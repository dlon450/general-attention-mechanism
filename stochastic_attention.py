#!/usr/bin/env python3
"""Stochastic (Gumbel-softmax) attention: differentiable samples from the attention matrix.

Where ordinary attention consumes the *dense* softmax weights

    w = softmax(a),   a_i = q . k_i / sqrt(head_dim)

this module consumes a differentiable **sample** from that distribution instead. Two
sampling rules, selected by ``topk``:

  mode ``gumbel`` (topk is None)
      Straight-through Gumbel-softmax: one hard categorical sample (argmax) per query
      row, gradients flowing through the relaxed distribution.

  mode ``gumbel_topk`` (topk = k >= 1)
      The k highest-scoring keys per query row, each given mass ``1 / k`` so the row
      stays a convex combination (weights sum to 1) rather than blowing the output up
      by a factor of k.

In both cases fresh, independent Gumbel(0, 1) noise is drawn for *every* entry of the
(L, S) attention matrix -- L * S samples per head, not one shared noise vector per row.
With ``hard=False`` the forward pass uses the relaxed weights directly (no
straight-through), which is fully differentiable but no longer a hard selection.

Relation to the rest of this repo: this is the *sampled* counterpart of
gated_attention.GatedSoftmaxAttention. The gated module collapses the subset expectation
E_{S ~ p(S)}[softmax restricted to S] to a deterministic mean-field marginal; this module
keeps the sampling but makes it differentiable via the Gumbel-softmax relaxation, so the
two bracket the same object from the deterministic and stochastic sides.

Parameter budget vs nn.MultiheadAttention (bias=True): *identical* -- same fused qkv +
out_proj (4*dim^2 + 4*dim), no extra parameters. tau is a hyperparameter (optionally
annealed over the run), not a learned weight.

Note on dropout: attention dropout is deliberately NOT applied to the sampled weights
(the sampling is itself the stochastic regularizer, and dropping entries of an already
one-hot row would zero the row). Callers that set attn_dropout are warned by the trainer.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class GumbelSoftmaxAttention(nn.Module):
    """Multi-head self-attention over Gumbel-softmax samples of the attention matrix.

    Input/output: (B, L, dim). Parameter-matched to nn.MultiheadAttention exactly.

    Args:
        dim: model width; must be divisible by ``num_heads``.
        num_heads: number of attention heads.
        proj_dropout: dropout on the output projection (as in the other attention modules).
        topk: ``None`` for one-hot sampling (mode ``gumbel``); ``k >= 1`` to keep the k
            best keys per row with mass ``1 / k`` each (mode ``gumbel_topk``). Clamped to
            the sequence length at runtime.
        tau: Gumbel-softmax relaxation temperature. Read at forward time from
            ``self.gumbel_tau``, so an annealing schedule only has to write the attribute.
        hard: ``True`` (default) for straight-through hard selection; ``False`` to use the
            relaxed weights directly.
        bias: whether the qkv projection carries a bias (out_proj always does, as in
            nn.MultiheadAttention).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        proj_dropout: float = 0.0,
        topk: Optional[int] = None,
        tau: float = 1.0,
        hard: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if topk is not None and topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV + output projection -- mirrors nn.MultiheadAttention exactly, and uses
        # the same layout as GatedSoftmaxAttention so load_from_mha() is a straight copy.
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.topk = None if topk is None else int(topk)
        # Plain attribute (not a buffer/parameter): the temperature anneal writes it
        # between steps and forward() reads it, exactly as in the reference implementation.
        self.gumbel_tau = float(tau)
        self.hard = bool(hard)

    def sample_attn_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """Turn attention logits into (straight-through) Gumbel-softmax weights.

        Args:
            scores: attention logits, shape (..., L, S). Every row of the result sums to 1.
        """
        # One independent Gumbel(0, 1) sample per logit -- L * S per head, never a single
        # per-row noise vector reused across the row.
        #
        # The Exp(1) draw is taken in float32 and clamped away from zero even when scores
        # is half precision: fp16 exponential_() returns exact zeros at a rate of ~2.5e-8,
        # and -log(0) = +inf poisons the whole softmax row with NaN. At this model's shapes
        # that is ~0.5 poisoned rows per forward under autocast, enough to destroy an AMP run.
        u = torch.empty_like(scores, dtype=torch.float32).exponential_()
        gumbels = -u.clamp_min(torch.finfo(torch.float32).tiny).log().to(scores.dtype)
        y_soft = ((scores + gumbels) / self.gumbel_tau).softmax(dim=-1)

        if self.topk is None:
            if not self.hard:
                # Soft relaxation: the full Gumbel-softmax distribution over all keys.
                return y_soft
            # Hard straight-through: one categorical sample (argmax) per query row.
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft

        # Top-k: restrict to the k best keys per query row.
        k = min(self.topk, scores.size(-1))
        index = y_soft.topk(k, dim=-1).indices
        if not self.hard:
            # Soft top-k: keep the selected soft weights and renormalize them to sum to 1
            # (convex combination over the k keys); fully differentiable.
            keep = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
            y_topk = y_soft * keep
            return y_topk / y_topk.sum(dim=-1, keepdim=True)
        # Hard straight-through: each selected key gets mass 1/k so the row stays a convex
        # combination (sums to 1) instead of scaling the output by k.
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0 / k)
        return y_hard - y_soft.detach() + y_soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if D != self.dim:
            raise ValueError(f"dim mismatch: got {D}, expected {self.dim}")

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, Lq, Lk)
        w = self.sample_attn_weights(scores)

        y = torch.matmul(w, v)                       # (B, H, Lq, head_dim)
        y = y.transpose(1, 2).reshape(B, L, D)       # (B, L, dim)
        y = self.out_proj(y)
        return self.proj_drop(y)

    @torch.no_grad()
    def load_from_mha(self, mha: nn.MultiheadAttention) -> None:
        """Copy nn.MultiheadAttention weights, so the only difference from ordinary
        attention is the weight rule (sampled vs dense). Used by the verification script."""
        self.qkv.weight.copy_(mha.in_proj_weight)
        if mha.in_proj_bias is not None and self.qkv.bias is not None:
            self.qkv.bias.copy_(mha.in_proj_bias)
        self.out_proj.weight.copy_(mha.out_proj.weight)
        self.out_proj.bias.copy_(mha.out_proj.bias)
