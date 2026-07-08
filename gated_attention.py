#!/usr/bin/env python3
"""Deterministic mean-field 'gated softmax' attention (Phase A).

Derived from the subset-attention framework by collapsing the Gibbs
expectation E_{S ~ p(S)}[ softmax-restricted-to-S ] to its mean-field marginal,
replacing the high-variance hard sampler with a deterministic, differentiable
functional:

    a_i  = q . k_i / sqrt(head_dim)                              (dot-product score)
    r_i  = sum_j g0_j * <k_i, k_j> / sqrt(head_dim)              (anti-redundancy; Phase B)
    g_i  = sigmoid( beta * (a_i - tau(q) - lambda * r_i) )       (inclusion marginal / gate)
    w_i  = g_i * exp(a_i) / sum_j g_j * exp(a_j)                 (gated softmax)
    y    = sum_i w_i v_i

Why this fixes the four confirmed failures of the sampled version:
  1. Forward variance:   deterministic -> variance is exactly 0 (was rel-std 0.955).
  2. Zero F2 gradient:    beta / tau / lambda enter the output analytically, so they
                          receive real gradients (the sampled restricted_softmax gave None).
  3. Sampler bias:        no Gibbs chain, no empty-init burn-in bias.
  4. Objective mismatch:  beta = 0  =>  g_i = 0.5  =>  the constant cancels in the
                          normaliser  =>  w = softmax(a) EXACTLY. beta is initialised to
                          0, so the layer *starts* as ordinary multi-head attention and
                          departs from it only if that lowers the loss.

Parameter budget vs nn.MultiheadAttention (bias=True): identical fused qkv + out_proj
(4*dim^2 + 4*dim), plus gate params beta(H) + w_tau(dim) + b_tau(H) [+ log_lambda(H) if
repulsion] = O(dim). For dim=192, H=3 that is +198 params on ~148k, i.e. ~0.13%.

Set repulsion=True for the Phase-B DPP-flavoured win-bet (the one inductive bias plain
softmax structurally cannot express: it scores keys independently, so it cannot
down-weight a key for being redundant with the other selected keys).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSoftmaxAttention(nn.Module):
    """Multi-head deterministic gated-softmax self-attention.

    Input/output: (B, L, dim). Parameter-matched to nn.MultiheadAttention up to
    O(dim) gate parameters.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        proj_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        bias: bool = True,
        beta_init: float = 0.0,
        repulsion: bool = False,
        lambda_init: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.eps = float(eps)

        # Fused QKV + output projection -- mirrors nn.MultiheadAttention exactly.
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Gate parameters (O(dim) total). beta = 0  =>  exact softmax at init.
        # NOTE (sticky-parity): at beta=0, d gate / d tau ∝ beta = 0, so tau/repulsion
        # gradients are frozen and beta=0 is an attractive fixed point -> the layer can
        # stay dense forever ("match but never beat"). Use beta_init=0.0 for the parity
        # proof; use a small positive beta_init (e.g. 0.3-1.0) for win experiments so the
        # gate is live and all gate params receive gradient from step 0.
        self.beta = nn.Parameter(torch.full((self.num_heads,), float(beta_init)))
        self.w_tau = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.b_tau = nn.Parameter(torch.zeros(self.num_heads))

        # Phase-B repulsion hook. softplus(log_lambda) >= 0; start near 0 so the
        # module begins in the pure-selection regime and only adds repulsion if it helps.
        self.repulsion = bool(repulsion)
        if self.repulsion:
            # softplus^{-1}(lambda_init); for lambda_init=0 use a large-negative logit.
            init = math.log(math.expm1(lambda_init)) if lambda_init > 0 else -8.0
            self.log_lambda = nn.Parameter(torch.full((self.num_heads,), float(init)))
        else:
            self.register_parameter("log_lambda", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if D != self.dim:
            raise ValueError(f"dim mismatch: got {D}, expected {self.dim}")

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, Lq, Lk)

        tau = torch.einsum("bhld,hd->bhl", q, self.w_tau) + self.b_tau[None, :, None]
        beta = self.beta[None, :, None, None]
        gate_logit = a - tau.unsqueeze(-1)  # (B, H, Lq, Lk), tau broadcast over keys

        if self.repulsion:
            lam = F.softplus(self.log_lambda)[None, :, None, None]
            g0 = torch.sigmoid(beta * gate_logit)                 # base per-(q,key) marginal
            # r[b,h,q,i] = sum_j g0[b,h,q,j] * <k_j, k_i> = <k_i, m_q>, m_q = sum_j g0 k_j.
            # Factored O(n^2 d) form (avoids the O(n^3) g0 @ (K K^T) Gram materialization):
            # down-weights key i when it is redundant with the keys this query includes.
            m = torch.matmul(g0, k)                               # (B, H, Lq, head_dim)
            r = torch.matmul(m, k.transpose(-1, -2)) * self.scale  # (B, H, Lq, Lk)
            gate_logit = gate_logit - lam * r

        gate = torch.sigmoid(beta * gate_logit)  # (B, H, Lq, Lk)

        # Gated softmax over keys (numerically stable). beta=0 -> gate=0.5 -> exact softmax.
        a_max = a.amax(dim=-1, keepdim=True)
        ex = torch.exp(a - a_max) * gate
        denom = ex.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        w = ex / denom
        w = self.attn_drop(w)

        y = torch.matmul(w, v)                       # (B, H, Lq, head_dim)
        y = y.transpose(1, 2).reshape(B, L, D)       # (B, L, dim)
        y = self.out_proj(y)
        return self.proj_drop(y)

    @torch.no_grad()
    def load_from_mha(self, mha: nn.MultiheadAttention) -> None:
        """Copy nn.MultiheadAttention weights so that, with beta=0, this module is
        bit-for-bit equivalent to ordinary attention. Used by the parity test."""
        self.qkv.weight.copy_(mha.in_proj_weight)
        if mha.in_proj_bias is not None and self.qkv.bias is not None:
            self.qkv.bias.copy_(mha.in_proj_bias)
        self.out_proj.weight.copy_(mha.out_proj.weight)
        self.out_proj.bias.copy_(mha.out_proj.bias)
        self.beta.zero_()
        self.w_tau.zero_()
        self.b_tau.zero_()
