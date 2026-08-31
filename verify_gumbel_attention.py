#!/usr/bin/env python3
"""Property tests for GumbelSoftmaxAttention (stochastic_attention.py).

Proves the four properties the sampled weight rule has to satisfy:
  1. CONVEX ROWS: every query row of the sampled attention matrix sums to 1 and is
     non-negative, in all four variants (gumbel/gumbel_topk x hard/soft). This is what
     stops the top-k rule from scaling the output by k.
  2. HARD SELECTION: with hard=True exactly k entries per row are nonzero (k=1 for
     gumbel), each carrying mass 1/k -- i.e. it really is a sample, not a dense reweighting.
  3. DIFFERENTIABLE: the straight-through estimator delivers finite, nonzero gradients to
     every parameter despite the forward pass being a hard, non-differentiable selection.
  4. PARAM-MATCHED: param count equals nn.MultiheadAttention EXACTLY (tau is a
     hyperparameter, not a weight), so accuracy deltas cannot be bought with capacity.

Plus two regression guards: no NaN under fp16 autocast (the -log(0) trap that kills AMP
runs), and the sampling is actually live (two forwards differ).

CPU only, tiny tensors. Run: python verify_gumbel_attention.py  (exits nonzero on failure)
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from stochastic_attention import GumbelSoftmaxAttention

VARIANTS = [(None, True), (None, False), (4, True), (4, False)]  # (topk, hard)


def test_convex_rows() -> bool:
    torch.manual_seed(0)
    B, H, L = 3, 4, 16
    scores = torch.randn(B, H, L, L)
    ok = True
    for topk, hard in VARIANTS:
        m = GumbelSoftmaxAttention(48, H, topk=topk, tau=0.7, hard=hard)
        w = m.sample_attn_weights(scores)
        rows = w.sum(dim=-1)
        err = (rows - 1.0).abs().max().item()
        good = err < 1e-5 and bool((w >= 0).all())
        ok &= good
        print(f"[1] convex rows  topk={topk} hard={hard}: max|rowsum-1|={err:.3e} "
              f"-> {'PASS' if good else 'FAIL'}")
    return ok


def test_hard_selection() -> bool:
    torch.manual_seed(0)
    B, H, L = 3, 4, 16
    scores = torch.randn(B, H, L, L)
    ok = True
    for topk in (None, 4):
        k = 1 if topk is None else topk
        m = GumbelSoftmaxAttention(48, H, topk=topk, tau=0.7, hard=True)
        w = m.sample_attn_weights(scores)
        nnz = (w > 0).sum(dim=-1)
        sel = w[w > 0]
        good = bool((nnz == k).all()) and torch.allclose(sel, torch.full_like(sel, 1.0 / k), atol=1e-5)
        ok &= good
        print(f"[2] hard selection  topk={topk}: {k} keys/row at mass 1/{k} "
              f"-> {'PASS' if good else 'FAIL'}")
    return ok


def test_straight_through_grads() -> bool:
    torch.manual_seed(0)
    B, L, D, H = 4, 17, 48, 4
    x = torch.randn(B, L, D)
    ok = True
    for topk, hard in VARIANTS:
        m = GumbelSoftmaxAttention(D, H, topk=topk, tau=0.7, hard=hard)
        m(x).pow(2).sum().backward()
        bad = [n for n, p in m.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
        ok &= not bad
        print(f"[3] straight-through grads  topk={topk} hard={hard}: "
              f"{'PASS' if not bad else 'FAIL ' + str(bad)}")
    return ok


def test_param_match() -> bool:
    D, H = 192, 3
    n_mha = sum(p.numel() for p in nn.MultiheadAttention(D, H, batch_first=True).parameters())
    n_gum = sum(p.numel() for p in GumbelSoftmaxAttention(D, H).parameters())
    good = n_mha == n_gum
    print(f"[4] param match (D={D},H={H}): MHA={n_mha:,}  gumbel={n_gum:,}  "
          f"extra={n_gum - n_mha} -> {'PASS' if good else 'FAIL'}")
    return good


def test_amp_safe() -> bool:
    if not torch.cuda.is_available():
        print("[5] fp16 autocast: SKIP (no CUDA)")
        return True
    torch.manual_seed(0)
    m = GumbelSoftmaxAttention(192, 3, tau=0.3).cuda()
    x = torch.randn(8, 65, 192, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = m(x)
    good = bool(torch.isfinite(y).all())
    print(f"[5] fp16 autocast: finite outputs -> {'PASS' if good else 'FAIL (-log(0) leak)'}")
    return good


def test_sampling_is_live() -> bool:
    torch.manual_seed(0)
    m = GumbelSoftmaxAttention(48, 4, tau=1.0).eval()
    x = torch.randn(4, 17, 48)
    good = not torch.allclose(m(x), m(x))
    print(f"[6] stochastic: two forwards differ -> {'PASS' if good else 'FAIL (not sampling)'}")
    return good


def main() -> None:
    ok = all([
        test_convex_rows(),
        test_hard_selection(),
        test_straight_through_grads(),
        test_param_match(),
        test_amp_safe(),
        test_sampling_is_live(),
    ])
    print("\nAll Gumbel-attention property tests passed." if ok else "\nFAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
