#!/usr/bin/env python3
"""Phase-A proof tests for GatedSoftmaxAttention.

Proves the four properties that fix the sampled version's failures:
  1. NESTS SOFTMAX: with weights copied from nn.MultiheadAttention and beta=0,
     the output matches ordinary attention bit-for-bit (< 1e-5).
  2. DIFFERENTIABLE GATE: beta / tau params receive nonzero gradient (the sampled
     restricted_softmax gave None) -- and the sticky-parity note is demonstrated.
  3. DETERMINISTIC: two forwards with different RNG seeds are identical
     (variance 0, vs the sampled version's relative std ~0.955).
  4. PARAM-MATCHED: param count equals nn.MultiheadAttention up to O(dim) gate params.

CPU only, tiny tensors.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from gated_attention import GatedSoftmaxAttention


def test_nests_softmax() -> None:
    torch.manual_seed(0)
    B, L, D, H = 4, 65, 48, 4
    x = torch.randn(B, L, D)

    mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True, bias=True)
    mha.eval()
    gated = GatedSoftmaxAttention(D, H, beta_init=0.0).eval()
    gated.load_from_mha(mha)

    with torch.no_grad():
        y_mha, _ = mha(x, x, x, need_weights=False)
        y_gated = gated(x)
    err = (y_mha - y_gated).abs().max().item()
    print(f"[1] nests softmax:      max_abs_err = {err:.3e}  ->  {'PASS' if err < 1e-5 else 'FAIL'}")
    assert err < 1e-5, err


def _grad_norm(p):
    return None if p.grad is None else float(p.grad.norm().item())


def test_gate_gradients() -> None:
    B, L, D, H = 4, 65, 48, 4
    x = torch.randn(B, L, D)

    # At beta=0: beta receives gradient, but tau is frozen (d/dtau ∝ beta = 0).
    g0 = GatedSoftmaxAttention(D, H, beta_init=0.0)
    y = g0(x); y.pow(2).mean().backward()
    gb0, gt0 = _grad_norm(g0.beta), _grad_norm(g0.w_tau)
    print(f"[2a] beta_init=0.0:  grad(beta)={gb0}  grad(w_tau)={gt0}  "
          f"(tau frozen at init -- the sticky-parity risk)")

    # At beta>0: the whole gate is live; every gate param gets gradient.
    g1 = GatedSoftmaxAttention(D, H, beta_init=0.5)
    y = g1(x); y.pow(2).mean().backward()
    gb1, gt1, gbt1 = _grad_norm(g1.beta), _grad_norm(g1.w_tau), _grad_norm(g1.b_tau)
    ok = gb1 and gb1 > 0 and gt1 and gt1 > 0 and gbt1 and gbt1 > 0
    print(f"[2b] beta_init=0.5:  grad(beta)={gb1:.3e}  grad(w_tau)={gt1:.3e}  "
          f"grad(b_tau)={gbt1:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    assert ok

    # Phase-B repulsion param also differentiable.
    g2 = GatedSoftmaxAttention(D, H, beta_init=0.5, repulsion=True)
    y = g2(x); y.pow(2).mean().backward()
    gl = _grad_norm(g2.log_lambda)
    print(f"[2c] repulsion on:   grad(log_lambda)={gl:.3e}  ->  "
          f"{'PASS' if gl and gl > 0 else 'FAIL'}")
    assert gl and gl > 0


def test_deterministic() -> None:
    B, L, D, H = 4, 65, 48, 4
    x = torch.randn(B, L, D)
    gated = GatedSoftmaxAttention(D, H, beta_init=0.5).eval()
    with torch.no_grad():
        torch.manual_seed(111)
        y1 = gated(x)
        torch.manual_seed(222)
        y2 = gated(x)
    err = (y1 - y2).abs().max().item()
    print(f"[3] deterministic:      max_abs_err across seeds = {err:.3e}  ->  "
          f"{'PASS' if err == 0.0 else 'FAIL'}  (sampled version: rel-std 0.955)")
    assert err == 0.0


def test_param_match() -> None:
    D, H = 192, 3
    mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True, bias=True)
    gated = GatedSoftmaxAttention(D, H)
    n_mha = sum(p.numel() for p in mha.parameters())
    n_gated = sum(p.numel() for p in gated.parameters())
    gate_params = gated.beta.numel() + gated.w_tau.numel() + gated.b_tau.numel()
    extra = n_gated - n_mha
    print(f"[4] param match (D={D},H={H}): MHA={n_mha:,}  gated={n_gated:,}  "
          f"extra={extra} (=dim+2H={D + 2 * H}, {100 * extra / n_mha:.3f}% of MHA)")
    assert extra == D + 2 * H == gate_params


if __name__ == "__main__":
    test_nests_softmax()
    test_gate_gradients()
    test_deterministic()
    test_param_match()
    print("\nAll Phase-A property tests passed.")
