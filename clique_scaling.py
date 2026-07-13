#!/usr/bin/env python3
"""E0 — theory/implementation reconciliation (tests the reviewer's central math claim).

For a bag of n DISTINCT signal keys + a clique of m IDENTICAL keys, measure the clique's TOTAL gate
mass W_m as m grows, under three gates:
  one_step  : g1 = sigma(b(a - tau - lam*K*g0)), g0=sigma(b(a-tau))         [what gated_attention.py ships]
  fixed_pt  : converged g = sigma(b(a - tau - lam*K*g))  (damped iteration) [what THEORY.md proves]
  dpp       : true DPP marginal K = L(L+I)^{-1}, g_i=K_ii, L=diag(e^{a/2}) Gram diag(e^{a/2})

Reviewer claim: one_step -> W_m ~ m e^{-cm} -> 0 ; fixed_pt -> Theta(log m) ; dpp -> const.
If confirmed, our trained 'clique gate -> 0' validates one_step, NOT the log-m theorem."""
from __future__ import annotations

import numpy as np


def build(n, m, d=64, a_val=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Ksig = rng.standard_normal((n, d)); Ksig /= np.linalg.norm(Ksig, axis=1, keepdims=True)
    kc = rng.standard_normal(d); kc /= np.linalg.norm(kc)
    K = np.concatenate([Ksig, np.tile(kc, (m, 1))], 0)          # (n+m, d)
    G = (K @ K.T)                                                # Gram / sqrt(d) already unit-norm
    a = np.full(n + m, a_val)
    clique = np.arange(n, n + m)
    return G, a, clique


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def one_step(G, a, beta, lam, tau):
    g0 = sig(beta * (a - tau))
    r = G @ g0
    g1 = sig(beta * (a - tau - lam * r))
    return g1


def fixed_pt(G, a, beta, lam, tau, iters=2000, damp=0.5):
    g = sig(beta * (a - tau))
    for _ in range(iters):
        gn = sig(beta * (a - tau - lam * (G @ g)))
        g_new = damp * gn + (1 - damp) * g
        if np.max(np.abs(g_new - g)) < 1e-10:
            g = g_new; break
        g = g_new
    return g


def dpp(G, a):
    q = np.exp(0.5 * a)
    L = (q[:, None] * G) * q[None, :]
    K = L @ np.linalg.inv(L + np.eye(len(a)))
    return np.clip(np.diag(K), 0, 1)


def main():
    n, beta, lam, tau = 8, 1.0, 0.5, 0.0
    ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"n_signal={n} beta={beta} lam={lam} tau={tau}")
    print(f"contraction beta*lam*||K||_inf<4 holds while ||K||_inf ~ m*s stays < {4/(beta*lam):.0f}\n")
    print(f"{'m':>6}{'W_onestep':>12}{'W_fixedpt':>12}{'W_dpp':>10}{'||K||inf':>10}")
    rows = []
    for m in ms:
        G, a, clq = build(n, m)
        Kinf = np.abs(G).sum(1).max()
        Wo = one_step(G, a, beta, lam, tau)[clq].sum()
        Wf = fixed_pt(G, a, beta, lam, tau)[clq].sum()
        Wd = dpp(G, a)[clq].sum()
        rows.append((m, Wo, Wf, Wd))
        print(f"{m:>6}{Wo:>12.4f}{Wf:>12.4f}{Wd:>10.4f}{Kinf:>10.1f}")
    # crude functional-form fit on the large-m tail
    R = np.array(rows, dtype=float); tail = R[R[:, 0] >= 32]
    lm = np.log(tail[:, 0])
    for name, col in [("one_step", 1), ("fixed_pt", 2), ("dpp", 3)]:
        y = tail[:, col]
        # fit y ~ A + B*log(m)  (log-law) and report R^2
        A = np.vstack([np.ones_like(lm), lm]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef; ss = 1 - ((y - pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
        print(f"  {name:9} vs (A+B*ln m): A={coef[0]:.3f} B={coef[1]:.3f} R^2={ss:.3f}  "
              f"(tail max {y.max():.3f}, m=4096 val {R[-1, col]:.3f})")


if __name__ == "__main__":
    main()
