#!/usr/bin/env python3
"""Numerically validate Theorem 1: the total attention mass on a size-m redundant clique grows
LINEARLY (Theta(m)) under softmax/first-order, but only LOGARITHMICALLY (Theta(log m)) under the
mean-field repulsion gate. Solves the self-consistent gate equation (no GPU, no training)."""
from __future__ import annotations

import math


def sigmoid(x):
    if x < -30.0:
        return 0.0
    if x > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def clique_gate_mass(m, beta=1.0, tau=0.0, lam=1.0, s=1.0, a=1.0):
    """Solve g = sigmoid(beta(a - tau - lam*s*m*g)) by bisection (f decreasing in g); return W=m*g."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        g = 0.5 * (lo + hi)
        f = sigmoid(beta * (a - tau - lam * s * m * g)) - g
        if f > 0:  # g too small
            lo = g
        else:
            hi = g
    g = 0.5 * (lo + hi)
    return m * g, g


def r2_loglinear(ms, ws):
    """R^2 of W ~ a + b*log(m)."""
    xs = [math.log(m) for m in ms]
    n = len(xs)
    mx, mw = sum(xs) / n, sum(ws) / n
    b = sum((x - mx) * (w - mw) for x, w in zip(xs, ws)) / sum((x - mx) ** 2 for x in xs)
    a = mw - b * mx
    ss_res = sum((w - (a + b * x)) ** 2 for x, w in zip(xs, ws))
    ss_tot = sum((w - mw) ** 2 for w in ws)
    return 1 - ss_res / ss_tot, a, b


def main():
    ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"{'m':>6} {'softmax W=m':>12} {'rep W (mass)':>13} {'rep g_D':>10} {'rep/soft':>10}")
    print("-" * 56)
    reps = []
    for m in ms:
        W, g = clique_gate_mass(m)
        reps.append(W)
        print(f"{m:>6} {m:>12} {W:>13.3f} {g:>10.4g} {W/m:>10.4g}")
    r2, a, b = r2_loglinear(ms, reps)
    print(f"\nrep clique mass fit  W ≈ {a:.3f} + {b:.3f}·ln(m)   R² = {r2:.5f}")
    # linear-fit R^2 for comparison (should be worse for rep)
    n = len(ms); mm = sum(ms) / n; mw = sum(reps) / n
    bl = sum((x - mm) * (w - mw) for x, w in zip(ms, reps)) / sum((x - mm) ** 2 for x in ms)
    al = mw - bl * mm
    ss_res = sum((w - (al + bl * x)) ** 2 for x, w in zip(ms, reps))
    ss_tot = sum((w - mw) ** 2 for w in reps)
    print(f"rep clique mass LINEAR fit W ≈ {al:.3f} + {bl:.4f}·m        R² = {1-ss_res/ss_tot:.5f}")
    print("\nVERDICT: softmax clique mass = m (linear, unbounded); repulsion clique mass ~ log m")
    print(f"  e.g. m=4096: softmax gives 4096 'votes', repulsion gives {reps[-1]:.1f} -> "
          f"{4096/reps[-1]:.0f}x suppression.")


if __name__ == "__main__":
    main()
