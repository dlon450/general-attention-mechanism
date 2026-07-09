#!/usr/bin/env python3
"""STRUCTURE CHECK (before any FL build): does rep's decorrelation beat specialized Byzantine-robust
aggregators when attackers COLLUDE (correlated updates)? Pure robust-mean-estimation simulation.

Setup: n workers submit d-dim update vectors. Honest workers ~ N(mu, sigma_h^2 I) — DIVERSE (non-IID
shards). Byzantine workers COLLUDE: a tight cluster (sigma_b << sigma_h) aimed to pull the aggregate
away from mu. Metric = ||aggregate - mu|| / ||mu|| (lower = more robust).

Aggregators (all parameter-free rules on the update SET):
  mean       : baseline (no defense)
  comedian   : coordinate-wise median
  trimmed    : coordinate-wise trimmed mean (trim = byz frac)
  krum       : Krum (Blanchard'17) — update closest to its n-f-2 neighbors
  foolsgold  : Sybil defense — down-weight workers whose direction is similar to others (THE cousin)
  rep        : OUR soft/global repulsion — down-weight updates redundant with the (gated) aggregate

Key question: is there a realistic regime (tight collusion, diverse honest) where rep beats BOTH the
classic defenses (median/Krum) AND its nearest cousin FoolsGold? If not, this niche fails too."""
from __future__ import annotations

import numpy as np


def gen(d, n, fbyz, sigma_h, sigma_b, rng, cmag=6.0):
    mu = rng.standard_normal(d); mu /= np.linalg.norm(mu)
    nb = int(round(fbyz * n)); nh = n - nb
    honest = mu[None] + sigma_h * rng.standard_normal((nh, d))
    # colluding attack: tight cluster pushed along a fixed direction far from mu
    u = rng.standard_normal(d); u -= (u @ mu) * mu; u /= np.linalg.norm(u)   # orthogonal push
    center = mu + cmag * u
    byz = center[None] + sigma_b * rng.standard_normal((nb, d))
    G = np.concatenate([honest, byz], 0)
    perm = rng.permutation(n)
    return G[perm], mu, nb


def agg_mean(G, nb):
    return G.mean(0)


def agg_comedian(G, nb):
    return np.median(G, 0)


def agg_trimmed(G, nb):
    k = nb
    S = np.sort(G, 0)
    return S[k:len(G) - k].mean(0) if len(G) - 2 * k > 0 else np.median(G, 0)


def agg_krum(G, nb):
    n = len(G); m = n - nb - 2
    if m < 1:
        return np.median(G, 0)
    D = ((G[:, None] - G[None]) ** 2).sum(-1)
    scores = np.array([np.sort(D[i])[1:m + 1].sum() for i in range(n)])
    return G[scores.argmin()]


def agg_foolsgold(G, nb):
    N = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    CS = N @ N.T; np.fill_diagonal(CS, 0.0)
    mx = CS.max(1)                              # max similarity to any other worker
    # FoolsGold pardoning + logit rescale
    cv = mx.copy()
    for i in range(len(G)):
        for j in range(len(G)):
            if mx[i] > mx[j] and mx[j] > 0:
                CS[i, j] *= mx[j] / mx[i]
    cv = CS.max(1)
    w = 1.0 - cv
    w = np.clip(w, 0, 1); w /= w.max() + 1e-9
    w = np.clip(w, 1e-6, 1)
    w = np.log(w / (1 - w) + 1e-9) + 0.5
    w = np.clip(w, 0, 1); w /= w.sum() + 1e-9
    return (w[:, None] * G).sum(0)


def agg_rep(G, nb, iters=5, beta=6.0, lam=1.0):
    N = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    w = np.ones(len(G)) / len(G)
    for _ in range(iters):
        m = (w[:, None] * G).sum(0); md = m / (np.linalg.norm(m) + 1e-9)
        r = N @ md                              # redundancy = alignment with current aggregate
        g = 1.0 / (1.0 + np.exp(beta * lam * (r - np.median(r))))   # suppress > median redundancy
        w = g / (g.sum() + 1e-9)
    return (w[:, None] * G).sum(0)


def agg_cclip(G, nb, iters=5):
    v = np.median(G, 0)                          # robust init
    tau = np.median(np.linalg.norm(G - v, axis=1)) + 1e-9
    for _ in range(iters):
        diff = G - v
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        v = v + (diff * np.minimum(1.0, tau / (norms + 1e-9))).mean(0)
    return v


AGG = {"mean": agg_mean, "comedian": agg_comedian, "trimmed": agg_trimmed,
       "krum": agg_krum, "cclip": agg_cclip, "foolsgold": agg_foolsgold, "rep": agg_rep}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=50)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sigma-h", type=float, default=1.0, help="honest diversity")
    ap.add_argument("--sigma-b", type=float, default=0.05, help="collusion tightness")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    print(f"d={args.d} n={args.n} honest-spread={args.sigma_h} collusion-tightness={args.sigma_b} "
          f"({args.seeds} seeds)\nerror = ||agg-mu||/||mu||, lower=better\n")
    print(f"{'byz frac':10}" + "".join(f"{k:>11}" for k in AGG))
    for f in fracs:
        errs = {k: [] for k in AGG}
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 * s + int(100 * f))
            G, mu, nb = gen(args.d, args.n, f, args.sigma_h, args.sigma_b, rng)
            for k, fn in AGG.items():
                errs[k].append(np.linalg.norm(fn(G, nb) - mu) / np.linalg.norm(mu))
        print(f"{f:<10.2f}" + "".join(f"{np.mean(errs[k]):>11.3f}" for k in AGG))


if __name__ == "__main__":
    main()
