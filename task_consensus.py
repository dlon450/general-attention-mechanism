#!/usr/bin/env python3
"""Consensus-under-adversarial-duplication benchmark (numpy, CPU, deterministic).

Binary latent truth: two content clusters are present per example, at two prototypes (truth theta_t,
lie theta_l). The two clusters are made EXCHANGEABLE in everything a (content, surface-id) rule can
see — same #items, same #distinct surface ids, same per-item content spread — so NO function of (V,S)
can tell which is truth (swap-symmetry => chance). They differ ONLY in true-origin structure:
  - HONEST cluster = n independent true origins (each a single item)  -> within-cluster same-origin ~0
  - SYBIL  cluster = ONE true origin relabeled across n surface ids   -> within-cluster same-origin ~1
The origin structure is revealed only through a NOISY same-origin graph Pgraph (each entry correct
w.p. gamma). alpha = sigma_lie/sigma_src sets the Sybil per-item spread; alpha=1 makes the two clusters
content-identical (the hard, swap-symmetric slice). A noise-ROBUST reader averages Pgraph within each
content cluster (honest low, Sybil high) -> predicts truth = lower-same-origin cluster.

Success window we need: cheap (content/count/surface) baselines ~= chance(50%); oracle(provenance)
well above, degrading gracefully with gamma."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HON, SYB, BG, PAD = 1, 2, 3, 0


@dataclass
class Cfg:
    C: int = 4
    d: int = 32
    L: int = 64
    proto_scale: float = 3.0
    sigma_src: float = 1.0       # honest per-origin content spread (reference)
    sigma_item: float = 0.3      # shared item jitter (both clusters)
    n_orig: tuple = (3, 10)      # #origins = #items = #surface-ids for BOTH clusters (matched)
    nbg: tuple = (0, 6)          # background distractors (REAL, off both prototypes)
    bg_scale: float = 4.0


def prototypes(cfg, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((cfg.d, cfg.d)))
    return (Q[:cfg.C] * cfg.proto_scale).astype(np.float32)      # (C,d) orthonormal * scale


def gen_batch(n, cfg, rng, alpha, gamma, mu):
    L, d, C = cfg.L, cfg.d, cfg.C
    sig_lie = alpha * cfg.sigma_src
    V = np.zeros((n, L, d), np.float32); S = np.zeros((n, L), np.int64)
    origin = np.full((n, L), -1, np.int64); role = np.zeros((n, L), np.int64)
    mask = np.zeros((n, L), np.float32); y = np.zeros(n, np.int64)
    present = np.zeros((n, 2), np.int64)
    for b in range(n):
        yt = rng.integers(C); yl = (yt + 1 + rng.integers(C - 1)) % C
        y[b] = yt; present[b] = (yt, yl)
        k = rng.integers(cfg.n_orig[0], cfg.n_orig[1] + 1)      # matched size for both clusters
        items = []; oid = 0
        # honest: k independent origins, one item each
        for i in range(k):
            v = mu[yt] + cfg.sigma_src * rng.standard_normal(d) + cfg.sigma_item * rng.standard_normal(d)
            items.append((v, ("H", i), oid, HON)); oid += 1
        # sybil: ONE origin, k items across k surface ids, per-item spread sig_lie (matched at alpha=1)
        syb = oid; oid += 1
        for i in range(k):
            v = mu[yl] + sig_lie * rng.standard_normal(d) + cfg.sigma_item * rng.standard_normal(d)
            items.append((v, ("S", i), syb, SYB))
        # background: off-prototype distractors, each its own origin
        for _ in range(rng.integers(cfg.nbg[0], cfg.nbg[1] + 1)):
            items.append((cfg.bg_scale * rng.standard_normal(d), ("B", oid), oid, BG)); oid += 1
        items = items[:L]; rng.shuffle(items)
        relab = list(range(2000)); rng.shuffle(relab); uniq = {}
        for i, (v, sid, org, rl) in enumerate(items):
            if sid not in uniq:
                uniq[sid] = relab[len(uniq)]
            V[b, i] = v; S[b, i] = uniq[sid]; origin[b, i] = org; role[b, i] = rl; mask[b, i] = 1.0
    Pgraph = np.zeros((n, L, L), np.float32)
    for b in range(n):
        same = (origin[b][:, None] == origin[b][None, :]) & (origin[b][:, None] >= 0)
        flip = np.triu(rng.random((L, L)) > gamma, 1)
        P = same.astype(np.float32).copy()
        iu = np.triu_indices(L, 1)
        P[iu] = np.where(flip[iu], 1.0 - P[iu], P[iu])
        P = np.triu(P, 1); P = P + P.T; np.fill_diagonal(P, 1.0)
        m = mask[b]; P *= m[:, None] * m[None, :]
        Pgraph[b] = P
    return dict(V=V, S=S, Pgraph=Pgraph, mask=mask, y=y, origin=origin, role=role, mu=mu, present=present)


def _cluster(batch):
    """nearest-prototype class per item; the two 'present' classes are given (yt,yl in any order)."""
    V, mu = batch["V"], batch["mu"]
    return (((V[:, :, None, :] - mu[None, None]) ** 2).sum(-1)).argmin(-1)   # (n,L)


def _binary(batch, score_fn):
    """score_fn(B,b,cls,k)-> score for class k; higher wins; TIES broken RANDOMLY (deterministic per b)
    so matched-count clusters give true 50% chance (no answer-key leak via slot order)."""
    cls = _cluster(batch); pres = batch["present"]; out = np.zeros(len(cls), np.int64)
    for b in range(len(cls)):
        a, c = pres[b]
        rb = np.random.default_rng(9973 * b + 1)
        sa = score_fn(batch, b, cls[b], a) + 1e-6 * rb.standard_normal()
        sc = score_fn(batch, b, cls[b], c) + 1e-6 * rb.standard_normal()
        out[b] = a if sa >= sc else c
    return out


def base_item_majority(batch):
    return _binary(batch, lambda B, b, cls, k: ((cls == k) & (B["mask"][b] > 0)).sum())


def base_surface_count(batch):
    def sc(B, b, cls, k):
        sel = (cls == k) & (B["mask"][b] > 0)
        return len(np.unique(B["S"][b][sel])) if sel.any() else 0
    return _binary(batch, sc)


def base_content_centroid(batch):
    # distance of the cluster centroid to its own prototype is symmetric -> use item count proxy inverse
    def sc(B, b, cls, k):
        sel = (cls == k) & (B["mask"][b] > 0)
        return sel.sum()      # symmetric with item-majority; kept for completeness
    return _binary(batch, sc)


def oracle_robust(batch):
    """NOISE-ROBUST provenance reader: within-content-cluster MEAN of Pgraph. Honest cluster (many
    origins) -> low; Sybil (one origin) -> high. Predict truth = lower within-cluster same-origin."""
    cls = _cluster(batch); P = batch["Pgraph"]; m = batch["mask"]; pres = batch["present"]
    out = np.zeros(len(cls), np.int64)
    for b in range(len(cls)):
        def within(k):
            idx = np.where((cls[b] == k) & (m[b] > 0))[0]
            if len(idx) < 2:
                return 0.5
            sub = P[b][np.ix_(idx, idx)]
            n = len(idx)
            return (sub.sum() - np.trace(sub)) / (n * (n - 1))     # off-diagonal mean
        a, c = pres[b]
        out[b] = a if within(a) <= within(c) else c                # lower same-origin => honest
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000); a = ap.parse_args()
    cfg = Cfg(); rng = np.random.default_rng(0); mu = prototypes(cfg, rng)
    print(f"C={cfg.C} d={cfg.d} L={cfg.L}  binary chance=50%  (n={a.n}/cell)")
    print("want: cheap baselines ~50%, oracle(provenance) high & graceful in gamma\n")
    print(f"{'alpha':>6}{'gamma':>6}{'item-maj':>10}{'surf-cnt':>10}{'ORACLE':>9}")
    for alpha in (0.0, 0.5, 1.0, 2.0):
        for gamma in (1.0, 0.9, 0.8, 0.7, 0.6):
            rng = np.random.default_rng(1234 + int(alpha * 10) * 7 + int(gamma * 100))
            B = gen_batch(a.n, cfg, rng, alpha, gamma, mu)
            acc = lambda p: 100.0 * (p == B["y"]).mean()
            print(f"{alpha:>6.1f}{gamma:>6.2f}{acc(base_item_majority(B)):>10.1f}"
                  f"{acc(base_surface_count(B)):>10.1f}{acc(oracle_robust(B)):>9.1f}", flush=True)


if __name__ == "__main__":
    main()
