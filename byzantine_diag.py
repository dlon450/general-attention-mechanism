#!/usr/bin/env python3
"""WHY did rep fail the adaptive attack? Instrument the actual weights rep (and FoolsGold) assign to
honest vs Byzantine workers, per attack, on REAL CIFAR gradients. Diagnosis hypothesis: rep keys on
'alignment with the aggregate', which only flags attackers when they DOMINATE the aggregate; a
norm-bounded diversified MINORITY attack doesn't -> rep suppresses the honest consensus instead."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fl_byzantine import load_cifar, partition, CNN, apply_attack, ATTACKS


def rep_weights(G, iters=5, beta=6.0, lam=1.0):
    N = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    w = np.ones(len(G)) / len(G)
    for _ in range(iters):
        m = (w[:, None] * G).sum(0); md = m / (np.linalg.norm(m) + 1e-9)
        r = N @ md
        g = 1.0 / (1.0 + np.exp(beta * lam * (r - np.median(r))))
        w = g / (g.sum() + 1e-9)
    return w


def foolsgold_weights(G):
    N = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    CS = N @ N.T; np.fill_diagonal(CS, 0.0)
    cv = CS.max(1); w = np.clip(1.0 - cv, 0, 1); w /= w.max() + 1e-9
    w = np.clip(w, 1e-6, 1); w = np.clip(np.log(w / (1 - w) + 1e-9) + 0.5, 0, 1)
    return w / (w.sum() + 1e-9)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Ytr, Xte, Yte = load_cifar()
    rng = np.random.default_rng(0); shards = partition(Ytr, 16, 0.5, rng)
    N = 16; nb = 5; byz = list(range(nb)); honest = list(range(nb, N))
    model = CNN().to(device); params = list(model.parameters())
    crit = nn.CrossEntropyLoss()
    P = sum(p.numel() for p in params)

    def round_grads():
        G = np.empty((N, P), dtype=np.float32)
        for w in range(N):
            sh = shards[w]; bi = sh[rng.integers(0, len(sh), size=32)]
            xb = torch.from_numpy(Xtr[bi]).to(device); yb = torch.from_numpy(Ytr[bi]).to(device)
            model.zero_grad(set_to_none=True); crit(model(xb), yb).backward()
            G[w] = torch.cat([p.grad.reshape(-1) for p in params]).cpu().numpy()
        return G

    # warm up the model a little with clean mean-SGD so gradients aren't random
    for _ in range(40):
        G = round_grads(); agg = torch.from_numpy(G.mean(0)).to(device)
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel(); p.add_(agg[i:i + n].view_as(p), alpha=-0.1); i += n

    rng_t = np.random.default_rng(7)
    print("mean aggregation WEIGHT on honest vs byzantine workers (uniform would be 1/16=0.0625)")
    print(f"byz={nb}/{N}. Ideal defense: ~0 on byz, spread on honest.\n")
    print(f"{'attack':13}{'rep w_honest':>14}{'rep w_byz':>12}{'FG w_honest':>14}{'FG w_byz':>11}")
    for atk in ATTACKS:
        wh_r, wb_r, wh_f, wb_f = [], [], [], []
        for _ in range(15):
            G = apply_attack(atk, round_grads(), byz, honest, rng_t)
            wr = rep_weights(G); wf = foolsgold_weights(G)
            wh_r.append(wr[honest].mean()); wb_r.append(wr[byz].mean())
            wh_f.append(wf[honest].mean()); wb_f.append(wf[byz].mean())
        print(f"{atk:13}{np.mean(wh_r):>14.4f}{np.mean(wb_r):>12.4f}"
              f"{np.mean(wh_f):>14.4f}{np.mean(wb_f):>11.4f}")


if __name__ == "__main__":
    main()
