#!/usr/bin/env python3
"""Real federated CIFAR-10 under Byzantine attack — the honest test of rep as a robust aggregator.

FedSGD over N workers on non-IID (Dirichlet) shards. Each round every worker sends a gradient; a
fraction f are Byzantine (omniscient — they see the honest gradients). The server aggregates with a
defense and takes one SGD step. We report FINAL test accuracy per (defense x attack). The decisive
column is `adaptive_rep`: an attack designed to EVADE rep (shared malicious pull + per-worker
orthogonal noise so updates look mutually diverse / low-redundancy-with-aggregate).

Defenses (server-side aggregation rules, parameter-matched, reused from byzantine_check):
  mean, comedian(median), trimmed, krum, cclip, foolsgold, rep(ours)

Attacks (Byzantine gradient replacement):
  none      : no attack (reference ceiling)
  signflip  : -scale * honest_mean
  ipm       : -eps * honest_mean            (inner-product manipulation, Xie'20)
  alie      : honest_mean - z*honest_std    ("a little is enough", Baruch'19 — evasive, in-distribution)
  adaptive_rep : shared malicious dir + big per-worker orthogonal noise (crafted to evade rep/FoolsGold)
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from byzantine_check import (agg_mean, agg_comedian, agg_trimmed, agg_krum, agg_cclip,
                             agg_foolsgold, agg_rep)

BASE = "/data/users/dereklong/scratch/general-attention-mechanism/data/cifar-10-batches-py"
DEFENSES = {"mean": agg_mean, "comedian": agg_comedian, "trimmed": agg_trimmed,
            "krum": agg_krum, "cclip": agg_cclip, "foolsgold": agg_foolsgold, "rep": agg_rep}
ATTACKS = ["none", "signflip", "ipm", "alie", "adaptive_rep"]


def load_cifar():
    def rd(names):
        X, Y = [], []
        for n in names:
            d = pickle.load(open(f"{BASE}/{n}", "rb"), encoding="latin1")
            X.append(d["data"]); Y += d["labels"]
        X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        return X, np.array(Y, dtype=np.int64)
    Xtr, Ytr = rd([f"data_batch_{i}" for i in range(1, 6)])
    Xte, Yte = rd(["test_batch"])
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.247, 0.243, 0.261]).reshape(1, 3, 1, 1)
    return ((Xtr - mean) / std).astype(np.float32), Ytr, ((Xte - mean) / std).astype(np.float32), Yte


def partition(Y, N, alpha, rng):
    shards = [[] for _ in range(N)]
    for c in range(10):
        idx = np.where(Y == c)[0]; rng.shuffle(idx)
        props = rng.dirichlet([alpha] * N)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for w, chunk in enumerate(np.split(idx, cuts)):
            shards[w] += chunk.tolist()
    return [np.array(s) for s in shards]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.f1 = nn.Linear(32 * 8 * 8, 64); self.f2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.f1(x.flatten(1)))
        return self.f2(x)


def apply_attack(name, G, byz, honest, rng_t):
    if name == "none" or len(byz) == 0:
        return G
    H = G[honest]; hmean = H.mean(0); hstd = H.std(0) + 1e-9
    if name == "signflip":
        G[byz] = -5.0 * hmean
    elif name == "ipm":
        G[byz] = -0.5 * hmean
    elif name == "alie":
        z = 1.5
        G[byz] = hmean - z * hstd                     # within honest spread -> evasive
    elif name == "adaptive_rep":
        # evade rep/FoolsGold: shared malicious pull + per-worker orthogonal noise (look diverse),
        # then NORM-BOUND each byz update to the honest median norm (evade magnitude/clipping defenses).
        hnorm = np.median(np.linalg.norm(H, axis=1))
        base = -hmean / (np.linalg.norm(hmean) + 1e-9)
        noise = rng_t.standard_normal((len(byz), G.shape[1])).astype(np.float32)
        noise -= (noise @ base)[:, None] * base[None]  # orthogonalize to the malicious direction
        noise /= (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-9)
        mix = 0.5 * base[None] + 0.866 * noise         # 50% shared pull + diversified (unit norm)
        G[byz] = hnorm * mix                            # norm-matched to honest -> evades clipping
    return G


@torch.no_grad()
def test_acc(model, Xte, Yte, device, bs=1000):
    model.eval(); cor = 0
    for i in range(0, len(Xte), bs):
        xb = torch.from_numpy(Xte[i:i + bs]).to(device)
        cor += (model(xb).argmax(1).cpu().numpy() == Yte[i:i + bs]).sum()
    return 100.0 * cor / len(Xte)


def run(defense, attack, Xtr, Ytr, Xte, Yte, shards, device, N=16, fbyz=0.3, rounds=200,
        bs=32, lr=0.1, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); rng_t = np.random.default_rng(seed + 7)
    model = CNN().to(device)
    params = list(model.parameters())
    nb = int(round(fbyz * N)); byz = list(range(nb)); honest = list(range(nb, N))
    crit = nn.CrossEntropyLoss()
    aggfn = DEFENSES[defense]
    accs = []
    for t in range(rounds):
        G = np.empty((N, sum(p.numel() for p in params)), dtype=np.float32)
        for w in range(N):
            sh = shards[w]
            bi = sh[rng.integers(0, len(sh), size=min(bs, len(sh)))]
            xb = torch.from_numpy(Xtr[bi]).to(device); yb = torch.from_numpy(Ytr[bi]).to(device)
            model.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            G[w] = torch.cat([p.grad.reshape(-1) for p in params]).cpu().numpy()
        G = apply_attack(attack, G, byz, honest, rng_t)
        agg = torch.from_numpy(aggfn(G, nb)).to(device)
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel(); p.add_(agg[i:i + n].view_as(p), alpha=-lr); i += n
        if (t + 1) % 25 == 0:
            accs.append(test_acc(model, Xte, Yte, device))
    return float(np.mean(accs[-3:]))          # final test acc (avg of last 3 evals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--fbyz", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID (smaller=more non-IID)")
    ap.add_argument("--defenses", nargs="+", default=list(DEFENSES))
    ap.add_argument("--attacks", nargs="+", default=ATTACKS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Ytr, Xte, Yte = load_cifar()
    rng = np.random.default_rng(0)
    shards = partition(Ytr, args.N, args.alpha, rng)
    print(f"CIFAR FedSGD: N={args.N} byz-frac={args.fbyz} rounds={args.rounds} "
          f"non-IID alpha={args.alpha} | {device}")
    print(f"final test acc (avg last 3 evals), byz={int(round(args.fbyz*args.N))}/{args.N}\n")
    print(f"{'defense':11}" + "".join(f"{a:>13}" for a in args.attacks))
    for dfn in args.defenses:
        row = []
        for atk in args.attacks:
            row.append(run(dfn, atk, Xtr, Ytr, Xte, Yte, shards, device,
                           N=args.N, fbyz=args.fbyz, rounds=args.rounds))
        print(f"{dfn:11}" + "".join(f"{v:>13.1f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
