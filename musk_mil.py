#!/usr/bin/env python3
"""MUSK1/MUSK2 — the classic real MIL benchmark (Dietterich et al. 1997). Standard 10-fold CV.
A bag = a molecule; instances = its conformations (166 features); label = musk / non-musk.
Compares our repulsion pooling vs Gated-ABMIL and DPP/dedup baselines in the shared backbone.

Honest expectation: MUSK redundancy is benign, so we expect ~parity with ABMIL (a "no-harm on a
standard real benchmark" result), not a big win. Prints a per-mode 10-fold mean±std table."""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from mil_abmil import ABMILPool

DATA = {"musk1": "/home/dereklong/mil_data/musk1.data",
        "musk2": "/home/dereklong/mil_data/musk2.data"}


def load_musk(path):
    bags = {}
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 169:
            continue
        mol = p[0]
        feats = [float(x) for x in p[2:168]]         # 166 features
        cls = int(float(p[168]))                     # molecule class (musk/non-musk)
        bags.setdefault(mol, {"x": [], "y": cls})
        bags[mol]["x"].append(feats)
    mols = list(bags)
    X = [torch.tensor(bags[m]["x"], dtype=torch.float32) for m in mols]
    Y = torch.tensor([1 if bags[m]["y"] > 0 else 0 for m in mols])
    return X, Y


def standardize(X):
    allx = torch.cat(X, 0)
    mu, sd = allx.mean(0), allx.std(0).clamp_min(1e-6)
    return [(x - mu) / sd for x in X]


class MuskModel(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(166, dim), nn.GELU())
        self.pool = ABMILPool(dim, mode=mode)
        self.head = nn.Linear(dim, 2)

    def forward(self, bag):                            # bag: (N,166)
        H = self.embed(bag).unsqueeze(0)              # (1,N,dim)
        b, _ = self.pool(H)
        return self.head(b)                           # (1,2)


def run_fold(mode, Xtr, Ytr, Xte, Yte, device, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = MuskModel(64, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    is_rep = mode == "rep"
    idx = list(range(len(Xtr)))
    for ep in range(epochs):
        if is_rep:
            model.pool.rep_scale = min(1.0, max(0.0, (ep - 0.4 * epochs) / (0.2 * epochs)))
        model.train()
        g = torch.Generator().manual_seed(seed * 997 + ep)
        for i in torch.randperm(len(Xtr), generator=g).tolist():
            opt.zero_grad(set_to_none=True)
            lg = model(Xtr[i].to(device))
            crit(lg, Ytr[i:i + 1].to(device)).backward()
            opt.step()
    if is_rep:
        model.pool.rep_scale = 1.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(Xte)):
            correct += int(model(Xte[i].to(device)).argmax(1).item() == int(Yte[i]))
    return correct / len(Xte)


def main():
    import argparse, statistics as st
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["musk1", "musk2"], default="musk1")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=["abmil", "dedup", "dpp", "rep"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = load_musk(DATA[args.dataset]); X = standardize(X)
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    folds = [perm[i::args.folds] for i in range(args.folds)]
    print(f"{args.dataset}: {n} bags, pos={int(Y.sum())}, {args.folds}-fold CV\n")
    print(f"{'mode':10}{'acc mean±std':>16}")
    for mode in args.modes:
        accs = []
        for f in range(args.folds):
            te = folds[f]; tr = [i for i in range(n) if i not in te]
            Xtr = [X[i] for i in tr]; Ytr = Y[tr]
            Xte = [X[i] for i in te]; Yte = Y[te]
            accs.append(run_fold(mode, Xtr, Ytr, Xte, Yte, device))
        print(f"{mode:10}{100*st.mean(accs):>10.2f} ± {100*st.pstdev(accs):.2f}")


if __name__ == "__main__":
    main()
