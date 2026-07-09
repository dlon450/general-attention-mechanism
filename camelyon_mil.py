#!/usr/bin/env python3
"""Camelyon16 slide-level MIL on Owkin's Phikon features — the canonical LARGE real MIL benchmark
(ABMIL/CLAM/TransMIL/DTFD all report here). A bag = one whole-slide image = 1000 patch features
(768-d Phikon, frozen); label = tumor(1)/normal(0). Standard FIXED train/test split (269/130),
metric = AUC (+ accuracy). Trainable pooling on frozen features (standard MIL) — same backbone for
all methods, so the comparison is apples-to-apples and parameter-matched (rep adds 3 scalars).

Camelyon16 is a NEEDLE task (positive iff any tumor patch), so redundancy is benign -> honest
expectation is ~parity with ABMIL (a real-SCALE "no-harm on the standard benchmark" result)."""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from mil_abmil import ABMILPool

ROOT = "/home/dereklong/mil_data/camelyon16/data"
FEAT_DIM = 768                                   # Phikon; first 3 cols are (zoom,x,y) coords -> dropped


def load_split(pat):
    fs = sorted(glob.glob(f"{ROOT}/{pat}"))
    import pandas as pd
    df = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
    X, Y = [], []
    for i in range(len(df)):
        feats = np.stack([np.asarray(x, dtype=np.float32) for x in df.iloc[i]["features"]])  # (Ni,771)
        X.append(torch.from_numpy(feats[:, 3:].copy()))    # drop coord cols -> (Ni,768), variable Ni
        Y.append(int(df.iloc[i]["label"]))
    return X, torch.tensor(Y)


def auc(scores, labels):                          # rank-based (Mann-Whitney U)
    s = np.asarray(scores); y = np.asarray(labels)
    order = s.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


class CamModel(nn.Module):
    def __init__(self, dim, mode, lambda_init=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(FEAT_DIM, dim), nn.GELU())
        self.pool = ABMILPool(dim, mode=mode, lambda_init=lambda_init)
        self.head = nn.Linear(dim, 2)

    def forward(self, bags):                       # (B,N,768)
        H = self.embed(bags)
        b, _ = self.pool(H)
        return self.head(b)


def run(mode, Xtr, Ytr, Xte, Yte, device, epochs=50, lr=1e-4, seed=0, lambda_init=1.0, lambda_lr=None):
    torch.manual_seed(seed)
    model = CamModel(256, mode, lambda_init=lambda_init).to(device)
    if mode == "rep" and lambda_lr:                       # give lambda/tau/beta their own high LR
        rep_p = [model.pool.log_lambda, model.pool.tau, model.pool.beta]
        rep_ids = {id(p) for p in rep_p}
        base_p = [p for p in model.parameters() if id(p) not in rep_ids]
        opt = torch.optim.AdamW([{"params": base_p, "lr": lr},
                                 {"params": rep_p, "lr": lambda_lr}], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    is_rep = mode == "rep"
    n = len(Xtr); total = epochs * n; step = 0
    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 131 + ep)
        for i in torch.randperm(n, generator=g).tolist():
            if is_rep:
                model.pool.rep_scale = min(1.0, max(0.0, (step - 0.4 * total) / (0.2 * total)))
            xb = Xtr[i].unsqueeze(0).to(device); yb = Ytr[i:i + 1].to(device)  # (1,Ni,768)
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step(); step += 1
    if is_rep:
        model.pool.rep_scale = 1.0
        import torch.nn.functional as _F
        print(f"[diag seed{seed}] learned lambda="
              f"{_F.softplus(model.pool.log_lambda).item():.3f} beta={model.pool.beta.item():.3f}",
              file=sys.stderr, flush=True)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(len(Xte)):
            lg = model(Xte[i].unsqueeze(0).to(device))
            probs.append(torch.softmax(lg, 1)[0, 1].item())
    probs = np.array(probs)
    acc = float(((probs > 0.5).astype(int) == Yte.numpy()).mean())
    return auc(probs, Yte.numpy()), acc


def main():
    import statistics as st
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["abmil", "dedup", "dpp", "rep"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lambda-init", type=float, default=1.0)
    ap.add_argument("--lambda-lr", type=float, default=None, help="separate high LR for lambda/tau/beta")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading Camelyon16 Phikon features ...", flush=True)
    Xtr, Ytr = load_split("*train*"); Xte, Yte = load_split("*test*")
    allx = torch.cat(Xtr, 0)
    mu = allx.mean(0); sd = allx.std(0).clamp_min(1e-6)
    Xtr = [(x - mu) / sd for x in Xtr]; Xte = [(x - mu) / sd for x in Xte]
    print(f"train {len(Xtr)} bags, test {len(Xte)} bags, {device}\n")
    print(f"{'mode':10}{'test AUC':>16}{'test acc':>16}")
    for mode in args.modes:
        aucs, accs = [], []
        for s in range(args.seeds):
            a, c = run(mode, Xtr, Ytr, Xte, Yte, device, epochs=args.epochs, seed=s,
                       lambda_init=args.lambda_init, lambda_lr=args.lambda_lr)
            aucs.append(a); accs.append(c)
        sd_a = st.pstdev(aucs) if len(aucs) > 1 else 0.0
        sd_c = st.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{mode:10}{st.mean(aucs):>10.4f}±{sd_a:.3f}{100*st.mean(accs):>11.2f}±{100*sd_c:.2f}", flush=True)


if __name__ == "__main__":
    main()
