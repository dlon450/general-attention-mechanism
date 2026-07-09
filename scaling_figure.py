#!/usr/bin/env python3
"""Theory<->experiment scaling figure. Train rep and Gated-ABMIL on redundancy MIL with RANDOM
clique sizes, then on held-out bags sweep clique size m and measure (a) the fraction of attention
the TRAINED model puts on the clique, and (b) accuracy. Theorem 1 predicts the clique's attention
grows ~linearly->saturating for softmax/ABMIL but only ~log m for rep."""
from __future__ import annotations

import json
import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from mil_abmil import ABMILModel
from mil_mnist import NUM_CLASSES, _samp, load_by_class

N_SIG, N_BG = 3, 5


def make_bags(data, B, m, g, device, want_mask=False):
    X, P = data; C = NUM_CLASSES
    y = torch.randint(0, C, (B,), generator=g)
    yp = (y + 1 + torch.randint(0, C - 1, (B,), generator=g)) % C
    sig = _samp(P, y.unsqueeze(1).expand(B, N_SIG).contiguous(), X, g)   # (B,n_sig,784)
    dec = _samp(P, yp.unsqueeze(1), X, g).expand(B, m, 784)             # (B,m,784) identical clique
    a = torch.minimum(y, yp).unsqueeze(1); b = torch.maximum(y, yp).unsqueeze(1)
    bc = torch.randint(0, C - 2, (B, N_BG), generator=g)
    bc = bc + (bc >= a).long(); bc = bc + (bc >= b).long()
    bg = _samp(P, bc, X, g)
    bag = torch.cat([sig, dec, bg], dim=1)
    L = bag.shape[1]
    perm = torch.rand(B, L, generator=g).argsort(dim=1)
    bag = torch.gather(bag, 1, perm.unsqueeze(-1).expand(B, L, 784))
    if not want_mask:
        return bag.to(device), y.to(device)
    mask = torch.cat([torch.zeros(B, N_SIG), torch.ones(B, m), torch.zeros(B, N_BG)], dim=1)
    mask = torch.gather(mask, 1, perm)
    return bag.to(device), y.to(device), mask.to(device)


def train(mode, data, device, steps=2500, bs=64, lr=5e-4, seed=0):
    torch.manual_seed(seed)
    model = ABMILModel(128, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(1000 + seed)
    is_rep = mode == "rep"
    ws, wl = int(0.4 * steps), max(1, int(0.2 * steps))
    model.train()
    for step in range(steps):
        if is_rep:
            model.pool.rep_scale = min(1.0, max(0.0, (step - ws) / wl))
        m = int(torch.randint(2, 40, (1,), generator=g))            # RANDOM clique size
        bags, y = make_bags(data, bs, m, g, device)
        opt.zero_grad(set_to_none=True); crit(model(bags)[0], y).backward(); opt.step()
    if is_rep:
        model.pool.rep_scale = 1.0
    return model


@torch.no_grad()
def measure(model, data, device, m, batches=10, bs=128):
    model.eval()
    g = torch.Generator().manual_seed(999)
    gm = 0.0; correct = 0; tot = 0
    for _ in range(batches):
        bags, y, mask = make_bags(data, bs, m, g, device, want_mask=True)
        logits, _ = model(bags)
        gm += (model.pool.last_gate * mask).sum(dim=1).sum().item()   # Σ_clique gate (unnormalized)
        correct += (logits.argmax(1) == y).sum().item(); tot += y.numel()
    return gm / tot, 100.0 * correct / tot            # mean clique GATE MASS, accuracy


def r2(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)); sst = sum((y - my) ** 2 for y in ys)
    return 1 - ssr / sst


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load_by_class("train"); te = load_by_class("test")
    ms = [2, 4, 8, 16, 32, 64, 128]
    seeds = [0, 1, 2]
    agg = {"abmil": {m: [] for m in ms}, "rep": {m: [] for m in ms}}
    acc = {"abmil": {m: [] for m in ms}, "rep": {m: [] for m in ms}}
    for name in ("abmil", "rep"):
        for sd in seeds:
            model = train(name, tr, device, seed=sd)
            for m in ms:
                gm, a = measure(model, te, device, m)
                agg[name][m].append(gm); acc[name][m].append(a)
    import statistics as st
    print(f"{'m':>5} | {'ABMIL gate=Σν':>13} {'ABMIL acc':>10} | {'rep gate Σg':>12} {'rep acc':>9}")
    print("-" * 62)
    for m in ms:
        ag = st.mean(agg["abmil"][m]); rg = st.mean(agg["rep"][m])
        aa = st.mean(acc["abmil"][m]); ra = st.mean(acc["rep"][m])
        print(f"{m:>5} | {ag:>13.2f} {aa:>9.1f}% | {rg:>12.2f} {ra:>8.1f}%")
    rg = [st.mean(agg["rep"][m]) for m in ms]
    ag = [st.mean(agg["abmil"][m]) for m in ms]
    print(f"\n[THEORY] rep clique gate mass Σg:  vs log(m) R²={r2([math.log(m) for m in ms], rg):.4f}"
          f"  |  vs m R²={r2(ms, rg):.4f}   (predict: log wins)")
    print(f"[THEORY] ABMIL clique gate Σν=m:  vs m R²={r2(ms, ag):.4f} (should be ~1.0, linear)")
    print(json.dumps({"ms": ms, "rep_gate": rg, "abmil_gate": ag,
                      "rep_acc": [st.mean(acc['rep'][m]) for m in ms],
                      "abmil_acc": [st.mean(acc['abmil'][m]) for m in ms]}))


if __name__ == "__main__":
    main()
