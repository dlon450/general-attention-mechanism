#!/usr/bin/env python3
"""Full protocol for the consensus mechanism study (locks the E1b result). Writes JSONL results that
aggregate.py turns into CI'd tables. Shardable by --seeds so experiments run across GPUs.

  --exp A  learning curve (arms x n_train x seeds) at alpha=1,gamma=0.8   -> paired-bootstrap CIs
  --exp B  worst-case-over-alpha (arms x alpha x seeds) at n_train=800    -> robustness
  --exp C  OOD extrapolation: train n_orig in [3,6], test n_orig in [8,10] vs IID control
  --exp D  latency at L=64 (fwd and fwd+bwd wall-clock per arm)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import torch

from consensus_models import Model, make_split, run_arm
from task_consensus import Cfg, prototypes

ARMS_ALL = ["softmax", "set_transformer", "prov_concat", "relation_bias", "m2_prov", "m2_prov_x", "m2_prov_r"]


def train_eval(arm, cfg_tr, cfg_te, mu, alpha, gamma, n_train, seed, device, steps):
    tr = make_split(n_train, cfg_tr, mu, alpha, gamma, 100 + seed, device)
    va = make_split(1500, cfg_tr, mu, alpha, gamma, 500 + seed, device)
    te = make_split(2500, cfg_te, mu, alpha, gamma, 777, device)          # FROZEN test (fixed seed)
    return run_arm(arm, tr, va, te, cfg_tr.C, device, steps=steps, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS_ALL)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1600, 3200, 6000])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Cfg(); mu = prototypes(cfg, np.random.default_rng(0))
    f = open(args.out, "a")

    def rec(**kw):
        f.write(json.dumps(kw) + "\n"); f.flush()

    if args.exp == "A":
        for arm in args.arms:
            for nt in args.sizes:
                for s in args.seeds:
                    acc, npar = train_eval(arm, cfg, cfg, mu, 1.0, args.gamma, nt, s, device, args.steps)
                    rec(exp="A", arm=arm, n_train=nt, alpha=1.0, gamma=args.gamma, seed=s,
                        acc=acc, params=npar)
                    print(f"A {arm} n={nt} seed={s} -> {acc:.1f}", flush=True)
    elif args.exp == "B":
        for arm in args.arms:
            for al in args.alphas:
                for s in args.seeds:
                    acc, npar = train_eval(arm, cfg, cfg, mu, al, args.gamma, 800, s, device, args.steps)
                    rec(exp="B", arm=arm, n_train=800, alpha=al, gamma=args.gamma, seed=s, acc=acc)
                    print(f"B {arm} a={al} seed={s} -> {acc:.1f}", flush=True)
    elif args.exp == "C":
        cfg_small = replace(cfg, n_orig=(3, 6)); cfg_big = replace(cfg, n_orig=(8, 10))
        for arm in args.arms:
            for s in args.seeds:
                iid, _ = train_eval(arm, cfg_small, cfg_small, mu, 1.0, args.gamma, 3200, s, device, args.steps)
                ood, _ = train_eval(arm, cfg_small, cfg_big, mu, 1.0, args.gamma, 3200, s, device, args.steps)
                rec(exp="C", arm=arm, seed=s, iid=iid, ood=ood)
                print(f"C {arm} seed={s} -> iid {iid:.1f} ood {ood:.1f}", flush=True)
    elif args.exp == "D":
        for arm in args.arms:
            m = Model(arm, cfg.C).to(device)
            V, M, P, Y = make_split(128, cfg, mu, 1.0, args.gamma, 1, device)
            for _ in range(5):
                m(V, M, P)                                               # warmup
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            for _ in range(50):
                with torch.no_grad():
                    m(V, M, P)
            torch.cuda.synchronize() if device.type == "cuda" else None
            fwd = (time.perf_counter() - t0) / 50 * 1000
            crit = torch.nn.CrossEntropyLoss(); opt = torch.optim.SGD(m.parameters(), lr=0.0)
            t0 = time.perf_counter()
            for _ in range(50):
                opt.zero_grad(set_to_none=True); crit(m(V, M, P), Y).backward(); opt.step()
            torch.cuda.synchronize() if device.type == "cuda" else None
            fb = (time.perf_counter() - t0) / 50 * 1000
            rec(exp="D", arm=arm, fwd_ms=fwd, fwdbwd_ms=fb,
                params=sum(p.numel() for p in m.parameters()))
            print(f"D {arm} fwd {fwd:.2f}ms fwd+bwd {fb:.2f}ms", flush=True)
    f.close()


if __name__ == "__main__":
    main()
