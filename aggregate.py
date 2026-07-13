#!/usr/bin/env python3
"""Aggregate results_full/*.jsonl into CI'd tables (Exp A learning curve + paired bootstrap,
Exp B worst-case-over-alpha, Exp C OOD, Exp D latency)."""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

ROWS = []
for fn in glob.glob("results_full/*.jsonl"):
    for line in open(fn):
        line = line.strip()
        if line:
            ROWS.append(json.loads(line))


def boot_ci(deltas, iters=10000):
    rng = np.random.default_rng(0); a = np.array(deltas, float); n = len(a)
    ms = np.array([a[rng.integers(0, n, n)].mean() for _ in range(iters)])
    return a.mean(), np.percentile(ms, 2.5), np.percentile(ms, 97.5)


def exp_A():
    d = defaultdict(dict)                                    # (arm) -> {n_train: {seed: acc}}
    sizes = set()
    for r in ROWS:
        if r.get("exp") == "A":
            d[r["arm"]].setdefault(r["n_train"], {})[r["seed"]] = r["acc"]; sizes.add(r["n_train"])
    sizes = sorted(sizes)
    arms = [a for a in ["softmax", "set_transformer", "prov_concat", "relation_bias",
                        "m2_prov", "m2_prov_x", "m2_prov_r"] if a in d]
    print("== Exp A: learning curve (test acc mean±std over seeds; chance=50) ==")
    print(f"{'arm':16}" + "".join(f"{('n='+str(s)):>13}" for s in sizes))
    for a in arms:
        cells = []
        for s in sizes:
            v = list(d[a].get(s, {}).values())
            cells.append(f"{np.mean(v):.1f}±{np.std(v):.1f}" if v else "-")
        print(f"{a:16}" + "".join(f"{c:>13}" for c in cells))
    # paired bootstrap Delta(m2_prov_r - prov_concat) per size
    if "m2_prov_r" in d and "prov_concat" in d:
        print("\n  paired Δ(m2_prov_r − prov_concat) with 95% bootstrap CI:")
        for s in sizes:
            common = sorted(set(d["m2_prov_r"].get(s, {})) & set(d["prov_concat"].get(s, {})))
            if not common:
                continue
            deltas = [d["m2_prov_r"][s][k] - d["prov_concat"][s][k] for k in common]
            m, lo, hi = boot_ci(deltas)
            flag = "  *win*" if lo > 0 else ("  (baseline wins)" if hi < 0 else "  (tie)")
            print(f"    n={s:<5} Δ={m:+5.1f}  [{lo:+.1f}, {hi:+.1f}]  (k={len(common)}){flag}")


def exp_B():
    d = defaultdict(lambda: defaultdict(list))
    alphas = set()
    for r in ROWS:
        if r.get("exp") == "B":
            d[r["arm"]][r["alpha"]].append(r["acc"]); alphas.add(r["alpha"])
    if not d:
        return
    alphas = sorted(alphas)
    print("\n== Exp B: robustness over alpha (n=800; worst-case = adaptive adversary) ==")
    print(f"{'arm':16}" + "".join(f"{('α='+str(a)):>11}" for a in alphas) + f"{'WORST':>9}")
    for a in ["softmax", "prov_concat", "relation_bias", "m2_prov_r"]:
        if a not in d:
            continue
        means = [np.mean(d[a][al]) if d[a][al] else float("nan") for al in alphas]
        print(f"{a:16}" + "".join(f"{m:>11.1f}" for m in means) + f"{min(means):>9.1f}")


def exp_C():
    d = defaultdict(lambda: {"iid": [], "ood": []})
    for r in ROWS:
        if r.get("exp") == "C":
            d[r["arm"]]["iid"].append(r["iid"]); d[r["arm"]]["ood"].append(r["ood"])
    if not d:
        return
    print("\n== Exp C: OOD extrapolation (train n_orig∈[3,6], test n_orig∈[8,10]) ==")
    print(f"{'arm':16}{'IID':>10}{'OOD':>10}{'drop':>8}")
    for a in ["prov_concat", "relation_bias", "m2_prov", "m2_prov_r"]:
        if a not in d:
            continue
        i, o = np.mean(d[a]["iid"]), np.mean(d[a]["ood"])
        print(f"{a:16}{i:>10.1f}{o:>10.1f}{i-o:>8.1f}")


def exp_D():
    d = {r["arm"]: r for r in ROWS if r.get("exp") == "D"}
    if not d:
        return
    base = d.get("softmax", {}).get("fwd_ms", 1.0)
    print("\n== Exp D: latency at L=64 (per-batch ms) ==")
    print(f"{'arm':16}{'fwd_ms':>9}{'fwd+bwd':>10}{'params':>9}{'fwd× vs softmax':>17}")
    for a in ["softmax", "set_transformer", "prov_concat", "relation_bias", "m2_prov", "m2_prov_x", "m2_prov_r"]:
        if a not in d:
            continue
        r = d[a]
        print(f"{a:16}{r['fwd_ms']:>9.2f}{r['fwdbwd_ms']:>10.2f}{r['params']:>9,}{r['fwd_ms']/base:>16.2f}x")


if __name__ == "__main__":
    print(f"{len(ROWS)} records\n")
    exp_A(); exp_B(); exp_C(); exp_D()
