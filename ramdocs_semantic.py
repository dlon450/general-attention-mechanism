#!/usr/bin/env python3
"""RAMDocs semantic-poisoning robustness — the make-or-break test.

Real TinyLlama embeddings (centered + top-10 removed, L2-normed) as the semantic space. A trainable
question->doc relevance aggregator selects the answer by weighted doc voting. Attack = inject k
SURFACE-DIVERSE paraphrase copies of a misinfo doc (token-drop; copy-copy cos 0.76, copy-legit 0.00
-> evades bag-of-words dedup but semantically tight). All methods trained with poison augmentation.

Baselines vs OUR rep, all in the SAME frozen semantic space (parameter-matched relevance head):
  softmax   : relevance attention (count-weighted)
  majority  : uniform vote (pure count)
  sem-dedupT: SEMANTIC dedup at cosine threshold T (the STRONG cheap defense) -- swept T
  rep       : our soft/global/learned repulsion gate (redundancy in the semantic space)

Reports CLEAN (k=0) and POISONED accuracy. Honest win condition: rep beats the best sem-dedup
operating point AND sem-dedup pays a clean-accuracy cost (removing legit same-answer docs)."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB = "/data/users/dereklong/scratch/general-attention-mechanism/ramdocs_emb.npz"
PAR = "/data/users/dereklong/scratch/general-attention-mechanism/ramdocs_para_emb.npz"
PATH = "/home/dereklong/mil_data/ramdocs/RAMDocs_test.jsonl"


def norm_ans(a):
    return a.lower().strip()


def load():
    d = np.load(EMB); demb = d["demb"].astype(np.float32); qemb = d["qemb"].astype(np.float32)
    meta = d["meta"]
    par = np.load(PAR)
    pool = {k: par[f"v_{k}"].astype(np.float32) for k in par["keys"]}
    rows = [json.loads(l) for l in open(PATH)]
    per_q = {}; idx = 0
    for (qi, di) in meta:
        per_q.setdefault(int(qi), []).append((int(di), idx)); idx += 1
    ex = []
    for qi, r in enumerate(rows):
        gold = set(norm_ans(a) for a in r["gold_answers"])
        cands = []
        for d_ in r["documents"]:
            a = norm_ans(d_.get("answer", "unknown"))
            if d_["type"] != "noise" and a != "unknown" and a not in cands:
                cands.append(a)
        if not cands:
            continue
        gold_idx = {i for i, c in enumerate(cands) if c in gold}
        if not gold_idx:
            continue
        docs = []; misinfo = []
        for (di, gidx) in per_q[qi]:
            d_ = r["documents"][di]; a = norm_ans(d_.get("answer", "unknown"))
            ci = cands.index(a) if (d_["type"] != "noise" and a in cands) else -1
            docs.append((demb[gidx], ci))
            key = f"{qi}_{di}"
            if d_["type"] == "misinfo" and ci >= 0 and ci not in gold_idx and key in pool:
                misinfo.append((pool[key], ci))
        ex.append({"q": qemb[qi], "docs": docs, "ncand": len(cands),
                   "gold": gold_idx, "misinfo": misinfo})
    return ex


def fit_transform(ex, tr_idx):
    """center + remove top-10 PCs (fit on train docs), return a closure applying it + L2-norm."""
    X = np.stack([ex[i]["docs"][j][0] for i in tr_idx for j in range(len(ex[i]["docs"]))])
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    top = Vt[:10]

    def tf(v):                                   # v: (...,2048) np -> torch normalized
        m = v - mu
        m = m - (m @ top.T) @ top
        t = torch.from_numpy(m.astype(np.float32))
        return F.normalize(t, dim=-1)
    return tf


class Agg(nn.Module):
    def __init__(self, dim=2048, proj=256, mode="softmax", thr=0.5, lambda_init=1.0):
        super().__init__()
        self.mode = mode; self.thr = thr
        self.wq = nn.Linear(dim, proj); self.wk = nn.Linear(dim, proj)
        self.scale = proj ** -0.5
        if mode == "rep":
            self.beta = nn.Parameter(torch.tensor(0.5))
            self.tau = nn.Parameter(torch.tensor(0.0))
            self.log_lambda = nn.Parameter(torch.tensor(math.log(math.expm1(lambda_init))))
            self.rep_scale = 1.0

    def weights(self, q, E):                      # q (dim,), E (N,dim) normalized frozen embeddings
        a = (self.wk(E) @ self.wq(q)) * self.scale       # (N,) relevance
        if self.mode == "majority":
            return torch.ones(E.shape[0], device=E.device) / E.shape[0]
        if self.mode == "softmax":
            return torch.softmax(a, 0)
        if self.mode == "dedup":                          # SEMANTIC dedup at cosine thr
            dup = (E @ E.t()) > self.thr
            earlier = torch.tril(dup, -1).any(1)
            return torch.softmax(a.masked_fill(earlier, float("-inf")), 0)
        if self.mode == "rep":
            lam = F.softplus(self.log_lambda) * self.rep_scale
            base = self.beta * (a - self.tau)
            g = torch.sigmoid(base)
            m = (g.unsqueeze(1) * E).sum(0)
            r = E @ m
            g = torch.sigmoid(base - self.beta * lam * r)
            ex = torch.exp(a - a.max()) * g
            return ex / ex.sum().clamp_min(1e-9)
        raise ValueError(self.mode)

    def forward(self, q, E, cidx, ncand):
        w = self.weights(q, E)
        s = torch.zeros(ncand, device=E.device)
        valid = cidx >= 0
        s = s.index_add(0, cidx[valid], w[valid])
        return torch.log(s + 1e-9)


def build(e, k, tf, g):
    docs = list(e["docs"])
    cidx = [d[1] for d in docs]
    embs = [d[0] for d in docs]
    if k > 0 and e["misinfo"]:
        pool, ci = e["misinfo"][int(torch.randint(len(e["misinfo"]), (1,), generator=g))]
        sel = torch.randint(len(pool), (k,), generator=g).tolist()
        for j in sel:
            embs.append(pool[j]); cidx.append(ci)
    E = tf(np.stack(embs))                        # (N,2048) normalized torch
    q = tf(e["q"][None])[0]
    return q, E, torch.tensor(cidx), e["ncand"], e["gold"]


def run(mode, tr, te, tf, device, thr=0.5, epochs=30, lr=1e-3, seed=0, kmax=6, lambda_lr=0.02,
        eval_ks=(0, 1, 2, 4, 8)):
    torch.manual_seed(seed)
    model = Agg(mode=mode, thr=thr).to(device)
    if mode == "rep" and lambda_lr:
        rp = [model.beta, model.tau, model.log_lambda]; rid = {id(p) for p in rp}
        bp = [p for p in model.parameters() if id(p) not in rid]
        opt = torch.optim.AdamW([{"params": bp, "lr": lr}, {"params": rp, "lr": lambda_lr}])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    is_rep = mode == "rep"; total = epochs * len(tr); step = 0
    tep = 0 if mode == "majority" else epochs
    for ep in range(tep):
        model.train()
        g = torch.Generator().manual_seed(seed * 17 + ep)
        for i in torch.randperm(len(tr), generator=g).tolist():
            if is_rep:
                model.rep_scale = min(1.0, max(0.0, (step - 0.3 * total) / (0.2 * total)))
            k = int(torch.randint(0, kmax + 1, (1,), generator=g))
            q, E, cidx, nc, gold = build(tr[i], k, tf, g)
            logits = model(q.to(device), E.to(device), cidx.to(device), nc)
            gi = max(gold, key=lambda c: logits[c].item())
            opt.zero_grad(set_to_none=True)
            crit(logits.unsqueeze(0), torch.tensor([gi], device=device)).backward()
            opt.step(); step += 1
    if is_rep:
        model.rep_scale = 1.0
    model.eval()
    accs = {}
    with torch.no_grad():
        for k in eval_ks:
            ge = torch.Generator().manual_seed(999); cor = tot = 0
            for e in te:
                if not e["misinfo"]:
                    continue
                q, E, cidx, nc, gold = build(e, k, tf, ge)
                pred = model(q.to(device), E.to(device), cidx.to(device), nc).argmax().item()
                cor += int(pred in gold); tot += 1
            accs[k] = 100.0 * cor / max(tot, 1)
    lam = F.softplus(model.log_lambda).item() if is_rep else None
    return accs, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = load()
    perm = torch.randperm(len(ex), generator=torch.Generator().manual_seed(0)).tolist()
    ntr = int(0.8 * len(ex)); tr_idx = perm[:ntr]; te_idx = perm[ntr:]
    tf = fit_transform(ex, tr_idx)
    tr = [ex[i] for i in tr_idx]; te = [ex[i] for i in te_idx]
    npois = sum(bool(e["misinfo"]) for e in te)
    eval_ks = (0, 1, 2, 4, 8)
    print(f"usable Q {len(ex)} (train {len(tr)}, test {len(te)}; poisonable test {npois})")
    print(f"\n{'mode':11}" + "".join(f"{'k='+str(k):>8}" for k in eval_ks) + "   lambda")
    specs = [("softmax", None), ("majority", None), ("dedup", 0.3), ("dedup", 0.5),
             ("dedup", 0.7), ("rep", None)]
    for mode, thr in specs:
        acc = {k: [] for k in eval_ks}; lams = []
        for s in range(args.seeds):
            a, lam = run(mode, tr, te, tf, device, thr=(thr or 0.5), epochs=args.epochs,
                         seed=s, eval_ks=eval_ks)
            for k in eval_ks:
                acc[k].append(a[k])
            if lam is not None:
                lams.append(lam)
        name = f"{mode}@{thr}" if mode == "dedup" else mode
        row = "".join(f"{np.mean(acc[k]):>8.1f}" for k in eval_ks)
        print(f"{name:11}{row}" + (f"   {np.mean(lams):.2f}" if lams else ""), flush=True)


if __name__ == "__main__":
    main()
