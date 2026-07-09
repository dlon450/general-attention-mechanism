#!/usr/bin/env python3
"""RAMDocs (Wang et al. 2025, "RAG with Conflicting Evidence") — real conflicting-evidence RAG text.

Native RAMDocs is NOT adversarial-by-count (misinfo 0.61/Q vs correct 3.84/Q; majority-vote-by-count
is wrong in only 2% of Qs — the real difficulty is ambiguity). So a native run would be an
uninformative tie. Instead we test the realistic threat that IS our regime — RETRIEVAL POISONING
(PoisonedRAG threat model): an attacker injects k near-duplicate misinfo passages to win BY COUNT.

Setup: a trainable question->document attention aggregator selects an answer by weighted doc voting
(candidate score = sum of attention over docs supporting it). Same shared backbone for every
aggregation rule (parameter-matched), trained with poison AUGMENTATION (random k) so all methods are
robustly trained; then evaluated at fixed poison strengths k. rep uses a separate high LR for lambda.

  softmax  : ABMIL-style relevance attention (count-weighted)   -> fooled by duplicate poison
  majority : uniform weights (pure count vote)                  -> fooled by duplicate poison
  dedup    : drop near-duplicate docs, softmax over survivors   -> cheap defense
  dpp      : exact DPP marginal diversity gate                  -> diversity defense
  rep      : OUR repulsion gate w_i = g_i e^{a_i}/Z             -> anti-redundancy defense

Prints accuracy vs poison-k for each mode."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = "/home/dereklong/mil_data/ramdocs/RAMDocs_test.jsonl"
HASH_DIM = 4096
_tok = re.compile(r"[a-z0-9]+")


def _hash_tokens(toks):
    v = np.zeros(HASH_DIM, dtype=np.float32)
    for t in toks:
        v[hash(t) % HASH_DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def hash_feat(text):
    return _hash_tokens(_tok.findall(text.lower()))


def norm_ans(a):
    return a.lower().strip()


def load():
    """Each usable example -> dict(qfeat, docs=[(feat, cand_idx)], n_cand, gold_idx set, misinfo=[feats])."""
    ex = []
    for line in open(PATH):
        r = json.loads(line)
        gold = set(norm_ans(a) for a in r["gold_answers"])
        # candidate answers = distinct non-unknown answers asserted by docs
        cands = []
        for d in r["documents"]:
            a = norm_ans(d.get("answer", "unknown"))
            if d["type"] != "noise" and a != "unknown" and a not in cands:
                cands.append(a)
        if not cands:
            continue
        gold_idx = {i for i, c in enumerate(cands) if c in gold}
        if not gold_idx:                       # gold must be reachable via some doc
            continue
        docs = []
        misinfo = []
        for d in r["documents"]:
            a = norm_ans(d.get("answer", "unknown"))
            ci = cands.index(a) if (d["type"] != "noise" and a in cands) else -1
            f = hash_feat(d["text"])
            docs.append((f, ci))
            if d["type"] == "misinfo" and ci not in gold_idx and ci >= 0:
                misinfo.append((d["text"], f, ci))          # keep text for paraphrase attack
        ex.append({"q": hash_feat(r["question"]), "docs": docs, "ncand": len(cands),
                   "gold": gold_idx, "misinfo": misinfo})
    return ex


def build_bag(e, k, g, paraphrase=False, drop=0.4):
    """Return (qf, doc_feats, cand_idx, ncand, gold_set). k = # misinfo copies injected.
    paraphrase=True: each copy drops a random `drop` fraction of tokens (near-dup that evades dedup)."""
    docs = list(e["docs"])
    if k > 0 and e["misinfo"]:
        text, f, ci = e["misinfo"][int(torch.randint(len(e["misinfo"]), (1,), generator=g))]
        toks = _tok.findall(text.lower())
        for _ in range(k):
            if paraphrase and len(toks) > 4:
                keep = [t for t in toks if torch.rand(1, generator=g).item() > drop]
                docs.append((_hash_tokens(keep), ci))       # near-duplicate poison copy
            else:
                docs.append((f, ci))                        # exact-duplicate poison copy
    feats = torch.from_numpy(np.stack([d[0] for d in docs]))
    cidx = torch.tensor([d[1] for d in docs])
    qf = torch.from_numpy(e["q"])
    return qf, feats, cidx, e["ncand"], e["gold"]


class Aggregator(nn.Module):
    def __init__(self, dim=256, mode="softmax", lambda_init=1.0):
        super().__init__()
        self.mode = mode
        self.embed = nn.Sequential(nn.Linear(HASH_DIM, dim), nn.GELU())
        self.q = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.dedup_thr = 0.9
        if mode == "rep":
            self.beta = nn.Parameter(torch.tensor(0.5))
            self.tau = nn.Parameter(torch.tensor(0.0))
            self.log_lambda = nn.Parameter(torch.tensor(math.log(math.expm1(lambda_init))))
            self.rep_scale = 1.0

    def weights(self, qf, Hf):                 # qf (D,), Hf (N,D) -> w (N,)
        H = self.embed(Hf)
        q = self.q(self.embed(qf.unsqueeze(0)))            # (1,dim)
        a = (H @ q.squeeze(0)) * self.scale               # (N,)
        if self.mode == "majority":
            return torch.ones_like(a) / a.numel()
        if self.mode == "softmax":
            return torch.softmax(a, 0)
        Hn = F.normalize(H, dim=-1)
        if self.mode == "dedup":
            dup = (Hn @ Hn.t()) > self.dedup_thr
            earlier = torch.tril(dup, -1).any(1)
            return torch.softmax(a.masked_fill(earlier, float("-inf")), 0)
        if self.mode == "dpp":
            S = Hn @ Hn.t()
            qual = torch.exp(0.5 * (a - a.max()))
            L = qual.unsqueeze(1) * S * qual.unsqueeze(0)
            K = torch.linalg.solve(L + torch.eye(a.numel(), device=a.device), L)
            gg = torch.diagonal(K).clamp(0, 1)
            ex = torch.exp(a - a.max()) * gg
            return ex / ex.sum().clamp_min(1e-9)
        if self.mode == "rep":
            lam = F.softplus(self.log_lambda) * self.rep_scale
            base = self.beta * (a - self.tau)
            gg = torch.sigmoid(base)
            m = (gg.unsqueeze(1) * Hn).sum(0)
            r = Hn @ m
            gg = torch.sigmoid(base - self.beta * lam * r)
            ex = torch.exp(a - a.max()) * gg
            return ex / ex.sum().clamp_min(1e-9)
        raise ValueError(self.mode)

    def forward(self, qf, Hf, cidx, ncand):
        w = self.weights(qf, Hf)
        scores = torch.zeros(ncand, device=w.device)
        valid = cidx >= 0
        scores = scores.index_add(0, cidx[valid], w[valid])
        return torch.log(scores + 1e-9)                   # logits over candidates


def run(mode, tr, te, device, epochs=40, lr=1e-3, seed=0, kmax_train=4, lambda_lr=None,
        eval_ks=(0, 1, 2, 4, 8), paraphrase=False, drop=0.4):
    torch.manual_seed(seed)
    model = Aggregator(mode=mode).to(device)
    if mode == "rep" and lambda_lr:
        rep_p = [model.beta, model.tau, model.log_lambda]; rid = {id(p) for p in rep_p}
        base = [p for p in model.parameters() if id(p) not in rid]
        opt = torch.optim.AdamW([{"params": base, "lr": lr}, {"params": rep_p, "lr": lambda_lr}])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    is_rep = mode == "rep"; total = epochs * len(tr); step = 0
    train_epochs = 0 if mode == "majority" else epochs   # majority is a param-free fixed rule
    for ep in range(train_epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 17 + ep)
        for i in torch.randperm(len(tr), generator=g).tolist():
            if is_rep:
                model.rep_scale = min(1.0, max(0.0, (step - 0.3 * total) / (0.2 * total)))
            k = int(torch.randint(0, kmax_train + 1, (1,), generator=g))   # poison augmentation
            qf, feats, cidx, nc, gold = build_bag(tr[i], k, g, paraphrase=paraphrase, drop=drop)
            logits = model(qf.to(device), feats.to(device), cidx.to(device), nc)
            # multi-gold: train toward the highest-scoring gold candidate
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
            ge = torch.Generator().manual_seed(999)
            cor = tot = 0
            for e in te:
                if not e["misinfo"]:
                    continue                    # fixed poisonable population across all k
                qf, feats, cidx, nc, gold = build_bag(e, k, ge, paraphrase=paraphrase, drop=drop)
                pred = model(qf.to(device), feats.to(device), cidx.to(device), nc).argmax().item()
                cor += int(pred in gold); tot += 1
            accs[k] = 100.0 * cor / max(tot, 1)
    lam = F.softplus(model.log_lambda).item() if is_rep else None
    return accs, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["softmax", "majority", "dedup", "dpp", "rep"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lambda-lr", type=float, default=0.02)
    ap.add_argument("--poison", choices=["exact", "paraphrase"], default="exact")
    ap.add_argument("--drop", type=float, default=0.4, help="token-drop fraction for paraphrase poison")
    args = ap.parse_args()
    paraphrase = args.poison == "paraphrase"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = load()
    ntr = int(0.8 * len(ex))
    perm = torch.randperm(len(ex), generator=torch.Generator().manual_seed(0)).tolist()
    tr = [ex[i] for i in perm[:ntr]]; te = [ex[i] for i in perm[ntr:]]
    npois = sum(bool(e["misinfo"]) for e in te)
    print(f"usable Q: {len(ex)} (train {len(tr)}, test {len(te)}; test w/ misinfo doc = {npois})")
    print(f"poison mode: {args.poison}" + (f" (drop={args.drop})" if paraphrase else ""))
    eval_ks = (0, 1, 2, 4, 8)
    print(f"\n{'mode':9}" + "".join(f"{'k='+str(k):>9}" for k in eval_ks) + "   lambda")
    for mode in args.modes:
        acc_seeds = {k: [] for k in eval_ks}; lams = []
        for s in range(args.seeds):
            accs, lam = run(mode, tr, te, device, epochs=args.epochs, seed=s,
                            lambda_lr=args.lambda_lr, eval_ks=eval_ks,
                            paraphrase=paraphrase, drop=args.drop)
            for k in eval_ks:
                acc_seeds[k].append(accs[k])
            if lam is not None:
                lams.append(lam)
        row = "".join(f"{np.mean(acc_seeds[k]):>9.1f}" for k in eval_ks)
        lam_s = f"   {np.mean(lams):.2f}" if lams else ""
        print(f"{mode:9}{row}{lam_s}", flush=True)


if __name__ == "__main__":
    main()
