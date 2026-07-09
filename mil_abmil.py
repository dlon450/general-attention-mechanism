#!/usr/bin/env python3
"""Fair SOTA MIL comparison in ONE shared backbone (encoder -> gated-ABMIL scorer -> pool
-> head). Only the pooling normalization/gating differs, so the comparison is apples-to-apples
and parameter-matched (our method adds 3 scalars):

  abmil       : gated-ABMIL (Ilse et al. 2018) = gated attention score + softmax   [canonical MIL SOTA]
  aem         : abmil + Attention-Entropy-Maximization regularizer (2024)          [anti-concentration SOTA]
  sparsemax   : gated score + sparsemax normalization
  entmax15    : gated score + entmax-1.5 normalization
  rep         : gated score + OUR repulsion gate w_k = g_k e^{a_k}/Z,
                g_k = sigmoid(beta(a_k - tau - lambda*r_k)), r_k = <Hn_k, sum_j g0_j Hn_j>
                (embedding-space anti-redundancy, with a warmup ramp on lambda)

Same encoder, same steps, ~same FLOPs (pooling is tiny vs the encoder at N~32).
Prints one JSON line (test acc + eval CE + attention entropy).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from f2_sweep import entmax15, sparsemax
from mil_mnist import Encoder, NUM_CLASSES, load_by_class, make_bag_batch


class ABMILPool(nn.Module):
    def __init__(self, dim, att=128, mode="abmil", beta_init=0.5, lambda_init=1.0,
                 mf_iters=1, fixed_lambda=None):
        super().__init__()
        self.mode = mode
        self.mf_iters = int(mf_iters)          # # mean-field iterations for the repulsion gate
        self.fixed_lambda = fixed_lambda        # if set, use a FIXED (non-learned) repulsion strength
        self.V = nn.Linear(dim, att)      # gated-ABMIL scorer (shared by ALL modes)
        self.U = nn.Linear(dim, att)
        self.w = nn.Linear(att, 1)
        self.dedup_thr = 0.9              # cosine threshold for the dedup/countnorm baselines
        if mode == "rep":
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
            self.tau = nn.Parameter(torch.tensor(0.0))
            self.log_lambda = nn.Parameter(torch.tensor(math.log(math.expm1(lambda_init))))
            self.rep_scale = 1.0

    def scores(self, H):  # H (B,N,dim) -> a (B,N)
        return self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))).squeeze(-1)

    def forward(self, H):
        a = self.scores(H)
        if self.mode in ("abmil", "aem"):
            w = torch.softmax(a, dim=1)
        elif self.mode == "sparsemax":
            w = sparsemax(a, dim=1)
        elif self.mode == "entmax15":
            w = entmax15(a, dim=1)
        elif self.mode in ("dedup", "countnorm"):
            # CHEAP DEFENSES a reviewer demands: remove count-domination via embedding
            # similarity, no attention change. dedup=keep 1 per near-dup group; countnorm=
            # down-weight each instance by its near-dup cluster size (self-consistency voting).
            Hn = F.normalize(H, dim=-1)
            dup = torch.matmul(Hn, Hn.transpose(1, 2)) > self.dedup_thr  # (B,N,N)
            if self.mode == "dedup":
                earlier = torch.tril(dup, diagonal=-1).any(dim=2)        # has an earlier duplicate
                w = torch.softmax(a.masked_fill(earlier, float("-inf")), dim=1)
            else:  # countnorm
                csize = dup.sum(dim=2).clamp_min(1).float()
                w = torch.softmax(a, dim=1) / csize
                w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-9)
        elif self.mode == "dpp":
            # Exact DPP marginal inclusion (DppNet/DPP-A lineage): quality x diversity L-ensemble
            # L = diag(e^{a/2}) S diag(e^{a/2}), S=key-cosine; marginal K = L(L+I)^{-1}; g_i = K_ii.
            Hn = F.normalize(H, dim=-1)
            Ssim = torch.matmul(Hn, Hn.transpose(1, 2))
            qual = torch.exp(0.5 * (a - a.amax(1, keepdim=True)))
            Lk = qual.unsqueeze(2) * Ssim * qual.unsqueeze(1)
            eye = torch.eye(a.shape[1], device=a.device).unsqueeze(0)
            K = torch.linalg.solve(Lk + eye, Lk)
            g = torch.diagonal(K, dim1=1, dim2=2).clamp(0.0, 1.0)
            ex = torch.exp(a - a.amax(1, keepdim=True)) * g
            w = ex / ex.sum(1, keepdim=True).clamp_min(1e-9)
        elif self.mode == "tome":
            # ToMe (as published): merge near-dup clusters, keep 1 rep with PROPORTIONAL attention
            # (+log cluster-size) — preserves the merged mass, i.e. does NOT drop count-domination.
            Hn = F.normalize(H, dim=-1)
            dup = torch.matmul(Hn, Hn.transpose(1, 2)) > self.dedup_thr
            csize = dup.sum(dim=2).clamp_min(1).float()
            earlier = torch.tril(dup, diagonal=-1).any(dim=2)
            w = torch.softmax((a + torch.log(csize)).masked_fill(earlier, float("-inf")), dim=1)
        elif self.mode == "rep":
            Hn = F.normalize(H, dim=-1)
            lam_base = self.fixed_lambda if self.fixed_lambda is not None else F.softplus(self.log_lambda)
            lam = lam_base * self.rep_scale
            base = self.beta * (a - self.tau)
            g = torch.sigmoid(base)                                    # g0
            for _ in range(self.mf_iters):                            # mean-field iterations
                m = torch.einsum("bn,bnd->bd", g, Hn)                # sum_j g_j Hn_j
                r = torch.einsum("bnd,bd->bn", Hn, m)                # r_k = <Hn_k, m>
                g = torch.sigmoid(base - self.beta * lam * r)
            ex = torch.exp(a - a.amax(dim=1, keepdim=True)) * g
            w = ex / ex.sum(dim=1, keepdim=True).clamp_min(1e-9)
        else:
            raise ValueError(self.mode)
        # stash the (unnormalized) gate for analysis: rep -> g; first-order -> nu==1
        self.last_gate = g.detach() if self.mode == "rep" else torch.ones_like(w)
        bag = torch.einsum("bn,bnd->bd", w, H)
        return bag, w


class ABMILModel(nn.Module):
    def __init__(self, dim, mode, beta_init=0.5, lambda_init=1.0, mf_iters=1, fixed_lambda=None):
        super().__init__()
        self.enc = Encoder(dim)
        self.pool = ABMILPool(dim, mode=mode, beta_init=beta_init, lambda_init=lambda_init,
                              mf_iters=mf_iters, fixed_lambda=fixed_lambda)
        self.head = nn.Linear(dim, NUM_CLASSES)

    def forward(self, bags):  # (B,N,784)
        B, N = bags.shape[:2]
        H = self.enc(bags.reshape(B * N, 784)).reshape(B, N, -1)
        bag, w = self.pool(H)
        return self.head(bag), w


@torch.no_grad()
def evaluate(model, cls, args, device):
    model.eval()
    g = torch.Generator().manual_seed(999)
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = tot = 0; loss = 0.0; ent = 0.0
    for _ in range(args.test_batches):
        bags, y = make_bag_batch(cls, args.batch_size, args.task, args.N, g,
                                 n_sig=args.n_sig, m_dec=args.m_dec, device=device)
        lg, w = model(bags)
        loss += ce(lg, y).item()
        ent += (-(w.clamp_min(1e-9).log() * w).sum(1)).sum().item()
        correct += (lg.argmax(1) == y).sum().item(); tot += y.numel()
    return 100.0 * correct / tot, loss / tot, ent / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["abmil", "aem", "sparsemax", "entmax15", "dedup", "countnorm", "tome", "dpp", "rep"], default="rep")
    p.add_argument("--task", choices=["needle", "redundancy", "majority"], default="redundancy")
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--n-sig", type=int, default=3)
    p.add_argument("--m-dec", type=int, default=None)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--aem-coef", type=float, default=0.1, help="attention-entropy-max weight")
    p.add_argument("--mf-iters", type=int, default=1, help="mean-field iterations for rep gate")
    p.add_argument("--lambda-init", type=float, default=1.0, help="initial (learned) repulsion strength")
    p.add_argument("--lambda-lr", type=float, default=None, help="separate high LR for lambda/tau/beta")
    p.add_argument("--fixed-lambda", type=float, default=None, help="fixed (non-learned) repulsion strength")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-batches", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tr = load_by_class("train"); te = load_by_class("test")
    model = ABMILModel(args.dim, args.mode, lambda_init=args.lambda_init,
                       mf_iters=args.mf_iters, fixed_lambda=args.fixed_lambda).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    if args.mode == "rep" and args.lambda_lr:
        rep_p = [model.pool.log_lambda, model.pool.tau, model.pool.beta]
        rep_ids = {id(p) for p in rep_p}
        base_p = [p for p in model.parameters() if id(p) not in rep_ids]
        opt = torch.optim.AdamW([{"params": base_p, "lr": args.lr},
                                 {"params": rep_p, "lr": args.lambda_lr}], weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(10_000 + args.seed)
    is_rep = args.mode == "rep"
    warm_start = int(0.4 * args.steps); warm_len = max(1, int(0.2 * args.steps))
    model.train()
    for step in range(args.steps):
        if is_rep:
            model.pool.rep_scale = min(1.0, max(0.0, (step - warm_start) / warm_len))
        bags, y = make_bag_batch(tr, args.batch_size, args.task, args.N, g,
                                 n_sig=args.n_sig, m_dec=args.m_dec, device=device)
        opt.zero_grad(set_to_none=True)
        lg, w = model(bags)
        loss = crit(lg, y)
        if args.mode == "aem":  # maximize attention entropy (spread attention)
            ent = (-(w.clamp_min(1e-9).log() * w).sum(1)).mean()
            loss = loss - args.aem_coef * ent
        loss.backward(); opt.step()
    if is_rep:
        model.pool.rep_scale = 1.0
        print(f"[diag seed{args.seed}] learned lambda="
              f"{F.softplus(model.pool.log_lambda).item():.3f}", file=sys.stderr, flush=True)
    acc, loss, ent = evaluate(model, te, args, device)
    print(json.dumps({"task": args.task, "mode": args.mode, "N": args.N, "seed": args.seed,
                      "params": n_params, "test_acc": round(acc, 3), "test_loss": round(loss, 4),
                      "attn_entropy": round(ent, 3), "chance": round(100.0 / NUM_CLASSES, 2)}))


if __name__ == "__main__":
    main()
