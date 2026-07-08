#!/usr/bin/env python3
"""Sweep F2 families (as deterministic mean-field gates) across regimes.

Each F2 defines the gate logit inside  w_i = gate_i * exp(a_i) / sum_j gate_j exp(a_j):

  mha       : dense softmax (gate = 1)                                   [baseline]
  modular   : gate = sigmoid(beta (a_i - tau))                          per-token sparsity
  card      : gate = sigmoid(beta (a_i - tau - gamma * N)),  N=sum_j g0_j    learned budget
  rep_key   : gate = sigmoid(beta (a_i - tau - lam * r_i)),  r_i=sum_j g0_j <k_i,k_j>/sqrt d
  rep_val   : same but r_i uses <v_i,v_j>                                value-space repulsion
  submod    : gate = sigmoid(beta (delta_i - tau)),  delta_i=log(1+e^{a_i}/(Ssoft_excl+eps))
              (submodular / diminishing-returns coverage)

All variants are parameter-matched to nn.MultiheadAttention up to O(dim) gate params.
Tasks: needle (long-context dilution) and redundancy (identical-clique pathology).
Reports test accuracy AND eval cross-entropy. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_SEED = 1234
TEST_SEED = 999


# --------------------------- F2 attention module ---------------------------
class F2Attention(nn.Module):
    def __init__(self, dim, num_heads, f2, beta_init=0.5, coupling_init=None, eps=1e-6):
        super().__init__()
        self.dim, self.h = dim, num_heads
        self.hd = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.hd)
        self.f2 = f2
        self.eps = eps
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.beta = nn.Parameter(torch.full((num_heads,), float(beta_init)))
        self.w_tau = nn.Parameter(torch.zeros(num_heads, self.hd))
        self.b_tau = nn.Parameter(torch.zeros(num_heads))
        if coupling_init is None:
            coupling_init = {"card": 0.1, "rep_key": 1.0, "rep_val": 1.0, "submod": 1.0}.get(f2, 0.0)
        if f2 in ("card", "rep_key", "rep_val", "submod"):
            inv = math.log(math.expm1(coupling_init)) if coupling_init > 0 else -8.0
            self.log_coupling = nn.Parameter(torch.full((num_heads,), float(inv)))
        else:
            self.register_parameter("log_coupling", None)

    def coupling(self):
        return F.softplus(self.log_coupling)[None, :, None, None]

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,Lq,Lk)
        tau = (torch.einsum("bhld,hd->bhl", q, self.w_tau) + self.b_tau[None, :, None]).unsqueeze(-1)
        beta = self.beta[None, :, None, None]
        base = a - tau
        g0 = torch.sigmoid(beta * base)

        if self.f2 == "modular":
            logit = base
        elif self.f2 == "card":
            N = g0.sum(dim=-1, keepdim=True)                       # expected cardinality
            logit = base - self.coupling() * N
        elif self.f2 == "rep_key":
            # r_i = <k_i, sum_j g0_j k_j>  (factored O(n^2 d), not O(n^3) Gram)
            m = torch.matmul(g0, k)
            r = torch.matmul(m, k.transpose(-1, -2)) * self.scale
            logit = base - self.coupling() * r
        elif self.f2 == "rep_val":
            m = torch.matmul(g0, v)
            r = torch.matmul(m, v.transpose(-1, -2)) * self.scale
            logit = base - self.coupling() * r
        elif self.f2 == "submod":
            am = a.amax(dim=-1, keepdim=True)
            ex = torch.exp(a - am)
            ssoft = (g0 * ex).sum(dim=-1, keepdim=True)
            excl = ssoft - g0 * ex
            delta = torch.log1p(ex / (excl + self.eps))
            logit = self.coupling() * delta - tau
        else:
            raise ValueError(self.f2)

        gate = torch.sigmoid(beta * logit)
        am = a.amax(dim=-1, keepdim=True)
        ex = torch.exp(a - am) * gate
        w = ex / ex.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        y = torch.matmul(w, v).transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(y)


class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.m = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        return self.m(x, x, x, need_weights=False)[0]


def sparsemax(z, dim=-1):
    """Sparsemax (Martins & Astudillo 2016): Euclidean projection onto the simplex.
    Produces sparse weights that sum to 1; low-score entries are truncated to exactly 0."""
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cssv = z_sorted.cumsum(dim)
    n = z.size(dim)
    rng = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = n
    rng = rng.view(shape)
    support = (1 + rng * z_sorted) > cssv
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cssv.gather(dim, k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


def entmax15(z, dim=-1, n_iter=30):
    """1.5-entmax (Peters et al. 2019) via bisection: SOTA sparse attention, less
    aggressive than sparsemax. Also magnitude-based (truncates low-score entries)."""
    x = z * 0.5  # (alpha-1) with alpha=1.5
    hi = x.max(dim=dim, keepdim=True).values
    lo = hi - 12.0
    tau = (lo + hi) / 2
    for _ in range(n_iter):
        tau = (lo + hi) / 2
        s = torch.clamp(x - tau, min=0).pow(2).sum(dim=dim, keepdim=True)
        too_small = s < 1.0            # tau too large -> search lower half
        hi = torch.where(too_small, tau, hi)
        lo = torch.where(too_small, lo, tau)
    p = torch.clamp(x - (lo + hi) / 2, min=0).pow(2)
    return p / p.sum(dim=dim, keepdim=True).clamp_min(1e-9)


class SparseAttention(nn.Module):
    """Baseline: standard attention with softmax replaced by sparsemax OR entmax-1.5
    (magnitude-based sparsity). Param-matched to MHA. Represents the sparse-attention family."""

    def __init__(self, dim, heads, kind="sparsemax"):
        super().__init__()
        self.dim, self.h = dim, heads
        self.hd = dim // heads
        self.scale = 1.0 / math.sqrt(self.hd)
        self.kind = kind
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        w = sparsemax(a, dim=-1) if self.kind == "sparsemax" else entmax15(a, dim=-1)
        y = torch.matmul(w, v).transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(y)


class Model(nn.Module):
    def __init__(self, d_in, dim, C, heads, f2):
        super().__init__()
        self.embed = nn.Linear(d_in, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.n1 = nn.LayerNorm(dim)
        if f2 == "mha":
            self.attn: nn.Module = MHA(dim, heads)
        elif f2 == "sparsemax":
            self.attn = SparseAttention(dim, heads, "sparsemax")
        elif f2 == "entmax15":
            self.attn = SparseAttention(dim, heads, "entmax15")
        else:
            self.attn = F2Attention(dim, heads, f2)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, C)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

    def forward(self, x):
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1), self.embed(x)], dim=1)
        h = h + self.attn(self.n1(h))
        h = h + self.mlp(self.n2(h))
        return self.head(self.norm(h)[:, 0])


# --------------------------- tasks ---------------------------
def make_protos(d, C, device):
    g = torch.Generator(device=device).manual_seed(DATA_SEED)
    return torch.randn(C, d, generator=g, device=device)


def batch_needle(n, C, d, L, protos, g, device, K=1, noise=1.2):
    y = torch.randint(0, C, (n,), generator=g, device=device)
    X = torch.randn(n, L, d, generator=g, device=device)  # distractors (redundant pool of 4)
    dp = torch.randint(0, 4, (n, L), generator=g, device=device)
    dprotos = torch.randn(4, d, generator=torch.Generator(device=device).manual_seed(DATA_SEED + 1), device=device)
    X = dprotos[dp] + noise * X
    order = torch.rand(n, L, generator=g, device=device).argsort(dim=1)[:, :K]
    sig = protos[y].unsqueeze(1) + noise * torch.randn(n, K, d, generator=g, device=device)
    X.scatter_(1, order.unsqueeze(-1).expand(-1, -1, d), sig)
    return X, y


def batch_redundancy(n, C, d, protos, g, device, max_sig=5, max_dec=30, n_bg=30, min_cnt=2,
                     noise_sig=0.3, noise_bg=1.0):
    y = torch.randint(0, C, (n,), generator=g, device=device)
    yp = (y + torch.randint(1, C, (n,), generator=g, device=device)) % C
    n_sig = torch.randint(min_cnt, max_sig + 1, (n, 1), generator=g, device=device)
    m_dec = torch.randint(min_cnt, max_dec + 1, (n, 1), generator=g, device=device)
    sig = protos[y].unsqueeze(1) + noise_sig * torch.randn(n, max_sig, d, generator=g, device=device)
    sig = torch.where((torch.arange(max_sig, device=device) < n_sig).unsqueeze(-1), sig,
                      noise_bg * torch.randn(n, max_sig, d, generator=g, device=device))
    base = protos[yp] + noise_sig * torch.randn(n, d, generator=g, device=device)
    dec = base.unsqueeze(1).expand(n, max_dec, d).clone()
    dec = torch.where((torch.arange(max_dec, device=device) < m_dec).unsqueeze(-1), dec,
                      noise_bg * torch.randn(n, max_dec, d, generator=g, device=device))
    bg = noise_bg * torch.randn(n, n_bg, d, generator=g, device=device)
    tok = torch.cat([sig, dec, bg], dim=1).contiguous()
    perm = torch.rand(n, tok.shape[1], generator=g, device=device).argsort(dim=1)
    return torch.gather(tok, 1, perm.unsqueeze(-1).expand(-1, -1, d)), y


def get_batch(task, n, C, d, L, protos, g, device):
    if task == "needle":
        return batch_needle(n, C, d, L, protos, g, device)
    return batch_redundancy(n, C, d, protos, g, device)


@torch.no_grad()
def evaluate(model, task, n, C, d, L, protos, device, batches=20):
    model.eval()
    g = torch.Generator(device=device).manual_seed(TEST_SEED)
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = tot = 0
    loss = 0.0
    for _ in range(batches):
        X, y = get_batch(task, 256, C, d, L, protos, g, device)
        lg = model(X)
        loss += ce(lg, y).item()
        correct += (lg.argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * correct / tot, loss / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--f2", choices=["mha", "modular", "card", "rep_key", "rep_val", "submod", "sparsemax", "entmax15"], default="modular")
    p.add_argument("--task", choices=["needle", "redundancy"], default="redundancy")
    p.add_argument("--L", type=int, default=512)  # needle length
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    protos = make_protos(args.d, args.num_classes, device)
    model = Model(args.d, args.dim, args.num_classes, args.heads, args.f2).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    model.train()
    for _ in range(args.steps):
        X, y = get_batch(args.task, args.batch_size, args.num_classes, args.d, args.L, protos, g, device)
        opt.zero_grad(set_to_none=True)
        crit(model(X), y).backward()
        opt.step()
    acc, loss = evaluate(model, args.task, 256, args.num_classes, args.d, args.L, protos, device)
    coup = None
    if getattr(model.attn, "log_coupling", None) is not None:
        coup = round(F.softplus(model.attn.log_coupling).mean().item(), 4)
    print(json.dumps({"task": args.task, "f2": args.f2, "L": args.L, "seed": args.seed,
                      "params": n_params, "test_acc": round(acc, 3), "test_loss": round(loss, 4),
                      "coupling": coup, "chance": round(100.0 / args.num_classes, 2)}))


if __name__ == "__main__":
    main()
