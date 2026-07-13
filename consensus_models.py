#!/usr/bin/env python3
"""Decisive core for "beat regular attention" on the consensus benchmark (task_consensus.py).

All arms share a k=1 self-attention encoder over the L items, then a swappable pooling head -> C
logits (predict latent truth y). Chance = 50% (two clusters present). Arms:

  CONTENT-ONLY (P1 null: must be ~chance at alpha=1, since they see only V):
    softmax          : encoder + softmax attention pool
    set_transformer  : ISAB(m_ind) + PMA  (strong content-only set model)
  PROVENANCE (P2: fed the SAME noisy graph Pgraph):
    prov_concat      : per-item provenance degree feature concat to input + softmax pool (cheap-feature baseline)
    relation_bias    : encoder attention logits += mu*Pgraph + softmax pool (strong provenance baseline)
    m2_prov (OURS)   : pooling gate that down-weights items with high WITHIN-CONTENT-NEIGHBOR same-origin
                       density (soft, differentiable version of the robust oracle)

Trains each arm; reports test accuracy on a FROZEN test set. Primary cell alpha=1, gamma=0.8."""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_consensus import Cfg, prototypes, gen_batch

DIM, HEADS = 64, 4


def mha(q, k, v, mask, bias=None):
    # q,k,v: (B,L,H,dh); mask: (B,L) real=1; bias: (B,L,L) added to logits (relation bias)
    B, L, H, dh = q.shape
    logits = torch.einsum("blhd,bmhd->bhlm", q, k) / (dh ** 0.5)   # (B,H,L,L)
    if bias is not None:
        logits = logits + bias.unsqueeze(1)
    keymask = (mask < 0.5).unsqueeze(1).unsqueeze(1)               # (B,1,1,L)
    logits = logits.masked_fill(keymask, float("-inf"))
    w = torch.softmax(logits, dim=-1)
    return torch.einsum("bhlm,bmhd->blhd", w, v)


class Encoder(nn.Module):
    """One self-attention block over items. Optional additive Pgraph bias (relation_bias arm)."""
    def __init__(self, d_in, use_relation_bias=False):
        super().__init__()
        self.embed = nn.Linear(d_in, DIM)
        self.q = nn.Linear(DIM, DIM); self.k = nn.Linear(DIM, DIM); self.v = nn.Linear(DIM, DIM)
        self.o = nn.Linear(DIM, DIM); self.n1 = nn.LayerNorm(DIM); self.n2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))
        self.use_rb = use_relation_bias
        if use_relation_bias:
            self.mu = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask, Pgraph=None):
        h = self.embed(x)
        B, L, _ = h.shape
        hn = self.n1(h)
        sh = lambda t: t.view(B, L, HEADS, DIM // HEADS)
        bias = self.mu * Pgraph if (self.use_rb and Pgraph is not None) else None
        att = mha(sh(self.q(hn)), sh(self.k(hn)), sh(self.v(hn)), mask, bias).reshape(B, L, DIM)
        h = h + self.o(att)
        h = h + self.mlp(self.n2(h))
        return h


class SoftmaxPool(nn.Module):
    def __init__(self):
        super().__init__(); self.q = nn.Parameter(torch.randn(DIM) * 0.02)

    def forward(self, h, mask, Pgraph=None):
        a = (h @ self.q) / (DIM ** 0.5)
        a = a.masked_fill(mask < 0.5, float("-inf"))
        w = torch.softmax(a, dim=1)
        return (w.unsqueeze(-1) * h).sum(1)


class M2ProvPool(nn.Module):
    """OURS: gate down-weights items whose content-neighbors are same-origin (high Pgraph density in
    the content neighborhood) -> soft/differentiable analog of the robust oracle."""
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.randn(DIM) * 0.02)
        self.beta = nn.Parameter(torch.tensor(2.0)); self.lam = nn.Parameter(torch.tensor(2.0))
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, h, mask, Pgraph):
        a = (h @ self.q) / (DIM ** 0.5)                                # relevance (B,L)
        sim = torch.einsum("bld,bmd->blm", h, h) / (DIM ** 0.5)        # content similarity
        sim = sim.masked_fill((mask < 0.5).unsqueeze(1), float("-inf"))
        Cw = torch.softmax(sim, dim=-1)                               # content-neighborhood weights (B,L,L)
        dens = (Cw * Pgraph).sum(-1)                                   # within-neighbor same-origin density (B,L)
        g = torch.sigmoid(self.beta * (self.tau - self.lam * dens))    # down-weight high density
        ex = torch.exp(a - a.amax(1, keepdim=True)) * g
        ex = ex.masked_fill(mask < 0.5, 0.0)
        w = ex / ex.sum(1, keepdim=True).clamp_min(1e-9)
        return (w.unsqueeze(-1) * h).sum(1)


class M2ProvXPool(nn.Module):
    """Expressive version of the source-aware bias: learn g_i = MLP([within-neighbor same-origin
    density, global degree, relevance]). Keeps the structural prior (a per-item gate driven by the
    oracle-aligned density feature) but is flexible enough to recover the Bayes-optimal use of
    provenance -> aims to keep the low-data head start AND match the flexible ceiling at high data."""
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.randn(DIM) * 0.02)
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, h, mask, Pgraph):
        a = (h @ self.q) / (DIM ** 0.5)
        sim = torch.einsum("bld,bmd->blm", h, h) / (DIM ** 0.5)
        sim = sim.masked_fill((mask < 0.5).unsqueeze(1), float("-inf"))
        Cw = torch.softmax(sim, dim=-1)
        dens = (Cw * Pgraph).sum(-1)                                    # local same-origin density
        deg = Pgraph.sum(-1) / mask.sum(1, keepdim=True).clamp_min(1)   # global degree
        feat = torch.stack([dens, deg, a], dim=-1)                      # (B,L,3)
        g = torch.sigmoid(self.gate(feat).squeeze(-1))
        ex = torch.exp(a - a.amax(1, keepdim=True)) * g
        ex = ex.masked_fill(mask < 0.5, 0.0)
        w = ex / ex.sum(1, keepdim=True).clamp_min(1e-9)
        return (w.unsqueeze(-1) * h).sum(1)


class M2ProvRPool(nn.Module):
    """Rigid prior + zero-initialized residual MLP: g_i = σ(β(τ−λ·density_i) + MLP([density,deg,a])).
    At init MLP≈0 so it IS the rigid gate (max sample-efficiency); with data the residual adds
    flexibility (recovers the high-data ceiling). Aims to dominate the whole learning curve."""
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.randn(DIM) * 0.02)
        self.beta = nn.Parameter(torch.tensor(2.0)); self.lam = nn.Parameter(torch.tensor(2.0))
        self.tau = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)   # start = rigid

    def forward(self, h, mask, Pgraph):
        a = (h @ self.q) / (DIM ** 0.5)
        sim = torch.einsum("bld,bmd->blm", h, h) / (DIM ** 0.5)
        sim = sim.masked_fill((mask < 0.5).unsqueeze(1), float("-inf"))
        Cw = torch.softmax(sim, dim=-1)
        dens = (Cw * Pgraph).sum(-1)
        deg = Pgraph.sum(-1) / mask.sum(1, keepdim=True).clamp_min(1)
        resid = self.gate(torch.stack([dens, deg, a], dim=-1)).squeeze(-1)
        g = torch.sigmoid(self.beta * (self.tau - self.lam * dens) + resid)
        ex = torch.exp(a - a.amax(1, keepdim=True)) * g
        ex = ex.masked_fill(mask < 0.5, 0.0)
        w = ex / ex.sum(1, keepdim=True).clamp_min(1e-9)
        return (w.unsqueeze(-1) * h).sum(1)


class SetTransformer(nn.Module):
    """ISAB(m_ind) + PMA(1 seed). Content-only strong baseline."""
    def __init__(self, d_in, m_ind=8):
        super().__init__()
        self.embed = nn.Linear(d_in, DIM)
        self.I = nn.Parameter(torch.randn(m_ind, DIM) * 0.02)
        self.seed = nn.Parameter(torch.randn(1, DIM) * 0.02)
        self.q1 = nn.Linear(DIM, DIM); self.k1 = nn.Linear(DIM, DIM); self.v1 = nn.Linear(DIM, DIM)
        self.q2 = nn.Linear(DIM, DIM); self.k2 = nn.Linear(DIM, DIM); self.v2 = nn.Linear(DIM, DIM)
        self.qp = nn.Linear(DIM, DIM); self.kp = nn.Linear(DIM, DIM); self.vp = nn.Linear(DIM, DIM)

    def _mab(self, Q, K, ql, kl, vl, mask=None):
        B = Q.shape[0]
        sh = lambda t, L: t.view(B, L, HEADS, DIM // HEADS)
        if mask is None:
            mask = torch.ones(B, K.shape[1], device=K.device)
        att = mha(sh(ql(Q), Q.shape[1]), sh(kl(K), K.shape[1]), sh(vl(K), K.shape[1]), mask)
        return Q + att.reshape(B, Q.shape[1], DIM)

    def forward(self, x, mask, Pgraph=None):
        B, L, _ = x.shape
        h = self.embed(x)
        Iexp = self.I.unsqueeze(0).expand(B, -1, -1)
        Hh = self._mab(Iexp, h, self.q1, self.k1, self.v1, mask)       # induced (B,m,DIM)
        Z = self._mab(h, Hh, self.q2, self.k2, self.v2)               # (B,L,DIM)
        S = self.seed.unsqueeze(0).expand(B, -1, -1)
        pooled = self._mab(S, Z, self.qp, self.kp, self.vp, mask)     # (B,1,DIM)
        return pooled.squeeze(1)


class Model(nn.Module):
    def __init__(self, arm, C, d_in=32):
        super().__init__()
        self.arm = arm
        extra = 1 if arm == "prov_concat" else 0
        if arm == "set_transformer":
            self.net = SetTransformer(d_in)
        else:
            self.enc = Encoder(d_in + extra, use_relation_bias=(arm == "relation_bias"))
            self.pool = {"m2_prov": M2ProvPool, "m2_prov_x": M2ProvXPool,
                         "m2_prov_r": M2ProvRPool}.get(arm, SoftmaxPool)()
        self.head = nn.Linear(DIM, C)

    def forward(self, V, mask, Pgraph):
        if self.arm == "set_transformer":
            b = self.net(V, mask)
        else:
            x = V
            if self.arm == "prov_concat":
                deg = (Pgraph.sum(-1) / mask.sum(1, keepdim=True).clamp_min(1))  # (B,L) norm degree
                x = torch.cat([V, deg.unsqueeze(-1)], dim=-1)
            h = self.enc(x, mask, Pgraph if self.arm == "relation_bias" else None)
            b = self.pool(h, mask, Pgraph)
        return self.head(b)


def make_split(n, cfg, mu, alpha, gamma, seed, device):
    B = gen_batch(n, cfg, np.random.default_rng(seed), alpha, gamma, mu)
    t = lambda k, dt: torch.tensor(B[k], dtype=dt, device=device)
    return t("V", torch.float32), t("mask", torch.float32), t("Pgraph", torch.float32), t("y", torch.long)


def run_arm(arm, tr, va, te, C, device, steps=800, bs=128, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    model = Model(arm, C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Vtr, Mtr, Ptr, Ytr = tr; n = len(Vtr)
    g = torch.Generator().manual_seed(seed)
    best_va, best_te = 0.0, 0.0
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        model.train(); opt.zero_grad(set_to_none=True)
        crit(model(Vtr[idx], Mtr[idx], Ptr[idx]), Ytr[idx]).backward(); opt.step()
        if (s + 1) % 100 == 0:
            va_acc = evaluate(model, va);
            if va_acc >= best_va:
                best_va = va_acc; best_te = evaluate(model, te)
    n_par = sum(p.numel() for p in model.parameters())
    return best_te, n_par


@torch.no_grad()
def evaluate(model, split, bs=512):
    model.eval(); V, M, P, Y = split; cor = 0
    for i in range(0, len(V), bs):
        cor += (model(V[i:i+bs], M[i:i+bs], P[i:i+bs]).argmax(1) == Y[i:i+bs]).sum().item()
    return 100.0 * cor / len(V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--arms", nargs="+",
                    default=["softmax", "set_transformer", "prov_concat", "relation_bias", "m2_prov"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Cfg(); mu = prototypes(cfg, np.random.default_rng(0))
    te = make_split(2500, cfg, mu, args.alpha, args.gamma, 777, device)   # FROZEN test (fixed seed)
    print(f"consensus: alpha={args.alpha} gamma={args.gamma} chance=50%  {device}")
    print(f"{'arm':16}{'test acc (mean±std)':>22}{'params':>9}   [oracle ~98 @a1g0.8]")
    for arm in args.arms:
        accs = []; npar = 0
        for s in range(args.seeds):
            tr = make_split(args.n_train, cfg, mu, args.alpha, args.gamma, 100 + s, device)
            va = make_split(1500, cfg, mu, args.alpha, args.gamma, 500 + s, device)
            acc, npar = run_arm(arm, tr, va, te, cfg.C, device, steps=args.steps, seed=s)
            accs.append(acc)
        import statistics as st
        sd = st.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{arm:16}{st.mean(accs):>15.1f} ±{sd:>4.1f}{npar:>9,}", flush=True)


if __name__ == "__main__":
    main()
