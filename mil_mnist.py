#!/usr/bin/env python3
"""MNIST-bags Multiple-Instance-Learning (canonical ABMIL benchmark, Ilse et al. 2018)
on REAL MNIST images. A "bag" is a set of digit images; a CLS query attention-pools over
per-image embeddings to predict a bag label. Tests our anti-redundancy attention on real
image features against ABMIL-softmax + sparse-attention baselines, at equal params.

Bag constructions:
  needle     : 1 signal image of class y among N-1 diverse distractor digits -> label y
               (dilution; standard MIL flavor).
  redundancy : n_sig DISTINCT class-y images + one class-y' image copied m times
               (adversarial redundant clique) + diverse background -> label y (the class
               with more DISTINCT instances). Real-image analog of our synthetic win case.

Attention pooling: mha (=ABMIL softmax), sparsemax, entmax15, modular, rep_key, rep_val.
Prints one JSON line (test acc + eval cross-entropy).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from f2_sweep import F2Attention, MHA, SparseAttention

RAW = "/data/users/dereklong/scratch/general-attention-mechanism/data/mnist_dl_train/mnist_train/MNIST/raw"
NUM_CLASSES = 10


def _read_images(path):
    with open(path, "rb") as f:
        _, num, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, r * c)


def _read_labels(path):
    with open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_by_class(split):
    stem = "train" if split == "train" else "t10k"
    X = torch.from_numpy(_read_images(f"{RAW}/{stem}-images-idx3-ubyte").copy()).float() / 255.0
    y = torch.from_numpy(_read_labels(f"{RAW}/{stem}-labels-idx1-ubyte").copy()).long()
    per = [torch.where(y == c)[0] for c in range(NUM_CLASSES)]        # per-class indices
    maxlen = max(int(t.numel()) for t in per)
    P = torch.stack([t[torch.arange(maxlen) % t.numel()] for t in per])  # (C, maxlen), tiled
    return X, P


def _samp(P, cls, X, g):
    """Sample a random image of each requested class. cls: (...,) class ids -> (..., 784)."""
    r = torch.randint(0, P.shape[1], cls.shape, generator=g)
    return X[P[cls, r]]


def make_bag_batch(data, B, task, N, g, n_sig=3, m_dec=None, device="cpu"):
    """Fully-vectorized bag construction (no per-example Python loop)."""
    X, P = data
    C = NUM_CLASSES
    y = torch.randint(0, C, (B,), generator=g)
    if task == "needle":
        sig = _samp(P, y.unsqueeze(1), X, g)                                  # (B,1,784)
        dc = (y.unsqueeze(1) + 1 + torch.randint(0, C - 1, (B, N - 1), generator=g)) % C  # != y
        bag = torch.cat([sig, _samp(P, dc, X, g)], dim=1)                     # (B,N,784)
    else:  # redundancy
        m = m_dec if m_dec is not None else N - n_sig - 5
        yp = (y + 1 + torch.randint(0, C - 1, (B,), generator=g)) % C         # decoy class != y
        sig = _samp(P, y.unsqueeze(1).expand(B, n_sig).contiguous(), X, g)    # n_sig DISTINCT class-y
        dec = _samp(P, yp.unsqueeze(1), X, g).expand(B, m, 784)               # ONE class-y' img x m
        parts = [sig, dec]
        n_bg = N - n_sig - m
        if n_bg > 0:
            a = torch.minimum(y, yp).unsqueeze(1); b = torch.maximum(y, yp).unsqueeze(1)
            bc = torch.randint(0, C - 2, (B, n_bg), generator=g)
            bc = bc + (bc >= a).long(); bc = bc + (bc >= b).long()           # classes != y, yp
            parts.append(_samp(P, bc, X, g))
        bag = torch.cat(parts, dim=1)
    L = bag.shape[1]
    perm = torch.rand(B, L, generator=g).argsort(dim=1)                       # per-row shuffle
    bag = torch.gather(bag, 1, perm.unsqueeze(-1).expand(B, L, 784))
    return bag.to(device), y.to(device)


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.GELU(), nn.Linear(256, dim), nn.GELU())

    def forward(self, x):
        return self.net(x)


class MILModel(nn.Module):
    def __init__(self, dim, heads, attn, beta_init=0.5, lambda_init=None):
        super().__init__()
        self.enc = Encoder(dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.n1 = nn.LayerNorm(dim)
        if attn == "mha":
            self.attn: nn.Module = MHA(dim, heads)
        elif attn in ("sparsemax", "entmax15"):
            self.attn = SparseAttention(dim, heads, attn)
        else:
            self.attn = F2Attention(dim, heads, attn, beta_init=beta_init, coupling_init=lambda_init)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NUM_CLASSES)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

    def forward(self, bags):  # (B,N,784)
        B, N = bags.shape[:2]
        h = self.enc(bags.reshape(B * N, 784)).reshape(B, N, -1)
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        h = h + self.attn(self.n1(h))
        h = h + self.mlp(self.n2(h))
        return self.head(self.norm(h)[:, 0])


@torch.no_grad()
def evaluate(model, cls, args, device):
    model.eval()
    g = torch.Generator().manual_seed(999)
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = tot = 0; loss = 0.0
    for _ in range(args.test_batches):
        bags, y = make_bag_batch(cls, args.batch_size, args.task, args.N, g,
                                 n_sig=args.n_sig, m_dec=args.m_dec, device=device)
        lg = model(bags); loss += ce(lg, y).item()
        correct += (lg.argmax(1) == y).sum().item(); tot += y.numel()
    return 100.0 * correct / tot, loss / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["mha", "sparsemax", "entmax15", "modular", "rep_key", "rep_val"], default="rep_key")
    p.add_argument("--task", choices=["needle", "redundancy"], default="redundancy")
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--n-sig", type=int, default=3)
    p.add_argument("--m-dec", type=int, default=None)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--beta-init", type=float, default=0.5)
    p.add_argument("--lambda-init", type=float, default=None, help="repulsion init; None=default(1.0)")
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
    model = MILModel(args.dim, args.heads, args.attn,
                     beta_init=args.beta_init, lambda_init=args.lambda_init).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(10_000 + args.seed)
    # Repulsion warmup: keep rep off for the first 40% of steps (encoder learns), then
    # ramp it in over the next 20%. Avoids the active-lambda-at-init encoder corruption.
    is_rep = args.attn in ("rep_key", "rep_val")
    warm_start = int(0.4 * args.steps)
    warm_len = max(1, int(0.2 * args.steps))
    model.train()
    for step in range(args.steps):
        if is_rep:
            model.attn.rep_scale = min(1.0, max(0.0, (step - warm_start) / warm_len))
        bags, y = make_bag_batch(tr, args.batch_size, args.task, args.N, g,
                                 n_sig=args.n_sig, m_dec=args.m_dec, device=device)
        opt.zero_grad(set_to_none=True)
        crit(model(bags), y).backward()
        opt.step()
    if is_rep:
        model.attn.rep_scale = 1.0
    acc, loss = evaluate(model, te, args, device)
    print(json.dumps({"task": args.task, "attn": args.attn, "N": args.N, "seed": args.seed,
                      "params": n_params, "test_acc": round(acc, 3), "test_loss": round(loss, 4),
                      "chance": round(100.0 / NUM_CLASSES, 2)}))


if __name__ == "__main__":
    main()
