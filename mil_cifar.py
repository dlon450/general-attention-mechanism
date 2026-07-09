#!/usr/bin/env python3
"""CIFAR-bags Multiple-Instance-Learning: tests our anti-redundancy attention on REAL
image features (not Gaussian synthetics). A "bag" is a set of real CIFAR images; a CLS
query attention-pools over per-image embeddings to predict a bag label (ABMIL-style).

Two bag constructions:
  needle     : 1 signal image of class y among N-1 diverse distractor images -> label y.
               (long-context dilution; standard MIL flavor)
  redundancy : n_sig DISTINCT images of class y + one class-y' image copied m times
               (adversarial redundant clique) + diverse background -> label y
               (the class with more DISTINCT instances). Real-image analog of our
               synthetic redundancy task.

Attention pooling variants (param-matched): mha (=ABMIL softmax), sparsemax, entmax15,
rep_key, rep_val. Reports test accuracy + eval cross-entropy. Self-contained (uses local
CIFAR). Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from f2_sweep import F2Attention, MHA, SparseAttention

CIFAR = "/data/users/dereklong/scratch/general-attention-mechanism/data/cifar-10-batches-py"
NUM_CLASSES = 10


def load_by_class(split):
    files = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
    xs, ys = [], []
    for name in files:
        with open(os.path.join(CIFAR, name), "rb") as f:
            e = pickle.load(f, encoding="latin1")
        xs.append(torch.frombuffer(bytes(e["data"]), dtype=torch.uint8).reshape(-1, 3, 32, 32))
        ys.extend(e["labels"])
    X = torch.cat(xs, 0).float() / 255.0
    y = torch.tensor(ys)
    return [X[y == c] for c in range(NUM_CLASSES)]  # list of per-class image tensors


def pick(cls_imgs, c, n, g):
    idx = torch.randint(0, cls_imgs[c].shape[0], (n,), generator=g)
    return cls_imgs[c][idx]


def make_bag_batch(cls, B, task, N, g, n_sig=3, m_dec=None, device="cpu"):
    bags, labels = [], []
    for _ in range(B):
        y = int(torch.randint(0, NUM_CLASSES, (1,), generator=g))
        if task == "needle":
            others = [c for c in range(NUM_CLASSES) if c != y]
            distractor_cls = [others[int(torch.randint(0, 9, (1,), generator=g))] for _ in range(N - 1)]
            imgs = [pick(cls, y, 1, g)] + [pick(cls, c, 1, g) for c in distractor_cls]
            bag = torch.cat(imgs, 0)
        else:  # redundancy
            m = m_dec if m_dec is not None else N - n_sig - 5
            yp = (y + 1 + int(torch.randint(0, NUM_CLASSES - 1, (1,), generator=g))) % NUM_CLASSES
            sig = pick(cls, y, n_sig, g)                       # n_sig DISTINCT class-y images
            dec1 = pick(cls, yp, 1, g)                         # ONE class-y' image ...
            dec = dec1.repeat(m, 1, 1, 1)                      # ... copied m times (clique)
            n_bg = max(0, N - n_sig - m)
            bg_cls = [c for c in range(NUM_CLASSES) if c not in (y, yp)]
            bg = torch.cat([pick(cls, bg_cls[int(torch.randint(0, 8, (1,), generator=g))], 1, g)
                            for _ in range(n_bg)], 0) if n_bg > 0 else sig[:0]
            bag = torch.cat([sig, dec, bg], 0)
        perm = torch.randperm(bag.shape[0], generator=g)
        bags.append(bag[perm]); labels.append(y)
    return torch.stack(bags).to(device), torch.tensor(labels, device=device)


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),   # 16
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),  # 8
            nn.Conv2d(64, dim, 3, 2, 1), nn.GELU(), # 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):  # (M,3,32,32) -> (M,dim)
        return self.net(x)


class MILModel(nn.Module):
    def __init__(self, dim, heads, attn, num_classes):
        super().__init__()
        self.enc = Encoder(dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.n1 = nn.LayerNorm(dim)
        if attn == "mha":
            self.attn: nn.Module = MHA(dim, heads)
        elif attn in ("sparsemax", "entmax15"):
            self.attn = SparseAttention(dim, heads, attn)
        else:
            self.attn = F2Attention(dim, heads, attn)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

    def forward(self, bags):  # (B,N,3,32,32)
        B, N = bags.shape[:2]
        h = self.enc(bags.reshape(B * N, 3, 32, 32)).reshape(B, N, -1)
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
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-batches", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_cls = load_by_class("train"); test_cls = load_by_class("test")
    model = MILModel(args.dim, args.heads, args.attn, NUM_CLASSES).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(10_000 + args.seed)
    model.train()
    for _ in range(args.steps):
        bags, y = make_bag_batch(train_cls, args.batch_size, args.task, args.N, g,
                                 n_sig=args.n_sig, m_dec=args.m_dec, device=device)
        opt.zero_grad(set_to_none=True)
        crit(model(bags), y).backward()
        opt.step()
    acc, loss = evaluate(model, test_cls, args, device)
    print(json.dumps({"task": args.task, "attn": args.attn, "N": args.N, "seed": args.seed,
                      "params": n_params, "test_acc": round(acc, 3), "test_loss": round(loss, 4),
                      "chance": round(100.0 / NUM_CLASSES, 2)}))


if __name__ == "__main__":
    main()
