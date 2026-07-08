#!/usr/bin/env python3
"""Needle-in-distractors probe: the regime where softmax's density is a liability.

Each example is a set of L tokens. K "signal" tokens carry the class prototype
(+noise); the other L-K are distractors. With --redundant, distractors are drawn
from a SMALL pool of distractor prototypes (many near-duplicates) -- this is where
plain softmax over-counts the redundant cluster (mass ~ multiplicity) and a
repulsive/DPP gate can help, and where adaptive sparsity can suppress the tail.

A CLS query must find the K signal tokens among L-K distractors and classify.
As L grows, dense softmax dilutes; selection/diversity should start to win.

Fair, param-matched comparison of --attn {mha, gated, gated_rep}. Self-contained
(synthetic data, no external downloads). Prints one JSON result line.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from gated_attention import GatedSoftmaxAttention

DATA_SEED = 1234  # fixes the task (prototypes) across all arms/seeds
TEST_SEED = 999


def make_prototypes(d: int, num_classes: int, n_distractor_protos: int, device):
    g = torch.Generator(device=device).manual_seed(DATA_SEED)
    protos = torch.randn(num_classes, d, generator=g, device=device)
    dprotos = torch.randn(n_distractor_protos, d, generator=g, device=device)
    return protos, dprotos


def make_batch(n, L, K, d, num_classes, ndp, redundant, noise, protos, dprotos, g, device):
    ys = torch.randint(0, num_classes, (n,), generator=g, device=device)
    if redundant:
        idx = torch.randint(0, ndp, (n, L), generator=g, device=device)
        X = dprotos[idx] + noise * torch.randn(n, L, d, generator=g, device=device)
    else:
        X = torch.randn(n, L, d, generator=g, device=device)
    # place K signal tokens (same class prototype + indep noise) at random positions
    order = torch.rand(n, L, generator=g, device=device).argsort(dim=1)
    pos = order[:, :K]  # (n, K)
    sig = protos[ys].unsqueeze(1) + noise * torch.randn(n, K, d, generator=g, device=device)
    X.scatter_(1, pos.unsqueeze(-1).expand(-1, -1, d), sig)
    return X, ys


class MHA(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.m = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        y, _ = self.m(x, x, x, need_weights=False)
        return y


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, attn: str, repulsion: bool, beta_init: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        if attn == "mha":
            self.attn: nn.Module = MHA(dim, heads)
        else:
            self.attn = GatedSoftmaxAttention(
                dim, heads, beta_init=beta_init, repulsion=repulsion
            )
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class NeedleModel(nn.Module):
    def __init__(self, d_in, dim, num_classes, heads, attn, repulsion, beta_init, depth=1):
        super().__init__()
        self.embed = nn.Linear(d_in, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [Block(dim, heads, attn, repulsion, beta_init) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

    def forward(self, x):
        h = self.embed(x)
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h)[:, 0])


@torch.no_grad()
def evaluate(model, args, protos, dprotos, device):
    model.eval()
    g = torch.Generator(device=device).manual_seed(TEST_SEED)
    correct = tot = 0
    for _ in range(args.test_batches):
        X, y = make_batch(args.batch_size, args.L, args.K, args.d, args.num_classes,
                          args.ndp, args.redundant, args.noise, protos, dprotos, g, device)
        correct += (model(X).argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * correct / tot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["mha", "gated", "gated_rep"], default="gated")
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--ndp", type=int, default=4, help="# distractor prototypes (redundancy)")
    p.add_argument("--redundant", action="store_true")
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--beta-init", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batches", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    protos, dprotos = make_prototypes(args.d, args.num_classes, args.ndp, device)

    model = NeedleModel(
        args.d, args.dim, args.num_classes, args.heads,
        attn="mha" if args.attn == "mha" else "gated",
        repulsion=(args.attn == "gated_rep"), beta_init=args.beta_init, depth=args.depth,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=device).manual_seed(10_000 + args.seed)

    model.train()
    for step in range(args.steps):
        X, y = make_batch(args.batch_size, args.L, args.K, args.d, args.num_classes,
                          args.ndp, args.redundant, args.noise, protos, dprotos, g, device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(X), y)
        loss.backward()
        opt.step()

    acc = evaluate(model, args, protos, dprotos, device)
    print(json.dumps({
        "attn": args.attn, "L": args.L, "K": args.K, "ndp": args.ndp,
        "redundant": args.redundant, "seed": args.seed, "params": n_params,
        "test_acc": round(acc, 3), "final_train_loss": round(loss.item(), 4),
    }))


if __name__ == "__main__":
    main()
