#!/usr/bin/env python3
"""Redundancy-pathology probe: isolates higher-order F2 (DPP) from per-token gating.

Task "more DISTINCT wins, counts randomized":
  - TRUE class y: n_sig DISTINCT tokens (proto[y] + independent noise), n_sig ~ U[2, max_sig].
  - DECOY class y': ONE token copied m_dec times (identical -> redundant clique),
    m_dec ~ U[2, max_dec]. So DISTINCT(y)=n_sig>=2  vs  DISTINCT(y')=1  => label = y always.
  - n_bg pure-noise background tokens; total length fixed (inactive slots -> noise).
  - Counts are RANDOMIZED and overlapping, so TOTAL count does NOT identify y (sometimes
    n_sig>m_dec, usually m_dec>n_sig). Only DISTINCT count does -> requires de-duplication.

Why each rung should behave:
  - mha (dense softmax) / gated (per-token modular gate): weight tokens independently, so
    the pooled rep ~ n_sig*proto[y] + m_dec*proto[y']; with counts randomized the readout
    cannot tell which class is y -> picks the larger-count class -> usually y' -> low acc.
  - gated_rep (+ DPP repulsion, non-modular F2): collapses the identical clique to ~1
    effective token -> pooled ~ n_sig*proto[y] + 1*proto[y'] -> argmax = y -> high acc.

rung2(gated_rep) >> rung1(gated) ~ rung0(mha) is the ONLY result attributable to general
attention beyond gating. Self-contained; prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from gated_attention import GatedSoftmaxAttention

DATA_SEED = 1234
TEST_SEED = 999


def make_prototypes(d, C, device):
    g = torch.Generator(device=device).manual_seed(DATA_SEED)
    return torch.randn(C, d, generator=g, device=device)


def make_batch(n, C, d, max_sig, max_dec, n_bg, min_cnt, noise_sig, noise_bg, protos, g, device):
    L = max_sig + max_dec + n_bg
    y = torch.randint(0, C, (n,), generator=g, device=device)
    yp = (y + torch.randint(1, C, (n,), generator=g, device=device)) % C
    n_sig = torch.randint(min_cnt, max_sig + 1, (n, 1), generator=g, device=device)
    m_dec = torch.randint(min_cnt, max_dec + 1, (n, 1), generator=g, device=device)

    # signal block: max_sig DISTINCT class-y tokens; inactive slots -> noise
    sig = protos[y].unsqueeze(1) + noise_sig * torch.randn(n, max_sig, d, generator=g, device=device)
    sig_active = torch.arange(max_sig, device=device).unsqueeze(0) < n_sig
    sig = torch.where(sig_active.unsqueeze(-1), sig,
                      noise_bg * torch.randn(n, max_sig, d, generator=g, device=device))

    # decoy block: ONE class-y' exemplar copied max_dec times; inactive slots -> noise
    dec_base = protos[yp] + noise_sig * torch.randn(n, d, generator=g, device=device)
    dec = dec_base.unsqueeze(1).expand(n, max_dec, d).clone()
    dec_active = torch.arange(max_dec, device=device).unsqueeze(0) < m_dec
    dec = torch.where(dec_active.unsqueeze(-1), dec,
                      noise_bg * torch.randn(n, max_dec, d, generator=g, device=device))

    bg = noise_bg * torch.randn(n, n_bg, d, generator=g, device=device)
    tokens = torch.cat([sig, dec, bg], dim=1).contiguous()  # (n, L, d)
    perm = torch.rand(n, L, generator=g, device=device).argsort(dim=1)
    tokens = torch.gather(tokens, 1, perm.unsqueeze(-1).expand(-1, -1, d))
    return tokens, y


class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.m = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        y, _ = self.m(x, x, x, need_weights=False)
        return y


class Model(nn.Module):
    def __init__(self, d_in, dim, C, heads, attn, beta_init, lambda_init):
        super().__init__()
        self.embed = nn.Linear(d_in, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.n1 = nn.LayerNorm(dim)
        if attn == "mha":
            self.attn: nn.Module = MHA(dim, heads)
        else:
            self.attn = GatedSoftmaxAttention(
                dim, heads, beta_init=beta_init,
                repulsion=(attn == "gated_rep"), lambda_init=lambda_init,
            )
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, C)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

    def forward(self, x):
        h = self.embed(x)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        h = h + self.attn(self.n1(h))
        h = h + self.mlp(self.n2(h))
        return self.head(self.norm(h)[:, 0])


@torch.no_grad()
def evaluate(model, args, protos, device):
    model.eval()
    g = torch.Generator(device=device).manual_seed(TEST_SEED)
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = tot = 0
    loss_sum = 0.0
    for _ in range(args.test_batches):
        X, y = make_batch(args.batch_size, args.num_classes, args.d, args.max_sig, args.max_dec,
                          args.n_bg, args.min_count, args.noise_sig, args.noise_bg, protos, g, device)
        logits = model(X)
        loss_sum += ce(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * correct / tot, loss_sum / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["mha", "gated", "gated_rep"], default="gated_rep")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--max-sig", type=int, default=5)
    p.add_argument("--max-dec", type=int, default=30)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--n-bg", type=int, default=30)
    p.add_argument("--noise-sig", type=float, default=0.3)
    p.add_argument("--noise-bg", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--beta-init", type=float, default=0.5)
    p.add_argument("--lambda-init", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batches", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    protos = make_prototypes(args.d, args.num_classes, device)
    model = Model(args.d, args.dim, args.num_classes, args.heads, args.attn,
                  args.beta_init, args.lambda_init).to(device)
    n_params = sum(q.numel() for q in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    model.train()
    for _ in range(args.steps):
        X, y = make_batch(args.batch_size, args.num_classes, args.d, args.max_sig, args.max_dec,
                          args.n_bg, args.min_count, args.noise_sig, args.noise_bg, protos, g, device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(X), y)
        loss.backward()
        opt.step()

    acc, eval_loss = evaluate(model, args, protos, device)
    lam = round(F.softplus(model.attn.log_lambda).mean().item(), 4) if args.attn == "gated_rep" else None
    print(json.dumps({
        "attn": args.attn, "max_sig": args.max_sig, "max_dec": args.max_dec, "seed": args.seed,
        "params": n_params, "test_acc": round(acc, 3), "test_loss": round(eval_loss, 4),
        "final_train_loss": round(loss.item(), 4), "lambda_mean": lam,
        "chance": round(100.0 / args.num_classes, 2),
    }))


if __name__ == "__main__":
    main()
