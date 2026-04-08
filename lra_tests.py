#!/usr/bin/env python3
"""
LRA Ablation: All F1 × F2 combinations vs standard softmax attention.
Tasks: ListOps · Pathfinder · LRA Text · LRA Image (CIFAR-10)

Run from the repo root:
    python lra_tests.py

Common options:
    --tasks listops text image pathfinder   # subset of tasks
    --epochs 5                              # epochs per model (default 3)
    --skip-f2 neural_mlp                   # skip slow F2 types
    --skip-f1 transformer                  # skip slow F1 types
    --quick                                # 2 epochs, 50-batch limit (smoke test)
    --no-download-pathfinder               # skip the large Pathfinder download
    --data-root ./lra_data                 # where to store downloaded data
    --results-dir ./lra_results            # where to write result files
"""

import argparse
import csv
import json
import math
import random
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# general_attention.py lives in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
from general_attention import GeneralAttention, GibbsConfig

# ── F1 / F2 grids ────────────────────────────────────────────────────────────

F2_TYPES = [
    "full_set",
    "modular_dot",
    "modular_dot_hard_singleton",
    "modular_dot_first_free",
    "logsumexp",
    "dot_repulsion",
    "neural_mlp",
]
F1_TYPES = [
    "mean",
    "mlp_mean",
    "mlp_concat",
    "transformer",
    "restricted_softmax",
]
SOFTMAX_KEY = ("full_set", "restricted_softmax")

GIBBS_CFG = GibbsConfig(beta=1.0, gibbs_steps=32, runs=4)

# ── Device ────────────────────────────────────────────────────────────────────

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ── Model architecture ────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, attn: nn.Module, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = attn
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SequenceClassifier(nn.Module):
    """Embedding -> N transformer blocks -> mean-pool -> linear head."""

    def __init__(self, vocab_size, num_classes, max_seq_len, dim, depth,
                 attn_factory, ff_mult=4, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(dim, attn_factory(dim), ff_mult, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h   = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        h   = self.norm(self.blocks(h))
        return self.head(h.mean(dim=1))


def build_attn_module(dim: int, f1_type: str, f2_type: str) -> nn.Module:
    return GeneralAttention(
        d_model=dim,
        f2_type=f2_type,
        f1_type=f1_type,
        cfg=GIBBS_CFG,
        f1_concat_max_set_size=8,
        f1_concat_hidden=64,
        f2_neural_hidden=64,
    )

# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(-1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y   = x.to(device), y.to(device)
        logits  = model(x)
        loss    = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(-1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def run_ablation(task_name, train_loader, val_loader, make_model_fn,
                 epochs=3, lr=3e-4, device=None,
                 max_train_batches=None, max_val_batches=None,
                 skip_f2=None, skip_f1=None):
    device   = device or choose_device()
    skip_f2  = set(skip_f2 or [])
    skip_f1  = set(skip_f1 or [])
    criterion = nn.CrossEntropyLoss()
    results  = {}

    combos = [(f2, f1) for f2 in F2_TYPES if f2 not in skip_f2
                        for f1 in F1_TYPES if f1 not in skip_f1]

    for idx, (f2, f1) in enumerate(combos, 1):
        tag = " [SOFTMAX BASELINE]" if (f2, f1) == SOFTMAX_KEY else ""
        print(f"[{task_name}] ({idx}/{len(combos)}) f2={f2}  f1={f1}{tag}")

        model = make_model_fn(f1, f2).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_epoch(
                model, train_loader, opt, criterion, device, max_train_batches)
            vl_loss, vl_acc = evaluate(
                model, val_loader, criterion, device, max_val_batches)
            sched.step()
            print(f"  ep{epoch:02d}: train={tr_acc:.4f}  val={vl_acc:.4f}  "
                  f"(loss {vl_loss:.4f})  {time.time()-t0:.1f}s")

        results[(f2, f1)] = {"val_acc": vl_acc, "val_loss": vl_loss,
                             "train_acc": tr_acc, "train_loss": tr_loss}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

# ── Result display + saving ───────────────────────────────────────────────────

ABBREV = {
    "full_set":                   "full_set",
    "modular_dot":                "mod_dot",
    "modular_dot_hard_singleton": "mod_dot_hard",
    "modular_dot_first_free":     "mod_dot_free",
    "logsumexp":                  "logsumexp",
    "dot_repulsion":              "dot_repul",
    "neural_mlp":                 "neural_mlp",
}


def format_table(task_name, results, skip_f2=None, skip_f1=None) -> str:
    skip_f2   = set(skip_f2 or [])
    skip_f1   = set(skip_f1 or [])
    active_f2 = [f for f in F2_TYPES if f not in skip_f2]
    active_f1 = [f for f in F1_TYPES if f not in skip_f1]

    sm_acc = results.get(SOFTMAX_KEY, {}).get("val_acc", float("nan"))
    lines  = []
    lines.append("=" * 95)
    lines.append(f"  {task_name}  |  val accuracy  |  softmax baseline (*) = {sm_acc:.4f}")
    lines.append("=" * 95)

    col_w, f1_w = 13, 22
    header = f"{'F1 \\ F2':<{f1_w}}" + "".join(f"{ABBREV[f2]:>{col_w}}" for f2 in active_f2)
    lines.append(header)
    lines.append("─" * len(header))

    for f1 in active_f1:
        row = f"{f1:<{f1_w}}"
        for f2 in active_f2:
            key = (f2, f1)
            if key not in results:
                cell = "skip"
            else:
                acc    = results[key]["val_acc"]
                marker = "*" if key == SOFTMAX_KEY else " "
                cell   = f"{acc:.4f}{marker}"
            row += cell.rjust(col_w)
        lines.append(row)

    gam = {k: v["val_acc"] for k, v in results.items() if k != SOFTMAX_KEY}
    if gam:
        best_k = max(gam, key=gam.get)
        best_v = gam[best_k]
        lines.append(f"\nBest GAM: f2={best_k[0]}  f1={best_k[1]}  "
                     f"val_acc={best_v:.4f}  delta={best_v - sm_acc:+.4f}")
    return "\n".join(lines)


def save_results(all_results: dict, cfg: dict, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = results_dir / run_id
    run_dir.mkdir()

    # JSON — keyed by "f2|f1"
    serialisable = {
        "run_id": run_id,
        "config": cfg,
        "tasks":  {
            task: {
                f"{k[0]}|{k[1]}": v for k, v in res.items()
            }
            for task, res in all_results.items()
        },
    }
    json_path = run_dir / "results.json"
    json_path.write_text(json.dumps(serialisable, indent=2))
    print(f"\nResults JSON -> {json_path}")

    # Human-readable summary
    summary_lines = [f"LRA Ablation Results  ({run_id})", ""]
    for task, res in all_results.items():
        summary_lines.append(
            format_table(task, res,
                         skip_f2=cfg.get("skip_f2"), skip_f1=cfg.get("skip_f1"))
        )
        summary_lines.append("")

    summary_lines.append("=" * 70)
    summary_lines.append(f"{'Task':<15} {'Softmax':>10} {'Best GAM':>10} {'Delta':>8}  Best config")
    summary_lines.append("─" * 70)
    for task, res in all_results.items():
        sm_acc = res.get(SOFTMAX_KEY, {}).get("val_acc", float("nan"))
        gam    = {k: v["val_acc"] for k, v in res.items() if k != SOFTMAX_KEY}
        if not gam:
            continue
        best_k = max(gam, key=gam.get)
        best_v = gam[best_k]
        summary_lines.append(
            f"{task:<15} {sm_acc:>10.4f} {best_v:>10.4f} {best_v-sm_acc:>+8.4f}"
            f"  f2={best_k[0]}  f1={best_k[1]}"
        )

    txt_path = run_dir / "summary.txt"
    txt_path.write_text("\n".join(summary_lines))
    print(f"Summary      -> {txt_path}")

    # Also write a symlink / copy as "latest"
    latest = results_dir / "latest"
    if latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(run_dir.name)
    except Exception:
        pass   # symlinks may fail on some Windows setups

    return run_dir

# ── Data download ─────────────────────────────────────────────────────────────

LRA_RELEASE_URL = "https://storage.googleapis.com/long-range-arena/lra_release.gz"


def _stream_extract(url: str, prefix: str, dest_root: Path, stop_count=None) -> int:
    """Stream a .tar.gz from url, save members matching prefix into dest_root."""
    print(f"  Streaming {url} ...")
    resp  = requests.get(url, stream=True)
    resp.raise_for_status()
    count = 0
    with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
        for member in tar:
            if not member.name.startswith(prefix) or member.isdir():
                continue
            rel  = member.name[len(prefix):]
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            f = tar.extractfile(member)
            if f:
                dest.write_bytes(f.read())
                count += 1
                if count % 500 == 0:
                    print(f"    ... {count} files extracted")
            if stop_count and count >= stop_count:
                break
    return count


def download_listops(root: Path) -> Path:
    dest   = root / "listops-1000"
    needed = ["basic_train.tsv", "basic_val.tsv", "basic_test.tsv"]
    if all((dest / f).exists() for f in needed):
        print(f"ListOps already present at {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    n = _stream_extract(LRA_RELEASE_URL,
                        "lra_release/lra_data/listops-1000/",
                        dest, stop_count=len(needed))
    print(f"  ListOps: {n} files -> {dest}")
    return dest


def download_pathfinder(root: Path, img_size: int = 32) -> Path:
    dest = root / f"pathfinder{img_size}" / "curv_contour_length_14"
    if dest.exists() and any(dest.rglob("*.png")):
        print(f"Pathfinder already present at {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"lra_release/lra_data/pathfinder{img_size}/curv_contour_length_14/"
    n = _stream_extract(LRA_RELEASE_URL, prefix, dest)
    print(f"  Pathfinder{img_size}: {n} files -> {dest}")
    return dest


def prepare_imdb_bytes(root: Path, seq_len: int = 4096) -> Path:
    dest       = root / "text"
    train_path = dest / "imdb_train.pt"
    val_path   = dest / "imdb_val.pt"
    if train_path.exists() and val_path.exists():
        print(f"IMDb bytes already present at {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  (needed for LRA Text)")
    print("Downloading IMDb via HuggingFace datasets ...")
    imdb = load_dataset("imdb")
    for split, hf_key, out in [("train", "train", train_path),
                                ("val",   "test",  val_path)]:
        seqs, labels = [], []
        for item in imdb[hf_key]:
            ids = list(item["text"].encode("utf-8", errors="replace"))[:seq_len]
            ids += [0] * (seq_len - len(ids))
            seqs.append(ids)
            labels.append(item["label"])
        torch.save({"seqs":   torch.tensor(seqs,   dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)}, out)
        print(f"  {split}: {len(seqs)} samples -> {out}")
    return dest

# ── Datasets ──────────────────────────────────────────────────────────────────

LISTOPS_TOKENS = {
    "[MIN": 0, "[MAX": 1, "[MED": 2, "[SM": 3, "]": 4, "PAD": 5,
    **{str(i): 6 + i for i in range(10)},
}
LISTOPS_VOCAB = len(LISTOPS_TOKENS)


class ListOpsSynthetic(Dataset):
    def __init__(self, size=2000, seq_len=512, seed=42):
        rng    = random.Random(seed)
        ops    = ["[MIN", "[MAX", "[MED", "[SM"]
        tok2id = LISTOPS_TOKENS
        self.samples = []

        def gen(depth):
            if depth == 0 or rng.random() < 0.4:
                v = rng.randint(0, 9)
                return [str(v)], v
            op = rng.choice(ops)
            toks, vals = [op], []
            for _ in range(rng.randint(2, 4)):
                st, sv = gen(depth - 1)
                toks += st
                vals.append(sv)
            toks.append("]")
            r = {"[MIN": min, "[MAX": max,
                 "[MED": lambda v: sorted(v)[len(v) // 2]}.get(
                     op, lambda v: sum(v) % 10)(vals)
            return toks, r

        for _ in range(size):
            toks, label = gen(4)
            ids = [tok2id.get(t, tok2id["PAD"]) for t in toks][:seq_len]
            ids += [tok2id["PAD"]] * (seq_len - len(ids))
            self.samples.append((ids, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids, lbl = self.samples[i]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lbl, dtype=torch.long)


def load_listops_real(data_dir: Path, split: str, seq_len: int) -> Dataset:
    fname  = {"train": "basic_train.tsv", "val": "basic_val.tsv",
              "test": "basic_test.tsv"}[split]
    tok2id = LISTOPS_TOKENS
    samples = []
    with open(data_dir / fname) as f:
        next(f)
        for line in f:
            lbl_str, seq_str = line.strip().split("\t", 1)
            ids = [tok2id.get(t, tok2id["PAD"]) for t in seq_str.split()][:seq_len]
            ids += [tok2id["PAD"]] * (seq_len - len(ids))
            samples.append((ids, int(lbl_str)))

    class _DS(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            ids, lbl = samples[i]
            return (torch.tensor(ids, dtype=torch.long),
                    torch.tensor(lbl, dtype=torch.long))
    return _DS()


class PathfinderSynthetic(Dataset):
    def __init__(self, size=2000, img_size=32, seed=0):
        torch.manual_seed(seed)
        seq_len      = img_size * img_size
        imgs         = torch.randint(0, 2, (size, seq_len))
        centre       = imgs[:, seq_len // 4: 3 * seq_len // 4].float().mean(dim=1)
        self.data    = imgs
        self.labels  = (centre > 0.5).long()

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]


def load_pathfinder_real(data_dir: Path, split: str, img_size: int) -> Dataset:
    import numpy as np
    from PIL import Image as PILImage

    meta_dir = data_dir / "metadata"
    img_dir  = data_dir / "imgs"

    all_entries = []
    for meta_file in sorted(meta_dir.glob("*.npy")):
        all_entries.extend(np.load(meta_file, allow_pickle=True))

    if not all_entries:
        raise RuntimeError(f"No metadata .npy files found in {meta_dir}")

    n = len(all_entries)
    splits = {
        "train": all_entries[:int(0.8 * n)],
        "val":   all_entries[int(0.8 * n):int(0.9 * n)],
        "test":  all_entries[int(0.9 * n):],
    }
    entries = splits[split]

    samples = []
    for entry in entries:
        img_rel = str(entry[0])
        label   = int(entry[1])
        img     = np.array(PILImage.open(img_dir / img_rel).convert("L"))
        if img.shape[0] != img_size:
            img = np.array(PILImage.fromarray(img).resize((img_size, img_size)))
        pixels = (img.flatten() > 128).astype(int)
        samples.append((pixels, label))

    class _DS(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            px, lbl = samples[i]
            return (torch.tensor(px, dtype=torch.long),
                    torch.tensor(lbl, dtype=torch.long))
    return _DS()


class TextSynthetic(Dataset):
    def __init__(self, size=2000, seq_len=1024, seed=0):
        torch.manual_seed(seed)
        self.data   = torch.randint(0, 256, (size, seq_len))
        self.labels = (self.data.float().mean(dim=1) > 128).long()

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]


def load_imdb_bytes(data_dir: Path, split: str) -> Dataset:
    data   = torch.load(data_dir / f"imdb_{split}.pt", weights_only=True)
    seqs   = data["seqs"]
    labels = data["labels"]

    class _DS(Dataset):
        def __len__(self): return len(seqs)
        def __getitem__(self, i): return seqs[i], labels[i]
    return _DS()


class CIFAR10PixelSequence(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True):
        from torchvision.datasets import CIFAR10
        import torchvision.transforms as T
        ds = CIFAR10(root=root, train=train, download=download,
                     transform=T.Compose([T.Grayscale(), T.ToTensor()]))
        self.seqs, self.labels = [], []
        for img, lbl in ds:
            self.seqs.append((img.flatten() * 255).long())
            self.labels.append(lbl)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        return self.seqs[i], torch.tensor(self.labels[i], dtype=torch.long)


class ImageSynthetic(Dataset):
    def __init__(self, size=2000, seed=0):
        torch.manual_seed(seed)
        self.data   = torch.randint(0, 256, (size, 1024))
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

# ── Task runners ──────────────────────────────────────────────────────────────

def run_task(task_name, train_ds, val_ds, make_model_fn,
             batch_size, epochs, device,
             max_train_batches, max_val_batches,
             skip_f2, skip_f1):
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_ldr   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    results = run_ablation(
        task_name, train_ldr, val_ldr, make_model_fn,
        epochs=epochs, device=device,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        skip_f2=skip_f2, skip_f1=skip_f1,
    )

    table = format_table(task_name, results, skip_f2, skip_f1)
    print("\n" + table + "\n")
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+",
                   choices=["listops", "pathfinder", "text", "image"],
                   default=["listops", "pathfinder", "text", "image"],
                   help="Which LRA tasks to run (default: all)")
    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs per model (default: 3)")
    p.add_argument("--dim", type=int, default=64,
                   help="Transformer hidden dimension (default: 64)")
    p.add_argument("--depth", type=int, default=2,
                   help="Number of transformer blocks (default: 2)")
    p.add_argument("--skip-f2", nargs="+", default=[],
                   choices=F2_TYPES, metavar="F2",
                   help="F2 types to skip (e.g. --skip-f2 neural_mlp)")
    p.add_argument("--skip-f1", nargs="+", default=[],
                   choices=F1_TYPES, metavar="F1",
                   help="F1 types to skip (e.g. --skip-f1 transformer)")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test mode: 2 epochs, 50-batch limit per epoch")
    p.add_argument("--no-download-pathfinder", action="store_true",
                   help="Skip downloading the large Pathfinder dataset (~5 GB)")
    p.add_argument("--data-root", type=Path, default=Path("./lra_data"),
                   help="Directory for downloaded LRA data (default: ./lra_data)")
    p.add_argument("--results-dir", type=Path, default=Path("./lra_results"),
                   help="Directory to save result files (default: ./lra_results)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = choose_device()

    if args.quick:
        args.epochs          = 2
        max_train_batches    = 50
        max_val_batches      = 20
    else:
        max_train_batches    = None
        max_val_batches      = None

    print(f"Device : {device}")
    print(f"Tasks  : {args.tasks}")
    print(f"Epochs : {args.epochs}")
    print(f"Dim    : {args.dim}   Depth: {args.depth}")
    n_combos = (len(F2_TYPES) - len(args.skip_f2)) * (len(F1_TYPES) - len(args.skip_f1))
    print(f"Combos : {n_combos} per task  ({len(args.tasks)} tasks = {n_combos * len(args.tasks)} total runs)")
    print()

    # ── Download data ────────────────────────────────────────────────────────
    args.data_root.mkdir(parents=True, exist_ok=True)
    listops_dir = pathfinder_dir = text_dir = None

    if "listops" in args.tasks:
        print("=== ListOps data ===")
        listops_dir = download_listops(args.data_root)

    if "text" in args.tasks:
        print("\n=== LRA Text data (IMDb bytes) ===")
        text_dir = prepare_imdb_bytes(args.data_root, seq_len=4096)

    if "pathfinder" in args.tasks and not args.no_download_pathfinder:
        print("\n=== Pathfinder32 data (large download) ===")
        pathfinder_dir = download_pathfinder(args.data_root, img_size=32)
    elif "pathfinder" in args.tasks:
        # Check if already downloaded
        candidate = args.data_root / "pathfinder32" / "curv_contour_length_14"
        if candidate.exists() and any(candidate.rglob("*.png")):
            pathfinder_dir = candidate

    print()

    # ── Run tasks ────────────────────────────────────────────────────────────
    all_results  = {}
    data_sources = {}

    # ── ListOps ──────────────────────────────────────────────────────────────
    if "listops" in args.tasks:
        seq_len = 2048
        if listops_dir and (listops_dir / "basic_train.tsv").exists():
            print(f"Loading real ListOps (seq_len={seq_len}) ...")
            lo_tr = load_listops_real(listops_dir, "train", seq_len)
            lo_vl = load_listops_real(listops_dir, "val",   seq_len)
            data_sources["ListOps"] = "real"
        else:
            print("WARNING: ListOps data not found, using synthetic fallback.")
            lo_tr = ListOpsSynthetic(size=2000, seq_len=seq_len, seed=0)
            lo_vl = ListOpsSynthetic(size=500,  seq_len=seq_len, seed=99)
            data_sources["ListOps"] = "synthetic"

        print(f"  {len(lo_tr)} train / {len(lo_vl)} val\n")

        def make_listops(f1, f2):
            return SequenceClassifier(
                vocab_size=LISTOPS_VOCAB, num_classes=10,
                max_seq_len=seq_len, dim=args.dim, depth=args.depth,
                attn_factory=lambda d: build_attn_module(d, f1, f2),
            )

        all_results["ListOps"] = run_task(
            "ListOps", lo_tr, lo_vl, make_listops,
            batch_size=32, epochs=args.epochs, device=device,
            max_train_batches=max_train_batches, max_val_batches=max_val_batches,
            skip_f2=args.skip_f2, skip_f1=args.skip_f1,
        )

    # ── Pathfinder ───────────────────────────────────────────────────────────
    if "pathfinder" in args.tasks:
        img_size = 32
        seq_len  = img_size ** 2
        if pathfinder_dir and pathfinder_dir.exists() and any(pathfinder_dir.rglob("*.png")):
            print(f"Loading real Pathfinder (img={img_size}x{img_size}) ...")
            pf_tr = load_pathfinder_real(pathfinder_dir, "train", img_size)
            pf_vl = load_pathfinder_real(pathfinder_dir, "val",   img_size)
            data_sources["Pathfinder"] = "real"
        else:
            print("WARNING: Pathfinder data not found, using synthetic fallback.")
            print("  Re-run without --no-download-pathfinder to get real data.")
            pf_tr = PathfinderSynthetic(size=2000, img_size=img_size, seed=1)
            pf_vl = PathfinderSynthetic(size=500,  img_size=img_size, seed=77)
            data_sources["Pathfinder"] = "synthetic"

        print(f"  {len(pf_tr)} train / {len(pf_vl)} val\n")

        def make_pathfinder(f1, f2):
            return SequenceClassifier(
                vocab_size=2, num_classes=2,
                max_seq_len=seq_len, dim=args.dim, depth=args.depth,
                attn_factory=lambda d: build_attn_module(d, f1, f2),
            )

        all_results["Pathfinder"] = run_task(
            "Pathfinder", pf_tr, pf_vl, make_pathfinder,
            batch_size=32, epochs=args.epochs, device=device,
            max_train_batches=max_train_batches, max_val_batches=max_val_batches,
            skip_f2=args.skip_f2, skip_f1=args.skip_f1,
        )

    # ── LRA Text ─────────────────────────────────────────────────────────────
    if "text" in args.tasks:
        seq_len = 4096
        if text_dir and (text_dir / "imdb_train.pt").exists():
            print(f"Loading real IMDb bytes (seq_len={seq_len}) ...")
            tx_tr = load_imdb_bytes(text_dir, "train")
            tx_vl = load_imdb_bytes(text_dir, "val")
            data_sources["LRA Text"] = "real"
        else:
            print("WARNING: IMDb byte data not found, using synthetic fallback.")
            tx_tr = TextSynthetic(size=2000, seq_len=seq_len, seed=2)
            tx_vl = TextSynthetic(size=500,  seq_len=seq_len, seed=55)
            data_sources["LRA Text"] = "synthetic"

        print(f"  {len(tx_tr)} train / {len(tx_vl)} val\n")

        def make_text(f1, f2):
            return SequenceClassifier(
                vocab_size=256, num_classes=2,
                max_seq_len=seq_len, dim=args.dim, depth=args.depth,
                attn_factory=lambda d: build_attn_module(d, f1, f2),
            )

        all_results["LRA Text"] = run_task(
            "LRA Text", tx_tr, tx_vl, make_text,
            batch_size=16, epochs=args.epochs, device=device,
            max_train_batches=max_train_batches, max_val_batches=max_val_batches,
            skip_f2=args.skip_f2, skip_f1=args.skip_f1,
        )

    # ── LRA Image ─────────────────────────────────────────────────────────────
    if "image" in args.tasks:
        seq_len = 1024
        print("Loading CIFAR-10 as pixel sequences ...")
        try:
            im_tr = CIFAR10PixelSequence(root=str(args.data_root / "cifar10"), train=True)
            im_vl = CIFAR10PixelSequence(root=str(args.data_root / "cifar10"), train=False)
            data_sources["LRA Image"] = "real"
        except Exception as e:
            print(f"WARNING: Could not load CIFAR-10 ({e}). Using synthetic fallback.")
            im_tr = ImageSynthetic(size=2000, seed=3)
            im_vl = ImageSynthetic(size=500,  seed=66)
            data_sources["LRA Image"] = "synthetic"

        print(f"  {len(im_tr)} train / {len(im_vl)} val\n")

        def make_image(f1, f2):
            return SequenceClassifier(
                vocab_size=256, num_classes=10,
                max_seq_len=seq_len, dim=args.dim, depth=args.depth,
                attn_factory=lambda d: build_attn_module(d, f1, f2),
            )

        all_results["LRA Image"] = run_task(
            "LRA Image", im_tr, im_vl, make_image,
            batch_size=32, epochs=args.epochs, device=device,
            max_train_batches=max_train_batches, max_val_batches=max_val_batches,
            skip_f2=args.skip_f2, skip_f1=args.skip_f1,
        )

    # ── Save results ─────────────────────────────────────────────────────────
    cfg = {
        "epochs":    args.epochs,
        "dim":       args.dim,
        "depth":     args.depth,
        "skip_f2":   args.skip_f2,
        "skip_f1":   args.skip_f1,
        "quick":     args.quick,
        "data_sources": data_sources,
    }
    run_dir = save_results(all_results, cfg, args.results_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Task':<15} {'Data':>10} {'Softmax':>10} {'Best GAM':>10} {'Delta':>8}  Best config")
    print("─" * 70)
    for task, res in all_results.items():
        sm_acc = res.get(SOFTMAX_KEY, {}).get("val_acc", float("nan"))
        gam    = {k: v["val_acc"] for k, v in res.items() if k != SOFTMAX_KEY}
        if not gam:
            continue
        best_k = max(gam, key=gam.get)
        best_v = gam[best_k]
        src    = data_sources.get(task, "?")
        print(f"{task:<15} {src:>10} {sm_acc:>10.4f} {best_v:>10.4f} {best_v-sm_acc:>+8.4f}"
              f"  f2={best_k[0]}  f1={best_k[1]}")

    print(f"\nFull results saved to: {run_dir}")


if __name__ == "__main__":
    main()
