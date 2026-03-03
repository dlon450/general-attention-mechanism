#!/usr/bin/env python3
"""
Plot training curves from train_vit_cifar.py JSON outputs.

Outputs:
  - accuracy_curves.png      (train/val accuracy vs epoch)
  - speed_curves.png         (images/s and epoch seconds vs epoch)
  - speed_summary_arch.png   (mean speed per architecture)

Example:
  python plot_training_curves.py \
    --inputs results/cifar10_mha.json results/cifar10_general.json \
    --out-dir results/plots
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunData:
    path: Path
    run_name: str
    attention: str
    dataset: str
    epochs: list[int]
    train_acc: list[float]
    val_acc: list[float]
    images_per_sec: list[float]
    epoch_seconds: list[float]

    @property
    def label(self) -> str:
        return f"{self.attention} ({self.run_name})"


def load_run(path: Path) -> RunData:
    with path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"{path}: missing or empty metrics")

    metrics = sorted(metrics, key=lambda x: int(x["epoch"]))
    epochs = [int(m["epoch"]) for m in metrics]
    train_acc = [float(m["train_acc"]) for m in metrics]
    val_acc = [float(m["val_acc"]) for m in metrics]
    images_per_sec = [float(m["images_per_sec"]) for m in metrics]
    epoch_seconds = [float(m["epoch_seconds"]) for m in metrics]

    return RunData(
        path=path,
        run_name=str(payload.get("run_name", path.stem)),
        attention=str(payload.get("attention", "unknown")),
        dataset=str(payload.get("dataset", "unknown")),
        epochs=epochs,
        train_acc=train_acc,
        val_acc=val_acc,
        images_per_sec=images_per_sec,
        epoch_seconds=epoch_seconds,
    )


def get_color_by_attention(runs: list[RunData]) -> dict[str, Any]:
    unique = sorted({r.attention for r in runs})
    cmap = plt.get_cmap("tab10")
    return {attn: cmap(i % 10) for i, attn in enumerate(unique)}


def safe_mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def accuracy_plot(runs: list[RunData], out_path: Path, title: str) -> None:
    colors = get_color_by_attention(runs)
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        c = colors[run.attention]
        ax.plot(
            run.epochs,
            run.train_acc,
            color=c,
            linestyle="-",
            linewidth=2.0,
            alpha=0.9,
            label=f"{run.label} train",
        )
        ax.plot(
            run.epochs,
            run.val_acc,
            color=c,
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            label=f"{run.label} val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title}: Train/Val Accuracy")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def speed_curves_plot(runs: list[RunData], out_path: Path, title: str) -> None:
    colors = get_color_by_attention(runs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)

    ax0, ax1 = axes
    for run in runs:
        c = colors[run.attention]
        ax0.plot(
            run.epochs,
            run.images_per_sec,
            color=c,
            linewidth=2.0,
            alpha=0.9,
            label=run.label,
        )
        ax1.plot(
            run.epochs,
            run.epoch_seconds,
            color=c,
            linewidth=2.0,
            alpha=0.9,
            label=run.label,
        )

    ax0.set_title(f"{title}: Throughput")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Images / sec")
    ax0.grid(True, linestyle=":", alpha=0.35)

    ax1.set_title(f"{title}: Epoch Time")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Seconds / epoch")
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def speed_summary_arch_plot(
    runs: list[RunData],
    out_path: Path,
    title: str,
    skip_first_n: int,
) -> None:
    grouped: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        k = run.attention
        grouped.setdefault(k, {"images_per_sec": [], "epoch_seconds": []})
        grouped[k]["images_per_sec"].extend(run.images_per_sec[skip_first_n:])
        grouped[k]["epoch_seconds"].extend(run.epoch_seconds[skip_first_n:])

    archs = sorted(grouped.keys())
    mean_imgs = [safe_mean(grouped[a]["images_per_sec"]) for a in archs]
    mean_secs = [safe_mean(grouped[a]["epoch_seconds"]) for a in archs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax0, ax1 = axes

    ax0.bar(archs, mean_imgs, color="steelblue", alpha=0.9)
    ax0.set_title(f"{title}: Mean Throughput")
    ax0.set_ylabel("Images / sec")
    ax0.grid(axis="y", linestyle=":", alpha=0.35)

    ax1.bar(archs, mean_secs, color="indianred", alpha=0.9)
    ax1.set_title(f"{title}: Mean Epoch Time")
    ax1.set_ylabel("Seconds / epoch")
    ax1.grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot train/val accuracy and speed from JSON logs.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSON files from train_vit_cifar.py")
    p.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    p.add_argument("--title", type=str, default="")
    p.add_argument(
        "--skip-first-n",
        type=int,
        default=1,
        help="Skip first N epochs when computing mean speed summary (warmup effects).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(x) for x in args.inputs]
    runs = [load_run(p) for p in paths]
    datasets = sorted({r.dataset for r in runs})
    if args.title.strip():
        title = args.title.strip()
    elif len(datasets) == 1:
        title = f"{datasets[0]} ViT Comparison"
    else:
        title = "ViT Comparison"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    acc_path = args.out_dir / "accuracy_curves.png"
    speed_path = args.out_dir / "speed_curves.png"
    speed_summary_path = args.out_dir / "speed_summary_arch.png"

    accuracy_plot(runs, acc_path, title=title)
    speed_curves_plot(runs, speed_path, title=title)
    speed_summary_arch_plot(
        runs,
        speed_summary_path,
        title=title,
        skip_first_n=max(0, int(args.skip_first_n)),
    )

    print("wrote:", acc_path)
    print("wrote:", speed_path)
    print("wrote:", speed_summary_path)


if __name__ == "__main__":
    main()
