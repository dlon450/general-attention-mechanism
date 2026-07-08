#!/usr/bin/env python3
"""Dependency-free CIFAR-10 loader (no torchvision).

Reads the standard `cifar-10-batches-py` pickle format and implements the usual
train/test transforms (RandomCrop(32, pad=4), RandomHorizontalFlip, ToTensor,
Normalize) in pure torch. Used as a fallback by train_vit_cifar.build_dataloaders
when torchvision is not installed.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10Local(Dataset):
    """CIFAR-10 from `root/cifar-10-batches-py`, transforms in pure torch."""

    _TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
    _TEST_BATCHES = ["test_batch"]

    def __init__(self, root: str | Path, train: bool = True):
        super().__init__()
        self.train = bool(train)
        base = Path(root) / "cifar-10-batches-py"
        if not base.is_dir():
            raise FileNotFoundError(f"missing {base} (expected extracted CIFAR-10)")

        batches = self._TRAIN_BATCHES if self.train else self._TEST_BATCHES
        data_chunks: list[torch.Tensor] = []
        targets: list[int] = []
        for name in batches:
            with open(base / name, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            # (N, 3072) uint8 -> (N, 32, 32, 3) HWC
            arr = torch.frombuffer(bytes(entry["data"]), dtype=torch.uint8)
            arr = arr.reshape(-1, 3, 32, 32)
            data_chunks.append(arr)
            targets.extend(entry["labels"])

        self.data = torch.cat(data_chunks, dim=0)  # (N, 3, 32, 32) uint8, CHW
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR_STD).view(3, 1, 1)
        self.classes = list(range(10))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.data[idx].float().div(255.0)  # (3, 32, 32) in [0, 1]
        if self.train:
            img = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="constant", value=0.0).squeeze(0)
            top = int(torch.randint(0, 9, (1,)).item())
            left = int(torch.randint(0, 9, (1,)).item())
            img = img[:, top : top + 32, left : left + 32]
            if torch.rand(()).item() < 0.5:
                img = torch.flip(img, dims=[2])
        img = (img - self.mean) / self.std
        return img, int(self.targets[idx].item())


def build_dataloaders_local(
    dataset_name: str,
    data_dir: str | Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, int]:
    if dataset_name != "cifar10":
        raise ValueError(
            f"dependency-free loader supports cifar10 only, got {dataset_name!r}"
        )
    train_ds = CIFAR10Local(data_dir, train=True)
    test_ds = CIFAR10Local(data_dir, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, 10
