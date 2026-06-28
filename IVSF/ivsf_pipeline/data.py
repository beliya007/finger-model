from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def cifar10_mean_std(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def build_cifar10_subset_loader(data_root: str, num_samples: int, num_workers: int) -> DataLoader:
    root = Path(data_root).expanduser()
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader
