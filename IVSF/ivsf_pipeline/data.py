from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def dataset_mean_std(dataset: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_name = str(dataset).lower()
    if dataset_name == "cifar10":
        values = ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif dataset_name == "gtsrb":
        values = ([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected cifar10 or gtsrb.")

    mean = torch.tensor(values[0], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(values[1], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def cifar10_mean_std(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return dataset_mean_std("cifar10", device)


def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def build_dataset_subset_loader(
    data_root: str,
    num_samples: int,
    num_workers: int,
    dataset: str = "cifar10",
    random_subset: bool = True,
    subset_seed: Optional[int] = None,
    fixed_indices: Optional[List[int]] = None,
) -> Tuple[DataLoader, List[int]]:
    root = Path(data_root).expanduser()
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset_name = str(dataset).lower()
    if dataset_name == "cifar10":
        source_dataset = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    elif dataset_name == "gtsrb":
        source_dataset = datasets.GTSRB(root=str(root), split="train", download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected cifar10 or gtsrb.")

    if fixed_indices is not None:
        sampled_indices = [int(i) for i in fixed_indices]
        if len(sampled_indices) == 0:
            raise ValueError("fixed_indices is empty.")
        for idx in sampled_indices:
            if idx < 0 or idx >= len(source_dataset):
                raise ValueError(f"fixed index out of range: {idx}")
    else:
        n = min(num_samples, len(source_dataset))
        if random_subset:
            rng = np.random.default_rng(subset_seed)
            sampled_indices = rng.choice(len(source_dataset), size=n, replace=False).tolist()
        else:
            sampled_indices = list(range(n))

    subset = Subset(source_dataset, sampled_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, sampled_indices


def build_cifar10_subset_loader(
    data_root: str,
    num_samples: int,
    num_workers: int,
    random_subset: bool = True,
    subset_seed: Optional[int] = None,
    fixed_indices: Optional[List[int]] = None,
) -> Tuple[DataLoader, List[int]]:
    return build_dataset_subset_loader(
        data_root=data_root,
        num_samples=num_samples,
        num_workers=num_workers,
        dataset="cifar10",
        random_subset=random_subset,
        subset_seed=subset_seed,
        fixed_indices=fixed_indices,
    )
