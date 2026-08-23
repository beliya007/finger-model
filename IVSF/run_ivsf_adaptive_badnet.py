from __future__ import annotations

import argparse
import csv
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ivsf_pipeline.data import cifar10_mean_std, normalize_batch
from ivsf_pipeline.model import load_resnet20_cifar10


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    want_cuda = str(device_arg).lower().startswith("cuda") and torch.cuda.is_available()
    return torch.device("cuda" if want_cuda else "cpu")


def _pointwise_mi_from_probs(selected_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(selected_probs.astype(np.float64), eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    p_marginal = probs.mean(axis=0, keepdims=True)
    p_marginal = np.clip(p_marginal, eps, 1.0)
    local_mi = np.sum(probs * (np.log(probs) - np.log(p_marginal)), axis=1)
    return local_mi.astype(np.float32)


def _predict_probs(model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    mean, std = cifar10_mean_std(device)
    model.eval()
    out_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            logits = model(normalize_batch(x[i : i + batch_size], mean, std))
            probs = torch.softmax(logits, dim=1)
            out_list.append(probs.detach().cpu().numpy())
    if not out_list:
        return np.empty((0, 10), dtype=np.float32)
    return np.concatenate(out_list, axis=0).astype(np.float32)


def _load_reference_fingerprints(path: Path, model: nn.Module, device: torch.device, batch_size: int = 256) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    if "selected_images" not in data.files:
        raise ValueError(f"{path} does not contain selected_images.")

    images = data["selected_images"].astype(np.float32)
    if "selected_probs" in data.files:
        probs = data["selected_probs"].astype(np.float32)
    else:
        probs = _predict_probs(model, images, device, batch_size=batch_size)

    labels = np.argmax(probs, axis=1).astype(np.int64)
    mi = _pointwise_mi_from_probs(probs)
    return {
        "images": images,
        "probs": probs,
        "labels": labels,
        "mi": mi,
    }


def _compute_detection_rate(attacked_probs: np.ndarray, ref: Dict[str, np.ndarray], tau: float) -> Tuple[float, float]:
    """Use ns=1 as the unified detection rule for all adaptive-attack sweeps.

    A sample is treated as detected if either its label differs from the clean
    reference label or its MI deviates beyond the threshold. This matches the
    single-sample evaluation required by the user's experiment definition.
    """
    attacked_labels = np.argmax(attacked_probs, axis=1).astype(np.int64)
    attacked_mi = _pointwise_mi_from_probs(attacked_probs)
    abs_mi_diff = np.abs(ref["mi"] - attacked_mi)
    detected = (attacked_labels != ref["labels"]) | (abs_mi_diff > tau)
    rate = 100.0 * float(detected.mean())
    return float(rate), float(rate)


def parse_sample_budgets(raw: str) -> List[int]:
    if not raw or not raw.strip():
        return [0, 50, 100, 200, 500, 1000, 2000, 5000]
    budgets = []
    for item in raw.split(","):
        s = item.strip()
        if s:
            budgets.append(int(s))
    return budgets


class AdaptiveBadNetDataset(Dataset):
    def __init__(
        self,
        clean_root: str,
        adaptive_images: np.ndarray,
        trigger_label: int = 0,
        trigger_size: int = 4,
        trigger_alpha: float = 1.0,
        seed: int = 1337,
        clean_ratio: float = 0.5,
    ) -> None:
        self.clean_ds = datasets.CIFAR10(root=clean_root, train=True, download=True, transform=transforms.ToTensor())
        self.adaptive_images = torch.from_numpy(np.asarray(adaptive_images, dtype=np.float32))
        self.trigger_label = int(trigger_label)
        self.trigger_size = int(trigger_size)
        self.trigger_alpha = float(trigger_alpha)
        self.clean_ratio = float(clean_ratio)
        self.rng = np.random.default_rng(seed)

        self.n_adaptive = self.adaptive_images.shape[0]
        self.n_clean = len(self.clean_ds)
        self.total = max(self.n_clean, self.n_adaptive)

    def _apply_badnet(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        patch = out[:, -self.trigger_size:, -self.trigger_size:]
        patched = patch * (1.0 - self.trigger_alpha) + self.trigger_alpha
        out[:, -self.trigger_size:, -self.trigger_size:] = patched
        return out

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.n_adaptive > 0 and idx < self.n_adaptive:
            x = self.adaptive_images[idx].clone()
            x = self._apply_badnet(x)
            return x, self.trigger_label

        # mix in clean CIFAR-10 images to keep the adaptation realistic
        clean_idx = int(idx - self.n_adaptive) % self.n_clean
        x, y = self.clean_ds[clean_idx]
        return x, int(y)


def train_adaptive_badnet(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
) -> nn.Module:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    mean, std = cifar10_mean_std(device)

    for _ in range(int(epochs)):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalize_batch(x, mean, std))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="IVSF adaptive BadNet experiment: increase fingerprint-sample budget and measure detection rate.")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resnet20_cifar10.pth")
    parser.add_argument("--fingerprints", type=str, default="./outputs/full_run/fingerprints.npz")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/adaptive_badnet_ivsf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample-budgets", type=str, default="0,50,100,200,500,1000,2000,5000")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trigger-label", type=int, default=0)
    parser.add_argument("--trigger-size", type=int, default=4)
    parser.add_argument("--trigger-alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_resnet20_cifar10(Path(args.checkpoint_path), device=device)
    ref = _load_reference_fingerprints(Path(args.fingerprints), model, device, batch_size=256)

    budgets = parse_sample_budgets(args.sample_budgets)
    max_budget = int(ref["images"].shape[0])
    budgets = sorted({int(b) for b in budgets if int(b) <= max_budget})
    if not budgets:
        budgets = [0]
    if 0 not in budgets:
        budgets = [0] + budgets

    rows: List[Dict[str, object]] = []

    for budget in budgets:
        budget = int(budget)
        adaptive_images = ref["images"][:budget].copy()

        dataset = AdaptiveBadNetDataset(
            clean_root=args.data_root,
            adaptive_images=adaptive_images,
            trigger_label=args.trigger_label,
            trigger_size=args.trigger_size,
            trigger_alpha=args.trigger_alpha,
            seed=args.seed,
            clean_ratio=0.5,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        adapted_model = train_adaptive_badnet(
            model=deepcopy(model),
            train_loader=loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )

        attacked_probs = _predict_probs(adapted_model, ref["images"], device, batch_size=256)
        det_rate, ns1 = _compute_detection_rate(attacked_probs, ref, tau=args.tau)
        rows.append(
            {
                "sample_budget": budget,
                "detection_rate": round(det_rate, 4),
                "ns1_rate": round(ns1, 4),
                "tau": args.tau,
                "epochs": args.epochs,
                "lr": args.lr,
                "trigger_label": args.trigger_label,
                "trigger_size": args.trigger_size,
            }
        )
        print(f"budget={budget}: detection_rate={det_rate:.2f}% ns1={ns1:.2f}%")

    out_csv = output_dir / "ivsf_adaptive_badnet_results.csv"
    write_csv(out_csv, rows)
    print(f"saved adaptive results to {out_csv}")


if __name__ == "__main__":
    main()
