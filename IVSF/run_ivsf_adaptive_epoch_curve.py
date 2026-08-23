from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from attack.data import PoisonedCIFAR10
from ivsf_pipeline.data import cifar10_mean_std, normalize_batch
from ivsf_pipeline.model import load_resnet20_cifar10


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pointwise_mi_from_probs(selected_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(selected_probs.astype(np.float64), eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    p_marginal = probs.mean(axis=0, keepdims=True)
    p_marginal = np.clip(p_marginal, eps, 1.0)
    local_mi = np.sum(probs * (np.log(probs) - np.log(p_marginal)), axis=1)
    return local_mi.astype(np.float32)


def predict_probs(model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    mean, std = cifar10_mean_std(device)
    model.eval()
    out_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            logits = model(normalize_batch(x[i : i + batch_size], mean, std))
            out_list.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    if not out_list:
        return np.empty((0, 10), dtype=np.float32)
    return np.concatenate(out_list, axis=0).astype(np.float32)


def load_reference_fingerprints(path: Path, model: nn.Module, device: torch.device) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    images = data["selected_images"].astype(np.float32)
    if "selected_probs" in data.files:
        probs = data["selected_probs"].astype(np.float32)
    else:
        probs = predict_probs(model, images, device)
    labels = np.argmax(probs, axis=1).astype(np.int64)
    mi = pointwise_mi_from_probs(probs)
    return {"images": images, "probs": probs, "labels": labels, "mi": mi}


def compute_detection_rate(attacked_probs: np.ndarray, ref: Dict[str, np.ndarray], tau: float) -> float:
    attacked_labels = np.argmax(attacked_probs, axis=1).astype(np.int64)
    attacked_mi = pointwise_mi_from_probs(attacked_probs)
    detected = (attacked_labels != ref["labels"]) | (np.abs(ref["mi"] - attacked_mi) > tau)
    return 100.0 * float(detected.mean())


def label_only_rate(attacked_probs: np.ndarray, ref: Dict[str, np.ndarray]) -> float:
    attacked_labels = np.argmax(attacked_probs, axis=1).astype(np.int64)
    return 100.0 * float(np.mean(attacked_labels != ref["labels"]))


class AdaptiveBadNetDataset(Dataset):
    def __init__(self, clean_root: str, adaptive_images: np.ndarray, trigger_label: int = 0, trigger_size: int = 4, trigger_alpha: float = 1.0) -> None:
        self.clean_ds = datasets.CIFAR10(root=clean_root, train=True, download=True, transform=transforms.ToTensor())
        self.adaptive_images = torch.from_numpy(np.asarray(adaptive_images, dtype=np.float32))
        self.trigger_label = int(trigger_label)
        self.trigger_size = int(trigger_size)
        self.trigger_alpha = float(trigger_alpha)
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

    def __getitem__(self, idx: int):
        if self.n_adaptive > 0 and idx < self.n_adaptive:
            x = self.adaptive_images[idx].clone()
            x = self._apply_badnet(x)
            return x, self.trigger_label
        clean_idx = int(idx - self.n_adaptive) % self.n_clean
        x, y = self.clean_ds[clean_idx]
        return x, int(y)


def train_for_epoch_count(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float = 1e-4) -> nn.Module:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    mean, std = cifar10_mean_std(device)
    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalize_batch(x, mean, std))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def make_normal_badnet_loader(data_root: str, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    ds = PoisonedCIFAR10(
        root=data_root,
        train=True,
        attack_type='badnet',
        poison_ratio=0.1,
        trigger_label=0,
        trigger_size=4,
        trigger_alpha=1.0,
        seed=1337,
        always_trigger=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'))


def make_leakage_loader(data_root: str, leaked_images: np.ndarray, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    ds = AdaptiveBadNetDataset(clean_root=data_root, adaptive_images=leaked_images, trigger_label=0, trigger_size=4, trigger_alpha=1.0)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'))


def write_csv(out_path: Path, rows: List[Dict[str, float]]) -> None:
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'Tampering', 'Validation_Leakage', 'Validation_NoLeakage'])
        writer.writeheader()
        writer.writerows(rows)


def make_plot(csv_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    epochs = [int(row['epoch']) for row in rows]
    tampering = [float(row['Tampering']) for row in rows]
    leakage = [float(row['Validation_Leakage']) for row in rows]
    no_leakage = [float(row['Validation_NoLeakage']) for row in rows]

    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, tampering, color='#dd2222', linestyle='--', marker='^', linewidth=2.5, markersize=7, label='Leakage + label-only')
    ax.plot(epochs, leakage, color='#0022cc', linestyle='--', marker='s', linewidth=2.5, markersize=7, label='Leakage + full IVSF')
    ax.plot(epochs, no_leakage, color='#000000', linestyle='--', marker='o', linewidth=2.5, markersize=6, label='No leakage + full IVSF')

    ax.set_xlabel('# of training epoch', fontsize=20)
    ax.set_ylabel('Detection rate (%) (ns=1)', fontsize=20)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, linestyle='--', linewidth=1.0, color='#ccccff')
    ax.legend(loc='lower right', fontsize=16)
    ax.set_title('(a) Adaptive backdoor attacks (30 leaked fingerprints, ns=1)', fontsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    print(f'Saved CSV: {csv_path}')
    print(f'Saved PNG: {png_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Real IVSF adaptive-attack epoch sweep under fixed 30 leaked fingerprints.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/resnet20_cifar10.pth')
    parser.add_argument('--fingerprints', type=str, default='./IVSF/outputs/full_run/fingerprints.npz')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./IVSF/outputs/adaptive_epoch_curve')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if args.device.lower().startswith('cuda') and torch.cuda.is_available() else 'cpu')
    clean_model, _ = load_resnet20_cifar10(Path(args.checkpoint_path), device=device)
    ref = load_reference_fingerprints(Path(args.fingerprints), clean_model, device)
    leaked = ref['images'][:30].copy()

    rows: List[Dict[str, float]] = []
    for epoch in range(0, 11):
        if epoch == 0:
            rows.append({'epoch': 0, 'Tampering': 0.0, 'Validation_Leakage': 0.0, 'Validation_NoLeakage': 0.0})
            continue

        normal_model = deepcopy(clean_model)
        normal_loader = make_normal_badnet_loader(args.data_root, args.batch_size, args.num_workers, device)
        normal_model = train_for_epoch_count(normal_model, normal_loader, device, epochs=epoch, lr=args.lr)
        normal_probs = predict_probs(normal_model, ref['images'], device)
        normal_rate = compute_detection_rate(normal_probs, ref, tau=args.tau)

        leakage_model = deepcopy(clean_model)
        leakage_loader = make_leakage_loader(args.data_root, leaked, args.batch_size, args.num_workers, device)
        leakage_model = train_for_epoch_count(leakage_model, leakage_loader, device, epochs=epoch, lr=args.lr)
        leakage_probs = predict_probs(leakage_model, ref['images'], device)
        leakage_rate = compute_detection_rate(leakage_probs, ref, tau=args.tau)
        tampering_rate = label_only_rate(leakage_probs, ref)

        rows.append({
            'epoch': epoch,
            'Tampering': round(float(tampering_rate), 2),
            'Validation_Leakage': round(float(leakage_rate), 2),
            'Validation_NoLeakage': round(float(normal_rate), 2),
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'ivsf_adaptive_epoch_curve.csv'
    png_path = out_dir / 'IVSF_adaptive_epoch.png'
    write_csv(csv_path, rows)
    make_plot(csv_path, png_path)


if __name__ == '__main__':
    main()
