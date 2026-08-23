from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from attack.data import BackupDataConfig, PoisonedCIFAR10, create_backdoor_data_loaders
from ivsf_pipeline.data import cifar10_mean_std, normalize_batch
from ivsf_pipeline.model import load_resnet20_cifar10


# NOTE: we intentionally keep the script simple and self-contained so it runs on the current project.
# It mirrors the three-curve experiment described by the user:
#   - black: standard IVSF detection under normal badnet, no leaked fingerprints
#   - blue: standard IVSF detection with 30 leaked fingerprints used in adaptive attack
#   - red: label-only tampering detection with 30 leaked fingerprints used in adaptive attack


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pointwise_mi_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(probs.astype(np.float64), eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    p_marginal = probs.mean(axis=0, keepdims=True)
    p_marginal = np.clip(p_marginal, eps, 1.0)
    mi = np.sum(probs * (np.log(probs) - np.log(p_marginal)), axis=1)
    return mi.astype(np.float32)


def predict_probs(model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    mean, std = cifar10_mean_std(device)
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            logits = model(normalize_batch(xb, mean, std))
            outs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def ivsf_detection_rate(clean_probs: np.ndarray, attacked_probs: np.ndarray, tau: float = 0.05) -> float:
    clean_labels = np.argmax(clean_probs, axis=1)
    attacked_labels = np.argmax(attacked_probs, axis=1)
    clean_mi = pointwise_mi_from_probs(clean_probs)
    attacked_mi = pointwise_mi_from_probs(attacked_probs)
    detected = (attacked_labels != clean_labels) | (np.abs(clean_mi - attacked_mi) > tau)
    return 100.0 * float(np.mean(detected))


def label_only_tampering_rate(clean_probs: np.ndarray, attacked_probs: np.ndarray) -> float:
    clean_labels = np.argmax(clean_probs, axis=1)
    attacked_labels = np.argmax(attacked_probs, axis=1)
    return 100.0 * float(np.mean(attacked_labels != clean_labels))


class LeakedBadNetDataset(Dataset):
    def __init__(self, clean_root: str, leaked_images: np.ndarray, trigger_label: int = 0, trigger_size: int = 4, trigger_alpha: float = 1.0):
        self.base = datasets.CIFAR10(root=clean_root, train=True, download=True, transform=transforms.ToTensor())
        self.leaked_images = torch.from_numpy(np.asarray(leaked_images, dtype=np.float32))
        self.trigger_label = int(trigger_label)
        self.trigger_size = int(trigger_size)
        self.trigger_alpha = float(trigger_alpha)

    def __len__(self) -> int:
        return max(len(self.base), self.leaked_images.shape[0])

    def _apply_trigger(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        patch = out[:, -self.trigger_size:, -self.trigger_size:]
        patched = patch * (1.0 - self.trigger_alpha) + self.trigger_alpha
        out[:, -self.trigger_size:, -self.trigger_size:] = patched
        return out

    def __getitem__(self, index: int):
        if index < self.leaked_images.shape[0]:
            x = self.leaked_images[index].clone()
            x = self._apply_trigger(x)
            return x, self.trigger_label
        x, y = self.base[index - self.leaked_images.shape[0]]
        return x, int(y)


def train_single_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lr: float = 1e-4, weight_decay: float = 1e-4) -> None:
    mean, std = cifar10_mean_std(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalize_batch(x, mean, std))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def make_dataloader_for_normal_badnet(data_root: str, batch_size: int = 128, num_workers: int = 2, device: torch.device = None) -> DataLoader:
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
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device is not None and device.type == 'cuda'))


def make_dataloader_for_leaked_badnet(data_root: str, leaked_images: np.ndarray, batch_size: int = 128, num_workers: int = 2, device: torch.device = None) -> DataLoader:
    ds = LeakedBadNetDataset(clean_root=data_root, leaked_images=leaked_images, trigger_label=0, trigger_size=4, trigger_alpha=1.0)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device is not None and device.type == 'cuda'))


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    fields = ['epoch', 'Tampering', 'Validation_Leakage', 'Validation_NoLeakage']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(csv_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    epochs = [int(row['epoch']) for row in data]
    red = [float(row['Tampering']) for row in data]
    blue = [float(row['Validation_Leakage']) for row in data]
    black = [float(row['Validation_NoLeakage']) for row in data]

    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, red, color='#dd2222', linestyle='--', marker='^', lw=2.5, ms=7, label='Tampering')
    ax.plot(epochs, blue, color='#0022cc', linestyle='--', marker='s', lw=2.5, ms=7, label='Validation w. Leakage')
    ax.plot(epochs, black, color='#000000', linestyle='--', marker='o', lw=2.5, ms=6, label='Validation w.o. Leakage')

    ax.set_xlabel('# of training epoch', fontsize=20)
    ax.set_ylabel('Detection rate (%)', fontsize=20)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, linestyle='--', linewidth=1.0, color='#ccccff')
    ax.legend(loc='lower right', fontsize=16)
    ax.set_title('(a) Adaptive backdoor attacks', fontsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    print(f'Saved CSV: {csv_path}')
    print(f'Saved PNG: {png_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='IVSF adaptive BadNet experiment with 30 leaked fingerprints across epochs 0-10.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/resnet20_cifar10.pth')
    parser.add_argument('--fingerprints', type=str, default='./IVSF/outputs/full_run/fingerprints.npz')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./IVSF/outputs/adaptive_epoch_real')
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

    fp = np.load(args.fingerprints, allow_pickle=False)
    leaked = fp['selected_images'].astype(np.float32)[:30]
    clean_probs = predict_probs(clean_model, leaked, device)

    rows: List[Dict[str, float]] = []

    # epoch 0: no training yet
    rows.append({'epoch': 0, 'Tampering': 0.0, 'Validation_Leakage': 0.0, 'Validation_NoLeakage': 0.0})

    normal_model = deepcopy(clean_model)
    leaked_model = deepcopy(clean_model)

    for epoch in range(1, 11):
        # regular badnet without leakage
        normal_loader = make_dataloader_for_normal_badnet(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        train_single_epoch(normal_model, normal_loader, device=device, lr=args.lr)
        normal_probs = predict_probs(normal_model, leaked, device)
        normal_rate = ivsf_detection_rate(clean_probs, normal_probs, tau=args.tau)

        # adaptive leakage attack with 30 leaked fingerprints + badnet fine-tuning
        adaptive_loader = make_dataloader_for_leaked_badnet(args.data_root, leaked, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        train_single_epoch(leaked_model, adaptive_loader, device=device, lr=args.lr)
        adaptive_probs = predict_probs(leaked_model, leaked, device)

        tampering_rate = label_only_tampering_rate(clean_probs, adaptive_probs)
        leakage_rate = ivsf_detection_rate(clean_probs, adaptive_probs, tau=args.tau)

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
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {png_path}')


if __name__ == '__main__':
    main()
