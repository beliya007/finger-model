from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ivsf_pipeline.data import cifar10_mean_std, normalize_batch


def _pointwise_mi_from_probs(selected_probs: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    probs = np.clip(selected_probs.astype(np.float64), eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    p_marginal = probs.mean(axis=0, keepdims=True)
    p_marginal = np.clip(p_marginal, eps, 1.0)

    local_mi = np.sum(probs * (np.log(probs) - np.log(p_marginal)), axis=1)
    return float(local_mi.mean()), local_mi.astype(np.float32)


def _predict_probs(
    model: nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    mean, std = cifar10_mean_std(device)

    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    probs_all: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            logits = model(normalize_batch(xb, mean, std))
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs.detach().cpu().numpy())

    return np.concatenate(probs_all, axis=0).astype(np.float32)


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare_fingerprints_on_models(
    fingerprint_npz_path: Path,
    clean_model: nn.Module,
    attacked_model: nn.Module,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(fingerprint_npz_path), allow_pickle=False)
    selected_images = data["selected_images"].astype(np.float32)

    if "selected_probs" in data.files:
        clean_probs = data["selected_probs"].astype(np.float32)
    else:
        clean_probs = _predict_probs(clean_model, selected_images, device)

    attacked_probs = _predict_probs(attacked_model, selected_images, device)

    clean_labels = np.argmax(clean_probs, axis=1).astype(np.int64)
    attacked_labels = np.argmax(attacked_probs, axis=1).astype(np.int64)

    clean_mi_ref, clean_mi = _pointwise_mi_from_probs(clean_probs)
    attacked_mi_ref, attacked_mi = _pointwise_mi_from_probs(attacked_probs)

    base_indices = data["selected_base_sample_indices"].astype(np.int64) if "selected_base_sample_indices" in data.files else np.full(clean_labels.shape[0], -1, dtype=np.int64)
    rank_in_sample = data["selected_rank_in_sample"].astype(np.int64) if "selected_rank_in_sample" in data.files else np.full(clean_labels.shape[0], -1, dtype=np.int64)

    rows: List[Dict[str, object]] = []
    for i in range(clean_labels.shape[0]):
        clean_mi_value = float(clean_mi[i])
        attacked_mi_value = float(attacked_mi[i])
        rows.append(
            {
                "fingerprint_idx": int(i),
                "base_sample_idx": int(base_indices[i]),
                "fingerprint_rank_in_sample": int(rank_in_sample[i]),
                "orig_pred_label": int(clean_labels[i]),
                "orig_mi": clean_mi_value,
                "attacked_pred_label": int(attacked_labels[i]),
                "attacked_mi": attacked_mi_value,
                "abs_mi_diff": abs(clean_mi_value - attacked_mi_value),
            }
        )

    csv_path = output_dir / "fingerprint_attack_comparison.csv"
    _save_csv(csv_path, rows)

    npz_path = output_dir / "fingerprint_attack_comparison.npz"
    np.savez_compressed(
        str(npz_path),
        selected_images=selected_images,
        clean_probs=clean_probs,
        attacked_probs=attacked_probs,
        orig_pred_labels=clean_labels,
        attacked_pred_labels=attacked_labels,
        orig_mi_per_fingerprint=clean_mi,
        attacked_mi_per_fingerprint=attacked_mi,
        orig_mi_ref=np.array([clean_mi_ref], dtype=np.float32),
        attacked_mi_ref=np.array([attacked_mi_ref], dtype=np.float32),
        selected_base_sample_indices=base_indices,
        selected_rank_in_sample=rank_in_sample,
    )

    return {
        "comparison_csv": csv_path,
        "comparison_npz": npz_path,
    }
