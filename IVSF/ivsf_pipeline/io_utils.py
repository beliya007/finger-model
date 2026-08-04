from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _format_prob_dist(probs: np.ndarray, decimals: int = 6) -> str:
    vals = [f"{float(v):.{decimals}f}" for v in probs.tolist()]
    return "[" + ", ".join(vals) + "]"


def save_prediction_report_md(
    path: Path,
    base_labels: np.ndarray,
    base_probs: np.ndarray,
    base_dataset_indices: np.ndarray,
    selected_probs_grouped: np.ndarray,
    selected_pred_labels_grouped: np.ndarray,
) -> None:
    lines: List[str] = []
    lines.append("# 样本与指纹预测分布报告")
    lines.append("")
    lines.append("本文档按基础样本分组，记录原始样本标签与预测概率分布，以及对应指纹的标签与预测概率分布。")
    lines.append("")

    n_base = int(base_labels.shape[0])
    for i in range(n_base):
        base_label = int(base_labels[i])
        dataset_idx = int(base_dataset_indices[i])
        base_prob = base_probs[i]
        base_pred = int(np.argmax(base_prob))

        lines.append(f"## 基础样本 {i}")
        lines.append("")
        lines.append(f"- dataset_idx: {dataset_idx}")
        lines.append(f"- 原始样本真实标签: {base_label}")
        lines.append(f"- 原始样本预测标签: {base_pred}")
        lines.append(f"- 原始样本预测概率分布: {_format_prob_dist(base_prob)}")
        lines.append("")

        n_fp = int(selected_probs_grouped.shape[1])
        for j in range(n_fp):
            fp_prob = selected_probs_grouped[i, j]
            fp_pred = int(selected_pred_labels_grouped[i, j])
            lines.append(f"### 对应指纹 {j}")
            lines.append("")
            lines.append(f"- 指纹标签(预测): {fp_pred}")
            lines.append(f"- 指纹预测概率分布: {_format_prob_dist(fp_prob)}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def save_meta(path: Path, meta: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_timing_summary(path: Path, timing_summary: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2, ensure_ascii=False)


def save_scores_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def _to_hwc(image_chw: np.ndarray) -> np.ndarray:
    img = np.transpose(image_chw, (1, 2, 0))
    return np.clip(img, 0.0, 1.0)


def save_preview_figure(
    path: Path,
    base_images: np.ndarray,
    selected_images: np.ndarray,
    deltas: np.ndarray,
    scores: np.ndarray,
    max_items: int = 12,
) -> None:
    n = min(max_items, len(base_images))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(10, 2.8 * n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        base = _to_hwc(base_images[i])
        fp = _to_hwc(selected_images[i])
        delta = np.transpose(deltas[i], (1, 2, 0))
        delta_vis = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)

        axes[i, 0].imshow(base)
        axes[i, 0].set_title("base")
        axes[i, 1].imshow(fp)
        axes[i, 1].set_title("fingerprint")
        axes[i, 2].imshow(delta_vis)
        axes[i, 2].set_title("delta")
        axes[i, 3].bar(["score"], [float(scores[i])])
        axes[i, 3].set_ylim(min(0.0, float(scores[i]) - 0.1), max(0.1, float(scores[i]) + 0.1))

        for j in range(4):
            axes[i, j].axis("off" if j < 3 else "on")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_one_base_with_three_fingerprints(
    path: Path,
    base_image: np.ndarray,
    fingerprint_images: np.ndarray,
    fingerprint_scores: np.ndarray,
    dataset_idx: int,
    label: int,
) -> None:
    n_fp = min(3, int(fingerprint_images.shape[0]))
    if n_fp <= 0:
        return

    fig, axes = plt.subplots(1, 1 + n_fp, figsize=(3.2 * (1 + n_fp), 3.4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].imshow(_to_hwc(base_image))
    axes[0].set_title(f"base (idx={dataset_idx}, y={label})")
    axes[0].axis("off")

    for i in range(n_fp):
        axes[i + 1].imshow(_to_hwc(fingerprint_images[i]))
        axes[i + 1].set_title(f"fp{i+1}  score={float(fingerprint_scores[i]):.4f}")
        axes[i + 1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_two_bases_with_three_fingerprints(
    path: Path,
    base_images: np.ndarray,
    base_dataset_indices: np.ndarray,
    base_labels: np.ndarray,
    fingerprint_images_grouped: np.ndarray,
    fingerprint_scores_grouped: np.ndarray,
    fingerprint_pred_labels_grouped: np.ndarray,
) -> None:
    n_rows = min(2, int(base_images.shape[0]))
    if n_rows <= 0:
        return

    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for r in range(n_rows):
        axes[r, 0].imshow(_to_hwc(base_images[r]))
        axes[r, 0].set_title(f"base (idx={int(base_dataset_indices[r])}, y={int(base_labels[r])})")
        axes[r, 0].axis("off")

        n_fp = min(3, int(fingerprint_images_grouped.shape[1]))
        for c in range(n_fp):
            axes[r, c + 1].imshow(_to_hwc(fingerprint_images_grouped[r, c]))
            axes[r, c + 1].set_title(
                f"fp{c+1} y_hat={int(fingerprint_pred_labels_grouped[r, c])}\n"
                f"score={float(fingerprint_scores_grouped[r, c]):.4f}"
            )
            axes[r, c + 1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
