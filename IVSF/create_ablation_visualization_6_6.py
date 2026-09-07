from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_hwc(image: np.ndarray) -> np.ndarray:
    return np.clip(np.transpose(image, (1, 2, 0)), 0.0, 1.0)


def _find_fingerprint(data: np.lib.npyio.NpzFile, dataset_index: int, rank: int) -> np.ndarray:
    base_indices = data["selected_base_sample_indices"].astype(np.int64)
    ranks = data["selected_rank_in_sample"].astype(np.int64)
    base_dataset_indices = data["base_dataset_indices"].astype(np.int64)
    matches = np.where(
        (base_dataset_indices[base_indices] == dataset_index) & (ranks == rank)
    )[0]
    if matches.size != 1:
        raise ValueError(f"Expected one fingerprint for dataset_idx={dataset_index}, rank={rank}; got {matches.size}.")
    return data["selected_images"][int(matches[0])].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Table 6-6 No-Stealth visual comparison.")
    parser.add_argument("--full-dir", default="./outputs/ablation_table_6_6/VMIISF")
    parser.add_argument("--no-stealth-dir", default="./outputs/ablation_table_6_6/No-Stealth")
    parser.add_argument("--output", default="./outputs/ablation_table_6_6/figure_6_6_no_stealth.png")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    full = np.load(Path(args.full_dir) / "fingerprints.npz", allow_pickle=False)
    no_stealth = np.load(Path(args.no_stealth_dir) / "fingerprints.npz", allow_pickle=False)
    dataset_indices = full["base_dataset_indices"].astype(np.int64)[: args.count]
    if not np.array_equal(dataset_indices, no_stealth["base_dataset_indices"].astype(np.int64)[: args.count]):
        raise ValueError("VMIISF and No-Stealth do not use the same base samples.")

    figure, axes = plt.subplots(3, args.count, figsize=(2.3 * args.count, 1.05 * 2.3 * 3), squeeze=False)
    for column, dataset_index in enumerate(dataset_indices):
        base_index = column
        original = full["base_images"][base_index].astype(np.float32)
        normal = _find_fingerprint(full, int(dataset_index), args.rank)
        ablation = _find_fingerprint(no_stealth, int(dataset_index), args.rank)
        for row, image in enumerate((original, normal, ablation)):
            axes[row, column].imshow(_to_hwc(image), interpolation="nearest")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_aspect("equal")
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.06, hspace=0.08)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="white", pad_inches=0)
    plt.close(figure)
    print(output_path)
    print("dataset_indices=" + ",".join(str(int(value)) for value in dataset_indices))


if __name__ == "__main__":
    main()