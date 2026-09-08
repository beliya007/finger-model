"""
GTSRB 方法对比报告：LPIPS + JS 隐蔽性对比（使用完全训练指纹）

对比方法：VMIISF / MiSentry / SSF / PublicCheck
指纹来源（完全训练规模，与 CIFAR-10 full preset 对齐：300→10→3 = 30 枚指纹）：
  - VMIISF:       ./outputs/gtsrb_full_run/fingerprints.npz
  - MiSentry:     ./outputs/gtsrb_baseline_fingerprints_full/misentry/fingerprints.npz
  - SSF:          ./outputs/gtsrb_baseline_fingerprints_full/ssf/fingerprints.npz
  - PublicCheck:  ./outputs/gtsrb_baseline_fingerprints_full/publiccheck/fingerprints.npz

输出：
  - CSV 表格：./outputs/gtsrb_method_comparison_full/method_lpips_js_comparison.csv
  - MD 表格：./outputs/gtsrb_method_comparison_full/method_lpips_js_comparison.md
  - 纵向对比图：./outputs/gtsrb_method_comparison_full/vertical_fingerprint_comparison_*.png
  - 元信息：./outputs/gtsrb_method_comparison_full/comparison_meta.json

生成完全训练指纹的命令（需先运行）：
  # VMIISF 指纹
  python run_ivsf_fingerprint.py --full --dataset gtsrb \\
      --checkpoint-path ./checkpoints/gtsrb_six_layer_cnn.pth \\
      --output-dir ./outputs/gtsrb_full_run --device cuda

  # 基线指纹（MiSentry / SSF / PublicCheck）
  python run_baseline_fingerprints.py --methods ssf,publiccheck,misentry --dataset gtsrb \\
      --checkpoint-path ./checkpoints/gtsrb_six_layer_cnn.pth \\
      --output-root ./outputs/gtsrb_baseline_fingerprints_full \\
      --num-screen-samples 300 --num-selected-bases 10 \\
      --num-candidates-per-base 3 --steps 220 --seed 1337 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lpips import LPIPS as LPIPSModel

from ivsf_pipeline.data import dataset_mean_std, normalize_batch
from ivsf_pipeline.losses import js_divergence, softmax_probs
from ivsf_pipeline.model import load_model


@dataclass
class MethodData:
    name: str
    path: Path
    selected_images: np.ndarray
    selected_probs: np.ndarray
    selected_base_indices: np.ndarray
    selected_rank_in_sample: np.ndarray
    base_images: np.ndarray
    base_dataset_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GTSRB method comparison: LPIPS + JS using fully-trained fingerprints.",
    )
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/gtsrb_six_layer_cnn.pth")
    parser.add_argument("--device", type=str, default="")

    parser.add_argument(
        "--ivsf-fingerprints",
        type=str,
        default="./outputs/gtsrb_full_run/fingerprints.npz",
    )
    parser.add_argument(
        "--misentry-fingerprints",
        type=str,
        default="./outputs/gtsrb_baseline_fingerprints_full/misentry/fingerprints.npz",
    )
    parser.add_argument(
        "--ssf-fingerprints",
        type=str,
        default="./outputs/gtsrb_baseline_fingerprints_full/ssf/fingerprints.npz",
    )
    parser.add_argument(
        "--publiccheck-fingerprints",
        type=str,
        default="./outputs/gtsrb_baseline_fingerprints_full/publiccheck/fingerprints.npz",
    )

    parser.add_argument("--output-dir", type=str, default="./outputs/gtsrb_method_comparison_full")
    parser.add_argument("--dataset", type=str, default="GTSRB")
    parser.add_argument("--rank", type=int, default=0, help="Fingerprint rank in each base sample for image composition.")
    parser.add_argument(
        "--dataset-indices",
        type=str,
        default="",
        help="Comma-separated dataset indices, one per output image. Defaults to the first two dataset indices shared by all methods.",
    )
    parser.add_argument("--image-with-text", action="store_true", help="If set, draw titles on the vertical image.")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        if device_arg.lower().startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_probs(model: torch.nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    mean, std = dataset_mean_std("gtsrb", device)
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    out: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            logits = model(normalize_batch(x[i : i + batch_size], mean, std))
            out.append(softmax_probs(logits).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _load_method_data(name: str, npz_path: Path, model: torch.nn.Module, device: torch.device) -> MethodData:
    data = np.load(npz_path, allow_pickle=False)
    selected_images = data["selected_images"].astype(np.float32)
    if "selected_probs" in data.files:
        selected_probs = data["selected_probs"].astype(np.float32)
    else:
        selected_probs = _predict_probs(model, selected_images, device)

    return MethodData(
        name=name,
        path=npz_path,
        selected_images=selected_images,
        selected_probs=selected_probs,
        selected_base_indices=data["selected_base_sample_indices"].astype(np.int64),
        selected_rank_in_sample=data["selected_rank_in_sample"].astype(np.int64),
        base_images=data["base_images"].astype(np.float32),
        base_dataset_indices=data["base_dataset_indices"].astype(np.int64),
    )


def _compute_js_lpips_rows(
    methods: List[MethodData],
    dataset_name: str,
    model: torch.nn.Module,
    device: torch.device,
    lpips_model: LPIPSModel,
) -> List[Dict[str, object]]:
    mean, std = dataset_mean_std("gtsrb", device)
    rows: List[Dict[str, object]] = []

    for method in methods:
        base_x = torch.from_numpy(method.base_images).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            base_probs = softmax_probs(model(normalize_batch(base_x, mean, std)))

        js_vals: List[float] = []
        lpips_vals: List[float] = []

        for i in range(method.selected_images.shape[0]):
            base_i = int(method.selected_base_indices[i])
            p_ref = base_probs[base_i : base_i + 1]
            p_fp = torch.from_numpy(method.selected_probs[i : i + 1]).to(device=device, dtype=torch.float32)
            js_vals.append(float(js_divergence(p_ref, p_fp).item()))

            base_img = torch.from_numpy(method.base_images[base_i : base_i + 1]).to(device=device, dtype=torch.float32)
            fp_img = torch.from_numpy(method.selected_images[i : i + 1]).to(device=device, dtype=torch.float32)
            lpips_vals.append(float(lpips_model(base_img * 2.0 - 1.0, fp_img * 2.0 - 1.0).item()))

        rows.append(
            {
                "Dataset": dataset_name,
                "Method": method.name,
                "LPIPS": float(np.mean(lpips_vals)),
                "JS": float(np.mean(js_vals)),
                "num_fingerprints": int(method.selected_images.shape[0]),
                "source_fingerprints": str(method.path).replace("\\", "/"),
            }
        )

    return rows


def _common_dataset_indices(methods: List[MethodData]) -> List[int]:
    common = set(int(x) for x in methods[0].base_dataset_indices.tolist())
    for m in methods[1:]:
        common &= set(int(x) for x in m.base_dataset_indices.tolist())
    return sorted(common)


def _pick_fp_index(method: MethodData, target_dataset_idx: int, rank: int) -> Tuple[int, int, int]:
    for i in range(method.selected_images.shape[0]):
        base_i = int(method.selected_base_indices[i])
        if int(method.base_dataset_indices[base_i]) == target_dataset_idx and int(method.selected_rank_in_sample[i]) == rank:
            return i, base_i, int(method.base_dataset_indices[base_i])

    for i in range(method.selected_images.shape[0]):
        base_i = int(method.selected_base_indices[i])
        if int(method.base_dataset_indices[base_i]) == target_dataset_idx:
            return i, base_i, int(method.base_dataset_indices[base_i])

    i = 0
    base_i = int(method.selected_base_indices[i])
    return i, base_i, int(method.base_dataset_indices[base_i])


def _to_hwc(image_chw: np.ndarray) -> np.ndarray:
    return np.clip(np.transpose(image_chw, (1, 2, 0)), 0.0, 1.0)


def _align_indices_to_ivsf(methods: List[MethodData], target_indices: List[int]) -> List[Dict[str, int]]:
    """For each target dataset index (taken from VMIISF), pick the closest available base
    index per method. Returns a list of {method_name: base_sample_idx_in_method} dicts, one per target.
    """
    ivsf = methods[0]  # VMIISF is methods[0]
    ivsf_indices = set(int(x) for x in ivsf.base_dataset_indices.tolist())
    aligned: List[Dict[str, int]] = []
    used: List[Dict[str, int]] = []

    for target in target_indices:
        if target in ivsf_indices:
            anchor_dataset_idx = target
        else:
            sorted_ivsf = sorted(int(x) for x in ivsf_indices)
            anchor_dataset_idx = min(sorted_ivsf, key=lambda x: abs(x - target))

        record: Dict[str, int] = {"target_dataset_idx": int(anchor_dataset_idx)}
        for method in methods:
            idx_list = method.base_dataset_indices
            exact_matches = [int(i) for i, di in enumerate(idx_list) if int(di) == anchor_dataset_idx]
            if exact_matches:
                record[method.name] = exact_matches[0]
            else:
                # Pick closest by absolute difference.
                closest_i = min(
                    range(len(idx_list)),
                    key=lambda i: abs(int(idx_list[i]) - anchor_dataset_idx),
                )
                record[method.name] = int(closest_i)
        aligned.append(record)
        used.append(record)
    return aligned


def _save_vertical_image(
    methods: List[MethodData],
    target_dataset_idx: int,
    base_idx_per_method: Dict[str, int],
    rank: int,
    output_path: Path,
    draw_text: bool,
) -> Dict[str, object]:
    # Required order: original, VMIISF, MiSentry, SSF, PublicCheck
    by_name = {m.name: m for m in methods}
    ivsf = by_name["VMIISF"]
    misentry = by_name["MiSentry"]
    ssf = by_name["SSF"]
    publiccheck = by_name["PublicCheck"]

    def _select(method: MethodData, base_idx: int):
        for i in range(method.selected_images.shape[0]):
            if int(method.selected_base_indices[i]) == base_idx and int(method.selected_rank_in_sample[i]) == rank:
                return i, int(method.base_dataset_indices[base_idx])
        # Fall back to first fingerprint from that base.
        for i in range(method.selected_images.shape[0]):
            if int(method.selected_base_indices[i]) == base_idx:
                return i, int(method.base_dataset_indices[base_idx])
        return 0, int(method.base_dataset_indices[int(method.selected_base_indices[0])])

    ivsf_base_idx = base_idx_per_method["VMIISF"]
    misentry_base_idx = base_idx_per_method["MiSentry"]
    ssf_base_idx = base_idx_per_method["SSF"]
    publiccheck_base_idx = base_idx_per_method["PublicCheck"]

    ivsf_fp_idx, ivsf_dataset_idx = _select(ivsf, ivsf_base_idx)
    orig_image = ivsf.base_images[ivsf_base_idx]

    mi_fp_idx, mi_dataset_idx = _select(misentry, misentry_base_idx)
    ssf_fp_idx, ssf_dataset_idx = _select(ssf, ssf_base_idx)
    pc_fp_idx, pc_dataset_idx = _select(publiccheck, publiccheck_base_idx)

    images = [
        orig_image,
        ivsf.selected_images[ivsf_fp_idx],
        misentry.selected_images[mi_fp_idx],
        ssf.selected_images[ssf_fp_idx],
        publiccheck.selected_images[pc_fp_idx],
    ]
    labels = ["Original sample", "VMIISF", "MiSentry", "SSF", "PublicCheck"]

    fig, axes = plt.subplots(5, 1, figsize=(4.0, 18.0))
    if not draw_text:
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.02)

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(_to_hwc(img))
        if draw_text:
            ax.set_title(label, fontsize=12)
        ax.axis("off")

    if draw_text:
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
    else:
        fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    return {
        "target_dataset_idx": int(target_dataset_idx),
        "requested_rank": int(rank),
        "selected": {
            "VMIISF": {"dataset_idx": int(ivsf_dataset_idx), "base_sample_idx": int(ivsf_base_idx), "fingerprint_idx": int(ivsf_fp_idx)},
            "MiSentry": {"dataset_idx": int(mi_dataset_idx), "base_sample_idx": int(misentry_base_idx), "fingerprint_idx": int(mi_fp_idx)},
            "SSF": {"dataset_idx": int(ssf_dataset_idx), "base_sample_idx": int(ssf_base_idx), "fingerprint_idx": int(ssf_fp_idx)},
            "PublicCheck": {"dataset_idx": int(pc_dataset_idx), "base_sample_idx": int(publiccheck_base_idx), "fingerprint_idx": int(pc_fp_idx)},
        },
    }


def _save_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Dataset", "Method", "LPIPS", "JS", "num_fingerprints", "source_fingerprints"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_rows_md(rows: List[Dict[str, object]], path: Path) -> None:
    lines = [
        "| Dataset | Method | LPIPS | JS | n | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {} | {} |".format(
                r["Dataset"],
                r["Method"],
                float(r["LPIPS"]),
                float(r["JS"]),
                int(r["num_fingerprints"]),
                r["source_fingerprints"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    model, _ = load_model("gtsrb", Path(args.checkpoint_path), device=device)
    model.eval()
    lpips_model = LPIPSModel(net="alex", version="0.1").to(device)
    lpips_model.eval()

    methods = [
        _load_method_data("VMIISF", Path(args.ivsf_fingerprints), model, device),
        _load_method_data("MiSentry", Path(args.misentry_fingerprints), model, device),
        _load_method_data("SSF", Path(args.ssf_fingerprints), model, device),
        _load_method_data("PublicCheck", Path(args.publiccheck_fingerprints), model, device),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _compute_js_lpips_rows(methods, args.dataset, model, device, lpips_model)
    csv_path = output_dir / "method_lpips_js_comparison.csv"
    md_path = output_dir / "method_lpips_js_comparison.md"
    _save_rows_csv(rows, csv_path)
    _save_rows_md(rows, md_path)

    ivsf_method = methods[0]
    ivsf_indices = sorted(int(x) for x in ivsf_method.base_dataset_indices.tolist())

    if args.dataset_indices.strip():
        requested = [int(x) for x in args.dataset_indices.split(",") if x.strip()]
    else:
        requested = ivsf_indices[:2]
    if len(requested) < 2:
        raise ValueError("Need at least two dataset indices to produce two comparison images.")

    aligned = _align_indices_to_ivsf(methods, requested)

    all_meta = []
    for order, (target_dataset_idx, base_idx_per_method) in enumerate(zip(requested, aligned), start=1):
        suffix = "with_text" if args.image_with_text else "no_text"
        vertical_img_path = output_dir / f"vertical_fingerprint_comparison_{order}_{suffix}.png"
        selection_meta = _save_vertical_image(
            methods=methods,
            target_dataset_idx=target_dataset_idx,
            base_idx_per_method=base_idx_per_method,
            rank=args.rank,
            output_path=vertical_img_path,
            draw_text=args.image_with_text,
        )
        all_meta.append({"image": str(vertical_img_path).replace("\\", "/"), **selection_meta})
        print(f"vertical_image[{order}]: {vertical_img_path}")

    meta = {
        "table_csv": str(csv_path).replace("\\", "/"),
        "table_md": str(md_path).replace("\\", "/"),
        "ivsf_dataset_indices_used": ivsf_indices,
        "requested_dataset_indices": requested,
        "images": all_meta,
    }
    meta_path = output_dir / "comparison_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("GTSRB fully-trained method comparison report finished.")
    print(f"table_csv: {csv_path}")
    print(f"table_md: {md_path}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
