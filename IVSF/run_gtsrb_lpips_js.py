"""纯 numpy/torch 版的 GTSRB LPIPS / JS 复现脚本

不依赖 matplotlib，便于在最小依赖环境下复现表 6-5 的 GTSRB 行。

输出：
- `outputs/gtsrb_method_comparison/method_lpips_js_comparison.csv` (覆盖)
- `outputs/gtsrb_method_comparison/method_lpips_js_comparison.md`
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from lpips import LPIPS as LPIPSModel

from ivsf_pipeline.data import dataset_mean_std, normalize_batch
from ivsf_pipeline.losses import js_divergence, softmax_probs
from ivsf_pipeline.model import load_model


METHODS = [
    ("MiSentry", "misentry"),
    ("SSF", "ssf"),
    ("PublicCheck", "publiccheck"),
    ("VMIISF", "ivsf"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTSRB LPIPS / JS pure numpy computation (no matplotlib).")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/gtsrb_six_layer_cnn.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input-root", type=str, default="./outputs/gtsrb_time_evaluation")
    parser.add_argument("--output-dir", type=str, default="./outputs/gtsrb_method_comparison")
    parser.add_argument("--dataset", type=str, default="GTSRB")
    return parser.parse_args()


def _resolve_device(arg: str) -> torch.device:
    if arg and arg.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_probs(model, images: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    mean, std = dataset_mean_std("gtsrb", device)
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            logits = model(normalize_batch(x[i : i + batch_size], mean, std))
            out.append(softmax_probs(logits).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    model, _ = load_model("gtsrb", Path(args.checkpoint_path), device=device)
    model.eval()
    lpips_model = LPIPSModel(net="alex", version="0.1").to(device)
    lpips_model.eval()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    mean, std = dataset_mean_std("gtsrb", device)

    for display_name, folder in METHODS:
        npz_path = input_root / folder / "fingerprints.npz"
        if not npz_path.is_file():
            print("[跳过] 找不到 %s" % npz_path)
            continue

        data = np.load(npz_path, allow_pickle=False)
        selected_images = data["selected_images"].astype(np.float32)
        if "selected_probs" in data.files:
            selected_probs = data["selected_probs"].astype(np.float32)
        else:
            selected_probs = _predict_probs(model, selected_images, device)

        base_images = data["base_images"].astype(np.float32)
        base_indices = data["selected_base_sample_indices"].astype(np.int64)

        base_x = torch.from_numpy(base_images).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            base_probs = softmax_probs(model(normalize_batch(base_x, mean, std)))

        lpips_vals: List[float] = []
        js_vals: List[float] = []

        for i in range(selected_images.shape[0]):
            base_i = int(base_indices[i])
            p_ref = base_probs[base_i : base_i + 1]
            p_fp = torch.from_numpy(selected_probs[i : i + 1]).to(device=device, dtype=torch.float32)
            js_vals.append(float(js_divergence(p_ref, p_fp).item()))

            base_img = torch.from_numpy(base_images[base_i : base_i + 1]).to(device=device, dtype=torch.float32)
            fp_img = torch.from_numpy(selected_images[i : i + 1]).to(device=device, dtype=torch.float32)
            lpips_vals.append(float(lpips_model(base_img * 2.0 - 1.0, fp_img * 2.0 - 1.0).item()))

        rows.append(
            {
                "Dataset": args.dataset,
                "Method": display_name,
                "LPIPS": float(np.mean(lpips_vals)),
                "JS": float(np.mean(js_vals)),
                "num_fingerprints": int(selected_images.shape[0]),
                "source_fingerprints": str(npz_path).replace("\\", "/"),
            }
        )

    csv_path = output_dir / "method_lpips_js_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Dataset", "Method", "LPIPS", "JS", "num_fingerprints", "source_fingerprints"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "method_lpips_js_comparison.md"
    md_lines = [
        "| Dataset | Method | LPIPS | JS | n | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {} | {} |".format(
                r["Dataset"],
                r["Method"],
                float(r["LPIPS"]),
                float(r["JS"]),
                int(r["num_fingerprints"]),
                r["source_fingerprints"],
            )
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("=" * 80)
    print("  GTSRB 表 6-5 LPIPS / JS 实验结果")
    print("=" * 80)
    print("%-12s %-15s %12s %12s" % ("Dataset", "Method", "LPIPS", "JS"))
    print("-" * 60)
    for r in rows:
        print("%-12s %-15s %12.6f %12.6f" % (r["Dataset"], r["Method"], float(r["LPIPS"]), float(r["JS"])))
    print("=" * 80)
    print("CSV: %s" % csv_path)
    print("Markdown: %s" % md_path)


if __name__ == "__main__":
    main()
