"""对同一批 GTSRB 指纹，分别以「单张逐一输入」和「整批输入」两种方式前向传播干净模型，
记录标签与互信息值。两种方式对应指纹生成阶段（单张）与验证/重跑阶段（整批）的真实计算路径，
即便模型和图像完全相同，卷积/GEMM 在不同 batch size 下的算子实现也会引入浮点误差，
因此两次互信息值通常不完全相等（这与 verification_info.csv 中 mi_input_output vs
mi_input_output_moin 的差异是同一现象）。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ivsf_pipeline.data import dataset_mean_std, normalize_batch
from ivsf_pipeline.model import load_model
from run_gtsrb_damage_detection import pointwise_mi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record label/MI for the same GTSRB fingerprints across two clean-model passes.")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/gtsrb_six_layer_cnn.pth")
    parser.add_argument("--fingerprint-path", type=Path, default=Path("./outputs/gtsrb_table_6_4/fingerprints.npz"))
    parser.add_argument("--output-csv", type=Path, default=Path("./outputs/gtsrb_clean_repeat_check.csv"))
    parser.add_argument("--tau", type=float, default=0.0003907681, help="与 run_gtsrb_damage_detection.py 一致的 MI 检测阈值。")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def predict_probabilities(model: torch.nn.Module, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    mean, std = dataset_mean_std("gtsrb", device)
    tensor = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            logits = model(normalize_batch(tensor[start : start + batch_size], mean, std))
            predictions.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.device.lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    data = np.load(args.fingerprint_path, allow_pickle=False)
    images = data["selected_images"].astype(np.float32)

    clean_model, _ = load_model("gtsrb", Path(args.checkpoint_path), device)

    # run1: 逐张单样本前向（对应生成阶段的计算路径）
    probs_run1 = predict_probabilities(clean_model, images, device, batch_size=1)
    # run2: 整批一次性前向（对应验证/重跑阶段的计算路径）
    probs_run2 = predict_probabilities(clean_model, images, device, batch_size=images.shape[0])

    labels_run1 = probs_run1.argmax(axis=1)
    labels_run2 = probs_run2.argmax(axis=1)
    mi_run1 = pointwise_mi(probs_run1)
    mi_run2 = pointwise_mi(probs_run2)

    rows = []
    for idx in range(images.shape[0]):
        rows.append(
            {
                "fingerprint_index": idx,
                "run1_label": int(labels_run1[idx]),
                "run1_mi": float(mi_run1[idx]),
                "run2_label": int(labels_run2[idx]),
                "run2_mi": float(mi_run2[idx]),
                "label_changed": labels_run1[idx] != labels_run2[idx],
                "mi_abs_diff": float(abs(mi_run1[idx] - mi_run2[idx])),
                "mi_threshold": args.tau,
                "false_positive_under_tau": bool(
                    labels_run1[idx] != labels_run2[idx] or abs(mi_run1[idx] - mi_run2[idx]) > args.tau
                ),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    num_label_changed = sum(1 for row in rows if row["label_changed"])
    max_mi_diff = max(row["mi_abs_diff"] for row in rows)
    num_false_positive = sum(1 for row in rows if row["false_positive_under_tau"])
    print(f"指纹数量: {len(rows)}")
    print(f"两次前向传播标签不一致的数量: {num_label_changed}")
    print(f"两次前向传播互信息绝对差的最大值: {max_mi_diff:.10f}")
    print(f"检测阈值 tau={args.tau}: 因干净模型自身噪声被误判为「已损坏」的指纹数量: {num_false_positive}/{len(rows)}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
