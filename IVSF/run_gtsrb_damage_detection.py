from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from attack.badnet_attack import (
    AttackConfig,
    get_default_attacked_checkpoint_path,
    load_attacked_model_checkpoint,
    train_attack,
)
from ivsf_pipeline.data import dataset_mean_std, normalize_batch
from ivsf_pipeline.model import load_model


METHODS = ["misentry", "ssf", "publiccheck", "ivsf"]
ATTACKS = [
    ("BadNet", "badnet"),
    ("TrojanNN", "trojan"),
    ("CleanLabel", "clean_label_backdoor"),
    ("Attack4 (Pruning)", "weight_pruning"),
    ("Attack5 (Fine-tuning)", "ftll"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 6-2 GTSRB model-damage detection experiments.")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/gtsrb_six_layer_cnn.pth")
    parser.add_argument("--fingerprint-root", type=Path, default=Path("./outputs/gtsrb_table_6_4"))
    parser.add_argument("--methods", type=str, default="ivsf", help="Comma-separated methods to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/gtsrb_damage_detection"))
    parser.add_argument(
        "--attack-checkpoint-root",
        type=Path,
        default=Path("./outputs/gtsrb_attack_models_400"),
        help="Directory holding reusable attack-model checkpoints.",
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ftll-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ftll-lr", type=float, default=1e-5)
    parser.add_argument("--poison-ratio", type=float, default=0.1)
    parser.add_argument("--trojan-steps", type=int, default=200)
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0003907681,
        help="Local MI deviation threshold for GTSRB detection (set to the measured clean-model noise ceiling from run_gtsrb_clean_repeat_check.py).",
    )
    parser.add_argument("--retrain", action="store_true", help="Retrain and overwrite existing attack checkpoints.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def pointwise_mi(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probabilities = np.clip(probabilities.astype(np.float64), eps, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    marginal = np.clip(probabilities.mean(axis=0, keepdims=True), eps, 1.0)
    return np.sum(probabilities * (np.log(probabilities) - np.log(marginal)), axis=1).astype(np.float32)


def predict_probabilities(model: torch.nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    mean, std = dataset_mean_std("gtsrb", device)
    tensor = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    predictions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensor.shape[0], 256):
            logits = model(normalize_batch(tensor[start : start + 256], mean, std))
            predictions.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def detection_rates(
    clean_probs: np.ndarray,
    attacked_probs: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    invalid_output = ~np.isfinite(attacked_probs).all(axis=1)
    safe_attacked_probs = np.nan_to_num(attacked_probs, nan=1.0 / attacked_probs.shape[1], posinf=1.0, neginf=0.0)
    safe_attacked_probs /= safe_attacked_probs.sum(axis=1, keepdims=True)
    changed_label = clean_probs.argmax(axis=1) != safe_attacked_probs.argmax(axis=1)
    attacked_mi_diff = np.abs(pointwise_mi(clean_probs) - pointwise_mi(safe_attacked_probs))
    changed_mi = attacked_mi_diff > tau
    detected = invalid_output | changed_label | changed_mi
    total = int(detected.size)
    hits = int(detected.sum())
    ns1 = 100.0 * hits / max(total, 1)
    miss_ns5 = 0.0 if total - hits < 5 else math.comb(total - hits, 5) / math.comb(total, 5)
    return {
        "ns1": ns1,
        "ns5": 100.0 * (1.0 - miss_ns5),
        "hits": hits,
        "total": total,
        "mi_threshold": float(tau),
    }


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    methods = [("MiSentry", "misentry"), ("SSF", "ssf"), ("PublicCheck", "publiccheck"), ("VMIISF", "ivsf")]
    lines = [
        "# 表6-2 在6-layer CNN（GTSRB）上模型损坏的检测准确率（%）",
        "",
        f"检测规则与 ResNet20 实验一致：对每种方法的 15 枚指纹，依据输出概率边缘分布计算局部互信息。单枚指纹的预测标签改变，或其局部 MI 相对干净模型的绝对偏差超过固定阈值 $\\tau={args.tau}$，即为检测成功。NS=1 为单枚检测成功率；NS=5 为随机无放回抽取 5 枚时至少检测一次成功的概率。",
        "",
        "| Attack Types / NS | MiSentry 1 | MiSentry 5 | SSF 1 | SSF 5 | PublicCheck 1 | PublicCheck 5 | VMIISF 1 | VMIISF 5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for attack_name, _ in ATTACKS:
        subset = {str(row["method"]): row for row in rows if str(row["attack_name"]) == attack_name}
        values = []
        for _, method in methods:
            if method in subset:
                values.extend([f"{float(subset[method]['ns1']):.1f}", f"{float(subset[method]['ns5']):.1f}"])
            else:
                values.extend(["-", "-"])
        lines.append(f"| {attack_name} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            f"攻击训练配置：BadNet、TrojanNN 和 CleanLabel 均训练 {args.epochs} 轮；Attack4 对模型可训练参数执行 15% 幅度剪枝；Attack5 仅微调最后分类层 {args.ftll_epochs} 轮，学习率为 {args.ftll_lr:.0e}。每种方法均使用 15 枚 GTSRB 指纹。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.tau < 0.0:
        raise ValueError("--tau must be non-negative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() and args.device.lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    methods = [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"--methods must be a non-empty subset of: {', '.join(METHODS)}")

    references: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        fingerprint_path = args.fingerprint_root / method / "fingerprints.npz"
        if method == "ivsf" and not fingerprint_path.is_file():
            fingerprint_path = args.fingerprint_root / "fingerprints.npz"
        data = np.load(fingerprint_path, allow_pickle=False)
        images = data["selected_images"].astype(np.float32)
        clean_model, _ = load_model("gtsrb", Path(args.checkpoint_path), device)
        clean_probs = predict_probabilities(clean_model, images, device)
        references[method] = {
            "images": images,
            "probs": clean_probs,
        }

    rows: List[Dict[str, object]] = []
    for attack_name, attack_type in ATTACKS:
        print(f"[attack] {attack_name}")
        attack_dir = args.attack_checkpoint_root / attack_type
        checkpoint_path = get_default_attacked_checkpoint_path(str(attack_dir), attack_type, "gtsrb")
        if checkpoint_path.is_file() and not args.retrain:
            loaded = load_attacked_model_checkpoint(checkpoint_path, device, "gtsrb")
            result = {
                "attacked_model": loaded["attacked_model"],
                "attacked_checkpoint": loaded["attacked_checkpoint"],
            }
            print(f"  reusing checkpoint: {checkpoint_path}")
        else:
            result = train_attack(
                AttackConfig(
                    dataset="gtsrb",
                    clean_checkpoint_path=args.checkpoint_path,
                    data_root=args.data_root,
                    output_dir=str(attack_dir),
                    device=args.device,
                    seed=args.seed,
                    attack_type=attack_type,
                    poison_ratio=args.poison_ratio,
                    trojan_steps=args.trojan_steps,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    ftll_epochs=args.ftll_epochs,
                    ftll_lr=args.ftll_lr,
                    num_workers=args.num_workers,
                )
            )
        attacked_model = result["attacked_model"]
        for method, reference in references.items():
            rates = detection_rates(
                reference["probs"],
                predict_probabilities(attacked_model, reference["images"], device),
                args.tau,
            )
            rows.append(
                {
                    "attack_name": attack_name,
                    "attack_type": attack_type,
                    "method": method,
                    "ns1": rates["ns1"],
                    "ns5": rates["ns5"],
                    "detected_fingerprints": rates["hits"],
                    "total_fingerprints": rates["total"],
                    "mi_threshold": rates["mi_threshold"],
                    "attack_checkpoint": str(result["attacked_checkpoint"]),
                }
            )
            print(f"  {method}: NS=1 {rates['ns1']:.1f}%, NS=5 {rates['ns5']:.1f}%")

    csv_path = args.output_dir / "gtsrb_damage_detection_results.csv"
    markdown_path = args.output_dir / "table_6_2_gtsrb_damage_detection.md"
    save_csv(csv_path, rows)
    save_markdown(markdown_path, rows, args)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()