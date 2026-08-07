from __future__ import annotations

import argparse
from pathlib import Path

import torch

from attack.badnet_attack import get_default_attacked_checkpoint_path, load_attacked_model_checkpoint
from attack.fingerprint_compare import compare_fingerprints_on_models
from ivsf_pipeline.model import load_resnet20_cifar10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entry-3: compare fingerprints on clean vs attacked model")
    parser.add_argument("--fingerprints", type=str, default="./outputs/quick_run/fingerprints.npz", help="Path to fingerprints.npz from entry-1")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resnet20_cifar10.pth", help="Path to clean ResNet20 checkpoint")
    parser.add_argument("--output-dir", type=str, default="./outputs/attack_run", help="Output directory for comparison files")
    parser.add_argument("--attacked-checkpoint", type=str, default="", help="Path to attacked model checkpoint from entry-2")
    parser.add_argument(
        "--attack-type",
        type=str,
        default="badnet",
        choices=["badnet", "trojan", "clean_label_backdoor", "weight_pruning", "ftll"],
        help="Used only when --attacked-checkpoint is empty",
    )
    parser.add_argument("--device", type=str, default="", help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_cuda = torch.cuda.is_available() and ((args.device == "") or args.device.lower().startswith("cuda"))
    device = torch.device("cuda" if use_cuda else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attacked_ckpt = (
        Path(args.attacked_checkpoint)
        if args.attacked_checkpoint
        else get_default_attacked_checkpoint_path(args.output_dir, args.attack_type)
    )
    loaded = load_attacked_model_checkpoint(attacked_ckpt, device=device)
    attacked_model = loaded["attacked_model"]

    clean_model, _ = load_resnet20_cifar10(Path(args.checkpoint_path), device=device)

    comparison = compare_fingerprints_on_models(
        fingerprint_npz_path=Path(args.fingerprints),
        clean_model=clean_model,
        attacked_model=attacked_model,
        device=device,
        output_dir=output_dir,
    )

    print("IVSF fingerprint comparison finished.")
    print(f"attacked_checkpoint: {loaded['attacked_checkpoint']}")
    print(f"comparison_csv: {comparison['comparison_csv']}")
    print(f"comparison_npz: {comparison['comparison_npz']}")


if __name__ == "__main__":
    main()
