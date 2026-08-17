from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from attack.badnet_attack import AttackConfig, train_attack


DEFAULT_ATTACKS = [
    "badnet",
    "trojan",
    "clean_label_backdoor",
    "weight_pruning",
    "ftll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-train all IVSF attacks")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resnet20_cifar10.pth")
    parser.add_argument("--output-root", type=str, default="./outputs/attack_runs_400ep")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    parser.add_argument(
        "--attacks",
        type=str,
        default=",".join(DEFAULT_ATTACKS),
        help="Comma-separated attack list. Supported: badnet,trojan,clean_label_backdoor,weight_pruning,ftll",
    )

    parser.add_argument("--epochs", type=int, default=400, help="Epochs for badnet/trojan/clean_label_backdoor")
    parser.add_argument("--ftll-epochs", type=int, default=400, help="Epochs for FTLL")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ftll-lr", type=float, default=1e-5)

    parser.add_argument("--poison-ratio", type=float, default=0.1)
    parser.add_argument("--trigger-label", type=int, default=0)
    parser.add_argument("--trigger-size", type=int, default=4)
    parser.add_argument("--trigger-alpha", type=float, default=1.0)

    parser.add_argument("--trojan-steps", type=int, default=200)
    parser.add_argument("--trojan-lr", type=float, default=0.1)
    parser.add_argument("--trojan-batch-size", type=int, default=64)

    parser.add_argument("--clean-label-base-label", type=int, default=1)
    parser.add_argument("--clean-label-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--clean-label-alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--clean-label-steps", type=int, default=5)
    parser.add_argument("--clean-label-max-poison", type=int, default=256)

    parser.add_argument("--pruning-ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision")
    return parser.parse_args()


def _parse_attack_list(raw_value: str) -> List[str]:
    attacks = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not attacks:
        raise ValueError("--attacks is empty.")

    unsupported = [item for item in attacks if item not in DEFAULT_ATTACKS]
    if unsupported:
        raise ValueError(f"Unsupported attack(s): {unsupported}. Supported: {DEFAULT_ATTACKS}")
    return attacks


def _build_attack_config(args: argparse.Namespace, attack_type: str, output_root: Path) -> AttackConfig:
    return AttackConfig(
        clean_checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        output_dir=str(output_root / attack_type),
        device=args.device,
        seed=args.seed,
        attack_type=attack_type,
        trigger_label=args.trigger_label,
        poison_ratio=args.poison_ratio,
        trigger_size=args.trigger_size,
        trigger_alpha=args.trigger_alpha,
        trojan_steps=args.trojan_steps,
        trojan_lr=args.trojan_lr,
        trojan_batch_size=args.trojan_batch_size,
        clean_label_base_label=args.clean_label_base_label,
        clean_label_eps=args.clean_label_eps,
        clean_label_alpha=args.clean_label_alpha,
        clean_label_steps=args.clean_label_steps,
        clean_label_max_poison=args.clean_label_max_poison,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pruning_ratio=args.pruning_ratio,
        ftll_epochs=args.ftll_epochs,
        ftll_lr=args.ftll_lr,
        num_workers=args.num_workers,
        use_amp=(not args.no_amp),
    )


def main() -> None:
    args = parse_args()
    attack_types = _parse_attack_list(args.attacks)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root.resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "device": args.device,
        "epochs": int(args.epochs),
        "ftll_epochs": int(args.ftll_epochs),
        "attacks": [],
    }

    total = len(attack_types)
    for index, attack_type in enumerate(attack_types, start=1):
        print(f"[{index}/{total}] training attack: {attack_type}")
        cfg = _build_attack_config(args, attack_type, output_root)
        result = train_attack(cfg)

        item = {
            "attack_type": attack_type,
            "output_dir": cfg.output_dir,
            "attacked_checkpoint": str(result["attacked_checkpoint"]),
            "clean_checkpoint": str(result["clean_checkpoint"]),
            "meta": result.get("meta", {}),
            "config": asdict(cfg),
        }
        summary["attacks"].append(item)

        print(f"[{index}/{total}] finished: {attack_type}")
        print(f"  output_dir: {cfg.output_dir}")
        print(f"  attacked_checkpoint: {result['attacked_checkpoint']}")

    summary_path = output_root / "batch_attack_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Batch attack training finished.")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()