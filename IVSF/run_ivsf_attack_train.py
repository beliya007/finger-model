from __future__ import annotations

import argparse

from attack.badnet_attack import AttackConfig, train_badnet_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entry-2: train attacked model only")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resnet20_cifar10.pth", help="Path to clean ResNet20 checkpoint")
    parser.add_argument("--output-dir", type=str, default="./outputs/attack_run", help="Output directory for attacked checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--poison-ratio", type=float, default=0.1)
    parser.add_argument("--trigger-label", type=int, default=0)
    parser.add_argument("--trigger-size", type=int, default=4)
    parser.add_argument("--trigger-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = AttackConfig(
        clean_checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        trigger_label=args.trigger_label,
        poison_ratio=args.poison_ratio,
        trigger_size=args.trigger_size,
        trigger_alpha=args.trigger_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        use_amp=(not args.no_amp),
    )

    result = train_badnet_attack(cfg)
    print("IVSF attack model training finished.")
    print(f"attacked_checkpoint: {result['attacked_checkpoint']}")
    print(f"clean_checkpoint: {result['clean_checkpoint']}")
    print(f"device: {result['device']}")


if __name__ == "__main__":
    main()
