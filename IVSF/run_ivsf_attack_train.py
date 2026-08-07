from __future__ import annotations

import argparse

from attack.badnet_attack import AttackConfig, train_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entry-2: train attacked model only")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resnet20_cifar10.pth", help="Path to clean ResNet20 checkpoint")
    parser.add_argument("--output-dir", type=str, default="./outputs/attack_run", help="Output directory for attacked checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    parser.add_argument(
        "--attack-type",
        type=str,
        default="badnet",
        choices=["badnet", "trojan", "clean_label_backdoor", "weight_pruning", "ftll"],
        help="Attack method to apply",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    parser.add_argument("--pruning-ratio", type=float, default=0.15, help="Weight pruning ratio")
    parser.add_argument("--ftll-epochs", type=int, default=3)
    parser.add_argument("--ftll-lr", type=float, default=1e-5)

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
        attack_type=args.attack_type,
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

    result = train_attack(cfg)
    print("IVSF attack model training finished.")
    print(f"attack_type: {result['attack_type']}")
    print(f"attacked_checkpoint: {result['attacked_checkpoint']}")
    print(f"clean_checkpoint: {result['clean_checkpoint']}")
    print(f"device: {result['device']}")


if __name__ == "__main__":
    main()
