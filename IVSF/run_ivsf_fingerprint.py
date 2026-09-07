from __future__ import annotations

import argparse
import json

from ivsf_pipeline import IVSFConfig, IVSFFingerprintGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IVSF fingerprint generation")
    parser.add_argument("--quick", action="store_true", help="Use quick preset.")
    parser.add_argument("--full", action="store_true", help="Use full preset (same sample/fingerprint count).")
    parser.add_argument("--time-evaluation", action="store_true", help="Use the CIFAR-10 time-evaluation preset (100 -> 5 -> 15).")
    parser.add_argument("--config-json", type=str, default="", help="JSON string to override config fields.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory override.")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Path to trained ResNet20 checkpoint.")
    parser.add_argument("--dataset", choices=["cifar10", "gtsrb"], default="", help="Dataset/model pair to use.")
    parser.add_argument("--data-root", type=str, default="", help="Dataset root directory.")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if sum((args.quick, args.full, args.time_evaluation)) > 1:
        raise ValueError("Only one preset can be selected: --quick, --full, or --time-evaluation.")

    if args.time_evaluation:
        cfg = IVSFConfig.time_evaluation()
    elif args.full:
        cfg = IVSFConfig.full()
    elif args.quick:
        cfg = IVSFConfig.quick()
    else:
        cfg = IVSFConfig()
    if args.config_json:
        cfg = IVSFConfig.from_overrides(json.loads(args.config_json))

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    if args.dataset:
        cfg.dataset = args.dataset
    if args.data_root:
        cfg.data_root = args.data_root
    if args.device:
        cfg.device = args.device

    generator = IVSFFingerprintGenerator(config=cfg)
    outputs = generator.run()

    print("IVSF fingerprint generation finished.")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
