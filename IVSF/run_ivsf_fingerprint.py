from __future__ import annotations

import argparse
import json
from pathlib import Path

from ivsf_pipeline import IVSFConfig, IVSFFingerprintGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IVSF fingerprint generation (ResNet20 + CIFAR-10)")
    parser.add_argument("--quick", action="store_true", help="Use quick preset.")
    parser.add_argument("--config-json", type=str, default="", help="JSON string to override config fields.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory override.")
    parser.add_argument("--model-source", type=str, default="", choices=["checkpoint", "train"], help="Model source.")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Path to checkpoint if model-source=checkpoint.")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = IVSFConfig.quick() if args.quick else IVSFConfig()
    if args.config_json:
        cfg = IVSFConfig.from_overrides(json.loads(args.config_json))

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model_source:
        cfg.model_source = args.model_source
    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    if args.device:
        cfg.device = args.device

    repo_root = Path(__file__).resolve().parents[1]
    generator = IVSFFingerprintGenerator(repo_root=repo_root, config=cfg)
    outputs = generator.run()

    print("IVSF fingerprint generation finished.")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
