from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from lpips import LPIPS as LPIPSModel

from attack.fingerprint_compare import _pointwise_mi_from_probs, _predict_probs
from ivsf_pipeline import IVSFConfig, IVSFFingerprintGenerator
from ivsf_pipeline.losses import js_divergence
from ivsf_pipeline.model import load_resnet20_cifar10


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(data: np.lib.npyio.NpzFile, model: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    images = data["selected_images"].astype(np.float32)
    bases = data["base_images"].astype(np.float32)
    base_indices = data["selected_base_sample_indices"].astype(np.int64)
    fp_probs = _predict_probs(model, images, device)
    base_probs = _predict_probs(model, bases, device)
    lpips_model = LPIPSModel(net="alex", version="0.1").to(device)
    lpips_model.eval()
    js_values: List[float] = []
    lpips_values: List[float] = []
    with torch.no_grad():
        for index in range(images.shape[0]):
            base_i = int(base_indices[index])
            js_values.append(float(js_divergence(
                torch.from_numpy(base_probs[base_i]).to(device).unsqueeze(0),
                torch.from_numpy(fp_probs[index]).to(device).unsqueeze(0),
            ).item()))
            base_tensor = torch.from_numpy(bases[base_i]).to(device).unsqueeze(0) * 2.0 - 1.0
            fp_tensor = torch.from_numpy(images[index]).to(device).unsqueeze(0) * 2.0 - 1.0
            lpips_values.append(float(lpips_model(base_tensor, fp_tensor).item()))
    return {"LPIPS": float(np.mean(lpips_values)), "JS": float(np.mean(js_values))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 ablation results for Table 6-6.")
    parser.add_argument("--output-root", default="./outputs/ablation_table_6_6")
    parser.add_argument("--checkpoint-path", default="./checkpoints/resnet20_cifar10.pth")
    parser.add_argument("--attacked-checkpoint", default="./outputs/attack_run_400ep/attacked_resnet20_cifar10.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-screen-samples", type=int, default=300)
    parser.add_argument("--num-selected-bases", type=int, default=10)
    parser.add_argument("--num-candidates-per-sample", type=int, default=3)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    use_cuda = args.device.lower().startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    clean_model, _ = load_resnet20_cifar10(Path(args.checkpoint_path), device=device)
    attacked_model, _ = load_resnet20_cifar10(Path(args.attacked_checkpoint), device=device)
    output_root = Path(args.output_root)
    variants = {
        "VMIISF": {},
        "No-CS": {"enable_candidate_screening": False},
        "No-Detect": {"lambda_detect": 0.0},
        "No-Stealth": {"lambda_stealth": 0.0},
    }
    rows: List[Dict[str, object]] = []
    for name, overrides in variants.items():
        cfg = IVSFConfig.full(output_dir=str(output_root / name))
        cfg.num_screen_samples = args.num_screen_samples
        cfg.num_selected_bases = args.num_selected_bases
        cfg.num_candidates_per_sample = args.num_candidates_per_sample
        cfg.steps = args.steps
        cfg.base_sample_seed = 1337
        cfg.checkpoint_path = args.checkpoint_path
        cfg.device = args.device
        for key, value in overrides.items():
            setattr(cfg, key, value)
        IVSFFingerprintGenerator(cfg).run()
        fp_path = Path(cfg.output_dir) / "fingerprints.npz"
        data = np.load(fp_path, allow_pickle=False)
        metrics = _metrics(data, clean_model, device)
        images = data["selected_images"].astype(np.float32)
        clean_probs = _predict_probs(clean_model, images, device)
        attacked_probs = _predict_probs(attacked_model, images, device)
        clean_mi = _pointwise_mi_from_probs(clean_probs)[1]
        attacked_mi = _pointwise_mi_from_probs(attacked_probs)[1]
        detected = (np.argmax(attacked_probs, axis=1) != np.argmax(clean_probs, axis=1)) | (np.abs(clean_mi - attacked_mi) > args.tau)
        rows.append({"Method": name, "ACC": round(100.0 * float(detected.mean()), 4), "LPIPS": round(metrics["LPIPS"], 6), "JS": round(metrics["JS"], 6), "num_fingerprints": int(images.shape[0]), "source": str(fp_path)})

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "table_6_6.csv"
    _write_csv(csv_path, rows)
    md_path = output_root / "table_6_6.md"
    md_path.write_text("| Method | ACC | LPIPS | JS |\n|---|---:|---:|---:|\n" + "\n".join(f"| {r['Method']} | {r['ACC']:.4f}% | {r['LPIPS']:.6f} | {r['JS']:.6f} |" for r in rows) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()