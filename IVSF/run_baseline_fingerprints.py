from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ivsf_pipeline.data import build_cifar10_subset_loader, cifar10_mean_std, normalize_batch
from ivsf_pipeline.losses import js_divergence, softmax_probs
from ivsf_pipeline.model import load_resnet20_cifar10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline fingerprints for SSF/PublicCheck/MiSentry.")
    parser.add_argument("--methods", type=str, default="ssf,publiccheck,misentry", help="Comma-separated methods: ssf,publiccheck,misentry")
    parser.add_argument("--output-root", type=str, default="./outputs/baseline_fingerprints", help="Root output directory")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resnet20_cifar10.pth", help="Path to clean ResNet20 checkpoint")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default="./data", help="CIFAR-10 root")
    parser.add_argument("--num-screen-samples", type=int, default=60, help="How many train samples to screen")
    parser.add_argument("--num-selected-bases", type=int, default=10, help="How many base samples to keep")
    parser.add_argument("--num-candidates-per-base", type=int, default=3, help="Fingerprints per base sample")
    parser.add_argument("--steps", type=int, default=80, help="Optimization steps per fingerprint")
    parser.add_argument("--lr", type=float, default=0.01, help="Optimizer LR")
    parser.add_argument("--l2-radius", type=float, default=3.0, help="L2 radius bound in pixel space [0,1]")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for final probability prediction")
    parser.add_argument(
        "--base-sample-indices",
        type=str,
        default="",
        help="Optional comma-separated CIFAR-10 train indices to force as base samples (e.g. '3879,6045,...').",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(arg_device: str) -> torch.device:
    if arg_device:
        if arg_device.lower().startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _project_l2(delta: torch.Tensor, radius: float) -> torch.Tensor:
    flat = delta.view(delta.shape[0], -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True)
    scale = torch.clamp(radius / (norm + 1e-12), max=1.0)
    flat = flat * scale
    return flat.view_as(delta)


def _predict_probs(model: torch.nn.Module, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    mean, std = cifar10_mean_std(device)
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    probs_all: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            logits = model(normalize_batch(xb, mean, std))
            probs = softmax_probs(logits)
            probs_all.append(probs.detach().cpu().numpy())

    return np.concatenate(probs_all, axis=0).astype(np.float32)


def _optimize_ssf(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    steps: int,
    lr: float,
    l2_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    delta = torch.randn_like(x) * 0.01
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(steps):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        logits_adv = model(normalize_batch(x_adv, mean, std))
        p_adv = softmax_probs(logits_adv)

        entropy = -(p_adv * torch.clamp(p_adv, 1e-12, 1.0).log()).sum(dim=1).mean()
        ce_keep = F.cross_entropy(logits_adv, y)
        l2 = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).mean()

        # SSF-style objective: increase sensitivity (entropy) under label+L2 constraints.
        loss = -entropy + 1.2 * ce_keep + 0.06 * l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.copy_(_project_l2(delta, l2_radius))

    with torch.no_grad():
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        probs = softmax_probs(model(normalize_batch(x_adv, mean, std)))
    return x_adv[0].detach().cpu().numpy().astype(np.float32), probs[0].detach().cpu().numpy().astype(np.float32)


def _optimize_publiccheck(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    steps: int,
    lr: float,
    l2_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    delta = torch.randn_like(x) * 0.005
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(steps):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        logits_adv = model(normalize_batch(x_adv, mean, std))

        f_y = logits_adv.gather(1, y.view(-1, 1)).squeeze(1)
        mask = torch.ones_like(logits_adv, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), False)
        f_other = logits_adv[mask].view(logits_adv.shape[0], -1).max(dim=1).values
        margin = f_y - f_other

        ce_keep = F.cross_entropy(logits_adv, y)
        l2 = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).mean()

        # PublicCheck-style boundary-near objective.
        loss = margin.mean() + 0.7 * ce_keep + 0.04 * l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.copy_(_project_l2(delta, l2_radius))

    with torch.no_grad():
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        probs = softmax_probs(model(normalize_batch(x_adv, mean, std)))
    return x_adv[0].detach().cpu().numpy().astype(np.float32), probs[0].detach().cpu().numpy().astype(np.float32)


def _optimize_misentry(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    steps: int,
    lr: float,
    l2_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        p_ref = softmax_probs(model(normalize_batch(x, mean, std)))

    delta = torch.randn_like(x) * 0.008
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(steps):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        logits_adv = model(normalize_batch(x_adv, mean, std))
        p_adv = softmax_probs(logits_adv)

        js = js_divergence(p_ref, p_adv).mean()
        ce_keep = F.cross_entropy(logits_adv, y)
        l2 = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).mean()

        # MiSentry-style compromise: distribution shift with label retention.
        loss = -js + 1.0 * ce_keep + 0.05 * l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.copy_(_project_l2(delta, l2_radius))

    with torch.no_grad():
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        probs = softmax_probs(model(normalize_batch(x_adv, mean, std)))
    return x_adv[0].detach().cpu().numpy().astype(np.float32), probs[0].detach().cpu().numpy().astype(np.float32)


def _save_method_outputs(
    output_dir: Path,
    method: str,
    selected_images: List[np.ndarray],
    selected_probs: List[np.ndarray],
    selected_pred_labels: List[int],
    selected_base_indices: List[int],
    selected_rank_in_sample: List[int],
    base_images: List[np.ndarray],
    base_labels: List[int],
    base_dataset_indices: List[int],
    config_dump: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_images_arr = np.stack(selected_images, axis=0).astype(np.float32)
    selected_probs_arr = np.stack(selected_probs, axis=0).astype(np.float32)

    payload = {
        "selected_images": selected_images_arr,
        "selected_probs": selected_probs_arr,
        "selected_pred_labels": np.array(selected_pred_labels, dtype=np.int64),
        "selected_base_sample_indices": np.array(selected_base_indices, dtype=np.int64),
        "selected_rank_in_sample": np.array(selected_rank_in_sample, dtype=np.int64),
        "base_images": np.stack(base_images, axis=0).astype(np.float32),
        "base_labels": np.array(base_labels, dtype=np.int64),
        "base_dataset_indices": np.array(base_dataset_indices, dtype=np.int64),
    }

    npz_path = output_dir / "fingerprints.npz"
    np.savez_compressed(npz_path, **payload)

    # BIVBench online-learning script expects fingerprints.npy directly.
    npy_path = output_dir / "fingerprints.npy"
    np.save(npy_path, selected_images_arr)

    meta = {
        "method": method,
        "num_fingerprints": int(selected_images_arr.shape[0]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "npz": str(npz_path),
            "npy": str(npy_path),
        },
        "config": config_dump,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _to_hwc(image_chw: np.ndarray) -> np.ndarray:
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    return np.clip(image_hwc, 0.0, 1.0)


def _save_fingerprint_images(
    output_dir: Path,
    base_images: List[np.ndarray],
    selected_images: List[np.ndarray],
    selected_base_indices: List[int],
    selected_rank_in_sample: List[int],
    max_show: int = 24,
) -> None:
    vis_dir = output_dir / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    total = min(len(selected_images), max_show)
    if total <= 0:
        return

    cols = 4
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    if rows == 1:
        axes = np.array([axes])

    axes_flat = axes.reshape(-1)
    for i in range(total):
        ax = axes_flat[i]
        fp = selected_images[i]
        base_i = selected_base_indices[i]
        rank_i = selected_rank_in_sample[i]
        ax.imshow(_to_hwc(fp))
        ax.set_title(f"fp#{i} base={base_i} rank={rank_i}", fontsize=10)
        ax.axis("off")

    for j in range(total, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(vis_dir / "fingerprints_grid.png", dpi=180)
    plt.close(fig)

    # Save per-base comparison strips: base + all fingerprints from that base.
    unique_bases = sorted(set(selected_base_indices))
    for base_i in unique_bases:
        fp_indices = [k for k, b in enumerate(selected_base_indices) if b == base_i]
        if not fp_indices:
            continue

        n = 1 + len(fp_indices)
        fig2, axes2 = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
        if n == 1:
            axes2 = np.array([axes2])

        axes2[0].imshow(_to_hwc(base_images[base_i]))
        axes2[0].set_title(f"base#{base_i}", fontsize=10)
        axes2[0].axis("off")

        for col, fp_idx in enumerate(fp_indices, start=1):
            axes2[col].imshow(_to_hwc(selected_images[fp_idx]))
            axes2[col].set_title(f"fp#{fp_idx}", fontsize=10)
            axes2[col].axis("off")

        fig2.tight_layout()
        fig2.savefig(vis_dir / f"base_{base_i}_with_fingerprints.png", dpi=180)
        plt.close(fig2)


def _screen_base_samples(
    model: torch.nn.Module,
    loader,
    sampled_indices: List[int],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    n_keep: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            dataset_idx = int(sampled_indices[i])
            x = x.to(device)
            y = y.to(device)

            logits = model(normalize_batch(x, mean, std))
            probs = softmax_probs(logits)
            pred = torch.argmax(probs, dim=1)
            conf = torch.max(probs, dim=1).values
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]

            # Keep correctly classified and near-boundary samples first.
            records.append(
                {
                    "dataset_idx": dataset_idx,
                    "x": x,
                    "y": y,
                    "pred_ok": bool((pred == y).item()),
                    "margin": float(margin.item()),
                    "confidence": float(conf.item()),
                }
            )

    correct = [r for r in records if r["pred_ok"]]
    correct.sort(key=lambda r: (r["margin"], r["confidence"]))
    chosen = correct[: min(n_keep, len(correct))]

    if len(chosen) < n_keep:
        remaining = [r for r in records if r not in chosen]
        remaining.sort(key=lambda r: (r["margin"], r["confidence"]))
        chosen.extend(remaining[: n_keep - len(chosen)])
    return chosen


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    valid = {"ssf", "publiccheck", "misentry"}
    invalid = [m for m in methods if m not in valid]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Valid: ssf, publiccheck, misentry")

    device = _resolve_device(args.device)
    model, resolved_ckpt = load_resnet20_cifar10(Path(args.checkpoint_path), device=device)
    mean, std = cifar10_mean_std(device)

    fixed_indices = None
    if args.base_sample_indices.strip():
        fixed_indices = [int(x.strip()) for x in args.base_sample_indices.split(",") if x.strip()]

    loader, sampled_indices = build_cifar10_subset_loader(
        data_root=args.data_root,
        num_samples=args.num_screen_samples,
        num_workers=2,
        random_subset=(fixed_indices is None),
        subset_seed=args.seed,
        fixed_indices=fixed_indices,
    )

    chosen = _screen_base_samples(
        model=model,
        loader=loader,
        sampled_indices=sampled_indices,
        device=device,
        mean=mean,
        std=std,
        n_keep=args.num_selected_bases,
    )
    if not chosen:
        raise RuntimeError("No base samples available.")

    optimizers = {
        "ssf": _optimize_ssf,
        "publiccheck": _optimize_publiccheck,
        "misentry": _optimize_misentry,
    }

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for method in methods:
        selected_images: List[np.ndarray] = []
        selected_probs: List[np.ndarray] = []
        selected_pred_labels: List[int] = []
        selected_base_indices: List[int] = []
        selected_rank_in_sample: List[int] = []

        base_images: List[np.ndarray] = []
        base_labels: List[int] = []
        base_dataset_indices: List[int] = []

        optimize_fn = optimizers[method]

        for base_i, rec in enumerate(chosen):
            x = rec["x"]
            y = rec["y"]
            base_images.append(x[0].detach().cpu().numpy().astype(np.float32))
            base_labels.append(int(y.item()))
            base_dataset_indices.append(int(rec["dataset_idx"]))

            for rank_i in range(args.num_candidates_per_base):
                img_np, prob_np = optimize_fn(
                    model=model,
                    x=x,
                    y=y,
                    mean=mean,
                    std=std,
                    steps=args.steps,
                    lr=args.lr,
                    l2_radius=args.l2_radius,
                )
                selected_images.append(img_np)
                selected_probs.append(prob_np)
                selected_pred_labels.append(int(np.argmax(prob_np)))
                selected_base_indices.append(base_i)
                selected_rank_in_sample.append(rank_i)

        out_dir = output_root / method
        _save_method_outputs(
            output_dir=out_dir,
            method=method,
            selected_images=selected_images,
            selected_probs=selected_probs,
            selected_pred_labels=selected_pred_labels,
            selected_base_indices=selected_base_indices,
            selected_rank_in_sample=selected_rank_in_sample,
            base_images=base_images,
            base_labels=base_labels,
            base_dataset_indices=base_dataset_indices,
            config_dump={
                "methods": methods,
                "checkpoint_path": str(resolved_ckpt),
                "device": str(device),
                "num_screen_samples": args.num_screen_samples,
                "num_selected_bases": args.num_selected_bases,
                "num_candidates_per_base": args.num_candidates_per_base,
                "steps": args.steps,
                "lr": args.lr,
                "l2_radius": args.l2_radius,
                "seed": args.seed,
                "base_sample_indices": fixed_indices,
            },
        )
        _save_fingerprint_images(
            output_dir=out_dir,
            base_images=base_images,
            selected_images=selected_images,
            selected_base_indices=selected_base_indices,
            selected_rank_in_sample=selected_rank_in_sample,
        )

        print(f"[{method}] fingerprints saved to: {out_dir}")

    print("Baseline fingerprint generation finished.")


if __name__ == "__main__":
    main()
