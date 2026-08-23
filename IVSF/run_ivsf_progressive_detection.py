from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from attack.data import BackdoorDataConfig, create_backdoor_data_loaders, create_clean_data_loaders
from ivsf_pipeline.data import cifar10_mean_std, normalize_batch
from ivsf_pipeline.model import load_resnet20_cifar10


@dataclass
class FingerprintReference:
    images: np.ndarray
    clean_probs: np.ndarray
    clean_labels: np.ndarray
    clean_mi: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IVSF progressive detection curve generator")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resnet20_cifar10.pth")
    parser.add_argument("--fingerprints", type=str, default="./outputs/full_run/fingerprints.npz")
    parser.add_argument("--output-dir", type=str, default="./outputs/progressive_detection")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    parser.add_argument(
        "--attacks",
        type=str,
        default="badnet,trojan,ftll",
        help="Comma-separated subset of: badnet,trojan,ftll",
    )
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.05, help="MI deviation threshold")

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

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    use_cuda = torch.cuda.is_available() and str(device_arg).lower().startswith("cuda")
    return torch.device("cuda" if use_cuda else "cpu")


def _pointwise_mi_from_probs(selected_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(selected_probs.astype(np.float64), eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    p_marginal = probs.mean(axis=0, keepdims=True)
    p_marginal = np.clip(p_marginal, eps, 1.0)
    local_mi = np.sum(probs * (np.log(probs) - np.log(p_marginal)), axis=1)
    return local_mi.astype(np.float32)


def _predict_probs(model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    mean, std = cifar10_mean_std(device)
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    outputs: List[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            logits = model(normalize_batch(x[i : i + batch_size], mean, std))
            probs = torch.softmax(logits, dim=1)
            outputs.append(probs.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)


def _load_fingerprint_reference(path: Path, clean_model: nn.Module, device: torch.device) -> FingerprintReference:
    data = np.load(str(path), allow_pickle=False)
    images = data["selected_images"].astype(np.float32)
    if "selected_probs" in data.files:
        clean_probs = data["selected_probs"].astype(np.float32)
    else:
        clean_probs = _predict_probs(clean_model, images, device)

    clean_labels = np.argmax(clean_probs, axis=1).astype(np.int64)
    clean_mi = _pointwise_mi_from_probs(clean_probs)
    return FingerprintReference(images=images, clean_probs=clean_probs, clean_labels=clean_labels, clean_mi=clean_mi)


def _compute_detection_metrics(
    attacked_probs: np.ndarray,
    ref: FingerprintReference,
    tau: float,
    ns: int,
) -> Dict[str, float]:
    attacked_labels = np.argmax(attacked_probs, axis=1).astype(np.int64)
    attacked_mi = _pointwise_mi_from_probs(attacked_probs)
    abs_mi_diff = np.abs(ref.clean_mi - attacked_mi)

    detected = (attacked_labels != ref.clean_labels) | (abs_mi_diff > float(tau))
    n = int(detected.shape[0])
    k = int(detected.sum())

    ns1 = 100.0 * float(k) / float(max(n, 1))

    if ns <= 1:
        ns_detect = ns1
    else:
        miss = 0.0 if (n - k) < ns else (math.comb(n - k, ns) / math.comb(n, ns))
        ns_detect = 100.0 * (1.0 - miss)

    return {
        "detected_single": float(k),
        "n_fingerprints": float(n),
        "ns1_rate": float(ns1),
        f"ns{ns}_rate": float(ns_detect),
    }


def _run_one_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(normalize_batch(x, mean, std))
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        steps += 1

    return running_loss / float(max(steps, 1))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    attacks = [x.strip().lower() for x in args.attacks.split(",") if x.strip()]
    supported = {"badnet", "trojan", "ftll"}
    unknown = [x for x in attacks if x not in supported]
    if unknown:
        raise ValueError(f"Unsupported attacks: {unknown}. Supported: {sorted(supported)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    use_amp = bool((not args.no_amp) and device.type == "cuda")

    # Build reference from clean model once.
    clean_model, _ = load_resnet20_cifar10(Path(args.checkpoint_path), device=device)
    clean_model.eval()
    ref = _load_fingerprint_reference(Path(args.fingerprints), clean_model, device)

    eval_epochs = set(range(0, int(args.max_epochs) + 1, int(args.eval_interval)))
    if int(args.max_epochs) not in eval_epochs:
        eval_epochs.add(int(args.max_epochs))

    combined_rows: List[Dict[str, object]] = []

    for attack_type in attacks:
        print(f"[progressive] start attack={attack_type}")
        model, _ = load_resnet20_cifar10(Path(args.checkpoint_path), device=device)
        model.train()

        data_cfg = BackdoorDataConfig(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            attack_type=attack_type,
            poison_ratio=args.poison_ratio,
            trigger_label=args.trigger_label,
            trigger_size=args.trigger_size,
            trigger_alpha=args.trigger_alpha,
            trojan_steps=args.trojan_steps,
            trojan_lr=args.trojan_lr,
            trojan_batch_size=args.trojan_batch_size,
        )

        if attack_type in {"badnet", "trojan"}:
            loaders = create_backdoor_data_loaders(data_cfg, device=device, model_for_poison=model)
            train_loader = loaders["train"]
            optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
        else:
            loaders = create_clean_data_loaders(data_cfg, device=device)
            train_loader = loaders["train"]
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=float(args.ftll_lr))

        mean, std = cifar10_mean_std(device)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        rows: List[Dict[str, object]] = []

        # epoch 0 (clean checkpoint before attack optimization)
        probs = _predict_probs(model, ref.images, device)
        m = _compute_detection_metrics(probs, ref, tau=float(args.tau), ns=5)
        row0 = {
            "attack_type": attack_type,
            "epoch": 0,
            "train_loss": 0.0,
            "detected_single": int(m["detected_single"]),
            "n_fingerprints": int(m["n_fingerprints"]),
            "ivsf_ns1": float(m["ns1_rate"]),
            "ivsf_ns5": float(m["ns5_rate"]),
            "tau": float(args.tau),
        }
        rows.append(row0)
        combined_rows.append(row0)

        for epoch in range(1, int(args.max_epochs) + 1):
            train_loss = _run_one_train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                mean=mean,
                std=std,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )

            if epoch in eval_epochs:
                probs = _predict_probs(model, ref.images, device)
                m = _compute_detection_metrics(probs, ref, tau=float(args.tau), ns=5)
                row = {
                    "attack_type": attack_type,
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "detected_single": int(m["detected_single"]),
                    "n_fingerprints": int(m["n_fingerprints"]),
                    "ivsf_ns1": float(m["ns1_rate"]),
                    "ivsf_ns5": float(m["ns5_rate"]),
                    "tau": float(args.tau),
                }
                rows.append(row)
                combined_rows.append(row)
                print(
                    f"[progressive] attack={attack_type} epoch={epoch} "
                    f"ns1={row['ivsf_ns1']:.2f} ns5={row['ivsf_ns5']:.2f}"
                )

        attack_csv = output_dir / f"ivsf_progressive_{attack_type}.csv"
        _write_csv(attack_csv, rows)
        print(f"[progressive] saved {attack_csv}")

    combined_csv = output_dir / "ivsf_progressive_detection_all.csv"
    _write_csv(combined_csv, combined_rows)
    print(f"[progressive] saved {combined_csv}")


if __name__ == "__main__":
    main()
