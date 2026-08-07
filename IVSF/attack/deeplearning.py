from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ivsf_pipeline.data import cifar10_mean_std, normalize_batch


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-3
    trigger_label: int = 0
    use_amp: bool = True


@dataclass
class FTLLConfig:
    epochs: int = 3
    lr: float = 1e-5
    use_amp: bool = True


def _evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_label: Optional[int] = None,
) -> float:
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(normalize_batch(x, mean, std))
            pred = torch.argmax(logits, dim=1)

            if target_label is None:
                correct += int((pred == y).sum().item())
                total += int(y.shape[0])
            else:
                correct += int((pred == int(target_label)).sum().item())
                total += int(pred.shape[0])

    return float(correct) / float(max(total, 1))


def evaluate_clean_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    mean, std = cifar10_mean_std(device)
    return _evaluate_accuracy(model, loader, device, mean, std, target_label=None)


def backdoor_model_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    test_clean_loader: DataLoader,
    test_trigger_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, object]:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mean, std = cifar10_mean_std(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    use_amp = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_asr = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
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

        clean_acc = _evaluate_accuracy(model, test_clean_loader, device, mean, std, target_label=None)
        attack_success = _evaluate_accuracy(
            model,
            test_trigger_loader,
            device,
            mean,
            std,
            target_label=cfg.trigger_label,
        )

        avg_loss = running_loss / float(max(len(train_loader), 1))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(avg_loss),
                "clean_acc": float(clean_acc),
                "attack_success_rate": float(attack_success),
            }
        )

        if attack_success > best_asr:
            best_asr = attack_success
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_asr": float(best_asr),
    }


def fine_tune_last_layer(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: FTLLConfig,
) -> Dict[str, object]:
    if not hasattr(model, "fc"):
        raise ValueError("FTLL expects model to have a final layer named 'fc'.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    mean, std = cifar10_mean_std(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=float(cfg.lr))

    use_amp = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict[str, float]] = []
    best_acc = -1.0
    best_state = None

    for epoch in range(int(cfg.epochs)):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
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

        clean_acc = _evaluate_accuracy(model, test_loader, device, mean, std, target_label=None)
        avg_loss = running_loss / float(max(len(train_loader), 1))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(avg_loss),
                "clean_acc": float(clean_acc),
            }
        )

        if clean_acc > best_acc:
            best_acc = clean_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_clean_acc": float(best_acc),
    }
