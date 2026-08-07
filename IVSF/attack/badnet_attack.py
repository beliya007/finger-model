from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from attack.data import BackdoorDataConfig, create_backdoor_data_loaders, create_clean_data_loaders
from attack.deeplearning import FTLLConfig, TrainConfig, backdoor_model_trainer, evaluate_clean_accuracy, fine_tune_last_layer
from ivsf_pipeline.model import ResNet20CIFAR10, load_resnet20_cifar10


@dataclass
class AttackConfig:
    clean_checkpoint_path: str = "./checkpoints/resnet20_cifar10.pth"
    data_root: str = "./data"
    output_dir: str = "./outputs/attack_run"
    device: str = "cuda"
    seed: int = 1337

    # attack options: badnet / trojan / clean_label_backdoor / weight_pruning / ftll
    attack_type: str = "badnet"

    # BadNets-style setup
    trigger_label: int = 0
    poison_ratio: float = 0.1
    trigger_size: int = 4
    trigger_alpha: float = 1.0
    trojan_steps: int = 200
    trojan_lr: float = 0.1
    trojan_batch_size: int = 64
    clean_label_base_label: int = 1
    clean_label_eps: float = 8.0 / 255.0
    clean_label_alpha: float = 1.0 / 255.0
    clean_label_steps: int = 5
    clean_label_max_poison: int = 256

    # Fine-tuning setup
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3

    # Weight pruning / FTLL specific setup
    pruning_ratio: float = 0.15
    ftll_epochs: int = 3
    ftll_lr: float = 1e-5

    num_workers: int = 2
    use_amp: bool = True

    def ensure_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_default_attacked_checkpoint_path(output_dir: str, attack_type: str = "badnet") -> Path:
    if str(attack_type) == "badnet":
        # Keep old default path for backward compatibility.
        return Path(output_dir) / "attacked_resnet20_cifar10.pth"
    return Path(output_dir) / f"attacked_resnet20_cifar10_{attack_type}.pth"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_weight_pruning(model: nn.Module, sparsity: float, prune_last_layer: bool = True) -> Dict[str, float]:
    sparsity = float(sparsity)
    if sparsity <= 0.0:
        return {"pruned_params": 0.0, "total_params": 0.0, "actual_sparsity": 0.0}

    bn_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_param_names.add(f"{module_name}.weight")
            bn_param_names.add(f"{module_name}.bias")

    total_params = 0
    pruned_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in bn_param_names:
                continue
            if (not prune_last_layer) and name.startswith("fc."):
                continue
            if not param.is_floating_point():
                continue
            if param.numel() == 0:
                continue

            flat = param.view(-1)
            num_pruned = int(ceil(flat.numel() * sparsity))
            if num_pruned <= 0:
                continue

            idx = torch.argsort(flat.abs())[:num_pruned]
            flat[idx] = 0.0

            total_params += int(flat.numel())
            pruned_params += int(num_pruned)

    actual = float(pruned_params) / float(max(total_params, 1))
    return {
        "pruned_params": float(pruned_params),
        "total_params": float(total_params),
        "actual_sparsity": actual,
    }


def train_attack(cfg: AttackConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    cfg.attack_type = str(cfg.attack_type).lower()
    output_dir = cfg.ensure_output_dir()

    use_cuda = torch.cuda.is_available() and str(cfg.device).lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    model, clean_ckpt_path = load_resnet20_cifar10(Path(cfg.clean_checkpoint_path), device=device)

    history = []
    extra_meta: Dict[str, object] = {}

    if cfg.attack_type in {"badnet", "trojan", "clean_label_backdoor"}:
        data_cfg = BackdoorDataConfig(
            data_root=cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            attack_type=cfg.attack_type,
            poison_ratio=cfg.poison_ratio,
            trigger_label=cfg.trigger_label,
            trigger_size=cfg.trigger_size,
            trigger_alpha=cfg.trigger_alpha,
            trojan_steps=cfg.trojan_steps,
            trojan_lr=cfg.trojan_lr,
            trojan_batch_size=cfg.trojan_batch_size,
            clean_label_base_label=cfg.clean_label_base_label,
            clean_label_eps=cfg.clean_label_eps,
            clean_label_alpha=cfg.clean_label_alpha,
            clean_label_steps=cfg.clean_label_steps,
            clean_label_max_poison=cfg.clean_label_max_poison,
        )
        loaders = create_backdoor_data_loaders(data_cfg, device=device, model_for_poison=model)

        train_cfg = TrainConfig(
            epochs=cfg.epochs,
            lr=cfg.lr,
            trigger_label=cfg.trigger_label,
            use_amp=cfg.use_amp,
        )
        train_result = backdoor_model_trainer(
            model=model,
            train_loader=loaders["train"],
            test_clean_loader=loaders["test_clean"],
            test_trigger_loader=loaders["test_trigger"],
            device=device,
            cfg=train_cfg,
        )

        model = train_result["model"]
        history = train_result["history"]
        extra_meta["best_asr"] = float(train_result["best_asr"])
    elif cfg.attack_type == "weight_pruning":
        data_cfg = BackdoorDataConfig(
            data_root=cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            attack_type=cfg.attack_type,
            poison_ratio=0.0,
            trigger_label=cfg.trigger_label,
            trigger_size=cfg.trigger_size,
            trigger_alpha=cfg.trigger_alpha,
        )
        loaders = create_clean_data_loaders(data_cfg, device=device)
        prune_stats = _apply_weight_pruning(model=model, sparsity=cfg.pruning_ratio, prune_last_layer=True)
        clean_acc = evaluate_clean_accuracy(model, loaders["test"], device)

        history = [
            {
                "clean_acc": float(clean_acc),
                "pruning_ratio": float(cfg.pruning_ratio),
                "actual_sparsity": float(prune_stats["actual_sparsity"]),
            }
        ]
        extra_meta.update(prune_stats)
    elif cfg.attack_type == "ftll":
        data_cfg = BackdoorDataConfig(
            data_root=cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            attack_type=cfg.attack_type,
            poison_ratio=0.0,
            trigger_label=cfg.trigger_label,
            trigger_size=cfg.trigger_size,
            trigger_alpha=cfg.trigger_alpha,
        )
        loaders = create_clean_data_loaders(data_cfg, device=device)
        ftll_result = fine_tune_last_layer(
            model=model,
            train_loader=loaders["train"],
            test_loader=loaders["test"],
            device=device,
            cfg=FTLLConfig(epochs=cfg.ftll_epochs, lr=cfg.ftll_lr, use_amp=cfg.use_amp),
        )
        model = ftll_result["model"]
        history = ftll_result["history"]
        extra_meta["best_clean_acc"] = float(ftll_result["best_clean_acc"])
    else:
        raise ValueError(
            "Unsupported attack_type. Expected one of: badnet, trojan, clean_label_backdoor, weight_pruning, ftll"
        )

    attacked_ckpt_path = get_default_attacked_checkpoint_path(cfg.output_dir, cfg.attack_type)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "attack_type": cfg.attack_type,
                "clean_checkpoint": str(clean_ckpt_path),
                "trigger_label": cfg.trigger_label,
                "poison_ratio": cfg.poison_ratio,
                "trigger_size": cfg.trigger_size,
                "trigger_alpha": cfg.trigger_alpha,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "ftll_epochs": cfg.ftll_epochs,
                "ftll_lr": cfg.ftll_lr,
                "pruning_ratio": cfg.pruning_ratio,
                "use_amp": cfg.use_amp,
                "device": str(device),
                "history": history,
                **extra_meta,
            },
        },
        str(attacked_ckpt_path),
    )

    return {
        "attacked_model": model,
        "clean_checkpoint": clean_ckpt_path,
        "attacked_checkpoint": attacked_ckpt_path,
        "history": history,
        "device": device,
        "attack_type": cfg.attack_type,
        "meta": extra_meta,
    }


def train_badnet_attack(cfg: AttackConfig) -> Dict[str, object]:
    cfg.attack_type = "badnet"
    return train_attack(cfg)


def load_attacked_model_checkpoint(attacked_checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    ckpt_path = attacked_checkpoint_path.resolve() if not attacked_checkpoint_path.is_absolute() else attacked_checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Attacked checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Unsupported attacked checkpoint format, expected dict with key 'model'.")

    model = ResNet20CIFAR10(num_classes=10)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    model.eval()

    return {
        "attacked_model": model,
        "attacked_checkpoint": ckpt_path,
        "meta": checkpoint.get("meta", {}),
        "device": device,
    }
