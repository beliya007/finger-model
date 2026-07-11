from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from attack.data import BackdoorDataConfig, create_backdoor_data_loaders
from attack.deeplearning import TrainConfig, backdoor_model_trainer
from ivsf_pipeline.model import ResNet20CIFAR10, load_resnet20_cifar10


@dataclass
class AttackConfig:
    clean_checkpoint_path: str = "./checkpoints/resnet20_cifar10.pth"
    data_root: str = "./data"
    output_dir: str = "./outputs/attack_run"
    device: str = "cuda"
    seed: int = 1337

    # BadNets-style setup
    trigger_label: int = 0
    poison_ratio: float = 0.1
    trigger_size: int = 4
    trigger_alpha: float = 1.0

    # Fine-tuning setup
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    num_workers: int = 2
    use_amp: bool = True

    def ensure_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_default_attacked_checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir) / "attacked_resnet20_cifar10.pth"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_badnet_attack(cfg: AttackConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    output_dir = cfg.ensure_output_dir()

    use_cuda = torch.cuda.is_available() and str(cfg.device).lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    model, clean_ckpt_path = load_resnet20_cifar10(Path(cfg.clean_checkpoint_path), device=device)

    data_cfg = BackdoorDataConfig(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        poison_ratio=cfg.poison_ratio,
        trigger_label=cfg.trigger_label,
        trigger_size=cfg.trigger_size,
        trigger_alpha=cfg.trigger_alpha,
    )
    loaders = create_backdoor_data_loaders(data_cfg, device=device)

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
    best_asr = train_result["best_asr"]

    attacked_ckpt_path = get_default_attacked_checkpoint_path(cfg.output_dir)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "clean_checkpoint": str(clean_ckpt_path),
                "trigger_label": cfg.trigger_label,
                "poison_ratio": cfg.poison_ratio,
                "trigger_size": cfg.trigger_size,
                "trigger_alpha": cfg.trigger_alpha,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "use_amp": cfg.use_amp,
                "device": str(device),
                "history": history,
                "best_asr": best_asr,
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
    }


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
