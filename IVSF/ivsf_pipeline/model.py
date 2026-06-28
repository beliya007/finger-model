from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel: int, outchannel: int, stride: int = 1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inchannel, channels, s))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def _load_state_dict_from_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def maybe_train_model(repo_root: Path, train_config_path: Path, num_models: int = 1) -> Path:
    script = repo_root / "BIVBench-main" / "train_protect_model.py"
    cmd = [
        sys.executable,
        str(script),
        "-c",
        str(train_config_path),
        "--num",
        str(num_models),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root / "BIVBench-main"))
    default_ckpt = repo_root / "BIVBench-main" / "outputs" / "cifar10" / "random_start_models" / "resnet20" / "best_0.pth"
    if not default_ckpt.exists():
        raise FileNotFoundError(f"Training finished but checkpoint not found: {default_ckpt}")
    return default_ckpt


def load_resnet20_cifar10(
    repo_root: Path,
    model_source: str,
    checkpoint_path: Path,
    train_config_path: Path,
    device: torch.device,
    train_num_models: int = 1,
) -> Tuple[nn.Module, Path]:
    if model_source not in {"checkpoint", "train"}:
        raise ValueError("model_source must be one of: checkpoint, train")

    ckpt_path = checkpoint_path
    if model_source == "train":
        ckpt_path = maybe_train_model(repo_root, train_config_path, train_num_models)

    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = ResNet20CIFAR10(num_classes=10)
    state_dict = _load_state_dict_from_checkpoint(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt_path
