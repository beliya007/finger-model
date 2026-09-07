from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

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


class SixLayerCNNGTSRB(nn.Module):
    """Six-convolution classifier matching BIVBench's gtsrb_convnet architecture."""

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        self.dropout_1 = nn.Dropout(0.2)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.dropout_2 = nn.Dropout(0.2)
        self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm2d(128)
        self.conv_6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_6 = nn.BatchNorm2d(128)
        self.dropout_3 = nn.Dropout(0.2)
        self.fc_1 = nn.Linear(4 * 4 * 128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_1(F.relu(self.conv_1(x)))
        x = self.bn_2(F.relu(self.conv_2(x)))
        x = self.dropout_1(F.max_pool2d(x, 2, 2))
        x = self.bn_3(F.relu(self.conv_3(x)))
        x = self.bn_4(F.relu(self.conv_4(x)))
        x = self.dropout_2(F.max_pool2d(x, 2, 2))
        x = self.bn_5(F.relu(self.conv_5(x)))
        x = self.bn_6(F.relu(self.conv_6(x)))
        x = self.dropout_3(F.max_pool2d(x, 2, 2))
        return self.fc_1(x.flatten(1))


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handle checkpoints saved from DataParallel or wrapped modules.
    normalized = OrderedDict()
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        normalized[nk] = v
    return normalized


def _convert_public_resnet20_to_local_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Convert keys like conv1/bn1/layerX.Y.conv1 to local keys used in this file.
    converted = OrderedDict()
    for k, v in state_dict.items():
        nk = k

        if nk.startswith("conv1."):
            nk = nk.replace("conv1.", "conv1.0.", 1)
        elif nk.startswith("bn1."):
            nk = nk.replace("bn1.", "conv1.1.", 1)
        elif nk.startswith("layer"):
            nk = nk.replace(".conv1.", ".left.0.")
            nk = nk.replace(".bn1.", ".left.1.")
            nk = nk.replace(".conv2.", ".left.3.")
            nk = nk.replace(".bn2.", ".left.4.")
            nk = nk.replace(".downsample.0.", ".shortcut.0.")
            nk = nk.replace(".downsample.1.", ".shortcut.1.")

        converted[nk] = v
    return converted


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    # BIVBench usually saves as {"model": ..., "optimizer": ...}; keep this as first priority.
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net", "weights"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = _normalize_state_dict_keys(checkpoint[key])
                if "bn1.weight" in state_dict and "conv1.0.weight" not in state_dict:
                    state_dict = _convert_public_resnet20_to_local_keys(state_dict)
                return state_dict

        # Some checkpoints are plain state_dict directly.
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = _normalize_state_dict_keys(checkpoint)
            if "bn1.weight" in state_dict and "conv1.0.weight" not in state_dict:
                state_dict = _convert_public_resnet20_to_local_keys(state_dict)
            return state_dict

    raise ValueError("Unsupported checkpoint format. Expected dict with model/state_dict or plain state_dict.")


def _load_state_dict_from_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    return _extract_state_dict(checkpoint)


def _resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    # Resolve user-provided path first.
    provided = checkpoint_path.resolve() if not checkpoint_path.is_absolute() else checkpoint_path
    if provided.exists():
        return provided

    # Fallback to BIVBench common locations when users want to reuse its trained weights.
    cwd = Path.cwd()
    candidates = [
        cwd / ".." / "BIVBench-main" / "outputs" / "cifar10" / "random_start_models" / "resnet20" / "best_0.pth",
        cwd / ".." / "BIVBench-main" / "outputs" / "cifar10" / "random_start" / "resnet20" / "best_0.pth",
        cwd / "BIVBench-main" / "outputs" / "cifar10" / "random_start_models" / "resnet20" / "best_0.pth",
        cwd / "BIVBench-main" / "outputs" / "cifar10" / "random_start" / "resnet20" / "best_0.pth",
    ]
    for p in candidates:
        rp = p.resolve()
        if rp.exists():
            return rp

    # Final fallback: download a public CIFAR-10 ResNet20 checkpoint.
    # This keeps IVSF runnable even when no local/BIVBench weights are present.
    target = provided
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        hub_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        torch.save({"model": hub_model.state_dict()}, str(target))
        if target.exists():
            return target
    except Exception:
        pass

    tried = "\n".join([str(provided)] + [str(p.resolve()) for p in candidates])
    raise FileNotFoundError(
        "Checkpoint not found. Searched these paths:\n"
        f"{tried}\n"
        "Also tried to download public CIFAR-10 ResNet20 weights but failed. "
        "Place your trained weight in IVSF/checkpoints/resnet20_cifar10.pth or pass --checkpoint-path explicitly."
    )


def load_resnet20_cifar10(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Path]:
    ckpt_path = _resolve_checkpoint_path(checkpoint_path)

    model = ResNet20CIFAR10(num_classes=10)
    state_dict = _load_state_dict_from_checkpoint(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt_path


def load_model(dataset: str, checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Path]:
    dataset_name = str(dataset).lower()
    if dataset_name == "cifar10":
        return load_resnet20_cifar10(checkpoint_path, device)
    if dataset_name != "gtsrb":
        raise ValueError(f"Unsupported dataset: {dataset}. Expected cifar10 or gtsrb.")

    ckpt_path = checkpoint_path.resolve() if checkpoint_path.is_absolute() else checkpoint_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"GTSRB six-layer CNN checkpoint not found: {ckpt_path}. "
            "Train BIVBench gtsrb_convnet first, then pass its checkpoint with --checkpoint-path."
        )
    model = SixLayerCNNGTSRB(num_classes=43)
    model.load_state_dict(_load_state_dict_from_checkpoint(ckpt_path), strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt_path
