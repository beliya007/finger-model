from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ivsf_pipeline.model import SixLayerCNNGTSRB


GTSRB_MEAN = (0.3337, 0.3064, 0.3171)
GTSRB_STD = (0.2672, 0.2564, 0.2629)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GTSRB six-layer CNN used by IVSF.")
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints/gtsrb_six_layer_cnn.pth"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            correct += int((model(images).argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    set_seed(args.seed)
    use_cuda = args.device.lower().startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = device.type == "cuda"

    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
        ]
    )
    train_dataset = datasets.GTSRB(root=str(args.data_root), split="train", download=True, transform=train_transform)
    test_dataset = datasets.GTSRB(root=str(args.data_root), split="test", download=True, transform=test_transform)
    loader_options = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_options)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = SixLayerCNNGTSRB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 50], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())

        validation_accuracy = evaluate(model, test_loader, device)
        scheduler.step()
        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={running_loss / max(total, 1):.4f}, "
            f"train_acc={correct / max(total, 1):.4%}, val_acc={validation_accuracy:.4%}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "dataset": "gtsrb",
                        "architecture": "SixLayerCNNGTSRB",
                        "num_classes": 43,
                        "image_size": 32,
                        "mean": GTSRB_MEAN,
                        "std": GTSRB_STD,
                        "epoch": epoch,
                        "validation_accuracy": validation_accuracy,
                    },
                },
                args.checkpoint_path,
            )

    print(f"Best checkpoint: {args.checkpoint_path.resolve()}")
    print(f"Best validation accuracy: {best_accuracy:.4%}")


if __name__ == "__main__":
    main()