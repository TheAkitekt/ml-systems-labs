import importlib.metadata
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_versions(results_dir: Path) -> None:
    packages = ["numpy", "pandas", "matplotlib", "torch", "torchvision", "mlflow", "psutil", "pyyaml"]
    lines = [f"Python: {sys.version.split()[0]} ({platform.platform()})", "-" * 40]
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lines.append(f"{pkg}=={ver}")
    (results_dir / "versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_loaders(batch_size: int = 128, num_workers: int = 0, seed: int = 42):
    data_dir = Path(__file__).resolve().parents[2] / "data"
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tf)

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [55000, 5000], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
