import importlib.metadata
import platform
import random
import statistics
import sys
import time
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
    packages = ["numpy", "pandas", "matplotlib", "torch", "torchvision", "thop", "scipy", "psutil"]
    lines = [f"Python: {sys.version.split()[0]} ({platform.platform()})", "-" * 40]
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lines.append(f"{pkg}=={ver}")
    (results_dir / "versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TeacherCNN(nn.Module):
    """Oversized baseline CNN (~997k parameters)."""
    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, fc_dim: int = 144):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(c3 * 7 * 7, fc_dim)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        return self.fc2(x)


class StudentCNN(nn.Module):
    """Compact student CNN (~100k parameters)."""
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
            nn.Linear(32 * 7 * 7, 60),
            nn.ReLU(),
            nn.Linear(60, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_loaders(batch_size: int = 128, num_workers: int = 1, seed: int = 42):
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


def eval_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += int((model(xb).argmax(dim=1) == yb).sum())
            total += yb.size(0)
    return round(correct / total, 4)


def measure_cpu_latency_ms(model: nn.Module, batch_size: int = 1, runs: int = 100) -> float:
    cpu_model = model.to("cpu").eval()
    x = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        for _ in range(10):
            _ = cpu_model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = cpu_model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return round(statistics.median(times), 4)
