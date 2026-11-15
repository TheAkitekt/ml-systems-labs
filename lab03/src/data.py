import importlib.metadata
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_versions(results_dir: Path) -> None:
    packages = [
        "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
        "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
        "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
        "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
        "joblib", "tqdm",
    ]
    lines = [f"Python: {sys.version.split()[0]} ({platform.platform()})", "-" * 40]
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lines.append(f"{pkg}=={ver}")
    (results_dir / "versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_dataloaders(batch_size: int = 128, seed: int = 42):
    set_seeds(seed)
    data_dir = Path(__file__).resolve().parents[2] / "data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [55000, 5000], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_versions(results_dir)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, seed=42)
    xb, yb = next(iter(train_loader))
    print(f"Train samples: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print(f"Batch shape: {tuple(xb.shape)}, dtype: {xb.dtype}, mean: {xb.mean():.4f}, std: {xb.std():.4f}")
