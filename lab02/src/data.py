import importlib.metadata
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


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


def load_data():
    set_seeds(42)
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float64)
    y_onehot = np.eye(10, dtype=np.float64)[y]

    X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
        X, y_onehot, y, test_size=0.30, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, y_train_idx, y_test_idx


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_versions(results_dir)
    X_train, X_test, y_train, y_test, _, _ = load_data()
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, range: [{X_train.min()}, {X_train.max()}]")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
