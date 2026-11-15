import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

from autograd import Value


def get_moons(n_samples: int = 100, seed: int = 42):
    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=seed)
    return X.astype(np.float64), y.astype(np.float64)


def make_init_weights(seed: int = 42):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 0.5, size=(2, 8))
    b1 = np.zeros(8, dtype=np.float64)
    W2 = rng.normal(0.0, 0.5, size=(8, 1))
    b2 = np.zeros(1, dtype=np.float64)
    return W1, b1, W2, b2


def run_custom_autograd(X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 0.2) -> tuple[float, float]:
    W1_init, b1_init, W2_init, b2_init = make_init_weights(42)
    W1 = [[Value(W1_init[i, j]) for j in range(8)] for i in range(2)]
    b1 = [Value(b1_init[j]) for j in range(8)]
    W2 = [[Value(W2_init[j, 0]) for _ in range(1)] for j in range(8)]
    b2 = [Value(b2_init[0])]
    params = [p for row in W1 for p in row] + b1 + [p for row in W2 for p in row] + b2

    t0 = time.perf_counter()
    for _ in range(epochs):
        total_loss = Value(0.0)
        for xi, yi in zip(X, y):
            x0, x1 = Value(xi[0]), Value(xi[1])
            h = [(x0 * W1[0][j] + x1 * W1[1][j] + b1[j]).sigmoid() for j in range(8)]
            logit = b2[0]
            for j in range(8):
                logit = logit + h[j] * W2[j][0]
            pred = logit.sigmoid()
            # Binary cross-entropy: -(y * log(p) + (1 - y) * log(1 - p))
            if yi == 1.0:
                loss_i = -pred.log()
            else:
                loss_i = -(Value(1.0) - pred).log()
            total_loss = total_loss + loss_i
        mean_loss = total_loss * (1.0 / len(X))

        for p in params:
            p.grad = 0.0
        mean_loss.backward()
        for p in params:
            p.data -= lr * p.grad
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    correct = 0
    for xi, yi in zip(X, y):
        h = [1.0 / (1.0 + np.exp(-(xi[0] * W1[0][j].data + xi[1] * W1[1][j].data + b1[j].data))) for j in range(8)]
        logit = b2[0].data + sum(h[j] * W2[j][0].data for j in range(8))
        pred = 1.0 if (1.0 / (1.0 + np.exp(-logit))) >= 0.5 else 0.0
        correct += int(pred == yi)
    return elapsed_ms, correct / len(X)


def run_pytorch(X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 0.2) -> tuple[float, float]:
    W1_init, b1_init, W2_init, b2_init = make_init_weights(42)
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y[:, None], dtype=torch.float64)

    model = nn.Sequential(nn.Linear(2, 8), nn.Sigmoid(), nn.Linear(8, 1), nn.Sigmoid()).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1_init.T))
        model[0].bias.copy_(torch.tensor(b1_init))
        model[2].weight.copy_(torch.tensor(W2_init.T))
        model[2].bias.copy_(torch.tensor(b2_init))

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    t0 = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        preds = (model(X_t) >= 0.5).double()
        acc = float((preds == y_t).double().mean())
    return elapsed_ms, acc


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    X, y = get_moons(n_samples=100, seed=42)

    # Warm-up
    _ = run_custom_autograd(X, y)
    _ = run_pytorch(X, y)

    custom_times, pt_times = [], []
    custom_acc, pt_acc = 0.0, 0.0
    for _ in range(5):
        t_c, custom_acc = run_custom_autograd(X, y)
        t_p, pt_acc = run_pytorch(X, y)
        custom_times.append(t_c)
        pt_times.append(t_p)

    med_custom = statistics.median(custom_times)
    med_pt = statistics.median(pt_times)
    df = pd.DataFrame([
        {"engine": "Custom Scalar Value", "median_train_ms": round(med_custom, 3), "accuracy": round(custom_acc, 4), "slowdown_vs_pt": round(med_custom / med_pt, 1)},
        {"engine": "PyTorch Tensor Autograd", "median_train_ms": round(med_pt, 3), "accuracy": round(pt_acc, 4), "slowdown_vs_pt": 1.0},
    ])
    df.to_csv(results_dir / "moons_benchmark.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
