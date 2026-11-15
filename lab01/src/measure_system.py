import os
import random
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / (1024 ** 2)


def measure_model(name: str, base_model, X_train, y_train, X_test, results_dir: Path) -> dict:
    proc = psutil.Process(os.getpid())

    # 1. Training time: 1 warm-up + 5 timed runs
    clone(base_model).fit(X_train, y_train)

    train_times_ms = []
    peak_train_rss_mb = get_rss_mb(proc)
    trained_model = None

    for _ in range(5):
        model = clone(base_model)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        train_times_ms.append((t1 - t0) * 1000.0)
        peak_train_rss_mb = max(peak_train_rss_mb, get_rss_mb(proc))
        trained_model = model

    # 2. Model size: save with joblib.dump to results/model.joblib
    model_path = results_dir / "model.joblib"
    joblib.dump(trained_model, model_path)
    size_bytes = model_path.stat().st_size
    size_kb = round(size_bytes / 1024.0, 2)

    # 3. Single-sample inference: 1 warm-up + 100 timed runs
    sample = X_test[:1]
    _ = trained_model.predict(sample)

    infer_times_ms = []
    peak_infer_rss_mb = get_rss_mb(proc)

    for _ in range(100):
        t0 = time.perf_counter()
        _ = trained_model.predict(sample)
        t1 = time.perf_counter()
        infer_times_ms.append((t1 - t0) * 1000.0)
        peak_infer_rss_mb = max(peak_infer_rss_mb, get_rss_mb(proc))

    return {
        "model": name,
        "train_median_ms": round(statistics.median(train_times_ms), 4),
        "infer_median_ms": round(statistics.median(infer_times_ms), 4),
        "size_bytes": size_bytes,
        "size_kb": size_kb,
        "peak_train_rss_mb": round(peak_train_rss_mb, 2),
        "peak_infer_rss_mb": round(peak_infer_rss_mb, 2),
    }


def main() -> None:
    set_seeds(42)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    rows = [
        measure_model(name, clf, X_train, y_train, X_test, results_dir)
        for name, clf in models
    ]

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "system_metrics.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
