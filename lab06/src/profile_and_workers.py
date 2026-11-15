import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from model_data import FashionCNN, get_loaders, set_seeds


def profile_one_epoch(device: torch.device, results_dir: Path) -> pd.DataFrame:
    set_seeds(42)
    train_loader, _, _ = get_loaders(batch_size=128, num_workers=0, seed=42)
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model.train()
    data_iter = iter(train_loader)
    with profile(activities=activities, record_shapes=False) as prof:
        while True:
            with record_function("phase_data_loading"):
                try:
                    xb, yb = next(data_iter)
                except StopIteration:
                    break
                xb, yb = xb.to(device), yb.to(device)

            with record_function("phase_forward"):
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)

            with record_function("phase_backward"):
                loss.backward()

            with record_function("phase_optimizer_step"):
                optimizer.step()

    phases = {
        "Data Loading": "phase_data_loading",
        "Forward Pass": "phase_forward",
        "Backward Pass": "phase_backward",
        "Optimizer Step": "phase_optimizer_step",
    }
    raw_ms = {}
    for evt in prof.key_averages():
        for label, key in phases.items():
            if evt.key == key:
                raw_ms[label] = evt.cpu_time_total / 1000.0

    total_ms = sum(raw_ms.values())
    rows = []
    for label in phases:
        ms = raw_ms.get(label, 0.0)
        rows.append({
            "phase": label,
            "time_ms": round(ms, 2),
            "share_pct": round(100.0 * ms / total_ms, 2) if total_ms > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "profiler_breakdown.csv", index=False)
    return df


def benchmark_workers(device: torch.device, results_dir: Path) -> pd.DataFrame:
    rows = []
    for nw in [0, 1, 2, 4, 8]:
        set_seeds(42)
        train_loader, _, _ = get_loaders(batch_size=128, num_workers=nw, seed=42)
        model = FashionCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        t0 = time.perf_counter()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
        throughput = len(train_loader.dataset) / elapsed_s
        rows.append({
            "num_workers": nw,
            "epoch_time_s": round(elapsed_s, 3),
            "throughput_samples_s": round(throughput, 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "workers_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["num_workers"], df["epoch_time_s"], marker="o", linewidth=2)
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.set_title("Epoch Training Time vs. DataLoader num_workers (Windows)")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Epoch Time (seconds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "workers_plot.png", dpi=150)
    return df


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")
    prof_df = profile_one_epoch(device, results_dir)
    print("=== torch.profiler Epoch Breakdown ===")
    print(prof_df.to_string(index=False))
    workers_df = benchmark_workers(device, results_dir)
    print("\n=== num_workers Study ===")
    print(workers_df.to_string(index=False))
