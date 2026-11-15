import os
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
import torch.nn as nn

from model_data import FashionCNN, get_loaders, set_seeds


def eval_acc(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += int((model(xb).argmax(dim=1) == yb).sum())
            total += yb.size(0)
    return round(correct / total, 4)


def run_batch_study(device: torch.device, results_dir: Path) -> pd.DataFrame:
    proc = psutil.Process(os.getpid())
    rows = []
    for bs in [32, 64, 128, 256, 512]:
        set_seeds(42)
        train_loader, _, test_loader = get_loaders(batch_size=bs, num_workers=1, seed=42)
        model = FashionCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        peak_rss_mb = proc.memory_info().rss / (1024 ** 2)

        t0 = time.perf_counter()
        for _ in range(5):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            peak_rss_mb = max(peak_rss_mb, proc.memory_info().rss / (1024 ** 2))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mem_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            if device.type == "cuda"
            else round(peak_rss_mb, 2)
        )
        rows.append({
            "batch_size": bs,
            "throughput_samples_s": round((5 * len(train_loader.dataset)) / elapsed, 1),
            "peak_mem_mb": peak_mem_mb,
            "test_acc": eval_acc(model, test_loader, device),
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "batch_size_study.csv", index=False)
    return df


def run_grad_accum_check(device: torch.device, results_dir: Path) -> pd.DataFrame:
    configs = [("Native Batch 256 (accum=1)", 256, 1), ("Emulated Batch 256 (64 x accum=4)", 64, 4)]
    rows = []
    for label, bs, accum in configs:
        set_seeds(42)
        train_loader, _, test_loader = get_loaders(batch_size=bs, num_workers=1, seed=42)
        model = FashionCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epoch_losses = []
        for _ in range(5):
            model.train()
            optimizer.zero_grad()
            run_loss, n_seen = 0.0, 0
            for step, (xb, yb) in enumerate(train_loader, start=1):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                raw_loss = criterion(logits, yb)
                (raw_loss / accum).backward()
                if step % accum == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                run_loss += float(raw_loss) * yb.size(0)
                n_seen += yb.size(0)
            epoch_losses.append(round(run_loss / n_seen, 4))

        rows.append({
            "mode": label,
            "epoch1_loss": epoch_losses[0],
            "epoch3_loss": epoch_losses[2],
            "epoch5_loss": epoch_losses[4],
            "test_acc": eval_acc(model, test_loader, device),
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "grad_accum_study.csv", index=False)
    return df


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Batch-Size Study (5 Epochs) ===")
    print(run_batch_study(device, results_dir).to_string(index=False))
    print("\n=== Gradient Accumulation Check ===")
    print(run_grad_accum_check(device, results_dir).to_string(index=False))
