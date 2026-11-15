import statistics
import time
from pathlib import Path
import pandas as pd
from data import load_data, set_seeds
from nn import TwoLayerMLP


def benchmark_passes(width: int, batch_size: int = 32, runs: int = 100) -> dict:
    set_seeds(42)
    X_train, _, y_train, _, _, _ = load_data()
    xb = X_train[:batch_size]
    yb = y_train[:batch_size]

    net = TwoLayerMLP(width=width, seed=42)

    # 1 warm-up pass
    _, cache = net.forward(xb)
    _ = net.backward(cache, yb)

    fwd_us = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _, cache = net.forward(xb)
        t1 = time.perf_counter()
        fwd_us.append((t1 - t0) * 1e6)

    bwd_us = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = net.backward(cache, yb)
        t1 = time.perf_counter()
        bwd_us.append((t1 - t0) * 1e6)

    med_fwd = statistics.median(fwd_us)
    med_bwd = statistics.median(bwd_us)
    return {
        "width": width,
        "batch_size": batch_size,
        "forward_median_us": round(med_fwd, 2),
        "backward_median_us": round(med_bwd, 2),
        "backward_to_forward_ratio": round(med_bwd / med_fwd, 2),
    }


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([benchmark_passes(w) for w in [16, 64]])
    df.to_csv(results_dir / "timing.csv", index=False)
    print(df.to_string(index=False))
