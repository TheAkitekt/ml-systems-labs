import statistics
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bench_fn(fn, x: torch.Tensor, warmup: int = 20, runs: int = 200) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = fn(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
    return round(statistics.median(times), 2)


def main() -> None:
    torch.manual_seed(42)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model = SmallNet().eval()
    x = torch.randn(256, 64)

    # 1) Inspect traced graph & autograd grad_fn
    traced = torch.jit.trace(model, x)
    graph_str = str(traced.graph)
    (results_dir / "traced_graph.txt").write_text(graph_str, encoding="utf-8")

    x_grad = torch.randn(4, 64, requires_grad=True)
    y_grad = model(x_grad).sum()
    print("Autograd root grad_fn:", y_grad.grad_fn)
    print("Traced TorchScript graph:\n", graph_str.strip())

    # 2) Eager vs compiled (with fallback to torch.jit.trace on Windows if Inductor unavailable)
    eager_us = bench_fn(model, x)
    mode_name = "torch.compile"
    try:
        compiled_model = torch.compile(model)
        opt_us = bench_fn(compiled_model, x)
    except Exception as exc:
        print(f"\ntorch.compile unsupported ({type(exc).__name__}), falling back to torch.jit.trace.")
        mode_name = "torch.jit.trace"
        opt_us = bench_fn(traced, x)

    # 3) Tensor shape, dtype, contiguity & non-contiguous view speed
    base = torch.randn(1024, 1024)
    non_contig = base.t()
    contig = non_contig.contiguous()

    print(f"\nBase tensor:          shape={tuple(base.shape)}, dtype={base.dtype}, contiguous={base.is_contiguous()}")
    print(f"Transposed view:      shape={tuple(non_contig.shape)}, dtype={non_contig.dtype}, contiguous={non_contig.is_contiguous()}")
    print(f"Contiguous copy:      shape={tuple(contig.shape)}, dtype={contig.dtype}, contiguous={contig.is_contiguous()}")

    contig_op_us = bench_fn(lambda t: (t * 2.0 + 1.0).sum(dim=1), contig, warmup=10, runs=100)
    non_contig_op_us = bench_fn(lambda t: (t * 2.0 + 1.0).sum(dim=1), non_contig, warmup=10, runs=100)

    df = pd.DataFrame([
        {"benchmark": "Eager SmallNet (B=256)", "median_us": eager_us},
        {"benchmark": f"{mode_name} SmallNet (B=256)", "median_us": opt_us},
        {"benchmark": "Contiguous view reduction (1024x1024)", "median_us": contig_op_us},
        {"benchmark": "Non-contiguous transposed view (1024x1024)", "median_us": non_contig_op_us},
    ])
    df.to_csv(results_dir / "compile_and_contiguity.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
