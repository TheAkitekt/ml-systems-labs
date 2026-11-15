import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from thop import profile
from torchinfo import summary

from data import set_seeds
from models import FashionAttention, FashionCNN, FashionMLP


def measure_latency_ms(model: torch.nn.Module, batch_size: int, runs: int, device: torch.device) -> float:
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return round(statistics.median(times), 4)


def main() -> None:
    set_seeds(42)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    models = {
        "MLP": FashionMLP().to(device),
        "CNN": FashionCNN().to(device),
        "Attention": FashionAttention().to(device),
    }

    # Hand-calculated FLOPs (2 * MACs) for representative layers & full models:
    # 1) MLP Linear(784->256): 2 * 784 * 256 = 401,408
    #    Total MLP: 2*(784*256 + 256*128 + 128*10) = 469,504 FLOPs
    # 2) CNN Conv2d(16->32, 3x3, 14x14): 2 * 3 * 3 * 16 * 32 * 14 * 14 = 1,806,336
    #    Total CNN: Conv1 (2*9*1*16*28*28=225,792) + Conv2 (1,806,336) + FC (2*1568*10=31,360) = 2,063,488 FLOPs
    # 3) Attention single encoder layer (seq=16, d=64, ff=128):
    #    QKV (2*16*64*192=393,216) + QK^T (2*16*16*64=32,768) + AV (2*16*16*64=32,768)
    #    + OutProj (2*16*64*64=131,072) + FFN (2*16*64*128 + 2*16*128*64 = 524,288) = 1,114,112
    #    Total Attention: PatchProj (2*16*49*64=100,352) + 2*Enc (2,228,224) + Head (2*64*10=1,280) = 2,329,856 FLOPs
    hand_total_flops = {
        "MLP": 469_504,
        "CNN": 2_063_488,
        "Attention": 2_329_856,
    }

    rows = []
    dummy_input = torch.randn(1, 1, 28, 28, device=device)

    for name, model in models.items():
        info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        total_params = info.total_params

        macs, thop_params = profile(model, inputs=(dummy_input,), verbose=False)
        thop_flops = int(macs * 2)

        lat_b1 = measure_latency_ms(model, batch_size=1, runs=100, device=device)
        lat_b128 = measure_latency_ms(model, batch_size=128, runs=20, device=device)

        rows.append({
            "model": name,
            "params": total_params,
            "hand_total_flops": hand_total_flops[name],
            "thop_flops": thop_flops,
            "lat_b1_ms": lat_b1,
            "lat_b128_ms": lat_b128,
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "model_costs.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
