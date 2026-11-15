from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from thop import profile

from models_data import (
    StudentCNN,
    TeacherCNN,
    eval_accuracy,
    get_loaders,
    log_versions,
    measure_cpu_latency_ms,
    set_seeds,
)


def main() -> None:
    set_seeds(42)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_versions(results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(batch_size=128, num_workers=1, seed=42)

    teacher = TeacherCNN().to(device)
    student_preview = StudentCNN()
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student_preview.parameters())
    print(f"Device: {device} | Teacher params: {t_params:,} | Student params: {s_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)

    for epoch in range(1, 9):
        teacher.train()
        run_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(teacher(xb), yb)
            loss.backward()
            optimizer.step()
            run_loss += float(loss) * yb.size(0)
            n_seen += yb.size(0)
        val_acc = eval_accuracy(teacher, val_loader, device)
        print(f"Epoch {epoch}/8 | train_loss: {run_loss / n_seen:.4f} | val_acc: {val_acc:.4f}")

    test_acc = eval_accuracy(teacher, test_loader, device)
    ckpt_path = results_dir / "teacher_dense.pt"
    torch.save(teacher.state_dict(), ckpt_path)
    size_mb = round(ckpt_path.stat().st_size / (1024 ** 2), 4)

    macs, _ = profile(teacher.to("cpu"), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    flops = int(macs * 2)
    lat_b1 = measure_cpu_latency_ms(teacher, batch_size=1, runs=100)
    lat_b128 = measure_cpu_latency_ms(teacher, batch_size=128, runs=20)

    df = pd.DataFrame([{
        "model": "TeacherCNN (Baseline)",
        "params": t_params,
        "size_mb": size_mb,
        "flops": flops,
        "lat_b1_ms": lat_b1,
        "lat_b128_ms": lat_b128,
        "test_acc": test_acc,
    }])
    df.to_csv(results_dir / "teacher_baseline.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
