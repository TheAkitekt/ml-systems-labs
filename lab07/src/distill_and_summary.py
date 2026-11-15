from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from models_data import StudentCNN, TeacherCNN, eval_accuracy, get_loaders, measure_cpu_latency_ms, set_seeds


def train_student(
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    teacher: nn.Module | None = None,
    epochs: int = 8,
    T: float = 4.0,
    alpha: float = 0.5,
) -> tuple[StudentCNN, float]:
    set_seeds(42)
    student = StudentCNN().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    if teacher is not None:
        teacher = teacher.to(device).eval()

    for _ in range(epochs):
        student.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            s_logits = student(xb)
            hard_loss = ce_loss(s_logits, yb)
            if teacher is None:
                loss = hard_loss
            else:
                with torch.no_grad():
                    t_logits = teacher(xb)
                soft_loss = (T * T) * F.kl_div(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction="batchmean",
                )
                loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
            loss.backward()
            optimizer.step()

    test_acc = eval_accuracy(student, test_loader, device)
    return student, test_acc


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(batch_size=128, num_workers=1, seed=42)

    dense_mb = round((results_dir / "teacher_global90_dense.pt").stat().st_size / (1024 ** 2), 4)
    sparse_mb = round((results_dir / "teacher_global90_sparse.pt").stat().st_size / (1024 ** 2), 4)
    print(f"90% Global Pruned Disk Size -> Dense .pt: {dense_mb} MB | Sparse COO .pt: {sparse_mb} MB\n")

    teacher = TeacherCNN()
    teacher.load_state_dict(torch.load(results_dir / "teacher_dense.pt", map_location="cpu", weights_only=True))

    plain_student, plain_acc = train_student(train_loader, val_loader, test_loader, device, teacher=None, epochs=8)
    distill_student, distill_acc = train_student(train_loader, val_loader, test_loader, device, teacher=teacher, epochs=8, T=4.0, alpha=0.5)

    torch.save(plain_student.cpu().state_dict(), results_dir / "student_plain.pt")
    torch.save(distill_student.cpu().state_dict(), results_dir / "student_distilled.pt")

    s_params = sum(p.numel() for p in plain_student.parameters())
    s_size_mb = round((results_dir / "student_distilled.pt").stat().st_size / (1024 ** 2), 4)
    macs, _ = profile(plain_student.cpu(), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    s_flops = int(macs * 2)
    plain_lat_b1 = measure_cpu_latency_ms(plain_student, batch_size=1, runs=100)
    plain_lat_b128 = measure_cpu_latency_ms(plain_student, batch_size=128, runs=20)
    dist_lat_b1 = measure_cpu_latency_ms(distill_student, batch_size=1, runs=100)
    dist_lat_b128 = measure_cpu_latency_ms(distill_student, batch_size=128, runs=20)

    df = pd.DataFrame([
        {
            "model": "Plain StudentCNN (CE only)",
            "params": s_params,
            "size_mb": s_size_mb,
            "flops": s_flops,
            "lat_b1_ms": plain_lat_b1,
            "lat_b128_ms": plain_lat_b128,
            "test_acc": plain_acc,
        },
        {
            "model": "Distilled StudentCNN (T=4, alpha=0.5)",
            "params": s_params,
            "size_mb": s_size_mb,
            "flops": s_flops,
            "lat_b1_ms": dist_lat_b1,
            "lat_b128_ms": dist_lat_b128,
            "test_acc": distill_acc,
        },
    ])
    df.to_csv(results_dir / "distillation_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
