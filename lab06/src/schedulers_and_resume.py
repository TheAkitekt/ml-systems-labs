from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from model_data import FashionCNN, get_loaders, set_seeds


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += float(criterion(logits, yb)) * yb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum())
            total += yb.size(0)
    return total_loss / total, correct / total


def run_opt_sched_study(device: torch.device, results_dir: Path) -> pd.DataFrame:
    configs = [
        ("Adam + StepLR", "adam", 1e-3, "step"),
        ("Adam + CosineAnnealingLR", "adam", 1e-3, "cosine"),
        ("Adam + ReduceLROnPlateau", "adam", 1e-3, "plateau"),
        ("SGD+Momentum + StepLR", "sgd", 1e-2, "step"),
        ("SGD+Momentum + CosineAnnealingLR", "sgd", 1e-2, "cosine"),
        ("SGD+Momentum + ReduceLROnPlateau", "sgd", 1e-2, "plateau"),
    ]
    rows = []
    for label, opt_name, lr, sched_name in configs:
        set_seeds(42)
        train_loader, val_loader, test_loader = get_loaders(batch_size=128, num_workers=1, seed=42)
        model = FashionCNN().to(device)
        criterion = nn.CrossEntropyLoss()

        if opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        if sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        elif sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        final_train_loss, final_val_acc = 0.0, 0.0
        for _ in range(10):
            model.train()
            run_loss, n_seen = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                run_loss += float(loss) * yb.size(0)
                n_seen += yb.size(0)

            final_train_loss = run_loss / n_seen
            val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)
            if sched_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        _, test_acc = evaluate(model, test_loader, criterion, device)
        rows.append({
            "config": label,
            "final_lr": round(optimizer.param_groups[0]["lr"], 6),
            "final_train_loss": round(final_train_loss, 4),
            "val_acc": round(final_val_acc, 4),
            "test_acc": round(test_acc, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "opt_sched_study.csv", index=False)
    return df


def run_interrupt_resume_check(device: torch.device, results_dir: Path) -> pd.DataFrame:
    ckpt_path = results_dir / "interrupt_epoch3.pt"
    criterion = nn.CrossEntropyLoss()

    # 1) Uninterrupted 5-epoch baseline
    set_seeds(42)
    train_loader, val_loader, _ = get_loaders(batch_size=128, num_workers=1, seed=42)
    model_full = FashionCNN().to(device)
    opt_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)
    full_losses, full_accs = [], []
    for epoch in range(1, 6):
        model_full.train()
        run_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_full.zero_grad()
            loss = criterion(model_full(xb), yb)
            loss.backward()
            opt_full.step()
            run_loss += float(loss) * yb.size(0)
            n_seen += yb.size(0)
        full_losses.append(round(run_loss / n_seen, 5))
        _, va = evaluate(model_full, val_loader, criterion, device)
        full_accs.append(round(va, 4))

    # 2) Run epochs 1..3, interrupt & save full RNG + optimizer + model state, then resume 4..5
    set_seeds(42)
    train_loader, val_loader, _ = get_loaders(batch_size=128, num_workers=1, seed=42)
    model_part1 = FashionCNN().to(device)
    opt_part1 = torch.optim.Adam(model_part1.parameters(), lr=1e-3)
    resumed_losses, resumed_accs = [], []
    for epoch in range(1, 4):
        model_part1.train()
        run_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_part1.zero_grad()
            loss = criterion(model_part1(xb), yb)
            loss.backward()
            opt_part1.step()
            run_loss += float(loss) * yb.size(0)
            n_seen += yb.size(0)
        resumed_losses.append(round(run_loss / n_seen, 5))
        _, va = evaluate(model_part1, val_loader, criterion, device)
        resumed_accs.append(round(va, 4))

    torch.save({
        "epoch": 3,
        "model_state": model_part1.state_dict(),
        "optimizer_state": opt_part1.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }, ckpt_path)

    # Resume in a fresh model/optimizer instance
    ckpt = torch.load(ckpt_path, weights_only=False)
    model_resumed = FashionCNN().to(device)
    opt_resumed = torch.optim.Adam(model_resumed.parameters(), lr=1e-3)
    model_resumed.load_state_dict(ckpt["model_state"])
    opt_resumed.load_state_dict(ckpt["optimizer_state"])
    torch.set_rng_state(ckpt["torch_rng"])

    for epoch in range(4, 6):
        model_resumed.train()
        run_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_resumed.zero_grad()
            loss = criterion(model_resumed(xb), yb)
            loss.backward()
            opt_resumed.step()
            run_loss += float(loss) * yb.size(0)
            n_seen += yb.size(0)
        resumed_losses.append(round(run_loss / n_seen, 5))
        _, va = evaluate(model_resumed, val_loader, criterion, device)
        resumed_accs.append(round(va, 4))

    df = pd.DataFrame({
        "epoch": [1, 2, 3, 4, 5],
        "uninterrupted_loss": full_losses,
        "resumed_loss": resumed_losses,
        "uninterrupted_val_acc": full_accs,
        "resumed_val_acc": resumed_accs,
    })
    df.to_csv(results_dir / "resume_check.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["uninterrupted_loss"], marker="o", label="Uninterrupted (Epochs 1-5)")
    ax.plot(df["epoch"], df["resumed_loss"], linestyle="--", marker="x", label="Interrupted @ Epoch 3 & Resumed")
    ax.axvline(3, color="gray", linestyle=":", label="Checkpoint Save/Load")
    ax.set_title("Training Loss Continuity Across Epoch-3 Checkpoint Resume")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "resume_curve.png", dpi=150)
    return df


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== 10-Epoch Optimizer & Scheduler Study ===")
    print(run_opt_sched_study(device, results_dir).to_string(index=False))
    print("\n=== Epoch-3 Interrupt & Resume Verification ===")
    print(run_interrupt_resume_check(device, results_dir).to_string(index=False))
