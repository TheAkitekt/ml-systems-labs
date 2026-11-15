from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from data import get_dataloaders, set_seeds
from models import FashionAttention, FashionCNN, FashionMLP


def train_and_eval(model: nn.Module, train_loader, val_loader, test_loader, device: torch.device, epochs: int = 5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    def eval_acc(loader) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += int((preds == yb).sum())
                total += yb.size(0)
        return round(correct / total, 4)

    return eval_acc(val_loader), eval_acc(test_loader)


def main() -> None:
    set_seeds(42)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, seed=42)

    costs_df = pd.read_csv(results_dir / "model_costs.csv")
    models = {"MLP": FashionMLP(), "CNN": FashionCNN(), "Attention": FashionAttention()}

    val_accs, test_accs = [], []
    for name, model in models.items():
        set_seeds(42)
        val_acc, test_acc = train_and_eval(model.to(device), train_loader, val_loader, test_loader, device, epochs=5)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        print(f"{name:10s} | val_acc: {val_acc:.4f} | test_acc: {test_acc:.4f}")

    costs_df["val_acc"] = val_accs
    costs_df["test_acc"] = test_accs
    costs_df.to_csv(results_dir / "full_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for _, row in costs_df.iterrows():
        axes[0].scatter(row["params"] / 1e3, row["test_acc"] * 100, s=90)
        axes[0].annotate(row["model"], (row["params"] / 1e3, row["test_acc"] * 100), xytext=(5, 4), textcoords="offset points")
        axes[1].scatter(row["lat_b128_ms"], row["test_acc"] * 100, s=90)
        axes[1].annotate(row["model"], (row["lat_b128_ms"], row["test_acc"] * 100), xytext=(5, 4), textcoords="offset points")

    axes[0].set_title("Test Accuracy vs. Parameters")
    axes[0].set_xlabel("Parameters (Thousands)")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Test Accuracy vs. Batch-128 Latency")
    axes[1].set_xlabel("Inference Latency, Batch=128 (ms)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "tradeoffs.png", dpi=150)


if __name__ == "__main__":
    main()
