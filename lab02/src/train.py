from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import load_data, set_seeds
from nn import TwoLayerMLP


def train_model(X_train, y_train, X_test, y_test_idx, width: int, lr: float, epochs: int = 50, batch_size: int = 32):
    set_seeds(42)
    net = TwoLayerMLP(width=width, seed=42)
    rng = np.random.default_rng(42)
    n_samples = X_train.shape[0]
    epoch_losses = []

    for _ in range(epochs):
        perm = rng.permutation(n_samples)
        X_shuf, y_shuf = X_train[perm], y_train[perm]
        batch_losses = []

        for start in range(0, n_samples, batch_size):
            xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            probs, cache = net.forward(xb)
            batch_losses.append(net.cross_entropy_loss(probs, yb))
            grads = net.backward(cache, yb)
            net.sgd_step(grads, lr)

        epoch_losses.append(float(np.mean(batch_losses)))

    test_probs, _ = net.forward(X_test)
    test_acc = float(np.mean(np.argmax(test_probs, axis=1) == y_test_idx))
    return epoch_losses, test_acc


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, _, _, y_test_idx = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    rows = []

    for ax, width in zip(axes, [16, 64]):
        for lr in [0.001, 0.01, 0.1]:
            losses, acc = train_model(X_train, y_train, X_test, y_test_idx, width=width, lr=lr)
            ax.plot(range(1, 51), losses, label=f"lr={lr}")
            rows.append({
                "width": width,
                "lr": lr,
                "final_train_loss": round(losses[-1], 4),
                "test_accuracy": round(acc, 4),
            })
        ax.set_title(f"Hidden Width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Training Cross-Entropy Loss")
    fig.tight_layout()
    fig.savefig(results_dir / "loss_curves.png", dpi=150)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "training_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
