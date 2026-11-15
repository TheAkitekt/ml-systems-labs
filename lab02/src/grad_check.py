from pathlib import Path
import numpy as np
import pandas as pd
from data import load_data, set_seeds
from nn import TwoLayerMLP


def run_gradient_check(width: int = 16, eps: float = 1e-5, num_checks: int = 10) -> pd.DataFrame:
    set_seeds(42)
    X_train, _, y_train, _, _, _ = load_data()
    X_batch = X_train[:32]
    y_batch = y_train[:32]

    net = TwoLayerMLP(width=width, seed=42)
    probs, cache = net.forward(X_batch)
    grads = net.backward(cache, y_batch)

    params = [
        ("W1", net.W1, grads["dW1"]),
        ("W2", net.W2, grads["dW2"]),
        ("W3", net.W3, grads["dW3"]),
    ]

    rng = np.random.default_rng(42)
    rows = []
    for i in range(num_checks):
        layer_name, weight_mat, grad_mat = params[i % len(params)]
        r = int(rng.integers(0, weight_mat.shape[0]))
        c = int(rng.integers(0, weight_mat.shape[1]))

        orig = weight_mat[r, c]

        weight_mat[r, c] = orig + eps
        p_plus, _ = net.forward(X_batch)
        loss_plus = net.cross_entropy_loss(p_plus, y_batch)

        weight_mat[r, c] = orig - eps
        p_minus, _ = net.forward(X_batch)
        loss_minus = net.cross_entropy_loss(p_minus, y_batch)

        weight_mat[r, c] = orig

        num_grad = (loss_plus - loss_minus) / (2.0 * eps)
        ana_grad = float(grad_mat[r, c])
        denom = max(abs(ana_grad) + abs(num_grad), 1e-12)
        rel_err = abs(ana_grad - num_grad) / denom

        rows.append({
            "check_id": i + 1,
            "param": f"{layer_name}[{r},{c}]",
            "analytic_grad": ana_grad,
            "numerical_grad": num_grad,
            "rel_error": rel_err,
            "passed": rel_err < 1e-6,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df = run_gradient_check(width=16, eps=1e-5, num_checks=10)
    df.to_csv(results_dir / "grad_check.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nPassed: {df['passed'].sum()}/10")
