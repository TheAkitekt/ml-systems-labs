import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seeds(42)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = round(float(accuracy_score(y_test, preds)), 4)
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})

    results_df = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parents[1] / "results" / "baseline_accuracy.csv"
    results_df.to_csv(out_path, index=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
