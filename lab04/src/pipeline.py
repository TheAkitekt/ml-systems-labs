import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from validate import ingest_raw, log_versions, validate_and_audit


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.low_ = np.quantile(arr, self.lower, axis=0)
        self.high_ = np.quantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.low_, self.high_)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    out["hours_per_age"] = out["hours-per-week"] / (out["age"] - 16)
    return out


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def run_experiment(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, y_te: np.ndarray, with_eng: bool):
    if with_eng:
        X_tr, X_te = add_engineered_features(X_tr), add_engineered_features(X_te)
    num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    cat_cols = [c for c in X_tr.columns if c not in num_cols]

    prep = build_preprocessor(num_cols, cat_cols)
    X_tr_proc = prep.fit_transform(X_tr)
    X_te_proc = prep.transform(X_te)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr_proc, y_tr)
    preds = clf.predict(X_te_proc)

    return prep, X_tr_proc, X_te_proc, {
        "features": "with_engineered" if with_eng else "baseline_14",
        "num_transformed_cols": int(X_tr_proc.shape[1]),
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "f1": round(float(f1_score(y_te, preds)), 4),
    }


def main() -> None:
    lab_dir = Path(__file__).resolve().parents[1]
    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_versions(results_dir)

    cfg = yaml.safe_load((lab_dir / "config.yaml").read_text(encoding="utf-8"))
    raw_df = ingest_raw(lab_dir.parent / "data")
    val_report = validate_and_audit(raw_df, cfg)
    (results_dir / "validation_report.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")

    target_col = val_report["target_col_raw"]
    df = raw_df.replace("?", np.nan)
    X = df.drop(columns=[target_col])
    y = (df[target_col].astype(str).str.strip().str.rstrip(".") == ">50K").astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=cfg["seed"]
    )

    # Drop exact duplicates from train only; keep test untouched
    train_combined = X_train.copy()
    train_combined["__y__"] = y_train
    before_dedup = len(train_combined)
    train_combined = train_combined.drop_duplicates()
    dropped_train_dups = before_dedup - len(train_combined)
    y_train = train_combined["__y__"].to_numpy()
    X_train = train_combined.drop(columns=["__y__"])

    _, _, _, res_base = run_experiment(X_train, y_train, X_test, y_test, with_eng=False)
    prep_eng, X_tr_proc, X_te_proc, res_eng = run_experiment(X_train, y_train, X_test, y_test, with_eng=True)

    joblib.dump(prep_eng, results_dir / "preprocessor.joblib")

    # Step 7: SHA-256 hash of processed dataset + config manifest
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(X_tr_proc, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(X_te_proc, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(y_train, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(y_test, dtype=np.int64).tobytes())
    sha256_hex = hasher.hexdigest()

    metrics_df = pd.DataFrame([res_base, res_eng])
    metrics_df.to_csv(results_dir / "baseline_metrics.csv", index=False)

    manifest = {
        "processed_dataset_sha256": sha256_hex,
        "train_rows_after_dedup": int(len(X_train)),
        "train_dups_dropped": int(dropped_train_dups),
        "test_rows_untouched": int(len(X_test)),
        "config": cfg,
    }
    (results_dir / "processed_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    print(f"Train rows after dedup: {len(X_train)} (dropped {dropped_train_dups}), Test rows: {len(X_test)}")
    print(metrics_df.to_string(index=False))
    print(f"Processed SHA256: {sha256_hex}")


if __name__ == "__main__":
    main()
