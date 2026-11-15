import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import fetch_openml


def log_versions(results_dir: Path) -> None:
    packages = [
        "numpy", "pandas", "scikit-learn", "scipy", "pyyaml", "joblib", "torch"
    ]
    lines = [f"Python: {sys.version.split()[0]} ({platform.platform()})", "-" * 40]
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lines.append(f"{pkg}=={ver}")
    (results_dir / "versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ingest_raw(data_dir: Path) -> pd.DataFrame:
    try:
        bunch = fetch_openml(name="adult", version=2, as_frame=True, parser="auto", data_home=str(data_dir))
    except Exception:
        bunch = fetch_openml(data_id=1590, as_frame=True, parser="auto", data_home=str(data_dir))
    return bunch.frame.copy()


def validate_and_audit(df: pd.DataFrame, cfg: dict) -> dict:
    target_col = "income" if "income" in df.columns else "class"
    expected_types = {
        "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
        "education-num": "int", "marital-status": "category", "occupation": "category",
        "relationship": "category", "race": "category", "sex": "category",
        "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
        "native-country": "category", target_col: "category",
    }

    type_issues = []
    for col, exp in expected_types.items():
        if col not in df.columns:
            type_issues.append(f"Missing column: {col}")
            continue
        actual = str(df[col].dtype)
        if exp == "int" and not pd.api.types.is_integer_dtype(df[col]):
            type_issues.append(f"{col}: expected int, got {actual}")
        elif exp == "category" and not isinstance(df[col].dtype, pd.CategoricalDtype):
            type_issues.append(f"{col}: expected category, got {actual}")

    range_violations = {}
    for col, (lo, hi) in cfg["validation"]["ranges"].items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        range_violations[col] = {"min": float(df[col].min()), "max": float(df[col].max()), "violations": bad}

    missing_counts = {}
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        q_cnt = int((s == "?").sum())
        na_cnt = int(df[col].isna().sum())
        if q_cnt > 0 or na_cnt > 0:
            missing_counts[col] = {"question_mark": q_cnt, "null_na": na_cnt}

    feature_cols = [c for c in df.columns if c != target_col]
    norm_df = df.copy()
    for c in norm_df.columns:
        norm_df[c] = norm_df[c].astype(str).str.strip()

    exact_dups = int(norm_df.duplicated().sum())
    grouped = norm_df.groupby(feature_cols, dropna=False)[target_col].nunique()
    conflicting_groups = grouped[grouped > 1]
    conflicting_rows = int(norm_df.set_index(feature_cols).index.isin(conflicting_groups.index).sum())

    sample50 = df.sample(n=50, random_state=cfg["seed"])
    impossible_in_50 = 0
    for col, (lo, hi) in cfg["validation"]["ranges"].items():
        impossible_in_50 += int(((sample50[col] < lo) | (sample50[col] > hi)).sum())
    impossible_in_50 += int((sample50["education-num"] < 1).sum() + (sample50["education-num"] > 16).sum())

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "target_col_raw": target_col,
        "type_issues": type_issues,
        "range_violations": range_violations,
        "missing_counts": missing_counts,
        "exact_duplicate_rows": exact_dups,
        "conflicting_feature_groups": int(len(conflicting_groups)),
        "conflicting_label_rows": conflicting_rows,
        "random_50_impossible_values": impossible_in_50,
    }


if __name__ == "__main__":
    lab_dir = Path(__file__).resolve().parents[1]
    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_versions(results_dir)

    cfg = yaml.safe_load((lab_dir / "config.yaml").read_text(encoding="utf-8"))
    raw_df = ingest_raw(lab_dir.parent / "data")
    report = validate_and_audit(raw_df, cfg)
    (results_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
