# ML Systems Labs

Laboratory implementations and system benchmarks for the Machine Learning Systems course at Vistula University, Warsaw. Each lab evaluates ML models from an engineering perspective: correctness, latency, memory footprint, storage cost, and deployment constraints.

## Labs

| Lab | Topic | Summary |
| :--- | :--- | :--- |
| 01 | Environment and first measurements | Pinned Python 3.11.8 environment; Logistic Regression vs. Random Forest on Breast Cancer Wisconsin; training/inference latency, artifact size, and peak RSS across Cloud, Edge, Mobile, and TinyML budgets. |

## Setup

Python `3.11.8` is required:

    python3.11 -m venv .venv
    source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Repository Structure

    lab01/ ... lab11/
      report.md    Goal, method, results, and conclusions
      src/         Source scripts
      results/     Generated metrics, plots, and serialized artifacts
    project/       Final course project, data card, and model card

Datasets download automatically into `data/` (git-ignored) when running scripts.

## Reproducibility Standards
- **Deterministic Seeds:** Fixed seed `42` across Python `random`, NumPy, and PyTorch.
- **Pinned Environment:** Exact library versions specified in `requirements.txt`.
- **Timing Protocol:** Warm-up pass followed by repeated runs (median reported).
