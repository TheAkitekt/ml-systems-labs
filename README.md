# ML Systems Labs

Laboratory implementations and system benchmarks for the Machine Learning Systems course at Vistula University, Warsaw. Each lab evaluates ML models from an engineering perspective: correctness, latency, memory footprint, storage cost, and deployment constraints.

## Labs

| Lab | Topic | Summary |
| :--- | :--- | :--- |
| 01 | Environment and first measurements | Pinned Python 3.11.8 environment; Logistic Regression vs. Random Forest on Breast Cancer Wisconsin; training/inference latency, artifact size, and peak RSS across Cloud, Edge, Mobile, and TinyML budgets. |
| 02 | Neural network from scratch | Two-hidden-layer MLP in NumPy on `load_digits`; forward pass, cross-entropy loss, manual backpropagation, numerical gradient check, learning-rate/width sweep, and forward vs. backward pass timing. |
| 03 | Architectures and their costs | MLP, CNN, and Transformer Attention on Fashion-MNIST; parameter counts, hand-calculated vs. `thop` FLOPs, batch-1 vs. batch-128 latency, and accuracy/cost trade-off plots. |
| 04 | Data pipeline and quality | Leakage-free UCI Adult pipeline; schema/range validation, duplicate-conflict label audit, train-only imputation/clipping, derived feature evaluation, SHA-256 data hash, and data card. |
| 05 | Frameworks and automatic differentiation | Scalar `Value` reverse-mode autograd engine verified against PyTorch and numerical gradients; `make_moons` speed comparison; eager vs. `torch.jit.trace` and non-contiguous tensor view benchmarks. |

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
