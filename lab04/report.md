# Lab 4: Data Pipeline and Quality

## 1. Goal & Problem Definition
Build a reproducible, leakage-free tabular preprocessing and validation pipeline on the UCI Adult dataset (`fetch_openml`, `data_id=1590`)[cite: 10].

**Problem Definition:**
- **Inputs:** 14 demographic, educational, and employment features (`age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`)[cite: 10].
- **Output:** Binary classification target predicting whether an individual's annual income exceeds `$50K` (`>50K` = 1, `<=50K` = 0)[cite: 10].
- **Success Metrics:** Test set **F1 score** (primary metric due to ~76%/24% class imbalance) and **Accuracy** evaluated using a `LogisticRegression` baseline[cite: 10].
- **Engineering Constraints:** Strict zero-leakage separation (70/30 stratified split prior to imputation, outlier clipping, scaling, and deduplication), explicit schema/range validation without silent row deletion, artifact serialization (`results/preprocessor.joblib`), cryptographic lineage (`SHA-256` + `config.yaml`), and single-command execution (`python lab04/src/pipeline.py`)[cite: 10].

## 2. Method
- **Ingestion & Validation (`src/validate.py`):** Loaded raw data untouched (`48,842` rows, `15` columns) and verified column dtypes, numeric bounds (`age` 17–90, `hours-per-week` 1–99, `capital-gain` 0–99999, `capital-loss` 0–4356), missing-value counts, exact duplicates, conflicting-label duplicates, and a 50-row random dictionary audit[cite: 10].
- **Leakage-Free Cleaning & Feature Engineering (`src/pipeline.py`):** Performed a 70/30 stratified split (`seed=42`) first, dropped 24 exact duplicate rows from the training split only (`34,189 -> 34,165`), kept all `14,653` test rows untouched, engineered `capital_net = capital-gain - capital-loss` and `hours_per_age = hours-per-week / (age - 16)`, fit median/mode imputers, 1st/99th percentile clippers, standard scalers, and one-hot encoders strictly on training data, saved `results/preprocessor.joblib`, and hashed the processed arrays with `SHA-256`[cite: 10].

## 3. Results

### Validation & Label-Quality Audit (`results/validation_report.json`)
- **Shape & Schema:** `48,842` rows x `15` columns; `0` dtype mismatches and `0` out-of-range violations across all four checked numeric columns[cite: 10].
- **Missing Values:** `workclass` (`2,799`), `occupation` (`2,809`), and `native-country` (`857`) contain missing entries; all other columns are complete.
- **Duplicate & Label-Conflict Audit:** Found `52` exact duplicate rows overall and **5 duplicate feature groups (10 rows)** with conflicting income labels (`<=50K` vs. `>50K`). A 50-row manual sample check (`seed=42`) found `0` impossible values[cite: 10].

### Baseline Model Comparison (`LogisticRegression`, `max_iter=1000`)
| Feature Set | Transformed Columns | Test Accuracy | Test F1 |
| :--- | :---: | :---: | :---: |
| **Baseline (14 raw features)** | 105 | 0.8540 | 0.6601 |
| **With Engineered Features (`capital_net`, `hours_per_age`)** | 107 | **0.8578** | **0.6696** |

- **Processed Dataset SHA-256:** `c1dec012bc02b15e107fbc63281ae762418b3c1e556b2c8af36592268eeb37de`[cite: 10]

## 4. Conclusions
1. **Leakage-free splitting preserves honest evaluation:** Splitting before deduplication, imputation, and quantile clipping removed 24 duplicate training records while keeping all `14,653` test records intact, preventing optimistic bias from train-test duplicate overlap or leaked distribution statistics[cite: 10].
2. **Domain-specific derived features improve linear separability:** Adding `capital_net` and `hours_per_age` raised test F1 by `+0.0095` (`0.6601 -> 0.6696`) and accuracy by `+0.0038` (`0.8540 -> 0.8578`), capturing net investment gains and age-normalized work intensity that linear combinations of raw columns express less cleanly after quantile clipping.
3. **Label noise establishes an irreducible error floor:** The audit uncovered 10 rows with identical feature vectors but opposite income labels (likely due to unobserved factors such as state cost-of-living or spouse income), showing why data-quality audits and cryptographic dataset hashing are essential before tuning model complexity[cite: 10].
