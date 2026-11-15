# Data Card: UCI Adult (Census Income)

## 1. Source & Provenance
- **Dataset:** UCI Adult / Census Income (`openml` name `adult`, version 2, `data_id=1590`)[cite: 10].
- **Origin:** Extracted by Barry Becker from the 1994 U.S. Census Bureau database.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) via UCI Machine Learning Repository / OpenML.
- **One-Command Pipeline:** `python lab04/src/pipeline.py` (configured via `lab04/config.yaml`)[cite: 10].

## 2. Dataset Size & Structure
- **Raw Dimensions:** 48,842 rows and 15 columns (14 demographic/employment features + binary target `class`: `<=50K` vs. `>50K`)[cite: 10].
- **Splits:** Stratified 70/30 train/test split (`seed=42`)[cite: 10]:
  - **Train set:** 34,189 raw rows -> **34,165 rows** after dropping 24 exact duplicate rows[cite: 10].
  - **Test set:** **14,653 rows** kept completely untouched (no deduplication or row filtering)[cite: 10].
- **Processed Feature Matrix:** 105 features (baseline) / 107 features (with `capital_net` and `hours_per_age`) after train-only median/mode imputation, 1st–99th percentile clipping, standard scaling, and one-hot encoding[cite: 10].
- **Processed Dataset SHA-256:** `c1dec012bc02b15e107fbc63281ae762418b3c1e556b2c8af36592268eeb37de`[cite: 10].

## 3. Data Quality & Label Audit Findings
- **Missing Values:** Missing entries appear as nulls in `workclass` (2,799), `occupation` (2,809), and `native-country` (857). No rows were silently dropped during validation[cite: 10].
- **Range Checks:** Zero violations across `age` (`[17, 90]`), `hours-per-week` (`[1, 99]`), `capital-gain` (`[0, 99999]`), and `capital-loss` (`[0, 4356]`)[cite: 10].
- **Duplicates & Label Conflicts:** 52 exact duplicate rows exist in the full dataset, alongside **5 duplicate feature groups (10 total rows)** carrying conflicting income labels (`<=50K` and `>50K` for identical feature vectors)[cite: 10].
- **50-Row Manual Audit:** 0 impossible values found across a 50-row random sample (`seed=42`) checked against the data dictionary[cite: 10].

## 4. Known Biases & Limitations
- **Historical & Geographic Skew:** Reflects 1994 U.S. income distributions (`>50K` corresponds to roughly >$105k in 2026 dollars) and is heavily skewed toward U.S. respondents (`United-States` is the dominant `native-country`).
- **Demographic Representation:** Contains strong historical disparities across `sex` and `race` (e.g., higher representation of White and Male individuals in the `>50K` class). Models trained on this data can perpetuate wage and demographic biases if used for credit, hiring, or financial screening.
- **Sampling Weights (`fnlwgt`):** Census sampling weights are state-specific and not directly comparable across states without normalization.

## 5. Privacy & Ethical Considerations
- **Anonymization:** Direct identifiers (names, SSNs, exact addresses) were stripped by the U.S. Census Bureau prior to public release.
- **Re-identification Risk:** Quasi-identifiers (`age`, `sex`, `race`, `marital-status`, `occupation`, `education`, `native-country`, `hours-per-week`) can narrow down individuals in small demographic subgroups, so downstream artifacts must not be linked with external voter or public records.
