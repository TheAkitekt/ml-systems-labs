# Lab 1: Environment and First System Measurements

## 1. Goal
Set up an isolated Python 3.11.8 environment with pinned package versions and measure accuracy, training time, inference latency, model size, and peak RAM for Logistic Regression and Random Forest on the Breast Cancer Wisconsin dataset to see which deployment tiers they fit.

## 2. Method
- **Environment and Seeds:** Created a Python 3.11.8 virtual environment with fixed library versions and set seed `42` across `random`, `numpy`, and `torch`.
- **Dataset and Models:** Loaded the Breast Cancer Wisconsin dataset and split it 70/30 with stratification (`random_state=42`). Trained `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Measurements:** Recorded test accuracy to 4 decimal places, median training time over 5 runs after 1 warm-up, median single-sample inference time over 100 runs after 1 warm-up, saved file size with `joblib`, and peak process RSS memory with `psutil`.

## 3. Results

### System Measurements
| Model | Test Accuracy | Train Median (ms) | Single-Sample Inference Median (ms) | Model Size (Bytes) | Model Size (KB) | Peak Train RSS (MB) | Peak Inference RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression** | 0.9474 | 260.5506 | 0.0346 | 1055 | 1.03 | 267.80 | 267.82 |
| **RandomForestClassifier** | 0.9357 | 108.0504 | 1.6184 | 290889 | 284.07 | 268.85 | 269.33 |

### Resource Budget Fit
| Tier | Memory Budget | Latency Budget | Size Budget | LogisticRegression | RandomForestClassifier |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cloud** | >= 1 GB | <= 100 ms | <= 500 MB | **Pass** | **Pass** |
| **Edge** | 256-1024 MB | <= 50 ms | <= 50 MB | **Pass** | **Pass** |
| **Mobile** | 64-256 MB | <= 20 ms | <= 10 MB | **Fail (RAM)** | **Fail (RAM)** |
| **TinyML** | <= 256 KB | <= 10 ms | <= 100 KB | **Fail (RAM)** | **Fail (Size and RAM)** |

**Justification:**
- **Cloud and Edge:** Both models pass every check. Peak RSS is around 268-269 MB, which sits inside the 256-1024 MB Edge range and well under Cloud limits.
- **Mobile:** Both models easily beat the 20 ms latency limit and 10 MB file size limit, but fail the 256 MB RAM cap because loading the Python libraries in this environment takes about 268 MB of RSS. Exporting the weights to a lighter runtime like ONNX or C would fix this.
- **TinyML:** Logistic Regression fits the 100 KB storage limit (1.03 KB) and 10 ms latency limit (0.0346 ms), so its weights could run on a microcontroller if rewritten in C without Python. Random Forest fails both storage (284.07 KB > 100 KB) and RAM limits.

## 4. Conclusions
1. **Python runtime memory is the main bottleneck on small devices:** Even though Logistic Regression takes only 1.03 KB on disk, the Python process uses 267.82 MB of RAM, which blocks standard scikit-learn scripts from running on Mobile or TinyML hardware.
2. **Training speed and inference speed do not match:** Logistic Regression took over twice as long to train as Random Forest (260.55 ms vs. 108.05 ms) because `lbfgs` hit the 1000-iteration cap on unscaled features, yet it predicted a single sample about 47 times faster (0.0346 ms vs. 1.6184 ms).
3. **Tree ensembles use much more storage than linear models:** Saving 100 decision trees makes the Random Forest file roughly 275 times larger than Logistic Regression (284.07 KB vs. 1.03 KB), pushing it past TinyML flash storage limits even on a small 30-feature dataset.
