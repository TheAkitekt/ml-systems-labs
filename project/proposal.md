PROJECT PROPOSAL: EDGE-DEPLOYABLE SHONA KEYWORD SPOTTING

Student Name: Brendon Chisakuwana
Student ID: 70491
Course: Machine Learning Systems (Vistula University)

1. PROBLEM
Commercial offline voice assistants rarely support low-resource African languages like Shona (chiShona). Users have to rely on cloud speech APIs, which need constant internet, add network delay, and send raw audio to external servers. This project builds an offline voice-command recognizer that classifies 1-second spoken Shona clips locally on a standard CPU.

2. DATASET
- Sources: Google FLEURS (sn_zw Shona split) and Mozilla Common Voice (sn Shona split) 16 kHz audio recordings.
- Classes (10 total): 8 Shona commands (hongu, kwete, mira, enda, vhura, vhara, dzoka, rubatsiro), plus "unknown" (other Shona words) and "silence" (background noise).
- Preprocessing: Audio is cut into 1-second clips and converted into 40x101 log-Mel spectrograms. Train, validation, and test splits are separated by speaker ID so the same voice never appears in both training and testing. Normalization stats are computed on the training set only, and the final split is locked with a SHA-256 hash.

3. SUCCESS METRICS
- Top-1 Test Accuracy: at least 88%
- Macro F1-Score: at least 88% (so rare command classes are weighted equally against "unknown")

4. DEPLOYMENT CONSTRAINTS
- Inference Latency: 50 ms or less per 1-second clip on CPU (batch size 1, median of 100 runs after warm-up, served via FastAPI)
- Model Size: 5 MB or less on disk
- Peak Memory (RSS): 250 MB or less during inference

5. BASELINE AND PLANNED OPTIMIZATIONS
- Baseline Model: Depthwise-Separable Convolutional Neural Network (DS-CNN) trained in PyTorch using a YAML config file and tracked in MLflow.
- Quantization: Convert FP32 weights and activations to INT8 (ONNX Runtime) to shrink file size by roughly 4x and speed up CPU execution.
- Knowledge Distillation: Train the compact DS-CNN using softened logits (T=4) from a larger CNN teacher to improve generalization across limited Shona speakers.
- Structured Pruning: Remove low-norm convolutional channels to reduce FLOPs and meet the 50 ms CPU latency limit

