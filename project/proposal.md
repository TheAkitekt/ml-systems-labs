# Individual Course Project Proposal: Edge-Deployable Acoustic Keyword Spotting

- **Student Name:** Brendon Chisakuwana
- **Student ID:** 70491
- **Course:** Machine Learning Systems (Vistula University, Warsaw)

## 1. Problem Definition
Always-on voice interfaces on resource-constrained edge and mobile devices must recognize spoken command keywords locally without cloud round-trips, privacy exposure, or battery drain. This project builds, optimizes, and packages an end-to-end acoustic keyword spotting (KWS) inference service that classifies 1-second audio clips into discrete voice commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, plus `silence` and `unknown`).

## 2. Dataset & Preprocessing Pipeline
- **Dataset:** Google Speech Commands v0.02 (`torchaudio.datasets.SPEECHCOMMANDS`, ~105,829 1-second WAV utterances sampled at 16 kHz across 35 words, mapped to the standard 12-class KWS benchmark split with official speaker-disjoint train/validation/test lists).
- **Feature Extraction:** Deterministic conversion of 16 kHz waveforms into 40-bin log-Mel spectrograms (`n_fft=400`, `hop_length=160`, `40 x 101` time-frequency frames) with train-only mean/variance normalization and a SHA-256 dataset manifest.

## 3. Target Success Metrics & Deployment Constraints
- **Accuracy Metric:** Macro F1-score and Top-1 Test Accuracy >= **90.0%** on the 12-class test split.
- **Latency Budget:** Median single-utterance CPU inference latency <= **50 ms** (batch size 1, warm-up followed by 100 timed runs) exposed via a FastAPI inference endpoint.
- **Model Size Budget:** Serialized deployment artifact size <= **5 MB** (`ONNX` / quantized runtime format).
- **Memory Budget:** Peak process RSS <= **250 MB** during inference.

## 4. Baseline Architecture & Planned System Optimizations
- **Baseline Model:** Depthwise-Separable Convolutional Neural Network (DS-CNN) trained with PyTorch, YAML-driven configuration, and MLflow experiment tracking.
- **Planned Optimization Techniques:**
  1. **Post-Training Quantization (PTQ) & Quantization-Aware Training (QAT):** Convert FP32 weights and activations to INT8 (`ONNX Runtime` / PyTorch dynamic & static quantization) to cut artifact size by ~4x and accelerate CPU SIMD execution.
  2. **Structured Filter Pruning & Knowledge Distillation:** Prune low-magnitude depthwise/pointwise channels and distill from a wider teacher DS-CNN to recover accuracy while meeting the <= 50 ms and <= 5 MB edge deployment budgets.
