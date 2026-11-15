import importlib.metadata
import platform
import sys
from pathlib import Path

PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm",
]

def main():
    lines = [
        f"Python: {sys.version.split()[0]} ({platform.platform()})",
        "-" * 40,
    ]
    for pkg in PACKAGES:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lines.append(f"{pkg}=={ver}")

    output = "\n".join(lines)
    print(output)
    out_path = Path(__file__).resolve().parents[1] / "results" / "versions.txt"
    out_path.write_text(output + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
