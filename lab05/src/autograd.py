import importlib.metadata
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch


def log_versions(results_dir: Path) -> None:
    packages = ["numpy", "pandas", "scikit-learn", "torch"]
    lines = [f"Python: {sys.version.split()[0]} ({platform.platform()})", "-" * 40]
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lines.append(f"{pkg}=={ver}")
    (results_dir / "versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: float):
        assert isinstance(power, (int, float)), "Power must be int or float"
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.data))))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def log(self):
        clamped = max(1e-12, self.data)
        out = Value(math.log(clamped), (self,), "log")

        def _backward():
            self.grad += (1.0 / clamped) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class ValueVec:
    def __init__(self, values: list[Value]):
        self.values = values

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ValueVec":
        return cls([Value(float(x)) for x in arr])

    def __mul__(self, other: "ValueVec") -> "ValueVec":
        return ValueVec([x * y for x, y in zip(self.values, other.values)])

    def __add__(self, other: "ValueVec") -> "ValueVec":
        return ValueVec([x + y for x, y in zip(self.values, other.values)])

    def __pow__(self, p: float) -> "ValueVec":
        return ValueVec([x ** p for x in self.values])

    def relu(self) -> "ValueVec":
        return ValueVec([x.relu() for x in self.values])

    def sum(self) -> Value:
        total = Value(0.0)
        for x in self.values:
            total = total + x
        return total

    def grads(self) -> np.ndarray:
        return np.array([x.grad for x in self.values], dtype=np.float64)


def verify_gradients() -> None:
    rng = np.random.default_rng(42)
    a_np = rng.normal(size=5)
    b_np = rng.normal(size=5)
    c_np = rng.normal(size=5)

    # 1) Custom Value autograd
    a_v, b_v, c_v = ValueVec.from_array(a_np), ValueVec.from_array(b_np), ValueVec.from_array(c_np)
    out_v = ((a_v * b_v + c_v) ** 2).relu().sum()
    out_v.backward()

    # 2) PyTorch autograd
    a_t = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    c_t = torch.tensor(c_np, dtype=torch.float64, requires_grad=True)
    out_t = ((a_t * b_t + c_t) ** 2).relu().sum()
    out_t.backward()

    # 3) Numerical central difference (eps = 1e-6)
    eps = 1e-6
    def f_np(a, b, c):
        return float(np.sum(np.maximum(0.0, (a * b + c) ** 2)))

    num_grads = {}
    for name, arr in [("a", a_np), ("b", b_np), ("c", c_np)]:
        g = np.zeros_like(arr)
        for i in range(len(arr)):
            plus, minus = arr.copy(), arr.copy()
            plus[i] += eps
            minus[i] -= eps
            if name == "a":
                g[i] = (f_np(plus, b_np, c_np) - f_np(minus, b_np, c_np)) / (2 * eps)
            elif name == "b":
                g[i] = (f_np(a_np, plus, c_np) - f_np(a_np, minus, c_np)) / (2 * eps)
            else:
                g[i] = (f_np(a_np, b_np, plus) - f_np(a_np, b_np, minus)) / (2 * eps)
        num_grads[name] = g

    diff_pt_a = np.max(np.abs(a_v.grads() - a_t.grad.numpy()))
    diff_pt_b = np.max(np.abs(b_v.grads() - b_t.grad.numpy()))
    diff_pt_c = np.max(np.abs(c_v.grads() - c_t.grad.numpy()))
    diff_num = max(
        np.max(np.abs(a_v.grads() - num_grads["a"])),
        np.max(np.abs(b_v.grads() - num_grads["b"])),
        np.max(np.abs(c_v.grads() - num_grads["c"])),
    )
    max_diff = max(diff_pt_a, diff_pt_b, diff_pt_c, diff_num)
    print(f"Forward output diff vs PyTorch: {abs(out_v.data - out_t.item()):.2e}")
    print(f"Max grad diff vs PyTorch:   da={diff_pt_a:.2e}, db={diff_pt_b:.2e}, dc={diff_pt_c:.2e}")
    print(f"Max grad diff vs Numerical: {diff_num:.2e}")
    print(f"Overall max abs diff:       {max_diff:.2e} (passed < 1e-6: {max_diff < 1e-6})")


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_versions(results_dir)
    verify_gradients()
