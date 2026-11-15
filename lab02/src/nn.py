import numpy as np
from data import load_data, set_seeds


class TwoLayerMLP:
    def __init__(self, width: int, seed: int = 42):
        set_seeds(seed)
        self.width = width
        self.W1 = np.random.normal(0.0, 0.01, size=(64, width))
        self.b1 = np.zeros((width,), dtype=np.float64)
        self.W2 = np.random.normal(0.0, 0.01, size=(width, width))
        self.b2 = np.zeros((width,), dtype=np.float64)
        self.W3 = np.random.normal(0.0, 0.01, size=(width, 10))
        self.b3 = np.zeros((10,), dtype=np.float64)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        z3 = a2 @ self.W3 + self.b3
        probs = self.softmax(z3)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "probs": probs}
        return probs, cache

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, y_onehot: np.ndarray) -> float:
        eps = 1e-12
        clipped = np.clip(probs, eps, 1.0 - eps)
        return float(-np.sum(y_onehot * np.log(clipped)) / y_onehot.shape[0])

    def backward(self, cache: dict, y_onehot: np.ndarray) -> dict[str, np.ndarray]:
        batch_size = y_onehot.shape[0]
        X = cache["X"]
        z1, a1 = cache["z1"], cache["a1"]
        z2, a2 = cache["z2"], cache["a2"]
        probs = cache["probs"]

        dz3 = (probs - y_onehot) / batch_size
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0.0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0.0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def sgd_step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]
        self.W3 -= lr * grads["dW3"]
        self.b3 -= lr * grads["db3"]
