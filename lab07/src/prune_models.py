import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from thop import profile

from models_data import TeacherCNN, eval_accuracy, get_loaders, measure_cpu_latency_ms, set_seeds


def prunable_layers(model: TeacherCNN):
    return [
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.conv3, "weight"),
        (model.fc1, "weight"),
        (model.fc2, "weight"),
    ]


def fine_tune(model: nn.Module, train_loader, device: torch.device, epochs: int = 3, lr: float = 2e-4) -> None:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def build_structurally_pruned_teacher(base_model: TeacherCNN, ratio: float) -> TeacherCNN:
    """Physically remove lowest L2-norm filters/units by `ratio` and copy surviving weights."""
    keep = 1.0 - ratio
    c1_k = max(1, int(round(32 * keep)))
    c2_k = max(1, int(round(64 * keep)))
    c3_k = max(1, int(round(128 * keep)))
    fc_k = max(1, int(round(144 * keep)))

    def top_idx(w: torch.Tensor, k: int) -> torch.Tensor:
        norms = torch.norm(w.view(w.size(0), -1), p=2, dim=1)
        return torch.topk(norms, k=k, largest=True).indices.sort().values

    idx1 = top_idx(base_model.conv1.weight.data, c1_k)
    idx2 = top_idx(base_model.conv2.weight.data, c2_k)
    idx3 = top_idx(base_model.conv3.weight.data, c3_k)
    idxf = top_idx(base_model.fc1.weight.data, fc_k)

    slim = TeacherCNN(c1=c1_k, c2=c2_k, c3=c3_k, fc_dim=fc_k)
    with torch.no_grad():
        slim.conv1.weight.copy_(base_model.conv1.weight.data[idx1])
        slim.conv1.bias.copy_(base_model.conv1.bias.data[idx1])

        slim.conv2.weight.copy_(base_model.conv2.weight.data[idx2][:, idx1])
        slim.conv2.bias.copy_(base_model.conv2.bias.data[idx2])

        slim.conv3.weight.copy_(base_model.conv3.weight.data[idx3][:, idx2])
        slim.conv3.bias.copy_(base_model.conv3.bias.data[idx3])

        # Flatten indices corresponding to kept c3 channels over 7x7 spatial grid
        flat_idx = torch.cat([ch * 49 + torch.arange(49) for ch in idx3])
        slim.fc1.weight.copy_(base_model.fc1.weight.data[idxf][:, flat_idx])
        slim.fc1.bias.copy_(base_model.fc1.bias.data[idxf])

        slim.fc2.weight.copy_(base_model.fc2.weight.data[:, idxf])
        slim.fc2.bias.copy_(base_model.fc2.bias.data)
    return slim


def save_sparse_state(state_dict: dict, path: Path) -> float:
    sparse_dict = {}
    for k, v in state_dict.items():
        if v.ndim >= 2 and (v == 0).float().mean().item() > 0.5:
            sparse_dict[k] = v.to_sparse_coo().coalesce()
        else:
            sparse_dict[k] = v
    torch.save(sparse_dict, path)
    return round(path.stat().st_size / (1024 ** 2), 4)


def main() -> None:
    set_seeds(42)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, test_loader = get_loaders(batch_size=128, num_workers=1, seed=42)

    base = TeacherCNN()
    base.load_state_dict(torch.load(results_dir / "teacher_dense.pt", map_location="cpu", weights_only=True))
    base_acc = eval_accuracy(base.to(device), test_loader, device)
    base_lat_b1 = measure_cpu_latency_ms(base, batch_size=1, runs=100)
    base_lat_b128 = measure_cpu_latency_ms(base, batch_size=128, runs=20)

    records = []
    sparsities = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]

    for method in ["global_unstructured", "layerwise_unstructured"]:
        for sp in sparsities:
            if sp == 0.0:
                records.append({
                    "method": method, "target_sparsity": 0.0, "actual_sparsity": 0.0,
                    "params": 997434, "lat_b1_ms": base_lat_b1, "lat_b128_ms": base_lat_b128, "test_acc": base_acc
                })
                continue

            set_seeds(42)
            m = copy.deepcopy(base).cpu()
            params_to_prune = prunable_layers(m)
            if method == "global_unstructured":
                prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=sp)
            else:
                for mod, pname in params_to_prune:
                    prune.l1_unstructured(mod, name=pname, amount=sp)

            fine_tune(m, train_loader, device, epochs=3, lr=2e-4)
            for mod, pname in prunable_layers(m):
                prune.remove(mod, pname)

            zeros = sum(int((p == 0).sum()) for p in m.parameters())
            total = sum(p.numel() for p in m.parameters())
            actual_sp = round(zeros / total, 4)
            acc = eval_accuracy(m.to(device), test_loader, device)
            lat_b1 = measure_cpu_latency_ms(m, batch_size=1, runs=100)
            lat_b128 = measure_cpu_latency_ms(m, batch_size=128, runs=20)

            if method == "global_unstructured" and sp == 0.90:
                dense_p = results_dir / "teacher_global90_dense.pt"
                sparse_p = results_dir / "teacher_global90_sparse.pt"
                torch.save(m.cpu().state_dict(), dense_p)
                dense_mb = round(dense_p.stat().st_size / (1024 ** 2), 4)
                sparse_mb = save_sparse_state(m.cpu().state_dict(), sparse_p)
                print(f"90% Global Pruned On-Disk Size -> Dense .pt: {dense_mb} MB | Sparse COO .pt: {sparse_mb} MB")

            records.append({
                "method": method, "target_sparsity": sp, "actual_sparsity": actual_sp,
                "params": total, "lat_b1_ms": lat_b1, "lat_b128_ms": lat_b128, "test_acc": acc
            })

    # Structured L2 filter pruning at [0%, 10%, 25%, 50%]
    records.append({
        "method": "structured_l2", "target_sparsity": 0.0, "actual_sparsity": 0.0,
        "params": 997434, "lat_b1_ms": base_lat_b1, "lat_b128_ms": base_lat_b128, "test_acc": base_acc
    })
    for ratio in [0.10, 0.25, 0.50]:
        set_seeds(42)
        slim = build_structurally_pruned_teacher(base.cpu(), ratio)
        fine_tune(slim, train_loader, device, epochs=3, lr=2e-4)
        slim_params = sum(p.numel() for p in slim.parameters())
        param_reduction = round(1.0 - (slim_params / 997434), 4)
        acc = eval_accuracy(slim.to(device), test_loader, device)
        lat_b1 = measure_cpu_latency_ms(slim, batch_size=1, runs=100)
        lat_b128 = measure_cpu_latency_ms(slim, batch_size=128, runs=20)

        if ratio == 0.50:
            struct_p = results_dir / "teacher_struct50.pt"
            torch.save(slim.cpu().state_dict(), struct_p)
            macs, _ = profile(slim.cpu(), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
            print(f"50% Structured Pruned -> Params: {slim_params:,} | Size: {struct_p.stat().st_size/(1024**2):.4f} MB | FLOPs: {int(macs*2):,}")

        records.append({
            "method": "structured_l2", "target_sparsity": ratio, "actual_sparsity": param_reduction,
            "params": slim_params, "lat_b1_ms": lat_b1, "lat_b128_ms": lat_b128, "test_acc": acc
        })

    df = pd.DataFrame(records)
    df.to_csv(results_dir / "pruning_results.csv", index=False)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, label in [
        ("global_unstructured", "Global Unstructured (L1)"),
        ("layerwise_unstructured", "Layer-wise Unstructured (L1)"),
        ("structured_l2", "Structured Filter Pruning (L2)"),
    ]:
        sub = df[df["method"] == method]
        ax.plot(sub["target_sparsity"] * 100, sub["test_acc"] * 100, marker="o", linewidth=2, label=label)

    ax.set_title("Fashion-MNIST Accuracy vs. Pruning Sparsity (3-Epoch Fine-Tune)")
    ax.set_xlabel("Target Sparsity / Filter Removal (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "sparsity_curves.png", dpi=150)


if __name__ == "__main__":
    main()
