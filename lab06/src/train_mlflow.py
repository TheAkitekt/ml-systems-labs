import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml

from model_data import FashionCNN, get_loaders, log_versions, set_seeds


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += float(criterion(logits, yb)) * yb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum())
            total += yb.size(0)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lab06/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seeds(cfg["seed"])

    lab_dir = Path(__file__).resolve().parents[1]
    results_dir = lab_dir / "results"
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_versions(results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        seed=cfg["seed"],
    )

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    tracking_uri = (lab_dir / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name="baseline_yaml_run") as run:
        mlflow.log_params({
            "seed": cfg["seed"],
            "batch_size": cfg["dataset"]["batch_size"],
            "num_workers": cfg["dataset"]["num_workers"],
            "epochs": cfg["training"]["epochs"],
            "optimizer": cfg["training"]["optimizer"],
            "lr": cfg["training"]["lr"],
            "device": str(device),
        })

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            model.train()
            run_loss, n_seen = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                run_loss += float(loss) * yb.size(0)
                n_seen += yb.size(0)

            train_loss = run_loss / n_seen
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(f"Epoch {epoch}/{cfg['training']['epochs']} | train_loss: {train_loss:.4f} | val_acc: {val_acc:.4f}")

            ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        mlflow.log_artifact(args.config)
        mlflow.log_artifact(str(ckpt_dir / f"ckpt_epoch_{cfg['training']['epochs']}.pt"))
        print(f"MLflow Run ID: {run.info.run_id} | Final Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
