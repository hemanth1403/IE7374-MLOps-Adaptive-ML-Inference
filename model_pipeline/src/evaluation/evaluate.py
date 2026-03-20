from __future__ import annotations

from pathlib import Path
import json
import yaml

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_model_weights(model_cfg: dict, checkpoints_root: Path) -> Path | str:
    trained_weights = checkpoints_root / model_cfg["output_subdir"] / "weights" / "best.pt"
    if trained_weights.exists():
        return trained_weights
    return model_cfg["weights"]


def main() -> None:
    eval_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "eval" / "eval_config.yaml")
    train_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "train" / "train_config.yaml")

    dataset_yaml = REPO_ROOT / "model_pipeline" / "artifacts" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {dataset_yaml}")

    metrics_dir = REPO_ROOT / eval_cfg["outputs"]["metrics_dir"]
    ensure_dir(metrics_dir)

    checkpoints_root = REPO_ROOT / train_cfg["outputs"]["checkpoints_dir"]

    for model_cfg in train_cfg["models"]:
        model_name = model_cfg["name"]
        weights_source = resolve_model_weights(model_cfg, checkpoints_root)

        print(f"Evaluating {model_name}")
        print(f"Using weights: {weights_source}")

        model = YOLO(str(weights_source))
        results = model.val(
            data=str(dataset_yaml),
            split=eval_cfg["evaluation"]["split"],
            device=eval_cfg["benchmark"]["device"],
            verbose=False,
        )

        metrics = {
            "model_name": model_name,
            "weights_source": str(weights_source),
            "split": eval_cfg["evaluation"]["split"],
            "metrics": {
                "mAP50": float(results.box.map50),
                "mAP50_95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            },
        }

        out_path = metrics_dir / f"{model_name}_metrics.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()