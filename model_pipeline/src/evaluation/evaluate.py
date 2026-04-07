from __future__ import annotations

from pathlib import Path
import json
import yaml
import mlflow

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

def build_runtime_dataset_yaml(src_yaml: Path, out_yaml: Path) -> Path:
    with src_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data_pipeline_root = REPO_ROOT / "Data-Pipeline"
    data["path"] = str(data_pipeline_root)

    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return out_yaml

def main() -> None:
    eval_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "eval" / "eval_config.yaml")
    train_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "train" / "train_config.yaml")

    dataset_yaml = REPO_ROOT / "model_pipeline" / "artifacts" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {dataset_yaml}")

    runtime_dataset_yaml = REPO_ROOT / "model_pipeline" / "artifacts" / "dataset.runtime.yaml"
    build_runtime_dataset_yaml(dataset_yaml, runtime_dataset_yaml)

    metrics_dir = REPO_ROOT / eval_cfg["outputs"]["metrics_dir"]
    ensure_dir(metrics_dir)

    mlflow.set_experiment("pretrained_yolo_evaluation")

    checkpoints_root = REPO_ROOT / train_cfg["outputs"]["checkpoints_dir"]

    for model_cfg in train_cfg["models"]:
        model_name = model_cfg["name"]
        weights_source = resolve_model_weights(model_cfg, checkpoints_root)

        print(f"Evaluating {model_name}")
        print(f"Using weights: {weights_source}")

        model = YOLO(str(weights_source))
        results = model.val(
            data=str(runtime_dataset_yaml),
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

        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("weights_source", str(weights_source))
            mlflow.log_param("split", eval_cfg["evaluation"]["split"])
            mlflow.log_param("device", eval_cfg["benchmark"]["device"])

            mlflow.log_metric("mAP50", metrics["metrics"]["mAP50"])
            mlflow.log_metric("mAP50_95", metrics["metrics"]["mAP50_95"])
            mlflow.log_metric("precision", metrics["metrics"]["precision"])
            mlflow.log_metric("recall", metrics["metrics"]["recall"])

            mlflow.log_artifact(str(out_path), artifact_path="metrics")

        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()