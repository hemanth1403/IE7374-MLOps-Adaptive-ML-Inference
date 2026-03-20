from __future__ import annotations

from pathlib import Path
import yaml

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    train_cfg_path = REPO_ROOT / "model_pipeline" / "configs" / "train" / "train_config.yaml"
    train_cfg = load_yaml(train_cfg_path)

    dataset_yaml = REPO_ROOT / "model_pipeline" / "artifacts" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {dataset_yaml}. Run load_data.py first."
        )

    training = train_cfg["training"]
    outputs = train_cfg["outputs"]

    checkpoints_dir = REPO_ROOT / outputs["checkpoints_dir"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in train_cfg["models"]:
        model_name = model_cfg["name"]
        weights = model_cfg["weights"]
        output_subdir = model_cfg["output_subdir"]

        print(f"\nStarting training for: {model_name}")
        print(f"Weights: {weights}")
        print(f"Dataset YAML: {dataset_yaml}")

        model = YOLO(weights)
        model.train(
            data=str(dataset_yaml),
            epochs=training["epochs"],
            imgsz=training["image_size"],
            batch=training["batch_size"],
            workers=training["workers"],
            device=training["device"],
            patience=training["patience"],
            project=str(checkpoints_dir),
            name=output_subdir,
            exist_ok=True,
            seed=training["seed"],
            deterministic=training["deterministic"],
        )


if __name__ == "__main__":
    main()