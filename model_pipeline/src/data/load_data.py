from __future__ import annotations

from pathlib import Path
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(path_str: str) -> Path:
    return (REPO_ROOT / path_str).resolve()


def get_dataset_config() -> dict:
    config_path = REPO_ROOT / "model_pipeline" / "configs" / "data" / "dataset_config.yaml"
    return load_yaml(config_path)


def get_dataset_paths() -> dict[str, Path]:
    cfg = get_dataset_config()
    paths = cfg["paths"]
    return {key: resolve_repo_path(value) for key, value in paths.items()}


def build_yolo_dataset_yaml(output_path: Path | None = None) -> Path:
    cfg = get_dataset_config()
    dataset_paths = get_dataset_paths()

    if output_path is None:
        output_path = REPO_ROOT / "model_pipeline" / "artifacts" / "dataset.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    yolo_data = {
        "path": str((REPO_ROOT / cfg["data_pipeline_root"]).resolve()),
        "train": str(dataset_paths["images_train"]),
        "val": str(dataset_paths["images_val"]),
        "test": str(dataset_paths["images_val"]),
        "nc": cfg["dataset"]["num_classes"],
        "names": [str(i) for i in range(cfg["dataset"]["num_classes"])],
    }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yolo_data, f, sort_keys=False)

    return output_path


if __name__ == "__main__":
    out = build_yolo_dataset_yaml()
    print(f"YOLO dataset YAML written to: {out}")