from __future__ import annotations

from pathlib import Path
import json
import time
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


def get_sample_image() -> Path:
    data_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "data" / "dataset_config.yaml")
    split_train = REPO_ROOT / data_cfg["paths"]["split_train"]

    with split_train.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise RuntimeError("train.txt is empty")

    sample_path = Path(first_line)

    if sample_path.is_absolute():
        return sample_path

    data_pipeline_root = REPO_ROOT / data_cfg["data_pipeline_root"]
    return (data_pipeline_root / sample_path).resolve()


def main() -> None:
    eval_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "eval" / "eval_config.yaml")
    train_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "train" / "train_config.yaml")

    benchmark_dir = REPO_ROOT / eval_cfg["outputs"]["benchmark_dir"]
    ensure_dir(benchmark_dir)

    checkpoints_root = REPO_ROOT / train_cfg["outputs"]["checkpoints_dir"]
    sample_image = get_sample_image()

    warmup_runs = eval_cfg["benchmark"]["warmup_runs"]
    timed_runs = eval_cfg["benchmark"]["timed_runs"]
    device = eval_cfg["benchmark"]["device"]

    for model_cfg in train_cfg["models"]:
        model_name = model_cfg["name"]
        weights_source = resolve_model_weights(model_cfg, checkpoints_root)

        print(f"Benchmarking {model_name}")
        print(f"Using weights: {weights_source}")

        model = YOLO(str(weights_source))

        for _ in range(warmup_runs):
            model.predict(source=str(sample_image), device=device, verbose=False)

        start = time.perf_counter()
        for _ in range(timed_runs):
            model.predict(source=str(sample_image), device=device, verbose=False)
        end = time.perf_counter()

        total_time = end - start
        latency_ms = (total_time / timed_runs) * 1000.0
        throughput_fps = timed_runs / total_time if total_time > 0 else 0.0

        results = {
            "model_name": model_name,
            "weights_source": str(weights_source),
            "sample_image": str(sample_image),
            "timed_runs": timed_runs,
            "latency_ms": latency_ms,
            "throughput_fps": throughput_fps,
        }

        out_path = benchmark_dir / f"{model_name}_benchmark.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()