from __future__ import annotations

from pathlib import Path
import json
import random
import time
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


def resolve_split_image_path(raw_path: str, data_pipeline_root: Path) -> Path:
    p = Path(raw_path.strip())
    if p.is_absolute():
        return p
    return (data_pipeline_root / p).resolve()


def image_to_label_path(image_path: Path, data_pipeline_root: Path) -> Path:
    rel = image_path.relative_to(data_pipeline_root)
    rel_parts = list(rel.parts)

    # Convert data/processed/images/.../*.jpg -> data/processed/labels/.../*.txt
    rel_parts = [
        "labels" if part == "images" else part
        for part in rel_parts
    ]
    label_rel = Path(*rel_parts).with_suffix(".txt")
    return (data_pipeline_root / label_rel).resolve()


def count_objects_in_label(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    with label_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def complexity_bucket(obj_count: int) -> str:
    if obj_count <= 2:
        return "simple"
    if obj_count <= 7:
        return "moderate"
    return "complex"


def load_benchmark_images(
    split_name: str = "train",
    per_bucket: int = 10,
    seed: int = 42,
) -> list[dict]:
    data_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "data" / "dataset_config.yaml")
    paths = data_cfg["paths"]
    data_pipeline_root = (REPO_ROOT / data_cfg["data_pipeline_root"]).resolve()

    split_key = f"split_{split_name}"
    split_file = (REPO_ROOT / paths[split_key]).resolve()

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as f:
        raw_paths = [line.strip() for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(raw_paths)

    buckets = {"simple": [], "moderate": [], "complex": []}

    for raw_path in raw_paths:
        image_path = resolve_split_image_path(raw_path, data_pipeline_root)
        if not image_path.exists():
            continue

        label_path = image_to_label_path(image_path, data_pipeline_root)
        obj_count = count_objects_in_label(label_path)
        bucket = complexity_bucket(obj_count)

        if len(buckets[bucket]) < per_bucket:
            buckets[bucket].append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "object_count": obj_count,
                    "bucket": bucket,
                }
            )

        if all(len(v) >= per_bucket for v in buckets.values()):
            break

    selected = buckets["simple"] + buckets["moderate"] + buckets["complex"]

    if not selected:
        raise RuntimeError("No benchmark images could be selected.")

    print("Selected benchmark images:")
    for bucket_name, items in buckets.items():
        print(f"- {bucket_name}: {len(items)}")

    return selected


def summarize_timings(per_image_results: list[dict]) -> dict:
    summary = {
        "overall": {},
        "by_bucket": {},
    }

    if not per_image_results:
        return summary

    latencies = [item["latency_ms"] for item in per_image_results]
    avg_latency = sum(latencies) / len(latencies)
    total_time_sec = sum(item["elapsed_sec"] for item in per_image_results)
    throughput_fps = len(per_image_results) / total_time_sec if total_time_sec > 0 else 0.0

    summary["overall"] = {
        "num_images": len(per_image_results),
        "avg_latency_ms": avg_latency,
        "throughput_fps": throughput_fps,
    }

    bucket_names = sorted(set(item["bucket"] for item in per_image_results))
    for bucket in bucket_names:
        bucket_items = [item for item in per_image_results if item["bucket"] == bucket]
        bucket_latencies = [item["latency_ms"] for item in bucket_items]
        bucket_avg_latency = sum(bucket_latencies) / len(bucket_latencies)
        bucket_total_time = sum(item["elapsed_sec"] for item in bucket_items)
        bucket_throughput = len(bucket_items) / bucket_total_time if bucket_total_time > 0 else 0.0

        summary["by_bucket"][bucket] = {
            "num_images": len(bucket_items),
            "avg_latency_ms": bucket_avg_latency,
            "throughput_fps": bucket_throughput,
        }

    return summary


def main() -> None:
    eval_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "eval" / "eval_config.yaml")
    train_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "train" / "train_config.yaml")

    benchmark_dir = REPO_ROOT / eval_cfg["outputs"]["benchmark_dir"]
    ensure_dir(benchmark_dir)

    mlflow.set_experiment("pretrained_yolo_benchmark")

    checkpoints_root = REPO_ROOT / train_cfg["outputs"]["checkpoints_dir"]
    device = eval_cfg["benchmark"]["device"]

    # You can tune these later
    benchmark_images = load_benchmark_images(
        split_name="train",
        per_bucket=10,
        seed=42,
    )

    for model_cfg in train_cfg["models"]:
        model_name = model_cfg["name"]
        weights_source = resolve_model_weights(model_cfg, checkpoints_root)

        print(f"\nBenchmarking {model_name}")
        print(f"Using weights: {weights_source}")

        model = YOLO(str(weights_source))

        # Warm up on first image a few times
        warmup_image = benchmark_images[0]["image_path"]
        for _ in range(5):
            model.predict(source=warmup_image, device=device, verbose=False)

        per_image_results = []

        for item in benchmark_images:
            image_path = item["image_path"]

            start = time.perf_counter()
            model.predict(source=image_path, device=device, verbose=False)
            end = time.perf_counter()

            elapsed_sec = end - start
            latency_ms = elapsed_sec * 1000.0

            per_image_results.append(
                {
                    "image_path": image_path,
                    "object_count": item["object_count"],
                    "bucket": item["bucket"],
                    "elapsed_sec": elapsed_sec,
                    "latency_ms": latency_ms,
                }
            )

        summary = summarize_timings(per_image_results)

        results = {
            "model_name": model_name,
            "weights_source": str(weights_source),
            "device": device,
            "benchmark_config": {
                "split_name": "train",
                "per_bucket": 10,
                "bucket_definition": {
                    "simple": "<=2 objects",
                    "moderate": "3-7 objects",
                    "complex": ">=8 objects",
                },
            },
            "summary": summary,
            "per_image_results": per_image_results,
        }

        out_path = benchmark_dir / f"{model_name}_benchmark.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        overall = summary.get("overall", {})
        by_bucket = summary.get("by_bucket", {})

        with mlflow.start_run(run_name=f"{model_name}_benchmark"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("weights_source", str(weights_source))
            mlflow.log_param("device", device)
            mlflow.log_param("split_name", "train")
            mlflow.log_param("per_bucket", 10)

            if overall.get("avg_latency_ms") is not None:
                mlflow.log_metric("overall_avg_latency_ms", overall["avg_latency_ms"])
            if overall.get("throughput_fps") is not None:
                mlflow.log_metric("overall_throughput_fps", overall["throughput_fps"])

            for bucket in ["simple", "moderate", "complex"]:
                bucket_stats = by_bucket.get(bucket, {})
                if bucket_stats.get("avg_latency_ms") is not None:
                    mlflow.log_metric(f"{bucket}_avg_latency_ms", bucket_stats["avg_latency_ms"])
                if bucket_stats.get("throughput_fps") is not None:
                    mlflow.log_metric(f"{bucket}_throughput_fps", bucket_stats["throughput_fps"])

            mlflow.log_artifact(str(out_path), artifact_path="benchmarks")

        print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()