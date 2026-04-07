from __future__ import annotations

from pathlib import Path
import json
import yaml
import mlflow


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bucket_avg(summary: dict, bucket: str, metric: str):
    return summary.get("by_bucket", {}).get(bucket, {}).get(metric)


def main() -> None:
    eval_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "eval" / "eval_config.yaml")

    metrics_dir = REPO_ROOT / eval_cfg["outputs"]["metrics_dir"]
    benchmark_dir = REPO_ROOT / eval_cfg["outputs"]["benchmark_dir"]
    figures_dir = REPO_ROOT / eval_cfg["outputs"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("pretrained_yolo_summary")

    summary = []

    metrics_files = sorted(metrics_dir.glob("*_metrics.json"))
    benchmark_files = sorted(benchmark_dir.glob("*_benchmark.json"))

    model_names = set()
    model_names.update(f.stem.replace("_metrics", "") for f in metrics_files)
    model_names.update(f.stem.replace("_benchmark", "") for f in benchmark_files)

    for model_name in sorted(model_names):
        metrics_file = metrics_dir / f"{model_name}_metrics.json"
        benchmark_file = benchmark_dir / f"{model_name}_benchmark.json"

        metrics = load_json(metrics_file) if metrics_file.exists() else {}
        benchmark = load_json(benchmark_file) if benchmark_file.exists() else {}

        metrics_block = metrics.get("metrics", {})
        bench_summary = benchmark.get("summary", {})
        overall = bench_summary.get("overall", {})

        summary.append(
            {
                "model_name": model_name,
                "weights_source": metrics.get("weights_source") or benchmark.get("weights_source"),
                "mAP50": metrics_block.get("mAP50"),
                "mAP50_95": metrics_block.get("mAP50_95"),
                "precision": metrics_block.get("precision"),
                "recall": metrics_block.get("recall"),
                "overall_avg_latency_ms": overall.get("avg_latency_ms"),
                "overall_throughput_fps": overall.get("throughput_fps"),
                "simple_avg_latency_ms": bucket_avg(bench_summary, "simple", "avg_latency_ms"),
                "moderate_avg_latency_ms": bucket_avg(bench_summary, "moderate", "avg_latency_ms"),
                "complex_avg_latency_ms": bucket_avg(bench_summary, "complex", "avg_latency_ms"),
                "simple_throughput_fps": bucket_avg(bench_summary, "simple", "throughput_fps"),
                "moderate_throughput_fps": bucket_avg(bench_summary, "moderate", "throughput_fps"),
                "complex_throughput_fps": bucket_avg(bench_summary, "complex", "throughput_fps"),
            }
        )

    out_path = figures_dir / "model_comparison_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with mlflow.start_run(run_name="model_comparison_summary"):
        mlflow.log_param("num_models", len(summary))
        mlflow.log_artifact(str(out_path), artifact_path="summaries")

    print(f"Saved comparison summary to {out_path}")

if __name__ == "__main__":
    main()