from __future__ import annotations

from pathlib import Path
import csv
import json
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    eval_cfg = load_yaml(REPO_ROOT / "model_pipeline" / "configs" / "eval" / "eval_config.yaml")

    benchmark_dir = REPO_ROOT / eval_cfg["outputs"]["benchmark_dir"]
    figures_dir = REPO_ROOT / eval_cfg["outputs"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    benchmark_files = sorted(benchmark_dir.glob("*_benchmark.json"))

    rows = []
    summary = {
        "models": {},
        "buckets": ["simple", "moderate", "complex"],
        "notes": {
            "simple": "<=2 objects",
            "moderate": "3-7 objects",
            "complex": ">=8 objects",
        },
    }

    for benchmark_file in benchmark_files:
        benchmark = load_json(benchmark_file)
        model_name = benchmark.get("model_name", benchmark_file.stem.replace("_benchmark", ""))
        weights_source = benchmark.get("weights_source")
        by_bucket = benchmark.get("summary", {}).get("by_bucket", {})

        summary["models"][model_name] = {}

        for bucket in ["simple", "moderate", "complex"]:
            bucket_stats = by_bucket.get(bucket, {})

            row = {
                "model_name": model_name,
                "weights_source": weights_source,
                "complexity_bucket": bucket,
                "num_images": bucket_stats.get("num_images"),
                "avg_latency_ms": bucket_stats.get("avg_latency_ms"),
                "throughput_fps": bucket_stats.get("throughput_fps"),
            }
            rows.append(row)
            summary["models"][model_name][bucket] = row

    csv_path = figures_dir / "slice_comparison_table.csv"
    json_path = figures_dir / "slice_comparison_summary.json"

    fieldnames = [
        "model_name",
        "weights_source",
        "complexity_bucket",
        "num_images",
        "avg_latency_ms",
        "throughput_fps",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved slice comparison CSV to {csv_path}")
    print(f"Saved slice comparison JSON to {json_path}")


if __name__ == "__main__":
    main()