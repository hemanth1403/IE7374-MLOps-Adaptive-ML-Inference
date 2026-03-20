from __future__ import annotations

from pathlib import Path
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

    metrics_dir = REPO_ROOT / eval_cfg["outputs"]["metrics_dir"]
    benchmark_dir = REPO_ROOT / eval_cfg["outputs"]["benchmark_dir"]
    figures_dir = REPO_ROOT / eval_cfg["outputs"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for metrics_file in sorted(metrics_dir.glob("*_metrics.json")):
        model_name = metrics_file.stem.replace("_metrics", "")
        benchmark_file = benchmark_dir / f"{model_name}_benchmark.json"

        metrics = load_json(metrics_file)
        benchmark = load_json(benchmark_file) if benchmark_file.exists() else {}

        summary.append({
            "model_name": model_name,
            "weights_source": metrics.get("weights_source"),
            "mAP50": metrics["metrics"]["mAP50"],
            "mAP50_95": metrics["metrics"]["mAP50_95"],
            "precision": metrics["metrics"]["precision"],
            "recall": metrics["metrics"]["recall"],
            "latency_ms": benchmark.get("latency_ms"),
            "throughput_fps": benchmark.get("throughput_fps"),
        })

    out_path = figures_dir / "model_comparison_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved comparison summary to {out_path}")


if __name__ == "__main__":
    main()