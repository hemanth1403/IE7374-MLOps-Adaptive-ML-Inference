from __future__ import annotations

from pathlib import Path
import csv
import json


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct_gap(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((value - baseline) / baseline) * 100.0


def main() -> None:
    figures_dir = REPO_ROOT / "model_pipeline" / "reports" / "figures"
    bias_reports_dir = REPO_ROOT / "model_pipeline" / "reports" / "bias"
    bias_reports_dir.mkdir(parents=True, exist_ok=True)

    slice_summary_path = figures_dir / "slice_comparison_summary.json"
    slice_summary = load_json(slice_summary_path)

    bucket_order = ["simple", "moderate", "complex"]
    models = slice_summary["models"]

    rows = []
    bias_summary = {
        "slice_definitions": slice_summary.get("notes", {}),
        "thresholds": {
            "latency_gap_flag_pct": 100.0,
            "throughput_gap_flag_pct": -50.0,
        },
        "models": {},
    }

    for model_name, bucket_data in models.items():
        latencies = []
        throughputs = []

        for bucket in bucket_order:
            stats = bucket_data.get(bucket, {})
            lat = stats.get("avg_latency_ms")
            fps = stats.get("throughput_fps")

            if lat is not None:
                latencies.append(lat)
            if fps is not None:
                throughputs.append(fps)

        best_latency = min(latencies) if latencies else 0.0
        best_throughput = max(throughputs) if throughputs else 0.0

        bias_summary["models"][model_name] = {
            "best_latency_ms": best_latency,
            "best_throughput_fps": best_throughput,
            "slices": {},
        }

        for bucket in bucket_order:
            stats = bucket_data.get(bucket, {})
            num_images = stats.get("num_images")
            avg_latency_ms = stats.get("avg_latency_ms")
            throughput_fps = stats.get("throughput_fps")

            latency_gap_pct = pct_gap(avg_latency_ms, best_latency) if avg_latency_ms is not None else None
            throughput_gap_pct = pct_gap(throughput_fps, best_throughput) if throughput_fps is not None else None

            latency_flag = latency_gap_pct is not None and latency_gap_pct > 100.0
            throughput_flag = throughput_gap_pct is not None and throughput_gap_pct < -50.0
            bias_flag = latency_flag or throughput_flag

            row = {
                "model_name": model_name,
                "slice": bucket,
                "num_images": num_images,
                "avg_latency_ms": round(avg_latency_ms, 4) if avg_latency_ms is not None else None,
                "throughput_fps": round(throughput_fps, 4) if throughput_fps is not None else None,
                "latency_gap_pct": round(latency_gap_pct, 2) if latency_gap_pct is not None else None,
                "throughput_gap_pct": round(throughput_gap_pct, 2) if throughput_gap_pct is not None else None,
                "latency_flag": latency_flag,
                "throughput_flag": throughput_flag,
                "bias_flag": bias_flag,
            }
            rows.append(row)
            bias_summary["models"][model_name]["slices"][bucket] = row

    csv_path = bias_reports_dir / "bias_table.csv"
    json_path = bias_reports_dir / "bias_summary.json"
    md_path = bias_reports_dir / "bias_report.md"

    fieldnames = [
        "model_name",
        "slice",
        "num_images",
        "avg_latency_ms",
        "throughput_fps",
        "latency_gap_pct",
        "throughput_gap_pct",
        "latency_flag",
        "throughput_flag",
        "bias_flag",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(bias_summary, f, indent=2)

    report_lines = [
        "# Bias and Slice Performance Report",
        "",
        "This report evaluates model-performance disparities across workload slices.",
        "For this project, the slices are scene-complexity buckets:",
        "- simple: <= 2 objects",
        "- moderate: 3-7 objects",
        "- complex: >= 8 objects",
        "",
        "A slice is flagged when:",
        "- latency is more than 100% worse than that model's best slice",
        "- throughput is more than 50% worse than that model's best slice",
        "",
        "## Summary",
        "",
    ]

    for model_name, model_info in bias_summary["models"].items():
        report_lines.append(f"### {model_name}")
        report_lines.append(f"- Best latency: {model_info['best_latency_ms']:.4f} ms")
        report_lines.append(f"- Best throughput: {model_info['best_throughput_fps']:.4f} FPS")

        flagged = []
        for bucket in bucket_order:
            slice_row = model_info["slices"][bucket]
            if slice_row["bias_flag"]:
                flagged.append(bucket)

        if flagged:
            report_lines.append(f"- Flagged slices: {', '.join(flagged)}")
        else:
            report_lines.append("- Flagged slices: none")

        report_lines.append("")

    report_lines.extend(
        [
            "## Interpretation",
            "",
            "This bias analysis is based on workload slices rather than demographic groups.",
            "In this project, slice-based disparity matters because different scene complexities",
            "can cause materially different latency and throughput behavior across YOLO tiers.",
            "These results support adaptive routing rather than relying on one fixed model.",
            "",
        ]
    )

    md_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved bias CSV to {csv_path}")
    print(f"Saved bias JSON to {json_path}")
    print(f"Saved bias report to {md_path}")


if __name__ == "__main__":
    main()