#!/usr/bin/env python3
"""
drift_detector.py — Model performance monitoring and drift detection.

Pulls recent session metrics from MLflow, detects performance decay and
distribution drift using Evidently AI, writes JSON/HTML reports, and
optionally triggers model retraining via GitHub Actions.

Usage:
    python monitoring/drift_detector.py [--trigger-on-drift] [--report-dir /path]

Environment variables:
    MLFLOW_TRACKING_URI    MLflow server URI          (default: http://localhost:5000)
    DRIFT_THRESHOLD        Evidently drift score to trigger retrain (default: 0.3)
    MIN_SESSIONS           Min finished sessions before checks run  (default: 10)
    LATENCY_THRESHOLD_MS   Max acceptable avg adaptive latency ms   (default: 150)
    SAVINGS_THRESHOLD_MS   Min acceptable latency savings ms        (default: 5)
    CONFIDENCE_THRESHOLD   Min acceptable avg detection confidence  (default: 0.30)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import mlflow

MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DRIFT_THRESHOLD      = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
MIN_SESSIONS         = int(os.getenv("MIN_SESSIONS", "10"))
LATENCY_THRESHOLD_MS = float(os.getenv("LATENCY_THRESHOLD_MS", "150.0"))
SAVINGS_THRESHOLD_MS = float(os.getenv("SAVINGS_THRESHOLD_MS", "5.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))

EXPERIMENT_NAME  = "adaptive_inference"
REFERENCE_WINDOW = 50   # older runs that form the reference distribution
CURRENT_WINDOW   = 20   # most-recent runs compared against the reference


# ──────────────────────────────────────────────────────────────────────────────
# MLflow helpers
# ──────────────────────────────────────────────────────────────────────────────

def fetch_session_metrics(client: mlflow.MlflowClient, experiment_id: str) -> pd.DataFrame:
    """Return a DataFrame of per-session metrics sorted oldest-first."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time ASC"],
        max_results=500,
    )

    records = []
    for run in runs:
        m = run.data.metrics
        if m.get("total_frames", 0) < 10:
            continue  # skip empty / truncated sessions
        records.append({
            "run_id":                    run.info.run_id,
            "start_time":                run.info.start_time,
            "avg_adaptive_latency_ms":   m.get("avg_adaptive_latency_ms",   np.nan),
            "avg_baseline_latency_ms":   m.get("avg_baseline_latency_ms",   np.nan),
            "latency_savings_ms":        m.get("latency_savings_ms",        np.nan),
            "total_frames":              m.get("total_frames",              0),
            "model_pct_nano":            m.get("model_pct_nano",            np.nan),
            "model_pct_small":           m.get("model_pct_small",           np.nan),
            "model_pct_large":           m.get("model_pct_large",           np.nan),
            "avg_adaptive_confidence":   m.get("avg_adaptive_confidence",   np.nan),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Threshold-based decay checks
# ──────────────────────────────────────────────────────────────────────────────

def threshold_checks(current_df: pd.DataFrame) -> Dict[str, Any]:
    """Compare rolling averages against hard thresholds."""
    if current_df.empty:
        return {"status": "no_data", "violations": []}

    violations = []
    avg_latency    = current_df["avg_adaptive_latency_ms"].mean()
    avg_savings    = current_df["latency_savings_ms"].mean()
    avg_confidence = current_df["avg_adaptive_confidence"].dropna().mean()

    if not np.isnan(avg_latency) and avg_latency > LATENCY_THRESHOLD_MS:
        violations.append({
            "metric":      "avg_adaptive_latency_ms",
            "value":       round(float(avg_latency), 2),
            "threshold":   LATENCY_THRESHOLD_MS,
            "description": "Average adaptive latency exceeds threshold",
        })

    if not np.isnan(avg_savings) and avg_savings < SAVINGS_THRESHOLD_MS:
        violations.append({
            "metric":      "latency_savings_ms",
            "value":       round(float(avg_savings), 2),
            "threshold":   SAVINGS_THRESHOLD_MS,
            "description": "RL routing not saving enough latency vs baseline",
        })

    if not np.isnan(avg_confidence) and avg_confidence < CONFIDENCE_THRESHOLD:
        violations.append({
            "metric":      "avg_adaptive_confidence",
            "value":       round(float(avg_confidence), 4),
            "threshold":   CONFIDENCE_THRESHOLD,
            "description": "Detection confidence below acceptable threshold",
        })

    return {
        "status": "violations_found" if violations else "ok",
        "violations": violations,
        "summary": {
            "avg_latency_ms":  round(float(avg_latency),    2) if not np.isnan(avg_latency)    else None,
            "avg_savings_ms":  round(float(avg_savings),    2) if not np.isnan(avg_savings)    else None,
            "avg_confidence":  round(float(avg_confidence), 4) if not np.isnan(avg_confidence) else None,
            "sessions_evaluated": len(current_df),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evidently distribution drift report
# ──────────────────────────────────────────────────────────────────────────────

def evidently_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_path: Path,
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset on session-metric distributions.
    Writes an HTML report to report_path and returns a drift summary dict.
    Returns a safe no-op dict if Evidently is not installed.
    """
    try:
        from evidently import ColumnMapping
        from evidently.report import Report
        from evidently.metric_presets import DataDriftPreset
    except ImportError:
        print("[DriftDetector] evidently not installed — skipping distribution drift report")
        return {"status": "evidently_not_installed", "drift_detected": False, "drift_score": 0.0}

    metric_cols = [
        "avg_adaptive_latency_ms",
        "latency_savings_ms",
        "model_pct_nano",
        "model_pct_small",
        "model_pct_large",
    ]
    if "avg_adaptive_confidence" in reference_df.columns:
        metric_cols.append("avg_adaptive_confidence")

    ref = reference_df[metric_cols].dropna()
    cur = current_df[metric_cols].dropna()

    if len(ref) < 10 or len(cur) < 5:
        return {"status": "insufficient_data", "drift_detected": False, "drift_score": 0.0}

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=ColumnMapping())
    report.save_html(str(report_path))

    result_dict = report.as_dict()
    drifted_cols = 0
    total_cols   = 0
    for m in result_dict.get("metrics", []):
        if m.get("metric") == "DataDriftTable":
            for _col, col_data in m.get("result", {}).get("drift_by_columns", {}).items():
                total_cols += 1
                if col_data.get("drift_detected", False):
                    drifted_cols += 1

    drift_score    = drifted_cols / total_cols if total_cols > 0 else 0.0
    drift_detected = drift_score >= DRIFT_THRESHOLD

    return {
        "status":          "completed",
        "drift_detected":  drift_detected,
        "drift_score":     round(drift_score, 3),
        "drifted_columns": drifted_cols,
        "total_columns":   total_cols,
        "report_path":     str(report_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Drift detector for Adaptive ML Inference")
    parser.add_argument("--trigger-on-drift", action="store_true",
                        help="Invoke retrain_trigger.py when drift or decay is detected")
    parser.add_argument("--report-dir", default="/tmp/drift-reports",
                        help="Directory to write HTML/JSON reports")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"[DriftDetector] Experiment '{EXPERIMENT_NAME}' not found — nothing to check")
        return 0

    df = fetch_session_metrics(client, experiment.experiment_id)
    if len(df) < MIN_SESSIONS:
        print(f"[DriftDetector] Only {len(df)} sessions (need {MIN_SESSIONS}) — skipping")
        return 0

    current_df   = df.tail(CURRENT_WINDOW)
    reference_df = df.iloc[
        max(0, len(df) - REFERENCE_WINDOW - CURRENT_WINDOW) : len(df) - CURRENT_WINDOW
    ]

    # Threshold checks (always run)
    threshold_result = threshold_checks(current_df)
    print(f"[DriftDetector] Threshold check: {threshold_result['status']}")
    for v in threshold_result["violations"]:
        print(f"  VIOLATION: {v['description']} (value={v['value']}, threshold={v['threshold']})")

    # Evidently distribution drift report
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    html_path   = report_dir / f"drift_report_{timestamp}.html"

    drift_result = evidently_drift_report(reference_df, current_df, html_path)
    print(
        f"[DriftDetector] Evidently: status={drift_result['status']} "
        f"score={drift_result.get('drift_score', 'N/A')} "
        f"detected={drift_result.get('drift_detected', False)}"
    )

    retrain_needed = (
        threshold_result["status"] == "violations_found"
        or drift_result.get("drift_detected", False)
    )

    # Write combined JSON report
    report = {
        "timestamp":          timestamp,
        "sessions_analyzed":  len(df),
        "threshold_check":    threshold_result,
        "evidently_drift":    drift_result,
        "retrain_needed":     retrain_needed,
    }
    json_path = report_dir / f"drift_report_{timestamp}.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"[DriftDetector] Report written → {json_path}")

    if retrain_needed:
        print("[DriftDetector] DRIFT/DECAY DETECTED — retraining needed")
        if args.trigger_on_drift:
            reason = (
                "; ".join(v["description"] for v in threshold_result["violations"])
                or f"Evidently drift score {drift_result.get('drift_score', 0):.3f} >= {DRIFT_THRESHOLD}"
            )
            trigger_script = Path(__file__).parent / "retrain_trigger.py"
            subprocess.run(
                [
                    sys.executable, str(trigger_script),
                    "--reason",      reason,
                    "--drift-score", str(drift_result.get("drift_score", 0.0)),
                ],
                check=False,
            )
        return 1  # non-zero so the K8s CronJob marks the Job as Failed (visible in alerts)

    print("[DriftDetector] No drift detected — model healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
