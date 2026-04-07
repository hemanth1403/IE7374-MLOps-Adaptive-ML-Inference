"""
tracking.py — MLflow session tracking for the Adaptive ML Inference System.

Accumulates per-frame dual-path results during a WebSocket session and logs
a summary to MLflow when the session ends (finalize()).

Logged metrics
--------------
avg_adaptive_latency_ms   — mean latency of the RL-selected model
avg_baseline_latency_ms   — mean latency of the fixed-Small baseline
latency_savings_ms        — baseline_avg − adaptive_avg  (positive = faster)
total_frames              — total frames processed in the session
model_pct_nano/small/large — percentage of frames routed to each YOLO variant

Logged params
-------------
model_distribution        — raw counts per model as a string
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import mlflow


class SessionTracker:
    """
    One instance per WebSocket connection.

    Usage
    -----
        tracker = SessionTracker()
        for frame_result in session:
            tracker.record(frame_result)
        tracker.finalize()   # called in the WebSocket disconnect handler
    """

    def __init__(self, experiment_name: str = "adaptive_inference") -> None:
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run()

        self._adaptive_latencies: List[float] = []
        self._baseline_latencies: List[float] = []
        self._model_counts: Dict[str, int] = defaultdict(int)
        self._frame_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────

    def record(self, result: Dict[str, Any]) -> None:
        """
        Accumulate one frame's dual-path inference result.

        Parameters
        ----------
        result : dict
            The dict returned by AdaptiveInferenceSystem.infer().
            Expected keys: "adaptive" and "baseline", each with
            "latency_ms" and "model_name".
        """
        self._frame_count += 1
        self._adaptive_latencies.append(result["adaptive"]["latency_ms"])
        self._baseline_latencies.append(result["baseline"]["latency_ms"])
        self._model_counts[result["adaptive"]["model_name"]] += 1

    def finalize(self) -> None:
        """
        Compute session summary, log to MLflow, and close the active run.
        Safe to call even if no frames were recorded.
        """
        if self._frame_count == 0:
            mlflow.end_run()
            return

        n = self._frame_count
        avg_adaptive = sum(self._adaptive_latencies) / n
        avg_baseline = sum(self._baseline_latencies) / n
        savings = avg_baseline - avg_adaptive

        mlflow.log_metrics(
            {
                "avg_adaptive_latency_ms": round(avg_adaptive, 2),
                "avg_baseline_latency_ms": round(avg_baseline, 2),
                "latency_savings_ms":      round(savings, 2),
                "total_frames":            n,
            }
        )

        # Model selection distribution (as percentage of frames)
        for model_name, count in self._model_counts.items():
            pct = (count / n) * 100.0
            mlflow.log_metric(f"model_pct_{model_name.lower()}", round(pct, 2))

        mlflow.log_param("model_distribution", str(dict(self._model_counts)))

        mlflow.end_run()

        print(
            f"[Tracking] Session complete — {n} frames | "
            f"adaptive avg {avg_adaptive:.1f} ms | "
            f"baseline avg {avg_baseline:.1f} ms | "
            f"savings {savings:+.1f} ms | "
            f"distribution {dict(self._model_counts)}"
        )
