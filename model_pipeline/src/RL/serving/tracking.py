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
        # End any stale run left over from a session that disconnected without finalizing
        if mlflow.active_run() is not None:
            mlflow.end_run()
        self._run = mlflow.start_run()

        self._adaptive_latencies: List[float] = []
        self._baseline_latencies: List[float] = []
        self._adaptive_confidences: List[float] = []
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
        conf = result["adaptive"].get("avg_confidence")
        if conf is not None:
            self._adaptive_confidences.append(float(conf))

    def finalize(self) -> None:
        """
        Compute session summary, log to MLflow, and close the active run.
        Safe to call even if no frames were recorded or if MLflow is unavailable.
        """
        try:
            if self._frame_count == 0:
                mlflow.end_run()
                return

            n = self._frame_count
            avg_adaptive = sum(self._adaptive_latencies) / n
            avg_baseline = sum(self._baseline_latencies) / n
            savings = avg_baseline - avg_adaptive

            metrics_payload: Dict[str, float] = {
                "avg_adaptive_latency_ms": round(avg_adaptive, 2),
                "avg_baseline_latency_ms": round(avg_baseline, 2),
                "latency_savings_ms":      round(savings, 2),
                "total_frames":            n,
            }
            if self._adaptive_confidences:
                avg_conf = sum(self._adaptive_confidences) / len(self._adaptive_confidences)
                metrics_payload["avg_adaptive_confidence"] = round(avg_conf, 4)

            mlflow.log_metrics(metrics_payload)

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
        except Exception as exc:
            print(f"[Tracking] MLflow logging failed (non-fatal): {exc}")
            try:
                mlflow.end_run()
            except Exception:
                pass
