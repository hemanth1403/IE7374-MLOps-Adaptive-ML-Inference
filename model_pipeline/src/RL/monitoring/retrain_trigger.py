#!/usr/bin/env python3
"""
retrain_trigger.py — Triggers automated model retraining via GitHub Actions
and sends a Slack notification.

Fires a repository_dispatch event that starts the retrain.yml workflow.
Requires no third-party libraries (stdlib urllib only).

Environment variables:
    GITHUB_TOKEN        GitHub PAT with `repo` scope (required)
    GITHUB_REPO         Owner/repo slug                (default: hemanth1403/IE7374-...)
    SLACK_WEBHOOK_URL   Incoming webhook URL           (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO      = os.getenv(
    "GITHUB_REPO",
    "hemanth1403/IE7374-MLOps-Adaptive-ML-Inference",
)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def trigger_github_retraining(reason: str, drift_score: float) -> bool:
    """POST a repository_dispatch event to start retrain.yml."""
    if not GITHUB_TOKEN:
        print("[RetrainTrigger] GITHUB_TOKEN not set — cannot trigger GitHub Actions")
        return False

    url     = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
    payload = json.dumps({
        "event_type": "retrain-triggered",
        "client_payload": {
            "reason":       reason,
            "drift_score":  drift_score,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
        },
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization":       f"Bearer {GITHUB_TOKEN}",
            "Accept":              "application/vnd.github+json",
            "Content-Type":        "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"[RetrainTrigger] GitHub dispatch accepted (HTTP {resp.status})")
            return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"[RetrainTrigger] GitHub dispatch failed: HTTP {exc.code} — {body}")
        return False
    except Exception as exc:
        print(f"[RetrainTrigger] GitHub dispatch error: {exc}")
        return False


def notify_slack(message: str) -> None:
    """Send a plain-text message to Slack via incoming webhook (best-effort)."""
    if not SLACK_WEBHOOK_URL:
        return

    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req):
            print("[RetrainTrigger] Slack notification sent")
    except Exception as exc:
        print(f"[RetrainTrigger] Slack notification failed (non-fatal): {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Trigger model retraining")
    parser.add_argument("--reason",      default="Drift detected by monitoring",
                        help="Human-readable reason for retraining")
    parser.add_argument("--drift-score", type=float, default=0.0,
                        help="Numeric drift score from drift_detector")
    args = parser.parse_args()

    print(f"[RetrainTrigger] Triggering retraining — reason: {args.reason}")

    success = trigger_github_retraining(args.reason, args.drift_score)

    slack_msg = (
        f":warning: *Adaptive Inference — Retraining Triggered*\n"
        f"*Reason:* {args.reason}\n"
        f"*Drift Score:* {args.drift_score:.3f}\n"
        f"*Repository:* `{GITHUB_REPO}`\n"
        f"GitHub Actions workflow `retrain.yml` has been dispatched."
    )
    notify_slack(slack_msg)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
