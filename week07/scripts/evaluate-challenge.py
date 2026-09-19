#!/usr/bin/env python3
"""Evaluate the Week 7 challenge against explicit, evidence-backed criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--flow-metrics", type=Path, required=True)
    parser.add_argument("--fairness", type=Path, required=True)
    parser.add_argument("--fairness-metrics", type=Path, required=True)
    parser.add_argument("--max-queue-requests", type=int, default=32)
    parser.add_argument("--max-queue-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    direct = load(args.direct)["results"]
    flow = load(args.flow)["results"]
    flow_metrics = load(args.flow_metrics)
    fairness = load(args.fairness)["results"]
    fairness_metrics = load(args.fairness_metrics)

    flow_statuses = flow["overall"]["http_status_counts"]
    rejections = int(flow_statuses.get("429", 0)) + int(flow_statuses.get("503", 0))
    interactive = flow["by_class"]["interactive"]
    batch = flow["by_class"]["batch"]
    direct_interactive = direct["by_class"]["interactive"]
    noisy = fairness["by_fairness_id"]["noisy-tenant"]
    quiet = fairness["by_fairness_id"]["quiet-tenant"]
    fairness_offered = fairness["overall"]["offered"]
    fairness_completed = fairness["overall"]["completed"]
    quiet_offered_share = quiet["offered"] / fairness_offered if fairness_offered else 0
    quiet_completed_share = quiet["completed"] / fairness_completed if fairness_completed else 0
    quiet_share_preservation = (
        quiet_completed_share / quiet_offered_share if quiet_offered_share else 0
    )
    direct_ttft_p95 = direct_interactive["ttft_p95_s"]
    flow_ttft_p95 = interactive["ttft_p95_s"]
    flow_queue_size = flow_metrics.get("max_queue_size")
    fairness_queue_size = fairness_metrics.get("max_queue_size")
    flow_queue_bytes = flow_metrics.get("max_queue_bytes")
    fairness_queue_bytes = fairness_metrics.get("max_queue_bytes")
    flow_saturation = flow_metrics.get("max_pool_saturation")

    checks = {
        "open_loop_launch_lag_p99_under_100ms": {
            "pass": max(
                direct["overall"]["launch_lag_p99_s"],
                flow["overall"]["launch_lag_p99_s"],
                fairness["overall"]["launch_lag_p99_s"],
            ) < 0.1,
            "observed_s": max(
                direct["overall"]["launch_lag_p99_s"],
                flow["overall"]["launch_lag_p99_s"],
                fairness["overall"]["launch_lag_p99_s"],
            ),
        },
        "central_queue_observed": {
            "pass": flow_queue_size is not None and flow_queue_size > 0,
            "observed_requests": flow_queue_size,
        },
        "queue_bound_respected": {
            "pass": all(value is not None for value in (flow_queue_size, fairness_queue_size))
            and max(flow_queue_size, fairness_queue_size) <= args.max_queue_requests,
            "configured_max_requests": args.max_queue_requests,
            "observed_max_requests": max(
                (value for value in (flow_queue_size, fairness_queue_size) if value is not None),
                default=None,
            ),
        },
        "queue_byte_bound_respected": {
            "pass": all(value is not None for value in (flow_queue_bytes, fairness_queue_bytes))
            and max(flow_queue_bytes, fairness_queue_bytes) <= args.max_queue_bytes,
            "configured_max_bytes": args.max_queue_bytes,
            "observed_max_bytes": max(
                (value for value in (flow_queue_bytes, fairness_queue_bytes) if value is not None),
                default=None,
            ),
        },
        "saturation_signal_observed": {
            "pass": flow_saturation is not None and flow_saturation >= 1.0,
            "observed": flow_saturation,
        },
        "metric_scrapes_clean": {
            "pass": not flow_metrics.get("errors") and not fairness_metrics.get("errors"),
            "flow_errors": flow_metrics.get("errors", []),
            "fairness_errors": fairness_metrics.get("errors", []),
        },
        "overload_shed_with_explicit_status": {
            "pass": rejections > 0,
            "http_429_or_503": rejections,
            "drop_reasons": flow["overall"]["drop_reason_counts"],
        },
        "interactive_completion_rate_not_below_batch": {
            "pass": interactive["completion_rate"] >= batch["completion_rate"],
            "interactive": interactive["completion_rate"],
            "batch": batch["completion_rate"],
        },
        "interactive_ttft_p95_better_than_direct": {
            "pass": direct_ttft_p95 is not None
            and flow_ttft_p95 is not None
            and flow_ttft_p95 < direct_ttft_p95,
            "direct_s": direct_ttft_p95,
            "flow_control_s": flow_ttft_p95,
        },
        "quiet_tenant_not_starved": {
            "pass": quiet["completed"] > 0 and quiet_share_preservation >= 0.8,
            "quiet_completed": quiet["completed"],
            "quiet_offered_share": quiet_offered_share,
            "quiet_completed_share": quiet_completed_share,
            "offered_share_preservation": quiet_share_preservation,
            "quiet_completion_rate": quiet["completion_rate"],
            "noisy_completion_rate": noisy["completion_rate"],
        },
    }
    verdict = {
        "challenge": "Protect a single-GPU interactive SLO with llm-d Flow Control",
        "verdict": "PASS" if all(item["pass"] for item in checks.values()) else "FAIL",
        "checks": checks,
        "limits": [
            "One backend cannot demonstrate load-aware endpoint selection or failover.",
            "Flow Control moves overload waiting to the EPP; it does not add GPU capacity.",
            "Fairness evidence is workload-specific and is not a production SLO guarantee.",
        ],
    }
    args.output.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    if verdict["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
