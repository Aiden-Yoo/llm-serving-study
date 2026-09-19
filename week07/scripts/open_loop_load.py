#!/usr/bin/env python3
"""Dependency-free open-loop load generator for the Week 7 Flow Control challenge."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "request_id",
    "traffic_class",
    "fairness_id",
    "objective",
    "max_tokens",
    "scheduled_at_s",
    "sent_at_s",
    "launch_lag_s",
    "http_status",
    "outcome",
    "drop_reason",
    "retry_after",
    "ttft_s",
    "e2e_s",
    "mean_itl_s",
    "completion_tokens",
    "error",
]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def workload(request_id: int, scenario: str) -> dict[str, str | int]:
    if scenario == "mixed":
        pattern = (
            ("interactive", "tenant-a", "premium-traffic", 64),
            ("batch", "batch-a", "best-effort-traffic", 128),
            ("interactive", "tenant-a", "premium-traffic", 64),
            ("batch", "batch-b", "best-effort-traffic", 128),
            ("batch", "batch-a", "best-effort-traffic", 128),
            ("interactive", "tenant-b", "premium-traffic", 64),
            ("batch", "batch-b", "best-effort-traffic", 128),
            ("interactive", "tenant-a", "premium-traffic", 64),
            ("batch", "batch-a", "best-effort-traffic", 128),
            ("batch", "batch-b", "best-effort-traffic", 128),
        )
    elif scenario == "fairness":
        pattern = tuple(
            [("standard", "noisy-tenant", "standard-traffic", 96)] * 5
            + [("standard", "quiet-tenant", "standard-traffic", 96)]
            + [("standard", "noisy-tenant", "standard-traffic", 96)] * 4
        )
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    traffic_class, fairness_id, objective, max_tokens = pattern[request_id % len(pattern)]
    return {
        "traffic_class": traffic_class,
        "fairness_id": fairness_id,
        "objective": objective,
        "max_tokens": max_tokens,
    }


def request_once(
    *,
    endpoint: str,
    model: str,
    request_id: int,
    spec: dict[str, str | int],
    schedule_origin: float,
    scheduled_at_s: float,
    timeout_s: float,
) -> dict:
    sent = time.perf_counter()
    prompt = (
        "Explain one practical benefit of centralized LLM admission control in two concise "
        f"sentences. Request identifier: {request_id}."
    )
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": spec["max_tokens"],
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-llm-d-inference-objective": str(spec["objective"]),
            "x-llm-d-inference-fairness-id": str(spec["fairness_id"]),
        },
        method="POST",
    )
    first_token_at = None
    token_chunk_times: list[float] = []
    completion_tokens = 0
    http_status = 0
    outcome = "error"
    drop_reason = ""
    retry_after = ""
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            http_status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or {}
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                choices = chunk.get("choices") or []
                content = choices[0].get("delta", {}).get("content") if choices else None
                if content:
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    token_chunk_times.append(now)
        if first_token_at is None:
            error = "stream ended without a content token"
        elif completion_tokens < 1:
            error = "stream ended without completion token usage"
        else:
            outcome = "completed"
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        drop_reason = exc.headers.get("x-llm-d-request-dropped-reason", "")
        retry_after = exc.headers.get("Retry-After", "")
        detail = exc.read(1024).decode("utf-8", "replace").strip()
        outcome = "rejected" if exc.code in (429, 503) else "error"
        error = detail or f"HTTP {exc.code}"
    except Exception as exc:  # request-level failures are benchmark evidence
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    itls = [b - a for a, b in zip(token_chunk_times, token_chunk_times[1:])]
    sent_at_s = sent - schedule_origin
    return {
        "request_id": request_id,
        **spec,
        "scheduled_at_s": scheduled_at_s,
        "sent_at_s": sent_at_s,
        "launch_lag_s": sent_at_s - scheduled_at_s,
        "http_status": http_status,
        "outcome": outcome,
        "drop_reason": drop_reason,
        "retry_after": retry_after,
        "ttft_s": first_token_at - sent if first_token_at else None,
        "e2e_s": finished - sent,
        "mean_itl_s": statistics.fmean(itls) if itls else None,
        "completion_tokens": completion_tokens,
        "error": error,
    }


def parse_metric_sample(text: str, elapsed_s: float) -> dict:
    queue_size = 0.0
    queue_bytes = 0.0
    saturation = 0.0
    request_counters: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            metric, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        name = metric.split("{", 1)[0]
        if name == "llm_d_epp_flow_control_queue_size":
            queue_size += value
        elif name == "llm_d_epp_flow_control_queue_bytes":
            queue_bytes += value
        elif name == "llm_d_epp_flow_control_pool_saturation":
            saturation = max(saturation, value)
        elif name == "llm_d_epp_flow_control_requests_total":
            request_counters[metric] = value
    return {
        "elapsed_s": elapsed_s,
        "queue_size": queue_size,
        "queue_bytes": queue_bytes,
        "pool_saturation": saturation,
        "request_counters": request_counters,
    }


def sample_metrics(
    endpoint: str,
    stop: threading.Event,
    samples: list[dict],
    errors: list[str],
    started: float,
) -> None:
    while not stop.is_set():
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                samples.append(
                    parse_metric_sample(response.read().decode("utf-8", "replace"), time.perf_counter() - started)
                )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        stop.wait(0.25)


def group_summary(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["outcome"] == "completed"]
    status_counts = Counter(str(row["http_status"]) for row in rows)
    drop_reasons = Counter(row["drop_reason"] or "none" for row in rows if row["outcome"] == "rejected")
    ttft = [float(row["ttft_s"]) for row in completed if row["ttft_s"] is not None]
    e2e = [float(row["e2e_s"]) for row in completed]
    itl = [float(row["mean_itl_s"]) for row in completed if row["mean_itl_s"] is not None]
    return {
        "offered": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else 0,
        "http_status_counts": dict(sorted(status_counts.items())),
        "drop_reason_counts": dict(sorted(drop_reasons.items())),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in completed),
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "ttft_p99_s": percentile(ttft, 0.99),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p95_s": percentile(e2e, 0.95),
        "e2e_p99_s": percentile(e2e, 0.99),
        "mean_itl_p95_s": percentile(itl, 0.95),
    }


def summarize(rows: list[dict], wall_s: float) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    by_fairness: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[str(row["traffic_class"])].append(row)
        by_fairness[str(row["fairness_id"])].append(row)
    overall = group_summary(rows)
    overall["wall_s"] = wall_s
    overall["output_tokens_per_s"] = overall["completion_tokens"] / wall_s if wall_s else 0
    overall["launch_lag_p99_s"] = percentile([float(row["launch_lag_s"]) for row in rows], 0.99)
    return {
        "overall": overall,
        "by_class": {name: group_summary(group) for name, group in sorted(by_class.items())},
        "by_fairness_id": {name: group_summary(group) for name, group in sorted(by_fairness.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--endpoint-label", required=True)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--scenario", choices=("mixed", "fairness"), required=True)
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--metrics-endpoint")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if not math.isfinite(args.rps) or args.rps <= 0:
        parser.error("rps must be a positive finite number")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be a positive finite number")
    total_requests = max(1, round(args.rps * args.duration))
    if args.max_workers < 1:
        parser.error("max-workers must be positive")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics_samples: list[dict] = []
    metrics_errors: list[str] = []
    metrics_stop = threading.Event()
    schedule_origin = time.perf_counter() + 0.25
    metrics_thread = None
    if args.metrics_endpoint:
        metrics_thread = threading.Thread(
            target=sample_metrics,
            args=(args.metrics_endpoint, metrics_stop, metrics_samples, metrics_errors, schedule_origin),
            daemon=True,
        )
        metrics_thread.start()

    futures: list[concurrent.futures.Future] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for request_id in range(total_requests):
            scheduled_at_s = request_id / args.rps
            delay = schedule_origin + scheduled_at_s - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(
                pool.submit(
                    request_once,
                    endpoint=args.endpoint,
                    model=args.model,
                    request_id=request_id,
                    spec=workload(request_id, args.scenario),
                    schedule_origin=schedule_origin,
                    scheduled_at_s=scheduled_at_s,
                    timeout_s=args.timeout,
                )
            )
        rows = [future.result() for future in futures]
    wall_s = time.perf_counter() - schedule_origin

    metrics_stop.set()
    if metrics_thread:
        metrics_thread.join(timeout=10)
        if metrics_thread.is_alive():
            raise RuntimeError("metrics sampler did not stop within 10 seconds")
    rows.sort(key=lambda row: int(row["request_id"]))

    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "endpoint_label": args.endpoint_label,
            "model": args.model,
            "scenario": args.scenario,
            "offered_rps": args.rps,
            "duration_s": args.duration,
            "total_requests": total_requests,
            "timeout_s": args.timeout,
            "max_workers": args.max_workers,
        },
        "results": summarize(rows, wall_s),
    }
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics_path = None
    if args.metrics_endpoint:
        metrics_path = args.output_prefix.parent / f"{args.output_prefix.name}-metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "sample_interval_s": 0.25,
                    "samples": metrics_samples,
                    "errors": metrics_errors,
                    "max_queue_size": max((sample["queue_size"] for sample in metrics_samples), default=None),
                    "max_queue_bytes": max((sample["queue_bytes"] for sample in metrics_samples), default=None),
                    "max_pool_saturation": max(
                        (sample["pool_saturation"] for sample in metrics_samples), default=None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(summary["results"]["overall"], ensure_ascii=False))
    print(f"csv={csv_path}")
    print(f"summary={json_path}")
    if metrics_path:
        print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
