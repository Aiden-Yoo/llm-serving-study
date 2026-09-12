#!/usr/bin/env python3
"""Dependency-free streaming benchmark for an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import statistics
import time
import urllib.request
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def request_once(endpoint: str, model: str, prompt: str, max_tokens: int, request_id: int) -> dict:
    unique_prompt = f"Request {request_id}: {prompt}"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": unique_prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    chunk_times: list[float] = []
    completion_tokens = 0
    status = "ok"
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
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
                    chunk_times.append(now)
    except Exception as exc:  # benchmark output must retain failures
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    if status == "ok" and first_token_at is None:
        status, error = "error", "stream ended without a content token"
    elif status == "ok" and completion_tokens < 1:
        status, error = "error", "stream ended without completion token usage"
    itls = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    return {
        "request_id": request_id,
        "status": status,
        "error": error,
        "e2e_s": finished - started,
        "ttft_s": first_token_at - started if first_token_at else None,
        "mean_itl_s": statistics.fmean(itls) if itls else None,
        "completion_tokens": completion_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Explain continuous batching in three concise sentences.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if args.warmup < 0:
        parser.error("warmup must not be negative")
    try:
        concurrencies = [int(v) for v in args.concurrencies.split(",")]
    except ValueError:
        parser.error("concurrencies must be comma-separated integers")
    if not concurrencies or min(concurrencies) < 1:
        parser.error("concurrencies must contain positive integers")
    if args.requests_per_level < max(concurrencies):
        parser.error("requests-per-level must be at least the largest concurrency")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.warmup):
        request_once(args.endpoint, args.model, args.prompt, min(args.max_tokens, 32), -(i + 1))

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for concurrency in concurrencies:
        level_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(
                pool.map(
                    lambda request_id: request_once(
                        args.endpoint,
                        args.model,
                        args.prompt,
                        args.max_tokens,
                        concurrency * 1_000_000 + request_id,
                    ),
                    range(args.requests_per_level),
                )
            )
        wall_s = time.perf_counter() - level_started
        for row in rows:
            row["concurrency"] = concurrency
        all_rows.extend(rows)
        ok = [row for row in rows if row["status"] == "ok"]
        ttft = [row["ttft_s"] for row in ok if row["ttft_s"] is not None]
        e2e = [row["e2e_s"] for row in ok]
        itl = [row["mean_itl_s"] for row in ok if row["mean_itl_s"] is not None]
        tokens = sum(row["completion_tokens"] for row in ok)
        summaries.append(
            {
                "concurrency": concurrency,
                "requests": len(rows),
                "successes": len(ok),
                "errors": len(rows) - len(ok),
                "wall_s": wall_s,
                "mean_completion_tokens": tokens / len(ok) if ok else 0,
                "output_tokens_per_s": tokens / wall_s if wall_s else 0,
                "ttft_p50_s": percentile(ttft, 0.50),
                "ttft_p95_s": percentile(ttft, 0.95),
                "ttft_p99_s": percentile(ttft, 0.99),
                "e2e_p50_s": percentile(e2e, 0.50),
                "e2e_p95_s": percentile(e2e, 0.95),
                "e2e_p99_s": percentile(e2e, 0.99),
                "mean_itl_p95_s": percentile(itl, 0.95),
            }
        )
        print(json.dumps(summaries[-1], ensure_ascii=False))

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    raw_path = args.output_dir / f"benchmark-{stamp}.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = args.output_dir / f"benchmark-{stamp}.json"
    summary_path.write_text(
        json.dumps({"configuration": vars(args) | {"output_dir": str(args.output_dir)}, "results": summaries}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "latest.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"raw={raw_path}\nsummary={summary_path}")


if __name__ == "__main__":
    main()
