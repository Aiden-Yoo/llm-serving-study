#!/usr/bin/env python3
"""Compare a blocking gateway, an async gateway, and direct vLLM requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import platform
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Measurement:
    endpoint: str
    label: str
    e2e_s: float
    prompt_tokens: int
    output_tokens: int


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def post_json(url: str, payload: dict[str, Any], label: str) -> Measurement:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {label}: {body}") from exc
    finished = time.perf_counter()
    usage = result.get("usage") or {}
    return Measurement(
        endpoint=url,
        label=label,
        e2e_s=finished - started,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
    )


def make_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "간결하고 정확하게 답하라."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": False,
    }


def run_concurrent(
    endpoint: str,
    endpoint_name: str,
    model: str,
    concurrency: int,
    request_count: int,
    max_tokens: int,
) -> tuple[list[Measurement], float]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for index in range(request_count):
            prompt = (
                f"실험 {endpoint_name}, 동시성 {concurrency}, 요청 {index}. "
                "continuous batching이 처리량을 높이는 이유를 두 문장으로 설명하라."
            )
            futures.append(
                executor.submit(
                    post_json,
                    endpoint,
                    make_payload(model, prompt, max_tokens),
                    f"{endpoint_name}-{concurrency}-{index}",
                )
            )
        measurements = [future.result() for future in futures]
    return measurements, time.perf_counter() - started


def summarize(
    endpoint_name: str,
    concurrency: int,
    measurements: list[Measurement],
    wall_s: float,
) -> dict[str, Any]:
    latencies = [item.e2e_s for item in measurements]
    output_tokens = sum(item.output_tokens for item in measurements)
    prompt_tokens = sum(item.prompt_tokens for item in measurements)
    return {
        "endpoint": endpoint_name,
        "concurrency": concurrency,
        "requests": len(measurements),
        "wall_s": round(wall_s, 4),
        "requests_per_s": round(len(measurements) / wall_s, 4),
        "output_tokens_per_s": round(output_tokens / wall_s, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "e2e_mean_s": round(statistics.fmean(latencies), 4),
        "e2e_p50_s": round(percentile(latencies, 0.50), 4),
        "e2e_p95_s": round(percentile(latencies, 0.95), 4),
        "measurements": [asdict(item) for item in measurements],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    settings = result["metadata"]["server_settings"]
    lines = [
        f"# R9700 serving-mode benchmark: {result['metadata']['profile']}",
        "",
        f"- Timestamp: `{result['metadata']['timestamp']}`",
        f"- Model: `{result['metadata']['model']}`",
        f"- Profile: `{result['metadata']['profile']}`",
        f"- `max_num_seqs`: `{settings['max_num_seqs']}`",
        f"- `max_num_batched_tokens`: `{settings['max_num_batched_tokens']}`",
        f"- Output length: `{result['metadata']['max_tokens']} tokens/request`",
        "",
        "| Endpoint | Concurrency | Requests | RPS | output tok/s | E2E p50/p95 (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["runs"]:
        lines.append(
            f"| {row['endpoint']} | {row['concurrency']} | {row['requests']} | "
            f"{row['requests_per_s']:.2f} | {row['output_tokens_per_s']:.1f} | "
            f"{row['e2e_p50_s']:.3f}/{row['e2e_p95_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "- `blocking`: `async def` 내부에서 동기 HTTP 호출을 직접 실행하는 의도적 negative control",
            "- `threaded`: 같은 동기 호출을 `asyncio.to_thread()`로 격리",
            "- `async`: `httpx.AsyncClient`로 upstream까지 non-blocking 호출",
            "- `direct`: gateway 없이 vLLM OpenAI endpoint 직접 호출",
            "",
            "> 단일 실행 결과이며 보편적인 성능 수치가 아니라 같은 장비·모델에서 설계 차이를 비교하기 위한 결과다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen3-4b-vllm")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vllm_url = args.vllm_url.rstrip("/")
    gateway_url = args.gateway_url.rstrip("/")
    endpoints = {
        "blocking": f"{gateway_url}/blocking",
        "threaded": f"{gateway_url}/threaded",
        "async": f"{gateway_url}/async",
        "direct": f"{vllm_url}/v1/chat/completions",
    }

    result: dict[str, Any] = {
        "metadata": {
            "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "profile": args.profile,
            "model": args.model,
            "model_path": "./models/Qwen3-4B-Instruct-2507",
            "python": platform.python_version(),
            "max_tokens": args.max_tokens,
            "server_settings": {
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.80,
                "max_model_len": 16384,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "prefix_caching": True,
            },
        },
        "runs": [],
    }

    for name, endpoint in endpoints.items():
        print(f"Warm-up: {name}", flush=True)
        post_json(endpoint, make_payload(args.model, f"{name} 워밍업", 16), f"warmup-{name}")

    for concurrency in args.concurrency:
        request_count = max(4, concurrency)
        for name, endpoint in endpoints.items():
            print(
                f"Profile={args.profile} endpoint={name} concurrency={concurrency} requests={request_count}",
                flush=True,
            )
            measurements, wall_s = run_concurrent(
                endpoint,
                name,
                args.model,
                concurrency,
                request_count,
                args.max_tokens,
            )
            result["runs"].append(summarize(name, concurrency, measurements, wall_s))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_markdown)
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
