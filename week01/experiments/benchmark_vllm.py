#!/usr/bin/env python3
"""Reproducible vLLM benchmark for the single-GPU R9700 study environment."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RequestMeasurement:
    label: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    e2e_s: float
    tpot_s: float | None

    @property
    def output_tokens_per_s(self) -> float:
        return self.output_tokens / self.e2e_s if self.e2e_s else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 600) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    label: str,
) -> RequestMeasurement:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "간결하고 정확하게 답하라."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_content_at: float | None = None
    usage: dict[str, int] = {}
    content_events = 0

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        content_events += 1
                        if first_content_at is None:
                            first_content_at = time.perf_counter()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {label}: {body}") from exc

    finished = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError(f"No streamed content received for {label}")
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", content_events))
    ttft = first_content_at - started
    e2e = finished - started
    tpot = (finished - first_content_at) / (output_tokens - 1) if output_tokens > 1 else None
    return RequestMeasurement(label, prompt_tokens, output_tokens, ttft, e2e, tpot)


def summarize_measurements(measurements: list[RequestMeasurement], wall_s: float) -> dict[str, Any]:
    ttfts = [item.ttft_s for item in measurements]
    e2es = [item.e2e_s for item in measurements]
    tpots = [item.tpot_s for item in measurements if item.tpot_s is not None]
    prompt_tokens = sum(item.prompt_tokens for item in measurements)
    output_tokens = sum(item.output_tokens for item in measurements)
    return {
        "requests": len(measurements),
        "wall_s": round(wall_s, 4),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "requests_per_s": round(len(measurements) / wall_s, 4),
        "output_tokens_per_s": round(output_tokens / wall_s, 3),
        "ttft_p50_s": round(percentile(ttfts, 0.50), 4),
        "ttft_p95_s": round(percentile(ttfts, 0.95), 4),
        "e2e_p50_s": round(percentile(e2es, 0.50), 4),
        "e2e_p95_s": round(percentile(e2es, 0.95), 4),
        "tpot_p50_ms": round(percentile(tpots, 0.50) * 1000, 3),
        "tpot_p95_ms": round(percentile(tpots, 0.95) * 1000, 3),
    }


def run_concurrent(
    base_url: str,
    model: str,
    concurrency: int,
    requests: int,
    max_tokens: int,
) -> tuple[list[RequestMeasurement], float]:
    prompts = [
        f"요청 식별자 {concurrency}-{index}. KV Cache와 continuous batching의 차이를 한 문단으로 설명하라."
        for index in range(requests)
    ]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_chat,
                base_url,
                model,
                prompt,
                max_tokens,
                f"concurrency-{concurrency}-{index}",
            )
            for index, prompt in enumerate(prompts)
        ]
        measurements = [future.result() for future in futures]
    return measurements, time.perf_counter() - started


def make_target_prompt(tokenizer: Any, target_tokens: int, nonce: str) -> str:
    seed = (
        f"고유 식별자 {nonce}. 다음은 LLM 서빙 부하 측정을 위한 입력이다. "
        "모델 추론에서는 입력 처리와 출력 생성의 병목이 서로 다르다. "
    )
    text = seed
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += " KV Cache는 이전 토큰의 Key와 Value를 재사용하고 배칭은 GPU 활용률을 높인다."
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True) + "\n핵심을 한 문장으로 요약하라."


METRIC_NAMES = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def metric_counters(base_url: str) -> dict[str, float]:
    metrics = read_text(f"{base_url}/metrics")
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        matches = re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", metrics, re.MULTILINE)
        result[name] = sum(float(value) for value in matches)
    return result


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: round(after.get(name, 0) - before.get(name, 0), 3) for name in METRIC_NAMES}


def package_version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "unknown"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# vLLM R9700 benchmark",
        "",
        f"- Timestamp: `{result['metadata']['timestamp']}`",
        f"- Model: `{result['metadata']['model']}`",
        f"- vLLM: `{result['metadata']['vllm_version']}`",
        f"- PyTorch: `{result['metadata']['torch_version']}`",
        "",
        "## Concurrency sweep",
        "",
        "| Concurrency | Requests | output tok/s | TTFT P50/P95 (s) | TPOT P50/P95 (ms) | E2E P50/P95 (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["concurrency_sweep"]:
        lines.append(
            f"| {row['concurrency']} | {row['requests']} | {row['output_tokens_per_s']:.1f} | "
            f"{row['ttft_p50_s']:.3f}/{row['ttft_p95_s']:.3f} | "
            f"{row['tpot_p50_ms']:.2f}/{row['tpot_p95_ms']:.2f} | "
            f"{row['e2e_p50_s']:.3f}/{row['e2e_p95_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Prompt-length sweep",
            "",
            "| Target user tokens | Actual prompt tokens | TTFT (s) | E2E (s) |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["prompt_length_sweep"]:
        lines.append(
            f"| {row['target_user_tokens']} | {row['prompt_tokens']} | {row['ttft_s']:.3f} | {row['e2e_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Output-length sweep",
            "",
            "| Requested output tokens | Actual output tokens | TTFT (s) | TPOT (ms) | E2E (s) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["output_length_sweep"]:
        lines.append(
            f"| {row['requested_output_tokens']} | {row['output_tokens']} | {row['ttft_s']:.3f} | "
            f"{(row['tpot_s'] or 0) * 1000:.2f} | {row['e2e_s']:.3f} |"
        )
    prefix = result["prefix_cache"]
    lines.extend(
        [
            "",
            "## Prefix cache",
            "",
            f"- Cold TTFT: `{prefix['cold']['ttft_s']:.4f}s`",
            f"- Warm TTFT: `{prefix['warm']['ttft_s']:.4f}s`",
            f"- Warm request cache-hit tokens: `{prefix['warm_metric_delta']['vllm:prefix_cache_hits_total']:.0f}`",
            f"- TTFT change: `{prefix['ttft_change_percent']:.1f}%`",
            "",
            "> This is a single run on one machine, not a universal performance claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-4b-vllm")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "VLLM_MODEL_PATH",
            str(project_root / "models" / "Qwen3-4B-Instruct-2507"),
        ),
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    try:
        models = http_json(f"{base_url}/v1/models")
    except Exception as exc:
        print(f"vLLM server is unavailable at {base_url}: {exc}", file=sys.stderr)
        return 2
    if args.model not in [item["id"] for item in models.get("data", [])]:
        print(f"Model {args.model!r} is not served", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    timestamp = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    result: dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp,
            "base_url": base_url,
            "model": args.model,
            "model_path": "./models/Qwen3-4B-Instruct-2507",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vllm_version": package_version("vllm"),
            "torch_version": package_version("torch"),
            "transformers_version": package_version("transformers"),
            "rocm_smi": command_output(["rocm-smi", "--showproductname", "--showmeminfo", "vram"]),
            "server_settings": {
                "dtype": "bfloat16",
                "max_model_len": 16384,
                "gpu_memory_utilization": 0.80,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "prefix_caching": True,
            },
        }
    }

    print("Warm-up: 2 requests", flush=True)
    for index in range(2):
        stream_chat(base_url, args.model, f"워밍업 요청 {index}", 64, f"warmup-{index}")

    result["concurrency_sweep"] = []
    for concurrency in (1, 4, 8, 16):
        requests = max(4, concurrency * 2)
        print(f"Concurrency {concurrency}: {requests} requests x 128 tokens", flush=True)
        measurements, wall_s = run_concurrent(base_url, args.model, concurrency, requests, 128)
        summary = summarize_measurements(measurements, wall_s)
        summary["concurrency"] = concurrency
        result["concurrency_sweep"].append(summary)

    result["prompt_length_sweep"] = []
    for target in (128, 512, 2048, 4096):
        print(f"Prompt length target: {target} user tokens", flush=True)
        prompt = make_target_prompt(tokenizer, target, f"prefill-{target}-{time.time_ns()}")
        measurement = stream_chat(base_url, args.model, prompt, 32, f"prompt-{target}")
        row = asdict(measurement)
        row["target_user_tokens"] = target
        row["ttft_s"] = round(row["ttft_s"], 4)
        row["e2e_s"] = round(row["e2e_s"], 4)
        row["tpot_s"] = round_or_none(row["tpot_s"])
        result["prompt_length_sweep"].append(row)

    result["output_length_sweep"] = []
    for output_tokens in (32, 128, 256):
        print(f"Output length: {output_tokens} tokens", flush=True)
        prompt = f"고유 식별자 decode-{output_tokens}-{time.time_ns()}. LLM 서빙 최적화의 핵심을 설명하라."
        measurement = stream_chat(base_url, args.model, prompt, output_tokens, f"output-{output_tokens}")
        row = asdict(measurement)
        row["requested_output_tokens"] = output_tokens
        row["ttft_s"] = round(row["ttft_s"], 4)
        row["e2e_s"] = round(row["e2e_s"], 4)
        row["tpot_s"] = round_or_none(row["tpot_s"])
        result["output_length_sweep"].append(row)

    print("Prefix cache: cold then warm shared prefix", flush=True)
    shared_prefix = make_target_prompt(tokenizer, 2048, f"prefix-{time.time_ns()}")
    metrics_before = metric_counters(base_url)
    cold = stream_chat(base_url, args.model, shared_prefix + "\n첫 번째 질문에 답하라.", 32, "prefix-cold")
    metrics_after_cold = metric_counters(base_url)
    warm = stream_chat(base_url, args.model, shared_prefix + "\n두 번째 질문에 답하라.", 32, "prefix-warm")
    metrics_after_warm = metric_counters(base_url)
    result["prefix_cache"] = {
        "cold": {key: round_or_none(value) if isinstance(value, float) else value for key, value in asdict(cold).items()},
        "warm": {key: round_or_none(value) if isinstance(value, float) else value for key, value in asdict(warm).items()},
        "cold_metric_delta": metric_delta(metrics_before, metrics_after_cold),
        "warm_metric_delta": metric_delta(metrics_after_cold, metrics_after_warm),
        "ttft_change_percent": round((warm.ttft_s / cold.ttft_s - 1) * 100, 2),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"vllm_r9700_{stamp}.json"
    markdown_path = output_dir / f"vllm_r9700_{stamp}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, markdown_path)
    (output_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "latest.md").write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
