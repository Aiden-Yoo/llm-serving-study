#!/usr/bin/env python3
"""Run a real GPU lazy-loading and LRU-eviction experiment with two models."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import platform
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GIB = 1024**3


def gib(value: int) -> float:
    return round(value / GIB, 4)


def gpu_memory() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": gib(torch.cuda.memory_allocated()),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
        "free_gib": gib(free),
        "total_gib": gib(total),
    }


def synchronize() -> None:
    torch.cuda.synchronize()


@dataclass
class ModelSpec:
    name: str
    public_path: str
    local_path: Path


@dataclass
class LoadedModel:
    spec: ModelSpec
    tokenizer: Any
    model: Any


class LRUModelManager:
    def __init__(self, specs: dict[str, ModelSpec], capacity: int) -> None:
        self.specs = specs
        self.capacity = capacity
        self.cache: OrderedDict[str, LoadedModel] = OrderedDict()

    def unload_lru(self) -> dict[str, Any] | None:
        if not self.cache:
            return None
        name, entry = self.cache.popitem(last=False)
        started = time.perf_counter()
        memory_before = gpu_memory()
        del entry.model
        del entry.tokenizer
        del entry
        gc.collect()
        torch.cuda.empty_cache()
        synchronize()
        return {
            "model": name,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "memory_before": memory_before,
            "memory_after": gpu_memory(),
        }

    def get_or_load(self, name: str) -> tuple[LoadedModel, bool, dict[str, Any] | None, float]:
        if name in self.cache:
            entry = self.cache.pop(name)
            self.cache[name] = entry
            return entry, True, None, 0.0

        eviction = self.unload_lru() if len(self.cache) >= self.capacity else None
        spec = self.specs[name]
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(spec.local_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec.local_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        model.eval()
        model.to("cuda")
        synchronize()
        load_s = time.perf_counter() - started
        entry = LoadedModel(spec=spec, tokenizer=tokenizer, model=model)
        self.cache[name] = entry
        return entry, False, eviction, load_s

    def clear(self) -> list[dict[str, Any]]:
        evictions: list[dict[str, Any]] = []
        while self.cache:
            eviction = self.unload_lru()
            if eviction:
                evictions.append(eviction)
        return evictions


def generate(entry: LoadedModel, prompt: str, max_new_tokens: int) -> tuple[str, float, dict[str, float]]:
    messages = [
        {"role": "system", "content": "간결하고 정확하게 답하라."},
        {"role": "user", "content": prompt},
    ]
    rendered = entry.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = entry.tokenizer(rendered, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = entry.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=entry.tokenizer.eos_token_id,
        )
    synchronize()
    elapsed_s = time.perf_counter() - started
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    text = entry.tokenizer.decode(new_tokens, skip_special_tokens=True)
    memory = gpu_memory()
    memory["peak_allocated_gib"] = gib(torch.cuda.max_memory_allocated())
    del inputs
    del output
    del new_tokens
    torch.cuda.empty_cache()
    return text, elapsed_s, memory


def parse_model(value: str) -> ModelSpec:
    try:
        name, public_path, local_path = value.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Model must be NAME=PUBLIC_PATH=LOCAL_PATH"
        ) from exc
    path = Path(local_path)
    if not (path / "config.json").is_file():
        raise argparse.ArgumentTypeError(f"Model config not found: {path / 'config.json'}")
    return ModelSpec(name=name, public_path=public_path, local_path=path)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# R9700 멀티 모델 lazy loading·LRU 결과",
        "",
        f"- Timestamp: `{result['metadata']['timestamp']}`",
        f"- GPU: `{result['metadata']['gpu']}`",
        f"- Cache capacity: `{result['metadata']['capacity']} model`",
        f"- Access sequence: `{' → '.join(result['metadata']['sequence'])}`",
        "",
        "| Step | Model | Cache | Evicted | Load (s) | Inference (s) | Allocated after load (GiB) | Peak inference (GiB) |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for event in result["events"]:
        lines.append(
            f"| {event['step']} | {event['model']} | {'hit' if event['cache_hit'] else 'miss'} | "
            f"{event['eviction']['model'] if event['eviction'] else '-'} | "
            f"{event['load_s']:.3f} | {event['inference_s']:.3f} | "
            f"{event['memory_after_load']['allocated_gib']:.3f} | "
            f"{event['memory_after_inference']['peak_allocated_gib']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Eviction memory",
            "",
            "| Step | Evicted model | Reserved before/after (GiB) | Free before/after (GiB) | Eviction (s) |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for event in result["events"]:
        eviction = event["eviction"]
        if not eviction:
            continue
        lines.append(
            f"| {event['step']} | {eviction['model']} | "
            f"{eviction['memory_before']['reserved_gib']:.3f}/{eviction['memory_after']['reserved_gib']:.3f} | "
            f"{eviction['memory_before']['free_gib']:.3f}/{eviction['memory_after']['free_gib']:.3f} | "
            f"{eviction['elapsed_s']:.3f} |"
        )

    summary = result["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Cache hits: `{summary['cache_hits']}`",
            f"- Cache misses: `{summary['cache_misses']}`",
            f"- LRU evictions during requests: `{summary['evictions']}`",
            f"- Final cleanup reserved VRAM: `{summary['final_memory']['reserved_gib']:.4f} GiB`",
            f"- Final cleanup free VRAM: `{summary['final_memory']['free_gib']:.4f} GiB`",
            "",
            "## Interpretation",
            "",
            "- 두 번째 동일 모델 요청은 cache hit이므로 모델 load 시간이 발생하지 않는다.",
            "- capacity가 1이므로 다른 모델 요청은 기존 모델을 LRU로 제거한 뒤 새 모델을 로드한다.",
            "- `del`만 하지 않고 `gc.collect()`와 `torch.cuda.empty_cache()` 후 VRAM 변화를 측정했다.",
            "- 동적 로딩은 유휴 VRAM을 줄일 수 있지만 cache miss마다 load latency가 사용자 요청에 추가된다.",
            "",
            "> 단일 실행 결과이며 모델 저장장치와 OS cache 상태에 따라 load 시간은 달라질 수 있다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--sequence", nargs="+", required=True)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU is unavailable through torch.cuda")
    specs = {spec.name: spec for spec in args.model}
    unknown = [name for name in args.sequence if name not in specs]
    if unknown:
        raise SystemExit(f"Unknown models in sequence: {unknown}")
    if args.capacity < 1:
        raise SystemExit("capacity must be at least 1")

    torch.cuda.empty_cache()
    manager = LRUModelManager(specs, args.capacity)
    result: dict[str, Any] = {
        "metadata": {
            "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "dtype": "bfloat16",
            "capacity": args.capacity,
            "sequence": args.sequence,
            "models": {name: spec.public_path for name, spec in specs.items()},
            "initial_memory": gpu_memory(),
        },
        "events": [],
    }

    try:
        for step, name in enumerate(args.sequence, start=1):
            print(f"Step {step}: access {name}", flush=True)
            entry, cache_hit, eviction, load_s = manager.get_or_load(name)
            memory_after_load = gpu_memory()
            output, inference_s, memory_after_inference = generate(
                entry,
                f"요청 {step}. 모델 서빙에서 LRU cache의 장단점을 한 문장으로 설명하라.",
                args.max_new_tokens,
            )
            result["events"].append(
                {
                    "step": step,
                    "model": name,
                    "public_path": entry.spec.public_path,
                    "cache_hit": cache_hit,
                    "eviction": eviction,
                    "load_s": round(load_s, 4),
                    "inference_s": round(inference_s, 4),
                    "memory_after_load": memory_after_load,
                    "memory_after_inference": memory_after_inference,
                    "output_preview": output[:240],
                    "cache_order_lru_to_mru": list(manager.cache),
                }
            )
    finally:
        final_evictions = manager.clear()

    result["summary"] = {
        "cache_hits": sum(1 for event in result["events"] if event["cache_hit"]),
        "cache_misses": sum(1 for event in result["events"] if not event["cache_hit"]),
        "evictions": sum(1 for event in result["events"] if event["eviction"]),
        "final_cleanup": final_evictions,
        "final_memory": gpu_memory(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_markdown)
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
