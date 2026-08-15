#!/usr/bin/env python3
"""Combine constrained and balanced vLLM profile results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def find_run(result: dict[str, Any], endpoint: str, concurrency: int) -> dict[str, Any]:
    for row in result["runs"]:
        if row["endpoint"] == endpoint and row["concurrency"] == concurrency:
            return row
    raise KeyError(f"No result for endpoint={endpoint}, concurrency={concurrency}")


def change_percent(before: float, after: float) -> float:
    return round((after / before - 1) * 100, 1) if before else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constrained", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    constrained = json.loads(args.constrained.read_text(encoding="utf-8"))
    balanced = json.loads(args.balanced.read_text(encoding="utf-8"))
    concurrency = max(row["concurrency"] for row in balanced["runs"])

    blocking = find_run(balanced, "blocking", concurrency)
    threaded = find_run(balanced, "threaded", concurrency)
    async_row = find_run(balanced, "async", concurrency)
    direct = find_run(balanced, "direct", concurrency)
    constrained_direct = find_run(constrained, "direct", concurrency)

    summary = {
        "concurrency": concurrency,
        "blocking_to_async": {
            "output_tokens_per_s_change_percent": change_percent(
                blocking["output_tokens_per_s"], async_row["output_tokens_per_s"]
            ),
            "wall_time_change_percent": change_percent(blocking["wall_s"], async_row["wall_s"]),
        },
        "blocking_to_threaded": {
            "output_tokens_per_s_change_percent": change_percent(
                blocking["output_tokens_per_s"], threaded["output_tokens_per_s"]
            ),
        },
        "constrained_to_balanced_direct": {
            "output_tokens_per_s_change_percent": change_percent(
                constrained_direct["output_tokens_per_s"], direct["output_tokens_per_s"]
            ),
            "e2e_p95_change_percent": change_percent(
                constrained_direct["e2e_p95_s"], direct["e2e_p95_s"]
            ),
        },
        "profiles": {
            "constrained": constrained["metadata"]["server_settings"],
            "balanced": balanced["metadata"]["server_settings"],
        },
    }
    combined = {"summary": summary, "constrained": constrained, "balanced": balanced}
    args.output_json.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# R9700 비동기 호출·스케줄링 비교 결과",
        "",
        f"- Model: `{balanced['metadata']['model']}`",
        f"- 비교 동시성: `{concurrency}`",
        "",
        "## API 호출 방식 비교 — balanced profile",
        "",
        "| 방식 | Wall time (s) | output tok/s | E2E p50/p95 (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in (blocking, threaded, async_row, direct):
        lines.append(
            f"| {row['endpoint']} | {row['wall_s']:.3f} | {row['output_tokens_per_s']:.1f} | "
            f"{row['e2e_p50_s']:.3f}/{row['e2e_p95_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- blocking 대비 async output TPS 변화: "
            f"`{summary['blocking_to_async']['output_tokens_per_s_change_percent']:+.1f}%`",
            f"- blocking 대비 threaded output TPS 변화: "
            f"`{summary['blocking_to_threaded']['output_tokens_per_s_change_percent']:+.1f}%`",
            "",
            "## vLLM 스케줄링 profile 비교 — direct endpoint",
            "",
            "| Profile | max_num_seqs | max_num_batched_tokens | output tok/s | E2E p95 (s) |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| constrained | {constrained['metadata']['server_settings']['max_num_seqs']} | "
            f"{constrained['metadata']['server_settings']['max_num_batched_tokens']} | "
            f"{constrained_direct['output_tokens_per_s']:.1f} | {constrained_direct['e2e_p95_s']:.3f} |",
            f"| balanced | {balanced['metadata']['server_settings']['max_num_seqs']} | "
            f"{balanced['metadata']['server_settings']['max_num_batched_tokens']} | "
            f"{direct['output_tokens_per_s']:.1f} | {direct['e2e_p95_s']:.3f} |",
            "",
            f"- constrained 대비 balanced output TPS 변화: "
            f"`{summary['constrained_to_balanced_direct']['output_tokens_per_s_change_percent']:+.1f}%`",
            f"- constrained 대비 balanced E2E p95 변화: "
            f"`{summary['constrained_to_balanced_direct']['e2e_p95_change_percent']:+.1f}%`",
            "",
            "## 해석 기준",
            "",
            "- blocking 경로는 `async def` 안의 동기 호출이 event loop를 막는 negative control이다.",
            "- threaded와 async 경로는 여러 요청을 vLLM에 동시에 전달해 continuous batching 기회를 만든다.",
            "- profile 비교에서는 호출 경로를 direct로 고정하고 vLLM scheduling 한도만 변경했다.",
            "- 단일 실행 결과이므로 절대 수치보다 같은 장비에서 관찰한 상대 차이를 해석한다.",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
