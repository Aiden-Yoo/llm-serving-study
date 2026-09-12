#!/usr/bin/env python3
"""Verify benchmark summary statistics against the published request-level CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def assert_close(label: str, actual: float | None, expected: float | None) -> None:
    if actual is None or expected is None:
        assert_equal(label, actual, expected)
        return
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="request-level benchmark CSV")
    parser.add_argument("--summary", type=Path, required=True, help="benchmark summary JSON")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["concurrency"])].append(row)
    summaries = {int(item["concurrency"]): item for item in summary["results"]}
    assert_equal("concurrency levels", sorted(grouped), sorted(summaries))

    for concurrency, level_rows in sorted(grouped.items()):
        expected = summaries[concurrency]
        ok = [row for row in level_rows if row["status"] == "ok"]
        request_ids = [row["request_id"] for row in level_rows]
        assert_equal(f"c={concurrency} unique request IDs", len(set(request_ids)), len(request_ids))
        assert_equal(f"c={concurrency} requests", len(level_rows), expected["requests"])
        assert_equal(f"c={concurrency} successes", len(ok), expected["successes"])
        assert_equal(f"c={concurrency} errors", len(level_rows) - len(ok), expected["errors"])

        token_count = sum(int(row["completion_tokens"]) for row in ok)
        mean_tokens = token_count / len(ok) if ok else 0
        assert_close(
            f"c={concurrency} mean completion tokens",
            mean_tokens,
            expected["mean_completion_tokens"],
        )
        assert_close(
            f"c={concurrency} output TPS formula",
            token_count / expected["wall_s"],
            expected["output_tokens_per_s"],
        )

        metrics = {
            "ttft": [optional_float(row["ttft_s"]) for row in ok],
            "e2e": [optional_float(row["e2e_s"]) for row in ok],
            "mean_itl": [optional_float(row["mean_itl_s"]) for row in ok],
        }
        for name, values in metrics.items():
            measured = [value for value in values if value is not None]
            quantiles = (0.95,) if name == "mean_itl" else (0.50, 0.95, 0.99)
            for quantile in quantiles:
                key = f"{name}_p{int(quantile * 100)}_s"
                assert_close(
                    f"c={concurrency} {key}",
                    percentile(measured, quantile),
                    expected[key],
                )

        print(
            f"c={concurrency}: requests={len(level_rows)}, successes={len(ok)}, "
            f"tokens={token_count}, summary=verified"
        )

    print("BENCHMARK_RESULTS_VERIFIED")


if __name__ == "__main__":
    main()
