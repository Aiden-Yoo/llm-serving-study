#!/usr/bin/env python3
"""Recompute every published Week 7 summary from its request-level CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from open_loop_load import summarize


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["request_id"] = int(row["request_id"])
        row["max_tokens"] = int(row["max_tokens"])
        row["scheduled_at_s"] = float(row["scheduled_at_s"])
        row["sent_at_s"] = float(row["sent_at_s"])
        row["launch_lag_s"] = float(row["launch_lag_s"])
        row["http_status"] = int(row["http_status"])
        row["ttft_s"] = optional_float(row["ttft_s"])
        row["e2e_s"] = float(row["e2e_s"])
        row["mean_itl_s"] = optional_float(row["mean_itl_s"])
        row["completion_tokens"] = int(row["completion_tokens"])
    return rows


def compare(path: str, actual: object, expected: object) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: keys differ: {set(actual) ^ set(expected)}")
        for key in actual:
            compare(f"{path}.{key}", actual[key], expected[key])
        return
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{path}: actual={actual!r}, expected={expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: actual={actual!r}, expected={expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    document = json.loads(args.summary.read_text(encoding="utf-8"))
    expected = document["results"]
    recorded_wall_s = float(expected["overall"]["wall_s"])
    derived_wall_s = max((row["sent_at_s"] + row["e2e_s"] for row in rows), default=0)
    if not math.isclose(recorded_wall_s, derived_wall_s, rel_tol=0, abs_tol=0.1):
        raise AssertionError(
            f"wall_s differs from request timestamps: recorded={recorded_wall_s}, derived={derived_wall_s}"
        )
    actual = summarize(rows, recorded_wall_s)
    compare("results", actual, expected)

    request_ids = [row["request_id"] for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise AssertionError("request IDs are not unique")
    if len(rows) != document["configuration"]["total_requests"]:
        raise AssertionError("CSV row count does not match configured request count")
    print(f"{args.csv.name}: rows={len(rows)}, summary=verified")
    print("WEEK7_RESULTS_VERIFIED")


if __name__ == "__main__":
    main()
