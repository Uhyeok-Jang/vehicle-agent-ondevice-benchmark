#!/usr/bin/env python3
"""Aggregate model-agnostic on-device runtime measurements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence


class RuntimeMetricError(ValueError):
    """Raised when runtime measurement input is invalid."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeMetricError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error

            if not isinstance(value, dict):
                raise RuntimeMetricError(
                    f"{path}:{line_number}: expected JSON object"
                )

            records.append(value)

    return records


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "example_id",
        "ttfc_ms",
        "peak_memory_mb",
    }

    missing = sorted(required - set(record))
    if missing:
        raise RuntimeMetricError(
            f"runtime record missing required fields: {missing}"
        )

    example_id = record["example_id"]

    if not isinstance(example_id, str) or not example_id:
        raise RuntimeMetricError(
            "example_id must be a non-empty string"
        )

    for key in ("ttfc_ms", "peak_memory_mb"):
        value = record[key]

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuntimeMetricError(
                f"{example_id}: {key} must be a finite non-negative number"
            )


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, where q is in [0, 1]."""

    if not values:
        raise RuntimeMetricError(
            "cannot compute percentile of empty sequence"
        )

    if not 0.0 <= q <= 1.0:
        raise RuntimeMetricError(
            "percentile q must be between 0 and 1"
        )

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def artifact_size_bytes(path: Path) -> int:
    """Return total deployed model artifact size.

    A file returns its file size.
    A directory returns the recursive sum of all files.
    """

    if not path.exists():
        raise RuntimeMetricError(
            f"artifact path does not exist: {path}"
        )

    if path.is_file():
        return path.stat().st_size

    if path.is_dir():
        return sum(
            child.stat().st_size
            for child in path.rglob("*")
            if child.is_file()
        )

    raise RuntimeMetricError(
        f"unsupported artifact path: {path}"
    )


def evaluate_runtime(
    records: Sequence[dict[str, Any]],
    *,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    if not records:
        raise RuntimeMetricError(
            "runtime measurement file is empty"
        )

    for record in records:
        validate_record(record)

    ttfc_values = [
        float(record["ttfc_ms"])
        for record in records
    ]

    memory_values = [
        float(record["peak_memory_mb"])
        for record in records
    ]

    summary: dict[str, Any] = {
        "samples": len(records),
        "unique_examples": len(
            {
                record["example_id"]
                for record in records
            }
        ),
        "ttfc_ms": {
            "mean": mean(ttfc_values),
            "p50": median(ttfc_values),
            "p95": percentile(ttfc_values, 0.95),
            "min": min(ttfc_values),
            "max": max(ttfc_values),
        },
        "peak_memory_mb": {
            "max": max(memory_values),
        },
    }

    if artifact_path is not None:
        size_bytes = artifact_size_bytes(artifact_path)

        summary["model_artifact"] = {
            "bytes": size_bytes,
            "mb": size_bytes / (1024 * 1024),
        }

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--measurements",
        type=Path,
        required=True,
        help="Runtime measurement JSONL",
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        help="Deployed model artifact file or directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output summary JSON",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    records = load_jsonl(args.measurements)

    summary = evaluate_runtime(
        records,
        artifact_path=args.artifact,
    )

    print(
        f"samples: {summary['samples']}, "
        f"unique examples: {summary['unique_examples']}"
    )

    print(
        "TTFC (ms): "
        f"p50={summary['ttfc_ms']['p50']:.3f}, "
        f"p95={summary['ttfc_ms']['p95']:.3f}"
    )

    print(
        "peak memory (MB): "
        f"{summary['peak_memory_mb']['max']:.3f}"
    )

    if "model_artifact" in summary:
        print(
            "model artifact (MB): "
            f"{summary['model_artifact']['mb']:.3f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(f"summary written: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
