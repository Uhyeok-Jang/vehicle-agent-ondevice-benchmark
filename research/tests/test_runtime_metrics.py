from pathlib import Path
import sys

import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = RESEARCH_ROOT / "evaluation"

if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import runtime_metrics


def test_runtime_summary():
    records = [
        {
            "example_id": "a",
            "ttfc_ms": 10.0,
            "peak_memory_mb": 100.0,
        },
        {
            "example_id": "b",
            "ttfc_ms": 20.0,
            "peak_memory_mb": 120.0,
        },
        {
            "example_id": "c",
            "ttfc_ms": 30.0,
            "peak_memory_mb": 110.0,
        },
    ]

    result = runtime_metrics.evaluate_runtime(records)

    assert result["samples"] == 3
    assert result["unique_examples"] == 3
    assert result["ttfc_ms"]["p50"] == 20.0
    assert result["ttfc_ms"]["p95"] == pytest.approx(29.0)
    assert result["peak_memory_mb"]["max"] == 120.0


def test_repeated_examples_are_allowed():
    records = [
        {
            "example_id": "a",
            "ttfc_ms": 10.0,
            "peak_memory_mb": 100.0,
        },
        {
            "example_id": "a",
            "ttfc_ms": 12.0,
            "peak_memory_mb": 105.0,
        },
    ]

    result = runtime_metrics.evaluate_runtime(records)

    assert result["samples"] == 2
    assert result["unique_examples"] == 1
    assert result["ttfc_ms"]["p50"] == 11.0


def test_invalid_negative_ttfc():
    records = [
        {
            "example_id": "a",
            "ttfc_ms": -1.0,
            "peak_memory_mb": 100.0,
        }
    ]

    with pytest.raises(
        runtime_metrics.RuntimeMetricError
    ):
        runtime_metrics.evaluate_runtime(records)


def test_artifact_directory_size(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"a" * 100)
    (tmp_path / "b.bin").write_bytes(b"b" * 200)

    records = [
        {
            "example_id": "a",
            "ttfc_ms": 10.0,
            "peak_memory_mb": 100.0,
        }
    ]

    result = runtime_metrics.evaluate_runtime(
        records,
        artifact_path=tmp_path,
    )

    assert result["model_artifact"]["bytes"] == 300
