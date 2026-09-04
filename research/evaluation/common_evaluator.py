#!/usr/bin/env python3
"""Model-agnostic evaluator for canonical Vehicle API predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_ROOT = RESEARCH_ROOT / "preprocessing"

if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_ROOT))

import canonical_vehicle_api as canonical


DEFAULT_SCHEMA = RESEARCH_ROOT / "schema" / "vehicle_api_schema.v0.1.0.json"
DEFAULT_REGISTRY = RESEARCH_ROOT / "schema" / "vehicle_api_registry.v0.1.0.json"


class EvaluationError(ValueError):
    """Raised when evaluation input violates the evaluator contract."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file containing one JSON object per non-empty line."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error

            if not isinstance(value, dict):
                raise EvaluationError(
                    f"{path}:{line_number}: expected JSON object"
                )

            records.append(value)

    return records


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the frozen canonical Vehicle API contract."""
    schema = canonical.load_json_object(DEFAULT_SCHEMA)
    registry = canonical.load_json_object(DEFAULT_REGISTRY)

    canonical.validate_registry(schema, registry)

    return schema, registry


def validate_gold_record(
    record: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    """Validate one processed benchmark gold record."""
    required = {
        "example_id",
        "canonical_calls",
        "call_count",
    }

    missing = sorted(required - set(record))
    if missing:
        raise EvaluationError(
            f"gold record missing required fields: {missing}"
        )

    example_id = record["example_id"]
    if not isinstance(example_id, str) or not example_id:
        raise EvaluationError(
            "gold example_id must be non-empty string"
        )

    calls = record["canonical_calls"]
    if not isinstance(calls, list):
        raise EvaluationError(
            f"{example_id}: canonical_calls must be a list"
        )

    call_count = record["call_count"]
    if not isinstance(call_count, int) or isinstance(call_count, bool):
        raise EvaluationError(
            f"{example_id}: call_count must be an integer"
        )

    if call_count != len(calls):
        raise EvaluationError(
            f"{example_id}: call_count={call_count}, "
            f"but canonical_calls has {len(calls)} calls"
        )

    payload = {"calls": calls}

    try:
        canonical.canonicalize_payload(
            payload,
            schema=schema,
            registry=registry,
        )
    except canonical.CanonicalValidationError as error:
        raise EvaluationError(
            f"{example_id}: invalid canonical gold: {error}"
        ) from error


def validate_gold_file(path: Path) -> list[dict[str, Any]]:
    """Load and validate all gold records."""
    schema, registry = load_contract()
    records = load_jsonl(path)

    seen_ids: set[str] = set()

    for record in records:
        validate_gold_record(
            record,
            schema=schema,
            registry=registry,
        )

        example_id = record["example_id"]

        if example_id in seen_ids:
            raise EvaluationError(
                f"duplicate gold example_id: {example_id}"
            )

        seen_ids.add(example_id)

    return records


def validate_prediction_record(
    record: Mapping[str, Any],
) -> None:
    """Validate only the prediction-file envelope.

    Canonical Vehicle API validity is deliberately NOT checked here.
    Schema-invalid predictions must survive loading so that the evaluator
    can count them in Invalid Call Rate.
    """
    required = {
        "example_id",
        "prediction",
    }

    missing = sorted(required - set(record))
    if missing:
        raise EvaluationError(
            f"prediction record missing required fields: {missing}"
        )

    example_id = record["example_id"]

    if not isinstance(example_id, str) or not example_id:
        raise EvaluationError(
            "prediction example_id must be non-empty string"
        )

    prediction = record["prediction"]

    if prediction is not None and not isinstance(prediction, dict):
        raise EvaluationError(
            f"{example_id}: prediction must be an object or null"
        )


def load_prediction_file(
    path: Path,
    *,
    gold_ids: set[str],
) -> dict[str, dict[str, Any] | None]:
    """Load predictions indexed by example_id.

    Missing gold IDs are allowed and will later be evaluated as invalid.
    Unknown or duplicate prediction IDs are input-contract errors.
    """
    records = load_jsonl(path)

    predictions: dict[str, dict[str, Any] | None] = {}

    for record in records:
        validate_prediction_record(record)

        example_id = record["example_id"]

        if example_id in predictions:
            raise EvaluationError(
                f"duplicate prediction example_id: {example_id}"
            )

        if example_id not in gold_ids:
            raise EvaluationError(
                f"unknown prediction example_id: {example_id}"
            )

        predictions[example_id] = record["prediction"]

    return predictions

def ordered_full_call_exact_match(
    gold_record: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    """Return whether prediction exactly matches the ordered canonical gold."""

    if prediction is None:
        return False

    gold_payload = {
        "calls": gold_record["canonical_calls"],
    }

    try:
        gold_canonical = canonical.canonical_json(
            gold_payload,
            schema=schema,
            registry=registry,
        )

        prediction_canonical = canonical.canonical_json(
            prediction,
            schema=schema,
            registry=registry,
        )

    except canonical.CanonicalValidationError:
        # Gold has already been validated.
        # Therefore this normally means prediction is schema-invalid.
        return False

    return prediction_canonical == gold_canonical


def evaluate_ordered_full_call_exact_match(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute sample-level Ordered Full Call Exact Match."""

    correct = 0

    for gold_record in gold_records:
        example_id = gold_record["example_id"]

        # Missing predictions are intentionally treated like prediction=null.
        prediction = predictions.get(example_id)

        if ordered_full_call_exact_match(
            gold_record,
            prediction,
            schema=schema,
            registry=registry,
        ):
            correct += 1

    total = len(gold_records)
    accuracy = correct / total if total else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }

def function_exact_match(
    gold_record: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    """Return whether the ordered function sequence exactly matches gold."""

    if prediction is None:
        return False

    try:
        prediction_canonical = canonical.canonicalize_payload(
            prediction,
            schema=schema,
            registry=registry,
        )
    except canonical.CanonicalValidationError:
        return False

    gold_functions = [
        call["function"]
        for call in gold_record["canonical_calls"]
    ]

    prediction_functions = [
        call["function"]
        for call in prediction_canonical["calls"]
    ]

    return prediction_functions == gold_functions


def evaluate_function_exact_match(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute sample-level ordered Function Exact Match."""

    correct = 0

    for gold_record in gold_records:
        example_id = gold_record["example_id"]
        prediction = predictions.get(example_id)

        if function_exact_match(
            gold_record,
            prediction,
            schema=schema,
            registry=registry,
        ):
            correct += 1

    total = len(gold_records)
    accuracy = correct / total if total else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }

def count_argument_exact_matches(
    gold_record: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[int, int]:
    """Count aligned calls with exactly matching function and arguments."""

    gold_payload = {
        "calls": gold_record["canonical_calls"],
    }

    gold_canonical = canonical.canonicalize_payload(
        gold_payload,
        schema=schema,
        registry=registry,
    )

    gold_calls = gold_canonical["calls"]
    total_gold_calls = len(gold_calls)

    if prediction is None:
        return 0, total_gold_calls

    try:
        prediction_canonical = canonical.canonicalize_payload(
            prediction,
            schema=schema,
            registry=registry,
        )
    except canonical.CanonicalValidationError:
        return 0, total_gold_calls

    prediction_calls = prediction_canonical["calls"]

    correct = 0

    for index, gold_call in enumerate(gold_calls):
        # Missing prediction call -> incorrect.
        if index >= len(prediction_calls):
            continue

        predicted_call = prediction_calls[index]

        if predicted_call["function"] != gold_call["function"]:
            continue

        if predicted_call["arguments"] != gold_call["arguments"]:
            continue

        correct += 1

    return correct, total_gold_calls


def evaluate_argument_exact_match(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute call-level Argument Exact Match over all gold calls."""

    correct_calls = 0
    total_gold_calls = 0

    for gold_record in gold_records:
        example_id = gold_record["example_id"]
        prediction = predictions.get(example_id)

        correct, total = count_argument_exact_matches(
            gold_record,
            prediction,
            schema=schema,
            registry=registry,
        )

        correct_calls += correct
        total_gold_calls += total

    accuracy = (
        correct_calls / total_gold_calls
        if total_gold_calls
        else 0.0
    )

    return {
        "correct_calls": correct_calls,
        "total_gold_calls": total_gold_calls,
        "accuracy": accuracy,
    }

def is_invalid_prediction(
    prediction: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    """Return whether a prediction is missing/null or schema-invalid."""

    if prediction is None:
        return True

    try:
        canonical.canonicalize_payload(
            prediction,
            schema=schema,
            registry=registry,
        )
    except canonical.CanonicalValidationError:
        return True

    return False


def evaluate_invalid_call_rate(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute sample-level Invalid Call Rate."""

    invalid = 0

    for gold_record in gold_records:
        example_id = gold_record["example_id"]

        # Missing prediction behaves exactly like prediction=null.
        prediction = predictions.get(example_id)

        if is_invalid_prediction(
            prediction,
            schema=schema,
            registry=registry,
        ):
            invalid += 1

    total = len(gold_records)
    rate = invalid / total if total else 0.0

    return {
        "invalid": invalid,
        "total": total,
        "rate": rate,
    }

def evaluate_call_count_full_em(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute Ordered Full Call EM grouped by gold call count."""

    buckets = {
        "1_call": {
            "examples": 0,
            "correct": 0,
        },
        "2_call": {
            "examples": 0,
            "correct": 0,
        },
        "3_plus_calls": {
            "examples": 0,
            "correct": 0,
        },
    }

    for gold_record in gold_records:
        call_count = gold_record["call_count"]

        if call_count == 1:
            bucket_name = "1_call"
        elif call_count == 2:
            bucket_name = "2_call"
        else:
            bucket_name = "3_plus_calls"

        buckets[bucket_name]["examples"] += 1

        example_id = gold_record["example_id"]
        prediction = predictions.get(example_id)

        if ordered_full_call_exact_match(
            gold_record,
            prediction,
            schema=schema,
            registry=registry,
        ):
            buckets[bucket_name]["correct"] += 1

    result: dict[str, Any] = {}

    for bucket_name, values in buckets.items():
        examples = values["examples"]
        correct = values["correct"]

        accuracy = (
            correct / examples
            if examples
            else 0.0
        )

        result[bucket_name] = {
            "examples": examples,
            "correct": correct,
            "accuracy": accuracy,
        }

    return result

def unordered_call_multiset_exact_match(
    gold_record: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    """Return whether prediction matches gold ignoring call order."""

    if prediction is None:
        return False

    gold_payload = {
        "calls": gold_record["canonical_calls"],
    }

    try:
        gold_canonical = canonical.canonical_json(
            gold_payload,
            schema=schema,
            registry=registry,
            unordered_calls=True,
        )

        prediction_canonical = canonical.canonical_json(
            prediction,
            schema=schema,
            registry=registry,
            unordered_calls=True,
        )

    except canonical.CanonicalValidationError:
        return False

    return prediction_canonical == gold_canonical


def evaluate_unordered_call_multiset_exact_match(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute sample-level unordered full-call multiset exact match."""

    correct = 0

    for gold_record in gold_records:
        example_id = gold_record["example_id"]
        prediction = predictions.get(example_id)

        if unordered_call_multiset_exact_match(
            gold_record,
            prediction,
            schema=schema,
            registry=registry,
        ):
            correct += 1

    total = len(gold_records)
    accuracy = correct / total if total else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Processed benchmark gold JSONL",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        help="Model prediction JSONL",
    )

    parser.add_argument(
    "--output",
    type=Path,
    help="Optional path to write evaluation summary JSON",
)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    gold_records = validate_gold_file(args.gold)

    total_calls = sum(
        record["call_count"]
        for record in gold_records
    )

    print(f"gold examples: {len(gold_records)}")
    print(f"gold calls: {total_calls}")

    if args.predictions is None:
        print("PASS: canonical gold validation")
        return 0

    gold_ids = {
        record["example_id"]
        for record in gold_records
    }

    predictions = load_prediction_file(
        args.predictions,
        gold_ids=gold_ids,
    )

    missing_predictions = gold_ids - set(predictions)

    schema, registry = load_contract()

    ordered_em = evaluate_ordered_full_call_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    function_em = evaluate_function_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    argument_em = evaluate_argument_exact_match(
    gold_records,
    predictions,
    schema=schema,
    registry=registry,
    )

    invalid_rate = evaluate_invalid_call_rate(
    gold_records,
    predictions,
    schema=schema,
    registry=registry,
)
    call_count_em = evaluate_call_count_full_em(
    gold_records,
    predictions,
    schema=schema,
    registry=registry,
)
    unordered_em = evaluate_unordered_call_multiset_exact_match(
    gold_records,
    predictions,
    schema=schema,
    registry=registry,
)

    summary = {
    "examples": len(gold_records),
    "gold_calls": total_calls,
    "prediction_records": len(predictions),
    "missing_predictions": len(missing_predictions),

    "ordered_full_call_exact_match": ordered_em,
    "function_exact_match": function_em,
    "argument_exact_match": argument_em,
    "invalid_call_rate": invalid_rate,
    "unordered_call_multiset_exact_match": unordered_em,
    "call_count_full_em": call_count_em,
}

    print(f"prediction records: {len(predictions)}")
    print(f"missing predictions: {len(missing_predictions)}")
    print(
        "ordered full call EM: "
        f"{ordered_em['correct']}/{ordered_em['total']} "
        f"= {ordered_em['accuracy']:.6f}"
    )
    print(
        "function EM: "
        f"{function_em['correct']}/{function_em['total']} "
        f"= {function_em['accuracy']:.6f}"
    )
    print(
    "argument EM: "
    f"{argument_em['correct_calls']}/"
    f"{argument_em['total_gold_calls']} "
    f"= {argument_em['accuracy']:.6f}"
    )
    print(
    "invalid call rate: "
    f"{invalid_rate['invalid']}/{invalid_rate['total']} "
    f"= {invalid_rate['rate']:.6f}"
)
    print("call-count full EM:")
    for bucket_name in (
    "1_call",
    "2_call",
    "3_plus_calls",
):
        bucket = call_count_em[bucket_name]

        print(
            f"  {bucket_name}: "
            f"{bucket['correct']}/{bucket['examples']} "
            f"= {bucket['accuracy']:.6f}"
        )
    print(
    "unordered call multiset EM: "
    f"{unordered_em['correct']}/{unordered_em['total']} "
    f"= {unordered_em['accuracy']:.6f}"
)
    print("PASS: ordered full call exact match evaluation")

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
