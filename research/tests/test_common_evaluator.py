from pathlib import Path
import sys
import json
import pytest

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = RESEARCH_ROOT / "evaluation"

if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import common_evaluator as evaluator


GOLD_PATH = (
    RESEARCH_ROOT
    / "data"
    / "processed"
    / "macslu_korean_v0.1"
    / "test.jsonl"
)


def load_test_context():
    schema, registry = evaluator.load_contract()
    gold_records = evaluator.validate_gold_file(GOLD_PATH)
    return schema, registry, gold_records


def perfect_predictions(gold_records):
    return {
        row["example_id"]: {
            "calls": row["canonical_calls"],
        }
        for row in gold_records
    }


def test_perfect_predictions():
    schema, registry, gold_records = load_test_context()
    predictions = perfect_predictions(gold_records)

    ordered = evaluator.evaluate_ordered_full_call_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    function = evaluator.evaluate_function_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    argument = evaluator.evaluate_argument_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    invalid = evaluator.evaluate_invalid_call_rate(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    unordered = evaluator.evaluate_unordered_call_multiset_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    assert ordered["accuracy"] == 1.0
    assert function["accuracy"] == 1.0
    assert argument["accuracy"] == 1.0
    assert invalid["rate"] == 0.0
    assert unordered["accuracy"] == 1.0


def test_null_prediction_is_invalid():
    schema, registry, gold_records = load_test_context()
    predictions = perfect_predictions(gold_records)

    example_id = gold_records[0]["example_id"]
    predictions[example_id] = None

    ordered = evaluator.evaluate_ordered_full_call_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    invalid = evaluator.evaluate_invalid_call_rate(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    assert ordered["correct"] == 60
    assert invalid["invalid"] == 1


def test_wrong_argument_preserves_function_em():
    schema, registry, gold_records = load_test_context()
    predictions = perfect_predictions(gold_records)

    first = gold_records[0]
    example_id = first["example_id"]

    calls = [
        {
            "function": call["function"],
            "arguments": dict(call["arguments"]),
        }
        for call in first["canonical_calls"]
    ]

    # test 첫 sample의 두 번째 call은 set_hvac_power(state=on)
    calls[1]["arguments"]["state"] = "off"

    predictions[example_id] = {"calls": calls}

    ordered = evaluator.evaluate_ordered_full_call_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    function = evaluator.evaluate_function_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    argument = evaluator.evaluate_argument_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    invalid = evaluator.evaluate_invalid_call_rate(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    assert ordered["correct"] == 60
    assert function["correct"] == 61
    assert argument["correct_calls"] == 113
    assert invalid["invalid"] == 0


def test_reordered_calls_only_affect_ordered_metrics():
    schema, registry, gold_records = load_test_context()
    predictions = perfect_predictions(gold_records)

    first = gold_records[0]
    example_id = first["example_id"]

    predictions[example_id] = {
        "calls": list(reversed(first["canonical_calls"]))
    }

    ordered = evaluator.evaluate_ordered_full_call_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    function = evaluator.evaluate_function_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    argument = evaluator.evaluate_argument_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    unordered = evaluator.evaluate_unordered_call_multiset_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    assert ordered["correct"] == 60
    assert function["correct"] == 60
    assert argument["correct_calls"] == 112
    assert unordered["correct"] == 61


def test_call_count_distribution():
    schema, registry, gold_records = load_test_context()
    predictions = perfect_predictions(gold_records)

    result = evaluator.evaluate_call_count_full_em(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    assert result["1_call"]["examples"] == 10
    assert result["2_call"]["examples"] == 49
    assert result["3_plus_calls"]["examples"] == 2

    assert result["1_call"]["accuracy"] == 1.0
    assert result["2_call"]["accuracy"] == 1.0
    assert result["3_plus_calls"]["accuracy"] == 1.0

def test_schema_invalid_prediction_is_invalid():
    schema, registry, gold_records = load_test_context()
    predictions = perfect_predictions(gold_records)

    first = gold_records[0]
    example_id = first["example_id"]

    calls = [
        {
            "function": call["function"],
            "arguments": dict(call["arguments"]),
        }
        for call in first["canonical_calls"]
    ]

    # set_hvac_power.state는 on/off만 허용되므로 schema-invalid.
    calls[1]["arguments"]["state"] = "definitely_invalid"

    predictions[example_id] = {
        "calls": calls,
    }

    ordered = evaluator.evaluate_ordered_full_call_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    invalid = evaluator.evaluate_invalid_call_rate(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    assert ordered["correct"] == 60
    assert invalid["invalid"] == 1


def test_missing_prediction_is_invalid():
    schema, registry, gold_records = load_test_context()
    predictions = perfect_predictions(gold_records)

    first_id = gold_records[0]["example_id"]
    del predictions[first_id]

    ordered = evaluator.evaluate_ordered_full_call_exact_match(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    invalid = evaluator.evaluate_invalid_call_rate(
        gold_records,
        predictions,
        schema=schema,
        registry=registry,
    )

    assert len(predictions) == 60
    assert ordered["correct"] == 60
    assert invalid["invalid"] == 1


def test_duplicate_prediction_id_is_rejected(tmp_path):
    _, _, gold_records = load_test_context()

    first = gold_records[0]

    record = {
        "example_id": first["example_id"],
        "prediction": {
            "calls": first["canonical_calls"],
        },
    }

    prediction_path = tmp_path / "duplicate_predictions.jsonl"

    prediction_path.write_text(
        json.dumps(record, ensure_ascii=False)
        + "\n"
        + json.dumps(record, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    gold_ids = {
        row["example_id"]
        for row in gold_records
    }

    with pytest.raises(
        evaluator.EvaluationError,
        match="duplicate prediction example_id",
    ):
        evaluator.load_prediction_file(
            prediction_path,
            gold_ids=gold_ids,
        )