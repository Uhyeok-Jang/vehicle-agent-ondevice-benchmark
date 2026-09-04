#!/usr/bin/env python3
"""Factorized label contract for the mmBERT semantic parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_ROOT = RESEARCH_ROOT / "preprocessing"

if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_ROOT))

import canonical_vehicle_api as canonical


DATA_ROOT = (
    RESEARCH_ROOT
    / "data"
    / "processed"
    / "macslu_korean_v0.1"
)

SCHEMA_PATH = (
    RESEARCH_ROOT
    / "schema"
    / "vehicle_api_schema.v0.1.0.json"
)

REGISTRY_PATH = (
    RESEARCH_ROOT
    / "schema"
    / "vehicle_api_registry.v0.1.0.json"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "label_schema.v0.1.json"
)

MAX_CALLS = 4
NONE_ZONE = "__NONE__"

FUNCTIONS_WITH_OPTIONAL_ZONE = {
    "set_hvac_power",
    "set_hvac_temperature",
    "set_hvac_fan_speed",
}

TARGET_FUNCTIONS = {
    "set_hvac_temperature",
    "set_hvac_fan_speed",
    "set_window_position",
    "set_sunroof_position",
    "set_sunshade_position",
}

SEAT_FUNCTIONS = {
    "set_seat_climate",
    "set_seat_massage",
}

LABEL_FIELDS = (
    "function",
    "zone",
    "feature",
    "state",
    "target_kind",
    "target_value",
    "target_direction",
    "target_magnitude",
    "setting_value",
)


class LabelError(ValueError):
    """Raised when a call cannot be represented by the frozen label contract."""


def load_split(split: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / f"{split}.jsonl"

    with path.open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def value_key(value: Any) -> str:
    """Stable typed key for values such as 1 vs '1'."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def extract_call_labels(
    call: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one canonical call into factorized semantic labels."""

    function = call["function"]
    args = call["arguments"]

    labels: dict[str, Any] = {
        "function": function,
    }

    # Zone
    if "zone" in args:
        labels["zone"] = args["zone"]
    elif function in FUNCTIONS_WITH_OPTIONAL_ZONE:
        labels["zone"] = NONE_ZONE

    # Direct state, used by HVAC power.
    if "state" in args:
        labels["state"] = args["state"]

    # Seat climate feature.
    if "feature" in args:
        labels["feature"] = args["feature"]

    # Target structure.
    target = args.get("target")

    if target is not None:
        kind = target["kind"]
        labels["target_kind"] = kind

        if kind in {
            "absolute",
            "extreme",
            "named",
        }:
            labels["target_value"] = target["value"]

        elif kind == "relative":
            labels["target_direction"] = target["direction"]
            labels["target_magnitude"] = target["magnitude"]

        else:
            raise LabelError(
                f"unsupported frozen-benchmark target kind: {kind}"
            )

    # Frozen 597-group benchmark only contains state-style
    # seat settings. The kind is therefore deterministic.
    setting = args.get("setting")

    if setting is not None:
        if setting.get("kind") != "state":
            raise LabelError(
                "frozen benchmark contains non-state seat setting: "
                f"{setting}"
            )

        labels["setting_value"] = setting["value"]

    return labels


def assemble_target(
    function: str,
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct canonical target from factorized labels."""

    kind = labels["target_kind"]

    if kind == "absolute":
        target = {
            "kind": "absolute",
            "value": labels["target_value"],
        }

        if function == "set_hvac_temperature":
            target["unit"] = "celsius"
        elif function == "set_hvac_fan_speed":
            target["unit"] = "level"
        else:
            raise LabelError(
                f"absolute target is invalid for {function}"
            )

        return target

    if kind == "extreme":
        return {
            "kind": "extreme",
            "value": labels["target_value"],
        }

    if kind == "named":
        return {
            "kind": "named",
            "value": labels["target_value"],
        }

    if kind == "relative":
        return {
            "kind": "relative",
            "direction": labels["target_direction"],
            "magnitude": labels["target_magnitude"],
        }

    raise LabelError(
        f"unsupported target kind: {kind}"
    )


def assemble_call(
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert factorized semantic labels back into one canonical call."""

    function = labels["function"]
    args: dict[str, Any] = {}

    if function == "set_hvac_power":
        args["state"] = labels["state"]

        zone = labels.get("zone", NONE_ZONE)
        if zone != NONE_ZONE:
            args["zone"] = zone

    elif function in {
        "set_hvac_temperature",
        "set_hvac_fan_speed",
    }:
        args["target"] = assemble_target(
            function,
            labels,
        )

        zone = labels.get("zone", NONE_ZONE)
        if zone != NONE_ZONE:
            args["zone"] = zone

    elif function == "set_window_position":
        args["zone"] = labels["zone"]
        args["target"] = assemble_target(
            function,
            labels,
        )

    elif function in {
        "set_sunroof_position",
        "set_sunshade_position",
    }:
        args["target"] = assemble_target(
            function,
            labels,
        )

    elif function == "set_seat_climate":
        args["zone"] = labels["zone"]
        args["feature"] = labels["feature"]
        args["setting"] = {
            "kind": "state",
            "value": labels["setting_value"],
        }

    elif function == "set_seat_massage":
        args["zone"] = labels["zone"]
        args["setting"] = {
            "kind": "state",
            "value": labels["setting_value"],
        }

    else:
        raise LabelError(
            f"unknown function: {function}"
        )

    return {
        "function": function,
        "arguments": args,
    }


def build_label_schema(
    train_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build classification vocabularies from TRAIN only."""

    vocabs: dict[str, dict[str, Any]] = {
        field: {}
        for field in LABEL_FIELDS
    }

    # Optional HVAC zone needs an explicit NONE class,
    # even if absence is rare.
    vocabs["zone"][value_key(NONE_ZONE)] = NONE_ZONE

    for row in train_rows:
        for call in row["canonical_calls"]:
            labels = extract_call_labels(call)

            for field, value in labels.items():
                if field not in vocabs:
                    continue

                vocabs[field][value_key(value)] = value

    label_vocab: dict[str, list[Any]] = {}

    for field in LABEL_FIELDS:
        values = list(vocabs[field].values())

        values.sort(
            key=lambda value: value_key(value)
        )

        label_vocab[field] = values

    return {
        "version": "0.1.0",
        "model_family": "mmbert_factorized_semantic_parser",
        "max_calls": MAX_CALLS,
        "call_count_labels": [1, 2, 3, 4],
        "none_zone_token": NONE_ZONE,
        "fields": label_vocab,
    }


def check_vocab_coverage(
    rows: list[dict[str, Any]],
    label_schema: Mapping[str, Any],
    *,
    split: str,
) -> None:
    """Assert that validation/test use no primitive unseen in train."""

    allowed = {
        field: {
            value_key(value)
            for value in values
        }
        for field, values
        in label_schema["fields"].items()
    }

    failures = []

    for row in rows:
        for call_index, call in enumerate(
            row["canonical_calls"]
        ):
            labels = extract_call_labels(call)

            for field, value in labels.items():
                if field not in allowed:
                    continue

                if value_key(value) not in allowed[field]:
                    failures.append(
                        (
                            row["example_id"],
                            call_index,
                            field,
                            value,
                        )
                    )

    if failures:
        raise LabelError(
            f"{split}: primitive OOV detected: "
            f"{failures[:10]}"
        )


def check_roundtrip(
    rows: list[dict[str, Any]],
    *,
    split: str,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    """Assert factorized labels reconstruct canonical gold exactly."""

    for row in rows:
        if row["call_count"] > MAX_CALLS:
            raise LabelError(
                f"{row['example_id']}: "
                f"call_count exceeds MAX_CALLS={MAX_CALLS}"
            )

        reconstructed = [
            assemble_call(
                extract_call_labels(call)
            )
            for call in row["canonical_calls"]
        ]

        gold_payload = {
            "calls": row["canonical_calls"],
        }

        reconstructed_payload = {
            "calls": reconstructed,
        }

        gold_json = canonical.canonical_json(
            gold_payload,
            schema=schema,
            registry=registry,
        )

        reconstructed_json = canonical.canonical_json(
            reconstructed_payload,
            schema=schema,
            registry=registry,
        )

        if gold_json != reconstructed_json:
            raise LabelError(
                f"{split}: roundtrip mismatch: "
                f"{row['example_id']}"
            )


def main() -> int:
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    schema = canonical.load_json_object(
        SCHEMA_PATH
    )

    registry = canonical.load_json_object(
        REGISTRY_PATH
    )

    canonical.validate_registry(
        schema,
        registry,
    )

    label_schema = build_label_schema(train)

    # IMPORTANT:
    # vocab is constructed from train only.
    check_vocab_coverage(
        validation,
        label_schema,
        split="validation",
    )

    check_vocab_coverage(
        test,
        label_schema,
        split="test",
    )

    for split, rows in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        check_roundtrip(
            rows,
            split=split,
            schema=schema,
            registry=registry,
        )

        print(
            f"{split} roundtrip PASS: "
            f"{len(rows)} examples"
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            label_schema,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nlabel vocab sizes:")

    for field, values in label_schema["fields"].items():
        print(
            f"  {field}: {len(values)}"
        )

    print(f"\nMAX_CALLS: {MAX_CALLS}")
    print(f"label schema: {OUTPUT_PATH}")
    print(
        "PASS: factorized label contract "
        "covers all 597 source groups"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
