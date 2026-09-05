#!/usr/bin/env python3
"""Validate, deduplicate, and split the augmented Korean benchmark v0.3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import canonical_vehicle_api as canonical


RESEARCH_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ORIGINAL_ROOT = RESEARCH_ROOT / "data" / "processed" / "macslu_korean_v0.2"
DEFAULT_CANDIDATES = (
    RESEARCH_ROOT / "data" / "synthetic" / "synthetic_candidates_v0.3.jsonl"
)
DEFAULT_ORIGINAL_POOL = (
    RESEARCH_ROOT / "data" / "synthetic" / "original_source_pool_v0.3.jsonl"
)
DEFAULT_SYNTHETIC_VALID = (
    RESEARCH_ROOT / "data" / "synthetic" / "synthetic_valid_v0.3.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    RESEARCH_ROOT / "data" / "processed" / "macslu_korean_augmented_v0.3"
)
DEFAULT_SCHEMA = RESEARCH_ROOT / "schema" / "vehicle_api_schema.v0.1.0.json"
DEFAULT_REGISTRY = RESEARCH_ROOT / "schema" / "vehicle_api_registry.v0.1.0.json"

SEED = 20260905
BENCHMARK_VERSION = "macslu_korean_augmented_v0.3"
SOURCE_BENCHMARK_VERSION = "macslu_korean_v0.2"
GENERATOR = "astra_vehicle_aug_v0.3"
GENERATOR_VERSION = "0.3.0"

SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}
ORIGINAL_SPLIT_COUNTS = {"train": 353, "validation": 43, "test": 43}
ORIGINAL_INPUT_SHA256 = {
    "train": "8779e51ab4c6f2fe67b861eb665a1f5108b95c8a150b963c8e9581f921ccada6",
    "validation": "96797a134a83d62b34fdc7ceb8b45164b10904d61381cdd43c1320332609a8a2",
    "test": "e5243ae656207b69df3209097898df0d227746c6723653a80760b13cf0729461",
}
SYNTHETIC_TARGETS = {1: 500, 2: 800, 3: 600, 4: 400}

FUNCTION_FAMILIES = {
    "set_hvac_power": "HVAC",
    "set_hvac_temperature": "HVAC",
    "set_hvac_fan_speed": "HVAC",
    "set_window_position": "Aperture",
    "set_sunroof_position": "Aperture",
    "set_sunshade_position": "Aperture",
    "set_seat_climate": "Seat",
    "set_seat_massage": "Seat",
}
FAMILY_ORDER = ("HVAC", "Aperture", "Seat")

MIN_SYNTHETIC_NORMALIZED_CHARS = 6
MIN_SYNTHETIC_HANGUL_SYLLABLES = 4
HANGUL_RE = re.compile(r"[가-힣]")
CJK_IDEOGRAPH_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
INCOMPLETE_ENDING_RE = re.compile(r"(?:그리고|한 다음|하면서|하고|이며|또)\s*$")
ARTIFACT_PATTERNS = (
    re.compile(r"[{}<>]"),
    re.compile(r"\$"),
    re.compile(r"%\([^)]+\)[a-zA-Z]"),
    re.compile(r"__[A-Za-z0-9_]+__"),
    re.compile(r"(?:TODO|FIXME|TBD|PLACEHOLDER|TEMPLATE|undefined|None|null)", re.I),
    re.compile(r"\[(?:zone|target|state|feature|setting|value)[^]]*]", re.I),
    re.compile(r"\b(?:set_hvac|set_window|set_sunroof|set_sunshade|set_seat)_"),
)

HVAC_ZONE_ATOMS = {
    # Omitted HVAC scope conservatively overlaps every explicit HVAC zone.
    "__IMPLICIT__": frozenset({"driver", "front_passenger", "rear"}),
    "driver": frozenset({"driver"}),
    "front_passenger": frozenset({"front_passenger"}),
    "rear": frozenset({"rear"}),
    "all": frozenset({"driver", "front_passenger", "rear"}),
}
WINDOW_ZONE_ATOMS = {
    "driver": frozenset({"driver"}),
    "front_passenger": frozenset({"front_passenger"}),
    "rear_left": frozenset({"rear_left"}),
    "rear_right": frozenset({"rear_right"}),
    "front_row": frozenset({"driver", "front_passenger"}),
    "rear_row": frozenset({"rear_left", "rear_right"}),
    "left_side": frozenset({"driver", "rear_left"}),
    "right_side": frozenset({"front_passenger", "rear_right"}),
    "all": frozenset({"driver", "front_passenger", "rear_left", "rear_right"}),
}
SEAT_ZONE_ATOMS = {
    "driver": frozenset({"driver"}),
    "front_passenger": frozenset({"front_passenger"}),
    "rear_left": frozenset({"rear_left"}),
    "rear_right": frozenset({"rear_right"}),
    "rear_row": frozenset({"rear_left", "rear_right"}),
    "all": frozenset({"driver", "front_passenger", "rear_left", "rear_right"}),
}


class BuildError(ValueError):
    """Raised when an input or release invariant is violated."""


def normalize_text(text: str) -> str:
    """Apply the exact v0.2 normalization policy."""

    return re.sub(r"\s+", " ", text.strip())


def canonical_calls_key(calls: Any) -> str:
    """Return an order-sensitive, key-order-insensitive gold key."""

    return json.dumps(
        calls,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise BuildError(f"cannot read {path}: {error}") from error

    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BuildError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise BuildError(f"{path}:{line_number}: expected JSON object")
            records.append(value)

    return records


def load_contract(
    schema_path: Path,
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = canonical.load_json_object(schema_path)
    registry = canonical.load_json_object(registry_path)
    canonical.validate_registry(schema, registry)

    expected_functions = set(FUNCTION_FAMILIES)
    actual_functions = canonical.schema_function_names(schema)
    if actual_functions != expected_functions:
        raise BuildError(
            "unexpected canonical function set: "
            f"expected={sorted(expected_functions)}, actual={sorted(actual_functions)}"
        )

    return schema, registry


def _typed_value(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _flatten_leaves(value: Any, path: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _flatten_leaves(value[key], f"{path}.{key}")
        return
    yield path, value


def extract_api_primitives(
    call: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> set[str]:
    """Extract every typed API leaf, including omitted optional arguments.

    Tokens are function-qualified so, for example, a seat level of 1 does not
    accidentally cover a fan level of 1. This intentionally does not use the
    narrower mmBERT v0.2 factorizer.
    """

    function_name = call["function"]
    arguments = call["arguments"]
    argument_order = registry["functions"][function_name]["argument_order"]
    primitives = {f"function={_typed_value(function_name)}"}

    for argument_name in argument_order:
        path = f"{function_name}.arguments.{argument_name}"
        if argument_name not in arguments:
            primitives.add(f"{path}=<ABSENT>")
            continue
        for leaf_path, value in _flatten_leaves(arguments[argument_name], path):
            primitives.add(f"{leaf_path}={_typed_value(value)}")

    return primitives


def record_api_primitives(
    record: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> set[str]:
    result: set[str] = set()
    for call in record["canonical_calls"]:
        result.update(extract_api_primitives(call, registry))
    return result


def _resource_scope(
    call: Mapping[str, Any],
) -> tuple[str, frozenset[str]]:
    function_name = call["function"]
    arguments = call["arguments"]

    if function_name in {
        "set_hvac_power",
        "set_hvac_temperature",
        "set_hvac_fan_speed",
    }:
        zone = arguments.get("zone", "__IMPLICIT__")
        return function_name, HVAC_ZONE_ATOMS[zone]

    if function_name == "set_window_position":
        return function_name, WINDOW_ZONE_ATOMS[arguments["zone"]]

    if function_name in {"set_sunroof_position", "set_sunshade_position"}:
        return function_name, frozenset({"singleton"})

    if function_name == "set_seat_climate":
        namespace = f"{function_name}:{arguments['feature']}"
        return namespace, SEAT_ZONE_ATOMS[arguments["zone"]]

    if function_name == "set_seat_massage":
        return function_name, SEAT_ZONE_ATOMS[arguments["zone"]]

    raise BuildError(f"unregistered resource function: {function_name!r}")


def find_effective_resource_conflicts(
    calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Find calls that redundantly or contradictorily touch overlapping scope."""

    resources = [_resource_scope(call) for call in calls]
    conflicts: list[dict[str, Any]] = []

    for left in range(len(calls)):
        left_namespace, left_atoms = resources[left]
        for right in range(left + 1, len(calls)):
            right_namespace, right_atoms = resources[right]
            overlap = sorted(left_atoms & right_atoms)
            if left_namespace != right_namespace or not overlap:
                continue
            classification = (
                "redundant"
                if canonical_calls_key(calls[left]) == canonical_calls_key(calls[right])
                else "overlapping_or_conflicting"
            )
            conflicts.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "resource": left_namespace,
                    "overlap": overlap,
                    "classification": classification,
                }
            )

    return conflicts


def _utterance_errors(text: Any, *, synthetic: bool) -> list[str]:
    if not isinstance(text, str):
        return ["utterance_type: utterance_ko must be a string"]

    normalized = normalize_text(text)
    errors: list[str] = []
    if not normalized:
        errors.append("empty_utterance: utterance_ko is empty")
        return errors
    if not HANGUL_RE.search(normalized):
        errors.append("non_korean_utterance: no Hangul syllable found")
    if CJK_IDEOGRAPH_RE.search(normalized):
        errors.append("cjk_ideograph: CJK ideograph found in Korean utterance")
    if "\ufffd" in text or any(ord(character) < 32 for character in text):
        errors.append("broken_utterance: replacement or control character found")

    if synthetic:
        hangul_count = len(HANGUL_RE.findall(normalized))
        if (
            len(normalized) < MIN_SYNTHETIC_NORMALIZED_CHARS
            or hangul_count < MIN_SYNTHETIC_HANGUL_SYLLABLES
        ):
            errors.append(
                "short_utterance: synthetic utterance is not minimally complete"
            )
        if INCOMPLETE_ENDING_RE.search(normalized):
            errors.append("incomplete_utterance: utterance ends in a connector")
        for pattern in ARTIFACT_PATTERNS:
            if pattern.search(text):
                errors.append(
                    f"template_artifact: matched artifact pattern {pattern.pattern!r}"
                )
                break

    return errors


def _canonical_record_errors(
    record: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    check_resource_conflicts: bool,
) -> list[str]:
    errors: list[str] = []
    example_id = record.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        errors.append("example_id: must be a non-empty string")

    calls = record.get("canonical_calls")
    call_count = record.get("call_count")
    if not isinstance(calls, list):
        errors.append("canonical_calls_type: canonical_calls must be a list")
    if not isinstance(call_count, int) or isinstance(call_count, bool):
        errors.append("call_count_type: call_count must be an integer")
    elif not 1 <= call_count <= 4:
        errors.append("call_count_range: call_count must be between 1 and 4")
    if (
        isinstance(calls, list)
        and isinstance(call_count, int)
        and not isinstance(call_count, bool)
    ):
        if call_count != len(calls):
            errors.append(
                "call_count_mismatch: "
                f"call_count={call_count}, canonical_calls={len(calls)}"
            )

    if isinstance(calls, list) and calls:
        try:
            canonicalized = canonical.canonicalize_payload(
                {"calls": calls},
                schema=schema,
                registry=registry,
            )
        except canonical.CanonicalValidationError as error:
            errors.append(f"canonical_schema: {error}")
        else:
            if canonicalized["calls"] != calls:
                errors.append("canonical_representation: calls are not canonical")
            resource_conflicts = (
                find_effective_resource_conflicts(calls)
                if check_resource_conflicts
                else []
            )
            if resource_conflicts:
                errors.append(
                    "effective_resource_conflict: "
                    + json.dumps(
                        resource_conflicts,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
    elif isinstance(calls, list):
        errors.append("canonical_calls_empty: canonical_calls must not be empty")

    errors.extend(_utterance_errors(record.get("utterance_ko"), synthetic=False))
    return errors


def _synthetic_metadata_errors(
    record: Mapping[str, Any],
    *,
    seed: int,
) -> list[str]:
    errors: list[str] = []
    example_id = record.get("example_id")

    if record.get("source_type") != "synthetic":
        errors.append("source_type: expected 'synthetic'")
    if record.get("source_split") != "synthetic":
        errors.append("source_split: expected 'synthetic'")
    if record.get("source_group_id") != example_id:
        errors.append("source_group_id: must equal example_id for synthetic data")
    if "benchmark_split" in record or "benchmark_version" in record:
        errors.append("premature_benchmark_assignment: candidate is already split")

    metadata = record.get("synthetic_generation")
    if not isinstance(metadata, Mapping):
        errors.append("synthetic_generation: must be an object")
        return errors

    expected_scalars = {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
    }
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            errors.append(
                f"synthetic_generation.{key}: expected {expected!r}, "
                f"got {metadata.get(key)!r}"
            )

    for key in (
        "generation_family_id",
        "template_id",
        "function_family_pattern",
    ):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"synthetic_generation.{key}: must be non-empty string")

    calls = record.get("canonical_calls")
    if (
        isinstance(calls, list)
        and calls
        and all(
            isinstance(call, Mapping) and call.get("function") in FUNCTION_FAMILIES
            for call in calls
        )
    ):
        present_families = {FUNCTION_FAMILIES[call["function"]] for call in calls}
        expected_pattern = "+".join(
            family for family in FAMILY_ORDER if family in present_families
        )
        if metadata.get("function_family_pattern") != expected_pattern:
            errors.append(
                "synthetic_generation.function_family_pattern: "
                f"expected {expected_pattern!r} from canonical_calls"
            )

    return errors


def validate_synthetic_record(
    record: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    seed: int,
) -> list[str]:
    errors = _canonical_record_errors(
        record,
        schema=schema,
        registry=registry,
        check_resource_conflicts=True,
    )
    errors.extend(_utterance_errors(record.get("utterance_ko"), synthetic=True))

    # Avoid listing the common basic language errors twice.
    errors = list(dict.fromkeys(errors))
    errors.extend(_synthetic_metadata_errors(record, seed=seed))
    return errors


def prepare_original_source_pool(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    expected_counts: Mapping[str, int] | None = ORIGINAL_SPLIT_COUNTS,
) -> list[dict[str, Any]]:
    """Combine v0.2 splits while retaining their former assignment as provenance."""

    if set(rows_by_split) != set(SPLITS):
        raise BuildError(f"original inputs must contain exactly {list(SPLITS)}")

    if expected_counts is not None:
        actual = {split: len(rows_by_split[split]) for split in SPLITS}
        if actual != dict(expected_counts):
            raise BuildError(
                f"unexpected v0.2 split counts: expected={dict(expected_counts)}, "
                f"actual={actual}"
            )

    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    normalized_rows: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        for input_row in rows_by_split[split]:
            errors = _canonical_record_errors(
                input_row,
                schema=schema,
                registry=registry,
                check_resource_conflicts=False,
            )
            if errors:
                raise BuildError(
                    f"invalid original {input_row.get('example_id', '<missing>')}: "
                    + "; ".join(errors)
                )

            row = copy.deepcopy(dict(input_row))
            example_id = row["example_id"]
            if example_id in seen_ids:
                raise BuildError(f"duplicate original example_id: {example_id}")
            seen_ids.add(example_id)

            if row.get("benchmark_split") != split:
                raise BuildError(
                    f"{example_id}: benchmark_split does not match source file {split}"
                )
            if row.get("benchmark_version") != SOURCE_BENCHMARK_VERSION:
                raise BuildError(
                    f"{example_id}: expected benchmark_version "
                    f"{SOURCE_BENCHMARK_VERSION!r}"
                )

            utterance_key = normalize_text(row["utterance_ko"])
            previous = normalized_rows.get(utterance_key)
            if previous is not None:
                relation = (
                    "same gold"
                    if canonical_calls_key(previous["canonical_calls"])
                    == canonical_calls_key(row["canonical_calls"])
                    else "conflicting gold"
                )
                raise BuildError(
                    "v0.2 source pool is not globally unique: "
                    f"{utterance_key!r} ({relation})"
                )
            normalized_rows[utterance_key] = row

            if "previous_benchmark_split" in row or "previous_benchmark_version" in row:
                raise BuildError(
                    f"{example_id}: reserved previous benchmark provenance exists"
                )
            row["previous_benchmark_split"] = row.pop("benchmark_split")
            row["previous_benchmark_version"] = row.pop("benchmark_version")
            if "source_type" in row and row["source_type"] != "original":
                raise BuildError(f"{example_id}: conflicting source_type")
            row["source_type"] = "original"
            output.append(row)

    output.sort(key=lambda row: row["example_id"])
    return output


def _reason_code(reason: str) -> str:
    return reason.split(":", 1)[0]


def validate_and_deduplicate_synthetic(
    candidates: Sequence[Mapping[str, Any]],
    originals: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return release-valid synthetic rows and an explicit rejection report."""

    original_ids = {row["example_id"] for row in originals}
    original_by_text = {normalize_text(row["utterance_ko"]): row for row in originals}

    candidate_id_counts = Counter(
        row.get("example_id")
        for row in candidates
        if isinstance(row.get("example_id"), str)
        and bool(row.get("example_id").strip())
    )
    duplicate_candidate_ids = {
        example_id for example_id, count in candidate_id_counts.items() if count > 1
    }

    invalid_examples: list[dict[str, Any]] = []
    invalid_reason_counts: Counter[str] = Counter()
    structurally_valid: list[dict[str, Any]] = []

    for input_index, input_row in enumerate(candidates, 1):
        row = copy.deepcopy(dict(input_row))
        errors = validate_synthetic_record(
            row,
            schema=schema,
            registry=registry,
            seed=seed,
        )
        example_id = row.get("example_id")
        if isinstance(example_id, str):
            if example_id in original_ids:
                errors.append("global_example_id_collision: collides with original")
            if example_id in duplicate_candidate_ids:
                errors.append("duplicate_candidate_example_id: appears more than once")

        errors = list(dict.fromkeys(errors))
        if errors:
            invalid_examples.append(
                {
                    "candidate_line": input_index,
                    "example_id": example_id,
                    "reasons": errors,
                }
            )
            invalid_reason_counts.update(_reason_code(reason) for reason in errors)
            continue

        row["utterance_ko"] = normalize_text(row["utterance_ko"])
        structurally_valid.append(row)

    candidates_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in structurally_valid:
        candidates_by_text[normalize_text(row["utterance_ko"])].append(row)

    kept: list[dict[str, Any]] = []
    duplicate_examples: list[dict[str, Any]] = []
    conflict_examples: list[dict[str, Any]] = []

    for utterance_key in sorted(candidates_by_text):
        members = sorted(
            candidates_by_text[utterance_key], key=lambda row: row["example_id"]
        )
        original = original_by_text.get(utterance_key)

        if original is not None:
            original_gold = canonical_calls_key(original["canonical_calls"])
            conflicting_members = [
                row
                for row in members
                if canonical_calls_key(row["canonical_calls"]) != original_gold
            ]
            matching_members = [
                row
                for row in members
                if canonical_calls_key(row["canonical_calls"]) == original_gold
            ]
            for row in matching_members:
                duplicate_examples.append(
                    {
                        "example_id": row["example_id"],
                        "duplicate_of": original["example_id"],
                        "normalized_utterance": utterance_key,
                        "scope": "original_synthetic",
                    }
                )
            if conflicting_members:
                gold_groups: dict[str, list[str]] = defaultdict(list)
                gold_groups[original_gold].append(original["example_id"])
                for row in conflicting_members:
                    gold_groups[canonical_calls_key(row["canonical_calls"])].append(
                        row["example_id"]
                    )
                conflict_examples.append(
                    {
                        "normalized_utterance": utterance_key,
                        "scope": "original_synthetic",
                        "gold_groups": [
                            {
                                "canonical_calls": json.loads(gold),
                                "example_ids": sorted(example_ids),
                            }
                            for gold, example_ids in sorted(gold_groups.items())
                        ],
                        "removed_synthetic_example_ids": sorted(
                            row["example_id"] for row in conflicting_members
                        ),
                    }
                )
            continue

        gold_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            gold_groups[canonical_calls_key(row["canonical_calls"])].append(row)

        if len(gold_groups) > 1:
            conflict_examples.append(
                {
                    "normalized_utterance": utterance_key,
                    "scope": "synthetic_synthetic",
                    "gold_groups": [
                        {
                            "canonical_calls": json.loads(gold),
                            "example_ids": sorted(
                                row["example_id"] for row in gold_members
                            ),
                        }
                        for gold, gold_members in sorted(gold_groups.items())
                    ],
                    "removed_synthetic_example_ids": [
                        row["example_id"] for row in members
                    ],
                }
            )
            continue

        representative = members[0]
        kept.append(representative)
        for row in members[1:]:
            duplicate_examples.append(
                {
                    "example_id": row["example_id"],
                    "duplicate_of": representative["example_id"],
                    "normalized_utterance": utterance_key,
                    "scope": "synthetic_synthetic",
                }
            )

    removed_conflict_ids = {
        example_id
        for conflict in conflict_examples
        for example_id in conflict["removed_synthetic_example_ids"]
    }
    kept.sort(key=lambda row: row["example_id"])

    report = {
        "generated_candidates": len(candidates),
        "rejected_invalid": len(invalid_examples),
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "invalid_examples": invalid_examples,
        "removed_duplicate": len(duplicate_examples),
        "removed_duplicate_by_scope": dict(
            sorted(Counter(item["scope"] for item in duplicate_examples).items())
        ),
        "duplicate_examples": duplicate_examples,
        "removed_conflict": len(removed_conflict_ids),
        "removed_conflict_by_scope": dict(
            sorted(
                Counter(
                    conflict["scope"]
                    for conflict in conflict_examples
                    for _ in conflict["removed_synthetic_example_ids"]
                ).items()
            )
        ),
        "conflicts": conflict_examples,
        "final_synthetic": len(kept),
    }

    accounted = (
        report["rejected_invalid"]
        + report["removed_duplicate"]
        + report["removed_conflict"]
        + report["final_synthetic"]
    )
    if accounted != len(candidates):
        raise BuildError(
            "synthetic validation accounting mismatch: "
            f"candidates={len(candidates)}, accounted={accounted}"
        )

    return kept, report


def assert_synthetic_targets(
    rows: Sequence[Mapping[str, Any]],
    targets: Mapping[int, int] = SYNTHETIC_TARGETS,
) -> dict[str, Any]:
    actual = Counter(int(row["call_count"]) for row in rows)
    shortfall = {
        str(call_count): target - actual.get(call_count, 0)
        for call_count, target in sorted(targets.items())
        if actual.get(call_count, 0) < target
    }
    excess = {
        str(call_count): actual.get(call_count, 0) - target
        for call_count, target in sorted(targets.items())
        if actual.get(call_count, 0) > target
    }
    result = {
        "target": {str(key): value for key, value in sorted(targets.items())},
        "actual": {str(key): actual.get(key, 0) for key in sorted(targets)},
        "shortfall": shortfall,
        "excess": excess,
        "exact_target_met": not shortfall and not excess,
    }
    if shortfall or excess:
        raise BuildError(
            "synthetic call-count target mismatch: "
            f"shortfall={shortfall}, excess={excess}"
        )
    return result


def _schema_enum(schema: Mapping[str, Any], definition: str) -> set[Any]:
    values = schema["$defs"][definition]["enum"]
    return set(values)


def _schema_branches(
    schema: Mapping[str, Any], definition: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for branch in schema["$defs"][definition]["oneOf"]:
        kind = branch["properties"]["kind"]["const"]
        result[kind] = branch
    return result


def _property_values(
    schema: Mapping[str, Any], property_schema: Mapping[str, Any]
) -> set[Any]:
    if "enum" in property_schema:
        return set(property_schema["enum"])
    if "const" in property_schema:
        return {property_schema["const"]}
    reference = property_schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        return _schema_enum(schema, reference.rsplit("/", 1)[-1])
    raise BuildError(
        f"coverage gate cannot enumerate schema property: {property_schema}"
    )


def _call_nested_value(call: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = call["arguments"]
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def audit_synthetic_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Enforce the quantitative coverage policy against the full API schema."""

    calls_by_function: dict[str, list[Mapping[str, Any]]] = {
        function_name: [] for function_name in FUNCTION_FAMILIES
    }
    for row in rows:
        for call in row["canonical_calls"]:
            calls_by_function[call["function"]].append(call)

    checks: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[str]] = {}

    def check(name: str, actual: Iterable[Any], expected: Iterable[Any]) -> None:
        actual_set = set(actual)
        expected_set = set(expected)
        missing = expected_set - actual_set
        checks[name] = {
            "expected": sorted((_typed_value(value) for value in expected_set)),
            "observed": sorted((_typed_value(value) for value in actual_set)),
            "missing": sorted((_typed_value(value) for value in missing)),
            "passed": not missing,
        }
        if missing:
            failures[name] = checks[name]["missing"]

    function_counts = {
        function_name: len(function_calls)
        for function_name, function_calls in sorted(calls_by_function.items())
    }
    check(
        "functions",
        (name for name, count in function_counts.items() if count),
        FUNCTION_FAMILIES,
    )
    check(
        "function_family_patterns",
        (row["synthetic_generation"]["function_family_pattern"] for row in rows),
        {
            "HVAC",
            "Aperture",
            "Seat",
            "HVAC+Aperture",
            "HVAC+Seat",
            "Aperture+Seat",
            "HVAC+Aperture+Seat",
        },
    )

    nonzero_function_counts = [count for count in function_counts.values() if count]
    function_ratio = (
        max(nonzero_function_counts) / min(nonzero_function_counts)
        if len(nonzero_function_counts) == len(FUNCTION_FAMILIES)
        else None
    )
    if function_ratio is None or function_ratio > 1.10 + 1e-12:
        failures["function_frequency_max_min_ratio"] = [str(function_ratio)]

    hvac_zones = _schema_enum(schema, "hvac_zone") | {"<ABSENT>"}
    for function_name in (
        "set_hvac_power",
        "set_hvac_temperature",
        "set_hvac_fan_speed",
    ):
        check(
            f"{function_name}.zone",
            (
                call["arguments"].get("zone", "<ABSENT>")
                for call in calls_by_function[function_name]
            ),
            hvac_zones,
        )

    check(
        "set_window_position.zone",
        (
            call["arguments"]["zone"]
            for call in calls_by_function["set_window_position"]
        ),
        _schema_enum(schema, "window_zone"),
    )
    for function_name in ("set_seat_climate", "set_seat_massage"):
        check(
            f"{function_name}.zone",
            (call["arguments"]["zone"] for call in calls_by_function[function_name]),
            _schema_enum(schema, "seat_zone"),
        )

    on_off = _schema_enum(schema, "on_off")
    check(
        "set_hvac_power.state",
        (call["arguments"]["state"] for call in calls_by_function["set_hvac_power"]),
        on_off,
    )
    check(
        "set_seat_climate.feature",
        (
            call["arguments"]["feature"]
            for call in calls_by_function["set_seat_climate"]
        ),
        {"heating", "ventilation"},
    )

    target_definitions = {
        "set_hvac_temperature": "temperature_target",
        "set_hvac_fan_speed": "fan_speed_target",
        "set_window_position": "aperture_target",
        "set_sunroof_position": "aperture_target",
        "set_sunshade_position": "sunshade_target",
    }
    for function_name, definition in target_definitions.items():
        branches = _schema_branches(schema, definition)
        function_calls = calls_by_function[function_name]
        check(
            f"{function_name}.target.kind",
            (_call_nested_value(call, ("target", "kind")) for call in function_calls),
            branches,
        )
        for kind, branch in branches.items():
            kind_calls = [
                call
                for call in function_calls
                if _call_nested_value(call, ("target", "kind")) == kind
            ]
            for property_name in ("direction", "magnitude"):
                property_schema = branch["properties"].get(property_name)
                if property_schema is None:
                    continue
                check(
                    f"{function_name}.target.{kind}.{property_name}",
                    (
                        _call_nested_value(call, ("target", property_name))
                        for call in kind_calls
                    ),
                    _property_values(schema, property_schema),
                )
            if kind in {"named", "extreme"}:
                check(
                    f"{function_name}.target.{kind}.value",
                    (
                        _call_nested_value(call, ("target", "value"))
                        for call in kind_calls
                    ),
                    _property_values(schema, branch["properties"]["value"]),
                )

    seat_branches = _schema_branches(schema, "seat_setting")
    for function_name in ("set_seat_climate", "set_seat_massage"):
        function_calls = calls_by_function[function_name]
        check(
            f"{function_name}.setting.kind",
            (_call_nested_value(call, ("setting", "kind")) for call in function_calls),
            seat_branches,
        )
        for kind, branch in seat_branches.items():
            kind_calls = [
                call
                for call in function_calls
                if _call_nested_value(call, ("setting", "kind")) == kind
            ]
            for property_name in ("value", "direction", "magnitude"):
                property_schema = branch["properties"].get(property_name)
                if property_schema is None or (
                    property_name == "value" and kind == "absolute_level"
                ):
                    continue
                check(
                    f"{function_name}.setting.{kind}.{property_name}",
                    (
                        _call_nested_value(call, ("setting", property_name))
                        for call in kind_calls
                    ),
                    _property_values(schema, property_schema),
                )

    temperature_values = {16 + 0.5 * step for step in range(33)}
    check(
        "set_hvac_temperature.absolute_grid",
        (
            _call_nested_value(call, ("target", "value"))
            for call in calls_by_function["set_hvac_temperature"]
            if _call_nested_value(call, ("target", "kind")) == "absolute"
        ),
        temperature_values,
    )
    check(
        "set_hvac_fan_speed.absolute_levels",
        (
            _call_nested_value(call, ("target", "value"))
            for call in calls_by_function["set_hvac_fan_speed"]
            if _call_nested_value(call, ("target", "kind")) == "absolute"
        ),
        range(1, 9),
    )
    for function_name in ("set_seat_climate", "set_seat_massage"):
        check(
            f"{function_name}.absolute_levels",
            (
                _call_nested_value(call, ("setting", "value"))
                for call in calls_by_function[function_name]
                if _call_nested_value(call, ("setting", "kind")) == "absolute_level"
            ),
            range(1, 4),
        )

    absolute_percent_grid = {0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100}
    relative_percent_grid = {5, 10, 15, 20, 25, 30, 40, 50}
    for function_name in (
        "set_window_position",
        "set_sunroof_position",
        "set_sunshade_position",
    ):
        function_calls = calls_by_function[function_name]
        check(
            f"{function_name}.absolute_percent_grid",
            (
                _call_nested_value(call, ("target", "value"))
                for call in function_calls
                if _call_nested_value(call, ("target", "kind")) == "absolute_percent"
            ),
            absolute_percent_grid,
        )
        check(
            f"{function_name}.relative_percent_grid",
            (
                _call_nested_value(call, ("target", "value"))
                for call in function_calls
                if _call_nested_value(call, ("target", "kind")) == "relative_percent"
            ),
            relative_percent_grid,
        )

    report = {
        "passed": not failures,
        "function_call_counts": function_counts,
        "function_frequency_max_min_ratio": (
            round(function_ratio, 6) if function_ratio is not None else None
        ),
        "maximum_allowed_function_ratio": 1.10,
        "checks": checks,
        "failures": failures,
    }
    if failures:
        raise BuildError(
            "synthetic quantitative coverage gate failed: "
            + json.dumps(failures, ensure_ascii=False, sort_keys=True)
        )
    return report


def audit_canonical_signature_policy(
    originals: Sequence[Mapping[str, Any]],
    synthetic_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original_signatures: dict[str, list[str]] = defaultdict(list)
    synthetic_signatures: dict[str, list[str]] = defaultdict(list)
    for row in originals:
        original_signatures[canonical_calls_key(row["canonical_calls"])].append(
            row["example_id"]
        )
    for row in synthetic_rows:
        synthetic_signatures[canonical_calls_key(row["canonical_calls"])].append(
            row["example_id"]
        )

    duplicate_synthetic = {
        signature: sorted(example_ids)
        for signature, example_ids in synthetic_signatures.items()
        if len(example_ids) > 1
    }
    overlap = sorted(set(original_signatures) & set(synthetic_signatures))
    report = {
        "ordered_signature_definition": "ordered canonical_calls; JSON object keys sorted",
        "original_examples": len(originals),
        "unique_original_signatures": len(original_signatures),
        "synthetic_examples": len(synthetic_rows),
        "unique_synthetic_signatures": len(synthetic_signatures),
        "duplicate_synthetic_signature_count": len(duplicate_synthetic),
        "duplicate_synthetic_signatures": [
            {
                "canonical_calls": json.loads(signature),
                "example_ids": example_ids,
            }
            for signature, example_ids in sorted(duplicate_synthetic.items())
        ],
        "synthetic_original_signature_overlap_count": len(overlap),
        "synthetic_original_signature_overlaps": [
            {
                "canonical_calls": json.loads(signature),
                "original_example_ids": sorted(original_signatures[signature]),
                "synthetic_example_ids": sorted(synthetic_signatures[signature]),
            }
            for signature in overlap
        ],
        "all_synthetic_signatures_unique": not duplicate_synthetic,
        "all_synthetic_signatures_novel": not overlap,
    }
    if duplicate_synthetic or overlap:
        raise BuildError(
            "synthetic ordered canonical signatures must be unique and novel: "
            f"duplicates={len(duplicate_synthetic)}, original_overlap={len(overlap)}"
        )
    return report


def _stable_tie(seed: int, *parts: str) -> int:
    value = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _integer_targets(total: int) -> dict[str, int]:
    raw = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
    result = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = total - sum(result.values())
    order = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - result[split]), SPLITS.index(split)),
    )
    for split in order[:remaining]:
        result[split] += 1
    return result


def _balance_features(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        for call in row["canonical_calls"]:
            function_name = call["function"]
            result[f"function:{function_name}"] += 1
            result[f"family:{FUNCTION_FAMILIES[function_name]}"] += 1
    return result


def _make_split_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        if row["source_type"] == "original":
            group_id = f"original::{row['example_id']}"
        else:
            family_id = row["synthetic_generation"]["generation_family_id"]
            group_id = f"synthetic::{family_id}"
        grouped[group_id].append(row)

    groups: list[dict[str, Any]] = []
    for group_id in sorted(grouped):
        members = sorted(grouped[group_id], key=lambda row: row["example_id"])
        source_types = {row["source_type"] for row in members}
        call_counts = {int(row["call_count"]) for row in members}
        if len(source_types) != 1 or len(call_counts) != 1:
            raise BuildError(
                f"split group {group_id!r} crosses source_type or call_count strata"
            )
        if next(iter(source_types)) == "synthetic":
            templates = {row["synthetic_generation"]["template_id"] for row in members}
            skeletons = {
                tuple(call["function"] for call in row["canonical_calls"])
                for row in members
            }
            declared_patterns = {
                row["synthetic_generation"]["function_family_pattern"]
                for row in members
            }
            if (
                len(templates) != 1
                or len(skeletons) != 1
                or len(declared_patterns) != 1
            ):
                raise BuildError(
                    f"generation family {group_id!r} crosses template, ordered "
                    "function skeleton, or family-pattern boundaries"
                )

        primitives: set[str] = set()
        for row in members:
            primitives.update(record_api_primitives(row, registry))

        groups.append(
            {
                "group_id": group_id,
                "source_type": next(iter(source_types)),
                "call_count": next(iter(call_counts)),
                "rows": members,
                "size": len(members),
                "primitives": primitives,
                "balance_features": _balance_features(members),
            }
        )

    return groups


def group_aware_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    registry: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Split atomic families with source/call-count stratified greedy packing."""

    groups = _make_split_groups(rows, registry=registry)
    strata_totals: Counter[tuple[str, int]] = Counter(
        {
            (source_type, call_count): 0
            for source_type in ("original", "synthetic")
            for call_count in range(1, 5)
        }
    )
    for group in groups:
        strata_totals[(group["source_type"], group["call_count"])] += group["size"]

    stratum_targets = {
        stratum: _integer_targets(total) for stratum, total in strata_totals.items()
    }
    global_targets = _integer_targets(len(rows))
    total_balance = _balance_features(rows)

    assignments: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    stratum_counts: Counter[tuple[str, int, str]] = Counter()
    balance_counts = {split: Counter() for split in SPLITS}

    def assign(group: Mapping[str, Any], split: str) -> None:
        assignments[group["group_id"]] = split
        split_counts[split] += group["size"]
        stratum_counts[(group["source_type"], group["call_count"], split)] += group[
            "size"
        ]
        balance_counts[split].update(group["balance_features"])

    # Release evaluations cannot introduce unseen typed API leaves. Select a
    # deterministic compact set of train groups that covers the full pool first.
    uncovered = set().union(*(group["primitives"] for group in groups))
    unassigned = {group["group_id"]: group for group in groups}
    coverage_group_ids: list[str] = []

    while uncovered:
        eligible = [
            group for group in unassigned.values() if group["primitives"] & uncovered
        ]
        if not eligible:
            raise BuildError("cannot construct primitive-complete train split")
        chosen = min(
            eligible,
            key=lambda group: (
                -len(group["primitives"] & uncovered),
                group["size"],
                _stable_tie(seed, "coverage", group["group_id"]),
                group["group_id"],
            ),
        )
        assign(chosen, "train")
        coverage_group_ids.append(chosen["group_id"])
        uncovered.difference_update(chosen["primitives"])
        del unassigned[chosen["group_id"]]

    ordered_remaining = sorted(
        unassigned.values(),
        key=lambda group: (
            -group["size"],
            _stable_tie(seed, "group-order", group["group_id"]),
            group["group_id"],
        ),
    )

    def placement_score(group: Mapping[str, Any], split: str) -> tuple[Any, ...]:
        stratum = (group["source_type"], group["call_count"])
        target = stratum_targets[stratum][split]
        current = stratum_counts[(*stratum, split)]
        projected = current + group["size"]
        stratum_overflow = max(0, projected - target)
        stratum_fill = projected / target if target else float("inf")

        global_target = global_targets[split]
        global_projected = split_counts[split] + group["size"]
        global_overflow = max(0, global_projected - global_target)
        global_fill = (
            global_projected / global_target if global_target else float("inf")
        )

        balance_fills = []
        for feature, amount in group["balance_features"].items():
            desired = total_balance[feature] * SPLIT_RATIOS[split]
            balance_fills.append(
                (balance_counts[split][feature] + amount) / desired
                if desired
                else float("inf")
            )
        balance_fill = statistics.mean(balance_fills) if balance_fills else 0.0

        return (
            bool(stratum_overflow),
            stratum_overflow / max(target, 1),
            stratum_fill,
            bool(global_overflow),
            global_overflow / max(global_target, 1),
            global_fill,
            balance_fill,
            _stable_tie(seed, "placement", group["group_id"], split),
        )

    for group in ordered_remaining:
        chosen_split = min(SPLITS, key=lambda split: placement_score(group, split))
        assign(group, chosen_split)

    output = {split: [] for split in SPLITS}
    for group in groups:
        split = assignments[group["group_id"]]
        for input_row in group["rows"]:
            row = copy.deepcopy(dict(input_row))
            row["benchmark_split"] = split
            row["benchmark_version"] = BENCHMARK_VERSION
            output[split].append(row)

    for split in SPLITS:
        output[split].sort(key=lambda row: row["example_id"])

    diagnostics = {
        "method": "primitive-cover-first seeded greedy atomic group packing",
        "ratios": SPLIT_RATIOS,
        "target_counts": global_targets,
        "actual_counts": {split: len(output[split]) for split in SPLITS},
        "stratum_targets": {
            f"{source_type}|{call_count}": targets
            for (source_type, call_count), targets in sorted(stratum_targets.items())
        },
        "stratum_actual": {
            f"{source_type}|{call_count}": {
                split: stratum_counts[(source_type, call_count, split)]
                for split in SPLITS
            }
            for source_type, call_count in sorted(strata_totals)
        },
        "atomic_group_count": len(groups),
        "primitive_coverage_train_group_count": len(coverage_group_ids),
        "primitive_coverage_train_group_ids": coverage_group_ids,
    }
    return output, diagnostics


def audit_split_integrity(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    registry: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    text_sets: dict[str, set[str]] = {}
    family_sets: dict[str, set[str]] = {}
    signature_sets: dict[str, set[str]] = {}

    expected_by_id = {row["example_id"]: row for row in expected_rows}
    if len(expected_by_id) != len(expected_rows):
        raise BuildError("expected final pool contains duplicate example_id")
    final_rows = [row for split in SPLITS for row in splits[split]]
    final_id_counts = Counter(row["example_id"] for row in final_rows)
    duplicate_final_ids = sorted(
        example_id for example_id, count in final_id_counts.items() if count != 1
    )
    missing_ids = sorted(set(expected_by_id) - set(final_id_counts))
    extra_ids = sorted(set(final_id_counts) - set(expected_by_id))
    if duplicate_final_ids or missing_ids or extra_ids:
        raise BuildError(
            "final split union identity failed: "
            f"non_singleton={duplicate_final_ids}, missing={missing_ids}, extra={extra_ids}"
        )

    for row in final_rows:
        if row.get("benchmark_version") != BENCHMARK_VERSION:
            raise BuildError(f"{row['example_id']}: wrong final benchmark_version")
        if row.get("benchmark_split") not in SPLITS:
            raise BuildError(f"{row['example_id']}: wrong final benchmark_split")
        unsplit_row = copy.deepcopy(dict(row))
        unsplit_row.pop("benchmark_version")
        unsplit_row.pop("benchmark_split")
        if unsplit_row != expected_by_id[row["example_id"]]:
            raise BuildError(
                f"{row['example_id']}: final split mutated non-split record fields"
            )

    for split in SPLITS:
        rows = splits[split]
        for row in rows:
            if row.get("benchmark_split") != split:
                raise BuildError(
                    f"{row['example_id']}: benchmark_split does not match {split} file"
                )
        texts = {normalize_text(row["utterance_ko"]) for row in rows}
        if len(texts) != len(rows):
            raise BuildError(f"{split}: internal normalized utterance duplicate")
        text_sets[split] = texts
        family_sets[split] = {
            row["synthetic_generation"]["generation_family_id"]
            for row in rows
            if row["source_type"] == "synthetic"
        }
        signature_sets[split] = {
            canonical_calls_key(row["canonical_calls"]) for row in rows
        }

    utterance_overlap: dict[str, int] = {}
    family_overlap: dict[str, int] = {}
    signature_overlap: dict[str, int] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        key = f"{left}_{right}"
        utterance_overlap[key] = len(text_sets[left] & text_sets[right])
        family_overlap[key] = len(family_sets[left] & family_sets[right])
        signature_overlap[key] = len(signature_sets[left] & signature_sets[right])

    if any(utterance_overlap.values()):
        raise BuildError(f"normalized utterance split overlap: {utterance_overlap}")
    if any(family_overlap.values()):
        raise BuildError(f"synthetic generation-family overlap: {family_overlap}")

    primitive_sets = {
        split: set().union(
            *(record_api_primitives(row, registry) for row in splits[split])
        )
        for split in SPLITS
    }
    primitive_missing = {
        split: sorted(primitive_sets[split] - primitive_sets["train"])
        for split in ("validation", "test")
    }
    if any(primitive_missing.values()):
        raise BuildError(f"evaluation primitive OOV exists: {primitive_missing}")

    synthetic_family_sizes = Counter(
        row["synthetic_generation"]["generation_family_id"]
        for row in expected_rows
        if row["source_type"] == "synthetic"
    )
    largest_atomic_group_size = max([1, *synthetic_family_sizes.values()])
    total = len(expected_rows)
    largest_atomic_share = largest_atomic_group_size / total
    split_ratio_report: dict[str, Any] = {}
    ratio_failures: dict[str, float] = {}
    for split in SPLITS:
        actual_share = len(splits[split]) / total
        deviation = abs(actual_share - SPLIT_RATIOS[split])
        within_one_point = deviation <= 0.01 + 1e-12
        atomic_exception = (
            not within_one_point and deviation <= largest_atomic_share + 1e-12
        )
        split_ratio_report[split] = {
            "target_share": SPLIT_RATIOS[split],
            "actual_share": round(actual_share, 8),
            "absolute_deviation_percentage_points": round(deviation * 100, 6),
            "within_one_percentage_point": within_one_point,
            "atomic_group_exception": atomic_exception,
        }
        if not within_one_point and not atomic_exception:
            ratio_failures[split] = deviation
    if ratio_failures:
        raise BuildError(f"final split ratio gate failed: {ratio_failures}")

    full_source_counts = Counter(row["source_type"] for row in expected_rows)
    full_source_shares = {
        source_type: full_source_counts[source_type] / total
        for source_type in ("original", "synthetic")
    }
    source_ratio_report: dict[str, Any] = {}
    source_ratio_failures: dict[str, Any] = {}
    for split in SPLITS:
        if not splits[split]:
            raise BuildError(f"{split}: empty final split")
        counts = Counter(row["source_type"] for row in splits[split])
        source_ratio_report[split] = {}
        for source_type in ("original", "synthetic"):
            actual_share = counts[source_type] / len(splits[split])
            deviation = abs(actual_share - full_source_shares[source_type])
            source_ratio_report[split][source_type] = {
                "full_pool_share": round(full_source_shares[source_type], 8),
                "split_share": round(actual_share, 8),
                "absolute_deviation_percentage_points": round(deviation * 100, 6),
                "within_two_percentage_points": deviation <= 0.02 + 1e-12,
            }
            if deviation > 0.02 + 1e-12:
                source_ratio_failures[f"{split}.{source_type}"] = deviation
    if source_ratio_failures:
        raise BuildError(f"source-type ratio gate failed: {source_ratio_failures}")

    return {
        "normalized_utterance_overlap": utterance_overlap,
        "synthetic_generation_family_overlap": family_overlap,
        "ordered_canonical_signature_overlap_diagnostic": signature_overlap,
        "final_union_identity": {
            "expected_examples": len(expected_rows),
            "observed_examples": len(final_rows),
            "unique_observed_example_ids": len(final_id_counts),
            "missing_example_ids": missing_ids,
            "extra_example_ids": extra_ids,
            "non_singleton_example_ids": duplicate_final_ids,
            "all_non_split_fields_preserved": True,
        },
        "split_ratio_gate": {
            "largest_atomic_group_size": largest_atomic_group_size,
            "largest_atomic_group_share": round(largest_atomic_share, 8),
            "splits": split_ratio_report,
        },
        "source_type_ratio_gate": source_ratio_report,
        "primitive_api_value_coverage": {
            "extraction": "function-qualified typed recursive API leaves plus absent optional arguments",
            "unique_train_primitives": len(primitive_sets["train"]),
            "unique_validation_primitives": len(primitive_sets["validation"]),
            "unique_test_primitives": len(primitive_sets["test"]),
            "validation_missing_from_train": primitive_missing["validation"],
            "test_missing_from_train": primitive_missing["test"],
        },
        "assertions": {
            "zero_normalized_utterance_overlap": True,
            "zero_synthetic_generation_family_overlap": True,
            "validation_test_primitives_subset_of_train": True,
            "final_union_identity_exactly_once": True,
            "source_type_ratio_within_two_percentage_points": True,
            "split_ratio_within_one_percentage_point_or_atomic_exception": True,
        },
    }


def _distribution(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    values: Iterable[Any],
) -> dict[str, int]:
    counts = Counter(row[key] for row in rows)
    return {str(value): counts.get(value, 0) for value in values}


def _function_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        call["function"] for row in rows for call in row["canonical_calls"]
    )
    return {name: counts.get(name, 0) for name in sorted(FUNCTION_FAMILIES)}


def _family_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        FUNCTION_FAMILIES[call["function"]]
        for row in rows
        for call in row["canonical_calls"]
    )
    return {family: counts.get(family, 0) for family in FAMILY_ORDER}


def _family_combinations(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        present = {
            FUNCTION_FAMILIES[call["function"]] for call in row["canonical_calls"]
        }
        key = "+".join(family for family in FAMILY_ORDER if family in present)
        counts[key] += 1
    return dict(sorted(counts.items()))


def _argument_coverage(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for call in row["canonical_calls"]:
            function_name = call["function"]
            arguments = call["arguments"]
            for argument_name in registry["functions"][function_name]["argument_order"]:
                base_path = f"{function_name}.{argument_name}"
                if argument_name not in arguments:
                    counts[base_path]["<ABSENT>"] += 1
                    continue
                for path, value in _flatten_leaves(arguments[argument_name], base_path):
                    counts[path][_typed_value(value)] += 1
    return {
        path: dict(sorted(value_counts.items()))
        for path, value_counts in sorted(counts.items())
    }


def _length_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = sorted(len(normalize_text(row["utterance_ko"])) for row in rows)
    if not lengths:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    p95_index = max(0, math.ceil(0.95 * len(lengths)) - 1)
    return {
        "count": len(lengths),
        "min": lengths[0],
        "mean": round(statistics.mean(lengths), 6),
        "median": statistics.median(lengths),
        "p95": lengths[p95_index],
        "max": lengths[-1],
    }


def select_review_samples(
    synthetic_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    sample_count: int = 32,
) -> list[dict[str, Any]]:
    """Select eight seeded examples per call count while maximizing coverage."""

    if sample_count % 4:
        raise BuildError("review sample count must be divisible by four")
    by_call_count: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(synthetic_rows, key=lambda item: item["example_id"]):
        by_call_count[int(row["call_count"])].append(row)

    required_features = {
        f"function:{call['function']}"
        for row in synthetic_rows
        for call in row["canonical_calls"]
    } | {
        "pattern:" + row["synthetic_generation"]["function_family_pattern"]
        for row in synthetic_rows
    }
    uncovered = set(required_features)
    selected: list[Mapping[str, Any]] = []
    quota = sample_count // 4

    def features(row: Mapping[str, Any]) -> set[str]:
        return {f"function:{call['function']}" for call in row["canonical_calls"]} | {
            "pattern:" + row["synthetic_generation"]["function_family_pattern"]
        }

    for call_count in range(1, 5):
        remaining = list(by_call_count[call_count])
        if len(remaining) < quota:
            raise BuildError(
                f"call_count={call_count}: fewer than {quota} review candidates"
            )
        for position in range(quota):
            chosen = min(
                remaining,
                key=lambda row: (
                    -len(features(row) & uncovered),
                    _stable_tie(
                        seed,
                        "review",
                        str(call_count),
                        str(position),
                        row["example_id"],
                    ),
                    row["example_id"],
                ),
            )
            selected.append(chosen)
            uncovered.difference_update(features(chosen))
            remaining.remove(chosen)

    if uncovered:
        raise BuildError(
            "32-row review sample cannot cover all functions/family patterns: "
            f"{sorted(uncovered)}"
        )
    selected.sort(key=lambda row: _stable_tie(seed, "review-order", row["example_id"]))
    samples = []
    for row in selected:
        generation = row["synthetic_generation"]
        samples.append(
            {
                "example_id": row["example_id"],
                "benchmark_split": row.get("benchmark_split"),
                "call_count": row["call_count"],
                "utterance_ko": row["utterance_ko"],
                "canonical_calls": row["canonical_calls"],
                "generation_family_id": generation["generation_family_id"],
                "template_id": generation["template_id"],
                "function_family_pattern": generation["function_family_pattern"],
            }
        )
    return samples


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(RESEARCH_ROOT.parent.resolve()))
    except ValueError:
        return str(path)


def build_report(
    *,
    original_pool: Sequence[Mapping[str, Any]],
    synthetic_valid: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    validation_report: Mapping[str, Any],
    target_report: Mapping[str, Any],
    synthetic_coverage_report: Mapping[str, Any],
    canonical_signature_report: Mapping[str, Any],
    split_diagnostics: Mapping[str, Any],
    integrity_report: Mapping[str, Any],
    registry: Mapping[str, Any],
    seed: int,
    input_paths: Mapping[str, Path],
    artifact_bytes: Mapping[str, bytes],
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    all_rows = [row for split in SPLITS for row in splits[split]]
    final_synthetic_rows = [
        row for row in all_rows if row["source_type"] == "synthetic"
    ]
    review_samples = select_review_samples(final_synthetic_rows, seed=seed)
    if len(review_samples) != 32:
        raise BuildError("review sample must contain exactly 32 examples")
    review_functions = sorted(
        {call["function"] for row in review_samples for call in row["canonical_calls"]}
    )
    review_patterns = sorted({row["function_family_pattern"] for row in review_samples})

    return {
        "dataset": BENCHMARK_VERSION,
        "seed": seed,
        "canonical_api": {
            "schema": _display_path(input_paths["schema"]),
            "registry": _display_path(input_paths["registry"]),
            "registry_version": registry["registry_version"],
        },
        "counts": {
            "total_examples": len(all_rows),
            "original": len(original_pool),
            "synthetic": len(synthetic_valid),
            "splits": {split: len(splits[split]) for split in SPLITS},
            "source_type_by_split": {
                split: _distribution(
                    splits[split], "source_type", ("original", "synthetic")
                )
                for split in SPLITS
            },
        },
        "call_count_distribution": {
            "overall": _distribution(all_rows, "call_count", range(1, 5)),
            "original": _distribution(original_pool, "call_count", range(1, 5)),
            "synthetic": _distribution(synthetic_valid, "call_count", range(1, 5)),
            "by_split": {
                split: _distribution(splits[split], "call_count", range(1, 5))
                for split in SPLITS
            },
        },
        "synthetic_targets": dict(target_report),
        "synthetic_release_coverage": dict(synthetic_coverage_report),
        "canonical_signature_policy": dict(canonical_signature_report),
        "function_counts": {
            "overall": _function_counts(all_rows),
            "by_split": {split: _function_counts(splits[split]) for split in SPLITS},
        },
        "family_counts": {
            "overall": _family_counts(all_rows),
            "by_split": {split: _family_counts(splits[split]) for split in SPLITS},
            "example_combinations": _family_combinations(all_rows),
        },
        "argument_coverage": _argument_coverage(all_rows, registry),
        "utterance_character_length": {
            "definition": "Unicode character count after v0.2 whitespace normalization",
            "overall": _length_statistics(all_rows),
            "by_source_type": {
                source_type: _length_statistics(
                    [row for row in all_rows if row["source_type"] == source_type]
                )
                for source_type in ("original", "synthetic")
            },
            "by_split": {split: _length_statistics(splits[split]) for split in SPLITS},
        },
        "deduplication_and_validation": dict(validation_report),
        "split": dict(split_diagnostics),
        "split_integrity": dict(integrity_report),
        "synthetic_generation": {
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "generation_family_count": len(
                {
                    row["synthetic_generation"]["generation_family_id"]
                    for row in synthetic_valid
                }
            ),
            "template_count": len(
                {row["synthetic_generation"]["template_id"] for row in synthetic_valid}
            ),
            "function_family_pattern_counts": dict(
                sorted(
                    Counter(
                        row["synthetic_generation"]["function_family_pattern"]
                        for row in synthetic_valid
                    ).items()
                )
            ),
        },
        "review_sample": {
            "selection": "32 deterministic seeded synthetic examples: eight per call_count, greedily covering functions and generation family patterns",
            "count": len(review_samples),
            "call_count_distribution": _distribution(
                review_samples, "call_count", range(1, 5)
            ),
            "covered_functions": review_functions,
            "covered_function_family_patterns": review_patterns,
            "examples": review_samples,
        },
        "hashes": {
            "algorithm": "sha256",
            "v0_2_inputs": {
                split: {
                    "path": _display_path(input_paths[f"original_{split}"]),
                    "sha256": _sha256_file(input_paths[f"original_{split}"]),
                    "expected_sha256": ORIGINAL_INPUT_SHA256[split],
                    "matches_frozen_input": True,
                }
                for split in SPLITS
            },
            "inputs": {
                name: {
                    "path": _display_path(path),
                    "sha256": _sha256_file(path),
                }
                for name, path in sorted(input_paths.items())
            },
            "artifacts": {
                name: {
                    "path": _display_path(artifact_paths[name]),
                    "sha256": _sha256_bytes(value),
                }
                for name, value in sorted(artifact_bytes.items())
            },
        },
    }


def build_from_paths(
    *,
    original_root: Path,
    candidates_path: Path,
    schema_path: Path,
    registry_path: Path,
    original_pool_path: Path,
    synthetic_valid_path: Path,
    output_root: Path,
    report_path: Path,
    seed: int,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    schema, registry = load_contract(schema_path, registry_path)
    original_paths = {split: original_root / f"{split}.jsonl" for split in SPLITS}
    frozen_hashes = {
        split: _sha256_file(path) for split, path in original_paths.items()
    }
    frozen_hash_mismatches = {
        split: {
            "expected": ORIGINAL_INPUT_SHA256[split],
            "actual": frozen_hashes[split],
        }
        for split in SPLITS
        if frozen_hashes[split] != ORIGINAL_INPUT_SHA256[split]
    }
    if frozen_hash_mismatches:
        raise BuildError(f"frozen v0.2 input hash mismatch: {frozen_hash_mismatches}")

    planned_outputs = {
        original_pool_path,
        synthetic_valid_path,
        report_path,
        *(output_root / f"{split}.jsonl" for split in SPLITS),
    }
    protected_inputs = {
        candidates_path,
        schema_path,
        registry_path,
        *original_paths.values(),
    }
    protected_resolved = {path.resolve() for path in protected_inputs}
    overlap = {
        output for output in planned_outputs if output.resolve() in protected_resolved
    }
    if overlap:
        raise BuildError(
            "output path collides with protected input: "
            + ", ".join(str(path) for path in sorted(overlap))
        )

    rows_by_split = {split: load_jsonl(original_paths[split]) for split in SPLITS}
    original_pool = prepare_original_source_pool(
        rows_by_split,
        schema=schema,
        registry=registry,
    )
    candidates = load_jsonl(candidates_path)
    synthetic_valid, validation_report = validate_and_deduplicate_synthetic(
        candidates,
        original_pool,
        schema=schema,
        registry=registry,
        seed=seed,
    )
    target_report = assert_synthetic_targets(synthetic_valid)
    synthetic_coverage_report = audit_synthetic_coverage(
        synthetic_valid,
        schema=schema,
    )
    canonical_signature_report = audit_canonical_signature_policy(
        original_pool,
        synthetic_valid,
    )

    combined = [*original_pool, *synthetic_valid]
    combined_ids = [row["example_id"] for row in combined]
    if len(combined_ids) != len(set(combined_ids)):
        raise BuildError("global example_id uniqueness invariant failed")

    splits, split_diagnostics = group_aware_split(
        combined,
        registry=registry,
        seed=seed,
    )
    integrity_report = audit_split_integrity(
        splits,
        registry=registry,
        expected_rows=combined,
    )

    artifact_paths = {
        "original_source_pool": original_pool_path,
        "synthetic_valid": synthetic_valid_path,
        **{split: output_root / f"{split}.jsonl" for split in SPLITS},
    }
    artifact_bytes = {
        "original_source_pool": _jsonl_bytes(original_pool),
        "synthetic_valid": _jsonl_bytes(synthetic_valid),
        **{split: _jsonl_bytes(splits[split]) for split in SPLITS},
    }
    input_paths = {
        **{f"original_{split}": path for split, path in original_paths.items()},
        "synthetic_candidates": candidates_path,
        "schema": schema_path,
        "registry": registry_path,
    }
    report = build_report(
        original_pool=original_pool,
        synthetic_valid=synthetic_valid,
        splits=splits,
        validation_report=validation_report,
        target_report=target_report,
        synthetic_coverage_report=synthetic_coverage_report,
        canonical_signature_report=canonical_signature_report,
        split_diagnostics=split_diagnostics,
        integrity_report=integrity_report,
        registry=registry,
        seed=seed,
        input_paths=input_paths,
        artifact_bytes=artifact_bytes,
        artifact_paths=artifact_paths,
    )

    outputs = {artifact_paths[name]: value for name, value in artifact_bytes.items()}
    outputs[report_path] = _json_bytes(report)
    return outputs, report


def write_artifacts_atomic(
    artifacts: Mapping[Path, bytes],
    *,
    force: bool,
) -> None:
    paths = list(artifacts)
    if len(paths) != len(set(path.resolve() for path in paths)):
        raise BuildError("output paths must be distinct")

    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise BuildError(
            "refusing to overwrite existing output(s) without --force: "
            + ", ".join(str(path) for path in existing)
        )

    staged: list[tuple[Path, Path]] = []
    try:
        for path, value in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary_path, path))

        for temporary_path, path in staged:
            os.replace(temporary_path, path)
    finally:
        for temporary_path, _ in staged:
            if temporary_path.exists():
                temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument(
        "--synthetic-candidates",
        "--candidates",
        dest="candidates_path",
        type=Path,
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--original-pool-output", type=Path, default=DEFAULT_ORIGINAL_POOL
    )
    parser.add_argument(
        "--synthetic-valid-output", type=Path, default=DEFAULT_SYNTHETIC_VALID
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="default: <output-root>/dataset_report.json",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report_output or args.output_root / "dataset_report.json"
    artifacts, report = build_from_paths(
        original_root=args.original_root,
        candidates_path=args.candidates_path,
        schema_path=args.schema,
        registry_path=args.registry,
        original_pool_path=args.original_pool_output,
        synthetic_valid_path=args.synthetic_valid_output,
        output_root=args.output_root,
        report_path=report_path,
        seed=args.seed,
    )
    write_artifacts_atomic(artifacts, force=args.force)

    counts = report["counts"]
    print(
        f"PASS: {BENCHMARK_VERSION} built with "
        f"{counts['total_examples']} examples "
        f"({counts['original']} original + {counts['synthetic']} synthetic)"
    )
    for split in SPLITS:
        print(f"{split}: {counts['splits'][split]}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
