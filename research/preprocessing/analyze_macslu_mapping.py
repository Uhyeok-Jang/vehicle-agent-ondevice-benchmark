#!/usr/bin/env python3
"""Measure deterministic MAC-SLU-to-canonical mapping coverage."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_macslu as audit
import build_macslu_inventory as inventory_builder
import canonical_vehicle_api as canonical
import map_macslu_vehicle as mapping


SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
UNIT_STATUSES = ("mapped", "ambiguous", "unsupported", "needs_context")
ROW_STATUSES = ("fully_mapped", "partially_mapped", "zero_mapped")
UNRESOLVED_FIELDS = (
    "slot_name",
    "slot_value",
    "total",
    "train",
    "validation",
    "test",
)
FAILURE_FIELDS = (
    "status",
    "reason_codes",
    "normalized_signature",
    "total",
    "train",
    "validation",
    "test",
)


def _split_key(name: str) -> tuple[int, str]:
    return SPLIT_ORDER.get(name, len(SPLIT_ORDER)), name


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _canonical_signature(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _blank_counter(names: Sequence[str]) -> Counter[str]:
    return Counter({name: 0 for name in names})


def _row_status(mapped_units: int, total_units: int) -> str:
    if total_units <= 0:
        raise ValueError("vehicle rows must contain at least one semantic unit")
    if mapped_units == total_units:
        return "fully_mapped"
    if mapped_units:
        return "partially_mapped"
    return "zero_mapped"


def analyze_mapping(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    revision: str,
    mapper: mapping.MacsluVehicleMapper,
    initial_status_by_example_id: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return aggregate coverage plus unresolved-value and failure ledgers."""

    if not revision:
        raise ValueError("a non-empty dataset revision is required")

    source_rows = 0
    vehicle_rows = 0
    vehicle_units = 0
    unit_statuses = _blank_counter(UNIT_STATUSES)
    row_statuses = _blank_counter(ROW_STATUSES)
    reason_counts: Counter[str] = Counter()
    mapped_functions: Counter[str] = Counter()
    call_counts: Counter[int] = Counter()
    unresolved: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    failures: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    initial_matrix: dict[str, Counter[str]] = defaultdict(
        lambda: _blank_counter(ROW_STATUSES)
    )
    split_totals: dict[str, dict[str, Any]] = {}

    for split in sorted(splits, key=_split_key):
        split_source_rows = 0
        split_vehicle_rows = 0
        split_vehicle_units = 0
        split_unit_statuses = _blank_counter(UNIT_STATUSES)
        split_row_statuses = _blank_counter(ROW_STATUSES)
        split_functions: Counter[str] = Counter()

        for row in splits[split]:
            source_rows += 1
            split_source_rows += 1
            frames = mapping.adapt_macslu_row(
                row,
                revision=revision,
                split=split,
                vehicle_domain=mapper.registry["vehicle_domain"],
            )
            if not frames:
                continue

            result = mapper.map_row(row, revision=revision, split=split)
            units = result["units"]
            if len(frames) != len(units):
                raise RuntimeError("adapter and mapper returned different unit counts")

            initial_status = initial_status_by_example_id.get(result["example_id"])
            if initial_status is None:
                raise RuntimeError(
                    f"missing inventory disposition for {result['example_id']}"
                )

            vehicle_rows += 1
            split_vehicle_rows += 1
            vehicle_units += len(units)
            split_vehicle_units += len(units)
            mapped_count = 0

            for frame, unit in zip(frames, units, strict=True):
                decision = unit["decision"]
                status = str(decision["status"])
                if status not in UNIT_STATUSES:
                    raise RuntimeError(f"unknown mapping status: {status}")
                unit_statuses[status] += 1
                split_unit_statuses[status] += 1
                reason_codes = tuple(str(code) for code in decision["reason_codes"])
                reason_counts.update(reason_codes)

                if status == "mapped":
                    mapped_count += 1
                    function = str(decision["call"]["function"])
                    mapped_functions[function] += 1
                    split_functions[function] += 1
                    continue

                failures[
                    (
                        status,
                        "|".join(reason_codes),
                        _canonical_signature(unit["normalized"]),
                    )
                ][split] += 1

                if "unrecognized_source_value" not in reason_codes:
                    continue
                traced_ordinals = {
                    ordinal
                    for trace in decision["trace"]
                    for ordinal in trace["source_slot_ordinals"]
                }
                for slot in frame.slots:
                    if slot.ordinal not in traced_ordinals:
                        unresolved[(slot.name, slot.value)][split] += 1

            row_status = _row_status(mapped_count, len(units))
            row_statuses[row_status] += 1
            split_row_statuses[row_status] += 1
            initial_matrix[initial_status][row_status] += 1

            has_payload = result["canonical_payload"] is not None
            if has_payload != (row_status == "fully_mapped"):
                raise RuntimeError("partial or failed row emitted a canonical payload")
            if has_payload:
                call_counts[len(result["canonical_payload"]["calls"])] += 1

        split_totals[split] = {
            "source_rows": split_source_rows,
            "vehicle_rows": split_vehicle_rows,
            "vehicle_units": split_vehicle_units,
            "unit_outcomes": {
                "counts": dict(split_unit_statuses),
                "mapped_ratio": _ratio(
                    split_unit_statuses["mapped"], split_vehicle_units
                ),
            },
            "row_outcomes": {
                "counts": dict(split_row_statuses),
                "fully_mapped_ratio": _ratio(
                    split_row_statuses["fully_mapped"], split_vehicle_rows
                ),
            },
            "mapped_functions": dict(sorted(split_functions.items())),
        }

    unresolved_rows = []
    for (slot_name, slot_value), split_counts in unresolved.items():
        unresolved_rows.append(
            {
                "slot_name": slot_name,
                "slot_value": slot_value,
                "total": sum(split_counts.values()),
                **{split: split_counts[split] for split in SPLIT_ORDER},
            }
        )
    unresolved_rows.sort(
        key=lambda row: (-int(row["total"]), str(row["slot_name"]), str(row["slot_value"]))
    )

    failure_rows = []
    for (status, reason_codes, signature), split_counts in failures.items():
        failure_rows.append(
            {
                "status": status,
                "reason_codes": reason_codes,
                "normalized_signature": signature,
                "total": sum(split_counts.values()),
                **{split: split_counts[split] for split in SPLIT_ORDER},
            }
        )
    failure_rows.sort(
        key=lambda row: (
            -int(row["total"]),
            str(row["status"]),
            str(row["reason_codes"]),
            str(row["normalized_signature"]),
        )
    )

    summary = {
        "schema_version": 1,
        "population": "all verified MAC-SLU vehicle semantic units",
        "mapping_registry_version": mapper.registry["registry_version"],
        "counts": {
            "source_rows": source_rows,
            "vehicle_rows": vehicle_rows,
            "vehicle_units": vehicle_units,
        },
        "unit_outcomes": {
            "counts": dict(unit_statuses),
            "mapped_ratio": _ratio(unit_statuses["mapped"], vehicle_units),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "row_outcomes": {
            "counts": dict(row_statuses),
            "fully_mapped_ratio": _ratio(
                row_statuses["fully_mapped"], vehicle_rows
            ),
            "fully_mapped_call_count_distribution": {
                str(count): rows for count, rows in sorted(call_counts.items())
            },
        },
        "mapped_functions": dict(sorted(mapped_functions.items())),
        "mapping_by_initial_status": {
            status: {
                "vehicle_rows": sum(counts.values()),
                "row_outcomes": dict(counts),
                "fully_mapped_ratio": _ratio(
                    counts["fully_mapped"], sum(counts.values())
                ),
            }
            for status, counts in sorted(initial_matrix.items())
        },
        "splits": split_totals,
        "final_eligibility": {
            "status": "not_adjudicated",
            "eligible_rows": None,
            "note": (
                "mapped is a transformation outcome, not release eligibility; "
                "translation review and final adjudication are still required"
            ),
        },
    }
    return summary, unresolved_rows, failure_rows


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    unresolved_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        unresolved_path = temporary / "unresolved_values.csv"
        with unresolved_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=UNRESOLVED_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(unresolved_rows)

        failures_path = temporary / "failure_signatures.csv"
        with failures_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FAILURE_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(failure_rows)

        output_summary = dict(summary)
        output_summary["artifacts"] = {
            "failure_signatures": {
                "name": failures_path.name,
                "rows": len(failure_rows),
                "sha256": audit.sha256_file(failures_path),
            },
            "unresolved_values": {
                "name": unresolved_path.name,
                "rows": len(unresolved_rows),
                "sha256": audit.sha256_file(unresolved_path),
            },
        }
        (temporary / "summary.json").write_text(
            json.dumps(output_summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mapping-registry",
        type=Path,
        default=mapping.DEFAULT_MAPPING_REGISTRY,
    )
    parser.add_argument(
        "--mapping-schema",
        type=Path,
        default=mapping.DEFAULT_MAPPING_SCHEMA,
    )
    parser.add_argument(
        "--canonical-schema",
        type=Path,
        default=canonical.DEFAULT_SCHEMA,
    )
    parser.add_argument(
        "--canonical-registry",
        type=Path,
        default=canonical.DEFAULT_REGISTRY,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits, source_provenance, manifest = audit.load_local_jsonl(
        args.manifest, args.source_root
    )
    verification, _ = audit.verify_sources(source_provenance["files"])
    audit.require_verified_source(verification)

    dataset = manifest["dataset"]
    revision = str(dataset["revision"])
    claims = manifest.get("paper_claims", {})
    inventory_rows, _ = inventory_builder.build_inventory(
        splits,
        revision=revision,
        vehicle_domain=dataset.get("vehicle_domain", audit.DEFAULT_VEHICLE_DOMAIN),
        max_intents=claims.get(
            "maximum_intents_per_utterance", audit.DEFAULT_MAX_INTENTS
        ),
        allowed_vehicle_slots=manifest.get(
            "known_vehicle_slot_names", tuple(audit.DEFAULT_VEHICLE_SLOTS)
        ),
    )
    initial_status_by_example_id = {
        str(row["example_id"]): str(row["initial_status"]) for row in inventory_rows
    }

    mapping_registry = canonical.load_json_object(args.mapping_registry)
    mapping_schema = canonical.load_json_object(args.mapping_schema)
    canonical_schema = canonical.load_json_object(args.canonical_schema)
    canonical_registry = canonical.load_json_object(args.canonical_registry)
    mapper = mapping.MacsluVehicleMapper(
        registry=mapping_registry,
        mapping_schema=mapping_schema,
        canonical_schema=canonical_schema,
        canonical_registry=canonical_registry,
    )
    summary, unresolved_rows, failure_rows = analyze_mapping(
        splits,
        revision=revision,
        mapper=mapper,
        initial_status_by_example_id=initial_status_by_example_id,
    )
    summary["provenance"] = {
        "dataset": dataset["id"],
        "revision": revision,
        "source": source_provenance,
        "source_verification": verification,
        "generator": {
            "name": Path(__file__).name,
            "sha256": audit.sha256_file(Path(__file__)),
            "dependencies": {
                "audit_macslu.py": audit.sha256_file(Path(audit.__file__)),
                "build_macslu_inventory.py": audit.sha256_file(
                    Path(inventory_builder.__file__)
                ),
                "map_macslu_vehicle.py": audit.sha256_file(Path(mapping.__file__)),
            },
        },
        "mapping_registry": {
            "name": args.mapping_registry.name,
            "sha256": audit.sha256_file(args.mapping_registry),
        },
        "mapping_schema": {
            "name": args.mapping_schema.name,
            "sha256": audit.sha256_file(args.mapping_schema),
        },
        "canonical_schema": {
            "name": args.canonical_schema.name,
            "sha256": audit.sha256_file(args.canonical_schema),
        },
        "canonical_registry": {
            "name": args.canonical_registry.name,
            "sha256": audit.sha256_file(args.canonical_registry),
        },
    }
    write_outputs(args.output_dir, summary, unresolved_rows, failure_rows)
    print(
        f"Wrote mapping coverage for {summary['counts']['vehicle_units']} units "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
