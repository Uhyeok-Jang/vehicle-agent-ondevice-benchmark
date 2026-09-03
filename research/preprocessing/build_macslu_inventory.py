#!/usr/bin/env python3
"""Build a deterministic review inventory from pinned MAC-SLU JSONL files."""

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


INVENTORY_FIELDS = (
    "example_id",
    "source_group_id",
    "split",
    "id",
    "query",
    "active_unit_count",
    "vehicle_unit_count",
    "has_vehicle_target",
    "domains",
    "issue_codes",
    "initial_status",
    "final_status",
)
OVERLAP_CODES = {
    "cross_split_query_overlap_exact",
    "cross_split_query_overlap_normalized",
}
REVIEW_OVERLAP_CODE = "cross_split_query_overlap_review_normalized"
MANUAL_REVIEW_CODES = {
    "split_sens_gt_semantic_units",
    "split_sens_lt_semantic_units",
    "mixed_domain_example",
    "unexpected_vehicle_slot",
    "max_intent_claim_violation",
    "multi_domain_per_intent",
    REVIEW_OVERLAP_CODE,
}
STATUS_NAMES = ("candidate", "manual_review", "quarantined", "excluded")
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def _split_key(name: str) -> tuple[int, str]:
    return SPLIT_ORDER.get(name, len(SPLIT_ORDER)), name


def _sequence_length(value: Any) -> int:
    return (
        len(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else 0
    )


def _overlap_flags(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, int], set[str]]:
    exact = defaultdict(list)
    normalized = defaultdict(list)
    review_normalized = defaultdict(list)
    for split_name, examples in splits.items():
        for index, example in enumerate(examples):
            query = str(example.get("query") or "")
            if not query:
                continue
            exact[query].append((split_name, index))
            normalized_query = audit.normalize_query(query)
            if normalized_query:
                normalized[normalized_query].append((split_name, index))
            review_key = audit.normalize_query_for_review(query)
            if review_key:
                review_normalized[review_key].append((split_name, index))

    flags: dict[tuple[str, int], set[str]] = defaultdict(set)
    for groups, code in (
        (exact, "cross_split_query_overlap_exact"),
        (normalized, "cross_split_query_overlap_normalized"),
        (review_normalized, REVIEW_OVERLAP_CODE),
    ):
        for locations in groups.values():
            participating_splits = {split for split, _ in locations}
            if len(participating_splits) < 2:
                continue
            earliest_split = min(participating_splits, key=_split_key)
            for split_name, index in locations:
                if split_name != earliest_split:
                    flags[(split_name, index)].add(code)
    return flags


def _base_issue_codes(
    example: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    vehicle_domain: str,
    max_intents: int,
    allowed_slots: frozenset[str],
) -> set[str]:
    codes = set()
    split_sens_count = _sequence_length(example.get("split_sens"))
    if split_sens_count > len(units):
        codes.add("split_sens_gt_semantic_units")
    elif split_sens_count < len(units):
        codes.add("split_sens_lt_semantic_units")

    intent_domains: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        intent_domains[str(unit["intent_index"])].add(str(unit["domain"]))
    if any(len(domains) > 1 for domains in intent_domains.values()):
        codes.add("multi_domain_per_intent")
    if len(intent_domains) > max_intents:
        codes.add("max_intent_claim_violation")

    domains = {str(unit["domain"]) for unit in units}
    if vehicle_domain in domains and len(domains) > 1:
        codes.add("mixed_domain_example")

    for unit in units:
        if unit["domain"] != vehicle_domain:
            continue
        for slot in unit["slots"]:
            name = slot.get("name") if isinstance(slot, Mapping) else None
            if name is not None and str(name) not in allowed_slots:
                codes.add("unexpected_vehicle_slot")
    return codes


def initial_status(
    split_name: str, issue_codes: set[str], *, has_vehicle_target: bool = True
) -> str:
    if not has_vehicle_target:
        return "excluded"
    if split_name != "train" and issue_codes & OVERLAP_CODES:
        return "quarantined"
    if issue_codes & MANUAL_REVIEW_CODES:
        return "manual_review"
    return "candidate"


def build_inventory(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    revision: str,
    vehicle_domain: str = audit.DEFAULT_VEHICLE_DOMAIN,
    max_intents: int = audit.DEFAULT_MAX_INTENTS,
    allowed_vehicle_slots: Sequence[str] = tuple(audit.DEFAULT_VEHICLE_SLOTS),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not revision:
        raise ValueError("a non-empty dataset revision is required")

    overlap_flags = _overlap_flags(splits)
    allowed_slots = frozenset(allowed_vehicle_slots)
    inventory = []
    status_counts = Counter({name: 0 for name in STATUS_NAMES})
    issue_counts = Counter()
    split_summaries = {}
    excluded_unannotated = 0
    excluded_no_vehicle = 0
    source_rows = 0

    for split_name in sorted(splits, key=_split_key):
        split_counts = Counter({name: 0 for name in STATUS_NAMES})
        for index, example in enumerate(splits[split_name]):
            source_rows += 1
            split_counts["source_rows"] += 1
            units = audit.extract_active_semantic_units(example)
            vehicle_units = [
                unit for unit in units if unit["domain"] == vehicle_domain
            ]
            if not units:
                excluded_unannotated += 1
                split_counts["excluded_unannotated_rows"] += 1
            if not vehicle_units:
                excluded_no_vehicle += 1
                split_counts["excluded_no_vehicle_rows"] += 1

            sample_id = str(example.get("id") or "")
            domains = sorted({str(unit["domain"]) for unit in units})
            codes = _base_issue_codes(
                example,
                units,
                vehicle_domain=vehicle_domain,
                max_intents=max_intents,
                allowed_slots=allowed_slots,
            )
            codes.update(overlap_flags.get((split_name, index), set()))
            if not vehicle_units:
                codes.add("no_vehicle_target")
            if not units:
                codes.add("unannotated_source")
            status = initial_status(
                split_name,
                codes,
                has_vehicle_target=bool(vehicle_units),
            )
            status_counts[status] += 1
            split_counts[status] += 1
            issue_counts.update(codes)
            inventory.append(
                {
                    "example_id": f"macslu:{revision}:{split_name}:{sample_id}",
                    "source_group_id": f"macslu:{revision}:{split_name}:{sample_id}",
                    "split": split_name,
                    "id": sample_id,
                    "query": str(example.get("query") or ""),
                    "active_unit_count": len(units),
                    "vehicle_unit_count": len(vehicle_units),
                    "has_vehicle_target": bool(vehicle_units),
                    "domains": domains,
                    "issue_codes": sorted(codes),
                    "initial_status": status,
                    "final_status": "",
                }
            )

        split_counts["inventory_rows"] = sum(split_counts[name] for name in STATUS_NAMES)
        split_counts["vehicle_rows"] = (
            split_counts["candidate"]
            + split_counts["manual_review"]
            + split_counts["quarantined"]
        )
        split_counts["excluded_annotated_non_vehicle_rows"] = (
            split_counts["excluded_no_vehicle_rows"]
            - split_counts["excluded_unannotated_rows"]
        )
        split_summaries[split_name] = dict(sorted(split_counts.items()))

    summary = {
        "schema_version": 2,
        "dataset_revision": revision,
        "vehicle_domain": vehicle_domain,
        "counts": {
            "source_rows": source_rows,
            "inventory_rows": len(inventory),
            "vehicle_rows": source_rows - excluded_no_vehicle,
            "excluded_no_vehicle_rows": excluded_no_vehicle,
            "excluded_unannotated_rows": excluded_unannotated,
            "excluded_annotated_non_vehicle_rows": (
                excluded_no_vehicle - excluded_unannotated
            ),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "issue_code_counts": dict(sorted(issue_counts.items())),
        "splits": split_summaries,
        "status_policy": {
            "priority": [
                "excluded",
                "quarantined",
                "manual_review",
                "candidate"
            ],
            "excluded": [
                "no_vehicle_target"
            ],
            "quarantined": sorted(OVERLAP_CODES),
            "manual_review": sorted(MANUAL_REVIEW_CODES),
            "candidate": "no quarantine or manual-review issue",
            "final_status": "blank until mapping and human review are complete",
        },
    }
    return inventory, summary


def write_outputs(
    output_dir: Path,
    inventory: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        inventory_path = temporary / "inventory.csv"
        with inventory_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for row in inventory:
                serialized = dict(row)
                serialized["domains"] = "|".join(row["domains"])
                serialized["issue_codes"] = "|".join(row["issue_codes"])
                writer.writerow(serialized)

        output_summary = dict(summary)
        output_summary["artifacts"] = {
            "inventory": {
                "name": inventory_path.name,
                "rows": len(inventory),
                "sha256": audit.sha256_file(inventory_path),
            }
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
        "--allow-unverified-source",
        action="store_true",
        help="continue after source verification failure (diagnostics only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits, provenance, manifest = audit.load_local_jsonl(
        args.manifest, args.source_root
    )
    dataset = manifest.get("dataset", {})
    claims = manifest.get("paper_claims", {})
    revision = dataset.get("revision")
    source_verification, _ = audit.verify_sources(provenance["files"])
    audit.require_verified_source(
        source_verification,
        allow_unverified=args.allow_unverified_source,
    )
    inventory, summary = build_inventory(
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
    summary["provenance"] = {
        **provenance,
        "dataset": dataset.get("id"),
        "generator": {
            "dependencies": {
                "audit_macslu.py": audit.sha256_file(Path(audit.__file__)),
            },
            "name": Path(__file__).name,
            "sha256": audit.sha256_file(Path(__file__)),
        },
        "normalization_version": audit.NORMALIZATION_VERSION,
        "revision": revision,
    }
    summary["source_verification"] = source_verification
    write_outputs(args.output_dir, inventory, summary)
    print(f"Wrote {len(inventory)} inventory rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
