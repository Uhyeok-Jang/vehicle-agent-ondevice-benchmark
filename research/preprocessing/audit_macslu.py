#!/usr/bin/env python3
"""Deterministic, source-preserving audit for MAC-SLU style datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DATASET = "Gatsby1984/MAC_SLU"
DEFAULT_VEHICLE_DOMAIN = "车载控制"
DEFAULT_MAX_INTENTS = 5
NORMALIZATION_VERSION = "2"
DEFAULT_VEHICLE_SLOTS = frozenset(
    {
        "intent",
        "value",
        "位置",
        "功能",
        "子功能",
        "对象",
        "对象功能",
        "操作",
        "操作_concrete",
        "方向偏移量",
        "模式",
        "调节内容",
        "车内灯类型",
        "车外灯类型",
        "车机模块",
        "摄像头模式",
        "座椅记忆位置",
        "序列号",
        "身体位置",
        "页面",
        "音效",
    }
)
ISSUE_FIELDS = (
    "code",
    "split",
    "id",
    "intent_index",
    "domain",
    "query",
    "detail",
)
SPLIT_COUNT_FIELDS = (
    "examples",
    "active_semantic_units",
    "active_intents",
    "vehicle_examples",
    "vehicle_semantic_units",
    "unannotated_examples",
    "count_mismatches",
    "split_sens_gt_semantic_units",
    "split_sens_lt_semantic_units",
    "vehicle_count_mismatches",
    "vehicle_split_sens_gt_semantic_units",
    "vehicle_split_sens_lt_semantic_units",
    "multi_domain_intents",
    "mixed_domain_examples",
    "max_intent_claim_violations",
    "unexpected_slot_occurrences",
    "missing_ids",
    "duplicate_id_records",
    "duplicate_id_values",
    "unique_ids",
    "max_active_intents",
    "max_active_semantic_units",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceVerificationError(RuntimeError):
    """Raised when a pinned source cannot be verified against its manifest."""


def normalize_query(query: Any) -> str:
    """Normalize width, case, and whitespace without erasing semantics."""
    text = unicodedata.normalize("NFKC", str(query or "")).lower()
    return "".join(char for char in text if not char.isspace())


def normalize_query_for_review(query: Any) -> str:
    """Aggressive review key that preserves decimal points and numeric signs."""
    text = normalize_query(query)
    kept = []
    for index, char in enumerate(text):
        if not unicodedata.category(char).startswith("P"):
            kept.append(char)
            continue
        previous_is_digit = index > 0 and text[index - 1].isdigit()
        next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
        if char == "." and previous_is_digit and next_is_digit:
            kept.append(char)
        elif char in {"+", "-"} and (previous_is_digit or next_is_digit):
            kept.append(char)
    return "".join(kept)


def extract_active_semantic_units(example: Mapping[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    semantics = example.get("semantics")
    if not isinstance(semantics, Mapping):
        return units

    for intent_index, intent_data in semantics.items():
        if not isinstance(intent_data, Mapping):
            continue
        for domain, slots in intent_data.items():
            if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
                continue
            if slots:
                units.append(
                    {
                        "intent_index": str(intent_index),
                        "domain": str(domain),
                        "slots": list(slots),
                    }
                )
    return units


def _issue(
    code: str,
    *,
    split: str = "",
    sample_id: str = "",
    intent_index: str = "",
    domain: str = "",
    query: str = "",
    detail: Any = "",
) -> dict[str, str]:
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False, sort_keys=True)
    return {
        "code": code,
        "split": split,
        "id": sample_id,
        "intent_index": intent_index,
        "domain": domain,
        "query": query,
        "detail": detail,
    }


def _duplicate_id_issues(
    records: Mapping[str, Sequence[tuple[str, str]]], code: str
) -> list[dict[str, str]]:
    issues = []
    for sample_id, locations in records.items():
        if sample_id and len(locations) > 1:
            issues.append(
                _issue(
                    code,
                    sample_id=sample_id,
                    split="|".join(sorted({split for split, _ in locations})),
                    detail={"locations": sorted(f"{s}:{q}" for s, q in locations)},
                )
            )
    return issues


def _overlap_issues(
    records: Mapping[str, Sequence[tuple[str, str, str]]], code: str
) -> list[dict[str, str]]:
    issues = []
    for key, locations in records.items():
        splits = sorted({split for split, _, _ in locations})
        if len(splits) < 2:
            continue
        variants = sorted({query for _, _, query in locations})
        issues.append(
            _issue(
                code,
                split="|".join(splits),
                sample_id="|".join(sorted(f"{s}:{i}" for s, i, _ in locations)),
                query=variants[0],
                detail={"key": key, "variants": variants},
            )
        )
    return issues


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def composition_overlap(
    fragments_by_split: Mapping[str, Sequence[Sequence[str]]],
    *,
    train_split: str = "train",
) -> dict[str, Any]:
    """Measure normalized atomic-fragment reuse; this is not a leakage verdict."""
    train_fragments = {
        fragment
        for sample in fragments_by_split.get(train_split, ())
        for fragment in sample
        if fragment
    }
    evaluation: dict[str, dict[str, Any]] = {}
    aggregate = Counter()
    for split_name in sorted(name for name in fragments_by_split if name != train_split):
        samples = [
            [fragment for fragment in sample if fragment]
            for sample in fragments_by_split[split_name]
        ]
        samples = [sample for sample in samples if sample]
        occurrences = sum(len(sample) for sample in samples)
        matched = sum(
            fragment in train_fragments for sample in samples for fragment in sample
        )
        all_matched = sum(
            all(fragment in train_fragments for fragment in sample) for sample in samples
        )
        counts = {
            "eval_fragment_occurrences": occurrences,
            "fragment_occurrences_in_train": matched,
            "fragment_occurrence_overlap_ratio": _ratio(matched, occurrences),
            "eval_samples_with_nonempty_fragments": len(samples),
            "samples_all_nonempty_fragments_in_train": all_matched,
            "sample_all_fragments_overlap_ratio": _ratio(all_matched, len(samples)),
        }
        evaluation[split_name] = counts
        aggregate.update(
            {key: value for key, value in counts.items() if isinstance(value, int)}
        )

    occurrences = aggregate["eval_fragment_occurrences"]
    matched = aggregate["fragment_occurrences_in_train"]
    sample_count = aggregate["eval_samples_with_nonempty_fragments"]
    all_matched = aggregate["samples_all_nonempty_fragments_in_train"]
    evaluation["all"] = {
        "eval_fragment_occurrences": occurrences,
        "fragment_occurrences_in_train": matched,
        "fragment_occurrence_overlap_ratio": _ratio(matched, occurrences),
        "eval_samples_with_nonempty_fragments": sample_count,
        "samples_all_nonempty_fragments_in_train": all_matched,
        "sample_all_fragments_overlap_ratio": _ratio(all_matched, sample_count),
    }
    return {
        "diagnostic": "composition_overlap",
        "normalization": (
            "review key: NFKC + lowercase + remove whitespace and non-numeric "
            "punctuation"
        ),
        "train_split": train_split,
        "train_unique_nonempty_fragments": len(train_fragments),
        "evaluation": evaluation,
    }


def audit_dataset(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    vehicle_domain: str = DEFAULT_VEHICLE_DOMAIN,
    max_intents: int = DEFAULT_MAX_INTENTS,
    allowed_vehicle_slots: Iterable[str] = DEFAULT_VEHICLE_SLOTS,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    allowed_slots = frozenset(allowed_vehicle_slots)
    issues: list[dict[str, str]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    global_ids: dict[str, list[tuple[str, str]]] = defaultdict(list)
    exact_queries: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    normalized_queries: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    review_normalized_queries: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    fragments_by_split: dict[str, list[list[str]]] = defaultdict(list)
    clean_fragments_by_split: dict[str, list[list[str]]] = defaultdict(list)

    for split_name in sorted(splits):
        stats: Counter[str] = Counter()
        ids: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for example in splits[split_name]:
            stats["examples"] += 1
            sample_id = str(example.get("id") or "")
            query = str(example.get("query") or "")
            units = extract_active_semantic_units(example)
            intent_domains: dict[str, set[str]] = defaultdict(set)
            for unit in units:
                intent_domains[unit["intent_index"]].add(unit["domain"])

            active_intents = len(intent_domains)
            vehicle_units = [unit for unit in units if unit["domain"] == vehicle_domain]
            split_sens = example.get("split_sens")
            split_sens_count = (
                len(split_sens)
                if isinstance(split_sens, Sequence)
                and not isinstance(split_sens, (str, bytes))
                else 0
            )
            domains = sorted({unit["domain"] for unit in units})
            normalized_fragments = (
                [
                    normalized
                    for fragment in split_sens or []
                    if (normalized := normalize_query_for_review(fragment))
                ]
                if isinstance(split_sens, Sequence)
                and not isinstance(split_sens, (str, bytes))
                else []
            )
            if vehicle_units:
                fragments_by_split[split_name].append(normalized_fragments)
            if (
                vehicle_units
                and domains == [vehicle_domain]
                and split_sens_count == len(units)
            ):
                clean_fragments_by_split[split_name].append(normalized_fragments)

            stats["active_semantic_units"] += len(units)
            stats["active_intents"] += active_intents
            stats["vehicle_semantic_units"] += len(vehicle_units)
            stats["vehicle_examples"] += bool(vehicle_units)
            stats["unannotated_examples"] += not units
            stats["max_active_semantic_units"] = max(
                stats["max_active_semantic_units"], len(units)
            )
            stats["max_active_intents"] = max(stats["max_active_intents"], active_intents)

            if sample_id:
                ids[sample_id].append((split_name, query))
                global_ids[sample_id].append((split_name, query))
            else:
                stats["missing_ids"] += 1
                issues.append(
                    _issue("missing_id", split=split_name, query=query)
                )

            if split_sens_count != len(units):
                relation = (
                    "split_sens_gt_semantic_units"
                    if split_sens_count > len(units)
                    else "split_sens_lt_semantic_units"
                )
                stats["count_mismatches"] += 1
                stats[relation] += 1
                if vehicle_units:
                    stats["vehicle_count_mismatches"] += 1
                    stats[f"vehicle_{relation}"] += 1
                issues.append(
                    _issue(
                        relation,
                        split=split_name,
                        sample_id=sample_id,
                        query=query,
                        detail={
                            "active_semantic_units": len(units),
                            "split_sens": split_sens_count,
                        },
                    )
                )

            if vehicle_units and len(domains) > 1:
                stats["mixed_domain_examples"] += 1
                issues.append(
                    _issue(
                        "mixed_domain_example",
                        split=split_name,
                        sample_id=sample_id,
                        query=query,
                        detail={"domains": domains},
                    )
                )

            for intent_index, domains in sorted(intent_domains.items()):
                if len(domains) > 1:
                    stats["multi_domain_intents"] += 1
                    issues.append(
                        _issue(
                            "multi_domain_per_intent",
                            split=split_name,
                            sample_id=sample_id,
                            intent_index=intent_index,
                            query=query,
                            detail={"domains": sorted(domains)},
                        )
                    )

            if active_intents > max_intents:
                stats["max_intent_claim_violations"] += 1
                issues.append(
                    _issue(
                        "max_intent_claim_violation",
                        split=split_name,
                        sample_id=sample_id,
                        query=query,
                        detail={"active_intents": active_intents, "claim": max_intents},
                    )
                )

            for unit in vehicle_units:
                for slot in unit["slots"]:
                    name = slot.get("name") if isinstance(slot, Mapping) else None
                    if name is not None and str(name) not in allowed_slots:
                        stats["unexpected_slot_occurrences"] += 1
                        issues.append(
                            _issue(
                                "unexpected_vehicle_slot",
                                split=split_name,
                                sample_id=sample_id,
                                intent_index=unit["intent_index"],
                                domain=vehicle_domain,
                                query=query,
                                detail={"slot": str(name)},
                            )
                        )

            if query:
                exact_queries[query].append((split_name, sample_id, query))
                normalized = normalize_query(query)
                if normalized:
                    normalized_queries[normalized].append(
                        (split_name, sample_id, query)
                    )
                review_normalized = normalize_query_for_review(query)
                if review_normalized:
                    review_normalized_queries[review_normalized].append(
                        (split_name, sample_id, query)
                    )

        duplicate_records = sum(len(v) - 1 for v in ids.values() if len(v) > 1)
        duplicate_values = sum(len(v) > 1 for v in ids.values())
        stats["unique_ids"] = len(ids)
        stats["duplicate_id_records"] = duplicate_records
        stats["duplicate_id_values"] = duplicate_values
        issues.extend(_duplicate_id_issues(ids, "duplicate_id_within_split"))
        split_summary = {field: int(stats[field]) for field in SPLIT_COUNT_FIELDS}
        split_summary["ids_unique"] = not duplicate_values and not stats["missing_ids"]
        split_summaries[split_name] = split_summary

    issues.extend(_duplicate_id_issues(global_ids, "duplicate_id_globally"))
    exact_issues = _overlap_issues(exact_queries, "cross_split_query_overlap_exact")
    normalized_issues = _overlap_issues(
        normalized_queries, "cross_split_query_overlap_normalized"
    )
    review_normalized_issues = _overlap_issues(
        review_normalized_queries,
        "cross_split_query_overlap_review_normalized",
    )
    issues.extend(exact_issues)
    issues.extend(normalized_issues)
    issues.extend(review_normalized_issues)
    issues.sort(key=lambda row: tuple(row[field] for field in ISSUE_FIELDS))

    additive_keys = {
        key
        for stats in split_summaries.values()
        for key in stats
        if key
        not in {
            "max_active_intents",
            "max_active_semantic_units",
            "unique_ids",
            "ids_unique",
        }
    }
    totals = {
        key: sum(int(stats.get(key, 0)) for stats in split_summaries.values())
        for key in sorted(additive_keys)
    }
    totals.update(
        {
            "unique_ids": len(global_ids),
            "duplicate_id_values_globally": sum(
                len(v) > 1 for v in global_ids.values()
            ),
            "duplicate_id_records_globally": sum(
                len(v) - 1 for v in global_ids.values() if len(v) > 1
            ),
            "ids_unique_globally": not any(
                len(v) > 1 for v in global_ids.values()
            )
            and not any(stats["missing_ids"] for stats in split_summaries.values()),
            "max_active_intents": max(
                (stats.get("max_active_intents", 0) for stats in split_summaries.values()),
                default=0,
            ),
            "max_active_semantic_units": max(
                (
                    stats.get("max_active_semantic_units", 0)
                    for stats in split_summaries.values()
                ),
                default=0,
            ),
            "cross_split_query_overlap_exact_groups": len(exact_issues),
            "cross_split_query_overlap_normalized_groups": len(normalized_issues),
            "cross_split_query_overlap_review_normalized_groups": len(
                review_normalized_issues
            ),
        }
    )
    summary = {
        "schema_version": 2,
        "audit_config": {
            "allowed_vehicle_slots": sorted(allowed_slots),
            "max_intent_claim": max_intents,
            "normalization_version": NORMALIZATION_VERSION,
            "quarantine_normalization": "NFKC + lowercase + remove whitespace",
            "review_normalization": (
                "quarantine normalization + remove non-numeric punctuation"
            ),
            "vehicle_domain": vehicle_domain,
        },
        "composition_overlap": composition_overlap(fragments_by_split),
        "composition_overlap_clean_vehicle_aligned": composition_overlap(
            clean_fragments_by_split
        ),
        "issue_counts_by_code": dict(
            sorted(Counter(issue["code"] for issue in issues).items())
        ),
        "splits": split_summaries,
        "totals": dict(sorted(totals.items())),
    }
    return summary, issues


def _manifest_paths(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset = manifest.get("dataset")
    paths = dataset.get("source_files") if isinstance(dataset, Mapping) else None
    if paths is None:
        paths = manifest.get("splits", manifest.get("files"))
    if not isinstance(paths, Mapping) or not paths:
        raise ValueError("manifest must contain dataset.source_files or a splits mapping")
    return paths


def load_local_jsonl(
    manifest_path: Path,
    source_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    source_root = source_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest root must be an object")

    splits: dict[str, list[dict[str, Any]]] = {}
    files = []
    for split_name, specification in sorted(_manifest_paths(manifest).items()):
        specifications = specification if isinstance(specification, list) else [specification]
        rows: list[dict[str, Any]] = []
        for item in specifications:
            raw_path = (
                item.get("relative_path", item.get("path"))
                if isinstance(item, Mapping)
                else item
            )
            if not isinstance(raw_path, str):
                raise ValueError(f"invalid path for split {split_name!r}")
            path = (source_root / raw_path).resolve()
            before = len(rows)
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number}: expected JSON object")
                    rows.append(row)
            observed_sha256 = sha256_file(path)
            observed_records = len(rows) - before
            expected_sha256 = item.get("sha256") if isinstance(item, Mapping) else None
            expected_records = item.get("records") if isinstance(item, Mapping) else None
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "expected_records": expected_records,
                    "expected_sha256": expected_sha256,
                    "records": observed_records,
                    "records_match_manifest": (
                        observed_records == expected_records
                        if expected_records is not None
                        else None
                    ),
                    "relative_path": Path(raw_path).as_posix(),
                    "sha256": observed_sha256,
                    "sha256_matches_manifest": (
                        observed_sha256 == expected_sha256
                        if expected_sha256 is not None
                        else None
                    ),
                    "split": str(split_name),
                }
            )
        splits[str(split_name)] = rows

    provenance = {
        "loader": "local_jsonl",
        "python_version": platform.python_version(),
        "files": sorted(
            files, key=lambda row: (row["split"], row["relative_path"])
        ),
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "name": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
    }
    return splits, provenance, dict(manifest)


def load_hf_dataset(
    dataset_name: str, revision: str | None, *, offline: bool = False
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        import datasets
    except ImportError as error:
        raise RuntimeError("Hugging Face loading requires the optional 'datasets' package") from error

    dataset = datasets.load_dataset(dataset_name, revision=revision)
    splits = {str(name): [dict(row) for row in dataset[name]] for name in sorted(dataset)}
    cache_files = []
    seen_paths = set()
    for split_name in sorted(dataset):
        for entry in getattr(dataset[split_name], "cache_files", []):
            path = Path(entry["filename"]).resolve()
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            cache_files.append(
                {
                    "bytes": path.stat().st_size,
                    "name": path.name,
                    "sha256": sha256_file(path),
                    "split": str(split_name),
                }
            )
    return splits, {
        "datasets_version": datasets.__version__,
        "loader": "huggingface",
        "python_version": platform.python_version(),
        "files": sorted(cache_files, key=lambda row: (row["split"], row["name"])),
    }


def verify_sources(
    files: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    verified_files = []
    issues = []
    for source in files:
        if "expected_sha256" not in source and "expected_records" not in source:
            continue
        result = {
            "split": source["split"],
            "relative_path": source["relative_path"],
            "expected_records": source.get("expected_records"),
            "observed_records": source.get("records"),
            "records_match": source.get("records_match_manifest"),
            "expected_sha256": source.get("expected_sha256"),
            "observed_sha256": source.get("sha256"),
            "sha256_match": source.get("sha256_matches_manifest"),
        }
        verified_files.append(result)
        if result["records_match"] is False:
            issues.append(
                _issue(
                    "source_record_count_mismatch",
                    split=str(result["split"]),
                    detail=result,
                )
            )
        if result["sha256_match"] is False:
            issues.append(
                _issue(
                    "source_sha256_mismatch",
                    split=str(result["split"]),
                    detail=result,
                )
            )
    matches = [
        result[check]
        for result in verified_files
        for check in ("records_match", "sha256_match")
        if result[check] is not None
    ]
    status = "verified" if matches and all(matches) else "mismatch" if matches else "unavailable"
    return {"status": status, "files": verified_files}, issues


def require_verified_source(
    verification: Mapping[str, Any], *, allow_unverified: bool = False
) -> None:
    status = verification.get("status")
    if status == "verified" or allow_unverified:
        return
    raise SourceVerificationError(
        "source verification is "
        f"{status!r}; use --allow-unverified-source only for diagnostics"
    )


def write_outputs(
    output_dir: Path, summary: Mapping[str, Any], issues: Sequence[Mapping[str, str]]
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        issues_path = temporary / "issues.csv"
        with issues_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ISSUE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(issues)
        output_summary = dict(summary)
        output_summary["artifacts"] = {
            "issues": {
                "name": issues_path.name,
                "rows": len(issues),
                "sha256": sha256_file(issues_path),
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
    parser.add_argument("--dataset", help=f"HF dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--revision", help="exact HF dataset revision")
    parser.add_argument("--vehicle-domain", help="vehicle domain label")
    parser.add_argument("--manifest", type=Path, help="source manifest and audit policy")
    parser.add_argument(
        "--source-root", type=Path, help="local root for manifest-relative JSONL files"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--offline", action="store_true", help="forbid HF network access")
    parser.add_argument("--max-intents", type=int, help="published maximum-intent claim")
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="continue after unavailable or mismatched source verification (diagnostics only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest: dict[str, Any] = {}
    if args.manifest:
        loaded = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("manifest root must be an object")
        manifest = loaded

    dataset_metadata = manifest.get("dataset", {})
    if not isinstance(dataset_metadata, Mapping):
        dataset_metadata = {}
    paper_claims = manifest.get("paper_claims", {})
    if not isinstance(paper_claims, Mapping):
        paper_claims = {}

    dataset_name = args.dataset or dataset_metadata.get("id") or DEFAULT_DATASET
    revision = args.revision or dataset_metadata.get("revision")
    vehicle_domain = (
        args.vehicle_domain
        or dataset_metadata.get("vehicle_domain")
        or DEFAULT_VEHICLE_DOMAIN
    )
    max_intents = (
        args.max_intents
        or paper_claims.get("maximum_intents_per_utterance")
        or DEFAULT_MAX_INTENTS
    )
    allowed_slots = manifest.get("known_vehicle_slot_names", DEFAULT_VEHICLE_SLOTS)

    if args.source_root:
        if not args.manifest:
            parser.error("--source-root requires --manifest")
        splits, provenance, manifest = load_local_jsonl(
            args.manifest, args.source_root
        )
    else:
        splits, provenance = load_hf_dataset(
            dataset_name, revision, offline=args.offline
        )
        if args.manifest:
            provenance["manifest"] = {
                "bytes": args.manifest.stat().st_size,
                "name": args.manifest.name,
                "sha256": sha256_file(args.manifest),
            }

    summary, issues = audit_dataset(
        splits,
        vehicle_domain=vehicle_domain,
        max_intents=int(max_intents),
        allowed_vehicle_slots=allowed_slots,
    )
    source_verification, source_issues = verify_sources(provenance.get("files", []))
    if args.manifest:
        require_verified_source(
            source_verification,
            allow_unverified=args.allow_unverified_source,
        )
    issues.extend(source_issues)
    issues.sort(key=lambda row: tuple(row[field] for field in ISSUE_FIELDS))
    summary["issue_counts_by_code"] = dict(
        sorted(Counter(issue["code"] for issue in issues).items())
    )
    summary["source_verification"] = source_verification
    summary["provenance"] = {
        **provenance,
        "dataset": dataset_name,
        "declared_source_files": dataset_metadata.get("source_files"),
        "generator": {
            "name": Path(__file__).name,
            "sha256": sha256_file(Path(__file__)),
        },
        "revision": revision,
    }
    write_outputs(args.output_dir, summary, issues)
    print(f"Wrote {len(issues)} issues to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
