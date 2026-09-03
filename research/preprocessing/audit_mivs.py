#!/usr/bin/env python3
"""Audit a pinned MIVS archive against verified raw MAC-SLU sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import audit_macslu


SCHEMA_VERSION = 1
STABLE_ID_SPEC = "mivs-stable-id-v1"
ONTOLOGY_MEMBER = "aispeech/ontology.json"
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
ISSUE_FIELDS = (
    "code",
    "severity",
    "split",
    "member",
    "line",
    "record_id",
    "unit_index",
    "slot_index",
    "detail",
)


class SourceVerificationError(RuntimeError):
    """Raised before output when a pinned input cannot be verified."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_query(value: Any) -> str:
    """Return the MAC v2 quarantine key (punctuation remains significant)."""
    return audit_macslu.normalize_query(value)


def normalize_query_for_review(value: Any) -> str:
    """Return the broader MAC v2 review key."""
    return audit_macslu.normalize_query_for_review(value)


def stable_record_identity(
    revision: str,
    member: str,
    line: int,
    query: str,
    semantics: Any,
) -> dict[str, str]:
    query_raw_sha256 = sha256_bytes(query.encode("utf-8"))
    query_quarantine_sha256 = sha256_bytes(normalize_query(query).encode("utf-8"))
    query_review_sha256 = sha256_bytes(
        normalize_query_for_review(query).encode("utf-8")
    )
    semantics_sha256 = sha256_bytes(canonical_json_bytes(semantics))
    payload = [
        revision,
        PurePosixPath(member).as_posix(),
        line,
        query_quarantine_sha256,
        semantics_sha256,
    ]
    return {
        "record_id": f"mivs:{sha256_bytes(canonical_json_bytes(payload))}",
        "query_quarantine_sha256": query_quarantine_sha256,
        "query_raw_sha256": query_raw_sha256,
        "query_review_sha256": query_review_sha256,
        "semantics_sha256": semantics_sha256,
    }


def _issue(
    code: str,
    *,
    severity: str = "warning",
    split: str = "",
    member: str = "",
    line: int | str = "",
    record_id: str = "",
    unit_index: int | str = "",
    slot_index: int | str = "",
    detail: Any = "",
) -> dict[str, str]:
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False, sort_keys=True)
    return {
        "code": code,
        "severity": severity,
        "split": split,
        "member": member,
        "line": str(line),
        "record_id": record_id,
        "unit_index": str(unit_index),
        "slot_index": str(slot_index),
        "detail": detail,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SourceVerificationError(f"manifest root must be an object: {path.name}")
    return loaded


def _jsonl_rows(raw: bytes, member: str) -> list[tuple[int, dict[str, Any]]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SourceVerificationError(f"member is not UTF-8: {member}") from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SourceVerificationError(
                f"invalid JSON object at {member}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise SourceVerificationError(
                f"expected JSON object at {member}:{line_number}"
            )
        rows.append((line_number, row))
    return rows


def _member_parts(member: str) -> tuple[str, str, str] | None:
    parts = PurePosixPath(member).parts
    if len(parts) != 4 or parts[0] != "aispeech" or not parts[3].endswith(".json"):
        return None
    return parts[1], parts[2], parts[3]


def _ontology_paths(
    ontology: Mapping[str, Any], vehicle_domain: str
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    domain = ontology.get(vehicle_domain)
    if not isinstance(domain, Mapping):
        raise SourceVerificationError(
            f"ontology is missing vehicle domain {vehicle_domain!r}"
        )
    hierarchy = domain.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise SourceVerificationError("vehicle ontology hierarchy must be an object")
    intents = set()
    slots = set()
    paths = set()
    for intent, slot_counts in hierarchy.items():
        if not isinstance(intent, str) or not isinstance(slot_counts, Mapping):
            raise SourceVerificationError("invalid vehicle ontology hierarchy")
        intents.add(intent)
        for slot in slot_counts:
            if not isinstance(slot, str):
                raise SourceVerificationError("vehicle ontology slot must be a string")
            slots.add(slot)
            paths.add((intent, slot))
    return intents, slots, paths


def _vehicle_units(
    row: Mapping[str, Any], member: str, line: int, vehicle_domain: str
) -> tuple[str, list[dict[str, Any]], Any]:
    query = row.get("input")
    semantics = row.get("semantics")
    if not isinstance(query, str) or not isinstance(semantics, list):
        raise SourceVerificationError(f"invalid MIVS record shape at {member}:{line}")
    units = []
    for domain_index, domain in enumerate(semantics):
        if not isinstance(domain, Mapping):
            raise SourceVerificationError(
                f"invalid domain at {member}:{line}:{domain_index}"
            )
        domain_name = domain.get("domain")
        intents = domain.get("intents")
        if not isinstance(domain_name, str) or not isinstance(intents, list):
            raise SourceVerificationError(
                f"invalid domain fields at {member}:{line}:{domain_index}"
            )
        for intent_index, intent in enumerate(intents):
            if not isinstance(intent, Mapping):
                raise SourceVerificationError(
                    f"invalid intent at {member}:{line}:{intent_index}"
                )
            intent_name = intent.get("intent")
            slots = intent.get("slots")
            if not isinstance(intent_name, str) or not isinstance(slots, list):
                raise SourceVerificationError(
                    f"invalid intent fields at {member}:{line}:{intent_index}"
                )
            normalized_slots = []
            for slot_index, slot in enumerate(slots):
                if not isinstance(slot, Mapping):
                    raise SourceVerificationError(
                        f"invalid slot at {member}:{line}:{intent_index}:{slot_index}"
                    )
                name = slot.get("name")
                value = slot.get("value")
                if not isinstance(name, str) or not isinstance(value, str):
                    raise SourceVerificationError(
                        f"invalid slot fields at {member}:{line}:{intent_index}:{slot_index}"
                    )
                normalized_slots.append(dict(slot))
            units.append(
                {
                    "domain": domain_name,
                    "intent": intent_name,
                    "slots": normalized_slots,
                }
            )
    if not units:
        raise SourceVerificationError(
            f"vehicle record has no intent at {member}:{line}"
        )
    return query, units, semantics


def _canonical_mivs_frame(units: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            str(unit["intent"]),
            tuple(
                sorted(
                    (str(slot["name"]), str(slot["value"])) for slot in unit["slots"]
                )
            ),
        )
        for unit in units
    )


def _canonical_mac_frame(units: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    result = []
    for unit in units:
        intents = []
        slots = []
        for slot in unit["slots"]:
            if not isinstance(slot, Mapping):
                continue
            name = slot.get("name")
            value = slot.get("value")
            if name == "intent":
                intents.append(str(value))
            elif name is not None and value is not None:
                slots.append((str(name), str(value)))
        result.append(("|".join(intents), tuple(sorted(slots))))
    return tuple(result)


def _intent_sequence(frame: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(unit[0]) for unit in frame)


def _audit_slot(
    query: str,
    slot: Mapping[str, Any],
    *,
    split: str,
    member: str,
    line: int,
    record_id: str,
    unit_index: int,
    slot_index: int,
) -> tuple[str, dict[str, str] | None]:
    if "pos" not in slot or slot.get("pos") is None:
        return "missing", None
    position = slot["pos"]
    if (
        not isinstance(position, list)
        or len(position) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in position
        )
    ):
        return "invalid", _issue(
            "invalid_pos_shape",
            severity="error",
            split=split,
            member=member,
            line=line,
            record_id=record_id,
            unit_index=unit_index,
            slot_index=slot_index,
            detail={"slot_name": slot["name"]},
        )
    start, end = position
    if start < 0 or end < start or end >= len(query):
        return "invalid", _issue(
            "pos_out_of_bounds",
            severity="error",
            split=split,
            member=member,
            line=line,
            record_id=record_id,
            unit_index=unit_index,
            slot_index=slot_index,
            detail={
                "end": end,
                "query_length": len(query),
                "slot_name": slot["name"],
                "start": start,
            },
        )
    value = str(slot["value"])
    span = query[start : end + 1]
    if span == value:
        return "exact", None
    return "mismatch", _issue(
        "slot_value_span_mismatch",
        split=split,
        member=member,
        line=line,
        record_id=record_id,
        unit_index=unit_index,
        slot_index=slot_index,
        detail={
            "slot_name": slot["name"],
            "span_length": len(span),
            "span_sha256": sha256_bytes(span.encode("utf-8")),
            "value_length": len(value),
            "value_sha256": sha256_bytes(value.encode("utf-8")),
        },
    )


def _count_mismatch(label: str, expected: Any, observed: Any) -> str | None:
    if expected == observed:
        return None
    return f"{label}: expected {expected!r}, observed {observed!r}"


def _verify_expected_counts(
    manifest: Mapping[str, Any],
    release_by_split: Mapping[str, int],
    release_by_component: Mapping[str, Mapping[str, Any]],
    vehicle_by_split: Mapping[str, int],
) -> None:
    declared = manifest.get("release_counts")
    vehicle = manifest.get("vehicle_subset")
    if not isinstance(declared, Mapping) or not isinstance(vehicle, Mapping):
        raise SourceVerificationError("MIVS manifest is missing count declarations")
    errors = []
    checks = [
        (
            "release total",
            declared.get("total_records"),
            sum(release_by_split.values()),
        ),
        ("release splits", dict(declared.get("by_split", {})), dict(release_by_split)),
        (
            "vehicle total",
            vehicle.get("total_records"),
            sum(vehicle_by_split.values()),
        ),
        ("vehicle splits", dict(vehicle.get("by_split", {})), dict(vehicle_by_split)),
    ]
    components = declared.get("components")
    if not isinstance(components, Mapping):
        raise SourceVerificationError("MIVS manifest components must be an object")
    for name, specification in components.items():
        if not isinstance(specification, Mapping):
            raise SourceVerificationError(f"invalid component declaration: {name}")
        observed = release_by_component.get(
            str(name), {"total_records": 0, "by_split": {}}
        )
        checks.extend(
            [
                (
                    f"component {name} total",
                    specification.get("total_records"),
                    observed["total_records"],
                ),
                (
                    f"component {name} splits",
                    dict(specification.get("by_split", {})),
                    dict(observed["by_split"]),
                ),
            ]
        )
    for label, expected, observed in checks:
        if mismatch := _count_mismatch(label, expected, observed):
            errors.append(mismatch)
    if errors:
        raise SourceVerificationError("; ".join(errors))


def _load_verified_mac(
    manifest_path: Path, source_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        splits, provenance, manifest = audit_macslu.load_local_jsonl(
            manifest_path, source_root
        )
        verification, _ = audit_macslu.verify_sources(provenance.get("files", []))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SourceVerificationError(f"cannot load raw MAC source: {error}") from error
    if verification.get("status") != "verified":
        raise SourceVerificationError("raw MAC source does not match its manifest")
    source_files = provenance.get("files", [])
    if not source_files or any(
        not isinstance(source.get("expected_sha256"), str)
        or not isinstance(source.get("expected_records"), int)
        or source.get("sha256_matches_manifest") is not True
        or source.get("records_match_manifest") is not True
        for source in source_files
    ):
        raise SourceVerificationError(
            "every raw MAC JSONL source requires matching SHA-256 and record count"
        )

    dataset = manifest.get("dataset", {})
    vehicle_domain = (
        dataset.get("vehicle_domain") if isinstance(dataset, Mapping) else None
    )
    if not isinstance(vehicle_domain, str):
        raise SourceVerificationError("MAC manifest is missing dataset.vehicle_domain")
    revision = dataset.get("revision")
    if not isinstance(revision, str):
        raise SourceVerificationError("MAC manifest is missing dataset.revision")

    records = []
    for split in sorted(splits):
        for row in splits[split]:
            query = row.get("query")
            if not isinstance(query, str):
                raise SourceVerificationError(f"MAC query is not a string in {split}")
            units = [
                unit
                for unit in audit_macslu.extract_active_semantic_units(row)
                if unit["domain"] == vehicle_domain
            ]
            if not units:
                continue
            frame = _canonical_mac_frame(units)
            raw_id = str(row.get("id") or "")
            record_key_payload = [
                revision,
                split,
                sha256_bytes(raw_id.encode("utf-8")),
                sha256_bytes(query.encode("utf-8")),
                sha256_bytes(canonical_json_bytes(frame)),
            ]
            record_key = f"mac:{sha256_bytes(canonical_json_bytes(record_key_payload))}"
            records.append(
                {
                    "split": split,
                    "record_key": record_key,
                    "query": query,
                    "quarantine_query": normalize_query(query),
                    "review_query": normalize_query_for_review(query),
                    "frame": frame,
                }
            )
    files = []
    for source in provenance.get("files", []):
        files.append(
            {
                "bytes": source["bytes"],
                "expected_records": source.get("expected_records"),
                "expected_sha256": source.get("expected_sha256"),
                "records": source.get("records"),
                "relative_path": source["relative_path"],
                "sha256": source["sha256"],
                "split": source["split"],
            }
        )
    return records, {
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "name": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "revision": revision,
        "source_files": sorted(
            files, key=lambda item: (item["split"], item["relative_path"])
        ),
        "total_records": sum(len(rows) for rows in splits.values()),
        "unique_vehicle_queries": len({record["query"] for record in records}),
        "vehicle_records": len(records),
    }


def _cross_split_overlap(
    records: Sequence[dict[str, Any]],
    *,
    key_field: str,
    flag: str,
    issue_code: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = record[key_field]
        if key:
            groups[key].append(record)

    pairwise = {"train_valid": 0, "train_test": 0, "valid_test": 0}
    by_split_set: Counter[str] = Counter()
    issues = []
    for key, matches in groups.items():
        splits = sorted(
            {record["split"] for record in matches},
            key=lambda value: SPLIT_ORDER[value],
        )
        if len(splits) < 2:
            continue
        by_split_set["|".join(splits)] += 1
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if left in splits and right in splits:
                pairwise[f"{left}_{right}"] += 1
        for record in matches:
            record["public"]["overlap"][flag] = True
        issues.append(
            _issue(
                issue_code,
                split="|".join(splits),
                detail={
                    "query_key_sha256": sha256_bytes(key.encode("utf-8")),
                    "record_ids": sorted(
                        record["public"]["record_id"] for record in matches
                    ),
                },
            )
        )
    return {
        "groups": sum(by_split_set.values()),
        "groups_by_split_set": dict(sorted(by_split_set.items())),
        "pairwise_unique_query_keys": pairwise,
    }, issues


def _mac_overlap(
    mivs_records: Sequence[dict[str, Any]],
    mac_records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    tiers = {
        "exact": "query",
        "quarantine_normalized": "quarantine_query",
        "review_normalized": "review_query",
    }
    mivs_maps: dict[str, dict[str, list[dict[str, Any]]]] = {}
    mac_maps: dict[str, dict[str, list[dict[str, Any]]]] = {}
    matched_keys: dict[str, list[str]] = {}
    for tier, field in tiers.items():
        mivs_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        mac_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in mivs_records:
            if key := record[field]:
                mivs_map[key].append(record)
        for record in mac_records:
            if key := record[field]:
                mac_map[key].append(record)
        mivs_maps[tier] = mivs_map
        mac_maps[tier] = mac_map
        matched_keys[tier] = sorted(set(mivs_map) & set(mac_map))

    exact_keys = matched_keys["exact"]
    intent_equal = 0
    frame_equal = 0
    issues: list[dict[str, str]] = []
    for key in exact_keys:
        left = mivs_maps["exact"][key]
        right = mac_maps["exact"][key]
        if any(
            _intent_sequence(mivs["frame"]) == _intent_sequence(mac["frame"])
            for mivs in left
            for mac in right
        ):
            intent_equal += 1
        if any(mivs["frame"] == mac["frame"] for mivs in left for mac in right):
            frame_equal += 1
    for tier in tiers:
        for key in matched_keys[tier]:
            left = mivs_maps[tier][key]
            right = mac_maps[tier][key]
            for record in left:
                record["public"]["overlap"][f"mac_{tier}"] = True
            issues.append(
                _issue(
                    f"mac_query_overlap_{tier}",
                    detail={
                        "mac_record_keys": sorted(
                            record["record_key"] for record in right
                        ),
                        "mivs_record_ids": sorted(
                            record["public"]["record_id"] for record in left
                        ),
                        "query_key_sha256": sha256_bytes(key.encode("utf-8")),
                    },
                )
            )

    quarantine_additional = sum(
        not any(
            mivs["query"] == mac["query"]
            for mivs in mivs_maps["quarantine_normalized"][key]
            for mac in mac_maps["quarantine_normalized"][key]
        )
        for key in matched_keys["quarantine_normalized"]
    )
    review_additional = sum(
        not any(
            mivs["quarantine_query"] == mac["quarantine_query"]
            for mivs in mivs_maps["review_normalized"][key]
            for mac in mac_maps["review_normalized"][key]
        )
        for key in matched_keys["review_normalized"]
    )

    exact_set = set(exact_keys)
    test_matches = [
        record
        for record in mivs_records
        if record["split"] == "test" and record["query"] in exact_set
    ]
    by_mac_split: Counter[str] = Counter()
    for record in test_matches:
        for split in {item["split"] for item in mac_maps["exact"][record["query"]]}:
            by_mac_split[split] += 1

    test_records = [record for record in mivs_records if record["split"] == "test"]
    exact_records = sum(
        record["public"]["overlap"]["mac_exact"] for record in test_records
    )
    quarantine_records = sum(
        record["public"]["overlap"]["mac_quarantine_normalized"]
        for record in test_records
    )
    review_records = sum(
        record["public"]["overlap"]["mac_review_normalized"] for record in test_records
    )

    return {
        "full_vehicle": {
            "exact_unique_queries": len(exact_keys),
            "full_frame_equal_unique_queries": frame_equal,
            "intent_sequence_equal_unique_queries": intent_equal,
            "quarantine_normalized_additional_unique_keys": quarantine_additional,
            "quarantine_normalized_total_unique_keys": len(
                matched_keys["quarantine_normalized"]
            ),
            "review_normalized_additional_unique_keys": review_additional,
            "review_normalized_total_unique_keys": len(
                matched_keys["review_normalized"]
            ),
        },
        "recommended_test": {
            "exact_records": exact_records,
            "exact_records_by_mac_split_nonexclusive": dict(
                sorted(by_mac_split.items())
            ),
            "multi_exact_records": sum(
                record["partition"] == "multi" for record in test_matches
            ),
            "quarantine_normalized_additional_records": (
                quarantine_records - exact_records
            ),
            "quarantine_normalized_records": quarantine_records,
            "review_normalized_additional_records": (
                review_records - quarantine_records
            ),
            "review_normalized_records": review_records,
        },
    }, issues


def audit_sources(
    manifest_path: Path,
    archive_path: Path,
    mac_manifest_path: Path,
    mac_source_root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    archive_path = archive_path.resolve()
    mac_manifest_path = mac_manifest_path.resolve()
    mac_source_root = mac_source_root.resolve()
    manifest = _load_manifest(manifest_path)

    dataset = manifest.get("dataset")
    vehicle_spec = manifest.get("vehicle_subset")
    release_spec = manifest.get("release_counts")
    if not isinstance(dataset, Mapping) or not isinstance(vehicle_spec, Mapping):
        raise SourceVerificationError("MIVS manifest is missing dataset metadata")
    repository = dataset.get("repository")
    archive_spec = dataset.get("archive")
    if not isinstance(repository, Mapping) or not isinstance(archive_spec, Mapping):
        raise SourceVerificationError(
            "MIVS manifest is missing repository/archive metadata"
        )
    revision = repository.get("revision")
    expected_archive_sha256 = archive_spec.get("sha256")
    vehicle_domain = vehicle_spec.get("domain")
    if not all(
        isinstance(value, str)
        for value in (revision, expected_archive_sha256, vehicle_domain)
    ):
        raise SourceVerificationError(
            "MIVS revision, archive hash, and vehicle domain are required"
        )

    observed_archive_sha256 = sha256_file(archive_path)
    if observed_archive_sha256 != expected_archive_sha256:
        raise SourceVerificationError("MIVS archive SHA-256 does not match manifest")

    if not isinstance(release_spec, Mapping) or not isinstance(
        release_spec.get("components"), Mapping
    ):
        raise SourceVerificationError("MIVS manifest components are required")
    directory_to_component = {}
    for component, specification in release_spec["components"].items():
        if not isinstance(specification, Mapping) or not isinstance(
            specification.get("source_directory"), str
        ):
            raise SourceVerificationError(f"invalid MIVS component: {component}")
        directory = specification["source_directory"]
        if directory in directory_to_component:
            raise SourceVerificationError(f"duplicate source directory: {directory}")
        directory_to_component[directory] = str(component)

    issues: list[dict[str, str]] = []
    members = []
    release_by_split: Counter[str] = Counter({split: 0 for split in SPLIT_ORDER})
    release_by_component: dict[str, dict[str, Any]] = {
        component: {
            "total_records": 0,
            "by_split": {split: 0 for split in SPLIT_ORDER},
        }
        for component in release_spec["components"]
    }
    vehicle_by_split: Counter[str] = Counter({split: 0 for split in SPLIT_ORDER})
    vehicle_units_by_split: Counter[str] = Counter({split: 0 for split in SPLIT_ORDER})
    partition_records: Counter[str] = Counter()
    partition_units: Counter[str] = Counter()
    source_unit_counts: Counter[int] = Counter()
    offset_totals: Counter[str] = Counter()
    offset_by_slot: dict[str, Counter[str]] = defaultdict(Counter)
    observed_intents = set()
    observed_slots = set()
    internal_records: list[dict[str, Any]] = []

    try:
        archive = zipfile.ZipFile(archive_path, metadata_encoding="utf-8")
    except (OSError, zipfile.BadZipFile) as error:
        raise SourceVerificationError("cannot open MIVS archive") from error
    with archive:
        infos = archive.infolist()
        duplicate_members = sorted(
            name
            for name, count in Counter(info.filename for info in infos).items()
            if count > 1
        )
        if duplicate_members:
            raise SourceVerificationError(
                f"archive contains duplicate members: {duplicate_members!r}"
            )
        by_name = {info.filename: info for info in infos}
        ontology_info = by_name.get(ONTOLOGY_MEMBER)
        if ontology_info is None:
            raise SourceVerificationError(f"archive is missing {ONTOLOGY_MEMBER}")
        ontology_raw = archive.read(ontology_info)
        try:
            ontology = json.loads(ontology_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SourceVerificationError("invalid MIVS ontology JSON") from error
        if not isinstance(ontology, Mapping):
            raise SourceVerificationError("MIVS ontology root must be an object")
        ontology_intents, ontology_slots, ontology_paths = _ontology_paths(
            ontology, vehicle_domain
        )
        members.append(
            {
                "bytes": ontology_info.file_size,
                "compressed_bytes": ontology_info.compress_size,
                "kind": "ontology",
                "path": ONTOLOGY_MEMBER,
                "sha256": sha256_bytes(ontology_raw),
            }
        )

        data_infos = []
        for info in infos:
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            parts = _member_parts(info.filename)
            if parts is None:
                continue
            split, directory, _ = parts
            if split not in SPLIT_ORDER:
                continue
            if directory not in directory_to_component:
                raise SourceVerificationError(
                    f"unrecognized JSON data directory in archive: {directory}"
                )
            data_infos.append(info)
        if not data_infos:
            raise SourceVerificationError(
                "archive contains no declared JSONL data members"
            )

        expected_vehicle_members = {
            f"aispeech/{split}/one_domain_data/{vehicle_domain}{suffix}.json"
            for split in SPLIT_ORDER
            for suffix in ("", "_multi")
        }
        missing_vehicle_members = sorted(expected_vehicle_members - set(by_name))
        if missing_vehicle_members:
            raise SourceVerificationError(
                f"archive is missing vehicle members: {missing_vehicle_members!r}"
            )

        for info in sorted(data_infos, key=lambda item: item.filename):
            split, directory, filename = _member_parts(info.filename) or ("", "", "")
            component = directory_to_component[directory]
            raw = archive.read(info)
            rows = _jsonl_rows(raw, info.filename)
            row_count = len(rows)
            release_by_split[split] += row_count
            release_by_component[component]["total_records"] += row_count
            release_by_component[component]["by_split"][split] += row_count
            members.append(
                {
                    "bytes": info.file_size,
                    "component": component,
                    "compressed_bytes": info.compress_size,
                    "kind": "data",
                    "path": info.filename,
                    "records": row_count,
                    "sha256": sha256_bytes(raw),
                    "split": split,
                }
            )
            if info.filename not in expected_vehicle_members:
                continue

            partition = "multi" if filename.endswith("_multi.json") else "single"
            vehicle_by_split[split] += row_count
            partition_records[partition] += row_count
            for line, row in rows:
                query, units, semantics = _vehicle_units(
                    row, info.filename, line, vehicle_domain
                )
                identity = stable_record_identity(
                    revision, info.filename, line, query, semantics
                )
                record_id = identity["record_id"]
                public = {
                    **identity,
                    "intent_count": len(units),
                    "line": line,
                    "member": info.filename,
                    "offsets": {
                        "exact": 0,
                        "invalid": 0,
                        "mismatch": 0,
                        "missing": 0,
                        "present": 0,
                    },
                    "ontology_valid": True,
                    "overlap": {
                        "mac_exact": False,
                        "mac_quarantine_normalized": False,
                        "mac_review_normalized": False,
                        "mivs_exact": False,
                        "mivs_quarantine_normalized": False,
                        "mivs_review_normalized": False,
                    },
                    "partition": partition,
                    "revision": revision,
                    "slot_count": sum(len(unit["slots"]) for unit in units),
                    "split": split,
                    "unit_ids": [
                        f"{record_id}:i{unit_index:02d}"
                        for unit_index in range(len(units))
                    ],
                }
                for unit_index, unit in enumerate(units):
                    intent = unit["intent"]
                    observed_intents.add(intent)
                    if (
                        unit["domain"] != vehicle_domain
                        or intent not in ontology_intents
                    ):
                        public["ontology_valid"] = False
                        issues.append(
                            _issue(
                                "ontology_intent_violation",
                                severity="error",
                                split=split,
                                member=info.filename,
                                line=line,
                                record_id=record_id,
                                unit_index=unit_index,
                                detail={
                                    "domain": unit["domain"],
                                    "intent": intent,
                                },
                            )
                        )
                    for slot_index, slot in enumerate(unit["slots"]):
                        slot_name = slot["name"]
                        observed_slots.add(slot_name)
                        if (intent, slot_name) not in ontology_paths:
                            public["ontology_valid"] = False
                            issues.append(
                                _issue(
                                    "ontology_slot_path_violation",
                                    severity="error",
                                    split=split,
                                    member=info.filename,
                                    line=line,
                                    record_id=record_id,
                                    unit_index=unit_index,
                                    slot_index=slot_index,
                                    detail={
                                        "intent": intent,
                                        "slot_name": slot_name,
                                    },
                                )
                            )
                        offset_totals["slots"] += 1
                        offset_by_slot[slot_name]["slots"] += 1
                        status, issue = _audit_slot(
                            query,
                            slot,
                            split=split,
                            member=info.filename,
                            line=line,
                            record_id=record_id,
                            unit_index=unit_index,
                            slot_index=slot_index,
                        )
                        offset_totals[status] += 1
                        offset_by_slot[slot_name][status] += 1
                        public["offsets"][status] += 1
                        if status != "missing":
                            public["offsets"]["present"] += 1
                        if issue is not None:
                            issues.append(issue)
                vehicle_units_by_split[split] += len(units)
                partition_units[partition] += len(units)
                if split == "test":
                    source_unit_counts[len(units)] += 1
                internal_records.append(
                    {
                        "frame": _canonical_mivs_frame(units),
                        "partition": partition,
                        "public": public,
                        "quarantine_query": normalize_query(query),
                        "query": query,
                        "review_query": normalize_query_for_review(query),
                        "split": split,
                    }
                )

    _verify_expected_counts(
        manifest,
        dict(release_by_split),
        release_by_component,
        dict(vehicle_by_split),
    )
    ids = [record["public"]["record_id"] for record in internal_records]
    if len(ids) != len(set(ids)):
        raise SourceVerificationError("stable record ID collision")

    mac_records, mac_provenance = _load_verified_mac(mac_manifest_path, mac_source_root)
    exact_overlap, exact_issues = _cross_split_overlap(
        internal_records,
        key_field="query",
        flag="mivs_exact",
        issue_code="mivs_cross_split_query_exact",
    )
    quarantine_overlap, quarantine_issues = _cross_split_overlap(
        internal_records,
        key_field="quarantine_query",
        flag="mivs_quarantine_normalized",
        issue_code="mivs_cross_split_query_quarantine_normalized",
    )
    review_overlap, review_issues = _cross_split_overlap(
        internal_records,
        key_field="review_query",
        flag="mivs_review_normalized",
        issue_code="mivs_cross_split_query_review_normalized",
    )
    mac_overlap, mac_issues = _mac_overlap(internal_records, mac_records)
    issues.extend(exact_issues)
    issues.extend(quarantine_issues)
    issues.extend(review_issues)
    issues.extend(mac_issues)
    issues.sort(key=lambda row: tuple(row[field] for field in ISSUE_FIELDS))

    declared_intents = set(map(str, vehicle_spec.get("intents", [])))
    declared_slots = set(map(str, vehicle_spec.get("slots", [])))
    by_slot = {}
    for slot, counts in sorted(offset_by_slot.items()):
        present = counts["exact"] + counts["mismatch"] + counts["invalid"]
        by_slot[slot] = {
            "pos_exact": counts["exact"],
            "pos_invalid": counts["invalid"],
            "pos_mismatch": counts["mismatch"],
            "pos_missing": counts["missing"],
            "pos_present": present,
            "slots": counts["slots"],
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_config": {
            "exact_key": "raw Unicode string equality",
            "quarantine_normalization": (
                "Unicode NFKC + lowercase + remove whitespace; preserve punctuation"
            ),
            "review_normalization": (
                "quarantine normalization + remove punctuation except decimal points "
                "and numeric signs"
            ),
            "offset_convention": "zero-based inclusive Unicode code-point indices",
            "revision": revision,
            "stable_id_spec": STABLE_ID_SPEC,
            "vehicle_domain": vehicle_domain,
        },
        "counts": {
            "recommended_test": {
                "source_unit_count_distribution": {
                    str(count): amount
                    for count, amount in sorted(source_unit_counts.items())
                },
                "records": vehicle_by_split["test"],
                "units": vehicle_units_by_split["test"],
            },
            "release": {
                "by_component": release_by_component,
                "by_split": dict(release_by_split),
                "records": sum(release_by_split.values()),
            },
            "vehicle": {
                "by_split": dict(vehicle_by_split),
                "partition_records": dict(sorted(partition_records.items())),
                "partition_units": dict(sorted(partition_units.items())),
                "records": sum(vehicle_by_split.values()),
                "units": sum(vehicle_units_by_split.values()),
                "units_by_split": dict(vehicle_units_by_split),
            },
        },
        "issue_counts_by_code": dict(
            sorted(Counter(issue["code"] for issue in issues).items())
        ),
        "offsets": {
            "by_slot": by_slot,
            "pos_exact": offset_totals["exact"],
            "pos_invalid": offset_totals["invalid"],
            "pos_mismatch": offset_totals["mismatch"],
            "pos_missing": offset_totals["missing"],
            "pos_present": (
                offset_totals["exact"]
                + offset_totals["mismatch"]
                + offset_totals["invalid"]
            ),
            "slots": offset_totals["slots"],
        },
        "ontology": {
            "declared_intents_match_observed": declared_intents == observed_intents,
            "declared_slots_match_observed": declared_slots == observed_slots,
            "intent_count": len(ontology_intents),
            "manifest_intents_match_ontology": declared_intents == ontology_intents,
            "manifest_slots_match_ontology": declared_slots == ontology_slots,
            "observed_path_violations": sum(
                issue["code"].startswith("ontology_") for issue in issues
            ),
            "slot_count": len(ontology_slots),
            "vehicle_intents": sorted(ontology_intents),
            "vehicle_slots": sorted(ontology_slots),
        },
        "overlap": {
            "mac": mac_overlap,
            "mivs_cross_split": {
                "exact": exact_overlap,
                "quarantine_normalized": quarantine_overlap,
                "review_normalized": review_overlap,
            },
        },
        "provenance": {
            "archive": {
                "bytes": archive_path.stat().st_size,
                "expected_sha256": expected_archive_sha256,
                "observed_sha256": observed_archive_sha256,
                "sha256_match": True,
            },
            "mac": mac_provenance,
            "manifest": {
                "bytes": manifest_path.stat().st_size,
                "name": manifest_path.name,
                "sha256": sha256_file(manifest_path),
            },
            "members": sorted(members, key=lambda item: item["path"]),
        },
        "source_verification": {
            "archive_sha256": "verified",
            "declared_counts": "verified",
            "mac_raw_sources": "verified",
            "status": "verified",
        },
    }
    public_records = [record["public"] for record in internal_records]
    return summary, issues, public_records


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    issues: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, Any]],
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
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (temporary / "issues.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=ISSUE_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(issues)
        ordered_records = sorted(
            records,
            key=lambda record: (
                SPLIT_ORDER[str(record["split"])],
                str(record["member"]),
                int(record["line"]),
            ),
        )
        with (temporary / "record_index.jsonl").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            for record in ordered_records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mac-manifest", type=Path, required=True)
    parser.add_argument("--mac-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    try:
        summary, issues, records = audit_sources(
            args.manifest,
            args.archive,
            args.mac_manifest,
            args.mac_source_root,
        )
    except (OSError, SourceVerificationError) as error:
        parser.exit(2, f"audit_mivs: {error}\n")
    write_outputs(args.output_dir, summary, issues, records)
    print(
        f"Wrote {len(records)} record identities and {len(issues)} findings "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
