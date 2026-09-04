#!/usr/bin/env python3
"""Build a globally deduplicated deterministic benchmark split."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    RESEARCH_ROOT
    / "data"
    / "processed"
    / "macslu_korean_v0.1"
)

OUTPUT_ROOT = (
    RESEARCH_ROOT
    / "data"
    / "processed"
    / "macslu_korean_v0.2"
)

MMBERT_ROOT = (
    RESEARCH_ROOT
    / "models"
    / "mmbert"
)

if str(MMBERT_ROOT) not in sys.path:
    sys.path.insert(0, str(MMBERT_ROOT))

from mmbert_labels import extract_call_labels, value_key


SEED = 20260904


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.strip(),
    )


def canonical_key(calls: Any) -> str:
    return json.dumps(
        calls,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_all() -> list[dict[str, Any]]:
    rows = []

    for split in ("train", "validation", "test"):
        path = INPUT_ROOT / f"{split}.jsonl"

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                row = json.loads(line)
                row["_original_benchmark_split"] = split
                rows.append(row)

    return rows


def deduplicate(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = normalize_text(row["utterance_ko"])
        groups[key].append(row)

    unique_rows = []

    for utterance_key, members in sorted(groups.items()):
        golds = {
            canonical_key(row["canonical_calls"])
            for row in members
        }

        if len(golds) != 1:
            raise RuntimeError(
                "identical Korean utterance has conflicting gold:\n"
                f"{utterance_key}\n"
                f"{sorted(golds)}"
            )

        call_counts = {
            int(row["call_count"])
            for row in members
        }

        if len(call_counts) != 1:
            raise RuntimeError(
                "duplicate group has conflicting call_count: "
                f"{utterance_key}"
            )

        # Deterministic representative.
        members = sorted(
            members,
            key=lambda row: row["example_id"],
        )

        representative = dict(members[0])

        representative["deduplication"] = {
            "version": "v0.2",
            "normalized_utterance": utterance_key,
            "group_size": len(members),
            "original_example_ids": [
                row["example_id"]
                for row in members
            ],
            "original_splits": sorted(
                {
                    row["_original_benchmark_split"]
                    for row in members
                }
            ),
        }

        representative.pop(
            "_original_benchmark_split",
            None,
        )

        unique_rows.append(representative)

    return unique_rows


def split_rows(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic call-count-stratified ~80/10/10 split."""

    rng = random.Random(SEED)

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        buckets[int(row["call_count"])].append(row)

    output = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for call_count in sorted(buckets):
        bucket = sorted(
            buckets[call_count],
            key=lambda row: row["example_id"],
        )

        rng.shuffle(bucket)

        n = len(bucket)

        # Keep very small strata in train.
        if n >= 10:
            n_test = max(
                1,
                round(n * 0.10),
            )
            n_validation = max(
                1,
                round(n * 0.10),
            )
        elif n >= 6:
            n_test = 1
            n_validation = 1
        else:
            n_test = 0
            n_validation = 0

        test_rows = bucket[:n_test]

        validation_rows = bucket[
            n_test:
            n_test + n_validation
        ]

        train_rows = bucket[
            n_test + n_validation:
        ]

        output["test"].extend(test_rows)
        output["validation"].extend(validation_rows)
        output["train"].extend(train_rows)

        print(
            f"call_count={call_count}: "
            f"total={n}, "
            f"train={len(train_rows)}, "
            f"validation={len(validation_rows)}, "
            f"test={len(test_rows)}"
        )

    for split in output:
        output[split].sort(
            key=lambda row: row["example_id"]
        )

        for row in output[split]:
            row["benchmark_split"] = split
            row["benchmark_version"] = "macslu_korean_v0.2"

    return output


def audit_no_leakage(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    print("\n===== LEAKAGE AUDIT =====")

    keys = {}

    for split, rows in splits.items():
        values = {
            normalize_text(row["utterance_ko"])
            for row in rows
        }

        if len(values) != len(rows):
            raise RuntimeError(
                f"{split}: internal utterance duplicate remains"
            )

        keys[split] = values

        print(
            f"{split}: "
            f"examples={len(rows)}, "
            f"unique utterances={len(values)}"
        )

    for a, b in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = keys[a] & keys[b]

        print(
            f"{a} vs {b} utterance overlap:",
            len(overlap),
        )

        if overlap:
            raise RuntimeError(
                f"{a}/{b}: leakage remains"
            )


def primitive_vocab(
    rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    vocab: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        for call in row["canonical_calls"]:
            labels = extract_call_labels(call)

            for field, value in labels.items():
                vocab[field].add(
                    value_key(value)
                )

    return vocab


def audit_primitive_oov(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    train_vocab = primitive_vocab(
        splits["train"]
    )

    print("\n===== PRIMITIVE OOV AUDIT =====")

    total = 0

    for split in ("validation", "test"):
        vocab = primitive_vocab(
            splits[split]
        )

        split_total = 0

        for field in sorted(vocab):
            oov = (
                vocab[field]
                - train_vocab.get(field, set())
            )

            if oov:
                print(
                    f"{split} {field}:",
                    sorted(oov),
                )
                split_total += len(oov)

        print(
            f"{split} primitive OOV:",
            split_total,
        )

        total += split_total

    if total != 0:
        raise RuntimeError(
            "primitive OOV exists; do not freeze this split"
        )


def write_splits(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, rows in splits.items():
        path = OUTPUT_ROOT / f"{split}.jsonl"

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        print(
            f"written {split}: "
            f"{len(rows)} -> {path}"
        )


def main() -> int:
    rows = load_all()

    print("original examples:", len(rows))

    unique_rows = deduplicate(rows)

    print(
        "globally unique Korean utterances:",
        len(unique_rows),
    )

    print(
        "removed exact duplicate examples:",
        len(rows) - len(unique_rows),
    )

    print("\n===== STRATIFIED SPLIT =====")

    splits = split_rows(unique_rows)

    audit_no_leakage(splits)
    audit_primitive_oov(splits)

    print("\n===== FINAL COUNTS =====")

    for split in (
        "train",
        "validation",
        "test",
    ):
        calls = sum(
            row["call_count"]
            for row in splits[split]
        )

        print(
            f"{split}: "
            f"examples={len(splits[split])}, "
            f"calls={calls}"
        )

    write_splits(splits)

    print(
        "\nPASS: leakage-safe benchmark v0.2 created"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
