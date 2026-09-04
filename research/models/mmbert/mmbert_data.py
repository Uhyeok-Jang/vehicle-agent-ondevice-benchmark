#!/usr/bin/env python3
"""Dataset and tensor labels for the mmBERT semantic parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from mmbert_labels import (
    DATA_ROOT,
    LABEL_FIELDS,
    MAX_CALLS,
    extract_call_labels,
    value_key,
)


IGNORE_INDEX = -100
DEFAULT_MAX_LENGTH = 128


def load_rows(split: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / f"{split}.jsonl"

    with path.open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def build_label_to_id(
    label_schema: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Build typed value -> class ID mappings."""

    mappings: dict[str, dict[str, int]] = {}

    for field, values in label_schema["fields"].items():
        mappings[field] = {
            value_key(value): index
            for index, value in enumerate(values)
        }

    return mappings


def encode_semantic_labels(
    row: Mapping[str, Any],
    label_schema: Mapping[str, Any],
    label_to_id: Mapping[str, Mapping[str, int]],
) -> dict[str, torch.Tensor]:
    """Encode one gold example into masked classification targets."""

    call_count = int(row["call_count"])

    if not 1 <= call_count <= MAX_CALLS:
        raise ValueError(
            f"{row['example_id']}: invalid call_count={call_count}"
        )

    call_count_labels = label_schema["call_count_labels"]

    try:
        call_count_target = call_count_labels.index(call_count)
    except ValueError as error:
        raise ValueError(
            f"{row['example_id']}: unsupported call_count={call_count}"
        ) from error

    field_targets = {
        field: torch.full(
            (MAX_CALLS,),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        for field in LABEL_FIELDS
    }

    for position, call in enumerate(row["canonical_calls"]):
        labels = extract_call_labels(call)

        for field, value in labels.items():
            if field not in field_targets:
                continue

            key = value_key(value)

            try:
                class_id = label_to_id[field][key]
            except KeyError as error:
                raise ValueError(
                    f"{row['example_id']} position={position}: "
                    f"unknown {field} label {value!r}"
                ) from error

            field_targets[field][position] = class_id

    result = {
        "call_count_labels": torch.tensor(
            call_count_target,
            dtype=torch.long,
        )
    }

    for field, tensor in field_targets.items():
        result[f"{field}_labels"] = tensor

    return result


class MMBertVehicleDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer: Any,
        label_schema: Mapping[str, Any],
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_schema = label_schema
        self.max_length = max_length

        self.label_to_id = build_label_to_id(
            label_schema
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.rows[index]

        encoded = self.tokenizer(
            row["utterance_ko"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item: dict[str, Any] = {
            "example_id": row["example_id"],
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded[
                "attention_mask"
            ].squeeze(0),
        }

        item.update(
            encode_semantic_labels(
                row,
                self.label_schema,
                self.label_to_id,
            )
        )

        return item
