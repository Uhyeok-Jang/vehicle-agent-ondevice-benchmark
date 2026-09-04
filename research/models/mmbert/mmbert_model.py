#!/usr/bin/env python3
"""mmBERT factorized semantic parser for closed-set vehicle control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from transformers import AutoModel


MODEL_ID = "jhu-clsp/mmBERT-small"
MODEL_REVISION = "abc32620dd4f6ab06f5fbe905dc25f310618e09f"

HERE = Path(__file__).resolve().parent
DEFAULT_LABEL_SCHEMA = HERE / "label_schema.v0.1.json"


class MMBertSemanticParser(nn.Module):
    """Encoder-only semantic parser with shared heads across call positions."""

    def __init__(
        self,
        label_schema: Mapping[str, Any],
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.label_schema = dict(label_schema)
        self.max_calls = int(label_schema["max_calls"])

        self.backbone = AutoModel.from_pretrained(
            model_id,
            revision=revision,
        )

        hidden_size = int(self.backbone.config.hidden_size)

        self.hidden_size = hidden_size

        # One learned representation for each ordered call position.
        self.call_position_embedding = nn.Embedding(
            self.max_calls,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        # Predict number of calls from pooled utterance representation.
        self.call_count_head = nn.Linear(
            hidden_size,
            len(label_schema["call_count_labels"]),
        )

        # Same semantic classifiers are shared across all call positions.
        self.semantic_heads = nn.ModuleDict(
            {
                field: nn.Linear(
                    hidden_size,
                    len(values),
                )
                for field, values
                in label_schema["fields"].items()
            }
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state

        # mmBERT tokenizer places <bos> at position 0.
        pooled = hidden[:, 0, :]
        pooled = self.dropout(pooled)

        call_count_logits = self.call_count_head(pooled)

        positions = torch.arange(
            self.max_calls,
            device=pooled.device,
        )

        position_repr = self.call_position_embedding(
            positions
        )

        # [batch, max_calls, hidden]
        call_repr = (
            pooled.unsqueeze(1)
            + position_repr.unsqueeze(0)
        )

        call_repr = self.dropout(call_repr)

        semantic_logits = {
            field: head(call_repr)
            for field, head in self.semantic_heads.items()
        }

        return {
            "call_count_logits": call_count_logits,
            "semantic_logits": semantic_logits,
        }


def load_label_schema(
    path: Path = DEFAULT_LABEL_SCHEMA,
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"label schema must be a JSON object: {path}"
        )

    return value
