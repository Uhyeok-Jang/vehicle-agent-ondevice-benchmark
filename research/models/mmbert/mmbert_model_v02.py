#!/usr/bin/env python3
"""mmBERT v0.2 with position-specific token attention pooling."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from transformers import AutoModel

from mmbert_model import (
    MODEL_ID,
    MODEL_REVISION,
    load_label_schema,
)


class MMBertSemanticParser(nn.Module):
    """Encoder semantic parser with one learned token-attention query per call."""

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

        hidden_size = int(
            self.backbone.config.hidden_size
        )

        self.hidden_size = hidden_size

        # Global representation for call-count prediction.
        self.call_count_head = nn.Linear(
            hidden_size,
            len(label_schema["call_count_labels"]),
        )

        # One learned attention query for each ordered call position.
        self.call_queries = nn.Embedding(
            self.max_calls,
            hidden_size,
        )

        # Position identity is also retained in the final call representation.
        self.call_position_embedding = nn.Embedding(
            self.max_calls,
            hidden_size,
        )

        self.call_norm = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Shared semantic heads across all call positions.
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

        # [B, L, H]
        hidden = outputs.last_hidden_state

        # <bos> global representation.
        pooled = hidden[:, 0, :]
        pooled = self.dropout(pooled)

        call_count_logits = self.call_count_head(
            pooled
        )

        batch_size = hidden.shape[0]

        # [C, H]
        queries = self.call_queries.weight

        # Each call query independently attends to all utterance tokens.
        # [B, C, L]
        attention_scores = torch.einsum(
            "ch,blh->bcl",
            queries,
            hidden,
        ) / math.sqrt(self.hidden_size)

        if attention_mask is not None:
            valid_mask = attention_mask.bool().clone()

            # Do not let every call query collapse onto <bos>.
            if valid_mask.shape[1] > 0:
                valid_mask[:, 0] = False

            attention_scores = (
                attention_scores.masked_fill(
                    ~valid_mask.unsqueeze(1),
                    torch.finfo(
                        attention_scores.dtype
                    ).min,
                )
            )

        # Compute softmax in fp32 for stability.
        attention_weights = torch.softmax(
            attention_scores.float(),
            dim=-1,
        ).to(hidden.dtype)

        # [B, C, H]
        call_context = torch.einsum(
            "bcl,blh->bch",
            attention_weights,
            hidden,
        )

        positions = torch.arange(
            self.max_calls,
            device=hidden.device,
        )

        position_embedding = (
            self.call_position_embedding(
                positions
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # Global utterance meaning + position-specific token evidence.
        call_repr = self.call_norm(
            call_context
            + pooled.unsqueeze(1)
            + position_embedding
        )

        call_repr = self.dropout(
            call_repr
        )

        semantic_logits = {
            field: head(call_repr)
            for field, head
            in self.semantic_heads.items()
        }

        return {
            "call_count_logits":
                call_count_logits,
            "semantic_logits":
                semantic_logits,

            # Diagnostic only; not needed by evaluator.
            "call_attention_weights":
                attention_weights,
        }
