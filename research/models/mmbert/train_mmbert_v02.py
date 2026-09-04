#!/usr/bin/env python3
"""Fine-tune mmBERT factorized semantic parser."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mmbert_data import (
    IGNORE_INDEX,
    MMBertVehicleDataset,
    load_rows,
)
from mmbert_labels import (
    FUNCTIONS_WITH_OPTIONAL_ZONE,
    LabelError,
    assemble_call,
)
from mmbert_model_v02 import (
    MODEL_ID,
    MODEL_REVISION,
    MMBertSemanticParser,
    load_label_schema,
)

from mmbert_constrained_decode import decode_batch as constrained_decode_batch


HERE = Path(__file__).resolve().parent
RESEARCH_ROOT = HERE.parents[1]

PREPROCESSING_ROOT = RESEARCH_ROOT / "preprocessing"
if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_ROOT))

import canonical_vehicle_api as canonical


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

CHECKPOINT_PATH = (
    RESEARCH_ROOT
    / "checkpoints"
    / "mmbert"
    / "best_v02.pt"
)

RESULT_ROOT = (
    RESEARCH_ROOT
    / "results"
    / "mmbert_attention_v0.2"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    result = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(
                device,
                non_blocking=True,
            )
        else:
            result[key] = value

    return result


def compute_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    call_count_loss = F.cross_entropy(
        outputs["call_count_logits"],
        batch["call_count_labels"],
    )

    semantic_losses: list[torch.Tensor] = []
    log_values = {
        "call_count": float(
            call_count_loss.detach().item()
        )
    }

    for field, logits in outputs[
        "semantic_logits"
    ].items():
        targets = batch[f"{field}_labels"]

        active = targets != IGNORE_INDEX

        # Some sparse heads may have no active label in a batch.
        if not active.any():
            continue

        field_loss = F.cross_entropy(
            logits[active],
            targets[active],
        )

        semantic_losses.append(field_loss)

        log_values[field] = float(
            field_loss.detach().item()
        )

    if semantic_losses:
        semantic_loss = torch.stack(
            semantic_losses
        ).mean()
    else:
        semantic_loss = torch.zeros(
            (),
            device=call_count_loss.device,
        )

    # Give call-count prediction and semantic parsing
    # comparable total weight.
    total_loss = (
        call_count_loss
        + semantic_loss
    )

    return total_loss, log_values


def class_value(
    label_schema: Mapping[str, Any],
    field: str,
    class_id: int,
) -> Any:
    return label_schema["fields"][field][
        class_id
    ]


def decode_batch(
    outputs: Mapping[str, Any],
    label_schema: Mapping[str, Any],
) -> list[dict[str, Any] | None]:
    call_count_ids = (
        outputs["call_count_logits"]
        .argmax(dim=-1)
        .detach()
        .cpu()
    )

    field_ids = {
        field: logits.argmax(dim=-1)
        .detach()
        .cpu()
        for field, logits
        in outputs["semantic_logits"].items()
    }

    batch_size = call_count_ids.shape[0]

    predictions: list[
        dict[str, Any] | None
    ] = []

    for batch_index in range(batch_size):
        count_class = int(
            call_count_ids[batch_index].item()
        )

        call_count = int(
            label_schema["call_count_labels"][
                count_class
            ]
        )

        calls = []
        failed = False

        def value(
            field: str,
            position: int,
        ) -> Any:
            class_id = int(
                field_ids[field][
                    batch_index,
                    position,
                ].item()
            )

            return class_value(
                label_schema,
                field,
                class_id,
            )

        for position in range(call_count):
            function = value(
                "function",
                position,
            )

            labels: dict[str, Any] = {
                "function": function,
            }

            if function in FUNCTIONS_WITH_OPTIONAL_ZONE:
                labels["zone"] = value(
                    "zone",
                    position,
                )

            if function == "set_hvac_power":
                labels["state"] = value(
                    "state",
                    position,
                )

            elif function in {
                "set_hvac_temperature",
                "set_hvac_fan_speed",
                "set_window_position",
                "set_sunroof_position",
                "set_sunshade_position",
            }:
                if function == "set_window_position":
                    labels["zone"] = value(
                        "zone",
                        position,
                    )

                kind = value(
                    "target_kind",
                    position,
                )

                labels["target_kind"] = kind

                if kind in {
                    "absolute",
                    "extreme",
                    "named",
                }:
                    labels["target_value"] = value(
                        "target_value",
                        position,
                    )

                elif kind == "relative":
                    labels[
                        "target_direction"
                    ] = value(
                        "target_direction",
                        position,
                    )

                    labels[
                        "target_magnitude"
                    ] = value(
                        "target_magnitude",
                        position,
                    )

            elif function == "set_seat_climate":
                labels["zone"] = value(
                    "zone",
                    position,
                )

                labels["feature"] = value(
                    "feature",
                    position,
                )

                labels["setting_value"] = value(
                    "setting_value",
                    position,
                )

            elif function == "set_seat_massage":
                labels["zone"] = value(
                    "zone",
                    position,
                )

                labels["setting_value"] = value(
                    "setting_value",
                    position,
                )

            try:
                calls.append(
                    assemble_call(labels)
                )
            except (LabelError, KeyError):
                failed = True
                break

        if failed:
            predictions.append(None)
        else:
            predictions.append(
                {"calls": calls}
            )

    return predictions


def payload_exact(
    prediction: dict[str, Any] | None,
    gold_calls: list[dict[str, Any]],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    if prediction is None:
        return False

    gold_payload = {
        "calls": gold_calls,
    }

    try:
        pred_json = canonical.canonical_json(
            prediction,
            schema=schema,
            registry=registry,
        )

        gold_json = canonical.canonical_json(
            gold_payload,
            schema=schema,
            registry=registry,
        )

    except Exception:
        return False

    return pred_json == gold_json


@torch.no_grad()
def evaluate(
    model: MMBertSemanticParser,
    loader: DataLoader,
    gold_by_id: Mapping[str, Any],
    label_schema: Mapping[str, Any],
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    batches = 0

    correct = 0
    total = 0

    for batch in loader:
        example_ids = list(
            batch["example_id"]
        )

        batch = move_batch(
            batch,
            device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
            )

            loss, _ = compute_loss(
                outputs,
                batch,
            )

        total_loss += float(
            loss.detach().item()
        )
        batches += 1

        predictions = constrained_decode_batch(
            outputs,
            label_schema,
        )

        for example_id, prediction in zip(
            example_ids,
            predictions,
        ):
            gold = gold_by_id[example_id]

            if payload_exact(
                prediction,
                gold["canonical_calls"],
                schema=schema,
                registry=registry,
            ):
                correct += 1

            total += 1

    return (
        total_loss / max(batches, 1),
        correct / max(total, 1),
    )


@torch.no_grad()
def predict_records(
    model: MMBertSemanticParser,
    loader: DataLoader,
    label_schema: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()

    records = []

    for batch in loader:
        example_ids = list(
            batch["example_id"]
        )

        batch = move_batch(
            batch,
            device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
            )

        predictions = constrained_decode_batch(
            outputs,
            label_schema,
        )

        for example_id, prediction in zip(
            example_ids,
            predictions,
        ):
            records.append(
                {
                    "example_id": example_id,
                    "prediction": prediction,
                }
            )

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = max(
        1,
        int(total_steps * warmup_ratio),
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(
                warmup_steps
            )

        remaining = total_steps - step
        decay_steps = max(
            1,
            total_steps - warmup_steps,
        )

        return max(
            0.0,
            remaining / decay_steps,
        )

    return LambdaLR(
        optimizer,
        lr_lambda,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this baseline run"
        )

    seed_everything(args.seed)

    torch.set_float32_matmul_precision(
        "high"
    )

    device = torch.device("cuda")

    label_schema = load_label_schema()

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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    train_rows = load_rows("train")
    validation_rows = load_rows(
        "validation"
    )
    test_rows = load_rows("test")

    train_dataset = MMBertVehicleDataset(
        train_rows,
        tokenizer=tokenizer,
        label_schema=label_schema,
    )

    validation_dataset = MMBertVehicleDataset(
        validation_rows,
        tokenizer=tokenizer,
        label_schema=label_schema,
    )

    test_dataset = MMBertVehicleDataset(
        test_rows,
        tokenizer=tokenizer,
        label_schema=label_schema,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = MMBertSemanticParser(
        label_schema
    ).to(device)

    head_parameters = [
        *model.call_position_embedding.parameters(),
        *model.call_count_head.parameters(),
        *model.semantic_heads.parameters(),
    ]

    optimizer = AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": args.backbone_lr,
            },
            {
                "params": head_parameters,
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = (
        len(train_loader)
        * args.epochs
    )

    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    gold_validation = {
        row["example_id"]: row
        for row in validation_rows
    }

    best_val_em = -1.0
    best_epoch = -1
    history = []

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_step = 0

    print("=== training configuration ===")
    print("model:", MODEL_ID)
    print("revision:", MODEL_REVISION)
    print("train:", len(train_dataset))
    print(
        "validation:",
        len(validation_dataset),
    )
    print("test:", len(test_dataset))
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print(
        "backbone_lr:",
        args.backbone_lr,
    )
    print("head_lr:", args.head_lr)
    print("device:", device)
    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0

        for batch_index, batch in enumerate(
            train_loader,
            1,
        ):
            batch = move_batch(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch[
                        "attention_mask"
                    ],
                )

                loss, _ = compute_loss(
                    outputs,
                    batch,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at "
                    f"epoch={epoch}, "
                    f"batch={batch_index}: "
                    f"{loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            running_loss += float(
                loss.detach().item()
            )

            global_step += 1

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_loss, val_em = evaluate(
            model,
            validation_loader,
            gold_validation,
            label_schema,
            schema,
            registry,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_ordered_full_em": val_em,
            }
        )

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_full_em={val_em:.4f}"
        )

        if val_em > best_val_em:
            best_val_em = val_em
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch": epoch,
                    "validation_ordered_full_em":
                        val_em,
                    "model_id": MODEL_ID,
                    "model_revision":
                        MODEL_REVISION,
                    "label_schema_version":
                        label_schema["version"],
                    "args": vars(args),
                },
                CHECKPOINT_PATH,
            )

            print(
                "  saved best checkpoint"
            )

    print()
    print("=== best checkpoint ===")
    print("epoch:", best_epoch)
    print(
        "validation Ordered Full EM:",
        f"{best_val_em:.4f}",
    )
    print(
        "checkpoint:",
        CHECKPOINT_PATH,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_predictions = predict_records(
        model,
        validation_loader,
        label_schema,
        device,
    )

    test_predictions = predict_records(
        model,
        test_loader,
        label_schema,
        device,
    )

    write_jsonl(
        RESULT_ROOT
        / "validation_predictions.jsonl",
        validation_predictions,
    )

    write_jsonl(
        RESULT_ROOT
        / "test_predictions.jsonl",
        test_predictions,
    )

    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "best_epoch": best_epoch,
        "best_validation_ordered_full_em":
            best_val_em,
        "training_args": vars(args),
        "history": history,
    }

    (
        RESULT_ROOT
        / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "validation predictions:",
        RESULT_ROOT
        / "validation_predictions.jsonl",
    )

    print(
        "test predictions:",
        RESULT_ROOT
        / "test_predictions.jsonl",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
