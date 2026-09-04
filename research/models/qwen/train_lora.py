#!/usr/bin/env python3
"""LoRA fine-tuning for Qwen3-4B vehicle function calling."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_prompt import (
    MODEL_ID,
    MODEL_REVISION,
    build_messages,
)
from run_zero_shot import parse_json_output


HERE = Path(__file__).resolve().parent
RESEARCH_ROOT = HERE.parents[1]

DATA_ROOT = (
    RESEARCH_ROOT
    / "data"
    / "processed"
    / "macslu_korean_v0.2"
)

ADAPTER_DIR = (
    RESEARCH_ROOT
    / "checkpoints"
    / "qwen_lora_v0.1"
)

RESULT_ROOT = (
    RESEARCH_ROOT
    / "results"
    / "qwen_lora_v0.1"
)

PREPROCESSING_ROOT = (
    RESEARCH_ROOT
    / "preprocessing"
)

if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PREPROCESSING_ROOT),
    )

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

IGNORE_INDEX = -100


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(
    split: str,
) -> list[dict[str, Any]]:
    path = DATA_ROOT / f"{split}.jsonl"

    with path.open(
        encoding="utf-8",
    ) as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def gold_json(
    row: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "calls": row[
                "canonical_calls"
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class QwenSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.rows[index]

        prompt_messages = build_messages(
            row["utterance_ko"]
        )

        full_messages = [
            *prompt_messages,
            {
                "role": "assistant",
                "content": gold_json(row),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )["input_ids"]

        full = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )["input_ids"]
        
        if len(full) > self.max_length:
            raise RuntimeError(
                f"{row['example_id']}: "
                f"sequence length {len(full)} "
                f"> max_length {self.max_length}"
            )

        input_ids = torch.tensor(
            full,
            dtype=torch.long,
        )

        labels = input_ids.clone()

        prompt_length = min(
            len(prompt),
            len(full),
        )

        labels[
            :prompt_length
        ] = IGNORE_INDEX

        attention_mask = torch.ones_like(
            input_ids
        )

        return {
            "example_id":
                row["example_id"],
            "input_ids":
                input_ids,
            "attention_mask":
                attention_mask,
            "labels":
                labels,
        }


class Collator:
    def __init__(
        self,
        pad_token_id: int,
    ) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        examples: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        max_len = max(
            len(x["input_ids"])
            for x in examples
        )

        batch_size = len(examples)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )

        labels = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=torch.long,
        )

        example_ids = []

        for i, example in enumerate(
            examples
        ):
            length = len(
                example["input_ids"]
            )

            input_ids[
                i,
                :length,
            ] = example[
                "input_ids"
            ]

            attention_mask[
                i,
                :length,
            ] = 1

            labels[
                i,
                :length,
            ] = example[
                "labels"
            ]

            example_ids.append(
                example[
                    "example_id"
                ]
            )

        return {
            "example_id":
                example_ids,
            "input_ids":
                input_ids,
            "attention_mask":
                attention_mask,
            "labels":
                labels,
        }


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: (
            value.to(
                device,
                non_blocking=True,
            )
            if isinstance(
                value,
                torch.Tensor,
            )
            else value
        )
        for key, value
        in batch.items()
    }


def payload_exact(
    prediction:
        dict[str, Any] | None,
    gold_calls:
        list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    registry: dict[str, Any],
) -> bool:
    if prediction is None:
        return False

    try:
        pred = canonical.canonical_json(
            prediction,
            schema=schema,
            registry=registry,
        )

        gold = canonical.canonical_json(
            {
                "calls":
                    gold_calls
            },
            schema=schema,
            registry=registry,
        )

    except Exception:
        return False

    return pred == gold


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    rows,
    *,
    schema,
    registry,
    device,
    max_new_tokens,
) -> tuple[
    float,
    list[dict[str, Any]],
]:
    model.eval()

    # Generation needs KV cache.
    model.config.use_cache = True

    correct = 0
    predictions = []

    for index, row in enumerate(
        rows,
        1,
    ):
        messages = build_messages(
            row["utterance_ko"]
        )

        inputs = (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        generated = model.generate(
            **inputs,
            max_new_tokens=
                max_new_tokens,
            do_sample=False,
        )

        new_tokens = generated[0][
            inputs[
                "input_ids"
            ].shape[-1]:
        ]

        raw = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        prediction = (
            parse_json_output(raw)
        )

        if payload_exact(
            prediction,
            row[
                "canonical_calls"
            ],
            schema=schema,
            registry=registry,
        ):
            correct += 1

        predictions.append(
            {
                "example_id":
                    row[
                        "example_id"
                    ],
                "prediction":
                    prediction,
                "raw_generation":
                    raw,
            }
        )

        print(
            f"  val "
            f"[{index:02d}/"
            f"{len(rows)}]"
        )

    model.config.use_cache = False

    return (
        correct / len(rows),
        predictions,
    )


def write_predictions(
    path: Path,
    predictions:
        list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for item in predictions:
            record = {
                "example_id":
                    item[
                        "example_id"
                    ],
                "prediction":
                    item[
                        "prediction"
                    ],
            }

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def build_scheduler(
    optimizer,
    *,
    total_steps,
    warmup_ratio,
):
    warmup_steps = max(
        1,
        int(
            total_steps
            * warmup_ratio
        ),
    )

    def lr_lambda(
        step: int,
    ) -> float:
        if step < warmup_steps:
            return (
                (step + 1)
                / warmup_steps
            )

        remaining = (
            total_steps - step
        )

        decay_steps = max(
            1,
            total_steps
            - warmup_steps,
        )

        return max(
            0.0,
            remaining
            / decay_steps,
        )

    return LambdaLR(
        optimizer,
        lr_lambda,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID,
            revision=
                MODEL_REVISION,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    train_rows = load_rows(
        "train"
    )

    validation_rows = load_rows(
        "validation"
    )

    train_dataset = (
        QwenSFTDataset(
            train_rows,
            tokenizer,
            max_length=
                args.max_length,
        )
    )

    # Fail early if a sample exceeds
    # max_length.
    max_observed = 0

    for i in range(
        len(train_dataset)
    ):
        item = train_dataset[i]
        max_observed = max(
            max_observed,
            len(
                item[
                    "input_ids"
                ]
            ),
        )

    print(
        "max train tokens:",
        max_observed,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=Collator(
            tokenizer.pad_token_id
        ),
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            revision=
                MODEL_REVISION,
            dtype=
                torch.bfloat16,
        )
    )

    base_model.config.use_cache = (
        False
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(
        base_model,
        lora_config,
    )

    model = model.to(device)

    model.print_trainable_parameters()

    optimizer = AdamW(
        [
            p
            for p in
            model.parameters()
            if p.requires_grad
        ],
        lr=args.lr,
        weight_decay=0.01,
    )

    update_steps_per_epoch = (
        math.ceil(
            len(loader)
            / args.grad_accum
        )
    )

    total_steps = (
        update_steps_per_epoch
        * args.epochs
    )

    scheduler = build_scheduler(
        optimizer,
        total_steps=
            total_steps,
        warmup_ratio=0.1,
    )

    schema = (
        canonical.load_json_object(
            SCHEMA_PATH
        )
    )

    registry = (
        canonical.load_json_object(
            REGISTRY_PATH
        )
    )

    canonical.validate_registry(
        schema,
        registry,
    )

    best_em = -1.0
    best_epoch = -1

    history = []

    optimizer.zero_grad(
        set_to_none=True
    )

    print()
    print(
        "=== training ==="
    )
    print(
        "train:",
        len(train_rows),
    )
    print(
        "validation:",
        len(validation_rows),
    )
    print(
        "epochs:",
        args.epochs,
    )
    print(
        "batch_size:",
        args.batch_size,
    )
    print(
        "grad_accum:",
        args.grad_accum,
    )
    print(
        "effective batch:",
        args.batch_size
        * args.grad_accum,
    )
    print(
        "lr:",
        args.lr,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()
        model.config.use_cache = (
            False
        )

        running_loss = 0.0
        batch_count = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        for batch_index, batch in enumerate(
            loader,
            1,
        ):
            batch = move_batch(
                batch,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    input_ids=
                        batch[
                            "input_ids"
                        ],
                    attention_mask=
                        batch[
                            "attention_mask"
                        ],
                    labels=
                        batch[
                            "labels"
                        ],
                )

                loss = (
                    outputs.loss
                    / args.grad_accum
                )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "non-finite loss"
                )

            loss.backward()

            running_loss += (
                float(
                    outputs.loss
                    .detach()
                    .item()
                )
            )

            batch_count += 1

            should_step = (
                batch_index
                % args.grad_accum
                == 0
                or batch_index
                == len(loader)
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

        train_loss = (
            running_loss
            / batch_count
        )

        print(
            f"\nepoch {epoch:02d} "
            f"train_loss="
            f"{train_loss:.4f}"
        )

        val_em, val_predictions = (
            evaluate_generation(
                model,
                tokenizer,
                validation_rows,
                schema=schema,
                registry=registry,
                device=device,
                max_new_tokens=
                    args.max_new_tokens,
            )
        )

        print(
            f"epoch {epoch:02d} "
            f"val_full_em="
            f"{val_em:.4f}"
        )

        history.append(
            {
                "epoch":
                    epoch,
                "train_loss":
                    train_loss,
                "validation_ordered_full_em":
                    val_em,
            }
        )

        if val_em > best_em:
            best_em = val_em
            best_epoch = epoch

            if ADAPTER_DIR.exists():
                import shutil
                shutil.rmtree(
                    ADAPTER_DIR
                )

            model.save_pretrained(
                ADAPTER_DIR
            )

            tokenizer.save_pretrained(
                ADAPTER_DIR
            )

            RESULT_ROOT.mkdir(
                parents=True,
                exist_ok=True,
            )

            write_predictions(
                RESULT_ROOT
                / "validation_predictions.jsonl",
                val_predictions,
            )

            print(
                "saved best adapter"
            )

        if best_em == 1.0:
            print(
                "validation EM reached 1.0; "
                "stopping early"
            )
            break

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model_id":
            MODEL_ID,
        "model_revision":
            MODEL_REVISION,
        "adapter":
            "LoRA",
        "lora_r":
            16,
        "lora_alpha":
            32,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        "best_epoch":
            best_epoch,
        "best_validation_ordered_full_em":
            best_em,
        "training_args":
            vars(args),
        "history":
            history,
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
        "=== best ==="
    )
    print(
        "epoch:",
        best_epoch,
    )
    print(
        "validation Ordered Full EM:",
        f"{best_em:.4f}",
    )
    print(
        "adapter:",
        ADAPTER_DIR,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
