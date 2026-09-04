#!/usr/bin/env python3
"""Run Qwen zero-shot vehicle function calling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from qwen_prompt import (
    MODEL_ID,
    MODEL_REVISION,
    build_messages,
)


DATA_ROOT = Path(
    "research/data/processed/macslu_korean_v0.2"
)

DEFAULT_OUTPUT = Path(
    "research/results/qwen_zero_shot_v0.1"
)


def load_rows(split: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / f"{split}.jsonl"

    with path.open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def parse_json_output(
    text: str,
) -> dict[str, Any] | None:
    text = text.strip()

    # Defensive handling in case the model emits code fences.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # First try strict whole-output JSON.
    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Minimal recovery: take outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            value = json.loads(
                text[start:end + 1]
            )

            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    return None


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    rows = load_rows(args.split)

    output_root = DEFAULT_OUTPUT
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_root
        / f"{args.split}_predictions.jsonl"
    )

    raw_path = (
        output_root
        / f"{args.split}_raw_generations.jsonl"
    )

    print("=== configuration ===")
    print("model:", MODEL_ID)
    print("revision:", MODEL_REVISION)
    print("split:", args.split)
    print("examples:", len(rows))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
    )

    model = model.to("cuda")
    model.eval()

    parsed_count = 0

    with (
        output_path.open(
            "w",
            encoding="utf-8",
        ) as prediction_file,
        raw_path.open(
            "w",
            encoding="utf-8",
        ) as raw_file,
    ):
        for index, row in enumerate(
            rows,
            1,
        ):
            messages = build_messages(
                row["utterance_ko"]
            )

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = {
                key: value.to("cuda")
                for key, value in inputs.items()
            }

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            new_tokens = generated[0][
                inputs["input_ids"].shape[-1]:
            ]

            raw_text = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            prediction = parse_json_output(
                raw_text
            )

            if prediction is not None:
                parsed_count += 1

            prediction_record = {
                "example_id": row["example_id"],
                "prediction": prediction,
            }

            raw_record = {
                "example_id": row["example_id"],
                "utterance_ko": row["utterance_ko"],
                "raw_generation": raw_text,
                "parsed": prediction is not None,
            }

            prediction_file.write(
                json.dumps(
                    prediction_record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

            raw_file.write(
                json.dumps(
                    raw_record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

            print(
                f"[{index:02d}/{len(rows)}] "
                f"parsed={prediction is not None} "
                f"{row['utterance_ko']}"
            )

    print()
    print(
        "JSON parsed:",
        f"{parsed_count}/{len(rows)}",
    )
    print("predictions:", output_path)
    print("raw generations:", raw_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
