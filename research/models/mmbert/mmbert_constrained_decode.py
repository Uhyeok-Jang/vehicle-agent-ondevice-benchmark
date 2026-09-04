"""Schema-constrained decoding for the frozen mmBERT benchmark."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from mmbert_labels import (
    NONE_ZONE,
    LabelError,
    assemble_call,
)


HVAC_ZONES = {
    NONE_ZONE,
    "driver",
    "front_passenger",
    "rear",
    "all",
}

WINDOW_ZONES = {
    "driver",
    "front_passenger",
    "rear_left",
    "rear_right",
    "front_row",
    "rear_row",
    "left_side",
    "right_side",
    "all",
}

SEAT_ZONES = {
    "driver",
    "front_passenger",
    "rear_left",
    "rear_right",
    "rear_row",
    "all",
}


def _choose(
    logits: torch.Tensor,
    values: list[Any],
    predicate=None,
) -> Any:
    if predicate is None:
        allowed = list(range(len(values)))
    else:
        allowed = [
            i
            for i, value in enumerate(values)
            if predicate(value)
        ]

    if not allowed:
        raise LabelError(
            "no label satisfies decoding constraint"
        )

    allowed_tensor = torch.tensor(
        allowed,
        device=logits.device,
        dtype=torch.long,
    )

    selected = logits.index_select(
        0,
        allowed_tensor,
    )

    local_index = int(
        selected.argmax().item()
    )

    return values[allowed[local_index]]


def decode_batch(
    outputs: Mapping[str, Any],
    label_schema: Mapping[str, Any],
) -> list[dict[str, Any] | None]:

    call_count_ids = (
        outputs["call_count_logits"]
        .argmax(dim=-1)
    )

    semantic_logits = outputs[
        "semantic_logits"
    ]

    batch_size = int(
        call_count_ids.shape[0]
    )

    predictions = []

    for b in range(batch_size):
        count_class = int(
            call_count_ids[b].item()
        )

        call_count = int(
            label_schema[
                "call_count_labels"
            ][count_class]
        )

        calls = []
        failed = False

        def choose(
            field: str,
            position: int,
            predicate=None,
        ) -> Any:
            return _choose(
                semantic_logits[field][
                    b,
                    position,
                ],
                label_schema[
                    "fields"
                ][field],
                predicate,
            )

        for position in range(call_count):
            try:
                function = choose(
                    "function",
                    position,
                )

                labels: dict[str, Any] = {
                    "function": function,
                }

                if function == "set_hvac_power":
                    labels["state"] = choose(
                        "state",
                        position,
                    )

                    labels["zone"] = choose(
                        "zone",
                        position,
                        lambda x: x in HVAC_ZONES,
                    )

                elif function in {
                    "set_hvac_temperature",
                    "set_hvac_fan_speed",
                }:
                    labels["zone"] = choose(
                        "zone",
                        position,
                        lambda x: x in HVAC_ZONES,
                    )

                    kind = choose(
                        "target_kind",
                        position,
                        lambda x: x in {
                            "absolute",
                            "relative",
                            "extreme",
                        },
                    )

                    labels["target_kind"] = kind

                    if kind == "absolute":
                        if (
                            function
                            == "set_hvac_temperature"
                        ):
                            labels[
                                "target_value"
                            ] = choose(
                                "target_value",
                                position,
                                lambda x: (
                                    isinstance(
                                        x,
                                        (int, float),
                                    )
                                    and not isinstance(
                                        x,
                                        bool,
                                    )
                                    and 16 <= x <= 32
                                ),
                            )

                        else:
                            labels[
                                "target_value"
                            ] = choose(
                                "target_value",
                                position,
                                lambda x: (
                                    type(x) is int
                                    and 1 <= x <= 8
                                ),
                            )

                    elif kind == "extreme":
                        labels[
                            "target_value"
                        ] = choose(
                            "target_value",
                            position,
                            lambda x: x in {
                                "min",
                                "max",
                            },
                        )

                    else:
                        labels[
                            "target_direction"
                        ] = choose(
                            "target_direction",
                            position,
                            lambda x: x in {
                                "increase",
                                "decrease",
                            },
                        )

                        labels[
                            "target_magnitude"
                        ] = choose(
                            "target_magnitude",
                            position,
                        )

                elif function == "set_window_position":
                    labels["zone"] = choose(
                        "zone",
                        position,
                        lambda x: x in WINDOW_ZONES,
                    )

                    # Frozen 597-example benchmark
                    # contains only named aperture targets.
                    labels["target_kind"] = "named"

                    labels["target_value"] = choose(
                        "target_value",
                        position,
                        lambda x: x in {
                            "open",
                            "closed",
                        },
                    )

                elif function in {
                    "set_sunroof_position",
                    "set_sunshade_position",
                }:
                    labels["target_kind"] = "named"

                    labels["target_value"] = choose(
                        "target_value",
                        position,
                        lambda x: x in {
                            "open",
                            "closed",
                        },
                    )

                elif function == "set_seat_climate":
                    labels["zone"] = choose(
                        "zone",
                        position,
                        lambda x: x in SEAT_ZONES,
                    )

                    labels["feature"] = choose(
                        "feature",
                        position,
                    )

                    labels["setting_value"] = choose(
                        "setting_value",
                        position,
                    )

                elif function == "set_seat_massage":
                    labels["zone"] = choose(
                        "zone",
                        position,
                        lambda x: x in SEAT_ZONES,
                    )

                    labels["setting_value"] = choose(
                        "setting_value",
                        position,
                    )

                else:
                    raise LabelError(
                        f"unknown function: {function}"
                    )

                calls.append(
                    assemble_call(labels)
                )

            except (
                LabelError,
                KeyError,
                ValueError,
            ):
                failed = True
                break

        if failed:
            predictions.append(None)
        else:
            predictions.append(
                {"calls": calls}
            )

    return predictions
