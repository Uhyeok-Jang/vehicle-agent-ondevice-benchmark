import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "Gatsby1984/MAC_SLU"
VEHICLE_DOMAIN = "车载控制"

OUTPUT_DIR = Path(
    "research/analysis/dataset_statistics/macslu_vehicle"
)


def extract_active_semantics(example):
    active = []

    semantics = example.get("semantics")

    if not semantics:
        return active

    for intent_index, intent_data in semantics.items():
        if not intent_data:
            continue

        for domain, slots in intent_data.items():
            if not slots:
                continue

            active.append(
                {
                    "intent_index": intent_index,
                    "domain": domain,
                    "slots": slots,
                }
            )

    return active


def slots_to_multidict(slots):
    result = defaultdict(list)

    for slot in slots:
        name = slot.get("name")
        value = slot.get("value")

        if name is not None:
            result[name].append(value)

    return dict(result)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME)

    slot_names = Counter()
    slot_values = defaultdict(Counter)

    intent_values = Counter()
    slot_patterns = Counter()

    object_action = Counter()
    object_function = Counter()
    adjust_value = Counter()

    position_values = Counter()
    integrity_counter = Counter()

    sample_rows = []

    for split_name, split in dataset.items():
        for example in split:
            active = extract_active_semantics(example)

            vehicle_units = [
                x for x in active
                if x["domain"] == VEHICLE_DOMAIN
            ]

            if not vehicle_units:
                continue

            split_sens = example.get("split_sens") or []

            active_count = len(active)
            split_sens_count = len(split_sens)

            domains = sorted(
                set(x["domain"] for x in active)
            )

            if split_sens_count == active_count:
                integrity = "count_match"
            elif split_sens_count > active_count:
                integrity = "semantics_missing"
            else:
                integrity = "semantics_exceed_split"

            integrity_counter[integrity] += 1

            mixed_domain = len(domains) > 1

            if mixed_domain:
                integrity_counter["mixed_domain"] += 1

            vehicle_serialized = []

            for unit in vehicle_units:
                slots = slots_to_multidict(
                    unit["slots"]
                )

                vehicle_serialized.append(
                    {
                        "intent_index": unit["intent_index"],
                        "slots": slots,
                    }
                )

                names = []

                for slot in unit["slots"]:
                    name = slot.get("name")
                    value = slot.get("value")

                    if name is None:
                        continue

                    names.append(name)
                    slot_names[name] += 1

                    if value is not None:
                        slot_values[name][value] += 1

                    if name == "intent":
                        intent_values[value] += 1

                    if name == "位置":
                        position_values[value] += 1

                slot_patterns[
                    tuple(sorted(names))
                ] += 1

                objects = slots.get("对象", [])
                actions = slots.get("操作", [])
                functions = slots.get("对象功能", [])
                adjusts = slots.get("调节内容", [])
                values = slots.get("value", [])

                for obj in objects:
                    for action in actions:
                        object_action[(obj, action)] += 1

                for obj in objects:
                    for function in functions:
                        object_function[(obj, function)] += 1

                for adjust in adjusts:
                    for value in values:
                        adjust_value[(adjust, value)] += 1

            sample_rows.append(
                {
                    "split": split_name,
                    "id": example["id"],
                    "query": example["query"],
                    "split_sens_count": split_sens_count,
                    "active_semantics_count": active_count,
                    "vehicle_semantics_count": len(vehicle_units),
                    "domains": "|".join(domains),
                    "mixed_domain": mixed_domain,
                    "integrity": integrity,
                    "split_sens": json.dumps(
                        split_sens,
                        ensure_ascii=False,
                    ),
                    "vehicle_semantics": json.dumps(
                        vehicle_serialized,
                        ensure_ascii=False,
                    ),
                }
            )

    ontology = {
        "intent_values":
            dict(intent_values.most_common()),

        "slot_names":
            dict(slot_names.most_common()),

        "slot_values": {
            name: dict(counter.most_common())
            for name, counter
            in slot_values.items()
        },

        "position_values":
            dict(position_values.most_common()),

        "integrity":
            dict(integrity_counter),
    }

    with open(
        OUTPUT_DIR / "ontology.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            ontology,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        OUTPUT_DIR / "vehicle_samples.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=sample_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(sample_rows)

    def save_counter(filename, header, counter):
        with open(
            OUTPUT_DIR / filename,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                list(header) + ["count"]
            )

            for key, count in counter.most_common():
                if not isinstance(key, tuple):
                    key = (key,)

                writer.writerow(
                    list(key) + [count]
                )

    save_counter(
        "object_action.csv",
        ("object", "action"),
        object_action,
    )

    save_counter(
        "object_function.csv",
        ("object", "function"),
        object_function,
    )

    save_counter(
        "adjust_value.csv",
        ("adjust_content", "value"),
        adjust_value,
    )

    save_counter(
        "slot_patterns.csv",
        ("slot_pattern",),
        Counter(
            {
                " | ".join(pattern): count
                for pattern, count
                in slot_patterns.items()
            }
        ),
    )

    print("=" * 70)
    print("MAC-SLU VEHICLE ANALYSIS")
    print("=" * 70)

    print(
        f"Vehicle samples: "
        f"{len(sample_rows)}"
    )

    print("\n[Annotation integrity]")

    for k, v in integrity_counter.most_common():
        print(f"{k:25s}: {v}")

    print("\n[Vehicle intent values]")

    for k, v in intent_values.most_common():
        print(f"{k:20s}: {v}")

    print("\n[Vehicle slot names]")

    for k, v in slot_names.most_common():
        print(f"{k:20s}: {v}")

    print(
        f"\nOutputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
