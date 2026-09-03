from collections import Counter, defaultdict
from datasets import load_dataset


DATASET_NAME = "Gatsby1984/MAC_SLU"
VEHICLE_DOMAIN = "车载控制"


def extract_active_intents(example):
    """
    Returns a list of active semantic intents.

    Each element:
    {
        "intent_index": "意图1",
        "domain": "车载控制",
        "slots": [...]
    }
    """
    active = []

    semantics = example["semantics"]

    if semantics is None:
        return active

    for intent_index, intent_data in semantics.items():

        if intent_data is None:
            continue

        for domain, slots in intent_data.items():

            if slots is None:
                continue

            if len(slots) == 0:
                continue

            active.append({
                "intent_index": intent_index,
                "domain": domain,
                "slots": slots,
            })

    return active


def slots_to_dict(slots):
    result = defaultdict(list)

    for slot in slots:
        name = slot.get("name")
        value = slot.get("value")

        if name is not None:
            result[name].append(value)

    return dict(result)


def inspect_split(split_name, dataset):

    domain_counter = Counter()
    slot_name_counter = Counter()
    intent_value_counter = Counter()

    num_intents_counter = Counter()
    num_slots_counter = Counter()

    vehicle_samples = 0
    vehicle_intents = 0

    unannotated_samples = 0

    vehicle_examples = []
    multi_intent_examples = []

    for example in dataset:

        active_intents = extract_active_intents(example)

        if len(active_intents) == 0:
            unannotated_samples += 1

        num_intents_counter[len(active_intents)] += 1

        total_slots = 0

        has_vehicle_domain = False

        for semantic_intent in active_intents:

            domain = semantic_intent["domain"]
            slots = semantic_intent["slots"]

            domain_counter[domain] += 1

            total_slots += len(slots)

            for slot in slots:

                slot_name = slot.get("name")
                slot_value = slot.get("value")

                if slot_name is not None:
                    slot_name_counter[slot_name] += 1

                if slot_name == "intent":
                    intent_value_counter[
                        f"{domain}::{slot_value}"
                    ] += 1

            if domain == VEHICLE_DOMAIN:

                has_vehicle_domain = True
                vehicle_intents += 1

        num_slots_counter[total_slots] += 1

        if has_vehicle_domain:

            vehicle_samples += 1

            if len(vehicle_examples) < 20:

                vehicle_semantics = []

                for semantic_intent in active_intents:

                    if semantic_intent["domain"] == VEHICLE_DOMAIN:

                        vehicle_semantics.append({
                            "intent_index":
                                semantic_intent["intent_index"],

                            "slots":
                                slots_to_dict(
                                    semantic_intent["slots"]
                                ),
                        })

                vehicle_examples.append({
                    "id": example["id"],
                    "query": example["query"],
                    "split_sens": example["split_sens"],
                    "vehicle_semantics":
                        vehicle_semantics,
                })

        if len(active_intents) >= 2:

            if len(multi_intent_examples) < 20:

                multi_intent_examples.append({
                    "id": example["id"],
                    "query": example["query"],
                    "split_sens": example["split_sens"],
                    "active_intents":
                        active_intents,
                })

    print()
    print("=" * 80)
    print(f"SPLIT: {split_name}")
    print("=" * 80)

    print(f"Total samples          : {len(dataset)}")
    print(f"Unannotated samples    : {unannotated_samples}")
    print(f"Vehicle samples        : {vehicle_samples}")
    print(f"Vehicle semantic units : {vehicle_intents}")

    print()
    print("[Domains]")

    for key, value in domain_counter.most_common():
        print(f"{key:20s} {value}")

    print()
    print("[Intent values]")

    for key, value in intent_value_counter.most_common():
        print(f"{key:40s} {value}")

    print()
    print("[Slot names]")

    for key, value in slot_name_counter.most_common():
        print(f"{key:30s} {value}")

    print()
    print("[# semantic intents per utterance]")

    for key in sorted(num_intents_counter):
        print(f"{key}: {num_intents_counter[key]}")

    print()
    print("[# slots per utterance]")

    for key in sorted(num_slots_counter):
        print(f"{key}: {num_slots_counter[key]}")

    print()
    print("[Vehicle examples]")

    for example in vehicle_examples[:10]:
        print("-" * 80)
        print(example)

    print()
    print("[Multi-intent examples]")

    for example in multi_intent_examples[:10]:
        print("-" * 80)
        print(example)


def main():

    dataset = load_dataset(DATASET_NAME)

    print(dataset)

    for split_name, split in dataset.items():
        inspect_split(split_name, split)


if __name__ == "__main__":
    main()
