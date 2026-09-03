import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
import audit_macslu as audit
import build_macslu_inventory as inventory


def semantic_unit(domain, *slot_names):
    return {domain: [{"name": name, "value": name} for name in slot_names]}


def example(sample_id, query, split_sens, semantics):
    return {
        "id": sample_id,
        "query": query,
        "split_sens": split_sens,
        "semantics": semantics,
    }


class InventoryPolicyTest(unittest.TestCase):
    def fixture(self):
        six_vehicle_intents = {
            f"intent-{index}": semantic_unit("vehicle", "intent")
            for index in range(1, 7)
        }
        return {
            "train": [
                example(
                    "t-overlap",
                    "Open, Window!",
                    ["Open, Window!"],
                    {"intent-1": semantic_unit("vehicle", "intent")},
                ),
                example(
                    "t-manual",
                    "mixed",
                    ["mixed", "extra"],
                    {
                        "intent-1": {
                            **semantic_unit("vehicle", "unexpected"),
                            **semantic_unit("music", "intent"),
                        }
                    },
                ),
                example("t-max", "six", ["six"], six_vehicle_intents),
                example(
                    "nonvehicle",
                    "music",
                    ["music"],
                    {"intent-1": semantic_unit("music", "intent")},
                ),
                example("unannotated", "none", [], {}),
            ],
            "validation": [
                example(
                    "v-exact",
                    "Open, Window!",
                    ["Open, Window!"],
                    {"intent-1": semantic_unit("vehicle", "intent")},
                ),
                example(
                    "v-candidate",
                    "unique",
                    ["unique"],
                    {"intent-1": semantic_unit("vehicle", "intent")},
                ),
                example(
                    "v-eval-pair",
                    "Eval Pair",
                    ["Eval Pair"],
                    {"intent-1": semantic_unit("vehicle", "intent")},
                ),
            ],
            "test": [
                example(
                    "x-normalized-manual",
                    " openwindow ",
                    ["openwindow", "extra"],
                    {"intent-1": semantic_unit("vehicle", "intent")},
                ),
                example(
                    "x-eval-pair",
                    "Eval Pair",
                    ["Eval Pair"],
                    {"intent-1": semantic_unit("vehicle", "intent")},
                ),
            ],
        }

    def test_inventory_scope_flags_and_status_priority(self):
        rows, summary = inventory.build_inventory(
            self.fixture(),
            revision="rev",
            vehicle_domain="vehicle",
            max_intents=5,
            allowed_vehicle_slots=("intent",),
        )
        by_id = {row["id"]: row for row in rows}

        self.assertEqual(len(rows), 10)
        self.assertEqual(
            by_id["v-exact"]["example_id"], "macslu:rev:validation:v-exact"
        )
        self.assertEqual(by_id["t-overlap"]["initial_status"], "candidate")
        self.assertEqual(by_id["v-exact"]["initial_status"], "quarantined")
        self.assertEqual(
            by_id["v-exact"]["issue_codes"],
            [
                "cross_split_query_overlap_exact",
                "cross_split_query_overlap_normalized",
                "cross_split_query_overlap_review_normalized",
            ],
        )
        self.assertEqual(
            by_id["x-normalized-manual"]["initial_status"], "manual_review"
        )
        self.assertIn(
            "split_sens_gt_semantic_units",
            by_id["x-normalized-manual"]["issue_codes"],
        )
        self.assertIn(
            "cross_split_query_overlap_review_normalized",
            by_id["x-normalized-manual"]["issue_codes"],
        )
        self.assertEqual(by_id["t-manual"]["initial_status"], "manual_review")
        self.assertEqual(by_id["t-max"]["initial_status"], "manual_review")
        self.assertIn("mixed_domain_example", by_id["t-manual"]["issue_codes"])
        self.assertIn("multi_domain_per_intent", by_id["t-manual"]["issue_codes"])
        self.assertIn("unexpected_vehicle_slot", by_id["t-manual"]["issue_codes"])
        self.assertIn("max_intent_claim_violation", by_id["t-max"]["issue_codes"])
        self.assertEqual(by_id["v-candidate"]["initial_status"], "candidate")
        self.assertEqual(by_id["v-eval-pair"]["initial_status"], "candidate")
        self.assertEqual(by_id["v-eval-pair"]["issue_codes"], [])
        self.assertEqual(by_id["x-eval-pair"]["initial_status"], "quarantined")
        self.assertEqual(
            by_id["x-eval-pair"]["issue_codes"],
            [
                "cross_split_query_overlap_exact",
                "cross_split_query_overlap_normalized",
                "cross_split_query_overlap_review_normalized",
            ],
        )
        self.assertEqual(by_id["nonvehicle"]["initial_status"], "excluded")
        self.assertEqual(
            by_id["nonvehicle"]["issue_codes"],
            ["no_vehicle_target"],
        )
        self.assertEqual(by_id["unannotated"]["initial_status"], "excluded")
        self.assertEqual(
            by_id["unannotated"]["issue_codes"],
            ["no_vehicle_target", "unannotated_source"],
        )
        self.assertEqual(
            by_id["v-candidate"]["source_group_id"],
            by_id["v-candidate"]["example_id"],
        )
        self.assertEqual(by_id["v-candidate"]["final_status"], "")

        self.assertEqual(summary["counts"]["source_rows"], 10)
        self.assertEqual(summary["counts"]["inventory_rows"], 10)
        self.assertEqual(summary["counts"]["vehicle_rows"], 8)
        self.assertEqual(summary["counts"]["excluded_no_vehicle_rows"], 2)
        self.assertEqual(summary["counts"]["excluded_unannotated_rows"], 1)
        self.assertEqual(
            summary["counts"]["excluded_annotated_non_vehicle_rows"], 1
        )
        self.assertEqual(
            summary["status_counts"],
            {
                "candidate": 3,
                "excluded": 2,
                "manual_review": 3,
                "quarantined": 2,
            },
        )


class InventoryIoTest(unittest.TestCase):
    def test_manifest_cli_is_offline_deterministic_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            rows = [
                example(
                    "1",
                    "open",
                    ["open"],
                    {"intent-1": semantic_unit("vehicle", "intent")},
                )
            ]
            source_path = source_root / "train.jsonl"
            source_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest = {
                "dataset": {
                    "id": "fixture",
                    "revision": "fixed",
                    "vehicle_domain": "vehicle",
                    "source_files": {
                        "train": {
                            "relative_path": "train.jsonl",
                            "records": 1,
                            "sha256": audit.sha256_file(source_path),
                        }
                    },
                },
                "paper_claims": {"maximum_intents_per_utterance": 5},
                "known_vehicle_slot_names": ["intent"],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            first = root / "first"
            second = root / "second"
            arguments = [
                "--manifest",
                str(manifest_path),
                "--source-root",
                str(source_root),
                "--output-dir",
            ]

            self.assertEqual(inventory.main(arguments + [str(first)]), 0)
            self.assertEqual(inventory.main(arguments + [str(second)]), 0)
            self.assertEqual(
                (first / "inventory.csv").read_bytes(),
                (second / "inventory.csv").read_bytes(),
            )
            self.assertEqual(
                (first / "summary.json").read_bytes(),
                (second / "summary.json").read_bytes(),
            )
            with (first / "inventory.csv").open(encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written[0]["example_id"], "macslu:fixed:train:1")
            self.assertEqual(
                written[0]["source_group_id"],
                "macslu:fixed:train:1",
            )
            self.assertEqual(written[0]["final_status"], "")
            saved_summary = json.loads((first / "summary.json").read_text())
            self.assertEqual(saved_summary["source_verification"]["status"], "verified")
            self.assertNotIn(str(root), json.dumps(saved_summary))
            with self.assertRaises(FileExistsError):
                inventory.main(arguments + [str(first)])


if __name__ == "__main__":
    unittest.main()
