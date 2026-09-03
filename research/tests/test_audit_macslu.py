import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
import audit_macslu as audit


def unit(domain, *slot_names):
    return {domain: [{"name": name, "value": name} for name in slot_names]}


class AuditRulesTest(unittest.TestCase):
    def fixture(self):
        six_intents = {
            f"intent-{index}": unit("vehicle", "intent", "操作")
            for index in range(1, 7)
        }
        return {
            "train": [
                {
                    "id": "duplicate",
                    "query": "Hello, World!",
                    "split_sens": ["Hello, World!"],
                    "semantics": {"intent-1": unit("vehicle", "intent", "操作")},
                },
                {
                    "id": "duplicate",
                    "query": "six commands",
                    "split_sens": [str(i) for i in range(6)],
                    "semantics": six_intents,
                },
                {
                    "id": "mixed",
                    "query": "mixed domain",
                    "split_sens": ["mixed domain"],
                    "semantics": {
                        "intent-1": {
                            **unit("vehicle", "intent", "unexpected_slot"),
                            **unit("music", "intent"),
                        }
                    },
                },
            ],
            "validation": [
                {
                    "id": "validation-1",
                    "query": "  hello world  ",
                    "split_sens": [],
                    "semantics": {},
                }
            ],
            "test": [
                {
                    "id": "test-1",
                    "query": "Hello, World!",
                    "split_sens": ["Hello, World!"],
                    "semantics": {"intent-1": unit("vehicle", "intent", "操作")},
                }
            ],
        }

    def test_normalization(self):
        self.assertEqual(
            audit.normalize_query("  ＨＥＬＬＯ，  WORLD!! "),
            "hello,world!!",
        )
        self.assertEqual(
            audit.normalize_query_for_review("  ＨＥＬＬＯ，  WORLD!! "),
            "helloworld",
        )
        self.assertNotEqual(
            audit.normalize_query_for_review("set 2.5"),
            audit.normalize_query_for_review("set 25"),
        )
        self.assertNotEqual(
            audit.normalize_query_for_review("set -10"),
            audit.normalize_query_for_review("set 10"),
        )

    def test_core_audit_rules(self):
        summary, issues = audit.audit_dataset(
            self.fixture(),
            vehicle_domain="vehicle",
            max_intents=5,
            allowed_vehicle_slots={"intent", "操作"},
        )
        codes = [issue["code"] for issue in issues]

        self.assertEqual(summary["totals"]["examples"], 5)
        self.assertEqual(summary["totals"]["active_semantic_units"], 10)
        self.assertEqual(summary["totals"]["vehicle_examples"], 4)
        self.assertEqual(summary["totals"]["vehicle_semantic_units"], 9)
        self.assertEqual(summary["totals"]["count_mismatches"], 1)
        self.assertEqual(summary["totals"]["max_intent_claim_violations"], 1)
        self.assertEqual(summary["totals"]["multi_domain_intents"], 1)
        self.assertEqual(summary["totals"]["mixed_domain_examples"], 1)
        self.assertEqual(summary["totals"]["duplicate_id_values"], 1)
        self.assertEqual(summary["totals"]["cross_split_query_overlap_exact_groups"], 1)
        self.assertEqual(
            summary["totals"]["cross_split_query_overlap_normalized_groups"], 1
        )
        self.assertEqual(
            summary["totals"][
                "cross_split_query_overlap_review_normalized_groups"
            ],
            1,
        )
        self.assertIn("duplicate_id_within_split", codes)
        self.assertIn("multi_domain_per_intent", codes)
        self.assertIn("mixed_domain_example", codes)
        self.assertIn("max_intent_claim_violation", codes)
        self.assertIn("unexpected_vehicle_slot", codes)
        self.assertIn("cross_split_query_overlap_exact", codes)
        self.assertIn("cross_split_query_overlap_normalized", codes)
        self.assertIn("cross_split_query_overlap_review_normalized", codes)
        self.assertEqual(
            summary["issue_counts_by_code"]["split_sens_lt_semantic_units"], 1
        )

    def test_composition_overlap_is_a_separate_diagnostic(self):
        splits = {
            "train": [
                {
                    "id": "t",
                    "query": "t",
                    "split_sens": ["Open, window", "fan"],
                    "semantics": {"intent-1": unit("vehicle", "intent")},
                },
                {
                    "id": "nonvehicle",
                    "query": "nonvehicle",
                    "split_sens": ["should not count"],
                    "semantics": {"intent-1": unit("music", "intent")},
                },
            ],
            "validation": [
                {
                    "id": "v1",
                    "query": "v1",
                    "split_sens": ["open window", "new"],
                    "semantics": {"intent-1": unit("vehicle", "intent")},
                },
                {
                    "id": "v2",
                    "query": "v2",
                    "split_sens": ["FAN"],
                    "semantics": {"intent-1": unit("vehicle", "intent")},
                },
                {
                    "id": "v3",
                    "query": "v3",
                    "split_sens": ["!!!"],
                    "semantics": {"intent-1": unit("vehicle", "intent")},
                },
                {
                    "id": "v4",
                    "query": "v4",
                    "split_sens": ["should not count"],
                    "semantics": {"intent-1": unit("music", "intent")},
                },
            ],
            "test": [
                {
                    "id": "x",
                    "query": "x",
                    "split_sens": ["ＯＰＥＮ， WINDOW"],
                    "semantics": {"intent-1": unit("vehicle", "intent")},
                }
            ],
        }
        summary, _ = audit.audit_dataset(splits, vehicle_domain="vehicle")
        diagnostic = summary["composition_overlap"]
        validation = diagnostic["evaluation"]["validation"]
        combined = diagnostic["evaluation"]["all"]
        clean = summary["composition_overlap_clean_vehicle_aligned"]

        self.assertEqual(diagnostic["diagnostic"], "composition_overlap")
        self.assertEqual(validation["eval_fragment_occurrences"], 3)
        self.assertEqual(validation["fragment_occurrences_in_train"], 2)
        self.assertAlmostEqual(validation["fragment_occurrence_overlap_ratio"], 2 / 3)
        self.assertEqual(validation["eval_samples_with_nonempty_fragments"], 2)
        self.assertEqual(validation["samples_all_nonempty_fragments_in_train"], 1)
        self.assertEqual(validation["sample_all_fragments_overlap_ratio"], 0.5)
        self.assertEqual(combined["eval_fragment_occurrences"], 4)
        self.assertEqual(combined["fragment_occurrences_in_train"], 3)
        self.assertEqual(clean["train_unique_nonempty_fragments"], 0)
        self.assertEqual(
            clean["evaluation"]["all"]["fragment_occurrences_in_train"],
            0,
        )

    def test_empty_queries_are_not_overlap_candidates(self):
        summary, issues = audit.audit_dataset(
            {
                "train": [{"id": "t", "query": "", "split_sens": []}],
                "test": [{"id": "x", "query": "", "split_sens": []}],
            }
        )
        self.assertEqual(summary["totals"]["cross_split_query_overlap_exact_groups"], 0)
        self.assertFalse(any("query_overlap" in issue["code"] for issue in issues))


class LocalIoTest(unittest.TestCase):
    def test_local_loader_provenance_and_atomic_non_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {
                "id": "1",
                "query": "open window",
                "split_sens": ["open window"],
                "semantics": {"intent-1": unit("vehicle", "intent")},
            }
            (root / "train.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            source_hash = audit.sha256_file(root / "train.jsonl")
            manifest = {
                "dataset": {
                    "id": "fixture",
                    "revision": "fixed",
                    "vehicle_domain": "vehicle",
                    "source_files": {
                        "train": {
                            "relative_path": "train.jsonl",
                            "records": 1,
                            "sha256": source_hash,
                        }
                    },
                },
                "paper_claims": {"maximum_intents_per_utterance": 5},
                "known_vehicle_slot_names": ["intent"],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            splits, provenance, loaded_manifest = audit.load_local_jsonl(
                manifest_path, root
            )
            self.assertEqual(splits["train"], [row])
            self.assertEqual(loaded_manifest["dataset"]["revision"], "fixed")
            self.assertEqual(len(provenance["files"][0]["sha256"]), 64)
            self.assertTrue(provenance["files"][0]["sha256_matches_manifest"])
            self.assertTrue(provenance["files"][0]["records_match_manifest"])
            self.assertEqual(len(provenance["manifest"]["sha256"]), 64)
            self.assertEqual(provenance["manifest"]["name"], "manifest.json")
            self.assertNotIn(str(root), json.dumps(provenance))
            self.assertTrue(provenance["python_version"])

            verification, verification_issues = audit.verify_sources(
                provenance["files"]
            )
            self.assertEqual(verification["status"], "verified")
            self.assertEqual(verification_issues, [])

            summary, issues = audit.audit_dataset(
                splits,
                vehicle_domain="vehicle",
                allowed_vehicle_slots={"intent"},
            )
            first = root / "audit-one"
            second = root / "audit-two"
            audit.write_outputs(first, summary, issues)
            audit.write_outputs(second, summary, issues)
            self.assertEqual(
                (first / "summary.json").read_bytes(),
                (second / "summary.json").read_bytes(),
            )
            self.assertEqual(
                (first / "issues.csv").read_bytes(),
                (second / "issues.csv").read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                audit.write_outputs(first, summary, issues)

    def test_source_verification_reports_both_mismatch_types(self):
        verification, issues = audit.verify_sources(
            [
                {
                    "split": "train",
                    "relative_path": "train.jsonl",
                    "expected_records": 2,
                    "records": 1,
                    "records_match_manifest": False,
                    "expected_sha256": "expected",
                    "sha256": "observed",
                    "sha256_matches_manifest": False,
                }
            ]
        )
        self.assertEqual(verification["status"], "mismatch")
        self.assertEqual(
            {issue["code"] for issue in issues},
            {"source_record_count_mismatch", "source_sha256_mismatch"},
        )
        with self.assertRaises(audit.SourceVerificationError):
            audit.require_verified_source(verification)
        audit.require_verified_source(verification, allow_unverified=True)

    def test_manifest_cli_fails_closed_before_writing_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {
                "id": "1",
                "query": "open window",
                "split_sens": ["open window"],
                "semantics": {"intent-1": unit("vehicle", "intent")},
            }
            (root / "train.jsonl").write_text(
                json.dumps(row) + "\n",
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
                            "sha256": "0" * 64,
                        }
                    },
                },
                "known_vehicle_slot_names": ["intent"],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_dir = root / "audit"

            with self.assertRaises(audit.SourceVerificationError):
                audit.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--source-root",
                        str(root),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
