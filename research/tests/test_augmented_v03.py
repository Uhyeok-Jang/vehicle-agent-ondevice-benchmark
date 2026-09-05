import copy
import hashlib
import json
import random
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = RESEARCH_ROOT.parent
PREPROCESSING_ROOT = RESEARCH_ROOT / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_ROOT))

import build_augmented_v03 as builder
import generate_synthetic_v03 as generator


ORIGINAL_ROOT = RESEARCH_ROOT / "data" / "processed" / "macslu_korean_v0.2"
SYNTHETIC_ROOT = RESEARCH_ROOT / "data" / "synthetic"
FINAL_ROOT = RESEARCH_ROOT / "data" / "processed" / "macslu_korean_augmented_v0.3"
CANDIDATES_PATH = SYNTHETIC_ROOT / "synthetic_candidates_v0.3.jsonl"
VALID_PATH = SYNTHETIC_ROOT / "synthetic_valid_v0.3.jsonl"
ORIGINAL_POOL_PATH = SYNTHETIC_ROOT / "original_source_pool_v0.3.jsonl"
GENERATION_REPORT_PATH = SYNTHETIC_ROOT / "synthetic_generation_report_v0.3.json"
DATASET_REPORT_PATH = FINAL_ROOT / "dataset_report.json"


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AugmentedV03ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema, cls.registry = builder.load_contract(
            builder.DEFAULT_SCHEMA,
            builder.DEFAULT_REGISTRY,
        )
        cls.original_by_split = {
            split: load_jsonl(ORIGINAL_ROOT / f"{split}.jsonl")
            for split in builder.SPLITS
        }
        cls.original_pool = load_jsonl(ORIGINAL_POOL_PATH)
        cls.candidates = load_jsonl(CANDIDATES_PATH)
        cls.synthetic_valid = load_jsonl(VALID_PATH)
        cls.final_by_split = {
            split: load_jsonl(FINAL_ROOT / f"{split}.jsonl") for split in builder.SPLITS
        }
        cls.final_rows = [
            row for split in builder.SPLITS for row in cls.final_by_split[split]
        ]
        cls.generation_report = json.loads(
            GENERATION_REPORT_PATH.read_text(encoding="utf-8")
        )
        cls.dataset_report = json.loads(DATASET_REPORT_PATH.read_text(encoding="utf-8"))

    def test_original_pool_is_exact_frozen_v02_pool_with_flat_provenance(self):
        self.assertEqual(len(self.original_pool), 439)
        self.assertEqual(
            Counter(row["previous_benchmark_split"] for row in self.original_pool),
            Counter(builder.ORIGINAL_SPLIT_COUNTS),
        )

        expected_by_id = {}
        for split in builder.SPLITS:
            for original in self.original_by_split[split]:
                expected = copy.deepcopy(original)
                expected["previous_benchmark_split"] = expected.pop("benchmark_split")
                expected["previous_benchmark_version"] = expected.pop(
                    "benchmark_version"
                )
                expected["source_type"] = "original"
                expected_by_id[expected["example_id"]] = expected

        self.assertEqual(len(expected_by_id), 439)
        self.assertEqual(
            {row["example_id"]: row for row in self.original_pool},
            expected_by_id,
        )
        for row in self.original_pool:
            self.assertNotIn("benchmark_split", row)
            self.assertNotIn("benchmark_version", row)
            self.assertEqual(row["previous_benchmark_version"], "macslu_korean_v0.2")

    def test_synthetic_candidates_meet_quota_schema_and_language_contract(self):
        self.assertEqual(len(self.candidates), 2300)
        self.assertEqual(
            Counter(row["call_count"] for row in self.candidates),
            Counter(builder.SYNTHETIC_TARGETS),
        )
        self.assertEqual(
            len({row["example_id"] for row in self.candidates}),
            len(self.candidates),
        )

        validator = Draft202012Validator(self.schema)
        for row in self.candidates:
            with self.subTest(example_id=row["example_id"]):
                self.assertEqual(row["source_type"], "synthetic")
                self.assertEqual(row["source_split"], "synthetic")
                self.assertEqual(row["source_group_id"], row["example_id"])
                self.assertEqual(row["call_count"], len(row["canonical_calls"]))
                self.assertFalse(
                    list(validator.iter_errors({"calls": row["canonical_calls"]}))
                )
                self.assertIsNone(generator.language_issue(row["utterance_ko"]))
                self.assertFalse(
                    generator.has_resource_conflict(row["canonical_calls"])
                )

    def test_synthetic_ordered_signatures_are_unique_and_novel(self):
        original_signatures = {
            builder.canonical_calls_key(row["canonical_calls"])
            for row in self.original_pool
        }
        synthetic_signatures = [
            builder.canonical_calls_key(row["canonical_calls"])
            for row in self.synthetic_valid
        ]
        self.assertEqual(len(synthetic_signatures), 2300)
        self.assertEqual(len(set(synthetic_signatures)), 2300)
        self.assertTrue(original_signatures.isdisjoint(synthetic_signatures))
        self.assertEqual(CANDIDATES_PATH.read_bytes(), VALID_PATH.read_bytes())

    def test_final_splits_are_an_exact_leakage_safe_union(self):
        source_rows = [*self.original_pool, *self.synthetic_valid]
        source_by_id = {row["example_id"]: row for row in source_rows}
        final_ids = [row["example_id"] for row in self.final_rows]

        self.assertEqual(len(source_rows), 2739)
        self.assertEqual(len(final_ids), len(set(final_ids)))
        self.assertEqual(set(final_ids), set(source_by_id))
        self.assertEqual(
            {split: len(rows) for split, rows in self.final_by_split.items()},
            {"train": 2192, "validation": 274, "test": 273},
        )

        normalized_by_split = {}
        family_splits = defaultdict(set)
        for split, rows in self.final_by_split.items():
            normalized_by_split[split] = {
                builder.normalize_text(row["utterance_ko"]) for row in rows
            }
            self.assertEqual(len(normalized_by_split[split]), len(rows))
            for row in rows:
                self.assertEqual(row["benchmark_split"], split)
                self.assertEqual(row["benchmark_version"], builder.BENCHMARK_VERSION)
                unsplit = copy.deepcopy(row)
                unsplit.pop("benchmark_split")
                unsplit.pop("benchmark_version")
                self.assertEqual(unsplit, source_by_id[row["example_id"]])
                if row["source_type"] == "synthetic":
                    family_splits[
                        row["synthetic_generation"]["generation_family_id"]
                    ].add(split)

        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertTrue(
                normalized_by_split[left].isdisjoint(normalized_by_split[right])
            )
        self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))

    def test_reports_match_artifacts_and_release_assertions(self):
        report = self.dataset_report
        self.assertEqual(
            report["counts"],
            {
                "total_examples": 2739,
                "original": 439,
                "synthetic": 2300,
                "splits": {"train": 2192, "validation": 274, "test": 273},
                "source_type_by_split": {
                    "train": {"original": 352, "synthetic": 1840},
                    "validation": {"original": 44, "synthetic": 230},
                    "test": {"original": 43, "synthetic": 230},
                },
            },
        )
        self.assertTrue(report["synthetic_targets"]["exact_target_met"])
        self.assertTrue(report["synthetic_release_coverage"]["passed"])
        self.assertTrue(
            report["canonical_signature_policy"]["all_synthetic_signatures_unique"]
        )
        self.assertTrue(
            report["canonical_signature_policy"]["all_synthetic_signatures_novel"]
        )
        self.assertTrue(all(report["split_integrity"]["assertions"].values()))
        self.assertEqual(report["review_sample"]["count"], 32)
        self.assertEqual(
            report["review_sample"]["call_count_distribution"],
            {"1": 8, "2": 8, "3": 8, "4": 8},
        )

        for split, expected_hash in builder.ORIGINAL_INPUT_SHA256.items():
            path = ORIGINAL_ROOT / f"{split}.jsonl"
            self.assertEqual(sha256(path), expected_hash)
            self.assertEqual(
                report["hashes"]["v0_2_inputs"][split]["sha256"], expected_hash
            )

        artifact_paths = {
            "original_source_pool": ORIGINAL_POOL_PATH,
            "synthetic_valid": VALID_PATH,
            **{split: FINAL_ROOT / f"{split}.jsonl" for split in builder.SPLITS},
        }
        for name, path in artifact_paths.items():
            entry = report["hashes"]["artifacts"][name]
            self.assertEqual(REPOSITORY_ROOT / entry["path"], path)
            self.assertEqual(entry["sha256"], sha256(path))

        self.assertEqual(
            self.generation_report["artifact"]["candidate_sha256"],
            sha256(CANDIDATES_PATH),
        )
        self.assertEqual(
            self.generation_report["canonical_signature_novelty"][
                "synthetic_unique_signatures"
            ],
            2300,
        )
        self.assertEqual(
            self.generation_report["canonical_signature_novelty"][
                "original_synthetic_overlap"
            ],
            0,
        )


class AugmentedV03HelperContractTest(unittest.TestCase):
    def test_renderer_preserves_scope_and_target_semantic_contrasts(self):
        omitted = {
            "function": "set_hvac_temperature",
            "arguments": {
                "target": {"kind": "absolute", "value": 22.5, "unit": "celsius"}
            },
        }
        all_zone = copy.deepcopy(omitted)
        all_zone["arguments"]["zone"] = "all"
        omitted_text = generator.render_call(omitted, random.Random(1)).final
        all_text = generator.render_call(all_zone, random.Random(1)).final
        self.assertNotIn("전체", omitted_text)
        self.assertIn("전체", all_text)
        self.assertIn("22.5도", omitted_text)

        named_open = {
            "function": "set_sunroof_position",
            "arguments": {"target": {"kind": "named", "value": "open"}},
        }
        absolute_percent = {
            "function": "set_sunroof_position",
            "arguments": {"target": {"kind": "absolute_percent", "value": 50}},
        }
        named_text = generator.render_call(named_open, random.Random(2)).final
        percent_text = generator.render_call(absolute_percent, random.Random(2)).final
        self.assertNotIn("%", named_text)
        self.assertIn("50%", percent_text)

        seat_state = {
            "function": "set_seat_massage",
            "arguments": {
                "zone": "driver",
                "setting": {"kind": "state", "value": "on"},
            },
        }
        seat_level = copy.deepcopy(seat_state)
        seat_level["arguments"]["setting"] = {
            "kind": "absolute_level",
            "value": 2,
        }
        state_text = generator.render_call(seat_state, random.Random(3)).final
        level_text = generator.render_call(seat_level, random.Random(3)).final
        self.assertNotIn("2단", state_text)
        self.assertIn("2단", level_text)

    def test_resource_overlap_helpers_cover_scope_lattice_and_seat_feature(self):
        hvac_implicit = {
            "function": "set_hvac_power",
            "arguments": {"state": "on"},
        }
        hvac_driver = {
            "function": "set_hvac_power",
            "arguments": {"state": "off", "zone": "driver"},
        }
        window_front = {
            "function": "set_window_position",
            "arguments": {
                "zone": "front_row",
                "target": {"kind": "named", "value": "open"},
            },
        }
        window_left = copy.deepcopy(window_front)
        window_left["arguments"]["zone"] = "left_side"
        window_rear_right = copy.deepcopy(window_front)
        window_rear_right["arguments"]["zone"] = "rear_right"
        seat_heating = {
            "function": "set_seat_climate",
            "arguments": {
                "zone": "driver",
                "feature": "heating",
                "setting": {"kind": "state", "value": "on"},
            },
        }
        seat_ventilation = copy.deepcopy(seat_heating)
        seat_ventilation["arguments"]["feature"] = "ventilation"

        conflicting_pairs = (
            [hvac_implicit, hvac_driver],
            [window_front, window_left],
        )
        disjoint_pairs = (
            [window_front, window_rear_right],
            [seat_heating, seat_ventilation],
        )
        for calls in conflicting_pairs:
            self.assertTrue(generator.has_resource_conflict(calls))
            self.assertTrue(builder.find_effective_resource_conflicts(calls))
        for calls in disjoint_pairs:
            self.assertFalse(generator.has_resource_conflict(calls))
            self.assertFalse(builder.find_effective_resource_conflicts(calls))


if __name__ == "__main__":
    unittest.main()
