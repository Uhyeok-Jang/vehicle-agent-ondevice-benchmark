import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PREPROCESSING = Path(__file__).resolve().parents[1] / "preprocessing"
sys.path.insert(0, str(PREPROCESSING))
import audit_mivs as audit  # noqa: E402


def mivs_record(query, *intents):
    return {
        "input": query,
        "semantics": [
            {
                "domain": "vehicle",
                "intents": [{"intent": "body", "slots": slots} for slots in intents],
            }
        ],
    }


def mac_unit(*slots):
    return {
        "vehicle": [
            {"name": "intent", "value": "body"},
            *({"name": name, "value": value} for name, value in slots),
        ]
    }


def mac_record(sample_id, query, *units):
    return {
        "id": sample_id,
        "query": query,
        "split_sens": [query] * len(units),
        "semantics": {f"intent-{index}": unit for index, unit in enumerate(units, 1)},
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_fixture(root):
    archive_path = root / "aispeech.zip"
    members = {
        "aispeech/train/one_domain_data/vehicle.json": [
            mivs_record(
                "Open, Window",
                [
                    {"name": "operation", "value": "Open", "pos": [0, 3]},
                    {"name": "object", "value": "Window", "pos": [6, 11]},
                ],
            ),
            mivs_record("same", [{"name": "value", "value": "implicit"}]),
        ],
        "aispeech/train/one_domain_data/vehicle_multi.json": [],
        "aispeech/valid/one_domain_data/vehicle.json": [
            mivs_record(
                "same", [{"name": "operation", "value": "same", "pos": [0, 3]}]
            ),
            mivs_record(
                "open window",
                [{"name": "operation", "value": "open", "pos": [0, 3]}],
            ),
        ],
        "aispeech/valid/one_domain_data/vehicle_multi.json": [],
        "aispeech/test/one_domain_data/vehicle.json": [
            mivs_record(
                "shared",
                [{"name": "operation", "value": "shared", "pos": [0, 5]}],
            ),
            mivs_record("abc", [{"name": "value", "value": "z", "pos": [0, 0]}]),
        ],
        "aispeech/test/one_domain_data/vehicle_multi.json": [
            mivs_record(
                "twoparts",
                [{"name": "operation", "value": "two", "pos": [0, 2]}],
                [{"name": "object", "value": "parts", "pos": [3, 7]}],
            )
        ],
    }
    ontology = {
        "vehicle": {"hierarchy": {"body": {"object": 1, "operation": 1, "value": 1}}}
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(
            audit.ONTOLOGY_MEMBER,
            json.dumps(ontology, ensure_ascii=False).encode("utf-8"),
        )
        for member, rows in members.items():
            handle.writestr(
                member,
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ).encode("utf-8"),
            )

    split_counts = {"train": 2, "valid": 2, "test": 3}
    manifest = {
        "manifest_version": "fixture",
        "dataset": {
            "repository": {"revision": "fixture-revision"},
            "archive": {
                "relative_path": "aispeech.zip",
                "sha256": audit.sha256_file(archive_path),
            },
        },
        "release_counts": {
            "total_records": 7,
            "by_split": split_counts,
            "components": {
                "single_domain": {
                    "source_directory": "one_domain_data",
                    "total_records": 7,
                    "by_split": split_counts,
                }
            },
        },
        "vehicle_subset": {
            "domain": "vehicle",
            "total_records": 7,
            "by_split": split_counts,
            "intent_count": 1,
            "intents": ["body"],
            "slot_count": 3,
            "slots": ["object", "operation", "value"],
        },
    }
    manifest_path = root / "mivs_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    mac_root = root / "mac"
    mac_splits = {
        "train": [
            mac_record(
                "Open, Window",
                "Open, Window",
                mac_unit(("operation", "Open"), ("object", "Window")),
            )
        ],
        "validation": [mac_record("m2", "unrelated", mac_unit(("value", "x")))],
        "test": [
            mac_record("m3", "shared", mac_unit(("operation", "shared"))),
            mac_record(
                "m4",
                "two parts",
                mac_unit(("operation", "two")),
                mac_unit(("object", "parts")),
            ),
        ],
    }
    source_files = {}
    relative_paths = {
        "train": "label/train_set.jsonl",
        "validation": "label/dev_set.jsonl",
        "test": "label/test_set.jsonl",
    }
    for split, rows in mac_splits.items():
        path = mac_root / relative_paths[split]
        write_jsonl(path, rows)
        source_files[split] = {
            "relative_path": relative_paths[split],
            "records": len(rows),
            "sha256": audit.sha256_file(path),
        }
    mac_manifest = {
        "dataset": {
            "id": "fixture-mac",
            "revision": "fixture-mac-revision",
            "vehicle_domain": "vehicle",
            "source_files": source_files,
        }
    }
    mac_manifest_path = root / "mac_manifest.json"
    mac_manifest_path.write_text(json.dumps(mac_manifest), encoding="utf-8")
    return archive_path, manifest_path, mac_root, mac_manifest_path


class IdentityAndOffsetTest(unittest.TestCase):
    def test_identity_is_canonical_and_source_sensitive(self):
        first = audit.stable_record_identity(
            "revision", "a.json", 1, "Ａ B", {"b": 2, "a": 1}
        )
        reordered = audit.stable_record_identity(
            "revision", "a.json", 1, "ab", {"a": 1, "b": 2}
        )
        punctuated = audit.stable_record_identity(
            "revision", "a.json", 1, "ab!", {"a": 1, "b": 2}
        )
        moved = audit.stable_record_identity(
            "revision", "a.json", 2, "ab", {"a": 1, "b": 2}
        )
        self.assertEqual(first["record_id"], reordered["record_id"])
        self.assertNotEqual(first["query_raw_sha256"], reordered["query_raw_sha256"])
        self.assertNotEqual(reordered["record_id"], punctuated["record_id"])
        self.assertNotEqual(first["record_id"], moved["record_id"])
        self.assertEqual(len(first["record_id"].removeprefix("mivs:")), 64)

    def test_normalization_matches_mac_v2_protocol(self):
        query = " Ａ-B, 1.5 +2 "
        self.assertEqual(audit.normalize_query(query), "a-b,1.5+2")
        self.assertEqual(audit.normalize_query_for_review(query), "ab1.5+2")

    def test_offsets_use_inclusive_unicode_code_points(self):
        common = {
            "split": "test",
            "member": "member",
            "line": 1,
            "record_id": "id",
            "unit_index": 0,
            "slot_index": 0,
        }
        status, issue = audit._audit_slot(
            "가나다", {"name": "object", "value": "나다", "pos": [1, 2]}, **common
        )
        self.assertEqual((status, issue), ("exact", None))
        status, issue = audit._audit_slot(
            "가나다", {"name": "object", "value": "가", "pos": [True, 0]}, **common
        )
        self.assertEqual(status, "invalid")
        self.assertEqual(issue["code"], "invalid_pos_shape")
        status, issue = audit._audit_slot(
            "가나다", {"name": "object", "value": "가", "pos": [0, 3]}, **common
        )
        self.assertEqual(status, "invalid")
        self.assertEqual(issue["code"], "pos_out_of_bounds")
        self.assertEqual(
            audit._audit_slot(
                "가나다", {"name": "object", "value": "implicit"}, **common
            ),
            ("missing", None),
        )


class FixtureAuditTest(unittest.TestCase):
    def run_fixture(self, root):
        archive, manifest, mac_root, mac_manifest = build_fixture(root)
        result = audit.audit_sources(manifest, archive, mac_manifest, mac_root)
        return (*result, archive, manifest, mac_root, mac_manifest)

    def test_audit_reproduces_counts_overlap_and_redacted_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, issues, records, _, _, _, _ = self.run_fixture(root)

            self.assertEqual(summary["counts"]["release"]["records"], 7)
            self.assertEqual(summary["counts"]["vehicle"]["units"], 8)
            self.assertEqual(
                summary["counts"]["recommended_test"]["source_unit_count_distribution"],
                {"1": 2, "2": 1},
            )
            self.assertEqual(summary["offsets"]["slots"], 9)
            self.assertEqual(summary["offsets"]["pos_present"], 8)
            self.assertEqual(summary["offsets"]["pos_missing"], 1)
            self.assertEqual(summary["offsets"]["pos_exact"], 7)
            self.assertEqual(summary["offsets"]["pos_mismatch"], 1)
            self.assertEqual(summary["ontology"]["observed_path_violations"], 0)
            self.assertEqual(
                summary["overlap"]["mivs_cross_split"]["exact"][
                    "pairwise_unique_query_keys"
                ]["train_valid"],
                1,
            )
            self.assertEqual(
                summary["overlap"]["mivs_cross_split"]["quarantine_normalized"][
                    "pairwise_unique_query_keys"
                ]["train_valid"],
                1,
            )
            self.assertEqual(
                summary["overlap"]["mivs_cross_split"]["review_normalized"][
                    "pairwise_unique_query_keys"
                ]["train_valid"],
                2,
            )
            mac = summary["overlap"]["mac"]
            self.assertEqual(mac["full_vehicle"]["exact_unique_queries"], 2)
            self.assertEqual(
                mac["full_vehicle"]["intent_sequence_equal_unique_queries"], 2
            )
            self.assertEqual(mac["full_vehicle"]["full_frame_equal_unique_queries"], 2)
            self.assertEqual(
                mac["full_vehicle"]["quarantine_normalized_additional_unique_keys"],
                1,
            )
            self.assertEqual(
                mac["full_vehicle"]["quarantine_normalized_total_unique_keys"], 3
            )
            self.assertEqual(
                mac["full_vehicle"]["review_normalized_additional_unique_keys"],
                0,
            )
            self.assertEqual(
                mac["full_vehicle"]["review_normalized_total_unique_keys"], 3
            )
            self.assertEqual(mac["recommended_test"]["exact_records"], 1)
            self.assertEqual(mac["recommended_test"]["multi_exact_records"], 0)
            self.assertEqual(
                mac["recommended_test"]["quarantine_normalized_records"], 2
            )
            self.assertEqual(mac["recommended_test"]["review_normalized_records"], 2)
            self.assertEqual(len(records), 7)
            self.assertEqual(len({record["record_id"] for record in records}), 7)
            self.assertTrue(
                all(record["record_id"].startswith("mivs:") for record in records)
            )
            self.assertIn(
                "slot_value_span_mismatch", {issue["code"] for issue in issues}
            )
            issue_codes = {issue["code"] for issue in issues}
            self.assertIn("mac_query_overlap_quarantine_normalized", issue_codes)
            self.assertIn("mac_query_overlap_review_normalized", issue_codes)
            self.assertIn("mivs_cross_split_query_quarantine_normalized", issue_codes)
            self.assertIn("mivs_cross_split_query_review_normalized", issue_codes)

            output = root / "output"
            audit.write_outputs(output, summary, issues, records)
            emitted = b"".join(
                (output / name).read_bytes()
                for name in ("summary.json", "issues.csv", "record_index.jsonl")
            )
            for raw in (
                b"Open, Window",
                b"shared",
                b"implicit",
                b"twoparts",
                b"Window",
                b"two parts",
            ):
                self.assertNotIn(raw, emitted)
            self.assertEqual(
                len(summary["provenance"]["archive"]["observed_sha256"]), 64
            )
            self.assertTrue(
                all(
                    len(member["sha256"]) == 64
                    for member in summary["provenance"]["members"]
                )
            )

    def test_outputs_are_atomic_deterministic_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, issues, records, _, _, _, _ = self.run_fixture(root)
            first = root / "first"
            second = root / "second"
            audit.write_outputs(first, summary, issues, records)
            audit.write_outputs(second, summary, issues, records)
            for name in ("summary.json", "issues.csv", "record_index.jsonl"):
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )
            with self.assertRaises(FileExistsError):
                audit.write_outputs(first, summary, issues, records)

    def test_archive_hash_mismatch_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, manifest, mac_root, mac_manifest = build_fixture(root)
            archive.write_bytes(archive.read_bytes() + b"tamper")
            output = root / "must-not-exist"
            with self.assertRaises(SystemExit) as raised:
                audit.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--archive",
                        str(archive),
                        "--mac-manifest",
                        str(mac_manifest),
                        "--mac-source-root",
                        str(mac_root),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_mac_hash_mismatch_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, manifest, mac_root, mac_manifest = build_fixture(root)
            path = mac_root / "label/train_set.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            output = root / "must-not-exist"
            with self.assertRaises(SystemExit) as raised:
                audit.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--archive",
                        str(archive),
                        "--mac-manifest",
                        str(mac_manifest),
                        "--mac-source-root",
                        str(mac_root),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())


@unittest.skipUnless(
    os.environ.get("MIVS_AISPEECH_ZIP") and os.environ.get("MAC_SLU_SOURCE_ROOT"),
    "pinned raw MIVS and MAC sources are not configured",
)
class PinnedSourceIntegrationTest(unittest.TestCase):
    def test_pinned_decision_counts(self):
        repo = Path(__file__).resolve().parents[2]
        summary, _, records = audit.audit_sources(
            repo / "research/config/mivs_source_manifest.json",
            Path(os.environ["MIVS_AISPEECH_ZIP"]),
            repo / "research/config/macslu_source_manifest.json",
            Path(os.environ["MAC_SLU_SOURCE_ROOT"]),
        )
        self.assertEqual(summary["counts"]["release"]["records"], 105240)
        self.assertEqual(summary["counts"]["vehicle"]["records"], 20000)
        self.assertEqual(summary["counts"]["vehicle"]["units"], 44880)
        self.assertEqual(summary["counts"]["recommended_test"]["records"], 2000)
        self.assertEqual(summary["counts"]["recommended_test"]["units"], 4485)
        self.assertEqual(
            summary["counts"]["recommended_test"]["source_unit_count_distribution"],
            {"1": 1000, "2": 19, "3": 477, "4": 504},
        )
        self.assertEqual(summary["offsets"]["slots"], 137106)
        self.assertEqual(summary["offsets"]["pos_present"], 115496)
        self.assertEqual(summary["offsets"]["pos_missing"], 21610)
        self.assertEqual(summary["offsets"]["pos_exact"], 115493)
        self.assertEqual(summary["offsets"]["pos_mismatch"], 3)
        exact = summary["overlap"]["mivs_cross_split"]["exact"]
        self.assertEqual(
            exact["pairwise_unique_query_keys"],
            {"train_valid": 24, "train_test": 6, "valid_test": 2},
        )
        mac = summary["overlap"]["mac"]
        self.assertEqual(mac["full_vehicle"]["exact_unique_queries"], 96)
        self.assertEqual(
            mac["full_vehicle"]["intent_sequence_equal_unique_queries"], 96
        )
        self.assertEqual(mac["full_vehicle"]["full_frame_equal_unique_queries"], 28)
        self.assertEqual(mac["recommended_test"]["exact_records"], 13)
        self.assertEqual(mac["recommended_test"]["multi_exact_records"], 0)
        self.assertEqual(len(records), 20000)


if __name__ == "__main__":
    unittest.main()
