import csv
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT / "preprocessing"))
import audit_macslu as audit
import audit_mivs


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Stage3ArtifactFreshnessTest(unittest.TestCase):
    def test_macslu_audit_artifacts_match_generator_and_summary(self):
        directory = (
            RESEARCH_ROOT
            / "analysis"
            / "dataset_statistics"
            / "macslu_audit_v2"
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        issues_path = directory / "issues.csv"

        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["source_verification"]["status"], "verified")
        self.assertEqual(
            summary["provenance"]["generator"]["sha256"],
            sha256(RESEARCH_ROOT / "preprocessing" / "audit_macslu.py"),
        )
        self.assertEqual(
            summary["artifacts"]["issues"]["sha256"],
            sha256(issues_path),
        )
        with issues_path.open(encoding="utf-8", newline="") as handle:
            issue_rows = sum(1 for _ in csv.DictReader(handle))
        self.assertEqual(summary["artifacts"]["issues"]["rows"], issue_rows)

    def test_macslu_inventory_is_a_complete_fresh_ledger(self):
        directory = (
            RESEARCH_ROOT
            / "analysis"
            / "dataset_statistics"
            / "macslu_inventory_v2"
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        inventory_path = directory / "inventory.csv"
        generator = summary["provenance"]["generator"]

        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["source_verification"]["status"], "verified")
        self.assertEqual(
            generator["sha256"],
            sha256(RESEARCH_ROOT / "preprocessing" / "build_macslu_inventory.py"),
        )
        self.assertEqual(
            generator["dependencies"]["audit_macslu.py"],
            sha256(Path(audit.__file__)),
        )
        self.assertEqual(
            summary["artifacts"]["inventory"]["sha256"],
            sha256(inventory_path),
        )

        statuses = Counter()
        ids = set()
        with inventory_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            statuses[row["initial_status"]] += 1
            ids.add(row["example_id"])
            self.assertEqual(row["source_group_id"], row["example_id"])
            self.assertEqual(row["final_status"], "")

        self.assertEqual(len(rows), summary["counts"]["source_rows"])
        self.assertEqual(len(ids), len(rows))
        self.assertEqual(dict(sorted(statuses.items())), summary["status_counts"])
        self.assertEqual(
            summary["counts"]["source_rows"],
            summary["counts"]["vehicle_rows"]
            + summary["counts"]["excluded_no_vehicle_rows"],
        )

    def test_mivs_audit_is_fresh_complete_and_redacted(self):
        directory = (
            RESEARCH_ROOT
            / "analysis"
            / "dataset_statistics"
            / "mivs_audit_v1"
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        provenance = summary["provenance"]

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["source_verification"]["status"], "verified")
        self.assertEqual(summary["counts"]["release"]["records"], 105240)
        self.assertEqual(summary["counts"]["vehicle"]["records"], 20000)
        self.assertEqual(summary["counts"]["vehicle"]["units"], 44880)
        self.assertEqual(
            provenance["generator"]["sha256"],
            sha256(Path(audit_mivs.__file__)),
        )
        self.assertEqual(
            provenance["generator"]["dependencies"]["audit_macslu.py"],
            sha256(Path(audit.__file__)),
        )
        self.assertEqual(
            provenance["manifest"]["sha256"],
            sha256(RESEARCH_ROOT / "config" / "mivs_source_manifest.json"),
        )

        for key in ("issues", "record_index"):
            artifact = summary["artifacts"][key]
            self.assertEqual(artifact["sha256"], sha256(directory / artifact["name"]))

        with (directory / "issues.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(
                summary["artifacts"]["issues"]["rows"],
                sum(1 for _ in csv.DictReader(handle)),
            )
        forbidden_fields = {"query", "input", "semantics", "slots", "slot_value"}
        record_ids = set()
        with (directory / "record_index.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.assertTrue(forbidden_fields.isdisjoint(record))
                record_ids.add(record["record_id"])
        self.assertEqual(
            len(record_ids), summary["artifacts"]["record_index"]["rows"]
        )
        self.assertEqual(len(record_ids), summary["counts"]["vehicle"]["records"])

    def test_current_macslu_mapping_artifacts_are_fresh(self):
        directory = (
            RESEARCH_ROOT
            / "analysis"
            / "dataset_statistics"
            / "macslu_mapping_v0.1.0_r3"
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        provenance = summary["provenance"]
        generator = provenance["generator"]

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(provenance["source_verification"]["status"], "verified")
        self.assertEqual(summary["counts"]["vehicle_rows"], 8057)
        self.assertEqual(summary["counts"]["vehicle_units"], 11471)
        self.assertEqual(summary["unit_outcomes"]["counts"]["mapped"], 2486)
        self.assertEqual(summary["row_outcomes"]["counts"]["fully_mapped"], 625)
        self.assertEqual(summary["final_eligibility"]["status"], "not_adjudicated")
        self.assertIsNone(summary["final_eligibility"]["eligible_rows"])

        self.assertEqual(
            generator["sha256"],
            sha256(RESEARCH_ROOT / "preprocessing" / "analyze_macslu_mapping.py"),
        )
        for name, relative_path in {
            "audit_macslu.py": "preprocessing/audit_macslu.py",
            "build_macslu_inventory.py": "preprocessing/build_macslu_inventory.py",
            "map_macslu_vehicle.py": "preprocessing/map_macslu_vehicle.py",
        }.items():
            self.assertEqual(
                generator["dependencies"][name], sha256(RESEARCH_ROOT / relative_path)
            )
        for key, relative_path in {
            "mapping_registry": "schema/macslu_vehicle_mapping.v0.1.0.json",
            "mapping_schema": "schema/vehicle_mapping_registry_schema.v0.1.0.json",
            "canonical_schema": "schema/vehicle_api_schema.v0.1.0.json",
            "canonical_registry": "schema/vehicle_api_registry.v0.1.0.json",
        }.items():
            self.assertEqual(
                provenance[key]["sha256"], sha256(RESEARCH_ROOT / relative_path)
            )

        for key in ("failure_signatures", "unresolved_values"):
            artifact = summary["artifacts"][key]
            path = directory / artifact["name"]
            self.assertEqual(artifact["sha256"], sha256(path))
            with path.open(encoding="utf-8", newline="") as handle:
                rows = sum(1 for _ in csv.DictReader(handle))
            self.assertEqual(artifact["rows"], rows)


if __name__ == "__main__":
    unittest.main()
