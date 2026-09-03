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


if __name__ == "__main__":
    unittest.main()
