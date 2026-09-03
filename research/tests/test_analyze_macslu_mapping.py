import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
import analyze_macslu_mapping as analysis
import map_macslu_vehicle as mapping


VEHICLE = "车载控制"
REVISION = "fixture-revision"


def slot(name, value):
    return {"name": name, "value": value}


def frame(intent, *slots_):
    return {VEHICLE: [slot("intent", intent), *slots_]}


def row(source_id, *frames):
    return {
        "id": source_id,
        "query": "ignored",
        "split_sens": ["ignored"],
        "semantics": {
            f"意图{index}": value for index, value in enumerate(frames, 1)
        },
    }


class MappingCoverageTest(unittest.TestCase):
    def test_unit_row_and_failure_attribution(self):
        splits = {
            "train": [
                row(
                    "full",
                    frame(
                        "车身控制",
                        slot("对象", "空调"),
                        slot("操作", "打开"),
                    ),
                ),
                row(
                    "partial",
                    frame(
                        "车身控制",
                        slot("对象", "空调"),
                        slot("操作", "打开"),
                    ),
                    frame(
                        "车身控制",
                        slot("对象", "车窗"),
                        slot("操作", "打开"),
                    ),
                ),
                row(
                    "zero",
                    frame(
                        "车机控制",
                        slot("操作", "打开"),
                        slot("页面", "设置"),
                    ),
                ),
                row(
                    "unknown",
                    frame(
                        "车身控制",
                        slot("对象", "空调"),
                        slot("操作", "打开"),
                        slot("模式", "fixture-unknown"),
                    ),
                ),
            ]
        }
        statuses = {
            f"macslu:{REVISION}:train:{source_id}": "candidate"
            for source_id in ("full", "partial", "zero", "unknown")
        }
        summary, unresolved, failures = analysis.analyze_mapping(
            splits,
            revision=REVISION,
            mapper=mapping.MacsluVehicleMapper(),
            initial_status_by_example_id=statuses,
        )

        self.assertEqual(summary["counts"]["vehicle_rows"], 4)
        self.assertEqual(summary["counts"]["vehicle_units"], 5)
        self.assertEqual(
            summary["unit_outcomes"]["counts"],
            {
                "mapped": 2,
                "ambiguous": 1,
                "unsupported": 1,
                "needs_context": 1,
            },
        )
        self.assertEqual(
            summary["row_outcomes"]["counts"],
            {
                "fully_mapped": 1,
                "partially_mapped": 1,
                "zero_mapped": 2,
            },
        )
        self.assertEqual(
            summary["row_outcomes"]["fully_mapped_call_count_distribution"],
            {"1": 1},
        )
        self.assertEqual(summary["final_eligibility"]["eligible_rows"], None)
        self.assertIn(
            {
                "slot_name": "模式",
                "slot_value": "fixture-unknown",
                "total": 1,
                "train": 1,
                "validation": 0,
                "test": 0,
            },
            unresolved,
        )
        self.assertTrue(
            any(row["reason_codes"] == "function_outside_schema" for row in failures)
        )

    def test_outputs_are_deterministic_and_non_overwriting(self):
        summary = {"schema_version": 1, "counts": {"vehicle_units": 1}}
        unresolved = [
            {
                "slot_name": "value",
                "slot_value": "x",
                "total": 1,
                "train": 1,
                "validation": 0,
                "test": 0,
            }
        ]
        failures = [
            {
                "status": "ambiguous",
                "reason_codes": "unrecognized_source_value",
                "normalized_signature": json.dumps(
                    {"entity": "hvac"}, separators=(",", ":"), sort_keys=True
                ),
                "total": 1,
                "train": 1,
                "validation": 0,
                "test": 0,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            analysis.write_outputs(first, summary, unresolved, failures)
            analysis.write_outputs(second, summary, unresolved, failures)
            for name in ("summary.json", "unresolved_values.csv", "failure_signatures.csv"):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )
            with self.assertRaises(FileExistsError):
                analysis.write_outputs(first, summary, unresolved, failures)


if __name__ == "__main__":
    unittest.main()
