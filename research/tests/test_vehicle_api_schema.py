import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
import canonical_vehicle_api as canonical


class VehicleApiSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = canonical.load_json_object(canonical.DEFAULT_SCHEMA)
        cls.registry = canonical.load_json_object(canonical.DEFAULT_REGISTRY)

    def assert_valid_call(self, call):
        canonical.validate_payload({"calls": [call]}, self.schema)

    def assert_invalid_payload(self, payload):
        with self.assertRaises(canonical.CanonicalValidationError):
            canonical.validate_payload(payload, self.schema)

    def test_schema_and_registry_have_the_same_eight_functions(self):
        canonical.validate_registry(self.schema, self.registry)
        self.assertEqual(
            canonical.schema_function_names(self.schema),
            {
                "set_hvac_power",
                "set_hvac_temperature",
                "set_hvac_fan_speed",
                "set_window_position",
                "set_sunroof_position",
                "set_sunshade_position",
                "set_seat_climate",
                "set_seat_massage",
            },
        )

    def test_each_function_has_a_valid_fixture(self):
        calls = [
            {
                "function": "set_hvac_power",
                "arguments": {"state": "on", "zone": "driver"},
            },
            {
                "function": "set_hvac_temperature",
                "arguments": {
                    "target": {
                        "kind": "absolute",
                        "value": 22.5,
                        "unit": "celsius",
                    }
                },
            },
            {
                "function": "set_hvac_fan_speed",
                "arguments": {
                    "target": {
                        "kind": "relative",
                        "direction": "increase",
                        "magnitude": "small",
                    },
                    "zone": "rear",
                },
            },
            {
                "function": "set_window_position",
                "arguments": {
                    "zone": "rear_left",
                    "target": {"kind": "absolute_percent", "value": 40},
                },
            },
            {
                "function": "set_sunroof_position",
                "arguments": {"target": {"kind": "named", "value": "vent"}},
            },
            {
                "function": "set_sunshade_position",
                "arguments": {
                    "target": {
                        "kind": "relative",
                        "direction": "close",
                        "magnitude": "default",
                    }
                },
            },
            {
                "function": "set_seat_climate",
                "arguments": {
                    "zone": "driver",
                    "feature": "ventilation",
                    "setting": {"kind": "absolute_level", "value": 2},
                },
            },
            {
                "function": "set_seat_massage",
                "arguments": {
                    "zone": "rear_row",
                    "setting": {"kind": "state", "value": "off"},
                },
            },
        ]
        for call in calls:
            with self.subTest(function=call["function"]):
                self.assert_valid_call(call)

    def test_envelope_and_closed_world_rejections(self):
        self.assert_invalid_payload({"calls": []})
        self.assert_invalid_payload({"calls": None})
        self.assert_invalid_payload({"calls": [], "schema_version": "0.1.0"})
        self.assert_invalid_payload(
            {"calls": [{"function": "unknown", "arguments": {}}]}
        )
        self.assert_invalid_payload(
            {
                "calls": [
                    {
                        "function": "set_hvac_power",
                        "arguments": {"state": "on"},
                        "note": "extra",
                    }
                ]
            }
        )
        self.assert_invalid_payload(
            {
                "calls": [
                    {
                        "function": "set_hvac_power",
                        "arguments": {"state": "on", "guess": True},
                    }
                ]
            }
        )

    def test_typed_targets_reject_lossy_or_out_of_range_values(self):
        invalid_calls = [
            {
                "function": "set_hvac_temperature",
                "arguments": {
                    "target": {
                        "kind": "absolute",
                        "value": 22.2,
                        "unit": "celsius",
                    }
                },
            },
            {
                "function": "set_hvac_temperature",
                "arguments": {
                    "target": {
                        "kind": "relative",
                        "direction": "increase",
                    }
                },
            },
            {
                "function": "set_hvac_fan_speed",
                "arguments": {
                    "target": {"kind": "absolute", "value": 0, "unit": "level"}
                },
            },
            {
                "function": "set_window_position",
                "arguments": {
                    "target": {"kind": "absolute_percent", "value": 50}
                },
            },
            {
                "function": "set_window_position",
                "arguments": {
                    "zone": "driver",
                    "target": {
                        "kind": "relative_percent",
                        "direction": "open",
                        "value": 0.5,
                    },
                },
            },
            {
                "function": "set_sunshade_position",
                "arguments": {"target": {"kind": "named", "value": "vent"}},
            },
            {
                "function": "set_seat_climate",
                "arguments": {
                    "zone": "driver",
                    "feature": "heating",
                    "setting": {"kind": "absolute_level", "value": 4},
                },
            },
            {
                "function": "set_seat_massage",
                "arguments": {
                    "zone": None,
                    "setting": {"kind": "state", "value": "on"},
                },
            },
        ]
        for call in invalid_calls:
            with self.subTest(function=call["function"], call=call):
                self.assert_invalid_payload({"calls": [call]})

    def test_canonicalization_is_idempotent_and_order_aware(self):
        first_call = {
            "arguments": {
                "zone": "driver",
                "target": {
                    "value": 22.5,
                    "unit": "celsius",
                    "kind": "absolute",
                },
            },
            "function": "set_hvac_temperature",
        }
        second_call = {
            "arguments": {
                "setting": {"value": "off", "kind": "state"},
                "zone": "rear_row",
            },
            "function": "set_seat_massage",
        }
        payload = {"calls": [first_call, second_call]}
        ordered = canonical.canonical_json(
            payload,
            schema=self.schema,
            registry=self.registry,
        )
        reversed_payload = {"calls": [second_call, first_call]}
        reversed_ordered = canonical.canonical_json(
            reversed_payload,
            schema=self.schema,
            registry=self.registry,
        )
        self.assertNotEqual(ordered, reversed_ordered)
        self.assertEqual(
            canonical.canonical_json(
                payload,
                schema=self.schema,
                registry=self.registry,
                unordered_calls=True,
            ),
            canonical.canonical_json(
                reversed_payload,
                schema=self.schema,
                registry=self.registry,
                unordered_calls=True,
            ),
        )
        canonical_payload = canonical.canonicalize_payload(
            payload,
            schema=self.schema,
            registry=self.registry,
        )
        self.assertEqual(
            canonical_payload,
            canonical.canonicalize_payload(
                canonical_payload,
                schema=self.schema,
                registry=self.registry,
            ),
        )
        self.assertIn(
            '"target":{"kind":"absolute","unit":"celsius","value":22.5}',
            ordered,
        )

    def test_registry_mismatch_is_rejected(self):
        registry = copy.deepcopy(self.registry)
        registry["functions"].pop("set_hvac_power")
        with self.assertRaises(canonical.CanonicalValidationError):
            canonical.validate_registry(self.schema, registry)


if __name__ == "__main__":
    unittest.main()
