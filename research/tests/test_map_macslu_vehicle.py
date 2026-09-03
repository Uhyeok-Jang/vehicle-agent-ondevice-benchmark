import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
import canonical_vehicle_api as canonical
import map_macslu_vehicle as mapping


REVISION = "fixture-revision"
VEHICLE = "车载控制"


def slot(name, value):
    return {"name": name, "value": value}


def vehicle_frame(intent, *slots_):
    return {
        VEHICLE: [
            slot("intent", intent),
            *slots_,
        ]
    }


def row(*frames, source_id="1", query="not used by mapping"):
    return {
        "id": source_id,
        "query": query,
        "split_sens": [query],
        "semantics": {
            f"意图{index}": frame
            for index, frame in enumerate(frames, 1)
        },
    }


class MacsluVehicleMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = mapping.MacsluVehicleMapper()

    def map_one(self, frame):
        result = self.mapper.map_row(
            row(frame),
            revision=REVISION,
            split="train",
        )
        self.assertEqual(len(result["units"]), 1)
        return result["units"][0]

    def assert_mapped(self, frame, function, arguments):
        unit = self.map_one(frame)
        decision = unit["decision"]
        self.assertEqual(decision["status"], "mapped")
        self.assertEqual(decision["reason_codes"], [])
        self.assertEqual(
            decision["call"],
            {"function": function, "arguments": arguments},
        )
        raw_slot_count = len(frame[VEHICLE])
        self.assertEqual(
            decision["consumed_slot_ordinals"],
            list(range(raw_slot_count)),
        )
        canonical.validate_payload(
            {"calls": [decision["call"]]},
            self.mapper.canonical_schema,
        )
        return unit

    def test_all_eight_pilot_functions_have_mapped_fixtures(self):
        fixtures = [
            (
                vehicle_frame(
                    "车身控制",
                    slot("对象", "空调"),
                    slot("操作", "打开"),
                ),
                "set_hvac_power",
                {"state": "on"},
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("位置", "主驾"),
                    slot("对象", "空调"),
                    slot("操作", "调"),
                    slot("操作_concrete", "到"),
                    slot("value", "二十三度"),
                    slot("调节内容", "温度"),
                ),
                "set_hvac_temperature",
                {
                    "target": {
                        "kind": "absolute",
                        "unit": "celsius",
                        "value": 23,
                    },
                    "zone": "driver",
                },
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("对象", "空调"),
                    slot("操作", "设置"),
                    slot("value", "三挡"),
                    slot("调节内容", "风量"),
                ),
                "set_hvac_fan_speed",
                {
                    "target": {
                        "kind": "absolute",
                        "unit": "level",
                        "value": 3,
                    }
                },
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("位置", "左后"),
                    slot("对象", "车窗"),
                    slot("操作", "关闭"),
                ),
                "set_window_position",
                {
                    "zone": "rear_left",
                    "target": {"kind": "named", "value": "closed"},
                },
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("对象", "天窗"),
                    slot("操作", "打开"),
                ),
                "set_sunroof_position",
                {"target": {"kind": "named", "value": "open"}},
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("对象", "遮阳帘"),
                    slot("操作", "关闭"),
                ),
                "set_sunshade_position",
                {"target": {"kind": "named", "value": "closed"}},
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("位置", "副驾"),
                    slot("对象", "座椅"),
                    slot("对象功能", "通风"),
                    slot("操作", "打开"),
                ),
                "set_seat_climate",
                {
                    "zone": "front_passenger",
                    "feature": "ventilation",
                    "setting": {"kind": "state", "value": "on"},
                },
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("位置", "后排"),
                    slot("对象", "座椅"),
                    slot("对象功能", "按摩"),
                    slot("操作", "关闭"),
                ),
                "set_seat_massage",
                {
                    "zone": "rear_row",
                    "setting": {"kind": "state", "value": "off"},
                },
            ),
        ]
        for frame, function, arguments in fixtures:
            with self.subTest(function=function):
                self.assert_mapped(frame, function, arguments)

    def test_reviewed_window_amplitude_closed_pattern(self):
        self.assert_mapped(
            vehicle_frame(
                "车身控制",
                slot("value", "关闭"),
                slot("位置", "前排"),
                slot("对象", "车窗"),
                slot("操作", "调节"),
                slot("调节内容", "幅度"),
            ),
            "set_window_position",
            {
                "zone": "front_row",
                "target": {"kind": "named", "value": "closed"},
            },
        )

        unresolved_zone = self.map_one(
            vehicle_frame(
                "车身控制",
                slot("value", "关闭"),
                slot("位置", "前车"),
                slot("对象", "车窗"),
                slot("操作", "调节"),
                slot("调节内容", "幅度"),
            )
        )
        self.assertEqual(unresolved_zone["decision"]["status"], "ambiguous")
        self.assertEqual(
            unresolved_zone["decision"]["reason_codes"],
            ["unrecognized_source_value"],
        )

        unresolved_percent = self.map_one(
            vehicle_frame(
                "车身控制",
                slot("value", "一半"),
                slot("位置", "前排"),
                slot("对象", "车窗"),
                slot("操作", "调节"),
                slot("调节内容", "幅度"),
            )
        )
        self.assertEqual(unresolved_percent["decision"]["status"], "ambiguous")
        self.assertEqual(
            unresolved_percent["decision"]["reason_codes"],
            ["unrecognized_source_value"],
        )

    def test_reviewed_window_group_zone_aliases_are_entity_scoped(self):
        for source_zone, canonical_zone in (
            ("前排前排", "front_row"),
            ("四个", "all"),
            ("四门", "all"),
        ):
            with self.subTest(source_zone=source_zone):
                self.assert_mapped(
                    vehicle_frame(
                        "车身控制",
                        slot("位置", source_zone),
                        slot("对象", "车窗"),
                        slot("操作", "关闭"),
                    ),
                    "set_window_position",
                    {
                        "zone": canonical_zone,
                        "target": {"kind": "named", "value": "closed"},
                    },
                )

        seat = self.map_one(
            vehicle_frame(
                "车身控制",
                slot("位置", "四门"),
                slot("对象", "座椅"),
                slot("对象功能", "按摩"),
                slot("操作", "关闭"),
            )
        )
        self.assertEqual(seat["decision"]["status"], "ambiguous")
        self.assertEqual(
            seat["decision"]["reason_codes"],
            ["unrecognized_source_value"],
        )

    def test_information_fragment_inherits_only_unique_hvac_row_context(self):
        information = vehicle_frame(
            "提供信息",
            slot("调节内容", "风量"),
            slot("操作", "调"),
            slot("操作_concrete", "到"),
            slot("value", "三挡"),
        )
        hvac = vehicle_frame(
            "车身控制",
            slot("对象", "空调"),
            slot("操作", "打开"),
        )
        contextual = self.mapper.map_row(
            row(hvac, information),
            revision=REVISION,
            split="train",
        )
        self.assertEqual(
            contextual["canonical_payload"]["calls"],
            [
                {"function": "set_hvac_power", "arguments": {"state": "on"}},
                {
                    "function": "set_hvac_fan_speed",
                    "arguments": {
                        "target": {
                            "kind": "absolute",
                            "unit": "level",
                            "value": 3,
                        }
                    },
                },
            ],
        )
        context_trace = contextual["units"][1]["decision"]["trace"][-1]
        self.assertEqual(
            context_trace["normalizer_id"], "context.entity.unique_in_row"
        )
        self.assertEqual(
            context_trace["source_unit_ids"],
            [contextual["units"][0]["unit_id"]],
        )

        standalone = self.map_one(information)
        self.assertEqual(standalone["decision"]["status"], "needs_context")
        self.assertEqual(
            standalone["decision"]["reason_codes"], ["missing_entity_context"]
        )

        seat = vehicle_frame(
            "车身控制",
            slot("位置", "副驾"),
            slot("对象", "座椅"),
            slot("对象功能", "通风"),
            slot("操作", "打开"),
        )
        non_hvac = self.mapper.map_row(
            row(seat, information), revision=REVISION, split="train"
        )
        self.assertEqual(
            non_hvac["units"][1]["decision"]["reason_codes"],
            ["missing_entity_context"],
        )

        conflicting = self.mapper.map_row(
            row(hvac, seat, information), revision=REVISION, split="train"
        )
        self.assertEqual(
            conflicting["units"][2]["decision"]["reason_codes"],
            ["missing_entity_context"],
        )

    def test_adapter_orders_semantic_units_numerically(self):
        fixture = {
            "id": "ordered",
            "query": "ignored",
            "semantics": {
                "意图2": vehicle_frame(
                    "车身控制",
                    slot("对象", "天窗"),
                    slot("操作", "关闭"),
                ),
                "意图1": vehicle_frame(
                    "车身控制",
                    slot("对象", "空调"),
                    slot("操作", "打开"),
                ),
            },
        }
        result = self.mapper.map_row(
            fixture,
            revision=REVISION,
            split="test",
        )
        self.assertEqual(
            [call["function"] for call in result["canonical_payload"]["calls"]],
            ["set_hvac_power", "set_sunroof_position"],
        )
        self.assertEqual(
            [unit["unit_order"] for unit in result["units"]],
            [0, 1],
        )

    def test_query_split_and_id_do_not_affect_the_call(self):
        frame = vehicle_frame(
            "车身控制",
            slot("对象", "空调"),
            slot("操作", "打开"),
        )
        first = self.mapper.map_row(
            row(frame, source_id="first", query="OPEN HVAC"),
            revision="revision-a",
            split="train",
        )
        second = self.mapper.map_row(
            row(frame, source_id="second", query="unrelated words"),
            revision="revision-b",
            split="test",
        )
        self.assertNotEqual(first["example_id"], second["example_id"])
        self.assertEqual(first["canonical_payload"], second["canonical_payload"])

    def test_alias_matching_is_nfkc_exact_without_fuzzy_or_case_folding(self):
        full_width_min = vehicle_frame(
            "车身控制",
            slot("对象", "空调"),
            slot("操作", "调"),
            slot("value", "ＭＩＮ"),
            slot("调节内容", "温度"),
        )
        mapped = self.assert_mapped(
            full_width_min,
            "set_hvac_temperature",
            {"target": {"kind": "extreme", "value": "min"}},
        )
        self.assertEqual(mapped["normalized"]["target"]["value"], "min")

        for value in ("Min", " MIN ", "二十三度左右"):
            with self.subTest(value=value):
                unit = self.map_one(
                    vehicle_frame(
                        "车身控制",
                        slot("对象", "空调"),
                        slot("操作", "调"),
                        slot("value", value),
                        slot("调节内容", "温度"),
                    )
                )
                self.assertEqual(unit["decision"]["status"], "ambiguous")
                self.assertEqual(
                    unit["decision"]["reason_codes"],
                    ["unrecognized_source_value"],
                )

    def test_non_mapped_outcomes_are_explicit(self):
        cases = [
            (
                vehicle_frame(
                    "提供信息",
                    slot("操作", "打开"),
                    slot("模式", "内循环"),
                    slot("调节内容", "模式"),
                ),
                "needs_context",
                "missing_entity_context",
            ),
            (
                vehicle_frame(
                    "车机控制",
                    slot("操作", "打开"),
                    slot("页面", "设置"),
                ),
                "unsupported",
                "function_outside_schema",
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("对象", "车窗"),
                    slot("操作", "打开"),
                ),
                "needs_context",
                "missing_required_argument",
            ),
            (
                vehicle_frame(
                    "车身控制",
                    slot("对象", "座椅"),
                    slot("对象功能", "按摩"),
                    slot("操作", "打开"),
                ),
                "needs_context",
                "missing_required_argument",
            ),
        ]
        for frame, status, reason in cases:
            with self.subTest(status=status, reason=reason):
                decision = self.map_one(frame)["decision"]
                self.assertEqual(decision["status"], status)
                self.assertEqual(decision["reason_codes"], [reason])
                self.assertIsNone(decision["call"])

    def test_unknown_and_unconsumed_slots_fail_closed(self):
        unknown = self.map_one(
            vehicle_frame(
                "车身控制",
                slot("对象", "空调"),
                slot("操作", "打开"),
                slot("new_slot", "surprise"),
            )
        )
        self.assertEqual(unknown["decision"]["status"], "ambiguous")
        self.assertEqual(
            unknown["decision"]["reason_codes"],
            ["unexpected_source_slot"],
        )

        unconsumed = self.map_one(
            vehicle_frame(
                "车身控制",
                slot("对象", "空调"),
                slot("操作", "打开"),
                slot("操作_concrete", "到"),
            )
        )
        self.assertEqual(unconsumed["decision"]["status"], "ambiguous")
        self.assertEqual(
            unconsumed["decision"]["reason_codes"],
            ["unconsumed_executable_slot"],
        )

    def test_mixed_domain_and_duplicate_slots_fail_closed(self):
        mixed = {
            "id": "mixed",
            "query": "ignored",
            "semantics": {
                "意图1": {
                    **vehicle_frame(
                        "车身控制",
                        slot("对象", "空调"),
                        slot("操作", "打开"),
                    ),
                    "音乐": [slot("intent", "播放音乐")],
                }
            },
        }
        mixed_unit = self.mapper.map_row(
            mixed,
            revision=REVISION,
            split="train",
        )["units"][0]
        self.assertEqual(mixed_unit["decision"]["status"], "ambiguous")
        self.assertEqual(
            mixed_unit["decision"]["reason_codes"],
            ["mixed_domain_example", "multi_domain_per_intent"],
        )

        duplicate = self.map_one(
            vehicle_frame(
                "车身控制",
                slot("对象", "空调"),
                slot("对象", "座椅"),
                slot("操作", "打开"),
            )
        )
        self.assertEqual(duplicate["decision"]["status"], "ambiguous")
        self.assertEqual(
            duplicate["decision"]["reason_codes"],
            ["source_annotation_conflict"],
        )

    def test_partial_multi_call_rows_never_emit_partial_payloads(self):
        result = self.mapper.map_row(
            row(
                vehicle_frame(
                    "车身控制",
                    slot("对象", "空调"),
                    slot("操作", "打开"),
                ),
                vehicle_frame(
                    "车身控制",
                    slot("对象", "车窗"),
                    slot("操作", "打开"),
                ),
            ),
            revision=REVISION,
            split="validation",
        )
        self.assertEqual(
            [unit["decision"]["status"] for unit in result["units"]],
            ["mapped", "needs_context"],
        )
        self.assertIsNone(result["canonical_payload"])

    def test_emitted_calls_are_checked_against_canonical_schema(self):
        registry = copy.deepcopy(self.mapper.registry)
        power_rule = next(
            rule
            for rule in registry["rules"]
            if rule["id"] == "map.hvac_power.activate"
        )
        power_rule["emit"]["arguments"]["state"]["const"] = "enabled"
        mapper = mapping.MacsluVehicleMapper(registry=registry)
        decision = mapper.map_row(
            row(
                vehicle_frame(
                    "车身控制",
                    slot("对象", "空调"),
                    slot("操作", "打开"),
                )
            ),
            revision=REVISION,
            split="train",
        )["units"][0]["decision"]
        self.assertEqual(decision["status"], "ambiguous")
        self.assertEqual(
            decision["reason_codes"],
            ["canonical_schema_validation_failed"],
        )
        self.assertIsNone(decision["call"])

    def test_registry_rejects_metadata_selectors_and_exact_alias_overlap(self):
        metadata_registry = copy.deepcopy(self.mapper.registry)
        metadata_registry["normalizers"][0]["input"]["path"] = "query"
        with self.assertRaises(mapping.MappingRegistryError):
            mapping.MacsluVehicleMapper(registry=metadata_registry)

        overlap_registry = copy.deepcopy(self.mapper.registry)
        duplicate_normalizer = copy.deepcopy(overlap_registry["normalizers"][0])
        duplicate_normalizer["id"] = "intent.duplicate"
        overlap_registry["normalizers"].append(duplicate_normalizer)
        with self.assertRaises(mapping.MappingRegistryError):
            mapping.MacsluVehicleMapper(registry=overlap_registry)

    def test_multiple_mapping_rules_are_reported_not_resolved_by_order(self):
        registry = copy.deepcopy(self.mapper.registry)
        duplicate_rule = copy.deepcopy(registry["rules"][0])
        duplicate_rule["id"] = "map.hvac_power.activate_duplicate"
        registry["rules"].append(duplicate_rule)
        mapper = mapping.MacsluVehicleMapper(registry=registry)
        decision = mapper.map_row(
            row(
                vehicle_frame(
                    "车身控制",
                    slot("对象", "空调"),
                    slot("操作", "打开"),
                )
            ),
            revision=REVISION,
            split="train",
        )["units"][0]["decision"]
        self.assertEqual(decision["status"], "ambiguous")
        self.assertEqual(
            decision["reason_codes"],
            ["multiple_mapping_rules"],
        )
        self.assertEqual(
            decision["matched_rule_ids"],
            ["map.hvac_power.activate", "map.hvac_power.activate_duplicate"],
        )


if __name__ == "__main__":
    unittest.main()
