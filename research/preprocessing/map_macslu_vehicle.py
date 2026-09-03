#!/usr/bin/env python3
"""Deterministically map MAC-SLU vehicle frames to the canonical pilot API."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

import canonical_vehicle_api as canonical


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_SCHEMA = (
    RESEARCH_ROOT / "schema" / "vehicle_mapping_registry_schema.v0.1.0.json"
)
DEFAULT_MAPPING_REGISTRY = (
    RESEARCH_ROOT / "schema" / "macslu_vehicle_mapping.v0.1.0.json"
)
FORBIDDEN_SELECTOR_TERMS = {"query", "split", "id", "example_id", "unit_id"}


class MappingRegistryError(ValueError):
    """Raised when a source mapping registry is invalid or non-deterministic."""


@dataclass(frozen=True)
class SourceSlot:
    ordinal: int
    name: str
    value: str
    pos: tuple[int, int] | None = None


@dataclass(frozen=True)
class RawFrame:
    unit_id: str
    unit_order: int
    semantic_key: str
    domain: str
    intent: str | None
    slots: tuple[SourceSlot, ...]
    structural_errors: tuple[str, ...] = ()


def _nfkc_exact(value: str) -> str:
    """Apply only the registry's exact-match normalization contract."""

    return unicodedata.normalize("NFKC", value)


def _semantic_sort_key(key: str) -> tuple[int, str]:
    normalized = _nfkc_exact(key)
    match = re.search(r"(\d+)$", normalized)
    return (int(match.group(1)), normalized) if match else (10**9, normalized)


def _example_id(revision: str, split: str, source_id: str) -> str:
    return f"macslu:{revision}:{split}:{source_id}"


def adapt_macslu_row(
    row: Mapping[str, Any],
    *,
    revision: str,
    split: str,
    vehicle_domain: str = "车载控制",
) -> list[RawFrame]:
    """Convert a MAC-SLU row into ordered, query-independent vehicle frames."""

    source_id = str(row.get("id", ""))
    example_id = _example_id(revision, split, source_id)
    semantics = row.get("semantics")
    if not isinstance(semantics, Mapping):
        return []

    ordered_frames = sorted(
        semantics.items(),
        key=lambda item: _semantic_sort_key(str(item[0])),
    )
    row_domains: set[str] = set()
    for _, domains in ordered_frames:
        if isinstance(domains, Mapping):
            row_domains.update(str(domain) for domain in domains)
    mixed_row = bool(row_domains - {vehicle_domain})

    output: list[RawFrame] = []
    for semantic_key, domains in ordered_frames:
        if not isinstance(domains, Mapping) or vehicle_domain not in domains:
            continue

        errors: list[str] = []
        normalized_key = _nfkc_exact(str(semantic_key))
        if not re.search(r"\d+$", normalized_key):
            errors.append("source_annotation_conflict")
        if mixed_row:
            errors.append("mixed_domain_example")
        if len(domains) != 1:
            errors.append("multi_domain_per_intent")

        raw_slots = domains.get(vehicle_domain)
        slots: list[SourceSlot] = []
        if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
            errors.append("source_annotation_conflict")
            raw_slots = []
        for ordinal, raw_slot in enumerate(raw_slots):
            if not isinstance(raw_slot, Mapping):
                errors.append("source_annotation_conflict")
                continue
            name = raw_slot.get("name")
            value = raw_slot.get("value")
            if not isinstance(name, str) or not isinstance(value, str):
                errors.append("source_annotation_conflict")
                continue
            raw_pos = raw_slot.get("pos")
            pos = None
            if (
                isinstance(raw_pos, Sequence)
                and not isinstance(raw_pos, (str, bytes))
                and len(raw_pos) == 2
                and all(isinstance(component, int) for component in raw_pos)
            ):
                pos = (raw_pos[0], raw_pos[1])
            slots.append(
                SourceSlot(
                    ordinal=ordinal,
                    name=_nfkc_exact(name),
                    value=_nfkc_exact(value),
                    pos=pos,
                )
            )

        slot_counts = Counter(slot.name for slot in slots)
        if any(count != 1 for count in slot_counts.values()):
            errors.append("source_annotation_conflict")
        intent_slots = [slot for slot in slots if slot.name == "intent"]
        intent = intent_slots[0].value if len(intent_slots) == 1 else None
        if len(intent_slots) != 1:
            errors.append("source_annotation_conflict")

        output.append(
            RawFrame(
                unit_id=f"{example_id}:semantic:{normalized_key}",
                unit_order=len(output),
                semantic_key=normalized_key,
                domain=vehicle_domain,
                intent=intent,
                slots=tuple(slots),
                structural_errors=tuple(dict.fromkeys(errors)),
            )
        )
    return output


def _schema_error_preview(errors: Sequence[Any]) -> str:
    return "; ".join(error.message for error in errors[:3])


def validate_mapping_registry(
    registry: Mapping[str, Any],
    *,
    mapping_schema: Mapping[str, Any],
    canonical_schema: Mapping[str, Any],
    canonical_registry: Mapping[str, Any],
) -> None:
    """Validate structure plus deterministic constraints not expressible in JSON Schema."""

    Draft202012Validator.check_schema(dict(mapping_schema))
    errors = sorted(
        Draft202012Validator(dict(mapping_schema)).iter_errors(registry),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        raise MappingRegistryError(_schema_error_preview(errors))
    canonical.validate_registry(canonical_schema, canonical_registry)

    all_ids: list[str] = []
    aliases: set[tuple[int, str, str, str]] = set()
    for normalizer in registry["normalizers"]:
        all_ids.append(normalizer["id"])
        requirements = json.dumps(
            normalizer.get("requires", {}),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for alias in normalizer["input"]["values"]:
            if alias != _nfkc_exact(alias):
                raise MappingRegistryError(
                    f"{normalizer['id']}: alias is not canonical NFKC text"
                )
            signature = (
                normalizer["stage"],
                normalizer["input"]["path"],
                alias,
                requirements,
            )
            if signature in aliases:
                raise MappingRegistryError(
                    f"overlapping exact normalizer selector: {signature}"
                )
            aliases.add(signature)

    canonical_functions = canonical.schema_function_names(canonical_schema)
    registered_functions = set(canonical_registry["functions"])
    for rule in registry["rules"]:
        all_ids.append(rule["id"])
        function_name = rule["emit"]["function"]
        if function_name not in canonical_functions:
            raise MappingRegistryError(
                f"{rule['id']}: function absent from canonical schema: {function_name}"
            )
        argument_names = set(rule["emit"]["arguments"])
        allowed_arguments = set(
            canonical_registry["functions"][function_name]["argument_order"]
        )
        if argument_names - allowed_arguments:
            raise MappingRegistryError(
                f"{rule['id']}: unregistered arguments: "
                f"{sorted(argument_names - allowed_arguments)}"
            )
        match_fields = set(rule["match"].get("equals", {}))
        match_fields.update(rule["match"].get("in", {}))
        match_fields.update(rule["match"].get("present", []))
        emission_fields = {
            binding["from"]
            for binding in rule["emit"]["arguments"].values()
            if "from" in binding
        }
        missing_consumption = (match_fields | emission_fields) - set(rule["consume"])
        if missing_consumption:
            raise MappingRegistryError(
                f"{rule['id']}: matched/emitted fields are not consumed: "
                f"{sorted(missing_consumption)}"
            )

    valid_reasons = registry["reason_codes"]
    for outcome in registry["outcome_rules"]:
        all_ids.append(outcome["id"])
        if outcome["reason_code"] not in valid_reasons[outcome["status"]]:
            raise MappingRegistryError(
                f"{outcome['id']}: reason is not registered for "
                f"{outcome['status']}"
            )
    if len(all_ids) != len(set(all_ids)):
        raise MappingRegistryError("registry ids must be globally unique")
    if canonical_functions != registered_functions:
        raise MappingRegistryError("canonical schema and registry functions differ")

    serialized = json.dumps(registry, ensure_ascii=False)
    for forbidden in FORBIDDEN_SELECTOR_TERMS:
        if f'"path": "{forbidden}"' in serialized:
            raise MappingRegistryError(
                f"source metadata cannot be used as a selector: {forbidden}"
            )


def _matches(values: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    for field, expected in condition.get("equals", {}).items():
        if values.get(field) != expected:
            return False
    for field, options in condition.get("in", {}).items():
        if values.get(field) not in options:
            return False
    if any(field not in values for field in condition.get("present", [])):
        return False
    if any(field in values for field in condition.get("absent", [])):
        return False
    return True


class MacsluVehicleMapper:
    """Apply the declarative v0.1 source registry without heuristic fallbacks."""

    def __init__(
        self,
        *,
        registry: Mapping[str, Any] | None = None,
        mapping_schema: Mapping[str, Any] | None = None,
        canonical_schema: Mapping[str, Any] | None = None,
        canonical_registry: Mapping[str, Any] | None = None,
    ) -> None:
        self.registry = dict(
            registry
            if registry is not None
            else canonical.load_json_object(DEFAULT_MAPPING_REGISTRY)
        )
        self.mapping_schema = dict(
            mapping_schema
            if mapping_schema is not None
            else canonical.load_json_object(DEFAULT_MAPPING_SCHEMA)
        )
        self.canonical_schema = dict(
            canonical_schema
            if canonical_schema is not None
            else canonical.load_json_object(canonical.DEFAULT_SCHEMA)
        )
        self.canonical_registry = dict(
            canonical_registry
            if canonical_registry is not None
            else canonical.load_json_object(canonical.DEFAULT_REGISTRY)
        )
        validate_mapping_registry(
            self.registry,
            mapping_schema=self.mapping_schema,
            canonical_schema=self.canonical_schema,
            canonical_registry=self.canonical_registry,
        )

    @staticmethod
    def _path_values(
        frame: RawFrame, path: str
    ) -> list[tuple[str, tuple[int, ...]]]:
        if path == "intent":
            matches = [
                slot for slot in frame.slots if slot.name == "intent"
            ]
        else:
            slot_name = path.removeprefix("slots.")
            matches = [slot for slot in frame.slots if slot.name == slot_name]
        return [
            (slot.value, (slot.ordinal,))
            for slot in matches
        ]

    def _normalize(
        self, frame: RawFrame
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
        normalized: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        for stage in (1, 2):
            snapshot = copy.deepcopy(normalized)
            candidates: dict[str, list[tuple[Any, str, str, tuple[int, ...]]]] = {}
            for normalizer in self.registry["normalizers"]:
                if normalizer["stage"] != stage:
                    continue
                if any(
                    snapshot.get(field) != expected
                    for field, expected in normalizer.get("requires", {}).items()
                ):
                    continue
                aliases = set(normalizer["input"]["values"])
                for value, ordinals in self._path_values(
                    frame, normalizer["input"]["path"]
                ):
                    if _nfkc_exact(value) not in aliases:
                        continue
                    field, emitted = next(iter(normalizer["emit"].items()))
                    candidates.setdefault(field, []).append(
                        (
                            copy.deepcopy(emitted),
                            normalizer["id"],
                            normalizer["input"]["path"],
                            ordinals,
                        )
                    )
            for field, matches in candidates.items():
                if len(matches) != 1 or field in normalized:
                    return normalized, trace, "multiple_normalizations"
                emitted, normalizer_id, source_path, ordinals = matches[0]
                normalized[field] = emitted
                trace.append(
                    {
                        "normalizer_id": normalizer_id,
                        "source_path": source_path,
                        "source_slot_ordinals": list(ordinals),
                        "output_field": field,
                    }
                )
        return normalized, trace, None

    def _result(
        self,
        frame: RawFrame,
        normalized: Mapping[str, Any],
        trace: Sequence[Mapping[str, Any]],
        *,
        status: str,
        reason_codes: Sequence[str] = (),
        matched_rule_ids: Sequence[str] = (),
        call: Mapping[str, Any] | None = None,
        consumed_slot_ordinals: Sequence[int] = (),
    ) -> dict[str, Any]:
        registered = set(self.registry["reason_codes"][status])
        if any(reason not in registered for reason in reason_codes):
            raise MappingRegistryError(
                f"unregistered {status} reason: {list(reason_codes)}"
            )
        return {
            "unit_id": frame.unit_id,
            "unit_order": frame.unit_order,
            "mapping_registry_version": self.registry["registry_version"],
            "normalized": copy.deepcopy(dict(normalized)),
            "decision": {
                "status": status,
                "reason_codes": list(dict.fromkeys(reason_codes)),
                "matched_rule_ids": list(matched_rule_ids),
                "call": copy.deepcopy(call),
                "consumed_slot_ordinals": sorted(set(consumed_slot_ordinals)),
                "trace": copy.deepcopy(list(trace)),
            },
        }

    def _outcome(
        self,
        frame: RawFrame,
        normalized: Mapping[str, Any],
        trace: Sequence[Mapping[str, Any]],
        phase: str,
    ) -> dict[str, Any] | None:
        matches = [
            outcome
            for outcome in self.registry["outcome_rules"]
            if outcome["phase"] == phase
            and _matches(normalized, outcome["match"])
        ]
        if len(matches) > 1:
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=["multiple_outcome_rules"],
                matched_rule_ids=[outcome["id"] for outcome in matches],
            )
        if not matches:
            return None
        outcome = matches[0]
        return self._result(
            frame,
            normalized,
            trace,
            status=outcome["status"],
            reason_codes=[outcome["reason_code"]],
            matched_rule_ids=[outcome["id"]],
        )

    @staticmethod
    def _emitted_call(
        rule: Mapping[str, Any], normalized: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        arguments: dict[str, Any] = {}
        for name, binding in rule["emit"]["arguments"].items():
            if "const" in binding:
                arguments[name] = copy.deepcopy(binding["const"])
                continue
            field = binding["from"]
            if field in normalized:
                arguments[name] = copy.deepcopy(normalized[field])
            elif not binding.get("optional", False):
                return None
        return {
            "function": rule["emit"]["function"],
            "arguments": arguments,
        }

    def map_frame(
        self,
        frame: RawFrame,
        *,
        inherited_entity: str | None = None,
        context_source_unit_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        if frame.structural_errors:
            return self._result(
                frame,
                {},
                [],
                status="ambiguous",
                reason_codes=frame.structural_errors,
            )

        allowed_slots = set(self.registry["executable_slot_names"])
        if any(slot.name not in allowed_slots for slot in frame.slots):
            return self._result(
                frame,
                {},
                [],
                status="ambiguous",
                reason_codes=["unexpected_source_slot"],
            )

        normalized, trace, normalization_error = self._normalize(frame)
        if normalization_error is not None:
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=[normalization_error],
            )

        if inherited_entity is not None and "entity" not in normalized:
            normalized["entity"] = inherited_entity
            trace.append(
                {
                    "normalizer_id": "context.entity.unique_in_row",
                    "source_path": "row_context.entity",
                    "source_slot_ordinals": [],
                    "source_unit_ids": list(context_source_unit_ids),
                    "output_field": "entity",
                }
            )

        early_outcome = self._outcome(
            frame,
            normalized,
            trace,
            "pre_normalization_errors",
        )
        if early_outcome is not None:
            return early_outcome

        rules = [
            rule
            for rule in self.registry["rules"]
            if _matches(normalized, rule["match"])
        ]
        if len(rules) > 1:
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=["multiple_mapping_rules"],
                matched_rule_ids=[rule["id"] for rule in rules],
            )

        traced_ordinals = {
            ordinal
            for item in trace
            for ordinal in item["source_slot_ordinals"]
        }
        unnormalized_ordinals = {
            slot.ordinal for slot in frame.slots
        } - traced_ordinals

        if not rules:
            if unnormalized_ordinals:
                return self._result(
                    frame,
                    normalized,
                    trace,
                    status="ambiguous",
                    reason_codes=["unrecognized_source_value"],
                )
            late_outcome = self._outcome(
                frame,
                normalized,
                trace,
                "post_mapping",
            )
            if late_outcome is not None:
                return late_outcome
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=["no_mapping_rule"],
            )

        rule = rules[0]
        consumed_fields = set(rule["consume"])
        consumed_ordinals = {
            ordinal
            for item in trace
            if item["output_field"] in consumed_fields
            for ordinal in item["source_slot_ordinals"]
        }
        all_ordinals = {slot.ordinal for slot in frame.slots}
        if unnormalized_ordinals:
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=["unrecognized_source_value"],
                matched_rule_ids=[rule["id"]],
                consumed_slot_ordinals=consumed_ordinals,
            )
        if consumed_ordinals != all_ordinals:
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=["unconsumed_executable_slot"],
                matched_rule_ids=[rule["id"]],
                consumed_slot_ordinals=consumed_ordinals,
            )

        call = self._emitted_call(rule, normalized)
        if call is None:
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=["missing_emission_value"],
                matched_rule_ids=[rule["id"]],
                consumed_slot_ordinals=consumed_ordinals,
            )
        try:
            canonical_payload = canonical.canonicalize_payload(
                {"calls": [call]},
                schema=self.canonical_schema,
                registry=self.canonical_registry,
            )
        except canonical.CanonicalValidationError:
            return self._result(
                frame,
                normalized,
                trace,
                status="ambiguous",
                reason_codes=["canonical_schema_validation_failed"],
                matched_rule_ids=[rule["id"]],
                consumed_slot_ordinals=consumed_ordinals,
            )
        canonical_call = canonical_payload["calls"][0]
        return self._result(
            frame,
            normalized,
            trace,
            status="mapped",
            matched_rule_ids=[rule["id"]],
            call=canonical_call,
            consumed_slot_ordinals=consumed_ordinals,
        )

    def map_row(
        self,
        row: Mapping[str, Any],
        *,
        revision: str,
        split: str,
    ) -> dict[str, Any]:
        frames = adapt_macslu_row(
            row,
            revision=revision,
            split=split,
            vehicle_domain=self.registry["vehicle_domain"],
        )
        allowed_slots = set(self.registry["executable_slot_names"])
        entity_sources: dict[str, list[str]] = {}
        normalized_frames: list[dict[str, Any]] = []
        for frame in frames:
            normalized, _, normalization_error = self._normalize(frame)
            normalized_frames.append(normalized)
            if (
                frame.structural_errors
                or normalization_error is not None
                or any(slot.name not in allowed_slots for slot in frame.slots)
                or "entity" not in normalized
            ):
                continue
            entity_sources.setdefault(str(normalized["entity"]), []).append(
                frame.unit_id
            )

        inherited_entity = None
        context_source_unit_ids: tuple[str, ...] = ()
        if set(entity_sources) == {"hvac"}:
            inherited_entity = "hvac"
            context_source_unit_ids = tuple(entity_sources["hvac"])

        units = []
        for frame, normalized in zip(frames, normalized_frames, strict=True):
            use_context = (
                inherited_entity is not None
                and normalized.get("intent_class") == "information_fragment"
                and "entity" not in normalized
            )
            units.append(
                self.map_frame(
                    frame,
                    inherited_entity=inherited_entity if use_context else None,
                    context_source_unit_ids=(
                        context_source_unit_ids if use_context else ()
                    ),
                )
            )
        payload = None
        if units and all(
            unit["decision"]["status"] == "mapped" for unit in units
        ):
            payload = canonical.canonicalize_payload(
                {
                    "calls": [
                        unit["decision"]["call"]
                        for unit in units
                    ]
                },
                schema=self.canonical_schema,
                registry=self.canonical_registry,
            )
        return {
            "example_id": _example_id(str(revision), str(split), str(row.get("id", ""))),
            "mapping_registry_version": self.registry["registry_version"],
            "units": units,
            "canonical_payload": payload,
        }
