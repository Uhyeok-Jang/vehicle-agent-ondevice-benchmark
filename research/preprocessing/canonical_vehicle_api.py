#!/usr/bin/env python3
"""Validate and deterministically serialize canonical Vehicle API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = RESEARCH_ROOT / "schema" / "vehicle_api_schema.v0.1.0.json"
DEFAULT_REGISTRY = RESEARCH_ROOT / "schema" / "vehicle_api_registry.v0.1.0.json"


class CanonicalValidationError(ValueError):
    """Raised when a payload or registry violates the canonical contract."""


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanonicalValidationError(f"{path}: expected a JSON object")
    return value


def schema_function_names(schema: Mapping[str, Any]) -> set[str]:
    definitions = schema.get("$defs")
    if not isinstance(definitions, Mapping):
        raise CanonicalValidationError("schema has no $defs object")
    canonical = definitions.get("canonical_call")
    if not isinstance(canonical, Mapping):
        raise CanonicalValidationError("schema has no canonical_call definition")
    branches = canonical.get("oneOf")
    if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
        raise CanonicalValidationError("canonical_call.oneOf must be an array")

    names = set()
    for branch in branches:
        reference = branch.get("$ref") if isinstance(branch, Mapping) else None
        prefix = "#/$defs/"
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise CanonicalValidationError("every call branch must be a local $defs ref")
        definition = definitions.get(reference[len(prefix) :])
        try:
            function_name = definition["properties"]["function"]["const"]
        except (KeyError, TypeError) as error:
            raise CanonicalValidationError(
                f"{reference}: missing function const discriminator"
            ) from error
        if not isinstance(function_name, str) or not function_name:
            raise CanonicalValidationError(
                f"{reference}: function discriminator must be a non-empty string"
            )
        if function_name in names:
            raise CanonicalValidationError(
                f"duplicate function discriminator: {function_name}"
            )
        names.add(function_name)
    return names


def validate_registry(
    schema: Mapping[str, Any], registry: Mapping[str, Any]
) -> None:
    Draft202012Validator.check_schema(dict(schema))
    functions = registry.get("functions")
    if not isinstance(functions, Mapping):
        raise CanonicalValidationError("registry.functions must be an object")
    schema_names = schema_function_names(schema)
    registry_names = set(functions)
    if schema_names != registry_names:
        raise CanonicalValidationError(
            "schema/registry function mismatch: "
            f"schema_only={sorted(schema_names - registry_names)}, "
            f"registry_only={sorted(registry_names - schema_names)}"
        )
    for name, specification in functions.items():
        if not isinstance(specification, Mapping):
            raise CanonicalValidationError(f"registry function {name!r} must be an object")
        order = specification.get("argument_order")
        if (
            not isinstance(order, Sequence)
            or isinstance(order, (str, bytes))
            or len(order) != len(set(order))
            or not all(isinstance(item, str) and item for item in order)
        ):
            raise CanonicalValidationError(
                f"registry function {name!r} has invalid argument_order"
            )


def _error_path(error: Any) -> str:
    path = "$"
    for component in error.absolute_path:
        path += f"[{component}]" if isinstance(component, int) else f".{component}"
    return path


def validate_payload(payload: Any, schema: Mapping[str, Any]) -> None:
    validator = Draft202012Validator(dict(schema))
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (
            tuple(str(component) for component in error.absolute_path),
            error.message,
        ),
    )
    if errors:
        preview = "; ".join(
            f"{_error_path(error)}: {error.message}" for error in errors[:3]
        )
        if len(errors) > 3:
            preview += f"; and {len(errors) - 3} more"
        raise CanonicalValidationError(preview)


def _ordered_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _ordered_nested(value[key])
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [_ordered_nested(item) for item in value]
    return value


def canonicalize_payload(
    payload: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    unordered_calls: bool = False,
) -> dict[str, Any]:
    validate_registry(schema, registry)
    validate_payload(payload, schema)
    registered = registry["functions"]
    calls = []
    for call in payload["calls"]:
        function_name = call["function"]
        arguments = call["arguments"]
        argument_order = registered[function_name]["argument_order"]
        unregistered_arguments = sorted(set(arguments) - set(argument_order))
        if unregistered_arguments:
            raise CanonicalValidationError(
                f"{function_name}: arguments absent from registry order: "
                f"{unregistered_arguments}"
            )
        ordered_arguments = {
            name: _ordered_nested(arguments[name])
            for name in argument_order
            if name in arguments
        }
        calls.append(
            {
                "function": function_name,
                "arguments": ordered_arguments,
            }
        )
    if unordered_calls:
        calls.sort(
            key=lambda call: json.dumps(
                call,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    canonical = {"calls": calls}
    validate_payload(canonical, schema)
    return canonical


def canonical_json(
    payload: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    unordered_calls: bool = False,
) -> str:
    canonical = canonicalize_payload(
        payload,
        schema=schema,
        registry=registry,
        unordered_calls=unordered_calls,
    )
    return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))


def _read_payloads(path: str, *, jsonl: bool) -> Iterable[dict[str, Any]]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    if jsonl:
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CanonicalValidationError(
                    f"line {line_number}: expected a JSON object"
                )
            yield value
        return
    value = json.loads(text)
    if not isinstance(value, dict):
        raise CanonicalValidationError("expected a JSON object")
    yield value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON/JSONL path, or - for stdin")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument(
        "--unordered-calls",
        action="store_true",
        help="sort calls for the secondary unordered-multiset diagnostic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = load_json_object(args.schema)
    registry = load_json_object(args.registry)
    validate_registry(schema, registry)
    for payload in _read_payloads(args.input, jsonl=args.jsonl):
        print(
            canonical_json(
                payload,
                schema=schema,
                registry=registry,
                unordered_calls=args.unordered_calls,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
