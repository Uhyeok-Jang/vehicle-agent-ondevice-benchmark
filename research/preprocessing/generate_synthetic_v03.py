#!/usr/bin/env python3
"""Generate deterministic, gold-first Korean Vehicle API augmentation candidates.

Canonical calls are sampled from the frozen v0.1.0 API before any Korean text is
rendered.  The resulting file is a candidate artifact; build_augmented_v03.py is
responsible for the release validation and final group-aware split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from canonical_vehicle_api import (
        canonicalize_payload,
        load_json_object,
        validate_registry,
    )
except ModuleNotFoundError:  # Supports namespace-package imports in tests.
    from .canonical_vehicle_api import (
        canonicalize_payload,
        load_json_object,
        validate_registry,
    )


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_ROOT = RESEARCH_ROOT / "data" / "processed" / "macslu_korean_v0.2"
DEFAULT_OUTPUT = RESEARCH_ROOT / "data" / "synthetic" / "synthetic_candidates_v0.3.jsonl"
DEFAULT_REPORT = (
    RESEARCH_ROOT / "data" / "synthetic" / "synthetic_generation_report_v0.3.json"
)
DEFAULT_SCHEMA = RESEARCH_ROOT / "schema" / "vehicle_api_schema.v0.1.0.json"
DEFAULT_REGISTRY = RESEARCH_ROOT / "schema" / "vehicle_api_registry.v0.1.0.json"

DEFAULT_SEED = 20260905
GENERATOR = "astra_vehicle_aug_v0.3"
GENERATOR_VERSION = "0.3.0"
CALL_COUNT_TARGETS = {1: 500, 2: 800, 3: 600, 4: 400}

FUNCTIONS = (
    "set_hvac_power",
    "set_hvac_temperature",
    "set_hvac_fan_speed",
    "set_window_position",
    "set_sunroof_position",
    "set_sunshade_position",
    "set_seat_climate",
    "set_seat_massage",
)

FUNCTION_FAMILY = {
    "set_hvac_power": "HVAC",
    "set_hvac_temperature": "HVAC",
    "set_hvac_fan_speed": "HVAC",
    "set_window_position": "Aperture",
    "set_sunroof_position": "Aperture",
    "set_sunshade_position": "Aperture",
    "set_seat_climate": "Seat",
    "set_seat_massage": "Seat",
}
FAMILY_ORDER = ("HVAC", "Aperture", "Seat")

HVAC_ZONES: tuple[str | None, ...] = (
    None,
    "driver",
    "front_passenger",
    "rear",
    "all",
)
WINDOW_ZONES = (
    "driver",
    "front_passenger",
    "rear_left",
    "rear_right",
    "front_row",
    "rear_row",
    "left_side",
    "right_side",
    "all",
)
SEAT_ZONES = (
    "driver",
    "front_passenger",
    "rear_left",
    "rear_right",
    "rear_row",
    "all",
)
ATOMIC_HVAC_ZONES = frozenset(("driver", "front_passenger", "rear"))
ATOMIC_WINDOW_ZONES = frozenset(
    ("driver", "front_passenger", "rear_left", "rear_right")
)
ATOMIC_SEAT_ZONES = ATOMIC_WINDOW_ZONES

RELATIVE_MAGNITUDES = ("default", "small", "medium", "large")
ABSOLUTE_PERCENT_GRID = (0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100)
RELATIVE_PERCENT_GRID = (5, 10, 15, 20, 25, 30, 40, 50)
TEMPERATURE_GRID = tuple(
    half_degree // 2 if half_degree % 2 == 0 else half_degree / 2
    for half_degree in range(32, 65)
)
FAN_LEVELS = tuple(range(1, 9))
SEAT_LEVELS = (1, 2, 3)

WINDOW_ATOMS = {
    "driver": frozenset(("driver",)),
    "front_passenger": frozenset(("front_passenger",)),
    "rear_left": frozenset(("rear_left",)),
    "rear_right": frozenset(("rear_right",)),
    "front_row": frozenset(("driver", "front_passenger")),
    "rear_row": frozenset(("rear_left", "rear_right")),
    "left_side": frozenset(("driver", "rear_left")),
    "right_side": frozenset(("front_passenger", "rear_right")),
    "all": ATOMIC_WINDOW_ZONES,
}
SEAT_ATOMS = {
    "driver": frozenset(("driver",)),
    "front_passenger": frozenset(("front_passenger",)),
    "rear_left": frozenset(("rear_left",)),
    "rear_right": frozenset(("rear_right",)),
    "rear_row": frozenset(("rear_left", "rear_right")),
    "all": ATOMIC_SEAT_ZONES,
}
HVAC_ATOMS = {
    None: ATOMIC_HVAC_ZONES,
    "driver": frozenset(("driver",)),
    "front_passenger": frozenset(("front_passenger",)),
    "rear": frozenset(("rear",)),
    "all": ATOMIC_HVAC_ZONES,
}

UNREPEATABLE_FUNCTIONS = frozenset(
    ("set_sunroof_position", "set_sunshade_position")
)
REPEATABLE_FUNCTIONS = tuple(
    function for function in FUNCTIONS if function not in UNREPEATABLE_FUNCTIONS
)

WHITESPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
PLACEHOLDER_RE = re.compile(r"[{}<>$]|\b(?:TODO|None|null)\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Apply the exact v0.2 whitespace-only normalization policy."""

    return WHITESPACE_RE.sub(" ", text.strip())

def language_issue(text: str) -> str | None:
    normalized = normalize_text(text)
    if len(normalized) < 6:
        return "fewer_than_6_normalized_characters"
    if len(HANGUL_RE.findall(normalized)) < 4:
        return "fewer_than_4_hangul_syllables"
    if CJK_RE.search(normalized):
        return "contains_cjk_ideograph"
    if PLACEHOLDER_RE.search(normalized):
        return "contains_placeholder_artifact"
    return None



def stable_seed(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def canonical_calls_key(calls: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        calls,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_original_pool(root: Path) -> tuple[dict[str, str], set[str]]:
    """Return normalized utterances and ordered gold signatures from v0.2."""

    utterances: dict[str, str] = {}
    signatures: set[str] = set()
    paths = [root / f"{split}.jsonl" for split in ("train", "validation", "test")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing original split: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                normalized = normalize_text(row["utterance_ko"])
                gold = canonical_calls_key(row["canonical_calls"])
                previous = utterances.get(normalized)
                if previous is not None and previous != gold:
                    raise RuntimeError(
                        f"original conflicting utterance at {path}:{line_number}: "
                        f"{normalized!r}"
                    )
                utterances[normalized] = gold
                signatures.add(gold)
    return utterances, signatures


def load_original_utterances(root: Path) -> dict[str, str]:
    """Compatibility helper for callers that need only normalized text keys."""

    return load_original_pool(root)[0]


def family_pattern(functions: Sequence[str]) -> str:
    present = {FUNCTION_FAMILY[function] for function in functions}
    return "+".join(family for family in FAMILY_ORDER if family in present)


def _eligible_function(
    function: str,
    current: Sequence[str],
) -> bool:
    return function not in current


def _least_used_choice(
    usage: Counter[str],
    current: Sequence[str],
    rng: random.Random,
    *,
    allowed: Iterable[str] = FUNCTIONS,
) -> str:
    eligible = [
        function
        for function in allowed
        if _eligible_function(function, current)
    ]
    if not eligible:
        raise RuntimeError(f"no compatible function for partial skeleton {current}")
    minimum = min(usage[function] for function in eligible)
    tied = [function for function in eligible if usage[function] == minimum]
    return tied[rng.randrange(len(tied))]


REQUIRED_SKELETONS: dict[int, tuple[tuple[str, ...], ...]] = {
    1: tuple((function,) for function in FUNCTIONS),
    2: (
        ("set_hvac_power", "set_window_position"),
        ("set_hvac_temperature", "set_seat_climate"),
        ("set_sunroof_position", "set_seat_massage"),
        ("set_hvac_power", "set_hvac_fan_speed"),
        ("set_window_position", "set_sunshade_position"),
        ("set_seat_climate", "set_seat_massage"),
        ("set_hvac_power", "set_hvac_power"),
        ("set_hvac_temperature", "set_hvac_temperature"),
        ("set_hvac_fan_speed", "set_hvac_fan_speed"),
        ("set_window_position", "set_window_position"),
        ("set_seat_climate", "set_seat_climate"),
        ("set_seat_massage", "set_seat_massage"),
    ),
    3: (
        ("set_hvac_power", "set_window_position", "set_seat_climate"),
        ("set_seat_massage", "set_sunroof_position", "set_hvac_temperature"),
        ("set_hvac_power", "set_hvac_temperature", "set_window_position"),
        ("set_hvac_fan_speed", "set_seat_climate", "set_seat_massage"),
        ("set_window_position", "set_sunroof_position", "set_seat_climate"),
        ("set_hvac_power", "set_hvac_temperature", "set_hvac_fan_speed"),
        ("set_window_position", "set_sunroof_position", "set_sunshade_position"),
        ("set_seat_climate", "set_seat_massage", "set_seat_climate"),
    ),
    4: (
        (
            "set_hvac_power",
            "set_window_position",
            "set_seat_climate",
            "set_hvac_temperature",
        ),
        (
            "set_seat_massage",
            "set_sunshade_position",
            "set_hvac_fan_speed",
            "set_window_position",
        ),
        (
            "set_hvac_power",
            "set_hvac_temperature",
            "set_window_position",
            "set_sunroof_position",
        ),
        (
            "set_hvac_power",
            "set_hvac_fan_speed",
            "set_seat_climate",
            "set_seat_massage",
        ),
        (
            "set_window_position",
            "set_sunroof_position",
            "set_seat_climate",
            "set_seat_massage",
        ),
    ),
}


def build_function_skeletons(
    targets: Mapping[int, int],
    seed: int,
) -> dict[int, list[tuple[str, ...]]]:
    """Build balanced ordered function skeletons before sampling arguments."""

    output: dict[int, list[tuple[str, ...]]] = {}
    global_usage: Counter[str] = Counter()
    repeat_usage: Counter[str] = Counter()
    for call_count in sorted(targets):
        target = int(targets[call_count])
        if call_count not in (1, 2, 3, 4) or target < 0:
            raise ValueError(f"invalid call-count target: {call_count}={target}")
        required = list(REQUIRED_SKELETONS.get(call_count, ()))
        if call_count == 1:
            # A power call has only two signatures novel to v0.2. A
            # uniform single-call stratum would violate signature uniqueness.
            # Multi-call strata subsequently restore global function balance.
            single_targets = {
                "set_hvac_power": 2,
                "set_hvac_temperature": 75,
                "set_hvac_fan_speed": 65,
                "set_window_position": 103,
                "set_sunroof_position": 38,
                "set_sunshade_position": 37,
                "set_seat_climate": 100,
                "set_seat_massage": 80,
            }
            if target != sum(single_targets.values()):
                required = required[:target]
            else:
                required = [
                    (function,)
                    for function in FUNCTIONS
                    for _ in range(single_targets[function])
                ]
                random.Random(stable_seed(seed, "single-skeleton-order")).shuffle(
                    required
                )
        if len(required) > target:
            required = required[:target]
        skeletons: list[tuple[str, ...]] = []
        for skeleton in required:
            skeletons.append(skeleton)
            global_usage.update(skeleton)
            for function, occurrences in Counter(skeleton).items():
                if occurrences > 1:
                    repeat_usage[function] += 1

        rng = random.Random(stable_seed(seed, "function-skeleton", call_count))
        while len(skeletons) < target:
            ordinal = len(skeletons)
            current: list[str] = []

            # Deliberately retain a modest number of same-function compositions.
            # They are rendered only after distinct, non-overlapping zones pass
            # the effective-resource check below.
            force_repeat = (
                call_count >= 2
                and ordinal % 19 == 7
                and ordinal < target - 16
            )
            if force_repeat:
                repeated = _least_used_choice(
                    repeat_usage,
                    current,
                    rng,
                    allowed=REPEATABLE_FUNCTIONS,
                )
                current.extend((repeated, repeated))
                global_usage.update((repeated, repeated))
                repeat_usage[repeated] += 1

            while len(current) < call_count:
                function = _least_used_choice(global_usage, current, rng)
                current.append(function)
                global_usage[function] += 1

            # Vary where a repeated pair appears without changing counts.
            if force_repeat and call_count > 2:
                rng.shuffle(current)
            skeletons.append(tuple(current))

        output[call_count] = skeletons
    if targets == CALL_COUNT_TARGETS:
        minimum = min(global_usage.values())
        maximum = max(global_usage.values())
        if minimum == 0 or maximum / minimum > 1.10:
            raise RuntimeError(
                "release function schedule exceeds max/min <= 1.10: "
                f"{dict(sorted(global_usage.items()))}"
            )
    return output


def _target_catalog(
    *,
    named_values: Sequence[str],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = [
        {"kind": "named", "value": value} for value in named_values
    ]
    targets.extend(
        {"kind": "absolute_percent", "value": value}
        for value in ABSOLUTE_PERCENT_GRID
    )
    targets.extend(
        {"kind": "relative", "direction": direction, "magnitude": magnitude}
        for direction in ("open", "close")
        for magnitude in RELATIVE_MAGNITUDES
    )
    targets.extend(
        {"kind": "relative_percent", "direction": direction, "value": value}
        for direction in ("open", "close")
        for value in RELATIVE_PERCENT_GRID
    )
    return targets


def _seat_settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = [
        {"kind": "state", "value": state} for state in ("on", "off")
    ]
    settings.extend(
        {"kind": "absolute_level", "value": value} for value in SEAT_LEVELS
    )
    settings.extend(
        {"kind": "relative", "direction": direction, "magnitude": magnitude}
        for direction in ("increase", "decrease")
        for magnitude in RELATIVE_MAGNITUDES
    )
    settings.extend(
        {"kind": "extreme", "value": value} for value in ("min", "max")
    )
    return settings


def build_call_catalogs() -> dict[str, list[dict[str, Any]]]:
    """Enumerate the documented finite augmentation surface for every function."""

    catalogs: dict[str, list[dict[str, Any]]] = {function: [] for function in FUNCTIONS}

    for zone in HVAC_ZONES:
        for state in ("on", "off"):
            arguments: dict[str, Any] = {"state": state}
            if zone is not None:
                arguments["zone"] = zone
            catalogs["set_hvac_power"].append(
                {"function": "set_hvac_power", "arguments": arguments}
            )

        temperature_targets: list[dict[str, Any]] = [
            {"kind": "absolute", "value": value, "unit": "celsius"}
            for value in TEMPERATURE_GRID
        ]
        temperature_targets.extend(
            {"kind": "relative", "direction": direction, "magnitude": magnitude}
            for direction in ("increase", "decrease")
            for magnitude in RELATIVE_MAGNITUDES
        )
        temperature_targets.extend(
            {"kind": "extreme", "value": value} for value in ("min", "max")
        )
        for target in temperature_targets:
            arguments = {"target": target}
            if zone is not None:
                arguments["zone"] = zone
            catalogs["set_hvac_temperature"].append(
                {"function": "set_hvac_temperature", "arguments": arguments}
            )

        fan_targets: list[dict[str, Any]] = [
            {"kind": "absolute", "value": value, "unit": "level"}
            for value in FAN_LEVELS
        ]
        fan_targets.extend(
            {"kind": "relative", "direction": direction, "magnitude": magnitude}
            for direction in ("increase", "decrease")
            for magnitude in RELATIVE_MAGNITUDES
        )
        fan_targets.extend(
            {"kind": "extreme", "value": value} for value in ("min", "max")
        )
        for target in fan_targets:
            arguments = {"target": target}
            if zone is not None:
                arguments["zone"] = zone
            catalogs["set_hvac_fan_speed"].append(
                {"function": "set_hvac_fan_speed", "arguments": arguments}
            )

    for zone in WINDOW_ZONES:
        for target in _target_catalog(named_values=("open", "closed", "vent")):
            catalogs["set_window_position"].append(
                {
                    "function": "set_window_position",
                    "arguments": {"zone": zone, "target": target},
                }
            )

    for target in _target_catalog(named_values=("open", "closed", "vent")):
        catalogs["set_sunroof_position"].append(
            {"function": "set_sunroof_position", "arguments": {"target": target}}
        )

    for target in _target_catalog(named_values=("open", "closed")):
        catalogs["set_sunshade_position"].append(
            {"function": "set_sunshade_position", "arguments": {"target": target}}
        )

    settings = _seat_settings()
    for zone in SEAT_ZONES:
        for feature in ("heating", "ventilation"):
            for setting in settings:
                catalogs["set_seat_climate"].append(
                    {
                        "function": "set_seat_climate",
                        "arguments": {
                            "zone": zone,
                            "feature": feature,
                            "setting": setting,
                        },
                    }
                )
        for setting in settings:
            catalogs["set_seat_massage"].append(
                {
                    "function": "set_seat_massage",
                    "arguments": {"zone": zone, "setting": setting},
                }
            )

    if any(not catalogs[function] for function in FUNCTIONS):
        raise RuntimeError("at least one function catalog is empty")
    return catalogs


def _resource_atoms(call: Mapping[str, Any]) -> frozenset[str]:
    function = call["function"]
    arguments = call["arguments"]
    if function.startswith("set_hvac_"):
        return HVAC_ATOMS[arguments.get("zone")]
    if function == "set_window_position":
        return WINDOW_ATOMS[arguments["zone"]]
    if function in UNREPEATABLE_FUNCTIONS:
        return frozenset((function,))
    return SEAT_ATOMS[arguments["zone"]]


def calls_share_effective_resource(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    if left["function"] != right["function"]:
        return False
    if left["function"] == "set_seat_climate":
        if left["arguments"]["feature"] != right["arguments"]["feature"]:
            return False
    return bool(_resource_atoms(left) & _resource_atoms(right))


def has_resource_conflict(calls: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        calls_share_effective_resource(calls[left], calls[right])
        for left in range(len(calls))
        for right in range(left + 1, len(calls))
    )


def _supports_repeated_scope(call: Mapping[str, Any]) -> bool:
    function = call["function"]
    zone = call["arguments"].get("zone")
    if function.startswith("set_hvac_"):
        return zone in ATOMIC_HVAC_ZONES
    if function == "set_window_position":
        return zone in ATOMIC_WINDOW_ZONES
    if function in ("set_seat_climate", "set_seat_massage"):
        return zone in ATOMIC_SEAT_ZONES
    return False


class SemanticSampler:
    """Cycle each valid-call catalog without losing rejected conflict choices."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.catalogs = build_call_catalogs()
        self.queues: dict[str, deque[dict[str, Any]]] = {}
        self.cycles: Counter[str] = Counter()
        for function in FUNCTIONS:
            self._refill(function)

    def _refill(self, function: str) -> None:
        cycle = self.cycles[function]
        values = copy.deepcopy(self.catalogs[function])
        random.Random(
            stable_seed(self.seed, "semantic-catalog", function, cycle)
        ).shuffle(values)
        self.queues.setdefault(function, deque()).extend(values)
        self.cycles[function] += 1

    def take(
        self,
        function: str,
        existing: Sequence[Mapping[str, Any]],
        *,
        repeated: bool,
        forced_zone: str | None | object = Ellipsis,
    ) -> dict[str, Any]:
        for _cycle_attempt in range(4):
            if not self.queues[function]:
                self._refill(function)
            queue = self.queues[function]
            for _ in range(len(queue)):
                candidate = queue.popleft()
                arguments = candidate["arguments"]
                if forced_zone is not Ellipsis and arguments.get("zone") != forced_zone:
                    queue.append(candidate)
                    continue
                if repeated and not _supports_repeated_scope(candidate):
                    queue.append(candidate)
                    continue
                if any(
                    calls_share_effective_resource(candidate, call)
                    for call in existing
                ):
                    queue.append(candidate)
                    continue
                return candidate
            self._refill(function)
        raise RuntimeError(
            f"catalog has no non-conflicting {function} call for {existing!r}"
        )

    def sample_calls(
        self,
        skeleton: Sequence[str],
        ordinal: int,
    ) -> tuple[list[dict[str, Any]], frozenset[int]]:
        repeated = Counter(skeleton)
        shared_scope_indices: frozenset[int] = frozenset()
        forced_zones: dict[int, str] = {}

        # One contiguous run of distinct HVAC functions occasionally shares an
        # explicit scope.  Later renderers can safely elide that repeated scope.
        runs: list[list[int]] = []
        run: list[int] = []
        for index, function in enumerate(skeleton):
            if FUNCTION_FAMILY[function] == "HVAC":
                run.append(index)
            else:
                if len(run) >= 2:
                    runs.append(run)
                run = []
        if len(run) >= 2:
            runs.append(run)
        eligible_runs = [
            indices
            for indices in runs
            if len({skeleton[index] for index in indices}) == len(indices)
        ]
        if eligible_runs and ordinal % 4 == 0:
            selected = eligible_runs[0]
            zone = ("driver", "front_passenger", "rear", "all")[
                (ordinal // 4) % 4
            ]
            forced_zones.update({index: zone for index in selected})
            shared_scope_indices = frozenset(selected)

        calls: list[dict[str, Any]] = []
        for index, function in enumerate(skeleton):
            calls.append(
                self.take(
                    function,
                    calls,
                    repeated=repeated[function] > 1,
                    forced_zone=forced_zones.get(index, Ellipsis),
                )
            )
        if has_resource_conflict(calls):
            raise RuntimeError(f"internal resource-conflict bug: {calls!r}")
        return calls, shared_scope_indices


def _josa(text: str, consonant: str, vowel: str) -> str:
    last = text[-1]
    codepoint = ord(last)
    has_final = 0xAC00 <= codepoint <= 0xD7A3 and (codepoint - 0xAC00) % 28 != 0
    return consonant if has_final else vowel


def object_form(text: str) -> str:
    return text + _josa(text, "을", "를")


def _pick(values: Sequence[str], rng: random.Random) -> tuple[str, int]:
    index = rng.randrange(len(values))
    return values[index], index


@dataclass(frozen=True)
class Clause:
    final: str
    link: str
    after: str
    while_form: str
    template_key: str


VERB_FORMS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "on": (
        ("켜 줘", "켜고", "켠 다음", "켜면서"),
        ("작동시켜 줘", "작동시키고", "작동시킨 다음", "작동시키면서"),
    ),
    "off": (
        ("꺼 줘", "끄고", "끈 다음", "끄면서"),
    ),
    "set": (
        ("맞춰 줘", "맞추고", "맞춘 다음", "맞추면서"),
        ("설정해 줘", "설정하고", "설정한 다음", "설정하면서"),
    ),
    "increase": (
        ("높여 줘", "높이고", "높인 다음", "높이면서"),
        ("올려 줘", "올리고", "올린 다음", "올리면서"),
    ),
    "decrease": (
        ("낮춰 줘", "낮추고", "낮춘 다음", "낮추면서"),
        ("내려 줘", "내리고", "내린 다음", "내리면서"),
    ),
    "open": (
        ("열어 줘", "열고", "연 다음", "열면서"),
        ("열어 둬", "열어 두고", "열어 둔 다음", "열어 두면서"),
    ),
    "close": (
        ("닫아 줘", "닫고", "닫은 다음", "닫으면서"),
        ("닫아 둬", "닫아 두고", "닫아 둔 다음", "닫아 두면서"),
    ),
}


def make_clause(
    prefix: str,
    action: str,
    template_key: str,
    rng: random.Random,
) -> Clause:
    forms = VERB_FORMS[action]
    selected = forms[rng.randrange(len(forms))]
    return Clause(
        final=prefix + selected[0],
        link=prefix + selected[1],
        after=prefix + selected[2],
        while_form=prefix + selected[3],
        template_key=template_key,
    )


def _as_also_clause(clause: Clause) -> Clause:
    def replace_particle(text: str) -> str:
        replaced, count = re.subn(
            r"^(.+?)(?:을|를) ",
            lambda match: match.group(1) + "도 ",
            text,
            count=1,
        )
        if count != 1:
            raise ValueError(f"clause has no object particle for 도: {text!r}")
        return replaced

    return Clause(
        final=replace_particle(clause.final),
        link=replace_particle(clause.link),
        after=replace_particle(clause.after),
        while_form=replace_particle(clause.while_form),
        template_key=clause.template_key,
    )


HVAC_SCOPE = {
    None: ("", ""),
    "driver": ("운전석 ", "운전석 쪽 "),
    "front_passenger": ("조수석 ", "조수석 쪽 "),
    "rear": ("뒷좌석 ", "뒤쪽 "),
    "all": ("차량 전체의 ", "전체 "),
}
WINDOW_OBJECTS = {
    "driver": ("운전석 창문",),
    "front_passenger": ("조수석 창문",),
    "rear_left": ("왼쪽 뒷좌석 창문", "뒷좌석 왼쪽 창문"),
    "rear_right": ("오른쪽 뒷좌석 창문", "뒷좌석 오른쪽 창문"),
    "front_row": ("앞좌석 쪽 창문", "앞좌석 창문"),
    "rear_row": ("뒷좌석 쪽 창문", "뒷좌석 창문"),
    "left_side": ("왼쪽 창문", "차량 왼편 창문"),
    "right_side": ("오른쪽 창문", "차량 오른편 창문"),
    "all": ("모든 창문", "전체 창문"),
}
SEAT_OBJECTS = {
    "driver": ("운전석 시트", "운전석 좌석"),
    "front_passenger": ("조수석 시트", "조수석 좌석"),
    "rear_left": ("왼쪽 뒷좌석 시트", "뒷좌석 왼쪽 좌석"),
    "rear_right": ("오른쪽 뒷좌석 시트", "뒷좌석 오른쪽 좌석"),
    "rear_row": ("뒷좌석 시트", "뒷좌석 전체"),
    "all": ("모든 좌석", "전체 좌석"),
}

RELATIVE_ADVERBS = {
    "default": ("",),
    "small": ("조금 ", "살짝 "),
    "medium": ("중간 정도로 ", "적당히 "),
    "large": ("많이 ", "크게 "),
}
APERTURE_RELATIVE_ADVERBS = {
    "default": ("더 ",),
    "small": ("조금 더 ", "살짝 더 "),
    "medium": ("중간 정도 더 ", "적당한 폭으로 더 "),
    "large": ("많이 더 ", "큰 폭으로 더 "),
}


def _format_number(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _hvac_noun(
    zone: str | None,
    kind: str,
    rng: random.Random,
    *,
    omit_scope: bool,
) -> str:
    scope = "" if omit_scope else _pick(HVAC_SCOPE[zone], rng)[0]
    nouns = {
        "power": ("에어컨", "공조기"),
        "temperature": ("에어컨 온도", "공조 온도"),
        "fan": ("에어컨 풍량", "공조기 바람 세기"),
    }[kind]
    return scope + _pick(nouns, rng)[0]


def _render_hvac(
    call: Mapping[str, Any],
    rng: random.Random,
    *,
    omit_scope: bool,
) -> Clause:
    function = call["function"]
    arguments = call["arguments"]
    zone = arguments.get("zone")
    if function == "set_hvac_power":
        noun = _hvac_noun(zone, "power", rng, omit_scope=omit_scope)
        state = arguments["state"]
        return make_clause(
            object_form(noun) + " ",
            state,
            f"set_hvac_power.state_{state}",
            rng,
        )

    kind = "temperature" if function == "set_hvac_temperature" else "fan"
    noun = _hvac_noun(zone, kind, rng, omit_scope=omit_scope)
    target = arguments["target"]
    target_kind = target["kind"]
    prefix = object_form(noun) + " "
    if target_kind == "absolute":
        suffix = (
            f"{_format_number(target['value'])}도로 "
            if kind == "temperature"
            else f"{target['value']}단으로 "
        )
        return make_clause(prefix + suffix, "set", f"{function}.absolute", rng)
    if target_kind == "relative":
        adverb = _pick(RELATIVE_ADVERBS[target["magnitude"]], rng)[0]
        action = "increase" if target["direction"] == "increase" else "decrease"
        return make_clause(
            prefix + adverb,
            action,
            f"{function}.relative",
            rng,
        )
    if target_kind == "extreme":
        extreme = "최저로 " if target["value"] == "min" else "최대로 "
        return make_clause(prefix + extreme, "set", f"{function}.extreme", rng)
    raise ValueError(f"unsupported HVAC target: {target!r}")


def _render_aperture_target(
    object_name: str,
    target: Mapping[str, Any],
    rng: random.Random,
    *,
    function: str,
) -> Clause:
    object_with_particle = object_form(object_name) + " "
    kind = target["kind"]
    if kind == "named":
        value = target["value"]
        if value == "open":
            return make_clause(
                object_with_particle, "open", f"{function}.named_open", rng
            )
        if value == "closed":
            return make_clause(
                object_with_particle, "close", f"{function}.named_closed", rng
            )
        if value == "vent":
            vent = "틸트 환기 위치로 " if function == "set_sunroof_position" else "환기 위치로 "
            return make_clause(
                object_with_particle + vent,
                "set",
                f"{function}.named_vent",
                rng,
            )
    if kind == "absolute_percent":
        prefix = f"{object_name} 열림 정도를 {target['value']}%로 "
        return make_clause(prefix, "set", f"{function}.absolute_percent", rng)
    if kind == "relative":
        adverb = _pick(APERTURE_RELATIVE_ADVERBS[target["magnitude"]], rng)[0]
        action = "open" if target["direction"] == "open" else "close"
        return make_clause(
            object_with_particle + adverb,
            action,
            f"{function}.relative",
            rng,
        )
    if kind == "relative_percent":
        adverb = f"지금보다 {target['value']}% 더 "
        action = "open" if target["direction"] == "open" else "close"
        return make_clause(
            object_with_particle + adverb,
            action,
            f"{function}.relative_percent",
            rng,
        )
    raise ValueError(f"unsupported aperture target: {target!r}")


def _render_seat_setting(
    object_name: str,
    setting: Mapping[str, Any],
    rng: random.Random,
    *,
    function: str,
) -> Clause:
    state_prefix = object_form(object_name) + " "
    kind = setting["kind"]
    if kind == "state":
        state = setting["value"]
        return make_clause(state_prefix, state, f"{function}.state_{state}", rng)
    if kind == "absolute_level":
        return make_clause(
            f"{object_name} 세기를 {setting['value']}단으로 ",
            "set",
            f"{function}.absolute_level",
            rng,
        )
    if kind == "relative":
        adverb = _pick(RELATIVE_ADVERBS[setting["magnitude"]], rng)[0]
        action = "increase" if setting["direction"] == "increase" else "decrease"
        return make_clause(
            f"{object_name} 세기를 " + adverb,
            action,
            f"{function}.relative",
            rng,
        )
    if kind == "extreme":
        extreme = "최저로 " if setting["value"] == "min" else "최대로 "
        return make_clause(
            f"{object_name} 세기를 " + extreme,
            "set",
            f"{function}.extreme",
            rng,
        )
    raise ValueError(f"unsupported seat setting: {setting!r}")


def render_call(
    call: Mapping[str, Any],
    rng: random.Random,
    *,
    omit_scope: bool = False,
) -> Clause:
    function = call["function"]
    arguments = call["arguments"]
    if function.startswith("set_hvac_"):
        return _render_hvac(call, rng, omit_scope=omit_scope)
    if function == "set_window_position":
        object_name = _pick(WINDOW_OBJECTS[arguments["zone"]], rng)[0]
        return _render_aperture_target(
            object_name, arguments["target"], rng, function=function
        )
    if function == "set_sunroof_position":
        return _render_aperture_target(
            "선루프", arguments["target"], rng, function=function
        )
    if function == "set_sunshade_position":
        return _render_aperture_target(
            "선셰이드", arguments["target"], rng, function=function
        )
    if function in ("set_seat_climate", "set_seat_massage"):
        seat = _pick(SEAT_OBJECTS[arguments["zone"]], rng)[0]
        if function == "set_seat_climate":
            feature = "열선" if arguments["feature"] == "heating" else "통풍"
            separator = "의 " if rng.randrange(2) else " "
            object_name = seat + separator + feature
        else:
            separator = "의 " if rng.randrange(2) else " "
            object_name = seat + separator + "마사지"
        return _render_seat_setting(
            object_name, arguments["setting"], rng, function=function
        )
    raise ValueError(f"unknown function: {function}")


def _final_style(text: str, style: int) -> str:
    ending = next((value for value in (" 줘", " 둬") if text.endswith(value)), None)
    if ending is None:
        raise ValueError(f"unsupported clause final: {text!r}")
    if style == 0:
        return text
    if style == 1:
        return text[:-2] + ending.strip()
    if style == 2:
        polite = " 주세요" if ending == " 줘" else " 두세요"
        return text[:-2] + polite
    if style == 3:
        return text[:-2]
    raise ValueError(f"unknown ending style: {style}")


def render_utterance(
    calls: Sequence[Mapping[str, Any]],
    rng: random.Random,
    shared_scope_indices: frozenset[int] = frozenset(),
) -> tuple[str, str]:
    if not 1 <= len(calls) <= 4:
        raise ValueError(f"rendering supports 1-4 calls, got {len(calls)}")
    first_shared = min(shared_scope_indices) if shared_scope_indices else None
    clauses = [
        render_call(
            call,
            rng,
            omit_scope=(index in shared_scope_indices and index != first_shared),
        )
        for index, call in enumerate(calls)
    ]
    count = len(clauses)
    also_last = count >= 2 and rng.randrange(5) == 0
    if also_last:
        clauses[-1] = _as_also_clause(clauses[-1])
    ending_style = rng.randrange(4)
    final = _final_style(clauses[-1].final, ending_style)
    shared_suffix = ".shared_scope" if shared_scope_indices else ""
    also_suffix = ".also_last" if also_last else ""

    if count == 1:
        composition = "single"
        utterance = final
    elif count == 2:
        composition = ("chain", "then", "continue", "while", "and")[rng.randrange(5)]
        if composition == "chain":
            utterance = f"{clauses[0].link} {final}"
        elif composition == "then":
            utterance = f"{clauses[0].after} {final}"
        elif composition == "continue":
            utterance = f"{clauses[0].link} 이어서 {final}"
        elif composition == "while":
            utterance = f"{clauses[0].while_form} {final}"
        else:
            first_final = _final_style(clauses[0].final, ending_style)
            utterance = f"{first_final}. 그리고 {final}"
    elif count == 3:
        composition = ("chain", "middle_then", "first_then", "staged", "and")[
            rng.randrange(5)
        ]
        if composition == "chain":
            utterance = f"{clauses[0].link} {clauses[1].link} {final}"
        elif composition == "middle_then":
            utterance = f"{clauses[0].link} {clauses[1].after} {final}"
        elif composition == "first_then":
            utterance = f"{clauses[0].after} {clauses[1].link} {final}"
        elif composition == "staged":
            utterance = f"{clauses[0].link} 그다음 {clauses[1].link} {final}"
        else:
            utterance = f"{clauses[0].link} {clauses[1].link} 그리고 {final}"
    else:
        composition = ("chain", "middle_then", "first_then", "staged", "and")[
            rng.randrange(5)
        ]
        if composition == "chain":
            utterance = (
                f"{clauses[0].link} {clauses[1].link} "
                f"{clauses[2].link} {final}"
            )
        elif composition == "middle_then":
            utterance = (
                f"{clauses[0].link} {clauses[1].after} "
                f"{clauses[2].link} {final}"
            )
        elif composition == "first_then":
            utterance = (
                f"{clauses[0].after} {clauses[1].link} "
                f"{clauses[2].link} {final}"
            )
        elif composition == "staged":
            utterance = (
                f"{clauses[0].link} 그다음 {clauses[1].link} "
                f"{clauses[2].link} {final}"
            )
        else:
            utterance = (
                f"{clauses[0].link} {clauses[1].link} "
                f"{clauses[2].link} 그리고 {final}"
            )

    utterance = normalize_text(utterance)
    clause_templates = "__".join(clause.template_key for clause in clauses)
    template_id = f"compose.c{count}.{composition}{shared_suffix}{also_suffix}__{clause_templates}"
    return utterance, template_id


def make_generation_family_id(
    template_id: str,
    functions: Sequence[str],
) -> str:
    material = json.dumps(
        {
            "ordered_function_skeleton": list(functions),
            "template_id": template_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"astra-v03-family-{digest}"


def make_example_id(
    calls: Sequence[Mapping[str, Any]],
    normalized_utterance: str,
) -> str:
    material = canonical_calls_key(calls) + "\x1f" + normalized_utterance
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"synthetic:{GENERATOR}:{digest}"


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    function_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    zones: dict[str, set[str]] = {function: set() for function in FUNCTIONS}
    target_kinds: dict[str, set[str]] = {function: set() for function in FUNCTIONS}
    target_values: dict[str, set[Any]] = {function: set() for function in FUNCTIONS}
    magnitudes: dict[str, set[str]] = {function: set() for function in FUNCTIONS}
    directions: dict[str, set[str]] = {function: set() for function in FUNCTIONS}
    features: set[str] = set()
    setting_kinds: dict[str, set[str]] = {
        "set_seat_climate": set(),
        "set_seat_massage": set(),
    }
    template_counts: Counter[str] = Counter()
    family_ids: set[str] = set()

    for row in rows:
        metadata = row["synthetic_generation"]
        patterns[metadata["function_family_pattern"]] += 1
        template_counts[metadata["template_id"]] += 1
        family_ids.add(metadata["generation_family_id"])
        for call in row["canonical_calls"]:
            function = call["function"]
            arguments = call["arguments"]
            function_counts[function] += 1
            family_counts[FUNCTION_FAMILY[function]] += 1
            zones[function].add(arguments.get("zone", "__OMITTED__"))
            if "feature" in arguments:
                features.add(arguments["feature"])
            value = arguments.get("target", arguments.get("setting"))
            if isinstance(value, Mapping):
                target_kinds[function].add(value["kind"])
                if "value" in value:
                    target_values[function].add(value["value"])
                if "magnitude" in value:
                    magnitudes[function].add(value["magnitude"])
                if "direction" in value:
                    directions[function].add(value["direction"])
                if function in setting_kinds:
                    setting_kinds[function].add(value["kind"])
            elif "state" in arguments:
                target_kinds[function].add("state")
                target_values[function].add(arguments["state"])

    def sorted_values(values: set[Any]) -> list[Any]:
        return sorted(values, key=lambda value: (str(type(value)), str(value)))

    return {
        "function_call_counts": dict(sorted(function_counts.items())),
        "function_family_call_counts": dict(sorted(family_counts.items())),
        "function_family_pattern_examples": dict(sorted(patterns.items())),
        "zones": {
            function: sorted_values(values) for function, values in zones.items()
        },
        "target_kinds": {
            function: sorted_values(values)
            for function, values in target_kinds.items()
        },
        "target_values": {
            function: sorted_values(values)
            for function, values in target_values.items()
        },
        "relative_magnitudes": {
            function: sorted_values(values)
            for function, values in magnitudes.items()
        },
        "directions": {
            function: sorted_values(values)
            for function, values in directions.items()
        },
        "seat_climate_features": sorted(features),
        "seat_setting_kinds": {
            function: sorted(values) for function, values in setting_kinds.items()
        },
        "template_count": len(template_counts),
        "generation_family_count": len(family_ids),
    }


def assert_required_coverage(
    rows: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
) -> None:
    function_counts = coverage["function_call_counts"]
    if set(function_counts) != set(FUNCTIONS):
        raise RuntimeError("not every canonical function was generated")
    if max(function_counts.values()) - min(function_counts.values()) > 3:
        raise RuntimeError(f"function coverage is imbalanced: {function_counts}")

    expected_patterns = {
        "HVAC",
        "Aperture",
        "Seat",
        "HVAC+Aperture",
        "HVAC+Seat",
        "Aperture+Seat",
        "HVAC+Aperture+Seat",
    }
    present_patterns = set(coverage["function_family_pattern_examples"])
    if missing := expected_patterns - present_patterns:
        raise RuntimeError(f"missing function-family patterns: {sorted(missing)}")

    expected_zones = {
        "set_hvac_power": {"__OMITTED__", "driver", "front_passenger", "rear", "all"},
        "set_hvac_temperature": {"__OMITTED__", "driver", "front_passenger", "rear", "all"},
        "set_hvac_fan_speed": {"__OMITTED__", "driver", "front_passenger", "rear", "all"},
        "set_window_position": set(WINDOW_ZONES),
        "set_seat_climate": set(SEAT_ZONES),
        "set_seat_massage": set(SEAT_ZONES),
    }
    for function, expected in expected_zones.items():
        actual = set(coverage["zones"][function])
        if actual != expected:
            raise RuntimeError(
                f"incomplete zone coverage for {function}: missing={sorted(expected - actual)}"
            )

    expected_kinds = {
        "set_hvac_power": {"state"},
        "set_hvac_temperature": {"absolute", "relative", "extreme"},
        "set_hvac_fan_speed": {"absolute", "relative", "extreme"},
        "set_window_position": {"named", "absolute_percent", "relative", "relative_percent"},
        "set_sunroof_position": {"named", "absolute_percent", "relative", "relative_percent"},
        "set_sunshade_position": {"named", "absolute_percent", "relative", "relative_percent"},
        "set_seat_climate": {"state", "absolute_level", "relative", "extreme"},
        "set_seat_massage": {"state", "absolute_level", "relative", "extreme"},
    }
    for function, expected in expected_kinds.items():
        actual = set(coverage["target_kinds"][function])
        if actual != expected:
            raise RuntimeError(
                f"incomplete target-kind coverage for {function}: "
                f"missing={sorted(expected - actual)}"
            )

    if set(coverage["seat_climate_features"]) != {"heating", "ventilation"}:
        raise RuntimeError("seat climate does not cover both features")

    temperature_values = set(coverage["target_values"]["set_hvac_temperature"])
    if not set(TEMPERATURE_GRID).issubset(temperature_values):
        raise RuntimeError("temperature coverage is missing values from the 0.5-degree grid")
    fan_values = set(coverage["target_values"]["set_hvac_fan_speed"])
    if not set(FAN_LEVELS).issubset(fan_values):
        raise RuntimeError("fan coverage is missing absolute levels")

    for function in (
        "set_window_position",
        "set_sunroof_position",
        "set_sunshade_position",
    ):
        values = set(coverage["target_values"][function])
        if not set(ABSOLUTE_PERCENT_GRID).issubset(values):
            raise RuntimeError(f"{function} is missing absolute percentage grid points")
        if not set(RELATIVE_PERCENT_GRID).issubset(values):
            raise RuntimeError(f"{function} is missing relative percentage grid points")

    magnitude_functions = (
        "set_hvac_temperature",
        "set_hvac_fan_speed",
        "set_window_position",
        "set_sunroof_position",
        "set_sunshade_position",
        "set_seat_climate",
        "set_seat_massage",
    )
    for function in magnitude_functions:
        if set(coverage["relative_magnitudes"][function]) != set(RELATIVE_MAGNITUDES):
            raise RuntimeError(f"{function} is missing relative magnitudes")

    if any(has_resource_conflict(row["canonical_calls"]) for row in rows):
        raise RuntimeError("generated rows contain an effective-resource conflict")


def generate_candidates(
    *,
    seed: int,
    targets: Mapping[int, int],
    original_utterances: Mapping[str, str],
    original_signatures: set[str],
    schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate validated candidate records and deterministic diagnostics."""

    validate_registry(schema, registry)
    skeletons = build_function_skeletons(targets, seed)
    sampler = SemanticSampler(seed)
    rows: list[dict[str, Any]] = []
    generated_text: dict[str, str] = {}
    generated_signatures: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    repeated_function_examples = 0
    shared_scope_examples = 0
    global_ordinal = 0

    for call_count in sorted(skeletons):
        for skeleton in skeletons[call_count]:
            for signature_attempt in range(2048):
                calls, shared_scope_indices = sampler.sample_calls(
                    skeleton,
                    global_ordinal,
                )
                payload = canonicalize_payload(
                    {"calls": calls},
                    schema=schema,
                    registry=registry,
                )
                calls = payload["calls"]
                gold = canonical_calls_key(calls)
                if gold in original_signatures:
                    rejection_counts["original_canonical_signature"] += 1
                    continue
                if gold in generated_signatures:
                    rejection_counts["synthetic_canonical_signature"] += 1
                    continue
                break
            else:
                raise RuntimeError(
                    "could not sample a novel ordered canonical signature after "
                    f"2048 attempts for skeleton {skeleton!r}"
                )

            selected: tuple[str, str] | None = None
            for surface_attempt in range(512):
                surface_rng = random.Random(
                    stable_seed(seed, "surface", global_ordinal, surface_attempt, gold)
                )
                utterance, template_id = render_utterance(
                    calls,
                    surface_rng,
                    shared_scope_indices,
                )
                normalized = normalize_text(utterance)
                if issue := language_issue(normalized):
                    rejection_counts[f"language:{issue}"] += 1
                    continue
                if normalized in original_utterances:
                    rejection_counts["original_utterance_collision"] += 1
                    continue
                previous_gold = generated_text.get(normalized)
                if previous_gold is not None:
                    key = (
                        "candidate_duplicate_same_gold"
                        if previous_gold == gold
                        else "candidate_utterance_gold_conflict"
                    )
                    rejection_counts[key] += 1
                    continue
                selected = normalized, template_id
                break
            if selected is None:
                raise RuntimeError(
                    f"could not render a unique utterance after 512 attempts: {calls!r}"
                )

            utterance, template_id = selected
            family_id = make_generation_family_id(template_id, skeleton)
            example_id = make_example_id(calls, utterance)
            row = {
                "call_count": call_count,
                "canonical_calls": calls,
                "example_id": example_id,
                "source_group_id": example_id,
                "source_split": "synthetic",
                "source_type": "synthetic",
                "synthetic_generation": {
                    "function_family_pattern": family_pattern(skeleton),
                    "generation_family_id": family_id,
                    "generator": GENERATOR,
                    "generator_version": GENERATOR_VERSION,
                    "seed": seed,
                    "template_id": template_id,
                },
                "utterance_ko": utterance,
            }
            generated_text[utterance] = gold
            generated_signatures.add(gold)
            rows.append(row)
            repeated_function_examples += int(len(set(skeleton)) < len(skeleton))
            shared_scope_examples += int(bool(shared_scope_indices))
            global_ordinal += 1

    expected_total = sum(targets.values())
    if len(rows) != expected_total:
        raise RuntimeError(f"generated {len(rows)} rows, expected {expected_total}")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise RuntimeError("content-derived example_id collision")
    if len(generated_signatures) != len(rows):
        raise RuntimeError("synthetic ordered canonical signatures are not unique")
    if overlap := generated_signatures & original_signatures:
        raise RuntimeError(
            f"synthetic ordered canonical signatures overlap original data: {len(overlap)}"
        )

    counts = Counter(row["call_count"] for row in rows)
    expected_counts = {int(key): int(value) for key, value in targets.items()}
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        raise RuntimeError(f"call-count distribution mismatch: {counts}")

    coverage = _coverage(rows)
    assert_required_coverage(rows, coverage)
    diagnostics = {
        "canonical_signature_novelty": {
            "original_unique_signatures": len(original_signatures),
            "synthetic_unique_signatures": len(generated_signatures),
            "original_synthetic_overlap": 0,
        },
        "call_count_distribution": {
            str(key): counts[key] for key in sorted(counts)
        },
        "coverage": coverage,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "repeated_function_examples": repeated_function_examples,
        "shared_scope_examples": shared_scope_examples,
    }
    return rows, diagnostics


def serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _atomic_write_many(contents: Mapping[Path, str], *, force: bool) -> None:
    paths = list(contents)
    if len(paths) != len(set(paths)):
        raise ValueError("output paths must be distinct")
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing artifact(s): {joined}")

    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in contents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append((Path(handle.name), path))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing candidate/report artifacts atomically",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = load_json_object(args.schema)
    registry = load_json_object(args.registry)
    original_utterances, original_signatures = load_original_pool(args.original_root)
    rows, diagnostics = generate_candidates(
        seed=args.seed,
        targets=CALL_COUNT_TARGETS,
        original_utterances=original_utterances,
        original_signatures=original_signatures,
        schema=schema,
        registry=registry,
    )
    candidate_text = serialize_jsonl(rows)
    candidate_sha256 = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
    report = {
        "artifact": {
            "candidate_jsonl": str(args.output),
            "candidate_sha256": candidate_sha256,
        },
        "call_count_targets": {
            str(key): CALL_COUNT_TARGETS[key] for key in sorted(CALL_COUNT_TARGETS)
        },
        "generated_candidates": len(rows),
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "original_normalized_utterance_count": len(original_utterances),
        "percentage_grids": {
            "absolute": list(ABSOLUTE_PERCENT_GRID),
            "relative": list(RELATIVE_PERCENT_GRID),
        },
        "registry_version": registry.get("registry_version"),
        "schema_id": schema.get("$id"),
        "seed": args.seed,
        **diagnostics,
    }
    report_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    _atomic_write_many(
        {args.output: candidate_text, args.report: report_text},
        force=args.force,
    )
    print(f"generated candidates: {len(rows)}")
    print(f"call-count distribution: {diagnostics['call_count_distribution']}")
    print(f"candidate sha256: {candidate_sha256}")
    print(f"written: {args.output}")
    print(f"written: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
