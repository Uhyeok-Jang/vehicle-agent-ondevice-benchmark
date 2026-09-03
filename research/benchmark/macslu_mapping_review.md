# MAC-SLU Mapping Review Log

Status: Stage 3 working draft  
Mapping registry: `macslu_vehicle_mapping.v0.1.0.json`

This log records human-reviewed, fail-closed changes made after measuring the complete verified MAC-SLU vehicle population. Each snapshot embeds the exact generator, dependency, source, schema, and mapping-registry hashes used to produce it.

## Coverage history

| Snapshot | Reviewed change | Mapped units | Fully mapped rows |
| --- | --- | ---: | ---: |
| `baseline` | Initial closed eight-function pilot | 2,240 / 11,471 (19.53%) | 479 / 8,057 (5.95%) |
| `r1` | Window `调节内容=幅度`, `value=关闭` as named closed | 2,273 (19.82%) | 499 (6.19%) |
| `r2` | Entity-scoped window zone aliases `前排前排`, `四个`, and `四门` | 2,282 (19.89%) | 506 (6.28%) |
| `r3` | Unique-HVAC row-context carry-over for `提供信息` temperature and fan-speed units | 2,486 (21.67%) | 625 (7.76%) |

These are transformation-coverage results, not release-eligibility counts. Every snapshot keeps `final_eligibility.status=not_adjudicated` and `eligible_rows=null`.

## Accepted decisions

### Window amplitude closure

`操作=调节`, `调节内容=幅度`, and `value=关闭` is mapped to `set_window_position` with a named `closed` target only when a recognized window entity and zone are present. The mapper does not infer a percentage or relative displacement.

`前排前排` is treated as a duplicated `front_row` annotation. `四个` and `四门` are treated as `all` only under an explicitly normalized window entity. Tests verify that those aliases do not apply to seat or other entities.

### Information-fragment HVAC context

The source label `提供信息` frequently carries executable temperature or fan-speed arguments without an object slot. Row-level entity carry-over is allowed only when all structurally valid, explicitly normalized entity-bearing vehicle units in that row identify HVAC and no other entity is present. The target fragment must then independently satisfy the closed temperature or fan-speed mapping rule.

Context evidence is recorded in the unit trace as `context.entity.unique_in_row` with the contributing source unit IDs. A standalone fragment, a row with only a non-HVAC entity, or a row containing HVAC plus any other entity remains `needs_context`. This deliberately excludes 279 superficially compatible candidates that lack unique HVAC row context; two of those demonstrate that schema-unique mapping alone can misclassify seat commands as windows.

The `r3` change maps 204 additional units relative to `r2`: 147 fan-speed and 57 temperature calls, split as 172 train, 15 validation, and 17 test units. It also exposes 209 previously hidden normalization or rule failures under rows whose HVAC context is known; these remain non-mapped and become evidence for later remediation.

## Deferred decisions

- Window amplitude percentages and relative expressions remain unmapped because `操作` and `操作_concrete` patterns mix absolute and relative semantics.
- The window zone value `前车` remains ambiguous.
- Mixed-domain rows remain non-mapped even when an individual vehicle unit looks compatible.
- A mapping that is unique only because the current eight-function schema lacks competing functions is not accepted as semantic evidence.
- `function_outside_schema` and residual `no_mapping_rule` groups are reviewed only after normalization remediation, before any E1/E2 API extension decision.
