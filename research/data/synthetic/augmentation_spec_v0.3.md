# MAC-SLU Korean Vehicle-Control Augmentation Specification v0.3

Status: implementation specification  
Benchmark version: `macslu_korean_augmented_v0.3`  
Canonical Vehicle API: `vehicle_api_schema.v0.1.0.json` / registry `0.1.0`  
Generator: `astra_vehicle_aug_v0.3`  
Random seed: `20260905`

## Purpose

Version 0.3 rebuilds the benchmark from a single source pool rather than extending
the frozen v0.2 train split. The 439 globally unique Korean examples in v0.2 are
pooled, about 2,300 gold-first synthetic Korean vehicle commands are added, and a
new deterministic train/validation/test split is created. The old v0.2 splits and
all pilot model results remain unchanged and are not treated as the final test.

The primary objective is schema-valid compositional coverage, not paraphrase
volume. Every synthetic example starts from canonical calls and is rendered into
Korean only after its gold structure is fixed.

## Original source pool

Inputs are all records from:

- `research/data/processed/macslu_korean_v0.2/train.jsonl` (353)
- `research/data/processed/macslu_korean_v0.2/validation.jsonl` (43)
- `research/data/processed/macslu_korean_v0.2/test.jsonl` (43)

The combined pool therefore contains exactly 439 records. The active v0.2
`benchmark_version` and `benchmark_split` keys are removed from source-pool rows
and retained explicitly as `previous_benchmark_version` and
`previous_benchmark_split`. Final split rows then receive
`benchmark_version: "macslu_korean_augmented_v0.3"` and a newly assigned
`benchmark_split`. Existing `example_id`, `source_group_id`, `source_split`,
`translation`, `deduplication`, `utterance_ko`, and `canonical_calls` values are
retained. Original records receive `source_type: "original"`.

Normalization follows the v0.2 policy exactly: trim leading/trailing whitespace
and collapse each run of whitespace to one ASCII space. No punctuation stripping,
case folding, or morphological normalization is added.

## Synthetic target

| Canonical call count | Target examples |
|---:|---:|
| 1 | 500 |
| 2 | 800 |
| 3 | 600 |
| 4 | 400 |
| **Total** | **2,300** |

The generator uses deterministic rejection sampling until each target is met.
The downstream builder remains authoritative: candidates that fail schema,
language, duplicate, or conflict checks are excluded. If exclusions prevent a
target from being met, the build fails with a report rather than silently changing
the intended distribution.

## Gold-first generation strategy

1. Sample a call-count stratum and a function-family composition pattern.
2. Sample one to four canonical calls from the existing API contract.
3. Reject redundant or contradictory calls that address the same effective
   resource in one command. Repeating a function across distinct, non-overlapping
   zones is allowed and intentionally represented.
4. Validate `{"calls": canonical_calls}` with the repository canonicalizer.
5. Render each call, or a safe shared-scope call group, with Korean phrase banks.
6. Join clauses with a deterministic composition template while preserving call
   order.
7. Reject empty, malformed, duplicate, conflicting, or placeholder-bearing text.
8. Reject an ordered canonical signature already emitted synthetically or already
   present in the 439-record original pool.

The generator never infers gold calls from a completed sentence.

Effective-resource overlap is checked conservatively. An omitted HVAC zone is
treated as overlapping any explicit HVAC zone for repeated operations. For
windows, `all` overlaps every zone, `front_row` overlaps driver/front-passenger,
`rear_row` overlaps rear-left/rear-right, and left/right-side scopes overlap their
corresponding front and rear windows. For seats, `all` overlaps every seat and
`rear_row` overlaps rear-left/rear-right. Repeated operations are emitted only for
disjoint scopes. Seat-climate resource identity also includes `feature`.

## Canonical coverage policy

All eight functions are scheduled with a least-represented-first tie-breaking
policy so no function dominates solely because its argument space is larger:

- `set_hvac_power`
- `set_hvac_temperature`
- `set_hvac_fan_speed`
- `set_window_position`
- `set_sunroof_position`
- `set_sunshade_position`
- `set_seat_climate`
- `set_seat_massage`

Family schedules include HVAC-only, Aperture-only, Seat-only, all pairwise
cross-family combinations, and HVAC + Aperture + Seat. Multi-call schedules also
include:

- one explicit scope shared by multiple compatible HVAC calls;
- different scopes in the same utterance;
- the same function applied to distinct non-overlapping zones;
- mixed-family commands;
- varied ordered function sequences.

The authoritative API schema, not the observed v0.2 vocabulary, defines validity.
Coverage includes all enum branches and representative numeric points:

- HVAC temperature: absolute Celsius values on the valid 0.5-degree grid,
  relative increase/decrease with all four magnitudes, and min/max extremes;
- HVAC fan: every valid level 1–8, relative increase/decrease with all four
  magnitudes, and min/max extremes;
- windows/sunroof: named targets including `vent`, absolute percentages,
  qualitative relative movement, and relative percentages;
- sunshade: named open/closed, absolute percentages, qualitative relative
  movement, and relative percentages;
- seats: state, levels 1–3, relative increase/decrease with all four magnitudes,
  and min/max extremes, across both climate features and massage;
- every valid HVAC, window, and seat zone, including omitted HVAC scope where
  allowed.

Absolute percentages are sampled from
`[0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]`; relative percentages use
`[5, 10, 15, 20, 25, 30, 40, 50]`. All emitted values satisfy the canonical
schema. Release gates require all eight functions, every enum branch, every HVAC
temperature grid point from 16 to 32 by 0.5, every fan level, every seat level,
and every value in both percentage grids to appear in the validated synthetic
set. Synthetic function-frequency `max/min` must not exceed 1.10.

All 2,300 synthetic ordered canonical signatures must be unique and novel with
respect to the original pool. This is a direct guard against inflating the data
with multiple surface paraphrases of a small gold set.

## Linguistic variation policy

Korean is generated from function-specific, argument-aware phrase banks and
composition templates. Variation covers polite requests, short colloquial
commands, explicit instructions, safe object ellipsis, different word orders,
and connectors such as `하고`, `한 다음`, `그리고`, `면서`, and `도`.

Scope and target information may be omitted only when a renderer has an explicit
safe shared-scope rule. `zone` omission in canonical HVAC calls is rendered as an
unqualified command; it is never verbalized as `all`. Relative direction,
magnitude, numeric value, unit, feature, state, and call order must remain
recoverable without guessing.

Surface diversity is subordinate to semantic clarity. The generator does not use
pronouns with unclear antecedents, unsupported vehicle features, or expressions
that collapse two canonical calls into one ambiguous operation.

The language validator requires at least four Hangul syllables, at least six
normalized characters including spaces, no CJK ideographs, and no braces, angle
brackets, `$` placeholders, `TODO`, `None`, or `null` artifacts.

## Provenance policy

Every synthetic record uses the existing core fields and adds explicit metadata:

- `source_type: "synthetic"`
- `example_id` and `source_group_id`
- `source_split: "synthetic"`
- `call_count`
- `canonical_calls`
- `utterance_ko`
- `synthetic_generation.generator`
- `synthetic_generation.generator_version`
- `synthetic_generation.seed`
- `synthetic_generation.generation_family_id`
- `synthetic_generation.template_id`
- `synthetic_generation.function_family_pattern`

`generation_family_id` groups examples that share the same surface template and
ordered function skeleton. Argument and zone substitutions within that skeleton
remain in the same group. The identifier is stable across repeated runs.
Each synthetic `source_group_id` is the row's unique `example_id`; it must not be
used as the near-duplicate split group.

## Validation and deduplication policy

Before inclusion, the builder checks:

1. canonical JSON Schema validity;
2. registered function names;
3. exact argument keys and typed enum/value constraints;
4. `call_count == len(canonical_calls)` and `1 <= call_count <= 4`;
5. non-empty, minimally complete Korean utterances;
6. absence of unresolved placeholders or template artifacts;
7. absence of redundant/contradictory operations on the same effective resource;
8. exact normalized-utterance duplicates within synthetic data;
9. duplicate utterances across original and synthetic data;
10. one normalized utterance mapping to more than one canonical gold;
11. global `example_id` uniqueness.

For a same-text/same-gold duplicate, the original record has precedence and the
synthetic record is removed. A same-text/different-gold conflict is never resolved
by choosing a label silently: the conflict is recorded and every conflicting
synthetic candidate is excluded. A conflict between original records is a hard
error because v0.2 is already frozen as globally conflict-free.

Validation counts and conflict details are recorded in
`synthetic_generation_report_v0.3.json` and `dataset_report.json`.

## Split policy

The final pool is split from scratch with seed `20260905`.

- Target ratio: train 80%, validation 10%, test 10%.
- Original and synthetic examples are stratified separately by `call_count`, then
  combined, preserving approximately the same source-type ratio in each split.
- Original records are singleton groups. Synthetic records are atomic by
  `generation_family_id`.
- Within each call-count/source-type stratum, deterministic greedy group packing
  assigns groups toward 80/10/10 target counts. Seeded tie-breaking and
  function-family deficits are used only to break equivalent placements.
- No generation family may appear in more than one split.
- Every primitive function/argument label present in validation or test must also
  occur in train; this is an asserted release gate.
- Exact normalized utterance overlap for train/validation, train/test, and
  validation/test must be zero.
- Each final split's share must be within one percentage point of 80/10/10 unless
  an atomic generation family makes that impossible; any deviation and the
  largest group size are reported. The original/synthetic share in each split
  must be within two percentage points of the full-pool share.
- The union of the three splits must equal the 439-row original pool plus the
  validated synthetic file exactly once by `example_id`.

Once generated and reviewed, the v0.3 test file is treated as frozen final test
data.

## Reporting and review

The final report includes total/source/split counts, 1–4 call distributions,
function and family frequencies, major argument coverage, character-length
statistics, dedup/conflict counts, exact split overlap, generation-family overlap,
canonical-signature novelty/overlap, primitive train coverage, input and artifact
hashes, and 32 deterministic review samples (eight per call-count stratum)
spanning all functions and family patterns.

The review sample is manually inspected before release. If systematic language
issues are found, phrase banks are corrected and both scripts are rerun with the
same seed.

## Known limitations

- Phrase-bank generation cannot reproduce the full pragmatic variety of human
  Korean commands, even with many templates.
- `generation_family_id` reduces template leakage but is not semantic embedding
  clustering; unrelated templates can still express similar intent.
- Numeric percentage sampling is representative, not exhaustive.
- Original Korean translations and canonical calls are inherited from v0.2 and
  are not re-annotated in this stage.
- The frozen mmBERT v0.2 factorizer and constrained decoder cannot represent
  aperture percentages/`vent`, non-state seat settings, or `medium`/`large`
  magnitudes. Existing model files remain untouched. Training on v0.3 will require
  a separate versioned mmBERT v0.3 label/factorization update; Qwen and the common
  evaluator already consume the canonical API structure directly.
- This stage builds and validates data only. It does not train or evaluate a
  model.
