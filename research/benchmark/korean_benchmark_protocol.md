# Korean Vehicle Command Benchmark Protocol

Status: Stage 3 working draft, protocol version `0.1.0`

Bulk translation status: **NO-GO** until the MAC-SLU and MIVS audits are accepted and the canonical Vehicle API schema is frozen. Only a versioned translation pilot may begin after those gates.

This protocol turns the Chinese MAC-SLU vehicle subset into a Korean text-to-API benchmark. It preserves source provenance while producing one language-independent canonical target shared by the Encoder and generative SLM baselines. ASR, TTS, dialogue management, and open-ended generation are outside scope.

The pinned source is recorded in [`macslu_source_manifest.json`](../config/macslu_source_manifest.json). The source paper describes a Chinese, eight-domain benchmark with up to five intents, while the pinned release contains 20,542 rows and two rows above five active semantic units. The release also differs from the paper by three training rows. Those differences are dataset-version facts, not records to repair implicitly. MIVS is governed by a separate [source manifest](../config/mivs_source_manifest.json) and [conditional-adoption decision](mivs_adoption_decision.md).

## Stage 3 work packages

| Package | Output | Exit condition |
| --- | --- | --- |
| 3A. Source audit | pinned MAC-SLU/MIVS manifests, deterministic audit, row-level flags | source counts and hashes reproduce; leakage, licensing, and annotation risks are enumerated |
| 3B. Inclusion policy | immutable initial status and reason codes per source row | every source row has one initial disposition; final eligibility remains separate |
| 3C. Canonical Vehicle API | versioned JSON Schema and serialization rules | every supported call validates and has one deterministic representation |
| 3D. Source mapping | MAC-SLU semantic unit to canonical call mapping | supported vehicle units have complete, reviewed mappings |
| 3E. Korean conversion | Korean utterance plus translation provenance | meaning and all executable arguments are preserved |
| 3F. Freeze | released split files, hashes, QA report | validation/test are sealed before model tuning or comparison |

MAC-SLU packages 3A and 3B are complete at the working-draft level. The reproducible MIVS audit remains open, and the current API checkpoint is the 3C eight-function pilot. API names, argument enums, and final inclusion decisions remain unfrozen until the 3C/3D review gates pass.

## Record and group identity

Each derived record must contain, at minimum:

```json
{
  "example_id": "macslu:<revision>:<source_split>:<source_id>",
  "source": {
    "dataset": "Gatsby1984/MAC_SLU",
    "revision": "40670d121a89ad7142e3536ee6dc05374d095f6b",
    "split": "train|validation|test",
    "id": "<source id>",
    "text_zh": "<source query>"
  },
  "source_group_id": "macslu:<revision>:<source_split>:<source_id>",
  "utterance_ko": "<reviewed Korean command>",
  "calls": [],
  "quality": {
    "initial_status": "candidate|manual_review|quarantined|excluded",
    "final_status": "eligible|quarantined|excluded|null",
    "reason_codes": []
  },
  "translation": {
    "method": "human|machine_then_human",
    "system": "<model or translator identifier>",
    "system_version": "<immutable version>",
    "prompt_version": "<version or null>",
    "reviewer_id": "<pseudonymous id>",
    "review_status": "pending|accepted|rejected"
  }
}
```

All paraphrases derived from one source row share one `source_group_id`. A source group, never an individual paraphrase, is the unit of splitting and de-duplication.

`candidate` is an automatic audit outcome, not a clean/gold claim. A row becomes `eligible` only after source-to-canonical mapping, schema validation, translation review, and final adjudication. `manual_review` is a workflow state and cannot remain in a frozen release.

## Inclusion and quarantine policy

No audit flag causes an automatic correction of the source label.

| Condition | Initial disposition | Reason code |
| --- | --- | --- |
| Vehicle-only, structurally valid annotation | candidate | `none` |
| Empty semantics or no vehicle unit | exclude from the core vehicle benchmark | `no_vehicle_target` |
| Vehicle plus another domain | manual review; exclude from core and retain for a later challenge set | `mixed_domain_example` |
| `split_sens` count is greater than active semantic-unit count | manual review; do not infer missing semantics from the count alone | `split_sens_gt_semantic_units` |
| `split_sens` count is less than active semantic-unit count | manual review; do not infer incorrect semantics from the count alone | `split_sens_lt_semantic_units` |
| One intent index contains more than one populated domain | manual review | `multi_domain_per_intent` |
| Slot name outside the reviewed source allowlist | manual review | `unexpected_vehicle_slot` |
| More than the paper-stated five active intents | manual review | `max_intent_claim_violation` |
| Exact query or width/case/whitespace-normalized query occurs in more than one split | quarantine the later evaluation occurrence; never move it into another split | `cross_split_query_overlap_exact`, `cross_split_query_overlap_normalized` |
| Match appears only after punctuation removal | manual review; do not auto-quarantine because punctuation can change numbers or polarity | `cross_split_query_overlap_review_normalized` |
| Meaning cannot be mapped without guessing | exclude with retained evidence | `unresolvable_target` |

“Count match” means only that two list lengths agree; it must not be reported as proof that the annotation is semantically correct. Flags can overlap, so exclusion counts are computed from row identities rather than by summing flag totals.

## Split isolation

1. Retain the official source split on every record.
2. Detect cross-split overlap before translation using exact text and a quarantine key that applies Unicode NFKC, lowercasing, and whitespace removal. Use punctuation removal only as a review key while preserving decimal points and numeric signs.
3. Do not relocate duplicate evaluation rows. Quarantine the validation/test occurrence that overlaps an earlier split.
4. Translate and create paraphrases only after assigning `source_group_id`; every derivative remains in its source split.
5. Repeat overlap detection on normalized Korean text after translation. Newly colliding evaluation groups are quarantined or rewritten before freeze.
6. Fit preprocessing, mapping heuristics, thresholds, prompts, and models on train; use validation for selection. After schema compatibility and QA are complete, test text and labels are sealed and used once for the registered comparison.

Aggregate test-set auditing needed to construct and validate the benchmark is permitted during Stage 3 and must be logged. It must not be presented as a blind test created by parties who never inspected the source data.

Exact whole-query isolation is necessary but insufficient for RQ3. The audit also measures whether validation/test command fragments already occur in train. The main source-compatible track keeps the official splits after quarantine. A separately frozen compositional track may hold out call combinations or template families while retaining every atomic function and categorical enum in train; results from that track are reported separately and must not be described as function-OOV generalization.

## Korean conversion

Translation preserves executable meaning, not Chinese word order.

- The translation pass receives the full Chinese query, ordered `split_sens`, and the complete source vehicle semantics/slots as semantic constraints, but not the canonical API serialization.
- A separate semantic review pass compares the Chinese source, Korean result, and source annotation.
- Actions, negation, object, cabin position, direction, level, numeric value, unit, mode, and multi-command order must be preserved explicitly.
- Canonical enum values and SI/unit-normalized arguments are produced by the mapping stage, not embedded as artificial phrasing in the Korean command.
- Ambiguous source language remains flagged; the translator must not resolve ambiguity by inventing an argument.
- A frozen prompt and immutable model/version identifier are required for machine-assisted translation. Retries and manual edits remain in an append-only provenance log.
- Natural Korean is required. Literal calques that make the command easy only because they expose source-slot order are rejected.

The validation and test splits receive full bilingual review. All flagged training rows receive full review; the remaining training rows receive a seeded, intent-and-complexity-stratified audit. The sampling seed, strata, sample size, observed error rate, and adjudications are recorded before freeze.

Before bulk conversion, a 200–300 example coverage-oriented pilot spans functions, call counts, implicit values, ellipsis, and every flagged edge-case class. Proposed acceptance gates are Korean-only blind re-annotation call exact match of at least 95%, categorical agreement of at least 0.80, naturalness score at least 4/5 for at least 95% of items, and zero unresolved critical semantic drift after adjudication. These thresholds remain proposals until registered in the Stage 3 QA plan.

## Canonical-target constraints

The current pilot schema and its staged extension gates are defined by [`vehicle_api_contract.v0.1.0.md`](../schema/vehicle_api_contract.v0.1.0.md). The pilot contains eight high-support HVAC, aperture, and seat functions. It is an implementation checkpoint, not the final function inventory.

- Function names and argument keys use stable English `snake_case` identifiers and do not depend on Korean or Chinese wording.
- One source semantic unit maps to zero or one canonical call. A zero-call outcome requires an explicit exclusion or unsupported reason.
- Multiple calls retain source intent order and serialize deterministically.
- Each function has a closed argument set, explicit required fields, enums, numeric bounds, and canonical units.
- Surface aliases belong to preprocessing or training data, never to the gold API schema.
- Missing arguments remain missing; defaults are applied only when the Vehicle API contract defines them.
- Unsupported, contradictory, or unresolvable commands are flagged rather than coerced into the closest function.
- Encoder and SLM outputs are normalized by the same evaluator before exact function and argument scoring.

For an Encoder baseline that uses span supervision, only explicit mentions receive tokenizer-independent Unicode character offsets. Implicit, normalized, defaulted, or context-derived arguments are labeled by provenance and trained through a span-free canonical argument head. Every explicit offset must round-trip to the intended substring. A query/API pair alone must not be treated as complete BIO supervision.

## Freeze gate

Stage 3 is complete only when all of the following hold:

- source file hashes, row counts, tool versions, and audit command are reproducible;
- every source row has one disposition and machine-readable reason codes;
- cross-split source and Korean-text overlap is zero in released validation/test data;
- every released call validates against the frozen Vehicle API schema;
- source-to-canonical mapping coverage is 100% for released rows;
- validation and test translations have completed bilingual review;
- released JSONL files, schema, mapping tables, prompt, and QA report have content hashes;
- test artifacts are sealed before any Stage 5 model or hyperparameter choice uses test results.

## Primary sources

- [MAC-SLU paper](https://arxiv.org/abs/2512.01603)
- [MAC-SLU dataset](https://huggingface.co/datasets/Gatsby1984/MAC_SLU)
- [MAC-SLU reference repository](https://github.com/Gatsby-web/MAC_SLU)
- [MIVS/BiRGAT paper](https://doi.org/10.1109/ICASSP48485.2024.10446325)
- [MIVS reference repository](https://github.com/X-LANCE/MIVS_BIRGAT)
