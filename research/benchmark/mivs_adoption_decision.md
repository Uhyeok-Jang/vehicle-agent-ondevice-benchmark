# MIVS Adoption Decision

Status: conditionally adopted for supplementary evaluation  
Decision date: 2026-09-03  
Evidence manifest: [`mivs_source_manifest.json`](../config/mivs_source_manifest.json)
Reproduced audit: [`mivs_audit_v1/summary.json`](../analysis/dataset_statistics/mivs_audit_v1/summary.json)

## Decision

MIVS may be used as a **supplementary compositional transfer test** after the gates below are satisfied. It must not be described as an independent validation dataset, independent external validation, or a naturally collected Korean benchmark.

The recommended source slice is the official 2,000-record vehicle test partition:

- `aispeech/test/one_domain_data/车载控制.json`: 1,000 single-intent records.
- `aispeech/test/one_domain_data/车载控制_multi.json`: 1,000 multi-intent records. Its intent-count distribution is 19 two-intent, 477 three-intent, and 504 four-intent records.

This slice is useful for testing function-call composition and complexity transfer. It is not a drop-in benchmark: the Chinese utterances and hierarchical intent-slot annotations must be converted to reviewed Korean utterances and the same canonical Vehicle API used by the primary MAC-SLU benchmark.

The counts above describe source semantic units. They must not be reported as canonical call counts: unsupported or context-dependent source units can map to zero calls, and canonical call counts exist only after the mapping registry is frozen and applied.

## Basis for conditional adoption

The [MIVS paper](https://arxiv.org/abs/2402.18258) describes a Chinese hierarchical SLU dataset with five domains and 105,240 records. The [official author repository](https://github.com/X-LANCE/MIVS_BIRGAT) distributes the dataset as [`data/aispeech.zip`](https://github.com/X-LANCE/MIVS_BIRGAT/blob/d1055a58d673468b4471e020626e8306ae6e4cb6/data/aispeech.zip). Direct inspection of the pinned archive reproduced all 105,240 rows, including 20,000 single-domain vehicle rows split 16,000/2,000/2,000 across train/valid/test.

The released vehicle annotation is structurally suitable for semantic-to-API conversion:

- Three intent labels and 18 slot labels are represented as a `domain -> intents -> slots` hierarchy.
- All 20,000 vehicle records satisfy the released ontology's domain-intent-slot constraints.
- The difficult vehicle partition has approximately 3.49 intents and 10.20 slots per utterance on average, providing a useful multi-call stress condition.
- The release contains text and annotations, not audio, which matches this project's text-to-API scope.

Cross-domain partitions are automatically concatenated according to the paper and repository documentation. They may be retained only as an explicitly synthetic, optional robustness set. `null_data` may be used only if out-of-scope rejection is separately specified.

## Independence and overlap risk

MIVS does not establish an independent provenance boundary from MAC-SLU. The two vehicle ontologies have all three intent names in common, and 16 of MIVS's 18 slot names appear exactly in the observed MAC-SLU vehicle annotations. Their official publications do not document a corpus-lineage relationship, so independence cannot be inferred.

A local exact-text comparison against the current [MAC-SLU vehicle inventory](../analysis/dataset_statistics/macslu_vehicle/vehicle_samples.csv) found:

- 96 of 8,055 unique MAC-SLU vehicle queries occur exactly somewhere in the complete MIVS release, or 1.1918%. Removing case, whitespace, and punctuation did not add further matches.
- All 96 overlapping records have the same intent sequence; only 28 have fully identical slot-value semantics, demonstrating annotation drift despite their common ontology.
- Thirteen of the 2,000 proposed MIVS vehicle test records also occur in MAC-SLU: 11 in MAC-SLU train, one in validation, and one in test.
- None of the 1,000 MIVS multi-intent vehicle test records exactly overlaps the MAC-SLU vehicle inventory.

MIVS itself also contains exact cross-split vehicle-query overlap: 24 validation/train, six test/train, and two test/validation intersections. The 20,000 vehicle rows therefore contain 19,968 unique query strings despite having no duplicates within an individual split.

These facts prohibit an “independent validation” claim. Any result should instead be reported as performance on a **MIVS-derived supplementary compositional transfer set**.

## Offset and annotation QA

The 20,000 vehicle records contain 137,106 slot annotations:

| Check | Result |
| --- | ---: |
| Slots with `pos` | 115,496 (84.24%) |
| Slots without `pos` | 21,610 (15.76%) |
| Invalid `pos` ranges | 0 |
| Exact value/span matches among positioned slots | 115,493 of 115,496 (99.9974%) |
| Value/span mismatches | 3 |
| Ontology path violations | 0 |

The three value/span mismatches involve whitespace-bearing Latin values. Missing offsets are common because MIVS permits implicit or normalized slot values that do not appear as a continuous source span. Conversion must use the complete hierarchical annotation, not assume that every executable argument can be recovered by span extraction.

## License caveat

The repository carries an [MIT license](https://github.com/X-LANCE/MIVS_BIRGAT/blob/d1055a58d673468b4471e020626e8306ae6e4cb6/LICENSE), but the archive contains no separate dataset license and the repository does not separately document the upstream AISpeech corpus rights. Internal research use is operationally low-risk when the copyright and license notice are retained. Public redistribution of translated or derived corpus records remains ambiguous and requires author confirmation.

Until that confirmation exists, publish only the pinned download reference, transformation code, ontology/API mappings, source identifiers and hashes, aggregate QA, and required notices. Do not commit or redistribute MIVS utterance text or Korean translations derived from it.

This is a provenance and licensing risk assessment, not legal advice.

## Mandatory adoption gates

1. Pin repository revision `d1055a58d673468b4471e020626e8306ae6e4cb6` and verify the archive SHA-256 recorded in the source manifest before processing.
2. Assign stable derived identifiers using source path, line index, normalized Chinese text, and semantic-content hashes because MIVS does not provide record IDs.
3. Exclude or quarantine the 13 proposed test records that exactly overlap MAC-SLU, including the 11 that overlap MAC-SLU train.
4. Run semantic-frame and near-duplicate detection across all MAC-SLU splits before Korean translation; exact text matching is only a lower bound on leakage.
5. Freeze the MIVS source selection and the three-intent/18-slot-to-canonical-API mapping before evaluating either model family.
6. Map only supported, executable vehicle semantics. Record unsupported, ambiguous, contradictory, or unresolvable items with explicit reason codes instead of coercing them to the nearest API.
7. Produce Korean commands through meaning-preserving adaptation and independent bilingual review. Preserve action, negation, object, cabin position, direction, value, unit, mode, and multi-command order.
8. Keep every derivative of one source record in one source group and repeat leakage detection on normalized Korean text after translation.
9. Do not train, prompt-select, map-select, threshold-select, or tune on the frozen MIVS supplementary test.
10. Evaluate with the primary benchmark's canonical function-call exact match and argument metrics. Do not compare the transformed target using MIVS's native serialized-frame accuracy.
11. Report single-intent and multi-intent results separately, with the combined number treated as secondary.
12. Obtain author confirmation before publicly redistributing source text or Korean derivative text; otherwise release only reproducible transformation materials and hashes.

## Permitted reporting language

Permitted:

> We additionally evaluated both approaches on a MIVS-derived Korean supplementary set designed to test multi-intent compositional transfer.

Not permitted:

> We confirmed the result on an independent external Korean benchmark.

## Primary sources

- [Peer-reviewed paper DOI](https://doi.org/10.1109/ICASSP48485.2024.10446325)
- [Paper preprint](https://arxiv.org/abs/2402.18258)
- [Author publication page](https://rhythmcao.github.io/publication/2024-birgat/)
- [Official repository](https://github.com/X-LANCE/MIVS_BIRGAT)
- [Pinned MIVS archive page](https://github.com/X-LANCE/MIVS_BIRGAT/blob/d1055a58d673468b4471e020626e8306ae6e4cb6/data/aispeech.zip)
- [Pinned repository license](https://github.com/X-LANCE/MIVS_BIRGAT/blob/d1055a58d673468b4471e020626e8306ae6e4cb6/LICENSE)

## Verification note

Research verification and artifact drafting were AI-assisted. Dataset facts were checked against the primary paper, author-linked repository, and a direct structural audit of the pinned archive; uncertainty about corpus lineage and dataset-specific redistribution rights is retained explicitly.
