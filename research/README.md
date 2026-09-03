# Research workspace

The current branch is building Stage 3 of the study: a Korean vehicle-command benchmark and a language-independent canonical Vehicle API. Bulk translation is intentionally blocked until source auditing, dataset selection, and schema freeze are complete.

## Current Stage 3 artifacts

- [`config/macslu_source_manifest.json`](config/macslu_source_manifest.json): pinned MAC-SLU revision, raw-file hashes, published claims, and reviewed vehicle slot names.
- [`preprocessing/audit_macslu.py`](preprocessing/audit_macslu.py): deterministic, fail-closed source and split audit.
- [`analysis/dataset_statistics/macslu_audit_v2/summary.json`](analysis/dataset_statistics/macslu_audit_v2/summary.json): reproduced audit summary using semantics-preserving overlap normalization.
- [`analysis/dataset_statistics/macslu_audit_v2/issues.csv`](analysis/dataset_statistics/macslu_audit_v2/issues.csv): row/group-level review flags; flags are not automatic error verdicts.
- [`analysis/dataset_statistics/macslu_inventory_v2/inventory.csv`](analysis/dataset_statistics/macslu_inventory_v2/inventory.csv): all 20,542 source rows with immutable initial disposition and blank final status.
- [`analysis/dataset_statistics/macslu_inventory_v2/summary.json`](analysis/dataset_statistics/macslu_inventory_v2/summary.json): ledger counts and artifact hash.
- [`benchmark/korean_benchmark_protocol.md`](benchmark/korean_benchmark_protocol.md): inclusion, translation, leakage, target, and freeze policy.
- [`config/mivs_source_manifest.json`](config/mivs_source_manifest.json) and [`benchmark/mivs_adoption_decision.md`](benchmark/mivs_adoption_decision.md): MIVS provenance and conditional supplementary-test decision.
- [`preprocessing/audit_mivs.py`](preprocessing/audit_mivs.py) and [`analysis/dataset_statistics/mivs_audit_v1/summary.json`](analysis/dataset_statistics/mivs_audit_v1/summary.json): deterministic MIVS release, ontology, offset, split-overlap, and MAC-overlap audit. The artifact index contains hashes and structural metadata, not source utterances or slot values.
- [`schema/vehicle_api_schema.v0.1.0.json`](schema/vehicle_api_schema.v0.1.0.json), [`schema/vehicle_api_registry.v0.1.0.json`](schema/vehicle_api_registry.v0.1.0.json), and [`schema/vehicle_api_contract.v0.1.0.md`](schema/vehicle_api_contract.v0.1.0.md): closed eight-function pilot contract, deterministic ordering, and staged extension gates.
- [`preprocessing/canonical_vehicle_api.py`](preprocessing/canonical_vehicle_api.py): shared schema validator and canonical serializer.
- [`preprocessing/map_macslu_vehicle.py`](preprocessing/map_macslu_vehicle.py), [`schema/macslu_vehicle_mapping.v0.1.0.json`](schema/macslu_vehicle_mapping.v0.1.0.json), and [`preprocessing/analyze_macslu_mapping.py`](preprocessing/analyze_macslu_mapping.py): fail-closed structural adapter, declarative source mapping, and full-population coverage analyzer.
- [`analysis/dataset_statistics/macslu_mapping_v0.1.0_baseline/summary.json`](analysis/dataset_statistics/macslu_mapping_v0.1.0_baseline/summary.json), [`analysis/dataset_statistics/macslu_mapping_v0.1.0_r3/summary.json`](analysis/dataset_statistics/macslu_mapping_v0.1.0_r3/summary.json), and [`benchmark/macslu_mapping_review.md`](benchmark/macslu_mapping_review.md): pre-remediation baseline, current mapping snapshot, and reviewed decision log. `mapped` is not final benchmark eligibility.

## Reproduce the MAC-SLU audit

The preferred path verifies the three raw JSONL files against the committed manifest. `<macslu-root>` must contain `label/train_set.jsonl`, `label/dev_set.jsonl`, and `label/test_set.jsonl` from the pinned dataset revision.

```bash
python research/preprocessing/audit_macslu.py \
  --manifest research/config/macslu_source_manifest.json \
  --source-root <macslu-root> \
  --output-dir /tmp/macslu_audit_check
```

The command refuses to overwrite an existing output directory and aborts before writing if the manifest record count or SHA-256 does not match. Compare the fresh result with `research/analysis/dataset_statistics/macslu_audit_v2/`; both files should match byte-for-byte when the Python version, generator, manifest, and inputs are unchanged.

Build the complete row ledger with:

```bash
python research/preprocessing/build_macslu_inventory.py \
  --manifest research/config/macslu_source_manifest.json \
  --source-root <macslu-root> \
  --output-dir /tmp/macslu_inventory_check
```

The ledger includes non-vehicle and unannotated rows as `excluded`; `candidate` means only that no automatic blocking flag fired. Its `final_status` remains empty until mapping and human review are complete.

## Reproduce the MIVS audit

The MIVS audit verifies the pinned archive and the raw MAC-SLU files before measuring ontology consistency, source offsets, internal split overlap, and cross-dataset overlap:

```bash
python research/preprocessing/audit_mivs.py \
  --manifest research/config/mivs_source_manifest.json \
  --archive <mivs-repository>/data/aispeech.zip \
  --mac-manifest research/config/macslu_source_manifest.json \
  --mac-source-root <macslu-root> \
  --output-dir /tmp/mivs_audit_check
```

The current result covers all 105,240 release records and 20,000 vehicle records (44,880 source semantic units). The recommended test partition contains 2,000 records and 4,485 source semantic units. That source-unit distribution is descriptive input structure, not a canonical API call-count distribution; call counts are defined only after mapping.

## Reproduce MAC-SLU mapping coverage

```bash
python research/preprocessing/analyze_macslu_mapping.py \
  --manifest research/config/macslu_source_manifest.json \
  --source-root <macslu-root> \
  --mapping-registry research/schema/macslu_vehicle_mapping.v0.1.0.json \
  --output-dir /tmp/macslu_mapping_check
```

The current `r3` snapshot maps 2,486 of 11,471 vehicle semantic units (21.67%) and fully maps 625 of 8,057 vehicle rows (7.76%). The analyzer records unresolved values and normalized failure signatures across the complete verified population. Final eligibility remains `not_adjudicated`; mapping success alone does not admit a row to the benchmark.

The optional Hugging Face loader requires the version recorded below. Raw-file verification is strongest with `--source-root`, because the Hugging Face loader exposes processed cache files rather than the three release JSONL files.

```bash
python -m pip install -r research/requirements-stage3.txt
python research/preprocessing/audit_macslu.py \
  --dataset Gatsby1984/MAC_SLU \
  --revision 40670d121a89ad7142e3536ee6dc05374d095f6b \
  --manifest research/config/macslu_source_manifest.json \
  --allow-unverified-source \
  --output-dir /tmp/macslu_audit_hf
```

That Hugging Face path is diagnostic only: processed cache files cannot verify the three raw release JSONL hashes, so the explicit `--allow-unverified-source` override is required and its result is not release-grade.

Validate or canonicalize a prediction with:

```bash
python research/preprocessing/canonical_vehicle_api.py prediction.json
```

Run the network-free regression tests with:

```bash
python -m unittest discover -s research/tests -p 'test_*.py' -v
```

## Next gate

Continue conservative MAC-SLU normalizer remediation using the newly exposed unresolved values, and remeasure after every registry change. Then review `function_outside_schema` and `no_mapping_rule` evidence before deciding whether the E1/E2 API extension gates are justified. A versioned MAC-SLU translation pilot may begin once its selected slice has frozen mappings and QA criteria; MIVS supplementary conversion has its own adoption gates and does not block the primary pilot.

The older `inspect_macslu.py` and `analyze_macslu_vehicle.py` files are exploratory EDA utilities. Their unpinned loader and legacy “missing semantics” label are not authoritative inputs to the release pipeline.
