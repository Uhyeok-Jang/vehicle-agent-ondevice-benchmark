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
- [`schema/vehicle_api_schema.v0.1.0.json`](schema/vehicle_api_schema.v0.1.0.json), [`schema/vehicle_api_registry.v0.1.0.json`](schema/vehicle_api_registry.v0.1.0.json), and [`schema/vehicle_api_contract.v0.1.0.md`](schema/vehicle_api_contract.v0.1.0.md): closed eight-function pilot contract, deterministic ordering, and staged extension gates.
- [`preprocessing/canonical_vehicle_api.py`](preprocessing/canonical_vehicle_api.py): shared schema validator and canonical serializer.

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

Implement the versioned MAC-SLU/MIVS structural adapters and source-to-canonical mapping registry, then measure unit- and row-level `mapped|ambiguous|unsupported|needs_context` coverage. Only after the API extension gates and reproducible MIVS audit pass should the Korean translation pilot begin.

The older `inspect_macslu.py` and `analyze_macslu_vehicle.py` files are exploratory EDA utilities. Their unpinned loader and legacy “missing semantics” label are not authoritative inputs to the release pipeline.
