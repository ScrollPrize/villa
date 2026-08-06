# Task log: las_manager Phase 7 — Lasagna backend

## Scope

- Add Lasagna checkpoint discovery and command construction to the existing
  backend-neutral manager workflow.
- Make direct Lasagna `predict3d` author the shared portable provenance format.
- Reuse catalog, prefetch, run records, tmux, completion, artifact inventory,
  staging upload, and Atlas Lasagna ingestion without duplicating them.

## Implementation notes

- Phase 7 follows the independently reviewed seven-phase task plan.
- Existing user changes and the untracked compiled `monotone_norm` extension
  are preserved.
- Checkpoint classification uses embedded structure, never filenames. Fiber
  records keep their current embedded-config path; Lasagna records extract
  patch/architecture/precision/validation metadata and use a `lasagna/...`
  selector.
- `launch_inference` remains the single run/tmux/lifecycle implementation. Its
  only backend branch selects Fiber or Lasagna command construction and the
  portable artifact kind.
- Direct `predict3d` writes `inference.json` beside its manifest, including
  source scales, checkpoint/runtime identity, numerical settings, preserved
  Lasagna decoding metadata, and bounded structural output inventory.
- Phase closeout parametrizes the atomic staging/idempotency and Atlas-ingest
  lifecycle regressions over both `fiber3d-prediction` and `lasagna`, proving
  that Lasagna uses the complete shared publication path rather than only its
  validator entry point.

## Deviations and findings

- No real download, GPU inference, S3 staging, Atlas mutation, or publication
  was run. The user explicitly deferred the real run and will test it later;
  remote mutation and publication remain operator-controlled.
- The full `test_preprocess_cos_omezarr.py` run stalls in the known local Zarr
  3.2.1 synchronous `zarr.open(..., mode="w")` path before reaching Phase 7
  logic. The new direct-provenance test avoids that unrelated constructor and
  passes. This is the same bounded-real-run blocker already recorded in status.

## Validation

- `33 passed, 59 deselected, 6 warnings` after the Phase 7 closeout added
  Lasagna coverage for atomic staging/idempotency and Atlas-ingest lifecycle:
  `pytest -q test_manager.py test_manager_open_data.py
  test_inference_provenance.py test_packaging.py
  test_preprocess_cos_omezarr.py -k 'manager or open_data or
  inference_provenance or packaging or direct_predict3d'`.
- `19 passed`: full manager unit/integration file.
- `1 passed, 59 deselected`: direct Lasagna provenance regression.
- `python -m preprocess_cos_omezarr predict3d --help` exposes checkpoint,
  compressor, and provenance-context arguments.
- `lasagna.manager.cli inference run --help` exposes both backends; generated
  Bash completion passes `bash -n`.
- `python -m py_compile` passes for all changed Python implementation modules.
- `git diff --check` passes.
- A clean temporary-config smoke test passed for `config init`, `config show`,
  top-level help, and generated Bash completion without network access or an
  installation/bootstrap step.
