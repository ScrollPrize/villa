# Request concurrency instrumentation — 2026-09-10

- Prior layout/prefault findings retained in docs/live_inference_chunk_major_20260910.md and changelog.
- User asks whether fast loader is used and requests in-flight visibility. No concurrency increase yet.
- NativeTileReader calls Vc3dCoordinateSampler.sample_block_zyx. Reads reserve input slots before submission; current8devices×1slot=8inputs,8executor threads.128result slots do not increase input capacity.
- Existing short profile prequeues1599 reference-intersecting chunks, not100 actual native trace directions. Distinguish RPCs/chunks/tiles/reads.
- Independent plan review approved. Implemented metadata-only shared snapshot,
  formatter and profile250ms sampled series/distributions. Added synthetic
  accounting/read-state regression. No scheduling or capacity change.
- PAUSED by user request before live JSONL integration and before running the
  new instrumented profile. Those pieces remain unfinished; no new throughput
  result claimed. User requests commit current work, then merge
  fiber-training-manager and integrate high-priority exclusive GPU acquisition.
- Pre-commit validation:51 tests pass in22.30s (demand, profile/config and split
  fixtures); git diff --check passes. Earlier native/layout suites are recorded
  in the layout report. Root playbook's named volume-cartographer dependency
  script is absent in this checkout; no setup/install attempted.
