# Task plan

1. Audit the approved evaluation runner modes, active benchmark configs and
   existing benchmark result documents.
2. Add one benchmark runbook under `docs/` with an explicit taxonomy:
   authoritative five-scroll/50-fiber evaluation; frozen short200 regression;
   beam versus bounded-fiberlet short200; anchor coverage; corrected mini5;
   explicit contrast diagnostics; live-inference performance profiles; and
   historical model/embedding sweeps.
3. For each current benchmark, record its exact runner mode, config, immutable
   input/manifest, checkpoint identity, output root, expected coverage, result
   report, cache behavior and completion evidence. Mark every entry canonical,
   diagnostic, superseded, incomplete or historical.
4. Document the distinct lifecycles: ordinary full-cohort prepare/validate/run,
   frozen-manifest `benchmark-full`, saved rendering/review, extraction,
   anchor/fiberlet sweeps, profiling and contrast packaging/connectivity.
5. Keep the frozen short200 v3 evidence separate from the versioned GT-review
   draft; the latter is annotation-audit material unless explicitly promoted as
   a new frozen benchmark.
6. Document immutable/host-local evidence rules: never overwrite completed
   roots; a fresh run uses a copied config and new output/cache namespace while
   preserving scientific parameters, manifests and checkpoint hashes.
7. List helper/reporting CLI modes separately so they are not mistaken for
   benchmark families.
8. Link the runbook from `short_failure_benchmark.md`,
   `fiberlet_evaluation.md`, `anchor_coverage.md` and
   `live_inference_trace_evaluation.md`, and add a changelog entry.
9. Validate every documented mode/config against the runner and check links,
   completion references and formatting without launching a benchmark.

## Testing

- Parse each documented JSON configuration and map it to an actual argparse
  mode/config schema.
- Check repository paths and host-local canonical artifacts when available;
  absence of `/ephemeral` must not make repository documentation invalid.
- Invoke runner `--help` and run focused `test_classic_fiber_evaluation.py` and
  `test_short_benchmark_evaluate.py`; do not start GPU evaluations.
- Verify Markdown links and canonical config/report/output/count references.
- Run `git diff --check`.

## Spec update

No scientific benchmark semantics change. Require the benchmark runbook to label
scientific authority, selection bias, supersession/history, exact
config/mode/output identity and completion evidence.

## Docs updates

- Add `docs/benchmark_runbook.md`.
- Link existing detailed result documents from the runbook rather than copying
  their full result tables, and link the central index from the four general
  benchmark entry documents.

## Changelog

Record the new benchmark inventory/runbook.
