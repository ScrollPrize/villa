# Plan: resident training trace evaluation

1. Commit existing evaluation/config changes independently (done: e5245a30c).
2. Extract prepared-workload audit/execution from the CLI into a reusable
   library; CLI and training call the same preparation, audit and live runner.
3. Extend the shared tiled worker with an optional resident model, retaining
   preprocessing, product conversion, blending and shared-buffer behavior.
   Restore model mode and numerical backend flags after evaluation.
4. Give the demand pool an external GPU executor interface. The training
   implementation routes shared-slot descriptors to existing GPU workers
   over dedicated pipes, retaining model/optimizer state. CPU loading,
   accumulation and native tracing retain the optimized implementations.
5. Run CPU evaluation coordination in a cancellable supervisor thread. Keep
   manager polling responsive. Hold the cohort at its trace boundary, service
   inference in resident replicas, then resume the same cohort. On actual
   resource revocation, cancel/join evaluation and then perform normal drain;
   retain the pending trace request for retry after regrant.
6. Preserve immutable checkpoint provenance and best-by-restarts selection.
   Do not launch a fresh Python benchmark/model process or reload weights for
   each test. Existing CPU workers and native C++ tracing remain parallel.
7. Test library/CLI parity, model/optimizer identity and mode restoration,
   descriptor delivery, cancellation/reallocation, successful continuation,
   and trace result/selection using the approved test launcher. GPU validation
   requests GPUs 0–5 through the manager; 6–7 are excluded.

## Spec update

Replace trace-driven cohort teardown with resident inference. Only resource
changes/shutdown trigger model evacuation. Keep the manager protocol unchanged.

## Docs updates

Document library entrypoints, external inference workers, resource ownership,
and pause/retry behavior in cooperative_validation.md.

## Changelog

Record resident-model trace evaluation and validation results.
