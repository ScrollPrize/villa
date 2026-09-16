# Short directed failure benchmark

## Accepted dataset (2026-09-16)

Saved at `/ephemeral/hendrik/fiber3d_train2/evaluations/short_failures_200_20260916_v3`:
`fibers/` holds200directed JSONpieces; `manifest.json` records hashes, source/model,
original arcs and minimal/padded confirming trials. Completed and validated.
Composition41best/50other/109Classic. Total275921.659834basevoxels/662.089848mm,
19.2815% of original one-way geometry,9.6407% of original bidirectional eval distance.
Median900basevoxels; range231.696–11100. Final resumed extraction21m34s.
Self-proximity excludes4wholeoriginalfibers. Best99reported events reduce to64
after exclusions,50passing reproduction/padding and41after deduplication.
The100voxel post-failure limit may prevent confirming some160voxel grace excursions;
rejection does not prove absence of a genuine historical failure. Accepted by user.

Common-model evaluation (not extraction producer scores):
`run_classic_fiber_evaluation.py benchmark-full --config
vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_best_u22000_20260916_v1.json`.
Reuses original cache with shared demand inference, one exclusive manager grant;
all200pieces use EMA999u22000 without embedding. Outputs and provenance under
`/ephemeral/hendrik/fiber3d_train2/evaluations/short200_best_u22000_20260916_v1`.
Selected failure dataset, not an unbiased replacement for original full evaluation.

Current execution uses config/output`v3`: `fill_missing: true` (default). Cache
misses invoke the existing on-demand inference pipeline with the original frozen
producer snapshot/settings and one exclusive manager GPU grant for the entire
extraction; published chunks are reused. All configured GPUs remain allocated
between batches, avoiding repeated handoffs with training.
Trials now use a rolling asynchronous queue across snapshots, not sequential
32-candidate groups. Up to eight model jobs run concurrently on disjoint GPU
subsets; a single-model category can use all GPUs. A slow search does not block
other candidates. A rolling frontier of up to128unresolved candidates limits
speculative cache expansion, admitting replacements as candidates resolve rather
than waiting for a whole fixed group. Winners are committed in original order, so
completion order cannot change deduplication or quotas. Canonical inference-cache
paths are reserved as a set to prevent simultaneous writers across model aliases.
Native trace-worker caps sum to the manager CPU budget (default128), decoded caches
share64GiB total, and loader/publication pools have separate per-job caps. Those
auxiliary pools add threads beyond the native-worker count;128 is not an aggregate
thread-count cap. Each lane is now a persistent spawned coordinator (separate
Python interpreter), retaining up to two model sessions. Readers, mmap stores and
GPU worker pools survive replay batches. GPU workers start lazily on first missing
inference, then remain loaded. The model's source catalog is opened once per
session. LRU eviction and overlapping cache-writer ownership close old sessions;
ordinary batch completion does not. Prior batch8 measurements reserved about50GiB,
so only one session per lane retains active CUDA workers. A model switch pauses
the old GPU workers but retains its CPU/readers/mmap session; switching back
restarts only its GPU workers. Queue scheduling favors the resident model until
its available work is drained, without fixed batch-count rotation. The full per-GPU budget is enforced by the existing allocator
guard; without an explicit budget, retain only one session. Cache-overlap components share a lane, preventing two
processes from retaining the same writer lock. A single-model stage assigns all
configured GPUs to that model. Native replay processes still restart per batch.
`progress.json` reports active model jobs, capacity, frontier and elapsed time.
`inference_summary.json` gives per-batch tile/publication counter deltas and
`reused_worker_pool`; worker-pool statistics are explicitly session-cumulative.
Use `--extraction-time-limit-seconds 60` for a bounded scheduling measurement;
it checkpoints and cancels/drains active work before releasing the grant. Cleanup
may extend beyond the requested duration. Resume with the same config.
Producer snapshots resolve from each evaluation's`full_bindings.json`; optional
`short_failures.checkpoints` maps producer names to alternative paths with identical
SHA. Optional`gpu_manager`settings and top-level`devices`control managed grants.
v2cached-only trials are imported only after scientific identity validation;
uncached trials and prior winner decisions are retried, never treated as failures
or successful traces. `reuse_trial_root`is optional; omit for a fresh extraction.

This is a targeted regression set, not an unbiased substitute for the full50-fiber test set.
Sources are the current-protocol saved beam evaluations listed in
`benchmarks/refinement_212757_ema999_best_current_20260916.json`.

## Selection and search

Before failure selection, screen eligible GT with a native uniform-grid segment
index. Query spacing<=4basevoxels; ignore arc separations<500basevoxels. Minimize
nearby-segment distance in the shared40/80normal-aware metric, constrained at500.
Any hit excludes the entire original fiber across directions/snapshots. Excluded
Lasagna/extrapolated geometry is not indexed; valid separate pieces are compared.
`self_proximity.json` records locations, ratio, normal validity, check counts/time.
This sampled test is not a certified continuous curve/curve intersection test.

Current config/output suffix is`v3` with on-demand inference. Earlier`v1/v2` trial
artifacts remain for audit; they are not a completed benchmark. Scientific identity
changes require a fresh root; batching-only implementation changes retain prior
revision metadata. Interrupted batches get new attempt directories. Speculative
cache misses trigger managed inference, never zero-filled or silently excluded.

Target200 unique locations: up to99 best-u22000 failures, up to50 failures from
other non-embedding evaluations where Classic has no raw failure within200base
voxels and complete eligible-piece coverage in both directions, then Classic
fills the remainder. Dedup distance200basevoxels within the same original fiber
and eligible piece. Historical candidates use the reported80-voxel restart episodes.
Dedup locations are reserved only after successful reproduction.

Search fresh starts in both directions at100-base-voxel distance increments.
Keep only first failures within±100original GT arc voxels of target; require full
160-voxel outside travel from saved geometry, rejecting terminal-only unresolved
excursions. Also verify the start100voxels farther upstream. Pick the shortest
verified padded piece; its end is at most100 beyond target. Stop the other
direction only when its shortest possible padded length cannot beat the winner.
No monotonicity assumption/binary search. A minimum is the shortest verified
result on this100-voxel grid, not a continuous optimum. Missing inference is filled
before deciding whether a smaller start reproduces the failure.
Boundaries cannot cross original CP/Lasagna splits.

## Run/resume

The single public entrypoint performs the entire pipeline, including the
self-proximity prefilter; no manual intermediate scripts are required. Native
tracing remains C++ and shared with ordinary evaluation. Python orchestrates it.
For another dataset use the same command with a different config. A minimal config:

```json
{
  "version": 1,
  "native_binary": "/path/to/vc_fiber_trace_metric",
  "output_root": "/path/to/new-benchmark",
  "short_failures": {
    "history_report": "/path/to/comparison.json",
    "baseline_name": "Classic",
    "total_cases": 200,
    "best_quota": 99,
    "other_quota": 50,
    "self_proximity": {
      "minimum_arc_separation_base": 500,
      "sample_spacing_base": 4
    }
  }
}
```

The comparison JSON uses the existing benchmark report schema: `rows`, each with
`name`, `path` to a completed native summary, and `metrics.global_result.failures`.
Original annotations, trace artifacts, normal manifests and inference caches must
remain available. `best_name` optionally pins the best producer instead of choosing
minimum failures; `evaluation_names` optionally chooses the other producers.
Baseline and best are always retained. Otherwise eligible non-`tuned` rows no worse
than baseline are considered. Classic/baseline backfills missing best/other slots.

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py short-failures --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short_failures_200_20260916_v3.json
```

CPU tracing uses shared PublishedStore/readiness/native beam replay. Missing cached
inference triggers the shared demand pipeline using the extraction's existing
exclusive GPU grant; it does not request a nested grant.
Published inference is reused; newly completed chunks extend that same cache.
The workload retains its source bindings; each trial actually uses the originating
evaluation's model/normal/trace configuration. No fiberlet tracing.
Old `short_failures.workers`/`candidate_batch` values in historical v3 configuration
are retained for identity-compatible resume, but no longer control scheduling.
The CPU budget is `evaluation.gpu_manager.cpu_cores`; GPU lanes use `devices`.

Outputs under `/ephemeral/hendrik/fiber3d_train2/evaluations/short_failures_200_20260916_v3`:
`protocol.json`, `candidates.json`, checkpoint `state.json`, individual trial audits
and native artifacts under `trials/` and `batches/`, then `fibers/` and `manifest.json`.
Manifest links original fiber/piece/arcs, original direction, model/config and both
successful trials. Exported line order is the selected tracing direction, with
`evaluation_direction: forward`; ordinary annotations still default to both.
Geometry is copied/sliced from eligible GT, not predicted trace geometry.
`optimization_mode` identifies the already-eligible non-Lasagna piece; full source
provenance remains attached rather than inventing a new refinement result.

Identity binds protocol, implementation/native hashes, historical summaries,
normal/inference manifests/cache identity. Historical annotations and consumed
trace artifacts are hash-checked. Incompatible resumes require a new output root.
Fewer than200 distinct reproducible locations are reported as a shortfall, never
filled with duplicates. Final lengths/results are pending real extraction.

Validation: approved native build;244Python tests passed/8skipped,23native replay,
84trace and11reference cases passed, including directed annotation and grace audit.
