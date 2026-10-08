# Cached fiberlet evaluation — 2026-09-16

Fullv2 at~980native seconds:18/180cases, native lifetime CPU5352% versus2671%
in v1. These are process-average utilization samples, not like-for-like timing
benchmarks (v2 reuses v1 caches). Observed CPU activity doubled after24workers/source.

Rebuild completed; approved regression runner again passes234Python/8skipped,
23replay/84trace/11reference native cases (72.78s Python). Fullv2 manager task
2a7d6990-28df-44eb-b9f3-fbe6b319e81a acquired all8GPUs exclusively and resumes
the existing inference namespace, now24construction workers/source.

Full v1 interrupted after roughly43min:9/180 cases complete,~57000tiles,
~62000published chunks. Native CPU averaged2671% with only8 construction workers
per physical source: unnecessarily low for128 allocated logical cores. Resume
in full_v2 output with24workers/source (no math/cache-identity change), preserving
all published caches and the partial v1 artifacts. Manager Ctrl-C cleanup emitted
a child-wait TimeoutExpired; verified native evaluator exited before rebuild.
Coverage regression compiled separately without replacing the live library and
passed all11reference benchmark tests; then approved workspace rebuild started.

Baseline attribution verified: extracted the randomized local-scoring test and
its legacy helpers verbatim from HEAD into /tmp/fiberlet_head_local_scoring_repro.cpp,
compiled with g++13/O2/x86-64-v3/LTO against the unchanged local-scoring code.
It reproduces exactly105bitwise failures. git diff HEAD is empty for
FiberLocalScoring.cpp/.hpp/Internal.hpp. This is not introduced by integration;
no tolerances or math were changed. Temporary repro is outside the repository.
Retained full-volume producer_toolchain diagnostic metadata in its CLI wrapper.
Pilot coverage rechecked directly from actual graph segment starts:100% bothways.

Final independent review found graph credited coverage used the seed's search
start (resetArc), incorrectly crediting missing_seed_gap. Native outcome now uses
the actual seeded segment start, including failure-span accounting. Updated the
existing coverage regression to reproduce resetArc < seedArc explicitly. Full
cold run was already underway with the old reporting-only calculation; its paths
remain valid. Defer rebuilding until it exits, then run corrected warm cached
evaluation (no new inference) for authoritative coverage. Cold pilot's original
100% credited figure is provisional until the same reporting correction; restart
and complete-reference partition statistics are unaffected.

Cold pilot completed: PHerc1451 fiber dj_20260903T200017539_000006, two directions,
22.257707 directed mm, zero restarts, 100% credited coverage. Native342.169s,
coordinator346.133s; 2104 new inference tiles,2541 published chunks,67 warm chunks.
Warm output fiberlet_u22000_pilot_warm_20260916_v1: native1.067s,total4.198s,
zero inference/publication,306 warm chunks, both graph.json objects identical.
One cold/warm measurement each, not a statistical performance estimate.
Latest approved regression runner:234passed/8skipped plus23replay/84trace native
cases. New tests include inline graph rendering and empty-seed zero coverage.
Sandbox-only pytest invocations stalled before output; approved unsandboxed runner
completed normally. Full50-fiber job submitted as2c3c3d35-bd21-4a8e-b964-76adf734e943.
Docs added at docs/fiberlet_evaluation.md, including split-layout limitation,
canonical-only construction knobs, graph-specific seeding and effective_trace caveat.

Resumed integration. Independent review identified specs.md (plural), construction-normal lattice mismatch, cache identity/layout distinctions and CPU-permit waits. Fixed construction normals to predictionToBaseScale (evaluation normals retain traceScale), included manifest hash, reused metadata factory in full-volume CLI, canonical chunk settings and compact fixed-cost codecs. Graph cache waits suspend case CPU permits; separate bounded anchor/path services avoid dependency deadlock. Shared thread-local publication proof helper gates stored-grid construction reads; offline whole-volume scan remains guarded.

Validation: approved workspace build succeeds including vc_fiberlets. Approved replay runner: 23 replay and 84 trace native cases, 233 Python passed/8 skipped (includes new synthetic canonical cold/warm reuse test). Storage suite41passed. Fiberlet paths suite has105bitwise mismatches exclusively in existing randomized legacy local-scoring test; new grace-across-edges and all other cases have no failed checks. No local-scoring math was modified; baseline attribution still to be verified. Real PHerc1451 shortest-whole-fiber pilot running under manager task aca3a057-25ba-4b4e-9853-08ee2cb3c3cc, approved pilot command/config eval_fiberlet_u22000_pilot_20260916_v1.json; same frozen u22000 inference namespace, extra construction tiles requested.

Explicit layout distinction: live evaluation uses existing split Anchors/Fiberlets cache datasets, not the whole-volume Combined final export. Kernels, full-chunk generation, compact direction/fixed-cost representation and metadata builder are shared. Full-volume combined import/export is not added by this integration. Graph-specific seed scan/reset semantics remain explicit; report credited/untraced coverage alongside restart metrics so seed absence cannot masquerade as successful tracing.

Inspection follow-up: both u22000 piece and whole-fiber rankings show zero-restart entries as infinity mean distance and zero failures/mm, sorted last. Nonzero entries retain directed distance/(restarts+1). Validated all table rows and zero-density ordering programmatically; underlying trace artifacts and aggregate evaluation metrics unchanged.

User paused integration for an inspection-report update. Generated per-eligible-piece best-u22000 ranking from completed summary using shared reference_segments/segment_statistics; paired directions by source/fiber/run_index and joined case IDs from trace provenance. Verified 90 pieces, 180 cases, 99 counted restarts and unchanged 6867.627253 directed mm. Saved docs/benchmarks/refinement_212757_best_u22000_pieces_by_failure_density_20260916.md and linked from older fiber-level report. No evaluation rerun. All 180 visualization render metadata files now present. Integration edits are partial, unbuilt and intentionally paused.

Prior work committed as e8d53c15d. User authorized implementation, reusing fast full-volume preprocessing/tunings. Existing OnDemandPreprocessor, ChunkDataset and graph replay will be reused. Full-volume uses bounded chunk workers and single-threaded inner extraction. planning/spec.md and volume-cartographer/scripts/build_dependencies.sh do not exist; use existing workspace build runner without installing dependencies.
