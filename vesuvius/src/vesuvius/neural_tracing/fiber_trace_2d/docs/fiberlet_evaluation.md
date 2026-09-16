# Cached fiberlet evaluation

The existing live evaluation runner supports `evaluation.backend: "fiberlet"`.
Omitting it retains the ordinary beam tracer. Both consume the same frozen model,
source bindings, locked GT split, inference cache and eligible-reference plan.

## Pipeline and caches

1. Native graph replay asks the existing on-demand preprocessor for anchors and
   fiberlet paths. It uses the fast full-volume extraction/path kernels, not a
   Python implementation.
2. Missing prediction dependencies go through the existing priority queue and
   GPU inference/publication pipeline. Construction halos can extend well beyond
   the reference-only corridor. Cached inference remains usable.
3. Construction produces complete canonical storage chunks, never GT-cropped
   chunks. The shared metadata builder fingerprints source/model/normal identity,
   grid, construction parameters and storage representation. Replay thresholds,
   beam settings and worker counts do not invalidate construction.
4. Construction caches live at `<inference-cache>/fiberlets/<fingerprint>/`, with
   `anchors.zarr` and `fiberlets.zarr`. Identical bindings share the preprocessor
   in-process; an exclusive directory lock prevents competing producers.

This uses the existing split anchor/fiberlet cache layout. It does **not** import
the full-volume `Combined` export. Shared production code covers metadata,
canonical generation, float positions, compact directions and fixed-cost codecs;
the final packaging differs. No new construction math is introduced.

Each source has separate bounded anchor and path services; inner extraction uses
one thread as in full-volume chunk-parallel preprocessing. Decoded bytes share
the process cache budget. Waiting graph cases release their CPU permits.
Construction normals use the prediction lattice, while evaluation normals retain
the tracing lattice. Publication readiness also covers stored-grid reads and
hot-cache anchor resampling.

For the short200runner, `scroll_sequential: true` now evaluates one physical
volume at a time (bothPariscollections share that group). `construction_workers`
sets the active scroll's stage admission; the current config uses224logicalCPUs.
Anchor/path principal kernels and graph replay share compute permits. Path
dependencies are resolved before acquiring permits; inference-readiness waits
release them. Stage cache service threads remain separate so waiting paths cannot
block every anchor worker. Halo assembly/anchor-view resampling and I/O support
threads are outside this principal-compute budget. This is not a strict total
thread-count cap and not the full-volume finite-Z-slab dispatcher.

Path owner requests coalesce through exact-owner paired futures, including sparse
empty outcomes. No64hashed-lock collisions. Empty owners are memoized in RAM for
the preprocessor lifetime; nonempty canonical pair publication is unchanged.
Within each chunk, the shared path kernel deduplicates interpolation coordinates,
sorts them spatially and samples in batches; cross-owner halos can still reuse
the same decoded source chunks. Cross-chunk DP/sample batching is not introduced.

`benchmark-full` uses the existing demand GPU controller by default: no registration
until inference input is ready, pause/release after30seconds of GPUinactivity,
resume on more work. CPUtracing/construction and input/accumulation buffers remain.
`--unmanaged` explicitly bypasses this. One retained inference session spans scrolls.

## Configuration

```json
{
  "backend": "fiberlet",
  "fiberlet": {
    "storage_chunk_side_base": 512,
    "workers_per_source": 24,
    "beam_width": 16,
    "lookahead_base_voxels": 384,
    "beam_step_base_voxels": 48
  }
}
```

Put this under `evaluation` in a runner workload, or at top level in a native
`replay-set` config. Anchor/path parameters currently use the canonical production
defaults (anchor cells 4 prediction voxels; path neighborhood radius 4 cells).
Arbitrary anchor/path overrides and embedding graph scoring are not exposed by
this integration. Ordinary `trace_args` are **not** graph beam parameters;
use the `fiberlet` fields above. The historical `effective_trace` summary field
still describes those ordinary options; authoritative graph settings are in
each case's `graph.json`.

## Protocol and inspection

GT eligibility and both directions remain unchanged: CP-bounded, no extrapolation,
no Lasagna spans. The current protocol uses normal radius 40, tangent radius 80,
160-base-voxel excursion grace and 80-reference-voxel restart episodes. Grace
state survives graph edge boundaries. Terminal excursions outside the ellipse
are failures.

Graph seeding/reset follows the existing graph replay implementation: it selects
an eligible anchor rather than fabricating a point exactly on GT. A missing seed
can leave reference distance untraced. Therefore report `traced_length_mm` and
`length_weighted_success_percent` alongside restarts and length statistics; a
completed evaluation is not necessarily complete traced coverage.

Each case saves authoritative `graph.json` plus a `trace.json` geometry adapter
for the existing renderer. The adapter explicitly marks `geometry_backend` as
`fiberlet`; graph geometry has no fabricated per-point cumulative losses. Existing
CT/presence top/side strips and restart cross-sections are available with
`evaluation.render: true`, including empty-seed cases.

## Commands

From `/home/hendrik/villa2`, use the existing approved entrypoints:

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fiberlet_u22000_full_20260916_v2.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fiberlet_u22000_full_20260916_v2.json
```

The full workload references the completed bidirectional PHerc1451 pilot. Pilot
configs and a separate warm-cache output config are alongside it. The inference
namespace and frozen u22000 snapshot match the beam reference; full runs use the
GPU manager exclusively. See the benchmark report for measured results.
