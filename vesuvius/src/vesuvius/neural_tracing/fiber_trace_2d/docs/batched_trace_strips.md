# Many-fiber strip visualization

## Stable approved runners

Run tests without changing the approved command:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
```

Build/Python/suite settings are in `vesuvius/scripts/fiber3d_native_replay_tests.json`.
The inspection suite uses existing binaries and runs native replay, CLI, reporting,
metric and cache regression tests. No build/install or GPU evaluation is launched.
Explicit build-directory/Python arguments retain the original full test suite.
Evaluations continue through the existing approved
`/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py`
entrypoint and its workload configs; do not replace it with ad-hoc raw Python commands.

## Eligible reference coverage

`replay-set` now evaluates maximal contiguous runs between actual control points,
excluding extrapolated tails and CP spans with actual `interp_mode=lasagna`.
`interp_goal` does not determine eligibility. Adjacent retained trace/cspline spans
merge into one run. Legacy v1 uses its global optimization mode. Exact CP-to-line
matching is shared with the existing native fiber loader; invalid mappings fail.

Each run is evaluated in both directions, with fresh state, no synthetic edge or
failure across excluded gaps. The selected cohort still has 50 original fibers;
the filtered cohort has 90 eligible runs / 180 directed cases. Reports pool eligible
distance divided by failures plus directed runs. All-excluded fibers remain in the
selection audit, with no score. Old unfiltered results are not directly comparable.

`replay-references --config ...` exports the same native selection used for tracing.
The Python inference coordinator consumes it for low-priority reference prefetch,
records annotation SHA256s, checks them before launch, and native replay rejects
changed selection. Existing inference chunks are retained; legitimate tile overlap
may still touch excluded regions. The stable evaluation runner's `reference-plan`
mode audits and saves this selection without inference.

Annotation identity uses the canonical existing file path in both native planning
and Python binding. Test-split file/directory symlinks therefore match their resolved
targets; missing targets fail. Alias membership and annotation-content auditing are
retained separately; resolving identity does not deduplicate selected fibers.

`reference_run` records the original line indices and absolute arc endpoints.
Reverse local arc u maps to original_run_end−u. Images contain eligible runs only;
restart IDs are scoped by case/run and both references are retained in render.json.

Named `inference_cache_namespace` directories are resumable: rerunning reuses
completed chunks with matching source/model identities and computes missing
coverage. An existing directory is not an error. Fresh-inference timing tests
must choose an unused namespace. No metadata is relabeled to force compatibility.
Set `inference.flush_workers: 128` to increase the existing publication pool;
this setting does not change model identity. Model identity uses the checkpoint
and inference parameters, not implementation-file/runtime hashes. Existing
code-hashed directories are matched by their recorded parameters without moving
chunks or relabeling provenance; source and geometry checks remain strict.
Current implementation/runtime are recorded separately in the evaluation summary.

Enable `--render` on `live_evaluate` or native `replay-set`, or set
`"render": true` in the evaluation JSON (`evaluation.render` in a classic-runner
workload). This renders each completed direction directly from its in-memory
replay, without loading saved traces or changing the tracing policy.

Each `cases/<index>/` contains:

- `rendered/strips_N.jpg`: combined, aligned CT and presence rows.
- `rendered/render.json`: version3 inspection metadata, row/page ranges,
  source transforms, component placements, restarts and cross-section locations.

Each view (reference top/side, traced top/side) has adjacent CT and presence rows.
Reference is yellow; trace and failure markers are cyan. Reset components remain
disconnected, with eight black separator columns. Failure bands use the stored
pre-reset reference arc. The existing native overview builder, transported
frames and matched-point overlays are shared with the older renderer. A common
CT-coordinate raster targets **two base voxels per output pixel**, with no
subsequent enlargement (render_scale=1). Endpoint-aligned grids can have slightly
finer spacing. Base-pyramid CT starts at level1 when present, otherwise level0;
explicit selected arrays retain their declared transform. Labels name the actual
source level, trilinear interpolation and fine-to-coarse fallback. Presence samples
those SAME physical points.
Presence intensity is0–255, maximum across options. Legacy post-hoc CT-only
rendering retains its original8x API; new inline pages are intentionally different.

Cross-sections are aligned under the **traced top strip**, not an independent
gallery. Place dense thumbnail centers into free layout columns, then invert the
component raster and LineStripPositionMap to obtain the actual traced point.
Never interpolate across component gaps. Every restart has R<index>- at the old
committed endpoint and R<index>+ at the actual next start (if it exists).
Overlapping mandatory sections use additional lanes; section pages repeat their
parent traced strip when needed. CT/presence rows remain paired. Sections are
perpendicular to the local trace tangent, with the long acceptance axis vertical.
Their full extent is twice each acceptance radius (the ellipse diameter).
The faint ellipse is the trace-centered display tolerance, not the historical
GT-centered rejection ellipse. It uses the exact plane intersection of the metric's3D ellipsoid,
not an assumed fixed20x80 ellipse when tangent and normal are oblique. Invalid
normals give the metric's isotropic circle; longitudinal strip construction still
requires valid nonparallel anchor normals. Rendering currently requires positive
acceptance thresholds.

Restart labels `R<index>` refer directly to `replay.failures[].index` in trace.json.
Metadata retains directional reference arc, original-fiber arc (reversed back for
reverse traces), original-fiber arc inmm, reference/evaluator XYZ and reason.
Cross-sections record axes, radii, exact trace position/XYZ, role, restart ID,
strip column, parent strip page/rows, pixel bounds and section page/rows. Singleton
zero-progress segments have no longitudinal strip: their exact point is retained
and explicitly anchored to the nearest real component endpoint (has_longitudinal_strip=false,
anchor_segment). Their orientation falls back to the reference tangent; duplicate
trace points otherwise use the nearest distinct trace segment.

`trace_progress.md`, `.csv`, `.json` at the run root are generated from saved
traces while rendering continues; they include per-fiber length, restarts,
complete-partition mean and case/image links. Refresh manually without inference:
`run_classic_fiber_evaluation.py report-progress --config <workload.json>`.
The final `README.md` and `per_fiber.*` still audit all completed pages. Old
separate-modality directories remain readable by the reporting tool.

## Running

Native, precomputed predictions:

```sh
/tmp/vc-strip-vis-cmake/bin/vc_fiber_trace_metric replay-set --config /path/to/evaluation.json --render
```

On-demand inference (retain the normal GPU-manager configuration in the JSON):

```sh
PYTHONPATH=/home/hendrik/villa2/vesuvius/src:/home/hendrik/villa2 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.live_evaluate --config /path/to/evaluation.json --checkpoint /path/to/snapshot.pt --devices 0,1,2,3,4,5,6,7 --native-binary /tmp/vc-strip-vis-cmake/bin/vc_fiber_trace_metric --render
```

Use a new output directory. Existing running evaluations were not restarted or
modified to turn rendering on.

If a live source has no `ct_volume`, its existing `input_volume` pyramid is used
with `ct_base_volume: true`. This accepts local and remote CT through VC3D's
regular Volume/cache, checks uint8 level zero and matching base shape. Explicit
local selected CT arrays retain the legacy OME metadata transform validation.
Remote selected arrays are not supported as independent CT bindings: supply
their base pyramid. No silently guessed group transform is used.

## Readiness and limits

Strip samples may lie outside inference already requested by tracing. Presence
uses the case-local prediction field and waits for complete coherent products
before reading. CPU permits are released during inference and render-mutex
waits. Rendering finishes before native completion, so the inference coordinator
remains available. A source-local mutex limits concurrent full-ribbon memory;
different sources may render concurrently.

Presence rendering currently requires canonical presence/nx/ny bindings and
explicit source scale, checked before tracing. It samples complete prediction
products, not independent presence-only arrays: missing direction channel data
can suppress presence in offline partial datasets. Live publication provides all
channels together. Invalid direction vectors alone do not suppress presence.
The legacy post-hoc `render-replay` remains CT-only; the two-modality path is the
requested inline batch flag.

Rendering adds CT reads, CPU work and possibly inference tiles, so render-enabled
runtime is not a clean tracing-speed benchmark. Extra live tile requests can
change asynchronous floating-point accumulation order; fixed-data tests verify
unchanged replay artifacts. No fresh full-cohort rendering run was launched.

## Validation (2026-09-14)

Isolated RelWithDebInfo GCC13 build `/tmp/vc-strip-vis-cmake`, including its own
shared libraries (verified with `ldd`); ongoing evaluation builds untouched.

```sh
cmake --build /tmp/vc-strip-vis-cmake --target vc_fiber_trace_metric test_fiber_replay -j 8
ctest --test-dir /tmp/vc-strip-vis-cmake -R '^test_fiber_replay$' --output-on-failure
VC_REPLAY_TEST_BINARY=/tmp/vc-strip-vis-cmake/bin/vc_fiber_trace_metric PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest --noconftest -x -q vesuvius/tests/neural_tracing/test_native_replay_evaluation.py
```

Native replay suite passed, including source-to-working affine presence sampling
and existing disconnected components/long-strip pagination tests. Python:
17 passed, 1 skipped; covers identical CT inline/post-hoc bytes, unchanged traces,
both modalities, reverse direction, local HTTP CT, one-permit live demand waits,
and cold/warm identity-model inference. The skip is an existing optional case.
The identity adapter fixture was updated for the existing `emit_embeddings`
constructor argument; no training or production inference change was required.
# Traced restart labels

Traced top/side strips label the committed old endpoint `Rn-` and the next
trace start `Rn+`, matching cross-section restart IDs. Labels use exact segment
identity and are drawn after component assembly so short components do not clip
text. Singleton traces without a longitudinal strip retain cross-section labels
only; they are not reassigned to a neighboring segment. Existing JPEGs are not
modified by rebuilding the renderer.
# Reported restart counts versus visual IDs

Statistics group successive resets less than80 base voxels apart into one
episode (gap from preceding actual reset). Both direction and eligible-run
boundaries reset grouping. All visual reset IDs remain visible; reported
restarts and all distance quantiles use episode partitions including terminal
spans. Raw restart counts are not displayed. The CPU-only `cached-full` and
report-only `report-metrics` modes of the approved evaluation runner accept the
restart80 workload configs; neither uses GPUs or renders images.
## Current-policy ellipse support

Inspection uses the replay's `tangentialThresholdFactor` for cross-section
radii, threshold metadata and layout margins. Threshold40/factor2 therefore
renders40/80 acceptance radii, not40/160. The frame's long axis is vertical.
Defaultfactor4 remains supported; invalid factors fail validation. This changes
only rendering; the inference cache identity and tracing policy are unchanged.
# Cross-section horizontal alignment

Inspection top and side views share one longitudinal sample count per component,
derived from the ribbon centerline at two base voxels per pixel. Previously the
longest offset row set each view's width independently: curved ribbons stretched
the side and top differently, whereas sections followed top. Both views now use
identical component boundaries, gaps and panel crops. Own-trace failure bands use
the actual segment endpoint, including reference-match plateaus. Reference and
traced paths retain independent arc axes; inference and trace metrics are unchanged.
