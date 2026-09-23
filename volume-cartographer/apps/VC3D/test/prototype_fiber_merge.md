# Fiber review merge prototype

This experimental CLI consumes one fiber's `linked` entries from a
`vc_fiber_patch_links` v1 JSON. It calls the production `proposeEdit`,
`addReviewOverlapTargets`, and `fitEdit(..., FitPolicy::FiberReview)` functions.
It does not call the old spiral-fitting rigid patch merger, copy its code, or
change the interactive workspace's fitting behavior. Source data is read-only.

The first pass fits in the workspace's deterministic proposal order. One
additional sweep reopens each placement against all currently fitted neighbors.
The fiber remains fixed. All objective weights, robust losses, distortion
limits, and validity checks are the existing review defaults. Output spacing
is independent of the review's fitting spacing.

Groups use the shared review matcher on original XYZ: matches within twice
review proximity (3 working voxels in the supplied example) and absolute normal
dot product at least 0.7. A chain of these matches stays in one group even if a
rigid fit would reject it. Dense
fiber-contact runs are recorded as diagnostics; distance-threshold dropouts do
not split otherwise overlapping patches. A patch without discovery eligibility
is reported as unlocated and blocks publication, rather than being dropped.

This is **per-fiber** physical connectivity, not a global union across every
fiber. Overlap anywhere on the assigned patches may connect them; it is not
limited to a narrow strip around the centerline. Patches can occur in results
from multiple fibers. Patch groups and exported surface components are distinct:
a group records assignment/contact connectivity; each patch-only TIFXYZ contains
one edge-connected region of complete quads. Single-quad islands are omitted,
including from diagnostic layer exports. Larger disconnected regions get
separate `_part_N` exports, preserving their graph origins and XYZ samples.

The opt-in shared renderer `renderMergedSurface` keeps fitted UV fixed, tapers
source weights at patch boundaries, and integrates source tangents across seams
before applying the shared metric regularizer. Merge position fidelity uses a
Huber loss at the review proximity radius so an inconsistent source edge cannot
dominate a seam. A signed-area penalty resists triangle reversal relative to
the blended source normals. Existing inversion/unsupported-cell checks still run.
The merge objective uses position weight `5/spacing`, tangent weight
`50/spacing`, and an orientation hinge at 5% of squared grid spacing with
weight 1000. A second hinge (weight 1000, minimum cosine 0.1) prevents
the two triangles of a quad from opposing each other. The exported quads must
pass a positive triangle-pair dot-product check. It allows up to 100 metric iterations; the ordinary renderer keeps
its original 12. These are prototype fitting choices, not relaxed validation
tolerances or guarantees of anatomical accuracy. Ordinary
workspace rendering and individual review fitting retain their existing behavior.
A fusion failure is reported, never treated as permission to export a bad mesh.

Without `--with-fiber`, only measured patch support is exported: real openings
remain. With `--with-fiber`, an additional `continuous/` export combines **all**
assigned patches with the workspace's generated fiber strip. It keeps the
component containing the most dense fiber samples and removes all detached
islands (including larger ones). The report records discarded component counts.
This surface contains synthetic strip geometry, explicitly marked in metadata;
it should be inspected against CT before use as measured sheet geometry.

## Build with existing dependencies

The harness uses the existing VC3D build objects and the repository's isolated
benchmark build helper. It needs a built `volume-cartographer/build/bin/VC3D`
and matching libraries; it does not install dependencies or replace VC3D.
From the repository root:

```sh
AGENTS_AGENT_MODE=1 python volume-cartographer/apps/VC3D/test/prepare_fiber_review_benchmark.py \
  --build-dir volume-cartographer/build --source prototype_fiber_merge.cpp
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build vc_constraints test_constraint_graph -j 2
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build \
  -f fiber-review-prototype_fiber_merge/build.ninja
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/prototype_fiber_merge --self-test
```

## Run and inspect

Output must be a new directory. Arguments after it are output grid spacing and
additional refit sweeps (defaults 10 and 1). Append `--with-fiber` to also export
a continuous strip-backed surface. Coordinates use the JSON's settings.

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/prototype_fiber_merge \
  /home/sean/Desktop/fiber_patch_links.json \
  anon_20260815T014816946_000002.json \
  spiral-fitting/outputs/fiber_merge_prototype/example_000002 10 1

AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-merge-matplotlib \
  spiral-fitting/.venv/bin/python spiral-fitting/preview_fiber_merge.py \
  spiral-fitting/outputs/fiber_merge_prototype/example_000002 \
  --ct-array /mnt/bigpc/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/0 \
  --xyz-scale 1
```

The chosen CT array is ZYX, shape `[18946,8174,8174]`, at the same coordinate
scale as these patch XYZ values. For a different array, set `--xyz-scale`
explicitly; the preview never infers it from a folder name. CT preview requires
a local, C-order, unfiltered uint8 Zarr v2 array and uses the existing
`zarr_chunk_io` decoder, including VCZ1 support. Missing chunks stay blank and
are counted. CT sampling is nearest-voxel after bilinear interpolation of
complete output quads; this does not alter the exported surface. Omit
`--ct-array` for a geometry-only preview.

Artifacts:

- `preview.html`: self-contained images and measurements, including failed
  groups shown as independent fitted layers, explicitly labeled as diagnostic.
- `group_*_preview.png`: source outlines, first-pass fusion, refined fusion,
  and source XYZ disagreement at common UV.
- `groups/group_*_refined/` and `_part_N`: separate validated TIFXYZ components.
- `continuous/{initial,refined}/`: optional single-component strip-backed TIFXYZ.
- `continuous_preview.png`: comparison of the two strip-backed exports.
- `layers/` and `initial_layers/`: independent registered patch layers.
- `review.json`: fitted placements reopenable in VC3D Fiber review.
- `report.json`: every fit, failure, assignment, and group, with original paths.
- `validation.json`: per-file quad connectivity, island and twist checks.
- `preview_metrics.json`: source-disagreement distributions, quad components,
  and shared-surface displacement relative to the source mean.

Source-disagreement before/after distributions use each pass's overlap support;
their sample counts may differ. They are diagnostics, not ground-truth accuracy
measurements. Additional paired distributions restrict comparison to identical
source membership at the same UV positions. The fit log's per-patch before/after RMS uses the
same targets within each refit, but neighbors change during the sweep.
An extra refit is not automatically better: the first real 11-patch example
reduces constraint residuals while increasing common-UV XYZ disagreement.
Inspect both `initial` and `refined` exports before choosing a fit objective.

To re-fuse saved placements without rerunning review registration:

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/prototype_fiber_merge \
  --fuse spiral-fitting/outputs/fiber_merge_prototype/example_000129 0 /tmp/group-0-merge
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/prototype_fiber_merge \
  --fuse spiral-fitting/outputs/fiber_merge_prototype/example_000129 all /tmp/continuous-merge --with-fiber
```

`GROUP_INDEX` restricts fusion to a group; `all` uses all groups. This replay
uses refined placements. Its compact report describes the exported components;
the full CLI produces the report/layers required by the CT preview script.

## Validation

Every preview run validates exported components first. To validate without CT
I/O or plots, pass `--validate-only` to `preview_fiber_merge.py`.

```sh
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 \
  volume-cartographer/build/bin/test_constraint_graph

AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-merge-matplotlib PYTHONPATH=spiral-fitting \
  spiral-fitting/.venv/bin/python -m pytest -q spiral-fitting/tests/test_preview_fiber_merge.py
```

The C++ self-test checks physical-overlap chains, separation without matches,
and disjoint contact runs. Python tests check interpolation without bridging
missing quads, shared UV-grid placement, XYZ/ZYX CT coordinates, scaling, and
missing chunks. Real-data runs use fibers `…000002` (14 patches) and `…000129`
(27 patches) from the supplied JSON. This build was exercised on Ubuntu arm64,
RelWithDebInfo, one OpenMP thread. macOS/amd64 have not been exercised.

The 14-patch example has physical groups of 1, 2, and 11 patches; the 27-patch
example has groups of 22, 1, and 4. All group merges now validate. Previously,
the 22-patch group failed its inversion check. Its refined output now has
53,095 quads in one connected component, with two single-quad islands discarded.
The gap along the fiber is an opening inside that larger connected patch region;
it is not filled in the patch-only output. The optional continuous 27-patch
output includes the generated strip, has 65,249 quads in one component, and
discards three single-quad islands. Both outputs have zero opposing triangle
pairs within a quad and pass the renderer's source-normal inversion check.

Regression tests cover a seam with inconsistent source positions, preservation
of patch-only gaps, connection through an explicitly added fiber strip, exact
component cropping without changing XYZ, omission of isolated quads, and
previewing all components without closing the gaps between them.

Validation for these regenerated examples: 71 C++ tests and six Python tests
pass. All 113 exported TIFXYZ files (including diagnostic layers and both
passes) contain exactly one component with at least two quads and zero opposing
triangle pairs. All 82 individual review-fit records match the earlier outputs;
the new fitting applies only during fusion. CT previews have no missing chunks.
Original example folders are retained with the `_before_merge_fix` suffix.

## Continuous-only batch

`--continuous-only` runs the same review fitting and final fusion as
`--with-fiber`, but writes only the refined continuous surface, its report,
and the reopenable review. It omits initial surfaces, individual patch layers,
and patch-only components. Incomplete assignments or failed continuous fusion
return a nonzero exit status. `report.json.complete` becomes true after the
review and final report have been saved.

The batch runner snapshots the links JSON, checks every exported surface,
and publishes one folder per assigned fiber under `strips/`. Fibers with no
assignments are recorded and skipped. Failures keep their reports and logs;
no invalid surface is published. Source patches/fibers remain unchanged.

```sh
AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-merge-matplotlib \
  spiral-fitting/.venv/bin/python spiral-fitting/batch_fiber_merge.py \
  /home/sean/Desktop/fiber_patch_links.json \
  spiral-fitting/outputs/fiber_merge_continuous_all --workers 6
```

Repeat the same command to resume unfinished jobs; add `--retry-failed` to
retry recorded failures. Each attempt gets a new directory. Resume requires
matching input, executable, shared constraint library, spacing, and refit
settings. `status.json` tracks batch progress; `jobs/*/status.json` identifies
running attempts and their logs. A filesystem lock prevents two runners from
using the same batch directory. Worker processes each use one fit/OpenMP/BLAS
thread. Concurrency does not change within-fiber fitting order.

After **all** export attempts finish, the batch assigns each published
`meta.json.uuid` to its strip folder's stem (the original fiber basename).
VC3D uses this field as its segment ID; the fitter's intermediate `refined`
directory name must not be shared by every published segment. The final pass
backs up the original metadata under `metadata_before_unique_ids/`, changes
only `uuid`, and verifies uniqueness in `metadata_ids_status.json`.
The pass is idempotent and also runs after resuming a batch.

For a batch already running under an older runner, queue the same final pass
without interrupting exports:

```sh
AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-merge-matplotlib \
  spiral-fitting/.venv/bin/python spiral-fitting/finalize_fiber_merge_ids.py \
  spiral-fitting/outputs/fiber_merge_continuous_all --wait
```

`--wait` acquires the export lock after the runner releases it. An incomplete
or interrupted batch is refused without touching segment metadata. Reload the
segments in VC3D after this pass so cached segment IDs are refreshed.

## Collision recovery and horizontal fallback

The September 2026 continuous batch published 526 merged strips. Of its 108
failures, 72 stopped at unlinked patch projections into the same output grid
cell; 36 stopped during patch fitting. The collision count alone is not a safe
merge criterion: the recorded pairs range from a single disagreement of about
two working voxels to tens of thousands of cells separated by over a hundred
voxels in 3D. Failed merge reports now include per-pair counts, UV location,
3D separation, normal separation, and local tangent projection diagnostics.
The renderer still rejects every unresolved unlinked collision.

`--shape-weight N` is an opt-in multiplier on the review patch's existing
symmetric Dirichlet shape energy. It keeps the original contacts, 4× maximum
stretch/compression check, and quad validation. With `N=16`, a retry of the 72
collision failures exported nine valid merged strips to
`spiral-fitting/outputs/fiber_merge_collision_recovery/`; the other 63 still
collided. The retry input contains only those nine successful fibers. This is
the command used before moving their source patches; resuming the completed
batch skips those exports, while a fresh refit requires restoring the moved
patches to the catalog folder:

```sh
AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-merge-matplotlib \
  spiral-fitting/.venv/bin/python spiral-fitting/batch_fiber_merge.py \
  spiral-fitting/outputs/fiber_merge_collision_recovery_links.json \
  spiral-fitting/outputs/fiber_merge_collision_recovery \
  --workers 3 --shape-weight 16
```

`--fiber-only` exports the generated strip from a horizontal fiber without
using any patch. Its metadata marks `constraint_graph.fiber_only=true`.
After excluding the six horizontal fibers recovered above, 59 horizontal
fibers with no assignments and 71 with failed merges received validated
fiber-only strips in `spiral-fitting/outputs/fiber_merge_horizontal_fallback/`:

```sh
AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-merge-matplotlib \
  spiral-fitting/.venv/bin/python spiral-fitting/batch_fiber_merge.py \
  spiral-fitting/outputs/fiber_merge_horizontal_fallback_links.json \
  spiral-fitting/outputs/fiber_merge_horizontal_fallback \
  --workers 6 --fiber-only
```

Both new batches verified one connected component, at least two quads, no
opposing triangle pair within a quad, and unique segment IDs. They have not
been visually reviewed against CT. The original 526 published strips remain
in their original batch; the recovery and fiber-only directories identify
their different geometry and provenance.
Saved `review.json` workspaces embed source geometry. The `--fuse` replay of
one recovered fiber was verified after its source patches moved; it can
rebuild a merged raster without repeating review registration.

The move plan `spiral-fitting/outputs/fiber_merge_patch_move_plan.json` lists
the source patches used by successful merged strips. It moves only patches
unused by every unresolved merge fiber. This moved 9,063 directories from
`graph_patches_only` to sibling `graph_patches_merged`; 986 patches shared with
unresolved fibers stayed in the original folder. All 4,562 patches required
by the 99 unresolved fibers remain available there. The move is resumable and
its status is recorded in `fiber_merge_patch_move_status.json`:

```sh
AGENTS_AGENT_MODE=1 spiral-fitting/.venv/bin/python \
  spiral-fitting/move_merged_fiber_patches.py \
  spiral-fitting/outputs/fiber_merge_patch_move_plan.json
# Add --execute to perform or resume the moves after reviewing the dry run.
```

```sh
AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-merge-matplotlib PYTHONPATH=spiral-fitting \
  spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_batch_fiber_merge.py \
  spiral-fitting/tests/test_preview_fiber_merge.py
```
