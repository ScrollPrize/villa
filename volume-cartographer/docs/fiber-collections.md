# Fibers in an object store

Fiber geometry, annotations, stable control point identities, revisions, and
shared branch relationships live in an object store (`.vcobj`): by default the
open project's. Another store can be attached beside the project (**File →
Stores → Attach Store...**, or launch `VC3D --project p.vcobj --attach-store
/data/fibers.vcobj`) and chosen for fibers with **File → Stores → Show Fibers
From**, or with the fiber pane's store buttons. Stores attached beside a project
and the store chosen for each kind are remembered with the project on this
machine. Changing the fiber store cancels old background reads and closes the
fiber panes after accepted writes finish.

To share fibers with other people, keep the store as a hosted hub on a network
share. **File → Stores → Clone Hosted Store...** copies the hub into a local
replica and attaches it beside the project; **File → Clone Hosted Project...**
opens it as the project itself, so the team also shares its volumes. VC3D syncs
replicas in the background: every 15 s, and shortly after each local edit. The
status bar shows the sync state. **File → Stores → Sync Hosted Stores Now** syncs
immediately. Remote volumes (URLs) of a shared project are everyone's; a volume
at a local path is listed only for the person who attached it.

Enabling editing of a fiber, and any edit, tag or retire action, first leases
that fiber from the hub. If someone else holds the lease, VC3D reports who and
the fiber stays read-only. Leases of fibers still open for editing are renewed.
Other leases are released once their changes are pushed. When a fiber changed
both here and on the hub, VC3D asks whether to use the hub's version or keep
yours. See the store's [hosted hub
guide](../../vc_object_store/docs/README.md#hosted-hub-and-replicas).

Create the database outside VC3D, using the sibling object-store tools:

```sh
vc_object_ingest --db /data/fibers.vcobj --author reviewer --fiber /data/native-fibers
```

Inspect the reported problems before proceeding. Ingestion preserves source UUIDs
and imports a directory's branch graph atomically. Fiber transfer accepts the
current version-4 format; there is no legacy-file conversion or refresh path. Native JSON import/export in
VC3D is an explicit transfer operation; exported files are not live storage.
Surfaces and the point collections attached to them (corrections, tool results)
are objects of the same store; CT and inference volumes retain their existing
storage.

Open a CT volume and select a fiber in the **Fibers** panel of the status bar. When its
coordinate identity or scale is missing, the binding dialog shows the active
volume identity, dimensions and resolution, alongside known fiber metadata and
bounds. Enter **stored fiber units per active-volume voxel**. For geometry in a
level-0 grid viewed at level 2, this is 4. Only use that value when those frames
are known to match. The assignment is a checked database edit. Unbound fibers
remain listed but do not contribute to overlays or slice queries. Each displayed
fiber uses its own binding to the viewer's actual volume, including hit testing.
`meta.coordinates` is authoritative: `volume_uuid` stores the volume's `id` from
open-data `metadata.json` verbatim (for example `20260411134726`). VC3D compares
it directly with `Volume::id()`; it does not generate or canonicalize a UUID.
Its scale must be finite and positive. Advisory labels cannot override a different
volume ID. Existing derived-UUID bindings require an explicit metadata correction. Transfer preserves known and unknown coordinates for every
geometry kind.

Completed interactions and accepted solver results commit immediately. Save waits
for outstanding work and writes; it is not required to persist an edit. Conflicts
leave current database contents intact and refresh the affected working views.
A superseded solver result is discarded and outstanding work restarts against the
current revision. Split and merge publish their geometry, annotations, branch
changes and source retirement together. Ordinary and adjacent branches retain
their distinct winding semantics.

Catalog, overlay and spatial requests use `ObjectStore` directly through the
application's shared `ObjectStoreSession`. There is no fiber repository. Runtime
numeric handles map to store/object UUIDs; names and transport paths are labels.
Spiral requests
serialize pinned revisions into temporary transport artifacts while retaining
their existing wire formats; the Lasagna service reads surfaces from the store
itself (`{type: "surface", uuid, revision}` references). Fiber transport uses UUID filenames, with
display names stored separately. The object-store `fiberNetworkJson` exporter
reads pinned relationship edges and resolves stable control-point IDs into wire
indices, preserving reciprocal ordinary, adjacent, and pending branches. Adding
fibers to Spiral stages their complete linked components before applying inputs.
Changes refresh the tracked drafts; retirement removes the corresponding UUID
input. Explicit collection export/import uses the same UUID identities, so equal
display names do not collide. Service-only fibers are imported, with their linked
component, into the attached store before editing. Existing object UUIDs retain
their store contents. Reopened service inputs retain their service IDs through an
explicit object UUID association. Discarding a service draft does not undo edits
already committed to the object store.

The shared `GeometryIngest.hpp` API accepts either native files or decoded batches.
Generic `updateGeometryArrays` handles array/sequence edits across geometry kinds;
VC3D retains editing semantics and UI state. The same coordinate and artifact helpers
serve point collections and surfaces. Optional native
fiber annotations remain optional; import/export preserves unknown metadata and
branch identities, approval state, and stable control-point IDs.

The object store is the only source of fiber data; VC3D keeps no loaded fiber
set. The **Fibers** panel (status bar) is the single fiber UI: it attaches and
detaches the store, filters by minimum length and H/V family, sets the display,
and lists the store's fibers in pages of 200, longest first in the active
volume's voxels, followed by fibers not bound to that volume. Each row's details
(spans, metrics, H/V scores) are read from the store when it is selected or
expanded. Every edit, tag and link action reads the fiber's current revision and
commits a checked action; a link is one shared edge, so changing it never rewrites
the fiber at the other end. Opening a fiber in Line Annotation reads it and its
linked peers in one snapshot. Whole-store tools (detailed export, the fiber map)
query the whole store when they run.

VC3D queries in the active volume's voxels through the store's volume target
(`ObjectSelection::volume`, `RegionOptions::volume`), which spans every
coordinate frame bound to that volume; VC3D does not iterate frames itself.

The viewers draw the store's fibers near each slice from bounded region
queries: sampled lines, the control points of annotated fibers, and, with
**Show linked**, link-component colours and pending/approved link rings. Curved
surfaces (flattened and segmentation views) draw the control-point chains of
annotated fibers within the surface's bounds. Right-clicking a drawn control
point offers to open the fiber there. **Slice ±** is the one distance control.

**Max displayed fibers** defaults to **5,000 per viewport**, persists in settings,
and applies before geometry decoding. Zero means unlimited. Selected fibers have
priority within the limit when they intersect the view. Work/byte budgets can
return fewer fibers; the overlay reports partial coverage. Panning cancels obsolete
requests and discards results from old attachments. Drawn geometry retains the
same float64 values and exact clipping predicates.

Slice queries apply UUID inclusion before priority/limits and report incomplete
results for cancellation or resource/display limits. Their geometric clipping
and acceptance predicates are shared with the existing numerical helpers.

## Building and checking

CMake uses an installed `vc_object_store` package or the sibling checkout selected
by `VC_OBJECT_STORE_SOURCE`. It does not install or fetch that dependency.

```sh
AGENTS_AGENT_MODE=1 ninja -C build -j2
AGENTS_AGENT_MODE=1 ninja -C ../volume-cartographer/build -j6 VC3D \
  test_viewer_overlay_surface_primitives test_line_annotation_structural_edits \
  test_line_annotation_link_validation test_line_annotation_generated_views \
  test_line_annotation_fiber_deletion test_spiral_input_workflow test_object_store_fiber_json
ctest --test-dir build --output-on-failure
PYTHONPATH=build/python python3 -m pytest tests/python -q
ctest --test-dir ../volume-cartographer/build --output-on-failure -R '(fiber|line_annotation|spiral|agent_bridge|object_store|viewer_overlay_surface_primitives)'
```

With `VC_TESTING` enabled, the production controller test can additionally use a
**disposable** real database copy and CT volume:

```sh
QT_QPA_PLATFORM=offscreen \
VC3D_OBJECT_STORE_SMOKE_DB=/tmp/fibers-test.vcobj \
VC3D_OBJECT_STORE_SMOKE_CT=/tmp/ct-scale2 \
../volume-cartographer/build/bin/VC3D --test-object-store
```

The test edits its database. The optional CT input is a normal VC volume whose
base grid is the supplied level-2 data; its binding dialog is exercised with an
explicit factor of 4. The harness checks immediate publication, stale conflicts,
links, split/merge, retirement, pinned transport, shared workspace lifetime,
coordinate assignment, a CT read, and reopening.

Recorded datasets, exact commands, results and remaining platform/service checks
are in [the verification record](object-store-verification.md).

For a read-only large-store attachment check (no source database edits):

```sh
mkdir -p /tmp/vc3d-fiber-bench-config
QT_QPA_PLATFORM=offscreen VC3D_CONFIG_DIR=/tmp/vc3d-fiber-bench-config \
VC3D_OBJECT_STORE_READ_ONLY_BENCH=/path/to/paris4-fibers.vcobj \
  build/bin/VC3D --test-object-store
```

Use fresh settings for the default-limit assertion. The check prints attachment,
first-page and selected-fiber timings and verifies UI heartbeats, deferred line
geometry, bounded catalog pages, and clearing runtime identities on detach.
It also dispatches 20 status-only hub events against the loaded catalog and
reports mean, median and p95 time until the catalog is ready, plus catalog,
fiber-refresh and selection-reset counts. These events must preserve selection
and warm fiber snapshots without reloading geometry. Use a local replica when
testing a hosted store. This measures notification handling, not rendering FPS;
the regular smoke test separately checks that actual commits refresh the views.

The PHerc1451 save/cache regression uses the fixed 589-control fiber
`6af54546-cad8-4696-9f4a-763334378476` and its real relationship neighbourhood.
It clones and detaches the supplied local seed into a temporary directory;
the source is not edited. Run from the repository root with `VC_TESTING=ON`:

```sh
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target VC3D -j 6
mkdir -p /tmp/vc3d-fiber-cache-config
QT_QPA_PLATFORM=offscreen VC3D_CONFIG_DIR=/tmp/vc3d-fiber-cache-config \
XDG_CONFIG_HOME=/tmp/vc3d-fiber-cache-config \
VC3D_FIBER_CACHE_TEST_STORE="$PWD/local/benchmarks/pherc1451/seed/baseline.vcobj" \
  volume-cartographer/build/bin/VC3D --test-object-store
```

Fiber transport artifacts and JSON exports use the core named-temporary-file
atomic writer (fsync and rename), so publishing them does not require Linux
`O_TMPFILE` linking through `/proc/self/fd`. The regression checks exact transport
bytes on the real linked component, path reuse, and bundle creation/replacement.

The check also covers unchanged saves with reordered branches, exact coordinate and
annotation preservation, one-ULP edits, metadata and approval edits, stale
revision conflicts, peer edits before event delivery, link deletion/creation,
retirement, and transform invalidation. It compares cached reads with independent
fresh reads and checks that unrelated decoded fibers and side-strip snapshots
retain their allocations after a local edit. No CT sampling or solver runs are
needed. Save comparisons use the existing batch canonicalization and exact
values; decoded caches track peer generations and relationship identities.
