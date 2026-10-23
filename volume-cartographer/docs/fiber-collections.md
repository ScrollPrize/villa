# Fibers in an object store

VC3D uses one writable `.vcobj` for fiber geometry, annotations, stable control
point identities, revisions, and shared branch relationships. Choose **File →
Attach Object Store**, or launch `VC3D --object-store /data/fibers.vcobj`.
Attachment lasts for the session. A failed replacement leaves the current store
attached. Detaching or replacing it cancels old background reads and closes the
fiber panes after accepted writes finish.

Create the database outside VC3D, using the sibling object-store tools:

```sh
vc_object_ingest --db /data/fibers.vcobj --author reviewer --fiber /data/native-fibers
```

Inspect the reported problems before proceeding. Ingestion preserves source UUIDs
and imports a directory's branch graph atomically. Fiber transfer accepts the
current version-4 format; there is no legacy-file conversion or refresh path. Native JSON import/export in
VC3D is an explicit transfer operation; exported files are not live storage.
Surfaces, point collections, CT, inference volumes and atlas directories retain
their existing storage.

Open a CT volume and select a fiber in the Object Store Fibers dock. When its
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
geometry kind; atlas export adjusts the declared scale with the exported XYZ.

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
Atlas mappings record object and geometry revision pins. Spiral and service
requests serialize pinned revisions into temporary transport artifacts while
retaining their existing wire formats. Fiber transport uses UUID filenames, with
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
VC3D retains editing semantics and UI state. Shared coordinate and artifact helpers
are reusable by future point-collection and tifxyz integrations. Optional native
fiber annotations remain optional; import/export preserves unknown metadata and
branch identities, approval state, and stable control-point IDs.

Attachment does not load fiber controls, annotations or line arrays. The dock
aggregates coordinate groups in a background read, fetches catalog pages of 200,
and reads geometry from the spatial index for each visible viewport. The editor
hydrates a selected fiber and its connected branch peers on demand. Explicit
whole-store operations (such as detailed export or the complete fiber map) still
load the data they need.

**Max displayed fibers** defaults to **5,000 per viewport**, persists in settings,
and applies before geometry decoding. Zero means unlimited. Selected fibers have
priority within the limit when they intersect the view. Work/byte budgets can
return fewer fibers; the overlay reports partial coverage. Panning cancels obsolete
requests and discards results from old attachments. Resident geometry retains the
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
ctest --test-dir ../volume-cartographer/build --output-on-failure -R '(fiber|line_annotation|atlas|spiral|agent_bridge|object_store|viewer_overlay_surface_primitives)'
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
first-page and selected-fiber timings and verifies UI heartbeats, deferred editor
geometry, bounded catalog pages, and clearing the resident cache on detach.
