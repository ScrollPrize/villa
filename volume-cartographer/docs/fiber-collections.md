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
and imports a directory's branch graph atomically. Native JSON import/export in
VC3D is an explicit transfer operation; exported files are not live storage.
Surfaces, point collections, CT, inference volumes and atlas directories retain
their existing storage.

Open a CT volume and select a fiber in the Object Store Fibers dock. When its
coordinate identity or scale is missing, the binding dialog shows the active
volume identity, dimensions and resolution, alongside known fiber metadata and
bounds. Enter **stored fiber units per active-volume voxel**. For geometry in a
level-0 grid viewed at level 2, this is 4. Only use that value when those frames
are known to match. The assignment is a checked database edit. Unbound fibers
remain listed but do not contribute to slice queries.

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
retaining their existing wire formats.

Slice queries apply UUID inclusion before priority/limits and report incomplete
results for cancellation or resource/display limits. Their geometric clipping
and acceptance predicates are shared with the existing numerical helpers.

## Building and checking

CMake uses an installed `vc_object_store` package or the sibling checkout selected
by `VC_OBJECT_STORE_SOURCE`. It does not install or fetch that dependency.

```sh
AGENTS_AGENT_MODE=1 ninja -C ../volume-cartographer/build
ctest --test-dir ../volume-cartographer/build --output-on-failure -R '(fiber|line_annotation|atlas|spiral|agent_bridge|object_store)'
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
