# Object-store cutover verification

The cleanup uses `ObjectStore` transactions for mutations and read-only typed
views for queries. VC3D shares the store's native source preparation, fiber
collection and linked-component import, revision-pinned native export, and
segment clipping. Branch endpoints require UUIDs; there is no filename identity
repair, bare-array collection fallback, database migration, or typed editor
commit path. Retired publishing/sync workflows are outside this cleanup.

Surfaces follow the same rules: VC3D lists, opens and edits `surface` objects
only (no tifxyz reads), Open Data segments are surfaces of read-only per-sample
catalog stores, and tools run in `--store` mode.

Native export accepts an **object revision** pin, so metadata, coordinates and
applicable topology agree with the selected geometry. Python names this argument
`object_revision_uuid`; the CLI retains `--revision`. The return value is the
geometry revision UUID. Historical exports do not read current annotations.

## Build and checks

Use the existing `RelWithDebInfo` builds with `VC_TESTING=ON`. No setup or install
step is required. Run from `vc_object_store`:

```sh
AGENTS_AGENT_MODE=1 ninja -C build -j2
AGENTS_AGENT_MODE=1 ninja -C ../volume-cartographer/build -j2 \
  VC3D test_atlas test_fiber_widget test_fiber_view \
  test_line_annotation_structural_edits test_line_annotation_link_validation \
  test_line_annotation_generated_views test_line_annotation_fiber_deletion \
  test_spiral_input_workflow test_open_data_manifest test_ink_detection_overlay \
  test_segmentation_lasagna_panel_ui
AGENTS_AGENT_MODE=1 ctest --test-dir build --output-on-failure
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=build/python \
  python3 -m pytest tests/python -q -p no:cacheprovider
QT_QPA_PLATFORM=offscreen SPIRAL_TEST_PYTHON=/home/sean/villa/spiral-fitting/.venv/bin/python \
  ctest --test-dir ../volume-cartographer/build --output-on-failure \
  -R '(fiber|line_annotation|atlas|spiral|agent_bridge|object_store|surface_panel_store|surface_catalog|open_data|ink_detection|lasagna)'
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=build/python \
  python3 ../volume-cartographer/apps/VC3D/agent_bridge/test/object_store_smoke.py \
  --vc3d /home/sean/villa/volume-cartographer/build/bin/VC3D
QT_QPA_PLATFORM=offscreen python3 \
  ../volume-cartographer/apps/VC3D/agent_bridge/test/smoke_offscreen.py \
  --vc3d ../volume-cartographer/build/bin/VC3D
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  ../spiral-fitting/tests/test_bracket_fiber_links.py -q -p no:cacheprovider
```

The bridge smoke creates a disposable current-format fiber database and isolated
settings. It checks that `fiber.list` returns spans before any edit, VC3D's
collection export imports through the store, edits persist before Save, external
journal updates are observed, and edits survive a fresh VC3D process. For
surfaces it lists a user surface beside a read-only catalog store surface;
activation, tags, rename and review commit to the store; cropping commits a child
geometry revision; the catalog surface refuses edits until
`segments.create_editable_copy` copies it into the user's store. Two hosted
replicas exchange fiber and surface edits through a hub, and a lease held
elsewhere blocks editing. Supplying `--database PATH` instead runs against a
disposable backup of a current database.

Regression tests also cover advisory-coordinate differences during merge, the
clickable deferred-length label, exact auxiliary-channel edits, immutable history,
strict branch UUIDs, linked import, and historical native metadata/topology.

Results (2026-10-25): 9/9 store C++ tests, 36 store Python tests passed (one
skipped), 232/233 VC CTest entries (`python_sftp_volume` fails as it did before
the cutover), the store bridge smoke and 50/50 offscreen bridge smoke checks
passed. Surface tool store modes are covered by `test_tool_store_modes` and
`test_grow_seg_store_mode`. The `vc3d-mcp` tests were not run (the `mcp` package
is not installed).

## Fiber fix regressions

The journal-kind backport uses schema 16. Older stores are rejected; recreate or
reimport them rather than opening them with the new binaries. The dependent
tifxyz branch uses a separate schema version for its surface changes.

Build and run the focused checks from `volume-cartographer` with `VC_TESTING=ON`:

```sh
ninja -C build VC3D vc_fiber_constraints test_fiber_constraint_geometry \
  test_object_store_session test_object_store_fiber_json test_object_store_fiber_session
ctest --test-dir build --output-on-failure \
  -R '^(object_store_(session|fiber_json|fiber_session|vc3d_controller)|test_fiber_constraint_geometry)$'
VC_FIBER_CONSTRAINT_FIXTURE=/path/to/native-fiber-collection \
  ctest --test-dir build -V -R '^test_fiber_constraint_geometry$'
```

The optional collection check imports into a disposable store and trims an
unlinked fiber through the two-point threshold. It verifies that extraction
selects the current geometry despite the unchanged ingest summary. Input files
are only read. The standard tests also check narrow import pins, conflicts on
linked endpoints, journal kinds after deletion, old-schema rejection, and cache
preservation on unrelated, foreign-store, and hub-status events.

## Limits

Validation is on Ubuntu aarch64, using offscreen Qt. macOS and amd64 have not been
run; visible rendering and manual mouse interaction have not been inspected.
The old 765-fiber database at `/tmp/vc_object_store_smoke/spiral-complete.vcobj`
and its desktop backup lack required stable control-ID arrays. They cannot verify
the current schema and were not converted. Earlier real-CT claims in this document
predate the strict schema and do not establish current cutover coverage. A fresh
current-format real-data fixture is still needed for that check. No performance
speedup is claimed. Real Lasagna fits, a VC3D fetch of a real remote Open Data
tifxyz, and the background fetch in the UI have not been exercised end to end.
