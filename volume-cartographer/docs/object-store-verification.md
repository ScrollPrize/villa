# Object-store fiber cutover verification

The cleanup uses `ObjectStore` transactions for mutations and read-only typed
views for queries. VC3D shares the store's native source preparation, fiber
collection and linked-component import, revision-pinned native export, and
segment clipping. Branch endpoints require UUIDs; there is no filename identity
repair, bare-array collection fallback, database migration, or typed editor
commit path. Retired publishing/sync workflows are outside this cleanup.

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
  test_spiral_input_workflow
AGENTS_AGENT_MODE=1 ctest --test-dir build --output-on-failure
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=build/python \
  python3 -m pytest tests/python -q -p no:cacheprovider
QT_QPA_PLATFORM=offscreen SPIRAL_TEST_PYTHON=/home/sean/villa/spiral-fitting/.venv/bin/python \
  ctest --test-dir ../volume-cartographer/build --output-on-failure \
  -R '(fiber|line_annotation|atlas|spiral|agent_bridge|object_store)'
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=build/python \
  python3 ../volume-cartographer/apps/VC3D/agent_bridge/test/object_store_smoke.py \
  --vc3d /home/sean/villa/volume-cartographer/build/bin/VC3D
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  ../spiral-fitting/tests/test_bracket_fiber_links.py -q -p no:cacheprovider
```

The bridge smoke creates a disposable current-format fiber database and isolated
settings. It checks that `fiber.list` returns spans before any edit, VC3D's
collection export imports through the store, edits persist before Save, external
journal updates are observed, and edits survive a fresh VC3D process. Supplying
`--database PATH` instead runs against a disposable backup of a current database.

Regression tests also cover advisory-coordinate differences during merge, the
clickable deferred-length label, exact auxiliary-channel edits, immutable history,
strict branch UUIDs, linked import, and historical native metadata/topology.

Results: 5/5 store C++ tests, 29/29 store Python tests, 54/54 selected VC CTest
entries, 4/4 branch-producer tests, and the fresh-database bridge smoke passed.
The Spiral workflow skips three optional real-fixture checks, and the brush
patch suite skips one; both test entries pass. `git diff --check` is clean.

## Limits

Validation is on Ubuntu aarch64, using offscreen Qt. macOS and amd64 have not been
run; visible rendering and manual mouse interaction have not been inspected.
The old 765-fiber database at `/tmp/vc_object_store_smoke/spiral-complete.vcobj`
and its desktop backup lack required stable control-ID arrays. They cannot verify
the current schema and were not converted. Earlier real-CT claims in this document
predate the strict schema and do not establish current cutover coverage. A fresh
current-format real-data fixture is still needed for that check. No performance
speedup is claimed.
