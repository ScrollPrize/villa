# Object-store fiber cutover verification

Verified on Ubuntu 24.04, aarch64, with the existing `RelWithDebInfo` builds and
`VC_TESTING=ON`. No dependencies were installed or downloaded. Fiber consumers
use `ObjectStore` APIs through the shared, object-type-independent
`ObjectStoreSession`; no `FiberRepository` is introduced.

## Build and automated checks

Commands run from `/home/sean/villa/vc_object_store`:

```sh
AGENTS_AGENT_MODE=1 ninja -C build
AGENTS_AGENT_MODE=1 ninja -C ../volume-cartographer/build
ctest --test-dir build --output-on-failure
PYTHONPATH=build/python python3 -m pytest tests/python -q
QT_QPA_PLATFORM=offscreen SPIRAL_TEST_PYTHON=/home/sean/villa/spiral-fitting/.venv/bin/python ctest --test-dir ../volume-cartographer/build --output-on-failure -R '(fiber|line_annotation|atlas|spiral|agent_bridge|object_store)'
QT_QPA_PLATFORM=offscreen python3 ../volume-cartographer/apps/VC3D/agent_bridge/test/smoke_offscreen.py --vc3d /home/sean/villa/volume-cartographer/build/bin/VC3D
```

Final results: 5/5 object-store C++ tests, 26/26 object-store Python tests, and
54/54 selected VC CTest entries pass. The shared
Python reader and Spiral service tests pass: 234 tests and 48 subtests. The local
Spiral workflow suite uses its existing Python environment via
`SPIRAL_TEST_PYTHON=/home/sean/villa/spiral-fitting/.venv/bin/python`.
The local Spiral workflow reports 13 passed and three optional real-fixture
checks skipped. Existing Atlas21 fixture coverage is conditional and was not exercised because
that fixture was unavailable.

Shared Python/service checks, from `vc_object_store`:

```sh
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=../vesuvius/src:../spiral-fitting \
../spiral-fitting/.venv/bin/python -m pytest \
  ../vesuvius/tests/test_vc3d_fiber_format.py \
  ../vesuvius/tests/test_vc3d_fiber_format_tags.py \
  ../spiral-fitting/tests/test_service_editing.py \
  ../spiral-fitting/tests/test_service_editing_http.py \
  ../spiral-fitting/tests/test_spiral_helpers.py \
  ../spiral-fitting/tests/test_spiral_service_v2.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 \
  ../volume-cartographer/apps/VC3D/test/check_spiral_discard_and_mutation_state.py
```

Cutover regressions cover authoritative coordinate round trips for fibers, PCLs,
tracks and tifxyz, exact scale declarations, generic array edits, branch UUID and
approval preservation, stable control IDs, historical annotation reads, optional
fiber metadata, service-only imports, and existing service input identity.

Added coverage includes mixed-object attachment/rollback/conflicts, serialized
writes and draining/cancellation, external journal notifications, UUID filtering
before spatial limits, budget/cancellation reporting, exact numerical comparison
with existing intersection predicates, transformed coordinates, historical atlas
reads after native files are removed, revision-pinned transport, branch editing,
control-point identity, immediate manual edits, split/merge and retirement.

## Real fibers and CT

The real-data fixture is an existing database containing 765 fibers and 186
shared branch relationships, at
`/tmp/vc_object_store_smoke/spiral-complete.vcobj`, with a pristine backup at
`/home/sean/Desktop/spiral_dataset_graph_uuid.vcobj`. It predates strict version-4
transfer validation. The original JSON directory contains older wire data and is
not a supported current-ingestion input; no conversion or compatibility path was
added. Tests use disposable SQLite backups and leave the originals unchanged.
This database intentionally retains missing coordinate bindings; the application
requests explicit assignment when opening a fiber against CT.

CT source:
`/mnt/bigpc/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr`,
level `2`. For the controller smoke, `/tmp/vc_object_store_smoke/ct-scale2/0`
is a symlink to that level. Its `meta.json` is:

```json
{"type":"vol","format":"zarr","uuid":"2c4aae9a-4bb1-43f7-9a1d-d20afed20a02","name":"PHercParis4 scale 2 smoke","width":8174,"height":8174,"slices":18946,"voxelsize":9.6,"min":0,"max":255}
```

Before each real controller run, create a fresh disposable database copy:

```sh
python3 - <<'PY'
import sqlite3
with sqlite3.connect('/tmp/vc_object_store_smoke/spiral-complete.vcobj') as src:
    with sqlite3.connect('/tmp/vc_object_store_smoke/vc3d-real-final.vcobj') as dst:
        src.backup(dst)
PY
QT_QPA_PLATFORM=offscreen \
VC3D_OBJECT_STORE_SMOKE_DB=/tmp/vc_object_store_smoke/vc3d-real-final.vcobj \
VC3D_OBJECT_STORE_SMOKE_CT=/tmp/vc_object_store_smoke/ct-scale2 \
../volume-cartographer/build/bin/VC3D --test-object-store
```

The production controller loaded all 765 fibers, exercised the missing-binding
dialog with an explicit factor of 4 stored units per active voxel, and read a
2×2×2 CT crop at XYZ `[2852.54, 5537.44, 18095.1]`. It edited and reopened fiber
`003f5992-41da-44cf-add0-825f12961a3b`, then split, linked and merged its geometry.
After reopening, all 78 controls and four external branch relationships remained.
The 7606 resulting line points match the existing structural planner exactly;
they are not claimed to equal the original 7692 points because the existing
split/merge planner trims the join span/tails.

The reusable real bridge test uses a disposable backup and isolated settings:

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=build/python python3 \
  ../volume-cartographer/apps/VC3D/agent_bridge/test/object_store_smoke.py \
  --vc3d /home/sean/villa/volume-cartographer/build/bin/VC3D \
  --database /tmp/vc_object_store_smoke/spiral-complete.vcobj
```

It verifies browsing, a tag visible to an independent database reader before
Save, a subsequent edit after an external database commit, the Save barrier,
and both tags surviving a new VC3D process. A separate bounded `fiberRegion`
query against the pristine desktop database returned complete results with a
single-UUID inclusion filter.

## Verification limits

The actual application, Qt dialogs and production controllers ran offscreen.
Visible rendering and manual mouse interaction were not visually inspected.
macOS and amd64 builds were not available. Remote Spiral/Lasagna/atlas services
were not contacted; transport and stale-result handling use local regression
coverage. Superseded fiber solver work is rescheduled; a stale atlas request is
rejected and currently requires an explicit rerun/rebuild of its pinned atlas.
The editing controller currently materializes revision-backed working views
with a bounded read per object; very large catalogs may need lazy loading later.
No performance speedup is claimed.
