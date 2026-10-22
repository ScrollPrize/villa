# Object-store fiber migration verification

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
ctest --test-dir ../volume-cartographer/build --output-on-failure -R '(fiber|line_annotation|atlas|spiral|agent_bridge|object_store)'
QT_QPA_PLATFORM=offscreen python3 ../volume-cartographer/apps/VC3D/agent_bridge/test/smoke_offscreen.py --vc3d /home/sean/villa/volume-cartographer/build/bin/VC3D
```

Results: 4/4 object-store C++ tests, 25/25 Python tests, and 54/54 selected VC
CTest entries pass. The separate bridge smoke reports 63 successful checks,
including 467 malformed-input probes. Existing Atlas21 fixture coverage is
conditional and was not exercised because that fixture was unavailable.

Added coverage includes mixed-object attachment/rollback/conflicts, serialized
writes and draining/cancellation, external journal notifications, UUID filtering
before spatial limits, budget/cancellation reporting, exact numerical comparison
with existing intersection predicates, transformed coordinates, historical atlas
reads after native files are removed, revision-pinned transport, branch editing,
control-point identity, immediate manual edits, split/merge and retirement.

## Real fibers and CT

Input fibers: `/home/sean/Desktop/spiral_dataset_graph_uuid/fibers`.
Import command:

```sh
AGENTS_AGENT_MODE=1 build/bin/vc_object_ingest \
  --db /tmp/vc_object_store_smoke/spiral-complete.vcobj \
  --author codex-smoke \
  --fiber /home/sean/Desktop/spiral_dataset_graph_uuid/fibers --no-progress
```

The import reported 765 fibers, 186 shared branch relationships and zero
problems. A pristine SQLite backup was saved for normal use at
`/home/sean/Desktop/spiral_dataset_graph_uuid.vcobj`. The native input files were
not modified. This database intentionally retains missing coordinate bindings;
the application requests explicit assignment when opening a fiber against CT.

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
