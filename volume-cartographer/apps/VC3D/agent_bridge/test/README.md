# Agent bridge tests

`smoke_offscreen.py` is the hermetic integration test used by CI. It launches
the compiled VC3D binary with Qt's offscreen platform, checks all method
descriptors, and exercises transport and headless error paths without local
volume fixtures.

Run it in the bridge container:

```sh
docker exec vc3d-bridge bash -lc 'cd /work && QT_QPA_PLATFORM=offscreen python3 \
  apps/VC3D/agent_bridge/test/smoke_offscreen.py \
  --vc3d build/ci-release-gcc/bin/VC3D'
```

After changing a method descriptor, regenerate the checked-in description with
the same command plus `--update-description-snapshot`.

`object_store_smoke.py` runs the bridge's fiber and surface workflows against
real stores on a disposable database it builds itself. It needs the
vc_object_store Python extension on `PYTHONPATH`:

```sh
QT_QPA_PLATFORM=offscreen python3 apps/VC3D/agent_bridge/test/object_store_smoke.py \
  --vc3d build/bin/VC3D
```
