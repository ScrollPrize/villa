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

`manual_add_method_smoke.py` checks that applying or canceling a Manual Add
extension, or disabling editing during Manual Add, ends the session without
changing the selected growth method. It uses a 4x4 crop from a real tifxyz
surface in a disposable project; the source surface and user settings stay
untouched. Requires the existing `numpy`, `tifffile`, and `vc_object_store`
Python packages. Supply the matching CT volume (a local zarr or URL):

```sh
python apps/VC3D/agent_bridge/test/manual_add_method_smoke.py \
  --vc3d "$PWD/build/bin/VC3D" \
  --surface /path/to/real-segment.tifxyz --volume /path/to/scan.zarr \
  --artifacts "$PWD/build/manual-add-method"
```

The artifacts directory holds application screenshots after each operation and
the VC3D log, including when an assertion fails.

Pass `--follow-sheet` to exercise the asynchronous RK4 path as well: attempts
to apply, cancel, or disable editing must leave it running until completion.
Use `--platform xcb` on an X display if offscreen Qt cannot create an OpenGL
context. A software-rendered run can use `LIBGL_ALWAYS_SOFTWARE=1`,
`__GLX_VENDOR_LIBRARY_NAME=mesa`, and `QT_XCB_GL_INTEGRATION=xcb_glx`.
Use the `vc_object_store` build matching the C++ application's schema (for example,
`PYTHONPATH=../vc_object_store/build/python` from `volume-cartographer/`).
