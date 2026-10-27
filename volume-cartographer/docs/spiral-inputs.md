# Spiral input list

A Spiral fit's patches, point collections and fibers come from the Spiral
service's object store and are frozen for the resident fit; see "Object store
and frozen inputs" and "Connecting from VC3D" in the
[Spiral fitting README](../../spiral-fitting/README.md). The workspace shows
them from the attached store at the revisions the fit pinned, and *Fit inputs*
in the Spiral panel reports their digest, counts and freshness.

The input list holds only local drawings: brush patches and drawn point
collections that a frozen fit does not take. The text filter applies to it.
Remove (or *Discard Local Changes* in its context menu) discards a drawing.
Leaving the workspace with drawings asks before discarding them.

Build and run the Spiral tests from the repository root using the existing
configured build directory:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build -j2 VC3D \
  test_spiral_brush_patch test_spiral_store_inputs test_spiral_service_protocol
QT_QPA_PLATFORM=offscreen \
  SPIRAL_TEST_PYTHON="$PWD/spiral-fitting/.venv/bin/python" \
  PYTHONPATH=<vc_object_store build>/python \
  ctest --test-dir volume-cartographer/build -R '^spiral_' --output-on-failure
```

`spiral_service_protocol` runs its own temporary service on loopback and is
skipped without `SPIRAL_TEST_PYTHON`.
