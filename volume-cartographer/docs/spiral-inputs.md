# Spiral input list

A Spiral fit's patches, point collections and fibers come from the Spiral
service's object store; see "Object store, positions and input updates"
and "Connecting from VC3D" in the
[Spiral fitting README](../../spiral-fitting/README.md). The workspace shows
them as the project's store for those kinds (all three routed to one store)
holds them now, updating each one a commit changes (fibers
with *Show fitted fibers* in Display), and *Fit inputs* in the Spiral panel
reports their digest, counts and what differs from what the fit read,
including its links.

The input list holds local drawings: brush patches and drawn point
collections. *Save Drafts to Store* (Shift+E) saves the ready ones into that
store as checked revisions, each pinned at the revision its edit
started from: a save is refused, overwriting nothing, when another editor
changed or retired that object since. *Update Fit Inputs* (or *Save and
Update Fit Inputs*) updates the resident fit's inputs in place at a position
including them, keeping its model and optimiser: a local service shares the
store file, so the saves are already there; with a hub-bound service the
update pushes them first. A service that can be named no position (a remote
one on a plain store) sees saved edits when it is rebuilt with current inputs. The text filter applies to the list. Remove (or
*Discard Local Changes* in its context menu) discards an unsaved drawing.
Leaving the workspace with unsaved drawings asks before discarding them.

Build and run the Spiral tests from the repository root using the existing
configured build directory:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build -j2 VC3D \
  test_spiral_brush_patch test_spiral_store_inputs test_spiral_store_edits \
  test_spiral_service_protocol
QT_QPA_PLATFORM=offscreen \
  SPIRAL_TEST_PYTHON="$PWD/spiral-fitting/.venv/bin/python" \
  PYTHONPATH=<vc_object_store build>/python \
  ctest --test-dir volume-cartographer/build -R '^spiral_' --output-on-failure
```

`spiral_service_protocol` runs its own temporary service on loopback and is
skipped without `SPIRAL_TEST_PYTHON`.
