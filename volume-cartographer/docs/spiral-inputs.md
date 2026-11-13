# Spiral input list

A Spiral fit's patches, point collections, fibers and tracks come from the Spiral
service's object store; see "Object store, positions and input updates"
and "Connecting from VC3D" in the
[Spiral fitting README](../../spiral-fitting/README.md). The workspace shows
patches, point collections and fibers as the project's store for those kinds
holds them now, updating each one a commit changes (fibers
with *Show fitted fibers* in Display), and *Fit inputs* in the Spiral panel
reports their digest, counts and what differs from what the fit read,
including its links. Tracks and their crossing relationships participate in
the same input comparison, but track geometry is not displayed by this view.
Use VC3D and the service from the same checkout (API 47); enable tracks in
the fit settings after importing them into the service's store. There is no
track-DBM path selector.

The scroll's facts, the umbilicus and the Lasagna stores are the open
project's: every build request sends the service, which cannot read the
project file, each project volume's voxel size, a catalog volume's
orientation and full-resolution shape, the umbilicus the project's umbilicus
store holds for it, and the selected Lasagna dataset (a local one with its
identity; the service identifies a remote one by its URL), its nx, ny
and grad_mag stores and their scale relative to each volume (see "The VC3D
project's inputs" in the README). The fit reads the dataset's packed pools
from the service's cache by that identity: pack them with
`pack_resident_pools.py` and copy them there first (see "Lasagna inputs must
be packed first"). With the fiber-direction loss on, it reads the samples
extracted from the selected fiber-inference dataset at the fit volume's scale,
also from the service's cache by that dataset's identity
(`fiber_direction_samples.py --cache`). The fit uses the entry of the volume its inputs
name; import an umbilicus for that volume (File > Import Umbilicus) and
select the Lasagna dataset before building. For a volume outside the
catalog, choose the spiral's *Outward sense* in the panel (a fit setting).
The outer shell is the store's surface tagged `spiral-outer-shell`, shown
from the fit's inputs. There is no dataset: VC3D neither picks nor sends a
dataset directory, and the service reads no `spiral-scroll.json`. Set the
winding count in the fit configuration.

Winding-model supervision reads the service's winding-inference artifact
from the path the service was launched with (`--winding-inference`); VC3D
never downloads or interprets it. For a local service VC3D owns, set the
optional *Winding inference* path in the launch settings (changing it
restarts the service); a remote service's binding is configured on its host.
The panel's *Winding inference* line says whether the service can run
winding-model supervision, and why not; a fit enabling it without a usable
binding fails preparation. The panel's *Scroll* and *Store layout* lines show the session's scroll,
volume, umbilicus version and Lasagna dataset. Line annotation reads its
normals from the same selected dataset.

The input list holds local drawings: brush patches and drawn point
collections. *Save Drafts to Store* (Shift+E) saves the ready ones into that
store as checked writes, each expecting the version its edit started from:
a save is refused, overwriting nothing, when another editor changed or
retired that object since. *Update Fit Inputs* (or *Save and Update Fit
Inputs*) updates the resident fit's inputs in place at a position including
them, keeping its model and optimiser: the service reads the same store (a
file shared on its host, or the `vc_object_server` endpoint attached here),
so the saves are already there. When the service's store is a file on
another host the panel says so once; an update then names no position and
reads that host's current head, which the edits saved here are not in. The
text filter applies to the list. Remove (or
*Discard Local Changes* in its context menu) discards an unsaved drawing.
Leaving the workspace with unsaved drawings asks before discarding them.

Changing tracks or their crossings requires *Rebuild Fit* → *Rebuild with
Current Inputs*, as does changing the outer shell. With tracks enabled and
`track_exclusion_radius > 0`, updates also require a rebuild because the
track exclusion masks depend on the fit's trusted geometry. To edit against
a service on another host, serve its store with `vc_object_server` (see the
[object-store guide](../../vc_object_store/docs/hosted.md#serving-a-store)),
bind the service to that endpoint and attach `ssh://host:port` in VC3D.

Build and run the Spiral tests from the repository root using the existing
configured build directory:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build -j2 VC3D \
  test_spiral_brush_patch test_spiral_store_inputs test_spiral_store_edits \
  test_spiral_project_inputs test_spiral_service_protocol
QT_QPA_PLATFORM=offscreen \
  SPIRAL_TEST_PYTHON="$PWD/spiral-fitting/.venv/bin/python" \
  PYTHONPATH=<vc_object_store build>/python \
  ctest --test-dir volume-cartographer/build -R '^spiral_' --output-on-failure
```

`spiral_service_protocol` runs its own temporary service on loopback and is
skipped without `SPIRAL_TEST_PYTHON`.
