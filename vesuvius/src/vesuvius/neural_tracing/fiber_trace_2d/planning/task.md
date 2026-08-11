# Task: reload replay artifacts in napari

Allow the fiber-presence napari viewer to reload a regenerated replay bundle
without restarting the application. Reload only replay JSON/OBJ artifacts; do
not reopen or reload the external fiber-presence Zarr volume.

The reload must update the same kinds of artifact layers in place, preserve
runtime display state, strictly reject incompatible bundle geometry, and
recompute display-only distance data derived from the reference and failed
trace.
