# Task log: circular winding visibility-mask controls

## Findings

- The previous implementation shifted only currently visible H/V keys and
  dropped unavailable targets. This did not model the operation as a circular
  roll of the entire visibility mask and left error/tie bits stationary.
- The H/V color helpers currently return bit-identical RGBA arrays, but the
  reported rendered result differs. Layer construction will therefore use one
  explicitly shared per-winding palette entry, with a fake-Napari regression
  on the actual arguments passed to both layer calls.
- Independent review fixed the rotation domain and direction: all managed
  nonempty state layers define sorted winding slots; Next moves each bit to the
  following slot and Previous to the preceding slot, both with wraparound.
  Sparse missing destination layers discard their incoming bit.

## Deviations

- None.

## Validation

- Focused viewer tests passed: 28/28 in 1.77 seconds with plugin autoload
  disabled to avoid the unrelated stale `zarr.testing` plugin dependency.
- Tests cover exact Next/Previous direction, wraparound, arbitrary masks,
  all-but-one-visible empty-space movement, all four states, sparse state
  layers, one-winding no-op, live fake-layer snapshots, and unmanaged-layer
  isolation.
- The fake Napari layer-construction regression confirms actual H and V
  `add_shapes` calls at one winding receive identical per-shape RGBA arrays.
- A real offscreen Napari object check could not run because `napari` is not
  installed in the current Python environment; no install was attempted.
- Ruff and `git diff --check` passed.
