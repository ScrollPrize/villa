# Plan: CI Repairs For 3D Lasagna And Fiber Inference

## Atlas Fixture

1. Replace the packed four-channel CZYX pred-snap fixture with four independent
   uint8 ZYX arrays and manifest groups.
2. Keep production channel binding strict; do not restore packed-array
   projection or other 4D handling.

## Fiber Inference Bounds

1. Keep shared `_ds_size` floor behavior for model/interpolation tensor sizes.
2. Add storage-coordinate ceil division for full OME level shapes and absolute
   output-region endpoints.
3. Add an odd-dimension regression that crosses an output-chunk boundary and
   verifies the last storage plane is included.

## Project Volume Reuse

1. Locate the loaded independently owned volume backing an incoming Lasagna
   channel location.
2. Accept reuse only when shape, dtype, fill value, base level, present levels,
   per-level shapes/chunks, and manifest-authoritative spacing match.
3. Do not require UUID equality: the Lasagna prepared wrapper has an intentional
   runtime identity distinct from an ordinary attachment of the same source.
4. Throw on missing or incompatible runtime backing so the existing attachment
   rollback restores project state.
5. Add tests for compatible independent ownership and incompatible spacing.

## Tests And Validation

1. Build `test_atlas` and `test_volume_pkg` with 32 jobs.
2. Run both C++ test binaries through CTest.
3. Run focused Fiber 3D inference tests, including the odd-edge regression.
4. Run `git diff --check`.

## Spec Update

- Clarify that OME storage shapes and region endpoints use ceil pyramid
  geometry even though model tensor downsampling remains floor-sized.
- State that VC3D Lasagna project channels are independent 3D ZYX volumes and
  independently attached source reuse requires metadata compatibility.

## Docs Updates

- Document odd-edge OME output bounds and compatible independent-volume reuse
  in the code-structure documentation.

## Changelog

- Record the CI fixture repair, odd-edge inference coverage, and validated
  Lasagna source-volume reuse.
