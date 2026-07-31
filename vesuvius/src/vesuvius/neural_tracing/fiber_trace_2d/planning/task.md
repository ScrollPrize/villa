# CI Repairs For 3D Lasagna And Fiber Inference

- Update the Atlas pred-snap fixture to the supported Lasagna representation:
  one 3D ZYX array per channel, with no packed CZYX compatibility.
- Make Fiber 3D inference cover ceil-sized OME-Zarr edge voxels for odd source
  dimensions while preserving floor-sized model tensor semantics.
- Before reusing an independently attached project volume for a Lasagna
  channel, validate its source geometry, dtype, level/chunk layout, and
  manifest-authoritative voxel spacing. Reject incompatible reuse atomically.
