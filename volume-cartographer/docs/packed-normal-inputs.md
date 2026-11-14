# Packed prediction normals

`LasagnaDataset::open` accepts a standalone manifest or a local Zarr root
containing a version-3 `lasagna` attribute in `.zattrs`. Relative channel paths
are resolved against the attribute file's parent. Remote inputs use the explicit
`.zattrs` URL.

The channel sampler accepts separate ZYX arrays or packed CZYX arrays. Manifest
channel order must match the packed C extent. Channel views decode the original
chunk with its original codec, then select the requested channel without writing
derived volumes. Missing chunks retain their missing/fill semantics. Existing
callers of `openLasagnaChannelArray` without a channel index still receive the
original array.

Fiber predictions distinguish **fiber direction** (`nx`, `ny`) from **sheet
normal** (`sheet_nx`, `sheet_ny`). Use `fiberSheetNormalDataset(prediction)` to
select the latter. The full channel list is preserved, including its indices.
`sheet_grad_mag` supplies normal validity; it is **not a measured winding
density** and must not replace Lasagna density in winding extraction.

Python sampling, in base XYZ coordinates:

```python
from vc.volume import NormalSampler
sampler = NormalSampler("prediction.zarr", normal_source="fiber")
normals, valid = sampler.sample(points_xyz_float32, threads=8)
```

The default `normal_source="lasagna"` retains the previous behavior. No source
is silently substituted. Fiber sheet normals always mask corners with zero
`sheet_grad_mag` before sign-invariant tensor interpolation. All-invalid
stencils remain invalid; no normal is invented or filled.

Spatial shapes use the declared base size and scaledown. Odd base extents use
ceiling division in this prediction format; do not infer a new voxel scale by
dividing rounded extents. AFV shape validation uses the existing same-level
coordinate-domain check (at most one base voxel difference per axis); it does
not rescale coordinates or allow a different pyramid level.

Validation:

```sh
cmake --build volume-cartographer/build/fiber-evaluation --target test_lasagna_normal_sampler test_lasagna_manifest -j 8
volume-cartographer/build/fiber-evaluation/bin/test_lasagna_normal_sampler
volume-cartographer/build/fiber-evaluation/bin/test_lasagna_manifest
```
