# Scaling TIFXYZ coordinates without resampling

`vc_transform_geom --no-resample` converts coordinate units while preserving
every grid sample. Existing transform behavior is unchanged without this flag.
This mode supports positive uniform scaling, not affine transforms or OBJ files.
It also supports a directory containing multiple TIFXYZ directories.

For the Paris4 9.6 um surface and the 2.4 um scan:

```bash
volume-cartographer/build/bin/vc_transform_geom \
  --input /path/to/original.tifxyz \
  --output /path/to/scaled.tifxyz \
  --scale-before-affine 4 --no-resample \
  --target-shape-zyx 75784 32693 32693
```

XYZ is multiplied by 4; `scale` (samples per coordinate unit) is divided by
4. Thus a 20-voxel grid step becomes 80 and a 956 x 56913 grid stays that size.
Invalid XYZ samples are preserved. Bbox and voxel-space area/step metadata are
scaled, physical area stays unchanged, and voxel size is divided by the factor.
Historical `fit_config` remains unchanged; current voxel size is written at the
top level, using the fit-config voxel size if no top-level value existed.
The destination name comes from the output directory name; its UUID is newly
generated. Other coordinate/provenance metadata is retained.

If the input records `base_shape_zyx`, the actual target shape is required:
rounding at lower pyramid levels cannot be inverted to recover exact dimensions.
The shape option does not determine the scaling factor.

The streaming path supports float32 single-channel, top-left-oriented XYZ TIFFs.
Uncompressed stripped inputs use direct row reads, including byte-order handling;
tiled inputs retain one tile row. Compressed stripped input may require libtiff
to buffer a complete strip. Output uses uncompressed, native-endian stripped
BigTIFF, compatible with VC3D's mmap reader. No full XYZ grid is allocated.
Ancillary files (including masks and winding IDs) are copied without resampling.
Inputs with `corrections.json` are rejected rather than copying unscaled spatial
corrections. Subdirectories are not copied.

Output is staged beside the destination and renamed only after successful writes.
Existing output is never intentionally overwritten; caught errors remove staging.
A killed process may leave a uniquely named `.scale-*.tmp` directory, which is not
a completed output and is not reused by subsequent runs.

VC3D does not rescale or transform surfaces itself: stored surfaces open in any
volume the object store relates to theirs (by coordinate scale or a registration),
through the store's views. Its Transforms panel lists those volumes, registers a
new affine to the open volume and chooses among a pair's routes; the store does
the mapping. Renders launched from VC3D read the surface the same way
(`vc_render_tifxyz --store … --surface … --volume-uuid <volume>`).

For tifxyz folders and OBJs, `vc_transform_geom --affine transform.json [--invert]`
and `vc_render_tifxyz --segmentation … --affine transform.json[:inv]` read a
`{"transformation_matrix": [3x4 or 4x4 rows]}` file with vc_reader's `Affine`
(the type the store's registrations use). A singular matrix, or a file without
`transformation_matrix`, is an error.

Build and test:

```bash
cmake --build volume-cartographer/build --target vc_transform_geom test_transform_surface test_tool_store_modes VC3D -j8
ctest --test-dir volume-cartographer/build -R '^test_(transform_surface|tool_store_modes)$' --output-on-failure
```
