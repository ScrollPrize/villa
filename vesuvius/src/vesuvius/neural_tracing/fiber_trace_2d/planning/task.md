Native 3D whole-fiber Trace2CP per-meter reporting must use only VC3D-supplied
volume metadata.

Use `record.sampler.volume.metadata["voxelsize"]` as the only physical voxel
size source. Interpret it as micrometers, convert to meters with
`voxelsize_um * 1e-6`, and use it to normalize the original loaded fiber-line
reference length.

If the VC3D sampler, volume metadata, or `voxelsize` field is unavailable,
missing, invalid, or non-positive, do not report physical units. Do not parse
Zarr/OME JSON directly in fiber code, do not inspect dataset config or record
metadata, do not accept alternate keys, and do not infer physical size from
filenames.

VC3D itself should always normalize the public Vesuvius
`scan/tomo/acquisition/detector/samplePixelSize` metadata field into
`metadata["voxelsize"]` when no explicit positive `voxelsize` is already
available. This should not depend on the remote volume base-scale mode.

Human whole-fiber progress/stdout should use compact labels `err/kvx` and
`err/m`, rounded to three decimals. Do not print `physical_unit=m` or reference
lengths in the progress line. Live progress must overwrite one terminal line
with carriage returns and only emit a newline at completion.
