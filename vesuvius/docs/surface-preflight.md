# Fail-closed TIFXYZ/volume preflight

`vesuvius.surface_preflight` checks that a TIFXYZ surface is structurally
usable and paired with the intended Zarr/OME-Zarr CT volume before an
expensive render, label transfer, or inference run starts.

Install the geometry I/O dependencies and run:

```bash
pip install "vesuvius[label-transfer]"
vesuvius.surface_preflight \
  --surface /path/to/segment.tifxyz \
  --volume /path/to/scroll.zarr \
  --output preflight.json
```

For an OME-Zarr with a nonstandard hierarchy, `--array-key` may explicitly
select its base-resolution array. Non-base arrays are rejected because TIFXYZ
coordinates are expressed in base-resolution voxel space. By default the
command selects the first dataset declared by OME-Zarr `multiscales`, then
falls back to array `0` or the only array in the group.

The command checks:

- required TIFXYZ files, metadata, coordinate shapes, and optional mask
  compatibility, including canonical multipage and integer-scaled masks;
- at least one valid vertex and connected quad;
- finite selected coordinates;
- exact valid-coordinate bounds in the selected CT array, including an
  optional `--margin`;
- deterministic, evenly ranked surface samples for nonzero CT signal support.

The JSON report records every required gate and its observed value. Exit code
`0` means every gate passed; exit code `2` means at least one gate failed or an
input could not be read. The report is written atomically, so downstream jobs
can require a complete `PASS` report rather than guessing from partial output.

Signal support defaults to at least 95% of 1,024 deterministic samples with
absolute CT value greater than zero. Use `--minimum-support-fraction`,
`--max-samples`, or `--support-threshold` when a volume has a documented
different fill-value convention. Threshold changes are recorded in the
report.

A signal-support failure means the selected volume does not contain enough
nonzero signal at the sampled surface coordinates. This can indicate a wrong
pairing, a cropped, sparse, or partially populated volume store, or genuinely
unsupported surface coverage; it is not by itself evidence that the surface
geometry is defective.

This is an input-pairing preflight, not a proof of surface correctness. It does
not replace geometric diagnostics such as self-intersection or local
orientation analysis.
