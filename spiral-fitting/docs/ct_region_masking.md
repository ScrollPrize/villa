# Mask prediction Zarrs with a CT envelope

`mask_zarr_from_ct.py` takes a CT **array path** (including the desired pyramid
level), followed by one or more target array paths or OME Zarr roots. It creates
new masked stores under `--output-dir`. Source stores are opened read-only;
existing destinations are rejected. For OME roots, **all declared pyramid
levels** are processed and their metadata is retained.

For exported winding rays instead of voxel arrays, see
[filter winding-inference rays](#filter-winding-inference-rays).

Run from `spiral-fitting/` with its existing Python environment:

```sh
python mask_zarr_from_ct.py \
  /home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr/2 \
  /home/sean/Desktop/spiral_dataset/lasagna_inputs/las_008_nx.ome.zarr \
  /home/sean/Desktop/spiral_dataset/lasagna_inputs/las_008_ny.ome.zarr \
  --output-dir /tmp/masked_lasagna_inputs
```

This creates `las_008_nx.ome.zarr`, `las_008_ny.ome.zarr`, and
`mask_report.json` in the output directory. The two channels share each computed
mask. Inside the mask, original target values are retained exactly; outside it,
values become zero. Existing resident-pool sidecars are not copied; build new
ones from these new stores if using the spiral fitter.

## The reviewed mask

All settings are in **selected CT-array voxels**, not level-0 coordinates:

1. Gaussian-smooth CT intensity (`--ct-sigma 1`), then threshold (`--threshold 65`).
2. Close narrow gaps (`--seed-radius 3`), fill enclosed holes, and retain the
   largest 4-connected component. Equal-size ties follow raster label order.
3. Apply wider closing (`--closing-radius 12`) and fill holes again.
4. Smooth the signed Euclidean distance field (`--boundary-sigma 4`), retaining
   its positive region. This smooths bridges and protrusions without replacing
   the whole contour with a convex hull.
5. Constrain the outer boundary to positive finite CT support, then fill holes
   and retain the largest component again. This order includes even zero-valued
   enclosed CT cavities. An empty or below-threshold slice stays empty.

The dim rim is excluded, detached fragments are removed, and enclosed internal
gaps are retained. The **mask on the CT grid** has at most one component;
nearest-neighbor sampling onto a coarser target grid can change topology.
Background hole filling uses an explicitly padded, 4-connected OpenCV flood;
it matches SciPy's default hole-filling result and is substantially faster on
these slices. Intensities are not automatically normalized: threshold 65 is
the reviewed setting for this uint8 CT, not a general CT intensity convention.

## Batches, threads, and memory

The default batch depth is the least common multiple of the CT chunk depth and
all target chunk depths expressed in CT coordinates. For this dataset it is
**128 CT slices**. Thus each CT chunk is decoded once per batch and each target
chunk is written by exactly one worker. There are no overlapping chunk writes.

The batch is read with parallel chunk workers, its independent 2D masks are
built with `--mask-workers 4`, and target chunks are then processed with
`--workers 2`. Queues are bounded, and workers share the in-memory mask batch
without process serialization. CT storage is released before applying masks.
OpenCV and Blosc internal thread counts are limited to one to avoid nested
thread pools.

For the 2044 x 2044 uint8 CT and a batch depth of 128, CT + mask buffers together
occupy about **1020 MiB**, plus slice-worker scratch space. Lowering
`--mask-workers` reduces concurrent scratch space. `--batch-depth 64` reduces
the batch buffers to about 510 MiB but decodes some CT chunks twice. A custom
depth must align all target chunk grids; invalid values are rejected.

Chunk handling avoids unnecessary work:

- Fully excluded chunks are left absent (implicit zero).
- Fully retained, present chunks are copied without decompression/recompression.
- Boundary chunks are decoded, masked, and re-encoded.
- Missing source chunks honor the source fill value; output fill is always zero.
- Targets with the same shape, chunks, and mapping share each chunk's mask lookup.

## Grid alignment and choosing a CT scale

The default assumes coincident origins and identical finest target/CT shapes.
For the supplied stores, **CT level 2 matches normal level 4**, and normal level
5 has twice the voxel spacing. OME scale ratios are used *within each target
pyramid*, rather than comparing absolute level numbers or incompatible root
scale conventions. Translated targets are rejected.

If the finest target shape differs from the CT, explicitly specify
`--target-scale Z Y X`: **CT voxels per finest target voxel**. For example, when
using a CT grid twice as coarse as the finest normals, use
`--target-scale 1/2 1/2 1/2`. Shape/extent compatibility is checked. All input
targets must follow the same declared finest-grid mapping.

Sampling is zero-origin, cell-aligned nearest neighbor:
`ct_index = floor(target_index * scale)`. Masks are sampled directly for each
chunk; no full-resolution upsampled volume is allocated. Different origins,
axis orders, rotations, and arbitrary registrations are not inferred from
shape. Input CT arrays must be ZYX.

Selecting a coarser CT level also changes the units of the Gaussian sigmas and
closing radii. Adjust those options and inspect the resulting masks before
using a different level. The default full command above preserves the exact
native-scale method that was reviewed; there is no automatic downsampling.

## Partial tests and reports

Add `--z-range 2048 2176` for a 128-slice test in **CT coordinates**, with an
exclusive upper bound. These output stores retain full shapes and coordinates,
but are **zero outside the requested range**, not copies of untouched source
data there. Partial outputs are marked in both root attributes and the report.

The report records grid mappings, mask settings, Z coverage, worker counts,
timings for each stage, and chunk counts. `boundary_zeroed` counts nonzero
voxels zeroed in decoded boundary chunks only; it does not count voxels in
chunks skipped without reading. `complete: true` means the requested range
finished; check `partial` to distinguish a test from a full-volume run. Failed
runs remain marked incomplete and cannot be silently resumed or overwritten.

## Supported stores and verification

Supported inputs are local numeric 3D Zarr v2 arrays without array filters,
using `.` or `/` chunk keys and either C or Fortran array order. Standard
numcodecs compression and legacy VCZ1 CT reads are supported. Targets using
VCZ1/VC-Delta3D codecs are rejected; use lossless Blosc, gzip, zstd, or
uncompressed target arrays. Zarr v3, sharded/remote stores, channels/time axes,
and translated grids are outside this script's scope.

The new implementation was checked pixel-for-pixel against all eight reviewed
masks, including the empty slice, broken end, and small last-slice region.
Regression tests cover hole-filling equivalence, component selection, rims,
notches, sparse fills, partial edge chunks, uint16 and Fortran-order values,
multiscale mapping, worker counts, and unchanged sources:

```sh
python -m pytest tests/test_ct_region_mask.py -q
python -m pytest tests/test_sparse_cuda_cache.py -q -k legacy_vcz1_reader
```

`zarr_chunk_io.py` contains the shared chunk-reading helpers extracted unchanged
from `pack_resident_pools.py`; both tools now use that implementation.

## Measurements on the supplied normals

Measured on this Linux aarch64 workstation using the existing Python 3.14
environment and installed native OpenCV/SciPy/numcodecs builds, without changing
precision or morphology settings. Inputs were the CT and nx/ny roots in the
full command above, including normal levels 4 and 5. No cache flushing was
performed, so these are warm/partly warm local-filesystem measurements.

The eight-slice range `--z-range 2104 2112` was run three times per configuration
into distinct new output directories:

| Configuration | Mean | p50 | p95 | Min–max |
| --- | ---: | ---: | ---: | ---: |
| Initial SciPy hole filling, 1 mask / 1 chunk worker | 4.579 s | 4.202 s | 5.295 s | 4.118–5.417 s |
| Equivalent flood filling, 4 mask / 2 chunk workers | 1.714 s | 1.639 s | 1.909 s | 1.562–1.939 s |

That is a 2.67x improvement in mean elapsed time for this small partial test.
The serial mask profile at z=4500 initially spent 0.197 of 0.382 seconds in
hole filling. After the equivalent flood implementation, all three hole fills
took 0.008 seconds and the complete mask took 0.190 seconds. Top remaining
costs were Gaussian filtering, connected components, closing, and distances.
Both implementations were checked against all reviewed masks; 250 randomized
topology/edge cases also compare the flood directly with SciPy.

A separate 128-slice batch (`--z-range 2048 2176`, four mask workers) was run
once per chunk-worker setting to tune I/O:

| Chunk workers | Total | CT read | Mask generation | Chunk application |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.93 s | 0.53 s | 10.13 s | 5.24 s |
| 2 | 11.76 s | 0.52 s | 9.17 s | 2.06 s |
| 4 | 12.95 s | 0.38 s | 8.42 s | 4.14 s |
| 8 | 13.46 s | 0.40 s | 8.30 s | 4.75 s |

These one-run tuning measurements have cache/order noise; they favor two I/O
workers for these tiny chunks, rather than proving a universal optimum. Each
run copied 8,466 chunks, encoded 1,836 boundary chunks, and skipped 26,562 chunks.
All 36,864 chunks were then checked voxel-for-voxel against source values and
the expected mask. All 10,302 stored chunk payloads matched byte-for-byte across
1/2/4/8 I/O workers. A separate 32-slice test at z=4480:4512 exercised the broken
end. Detailed local reports are in `/tmp/ct_mask_script_test/`.

To repeat a current small benchmark, use the full example command with
`--z-range 2104 2112 --workers 2 --mask-workers 4` and a fresh output directory
for each repetition. Each run writes its own timings to `mask_report.json`.

The final full-volume run processed all **4,737 CT slices and both levels of
both normal stores in 409.81 seconds (6 min 50 s)**, using the default 128-slice
batches, two chunk workers, and four mask workers. It copied 280,300 chunks,
masked 74,068 boundary chunks, and skipped 1,019,840 chunk positions. The output
copies are in `/tmp/ct_mask_script_test/full_volume/`, about 4.1 GiB per channel
on disk. Originals were not modified.

The full run's central 128-slice batch matches every stored payload from the
exhaustively verified partial test. Another 28 complete output planes across
seven Z locations (including empty slices and both ends) were compared against
source values and the expected masks, across both channels and both levels.
`verification.json` and `batch_verification.json` in the full output directory
record these checks. The full run includes internal cavities even when their
CT values are exactly zero; that clarification did not change any of the eight
reviewed example masks.


## Filter winding-inference rays

`mask_winding_from_ct.py` uses the same `ct_region_mask.make_ct_mask` defaults
and writes a new exported crossing store. It removes an **entire ray and every
crossing on it** when any part of the original ray leaves the CT mask. Both
relative-winding and adjacent-passage density supervision use those same
records, so both are filtered together.

```sh
/home/sean/villa/spiral-fitting/.venv/bin/python \
  /home/sean/villa/spiral-fitting/mask_winding_from_ct.py \
  /home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr/2 \
  /home/sean/Desktop/spiral_dataset/winding_inference \
  --coordinate-scale 0.25 0.25 0.25 \
  --output-dir /home/sean/Desktop/spiral_dataset/winding_inference_masked
```

These rays use reference **scale-0** coordinates, while this CT array is
**level 2**, so the explicit ZYX factors are one quarter. Output coordinates
stay in their original reference system. This requires matching axes and a
shared zero origin; no registration or translation is inferred. The required
`--coordinate-scale` factors specify CT voxels per reference voxel. Other
scans need their own mapping and reviewed mask parameters.

For this artifact, `source_attributes.ray_length` is 384, so the checked
segment is `origin + t * step` for **all t from 0 through 383**. This includes
unused tails beyond the first and last exported crossings. Even if every
exported crossing is inside, an exterior tail removes the whole ray. The
script refuses artifacts missing the full ray length rather than guessing it.

The segment test traverses every intersected CT grid cell, including the
spaces between crossings, instead of using a fixed sampling interval or
checking only endpoints. At simultaneous grid-plane crossings, neighboring
cells are conservatively checked too. Grid cells use the same `floor`
convention as the Zarr masker. Enclosed internal cavities remain inside the
mask. Nonfinite or out-of-volume rays are rejected.
Stored crossing points are also checked using the fitting loader's float32
arithmetic, guarding against rounding across a mask boundary.

Default staged thread pools are **2 CT read workers, 4 mask workers, and 4 ray
workers** (`--workers`, `--mask-workers`, `--ray-workers`). Rays are checked in
bounded batches of 16,384 (`--rays-per-task`). The mask is generated in CT
chunk-depth slabs, then held in a temporary, memory-mapped **bit-packed**
volume restricted to the necessary Z interval. At 2044 x 2044, storage is
about 0.5 MiB per slice, and a 128-slice uint8 CT slab occupies 510 MiB, plus
slice worker scratch. `--temp-dir` selects the temporary storage location.
The temporary mask is removed automatically when the operation ends.

Input checksums and manifest identity are verified. Retained ray origins,
steps, seed windings, crossing positions, winding labels, and order remain
unchanged; only ragged offsets and counts are rebuilt. Arrays receive new
checksums and the store gets a new fingerprint. The shared identity helpers
live in `winding_inference_io.py` and are also used by the fitting loader.
`filter_audit/*_source_ray_indices.npy` maps retained rows back to each input
shard. `mask_report.json` records counts, settings, mappings, and timing.
Existing output directories are rejected. An interrupted output lacks its
root `manifest.json`, preventing the fitting loader from using partial data.

To actually use the filtered store in fitting, set **`paths.winding_inference`**
in the dataset's `spiral-scroll.json` to the new directory, then start/restart
the fit. A currently running fitter has already loaded the old rays into GPU
memory and will not notice new files automatically. This affects the exported
winding-inference input; independent normal, fiber, or other input stores must
be masked separately with their corresponding tool.

Verification:

```sh
python -m pytest tests/test_mask_winding_from_ct.py \
  tests/test_ct_region_mask.py tests/test_winding_supervision.py -q
```

Tests cover thin excursions between samples, leaving and re-entering the
mask, simultaneous grid-plane crossings, integer endpoints, nonfinite and
out-of-volume coordinates, whole-record retention, empty outputs, checksums,
source preservation, and loading/sampling the filtered artifact for both loss
types. A separate plane-intersection oracle checks randomized segment tests.


The supplied seven-shard store was filtered into
`/home/sean/Desktop/spiral_dataset/winding_inference_masked`. The run checked
1,460,171 full rays and retained 1,147,551 (removed 312,620, or 21.41%). It
retained 22,706,484 of 26,743,553 crossings. Mask generation covered CT Z
`[1829, 4423)` and took 178.9 seconds; the complete initial run took 188.1
seconds, with default worker counts and a warm/partly warm local cache.
Independent validation took another 28.7 seconds: all source rays were
rechecked with the final filter, every retained record was compared exactly,
every retained float32 crossing was tested against the mask, 9,461 kept and
rejected rays were checked by a separate plane-intersection oracle, and the
real CPU fitting loader sampled 20,000 valid pairs across both loss types.
The delivered copy includes `verification.json` and `mask_report.json`.
These are single-run timings, not repeated benchmark statistics.
