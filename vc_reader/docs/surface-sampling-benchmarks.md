# Surface sampling validation and measurements

Validated on Linux amd64, Python 3.14 and an RTX 5090 with the rebuilt native
CPU/CUDA raster kernels. The projection regression suite covers both backends;
related aligned-sampling and training tests also exercise DataLoader workers.
No dependencies were installed.

## Correctness and registration

Synthetic tests cover physical offsets, anisotropic and cross-volume frames,
explicit normals/masks, holes, degenerate geometry, rotated label footprints,
exact voxel-face contacts, thickness versus padding, folded conflicts, dense
versus tiled equality, output ownership, worker reopening, bounded caches,
multichannel categorical values, byte order, and large coordinate/UV origins.
Production geometry uses float32 local coordinates. Integer labels preserve
their stored dtype, including uint8 ink. The uint64 case tests exact preservation
of unusually large categorical IDs; it does not promote ordinary labels.
Stable triangle IDs and global origins use integer indexing. The independent
scalar oracle uses float64 to expose float32 boundary/rounding mistakes.

The PHercParis4 layer workload uses the bundled `fiber_132487` tifxyz and volume
`20260411134726`, level 2 (9.6 µm voxels). Six 32×32 UV patches begin at
`(30, x)` for x = 1000, 2000, 3000, 4000, 5000, 6000, with offsets
`(-9.6, 0, 9.6)` µm. TIFF/SciPy interpolation independently checks geometry
within 0.0001 voxel; direct volume samples check CT with `atol=.02, rtol=2e-5`.
CPU/CUDA CT comparisons use `atol=.001, rtol=2e-5`; validity agrees exactly.
Tifxyz scale is XY, as used by QuadSurface: columns use scale[0], rows scale[1].
Correcting the benchmark's axis interpretation changes its selected requests,
so earlier surface-workload hashes/timings are not a before/after baseline.

Projection uses real PHerc0009B volume `20250820154339`, segment
`20250919125754`, and label set `20260918`. Serializable source fixtures live
in `benchmarks/fixtures/surface_projection/`. Six 16×16 label patches start at
`(18576 + 2*i, 7984 + 2*i)` for i = 0…5. Thickness and padding are each 2.4 µm.
The first crop has origin ZYX `(23696, 15856, 18653)`, shape `(20, 11, 19)`,
4,180 valid CT voxels and 941 valid labeled sheet voxels. The seed rectangle
defines the crop bounds; projection includes the registered sheet throughout
that crop, including the padded margins.

The fixture explicitly declares `pixel_to_uv=((1,0,.5),(0,1,.5))`, derived
from the full-canvas, unit-scale, unrotated `vc_render_tifxyz` convention:
its canvas origin is `-(size-1)/2`, while QuadSurface centers the grid using
`grid_size/(2*scale)`. Published label metadata does not itself provide this
affine. This is a declared fixture registration based on that rendering
convention, not an independent anatomical alignment measurement or automatic
inference from image dimensions. Production discovery requires explicit
registration. The source snapshot SHA256 was
`55aa3d217732a3c75401765f7d425738a35fecab75baad6e4ef5cb8d8b9844e3`.
The fixture declares `tifxyz_mask=False` after the optional mask returned HTTP
404, allowing unambiguous offline reopening.

The first zero-thickness projection agrees exactly with an independent scalar
triangle-intersection/closest-point oracle for categorical values,
label validity, geometric coverage and conflicts on both backends. This oracle
streams the complete raw TIFF geometry and samples label tiles independently
of the production spatial index. All six CT crops match direct reads.
CPU/CUDA label values, validity, conflicts and companion masks agree exactly
for the positive-thickness benchmark. The existing real paired-reading script
also passes on CUDA: 25×128×128 CT, 128×128 labels with 9,926 nonzero pixels,
and independent supervision/validation mask checks.

## End-to-end timings

Each workload uses six fixed requests, two warmup passes, and ten measured
passes (60 requests). Correctness, cProfile and tracemalloc runs are separate
from these wall timings. CUDA calls synchronize before returning. Disk caches
were populated and OS caches were not cleared; these are not cold-network or
cold-disk measurements. These operations are new, so no prior equivalent
end-to-end API baseline exists.

| Workload | CPU mean / p50 / p95 ms | CUDA mean / p50 / p95 ms |
| --- | --- | --- |
| Three surface layers, 32×32 | 0.938 / 0.930 / 0.983 | 0.906 / 0.851 / 1.007 |
| Label projection, 16×16 | 6.403 / 6.318 / 6.671 | 3.954 / 3.834 / 4.593 |

Projection is about 1.62× faster on CUDA for these requests, including CT,
labels and companion masks. Reader opening took 13.8 ms CPU / 22.7 ms CUDA.
The first projection took 128.5 / 213.6 ms, including construction of the
whole-sheet spatial index; CUDA also includes initial device setup. Subsequent
first visits took 10.5–11.3 ms CPU and 8.0–8.8 ms CUDA. Warm timings above reuse
the index, coordinate tiles and label tiles.

The six PHercParis4 notebook crops were also checked on both backends. Each uses
a 64×64 seed, CT level 2, a 9.6 µm band and 19.2 µm padding. Sheet coverage reaches
all six crop faces for these particular patches, and CPU/CUDA labels, validity,
geometric coverage, supervision and conflicts agree exactly. One padded crop
also matches the independent float64 oracle at zero thickness. Building the
index and reading the first crop took 466–556 ms across the two segments/backends.
Three repeats of each segment's first crop had median times of 22.2 / 12.6 ms
on CPU and 16.2 / 7.9 ms on CUDA. These small repeat counts are a cache-reuse check,
separate from the 60-request benchmark table.

## Memory and execution phases

For the 16×16 projection workload, both backends retained 20,443,541 bytes of
sampling tiles and 714,752 bytes of plans; reported peak sampling work was
5,980,240 bytes. Tracemalloc peaks during the instrumented warm pass were
271,976 bytes CPU and 209,600 bytes CUDA. These are distinct budgets/counters,
not whole-process or total GPU memory measurements. TIFF mappings and decoder
caches are separate.

Geometry logical reads totaled 15,772,908 bytes during preparation and remained
unchanged throughout repeated requests: the whole-sheet index is reused.
Across correctness, timing and instrumented passes, preparation consumed
119.87 ms CPU / 117.09 ms CUDA; candidate indexing 1.42 / 1.40 ms; raster calls
318.52 / 29.99 ms. Those cumulative wall times include dispatch and uploads.
CUDA candidate payload uploads totaled 6,194,432 bytes and shared source/state
uploads 9,839,320 bytes. Logical geometry bytes exclude TIFF headers and physical
file/network reads; payload counters exclude some native dispatch transfers.

The regression suite also checks that larger padding preserves annotations in
the overlapping crop, actual geometry holes remain uncovered, missing labels do
not become valid negatives, distant folds are included, and the index is reused
for different seed rectangles. Coverage is produced by the same CPU/CUDA raster
pass as label validity, without a separate geometry rasterization.

## Reproduction

Using the existing configured build directories:

```bash
cmake --build build/cp314-cp314-linux_x86_64 --target _native -j4
cmake --build /tmp/vc-reader-cuda --target _native -j4
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks VC_READER_NATIVE_DIR=build/cp314-cp314-linux_x86_64 .venv/bin/python -m pytest tests -q
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda .venv/bin/python -m pytest tests -q

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_surface.py --native-dir /tmp/vc-reader-cuda --backend both --output /tmp/plan04-layers-final.json
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_surface.py --native-dir /tmp/vc-reader-cuda --backend both --label-source benchmarks/fixtures/surface_projection/label.json --reference-source benchmarks/fixtures/surface_projection/reference.json --start 18576 7984 --size 16 16 --cache-dir /tmp/vc-reader-surface-cache --offline --output /tmp/plan04-projection-final.json

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o /tmp/plan04-projection-profile .venv/bin/python benchmarks/bench_surface.py --native-dir /tmp/vc-reader-cuda --backend cuda --label-source benchmarks/fixtures/surface_projection/label.json --reference-source benchmarks/fixtures/surface_projection/reference.json --start 18576 7984 --size 16 16 --cache-dir /tmp/vc-reader-surface-cache --offline --output /tmp/plan04-projection-profile.json
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv /tmp/plan04-projection-profile.nsys-rep

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python scripts/paired_reading.py --backend cuda --native-dir /tmp/vc-reader-cuda --offline
```

Remove `--offline` for the first projection/paired run to populate its cache.
The default layer workload requires the local PHercParis4 volume path recorded
in `bench_surface.py`. Reports include source descriptors, requests, raw timing
samples, profiles, counters and output hashes. Generated local copies are saved
under `docs/benchmarks/surface-reading/` (ignored by Git).
