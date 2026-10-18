# Surface sampling validation and measurements

Validated with Release CPU/CUDA builds on Linux amd64, Python 3.14 and an
RTX 5090. The complete suite passed 140 tests and 2,908 subtests with the CUDA
build; the CPU build passed 119 tests and 1,374 subtests, with 21 CUDA skips.
After naming the layer operation `render_surface`, its 13 tests and four
subtests passed again on the CUDA build, exercising both backends. macOS and
arm64 runtime validation was unavailable. No dependencies were installed.

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
4,180 valid CT voxels and 701 covered label voxels.

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
triangle-clipping/intersection/closest-point oracle for categorical values,
coverage and conflicts on both backends. All six CT crops match direct reads.
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
| Label projection, 16×16 | 3.206 / 3.254 / 3.504 | 1.718 / 1.654 / 2.174 |

Projection is about 1.87× faster on CUDA for these small requests, including
CT, labels and companion masks. Reader opening took 13.6 ms CPU / 23.2 ms CUDA
for projection and 5.1 / 13.8 ms for layers. The first projection took
6.5 / 102.6 ms and first layer request 5.6 / 92.9 ms; CUDA includes initial
device setup. Subsequent first visits to projection patches took 3.6–4.6 ms
CPU and 2.3–3.5 ms CUDA.

## Memory and execution phases

Projection retained 16,024 bytes of sampling tiles and 15,360 bytes of plans;
reported peak sampling work was 138,696 bytes. Tracemalloc peaks were 258,540
bytes CPU and 199,132 bytes CUDA. Layers retained 401,192 tile bytes, no plans,
and peaked at 196,608 sampling work bytes; Python peaks were 183,012 / 168,372
bytes. These figures are distinct budgets/counters, not whole-process or total
GPU memory measurements. TIFF mappings and decoder caches are separate.

Across the projection run, including correctness and instrumented passes,
geometry preparation consumed 4.99 ms CPU / 4.65 ms CUDA; index construction
1.18 / 1.19 ms; raster calls 226.21 / 29.77 ms. Those cumulative wall times
include dispatch and uploads and are not unbiased per-request kernel timings.
Geometry logical reads totaled 288 bytes; CUDA candidate payload uploads
totaled 5,042,952 bytes and shared source/state uploads 9,443,744 bytes.
Logical geometry bytes exclude TIFF headers and physical file/network reads;
payload counters exclude some native dispatch-table transfers.

A separate Nsight Systems projection profile, including startup and validation,
measured 182 raster launches at mean 137.1 µs (median 117.2 µs), 364 aligned
sampling launches at mean 5.77 µs, and 182 box-validity launches at mean
0.707 µs. Rasterization dominated GPU kernel time (24.96 ms total).
H2D copies totaled 24.063 MB / 1.758 ms, D2D 4.717 MB / 9.673 ms, and D2H
3.256 MB / 0.129 ms across that entire profile. Small copies and Python/dispatch
overhead remain significant at this patch size; this is not a large-patch
throughput claim. Host decoding, UV planning and spatial indexing remain CPU
work; interpolation, volume/categorical sampling and raster reduction have
CUDA implementations.

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
