# Aligned sampling validation and measurements

Validated on Linux amd64 with Python 3.14.4, NumPy 2.5.4, and the existing
Release CPU/CUDA builds. No dependencies were installed. The complete CUDA
suite passed 108 tests; the CPU-only suite passed with 21 CUDA-dependent skips.
The final sampling subset separately passed 15 tests on CUDA and 14 with one
CUDA skip on the CPU-only build. macOS/arm64 runtime validation was unavailable.

Tests use a scalar independent reference and analytic vector/normal/coordinate
examples. Categorical values and validity decisions agree exactly across CPU
and CUDA, including uint64 IDs, color encodings, half ties, sparse array origins
above `2**26`, and multiple native output blocks. Float32 comparisons use
`rtol=2e-6, atol=2e-5` for small-valued fields: the relative allowance is about
16 float32 epsilons, covering weight/accumulation rounding against the double
reference and CUDA contraction. The real external-coordinate fields use an
absolute allowance of `.002` voxels, approximately two float32 ULPs near their
largest coordinates, with the same relative allowance. This is not a change to
nearest-neighbor decisions or geometric validity.

## Real volume workload

Input: existing local PHercParis4 volume `20260411134726`, level 2, 9.6 um voxels:

```text
/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2
```

Six `(64, 64, 64)` crops start at `(9000+j, 3600+j, 3600+j)` for
`j = 0, 2, 4, 6, 8, 10`. The affine target uses controlled quarter-voxel
translation and shear on these real CT values. This is a sampling workload,
not evidence of registration between different scans. The report records the
complete source descriptors, affine, requests, native binary hash, and crop
hashes. An affine call returns both the direct reference and transformed target.

Each method has two warmup passes and ten measured passes (60 crops). Timing
excludes separate correctness, cProfile, and allocation runs. Application
caches start empty for the first-read measurements; OS caches were not cleared.
The local decoded chunk cache is the input, so these are not remote-download or
physical disk-cold measurements.

| Workload | Low-level baseline mean / p50 / p95 ms | Aligned mean / p50 / p95 ms |
| --- | --- | --- |
| CPU direct | 0.0165 / 0.0166 / 0.0173 | 0.0383 / 0.0346 / 0.0434 |
| CPU reference + affine target | 4.079 / 4.050 / 4.585 | 6.820 / 6.850 / 7.076 |
| CUDA direct | 0.0518 / 0.0518 / 0.0532 | 0.0574 / 0.0573 / 0.0591 |
| CUDA reference + affine target | 0.0706 / 0.0705 / 0.0731 | 0.3063 / 0.3049 / 0.3133 |

The CPU affine baseline lacks a validity output; aligned sampling also computes
that mask. CUDA baselines include existing low-level uint8 validity masks.
The general CUDA tile kernel supports channels, encodings, arbitrary integer
types, and embedded planes, but remains slower than the specialized low-level
uint8 kernel. Direct crops retain the batched source-reader path.

An independent upload-only measurement took 0.0178 ms for a uint8 crop and
0.0475 ms for a float32 crop. Warm affine targets reuse resident device tiles;
the returned reference image still comes from the direct CPU crop/upload path.
Thus the table's CUDA timing includes that transfer and wrapper overhead.

CPU direct/affine open took 3.38/0.84 ms, with first reads of 1.20/7.58 ms.
CUDA direct/affine open took 13.44/0.75 ms; first reads were 89.12/23.99 ms,
including initial device/runtime allocation and tile planning. Subsequent
new affine requests took approximately 22–23 ms before their plans were cached.
These initialization costs are reported separately from warm execution.

The six requests loaded one 2,097,152-byte decoded source chunk. There were zero
remote fetches. Warm source reads hit the native memory cache; CUDA affine
sampling retained 696,320 tile bytes and 28,320 native plan bytes, with a maximum
3,170,304-byte conservative tile working estimate. Final tile hits were 21,970
across all timing, validation, and profiling passes. The source cache is shared
between reference and target; its per-source reported byte counts must not be
summed. Retained tiles represent the touched tile subset, rather than a whole
rotated bounding box. Counters do not measure physical filesystem traffic.

Separate Python allocation peaks were 264,952 bytes (CPU direct), 528,734
(CPU affine), 2,912 (CUDA direct), and 9,305 (CUDA affine). Whole-process peak RSS
was 642,236,416 bytes, including runtime/native initialization. Neither measure
is a native decoder or GPU-memory peak; native cache and tile budgets are
reported separately.

Reproduce:

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_sampling.py \
  --array /mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2 \
  --start 9000 3600 3600 --size 64 64 64 \
  --native-dir /tmp/vc-reader-cuda --output /tmp/plan03-real-ct.json
```

## Real multichannel field values

`bench_sampling_fields.py` uses the bundled `fiber_132487` X/Y/Z TIFFs, preserving
their actual PHercParis4 level-2 XYZ values and recording each input hash. It
re-encodes them as a temporary `(3, 1, 25, 4800)` float32 TIFF. The reference is
the fixture's mask with singleton depth; its declared UV lattice is a test grid,
not an inferred CT surface placement. Coordinates use an external value frame
so their XYZ units remain 9.6 um voxels. This does not implement surface sampling.

Six `(1, 16, 512)` requests start at `(0, j, 300+600*j)`, `j=0..5`, with a
controlled in-plane shear and quarter-voxel translation. The same two-warmup,
ten-repetition protocol applies.

| Backend | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| CPU | 0.7749 | 0.7674 | 0.7796 |
| CUDA | 0.3396 | 0.3225 | 0.4198 |

Direct TIFF field reads took 0.4718 ms mean; this is an I/O lower bound, not an
equivalent interpolation/validity baseline. Both backends retained 1,216,512
field tile bytes. Singleton-aware tiles avoid allocating and uploading padded
depth planes: the initial fixed-depth tiling retained 19,464,192 bytes for the
same requests. The final conservative tile working peak was 92,160 bytes on CPU
and 737,280 bytes for batched CUDA execution. CUDA uploaded 1,953,792 bytes over
all passes, including repeated reference crops. Python peaks were 183,619 and
184,721 bytes; whole-process peak RSS was 642,236,416 bytes. TIFF payload counters
report logical mapped/read regions, not physical disk bytes.

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_sampling_fields.py \
  --native-dir /tmp/vc-reader-cuda --output /tmp/plan03-real-fields.json
```

## Profiling and catalog acceptance

cProfile shows CPU affine time in the existing native sampler plus validity
calculation. Generic field sampling spends time in tile sampling, TIFF reference
reads, and Python request preparation. Native planning costs occur on new
requests; cached plans avoid repeating stencil traversal.

Nsight Systems measured the general CUDA affine kernel at 119.3 us mean per
64-cubed request, versus 12.1 us for the specialized low-level uint8 kernel.
The direct validity kernel averaged 1.15 us. The direct CUDA/CPU latency gap is
mostly host work, upload, and synchronization, rather than a slow validity
kernel. All host-to-device transfers in the trace totaled 4.36 ms across 1,325
copies, including initial tiles, pointer tables, reference crops, and the
upload-only measurement. Initial asynchronous allocation dominated CUDA API
time; profiling timings are not substituted for the unprofiled table above.

```bash
AGENTS_AGENT_MODE=1 nsys profile --sample=none --cpuctxsw=none --trace=cuda \
  --force-overwrite=true -o /tmp/plan03-sampling-profile \
  .venv/bin/python benchmarks/bench_sampling.py \
  --array /mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2 \
  --start 9000 3600 3600 --size 64 64 64 --backend cuda \
  --native-dir /tmp/vc-reader-cuda --output /tmp/plan03-profile-timing.json
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  --format csv /tmp/plan03-sampling-profile.nsys-rep
```

The executable catalog example also opened PHercParis4 volume `20260411134726`
level 2 through its published group metadata and read a `(16, 32, 32)` region
using existing cached chunks, with cache writes disabled. Snapshot SHA-256:
`de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d`.

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python scripts/sampling_examples.py \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --sample PHercParis4 --volume 20260411134726 --level 2 \
  --slice 9000:9016 3600:3632 3600:3632 \
  --cache-dir /mnt/raid_nvme/volume_cache --native-dir /tmp/vc-reader-cuda
```

Pass `--target name=source.json` for an explicitly registered user source or
`--catalog-target name=source-id` for an explicitly chosen catalog descriptor.
New surface-layer generation and volume projection remain
[planned geometry work](../plans/04-surface-labels-and-projection.md). Reading
published flattened render/label pairs is
[implemented and verified](scroll-reading-benchmarks.md). Real scan-to-scan label transfer
is not claimed by the controlled affine benchmark; cross-volume registration
and independent component frames are covered by deterministic fixtures.

Raw local reports are retained in `docs/benchmarks/aligned-sampling/`, following
the repository's ignored benchmark-artifact convention. Exact reproduction
commands above regenerate them. CPU-only imports and runtime were tested on
Linux; macOS/arm64 and real remote cross-volume acceptance remain unverified.
