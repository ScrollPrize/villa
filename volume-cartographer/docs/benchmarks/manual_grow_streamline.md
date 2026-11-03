# Reusing streamline endpoint normals

The integrator reuses a normal that it already evaluated at the exact same
position. The first derivative uses the normal just fetched to project the
starting tangent. A successful substep's endpoint normal becomes the next
substep's starting normal. This value belongs to one line and is discarded on
failure before projection fallback, and at the end of the line.

The structure-tensor implementation, finite-difference stencil, integrator
stages, step lengths, floating-point arithmetic order, and projection path are
unchanged. For N successful substeps per line, normal queries fall from 8N to
6N+1 for midpoint/Heun and from 14N to 12N+1 for RK4. Normal fields must remain
stable during the walk, as they do for VC3D's immutable source volume.

## Measurement protocol

Baseline: commit `1a59b2c6d`, including the 512 MiB prepared-patch cache. Both
binaries use the existing **RelWithDebInfo** build (`-O2 -g -DNDEBUG`, LTO), as
required by the local build instructions. These are not Release measurements.
Host: Ubuntu 24.04, aarch64, Cortex-X925/A725.

Input: the existing real PHerc0139 4x4 tifxyz crop at
`build/streamline-bench/input`, whose metadata identifies source surface
`w020_20260219135139420`. CT source:
`https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr`.
Each run grows the right edge, one round, with a maximum internal step of 2
voxels. Two cells add 8 vertices; sixteen cells add 64 vertices.

`scripts/benchmark_streamline.py` uses one unmeasured warmup per binary, then
five runs each, alternating which binary runs first. The on-disk CT chunk cache
is warm; each process starts with empty normal caches. OpenMP and OpenBLAS use
one thread each; `--threads` controls the independent integration workers.
No build or profiler runs concurrently with the measurements. Timings are the
CLI's reported time, including input/volume loading, growth, and output saving,
but excluding process startup.
The harness requires all three output XYZ TIFF files to match byte-for-byte
for every before/after pair, including warmup. Reports retain input hashes and
the full commands.

## Results

Seconds, five measured runs per binary. All XYZ TIFF comparisons passed.

| Method | Cells / workers | Before mean (min / median / max) | After mean (min / median / max) | Time reduction |
| --- | --- | --- | --- | --- |
| midpoint | 2 / 1 | 1.3720 (1.3666 / 1.3719 / 1.3759) | 1.2306 (1.2221 / 1.2311 / 1.2365) | 10.3% |
| heun | 2 / 1 | 1.2826 (1.2739 / 1.2852 / 1.2877) | 1.1405 (1.1357 / 1.1376 / 1.1521) | 11.1% |
| rk4 | 2 / 1 | 1.8239 (1.8129 / 1.8226 / 1.8377) | 1.6857 (1.6815 / 1.6851 / 1.6903) | 7.6% |
| rk4 | 2 / 4 | 0.5099 (0.4950 / 0.5115 / 0.5250) | 0.4804 (0.4618 / 0.4849 / 0.4898) | 5.8% |
| rk4 | 16 / 4 | 3.6900 (3.6593 / 3.6725 / 3.7271) | 3.4016 (3.3719 / 3.4071 / 3.4258) | 7.8% |

## Profiling

A gprofng CPU sample profile of baseline RK4, 16 cells, one worker, attributes
59.1% exclusive CPU time to `SheetNormalField::normalAt` and 13.8% to `exp`
(72.9% inclusive in `normalAt`); an unidentified libc routine accounts for another
6.6%. The optimization reduces repeated calls into this expensive path without
changing its computation.

## Reproduce

From `volume-cartographer`, preserve the baseline binary before applying the
integration change:

```sh
ninja -C build vc_extend_edge test_manual_add_extend
cp build/bin/vc_extend_edge build/bin/vc_extend_edge-pass2-before
# Apply the integration change, then:
ninja -C build VC3D vc_extend_edge test_manual_add_extend
ctest --test-dir build -R '^manual_add_extend$' --output-on-failure
python scripts/benchmark_streamline.py \
  --before build/bin/vc_extend_edge-pass2-before \
  --after build/bin/vc_extend_edge \
  --input build/streamline-bench/input \
  --output build/streamline-bench/comparison-rk4 \
  --config-dir build/streamline-bench/config \
  --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr \
  --integrator rk4 --cells 2 --threads 1 --iterations 5
```

The output directory must be new. The config directory supplies the existing
VC3D disk-cache configuration. Repeat with `--integrator midpoint` or `heun`,
and with `--threads 4` / `--cells 16`, using a different output directory each
time. `--help` lists the harness options; it uses only Python's standard library.

For profiling, prepend `gprofng collect app -o profile.er` to the `vc_extend_edge`
command saved in a benchmark summary, selecting a fresh output path. Inspect
with `gprofng display text -functions profile.er`; do not include profiler runs
in the benchmark timing set.

## Validation and limits

`manual_add_extend` covers all three integration methods, convergence, unsigned
normal sign changes, missing normals, serial/parallel equivalence, per-line
normal reuse, and invalidation after projection fallback. The real-data harness
checks byte-identical output geometry in addition to the unit tests.

This is a small real-surface crop and a warm disk-cache measurement. It does not
establish throughput on large surfaces or cold remote reads. The changed code
uses portable C++ and no new dependencies or architecture-specific instructions;
macOS and amd64 were not tested here.

## Eight versus twelve workers

A separate worker-scaling check uses an unresampled 48x8 crop of real surface
`20251226000000-w055_2025122611`, published in
`PHerc0139/20250728140407@L0` coordinates. The crop begins at row 192, column 256.
Both worker counts use the optimized binary above, RK4, a 2-voxel maximum
substep, and eight added cells on the right edge (384 vertices on 48 lines).
Five alternating measured runs follow one warmup for each worker count, with
the same build, CT source, and environment described above.

| Workers | Mean (s) | Minimum (s) | Median (s) | Maximum (s) |
| --- | --- | --- | --- | --- |
| 8 | 9.3602 | 9.0558 | 9.1535 | 9.7867 |
| 12 | 7.7053 | 7.5137 | 7.6757 | 7.9930 |

Twelve workers reduce mean runtime by 17.7% on this machine and workload. All
XYZ TIFFs match byte-for-byte across both worker counts and all runs; no normal
queries lacked a usable normal. This does not establish an optimal worker count
on other CPUs. The Manual Grow Workers control defaults to 8 and permits users
to select 12 or another limit.

The per-run command, from `volume-cartographer`, is:

```sh
VC3D_CONFIG_DIR="$PWD/build/streamline-bench/config" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
build/bin/vc_extend_edge \
  -i build/streamline-bench/threads-8-vs-12/input \
  -n https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr \
  --sides right --cells 8 --rounds 1 --normal-step 2 \
  --walk-mode streamline --integrator rk4 --threads 8 \
  -o build/streamline-bench/threads-8-vs-12/repeat-8 \
  --report build/streamline-bench/threads-8-vs-12/repeat-8.json
```

Repeat with `--threads 12` and a fresh output/report path. The local measurement
artifacts, including crop provenance, individual reports, and the alternating
run script, are under `build/streamline-bench/threads-8-vs-12/`.
