# CPU elastic experiment

Compare alternative elastic displacement fields on real Paris 4 CT crops.
The goal is smooth bending and changes in sheet spacing, with enough control
to distinguish mild waviness from larger deformations. The original comparison used an optional benchmark module. Production
`Elastic` now uses its cubic B-spline method, with shared field construction
and normalization in `cpp/augment/elastic.hpp`. See the
[production optimization report](../elastic_production_results.md).

The comparison images and measured results are in
[../../elastic_comparison/README.md](../../elastic_comparison/README.md).
For movement that is easier to inspect, open the
[animated viewer](../../elastic_comparison/animations/index.html).

## Build and run

The standalone CMake project uses C++20, nanobind, Python development headers,
and OpenCV **core**. The ordinary vc_reader build does not acquire an OpenCV
dependency. No CUDA, OpenMP, architecture-specific intrinsics, fast-math, or
dependency installation is involved. The kernels use portable C++; this run
was validated on Linux amd64, not macOS or arm64.

Exact build and validation commands used, from the repository root:

```sh
AGENTS_AGENT_MODE=1 cmake -S vc_reader/benchmarks/elastic -B /tmp/vc-elastic-lab \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python"
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-elastic-lab -j 4

AGENTS_AGENT_MODE=1 PYTHONPATH=/tmp/vc-elastic-lab \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  spiral-fitting/.venv/bin/python -m pytest \
  vc_reader/benchmarks/elastic/test_lab.py -q --tb=short
```

The existing local environments supplied all dependencies. These paths are
specific to this checkout. The benchmark uses `-S` to bypass its stale editable
package hook, and imports Matplotlib from the other existing environment:

```sh
AGENTS_AGENT_MODE=1 \
  PYTHONPATH=/tmp/vc-elastic-lab:vc_reader/src:vc_reader/.venv/lib/python3.14/site-packages:spiral-fitting/.venv/lib/python3.14/site-packages \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vc-elastic-mpl \
  taskset -c 0 vc_reader/.venv/bin/python -S \
  vc_reader/benchmarks/elastic/compare.py --reps 5 --warmup 1
```

Append `--visuals-only` to regenerate figures and diagnostics while preserving
the original timing samples. It checks CT hashes and the extension hash before
replacing the report. `--no-plots` suppresses plotting; `--chunks parallel`
selects one region. `--array` and `--output` override the default paths.
The reader intentionally rejects missing cache chunks. It supports this raw,
uint8, uncompressed Zarr v2 layout only; it never downloads data or writes to
the source volume. `taskset` is a Linux benchmark control, not a kernel
requirement; omit it on macOS.

Generate the enlarged smooth-motion and before/after GIFs from the saved
benchmark cases (also requires Pillow, already available in these environments):

```sh
AGENTS_AGENT_MODE=1 \
  PYTHONPATH=/tmp/vc-elastic-lab:vc_reader/src:vc_reader/.venv/lib/python3.14/site-packages:spiral-fitting/.venv/lib/python3.14/site-packages \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 vc_reader/.venv/bin/python -S \
  vc_reader/benchmarks/elastic/animate.py
```

`animate.py` reuses the comparison reader, field generation, normalization and
native resampler. It verifies source/build hashes, exact endpoint agreement
with all 18 selected benchmark cases, zero-displacement identity, and every
decoded GIF frame against its rendered source. It writes 24-frame smooth loops,
two-frame blink loops, still endpoints, a manifest and a local HTML viewer to
`vc_reader/elastic_comparison/animations/`. Intermediate frames apply a scaled
field to the original CT, never a repeated warp or an image crossfade. The stored GIFs and timings are historical artifacts. Regeneration checks the
original binary hash; use that preserved build or generate a fresh comparison
in a separate output directory before regenerating its animations.

The current public API is benchmarked by `bench_production.py`; the frozen GIF
prototype versus the shared optimized kernel is benchmarked by `bench_kernel.py`.
Exact commands and raw results are in the [production report](../elastic_production_results.md).
Both accept `--reps` and `--warmup`; `bench_kernel.py --method python --scales 32`
runs the original Python Gaussian at the strongest preview scale.

## Methods and parameter meanings

All fields map **output coordinates to source sampling coordinates**, in ZYX
voxel units. All four C++ methods reuse the existing C++ image resampler. They produce
an owning float32 output and leave the input untouched.

| Method | Field construction | Meaning of `scale` |
|---|---|---|
| `dense` | Three full-resolution random fields and the existing separable Gaussian kernel | Gaussian sigma in voxels |
| `coarse` | Gaussian noise on a four-voxel grid, blurred there, expanded linearly | Full-resolution equivalent Gaussian sigma |
| `bspline` | Random controls expanded by separable cardinal cubic B-splines | Half the knot spacing |
| `normal_bspline` | Same cubic controls, with noise projected toward estimated sheet normals | Half the knot spacing |

The dense kernel preserves the former production Gaussian computation, but its amplitude
normalization is deliberately changed for this experiment. The different
methods are semantic alternatives, not bitwise substitutes. A common nominal
scale does not guarantee identical spatial correlations; `results.json`
records correlations at lags 8, 16, and 32. The B-splines tend to be smoother
at equal nominal scale. Coarse linear interpolation has slope discontinuities
at grid boundaries; cubic control interpolation avoids that at the continuous
field-construction stage. All final fields are sampled on a voxel grid.

`rms` means root-mean-square **vector** displacement after removing global
translation. It is not the old per-component maximum `magnitude`. The field
is uniformly reduced if necessary to keep a conservative infinity-norm
Lipschitz bound at or below 0.8. For trilinearly interpolated fields, derivative
components are convex combinations of grid-edge differences. Summing their
componentwise maxima bounds every Jacobian row. A bound below one makes
`identity + displacement` injective within the crop. This is a sufficient,
conservative condition; it can reject strong bends that would actually be
safe. Titles show **achieved** RMS and identify limited settings. Identical
panels at different requested strengths are an intentional consequence of
this limit. Stronger compression would require a less conservative bound or
a different deformation parameterization, with fresh validation.

The sampled determinant diagnostic uses stride-two finite differences. It is
an additional check, not the proof of injectivity. A determinant greater than
one here describes a larger **source** sampling volume and thus compression
in the output. It is not a measurement of sheet-normal spacing or material
density. No conservation-of-mass intensity correction is applied.

## Tensor provenance and limitations

`ct_normal.hpp` imports the numerical kernels from local
`origin/graph-unification`, commit `de1701f1a`,
`volume-cartographer/core/src/CtSheetNormal.cpp`. Method identifier:
`ct-anisotropic-1-24-4-grid3-r48-v4`. Copying was explicitly authorized.
The Volume/IO/LRU services are replaced with an in-memory crop adapter;
the global three-voxel sampling lattice and gradient/integration kernels
are retained. This experiment uses one worker.

The estimator uses a broad isotropic seed followed by three anisotropic
refinements. At control sites, the displacement is
`0.1*v + 0.9*n*(n dot v)`, which is invariant to the arbitrary normal sign.
Low confidence (eigenvalue gap below 0.2) or invalid estimates fall back to
isotropic noise. Control halo sites use the closest location inside the
output crop. All queries have real CT context for the estimator's full support.

Normals are prepared once per crop and scale, then reused in warm benchmark
iterations. Their preparation time is reported separately and must be added
for a first use. This harness is not a persistent production normal cache.
Normals must be recomputed after a geometry change or keyed to the unchanged
source geometry. Projection at control sites favors normal motion but does
not guarantee exact alignment of every interpolated displacement with the
local sheet. Synthetic-plane accuracy and overlapping-crop consistency are
tested; mesh-normal accuracy on these real crops has not been measured.

The Python baseline calls the actual Vesuvius `SpatialTransform` on CPU,
with sigma `scale` and legacy maximum magnitude 4. Its original timing is
reported separately from a version that applies the experiment's RMS and
gradient-limit normalization before using the same Python resampler.
Images use the latter. That version deliberately calls the shared C++
normalization helper so normalization policy cannot differ across methods;
it is therefore a comparison with the original Python Gaussian generation
and resampling, not an entirely Python implementation of the new policy.

## Profiling

`profile.cpp` runs the same field kernels and production resampler on the
central raw 128³ chunk. Unlike the image comparison, the profiler has no
64-voxel source halo, so boundary sampling adds work. Use profiles to identify
hotspots, not as the release latency measurements.

```sh
AGENTS_AGENT_MODE=1 cmake -S vc_reader/benchmarks/elastic -B /tmp/vc-elastic-gprof \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-pg \
  -DCMAKE_EXE_LINKER_FLAGS=-pg \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python"
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-elastic-gprof --target profile_elastic -j 4

AGENTS_AGENT_MODE=1 GMON_OUT_PREFIX=/tmp/elastic-gprof-bspline taskset -c 2 \
  /tmp/vc-elastic-gprof/profile_elastic \
  /mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2/70/35/25 \
  bspline 16 100
gprof -p -b /tmp/vc-elastic-gprof/profile_elastic /tmp/elastic-gprof-bspline.2
```

The `.2` suffix is the sandbox process ID; use the actual generated suffix
when running elsewhere. Repeat with `coarse 16 100` and prefix
`/tmp/elastic-gprof-coarse`; dense used `dense 16 20`, CPU 3 and prefix
`/tmp/elastic-gprof-dense`. Profiles use GCC `-O2 -g -pg`; release timings use
`-O3 -DNDEBUG`. Shared-library math is not fully attributed by gprof.
A gprofng sampling attempt reported that its timer had changed to zero;
those unreliable samples were discarded.
