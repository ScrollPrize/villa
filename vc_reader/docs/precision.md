# Numerical precision

Float32 is the normal execution/storage precision for float32 ML images.
Compatibility with old floating-point outputs or old replay records is **not**
a reason to use float64. New wider arithmetic must identify a range, conditioning,
accumulation, or explicit-input requirement, with a numerical example or test.

Elastic controls/displacements, separable local intensity coefficients, dropout fields,
CPU image workspaces, FFT spectra, reference images and ordinary geometry now
follow float32 rather than being unconditionally widened. Explicit float64 image
inputs retain their precision. Label indices use integer storage. Exact flips
use integer addressing, without interpolation. Nearest resize uses a shared
integer ratio in `resize_index.hpp`, avoiding floating-point half-voxel ties.

Local brightness, contrast, gamma and smoothing fields store four float32
normalization scalars and one float32 vector per spatial axis. Both CPU and CUDA
evaluate their separable product while consuming the image. Centered brightness
fields use the product of axis means; when the unnormalized range is below .01,
a scan of evaluated weights avoids amplifying product rounding in an almost
constant field. That exceptional scan allocates no dense field. Axis means use
double for accumulation and cancellation; stored coefficients and evaluated
weights remain float32. LocalContrast also retains double weighted accumulation.

## Where wider arithmetic remains

| Location / operation | Numerical reason |
| --- | --- |
| `spatial.hpp`, `cuda/geometry.cu`: coordinate fallback | Affine terms can lose subvoxel offsets before interpolation. The dispatch estimates coordinate magnitude including cancellation and uses float32 when `8 * epsilon_float * bound <= 1e-3` voxel; otherwise coordinates use double. This is a conservative dispatch heuristic, not a guarantee of identical nearest-neighbor choices. At 4096, float32 spacing is 1/2048: a 1/8192 translation disappears. A regression exercises this case. |
| `affine.py`, reader origins, sparse points, transform matrices, `sampling_math.hpp`, `array.hpp` | Global coordinates must retain fractional offsets before rebasing to local float32 coordinates. At 2^24 even adjacent integers are no longer all representable in float32. Affine inversion/composition and inverse deformation also amplify error near singular matrices. CT uint8 sampling already rebases into local float32 geometry. Generic reader paths also support explicit float64 source/output values. |
| `normalization.hpp`, `cuda/statistics*`, `channel_statistics.cuh`, CPU statistics/elastic energies | Long sums and sums of squares need accumulation precision beyond each stored voxel. Subtracting a mean rounded to float32 can erase small deviations on a large baseline. The regression uses a float32 image near 1e6 with deviations of .125. Robust deviations, percentile interpolation and retained-statistics gamma use wide scalars for the same cancellation/range reasons. Ordinary min-max arithmetic uses float32 when its range fits. |
| `histogram.hpp`, histogram/CDF kernels, rank and label-value comparisons | Counts/rank positions may exceed 2^24; float32 can select a different order statistic or merge distinct integer labels. Counts and component indices stay integer where possible. Double-valued carriers support exact integers through 2^53; this does not imply arbitrary uint64 values can be represented exactly. |
| `precision.hpp`, channel execution, pointwise/noise/geometry fallbacks | Finite float32 inputs can overflow intermediate filter sums, FFT coefficients or range subtraction even when the final value fits. Bounds promote exceptional inputs; pointwise/interpolation code retries overflowing intermediates in double. Tests include FLT_MAX constant images and large Rician inputs. Rician normally uses `hypot`, avoiding unnecessary promotion. |
| `distance_transform.hpp`, `cuda/distance_transform.cu` | Squared integer distances and lower-envelope intersection ordering can exceed float32's exact integer range on long axes. The distance calculation retains double; finished distances and normal fields are float32. |
| `structure_tensor.hpp`, `cuda/structure_tensor.cu` | The small symmetric eigensolver repeatedly subtracts nearly equal quantities and resolves small eigengaps. Gaussian point evaluation sums over the whole volume, including very small weights before normalization. These scalars/reductions remain double; gradient, component, eigenvector and confidence volumes follow the image dtype. |
| `separable_filter.hpp` | CPU signed derivative sums can suffer cancellation on high-baseline inputs. Accumulation and host tap construction stay double; image buffers follow their dtype. CUDA convolution uses image-precision arithmetic and taps. CPU/CUDA agreement is tolerance-based, not bitwise. |
| `fourier.hpp` | Bluestein phase angles grow with squared indices; float32 phase rounding and repeated twiddle multiplication accumulate phase error. Phase generation/rotation state stay double. Full-volume spectra, line scratch, chirps and butterfly arithmetic follow the image dtype. |
| `surface_displacement.hpp`, `tracing.hpp`, surface/strip CUDA code, directional mapping | Surface normals subtract coordinates and normalize near-parallel cross products. Squared distances choose support sets; many signed blobs accumulate into displacement. Strip control coordinates can be large and its hash arithmetic exceeds float32's exact integer range. Jacobian inversion and sparse inverse mapping need conditioning/convergence precision. Final dense displacement fields are float32. |
| `volume_grid.hpp`, resize tables, oriented-blur frame construction | Index/coordinate construction must resolve large axes and rounding boundaries. Small rotation frames need stable normalization and cross products. Ordinary oriented-blur samples use float32 coordinates with a wide-coordinate fallback; image accumulators/workspaces follow the image dtype. |
| `random_sampling.hpp`, Laplace/beta/Poisson sampling | Poisson counts can exceed 2^24 and its log rejection test subtracts large terms. Gamma rejection subtracts nearly equal terms; Laplace inverse-CDF tails can become infinite if a uniform draw rounds to an endpoint. These distribution calculations retain double; Gaussian and uniform pixel noise and lattice values follow the image precision. |
| `core.hpp`, scalar configs/bindings, sampling and replay scalar carriers | The shared scalar representation also carries global geometry, label values, explicit float64 parameters and exceptional values outside float32 range. Probability draws retain 53-bit resolution so rare-event probabilities are not quantized to a 24-bit grid. Gaussian pixel-noise transcendental arithmetic follows the image dtype; scalar distribution sampling retains the wider range. |

These exceptions do not promise arbitrary-precision geometry or exact results
for ill-conditioned inputs. They identify where narrowing would remove a useful
numerical capability. Asynchronous batch execution flags exceptional channel magnitudes that would
require a statistics readback and rejects them at completion; the synchronous
channel path can promote them. Ordinary CT batches avoid that readback.

There is no public precision mode to tune for ordinary ML
use, and no fast-math compiler change.

## Replay and numerical changes

New replay records use `VCA2`. Local intensity fields serialize their compact
axis coefficients and normalization scalars. Dense fields are serialized as float32; component
indices are integers. Matrices and shared scalar parameters retain the reasons
listed above. `VCA1` records are rejected: regenerate old plans. A new plan
reproduces its recorded execution exactly on the tested backend; CPU/CUDA and
older-version outputs need not be bitwise equal.

For three 128³ random images (seeds 0, 123, 456), the geometry precision change
produced maximum image error of about 7.5e-6 against the preceding double-geometry
implementation. Thirty-one of 6,291,456 nearest-neighbor label voxels selected a
neighboring source voxel. A 128³ elastic field's replay record fell from
50,331,945 to 25,166,121 bytes. These are numerical checks, not model-quality
measurements. Nearest resize now resolves exact half-voxel ties identically on
CPU/CUDA (for example 128 → 109 selects source 64 at output 54). This can change
whole sampled planes relative to previous floating-point tie rounding; it is
a deliberate index-semantics correction, separate from small interpolation
roundoff. See [performance measurements](../benchmarks/README.md#cuda-geometry-optimizations).

The subsequent [CUDA work-reduction pass](../benchmarks/README.md#cuda-geometry-work-reduction)
factors interior interpolation into axis blends, shares B-spline expansion work
across components, fuses field centering with statistics/scaling, and changes
the reduction grid. The lattice, float32 field storage, RMS convention and
gradient cap remain the same. Arithmetic grouping/reduction order changes are
intentional: maximum image difference against the preceding CUDA implementation
was 7.99e-6 on three random 128³ images, with no changed labels in that sample.
VCA2 records retain their format and size; old records still load, but image
interpolation can round differently across versions. Same-version replay is exact.

## Code boundaries and validation

Structure-tensor definitions and CPU/CUDA implementations live in
`structure_tensor.hpp` / `cuda/structure_tensor.cu`. Distance transforms live in
`distance_transform.hpp` / `cuda/distance_transform.cu`. They share only
`separable_filter.hpp` / `cuda/separable_filter.cuh` and `derived_field.hpp`;
neither transform includes the other's implementation.

Using existing configured Release builds, without installing dependencies:

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --parallel 4
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python tests/test_precision.py -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python tests/test_torch.py -v
# Repeat with a configured VC_READER_ENABLE_CUDA=OFF build for CPU-only coverage.
```

Precision tests cover float32/float64 transforms, CPU/CUDA agreement, replay,
derived fields, resize ties, coordinate cancellation, normalization cancellation, overflow
fallbacks and integer component indices. Loader tests cover the public batch
API, stream lifetime, spawned workers, and backward execution.
