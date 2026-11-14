# Point interpolation policies

`Array::sample_interpolated_coordinates(coordinates, count, elementType, policy, missing)` is a C++ extension of the existing linear-sampling traversal. It uses the same coordinate providers, chunk planner, caches and split-boundary handling as `sample_linear_coordinates`. It does not expand every input point into a batch-wide corner array.

Points whose footprint fits in one chunk are evaluated while that chunk is resolved. Split points gather only their required elements into the existing bounded per-batch scratch and invoke the same policy afterward. With channel-selective caching, policies address the selected cached planes directly; no contiguous copy of those planes is assembled. The caller must register the relevant cache channels as usual.

The policy is a statically typed C++ object, not a Python callback. It provides:

| Member | Meaning |
| --- | --- |
| `static constexpr bool custom = true` | Select the policy specialization. |
| `real_type` | Coordinate-fraction and element-load arithmetic type. |
| `static constexpr size_t components` | Elements required at each corner. |
| `offset(component, axis)` | Nonnegative integer displacement from the supplied coordinate for that component. Choose the lowest channel as the coordinate base for channel tuples. |
| `prepare_cell(cell)` | Select interpolation axes/corner order using `cell.k` and `cell.axes`; fractions and integer coordinates are already populated. Can retain zero-fraction axes when the domain requires all corners. |
| `evaluate(cell, read)` | Return the final domain result. `read(corner, component)` loads one required element. Masking, decoding and interpolation belong here. No chunk pointer escapes the call. |
| `store(index, result)` | Write one output. Concurrent calls address distinct indices. |
| `outside(index)` | Handle nonfinite or out-of-domain coordinates. |
| `static constexpr bool clamp_boundary` | When true, require the supplied coordinate inside the source, then replicate footprint elements at array edges. When false, use ordinary scalar-sampler fill boundaries. Missing chunks supply the configured fill value. |

Store/evaluation operations must be safe for concurrent calls. Reader failures and callback exceptions propagate through the existing worker pool. The optional cache-only `missing` flags retain the scalar sampler's behavior: unavailable samples are flagged and their output is left unchanged.

Policies that combine another source's prepared data may implement `evaluate(cell, read, pointIndex)` instead. The reader supplies the original point index so the final result can be evaluated directly without copying cells or retaining the second source's corners.

Ordinary scalar interpolation uses an internal identity policy and retains its existing optimized arithmetic and boundary behavior. Policies are host C++ specializations; this entry point does not compile arbitrary policies into CUDA kernels or expose per-point Python callbacks.

For nonlinear interpolation of a pair of encoded fields stored in independent arrays, `interpolation.hpp` provides `sample_interpolated_pair_coordinates`. It invokes the same traversal for both sources and returns the policy's final result. One component's corner values are temporarily retained inside the reader in tiles of at most 65536 points (2 MiB for float32 corners); no batch-wide corner result is exposed to the caller. Tiles use the first array's existing shared worker pool, with leased, reusable per-thread scratch that is safe for nested reads and released after inactivity. The bound applies to interpolation workspace, not decoder slabs or cached chunk sizes. The spatial grids must agree, but chunk shapes and element types may differ. The paired policy additionally accepts a source index in `prepare_cell` to select the corresponding array's spatial axes. Policies must support concurrent stores to distinct output indices, as with ordinary parallel sampling. This CPU helper does not implement cache-only mode.

The Lasagna adapter supplies shared policies for validity-masked projective interpolation of encoded sheet normals and unmasked density/normal samples used by winding integration. Both packed and independent-array CPU winding sources return interpolated field values, which the integrator projects along the connector and accumulates using its unchanged trapezoidal rule. The domain math remains in `volume-cartographer`; `vc_reader` owns the sampling traversal. Scalar double-precision sampling and CUDA paths retain their existing implementations.
