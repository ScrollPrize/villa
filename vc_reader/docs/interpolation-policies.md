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

Ordinary scalar interpolation uses an internal identity policy and retains its existing optimized arithmetic and boundary behavior. Policies are host C++ specializations; this entry point does not compile arbitrary policies into CUDA kernels or expose per-point Python callbacks.

The Lasagna adapter supplies a shared policy for validity-masked projective interpolation of encoded sheet normals. Its packed-array batch path returns final normals and validity directly. The domain math remains in `volume-cartographer`; `vc_reader` owns the sampling traversal. Scalar double-precision sampling, CUDA paths and channels stored in separate arrays retain their existing implementations.
