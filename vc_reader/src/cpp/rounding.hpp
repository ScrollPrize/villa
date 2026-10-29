// The two tie rules used throughout vc_reader, on CPU and CUDA. Every rounding
// of a coordinate to a voxel or of a value to an integer goes through one of these,
// so a tie resolves the same way in every sampler, backend and transform.
//
// Spatial ties (choosing the voxel nearest a coordinate; voxel i at coordinate i)
// round up, towards +infinity. Integer shifts of the input then shift the result
// by the same voxels, the answer does not depend on a crop, tile or brick origin,
// and zoomed nearest planes give every voxel the same width (ties to even would
// alternate 1 and 3 pixels at 2x). This is SciPy ndimage's order-0 rule.
//
// Value ties (rounding a number: sizes, counts, quantized intensities) round to
// even, as NumPy (np.rint, np.round) and Torch (torch.round) do.
#pragma once
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define VCR_ROUNDING_HD __host__ __device__ inline
#else
#define VCR_ROUNDING_HD inline
#endif

namespace vcr {

// Whether a coordinate whose fraction above its floor is `fraction` (in [0, 1))
// takes the upper voxel. Callers that already split floor and fraction use this;
// c - floor(c) is exact, so there is no c + 0.5 addition rounding.
template <class Real>
VCR_ROUNDING_HD bool nearest_rounds_up(Real fraction)
{
    return fraction >= Real(0.5);
}

VCR_ROUNDING_HD std::int64_t nearest_index(float c)
{
    const float low = floorf(c);
    return std::int64_t(low) + nearest_rounds_up(c - low);
}

VCR_ROUNDING_HD std::int64_t nearest_index(double c)
{
    const double low = floor(c);
    return std::int64_t(low) + nearest_rounds_up(c - low);
}

// Nearest integer to a value, ties to even, independent of the floating-point
// rounding mode.
VCR_ROUNDING_HD double round_half_even(double v)
{
    const double low = floor(v), fraction = v - low;
    return low + (fraction > .5 || (fraction == .5 && fmod(low, 2.) != 0));
}

}  // namespace vcr
