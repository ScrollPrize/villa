#pragma once

#include <cmath>

// Catmull-Rom (Cardinal spline, tension 1/2) basis, shared by the volume
// sampler (ChunkSampler::sampleTricubic in core/src/Slicing.cpp) and the
// bicubic surface warp (QuadSurface::gen's Smooth interpolation mode).
//
// Both callers evaluate the basis as a function of the signed distance from a
// tap to the sample point, i.e. weight(f - d) for taps at integer offsets d.
// Keeping one definition here means the two can never drift apart.

// Slicing.cpp defines its own VC_FORCE_INLINE *after* its includes, so this
// header cannot use it (and must not collide with it).
#if defined(_MSC_VER)
#define VC_INTERP_INLINE __forceinline
#else
#define VC_INTERP_INLINE __attribute__((always_inline)) inline
#endif

namespace vc::interp
{

// The basis function. Even in |t| and zero outside (-2, 2).
VC_INTERP_INLINE float catmullRomWeight(float t)
{
    float at = std::abs(t);
    if (at < 1.0f) return 1.5f*at*at*at - 2.5f*at*at + 1.0f;
    if (at < 2.0f) return -0.5f*at*at*at + 2.5f*at*at - 4.0f*at + 2.0f;
    return 0.0f;
}

// d/dt of catmullRomWeight. The basis is even in |t|, so the chain rule leaves
// an explicit sign factor. Both branches agree at |t| == 1 (value 0, slope
// -1/2), so the basis and this derivative are continuous there.
VC_INTERP_INLINE float catmullRomWeightDerivative(float t)
{
    float at = std::abs(t);
    float s  = t < 0.0f ? -1.0f : 1.0f;
    if (at < 1.0f) return s * (4.5f*at*at - 5.0f*at);
    if (at < 2.0f) return s * (-1.5f*at*at + 5.0f*at - 4.0f);
    return 0.0f;
}

// Weights for the four taps at integer offsets {-1, 0, 1, 2} from floor(x),
// where f = x - floor(x) is in [0, 1). Sums to 1 for every f.
VC_INTERP_INLINE void catmullRomWeights4(float f, float w[4])
{
    w[0] = catmullRomWeight(f + 1.0f);
    w[1] = catmullRomWeight(f);
    w[2] = catmullRomWeight(f - 1.0f);
    w[3] = catmullRomWeight(f - 2.0f);
}

// The same four weights plus d/df of each, for analytic differentiation of the
// interpolant. dw sums to 0 for every f. At f == 0 these reduce exactly to
// w = (0, 1, 0, 0) and dw = (-1/2, 0, 1/2, 0), i.e. the interpolant passes
// through the sample and its derivative is the central difference — which is
// what grid_normal_int computes (see core/src/Geometry.cpp).
VC_INTERP_INLINE void catmullRomWeights4WithDerivative(float f, float w[4], float dw[4])
{
    w[0]  = catmullRomWeight(f + 1.0f);
    w[1]  = catmullRomWeight(f);
    w[2]  = catmullRomWeight(f - 1.0f);
    w[3]  = catmullRomWeight(f - 2.0f);
    dw[0] = catmullRomWeightDerivative(f + 1.0f);
    dw[1] = catmullRomWeightDerivative(f);
    dw[2] = catmullRomWeightDerivative(f - 1.0f);
    dw[3] = catmullRomWeightDerivative(f - 2.0f);
}

} // namespace vc::interp
