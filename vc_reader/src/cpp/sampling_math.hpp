// Shared CPU/CUDA sampler arithmetic. The coordinate scalar selects float32 or float64 arithmetic.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#ifdef __CUDACC__
#define VCR_SAMPLE_HD __host__ __device__
#else
#define VCR_SAMPLE_HD
#endif
namespace vcr::sampling {
// Coordinates use separate float32 rounding on every architecture. Interpolation
// may use FMA, but coordinate contraction can change nearest half-voxel ties.
VCR_SAMPLE_HD inline float step(float value, float axis, float index) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(value,__fmul_rn(axis,index));
#else
    volatile float product=axis*index;
    return value+product;
#endif
}
VCR_SAMPLE_HD inline float affine(float origin, const float* axes, float z, float y, float x) {
    return step(step(step(origin,axes[0],z),axes[1],y),axes[2],x);
}
// The weighted sum of a cell's 2^k neighbours; value(corner) is neighbour `corner`
// (bit a set: one step up along axes[a]). The weights and the sum are always formed in
// this order, so a point's result does not depend on where its neighbours were read.
template <class Cell, class Value>
VCR_SAMPLE_HD inline typename Cell::value_type interpolate(const Cell& cell, Value&& value)
{
    using Real = std::remove_cvref_t<decltype(cell.frac[0])>;
    const std::size_t count = std::size_t{1} << cell.k;
    Real v = 0;
    for (std::size_t corner = 0; corner < count; ++corner) {
        Real w = 1;
        for (std::size_t a = 0; a < cell.k; ++a) {
            const Real f = cell.frac[cell.axes[a]];
            if (corner >> a & 1)
                w *= f;
            else
                w *= Real(1) - f;
        }
        v += w * value(corner);
    }
    return v;
}

// interpolate() for a 3-D cell off the grid on every axis, unrolled: the same weights
// and the same sum, in the same order. `at` is the near corner, `step` the byte strides.
template <class Real, class Load>
VCR_SAMPLE_HD inline Real trilinear(const Real* f, const std::uint8_t* at, const std::int64_t* step, Load& load)
{
    const Real g0 = Real(1) - f[0], g1 = Real(1) - f[1], g2 = Real(1) - f[2];
    const Real w00 = g0 * g1, w10 = f[0] * g1, w01 = g0 * f[1], w11 = f[0] * f[1];
    const std::int64_t s0 = step[0], s1 = step[1], s2 = step[2];
    Real v = 0;
    v += w00 * g2 * load(at);
    v += w10 * g2 * load(at + s0);
    v += w01 * g2 * load(at + s1);
    v += w11 * g2 * load(at + s0 + s1);
    v += w00 * f[2] * load(at + s2);
    v += w10 * f[2] * load(at + s0 + s2);
    v += w01 * f[2] * load(at + s1 + s2);
    v += w11 * f[2] * load(at + s0 + s1 + s2);
    return v;
}

} // namespace vcr::sampling
#undef VCR_SAMPLE_HD
