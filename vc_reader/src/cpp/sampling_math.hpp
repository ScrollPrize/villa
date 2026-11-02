// Shared CPU/CUDA sampler arithmetic. The coordinate scalar selects float32 or float64 arithmetic.
#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "rounding.hpp"
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
template<class Real, class Axes>
VCR_SAMPLE_HD inline Real cornerWeight(std::size_t corner, const Real* fractions, const Axes& axes, std::size_t dimensions)
{
    Real weight = 1;
    for (std::size_t a = 0; a < dimensions; ++a) {
        const Real f = fractions[axes[a]];
        weight *= (corner >> a & 1) ? f : Real(1) - f;
    }
    return weight;
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
        const Real w = cornerWeight(corner, &cell.frac[0], cell.axes, cell.k);
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


template<class V, class T>
VCR_SAMPLE_HD inline V bilinear(const V& p00, const V& p01, const V& p10, const V& p11, const T& x, const T& y)
{
    const V top = p00 * (T(1) - x) + p01 * x;
    const V bottom = p10 * (T(1) - x) + p11 * x;
    return top * (T(1) - y) + bottom * y;
}

// Nested linear interpolation keeps the rounding used by coordinate fields.
// The byte-strided overload above uses the array sampler's weighted-sum order.
template<class Values, class Fractions>
VCR_SAMPLE_HD inline auto trilinear(const Values& v, const Fractions& f)
{
    using T = std::remove_cvref_t<decltype(f[0])>;
    const T lower = bilinear(T(v[0]), T(v[1]), T(v[2]), T(v[3]), f[0], f[1]);
    const T upper = bilinear(T(v[4]), T(v[5]), T(v[6]), T(v[7]), f[0], f[1]);
    return lower * (T(1) - f[2]) + upper * f[2];
}

// Clamp coordinates to the nearest complete voxel cell; all eight corners must
// pass `valid(z,y,x)`, even when a corner's fractional weight is zero.
template<class At, class Valid, class T>
VCR_SAMPLE_HD inline bool sampleCube(At&& at, Valid&& valid, const int* shapeZYX, const T* xyz, T& out)
{
    int low[3]; T f[3];
    for(int a=0;a<3;++a) {
        if(shapeZYX[2-a]<2 || !std::isfinite(xyz[a]))return false;
        const T base=std::clamp(std::floor(xyz[a]),T(0),T(shapeZYX[2-a]-2));
        low[a]=int(base);f[a]=std::clamp(xyz[a]-base,T(0),T(1));
    }
    T values[8];
    for(int k=0;k<8;++k) {
        const int x=low[0]+(k&1),y=low[1]+((k>>1)&1),z=low[2]+((k>>2)&1);
        if(!valid(z,y,x))return false;
        values[k]=at(z,y,x);
    }
    out=trilinear(values,f);return true;
}

template<class T>
VCR_SAMPLE_HD inline T cornerWeight(std::size_t k, const T* f)
{
    const int axes[3]{0, 1, 2};
    return cornerWeight(k, f, axes, 3);
}

// Z,Y,X multiplication order, used by float tensor and presence samplers.
template<class T>
VCR_SAMPLE_HD inline T cornerWeightZYX(std::size_t k, const T* f)
{
    const int axes[3]{2, 1, 0};
    const std::size_t reversed = ((k & 1) << 2) | (k & 2) | ((k & 4) >> 2);
    return cornerWeight(reversed, f, axes, 3);
}

template<class Values, class Weights>
VCR_SAMPLE_HD inline auto weightedCorners(const Values& values, const Weights& weights)
{
    using T = std::remove_cvref_t<decltype(weights[0])>;
    T value = 0;
    for (int k = 0; k < 8; ++k) value = std::fma(weights[k], T(values[k]), value);
    return value;
}

enum class Accumulation { Nested, Weighted };

template<class Values, class Mask, class T>
VCR_SAMPLE_HD inline bool maskedTrilinear(const Values& values, const Mask& mask, const T* f, T& out, Accumulation accumulation = Accumulation::Nested)
{
    bool all = true;
    for (int k = 0; k < 8; ++k) all &= bool(mask[k]);
    if (all && accumulation == Accumulation::Nested) { out = trilinear(values, f); return true; }
    T sum = 0, denominator = 0;
    for (int k = 0; k < 8; ++k) {
        if (!mask[k]) continue;
        const T w = cornerWeight(k, f);
        if (w > 0) { sum += w * values[k]; denominator += w; }
    }
    if (!(denominator > 0)) return false;
    out = sum / denominator;
    return true;
}

template<class Fractions>
inline std::array<float, 8> cornerWeights(const Fractions& f)
{
    const float xyz[3]{float(f[0]), float(f[1]), float(f[2])};
    std::array<float, 8> out;
    for (int k = 0; k < 8; ++k) out[k] = cornerWeightZYX(k, xyz);
    return out;
}
inline std::array<float, 8> cornerWeights(const std::array<float, 3>& f)
{
    return cornerWeights<std::array<float, 3>>(f);
}

template<class Values, class Fractions, class Mask>
inline auto maskedTrilinear(const Values& values, const Fractions& f, const Mask& mask)
{
    using T = std::remove_cvref_t<decltype(f[0])>;
    const T xyz[3]{f[0], f[1], f[2]};
    T out;
    return maskedTrilinear(values, mask, xyz, out) ? std::optional<T>(out) : std::nullopt;
}

} // namespace vcr::sampling
#undef VCR_SAMPLE_HD
