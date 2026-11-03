#pragma once
#if defined(__SSE2__) && !defined(__CUDACC__)
#include <emmintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(__CUDACC__)
#include <arm_neon.h>
#endif
#include "../ops/distance_transform.hpp"
#include "labels.hpp"
#include "../shared/separable_filter.hpp"
#include "../shared/derived_field.hpp"

namespace vcr::transforms {

namespace distance_detail {
using namespace filter_detail;
using namespace derived_field_detail;
// Scharr (skimage, ndi.convolve): x[i+1] - x[i-1]; [3, 10, 3]/16 smoothing.
inline Taps scharr_derivative() { return {{-1., 0., 1.}, 1}; }
inline Taps scharr_smooth() { return {{3 / 16., 10 / 16., 3 / 16.}, 1}; }
// Felzenszwalb-Huttenlocher squared distance of one line in place (stride
// apart). v/z are scratch of n and n+1 entries, `ss` apart (CUDA interleaves
// the scratch of neighbouring lines). Infinite lines stay infinite. With `index`
// (and n entries of index_scratch), each voxel also takes the nearest feature
// index of the line voxel its distance comes from.
template<class T, class V, class Z, class I = long long> VCR_HD void envelope(T* f, std::size_t n, std::size_t stride, V* v, Z* z, T* scratch,
                                                                         I* index = nullptr, I* index_scratch = nullptr, std::size_t ss = 1) {
    const double inf = 1e300;
    long long k = -1;
    for (long long q = 0; q < (long long)n; ++q) {
        const double fq = double(f[q * stride]);
        if (!(fq < inf)) continue;
        while (k >= 0) {
            const auto vk = (long long)v[k * ss];
            const double fv = double(f[vk * stride]);
            const double s = ((fq + double(q) * q) - (fv + double(vk) * vk)) / (2. * q - 2. * vk);
            if (s <= z[k * ss]) --k; else { ++k; v[k * ss] = q; z[k * ss] = s; z[(k + 1) * ss] = inf; break; }
        }
        if (k < 0) { k = 0; v[0] = q; z[0] = -inf; z[ss] = inf; }
    }
    for (std::size_t q = 0; q < n; ++q) scratch[q * ss] = f[q * stride];
    if (k < 0) return;
    if (index) for (std::size_t q = 0; q < n; ++q) index_scratch[q * ss] = index[q * stride];
    long long j = 0;
    for (long long q = 0; q < (long long)n; ++q) {
        while (z[(j + 1) * ss] < double(q)) ++j;
        const auto vj = (long long)v[j * ss];
        const double dq = double(q) - double(vj);
        f[q * stride] = T(dq * dq + double(scratch[vj * ss]));
        if (index) index[q * stride] = index_scratch[vj * ss];
    }
}
// envelope() for one line of integer-valued squared distances (features 0, sums of squared
// integer offsets, or the 1e300 sentinel) without its division. Each intersection is a ratio of
// integers with a positive denominator below 2n, so exact cross-multiplied comparisons decide
// exactly as envelope()'s correctly rounded quotients do (distinct ratios differ by at least
// 1/(4n^2); equal ratios round alike), and the resulting q^2 + f values are identical.
inline void envelope_integer(double* f, std::size_t n, std::size_t stride, long long* v, long long* zn, long long* zd, long long* g) {
    const double inf = 1e300;
    long long k = -1;
    for (long long q = 0; q < (long long)n; ++q) {
        const double fq = f[q * stride];
        if (!(fq < inf)) { g[q] = -1; continue; }
        g[q] = (long long)fq;
        const long long hq = g[q] + q * q;
        while (k >= 0) {
            const long long vk = v[k], num = hq - (g[vk] + vk * vk), den = 2 * (q - vk);
            // s <= z[k], with z[0] = -inf.
            const bool pop = k > 0 && num * zd[k] <= zn[k] * den;
            if (pop) --k; else { ++k; v[k] = q; zn[k] = num; zd[k] = den; break; }
        }
        if (k < 0) { k = 0; v[0] = q; }
    }
    if (k < 0) return;
    // z[k + 1] = +inf; advance while z[j + 1] < q.
    long long j = 0;
    for (long long q = 0; q < (long long)n; ++q) {
        while (j < k && zn[j + 1] < q * zd[j + 1]) ++j;
        const long long vj = v[j], dq = q - vj;
        f[q * stride] = double(dq * dq + g[vj]);
    }
}
// float(std::sqrt(x)) of nonnegative values. Square roots are correctly rounded (and the
// float conversion rounds to nearest) everywhere; std::sqrt's errno handling, which only
// negative inputs need, otherwise prevents vectorization.
inline void sqrt_to_float(const double* in, float* out, std::size_t n) {
    std::size_t i = 0;
#if defined(__SSE2__) && !defined(__CUDACC__)
    for (; i + 2 <= n; i += 2) {
        const __m128 r = _mm_cvtpd_ps(_mm_sqrt_pd(_mm_loadu_pd(in + i)));
        _mm_storel_pi(reinterpret_cast<__m64*>(out + i), r);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(__CUDACC__)
    for (; i + 2 <= n; i += 2) vst1_f32(out + i, vcvt_f32_f64(vsqrtq_f64(vld1q_f64(in + i))));
#endif
    for (; i < n; ++i) out[i] = float(std::sqrt(in[i]));
}
// Squared EDT to the nearest voxel where `feature` holds, all axes; with
// `index`, also each voxel's nearest feature as a flat index.
// The first axis of a 0/inf (feature/background) volume: envelope() then yields, on lines with
// a feature, (q - nearest feature)^2 + 0, an exact integer square, and leaves featureless lines
// unchanged. Forward/backward integer scans give the same values, vectorized across the lines.
VCR_TARGET_CLONES inline void first_axis_binary(std::vector<double>& f, const Grid& g) {
    const auto extent = g.shape[0], lines = g.size / extent;
    const std::int32_t none = std::int32_t(1) << 30;  // beyond any extent: lines with no feature
    std::vector<std::int32_t> forward(lines, none), backward(lines, none), distance(g.size);
    std::vector<std::uint8_t> any(lines, 0);
    for (std::size_t q = 0; q < extent; ++q) {
        const double* __restrict row = f.data() + q * lines;
        std::int32_t* __restrict out = distance.data() + q * lines;
        std::int32_t* __restrict last = forward.data();
        std::uint8_t* __restrict seen = any.data();
        for (std::size_t i = 0; i < lines; ++i) {
            const bool feature = row[i] == 0;
            last[i] = feature ? 0 : last[i] + 1;
            out[i] = last[i];
            seen[i] |= std::uint8_t(feature);
        }
    }
    for (std::size_t q = extent; q-- > 0;) {
        double* row = f.data() + q * lines;
        const std::int32_t* __restrict out = distance.data() + q * lines;
        std::int32_t* __restrict next = backward.data();
        const std::uint8_t* __restrict seen = any.data();
        for (std::size_t i = 0; i < lines; ++i) {
            const double value = row[i];
            next[i] = value == 0 ? 0 : next[i] + 1;
            const double dq = double(out[i] < next[i] ? out[i] : next[i]);
            row[i] = seen[i] ? dq * dq : value;
        }
    }
}
inline void squared_edt(std::vector<double>& f, const Grid& g, std::vector<long long>* index = nullptr, bool binary = false) {
    const auto n = *std::max_element(g.shape.begin(), g.shape.end());
    std::vector<long long> v(n), index_scratch(index ? n : 0); std::vector<double> z(n + 1), scratch(n);
    const bool first_binary = binary && !index;
    // Binary inputs keep every finite value an integer; under 2^16 per axis, values stay below
    // 3 * 2^32 and envelope_integer's cross products below 2^51.
    const bool integer = binary && n < (std::size_t(1) << 16);
    std::vector<long long> zn(integer ? n + 1 : 0), zd(integer ? n + 1 : 0), values(integer ? n : 0);
    if (first_binary) first_axis_binary(f, g);
    for (std::size_t a = first_binary ? 1 : 0; a < g.shape.size(); ++a) {
        const auto stride = g.stride[a], extent = g.shape[a], block = stride * extent;
        // Line starts (coordinate a is 0) in ascending order, without a div/mod per voxel.
        for (std::size_t outer = 0; outer < g.size; outer += block)
            for (std::size_t i = outer; i < outer + stride; ++i) {
                if (integer && !index) { envelope_integer(f.data() + i, extent, stride, v.data(), zn.data(), zd.data(), values.data()); continue; }
                envelope(f.data() + i, extent, stride, v.data(), z.data(), scratch.data(),
                         index ? index->data() + i : nullptr, index_scratch.data());
            }
    }
}
// Distances to the nearest voxel of the other class. A class absent from the
// grid contributes zero (SciPy's result is undefined there). `nearest` takes
// each voxel's nearest feature as a flat index; -1 stands for (-1, 0, ...),
// the implicit background, and an absent class otherwise gives the voxel itself.
inline std::vector<float> distance(const std::vector<std::uint8_t>& foreground, const Grid& g, bool inside, bool implicit_background = false,
                                   std::vector<long long>* nearest = nullptr) {
    std::vector<double> f(g.size); bool any = false;
    if (nearest) nearest->assign(g.size, 0);
    for (std::size_t i = 0; i < g.size; ++i) {
        const bool feature = bool(foreground[i]) != inside; f[i] = feature ? 0 : 1e300; any |= feature;
        if (nearest) (*nearest)[i] = (long long)i;
    }
    if (!any) {
        if (nearest && implicit_background) std::fill(nearest->begin(), nearest->end(), -1);
        std::vector<float> result(g.size, 0.f);
        if (implicit_background) for (std::size_t i=0;i<g.size;++i) {
            double sum=0;for(std::size_t a=0;a<g.shape.size();++a){const double x=double(g.coord(i,a))+(a==0?1:0);sum+=x*x;}
            result[i]=float(std::sqrt(sum));
        }
        return result;
    }
    squared_edt(f, g, nearest, true);
    std::vector<float> result(g.size);
    sqrt_to_float(f.data(), result.data(), g.size);
    return result;
}
} // namespace distance_detail

inline Parameters sample(const DistanceTransform&, const Sample&, Random&) { return {}; }
inline void apply(const DistanceTransform& t, Sample& s, const Parameters&) {
    using namespace distance_detail;
    std::vector<std::pair<std::string, Field>> outputs;
    for (const auto& [name, f] : s) if (selected_label(name, f, t.on)) {
        const auto& a = f.array; Grid g(a.spatial()); const auto d = g.shape.size();
        require(t.channel < a.shape[0], "DistanceTransform channel out of bounds");
        const auto values = a.channel(t.channel);
        std::vector<std::uint8_t> foreground(g.size);
        for (std::size_t i = 0; i < g.size; ++i) foreground[i] = values[i] > t.threshold;
        std::vector<float> result(g.size, 0.f);
        std::vector<long long> outside_nearest, inside_nearest;
        const bool indices = !t.indices.empty();
        if (t.mode != DistanceMode::Inside) result = distance(foreground, g, false, t.implicit_background, indices ? &outside_nearest : nullptr);
        if (t.mode != DistanceMode::Outside) {
            auto inside = distance(foreground, g, true, t.implicit_background, indices ? &inside_nearest : nullptr);
            for (std::size_t i = 0; i < g.size; ++i) result[i] -= inside[i];
        }
        if (t.mode == DistanceMode::Inside) for (auto& x : result) x = -x;
        if (indices) {
            // The nearest voxel of the other class: foreground voxels (signed, inside)
            // from the inside pass, background voxels from the outside pass.
            Shape ishape{d}; ishape.insert(ishape.end(), g.shape.begin(), g.shape.end());
            Field nearest{Array(ishape, 'i', 4), derived(Role::Label), name};
            auto* out = static_cast<std::int32_t*>(nearest.array.data());
            for (std::size_t i = 0; i < g.size; ++i) {
                const bool from_inside = t.mode == DistanceMode::Inside || (t.mode == DistanceMode::Signed && foreground[i]);
                const auto j = (from_inside ? inside_nearest : outside_nearest)[i];
                for (std::size_t a = 0; a < d; ++a) out[a * g.size + i] = j < 0 ? (a == 0 ? -1 : 0) : std::int32_t(g.coord(std::size_t(j), a));
            }
            outputs.emplace_back(name + t.indices, std::move(nearest));
        }
        Shape shape{1}; shape.insert(shape.end(), g.shape.begin(), g.shape.end());
        Field out{Array(shape), derived(Role::Scalar), name}; out.array.channel(0, result);
        if (!t.normals.empty()) {
            Shape vshape{d}; vshape.insert(vshape.end(), g.shape.begin(), g.shape.end());
            Field normals{Array(vshape), derived(Role::NormalField, t.components), name};
            std::vector<std::vector<float>> gradient(d, result);
            for (std::size_t a = 0; a < d; ++a) {
                std::vector<Taps> taps(d, scharr_smooth()); taps[a] = scharr_derivative();
                separable(gradient[a], g, taps, Boundary::Reflect);
            }
            for (std::size_t i = 0; i < g.size; ++i) {
                double norm = 0; for (std::size_t a = 0; a < d; ++a) norm += gradient[a][i] * gradient[a][i];
                norm = std::max(std::sqrt(norm), 1e-6);
                for (std::size_t a = 0; a < d; ++a) gradient[a][i] /= norm;
            }
            for (std::size_t j = 0; j < d; ++j) normals.array.channel(j, gradient[std::size_t(axis(t.components[j], d))]);
            outputs.emplace_back(name + t.normals, std::move(normals));
        }
        outputs.emplace_back(name + t.suffix, std::move(out));
    }
    for (auto& [name, f] : outputs) { require(!s.count(name), "derived field already exists: " + name); s.emplace(name, std::move(f)); }
}

} // namespace vcr::transforms
