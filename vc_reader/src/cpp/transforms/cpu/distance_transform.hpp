#pragma once
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
// apart). v/z are scratch of n and n+1 entries. Infinite lines stay infinite.
template<class T, class V, class Z> VCR_HD void envelope(T* f, std::size_t n, std::size_t stride, V* v, Z* z, T* scratch) {
    const double inf = 1e300;
    long long k = -1;
    for (long long q = 0; q < (long long)n; ++q) {
        const double fq = double(f[q * stride]);
        if (!(fq < inf)) continue;
        while (k >= 0) {
            const double fv = double(f[(long long)v[k] * stride]);
            const double s = ((fq + double(q) * q) - (fv + double(v[k]) * v[k])) / (2. * q - 2. * v[k]);
            if (s <= z[k]) --k; else { ++k; v[k] = q; z[k] = s; z[k + 1] = inf; break; }
        }
        if (k < 0) { k = 0; v[0] = q; z[0] = -inf; z[1] = inf; }
    }
    for (std::size_t q = 0; q < n; ++q) scratch[q] = f[q * stride];
    if (k < 0) return;
    long long j = 0;
    for (long long q = 0; q < (long long)n; ++q) {
        while (z[j + 1] < double(q)) ++j;
        const double dq = double(q) - double(v[j]);
        f[q * stride] = T(dq * dq + double(scratch[v[j]]));
    }
}
// Squared EDT to the nearest voxel where `feature` holds, all axes.
inline void squared_edt(std::vector<double>& f, const Grid& g) {
    const auto n = *std::max_element(g.shape.begin(), g.shape.end());
    std::vector<long long> v(n); std::vector<double> z(n + 1), scratch(n);
    for (std::size_t a = 0; a < g.shape.size(); ++a) {
        const auto stride = g.stride[a], extent = g.shape[a];
        for (std::size_t i = 0; i < g.size; ++i) {
            if (g.coord(i, a) != 0) continue;
            envelope(f.data() + i, extent, stride, v.data(), z.data(), scratch.data());
        }
    }
}
// Distances to the nearest voxel of the other class. A class absent from the
// grid contributes zero (SciPy's result is undefined there).
inline std::vector<float> distance(const std::vector<std::uint8_t>& foreground, const Grid& g, bool inside) {
    std::vector<double> f(g.size); bool any = false;
    for (std::size_t i = 0; i < g.size; ++i) { const bool feature = bool(foreground[i]) != inside; f[i] = feature ? 0 : 1e300; any |= feature; }
    if (!any) return std::vector<float>(g.size, 0.f);
    squared_edt(f, g);
    std::vector<float> result(g.size);
    for (std::size_t i = 0; i < g.size; ++i) result[i] = float(std::sqrt(f[i]));
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
        if (t.mode != DistanceMode::Inside) result = distance(foreground, g, false);
        if (t.mode != DistanceMode::Outside) { auto inside = distance(foreground, g, true); for (std::size_t i = 0; i < g.size; ++i) result[i] -= inside[i]; }
        if (t.mode == DistanceMode::Inside) for (auto& x : result) x = -x;
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
