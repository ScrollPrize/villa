#pragma once
#include "core.hpp"

// Separable correlation used by derived-volume transforms.
namespace vcr::augment::filter_detail {
struct Taps { std::vector<double> weights; int radius = 0; };
// Index of a neighbour, or -1 for a zero-padded sample.
VCR_HD inline long long neighbour(long long x, long long n, Boundary edge) {
    if (x >= 0 && x < n) return x;
    if (edge == Boundary::Zero) return -1;
    if (edge == Boundary::Nearest) return x < 0 ? 0 : n - 1;
    if (n == 1) return 0;
    const bool mirror = edge == Boundary::Mirror;
    const long long period = mirror ? 2 * (n - 1) : 2 * n;
    x = (x % period + period) % period;
    return x < n ? x : period - x - (mirror ? 0 : 1);
}
template<class T> void correlate(const T* in, T* out, const Grid& g, std::size_t axis, const Taps& t, Boundary edge) {
    const auto n = (long long)g.shape[axis]; const auto stride = g.stride[axis];
    for (std::size_t i = 0; i < g.size; ++i) {
        const auto x = (long long)g.coord(i, axis); const auto base = i - std::size_t(x) * stride;
        double sum = 0;
        for (int k = -t.radius; k <= t.radius; ++k) {
            const auto j = neighbour(x + k, n, edge);
            if (j >= 0) sum += t.weights[k + t.radius] * double(in[base + std::size_t(j) * stride]);
        }
        out[i] = T(sum);
    }
}
// Applies taps[a] along each axis a (identity taps are skipped).
template<class T> void separable(std::vector<T>& data, const Grid& g, const std::vector<Taps>& taps, Boundary edge) {
    std::vector<T> tmp(data.size());
    for (std::size_t a = 0; a < taps.size(); ++a) {
        if (taps[a].radius == 0 && taps[a].weights[0] == 1) continue;
        correlate(data.data(), tmp.data(), g, a, taps[a], edge); data.swap(tmp);
    }
}
} // namespace vcr::augment::filter_detail
