#pragma once
#include "../core.hpp"

// Separable correlation used by derived-volume transforms.
namespace vcr::transforms::filter_detail {
struct Taps { std::vector<double> weights; int radius = 0; };
template<class T> void correlate(const T* in, T* out, const Grid& g, std::size_t axis, const Taps& t, Boundary edge) {
    convolve_axis(in, out, g, axis, t.weights, edge);
}
// Applies taps[a] along each axis a (identity taps are skipped).
template<class T> void separable(std::vector<T>& data, const Grid& g, const std::vector<Taps>& taps, Boundary edge) {
    std::vector<T> tmp(data.size());
    for (std::size_t a = 0; a < taps.size(); ++a) {
        if (taps[a].radius == 0 && taps[a].weights[0] == 1) continue;
        correlate(data.data(), tmp.data(), g, a, taps[a], edge); data.swap(tmp);
    }
}
} // namespace vcr::transforms::filter_detail
