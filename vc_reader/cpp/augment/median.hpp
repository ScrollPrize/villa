#pragma once
#include "core.hpp"

namespace vcr::augment::median_detail {
struct Comparator { std::size_t low, high; bool minimum = false, maximum = false; };

// A 32-input bitonic sorting network, pruned backwards to the 14th output.
// Padding 27 neighbors with +infinity makes that output the exact 3D median.
// Construct the schedule rather than maintaining an opaque hand-written list.
inline constexpr auto network = [] {
    std::array<Comparator, 240> result{};
    std::size_t n = 0;
    for (std::size_t size = 2; size <= 32; size *= 2)
        for (std::size_t step = size / 2; step; step /= 2)
            for (std::size_t i = 0; i < 32; ++i) {
                const auto j = i ^ step;
                if (j > i) result[n++] = (i & size) ? Comparator{j, i} : Comparator{i, j};
            }
    std::array<bool, 32> needed{}; needed[13] = true;
    for (std::size_t i = n; i-- > 0;) {
        auto& c = result[i];
        c.minimum = needed[c.low]; c.maximum = needed[c.high];
        if (c.minimum || c.maximum) needed[c.low] = needed[c.high] = true;
    }
    return result;
}();

template<class T, std::size_t L>
inline void select(std::array<std::array<T, L>, 32>& values) {
    for (const auto& c : network) {
        if (!c.minimum && !c.maximum) continue;
        for (std::size_t lane = 0; lane < L; ++lane) {
            const auto a = values[c.low][lane], b = values[c.high][lane];
            if (c.minimum) values[c.low][lane] = std::min(a, b);
            if (c.maximum) values[c.high][lane] = std::max(a, b);
        }
    }
}

template<class T>
bool median3(std::vector<T>& values, const Grid& grid,
             const std::vector<std::size_t>& iz, const std::vector<std::size_t>& iy,
             const std::vector<std::size_t>& ix) {
    // nth_element's choice among signed-zero ties is implementation-specific.
    // Preserve that path when both signs occur instead of changing output bits.
    bool positive_zero = false, negative_zero = false;
    for (auto value : values) if (value == 0) {
        if (std::signbit(value)) negative_zero = true; else positive_zero = true;
        if (positive_zero && negative_zero) return false;
    }
    const auto source = values;
    const auto nz = grid.shape[0], ny = grid.shape[1], nx = grid.shape[2];
    constexpr std::size_t lanes = 32;
    std::array<std::array<T, lanes>, 32> window;
    for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; x += lanes) {
            const auto count = std::min(lanes, nx - x);
            for (std::size_t dz = 0; dz < 3; ++dz) for (std::size_t dy = 0; dy < 3; ++dy) {
                const auto* row = source.data() + iz[z * 3 + dz] + iy[y * 3 + dy];
                for (std::size_t dx = 0; dx < 3; ++dx)
                    for (std::size_t lane = 0; lane < lanes; ++lane)
                        window[(dz * 3 + dy) * 3 + dx][lane] = row[ix[(x + std::min(lane, count - 1)) * 3 + dx]];
            }
            // Comparisons overwrite padding as well as neighbors.
            for (std::size_t i = 27; i < 32; ++i) window[i].fill(std::numeric_limits<T>::infinity());
            select(window);
            for (std::size_t lane = 0; lane < count; ++lane)
                values[(z * ny + y) * nx + x + lane] = window[13][lane];
        }
    return true;
}
} // namespace vcr::augment::median_detail
