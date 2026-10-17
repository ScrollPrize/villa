#pragma once
#include "core.hpp"

namespace vcr::augment::median_detail {
struct Comparator { std::size_t low, high; bool minimum = false, maximum = false; };

// A 32-input bitonic sorting network, pruned backwards to the 14th output.
// Padding 27 neighbors with +infinity (255 for byte codes) makes that output
// the exact 3D median, even when real neighbors equal the padding value.
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
        auto& low = values[c.low]; auto& high = values[c.high];
        if constexpr (!std::is_same_v<T, std::uint8_t>) {
            // Retain the existing floating selector's code generation.
            for (std::size_t lane = 0; lane < L; ++lane) {
                const auto a = low[lane], b = high[lane];
                if (c.minimum) low[lane] = std::min(a, b);
                if (c.maximum) high[lane] = std::max(a, b);
            }
            continue;
        }
        // Keep schedule decisions outside the lane loops so vectorization
        // does not depend on the compiler's loop-unswitching heuristics.
        if (c.minimum && c.maximum) {
            for (std::size_t lane = 0; lane < L; ++lane) {
                const auto a = low[lane], b = high[lane];
                low[lane] = std::min(a, b); high[lane] = std::max(a, b);
            }
        } else if (c.minimum) {
            for (std::size_t lane = 0; lane < L; ++lane) low[lane] = std::min(low[lane], high[lane]);
        } else {
            for (std::size_t lane = 0; lane < L; ++lane) high[lane] = std::max(low[lane], high[lane]);
        }
    }
}

// Shared stencil and selector for floating values and sorted palette codes.
// Decode only the selected rank; byte codes use wider blocks with the same
// bounded scratch space. Source and destination must not alias.
template<std::size_t lanes = 32, class T, class Output, class Decode>
void median3_filter(const std::vector<T>& source, Output& values, const Grid& grid,
                    const std::vector<std::size_t>& iz, const std::vector<std::size_t>& iy,
                    const std::vector<std::size_t>& ix, Decode decode) {
    const auto nz = grid.shape[0], ny = grid.shape[1], nx = grid.shape[2];
    std::array<std::array<T, lanes>, 32> window;
    for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; x += lanes) {
            const auto count = std::min(lanes, nx - x);
            for (std::size_t dz = 0; dz < 3; ++dz) for (std::size_t dy = 0; dy < 3; ++dy) {
                const auto* row = source.data() + iz[z * 3 + dz] + iy[y * 3 + dy];
                for (std::size_t dx = 0; dx < 3; ++dx) {
                    auto& neighbors = window[(dz * 3 + dy) * 3 + dx];
                    if constexpr (!std::is_same_v<T, std::uint8_t>) {
                        for (std::size_t lane = 0; lane < lanes; ++lane)
                            neighbors[lane] = row[ix[(x + std::min(lane, count - 1)) * 3 + dx]];
                        continue;
                    }
                    // Only the block endpoints can need reflection. Interior
                    // lanes read contiguous neighbors, including in edge blocks.
                    neighbors[0] = row[ix[x * 3 + dx]];
                    neighbors[count - 1] = row[ix[(x + count - 1) * 3 + dx]];
                    if (count > 2) std::copy_n(row + ix[(x + 1) * 3 + dx], count - 2, neighbors.begin() + 1);
                    std::fill(neighbors.begin() + count, neighbors.end(), neighbors[count - 1]);
                }
            }
            // Comparisons overwrite padding as well as neighbors.
            for (std::size_t i = 27; i < 32; ++i) window[i].fill(std::numeric_limits<T>::has_infinity ?
                std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max());
            select(window);
            for (std::size_t lane = 0; lane < count; ++lane)
                values[(z * ny + y) * nx + x + lane] = decode(window[13][lane]);
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
    median3_filter(source, values, grid, iz, iy, ix, [](T value) { return value; });
    return true;
}
} // namespace vcr::augment::median_detail
