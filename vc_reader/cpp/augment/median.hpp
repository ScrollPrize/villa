#pragma once
#include "core.hpp"

namespace vcr::augment::median_detail {
struct Comparator { std::size_t low, high; bool minimum = false, maximum = false; };

// Generate an odd-even merge selector, eliminate maximum padding values,
// and prune backwards to the requested rank. SortedGroup permits callers to
// reuse sorted columns instead of sorting every overlapping window again.
template<std::size_t W> struct Network {
    std::array<Comparator, W * 9> comparisons{};
    std::size_t count = 0, output = 0;
};
template<std::size_t N, std::size_t SortedGroup = 1>
constexpr auto make_network() {
    constexpr auto group_width = std::bit_ceil(SortedGroup);
    constexpr auto width = std::bit_ceil((N + SortedGroup - 1) / SortedGroup * group_width);
    Network<width> result;
    std::array<std::size_t, width> wires{};
    for (std::size_t i = 0; i < width; ++i)
        wires[i] = i % group_width < SortedGroup ? i / group_width * SortedGroup + i % group_width : N;
    auto compare = [&](std::size_t low, std::size_t high) {
        if (wires[high] >= N) return;
        if (wires[low] >= N) { std::swap(wires[low], wires[high]); return; }
        result.comparisons[result.count++] = {wires[low], wires[high]};
    };
    auto merge = [&](auto&& self, std::size_t first, std::size_t size, std::size_t stride) -> void {
        const auto step = 2 * stride;
        if (step < size) {
            self(self, first, size, step);
            self(self, first + stride, size, step);
            for (auto i = first + stride; i + stride < first + size; i += step) compare(i, i + stride);
        } else compare(first, first + stride);
    };
    auto sort = [&](auto&& self, std::size_t first, std::size_t size) -> void {
        if (size == group_width) return;
        self(self, first, size / 2);
        self(self, first + size / 2, size / 2);
        merge(merge, first, size, 1);
    };
    sort(sort, 0, width);
    result.output = wires[N / 2];
    std::array<bool, width> needed{}; needed[result.output] = true;
    for (std::size_t i = result.count; i-- > 0;) {
        auto& c = result.comparisons[i];
        c.minimum = needed[c.low]; c.maximum = needed[c.high];
        if (c.minimum || c.maximum) needed[c.low] = needed[c.high] = true;
    }
    std::size_t count = 0;
    for (std::size_t i = 0; i < result.count; ++i)
        if (result.comparisons[i].minimum || result.comparisons[i].maximum)
            result.comparisons[count++] = result.comparisons[i];
    result.count = count;
    return result;
}
inline constexpr auto network = make_network<27>();
inline constexpr auto column_network = make_network<27, 3>();
inline constexpr auto plane_network = make_network<9, 3>();

template<class T, std::size_t L>
inline void select(std::array<std::array<T, L>, 32>& values) {
    for (std::size_t i = 0; i < network.count; ++i) {
        const auto& c = network.comparisons[i];
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
            // Padding has been eliminated from the generated schedule.
            select(window);
            for (std::size_t lane = 0; lane < count; ++lane)
                values[(z * ny + y) * nx + x + lane] = decode(window[network.output][lane]);
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
