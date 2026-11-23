#pragma once
#include "../core.hpp"

namespace vcr::transforms::median_detail {
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

template<class T, std::size_t L, std::size_t S, std::size_t W>
inline void select(std::array<std::array<T, L>, S>& values, const Network<W>& network) {
    for (std::size_t i = 0; i < network.count; ++i) {
        const auto& c = network.comparisons[i];
        if (!c.minimum && !c.maximum) continue;
        auto& low = values[c.low]; auto& high = values[c.high];
        // Keep schedule decisions outside the lane loops so vectorization
        // does not depend on the compiler's loop-unswitching heuristics.
        // std::min/std::max select the same element in every lane either way.
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

// 3x3 median of a plane: the 19-exchange median-of-9 selector on values held in registers,
// vectorized across contiguous interior columns; the two edge columns use reflected indices.
// It selects the same element as any exact median for the finite, sign-consistent inputs
// that reach it (median3 rejects mixed signed zeros; samples are validated finite).
// By value, with std::min/std::max's exact comparisons ((b < a) ? b : a and (a < b) ? b : a).
template<class T> inline T lesser(T a, T b) { return b < a ? b : a; }
template<class T> inline T greater(T a, T b) { return a < b ? b : a; }
template<class T> inline void exchange(T& a, T& b) { const T low = lesser(a, b), high = greater(a, b); a = low; b = high; }
template<class T> inline T median9(T p0, T p1, T p2, T p3, T p4, T p5, T p6, T p7, T p8) {
    exchange(p1, p2); exchange(p4, p5); exchange(p7, p8);
    exchange(p0, p1); exchange(p3, p4); exchange(p6, p7);
    exchange(p1, p2); exchange(p4, p5); exchange(p7, p8);
    exchange(p0, p3); exchange(p5, p8); exchange(p4, p7);
    exchange(p3, p6); exchange(p1, p4); exchange(p2, p5);
    exchange(p4, p7); exchange(p4, p2); exchange(p6, p4);
    exchange(p4, p2);
    return p4;
}
template<class T> inline T median3_of(T a, T b, T c) { return greater(lesser(a, b), lesser(greater(a, b), c)); }
// Each vertical triple is sorted once and shared by the three windows containing it: the
// median of nine is then med3(max of lows, med3 of middles, min of highs), exactly.
template<class T, class Output, class Decode, bool CheckZeros = false>
VCR_TARGET_CLONES bool median9_plane(const T* source, Output& values, std::size_t ny, std::size_t nx,
                                     const std::vector<std::size_t>& iy, const std::vector<std::size_t>& ix, Decode decode) {
    // With CheckZeros, also report whether both signed zeros occur (each row is read once as
    // the middle row; the result is then to be discarded, as median3 would not use this path).
    using Bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
    Bits positive = 0, negative = 0;
    std::vector<T> scratch(3 * nx);
    T* __restrict low = scratch.data(); T* __restrict middle = low + nx; T* __restrict high = middle + nx;
    for (std::size_t y = 0; y < ny; ++y) {
        const T* __restrict a = source + iy[y * 3];
        const T* __restrict b = source + iy[y * 3 + 1];
        const T* __restrict c = source + iy[y * 3 + 2];
        for (std::size_t x = 0; x < nx; ++x) {
            T p = a[x], q = b[x], r = c[x];
            if constexpr (CheckZeros) {
                constexpr Bits sign = Bits(1) << (8 * sizeof(T) - 1);
                const auto bits = std::bit_cast<Bits>(q);
                positive |= Bits(bits == 0); negative |= Bits(bits == sign);
            }
            exchange(p, q); exchange(q, r); exchange(p, q);
            low[x] = p; middle[x] = q; high[x] = r;
        }
        const auto pixel = [&](std::size_t l, std::size_t m, std::size_t r) {
            return median3_of(greater(greater(low[l], low[m]), low[r]), median3_of(middle[l], middle[m], middle[r]),
                              lesser(lesser(high[l], high[m]), high[r]));
        };
        const std::size_t row = y * nx;
        values[row] = decode(pixel(ix[0], ix[1], ix[2]));
        if constexpr (std::is_pointer_v<std::remove_reference_t<decltype(&values[0])>> && std::is_same_v<std::remove_cvref_t<decltype(values[0])>, T>) {
            T* __restrict out = &values[row];
            for (std::size_t x = 1; x + 1 < nx; ++x) out[x] = decode(pixel(x - 1, x, x + 1));
        } else for (std::size_t x = 1; x + 1 < nx; ++x) values[row + x] = decode(pixel(x - 1, x, x + 1));
        if (nx > 1) values[row + nx - 1] = decode(pixel(ix[(nx - 1) * 3], ix[(nx - 1) * 3 + 1], ix[(nx - 1) * 3 + 2]));
    }
    return positive && negative;
}

// median9_plane in place. Row y reads rows y-1, y and y+1 (reflected); of these only y-1 has
// been overwritten, so its original is kept in a row buffer. Results equal median9_plane's.
template<class T>
VCR_TARGET_CLONES void median9_plane_inplace(T* data, std::size_t ny, std::size_t nx,
                                             const std::vector<std::size_t>& iy, const std::vector<std::size_t>& ix) {
    std::vector<T> scratch(5 * nx);
    T* __restrict low = scratch.data(); T* __restrict middle = low + nx; T* __restrict high = middle + nx;
    T* previous = high + nx; T* row = previous + nx;
    for (std::size_t y = 0; y < ny; ++y) {
      {  // restrict-qualified reads end before the row buffers are rewritten below
        const auto source = [&](std::size_t offset) -> const T* { return offset / nx < y ? previous : data + offset; };
        const T* __restrict a = source(iy[y * 3]); const T* __restrict b = source(iy[y * 3 + 1]); const T* __restrict c = source(iy[y * 3 + 2]);
        for (std::size_t x = 0; x < nx; ++x) {
            T p = a[x], q = b[x], r = c[x];
            exchange(p, q); exchange(q, r); exchange(p, q);
            low[x] = p; middle[x] = q; high[x] = r;
        }
        const auto pixel = [&](std::size_t l, std::size_t m, std::size_t r) {
            return median3_of(greater(greater(low[l], low[m]), low[r]), median3_of(middle[l], middle[m], middle[r]),
                              lesser(lesser(high[l], high[m]), high[r]));
        };
        T* __restrict out = row;
        out[0] = pixel(ix[0], ix[1], ix[2]);
        for (std::size_t x = 1; x + 1 < nx; ++x) out[x] = pixel(x - 1, x, x + 1);
        if (nx > 1) out[nx - 1] = pixel(ix[(nx - 1) * 3], ix[(nx - 1) * 3 + 1], ix[(nx - 1) * 3 + 2]);
      }
        std::copy_n(data + y * nx, nx, previous);
        std::copy_n(row, nx, data + y * nx);
    }
}

// One stencil depth (1 for 3x3, 3 for 3x3x3); padding lanes repeat the last voxel.
template<std::size_t lanes, std::size_t depth, class T, class Output, class Decode, std::size_t S, std::size_t W>
VCR_TARGET_CLONES void median3_rows(const T* source, Output& values, std::size_t nz, std::size_t ny, std::size_t nx,
                  const std::vector<std::size_t>& iz, const std::vector<std::size_t>& iy,
                  const std::vector<std::size_t>& ix, Decode decode,
                  std::array<std::array<T, lanes>, S>& window, const Network<W>& selector) {
    for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; x += lanes) {
            const auto count = std::min(lanes, nx - x);
            for (std::size_t dz = 0; dz < depth; ++dz) for (std::size_t dy = 0; dy < 3; ++dy) {
                const auto* row = source + (depth == 3 ? iz[z * 3 + dz] : 0) + iy[y * 3 + dy];
                for (std::size_t dx = 0; dx < 3; ++dx) {
                    auto& neighbors = window[(dz * 3 + dy) * 3 + dx];
                    // Only the block endpoints can need reflection. Interior
                    // lanes read contiguous neighbors, including in edge blocks.
                    neighbors[0] = row[ix[x * 3 + dx]];
                    neighbors[count - 1] = row[ix[(x + count - 1) * 3 + dx]];
                    const auto* interior = row + ix[(x + 1) * 3 + dx];
                    // Full blocks copy a compile-time length (vector moves, not a memmove call per row).
                    if (count == lanes) for (std::size_t lane = 1; lane + 1 < lanes; ++lane) neighbors[lane] = interior[lane - 1];
                    else if (count > 2) std::copy_n(interior, count - 2, neighbors.begin() + 1);
                    std::fill(neighbors.begin() + count, neighbors.end(), neighbors[count - 1]);
                }
            }
            // Padding has been eliminated from the generated schedule.
            select(window, selector);
            for (std::size_t lane = 0; lane < count; ++lane)
                values[(z * ny + y) * nx + x + lane] = decode(window[selector.output][lane]);
        }
}

// Shared stencil and selector for floating values and sorted palette codes:
// 3x3x3 for 3D grids, 3x3 for 2D. Decode only the selected rank; byte codes use
// wider blocks with the same bounded scratch space. Source and destination must not alias.
template<std::size_t lanes = 32, class T, class Output, class Decode>
void median3_filter(const T* source, Output& values, const Grid& grid,
                    const std::vector<std::size_t>& iz, const std::vector<std::size_t>& iy,
                    const std::vector<std::size_t>& ix, Decode decode) {
    if (grid.shape.size() == 2) {
        median9_plane(source, values, grid.shape[0], grid.shape[1], iy, ix, decode);
        return;
    }
    std::array<std::array<T, lanes>, 32> window;
    median3_rows<lanes, 3>(source, values, grid.shape[0], grid.shape[1], grid.shape[2], iz, iy, ix, decode, window, network);
}

// Whether both signed zeros occur: nth_element's choice among them is implementation-specific.
template<class T> VCR_TARGET_CLONES bool mixed_zeros(const T* values, std::size_t n) {
    if constexpr (std::is_floating_point_v<T>) {
        // +0 and -0 by bit pattern, as OR reductions (vectorizable, no early exit needed).
        using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
        constexpr Bits sign = Bits(1) << (8 * sizeof(T) - 1);
        Bits positive = 0, negative = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto bits = std::bit_cast<Bits>(values[i]);
            positive |= Bits(bits == 0); negative |= Bits(bits == sign);
        }
        return positive && negative;
    } else return false;  // integers have one zero
}
template<class T>
bool median3(std::vector<T>& values, const Grid& grid,
             const std::vector<std::size_t>& iz, const std::vector<std::size_t>& iy,
             const std::vector<std::size_t>& ix) {
    // nth_element's choice among signed-zero ties is implementation-specific.
    // Preserve that path when both signs occur instead of changing output bits.
    if (mixed_zeros(values.data(), values.size())) return false;
    const auto source = values;
    median3_filter(source.data(), values, grid, iz, iy, ix, [](T value) { return value; });
    return true;
}
} // namespace vcr::transforms::median_detail
