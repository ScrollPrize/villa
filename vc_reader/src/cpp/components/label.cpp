#include <vc_reader/components.hpp>
#include "../thread_pool.hpp"
#include "rows.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Run-based two-pass labelling over row bit masks. Each row's foreground runs are
// joined with the runs of the previously scanned rows they touch, found with word
// operations; the union-find keeps the smallest run index as root, so the first
// run of each component, and hence the numbering, does not depend on the order of
// unions or on threads.
namespace vcr::components {
namespace {
struct Run { std::uint32_t begin, end; };
using detail::neighbor_rows;

std::uint32_t find(std::uint32_t* parent, std::uint32_t i) {
    while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return i;
}

static_assert(std::endian::native == std::endian::little, "nonzero_bits assumes little-endian loads");
// Bit k is set when byte k of p[0..8) is nonzero. Per byte, (b & 0x7f) + 0x7f
// sets the high bit for any low bits and cannot carry; the multiply gathers the
// eight high bits into the top byte.
inline std::uint64_t nonzero_bits(const std::uint8_t* p) {
    std::uint64_t w; std::memcpy(&w, p, 8);
    constexpr std::uint64_t low = 0x7f7f7f7f7f7f7f7fULL;
    const auto high = (((w & low) + low) | w) >> 7 & 0x0101010101010101ULL;
    return (high * 0x0102040810204080ULL) >> 56;
}
// One row as 64-voxel words: nonzero voxels, run starts (set after clear), run
// last voxels (set before clear) and the starts in earlier words. Returns the
// row's run count.
std::size_t row_bits(const std::uint8_t* m, std::size_t nx, std::uint64_t* bits, std::uint64_t* starts,
                     std::uint64_t* lasts, std::uint32_t* before) {
    const std::size_t words = (nx + 63) / 64;
    for (std::size_t base = 0, w = 0; base < nx; base += 64, ++w) {
        const auto n = std::min<std::size_t>(64, nx - base);
        std::uint64_t x = 0; std::size_t k = 0;
        for (; k + 8 <= n; k += 8) x |= nonzero_bits(m + base + k) << k;
        for (; k < n; ++k) x |= std::uint64_t(m[base + k] != 0) << k;
        bits[w] = x;
    }
    std::uint32_t count = 0;
    for (std::size_t w = 0; w < words; ++w) {
        const auto x = bits[w];
        const auto left = x << 1 | (w ? bits[w - 1] >> 63 : 0), right = x >> 1 | (w + 1 < words ? bits[w + 1] << 63 : 0);
        starts[w] = x & ~left; lasts[w] = x & ~right;
        before[w] = count; count += std::uint32_t(std::popcount(starts[w]));
    }
    return count;
}
// The runs of one row from its words: starts and ends (a clear bit after a set
// one) alternate, so each is written in order; a run reaching the row end ends at nx.
void row_runs(const std::uint64_t* words, std::size_t nx, Run* runs) {
    std::size_t begins = 0, ends = 0; std::uint64_t carry = 0;
    for (std::size_t base = 0, w = 0; base < nx; base += 64, ++w) {
        const auto bits = words[w], previous = bits << 1 | carry;
        for (auto s = bits & ~previous; s; s &= s - 1) runs[begins++].begin = std::uint32_t(base + std::size_t(std::countr_zero(s)));
        // Bits past the row end are clear, so an end at bit n of a partial word is nx.
        for (auto e = ~bits & previous; e; e &= e - 1) runs[ends++].end = std::uint32_t(base + std::size_t(std::countr_zero(e)));
        carry = bits >> 63;
    }
    if (ends < begins) runs[ends].end = std::uint32_t(nx);
}

ThreadPool& pool() {
    // Never destroyed; see samplingPool() in thread_pool.hpp.
    static auto* workers = new ThreadPool(std::max(1u, std::thread::hardware_concurrency()) - 1);
    return *workers;
}
}

std::size_t label(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
                  std::int32_t* labels, std::size_t threads) {
    const auto dim = shape.size();
    if (dim != 2 && dim != 3) throw std::invalid_argument("label expects a 2D or 3D mask");
    connectivity = detail::planar_connectivity(dim, connectivity);
    const auto table = neighbor_rows(dim, connectivity);
    const std::span neighbors(table.rows.data(), std::size_t(table.count));
    const std::size_t nz = dim == 3 ? shape[0] : 1, ny = shape[dim - 2], nx = shape[dim - 1];
    if (nx >= UINT32_MAX - 1) throw std::invalid_argument("label rows are limited to 2^32 - 2 voxels");
    const std::size_t rows = nz * ny;
    if (!rows || !nx) return 0;

    // Contiguous row chunks; a chunk unions runs within itself only, so chunks run
    // concurrently, and the rows reaching into an earlier chunk are joined after.
    if (!threads) threads = pool().size() + 1;
    const std::size_t chunks = std::max<std::size_t>(1, std::min({threads, rows, rows * nx / 65536 + 1}));
    std::vector<std::size_t> first_row(chunks + 1);
    for (std::size_t c = 0; c <= chunks; ++c) first_row[c] = rows * c / chunks;
    const auto for_chunks = [&](const auto& fn) {
        if (chunks == 1) fn(std::size_t(0)); else pool().parallel_for(chunks, fn);
    };

    // Row bit masks and run counts; a run is identified by its row and rank.
    const std::size_t words = (nx + 63) / 64;
    std::vector<std::uint64_t> bits(rows * words), starts(rows * words), lasts(rows * words);
    std::vector<std::uint32_t> before(rows * words);
    std::vector<std::size_t> row_start(rows + 1, 0);
    for_chunks([&](std::size_t c) {
        for (auto r = first_row[c]; r < first_row[c + 1]; ++r) {
            const auto o = r * words;
            row_start[r + 1] = row_bits(mask + r * nx, nx, bits.data() + o, starts.data() + o, lasts.data() + o, before.data() + o);
        }
    });
    for (std::size_t r = 0; r < rows; ++r) row_start[r + 1] += row_start[r];
    const auto total = row_start[rows];
    if (total >= UINT32_MAX) throw std::invalid_argument("label supports fewer than 2^32 - 1 runs");
    std::vector<std::uint32_t> parent(total);
    for_chunks([&](std::size_t c) {
        for (auto i = row_start[first_row[c]]; i < row_start[first_row[c + 1]]; ++i) parent[i] = std::uint32_t(i);
    });

    // Joins each run of row r with the runs of its neighbour rows q in [low, high)
    // that overlap it, or touch it diagonally, found from the row bits: each such
    // pair shows at one bit, the first voxel of their overlap or, touching only
    // diagonally, the left run's last voxel where the right run starts one voxel
    // on. A run's index in its row is the number of run starts up to any of its
    // voxels, less one. Linking the larger root under the smaller keeps each
    // component's first run as its root.
    //
    // With 26-connectivity, where the row below (z - 1, y) has a voxel at a contact
    // (or either side of a diagonal one), that voxel's run touches both runs, and
    // is joined with each as a neighbour row of both; the pair itself is skipped.
    // Likewise for pairs with row (z - 1, y - 1) where row (z, y - 1) has a voxel:
    // its pair with this row is kept or implied by the row below, and its pair
    // with (z - 1, y - 1), the row below it, is never skipped. With 18-connectivity
    // the same holds when that voxel is in line (same x) with the other row's voxel.
    const bool below_skips = dim == 3 && connectivity != 6, full = connectivity == 26;
    const auto join_row = [&](std::size_t r, std::size_t low, std::size_t high) {
        const auto z = r / ny, y = r % ny;
        const auto a0 = row_start[r];
        if (a0 == row_start[r + 1]) return;
        auto* up = parent.data();
        const auto* A = bits.data() + r * words; const auto* SA = starts.data() + r * words;
        const auto* LA = lasts.data() + r * words; const auto* PA = before.data() + r * words;
        const auto rank = [](const std::uint64_t* S, const std::uint32_t* P, std::size_t x) {
            return P[x >> 6] + std::uint32_t(std::popcount(S[x >> 6] & (~0ULL >> (63 - (x & 63))))) - 1;
        };
        const std::uint64_t* below = below_skips && z ? bits.data() + (r - ny) * words : nullptr;
        for (const auto& n : neighbors) {
            const auto zz = std::ptrdiff_t(z) + n.dz, yy = std::ptrdiff_t(y) + n.dy;
            if (zz < 0 || yy < 0 || yy >= std::ptrdiff_t(ny)) continue;
            const auto q = std::size_t(zz) * ny + std::size_t(yy);
            if (q < low || q >= high) continue;
            const auto* skip_by = below && !(n.dz == -1 && n.dy == 0) ? below : nullptr;
            const auto* skip_also = below_skips && n.dz == -1 && n.dy == -1 && row_start[r - 1] != row_start[r]
                ? bits.data() + (r - 1) * words : nullptr;  // row (z, y - 1)
            const auto b0 = row_start[q];
            if (b0 == row_start[q + 1]) continue;
            const auto* B = bits.data() + q * words; const auto* SB = starts.data() + q * words;
            const auto* LB = lasts.data() + q * words; const auto* PB = before.data() + q * words;
            const auto join = [&](std::uint32_t run, std::uint32_t other) {
                auto ra = up[run]; if (up[ra] != ra) ra = find(up, ra);
                auto rb = up[other];
                if (rb == ra) return;
                if (up[rb] != rb) rb = find(up, rb);
                if (ra < rb) up[rb] = ra; else if (rb < ra) up[ra] = rb;
            };
            std::uint64_t overlap_left = 0;
            const std::uint64_t diagonal = n.diagonal ? ~0ULL : 0;
            for (std::size_t w = 0; w < words; ++w) {
                const auto overlap = A[w] & B[w];
                const auto by = skip_by ? skip_by[w] : 0, also = skip_also ? skip_also[w] : 0;
                const auto by_next = skip_by ? skip_by[w] >> 1 | (w + 1 < words ? skip_by[w + 1] << 63 : 0) : 0;
                const auto also_next = skip_also ? skip_also[w] >> 1 | (w + 1 < words ? skip_also[w + 1] << 63 : 0) : 0;
                const auto at = by | also;
                // 18-connectivity needs the skipping voxel in line with this row's (x + 1 or x).
                const auto skip_b_right = full ? at | by_next | also_next : by_next;
                const auto skip_a_right = full ? at | by_next | also_next : by;
                const auto first = overlap & ~(overlap << 1 | overlap_left) & ~at;
                overlap_left = overlap >> 63;
                // A run of A ends at x and one of B starts at x + 1, or the reverse.
                // The three contact kinds are disjoint, so one pass takes them all.
                const auto b_next = SB[w] >> 1 | (w + 1 < words ? SB[w + 1] << 63 : 0);
                const auto a_next = SA[w] >> 1 | (w + 1 < words ? SA[w + 1] << 63 : 0);
                const auto b_right = LA[w] & b_next & ~skip_b_right & diagonal, a_right = LB[w] & a_next & ~skip_a_right & diagonal;
                for (auto f = first | b_right | a_right; f; f &= f - 1) {
                    const auto bit = std::countr_zero(f);
                    const auto x = w * 64 + std::size_t(bit);
                    join(std::uint32_t(a0 + rank(SA, PA, x + (a_right >> bit & 1))),
                         std::uint32_t(b0 + rank(SB, PB, x + (b_right >> bit & 1))));
                }
            }
        }
        // Later rows join with this one: point its runs at their roots.
        for (auto i = a0; i < row_start[r + 1]; ++i) up[i] = find(up, std::uint32_t(i));
    };
    for_chunks([&](std::size_t c) {
        for (auto r = first_row[c]; r < first_row[c + 1]; ++r) join_row(r, first_row[c], rows);
    });
    // Then rows reaching into earlier chunks; neighbour rows are at most ny + 1 rows back.
    for (std::size_t c = 1; c < chunks; ++c)
        for (auto r = first_row[c]; r < std::min(first_row[c + 1], first_row[c] + ny + 1); ++r) join_row(r, 0, first_row[c]);

    // Roots are each component's first run in scan order: number them in that order.
    // Unions point to smaller indices, so in scan order each run's parent slot
    // already holds its label when the run is reached.
    std::size_t count = 0;
    for (std::size_t i = 0; i < total; ++i) {
        const auto up = parent[i];
        if (up == i) {
            if (count == std::size_t(INT32_MAX)) throw std::overflow_error("label count exceeds INT32_MAX");
            parent[i] = std::uint32_t(++count);
        } else parent[i] = parent[up];
    }
    for_chunks([&](std::size_t c) {
        std::vector<Run> row((nx + 1) / 2);
        for (auto r = first_row[c]; r < first_row[c + 1]; ++r) {
            auto* out = labels + r * nx;
            std::fill(out, out + nx, 0);
            row_runs(bits.data() + r * words, nx, row.data());
            for (auto i = row_start[r]; i < row_start[r + 1]; ++i) {
                const auto& run = row[i - row_start[r]];
                std::fill(out + run.begin, out + run.end, std::int32_t(parent[i]));
            }
        }
    });
    return count;
}

void fillHoles(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
               std::uint8_t* filled, std::size_t threads) {
    if (!connectivity) connectivity = shape.size() == 3 ? 6 : 4;
    std::size_t n = 1; for (auto e : shape) n *= e;
    if (!n) { label(mask, shape, connectivity, nullptr, threads); return; }  // validates shape and connectivity
    if (!threads) threads = pool().size() + 1;
    const std::size_t chunks = std::max<std::size_t>(1, std::min(threads, n / 65536 + 1));
    const auto for_ranges = [&](const auto& fn) {
        const auto each = [&](std::size_t c) { fn(n * c / chunks, n * (c + 1) / chunks); };
        if (chunks == 1) each(0); else pool().parallel_for(chunks, each);
    };
    std::vector<std::uint8_t> background(n);
    for_ranges([&](std::size_t a, std::size_t b) { for (auto i = a; i < b; ++i) background[i] = !mask[i]; });
    std::vector<std::int32_t> labels(n);
    const auto count = label(background.data(), shape, connectivity, labels.data(), threads);
    // Background components with a voxel on the border are not holes.
    std::vector<std::uint8_t> open(count + 1, 0);
    const auto dim = shape.size();
    const std::size_t nz = dim == 3 ? shape[0] : 1, ny = shape[dim - 2], nx = shape[dim - 1];
    for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y) {
        const auto* row = labels.data() + (z * ny + y) * nx;
        if ((dim == 3 && (z == 0 || z + 1 == nz)) || y == 0 || y + 1 == ny) { for (std::size_t x = 0; x < nx; ++x) open[std::size_t(row[x])] = 1; }
        else { open[std::size_t(row[0])] = 1; open[std::size_t(row[nx - 1])] = 1; }
    }
    open[0] = 1;  // foreground stays as it is
    for_ranges([&](std::size_t a, std::size_t b) {
        for (auto i = a; i < b; ++i) filled[i] = mask[i] || (labels[i] && !open[std::size_t(labels[i])]);
    });
}
}
