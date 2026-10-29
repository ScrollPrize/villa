// Groups the pixels of a sampling request into tiles whose dense blocks stay
// compact, in waves of bounded block memory. Both backends execute these plans.
#pragma once
#include "math.hpp"
#include "../array.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace vcr::slice::detail {

struct Wave {
    std::vector<Tile> tiles;
    std::int64_t elements = 0;  // float32 block elements
};

inline constexpr std::int64_t kWaveElements = 32ll << 20;  // 128 MiB of float32 blocks

// Calls run(wave) for consecutive waves covering every unit of `src`.
template <class Run>
void plan(const Source& src, const Index& shape, int order, Run&& run)
{
    if (shape.size() != 3)
        throw std::invalid_argument("sampling requires a 3-D array");
    const std::int64_t W = src.width > 0 ? src.width : std::max<std::int64_t>(src.n, 1);
    const std::int64_t H = src.width > 0 ? (src.n + W - 1) / W : 1;
    // Band extremes: the samples of a pixel lie between its first and last offset.
    int kLo = 0, kHi = 0;
    if (src.kind == Source::Band)
        for (int k = 1; k < src.K; ++k) {
            if (src.offsets[k] < src.offsets[kLo])
                kLo = k;
            if (src.offsets[k] > src.offsets[kHi])
                kHi = k;
        }
    // How far the neighbourhood of a point touching the array can reach past its edge.
    const std::int64_t margin = order == 3 ? 3 : order == 1 ? 1 : 0;
    std::vector<Tile> pending;
    const std::int64_t tw = src.width > 0 ? (src.K == 1 ? 64 : 32) : (src.K == 1 ? 4096 : 1024);
    const std::int64_t th = src.width > 0 ? tw : 1;
    for (std::int64_t y = 0; y < H; y += th)
        for (std::int64_t x = 0; x < W; x += tw)
            pending.push_back({x, y, std::min(tw, W - x), std::min(th, H - y)});
    std::reverse(pending.begin(), pending.end());  // pop_back walks the image in order

    Wave wave;
    while (!pending.empty()) {
        Tile t = pending.back();
        pending.pop_back();
        std::int64_t lo[3] = {std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max(),
                              std::numeric_limits<std::int64_t>::max()};
        std::int64_t hi[3] = {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::min(),
                              std::numeric_limits<std::int64_t>::min()};
        std::int64_t units = 0;
        for (std::int64_t yy = t.y0; yy < t.y0 + t.h; ++yy)
            for (std::int64_t xx = t.x0; xx < t.x0 + t.w; ++xx) {
                const std::int64_t p = yy * W + xx;
                if (p >= src.n)
                    continue;
                ++units;
                for (const int k : {kLo, kHi}) {
                    float x, y, z;
                    if (!src.point(p, k, x, y, z))
                        continue;
                    // A band's extreme may leave the array while its middle does not.
                    if (src.kind != Source::Band && outside(x, y, z, shape[0], shape[1], shape[2], order))
                        continue;
                    const float c[3] = {z, y, x};
                    for (int a = 0; a < 3; ++a) {
                        if (!(fabsf(c[a]) < 1.0e15f))
                            continue;
                        std::int64_t first, last;
                        footprint(c[a], order, first, last);
                        lo[a] = std::min(lo[a], std::max(first, -margin));
                        hi[a] = std::max(hi[a], std::min(last, shape[a] - 1 + margin));
                    }
                }
            }
        std::int64_t voxels = 0;
        bool block = lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2];
        if (block)
            voxels = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1);
        const std::int64_t budget = std::max<std::int64_t>(32768, 16 * units * src.K);
        if (block && voxels > budget && t.w * t.h > 1) {
            Tile a = t, b = t;  // halve the longer side
            if (t.w >= t.h) {
                a.w = t.w / 2;
                b.x0 = t.x0 + a.w;
                b.w = t.w - a.w;
            } else {
                a.h = t.h / 2;
                b.y0 = t.y0 + a.h;
                b.h = t.h - a.h;
            }
            pending.push_back(b);
            pending.push_back(a);
            continue;
        }
        if (block) {
            if (voxels > (std::int64_t(1) << 31))
                throw std::length_error("a sampled pixel spans more than 2^31 voxels");
            t.hasBlock = true;
            t.block = Block{wave.elements, lo[0], lo[1], lo[2], std::int32_t(hi[0] - lo[0] + 1),
                            std::int32_t(hi[1] - lo[1] + 1), std::int32_t(hi[2] - lo[2] + 1)};
            if (wave.elements + voxels > kWaveElements && !wave.tiles.empty()) {
                run(wave);
                wave = Wave{};
                t.block.offset = 0;
            }
            wave.elements += voxels;
        }
        wave.tiles.push_back(t);
    }
    if (!wave.tiles.empty())
        run(wave);
}

// Reads every block of a wave as float32 (fill outside the array) into `buffer`.
inline void readBlocks(Array& array, const ElementType& type, const Wave& wave, std::vector<float>& buffer)
{
    buffer.resize(std::size_t(wave.elements));
    std::vector<Index> starts, sizes;
    std::vector<std::uint8_t*> outs;
    for (const auto& t : wave.tiles) {
        if (!t.hasBlock)
            continue;
        starts.push_back({t.block.oz, t.block.oy, t.block.ox});
        sizes.push_back({t.block.nz, t.block.ny, t.block.nx});
        outs.push_back(reinterpret_cast<std::uint8_t*>(buffer.data() + t.block.offset));
    }
    if (!starts.empty())
        array.read_many_float(starts, sizes, outs, type, false);
}

inline Source checkedBand(const float* base, const float* dirs, std::size_t n, std::size_t width,
                          const float* offsets, std::size_t count)
{
    if (count == 0 || count > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("a band needs at least one offset");
    Source s;
    s.kind = Source::Band;
    s.base = base;
    s.dirs = dirs;
    s.offsets = offsets;
    s.K = int(count);
    s.n = std::int64_t(n);
    s.width = std::int64_t(width);
    return s;
}

}  // namespace vcr::slice::detail
