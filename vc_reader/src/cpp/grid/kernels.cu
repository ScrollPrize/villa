// CUDA sampling of vcr::grid rasters: one thread per pixel takes its point (grid::point
// or the plane), walks its layers, samples each through sampling::point from the first
// level whose bricks are resident, reduces and writes once: the CPU tile loop's code.
#include "cuda.hpp"
#include "../composite/reducer.hpp"
#include "../cuda_buffer.hpp"
#include "../sample_point.hpp"

#include <algorithm>
#include <cmath>

namespace vcr::grid::cuda_detail {
namespace {

constexpr std::int8_t kNone = -2, kMissing = -1;

// Notes `key` in the set once. A full set counts the key past its capacity instead.
__device__ void noteLacking(const LackingSet& set, std::int64_t key)
{
    const unsigned mask = set.capacity - 1;
    unsigned slot = unsigned(reader_cuda::home_slot(key, mask));
    for (unsigned probe = 0; probe < set.capacity; ++probe, slot = (slot + 1) & mask) {
        const auto previous = atomicCAS(set.table + slot, LackingSet::kEmpty, (unsigned long long)key);
        if (previous == (unsigned long long)key)
            return;
        if (previous == LackingSet::kEmpty) {
            const unsigned at = atomicAdd(set.count, 1u);
            if (at < set.capacity)
                set.keys[at] = key;
            return;
        }
    }
    atomicAdd(set.count, 1u);
}

// A level's resident bricks for sampling::point. A brick that is not resident is
// unavailable; with a lacking set, it is noted there.
struct BrickLookup {
    const reader_cuda::Source& s;
    const LackingSet* lacking;
    std::int64_t shape[3], block[3], stride[3];
    float fill;

    __device__ BrickLookup(const reader_cuda::Source& source, const LackingSet* lack) : s(source), lacking(lack)
    {
        for (int a = 0; a < 3; ++a) {
            shape[a] = s.shape[a];
            block[a] = s.brick[a];
        }
        stride[2] = 1;
        stride[1] = block[2];
        stride[0] = block[1] * block[2];
        fill = float(s.fill);
    }
    __device__ float load(const unsigned char* p) const { return float(*p); }
    __device__ const unsigned char* voxel(const std::int64_t* v, bool& available) const
    {
        const std::int64_t key = (v[0] / s.brick[0] * s.grid[1] + v[1] / s.brick[1]) * s.grid[2] + v[2] / s.brick[2];
        const std::int64_t local = ((v[0] % s.brick[0]) * s.brick[1] + v[1] % s.brick[1]) * s.brick[2] + v[2] % s.brick[2];
        for (std::size_t slot = reader_cuda::home_slot(key, s.mask);; slot = (slot + 1) & s.mask) {
            const reader_cuda::Slot entry = s.table[slot];
            if (entry.key == key)
                return entry.pointer + local;
            if (entry.key < 0)
                break;
        }
        available = false;
        if (lacking)
            noteLacking(*lacking, key);
        return nullptr;
    }
};

__global__ void samplePixels(Params P)
{
    const std::size_t pixels = std::size_t(P.width) * std::size_t(P.height);
    const std::size_t q = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= pixels)
        return;
    if (P.reduce)
        P.values[q] = nanf("");
    else
        for (unsigned k = 0; k < P.layers; ++k)
            P.values[std::size_t(k) * pixels + q] = nanf("");
    P.level[q] = kNone;
    if (P.pixels && !P.pixels[q])
        return;
    const int i = int(q % std::size_t(P.width)), j = int(q / std::size_t(P.width));
    float p[3], n[3];
    if (P.prepared) {
        for (int a = 0; a < 3; ++a) {
            p[a] = P.xyz[q * 3 + a];
            n[a] = P.normals ? P.normals[q * 3 + a] : 0;
        }
    } else if (P.plane) {
        for (int a = 0; a < 3; ++a) {
            p[a] = P.origin[a] + P.dy[a] * float(j) + P.dx[a] * float(i);
            n[a] = P.normal[a];
        }
    } else if (!point(P.map, P.grid, float(P.x0 + double(j) * P.kx + double(i) * P.sx),
                      float(P.y0 + double(j) * P.sy + double(i) * P.ky), p, n)) {
        return;  // no surface point here
    }
    composite::detail::Reducer reducer(P.composite);
    std::int8_t level = kNone;
    bool missing = false;
    for (unsigned k = 0; k < P.layers && !(P.reduce && reducer.done); ++k) {
        const float off = P.offsets[k];
        const float at[3]{p[0] + (off != 0 ? n[0] * off : 0.0f), p[1] + (off != 0 ? n[1] * off : 0.0f),
                          p[2] + (off != 0 ? n[2] * off : 0.0f)};
        if (!(isfinite(at[0]) && isfinite(at[1]) && isfinite(at[2])))
            continue;
        float value = 0;
        int li = 0;
        for (; li < P.levelCount; ++li) {
            BrickLookup lookup(P.levels[li].source, li == 0 && P.lacking.table ? &P.lacking : nullptr);
            float c[3];
            levelIndex(P.levels[li].map, at[0], at[1], at[2], c);
            if (sampling::point(lookup, P.levels[li].map.base, c, P.order, value))
                break;
        }
        if (li == P.levelCount) {
            missing = true;
            continue;
        }
        level = level > li ? level : std::int8_t(li);
        if (P.reduce)
            reducer.add(value);
        else
            P.values[std::size_t(k) * pixels + q] = value;
    }
    if (P.reduce)
        P.values[q] = reducer.result();
    P.level[q] = missing ? kMissing : level;
}

}  // namespace

void launch(const Params& params)
{
    const std::size_t pixels = std::size_t(params.width) * std::size_t(params.height);
    if (!pixels)
        return;
    samplePixels<<<unsigned((pixels + 255) / 256), 256, 0, vcr::device::stream()>>>(params);
    device::check(cudaGetLastError());
}

}  // namespace vcr::grid::cuda_detail
