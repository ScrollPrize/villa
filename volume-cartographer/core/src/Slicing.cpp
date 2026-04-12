#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/cache/BlockCache.hpp"
#include "vc/core/cache/BlockPipeline.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>

#if defined(_MSC_VER)
#define VC_FORCE_INLINE __forceinline
#else
#define VC_FORCE_INLINE __attribute__((always_inline)) inline
#endif

namespace {

using vc::cache::Block;
using vc::cache::BlockKey;
using vc::cache::BlockPtr;
using vc::cache::BlockPipeline;
using vc::cache::kBlockSize;

constexpr int kBlockShift = 4;       // log2(16)
constexpr int kBlockMask = 15;
constexpr size_t kStrideY = kBlockSize;              // 16
constexpr size_t kStrideZ = kBlockSize * kBlockSize; // 256

VC_FORCE_INLINE bool isnan_bitwise(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0;
}

VC_FORCE_INLINE bool isfinite_bitwise(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return (u & 0x7f800000u) != 0x7f800000u;
}

template<typename T>
VC_FORCE_INLINE void nt_store(T* dst, T val) {
#if __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(val, dst);
#else
    *dst = val;
#endif
}

VC_FORCE_INLINE void nt_store_u32(uint32_t* dst, uint32_t val) {
#if __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(val, dst);
#else
    *dst = val;
#endif
}

// Volume dimensions at a given level, used for bounds checks.
struct VolumeShape {
    int sz = 0, sy = 0, sx = 0;
    float szf = 0.f, syf = 0.f, sxf = 0.f;

    VolumeShape() = default;
    explicit VolumeShape(BlockPipeline& cache, int level) {
        auto s = cache.levelShape(level);
        sz = s[0]; sy = s[1]; sx = s[2];
        szf = float(sz); syf = float(sy); sxf = float(sx);
    }
};

// Direct-mapped block cache keyed by (bz, by, bx). On miss, consults the
// block cache only — never blocks on disk or network. Missing blocks make
// sampleInt return 0 (black) for that voxel. Viewport-demand fetches feed
// the cache asynchronously via BlockPipeline::fetchInteractive.
template<typename T, int kSlots = 1024>
struct BlockSampler {
    static_assert((kSlots & (kSlots - 1)) == 0, "kSlots must be power of 2");
    static constexpr int kSlotMask = kSlots - 1;

    // Hot slot: packed key + data pointer, 16 bytes. 512 * 16 = 8KB, fits
    // comfortably in L1D. The shared_ptr (refcounting keep-alive) lives
    // in a cold parallel array, touched only on miss.
    struct HotSlot {
        uint64_t key = UINT64_MAX;
        const T* data = nullptr;
    };

    BlockPipeline& cache;
    int level;
    VolumeShape shape;
    HotSlot slots[kSlots];
    BlockPtr slotBlocks[kSlots];  // cold: refcount keep-alive
    uint64_t lastKey = UINT64_MAX;
    const T* data = nullptr;

    BlockSampler(BlockPipeline& c, int lvl)
        : cache(c), level(lvl), shape(c, lvl) {}

    VC_FORCE_INLINE static uint64_t packKey(int bz, int by, int bx) {
        return (uint64_t(uint32_t(bz)) << 42) | (uint64_t(uint32_t(by)) << 21) | uint64_t(uint32_t(bx));
    }

    VC_FORCE_INLINE int slotIndex(int bz, int by, int bx) const {
        uint32_t h = uint32_t(bz) * 73856093u ^ uint32_t(by) * 19349663u ^ uint32_t(bx) * 83492791u;
        return int(h) & kSlotMask;
    }

    // Fetch the block pointer for (bz, by, bx). Non-blocking: returns null
    // data if the block isn't resident in the cache.
    VC_FORCE_INLINE void updateBlock(int bz, int by, int bx) {
        tryUpdateBlockNonBlocking(bz, by, bx);
    }

    // Identical; kept for callers that want to be explicit about intent.
    VC_FORCE_INLINE void tryUpdateBlockNonBlocking(int bz, int by, int bx) {
        uint64_t key = packKey(bz, by, bx);
        if (key == lastKey) [[likely]] return;

        int idx = slotIndex(bz, by, bx);
        HotSlot& slot = slots[idx];
        if (slot.key == key) [[likely]] {
            data = slot.data;
            lastKey = key;
            return;
        }

        BlockKey bk{level, bz, by, bx};
        slotBlocks[idx] = cache.blockAt(bk);
        slot.data = slotBlocks[idx] ? reinterpret_cast<const T*>(slotBlocks[idx]->data) : nullptr;
        slot.key = key;
        data = slot.data;
        lastKey = key;
    }

    VC_FORCE_INLINE static size_t voxelOffset(int lz, int ly, int lx) {
        return size_t(lz) * kStrideZ + size_t(ly) * kStrideY + size_t(lx);
    }

    VC_FORCE_INLINE bool inBounds(float vz, float vy, float vx) const {
        return vz >= 0 && vy >= 0 && vx >= 0
            && vz < shape.szf && vy < shape.syf && vx < shape.sxf;
    }

    // Integer-coord fetch. Out-of-bounds and missing-block return 0.
    VC_FORCE_INLINE T sampleInt(int iz, int iy, int ix) {
        if (unsigned(iz) >= unsigned(shape.sz) ||
            unsigned(iy) >= unsigned(shape.sy) ||
            unsigned(ix) >= unsigned(shape.sx))
            return 0;

        int bz = iz >> kBlockShift;
        int by = iy >> kBlockShift;
        int bx = ix >> kBlockShift;
        updateBlock(bz, by, bx);
        if (!data) return 0;

        int lz = iz & kBlockMask;
        int ly = iy & kBlockMask;
        int lx = ix & kBlockMask;
        return data[voxelOffset(lz, ly, lx)];
    }

    // Non-blocking version — skips blocks not yet in RAM.
    VC_FORCE_INLINE bool sampleIntNB(int iz, int iy, int ix, T& out) {
        if (unsigned(iz) >= unsigned(shape.sz) ||
            unsigned(iy) >= unsigned(shape.sy) ||
            unsigned(ix) >= unsigned(shape.sx)) {
            out = 0; return true;
        }
        int bz = iz >> kBlockShift;
        int by = iy >> kBlockShift;
        int bx = ix >> kBlockShift;
        tryUpdateBlockNonBlocking(bz, by, bx);
        if (!data) return false;
        int lz = iz & kBlockMask, ly = iy & kBlockMask, lx = ix & kBlockMask;
        out = data[voxelOffset(lz, ly, lx)];
        return true;
    }

    VC_FORCE_INLINE T sampleNearest(float vz, float vy, float vx) {
        int iz = int(vz + 0.5f);
        int iy = int(vy + 0.5f);
        int ix = int(vx + 0.5f);
        if (iz >= shape.sz) iz = shape.sz - 1;
        if (iy >= shape.sy) iy = shape.sy - 1;
        if (ix >= shape.sx) ix = shape.sx - 1;
        return sampleInt(iz, iy, ix);
    }

    VC_FORCE_INLINE float sampleTrilinear(float vz, float vy, float vx) {
        int iz = int(vz);
        int iy = int(vy);
        int ix = int(vx);

        float c000 = sampleInt(iz,     iy,     ix);
        float c100 = sampleInt(iz + 1, iy,     ix);
        float c010 = sampleInt(iz,     iy + 1, ix);
        float c110 = sampleInt(iz + 1, iy + 1, ix);
        float c001 = sampleInt(iz,     iy,     ix + 1);
        float c101 = sampleInt(iz + 1, iy,     ix + 1);
        float c011 = sampleInt(iz,     iy + 1, ix + 1);
        float c111 = sampleInt(iz + 1, iy + 1, ix + 1);

        float fz = vz - float(iz);
        float fy = vy - float(iy);
        float fx = vx - float(ix);

        float c00 = std::fma(fx, c001 - c000, c000);
        float c01 = std::fma(fx, c011 - c010, c010);
        float c10 = std::fma(fx, c101 - c100, c100);
        float c11 = std::fma(fx, c111 - c110, c110);
        float c0  = std::fma(fy, c01 - c00, c00);
        float c1  = std::fma(fy, c11 - c10, c10);
        return std::fma(fz, c1 - c0, c0);
    }

    static VC_FORCE_INLINE float catmullRom(float t) {
        float at = std::abs(t);
        if (at < 1.0f) return 1.5f*at*at*at - 2.5f*at*at + 1.0f;
        if (at < 2.0f) return -0.5f*at*at*at + 2.5f*at*at - 4.0f*at + 2.0f;
        return 0.0f;
    }

    float sampleTricubic(float vz, float vy, float vx) {
        int iz = int(std::floor(vz));
        int iy = int(std::floor(vy));
        int ix = int(std::floor(vx));
        float fz = vz - float(iz), fy = vy - float(iy), fx = vx - float(ix);

        float result = 0.0f;
        for (int dz = -1; dz <= 2; dz++) {
            float wz = catmullRom(fz - float(dz));
            for (int dy = -1; dy <= 2; dy++) {
                float wy = catmullRom(fy - float(dy));
                float wzy = wz * wy;
                for (int dx = -1; dx <= 2; dx++) {
                    float wx = catmullRom(fx - float(dx));
                    result += wzy * wx * float(sampleInt(iz + dz, iy + dy, ix + dx));
                }
            }
        }
        return std::clamp(result, 0.0f, float(std::numeric_limits<T>::max()));
    }
};

// Append the chunk keys that enclose a world-space bbox to `out`. Callers
// that submit multiple regions/levels in one frame should accumulate into
// a single vector and call cache.fetchInteractive(keys) exactly once —
// every fetchInteractive call rebuilds the IOPool priority queue, so
// batching matters.
void appendChunksForRegion(BlockPipeline& cache, int level,
                           float minVx, float minVy, float minVz,
                           float maxVx, float maxVy, float maxVz,
                           std::vector<vc::cache::ChunkKey>& out) {
    auto cs = cache.chunkShape(level);
    if (cs[0] <= 0) return;
    auto ls = cache.levelShape(level);
    int chunksZ = (ls[0] + cs[0] - 1) / cs[0];
    int chunksY = (ls[1] + cs[1] - 1) / cs[1];
    int chunksX = (ls[2] + cs[2] - 1) / cs[2];

    int iMinX = std::max(0, int(std::floor(minVx)));
    int iMinY = std::max(0, int(std::floor(minVy)));
    int iMinZ = std::max(0, int(std::floor(minVz)));
    int iMaxX = std::min(ls[2] - 1, int(std::ceil(maxVx)));
    int iMaxY = std::min(ls[1] - 1, int(std::ceil(maxVy)));
    int iMaxZ = std::min(ls[0] - 1, int(std::ceil(maxVz)));
    if (iMinX > iMaxX || iMinY > iMaxY || iMinZ > iMaxZ) return;

    int cMinX = iMinX / cs[2], cMaxX = iMaxX / cs[2];
    int cMinY = iMinY / cs[1], cMaxY = iMaxY / cs[1];
    int cMinZ = iMinZ / cs[0], cMaxZ = iMaxZ / cs[0];

    for (int iz = cMinZ; iz <= cMaxZ && iz < chunksZ; iz++)
        for (int iy = cMinY; iy <= cMaxY && iy < chunksY; iy++)
            for (int ix = cMinX; ix <= cMaxX && ix < chunksX; ix++)
                out.push_back({level, iz, iy, ix});
}

// Convenience wrapper: single-region, single-level. Submits immediately.
void prefetchRegion(BlockPipeline& cache, int level,
                    float minVx, float minVy, float minVz,
                    float maxVx, float maxVy, float maxVz) {
    std::vector<vc::cache::ChunkKey> keys;
    appendChunksForRegion(cache, level, minVx, minVy, minVz,
                          maxVx, maxVy, maxVz, keys);
    if (!keys.empty()) cache.fetchInteractive(keys);
}

// prefetchCoordsRegion / prefetchPlaneRegion: inputs are already in
// LEVEL-space voxels (callers either pass already-scaled args or operate
// at a single level). For the world-space → multi-level adaptive path,
// see samplePixelsAdaptiveARGB32 which scales per-level before prefetching.
void prefetchCoordsRegion(BlockPipeline& cache, int level,
                          const cv::Mat_<cv::Vec3f>& coords) {
    float minVx = FLT_MAX, minVy = FLT_MAX, minVz = FLT_MAX;
    float maxVx = -FLT_MAX, maxVy = -FLT_MAX, maxVz = -FLT_MAX;
    for (int r = 0; r < coords.rows; r++) {
        const cv::Vec3f* row = coords.ptr<cv::Vec3f>(r);
        for (int c = 0; c < coords.cols; c++) {
            const auto& v = row[c];
            if (!isfinite_bitwise(v[0])) continue;
            minVx = std::min(minVx, v[0]); maxVx = std::max(maxVx, v[0]);
            minVy = std::min(minVy, v[1]); maxVy = std::max(maxVy, v[1]);
            minVz = std::min(minVz, v[2]); maxVz = std::max(maxVz, v[2]);
        }
    }
    if (maxVx >= minVx)
        prefetchRegion(cache, level, minVx, minVy, minVz, maxVx, maxVy, maxVz);
}

void prefetchPlaneRegion(BlockPipeline& cache, int level,
                         const cv::Vec3f& origin,
                         const cv::Vec3f& vx_step,
                         const cv::Vec3f& vy_step,
                         int w, int h) {
    cv::Vec3f corners[4] = {
        origin,
        origin + vx_step * float(w - 1),
        origin + vy_step * float(h - 1),
        origin + vx_step * float(w - 1) + vy_step * float(h - 1),
    };
    float minVx = corners[0][0], maxVx = corners[0][0];
    float minVy = corners[0][1], maxVy = corners[0][1];
    float minVz = corners[0][2], maxVz = corners[0][2];
    for (int i = 1; i < 4; i++) {
        minVx = std::min(minVx, corners[i][0]); maxVx = std::max(maxVx, corners[i][0]);
        minVy = std::min(minVy, corners[i][1]); maxVy = std::max(maxVy, corners[i][1]);
        minVz = std::min(minVz, corners[i][2]); maxVz = std::max(maxVz, corners[i][2]);
    }
    prefetchRegion(cache, level, minVx, minVy, minVz, maxVx, maxVy, maxVz);
}

enum class SampleMode : std::uint8_t { Nearest, Trilinear, Tricubic };

template<typename T, SampleMode Mode>
VC_FORCE_INLINE T sampleOne(BlockSampler<T>& s, float vz, float vy, float vx) {
    if constexpr (Mode == SampleMode::Nearest) {
        if (!s.inBounds(vz, vy, vx)) return 0;
        return s.sampleNearest(vz, vy, vx);
    } else if constexpr (Mode == SampleMode::Trilinear) {
        if (!s.inBounds(vz, vy, vx)) return 0;
        float v = s.sampleTrilinear(vz, vy, vx);
        if constexpr (std::is_same_v<T, uint16_t>) {
            if (v < 0.f) v = 0.f; if (v > 65535.f) v = 65535.f;
            return uint16_t(v + 0.5f);
        } else {
            if (v < 0.f) v = 0.f; if (v > 255.f) v = 255.f;
            return T(v);
        }
    } else {
        if (!s.inBounds(vz, vy, vx)) return 0;
        float v = s.sampleTricubic(vz, vy, vx);
        return T(v);
    }
}

template<typename T, SampleMode Mode>
void readVolumeImpl(cv::Mat_<T>& out, BlockPipeline& cache, int level,
                    const cv::Mat_<cv::Vec3f>& coords)
{
    prefetchCoordsRegion(cache, level, coords);

    const int h = coords.rows;
    const int w = coords.cols;
    #pragma omp parallel
    {
        BlockSampler<T> s(cache, level);
        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            const cv::Vec3f* row = coords.ptr<cv::Vec3f>(y);
            T* outRow = out.template ptr<T>(y);
            for (int x = 0; x < w; x++) {
                const auto& c = row[x];
                outRow[x] = sampleOne<T, Mode>(s, c[2], c[1], c[0]);
            }
        }
    }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

void readInterpolated3D(cv::Mat_<uint8_t>& out, BlockPipeline* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, bool nearest_neighbor) {
    if (nearest_neighbor)
        readVolumeImpl<uint8_t, SampleMode::Nearest>(out, *cache, level, coords);
    else
        readVolumeImpl<uint8_t, SampleMode::Trilinear>(out, *cache, level, coords);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, BlockPipeline* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, bool nearest_neighbor) {
    if (nearest_neighbor)
        readVolumeImpl<uint16_t, SampleMode::Nearest>(out, *cache, level, coords);
    else
        readVolumeImpl<uint16_t, SampleMode::Trilinear>(out, *cache, level, coords);
}

void readInterpolated3D(cv::Mat_<uint8_t>& out, BlockPipeline* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, vc::Sampling method) {
    switch (method) {
        case vc::Sampling::Nearest:
            readVolumeImpl<uint8_t, SampleMode::Nearest>(out, *cache, level, coords); break;
        case vc::Sampling::Tricubic:
            readVolumeImpl<uint8_t, SampleMode::Tricubic>(out, *cache, level, coords); break;
        default:
            readVolumeImpl<uint8_t, SampleMode::Trilinear>(out, *cache, level, coords); break;
    }
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, BlockPipeline* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, vc::Sampling method) {
    switch (method) {
        case vc::Sampling::Nearest:
            readVolumeImpl<uint16_t, SampleMode::Nearest>(out, *cache, level, coords); break;
        case vc::Sampling::Tricubic:
            readVolumeImpl<uint16_t, SampleMode::Tricubic>(out, *cache, level, coords); break;
        default:
            readVolumeImpl<uint16_t, SampleMode::Trilinear>(out, *cache, level, coords); break;
    }
}

namespace {

template<typename T>
void readArea3DImpl(Array3D<T>& out, const cv::Vec3i& offset,
                    BlockPipeline* cache, int level) {
    int d = int(out.shape()[0]), h = int(out.shape()[1]), w = int(out.shape()[2]);
    // Prefetch
    prefetchRegion(*cache, level,
                   float(offset[0]), float(offset[1]), float(offset[2]),
                   float(offset[0] + w - 1), float(offset[1] + h - 1), float(offset[2] + d - 1));

    #pragma omp parallel
    {
        BlockSampler<T> s(*cache, level);
        #pragma omp for schedule(dynamic, 4) collapse(2)
        for (int z = 0; z < d; z++) {
            for (int y = 0; y < h; y++) {
                int iz = offset[2] + z;
                int iy = offset[1] + y;
                for (int x = 0; x < w; x++) {
                    int ix = offset[0] + x;
                    out(z, y, x) = s.sampleInt(iz, iy, ix);
                }
            }
        }
    }
}

} // namespace

void readArea3D(Array3D<uint8_t>& out, const cv::Vec3i& offset,
                BlockPipeline* cache, int level) {
    readArea3DImpl(out, offset, cache, level);
}

void readArea3D(Array3D<uint16_t>& out, const cv::Vec3i& offset,
                BlockPipeline* cache, int level) {
    readArea3DImpl(out, offset, cache, level);
}

// ----------------------------------------------------------------------------
// Plane sampling (uint8 + ARGB32 variants)
// ----------------------------------------------------------------------------

namespace {

template<SampleMode Mode>
void samplePlaneImpl(cv::Mat_<uint8_t>& out, BlockPipeline& cache, int level,
                     const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                     int w, int h) {
    prefetchPlaneRegion(cache, level, origin, vx_step, vy_step, w, h);

    #pragma omp parallel
    {
        BlockSampler<uint8_t> s(cache, level);
        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            uint8_t* outRow = out.ptr<uint8_t>(y);
            cv::Vec3f base = origin + vy_step * float(y);
            for (int x = 0; x < w; x++) {
                cv::Vec3f c = base + vx_step * float(x);
                outRow[x] = sampleOne<uint8_t, Mode>(s, c[2], c[1], c[0]);
            }
        }
    }
}

} // namespace

void samplePlane(cv::Mat_<uint8_t>& out, BlockPipeline* cache, int level,
                 const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                 int w, int h, vc::Sampling method) {
    switch (method) {
        case vc::Sampling::Nearest:
            samplePlaneImpl<SampleMode::Nearest>(out, *cache, level, origin, vx_step, vy_step, w, h); break;
        case vc::Sampling::Tricubic:
            samplePlaneImpl<SampleMode::Tricubic>(out, *cache, level, origin, vx_step, vy_step, w, h); break;
        default:
            samplePlaneImpl<SampleMode::Trilinear>(out, *cache, level, origin, vx_step, vy_step, w, h); break;
    }
}

// ----------------------------------------------------------------------------
// Adaptive plane/coords sampling (per-pixel level fallback, non-blocking)
// ----------------------------------------------------------------------------

namespace {

// Attempt a non-blocking fetch at level L; returns true and writes LUT pixel
// if all needed blocks are present. Trilinear path uses 8 corners.
// Nearest sample with caller-guaranteed in-bounds coords. Caller must have
// verified 0 <= v{z,y,x} < shape.{sz,sy,sx}. Skips all bounds checks.
VC_FORCE_INLINE bool trySampleNearestUnchecked(BlockSampler<uint8_t>& s,
                                               float vz, float vy, float vx,
                                               uint8_t& out) {
    int iz = int(vz + 0.5f), iy = int(vy + 0.5f), ix = int(vx + 0.5f);
    if (iz >= s.shape.sz) iz = s.shape.sz - 1;
    if (iy >= s.shape.sy) iy = s.shape.sy - 1;
    if (ix >= s.shape.sx) ix = s.shape.sx - 1;
    int bz = iz >> kBlockShift, by = iy >> kBlockShift, bx = ix >> kBlockShift;
    s.tryUpdateBlockNonBlocking(bz, by, bx);
    if (!s.data) return false;
    int lz = iz & kBlockMask, ly = iy & kBlockMask, lx = ix & kBlockMask;
    out = s.data[size_t(lz) * kStrideZ + size_t(ly) * kStrideY + size_t(lx)];
    return true;
}

template<SampleMode Mode>
VC_FORCE_INLINE bool trySampleNB(BlockSampler<uint8_t>& s, float vz, float vy, float vx,
                                 uint8_t& out) {
    if constexpr (Mode == SampleMode::Nearest) {
        // Skip the float inBounds + per-axis clamp. sampleIntNB already
        // rejects OOB via unsigned compare. Only need to guard negatives
        // (truncation rounds small negatives to 0 incorrectly).
        if (vz < 0.f || vy < 0.f || vx < 0.f) { out = 0; return true; }
        int iz = int(vz + 0.5f), iy = int(vy + 0.5f), ix = int(vx + 0.5f);
        return s.sampleIntNB(iz, iy, ix, out);
    }
    if (!s.inBounds(vz, vy, vx)) { out = 0; return true; }
    {
        int iz = int(vz), iy = int(vy), ix = int(vx);
        float v000f, v100f, v010f, v110f, v001f, v101f, v011f, v111f;

        // Fast path: all 8 corners share one block (the common case — ~82%
        // of interior samples). Single block lookup, 8 direct reads.
        const int lz = iz & kBlockMask;
        const int ly = iy & kBlockMask;
        const int lx = ix & kBlockMask;
        if (lz != kBlockMask && ly != kBlockMask && lx != kBlockMask &&
            iz >= 0 && iy >= 0 && ix >= 0 &&
            iz + 1 < s.shape.sz && iy + 1 < s.shape.sy && ix + 1 < s.shape.sx) {
            int bz = iz >> kBlockShift, by = iy >> kBlockShift, bx = ix >> kBlockShift;
            s.tryUpdateBlockNonBlocking(bz, by, bx);
            if (!s.data) return false;
            const uint8_t* b = s.data + size_t(lz) * kStrideZ
                                      + size_t(ly) * kStrideY
                                      + size_t(lx);
            v000f = float(b[0]);
            v001f = float(b[1]);
            v010f = float(b[kStrideY]);
            v011f = float(b[kStrideY + 1]);
            v100f = float(b[kStrideZ]);
            v101f = float(b[kStrideZ + 1]);
            v110f = float(b[kStrideZ + kStrideY]);
            v111f = float(b[kStrideZ + kStrideY + 1]);
        } else {
            uint8_t v000, v100, v010, v110, v001, v101, v011, v111;
            if (!s.sampleIntNB(iz,     iy,     ix,     v000)) return false;
            if (!s.sampleIntNB(iz + 1, iy,     ix,     v100)) return false;
            if (!s.sampleIntNB(iz,     iy + 1, ix,     v010)) return false;
            if (!s.sampleIntNB(iz + 1, iy + 1, ix,     v110)) return false;
            if (!s.sampleIntNB(iz,     iy,     ix + 1, v001)) return false;
            if (!s.sampleIntNB(iz + 1, iy,     ix + 1, v101)) return false;
            if (!s.sampleIntNB(iz,     iy + 1, ix + 1, v011)) return false;
            if (!s.sampleIntNB(iz + 1, iy + 1, ix + 1, v111)) return false;
            v000f = float(v000); v100f = float(v100);
            v010f = float(v010); v110f = float(v110);
            v001f = float(v001); v101f = float(v101);
            v011f = float(v011); v111f = float(v111);
        }

        float fz = vz - float(iz), fy = vy - float(iy), fx = vx - float(ix);
        float c00 = std::fma(fx, v001f - v000f, v000f);
        float c01 = std::fma(fx, v011f - v010f, v010f);
        float c10 = std::fma(fx, v101f - v100f, v100f);
        float c11 = std::fma(fx, v111f - v110f, v110f);
        float c0  = std::fma(fy, c01 - c00, c00);
        float c1  = std::fma(fy, c11 - c10, c10);
        float v   = std::fma(fz, c1 - c0, c0);
        if (v < 0.f) v = 0.f; if (v > 255.f) v = 255.f;
        out = uint8_t(v);
        return true;
    }
}

template<SampleMode Mode>
void samplePixelsAdaptiveARGB32(uint32_t* outBuf, int outStride,
                                BlockPipeline& cache,
                                int desiredLevel, int numLevels,
                                const cv::Mat_<cv::Vec3f>* coords,  // may be nullptr
                                const cv::Vec3f* origin, const cv::Vec3f* vx_step, const cv::Vec3f* vy_step,
                                int w, int h, const uint32_t lut[256])
{
    // Pre-start fetches for all levels. Coords/origin are in world (level-0)
    // voxel space; scale to each level before enumerating chunks. Batch
    // everything into one fetchInteractive call — the IOPool's queue
    // rebuild is O(N) and we'd otherwise pay it once per level.
    auto levelScale = [](int lvl) { return (lvl > 0) ? 1.0f / float(1 << lvl) : 1.0f; };
    std::vector<vc::cache::ChunkKey> prefetchKeys;
    if (coords) {
        // Compute world-space bbox once, subsample like the composite path.
        float minVx = FLT_MAX, minVy = FLT_MAX, minVz = FLT_MAX;
        float maxVx = -FLT_MAX, maxVy = -FLT_MAX, maxVz = -FLT_MAX;
        const int stride = (coords->rows > 256) ? 8 : 1;
        for (int r = 0; r < coords->rows; r += stride) {
            const cv::Vec3f* row = coords->ptr<cv::Vec3f>(r);
            for (int c = 0; c < coords->cols; c += stride) {
                const auto& v = row[c];
                if (!isfinite_bitwise(v[0])) continue;
                minVx = std::min(minVx, v[0]); maxVx = std::max(maxVx, v[0]);
                minVy = std::min(minVy, v[1]); maxVy = std::max(maxVy, v[1]);
                minVz = std::min(minVz, v[2]); maxVz = std::max(maxVz, v[2]);
            }
        }
        if (maxVx >= minVx) {
            for (int lvl = desiredLevel; lvl < numLevels; lvl++) {
                float s = levelScale(lvl);
                appendChunksForRegion(cache, lvl,
                    minVx*s, minVy*s, minVz*s, maxVx*s, maxVy*s, maxVz*s,
                    prefetchKeys);
            }
        }
    } else {
        // Plane bbox from corners, compute once then scale per level.
        cv::Vec3f p0 = *origin;
        cv::Vec3f p1 = *origin + (*vx_step) * float(w-1) + (*vy_step) * float(h-1);
        float minVx = std::min(p0[0], p1[0]), maxVx = std::max(p0[0], p1[0]);
        float minVy = std::min(p0[1], p1[1]), maxVy = std::max(p0[1], p1[1]);
        float minVz = std::min(p0[2], p1[2]), maxVz = std::max(p0[2], p1[2]);
        for (int lvl = desiredLevel; lvl < numLevels; lvl++) {
            float s = levelScale(lvl);
            appendChunksForRegion(cache, lvl,
                minVx*s, minVy*s, minVz*s, maxVx*s, maxVy*s, maxVz*s,
                prefetchKeys);
        }
    }
    if (!prefetchKeys.empty()) cache.fetchInteractive(prefetchKeys);

    float scales[32] = {};
    const int nSamplersTotal = numLevels - desiredLevel;
    for (int i = 0; i < nSamplersTotal && i < 32; i++)
        scales[i] = levelScale(desiredLevel + i);

    #pragma omp parallel
    {
        const int nSamplers = numLevels - desiredLevel;
        std::array<std::optional<BlockSampler<uint8_t>>, 32> samplers;
        if (nSamplers > 0) samplers[0].emplace(cache, desiredLevel);
        auto sampler = [&](int i) -> BlockSampler<uint8_t>& {
            if (!samplers[i].has_value())
                samplers[i].emplace(cache, desiredLevel + i);
            return *samplers[i];
        };

        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            uint32_t* outRow = outBuf + size_t(y) * size_t(outStride);
            for (int x = 0; x < w; x++) {
                cv::Vec3f c;
                if (coords) c = (*coords)(y, x);
                else        c = *origin + *vx_step * float(x) + *vy_step * float(y);

                uint8_t pix = 0;
                bool got = false;
                for (int i = 0; i < nSamplers; i++) {
                    float scale = scales[i];
                    float vx = c[0] * scale, vy = c[1] * scale, vz = c[2] * scale;
                    if (trySampleNB<Mode>(sampler(i), vz, vy, vx, pix)) {
                        got = true;
                        break;
                    }
                }
                (void)got;
                outRow[x] = lut[pix];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Unified composite-capable adaptive sampler.
// numLayers=1 + nullptr normals = non-composite (same cost as the plain
// adaptive path, no per-layer overhead).
// ----------------------------------------------------------------------------

enum class AccumMode2 : std::uint8_t { Max, Min, Mean, LayerStorage };

static AccumMode2 accumModeFor(const std::string& m) {
    if (m == "max") return AccumMode2::Max;
    if (m == "min") return AccumMode2::Min;
    if (m == "median" || m == "alpha" || m == "minabs") return AccumMode2::LayerStorage;
    return AccumMode2::Mean;
}

template<SampleMode SMode, AccumMode2 AMode>
void sampleCompositeAdaptiveImpl(
    uint32_t* outBuf, int outStride,
    BlockPipeline& cache, int desiredLevel, int numLevels,
    const cv::Mat_<cv::Vec3f>* coords,
    const cv::Vec3f* origin, const cv::Vec3f* vx_step, const cv::Vec3f* vy_step,
    const cv::Mat_<cv::Vec3f>* normals,
    const cv::Vec3f* planeNormal,
    int numLayers, int zStart, float zStep,
    int w, int h,
    const std::string& compositeMethod,
    const uint32_t lut[256])
{
    auto levelScale = [](int lvl) { return (lvl > 0) ? 1.0f / float(1 << lvl) : 1.0f; };
    const float zLo = float(zStart) * zStep;
    const float zHi = float(zStart + numLayers - 1) * zStep;
    const float zMin = std::min(zLo, zHi), zMax = std::max(zLo, zHi);

    // Prefetch covered bbox across all fallback levels.
    if (coords) {
        float minVx=FLT_MAX, minVy=FLT_MAX, minVz=FLT_MAX;
        float maxVx=-FLT_MAX, maxVy=-FLT_MAX, maxVz=-FLT_MAX;
        // Subsample the coord grid for bbox computation: for a ~1920x1080
        // coords matrix, an 8x8 stride gives 30k samples instead of 2M
        // and still bounds the actual bbox within a voxel of truth
        // (prefetch is block-granular anyway).
        const int stride = (coords->rows > 256) ? 8 : 1;
        for (int r=0; r<coords->rows; r += stride) {
            const cv::Vec3f* row = coords->ptr<cv::Vec3f>(r);
            const cv::Vec3f* nrow = normals ? normals->ptr<cv::Vec3f>(r) : nullptr;
            for (int c=0; c<coords->cols; c += stride) {
                const auto& v = row[c];
                if (!isfinite_bitwise(v[0])) continue;
                cv::Vec3f n = nrow ? nrow[c] : cv::Vec3f(0,0,0);
                if (nrow && !isfinite_bitwise(n[0])) continue;
                float lox = v[0]+n[0]*zMin, hix = v[0]+n[0]*zMax;
                float loy = v[1]+n[1]*zMin, hiy = v[1]+n[1]*zMax;
                float loz = v[2]+n[2]*zMin, hiz = v[2]+n[2]*zMax;
                minVx=std::min(minVx,std::min(lox,hix)); maxVx=std::max(maxVx,std::max(lox,hix));
                minVy=std::min(minVy,std::min(loy,hiy)); maxVy=std::max(maxVy,std::max(loy,hiy));
                minVz=std::min(minVz,std::min(loz,hiz)); maxVz=std::max(maxVz,std::max(loz,hiz));
            }
        }
        if (maxVx >= minVx) {
            std::vector<vc::cache::ChunkKey> keys;
            for (int lvl=desiredLevel; lvl<numLevels; lvl++) {
                float s = levelScale(lvl);
                appendChunksForRegion(cache, lvl,
                    minVx*s, minVy*s, minVz*s, maxVx*s, maxVy*s, maxVz*s, keys);
            }
            if (!keys.empty()) cache.fetchInteractive(keys);
        }
    } else {
        cv::Vec3f p0 = *origin + (*planeNormal) * zMin;
        cv::Vec3f p1 = *origin + (*vx_step)*float(w-1) + (*vy_step)*float(h-1) + (*planeNormal)*zMax;
        float minVx=std::min(p0[0],p1[0]), maxVx=std::max(p0[0],p1[0]);
        float minVy=std::min(p0[1],p1[1]), maxVy=std::max(p0[1],p1[1]);
        float minVz=std::min(p0[2],p1[2]), maxVz=std::max(p0[2],p1[2]);
        std::vector<vc::cache::ChunkKey> keys;
        for (int lvl=desiredLevel; lvl<numLevels; lvl++) {
            float s = levelScale(lvl);
            appendChunksForRegion(cache, lvl,
                minVx*s, minVy*s, minVz*s, maxVx*s, maxVy*s, maxVz*s, keys);
        }
        if (!keys.empty()) cache.fetchInteractive(keys);
    }

    // Precompute per-level scale factor once (1.0 / 2^lvl). Hoists the
    // integer shift + int->float convert + fdiv out of the hot inner loop.
    float scales[32] = {};
    const int nSamplersTotal = numLevels - desiredLevel;
    for (int i = 0; i < nSamplersTotal && i < 32; i++) {
        int lvl = desiredLevel + i;
        scales[i] = (lvl > 0) ? 1.0f / float(1 << lvl) : 1.0f;
    }

    #pragma omp parallel
    {
        // Lazy per-level samplers: construct the level-0 sampler eagerly
        // (always used) and leave higher levels unconstructed until the
        // adaptive fallback actually needs them. Each sampler carries a
        // ~4KB hot slot cache; we don't want to pay that cost for levels
        // we never touch.
        const int nSamplers = numLevels - desiredLevel;
        std::array<std::optional<BlockSampler<uint8_t>>, 32> samplers;
        if (nSamplers > 0) samplers[0].emplace(cache, desiredLevel);
        auto sampler = [&](int i) -> BlockSampler<uint8_t>& {
            if (!samplers[i].has_value())
                samplers[i].emplace(cache, desiredLevel + i);
            return *samplers[i];
        };
        std::vector<float> layerVals;
        if constexpr (AMode == AccumMode2::LayerStorage) layerVals.resize(numLayers);

        #pragma omp for schedule(dynamic, 16)
        for (int y=0; y<h; y++) {
            uint32_t* outRow = outBuf + size_t(y) * size_t(outStride);
            const cv::Vec3f* crow = coords ? coords->ptr<cv::Vec3f>(y) : nullptr;
            const cv::Vec3f* nrow = normals ? normals->ptr<cv::Vec3f>(y) : nullptr;
            for (int x=0; x<w; x++) {
                cv::Vec3f base = crow ? crow[x] : (*origin + *vx_step*float(x) + *vy_step*float(y));
                if (!isfinite_bitwise(base[0])) { outRow[x] = lut[0]; continue; }
                cv::Vec3f nrm = nrow ? nrow[x] : (planeNormal ? *planeNormal : cv::Vec3f(0,0,0));
                if (nrow && !isfinite_bitwise(nrm[0])) { outRow[x] = lut[0]; continue; }

                float accum=0.f, mx=0.f, mn=255.f;
                int count=0;
                // Incremental per-layer offset along the normal direction:
                // one vector add per layer instead of 3 FMAs.
                const float dx = nrm[0] * zStep;
                const float dy = nrm[1] * zStep;
                const float dz = nrm[2] * zStep;
                float wx = base[0] + nrm[0] * float(zStart) * zStep;
                float wy = base[1] + nrm[1] * float(zStart) * zStep;
                float wz = base[2] + nrm[2] * float(zStart) * zStep;

                // Pixel-level bounds precheck at level 0: if the entire
                // z-line fits in bounds, skip all per-sample float compares.
                // Reserves 1-voxel margin for nearest rounding.
                const auto& sh0 = sampler(0).shape;
                const float endScale = scales[0];
                const float fwx = wx * endScale, fwy = wy * endScale, fwz = wz * endScale;
                const float tailFx = (wx + dx * float(numLayers - 1)) * endScale;
                const float tailFy = (wy + dy * float(numLayers - 1)) * endScale;
                const float tailFz = (wz + dz * float(numLayers - 1)) * endScale;
                const float minFx = std::min(fwx, tailFx);
                const float maxFx = std::max(fwx, tailFx);
                const float minFy = std::min(fwy, tailFy);
                const float maxFy = std::max(fwy, tailFy);
                const float minFz = std::min(fwz, tailFz);
                const float maxFz = std::max(fwz, tailFz);
                const bool fullyInBounds = SMode == SampleMode::Nearest
                    && minFx >= 0.5f && maxFx < float(sh0.sx) - 0.5f
                    && minFy >= 0.5f && maxFy < float(sh0.sy) - 0.5f
                    && minFz >= 0.5f && maxFz < float(sh0.sz) - 0.5f;

                for (int li=0; li<numLayers; li++) {
                    uint8_t v = 0;
                    bool got = false;
                    if (fullyInBounds) {
                        // Hot path: skip per-sample bounds check. Scale
                        // world coords into the desiredLevel sampler's
                        // space (scales[0] = 1/2^desiredLevel).
                        got = trySampleNearestUnchecked(*samplers[0],
                            wz * endScale, wy * endScale, wx * endScale, v);
                    }
                    if (!got) {
                        // Fallback: either we're near a boundary or the
                        // desired-level block isn't resident yet. Walk the
                        // fallback chain from finest to coarsest — adaptive
                        // sampling fills in from whichever level is ready.
                        for (int i=0; i<nSamplers; i++) {
                            float scl = scales[i];
                            if (trySampleNB<SMode>(sampler(i), wz*scl, wy*scl, wx*scl, v)) break;
                        }
                    }
                    if constexpr (AMode == AccumMode2::Max) { mx = std::max(mx, float(v)); }
                    else if constexpr (AMode == AccumMode2::Min) { mn = std::min(mn, float(v)); }
                    else if constexpr (AMode == AccumMode2::Mean) { accum += float(v); count++; }
                    else { layerVals[li] = float(v); }
                    wx += dx; wy += dy; wz += dz;
                }

                float val = 0.f;
                if constexpr (AMode == AccumMode2::Max) val = mx;
                else if constexpr (AMode == AccumMode2::Min) val = mn;
                else if constexpr (AMode == AccumMode2::Mean) val = count ? accum/float(count) : 0.f;
                else {
                    if (compositeMethod == "median") {
                        std::nth_element(layerVals.begin(), layerVals.begin()+numLayers/2, layerVals.end());
                        val = layerVals[numLayers/2];
                    } else if (compositeMethod == "minabs") {
                        float best = layerVals[0];
                        for (int i=1; i<numLayers; i++)
                            if (std::abs(layerVals[i]-127.5f) < std::abs(best-127.5f)) best = layerVals[i];
                        val = best;
                    } else {
                        float s=0.f; for (float v : layerVals) s += v;
                        val = numLayers>0 ? s/float(numLayers) : 0.f;
                    }
                }
                if (val < 0.f) val = 0.f; if (val > 255.f) val = 255.f;
                outRow[x] = lut[uint8_t(val)];
            }
        }
    }
}

template<SampleMode SMode>
void dispatchCompositeAdaptive(
    uint32_t* outBuf, int outStride,
    BlockPipeline& cache, int desiredLevel, int numLevels,
    const cv::Mat_<cv::Vec3f>* coords,
    const cv::Vec3f* origin, const cv::Vec3f* vx_step, const cv::Vec3f* vy_step,
    const cv::Mat_<cv::Vec3f>* normals,
    const cv::Vec3f* planeNormal,
    int numLayers, int zStart, float zStep,
    int w, int h,
    const std::string& method,
    const uint32_t lut[256])
{
    switch (accumModeFor(method)) {
        case AccumMode2::Max:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::Max>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut); break;
        case AccumMode2::Min:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::Min>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut); break;
        case AccumMode2::LayerStorage:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::LayerStorage>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut); break;
        default:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::Mean>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut); break;
    }
}

} // namespace

void sampleAdaptiveARGB32(
    uint32_t* outBuf, int outStride,
    vc::cache::BlockPipeline* cache,
    int desiredLevel, int numLevels,
    const cv::Mat_<cv::Vec3f>* coords,
    const cv::Vec3f* origin, const cv::Vec3f* vx_step, const cv::Vec3f* vy_step,
    const cv::Mat_<cv::Vec3f>* normals,
    const cv::Vec3f* planeNormal,
    int numLayers, int zStart, float zStep,
    int width, int height,
    const std::string& compositeMethod,
    const uint32_t lut[256],
    vc::Sampling method)
{
    if (numLayers <= 0) numLayers = 1;
    // Composite rendering forces Nearest: averaging N layers already
    // low-passes aliasing so per-voxel trilinear precision is wasted.
    // Pin the template instantiation to SampleMode::Nearest so the entire
    // inner loop compiles without any interp branch.
    const bool composite = numLayers > 1;
    if (composite || method == vc::Sampling::Nearest) {
        dispatchCompositeAdaptive<SampleMode::Nearest>(
            outBuf, outStride, *cache, desiredLevel, numLevels,
            coords, origin, vx_step, vy_step, normals, planeNormal,
            numLayers, zStart, zStep, width, height, compositeMethod, lut);
    } else {
        dispatchCompositeAdaptive<SampleMode::Trilinear>(
            outBuf, outStride, *cache, desiredLevel, numLevels,
            coords, origin, vx_step, vy_step, normals, planeNormal,
            numLayers, zStart, zStep, width, height, compositeMethod, lut);
    }
}

int samplePlaneAdaptiveARGB32(uint32_t* outBuf, int outStride,
                              BlockPipeline* cache,
                              int desiredLevel, int numLevels,
                              const cv::Vec3f& origin,
                              const cv::Vec3f& vx_step,
                              const cv::Vec3f& vy_step,
                              int w, int h,
                              const uint32_t lut[256],
                              vc::Sampling method)
{
    switch (method) {
        case vc::Sampling::Nearest:
            samplePixelsAdaptiveARGB32<SampleMode::Nearest>(
                outBuf, outStride, *cache, desiredLevel, numLevels,
                nullptr, &origin, &vx_step, &vy_step, w, h, lut);
            break;
        default:
            samplePixelsAdaptiveARGB32<SampleMode::Trilinear>(
                outBuf, outStride, *cache, desiredLevel, numLevels,
                nullptr, &origin, &vx_step, &vy_step, w, h, lut);
            break;
    }
    return desiredLevel;
}

void sampleCoordsAdaptiveARGB32(uint32_t* outBuf, int outStride,
                                BlockPipeline* cache,
                                int desiredLevel, int numLevels,
                                const cv::Mat_<cv::Vec3f>& coords,
                                const uint32_t lut[256],
                                vc::Sampling method)
{
    int w = coords.cols, h = coords.rows;
    switch (method) {
        case vc::Sampling::Nearest:
            samplePixelsAdaptiveARGB32<SampleMode::Nearest>(
                outBuf, outStride, *cache, desiredLevel, numLevels,
                &coords, nullptr, nullptr, nullptr, w, h, lut);
            break;
        default:
            samplePixelsAdaptiveARGB32<SampleMode::Trilinear>(
                outBuf, outStride, *cache, desiredLevel, numLevels,
                &coords, nullptr, nullptr, nullptr, w, h, lut);
            break;
    }
}

// ----------------------------------------------------------------------------
// Composite rendering
// ----------------------------------------------------------------------------

namespace {

enum class AccumMode : std::uint8_t { Max, Min, Mean, LayerStorage };

static bool needsLayerStorage(const std::string& m) {
    return m == "median" || m == "alpha" || m == "minabs";
}

template<typename T, SampleMode Mode>
void readCompositeFastImpl(
    cv::Mat_<uint8_t>& out,
    BlockPipeline& cache,
    int level,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params)
{
    const int h = baseCoords.rows, w = baseCoords.cols;
    const int numLayers = zEnd - zStart + 1;
    AccumMode mode = AccumMode::Mean;
    if (needsLayerStorage(params.method)) mode = AccumMode::LayerStorage;
    else if (params.method == "max") mode = AccumMode::Max;
    else if (params.method == "min") mode = AccumMode::Min;

    // Prefetch all layers' coverage.
    // Approximation: bbox of baseCoords ± numLayers * zStep in each normal direction.
    prefetchCoordsRegion(cache, level, baseCoords);

    #pragma omp parallel
    {
        BlockSampler<T> s(cache, level);
        std::vector<float> layerVals(numLayers);
        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            const cv::Vec3f* bRow = baseCoords.ptr<cv::Vec3f>(y);
            const cv::Vec3f* nRow = normals.ptr<cv::Vec3f>(y);
            uint8_t* outRow = out.ptr<uint8_t>(y);
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& base = bRow[x];
                const cv::Vec3f& n = nRow[x];
                if (!isfinite_bitwise(base[0]) || !isfinite_bitwise(n[0])) continue;

                float accum = 0.f, mx = 0.f, mn = float(std::numeric_limits<T>::max());
                int count = 0;
                for (int li = 0; li < numLayers; li++) {
                    float z = float(zStart + li) * zStep;
                    float vx = base[0] + n[0] * z;
                    float vy = base[1] + n[1] * z;
                    float vz = base[2] + n[2] * z;
                    float v = float(sampleOne<T, Mode>(s, vz, vy, vx));
                    switch (mode) {
                        case AccumMode::Max: mx = std::max(mx, v); break;
                        case AccumMode::Min: mn = std::min(mn, v); break;
                        case AccumMode::Mean: accum += v; count++; break;
                        case AccumMode::LayerStorage: layerVals[li] = v; break;
                    }
                }

                float val = 0.f;
                switch (mode) {
                    case AccumMode::Max:  val = mx; break;
                    case AccumMode::Min:  val = mn; break;
                    case AccumMode::Mean: val = count > 0 ? accum / float(count) : 0.f; break;
                    case AccumMode::LayerStorage: {
                        if (params.method == "median") {
                            std::nth_element(layerVals.begin(),
                                             layerVals.begin() + numLayers / 2,
                                             layerVals.end());
                            val = layerVals[numLayers / 2];
                        } else if (params.method == "minabs") {
                            float best = layerVals[0];
                            for (int i = 1; i < numLayers; i++)
                                if (std::abs(layerVals[i] - 127.5f) < std::abs(best - 127.5f))
                                    best = layerVals[i];
                            val = best;
                        } else {
                            val = count > 0 ? accum / float(count) : 0.f;
                        }
                        break;
                    }
                }
                if (val < 0.f) val = 0.f; if (val > 255.f) val = 255.f;
                outRow[x] = uint8_t(val);
            }
        }
    }
}

} // namespace

void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    vc::Sampling method)
{
    switch (method) {
        case vc::Sampling::Tricubic:
            readCompositeFastImpl<uint8_t, SampleMode::Tricubic>(out, *cache, level,
                baseCoords, normals, zStep, zStart, zEnd, params); break;
        case vc::Sampling::Trilinear:
            readCompositeFastImpl<uint8_t, SampleMode::Trilinear>(out, *cache, level,
                baseCoords, normals, zStep, zStart, zEnd, params); break;
        default:
            readCompositeFastImpl<uint8_t, SampleMode::Nearest>(out, *cache, level,
                baseCoords, normals, zStep, zStart, zEnd, params); break;
    }
}

void samplePlaneCompositeARGB32(
    uint32_t* outBuf, int outStride,
    BlockPipeline* cache, int level,
    const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
    const cv::Vec3f& normal, float zStep, int zStart, int numLayers,
    int w, int h,
    const std::string& compositeMethod,
    const uint32_t lut[256])
{
    AccumMode mode = AccumMode::Mean;
    if (needsLayerStorage(compositeMethod)) mode = AccumMode::LayerStorage;
    else if (compositeMethod == "max") mode = AccumMode::Max;
    else if (compositeMethod == "min") mode = AccumMode::Min;

    // Prefetch outer bbox including all layers.
    cv::Vec3f layerOffsetMin = normal * (float(zStart) * zStep);
    cv::Vec3f layerOffsetMax = normal * (float(zStart + numLayers - 1) * zStep);
    cv::Vec3f p0 = origin + layerOffsetMin;
    cv::Vec3f p1 = origin + vx_step * float(w - 1) + vy_step * float(h - 1) + layerOffsetMax;
    float minVx = std::min(p0[0], p1[0]), maxVx = std::max(p0[0], p1[0]);
    float minVy = std::min(p0[1], p1[1]), maxVy = std::max(p0[1], p1[1]);
    float minVz = std::min(p0[2], p1[2]), maxVz = std::max(p0[2], p1[2]);
    prefetchRegion(*cache, level, minVx, minVy, minVz, maxVx, maxVy, maxVz);

    #pragma omp parallel
    {
        BlockSampler<uint8_t> s(*cache, level);
        std::vector<float> layerVals(numLayers);
        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            uint32_t* outRow = outBuf + size_t(y) * size_t(outStride);
            for (int x = 0; x < w; x++) {
                cv::Vec3f base = origin + vx_step * float(x) + vy_step * float(y);

                float accum = 0.f, mx = 0.f, mn = 255.f;
                int count = 0;
                for (int li = 0; li < numLayers; li++) {
                    float z = float(zStart + li) * zStep;
                    float vx = base[0] + normal[0] * z;
                    float vy = base[1] + normal[1] * z;
                    float vz = base[2] + normal[2] * z;
                    float v = float(sampleOne<uint8_t, SampleMode::Nearest>(s, vz, vy, vx));
                    switch (mode) {
                        case AccumMode::Max: mx = std::max(mx, v); break;
                        case AccumMode::Min: mn = std::min(mn, v); break;
                        case AccumMode::Mean: accum += v; count++; break;
                        case AccumMode::LayerStorage: layerVals[li] = v; break;
                    }
                }

                float val = 0.f;
                switch (mode) {
                    case AccumMode::Max:  val = mx; break;
                    case AccumMode::Min:  val = mn; break;
                    case AccumMode::Mean: val = count > 0 ? accum / float(count) : 0.f; break;
                    case AccumMode::LayerStorage: {
                        if (compositeMethod == "median") {
                            std::nth_element(layerVals.begin(),
                                             layerVals.begin() + numLayers / 2,
                                             layerVals.end());
                            val = layerVals[numLayers / 2];
                        } else if (compositeMethod == "minabs") {
                            float best = layerVals[0];
                            for (int i = 1; i < numLayers; i++)
                                if (std::abs(layerVals[i] - 127.5f) < std::abs(best - 127.5f))
                                    best = layerVals[i];
                            val = best;
                        } else {
                            val = count > 0 ? accum / float(count) : 0.f;
                        }
                        break;
                    }
                }
                if (val < 0.f) val = 0.f; if (val > 255.f) val = 255.f;
                outRow[x] = lut[uint8_t(val)];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Multi-slice readers
// ----------------------------------------------------------------------------

namespace {

template<typename T>
void readMultiSliceImpl(
    std::vector<cv::Mat_<T>>& out,
    BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    const int h = basePoints.rows, w = basePoints.cols;
    const int nSlices = int(offsets.size());
    out.resize(nSlices);
    for (int i = 0; i < nSlices; i++)
        out[i].create(h, w);

    // Prefetch union bbox.
    float minOff = *std::min_element(offsets.begin(), offsets.end());
    float maxOff = *std::max_element(offsets.begin(), offsets.end());
    float minVx = FLT_MAX, minVy = FLT_MAX, minVz = FLT_MAX;
    float maxVx = -FLT_MAX, maxVy = -FLT_MAX, maxVz = -FLT_MAX;
    for (int y = 0; y < h; y++) {
        const cv::Vec3f* bRow = basePoints.ptr<cv::Vec3f>(y);
        const cv::Vec3f* sRow = stepDirs.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; x++) {
            if (!isfinite_bitwise(bRow[x][0])) continue;
            cv::Vec3f lo = bRow[x] + sRow[x] * minOff;
            cv::Vec3f hi = bRow[x] + sRow[x] * maxOff;
            minVx = std::min(minVx, std::min(lo[0], hi[0]));
            minVy = std::min(minVy, std::min(lo[1], hi[1]));
            minVz = std::min(minVz, std::min(lo[2], hi[2]));
            maxVx = std::max(maxVx, std::max(lo[0], hi[0]));
            maxVy = std::max(maxVy, std::max(lo[1], hi[1]));
            maxVz = std::max(maxVz, std::max(lo[2], hi[2]));
        }
    }
    if (maxVx >= minVx)
        prefetchRegion(*cache, level, minVx, minVy, minVz, maxVx, maxVy, maxVz);

    #pragma omp parallel
    {
        BlockSampler<T> s(*cache, level);
        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            const cv::Vec3f* bRow = basePoints.ptr<cv::Vec3f>(y);
            const cv::Vec3f* sRow = stepDirs.ptr<cv::Vec3f>(y);
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& b = bRow[x];
                const cv::Vec3f& d = sRow[x];
                for (int i = 0; i < nSlices; i++) {
                    float off = offsets[i];
                    float vx = b[0] + d[0] * off;
                    float vy = b[1] + d[1] * off;
                    float vz = b[2] + d[2] * off;
                    T val = 0;
                    if (s.inBounds(vz, vy, vx)) {
                        float v = s.sampleTrilinear(vz, vy, vx);
                        if (v < 0.f) v = 0.f;
                        float maxV = float(std::numeric_limits<T>::max());
                        if (v > maxV) v = maxV;
                        val = T(v + (std::is_same_v<T, uint16_t> ? 0.5f : 0.f));
                    }
                    out[i].template at<T>(y, x) = val;
                }
            }
        }
    }
}

template<typename T>
void sampleTileSlicesImpl(
    std::vector<cv::Mat_<T>>& out,
    BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    const int h = basePoints.rows, w = basePoints.cols;
    const int nSlices = int(offsets.size());
    out.resize(nSlices);
    for (int i = 0; i < nSlices; i++) out[i].create(h, w);

    BlockSampler<T> s(*cache, level);
    for (int y = 0; y < h; y++) {
        const cv::Vec3f* bRow = basePoints.ptr<cv::Vec3f>(y);
        const cv::Vec3f* sRow = stepDirs.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; x++) {
            const cv::Vec3f& b = bRow[x];
            const cv::Vec3f& d = sRow[x];
            for (int i = 0; i < nSlices; i++) {
                float off = offsets[i];
                float vx = b[0] + d[0] * off;
                float vy = b[1] + d[1] * off;
                float vz = b[2] + d[2] * off;
                T val = 0;
                if (s.inBounds(vz, vy, vx)) {
                    float v = s.sampleTrilinear(vz, vy, vx);
                    if (v < 0.f) v = 0.f;
                    float maxV = float(std::numeric_limits<T>::max());
                    if (v > maxV) v = maxV;
                    val = T(v + (std::is_same_v<T, uint16_t> ? 0.5f : 0.f));
                }
                out[i].template at<T>(y, x) = val;
            }
        }
    }
}

} // namespace

void readMultiSlice(std::vector<cv::Mat_<uint8_t>>& out, BlockPipeline* cache, int level,
                    const cv::Mat_<cv::Vec3f>& basePoints, const cv::Mat_<cv::Vec3f>& stepDirs,
                    const std::vector<float>& offsets) {
    readMultiSliceImpl(out, cache, level, basePoints, stepDirs, offsets);
}

void readMultiSlice(std::vector<cv::Mat_<uint16_t>>& out, BlockPipeline* cache, int level,
                    const cv::Mat_<cv::Vec3f>& basePoints, const cv::Mat_<cv::Vec3f>& stepDirs,
                    const std::vector<float>& offsets) {
    readMultiSliceImpl(out, cache, level, basePoints, stepDirs, offsets);
}

void sampleTileSlices(std::vector<cv::Mat_<uint8_t>>& out, BlockPipeline* cache, int level,
                      const cv::Mat_<cv::Vec3f>& basePoints, const cv::Mat_<cv::Vec3f>& stepDirs,
                      const std::vector<float>& offsets) {
    sampleTileSlicesImpl(out, cache, level, basePoints, stepDirs, offsets);
}

void sampleTileSlices(std::vector<cv::Mat_<uint16_t>>& out, BlockPipeline* cache, int level,
                      const cv::Mat_<cv::Vec3f>& basePoints, const cv::Mat_<cv::Vec3f>& stepDirs,
                      const std::vector<float>& offsets) {
    sampleTileSlicesImpl(out, cache, level, basePoints, stepDirs, offsets);
}
