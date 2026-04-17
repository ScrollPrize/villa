#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/cache/BlockCache.hpp"
#include "vc/core/cache/BlockPipeline.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>
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
using vc::cache::kMaxLevels;

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
template<typename T, int kSlots = 4096>
struct BlockSampler {
    static_assert((kSlots & (kSlots - 1)) == 0, "kSlots must be power of 2");
    static constexpr int kSlotMask = kSlots - 1;

    // Hot slot: packed key + data pointer, 16 bytes. 4096 slots × 16B = 64KB.
    // Fits in L2 with room to spare; L1D (typically 32-64KB) takes collisions.
    // Raised from 1024 because the 12-thread render's per-thread working set
    // is 1400-2700 unique blocks per frame — at 1024 slots the direct-mapped
    // cache collided hard, and every collision miss fell through to blockAt,
    // taking a BlockCache shard shared_lock. The shard shared_lock release
    // (CAS-rel on its reader counter) was ~9% of CPU. 4096 slots hold the
    // full working set so most lookups stay in the sampler's private cache.
    // The shared_ptr (refcounting keep-alive) lives in a cold parallel array,
    // touched only on miss.
    struct HotSlot {
        uint64_t key = UINT64_MAX;
        const T* data = nullptr;
    };

    BlockPipeline& cache;
    int level;
    VolumeShape shape;
    HotSlot slots[kSlots];
    BlockPtr slotBlocks[kSlots];  // cold: refcount keep-alive
    // Last-block (bz,by,bx) cache as separate ints. Most pixels in a tile
    // sample the same block, so comparing three ints lets us skip packKey's
    // 3 shifts + 2 ORs on every same-block call. lastBz=INT_MIN seeds a
    // guaranteed miss on the first access.
    int lastBz = std::numeric_limits<int>::min();
    int lastBy = 0;
    int lastBx = 0;
    uint64_t lastKey = UINT64_MAX;
    const T* data = nullptr;

    BlockSampler(BlockPipeline& c, int lvl)
        : cache(c), level(lvl), shape(c, lvl) {}

    VC_FORCE_INLINE static uint64_t packKey(int bz, int by, int bx) {
        return (uint64_t(uint32_t(bz)) << 42) | (uint64_t(uint32_t(by)) << 21) | uint64_t(uint32_t(bx));
    }

    // Cheap finalizer over the already-packed key. Reuses packKey's output
    // instead of re-hashing the three int coords with three multiplies;
    // the xor-fold mixes all three 21/22-bit axis fields into the low bits.
    VC_FORCE_INLINE int slotIndexFromKey(uint64_t key) const {
        uint64_t h = key ^ (key >> 21) ^ (key >> 42);
        h ^= h >> 15;
        h *= 0x2545F4914F6CDD1DULL;
        return int(h) & kSlotMask;
    }

    VC_FORCE_INLINE int slotIndex(int bz, int by, int bx) const {
        return slotIndexFromKey(packKey(bz, by, bx));
    }

    // Fetch the block pointer for (bz, by, bx). Non-blocking: returns null
    // data if the block isn't resident in the cache.
    VC_FORCE_INLINE void updateBlock(int bz, int by, int bx) {
        tryUpdateBlockNonBlocking(bz, by, bx);
    }

    // Identical; kept for callers that want to be explicit about intent.
    VC_FORCE_INLINE void tryUpdateBlockNonBlocking(int bz, int by, int bx) {
        // Int-level fast path: consecutive samples within a tile almost
        // always land in the same 16³ block. Compare three ints and skip
        // packKey + slot hash entirely on a match — saves ~5 cycles/sample
        // on the ~80% of samples that hit the same block as the last.
        if (bz == lastBz && by == lastBy && bx == lastBx) [[likely]] return;

        const uint64_t key = packKey(bz, by, bx);
        lastBz = bz; lastBy = by; lastBx = bx;
        lastKey = key;

        int idx = slotIndexFromKey(key);
        HotSlot& slot = slots[idx];
        if (slot.key == key) [[likely]] {
            data = slot.data;
            return;
        }

        BlockKey bk{level, bz, by, bx};
        slotBlocks[idx] = cache.blockAt(bk);
        slot.data = slotBlocks[idx] ? reinterpret_cast<const T*>(slotBlocks[idx]->data) : nullptr;
        slot.key = key;
        data = slot.data;
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

// Surface-aware chunk enumeration. Bbox enumeration is correct for planes
// (the plane really does span its bbox) but disastrous for curved surfaces:
// a flattened scroll surface might span a 4000x4000x500 voxel bbox while
// actually crossing only a few hundred chunks. The other thousands of
// bbox-interior chunks are off-surface and would trigger pointless S3
// fetches. Walk per-pixel coords, map each to its chunk, dedup.
// Sub-sample to keep the dedup set cheap.
void appendChunksForCoordsSurface(BlockPipeline& cache, int level,
                                  const cv::Mat_<cv::Vec3f>& coords,
                                  std::vector<vc::cache::ChunkKey>& out) {
    auto cs = cache.chunkShape(level);
    if (cs[0] <= 0 || cs[1] <= 0 || cs[2] <= 0) return;
    auto ls = cache.levelShape(level);
    const int chunksZ = (ls[0] + cs[0] - 1) / cs[0];
    const int chunksY = (ls[1] + cs[1] - 1) / cs[1];
    const int chunksX = (ls[2] + cs[2] - 1) / cs[2];

    const float scale = (level > 0) ? 1.0f / float(1 << level) : 1.0f;
    // Sub-sample stride for the prefetch walk. Missing a chunk here is not
    // a correctness bug — the render sampler faults it in lazily — just a
    // prefetch miss, so stride=8 trades a rare lazy fetch on steeply
    // oblique surfaces for ~64x less work on large surfaces.
    const int stride = (coords.rows > 256) ? 8 : 1;

    // Thread-local dedup set to avoid heap churn across frames. Cleared on
    // entry; capacity grows monotonically with the largest surface ever seen
    // on this thread, which is exactly what we want (amortizes allocations).
    thread_local std::unordered_set<vc::cache::ChunkKey, vc::cache::ChunkKeyHash> seen;
    seen.clear();
    const size_t sampleEstimate =
        size_t((coords.rows + stride - 1) / stride) *
        size_t((coords.cols + stride - 1) / stride);
    // Rough: spatial coherence keeps unique-chunk count ≪ sample count.
    seen.reserve(std::max<size_t>(512, sampleEstimate / 8));

    for (int r = 0; r < coords.rows; r += stride) {
        const cv::Vec3f* row = coords.ptr<cv::Vec3f>(r);
        for (int c = 0; c < coords.cols; c += stride) {
            const cv::Vec3f& v = row[c];
            // Skip NaN / zero-sentinel pixels (off-surface).
            if (!isfinite_bitwise(v[0])) continue;
            if (v[0] == 0.f && v[1] == 0.f && v[2] == 0.f) continue;
            const int ix_ = int(std::floor(v[0] * scale));
            const int iy_ = int(std::floor(v[1] * scale));
            const int iz_ = int(std::floor(v[2] * scale));
            if (ix_ < 0 || iy_ < 0 || iz_ < 0) continue;
            const int cx = ix_ / cs[2];
            const int cy = iy_ / cs[1];
            const int cz = iz_ / cs[0];
            if (cx >= chunksX || cy >= chunksY || cz >= chunksZ) continue;
            vc::cache::ChunkKey k{level, cz, cy, cx};
            if (seen.insert(k).second) out.push_back(k);
        }
    }
}

// Sort prefetch keys by 3D distance from the viewport-center voxel (in
// level-0 space). The IOPool preserves submission order when rebuilding
// the front of its priority queue, so center-most chunks land at the head
// of the download/load queue and stream in first. `centerL0` is in level-0
// voxel coordinates.
void sortKeysByCenterDistance(std::vector<vc::cache::ChunkKey>& keys,
                              const std::array<int, 3>* chunkShapesByLevel,
                              int numLevels,
                              const cv::Vec3f& centerL0) {
    if (keys.size() < 2) return;
    auto sqDist = [&](const vc::cache::ChunkKey& k) -> float {
        if (k.level < 0 || k.level >= numLevels) return std::numeric_limits<float>::max();
        const auto& cs = chunkShapesByLevel[k.level];
        if (cs[0] <= 0) return std::numeric_limits<float>::max();
        const float scale = float(1 << k.level);
        // Chunk center in level-0 voxel space.
        float cx = (float(k.ix) + 0.5f) * float(cs[2]) * scale;
        float cy = (float(k.iy) + 0.5f) * float(cs[1]) * scale;
        float cz = (float(k.iz) + 0.5f) * float(cs[0]) * scale;
        float dx = cx - centerL0[0];
        float dy = cy - centerL0[1];
        float dz = cz - centerL0[2];
        return dx*dx + dy*dy + dz*dz;
    };
    std::sort(keys.begin(), keys.end(),
        [&](const vc::cache::ChunkKey& a, const vc::cache::ChunkKey& b) {
            return sqDist(a) < sqDist(b);
        });
}

// Convenience wrapper: single-region, single-level. Submits immediately.
void prefetchRegion(BlockPipeline& cache, int level,
                    float minVx, float minVy, float minVz,
                    float maxVx, float maxVy, float maxVz) {
    thread_local std::vector<vc::cache::ChunkKey> keys;
    keys.clear();
    appendChunksForRegion(cache, level, minVx, minVy, minVz,
                          maxVx, maxVy, maxVz, keys);
    if (!keys.empty()) cache.fetchInteractive(keys, level);
}

// prefetchCoordsRegion / prefetchPlaneRegion: inputs are already in
// LEVEL-space voxels (callers either pass already-scaled args or operate
// at a single level). For the world-space → multi-level adaptive path,
// see sampleCompositeAdaptiveImpl / sampleSingleLayerAdaptiveImpl which
// scale per-level before prefetching.
void prefetchCoordsRegion(BlockPipeline& cache, int level,
                          const cv::Mat_<cv::Vec3f>& coords) {
    // Unconditional min/max reductions so the vectorizer can fold the
    // inner loop: invalid points are sentinels (usually NaN), and
    // fminnm/fmaxnm on NEON treat NaN as "not a number, propagate the
    // other operand" — same end result as a skip but no branch.
    float minVx = FLT_MAX, minVy = FLT_MAX, minVz = FLT_MAX;
    float maxVx = -FLT_MAX, maxVy = -FLT_MAX, maxVz = -FLT_MAX;
    for (int r = 0; r < coords.rows; r++) {
        const cv::Vec3f* row = coords.ptr<cv::Vec3f>(r);
        for (int c = 0; c < coords.cols; c++) {
            const auto& v = row[c];
            minVx = std::fmin(minVx, v[0]); maxVx = std::fmax(maxVx, v[0]);
            minVy = std::fmin(minVy, v[1]); maxVy = std::fmax(maxVy, v[1]);
            minVz = std::fmin(minVz, v[2]); maxVz = std::fmax(maxVz, v[2]);
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

// Manual tile parallelism: persistent worker pool that `runRenderThreads`
// dispatches work to. Each render call fans body(tid) across N-1 workers
// and body(0) on the calling thread, then blocks until all finish.
//
// We cannot spawn fresh std::jthreads per render call — at interactive
// rates (60+ fps), pthread_create+join overhead regresses the 12-core
// speedup from ~4x down to ~1.1x. OpenMP's implicit pool hid this cost
// before; the custom pool reclaims it without dragging the libgomp
// runtime back into the hot path.
inline int renderThreadCount() {
    static const int n = []() {
        int hw = int(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 4;
        // 16 is empirical: 129-layer composite saturates L1/L2 before we
        // hit contention on the shared block cache; more threads waste
        // schedule slots without shortening the critical path.
        return hw < 16 ? hw : 16;
    }();
    return n;
}

class RenderThreadPool {
public:
    static RenderThreadPool& instance() {
        static RenderThreadPool p;
        return p;
    }

    template<typename Body>
    void run(Body&& body) {
        const int nWorkers = int(workers_.size());
        if (nWorkers == 0) { body(0); return; }
        // Serialize concurrent callers — the render path is the only caller
        // today, but guarding keeps the pool composable if another hot path
        // ever shares it.
        std::lock_guard<std::mutex> callLock(callMutex_);
        body_ = std::function<void(int)>(std::forward<Body>(body));
        for (int i = 0; i < nWorkers; ++i) startSem_.release();
        body_(0);
        for (int i = 0; i < nWorkers; ++i) doneSem_.acquire();
    }

    int workerThreads() const { return int(workers_.size()); }

private:
    RenderThreadPool() {
        const int nT = renderThreadCount();
        const int nWorkers = nT > 1 ? nT - 1 : 0;
        workers_.reserve(size_t(nWorkers));
        for (int i = 1; i <= nWorkers; ++i) {
            workers_.emplace_back([this, i]() {
                while (true) {
                    startSem_.acquire();
                    if (shutdown_.load(std::memory_order_acquire)) return;
                    body_(i);
                    doneSem_.release();
                }
            });
        }
    }

    ~RenderThreadPool() {
        shutdown_.store(true, std::memory_order_release);
        for (size_t i = 0; i < workers_.size(); ++i) startSem_.release();
        // jthread destructor joins automatically.
    }

    RenderThreadPool(const RenderThreadPool&) = delete;
    RenderThreadPool& operator=(const RenderThreadPool&) = delete;

    std::vector<std::jthread> workers_;
    std::function<void(int)> body_;
    std::counting_semaphore<> startSem_{0};
    std::counting_semaphore<> doneSem_{0};
    std::atomic<bool> shutdown_{false};
    std::mutex callMutex_;
};

template<typename Body>
inline void runRenderThreads(Body&& body) {
    RenderThreadPool::instance().run(std::forward<Body>(body));
}

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
            // Slow path: at least one axis's high corner crosses a block
            // boundary. In the common case only ONE axis crosses (~80% of
            // slow-path samples); specialize to 2 block lookups instead of
            // 8 independent sampleIntNB calls.
            const bool zCross = (lz == kBlockMask);
            const bool yCross = (ly == kBlockMask);
            const bool xCross = (lx == kBlockMask);
            const int nCross = int(zCross) + int(yCross) + int(xCross);

            // Full-bounds guard for the specialized paths. Near-image-edge
            // samples fall through to the generic 8-call path so we don't
            // duplicate its bounds handling here.
            const bool edgeInBounds =
                iz >= 0 && iy >= 0 && ix >= 0
                && iz + 1 < s.shape.sz && iy + 1 < s.shape.sy && ix + 1 < s.shape.sx;

            auto loadFromBlock = [](const uint8_t* b, int tz, int ty, int tx) {
                return float(b[size_t(tz) * kStrideZ + size_t(ty) * kStrideY + size_t(tx)]);
            };

            if (nCross == 1 && edgeInBounds) {
                const int bz = iz >> kBlockShift;
                const int by = iy >> kBlockShift;
                const int bx = ix >> kBlockShift;
                if (zCross) {
                    s.tryUpdateBlockNonBlocking(bz, by, bx);
                    if (!s.data) return false;
                    const uint8_t* bA = s.data;
                    v000f = loadFromBlock(bA, lz,     ly,     lx);
                    v001f = loadFromBlock(bA, lz,     ly,     lx + 1);
                    v010f = loadFromBlock(bA, lz,     ly + 1, lx);
                    v011f = loadFromBlock(bA, lz,     ly + 1, lx + 1);
                    s.tryUpdateBlockNonBlocking(bz + 1, by, bx);
                    if (!s.data) return false;
                    const uint8_t* bB = s.data;
                    v100f = loadFromBlock(bB, 0,      ly,     lx);
                    v101f = loadFromBlock(bB, 0,      ly,     lx + 1);
                    v110f = loadFromBlock(bB, 0,      ly + 1, lx);
                    v111f = loadFromBlock(bB, 0,      ly + 1, lx + 1);
                } else if (yCross) {
                    s.tryUpdateBlockNonBlocking(bz, by, bx);
                    if (!s.data) return false;
                    const uint8_t* bA = s.data;
                    v000f = loadFromBlock(bA, lz,     ly,     lx);
                    v001f = loadFromBlock(bA, lz,     ly,     lx + 1);
                    v100f = loadFromBlock(bA, lz + 1, ly,     lx);
                    v101f = loadFromBlock(bA, lz + 1, ly,     lx + 1);
                    s.tryUpdateBlockNonBlocking(bz, by + 1, bx);
                    if (!s.data) return false;
                    const uint8_t* bB = s.data;
                    v010f = loadFromBlock(bB, lz,     0,      lx);
                    v011f = loadFromBlock(bB, lz,     0,      lx + 1);
                    v110f = loadFromBlock(bB, lz + 1, 0,      lx);
                    v111f = loadFromBlock(bB, lz + 1, 0,      lx + 1);
                } else {  // xCross
                    s.tryUpdateBlockNonBlocking(bz, by, bx);
                    if (!s.data) return false;
                    const uint8_t* bA = s.data;
                    v000f = loadFromBlock(bA, lz,     ly,     lx);
                    v010f = loadFromBlock(bA, lz,     ly + 1, lx);
                    v100f = loadFromBlock(bA, lz + 1, ly,     lx);
                    v110f = loadFromBlock(bA, lz + 1, ly + 1, lx);
                    s.tryUpdateBlockNonBlocking(bz, by, bx + 1);
                    if (!s.data) return false;
                    const uint8_t* bB = s.data;
                    v001f = loadFromBlock(bB, lz,     ly,     0);
                    v011f = loadFromBlock(bB, lz,     ly + 1, 0);
                    v101f = loadFromBlock(bB, lz + 1, ly,     0);
                    v111f = loadFromBlock(bB, lz + 1, ly + 1, 0);
                }
            } else {
                // 2- or 3-axis crossing, or image-edge adjacent: fall back
                // to per-corner sampleIntNB (handles OOB, negative indices,
                // multi-block fetch uniformly).
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


// ----------------------------------------------------------------------------
// Unified composite-capable adaptive sampler.
// numLayers=1 + nullptr normals = non-composite (same cost as the plain
// adaptive path, no per-layer overhead).
// ----------------------------------------------------------------------------

enum class AccumMode2 : std::uint8_t { Max, Min, Mean, LayerStorage, Volumetric };

static AccumMode2 accumModeFor(const std::string& m) {
    if (m == "max") return AccumMode2::Max;
    if (m == "min") return AccumMode2::Min;
    if (m == "volumetric") return AccumMode2::Volumetric;
    if (m == "median" || m == "alpha" || m == "beerLambert" || m == "minabs") return AccumMode2::LayerStorage;
    return AccumMode2::Mean;
}

// Specialized single-layer (nL==1) kernel. The composite scaffolding —
// accumulator init, layer loop, sdx/sdy/sdz step vectors, AMode finalize
// switch, Volumetric integration — all collapses to "sample one voxel at
// base + nrm*zStart*zStep" when there's only one layer. Template still takes
// SMode so Nearest/Trilinear compile separately; AMode is irrelevant at
// nL=1 (every accumulator reduces to the single sampled value) so this
// kernel is shared across Max/Min/Mean/LayerStorage dispatches.
// Volumetric is explicitly excluded — the multi-layer path still handles it.
template<SampleMode SMode>
void sampleSingleLayerAdaptiveImpl(
    uint32_t* outBuf, int outStride,
    BlockPipeline& cache, int desiredLevel, int numLevels,
    const cv::Mat_<cv::Vec3f>* coords,
    const cv::Vec3f* origin, const cv::Vec3f* vx_step, const cv::Vec3f* vy_step,
    const cv::Mat_<cv::Vec3f>* normals,
    const cv::Vec3f* planeNormal,
    int zStart, float zStep,
    int w, int h,
    const uint32_t lut[256],
    const CompositeParams* lightParams,
    uint8_t* levelOut,
    int levelStride)
{
    auto levelScale = [](int lvl) { return (lvl > 0) ? 1.0f / float(1 << lvl) : 1.0f; };
    const float zOffConst = float(zStart) * zStep;

    // Prefetch the chunks the sampled plane touches. Same as the multi-layer
    // version but the bbox collapses to the single-z-slab defined by zStart.
    cv::Vec3f viewCenterL0(0, 0, 0);
    bool haveCenter = false;
    if (coords) {
        thread_local std::vector<vc::cache::ChunkKey> keys;
        keys.clear();
        for (int lvl = desiredLevel; lvl < numLevels; ++lvl) {
            appendChunksForCoordsSurface(cache, lvl, *coords, keys);
        }
        const cv::Vec3f cvCenter = (*coords)(coords->rows / 2, coords->cols / 2);
        if (isfinite_bitwise(cvCenter[0])
            && (cvCenter[0] * cvCenter[0] + cvCenter[1] * cvCenter[1]
                + cvCenter[2] * cvCenter[2]) > 0.25f) {
            viewCenterL0 = cvCenter;
            haveCenter = true;
        }
        if (!keys.empty()) {
            if (haveCenter) {
                std::array<std::array<int, 3>, vc::cache::kMaxLevels> shapes{};
                for (int lvl = 0; lvl < numLevels && lvl < vc::cache::kMaxLevels; ++lvl)
                    shapes[lvl] = cache.chunkShape(lvl);
                sortKeysByCenterDistance(keys, shapes.data(),
                                         std::min(numLevels, int(vc::cache::kMaxLevels)),
                                         viewCenterL0);
            }
            cache.fetchInteractive(keys, desiredLevel);
        }
    } else {
        cv::Vec3f p0 = *origin + (*planeNormal) * zOffConst;
        cv::Vec3f p1 = *origin + (*vx_step)*float(w-1) + (*vy_step)*float(h-1) + (*planeNormal)*zOffConst;
        float minVx=std::min(p0[0],p1[0]), maxVx=std::max(p0[0],p1[0]);
        float minVy=std::min(p0[1],p1[1]), maxVy=std::max(p0[1],p1[1]);
        float minVz=std::min(p0[2],p1[2]), maxVz=std::max(p0[2],p1[2]);
        thread_local std::vector<vc::cache::ChunkKey> keys;
        keys.clear();
        for (int lvl=desiredLevel; lvl<numLevels; lvl++) {
            float s = levelScale(lvl);
            appendChunksForRegion(cache, lvl,
                minVx*s, minVy*s, minVz*s, maxVx*s, maxVy*s, maxVz*s, keys);
        }
        viewCenterL0 = *origin
            + (*vx_step) * (float(w) * 0.5f)
            + (*vy_step) * (float(h) * 0.5f);
        haveCenter = true;
        if (!keys.empty()) {
            if (haveCenter) {
                std::array<std::array<int, 3>, vc::cache::kMaxLevels> shapes{};
                for (int lvl = 0; lvl < numLevels && lvl < vc::cache::kMaxLevels; ++lvl)
                    shapes[lvl] = cache.chunkShape(lvl);
                sortKeysByCenterDistance(keys, shapes.data(),
                                         std::min(numLevels, int(vc::cache::kMaxLevels)),
                                         viewCenterL0);
            }
            cache.fetchInteractive(keys, desiredLevel);
        }
    }

    float scales[kMaxLevels] = {};
    const int nSamplersTotal = numLevels - desiredLevel;
    for (int i = 0; i < nSamplersTotal && i < kMaxLevels; i++) {
        int lvl = desiredLevel + i;
        scales[i] = (lvl > 0) ? 1.0f / float(1 << lvl) : 1.0f;
    }

    const bool lightingEnabled = lightParams && lightParams->lightingEnabled;
    const int  lightNormalSource = lightParams ? lightParams->lightNormalSource : 0;

    constexpr int kTile = 32;
    const int nTilesY = (h + kTile - 1) / kTile;
    const int nTilesX = (w + kTile - 1) / kTile;
    const int totalTiles = nTilesY * nTilesX;
    std::atomic<int> nextTile{0};

    runRenderThreads([&](int /*tid*/) {
        const int nSamplers = numLevels - desiredLevel;
        std::array<std::optional<BlockSampler<uint8_t>>, kMaxLevels> samplers;
        if (nSamplers > 0) samplers[0].emplace(cache, desiredLevel);
        auto sampler = [&](int i) -> BlockSampler<uint8_t>& {
            if (!samplers[i].has_value())
                samplers[i].emplace(cache, desiredLevel + i);
            return *samplers[i];
        };

        VolumeShape sh0{};
        if (nSamplers > 0) sh0 = sampler(0).shape;
        const float sh0xF = float(sh0.sx), sh0yF = float(sh0.sy), sh0zF = float(sh0.sz);

        float scalesRatio[kMaxLevels] = {};
        for (int i = 0; i < nSamplers && i < kMaxLevels; i++)
            scalesRatio[i] = (i > 0) ? 1.0f / float(1 << i) : 1.0f;

        const cv::Vec3f constNrm = planeNormal ? *planeNormal : cv::Vec3f(0, 0, 0);
        const float endScale = scales[0];
        const float wxNrmConst = constNrm[0] * zOffConst;
        const float wyNrmConst = constNrm[1] * zOffConst;
        const float wzNrmConst = constNrm[2] * zOffConst;

        while (true) {
            const int idx = nextTile.fetch_add(1, std::memory_order_relaxed);
            if (idx >= totalTiles) break;
            const int tyi = idx / nTilesX;
            const int txi = idx % nTilesX;
            const int ty = tyi * kTile;
            const int tx = txi * kTile;
            const int yEnd = std::min(ty + kTile, h);
            const int xEnd = std::min(tx + kTile, w);
        for (int y=ty; y<yEnd; y++) {
            uint32_t* outRow = outBuf + size_t(y) * size_t(outStride);
            uint8_t* lvlRow = levelOut ? (levelOut + size_t(y) * size_t(levelStride)) : nullptr;
            const cv::Vec3f* crow = coords ? coords->ptr<cv::Vec3f>(y) : nullptr;
            const cv::Vec3f* nrow = normals ? normals->ptr<cv::Vec3f>(y) : nullptr;
            for (int x=tx; x<xEnd; x++) {
                cv::Vec3f base = crow ? crow[x] : (*origin + *vx_step*float(x) + *vy_step*float(y));
                if (!isfinite_bitwise(base[0])
                    || (base[0] == 0.f && base[1] == 0.f && base[2] == 0.f)) {
                    outRow[x] = lut[0];
                    if (lvlRow) lvlRow[x] = 0;
                    continue;
                }
                cv::Vec3f nrm = nrow ? nrow[x] : (planeNormal ? *planeNormal : cv::Vec3f(0,0,0));
                if (nrow && !isfinite_bitwise(nrm[0])) {
                    outRow[x] = lut[0];
                    if (lvlRow) lvlRow[x] = 0;
                    continue;
                }
                uint8_t pxLevel = 0;

                const float wxNrm = nrow ? (nrm[0] * zOffConst) : wxNrmConst;
                const float wyNrm = nrow ? (nrm[1] * zOffConst) : wyNrmConst;
                const float wzNrm = nrow ? (nrm[2] * zOffConst) : wzNrmConst;
                const float swx = (base[0] + wxNrm) * endScale;
                const float swy = (base[1] + wyNrm) * endScale;
                const float swz = (base[2] + wzNrm) * endScale;

                uint8_t v = 0;
                bool got = false;
                if constexpr (SMode == SampleMode::Nearest) {
                    if (swx >= 0.5f && swx < sh0xF - 0.5f
                     && swy >= 0.5f && swy < sh0yF - 0.5f
                     && swz >= 0.5f && swz < sh0zF - 0.5f) {
                        got = trySampleNearestUnchecked(*samplers[0], swz, swy, swx, v);
                    }
                }
                if (!got) {
                    for (int i=0; i<nSamplers; i++) {
                        const float r = scalesRatio[i];
                        if (trySampleNB<SMode>(sampler(i),
                            swz * r, swy * r, swx * r, v)) {
                            if (uint8_t(i) > pxLevel) pxLevel = uint8_t(i);
                            break;
                        }
                    }
                }
                float val = float(v);

                if (lightingEnabled) {
                    cv::Vec3f lnrm;
                    if (lightNormalSource == 1) {
                        const float bx = base[0] * endScale;
                        const float by = base[1] * endScale;
                        const float bz = base[2] * endScale;
                        uint8_t gx0=0, gx1=0, gy0=0, gy1=0, gz0=0, gz1=0;
                        const bool gok =
                            trySampleNB<SMode>(*samplers[0], bz, by, bx - 1.f, gx0)
                         && trySampleNB<SMode>(*samplers[0], bz, by, bx + 1.f, gx1)
                         && trySampleNB<SMode>(*samplers[0], bz, by - 1.f, bx, gy0)
                         && trySampleNB<SMode>(*samplers[0], bz, by + 1.f, bx, gy1)
                         && trySampleNB<SMode>(*samplers[0], bz - 1.f, by, bx, gz0)
                         && trySampleNB<SMode>(*samplers[0], bz + 1.f, by, bx, gz1);
                        if (gok) {
                            lnrm = cv::Vec3f(
                                float(gx0) - float(gx1),
                                float(gy0) - float(gy1),
                                float(gz0) - float(gz1));
                        } else {
                            lnrm = nrm;
                        }
                    } else {
                        lnrm = nrm;
                    }
                    val *= computeLightingFactor(lnrm, *lightParams);
                }
                if (val < 0.f) val = 0.f; if (val > 255.f) val = 255.f;
                outRow[x] = lut[uint8_t(val)];
                if (lvlRow) lvlRow[x] = pxLevel;
            }
        }
        }  // while tiles
    });
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
    const uint32_t lut[256],
    const CompositeParams* lightParams,
    uint8_t* levelOut,
    int levelStride)
{
    auto levelScale = [](int lvl) { return (lvl > 0) ? 1.0f / float(1 << lvl) : 1.0f; };
    const float zLo = float(zStart) * zStep;
    const float zHi = float(zStart + numLayers - 1) * zStep;
    const float zMin = std::min(zLo, zHi), zMax = std::max(zLo, zHi);

    // Prefetch covered bbox across all fallback levels.
    cv::Vec3f viewCenterL0(0, 0, 0);
    bool haveCenter = false;
    if (coords) {
        // Surface-aware enumeration replaces bbox: a curved scroll surface
        // spans a huge bbox while only crossing a few hundred chunks. Use
        // the per-pixel coords to enqueue exactly the chunks the surface
        // touches; bbox enumeration is left in place only to compute the
        // view center fallback when the central pixel is NaN.
        thread_local std::vector<vc::cache::ChunkKey> keys;
        keys.clear();
        for (int lvl = desiredLevel; lvl < numLevels; ++lvl) {
            appendChunksForCoordsSurface(cache, lvl, *coords, keys);
        }
        const cv::Vec3f cvCenter = (*coords)(coords->rows / 2, coords->cols / 2);
        // Guard against the (0,0,0) off-surface sentinel — we don't want
        // it biasing the viewport-centre prefetch priority toward origin.
        if (isfinite_bitwise(cvCenter[0])
            && (cvCenter[0] * cvCenter[0] + cvCenter[1] * cvCenter[1]
                + cvCenter[2] * cvCenter[2]) > 0.25f) {
            viewCenterL0 = cvCenter;
            haveCenter = true;
        }
        if (!keys.empty()) {
            if (haveCenter) {
                // Without a real center (viewCenterL0 would fall back to
                // the (0,0,0) default), the distance sort would wrongly
                // bias prefetch toward the volume origin.
                std::array<std::array<int, 3>, vc::cache::kMaxLevels> shapes{};
                for (int lvl = 0; lvl < numLevels && lvl < vc::cache::kMaxLevels; ++lvl)
                    shapes[lvl] = cache.chunkShape(lvl);
                sortKeysByCenterDistance(keys, shapes.data(),
                                         std::min(numLevels, int(vc::cache::kMaxLevels)),
                                         viewCenterL0);
            }
            cache.fetchInteractive(keys, desiredLevel);
        }
    } else {
        cv::Vec3f p0 = *origin + (*planeNormal) * zMin;
        cv::Vec3f p1 = *origin + (*vx_step)*float(w-1) + (*vy_step)*float(h-1) + (*planeNormal)*zMax;
        float minVx=std::min(p0[0],p1[0]), maxVx=std::max(p0[0],p1[0]);
        float minVy=std::min(p0[1],p1[1]), maxVy=std::max(p0[1],p1[1]);
        float minVz=std::min(p0[2],p1[2]), maxVz=std::max(p0[2],p1[2]);
        thread_local std::vector<vc::cache::ChunkKey> keys;
        keys.clear();
        for (int lvl=desiredLevel; lvl<numLevels; lvl++) {
            float s = levelScale(lvl);
            appendChunksForRegion(cache, lvl,
                minVx*s, minVy*s, minVz*s, maxVx*s, maxVy*s, maxVz*s, keys);
        }
        viewCenterL0 = *origin
            + (*vx_step) * (float(w) * 0.5f)
            + (*vy_step) * (float(h) * 0.5f);
        haveCenter = true;
        if (!keys.empty()) {
            if (haveCenter) {
                std::array<std::array<int, 3>, vc::cache::kMaxLevels> shapes{};
                for (int lvl = 0; lvl < numLevels && lvl < vc::cache::kMaxLevels; ++lvl)
                    shapes[lvl] = cache.chunkShape(lvl);
                sortKeysByCenterDistance(keys, shapes.data(),
                                         std::min(numLevels, int(vc::cache::kMaxLevels)),
                                         viewCenterL0);
            }
            cache.fetchInteractive(keys, desiredLevel);
        }
    }

    // Precompute per-level scale factor once (1.0 / 2^lvl). Hoists the
    // integer shift + int->float convert + fdiv out of the hot inner loop.
    float scales[kMaxLevels] = {};
    const int nSamplersTotal = numLevels - desiredLevel;
    for (int i = 0; i < nSamplersTotal && i < kMaxLevels; i++) {
        int lvl = desiredLevel + i;
        scales[i] = (lvl > 0) ? 1.0f / float(1 << lvl) : 1.0f;
    }

    // Parse compositeMethod once. Pixel loop previously did string compare
    // per pixel for the LayerStorage path; convert to a small enum up front.
    // Max and Min also reach here when preprocess is enabled — the per-ray
    // layer preprocess needs layerVals[] populated, which Max/Min's direct
    // accumulate path skips.
    enum class LayerAgg : uint8_t { Median, MinAbs, Alpha, BeerLambert, Mean, Max, Min };
    const LayerAgg layerAgg = (compositeMethod == "median") ? LayerAgg::Median
                            : (compositeMethod == "minabs") ? LayerAgg::MinAbs
                            : (compositeMethod == "alpha") ? LayerAgg::Alpha
                            : (compositeMethod == "beerLambert") ? LayerAgg::BeerLambert
                            : (compositeMethod == "max") ? LayerAgg::Max
                            : (compositeMethod == "min") ? LayerAgg::Min
                            : LayerAgg::Mean;

    // Per-ray layer preprocess flags (applied before the aggregation below).
    const bool preNormalize = lightParams && lightParams->preNormalizeLayers;
    const bool preHistEq    = lightParams && lightParams->preHistEqLayers;

    // UI caps composite layers at 64 front + 64 behind + center = 129. Bound
    // once at the function level so the per-pixel loop and the bounds
    // precheck both see the same compile-time-friendly trip count.
    constexpr int kMaxLayers = 129;
    const int nLHoisted = numLayers > kMaxLayers ? kMaxLayers : numLayers;

    // Hoist lightParams fields into locals so the inner pixel loop doesn't
    // reload them through the pointer each iteration.
    const bool lightingEnabled = lightParams && lightParams->lightingEnabled;
    const int  lightNormalSource = lightParams ? lightParams->lightNormalSource : 0;

    // Volumetric-mode exp LUTs: exp(-extN * k) for k in [0..255]. extN is
    // constant per call (from lightParams->blExtinction/255). Pre-computing
    // 256 entries replaces 2*numLayers std::exp() calls per pixel with two
    // cached table lookups.
    alignas(64) float volExpLUT[256];
    if constexpr (AMode == AccumMode2::Volumetric) {
        const float extinction = lightParams ? lightParams->blExtinction : 1.5f;
        const float extN = extinction / 255.0f;
        for (int k = 0; k < 256; ++k)
            volExpLUT[k] = std::exp(-extN * float(k));
    }

    // Tile the output into 32x32 blocks. Most pixels in a tile map
    // into the same 1-4 level-0 blocks, so the sampler's slot cache
    // stays hot — vs row-major which touches ~120 blocks across a
    // row before cycling back to the same y-row.
    constexpr int kTile = 32;
    const int nTilesY = (h + kTile - 1) / kTile;
    const int nTilesX = (w + kTile - 1) / kTile;
    const int totalTiles = nTilesY * nTilesX;
    std::atomic<int> nextTile{0};

    runRenderThreads([&](int /*tid*/) {
        // Lazy per-level samplers: construct the level-0 sampler eagerly
        // (always used) and leave higher levels unconstructed until the
        // adaptive fallback actually needs them. Each sampler carries a
        // ~4KB hot slot cache; we don't want to pay that cost for levels
        // we never touch.
        const int nSamplers = numLevels - desiredLevel;
        std::array<std::optional<BlockSampler<uint8_t>>, kMaxLevels> samplers;
        if (nSamplers > 0) samplers[0].emplace(cache, desiredLevel);
        auto sampler = [&](int i) -> BlockSampler<uint8_t>& {
            if (!samplers[i].has_value())
                samplers[i].emplace(cache, desiredLevel + i);
            return *samplers[i];
        };
        std::vector<float> layerVals;
        if constexpr (AMode == AccumMode2::LayerStorage) layerVals.resize(numLayers);

        // Hoist desiredLevel sampler's shape out of the per-pixel loop.
        // In hot paths we dereference sh0 once per pixel for bounds; putting
        // it in a local makes sure the compiler hoists past the per-layer
        // loop.
        VolumeShape sh0{};
        if (nSamplers > 0) sh0 = sampler(0).shape;
        const float sh0xF = float(sh0.sx), sh0yF = float(sh0.sy), sh0zF = float(sh0.sz);

        // Ratios for scaling desiredLevel-sampler coords into coarser-level
        // coords: scalesRatio[i] = scales[i] / scales[0] = 1 / 2^i. Used in
        // the fallback loop so the hot path never computes scales[i]/endScale.
        float scalesRatio[kMaxLevels] = {};
        for (int i = 0; i < nSamplers && i < kMaxLevels; i++)
            scalesRatio[i] = (i > 0) ? 1.0f / float(1 << i) : 1.0f;

        // When there's no per-pixel normals map (common case: plane viewer),
        // nrm is constant across every pixel and the pre-scaled step vector
        // (sdx,sdy,sdz) can be computed once here instead of per pixel.
        const cv::Vec3f constNrm =
            planeNormal ? *planeNormal : cv::Vec3f(0, 0, 0);
        const float endScale0 = scales[0];
        const float sdxConst = constNrm[0] * zStep * endScale0;
        const float sdyConst = constNrm[1] * zStep * endScale0;
        const float sdzConst = constNrm[2] * zStep * endScale0;
        const float zOffStart = float(zStart) * zStep;
        const float wxNrmStartConst = constNrm[0] * zOffStart;
        const float wyNrmStartConst = constNrm[1] * zOffStart;
        const float wzNrmStartConst = constNrm[2] * zOffStart;

        while (true) {
            const int idx = nextTile.fetch_add(1, std::memory_order_relaxed);
            if (idx >= totalTiles) break;
            const int tyi = idx / nTilesX;
            const int txi = idx % nTilesX;
            const int ty = tyi * kTile;
            const int tx = txi * kTile;
            const int yEnd = std::min(ty + kTile, h);
            const int xEnd = std::min(tx + kTile, w);
        for (int y=ty; y<yEnd; y++) {
            uint32_t* outRow = outBuf + size_t(y) * size_t(outStride);
            uint8_t* lvlRow = levelOut ? (levelOut + size_t(y) * size_t(levelStride)) : nullptr;
            const cv::Vec3f* crow = coords ? coords->ptr<cv::Vec3f>(y) : nullptr;
            const cv::Vec3f* nrow = normals ? normals->ptr<cv::Vec3f>(y) : nullptr;
            for (int x=tx; x<xEnd; x++) {
                cv::Vec3f base = crow ? crow[x] : (*origin + *vx_step*float(x) + *vy_step*float(y));
                // A surface can report NaN or (0,0,0) for "no data here" —
                // both map to a black output pixel.
                if (!isfinite_bitwise(base[0])
                    || (base[0] == 0.f && base[1] == 0.f && base[2] == 0.f)) {
                    outRow[x] = lut[0];
                    if (lvlRow) lvlRow[x] = 0;
                    continue;
                }
                cv::Vec3f nrm = nrow ? nrow[x] : (planeNormal ? *planeNormal : cv::Vec3f(0,0,0));
                if (nrow && !isfinite_bitwise(nrm[0])) {
                    outRow[x] = lut[0];
                    if (lvlRow) lvlRow[x] = 0;
                    continue;
                }
                uint8_t pxLevel = 0;

                if constexpr (AMode == AccumMode2::Volumetric) {
                    // Volumetric Beer-Lambert with secondary shadow rays.
                    // Walks the view ray front-to-back through the slab;
                    // at each view-ray sample, integrates density along a
                    // shadow ray toward the light to attenuate the voxel's
                    // emission. Materials with high density in front of the
                    // light end up darker (self-shadowing), which reveals
                    // micro-structure — fibers, ink, crackle — as relief.
                    // Extinction is baked into volExpLUT once above; keep
                    // the rest of the volumetric constants per-pixel-local.
                    const CompositeParams* p = lightParams;
                    const float emissionScale = p ? p->blEmission : 1.5f;
                    const float ambient = p ? p->blAmbient : 0.1f;
                    const float diffuse = p ? p->lightDiffuse : 1.0f;
                    const int shadowSteps = p ? std::max(1, p->shadowSteps) : 8;
                    const float lDx = p ? p->lightDirX : 0.5f;
                    const float lDy = p ? p->lightDirY : 0.5f;
                    const float lDz = p ? p->lightDirZ : 0.707f;

                    const float emiN = emissionScale / 255.0f;

                    const float vdx = nrm[0] * zStep;
                    const float vdy = nrm[1] * zStep;
                    const float vdz = nrm[2] * zStep;
                    float wxV = base[0] + nrm[0] * float(zStart) * zStep;
                    float wyV = base[1] + nrm[1] * float(zStart) * zStep;
                    float wzV = base[2] + nrm[2] * float(zStart) * zStep;

                    // All sampling happens at the desired (finest) level.
                    // Sampler coords are world * scl0; one sampler-voxel
                    // step along the light direction is just adding lDir
                    // since it's a unit vector.
                    const float scl0 = scales[0];
                    float transmittance = 1.0f;
                    float accumC = 0.0f;
                    for (int li = 0; li < numLayers; li++) {
                        const float svx = wxV * scl0;
                        const float svy = wyV * scl0;
                        const float svz = wzV * scl0;
                        uint8_t v = 0;
                        bool got = trySampleNearestUnchecked(*samplers[0],
                            svz, svy, svx, v);
                        if (!got) {
                            for (int i = 0; i < nSamplers; i++) {
                                float scl = scales[i];
                                if (trySampleNB<SMode>(sampler(i),
                                    wzV * scl, wyV * scl, wxV * scl, v)) {
                                    if (uint8_t(i) > pxLevel) pxLevel = uint8_t(i);
                                    break;
                                }
                            }
                        }
                        if (v > 0) {
                            // Shadow ray: desired-level samples only. Missing
                            // chunks read as zero density (no shadow) — good
                            // enough while coarse levels stream in.
                            int shadowAccum = 0;
                            float shx = svx, shy = svy, shz = svz;
                            for (int k = 1; k <= shadowSteps; k++) {
                                shx += lDx; shy += lDy; shz += lDz;
                                uint8_t sv = 0;
                                trySampleNB<SMode>(*samplers[0],
                                    shz, shy, shx, sv);
                                shadowAccum += int(sv);
                            }
                            // exp(-extN * x) via 256-entry LUT. shadowSteps
                            // is capped so shadowAccum can exceed 255; clamp
                            // saturates to the darkest entry (matches the
                            // physical "fully opaque" limit).
                            const int shIdx = shadowAccum < 255 ? shadowAccum : 255;
                            const float L = diffuse * volExpLUT[shIdx];
                            const float emission = float(v) * emiN * (L + ambient);
                            const float layerT = volExpLUT[v];
                            accumC += emission * transmittance * (1.0f - layerT);
                            transmittance *= layerT;
                            if (transmittance < 0.001f) break;
                        }
                        wxV += vdx; wyV += vdy; wzV += vdz;
                    }
                    accumC += ambient * transmittance;
                    float vF = std::min(255.0f, accumC * 255.0f);
                    if (vF < 0.f) vF = 0.f;
                    outRow[x] = lut[uint8_t(vF)];
                    if (lvlRow) lvlRow[x] = pxLevel;
                    continue;
                }

                float accum=0.f, mx=0.f, mn=255.f;
                int count=0;
                // Incremental per-layer offset along the normal direction:
                // one vector add per layer instead of 3 FMAs.
                const float endScale = endScale0;
                // Pre-scale step + start into desiredLevel-sampler space so
                // the per-layer hot loop becomes a pure add instead of
                // multiplying three coords by endScale every iteration.
                // When the caller supplied no per-pixel normals, sdx/sdy/sdz
                // are pixel-independent — use the hoisted constants.
                const float sdx = nrow ? (nrm[0] * zStep * endScale) : sdxConst;
                const float sdy = nrow ? (nrm[1] * zStep * endScale) : sdyConst;
                const float sdz = nrow ? (nrm[2] * zStep * endScale) : sdzConst;
                const float wxNrmStart = nrow ? (nrm[0] * zOffStart) : wxNrmStartConst;
                const float wyNrmStart = nrow ? (nrm[1] * zOffStart) : wyNrmStartConst;
                const float wzNrmStart = nrow ? (nrm[2] * zOffStart) : wzNrmStartConst;
                float swx = (base[0] + wxNrmStart) * endScale;
                float swy = (base[1] + wyNrmStart) * endScale;
                float swz = (base[2] + wzNrmStart) * endScale;

                // Pixel-level bounds precheck at level 0: if the entire
                // z-line fits in bounds, skip all per-sample float compares.
                // Reserves 1-voxel margin for nearest rounding. Only the
                // Nearest path uses the resulting flag, so compile the
                // precheck out of the trilinear instantiation entirely.
                bool fullyInBounds = false;
                if constexpr (SMode == SampleMode::Nearest) {
                    const float tailSx = swx + sdx * float(numLayers - 1);
                    const float tailSy = swy + sdy * float(numLayers - 1);
                    const float tailSz = swz + sdz * float(numLayers - 1);
                    const float minSx = std::min(swx, tailSx);
                    const float maxSx = std::max(swx, tailSx);
                    const float minSy = std::min(swy, tailSy);
                    const float maxSy = std::max(swy, tailSy);
                    const float minSz = std::min(swz, tailSz);
                    const float maxSz = std::max(swz, tailSz);
                    fullyInBounds = minSx >= 0.5f && maxSx < sh0xF - 0.5f
                                 && minSy >= 0.5f && maxSy < sh0yF - 0.5f
                                 && minSz >= 0.5f && maxSz < sh0zF - 0.5f;
                }
                const int nL = nLHoisted;
                // Block-run batching: the z-ray advances by (sdx,sdy,sdz)
                // per layer — typically <1 voxel/layer for composite views
                // so the whole 129-layer ray sits in 1-5 blocks. Cache the
                // current block ptr and only re-resolve on (bz,by,bx)
                // change. Saves packKey + lastKey compare (~5 cycles/sample)
                // on every same-block step — i.e. most steps.
                BlockSampler<uint8_t>& s0 = *samplers[0];
                if (fullyInBounds) {
                    // Chunk-grouped sampling: precompute per-layer block
                    // coordinates and in-block offsets in a pure-linear pass
                    // (compiler vectorizes), then walk layers grouped by
                    // block — one tryUpdateBlock call per distinct block,
                    // followed by a branch-free inner loop that just hits
                    // `cdata[offset[i]]` and feeds the accumulator.
                    //
                    // This wins over the original interleaved loop for
                    // long rays (nL=65): the per-sample block-change check
                    // fuses into a single run-length scan over the packed
                    // block-key array, and the inner byte-load loop is
                    // tight enough that clang pipelines it aggressively.
                    alignas(64) int64_t bkey[kMaxLayers];
                    alignas(64) int16_t offset[kMaxLayers];
                    for (int li = 0; li < nL; ++li) {
                        const int iz = int(swz + 0.5f);
                        const int iy = int(swy + 0.5f);
                        const int ix = int(swx + 0.5f);
                        const int bz = iz >> kBlockShift;
                        const int by = iy >> kBlockShift;
                        const int bx = ix >> kBlockShift;
                        // 20 bits per axis is well beyond any realistic
                        // block count (2^20 × 16 = 16M voxels/axis).
                        bkey[li] = (int64_t(bz) << 40)
                                 | (int64_t(by) << 20)
                                 |  int64_t(bx);
                        const int lz = iz & kBlockMask;
                        const int ly = iy & kBlockMask;
                        const int lx = ix & kBlockMask;
                        offset[li] = int16_t(lz * kStrideZ
                                           + ly * kStrideY
                                           + lx);
                        swx += sdx; swy += sdy; swz += sdz;
                    }

                    int li = 0;
                    while (li < nL) {
                        const int64_t key = bkey[li];
                        const int bz = int((key >> 40) & ((int64_t(1) << 20) - 1));
                        const int by = int((key >> 20) & ((int64_t(1) << 20) - 1));
                        const int bx = int( key        & ((int64_t(1) << 20) - 1));
                        s0.tryUpdateBlockNonBlocking(bz, by, bx);
                        const uint8_t* cdata = s0.data;
                        int liEnd = li + 1;
                        while (liEnd < nL && bkey[liEnd] == key) ++liEnd;
                        const int runLen = liEnd - li;
                        if (cdata) {
                            // Tight same-block inner loop: N byte loads
                            // into accumulator. Clang pipelines 4 per
                            // cycle comfortably on ARM — benchmarked
                            // identical to a manual 4-wide unroll so
                            // we keep the simpler scalar form.
                            if constexpr (AMode == AccumMode2::Max) {
                                uint8_t m = uint8_t(mx);
                                for (int i = li; i < liEnd; ++i) {
                                    const uint8_t v = cdata[offset[i]];
                                    m = v > m ? v : m;
                                }
                                mx = float(m);
                            } else if constexpr (AMode == AccumMode2::Min) {
                                uint8_t m = uint8_t(mn);
                                for (int i = li; i < liEnd; ++i) {
                                    const uint8_t v = cdata[offset[i]];
                                    m = v < m ? v : m;
                                }
                                mn = float(m);
                            } else if constexpr (AMode == AccumMode2::Mean) {
                                int sum = 0;
                                for (int i = li; i < liEnd; ++i) {
                                    sum += int(cdata[offset[i]]);
                                }
                                accum += float(sum);
                                count += runLen;
                            } else {
                                for (int i = li; i < liEnd; ++i) {
                                    layerVals[i] = float(cdata[offset[i]]);
                                }
                            }
                        } else if constexpr (AMode == AccumMode2::Mean) {
                            // Missing block → 0 voxels count toward mean
                            count += runLen;
                        } else if constexpr (AMode == AccumMode2::LayerStorage) {
                            for (int i = li; i < liEnd; ++i) layerVals[i] = 0.f;
                        }
                        li = liEnd;
                    }
                } else {
                    // Near-edge / partial-miss path: keep the original
                    // per-layer fallback chain (rare enough that the
                    // chunk-grouped fast path isn't worth forking here).
                    for (int li = 0; li < nL; li++) {
                        uint8_t v = 0;
                        for (int i = 0; i < nSamplers; i++) {
                            const float r = scalesRatio[i];
                            if (trySampleNB<SMode>(sampler(i),
                                swz * r, swy * r, swx * r, v)) {
                                if (uint8_t(i) > pxLevel) pxLevel = uint8_t(i);
                                break;
                            }
                        }
                        if constexpr (AMode == AccumMode2::Max) { mx = std::max(mx, float(v)); }
                        else if constexpr (AMode == AccumMode2::Min) { mn = std::min(mn, float(v)); }
                        else if constexpr (AMode == AccumMode2::Mean) { accum += float(v); count++; }
                        else { layerVals[li] = float(v); }
                        swx += sdx; swy += sdy; swz += sdz;
                    }
                }

                float val = 0.f;
                if constexpr (AMode == AccumMode2::Max) val = mx;
                else if constexpr (AMode == AccumMode2::Min) val = mn;
                else if constexpr (AMode == AccumMode2::Mean) val = count ? accum * (1.0f/float(count)) : 0.f;
                else {
                    // Per-ray layer preprocess: runs only in LayerStorage
                    // mode (we need all N layerVals to compute stats). When
                    // the user enables it for Max/Min/Mean, dispatchComposite
                    // routes the method through LayerStorage so this runs
                    // before the aggregation below.
                    //
                    // Void handling: missing-block samples land in layerVals
                    // as 0, and the iso cutoff threshold zeros out anything
                    // below it. Both get treated as "no data" here — they
                    // don't contribute to normalize min/max or to the hist-eq
                    // CDF (a ray with 100 voids + 29 real samples used to
                    // push the CDF toward max and blow the render out white),
                    // and they stay at 0 through the remap so the aggregation
                    // sees the same voids as without preprocess.
                    // Only build the void mask when preprocess is active;
                    // otherwise keep existing aggregation behavior (no void
                    // filtering) so enabling iso-cutoff alone doesn't silently
                    // change Max/Min/Mean results.
                    const bool preprocessActive = preNormalize || preHistEq;
                    const uint8_t isoCut = lightParams ? lightParams->isoCutoff : 0;
                    const float  isoF   = float(isoCut);
                    alignas(64) uint8_t voidMask[kMaxLayers];
                    int nValid = 0;
                    if (preprocessActive) {
                        for (int i = 0; i < nL; ++i) {
                            // Iso cutoff: <= cutoff → void. Captures both the
                            // user's explicit threshold and the implicit
                            // missing-block zero case (isoCut=0 still treats 0
                            // as void so block holes don't pollute the stats).
                            if (layerVals[i] <= isoF) {
                                layerVals[i] = 0.f;
                                voidMask[i] = 1;
                            } else {
                                voidMask[i] = 0;
                                ++nValid;
                            }
                        }
                    }

                    if (preNormalize && nValid >= 2) {
                        float mnL = 0.f, mxL = 0.f;
                        bool first = true;
                        for (int i = 0; i < nL; ++i) {
                            if (voidMask[i]) continue;
                            const float v = layerVals[i];
                            if (first) { mnL = mxL = v; first = false; }
                            else {
                                if (v < mnL) mnL = v;
                                if (v > mxL) mxL = v;
                            }
                        }
                        const float range = mxL - mnL;
                        if (range > 1e-4f) {
                            const float scl = 255.0f / range;
                            for (int i = 0; i < nL; ++i) {
                                if (voidMask[i]) continue;
                                layerVals[i] = (layerVals[i] - mnL) * scl;
                            }
                        }
                    }
                    if (preHistEq && nValid >= 2) {
                        // Build a 256-bin histogram from the N layer values,
                        // compute the CDF, remap each sample through it.
                        // Classic single-image hist-eq formula, applied to
                        // the N-sample ray: out = 255 * (cdf[v] - cdfMin) /
                        // (nValid - cdfMin). Void samples are skipped so
                        // they don't dominate the CDF.
                        uint32_t hist[256] = {0};
                        for (int i = 0; i < nL; ++i) {
                            if (voidMask[i]) continue;
                            float v = layerVals[i];
                            if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                            hist[uint8_t(v)]++;
                        }
                        uint32_t cdf[256];
                        uint32_t cum = 0;
                        for (int k = 0; k < 256; ++k) {
                            cum += hist[k];
                            cdf[k] = cum;
                        }
                        uint32_t cdfMin = 0;
                        for (int k = 0; k < 256; ++k) {
                            if (cdf[k] > 0) { cdfMin = cdf[k]; break; }
                        }
                        const float denom = float(nValid - int(cdfMin));
                        if (denom > 0.5f) {
                            const float scl = 255.0f / denom;
                            for (int i = 0; i < nL; ++i) {
                                if (voidMask[i]) continue;
                                float v = layerVals[i];
                                if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                                layerVals[i] = float(int(cdf[uint8_t(v)]) - int(cdfMin)) * scl;
                            }
                        }
                    }

                    if (layerAgg == LayerAgg::Median) {
                        // For the small N we see in practice (<=~129 layers),
                        // insertion-sort-up-to-the-median beats nth_element:
                        // its introselect setup costs more than sorting a
                        // few dozen floats. partial_sort gives us exactly
                        // that for small N without branching on size.
                        if (preprocessActive && nValid > 0) {
                            // Pack non-void values to the front, sort those
                            // only. Voids included would drag the median
                            // toward 0 whenever the ray sees block misses.
                            int k = 0;
                            for (int i = 0; i < nL; ++i) {
                                if (!voidMask[i]) layerVals[k++] = layerVals[i];
                            }
                            std::partial_sort(layerVals.begin(),
                                              layerVals.begin() + k/2 + 1,
                                              layerVals.begin() + k);
                            val = layerVals[k/2];
                        } else {
                            std::partial_sort(layerVals.begin(),
                                              layerVals.begin() + nL/2 + 1,
                                              layerVals.begin() + nL);
                            val = layerVals[nL/2];
                        }
                    } else if (layerAgg == LayerAgg::MinAbs) {
                        // Hoist abs(best-127.5) out of the loop and use std::fabs
                        // (hardware fabs opcode vs. std::abs's integer-path).
                        // Skip voids when preprocess is active so block misses
                        // (distance 127.5 from mid) don't win over real data.
                        float best = 0.f, bestAbs = std::numeric_limits<float>::max();
                        for (int i = 0; i < nL; ++i) {
                            if (preprocessActive && voidMask[i]) continue;
                            const float d = std::fabs(layerVals[i] - 127.5f);
                            if (d < bestAbs) { best = layerVals[i]; bestAbs = d; }
                        }
                        val = best;
                    } else if (layerAgg == LayerAgg::Alpha) {
                        const CompositeParams* p = lightParams;
                        const float alphaMin = p ? p->alphaMin * 255.0f : 0.0f;
                        const float alphaMax = p ? p->alphaMax * 255.0f : 255.0f;
                        const float alphaOpacity = p ? p->alphaOpacity : 1.0f;
                        const float alphaCutoff = p ? p->alphaCutoff : 1.0f;
                        const float range = alphaMax - alphaMin;
                        if (range != 0.0f) {
                            const float invRange = 1.0f / range;
                            const float offset = alphaMin / range;
                            float alpha = 0.0f;
                            float valueAcc = 0.0f;
                            for (int i=0; i<nL; i++) {
                                float normalized = layerVals[i] * invRange - offset;
                                if (normalized <= 0.0f) continue;
                                if (normalized > 1.0f) normalized = 1.0f;
                                if (alpha >= alphaCutoff) break;

                                float opacity = normalized * alphaOpacity;
                                if (opacity > 1.0f) opacity = 1.0f;
                                const float weight = (1.0f - alpha) * opacity;
                                valueAcc += weight * normalized;
                                alpha += weight;
                            }
                            val = valueAcc * 255.0f;
                        }
                    } else if (layerAgg == LayerAgg::BeerLambert) {
                        const CompositeParams* p = lightParams;
                        const float extinctionScaled = (p ? p->blExtinction : 1.5f) / 255.0f;
                        const float emissionScaled = (p ? p->blEmission : 1.5f) / 255.0f;
                        const float ambient = p ? p->blAmbient : 0.1f;
                        float transmittance = 1.0f;
                        float accumulatedColor = 0.0f;

                        for (int i=0; i<nL; i++) {
                            const float value = layerVals[i];
                            if (value < 0.255f) continue;

                            const float emission = value * emissionScaled;
                            const float layerTransmittance = std::exp(-extinctionScaled * value);
                            accumulatedColor += emission * transmittance * (1.0f - layerTransmittance);
                            transmittance *= layerTransmittance;
                            if (transmittance < 0.001f) break;
                        }

                        accumulatedColor += ambient * transmittance;
                        val = std::min(255.0f, accumulatedColor * 255.0f);
                    } else if (layerAgg == LayerAgg::Max) {
                        // Max isn't affected by voids (0 never beats a real
                        // sample) so no void gating needed here.
                        float m = layerVals[0];
                        for (int i=1; i<nL; i++) if (layerVals[i] > m) m = layerVals[i];
                        val = m;
                    } else if (layerAgg == LayerAgg::Min) {
                        // Without void gating, Min collapses to 0 as soon as
                        // a single block is missing. Seed from the first
                        // valid value instead when preprocess is on.
                        if (preprocessActive && nValid > 0) {
                            float m = std::numeric_limits<float>::max();
                            for (int i = 0; i < nL; ++i) {
                                if (voidMask[i]) continue;
                                if (layerVals[i] < m) m = layerVals[i];
                            }
                            val = m;
                        } else {
                            float m = layerVals[0];
                            for (int i=1; i<nL; i++) if (layerVals[i] < m) m = layerVals[i];
                            val = m;
                        }
                    } else {
                        // Mean: skip voids when preprocess is active so block
                        // holes don't pull the average toward 0. Fall back to
                        // the plain averaging when preprocess is off to keep
                        // existing behavior identical.
                        if (preprocessActive) {
                            float s = 0.f;
                            for (int i = 0; i < nL; ++i) {
                                if (!voidMask[i]) s += layerVals[i];
                            }
                            val = nValid > 0 ? s / float(nValid) : 0.f;
                        } else {
                            float s=0.f; for (int i=0; i<nL; i++) s += layerVals[i];
                            const float invN = nL>0 ? 1.0f/float(nL) : 0.f;
                            val = s * invN;
                        }
                    }
                }
                if (lightingEnabled) {
                    cv::Vec3f lnrm;
                    if (lightNormalSource == 1) {
                        // Volume-gradient normal: six cheap samples around
                        // base in the desired-level sampler's space. Reveals
                        // local density variation (fibers, ink, crackle)
                        // that a smoothed mesh normal can't.
                        const float bx = base[0] * endScale;
                        const float by = base[1] * endScale;
                        const float bz = base[2] * endScale;
                        uint8_t gx0=0, gx1=0, gy0=0, gy1=0, gz0=0, gz1=0;
                        const bool gok =
                            trySampleNB<SMode>(*samplers[0], bz, by, bx - 1.f, gx0)
                         && trySampleNB<SMode>(*samplers[0], bz, by, bx + 1.f, gx1)
                         && trySampleNB<SMode>(*samplers[0], bz, by - 1.f, bx, gy0)
                         && trySampleNB<SMode>(*samplers[0], bz, by + 1.f, bx, gy1)
                         && trySampleNB<SMode>(*samplers[0], bz - 1.f, by, bx, gz0)
                         && trySampleNB<SMode>(*samplers[0], bz + 1.f, by, bx, gz1);
                        if (gok) {
                            // Gradient from dense → sparse: serves as an
                            // outward-facing surface normal for Lambertian
                            // shading.
                            lnrm = cv::Vec3f(
                                float(gx0) - float(gx1),
                                float(gy0) - float(gy1),
                                float(gz0) - float(gz1));
                        } else {
                            lnrm = nrm;
                        }
                    } else {
                        lnrm = nrm;
                    }
                    val *= computeLightingFactor(lnrm, *lightParams);
                }
                if (val < 0.f) val = 0.f; if (val > 255.f) val = 255.f;
                outRow[x] = lut[uint8_t(val)];
                if (lvlRow) lvlRow[x] = pxLevel;
            }
        }
        }  // while tiles
    });
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
    const uint32_t lut[256],
    const CompositeParams* lightParams,
    uint8_t* levelOut,
    int levelStride)
{
    AccumMode2 mode = accumModeFor(method);
    // Per-ray layer preprocess (normalize / hist-eq over N composite samples)
    // needs the full layer array; force LayerStorage so the kernel stores all
    // N values in layerVals before the preprocess + aggregation stages run.
    // The fast Max/Min/Mean direct-accumulate paths skip storage, so they
    // can't support preprocess without a detour through LayerStorage.
    const bool preprocessActive = lightParams &&
        (lightParams->preNormalizeLayers || lightParams->preHistEqLayers);
    if (preprocessActive && mode != AccumMode2::Volumetric) {
        mode = AccumMode2::LayerStorage;
    }
    // nL=1 reduces Max/Min/Mean to the single sampled value (all three
    // collapse: max(x)=min(x)=mean(x)=x). Dispatch to the specialized kernel
    // to skip the layer loop, accumulator setup, and finalize switch — the
    // plane viewer's dominant path. LayerStorage is excluded because its
    // sub-modes (Alpha, BeerLambert, Median, MinAbs) apply a tone-mapping
    // transform to the single sample rather than returning it raw; Volumetric
    // is excluded because its shadow-ray + transmittance integration doesn't
    // degenerate to a single sample. Preprocess is also excluded — with a
    // single sample the per-ray normalize collapses to 0 and hist-eq to a
    // single bin, neither of which is useful.
    const bool singleLayerFast = numLayers <= 1 && !preprocessActive
        && (mode == AccumMode2::Max
         || mode == AccumMode2::Min
         || mode == AccumMode2::Mean);
    if (singleLayerFast) {
        sampleSingleLayerAdaptiveImpl<SMode>(
            outBuf, outStride, cache, desiredLevel, numLevels,
            coords, origin, vx_step, vy_step, normals, planeNormal,
            zStart, zStep, w, h, lut, lightParams, levelOut, levelStride);
        return;
    }
    switch (mode) {
        case AccumMode2::Max:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::Max>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut, lightParams, levelOut, levelStride); break;
        case AccumMode2::Min:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::Min>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut, lightParams, levelOut, levelStride); break;
        case AccumMode2::LayerStorage:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::LayerStorage>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut, lightParams, levelOut, levelStride); break;
        case AccumMode2::Volumetric:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::Volumetric>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut, lightParams, levelOut, levelStride); break;
        default:
            sampleCompositeAdaptiveImpl<SMode, AccumMode2::Mean>(
                outBuf, outStride, cache, desiredLevel, numLevels,
                coords, origin, vx_step, vy_step, normals, planeNormal,
                numLayers, zStart, zStep, w, h, method, lut, lightParams, levelOut, levelStride); break;
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
    vc::Sampling method,
    const CompositeParams* lightParams,
    uint8_t* levelOut,
    int levelStride)
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
            numLayers, zStart, zStep, width, height, compositeMethod, lut, lightParams, levelOut, levelStride);
    } else {
        dispatchCompositeAdaptive<SampleMode::Trilinear>(
            outBuf, outStride, *cache, desiredLevel, numLevels,
            coords, origin, vx_step, vy_step, normals, planeNormal,
            numLayers, zStart, zStep, width, height, compositeMethod, lut, lightParams, levelOut, levelStride);
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
                            // partial_sort matches the other median path
                            // (Slicing.cpp composite) and beats nth_element
                            // at the small N we run with (<=~65).
                            std::partial_sort(layerVals.begin(),
                                              layerVals.begin() + numLayers / 2 + 1,
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
                    out[i].template ptr<T>(y)[x] = val;
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
                out[i].template ptr<T>(y)[x] = val;
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
