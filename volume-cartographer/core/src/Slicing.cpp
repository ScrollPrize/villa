#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/cache/BlockCache.hpp"
#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/TickCoordinator.hpp"

#include <opencv2/core.hpp>

#if defined(__linux__)
#include <pthread.h>
#include <cstdio>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <semaphore>
#include <thread>
#include <unordered_set>
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
using vc::cache::ChunkKey;
using vc::cache::FrameState;
using vc::cache::kBlockSize;
using vc::cache::kMaxLevels;
using vc::cache::TickCoordinator;

// Shared static zero-block for chunks known to be all-zero. Using one
// instance keeps the cold-cache footprint at 4 KiB instead of one per
// sampler; the BlockPipeline already uses an identical pattern for its
// own empty-chunk short-circuit path.
inline constinit Block kSliceZeroBlock{};

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
template<typename T, int kSlots = 16384>
struct BlockSampler {
    static_assert((kSlots & (kSlots - 1)) == 0, "kSlots must be power of 2");
    static constexpr int kSlotMask = kSlots - 1;

    // Hot slot: packed key + data pointer, 16 bytes. 16384 slots × 16B =
    // 256 KB per sampler — fits in L2D on Oryon (typically 1-2 MB/core).
    // Sized up from 4096 after observing heavy Max-composite workloads
    // (65 layers × 1M pixels, ~500K unique blocks across the frame)
    // sending ~25% of lookups to the BlockCache slow path. The slow
    // path's pthread_rwlock_rdlock/unlock chain was ~44% of CPU in those
    // frames. Quadrupling the per-thread cache drops the direct-mapped
    // collision rate roughly 4× by lowering occupancy, and keeps far
    // more of the hot working set in the per-sampler private cache
    // (where lookups are 2 instructions, no atomics).
    // The BlockPtr (non-owning) lives in a cold parallel array,
    // touched only on miss. Both arrays are heap-allocated so the
    // sampler itself stays ~100 bytes — render workers stack an
    // array<optional<BlockSampler>, kMaxLevels=8> and inline 384 KB
    // per slot would blow past a macOS default pthread stack.
    struct HotSlot {
        uint64_t key = UINT64_MAX;
        const T* data = nullptr;
    };

    BlockPipeline& cache;
    int level;
    VolumeShape shape;
    int chunkBlocksZ = 1;
    int chunkBlocksY = 1;
    int chunkBlocksX = 1;
    // Frame snapshot captured at construction; released in the destructor.
    // When non-null we can bypass `cache.blockAt` on known-empty chunks
    // via a plain-memory binary search instead of an atomic probe loop.
    const FrameState* frame;
    std::unique_ptr<HotSlot[]> slots;
    std::unique_ptr<BlockPtr[]> slotBlocks;  // cold: refcount keep-alive
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
        : cache(c), level(lvl), shape(c, lvl),
          frame(TickCoordinator::currentFrameGlobal()),
          slots(std::make_unique<HotSlot[]>(kSlots)),
          slotBlocks(std::make_unique<BlockPtr[]>(kSlots))
    {
        const auto chunkShape = cache.chunkShape(level);
        chunkBlocksZ = std::max(1, chunkShape[0] / kBlockSize);
        chunkBlocksY = std::max(1, chunkShape[1] / kBlockSize);
        chunkBlocksX = std::max(1, chunkShape[2] / kBlockSize);
    }

    ~BlockSampler() {
        TickCoordinator::releaseFrameGlobal(frame);
    }

    BlockSampler(const BlockSampler&) = delete;
    BlockSampler& operator=(const BlockSampler&) = delete;

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

        // Known-empty chunk short-circuit. FrameState::emptyChunkKeys is
        // a sorted vector published once per tick by TickCoordinator; a
        // binary search is plain memory, vs. the atomic probe loop inside
        // BlockPipeline::blockAt. Chunk shapes are dataset-specific, so
        // map block coordinates back through the metadata-derived grid.
        if (frame) {
            const ChunkKey ck{
                level,
                bz / chunkBlocksZ,
                by / chunkBlocksY,
                bx / chunkBlocksX};
            if (std::binary_search(frame->emptyChunkKeys.begin(),
                                   frame->emptyChunkKeys.end(), ck)) {
                slotBlocks[idx] = &kSliceZeroBlock;
                slot.data = reinterpret_cast<const T*>(kSliceZeroBlock.data);
                slot.key  = key;
                data = slot.data;
                return;
            }
            // Slice L1: freshly-landed blocks at active pyramid levels.
            // Plain-memory binary search saves the atomic-heavy
            // BlockCache::get / isEmptyChunk probes for the first access
            // of hot data. The slice may contain multiple entries for a
            // given packedKey; scan the run looking for matching pipeline
            // and pyramid level.
            if (!frame->slice.empty()) {
                auto it = std::lower_bound(
                    frame->slice.begin(), frame->slice.end(), key,
                    [](const vc::cache::SliceEntry& e, std::uint64_t k) {
                        return e.packedKey < k;
                    });
                while (it != frame->slice.end() && it->packedKey == key) {
                    if (it->pipeline == &cache && it->level == level && it->block) {
                        slotBlocks[idx] = const_cast<BlockPtr>(it->block);
                        slot.data = reinterpret_cast<const T*>(it->block->data);
                        slot.key  = key;
                        data = slot.data;
                        return;
                    }
                    ++it;
                }
            }
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

struct VoxelBounds {
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();

    void include(const cv::Vec3f& p) {
        minX = std::min(minX, p[0]);
        minY = std::min(minY, p[1]);
        minZ = std::min(minZ, p[2]);
        maxX = std::max(maxX, p[0]);
        maxY = std::max(maxY, p[1]);
        maxZ = std::max(maxZ, p[2]);
    }
};

VoxelBounds planeViewportBounds(const cv::Vec3f& origin,
                                const cv::Vec3f& vxStep,
                                const cv::Vec3f& vyStep,
                                int w, int h,
                                const cv::Vec3f& normal,
                                float zMin, float zMax) {
    VoxelBounds b;
    const float xMax = float(std::max(0, w - 1));
    const float yMax = float(std::max(0, h - 1));
    const cv::Vec3f corners[4] = {
        origin,
        origin + vxStep * xMax,
        origin + vyStep * yMax,
        origin + vxStep * xMax + vyStep * yMax,
    };
    for (const cv::Vec3f& corner : corners) {
        b.include(corner + normal * zMin);
        if (zMax != zMin) {
            b.include(corner + normal * zMax);
        }
    }
    return b;
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
    if (!keys.empty()) TickCoordinator::enqueuePrefetchGlobal(&cache, keys, level);
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
