#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/types/Sampling.hpp"


#include "vc/core/types/VcDataset.hpp"
#include "vc/core/cache/ChunkData.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <climits>
#include <limits>
#include <unordered_set>
#include <omp.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

// Force-inline for hot-path functions.  -Oz suppresses inlining aggressively;
// these functions are called millions of times per frame and must be inlined.
#if defined(_MSC_VER)
#define VC_FORCE_INLINE __forceinline
#else
#define VC_FORCE_INLINE __attribute__((always_inline)) inline
#endif

// Non-temporal store for write-only output — bypasses cache,
// freeing cache lines for read-heavy chunk data.
template<typename T>
static inline void nt_store(T* dst, T val) {
#if __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(val, dst);
#else
    *dst = val;
#endif
}

// uint32_t specialization: use _mm_stream_si32 on x86 for guaranteed NT.
static inline void nt_store_u32(uint32_t* dst, uint32_t val) {
#if defined(__x86_64__)
    _mm_stream_si32(reinterpret_cast<int*>(dst), static_cast<int>(val));
#elif __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(val, dst);
#else
    *dst = val;
#endif
}


// Compile-time constants for the standard 128^3 chunk size.
// When chunk128 is true, the compiler emits immediate-operand shifts and masks
// instead of loading shift/mask values from registers.
static constexpr int kChunkDim = 128;
static constexpr int kChunkShift = 7;   // log2(128)
static constexpr int kChunkMask = 127;  // 128 - 1

// ============================================================================
// CacheParams — extract dataset constants once (ZYX ordering)
// ============================================================================

struct CacheParams {
    int cz, cy, cx, sz, sy, sx;
    int czShift, cyShift, cxShift, czMask, cyMask, cxMask;
    bool pow2;  // true when all chunk dims are powers of two (fast path)
    bool chunk128;  // true when all chunk dims are exactly 128 (fastest path)
    int chunksZ, chunksY, chunksX;

    // Precomputed strides for 16^3 block layout (used by trilinearBlockOffsets8)
    int blocksPerX = 0;  // (cx >> 4)
    int blocksPerZ = 0;  // (cy >> 4) * (cx >> 4)

    // Chunk index bounds from logical data extent.
    // Chunks outside [dbMinC*, dbMaxC*] are in zero-padded regions.
    int dbMinCz = 0, dbMaxCz = INT_MAX;
    int dbMinCy = 0, dbMaxCy = INT_MAX;
    int dbMinCx = 0, dbMaxCx = INT_MAX;
    bool dbValid = false;

    explicit CacheParams(vc::cache::TieredChunkCache* cache, int level) {
        auto cs = cache->chunkShape(level);
        cz = cs[0]; cy = cs[1]; cx = cs[2];
        auto shape = cache->levelShape(level);
        sz = shape[0]; sy = shape[1]; sx = shape[2];

        auto isPow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; };
        pow2 = isPow2(cz) && isPow2(cy) && isPow2(cx);
        chunk128 = (cz == kChunkDim && cy == kChunkDim && cx == kChunkDim);
        if (pow2) {
            czShift = log2_pow2(cz); cyShift = log2_pow2(cy); cxShift = log2_pow2(cx);
            czMask = cz - 1; cyMask = cy - 1; cxMask = cx - 1;
        } else {
            czShift = cyShift = cxShift = 0;
            czMask = cyMask = cxMask = 0;
        }

        chunksZ = (sz + cz - 1) / cz;
        chunksY = (sy + cy - 1) / cy;
        chunksX = (sx + cx - 1) / cx;

        blocksPerX = cx >> 4;
        blocksPerZ = (cy >> 4) * blocksPerX;

        // Compute chunk-level data bounds from level-0 bounds.
        // Dilate by 1 chunk on each side — the level-0 bounds are already
        // dilated by 1 coarsest voxel (32 pixels), but chunks are 128^3
        // so an extra chunk of margin avoids clipping at boundaries.
        auto db = cache->dataBounds();
        if (db.valid) {
            float scale = 1.0f / static_cast<float>(1 << level);
            dbMinCx = std::max(0, chunkIdx(static_cast<int>(std::floor(db.minX * scale)), cx, cxShift) - 1);
            dbMaxCx = std::min(chunksX - 1, chunkIdx(static_cast<int>(std::ceil(db.maxX * scale)), cx, cxShift) + 1);
            dbMinCy = std::max(0, chunkIdx(static_cast<int>(std::floor(db.minY * scale)), cy, cyShift) - 1);
            dbMaxCy = std::min(chunksY - 1, chunkIdx(static_cast<int>(std::ceil(db.maxY * scale)), cy, cyShift) + 1);
            dbMinCz = std::max(0, chunkIdx(static_cast<int>(std::floor(db.minZ * scale)), cz, czShift) - 1);
            dbMaxCz = std::min(chunksZ - 1, chunkIdx(static_cast<int>(std::ceil(db.maxZ * scale)), cz, czShift) + 1);
            dbValid = true;
        }
    }

    // Chunk index: shift for pow2, divide otherwise
    VC_FORCE_INLINE int chunkIdx(int v, int c, int shift) const {
        return __builtin_expect(pow2, 1) ? (v >> shift) : (v / c);
    }

    // Local offset within chunk: mask for pow2, modulo otherwise
    VC_FORCE_INLINE int localOff(int v, int c, int mask) const {
        return __builtin_expect(pow2, 1) ? (v & mask) : (v % c);
    }

    // Convenience: chunk indices from voxel coords
    // chunk128 path uses compile-time constants — the compiler emits
    // immediate-operand shifts (e.g. shr reg, 7) with no register loads.
    VC_FORCE_INLINE int chunkZ(int iz) const { return __builtin_expect(chunk128, 1) ? (iz >> kChunkShift) : __builtin_expect(pow2, 1) ? (iz >> czShift) : (iz / cz); }
    VC_FORCE_INLINE int chunkY(int iy) const { return __builtin_expect(chunk128, 1) ? (iy >> kChunkShift) : __builtin_expect(pow2, 1) ? (iy >> cyShift) : (iy / cy); }
    VC_FORCE_INLINE int chunkX(int ix) const { return __builtin_expect(chunk128, 1) ? (ix >> kChunkShift) : __builtin_expect(pow2, 1) ? (ix >> cxShift) : (ix / cx); }

    // Convenience: local offsets from voxel coords
    VC_FORCE_INLINE int localZ(int iz) const { return __builtin_expect(chunk128, 1) ? (iz & kChunkMask) : __builtin_expect(pow2, 1) ? (iz & czMask) : (iz % cz); }
    VC_FORCE_INLINE int localY(int iy) const { return __builtin_expect(chunk128, 1) ? (iy & kChunkMask) : __builtin_expect(pow2, 1) ? (iy & cyMask) : (iy % cy); }
    VC_FORCE_INLINE int localX(int ix) const { return __builtin_expect(chunk128, 1) ? (ix & kChunkMask) : __builtin_expect(pow2, 1) ? (ix & cxMask) : (ix % cx); }


    // 16^3 block layout offset: local coords (lz, ly, lx) -> flat offset.
    // Pure bit ops for pow2 chunks.
    VC_FORCE_INLINE size_t blockOffset(int lz, int ly, int lx) const {
        int bz = lz >> 4, by = ly >> 4, bx = lx >> 4;
        int blockIdx = bz * blocksPerZ + by * blocksPerX + bx;
        int localOff = ((lz & 0xF) << 8) | ((ly & 0xF) << 4) | (lx & 0xF);
        return (static_cast<size_t>(blockIdx) << 12) | localOff;
    }

    // Compute all 8 trilinear offsets at once for adjacent voxels
    // (lz0,ly0,lx0) and (lz0+1,ly0+1,lx0+1) in the same chunk.
    // When none of the +1 coords cross a 16-boundary, offsets are
    // simple additions from a base offset. Otherwise falls back to
    // individual blockOffset calls.
    VC_FORCE_INLINE void trilinearBlockOffsets8(
        int lz0, int ly0, int lx0,
        size_t (&off)[8]) const
    {
        // Check if +1 in any axis crosses a block-16 boundary
        if (__builtin_expect(((lz0 & 0xF) != 0xF) &
                             ((ly0 & 0xF) != 0xF) &
                             ((lx0 & 0xF) != 0xF), 1)) {
            // Fast path: all 8 corners in same 16^3 block
            int bz = lz0 >> 4, by = ly0 >> 4, bx = lx0 >> 4;
            size_t blockBase = static_cast<size_t>(bz * blocksPerZ + by * blocksPerX + bx) << 12;
            int z0 = (lz0 & 0xF), y0 = (ly0 & 0xF), x0 = (lx0 & 0xF);
            size_t base = blockBase | (z0 << 8) | (y0 << 4) | x0;
            off[0] = base;              // (z0, y0, x0)
            off[1] = base + 0x100;      // (z0+1, y0, x0)  = +1 in z = +(1<<8)
            off[2] = base + 0x010;      // (z0, y0+1, x0)  = +1 in y = +(1<<4)
            off[3] = base + 0x110;      // (z0+1, y0+1, x0)
            off[4] = base + 0x001;      // (z0, y0, x0+1)  = +1 in x
            off[5] = base + 0x101;      // (z0+1, y0, x0+1)
            off[6] = base + 0x011;      // (z0, y0+1, x0+1)
            off[7] = base + 0x111;      // (z0+1, y0+1, x0+1)
        } else {
            // Slow path: crosses block boundary
            int lz1 = lz0 + 1, ly1 = ly0 + 1, lx1 = lx0 + 1;
            off[0] = blockOffset(lz0, ly0, lx0);
            off[1] = blockOffset(lz1, ly0, lx0);
            off[2] = blockOffset(lz0, ly1, lx0);
            off[3] = blockOffset(lz1, ly1, lx0);
            off[4] = blockOffset(lz0, ly0, lx1);
            off[5] = blockOffset(lz1, ly0, lx1);
            off[6] = blockOffset(lz0, ly1, lx1);
            off[7] = blockOffset(lz1, ly1, lx1);
        }
    }

    static int log2_pow2(int v) {
        int r = 0;
        while ((v >> r) > 1) r++;
        return r;
    }
};


// ============================================================================
// ChunkSampler — thread-local fast voxel access via raw pointer + strides
//
// Direct-mapped cache: chunk coords (iz, iy, ix) map to a slot index via
//   (iz * NY * NX + iy * NX + ix) & (kSlots - 1)
// This is a single mask operation — faster than the previous hash table.
// kSlots must be a power of 2 (16, 32, or 64). With 16^3 = 4KB alignment
// per sampler slot, the total footprint is kSlots * sizeof(Slot).
// ============================================================================

template<typename T, int kSlots = 32>
struct ChunkSampler {
    static_assert((kSlots & (kSlots - 1)) == 0, "kSlots must be power of 2");
    static constexpr int kSlotMask = kSlots - 1;

    struct Slot {
        uint64_t key = UINT64_MAX;  // packed (iz,iy,ix) key
        vc::cache::ChunkDataPtr chunk;
        const T* data = nullptr;
        bool blockLayout = false;
    };

    const CacheParams& p;
    vc::cache::TieredChunkCache& cache;
    int level;
    Slot slots[kSlots];
    uint64_t lastKey = UINT64_MAX;  // MRU key for fast repeat check
    const T* data = nullptr;  // current data pointer
    size_t s0 = 0, s1 = 0;   // row-major strides (only used when !useBlocks)
    bool useBlocks = false;   // true = current chunk uses 16^3 block layout

    // Grid dimensions for direct index computation
    int NY = 0, NX = 0;

    VC_FORCE_INLINE static uint64_t packKey(int iz, int iy, int ix) {
        return (uint64_t(unsigned(iz)) << 40) | (uint64_t(unsigned(iy)) << 20) | uint64_t(unsigned(ix));
    }

    ChunkSampler(const CacheParams& p_, vc::cache::TieredChunkCache& cache_, int level_)
        : p(p_), cache(cache_), level(level_)
    {
        s0 = static_cast<size_t>(p.cy) * p.cx;
        s1 = static_cast<size_t>(p.cx);
        NY = p.chunksY;
        NX = p.chunksX;
    }


    // Compute flat offset from local coords, respecting current chunk layout.
    VC_FORCE_INLINE size_t voxelOffset(int lz, int ly, int lx) const {
        if (__builtin_expect(useBlocks, 1))
            return p.blockOffset(lz, ly, lx);
        return static_cast<size_t>(lz) * s0 + static_cast<size_t>(ly) * s1 + lx;
    }

    // Direct-mapped index: chunk coords → slot index via simple mask
    VC_FORCE_INLINE int directIndex(int iz, int iy, int ix) const {
        return (iz * NY * NX + iy * NX + ix) & kSlotMask;
    }

    VC_FORCE_INLINE void updateChunk(int iz, int iy, int ix) {
        uint64_t key = packKey(iz, iy, ix);

        // Fast MRU check — same chunk as last call
        if (key == lastKey) return;

        // Quick reject: chunk is entirely in zero-padded region
        if (p.dbValid && (iz < p.dbMinCz || iz > p.dbMaxCz ||
                          iy < p.dbMinCy || iy > p.dbMaxCy ||
                          ix < p.dbMinCx || ix > p.dbMaxCx)) {
            data = nullptr;
            useBlocks = false;
            lastKey = key;
            return;
        }

        // Direct-mapped lookup: single index computation + compare
        int idx = directIndex(iz, iy, ix);
        auto& slot = slots[idx];
        if (slot.key == key) {
            data = slot.data;
            useBlocks = slot.blockLayout;
            lastKey = key;
            return;
        }

        // Miss: evict this slot and load the new chunk
        slot.chunk = cache.getBlocking(vc::cache::ChunkKey{level, iz, iy, ix});
        slot.key = key;
        if (slot.chunk && slot.chunk->isEmpty) {
            // Empty chunk: all voxels zero — treat as null for fast skip
            slot.data = nullptr;
            slot.blockLayout = false;
        } else {
            slot.data = slot.chunk ? slot.chunk->template data<T>() : nullptr;
            slot.blockLayout = slot.chunk ? slot.chunk->blockLayout : false;
        }
        data = slot.data;
        useBlocks = slot.blockLayout;
        lastKey = key;

        // Prefetch chunk data into L1 cache
        if (data) {
            __builtin_prefetch(data, 0, 3);
            __builtin_prefetch(reinterpret_cast<const char*>(data) + 64, 0, 3);
        }
    }

    // Submit chunk keys for async IO+decode before the sampling loop starts.
    void prefetchKeys(const std::vector<vc::cache::ChunkKey>& keys) {
        cache.prefetch(keys);
    }

    // Compute and prefetch all chunk keys that a plane tile will touch.
    void prefetchPlaneChunks(const cv::Vec3f& origin,
                             const cv::Vec3f& vx_step,
                             const cv::Vec3f& vy_step,
                             int w, int h) {
        cv::Vec3f corners[4] = {
            origin,
            origin + vx_step * static_cast<float>(w - 1),
            origin + vy_step * static_cast<float>(h - 1),
            origin + vx_step * static_cast<float>(w - 1) + vy_step * static_cast<float>(h - 1)
        };
        float minVx = corners[0][0], maxVx = corners[0][0];
        float minVy = corners[0][1], maxVy = corners[0][1];
        float minVz = corners[0][2], maxVz = corners[0][2];
        for (int c = 1; c < 4; c++) {
            minVx = std::min(minVx, corners[c][0]); maxVx = std::max(maxVx, corners[c][0]);
            minVy = std::min(minVy, corners[c][1]); maxVy = std::max(maxVy, corners[c][1]);
            minVz = std::min(minVz, corners[c][2]); maxVz = std::max(maxVz, corners[c][2]);
        }
        float margin = 1.0f;
        minVx -= margin; minVy -= margin; minVz -= margin;
        maxVx += margin; maxVy += margin; maxVz += margin;

        if (maxVx < 0 || maxVy < 0 || maxVz < 0 ||
            minVx >= p.sx || minVy >= p.sy || minVz >= p.sz) return;

        int iMinX = std::max(0, static_cast<int>(std::floor(minVx)));
        int iMaxX = std::min(p.sx - 1, static_cast<int>(std::ceil(maxVx)));
        int iMinY = std::max(0, static_cast<int>(std::floor(minVy)));
        int iMaxY = std::min(p.sy - 1, static_cast<int>(std::ceil(maxVy)));
        int iMinZ = std::max(0, static_cast<int>(std::floor(minVz)));
        int iMaxZ = std::min(p.sz - 1, static_cast<int>(std::ceil(maxVz)));

        int minCx = p.chunkX(iMinX), maxCx = p.chunkX(iMaxX);
        int minCy = p.chunkY(iMinY), maxCy = p.chunkY(iMaxY);
        int minCz = p.chunkZ(iMinZ), maxCz = p.chunkZ(iMaxZ);

        std::vector<vc::cache::ChunkKey> keys;
        for (int cix = minCx; cix <= maxCx; cix++)
            for (int ciy = minCy; ciy <= maxCy; ciy++)
                for (int ciz = minCz; ciz <= maxCz; ciz++) {
                    if (p.dbValid && (ciz < p.dbMinCz || ciz > p.dbMaxCz ||
                                      ciy < p.dbMinCy || ciy > p.dbMaxCy ||
                                      cix < p.dbMinCx || cix > p.dbMaxCx)) continue;
                    keys.push_back(vc::cache::ChunkKey{level, ciz, ciy, cix});
                }
        if (!keys.empty()) cache.prefetch(keys);
    }

    bool inBounds(float vz, float vy, float vx) const {
        return vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx;
    }

    VC_FORCE_INLINE T sampleNearest(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz + 0.5f);
        int iy = static_cast<int>(vy + 0.5f);
        int ix = static_cast<int>(vx + 0.5f);
        if (__builtin_expect(iz >= p.sz, 0)) iz = p.sz - 1;
        if (__builtin_expect(iy >= p.sy, 0)) iy = p.sy - 1;
        if (__builtin_expect(ix >= p.sx, 0)) ix = p.sx - 1;

        int ciz, ciy, cix, lz, ly, lx;
        if (__builtin_expect(p.chunk128, 1)) {
            ciz = iz >> kChunkShift; ciy = iy >> kChunkShift; cix = ix >> kChunkShift;
            lz = iz & kChunkMask; ly = iy & kChunkMask; lx = ix & kChunkMask;
        } else if (__builtin_expect(p.pow2, 1)) {
            ciz = iz >> p.czShift; ciy = iy >> p.cyShift; cix = ix >> p.cxShift;
            lz = iz & p.czMask; ly = iy & p.cyMask; lx = ix & p.cxMask;
        } else {
            ciz = iz / p.cz; ciy = iy / p.cy; cix = ix / p.cx;
            lz = iz % p.cz; ly = iy % p.cy; lx = ix % p.cx;
        }

        // Inlined MRU check — avoids function call overhead on the hot path
        uint64_t key = packKey(ciz, ciy, cix);
        if (__builtin_expect(key != lastKey, 0)) {
            updateChunk(ciz, ciy, cix);
        }
        if (__builtin_expect(!data, 0)) return 0;

        return data[voxelOffset(lz, ly, lx)];
    }

    VC_FORCE_INLINE T sampleInt(int iz, int iy, int ix) {
        if (__builtin_expect(iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx, 0))
            return 0;

        int ciz, ciy, cix, lz, ly, lx;
        if (__builtin_expect(p.chunk128, 1)) {
            ciz = iz >> kChunkShift; ciy = iy >> kChunkShift; cix = ix >> kChunkShift;
            lz = iz & kChunkMask; ly = iy & kChunkMask; lx = ix & kChunkMask;
        } else if (__builtin_expect(p.pow2, 1)) {
            ciz = iz >> p.czShift; ciy = iy >> p.cyShift; cix = ix >> p.cxShift;
            lz = iz & p.czMask; ly = iy & p.cyMask; lx = ix & p.cxMask;
        } else {
            ciz = iz / p.cz; ciy = iy / p.cy; cix = ix / p.cx;
            lz = iz % p.cz; ly = iy % p.cy; lx = ix % p.cx;
        }

        // Inlined MRU check — avoids function call overhead on the hot path
        uint64_t key = packKey(ciz, ciy, cix);
        if (__builtin_expect(key != lastKey, 0)) {
            updateChunk(ciz, ciy, cix);
        }
        if (__builtin_expect(!data, 0)) return 0;

        return data[voxelOffset(lz, ly, lx)];
    }

    VC_FORCE_INLINE float sampleTrilinear(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz);
        int iy = static_cast<int>(vy);
        int ix = static_cast<int>(vx);

        float c000 = sampleInt(iz, iy, ix);
        float c100 = sampleInt(iz + 1, iy, ix);
        float c010 = sampleInt(iz, iy + 1, ix);
        float c110 = sampleInt(iz + 1, iy + 1, ix);
        float c001 = sampleInt(iz, iy, ix + 1);
        float c101 = sampleInt(iz + 1, iy, ix + 1);
        float c011 = sampleInt(iz, iy + 1, ix + 1);
        float c111 = sampleInt(iz + 1, iy + 1, ix + 1);

        float fz = vz - iz;
        float fy = vy - iy;
        float fx = vx - ix;

        float c00 = std::fma(fx, c001 - c000, c000);
        float c01 = std::fma(fx, c011 - c010, c010);
        float c10 = std::fma(fx, c101 - c100, c100);
        float c11 = std::fma(fx, c111 - c110, c110);

        float c0 = std::fma(fy, c01 - c00, c00);
        float c1 = std::fma(fy, c11 - c10, c10);

        return std::fma(fz, c1 - c0, c0);
    }

    // Fast path: when all 8 trilinear corners are in the same chunk
    VC_FORCE_INLINE float sampleTrilinearFast(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz);
        int iy = static_cast<int>(vy);
        int ix = static_cast<int>(vx);

        if (iz < 0 || iy < 0 || ix < 0 ||
            iz + 1 >= p.sz || iy + 1 >= p.sy || ix + 1 >= p.sx)
            return sampleTrilinear(vz, vy, vx);

        int ciz0 = p.chunkZ(iz), ciz1 = p.chunkZ(iz + 1);
        int ciy0 = p.chunkY(iy), ciy1 = p.chunkY(iy + 1);
        int cix0 = p.chunkX(ix), cix1 = p.chunkX(ix + 1);

        if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
            updateChunk(ciz0, ciy0, cix0);
            if (!data) return 0;

            int lz0 = p.localZ(iz), ly0 = p.localY(iy), lx0 = p.localX(ix);

            size_t off[8];
            if (__builtin_expect(useBlocks, 1)) {
                p.trilinearBlockOffsets8(lz0, ly0, lx0, off);
            } else {
                size_t base = static_cast<size_t>(lz0) * s0 + static_cast<size_t>(ly0) * s1 + lx0;
                off[0] = base;
                off[1] = base + s0;
                off[2] = base + s1;
                off[3] = base + s0 + s1;
                off[4] = base + 1;
                off[5] = base + s0 + 1;
                off[6] = base + s1 + 1;
                off[7] = base + s0 + s1 + 1;
            }

            float c000 = data[off[0]];
            float c100 = data[off[1]];
            float c010 = data[off[2]];
            float c110 = data[off[3]];
            float c001 = data[off[4]];
            float c101 = data[off[5]];
            float c011 = data[off[6]];
            float c111 = data[off[7]];

            float fz = vz - iz;
            float fy = vy - iy;
            float fx = vx - ix;

            float c00 = std::fma(fx, c001 - c000, c000);
            float c01 = std::fma(fx, c011 - c010, c010);
            float c10 = std::fma(fx, c101 - c100, c100);
            float c11 = std::fma(fx, c111 - c110, c110);

            float c0 = std::fma(fy, c01 - c00, c00);
            float c1 = std::fma(fy, c11 - c10, c10);

            return std::fma(fz, c1 - c0, c0);
        }

        return sampleTrilinear(vz, vy, vx);
    }

    // Same as sampleTrilinearFast but caller guarantees vz/vy/vx are in bounds
    // (i.e., iz >= 0, iy >= 0, ix >= 0, iz+1 < p.sz, iy+1 < p.sy, ix+1 < p.sx).
    VC_FORCE_INLINE float sampleTrilinearUnchecked(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz);
        int iy = static_cast<int>(vy);
        int ix = static_cast<int>(vx);

        int ciz0 = p.chunkZ(iz), ciz1 = p.chunkZ(iz + 1);
        int ciy0 = p.chunkY(iy), ciy1 = p.chunkY(iy + 1);
        int cix0 = p.chunkX(ix), cix1 = p.chunkX(ix + 1);

        if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
            updateChunk(ciz0, ciy0, cix0);
            if (!data) return 0;

            int lz0 = p.localZ(iz), ly0 = p.localY(iy), lx0 = p.localX(ix);

            size_t off[8];
            if (__builtin_expect(useBlocks, 1)) {
                p.trilinearBlockOffsets8(lz0, ly0, lx0, off);
            } else {
                size_t base = static_cast<size_t>(lz0) * s0 + static_cast<size_t>(ly0) * s1 + lx0;
                off[0] = base;
                off[1] = base + s0;
                off[2] = base + s1;
                off[3] = base + s0 + s1;
                off[4] = base + 1;
                off[5] = base + s0 + 1;
                off[6] = base + s1 + 1;
                off[7] = base + s0 + s1 + 1;
            }

            float c000 = data[off[0]];
            float c100 = data[off[1]];
            float c010 = data[off[2]];
            float c110 = data[off[3]];
            float c001 = data[off[4]];
            float c101 = data[off[5]];
            float c011 = data[off[6]];
            float c111 = data[off[7]];

            float fz = vz - iz;
            float fy = vy - iy;
            float fx = vx - ix;

            float c00 = std::fma(fx, c001 - c000, c000);
            float c01 = std::fma(fx, c011 - c010, c010);
            float c10 = std::fma(fx, c101 - c100, c100);
            float c11 = std::fma(fx, c111 - c110, c110);

            float c0 = std::fma(fy, c01 - c00, c00);
            float c1 = std::fma(fy, c11 - c10, c10);

            return std::fma(fz, c1 - c0, c0);
        }

        return sampleTrilinear(vz, vy, vx);
    }

    // Catmull-Rom weight function for tricubic interpolation
    static float catmullRom(float t) {
        float at = std::abs(t);
        if (at < 1.0f) return 1.5f*at*at*at - 2.5f*at*at + 1.0f;
        if (at < 2.0f) return -0.5f*at*at*at + 2.5f*at*at - 4.0f*at + 2.0f;
        return 0.0f;
    }

    float sampleTricubic(float vz, float vy, float vx) {
        int iz = static_cast<int>(std::floor(vz));
        int iy = static_cast<int>(std::floor(vy));
        int ix = static_cast<int>(std::floor(vx));
        float fz = vz - iz, fy = vy - iy, fx = vx - ix;

        float result = 0.0f;
        for (int dz = -1; dz <= 2; dz++) {
            float wz = catmullRom(fz - dz);
            for (int dy = -1; dy <= 2; dy++) {
                float wy = catmullRom(fy - dy);
                float wzy = wz * wy;
                for (int dx = -1; dx <= 2; dx++) {
                    float wx = catmullRom(fx - dx);
                    result += wzy * wx * static_cast<float>(sampleInt(iz+dz, iy+dy, ix+dx));
                }
            }
        }
        return std::clamp(result, 0.0f, static_cast<float>(std::numeric_limits<T>::max()));
    }
};


// ============================================================================
// readVolumeImpl — unified inner loop
// ============================================================================

enum class SampleMode { Nearest, Trilinear, Tricubic };

template<typename T, SampleMode Mode, typename NormalFn>
static void readVolumeImpl(
    cv::Mat_<T>& out,
    vc::cache::TieredChunkCache& cache,
    int level,
    const CacheParams& p,
    const cv::Mat_<cv::Vec3f>& coords,
    NormalFn getNormal,
    int numLayers,
    float zStep,
    int zStart,
    const CompositeParams* params)
{
    const int h = coords.rows;
    const int w = coords.cols;

    // Reuse existing buffer when size matches (avoids malloc/free per tile)
    if (out.rows != h || out.cols != w) {
        out.create(h, w);
    }
    out.setTo(0);

    // Phase 1: Discover needed chunks and prefetch (single-threaded).
    {
        const size_t totalChunks = static_cast<size_t>(p.chunksZ) * p.chunksY * p.chunksX;
        std::vector<uint8_t> needed(totalChunks, 0);
        int minIz = p.chunksZ, maxIz = -1;
        int minIy = p.chunksY, maxIy = -1;
        int minIx = p.chunksX, maxIx = -1;

        auto markVoxel = [&](float vz, float vy, float vx) {
            if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx)) return;
            int iz = static_cast<int>(vz + 0.5f);
            int iy = static_cast<int>(vy + 0.5f);
            int ix = static_cast<int>(vx + 0.5f);
            if (iz >= p.sz) iz = p.sz - 1;
            if (iy >= p.sy) iy = p.sy - 1;
            if (ix >= p.sx) ix = p.sx - 1;
            int ciz = p.chunkZ(iz);
            int ciy = p.chunkY(iy);
            int cix = p.chunkX(ix);
            // Skip chunks in zero-padded regions
            if (p.dbValid && (ciz < p.dbMinCz || ciz > p.dbMaxCz ||
                              ciy < p.dbMinCy || ciy > p.dbMaxCy ||
                              cix < p.dbMinCx || cix > p.dbMaxCx)) return;
            size_t idx = static_cast<size_t>(ciz) * p.chunksY * p.chunksX + static_cast<size_t>(ciy) * p.chunksX + cix;
            if (!needed[idx]) {
                needed[idx] = 1;
                minIz = std::min(minIz, ciz); maxIz = std::max(maxIz, ciz);
                minIy = std::min(minIy, ciy); maxIy = std::max(maxIy, ciy);
                minIx = std::min(minIx, cix); maxIx = std::max(maxIx, cix);
            }
        };

        // Sparse grid sampling for chunk discovery.
        // Coordinates are interpolated from a smooth surface, so a 32-pixel
        // stride is well within a 128-voxel chunk — guarantees we visit every
        // chunk boundary at least once.  Reduces O(w*h) to O((w/S+1)*(h/S+1)).
        constexpr int S = 32; // sparse stride (must be < chunk size)

        // Build sparse Y indices: edges every S pixels + last row
        std::vector<int> sparseY, sparseX;
        sparseY.reserve(h / S + 2);
        sparseX.reserve(w / S + 2);
        for (int y = 0; y < h; y += S) sparseY.push_back(y);
        if (h > 0 && sparseY.back() != h - 1) sparseY.push_back(h - 1);
        for (int x = 0; x < w; x += S) sparseX.push_back(x);
        if (w > 0 && sparseX.back() != w - 1) sparseX.push_back(w - 1);

        if (numLayers == 1) {
            for (int y : sparseY) {
                const auto* row = coords.ptr<cv::Vec3f>(y);
                for (int x : sparseX) {
                    float vx = row[x][0], vy = row[x][1], vz = row[x][2];
                    if constexpr (Mode == SampleMode::Tricubic) {
                        if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx)) continue;
                        int iz = static_cast<int>(std::floor(vz)), iy = static_cast<int>(std::floor(vy)), ix = static_cast<int>(std::floor(vx));
                        for (int dz = -1; dz <= 2; dz++)
                            for (int dy = -1; dy <= 2; dy++)
                                for (int dx = -1; dx <= 2; dx++)
                                    markVoxel(float(iz+dz), float(iy+dy), float(ix+dx));
                    } else if constexpr (Mode == SampleMode::Trilinear) {
                        if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx)) continue;
                        int iz = static_cast<int>(vz), iy = static_cast<int>(vy), ix = static_cast<int>(vx);
                        for (int dz = 0; dz <= 1; dz++)
                            for (int dy = 0; dy <= 1; dy++)
                                for (int dx = 0; dx <= 1; dx++)
                                    markVoxel(float(iz+dz), float(iy+dy), float(ix+dx));
                    } else {
                        markVoxel(vz, vy, vx);
                    }
                }
            }
        } else {
            std::vector<float> layerOffsets(numLayers);
            for (int l = 0; l < numLayers; l++)
                layerOffsets[l] = (zStart + l) * zStep;

            for (int y : sparseY) {
                const auto* row = coords.ptr<cv::Vec3f>(y);
                for (int x : sparseX) {
                    float bx = row[x][0], by = row[x][1], bz = row[x][2];
                    if (!(bz >= 0 && by >= 0 && bx >= 0 && bz < p.sz && by < p.sy && bx < p.sx)) continue;
                    cv::Vec3f n = getNormal(y, x);
                    for (int l = 0; l < numLayers; l++) {
                        float off = layerOffsets[l];
                        markVoxel(bz + n[2]*off, by + n[1]*off, bx + n[0]*off);
                    }
                }
            }
        }

        // Collect needed chunk keys for parallel loading
        std::vector<vc::cache::ChunkKey> neededKeys;
        for (int cix = minIx; cix <= maxIx; cix++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int ciz = minIz; ciz <= maxIz; ciz++)
                    if (needed[static_cast<size_t>(ciz) * p.chunksY * p.chunksX + static_cast<size_t>(ciy) * p.chunksX + cix])
                        neededKeys.push_back(vc::cache::ChunkKey{level, ciz, ciy, cix});

        // Submit async prefetch so IO+decode pipelines across all chunks.
        cache.prefetch(neededKeys);

        // Block until all chunks are available.
        bool anyMissing = false;
        for (auto& key : neededKeys) {
            if (!cache.get(key)) {
                anyMissing = true;
                break;
            }
        }
        if (anyMissing) {
            for (auto& key : neededKeys)
                (void)cache.getBlocking(key);
        }
    }

    // Phase 2: Sample (all chunks already cached)
    //
    // Process pixels in chunk-aligned tile blocks to maximize spatial locality
    // and minimize updateChunk calls. For a typical surface, adjacent pixels
    // map to adjacent voxels, so tiling by chunk size keeps most pixels in
    // the same chunk. We tile over the Y dimension (since X already has good
    // sequential locality within a row) using chunk-sized bands.
    const bool isComposite = (numLayers > 1);

    // Hoist composite method dispatch out of the per-pixel inner loop.
    // The mode never changes, so the compiler can jump-table the switch
    // and the branch predictor will lock on immediately.
    enum AccumMode { kLayerStorage, kMax, kMin, kMean };
    const AccumMode accumMode =
        (params && methodRequiresLayerStorage(params->method)) ? kLayerStorage
      : (params && params->method == "max") ? kMax
      : (params && params->method == "min") ? kMin
      : kMean;
    const bool needsLayerStorage = (accumMode == kLayerStorage);
    const float firstLayerOffset = zStart * zStep;

    // Tile height: use chunk Y size for good locality (chunks are typically 128)
    // but clamp to reasonable range for the OMP scheduler.
    const int tileH = std::max(8, std::min(p.cy, h));
    const int numTiles = (h + tileH - 1) / tileH;

    // Each OMP task processes a horizontal band of tileH rows.
    // The sampler persists across all rows in the band, keeping its cache warm.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int tile = 0; tile < numTiles; tile++) {
        const int yStart = tile * tileH;
        const int yEnd = std::min(yStart + tileH, h);

        // Per-thread sampler and layer stack — persists across rows in this band
        ChunkSampler<T> sampler(p, cache, level);
        LayerStack stack;
        if (needsLayerStorage) {
            stack.values.resize(numLayers);
        }

        for (int y = yStart; y < yEnd; y++) {
            const auto* coordRow = coords.template ptr<cv::Vec3f>(y);
            auto* outRow = out.template ptr<T>(y);

            if (numLayers == 1) {
                // ============================================================
                // Single-sample path — optimized per SampleMode
                // ============================================================
                if constexpr (Mode == SampleMode::Trilinear) {
                    const float* rawCoords = reinterpret_cast<const float*>(coordRow);

                    // Row-level same-chunk check: if first and last pixel
                    // (including +1 trilinear corners) map to the same chunk,
                    // skip all updateChunk calls for the entire row.
                    bool rowSingleChunk = false;
                    int rowCiz = 0, rowCiy = 0, rowCix = 0;
                    if (w > 1) {
                        float r0x = coordRow[0][0], r0y = coordRow[0][1], r0z = coordRow[0][2];
                        float rEx = coordRow[w-1][0], rEy = coordRow[w-1][1], rEz = coordRow[w-1][2];
                        if (r0z >= 0 && r0y >= 0 && r0x >= 0 &&
                            rEz >= 0 && rEy >= 0 && rEx >= 0 &&
                            r0z + 1 < p.sz && r0y + 1 < p.sy && r0x + 1 < p.sx &&
                            rEz + 1 < p.sz && rEy + 1 < p.sy && rEx + 1 < p.sx) {
                            int iz0 = static_cast<int>(r0z), iy0 = static_cast<int>(r0y), ix0 = static_cast<int>(r0x);
                            int izE = static_cast<int>(rEz), iyE = static_cast<int>(rEy), ixE = static_cast<int>(rEx);
                            int ciz0 = p.chunkZ(iz0), ciy0 = p.chunkY(iy0), cix0 = p.chunkX(ix0);
                            int ciz1 = p.chunkZ(izE + 1), ciy1 = p.chunkY(iyE + 1), cix1 = p.chunkX(ixE + 1);
                            if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
                                rowSingleChunk = true;
                                rowCiz = ciz0; rowCiy = ciy0; rowCix = cix0;
                            }
                        }
                    }

                    if (rowSingleChunk) {
                        // Entire row in one chunk — single updateChunk, direct access
                        sampler.updateChunk(rowCiz, rowCiy, rowCix);
                        if (sampler.data) {
                            const T* chunkData = sampler.data;
                            const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                            for (int x = 0; x < w; x++) {
                                float vx = coordRow[x][0], vy = coordRow[x][1], vz = coordRow[x][2];
                                if (!(vz >= 0 && vy >= 0 && vx >= 0 &&
                                      vz < p.sz && vy < p.sy && vx < p.sx)) continue;

                                int iz = static_cast<int>(vz);
                                int iy = static_cast<int>(vy);
                                int ix = static_cast<int>(vx);
                                int lz0 = p.localZ(iz), ly0 = p.localY(iy), lx0 = p.localX(ix);

                                float c000 = chunkData[vo(lz0, ly0, lx0)];
                                float c100 = chunkData[vo((lz0+1), ly0, lx0)];
                                float c010 = chunkData[vo(lz0, (ly0+1), lx0)];
                                float c110 = chunkData[vo((lz0+1), (ly0+1), lx0)];
                                float c001 = chunkData[vo(lz0, ly0, (lx0+1))];
                                float c101 = chunkData[vo((lz0+1), ly0, (lx0+1))];
                                float c011 = chunkData[vo(lz0, (ly0+1), (lx0+1))];
                                float c111 = chunkData[vo((lz0+1), (ly0+1), (lx0+1))];

                                float fz = vz - iz, fy = vy - iy, fx = vx - ix;
                                float c00 = std::fma(fx, c001 - c000, c000);
                                float c01 = std::fma(fx, c011 - c010, c010);
                                float c10 = std::fma(fx, c101 - c100, c100);
                                float c11 = std::fma(fx, c111 - c110, c110);
                                float c0 = std::fma(fy, c01 - c00, c00);
                                float c1 = std::fma(fy, c11 - c10, c10);
                                float v = std::fma(fz, c1 - c0, c0);

                                if constexpr (std::is_same_v<T, uint16_t>) {
                                    if (v < 0.f) v = 0.f;
                                    if (v > 65535.f) v = 65535.f;
                                    outRow[x] = static_cast<uint16_t>(v + 0.5f);
                                } else {
                                    outRow[x] = static_cast<T>(v);
                                }
                            }
                        }
                        // if data was null, output stays zero-initialized
                    } else {
                        // Multi-chunk path: SIMD batch of 4 pixels
                        int x = 0;

#if defined(__aarch64__)
                        // NEON: process 4 pixels at a time
                        const int32x4_t negCxShift = vdupq_n_s32(-p.cxShift);
                        const int32x4_t negCyShift = vdupq_n_s32(-p.cyShift);
                        const int32x4_t negCzShift = vdupq_n_s32(-p.czShift);
                        for (; x + 3 < w; x += 4) {
                            // vld3q_f32 deinterleaves 4 Vec3f into separate x,y,z
                            float32x4x3_t xyz = vld3q_f32(rawCoords + x * 3);
                            float32x4_t vx4 = xyz.val[0];
                            float32x4_t vy4 = xyz.val[1];
                            float32x4_t vz4 = xyz.val[2];

                            // Bounds check: all >= 0 and < size
                            uint32x4_t geZero = vandq_u32(vandq_u32(
                                vcgeq_f32(vx4, vdupq_n_f32(0.0f)),
                                vcgeq_f32(vy4, vdupq_n_f32(0.0f))),
                                vcgeq_f32(vz4, vdupq_n_f32(0.0f)));
                            uint32x4_t ltSize = vandq_u32(vandq_u32(
                                vcltq_f32(vx4, vdupq_n_f32(static_cast<float>(p.sx))),
                                vcltq_f32(vy4, vdupq_n_f32(static_cast<float>(p.sy)))),
                                vcltq_f32(vz4, vdupq_n_f32(static_cast<float>(p.sz))));
                            uint32x4_t validMask = vandq_u32(geZero, ltSize);

                            if (vmaxvq_u32(validMask) == 0) continue;

                            // Floor to int for chunk index computation
                            int32x4_t ix4 = vcvtq_s32_f32(vx4);
                            int32x4_t iy4 = vcvtq_s32_f32(vy4);
                            int32x4_t iz4 = vcvtq_s32_f32(vz4);

                            // Chunk indices (vshlq with negative = right shift)
                            int32x4_t cix4, ciy4, ciz4;
                            if (__builtin_expect(p.pow2, 1)) {
                                cix4 = vshlq_s32(ix4, negCxShift);
                                ciy4 = vshlq_s32(iy4, negCyShift);
                                ciz4 = vshlq_s32(iz4, negCzShift);
                                // Also check +1 corners
                                int32x4_t ones = vdupq_n_s32(1);
                                int32x4_t cix4_1 = vshlq_s32(vaddq_s32(ix4, ones), negCxShift);
                                int32x4_t ciy4_1 = vshlq_s32(vaddq_s32(iy4, ones), negCyShift);
                                int32x4_t ciz4_1 = vshlq_s32(vaddq_s32(iz4, ones), negCzShift);

                                // Extract all chunk indices to scalar for comparison
                                int cixArr[4], ciyArr[4], cizArr[4];
                                int cixArr1[4], ciyArr1[4], cizArr1[4];
                                vst1q_s32(cixArr, cix4); vst1q_s32(ciyArr, ciy4); vst1q_s32(cizArr, ciz4);
                                vst1q_s32(cixArr1, cix4_1); vst1q_s32(ciyArr1, ciy4_1); vst1q_s32(cizArr1, ciz4_1);

                                uint32_t vmask[4];
                                vst1q_u32(vmask, validMask);

                                // Check if all valid pixels are in the same chunk
                                bool allSame = true;
                                bool allValid = true;
                                for (int k = 0; k < 4; k++) {
                                    if (!vmask[k]) { allValid = false; continue; }
                                    if (cixArr[k] != cixArr[0] || ciyArr[k] != ciyArr[0] || cizArr[k] != cizArr[0] ||
                                        cixArr1[k] != cixArr[0] || ciyArr1[k] != ciyArr[0] || cizArr1[k] != cizArr[0]) {
                                        allSame = false; break;
                                    }
                                }

                                if (allValid && allSame) {
                                    // All 4 valid, same chunk — batch sample
                                    sampler.updateChunk(cizArr[0], ciyArr[0], cixArr[0]);
                                    if (sampler.data) {
                                        const T* cd = sampler.data;
                                        const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                                        float fvxArr[4], fvyArr[4], fvzArr[4];
                                        vst1q_f32(fvxArr, vx4); vst1q_f32(fvyArr, vy4); vst1q_f32(fvzArr, vz4);
                                        for (int k = 0; k < 4; k++) {
                                            int iz = static_cast<int>(fvzArr[k]), iy = static_cast<int>(fvyArr[k]), ix = static_cast<int>(fvxArr[k]);
                                            int lz0 = iz & p.czMask, ly0 = iy & p.cyMask, lx0 = ix & p.cxMask;
                                            float c000 = cd[vo(lz0, ly0, lx0)];
                                            float c100 = cd[vo((lz0+1), ly0, lx0)];
                                            float c010 = cd[vo(lz0, (ly0+1), lx0)];
                                            float c110 = cd[vo((lz0+1), (ly0+1), lx0)];
                                            float c001 = cd[vo(lz0, ly0, (lx0+1))];
                                            float c101 = cd[vo((lz0+1), ly0, (lx0+1))];
                                            float c011 = cd[vo(lz0, (ly0+1), (lx0+1))];
                                            float c111 = cd[vo((lz0+1), (ly0+1), (lx0+1))];
                                            float fz = fvzArr[k] - iz, fy = fvyArr[k] - iy, fx = fvxArr[k] - ix;
                                            float rc00 = std::fma(fx, c001 - c000, c000);
                                            float rc01 = std::fma(fx, c011 - c010, c010);
                                            float rc10 = std::fma(fx, c101 - c100, c100);
                                            float rc11 = std::fma(fx, c111 - c110, c110);
                                            float rc0 = std::fma(fy, rc01 - rc00, rc00);
                                            float rc1 = std::fma(fy, rc11 - rc10, rc10);
                                            float v = std::fma(fz, rc1 - rc0, rc0);
                                            if constexpr (std::is_same_v<T, uint16_t>) {
                                                if (v < 0.f) v = 0.f;
                                                if (v > 65535.f) v = 65535.f;
                                                outRow[x + k] = static_cast<uint16_t>(v + 0.5f);
                                            } else {
                                                outRow[x + k] = static_cast<T>(v);
                                            }
                                        }
                                        continue;
                                    }
                                }
                            }

                            // NEON fallback: process 4 pixels individually
                            for (int k = 0; k < 4; k++) {
                                float fvx = rawCoords[(x+k)*3], fvy = rawCoords[(x+k)*3+1], fvz = rawCoords[(x+k)*3+2];
                                if (!(fvz >= 0 && fvy >= 0 && fvx >= 0 &&
                                      fvz < p.sz - 1 && fvy < p.sy - 1 && fvx < p.sx - 1)) continue;
                                float v = sampler.sampleTrilinearUnchecked(fvz, fvy, fvx);
                                if constexpr (std::is_same_v<T, uint16_t>) {
                                    if (v < 0.f) v = 0.f;
                                    if (v > 65535.f) v = 65535.f;
                                    outRow[x + k] = static_cast<uint16_t>(v + 0.5f);
                                } else {
                                    outRow[x + k] = static_cast<T>(v);
                                }
                            }
                        }
#elif defined(__x86_64__)
                        // SSE: process 4 pixels at a time
                        for (; x + 3 < w; x += 4) {
                            // Load 12 floats (4 Vec3f), deinterleave to x,y,z
                            const float* base = rawCoords + x * 3;
                            float xs[4] = { base[0], base[3], base[6], base[9] };
                            float ys[4] = { base[1], base[4], base[7], base[10] };
                            float zs[4] = { base[2], base[5], base[8], base[11] };
                            __m128 vx4 = _mm_loadu_ps(xs);
                            __m128 vy4 = _mm_loadu_ps(ys);
                            __m128 vz4 = _mm_loadu_ps(zs);

                            // Bounds check
                            __m128 zero = _mm_setzero_ps();
                            __m128 geZero = _mm_and_ps(_mm_and_ps(
                                _mm_cmpge_ps(vx4, zero), _mm_cmpge_ps(vy4, zero)),
                                _mm_cmpge_ps(vz4, zero));
                            __m128 ltSize = _mm_and_ps(_mm_and_ps(
                                _mm_cmplt_ps(vx4, _mm_set1_ps(static_cast<float>(p.sx))),
                                _mm_cmplt_ps(vy4, _mm_set1_ps(static_cast<float>(p.sy)))),
                                _mm_cmplt_ps(vz4, _mm_set1_ps(static_cast<float>(p.sz))));
                            int validBits = _mm_movemask_ps(_mm_and_ps(geZero, ltSize));

                            if (validBits == 0) continue;

                            // Floor to int for chunk index
                            __m128i ix4 = _mm_cvttps_epi32(vx4);
                            __m128i iy4 = _mm_cvttps_epi32(vy4);
                            __m128i iz4 = _mm_cvttps_epi32(vz4);

                            if (__builtin_expect(p.pow2, 1) && validBits == 0xF) {
                                // All 4 valid — check same chunk
                                // _mm_sra_epi32 takes shift count in low 64 bits of __m128i
                                __m128i cxShiftV = _mm_cvtsi32_si128(p.cxShift);
                                __m128i cyShiftV = _mm_cvtsi32_si128(p.cyShift);
                                __m128i czShiftV = _mm_cvtsi32_si128(p.czShift);
                                __m128i cix4 = _mm_sra_epi32(ix4, cxShiftV);
                                __m128i ciy4 = _mm_sra_epi32(iy4, cyShiftV);
                                __m128i ciz4 = _mm_sra_epi32(iz4, czShiftV);
                                __m128i ones = _mm_set1_epi32(1);
                                __m128i cix4_1 = _mm_sra_epi32(_mm_add_epi32(ix4, ones), cxShiftV);
                                __m128i ciy4_1 = _mm_sra_epi32(_mm_add_epi32(iy4, ones), cyShiftV);
                                __m128i ciz4_1 = _mm_sra_epi32(_mm_add_epi32(iz4, ones), czShiftV);

                                // Broadcast lane 0 and compare all
                                __m128i cx0 = _mm_shuffle_epi32(cix4, 0);
                                __m128i cy0 = _mm_shuffle_epi32(ciy4, 0);
                                __m128i cz0 = _mm_shuffle_epi32(ciz4, 0);

                                __m128i same = _mm_and_si128(
                                    _mm_and_si128(_mm_cmpeq_epi32(cix4, cx0), _mm_cmpeq_epi32(ciy4, cy0)),
                                    _mm_cmpeq_epi32(ciz4, cz0));
                                __m128i same1 = _mm_and_si128(
                                    _mm_and_si128(_mm_cmpeq_epi32(cix4_1, cx0), _mm_cmpeq_epi32(ciy4_1, cy0)),
                                    _mm_cmpeq_epi32(ciz4_1, cz0));
                                int allSameMask = _mm_movemask_epi8(_mm_and_si128(same, same1));

                                if (allSameMask == 0xFFFF) {
                                    int ciz0v = _mm_cvtsi128_si32(ciz4);
                                    int ciy0v = _mm_cvtsi128_si32(ciy4);
                                    int cix0v = _mm_cvtsi128_si32(cix4);
                                    sampler.updateChunk(ciz0v, ciy0v, cix0v);
                                    if (sampler.data) {
                                        const T* cd = sampler.data;
                                        const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                                        for (int k = 0; k < 4; k++) {
                                            int iz = static_cast<int>(zs[k]), iy = static_cast<int>(ys[k]), ix = static_cast<int>(xs[k]);
                                            int lz0 = iz & p.czMask, ly0 = iy & p.cyMask, lx0 = ix & p.cxMask;
                                            float c000 = cd[vo(lz0, ly0, lx0)];
                                            float c100 = cd[vo((lz0+1), ly0, lx0)];
                                            float c010 = cd[vo(lz0, (ly0+1), lx0)];
                                            float c110 = cd[vo((lz0+1), (ly0+1), lx0)];
                                            float c001 = cd[vo(lz0, ly0, (lx0+1))];
                                            float c101 = cd[vo((lz0+1), ly0, (lx0+1))];
                                            float c011 = cd[vo(lz0, (ly0+1), (lx0+1))];
                                            float c111 = cd[vo((lz0+1), (ly0+1), (lx0+1))];
                                            float fz = zs[k] - iz, fy = ys[k] - iy, fx = xs[k] - ix;
                                            float rc00 = std::fma(fx, c001 - c000, c000);
                                            float rc01 = std::fma(fx, c011 - c010, c010);
                                            float rc10 = std::fma(fx, c101 - c100, c100);
                                            float rc11 = std::fma(fx, c111 - c110, c110);
                                            float rc0 = std::fma(fy, rc01 - rc00, rc00);
                                            float rc1 = std::fma(fy, rc11 - rc10, rc10);
                                            float v = std::fma(fz, rc1 - rc0, rc0);
                                            if constexpr (std::is_same_v<T, uint16_t>) {
                                                if (v < 0.f) v = 0.f;
                                                if (v > 65535.f) v = 65535.f;
                                                outRow[x + k] = static_cast<uint16_t>(v + 0.5f);
                                            } else {
                                                outRow[x + k] = static_cast<T>(v);
                                            }
                                        }
                                        continue;
                                    }
                                }
                            }

                            // SSE fallback: process 4 pixels individually
                            for (int k = 0; k < 4; k++) {
                                if (!(validBits & (1 << k))) continue;
                                float v = sampler.sampleTrilinearFast(zs[k], ys[k], xs[k]);
                                if constexpr (std::is_same_v<T, uint16_t>) {
                                    if (v < 0.f) v = 0.f;
                                    if (v > 65535.f) v = 65535.f;
                                    outRow[x + k] = static_cast<uint16_t>(v + 0.5f);
                                } else {
                                    outRow[x + k] = static_cast<T>(v);
                                }
                            }
                        }
#endif
                        // Scalar tail for remaining pixels (and entire loop on non-SIMD platforms)
                        for (; x < w; x++) {
                            float base_vx = coordRow[x][0], base_vy = coordRow[x][1], base_vz = coordRow[x][2];
                            if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0 &&
                                  base_vz < p.sz - 1 && base_vy < p.sy - 1 && base_vx < p.sx - 1)) continue;
                            float v = sampler.sampleTrilinearUnchecked(base_vz, base_vy, base_vx);
                            if constexpr (std::is_same_v<T, uint16_t>) {
                                if (v < 0.f) v = 0.f;
                                if (v > 65535.f) v = 65535.f;
                                outRow[x] = static_cast<uint16_t>(v + 0.5f);
                            } else {
                                outRow[x] = static_cast<T>(v);
                            }
                        }
                    }
                } else if constexpr (Mode == SampleMode::Tricubic) {
                    for (int x = 0; x < w; x++) {
                        float base_vx = coordRow[x][0], base_vy = coordRow[x][1], base_vz = coordRow[x][2];
                        if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0 &&
                              base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        float v = sampler.sampleTricubic(base_vz, base_vy, base_vx);
                        if constexpr (std::is_same_v<T, uint16_t>) {
                            if (v < 0.f) v = 0.f;
                            if (v > 65535.f) v = 65535.f;
                            outRow[x] = static_cast<uint16_t>(v + 0.5f);
                        } else {
                            outRow[x] = static_cast<T>(v);
                        }
                    }
                } else {
                    // Nearest — row-level same-chunk check
                    bool rowSingleChunk = false;
                    int rowCiz = 0, rowCiy = 0, rowCix = 0;
                    if (w > 1) {
                        float r0x = coordRow[0][0], r0y = coordRow[0][1], r0z = coordRow[0][2];
                        float rEx = coordRow[w-1][0], rEy = coordRow[w-1][1], rEz = coordRow[w-1][2];
                        if (r0z >= 0 && r0y >= 0 && r0x >= 0 &&
                            rEz >= 0 && rEy >= 0 && rEx >= 0 &&
                            r0z < p.sz && r0y < p.sy && r0x < p.sx &&
                            rEz < p.sz && rEy < p.sy && rEx < p.sx) {
                            int iz0 = std::min(static_cast<int>(r0z + 0.5f), p.sz - 1);
                            int iy0 = std::min(static_cast<int>(r0y + 0.5f), p.sy - 1);
                            int ix0 = std::min(static_cast<int>(r0x + 0.5f), p.sx - 1);
                            int izE = std::min(static_cast<int>(rEz + 0.5f), p.sz - 1);
                            int iyE = std::min(static_cast<int>(rEy + 0.5f), p.sy - 1);
                            int ixE = std::min(static_cast<int>(rEx + 0.5f), p.sx - 1);
                            if (iz0 >= 0 && iy0 >= 0 && ix0 >= 0 &&
                                izE >= 0 && iyE >= 0 && ixE >= 0) {
                                int ciz0 = p.chunkZ(iz0), ciy0 = p.chunkY(iy0), cix0 = p.chunkX(ix0);
                                int ciz1 = p.chunkZ(izE), ciy1 = p.chunkY(iyE), cix1 = p.chunkX(ixE);
                                if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
                                    rowSingleChunk = true;
                                    rowCiz = ciz0; rowCiy = ciy0; rowCix = cix0;
                                }
                            }
                        }
                    }

                    if (rowSingleChunk) {
                        sampler.updateChunk(rowCiz, rowCiy, rowCix);
                        if (sampler.data) {
                            const T* chunkData = sampler.data;
                            const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                            for (int x = 0; x < w; x++) {
                                float base_vx = coordRow[x][0], base_vy = coordRow[x][1], base_vz = coordRow[x][2];
                                if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0 &&
                                      base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                                int iz = static_cast<int>(base_vz + 0.5f);
                                int iy = static_cast<int>(base_vy + 0.5f);
                                int ix = static_cast<int>(base_vx + 0.5f);
                                if (__builtin_expect(iz >= p.sz, 0)) iz = p.sz - 1;
                                if (__builtin_expect(iy >= p.sy, 0)) iy = p.sy - 1;
                                if (__builtin_expect(ix >= p.sx, 0)) ix = p.sx - 1;
                                int lz = p.localZ(iz), ly = p.localY(iy), lx = p.localX(ix);
                                nt_store(&outRow[x], chunkData[vo(lz, ly, lx)]);
                            }
                        }
                    } else {
                        for (int x = 0; x < w; x++) {
                            float base_vx = coordRow[x][0], base_vy = coordRow[x][1], base_vz = coordRow[x][2];
                            if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0 &&
                                  base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                            if ((static_cast<int>(base_vz + 0.5f) | static_cast<int>(base_vy + 0.5f) | static_cast<int>(base_vx + 0.5f)) < 0) continue;
                            nt_store(&outRow[x], sampler.sampleNearest(base_vz, base_vy, base_vx));
                        }
                    }
                }
            } else {
                // ============================================================
                // Composite path (multi-layer) — per-pixel, no SIMD
                // ============================================================
                for (int x = 0; x < w; x++) {
                    float base_vx = coordRow[x][0], base_vy = coordRow[x][1], base_vz = coordRow[x][2];
                    if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0)) continue;

                    cv::Vec3f n = getNormal(y, x);
                    float nz = n[2], ny = n[1], nx = n[0];

                    float acc = (accumMode == kMin) ? 255.0f : 0.0f;
                    int validCount = 0;

                    if (needsLayerStorage) {
                        stack.validCount = 0;
                    }

                    float dz = nz * zStep;
                    float dy = ny * zStep;
                    float dx = nx * zStep;

                    float vz = base_vz + nz * firstLayerOffset;
                    float vy = base_vy + ny * firstLayerOffset;
                    float vx = base_vx + nx * firstLayerOffset;

                    for (int layer = 0; layer < numLayers; layer++) {
                        bool validSample = false;
                        float value = 0;

                        if (sampler.inBounds(vz, vy, vx)) {
                            uint8_t raw = sampler.sampleNearest(vz, vy, vx);
                            value = static_cast<float>(raw < params->isoCutoff ? 0 : raw);
                            validSample = true;
                        }

                        vz += dz; vy += dy; vx += dx;

                        if (validSample) {
                            switch (accumMode) {
                            case kLayerStorage: stack.values[stack.validCount++] = value; break;
                            case kMax: acc = value > acc ? value : acc; validCount++; break;
                            case kMin: acc = value < acc ? value : acc; validCount++; break;
                            case kMean: acc += value; validCount++; break;
                            }
                        }
                    }

                    float result = 0.0f;
                    switch (accumMode) {
                    case kLayerStorage: result = compositeLayerStack(stack, *params); break;
                    case kMax: // fallthrough
                    case kMin: result = acc; break;
                    case kMean: if (validCount > 0) result = acc / static_cast<float>(validCount); break;
                    }

                    if (params->lightingEnabled) {
                        float lightFactor = computeLightingFactor(n, *params);
                        result *= lightFactor;
                    }

                    outRow[x] = static_cast<T>(std::max(0.0f, std::min(255.0f, result)));
                }
            }
        }
    }
}


// ============================================================================
// readArea3DImpl
// ============================================================================

template<typename T>
static void readArea3DImpl(Array3D<T>& out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level) {

    CacheParams p(cache, level);

    cv::Vec3i size = {(int)out.shape()[0], (int)out.shape()[1], (int)out.shape()[2]};
    cv::Vec3i to = offset + size;

    // Step 1: List all required chunks
    std::vector<cv::Vec3i> chunks_to_process;
    cv::Vec3i start_chunk = {offset[0] / p.cz, offset[1] / p.cy, offset[2] / p.cx};
    cv::Vec3i end_chunk = {(to[0] - 1) / p.cz, (to[1] - 1) / p.cy, (to[2] - 1) / p.cx};

    for (int cz = start_chunk[0]; cz <= end_chunk[0]; ++cz) {
        for (int cy = start_chunk[1]; cy <= end_chunk[1]; ++cy) {
            for (int cx = start_chunk[2]; cx <= end_chunk[2]; ++cx) {
                chunks_to_process.push_back({cz, cy, cx});
            }
        }
    }

    // Step 2 & 3: Load and copy chunks (no inner OMP — called from parallel tile loop)
    for (const auto& idx : chunks_to_process) {
        int cz = idx[0], cy = idx[1], cx = idx[2];
        auto chunkPtr = cache->getBlocking(vc::cache::ChunkKey{level, cz, cy, cx});

        cv::Vec3i chunk_offset = {p.cz * cz, p.cy * cy, p.cx * cx};

        cv::Vec3i copy_from_start = {
            std::max(offset[0], chunk_offset[0]),
            std::max(offset[1], chunk_offset[1]),
            std::max(offset[2], chunk_offset[2])
        };

        cv::Vec3i copy_from_end = {
            std::min(to[0], chunk_offset[0] + p.cz),
            std::min(to[1], chunk_offset[1] + p.cy),
            std::min(to[2], chunk_offset[2] + p.cx)
        };

        if (chunkPtr) {
            const T* chunkData = chunkPtr->data<T>();
            bool blk = chunkPtr->blockLayout;
            for (int z = copy_from_start[0]; z < copy_from_end[0]; ++z) {
                for (int y = copy_from_start[1]; y < copy_from_end[1]; ++y) {
                    for (int x = copy_from_start[2]; x < copy_from_end[2]; ++x) {
                        int lz = z - chunk_offset[0];
                        int ly = y - chunk_offset[1];
                        int lx = x - chunk_offset[2];
                        size_t off = blk ? p.blockOffset(lz, ly, lx)
                                         : static_cast<size_t>(lz) * p.cy * p.cx + ly * p.cx + lx;
                        out(z - offset[0], y - offset[1], x - offset[2]) = chunkData[off];
                    }
                }
            }
        } else {
            for (int z = copy_from_start[0]; z < copy_from_end[0]; ++z) {
                for (int y = copy_from_start[1]; y < copy_from_end[1]; ++y) {
                    for (int x = copy_from_start[2]; x < copy_from_end[2]; ++x) {
                        out(z - offset[0], y - offset[1], x - offset[2]) = 0;
                    }
                }
            }
        }
    }
}

void readArea3D(Array3D<uint8_t>& out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level) {
    readArea3DImpl(out, offset, cache, level);
}

void readArea3D(Array3D<uint16_t>& out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level) {
    readArea3DImpl(out, offset, cache, level);
}


// ============================================================================
// samplePlaneImpl — fused plane coord generation + sampling (no coords Mat)
// ============================================================================

template<typename T, SampleMode Mode>
static void samplePlaneImpl(
    cv::Mat_<T>& out,
    vc::cache::TieredChunkCache& cache,
    int level,
    const CacheParams& p,
    const cv::Vec3f& origin,
    const cv::Vec3f& vx_step,
    const cv::Vec3f& vy_step,
    int w, int h)
{
    // Reuse existing buffer when size matches (avoids malloc/free per tile)
    if (out.rows != h || out.cols != w) {
        out.create(h, w);
    }
    out.setTo(0);

    // Phase 1: Prefetch needed chunks.
    // Compute world bounding box from 4 corners of the plane.
    {
        cv::Vec3f corners[4] = {
            origin,
            origin + vx_step * static_cast<float>(w - 1),
            origin + vy_step * static_cast<float>(h - 1),
            origin + vx_step * static_cast<float>(w - 1) + vy_step * static_cast<float>(h - 1)
        };

        float minVx = corners[0][0], maxVx = corners[0][0];
        float minVy = corners[0][1], maxVy = corners[0][1];
        float minVz = corners[0][2], maxVz = corners[0][2];
        for (int c = 1; c < 4; c++) {
            minVx = std::min(minVx, corners[c][0]); maxVx = std::max(maxVx, corners[c][0]);
            minVy = std::min(minVy, corners[c][1]); maxVy = std::max(maxVy, corners[c][1]);
            minVz = std::min(minVz, corners[c][2]); maxVz = std::max(maxVz, corners[c][2]);
        }

        // Add interpolation margin
        float margin = (Mode == SampleMode::Tricubic) ? 2.0f : 1.0f;
        minVx -= margin; minVy -= margin; minVz -= margin;
        maxVx += margin; maxVy += margin; maxVz += margin;

        // Clamp to volume bounds
        if (maxVx < 0 || maxVy < 0 || maxVz < 0 ||
            minVx >= p.sx || minVy >= p.sy || minVz >= p.sz) {
            return;  // Entirely out of bounds
        }

        int iMinX = std::max(0, static_cast<int>(std::floor(minVx)));
        int iMaxX = std::min(p.sx - 1, static_cast<int>(std::ceil(maxVx)));
        int iMinY = std::max(0, static_cast<int>(std::floor(minVy)));
        int iMaxY = std::min(p.sy - 1, static_cast<int>(std::ceil(maxVy)));
        int iMinZ = std::max(0, static_cast<int>(std::floor(minVz)));
        int iMaxZ = std::min(p.sz - 1, static_cast<int>(std::ceil(maxVz)));

        int minCx = p.chunkX(iMinX), maxCx = p.chunkX(iMaxX);
        int minCy = p.chunkY(iMinY), maxCy = p.chunkY(iMaxY);
        int minCz = p.chunkZ(iMinZ), maxCz = p.chunkZ(iMaxZ);

        // Collect needed chunk keys
        std::vector<vc::cache::ChunkKey> neededKeys;
        for (int cix = minCx; cix <= maxCx; cix++)
            for (int ciy = minCy; ciy <= maxCy; ciy++)
                for (int ciz = minCz; ciz <= maxCz; ciz++) {
                    if (p.dbValid && (ciz < p.dbMinCz || ciz > p.dbMaxCz ||
                                      ciy < p.dbMinCy || ciy > p.dbMaxCy ||
                                      cix < p.dbMinCx || cix > p.dbMaxCx)) continue;
                    neededKeys.push_back(vc::cache::ChunkKey{level, ciz, ciy, cix});
                }

        // Submit all needed keys for async IO+decode first.
        cache.prefetch(neededKeys);

        bool anyMissing = false;
        for (auto& key : neededKeys) {
            if (!cache.get(key)) {
                anyMissing = true;
                break;
            }
        }
        if (anyMissing) {
            for (auto& key : neededKeys)
                (void)cache.getBlocking(key);
        }
    }

    // Phase 2: Sample with inline coordinate computation
    const int tileH = std::max(8, std::min(p.cy, h));
    const int numTiles = (h + tileH - 1) / tileH;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int tile = 0; tile < numTiles; tile++) {
        const int yStart = tile * tileH;
        const int yEnd = std::min(yStart + tileH, h);

        ChunkSampler<T> sampler(p, cache, level);

        for (int y = yStart; y < yEnd; y++) {
            auto* outRow = out.template ptr<T>(y);
            cv::Vec3f row_base = origin + vy_step * static_cast<float>(y);

            if constexpr (Mode == SampleMode::Trilinear) {
                // Row-level same-chunk check: if first and last pixel
                // map to the same chunk, skip all updateChunk calls.
                cv::Vec3f first = row_base;
                cv::Vec3f last = row_base + vx_step * static_cast<float>(w - 1);
                bool rowSingleChunk = false;
                int rowCiz = 0, rowCiy = 0, rowCix = 0;

                if (w > 1 &&
                    first[2] >= 0 && first[1] >= 0 && first[0] >= 0 &&
                    last[2] >= 0 && last[1] >= 0 && last[0] >= 0 &&
                    first[2] + 1 < p.sz && first[1] + 1 < p.sy && first[0] + 1 < p.sx &&
                    last[2] + 1 < p.sz && last[1] + 1 < p.sy && last[0] + 1 < p.sx) {
                    int iz0 = static_cast<int>(first[2]), iy0 = static_cast<int>(first[1]), ix0 = static_cast<int>(first[0]);
                    int izE = static_cast<int>(last[2]), iyE = static_cast<int>(last[1]), ixE = static_cast<int>(last[0]);
                    int ciz0 = p.chunkZ(iz0), ciy0 = p.chunkY(iy0), cix0 = p.chunkX(ix0);
                    int ciz1 = p.chunkZ(izE + 1), ciy1 = p.chunkY(iyE + 1), cix1 = p.chunkX(ixE + 1);
                    if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
                        rowSingleChunk = true;
                        rowCiz = ciz0; rowCiy = ciy0; rowCix = cix0;
                    }
                }

                if (rowSingleChunk) {
                    sampler.updateChunk(rowCiz, rowCiy, rowCix);
                    if (sampler.data) {
                        const T* chunkData = sampler.data;
                        const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                        float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                        for (int x = 0; x < w; x++) {
                            if (vz >= 0 && vy >= 0 && vx >= 0 &&
                                vz < p.sz && vy < p.sy && vx < p.sx) {
                                int iz = static_cast<int>(vz);
                                int iy = static_cast<int>(vy);
                                int ix = static_cast<int>(vx);
                                int lz0 = p.localZ(iz), ly0 = p.localY(iy), lx0 = p.localX(ix);

                                float c000 = chunkData[vo(lz0, ly0, lx0)];
                                float c100 = chunkData[vo((lz0+1), ly0, lx0)];
                                float c010 = chunkData[vo(lz0, (ly0+1), lx0)];
                                float c110 = chunkData[vo((lz0+1), (ly0+1), lx0)];
                                float c001 = chunkData[vo(lz0, ly0, (lx0+1))];
                                float c101 = chunkData[vo((lz0+1), ly0, (lx0+1))];
                                float c011 = chunkData[vo(lz0, (ly0+1), (lx0+1))];
                                float c111 = chunkData[vo((lz0+1), (ly0+1), (lx0+1))];

                                float fz = vz - iz, fy = vy - iy, fx = vx - ix;
                                float c00 = std::fma(fx, c001 - c000, c000);
                                float c01 = std::fma(fx, c011 - c010, c010);
                                float c10 = std::fma(fx, c101 - c100, c100);
                                float c11 = std::fma(fx, c111 - c110, c110);
                                float c0 = std::fma(fy, c01 - c00, c00);
                                float c1 = std::fma(fy, c11 - c10, c10);
                                float v = std::fma(fz, c1 - c0, c0);

                                if constexpr (std::is_same_v<T, uint16_t>) {
                                    v = std::clamp(v, 0.f, 65535.f);
                                    outRow[x] = static_cast<uint16_t>(v + 0.5f);
                                } else {
                                    outRow[x] = static_cast<T>(v);
                                }
                            }
                            vx += vx_step[0]; vy += vx_step[1]; vz += vx_step[2];
                        }
                    }
                } else {
                    // Multi-chunk path: incremental coord computation
                    float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                    for (int x = 0; x < w; x++) {
                        if (vz >= 0 && vy >= 0 && vx >= 0 &&
                            vz < p.sz - 1 && vy < p.sy - 1 && vx < p.sx - 1) {
                            float v = sampler.sampleTrilinearUnchecked(vz, vy, vx);
                            if constexpr (std::is_same_v<T, uint16_t>) {
                                v = std::clamp(v, 0.f, 65535.f);
                                outRow[x] = static_cast<uint16_t>(v + 0.5f);
                            } else {
                                outRow[x] = static_cast<T>(v);
                            }
                        }
                        vx += vx_step[0]; vy += vx_step[1]; vz += vx_step[2];
                    }
                }
            } else if constexpr (Mode == SampleMode::Tricubic) {
                float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                for (int x = 0; x < w; x++) {
                    if (vz >= 0 && vy >= 0 && vx >= 0 &&
                        vz < p.sz && vy < p.sy && vx < p.sx) {
                        float v = sampler.sampleTricubic(vz, vy, vx);
                        if constexpr (std::is_same_v<T, uint16_t>) {
                            v = std::clamp(v, 0.f, 65535.f);
                            outRow[x] = static_cast<uint16_t>(v + 0.5f);
                        } else {
                            outRow[x] = static_cast<T>(v);
                        }
                    }
                    vx += vx_step[0]; vy += vx_step[1]; vz += vx_step[2];
                }
            } else {
                // Nearest — integer stepping with row-level bounds elimination.
                // Instead of float-to-int per pixel, track integer coords with
                // fractional accumulators (Bresenham-style).
                const float dx = vx_step[0], dy = vx_step[1], dz = vx_step[2];
                cv::Vec3f first = row_base;
                cv::Vec3f last = row_base + vx_step * static_cast<float>(w - 1);

                // Row-level bounds: after nearest-rounding (+0.5f), int coords
                // are in [0, sz-1] iff float coords are in [-0.5, sz-0.5).
                float rowMinX = std::min(first[0], last[0]);
                float rowMaxX = std::max(first[0], last[0]);
                float rowMinY = std::min(first[1], last[1]);
                float rowMaxY = std::max(first[1], last[1]);
                float rowMinZ = std::min(first[2], last[2]);
                float rowMaxZ = std::max(first[2], last[2]);
                bool rowInBounds = rowMinX >= -0.5f && rowMinY >= -0.5f && rowMinZ >= -0.5f &&
                                   rowMaxX < p.sx - 0.5f && rowMaxY < p.sy - 0.5f && rowMaxZ < p.sz - 0.5f;

                // Single-chunk detection from integer bounding box
                bool rowSingleChunk = false;
                int rowCiz = 0, rowCiy = 0, rowCix = 0;
                if (rowInBounds && w > 1) {
                    int iz0 = static_cast<int>(first[2] + 0.5f), iy0 = static_cast<int>(first[1] + 0.5f), ix0 = static_cast<int>(first[0] + 0.5f);
                    int izE = static_cast<int>(last[2] + 0.5f), iyE = static_cast<int>(last[1] + 0.5f), ixE = static_cast<int>(last[0] + 0.5f);
                    int izMin = std::min(iz0, izE), izMax = std::max(iz0, izE);
                    int iyMin = std::min(iy0, iyE), iyMax = std::max(iy0, iyE);
                    int ixMin = std::min(ix0, ixE), ixMax = std::max(ix0, ixE);
                    int ciz0 = p.chunkZ(izMin), ciz1 = p.chunkZ(izMax);
                    int ciy0 = p.chunkY(iyMin), ciy1 = p.chunkY(iyMax);
                    int cix0 = p.chunkX(ixMin), cix1 = p.chunkX(ixMax);
                    if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
                        rowSingleChunk = true;
                        rowCiz = ciz0; rowCiy = ciy0; rowCix = cix0;
                    }
                }

                if (rowSingleChunk) {
                    // Fast path: single chunk, no bounds checks, integer stepping.
                    // Local coords tracked incrementally — no localX/Y/Z per pixel.
                    sampler.updateChunk(rowCiz, rowCiy, rowCix);
                    if (sampler.data) {
                        const T* __restrict__ chunkData = sampler.data;
                        const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                        float fx = first[0] + 0.5f, fy = first[1] + 0.5f, fz = first[2] + 0.5f;
                        int ix = static_cast<int>(fx);
                        int iy = static_cast<int>(fy);
                        int iz = static_cast<int>(fz);
                        int lx = p.localX(ix), ly = p.localY(iy), lz = p.localZ(iz);
                        float fracX = fx - ix, fracY = fy - iy, fracZ = fz - iz;
                        for (int x = 0; x < w; x++) {
                            nt_store(&outRow[x], chunkData[vo(lz, ly, lx)]);
                            fracX += dx;
                            if (fracX >= 1.0f) { lx++; fracX -= 1.0f; }
                            else if (fracX < 0.0f) { lx--; fracX += 1.0f; }
                            fracY += dy;
                            if (fracY >= 1.0f) { ly++; fracY -= 1.0f; }
                            else if (fracY < 0.0f) { ly--; fracY += 1.0f; }
                            fracZ += dz;
                            if (fracZ >= 1.0f) { lz++; fracZ -= 1.0f; }
                            else if (fracZ < 0.0f) { lz--; fracZ += 1.0f; }
                        }
                        // Prefetch next row's starting voxel into L2
                        if (y + 1 < yEnd) {
                            cv::Vec3f nextBase = origin + vy_step * static_cast<float>(y + 1);
                            int niz = static_cast<int>(nextBase[2] + 0.5f);
                            int niy = static_cast<int>(nextBase[1] + 0.5f);
                            int nix = static_cast<int>(nextBase[0] + 0.5f);
                            if (niz >= 0 && niy >= 0 && nix >= 0 &&
                                niz < p.sz && niy < p.sy && nix < p.sx) {
                                int nlz = p.localZ(niz), nly = p.localY(niy), nlx = p.localX(nix);
                                __builtin_prefetch(chunkData + vo(nlz, nly, nlx), 0, 2);
                            }
                        }
                    }
                } else if (rowInBounds) {
                    // In-bounds multi-chunk: integer stepping, no per-pixel bounds checks
                    float fx = first[0] + 0.5f, fy = first[1] + 0.5f, fz = first[2] + 0.5f;
                    int ix = static_cast<int>(fx);
                    int iy = static_cast<int>(fy);
                    int iz = static_cast<int>(fz);
                    float fracX = fx - ix, fracY = fy - iy, fracZ = fz - iz;
                    uint64_t curKey = UINT64_MAX;
                    const T* __restrict__ curData = nullptr;
                    for (int x = 0; x < w; x++) {
                        int ciz = p.chunkZ(iz), ciy = p.chunkY(iy), cix = p.chunkX(ix);
                        uint64_t key = ChunkSampler<T>::packKey(ciz, ciy, cix);
                        if (key != curKey) {
                            sampler.updateChunk(ciz, ciy, cix);
                            curData = sampler.data;
                            curKey = sampler.lastKey;
                        }
                        if (__builtin_expect(curData != nullptr, 1)) {
                            int lz = p.localZ(iz), ly = p.localY(iy), lx = p.localX(ix);
                            nt_store(&outRow[x], curData[sampler.voxelOffset(lz, ly, lx)]);
                        }
                        fracX += dx;
                        if (fracX >= 1.0f) { ix++; fracX -= 1.0f; }
                        else if (fracX < 0.0f) { ix--; fracX += 1.0f; }
                        fracY += dy;
                        if (fracY >= 1.0f) { iy++; fracY -= 1.0f; }
                        else if (fracY < 0.0f) { iy--; fracY += 1.0f; }
                        fracZ += dz;
                        if (fracZ >= 1.0f) { iz++; fracZ -= 1.0f; }
                        else if (fracZ < 0.0f) { iz--; fracZ += 1.0f; }
                    }
                    // Prefetch next row's first chunk data from sampler cache
                    if (y + 1 < yEnd) {
                        cv::Vec3f nextBase = origin + vy_step * static_cast<float>(y + 1);
                        int niz = static_cast<int>(nextBase[2] + 0.5f);
                        int niy = static_cast<int>(nextBase[1] + 0.5f);
                        int nix = static_cast<int>(nextBase[0] + 0.5f);
                        if (niz >= 0 && niy >= 0 && nix >= 0 &&
                            niz < p.sz && niy < p.sy && nix < p.sx) {
                            int nciz = p.chunkZ(niz), nciy = p.chunkY(niy), ncix = p.chunkX(nix);
                            int nIdx = sampler.directIndex(nciz, nciy, ncix);
                            auto& nSlot = sampler.slots[nIdx];
                            if (nSlot.data)
                                __builtin_prefetch(nSlot.data, 0, 2);
                        }
                    }
                } else {
                    // Boundary row: per-pixel bounds checks, float coords
                    float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                    for (int x = 0; x < w; x++) {
                        if (vz >= 0 && vy >= 0 && vx >= 0 &&
                            vz < p.sz && vy < p.sy && vx < p.sx) {
                            nt_store(&outRow[x], sampler.sampleNearest(vz, vy, vx));
                        }
                        vx += dx; vy += dy; vz += dz;
                    }
                }
            }
        }
    }
}

void samplePlane(cv::Mat_<uint8_t>& out, vc::cache::TieredChunkCache* cache, int level,
                 const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                 int width, int height, vc::Sampling method) {
    CacheParams p(cache, level);
    switch (method) {
        case vc::Sampling::Nearest:
            samplePlaneImpl<uint8_t, SampleMode::Nearest>(out, *cache, level, p, origin, vx_step, vy_step, width, height);
            break;
        case vc::Sampling::Tricubic:
            samplePlaneImpl<uint8_t, SampleMode::Tricubic>(out, *cache, level, p, origin, vx_step, vy_step, width, height);
            break;
        default:
            samplePlaneImpl<uint8_t, SampleMode::Trilinear>(out, *cache, level, p, origin, vx_step, vy_step, width, height);
            break;
    }
}




// ============================================================================
// samplePlaneARGB32Impl — fused sampling + LUT: uint8 voxel -> uint32 ARGB32
//
// Identical sampling logic to samplePlaneImpl<uint8_t>, but writes
// lut[voxelValue] as uint32_t to an external buffer instead of uint8 to a Mat.
// This eliminates the intermediate Mat allocation and the second-pass LUT apply.
// ============================================================================

template<SampleMode Mode>
static void samplePlaneARGB32Impl(
    uint32_t* __restrict__ outBuf, int outStride,
    vc::cache::TieredChunkCache& cache,
    int level,
    const CacheParams& p,
    const cv::Vec3f& origin,
    const cv::Vec3f& vx_step,
    const cv::Vec3f& vy_step,
    int w, int h,
    const uint32_t lut[256])
{
    // Zero-fill with lut[0] (the background color for voxel value 0)
    const uint32_t bg = lut[0];
    for (int y = 0; y < h; y++) {
        uint32_t* row = outBuf + y * outStride;
        for (int x = 0; x < w; x++)
            row[x] = bg;
    }

    // Phase 1: Prefetch needed chunks (same as samplePlaneImpl)
    {
        cv::Vec3f corners[4] = {
            origin,
            origin + vx_step * static_cast<float>(w - 1),
            origin + vy_step * static_cast<float>(h - 1),
            origin + vx_step * static_cast<float>(w - 1) + vy_step * static_cast<float>(h - 1)
        };

        float minVx = corners[0][0], maxVx = corners[0][0];
        float minVy = corners[0][1], maxVy = corners[0][1];
        float minVz = corners[0][2], maxVz = corners[0][2];
        for (int c = 1; c < 4; c++) {
            minVx = std::min(minVx, corners[c][0]); maxVx = std::max(maxVx, corners[c][0]);
            minVy = std::min(minVy, corners[c][1]); maxVy = std::max(maxVy, corners[c][1]);
            minVz = std::min(minVz, corners[c][2]); maxVz = std::max(maxVz, corners[c][2]);
        }

        float margin = (Mode == SampleMode::Tricubic) ? 2.0f : 1.0f;
        minVx -= margin; minVy -= margin; minVz -= margin;
        maxVx += margin; maxVy += margin; maxVz += margin;

        if (maxVx < 0 || maxVy < 0 || maxVz < 0 ||
            minVx >= p.sx || minVy >= p.sy || minVz >= p.sz) {
            return;
        }

        int iMinX = std::max(0, static_cast<int>(std::floor(minVx)));
        int iMaxX = std::min(p.sx - 1, static_cast<int>(std::ceil(maxVx)));
        int iMinY = std::max(0, static_cast<int>(std::floor(minVy)));
        int iMaxY = std::min(p.sy - 1, static_cast<int>(std::ceil(maxVy)));
        int iMinZ = std::max(0, static_cast<int>(std::floor(minVz)));
        int iMaxZ = std::min(p.sz - 1, static_cast<int>(std::ceil(maxVz)));

        int minCx = p.chunkX(iMinX), maxCx = p.chunkX(iMaxX);
        int minCy = p.chunkY(iMinY), maxCy = p.chunkY(iMaxY);
        int minCz = p.chunkZ(iMinZ), maxCz = p.chunkZ(iMaxZ);

        std::vector<vc::cache::ChunkKey> neededKeys;
        for (int cix = minCx; cix <= maxCx; cix++)
            for (int ciy = minCy; ciy <= maxCy; ciy++)
                for (int ciz = minCz; ciz <= maxCz; ciz++) {
                    if (p.dbValid && (ciz < p.dbMinCz || ciz > p.dbMaxCz ||
                                      ciy < p.dbMinCy || ciy > p.dbMaxCy ||
                                      cix < p.dbMinCx || cix > p.dbMaxCx)) continue;
                    neededKeys.push_back(vc::cache::ChunkKey{level, ciz, ciy, cix});
                }

        cache.prefetch(neededKeys);

        bool anyMissing = false;
        for (auto& key : neededKeys) {
            if (!cache.get(key)) {
                anyMissing = true;
                break;
            }
        }
        if (anyMissing) {
            for (auto& key : neededKeys)
                (void)cache.getBlocking(key);
        }
    }

    // Phase 2: Sample with inline coordinate computation, write ARGB32 via LUT
    const int tileH = std::max(8, std::min(p.cy, h));
    const int numTiles = (h + tileH - 1) / tileH;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int tile = 0; tile < numTiles; tile++) {
        const int yStart = tile * tileH;
        const int yEnd = std::min(yStart + tileH, h);

        ChunkSampler<uint8_t> sampler(p, cache, level);

        for (int y = yStart; y < yEnd; y++) {
            uint32_t* __restrict__ outRow = outBuf + y * outStride;
            cv::Vec3f row_base = origin + vy_step * static_cast<float>(y);

            if constexpr (Mode == SampleMode::Trilinear) {
                cv::Vec3f first = row_base;
                cv::Vec3f last = row_base + vx_step * static_cast<float>(w - 1);
                bool rowSingleChunk = false;
                int rowCiz = 0, rowCiy = 0, rowCix = 0;

                if (w > 1 &&
                    first[2] >= 0 && first[1] >= 0 && first[0] >= 0 &&
                    last[2] >= 0 && last[1] >= 0 && last[0] >= 0 &&
                    first[2] + 1 < p.sz && first[1] + 1 < p.sy && first[0] + 1 < p.sx &&
                    last[2] + 1 < p.sz && last[1] + 1 < p.sy && last[0] + 1 < p.sx) {
                    int iz0 = static_cast<int>(first[2]), iy0 = static_cast<int>(first[1]), ix0 = static_cast<int>(first[0]);
                    int izE = static_cast<int>(last[2]), iyE = static_cast<int>(last[1]), ixE = static_cast<int>(last[0]);
                    int ciz0 = p.chunkZ(iz0), ciy0 = p.chunkY(iy0), cix0 = p.chunkX(ix0);
                    int ciz1 = p.chunkZ(izE + 1), ciy1 = p.chunkY(iyE + 1), cix1 = p.chunkX(ixE + 1);
                    if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
                        rowSingleChunk = true;
                        rowCiz = ciz0; rowCiy = ciy0; rowCix = cix0;
                    }
                }

                if (rowSingleChunk) {
                    sampler.updateChunk(rowCiz, rowCiy, rowCix);
                    if (sampler.data) {
                        const uint8_t* chunkData = sampler.data;
                        const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                        float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                        for (int x = 0; x < w; x++) {
                            if (vz >= 0 && vy >= 0 && vx >= 0 &&
                                vz < p.sz && vy < p.sy && vx < p.sx) {
                                int iz = static_cast<int>(vz);
                                int iy = static_cast<int>(vy);
                                int ix = static_cast<int>(vx);
                                int lz0 = p.localZ(iz), ly0 = p.localY(iy), lx0 = p.localX(ix);

                                float c000 = chunkData[vo(lz0, ly0, lx0)];
                                float c100 = chunkData[vo((lz0+1), ly0, lx0)];
                                float c010 = chunkData[vo(lz0, (ly0+1), lx0)];
                                float c110 = chunkData[vo((lz0+1), (ly0+1), lx0)];
                                float c001 = chunkData[vo(lz0, ly0, (lx0+1))];
                                float c101 = chunkData[vo((lz0+1), ly0, (lx0+1))];
                                float c011 = chunkData[vo(lz0, (ly0+1), (lx0+1))];
                                float c111 = chunkData[vo((lz0+1), (ly0+1), (lx0+1))];

                                float fz = vz - iz, fy = vy - iy, fx = vx - ix;
                                float c00 = std::fma(fx, c001 - c000, c000);
                                float c01 = std::fma(fx, c011 - c010, c010);
                                float c10 = std::fma(fx, c101 - c100, c100);
                                float c11 = std::fma(fx, c111 - c110, c110);
                                float c0 = std::fma(fy, c01 - c00, c00);
                                float c1 = std::fma(fy, c11 - c10, c10);
                                float v = std::fma(fz, c1 - c0, c0);

                                nt_store_u32(&outRow[x], lut[static_cast<uint8_t>(v)]);
                            }
                            vx += vx_step[0]; vy += vx_step[1]; vz += vx_step[2];
                        }
                    }
                } else {
                    float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                    for (int x = 0; x < w; x++) {
                        if (vz >= 0 && vy >= 0 && vx >= 0 &&
                            vz < p.sz - 1 && vy < p.sy - 1 && vx < p.sx - 1) {
                            float v = sampler.sampleTrilinearUnchecked(vz, vy, vx);
                            nt_store_u32(&outRow[x], lut[static_cast<uint8_t>(v)]);
                        }
                        vx += vx_step[0]; vy += vx_step[1]; vz += vx_step[2];
                    }
                }
            } else if constexpr (Mode == SampleMode::Tricubic) {
                float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                for (int x = 0; x < w; x++) {
                    if (vz >= 0 && vy >= 0 && vx >= 0 &&
                        vz < p.sz && vy < p.sy && vx < p.sx) {
                        float v = sampler.sampleTricubic(vz, vy, vx);
                        nt_store_u32(&outRow[x], lut[static_cast<uint8_t>(v)]);
                    }
                    vx += vx_step[0]; vy += vx_step[1]; vz += vx_step[2];
                }
            } else {
                // Nearest -- integer stepping
                const float dx = vx_step[0], dy = vx_step[1], dz = vx_step[2];
                cv::Vec3f first = row_base;
                cv::Vec3f last = row_base + vx_step * static_cast<float>(w - 1);

                float rowMinX = std::min(first[0], last[0]);
                float rowMaxX = std::max(first[0], last[0]);
                float rowMinY = std::min(first[1], last[1]);
                float rowMaxY = std::max(first[1], last[1]);
                float rowMinZ = std::min(first[2], last[2]);
                float rowMaxZ = std::max(first[2], last[2]);
                bool rowInBounds = rowMinX >= -0.5f && rowMinY >= -0.5f && rowMinZ >= -0.5f &&
                                   rowMaxX < p.sx - 0.5f && rowMaxY < p.sy - 0.5f && rowMaxZ < p.sz - 0.5f;

                bool rowSingleChunk = false;
                int rowCiz = 0, rowCiy = 0, rowCix = 0;
                if (rowInBounds && w > 1) {
                    int iz0 = static_cast<int>(first[2] + 0.5f), iy0 = static_cast<int>(first[1] + 0.5f), ix0 = static_cast<int>(first[0] + 0.5f);
                    int izE = static_cast<int>(last[2] + 0.5f), iyE = static_cast<int>(last[1] + 0.5f), ixE = static_cast<int>(last[0] + 0.5f);
                    int izMin = std::min(iz0, izE), izMax = std::max(iz0, izE);
                    int iyMin = std::min(iy0, iyE), iyMax = std::max(iy0, iyE);
                    int ixMin = std::min(ix0, ixE), ixMax = std::max(ix0, ixE);
                    int ciz0 = p.chunkZ(izMin), ciz1 = p.chunkZ(izMax);
                    int ciy0 = p.chunkY(iyMin), ciy1 = p.chunkY(iyMax);
                    int cix0 = p.chunkX(ixMin), cix1 = p.chunkX(ixMax);
                    if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
                        rowSingleChunk = true;
                        rowCiz = ciz0; rowCiy = ciy0; rowCix = cix0;
                    }
                }

                if (rowSingleChunk) {
                    sampler.updateChunk(rowCiz, rowCiy, rowCix);
                    if (sampler.data) {
                        const uint8_t* __restrict__ chunkData = sampler.data;
                        const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };
                        float fx = first[0] + 0.5f, fy = first[1] + 0.5f, fz = first[2] + 0.5f;
                        int ix = static_cast<int>(fx);
                        int iy = static_cast<int>(fy);
                        int iz = static_cast<int>(fz);
                        int lx = p.localX(ix), ly = p.localY(iy), lz = p.localZ(iz);
                        float fracX = fx - ix, fracY = fy - iy, fracZ = fz - iz;
                        for (int x = 0; x < w; x++) {
                            nt_store_u32(&outRow[x], lut[chunkData[vo(lz, ly, lx)]]);
                            fracX += dx;
                            if (fracX >= 1.0f) { lx++; fracX -= 1.0f; }
                            else if (fracX < 0.0f) { lx--; fracX += 1.0f; }
                            fracY += dy;
                            if (fracY >= 1.0f) { ly++; fracY -= 1.0f; }
                            else if (fracY < 0.0f) { ly--; fracY += 1.0f; }
                            fracZ += dz;
                            if (fracZ >= 1.0f) { lz++; fracZ -= 1.0f; }
                            else if (fracZ < 0.0f) { lz--; fracZ += 1.0f; }
                        }
                        // Prefetch next row's starting voxel into L2
                        if (y + 1 < yEnd) {
                            cv::Vec3f nextBase = origin + vy_step * static_cast<float>(y + 1);
                            int niz = static_cast<int>(nextBase[2] + 0.5f);
                            int niy = static_cast<int>(nextBase[1] + 0.5f);
                            int nix = static_cast<int>(nextBase[0] + 0.5f);
                            if (niz >= 0 && niy >= 0 && nix >= 0 &&
                                niz < p.sz && niy < p.sy && nix < p.sx) {
                                int nlz = p.localZ(niz), nly = p.localY(niy), nlx = p.localX(nix);
                                __builtin_prefetch(chunkData + vo(nlz, nly, nlx), 0, 2);
                            }
                        }
                    }
                } else if (rowInBounds) {
                    float fx = first[0] + 0.5f, fy = first[1] + 0.5f, fz = first[2] + 0.5f;
                    int ix = static_cast<int>(fx);
                    int iy = static_cast<int>(fy);
                    int iz = static_cast<int>(fz);
                    float fracX = fx - ix, fracY = fy - iy, fracZ = fz - iz;
                    uint64_t curKey = UINT64_MAX;
                    const uint8_t* __restrict__ curData = nullptr;
                    for (int x = 0; x < w; x++) {
                        int ciz = p.chunkZ(iz), ciy = p.chunkY(iy), cix = p.chunkX(ix);
                        uint64_t key = ChunkSampler<uint8_t>::packKey(ciz, ciy, cix);
                        if (key != curKey) {
                            sampler.updateChunk(ciz, ciy, cix);
                            curData = sampler.data;
                            curKey = sampler.lastKey;
                        }
                        if (__builtin_expect(curData != nullptr, 1)) {
                            int lz = p.localZ(iz), ly = p.localY(iy), lx = p.localX(ix);
                            nt_store_u32(&outRow[x], lut[curData[sampler.voxelOffset(lz, ly, lx)]]);
                        }
                        fracX += dx;
                        if (fracX >= 1.0f) { ix++; fracX -= 1.0f; }
                        else if (fracX < 0.0f) { ix--; fracX += 1.0f; }
                        fracY += dy;
                        if (fracY >= 1.0f) { iy++; fracY -= 1.0f; }
                        else if (fracY < 0.0f) { iy--; fracY += 1.0f; }
                        fracZ += dz;
                        if (fracZ >= 1.0f) { iz++; fracZ -= 1.0f; }
                        else if (fracZ < 0.0f) { iz--; fracZ += 1.0f; }
                    }
                    // Prefetch next row's first chunk data from sampler cache
                    if (y + 1 < yEnd) {
                        cv::Vec3f nextBase = origin + vy_step * static_cast<float>(y + 1);
                        int niz = static_cast<int>(nextBase[2] + 0.5f);
                        int niy = static_cast<int>(nextBase[1] + 0.5f);
                        int nix = static_cast<int>(nextBase[0] + 0.5f);
                        if (niz >= 0 && niy >= 0 && nix >= 0 &&
                            niz < p.sz && niy < p.sy && nix < p.sx) {
                            int nciz = p.chunkZ(niz), nciy = p.chunkY(niy), ncix = p.chunkX(nix);
                            int nIdx = sampler.directIndex(nciz, nciy, ncix);
                            auto& nSlot = sampler.slots[nIdx];
                            if (nSlot.data)
                                __builtin_prefetch(nSlot.data, 0, 2);
                        }
                    }
                } else {
                    float vx = row_base[0], vy = row_base[1], vz = row_base[2];
                    for (int x = 0; x < w; x++) {
                        if (vz >= 0 && vy >= 0 && vx >= 0 &&
                            vz < p.sz && vy < p.sy && vx < p.sx) {
                            nt_store_u32(&outRow[x], lut[sampler.sampleNearest(vz, vy, vx)]);
                        }
                        vx += dx; vy += dy; vz += dz;
                    }
                }
            }
        }
    }
}

void samplePlaneARGB32(uint32_t* outBuf, int outStride,
                       vc::cache::TieredChunkCache* cache, int level,
                       const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                       int width, int height, vc::Sampling method,
                       const uint32_t lut[256]) {
    CacheParams p(cache, level);
    switch (method) {
        case vc::Sampling::Nearest:
            samplePlaneARGB32Impl<SampleMode::Nearest>(outBuf, outStride, *cache, level, p, origin, vx_step, vy_step, width, height, lut);
            break;
        case vc::Sampling::Tricubic:
            samplePlaneARGB32Impl<SampleMode::Tricubic>(outBuf, outStride, *cache, level, p, origin, vx_step, vy_step, width, height, lut);
            break;
        default:
            samplePlaneARGB32Impl<SampleMode::Trilinear>(outBuf, outStride, *cache, level, p, origin, vx_step, vy_step, width, height, lut);
            break;
    }
}

// ============================================================================
// Public API — thin wrappers around readVolumeImpl
// ============================================================================

template<typename T>
static void readInterpolated3DImpl(cv::Mat_<T>& out, vc::cache::TieredChunkCache* cache, int level,
                                   const cv::Mat_<cv::Vec3f>& coords,
                                   vc::Sampling method) {
    CacheParams p(cache, level);
    auto noNormal = [](int, int) -> cv::Vec3f { return {}; };

    switch (method) {
        case vc::Sampling::Nearest:
            readVolumeImpl<T, SampleMode::Nearest>(out, *cache, level, p, coords,
                noNormal, 1, 0.f, 0, nullptr);
            break;
        case vc::Sampling::Tricubic:
            readVolumeImpl<T, SampleMode::Tricubic>(out, *cache, level, p, coords,
                noNormal, 1, 0.f, 0, nullptr);
            break;
        default:
            readVolumeImpl<T, SampleMode::Trilinear>(out, *cache, level, p, coords,
                noNormal, 1, 0.f, 0, nullptr);
            break;
    }
}

// Legacy bool overloads (backward compatible)
void readInterpolated3D(cv::Mat_<uint8_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, bool nearest_neighbor) {
    readInterpolated3DImpl(out, cache, level, coords,
                           nearest_neighbor ? vc::Sampling::Nearest : vc::Sampling::Trilinear);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, bool nearest_neighbor) {
    readInterpolated3DImpl(out, cache, level, coords,
                           nearest_neighbor ? vc::Sampling::Nearest : vc::Sampling::Trilinear);
}

// New overloads accepting vc::Sampling enum
void readInterpolated3D(cv::Mat_<uint8_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, vc::Sampling method) {
    readInterpolated3DImpl(out, cache, level, coords, method);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, vc::Sampling method) {
    readInterpolated3DImpl(out, cache, level, coords, method);
}


void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    vc::Sampling method)
{
    CacheParams p(cache, level);

    const bool hasNormals = !normals.empty() && normals.size() == baseCoords.size();
    const int numLayers = zEnd - zStart + 1;

    auto getNormal = [&](int y, int x) -> cv::Vec3f {
        if (hasNormals) {
            const cv::Vec3f& n = normals(y, x);
            if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                return n;
            }
        }
        return {1, 0, 0};
    };

    switch (method) {
        case vc::Sampling::Trilinear:
            readVolumeImpl<uint8_t, SampleMode::Trilinear>(out, *cache, level, p, baseCoords,
                getNormal, numLayers, zStep, zStart, &params);
            break;
        case vc::Sampling::Tricubic:
            readVolumeImpl<uint8_t, SampleMode::Tricubic>(out, *cache, level, p, baseCoords,
                getNormal, numLayers, zStep, zStart, &params);
            break;
        default:
            readVolumeImpl<uint8_t, SampleMode::Nearest>(out, *cache, level, p, baseCoords,
                getNormal, numLayers, zStep, zStart, &params);
            break;
    }
}


// ============================================================================
// readMultiSlice — bulk multi-slice trilinear sampling
// ============================================================================

template<typename T>
static void readMultiSliceImpl(
    std::vector<cv::Mat_<T>>& out,
    vc::cache::TieredChunkCache& cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    CacheParams p(&cache, level);
    const int h = basePoints.rows;
    const int w = basePoints.cols;
    const int numSlices = static_cast<int>(offsets.size());

    out.resize(numSlices);
    for (int s = 0; s < numSlices; s++)
        out[s] = cv::Mat_<T>(basePoints.size(), 0);

    if (numSlices == 0) return;

    // Phase 1: Discover needed chunks and prefetch in parallel.
    {
        const size_t totalChunks = static_cast<size_t>(p.chunksZ) * p.chunksY * p.chunksX;
        std::vector<uint8_t> needed(totalChunks, 0);
        int minIz = p.chunksZ, maxIz = -1;
        int minIy = p.chunksY, maxIy = -1;
        int minIx = p.chunksX, maxIx = -1;

        auto markVoxel = [&](int iz, int iy, int ix) {
            if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx) return;
            int ciz = p.chunkZ(iz);
            int ciy = p.chunkY(iy);
            int cix = p.chunkX(ix);
            size_t idx = static_cast<size_t>(ciz) * p.chunksY * p.chunksX + ciy * p.chunksX + cix;
            if (!needed[idx]) {
                needed[idx] = 1;
                minIz = std::min(minIz, ciz); maxIz = std::max(maxIz, ciz);
                minIy = std::min(minIy, ciy); maxIy = std::max(maxIy, ciy);
                minIx = std::min(minIx, cix); maxIx = std::max(maxIx, cix);
            }
        };

        const float fOff = offsets[0];
        const float lOff = offsets[numSlices - 1];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bp = basePoints(y, x);
                const cv::Vec3f& sd = stepDirs(y, x);
                if (std::isnan(bp[0])) continue;

                for (float off : {fOff, lOff}) {
                    float vx = bp[0] + sd[0] * off;
                    float vy = bp[1] + sd[1] * off;
                    float vz = bp[2] + sd[2] * off;
                    int iz = static_cast<int>(vz);
                    int iy = static_cast<int>(vy);
                    int ix = static_cast<int>(vx);
                    for (int dz = 0; dz <= 1; dz++)
                        for (int dy = 0; dy <= 1; dy++)
                            for (int dx = 0; dx <= 1; dx++)
                                markVoxel(iz+dz, iy+dy, ix+dx);
                }
            }
        }

        // Collect needed chunk keys
        std::vector<vc::cache::ChunkKey> neededKeys;
        for (int ciz = minIz; ciz <= maxIz; ciz++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int cix = minIx; cix <= maxIx; cix++)
                    if (needed[static_cast<size_t>(ciz) * p.chunksY * p.chunksX + ciy * p.chunksX + cix])
                        neededKeys.push_back(vc::cache::ChunkKey{level, ciz, ciy, cix});

        // Submit async prefetch so IO+decode pipelines across all chunks.
        cache.prefetch(neededKeys);

        // Block until all chunks are available.
        bool anyMissing = false;
        for (auto& key : neededKeys) {
            if (!cache.get(key)) {
                anyMissing = true;
                break;
            }
        }
        if (anyMissing) {
            for (auto& key : neededKeys)
                (void)cache.getBlocking(key);
        }
    }

    // Phase 2: Sample (all chunks for this band already cached).
    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;
    const float firstOff = offsets[0];
    const float lastOff = offsets[numSlices - 1];
    const int lsz = p.sz, lsy = p.sy, lsx = p.sx;

    {
        ChunkSampler<T, 2> sampler(p, cache, level);
        const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bp = basePoints(y, x);
                const cv::Vec3f& sd = stepDirs(y, x);
                if (std::isnan(bp[0])) continue;

                float vx0 = bp[0] + sd[0] * firstOff;
                float vy0 = bp[1] + sd[1] * firstOff;
                float vz0 = bp[2] + sd[2] * firstOff;
                float vx1 = bp[0] + sd[0] * lastOff;
                float vy1 = bp[1] + sd[1] * lastOff;
                float vz1 = bp[2] + sd[2] * lastOff;

                float minVz = std::min(vz0, vz1);
                float maxVz = std::max(vz0, vz1);
                float minVy = std::min(vy0, vy1);
                float maxVy = std::max(vy0, vy1);
                float minVx = std::min(vx0, vx1);
                float maxVx = std::max(vx0, vx1);

                int izMin = static_cast<int>(minVz);
                int izMax = static_cast<int>(maxVz) + 1;
                int iyMin = static_cast<int>(minVy);
                int iyMax = static_cast<int>(maxVy) + 1;
                int ixMin = static_cast<int>(minVx);
                int ixMax = static_cast<int>(maxVx) + 1;

                bool allInBounds = minVz >= 0 && minVy >= 0 && minVx >= 0 &&
                                   izMax < lsz && iyMax < lsy && ixMax < lsx &&
                                   izMin >= 0 && iyMin >= 0 && ixMin >= 0;

                bool singleChunk = allInBounds &&
                    p.chunkZ(izMin) == p.chunkZ(izMax) &&
                    p.chunkY(iyMin) == p.chunkY(iyMax) &&
                    p.chunkX(ixMin) == p.chunkX(ixMax);

                if (singleChunk) {
                    int ciz = p.chunkZ(izMin);
                    int ciy = p.chunkY(iyMin);
                    int cix = p.chunkX(ixMin);
                    sampler.updateChunk(ciz, ciy, cix);
                    const T* __restrict__ d = sampler.data;
                    if (!d) continue;

                    for (int si = 0; si < numSlices; si++) {
                        float off = offsets[si];
                        float vx = bp[0] + sd[0] * off;
                        float vy = bp[1] + sd[1] * off;
                        float vz = bp[2] + sd[2] * off;

                        int iz = static_cast<int>(vz);
                        int iy = static_cast<int>(vy);
                        int ix = static_cast<int>(vx);

                        int lz0 = p.localZ(iz), ly0 = p.localY(iy), lx0 = p.localX(ix);
                        float c000 = d[vo(lz0, ly0, lx0)];
                        float c100 = d[vo(lz0+1, ly0, lx0)];
                        float c010 = d[vo(lz0, ly0+1, lx0)];
                        float c110 = d[vo(lz0+1, ly0+1, lx0)];
                        float c001 = d[vo(lz0, ly0, lx0+1)];
                        float c101 = d[vo(lz0+1, ly0, lx0+1)];
                        float c011 = d[vo(lz0, ly0+1, lx0+1)];
                        float c111 = d[vo(lz0+1, ly0+1, lx0+1)];

                        float fz = vz - iz;
                        float fy = vy - iy;
                        float fx = vx - ix;

                        float c00 = std::fma(fx, c001 - c000, c000);
                        float c01 = std::fma(fx, c011 - c010, c010);
                        float c10 = std::fma(fx, c101 - c100, c100);
                        float c11 = std::fma(fx, c111 - c110, c110);

                        float c0 = std::fma(fy, c01 - c00, c00);
                        float c1 = std::fma(fy, c11 - c10, c10);

                        float v = std::fma(fz, c1 - c0, c0);
                        v = std::max(0.f, std::min(maxVal, v + 0.5f));
                        out[si](y, x) = static_cast<T>(v);
                    }
                } else {
                    for (int si = 0; si < numSlices; si++) {
                        float off = offsets[si];
                        float vx = bp[0] + sd[0] * off;
                        float vy = bp[1] + sd[1] * off;
                        float vz = bp[2] + sd[2] * off;

                        if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < lsz - 1 && vy < lsy - 1 && vx < lsx - 1))
                            continue;

                        float v = sampler.sampleTrilinearUnchecked(vz, vy, vx);
                        v = std::max(0.f, std::min(maxVal, v + 0.5f));
                        out[si](y, x) = static_cast<T>(v);
                    }
                }
            }
        }
    }

}

void readMultiSlice(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    readMultiSliceImpl(out, *cache, level, basePoints, stepDirs, offsets);
}

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    readMultiSliceImpl(out, *cache, level, basePoints, stepDirs, offsets);
}


// ============================================================================
// sampleTileSlices — single-threaded per-tile multi-slice sampler
// Called from within an OMP thread; no internal OMP parallelism.
// Same trilinear math as readMultiSliceImpl Phase 2.
// ============================================================================

template<typename T>
static void sampleTileSlicesImpl(
    std::vector<cv::Mat_<T>>& out,
    vc::cache::TieredChunkCache& cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    CacheParams p(&cache, level);
    const int h = basePoints.rows;
    const int w = basePoints.cols;
    const int numSlices = static_cast<int>(offsets.size());

    out.resize(numSlices);
    for (int s = 0; s < numSlices; s++)
        out[s] = cv::Mat_<T>(basePoints.size(), 0);

    if (numSlices == 0) return;

    // Phase 1: Discover needed chunks and prefetch serially.
    {
        std::vector<std::array<int,3>> neededChunks;
        neededChunks.reserve(16);

        std::unordered_set<uint64_t> seen;
        seen.reserve(16);
        auto markVoxel = [&](int iz, int iy, int ix) {
            if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx) return;
            int ciz = p.chunkZ(iz);
            int ciy = p.chunkY(iy);
            int cix = p.chunkX(ix);
            uint64_t key = (uint64_t(ciz) << 40) | (uint64_t(ciy) << 20) | uint64_t(cix);
            if (!seen.insert(key).second) return;
            neededChunks.push_back({ciz, ciy, cix});
        };

        const float fOff = offsets[0];
        const float lOff = offsets[numSlices - 1];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bp = basePoints(y, x);
                const cv::Vec3f& sd = stepDirs(y, x);
                if (std::isnan(bp[0])) continue;

                for (float off : {fOff, lOff}) {
                    float vx = bp[0] + sd[0] * off;
                    float vy = bp[1] + sd[1] * off;
                    float vz = bp[2] + sd[2] * off;
                    int iz = static_cast<int>(vz);
                    int iy = static_cast<int>(vy);
                    int ix = static_cast<int>(vx);
                    for (int dz = 0; dz <= 1; dz++)
                        for (int dy = 0; dy <= 1; dy++)
                            for (int dx = 0; dx <= 1; dx++)
                                markVoxel(iz+dz, iy+dy, ix+dx);
                }
            }
        }

        // Submit async prefetch so IO+decode pipelines across all chunks,
        // then block. Even though we're on an OMP thread, the async submit
        // lets the IO pool start fetching all chunks in parallel.
        {
            std::vector<vc::cache::ChunkKey> keys;
            keys.reserve(neededChunks.size());
            for (auto& c : neededChunks)
                keys.push_back(vc::cache::ChunkKey{level, c[0], c[1], c[2]});
            cache.prefetch(keys);
        }
        for (auto& c : neededChunks)
            (void)cache.getBlocking(vc::cache::ChunkKey{level, c[0], c[1], c[2]});
    }

    // Phase 2: Sample (single-threaded, all chunks already cached).
    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;
    const float firstOff = offsets[0];
    const float lastOff = offsets[numSlices - 1];
    const int lsz = p.sz, lsy = p.sy, lsx = p.sx;

    ChunkSampler<T, 8> sampler(p, cache, level);
    const auto vo = [&](int lz, int ly, int lx) { return sampler.voxelOffset(lz, ly, lx); };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const cv::Vec3f& bp = basePoints(y, x);
            const cv::Vec3f& sd = stepDirs(y, x);
            if (std::isnan(bp[0])) continue;

            float vx0 = bp[0] + sd[0] * firstOff;
            float vy0 = bp[1] + sd[1] * firstOff;
            float vz0 = bp[2] + sd[2] * firstOff;
            float vx1 = bp[0] + sd[0] * lastOff;
            float vy1 = bp[1] + sd[1] * lastOff;
            float vz1 = bp[2] + sd[2] * lastOff;

            float minVz = std::min(vz0, vz1);
            float maxVz = std::max(vz0, vz1);
            float minVy = std::min(vy0, vy1);
            float maxVy = std::max(vy0, vy1);
            float minVx = std::min(vx0, vx1);
            float maxVx = std::max(vx0, vx1);

            int izMin = static_cast<int>(minVz);
            int izMax = static_cast<int>(maxVz) + 1;
            int iyMin = static_cast<int>(minVy);
            int iyMax = static_cast<int>(maxVy) + 1;
            int ixMin = static_cast<int>(minVx);
            int ixMax = static_cast<int>(maxVx) + 1;

            bool allInBounds = minVz >= 0 && minVy >= 0 && minVx >= 0 &&
                               izMax < lsz && iyMax < lsy && ixMax < lsx &&
                               izMin >= 0 && iyMin >= 0 && ixMin >= 0;

            bool singleChunk = allInBounds &&
                p.chunkZ(izMin) == p.chunkZ(izMax) &&
                p.chunkY(iyMin) == p.chunkY(iyMax) &&
                p.chunkX(ixMin) == p.chunkX(ixMax);

            if (singleChunk) {
                int ciz = p.chunkZ(izMin);
                int ciy = p.chunkY(iyMin);
                int cix = p.chunkX(ixMin);
                sampler.updateChunk(ciz, ciy, cix);
                const T* __restrict__ d = sampler.data;
                if (!d) continue;

                for (int si = 0; si < numSlices; si++) {
                    float off = offsets[si];
                    float vx = bp[0] + sd[0] * off;
                    float vy = bp[1] + sd[1] * off;
                    float vz = bp[2] + sd[2] * off;

                    int iz = static_cast<int>(vz);
                    int iy = static_cast<int>(vy);
                    int ix = static_cast<int>(vx);

                    int lz0 = p.localZ(iz), ly0 = p.localY(iy), lx0 = p.localX(ix);
                    float c000 = d[vo(lz0, ly0, lx0)];
                    float c100 = d[vo(lz0+1, ly0, lx0)];
                    float c010 = d[vo(lz0, ly0+1, lx0)];
                    float c110 = d[vo(lz0+1, ly0+1, lx0)];
                    float c001 = d[vo(lz0, ly0, lx0+1)];
                    float c101 = d[vo(lz0+1, ly0, lx0+1)];
                    float c011 = d[vo(lz0, ly0+1, lx0+1)];
                    float c111 = d[vo(lz0+1, ly0+1, lx0+1)];

                    float fz = vz - iz;
                    float fy = vy - iy;
                    float fx = vx - ix;

                    float c00 = std::fma(fx, c001 - c000, c000);
                    float c01 = std::fma(fx, c011 - c010, c010);
                    float c10 = std::fma(fx, c101 - c100, c100);
                    float c11 = std::fma(fx, c111 - c110, c110);

                    float c0 = std::fma(fy, c01 - c00, c00);
                    float c1 = std::fma(fy, c11 - c10, c10);

                    float v = std::fma(fz, c1 - c0, c0);
                    v = std::max(0.f, std::min(maxVal, v + 0.5f));
                    out[si](y, x) = static_cast<T>(v);
                }
            } else {
                for (int si = 0; si < numSlices; si++) {
                    float off = offsets[si];
                    float vx = bp[0] + sd[0] * off;
                    float vy = bp[1] + sd[1] * off;
                    float vz = bp[2] + sd[2] * off;

                    if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < lsz - 1 && vy < lsy - 1 && vx < lsx - 1))
                        continue;

                    float v = sampler.sampleTrilinearUnchecked(vz, vy, vx);
                    v = std::max(0.f, std::min(maxVal, v + 0.5f));
                    out[si](y, x) = static_cast<T>(v);
                }
            }
        }
    }
}

void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    sampleTileSlicesImpl(out, *cache, level, basePoints, stepDirs, offsets);
}

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    sampleTileSlicesImpl(out, *cache, level, basePoints, stepDirs, offsets);
}

