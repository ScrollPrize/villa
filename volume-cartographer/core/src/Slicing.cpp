#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/types/Sampling.hpp"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "vc/core/types/VcDataset.hpp"
#include "vc/core/cache/ChunkData.hpp"

#include "z5/dataset.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <limits>
#include <unordered_set>


// ============================================================================
// CacheParams — extract dataset constants once (ZYX ordering)
// ============================================================================

template<typename T>
struct CacheParams {
    int cz, cy, cx, sz, sy, sx;
    int czShift, cyShift, cxShift, czMask, cyMask, cxMask;
    int chunksZ, chunksY, chunksX;

    explicit CacheParams(z5::Dataset* ds) {
        const auto& cs = ds->defaultChunkShape();
        cz = static_cast<int>(cs[0]);
        cy = static_cast<int>(cs[1]);
        cx = static_cast<int>(cs[2]);
        const auto& shape = ds->shape();
        sz = static_cast<int>(shape[0]);
        sy = static_cast<int>(shape[1]);
        sx = static_cast<int>(shape[2]);
        czShift = log2_pow2(cz); cyShift = log2_pow2(cy); cxShift = log2_pow2(cx);
        czMask = cz - 1; cyMask = cy - 1; cxMask = cx - 1;
        chunksZ = (sz + cz - 1) / cz;
        chunksY = (sy + cy - 1) / cy;
        chunksX = (sx + cx - 1) / cx;
    }

    static int log2_pow2(int v) {
        int r = 0;
        while ((v >> r) > 1) r++;
        return r;
    }
};


// ============================================================================
// ChunkSampler — thread-local fast voxel access via raw pointer + strides
// ============================================================================

template<typename T, int kSlots = 8>
struct ChunkSampler {
    struct Slot {
        int iz = -1, iy = -1, ix = -1;
        typename ChunkCache<T>::ChunkPtr chunk;
        const T* data = nullptr;
    };

    const CacheParams<T>& p;
    ChunkCache<T>& cache;
    z5::Dataset* ds;
    Slot slots[kSlots];
    int mru = 0;  // most-recently-used slot index
    const T* data = nullptr;  // current data pointer
    size_t s0 = 0, s1 = 0;   // strides (s2 is always 1, eliminated)

    ChunkSampler(const CacheParams<T>& p_, ChunkCache<T>& cache_, z5::Dataset* ds_)
        : p(p_), cache(cache_), ds(ds_)
    {
        s0 = static_cast<size_t>(p.cy) * p.cx;
        s1 = static_cast<size_t>(p.cx);
    }

    void updateChunk(int iz, int iy, int ix) {
        // Check MRU slot first
        auto& m = slots[mru];
        if (m.iz == iz && m.iy == iy && m.ix == ix) {
            data = m.data;
            return;
        }
        // Linear scan remaining slots
        for (int i = 0; i < kSlots; i++) {
            if (i == mru) continue;
            auto& s = slots[i];
            if (s.iz == iz && s.iy == iy && s.ix == ix) {
                mru = i;
                data = s.data;
                return;
            }
        }
        // Miss: evict LRU (slot furthest from mru in ring)
        int victim = (mru + 1) % kSlots;
        auto& v = slots[victim];
        v.chunk = cache.get(ds, iz, iy, ix);
        v.iz = iz; v.iy = iy; v.ix = ix;
        v.data = v.chunk ? v.chunk->data() : nullptr;
        mru = victim;
        data = v.data;
    }

    bool inBounds(float vz, float vy, float vx) const {
        return vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx;
    }

    T sampleNearest(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz + 0.5f);
        int iy = static_cast<int>(vy + 0.5f);
        int ix = static_cast<int>(vx + 0.5f);
        if (iz >= p.sz) iz = p.sz - 1;
        if (iy >= p.sy) iy = p.sy - 1;
        if (ix >= p.sx) ix = p.sx - 1;

        updateChunk(iz >> p.czShift, iy >> p.cyShift, ix >> p.cxShift);
        if (!data) return 0;

        return data[(iz & p.czMask) * s0 + (iy & p.cyMask) * s1 + (ix & p.cxMask)];
    }

    T sampleInt(int iz, int iy, int ix) {
        if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx)
            return 0;

        updateChunk(iz >> p.czShift, iy >> p.cyShift, ix >> p.cxShift);
        if (!data) return 0;

        return data[(iz & p.czMask) * s0 + (iy & p.cyMask) * s1 + (ix & p.cxMask)];
    }

    float sampleTrilinear(float vz, float vy, float vx) {
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

        float c00 = (1 - fx) * c000 + fx * c001;
        float c01 = (1 - fx) * c010 + fx * c011;
        float c10 = (1 - fx) * c100 + fx * c101;
        float c11 = (1 - fx) * c110 + fx * c111;

        float c0 = (1 - fy) * c00 + fy * c01;
        float c1 = (1 - fy) * c10 + fy * c11;

        return (1 - fz) * c0 + fz * c1;
    }

    // Fast path: when all 8 trilinear corners are in the same chunk
    float sampleTrilinearFast(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz);
        int iy = static_cast<int>(vy);
        int ix = static_cast<int>(vx);

        if (iz < 0 || iy < 0 || ix < 0 ||
            iz + 1 >= p.sz || iy + 1 >= p.sy || ix + 1 >= p.sx)
            return sampleTrilinear(vz, vy, vx);

        // Local copies of shifts for register allocation
        const int lczShift = p.czShift, lcyShift = p.cyShift, lcxShift = p.cxShift;
        int ciz0 = iz >> lczShift, ciz1 = (iz + 1) >> lczShift;
        int ciy0 = iy >> lcyShift, ciy1 = (iy + 1) >> lcyShift;
        int cix0 = ix >> lcxShift, cix1 = (ix + 1) >> lcxShift;

        if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
            updateChunk(ciz0, ciy0, cix0);
            if (!data) return 0;

            // Local stride copies — s2 is always 1, eliminated
            const size_t ls0 = s0, ls1 = s1;
            const int lczMask = p.czMask, lcyMask = p.cyMask, lcxMask = p.cxMask;
            int lz0 = iz & lczMask, ly0 = iy & lcyMask, lx0 = ix & lcxMask;
            int lz1 = lz0 + 1, ly1 = ly0 + 1, lx1 = lx0 + 1;

            float c000 = data[lz0*ls0 + ly0*ls1 + lx0];
            float c100 = data[lz1*ls0 + ly0*ls1 + lx0];
            float c010 = data[lz0*ls0 + ly1*ls1 + lx0];
            float c110 = data[lz1*ls0 + ly1*ls1 + lx0];
            float c001 = data[lz0*ls0 + ly0*ls1 + lx1];
            float c101 = data[lz1*ls0 + ly0*ls1 + lx1];
            float c011 = data[lz0*ls0 + ly1*ls1 + lx1];
            float c111 = data[lz1*ls0 + ly1*ls1 + lx1];

            float fz = vz - iz;
            float fy = vy - iy;
            float fx = vx - ix;

            float c00 = (1 - fx) * c000 + fx * c001;
            float c01 = (1 - fx) * c010 + fx * c011;
            float c10 = (1 - fx) * c100 + fx * c101;
            float c11 = (1 - fx) * c110 + fx * c111;

            float c0 = (1 - fy) * c00 + fy * c01;
            float c1 = (1 - fy) * c10 + fy * c11;

            return (1 - fz) * c0 + fz * c1;
        }

        return sampleTrilinear(vz, vy, vx);
    }
};


// ============================================================================
// readVolumeImpl — unified inner loop
// ============================================================================

enum class SampleMode { Nearest, Trilinear };

template<typename T, SampleMode Mode, typename NormalFn>
static void readVolumeImpl(
    cv::Mat_<T>& out,
    z5::Dataset* ds,
    ChunkCache<T>& cache,
    const CacheParams<T>& p,
    const cv::Mat_<cv::Vec3f>& coords,
    NormalFn getNormal,
    int numLayers,
    float zStep,
    int zStart,
    const CompositeParams* params)
{
    const int h = coords.rows;
    const int w = coords.cols;

    out = cv::Mat_<T>(coords.size(), 0);

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
            int ciz = iz >> p.czShift;
            int ciy = iy >> p.cyShift;
            int cix = ix >> p.cxShift;
            size_t idx = ciz * p.chunksY * p.chunksX + ciy * p.chunksX + cix;
            if (!needed[idx]) {
                needed[idx] = 1;
                minIz = std::min(minIz, ciz); maxIz = std::max(maxIz, ciz);
                minIy = std::min(minIy, ciy); maxIy = std::max(maxIy, ciy);
                minIx = std::min(minIx, cix); maxIx = std::max(maxIx, cix);
            }
        };

        if (numLayers == 1) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float vz = coords(y,x)[2], vy = coords(y,x)[1], vx = coords(y,x)[0];
                    if constexpr (Mode == SampleMode::Trilinear) {
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

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    const cv::Vec3f& bc = coords(y, x);
                    float bz = bc[2], by = bc[1], bx = bc[0];
                    if (!(bz >= 0 && by >= 0 && bx >= 0 && bz < p.sz && by < p.sy)) continue;
                    cv::Vec3f n = getNormal(y, x);
                    for (int l = 0; l < numLayers; l++) {
                        float off = layerOffsets[l];
                        markVoxel(bz + n[2]*off, by + n[1]*off, bx + n[0]*off);
                    }
                }
            }
        }

        // Collect needed chunk indices for parallel loading
        std::vector<std::array<int,3>> neededChunks;
        for (int cix = minIx; cix <= maxIx; cix++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int ciz = minIz; ciz <= maxIz; ciz++)
                    if (needed[ciz * p.chunksY * p.chunksX + ciy * p.chunksX + cix])
                        neededChunks.push_back({ciz, ciy, cix});

        // Load chunks — check if any are uncached before spawning threads.
        bool anyMissing = false;
        for (size_t ci = 0; ci < neededChunks.size(); ci++) {
            if (!cache.getIfCached(ds, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2])) {
                anyMissing = true;
                break;
            }
        }

        if (anyMissing) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t ci = 0; ci < neededChunks.size(); ci++)
                cache.get(ds, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]);
        }
    }

    // Phase 2: Sample (OMP parallel, all chunks already cached)
    const bool isComposite = (numLayers > 1);
    const bool isMin = params && (params->method == "min");
    const bool isMax = params && (params->method == "max");
    const bool isMean = params && (params->method == "mean");
    const bool needsLayerStorage = params && methodRequiresLayerStorage(params->method);
    const float firstLayerOffset = zStart * zStep;

    #pragma omp parallel
    {
        ChunkSampler<T> samplerNoPrefetch(p, cache, ds);
        ChunkSampler<T> samplerPrefetch(p, cache, ds);

        LayerStack stack;
        if (needsLayerStorage) {
            stack.values.resize(numLayers);
        }

        #pragma omp for collapse(2)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bc = coords(y, x);
                float base_vz = bc[2], base_vy = bc[1], base_vx = bc[0];

                if (numLayers == 1) {
                    // Single-sample path
                    if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0)) continue;

                    if constexpr (Mode == SampleMode::Trilinear) {
                        if (!(base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        float v = samplerNoPrefetch.sampleTrilinearFast(base_vz, base_vy, base_vx);
                        if constexpr (std::is_same_v<T, uint16_t>) {
                            if (v < 0.f) v = 0.f;
                            if (v > 65535.f) v = 65535.f;
                            out(y, x) = static_cast<uint16_t>(v + 0.5f);
                        } else {
                            out(y, x) = static_cast<T>(v);
                        }
                    } else {
                        if (!(base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        if ((static_cast<int>(base_vz + 0.5f) | static_cast<int>(base_vy + 0.5f) | static_cast<int>(base_vx + 0.5f)) < 0) continue;
                        out(y, x) = samplerNoPrefetch.sampleNearest(base_vz, base_vy, base_vx);
                    }
                } else {
                    // Composite path
                    if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0)) continue;

                    cv::Vec3f n = getNormal(y, x);
                    float nz = n[2], ny = n[1], nx = n[0];

                    float acc = isMin ? 255.0f : 0.0f;
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

                        if (samplerPrefetch.inBounds(vz, vy, vx)) {
                            uint8_t raw = samplerPrefetch.sampleNearest(vz, vy, vx);
                            value = static_cast<float>(raw < params->isoCutoff ? 0 : raw);
                            validSample = true;
                        }

                        vz += dz; vy += dy; vx += dx;

                        if (validSample) {
                            if (needsLayerStorage) {
                                stack.values[stack.validCount++] = value;
                            } else if (isMax) {
                                acc = value > acc ? value : acc;
                                validCount++;
                            } else if (isMin) {
                                acc = value < acc ? value : acc;
                                validCount++;
                            } else {
                                acc += value;
                                validCount++;
                            }
                        }
                    }

                    float result = 0.0f;
                    if (needsLayerStorage) {
                        result = compositeLayerStack(stack, *params);
                    } else if (isMax || isMin) {
                        result = acc;
                    } else if (isMean && validCount > 0) {
                        result = acc / static_cast<float>(validCount);
                    }

                    if (params->lightingEnabled) {
                        cv::Vec3f ln = getNormal(y, x);
                        float lightFactor = computeLightingFactor(ln, *params);
                        result *= lightFactor;
                    }

                    out(y, x) = static_cast<T>(std::max(0.0f, std::min(255.0f, result)));
                }
            }
        }
    }
}


// ============================================================================
// readArea3DImpl
// ============================================================================

template<typename T>
static void readArea3DImpl(xt::xtensor<T, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<T>* cache) {

    CacheParams<T> p(ds);

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
        auto chunkPtr = cache->get(ds, cz, cy, cx);
        xt::xarray<T>* chunk = chunkPtr.get();

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

        if (chunk) {
            for (int z = copy_from_start[0]; z < copy_from_end[0]; ++z) {
                for (int y = copy_from_start[1]; y < copy_from_end[1]; ++y) {
                    for (int x = copy_from_start[2]; x < copy_from_end[2]; ++x) {
                        int lz = z - chunk_offset[0];
                        int ly = y - chunk_offset[1];
                        int lx = x - chunk_offset[2];
                        out(z - offset[0], y - offset[1], x - offset[2]) = (*chunk)(lz, ly, lx);
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

void readArea3D(xt::xtensor<uint8_t, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<uint8_t>* cache) {
    readArea3DImpl(out, offset, ds, cache);
}

void readArea3D(xt::xtensor<uint16_t, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<uint16_t>* cache) {
    readArea3DImpl(out, offset, ds, cache);
}


// ============================================================================
// Public API — thin wrappers around readVolumeImpl
// ============================================================================

template<typename T>
static void readInterpolated3DImpl(cv::Mat_<T>& out, z5::Dataset* ds,
                                   const cv::Mat_<cv::Vec3f>& coords, ChunkCache<T>* cache, bool nearest_neighbor) {
    CacheParams<T> p(ds);

    if (nearest_neighbor) {
        readVolumeImpl<T, SampleMode::Nearest>(out, ds, *cache, p, coords,
            [](int, int) -> cv::Vec3f { return {}; }, 1, 0.f, 0, nullptr);
    } else {
        readVolumeImpl<T, SampleMode::Trilinear>(out, ds, *cache, p, coords,
            [](int, int) -> cv::Vec3f { return {}; }, 1, 0.f, 0, nullptr);
    }
}

void readInterpolated3D(cv::Mat_<uint8_t>& out, z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint8_t>* cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint16_t>* cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}


void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache)
{
    CacheParams<uint8_t> p(ds);

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

    readVolumeImpl<uint8_t, SampleMode::Nearest>(out, ds, cache, p, baseCoords,
        getNormal, numLayers, zStep, zStart, &params);
}


// ============================================================================
// readMultiSlice — bulk multi-slice trilinear sampling
// ============================================================================

template<typename T>
static void readMultiSliceImpl(
    std::vector<cv::Mat_<T>>& out,
    z5::Dataset* ds,
    ChunkCache<T>& cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    CacheParams<T> p(ds);
    const int h = basePoints.rows;
    const int w = basePoints.cols;
    const int numSlices = static_cast<int>(offsets.size());

    out.resize(numSlices);
    for (int s = 0; s < numSlices; s++)
        out[s] = cv::Mat_<T>(basePoints.size(), 0);

    if (numSlices == 0) return;

    // Phase 1: Discover needed chunks and prefetch in parallel.
    // This converts random I/O stalls during OMP sampling into a
    // concentrated upfront I/O burst.
    {
        const size_t totalChunks = static_cast<size_t>(p.chunksZ) * p.chunksY * p.chunksX;
        std::vector<uint8_t> needed(totalChunks, 0);
        int minIz = p.chunksZ, maxIz = -1;
        int minIy = p.chunksY, maxIy = -1;
        int minIx = p.chunksX, maxIx = -1;

        auto markVoxel = [&](int iz, int iy, int ix) {
            if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx) return;
            int ciz = iz >> p.czShift;
            int ciy = iy >> p.cyShift;
            int cix = ix >> p.cxShift;
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

                // Mark chunks for first and last offsets (covers the range)
                for (float off : {fOff, lOff}) {
                    float vx = bp[0] + sd[0] * off;
                    float vy = bp[1] + sd[1] * off;
                    float vz = bp[2] + sd[2] * off;
                    int iz = static_cast<int>(vz);
                    int iy = static_cast<int>(vy);
                    int ix = static_cast<int>(vx);
                    // Mark all 8 trilinear corners
                    for (int dz = 0; dz <= 1; dz++)
                        for (int dy = 0; dy <= 1; dy++)
                            for (int dx = 0; dx <= 1; dx++)
                                markVoxel(iz+dz, iy+dy, ix+dx);
                }
            }
        }

        // Collect and load needed chunks
        std::vector<std::array<int,3>> neededChunks;
        for (int ciz = minIz; ciz <= maxIz; ciz++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int cix = minIx; cix <= maxIx; cix++)
                    if (needed[static_cast<size_t>(ciz) * p.chunksY * p.chunksX + ciy * p.chunksX + cix])
                        neededChunks.push_back({ciz, ciy, cix});

        // Only spawn threads if any chunks are uncached
        bool anyMissing = false;
        for (size_t ci = 0; ci < neededChunks.size(); ci++) {
            if (!cache.getIfCached(ds, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2])) {
                anyMissing = true;
                break;
            }
        }

        if (anyMissing) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t ci = 0; ci < neededChunks.size(); ci++)
                cache.get(ds, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]);
        }
    }

    // Phase 2: Sample (all chunks for this band already cached).
    // Optimization: for each pixel, check if ALL slice samples fall within a
    // single chunk. If so, skip per-sample bounds/chunk checks entirely.

    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;
    const float firstOff = offsets[0];
    const float lastOff = offsets[numSlices - 1];
    const int lczShift = p.czShift, lcyShift = p.cyShift, lcxShift = p.cxShift;
    const int lczMask = p.czMask, lcyMask = p.cyMask, lcxMask = p.cxMask;
    const int lsz = p.sz, lsy = p.sy, lsx = p.sx;

    #pragma omp parallel
    {
        // 2-slot sampler: sequential slice pattern only needs current chunk +
        // occasionally one more for boundary crossings
        ChunkSampler<T, 2> sampler(p, cache, ds);
        const size_t ls0 = sampler.s0, ls1 = sampler.s1;

        #pragma omp for schedule(static)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bp = basePoints(y, x);
                const cv::Vec3f& sd = stepDirs(y, x);
                if (std::isnan(bp[0])) continue;

                // Compute first and last sample positions
                float vx0 = bp[0] + sd[0] * firstOff;
                float vy0 = bp[1] + sd[1] * firstOff;
                float vz0 = bp[2] + sd[2] * firstOff;
                float vx1 = bp[0] + sd[0] * lastOff;
                float vy1 = bp[1] + sd[1] * lastOff;
                float vz1 = bp[2] + sd[2] * lastOff;

                // Determine bounding box of all samples (including +1 for trilinear)
                float minVz = std::min(vz0, vz1);
                float maxVz = std::max(vz0, vz1);
                float minVy = std::min(vy0, vy1);
                float maxVy = std::max(vy0, vy1);
                float minVx = std::min(vx0, vx1);
                float maxVx = std::max(vx0, vx1);

                // Check if all samples are in bounds AND all trilinear corners
                // fall within a single chunk (the common case)
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
                    (izMin >> lczShift) == (izMax >> lczShift) &&
                    (iyMin >> lcyShift) == (iyMax >> lcyShift) &&
                    (ixMin >> lcxShift) == (ixMax >> lcxShift);

                if (singleChunk) {
                    // Fast path: all slices in one chunk. Load chunk once,
                    // then pure trilinear math with no checks.
                    int ciz = izMin >> lczShift;
                    int ciy = iyMin >> lcyShift;
                    int cix = ixMin >> lcxShift;
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

                        size_t base = (iz & lczMask)*ls0 + (iy & lcyMask)*ls1 + (ix & lcxMask);
                        float c000 = d[base];
                        float c100 = d[base + ls0];
                        float c010 = d[base + ls1];
                        float c110 = d[base + ls0 + ls1];
                        float c001 = d[base + 1];
                        float c101 = d[base + ls0 + 1];
                        float c011 = d[base + ls1 + 1];
                        float c111 = d[base + ls0 + ls1 + 1];

                        float fz = vz - iz;
                        float fy = vy - iy;
                        float fx = vx - ix;

                        float c00 = (1-fx)*c000 + fx*c001;
                        float c01 = (1-fx)*c010 + fx*c011;
                        float c10 = (1-fx)*c100 + fx*c101;
                        float c11 = (1-fx)*c110 + fx*c111;

                        float c0 = (1-fy)*c00 + fy*c01;
                        float c1 = (1-fy)*c10 + fy*c11;

                        float v = (1-fz)*c0 + fz*c1;
                        v = std::max(0.f, std::min(maxVal, v + 0.5f));
                        out[si](y, x) = static_cast<T>(v);
                    }
                } else {
                    // Slow path: per-sample bounds/chunk checks
                    for (int si = 0; si < numSlices; si++) {
                        float off = offsets[si];
                        float vx = bp[0] + sd[0] * off;
                        float vy = bp[1] + sd[1] * off;
                        float vz = bp[2] + sd[2] * off;

                        if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < lsz && vy < lsy && vx < lsx))
                            continue;

                        float v = sampler.sampleTrilinearFast(vz, vy, vx);
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
    z5::Dataset* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    readMultiSliceImpl(out, ds, *cache, basePoints, stepDirs, offsets);
}

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    readMultiSliceImpl(out, ds, *cache, basePoints, stepDirs, offsets);
}


// ============================================================================
// sampleTileSlices — single-threaded per-tile multi-slice sampler
// Called from within an OMP thread; no internal OMP parallelism.
// Same trilinear math as readMultiSliceImpl Phase 2.
// ============================================================================

template<typename T>
static void sampleTileSlicesImpl(
    std::vector<cv::Mat_<T>>& out,
    z5::Dataset* ds,
    ChunkCache<T>& cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    CacheParams<T> p(ds);
    const int h = basePoints.rows;
    const int w = basePoints.cols;
    const int numSlices = static_cast<int>(offsets.size());

    out.resize(numSlices);
    for (int s = 0; s < numSlices; s++)
        out[s] = cv::Mat_<T>(basePoints.size(), 0);

    if (numSlices == 0) return;

    // Phase 1: Discover needed chunks and prefetch serially.
    // A 128x128 tile typically touches only ~4-8 chunks, so collect into
    // a compact list rather than scanning the full chunk grid (~39M entries).
    {
        std::vector<std::array<int,3>> neededChunks;
        neededChunks.reserve(16);

        // Small local set: track seen chunks by packing (ciz,ciy,cix) into uint64
        std::vector<uint64_t> seen;
        seen.reserve(16);
        auto markVoxel = [&](int iz, int iy, int ix) {
            if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx) return;
            int ciz = iz >> p.czShift;
            int ciy = iy >> p.cyShift;
            int cix = ix >> p.cxShift;
            uint64_t key = (uint64_t(ciz) << 40) | (uint64_t(ciy) << 20) | uint64_t(cix);
            for (auto k : seen) if (k == key) return;
            seen.push_back(key);
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

        // Serial prefetch — we're already on an OMP thread
        for (auto& c : neededChunks)
            cache.get(ds, c[0], c[1], c[2]);
    }

    // Phase 2: Sample (single-threaded, all chunks already cached).
    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;
    const float firstOff = offsets[0];
    const float lastOff = offsets[numSlices - 1];
    const int lczShift = p.czShift, lcyShift = p.cyShift, lcxShift = p.cxShift;
    const int lczMask = p.czMask, lcyMask = p.cyMask, lcxMask = p.cxMask;
    const int lsz = p.sz, lsy = p.sy, lsx = p.sx;

    ChunkSampler<T, 8> sampler(p, cache, ds);
    const size_t ls0 = sampler.s0, ls1 = sampler.s1;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const cv::Vec3f& bp = basePoints(y, x);
            const cv::Vec3f& sd = stepDirs(y, x);
            if (std::isnan(bp[0])) continue;

            // Compute first and last sample positions
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
                (izMin >> lczShift) == (izMax >> lczShift) &&
                (iyMin >> lcyShift) == (iyMax >> lcyShift) &&
                (ixMin >> lcxShift) == (ixMax >> lcxShift);

            if (singleChunk) {
                int ciz = izMin >> lczShift;
                int ciy = iyMin >> lcyShift;
                int cix = ixMin >> lcxShift;
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

                    size_t base = (iz & lczMask)*ls0 + (iy & lcyMask)*ls1 + (ix & lcxMask);
                    float c000 = d[base];
                    float c100 = d[base + ls0];
                    float c010 = d[base + ls1];
                    float c110 = d[base + ls0 + ls1];
                    float c001 = d[base + 1];
                    float c101 = d[base + ls0 + 1];
                    float c011 = d[base + ls1 + 1];
                    float c111 = d[base + ls0 + ls1 + 1];

                    float fz = vz - iz;
                    float fy = vy - iy;
                    float fx = vx - ix;

                    float c00 = (1-fx)*c000 + fx*c001;
                    float c01 = (1-fx)*c010 + fx*c011;
                    float c10 = (1-fx)*c100 + fx*c101;
                    float c11 = (1-fx)*c110 + fx*c111;

                    float c0 = (1-fy)*c00 + fy*c01;
                    float c1 = (1-fy)*c10 + fy*c11;

                    float v = (1-fz)*c0 + fz*c1;
                    v = std::max(0.f, std::min(maxVal, v + 0.5f));
                    out[si](y, x) = static_cast<T>(v);
                }
            } else {
                for (int si = 0; si < numSlices; si++) {
                    float off = offsets[si];
                    float vx = bp[0] + sd[0] * off;
                    float vy = bp[1] + sd[1] * off;
                    float vz = bp[2] + sd[2] * off;

                    if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < lsz && vy < lsy && vx < lsx))
                        continue;

                    float v = sampler.sampleTrilinearFast(vz, vy, vx);
                    v = std::max(0.f, std::min(maxVal, v + 0.5f));
                    out[si](y, x) = static_cast<T>(v);
                }
            }
        }
    }
}

void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    sampleTileSlicesImpl(out, ds, *cache, basePoints, stepDirs, offsets);
}

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    sampleTileSlicesImpl(out, ds, *cache, basePoints, stepDirs, offsets);
}


cv::Mat_<cv::Vec3f> computeVolumeGradientsNative(
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& rawPoints,
    float dsScale)
{
    const int h = rawPoints.rows;
    const int w = rawPoints.cols;
    cv::Mat_<cv::Vec3f> gradients(h, w, cv::Vec3f(0, 0, 1));

    if (h == 0 || w == 0) return gradients;

    const auto volShape = ds->shape();
    const int volZ = static_cast<int>(volShape[0]);
    const int volY = static_cast<int>(volShape[1]);
    const int volX = static_cast<int>(volShape[2]);

    // Step 1: Find bounding box of all valid coordinates
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f& c = rawPoints(y, x);
            if (c[0] == -1.f) continue;

            const float cx = c[0] * dsScale;
            const float cy = c[1] * dsScale;
            const float cz = c[2] * dsScale;

            minX = std::min(minX, cx);
            minY = std::min(minY, cy);
            minZ = std::min(minZ, cz);
            maxX = std::max(maxX, cx);
            maxY = std::max(maxY, cy);
            maxZ = std::max(maxZ, cz);
        }
    }

    if (minX > maxX) return gradients;

    const int pad = 2;
    const int bboxX0 = std::max(0, static_cast<int>(std::floor(minX)) - pad);
    const int bboxY0 = std::max(0, static_cast<int>(std::floor(minY)) - pad);
    const int bboxZ0 = std::max(0, static_cast<int>(std::floor(minZ)) - pad);
    const int bboxX1 = std::min(volX, static_cast<int>(std::ceil(maxX)) + pad + 1);
    const int bboxY1 = std::min(volY, static_cast<int>(std::ceil(maxY)) + pad + 1);
    const int bboxZ1 = std::min(volZ, static_cast<int>(std::ceil(maxZ)) + pad + 1);

    const size_t localW = static_cast<size_t>(bboxX1 - bboxX0);
    const size_t localH = static_cast<size_t>(bboxY1 - bboxY0);
    const size_t localD = static_cast<size_t>(bboxZ1 - bboxZ0);

    if (localW == 0 || localH == 0 || localD == 0) return gradients;

    // Step 2: Batch read the volume data for the bounding box
    xt::xarray<uint8_t> localVolume = xt::empty<uint8_t>({localD, localH, localW});
    z5::types::ShapeType off = {static_cast<size_t>(bboxZ0), static_cast<size_t>(bboxY0), static_cast<size_t>(bboxX0)};
    z5::multiarray::readSubarray<uint8_t>(*ds, localVolume, off.begin());

    auto sampleLocal = [&](float gx, float gy, float gz) -> float {
        const int lx = static_cast<int>(std::round(gx)) - bboxX0;
        const int ly = static_cast<int>(std::round(gy)) - bboxY0;
        const int lz = static_cast<int>(std::round(gz)) - bboxZ0;

        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= static_cast<int>(localW) ||
            ly >= static_cast<int>(localH) ||
            lz >= static_cast<int>(localD)) {
            return 0.0f;
        }
        return static_cast<float>(localVolume(static_cast<size_t>(lz), static_cast<size_t>(ly), static_cast<size_t>(lx)));
    };

    // Step 3: Compute gradients in parallel at each raw grid point
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f& c = rawPoints(y, x);

            if (c[0] == -1.f) {
                gradients(y, x) = cv::Vec3f(0, 0, 1);
                continue;
            }

            const float cx = c[0] * dsScale;
            const float cy = c[1] * dsScale;
            const float cz = c[2] * dsScale;

            const float v_xp = sampleLocal(cx + 1, cy, cz);
            const float v_xm = sampleLocal(cx - 1, cy, cz);
            const float v_yp = sampleLocal(cx, cy + 1, cz);
            const float v_ym = sampleLocal(cx, cy - 1, cz);
            const float v_zp = sampleLocal(cx, cy, cz + 1);
            const float v_zm = sampleLocal(cx, cy, cz - 1);

            float gx = (v_xp - v_xm) / 2.0f;
            float gy = (v_yp - v_ym) / 2.0f;
            float gz = (v_zp - v_zm) / 2.0f;

            const float len2 = gx*gx + gy*gy + gz*gz;
            if (len2 > 1e-12f) {
                const float len = std::sqrt(len2);
                gradients(y, x) = cv::Vec3f(-gx/len, -gy/len, -gz/len);
            } else {
                gradients(y, x) = cv::Vec3f(0, 0, 1);
            }
        }
    }

    return gradients;
}


// ############################################################################
// TieredChunkCache-based implementations (TC suffix to avoid collision)
// ############################################################################


// ============================================================================
// CacheParamsTC — extract dataset constants once (ZYX ordering)
// ============================================================================

struct CacheParamsTC {
    int cz, cy, cx, sz, sy, sx;
    int czShift, cyShift, cxShift, czMask, cyMask, cxMask;
    int chunksZ, chunksY, chunksX;

    // Chunk index bounds from logical data extent.
    // Chunks outside [dbMinC*, dbMaxC*] are in zero-padded regions.
    int dbMinCz = 0, dbMaxCz = INT_MAX;
    int dbMinCy = 0, dbMaxCy = INT_MAX;
    int dbMinCx = 0, dbMaxCx = INT_MAX;
    bool dbValid = false;

    explicit CacheParamsTC(vc::cache::TieredChunkCache* cache, int level) {
        auto cs = cache->chunkShape(level);
        cz = cs[0]; cy = cs[1]; cx = cs[2];
        auto shape = cache->levelShape(level);
        sz = shape[0]; sy = shape[1]; sx = shape[2];
        czShift = log2_pow2(cz); cyShift = log2_pow2(cy); cxShift = log2_pow2(cx);
        czMask = cz - 1; cyMask = cy - 1; cxMask = cx - 1;
        chunksZ = (sz + cz - 1) / cz;
        chunksY = (sy + cy - 1) / cy;
        chunksX = (sx + cx - 1) / cx;

        // Compute chunk-level data bounds from level-0 bounds.
        auto db = cache->dataBounds();
        if (db.valid) {
            float scale = 1.0f / static_cast<float>(1 << level);
            dbMinCx = std::max(0, (static_cast<int>(std::floor(db.minX * scale)) >> cxShift) - 1);
            dbMaxCx = std::min(chunksX - 1, (static_cast<int>(std::ceil(db.maxX * scale)) >> cxShift) + 1);
            dbMinCy = std::max(0, (static_cast<int>(std::floor(db.minY * scale)) >> cyShift) - 1);
            dbMaxCy = std::min(chunksY - 1, (static_cast<int>(std::ceil(db.maxY * scale)) >> cyShift) + 1);
            dbMinCz = std::max(0, (static_cast<int>(std::floor(db.minZ * scale)) >> czShift) - 1);
            dbMaxCz = std::min(chunksZ - 1, (static_cast<int>(std::ceil(db.maxZ * scale)) >> czShift) + 1);
            dbValid = true;
        }
    }

    static int log2_pow2(int v) {
        int r = 0;
        while ((v >> r) > 1) r++;
        return r;
    }
};


// ============================================================================
// ChunkSamplerTC — thread-local fast voxel access via TieredChunkCache
// ============================================================================

template<typename T, int kSlots = 8>
struct ChunkSamplerTC {
    struct Slot {
        int iz = -1, iy = -1, ix = -1;
        vc::cache::ChunkDataPtr chunk;
        const T* data = nullptr;
    };

    const CacheParamsTC& p;
    vc::cache::TieredChunkCache& cache;
    int level;
    Slot slots[kSlots];
    int mru = 0;
    const T* data = nullptr;
    size_t s0 = 0, s1 = 0;

    ChunkSamplerTC(const CacheParamsTC& p_, vc::cache::TieredChunkCache& cache_, int level_)
        : p(p_), cache(cache_), level(level_)
    {
        s0 = static_cast<size_t>(p.cy) * p.cx;
        s1 = static_cast<size_t>(p.cx);
    }

    void updateChunk(int iz, int iy, int ix) {
        // Quick reject: chunk is entirely in zero-padded region
        if (p.dbValid && (iz < p.dbMinCz || iz > p.dbMaxCz ||
                          iy < p.dbMinCy || iy > p.dbMaxCy ||
                          ix < p.dbMinCx || ix > p.dbMaxCx)) {
            data = nullptr;
            return;
        }
        auto& m = slots[mru];
        if (m.iz == iz && m.iy == iy && m.ix == ix) {
            data = m.data;
            return;
        }
        for (int i = 0; i < kSlots; i++) {
            if (i == mru) continue;
            auto& s = slots[i];
            if (s.iz == iz && s.iy == iy && s.ix == ix) {
                mru = i;
                data = s.data;
                return;
            }
        }
        int victim = (mru + 1) % kSlots;
        auto& v = slots[victim];
        v.chunk = cache.getBlocking(vc::cache::ChunkKey{level, iz, iy, ix});
        v.iz = iz; v.iy = iy; v.ix = ix;
        v.data = v.chunk ? v.chunk->template data<T>() : nullptr;
        mru = victim;
        data = v.data;
    }

    bool inBounds(float vz, float vy, float vx) const {
        return vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx;
    }

    T sampleNearest(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz + 0.5f);
        int iy = static_cast<int>(vy + 0.5f);
        int ix = static_cast<int>(vx + 0.5f);
        if (iz >= p.sz) iz = p.sz - 1;
        if (iy >= p.sy) iy = p.sy - 1;
        if (ix >= p.sx) ix = p.sx - 1;

        updateChunk(iz >> p.czShift, iy >> p.cyShift, ix >> p.cxShift);
        if (!data) return 0;

        return data[(iz & p.czMask) * s0 + (iy & p.cyMask) * s1 + (ix & p.cxMask)];
    }

    T sampleInt(int iz, int iy, int ix) {
        if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx)
            return 0;

        updateChunk(iz >> p.czShift, iy >> p.cyShift, ix >> p.cxShift);
        if (!data) return 0;

        return data[(iz & p.czMask) * s0 + (iy & p.cyMask) * s1 + (ix & p.cxMask)];
    }

    float sampleTrilinear(float vz, float vy, float vx) {
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

    float sampleTrilinearFast(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz);
        int iy = static_cast<int>(vy);
        int ix = static_cast<int>(vx);

        if (iz < 0 || iy < 0 || ix < 0 ||
            iz + 1 >= p.sz || iy + 1 >= p.sy || ix + 1 >= p.sx)
            return sampleTrilinear(vz, vy, vx);

        const int lczShift = p.czShift, lcyShift = p.cyShift, lcxShift = p.cxShift;
        int ciz0 = iz >> lczShift, ciz1 = (iz + 1) >> lczShift;
        int ciy0 = iy >> lcyShift, ciy1 = (iy + 1) >> lcyShift;
        int cix0 = ix >> lcxShift, cix1 = (ix + 1) >> lcxShift;

        if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) {
            updateChunk(ciz0, ciy0, cix0);
            if (!data) return 0;

            const size_t ls0 = s0, ls1 = s1;
            const int lczMask = p.czMask, lcyMask = p.cyMask, lcxMask = p.cxMask;
            int lz0 = iz & lczMask, ly0 = iy & lcyMask, lx0 = ix & lcxMask;
            int lz1 = lz0 + 1, ly1 = ly0 + 1, lx1 = lx0 + 1;

            float c000 = data[lz0*ls0 + ly0*ls1 + lx0];
            float c100 = data[lz1*ls0 + ly0*ls1 + lx0];
            float c010 = data[lz0*ls0 + ly1*ls1 + lx0];
            float c110 = data[lz1*ls0 + ly1*ls1 + lx0];
            float c001 = data[lz0*ls0 + ly0*ls1 + lx1];
            float c101 = data[lz1*ls0 + ly0*ls1 + lx1];
            float c011 = data[lz0*ls0 + ly1*ls1 + lx1];
            float c111 = data[lz1*ls0 + ly1*ls1 + lx1];

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
// SampleModeTC enum + readVolumeImplTC — unified inner loop
// ============================================================================

enum class SampleModeTC { Nearest, Trilinear, Tricubic };

template<typename T, SampleModeTC Mode, typename NormalFn>
static void readVolumeImplTC(
    cv::Mat_<T>& out,
    vc::cache::TieredChunkCache& cache,
    int level,
    const CacheParamsTC& p,
    const cv::Mat_<cv::Vec3f>& coords,
    NormalFn getNormal,
    int numLayers,
    float zStep,
    int zStart,
    const CompositeParams* params)
{
    const int h = coords.rows;
    const int w = coords.cols;

    out = cv::Mat_<T>(coords.size(), 0);

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
            int ciz = iz >> p.czShift;
            int ciy = iy >> p.cyShift;
            int cix = ix >> p.cxShift;
            if (p.dbValid && (ciz < p.dbMinCz || ciz > p.dbMaxCz ||
                              ciy < p.dbMinCy || ciy > p.dbMaxCy ||
                              cix < p.dbMinCx || cix > p.dbMaxCx)) return;
            size_t idx = ciz * p.chunksY * p.chunksX + ciy * p.chunksX + cix;
            if (!needed[idx]) {
                needed[idx] = 1;
                minIz = std::min(minIz, ciz); maxIz = std::max(maxIz, ciz);
                minIy = std::min(minIy, ciy); maxIy = std::max(maxIy, ciy);
                minIx = std::min(minIx, cix); maxIx = std::max(maxIx, cix);
            }
        };

        if (numLayers == 1) {
            for (int y = 0; y < h; y++) {
                const auto* row = coords.ptr<cv::Vec3f>(y);
                for (int x = 0; x < w; x++) {
                    float vx = row[x][0], vy = row[x][1], vz = row[x][2];
                    if constexpr (Mode == SampleModeTC::Tricubic) {
                        if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx)) continue;
                        int iz = static_cast<int>(std::floor(vz)), iy = static_cast<int>(std::floor(vy)), ix = static_cast<int>(std::floor(vx));
                        for (int dz = -1; dz <= 2; dz++)
                            for (int dy = -1; dy <= 2; dy++)
                                for (int dx = -1; dx <= 2; dx++)
                                    markVoxel(float(iz+dz), float(iy+dy), float(ix+dx));
                    } else if constexpr (Mode == SampleModeTC::Trilinear) {
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

            for (int y = 0; y < h; y++) {
                const auto* row = coords.ptr<cv::Vec3f>(y);
                for (int x = 0; x < w; x++) {
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

        std::vector<std::array<int,3>> neededChunks;
        for (int cix = minIx; cix <= maxIx; cix++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int ciz = minIz; ciz <= maxIz; ciz++)
                    if (needed[ciz * p.chunksY * p.chunksX + ciy * p.chunksX + cix])
                        neededChunks.push_back({ciz, ciy, cix});

        bool anyMissing = false;
        for (size_t ci = 0; ci < neededChunks.size(); ci++) {
            if (!cache.get(vc::cache::ChunkKey{level, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]})) {
                anyMissing = true;
                break;
            }
        }

        if (anyMissing) {
            for (size_t ci = 0; ci < neededChunks.size(); ci++)
                (void)cache.getBlocking(vc::cache::ChunkKey{level, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]});
        }
    }

    // Phase 2: Sample (all chunks already cached)
    const bool isMin = params && (params->method == "min");
    const bool isMax = params && (params->method == "max");
    const bool isMean = params && (params->method == "mean");
    const bool needsLayerStorage = params && methodRequiresLayerStorage(params->method);
    const float firstLayerOffset = zStart * zStep;

    {
        ChunkSamplerTC<T> samplerNoPrefetch(p, cache, level);
        ChunkSamplerTC<T> samplerPrefetch(p, cache, level);

        LayerStack stack;
        if (needsLayerStorage) {
            stack.values.resize(numLayers);
        }

        for (int y = 0; y < h; y++) {
            const auto* coordRow = coords.template ptr<cv::Vec3f>(y);
            auto* outRow = out.template ptr<T>(y);
            for (int x = 0; x < w; x++) {
                float base_vx = coordRow[x][0], base_vy = coordRow[x][1], base_vz = coordRow[x][2];

                if (numLayers == 1) {
                    if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0)) continue;

                    if constexpr (Mode == SampleModeTC::Trilinear) {
                        if (!(base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        float v = samplerNoPrefetch.sampleTrilinearFast(base_vz, base_vy, base_vx);
                        if constexpr (std::is_same_v<T, uint16_t>) {
                            if (v < 0.f) v = 0.f;
                            if (v > 65535.f) v = 65535.f;
                            outRow[x] = static_cast<uint16_t>(v + 0.5f);
                        } else {
                            outRow[x] = static_cast<T>(v);
                        }
                    } else if constexpr (Mode == SampleModeTC::Tricubic) {
                        if (!(base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        float v = samplerNoPrefetch.sampleTricubic(base_vz, base_vy, base_vx);
                        if constexpr (std::is_same_v<T, uint16_t>) {
                            if (v < 0.f) v = 0.f;
                            if (v > 65535.f) v = 65535.f;
                            outRow[x] = static_cast<uint16_t>(v + 0.5f);
                        } else {
                            outRow[x] = static_cast<T>(v);
                        }
                    } else {
                        if (!(base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        if ((static_cast<int>(base_vz + 0.5f) | static_cast<int>(base_vy + 0.5f) | static_cast<int>(base_vx + 0.5f)) < 0) continue;
                        outRow[x] = samplerNoPrefetch.sampleNearest(base_vz, base_vy, base_vx);
                    }
                } else {
                    // Composite path
                    if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0)) continue;

                    cv::Vec3f n = getNormal(y, x);
                    float nz = n[2], ny = n[1], nx = n[0];

                    float acc = isMin ? 255.0f : 0.0f;
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

                        if (samplerPrefetch.inBounds(vz, vy, vx)) {
                            uint8_t raw = samplerPrefetch.sampleNearest(vz, vy, vx);
                            value = static_cast<float>(raw < params->isoCutoff ? 0 : raw);
                            validSample = true;
                        }

                        vz += dz; vy += dy; vx += dx;

                        if (validSample) {
                            if (needsLayerStorage) {
                                stack.values[stack.validCount++] = value;
                            } else if (isMax) {
                                acc = value > acc ? value : acc;
                                validCount++;
                            } else if (isMin) {
                                acc = value < acc ? value : acc;
                                validCount++;
                            } else {
                                acc += value;
                                validCount++;
                            }
                        }
                    }

                    float result = 0.0f;
                    if (needsLayerStorage) {
                        result = compositeLayerStack(stack, *params);
                    } else if (isMax || isMin) {
                        result = acc;
                    } else if (isMean && validCount > 0) {
                        result = acc / static_cast<float>(validCount);
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
// readArea3DImplTC
// ============================================================================

template<typename T>
static void readArea3DImplTC(xt::xtensor<T, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level) {

    CacheParamsTC p(cache, level);

    cv::Vec3i size = {(int)out.shape()[0], (int)out.shape()[1], (int)out.shape()[2]};
    cv::Vec3i to = offset + size;

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
            int strideZ = p.cy * p.cx;
            int strideY = p.cx;
            for (int z = copy_from_start[0]; z < copy_from_end[0]; ++z) {
                for (int y = copy_from_start[1]; y < copy_from_end[1]; ++y) {
                    for (int x = copy_from_start[2]; x < copy_from_end[2]; ++x) {
                        int lz = z - chunk_offset[0];
                        int ly = y - chunk_offset[1];
                        int lx = x - chunk_offset[2];
                        out(z - offset[0], y - offset[1], x - offset[2]) = chunkData[lz * strideZ + ly * strideY + lx];
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


// ============================================================================
// TC Public API — thin wrappers
// ============================================================================

template<typename T>
static void readInterpolated3DImplTC(cv::Mat_<T>& out, vc::cache::TieredChunkCache* cache, int level,
                                     const cv::Mat_<cv::Vec3f>& coords,
                                     vc::Sampling method) {
    CacheParamsTC p(cache, level);
    auto noNormal = [](int, int) -> cv::Vec3f { return {}; };

    switch (method) {
        case vc::Sampling::Nearest:
            readVolumeImplTC<T, SampleModeTC::Nearest>(out, *cache, level, p, coords,
                noNormal, 1, 0.f, 0, nullptr);
            break;
        case vc::Sampling::Tricubic:
            readVolumeImplTC<T, SampleModeTC::Tricubic>(out, *cache, level, p, coords,
                noNormal, 1, 0.f, 0, nullptr);
            break;
        default:
            readVolumeImplTC<T, SampleModeTC::Trilinear>(out, *cache, level, p, coords,
                noNormal, 1, 0.f, 0, nullptr);
            break;
    }
}

void readInterpolated3D(cv::Mat_<uint8_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, bool nearest_neighbor) {
    readInterpolated3DImplTC(out, cache, level, coords,
                             nearest_neighbor ? vc::Sampling::Nearest : vc::Sampling::Trilinear);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, bool nearest_neighbor) {
    readInterpolated3DImplTC(out, cache, level, coords,
                             nearest_neighbor ? vc::Sampling::Nearest : vc::Sampling::Trilinear);
}

void readInterpolated3D(cv::Mat_<uint8_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, vc::Sampling method) {
    readInterpolated3DImplTC(out, cache, level, coords, method);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, vc::cache::TieredChunkCache* cache, int level,
                        const cv::Mat_<cv::Vec3f>& coords, vc::Sampling method) {
    readInterpolated3DImplTC(out, cache, level, coords, method);
}


void readArea3D(xt::xtensor<uint8_t, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level) {
    readArea3DImplTC(out, offset, cache, level);
}

void readArea3D(xt::xtensor<uint16_t, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level) {
    readArea3DImplTC(out, offset, cache, level);
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
    CacheParamsTC p(cache, level);

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
            readVolumeImplTC<uint8_t, SampleModeTC::Trilinear>(out, *cache, level, p, baseCoords,
                getNormal, numLayers, zStep, zStart, &params);
            break;
        case vc::Sampling::Tricubic:
            readVolumeImplTC<uint8_t, SampleModeTC::Tricubic>(out, *cache, level, p, baseCoords,
                getNormal, numLayers, zStep, zStart, &params);
            break;
        default:
            readVolumeImplTC<uint8_t, SampleModeTC::Nearest>(out, *cache, level, p, baseCoords,
                getNormal, numLayers, zStep, zStart, &params);
            break;
    }
}


// ============================================================================
// readMultiSliceImplTC — bulk multi-slice trilinear sampling
// ============================================================================

template<typename T>
static void readMultiSliceImplTC(
    std::vector<cv::Mat_<T>>& out,
    vc::cache::TieredChunkCache& cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    CacheParamsTC p(&cache, level);
    const int h = basePoints.rows;
    const int w = basePoints.cols;
    const int numSlices = static_cast<int>(offsets.size());

    out.resize(numSlices);
    for (int s = 0; s < numSlices; s++)
        out[s] = cv::Mat_<T>(basePoints.size(), 0);

    if (numSlices == 0) return;

    // Phase 1: Discover needed chunks and prefetch.
    {
        const size_t totalChunks = static_cast<size_t>(p.chunksZ) * p.chunksY * p.chunksX;
        std::vector<uint8_t> needed(totalChunks, 0);
        int minIz = p.chunksZ, maxIz = -1;
        int minIy = p.chunksY, maxIy = -1;
        int minIx = p.chunksX, maxIx = -1;

        auto markVoxel = [&](int iz, int iy, int ix) {
            if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx) return;
            int ciz = iz >> p.czShift;
            int ciy = iy >> p.cyShift;
            int cix = ix >> p.cxShift;
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

        std::vector<std::array<int,3>> neededChunks;
        for (int ciz = minIz; ciz <= maxIz; ciz++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int cix = minIx; cix <= maxIx; cix++)
                    if (needed[static_cast<size_t>(ciz) * p.chunksY * p.chunksX + ciy * p.chunksX + cix])
                        neededChunks.push_back({ciz, ciy, cix});

        bool anyMissing = false;
        for (size_t ci = 0; ci < neededChunks.size(); ci++) {
            if (!cache.get(vc::cache::ChunkKey{level, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]})) {
                anyMissing = true;
                break;
            }
        }

        if (anyMissing) {
            for (size_t ci = 0; ci < neededChunks.size(); ci++)
                (void)cache.getBlocking(vc::cache::ChunkKey{level, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]});
        }
    }

    // Phase 2: Sample (all chunks already cached).
    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;
    const float firstOff = offsets[0];
    const float lastOff = offsets[numSlices - 1];
    const int lczShift = p.czShift, lcyShift = p.cyShift, lcxShift = p.cxShift;
    const int lczMask = p.czMask, lcyMask = p.cyMask, lcxMask = p.cxMask;
    const int lsz = p.sz, lsy = p.sy, lsx = p.sx;

    {
        ChunkSamplerTC<T, 2> sampler(p, cache, level);
        const size_t ls0 = sampler.s0, ls1 = sampler.s1;

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
                    (izMin >> lczShift) == (izMax >> lczShift) &&
                    (iyMin >> lcyShift) == (iyMax >> lcyShift) &&
                    (ixMin >> lcxShift) == (ixMax >> lcxShift);

                if (singleChunk) {
                    int ciz = izMin >> lczShift;
                    int ciy = iyMin >> lcyShift;
                    int cix = ixMin >> lcxShift;
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

                        size_t base = (iz & lczMask)*ls0 + (iy & lcyMask)*ls1 + (ix & lcxMask);
                        float c000 = d[base];
                        float c100 = d[base + ls0];
                        float c010 = d[base + ls1];
                        float c110 = d[base + ls0 + ls1];
                        float c001 = d[base + 1];
                        float c101 = d[base + ls0 + 1];
                        float c011 = d[base + ls1 + 1];
                        float c111 = d[base + ls0 + ls1 + 1];

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

                        if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < lsz && vy < lsy && vx < lsx))
                            continue;

                        float v = sampler.sampleTrilinearFast(vz, vy, vx);
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
    readMultiSliceImplTC(out, *cache, level, basePoints, stepDirs, offsets);
}

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    readMultiSliceImplTC(out, *cache, level, basePoints, stepDirs, offsets);
}


// ============================================================================
// sampleTileSlicesImplTC — single-threaded per-tile multi-slice sampler
// ============================================================================

template<typename T>
static void sampleTileSlicesImplTC(
    std::vector<cv::Mat_<T>>& out,
    vc::cache::TieredChunkCache& cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    CacheParamsTC p(&cache, level);
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
            int ciz = iz >> p.czShift;
            int ciy = iy >> p.cyShift;
            int cix = ix >> p.cxShift;
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

        // Serial prefetch — we're already on an OMP thread
        for (auto& c : neededChunks)
            (void)cache.getBlocking(vc::cache::ChunkKey{level, c[0], c[1], c[2]});
    }

    // Phase 2: Sample (single-threaded, all chunks already cached).
    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;
    const float firstOff = offsets[0];
    const float lastOff = offsets[numSlices - 1];
    const int lczShift = p.czShift, lcyShift = p.cyShift, lcxShift = p.cxShift;
    const int lczMask = p.czMask, lcyMask = p.cyMask, lcxMask = p.cxMask;
    const int lsz = p.sz, lsy = p.sy, lsx = p.sx;

    ChunkSamplerTC<T, 8> sampler(p, cache, level);
    const size_t ls0 = sampler.s0, ls1 = sampler.s1;

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
                (izMin >> lczShift) == (izMax >> lczShift) &&
                (iyMin >> lcyShift) == (iyMax >> lcyShift) &&
                (ixMin >> lcxShift) == (ixMax >> lcxShift);

            if (singleChunk) {
                int ciz = izMin >> lczShift;
                int ciy = iyMin >> lcyShift;
                int cix = ixMin >> lcxShift;
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

                    size_t base = (iz & lczMask)*ls0 + (iy & lcyMask)*ls1 + (ix & lcxMask);
                    float c000 = d[base];
                    float c100 = d[base + ls0];
                    float c010 = d[base + ls1];
                    float c110 = d[base + ls0 + ls1];
                    float c001 = d[base + 1];
                    float c101 = d[base + ls0 + 1];
                    float c011 = d[base + ls1 + 1];
                    float c111 = d[base + ls0 + ls1 + 1];

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

                    if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < lsz && vy < lsy && vx < lsx))
                        continue;

                    float v = sampler.sampleTrilinearFast(vz, vy, vx);
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
    sampleTileSlicesImplTC(out, *cache, level, basePoints, stepDirs, offsets);
}

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    sampleTileSlicesImplTC(out, *cache, level, basePoints, stepDirs, offsets);
}


// ============================================================================
// computeVolumeGradientsNative — VcDataset overload
// ============================================================================

cv::Mat_<cv::Vec3f> computeVolumeGradientsNative(
    vc::VcDataset* ds,
    const cv::Mat_<cv::Vec3f>& rawPoints,
    float dsScale)
{
    const int h = rawPoints.rows;
    const int w = rawPoints.cols;
    cv::Mat_<cv::Vec3f> gradients(h, w, cv::Vec3f(0, 0, 1));

    if (h == 0 || w == 0) return gradients;

    const auto volShape = ds->shape();
    const int volZ = static_cast<int>(volShape[0]);
    const int volY = static_cast<int>(volShape[1]);
    const int volX = static_cast<int>(volShape[2]);

    // Step 1: Find bounding box of all valid coordinates
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f& c = rawPoints(y, x);
            if (c[0] == -1.f) continue;

            const float cx = c[0] * dsScale;
            const float cy = c[1] * dsScale;
            const float cz = c[2] * dsScale;

            minX = std::min(minX, cx);
            minY = std::min(minY, cy);
            minZ = std::min(minZ, cz);
            maxX = std::max(maxX, cx);
            maxY = std::max(maxY, cy);
            maxZ = std::max(maxZ, cz);
        }
    }

    if (minX > maxX) return gradients;

    const int pad = 2;
    const int bboxX0 = std::max(0, static_cast<int>(std::floor(minX)) - pad);
    const int bboxY0 = std::max(0, static_cast<int>(std::floor(minY)) - pad);
    const int bboxZ0 = std::max(0, static_cast<int>(std::floor(minZ)) - pad);
    const int bboxX1 = std::min(volX, static_cast<int>(std::ceil(maxX)) + pad + 1);
    const int bboxY1 = std::min(volY, static_cast<int>(std::ceil(maxY)) + pad + 1);
    const int bboxZ1 = std::min(volZ, static_cast<int>(std::ceil(maxZ)) + pad + 1);

    const size_t localW = static_cast<size_t>(bboxX1 - bboxX0);
    const size_t localH = static_cast<size_t>(bboxY1 - bboxY0);
    const size_t localD = static_cast<size_t>(bboxZ1 - bboxZ0);

    if (localW == 0 || localH == 0 || localD == 0) return gradients;

    // Step 2: Batch read the volume data for the bounding box
    xt::xarray<uint8_t> localVolume = xt::empty<uint8_t>({localD, localH, localW});
    std::vector<size_t> regionOff = {static_cast<size_t>(bboxZ0), static_cast<size_t>(bboxY0), static_cast<size_t>(bboxX0)};
    std::vector<size_t> regionShape = {localD, localH, localW};
    ds->readRegion(regionOff, regionShape, localVolume.data());

    auto sampleLocal = [&](float gx, float gy, float gz) -> float {
        const int lx = static_cast<int>(std::round(gx)) - bboxX0;
        const int ly = static_cast<int>(std::round(gy)) - bboxY0;
        const int lz = static_cast<int>(std::round(gz)) - bboxZ0;

        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= static_cast<int>(localW) ||
            ly >= static_cast<int>(localH) ||
            lz >= static_cast<int>(localD)) {
            return 0.0f;
        }
        return static_cast<float>(localVolume(static_cast<size_t>(lz), static_cast<size_t>(ly), static_cast<size_t>(lx)));
    };

    // Step 3: Compute gradients at each raw grid point
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f& c = rawPoints(y, x);

            if (c[0] == -1.f) {
                gradients(y, x) = cv::Vec3f(0, 0, 1);
                continue;
            }

            const float cx = c[0] * dsScale;
            const float cy = c[1] * dsScale;
            const float cz = c[2] * dsScale;

            const float v_xp = sampleLocal(cx + 1, cy, cz);
            const float v_xm = sampleLocal(cx - 1, cy, cz);
            const float v_yp = sampleLocal(cx, cy + 1, cz);
            const float v_ym = sampleLocal(cx, cy - 1, cz);
            const float v_zp = sampleLocal(cx, cy, cz + 1);
            const float v_zm = sampleLocal(cx, cy, cz - 1);

            float gx = (v_xp - v_xm) / 2.0f;
            float gy = (v_yp - v_ym) / 2.0f;
            float gz = (v_zp - v_zm) / 2.0f;

            float len = std::sqrt(gx*gx + gy*gy + gz*gz);
            if (len > 1e-6f) {
                gradients(y, x) = cv::Vec3f(-gx/len, -gy/len, -gz/len);
            } else {
                gradients(y, x) = cv::Vec3f(0, 0, 1);
            }
        }
    }

    return gradients;
}
