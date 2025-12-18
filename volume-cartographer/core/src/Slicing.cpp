#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include XTENSORINCLUDE(io, xio.hpp)
#include XTENSORINCLUDE(generators, xbuilder.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/multiarray/xtensor_access.hxx"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <shared_mutex>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <random>


template<typename T>
static xt::xarray<T> *readChunk(const z5::Dataset & ds, const z5::types::ShapeType& chunkId)
{
    if (!ds.chunkExists(chunkId)) {
        return nullptr;
    }

    if (!ds.isZarr())
        throw std::runtime_error("only zarr datasets supported currently!");
    if (ds.getDtype() != z5::types::Datatype::uint8 && ds.getDtype() != z5::types::Datatype::uint16)
        throw std::runtime_error("only uint8_t/uint16 zarrs supported currently!");

    z5::types::ShapeType chunkShape;
    ds.getChunkShape(chunkId, chunkShape);
    const std::size_t maxChunkSize = ds.defaultChunkSize();
    const auto & maxChunkShape = ds.defaultChunkShape();

    xt::xarray<T> *out = new xt::xarray<T>();
    *out = xt::empty<T>(maxChunkShape);

    // Handle based on both dataset dtype and target type T
    if (ds.getDtype() == z5::types::Datatype::uint8) {
        // Dataset is uint8 - direct read for uint8_t, invalid for uint16_t
        if constexpr (std::is_same_v<T, uint8_t>) {
            ds.readChunk(chunkId, out->data());
        } else {
            throw std::runtime_error("Cannot read uint8 dataset into uint16 array");
        }
    }
    else if (ds.getDtype() == z5::types::Datatype::uint16) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            // Dataset is uint16, target is uint16 - direct read
            ds.readChunk(chunkId, out->data());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            // Dataset is uint16, target is uint8 - need conversion
            xt::xarray<uint16_t> tmp = xt::empty<uint16_t>(maxChunkShape);
            ds.readChunk(chunkId, tmp.data());

            uint8_t *p8 = out->data();
            uint16_t *p16 = tmp.data();
            for(size_t i = 0; i < maxChunkSize; i++)
                p8[i] = p16[i] / 257;
        }
    }

    return out;
}



template<typename T>
static void readArea3DImpl(xt::xtensor<T, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<T>* cache) {
    int group_idx = cache->groupIdx(ds->path());
    cv::Vec3i size = {(int)out.shape()[0], (int)out.shape()[1], (int)out.shape()[2]};
    auto chunksize = ds->chunking().blockShape();
    cv::Vec3i to = offset + size;

    // Step 1: List all required chunks
    std::vector<cv::Vec4i> chunks_to_process;
    cv::Vec3i start_chunk = {offset[0] / (int)chunksize[0], offset[1] / (int)chunksize[1], offset[2] / (int)chunksize[2]};
    cv::Vec3i end_chunk = {(to[0] - 1) / (int)chunksize[0], (to[1] - 1) / (int)chunksize[1], (to[2] - 1) / (int)chunksize[2]};

    for (int cz = start_chunk[0]; cz <= end_chunk[0]; ++cz) {
        for (int cy = start_chunk[1]; cy <= end_chunk[1]; ++cy) {
            for (int cx = start_chunk[2]; cx <= end_chunk[2]; ++cx) {
                chunks_to_process.push_back({group_idx, cz, cy, cx});
            }
        }
    }

    // Shuffle to reduce I/O contention from parallel requests
    std::shuffle(chunks_to_process.begin(), chunks_to_process.end(), std::mt19937(std::random_device()()));

    // Step 2 & 3: Combined parallel I/O and copy
    #pragma omp parallel for schedule(dynamic, 1)
    for (const auto& idx : chunks_to_process) {
        std::shared_ptr<xt::xarray<T>> chunk_ref;
        bool needs_read = false;

        {
            std::shared_lock<std::shared_mutex> lock(cache->mutex);
            if (cache->has(idx)) {
                chunk_ref = cache->get(idx);
            } else {
                needs_read = true;
            }
        }

        if (needs_read) {
            auto* new_chunk = readChunk<T>(*ds, {(size_t)idx[1], (size_t)idx[2], (size_t)idx[3]});
            std::unique_lock<std::shared_mutex> lock(cache->mutex);
            if (!cache->has(idx)) {
                cache->put(idx, new_chunk);
            } else {
                delete new_chunk; // Another thread might have cached it in the meantime
            }
            chunk_ref = cache->get(idx);
        }

        int cz = idx[1], cy = idx[2], cx = idx[3];
        cv::Vec3i chunk_offset = {(int)chunksize[0] * cz, (int)chunksize[1] * cy, (int)chunksize[2] * cx};

        cv::Vec3i copy_from_start = {
            std::max(offset[0], chunk_offset[0]),
            std::max(offset[1], chunk_offset[1]),
            std::max(offset[2], chunk_offset[2])
        };

        cv::Vec3i copy_from_end = {
            std::min(to[0], chunk_offset[0] + (int)chunksize[0]),
            std::min(to[1], chunk_offset[1] + (int)chunksize[1]),
            std::min(to[2], chunk_offset[2] + (int)chunksize[2])
        };

        if (chunk_ref) {
            for (int z = copy_from_start[0]; z < copy_from_end[0]; ++z) {
                for (int y = copy_from_start[1]; y < copy_from_end[1]; ++y) {
                    for (int x = copy_from_start[2]; x < copy_from_end[2]; ++x) {
                        int lz = z - chunk_offset[0];
                        int ly = y - chunk_offset[1];
                        int lx = x - chunk_offset[2];
                        out(z - offset[0], y - offset[1], x - offset[2]) = (*chunk_ref)(lz, ly, lx);
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

template<typename T>
static void readNearestNeighborImpl(cv::Mat_<T> &out, const z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<T> *cache) {
    out = cv::Mat_<T>(coords.size(), 0);
    int group_idx = cache->groupIdx(ds->path());

    const auto& blockShape = ds->chunking().blockShape();
    if (blockShape.size() < 3) {
        throw std::runtime_error("Unexpected chunk dimensionality for nearest-neighbor sampling: got " + std::to_string(blockShape.size()));
    }
    const int chunk_size_x = static_cast<int>(blockShape[0]);
    const int chunk_size_y = static_cast<int>(blockShape[1]);
    const int chunk_size_z = static_cast<int>(blockShape[2]);

    if (chunk_size_x <= 0 || chunk_size_y <= 0 || chunk_size_z <= 0) {
        throw std::runtime_error("Invalid chunk dimensions for nearest-neighbor sampling");
    }

    int w = coords.cols;
    int h = coords.rows;

    constexpr int TILE_SIZE = 32;

    #pragma omp parallel
    {
        // Thread-local variables
        cv::Vec4i last_idx = {-1,-1,-1,-1};
        xt::xarray<T> *chunk = nullptr;
        std::shared_ptr<xt::xarray<T>> chunk_ref;

        #pragma omp for schedule(static, 1) collapse(2)
        for(size_t tile_y = 0; tile_y < static_cast<size_t>(h); tile_y += TILE_SIZE) {
            for(size_t tile_x = 0; tile_x < static_cast<size_t>(w); tile_x += TILE_SIZE) {
                size_t y_end = std::min(tile_y + TILE_SIZE, static_cast<size_t>(h));
                size_t x_end = std::min(tile_x + TILE_SIZE, static_cast<size_t>(w));

                for(size_t y = tile_y; y < y_end; y++) {
                    if (y + 1 < y_end) {
                        __builtin_prefetch(&coords(y+1, tile_x), 0, 1);
                    }

                    for(size_t x = tile_x; x < x_end; x++) {
                        int ox = static_cast<int>(coords(y,x)[2] + 0.5f);
                        int oy = static_cast<int>(coords(y,x)[1] + 0.5f);
                        int oz = static_cast<int>(coords(y,x)[0] + 0.5f);

                        if ((ox | oy | oz) < 0)
                            continue;

                        int ix = ox / chunk_size_x;
                        int iy = oy / chunk_size_y;
                        int iz = oz / chunk_size_z;

                        cv::Vec4i idx = {group_idx, ix, iy, iz};

                        if (idx != last_idx) {
                            last_idx = idx;

                            #pragma omp critical(cache_access)
                            {
                                if (!cache->has(idx)) {
                                    auto* new_chunk = readChunk<T>(*ds, {size_t(ix), size_t(iy), size_t(iz)});
                                    cache->put(idx, new_chunk);
                                    chunk_ref = cache->get(idx);
                                } else {
                                    chunk_ref = cache->get(idx);
                                }
                            }
                            chunk = chunk_ref.get();
                        }

                        if (!chunk)
                            continue;

                        int lx = ox - ix * chunk_size_x;
                        int ly = oy - iy * chunk_size_y;
                        int lz = oz - iz * chunk_size_z;

                        if (lx < 0 || ly < 0 || lz < 0 ||
                            lx >= chunk_size_x || ly >= chunk_size_y || lz >= chunk_size_z) {
                            continue;
                        }

                        out(y,x) = chunk->operator()(lx, ly, lz);
                    }
                }
            }
        }
    }
}

void readNearestNeighbor(cv::Mat_<uint8_t> &out, const z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache) {
    readNearestNeighborImpl(out, ds, coords, cache);
}

static void readNearestNeighbor16(cv::Mat_<uint16_t> &out, const z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache) {
    readNearestNeighborImpl(out, ds, coords, cache);
}

template<typename T>
static void readInterpolated3DImpl(cv::Mat_<T> &out, z5::Dataset *ds,
                               const cv::Mat_<cv::Vec3f> &coords, ChunkCache<T> *cache, bool nearest_neighbor) {
    if (nearest_neighbor) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            return readNearestNeighbor(out,ds,coords,cache);
        } else {
            return readNearestNeighbor16(out,ds,coords,cache);
        }
    }

    out = cv::Mat_<T>(coords.size(), 0);

    if (!cache) {
        std::cout << "ERROR should use a shared chunk cache!" << std::endl;
        abort();
    }

    int group_idx = cache->groupIdx(ds->path());

    auto cw = ds->chunking().blockShape()[0];
    auto ch = ds->chunking().blockShape()[1];
    auto cd = ds->chunking().blockShape()[2];

    const auto& dsShape = ds->shape();
    const int sx = static_cast<int>(dsShape[0]);
    const int sy = static_cast<int>(dsShape[1]);
    const int sz = static_cast<int>(dsShape[2]);
    const int chunksX = (sx + static_cast<int>(cw) - 1) / static_cast<int>(cw);
    const int chunksY = (sy + static_cast<int>(ch) - 1) / static_cast<int>(ch);
    const int chunksZ = (sz + static_cast<int>(cd) - 1) / static_cast<int>(cd);

    int w = coords.cols;
    int h = coords.rows;

    std::shared_mutex mutex;
    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<T>>,vec4i_hash> chunks;

    // Lambda for retrieving single values (unchanged)
    auto retrieve_single_value_cached = [&cw,&ch,&cd,&group_idx,&chunks,&sx,&sy,&sz](
        int ox, int oy, int oz) -> T {

            if (ox < 0 || oy < 0 || oz < 0 ||
                ox >= sx || oy >= sy || oz >= sz) {
                return 0;
            }

            int ix = int(ox)/cw;
            int iy = int(oy)/ch;
            int iz = int(oz)/cd;

            cv::Vec4i idx = {group_idx,ix,iy,iz};
            auto it = chunks.find(idx);
            if (it == chunks.end()) {
                return 0;
            }

            xt::xarray<T> *chunk  = it->second.get();

            if (!chunk)
                return 0;

            int lx = ox-ix*cw;
            int ly = oy-iy*ch;
            int lz = oz-iz*cd;

            return chunk->operator()(lx,ly,lz);
        };

        // size_t done = 0;

        #pragma omp parallel
        {
            cv::Vec4i last_idx = {-1,-1,-1,-1};
            std::shared_ptr<xt::xarray<T>> chunk_ref;
            xt::xarray<T> *chunk = nullptr;
            std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<T>>,vec4i_hash> chunks_local;

            #pragma omp for collapse(2)
            for(size_t y = 0;y<h;y++) {
                for(size_t x = 0;x<w;x++) {
                    float ox = coords(y,x)[2];
                    float oy = coords(y,x)[1];
                    float oz = coords(y,x)[0];

                    if (ox < 0 || oy < 0 || oz < 0)
                        continue;

                    if (ox >= sx || oy >= sy || oz >= sz) {
                        continue;
                    }

                    int ix = int(ox)/cw;
                    int iy = int(oy)/ch;
                    int iz = int(oz)/cd;

                    cv::Vec4i idx = {group_idx,ix,iy,iz};

                    if (idx != last_idx) {
                        last_idx = idx;
                        if (ix >= 0 && ix < chunksX &&
                            iy >= 0 && iy < chunksY &&
                            iz >= 0 && iz < chunksZ) {
                            chunks_local[idx] = nullptr;
                        }
                    }

                    int lx = ox-ix*cw;
                    int ly = oy-iy*ch;
                    int lz = oz-iz*cd;

                    if (lx+1 >= cw || ly+1 >= ch || lz+1 >= cd) {
                        if (lx+1>=cw) {
                            cv::Vec4i idx2 = idx;
                            idx2[1]++;
                            if (idx2[1] >= 0 && idx2[1] < chunksX) {
                                chunks_local[idx2] = nullptr;
                            }
                        }
                        if (ly+1>=ch) {
                            cv::Vec4i idx2 = idx;
                            idx2[2]++;
                            if (idx2[2] >= 0 && idx2[2] < chunksY) {
                                chunks_local[idx2] = nullptr;
                            }
                        }

                        if (lz+1>=cd) {
                            cv::Vec4i idx2 = idx;
                            idx2[3]++;
                            if (idx2[3] >= 0 && idx2[3] < chunksZ) {
                                chunks_local[idx2] = nullptr;
                            }
                        }
                    }
                }
            }

#pragma omp barrier
#pragma omp critical
            chunks.merge(chunks_local);

        }

    std::vector<std::pair<cv::Vec4i,xt::xarray<T>*>> needs_io;

    cache->mutex.lock();
    for(auto &it : chunks) {
        xt::xarray<T> *chunk = nullptr;
        std::shared_ptr<xt::xarray<T>> chunk_ref;

        cv::Vec4i idx = it.first;

        if (!cache->has(idx)) {
            needs_io.push_back({idx,nullptr});
        } else {
            chunk_ref = cache->get(idx);
            chunks[idx] = chunk_ref;
        }
    }
    cache->mutex.unlock();

    #pragma omp parallel for schedule(dynamic, 1)
    for(auto &it : needs_io) {
        cv::Vec4i idx = it.first;
        std::shared_ptr<xt::xarray<T>> chunk_ref;
        it.second = readChunk<T>(*ds, {size_t(idx[1]),size_t(idx[2]),size_t(idx[3])});
    }

    cache->mutex.lock();
    for(auto &it : needs_io) {
        cv::Vec4i idx = it.first;
        cache->put(idx, it.second);
        chunks[idx] = cache->get(idx);
    }
    cache->mutex.unlock();


    #pragma omp parallel
    {
        cv::Vec4i last_idx = {-1,-1,-1,-1};
        std::shared_ptr<xt::xarray<T>> chunk_ref;
        xt::xarray<T> *chunk = nullptr;

        #pragma omp for collapse(2)
        for(size_t y = 0;y<h;y++) {
            for(size_t x = 0;x<w;x++) {
                float ox = coords(y,x)[2];
                float oy = coords(y,x)[1];
                float oz = coords(y,x)[0];

                if (ox < 0 || oy < 0 || oz < 0)
                    continue;

                if (ox >= sx || oy >= sy || oz >= sz) {
                    continue;
                }

                int ix = int(ox)/cw;
                int iy = int(oy)/ch;
                int iz = int(oz)/cd;

                cv::Vec4i idx = {group_idx,ix,iy,iz};

                if (idx != last_idx) {
                    last_idx = idx;
                    if (ix < 0 || ix >= chunksX ||
                        iy < 0 || iy >= chunksY ||
                        iz < 0 || iz >= chunksZ) {
                        chunk = nullptr;
                    } else {
                        chunk = chunks[idx].get();
                    }
                }

                int lx = ox-ix*cw;
                int ly = oy-iy*ch;
                int lz = oz-iz*cd;

                //valid - means zero!
                if (!chunk)
                    continue;

                float c000 = chunk->operator()(lx,ly,lz);
                float c100, c010, c110, c001, c101, c011, c111;

                // Handle edge cases for interpolation
                if (lx+1 >= cw || ly+1 >= ch || lz+1 >= cd) {
                    if (lx+1>=cw)
                        c100 = retrieve_single_value_cached(ox+1,oy,oz);
                    else
                        c100 = chunk->operator()(lx+1,ly,lz);

                    if (ly+1 >= ch)
                        c010 = retrieve_single_value_cached(ox,oy+1,oz);
                    else
                        c010 = chunk->operator()(lx,ly+1,lz);
                    if (lz+1 >= cd)
                        c001 = retrieve_single_value_cached(ox,oy,oz+1);
                    else
                        c001 = chunk->operator()(lx,ly,lz+1);

                    c110 = retrieve_single_value_cached(ox+1,oy+1,oz);
                    c101 = retrieve_single_value_cached(ox+1,oy,oz+1);
                    c011 = retrieve_single_value_cached(ox,oy+1,oz+1);
                    c111 = retrieve_single_value_cached(ox+1,oy+1,oz+1);
                } else {
                    c100 = chunk->operator()(lx+1,ly,lz);
                    c010 = chunk->operator()(lx,ly+1,lz);
                    c110 = chunk->operator()(lx+1,ly+1,lz);
                    c001 = chunk->operator()(lx,ly,lz+1);
                    c101 = chunk->operator()(lx+1,ly,lz+1);
                    c011 = chunk->operator()(lx,ly+1,lz+1);
                    c111 = chunk->operator()(lx+1,ly+1,lz+1);
                }

                // Trilinear interpolation
                float fx = ox-int(ox);
                float fy = oy-int(oy);
                float fz = oz-int(oz);

                float c00 = (1-fz)*c000 + fz*c001;
                float c01 = (1-fz)*c010 + fz*c011;
                float c10 = (1-fz)*c100 + fz*c101;
                float c11 = (1-fz)*c110 + fz*c111;

                float c0 = (1-fy)*c00 + fy*c01;
                float c1 = (1-fy)*c10 + fy*c11;

                float c = (1-fx)*c0 + fx*c1;

                if constexpr (std::is_same_v<T, uint16_t>) {
                    if (c < 0.f) c = 0.f;
                    if (c > 65535.f) c = 65535.f;
                    out(y,x) = static_cast<uint16_t>(c + 0.5f);
                } else {
                    out(y,x) = c;
                }
            }
        }
    }
}

void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds,
                               const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}

void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds,
                               const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}


// ============================================================================
// FastCompositeCache implementation - lock-free chunk caching for composite
// ============================================================================

void FastCompositeCache::clear() {
    _chunks.clear();
    _ds = nullptr;
    _cw = _ch = _cd = 0;
    _sx = _sy = _sz = 0;
    _chunksX = _chunksY = _chunksZ = 0;
}

void FastCompositeCache::setDataset(z5::Dataset* ds) {
    if (_ds == ds) return;  // Already set to this dataset

    clear();
    _ds = ds;

    if (!ds) return;

    const auto& blockShape = ds->chunking().blockShape();
    _cw = static_cast<int>(blockShape[0]);
    _ch = static_cast<int>(blockShape[1]);
    _cd = static_cast<int>(blockShape[2]);

    const auto& dsShape = ds->shape();
    _sx = static_cast<int>(dsShape[0]);
    _sy = static_cast<int>(dsShape[1]);
    _sz = static_cast<int>(dsShape[2]);

    _chunksX = (_sx + _cw - 1) / _cw;
    _chunksY = (_sy + _ch - 1) / _ch;
    _chunksZ = (_sz + _cd - 1) / _cd;
}

const xt::xarray<uint8_t>* FastCompositeCache::getChunk(int ix, int iy, int iz) {
    if (!_ds) return nullptr;
    if (ix < 0 || ix >= _chunksX || iy < 0 || iy >= _chunksY || iz < 0 || iz >= _chunksZ)
        return nullptr;

    uint64_t key = chunkKey(ix, iy, iz);
    auto it = _chunks.find(key);
    if (it != _chunks.end()) {
        return it->second.get();
    }

    // Load chunk directly - no mutex needed
    auto* chunk = readChunk<uint8_t>(*_ds, {static_cast<size_t>(ix), static_cast<size_t>(iy), static_cast<size_t>(iz)});
    if (chunk) {
        _chunks[key] = std::unique_ptr<xt::xarray<uint8_t>>(chunk);
        return chunk;
    }
    return nullptr;
}

// ============================================================================
// readCompositeFast - specialized fast path for nearest-neighbor compositing
// ============================================================================

// Helper to compute log2 for power-of-2 values (used for bit shift optimization)
static inline int log2_pow2(int v) {
    // For power-of-2 values, count trailing zeros gives log2
    int r = 0;
    while ((v >> r) > 1) r++;
    return r;
}

void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    FastCompositeCache& cache)
{
    cache.setDataset(ds);

    const int h = baseCoords.rows;
    const int w = baseCoords.cols;
    const int numLayers = zEnd - zStart + 1;

    const bool hasNormals = !normals.empty() && normals.size() == baseCoords.size();

    const int cw = cache.chunkSizeX();
    const int ch = cache.chunkSizeY();
    const int cd = cache.chunkSizeZ();
    const int sx = cache.datasetSizeX();
    const int sy = cache.datasetSizeY();
    const int sz = cache.datasetSizeZ();
    const int chunksX = (sx + cw - 1) / cw;
    const int chunksY = (sy + ch - 1) / ch;
    const int chunksZ = (sz + cd - 1) / cd;

    // Bit shift constants for power-of-2 chunk sizes (replaces expensive division)
    const int cwShift = log2_pow2(cw);
    const int chShift = log2_pow2(ch);
    const int cdShift = log2_pow2(cd);
    const int cwMask = cw - 1;  // For modulo: x % cw == x & cwMask when cw is power of 2
    const int chMask = ch - 1;
    const int cdMask = cd - 1;

    // Phase 1: Collect all needed chunks using a bitmap (much faster than unordered_set)
    // Use a flat array as a 3D bitmap - O(1) insert/lookup vs O(1) amortized with hash overhead
    const size_t totalChunks = static_cast<size_t>(chunksX) * chunksY * chunksZ;
    std::vector<uint8_t> chunkNeeded(totalChunks, 0);

    // Track bounding box for efficient array allocation (use atomics for parallel updates)
    std::atomic<int> minIx{chunksX}, maxIx{-1};
    std::atomic<int> minIy{chunksY}, maxIy{-1};
    std::atomic<int> minIz{chunksZ}, maxIz{-1};

    // Pre-compute layer offsets (constant for all pixels)
    std::vector<float> layerOffsets(numLayers);
    for (int layer = 0; layer < numLayers; layer++) {
        layerOffsets[layer] = (zStart + layer) * zStep;
    }

    // Parallel scan of all coordinates to find needed chunks
    #pragma omp parallel
    {
        // Thread-local min/max to reduce atomic contention
        int localMinIx = chunksX, localMaxIx = -1;
        int localMinIy = chunksY, localMaxIy = -1;
        int localMinIz = chunksZ, localMaxIz = -1;

        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& baseCoord = baseCoords(y, x);
                const float base_ox = baseCoord[2];
                const float base_oy = baseCoord[1];
                const float base_oz = baseCoord[0];

                if (base_oz < 0 || base_oy < 0 || base_ox < 0) continue;
                if (base_ox >= sx || base_oy >= sy) continue;

                float nx = 0, ny = 0, nz = 1;
                if (hasNormals) {
                    const cv::Vec3f& n = normals(y, x);
                    if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                        nx = n[0]; ny = n[1]; nz = n[2];
                    }
                }

                for (int layer = 0; layer < numLayers; layer++) {
                    const float layerOffset = layerOffsets[layer];
                    const float ox = base_ox + nz * layerOffset;
                    const float oy = base_oy + ny * layerOffset;
                    const float oz = base_oz + nx * layerOffset;

                    if (oz < 0 || oz >= sz || oy < 0 || oy >= sy || ox < 0 || ox >= sx) continue;

                    int iox = static_cast<int>(ox + 0.5f);
                    int ioy = static_cast<int>(oy + 0.5f);
                    int ioz = static_cast<int>(oz + 0.5f);

                    if (iox >= sx) iox = sx - 1;
                    if (ioy >= sy) ioy = sy - 1;
                    if (ioz >= sz) ioz = sz - 1;

                    // Use bit shifts instead of division (chunk sizes are power of 2)
                    const int ix = iox >> cwShift;
                    const int iy = ioy >> chShift;
                    const int iz = ioz >> cdShift;

                    if (ix >= 0 && ix < chunksX && iy >= 0 && iy < chunksY && iz >= 0 && iz < chunksZ) {
                        // Bitmap write is safe even with races - worst case we write 1 multiple times
                        chunkNeeded[ix + iy * chunksX + iz * chunksX * chunksY] = 1;
                        localMinIx = std::min(localMinIx, ix);
                        localMaxIx = std::max(localMaxIx, ix);
                        localMinIy = std::min(localMinIy, iy);
                        localMaxIy = std::max(localMaxIy, iy);
                        localMinIz = std::min(localMinIz, iz);
                        localMaxIz = std::max(localMaxIz, iz);
                    }
                }
            }
        }

        // Merge thread-local bounds into global atomics
        #pragma omp critical
        {
            if (localMinIx < minIx.load()) minIx.store(localMinIx);
            if (localMaxIx > maxIx.load()) maxIx.store(localMaxIx);
            if (localMinIy < minIy.load()) minIy.store(localMinIy);
            if (localMaxIy > maxIy.load()) maxIy.store(localMaxIy);
            if (localMinIz < minIz.load()) minIz.store(localMinIz);
            if (localMaxIz > maxIz.load()) maxIz.store(localMaxIz);
        }
    }

    const int finalMinIx = minIx.load(), finalMaxIx = maxIx.load();
    const int finalMinIy = minIy.load(), finalMaxIy = maxIy.load();
    const int finalMinIz = minIz.load(), finalMaxIz = maxIz.load();

    if (finalMaxIx < finalMinIx) {
        // No valid chunks needed
        out = cv::Mat_<uint8_t>(baseCoords.size(), 0);
        return;
    }

    // Phase 2: Pre-load all needed chunks and build lookup array
    const int arrSizeX = finalMaxIx - finalMinIx + 1;
    const int arrSizeY = finalMaxIy - finalMinIy + 1;
    const int arrSizeZ = finalMaxIz - finalMinIz + 1;
    std::vector<const xt::xarray<uint8_t>*> chunkArray(arrSizeX * arrSizeY * arrSizeZ, nullptr);

    // Iterate over bounding box and load chunks that are marked as needed
    for (int iz = finalMinIz; iz <= finalMaxIz; iz++) {
        for (int iy = finalMinIy; iy <= finalMaxIy; iy++) {
            for (int ix = finalMinIx; ix <= finalMaxIx; ix++) {
                if (chunkNeeded[ix + iy * chunksX + iz * chunksX * chunksY]) {
                    const int arrIdx = (ix - finalMinIx) + (iy - finalMinIy) * arrSizeX + (iz - finalMinIz) * arrSizeX * arrSizeY;
                    chunkArray[arrIdx] = cache.getChunk(ix, iy, iz);
                }
            }
        }
    }

    auto getChunk = [&](int ix, int iy, int iz) -> const xt::xarray<uint8_t>* {
        if (ix < finalMinIx || ix > finalMaxIx || iy < finalMinIy || iy > finalMaxIy || iz < finalMinIz || iz > finalMaxIz)
            return nullptr;
        return chunkArray[(ix - finalMinIx) + (iy - finalMinIy) * arrSizeX + (iz - finalMinIz) * arrSizeX * arrSizeY];
    };

    // Phase 3: Contrast enhancement
    // Identity LUT - no histogram enhancement (GLCAE removed)
    std::array<uint8_t, 256> equalizeLUT;
    for (int i = 0; i < 256; i++) equalizeLUT[i] = static_cast<uint8_t>(i);

    // Phase 4: Initialize output
    out = cv::Mat_<uint8_t>(baseCoords.size(), 0);

    // Determine which compositing path to use
    const bool isAlpha = (params.method == "alpha");
    const bool isMin = (params.method == "min");
    const bool isMax = (params.method == "max");
    const bool isMean = (params.method == "mean");
    const bool needsLayerStorage = methodRequiresLayerStorage(params.method);

    // Pre-compute alpha normalization constants
    const float alphaScale = 1.0f / (255.0f * (params.alphaMax - params.alphaMin));
    const float alphaOffset = params.alphaMin / (params.alphaMax - params.alphaMin);

    // Pre-compute incremental step for coordinate updates
    const float firstLayerOffset = zStart * zStep;

    // Pre-compute chunk stride for C-contiguous layout (common case)
    // This avoids 3 multiplies per sample when layout is standard
    const size_t chunkPlaneStride = static_cast<size_t>(cw) * ch;  // stride for z dimension
    const size_t chunkRowStride = static_cast<size_t>(cw);         // stride for y dimension

    // Phase 4: Process all layers for each pixel
    #pragma omp parallel
    {
        // Thread-local layer stack for methods that need it
        LayerStack stack;
        if (needsLayerStorage) {
            stack.values.resize(numLayers);
        }

        #pragma omp for collapse(2)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& baseCoord = baseCoords(y, x);
                const float base_ox = baseCoord[2];
                const float base_oy = baseCoord[1];
                const float base_oz = baseCoord[0];

                if (base_oz < 0 || base_oy < 0 || base_ox < 0) continue;

                // Get normal once per pixel
                float nx = 0, ny = 0, nz = 1;
                if (hasNormals) {
                    const cv::Vec3f& n = normals(y, x);
                    if (n[0] == n[0] && n[1] == n[1] && n[2] == n[2]) {
                        nx = n[0];
                        ny = n[1];
                        nz = n[2];
                    }
                }

                // Accumulators for simple methods
                float acc = isMin ? 255.0f : 0.0f;
                int validCount = 0;

                // Reset layer stack
                if (needsLayerStorage) {
                    stack.validCount = 0;
                }

                // Pre-compute incremental coordinate deltas for this pixel's normal
                const float dx = nz * zStep;
                const float dy = ny * zStep;
                const float dz = nx * zStep;

                // Start at first layer position
                float ox = base_ox + nz * firstLayerOffset;
                float oy = base_oy + ny * firstLayerOffset;
                float oz = base_oz + nx * firstLayerOffset;

                // Chunk caching: track current chunk to avoid re-lookup
                int cachedIx = -1, cachedIy = -1, cachedIz = -1;
                const uint8_t* cachedData = nullptr;
                size_t cachedStride0 = 0, cachedStride1 = 0, cachedStride2 = 0;

                // Sample all layers
                for (int layer = 0; layer < numLayers; layer++) {
                    float value;
                    bool validSample = false;

                    // Bounds check
                    if (oz >= 0 && oz < sz && oy >= 0 && oy < sy && ox >= 0 && ox < sx) {
                            // Fast rounding with clamping to valid range
                            int iox = static_cast<int>(ox + 0.5f);
                            int ioy = static_cast<int>(oy + 0.5f);
                            int ioz = static_cast<int>(oz + 0.5f);

                            // Clamp to valid range (rounding can push to boundary)
                            if (iox >= sx) iox = sx - 1;
                            if (ioy >= sy) ioy = sy - 1;
                            if (ioz >= sz) ioz = sz - 1;

                            // Use bit shifts for chunk index
                            const int ix = iox >> cwShift;
                            const int iy = ioy >> chShift;
                            const int iz = ioz >> cdShift;

                            // Check if we need to switch chunks
                            if (ix != cachedIx || iy != cachedIy || iz != cachedIz) {
                                const xt::xarray<uint8_t>* chunk = getChunk(ix, iy, iz);
                                if (chunk) {
                                    cachedData = chunk->data();
                                    const auto& strides = chunk->strides();
                                    cachedStride0 = strides[0];
                                    cachedStride1 = strides[1];
                                    cachedStride2 = strides[2];
                                    cachedIx = ix;
                                    cachedIy = iy;
                                    cachedIz = iz;

                                    // Prefetch next likely chunk (along z direction)
                                    if (iz + 1 <= finalMaxIz) {
                                        const xt::xarray<uint8_t>* nextChunk = getChunk(ix, iy, iz + 1);
                                        if (nextChunk) {
                                            __builtin_prefetch(nextChunk->data(), 0, 1);
                                        }
                                    }
                                } else {
                                    cachedData = nullptr;
                                }
                            }

                            if (cachedData) {
                                // Use bit masking for local coordinates
                                const int lx = iox & cwMask;
                                const int ly = ioy & chMask;
                                const int lz = ioz & cdMask;

                                // Direct pointer arithmetic with actual strides
                                const uint8_t rawValue = cachedData[lx * cachedStride0 + ly * cachedStride1 + lz * cachedStride2];

                                // Apply LUT (ISO cutoff)
                                value = static_cast<float>(equalizeLUT[rawValue < params.isoCutoff ? 0 : rawValue]);
                                validSample = true;
                            }
                        }

                        // Increment coordinates for next layer
                        ox += dx; oy += dy; oz += dz;

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
                            // mean
                            acc += value;
                            validCount++;
                        }
                    }
                }

                // Compute final value
                float result = 0.0f;

                if (needsLayerStorage) {
                    result = compositeLayerStack(stack, params);
                } else if (isMax || isMin) {
                    result = acc;
                } else if (isMean && validCount > 0) {
                    result = acc / static_cast<float>(validCount);
                }

                // Apply directional lighting if enabled
                if (params.lightingEnabled && hasNormals) {
                    const cv::Vec3f& n = normals(y, x);
                    float lightFactor = computeLightingFactor(n, params);
                    result *= lightFactor;
                }

                out(y, x) = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, result)));
            }
        }
    }
}

void readCompositeFastConstantNormal(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Vec3f& normal,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    FastCompositeCache& cache)
{
    cache.setDataset(ds);

    const int h = baseCoords.rows;
    const int w = baseCoords.cols;
    const int numLayers = zEnd - zStart + 1;

    // Extract constant normal components once
    const float nx = normal[0];
    const float ny = normal[1];
    const float nz = normal[2];

    const int cw = cache.chunkSizeX();
    const int ch = cache.chunkSizeY();
    const int cd = cache.chunkSizeZ();
    const int sx = cache.datasetSizeX();
    const int sy = cache.datasetSizeY();
    const int sz = cache.datasetSizeZ();
    const int chunksX = (sx + cw - 1) / cw;
    const int chunksY = (sy + ch - 1) / ch;
    const int chunksZ = (sz + cd - 1) / cd;

    // Bit shift constants for power-of-2 chunk sizes (replaces expensive division)
    const int cwShift = log2_pow2(cw);
    const int chShift = log2_pow2(ch);
    const int cdShift = log2_pow2(cd);
    const int cwMask = cw - 1;
    const int chMask = ch - 1;
    const int cdMask = cd - 1;

    // Pre-compute layer offsets as 3D vectors (constant normal * layer offset)
    // This is the key optimization for constant normals - we compute the delta once per layer
    struct LayerDelta {
        float dx, dy, dz;
    };
    std::vector<LayerDelta> layerDeltas(numLayers);
    for (int layer = 0; layer < numLayers; layer++) {
        const float offset = (zStart + layer) * zStep;
        layerDeltas[layer] = {nz * offset, ny * offset, nx * offset};
    }

    // Phase 1: Collect all needed chunks using bitmap
    const size_t totalChunks = static_cast<size_t>(chunksX) * chunksY * chunksZ;
    std::vector<uint8_t> chunkNeeded(totalChunks, 0);

    std::atomic<int> minIx{chunksX}, maxIx{-1};
    std::atomic<int> minIy{chunksY}, maxIy{-1};
    std::atomic<int> minIz{chunksZ}, maxIz{-1};

    #pragma omp parallel
    {
        int localMinIx = chunksX, localMaxIx = -1;
        int localMinIy = chunksY, localMaxIy = -1;
        int localMinIz = chunksZ, localMaxIz = -1;

        #pragma omp for schedule(dynamic, 16)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& baseCoord = baseCoords(y, x);
                const float base_ox = baseCoord[2];
                const float base_oy = baseCoord[1];
                const float base_oz = baseCoord[0];

                if (base_oz < 0 || base_oy < 0 || base_ox < 0) continue;
                if (base_ox >= sx || base_oy >= sy) continue;

                for (int layer = 0; layer < numLayers; layer++) {
                    const LayerDelta& delta = layerDeltas[layer];
                    const float ox = base_ox + delta.dx;
                    const float oy = base_oy + delta.dy;
                    const float oz = base_oz + delta.dz;

                    if (oz < 0 || oz >= sz || oy < 0 || oy >= sy || ox < 0 || ox >= sx) continue;

                    int iox = static_cast<int>(ox + 0.5f);
                    int ioy = static_cast<int>(oy + 0.5f);
                    int ioz = static_cast<int>(oz + 0.5f);

                    if (iox >= sx) iox = sx - 1;
                    if (ioy >= sy) ioy = sy - 1;
                    if (ioz >= sz) ioz = sz - 1;

                    // Use bit shifts instead of division
                    const int ix = iox >> cwShift;
                    const int iy = ioy >> chShift;
                    const int iz = ioz >> cdShift;

                    if (ix >= 0 && ix < chunksX && iy >= 0 && iy < chunksY && iz >= 0 && iz < chunksZ) {
                        chunkNeeded[ix + iy * chunksX + iz * chunksX * chunksY] = 1;
                        localMinIx = std::min(localMinIx, ix);
                        localMaxIx = std::max(localMaxIx, ix);
                        localMinIy = std::min(localMinIy, iy);
                        localMaxIy = std::max(localMaxIy, iy);
                        localMinIz = std::min(localMinIz, iz);
                        localMaxIz = std::max(localMaxIz, iz);
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (localMinIx < minIx.load()) minIx.store(localMinIx);
            if (localMaxIx > maxIx.load()) maxIx.store(localMaxIx);
            if (localMinIy < minIy.load()) minIy.store(localMinIy);
            if (localMaxIy > maxIy.load()) maxIy.store(localMaxIy);
            if (localMinIz < minIz.load()) minIz.store(localMinIz);
            if (localMaxIz > maxIz.load()) maxIz.store(localMaxIz);
        }
    }

    const int finalMinIx = minIx.load(), finalMaxIx = maxIx.load();
    const int finalMinIy = minIy.load(), finalMaxIy = maxIy.load();
    const int finalMinIz = minIz.load(), finalMaxIz = maxIz.load();

    if (finalMaxIx < finalMinIx) {
        out = cv::Mat_<uint8_t>(baseCoords.size(), 0);
        return;
    }

    // Phase 2: Pre-load chunks and build lookup array
    const int arrSizeX = finalMaxIx - finalMinIx + 1;
    const int arrSizeY = finalMaxIy - finalMinIy + 1;
    const int arrSizeZ = finalMaxIz - finalMinIz + 1;
    std::vector<const xt::xarray<uint8_t>*> chunkArray(arrSizeX * arrSizeY * arrSizeZ, nullptr);

    for (int iz = finalMinIz; iz <= finalMaxIz; iz++) {
        for (int iy = finalMinIy; iy <= finalMaxIy; iy++) {
            for (int ix = finalMinIx; ix <= finalMaxIx; ix++) {
                if (chunkNeeded[ix + iy * chunksX + iz * chunksX * chunksY]) {
                    const int arrIdx = (ix - finalMinIx) + (iy - finalMinIy) * arrSizeX + (iz - finalMinIz) * arrSizeX * arrSizeY;
                    chunkArray[arrIdx] = cache.getChunk(ix, iy, iz);
                }
            }
        }
    }

    auto getChunk = [&](int ix, int iy, int iz) -> const xt::xarray<uint8_t>* {
        if (ix < finalMinIx || ix > finalMaxIx || iy < finalMinIy || iy > finalMaxIy || iz < finalMinIz || iz > finalMaxIz)
            return nullptr;
        return chunkArray[(ix - finalMinIx) + (iy - finalMinIy) * arrSizeX + (iz - finalMinIz) * arrSizeX * arrSizeY];
    };

    // Phase 3: Identity LUT - no histogram enhancement (GLCAE removed)
    std::array<uint8_t, 256> equalizeLUT;
    for (int i = 0; i < 256; i++) equalizeLUT[i] = static_cast<uint8_t>(i);
    // Phase 4: Composite rendering
    out = cv::Mat_<uint8_t>(baseCoords.size(), 0);

    const bool isMin = (params.method == "min");
    const bool isMax = (params.method == "max");
    const bool isMean = (params.method == "mean");
    const bool needsLayerStorage = methodRequiresLayerStorage(params.method);

    #pragma omp parallel
    {
        LayerStack stack;
        if (needsLayerStorage) {
            stack.values.resize(numLayers);
        }

        #pragma omp for collapse(2)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& baseCoord = baseCoords(y, x);
                const float base_ox = baseCoord[2];
                const float base_oy = baseCoord[1];
                const float base_oz = baseCoord[0];

                if (base_oz < 0 || base_oy < 0 || base_ox < 0) continue;

                float acc = isMin ? 255.0f : 0.0f;
                int validCount = 0;

                if (needsLayerStorage) {
                    stack.validCount = 0;
                }

                // Chunk caching (only used for non-3DGLCAE path)
                int cachedIx = -1, cachedIy = -1, cachedIz = -1;
                const uint8_t* cachedData = nullptr;
                size_t cachedStride0 = 0, cachedStride1 = 0, cachedStride2 = 0;

                // Sample all layers
                for (int layer = 0; layer < numLayers; layer++) {
                    float value;
                    bool validSample = false;

                    const LayerDelta& delta = layerDeltas[layer];
                    const float ox = base_ox + delta.dx;
                    const float oy = base_oy + delta.dy;
                    const float oz = base_oz + delta.dz;

                    // Bounds check
                    if (oz >= 0 && oz < sz && oy >= 0 && oy < sy && ox >= 0 && ox < sx) {
                            // Fast rounding with clamping
                            int iox = static_cast<int>(ox + 0.5f);
                            int ioy = static_cast<int>(oy + 0.5f);
                            int ioz = static_cast<int>(oz + 0.5f);

                            if (iox >= sx) iox = sx - 1;
                            if (ioy >= sy) ioy = sy - 1;
                            if (ioz >= sz) ioz = sz - 1;

                            const int ix = iox >> cwShift;
                            const int iy = ioy >> chShift;
                            const int iz = ioz >> cdShift;

                            if (ix != cachedIx || iy != cachedIy || iz != cachedIz) {
                                const xt::xarray<uint8_t>* chunk = getChunk(ix, iy, iz);
                                if (chunk) {
                                    cachedData = chunk->data();
                                    const auto& strides = chunk->strides();
                                    cachedStride0 = strides[0];
                                    cachedStride1 = strides[1];
                                    cachedStride2 = strides[2];
                                    cachedIx = ix;
                                    cachedIy = iy;
                                    cachedIz = iz;

                                    // Prefetch next chunk along z
                                    if (iz + 1 <= finalMaxIz) {
                                        const xt::xarray<uint8_t>* nextChunk = getChunk(ix, iy, iz + 1);
                                        if (nextChunk) {
                                            __builtin_prefetch(nextChunk->data(), 0, 1);
                                        }
                                    }
                                } else {
                                    cachedData = nullptr;
                                }
                            }

                            if (cachedData) {
                                const int lx = iox & cwMask;
                                const int ly = ioy & chMask;
                                const int lz = ioz & cdMask;

                                const uint8_t rawValue = cachedData[lx * cachedStride0 + ly * cachedStride1 + lz * cachedStride2];
                                value = static_cast<float>(equalizeLUT[rawValue]);
                                validSample = true;
                            }
                        }

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
                    result = compositeLayerStack(stack, params);
                } else if (isMax || isMin) {
                    result = acc;
                } else if (isMean && validCount > 0) {
                    result = acc / static_cast<float>(validCount);
                }

                // Apply directional lighting if enabled (using constant normal)
                if (params.lightingEnabled) {
                    float lightFactor = computeLightingFactor(normal, params);
                    result *= lightFactor;
                }

                out(y, x) = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, result)));
            }
        }
    }
}
