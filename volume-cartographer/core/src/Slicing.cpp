#include "vc/core/util/Slicing.hpp"

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
