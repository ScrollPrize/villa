#include "vc/core/util/Slicing.hpp"

#include <nlohmann/json.hpp>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include XTENSORINCLUDE(views, xaxis_slice_iterator.hpp)
#include XTENSORINCLUDE(io, xio.hpp)
#include XTENSORINCLUDE(generators, xbuilder.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <shared_mutex>

#include <algorithm>
#include <random>


template<typename T>
static xt::xarray<T> *readChunk(const z5::Dataset & ds, z5::types::ShapeType chunkId)
{
    if (!ds.chunkExists(chunkId)) {
        return nullptr;
    }

    if (!ds.isZarr())
        throw std::runtime_error("only zarr datasets supported currently!");
    if (ds.getDtype() != z5::types::Datatype::uint8 && ds.getDtype() != z5::types::Datatype::uint16)
        throw std::runtime_error("only uint8_t/uint16 zarrs supported currently!");

    z5::types::ShapeType chunkShape;
    // size_t chunkSize;
    ds.getChunkShape(chunkId, chunkShape);
    // get the shape of the chunk (as stored it is stored)
    //for ZARR also edge chunks are always full size!
    const std::size_t maxChunkSize = ds.defaultChunkSize();
    const auto & maxChunkShape = ds.defaultChunkShape();

    // chunkSize = std::accumulate(chunkShape.begin(), chunkShape.end(), 1, std::multiplies<std::size_t>());

    xt::xarray<T> *out = new xt::xarray<T>();
    *out = xt::empty<T>(maxChunkShape);


    // read/decompress & convert data
    if (ds.getDtype() == z5::types::Datatype::uint8) {
        ds.readChunk(chunkId, out->data());
    }
    else if (ds.getDtype() == z5::types::Datatype::uint16) {
        xt::xarray<uint16_t> tmp = xt::empty<T>(maxChunkShape);
        ds.readChunk(chunkId, tmp.data());

        uint8_t *p8 = out->data();
        uint16_t *p16 = tmp.data();
        for(int i=0;i<maxChunkSize;i++)
            p8[i] = p16[i] / 257;
    }

    return out;
}


int ChunkCache::groupIdx(const std::string &name)
{
    if (!_group_store.count(name))
        _group_store[name] = _group_store.size()+1;
    
     return _group_store[name];
}
    
void ChunkCache::put(const cv::Vec4i &idx, xt::xarray<uint8_t> *ar)
{
    if (_stored >= _size) {
        using KP = std::pair<cv::Vec4i, uint64_t>;
        std::vector<KP> gen_list(_gen_store.begin(), _gen_store.end());
        std::sort(gen_list.begin(), gen_list.end(), [](KP &a, KP &b){ return a.second < b.second; });
        for(auto it : gen_list) {
            std::shared_ptr<xt::xarray<uint8_t>> ar = _store[it.first];
            //TODO we could remove this with lower probability so we dont store infiniteyl empty blocks but also keep more of them as they are cheap
            if (ar.get()) {
                size_t size = ar.get()->storage().size();
                ar.reset();
                _stored -= size;
            
                _store.erase(it.first);
                _gen_store.erase(it.first);
            }

            //we delete 10% of cache content to amortize sorting costs
            if (_stored < 0.9*_size) {
                break;
            }
        }
    }

    if (ar) {
        if (_store.count(idx)) {
            assert(_store[idx].get());
            _stored -= ar->size();
        }
        _stored += ar->size();
    }
    _store[idx].reset(ar);
    _generation++;
    _gen_store[idx] = _generation;
}

void readArea3D(xt::xtensor<uint8_t, 3, xt::layout_type::column_major>& out, const cv::Vec3i offset, z5::Dataset* ds, ChunkCache* cache) {
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
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;
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
            auto* new_chunk = readChunk<uint8_t>(*ds, {(size_t)idx[1], (size_t)idx[2], (size_t)idx[3]});
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


ChunkCache::~ChunkCache()
{
    for(auto &it : _store)
        it.second.reset();
}

void ChunkCache::reset()
{
    _gen_store.clear();
    _group_store.clear();
    _store.clear();

    _generation = 0;
    _stored = 0;
}

std::shared_ptr<xt::xarray<uint8_t>> ChunkCache::get(const cv::Vec4i &idx)
{
    auto res = _store.find(idx);
    if (res == _store.end())
        return nullptr;

    _generation++;
    _gen_store[idx] = _generation;
    
    return res->second;
}

bool ChunkCache::has(const cv::Vec4i &idx)
{
    return _store.count(idx);
}


void readNearestNeighbor(cv::Mat_<uint8_t> &out, const z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache) {
    out = cv::Mat_<uint8_t>(coords.size(), 0);
    int group_idx = cache->groupIdx(ds->path());

    const int chunk_size = ds->chunking().blockShape()[0];
    const int chunk_shift = __builtin_ctz(chunk_size);
    const int chunk_mask = chunk_size - 1;

    if ((chunk_size & (chunk_size - 1)) != 0 || chunk_size == 0) {
        throw std::runtime_error("Chunk size must be a power of 2, got: " + std::to_string(chunk_size));
    }

    int w = coords.cols;
    int h = coords.rows;

    constexpr int TILE_SIZE = 32;

    #pragma omp parallel
    {
        // Thread-local variables
        cv::Vec4i last_idx = {-1,-1,-1,-1};
        xt::xarray<uint8_t> *chunk = nullptr;
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;

        #pragma omp for schedule(static, 1) collapse(2)
        for(size_t tile_y = 0; tile_y < h; tile_y += TILE_SIZE) {
            for(size_t tile_x = 0; tile_x < w; tile_x += TILE_SIZE) {
                size_t y_end = std::min(tile_y + TILE_SIZE, (size_t)h);
                size_t x_end = std::min(tile_x + TILE_SIZE, (size_t)w);

                for(size_t y = tile_y; y < y_end; y++) {
                    if (y + 1 < y_end) {
                        __builtin_prefetch(&coords(y+1, tile_x), 0, 1);
                    }

                    for(size_t x = tile_x; x < x_end; x++) {
                        int ox = int(coords(y,x)[2] + 0.5f);
                        int oy = int(coords(y,x)[1] + 0.5f);
                        int oz = int(coords(y,x)[0] + 0.5f);

                        if ((ox | oy | oz) < 0)
                            continue;

                        int ix = ox >> chunk_shift;
                        int iy = oy >> chunk_shift;
                        int iz = oz >> chunk_shift;

                        cv::Vec4i idx = {group_idx, ix, iy, iz};

                        if (idx != last_idx) {
                            last_idx = idx;

                            #pragma omp critical(cache_access)
                            {
                                if (!cache->has(idx)) {
                                    auto* new_chunk = readChunk<uint8_t>(*ds, {size_t(ix), size_t(iy), size_t(iz)});
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

                        int lx = ox & chunk_mask;
                        int ly = oy & chunk_mask;
                        int lz = oz & chunk_mask;

                        out(y,x) = chunk->operator()(lx, ly, lz);
                    }
                }
            }
        }
    }
}

void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds,
                               const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache, bool nearest_neighbor) {
    if (nearest_neighbor) {
        return readNearestNeighbor(out,ds,coords,cache);
    }
  
    out = cv::Mat_<uint8_t>(coords.size(), 0);

    if (!cache) {
        std::cout << "ERROR should use a shared chunk cache!" << std::endl;
        abort();
    }

    int group_idx = cache->groupIdx(ds->path());

    auto cw = ds->chunking().blockShape()[0];
    auto ch = ds->chunking().blockShape()[1];
    auto cd = ds->chunking().blockShape()[2];

    int w = coords.cols;
    int h = coords.rows;

    std::shared_mutex mutex;
    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint8_t>>,vec4i_hash> chunks;

    // Lambda for retrieving single values (unchanged)
    auto retrieve_single_value_cached = [&cw,&ch,&cd,&group_idx,&chunks](
        int ox, int oy, int oz) -> uint8_t {

            int ix = int(ox)/cw;
            int iy = int(oy)/ch;
            int iz = int(oz)/cd;

            cv::Vec4i idx = {group_idx,ix,iy,iz};

            xt::xarray<uint8_t> *chunk  = chunks[idx].get();

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
            std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;
            xt::xarray<uint8_t> *chunk = nullptr;
            std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint8_t>>,vec4i_hash> chunks_local;

            #pragma omp for collapse(2)
            for(size_t y = 0;y<h;y++) {
                for(size_t x = 0;x<w;x++) {
                    float ox = coords(y,x)[2];
                    float oy = coords(y,x)[1];
                    float oz = coords(y,x)[0];

                    if (ox < 0 || oy < 0 || oz < 0)
                        continue;

                    int ix = int(ox)/cw;
                    int iy = int(oy)/ch;
                    int iz = int(oz)/cd;

                    cv::Vec4i idx = {group_idx,ix,iy,iz};

                    if (idx != last_idx) {
                        last_idx = idx;
                        chunks_local[idx] = nullptr;
                    }

                    int lx = ox-ix*cw;
                    int ly = oy-iy*ch;
                    int lz = oz-iz*cd;

                    if (lx+1 >= cw || ly+1 >= ch || lz+1 >= cd) {
                        if (lx+1>=cw) {
                            cv::Vec4i idx2 = idx;
                            idx2[1]++;
                            chunks_local[idx2] = nullptr;
                        }
                        if (ly+1>=ch) {
                            cv::Vec4i idx2 = idx;
                            idx2[2]++;
                            chunks_local[idx2] = nullptr;
                        }

                        if (lz+1>=cd) {
                            cv::Vec4i idx2 = idx;
                            idx2[3]++;
                            chunks_local[idx2] = nullptr;
                        }
                    }
                }
            }

#pragma omp barrier
#pragma omp critical
            chunks.merge(chunks_local);

        }

    std::vector<std::pair<cv::Vec4i,xt::xarray<uint8_t>*>> needs_io;

    cache->mutex.lock();
    for(auto &it : chunks) {
        xt::xarray<uint8_t> *chunk = nullptr;
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;

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
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;
        it.second = readChunk<uint8_t>(*ds, {size_t(idx[1]),size_t(idx[2]),size_t(idx[3])});
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
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;
        xt::xarray<uint8_t> *chunk = nullptr;

        #pragma omp for collapse(2)
        for(size_t y = 0;y<h;y++) {
            for(size_t x = 0;x<w;x++) {
                float ox = coords(y,x)[2];
                float oy = coords(y,x)[1];
                float oz = coords(y,x)[0];

                if (ox < 0 || oy < 0 || oz < 0)
                    continue;

                int ix = int(ox)/cw;
                int iy = int(oy)/ch;
                int iz = int(oz)/cd;

                cv::Vec4i idx = {group_idx,ix,iy,iz};

                if (idx != last_idx) {
                    last_idx = idx;
                    chunk = chunks[idx].get();
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

                out(y,x) = c;
            }
        }
    }
}

//somehow opencvs functions are pretty slow 
static cv::Vec3f normed(const cv::Vec3f v)
{
    return v/sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

static cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;
    
    cv::Vec3f p00 = points(y,x);
    cv::Vec3f p01 = points(y,x+1);
    cv::Vec3f p10 = points(y+1,x);
    cv::Vec3f p11 = points(y+1,x+1);
    
    cv::Vec3f p0 = (1-fx)*p00 + fx*p01;
    cv::Vec3f p1 = (1-fx)*p10 + fx*p11;
    
    return (1-fy)*p0 + fy*p1;
}

static cv::Vec2f vmin(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return {std::min(a[0],b[0]),std::min(a[1],b[1])};
}

static cv::Vec2f vmax(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return {std::max(a[0],b[0]),std::max(a[1],b[1])};
}

cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc)
{
    cv::Vec2f inb_loc = {loc[0], loc[1]};
    //move inside from the grid border so w can access required locations
    inb_loc = vmax(inb_loc, {1,1});
    inb_loc = vmin(inb_loc, {points.cols-3,points.rows-3});
    
    if (!loc_valid_xy(points, inb_loc))
        return {NAN,NAN,NAN};
    
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(-1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,1)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,-1)))
        return {NAN,NAN,NAN};
    
    cv::Vec3f xv = normed(at_int(points,inb_loc+cv::Vec2f(1,0))-at_int(points,inb_loc-cv::Vec2f(1,0)));
    cv::Vec3f yv = normed(at_int(points,inb_loc+cv::Vec2f(0,1))-at_int(points,inb_loc-cv::Vec2f(0,1)));
    
    cv::Vec3f n = yv.cross(xv);

    if (std::isnan(n[0]))
        return {NAN,NAN,NAN};
    
    return normed(n);
}

static float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

static void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, bool z_search = true)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = {-1,-1,-1};
        loc = {-1,-1};
        return;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);
    float res;
    
    std::vector<cv::Vec2f> search;
    if (z_search)
        search = {{0,-1},{0,1},{-1,0},{1,0}};
    else
        search = {{1,0},{-1,0}};
    
    float step = 1.0;
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!boundary.contains(cv::Point(cand))) {
                out = {-1,-1,-1};
                loc = {-1,-1};
                return;
            }
            
            
            val = at_int(points, cand);
            res = sdist(val,tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }
        
        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
}

//this works surprisingly well, though some artifacts where original there was a lot of skew
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points)
{
    cv::Mat_<cv::Vec3f> out = points.clone();
    cv::Mat_<cv::Vec3f> blur(points.cols, points.rows);
    cv::Mat_<cv::Vec2f> locs(points.size());
    
    cv::Mat trans = out.t();
    
    #pragma omp parallel for
    for(int j=0;j<trans.rows;j++) 
        cv::GaussianBlur(trans({0,j,trans.cols,1}), blur({0,j,trans.cols,1}), {255,1}, 0);
    
    blur = blur.t();
    
    #pragma omp parallel for
    for(int j=1;j<points.rows;j++)
        for(int i=1;i<points.cols-1;i++) {
            cv::Vec2f loc = {i,j};
            min_loc(points, loc, out(j,i), blur(j,i), false);
        }
        
        return out;
}

void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy)
{
    //so we get something somewhat meaningful by default
    double sum_x = 0;
    double sum_y = 0;
    int count = 0;
    //NOTE leave out bordes as these contain lots of artifacst if coming from smooth_segmentation() ... would need median or something ...
    int jmin = points.size().height*0.1+1;
    int jmax = points.size().height*0.9;
    int imin = points.size().width*0.1+1;
    int imax = points.size().width*0.9;
    int step = 4;
    if (points.size().height < 20) {
        std::cout << "small array vc scales " << std::endl;
        jmin = 1;
        jmax = points.size().height;
        imin = 1;
        imax = points.size().width;
        step = 1;
    }
#pragma omp parallel for
    for(int j=jmin;j<jmax;j+=step) {
        double _sum_x = 0;
        double _sum_y = 0;
        int _count = 0;
        cv::Vec3f *row = points.ptr<cv::Vec3f>(j);
        for(int i=imin;i<imax;i+=step) {
            cv::Vec3f v = points(j,i)-points(j,i-1);
            _sum_x += sqrt(v.dot(v));
            v = points(j,i)-points(j-1,i);
            _sum_y += sqrt(v.dot(v));
            _count++;
        }
#pragma omp critical
        {
            sum_x += _sum_x;
            sum_y += _sum_y;
            count += _count;
        }
    }

    sx = count/sum_x;
    sy = count/sum_y;
}

cv::Mat_<cv::Vec3f> vc_segmentation_calc_normals(const cv::Mat_<cv::Vec3f> &points) {
    int n_step = 1;
    cv::Mat_<cv::Vec3f> blur;
    cv::GaussianBlur(points, blur, {21,21}, 0);
    cv::Mat_<cv::Vec3f> normals(points.size());
#pragma omp parallel for
    for(int j=n_step;j<points.rows-n_step;j++)
        for(int i=n_step;i<points.cols-n_step;i++) {
            cv::Vec3f xv = normed(blur(j,i+n_step)-blur(j,i-n_step));
            cv::Vec3f yv = normed(blur(j+n_step,i)-blur(j-n_step,i));
            
            cv::Vec3f n = yv.cross(xv);
            n = n/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            
            normals(j,i) = n;
        }
        
        cv::GaussianBlur(normals, normals, {21,21}, 0);
        
#pragma omp parallel for
        for(int j=n_step;j<points.rows-n_step;j++)
            for(int i=n_step;i<points.cols-n_step;i++)
                normals(j,i) = normed(normals(j,i));
    
    return normals;
}
