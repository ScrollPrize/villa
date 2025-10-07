#pragma once

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xtensor.hpp)
#include XTENSORINCLUDE(containers, xadapt.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/multiarray/xtensor_access.hxx"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <sstream>

#ifndef CCI_TLS_MAX // Max number for ChunkedCachedInterpolator
#define CCI_TLS_MAX 256
#endif



struct passTroughComputor
{
    enum {BORDER = 0};
    enum {CHUNK_SIZE = 32};
    enum {FILL_V = 0};
    const std::string UNIQUE_ID_STRING = "";
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        small = view(large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};

static uint64_t miss = 0;
static uint64_t total = 0;
static uint64_t chunk_compute_collisions = 0;
static uint64_t chunk_compute_total = 0;

template <typename T, typename C> class Chunked3dAccessor;

static std::string tmp_name_proc_thread()
{
    std::stringstream ss;
    ss << "tmp_" << getpid() << "_" << std::this_thread::get_id();
    return ss.str();
}

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d {
public:
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
    };
    ~Chunked3d()
    {
        if (!_persistent)
            remove_all(_cache_dir);
    };
    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache, const std::filesystem::path &cache_root) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
        
        if (_ds)
            _shape = {_ds->shape()[0],_ds->shape()[1],_ds->shape()[2]};
        
        if (cache_root.empty())
            return;

        if (!_compute_f.UNIQUE_ID_STRING.size())
            throw std::runtime_error("requested fs cache for compute function without identifier");        
        
        std::filesystem::path root = cache_root/_compute_f.UNIQUE_ID_STRING;
        
        std::filesystem::create_directories(root);
        
        if (!_ds)
            _persistent = false;
        
        //create cache dir while others are competing to do the same
        for(int r=0;r<1000 && _cache_dir.empty();r++) {
            std::set<std::string> paths;
            if (_persistent) {
                for (auto const& entry : std::filesystem::directory_iterator(root))
                    if (std::filesystem::is_directory(entry) && std::filesystem::exists(entry.path()/"meta.json") && std::filesystem::is_regular_file(entry.path()/"meta.json")) {
                        paths.insert(entry.path());
                        try {
                            std::ifstream meta_f(entry.path()/"meta.json");
                            nlohmann::json meta = nlohmann::json::parse(meta_f);
                            // Skip entries with invalid or non-existent dataset paths
                            if (!meta.contains("dataset_source_path") || !meta["dataset_source_path"].is_string())
                                continue;
                            std::filesystem::path src_candidate(meta["dataset_source_path"].get<std::string>());
                            if (!std::filesystem::exists(src_candidate))
                                continue;
                            if (!std::filesystem::exists(ds->path()))
                                continue;
                            std::filesystem::path src = std::filesystem::canonical(src_candidate);
                            std::filesystem::path cur = std::filesystem::canonical(ds->path());
                            if (src == cur) {
                                _cache_dir = entry.path();
                                break;
                            }
                        } catch (const std::exception&) {
                            // Ignore malformed cache entries or paths we cannot canonicalize
                            continue;
                        }
                    }
                
                if (!_cache_dir.empty())
                    continue;
            }
            
            //try generating our own cache dir atomically
            std::filesystem::path tmp_dir = cache_root/tmp_name_proc_thread();
            std::filesystem::create_directories(tmp_dir);
            
            if (_persistent) {
                nlohmann::json meta;
                meta["dataset_source_path"] = std::filesystem::canonical(ds->path()).string();
                std::ofstream o(tmp_dir/"meta.json");
                o << std::setw(4) << meta << std::endl;
                
                std::filesystem::path tgt_path;
                for(int i=0;i<1000;i++) {
                    tgt_path = root/std::to_string(i);
                    if (paths.count(tgt_path.string()))
                        continue;
                    try {
                        std::filesystem::rename(tmp_dir, tgt_path);
                    }
                    catch (std::filesystem::filesystem_error&){
                        continue;
                    }
                    _cache_dir = tgt_path;
                    break;
                }
            }
            else {
                _cache_dir = tmp_dir;
            }
        }
        
        if (_cache_dir.empty())
            throw std::runtime_error("could not create cache dir - maybe too many caches in cache root (max 1000!)");
        
    };
    size_t calc_off(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        return p[0] + p[1]*s + p[2]*s*s;
    }
    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        if (!_chunks.count(id))
            cache_chunk(id);

        return _chunks[id][calc_off({p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s})];
    }
    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }
    T &safe_at(const cv::Vec3i &p)
    {
        const auto s = C::CHUNK_SIZE;
        const cv::Vec3i id{ p[0]/s, p[1]/s, p[2]/s };

        {
            std::shared_lock<std::shared_mutex> rlock(_mutex);
            if (auto it = _chunks.find(id); it != _chunks.end()) {
                T* chunk = it->second;
                return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
            }
        }
        // compute/load outside shared lock
        T* chunk = cache_chunk_safe(id);
        return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
    }
    T &safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    std::filesystem::path id_path(const std::filesystem::path &dir, const cv::Vec3i &id)
    {
        return dir / (std::to_string(id[0]) + "." + std::to_string(id[1]) + "." + std::to_string(id[2]));
    }

    T *cache_chunk_safe_mmap(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;

        std::filesystem::path tgt_path = id_path(_cache_dir, id);
        size_t len = s*s*s;
        size_t len_bytes = len*sizeof(T);

        if (std::filesystem::exists(tgt_path)) {
            int fd = open(tgt_path.string().c_str(), O_RDWR);
            T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);

            _mutex.lock();
            if (!_chunks.count(id)) {
                _chunks[id] = chunk;
            }
            else {
#pragma omp atomic
                chunk_compute_collisions++;
                munmap(chunk, len_bytes);
                chunk = _chunks[id];
            }
#pragma omp atomic
            chunk_compute_total++;
            _mutex.unlock();

            return chunk;
        }

        std::filesystem::path tmp_path;
        _mutex.lock();
        std::stringstream ss;
        ss << this << "_" << std::this_thread::get_id() << "_" << _tmp_counter++;
        tmp_path = std::filesystem::path(_cache_dir) / ss.str();
        _mutex.unlock();
        int fd = open(tmp_path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
        if (fd == -1) {
            const int err = errno;
            std::stringstream msg;
            msg << "Chunked3d: open failed for " << tmp_path << ": " << std::strerror(err);
            throw std::runtime_error(msg.str());
        }
        int ret = ftruncate(fd, len_bytes);
        if (ret != 0) {
            const int err = errno;
            close(fd);
            std::stringstream msg;
            msg << "Chunked3d: ftruncate failed for " << tmp_path
                << " (" << len_bytes << " bytes): " << std::strerror(err);
            throw std::runtime_error(msg.str());
        }
        T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

        CHUNKT small = xt::empty<T>({s,s,s});
        CHUNKT large;
        if (_ds) {
            large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
            large = xt::full_like(large, C::FILL_V);

            readArea3D(large, offset, _ds, _cache);
        }

        _compute_f.template compute<CHUNKT,T>(large, small, offset);

        for(int i=0;i<len;i++)
            chunk[i] = (&small(0,0,0))[i];

        _mutex.lock();
        if (!_chunks.count(id)) {
            _chunks[id] = chunk;
            int ret = rename(tmp_path.string().c_str(), tgt_path.string().c_str());

            if (ret)
                throw std::runtime_error("oops rename failed!");
        }
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            munmap(chunk, len_bytes);
            unlink(tmp_path.string().c_str());
            chunk = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return chunk;
    }


    T *cache_chunk_safe_alloc(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};
            
        CHUNKT large;
        if (_ds) {
            large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
            large = xt::full_like(large, C::FILL_V);
            
            readArea3D(large, offset, _ds, _cache);
        }

        _compute_f.template compute<CHUNKT,T>(large, small, offset);

        T *chunk = nullptr;

        _mutex.lock();
        if (!_chunks.count(id)) {
            chunk = (T*)malloc(s*s*s*sizeof(T));
            memcpy(chunk, &small(0,0,0), s*s*s*sizeof(T));
            _chunks[id] = chunk;
        }
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            chunk = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return chunk;
    }

    // T *cache_chunk_safe_alloc(const cv::Vec3i &id)
    // {
    //     auto s = C::CHUNK_SIZE;
    //     CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
    //     large = xt::full_like(large, C::FILL_V);
    //     CHUNKT small = xt::empty<T>({s,s,s});
    //
    //     cv::Vec3i offset =
    //     {id[0]*s-_border,
    //         id[1]*s-_border,
    //         id[2]*s-_border};
    //
    //     readArea3D(large, offset, _ds, _cache);
    //
    //     _compute_f.template compute<CHUNKT,T>(large, small);
    //
    //     _mutex.lock();
    //     if (!_chunks.count(id))
    //         _chunks[id] = small;
    //     else {
    //         #pragma omp atomic
    //         chunk_compute_collisions++;
    //         delete small;
    //         small = _chunks[id];
    //     }
    //     #pragma omp atomic
    //     chunk_compute_total++;
    //     _mutex.unlock();
    //
    //     return small;
    // }

    T *cache_chunk_safe(const cv::Vec3i &id)
    {
        if (_cache_dir.empty())
            return cache_chunk_safe_alloc(id);
        else
            return cache_chunk_safe_mmap(id);
    }

    T *cache_chunk(const cv::Vec3i &id) {
        return cache_chunk_safe(id);
        // auto s = C::CHUNK_SIZE;
        // CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        // large = xt::full_like(large, C::FILL_V);
        // CHUNKT *small = new CHUNKT();
        // *small = xt::empty<T>({s,s,s});
        //
        // cv::Vec3i offset =
        // {id[0]*s-_border,
        //     id[1]*s-_border,
        //     id[2]*s-_border};
        //
        //     readArea3D(large, offset, _ds, _cache);
        //
        //     _compute_f.template compute<CHUNKT,T>(large, *small);
        //
        //     _chunks[id] = small;
        //
        // return small;
    }

    T *chunk(const cv::Vec3i &id) {
        if (!_chunks.count(id))
            return cache_chunk(id);
        return _chunks[id];
    }

    T *chunk_safe(const cv::Vec3i &id) {
        T *chunk = nullptr;
        _mutex.lock_shared();
        if (_chunks.count(id)) {
            chunk = _chunks[id];
            _mutex.unlock();
        }
        else {
            _mutex.unlock();
            chunk = cache_chunk_safe(id);
        }

        return chunk;
    }
    
    std::vector<int> shape() {
        return _shape;
    }

    std::unordered_map<cv::Vec3i,T*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache *_cache;
    size_t _border;
    C &_compute_f;
    std::shared_mutex _mutex;
    uint64_t _tmp_counter = 0;
    std::filesystem::path _cache_dir;
    bool _persistent = true;
    std::vector<int> _shape;
};

void print_accessor_stats();

template <typename T, typename C>
class Chunked3dAccessor
{
public:
    using CHUNKT = typename Chunked3d<T,C>::CHUNKT;

    Chunked3dAccessor(Chunked3d<T,C> &ar) : _ar(ar) {};

    static Chunked3dAccessor create(Chunked3d<T,C> &ar)
    {
        return Chunked3dAccessor(ar);
    }

    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk(p);
        }

        total++;

        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }



    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }

    T& safe_at(const cv::Vec3i &p)
    {        
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk_safe(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk_safe(p);
        }

        #pragma omp atomic
        total++;

        // size_t pos_xt = &_chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]) - &_chunk->operator()(0,0,0);
        // if (pos_xt != _ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})) {
        //     std::cout << pos_xt << cv::Vec3i({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]}) << _ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]}) << std::endl;
        //     throw std::runtime_error("fix calc_off!");
        // }

        // return _chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]);
        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }

    T& safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    void get_chunk(const cv::Vec3i &p)
    {
        miss++;
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk(id);
        _corner = id*C::CHUNK_SIZE;
    }

    void get_chunk_safe(const cv::Vec3i &p)
    {
        #pragma omp atomic
        miss++;
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk_safe(id);
        _corner = id*C::CHUNK_SIZE;
    }

    Chunked3d<T,C> &_ar;
protected:
    T *_chunk;
    cv::Vec3i _corner = {-1,-1,-1};
};

// ────────────────────────────────────────────────────────────────────────────────
//  CachedChunked3dInterpolator – thread-safe version
// ────────────────────────────────────────────────────────────────────────────────

template <typename T, typename C>
class CachedChunked3dInterpolator
{
public:
    using Acc   = Chunked3dAccessor<T, C>;
    using ArRef = Chunked3d<T, C>&;

    explicit CachedChunked3dInterpolator(ArRef ar)
        : _ar(ar), _shape(ar.shape())
    {}

    CachedChunked3dInterpolator(const CachedChunked3dInterpolator&)            = delete;
    CachedChunked3dInterpolator& operator=(const CachedChunked3dInterpolator&) = delete;

    /** Trilinear interpolation. */
    template <typename V>
    inline void Evaluate(const V& z, const V& y, const V& x, V* out) const
    {
        // ---- 1. get *this* thread’s private accessor ------------------------
        Acc& a = local_accessor();

        // ---- 2. fast trilinear interpolation --------------------
        cv::Vec3d f { val(z), val(y), val(x) };

        cv::Vec3i corner { static_cast<int>(std::floor(f[0])),
                           static_cast<int>(std::floor(f[1])),
                           static_cast<int>(std::floor(f[2])) };

        for (int i=0; i<3; ++i) {
            corner[i] = std::max(corner[i], 0);
            if (!_shape.empty())
                corner[i] = std::min(corner[i], _shape[i]-2);
        }

        const V fx = z - V(corner[0]);
        const V fy = y - V(corner[1]);
        const V fz = x - V(corner[2]);

        // clamp only once – cheaper than three branches per component
        const V cx = std::clamp(fx, V(0), V(1));
        const V cy = std::clamp(fy, V(0), V(1));
        const V cz = std::clamp(fz, V(0), V(1));

        // fetch the eight lattice points
        const V c000 = V(a.safe_at(corner));
        const V c100 = V(a.safe_at(corner + cv::Vec3i(1,0,0)));
        const V c010 = V(a.safe_at(corner + cv::Vec3i(0,1,0)));
        const V c110 = V(a.safe_at(corner + cv::Vec3i(1,1,0)));
        const V c001 = V(a.safe_at(corner + cv::Vec3i(0,0,1)));
        const V c101 = V(a.safe_at(corner + cv::Vec3i(1,0,1)));
        const V c011 = V(a.safe_at(corner + cv::Vec3i(0,1,1)));
        const V c111 = V(a.safe_at(corner + cv::Vec3i(1,1,1)));

        // interpolate
        const V c00 = (V(1)-cz)*c000 + cz*c001;
        const V c01 = (V(1)-cz)*c010 + cz*c011;
        const V c10 = (V(1)-cz)*c100 + cz*c101;
        const V c11 = (V(1)-cz)*c110 + cz*c111;

        const V c0  = (V(1)-cy)*c00 + cy*c01;
        const V c1  = (V(1)-cy)*c10 + cy*c11;

        *out = (V(1)-cx)*c0 + cx*c1;
    }

    // -------------------------------------------------------------------------
    double val(const double& v) const { return v; }
    template <typename JetT> double val(const JetT& v) const { return v.a; }

private:
    /** Return the accessor that is *unique to this combination of
     *  (interpolator instance, thread)*. */
    Acc& local_accessor() const
    {
        // Per-thread, bounded cache keyed by the underlying array address.
        // (Multiple interpolators over the same array share one accessor.)
        struct TLS {
            std::unordered_map<const void*, Acc> map;
            std::deque<const void*> order;          // FIFO ~ LRU-ish
        };
        thread_local TLS tls;

        constexpr std::size_t kMax = CCI_TLS_MAX;

        const void* key = static_cast<const void*>(&_ar);

        if (auto it = tls.map.find(key); it != tls.map.end()) {
            return it->second;
        }

        // Evict oldest if at capacity
        if (tls.map.size() >= kMax && !tls.order.empty()) {
            const void* old = tls.order.front();
            tls.order.pop_front();
            tls.map.erase(old);
        }

        auto [it2, inserted] = tls.map.emplace(key, Acc{_ar});
        if (inserted) tls.order.push_back(key);
        return it2->second;
    }

    ArRef               _ar;
    std::vector<int>    _shape;
};

struct Chunked3dFloatFromUint8
{
    Chunked3dFloatFromUint8(std::unique_ptr<z5::Dataset> &&ds, float scale, ChunkCache *cache, std::string const &cache_root, std::string const &unique_id) :
        _passthrough{unique_id},
        _x(_passthrough, ds.get(), cache, cache_root),
        _scale(scale),  // multiplying by this maps indices of the 'canonical' volume to indices of our dataset
        _ds(std::move(ds))  // take ownership of the dataset, as Chunked3d accesses it through a bare pointer
    {
    }

    float operator()(cv::Vec3d p)
    {
        // p has zyx ordering!
        p *= _scale;
        cv::Vec3i i{lround(p[0]), lround(p[1]), lround(p[2])};
        uint8_t x = _x.safe_at(i);
        return float{x} / 255.f;
    }

    float operator()(double z, double y, double x)
    {
        return operator()({z,y,x});
    }

    passTroughComputor _passthrough;
    Chunked3d<uint8_t, passTroughComputor> _x;
    float _scale;
    std::unique_ptr<z5::Dataset> _ds;
};

struct Chunked3dVec3fFromUint8
{
    Chunked3dVec3fFromUint8(std::vector<std::unique_ptr<z5::Dataset>> &&dss, float scale, ChunkCache *cache, std::string const &cache_root, std::string const &unique_id) :
        _passthrough_x{unique_id + "_x"},
        _passthrough_y{unique_id + "_y"},
        _passthrough_z{unique_id + "_z"},
        _x(_passthrough_x, dss[0].get(), cache, cache_root),
        _y(_passthrough_y, dss[1].get(), cache, cache_root),
        _z(_passthrough_z, dss[2].get(), cache, cache_root),
        _scale(scale),  // multiplying by this maps indices of the 'canonical' volume to indices of our three volumes
        _dss(std::move(dss))  // take ownership of the datasets here, as Chunked3d accesses them through a bare pointer
    {
    }

    cv::Vec3f operator()(cv::Vec3d p)
    {
        // Both p and returned vector have zyx ordering!
        p *= _scale;
        cv::Vec3i i{lround(p[0]), lround(p[1]), lround(p[2])};
        uint8_t x = _x.safe_at(i);
        uint8_t y = _y.safe_at(i);
        uint8_t z = _z.safe_at(i);
        return (cv::Vec3f{z, y, x} - cv::Vec3f{128.f, 128.f, 128.f}) / 127.f;
    }

    cv::Vec3f operator()(double z, double y, double x)
    {
        return operator()({z,y,x});
    }

    passTroughComputor _passthrough_x, _passthrough_y, _passthrough_z;
    Chunked3d<uint8_t, passTroughComputor> _x, _y, _z;
    float _scale;
    std::vector<std::unique_ptr<z5::Dataset>> _dss;
};
