// C++ read path for one Zarr array: plan, then resolve every chunk through
// the memory cache, the disk layers (in their own formats) and the remote, and
// copy into the caller's output buffers.
#pragma once

#include "chunk_io.hpp"
#include "output.hpp"
#include "sampling_math.hpp"
#include "dense_affine.hpp"
#include "codecs.hpp"
#include "disk_cache.hpp"
#include "http.hpp"
#include "memory_cache.hpp"
#include "thread_pool.hpp"
#include "write_behind.hpp"
#include "zarr_storage.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vcr {

// Coordinate providers let point arrays and procedural grids share the sampler.
template <class T>
struct PointCoordinates {
    using value_type = T;
    const T* data;
    std::size_t ndim;
    std::int64_t base(std::size_t) const { return 0; }
    const T* operator()(std::size_t p, T*) const { return data + p * ndim; }
    void prefetch(std::size_t p) const {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(data + p * ndim);
#endif
    }
};

// A regular 3-D output grid in source-array coordinates. No voxel-sized
// coordinate buffer: each point is reconstructed in the chosen scalar type.
template<class Real>
struct BasicAffineCoordinates {
    using value_type = Real;
    std::int64_t base(std::size_t) const { return 0; }
    std::array<std::size_t, 3> shape;
    std::array<Real, 3> origin;
    std::array<Real, 9> axes;  // row major: columns are output-axis steps

    const Real* operator()(std::size_t p, Real* c) const {
        const auto x = p % shape[2];
        p /= shape[2];
        const auto y = p % shape[1], z = p / shape[1];
        for (std::size_t i = 0; i < 3; ++i) {
            if constexpr (std::is_same_v<Real,float>) c[i]=sampling::affine(origin[i],axes.data()+3*i,float(z),float(y),float(x));
            else c[i] = origin[i] + axes[3*i] * z + axes[3*i+1] * y + axes[3*i+2] * x;
        }
        return c;
    }
    void prefetch(std::size_t) const {}
};

using AffineCoordinates = BasicAffineCoordinates<double>;

// Float32 CT paths: global location is integer, arithmetic local.
template<class Coordinates>
struct RebasedCoordinates : Coordinates {
    std::array<std::int64_t, 3> offset{};
    std::int64_t base(std::size_t axis) const { return offset[axis]; }
};
using FloatPoints = RebasedCoordinates<PointCoordinates<float>>;
using FloatAffine = RebasedCoordinates<BasicAffineCoordinates<float>>;

// A chunk an offline read needs is in none of the disk layers.
struct CacheMiss : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An owning reference for the optional Python prefetch scheduler. No crop
// output is allocated; nullptr represents a chunk absent from the source.
struct ChunkHandle {
    ChunkPtr value;
};

// Linear chunk index -> dense id, for grouping point reads by chunk (open addressing).
class ChunkIds {
public:
    // The id of `key` (>= 0); if absent, inserts it with id `fresh` and returns that.
    std::uint32_t find_or_insert(std::int64_t key, std::uint32_t fresh)
    {
        if (2 * (size_ + 1) > keys_.size())
            grow();
        const std::size_t mask = keys_.size() - 1;
        for (std::size_t i = slot(key);; i = (i + 1) & mask) {
            if (keys_[i] == key)
                return ids_[i];
            if (keys_[i] == kEmpty) {
                keys_[i] = key;
                ids_[i] = fresh;
                ++size_;
                return fresh;
            }
        }
    }

private:
    static constexpr std::int64_t kEmpty = -1;

    std::size_t slot(std::int64_t key) const
    {
        return static_cast<std::size_t>((static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ull) >> shift_);
    }

    void grow()
    {
        std::vector<std::int64_t> keys = std::move(keys_);
        std::vector<std::uint32_t> ids = std::move(ids_);
        const std::size_t capacity = std::max<std::size_t>(64, 2 * keys.size());
        keys_.assign(capacity, kEmpty);
        ids_.assign(capacity, 0);
        shift_ = 64 - std::countr_zero(capacity);
        const std::size_t mask = capacity - 1;
        for (std::size_t j = 0; j < keys.size(); ++j) {
            if (keys[j] == kEmpty)
                continue;
            std::size_t i = slot(keys[j]);
            while (keys_[i] != kEmpty)
                i = (i + 1) & mask;
            keys_[i] = keys[j];
            ids_[i] = ids[j];
        }
    }

    std::vector<std::int64_t> keys_;
    std::vector<std::uint32_t> ids_;
    std::size_t size_ = 0;
    int shift_ = 64;
};

// One directory holding chunk files of the array, in one format.
struct LayerSpec {
    // "array": the array itself (a local store; an absent chunk is not stored).
    // "decoded": vc_reader / vesuvius decoded cache (<key>.missing markers).
    // "vc3d-mirror": VC3D source mirror (<key>.empty markers).
    // "vc3d-delta3d": VC3D Delta3D cache (<empty_root>/<key>.empty markers).
    std::string kind;
    std::string root;
    std::string codec;       // how this layer stores chunks: "" (raw), "blosc", "vc-delta3d"
    std::string empty_root;  // vc3d-delta3d only
    bool allow_lossy = false;
};

struct ArraySpec {
    Index shape, chunks;
    std::size_t itemsize = 1;
    std::vector<std::uint8_t> fill;  // one element's bytes
    std::string separator = ".";
    std::string identity;            // memory-cache namespace
    std::vector<LayerSpec> layers;
    int write_layer = -1;            // index into layers, or -1
    std::string remote_url;          // "" for a purely local array
    std::string remote_codec;
    std::string source_root, source_prefix;  // v3 physical objects; cache keys stay logical
    Index shard_chunks;
    bool shard_index_end = true, shard_index_crc = false;
    bool offline = false;            // never fetch: a chunk in no layer raises CacheMiss
    std::string read_mode = "slab";  // raw local chunks: "slab" (pread rows) or "mmap"
    bool hint = false;               // queue readahead for every piece before reading
    std::size_t threads = 16;
    std::size_t writer_threads = 4;                    // background disk-cache writes
    std::size_t max_pending_write_bytes = 1ull << 30;  // reads wait beyond this much unwritten data
    std::shared_ptr<DiskCache> disk_cache;
    HttpOptions http;
};

struct Stats {
    std::atomic<std::uint64_t> memory_hits{0}, pending_hits{0}, disk_hits{0}, remote_fetches{0},
        remote_missing{0}, remote_bytes{0}, write_errors{0};
};

class Array {
public:
    Array(ArraySpec spec, std::shared_ptr<MemoryCache> cache)
        : spec_(std::move(spec)), cache_(std::move(cache)), pool_(spec_.threads), maps_(8192)
    {
        const std::size_t nd = spec_.shape.size();
        if (nd == 0 || spec_.chunks.size() != nd)
            throw std::invalid_argument("shape and chunks must have the same, nonzero rank");
        if (spec_.fill.size() != spec_.itemsize)
            throw std::invalid_argument("fill must be one element");
        chunk_strides_ = c_strides(spec_.chunks, spec_.itemsize);
        chunk_bytes_ = static_cast<std::size_t>(chunk_strides_[0] * spec_.chunks[0]);
        grid_.resize(nd);
        chunk_shift_.resize(nd, -1);
        for (std::size_t i = 0; i < nd; ++i) {
            grid_[i] = (spec_.shape[i] + spec_.chunks[i] - 1) / spec_.chunks[i];
            const auto c = static_cast<std::uint64_t>(spec_.chunks[i]);
            if (std::has_single_bit(c))
                chunk_shift_[i] = std::countr_zero(c);
        }
        for (auto& l : spec_.layers)
            codecs_.push_back(parse_codec(l.codec));
        remote_codec_ = parse_codec(spec_.remote_codec);
        if (!spec_.remote_url.empty() && !spec_.offline)
            http_ = std::make_unique<HttpClient>(spec_.http);
        if (!spec_.source_root.empty() && !spec_.offline)
            source_ = std::make_unique<ZarrStorage>(spec_.source_root, spec_.source_prefix, spec_.separator,
                spec_.shard_chunks, spec_.shard_index_end, spec_.shard_index_crc, chunk_bytes_, http_.get(),
                stats_.remote_fetches, stats_.remote_bytes);
        if (spec_.write_layer >= 0 && codecs_.at(spec_.write_layer) == Codec::Delta3d && nd != 3)
            throw std::invalid_argument("Delta3D caches hold 3-D chunks only");
        if (spec_.write_layer >= 0)
            writes_ = std::make_unique<WriteBehind>(spec_.writer_threads, spec_.max_pending_write_bytes);
    }

    // Waits for every queued disk-cache write.
    void flush()
    {
        if (writes_)
            writes_->flush();
    }

    std::size_t pending_writes() { return writes_ ? writes_->outstanding() : 0; }
    std::size_t ndim() const noexcept { return spec_.shape.size(); }
    const ArraySpec& spec() const noexcept { return spec_; }
    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) { pool_.parallel_for(n,fn); }
    std::size_t itemsize() const noexcept { return spec_.itemsize; }

    void set_fetcher(std::function<ChunkHandle(const Index&)> fetcher)
    {
        fetcher_.store(std::make_shared<const Fetcher>(std::move(fetcher)));
    }

    // The scheduler calls this directly, bypassing its own demand callback.
    // It shares the normal decoder, memory cache, fetch deduplication and writes.
    ChunkHandle load_chunk(const Index& idx)
    {
        validate_chunk(idx);
        const auto k = key(idx), ck = spec_.identity + "|" + k;
        if (cache_) {
            auto [found, chunk] = cache_->get(ck);
            if (found) return {chunk};
        }
        if (writes_) {
            auto [found, chunk] = writes_->pending(k);
            if (found) return {chunk};
        }
        ChunkPtr chunk;
        bool found = false;
        for (std::size_t li = 0; li < spec_.layers.size(); ++li) {
            if (auto value = read_layer_chunk(li, k)) {
                chunk = *value;
                found = true;
                break;
            }
        }
        if (!found) {
            if (spec_.offline)
                throw CacheMiss("chunk " + k + " of " + spec_.identity + " is not in the disk cache (offline)");
            if (http_ || source_) chunk = fetch(k);
        }
        if (cache_) cache_->put(ck, chunk);
        return {chunk};
    }

    // Cross-process completion requires publication, including missing markers.
    // A refused/failed cache write returns false; demand still owns its chunk.
    bool publish_chunk(const Index& idx)
    {
        validate_chunk(idx);
        const auto k = key(idx);
        if (writes_) writes_->wait(k);
        for (const auto& layer : spec_.layers)
            if (layer.kind == "array" || exists(layer.root + "/" + k) || exists(marker_path(layer, k)))
                return true;
        return false;
    }

    // Credentials for later fetches (e.g. refreshed temporary keys).
    void set_auth(Auth auth)
    {
        if (http_)
            http_->set_auth(std::move(auth));
    }

    // Reads every region [start, start+size) into its C-order output buffer.
    // Parts outside the array, and chunks that are not stored, read as fill.
    // `stream`: raw local chunks missing from the memory cache are copied straight
    // into the outputs without being read whole and inserted (callers with their
    // own cache, e.g. the CUDA reader). Memory-cache hits are still used.
    void read_many(const std::vector<Index>& starts, const std::vector<Index>& sizes,
                   const std::vector<std::uint8_t*>& outs, const BoxOutput* output = nullptr,
                   bool stream = false)
    {
        const std::size_t nd = spec_.shape.size();
        std::vector<Index> out_strides(starts.size());
        std::map<std::int64_t, std::size_t> slot;  // linear chunk index -> work item (ordered: locality)
        std::vector<Work> work;
        for (std::size_t r = 0; r < starts.size(); ++r) {
            const Index& start = starts[r];
            const Index& size = sizes[r];
            if (start.size() != nd || size.size() != nd)
                throw std::invalid_argument("region rank does not match the array");
            out_strides[r] = c_strides(size, output ? output->itemsize : spec_.itemsize);
            if (std::any_of(size.begin(), size.end(), [](std::int64_t n) { return n <= 0; }))
                continue;  // nothing to read or fill
            Index lo(nd), hi(nd);
            bool empty = false, clipped = false;
            for (std::size_t i = 0; i < nd; ++i) {
                lo[i] = std::max<std::int64_t>(start[i], 0);
                hi[i] = std::min<std::int64_t>(start[i] + size[i], spec_.shape[i]);
                empty |= hi[i] <= lo[i];
                clipped |= lo[i] != start[i] || hi[i] != start[i] + size[i];
            }
            if (empty || clipped)  // outside the array reads as fill; the inside is overwritten below
                fill_box(outs[r], out_strides[r], size, output ? output->fill : spec_.fill);
            if (empty)
                continue;
            Index c0(nd), c1(nd);
            for (std::size_t i = 0; i < nd; ++i) {
                c0[i] = lo[i] / spec_.chunks[i];
                c1[i] = (hi[i] - 1) / spec_.chunks[i];
            }
            Index idx = c0;
            do {
                std::int64_t linear = 0;
                Piece p;
                p.region = r;
                p.src_lo.resize(nd);
                p.extent.resize(nd);
                Index dst_lo(nd);
                for (std::size_t i = 0; i < nd; ++i) {
                    linear = linear * grid_[i] + idx[i];
                    const std::int64_t base = idx[i] * spec_.chunks[i];
                    const std::int64_t a = std::max(lo[i], base), b = std::min(hi[i], base + spec_.chunks[i]);
                    p.src_lo[i] = a - base;
                    p.extent[i] = b - a;
                    dst_lo[i] = a - start[i];
                }
                p.dst = outs[r] + offset_of(dst_lo, out_strides[r]);
                auto [it, fresh] = slot.emplace(linear, work.size());
                if (fresh)
                    work.push_back(Work{idx, {}, {}});
                work[it->second].pieces.push_back(std::move(p));
            } while (next_index(idx, c0, c1));
        }
        std::vector<Work> ordered;  // chunk order: locality
        ordered.reserve(work.size());
        for (auto& [linear, i] : slot)
            ordered.push_back(std::move(work[i]));
        run(ordered, out_strides, nullptr, output, stream);
    }

    void read_many_float(const std::vector<Index>& starts, const std::vector<Index>& sizes,
                         const std::vector<std::uint8_t*>& outs, const ElementType& type, bool out64)
    {
        auto output = out64 ? BoxOutput::make<double>(type, spec_.fill) : BoxOutput::make<float>(type, spec_.fill);
        read_many(starts, sizes, outs, &output);
    }

    // Reads the elements at `n` points into `out` (one element each, in order).
    // `coords` holds each point's indices, C order. Points outside the array read as fill.
    void read_points(const std::int64_t* coords, std::size_t n, std::uint8_t* out)
    {
        if (ndim() == 3)
            read_points_as<3>(PointCoordinates{coords, ndim()}, n, out);
        else
            read_points_as<0>(PointCoordinates{coords, ndim()}, n, out);
    }

    // Element nearest to each real-valued point (voxel i at coordinate i, ties to even),
    // written as stored. Points outside the array, and non-finite ones, read as fill.
    void sample_nearest(const double* coords, std::size_t n, std::uint8_t* out)
    {
        if (ndim() == 3)
            read_points_as<3>(PointCoordinates{coords, ndim()}, n, out);
        else
            read_points_as<0>(PointCoordinates{coords, ndim()}, n, out);
    }

    // N-linear interpolation at real-valued points (voxel i at coordinate i): the weighted
    // sum of the 2^k neighbouring elements, k being the axes on which the point is not
    // on the grid. Neighbours outside the array read as fill. Non-finite points give NaN.
    // Computed in double; `out` holds float (out64 false) or double values.
    void sample_linear(const double* coords, std::size_t n, const ElementType& type, bool out64, void* out)
    {
        sample_linear_coordinates(PointCoordinates{coords, ndim()}, n, type, out64, out);
    }

    void read_affine(const AffineCoordinates& grid, std::size_t n, int order,
                     const ElementType& type, bool out64, std::uint8_t* out,
                     bool float_output = false)
    {
        if (ndim() != 3)
            throw std::invalid_argument("affine reads require a 3-D source array");
        const bool same_float = type.kind == 'f' && !type.swap && type.itemsize == (out64 ? 8 : 4);
        if (order == 0 && float_output && !same_float) {
            if (n > std::numeric_limits<std::size_t>::max() / spec_.itemsize)
                throw std::invalid_argument("nearest crop is too large");
            std::vector<std::uint8_t> raw(n * spec_.itemsize);
            read_points_as<3>(grid, n, raw.data());
            auto convert = out64 ? BoxOutput::make<double>(type, spec_.fill) : BoxOutput::make<float>(type, spec_.fill);
            if (n) convert.copy(raw.data(), {static_cast<std::int64_t>(spec_.itemsize)}, out,
                                {static_cast<std::int64_t>(convert.itemsize)}, {static_cast<std::int64_t>(n)});
        } else if (order == 0) {
            read_points_as<3>(grid, n, out);
        } else {
            sample_linear_coordinates(grid, n, type, out64, out);
        }
    }

    template <class Coordinates>
    void sample_linear_coordinates(const Coordinates& coords, std::size_t n, const ElementType& type,
                                   bool out64, void* out)
    {
        if (spec_.shape.size() > kMaxSampleDims)
            throw std::invalid_argument("sample() supports at most 16 dimensions");
        // The element loads, chosen once: native-order numbers inline, the rest through `type`.
        const auto dispatch = [&](auto load) {
            if (ndim() == 3)
                sample_linear_as<3>(coords, n, type, out64, out, load);
            else
                sample_linear_as<0>(coords, n, type, out64, out, load);
        };
        type.visit_load([&](auto load) {
            dispatch([load](const std::uint8_t* p) { return static_cast<double>(load(p)); });
        });
    }

    // Float32 uint8 CT entry: select a dense workspace only when its volume is
    // bounded relative to the requested samples; otherwise plan direct chunks.
    template<class Coordinates>
    void sample_u8_float32(const Coordinates& coords, std::size_t n, int order, void* out) {
        if (ndim() != 3 || spec_.itemsize != 1)
            throw std::invalid_argument("float32 sampling requires 3-D uint8 CT");
        if (order == 0) read_points_as<3>(coords,n,static_cast<std::uint8_t*>(out));
        else if (order == 1) {
            if constexpr (std::is_same_v<Coordinates,FloatAffine>) {
                sampling::DenseAffineBlock block;
                if (block.plan(coords,n)) {
                    std::vector<std::uint8_t> raw(block.count);
                    bool resident=true;
                    try {
                        read_many({Index(block.start.begin(),block.start.end())},
                                  {Index(block.size.begin(),block.size.end())},{raw.data()});
                    } catch (const CacheMiss&) {
                        // An offline cache may contain the sampled chunks but
                        // not every chunk in the enclosing box.
                        resident=false;
                    }
                    if (resident) {
                        block.sample(coords,raw.data(),static_cast<float*>(out),
                            [&](std::size_t count,const auto& fn) {
                                if (pool_.size()==1) for (std::size_t z=0;z<count;++z) fn(z);
                                else pool_.parallel_for(count,fn);
                            });
                        return;
                    }
                }
            }
            sample_linear_as<3>(coords,n,ElementType{'u',1,false},false,out,
                               [](const std::uint8_t* at) { return float(*at); });
        }
        else throw std::invalid_argument("order must be 0 or 1");
    }

    const Stats& stats() const noexcept { return stats_; }
    std::string last_write_error()
    {
        std::lock_guard lock(error_mutex_);
        return last_write_error_;
    }

private:
    // Advances idx through the box [c0, c1] in C order; false after the last index.
    static bool next_index(Index& idx, const Index& c0, const Index& c1)
    {
        for (std::size_t ax = idx.size(); ax-- > 0;) {
            if (++idx[ax] <= c1[ax])
                return true;
            idx[ax] = c0[ax];
        }
        return false;
    }

    static constexpr std::size_t kPointBatch = std::size_t{1} << 20;  // points per plan: bounds its memory
    static constexpr std::size_t kPlanGrain = std::size_t{1} << 14;   // points per planning segment, at least
    static constexpr std::size_t kMaxSampleDims = 16;
    // Tags of sample_linear's neighbour reads for points whose neighbours span chunks:
    // the flag, the planning segment, and the neighbour's slot within that segment.
    static constexpr std::uint64_t kSplit = std::uint64_t{1} << 63;
    static constexpr int kSegmentShift = 40;
    static constexpr std::uint64_t kSlotMask = (std::uint64_t{1} << kSegmentShift) - 1;

    struct Piece {
        std::size_t region = 0;
        Index src_lo, extent;
        std::uint8_t* dst = nullptr;
    };
    // Built in place, field by field: a temporary copied into a vector costs store-forwarding
    // stalls, and resizing a vector of these leaves them uninitialized.
    struct Point {
        std::int64_t src;   // byte offset in the chunk
        std::uint64_t tag;  // which output the element is for: the PointOp's business
        Point() {}
    };
    // Consecutive points [first, first + count) that one chunk answers alone (see plan_points).
    struct Run {
        std::uint64_t first, count;
        std::int64_t lo, hi;  // the lowest and highest byte offset in the chunk the points read from
    };
    // Points per run: shorter runs are planned as single points (recomputing a point costs
    // more than reading its planned offset); longer ones are cut, so runs can be shared out.
    static constexpr std::uint64_t kRunMin = 8, kRunMax = std::uint64_t{1} << 12;
    void validate_chunk(const Index& idx) const
    {
        if (idx.size() != grid_.size())
            throw std::invalid_argument("chunk index rank does not match the array");
        for (std::size_t i = 0; i < idx.size(); ++i)
            if (idx[i] < 0 || idx[i] >= grid_[i])
                throw std::out_of_range("chunk index outside the array");
    }

    // Everything one chunk supplies: boxes (read_many) or single elements (point reads).
    struct Work {
        Index idx;
        std::vector<Piece> pieces;
        std::span<const Point> points;
        std::span<const Run> runs;

        bool has_points() const noexcept { return !points.empty() || !runs.empty(); }
    };
    // What a point read does with a chunk's points and runs once the chunk is resolved.
    struct PointOp {
        std::int64_t reach = 0;  // bytes each point reads from its offset on
        // `data` holds the chunk's bytes from byte offset `origin` on.
        std::function<void(const std::uint8_t* data, std::int64_t origin, const Work&)> read;
        std::function<void(const Work&)> fill;  // the chunk is not stored
    };

    // Resolves every chunk in parallel, in the order given.
    void run(std::vector<Work>& work, const std::vector<Index>& out_strides, const PointOp* op,
             const BoxOutput* output = nullptr, bool stream = false)
    {
        if (spec_.hint && (!cache_ || stream))
            pool_.parallel_for(work.size(), [&](std::size_t i) { hint(work[i]); });
        pool_.parallel_for(work.size(), [&](std::size_t i) { resolve(work[i], out_strides, op, output, stream); });
    }

    std::size_t plan_segments(std::size_t n) const
    {
        return std::clamp<std::size_t>(n / kPlanGrain, 1, 4 * (pool_.size() + 1));
    }

    // Point planning only divides nonnegative voxel indices. Common power-of-two
    // chunk sizes need a shift; arbitrary chunk shapes keep exact integer division.
    std::int64_t chunk_index(std::int64_t voxel, std::size_t axis) const
    {
        return chunk_shift_[axis] >= 0 ? voxel >> chunk_shift_[axis] : voxel / spec_.chunks[axis];
    }

    // Groups point reads by chunk. The n points are split into plan_segments(n) contiguous
    // ranges, each planned in order on one thread: locate(segment, p, scratch, emit) reports
    // what point p reads (scratch: room for nd indices; index: nd indices inside the array):
    //  - emit.point(index, tag): the point is answered from the chunk holding `index` alone,
    //    and the PointOp can recompute everything it needs from the point's number, `tag`;
    //  - emit(index, tag): one element the point needs.
    // Consecutive points answered from one chunk become a Run, which stores no per-point data;
    // the other elements become Points (byte offset and tag). Returns one Work per chunk, in
    // chunk order, whose points and runs (in emission order) are stored in `points` and `runs`.
    template <std::size_t N = 0, class Locate>
    std::vector<Work> plan_points(std::size_t n, std::vector<Point>& points, std::vector<Run>& runs, Locate&& locate)
    {
        const std::size_t nd = N ? N : spec_.shape.size();
        const std::size_t segs = plan_segments(n);
        struct Entry {
            std::int64_t src;
            std::uint64_t tag;
            std::uint32_t id;
            Entry(std::int64_t src, std::uint64_t tag, std::uint32_t id) : src(src), tag(tag), id(id) {}
        };
        struct RunEntry {
            Run run;
            std::uint32_t id;
            RunEntry(const Run& run, std::uint32_t id) : run(run), id(id) {}
        };
        struct Segment {
            ChunkIds ids;
            std::vector<std::int64_t> linear;  // local id -> linear chunk index
            std::vector<std::size_t> next;     // local id -> element count; later its next slot in `points`
            std::vector<std::size_t> next_run; // local id -> run count; later its next slot in `runs`
            std::vector<Entry> entries;
            std::vector<RunEntry> run_entries;
        };
        struct Emit {
            const Array& a;
            Segment& g;
            std::int64_t last = -1;  // the last chunk looked up, and its local id
            std::uint32_t id = 0;
            Run run{0, 0, 0, 0};  // the open run, of chunk run_id with first element run_base
            std::uint32_t run_id = 0;
            Index run_base;
            std::int64_t run_src[kRunMin];  // the offsets of its first points

            // The local id of the chunk holding `c`; returns c's byte offset in it. base: the
            // chunk's first element, or null.
            std::int64_t find(const std::int64_t* c, std::uint32_t& cid, std::int64_t* base)
            {
                std::int64_t linear = 0, offset = 0;
                for (std::size_t i = 0; i < (N ? N : a.grid_.size()); ++i) {
                    const std::int64_t q = a.chunk_index(c[i], i);
                    const std::int64_t r = c[i] - q * a.spec_.chunks[i];
                    linear = linear * a.grid_[i] + q;
                    offset += r * a.chunk_strides_[i];
                    if (base)
                        base[i] = c[i] - r;
                }
                if (linear != last) {  // runs of points in one chunk skip the lookup
                    last = linear;
                    id = g.ids.find_or_insert(linear, static_cast<std::uint32_t>(g.linear.size()));
                    if (id == g.linear.size()) {
                        g.linear.push_back(linear);
                        g.next.push_back(0);
                        g.next_run.push_back(0);
                    }
                }
                cid = id;
                return offset;
            }
            void element(std::int64_t src, std::uint64_t tag, std::uint32_t cid)
            {
                ++g.next[cid];
                g.entries.emplace_back(src, tag, cid);
            }
            void operator()(const std::int64_t* c, std::uint64_t tag)
            {
                std::uint32_t cid;
                const std::int64_t src = find(c, cid, nullptr);
                element(src, tag, cid);
            }
            void point(const std::int64_t* c, std::uint64_t tag)
            {
                if (run.count && tag == run.first + run.count && run.count < kRunMax) {
                    bool inside = true;  // in the run's chunk: no division or lookup
                    std::int64_t src = 0;
                    for (std::size_t i = 0; i < (N ? N : run_base.size()); ++i) {
                        const std::int64_t d = c[i] - run_base[i];
                        inside &= d >= 0 && d < a.spec_.chunks[i];
                        src += d * a.chunk_strides_[i];
                    }
                    if (inside) {
                        if (run.count < kRunMin)
                            run_src[run.count] = src;
                        ++run.count;
                        run.lo = std::min(run.lo, src);
                        run.hi = std::max(run.hi, src);
                        return;
                    }
                }
                close();
                const std::int64_t src = find(c, run_id, run_base.data());
                run = Run{tag, 1, src, src};
                run_src[0] = src;
            }
            void close()
            {
                if (run.count < kRunMin) {
                    for (std::uint64_t j = 0; j < run.count; ++j)
                        element(run_src[j], run.first + j, run_id);
                } else {
                    ++g.next_run[run_id];
                    g.run_entries.emplace_back(run, run_id);
                }
                run.count = 0;
            }
        };
        std::vector<Segment> seg(segs);
        pool_.parallel_for(segs, [&](std::size_t s) {
            Segment& g = seg[s];
            const std::size_t p0 = n * s / segs, p1 = n * (s + 1) / segs;
            g.entries.reserve(p1 - p0);
            Index scratch(nd);
            Emit emit{*this, g};
            emit.run_base.resize(nd);
            for (std::size_t p = p0; p < p1; ++p)
                locate(s, p, scratch.data(), emit);
            emit.close();
        });
        // Chunks in order; within a chunk, each segment's elements after the earlier segments'.
        // Every segment's chunks get a dense id (in order of appearance), then their rank.
        ChunkIds all;
        std::vector<std::int64_t> seen;  // dense id -> linear chunk index
        std::vector<std::vector<std::uint32_t>> global(segs);
        for (std::size_t s = 0; s < segs; ++s) {
            const Segment& g = seg[s];
            global[s].resize(g.linear.size());
            for (std::size_t id = 0; id < g.linear.size(); ++id) {
                const auto fresh = static_cast<std::uint32_t>(seen.size());
                global[s][id] = all.find_or_insert(g.linear[id], fresh);
                if (global[s][id] == fresh)
                    seen.push_back(g.linear[id]);
            }
        }
        std::vector<std::uint32_t> order(seen.size()), rank(seen.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return seen[a] < seen[b]; });
        std::vector<std::int64_t> chunks(seen.size());
        for (std::size_t c = 0; c < order.size(); ++c) {
            chunks[c] = seen[order[c]];
            rank[order[c]] = static_cast<std::uint32_t>(c);
        }
        // Counting sort of the elements and of the runs by chunk.
        std::vector<std::size_t> start(chunks.size() + 1, 0), run_start(chunks.size() + 1, 0);
        for (std::size_t s = 0; s < segs; ++s) {
            const Segment& g = seg[s];
            for (std::size_t id = 0; id < g.linear.size(); ++id) {
                global[s][id] = rank[global[s][id]];
                start[global[s][id] + 1] += g.next[id];
                run_start[global[s][id] + 1] += g.next_run[id];
            }
        }
        for (std::size_t c = 0; c < chunks.size(); ++c) {
            start[c + 1] += start[c];
            run_start[c + 1] += run_start[c];
        }
        std::vector<std::size_t> cursor(start.begin(), start.end() - 1);
        std::vector<std::size_t> run_cursor(run_start.begin(), run_start.end() - 1);
        for (std::size_t s = 0; s < segs; ++s) {
            Segment& g = seg[s];
            for (std::size_t id = 0; id < g.linear.size(); ++id) {
                const std::size_t count = g.next[id], run_count = g.next_run[id];
                g.next[id] = cursor[global[s][id]];
                cursor[global[s][id]] += count;
                g.next_run[id] = run_cursor[global[s][id]];
                run_cursor[global[s][id]] += run_count;
            }
        }
        points.resize(start.back());
        runs.resize(run_start.back());
        pool_.parallel_for(segs, [&](std::size_t s) {
            Segment& g = seg[s];
            for (const Entry& e : g.entries) {
                Point& pt = points[g.next[e.id]++];
                pt.src = e.src;
                pt.tag = e.tag;
            }
            for (const RunEntry& e : g.run_entries)
                runs[g.next_run[e.id]++] = e.run;
        });
        std::vector<Work> work(chunks.size());
        for (std::size_t c = 0; c < chunks.size(); ++c) {
            Work& w = work[c];
            w.idx.resize(nd);
            std::int64_t linear = chunks[c];
            for (std::size_t i = nd; i-- > 0;) {
                w.idx[i] = linear % grid_[i];
                linear /= grid_[i];
            }
            w.points = std::span<const Point>(points.data() + start[c], start[c + 1] - start[c]);
            w.runs = std::span<const Run>(runs.data() + run_start[c], run_start[c + 1] - run_start[c]);
        }
        return work;
    }

    // Runs fn over `pts` in parts, in parallel when there are many (one chunk holding most
    // of a batch's points would otherwise leave the other threads idle).
    template <class Fn>
    void for_points(std::span<const Point> pts, Fn&& fn)
    {
        const std::size_t parts = pts.size() / kPlanGrain;
        if (parts < 2) {
            fn(pts);
            return;
        }
        pool_.parallel_for(parts, [&](std::size_t i) {
            const std::size_t a = pts.size() * i / parts, b = pts.size() * (i + 1) / parts;
            fn(pts.subspan(a, b - a));
        });
    }

    // for_points for runs: parts of about equal point counts, cut between runs.
    template <class Fn>
    void for_runs(std::span<const Run> runs, Fn&& fn)
    {
        std::uint64_t total = 0;
        for (const Run& r : runs)
            total += r.count;
        const std::uint64_t parts = total / kPlanGrain;
        if (parts < 2) {
            fn(runs);
            return;
        }
        std::vector<std::size_t> cut{0};
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i + 1 < runs.size(); ++i) {
            sum += runs[i].count;
            if (sum * parts >= total * cut.size())
                cut.push_back(i + 1);
        }
        cut.push_back(runs.size());
        pool_.parallel_for(cut.size() - 1, [&](std::size_t i) { fn(runs.subspan(cut[i], cut[i + 1] - cut[i])); });
    }

    // Byte offset of the element at `idx` in the chunk whose first element is `base`.
    template <std::size_t N = 0>
    std::int64_t offset_in(const std::int64_t* idx, const std::int64_t* base) const
    {
        std::int64_t src = 0;
        for (std::size_t i = 0; i < (N ? N : chunk_strides_.size()); ++i)
            src += (idx[i] - base[i]) * chunk_strides_[i];
        return src;
    }

    Index chunk_base(const Work& w) const
    {
        Index base(w.idx.size());
        for (std::size_t i = 0; i < base.size(); ++i)
            base[i] = w.idx[i] * spec_.chunks[i];
        return base;
    }

    template <std::size_t K>
    static void copy_points(const std::uint8_t* data, std::int64_t origin, std::span<const Point> pts,
                            std::uint8_t* out)
    {
        for (const Point& pt : pts)
            std::memcpy(out + pt.tag * K, data + (pt.src - origin), K);
    }

    // Reads one element per located point into out[tag] (see plan_points for `locate`, which
    // only emits elements here: planning a point costs less than recomputing it).
    template <std::size_t N, class Locate>
    void gather_points(std::size_t n, std::uint8_t* out, Locate&& locate)
    {
        std::vector<Point> points;
        std::vector<Run> runs;
        auto work = plan_points<N>(n, points, runs, locate);
        const std::size_t isz = spec_.itemsize;
        PointOp op;
        op.reach = static_cast<std::int64_t>(isz);
        op.read = [&](const std::uint8_t* data, std::int64_t origin, const Work& w) {
            for_points(w.points, [&](std::span<const Point> part) {
                switch (isz) {
                case 1: return copy_points<1>(data, origin, part, out);
                case 2: return copy_points<2>(data, origin, part, out);
                case 4: return copy_points<4>(data, origin, part, out);
                case 8: return copy_points<8>(data, origin, part, out);
                default:
                    for (const Point& pt : part)
                        std::memcpy(out + pt.tag * isz, data + (pt.src - origin), isz);
                }
            });
        };
        op.fill = [&](const Work& w) {
            for (const Point& pt : w.points)
                std::memcpy(out + pt.tag * isz, spec_.fill.data(), isz);
        };
        run(work, {}, &op);
    }

    // Keep the common 3-D loop bounds visible to the compiler, sharing the same
    // planner and bounds/rounding rules with the arbitrary-rank path.
    template <std::size_t N, class Coordinates>
    void read_points_as(const Coordinates& coords, std::size_t n, std::uint8_t* out)
    {
        using Coord = typename Coordinates::value_type;
        const std::size_t nd = N ? N : ndim();
        for (std::size_t b0 = 0; b0 < n; b0 += kPointBatch) {
            const std::size_t m = std::min(kPointBatch, n - b0);
            gather_points<N>(m, out, [&](std::size_t, std::size_t p, std::int64_t* idx, auto& emit) {
                Coord scratch[N ? N : kMaxSampleDims];
                const Coord* c = coords(b0 + p, scratch);
                const std::int64_t* at;
                if constexpr (std::is_floating_point_v<Coord>) {
                    for (std::size_t i = 0; i < nd; ++i) {
                        const auto base = coords.base(i);
                        if (!(c[i] > Coord(-1-base) && c[i] < Coord(spec_.shape[i]-base)+Coord(1))) {
                            idx[i] = -1;
                        } else if constexpr (std::is_same_v<Coord,float>) {
                            const float low = std::floor(c[i]), f = c[i]-low;
                            const auto integer = base+static_cast<std::int64_t>(low);
                            idx[i] = integer+(f > .5f || (f == .5f && (integer & 1)));
                        } else idx[i] = base+static_cast<std::int64_t>(std::nearbyint(c[i]));
                    }
                    at = idx;
                } else {
                    at = c;
                }
                bool inside = true;
                for (std::size_t i = 0; i < nd; ++i)
                    inside &= at[i] >= 0 && at[i] < spec_.shape[i];
                if (inside)
                    emit(at, b0 + p);
                else
                    std::memcpy(out + (b0 + p) * spec_.itemsize, spec_.fill.data(), spec_.itemsize);
            });
        }
    }

    // A point's grid cell: the corner below it, its fractional offsets, and the axes on
    // which it is off the grid.
    template <std::size_t N, class Real = double>
    struct Cell {
        using value_type = Real;
        std::int64_t lo[N ? N : kMaxSampleDims];
        Real frac[N ? N : kMaxSampleDims];
        std::size_t axes[N ? N : kMaxSampleDims];
        std::size_t k = 0;
    };

    template <std::size_t N, class Real, class Coordinates>
    static void cell_of(const Real* c, std::size_t nd, Cell<N,Real>& cell, const Coordinates& coords)
    {
        cell.k = 0;
        for (std::size_t i = 0; i < (N ? N : nd); ++i) {
            const Real f = std::floor(c[i]);
            cell.lo[i] = coords.base(i) + static_cast<std::int64_t>(f);
            cell.frac[i] = c[i] - f;
            if (cell.frac[i] != 0.0)
                cell.axes[cell.k++] = i;
        }
    }

    // sample_linear with element loads `load`. A point whose neighbours all lie in one chunk
    // is interpolated from that chunk while it is resolved. The other ("split") points read
    // their neighbours into `raw` (neighbour j of a split point at slot first + j of its
    // planning segment) and are interpolated after.
    template <std::size_t N, class Load, class Coordinates>
    void sample_linear_as(const Coordinates& coords, std::size_t n, const ElementType& type, bool out64, void* out,
                          Load load)
    {
        using Real = typename Coordinates::value_type;
        const std::size_t nd = N ? N : spec_.shape.size();
        const std::size_t isz = spec_.itemsize;
        const Real fill = static_cast<Real>(type.load(spec_.fill.data()));
        const auto store = [&](std::size_t p, Real v) {
            if (out64)
                static_cast<double*>(out)[p] = v;
            else
                static_cast<float*>(out)[p] = static_cast<float>(v);
        };
        struct Split {
            std::size_t p;
            std::uint64_t first;
        };
        struct Splits {
            std::vector<Split> points;
            std::uint64_t slots = 0;
        };
        std::int64_t reach = static_cast<std::int64_t>(isz);  // the far corner of a cell, from its near one
        for (std::size_t i = 0; i < nd; ++i)
            reach += chunk_strides_[i];
        std::vector<Splits> split;
        std::vector<std::uint64_t> base;  // segment -> its first slot in raw
        std::vector<std::uint8_t> raw;
        std::vector<Point> points;
        std::vector<Run> runs;
        // A point whose neighbours all lie in the chunk at `at`'s; `at` is the near corner.
        const auto value = [&](const Cell<N,Real>& cell, const std::uint8_t* at) {
            if (nd == 3 && cell.k == 3)
                return sampling::trilinear(cell.frac, at, chunk_strides_.data(), load);
            return sampling::interpolate(cell, [&](std::size_t corner) {
                std::int64_t offset = 0;
                for (std::size_t a = 0; a < cell.k; ++a)
                    if (corner >> a & 1)
                        offset += chunk_strides_[cell.axes[a]];
                return load(at + offset);
            });
        };
        for (std::size_t b0 = 0; b0 < n; b0 += kPointBatch) {
            const std::size_t m = std::min(kPointBatch, n - b0);
            split.assign(plan_segments(m), Splits{});
            auto work = plan_points<N>(m, points, runs, [&](std::size_t s, std::size_t p, std::int64_t* idx, auto& emit) {
                Real scratch[N ? N : kMaxSampleDims];
                const Real* c = coords(b0 + p, scratch);
                bool finite = true, outside = false;
                for (std::size_t i = 0; i < nd; ++i) {
                    finite &= std::isfinite(c[i]);
                    outside |= !(c[i] > Real(-1-coords.base(i)) && c[i] < Real(spec_.shape[i]-coords.base(i)));
                }
                if (!finite || outside) {  // no neighbour inside the array
                    store(b0 + p, finite ? fill : std::numeric_limits<Real>::quiet_NaN());
                    return;
                }
                Cell<N,Real> cell;
                cell_of<N>(c, nd, cell, coords);
                bool whole = true;  // every neighbour inside the array and in the near corner's chunk
                for (std::size_t i = 0; i < nd; ++i)
                    whole &= cell.lo[i] >= 0;
                for (std::size_t a = 0; a < cell.k; ++a) {
                    const std::size_t i = cell.axes[a];
                    const auto rem = chunk_shift_[i] >= 0
                                         ? cell.lo[i] & (spec_.chunks[i] - 1)
                                         : cell.lo[i] % spec_.chunks[i];
                    whole &= cell.lo[i] + 1 < spec_.shape[i] && rem + 1 < spec_.chunks[i];
                }
                if (whole) {
                    emit.point(cell.lo, b0 + p);
                    return;
                }
                Splits& sp = split[s];
                const std::uint64_t first = sp.slots;
                const std::size_t count = std::size_t{1} << cell.k;
                sp.points.push_back(Split{b0 + p, first});
                sp.slots += count;
                for (std::size_t corner = 0; corner < count; ++corner) {
                    bool inside = true;
                    for (std::size_t i = 0; i < nd; ++i)
                        idx[i] = cell.lo[i];
                    for (std::size_t a = 0; a < cell.k; ++a)
                        idx[cell.axes[a]] += corner >> a & 1;
                    for (std::size_t i = 0; i < nd; ++i)
                        inside &= idx[i] >= 0 && idx[i] < spec_.shape[i];
                    if (inside)  // outside neighbours keep raw's fill
                        emit(idx, kSplit | std::uint64_t{s} << kSegmentShift | (first + corner));
                }
            });
            base.assign(split.size(), 0);
            std::uint64_t slots = 0;
            for (std::size_t s = 0; s < split.size(); ++s) {
                base[s] = slots;
                slots += split[s].slots;
            }
            raw.resize(slots * isz);
            fill_elements(raw.data(), slots, spec_.fill);
            PointOp op;
            op.reach = reach;
            op.read = [&](const std::uint8_t* data, std::int64_t origin, const Work& w) {
                for_points(w.points, [&](std::span<const Point> part) {
                    Cell<N,Real> cell;
                    Real coordinate[N ? N : kMaxSampleDims];
                    for (std::size_t j = 0; j < part.size(); ++j) {
#if defined(__GNUC__) || defined(__clang__)
                        // Chunk grouping makes the voxel reads local, but random
                        // points still gather coordinates and four voxel rows.
                        if constexpr (N == 3) {
                            if (j + 8 < part.size()) {
                                const Point& next = part[j + 8];
                                if (!(next.tag & kSplit)) {
                                    coords.prefetch(next.tag);
                                    const auto* at = data + (next.src - origin);
                                    __builtin_prefetch(at);
                                    if (next.src + reach <= static_cast<std::int64_t>(chunk_bytes_)) {
                                        __builtin_prefetch(at + chunk_strides_[0]);
                                        __builtin_prefetch(at + chunk_strides_[1]);
                                        __builtin_prefetch(at + chunk_strides_[0] + chunk_strides_[1]);
                                    }
                                }
                            }
                        }
#endif
                        const Point& pt = part[j];
                        const std::uint8_t* at = data + (pt.src - origin);
                        if (pt.tag & kSplit) {
                            const std::uint64_t slot = base[(pt.tag & ~kSplit) >> kSegmentShift] + (pt.tag & kSlotMask);
                            std::memcpy(raw.data() + slot * isz, at, isz);
                            continue;
                        }
                        cell_of<N>(coords(pt.tag, coordinate), nd, cell, coords);
                        store(pt.tag, value(cell, at));
                    }
                });
                if (w.runs.empty())
                    return;
                const Index chunk = chunk_base(w);
                for_runs(w.runs, [&](std::span<const Run> part) {
                    Cell<N,Real> cell;
                    Real coordinate[N ? N : kMaxSampleDims];
                    for (const Run& r : part)
                        for (std::uint64_t p = r.first; p < r.first + r.count; ++p) {
                            cell_of<N>(coords(p, coordinate), nd, cell, coords);
                            store(p, value(cell, data + (offset_in<N>(cell.lo, chunk.data()) - origin)));
                        }
                });
            };
            op.fill = [&](const Work& w) {
                Cell<N,Real> cell;
                Real coordinate[N ? N : kMaxSampleDims];
                for (const Point& pt : w.points) {
                    if (pt.tag & kSplit)
                        continue;  // its raw slot holds fill already
                    cell_of<N>(coords(pt.tag, coordinate), nd, cell, coords);
                    store(pt.tag, sampling::interpolate(cell, [&](std::size_t) { return fill; }));
                }
                for (const Run& r : w.runs)
                    for (std::uint64_t p = r.first; p < r.first + r.count; ++p) {
                        cell_of<N>(coords(p, coordinate), nd, cell, coords);
                        store(p, sampling::interpolate(cell, [&](std::size_t) { return fill; }));
                    }
            };
            run(work, {}, &op);
            pool_.parallel_for(split.size(), [&](std::size_t s) {
                Cell<N,Real> cell;
                Real coordinate[N ? N : kMaxSampleDims];
                for (const Split& sp : split[s].points) {
                    const std::uint8_t* at = raw.data() + (base[s] + sp.first) * isz;
                    cell_of<N>(coords(sp.p, coordinate), nd, cell, coords);
                    store(sp.p, sampling::interpolate(cell, [&](std::size_t corner) { return load(at + corner * isz); }));
                }
            });
        }
    }

    enum class Found { Copied, Missing, NotCached };

    std::string key(const Index& idx) const
    {
        std::string k;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (i)
                k += spec_.separator;
            k += std::to_string(idx[i]);
        }
        return k;
    }

    std::string marker_path(const LayerSpec& l, const std::string& k) const
    {
        if (l.kind == "decoded")
            return l.root + "/" + k + ".missing";
        if (l.kind == "vc3d-mirror")
            return l.root + "/" + k + ".empty";
        if (l.kind == "vc3d-delta3d")
            return l.empty_root + "/" + k + ".empty";
        return {};
    }

    // Byte range of a piece inside the chunk file: first..last element touched.
    std::pair<std::int64_t, std::int64_t> span_of(const Piece& p) const
    {
        Index last(p.extent.size());
        for (std::size_t i = 0; i < last.size(); ++i)
            last[i] = p.src_lo[i] + p.extent[i] - 1;
        const std::int64_t first = offset_of(p.src_lo, chunk_strides_);
        return {first, offset_of(last, chunk_strides_) + static_cast<std::int64_t>(spec_.itemsize) - first};
    }

    std::pair<std::int64_t, std::int64_t> point_span(const Work& w, const PointOp& op) const
    {
        std::int64_t lo = INT64_MAX, hi = 0;
        for (const auto& pt : w.points) {
            lo = std::min(lo, pt.src);
            hi = std::max(hi, pt.src);
        }
        for (const auto& r : w.runs) {
            lo = std::min(lo, r.lo);
            hi = std::max(hi, r.hi);
        }
        hi = std::min(hi + op.reach, static_cast<std::int64_t>(chunk_bytes_));
        return {lo, hi - lo};
    }

    void hint(const Work& w)
    {
        if (spec_.layers.empty() || codecs_[0] != Codec::Raw || spec_.read_mode != "slab" || w.pieces.empty())
            return;
        File f = File::open_read(spec_.layers[0].root + "/" + key(w.idx));
        if (!f)
            return;
        std::int64_t lo = INT64_MAX, hi = 0;
        for (const auto& p : w.pieces) {
            auto [first, len] = span_of(p);
            lo = std::min(lo, first);
            hi = std::max(hi, first + len);
        }
        hint_willneed(f.fd(), lo, hi - lo);
    }

    void copy_piece(const std::uint8_t* src, const Piece& p, const std::vector<Index>& strides,
                    const BoxOutput* output)
    {
        if (output) {
            output->copy(src, chunk_strides_, p.dst, strides[p.region], p.extent);
        } else {
            copy_box(src, chunk_strides_, p.dst, strides[p.region], p.extent, spec_.itemsize);
        }
    }

    void apply(const ChunkPtr& chunk, const Work& w, const std::vector<Index>& out_strides, const PointOp* op, const BoxOutput* output)
    {
        if (!chunk) {
            fill_pieces(w, out_strides, op, output);
            return;
        }
        for (const auto& p : w.pieces)
            copy_piece(chunk->data() + offset_of(p.src_lo, chunk_strides_), p, out_strides, output);
        if (w.has_points())
            op->read(chunk->data(), 0, w);
    }

    void fill_pieces(const Work& w, const std::vector<Index>& out_strides, const PointOp* op, const BoxOutput* output)
    {
        for (const auto& p : w.pieces)
            fill_box(p.dst, out_strides[p.region], p.extent, output ? output->fill : spec_.fill);
        if (w.has_points())
            op->fill(w);
    }

    // Reads the pieces of a raw (uncompressed) chunk file straight into the outputs.
    Found read_raw_pieces(const std::string& path, const Work& w, const std::vector<Index>& out_strides,
                          const PointOp* op, const BoxOutput* output)
    {
        if (spec_.read_mode == "mmap") {
            bool fresh;
            auto m = maps_.get(path, fresh);
            if (!m)
                return Found::NotCached;
            if (m->size != chunk_bytes_)
                throw std::runtime_error(path + ": chunk file has the wrong size");
            const auto advise = [&](std::int64_t first, std::int64_t len) {
                const std::int64_t page = ::sysconf(_SC_PAGESIZE);
                const std::int64_t a = first / page * page;
                ::madvise(const_cast<std::uint8_t*>(m->data) + a, static_cast<std::size_t>(first + len - a),
                          MADV_WILLNEED);
            };
            // Many pieces of one chunk (sub-chunk blocks) share one hint over their
            // span unless it would advise much more than the pieces themselves.
            std::int64_t lo = INT64_MAX, hi = 0, total = 0;
            for (const auto& p : w.pieces) {
                auto [first, len] = span_of(p);
                lo = std::min(lo, first);
                hi = std::max(hi, first + len);
                total += len;
            }
            if (!w.pieces.empty() && hi - lo <= 2 * total)
                advise(lo, hi - lo);
            else
                for (const auto& p : w.pieces) {
                    auto [first, len] = span_of(p);
                    advise(first, len);
                }
            if (fresh && w.has_points()) {
                auto [first, len] = point_span(w, *op);
                advise(first, len);
            }
            for (const auto& p : w.pieces)
                copy_piece(m->data + offset_of(p.src_lo, chunk_strides_), p, out_strides, output);
            if (w.has_points())
                op->read(m->data, 0, w);
            return Found::Copied;
        }
        File f = File::open_read(path);
        if (!f)
            return Found::NotCached;
        thread_local std::vector<std::uint8_t> scratch;
        if (scratch.size() < chunk_bytes_)
            scratch.resize(chunk_bytes_);
        std::int64_t lo = INT64_MAX, hi = 0, total = 0;
        for (const auto& p : w.pieces) {
            auto [first, len] = span_of(p);
            lo = std::min(lo, first);
            hi = std::max(hi, first + len);
            total += len;
        }
        if (w.pieces.size() > 1 && hi - lo <= total) {  // overlapping spans (sub-chunk blocks): one read
            pread_exact(f.fd(), scratch.data(), static_cast<std::size_t>(hi - lo), lo, path);
            for (const auto& p : w.pieces)
                copy_piece(scratch.data() + (span_of(p).first - lo), p, out_strides, output);
        } else
            for (const auto& p : w.pieces) {
                auto [first, len] = span_of(p);
                pread_exact(f.fd(), scratch.data(), static_cast<std::size_t>(len), first, path);
                copy_piece(scratch.data(), p, out_strides, output);
            }
        if (w.has_points()) {  // one read of the span the points touch
            auto [lo, len] = point_span(w, *op);
            pread_exact(f.fd(), scratch.data(), static_cast<std::size_t>(len), lo, path);
            op->read(scratch.data(), lo, w);
        }
        return Found::Copied;
    }

    // A layer's whole chunk, decoded. nullopt: not in this layer.
    std::optional<ChunkPtr> read_layer_chunk(std::size_t li, const std::string& k)
    {
        const LayerSpec& l = spec_.layers[li];
        File f = File::open_read(l.root + "/" + k);
        if (!f) {
            if (l.kind == "array" || exists(marker_path(l, k)))
                return ChunkPtr{};
            return std::nullopt;
        }
        auto data = read_whole(f.fd(), l.root + "/" + k);
        if (codecs_[li] == Codec::Delta3d && !l.allow_lossy && !delta3d_lossless(data.data(), data.size()))
            return std::nullopt;  // quantized: not the source's values
        if (codecs_[li] == Codec::Raw) {
            check_raw_size(data.size(), chunk_bytes_);
            return std::make_shared<const Chunk>(std::move(data));
        }
        auto chunk = std::make_shared<Chunk>(chunk_bytes_);
        decode(codecs_[li], data.data(), data.size(), chunk->data(), chunk_bytes_);
        return ChunkPtr(std::move(chunk));
    }

    void resolve(const Work& w, const std::vector<Index>& out_strides, const PointOp* op, const BoxOutput* output,
                 bool stream = false)
    {
        const std::string k = key(w.idx);
        const std::string ck = spec_.identity + "|" + k;
        if (cache_) {
            auto [found, chunk] = cache_->get(ck);
            if (found) {
                ++stats_.memory_hits;
                apply(chunk, w, out_strides, op, output);
                return;
            }
        }
        if (writes_) {  // fetched, but its cache file is not written yet
            auto [found, chunk] = writes_->pending(k);
            if (found) {
                ++stats_.pending_hits;
                apply(chunk, w, out_strides, op, output);
                return;
            }
        }
        for (std::size_t li = 0; li < spec_.layers.size(); ++li) {
            const LayerSpec& l = spec_.layers[li];
            if ((!cache_ || stream) && codecs_[li] == Codec::Raw) {
                if (read_raw_pieces(l.root + "/" + k, w, out_strides, op, output) == Found::Copied) {
                    ++stats_.disk_hits;
                    return;
                }
                if (l.kind == "array" || exists(marker_path(l, k))) {
                    fill_pieces(w, out_strides, op, output);
                    return;
                }
                continue;
            }
            if (auto chunk = read_layer_chunk(li, k)) {
                ++stats_.disk_hits;
                if (cache_)
                    cache_->put(ck, *chunk);
                apply(*chunk, w, out_strides, op, output);
                return;
            }
        }
        if (auto fetcher = fetcher_.load()) {
            ChunkPtr chunk = (*fetcher)(w.idx).value;
            if (chunk && chunk->size() != chunk_bytes_)
                throw std::runtime_error("prefetch coordinator returned an invalid chunk size");
            if (cache_) cache_->put(ck, chunk);
            apply(chunk, w, out_strides, op, output);
            return;
        }
        if (spec_.offline)
            throw CacheMiss("chunk " + k + " of " + spec_.identity + " is not in the disk cache (offline)");
        if (!http_ && !source_) {
            fill_pieces(w, out_strides, op, output);
            return;
        }
        ChunkPtr chunk = fetch(k);
        if (cache_)
            cache_->put(ck, chunk);
        apply(chunk, w, out_strides, op, output);
    }

    // Fetches a chunk from the remote once, however many reads want it at the same time.
    ChunkPtr fetch(const std::string& k)
    {
        std::shared_future<ChunkPtr> future;
        std::promise<ChunkPtr> promise;
        bool owner = false;
        {
            std::lock_guard lock(inflight_mutex_);
            auto it = inflight_.find(k);
            if (it != inflight_.end()) {
                future = it->second;
            } else {
                future = promise.get_future().share();
                inflight_.emplace(k, future);
                owner = true;
            }
        }
        if (!owner)
            return future.get();
        try {
            ChunkPtr chunk = fetch_now(k);
            promise.set_value(chunk);
            std::lock_guard lock(inflight_mutex_);
            inflight_.erase(k);
            return chunk;
        } catch (...) {
            promise.set_exception(std::current_exception());
            std::lock_guard lock(inflight_mutex_);
            inflight_.erase(k);
            throw;
        }
    }

    ChunkPtr fetch_now(const std::string& k)
    {
        std::vector<std::uint8_t> body;
        bool found;
        if (source_) {
            found = source_->read(k, body);
            if (!http_ && found) ++stats_.disk_hits;
        } else {
            ++stats_.remote_fetches;
            found = http_->get(spec_.remote_url + "/" + k, body, chunk_bytes_);
            stats_.remote_bytes += body.size();
        }
        if (!found) {
            ++stats_.remote_missing;
            if (writes_ && !write_disabled_.load())
                writes_->submit(k, nullptr, 64, [this, k] { store_missing(k); });
            return nullptr;
        }
        ChunkPtr chunk;
        std::shared_ptr<const std::vector<std::uint8_t>> payload;
        if (remote_codec_ == Codec::Raw) {
            if (body.size() != chunk_bytes_)
                throw std::runtime_error(spec_.remote_url + "/" + k + ": body is not one chunk");
            chunk = std::make_shared<const Chunk>(std::move(body));
            payload = chunk;  // the stored bytes are the decoded chunk
        } else {
            auto decoded = std::make_shared<Chunk>(chunk_bytes_);
            decode(remote_codec_, body.data(), body.size(), decoded->data(), chunk_bytes_);
            chunk = std::move(decoded);
            payload = std::make_shared<const std::vector<std::uint8_t>>(std::move(body));
        }
        // The read returns now; the cache file is written (and encoded) in the background.
        if (writes_ && !write_disabled_.load()) {
            const std::size_t bytes = chunk->size() + (payload == chunk ? 0 : payload->size());
            writes_->submit(k, chunk, bytes, [this, k, chunk, payload] { store(k, *chunk, *payload); });
        }
        return chunk;
    }

    // Writes a fetched chunk to the write layer in that layer's format.
    void store(const std::string& k, const Chunk& decoded, const std::vector<std::uint8_t>& payload)
    {
        if (write_disabled_.load())
            return;
        const LayerSpec& l = spec_.layers[spec_.write_layer];
        try {
            const std::string path = l.root + "/" + k;
            if (l.kind == "vc3d-mirror") {
                atomic_write(path, payload.data(), payload.size());  // the source's bytes, verbatim
            } else if (codecs_[spec_.write_layer] == Codec::Delta3d) {  // compressed decoded / VC3D Delta3D
                const std::array<int, 3> shape{static_cast<int>(spec_.chunks[0]), static_cast<int>(spec_.chunks[1]),
                                               static_cast<int>(spec_.chunks[2])};
                auto enc = encode_delta3d(decoded.data(), decoded.size(), shape, spec_.itemsize);
                store_bytes(path, enc.data(), enc.size());
            } else {
                store_bytes(path, decoded.data(), decoded.size());
            }
        } catch (const std::exception& e) {
            write_failed(e.what());
        }
    }

    void store_bytes(const std::string& path, const std::uint8_t* data, std::size_t size)
    {
        if (spec_.disk_cache)
            spec_.disk_cache->write(path, data, size);
        else
            atomic_write(path, data, size);
    }

    void store_missing(const std::string& k)
    {
        if (write_disabled_.load())
            return;
        const std::string path = marker_path(spec_.layers[spec_.write_layer], k);
        if (path.empty())
            return;
        try {
            atomic_write(path, nullptr, 0);
        } catch (const std::exception& e) {
            write_failed(e.what());
        }
    }

    void write_failed(const char* what)
    {
        ++stats_.write_errors;
        write_disabled_ = true;
        std::lock_guard lock(error_mutex_);
        last_write_error_ = what;
    }

    ArraySpec spec_;
    std::shared_ptr<MemoryCache> cache_;
    ThreadPool pool_;
    MapCache maps_;
    std::unique_ptr<HttpClient> http_;
    std::vector<Codec> codecs_;
    Codec remote_codec_ = Codec::Raw;
    Index chunk_strides_, grid_;
    std::vector<int> chunk_shift_;
    std::size_t chunk_bytes_ = 0;
    Stats stats_;
    std::unique_ptr<ZarrStorage> source_;
    std::mutex inflight_mutex_;
    using Fetcher = std::function<ChunkHandle(const Index&)>;
    std::atomic<std::shared_ptr<const Fetcher>> fetcher_;
    std::unordered_map<std::string, std::shared_future<ChunkPtr>> inflight_;
    std::atomic<bool> write_disabled_{false};
    std::mutex error_mutex_;
    std::string last_write_error_;
    // Last member: destroyed first, so queued writes finish while everything they use still exists.
    std::unique_ptr<WriteBehind> writes_;
};

}  // namespace vcr
