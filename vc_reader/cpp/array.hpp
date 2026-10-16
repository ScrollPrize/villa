// Native read path for one zarr v2 array: plan, then resolve every chunk through
// the memory cache, the disk layers (in their own formats) and the remote, and
// copy into the caller's output buffers.
#pragma once

#include "chunk_io.hpp"
#include "codecs.hpp"
#include "http.hpp"
#include "memory_cache.hpp"
#include "thread_pool.hpp"
#include "write_behind.hpp"

#include <sys/mman.h>

#include <algorithm>
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
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vcr {

// How sample() reads stored elements as numbers.
struct ElementType {
    char kind = 'u';  // numpy kind: 'u', 'i', 'f' or 'b'
    std::size_t itemsize = 1;
    bool swap = false;  // stored in the other byte order

    double load(const std::uint8_t* p) const
    {
        std::uint8_t b[8];
        std::memcpy(b, p, itemsize);
        if (swap)
            std::reverse(b, b + itemsize);
        switch (kind) {
        case 'b':
            return b[0] != 0;
        case 'f':
            if (itemsize == 4) {
                float v;
                std::memcpy(&v, b, 4);
                return v;
            } else {
                double v;
                std::memcpy(&v, b, 8);
                return v;
            }
        case 'i':
            switch (itemsize) {
            case 1: return static_cast<double>(static_cast<std::int8_t>(b[0]));
            case 2: { std::int16_t v; std::memcpy(&v, b, 2); return v; }
            case 4: { std::int32_t v; std::memcpy(&v, b, 4); return v; }
            default: { std::int64_t v; std::memcpy(&v, b, 8); return static_cast<double>(v); }
            }
        default:
            switch (itemsize) {
            case 1: return b[0];
            case 2: { std::uint16_t v; std::memcpy(&v, b, 2); return v; }
            case 4: { std::uint32_t v; std::memcpy(&v, b, 4); return v; }
            default: { std::uint64_t v; std::memcpy(&v, b, 8); return static_cast<double>(v); }
            }
        }
    }
};

// A chunk an offline read needs is in none of the disk layers.
struct CacheMiss : std::runtime_error {
    using std::runtime_error::runtime_error;
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
    bool offline = false;            // never fetch: a chunk in no layer raises CacheMiss
    std::string read_mode = "slab";  // raw local chunks: "slab" (pread rows) or "mmap"
    bool hint = false;               // queue readahead for every piece before reading
    std::size_t threads = 16;
    std::size_t writer_threads = 4;                    // background disk-cache writes
    std::size_t max_pending_write_bytes = 1ull << 30;  // reads wait beyond this much unwritten data
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
        for (std::size_t i = 0; i < nd; ++i)
            grid_[i] = (spec_.shape[i] + spec_.chunks[i] - 1) / spec_.chunks[i];
        for (auto& l : spec_.layers)
            codecs_.push_back(parse_codec(l.codec));
        remote_codec_ = parse_codec(spec_.remote_codec);
        if (!spec_.remote_url.empty() && !spec_.offline)
            http_ = std::make_unique<HttpClient>(spec_.http);
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

    // Credentials for later fetches (e.g. refreshed temporary keys).
    void set_auth(Auth auth)
    {
        if (http_)
            http_->set_auth(std::move(auth));
    }

    // Reads every region [start, start+size) into its C-order output buffer.
    // Parts outside the array, and chunks that are not stored, read as fill.
    void read_many(const std::vector<Index>& starts, const std::vector<Index>& sizes,
                   const std::vector<std::uint8_t*>& outs)
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
            out_strides[r] = c_strides(size, spec_.itemsize);
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
                fill_box(outs[r], out_strides[r], size, spec_.fill);
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
        run(ordered, out_strides, nullptr);
    }

    // Reads the elements at `n` points into `out` (one element each, in order).
    // `coords` holds each point's indices, C order. Points outside the array read as fill.
    void read_points(const std::int64_t* coords, std::size_t n, std::uint8_t* out)
    {
        const std::size_t nd = spec_.shape.size();
        for (std::size_t b0 = 0; b0 < n; b0 += kPointBatch) {
            const std::size_t m = std::min(kPointBatch, n - b0);
            gather_points(m, out, [&](std::size_t, std::size_t p, std::int64_t*, auto& emit) {
                const std::int64_t* c = coords + (b0 + p) * nd;
                bool inside = true;
                for (std::size_t i = 0; i < nd; ++i)
                    inside &= c[i] >= 0 && c[i] < spec_.shape[i];
                if (inside)
                    emit(c, b0 + p);
                else
                    std::memcpy(out + (b0 + p) * spec_.itemsize, spec_.fill.data(), spec_.itemsize);
            });
        }
    }

    // Element nearest to each real-valued point (voxel i at coordinate i, ties to even),
    // written as stored. Points outside the array, and non-finite ones, read as fill.
    void sample_nearest(const double* coords, std::size_t n, std::uint8_t* out)
    {
        const std::size_t nd = spec_.shape.size();
        for (std::size_t b0 = 0; b0 < n; b0 += kPointBatch) {
            const std::size_t m = std::min(kPointBatch, n - b0);
            gather_points(m, out, [&](std::size_t, std::size_t p, std::int64_t* idx, auto& emit) {
                const double* c = coords + (b0 + p) * nd;
                bool inside = true;
                for (std::size_t i = 0; i < nd; ++i) {
                    // Out of range (or NaN): any index outside the array reads as fill.
                    idx[i] = c[i] > -1.0 && c[i] < static_cast<double>(spec_.shape[i]) + 1.0
                                 ? static_cast<std::int64_t>(std::nearbyint(c[i])) : -1;
                    inside &= idx[i] >= 0 && idx[i] < spec_.shape[i];
                }
                if (inside)
                    emit(idx, b0 + p);
                else
                    std::memcpy(out + (b0 + p) * spec_.itemsize, spec_.fill.data(), spec_.itemsize);
            });
        }
    }

    // N-linear interpolation at real-valued points (voxel i at coordinate i): the weighted
    // sum of the 2^k neighbouring elements, k being the axes on which the point is not
    // on the grid. Neighbours outside the array read as fill. Non-finite points give NaN.
    // Computed in double; `out` holds float (out64 false) or double values.
    void sample_linear(const double* coords, std::size_t n, const ElementType& type, bool out64, void* out)
    {
        if (spec_.shape.size() > kMaxSampleDims)
            throw std::invalid_argument("sample() supports at most 16 dimensions");
        // The element loads, chosen once: native-order numbers inline, the rest through `type`.
        const auto with = [&]<class T>() {
            sample_linear_as(coords, n, type, out64, out, [](const std::uint8_t* p) {
                T v;
                std::memcpy(&v, p, sizeof(T));
                return static_cast<double>(v);
            });
        };
        if (type.swap || type.kind == 'b')
            return sample_linear_as(coords, n, type, out64, out, [&type](const std::uint8_t* p) { return type.load(p); });
        switch (type.kind == 'f' ? 100 + type.itemsize : type.kind == 'i' ? 10 + type.itemsize : type.itemsize) {
        case 1: return with.template operator()<std::uint8_t>();
        case 2: return with.template operator()<std::uint16_t>();
        case 4: return with.template operator()<std::uint32_t>();
        case 8: return with.template operator()<std::uint64_t>();
        case 11: return with.template operator()<std::int8_t>();
        case 12: return with.template operator()<std::int16_t>();
        case 14: return with.template operator()<std::int32_t>();
        case 18: return with.template operator()<std::int64_t>();
        case 104: return with.template operator()<float>();
        case 108: return with.template operator()<double>();
        default:
            return sample_linear_as(coords, n, type, out64, out, [&type](const std::uint8_t* p) { return type.load(p); });
        }
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
    struct Point {
        std::int64_t src = 0;   // byte offset in the chunk
        std::uint64_t tag = 0;  // which output the element is for: the PointOp's business
    };
    // What a point read does with a chunk's points once the chunk is resolved.
    struct PointOp {
        std::int64_t reach = 0;  // bytes each point reads from its offset on
        // `data` holds the chunk's bytes from byte offset `origin` on.
        std::function<void(const std::uint8_t* data, std::int64_t origin, std::span<const Point>)> read;
        std::function<void(std::span<const Point>)> fill;  // the chunk is not stored
    };
    // Everything one chunk supplies: boxes (read_many) or single elements (point reads).
    struct Work {
        Index idx;
        std::vector<Piece> pieces;
        std::span<const Point> points;
    };

    // Resolves every chunk in parallel, in the order given.
    void run(std::vector<Work>& work, const std::vector<Index>& out_strides, const PointOp* op)
    {
        if (spec_.hint && !cache_)
            pool_.parallel_for(work.size(), [&](std::size_t i) { hint(work[i]); });
        pool_.parallel_for(work.size(), [&](std::size_t i) { resolve(work[i], out_strides, op); });
    }

    std::size_t plan_segments(std::size_t n) const
    {
        return std::clamp<std::size_t>(n / kPlanGrain, 1, pool_.size() + 1);
    }

    // Groups point reads by chunk. The n points are split into plan_segments(n) contiguous
    // ranges, each planned in order on one thread: locate(segment, p, scratch, emit) calls
    // emit(index, tag) for each element point p reads (index: nd indices inside the array;
    // scratch: room for nd indices). Returns one Work per chunk, in chunk order, whose
    // points (byte offset and tag, in emission order) are stored in `points`.
    template <class Locate>
    std::vector<Work> plan_points(std::size_t n, std::vector<Point>& points, Locate&& locate)
    {
        const std::size_t nd = spec_.shape.size();
        const std::size_t segs = plan_segments(n);
        struct Entry {
            std::int64_t src;
            std::uint64_t tag;
            std::uint32_t id;
        };
        struct Segment {
            ChunkIds ids;
            std::vector<std::int64_t> linear;  // local id -> linear chunk index
            std::vector<std::size_t> next;     // local id -> element count; later its next slot in `points`
            std::vector<Entry> entries;
        };
        std::vector<Segment> seg(segs);
        pool_.parallel_for(segs, [&](std::size_t s) {
            Segment& g = seg[s];
            const std::size_t p0 = n * s / segs, p1 = n * (s + 1) / segs;
            g.entries.reserve(p1 - p0);
            Index scratch(nd);
            std::int64_t last = -1;
            std::uint32_t id = 0;
            auto emit = [&](const std::int64_t* c, std::uint64_t tag) {
                std::int64_t linear = 0, offset = 0;
                for (std::size_t i = 0; i < nd; ++i) {
                    const std::int64_t q = c[i] / spec_.chunks[i];
                    linear = linear * grid_[i] + q;
                    offset += (c[i] - q * spec_.chunks[i]) * chunk_strides_[i];
                }
                if (linear != last) {  // runs of points in one chunk skip the lookup
                    last = linear;
                    id = g.ids.find_or_insert(linear, static_cast<std::uint32_t>(g.linear.size()));
                    if (id == g.linear.size()) {
                        g.linear.push_back(linear);
                        g.next.push_back(0);
                    }
                }
                ++g.next[id];
                g.entries.push_back(Entry{offset, tag, id});
            };
            for (std::size_t p = p0; p < p1; ++p)
                locate(s, p, scratch.data(), emit);
        });
        // Chunks in order; within a chunk, each segment's elements after the earlier segments'.
        std::vector<std::int64_t> chunks;
        for (const auto& g : seg)
            chunks.insert(chunks.end(), g.linear.begin(), g.linear.end());
        std::sort(chunks.begin(), chunks.end());
        chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());
        std::vector<std::vector<std::size_t>> global(segs);
        std::vector<std::size_t> start(chunks.size() + 1, 0);
        for (std::size_t s = 0; s < segs; ++s) {
            const Segment& g = seg[s];
            global[s].resize(g.linear.size());
            for (std::size_t id = 0; id < g.linear.size(); ++id) {
                global[s][id] = static_cast<std::size_t>(
                    std::lower_bound(chunks.begin(), chunks.end(), g.linear[id]) - chunks.begin());
                start[global[s][id] + 1] += g.next[id];
            }
        }
        for (std::size_t c = 0; c < chunks.size(); ++c)
            start[c + 1] += start[c];
        std::vector<std::size_t> cursor(start.begin(), start.end() - 1);
        for (std::size_t s = 0; s < segs; ++s) {
            Segment& g = seg[s];
            for (std::size_t id = 0; id < g.linear.size(); ++id) {
                const std::size_t count = g.next[id];
                g.next[id] = cursor[global[s][id]];
                cursor[global[s][id]] += count;
            }
        }
        points.resize(start.back());
        pool_.parallel_for(segs, [&](std::size_t s) {
            Segment& g = seg[s];
            for (const Entry& e : g.entries)
                points[g.next[e.id]++] = Point{e.src, e.tag};
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

    template <std::size_t K>
    static void copy_points(const std::uint8_t* data, std::int64_t origin, std::span<const Point> pts,
                            std::uint8_t* out)
    {
        for (const Point& pt : pts)
            std::memcpy(out + pt.tag * K, data + (pt.src - origin), K);
    }

    // Reads one element per located point into out[tag] (see plan_points for `locate`).
    template <class Locate>
    void gather_points(std::size_t n, std::uint8_t* out, Locate&& locate)
    {
        std::vector<Point> points;
        auto work = plan_points(n, points, locate);
        const std::size_t isz = spec_.itemsize;
        PointOp op;
        op.reach = static_cast<std::int64_t>(isz);
        op.read = [&](const std::uint8_t* data, std::int64_t origin, std::span<const Point> pts) {
            for_points(pts, [&](std::span<const Point> part) {
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
        op.fill = [&](std::span<const Point> pts) {
            for (const Point& pt : pts)
                std::memcpy(out + pt.tag * isz, spec_.fill.data(), isz);
        };
        run(work, {}, &op);
    }

    // A point's grid cell: the corner below it, its fractional offsets, and the axes on
    // which it is off the grid.
    struct Cell {
        std::int64_t lo[kMaxSampleDims];
        double frac[kMaxSampleDims];
        std::size_t axes[kMaxSampleDims];
        std::size_t k = 0;
    };

    static void cell_of(const double* c, std::size_t nd, Cell& cell)
    {
        cell.k = 0;
        for (std::size_t i = 0; i < nd; ++i) {
            const double f = std::floor(c[i]);
            cell.lo[i] = static_cast<std::int64_t>(f);
            cell.frac[i] = c[i] - f;
            if (cell.frac[i] != 0.0)
                cell.axes[cell.k++] = i;
        }
    }

    // The weighted sum of a cell's 2^k neighbours; value(corner) is neighbour `corner`
    // (bit a set: one step up along axes[a]). The weights and the sum are always formed in
    // this order, so a point's result does not depend on where its neighbours were read.
    template <class Value>
    static double interpolate(const Cell& cell, Value&& value)
    {
        const std::size_t count = std::size_t{1} << cell.k;
        double v = 0.0;
        for (std::size_t corner = 0; corner < count; ++corner) {
            double w = 1.0;
            for (std::size_t a = 0; a < cell.k; ++a) {
                const double f = cell.frac[cell.axes[a]];
                if (corner >> a & 1)
                    w *= f;
                else
                    w *= 1.0 - f;
            }
            v += w * value(corner);
        }
        return v;
    }

    // interpolate() for a 3-D cell off the grid on every axis, unrolled: the same weights
    // and the same sum, in the same order. `at` is the near corner, `step` the byte strides.
    template <class Load>
    static double trilinear(const double* f, const std::uint8_t* at, const std::int64_t* step, Load& load)
    {
        const double g0 = 1.0 - f[0], g1 = 1.0 - f[1], g2 = 1.0 - f[2];
        const double w00 = g0 * g1, w10 = f[0] * g1, w01 = g0 * f[1], w11 = f[0] * f[1];
        const std::int64_t s0 = step[0], s1 = step[1], s2 = step[2];
        double v = 0.0;
        v += w00 * g2 * load(at);
        v += w10 * g2 * load(at + s0);
        v += w01 * g2 * load(at + s1);
        v += w11 * g2 * load(at + s0 + s1);
        v += w00 * f[2] * load(at + s2);
        v += w10 * f[2] * load(at + s0 + s2);
        v += w01 * f[2] * load(at + s1 + s2);
        v += w11 * f[2] * load(at + s0 + s1 + s2);
        return v;
    }

    // sample_linear with element loads `load`. A point whose neighbours all lie in one chunk
    // is interpolated from that chunk while it is resolved. The other ("split") points read
    // their neighbours into `raw` (neighbour j of a split point at slot first + j of its
    // planning segment) and are interpolated after.
    template <class Load>
    void sample_linear_as(const double* coords, std::size_t n, const ElementType& type, bool out64, void* out,
                          Load load)
    {
        const std::size_t nd = spec_.shape.size();
        const std::size_t isz = spec_.itemsize;
        const double fill = type.load(spec_.fill.data());
        const auto store = [&](std::size_t p, double v) {
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
        for (std::size_t b0 = 0; b0 < n; b0 += kPointBatch) {
            const std::size_t m = std::min(kPointBatch, n - b0);
            split.assign(plan_segments(m), Splits{});
            auto work = plan_points(m, points, [&](std::size_t s, std::size_t p, std::int64_t* idx, auto& emit) {
                const double* c = coords + (b0 + p) * nd;
                bool finite = true, outside = false;
                for (std::size_t i = 0; i < nd; ++i) {
                    finite &= std::isfinite(c[i]);
                    outside |= !(c[i] > -1.0 && c[i] < static_cast<double>(spec_.shape[i]));
                }
                if (!finite || outside) {  // no neighbour inside the array
                    store(b0 + p, finite ? fill : std::numeric_limits<double>::quiet_NaN());
                    return;
                }
                Cell cell;
                cell_of(c, nd, cell);
                bool whole = true;  // every neighbour inside the array and in the near corner's chunk
                for (std::size_t i = 0; i < nd; ++i)
                    whole &= cell.lo[i] >= 0;
                for (std::size_t a = 0; a < cell.k; ++a) {
                    const std::size_t i = cell.axes[a];
                    whole &= cell.lo[i] + 1 < spec_.shape[i] && cell.lo[i] % spec_.chunks[i] + 1 < spec_.chunks[i];
                }
                if (whole) {
                    emit(cell.lo, b0 + p);
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
            for (std::uint64_t j = 0; j < slots; ++j)
                std::memcpy(raw.data() + j * isz, spec_.fill.data(), isz);
            PointOp op;
            op.reach = reach;
            op.read = [&](const std::uint8_t* data, std::int64_t origin, std::span<const Point> pts) {
                for_points(pts, [&](std::span<const Point> part) {
                    Cell cell;
                    for (const Point& pt : part) {
                        const std::uint8_t* at = data + (pt.src - origin);
                        if (pt.tag & kSplit) {
                            const std::uint64_t slot = base[(pt.tag & ~kSplit) >> kSegmentShift] + (pt.tag & kSlotMask);
                            std::memcpy(raw.data() + slot * isz, at, isz);
                            continue;
                        }
                        cell_of(coords + pt.tag * nd, nd, cell);
                        if (nd == 3 && cell.k == 3) {
                            store(pt.tag, trilinear(cell.frac, at, chunk_strides_.data(), load));
                            continue;
                        }
                        store(pt.tag, interpolate(cell, [&](std::size_t corner) {
                            std::int64_t offset = 0;
                            for (std::size_t a = 0; a < cell.k; ++a)
                                if (corner >> a & 1)
                                    offset += chunk_strides_[cell.axes[a]];
                            return load(at + offset);
                        }));
                    }
                });
            };
            op.fill = [&](std::span<const Point> pts) {
                Cell cell;
                for (const Point& pt : pts) {
                    if (pt.tag & kSplit)
                        continue;  // its raw slot holds fill already
                    cell_of(coords + pt.tag * nd, nd, cell);
                    store(pt.tag, interpolate(cell, [&](std::size_t) { return fill; }));
                }
            };
            run(work, {}, &op);
            pool_.parallel_for(split.size(), [&](std::size_t s) {
                Cell cell;
                for (const Split& sp : split[s].points) {
                    const std::uint8_t* at = raw.data() + (base[s] + sp.first) * isz;
                    cell_of(coords + sp.p * nd, nd, cell);
                    store(sp.p, interpolate(cell, [&](std::size_t corner) { return load(at + corner * isz); }));
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

    void apply(const ChunkPtr& chunk, const Work& w, const std::vector<Index>& out_strides, const PointOp* op)
    {
        if (!chunk) {
            fill_pieces(w, out_strides, op);
            return;
        }
        for (const auto& p : w.pieces)
            copy_box(chunk->data() + offset_of(p.src_lo, chunk_strides_), chunk_strides_, p.dst,
                     out_strides[p.region], p.extent, spec_.itemsize);
        if (!w.points.empty())
            op->read(chunk->data(), 0, w.points);
    }

    void fill_pieces(const Work& w, const std::vector<Index>& out_strides, const PointOp* op)
    {
        for (const auto& p : w.pieces)
            fill_box(p.dst, out_strides[p.region], p.extent, spec_.fill);
        if (!w.points.empty())
            op->fill(w.points);
    }

    // Reads the pieces of a raw (uncompressed) chunk file straight into the outputs.
    Found read_raw_pieces(const std::string& path, const Work& w, const std::vector<Index>& out_strides,
                          const PointOp* op)
    {
        if (spec_.read_mode == "mmap") {
            auto m = maps_.get(path);
            if (!m)
                return Found::NotCached;
            if (m->size != chunk_bytes_)
                throw std::runtime_error(path + ": chunk file has the wrong size");
            for (const auto& p : w.pieces) {
                auto [first, len] = span_of(p);
                const std::int64_t page = ::sysconf(_SC_PAGESIZE);
                const std::int64_t a = first / page * page;
                ::madvise(const_cast<std::uint8_t*>(m->data) + a, static_cast<std::size_t>(first + len - a),
                          MADV_WILLNEED);
            }
            for (const auto& p : w.pieces)
                copy_box(m->data + offset_of(p.src_lo, chunk_strides_), chunk_strides_, p.dst,
                         out_strides[p.region], p.extent, spec_.itemsize);
            if (!w.points.empty())
                op->read(m->data, 0, w.points);
            return Found::Copied;
        }
        File f = File::open_read(path);
        if (!f)
            return Found::NotCached;
        thread_local std::vector<std::uint8_t> scratch;
        if (scratch.size() < chunk_bytes_)
            scratch.resize(chunk_bytes_);
        for (const auto& p : w.pieces) {
            auto [first, len] = span_of(p);
            pread_exact(f.fd(), scratch.data(), static_cast<std::size_t>(len), first, path);
            copy_box(scratch.data(), chunk_strides_, p.dst, out_strides[p.region], p.extent, spec_.itemsize);
        }
        if (!w.points.empty()) {  // one read of the span the points touch
            std::int64_t lo = INT64_MAX, hi = 0;
            for (const auto& pt : w.points) {
                lo = std::min(lo, pt.src);
                hi = std::max(hi, pt.src);
            }
            hi = std::min(hi + op->reach, static_cast<std::int64_t>(chunk_bytes_));
            pread_exact(f.fd(), scratch.data(), static_cast<std::size_t>(hi - lo), lo, path);
            op->read(scratch.data(), lo, w.points);
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
        auto chunk = std::make_shared<Chunk>(chunk_bytes_);
        decode(codecs_[li], data.data(), data.size(), chunk->data(), chunk_bytes_);
        return ChunkPtr(std::move(chunk));
    }

    void resolve(const Work& w, const std::vector<Index>& out_strides, const PointOp* op)
    {
        const std::string k = key(w.idx);
        const std::string ck = spec_.identity + "|" + k;
        if (cache_) {
            auto [found, chunk] = cache_->get(ck);
            if (found) {
                ++stats_.memory_hits;
                apply(chunk, w, out_strides, op);
                return;
            }
        }
        if (writes_) {  // fetched, but its cache file is not written yet
            auto [found, chunk] = writes_->pending(k);
            if (found) {
                ++stats_.pending_hits;
                apply(chunk, w, out_strides, op);
                return;
            }
        }
        for (std::size_t li = 0; li < spec_.layers.size(); ++li) {
            const LayerSpec& l = spec_.layers[li];
            if (!cache_ && codecs_[li] == Codec::Raw) {
                if (read_raw_pieces(l.root + "/" + k, w, out_strides, op) == Found::Copied) {
                    ++stats_.disk_hits;
                    return;
                }
                if (l.kind == "array" || exists(marker_path(l, k))) {
                    fill_pieces(w, out_strides, op);
                    return;
                }
                continue;
            }
            if (auto chunk = read_layer_chunk(li, k)) {
                ++stats_.disk_hits;
                if (cache_)
                    cache_->put(ck, *chunk);
                apply(*chunk, w, out_strides, op);
                return;
            }
        }
        if (spec_.offline)
            throw CacheMiss("chunk " + k + " of " + spec_.identity + " is not in the disk cache (offline)");
        if (!http_) {
            fill_pieces(w, out_strides, op);
            return;
        }
        ChunkPtr chunk = fetch(k);
        if (cache_)
            cache_->put(ck, chunk);
        apply(chunk, w, out_strides, op);
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
        ++stats_.remote_fetches;
        const bool found = http_->get(spec_.remote_url + "/" + k, body, chunk_bytes_);
        if (!found) {
            ++stats_.remote_missing;
            if (writes_ && !write_disabled_.load())
                writes_->submit(k, nullptr, 64, [this, k] { store_missing(k); });
            return nullptr;
        }
        stats_.remote_bytes += body.size();
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
                atomic_write(path, enc.data(), enc.size());
            } else {
                atomic_write(path, decoded.data(), decoded.size());
            }
        } catch (const std::exception& e) {
            write_failed(e.what());
        }
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
    std::size_t chunk_bytes_ = 0;
    Stats stats_;
    std::mutex inflight_mutex_;
    std::unordered_map<std::string, std::shared_future<ChunkPtr>> inflight_;
    std::atomic<bool> write_disabled_{false};
    std::mutex error_mutex_;
    std::string last_write_error_;
    // Last member: destroyed first, so queued writes finish while everything they use still exists.
    std::unique_ptr<WriteBehind> writes_;
};

}  // namespace vcr
