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
#include <cmath>
#include <limits>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
        run(work, slot, out_strides);
    }

    // Reads the elements at `n` points into `out` (one element each, in order).
    // `coords` holds each point's indices, C order. Points outside the array read as fill.
    void read_points(const std::int64_t* coords, std::size_t n, std::uint8_t* out)
    {
        const std::size_t nd = spec_.shape.size();
        std::map<std::int64_t, std::size_t> slot;
        std::vector<Work> work;
        Index idx(nd);
        for (std::size_t p = 0; p < n; ++p) {
            const std::int64_t* c = coords + p * nd;
            std::uint8_t* dst = out + p * spec_.itemsize;
            bool inside = true;
            for (std::size_t i = 0; i < nd; ++i)
                inside &= c[i] >= 0 && c[i] < spec_.shape[i];
            if (!inside) {
                std::memcpy(dst, spec_.fill.data(), spec_.itemsize);
                continue;
            }
            std::int64_t linear = 0, offset = 0;
            for (std::size_t i = 0; i < nd; ++i) {
                idx[i] = c[i] / spec_.chunks[i];
                linear = linear * grid_[i] + idx[i];
                offset += (c[i] - idx[i] * spec_.chunks[i]) * chunk_strides_[i];
            }
            auto [it, fresh] = slot.emplace(linear, work.size());
            if (fresh)
                work.push_back(Work{idx, {}, {}});
            work[it->second].points.push_back(Point{offset, dst});
        }
        run(work, slot, {});
    }

    // Element nearest to each real-valued point (voxel i at coordinate i, ties to even),
    // written as stored. Points outside the array, and non-finite ones, read as fill.
    void sample_nearest(const double* coords, std::size_t n, std::uint8_t* out)
    {
        const std::size_t nd = spec_.shape.size();
        std::vector<std::int64_t> idx;
        for (std::size_t b0 = 0; b0 < n; b0 += kSampleBatch) {
            const std::size_t m = std::min(kSampleBatch, n - b0);
            idx.resize(m * nd);
            for (std::size_t p = 0; p < m; ++p)
                for (std::size_t i = 0; i < nd; ++i) {
                    const double c = coords[(b0 + p) * nd + i];
                    // Out of range (or NaN): any index outside the array reads as fill.
                    idx[p * nd + i] = c > -1.0 && c < static_cast<double>(spec_.shape[i]) + 1.0
                                          ? static_cast<std::int64_t>(std::nearbyint(c)) : -1;
                }
            read_points(idx.data(), m, out + b0 * spec_.itemsize);
        }
    }

    // N-linear interpolation at real-valued points (voxel i at coordinate i): the weighted
    // sum of the 2^k neighbouring elements, k being the axes on which the point is not
    // on the grid. Neighbours outside the array read as fill. Non-finite points give NaN.
    // Computed in double; `out` holds float (out64 false) or double values.
    void sample_linear(const double* coords, std::size_t n, const ElementType& type, bool out64, void* out)
    {
        const std::size_t nd = spec_.shape.size();
        if (nd > 16)
            throw std::invalid_argument("sample() supports at most 16 dimensions");
        const double fill = type.load(spec_.fill.data());
        std::vector<std::int64_t> corners;  // neighbour indices, nd per neighbour
        std::vector<double> weights;        // one per neighbour
        std::vector<std::uint32_t> first;   // each point's first neighbour; the end after the last
        std::vector<double> value;          // each point's value when it needs no neighbours
        std::vector<std::uint8_t> raw;
        std::vector<std::int64_t> lo(nd);
        std::vector<double> frac(nd);
        std::vector<std::size_t> axes;
        for (std::size_t b0 = 0; b0 < n; b0 += kSampleBatch) {
            const std::size_t m = std::min(kSampleBatch, n - b0);
            corners.clear();
            weights.clear();
            first.assign(m + 1, 0);
            value.assign(m, 0.0);
            for (std::size_t p = 0; p < m; ++p) {
                first[p] = static_cast<std::uint32_t>(weights.size());
                const double* c = coords + (b0 + p) * nd;
                bool finite = true, outside = false;
                axes.clear();
                for (std::size_t i = 0; i < nd; ++i) {
                    finite &= std::isfinite(c[i]);
                    outside |= !(c[i] > -1.0 && c[i] < static_cast<double>(spec_.shape[i]));
                }
                if (!finite || outside) {  // no neighbour inside the array
                    value[p] = finite ? fill : std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                for (std::size_t i = 0; i < nd; ++i) {
                    const double f = std::floor(c[i]);
                    lo[i] = static_cast<std::int64_t>(f);
                    frac[i] = c[i] - f;
                    if (frac[i] != 0.0)
                        axes.push_back(i);
                }
                const std::size_t count = std::size_t{1} << axes.size();
                for (std::size_t corner = 0; corner < count; ++corner) {
                    double w = 1.0;
                    const std::size_t at = corners.size();
                    corners.insert(corners.end(), lo.begin(), lo.end());
                    for (std::size_t a = 0; a < axes.size(); ++a) {
                        const std::size_t i = axes[a];
                        if (corner >> a & 1) {
                            ++corners[at + i];
                            w *= frac[i];
                        } else {
                            w *= 1.0 - frac[i];
                        }
                    }
                    weights.push_back(w);
                }
            }
            first[m] = static_cast<std::uint32_t>(weights.size());
            raw.resize(weights.size() * spec_.itemsize);
            read_points(corners.data(), weights.size(), raw.data());
            for (std::size_t p = 0; p < m; ++p) {
                double v = value[p];
                if (first[p] != first[p + 1]) {
                    v = 0.0;
                    for (std::uint32_t j = first[p]; j < first[p + 1]; ++j)
                        v += weights[j] * type.load(raw.data() + j * spec_.itemsize);
                }
                if (out64)
                    static_cast<double*>(out)[b0 + p] = v;
                else
                    static_cast<float*>(out)[b0 + p] = static_cast<float>(v);
            }
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

    static constexpr std::size_t kSampleBatch = 1 << 16;  // points per batch: bounds the neighbour lists

    struct Piece {
        std::size_t region = 0;
        Index src_lo, extent;
        std::uint8_t* dst = nullptr;
    };
    struct Point {
        std::int64_t src = 0;  // byte offset in the chunk
        std::uint8_t* dst = nullptr;
    };
    // Everything one chunk supplies: boxes (read_many) or single elements (read_points).
    struct Work {
        Index idx;
        std::vector<Piece> pieces;
        std::vector<Point> points;
    };

    // Resolves every chunk in parallel, in chunk order.
    void run(std::vector<Work>& work, const std::map<std::int64_t, std::size_t>& slot,
             const std::vector<Index>& out_strides)
    {
        std::vector<Work*> ordered;
        ordered.reserve(work.size());
        for (auto& [linear, i] : slot)
            ordered.push_back(&work[i]);
        if (spec_.hint && !cache_)
            pool_.parallel_for(ordered.size(), [&](std::size_t i) { hint(*ordered[i]); });
        pool_.parallel_for(ordered.size(), [&](std::size_t i) { resolve(*ordered[i], out_strides); });
    }

    void gather(const std::uint8_t* chunk, const Work& w) const
    {
        for (const auto& pt : w.points)
            std::memcpy(pt.dst, chunk + pt.src, spec_.itemsize);
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

    void apply(const ChunkPtr& chunk, const Work& w, const std::vector<Index>& out_strides)
    {
        if (!chunk) {
            fill_pieces(w, out_strides);
            return;
        }
        for (const auto& p : w.pieces)
            copy_box(chunk->data() + offset_of(p.src_lo, chunk_strides_), chunk_strides_, p.dst,
                     out_strides[p.region], p.extent, spec_.itemsize);
        gather(chunk->data(), w);
    }

    void fill_pieces(const Work& w, const std::vector<Index>& out_strides)
    {
        for (const auto& p : w.pieces)
            fill_box(p.dst, out_strides[p.region], p.extent, spec_.fill);
        for (const auto& pt : w.points)
            std::memcpy(pt.dst, spec_.fill.data(), spec_.itemsize);
    }

    // Reads the pieces of a raw (uncompressed) chunk file straight into the outputs.
    Found read_raw_pieces(const std::string& path, const Work& w, const std::vector<Index>& out_strides)
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
            gather(m->data, w);
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
                hi = std::max(hi, pt.src + static_cast<std::int64_t>(spec_.itemsize));
            }
            pread_exact(f.fd(), scratch.data(), static_cast<std::size_t>(hi - lo), lo, path);
            for (const auto& pt : w.points)
                std::memcpy(pt.dst, scratch.data() + (pt.src - lo), spec_.itemsize);
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

    void resolve(const Work& w, const std::vector<Index>& out_strides)
    {
        const std::string k = key(w.idx);
        const std::string ck = spec_.identity + "|" + k;
        if (cache_) {
            auto [found, chunk] = cache_->get(ck);
            if (found) {
                ++stats_.memory_hits;
                apply(chunk, w, out_strides);
                return;
            }
        }
        if (writes_) {  // fetched, but its cache file is not written yet
            auto [found, chunk] = writes_->pending(k);
            if (found) {
                ++stats_.pending_hits;
                apply(chunk, w, out_strides);
                return;
            }
        }
        for (std::size_t li = 0; li < spec_.layers.size(); ++li) {
            const LayerSpec& l = spec_.layers[li];
            if (!cache_ && codecs_[li] == Codec::Raw) {
                if (read_raw_pieces(l.root + "/" + k, w, out_strides) == Found::Copied) {
                    ++stats_.disk_hits;
                    return;
                }
                if (l.kind == "array" || exists(marker_path(l, k))) {
                    fill_pieces(w, out_strides);
                    return;
                }
                continue;
            }
            if (auto chunk = read_layer_chunk(li, k)) {
                ++stats_.disk_hits;
                if (cache_)
                    cache_->put(ck, *chunk);
                apply(*chunk, w, out_strides);
                return;
            }
        }
        if (spec_.offline)
            throw CacheMiss("chunk " + k + " of " + spec_.identity + " is not in the disk cache (offline)");
        if (!http_) {
            fill_pieces(w, out_strides);
            return;
        }
        ChunkPtr chunk = fetch(k);
        if (cache_)
            cache_->put(ck, chunk);
        apply(chunk, w, out_strides);
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
