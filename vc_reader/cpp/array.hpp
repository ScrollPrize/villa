// Native read path for one zarr v2 array: plan, then resolve every chunk through
// the memory cache, the disk layers (in their own formats) and the remote, and
// copy into the caller's output buffers.
#pragma once

#include "chunk_io.hpp"
#include "codecs.hpp"
#include "http.hpp"
#include "memory_cache.hpp"
#include "thread_pool.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
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
    std::string read_mode = "slab";  // raw local chunks: "slab" (pread rows) or "mmap"
    bool hint = false;               // queue readahead for every piece before reading
    std::size_t threads = 16;
    HttpOptions http;
};

struct Stats {
    std::atomic<std::uint64_t> memory_hits{0}, disk_hits{0}, remote_fetches{0}, remote_missing{0},
        remote_bytes{0}, write_errors{0};
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
        if (!spec_.remote_url.empty())
            http_ = std::make_unique<HttpClient>(spec_.http);
        if (spec_.write_layer >= 0 && spec_.layers.at(spec_.write_layer).kind == "vc3d-delta3d" && nd != 3)
            throw std::invalid_argument("Delta3D caches hold 3-D chunks only");
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
                    work.push_back(Work{idx, {}});
                work[it->second].pieces.push_back(std::move(p));
            } while (next_index(idx, c0, c1));
        }
        std::vector<Work*> ordered;
        ordered.reserve(work.size());
        for (auto& [linear, i] : slot)
            ordered.push_back(&work[i]);
        if (spec_.hint && !cache_)
            pool_.parallel_for(ordered.size(), [&](std::size_t i) { hint(*ordered[i]); });
        pool_.parallel_for(ordered.size(), [&](std::size_t i) { resolve(*ordered[i], out_strides); });
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

    struct Piece {
        std::size_t region = 0;
        Index src_lo, extent;
        std::uint8_t* dst = nullptr;
    };
    struct Work {
        Index idx;
        std::vector<Piece> pieces;
    };
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
        if (spec_.layers.empty() || codecs_[0] != Codec::Raw || spec_.read_mode != "slab")
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
        for (const auto& p : w.pieces) {
            if (!chunk)
                fill_box(p.dst, out_strides[p.region], p.extent, spec_.fill);
            else
                copy_box(chunk->data() + offset_of(p.src_lo, chunk_strides_), chunk_strides_, p.dst,
                         out_strides[p.region], p.extent, spec_.itemsize);
        }
    }

    void fill_pieces(const Work& w, const std::vector<Index>& out_strides)
    {
        for (const auto& p : w.pieces)
            fill_box(p.dst, out_strides[p.region], p.extent, spec_.fill);
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
            store_missing(k);
            return nullptr;
        }
        stats_.remote_bytes += body.size();
        std::shared_ptr<Chunk> chunk;
        if (remote_codec_ == Codec::Raw) {
            if (body.size() != chunk_bytes_)
                throw std::runtime_error(spec_.remote_url + "/" + k + ": body is not one chunk");
            chunk = std::make_shared<Chunk>(std::move(body));
            store(k, *chunk, *chunk);
        } else {
            chunk = std::make_shared<Chunk>(chunk_bytes_);
            decode(remote_codec_, body.data(), body.size(), chunk->data(), chunk_bytes_);
            store(k, *chunk, body);
        }
        return chunk;
    }

    void store(const std::string& k, const Chunk& decoded, const std::vector<std::uint8_t>& payload)
    {
        if (spec_.write_layer < 0 || write_disabled_.load())
            return;
        const LayerSpec& l = spec_.layers[spec_.write_layer];
        try {
            const std::string path = l.root + "/" + k;
            if (l.kind == "decoded")
                atomic_write(path, decoded.data(), decoded.size());
            else if (l.kind == "vc3d-mirror")
                atomic_write(path, payload.data(), payload.size());  // the source's bytes, verbatim
            else if (l.kind == "vc3d-delta3d") {
                const std::array<int, 3> shape{static_cast<int>(spec_.chunks[0]), static_cast<int>(spec_.chunks[1]),
                                               static_cast<int>(spec_.chunks[2])};
                auto enc = encode_delta3d(decoded.data(), decoded.size(), shape, spec_.itemsize);
                atomic_write(path, enc.data(), enc.size());
            }
        } catch (const std::exception& e) {
            write_failed(e.what());
        }
    }

    void store_missing(const std::string& k)
    {
        if (spec_.write_layer < 0 || write_disabled_.load())
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
};

}  // namespace vcr
