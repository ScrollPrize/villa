#pragma once
// Three-tier chunk cache: RAM -> SSD -> Network (S3/HTTP).
//
// Hot tier:  LRUCache of decompressed ChunkData in RAM.
// Cold tier: Shard files on local SSD (whole shards cached from network).
// Ice tier:  Remote shard files fetched via HTTP/S3.
//
// All I/O is async via ThreadPool. Callers get best-available data
// without blocking (coarser pyramid level if fine isn't cached yet).

#include "lru_cache.hpp"
#include "shard.hpp"
#include "thread_pool.hpp"
#include "types.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace vc {

// H265 decode function — provided by caller (links libde265).
// Decodes compressed bytes → 128^3 u8 voxels.
using DecodeFn = std::function<ChunkData(std::span<const uint8_t> compressed)>;

// Fetch function for remote shards — provided by caller (links libcurl).
// Downloads entire shard file given a URL.
using FetchShardFn = std::function<std::vector<uint8_t>(const std::string& url)>;

struct CacheStats {
    uint64_t hot_hits = 0, hot_misses = 0;
    uint64_t cold_hits = 0, cold_misses = 0;
    uint64_t ice_fetches = 0;
    uint64_t decode_count = 0;
    size_t hot_bytes = 0, hot_count = 0;
};

class ChunkCache {
    // Hot tier: decompressed chunks in RAM
    LRUCache<ChunkKey, std::shared_ptr<ChunkData>, ChunkKeyHash> hot_;

    // Cold tier: shard files on local SSD
    std::filesystem::path cold_root_;

    // Ice tier: remote base URL
    std::string remote_base_url_;
    FetchShardFn fetch_shard_;

    // Decode H265
    DecodeFn decode_;

    // Async I/O
    ThreadPool io_pool_;

    // Shard-level locking: only one thread fetches a given shard at a time
    std::mutex shard_mu_;
    std::unordered_map<ShardKey, bool, ShardKeyHash> shard_inflight_;

    // Stats
    mutable std::atomic<uint64_t> hot_hits_{0}, hot_misses_{0};
    mutable std::atomic<uint64_t> cold_hits_{0}, cold_misses_{0};
    mutable std::atomic<uint64_t> ice_fetches_{0}, decode_count_{0};

    // Volume metadata
    VolumeMeta meta_;

    // Chunk ready callback
    std::function<void()> on_chunk_ready_;

    // Decode a compressed chunk and insert into hot cache
    std::shared_ptr<ChunkData> decode_and_cache(const ChunkKey& key,
                                                  std::span<const uint8_t> compressed) {
        auto chunk = std::make_shared<ChunkData>(decode_(compressed));
        hot_.put(key, chunk);
        decode_count_.fetch_add(1, std::memory_order_relaxed);
        return chunk;
    }

    // Try to read chunk from cold tier (local shard file)
    std::shared_ptr<ChunkData> try_cold(const ChunkKey& key) {
        auto path = shard_path(cold_root_, key.level, key.sz, key.sy, key.sx);
        auto compressed = read_chunk_compressed(path, key.cz, key.cy, key.cx);
        if (compressed.empty()) return nullptr;
        cold_hits_.fetch_add(1, std::memory_order_relaxed);
        return decode_and_cache(key, compressed);
    }

    // Fetch shard from network, save to cold tier, decode requested chunk
    void fetch_shard_async(const ShardKey& sk) {
        io_pool_.enqueue([this, sk] {
            // Check if already fetched
            {
                std::lock_guard lk(shard_mu_);
                if (shard_inflight_.contains(sk)) return;
                shard_inflight_[sk] = true;
            }

            // Build URL and fetch
            auto url = std::format("{}/{}/{}.{}.{}.shard",
                remote_base_url_, sk.level, sk.sz, sk.sy, sk.sx);
            auto shard_data = fetch_shard_(url);
            ice_fetches_.fetch_add(1, std::memory_order_relaxed);

            // Save to cold tier
            if (!cold_root_.empty() && !shard_data.empty()) {
                auto path = shard_path(cold_root_, sk.level, sk.sz, sk.sy, sk.sx);
                std::filesystem::create_directories(path.parent_path());
                FILE* f = fopen(path.c_str(), "wb");
                if (f) { fwrite(shard_data.data(), 1, shard_data.size(), f); fclose(f); }
            }

            // Decode all chunks from this shard into hot cache
            if (!shard_data.empty()) {
                auto idx = read_shard_index(shard_data);
                for (int cz = 0; cz < CHUNKS_PER; ++cz)
                for (int cy = 0; cy < CHUNKS_PER; ++cy)
                for (int cx = 0; cx < CHUNKS_PER; ++cx) {
                    auto e = idx.at(cz, cy, cx);
                    if (e.length == 0) continue;
                    ChunkKey ck{sk.level, sk.sz, sk.sy, sk.sx, cz, cy, cx};
                    if (hot_.contains(ck)) continue;
                    auto span = std::span<const uint8_t>(
                        shard_data.data() + e.offset, e.length);
                    decode_and_cache(ck, span);
                }
            }

            {
                std::lock_guard lk(shard_mu_);
                shard_inflight_.erase(sk);
            }

            if (on_chunk_ready_) on_chunk_ready_();
        });
    }

public:
    struct Config {
        size_t hot_max_bytes = 2ULL << 30;  // 2 GB RAM cache
        int io_threads = 4;
        std::filesystem::path cold_root;     // empty = no SSD cache
        std::string remote_base_url;         // empty = local only
        DecodeFn decode;
        FetchShardFn fetch_shard;
    };

    explicit ChunkCache(Config cfg, VolumeMeta meta)
        : hot_(cfg.hot_max_bytes, [](const std::shared_ptr<ChunkData>& c) { return c->byte_size(); })
        , cold_root_(cfg.cold_root)
        , remote_base_url_(cfg.remote_base_url)
        , fetch_shard_(std::move(cfg.fetch_shard))
        , decode_(std::move(cfg.decode))
        , io_pool_(cfg.io_threads)
        , meta_(meta)
    {}

    // Get a chunk. Returns nullptr if not in hot cache.
    // Triggers async fetch from cold/ice if missing.
    std::shared_ptr<ChunkData> get(const ChunkKey& key) {
        // Hot tier
        if (auto v = hot_.get(key)) {
            hot_hits_.fetch_add(1, std::memory_order_relaxed);
            return *v;
        }
        hot_misses_.fetch_add(1, std::memory_order_relaxed);

        // Async: try cold then ice
        io_pool_.enqueue([this, key] {
            auto result = try_cold(key);
            if (result) {
                if (on_chunk_ready_) on_chunk_ready_();
                return;
            }
            cold_misses_.fetch_add(1, std::memory_order_relaxed);

            if (!remote_base_url_.empty() && fetch_shard_)
                fetch_shard_async(key.shard());
        });

        return nullptr;
    }

    // Get best available: try requested level, then coarser levels.
    std::shared_ptr<ChunkData> get_best(const ChunkKey& key) {
        if (auto v = hot_.get(key)) {
            hot_hits_.fetch_add(1, std::memory_order_relaxed);
            return *v;
        }
        // Try coarser levels
        for (int l = key.level + 1; l < meta_.levels; ++l) {
            int scale = 1 << (l - key.level);
            ChunkKey coarse{l,
                key.sz / scale, key.sy / scale, key.sx / scale,
                (key.cz / scale) % CHUNKS_PER,
                (key.cy / scale) % CHUNKS_PER,
                (key.cx / scale) % CHUNKS_PER};
            if (auto v = hot_.get(coarse)) return *v;
        }
        // Trigger async fetch of the fine level
        get(key);
        return nullptr;
    }

    // Prefetch all chunks in a world-space bounding box at a given level.
    void prefetch(Box3f bbox, int level) {
        int s = 1 << level;
        int sz0 = int(bbox.min.z) / (SHARD_DIM * s);
        int sy0 = int(bbox.min.y) / (SHARD_DIM * s);
        int sx0 = int(bbox.min.x) / (SHARD_DIM * s);
        int sz1 = int(bbox.max.z) / (SHARD_DIM * s);
        int sy1 = int(bbox.max.y) / (SHARD_DIM * s);
        int sx1 = int(bbox.max.x) / (SHARD_DIM * s);

        for (int sz = sz0; sz <= sz1; ++sz)
        for (int sy = sy0; sy <= sy1; ++sy)
        for (int sx = sx0; sx <= sx1; ++sx) {
            ShardKey sk{level, sz, sy, sx};
            // Check if any chunk from this shard is missing
            bool need = false;
            for (int c = 0; c < INDEX_COUNT && !need; ++c) {
                ChunkKey ck{level, sz, sy, sx,
                    c / 64, (c / 8) % 8, c % 8};
                if (!hot_.contains(ck)) need = true;
            }
            if (need) {
                if (!cold_root_.empty())
                    io_pool_.enqueue([this, sk] {
                        // Load entire cold shard
                        auto path = shard_path(cold_root_, sk.level, sk.sz, sk.sy, sk.sx);
                        auto data = read_shard_file(path);
                        if (!data.empty()) {
                            auto idx = read_shard_index(data);
                            for (int i = 0; i < INDEX_COUNT; ++i) {
                                if (idx.entries[i].length == 0) continue;
                                ChunkKey ck{sk.level, sk.sz, sk.sy, sk.sx,
                                    i / 64, (i / 8) % 8, i % 8};
                                if (hot_.contains(ck)) continue;
                                decode_and_cache(ck, std::span(
                                    data.data() + idx.entries[i].offset,
                                    idx.entries[i].length));
                            }
                            if (on_chunk_ready_) on_chunk_ready_();
                        } else if (!remote_base_url_.empty() && fetch_shard_) {
                            fetch_shard_async(sk);
                        }
                    });
                else if (!remote_base_url_.empty() && fetch_shard_)
                    fetch_shard_async(sk);
            }
        }
    }

    void set_on_chunk_ready(std::function<void()> cb) { on_chunk_ready_ = std::move(cb); }

    CacheStats stats() const {
        return {
            hot_hits_.load(), hot_misses_.load(),
            cold_hits_.load(), cold_misses_.load(),
            ice_fetches_.load(), decode_count_.load(),
            hot_.byte_size(), hot_.size()
        };
    }

    void clear() { hot_.clear(); }
};

} // namespace vc
