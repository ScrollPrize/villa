#pragma once
// Frame cache: RAM LRU of decoded 1024x1024 frames.
//
// When a frame is needed:
//   1. Check RAM LRU (hit → instant)
//   2. Find the shard file on disk, read the slab, decode ALL 128 frames,
//      cache them all (spatial locality means you'll need neighbors soon)
//   3. If shard is remote, fetch it first, save to SSD, then decode
//
// Key = (level, global_z) → Frame (1024x1024 u8, ~1MB)
// 2GB budget ≈ 2000 cached frames

#include "lru_cache.hpp"
#include "shard.hpp"
#include "thread_pool.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vc {

// Decode function: slab bitstream → vector of 128 Frames
using DecodeSlabFn = std::function<std::vector<Frame>(std::span<const uint8_t>)>;

// Fetch function: download shard file by URL
using FetchShardFn = std::function<std::vector<uint8_t>(const std::string& url)>;

struct CacheStats {
    uint64_t hits = 0, misses = 0, decodes = 0;
    size_t bytes = 0, count = 0;
};

class FrameCache {
    LRUCache<FrameKey, std::shared_ptr<Frame>, FrameKeyHash> lru_;
    std::filesystem::path volume_root_;
    std::filesystem::path cold_root_;
    std::string remote_url_;
    FetchShardFn fetch_;
    DecodeSlabFn decode_;
    ThreadPool io_;
    std::function<void()> on_ready_;

    // Track in-flight slab decodes to avoid duplicates
    std::mutex mu_;
    std::unordered_map<uint64_t, bool> inflight_;

    mutable std::atomic<uint64_t> hits_{0}, misses_{0}, decodes_{0};

    // Pack slab identity into a single key for inflight tracking
    static uint64_t slab_id(int level, int gz) {
        return uint64_t(level) << 48 | uint64_t(gz / SLAB_FRAMES);
    }

    // Decode an entire slab and cache all 128 frames
    void decode_slab(int level, int global_z_base, std::span<const uint8_t> compressed) {
        auto frames = decode_(compressed);
        for (int i = 0; i < int(frames.size()); ++i) {
            FrameKey fk{level, global_z_base + i};
            if (!lru_.contains(fk))
                lru_.put(fk, std::make_shared<Frame>(std::move(frames[i])));
        }
        decodes_.fetch_add(1, std::memory_order_relaxed);
        if (on_ready_) on_ready_();
    }

    // Find and decode a slab from local shard files
    bool try_local(int level, int global_z) {
        // Which shard and which slab within it?
        int shard_z = global_z / SHARD_DIM;
        int slab_in_shard = (global_z % SHARD_DIM) / SLAB_FRAMES;
        int global_z_base = shard_z * SHARD_DIM + slab_in_shard * SLAB_FRAMES;

        // Try volume root first, then cold cache
        for (auto& root : {volume_root_, cold_root_}) {
            if (root.empty()) continue;
            // We need to find the shard. For now, shard coords are (sz, 0, 0)
            // since we're addressing by global z. Full 3D shard addressing:
            // This cache stores FRAMES, so shard sy/sx are always implicit
            // from the volume layout. But the shard file contains one
            // 1024x1024 video, so sy/sx are baked into the video content.
            // For now: assume single-shard-wide volumes or caller provides paths.
            auto path = shard_path(root, level, shard_z, 0, 0);
            auto compressed = read_slab_compressed(path, slab_in_shard);
            if (!compressed.empty()) {
                decode_slab(level, global_z_base, compressed);
                return true;
            }
        }
        return false;
    }

public:
    struct Config {
        size_t max_bytes = 2ULL << 30;
        int io_threads = 4;
        std::filesystem::path volume_root;
        std::filesystem::path cold_root;
        std::string remote_url;
        DecodeSlabFn decode;
        FetchShardFn fetch;
    };

    explicit FrameCache(Config cfg, VolumeMeta meta)
        : lru_(cfg.max_bytes, [](const std::shared_ptr<Frame>& f) { return f->byte_size(); })
        , volume_root_(cfg.volume_root)
        , cold_root_(cfg.cold_root)
        , remote_url_(cfg.remote_url)
        , fetch_(std::move(cfg.fetch))
        , decode_(std::move(cfg.decode))
        , io_(cfg.io_threads)
    {}

    // Get a frame. Returns nullptr if not cached yet; triggers async decode.
    std::shared_ptr<Frame> get(FrameKey key) {
        if (auto v = lru_.get(key)) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return *v;
        }
        misses_.fetch_add(1, std::memory_order_relaxed);

        // Async decode
        auto sid = slab_id(key.level, key.z);
        {
            std::lock_guard lk(mu_);
            if (inflight_.contains(sid)) return nullptr;
            inflight_[sid] = true;
        }

        io_.enqueue([this, key, sid] {
            try_local(key.level, key.z);
            std::lock_guard lk(mu_);
            inflight_.erase(sid);
        });

        return nullptr;
    }

    // Get frame, or try coarser levels if not available
    std::shared_ptr<Frame> get_best(FrameKey key, int max_levels) {
        if (auto f = lru_.get(key)) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return *f;
        }
        for (int l = key.level + 1; l < max_levels; ++l) {
            FrameKey coarse{l, key.z >> (l - key.level)};
            if (auto f = lru_.get(coarse)) return *f;
        }
        get(key);  // trigger async fetch
        return nullptr;
    }

    void set_on_ready(std::function<void()> cb) { on_ready_ = std::move(cb); }

    CacheStats stats() const {
        return {hits_.load(), misses_.load(), decodes_.load(),
                lru_.byte_size(), lru_.size()};
    }

    void clear() { lru_.clear(); }
};

} // namespace vc
