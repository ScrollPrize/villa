#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ChunkKey.hpp"
#include <utils/lock_pool.hpp>

namespace vc::cache {

// Persistent on-disk cache for compressed chunk data.
// Directory layout:
//   <root>/<volumeId>/<level>/<iz>.<iy>.<ix>
//
// Thread-safe for concurrent read/write to different keys.
// Same-key writes are serialized via a lock pool.
class DiskStore {
public:
    struct Config {
        std::filesystem::path root;           // cache root directory
        size_t maxBytes = 100ULL << 30;       // 100 GB default
        bool persistent = true;               // if false, cleaned up on destruction
        bool directMode = false;              // skip volumeId subdirectory
        std::string delimiter = ".";          // chunk index separator ("." or "/")
    };

    explicit DiskStore(Config config);
    ~DiskStore();

    DiskStore(const DiskStore&) = delete;
    DiskStore& operator=(const DiskStore&) = delete;

    // Read compressed chunk bytes from disk. Returns nullopt on miss.
    [[nodiscard]] std::optional<std::vector<uint8_t>> get(
        const std::string& volumeId,
        const ChunkKey& key) const;

    // Write compressed chunk bytes to disk. Creates directories as needed.
    void put(
        const std::string& volumeId,
        const ChunkKey& key,
        const uint8_t* data,
        size_t size);

    void put(
        const std::string& volumeId,
        const ChunkKey& key,
        const std::vector<uint8_t>& data)
    {
        put(volumeId, key, data.data(), data.size());
    }

    // Remove a specific chunk from disk.
    void remove(const std::string& volumeId, const ChunkKey& key);

    // Evict oldest files until total size <= targetBytes.
    // Uses file modification time for ordering.
    void evictToSize(size_t targetBytes);

    // Total bytes currently stored on disk.
    // Returns the incrementally tracked value (fast, no scan).
    [[nodiscard]] size_t totalBytes() const;

    // Perform initial directory scan to populate totalBytes_.
    // Called automatically by the constructor if the cache directory exists.
    void initTotalBytes();

    // Remove all cached data for a volume.
    void clearVolume(const std::string& volumeId);

    // Remove all cached data.
    void clearAll();

    [[nodiscard]] const std::filesystem::path& root() const { return config_.root; }

private:
    std::filesystem::path chunkPath(
        const std::string& volumeId,
        const ChunkKey& key) const;

    // Remove the index entry for a file path (caller must hold indexMtx_).
    void removeFromIndex(const std::filesystem::path& path);

    // Per-key lock pool to serialize same-key writes
    mutable utils::LockPool<64> lockPool_;

    Config config_;

    // Incrementally tracked total bytes on disk.
    // Initialized to 0; populated by initTotalBytes() on construction.
    std::atomic<size_t> totalBytes_{0};

    // In-memory index of cached files, ordered by modification time.
    // Enables O(1) eviction of the oldest entry instead of O(n) dir scan.
    struct IndexEntry {
        std::filesystem::path path;
        size_t bytes;
    };
    std::mutex indexMtx_;
    // mtime -> entry (oldest first via default ordering)
    std::multimap<std::filesystem::file_time_type, IndexEntry> timeIndex_;
    // path -> iterator into timeIndex_ for O(1) lookup/removal by path
    std::unordered_map<std::string, decltype(timeIndex_)::iterator> pathIndex_;
};

}  // namespace vc::cache
