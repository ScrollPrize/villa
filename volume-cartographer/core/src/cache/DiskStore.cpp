#include "vc/core/cache/DiskStore.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/CacheUtils.hpp"
#include <utils/hash.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vc::cache {

DiskStore::DiskStore(Config config) : config_(std::move(config))
{
    if (!config_.root.empty()) {
        std::filesystem::create_directories(config_.root);
        if (std::filesystem::exists(config_.root)) {
            initTotalBytes();
        }
    }
}

DiskStore::~DiskStore()
{
    if (!config_.persistent && !config_.root.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(config_.root, ec);
    }
}

std::filesystem::path DiskStore::chunkPath(
    const std::string& volumeId,
    const ChunkKey& key) const
{
    auto base = config_.directMode
                    ? config_.root
                    : config_.root / volumeId;
    return base / std::to_string(key.level) / chunkFilename(key, config_.delimiter);
}

// Lock key combines volumeId and chunk key for per-key serialization.
static size_t diskLockKey(const std::string& volumeId, const ChunkKey& key) noexcept
{
    return utils::hash_combine(std::hash<std::string>()(volumeId), ChunkKeyHash()(key));
}

std::optional<std::vector<uint8_t>> DiskStore::get(
    const std::string& volumeId,
    const ChunkKey& key) const
{
    return readFileToVector(chunkPath(volumeId, key));
}

void DiskStore::put(
    const std::string& volumeId,
    const ChunkKey& key,
    const uint8_t* data,
    size_t size)
{
    auto path = chunkPath(volumeId, key);

    // Serialize writes to same key
    auto lock = lockPool_.lock<size_t>(diskLockKey(volumeId, key));

    // Create parent directories
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[DISK] put: create_directories FAILED for %s: %s\n",
                         path.parent_path().c_str(), ec.message().c_str());
        return;
    }

    // Write to temp file, then rename (atomic on same filesystem)
    auto tmpPath = path;
    tmpPath += ".tmp";

    int fd = ::open(tmpPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[DISK] put: open FAILED for %s errno=%d\n",
                         tmpPath.c_str(), errno);
        return;
    }

    size_t written = 0;
    while (written < size) {
        ssize_t n = ::write(fd, data + written, size - written);
        if (n <= 0) {
            int err = errno;
            if (err == ENOSPC) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[DISK] put: DISK FULL writing %s\n", tmpPath.c_str());
            } else {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[DISK] put: write FAILED for %s errno=%d (%s)\n",
                                 tmpPath.c_str(), err, strerror(err));
            }
            ::close(fd);
            ::unlink(tmpPath.c_str());
            return;
        }
        written += static_cast<size_t>(n);
    }
    ::close(fd);

    // Atomic rename
    std::filesystem::rename(tmpPath, path, ec);
    if (ec) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[DISK] put: rename FAILED %s -> %s: %s\n",
                         tmpPath.c_str(), path.c_str(), ec.message().c_str());
        ::unlink(tmpPath.c_str());
    } else {
        // Update in-memory index
        size_t oldSize = 0;
        auto mtime = std::filesystem::last_write_time(path, ec);
        if (!ec) {
            std::lock_guard<std::mutex> lk(indexMtx_);
            // Remove old entry if overwriting an existing file
            auto pit = pathIndex_.find(path.string());
            if (pit != pathIndex_.end()) {
                oldSize = pit->second->second.bytes;
                timeIndex_.erase(pit->second);
                pathIndex_.erase(pit);
            }
            auto it = timeIndex_.emplace(mtime, IndexEntry{path, size});
            pathIndex_.emplace(path.string(), it);
        }
        // Adjust totalBytes_: add new size, subtract old if overwriting
        if (oldSize > 0 && oldSize <= size) {
            totalBytes_.fetch_add(size - oldSize, std::memory_order_relaxed);
        } else if (oldSize > size) {
            auto prev = totalBytes_.load(std::memory_order_relaxed);
            auto sub = oldSize - size;
            while (prev >= sub &&
                   !totalBytes_.compare_exchange_weak(
                       prev, prev - sub, std::memory_order_relaxed))
                ;
        } else {
            totalBytes_.fetch_add(size, std::memory_order_relaxed);
        }
    }

    // Enforce disk budget: evict oldest entries if over maxBytes
    if (config_.maxBytes > 0 &&
        totalBytes_.load(std::memory_order_relaxed) > config_.maxBytes) {
        evictToSize(config_.maxBytes);
    }
}

void DiskStore::remove(
    const std::string& volumeId,
    const ChunkKey& key)
{
    auto path = chunkPath(volumeId, key);
    std::error_code ec;

    // Get file size before removing so we can update the tracked total
    auto sz = std::filesystem::file_size(path, ec);
    if (ec) {
        // File doesn't exist or can't stat — nothing to subtract
        std::filesystem::remove(path, ec);
        return;
    }

    std::filesystem::remove(path, ec);
    if (!ec) {
        // Remove from in-memory index
        {
            std::lock_guard<std::mutex> lk(indexMtx_);
            removeFromIndex(path);
        }

        auto prev = totalBytes_.load(std::memory_order_relaxed);
        auto sub = static_cast<size_t>(sz);
        // Clamp to avoid underflow
        while (prev >= sub &&
               !totalBytes_.compare_exchange_weak(
                   prev, prev - sub, std::memory_order_relaxed))
            ;
    }
}

void DiskStore::evictToSize(size_t targetBytes)
{
    if (config_.root.empty()) return;

    // Fast path: if tracked total is already within budget, skip eviction
    size_t tracked = totalBytes_.load(std::memory_order_relaxed);
    if (tracked > 0 && tracked <= targetBytes) return;

    // Use in-memory index to evict oldest entries (already sorted by mtime)
    std::lock_guard<std::mutex> lk(indexMtx_);

    std::error_code ec;
    auto it = timeIndex_.begin();
    while (it != timeIndex_.end()) {
        tracked = totalBytes_.load(std::memory_order_relaxed);
        if (tracked <= targetBytes) break;

        auto& entry = it->second;
        std::filesystem::remove(entry.path, ec);
        if (!ec) {
            // Update totalBytes_, clamping to avoid underflow
            auto prev = totalBytes_.load(std::memory_order_relaxed);
            while (prev >= entry.bytes &&
                   !totalBytes_.compare_exchange_weak(
                       prev, prev - entry.bytes, std::memory_order_relaxed))
                ;

            pathIndex_.erase(entry.path.string());
            it = timeIndex_.erase(it);
        } else {
            // File already gone or inaccessible; remove stale index entry
            pathIndex_.erase(entry.path.string());
            it = timeIndex_.erase(it);
        }
    }
}

size_t DiskStore::totalBytes() const
{
    return totalBytes_.load(std::memory_order_relaxed);
}

void DiskStore::initTotalBytes()
{
    if (config_.root.empty() || !std::filesystem::exists(config_.root)) {
        totalBytes_.store(0, std::memory_order_relaxed);
        return;
    }

    size_t total = 0;
    size_t fileCount = 0;
    std::error_code ec;

    std::lock_guard<std::mutex> lk(indexMtx_);
    timeIndex_.clear();
    pathIndex_.clear();

    for (auto& entry :
         std::filesystem::recursive_directory_iterator(config_.root, ec)) {
        if (!entry.is_regular_file()) continue;
        auto name = entry.path().filename().string();
        // Skip hidden, temp, and metadata files
        if (name.starts_with(".") || name.ends_with(".tmp") ||
            name.ends_with(".json"))
            continue;

        auto sz = static_cast<size_t>(entry.file_size());
        total += sz;
        fileCount++;

        auto mtime = entry.last_write_time();
        auto it = timeIndex_.emplace(mtime, IndexEntry{entry.path(), sz});
        pathIndex_.emplace(entry.path().string(), it);
    }
    totalBytes_.store(total, std::memory_order_relaxed);
    std::fprintf(stderr, "[DiskStore] initTotalBytes: root=%s files=%zu bytes=%zu (%.1f MB)\n",
                 config_.root.c_str(), fileCount, total, total / (1024.0 * 1024.0));
}

void DiskStore::clearVolume(const std::string& volumeId)
{
    auto volDir = config_.root / volumeId;
    std::error_code ec;
    std::filesystem::remove_all(volDir, ec);
    // Re-scan since we can't easily know how much was in this volume
    initTotalBytes();
}

void DiskStore::clearAll()
{
    if (config_.root.empty()) return;
    std::error_code ec;
    for (auto& entry : std::filesystem::directory_iterator(config_.root, ec)) {
        std::filesystem::remove_all(entry.path(), ec);
    }
    {
        std::lock_guard<std::mutex> lk(indexMtx_);
        timeIndex_.clear();
        pathIndex_.clear();
    }
    totalBytes_.store(0, std::memory_order_relaxed);
}

void DiskStore::removeFromIndex(const std::filesystem::path& path)
{
    // Caller must hold indexMtx_
    auto pit = pathIndex_.find(path.string());
    if (pit != pathIndex_.end()) {
        timeIndex_.erase(pit->second);
        pathIndex_.erase(pit);
    }
}

}  // namespace vc::cache
