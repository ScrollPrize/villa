#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ChunkKey.hpp"
#include "HttpMetadataFetcher.hpp"  // HttpAuth (= utils::AwsAuth)

namespace utils { class HttpClient; }
namespace utils::detail { struct ShardIndex; }

namespace vc::cache {

struct FetchResult {
    std::vector<uint8_t> bytes;
    bool wasAbsent = false;
    bool transientError = false;
};

class VolumeSource {
public:
    virtual ~VolumeSource() = default;
    [[nodiscard]] virtual FetchResult fetch(const ChunkKey& key) = 0;
    [[nodiscard]] virtual int numLevels() const noexcept = 0;
    [[nodiscard]] virtual std::array<int, 3> chunkShape(int level) const noexcept = 0;
    [[nodiscard]] virtual std::array<int, 3> levelShape(int level) const noexcept = 0;
    [[nodiscard]] virtual bool isSharded() const noexcept { return false; }
};

// Reads compressed chunks from a local zarr v2 directory.
class FileSystemSource : public VolumeSource {
public:
    struct LevelMeta {
        std::array<int, 3> shape;
        std::array<int, 3> chunkShape;
        std::string dirName;  // actual directory name (e.g. "3" for scale 3)
    };

    explicit FileSystemSource(
        const std::filesystem::path& zarrRoot,
        const std::string& delimiter = ".");

    FileSystemSource(
        const std::filesystem::path& zarrRoot,
        const std::string& delimiter,
        std::vector<LevelMeta> levels);

    [[nodiscard]] FetchResult fetch(const ChunkKey& key) override;
    [[nodiscard]] int numLevels() const noexcept override;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept override;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept override;

private:
    std::filesystem::path chunkPath(const ChunkKey& key) const;
    void discoverLevels();

    std::filesystem::path root_;
    std::string delimiter_;
    std::vector<LevelMeta> levels_;
};

struct ShardConfig;

// Fetches compressed chunks from an HTTP/HTTPS zarr store via utils::HttpClient.
class HttpSource : public VolumeSource {
public:
    using LevelMeta = FileSystemSource::LevelMeta;

    HttpSource(
        const std::string& baseUrl,
        const std::string& delimiter,
        std::vector<LevelMeta> levels,
        HttpAuth auth = {});

    ~HttpSource() override;

    void setShardConfig(const ShardConfig& config);

    [[nodiscard]] FetchResult fetch(const ChunkKey& key) override;
    [[nodiscard]] int numLevels() const noexcept override;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept override;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept override;

    [[nodiscard]] std::vector<uint8_t> fetchWholeShard(int level, int sz, int sy, int sx);
    [[nodiscard]] bool isSharded() const noexcept override { return sharded_; }
    [[nodiscard]] std::array<int, 3> shardsPerAxis(int level) const noexcept;
    [[nodiscard]] std::array<int, 3> chunksPerShard() const noexcept { return chunksPerShard_; }
    [[nodiscard]] std::array<int, 3> shardShape() const noexcept { return shardShape_; }

    [[nodiscard]] bool hadTransientError() const noexcept {
        return transientError_.load(std::memory_order_relaxed);
    }

private:
    std::string chunkUrl(const ChunkKey& key) const;
    std::string shardUrl(const ChunkKey& key) const;
    int innerChunkIndex(const ChunkKey& key) const noexcept;
    int totalChunksPerShard() const noexcept;
    FetchResult fetchFromShard(const ChunkKey& key);
    FetchResult httpGet(const std::string& url);
    FetchResult httpGetRange(const std::string& url,
                             std::size_t offset, std::size_t length);

    std::string baseUrl_;
    std::string delimiter_;
    std::vector<LevelMeta> levels_;
    std::shared_ptr<utils::HttpClient> client_;
    bool sharded_ = false;
    std::array<int, 3> shardShape_ = {0, 0, 0};
    std::array<int, 3> chunksPerShard_ = {1, 1, 1};

    struct ShardCacheEntry {
        std::shared_ptr<std::vector<uint8_t>> bytes;
        // Parsed shard index cached alongside the bytes so fetchFromShard
        // doesn't reparse the 16-byte-per-entry table on every inner-chunk
        // extraction. Populated lazily on first fetchFromShard call.
        std::shared_ptr<utils::detail::ShardIndex> index;
    };
    struct ShardLruNode {
        std::string url;
        ShardCacheEntry entry;
    };
    std::mutex shardCacheMutex_;
    std::list<ShardLruNode> shardCacheLru_;
    std::unordered_map<std::string, std::list<ShardLruNode>::iterator> shardCacheMap_;
    std::size_t shardCacheBytes_ = 0;

    std::atomic<bool> transientError_{false};
};

}  // namespace vc::cache
