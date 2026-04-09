#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ChunkKey.hpp"
#include "HttpMetadataFetcher.hpp"  // HttpAuth (= utils::AwsAuth)

namespace utils { class HttpClient; }

namespace vc::cache {

// Abstract interface for fetching raw compressed chunk bytes from a data source.
class ChunkSource {
public:
    virtual ~ChunkSource() = default;
    [[nodiscard]] virtual std::vector<uint8_t> fetch(const ChunkKey& key) = 0;
    [[nodiscard]] virtual int numLevels() const noexcept = 0;
    [[nodiscard]] virtual std::array<int, 3> chunkShape(int level) const noexcept = 0;
    [[nodiscard]] virtual std::array<int, 3> levelShape(int level) const noexcept = 0;
};

// Reads compressed chunks from a local zarr v2 directory.
class FileSystemChunkSource : public ChunkSource {
public:
    struct LevelMeta {
        std::array<int, 3> shape;
        std::array<int, 3> chunkShape;
    };

    explicit FileSystemChunkSource(
        const std::filesystem::path& zarrRoot,
        const std::string& delimiter = ".");

    FileSystemChunkSource(
        const std::filesystem::path& zarrRoot,
        const std::string& delimiter,
        std::vector<LevelMeta> levels);

    [[nodiscard]] std::vector<uint8_t> fetch(const ChunkKey& key) override;
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
class HttpChunkSource : public ChunkSource {
public:
    using LevelMeta = FileSystemChunkSource::LevelMeta;

    HttpChunkSource(
        const std::string& baseUrl,
        const std::string& delimiter,
        std::vector<LevelMeta> levels,
        HttpAuth auth = {});

    ~HttpChunkSource() override;

    void setShardConfig(const ShardConfig& config);

    [[nodiscard]] std::vector<uint8_t> fetch(const ChunkKey& key) override;
    [[nodiscard]] int numLevels() const noexcept override;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept override;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept override;

    // Download an entire shard as raw bytes (for bulk prefetch to disk).
    [[nodiscard]] std::vector<uint8_t> fetchWholeShard(int level, int sz, int sy, int sx);
    [[nodiscard]] bool isSharded() const noexcept { return sharded_; }
    [[nodiscard]] std::array<int, 3> shardsPerAxis(int level) const noexcept;
    [[nodiscard]] std::array<int, 3> chunksPerShard() const noexcept { return chunksPerShard_; }
    [[nodiscard]] std::array<int, 3> shardShape() const noexcept { return shardShape_; }

private:
    std::string chunkUrl(const ChunkKey& key) const;
    std::string shardUrl(const ChunkKey& key) const;
    int innerChunkIndex(const ChunkKey& key) const noexcept;
    int totalChunksPerShard() const noexcept;
    std::vector<uint8_t> fetchFromShard(const ChunkKey& key);
    std::vector<uint8_t> httpGet(const std::string& url);

    std::string baseUrl_;
    std::string delimiter_;
    std::vector<LevelMeta> levels_;
    std::shared_ptr<utils::HttpClient> client_;
    bool sharded_ = false;
    std::array<int, 3> shardShape_ = {0, 0, 0};
    std::array<int, 3> chunksPerShard_ = {1, 1, 1};

    std::mutex shardCacheMutex_;
    std::unordered_map<std::string, std::shared_ptr<std::vector<uint8_t>>> shardCache_;
};

}  // namespace vc::cache
