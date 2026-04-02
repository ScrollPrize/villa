#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ChunkKey.hpp"
#include "HttpMetadataFetcher.hpp"  // HttpAuth

namespace vc::cache {

// Abstract interface for fetching raw compressed chunk bytes from a data source.
// Implementations handle the details of file paths, network protocols, etc.
// All methods are blocking; async behavior is handled by the IOPool layer.
class ChunkSource {
public:
    virtual ~ChunkSource() = default;

    // Fetch raw compressed chunk bytes. Returns empty vector if not found.
    [[nodiscard]] virtual std::vector<uint8_t> fetch(const ChunkKey& key) = 0;

    // Number of pyramid levels available.
    [[nodiscard]] virtual int numLevels() const = 0;

    // Chunk shape at a given level, in {z, y, x} order.
    [[nodiscard]] virtual std::array<int, 3> chunkShape(int level) const = 0;

    // Full dataset shape at a given level, in {z, y, x} order.
    [[nodiscard]] virtual std::array<int, 3> levelShape(int level) const = 0;
};

// Reads compressed chunks from a local zarr v2 directory.
// Directory layout: <root>/<level>/<iz>.<iy>.<ix>
// Reads .zarray metadata per level for shape/chunk info.
class FileSystemChunkSource : public ChunkSource {
public:
    struct LevelMeta {
        std::array<int, 3> shape;       // dataset shape {z, y, x}
        std::array<int, 3> chunkShape;  // chunk dimensions {z, y, x}
    };

    // Construct from zarr root directory. Auto-discovers levels from subdirs.
    // delimiter: chunk index separator ("." for zarr, "/" for N5)
    explicit FileSystemChunkSource(
        const std::filesystem::path& zarrRoot,
        const std::string& delimiter = ".");

    // Construct with pre-supplied metadata (avoids reading .zarray files).
    FileSystemChunkSource(
        const std::filesystem::path& zarrRoot,
        const std::string& delimiter,
        std::vector<LevelMeta> levels);

    [[nodiscard]] std::vector<uint8_t> fetch(const ChunkKey& key) override;
    [[nodiscard]] int numLevels() const override;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const override;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const override;

private:
    std::filesystem::path chunkPath(const ChunkKey& key) const;
    void discoverLevels();

    std::filesystem::path root_;
    std::string delimiter_;
    std::vector<LevelMeta> levels_;
};

// Shard configuration for zarr v3 sharded storage (forward declaration).
struct ShardConfig;

// Fetches compressed chunks from an HTTP/HTTPS zarr store.
// URL layout: <baseUrl>/<level>/<iz>.<iy>.<ix>  (zarr v2)
//         or: <baseUrl>/<level>/c/<sz>/<sy>/<sx> (zarr v3 sharded)
// Requires libcurl at link time (gated behind VC_USE_CURL).
class HttpChunkSource : public ChunkSource {
public:
    using LevelMeta = FileSystemChunkSource::LevelMeta;

    // baseUrl: root URL of the zarr store (no trailing slash)
    // delimiter: chunk index separator
    // levels: pre-supplied metadata (HTTP source doesn't auto-discover)
    HttpChunkSource(
        const std::string& baseUrl,
        const std::string& delimiter,
        std::vector<LevelMeta> levels,
        HttpAuth auth = {});

    ~HttpChunkSource() override;

    // Enable zarr v3 sharded storage. When set, chunk URLs use
    // {level}/c/{sz}/{sy}/{sx} format with shard coordinate mapping.
    void setShardConfig(const ShardConfig& config);

    [[nodiscard]] std::vector<uint8_t> fetch(const ChunkKey& key) override;
    [[nodiscard]] int numLevels() const override;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const override;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const override;

    // Apply pre-computed auth to a CURL handle. outHeaders is appended to
    // (caller must free after curl_easy_perform).
    void applyCachedAuth(CURL* curl, struct curl_slist*& outHeaders) const;

private:
    std::string chunkUrl(const ChunkKey& key) const;
    std::string shardUrl(const ChunkKey& key) const;
    int innerChunkIndex(const ChunkKey& key) const;
    int totalChunksPerShard() const;

    // Fetch entire shard file, cache it, extract chunk by inner index.
    std::vector<uint8_t> fetchFromShard(const ChunkKey& key);

    // Pre-computed auth strings (built once in constructor, applied per request).
    void buildCachedAuth();
    std::string cachedSigv4_;
    std::string cachedUserpwd_;
    std::string cachedTokenHeader_;  // empty if no session token

    std::string baseUrl_;
    std::string delimiter_;
    std::vector<LevelMeta> levels_;
    HttpAuth auth_;
    bool sharded_ = false;
    std::array<int, 3> shardShape_ = {0, 0, 0};   // shard size in voxels
    std::array<int, 3> chunksPerShard_ = {1, 1, 1}; // chunks per shard dimension

    // Whole-shard cache: shard URL -> raw bytes.
    std::mutex shardCacheMutex_;
    std::unordered_map<std::string, std::shared_ptr<std::vector<uint8_t>>> shardCache_;
};

}  // namespace vc::cache
