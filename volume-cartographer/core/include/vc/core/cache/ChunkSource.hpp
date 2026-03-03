#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
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

// Fetches compressed chunks from an HTTP/HTTPS zarr store.
// URL layout: <baseUrl>/<level>/<iz>.<iy>.<ix>
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

    [[nodiscard]] std::vector<uint8_t> fetch(const ChunkKey& key) override;
    [[nodiscard]] int numLevels() const override;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const override;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const override;

private:
    std::string chunkUrl(const ChunkKey& key) const;

    std::string baseUrl_;
    std::string delimiter_;
    std::vector<LevelMeta> levels_;
    HttpAuth auth_;
};

}  // namespace vc::cache
