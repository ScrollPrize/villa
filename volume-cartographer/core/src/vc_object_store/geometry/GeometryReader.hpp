#pragma once
#include "internal/Database.hpp"
#include "geometry/GeometryInternal.hpp"
#include <functional>
#include <list>
#include <iterator>

namespace vc::objects::detail {

// Bound to one connection. The read pool may retain it across unchanged read
// snapshots; writes and internal integrity/publication paths use fresh readers.
class GeometryReader {
public:
    GeometryReader(const Txn& txn, const std::string& revision, bool cacheChunks = true);
    GeometryReader(sqlite3* db, const std::string& revision, bool cacheChunks = true);
    GeometryReader(const GeometryReader&) = delete;
    GeometryReader& operator=(const GeometryReader&) = delete;
    const nlohmann::json& manifest() const { return manifest_; }
    size_t retainedCharge() const { return metadataCharge_ + cachedBytes_ + chunks_.size() * 256; }
    std::vector<uint8_t> read(const std::string& name, int64_t offset, std::optional<int64_t> count = std::nullopt);

    std::vector<uint8_t> rectangle(const std::string& name, int64_t row, int64_t column,
                                   int64_t rows, int64_t columns);

private:
    using ChunkKey = std::pair<std::string, int64_t>;
    struct CachedChunk {
        std::vector<uint8_t> bytes;
        std::list<ChunkKey>::iterator recency;
    };
    void visitChunk(const std::string& name, int64_t index,
                    const std::function<void(std::span<const uint8_t>)>& visit);

    sqlite3* db_;
    std::array<unsigned char, 16> revision_;
    bool cacheChunks_;
    nlohmann::json manifest_;
    std::map<std::string, GeometryLayout> layouts_;
    std::map<ChunkKey, CachedChunk> chunks_;
    std::list<ChunkKey> recency_;
    size_t cachedBytes_ = 0;
    size_t metadataCharge_ = 0;
};
}
