#pragma once
#include "internal/Database.hpp"
#include "geometry/GeometryInternal.hpp"
#include <functional>
#include <optional>
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
    std::vector<uint8_t> chunk(const std::string& name, int64_t index);
    size_t retainedCharge() const { return metadataCharge_ + cachedBytes_ + chunks_.size() * 256; }
    std::vector<uint8_t> read(const std::string& name, int64_t offset, std::optional<int64_t> count = std::nullopt);
    std::vector<std::vector<uint8_t>> ranges(const std::string& name,
        const std::vector<std::array<int64_t, 2>>& ranges);

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
    std::string baseRevision_, materializedObject_;
    std::vector<std::pair<std::string, nlohmann::json>> deltas_;
    std::unique_ptr<GeometryReader> reverseSource_;
};
// A surface rectangle's XYZ and whether each vertex is valid by the store's
// rule (geometry::validSurfacePoint and the valid channel), row-major.
struct SurfaceTile {
    std::vector<std::array<float, 3>> points;
    std::vector<uint8_t> valid;
    // Bounds of the valid vertices of each band of bandCells cell
    // rows (its vertex rows and the next), absent for a band without any:
    // a plane or point that cannot reach a band skips its cells.
    static constexpr int64_t bandCells = 8;
    std::vector<std::optional<std::array<double, 6>>> bands;
};
// Fill tile.bands for a tile `columns` vertices wide.
void boundSurfaceTileBands(SurfaceTile& tile, int64_t columns);
SurfaceTile surfaceTile(GeometryReader&, int64_t row, int64_t column, int64_t rows, int64_t columns);
// The same through `cache` when there is one: decoded once for every reader.
std::shared_ptr<const SurfaceTile> surfaceTile(SurfaceTileCache* cache, const std::string& revision, GeometryReader&,
                                               int64_t row, int64_t column, int64_t rows, int64_t columns);

// The read connection's retained reader for `revision` (verified chunks kept
// across reads while the store is unchanged), or a new one. Handed back when
// the read ends normally; dropped when it throws, so partially verified data
// never poisons a later read.
class RetainedGeometryReader {
public:
    RetainedGeometryReader(sqlite3*, std::string revision);
    ~RetainedGeometryReader();
    RetainedGeometryReader(const RetainedGeometryReader&) = delete;
    RetainedGeometryReader& operator=(const RetainedGeometryReader&) = delete;
    GeometryReader& operator*() const { return *reader_; }
    GeometryReader* operator->() const { return reader_.get(); }
private:
    sqlite3* db_;
    std::string revision_;
    std::shared_ptr<GeometryReader> reader_;
    int exceptions_;
};
}
