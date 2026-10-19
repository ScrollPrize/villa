#include "geometry/GeometryReader.hpp"

namespace vc::objects {
using namespace detail;
namespace detail {
// One instance per leased reader connection: no shared mutable geometry or
// cross-snapshot lookups between concurrent readers. The SQLite data_version is
// connection-local and changes on commits by any *other* connection, including
// this store's writer. Writer connections never receive this cache.
class GeometryReadCache {
public:
    explicit GeometryReadCache(size_t budget) : budget_(budget) {}
    void begin(int version) {
        if (version != version_) {
            readers_.clear(); recency_.clear(); charge_ = 0; version_ = version;
        }
    }
    std::shared_ptr<GeometryReader> take(const std::string& revision) {
        const auto found = readers_.find(revision);
        if (found == readers_.end()) return {};
        auto reader = std::move(found->second.reader);
        charge_ -= found->second.charge;
        recency_.erase(found->second.recency);
        readers_.erase(found);
        return reader;
    }
    void retain(const std::string& revision, std::shared_ptr<GeometryReader> reader) {
        const auto charge = reader->retainedCharge();
        if (charge > budget_) return;
        // Also bound revision/node overhead, including empty geometries.
        while (charge_ > budget_ - charge || readers_.size() >= 64) {
            const auto old = readers_.find(recency_.front());
            charge_ -= old->second.charge;
            readers_.erase(old); recency_.pop_front();
        }
        recency_.push_back(revision);
        try {
            readers_.emplace(revision, Entry{std::move(reader), charge, std::prev(recency_.end())});
        } catch (...) { recency_.pop_back(); throw; }
        charge_ += charge;
    }
private:
    struct Entry {
        std::shared_ptr<GeometryReader> reader;
        size_t charge;
        std::list<std::string>::iterator recency;
    };
    size_t budget_, charge_ = 0;
    int version_ = -1;
    std::map<std::string, Entry> readers_;
    std::list<std::string> recency_;
};

void beginReadCaches(sqlite3* db, StatementPool& pool, size_t geometryBudget, size_t spatialBudget)
{
    const int version = geometryBudget || spatialBudget ? pragmaInt(db, "data_version") : 0;
    if (!geometryBudget) pool.geometryCache.reset();
    else {
        if (!pool.geometryCache) pool.geometryCache = std::make_shared<GeometryReadCache>(geometryBudget);
        pool.geometryCache->begin(version);
    }
    beginSpatialRead(pool, spatialBudget, version);
}

template<class F> auto withGeometryReader(const Txn& txn, const std::string& revision, F&& read)
{
    auto* db = ObjectStoreImplAccess::conn(txn);
    std::shared_ptr<GeometryReadCache> cache;
    if (const auto lease = activeStatementPools.find(db); lease != activeStatementPools.end())
        cache = lease->second->geometryCache;
    auto reader = cache ? cache->take(revision) : nullptr;
    if (!reader) reader = std::make_shared<GeometryReader>(db, revision);
    // A failed read discards its reader; partially verified data never poisons
    // a future attempt. Successful readers remain connection/snapshot scoped.
    auto result = read(*reader);
    if (cache) cache->retain(revision, std::move(reader));
    return result;
}

GeometryReader::GeometryReader(const Txn& txn, const std::string& revision, bool cacheChunks)
    : GeometryReader(ObjectStoreImplAccess::conn(txn), revision, cacheChunks) {}

GeometryReader::GeometryReader(sqlite3* db, const std::string& revision, bool cacheChunks)
    : db_(db), revision_(parseUuid(revision)), cacheChunks_(cacheChunks)
{
    Statement row(db_, "SELECT content_hash,manifest FROM geometry_revisions WHERE uuid=?");
    row.bindBlob(1, revision_);
    if (!row.row()) throw std::invalid_argument("geometry: revision not found");
    const auto text = row.text(1);
    // Charge JSON nodes/strings, layouts and cache bookkeeping conservatively,
    // in addition to the separately accounted retained payload bytes.
    metadataCharge_ = text.size() * 32 + 4096;
    if (geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size())) != row.text(0))
        throw std::runtime_error("geometry: corrupt revision manifest");
    manifest_ = nlohmann::json::parse(text);
    for (const auto& [name, array] : manifest_.at("arrays").items())
        layouts_.emplace(name, geometryLayout(array));
}

void GeometryReader::visitChunk(const std::string& name, int64_t index,
    const std::function<void(std::span<const uint8_t>)>& visit)
{
    const ChunkKey key{name, index};
    auto found = chunks_.find(key);
    if (found != chunks_.end()) {
        recency_.splice(recency_.end(), recency_, found->second.recency);
        visit(found->second.bytes);
        return;
    }
    const auto expectedSize = layouts_.at(name).chunkBytes(index);
    const auto& expectedHash = manifest_.at("arrays").at(name).at("chunks").at(index).get_ref<const std::string&>();
    Statement chunk(db_, "SELECT c.data,r.hash FROM geometry_revision_chunks r JOIN geometry_chunks c ON c.hash=r.hash "
                         "WHERE r.revision_uuid=? AND r.array_name=? AND r.chunk_index=?");
    chunk.bindBlob(1, revision_); chunk.bind(2, name); chunk.bind(3, index);
    if (!chunk.row()) throw std::runtime_error("geometry: missing payload chunk");
    const auto bytes = std::span(static_cast<const uint8_t*>(sqlite3_column_blob(chunk.value, 0)),
                                static_cast<size_t>(sqlite3_column_bytes(chunk.value, 0)));
    if (static_cast<int64_t>(bytes.size()) != expectedSize || chunk.text(1) != expectedHash || geometryHash(bytes) != expectedHash)
        throw std::runtime_error("geometry: corrupt payload chunk");
    if (!cacheChunks_) { visit(bytes); return; }
    // Bound bytes, not chunk count: tiled chunks are smaller than linear ones.
    constexpr size_t budget = 8 * geometryChunkSize;
    while (cachedBytes_ + bytes.size() > budget && !recency_.empty()) {
        const auto old = chunks_.find(recency_.front());
        cachedBytes_ -= old->second.bytes.size();
        chunks_.erase(old); recency_.pop_front();
    }
    recency_.push_back(key);
    auto inserted = chunks_.emplace(key, CachedChunk{std::vector<uint8_t>(bytes.begin(), bytes.end()), std::prev(recency_.end())}).first;
    cachedBytes_ += bytes.size();
    visit(inserted->second.bytes);
}

std::vector<uint8_t> GeometryReader::read(const std::string& name, int64_t offset, std::optional<int64_t> count)
{
    const auto found = layouts_.find(name);
    if (found == layouts_.end()) throw std::invalid_argument("geometry: array not found: " + name);
    const auto& layout = found->second;
    const auto length = layout.length;
    if (offset < 0 || offset > length) throw std::out_of_range("geometry: invalid byte offset");
    const int64_t size = count.value_or(length - offset);
    if (size < 0 || size > length - offset) throw std::out_of_range("geometry: invalid byte count");
    std::vector<uint8_t> result(static_cast<size_t>(size));
    if (!size) return result;
    if (!layout.tile) {
        for (int64_t index = offset / geometryChunkSize; index <= (offset + size - 1) / geometryChunkSize; ++index) {
            const int64_t base = index * geometryChunkSize;
            visitChunk(name, index, [&](auto bytes) {
                const auto begin = std::max(base, offset), end = std::min(base + int64_t(bytes.size()), offset + size);
                std::memcpy(result.data() + begin - offset, bytes.data() + begin - base, end - begin);
            });
        }
        return result;
    }
    const auto rowBytes = layout.cols * 12;
    const auto firstRow = offset / rowBytes, lastRow = (offset + size - 1) / rowBytes;
    const auto firstColTile = firstRow == lastRow ? offset % rowBytes / 12 / layout.tile : 0;
    const auto lastColTile = firstRow == lastRow ? (offset + size - 1) % rowBytes / 12 / layout.tile : layout.tileColumns() - 1;
    for (int64_t ty = firstRow / layout.tile; ty <= lastRow / layout.tile; ++ty)
        for (int64_t tx = firstColTile; tx <= lastColTile; ++tx) {
            const auto index = ty * layout.tileColumns() + tx;
            const auto [y, x, h, w] = layout.rectangle(index);
            const auto beginRow = std::max(y, firstRow), endRow = std::min(y + h, lastRow + 1);
            auto overlap = [&](int64_t row) {
                const auto base = (row * layout.cols + x) * 12;
                return std::array<int64_t, 2>{std::max(base, offset), std::min(base + w * 12, offset + size)};
            };
            bool needed = false;
            for (auto row = beginRow; row < endRow && !needed; ++row) {
                const auto [begin, end] = overlap(row); needed = begin < end;
            }
            if (!needed) continue;
            visitChunk(name, index, [&](auto bytes) {
                for (auto row = beginRow; row < endRow; ++row) {
                    const auto [begin, end] = overlap(row);
                    if (begin < end)
                        std::memcpy(result.data() + begin - offset,
                            bytes.data() + (row - y) * w * 12 + begin - (row * layout.cols + x) * 12, end - begin);
                }
            });
        }
    return result;
}

std::vector<uint8_t> GeometryReader::rectangle(const std::string& name, int64_t row, int64_t column,
                                             int64_t rows, int64_t columns)
{
    const auto found = layouts_.find(name);
    if (found == layouts_.end()) throw std::invalid_argument("geometry: array not found: " + name);
    const auto& array = manifest_.at("arrays").at(name);
    const auto shape = array.at("shape").get<std::vector<int64_t>>();
    if (manifest_.at("kind") != "patch" || array.at("dtype") != "<f4" || shape.size() != 3 || shape[2] != 3)
        throw std::invalid_argument("geometry: rectangle requires a patch float32 XYZ array");
    if (row < 0 || column < 0 || rows < 0 || columns < 0 || row > shape[0] || column > shape[1] ||
        rows > shape[0] - row || columns > shape[1] - column)
        throw std::out_of_range("geometry: invalid rectangle");
    std::vector<uint8_t> result(static_cast<size_t>(rows * columns * 12));
    if (!rows || !columns) return result;
    const auto& layout = found->second;
    if (!layout.tile) {
        for (int64_t r = 0; r < rows; ++r) {
            const auto bytes = read(name, ((row + r) * shape[1] + column) * 12, columns * 12);
            std::memcpy(result.data() + r * columns * 12, bytes.data(), bytes.size());
        }
        return result;
    }
    for (int64_t ty = row / layout.tile; ty <= (row + rows - 1) / layout.tile; ++ty)
        for (int64_t tx = column / layout.tile; tx <= (column + columns - 1) / layout.tile; ++tx) {
            const auto index = ty * layout.tileColumns() + tx;
            const auto [y, x, h, w] = layout.rectangle(index);
            const auto left = std::max(column, x), right = std::min(column + columns, x + w);
            visitChunk(name, index, [&](auto bytes) {
                for (auto r = std::max(row, y); r < std::min(row + rows, y + h); ++r)
                    std::memcpy(result.data() + ((r - row) * columns + left - column) * 12,
                        bytes.data() + ((r - y) * w + left - x) * 12, (right - left) * 12);
            });
        }
    return result;
}

} // namespace detail

std::vector<uint8_t> ObjectStore::readGeometryArray(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, int64_t byteOffset, const std::optional<int64_t>& byteCount) const
{
    (void)connection(txn);
    return detail::GeometryReader(txn, revisionUuid, false).read(arrayName, byteOffset, byteCount);
}

std::vector<std::vector<uint8_t>> ObjectStore::readGeometryRanges(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, const std::vector<std::array<int64_t, 2>>& ranges) const
{
    (void)connection(txn);
    return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
        std::vector<std::vector<uint8_t>> result;
        result.reserve(ranges.size());
        for (const auto& range : ranges) result.push_back(reader.read(arrayName, range[0], range[1]));
        return result;
    });
}

std::vector<uint8_t> ObjectStore::readGeometryRectangle(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, int64_t row, int64_t column, int64_t rows, int64_t columns) const
{
    (void)connection(txn);
    return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
        return reader.rectangle(arrayName, row, column, rows, columns);
    });
}

std::vector<std::vector<uint8_t>> ObjectStore::readGeometryRectangles(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, const std::vector<std::array<int64_t, 4>>& rectangles) const
{
    (void)connection(txn);
    return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
        std::vector<std::vector<uint8_t>> result;
        result.reserve(rectangles.size());
        for (const auto& r : rectangles) result.push_back(reader.rectangle(arrayName, r[0], r[1], r[2], r[3]));
        return result;
    });
}

} // namespace vc::objects
