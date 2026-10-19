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
    manifest_ = verifiedGeometryManifest(text, row.text(0));
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
    verifyGeometryChunk(bytes, chunk.text(1), expectedHash, expectedSize);
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
    if (offset == 0 && size == length) {
        for (int64_t index = 0; index < layout.chunkCount(); ++index)
            visitChunk(name, index, [&](auto bytes) { copyGeometryChunk(layout, index, bytes, result.data()); });
        return result;
    }
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

std::vector<std::vector<uint8_t>> GeometryReader::ranges(const std::string& name,
    const std::vector<std::array<int64_t, 2>>& ranges)
{
    const auto found = layouts_.find(name);
    if (found == layouts_.end()) throw std::invalid_argument("geometry: array not found: " + name);
    const auto& layout = found->second;
    struct Copy { size_t output; int64_t source, destination, count; };
    std::map<int64_t, std::vector<Copy>> plan;
    std::vector<std::vector<uint8_t>> result;
    result.reserve(ranges.size());
    for (const auto& range : ranges) {
        const auto [offset, size] = range;
        if (offset < 0 || offset > layout.length || size < 0 || size > layout.length - offset)
            throw std::out_of_range("geometry: invalid byte range");
        const auto output = result.size();
        result.emplace_back(static_cast<size_t>(size));
        if (!size) continue;
        if (!layout.tile) {
            for (auto chunk = offset / geometryChunkSize; chunk <= (offset + size - 1) / geometryChunkSize; ++chunk) {
                const int64_t base = chunk * geometryChunkSize;
                const auto begin = std::max(base, offset);
                const auto end = std::min(base + layout.chunkBytes(chunk), offset + size);
                plan[chunk].push_back({output, begin - base, begin - offset, end - begin});
            }
        } else {
            // Split logical row-major bytes into physical tile spans, including
            // partial XYZ elements and ranges crossing row/tile boundaries.
            const auto rowBytes = layout.cols * 12;
            for (auto row = offset / rowBytes; row <= (offset + size - 1) / rowBytes; ++row) {
                const auto first = std::max(offset, row * rowBytes);
                const auto last = std::min(offset + size, (row + 1) * rowBytes);
                for (auto col = (first % rowBytes) / (layout.tile * 12);
                     col <= (last - 1 - row * rowBytes) / (layout.tile * 12); ++col) {
                    const auto chunk = row / layout.tile * layout.tileColumns() + col;
                    const auto [y, x, h, w] = layout.rectangle(chunk);
                    const auto base = row * rowBytes + x * 12;
                    const auto begin = std::max(base, first), end = std::min(base + w * 12, last);
                    plan[chunk].push_back({output, (row - y) * w * 12 + begin - base,
                                          begin - offset, end - begin});
                }
            }
        }
    }
    // Each touched chunk is loaded and verified once, independently of caller
    // order or cache capacity. Duplicates/overlaps scatter to every destination.
    for (const auto& [chunk, copies] : plan)
        visitChunk(name, chunk, [&](auto bytes) {
            for (const auto& copy : copies)
                std::memcpy(result[copy.output].data() + copy.destination,
                            bytes.data() + copy.source, copy.count);
        });
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
    if (byteCount && *byteCount <= detail::geometryChunkSize)
        return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
            return reader.read(arrayName, byteOffset, byteCount);
        });
    return detail::GeometryReader(txn, revisionUuid, false).read(arrayName, byteOffset, byteCount);
}

PackedGeometryArray ObjectStore::readGeometryArrays(const Txn& txn,
    const std::vector<std::string>& revisions, const std::string& name) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    PackedGeometryArray result;
    struct ArrayPlan {
        GeometryLayout layout;
        std::vector<std::string> hashes;
        size_t offset;
    };
    std::vector<ArrayPlan> plan;
    plan.reserve(revisions.size());
    size_t totalBytes = 0;
    // Bound parameters/manifest parsing; retain only layouts and expected hashes
    // before allocating the final packed payload once.
    constexpr size_t window = 256;
    auto request = [](size_t count) {
        std::string sql = "WITH requested(pos,uuid) AS (VALUES ";
        for (size_t i = 0; i < count; ++i)
            sql += (i ? "," : "") + std::string("(") + std::to_string(i) + ",?" + std::to_string(i+1) + ")";
        return sql + ") ";
    };
    auto bind = [&](Statement& query, size_t begin, size_t count) {
        for (size_t i = 0; i < count; ++i) query.bindBlob(int(i+1), parseUuid(revisions[begin+i]));
    };
    for (size_t begin = 0; begin < revisions.size(); begin += window) {
        const auto count = std::min(window, revisions.size() - begin);
        Statement manifests(db, request(count) + "SELECT q.pos,r.content_hash,r.manifest FROM requested q "
            "LEFT JOIN geometry_revisions r ON r.uuid=q.uuid ORDER BY q.pos");
        bind(manifests, begin, count);
        while (manifests.row()) {
            if (manifests.isNull(1)) throw std::invalid_argument("geometry: revision not found");
            auto manifest = verifiedGeometryManifest(manifests.text(2), manifests.text(1));
            const auto& descriptors = manifest.at("arrays");
            if (!descriptors.contains(name)) throw std::invalid_argument("geometry: array not found: " + name);
            GeometryLayout layout;
            for (const auto& [key, array] : descriptors.items()) {
                const auto validated = geometryLayout(array);
                if (key == name) layout = validated;
            }
            const auto& array = descriptors.at(name);
            const auto shape = array.at("shape").get<std::vector<int64_t>>();
            const auto dtype = array.at("dtype").get<std::string>();
            if (result.offsets.size() == 1) {
                result.dtype = dtype; result.shape = shape; result.shape[0] = 0;
            }
            if (dtype != result.dtype || shape.size() != result.shape.size() ||
                !std::equal(shape.begin()+1, shape.end(), result.shape.begin()+1))
                throw std::invalid_argument("geometry: packed arrays require matching dtype and trailing dimensions");
            if (shape[0] > std::numeric_limits<int64_t>::max() - result.shape[0] ||
                uint64_t(layout.length) > result.data.max_size() - totalBytes)
                throw std::overflow_error("geometry: packed array size overflow");
            result.shape[0] += shape[0];
            result.offsets.push_back(uint64_t(result.shape[0]));
            plan.push_back({layout, array.at("chunks").get<std::vector<std::string>>(), totalBytes});
            totalBytes += size_t(layout.length);
        }
        if (plan.size() != begin + count) throw std::runtime_error("geometry: incomplete revision selection");
    }
    result.data.resize(totalBytes);
    for (size_t begin = 0; begin < revisions.size(); begin += window) {
        const auto count = std::min(window, revisions.size() - begin);
        // Keep requested revisions outermost: otherwise SQLite can scan the
        // entire reference table for every window, even with its primary key.
        Statement chunks(db, request(count) + "SELECT q.pos,r.chunk_index,r.hash,c.data FROM requested q "
            "CROSS JOIN geometry_revision_chunks r ON r.revision_uuid=q.uuid AND r.array_name=?" + std::to_string(count+1) +
            " LEFT JOIN geometry_chunks c ON c.hash=r.hash");
        bind(chunks, begin, count); chunks.bind(int(count+1), name);
        std::vector<int64_t> seen(count, 0);
        while (chunks.row()) {
            const auto row = size_t(chunks.i64(0));
            const auto index = chunks.i64(1);
            ++seen.at(row); // unique (revision,array,index), independent of SQL row order
            if (chunks.isNull(3)) throw std::runtime_error("geometry: missing payload chunk");
            const auto bytes = std::span(static_cast<const uint8_t*>(sqlite3_column_blob(chunks.value, 3)),
                                        size_t(sqlite3_column_bytes(chunks.value, 3)));
            const auto& array = plan.at(begin + row);
            const auto expectedSize = array.layout.chunkBytes(index);
            verifyGeometryChunk(bytes, chunks.text(2), array.hashes.at(index), expectedSize);
            copyGeometryChunk(array.layout, index, bytes, result.data.data() + array.offset);
        }
        for (size_t i = 0; i < count; ++i)
            if (seen[i] != plan[begin+i].layout.chunkCount()) throw std::runtime_error("geometry: missing payload chunk");
    }
    return result;
}

std::vector<std::vector<uint8_t>> ObjectStore::readGeometryRanges(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, const std::vector<std::array<int64_t, 2>>& ranges) const
{
    (void)connection(txn);
    return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
        return reader.ranges(arrayName, ranges);
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
