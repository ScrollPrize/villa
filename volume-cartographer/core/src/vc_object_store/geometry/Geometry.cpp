#include "internal/RecordSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/Dependencies.hpp"

namespace vc::objects {
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

void beginGeometryRead(sqlite3* db, StatementPool& pool, size_t budget)
{
    if (!budget) { pool.geometryCache.reset(); return; }
    if (!pool.geometryCache) pool.geometryCache = std::make_shared<GeometryReadCache>(budget);
    pool.geometryCache->begin(pragmaInt(db, "data_version"));
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
namespace {
GeometryRevision readGeometryRevision(const Statement& s)
{
    GeometryRevision r;
    r.uuid = formatUuid(s.blob16(0)); r.objectUuid = formatUuid(s.blob16(1));
    if (!s.isNull(2)) r.parentUuid = formatUuid(s.blob16(2));
    r.contentHash = s.text(3);
    const auto text = s.text(4);
    if (detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size())) != r.contentHash)
        throw std::runtime_error("geometry: corrupt revision manifest");
    const auto manifest = nlohmann::json::parse(text);
    r.kind = manifest.at("kind"); r.formatVersion = manifest.at("format_version");
    r.coordinateSystem = manifest.at("coordinate_system"); r.attributes = manifest.at("attributes").dump();
    r.objectGeneration = s.i64(5); r.createdBy = s.text(6); r.createdAt = s.text(7); r.createdHost = s.text(8);
    for (const auto& [name, a] : manifest.at("arrays").items())
        r.arrays.push_back({name, a.at("dtype"), a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length")});
    return r;
}
constexpr const char* geometryColumns = "uuid,object_uuid,parent_uuid,content_hash,manifest,object_generation,created_by,created_at,created_host";
}

std::optional<GeometryRevision> ObjectStore::geometryRevision(const Txn& txn, const std::string& revisionUuid) const
{
    (void)connection(txn);
    Statement s(static_cast<sqlite3*>(connection(txn)), std::string("SELECT ") + geometryColumns + " FROM geometry_revisions WHERE uuid=?");
    s.bindBlob(1, parseUuid(revisionUuid));
    return s.row() ? std::optional(readGeometryRevision(s)) : std::nullopt;
}

std::optional<GeometryRevision> ObjectStore::currentGeometry(const Txn& txn, const std::string& objectUuid) const
{
    (void)connection(txn);
    const auto head = geometryHead(static_cast<sqlite3*>(connection(txn)), objectUuid);
    return head ? geometryRevision(txn, *head) : std::nullopt;
}

std::vector<GeometryRevision> ObjectStore::geometryHistory(const Txn& txn, const std::string& objectUuid) const
{
    (void)connection(txn);
    Statement s(static_cast<sqlite3*>(connection(txn)), std::string("SELECT ") + geometryColumns + " FROM geometry_revisions WHERE object_uuid=? ORDER BY object_generation");
    s.bindBlob(1, parseUuid(objectUuid));
    std::vector<GeometryRevision> result;
    while (s.row()) result.push_back(readGeometryRevision(s));
    return result;
}

namespace {
int64_t revisionPatchTileSize(const Txn& txn, const GeometryRevision& revision)
{
    if (revision.kind != "patch") return 0;
    const detail::GeometryReader reader(txn, revision.uuid);
    const auto& points = reader.manifest().at("arrays").at("points");
    return points.contains("storage") ? points.at("storage").at("tile_size").get<int64_t>() : 0;
}
std::vector<EdgeRecord> affectedEdges(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
    const std::optional<GeometryRevision>& current, const detail::DependencyGeometry& after)
{
    const auto edges = store.edgesOf(txn, objectUuid);
    if (edges.empty()) return {};
    const auto before = current ? std::optional(detail::dependencyGeometry(store, txn, *current)) : std::nullopt;
    std::vector<EdgeRecord> result;
    for (const auto& edge : edges) {
        const auto changed = [&](const Anchor& anchor) {
            return !before || detail::dependenciesChanged(anchor, *before, after);
        };
        if ((edge.aUuid == objectUuid && changed(edge.aAnchor)) ||
            (edge.bUuid == objectUuid && changed(edge.bAnchor))) result.push_back(edge);
    }
    return result;
}
}

std::vector<EdgeRecord> ObjectStore::affectedGeometryEdges(const Txn& txn, const std::string& objectUuid,
                                                         const GeometryData& data) const
{
    detail::DependencyScope dependencyScope;
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || object->kind != data.kind) throw std::invalid_argument("geometry: object missing or kind mismatch");
    const auto current = currentGeometry(txn, objectUuid);
    const auto manifest = detail::geometryManifest(data, current ? revisionPatchTileSize(txn, *current) : GeometryStorageOptions{}.patchTileSize).dump();
    if (current && current->contentHash == detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size())))
        return {};
    return affectedEdges(*this, txn, object->uuid, current, detail::dependencyGeometry(data));
}

std::string ObjectStore::putGeometry(WriteTxn& txn, const std::string& objectUuid, const GeometryData& data,
    const std::optional<std::string>& expectedRevision, const std::vector<GeometryEdgeEdit>& edgeEdits,
    const GeometryStorageOptions& storage)
{
    detail::DependencyScope dependencyScope;
    (void)connection(txn);
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || object->kind != data.kind) throw std::invalid_argument("geometry: object missing or kind mismatch");
    const auto current = currentGeometry(txn, objectUuid);
    const auto expected = expectedRevision ? std::optional(formatUuid(parseUuid(*expectedRevision))) : std::nullopt;
    if ((current ? std::optional(current->uuid) : std::nullopt) != expected)
        throw std::runtime_error("geometry: current revision changed; reload before committing");
    const auto manifest = detail::geometryManifest(data, storage.patchTileSize);
    const auto manifestText = manifest.dump();
    const auto contentHash = detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(manifestText.data()), manifestText.size()));
    bool unchanged = current && current->contentHash == contentHash;
    if (current && !unchanged && data.kind == "patch") {
        const auto oldTile = revisionPatchTileSize(txn, *current);
        if (oldTile != storage.patchTileSize) {
            const auto oldManifest = detail::geometryManifest(data, oldTile).dump();
            unchanged = current->contentHash == detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(oldManifest.data()), oldManifest.size()));
        }
    }
    if (unchanged) {
        if (!edgeEdits.empty()) throw std::invalid_argument("geometry is unchanged; edit relationships directly");
        return current->uuid;
    }
    const auto after = detail::dependencyGeometry(data);
    const auto affected = affectedEdges(*this, txn, object->uuid, current, after);
    std::map<std::string, GeometryEdgeEdit> decisions;
    for (const auto& edit : edgeEdits) {
        const auto uuid = formatUuid(parseUuid(edit.edgeUuid));
        if (!decisions.emplace(uuid, edit).second) throw std::invalid_argument("duplicate geometry edge decision");
    }
    for (const auto& edge : affected) {
        const auto found = decisions.find(edge.uuid);
        if (found == decisions.end())
            throw std::runtime_error("geometry edit requires explicit retain/update/remove for affected edge " + edge.uuid);
        const auto& edit = found->second;
        if (edit.expectedGeneration != edge.generation)
            throw std::runtime_error("geometry edit: edge generation changed; reload decisions");
        auto canonical = [](const std::optional<std::string>& id) -> std::optional<std::string> {
            return id ? std::optional(formatUuid(parseUuid(*id))) : std::nullopt;
        };
        if (canonical(edit.aRevision) != geometryHead(db, edge.aUuid) || canonical(edit.bRevision) != geometryHead(db, edge.bUuid))
            throw std::runtime_error("geometry edit: endpoint revision changed; reload decisions");
        if (edit.action != "retain" && edit.action != "update" && edit.action != "remove")
            throw std::invalid_argument("geometry edge action must be retain, update, or remove");
        if (edit.action == "remove") continue;
        auto validate = [&](const std::string& endpoint, const Anchor& anchor) {
            if (endpoint == object->uuid) detail::validateDependencies(anchor, after);
            else if (const auto revision = currentGeometry(txn, endpoint))
                detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
            else if (!anchor.wholeGeometry || !anchor.points.empty())
                throw std::runtime_error("geometry edit: relationship endpoint has no geometry for its point references");
        };
        validate(edge.aUuid, edit.action == "retain" ? edge.aAnchor : edit.aAnchor);
        validate(edge.bUuid, edit.action == "retain" ? edge.bAnchor : edit.bAnchor);
    }
    if (decisions.size() != affected.size())
        throw std::invalid_argument("geometry decisions include an edge not affected by this edit");

    exec(db, "SAVEPOINT geometry_publish;");
    try {
        const auto revisionUuid = util::newUuid();
        Statement revision(db, "INSERT INTO geometry_revisions VALUES(?,?,?,?,?,?,?,?,?)");
        revision.bindBlob(1, parseUuid(revisionUuid)); revision.bindBlob(2, parseUuid(objectUuid));
        if (current) revision.bindBlob(3, parseUuid(current->uuid)); else revision.bindNull(3);
        revision.bind(4, contentHash); revision.bind(5, manifestText); revision.bind(6, object->generation + 1);
        revision.bind(7, author()); revision.bind(8, nowIsoUtc()); revision.bind(9, hostname()); revision.stepDone();
        Statement insert(db, "INSERT INTO geometry_chunks(hash,data) VALUES(?,?) ON CONFLICT(hash) DO NOTHING");
        Statement verify(db, "SELECT data FROM geometry_chunks WHERE hash=?");
        Statement reference(db, "INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)");
        for (const auto& array : data.arrays) {
            const auto layout = detail::geometryLayout(manifest.at("arrays").at(array.name));
            detail::forEachGeometryChunk(array, layout.tile, [&](int64_t index, std::span<const uint8_t> bytes) {
                const std::string hash = manifest.at("arrays").at(array.name).at("chunks").at(index);
                insert.reset(); insert.bind(1, hash);
                check(sqlite3_bind_blob64(insert.value, 2, bytes.data(), bytes.size(), SQLITE_STATIC), db, "bind geometry chunk");
                insert.stepDone();
                // Never silently reuse corrupt data, even on a hash collision.
                verify.reset(); verify.bind(1, hash);
                if (!verify.row() || sqlite3_column_bytes(verify.value, 0) != static_cast<int>(bytes.size()) ||
                    std::memcmp(sqlite3_column_blob(verify.value, 0), bytes.data(), bytes.size()) != 0)
                    throw std::runtime_error("geometry: corrupt or colliding payload hash");
                reference.reset(); reference.bindBlob(1, parseUuid(revisionUuid)); reference.bind(2, array.name);
                reference.bind(3, static_cast<int64_t>(index)); reference.bind(4, hash); reference.stepDone();
            });
        }

        Statement head(db, "INSERT INTO geometry_heads VALUES(?,?) ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid");
        head.bindBlob(1, parseUuid(objectUuid)); head.bindBlob(2, parseUuid(revisionUuid)); head.stepDone();
        const auto id = *objectIdByUuid(db, parseUuid(objectUuid));
        detail::indexGeometryBounds(db, id, revisionUuid, data);
        stampContentChange(db, author(), hostname(), "objects", id, parseUuid(objectUuid), "geometry");
        Statement update(db, "UPDATE objects SET storage_kind='sqlite',data_generation=generation,data_checksum=? WHERE id=?");
        update.bind(1, contentHash); update.bind(2, id); update.stepDone();
        for (const auto& edge : affected) {
            const auto& edit = decisions.at(edge.uuid);
            if (edit.action == "remove") { removeEdge(txn, edge.uuid); continue; }
            EdgeGeometryRepair repair;
            repair.edgeUuid = edge.uuid; repair.expectedGeneration = edge.generation;
            repair.aRevision = geometryHead(db, edge.aUuid); repair.bRevision = geometryHead(db, edge.bUuid);
            repair.aAnchor = edit.action == "retain" ? edge.aAnchor : edit.aAnchor;
            repair.bAnchor = edit.action == "retain" ? edge.bAnchor : edit.bAnchor;
            repairEdgeGeometry(txn, repair);
        }
        exec(db, "RELEASE geometry_publish;");
        return revisionUuid;
    } catch (...) {
        exec(db, "ROLLBACK TO geometry_publish; RELEASE geometry_publish;");
        throw;
    }
}

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
