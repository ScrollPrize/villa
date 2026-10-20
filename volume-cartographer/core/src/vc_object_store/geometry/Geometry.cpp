#include "internal/RecordSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/Dependencies.hpp"
#include "geometry/GeometryDelta.hpp"

namespace vc::objects {
using namespace detail;

namespace {
GeometryRevision readGeometryRevision(const Statement& s)
{
    GeometryRevision r;
    r.uuid = formatUuid(s.blob16(0)); r.objectUuid = formatUuid(s.blob16(1));
    if (!s.isNull(2)) r.parentUuid = formatUuid(s.blob16(2));
    r.contentHash = s.text(3);
    const auto text = s.text(4);
    const auto manifest = loadGeometryManifest(sqlite3_db_handle(s.value), r.uuid, text, r.contentHash);
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
std::map<std::string, GeometryEdgeEdit> validateEdgeEdits(const ObjectStore& store, const Txn& txn,
    sqlite3* db, const std::string& objectUuid, const detail::DependencyGeometry& after,
    const std::vector<EdgeRecord>& affected, const std::vector<GeometryEdgeEdit>& edgeEdits)
{
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
            if (endpoint == objectUuid) detail::validateDependencies(anchor, after);
            else if (const auto revision = store.currentGeometry(txn, endpoint))
                detail::validateDependencies(anchor, detail::dependencyGeometry(store, txn, *revision));
            else if (!anchor.wholeGeometry || !anchor.points.empty())
                throw std::runtime_error("geometry edit: relationship endpoint has no geometry for its point references");
        };
        validate(edge.aUuid, edit.action == "retain" ? edge.aAnchor : edit.aAnchor);
        validate(edge.bUuid, edit.action == "retain" ? edge.bAnchor : edit.bAnchor);
    }
    if (decisions.size() != affected.size())
        throw std::invalid_argument("geometry decisions include an edge not affected by this edit");

    return decisions;
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
    const auto decisions = validateEdgeEdits(*this, txn, db, object->uuid, after, affected, edgeEdits);

    Savepoint savepoint(db);
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

    detail::clearGeometryMaterialization(db, objectUuid);
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
    savepoint.release();
    return revisionUuid;
}

namespace {
struct SparseGeometryEdit {
    std::shared_ptr<GeometryReader> reader;
    nlohmann::json manifest;
    std::map<GeometryChunkKey, std::vector<uint8_t>> original, modified;
    std::vector<GeometryChunkChange> changes;
    DependencyGeometry after;
};
std::shared_ptr<SparseGeometryEdit> prepareEdit(const ObjectStore& store, const Txn& txn,
    const std::string& objectUuid, const std::string& expected, const std::vector<GeometryArrayEdit>& edits)
{
    const auto revision = store.currentGeometry(txn, objectUuid);
    if (!revision || revision->uuid != formatUuid(parseUuid(expected)))
        throw std::runtime_error("geometry: current revision changed; reload before committing");
    auto plan = std::make_shared<SparseGeometryEdit>();
    plan->reader = std::make_shared<GeometryReader>(txn, revision->uuid);
    plan->manifest = plan->reader->manifest();
    for (const auto& edit : edits) {
        if (!plan->manifest.at("arrays").contains(edit.array))
            throw std::invalid_argument("geometry edit: array not found: " + edit.array);
        const auto& array = plan->manifest.at("arrays").at(edit.array);
        const auto layout = geometryLayout(array);
        const auto scalar = geometryByteLength(array.at("dtype"), {1});
        if (edit.elementOffset < 0 || edit.elementOffset > layout.length / scalar || edit.bytes.size() % scalar ||
            edit.bytes.size() > uint64_t(layout.length - edit.elementOffset * scalar))
            throw std::out_of_range("geometry edit: invalid or unaligned element range");
        int64_t offset = edit.elementOffset * scalar;
        size_t consumed = 0;
        while (consumed < edit.bytes.size()) {
            int64_t index, within, available;
            if (!layout.tile) {
                index = offset / geometryChunkSize; within = offset % geometryChunkSize;
                available = layout.chunkBytes(index) - within;
            } else {
                const auto row = offset / (layout.cols * 12), col = offset % (layout.cols * 12) / 12;
                index = row / layout.tile * layout.tileColumns() + col / layout.tile;
                const auto [y, x, h, w] = layout.rectangle(index);
                within = (row - y) * w * 12 + offset % (layout.cols * 12) - x * 12;
                available = w * 12 - within % (w * 12);
            }
            const GeometryChunkKey key{edit.array, index};
            auto found = plan->modified.find(key);
            if (found == plan->modified.end()) {
                auto bytes = plan->reader->chunk(edit.array, index);
                plan->original.emplace(key, bytes);
                found = plan->modified.emplace(key, std::move(bytes)).first;
            }
            const auto count = std::min<size_t>(available, edit.bytes.size() - consumed);
            std::memcpy(found->second.data() + within, edit.bytes.data() + consumed, count);
            consumed += count; offset += count;
        }
    }
    for (auto it = plan->modified.begin(); it != plan->modified.end();) {
        const auto& old = plan->original.at(it->first);
        const auto& next = it->second;
        if (old == next) { it = plan->modified.erase(it); continue; }
        GeometryChunkChange change{it->first, geometryHash(old), geometryHash(next), {}};
        for (size_t i = 0; i < old.size();) {
            if (old[i] == next[i]) { ++i; continue; }
            const auto begin = i++;
            while (i < old.size() && old[i] != next[i]) ++i;
            change.changes.push_back({int64_t(begin), {old.begin()+begin, old.begin()+i}, {next.begin()+begin, next.begin()+i}});
        }
        plan->manifest["arrays"][it->first.first]["chunks"][it->first.second] = change.afterHash;
        plan->changes.push_back(std::move(change)); ++it;
    }
    plan->after = dependencyGeometry(store, txn, *revision);
    plan->after.pointIndex = std::make_shared<DependencyGeometry::PointIndex>();
    // Keep the dependency reader independent of the plan's lifetime; capture
    // only buffers and the original reader, avoiding a plan -> lambda cycle.
    auto reader = plan->reader;
    auto modified = std::make_shared<decltype(plan->modified)>(plan->modified);
    auto manifest = plan->manifest;
    plan->after.read = [reader, modified, manifest](const std::string& name, int64_t offset, int64_t count) {
        auto result = reader->read(name, offset, count);
        const auto layout = geometryLayout(manifest.at("arrays").at(name));
        for (const auto& [key, bytes] : *modified) {
            if (key.first != name) continue;
            auto copy = [&](int64_t logical, int64_t physical, int64_t length) {
                const auto begin = std::max(offset, logical), end = std::min(offset + count, logical + length);
                if (begin < end) std::memcpy(result.data() + begin - offset, bytes.data() + physical + begin - logical, end - begin);
            };
            if (!layout.tile) copy(key.second * geometryChunkSize, 0, bytes.size());
            else {
                const auto [y, x, h, w] = layout.rectangle(key.second);
                for (int64_t r = 0; r < h; ++r) copy(((y+r)*layout.cols+x)*12, r*w*12, w*12);
            }
        }
        return result;
    };
    return plan;
}
}

std::vector<EdgeRecord> ObjectStore::affectedGeometryEditEdges(const Txn& txn, const std::string& objectId,
    const std::vector<GeometryArrayEdit>& edits, const std::string& expected) const
{
    (void)connection(txn);
    const auto objectUuid = formatUuid(parseUuid(objectId));
    DependencyScope scope;
    const auto plan = prepareEdit(*this, txn, objectUuid, expected, edits);
    if (plan->changes.empty()) return {};
    return affectedEdges(*this, txn, objectUuid, currentGeometry(txn, objectUuid), plan->after);
}

std::string ObjectStore::editGeometry(WriteTxn& txn, const std::string& objectId,
    const std::vector<GeometryArrayEdit>& edits, const std::string& expected,
    const std::vector<GeometryEdgeEdit>& edgeEdits)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto objectUuid = formatUuid(parseUuid(objectId));
    DependencyScope scope;
    const auto plan = prepareEdit(*this, txn, objectUuid, expected, edits);
    const auto current = *currentGeometry(txn, objectUuid);
    if (plan->changes.empty()) {
        if (!edgeEdits.empty()) throw std::invalid_argument("geometry is unchanged; edit relationships directly");
        return current.uuid;
    }
    const auto object = *objectByUuid(txn, objectUuid);
    const auto affected = affectedEdges(*this, txn, objectUuid, current, plan->after);
    const auto decisions = validateEdgeEdits(*this, txn, db, objectUuid, plan->after, affected, edgeEdits);
    auto encoded = encodeGeometryDelta(plan->changes);
    Statement old(db, "SELECT manifest FROM geometry_revisions WHERE uuid=?");
    old.bindBlob(1, parseUuid(current.uuid)); old.row();
    const auto previous = loadGeometryManifest(db, current.uuid, old.text(0), current.contentHash);
    encoded.description["base"] = previous.contains("delta") ? previous.at("delta").at("base") : nlohmann::json(current.uuid);
    encoded.description["parent_hash"] = current.contentHash;
    encoded.description["result_hash"] = hashText(plan->manifest.dump());
    auto descriptor = plan->manifest;
    for (auto& a : descriptor.at("arrays")) a.erase("chunks");
    descriptor["delta"] = encoded.description;
    const auto text = descriptor.dump(), hash = hashText(plan->manifest.dump()), uuid = util::newUuid();
    Savepoint savepoint(db);
    Statement revision(db, "INSERT INTO geometry_revisions VALUES(?,?,?,?,?,?,?,?,?)");
    revision.bindBlob(1, parseUuid(uuid)); revision.bindBlob(2, parseUuid(objectUuid)); revision.bindBlob(3, parseUuid(current.uuid));
    revision.bind(4, hash); revision.bind(5, text); revision.bind(6, object.generation+1);
    revision.bind(7, author()); revision.bind(8, nowIsoUtc()); revision.bind(9, hostname()); revision.stepDone();
    Statement payload(db, "INSERT INTO geometry_deltas VALUES(?,?,?)");
    payload.bindBlob(1, parseUuid(uuid));
    check(sqlite3_bind_blob64(payload.value, 2, encoded.bytes.data(), encoded.bytes.size(), SQLITE_TRANSIENT), db);
    payload.bind(3, hashText(text)); payload.stepDone();
    // A snapshot head has no mutable overlay. Delta heads preserve unchanged
    // overlay chunks; replacing changed rows never retains full old tiles.
    if (!previous.contains("delta")) clearGeometryMaterialization(db, objectUuid);
    Statement put(db, "INSERT INTO geometry_current_chunks VALUES(?,?,?,?,?) ON CONFLICT(object_uuid,array_name,chunk_index) "
        "DO UPDATE SET hash=excluded.hash,data=excluded.data");
    for (const auto& change : plan->changes) {
        const auto& bytes = plan->modified.at(change.key);
        put.reset(); put.bindBlob(1, parseUuid(objectUuid)); put.bind(2, change.key.first); put.bind(3, change.key.second); put.bind(4, change.afterHash);
        check(sqlite3_bind_blob64(put.value, 5, bytes.data(), bytes.size(), SQLITE_TRANSIENT), db); put.stepDone();
    }
    Statement materialized(db, "INSERT INTO geometry_materializations VALUES(?,?,?) ON CONFLICT(object_uuid) "
        "DO UPDATE SET revision_uuid=excluded.revision_uuid,manifest=excluded.manifest");
    materialized.bindBlob(1, parseUuid(objectUuid)); materialized.bindBlob(2, parseUuid(uuid)); materialized.bind(3, plan->manifest.dump()); materialized.stepDone();
    Statement head(db, "UPDATE geometry_heads SET revision_uuid=? WHERE object_uuid=?");
    head.bindBlob(1, parseUuid(uuid)); head.bindBlob(2, parseUuid(objectUuid)); head.stepDone();
    const auto id = *objectIdByUuid(db, parseUuid(objectUuid));
    updateEditedGeometryBounds(db, id, uuid, edits);
    stampContentChange(db, author(), hostname(), "objects", id, parseUuid(objectUuid), "geometry");
    Statement update(db, "UPDATE objects SET storage_kind='sqlite',data_generation=generation,data_checksum=? WHERE id=?");
    update.bind(1, hash); update.bind(2, id); update.stepDone();
    for (const auto& edge : affected) {
        const auto& decision = decisions.at(edge.uuid);
        if (decision.action == "remove") { removeEdge(txn, edge.uuid); continue; }
        EdgeGeometryRepair repair;
        repair.edgeUuid = edge.uuid; repair.expectedGeneration = edge.generation;
        repair.aRevision = geometryHead(db, edge.aUuid); repair.bRevision = geometryHead(db, edge.bUuid);
        repair.aAnchor = decision.action == "retain" ? edge.aAnchor : decision.aAnchor;
        repair.bAnchor = decision.action == "retain" ? edge.bAnchor : decision.bAnchor;
        repairEdgeGeometry(txn, repair);
    }
    savepoint.release();
    return uuid;
}

} // namespace vc::objects
