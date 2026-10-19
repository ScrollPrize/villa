#include "internal/RecordSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/Dependencies.hpp"

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

} // namespace vc::objects
