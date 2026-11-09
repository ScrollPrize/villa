#include "internal/RecordJson.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/History.hpp"
#include "internal/Extensions.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects {
using namespace detail;

std::string ObjectStore::exportJson(const Txn& txn, const JsonExportOptions& options) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    rejectExtensionLoss(db,"metadata JSON export");
    nlohmann::json out{{"format", "vc_object_metadata"}, {"version", 2}, {"metadata_only", true},
        {"store_uuid", uuid(txn)}, {"objects", nlohmann::json::array()}, {"edges", nlohmann::json::array()}};
    const std::set<std::string> fields(options.fields.begin(), options.fields.end());
    if (!fields.empty()) out["projected"] = true;
    const std::set<std::string> commonFields{
        "uuid", "kind", "commit_sequence", "created_by", "created_at", "created_host", "changed_by", "changed_at", "changed_host",
        "approval_state", "approved_by", "approved_at", "approved_host", "approved_version", "tags", "meta"};
    auto projection = [&](std::set<std::string> valid) -> std::optional<std::vector<std::string>> {
        if (fields.empty()) return std::nullopt;
        valid.insert(commonFields.begin(), commonFields.end());
        std::vector<std::string> selected;
        for (const auto& field : fields) if (valid.contains(field)) selected.push_back(field);
        return selected;
    };
    const auto objectProjection = projection({"name", "format_version", "storage_kind", "storage_ref", "version", "data_checksum", "fiber", "surface", "track"});
    const auto edgeProjection = projection({"a", "b", "a_anchor", "b_anchor"});
    ObjectFields edgeFields = edgeProjection ? ObjectFields(std::in_place, edgeProjection->begin(), edgeProjection->end()) : std::nullopt;
    // Export canonicalizes endpoint order, including self-link dependency order.
    // Both anchors are needed for that even when only one is requested.
    if (edgeFields) { edgeFields->insert("a_anchor"); edgeFields->insert("b_anchor"); }
    nlohmann::json included = nlohmann::json::array();
    for (const auto& object : selectObjects(txn, options.selection, objectProjection)) {
        included.push_back(*objectIdByUuid(db, parseUuid(object.uuid)));
        auto record = objectToJson(object, fields.empty() ? nullptr : &fields);
        if (const auto geometry = currentGeometry(txn, object.uuid))
            record["geometry"] = {{"revision_uuid", geometry->uuid}, {"content_hash", geometry->contentHash}};
        out["objects"].push_back(std::move(record));
    }
    std::string members = "SELECT e.id FROM json_each(?1) selected CROSS JOIN objects selected_object ON selected_object.id=selected.value "
        "CROSS JOIN edges e INDEXED BY edges_a WHERE e.state=0 AND e.a_uuid=selected_object.uuid "
        "AND e.b_uuid IN (SELECT uuid FROM objects WHERE id IN (SELECT value FROM json_each(?1)))";
    if (!options.selection.includeRetired)
        members += " AND e.id IN (SELECT id FROM edges WHERE " + activeEdgeSql() + ")";
    if (!options.selection.edgeKinds.empty()) members += " AND e.kind IN (SELECT value FROM json_each(?2))";
    Statement states(db, "SELECT e.uuid,CASE WHEN a.geometry_manifest IS NOT NULL THEN a.storage_ref END,"
        "CASE WHEN z.geometry_manifest IS NOT NULL THEN z.storage_ref END FROM edges e "
        "JOIN objects a ON a.uuid=e.a_uuid JOIN objects z ON z.uuid=e.b_uuid "
        "WHERE e.id IN (" + members + ")", included.dump());
    if (!options.selection.edgeKinds.empty()) states.bind(2, nlohmann::json(options.selection.edgeKinds).dump());
    std::map<std::string, EdgeGeometryState> geometryStates;
    while (states.row()) {
        const auto id = formatUuid(states.blob16(0));
        auto revision = [&](int col) -> std::optional<std::string> {
            return states.isNull(col) ? std::nullopt : std::optional(readGeometryTokenColumn(states,col));
        };
        EdgeGeometryState state{revision(1), revision(2), false};
        geometryStates.emplace(id, std::move(state));
    }
    Statement edges(db, "SELECT " + projectedEdgeColumns(edgeFields) + " FROM edges WHERE id IN (" + members + ") ORDER BY uuid",
        included.dump());
    if (!options.selection.edgeKinds.empty()) edges.bind(2, nlohmann::json(options.selection.edgeKinds).dump());
    for (const auto& edge : readEdgeRows(db, edges, edgeFields)) {
        auto record = edgeToJson(edge, fields.empty() ? nullptr : &fields);
        const auto& state = geometryStates.at(edge.uuid);
        const bool swap = record["a"] != edge.aUuid;
        auto a = swap ? state.bRevision : state.aRevision, b = swap ? state.aRevision : state.bRevision;
        record["geometry_bindings"] = {{"a", a ? nlohmann::json(*a) : nlohmann::json()},
            {"b", b ? nlohmann::json(*b) : nlohmann::json()}, {"stale", state.stale}};
        out["edges"].push_back(std::move(record));
    }
    return out.dump(2);
}

void ObjectStore::importJson(WriteTxn& txn, const std::string& jsonText)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    const auto root = nlohmann::json::parse(jsonText);
    if (root.value("format", "") != "vc_object_metadata")
        throw std::runtime_error("object store: not a vc_object_metadata document");
    const int version = root.value("version", 0);
    if (version != 2)
        throw std::runtime_error("object store: unsupported export version");
    if (root.value("projected", false)) throw std::runtime_error("projected exports are for inspection, not import");
    for (const auto& object : root.at("objects")) {
        if (object.contains("geometry")) throw std::runtime_error("metadata export references stored geometry; use synchronize to transfer it");
        if (object.value("retired", false) || !object.value("lifecycle_epoch", "").empty())
            throw std::runtime_error("metadata export contains lifecycle state; use synchronize to preserve relationship bindings");
    }
    // A failed import must not leave half a snapshot, even if the caller
    // catches the error and commits other work in its enclosing transaction.
    Savepoint savepoint(db);
    std::set<std::string> seen;
    auto checkIdentity = [&](std::string& uuid) {
        uuid = formatUuid(parseUuid(uuid));
        if (!seen.insert(uuid).second)
            throw std::runtime_error("object store: duplicate snapshot UUID: " + uuid);
    };
    for (const auto& j : root.at("objects")) {
        auto object = objectFromJson(j);
        checkIdentity(object.uuid);
        commonFromJson(j, object);
        if (edgeByUuid(txn, object.uuid))
            throw std::runtime_error("object store: snapshot UUID belongs to an edge: " + object.uuid);
        if (auto existing = objectByUuid(txn, object.uuid)) {
            if (recordContent(objectToJson(*existing)) == recordContent(objectToJson(object))) continue;
            throw std::runtime_error("object store: snapshot conflicts with existing object: " + object.uuid);
        }
        upsertObjectImpl(db, author(), impl_->hostname, object, true);
    }
    for (const auto& j : root.at("edges")) {
        auto e = edgeFromJson(j);
        checkIdentity(e.uuid);
        if (objectByUuid(txn, e.uuid))
            throw std::runtime_error("object store: snapshot UUID belongs to an object: " + e.uuid);
        if (auto existing = edgeByUuid(txn, e.uuid)) {
            if (recordContent(edgeToJson(*existing)) == recordContent(edgeToJson(e))) continue;
            throw std::runtime_error("object store: snapshot conflicts with existing edge: " + e.uuid);
        }
        addEdgeImpl(db, impl_->options.author, impl_->hostname, e, true);
        recordRelationshipValidation(db, *edgeByUuid(txn, e.uuid));
    }
    savepoint.release();
}

} // namespace vc::objects
