#include "internal/RecordJson.hpp"
#include "internal/EdgeSql.hpp"

namespace vc::objects {
using namespace detail;

std::string ObjectStore::exportJson(const Txn& txn, const JsonExportOptions& options) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    nlohmann::json out{{"format", "vc_object_metadata"}, {"version", 2}, {"metadata_only", true},
        {"store_uuid", uuid(txn)}, {"objects", nlohmann::json::array()}, {"edges", nlohmann::json::array()}};
    const std::set<std::string> fields(options.fields.begin(), options.fields.end());
    if (!fields.empty()) out["projected"] = true;
    const std::set<std::string> commonFields{
        "uuid", "kind", "generation", "created_by", "created_at", "created_host", "changed_by", "changed_at", "changed_host",
        "approval_state", "approved_by", "approved_at", "approved_host", "approved_generation", "tags", "meta"};
    auto projection = [&](std::set<std::string> valid) -> std::optional<std::vector<std::string>> {
        if (fields.empty()) return std::nullopt;
        valid.insert(commonFields.begin(), commonFields.end());
        std::vector<std::string> selected;
        for (const auto& field : fields) if (valid.contains(field)) selected.push_back(field);
        return selected;
    };
    const auto objectProjection = projection({"name", "format_version", "storage_kind", "storage_ref", "data_generation", "data_checksum", "fiber", "patch", "track"});
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
    std::string members = "SELECT e.id FROM json_each(?1) selected CROSS JOIN edges e INDEXED BY edges_a "
        "WHERE e.a_id=selected.value AND e.b_id IN (SELECT value FROM json_each(?1))";
    if (!options.selection.edgeKinds.empty()) members += " AND e.kind IN (SELECT value FROM json_each(?2))";
    Statement states(db, "SELECT e.uuid,b.a_revision,b.b_revision,ha.revision_uuid,hb.revision_uuid FROM edges e "
        "LEFT JOIN edge_geometry_bindings b ON b.edge_id=e.id "
        "JOIN objects a ON a.id=e.a_id JOIN objects z ON z.id=e.b_id "
        "LEFT JOIN geometry_heads ha ON ha.object_uuid=a.uuid LEFT JOIN geometry_heads hb ON hb.object_uuid=z.uuid "
        "WHERE e.id IN (" + members + ")");
    states.bind(1, included.dump());
    if (!options.selection.edgeKinds.empty()) states.bind(2, nlohmann::json(options.selection.edgeKinds).dump());
    std::map<std::string, EdgeGeometryState> geometryStates;
    while (states.row()) {
        const auto id = formatUuid(states.blob16(0));
        auto revision = [&](int col) -> std::optional<std::string> {
            return states.isNull(col) ? std::nullopt : std::optional(formatUuid(states.blob16(col)));
        };
        EdgeGeometryState state{revision(1), revision(2), false};
        // Only endpoints whose head moved need dependency-aware inspection.
        if (state.aRevision != revision(3) || state.bRevision != revision(4)) state = edgeGeometryState(txn, id);
        geometryStates.emplace(id, std::move(state));
    }
    Statement edges(db, "SELECT " + projectedEdgeColumns(edgeFields) + " FROM edges WHERE id IN (" + members + ") ORDER BY uuid");
    edges.bind(1, included.dump());
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
    for (const auto& object : root.at("objects"))
        if (object.contains("geometry")) throw std::runtime_error("metadata export references stored geometry; use synchronize to transfer it");
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
            if (objectToJson(*existing) == objectToJson(object)) continue;
            throw std::runtime_error("object store: snapshot conflicts with existing object: " + object.uuid);
        }
        upsertObjectImpl(db, impl_->options.author, impl_->hostname, object, true);
    }
    for (const auto& j : root.at("edges")) {
        auto e = edgeFromJson(j);
        checkIdentity(e.uuid);
        if (objectByUuid(txn, e.uuid))
            throw std::runtime_error("object store: snapshot UUID belongs to an object: " + e.uuid);
        if (auto existing = edgeByUuid(txn, e.uuid)) {
            if (edgeToJson(*existing) == edgeToJson(e)) continue;
            throw std::runtime_error("object store: snapshot conflicts with existing edge: " + e.uuid);
        }
        addEdgeImpl(db, impl_->options.author, impl_->hostname, e, true);
    }
    savepoint.release();
}

} // namespace vc::objects
