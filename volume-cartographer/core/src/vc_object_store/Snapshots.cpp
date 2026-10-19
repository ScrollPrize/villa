#include "internal/RecordJson.hpp"

namespace vc::objects {


std::vector<ObjectRecord> ObjectStore::selectObjects(const Txn& txn, const ObjectSelection& selection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    std::set<std::string> selected, connected;
    for (const auto& id : selection.uuids) selected.insert(formatUuid(parseUuid(id)));
    if (selection.connectedTo) {
        for (const auto& id : componentOf(txn, *selection.connectedTo, selection.edgeKinds)) connected.insert(id);
    }
    std::string sql = std::string("SELECT ") + kObjectCols + " FROM objects WHERE 1";
    std::vector<std::string> parameters;
    auto bindFilter = [&](const std::string& clause, const std::string& value) {
        sql += clause; parameters.push_back(value);
    };
    if (!selected.empty() || selection.connectedTo) {
        nlohmann::json ids = nlohmann::json::array();
        const auto& candidates = selected.empty() ? connected : selected;
        for (const auto& uuid : candidates) {
            if (selection.connectedTo && !connected.contains(uuid)) continue;
            if (auto id = objectIdByUuid(db, parseUuid(uuid))) ids.push_back(*id);
        }
        // json_each is a read-only, single-parameter selection table. Unlike a
        // temporary table it also works on query_only pooled connections.
        bindFilter(" AND id IN (SELECT value FROM json_each(?))", ids.dump());
    }
    if (selection.kind) bindFilter(" AND kind=?", *selection.kind);
    if (selection.tag) bindFilter(" AND id IN (SELECT object_id FROM object_tags INDEXED BY object_tags_tag WHERE tag=?)", *selection.tag);
    sql += " ORDER BY uuid";
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    sql += " LIMIT ? OFFSET ?";
    Statement rows(db, sql);
    for (size_t i = 0; i < parameters.size(); ++i) rows.bind(int(i) + 1, parameters[i]);
    rows.bind(int(parameters.size()) + 1, selection.limit.value_or(-1));
    rows.bind(int(parameters.size()) + 2, selection.offset);
    return readObjectRows(db, rows);
}

std::string ObjectStore::exportJson(const Txn& txn, const JsonExportOptions& options) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    nlohmann::json out{{"format", "vc_object_metadata"}, {"version", 2}, {"metadata_only", true},
        {"store_uuid", uuid(txn)}, {"objects", nlohmann::json::array()}, {"edges", nlohmann::json::array()}};
    const std::set<std::string> fields(options.fields.begin(), options.fields.end());
    if (!fields.empty()) out["projected"] = true;
    nlohmann::json included = nlohmann::json::array();
    for (const auto& object : selectObjects(txn, options.selection)) {
        included.push_back(*objectIdByUuid(db, parseUuid(object.uuid)));
        auto record = objectToJson(object, fields.empty() ? nullptr : &fields);
        if (const auto geometry = currentGeometry(txn, object.uuid))
            record["geometry"] = {{"revision_uuid", geometry->uuid}, {"content_hash", geometry->contentHash}};
        out["objects"].push_back(std::move(record));
    }
    std::string edgeSql = std::string("SELECT ") + kEdgeCols + " FROM edges WHERE id IN ("
        "SELECT e.id FROM json_each(?1) selected CROSS JOIN edges e INDEXED BY edges_a "
        "WHERE e.a_id=selected.value AND e.b_id IN (SELECT value FROM json_each(?1))";
    if (!options.selection.edgeKinds.empty()) edgeSql += " AND e.kind IN (SELECT value FROM json_each(?2))";
    edgeSql += ") ORDER BY uuid";
    Statement edges(db, edgeSql);
    edges.bind(1, included.dump());
    if (!options.selection.edgeKinds.empty()) edges.bind(2, nlohmann::json(options.selection.edgeKinds).dump());
    while (edges.row()) {
        const auto edge = readEdgeRow(db, edges);
        auto record = edgeToJson(edge, fields.empty() ? nullptr : &fields);
        const auto state = edgeGeometryState(txn, edge.uuid);
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
    exec(db, "SAVEPOINT metadata_import;");
    try {
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
        exec(db, "RELEASE metadata_import;");
    } catch (...) {
        exec(db, "ROLLBACK TO metadata_import; RELEASE metadata_import;");
        throw;
    }
}

} // namespace vc::objects
