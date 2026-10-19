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
    std::vector<ObjectRecord> result;
    Statement rows(db, std::string("SELECT ") + kObjectCols + " FROM objects ORDER BY uuid");
    while (rows.row()) {
        auto o = readObjectRow(db, rows);
        if (selection.kind && o.kind != *selection.kind) continue;
        if (selection.tag && std::find(o.tags.begin(), o.tags.end(), *selection.tag) == o.tags.end()) continue;
        if (!selected.empty() && !selected.contains(o.uuid)) continue;
        if (selection.connectedTo && !connected.contains(o.uuid)) continue;
        result.push_back(std::move(o));
    }
    return result;
}

std::string ObjectStore::exportJson(const Txn& txn, const JsonExportOptions& options) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    nlohmann::json out{{"format", "vc_object_metadata"}, {"version", 1}, {"metadata_only", true},
        {"store_uuid", uuid(txn)}, {"objects", nlohmann::json::array()}, {"edges", nlohmann::json::array()}};
    const std::set<std::string> fields(options.fields.begin(), options.fields.end());
    if (!fields.empty()) out["projected"] = true;
    auto project = [&](nlohmann::json record) {
        if (!fields.empty()) for (auto it = record.begin(); it != record.end();) {
            if (it.key() != "uuid" && it.key() != "kind" && it.key() != "a" && it.key() != "b" &&
                it.key() != "geometry" && it.key() != "geometry_bindings" && !fields.contains(it.key())) it = record.erase(it);
            else ++it;
        }
        return record;
    };
    std::set<std::string> included;
    for (const auto& object : selectObjects(txn, options.selection)) {
        included.insert(object.uuid);
        auto record = objectToJson(object);
        if (const auto geometry = currentGeometry(txn, object.uuid))
            record["geometry"] = {{"revision_uuid", geometry->uuid}, {"content_hash", geometry->contentHash}};
        out["objects"].push_back(project(std::move(record)));
    }
    Statement edges(db, std::string("SELECT ") + kEdgeCols + " FROM edges ORDER BY uuid");
    while (edges.row()) {
        const auto edge = readEdgeRow(db, edges);
        if (!included.contains(edge.aUuid) || !included.contains(edge.bUuid)) continue;
        if (!options.selection.edgeKinds.empty() && std::find(options.selection.edgeKinds.begin(),
            options.selection.edgeKinds.end(), edge.kind) == options.selection.edgeKinds.end()) continue;
        auto record = edgeToJson(edge);
        const auto state = edgeGeometryState(txn, edge.uuid);
        const bool swap = record["a"] != edge.aUuid;
        auto a = swap ? state.bRevision : state.aRevision, b = swap ? state.aRevision : state.bRevision;
        record["geometry_bindings"] = {{"a", a ? nlohmann::json(*a) : nlohmann::json()},
            {"b", b ? nlohmann::json(*b) : nlohmann::json()}, {"stale", state.stale}};
        out["edges"].push_back(project(std::move(record)));
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
    if (version != 1)
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
