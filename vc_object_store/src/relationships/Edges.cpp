#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include "internal/History.hpp"

namespace vc::objects {
using namespace detail;
std::string ObjectStore::addEdge(WriteTxn& txn, const EdgeRecord& record)
{
    detail::DependencyScope dependencyScope;
    for (const auto& [uuid, anchor] : {std::pair{record.aUuid, record.aAnchor}, std::pair{record.bUuid, record.bAnchor}})
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    auto* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    const auto uuid = addEdgeImpl(db, impl_->options.author, impl_->hostname, record, false);
    recordRelationshipValidation(db, *edgeByUuid(txn, uuid));
    savepoint.release();
    return uuid;
}

void ObjectStore::updateEdge(WriteTxn& txn, const EdgeRecord& record)
{
    detail::DependencyScope dependencyScope;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto existing = edgeByUuid(txn, record.uuid);
    if (!existing) throw std::runtime_error("object store: edge not found: " + record.uuid);
    auto rec = record;
    canonicalizeEdge(rec);
    if (rec.aUuid != rec.bUuid && rec.aUuid == existing->bUuid && rec.bUuid == existing->aUuid)
        swapEndpoints(rec);
    if (rec.kind != existing->kind || rec.aUuid != existing->aUuid || rec.bUuid != existing->bUuid)
        throw std::runtime_error("object store: edge kind and endpoints cannot change");
    for (const auto& [uuid, anchor] : {std::pair{rec.aUuid, rec.aAnchor}, std::pair{rec.bUuid, rec.bAnchor}})
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    std::sort(rec.tags.begin(), rec.tags.end());
    rec.tags.erase(std::unique(rec.tags.begin(), rec.tags.end()), rec.tags.end());
    if (rec.aAnchor == existing->aAnchor && rec.bAnchor == existing->bAnchor &&
        rec.tags == existing->tags && nlohmann::json::parse(rec.meta) == nlohmann::json::parse(existing->meta)) {
        if (!lifecycleBindingMatches(db, rec))
            stampContentChange(db, author(), hostname(), "edges", *edgeIdByUuid(db, parseUuid(rec.uuid)),
                               parseUuid(rec.uuid), "lifecycle_revalidated");
        recordRelationshipValidation(db, rec);
        return;
    }
    const auto id = *edgeIdByUuid(db, parseUuid(rec.uuid));
    Savepoint savepoint(db);
    Statement s(db, "UPDATE edges SET a_anchor_x=?,a_anchor_y=?,a_anchor_z=?,a_dir_x=?,a_dir_y=?,a_dir_z=?,"
        "b_anchor_x=?,b_anchor_y=?,b_anchor_z=?,b_dir_x=?,b_dir_y=?,b_dir_z=?,meta=?,a_dependencies=?,b_dependencies=? WHERE id=?");
    bindAnchor(s, 1, rec.aAnchor); bindAnchor(s, 7, rec.bAnchor);
    s.bind(13, rec.meta);
    s.bind(14, dependenciesToJson(rec.aAnchor).dump()); s.bind(15, dependenciesToJson(rec.bAnchor).dump());
    s.bind(16, id); s.stepDone();
    writeTags(db, "edge_tags", "edge_id", id, rec.tags);
    stampContentChange(db, author(), hostname(), "edges", id, parseUuid(rec.uuid), "update_edge");
    recordRelationshipValidation(db, rec);
    savepoint.release();
}

std::optional<EdgeRecord> ObjectStore::edgeByUuid(const Txn& txn, const std::string& uuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE uuid=?");
    s.bindBlob(1, parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readEdgeRow(db, s);
}

void ObjectStore::removeEdge(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    const auto bytes = parseUuid(uuid);
    auto id = edgeIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("object store: edge not found: " + uuid);
    Statement s(db, "DELETE FROM edges WHERE id=?");
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "remove_edge");
    savepoint.release();
}

std::vector<EdgeRecord> ObjectStore::edgesOf(const Txn& txn, const std::string& objectUuid, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("object store: object not found: " + objectUuid);
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE (a_id=? OR b_id=?)" +
        (includeRetired ? std::string() : " AND " + activeEdgeSql(db)) + " ORDER BY id");
    s.bind(1, *id);
    s.bind(2, *id);
    return readEdgeRows(db, s);
}

std::vector<EdgeRecord> ObjectStore::edgesByKind(const Txn& txn, const std::string& edgeKind, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE kind=?" +
        (includeRetired ? std::string() : " AND " + activeEdgeSql(db)) + " ORDER BY id");
    s.bind(1, edgeKind);
    return readEdgeRows(db, s);
}

std::unique_ptr<Statement> detail::edgeSelectionQuery(sqlite3* db, const EdgeSelection& selection,
    const std::string& columns, const std::optional<std::vector<int64_t>>& endpointIds)
{
    if (selection.limit && *selection.limit < 0) throw std::invalid_argument("edge limit must be nonnegative");
    const bool candidates = endpointIds || selection.endpointUuids || selection.uuids || selection.tag;
    std::string sql = "SELECT " + columns + " FROM edges" +
        (candidates ? " NOT INDEXED" : "") + " WHERE 1";
    if (!selection.includeRetired) sql += " AND " + activeEdgeSql(db);
    std::vector<std::string> parameters;
    auto parameter = [&](const std::string& value) {
        parameters.push_back(value);
        return "?" + std::to_string(parameters.size());
    };
    if (selection.kind) sql += " AND kind=" + parameter(*selection.kind);
    for (const auto& state : selection.approvalStates)
        if (state != "pending" && state != "approved" && state != "rejected")
            throw std::invalid_argument("invalid edge approval state: " + state);
    if (selection.approvalStates.size() == 1)
        sql += " AND approval_state=" + parameter(selection.approvalStates.front());
    else if (!selection.approvalStates.empty())
        sql += " AND approval_state IN (SELECT value FROM json_each(" + parameter(nlohmann::json(selection.approvalStates).dump()) + "))";
    if (selection.tag)
        sql += " AND id IN (SELECT edge_id FROM edge_tags WHERE tag=" + parameter(*selection.tag) + ")";
    if (endpointIds || selection.endpointUuids) {
        nlohmann::json ids = endpointIds ? nlohmann::json(*endpointIds) : nlohmann::json::array();
        if (!endpointIds) for (const auto& uuid : *selection.endpointUuids)
            if (auto id = objectIdByUuid(db, parseUuid(uuid))) ids.push_back(*id);
        const auto members = " IN (SELECT value FROM json_each(" + parameter(ids.dump()) + "))";
        const std::string kind = selection.kind ? " AND kind=?1" : "";
        // Materialize endpoint-index candidates, then probe by primary key.
        // Otherwise SQLite may scan an entire kind's UUID index to avoid sorting.
        sql += " AND id IN (SELECT id FROM edges INDEXED BY edges_a WHERE a_id" + members + kind;
        if (selection.bothEndpoints) sql += " AND b_id" + members;
        else sql += " UNION SELECT id FROM edges INDEXED BY edges_b WHERE b_id" + members + kind;
        sql += ")";
    }
    if (selection.uuids) {
        nlohmann::json ids = nlohmann::json::array();
        for (const auto& uuid : *selection.uuids)
            if (auto id = edgeIdByUuid(db, parseUuid(uuid))) ids.push_back(*id);
        sql += " AND id IN (SELECT value FROM json_each(" + parameter(ids.dump()) + "))";
    }
    const auto cursorParameter = parameters.size() + 1;
    if (selection.afterUuid) sql += " AND uuid>?" + std::to_string(cursorParameter);
    sql += " ORDER BY uuid LIMIT ?" + std::to_string(cursorParameter + (selection.afterUuid ? 1 : 0));
    auto rows = std::make_unique<Statement>(db, sql);
    int next = 1;
    for (const auto& value : parameters) rows->bind(next++, value);
    if (selection.afterUuid) rows->bindBlob(next++, parseUuid(*selection.afterUuid));
    rows->bind(next, selection.limit.value_or(-1));
    return rows;
}

std::vector<EdgeRecord> ObjectStore::selectEdges(const Txn& txn, const EdgeSelection& selection,
    const std::optional<std::vector<std::string>>& projection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const ObjectFields fields = projection ? ObjectFields(std::in_place, projection->begin(), projection->end()) : std::nullopt;
    auto rows = edgeSelectionQuery(db, selection, projectedEdgeColumns(fields));
    return readEdgeRows(db, *rows, fields);
}

} // namespace vc::objects
