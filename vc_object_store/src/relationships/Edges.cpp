#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include "geometry/GeometryInternal.hpp"
#include "internal/History.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace vc::objects {
using namespace detail;
std::string ObjectStore::addEdge(WriteTxn& txn, const EdgeRecord& record)
{
    GeometryPhase phase("native.edge");
    detail::DependencyScope dependencyScope;
    {GeometryPhase geometryPhase("native.edge.geometry");
    for (const auto& [uuid, anchor] : {std::pair{record.aUuid, record.aAnchor}, std::pair{record.bUuid, record.bAnchor}}) {
        // validateDependencies immediately accepts this case. Anchor syntax
        // and finite values are still checked by canonicalizeEdge below.
        if (anchor.wholeGeometry && anchor.points.empty()) continue;
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    }
    }
    auto* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    const auto inserted = addEdgeImpl(db, impl_->options.author, impl_->hostname, record, false);
    {GeometryPhase validate("native.edge.validation");recordRelationshipValidation(db,inserted);}
    savepoint.release();
    return inserted.uuid;
}

std::string ObjectStore::insertEdgeWithId(WriteTxn& txn,const EdgeRecord& record,int64_t id)
{
    ExplicitRecordId allocation(static_cast<sqlite3*>(connection(txn)),"edges",record.uuid,id);
    const auto result=addEdge(txn,record);
    allocation.finish();return result;
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
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE uuid=? AND state=0", parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readEdgeRow(db, s);
}

void ObjectStore::removeEdge(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    const auto bytes = parseUuid(uuid);
    Statement row(db, "SELECT id,kind FROM edges WHERE uuid=? AND state=0", bytes);
    if (!row.row()) throw std::runtime_error("object store: edge not found: " + uuid);
    const auto kind = row.text(1);
    execute(db,"UPDATE edges SET state=2,generation=?,changed_by=?,changed_at=?,changed_host=? WHERE id=?",
        writeGeneration(db),author(),nowIsoUtc(),hostname(),row.i64(0));
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "remove_edge", kind);
    savepoint.release();
}

std::vector<EdgeRecord> ObjectStore::edgesOf(const Txn& txn, const std::string& objectUuid, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    const auto key=parseUuid(objectUuid);
    if(!objectIdByUuid(db,key)) {
        Statement present(db,"SELECT 1 FROM edges WHERE state=0 AND (a_uuid=?1 OR b_uuid=?1) LIMIT 1",key);
        if(!present.row())throw std::runtime_error("object store: object not found: "+objectUuid);
    }
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE state=0 AND (a_uuid=? OR b_uuid=?)" +
        (includeRetired ? std::string() : " AND " + activeEdgeSql()) + " ORDER BY id", key, key);
    return readEdgeRows(db, s);
}

std::vector<EdgeRecord> ObjectStore::edgesByKind(const Txn& txn, const std::string& edgeKind, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE state=0 AND kind=?" +
        (includeRetired ? std::string() : " AND " + activeEdgeSql()) + " ORDER BY id", edgeKind);
    return readEdgeRows(db, s);
}

std::unique_ptr<Statement> detail::edgeSelectionQuery(sqlite3* db, const EdgeSelection& selection,
    const std::string& columns, const std::optional<std::vector<int64_t>>& endpointIds)
{
    validateEdgeSelection(selection);
    const bool candidates = endpointIds || selection.endpointUuids || selection.uuids || selection.tag;
    std::string sql = "SELECT " + columns + " FROM edges" +
        (candidates ? " NOT INDEXED" : "") + " WHERE state=0";
    if (!selection.includeRetired) sql += " AND " + activeEdgeSql();
    sql += fiberAdjacencySql(selection.adjacent);
    std::vector<std::variant<std::string,std::vector<uint8_t>>> parameters;
    std::string prefix;
    auto parameter = [&](const std::string& value) {
        parameters.push_back(value);
        return "?" + std::to_string(parameters.size());
    };
    if (selection.kind) sql += " AND kind=" + parameter(*selection.kind);
    if (!selection.attachments) sql += " AND kind<>" + parameter(kAttachmentEdgeKind);
    if (selection.approvalStates.size() == 1)
        sql += " AND approval_state=" + parameter(selection.approvalStates.front());
    else if (!selection.approvalStates.empty())
        sql += " AND approval_state IN (SELECT value FROM json_each(" + parameter(nlohmann::json(selection.approvalStates).dump()) + "))";
    if (selection.tag)
        sql += " AND id IN (SELECT edge_id FROM edge_tags WHERE tag=" + parameter(*selection.tag) + ")";
    if (endpointIds || selection.endpointUuids) {
        std::string members;
        if(endpointIds) {
            members=" IN (SELECT uuid FROM objects WHERE state<>2 AND id IN (SELECT value FROM json_each("+
                parameter(nlohmann::json(*endpointIds).dump())+")))";
        } else {
            std::vector<uint8_t> keys;keys.reserve(selection.endpointUuids->size()*16);
            for(const auto& uuid:*selection.endpointUuids){const auto bytes=parseUuid(uuid);keys.insert(keys.end(),bytes.begin(),bytes.end());}
            parameters.emplace_back(std::move(keys));
            const auto blob="?"+std::to_string(parameters.size());
            prefix="WITH RECURSIVE requested_key_offsets(pos) AS (SELECT 1 WHERE length("+blob+
                ")>0 UNION ALL SELECT pos+16 FROM requested_key_offsets WHERE pos+16<=length("+blob+
                ")), requested_keys(uuid) AS MATERIALIZED (SELECT substr("+blob+",pos,16) FROM requested_key_offsets) ";
            members=" IN (SELECT uuid FROM requested_keys)";
        }
        const std::string kind = selection.kind ? " AND kind=?1" : "";
        // Materialize endpoint-index candidates, then probe by primary key.
        // Otherwise SQLite may scan an entire kind's UUID index to avoid sorting.
        sql += " AND id IN (SELECT id FROM edges INDEXED BY edges_a WHERE a_uuid" + members + kind;
        if (selection.bothEndpoints) sql += " AND b_uuid" + members;
        else sql += " UNION SELECT id FROM edges INDEXED BY edges_b WHERE b_uuid" + members + kind;
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
    auto rows = std::make_unique<Statement>(db, prefix+sql);
    int next = 1;
    for (const auto& value : parameters) std::visit([&](const auto& bound){rows->bind(next++,bound);},value);
    if (selection.afterUuid) rows->bind(next++, parseUuid(*selection.afterUuid));
    rows->bind(next, selection.limit.value_or(-1));
    return rows;
}

std::vector<EdgeRecord> ObjectStore::selectEdges(const Txn& txn, const EdgeSelection& selection,
    const std::optional<std::vector<std::string>>& projection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const ObjectFields fields = projection ? ObjectFields(std::in_place, projection->begin(), projection->end()) : std::nullopt;
    const auto columns = projectedEdgeColumns(fields);
    if (selection.applicableOnly) {
        validateEdgeSelection(selection);
        requirePublishedGraphRead(txn);
        // Apply validity before pagination; then hydrate only accepted records.
        auto candidates = selection; candidates.limit.reset(); candidates.applicableOnly = false;
        auto rows = edgeSelectionQuery(db, candidates, "uuid");
        std::vector<std::string> accepted;
        while ((!selection.limit || accepted.size() < static_cast<size_t>(*selection.limit)) && rows->row()) {
            auto uuid = formatUuid(rows->blob16(0));
            if (currentRelationshipApplicable(*this, txn, uuid)) accepted.push_back(std::move(uuid));
        }
        EdgeSelection selected; selected.uuids = std::move(accepted); selected.includeRetired = true;
        return selectEdges(txn, selected, projection);
    }
    auto rows = edgeSelectionQuery(db, selection, columns);
    return readEdgeRows(db, *rows, fields);
}

std::string ObjectStore::attach(WriteTxn& txn, const std::string& dataUuid, const std::string& ownerUuid, const std::string& role)
{
    EdgeRecord edge;
    edge.kind = kAttachmentEdgeKind;
    edge.aUuid = dataUuid;
    edge.bUuid = ownerUuid;
    edge.aAnchor.wholeGeometry = true;
    edge.bAnchor.wholeGeometry = true;
    edge.meta = nlohmann::json{{"role", role}}.dump();
    return addEdge(txn, edge);
}

std::vector<EdgeRecord> ObjectStore::attachments(const Txn& txn, const std::string& ownerUuid,
                                                 const std::optional<std::string>& role) const
{
    EdgeSelection selection;
    selection.kind = kAttachmentEdgeKind;
    selection.endpointUuids = std::vector<std::string>{ownerUuid};
    auto edges = selectEdges(txn, selection);
    if (role)
        std::erase_if(edges, [&](const EdgeRecord& edge) {
            return nlohmann::json::parse(edge.meta).value("role", std::string{}) != *role;
        });
    return edges;
}

} // namespace vc::objects
