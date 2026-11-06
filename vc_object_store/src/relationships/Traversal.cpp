#include "internal/EdgeSql.hpp"
#include "internal/History.hpp"
#include "geometry/GeometryToken.hpp"
#include "internal/ExternalEndpoint.hpp"

namespace vc::objects {
using namespace detail;
std::vector<std::string> ObjectStore::neighbours(const Txn& txn, const std::string& objectUuid,
                                             const std::optional<std::string>& edgeKind,
                                             const std::optional<std::string>& neighbourKind, bool includeRetired,
                                             std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto key=parseUuid(objectUuid);
    Statement own(db,"SELECT deleted FROM objects WHERE uuid=?",key);
    if(own.row()) {if(own.i64(0))return {};}
    else {Statement known(db,"SELECT 1 FROM edges WHERE deleted=0 AND (a_uuid=?1 OR b_uuid=?1) LIMIT 1",key);
        if(!known.row())throw std::runtime_error("object store: object not found: "+objectUuid);
        if(!includeRetired&&externalEndpointRecord(db,objectUuid).retired)return {};
    }
    if (applicableOnly) requirePublishedGraphRead(txn);
    const std::string filter = " AND deleted=0" + fiberAdjacencySql(adjacentFilter) +
        (edgeKind ? std::string(" AND kind=?2") : " AND kind<>'" + std::string(kAttachmentEdgeKind) + "'") +
        (includeRetired ? "" : " AND " + activeEdgeSql());
    // Filter before deduplication: a rejected parallel edge must not hide a
    // later applicable edge to the same neighbour. Preserve first-edge order.
    std::string sql = "SELECT e.neighbour,e.uuid,owned.id,owned.kind FROM (SELECT id,uuid,b_uuid AS neighbour FROM edges INDEXED BY edges_a WHERE a_uuid=?1" + filter +
        " UNION SELECT id,uuid,a_uuid AS neighbour FROM edges INDEXED BY edges_b WHERE b_uuid=?1" + filter +
        ") e LEFT JOIN objects owned ON owned.uuid=e.neighbour WHERE coalesce(owned.deleted,0)=0";
    sql += " ORDER BY e.id";
    Statement rows(db, sql, key);
    if (edgeKind) rows.bind(2, *edgeKind);
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    while (rows.row()) {
        const auto next = formatUuid(rows.blob16(0));
        if (seen.contains(next)) continue;
        if(rows.isNull(2)&&(!includeRetired||neighbourKind)){
            const auto endpoint=externalEndpointRecord(db,next);
            if((!includeRetired&&endpoint.retired)||(neighbourKind&&endpoint.kind!=*neighbourKind))continue;
        }else if(neighbourKind&&rows.text(3)!=*neighbourKind)continue;
        if (applicableOnly && !currentRelationshipApplicable(*this, txn, formatUuid(rows.blob16(1)))) continue;
        seen.insert(next); out.push_back(next);
    }
    return out;
}

namespace {

void checkTraversalCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load()) throw std::runtime_error("object store: traversal cancelled");
}

template<class Node, class Adjacent>
std::vector<Node> breadthFirst(Node start, int maxDepth, const std::atomic_bool* cancel, Adjacent adjacent)
{
    std::vector<Node> result;
    std::unordered_set<Node> visited{start};
    std::queue<std::pair<Node, int>> queue; queue.emplace(start, 0);
    while (!queue.empty()) {
        checkTraversalCancelled(cancel);
        const auto [node, depth] = queue.front(); queue.pop();
        result.push_back(node);
        if (depth >= maxDepth) continue;
        adjacent(node, [&](Node next, auto accepted) {
            checkTraversalCancelled(cancel);
            if (!visited.contains(next) && accepted()) {
                visited.insert(next); queue.emplace(next, depth+1);
            }
        });
    }
    return result;
}

struct ProgressGuard {
    sqlite3* db;
    ~ProgressGuard() { sqlite3_progress_handler(db, 0, nullptr, nullptr); }
};

std::vector<std::string> traverse(sqlite3* db, const std::string& startUuid, int maxDepth,
                                  const std::vector<std::string>& edgeKindFilter, const std::atomic_bool* cancel, bool includeRetired,
                                  std::optional<bool> adjacentFilter = std::nullopt,
                                  const ObjectStore* store = nullptr, const Txn* txn = nullptr)
{
    auto checkCancelled = [&] { checkTraversalCancelled(cancel); };
    checkCancelled();
    ProgressGuard guard{db};
    if (cancel)
        sqlite3_progress_handler(
            db, 1000, [](void* p) { return static_cast<const std::atomic_bool*>(p)->load() ? 1 : 0; },
            const_cast<std::atomic_bool*>(cancel));
    const auto startBytes = parseUuid(startUuid);
    auto active=[&](const std::string& uuid) {
        Statement local(db,"SELECT deleted,retired FROM objects WHERE uuid=?",parseUuid(uuid));
        if(local.row())return !local.i64(0)&&(includeRetired||!local.i64(1));
        return includeRetired||!externalEndpointRecord(db,uuid).retired;
    };
    Statement known(db,"SELECT 1 FROM objects WHERE uuid=?1 UNION ALL SELECT 1 FROM edges WHERE deleted=0 AND (a_uuid=?1 OR b_uuid=?1) LIMIT 1",startBytes);
    if(!known.row())throw std::runtime_error("object store: object not found: "+startUuid);
    if(!active(startUuid))return {};
    // Probe the endpoint indexes only for nodes reached by this traversal.
    // Visit neighbours in stable internal-ID order.
    std::string filter=" AND deleted=0";
    if (!edgeKindFilter.empty()) {
        filter += " AND kind IN (";
        for (size_t i = 0; i < edgeKindFilter.size(); ++i)
            filter += (i ? ",?" : "?") + std::to_string(i + 2);
        filter += ")";
    } else {
        filter += " AND kind<>'" + std::string(kAttachmentEdgeKind) + "'";
    }
    filter += fiberAdjacencySql(adjacentFilter);
    if (!includeRetired) filter += " AND " + activeEdgeSql();
    // Each UNION arm uses an endpoint-leading index even without ANALYZE.
    std::string sql = "SELECT id,b_uuid,uuid FROM edges INDEXED BY edges_a WHERE a_uuid=?1" + filter +
        " UNION SELECT id,a_uuid,uuid FROM edges INDEXED BY edges_b WHERE b_uuid=?1" + filter + " ORDER BY id";
    Statement adjacent(db, sql);
    std::unordered_map<std::string,bool> activeNodes;
    activeNodes.emplace(startUuid,true);
    try {
        return breadthFirst(formatUuid(startBytes), maxDepth, cancel, [&](const std::string& uuid, auto visit) {
            adjacent.reset(); adjacent.bind(1, parseUuid(uuid));
            for(size_t i=0;i<edgeKindFilter.size();++i)adjacent.bind(int(i)+2,edgeKindFilter[i]);
            while(adjacent.row()) {
                const auto endpoint=formatUuid(adjacent.blob16(1));
                visit(endpoint,[&] {
                    auto found=activeNodes.find(endpoint);
                    if(found==activeNodes.end())found=activeNodes.emplace(endpoint,active(endpoint)).first;
                    return found->second&&(!store||currentRelationshipApplicable(*store,*txn,formatUuid(adjacent.blob16(2))));
                });
            }
        });
    } catch (const std::runtime_error&) {
        checkCancelled(); // Translate SQLite interrupts into the cancellation contract.
        throw;
    }
}

}  // namespace

std::vector<int64_t> detail::componentIds(sqlite3* db, const std::string& start,
                                        const std::vector<std::string>& kinds, bool includeRetired, std::optional<bool> adjacentFilter)
{
    StatementPool* pool = nullptr;
    if (auto it = activeStatementPools.find(db); it != activeStatementPools.end() && it->second->readSnapshot)
        pool = it->second;
    const auto key = nlohmann::json({formatUuid(parseUuid(start)), kinds, includeRetired, adjacentFilter ? nlohmann::json(*adjacentFilter) : nlohmann::json(nullptr)}).dump();
    if (pool) if (auto it = pool->components.find(key); it != pool->components.end()) return it->second;
    const auto keys=traverse(db,start,INT_MAX,kinds,nullptr,includeRetired,adjacentFilter);
    std::vector<int64_t> ids;ids.reserve(keys.size());
    for(const auto& uuid:keys)if(const auto id=objectIdByUuid(db,parseUuid(uuid)))ids.push_back(*id);
    // Bound the transaction-local membership cache; never reuse across snapshots.
    if (pool && ids.size() <= 262144) {
        if (pool->componentNodes + ids.size() > 262144) { pool->components.clear(); pool->componentNodes = 0; }
        pool->componentNodes += ids.size();
        pool->components.emplace(key, ids);
    }
    return ids;
}

std::vector<std::string> ObjectStore::componentOf(const Txn& txn, const std::string& objectUuid,
                                              const std::vector<std::string>& edgeKindFilter,
                                              const std::atomic_bool* cancel, bool includeRetired,
                                              std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (applicableOnly) requirePublishedGraphRead(txn);
    return traverse(db,objectUuid,INT_MAX,edgeKindFilter,cancel,includeRetired,adjacentFilter,applicableOnly?this:nullptr,&txn);
}

std::vector<std::string> ObjectStore::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel, bool includeRetired,
                                              std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    if (maxDepth < 0) throw std::runtime_error("object store: maxDepth must be >= 0");
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (applicableOnly) requirePublishedGraphRead(txn);
    return traverse(db, objectUuid, maxDepth, edgeKindFilter, cancel, includeRetired,
                                adjacentFilter, applicableOnly ? this : nullptr, &txn);
}

namespace {
void finishGraph(GraphReadView& result, sqlite3* db)
{
    result.offsets.resize(result.objectUuids.size() + 1);
    for (const auto& [a, b] : result.endpoints) { ++result.offsets[a+1]; ++result.offsets[b+1]; }
    for (size_t i = 1; i < result.offsets.size(); ++i) result.offsets[i] += result.offsets[i-1];
    result.neighbours.resize(result.offsets.back()); result.edgeIndices.resize(result.offsets.back());
    auto next = result.offsets;
    for (size_t edge = 0; edge < result.endpoints.size(); ++edge) {
        const auto [a, b] = result.endpoints[edge];
        for (const auto& [from, to] : {std::pair{a, b}, std::pair{b, a}}) {
            const auto slot = next[from]++;
            result.neighbours[slot] = to; result.edgeIndices[slot] = edge;
        }
    }
    if(historyEnabled(db)) {
        Statement journal(db, "SELECT coalesce(max(seq),0) FROM journal");
        (void)journal.row(); result.journalSequence = journal.i64(0);
    }
}
}

GraphReadView ObjectStore::graphReadView(const Txn& txn, const ObjectSelection& objects,
                                       const EdgeSelection& edges) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    GraphReadView result;
    if (edges.endpointUuids)
        throw std::invalid_argument("graph endpoint scope is specified by its object selection");
    std::unordered_map<std::string, uint64_t> nodes;
    std::vector<int64_t> ids;
    auto objectsQuery = objectSelectionQuery(db, objects,
        "o.id,o.uuid,o.generation,CASE WHEN o.geometry_manifest IS NOT NULL THEN o.storage_ref END");
    while (objectsQuery->row()) {
        const auto node = result.objectUuids.size();
        const auto id = objectsQuery->i64(0);
        nodes.emplace(formatUuid(objectsQuery->blob16(1)), node); ids.push_back(id);
        result.objectUuids.push_back(formatUuid(objectsQuery->blob16(1)));
        result.generations.push_back(objectsQuery->i64(2));
        result.geometryRevisions.push_back(objectsQuery->isNull(3) ? std::nullopt :
            std::optional(readGeometryTokenColumn(*objectsQuery,3)));
    }
    auto selection = edges; selection.bothEndpoints = true;
    selection.includeRetired = objects.includeRetired;
    // Attachments are not graph links unless asked for by kind.
    if (!selection.kind) selection.attachments = false;
    if (selection.applicableOnly) {
        selection.endpointUuids = result.objectUuids;
        for (const auto& edge : selectEdges(txn, selection, std::vector<std::string>{})) {
            result.edgeUuids.push_back(edge.uuid);
            result.endpoints.push_back({nodes.at(edge.aUuid),
                                        nodes.at(edge.bUuid)});
        }
    } else {
        auto links = edgeSelectionQuery(db, selection, "uuid,a_uuid,b_uuid", ids);
        while (links->row()) {
            result.edgeUuids.push_back(formatUuid(links->blob16(0)));
            result.endpoints.push_back({nodes.at(formatUuid(links->blob16(1))), nodes.at(formatUuid(links->blob16(2)))});
        }
    }
    finishGraph(result, db);
    return result;
}

GraphReadView ObjectStore::graphReadViewAt(const Txn& txn, const RevisionSelection& selection,
                                         const EdgeSelection& edges) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    requirePublishedGraphRead(txn);
    validateEdgeSelection(edges);
    if (edges.endpointUuids)
        throw std::invalid_argument("graph endpoint scope is specified by its revision selection");
    GraphReadView result;
    result.selectedObjects.emplace(); result.selectedRelationships.emplace();
    RevisionSelection resolved;
    for (const auto& [object, revision] : selection) {
        const auto uuid = formatUuid(parseUuid(object));
        const auto value = revision ? objectRevision(txn, *revision) : currentObjectRevision(txn, uuid);
        if (!value || value->objectUuid != uuid || value->deleted)
            throw std::invalid_argument("object history: selection revision does not belong to object");
        if (!resolved.emplace(uuid, value->uuid).second)
            throw std::invalid_argument("duplicate object in revision selection");
    }
    std::map<std::string, uint64_t> nodes;
    for (const auto& [object, revision] : resolved) {
        auto value = *objectRevision(txn, *revision);
        nodes.emplace(object, result.objectUuids.size());
        result.objectUuids.push_back(object); result.generations.push_back(value.record.generation);
        result.geometryRevisions.push_back(value.geometryRevision);
        result.selectedObjects->push_back(std::move(value));
    }
    auto graphEdges = edges;
    if (!graphEdges.kind) graphEdges.attachments = false;
    for (auto& selected : relationshipsAt(txn, resolved))
        if (selected.applicable && matchesEdgeSelection(selected.relationship.record, graphEdges))
            result.selectedRelationships->push_back(std::move(selected.relationship));
    auto& relationships = *result.selectedRelationships;
    std::sort(relationships.begin(), relationships.end(), [](const auto& a, const auto& b) { return a.record.uuid < b.record.uuid; });
    if (edges.limit && relationships.size() > static_cast<size_t>(*edges.limit)) relationships.resize(*edges.limit);
    for (const auto& relationship : relationships) {
        const auto& edge = relationship.record;
        result.edgeUuids.push_back(edge.uuid);
        result.endpoints.push_back({nodes.at(edge.aUuid), nodes.at(edge.bUuid)});
    }
    finishGraph(result, db);
    return result;
}

std::vector<std::string> GraphReadView::reachableWithin(const std::string& objectUuid, int maxDepth,
                                                       const std::atomic_bool* cancel) const
{
    if (maxDepth < 0) throw std::invalid_argument("object store: maxDepth must be >= 0");
    checkTraversalCancelled(cancel);
    const auto found = std::find(objectUuids.begin(), objectUuids.end(), formatUuid(parseUuid(objectUuid)));
    if (found == objectUuids.end()) throw std::invalid_argument("object not in graph: " + objectUuid);
    const auto start = static_cast<size_t>(found - objectUuids.begin());
    const auto nodes = breadthFirst(start, maxDepth, cancel, [&](size_t node, auto visit) {
        for (auto i = offsets[node]; i < offsets[node+1]; ++i) visit(neighbours[i], [] { return true; });
    });
    std::vector<std::string> result; result.reserve(nodes.size());
    for (auto node : nodes) result.push_back(objectUuids[node]);
    return result;
}

std::vector<std::string> GraphReadView::componentOf(const std::string& objectUuid, const std::atomic_bool* cancel) const
{
    return reachableWithin(objectUuid, INT_MAX, cancel);
}

} // namespace vc::objects
