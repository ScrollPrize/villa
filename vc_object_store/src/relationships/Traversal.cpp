#include "internal/EdgeSql.hpp"
#include "internal/History.hpp"

namespace vc::objects {
using namespace detail;
std::vector<std::string> ObjectStore::neighbours(const Txn& txn, const std::string& objectUuid,
                                             const std::optional<std::string>& edgeKind,
                                             const std::optional<std::string>& neighbourKind, bool includeRetired,
                                             std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("object store: object not found: " + objectUuid);
    if (applicableOnly) requirePublishedGraphRead(txn);
    const std::string filter = fiberAdjacencySql(adjacentFilter) + (edgeKind ? std::string(" AND kind=?2") : "") +
        (includeRetired ? "" : " AND " + activeEdgeSql(db));
    // Filter before deduplication: a rejected parallel edge must not hide a
    // later applicable edge to the same neighbour. Preserve first-edge order.
    std::string sql = "SELECT o.uuid,e.uuid FROM (SELECT id,uuid,b_id AS neighbour FROM edges INDEXED BY edges_a WHERE a_id=?1" + filter +
        " UNION SELECT id,uuid,a_id AS neighbour FROM edges INDEXED BY edges_b WHERE b_id=?1" + filter +
        ") e JOIN objects o ON o.id=e.neighbour";
    if (neighbourKind) sql += " WHERE o.kind=?3";
    sql += " ORDER BY e.id";
    Statement rows(db, sql);
    rows.bind(1, *id);
    if (edgeKind) rows.bind(2, *edgeKind);
    if (neighbourKind) rows.bind(3, *neighbourKind);
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    while (rows.row()) {
        const auto next = formatUuid(rows.blob16(0));
        if (seen.contains(next)) continue;
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

std::vector<int64_t> traverse(sqlite3* db, const std::string& startUuid, int maxDepth,
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
    if (!objectIdByUuid(db, startBytes))
        throw std::runtime_error("object store: object not found: " + startUuid);
    if (!includeRetired) {
        Statement active(db, "SELECT 1 FROM objects WHERE uuid=? AND " + activeObjectSql(db, "objects.id"));
        active.bindBlob(1, startBytes);
        if (!active.row()) return {};
    }
    // Probe the endpoint indexes only for nodes reached by this traversal.
    // Visit neighbours in stable internal-ID order.
    std::string filter;
    if (!edgeKindFilter.empty()) {
        filter = " AND kind IN (";
        for (size_t i = 0; i < edgeKindFilter.size(); ++i)
            filter += (i ? ",?" : "?") + std::to_string(i + 2);
        filter += ")";
    }
    filter += fiberAdjacencySql(adjacentFilter);
    if (!includeRetired) filter += " AND " + activeEdgeSql(db);
    // Each UNION arm uses an endpoint-leading index even without ANALYZE.
    std::string sql = "SELECT id,b_id,uuid FROM edges INDEXED BY edges_a WHERE a_id=?1" + filter +
        " UNION SELECT id,a_id,uuid FROM edges INDEXED BY edges_b WHERE b_id=?1" + filter + " ORDER BY id";
    Statement adjacent(db, sql);
    const int64_t startId = *objectIdByUuid(db, startBytes);
    try {
        return breadthFirst(startId, maxDepth, cancel, [&](int64_t id, auto visit) {
            adjacent.reset(); adjacent.bind(1, id);
            for (size_t i = 0; i < edgeKindFilter.size(); ++i) adjacent.bind(int(i) + 2, edgeKindFilter[i]);
            while (adjacent.row())
                visit(adjacent.i64(1), [&] {
                    return !store || currentRelationshipApplicable(*store, *txn, formatUuid(adjacent.blob16(2)));
                });
        });
    } catch (const std::runtime_error&) {
        checkCancelled(); // Translate SQLite interrupts into the cancellation contract.
        throw;
    }
}

std::vector<std::string> uuidList(sqlite3* db, const std::vector<int64_t>& ids)
{
    Statement rows(db, "SELECT id,uuid FROM objects WHERE id IN (SELECT value FROM json_each(?))");
    rows.bind(1, nlohmann::json(ids).dump());
    std::unordered_map<int64_t, std::string> uuids;
    while (rows.row()) uuids.emplace(rows.i64(0), formatUuid(rows.blob16(1)));
    std::vector<std::string> result;
    result.reserve(ids.size());
    for (auto id : ids) result.push_back(uuids.at(id));
    return result;
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
    auto ids = traverse(db, start, INT_MAX, kinds, nullptr, includeRetired, adjacentFilter);
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
    return uuidList(db, (cancel || applicableOnly) ? traverse(db, objectUuid, INT_MAX, edgeKindFilter, cancel, includeRetired,
                                                           adjacentFilter, applicableOnly ? this : nullptr, &txn)
                              : componentIds(db, objectUuid, edgeKindFilter, includeRetired, adjacentFilter));
}

std::vector<std::string> ObjectStore::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel, bool includeRetired,
                                              std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    if (maxDepth < 0) throw std::runtime_error("object store: maxDepth must be >= 0");
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (applicableOnly) requirePublishedGraphRead(txn);
    return uuidList(db, traverse(db, objectUuid, maxDepth, edgeKindFilter, cancel, includeRetired,
                                adjacentFilter, applicableOnly ? this : nullptr, &txn));
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
    Statement journal(db, "SELECT coalesce(max(seq),0) FROM journal");
    (void)journal.row(); result.journalSequence = journal.i64(0);
}
}

GraphReadView ObjectStore::graphReadView(const Txn& txn, const ObjectSelection& objects,
                                       const EdgeSelection& edges) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    GraphReadView result;
    if (edges.endpointUuids)
        throw std::invalid_argument("graph endpoint scope is specified by its object selection");
    std::unordered_map<int64_t, uint64_t> nodes;
    std::vector<int64_t> ids;
    auto objectsQuery = objectSelectionQuery(db, objects,
        "o.id,o.uuid,o.generation,(SELECT revision_uuid FROM geometry_heads WHERE object_uuid=o.uuid)");
    while (objectsQuery->row()) {
        const auto node = result.objectUuids.size();
        const auto id = objectsQuery->i64(0);
        nodes.emplace(id, node); ids.push_back(id);
        result.objectUuids.push_back(formatUuid(objectsQuery->blob16(1)));
        result.generations.push_back(objectsQuery->i64(2));
        result.geometryRevisions.push_back(objectsQuery->isNull(3) ? std::nullopt :
            std::optional(formatUuid(objectsQuery->blob16(3))));
    }
    auto selection = edges; selection.bothEndpoints = true;
    selection.includeRetired = objects.includeRetired;
    if (selection.applicableOnly) {
        selection.endpointUuids = result.objectUuids;
        for (const auto& edge : selectEdges(txn, selection, std::vector<std::string>{})) {
            result.edgeUuids.push_back(edge.uuid);
            result.endpoints.push_back({nodes.at(*objectIdByUuid(db, parseUuid(edge.aUuid))),
                                        nodes.at(*objectIdByUuid(db, parseUuid(edge.bUuid)))});
        }
    } else {
        auto links = edgeSelectionQuery(db, selection, "uuid,a_id,b_id", ids);
        while (links->row()) {
            result.edgeUuids.push_back(formatUuid(links->blob16(0)));
            result.endpoints.push_back({nodes.at(links->i64(1)), nodes.at(links->i64(2))});
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
    for (auto& selected : relationshipsAt(txn, resolved))
        if (selected.applicable && matchesEdgeSelection(selected.relationship.record, edges))
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
