#include "internal/EdgeSql.hpp"

namespace vc::objects {
using namespace detail;
std::vector<std::string> ObjectStore::neighbours(const Txn& txn, const std::string& objectUuid,
                                             const std::optional<std::string>& edgeKind,
                                             const std::optional<std::string>& neighbourKind, bool includeRetired) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("object store: object not found: " + objectUuid);
    const std::string filter = (edgeKind ? std::string(" AND kind=?2") : "") +
        (includeRetired ? "" : " AND " + activeEdgeSql(db));
    // Order by the first incident edge, as edgesOf() did, without hydrating edges.
    std::string sql = "SELECT o.uuid FROM (SELECT id,b_id AS neighbour FROM edges INDEXED BY edges_a WHERE a_id=?1" + filter +
        " UNION SELECT id,a_id AS neighbour FROM edges INDEXED BY edges_b WHERE b_id=?1" + filter +
        ") e JOIN objects o ON o.id=e.neighbour";
    if (neighbourKind) sql += " WHERE o.kind=?3";
    sql += " GROUP BY o.id ORDER BY min(e.id)";
    Statement rows(db, sql);
    rows.bind(1, *id);
    if (edgeKind) rows.bind(2, *edgeKind);
    if (neighbourKind) rows.bind(3, *neighbourKind);
    std::vector<std::string> out;
    while (rows.row()) out.push_back(formatUuid(rows.blob16(0)));
    return out;
}

namespace {

struct ProgressGuard {
    sqlite3* db;
    ~ProgressGuard() { sqlite3_progress_handler(db, 0, nullptr, nullptr); }
};

std::vector<int64_t> traverse(sqlite3* db, const std::string& startUuid, int maxDepth,
                                  const std::vector<std::string>& edgeKindFilter, const std::atomic_bool* cancel, bool includeRetired)
{
    auto checkCancelled = [&] {
        if (cancel && cancel->load())
            throw std::runtime_error("object store: traversal cancelled");
    };
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
    if (!includeRetired) filter += " AND " + activeEdgeSql(db);
    // Each UNION arm uses an endpoint-leading index even without ANALYZE.
    std::string sql = "SELECT id,b_id FROM edges INDEXED BY edges_a WHERE a_id=?1" + filter +
        " UNION SELECT id,a_id FROM edges INDEXED BY edges_b WHERE b_id=?1" + filter + " ORDER BY id";
    Statement adjacent(db, sql);
    const int64_t startId = *objectIdByUuid(db, startBytes);
    std::vector<int64_t> out;
    std::unordered_set<int64_t> visited{startId};
    std::queue<std::pair<int64_t, int>> queue;
    queue.emplace(startId, 0);
    while (!queue.empty()) {
        checkCancelled();
        auto [id, depth] = queue.front();
        queue.pop();
        out.push_back(id);
        if (depth >= maxDepth) continue;
        adjacent.reset();
        adjacent.bind(1, id);
        for (size_t i = 0; i < edgeKindFilter.size(); ++i) adjacent.bind(int(i) + 2, edgeKindFilter[i]);
        try {
            while (adjacent.row()) {
                checkCancelled();
                const auto next = adjacent.i64(1);
                if (visited.insert(next).second) queue.emplace(next, depth + 1);
            }
        } catch (const std::runtime_error&) {
            checkCancelled(); // translate SQLite's interrupt into the cancellation contract
            throw;
        }
    }
    return out;
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
                                        const std::vector<std::string>& kinds, bool includeRetired)
{
    StatementPool* pool = nullptr;
    if (auto it = activeStatementPools.find(db); it != activeStatementPools.end() && it->second->readSnapshot)
        pool = it->second;
    const auto key = nlohmann::json({formatUuid(parseUuid(start)), kinds, includeRetired}).dump();
    if (pool) if (auto it = pool->components.find(key); it != pool->components.end()) return it->second;
    auto ids = traverse(db, start, INT_MAX, kinds, nullptr, includeRetired);
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
                                              const std::atomic_bool* cancel, bool includeRetired) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    return uuidList(db, cancel ? traverse(db, objectUuid, INT_MAX, edgeKindFilter, cancel, includeRetired)
                              : componentIds(db, objectUuid, edgeKindFilter, includeRetired));
}

std::vector<std::string> ObjectStore::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel, bool includeRetired) const
{
    if (maxDepth < 0) throw std::runtime_error("object store: maxDepth must be >= 0");
    auto* db = static_cast<sqlite3*>(connection(txn));
    return uuidList(db, traverse(db, objectUuid, maxDepth, edgeKindFilter, cancel, includeRetired));
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
    auto links = edgeSelectionQuery(db, selection, "uuid,a_id,b_id", ids);
    while (links->row()) {
        result.edgeUuids.push_back(formatUuid(links->blob16(0)));
        result.endpoints.push_back({nodes.at(links->i64(1)), nodes.at(links->i64(2))});
    }
    result.offsets.resize(nodes.size() + 1);
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
    return result;
}

} // namespace vc::objects
