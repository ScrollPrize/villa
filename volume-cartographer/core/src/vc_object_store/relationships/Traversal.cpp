#include "internal/EdgeSql.hpp"

namespace vc::objects {
using namespace detail;
std::vector<std::string> ObjectStore::neighbours(const Txn& txn, const std::string& objectUuid,
                                             const std::optional<std::string>& edgeKind,
                                             const std::optional<std::string>& neighbourKind) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("object store: object not found: " + objectUuid);
    const std::string filter = edgeKind ? " AND kind=?2" : "";
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

std::vector<std::string> traverse(sqlite3* db, const std::string& startUuid, int maxDepth,
                                  const std::vector<std::string>& edgeKindFilter, const std::atomic_bool* cancel)
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
    // Probe the endpoint indexes only for nodes reached by this traversal.
    // Visit neighbours in stable internal-ID order.
    std::string filter;
    if (!edgeKindFilter.empty()) {
        filter = " AND kind IN (";
        for (size_t i = 0; i < edgeKindFilter.size(); ++i)
            filter += (i ? ",?" : "?") + std::to_string(i + 2);
        filter += ")";
    }
    // Each UNION arm uses an endpoint-leading index even without ANALYZE.
    std::string sql = "SELECT id,b_id FROM edges INDEXED BY edges_a WHERE a_id=?1" + filter +
        " UNION SELECT id,a_id FROM edges INDEXED BY edges_b WHERE b_id=?1" + filter + " ORDER BY id";
    Statement adjacent(db, sql);
    const int64_t startId = *objectIdByUuid(db, startBytes);
    std::vector<std::string> out;
    std::unordered_set<int64_t> visited{startId};
    std::queue<std::pair<int64_t, int>> queue;
    queue.emplace(startId, 0);
    while (!queue.empty()) {
        checkCancelled();
        auto [id, depth] = queue.front();
        queue.pop();
        out.push_back(uuidTextByObjectId(db, id));
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

}  // namespace

std::vector<std::string> ObjectStore::componentOf(const Txn& txn, const std::string& objectUuid,
                                              const std::vector<std::string>& edgeKindFilter,
                                              const std::atomic_bool* cancel) const
{
    return traverse(static_cast<sqlite3*>(connection(txn)), objectUuid, INT_MAX, edgeKindFilter, cancel);
}

std::vector<std::string> ObjectStore::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel) const
{
    if (maxDepth < 0) throw std::runtime_error("object store: maxDepth must be >= 0");
    return traverse(static_cast<sqlite3*>(connection(txn)), objectUuid, maxDepth, edgeKindFilter, cancel);
}

} // namespace vc::objects
