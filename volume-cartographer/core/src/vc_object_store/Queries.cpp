#include "internal/RecordSql.hpp"
#include "internal/SpatialCache.hpp"

namespace vc::objects {
using namespace detail;
void detail::beginSpatialRead(StatementPool& pool, size_t budget, int version)
{
    if (!budget) { pool.spatialCache.reset(); return; }
    if (!pool.spatialCache) pool.spatialCache = std::make_shared<SpatialReadCache>(budget);
    pool.spatialCache->begin(version);
}

namespace {
struct SelectionFilter {
    std::string sql;
    std::vector<std::string> parameters;
};
SelectionFilter objectFilter(const ObjectStore& store, const Txn& txn, sqlite3* db, const ObjectSelection& selection)
{
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    std::set<std::string> selected, connected;
    for (const auto& id : selection.uuids) selected.insert(formatUuid(parseUuid(id)));
    if (selection.connectedTo)
        for (const auto& id : store.componentOf(txn, *selection.connectedTo, selection.edgeKinds)) connected.insert(id);
    SelectionFilter filter;
    auto bind = [&](const std::string& clause, const std::string& value) {
        filter.sql += clause; filter.parameters.push_back(value);
    };
    if (!selected.empty() || selection.connectedTo) {
        nlohmann::json ids = nlohmann::json::array();
        const auto& candidates = selected.empty() ? connected : selected;
        for (const auto& uuid : candidates) {
            if (selection.connectedTo && !connected.contains(uuid)) continue;
            if (auto id = objectIdByUuid(db, parseUuid(uuid))) ids.push_back(*id);
        }
        bind(" AND o.id IN (SELECT value FROM json_each(?))", ids.dump());
    }
    if (selection.kind) bind(" AND o.kind=?", *selection.kind);
    if (selection.tag) bind(" AND o.id IN (SELECT object_id FROM object_tags INDEXED BY object_tags_tag WHERE tag=?)", *selection.tag);
    return filter;
}
void bindSelection(Statement& rows, const SelectionFilter& filter, const ObjectSelection& selection)
{
    int next = 1;
    for (const auto& value : filter.parameters) rows.bind(next++, value);
    if (selection.bbox) for (double coordinate : *selection.bbox) rows.bind(next++, coordinate);
    rows.bind(next++, selection.limit.value_or(-1));
    rows.bind(next, selection.offset);
}
}

std::vector<ObjectRecord> ObjectStore::selectObjects(const Txn& txn, const ObjectSelection& selection,
    const std::optional<std::vector<std::string>>& projection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const ObjectFields fields = projection ? ObjectFields(std::in_place, projection->begin(), projection->end()) : std::nullopt;
    const auto columns = projectedObjectColumns(fields);
    auto filter = objectFilter(*this, txn, db, selection);
    const bool candidates = selection.bbox || !selection.uuids.empty() || selection.connectedTo || selection.tag;
    // Force primary-key candidate probes instead of scanning objects_kind_uuid
    // to avoid a sort when the R-tree returned only a few matches.
    std::string sql = "SELECT " + columns + " FROM objects o" + (candidates ? " NOT INDEXED" : "") + " WHERE 1" + filter.sql;
    if (selection.bbox) {
        detail::validateBounds(*selection.bbox);
        const auto version = pragmaInt(db, "user_version");
        if (version < 4)
            throw std::runtime_error("spatial queries require schema 4; open the store writable once to migrate");
        if (version == 4) // Read-only compatibility with the original object index.
            sql += " AND o.id IN (SELECT b.object_id FROM object_bounds_rtree r CROSS JOIN object_bounds b ON b.object_id=r.object_id WHERE ";
        else
            sql += " AND o.id IN (SELECT b.object_id FROM spatial_regions_rtree r CROSS JOIN spatial_regions b ON b.id=r.id WHERE b.array_name='' AND ";
        sql += detail::spatialOverlapSql(int(filter.parameters.size()) + 1) + ")";
    }
    sql += " ORDER BY uuid LIMIT ? OFFSET ?";
    Statement rows(db, sql);
    bindSelection(rows, filter, selection);
    return readObjectRows(db, rows, fields);
}

std::vector<SpatialRegion> ObjectStore::spatialRegions(const Txn& txn, const ObjectSelection& selection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (!selection.bbox) throw std::invalid_argument("spatial regions require a bbox");
    detail::validateBounds(*selection.bbox);
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (const auto lease = activeStatementPools.find(db); lease != activeStatementPools.end())
        if (const auto& cache = lease->second->spatialCache)
            if (auto result = cache->query(db, selection)) return std::move(*result);
    if (pragmaInt(db, "user_version") < 5)
        throw std::runtime_error("spatial regions require schema 5; open the store writable once to migrate");
    const auto filter = objectFilter(*this, txn, db, selection);
    // CROSS JOIN preserves the R-tree-first plan even with a kind filter.
    std::string sql = std::string("SELECT ") + detail::regionColumns + " FROM spatial_regions_rtree r "
        "CROSS JOIN spatial_regions b ON b.id=r.id CROSS JOIN objects o ON o.id=b.object_id "
        "CROSS JOIN object_bounds h ON h.object_id=b.object_id "
        "WHERE (b.array_name!='' OR h.revision_uuid IS NULL)" + filter.sql + " AND " +
        detail::spatialOverlapSql(int(filter.parameters.size()) + 1) +
        " ORDER BY o.uuid,b.array_name,b.row_start,b.col_start LIMIT ? OFFSET ?";
    Statement rows(db, sql);
    bindSelection(rows, filter, selection);
    std::vector<SpatialRegion> result;
    while (rows.row()) result.push_back(detail::readSpatialRegion(rows));
    return result;
}

} // namespace vc::objects
