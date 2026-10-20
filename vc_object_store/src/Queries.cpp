#include "internal/EdgeSql.hpp"
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
SelectionFilter objectFilter(sqlite3* db, const ObjectSelection& selection)
{
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (selection.orderBy != "uuid" && selection.orderBy != "length_desc")
        throw std::invalid_argument("order_by must be uuid or length_desc");
    if (selection.family || selection.minLength || selection.orderBy == "length_desc")
        if (!selection.kind || (*selection.kind != "fiber" && *selection.kind != "track"))
            throw std::invalid_argument("typed catalog filters require fiber or track kind");
    if (selection.minLength && (!std::isfinite(*selection.minLength) || *selection.minLength < 0))
        throw std::invalid_argument("min_length must be finite and nonnegative");
    if (selection.afterLength && !std::isfinite(*selection.afterLength))
        throw std::invalid_argument("after_length must be finite");
    if (selection.orderBy == "length_desc" ? bool(selection.afterUuid) != bool(selection.afterLength) : bool(selection.afterLength))
        throw std::invalid_argument("length cursor requires after_uuid and after_length with length_desc");
    if (selection.afterUuid && selection.offset)
        throw std::invalid_argument("cursor and offset cannot be combined");
    std::set<int64_t> selected, connected;
    for (const auto& uuid : selection.uuids)
        if (const auto id = objectIdByUuid(db, parseUuid(uuid))) selected.insert(*id);
    if (selection.connectedTo) {
        const auto ids = componentIds(db, *selection.connectedTo, selection.edgeKinds);
        connected.insert(ids.begin(), ids.end());
    }
    SelectionFilter filter;
    auto bind = [&](const std::string& clause, const std::string& value) {
        filter.sql += clause; filter.parameters.push_back(value);
    };
    if (!selection.uuids.empty() || selection.connectedTo) {
        nlohmann::json ids = nlohmann::json::array();
        const auto& candidates = selection.uuids.empty() ? connected : selected;
        for (const auto id : candidates) {
            if (selection.connectedTo && !connected.contains(id)) continue;
            ids.push_back(id);
        }
        bind(" AND o.id IN (SELECT value FROM json_each(?))", ids.dump());
    }
    if (selection.kind) bind(" AND o.kind=?", *selection.kind);
    if (selection.tag) bind(" AND o.id IN (SELECT object_id FROM object_tags INDEXED BY object_tags_tag WHERE tag=?)", *selection.tag);
    if (selection.family) bind(" AND d.family=?", *selection.family);
    if (selection.minLength) bind(" AND d.length>=?", nlohmann::json(*selection.minLength).dump());
    if (selection.afterUuid) {
        auto hex = formatUuid(parseUuid(*selection.afterUuid));
        hex.erase(std::remove(hex.begin(), hex.end(), '-'), hex.end());
        const auto uuid = "X'" + hex + "'";
        if (selection.orderBy == "length_desc") {
            bind(" AND d.length<=?", nlohmann::json(*selection.afterLength).dump());
            bind(" AND (d.length<?", nlohmann::json(*selection.afterLength).dump());
            filter.sql += " OR o.uuid>" + uuid + ")";
        } else filter.sql += " AND o.uuid>" + uuid;
    }
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

std::unique_ptr<Statement> detail::objectSelectionQuery(sqlite3* db, const ObjectSelection& selection,
                                                       const std::string& columns)
{
    auto filter = objectFilter(db, selection);
    const bool candidates = selection.bbox || !selection.uuids.empty() || selection.connectedTo || selection.tag;
    // Force primary-key candidate probes instead of scanning objects_kind_uuid
    // to avoid a sort when the R-tree returned only a few matches.
    const bool typed = selection.family || selection.minLength || selection.orderBy == "length_desc";
    const auto table = selection.kind == "fiber" ? "fibers" : "tracks";
    std::string from = "objects o" + std::string(candidates ? " NOT INDEXED" : "");
    if (typed) {
        if (!candidates) from = std::string(table) + " d CROSS JOIN objects o ON o.id=d.object_id";
        else from += std::string(" JOIN ") + table + " d ON d.object_id=o.id";
    }
    std::string sql = "SELECT " + columns + " FROM " + from + " WHERE 1" + filter.sql;
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
    sql += std::string(" ORDER BY ") + (selection.orderBy == "length_desc" ? "d.length DESC,o.uuid" : "o.uuid") + " LIMIT ? OFFSET ?";
    auto rows = std::make_unique<Statement>(db, sql);
    bindSelection(*rows, filter, selection);
    return rows;
}

std::vector<ObjectRecord> ObjectStore::selectObjects(const Txn& txn, const ObjectSelection& selection,
    const std::optional<std::vector<std::string>>& projection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const ObjectFields fields = projection ? ObjectFields(std::in_place, projection->begin(), projection->end()) : std::nullopt;
    auto rows = objectSelectionQuery(db, selection, projectedObjectColumns(fields));
    return readObjectRows(db, *rows, fields);
}

std::vector<SpatialRegion> ObjectStore::spatialRegions(const Txn& txn, const ObjectSelection& selection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (selection.orderBy != "uuid") throw std::invalid_argument("spatial regions use UUID order");
    if (!selection.bbox) throw std::invalid_argument("spatial regions require a bbox");
    detail::validateBounds(*selection.bbox);
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (const auto lease = activeStatementPools.find(db); lease != activeStatementPools.end())
        if (const auto& cache = lease->second->spatialCache)
            if (auto result = cache->query(db, selection)) return std::move(*result);
    if (pragmaInt(db, "user_version") < 5)
        throw std::runtime_error("spatial regions require schema 5; open the store writable once to migrate");
    const auto filter = objectFilter(db, selection);
    // CROSS JOIN preserves the R-tree-first plan even with a kind filter.
    std::string sql = std::string("SELECT ") + detail::regionColumns + " FROM spatial_regions_rtree r "
        "CROSS JOIN spatial_regions b ON b.id=r.id CROSS JOIN objects o ON o.id=b.object_id "
        "CROSS JOIN object_bounds h ON h.object_id=b.object_id " +
        ((selection.family || selection.minLength) ?
            std::string(" JOIN ") + (selection.kind == "fiber" ? "fibers" : "tracks") + " d ON d.object_id=o.id " : "") +
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
