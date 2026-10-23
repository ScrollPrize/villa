#include "internal/EdgeSql.hpp"
#include "internal/SpatialCache.hpp"
#include "internal/CoordinateFilter.hpp"
#include "internal/VolumeFrames.hpp"

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
SelectionFilter objectFilter(sqlite3* db, const ObjectSelection& selection, const FrameFilter* frame)
{
    if (selection.volume || selection.includeUnbound)
        throw std::invalid_argument("volume targets apply to object selections, spatial regions and geometry regions");
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
        const auto ids = componentIds(db, *selection.connectedTo, selection.edgeKinds, selection.includeRetired);
        connected.insert(ids.begin(), ids.end());
    }
    SelectionFilter filter;
    if (!selection.includeRetired) filter.sql = " AND " + activeObjectSql("o.id");
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
    if (selection.coordinates) {
        const auto parameter = "?" + std::to_string(filter.parameters.size() + 1);
        bind(coordinateFilterSql("o", parameter), coordinateFilterValue(*selection.coordinates));
    }
    if (selection.tag) bind(" AND o.id IN (SELECT object_id FROM object_tags INDEXED BY object_tags_tag WHERE tag=?)", *selection.tag);
    if (selection.family) bind(" AND d.family=?", *selection.family);
    if (selection.minLength) bind(" AND d.length>=?", nlohmann::json(*selection.minLength).dump());
    if (frame && frame->unboundVolume) {
        bind(" AND NOT (json_extract(o.meta,'$.coordinates.volume_uuid') IS ?", *frame->unboundVolume);
        filter.sql += " AND COALESCE(json_extract(o.meta,'$.coordinates.scale_factor')>0,0))";
    }
    if (frame && frame->cursor) {
        // Volume order within a frame: shorter volume length, or the same
        // volume length and either a shorter stored length or a later UUID.
        const auto& c = *frame->cursor;
        auto hex = formatUuid(parseUuid(c.uuid));
        hex.erase(std::remove(hex.begin(), hex.end(), '-'), hex.end());
        bind(" AND (d.length<?", nlohmann::json(c.tieLow).dump());
        bind(" OR (d.length>=?", nlohmann::json(c.tieLow).dump());
        bind(" AND d.length<=?", nlohmann::json(c.tieHigh).dump());
        bind(" AND (d.length<?", nlohmann::json(c.stored).dump());
        bind(" OR (d.length=?", nlohmann::json(c.stored).dump());
        filter.sql += " AND o.uuid>X'" + hex + "'))))";
    }
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

namespace {
void requireVolumeSelection(const ObjectSelection& selection)
{
    if (selection.includeUnbound && !selection.volume)
        throw std::invalid_argument("include_unbound requires a volume");
    if (selection.volume && selection.coordinates)
        throw std::invalid_argument("volume and coordinates filters are exclusive");
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (selection.afterUuid && selection.offset)
        throw std::invalid_argument("cursor and offset cannot be combined");
    const bool tail = selection.includeUnbound && selection.orderBy == "length_desc" && selection.afterUuid;
    if (selection.orderBy == "length_desc" && bool(selection.afterUuid) != bool(selection.afterLength) && !tail)
        throw std::invalid_argument("length cursor requires after_uuid and after_length with length_desc");
}
// One frame of a volume selection: the caller's filters in that frame's stored units.
ObjectSelection frameSelection(const ObjectSelection& selection, const VolumeFrame& frame)
{
    auto sub = selection;
    sub.volume.reset(); sub.includeUnbound = false;
    sub.coordinates = frame.coordinates;
    sub.afterUuid.reset(); sub.afterLength.reset();
    sub.offset = 0;
    if (selection.limit) sub.limit = *selection.limit + selection.offset;
    if (selection.bbox) {
        validateBounds(*selection.bbox);
        sub.bbox = storedBox(*selection.bbox, frame.toVolume);
    }
    if (selection.minLength) {
        if (!std::isfinite(*selection.minLength) || *selection.minLength < 0)
            throw std::invalid_argument("min_length must be finite and nonnegative");
        sub.minLength = lowestStoredAtLeast(*selection.minLength, frame.toVolume);
    }
    return sub;
}
struct VolumeRow {
    std::string uuid;
    double stored = 0, volume = 0;
    bool bound = true;
};
// The volume selection's UUIDs in its order, after the cursor or offset.
std::vector<std::string> volumeSelection(sqlite3* db, const ObjectSelection& selection)
{
    requireVolumeSelection(selection);
    const bool byLength = selection.orderBy == "length_desc";
    const bool inTail = byLength && selection.afterUuid && !selection.afterLength;
    std::optional<double> cursorStored;
    if (byLength && selection.afterLength) {
        if (!std::isfinite(*selection.afterLength)) throw std::invalid_argument("after_length must be finite");
        if (!selection.kind || (*selection.kind != "fiber" && *selection.kind != "track"))
            throw std::invalid_argument("typed catalog filters require fiber or track kind");
        // Equal volume lengths order by stored length: the cursor's own.
        Statement cursor(db, std::string("SELECT d.length FROM objects o JOIN ") +
                             (*selection.kind == "fiber" ? "fibers" : "tracks") + " d ON d.object_id=o.id WHERE o.uuid=?",
                         parseUuid(*selection.afterUuid));
        if (!cursor.row()) throw std::invalid_argument("after_uuid does not name a cataloged object");
        cursorStored = cursor.f64(0);
    }
    std::vector<VolumeRow> rows;
    if (!inTail) for (const auto& frame : volumeFrames(db, *selection.volume, selection.kind, selection.includeRetired)) {
        auto sub = frameSelection(selection, frame);
        FrameFilter filter;
        if (cursorStored) {
            filter.cursor = FrameCursor{lowestStoredAtLeast(*selection.afterLength, frame.toVolume),
                                        highestStoredAtMost(*selection.afterLength, frame.toVolume),
                                        *cursorStored, *selection.afterUuid};
        } else if (selection.afterUuid) {
            sub.afterUuid = selection.afterUuid;
        }
        auto query = objectSelectionQuery(db, sub, byLength ? "o.uuid,d.length" : "o.uuid", &filter);
        while (query->row()) {
            VolumeRow row{formatUuid(query->blob16(0))};
            if (byLength) { row.stored = query->f64(1); row.volume = row.stored * frame.toVolume; }
            rows.push_back(std::move(row));
        }
    }
    // Unbound objects have no volume position or length.
    if (selection.includeUnbound && !selection.bbox && !(selection.minLength && *selection.minLength > 0)) {
        auto sub = selection;
        sub.volume.reset(); sub.includeUnbound = false;
        sub.orderBy = "uuid"; sub.minLength.reset(); sub.afterLength.reset(); sub.offset = 0;
        if (selection.limit) sub.limit = *selection.limit + selection.offset;
        if (byLength && !inTail) sub.afterUuid.reset();
        FrameFilter filter;
        filter.unboundVolume = *selection.volume;
        auto query = objectSelectionQuery(db, sub, "o.uuid", &filter);
        while (query->row()) rows.push_back({formatUuid(query->blob16(0)), 0, 0, false});
    }
    std::sort(rows.begin(), rows.end(), [byLength](const VolumeRow& a, const VolumeRow& b) {
        if (byLength) {
            if (a.bound != b.bound) return a.bound;
            if (a.bound && a.volume != b.volume) return a.volume > b.volume;
            if (a.bound && a.stored != b.stored) return a.stored > b.stored;
        }
        return a.uuid < b.uuid;
    });
    const size_t begin = std::min(rows.size(), size_t(selection.offset));
    const size_t end = selection.limit ? std::min(rows.size(), begin + size_t(*selection.limit)) : rows.size();
    std::vector<std::string> uuids;
    for (size_t i = begin; i < end; ++i) uuids.push_back(std::move(rows[i].uuid));
    return uuids;
}
}

std::unique_ptr<Statement> detail::objectSelectionQuery(sqlite3* db, const ObjectSelection& selection,
                                                       const std::string& columns, const FrameFilter* frame)
{
    auto filter = objectFilter(db, selection, frame);
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
    if (!selection.volume && !selection.includeUnbound) {
        auto rows = objectSelectionQuery(db, selection, projectedObjectColumns(fields));
        return readObjectRows(db, *rows, fields);
    }
    const auto order = volumeSelection(db, selection);
    if (order.empty()) return {};
    ObjectSelection chosen;
    chosen.uuids = order;
    chosen.includeRetired = selection.includeRetired;
    auto rows = objectSelectionQuery(db, chosen, projectedObjectColumns(fields));
    auto records = readObjectRows(db, *rows, fields);
    std::map<std::string, size_t> position;
    for (size_t i = 0; i < order.size(); ++i) position.emplace(order[i], i);
    std::sort(records.begin(), records.end(), [&](const ObjectRecord& a, const ObjectRecord& b) {
        return position.at(a.uuid) < position.at(b.uuid);
    });
    return records;
}

std::vector<SpatialRegion> ObjectStore::spatialRegions(const Txn& txn, const ObjectSelection& selection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (selection.orderBy != "uuid") throw std::invalid_argument("spatial regions use UUID order");
    if (!selection.bbox) throw std::invalid_argument("spatial regions require a bbox");
    detail::validateBounds(*selection.bbox);
    if (selection.includeUnbound) throw std::invalid_argument("spatial regions place only objects bound to a volume");
    if (selection.volume) {
        requireVolumeSelection(selection);
        std::vector<SpatialRegion> regions;
        for (const auto& frame : volumeFrames(db, *selection.volume, selection.kind, selection.includeRetired)) {
            auto sub = frameSelection(selection, frame);
            sub.afterUuid = selection.afterUuid;
            for (auto region : spatialRegions(txn, sub)) {
                // Mapped stored bounds stay conservative: the map is monotonic.
                for (auto& bound : region.bbox) bound *= frame.toVolume;
                regions.push_back(std::move(region));
            }
        }
        std::sort(regions.begin(), regions.end(), [](const SpatialRegion& a, const SpatialRegion& b) {
            return std::tie(a.objectUuid, a.arrayName, a.start) < std::tie(b.objectUuid, b.arrayName, b.start);
        });
        const size_t begin = std::min(regions.size(), size_t(selection.offset));
        const size_t end = selection.limit ? std::min(regions.size(), begin + size_t(*selection.limit)) : regions.size();
        return {std::make_move_iterator(regions.begin() + begin), std::make_move_iterator(regions.begin() + end)};
    }
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (const auto lease = activeStatementPools.find(db); lease != activeStatementPools.end())
        if (const auto& cache = lease->second->spatialCache)
            if (auto result = cache->query(db, selection)) return *result;
    const auto filter = objectFilter(db, selection, nullptr);
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
