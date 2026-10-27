#include "internal/CoordinateFilter.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "geometry/ArrayLayout.hpp"
#include "geometry/GeometryReader.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include <algorithm>
#include <bit>
#include <set>
#include <tuple>

namespace vc::objects::detail {
nlohmann::json coordinateBounds(std::span<const uint8_t> bytes,const std::string& dtype,bool surface) {
    GeometryPhase phase("coordinateBounds");
    std::optional<std::array<double,6>> result;
    auto scan=[&]<class T>() {
        if (bytes.size()%(3*sizeof(T))) throw std::runtime_error("unaligned coordinate array");
        for (size_t offset=0;offset<bytes.size();offset+=3*sizeof(T)) {
            std::array<T,3> point; copyScalars<T>(bytes.subspan(offset,3*sizeof(T)),point);
            const double x=point[0],y=point[1],z=point[2];
            if (!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)) continue;
            if (surface&&!vc::geometry::validSurfacePoint({float(x),float(y),float(z)})) continue;
            if(!result)result=std::array<double,6>{x,y,z,x,y,z};
            else for(size_t axis=0;axis<3;++axis) {(*result)[axis]=std::min((*result)[axis],double(point[axis]));(*result)[axis+3]=std::max((*result)[axis+3],double(point[axis]));}
        }
    };
    if (dtype=="<f8") scan.template operator()<double>();
    else if (dtype=="<f4") scan.template operator()<float>();
    else if (dtype=="<i4") scan.template operator()<int32_t>();
    else throw std::runtime_error("unsupported coordinate dtype");
    return result?nlohmann::json(*result):nlohmann::json();
}
void validateBounds(const Bounds& bounds)
{
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(bounds[axis]) || !std::isfinite(bounds[axis + 3]) || bounds[axis] > bounds[axis + 3])
            throw std::invalid_argument("spatial bounds must be finite, ordered minXYZ,maxXYZ coordinates");
}

std::string spatialOverlapSql(int firstParameter)
{
    std::string sql;
    for (int axis = 0; axis < 3; ++axis) {
        const auto low = "?" + std::to_string(firstParameter + axis);
        const auto high = "?" + std::to_string(firstParameter + axis + 3);
        const std::string coordinate(1, "xyz"[axis]);
        if (axis) sql += " AND ";
        sql += "r.max_" + coordinate + ">=max(-1e30,min(1e30," + low + ")) AND r.min_" + coordinate +
            "<=max(-1e30,min(1e30," + high + ")) AND b.max_" + coordinate + ">=" + low +
            " AND b.min_" + coordinate + "<=" + high;
    }
    return sql;
}

std::vector<std::string> remoteSurfacesIn(sqlite3* db, const Bounds& bounds, const std::optional<std::string>& includedIds,
                                          const std::optional<std::string>& coordinates)
{
    std::string sql = "SELECT o.uuid FROM spatial_regions_rtree r CROSS JOIN spatial_regions b ON b.id=r.id "
        "CROSS JOIN objects o ON o.id=b.object_id LEFT JOIN geometry_heads h ON h.object_uuid=o.uuid "
        "LEFT JOIN object_lifecycle l ON l.object_id=o.id WHERE o.kind='surface' AND b.array_name='' "
        "AND h.object_uuid IS NULL AND o.storage_kind!='' AND COALESCE(l.retired,0)=0 AND " + spatialOverlapSql();
    if (includedIds) sql += " AND o.id IN (SELECT value FROM json_each(?8))";
    if (coordinates) sql += coordinateFilterSql("o", "?7");
    sql += " ORDER BY o.uuid";
    Statement rows(db, sql);
    for (int i = 0; i < 6; ++i) rows.bind(i + 1, bounds[i]);
    if (coordinates) rows.bind(7, coordinateFilterValue(*coordinates));
    if (includedIds) rows.bind(8, *includedIds);
    std::vector<std::string> uuids;
    while (rows.row()) uuids.push_back(formatUuid(rows.blob16(0)));
    return uuids;
}

std::optional<Bounds> recordBounds(const ObjectRecord& object)
{
    if (object.kind == "fiber" && object.fiber) return object.fiber->bbox;
    if (object.kind == "surface" && object.surface) return object.surface->bbox;
    if (object.kind == "track" && object.track) return object.track->bbox;
    return std::nullopt;
}

namespace {
struct BoundsBuilder {
    std::optional<Bounds> bounds;
    void merge(const Bounds& b) {
        if (!bounds) bounds = b;
        else for (size_t axis = 0; axis < 3; ++axis) {
            (*bounds)[axis] = std::min((*bounds)[axis], b[axis]);
            (*bounds)[axis + 3] = std::max((*bounds)[axis + 3], b[axis + 3]);
        }
    }
    void append(std::span<const uint8_t> bytes, const std::string& dtype, bool surface) {
        const auto value=coordinateBounds(bytes,dtype,surface);
        if(!value.is_null())merge(value.get<Bounds>());
    }
};
// Kinds whose points are a polyline: spatial blocks overlap by one point so
// every segment lies in a block.
bool polyline(const std::string& kind) { return kind == "fiber" || kind == "track" || kind == "umbilicus"; }
bool coordinates(const std::string& kind, const std::string& name) {
    return kind == "fiber" ? name == "line_points" || name == "control_points" : name == "points";
}
}

namespace {
void writeBounds(sqlite3* db, int64_t objectId, const std::optional<Bounds>& bounds,
                 const std::optional<std::string>& revision)
{
    if (!bounds) {
        execute(db, "DELETE FROM object_bounds WHERE object_id=?", objectId); return;
    }
    validateBounds(*bounds);
    Statement put(db, "INSERT INTO object_bounds VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(object_id) DO UPDATE SET "
        "revision_uuid=excluded.revision_uuid,min_x=excluded.min_x,min_y=excluded.min_y,min_z=excluded.min_z,"
        "max_x=excluded.max_x,max_y=excluded.max_y,max_z=excluded.max_z", objectId);
    if (revision) put.bind(2, parseUuid(*revision)); else put.bindNull(2);
    for (size_t i = 0; i < 6; ++i) put.bind(int(i) + 3, (*bounds)[i]);
    put.stepDone();
}

// Starts/counts exclude the final XYZ axis. Surfaces use rows and columns;
// one-dimensional arrays use row_start/count and zero column values.
struct Region {
    int64_t id = 0, rows = 0, columns = 0;
    Bounds bounds;
};
using RegionKey = std::tuple<std::string, int64_t, int64_t>;

class RegionWriter {
public:
    RegionWriter(sqlite3* db, int64_t objectId) : objectId_(objectId),
        put_(db, "INSERT INTO spatial_regions(object_id,array_name,row_start,col_start,row_count,col_count,"
            "min_x,min_y,min_z,max_x,max_y,max_z) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(object_id,array_name,row_start,col_start) DO UPDATE SET "
            "row_count=excluded.row_count,col_count=excluded.col_count,min_x=excluded.min_x,min_y=excluded.min_y,"
            "min_z=excluded.min_z,max_x=excluded.max_x,max_y=excluded.max_y,max_z=excluded.max_z"),
        remove_(db, "DELETE FROM spatial_regions WHERE object_id=? AND array_name=? AND row_start=? AND col_start=?") {}
    void update(const std::string& name, int64_t row, int64_t col, int64_t rows, int64_t cols,
                const std::optional<Bounds>& bounds) {
        auto& statement = bounds ? put_ : remove_;
        statement.reset(); statement.bind(1, objectId_); statement.bind(2, name); statement.bind(3, row); statement.bind(4, col);
        if (bounds) {
            statement.bind(5, rows); statement.bind(6, cols);
            for (int i = 0; i < 6; ++i) statement.bind(i+7, (*bounds)[i]);
        }
        statement.stepDone();
    }
private:
    int64_t objectId_;
    Statement put_, remove_;
};

template<class Read, class ReadRectangle>
void indexGeometry(sqlite3* db, int64_t objectId, const std::string& revision,
                   const std::string& kind, const std::vector<GeometryArrayInfo>& arrays, Read&& read, ReadRectangle&& readRectangle, bool tiledSurface = false)
{
    // Reuse unchanged entries rather than churning the R-tree on every edit.
    // Revision identity belongs to the object's head, so unchanged blocks do
    // not need a write just because a different block changed.
    std::map<RegionKey, Region> previous;
    {
        Statement old(db, "SELECT id,array_name,row_start,col_start,row_count,col_count,"
            "min_x,min_y,min_z,max_x,max_y,max_z FROM spatial_regions WHERE object_id=? AND array_name!=''",
            objectId);
        while (old.row()) {
            Region r{old.i64(0), old.i64(4), old.i64(5), {}};
            for (int i = 0; i < 6; ++i) r.bounds[i] = old.f64(i + 6);
            previous.emplace(RegionKey{old.text(1), old.i64(2), old.i64(3)}, r);
        }
    }
    RegionWriter writer(db, objectId);
    BoundsBuilder total;
    auto emit = [&](const std::string& name, int64_t row, int64_t col, int64_t rows, int64_t cols,
                    const BoundsBuilder& builder) {
        if (geometryWork) ++geometryWork->spatialRows;
        if (!builder.bounds) return;
        const auto& box = *builder.bounds;
        total.merge(box);
        auto old = previous.find({name, row, col});
        const bool unchanged = old != previous.end() && old->second.rows == rows && old->second.columns == cols
            && old->second.bounds == box;
        if (old != previous.end()) previous.erase(old);
        if (unchanged) return;
        writer.update(name, row, col, rows, cols, builder.bounds);
    };
    for (const auto& a : arrays) if (coordinates(kind, a.name)) {
        const int64_t stride = a.dtype == "<f8" ? 24 : 12;
        if (kind == "surface") {
            const auto rows = a.shape.at(0), cols = a.shape.at(1);
            if (!rows || !cols) continue;
            constexpr int64_t cells = 32;
            const int64_t tileColumns = (std::max<int64_t>(1, cols - 1) + cells - 1) / cells;
            for (int64_t row = 0; row < std::max<int64_t>(1, rows - 1); row += cells) {
                const int64_t height = std::min(cells + 1, rows - row);
                std::vector<BoundsBuilder> tiles(static_cast<size_t>(tileColumns));
                if (tiledSurface) {
                    // Tiled payloads are visited as rectangles, once per region.
                    for (int64_t t = 0; t < tileColumns; ++t) {
                        const auto col = t * cells, width = std::min(cells + 1, cols - col);
                        tiles[t].append(readRectangle(a.name, row, col, height, width), a.dtype, true);
                    }
                } else {
                    // Linear and in-memory arrays need row-major traversal:
                    // tile-major reads on a wide linear surface would repeatedly
                    // fetch dozens of 1 MiB chunks for each small region.
                    for (int64_t y = row; y < row + height; ++y)
                        for (int64_t t = 0; t < tileColumns; ++t) {
                            const auto col = t * cells, width = std::min(cells + 1, cols - col);
                            tiles[t].append(read(a.name, (y * cols + col) * stride, width * stride), a.dtype, true);
                        }
                }
                for (int64_t t = 0; t < tileColumns; ++t)
                    emit(a.name, row, t * cells, height, std::min(cells + 1, cols - t * cells), tiles[t]);
            }
        } else {
            const int64_t points = a.shape.at(0);
            const bool line = polyline(kind);
            constexpr int64_t block = 256;
            const auto end = line ? std::max<int64_t>(1, points - 1) : points;
            for (int64_t start = 0; points && start < end; start += block) {
                const int64_t count = std::min(block + (line ? 1 : 0), points - start);
                BoundsBuilder bounds;
                bounds.append(read(a.name, start * stride, count * stride), a.dtype, false);
                emit(a.name, start, 0, count, 0, bounds);
            }
        }
    }
    Statement remove(db, "DELETE FROM spatial_regions WHERE id=?");
    for (const auto& [key, region] : previous) {
        remove.bind(1, region.id); remove.stepDone(); remove.reset();
    }
    writeBounds(db, objectId, total.bounds, revision);
}
}

void setSpatialBounds(sqlite3* db, int64_t objectId, const std::optional<Bounds>& bounds,
                      const std::optional<std::string>& revision)
{
    writeBounds(db, objectId, bounds, revision);
}

void indexGeometryBounds(sqlite3* db, int64_t objectId, const std::string& revision, const GeometryData& geometry)
{
    std::vector<GeometryArrayInfo> arrays;
    std::map<std::string, const GeometryArray*> data;
    for (const auto& a : geometry.arrays) {
        arrays.push_back({a.name, a.dtype, a.shape, static_cast<int64_t>(a.bytes.size())});
        data.emplace(a.name, &a);
    }
    if (geometry.kind == "surface") refreshSurfaceArea(db, objectId, revision);
    indexGeometry(db, objectId, revision, geometry.kind, arrays,
        [&](const std::string& name, int64_t offset, int64_t count) {
            return std::span<const uint8_t>(data.at(name)->bytes).subspan(offset, count);
        }, [&](const std::string& name, int64_t row, int64_t col, int64_t rows, int64_t cols) {
            const auto& a = *data.at(name);
            std::vector<uint8_t> bytes(rows * cols * 12);
            for (int64_t y = 0; y < rows; ++y)
                std::memcpy(bytes.data() + y * cols * 12, a.bytes.data() + ((row + y) * a.shape[1] + col) * 12, cols * 12);
            return bytes;
        });
}

void refreshSurfaceArea(sqlite3* db, int64_t objectId, const std::string& revision)
{
    GeometryReader reader(db, revision, false);
    const auto& manifest = reader.manifest();
    if (manifest.at("kind") != "surface") return;
    const auto& arrays = manifest.at("arrays");
    const auto shape = arrays.at("points").at("shape").get<std::vector<int64_t>>();
    const int64_t rows = shape.at(0), columns = shape.at(1);
    const bool hasValid = arrays.contains(surfaceValidChannel);
    // Bands of about a million vertices, overlapping by a row.
    const int64_t band = std::max<int64_t>(2, (int64_t(1) << 20) / std::max<int64_t>(columns, 1));
    double total = 0;
    for (int64_t row = 0; row + 1 < rows; row += band - 1) {
        const int64_t count = std::min(band, rows - row);
        const auto points = xyzValues<float>(reader.rectangle("points", row, 0, count, columns));
        const auto valid = hasValid ? reader.rectangle(surfaceValidChannel, row, 0, count, columns) : std::vector<uint8_t>{};
        total = vc::geometry::surfaceGridArea(count, columns,
            [&](int64_t r, int64_t c) -> const std::array<float, 3>& { return points[size_t(r * columns + c)]; },
            [&](int64_t r, int64_t c) {
                const auto i = size_t(r * columns + c);
                return vc::geometry::validSurfacePoint(points[i]) && (!hasValid || valid[i]);
            }, total);
    }
    execute(db, "INSERT INTO surfaces(object_id,grid_width,grid_height,area) VALUES(?,?,?,?) ON CONFLICT(object_id) "
                "DO UPDATE SET grid_width=excluded.grid_width,grid_height=excluded.grid_height,area=excluded.area",
            objectId, columns, rows, total);
}

void refreshGeometryBounds(sqlite3* db, const std::string& objectUuid)
{
    Statement head(db, "SELECT o.id,h.revision_uuid FROM objects o JOIN geometry_heads h ON h.object_uuid=o.uuid WHERE o.uuid=?",
        parseUuid(objectUuid));
    if (!head.row()) return;
    const auto revision = formatUuid(head.blob16(1));
    GeometryReader reader(db, revision);
    const auto& manifest = reader.manifest();
    std::vector<GeometryArrayInfo> arrays;
    for (const auto& [name, a] : manifest.at("arrays").items())
        arrays.push_back({name, a.at("dtype"), a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length")});
    refreshSurfaceArea(db, head.i64(0), revision);
    indexGeometry(db, head.i64(0), revision, manifest.at("kind").get<std::string>(), arrays,
        [&](const std::string& name, int64_t offset, int64_t count) { return reader.read(name, offset, count); },
        [&](const std::string& name, int64_t row, int64_t col, int64_t rows, int64_t cols) {
            return reader.rectangle(name, row, col, rows, cols);
        }, manifest.at("kind") == "surface" && manifest.at("arrays").at("points").contains("storage"));
}

namespace {
// Of an edit list: does it change a surface's points or validity?
bool editsSurfaceArea(const std::vector<GeometryArrayEdit>& edits)
{
    return std::any_of(edits.begin(), edits.end(), [](const auto& edit) {
        return edit.array == "points" || edit.array == surfaceValidChannel;
    });
}

// The area of `cells` (row-major) in a surface revision, read row by row.
double cellsArea(GeometryReader& reader, const std::vector<std::pair<int64_t, int64_t>>& cells)
{
    const bool hasValid = reader.manifest().at("arrays").contains(surfaceValidChannel);
    double total = 0;
    for (size_t first = 0; first < cells.size();) {
        const auto row = cells[first].first;
        size_t last = first;
        while (last + 1 < cells.size() && cells[last + 1].first == row) ++last;
        const auto c0 = cells[first].second, columns = cells[last].second - c0 + 2;
        const auto points = xyzValues<float>(reader.rectangle("points", row, c0, 2, columns));
        const auto valid = hasValid ? reader.rectangle(surfaceValidChannel, row, c0, 2, columns) : std::vector<uint8_t>{};
        for (size_t i = first; i <= last; ++i) {
            const auto c = cells[i].second - c0;
            total = vc::geometry::surfaceGridArea(2, 2,
                [&](int64_t r, int64_t k) -> const std::array<float, 3>& { return points[size_t(r * columns + c + k)]; },
                [&](int64_t r, int64_t k) {
                    const auto at = size_t(r * columns + c + k);
                    return vc::geometry::validSurfacePoint(points[at]) && (!hasValid || valid[at]);
                }, total);
        }
        first = last + 1;
    }
    return total;
}
}

std::optional<EditedSurfaceCells> editedSurfaceCells(sqlite3* db, const std::string& revision,
                                                     const std::vector<GeometryArrayEdit>& edits)
{
    if (!editsSurfaceArea(edits)) return std::nullopt;
    GeometryReader reader(db, revision, false);
    if (reader.manifest().at("kind") != "surface") return std::nullopt;
    const auto shape = reader.manifest().at("arrays").at("points").at("shape").get<std::vector<int64_t>>();
    const int64_t rows = shape.at(0), columns = shape.at(1);
    if (rows < 2 || columns < 2) return std::nullopt;
    // Past about a quarter of the grid a full pass is no slower.
    const int64_t limit = std::max<int64_t>(1, (rows - 1) * (columns - 1) / 4);
    std::set<std::pair<int64_t, int64_t>> cells;
    for (const auto& edit : edits) {
        int64_t first = 0, count = 0;
        if (edit.array == "points") {
            first = edit.elementOffset / 3;
            count = (edit.elementOffset + int64_t(edit.bytes.size() / sizeof(float)) + 2) / 3 - first;
        } else if (edit.array == surfaceValidChannel) {
            first = edit.elementOffset;
            count = int64_t(edit.bytes.size());
        } else {
            continue;
        }
        if (count > limit) return std::nullopt;
        for (auto vertex = first; vertex < first + count; ++vertex) {
            const auto r = vertex / columns, c = vertex % columns;
            for (auto cr = std::max<int64_t>(0, r - 1); cr <= std::min(r, rows - 2); ++cr)
                for (auto cc = std::max<int64_t>(0, c - 1); cc <= std::min(c, columns - 2); ++cc) cells.emplace(cr, cc);
        }
        if (int64_t(cells.size()) > limit) return std::nullopt;
    }
    EditedSurfaceCells result{{cells.begin(), cells.end()}, 0};
    result.area = cellsArea(reader, result.cells);
    return result;
}

void updateEditedGeometryBounds(sqlite3* db, int64_t objectId, const std::string& revision,
                                const std::vector<GeometryArrayEdit>& edits, const std::optional<EditedSurfaceCells>& before)
{
    GeometryReader reader(db, revision);
    const auto& manifest = reader.manifest();
    const std::string kind = manifest.at("kind");
    // No edits list: a channel rewrite, which may replace validity.
    if (kind == "surface" && (edits.empty() || editsSurfaceArea(edits))) {
        Statement stored(db, "SELECT area FROM surfaces WHERE object_id=?", objectId);
        if (before && stored.row()) {
            const auto area = stored.f64(0) - before->area + cellsArea(reader, before->cells);
            execute(db, "UPDATE surfaces SET area=? WHERE object_id=?", std::max(0.0, area), objectId);
        } else {
            refreshSurfaceArea(db, objectId, revision);
        }
    }
    std::set<RegionKey> dirty;
    for (const auto& edit : edits) {
        if (edit.bytes.empty() || !coordinates(kind, edit.array)) continue;
        const auto& a = manifest.at("arrays").at(edit.array);
        const auto shape = a.at("shape").get<std::vector<int64_t>>();
        const auto scalar = geometryByteLength(a.at("dtype"), {1});
        const auto first = edit.elementOffset / 3;
        const auto last = (edit.elementOffset + int64_t(edit.bytes.size() / scalar) - 1) / 3;
        if (kind == "surface") {
            // A vertex dirties its 32-block and, on a block's first row or column,
            // the block before it (regions overlap by one vertex). Per run of a
            // row: the blocks from the first vertex's to the last vertex's.
            const auto rowLimit = std::max<int64_t>(1, shape[0]-1), colLimit = std::max<int64_t>(1, shape[1]-1);
            for (auto point = first; point <= last;) {
                const auto row = point / shape[1], firstCol = point % shape[1];
                const auto lastCol = std::min(shape[1] - 1, firstCol + (last - point));
                const auto fromCol = firstCol > 0 && firstCol % 32 == 0 ? firstCol - 32 : firstCol / 32 * 32;
                for (auto r : {row / 32 * 32, row > 0 && row % 32 == 0 ? row - 32 : row / 32 * 32})
                    for (auto c = fromCol; c <= lastCol / 32 * 32; c += 32)
                        if (r < rowLimit && c < colLimit) dirty.emplace(edit.array, r, c);
                point += lastCol - firstCol + 1;
            }
            continue;
        }
        for (auto point = first; point <= last; ++point) {
            const bool line = polyline(kind);
            const auto end = line ? std::max<int64_t>(1, shape[0]-1) : shape[0];
            const auto start = point / 256 * 256;
            if (start < end) dirty.emplace(edit.array, start, 0);
            if (line && point && point % 256 == 0) dirty.emplace(edit.array, point-256, 0);
        }
    }
    RegionWriter writer(db, objectId);
    for (const auto& [name, row, col] : dirty) {
        const auto& a = manifest.at("arrays").at(name);
        const auto shape = a.at("shape").get<std::vector<int64_t>>();
        const std::string dtype = a.at("dtype");
        const auto rows = std::min<int64_t>(kind == "surface" ? 33 : kind == "point_collection" ? 256 : 257, shape[0]-row);
        const auto cols = kind == "surface" ? std::min<int64_t>(33, shape[1]-col) : 0;
        BoundsBuilder bounds;
        const auto stride = geometryByteLength(dtype, {3});
        bounds.append(kind == "surface" ? reader.rectangle(name, row, col, rows, cols) : reader.read(name, row*stride, rows*stride),
                      dtype, kind == "surface");
        writer.update(name, row, col, rows, cols, bounds.bounds);
    }
    Statement total(db, "SELECT min(min_x),min(min_y),min(min_z),max(max_x),max(max_y),max(max_z) "
                        "FROM spatial_regions WHERE object_id=? AND array_name!=''", objectId); total.row();
    std::optional<Bounds> bounds;
    if (!total.isNull(0)) {
        bounds.emplace();
        for (int i = 0; i < 6; ++i) (*bounds)[i] = total.f64(i);
    }
    writeBounds(db, objectId, bounds, revision);
}

namespace {
void createRegionSchema(sqlite3* db)
{
    // One shared float R-tree holds both whole-object summaries and geometry
    // regions. Exact double bounds eliminate outward-rounding false positives.
    exec(db, R"SQL(
CREATE TABLE spatial_regions(
  id INTEGER PRIMARY KEY,
  object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  array_name TEXT NOT NULL,
  row_start INTEGER NOT NULL DEFAULT 0,col_start INTEGER NOT NULL DEFAULT 0,
  row_count INTEGER NOT NULL DEFAULT 0,col_count INTEGER NOT NULL DEFAULT 0,
  min_x REAL NOT NULL,min_y REAL NOT NULL,min_z REAL NOT NULL,
  max_x REAL NOT NULL,max_y REAL NOT NULL,max_z REAL NOT NULL,
  UNIQUE(object_id,array_name,row_start,col_start)
);
CREATE VIRTUAL TABLE spatial_regions_rtree USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
CREATE TRIGGER spatial_regions_insert AFTER INSERT ON spatial_regions BEGIN
  INSERT INTO spatial_regions_rtree VALUES(new.id,
    max(-1e30,min(1e30,new.min_x)),max(-1e30,min(1e30,new.max_x)),
    max(-1e30,min(1e30,new.min_y)),max(-1e30,min(1e30,new.max_y)),
    max(-1e30,min(1e30,new.min_z)),max(-1e30,min(1e30,new.max_z)));
END;
CREATE TRIGGER spatial_regions_update AFTER UPDATE ON spatial_regions BEGIN
  DELETE FROM spatial_regions_rtree WHERE id=old.id;
  INSERT INTO spatial_regions_rtree VALUES(new.id,
    max(-1e30,min(1e30,new.min_x)),max(-1e30,min(1e30,new.max_x)),
    max(-1e30,min(1e30,new.min_y)),max(-1e30,min(1e30,new.max_y)),
    max(-1e30,min(1e30,new.min_z)),max(-1e30,min(1e30,new.max_z)));
END;
CREATE TRIGGER spatial_regions_delete AFTER DELETE ON spatial_regions BEGIN
  DELETE FROM spatial_regions_rtree WHERE id=old.id;
END;
CREATE TRIGGER object_bounds_insert AFTER INSERT ON object_bounds BEGIN
  INSERT INTO spatial_regions(object_id,array_name,min_x,min_y,min_z,max_x,max_y,max_z)
    VALUES(new.object_id,'',new.min_x,new.min_y,new.min_z,new.max_x,new.max_y,new.max_z);
END;
CREATE TRIGGER object_bounds_update AFTER UPDATE ON object_bounds
WHEN old.min_x!=new.min_x OR old.min_y!=new.min_y OR old.min_z!=new.min_z
  OR old.max_x!=new.max_x OR old.max_y!=new.max_y OR old.max_z!=new.max_z BEGIN
  UPDATE spatial_regions SET min_x=new.min_x,min_y=new.min_y,min_z=new.min_z,
    max_x=new.max_x,max_y=new.max_y,max_z=new.max_z WHERE object_id=new.object_id AND array_name='';
END;
CREATE TRIGGER object_bounds_delete AFTER DELETE ON object_bounds BEGIN
  DELETE FROM spatial_regions WHERE object_id=old.object_id AND array_name='';
END;
)SQL");
}
}

void createSpatialSchema(sqlite3* db)
{
    // Exact double bounds remain authoritative. SQLite's float R-tree is only
    // the broad phase. Saturation preserves ordering for very large doubles;
    // queries apply the same saturation and then recheck exact double bounds.
    exec(db, R"SQL(
CREATE TABLE object_bounds(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  revision_uuid BLOB REFERENCES geometry_revisions(uuid),
  min_x REAL NOT NULL,min_y REAL NOT NULL,min_z REAL NOT NULL,
  max_x REAL NOT NULL,max_y REAL NOT NULL,max_z REAL NOT NULL
);
CREATE INDEX objects_kind_uuid ON objects(kind,uuid);
)SQL");
    createRegionSchema(db);
}

}
