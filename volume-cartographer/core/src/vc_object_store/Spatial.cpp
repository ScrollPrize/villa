#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "geometry/GeometryReader.hpp"
#include <bit>
#include <tuple>

namespace vc::objects::detail {
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

std::optional<Bounds> recordBounds(const ObjectRecord& object)
{
    if (object.kind == "fiber" && object.fiber) return object.fiber->bbox;
    if (object.kind == "patch" && object.patch) return object.patch->bbox;
    if (object.kind == "track" && object.track) return object.track->bbox;
    return std::nullopt;
}

namespace {
struct BoundsBuilder {
    std::optional<Bounds> bounds;
    template<class T> void append(std::span<const uint8_t> bytes, bool patch) {
        if (bytes.size() % (3 * sizeof(T))) throw std::runtime_error("spatial index: invalid XYZ array");
        for (size_t offset = 0; offset < bytes.size(); offset += 3 * sizeof(T)) {
            std::array<double, 3> xyz;
            for (size_t axis = 0; axis < 3; ++axis) {
                std::array<uint8_t, sizeof(T)> scalar;
                std::memcpy(scalar.data(), bytes.data() + offset + axis * sizeof(T), sizeof(T));
                if constexpr (std::endian::native != std::endian::little) std::reverse(scalar.begin(), scalar.end());
                xyz[axis] = std::bit_cast<T>(scalar);
            }
            // Conservatively cover both VC3D validity conventions: raw surfaces
            // use x != -1, mapped surfaces use z > 0. Narrow phases choose
            // their own convention; indexing their union cannot lose triangles.
            if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2]) || (patch && xyz[0] == -1 && xyz[2] <= 0)) continue;
            if (!bounds) bounds = Bounds{xyz[0], xyz[1], xyz[2], xyz[0], xyz[1], xyz[2]};
            else for (size_t axis = 0; axis < 3; ++axis) {
                (*bounds)[axis] = std::min((*bounds)[axis], xyz[axis]);
                (*bounds)[axis + 3] = std::max((*bounds)[axis + 3], xyz[axis]);
            }
        }
    }
    void merge(const Bounds& b) {
        if (!bounds) bounds = b;
        else for (size_t axis = 0; axis < 3; ++axis) {
            (*bounds)[axis] = std::min((*bounds)[axis], b[axis]);
            (*bounds)[axis + 3] = std::max((*bounds)[axis + 3], b[axis + 3]);
        }
    }
    void append(std::span<const uint8_t> bytes, const std::string& dtype, bool patch) {
        if (dtype == "<f4") append<float>(bytes, patch);
        else if (dtype == "<f8") append<double>(bytes, patch);
        else if (dtype == "<i4") append<int32_t>(bytes, patch);
        else throw std::runtime_error("spatial index: unsupported coordinate dtype");
    }
};
bool coordinates(const std::string& kind, const std::string& name) {
    return kind == "fiber" ? name == "line_points" || name == "control_points" : name == "points";
}
}

namespace {
void writeBounds(sqlite3* db, int64_t objectId, const std::optional<Bounds>& bounds,
                 const std::optional<std::string>& revision)
{
    if (!bounds) {
        Statement remove(db, "DELETE FROM object_bounds WHERE object_id=?");
        remove.bind(1, objectId); remove.stepDone(); return;
    }
    validateBounds(*bounds);
    Statement put(db, "INSERT INTO object_bounds VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(object_id) DO UPDATE SET "
        "revision_uuid=excluded.revision_uuid,min_x=excluded.min_x,min_y=excluded.min_y,min_z=excluded.min_z,"
        "max_x=excluded.max_x,max_y=excluded.max_y,max_z=excluded.max_z");
    put.bind(1, objectId);
    if (revision) put.bindBlob(2, parseUuid(*revision)); else put.bindNull(2);
    for (size_t i = 0; i < 6; ++i) put.bind(int(i) + 3, (*bounds)[i]);
    put.stepDone();
}

// Starts/counts exclude the final XYZ axis. Patches use rows and columns;
// one-dimensional arrays use row_start/count and zero column values.
struct Region {
    int64_t id = 0, rows = 0, columns = 0;
    Bounds bounds;
};
using RegionKey = std::tuple<std::string, int64_t, int64_t>;

template<class Read, class ReadRectangle>
void indexGeometry(sqlite3* db, int64_t objectId, const std::string& revision,
                   const std::string& kind, const std::vector<GeometryArrayInfo>& arrays, Read&& read, ReadRectangle&& readRectangle, bool tiledPatch = false)
{
    // Reuse unchanged entries rather than churning the R-tree on every edit.
    // Revision identity belongs to the object's head, so unchanged blocks do
    // not need a write just because a different block changed.
    std::map<RegionKey, Region> previous;
    {
        Statement old(db, "SELECT id,array_name,row_start,col_start,row_count,col_count,"
            "min_x,min_y,min_z,max_x,max_y,max_z FROM spatial_regions WHERE object_id=? AND array_name!=''");
        old.bind(1, objectId);
        while (old.row()) {
            Region r{old.i64(0), old.i64(4), old.i64(5), {}};
            for (int i = 0; i < 6; ++i) r.bounds[i] = old.f64(i + 6);
            previous.emplace(RegionKey{old.text(1), old.i64(2), old.i64(3)}, r);
        }
    }
    Statement put(db, "INSERT INTO spatial_regions(object_id,array_name,row_start,col_start,row_count,col_count,"
        "min_x,min_y,min_z,max_x,max_y,max_z) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(object_id,array_name,row_start,col_start) DO UPDATE SET "
        "row_count=excluded.row_count,col_count=excluded.col_count,min_x=excluded.min_x,min_y=excluded.min_y,"
        "min_z=excluded.min_z,max_x=excluded.max_x,max_y=excluded.max_y,max_z=excluded.max_z");
    BoundsBuilder total;
    auto emit = [&](const std::string& name, int64_t row, int64_t col, int64_t rows, int64_t cols,
                    const BoundsBuilder& builder) {
        if (!builder.bounds) return;
        const auto& box = *builder.bounds;
        total.merge(box);
        auto old = previous.find({name, row, col});
        const bool unchanged = old != previous.end() && old->second.rows == rows && old->second.columns == cols
            && old->second.bounds == box;
        if (old != previous.end()) previous.erase(old);
        if (unchanged) return;
        put.bind(1, objectId); put.bind(2, name); put.bind(3, row); put.bind(4, col);
        put.bind(5, rows); put.bind(6, cols);
        for (int i = 0; i < 6; ++i) put.bind(i + 7, box[i]);
        put.stepDone(); put.reset();
    };
    for (const auto& a : arrays) if (coordinates(kind, a.name)) {
        const int64_t stride = a.dtype == "<f8" ? 24 : 12;
        if (kind == "patch") {
            const auto rows = a.shape.at(0), cols = a.shape.at(1);
            if (!rows || !cols) continue;
            constexpr int64_t cells = 32;
            const int64_t tileColumns = (std::max<int64_t>(1, cols - 1) + cells - 1) / cells;
            for (int64_t row = 0; row < std::max<int64_t>(1, rows - 1); row += cells) {
                const int64_t height = std::min(cells + 1, rows - row);
                std::vector<BoundsBuilder> tiles(static_cast<size_t>(tileColumns));
                if (tiledPatch) {
                    // Tiled payloads are visited as rectangles, once per region.
                    for (int64_t t = 0; t < tileColumns; ++t) {
                        const auto col = t * cells, width = std::min(cells + 1, cols - col);
                        tiles[t].append(readRectangle(a.name, row, col, height, width), a.dtype, true);
                    }
                } else {
                    // Legacy and in-memory arrays need row-major traversal:
                    // tile-major reads on a wide legacy patch would repeatedly
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
            const bool line = kind == "fiber" || kind == "track";
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

void refreshGeometryBounds(sqlite3* db, const std::string& objectUuid)
{
    Statement head(db, "SELECT o.id,h.revision_uuid FROM objects o JOIN geometry_heads h ON h.object_uuid=o.uuid WHERE o.uuid=?");
    head.bindBlob(1, parseUuid(objectUuid));
    if (!head.row()) return;
    const auto revision = formatUuid(head.blob16(1));
    GeometryReader reader(db, revision);
    const auto& manifest = reader.manifest();
    std::vector<GeometryArrayInfo> arrays;
    for (const auto& [name, a] : manifest.at("arrays").items())
        arrays.push_back({name, a.at("dtype"), a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length")});
    indexGeometry(db, head.i64(0), revision, manifest.at("kind").get<std::string>(), arrays,
        [&](const std::string& name, int64_t offset, int64_t count) { return reader.read(name, offset, count); },
        [&](const std::string& name, int64_t row, int64_t col, int64_t rows, int64_t cols) {
            return reader.rectangle(name, row, col, rows, cols);
        }, manifest.at("kind") == "patch" && manifest.at("arrays").at("points").contains("storage"));
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

void migrateSpatialRegions(sqlite3* db)
{
    exec(db, "DROP TRIGGER object_bounds_insert; DROP TRIGGER object_bounds_update; "
        "DROP TRIGGER object_bounds_delete; DROP TABLE object_bounds_rtree;");
    createRegionSchema(db);
    exec(db, "INSERT INTO spatial_regions(object_id,array_name,min_x,min_y,min_z,max_x,max_y,max_z) "
        "SELECT object_id,'',min_x,min_y,min_z,max_x,max_y,max_z FROM object_bounds;");
    Statement heads(db, "SELECT object_uuid FROM geometry_heads");
    while (heads.row()) refreshGeometryBounds(db, formatUuid(heads.blob16(0)));
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

void backfillSpatialBounds(sqlite3* db)
{
    // Registered external objects retain their annotated bounds. Managed
    // geometry always wins, including empty geometries with no spatial extent.
    for (const auto* table : {"fibers", "patches", "tracks"}) {
        Statement rows(db, std::string("SELECT object_id,bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z FROM ") + table + " WHERE bbox_min_x IS NOT NULL");
        while (rows.row()) {
            Bounds box;
            bool complete = true;
            for (int i = 0; i < 6; ++i) { complete &= !rows.isNull(i + 1); box[i] = rows.f64(i + 1); }
            if (complete) setSpatialBounds(db, rows.i64(0), box);
        }
    }
    Statement heads(db, "SELECT object_uuid FROM geometry_heads");
    while (heads.row()) refreshGeometryBounds(db, formatUuid(heads.blob16(0)));
}
}
