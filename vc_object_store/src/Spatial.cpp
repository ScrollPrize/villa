#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/PersistentTree.hpp"
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
    void merge(const Bounds& b) {
        if (!bounds) bounds = b;
        else for (size_t axis = 0; axis < 3; ++axis) {
            (*bounds)[axis] = std::min((*bounds)[axis], b[axis]);
            (*bounds)[axis + 3] = std::max((*bounds)[axis + 3], b[axis + 3]);
        }
    }
    void append(std::span<const uint8_t> bytes, const std::string& dtype, bool patch) {
        const auto value=coordinateBounds(bytes,dtype,patch);
        if(!value.is_null())merge(value.get<Bounds>());
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
    bool indexed=true;for(const auto& a:manifest.at("arrays"))indexed &= a.contains("tree");
    if(indexed) {auto copy=manifest;indexSharedGeometry(db,head.i64(0),revision,copy);return;}

    std::vector<GeometryArrayInfo> arrays;
    for (const auto& [name, a] : manifest.at("arrays").items())
        arrays.push_back({name, a.at("dtype"), a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length")});
    indexGeometry(db, head.i64(0), revision, manifest.at("kind").get<std::string>(), arrays,
        [&](const std::string& name, int64_t offset, int64_t count) { return reader.read(name, offset, count); },
        [&](const std::string& name, int64_t row, int64_t col, int64_t rows, int64_t cols) {
            return reader.rectangle(name, row, col, rows, cols);
        }, manifest.at("kind") == "patch" && manifest.at("arrays").at("points").contains("storage"));
}

void updateEditedGeometryBounds(sqlite3* db, int64_t objectId, const std::string& revision,
                                const std::vector<GeometryArrayEdit>& edits)
{
    GeometryReader reader(db, revision);
    const auto& manifest = reader.manifest();
    const std::string kind = manifest.at("kind");
    std::set<RegionKey> dirty;
    for (const auto& edit : edits) {
        if (edit.bytes.empty() || !coordinates(kind, edit.array)) continue;
        const auto& a = manifest.at("arrays").at(edit.array);
        const auto shape = a.at("shape").get<std::vector<int64_t>>();
        const auto scalar = geometryByteLength(a.at("dtype"), {1});
        const auto first = edit.elementOffset / 3;
        const auto last = (edit.elementOffset + int64_t(edit.bytes.size() / scalar) - 1) / 3;
        for (auto point = first; point <= last; ++point) {
            if (kind == "patch") {
                const auto row = point / shape[1], col = point % shape[1];
                for (auto r : {row / 32 * 32, row > 0 && row % 32 == 0 ? row - 32 : row / 32 * 32})
                    for (auto c : {col / 32 * 32, col > 0 && col % 32 == 0 ? col - 32 : col / 32 * 32})
                        if (r < std::max<int64_t>(1, shape[0]-1) && c < std::max<int64_t>(1, shape[1]-1))
                            dirty.emplace(edit.array, r, c);
            } else {
                const bool line = kind == "fiber" || kind == "track";
                const auto end = line ? std::max<int64_t>(1, shape[0]-1) : shape[0];
                const auto start = point / 256 * 256;
                if (start < end) dirty.emplace(edit.array, start, 0);
                if (line && point && point % 256 == 0) dirty.emplace(edit.array, point-256, 0);
            }
        }
    }
    RegionWriter writer(db, objectId);
    for (const auto& [name, row, col] : dirty) {
        const auto& a = manifest.at("arrays").at(name);
        const auto shape = a.at("shape").get<std::vector<int64_t>>();
        const std::string dtype = a.at("dtype");
        const auto rows = std::min<int64_t>(kind == "patch" ? 33 : kind == "point_collection" ? 256 : 257, shape[0]-row);
        const auto cols = kind == "patch" ? std::min<int64_t>(33, shape[1]-col) : 0;
        BoundsBuilder bounds;
        const auto stride = geometryByteLength(dtype, {3});
        bounds.append(kind == "patch" ? reader.rectangle(name, row, col, rows, cols) : reader.read(name, row*stride, rows*stride),
                      dtype, kind == "patch");
        writer.update(name, row, col, rows, cols, bounds.bounds);
    }
    Statement total(db, "SELECT min(min_x),min(min_y),min(min_z),max(max_x),max(max_y),max(max_z) "
                        "FROM spatial_regions WHERE object_id=? AND array_name!=''");
    total.bind(1, objectId); total.row();
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

// Structural layouts keep conservative per-block bounds in their immutable
// manifests. Unchanged blocks reuse bounds as well as payload. Region offsets
// are rebuilt from descriptors, without reading unchanged coordinate payloads.
#include "geometry/SharedGeometry.hpp"
namespace vc::objects::detail {
void indexSharedGeometry(sqlite3* db, int64_t objectId, const std::string& revision, nlohmann::json& manifest)
{
    GeometryPhase phase("indexSharedGeometry");
    const std::string kind=manifest.at("kind");
    bool indexed=true; for(const auto& a:manifest.at("arrays"))indexed &= a.contains("tree");
    if(indexed) {
        // The immutable tree carries the fine-grained bounds. SQL stores one
        // coarse candidate per coordinate array, independent of object size.
        Statement cleanup(db,"DELETE FROM spatial_regions WHERE object_id=? AND array_name!='' AND array_name NOT LIKE '@tree:%'");
        cleanup.bind(1,objectId);cleanup.stepDone();
        RegionWriter writer(db,objectId); BoundsBuilder total;
        for(const auto& [name,a]:manifest.at("arrays").items())if(coordinates(kind,name)) {
            const auto& root=a.at("tree");std::optional<Bounds> bounds;
            if(!root.is_null()&&!root.at("bounds").is_null())bounds=root.at("bounds").get<Bounds>();
            const auto shape=a.at("shape").get<std::vector<int64_t>>();
            writer.update("@tree:"+name,0,0,shape[0],kind=="patch"?shape[1]:0,bounds);
            if(bounds)total.merge(*bounds);
            if(geometryWork)++geometryWork->spatialRows;
        }
        writeBounds(db,objectId,total.bounds,revision);return;
    }

    Statement remove(db,"DELETE FROM spatial_regions WHERE object_id=? AND array_name!=''");
    remove.bind(1,objectId); remove.stepDone();
    RegionWriter writer(db,objectId); BoundsBuilder total;
    for (auto& [name,a] : manifest.at("arrays").items()) {
        if (!coordinates(kind,name)) continue;
        const auto shape=a.at("shape").get<std::vector<int64_t>>();
        const std::string dtype=a.at("dtype"); const auto stride=geometryByteLength(dtype,{3});
        if (a.contains("blocks")) {
            int64_t offset=0;
            for (auto& b : a.at("blocks")) {
                const int64_t length=b.at("length");
                const auto begin=offset/stride, end=std::min(shape[0],(offset+length+stride-1)/stride);
                BoundsBuilder bounds;
                // Bounds cover complete points intersecting this block. A line
                // region also includes the next point to cover crossing spans.
                const auto count=std::min(shape[0]-begin,end-begin+(kind!="point_collection"));
                if (b.contains("bounds") && b.contains("bounds_count") && b.at("bounds_count")==count) {
                    if (!b.at("bounds").is_null()) bounds.merge(b.at("bounds").get<Bounds>());
                } else {
                    bounds.append(readSharedArray(db,a,name,begin*stride,count*stride),dtype,false);
                    b["bounds"]=bounds.bounds ? nlohmann::json(*bounds.bounds) : nlohmann::json(); b["bounds_count"]=count;
                }
                if (bounds.bounds) { total.merge(*bounds.bounds); writer.update(name,begin,0,count,0,bounds.bounds); }
                offset+=length;
                if (geometryWork) ++geometryWork->spatialRows;
            }
        } else {
            // Each tile's stored bounds cover its own vertices. For triangles
            // crossing tile boundaries, include neighbours conservatively.
            for (auto& t : a.at("tiles")) if (!t.contains("bounds")) {
                auto temporary=a; temporary["tiles"]=nlohmann::json::array({t});
                const int64_t r=t.at("row"), c=t.at("column"), h=t.at("rows"), w=t.at("columns");
                BoundsBuilder bounds;
                bounds.append(readSharedRectangle(db,temporary,name,r,c,h,w),dtype,true);
                t["bounds"]=bounds.bounds ? nlohmann::json(*bounds.bounds) : nlohmann::json();
            }
            std::map<std::pair<int64_t,int64_t>, const nlohmann::json*> neighbours;
            const int64_t tile=a.at("tile_size");
            const auto origin=manifest.value("grid_origin",std::array<int64_t,2>{0,0});
            auto cell=[&](int64_t n) { auto q=n/tile; if (n%tile<0) --q; return q; };
            for (const auto& t:a.at("tiles")) neighbours[{cell(origin[0]+t.at("row").get<int64_t>()),cell(origin[1]+t.at("column").get<int64_t>())}]=&t;
            for (const auto& t : a.at("tiles")) {
                const int64_t r=t.at("row"),c=t.at("column"),h=t.at("rows"),w=t.at("columns");
                BoundsBuilder bounds;
                for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
                    const auto found=neighbours.find({cell(origin[0]+r)+dr,cell(origin[1]+c)+dc});
                    if (found!=neighbours.end() && !found->second->at("bounds").is_null()) bounds.merge(found->second->at("bounds").get<Bounds>());
                }
                if (bounds.bounds) { total.merge(*bounds.bounds); writer.update(name,r,c,std::min(h+1,shape[0]-r),std::min(w+1,shape[1]-c),bounds.bounds); }
                if (geometryWork) ++geometryWork->spatialRows;
            }
        }
    }
    writeBounds(db,objectId,total.bounds,revision);
}
void visitIndexedRegions(sqlite3* db, const SpatialRegion& candidate, const Bounds& bounds,
                         GeometryReader& reader, const std::function<void(const SpatialRegion&)>& emit) {
    auto overlaps=[&](const nlohmann::json& box) {
        requestPoll();
        if(box.is_null())return false;
        for(int i=0;i<3;++i)if(box.at(i).get<double>()>bounds[i+3]||box.at(i+3).get<double>()<bounds[i])return false;
        return true;
    };
    auto region=candidate;
    if (!region.arrayName.starts_with("@tree:")) { emit(region); return; }
    region.arrayName.erase(0,6);
    const auto& a=reader.manifest().at("arrays").at(region.arrayName);
    if(!a.contains("tree")) {emit(region);return;}
    const auto stride=geometryByteLength(a.at("dtype"),{3});const bool patch=a.contains("tile_size");
    const auto shape=a.at("shape").get<std::vector<int64_t>>();
    std::function<void(const Tree&,int64_t)> visit=[&](const Tree& ref,int64_t offset) {
        if(ref.is_null()||!overlaps(ref.at("bounds")))return;
        const auto node=treeNode(db,ref);
        if(node.contains("value")) {
            auto r=region;r.bbox=ref.at("bounds").get<Bounds>();
            if(patch) {
                const auto origin=a.at("origin").get<std::array<int64_t,2>>();const auto& v=node.at("value");
                const int64_t row=v.at("row").get<int64_t>()-origin[0],col=v.at("column").get<int64_t>()-origin[1];
                const auto top=std::max<int64_t>(0,row-1),left=std::max<int64_t>(0,col-1);
                r.start={top,left};r.shape={std::min(shape[0],row+v.at("rows").get<int64_t>()+1)-top,std::min(shape[1],col+v.at("columns").get<int64_t>()+1)-left};
            } else {r.start={offset/stride};r.shape={treeSize(ref)/stride};}
            emit(r);return;
        }
        const auto& left=node.at("left");const auto& right=node.at("right");
        visit(left,offset);
        // In-order seams connect adjacent leaf spans exactly once.
        if(!patch&&region.kind!="point_collection") {
            const auto seam=mergeTreeBounds(left.at("last_bounds"),right.at("first_bounds"));
            if(overlaps(seam)) {auto r=region;r.bbox=seam.get<Bounds>();r.start={(offset+treeSize(left))/stride-1};r.shape={2};emit(r);}
        }
        visit(right,offset+treeSize(left));
    };
    visit(a.at("tree"),0);
}
std::vector<SpatialRegion> refineIndexedRegions(sqlite3* db,const std::vector<SpatialRegion>& candidates,const ObjectSelection& selection) {
    TreeCacheScope cache;
    std::vector<SpatialRegion> result;
    for(const auto& candidate:candidates) {
        if(!candidate.revisionUuid||!candidate.arrayName.starts_with("@tree:")) {result.push_back(candidate);continue;}
        GeometryReader reader(db,*candidate.revisionUuid);
        visitIndexedRegions(db,candidate,*selection.bbox,reader,[&](const auto& region){result.push_back(region);});
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return std::tie(a.objectUuid,a.arrayName,a.start)<std::tie(b.objectUuid,b.arrayName,b.start);});
    const auto begin=std::min<uint64_t>(selection.offset,result.size());
    const auto count=selection.limit?std::min<uint64_t>(*selection.limit,result.size()-begin):result.size()-begin;
    return {result.begin()+begin,result.begin()+begin+count};
}

}
