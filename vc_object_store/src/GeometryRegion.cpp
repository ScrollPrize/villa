#include "geometry/GeometryReader.hpp"
#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "internal/ReadRequest.hpp"
#include "internal/RecordSql.hpp"
#include "internal/CoordinateFilter.hpp"
#include "internal/VolumeFrames.hpp"
#include "internal/TransformedLength.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/SegmentClip.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "geometry/ArrayLayout.hpp"
#include "spatial/SurfaceTiles.hpp"
#include <map>
#include <exception>
#include <set>

namespace vc::objects {
using namespace detail;
const char* regionStopName(RegionStop reason)
{
    switch (reason) {
    case RegionStop::Exhausted: return "exhausted";
    case RegionStop::ObjectLimit: return "object_limit";
    case RegionStop::OutputBudget: return "output_budget";
    case RegionStop::ProcessingBudget: return "processing_budget";
    case RegionStop::CandidateBudget: return "candidate_budget";
    case RegionStop::SqlBudget: return "sql_budget";
    case RegionStop::Cancelled: return "cancelled";
    }
    throw std::logic_error("invalid region stop reason");
}
namespace {
using Point = std::array<double, 3>;
// Region geometry kinds: their typed summary table and coordinate arrays in
// precedence order (the first nonempty array is intersected, else the last).
// Polyline kinds clip segments to the slab; surfaces intersect its mid-plane.
struct RegionKind {
    const char* table;
    std::vector<std::string> arrays;
    const char* dtype;
    int64_t stride;
    bool surface = false;
};
RegionKind regionKind(const std::string& kind)
{
    if (kind == "fiber") return {"fibers", {"line_points", "control_points"}, "<f8", 24};
    if (kind == "track") return {"tracks", {"points"}, "<i4", 12};
    if (kind == "surface") return {"surfaces", {"points"}, "<f4", 12, true};
    throw std::invalid_argument("geometry region does not support kind: " + kind);
}
struct RequestScope {
    sqlite3* db;
    ReadRequest& request;
    std::exception_ptr error;
    RequestScope(sqlite3* connection, ReadRequest& r) : db(connection), request(r) {
        if (readRequest) throw std::logic_error("region queries cannot be nested");
        readRequest = &r;
        // Single-opcode granularity gives a strict instruction allowance. An
        // opcode (notably an SQLite I/O operation) has no wall-clock bound.
        sqlite3_progress_handler(db, 1, [](void* context) noexcept {
            auto& self = *static_cast<RequestScope*>(context);
            try {
                self.request.poll();
                if (self.request.result.stats.sqlSteps == self.request.options.sqlStepBudget)
                    self.request.stop(RegionStop::SqlBudget);
                ++self.request.result.stats.sqlSteps;
                return 0;
            } catch (...) { self.error = std::current_exception(); return 1; }
        }, this);
    }
    ~RequestScope() { sqlite3_progress_handler(db, 0, nullptr, nullptr); readRequest = nullptr; }
};
Point project(const RegionSlice& view, const Point& point) {
    Point p{};
    const std::array<Point, 3> axes{view.u, view.v, view.normal};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) p[i] += (point[j] - view.origin[j]) * axes[i][j];
    return p;
}
}

Bounds detail::slabBounds(const RegionSlice& view) {
    validateBounds(view.localBounds);
    const std::array<Point, 3> axes{view.u, view.v, view.normal};
    for (double x : view.origin) if (!std::isfinite(x)) throw std::invalid_argument("slice origin must be finite");
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        double dot = 0;
        for (int k = 0; k < 3; ++k) dot += axes[i][k] * axes[j][k];
        if (!std::isfinite(dot) || std::abs(dot - (i == j ? 1.0 : 0.0)) > 1e-12)
            throw std::invalid_argument("slice axes must be orthonormal");
    }
    // Invert the actual supplied projection, rather than treating approximately
    // orthogonal floating-point axes as an exact inverse.
    const auto cross = [](const Point& a, const Point& b) -> Point {
        return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    };
    const std::array<Point,3> inverse{cross(view.v,view.normal), cross(view.normal,view.u), cross(view.u,view.v)};
    double det = 0; for(int i=0;i<3;++i)det += view.u[i]*inverse[0][i];
    Bounds bounds;
    for (int i = 0; i < 3; ++i) {
        double lo=view.origin[i], hi=lo, magnitude=std::abs(lo);
        for (int j=0;j<3;++j) {
            const double a=inverse[j][i]/det*view.localBounds[j], b=inverse[j][i]/det*view.localBounds[j+3];
            lo+=std::min(a,b); hi+=std::max(a,b); magnitude+=std::max(std::abs(a),std::abs(b));
        }
        const double margin=32*std::numeric_limits<double>::epsilon()*std::max(1.0,magnitude);
        bounds[i]=lo-margin; bounds[i+3]=hi+margin;
    }
    validateBounds(bounds);
    return bounds;
}

RegionResult ObjectStore::geometryRegion(const ReadTxn& txn, const RegionSlice& view,
    const RegionOptions& requested, const RegionCancel& cancel) const
{
    auto options=requested;
    if(options.volume&&!options.transforms)options.transforms=std::make_shared<VolumeTransforms>(volumeTransforms(txn));
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (options.volume) {
        // One bounded query per declared frame, merged in UUID order with the
        // selected object first; the object limit applies to the merge.
        if (options.coordinates) throw std::invalid_argument("volume and coordinates filters are exclusive");
        if (!std::isfinite(options.targetScale) || options.targetScale <= 0)
            throw std::invalid_argument("region target scale must be finite and positive");
        RegionResult merged;
        std::optional<RegionStop> stopped;
        bool limited = false;
        for (const auto& frame : volumeFrames(db, *options.volume, options.kind, false, options.transforms.get())) {
            if (merged.stats.outputBytes >= options.outputByteBudget) { stopped = RegionStop::OutputBudget; break; }
            auto sub = options;
            sub.volume.reset();
            sub.coordinates = frame.coordinates;
            sub.outputByteBudget = options.outputByteBudget - merged.stats.outputBytes;
            sub.minLength = options.minLength;
            sub.toTarget=VolumeAffine::scale(options.targetScale)*frame.mapping;
            sub.processingByteBudget=options.processingByteBudget-std::min(options.processingByteBudget,merged.stats.processingBytes);
            sub.candidateWorkBudget=options.candidateWorkBudget-std::min(options.candidateWorkBudget,merged.stats.candidateWork);
            sub.sqlStepBudget=options.sqlStepBudget-std::min(options.sqlStepBudget,merged.stats.sqlSteps);
            auto part = geometryRegion(txn, view, sub, cancel);
            for (auto field : {&RegionStats::outputBytes, &RegionStats::processingBytes, &RegionStats::payloadBytes,
                               &RegionStats::verifiedBytes, &RegionStats::metadataBytes, &RegionStats::reconstructedBytes,
                               &RegionStats::cacheBytes, &RegionStats::candidateWork, &RegionStats::candidateVisits,
                               &RegionStats::segmentTests, &RegionStats::sqlSteps})
                merged.stats.*field += part.stats.*field;
            for (auto& run : part.runs) {
                merged.runs.push_back(std::move(run));
            }
            merged.remote.insert(merged.remote.end(), part.remote.begin(), part.remote.end());
            if (part.stop == RegionStop::ObjectLimit) limited = true;
            else if (!part.complete()) { stopped = part.stop; break; }
        }
        const auto selected = options.maximumObjects && options.selected
            ? std::optional(formatUuid(parseUuid(*options.selected))) : std::nullopt;
        std::stable_sort(merged.runs.begin(), merged.runs.end(), [&](const VisibleRun& a, const VisibleRun& b) {
            if ((a.objectUuid == selected) != (b.objectUuid == selected)) return a.objectUuid == selected;
            return a.objectUuid < b.objectUuid;
        });
        std::set<std::string> objects;
        size_t kept = 0;
        for (; kept < merged.runs.size(); ++kept) {
            const auto& uuid = merged.runs[kept].objectUuid;
            if (!objects.contains(uuid) && options.maximumObjects && objects.size() == options.maximumObjects) {
                limited = true;
                break;
            }
            objects.insert(uuid);
        }
        merged.runs.resize(kept);
        std::sort(merged.remote.begin(), merged.remote.end());
        merged.remote.erase(std::unique(merged.remote.begin(), merged.remote.end()), merged.remote.end());
        merged.intersectingObjects = objects.size();
        merged.stop = stopped ? *stopped : limited ? RegionStop::ObjectLimit : RegionStop::Exhausted;
        return merged;
    }
    const auto kind = regionKind(options.kind);
    const auto targetBounds = slabBounds(view);
    const auto bounds = options.toTarget ? transformBounds(targetBounds, options.toTarget->inverse()) : targetBounds;
    if (!std::isfinite(options.minLength) || options.minLength < 0 || options.minPoints < 2)
        throw std::invalid_argument("geometry region requires nonnegative finite length and minPoints >= 2");
    if (kind.surface && (options.family || options.minLength > 0))
        throw std::invalid_argument("surface regions have no family or length filter");
    std::optional<std::string> selected;
    if (options.maximumObjects && options.selected) selected = formatUuid(parseUuid(*options.selected));
    RegionResult result;
    ReadRequest request{options, result, cancel};
    RequestScope scope(db, request);
    // A surface's runs are its exact triangle mesh cut by the slab's mid-plane
    // (SurfaceCut), clipped to u/v.
    // Every surface tile in the slab, by object in row/column order: one R-tree
    // search per query instead of scanning each surface's tiles.
    struct Tile { int64_t row, column, rows, columns; Bounds box; };
    std::optional<std::map<int64_t, std::vector<Tile>>> surfaceTiles;
    auto tilesOf = [&](int64_t id) -> const std::vector<Tile>& {
        if (!surfaceTiles) {
            surfaceTiles.emplace();
            Statement tiles(db, "SELECT b.object_id,b.row_start,b.col_start,b.row_count,b.col_count,"
                "b.min_x,b.min_y,b.min_z,b.max_x,b.max_y,b.max_z FROM spatial_regions_rtree r "
                "CROSS JOIN spatial_regions b ON b.id=r.id WHERE b.array_name='points' AND " + spatialOverlapSql() +
                " ORDER BY b.object_id,b.row_start,b.col_start");
            for (int i = 0; i < 6; ++i) tiles.bind(i + 1, bounds[i]);
            while (tiles.row()) {
                Tile tile{tiles.i64(1), tiles.i64(2), tiles.i64(3), tiles.i64(4), {}};
                for (int i = 0; i < 6; ++i) tile.box[i] = tiles.f64(5 + i);
                (*surfaceTiles)[tiles.i64(0)].push_back(tile);
            }
        }
        static const std::vector<Tile> none;
        const auto found = surfaceTiles->find(id);
        return found == surfaceTiles->end() ? none : found->second;
    };
    auto processSurface = [&](const std::string& uuid, int64_t id, const std::string& revision) {
        if(options.toTarget && options.minLength>0 && transformedLength(db,revision,*options.toTarget,options.kind)<options.minLength)return;
        RetainedGeometryReader retained(db, revision);
        auto& reader = *retained;
        const auto& arrays = reader.manifest().at("arrays");
        const auto shape = arrays.at("points").at("shape").get<std::vector<int64_t>>();
        if (arrays.at("points").at("dtype") != kind.dtype) throw std::runtime_error("surface coordinates must be <f4");
        SurfaceCut cut(view, options.toTarget, shape.at(1));
        for (const auto& [row0, col0, rows, cols, box] : tilesOf(id)) {
            requestWork(&RegionStats::candidateVisits);
            if (rows < 2 || cols < 2 || !cut.reaches(box)) continue;
            const auto tile = surfaceTile(impl_->options.surfaceTiles.get(), revision, reader, row0, col0, rows, cols);
            result.stats.segmentTests += cut.add(*tile, row0, col0, rows, cols);
        }
        bool hit = false;
        cut.pieces([&](int64_t segment, bool continues, const auto& start, const auto& end) {
            // Each point is XYZ and its grid position.
            const uint64_t added = continues ? 40 : 80;
            if (added > options.outputByteBudget - result.stats.outputBytes) request.stop(RegionStop::OutputBudget);
            if (!continues) result.runs.push_back({uuid, revision, "points", "", segment, {start.point}, {start.grid}});
            result.runs.back().points.push_back(end.point);
            result.runs.back().grid.push_back(end.grid);
            result.stats.outputBytes += added;
            if (!hit) { hit = true; ++result.intersectingObjects; }
        });
        if (hit && options.maximumObjects && result.intersectingObjects == options.maximumObjects)
            request.stop(RegionStop::ObjectLimit);
    };
    auto process = [&](const std::string& uuid, int64_t id, const std::string& revision, const std::string& family) {
        if (kind.surface) return processSurface(uuid, id, revision);
        if(options.toTarget && options.minLength>0 && transformedLength(db,revision,*options.toTarget,options.kind)<options.minLength)return;
        RetainedGeometryReader retained(db, revision);
        auto& reader = *retained;
        const auto& arrays = reader.manifest().at("arrays");
        std::string name = kind.arrays.back();
        for (const auto& candidate : kind.arrays)
            if (arrays.at(candidate).at("shape").at(0).get<int64_t>()) { name = candidate; break; }
        const auto& array = arrays.at(name);
        const auto points = array.at("shape").at(0).get<int64_t>();
        if (points < options.minPoints) return;
        if (array.at("dtype") != kind.dtype)
            throw std::runtime_error(options.kind + " coordinates must be " + kind.dtype);
        bool hit = false;
        int64_t nextSegment = 0;
        auto appendRegion = [&](const SpatialRegion& region) {
            requestWork(&RegionStats::candidateVisits);
            const auto end = std::min(points - 1, region.start.at(0) + region.shape.at(0) - 1);
            auto start = std::max(nextSegment, region.start.at(0));
            for (; start < end;) {
                const auto count = std::min<int64_t>(256, end-start);
                auto bytes = reader.read(name, start*kind.stride, (count+1)*kind.stride);
                auto point = [&](int64_t i) {
                    const auto source = std::span<const uint8_t>(bytes).subspan(i*kind.stride,kind.stride);
                    Point p;
                    if (kind.stride == 24) copyScalars<double>(source,p);
                    else { std::array<int32_t,3> q; copyScalars<int32_t>(source,q); for(int k=0;k<3;++k)p[k]=q[k]; }
                    return options.toTarget?options.toTarget->apply(p):p;
                };
                Point a=point(0), pa=project(view,a);
                for (int64_t i=0;i<count;++i) {
                    requestWork(&RegionStats::segmentTests);
                    const Point b=point(i+1), pb=project(view,b);
                    double first,last;
                    if (vc::geometry::clipSegment(pa,pb,view.localBounds,first,last)) {
                        const int64_t index=start+i;
                        const bool contiguous=!result.runs.empty() && result.runs.back().objectUuid==uuid &&
                            result.runs.back().firstSegment+int64_t(result.runs.back().points.size())-1==index;
                        const uint64_t added=contiguous?24:48;
                        if (added>options.outputByteBudget-result.stats.outputBytes) request.stop(RegionStop::OutputBudget);
                        if (!contiguous) result.runs.push_back({uuid,revision,name,family,index,{a}});
                        result.runs.back().points.push_back(b);
                        result.stats.outputBytes+=added;
                        if (!hit) {hit=true;++result.intersectingObjects;}
                    }
                    a=b;pa=pb;
                }
                start+=count;
            }
            nextSegment=std::max(nextSegment,end);
        };
        // Regions overlap by one vertex, never by a segment.
        Statement regions(db, "SELECT b.array_name,b.row_start,b.row_count,b.min_x,b.min_y,b.min_z,b.max_x,b.max_y,b.max_z "
            "FROM spatial_regions b WHERE b.object_id=?7 AND b.array_name=?8 "
            "AND b.max_x>=?1 AND b.max_y>=?2 AND b.max_z>=?3 AND b.min_x<=?4 AND b.min_y<=?5 AND b.min_z<=?6 ORDER BY b.row_start");
        for(int i=0;i<6;++i)regions.bind(i+1,bounds[i]);
        regions.bind(7,id);regions.bind(8,name);
        while(regions.row()) {
            requestWork(&RegionStats::candidateVisits);
            SpatialRegion region;region.objectUuid=uuid;region.kind=options.kind;region.revisionUuid=revision;
            region.arrayName=regions.text(0);region.start={regions.i64(1)};region.shape={regions.i64(2)};
            for(int i=0;i<6;++i)region.bbox[i]=regions.f64(i+3);
            appendRegion(region);
        }
        if (hit && options.maximumObjects && result.intersectingObjects==options.maximumObjects)
            request.stop(RegionStop::ObjectLimit);
    };
    try {
        request.poll();
        std::optional<std::string> includedIds;
        if (options.uuids) {
            if (options.uuids->empty()) return result;
            std::string ids = "[";
            for (const auto& uuid : *options.uuids) {
                request.work(&RegionStats::candidateVisits);
                // Account for the ID vector, JSON representation, and SQL input.
                request.bytes(64, &RegionStats::metadataBytes);
                if (const auto id = objectIdByUuid(db, parseUuid(uuid))) {
                    const auto value = std::to_string(*id);
                    if (ids.size() + value.size() + 2 > options.maxUnitBytes)
                        request.stop(RegionStop::ProcessingBudget);
                    if (ids.size() > 1) ids += ',';
                    ids += value;
                }
            }
            if (ids.size() == 1) return result;
            ids += ']';
            includedIds = std::move(ids);
        }
        // Selected is attempted before any global priority-selection work.
        const auto query = [&](bool selectedOnly) {
            const std::string from = selectedOnly
                ? "objects o CROSS JOIN spatial_regions b ON b.object_id=o.id AND b.array_name='' CROSS JOIN spatial_regions_rtree r ON r.id=b.id "
                : "spatial_regions_rtree r CROSS JOIN spatial_regions b ON b.id=r.id CROSS JOIN objects o ON o.id=b.object_id ";
            // Surfaces have no family or length summary; their filters are rejected above.
            const std::string family = kind.surface ? "''" : "COALESCE(f.family,'')";
            std::string sql="SELECT o.uuid,o.id,h.revision_uuid," + family + " FROM " + from +
                "JOIN geometry_heads h ON h.object_uuid=o.uuid LEFT JOIN " + std::string(kind.table) + " f ON f.object_id=o.id "
                "LEFT JOIN object_lifecycle l ON l.object_id=o.id WHERE o.kind=?12 AND b.array_name='' "
                "AND COALESCE(l.retired,0)=0 AND "+spatialOverlapSql()+
                (kind.surface ? std::string() : " AND COALESCE(f.length,0)>=?7 AND (?8 IS NULL OR "+family+"=?8)");
            if(includedIds)sql+=" AND o.id IN (SELECT value FROM json_each(?10))";
            if(options.coordinates)sql+=coordinateFilterSql("o", "?11");
            if(selectedOnly)sql+=" AND o.uuid=?9";
            else if(selected)sql+=" AND o.uuid!=?9";
            // Start with spatial candidates, then order their metadata by UUID.
            // A small viewport must not scan every object before reaching it.
            // The sort and candidate probes remain inside the SQL work budget.
            if(!selectedOnly)sql+=" ORDER BY o.uuid";
            Statement candidates(db,sql);
            for(int i=0;i<6;++i)candidates.bind(i+1,bounds[i]);
            candidates.bind(7,options.toTarget?0.:options.minLength);
            if(options.family)candidates.bind(8,*options.family);else candidates.bindNull(8);
            if(selected)candidates.bind(9,parseUuid(*selected));
            if(includedIds)candidates.bind(10,*includedIds);
            if(options.coordinates)candidates.bind(11,coordinateFilterValue(*options.coordinates));
            candidates.bind(12,options.kind);
            while(candidates.row()) {
                requestWork(&RegionStats::candidateVisits);
                process(formatUuid(candidates.blob16(0)),candidates.i64(1),formatUuid(candidates.blob16(2)),candidates.text(3));
            }
        };
        if(selected)query(true);
        query(false);
        if (kind.surface) {
            result.remote = remoteSurfacesIn(db, bounds, includedIds, options.coordinates);
            if (!result.remote.empty() && impl_->options.remoteGeometryReached)
                impl_->options.remoteGeometryReached(result.remote);
        }
    } catch (...) {
        auto error=scope.error ? scope.error : std::current_exception();
        try { std::rethrow_exception(error); } catch (const ReadStopped&) {} // real failures still propagate
    }
    return result;
}
}
