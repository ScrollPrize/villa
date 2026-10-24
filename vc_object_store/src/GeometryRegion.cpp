#include "geometry/GeometryReader.hpp"
#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "internal/ReadRequest.hpp"
#include "internal/RecordSql.hpp"
#include "internal/CoordinateFilter.hpp"
#include "internal/VolumeFrames.hpp"
#include "geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/SegmentClip.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "geometry/ArrayLayout.hpp"
#include <deque>
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
// A crossing of the plane on the grid edge between two vertices (lower linear index first).
using GridEdge = std::pair<int64_t, int64_t>;
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
Bounds worldBounds(const RegionSlice& view) {
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
}

RegionResult ObjectStore::geometryRegion(const ReadTxn& txn, const RegionSlice& view,
    const RegionOptions& options, const RegionCancel& cancel) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (options.volume) {
        // One bounded query per declared frame, merged in UUID order with the
        // selected object first; the object limit applies to the merge.
        if (options.coordinates) throw std::invalid_argument("volume and coordinates filters are exclusive");
        RegionResult merged;
        std::optional<RegionStop> stopped;
        bool limited = false;
        for (const auto& frame : volumeFrames(db, *options.volume, options.kind, false)) {
            if (merged.stats.outputBytes >= options.outputByteBudget) { stopped = RegionStop::OutputBudget; break; }
            auto sub = options;
            sub.volume.reset();
            sub.coordinates = frame.coordinates;
            sub.outputByteBudget = options.outputByteBudget - merged.stats.outputBytes;
            sub.minLength = options.minLength > 0 ? lowestStoredAtLeast(options.minLength, frame.toVolume) : 0.0;
            RegionSlice stored = view;
            for (auto& coordinate : stored.origin) coordinate /= frame.toVolume;
            for (auto& bound : stored.localBounds) bound /= frame.toVolume;
            auto part = geometryRegion(txn, stored, sub, cancel);
            for (auto field : {&RegionStats::outputBytes, &RegionStats::processingBytes, &RegionStats::payloadBytes,
                               &RegionStats::verifiedBytes, &RegionStats::metadataBytes, &RegionStats::reconstructedBytes,
                               &RegionStats::cacheBytes, &RegionStats::candidateWork, &RegionStats::candidateVisits,
                               &RegionStats::segmentTests, &RegionStats::sqlSteps})
                merged.stats.*field += part.stats.*field;
            for (auto& run : part.runs) {
                for (auto& point : run.points) for (auto& coordinate : point) coordinate *= frame.toVolume;
                merged.runs.push_back(std::move(run));
            }
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
        merged.intersectingObjects = objects.size();
        merged.stop = stopped ? *stopped : limited ? RegionStop::ObjectLimit : RegionStop::Exhausted;
        return merged;
    }
    const auto kind = regionKind(options.kind);
    const auto bounds = worldBounds(view);
    if (!std::isfinite(options.minLength) || options.minLength < 0 || options.minPoints < 2)
        throw std::invalid_argument("geometry region requires nonnegative finite length and minPoints >= 2");
    if (kind.surface && (options.family || options.minLength > 0))
        throw std::invalid_argument("surface regions have no family or length filter");
    std::optional<std::string> selected;
    if (options.maximumObjects && options.selected) selected = formatUuid(parseUuid(*options.selected));
    RegionResult result;
    ReadRequest request{options, result, cancel};
    RequestScope scope(db, request);
    // A surface's runs are its exact triangle mesh cut by the slab's mid-plane:
    // cells whose four vertices are valid split along VC's diagonal into
    // (r,c),(r,c+1),(r+1,c) and (r,c+1),(r+1,c+1),(r+1,c). Crossings are computed
    // once per grid edge, so neighbouring triangles share exact points and chain
    // into polylines across tiles; the polylines are then clipped to u/v.
    const double planeOffset = (view.localBounds[2] + view.localBounds[5]) / 2;
    const Bounds planeBox{view.localBounds[0], view.localBounds[1], -1, view.localBounds[3], view.localBounds[4], 1};
    auto processSurface = [&](const std::string& uuid, int64_t id, const std::string& revision) {
        GeometryReader reader(db, revision);
        const auto& arrays = reader.manifest().at("arrays");
        const auto shape = arrays.at("points").at("shape").get<std::vector<int64_t>>();
        if (arrays.at("points").at("dtype") != kind.dtype) throw std::runtime_error("surface coordinates must be <f4");
        const bool hasValid = arrays.contains(surfaceValidChannel);
        const auto columns = shape.at(1);
        std::map<GridEdge, Point> crossings;
        std::vector<std::pair<GridEdge, GridEdge>> segments;
        Statement regions(db, "SELECT row_start,col_start,row_count,col_count FROM spatial_regions WHERE object_id=?7 "
            "AND array_name='points' AND max_x>=?1 AND max_y>=?2 AND max_z>=?3 AND min_x<=?4 AND min_y<=?5 AND min_z<=?6 "
            "ORDER BY row_start,col_start");
        for (int i = 0; i < 6; ++i) regions.bind(i + 1, bounds[i]);
        regions.bind(7, id);
        while (regions.row()) {
            requestWork(&RegionStats::candidateVisits);
            const auto row0 = regions.i64(0), col0 = regions.i64(1), rows = regions.i64(2), cols = regions.i64(3);
            if (rows < 2 || cols < 2) continue;
            const auto bytes = reader.rectangle("points", row0, col0, rows, cols);
            const auto valid = hasValid ? reader.rectangle(surfaceValidChannel, row0, col0, rows, cols) : std::vector<uint8_t>{};
            std::vector<Point> world(size_t(rows * cols));
            std::vector<double> distance(world.size());
            std::vector<uint8_t> usable(world.size());
            for (size_t i = 0; i < world.size(); ++i) {
                std::array<float, 3> q; copyScalars<float>(std::span(bytes).subspan(i * 12, 12), q);
                world[i] = {q[0], q[1], q[2]};
                usable[i] = vc::geometry::validSurfacePoint(q) && (!hasValid || valid[i]);
                distance[i] = project(view, world[i])[2] - planeOffset;
            }
            auto global = [&](int64_t r, int64_t c) { return (row0 + r) * columns + col0 + c; };
            auto crossing = [&](int64_t a, int64_t b, int64_t ga, int64_t gb) {
                if (ga > gb) { std::swap(a, b); std::swap(ga, gb); }
                const GridEdge edge{ga, gb};
                if (!crossings.contains(edge)) {
                    const double t = distance[a] / (distance[a] - distance[b]);
                    Point p;
                    for (int k = 0; k < 3; ++k) p[k] = world[a][k] + t * (world[b][k] - world[a][k]);
                    crossings.emplace(edge, p);
                }
                return edge;
            };
            for (int64_t r = 0; r + 1 < rows; ++r)
                for (int64_t c = 0; c + 1 < cols; ++c) {
                    const std::array<int64_t, 4> v{r * cols + c, r * cols + c + 1, (r + 1) * cols + c + 1, (r + 1) * cols + c};
                    if (!usable[v[0]] || !usable[v[1]] || !usable[v[2]] || !usable[v[3]]) continue;
                    const bool side0 = distance[v[0]] >= 0;
                    if (side0 == (distance[v[1]] >= 0) && side0 == (distance[v[2]] >= 0) && side0 == (distance[v[3]] >= 0)) continue;
                    const std::array<int64_t, 4> g{global(r, c), global(r, c + 1), global(r + 1, c + 1), global(r + 1, c)};
                    for (const auto& tri : {std::array<int, 3>{0, 1, 3}, std::array<int, 3>{1, 2, 3}}) {
                        ++result.stats.segmentTests;
                        std::vector<GridEdge> cut;
                        for (int e = 0; e < 3; ++e) {
                            const auto a = tri[e], b = tri[(e + 1) % 3];
                            if ((distance[v[a]] >= 0) != (distance[v[b]] >= 0)) cut.push_back(crossing(v[a], v[b], g[a], g[b]));
                        }
                        if (cut.size() == 2) segments.emplace_back(cut[0], cut[1]);
                    }
                }
        }
        // Chain segments through their shared grid edges, in emission order.
        std::map<GridEdge, std::vector<size_t>> incident;
        for (size_t i = 0; i < segments.size(); ++i) {
            incident[segments[i].first].push_back(i);
            incident[segments[i].second].push_back(i);
        }
        std::vector<bool> used(segments.size());
        bool hit = false;
        for (size_t start = 0; start < segments.size(); ++start) {
            if (used[start]) continue;
            used[start] = true;
            std::deque<GridEdge> chain{segments[start].first, segments[start].second};
            auto extend = [&](bool front) {
                for (;;) {
                    const auto end = front ? chain.front() : chain.back();
                    const auto& next = incident.at(end);
                    const auto it = std::find_if(next.begin(), next.end(), [&](size_t s) { return !used[s]; });
                    if (it == next.end()) return;
                    used[*it] = true;
                    const auto other = segments[*it].first == end ? segments[*it].second : segments[*it].first;
                    if (front) chain.push_front(other); else chain.push_back(other);
                }
            };
            extend(false); extend(true);
            // Clip the polyline to the slice's u/v extent; each inside stretch is a run.
            bool open = false;
            for (size_t i = 0; i + 1 < chain.size(); ++i) {
                const auto& a = crossings.at(chain[i]);
                const auto& b = crossings.at(chain[i + 1]);
                auto pa = project(view, a), pb = project(view, b);
                pa[2] = pb[2] = 0;
                double first, last;
                if (!vc::geometry::clipSegment(pa, pb, planeBox, first, last)) { open = false; continue; }
                auto at = [&](double t) {
                    if (t <= 0) return a;
                    if (t >= 1) return b;
                    Point p; for (int k = 0; k < 3; ++k) p[k] = a[k] + t * (b[k] - a[k]);
                    return p;
                };
                const uint64_t added = open && first <= 0 ? 24 : 48;
                if (added > options.outputByteBudget - result.stats.outputBytes) request.stop(RegionStop::OutputBudget);
                if (!open || first > 0) result.runs.push_back({uuid, revision, "points", "", int64_t(i), {at(first)}});
                result.runs.back().points.push_back(at(last));
                result.stats.outputBytes += added;
                open = last >= 1;
                if (!hit) { hit = true; ++result.intersectingObjects; }
            }
        }
        if (hit && options.maximumObjects && result.intersectingObjects == options.maximumObjects)
            request.stop(RegionStop::ObjectLimit);
    };
    auto process = [&](const std::string& uuid, int64_t id, const std::string& revision, const std::string& family) {
        if (kind.surface) return processSurface(uuid, id, revision);
        GeometryReader reader(db, revision);
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
                    return p;
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
            candidates.bind(7,options.minLength);
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
    } catch (...) {
        auto error=scope.error ? scope.error : std::current_exception();
        try { std::rethrow_exception(error); } catch (const ReadStopped&) {} // real failures still propagate
    }
    return result;
}
}
