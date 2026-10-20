#include "internal/Database.hpp"
#include "internal/Spatial.hpp"
#include "internal/ReadRequest.hpp"
#include "geometry/PersistentTree.hpp"
#include "geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/SegmentClip.hpp"
#include <exception>

namespace vc::objects {
using namespace detail;
const char* fiberRegionStopName(FiberRegionStop reason)
{
    switch (reason) {
    case FiberRegionStop::Exhausted: return "exhausted";
    case FiberRegionStop::FiberLimit: return "fiber_limit";
    case FiberRegionStop::OutputBudget: return "output_budget";
    case FiberRegionStop::ProcessingBudget: return "processing_budget";
    case FiberRegionStop::CandidateBudget: return "candidate_budget";
    case FiberRegionStop::SqlBudget: return "sql_budget";
    case FiberRegionStop::Cancelled: return "cancelled";
    }
    throw std::logic_error("invalid fiber region stop reason");
}
namespace {
using Point = std::array<double, 3>;
struct RequestScope {
    sqlite3* db;
    ReadRequest& request;
    std::exception_ptr error;
    RequestScope(sqlite3* connection, ReadRequest& r) : db(connection), request(r) {
        if (readRequest) throw std::logic_error("fiber region queries cannot be nested");
        readRequest = &r;
        // Single-opcode granularity gives a strict instruction allowance. An
        // opcode (notably an SQLite I/O operation) has no wall-clock bound.
        sqlite3_progress_handler(db, 1, [](void* context) noexcept {
            auto& self = *static_cast<RequestScope*>(context);
            try {
                self.request.poll();
                if (self.request.result.stats.sqlSteps == self.request.options.sqlStepBudget)
                    self.request.stop(FiberRegionStop::SqlBudget);
                ++self.request.result.stats.sqlSteps;
                return 0;
            } catch (...) { self.error = std::current_exception(); return 1; }
        }, this);
    }
    ~RequestScope() { sqlite3_progress_handler(db, 0, nullptr, nullptr); readRequest = nullptr; }
};
Point project(const FiberSlice& view, const Point& point) {
    Point p{};
    const std::array<Point, 3> axes{view.u, view.v, view.normal};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) p[i] += (point[j] - view.origin[j]) * axes[i][j];
    return p;
}
Bounds worldBounds(const FiberSlice& view) {
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

FiberRegionResult ObjectStore::fiberRegion(const ReadTxn& txn, const FiberSlice& view,
    const FiberRegionOptions& options, const FiberRegionCancel& cancel) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto bounds = worldBounds(view);
    if (!std::isfinite(options.minLength) || options.minLength < 0 || options.minPoints < 2)
        throw std::invalid_argument("fiber region requires nonnegative finite length and minPoints >= 2");
    std::optional<std::string> selected;
    if (options.maximumFibers && options.selected) selected = formatUuid(parseUuid(*options.selected));
    FiberRegionResult result;
    ReadRequest request{options, result, cancel};
    RequestScope scope(db, request);
    TreeCacheScope treeCache;
    auto process = [&](const std::string& uuid, int64_t id, const std::string& revision, const std::string& family) {
        GeometryReader reader(db, revision);
        const auto& arrays = reader.manifest().at("arrays");
        const std::string name = arrays.at("line_points").at("shape").at(0).get<int64_t>() ? "line_points" : "control_points";
        const auto& array = arrays.at(name);
        const auto points = array.at("shape").at(0).get<int64_t>();
        if (points < options.minPoints) return;
        if (array.at("dtype") != "<f8") throw std::runtime_error("fiber coordinates must be float64");
        bool hit = false;
        int64_t nextSegment = 0;
        auto appendRegion = [&](const SpatialRegion& region) {
            requestWork(&FiberRegionStats::candidateVisits);
            const auto end = std::min(points - 1, region.start.at(0) + region.shape.at(0) - 1);
            auto start = std::max(nextSegment, region.start.at(0));
            for (; start < end;) {
                const auto count = std::min<int64_t>(256, end-start);
                auto bytes = reader.read(name, start*24, (count+1)*24);
                auto point = [&](int64_t i) { Point p; copyScalars<double>(std::span<const uint8_t>(bytes).subspan(i*24,24),p); return p; };
                Point a=point(0), pa=project(view,a);
                for (int64_t i=0;i<count;++i) {
                    requestWork(&FiberRegionStats::segmentTests);
                    const Point b=point(i+1), pb=project(view,b);
                    double first,last;
                    if (vc::geometry::clipSegment(pa,pb,view.localBounds,first,last)) {
                        const int64_t index=start+i;
                        const bool contiguous=!result.runs.empty() && result.runs.back().objectUuid==uuid &&
                            result.runs.back().firstSegment+int64_t(result.runs.back().points.size())-1==index;
                        const uint64_t added=contiguous?24:48;
                        if (added>options.outputByteBudget-result.stats.outputBytes) request.stop(FiberRegionStop::OutputBudget);
                        if (!contiguous) result.runs.push_back({uuid,revision,name,family,index,{a}});
                        result.runs.back().points.push_back(b);
                        result.stats.outputBytes+=added;
                        if (!hit) {hit=true;++result.intersectingFibers;}
                    }
                    a=b;pa=pb;
                }
                start+=count;
            }
            nextSegment=std::max(nextSegment,end);
        };
        // Indexed regions are emitted in segment order, including structural
        // seams. Legacy regions overlap by one vertex, never by a segment.
        Statement regions(db, "SELECT b.array_name,b.row_start,b.row_count,b.min_x,b.min_y,b.min_z,b.max_x,b.max_y,b.max_z "
            "FROM spatial_regions b WHERE b.object_id=?7 AND b.array_name=?8 "
            "AND b.max_x>=?1 AND b.max_y>=?2 AND b.max_z>=?3 AND b.min_x<=?4 AND b.min_y<=?5 AND b.min_z<=?6 ORDER BY b.row_start");
        for(int i=0;i<6;++i)regions.bind(i+1,bounds[i]);
        regions.bind(7,id);regions.bind(8,array.contains("tree") ? "@tree:"+name : name);
        while(regions.row()) {
            requestWork(&FiberRegionStats::candidateVisits);
            SpatialRegion region;region.objectUuid=uuid;region.kind="fiber";region.revisionUuid=revision;
            region.arrayName=regions.text(0);region.start={regions.i64(1)};region.shape={regions.i64(2)};
            for(int i=0;i<6;++i)region.bbox[i]=regions.f64(i+3);
            visitIndexedRegions(db,region,bounds,reader,appendRegion);
        }
        if (hit && options.maximumFibers && result.intersectingFibers==options.maximumFibers)
            request.stop(FiberRegionStop::FiberLimit);
    };
    try {
        request.poll();
        // Selected is attempted before any global priority-selection work.
        const auto query = [&](bool selectedOnly) {
            std::string sql="SELECT o.uuid,o.id,h.revision_uuid,COALESCE(f.family,'') FROM objects o INDEXED BY objects_kind_uuid "
                "CROSS JOIN spatial_regions b ON b.object_id=o.id AND b.array_name='' "
                "CROSS JOIN spatial_regions_rtree r ON r.id=b.id "
                "JOIN geometry_heads h ON h.object_uuid=o.uuid LEFT JOIN fibers f ON f.object_id=o.id "
                "LEFT JOIN object_lifecycle l ON l.object_id=o.id WHERE o.kind='fiber' AND b.array_name='' "
                "AND COALESCE(l.retired,0)=0 AND "+spatialOverlapSql()+
                " AND COALESCE(f.length,0)>=?7 AND (?8 IS NULL OR COALESCE(f.family,'')=?8)";
            if(selectedOnly)sql+=" AND o.uuid=?9";
            else if(selected)sql+=" AND o.uuid!=?9";
            // Drive the UUID index to stream stable priorities without a sorter.
            // Nonspatial object probes also consume the SQL work budget.
            // There is no LIMIT/OFFSET replay and no candidate coverage cache.
            if(!selectedOnly)sql+=" ORDER BY o.uuid";
            Statement candidates(db,sql);
            for(int i=0;i<6;++i)candidates.bind(i+1,bounds[i]);
            candidates.bind(7,options.minLength);
            if(options.family)candidates.bind(8,*options.family);else candidates.bindNull(8);
            if(selected)candidates.bindBlob(9,parseUuid(*selected));
            while(candidates.row()) {
                requestWork(&FiberRegionStats::candidateVisits);
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
