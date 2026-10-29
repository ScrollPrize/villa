#include "vc/fiber_tracer/FiberSideStripIntersections.hpp"

#include <vc_object_store/geometry/Geometry.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace vc::fiber_tracer {
namespace {

constexpr double kEpsilon = 1.0e-12;

using Point3 = bg::model::point<double, 3, bg::cs::cartesian>;
using Box3 = bg::model::box<Point3>;

double dot(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const cv::Vec3d& v)
{
    return std::sqrt(std::max(0.0, dot(v, v)));
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    const double n = norm(v);
    if (n <= kEpsilon ||
        !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

bool finitePoint(const cv::Vec3d& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

Point3 bgPoint(const cv::Vec3d& p)
{
    return Point3(p[0], p[1], p[2]);
}

Box3 bgBox(const cv::Vec3d& mn, const cv::Vec3d& mx)
{
    return Box3(bgPoint(mn), bgPoint(mx));
}

std::vector<double> cumulativeArclengths(const FiberPolyline& fiber)
{
    std::vector<double> lengths(fiber.points.size(), 0.0);
    for (size_t i = 1; i < fiber.points.size(); ++i) {
        const double step = norm(fiber.points[i] - fiber.points[i - 1]);
        lengths[i] = lengths[i - 1] + (std::isfinite(step) ? step : 0.0);
    }
    return lengths;
}

double squaredDistance(const cv::Vec3d& a, const cv::Vec3d& b)
{
    const cv::Vec3d delta = a - b;
    return dot(delta, delta);
}

struct StripTriangle {
    cv::Vec3d p0{0.0, 0.0, 0.0};
    cv::Vec3d p1{0.0, 0.0, 0.0};
    cv::Vec3d p2{0.0, 0.0, 0.0};
    cv::Vec2d uv0{0.0, 0.0};
    cv::Vec2d uv1{0.0, 0.0};
    cv::Vec2d uv2{0.0, 0.0};
    cv::Vec3d aabbMin{0.0, 0.0, 0.0};
    cv::Vec3d aabbMax{0.0, 0.0, 0.0};
};

struct StripTriangleHit {
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec2d uv{0.0, 0.0};
    double lineParameter = 0.0;
};

using TriangleRTreeValue = std::pair<Box3, size_t>;
using TriangleTree = bgi::rtree<TriangleRTreeValue, bgi::quadratic<32>>;

cv::Vec3d toVec3d(const cv::Vec3f& point)
{
    return {static_cast<double>(point[0]),
            static_cast<double>(point[1]),
            static_cast<double>(point[2])};
}

cv::Vec3d componentMin(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return {std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2])};
}

cv::Vec3d componentMax(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return {std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2])};
}

Box3 paddedBox(const cv::Vec3d& mn, const cv::Vec3d& mx, double padding)
{
    const double p = std::isfinite(padding) && padding > 0.0 ? padding : 0.0;
    const cv::Vec3d pad{p, p, p};
    return bgBox(mn - pad, mx + pad);
}

size_t sideStripWorkerCount(size_t requestedWorkers, size_t workItems)
{
    if (workItems == 0) {
        return 1;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t defaultWorkers =
        std::max<size_t>(1, static_cast<size_t>(hw > 1 ? (hw + 1) / 2 : 1));
    const size_t requested =
        requestedWorkers > 0 ? requestedWorkers : defaultWorkers;
    return std::max<size_t>(1, std::min(requested, workItems));
}

bool makeStripTriangle(const cv::Vec3d& p0,
                       const cv::Vec3d& p1,
                       const cv::Vec3d& p2,
                       const cv::Vec2d& uv0,
                       const cv::Vec2d& uv1,
                       const cv::Vec2d& uv2,
                       StripTriangle& triangle)
{
    if (!finitePoint(p0) || !finitePoint(p1) || !finitePoint(p2)) {
        return false;
    }
    const cv::Vec3d normal = (p1 - p0).cross(p2 - p0);
    if (norm(normal) <= kEpsilon) {
        return false;
    }
    triangle.p0 = p0;
    triangle.p1 = p1;
    triangle.p2 = p2;
    triangle.uv0 = uv0;
    triangle.uv1 = uv1;
    triangle.uv2 = uv2;
    triangle.aabbMin = componentMin(componentMin(p0, p1), p2);
    triangle.aabbMax = componentMax(componentMax(p0, p1), p2);
    return true;
}

std::vector<StripTriangle> stripTrianglesForPoints(const cv::Mat_<cv::Vec3f>& points)
{
    std::vector<StripTriangle> triangles;
    if (points.rows < 2 || points.cols < 2) {
        return triangles;
    }
    triangles.reserve(static_cast<size_t>(points.rows - 1) *
                      static_cast<size_t>(points.cols - 1) * 2);
    for (int row = 0; row + 1 < points.rows; ++row) {
        for (int col = 0; col + 1 < points.cols; ++col) {
            const cv::Vec3f p00f = points(row, col);
            const cv::Vec3f p01f = points(row, col + 1);
            const cv::Vec3f p10f = points(row + 1, col);
            const cv::Vec3f p11f = points(row + 1, col + 1);
            if (!vc::geometry::validSurfacePoint(p00f) ||
                !vc::geometry::validSurfacePoint(p01f) ||
                !vc::geometry::validSurfacePoint(p10f) ||
                !vc::geometry::validSurfacePoint(p11f)) {
                continue;
            }
            const cv::Vec3d p00 = toVec3d(p00f);
            const cv::Vec3d p01 = toVec3d(p01f);
            const cv::Vec3d p10 = toVec3d(p10f);
            const cv::Vec3d p11 = toVec3d(p11f);
            StripTriangle triangle;
            if (makeStripTriangle(p00,
                                  p10,
                                  p11,
                                  {static_cast<double>(row), static_cast<double>(col)},
                                  {static_cast<double>(row + 1), static_cast<double>(col)},
                                  {static_cast<double>(row + 1), static_cast<double>(col + 1)},
                                  triangle)) {
                triangles.push_back(triangle);
            }
            if (makeStripTriangle(p00,
                                  p11,
                                  p01,
                                  {static_cast<double>(row), static_cast<double>(col)},
                                  {static_cast<double>(row + 1), static_cast<double>(col + 1)},
                                  {static_cast<double>(row), static_cast<double>(col + 1)},
                                  triangle)) {
                triangles.push_back(triangle);
            }
        }
    }
    return triangles;
}

std::optional<StripTriangleHit> intersectLineTriangle(const cv::Vec3d& origin,
                                                      const cv::Vec3d& direction,
                                                      const StripTriangle& triangle,
                                                      std::optional<std::pair<double, double>> lineRange)
{
    if (!finitePoint(origin) || !finitePoint(direction)) {
        return std::nullopt;
    }
    const cv::Vec3d edge1 = triangle.p1 - triangle.p0;
    const cv::Vec3d edge2 = triangle.p2 - triangle.p0;
    const cv::Vec3d h = direction.cross(edge2);
    const double det = dot(edge1, h);
    if (!std::isfinite(det) || std::abs(det) <= kEpsilon) {
        return std::nullopt;
    }

    const double invDet = 1.0 / det;
    const cv::Vec3d s = origin - triangle.p0;
    const double u = invDet * dot(s, h);
    if (!std::isfinite(u) || u < -kEpsilon || u > 1.0 + kEpsilon) {
        return std::nullopt;
    }

    const cv::Vec3d q = s.cross(edge1);
    const double v = invDet * dot(direction, q);
    if (!std::isfinite(v) || v < -kEpsilon || u + v > 1.0 + kEpsilon) {
        return std::nullopt;
    }

    const double t = invDet * dot(edge2, q);
    if (!std::isfinite(t)) {
        return std::nullopt;
    }
    if (lineRange && (t < lineRange->first - kEpsilon ||
                      t > lineRange->second + kEpsilon)) {
        return std::nullopt;
    }

    const double bary0 = std::clamp(1.0 - u - v, 0.0, 1.0);
    const double bary1 = std::clamp(u, 0.0, 1.0);
    const double bary2 = std::clamp(v, 0.0, 1.0);
    StripTriangleHit hit;
    hit.point = origin + direction * t;
    hit.uv = triangle.uv0 * bary0 + triangle.uv1 * bary1 + triangle.uv2 * bary2;
    hit.lineParameter = t;
    if (!finitePoint(hit.point) ||
        !std::isfinite(hit.uv[0]) ||
        !std::isfinite(hit.uv[1])) {
        return std::nullopt;
    }
    return hit;
}

double branchLinkProjectionScore(const FiberSideStripIntersection& intersection)
{
    if (finitePoint(intersection.projectionTarget) && finitePoint(intersection.point)) {
        return squaredDistance(intersection.projectionTarget, intersection.point);
    }
    return std::abs(intersection.distance);
}

bool sameBranchLinkProjection(const FiberSideStripIntersection& lhs,
                              const FiberSideStripIntersection& rhs)
{
    return lhs.source == FiberSideStripIntersectionSource::BranchLink &&
           rhs.source == FiberSideStripIntersectionSource::BranchLink &&
           lhs.fiberId == rhs.fiberId &&
           finitePoint(lhs.connectorStart) &&
           finitePoint(rhs.connectorStart) &&
           finitePoint(lhs.projectionTarget) &&
           finitePoint(rhs.projectionTarget) &&
           lhs.branchLinkIndex == rhs.branchLinkIndex &&
           lhs.branchLinkIndex != std::numeric_limits<size_t>::max() &&
           squaredDistance(lhs.connectorStart, rhs.connectorStart) <= 1.0e-12 &&
           squaredDistance(lhs.projectionTarget, rhs.projectionTarget) <= 1.0e-12;
}

std::vector<FiberSideStripIntersection> keepNearestBranchLinkProjections(
    std::vector<FiberSideStripIntersection> intersections)
{
    std::vector<FiberSideStripIntersection> filtered;
    filtered.reserve(intersections.size());
    for (auto& intersection : intersections) {
        if (intersection.source != FiberSideStripIntersectionSource::BranchLink ||
            !finitePoint(intersection.connectorStart) ||
            !finitePoint(intersection.projectionTarget)) {
            filtered.push_back(std::move(intersection));
            continue;
        }

        const auto existing = std::find_if(
            filtered.begin(),
            filtered.end(),
            [&intersection](const FiberSideStripIntersection& candidate) {
                return sameBranchLinkProjection(candidate, intersection);
            });
        if (existing == filtered.end()) {
            filtered.push_back(std::move(intersection));
            continue;
        }
        if (branchLinkProjectionScore(intersection) <
            branchLinkProjectionScore(*existing)) {
            *existing = std::move(intersection);
        }
    }
    return filtered;
}

} // namespace

std::vector<FiberSideStripIntersection> fiberSideStripIntersections(
    const FiberSideStripQueryOptions& options,
    FiberSideStripProgressCallback progressCallback,
    FiberSideStripCancelCallback cancelCallback)
{
    if (progressCallback) {
        progressCallback(FiberSideStripProgressPhase::BuildStripTriangles, 0, 0);
    }
    const std::vector<StripTriangle> triangles =
        stripTrianglesForPoints(options.stripPoints);
    if (triangles.empty()) {
        return {};
    }

    const double padding =
        std::isfinite(options.aabbPadding) && options.aabbPadding >= 0.0
            ? options.aabbPadding
            : 0.0;

    std::unordered_set<uint64_t> excluded(options.excludedFiberIds.begin(),
                                          options.excludedFiberIds.end());

    std::vector<FiberSideStripIntersection> intersections;

    struct PreparedBranchLink {
        const FiberSideStripLineQuery* link = nullptr;
        cv::Vec3d direction{0.0, 0.0, 0.0};
        bool valid = false;
    };

    std::vector<PreparedBranchLink> branchLinks;
    branchLinks.reserve(options.branchLinks.size());
    for (const auto& link : options.branchLinks) {
        PreparedBranchLink prepared;
        prepared.link = &link;
        prepared.direction = normalizedOrZero(link.direction);
        prepared.valid =
            link.fiberId != 0 &&
            finitePoint(link.point) &&
            norm(prepared.direction) > kEpsilon;
        branchLinks.push_back(prepared);
    }

    const size_t branchTriangleWork = branchLinks.size() * triangles.size();
    if (progressCallback) {
        progressCallback(FiberSideStripProgressPhase::BranchLinks,
                         0,
                         branchTriangleWork);
    }
    if (branchTriangleWork > 0) {
        constexpr size_t kBranchChunkSize = 1024;
        const size_t chunkCount =
            (branchTriangleWork + kBranchChunkSize - 1) / kBranchChunkSize;
        const size_t workerCount =
            sideStripWorkerCount(options.workerThreads, chunkCount);
        std::atomic<size_t> nextChunk{0};
        std::atomic<size_t> completedBranchWork{0};
        std::atomic<bool> cancelled{false};
        std::exception_ptr firstException;
        std::mutex exceptionMutex;
        std::vector<std::vector<FiberSideStripIntersection>> workerIntersections(workerCount);
        std::vector<std::thread> workers;
        workers.reserve(workerCount);

        auto scanBranchChunks = [&](size_t workerIndex) {
            try {
                auto& local = workerIntersections[workerIndex];
                while (!cancelled.load(std::memory_order_relaxed)) {
                    const size_t chunk = nextChunk.fetch_add(1, std::memory_order_relaxed);
                    if (chunk >= chunkCount) {
                        break;
                    }
                    const size_t begin = chunk * kBranchChunkSize;
                    const size_t end = std::min(branchTriangleWork, begin + kBranchChunkSize);
                    for (size_t flatIndex = begin; flatIndex < end; ++flatIndex) {
                        if (cancelCallback && cancelCallback()) {
                            cancelled.store(true, std::memory_order_relaxed);
                            break;
                        }
                        const size_t linkIndex = flatIndex / triangles.size();
                        const size_t triangleIndex = flatIndex % triangles.size();
                        const auto& link = branchLinks[linkIndex];
                        if (!link.valid || !link.link) {
                            continue;
                        }
                        const auto& triangle = triangles[triangleIndex];
                        const auto hit = intersectLineTriangle(link.link->point,
                                                               link.direction,
                                                               triangle,
                                                               std::nullopt);
                        if (!hit) {
                            continue;
                        }
                        local.push_back(FiberSideStripIntersection{
                            link.link->fiberId,
                            uint64_t{1},
                            -1,
                            hit->lineParameter,
                            hit->point,
                            hit->uv[0],
                            hit->uv[1],
                            std::abs(hit->lineParameter),
                            FiberSideStripIntersectionSource::BranchLink,
                            link.link->connectorStart,
                            link.link->point,
                            linkIndex});
                    }
                    completedBranchWork.fetch_add(end - begin, std::memory_order_relaxed);
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(exceptionMutex);
                    if (!firstException) {
                        firstException = std::current_exception();
                    }
                }
                cancelled.store(true, std::memory_order_relaxed);
            }
        };

        for (size_t worker = 0; worker < workerCount; ++worker) {
            workers.emplace_back(scanBranchChunks, worker);
        }
        size_t lastProgress = 0;
        while (completedBranchWork.load(std::memory_order_relaxed) < branchTriangleWork &&
               !cancelled.load(std::memory_order_relaxed)) {
            if (cancelCallback && cancelCallback()) {
                cancelled.store(true, std::memory_order_relaxed);
                break;
            }
            const size_t completed =
                completedBranchWork.load(std::memory_order_relaxed);
            if (progressCallback &&
                (completed == branchTriangleWork ||
                 completed >= lastProgress + 4096)) {
                progressCallback(FiberSideStripProgressPhase::BranchLinks,
                                 completed,
                                 branchTriangleWork);
                lastProgress = completed;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        if (firstException) {
            std::rethrow_exception(firstException);
        }
        if (cancelled.load(std::memory_order_relaxed) ||
            (cancelCallback && cancelCallback())) {
            return {};
        }
        for (auto& local : workerIntersections) {
            intersections.insert(intersections.end(), local.begin(), local.end());
        }
        if (progressCallback) {
            progressCallback(FiberSideStripProgressPhase::BranchLinks,
                             branchTriangleWork,
                             branchTriangleWork);
        }
    }

    std::vector<const FiberPolyline*> fibers;
    if (!options.queryFibers.empty()) {
        fibers.reserve(options.queryFibers.size());
        for (const FiberPolyline* fiber : options.queryFibers) {
            if (fiber) {
                fibers.push_back(fiber);
            }
        }
    }

    if (!fibers.empty()) {
        std::vector<TriangleRTreeValue> triangleValues;
            triangleValues.reserve(triangles.size());
            if (progressCallback) {
                progressCallback(FiberSideStripProgressPhase::BuildTriangleIndex, 0, triangles.size());
            }
            for (size_t i = 0; i < triangles.size(); ++i) {
                if (cancelCallback && i % 4096 == 0 && cancelCallback()) {
                    return {};
                }
                triangleValues.emplace_back(paddedBox(triangles[i].aabbMin,
                                                     triangles[i].aabbMax,
                                                     padding),
                                            i);
                if (progressCallback && ((i + 1) == triangles.size() || (i + 1) % 4096 == 0)) {
                    progressCallback(FiberSideStripProgressPhase::BuildTriangleIndex,
                                     i + 1,
                                     triangles.size());
                }
            }
            const TriangleTree triangleTree(triangleValues.begin(), triangleValues.end());

            struct SegmentJob {
                const FiberPolyline* fiber = nullptr;
                size_t pointIndex = 0;
                double startArclength = 0.0;
                double endArclength = 0.0;
            };
            std::vector<SegmentJob> segmentJobs;
            for (const FiberPolyline* fiber : fibers) {
                if (!fiber || excluded.find(fiber->id) != excluded.end() ||
                    fiber->points.size() < 2) {
                    continue;
                }
                const auto lengths = cumulativeArclengths(*fiber);
                for (size_t i = 1; i < fiber->points.size(); ++i) {
                    segmentJobs.push_back(SegmentJob{
                        fiber,
                        i,
                        lengths[i - 1],
                        lengths[i]});
                }
            }

            if (progressCallback) {
                progressCallback(FiberSideStripProgressPhase::FiberSegments,
                                 0,
                                 segmentJobs.size());
            }
            if (!segmentJobs.empty()) {
                const size_t workerCount =
                    sideStripWorkerCount(options.workerThreads, segmentJobs.size());
                std::atomic<size_t> nextJob{0};
                std::atomic<size_t> completedSegments{0};
                std::atomic<bool> cancelled{false};
                std::exception_ptr firstException;
                std::mutex exceptionMutex;
                std::vector<std::vector<FiberSideStripIntersection>> workerIntersections(workerCount);
                std::vector<std::thread> workers;
                workers.reserve(workerCount);

                auto scanSegments = [&](size_t workerIndex) {
                    try {
                        auto& local = workerIntersections[workerIndex];
                        while (!cancelled.load(std::memory_order_relaxed)) {
                            const size_t jobIndex =
                                nextJob.fetch_add(1, std::memory_order_relaxed);
                            if (jobIndex >= segmentJobs.size()) {
                                break;
                            }
                            if (cancelCallback && cancelCallback()) {
                                cancelled.store(true, std::memory_order_relaxed);
                                break;
                            }
                            const SegmentJob& job = segmentJobs[jobIndex];
                            const FiberPolyline* fiber = job.fiber;
                            if (!fiber || job.pointIndex == 0 ||
                                job.pointIndex >= fiber->points.size()) {
                                completedSegments.fetch_add(1, std::memory_order_relaxed);
                                continue;
                            }
                            const cv::Vec3d a =
                                fiber->points[job.pointIndex - 1];
                            const cv::Vec3d b =
                                fiber->points[job.pointIndex];
                            if (finitePoint(a) && finitePoint(b)) {
                                const cv::Vec3d segment = b - a;
                                const double segmentLength = norm(segment);
                                if (std::isfinite(segmentLength) &&
                                    segmentLength > kEpsilon) {
                                    const Box3 segmentBox =
                                        paddedBox(componentMin(a, b),
                                                  componentMax(a, b),
                                                  padding);
                                    std::vector<TriangleRTreeValue> candidates;
                                    triangleTree.query(bgi::intersects(segmentBox),
                                                       std::back_inserter(candidates));
                                    std::sort(candidates.begin(),
                                              candidates.end(),
                                              [](const auto& lhs, const auto& rhs) {
                                                  return lhs.second < rhs.second;
                                              });

                                    for (const auto& candidate : candidates) {
                                        if (cancelCallback && cancelCallback()) {
                                            cancelled.store(true,
                                                            std::memory_order_relaxed);
                                            break;
                                        }
                                        const auto& triangle = triangles[candidate.second];
                                        const auto hit =
                                            intersectLineTriangle(a,
                                                                  segment,
                                                                  triangle,
                                                                  std::make_pair(0.0, 1.0));
                                        if (!hit) {
                                            continue;
                                        }
                                        const double arclength =
                                            job.startArclength +
                                            (job.endArclength - job.startArclength) *
                                                hit->lineParameter;
                                        local.push_back(FiberSideStripIntersection{
                                            fiber->id,
                                            fiber->generation,
                                            static_cast<int>(job.pointIndex - 1),
                                            arclength,
                                            hit->point,
                                            hit->uv[0],
                                            hit->uv[1],
                                            arclength,
                                            FiberSideStripIntersectionSource::FiberSegment});
                                    }
                                }
                            }
                            completedSegments.fetch_add(1, std::memory_order_relaxed);
                        }
                    } catch (...) {
                        {
                            std::lock_guard<std::mutex> lock(exceptionMutex);
                            if (!firstException) {
                                firstException = std::current_exception();
                            }
                        }
                        cancelled.store(true, std::memory_order_relaxed);
                    }
                };

                for (size_t worker = 0; worker < workerCount; ++worker) {
                    workers.emplace_back(scanSegments, worker);
                }
                size_t lastProgress = 0;
                while (completedSegments.load(std::memory_order_relaxed) <
                           segmentJobs.size() &&
                       !cancelled.load(std::memory_order_relaxed)) {
                    if (cancelCallback && cancelCallback()) {
                        cancelled.store(true, std::memory_order_relaxed);
                        break;
                    }
                    const size_t completed =
                        completedSegments.load(std::memory_order_relaxed);
                    if (progressCallback &&
                        (completed == segmentJobs.size() ||
                         completed >= lastProgress + 16)) {
                        progressCallback(FiberSideStripProgressPhase::FiberSegments,
                                         completed,
                                         segmentJobs.size());
                        lastProgress = completed;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                }
                for (auto& worker : workers) {
                    if (worker.joinable()) {
                        worker.join();
                    }
                }
                if (firstException) {
                    std::rethrow_exception(firstException);
                }
                if (cancelled.load(std::memory_order_relaxed) ||
                    (cancelCallback && cancelCallback())) {
                    return {};
                }
                for (auto& local : workerIntersections) {
                    intersections.insert(intersections.end(), local.begin(), local.end());
                }
                if (progressCallback) {
                    progressCallback(FiberSideStripProgressPhase::FiberSegments,
                                     segmentJobs.size(),
                                     segmentJobs.size());
                }
        }
    }

    if (progressCallback) {
        progressCallback(FiberSideStripProgressPhase::Deduplicate, 0, intersections.size());
    }
    std::sort(intersections.begin(), intersections.end(), [](const auto& a, const auto& b) {
        if (a.source != b.source) {
            return static_cast<int>(a.source) < static_cast<int>(b.source);
        }
        if (a.fiberId != b.fiberId) return a.fiberId < b.fiberId;
        if (a.stripCol != b.stripCol) return a.stripCol < b.stripCol;
        if (a.stripRow != b.stripRow) return a.stripRow < b.stripRow;
        return a.arclength < b.arclength;
    });

    const double stripDistance =
        std::isfinite(options.deduplicateStripDistance) && options.deduplicateStripDistance > 0.0
            ? options.deduplicateStripDistance
            : 0.0;
    std::vector<FiberSideStripIntersection> clustered;
    clustered.reserve(intersections.size());
    for (const auto& intersection : intersections) {
        bool duplicate = false;
        for (const auto& kept : clustered) {
            if (kept.source != intersection.source) {
                continue;
            }
            if (intersection.source == FiberSideStripIntersectionSource::BranchLink) {
                if (!sameBranchLinkProjection(kept, intersection)) {
                    continue;
                }
            } else if (kept.fiberId != intersection.fiberId) {
                continue;
            }
            if (std::abs(kept.stripRow - intersection.stripRow) <= stripDistance &&
                std::abs(kept.stripCol - intersection.stripCol) <= stripDistance) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            clustered.push_back(intersection);
        }
    }

    if (progressCallback) {
        progressCallback(FiberSideStripProgressPhase::Deduplicate,
                         intersections.size(),
                         intersections.size());
    }
    auto filtered = keepNearestBranchLinkProjections(std::move(clustered));
    if (options.maxResults > 0 && filtered.size() > options.maxResults) {
        filtered.resize(options.maxResults);
    }
    return filtered;
}

} // namespace vc::fiber_tracer
