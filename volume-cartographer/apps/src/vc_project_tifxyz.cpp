#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <omp.h>

#include "utils/Json.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Umbilicus.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;
using Json = utils::Json;
using vc::core::util::Umbilicus;

namespace {

const cv::Vec3f kInvalid(-1.0f, -1.0f, -1.0f);

struct Config {
    fs::path source;
    fs::path umbilicus;
    fs::path patchDir;
    fs::path output;
    std::string direction = "out";
    std::optional<float> zMin;
    std::optional<float> zMax;
    float maxDistance = 500.0f;
    float minDistance = 1.0f;
    float dedupRadius = 2.0f;
    float sameWrapTolerance = 3.0f;
    float spacing = 20.0f;
    float neighborMaxDelta = 40.0f;
    float neighborMaxFartherDelta = 0.0f;
    int rayNeighborhoodRadius = 1;
    int rayNeighborhoodMinSamples = 1;
    float rayNeighborhoodMaxDelta = 40.0f;
    bool rayNeighborhoodAverageDirectHits = false;
    int smoothRadius = 2;
    int smoothIters = 2;
    int indexStride = 1;
    int selfHitIgnoreRadius = 2;
    float selfHitEpsilon = 0.5f;
    float bboxPadding = 1.0f;
    bool snapToPatch = false;
    float snapMinDistance = 0.0f;
    float snapMaxDistance = 20.0f;
    int snapInterpolateRadius = 1;
    int snapInterpolateIters = 2;
    int snapInterpolateMinNeighbors = 1;
    int repairIters = 2;
    int repairRadius = 2;
    int repairMinNeighbors = 3;
    float repairCloserDelta = 30.0f;
    float repairEdgeFactor = 3.0f;
    float repairBacktrack = 40.0f;
    float repairForward = 80.0f;
    float repairMinMove = 1.0f;
    float repairPatchBonus = 10.0f;
    float repairPatchPenalty = 50.0f;
    bool noIndexCache = false;
    fs::path indexCacheDir;
    int threads = 0;
};

struct RayHit {
    float t = 0.0f;
    cv::Vec3f point = kInvalid;
    cv::Vec3f surfaceParam = kInvalid;
    SurfacePatchIndex::SurfacePtr surface;
    int i = 0;
    int j = 0;
    int triangleIndex = 0;
};

struct RayTriangleIntersection {
    float t = 0.0f;
    cv::Vec3f bary = {0.0f, 0.0f, 0.0f};
};

bool valid_point(const cv::Vec3f& p)
{
    return p[0] != -1.0f && p[1] != -1.0f && p[2] != -1.0f
        && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

bool finite_vec(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

bool valid_surface_param(const cv::Vec3f& p)
{
    return finite_vec(p) && std::abs(p[2]) <= 1e-4f;
}

float z_min_from_bbox(const Json& meta)
{
    return meta["bbox"][0][2].get_float();
}

float z_max_from_bbox(const Json& meta)
{
    return meta["bbox"][1][2].get_float();
}

bool bbox_intersects_z(const Json& meta, float zMin, float zMax)
{
    if (!meta.contains("bbox") || !meta["bbox"].is_array()) {
        return true;
    }
    return z_max_from_bbox(meta) >= zMin && z_min_from_bbox(meta) <= zMax;
}

cv::Vec3f json_bbox_high(const Json& meta)
{
    if (!meta.contains("bbox") || !meta["bbox"].is_array()) {
        return {0, 0, 0};
    }
    return {
        meta["bbox"][1][0].get_float(),
        meta["bbox"][1][1].get_float(),
        meta["bbox"][1][2].get_float()
    };
}

std::string now_id()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y%m%d%H%M%S");
    oss << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::vector<SurfacePatchIndex::SurfacePtr> discover_patch_surfaces(
    const Config& cfg,
    float zMin,
    float zMax,
    const fs::path& sourceAbs,
    cv::Vec3f& maxCoord)
{
    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    std::size_t scanned = 0;
    std::size_t skippedZ = 0;
    std::size_t skippedSource = 0;
    std::size_t skippedBad = 0;

    for (const auto& entry : fs::directory_iterator(cfg.patchDir)) {
        if (!entry.is_directory()) {
            continue;
        }
        ++scanned;

        std::error_code ec;
        fs::path absPath = fs::absolute(entry.path(), ec).lexically_normal();
        if (!ec && absPath == sourceAbs) {
            ++skippedSource;
            continue;
        }

        const fs::path metaPath = entry.path() / "meta.json";
        if (!fs::is_regular_file(metaPath)) {
            continue;
        }

        try {
            Json meta = Json::parse_file(metaPath);
            if (meta.value("format", std::string{}) != "tifxyz") {
                ++skippedBad;
                continue;
            }
            if (!bbox_intersects_z(meta, zMin, zMax)) {
                ++skippedZ;
                continue;
            }

            const cv::Vec3f high = json_bbox_high(meta);
            for (int i = 0; i < 3; ++i) {
                maxCoord[i] = std::max(maxCoord[i], high[i]);
            }
            surfaces.push_back(std::make_shared<QuadSurface>(entry.path(), meta));
        } catch (const std::exception& ex) {
            ++skippedBad;
            std::cerr << "WARNING: skipping " << entry.path() << ": " << ex.what() << '\n';
        }
    }

    std::cout << "Patch discovery: scanned " << scanned
              << ", selected " << surfaces.size()
              << ", skipped z " << skippedZ
              << ", skipped source " << skippedSource
              << ", skipped invalid " << skippedBad << '\n';
    return surfaces;
}

std::optional<RayTriangleIntersection> ray_triangle_intersection(
    const cv::Vec3f& origin,
    const cv::Vec3f& dir,
    const std::array<cv::Vec3f, 3>& tri,
    float minT,
    float maxT)
{
    constexpr float eps = 1e-6f;
    const cv::Vec3f e1 = tri[1] - tri[0];
    const cv::Vec3f e2 = tri[2] - tri[0];
    const cv::Vec3f p = dir.cross(e2);
    const float det = e1.dot(p);
    if (std::abs(det) < eps) {
        return std::nullopt;
    }

    const float invDet = 1.0f / det;
    const cv::Vec3f tv = origin - tri[0];
    const float u = tv.dot(p) * invDet;
    if (u < -eps || u > 1.0f + eps) {
        return std::nullopt;
    }

    const cv::Vec3f q = tv.cross(e1);
    const float v = dir.dot(q) * invDet;
    if (v < -eps || u + v > 1.0f + eps) {
        return std::nullopt;
    }

    const float t = e2.dot(q) * invDet;
    if (t < minT || t > maxT || !std::isfinite(t)) {
        return std::nullopt;
    }
    return RayTriangleIntersection{t, {1.0f - u - v, u, v}};
}

std::optional<float> ray_triangle_t(const cv::Vec3f& origin,
                                    const cv::Vec3f& dir,
                                    const std::array<cv::Vec3f, 3>& tri,
                                    float minT,
                                    float maxT)
{
    auto hit = ray_triangle_intersection(origin, dir, tri, minT, maxT);
    if (!hit) {
        return std::nullopt;
    }
    return hit->t;
}

bool ray_intersects_patch_bounds(const SurfacePatchIndex::PatchBounds& bounds,
                                 const cv::Vec3f& origin,
                                 const cv::Vec3f& dir,
                                 float minT,
                                 float maxT)
{
    float t0 = minT;
    float t1 = maxT;
    for (int ax = 0; ax < 3; ++ax) {
        const float d = dir[ax];
        if (std::abs(d) <= 1e-8f) {
            if (origin[ax] < bounds.low[ax] || origin[ax] > bounds.high[ax]) {
                return false;
            }
            continue;
        }

        const float invD = 1.0f / d;
        float nearT = (bounds.low[ax] - origin[ax]) * invD;
        float farT = (bounds.high[ax] - origin[ax]) * invD;
        if (nearT > farT) {
            std::swap(nearT, farT);
        }
        t0 = std::max(t0, nearT);
        t1 = std::min(t1, farT);
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

cv::Vec3f interpolate_surface_param(const SurfacePatchIndex::TriangleCandidate& tri,
                                    const cv::Vec3f& bary)
{
    return tri.surfaceParams[0] * bary[0]
         + tri.surfaceParams[1] * bary[1]
         + tri.surfaceParams[2] * bary[2];
}

bool is_local_source_triangle(const SurfacePatchIndex::TriangleCandidate& tri,
                              int row,
                              int col,
                              int indexStride,
                              int ignoreRadius)
{
    const int stride = std::max(1, indexStride);
    const int radius = std::max(ignoreRadius, stride + 1);
    const int triRow0 = tri.j;
    const int triRow1 = tri.j + stride;
    const int triCol0 = tri.i;
    const int triCol1 = tri.i + stride;

    return row >= triRow0 - radius && row <= triRow1 + radius &&
           col >= triCol0 - radius && col <= triCol1 + radius;
}

void dedup_hits(std::vector<RayHit>& hits, float radius)
{
    if (hits.empty()) {
        return;
    }
    std::sort(hits.begin(), hits.end(), [](const RayHit& a, const RayHit& b) {
        return a.t < b.t;
    });

    if (radius <= 0.0f) {
        return;
    }

    std::vector<RayHit> deduped;
    deduped.reserve(hits.size());

    std::size_t clusterBegin = 0;
    while (clusterBegin < hits.size()) {
        std::size_t clusterEnd = clusterBegin + 1;
        float clusterSum = hits[clusterBegin].t;
        while (clusterEnd < hits.size() &&
               hits[clusterEnd].t - hits[clusterBegin].t <= radius) {
            clusterSum += hits[clusterEnd].t;
            ++clusterEnd;
        }

        const float clusterMean = clusterSum / static_cast<float>(clusterEnd - clusterBegin);
        std::size_t best = clusterBegin;
        float bestErr = std::abs(hits[clusterBegin].t - clusterMean);
        for (std::size_t idx = clusterBegin + 1; idx < clusterEnd; ++idx) {
            const float err = std::abs(hits[idx].t - clusterMean);
            if (err < bestErr) {
                best = idx;
                bestErr = err;
            }
        }
        deduped.push_back(hits[best]);
        clusterBegin = clusterEnd;
    }
    hits.swap(deduped);
}

std::vector<float> neighbor_distances(const cv::Mat_<float>& dist,
                                      const cv::Mat_<uchar>& srcValid,
                                      int row,
                                      int col,
                                      int radius)
{
    std::vector<float> values;
    values.reserve((radius * 2 + 1) * (radius * 2 + 1));
    const int r0 = std::max(0, row - radius);
    const int r1 = std::min(dist.rows - 1, row + radius);
    const int c0 = std::max(0, col - radius);
    const int c1 = std::min(dist.cols - 1, col + radius);
    for (int r = r0; r <= r1; ++r) {
        for (int c = c0; c <= c1; ++c) {
            if (r == row && c == col) {
                continue;
            }
            if (!srcValid(r, c)) {
                continue;
            }
            const float d = dist(r, c);
            if (d > 0.0f && std::isfinite(d)) {
                values.push_back(d);
            }
        }
    }
    return values;
}

bool surfaces_overlap(const SurfacePatchIndex::SurfacePtr& a,
                      const SurfacePatchIndex::SurfacePtr& b)
{
    if (!a || !b) {
        return false;
    }
    if (a == b || a.get() == b.get()) {
        return true;
    }
    if (!a->id.empty() && b->overlappingIds().count(a->id) != 0) {
        return true;
    }
    if (!b->id.empty() && a->overlappingIds().count(b->id) != 0) {
        return true;
    }
    return false;
}

bool same_surface(const SurfacePatchIndex::SurfacePtr& a,
                  const SurfacePatchIndex::SurfacePtr& b)
{
    return a && b && (a == b || a.get() == b.get());
}

std::optional<float> median(std::vector<float>& values)
{
    if (values.empty()) {
        return std::nullopt;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float med = values[mid];
    if ((values.size() % 2) == 0) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        med = 0.5f * (med + values[mid - 1]);
    }
    return med;
}

void enforce_neighbor_consistency(const std::vector<std::vector<RayHit>>& allHits,
                                  const cv::Mat_<uchar>& srcValid,
                                  float maxDelta,
                                  float maxFartherDelta,
                                  int radius,
                                  int iterations,
                                  cv::Mat_<float>& chosenDist,
                                  cv::Mat_<cv::Vec3f>& chosenPoint)
{
    if (maxDelta <= 0.0f || radius <= 0 || iterations <= 0) {
        return;
    }

    const int rows = chosenDist.rows;
    const int cols = chosenDist.cols;
    for (int iter = 0; iter < iterations; ++iter) {
        int changed = 0;
        cv::Mat_<float> nextDist = chosenDist.clone();
        cv::Mat_<cv::Vec3f> nextPoint = chosenPoint.clone();

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!srcValid(r, c)) {
                    continue;
                }
                auto values = neighbor_distances(chosenDist, srcValid, r, c, radius);
                auto med = median(values);
                if (!med) {
                    continue;
                }

                const int idx = r * cols + c;
                const auto& candidates = allHits[idx];
                if (candidates.empty()) {
                    continue;
                }

                const float curr = chosenDist(r, c);
                if (curr > 0.0f) {
                    const float delta = curr - *med;
                    const bool tooClose = delta < -maxDelta;
                    const bool tooFar = maxFartherDelta > 0.0f && delta > maxFartherDelta;
                    if (!tooClose && !tooFar) {
                        continue;
                    }
                }

                const RayHit* best = nullptr;
                float bestErr = std::numeric_limits<float>::max();
                for (const RayHit& hit : candidates) {
                    const float err = std::abs(hit.t - *med);
                    if (err < bestErr) {
                        bestErr = err;
                        best = &hit;
                    }
                }

                if (best) {
                    const float bestDelta = best->t - *med;
                    const bool bestAccepted = bestDelta >= 0.0f
                        ? (maxFartherDelta <= 0.0f || bestDelta <= maxFartherDelta)
                        : (-bestDelta <= maxDelta);
                    if (!bestAccepted) {
                        best = nullptr;
                    }
                }

                if (best) {
                    nextDist(r, c) = best->t;
                    nextPoint(r, c) = best->point;
                    ++changed;
                } else if (curr > 0.0f) {
                    nextDist(r, c) = -1.0f;
                    nextPoint(r, c) = kInvalid;
                    ++changed;
                }
            }
        }

        chosenDist = std::move(nextDist);
        chosenPoint = std::move(nextPoint);
        std::cout << "Neighbor consistency pass " << (iter + 1)
                  << ": changed " << changed << " vertices\n";
        if (changed == 0) {
            break;
        }
    }
}

std::optional<cv::Vec3f> neighbor_point_average(const cv::Mat_<cv::Vec3f>& points,
                                                const cv::Mat_<uchar>& srcValid,
                                                int row,
                                                int col,
                                                int radius,
                                                int minNeighbors)
{
    cv::Vec3f sum(0.0f, 0.0f, 0.0f);
    int count = 0;
    const int r0 = std::max(0, row - radius);
    const int r1 = std::min(points.rows - 1, row + radius);
    const int c0 = std::max(0, col - radius);
    const int c1 = std::min(points.cols - 1, col + radius);
    for (int r = r0; r <= r1; ++r) {
        for (int c = c0; c <= c1; ++c) {
            if (r == row && c == col) {
                continue;
            }
            if (!srcValid(r, c) || !valid_point(points(r, c))) {
                continue;
            }
            sum += points(r, c);
            ++count;
        }
    }
    if (count < minNeighbors) {
        return std::nullopt;
    }
    return sum * (1.0f / static_cast<float>(count));
}

float local_edge_score(const cv::Mat_<cv::Vec3f>& points,
                       const cv::Mat_<uchar>& srcValid,
                       const cv::Vec3f& candidate,
                       int row,
                       int col,
                       float targetSpacing)
{
    static constexpr int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    float score = 0.0f;
    int count = 0;
    for (const auto& off : offsets) {
        const int nr = row + off[0];
        const int nc = col + off[1];
        if (nr < 0 || nr >= points.rows || nc < 0 || nc >= points.cols) {
            continue;
        }
        if (!srcValid(nr, nc) || !valid_point(points(nr, nc))) {
            continue;
        }
        score += std::abs(cv::norm(candidate - points(nr, nc)) - targetSpacing);
        ++count;
    }
    return count > 0 ? score / static_cast<float>(count) : 0.0f;
}

float local_patch_score(const std::vector<RayHit>& chosenHits,
                        const cv::Mat_<uchar>& chosenHitValid,
                        const SurfacePatchIndex::SurfacePtr& candidateSurface,
                        int row,
                        int col,
                        int radius,
                        float compatibleBonus,
                        float incompatiblePenalty)
{
    if (!candidateSurface) {
        return 0.0f;
    }
    float score = 0.0f;
    const int rows = chosenHitValid.rows;
    const int cols = chosenHitValid.cols;
    const int r0 = std::max(0, row - radius);
    const int r1 = std::min(rows - 1, row + radius);
    const int c0 = std::max(0, col - radius);
    const int c1 = std::min(cols - 1, col + radius);
    for (int r = r0; r <= r1; ++r) {
        for (int c = c0; c <= c1; ++c) {
            if (r == row && c == col) {
                continue;
            }
            if (!chosenHitValid(r, c)) {
                continue;
            }
            const RayHit& neighbor = chosenHits[static_cast<std::size_t>(r) * cols + c];
            if (!neighbor.surface) {
                continue;
            }
            score += surfaces_overlap(candidateSurface, neighbor.surface)
                ? -compatibleBonus
                : incompatiblePenalty;
        }
    }
    return score;
}

cv::Vec3f oriented_grid_normal(const cv::Mat_<cv::Vec3f>& points,
                               int row,
                               int col,
                               const Umbilicus& umb,
                               bool castOut);

void repair_shifted_ray_outliers(const Config& cfg,
                                 const SurfacePatchIndex& index,
                                 const Umbilicus& umb,
                                 bool castOut,
                                 const cv::Mat_<uchar>& srcValid,
                                 const cv::Mat_<cv::Vec3f>& rayDirs,
                                 cv::Mat_<float>& chosenDist,
                                 cv::Mat_<cv::Vec3f>& chosenPoints,
                                 std::vector<RayHit>& chosenHits,
                                 cv::Mat_<uchar>& chosenHitValid,
                                 std::size_t& tested,
                                 std::size_t& repaired)
{
    tested = 0;
    repaired = 0;
    if (cfg.repairIters <= 0 || cfg.repairRadius <= 0) {
        return;
    }

    for (int iter = 0; iter < cfg.repairIters; ++iter) {
        cv::Mat_<float> nextDist = chosenDist.clone();
        cv::Mat_<cv::Vec3f> nextPoints = chosenPoints.clone();
        std::vector<RayHit> nextHits = chosenHits;
        cv::Mat_<uchar> nextHitValid = chosenHitValid.clone();
        std::size_t iterTested = 0;
        std::size_t iterRepaired = 0;

        for (int r = 0; r < chosenPoints.rows; ++r) {
            for (int c = 0; c < chosenPoints.cols; ++c) {
                if (!srcValid(r, c) || !valid_point(chosenPoints(r, c))) {
                    continue;
                }

                auto distances = neighbor_distances(chosenDist, srcValid, r, c, cfg.repairRadius);
                auto medDist = median(distances);
                auto predicted = neighbor_point_average(chosenPoints, srcValid, r, c,
                                                        cfg.repairRadius,
                                                        cfg.repairMinNeighbors);
                if (!medDist || !predicted) {
                    continue;
                }

                const float tooClose = *medDist - chosenDist(r, c);
                const float expectedMaxEdge = std::max(1.0f, cfg.spacing * cfg.repairEdgeFactor);
                const float predictedGap = cv::norm(chosenPoints(r, c) - *predicted);
                if (tooClose <= cfg.repairCloserDelta && predictedGap <= expectedMaxEdge) {
                    continue;
                }

                cv::Vec3f dir = oriented_grid_normal(chosenPoints, r, c, umb, castOut);
                if (!finite_vec(dir)) {
                    dir = rayDirs(r, c);
                }
                const float dirLen = cv::norm(dir);
                if (!std::isfinite(dirLen) || dirLen <= 1e-6f) {
                    continue;
                }
                dir *= 1.0f / dirLen;

                ++iterTested;
                const cv::Vec3f origin = *predicted - dir * cfg.repairBacktrack;
                const float maxT = cfg.repairBacktrack + cfg.repairForward;
                const cv::Vec3f end = origin + dir * maxT;
                Rect3D bounds;
                for (int ax = 0; ax < 3; ++ax) {
                    bounds.low[ax] = std::min(origin[ax], end[ax]) - cfg.bboxPadding;
                    bounds.high[ax] = std::max(origin[ax], end[ax]) + cfg.bboxPadding;
                }

                std::vector<RayHit> hits;
                SurfacePatchIndex::TriangleQuery query;
                query.bounds = bounds;
                query.patchFilter = [&](const SurfacePatchIndex::PatchBounds& patchBounds) {
                    return ray_intersects_patch_bounds(patchBounds, origin, dir, 0.0f, maxT);
                };
                index.forEachTriangle(
                    query,
                    [&](const SurfacePatchIndex::TriangleCandidate& tri) {
                        auto intersection = ray_triangle_intersection(origin, dir, tri.world, 0.0f, maxT);
                        if (!intersection) {
                            return;
                        }
                        hits.push_back(RayHit{intersection->t,
                                               origin + dir * intersection->t,
                                               interpolate_surface_param(tri, intersection->bary),
                                               tri.surface, tri.i, tri.j, tri.triangleIndex});
                    });
                dedup_hits(hits, cfg.dedupRadius);
                if (hits.empty()) {
                    continue;
                }

                const int idx = r * chosenPoints.cols + c;
                const cv::Vec3f current = chosenPoints(r, c);
                const RayHit& currentHit = chosenHits[static_cast<std::size_t>(idx)];
                float bestCurrentScore = cv::norm(current - *predicted)
                    + 0.5f * local_edge_score(chosenPoints, srcValid, current, r, c, cfg.spacing);
                if (chosenHitValid(r, c)) {
                    bestCurrentScore += local_patch_score(chosenHits, chosenHitValid,
                                                          currentHit.surface, r, c,
                                                          cfg.repairRadius,
                                                          cfg.repairPatchBonus,
                                                          cfg.repairPatchPenalty);
                }

                const RayHit* best = nullptr;
                float bestScore = std::numeric_limits<float>::max();
                for (const RayHit& hit : hits) {
                    const float move = cv::norm(hit.point - current);
                    if (move < cfg.repairMinMove) {
                        continue;
                    }
                    float score = cv::norm(hit.point - *predicted);
                    score += 0.5f * local_edge_score(chosenPoints, srcValid, hit.point, r, c, cfg.spacing);
                    score += local_patch_score(chosenHits, chosenHitValid,
                                               hit.surface, r, c,
                                               cfg.repairRadius,
                                               cfg.repairPatchBonus,
                                               cfg.repairPatchPenalty);
                    if (score < bestScore) {
                        bestScore = score;
                        best = &hit;
                    }
                }

                if (best && bestScore + 1e-3f < bestCurrentScore) {
                    nextDist(r, c) = *medDist;
                    nextPoints(r, c) = best->point;
                    nextHits[static_cast<std::size_t>(idx)] = *best;
                    nextHitValid(r, c) = 1;
                    ++iterRepaired;
                }
            }
        }

        chosenDist = std::move(nextDist);
        chosenPoints = std::move(nextPoints);
        chosenHits = std::move(nextHits);
        chosenHitValid = std::move(nextHitValid);
        tested += iterTested;
        repaired += iterRepaired;
        std::cout << "Shifted-ray repair pass " << (iter + 1)
                  << ": tested " << iterTested
                  << ", repaired " << iterRepaired << " vertices\n";
        if (iterRepaired == 0) {
            break;
        }
    }
}

float median_grid_spacing(const cv::Mat_<cv::Vec3f>& points)
{
    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(points.rows + points.cols) * 2);
    const int rowStep = std::max(1, points.rows / 512);
    const int colStep = std::max(1, points.cols / 512);
    for (int r = 0; r < points.rows; r += rowStep) {
        for (int c = 0; c < points.cols; c += colStep) {
            const cv::Vec3f& p = points(r, c);
            if (!valid_point(p)) {
                continue;
            }
            if (c + 1 < points.cols && valid_point(points(r, c + 1))) {
                samples.push_back(cv::norm(points(r, c + 1) - p));
            }
            if (r + 1 < points.rows && valid_point(points(r + 1, c))) {
                samples.push_back(cv::norm(points(r + 1, c) - p));
            }
        }
    }
    auto med = median(samples);
    return med.value_or(0.0f);
}

cv::Mat_<cv::Vec3f> resample_points(const cv::Mat_<cv::Vec3f>& src, float factor)
{
    if (std::abs(factor - 1.0f) < 0.01f || src.empty()) {
        return src;
    }

    const int newRows = std::max(1, static_cast<int>(std::round(src.rows * factor)));
    const int newCols = std::max(1, static_cast<int>(std::round(src.cols * factor)));
    cv::Mat_<cv::Vec3f> dst(newRows, newCols);
    dst.setTo(kInvalid);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < newRows; ++r) {
        for (int c = 0; c < newCols; ++c) {
            float sr = (static_cast<float>(r) + 0.5f) / factor - 0.5f;
            float sc = (static_cast<float>(c) + 0.5f) / factor - 0.5f;
            sr = std::clamp(sr, 0.0f, static_cast<float>(src.rows - 1));
            sc = std::clamp(sc, 0.0f, static_cast<float>(src.cols - 1));

            const int r0 = static_cast<int>(std::floor(sr));
            const int c0 = static_cast<int>(std::floor(sc));
            const int r1 = std::min(r0 + 1, src.rows - 1);
            const int c1 = std::min(c0 + 1, src.cols - 1);
            if (!valid_point(src(r0, c0)) || !valid_point(src(r0, c1)) ||
                !valid_point(src(r1, c0)) || !valid_point(src(r1, c1))) {
                continue;
            }

            const float tr = sr - static_cast<float>(r0);
            const float tc = sc - static_cast<float>(c0);
            const cv::Vec3f top = src(r0, c0) * (1.0f - tc) + src(r0, c1) * tc;
            const cv::Vec3f bot = src(r1, c0) * (1.0f - tc) + src(r1, c1) * tc;
            dst(r, c) = top * (1.0f - tr) + bot * tr;
        }
    }
    return dst;
}

cv::Vec3f oriented_grid_normal(const cv::Mat_<cv::Vec3f>& points,
                               int row,
                               int col,
                               const Umbilicus& umb,
                               bool castOut)
{
    const cv::Vec3f& p = points(row, col);
    if (!valid_point(p)) {
        return {NAN, NAN, NAN};
    }

    cv::Vec3f n = grid_normal(points, cv::Vec3f(static_cast<float>(col),
                                                static_cast<float>(row),
                                                0.0f));
    const float nLen = cv::norm(n);
    if (!std::isfinite(nLen) || nLen <= 1e-6f) {
        return {NAN, NAN, NAN};
    }
    n *= 1.0f / nLen;

    cv::Vec3f away = p - umb.center_at(static_cast<int>(std::lround(p[2])));
    const float awayLen = cv::norm(away);
    if (std::isfinite(awayLen) && awayLen > 1e-6f) {
        away *= 1.0f / awayLen;
        if (n.dot(away) < 0.0f) {
            n *= -1.0f;
        }
    }
    if (!castOut) {
        n *= -1.0f;
    }
    return n;
}

cv::Mat_<cv::Vec3f> interpolate_snap_misses(const cv::Mat_<cv::Vec3f>& original,
                                            const cv::Mat_<cv::Vec3f>& snapped,
                                            int radius,
                                            int iterations,
                                            int minNeighbors,
                                            std::size_t& interpolated)
{
    interpolated = 0;
    if (radius <= 0 || iterations <= 0 || minNeighbors <= 0) {
        return snapped;
    }

    cv::Mat_<cv::Vec3f> current = snapped.clone();
    for (int iter = 0; iter < iterations; ++iter) {
        cv::Mat_<cv::Vec3f> next = current.clone();
        std::size_t changed = 0;

        for (int r = 0; r < current.rows; ++r) {
            for (int c = 0; c < current.cols; ++c) {
                if (!valid_point(original(r, c)) || valid_point(current(r, c))) {
                    continue;
                }

                cv::Vec3f sum(0.0f, 0.0f, 0.0f);
                int count = 0;
                const int r0 = std::max(0, r - radius);
                const int r1 = std::min(current.rows - 1, r + radius);
                const int c0 = std::max(0, c - radius);
                const int c1 = std::min(current.cols - 1, c + radius);
                for (int nr = r0; nr <= r1; ++nr) {
                    for (int nc = c0; nc <= c1; ++nc) {
                        if (nr == r && nc == c) {
                            continue;
                        }
                        if (!valid_point(current(nr, nc))) {
                            continue;
                        }
                        sum += current(nr, nc);
                        ++count;
                    }
                }

                if (count >= minNeighbors) {
                    next(r, c) = sum * (1.0f / static_cast<float>(count));
                    ++changed;
                }
            }
        }

        current = std::move(next);
        interpolated += changed;
        std::cout << "Snap interpolation pass " << (iter + 1)
                  << ": filled " << changed << " vertices\n";
        if (changed == 0) {
            break;
        }
    }

    return current;
}

cv::Mat_<cv::Vec3f> snap_points_to_patch_rays(const cv::Mat_<cv::Vec3f>& points,
                                              const Config& cfg,
                                              const SurfacePatchIndex& index,
                                              const Umbilicus& umb,
                                              bool castOut,
                                              std::size_t& snappedCount,
                                              std::size_t& interpolatedCount)
{
    snappedCount = 0;
    interpolatedCount = 0;

    cv::Mat_<cv::Vec3f> snapped(points.rows, points.cols);
    snapped.setTo(kInvalid);
    const float maxDistance = std::max(cfg.snapMinDistance, cfg.snapMaxDistance);

    #pragma omp parallel for schedule(dynamic, 32) reduction(+:snappedCount)
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f origin = points(r, c);
            if (!valid_point(origin)) {
                continue;
            }

            const cv::Vec3f dir = oriented_grid_normal(points, r, c, umb, castOut);
            if (!finite_vec(dir)) {
                continue;
            }

            const cv::Vec3f end = origin + dir * maxDistance;
            Rect3D bounds;
            for (int ax = 0; ax < 3; ++ax) {
                bounds.low[ax] = std::min(origin[ax], end[ax]) - cfg.bboxPadding;
                bounds.high[ax] = std::max(origin[ax], end[ax]) + cfg.bboxPadding;
            }

            float bestT = std::numeric_limits<float>::max();
            cv::Vec3f bestPoint = kInvalid;
            SurfacePatchIndex::TriangleQuery query;
            query.bounds = bounds;
            query.patchFilter = [&](const SurfacePatchIndex::PatchBounds& patchBounds) {
                return ray_intersects_patch_bounds(patchBounds, origin, dir,
                                                   cfg.snapMinDistance, maxDistance);
            };
            index.forEachTriangle(
                query,
                [&](const SurfacePatchIndex::TriangleCandidate& tri) {
                    auto t = ray_triangle_t(origin, dir, tri.world, cfg.snapMinDistance, maxDistance);
                    if (t && *t < bestT) {
                        bestT = *t;
                        bestPoint = origin + dir * (*t);
                    }
                });

            if (valid_point(bestPoint)) {
                snapped(r, c) = bestPoint;
                ++snappedCount;
            }
        }
    }

    return interpolate_snap_misses(points, snapped, cfg.snapInterpolateRadius,
                                   cfg.snapInterpolateIters,
                                   cfg.snapInterpolateMinNeighbors,
                                   interpolatedCount);
}

std::size_t count_valid(const cv::Mat_<cv::Vec3f>& points)
{
    std::size_t count = 0;
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            if (valid_point(points(r, c))) {
                ++count;
            }
        }
    }
    return count;
}

Config parse_args(int argc, char** argv)
{
    Config cfg;
    po::options_description desc("Project a tifxyz along umbilicus-oriented normals onto a folder of tifxyz patches");
    desc.add_options()
        ("help,h", "show help")
        ("source,s", po::value<std::string>()->required(), "source tifxyz directory")
        ("umbilicus,u", po::value<std::string>()->required(), "umbilicus json/text path")
        ("patch-dir,p", po::value<std::string>()->required(), "folder containing tifxyz patch subdirectories")
        ("output,o", po::value<std::string>()->required(), "output tifxyz directory")
        ("direction,d", po::value<std::string>(&cfg.direction)->default_value(cfg.direction), "out or in")
        ("z-min", po::value<float>(), "minimum z slice to index")
        ("z-max", po::value<float>(), "maximum z slice to index")
        ("max-distance", po::value<float>(&cfg.maxDistance)->default_value(cfg.maxDistance), "maximum ray distance in voxels")
        ("min-distance", po::value<float>(&cfg.minDistance)->default_value(cfg.minDistance), "minimum accepted ray distance")
        ("dedup-radius", po::value<float>(&cfg.dedupRadius)->default_value(cfg.dedupRadius), "merge hits on the same ray within this many voxels")
        ("same-wrap-tolerance", po::value<float>(&cfg.sameWrapTolerance)->default_value(cfg.sameWrapTolerance), "drop candidate patch hits this close to the source surface; <=0 disables")
        ("spacing", po::value<float>(&cfg.spacing)->default_value(cfg.spacing), "target output vertex spacing in voxels")
        ("neighbor-max-delta", po::value<float>(&cfg.neighborMaxDelta)->default_value(cfg.neighborMaxDelta), "maximum allowed closer-than-median distance jump; <=0 disables neighbor consistency")
        ("neighbor-max-farther-delta", po::value<float>(&cfg.neighborMaxFartherDelta)->default_value(cfg.neighborMaxFartherDelta), "maximum allowed farther-than-median distance jump; <=0 allows farther hits")
        ("ray-neighborhood-radius", po::value<int>(&cfg.rayNeighborhoodRadius)->default_value(cfg.rayNeighborhoodRadius), "source grid radius around each vertex to cast and average")
        ("ray-neighborhood-min-samples", po::value<int>(&cfg.rayNeighborhoodMinSamples)->default_value(cfg.rayNeighborhoodMinSamples), "minimum accepted neighborhood ray hits needed to emit a vertex")
        ("ray-neighborhood-max-delta", po::value<float>(&cfg.rayNeighborhoodMaxDelta)->default_value(cfg.rayNeighborhoodMaxDelta), "drop neighborhood ray hits this far from the neighborhood median distance; <=0 disables")
        ("ray-neighborhood-average-direct-hits", po::bool_switch(&cfg.rayNeighborhoodAverageDirectHits)->default_value(cfg.rayNeighborhoodAverageDirectHits), "average neighborhood hits even when the center vertex has a direct ray hit")
        ("smooth-radius", po::value<int>(&cfg.smoothRadius)->default_value(cfg.smoothRadius), "local radius for distance consistency")
        ("smooth-iters", po::value<int>(&cfg.smoothIters)->default_value(cfg.smoothIters), "distance consistency iterations")
        ("index-stride", po::value<int>(&cfg.indexStride)->default_value(cfg.indexStride), "SurfacePatchIndex sampling stride")
        ("self-hit-ignore-radius", po::value<int>(&cfg.selfHitIgnoreRadius)->default_value(cfg.selfHitIgnoreRadius), "source-grid radius to ignore around the emitting vertex for self-hit rejection")
        ("self-hit-epsilon", po::value<float>(&cfg.selfHitEpsilon)->default_value(cfg.selfHitEpsilon), "discard if source self-hit is this many voxels before the first patch hit")
        ("bbox-padding", po::value<float>(&cfg.bboxPadding)->default_value(cfg.bboxPadding), "index/query bbox padding")
        ("snap-to-patch", po::bool_switch(&cfg.snapToPatch)->default_value(cfg.snapToPatch), "after resampling, snap vertices to triangle intersections along oriented projection normals")
        ("snap-min-distance", po::value<float>(&cfg.snapMinDistance)->default_value(cfg.snapMinDistance), "minimum ray distance for post-resample patch snapping")
        ("snap-max-distance", po::value<float>(&cfg.snapMaxDistance)->default_value(cfg.snapMaxDistance), "maximum ray distance for post-resample patch snapping")
        ("snap-interpolate-radius", po::value<int>(&cfg.snapInterpolateRadius)->default_value(cfg.snapInterpolateRadius), "neighbor radius for filling post-snap misses")
        ("snap-interpolate-iters", po::value<int>(&cfg.snapInterpolateIters)->default_value(cfg.snapInterpolateIters), "iterations for filling post-snap misses")
        ("snap-interpolate-min-neighbors", po::value<int>(&cfg.snapInterpolateMinNeighbors)->default_value(cfg.snapInterpolateMinNeighbors), "minimum snapped neighbors needed to fill a post-snap miss")
        ("repair-iters", po::value<int>(&cfg.repairIters)->default_value(cfg.repairIters), "shifted-ray wrong-patch repair iterations before resampling; 0 disables")
        ("repair-radius", po::value<int>(&cfg.repairRadius)->default_value(cfg.repairRadius), "neighbor radius for shifted-ray wrong-patch repair")
        ("repair-min-neighbors", po::value<int>(&cfg.repairMinNeighbors)->default_value(cfg.repairMinNeighbors), "minimum valid neighbors required to predict a shifted repair ray")
        ("repair-closer-delta", po::value<float>(&cfg.repairCloserDelta)->default_value(cfg.repairCloserDelta), "repair vertices whose projection distance is this much closer than the local median")
        ("repair-edge-factor", po::value<float>(&cfg.repairEdgeFactor)->default_value(cfg.repairEdgeFactor), "repair vertices farther than spacing*factor from the neighbor-predicted point")
        ("repair-backtrack", po::value<float>(&cfg.repairBacktrack)->default_value(cfg.repairBacktrack), "voxels to move behind the neighbor-predicted point before recasting")
        ("repair-forward", po::value<float>(&cfg.repairForward)->default_value(cfg.repairForward), "voxels to cast forward from the neighbor-predicted point during repair")
        ("repair-min-move", po::value<float>(&cfg.repairMinMove)->default_value(cfg.repairMinMove), "minimum replacement movement required for shifted-ray repair")
        ("repair-patch-bonus", po::value<float>(&cfg.repairPatchBonus)->default_value(cfg.repairPatchBonus), "score bonus for repair candidates on neighbor-compatible/overlapping patches")
        ("repair-patch-penalty", po::value<float>(&cfg.repairPatchPenalty)->default_value(cfg.repairPatchPenalty), "score penalty for repair candidates on unrelated patches")
        ("index-cache-dir", po::value<std::string>(), "directory for reusable SurfacePatchIndex cache files; defaults to the output parent directory")
        ("no-index-cache", po::bool_switch(&cfg.noIndexCache)->default_value(cfg.noIndexCache), "disable reusable patch-index cache")
        ("threads,j", po::value<int>(&cfg.threads)->default_value(cfg.threads), "OpenMP threads, 0 keeps current default");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << '\n';
        std::exit(EXIT_SUCCESS);
    }
    po::notify(vm);

    cfg.source = vm["source"].as<std::string>();
    cfg.umbilicus = vm["umbilicus"].as<std::string>();
    cfg.patchDir = vm["patch-dir"].as<std::string>();
    cfg.output = vm["output"].as<std::string>();
    if (vm.count("index-cache-dir")) {
        cfg.indexCacheDir = vm["index-cache-dir"].as<std::string>();
    } else {
        cfg.indexCacheDir = cfg.output.parent_path();
        if (cfg.indexCacheDir.empty()) {
            cfg.indexCacheDir = ".";
        }
    }
    if (vm.count("z-min")) {
        cfg.zMin = vm["z-min"].as<float>();
    }
    if (vm.count("z-max")) {
        cfg.zMax = vm["z-max"].as<float>();
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv)
{
    Config cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    if (cfg.direction != "out" && cfg.direction != "in") {
        std::cerr << "ERROR: --direction must be 'out' or 'in'\n";
        return EXIT_FAILURE;
    }
    if (cfg.maxDistance <= 0.0f || cfg.spacing <= 0.0f || cfg.indexStride <= 0) {
        std::cerr << "ERROR: --max-distance, --spacing, and --index-stride must be positive\n";
        return EXIT_FAILURE;
    }
    cfg.rayNeighborhoodRadius = std::max(0, cfg.rayNeighborhoodRadius);
    cfg.rayNeighborhoodMinSamples = std::max(1, cfg.rayNeighborhoodMinSamples);
    cfg.snapMinDistance = std::max(0.0f, cfg.snapMinDistance);
    cfg.snapMaxDistance = std::max(cfg.snapMinDistance, cfg.snapMaxDistance);
    cfg.snapInterpolateRadius = std::max(0, cfg.snapInterpolateRadius);
    cfg.snapInterpolateIters = std::max(0, cfg.snapInterpolateIters);
    cfg.snapInterpolateMinNeighbors = std::max(1, cfg.snapInterpolateMinNeighbors);
    cfg.repairIters = std::max(0, cfg.repairIters);
    cfg.repairRadius = std::max(0, cfg.repairRadius);
    cfg.repairMinNeighbors = std::max(1, cfg.repairMinNeighbors);
    cfg.repairCloserDelta = std::max(0.0f, cfg.repairCloserDelta);
    cfg.repairEdgeFactor = std::max(0.1f, cfg.repairEdgeFactor);
    cfg.repairBacktrack = std::max(0.0f, cfg.repairBacktrack);
    cfg.repairForward = std::max(0.0f, cfg.repairForward);
    cfg.repairMinMove = std::max(0.0f, cfg.repairMinMove);
    cfg.repairPatchBonus = std::max(0.0f, cfg.repairPatchBonus);
    cfg.repairPatchPenalty = std::max(0.0f, cfg.repairPatchPenalty);
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }

    const fs::path initialSource = cfg.source;

    std::cout << "Loading initial source for patch index bounds: " << initialSource << '\n';
    std::shared_ptr<QuadSurface> initialSrc;
    try {
        initialSrc = std::shared_ptr<QuadSurface>(load_quad_from_tifxyz(initialSource.string()).release());
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: failed to load initial source tifxyz: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    if (!initialSrc->meta.contains("bbox") || !initialSrc->meta["bbox"].is_array()) {
        std::cerr << "ERROR: source tifxyz meta.json must contain bbox for z-range patch filtering\n";
        return EXIT_FAILURE;
    }

    cv::Vec3f maxCoord = json_bbox_high(initialSrc->meta);
    const float initialSourceZMin = z_min_from_bbox(initialSrc->meta);
    const float initialSourceZMax = z_max_from_bbox(initialSrc->meta);
    float indexZMin = initialSourceZMin;
    float indexZMax = initialSourceZMax;
    if (cfg.zMin) {
        indexZMin = std::max(indexZMin, *cfg.zMin);
    }
    if (cfg.zMax) {
        indexZMax = std::min(indexZMax, *cfg.zMax);
    }
    if (indexZMin > indexZMax) {
        std::cerr << "ERROR: requested z range does not intersect source z range ["
                  << initialSourceZMin << ", " << initialSourceZMax << "]\n";
        return EXIT_FAILURE;
    }
    std::cout << "Initial source z range: [" << initialSourceZMin << ", " << initialSourceZMax
              << "], index z range: [" << indexZMin << ", " << indexZMax << "]\n";

    std::error_code ec;
    const fs::path initialSourceAbs = fs::absolute(initialSource, ec).lexically_normal();
    std::cout << "Discovering patch tifxyz folders: " << cfg.patchDir << '\n';
    auto patchSurfaces = discover_patch_surfaces(cfg, indexZMin, indexZMax, initialSourceAbs, maxCoord);
    if (patchSurfaces.empty()) {
        std::cerr << "ERROR: no tifxyz patches selected for indexing\n";
        return EXIT_FAILURE;
    }

    const cv::Vec3i volumeShape{
        std::max(1, static_cast<int>(std::ceil(maxCoord[2])) + 2),
        std::max(1, static_cast<int>(std::ceil(maxCoord[1])) + 2),
        std::max(1, static_cast<int>(std::ceil(maxCoord[0])) + 2)
    };
    Umbilicus umb = Umbilicus::FromFile(cfg.umbilicus, volumeShape);
    std::cout << "Umbilicus loaded with inferred volume shape [z,y,x]=["
              << volumeShape[0] << "," << volumeShape[1] << "," << volumeShape[2] << "]\n";

    std::cout << "Building surface patch index with " << patchSurfaces.size()
              << " surfaces, stride " << cfg.indexStride << "...\n";
    SurfacePatchIndex index;
    index.setReadOnly(true);
    index.setSamplingStride(cfg.indexStride);
    bool cacheHit = false;
    fs::path cachePath;
    std::string cacheKey;
    if (!cfg.noIndexCache) {
        cacheKey = SurfacePatchIndex::cacheKeyForSurfaces(
            patchSurfaces, cfg.indexStride, cfg.bboxPadding);
        cachePath = cfg.indexCacheDir / ("surface_patch_index_" + cacheKey + ".bin");
        const auto cacheLoadStart = std::chrono::steady_clock::now();
        cacheHit = index.loadCache(cachePath, patchSurfaces, cacheKey);
        const double cacheLoadElapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - cacheLoadStart).count();
        std::cout << "SurfacePatchIndex cache "
                  << (cacheHit ? "hit" : "miss")
                  << " key=" << cacheKey
                  << " time=" << cacheLoadElapsed << "s"
                  << " path=" << cachePath << '\n';
    }
    if (!cacheHit) {
        const auto rebuildStart = std::chrono::steady_clock::now();
        index.rebuild(patchSurfaces, cfg.bboxPadding);
        const double rebuildElapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - rebuildStart).count();
        std::cout << "SurfacePatchIndex rebuild time=" << rebuildElapsed << "s\n";
        if (!cfg.noIndexCache && !cachePath.empty()) {
            const auto cacheSaveStart = std::chrono::steady_clock::now();
            const bool saved = index.saveCache(cachePath, cacheKey);
            const double cacheSaveElapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - cacheSaveStart).count();
            std::cout << "SurfacePatchIndex cache save "
                      << (saved ? "ok" : "failed")
                      << " time=" << cacheSaveElapsed << "s"
                      << " path=" << cachePath << '\n';
        }
    }
    std::cout << "Index ready: " << index.patchCount() << " patches across "
              << index.surfaceCount() << " surfaces\n";

    {
        std::cout << "Loading source: " << cfg.source << '\n';
        std::shared_ptr<QuadSurface> src;
        try {
            src = std::shared_ptr<QuadSurface>(load_quad_from_tifxyz(cfg.source.string()).release());
        } catch (const std::exception& ex) {
            std::cerr << "ERROR: failed to load source tifxyz: " << ex.what() << '\n';
            return EXIT_FAILURE;
        }

        const cv::Mat_<cv::Vec3f> srcPoints = src->rawPoints();
        const int rows = srcPoints.rows;
        const int cols = srcPoints.cols;
        const std::size_t srcValidCount = count_valid(srcPoints);
        std::cout << "Source grid: " << cols << "x" << rows
                  << ", valid vertices: " << srcValidCount << '\n';

        if (!src->meta.contains("bbox") || !src->meta["bbox"].is_array()) {
            std::cerr << "ERROR: source tifxyz meta.json must contain bbox\n";
            return EXIT_FAILURE;
        }
        const float sourceZMin = z_min_from_bbox(src->meta);
        const float sourceZMax = z_max_from_bbox(src->meta);
        std::cout << "Source z range: [" << sourceZMin << ", " << sourceZMax
                  << "], reused index z range: [" << indexZMin << ", " << indexZMax << "]\n";

        std::cout << "Building source self-hit index...\n";
        SurfacePatchIndex sourceIndex;
        sourceIndex.setReadOnly(true);
        sourceIndex.setSamplingStride(cfg.indexStride);
        sourceIndex.rebuild(std::vector<SurfacePatchIndex::SurfacePtr>{src}, cfg.bboxPadding);
        std::cout << "Source index ready: " << sourceIndex.patchCount() << " patches\n";

        cv::Mat_<uchar> srcValid(rows, cols);
        srcValid.setTo(0);
        cv::Mat_<cv::Vec3f> rayDirs(rows, cols);
        rayDirs.setTo(kInvalid);

        const bool castOut = cfg.direction == "out";
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const cv::Vec3f p = srcPoints(r, c);
                if (!valid_point(p)) {
                    continue;
                }
                cv::Vec3f n = src->gridNormal(r, c);
                const float nLen = cv::norm(n);
                if (!std::isfinite(nLen) || nLen <= 1e-6f) {
                    continue;
                }
                n *= 1.0f / nLen;

                cv::Vec3f away = p - umb.center_at(static_cast<int>(std::lround(p[2])));
                const float awayLen = cv::norm(away);
                if (std::isfinite(awayLen) && awayLen > 1e-6f) {
                    away *= 1.0f / awayLen;
                    if (n.dot(away) < 0.0f) {
                        n *= -1.0f;
                    }
                }
                if (!castOut) {
                    n *= -1.0f;
                }
                rayDirs(r, c) = n;
                srcValid(r, c) = 1;
            }
        }

        std::cout << "Casting rays " << (castOut ? "outwards" : "inwards")
                  << " up to " << cfg.maxDistance << " voxels"
                  << " using neighborhood radius " << cfg.rayNeighborhoodRadius
                  << " and min samples " << cfg.rayNeighborhoodMinSamples << "...\n";
        std::vector<std::vector<RayHit>> allHits(static_cast<std::size_t>(rows) * cols);
        std::vector<RayHit> chosenHits(static_cast<std::size_t>(rows) * cols);
        cv::Mat_<uchar> chosenHitValid(rows, cols);
        chosenHitValid.setTo(0);
        cv::Mat_<float> chosenDist(rows, cols);
        chosenDist.setTo(-1.0f);
        cv::Mat_<cv::Vec3f> chosenPoints(rows, cols);
        chosenPoints.setTo(kInvalid);
        std::atomic<std::size_t> raysDone{0};
        std::atomic<std::size_t> raysHit{0};
        std::atomic<std::size_t> rawHits{0};
        std::atomic<std::size_t> sameWrapDiscarded{0};
        std::atomic<std::size_t> selfDiscarded{0};
        std::atomic<std::size_t> sourceRaysHit{0};
        std::atomic<std::size_t> neighborPreferredHits{0};

        auto cast_one = [&](int sr, int sc) -> std::vector<RayHit> {
            if (sr < 0 || sr >= rows || sc < 0 || sc >= cols || !srcValid(sr, sc)) {
                return {};
            }
            const cv::Vec3f origin = srcPoints(sr, sc);
            const cv::Vec3f dir = rayDirs(sr, sc);
            const cv::Vec3f end = origin + dir * cfg.maxDistance;
            Rect3D bounds;
            for (int ax = 0; ax < 3; ++ax) {
                bounds.low[ax] = std::min(origin[ax], end[ax]) - cfg.bboxPadding;
                bounds.high[ax] = std::max(origin[ax], end[ax]) + cfg.bboxPadding;
            }

            std::vector<RayHit> hits;
            SurfacePatchIndex::TriangleQuery query;
            query.bounds = bounds;
            query.patchFilter = [&](const SurfacePatchIndex::PatchBounds& patchBounds) {
                return ray_intersects_patch_bounds(patchBounds, origin, dir,
                                                   cfg.minDistance, cfg.maxDistance);
            };
            index.forEachTriangle(
                query,
                [&](const SurfacePatchIndex::TriangleCandidate& tri) {
                    auto intersection = ray_triangle_intersection(origin, dir, tri.world,
                                                                  cfg.minDistance, cfg.maxDistance);
                    if (!intersection) {
                        return;
                    }
                    hits.push_back(RayHit{intersection->t,
                                           origin + dir * intersection->t,
                                           interpolate_surface_param(tri, intersection->bary),
                                           tri.surface, tri.i, tri.j, tri.triangleIndex});
                });
            rawHits.fetch_add(hits.size(), std::memory_order_relaxed);
            dedup_hits(hits, cfg.dedupRadius);

            if (!hits.empty() && cfg.sameWrapTolerance > 0.0f) {
                const std::size_t before = hits.size();
                hits.erase(std::remove_if(hits.begin(), hits.end(), [&](const RayHit& hit) {
                    return sourceIndex.locate(hit.point, cfg.sameWrapTolerance, src).has_value();
                }), hits.end());
                sameWrapDiscarded.fetch_add(before - hits.size(), std::memory_order_relaxed);
            }

            if (!hits.empty()) {
                const float firstPatchT = hits.front().t;
                const cv::Vec3f selfEnd = origin + dir * firstPatchT;
                Rect3D selfBounds;
                for (int ax = 0; ax < 3; ++ax) {
                    selfBounds.low[ax] = std::min(origin[ax], selfEnd[ax]) - cfg.bboxPadding;
                    selfBounds.high[ax] = std::max(origin[ax], selfEnd[ax]) + cfg.bboxPadding;
                }
                bool selfHitFirst = false;
                SurfacePatchIndex::TriangleQuery selfQuery;
                selfQuery.bounds = selfBounds;
                selfQuery.targetSurface = src;
                selfQuery.patchFilter = [&](const SurfacePatchIndex::PatchBounds& patchBounds) {
                    return ray_intersects_patch_bounds(patchBounds, origin, dir,
                                                       cfg.minDistance, firstPatchT);
                };
                sourceIndex.forEachTriangle(
                    selfQuery,
                    [&](const SurfacePatchIndex::TriangleCandidate& tri) {
                        if (selfHitFirst || is_local_source_triangle(tri, sr, sc, cfg.indexStride, cfg.selfHitIgnoreRadius)) {
                            return;
                        }
                        auto t = ray_triangle_t(origin, dir, tri.world, cfg.minDistance, firstPatchT);
                        if (t && *t + cfg.selfHitEpsilon < firstPatchT) {
                            selfHitFirst = true;
                        }
                    });
                if (selfHitFirst) {
                    selfDiscarded.fetch_add(1, std::memory_order_relaxed);
                    return {};
                }
            }

            return hits;
        };

        std::vector<std::vector<RayHit>> cachedRayHits(static_cast<std::size_t>(rows) * cols);
        cv::Mat_<uchar> cachedRayHitValid(rows, cols);
        cachedRayHitValid.setTo(0);
        std::vector<RayHit> directRayHits(static_cast<std::size_t>(rows) * cols);
        cv::Mat_<uchar> directRayHitValid(rows, cols);
        directRayHitValid.setTo(0);

        #pragma omp parallel for schedule(dynamic, 32)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!srcValid(r, c)) {
                    continue;
                }
                auto hits = cast_one(r, c);
                if (!hits.empty()) {
                    cachedRayHits[static_cast<std::size_t>(r) * cols + c] = std::move(hits);
                    cachedRayHitValid(r, c) = 1;
                    sourceRaysHit.fetch_add(1, std::memory_order_relaxed);
                }

                const std::size_t done = raysDone.fetch_add(1, std::memory_order_relaxed) + 1;
                if ((done % 100000) == 0) {
                    #pragma omp critical
                    std::cout << "  source rays processed: " << done << " / " << srcValidCount
                              << ", source rays hit: " << sourceRaysHit.load(std::memory_order_relaxed) << '\n';
                }
            }
        }

        std::cout << "Source ray cache complete: " << raysDone.load() << " rays, "
                  << sourceRaysHit.load() << " source rays hit\n";

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!cachedRayHitValid(r, c)) {
                    continue;
                }
                const std::size_t idx = static_cast<std::size_t>(r) * cols + c;
                directRayHits[idx] = cachedRayHits[idx].front();
                directRayHitValid(r, c) = 1;
            }
        }

        auto choose_hit = [&](const std::vector<RayHit>& hits,
                              int row,
                              int col) -> std::optional<RayHit> {
            if (hits.empty()) {
                return std::nullopt;
            }
            if (hits.size() == 1) {
                return hits.front();
            }

            struct CandidateScore {
                bool hasNeighborSurface = false;
                int neighborCount = 0;
                float continuity = std::numeric_limits<float>::max();
                float t = std::numeric_limits<float>::max();
            };

            auto score_hit = [&](const RayHit& candidate) {
                CandidateScore score;
                score.t = candidate.t;
                if (!candidate.surface || !valid_surface_param(candidate.surfaceParam)) {
                    return score;
                }

                static constexpr int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                float gridDistanceSum = 0.0f;
                for (const auto& off : offsets) {
                    const int nr = row + off[0];
                    const int nc = col + off[1];
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols || !directRayHitValid(nr, nc)) {
                        continue;
                    }

                    const RayHit& neighbor = directRayHits[static_cast<std::size_t>(nr) * cols + nc];
                    if (!same_surface(candidate.surface, neighbor.surface) ||
                        !valid_surface_param(neighbor.surfaceParam)) {
                        continue;
                    }
                    gridDistanceSum += static_cast<float>(
                        cv::norm(candidate.surfaceParam - neighbor.surfaceParam));
                    ++score.neighborCount;
                }

                score.hasNeighborSurface = score.neighborCount > 0;
                if (score.hasNeighborSurface) {
                    score.continuity = gridDistanceSum / static_cast<float>(score.neighborCount);
                }
                return score;
            };

            std::size_t bestIdx = 0;
            CandidateScore bestScore = score_hit(hits[0]);
            for (std::size_t idx = 1; idx < hits.size(); ++idx) {
                CandidateScore score = score_hit(hits[idx]);
                const bool better =
                    (score.hasNeighborSurface != bestScore.hasNeighborSurface)
                        ? score.hasNeighborSurface
                        : (score.hasNeighborSurface
                            ? (score.continuity < bestScore.continuity - 1e-4f ||
                               (std::abs(score.continuity - bestScore.continuity) <= 1e-4f &&
                                (score.neighborCount > bestScore.neighborCount ||
                                 (score.neighborCount == bestScore.neighborCount &&
                                  score.t < bestScore.t))))
                            : score.t < bestScore.t);
                if (better) {
                    bestIdx = idx;
                    bestScore = score;
                }
            }

            if (bestIdx != 0 && bestScore.hasNeighborSurface) {
                neighborPreferredHits.fetch_add(1, std::memory_order_relaxed);
            }
            return hits[bestIdx];
        };

        std::atomic<std::size_t> outputDone{0};
        #pragma omp parallel for schedule(dynamic, 8)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!srcValid(r, c)) {
                    continue;
                }

                const std::size_t centerIdx = static_cast<std::size_t>(r) * cols + c;
                const bool hasCenterHit = cachedRayHitValid(r, c) != 0;
                std::vector<RayHit> neighborhoodHits;
                const int maxNeighborhoodSamples = (cfg.rayNeighborhoodRadius * 2 + 1) *
                                                   (cfg.rayNeighborhoodRadius * 2 + 1);
                neighborhoodHits.reserve(std::max(1, maxNeighborhoodSamples));
                std::vector<RayHit> averagedCandidates;
                averagedCandidates.reserve(1);
                if (hasCenterHit && !cfg.rayNeighborhoodAverageDirectHits) {
                    auto hit = choose_hit(cachedRayHits[centerIdx], r, c);
                    if (hit) {
                        averagedCandidates.push_back(*hit);
                    }
                } else {
                    const int nrad = cfg.rayNeighborhoodRadius;
                    for (int sr = r - nrad; sr <= r + nrad; ++sr) {
                        for (int sc = c - nrad; sc <= c + nrad; ++sc) {
                            if (sr < 0 || sr >= rows || sc < 0 || sc >= cols || !cachedRayHitValid(sr, sc)) {
                                continue;
                            }
                            auto hit = choose_hit(cachedRayHits[static_cast<std::size_t>(sr) * cols + sc],
                                                  sr, sc);
                            if (hit) {
                                neighborhoodHits.push_back(*hit);
                            }
                        }
                    }

                    if (neighborhoodHits.size() >= static_cast<std::size_t>(cfg.rayNeighborhoodMinSamples) &&
                        cfg.rayNeighborhoodMaxDelta > 0.0f) {
                        std::vector<float> hitDistances;
                        hitDistances.reserve(neighborhoodHits.size());
                        for (const RayHit& hit : neighborhoodHits) {
                            hitDistances.push_back(hit.t);
                        }
                        auto med = median(hitDistances);
                        if (med) {
                            neighborhoodHits.erase(
                                std::remove_if(neighborhoodHits.begin(), neighborhoodHits.end(), [&](const RayHit& hit) {
                                    return std::abs(hit.t - *med) > cfg.rayNeighborhoodMaxDelta;
                                }),
                                neighborhoodHits.end());
                        }
                    }

                    if (neighborhoodHits.size() >= static_cast<std::size_t>(cfg.rayNeighborhoodMinSamples)) {
                        cv::Vec3f pointSum(0.0f, 0.0f, 0.0f);
                        float distSum = 0.0f;
                        for (const RayHit& hit : neighborhoodHits) {
                            pointSum += hit.point;
                            distSum += hit.t;
                        }
                        const float invCount = 1.0f / static_cast<float>(neighborhoodHits.size());
                        averagedCandidates.push_back(RayHit{distSum * invCount, pointSum * invCount});
                    }
                }

                const int idx = r * cols + c;
                if (!averagedCandidates.empty()) {
                    chosenDist(r, c) = averagedCandidates.front().t;
                    chosenPoints(r, c) = averagedCandidates.front().point;
                    if (averagedCandidates.front().surface) {
                        chosenHits[static_cast<std::size_t>(idx)] = averagedCandidates.front();
                        chosenHitValid(r, c) = 1;
                    }
                    allHits[idx] = std::move(averagedCandidates);
                    raysHit.fetch_add(1, std::memory_order_relaxed);
                }

                const std::size_t done = outputDone.fetch_add(1, std::memory_order_relaxed) + 1;
                if ((done % 100000) == 0) {
                    #pragma omp critical
                    std::cout << "  output vertices processed: " << done << " / " << srcValidCount
                              << ", hit vertices: " << raysHit.load(std::memory_order_relaxed) << '\n';
                }
            }
        }

        std::cout << "Ray casting complete: " << raysDone.load() << " source rays, "
                  << raysHit.load() << " vertices hit, "
                  << rawHits.load() << " raw triangle hits before dedup, "
                  << sameWrapDiscarded.load() << " discarded same-wrap hits, "
                  << selfDiscarded.load() << " discarded source self-hits, "
                  << neighborPreferredHits.load() << " neighbor-preferred hit choices\n";

        enforce_neighbor_consistency(allHits, srcValid, cfg.neighborMaxDelta,
                                     cfg.neighborMaxFartherDelta,
                                     cfg.smoothRadius, cfg.smoothIters,
                                     chosenDist, chosenPoints);

        std::size_t repairTested = 0;
        std::size_t repairChanged = 0;
        repair_shifted_ray_outliers(cfg, index, umb, castOut, srcValid, rayDirs,
                                    chosenDist, chosenPoints,
                                    chosenHits, chosenHitValid,
                                    repairTested, repairChanged);

        const std::size_t validBeforeResample = count_valid(chosenPoints);
        const float currentSpacing = median_grid_spacing(chosenPoints);
        float resampleFactor = 1.0f;
        if (currentSpacing > 1e-3f) {
            resampleFactor = currentSpacing / cfg.spacing;
        }
        resampleFactor = std::clamp(resampleFactor, 0.05f, 20.0f);
        std::cout << "Output before resample: " << cols << "x" << rows
                  << ", valid vertices: " << validBeforeResample
                  << ", median spacing: " << currentSpacing
                  << ", target spacing: " << cfg.spacing
                  << ", resample factor: " << resampleFactor << '\n';

        cv::Mat_<cv::Vec3f> finalPoints = resample_points(chosenPoints, resampleFactor);
        std::cout << "Output after resample: " << finalPoints.cols << "x" << finalPoints.rows
                  << ", valid vertices: " << count_valid(finalPoints) << '\n';

        std::size_t snapHitCount = 0;
        std::size_t snapInterpolatedCount = 0;
        if (cfg.snapToPatch) {
            std::cout << "Snapping resampled vertices to patch triangle intersections along "
                      << (castOut ? "outward" : "inward")
                      << " normals, distance range [" << cfg.snapMinDistance
                      << ", " << cfg.snapMaxDistance << "]...\n";
            finalPoints = snap_points_to_patch_rays(finalPoints, cfg, index, umb, castOut,
                                                    snapHitCount, snapInterpolatedCount);
            std::cout << "Output after snap: valid vertices: " << count_valid(finalPoints)
                      << ", direct snap hits: " << snapHitCount
                      << ", interpolated misses: " << snapInterpolatedCount << '\n';
        }

        const std::string uuid = cfg.output.filename().empty()
            ? std::string("projected_tifxyz_") + now_id()
            : cfg.output.filename().string();
        auto out = std::make_unique<QuadSurface>(finalPoints, src->scale());
        Json meta;
        meta["source"] = "vc_project_tifxyz";
        meta["source_tifxyz"] = cfg.source.string();
        meta["umbilicus"] = cfg.umbilicus.string();
        meta["patch_dir"] = cfg.patchDir.string();
        meta["direction"] = cfg.direction;
        meta["z_min"] = indexZMin;
        meta["z_max"] = indexZMax;
        meta["max_distance"] = cfg.maxDistance;
        meta["min_distance"] = cfg.minDistance;
        meta["dedup_radius"] = cfg.dedupRadius;
        meta["same_wrap_tolerance"] = cfg.sameWrapTolerance;
        meta["spacing"] = cfg.spacing;
        meta["neighbor_max_delta"] = cfg.neighborMaxDelta;
        meta["neighbor_max_farther_delta"] = cfg.neighborMaxFartherDelta;
        meta["ray_neighborhood_radius"] = cfg.rayNeighborhoodRadius;
        meta["ray_neighborhood_min_samples"] = cfg.rayNeighborhoodMinSamples;
        meta["ray_neighborhood_max_delta"] = cfg.rayNeighborhoodMaxDelta;
        meta["ray_neighborhood_average_direct_hits"] = cfg.rayNeighborhoodAverageDirectHits;
        meta["neighbor_preferred_hit_choices"] = static_cast<double>(neighborPreferredHits.load());
        meta["index_stride"] = cfg.indexStride;
        meta["self_hit_ignore_radius"] = cfg.selfHitIgnoreRadius;
        meta["self_hit_epsilon"] = cfg.selfHitEpsilon;
        meta["snap_to_patch"] = cfg.snapToPatch;
        meta["snap_min_distance"] = cfg.snapMinDistance;
        meta["snap_max_distance"] = cfg.snapMaxDistance;
        meta["snap_interpolate_radius"] = cfg.snapInterpolateRadius;
        meta["snap_interpolate_iters"] = cfg.snapInterpolateIters;
        meta["snap_interpolate_min_neighbors"] = cfg.snapInterpolateMinNeighbors;
        meta["snap_direct_hits"] = static_cast<double>(snapHitCount);
        meta["snap_interpolated_misses"] = static_cast<double>(snapInterpolatedCount);
        meta["repair_iters"] = cfg.repairIters;
        meta["repair_radius"] = cfg.repairRadius;
        meta["repair_min_neighbors"] = cfg.repairMinNeighbors;
        meta["repair_closer_delta"] = cfg.repairCloserDelta;
        meta["repair_edge_factor"] = cfg.repairEdgeFactor;
        meta["repair_backtrack"] = cfg.repairBacktrack;
        meta["repair_forward"] = cfg.repairForward;
        meta["repair_min_move"] = cfg.repairMinMove;
        meta["repair_patch_bonus"] = cfg.repairPatchBonus;
        meta["repair_patch_penalty"] = cfg.repairPatchPenalty;
        meta["repair_tested"] = static_cast<double>(repairTested);
        meta["repair_changed"] = static_cast<double>(repairChanged);
        meta["index_cache_enabled"] = !cfg.noIndexCache;
        meta["index_cache_dir"] = cfg.indexCacheDir.string();
        meta["index_cache_key"] = cacheKey;
        out->meta = meta;

        try {
            std::cout << "Saving tifxyz: " << cfg.output << '\n';
            out->save(cfg.output.string(), uuid, true);
        } catch (const std::exception& ex) {
            std::cerr << "ERROR: failed to save output tifxyz: " << ex.what() << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
