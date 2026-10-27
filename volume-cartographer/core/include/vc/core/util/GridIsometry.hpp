#pragma once

// Grid isometry metrics for a tifxyz surface, the C++ half of ScrollPrize/villa
// issue #2279. A tifxyz grid is a 2D array of 3D points whose meta.json promises,
// via `scale`, that neighbouring nodes sit `1 / scale` voxels apart. Nothing in the
// format records whether the grid keeps that promise, so segments cannot be compared
// for stretch by a standard number. This header computes the block the issue proposes
// and writes it under the `isometry` key of a surface's metadata.
//
// The metric definitions mirror `vc::surface::SurfaceArea` for validity (a corner is
// invalid when it is non-finite or the (-1, -1, -1) sentinel) and the issue's field
// list, and they are stated explicitly here so the block can be compared across
// segments and against the reference numpy implementation the issue links to
// (`kit tifxyz isometry`). The same numbers are produced in Python by
// `vesuvius.tifxyz.compute_grid_isometry`. The flatboi metrics (l2_mean,
// l2_median, linf, area_error) come from the shared per-triangle core in
// `TriangleStretch.hpp` (villa #2279 item 4), which flatboi itself also calls, so
// the grid and flatboi cannot drift.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/TriangleStretch.hpp"

namespace vc::surface
{
// Bumped when thresholds/definitions change, so a stored block can be read against
// the rules that produced it.
inline constexpr int kIsometryVersion = 1;

// An edge is "stretched" (the sheet is pulled apart or collapsed there) when it is
// more than 1.5x or less than 1/1.5x the promised step.
inline constexpr double kIsometryStretchFactor = 1.5;

// within_10pct / within_25pct count an edge as inside the band when its length ratio
// to the promised step is in [1/1.1, 1.1] / [1/1.25, 1.25].
inline constexpr double kIsometryWithinFactor10 = 1.1;
inline constexpr double kIsometryWithinFactor25 = 1.25;

inline constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

struct GridIsometryStats
{
    int version = kIsometryVersion;
    double step_u = 1.0;
    double step_v = 1.0;
    std::size_t edges = 0;
    double within_10pct = 0.0;
    double within_25pct = 0.0;
    double median_abs_log_ratio = 0.0;
    double area_ratio = 0.0;
    double angle_p5 = 0.0;
    double angle_p50 = 0.0;
    double angle_p95 = 0.0;
    std::size_t stretched_nodes = 0;
    double stretched_area_vx2 = 0.0;
    double l2_mean = 0.0;
    double l2_median = 0.0;
    double linf = 0.0;
    double area_error = 0.0;
};

namespace detail
{
template <typename Vec>
inline bool nodeValid(const Vec& v)
{
    return isFiniteVec(v) && !isSentinelInvalid(v);
}

// numpy's linear-method percentile of an ascending-sorted sample, so the C++ block
// matches the Python writer bit for bit. numpy evaluates the position as
// (n - 1) * q in double and interpolates with the >= 0.5 branch taken from the upper
// value, which is what `_lerp` does.
inline double percentileLinear(const std::vector<double>& sorted, double percentile)
{
    const std::size_t n = sorted.size();
    if (n == 0) {
        return 0.0;
    }
    if (n == 1) {
        return sorted[0];
    }
    const double q = percentile / 100.0;
    const double position = static_cast<double>(n - 1) * q;
    const auto lower = static_cast<std::size_t>(std::floor(position));
    if (lower >= n - 1) {
        return sorted[n - 1];
    }
    const double fraction = position - static_cast<double>(lower);
    if (fraction == 0.0) {
        return sorted[lower];
    }
    const double a = sorted[lower];
    const double b = sorted[lower + 1];
    const double diff = b - a;
    return fraction >= 0.5 ? b - diff * (1.0 - fraction) : a + diff * fraction;
}

inline void pushArray3(utils::Json& out, double a, double b, double c)
{
    out = utils::Json::array();
    out.push_back(a);
    out.push_back(b);
    out.push_back(c);
}

inline void pushArray2(utils::Json& out, double a, double b)
{
    out = utils::Json::array();
    out.push_back(a);
    out.push_back(b);
}
} // namespace detail

// One pass over the grid: every edge with both ends valid, every cell with all four
// corners valid, exactly as SurfaceArea skips cells for the area metric.
template <typename Vec>
inline GridIsometryStats computeGridIsometryStats(const cv::Mat_<Vec>& points,
                                                  double step_u,
                                                  double step_v)
{
    GridIsometryStats stats;
    stats.step_u = step_u;
    stats.step_v = step_v;
    if (points.empty() || points.rows < 2 || points.cols < 2) {
        return stats;
    }

    const int rows = points.rows;
    const int cols = points.cols;

    // Node validity, reusing the SI/area convention.
    cv::Mat_<std::uint8_t> ok(rows, cols, std::uint8_t(0));
    bool anyValid = false;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (detail::nodeValid(points(y, x))) {
                ok(y, x) = 1;
                anyValid = true;
            }
        }
    }
    if (!anyValid) {
        return stats;
    }

    // Edges: along columns (u) and along rows (v), each with both ends valid.
    std::vector<double> ratios;
    ratios.reserve(static_cast<std::size_t>(rows) * std::max(0, cols - 1) +
                   static_cast<std::size_t>(cols) * std::max(0, rows - 1));
    cv::Mat_<std::uint8_t> stretched(rows, cols, std::uint8_t(0));

    auto edgeStep = [&](const Vec& a, const Vec& b, double step, double& ratio, bool& stretchedEdge) {
        const double dx = static_cast<double>(b[0]) - static_cast<double>(a[0]);
        const double dy = static_cast<double>(b[1]) - static_cast<double>(a[1]);
        const double dz = static_cast<double>(b[2]) - static_cast<double>(a[2]);
        const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
        ratio = step > 0.0 ? len / step : len;
        stretchedEdge = ratio > kIsometryStretchFactor || ratio < 1.0 / kIsometryStretchFactor;
    };

    auto collectEdge = [&](int y, int x, int y2, int x2, double step) {
        double ratio = 0.0;
        bool bad = false;
        edgeStep(points(y, x), points(y2, x2), step, ratio, bad);
        ratios.push_back(ratio);
        if (bad) {
            stretched(y, x) = 1;
            stretched(y2, x2) = 1;
        }
    };

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x + 1 < cols; ++x) {
            if (ok(y, x) && ok(y, x + 1)) {
                collectEdge(y, x, y, x + 1, step_u);
            }
        }
    }
    for (int y = 0; y + 1 < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (ok(y, x) && ok(y + 1, x)) {
                collectEdge(y, x, y + 1, x, step_v);
            }
        }
    }

    stats.edges = ratios.size();
    if (!ratios.empty()) {
        std::size_t within10 = 0;
        std::size_t within25 = 0;
        for (double r : ratios) {
            if (r >= 1.0 / kIsometryWithinFactor10 && r <= kIsometryWithinFactor10) {
                ++within10;
            }
            if (r >= 1.0 / kIsometryWithinFactor25 && r <= kIsometryWithinFactor25) {
                ++within25;
            }
        }
        const double n = static_cast<double>(ratios.size());
        stats.within_10pct = static_cast<double>(within10) / n;
        stats.within_25pct = static_cast<double>(within25) / n;

        std::vector<double> logs(ratios.size());
        for (std::size_t i = 0; i < ratios.size(); ++i) {
            logs[i] = std::abs(std::log(ratios[i]));
        }
        stats.median_abs_log_ratio = detail::median(logs);
    }

    std::size_t stretchedNodes = 0;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (ok(y, x) && stretched(y, x)) {
                ++stretchedNodes;
            }
        }
    }
    stats.stretched_nodes = stretchedNodes;
    // #2279: "their area at one grid cell per node" -- the node count times the
    // promised cell area, not the 3D area of every cell a stretched node touches.
    stats.stretched_area_vx2 = static_cast<double>(stretchedNodes) * step_u * step_v;

    // Cells: all four corners valid. area_ratio and the flatboi metrics are taken
    // over the same cells; angles at each cell's (r, c) corner.
    std::vector<double> angles;
    angles.reserve(static_cast<std::size_t>(std::max(0, rows - 1)) * std::max(0, cols - 1));
    double cellAreaSum = 0.0;
    std::size_t cellCount = 0;

    // flatboi's stretch metrics come from the shared per-triangle core
    // (TriangleStretch.hpp, villa #2279 item 4), so the grid and flatboi agree by
    // construction. Both triangles of a cell have the same diagonal split as
    // SurfaceArea's quadArea (p00-p10-p11 and p00-p11-p01); their parameter
    // coordinates are the cell's own uniform ones, column x step_u and row x step_v.
    std::vector<TriangleStretch> triangles;
    triangles.reserve(2 * static_cast<std::size_t>(std::max(0, rows - 1)) *
                      static_cast<std::size_t>(std::max(0, cols - 1)));
    const double tri1_uv[3][2] = {{0.0, 0.0}, {step_u, 0.0}, {step_u, step_v}};
    const double tri2_uv[3][2] = {{0.0, 0.0}, {step_u, step_v}, {0.0, step_v}};
    for (int y = 0; y + 1 < rows; ++y) {
        for (int x = 0; x + 1 < cols; ++x) {
            if (!(ok(y, x) && ok(y, x + 1) && ok(y + 1, x) && ok(y + 1, x + 1))) {
                continue;
            }
            const Vec& p00 = points(y, x);
            const Vec& p10 = points(y, x + 1);
            const Vec& p01 = points(y + 1, x);
            const Vec& p11 = points(y + 1, x + 1);

            ++cellCount;
            cellAreaSum += detail::quadArea(p00, p10, p01, p11);

            // Grid angle at this cell's (r, c) corner, between its u and v edges.
            const double eux = static_cast<double>(p10[0]) - p00[0];
            const double euy = static_cast<double>(p10[1]) - p00[1];
            const double euz = static_cast<double>(p10[2]) - p00[2];
            const double evx = static_cast<double>(p01[0]) - p00[0];
            const double evy = static_cast<double>(p01[1]) - p00[1];
            const double evz = static_cast<double>(p01[2]) - p00[2];
            const double nu = std::sqrt(eux * eux + euy * euy + euz * euz);
            const double nv = std::sqrt(evx * evx + evy * evy + evz * evz);
            const double denom = nu * nv;
            if (denom > 0.0) {
                const double cosv = (eux * evx + euy * evy + euz * evz) / denom;
                const double clamped = std::min(1.0, std::max(-1.0, cosv));
                angles.push_back(std::acos(clamped) * kRadiansToDegrees);
            }

            const double p1[3][3] = {{p00[0], p00[1], p00[2]},
                                     {p10[0], p10[1], p10[2]},
                                     {p11[0], p11[1], p11[2]}};
            const double p2[3][3] = {{p00[0], p00[1], p00[2]},
                                     {p11[0], p11[1], p11[2]},
                                     {p01[0], p01[1], p01[2]}};
            triangles.push_back(triangleStretch(p1, tri1_uv));
            triangles.push_back(triangleStretch(p2, tri2_uv));
        }
    }

    if (cellCount > 0 && step_u > 0.0 && step_v > 0.0) {
        stats.area_ratio = cellAreaSum / (step_u * step_v * static_cast<double>(cellCount));
    }

    if (!angles.empty()) {
        std::sort(angles.begin(), angles.end());
        stats.angle_p5 = detail::percentileLinear(angles, 5.0);
        stats.angle_p50 = detail::percentileLinear(angles, 50.0);
        stats.angle_p95 = detail::percentileLinear(angles, 95.0);
    }

    const StretchMetrics stretch = aggregateStretch(triangles);
    stats.l2_mean = stretch.l2_mean;
    stats.l2_median = stretch.l2_median;
    stats.linf = stretch.linf;
    stats.area_error = stretch.area_error;

    return stats;
}

inline GridIsometryStats computeGridIsometryStats(const QuadSurface& surface)
{
    const auto* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return GridIsometryStats{};
    }
    const cv::Vec2f scale = surface.scale();
    const double step_u = scale[0] != 0.0f ? 1.0 / static_cast<double>(scale[0]) : 1.0;
    const double step_v = scale[1] != 0.0f ? 1.0 / static_cast<double>(scale[1]) : 1.0;
    return computeGridIsometryStats(*points, step_u, step_v);
}

// The block written under `isometry` in meta.json: exactly the issue's field list.
inline utils::Json gridIsometryToJson(const GridIsometryStats& s)
{
    utils::Json out;
    out["version"] = s.version;
    detail::pushArray2(out["step_vx"], s.step_u, s.step_v);
    out["edges"] = s.edges;
    out["within_10pct"] = s.within_10pct;
    out["within_25pct"] = s.within_25pct;
    out["median_abs_log_ratio"] = s.median_abs_log_ratio;
    out["area_ratio"] = s.area_ratio;
    detail::pushArray3(out["angle_deg"], s.angle_p5, s.angle_p50, s.angle_p95);
    out["stretched_nodes"] = s.stretched_nodes;
    out["stretched_area_vx2"] = s.stretched_area_vx2;
    out["l2_mean"] = s.l2_mean;
    out["l2_median"] = s.l2_median;
    out["linf"] = s.linf;
    out["area_error"] = s.area_error;
    return out;
}

template <typename Vec>
inline utils::Json computeGridIsometry(const cv::Mat_<Vec>& points, double step_u, double step_v)
{
    return gridIsometryToJson(computeGridIsometryStats(points, step_u, step_v));
}

// The promised step is `1 / scale`, and `scale` is what the metadata stores. It is
// read here from the metadata itself (a JSON double) when present, so the block is
// measured against exactly the number a reader sees -- the surface's own scale is a
// float32 copy, which would perturb the step it is being compared to.
inline bool isometryStepsFromMeta(const utils::Json& meta, double& step_u, double& step_v)
{
    if (!meta.is_object() || !meta.contains("scale")) {
        return false;
    }
    const utils::Json& scale = meta["scale"];
    if (!scale.is_array() || scale.size() < 2) {
        return false;
    }
    const utils::Json& s0j = scale.at(0);
    const utils::Json& s1j = scale.at(1);
    if (!s0j.is_number() || !s1j.is_number()) {
        return false;
    }
    const double s0 = s0j.get_double();
    const double s1 = s1j.get_double();
    if (!(std::isfinite(s0) && std::isfinite(s1)) || s0 <= 0.0 || s1 <= 0.0) {
        return false;
    }
    step_u = 1.0 / s0;
    step_v = 1.0 / s1;
    return true;
}

// Write the block into an object's metadata next to storeAreaMeta. Empty-scale-safe:
// a 0 step falls back to length ratios so a block is still produced.
inline void storeIsometryMeta(utils::Json& meta, const cv::Mat_<cv::Vec3f>& points, double step_u, double step_v)
{
    if (step_u <= 0.0) {
        step_u = 1.0;
    }
    if (step_v <= 0.0) {
        step_v = 1.0;
    }
    meta["isometry"] = computeGridIsometry(points, step_u, step_v);
}

inline void storeIsometryMeta(utils::Json& meta, const cv::Mat_<cv::Vec3f>& points, const cv::Vec2f& scale)
{
    const double step_u = scale[0] != 0.0f ? 1.0 / static_cast<double>(scale[0]) : 1.0;
    const double step_v = scale[1] != 0.0f ? 1.0 / static_cast<double>(scale[1]) : 1.0;
    storeIsometryMeta(meta, points, step_u, step_v);
}

inline void storeIsometryMeta(utils::Json& meta, const QuadSurface& surface)
{
    const auto* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }
    double step_u = 1.0;
    double step_v = 1.0;
    if (!isometryStepsFromMeta(meta, step_u, step_v)) {
        const cv::Vec2f scale = surface.scale();
        step_u = scale[0] != 0.0f ? 1.0 / static_cast<double>(scale[0]) : 1.0;
        step_v = scale[1] != 0.0f ? 1.0 / static_cast<double>(scale[1]) : 1.0;
    }
    storeIsometryMeta(meta, *points, step_u, step_v);
}
} // namespace vc::surface
