#include "vc/fiber_tracer/FiberSkeletonRegions.hpp"

#include "vc/fiber_tracer/FiberSkeletonChoice.hpp"
#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <queue>
#include <set>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-9;

cv::Vec3d normalizedOrZero(const cv::Vec3d& value)
{
    const double length = cv::norm(value);
    if (!std::isfinite(length) || !(length > kEpsilon))
        return {0.0, 0.0, 0.0};
    return value / length;
}

// Unit tangent of a polyline at `arc`, oriented towards increasing arc.
cv::Vec3d polylineTangentAtArc(
    const std::vector<cv::Vec3d>& points, double arc, double span)
{
    const double length = fiberPolylineLength(points);
    const double half = std::max(span, kEpsilon);
    const double back = std::clamp(arc - half, 0.0, length);
    const double front = std::clamp(arc + half, 0.0, length);
    cv::Vec3d tangent = normalizedOrZero(
        fiberPolylinePointAtArc(points, front) -
        fiberPolylinePointAtArc(points, back));
    if (tangent.dot(tangent) <= 0.0 && points.size() >= 2)
        tangent = normalizedOrZero(points.back() - points.front());
    return tangent;
}

cv::Vec3d orthogonalTo(const cv::Vec3d& axis)
{
    const cv::Vec3d candidate = std::abs(axis[0]) < 0.9
        ? cv::Vec3d{1.0, 0.0, 0.0}
        : cv::Vec3d{0.0, 1.0, 0.0};
    return normalizedOrZero(candidate - axis * candidate.dot(axis));
}

void appendSamples(
    std::vector<cv::Vec3d>& out, const std::vector<cv::Vec3d>& points,
    double beginArc, double endArc, double spacing)
{
    if (points.size() < 2 || !(endArc > beginArc + kEpsilon))
        return;
    const double step = std::max(spacing, kEpsilon);
    for (double arc = beginArc; arc <= endArc + kEpsilon; arc += step)
        out.push_back(fiberPolylinePointAtArc(points, arc));
}


int presenceAtBase(const FiberSkeletonCrop& crop, const cv::Vec3d& pointBaseXYZ)
{
    const double scale = crop.predictionToBaseScale;
    if (!(scale > 0.0) || crop.presence.empty())
        return 0;
    const auto x = static_cast<long long>(
        std::llround(pointBaseXYZ[0] / scale) - crop.originZYX[2]);
    const auto y = static_cast<long long>(
        std::llround(pointBaseXYZ[1] / scale) - crop.originZYX[1]);
    const auto z = static_cast<long long>(
        std::llround(pointBaseXYZ[2] / scale) - crop.originZYX[0]);
    if (x < 0 || y < 0 || z < 0)
        return 0;
    if (static_cast<std::size_t>(x) >= crop.shapeZYX[2] ||
        static_cast<std::size_t>(y) >= crop.shapeZYX[1] ||
        static_cast<std::size_t>(z) >= crop.shapeZYX[0])
        return 0;
    return static_cast<int>(crop.presence[crop.flatIndex(
        static_cast<std::size_t>(z), static_cast<std::size_t>(y),
        static_cast<std::size_t>(x))]);
}

double medianOf(std::vector<double> values)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

FiberSkeletonSection fiberSkeletonSectionAt(
    const FiberSkeletonCrop& crop, int thresholdU8, const cv::Vec3d& pointBaseXYZ,
    const cv::Vec3d& tangentXYZ, double searchRadiusBase)
{
    FiberSkeletonSection section;
    const double scale = crop.predictionToBaseScale;
    if (!(scale > 0.0) || !(searchRadiusBase > 0.0) || crop.presence.empty())
        return section;
    cv::Vec3d tangent = tangentXYZ;
    const double tangentNorm = cv::norm(tangent);
    if (!(tangentNorm > kEpsilon))
        return section;
    tangent /= tangentNorm;

    // Any pair of axes spanning the normal plane will do to sample it; the
    // section's own axes are recovered from its second moments below.
    cv::Vec3d helper = std::abs(tangent[0]) < 0.9 ? cv::Vec3d{1.0, 0.0, 0.0}
                                                  : cv::Vec3d{0.0, 1.0, 0.0};
    cv::Vec3d axisU = helper - tangent * tangent.dot(helper);
    const double axisUNorm = cv::norm(axisU);
    if (!(axisUNorm > kEpsilon))
        return section;
    axisU /= axisUNorm;
    const cv::Vec3d axisV = tangent.cross(axisU);

    const int radiusSamples =
        std::max(1, static_cast<int>(std::lround(searchRadiusBase / scale)));
    const int side = 2 * radiusSamples + 1;
    const auto index = [&](int iu, int iv) {
        return static_cast<std::size_t>((iv + radiusSamples) * side +
                                        (iu + radiusSamples));
    };

    std::vector<std::uint8_t> inside(static_cast<std::size_t>(side) * side, 0);
    for (int iv = -radiusSamples; iv <= radiusSamples; ++iv) {
        for (int iu = -radiusSamples; iu <= radiusSamples; ++iu) {
            const double du = static_cast<double>(iu) * scale;
            const double dv = static_cast<double>(iv) * scale;
            if (du * du + dv * dv > searchRadiusBase * searchRadiusBase)
                continue;
            const cv::Vec3d sample = pointBaseXYZ + axisU * du + axisV * dv;
            const auto x = static_cast<long long>(
                std::llround(sample[0] / scale) - crop.originZYX[2]);
            const auto y = static_cast<long long>(
                std::llround(sample[1] / scale) - crop.originZYX[1]);
            const auto z = static_cast<long long>(
                std::llround(sample[2] / scale) - crop.originZYX[0]);
            if (x < 0 || y < 0 || z < 0)
                continue;
            if (static_cast<std::size_t>(x) >= crop.shapeZYX[2] ||
                static_cast<std::size_t>(y) >= crop.shapeZYX[1] ||
                static_cast<std::size_t>(z) >= crop.shapeZYX[0])
                continue;
            const auto value = crop.presence[crop.flatIndex(
                static_cast<std::size_t>(z), static_cast<std::size_t>(y),
                static_cast<std::size_t>(x))];
            if (static_cast<int>(value) >= thresholdU8)
                inside[index(iu, iv)] = 1;
        }
    }

    // Only the blob the point sits in counts. A fibre that merely passes nearby
    // without touching must not be mistaken for part of this section.
    std::vector<std::pair<int, int>> stack;
    if (inside[index(0, 0)] != 0) {
        stack.emplace_back(0, 0);
    } else {
        int bestU = 0;
        int bestV = 0;
        double bestDistance = std::numeric_limits<double>::max();
        for (int iv = -radiusSamples; iv <= radiusSamples; ++iv) {
            for (int iu = -radiusSamples; iu <= radiusSamples; ++iu) {
                if (inside[index(iu, iv)] == 0)
                    continue;
                const double distance =
                    static_cast<double>(iu) * iu + static_cast<double>(iv) * iv;
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestU = iu;
                    bestV = iv;
                }
            }
        }
        if (bestDistance == std::numeric_limits<double>::max())
            return section;
        stack.emplace_back(bestU, bestV);
    }

    std::vector<std::uint8_t> visited(inside.size(), 0);
    visited[index(stack.front().first, stack.front().second)] = 1;
    std::vector<std::pair<double, double>> blob;
    while (!stack.empty()) {
        const auto [iu, iv] = stack.back();
        stack.pop_back();
        blob.emplace_back(static_cast<double>(iu) * scale,
                          static_cast<double>(iv) * scale);
        for (int dvStep = -1; dvStep <= 1; ++dvStep) {
            for (int duStep = -1; duStep <= 1; ++duStep) {
                if (duStep == 0 && dvStep == 0)
                    continue;
                const int nu = iu + duStep;
                const int nv = iv + dvStep;
                if (nu < -radiusSamples || nu > radiusSamples ||
                    nv < -radiusSamples || nv > radiusSamples)
                    continue;
                const auto flat = index(nu, nv);
                if (visited[flat] != 0 || inside[flat] == 0)
                    continue;
                visited[flat] = 1;
                stack.emplace_back(nu, nv);
            }
        }
    }
    if (blob.empty())
        return section;

    double sumU = 0.0;
    double sumV = 0.0;
    double sumSquares = 0.0;
    for (const auto& [du, dv] : blob) {
        sumU += du;
        sumV += dv;
        sumSquares += du * du + dv * dv;
    }
    const double count = static_cast<double>(blob.size());
    const double meanU = sumU / count;
    const double meanV = sumV / count;
    double mUU = 0.0;
    double mUV = 0.0;
    double mVV = 0.0;
    for (const auto& [du, dv] : blob) {
        const double cu = du - meanU;
        const double cv = dv - meanV;
        mUU += cu * cu;
        mUV += cu * cv;
        mVV += cv * cv;
    }
    mUU /= count;
    mUV /= count;
    mVV /= count;
    // Principal axes of the section. The fibre presence is ribbon-shaped: it is
    // narrow across the sheet and wide along it, and only the narrow direction
    // localises the fibre.
    const double trace = mUU + mVV;
    const double gap = std::sqrt(std::max(
        0.0, (mUU - mVV) * (mUU - mVV) * 0.25 + mUV * mUV));
    const double eigLong = trace * 0.5 + gap;
    const double eigShort = std::max(0.0, trace * 0.5 - gap);
    cv::Vec2d longUV{mUV, eigLong - mUU};
    if (cv::norm(longUV) <= kEpsilon)
        longUV = cv::Vec2d{1.0, 0.0};
    longUV /= cv::norm(longUV);
    const cv::Vec2d shortUV{-longUV[1], longUV[0]};

    section.valid = true;
    section.centreBaseXYZ = pointBaseXYZ + axisU * meanU + axisV * meanV;
    section.longAxisXYZ = axisU * longUV[0] + axisV * longUV[1];
    section.shortAxisXYZ = axisU * shortUV[0] + axisV * shortUV[1];
    section.longSigmaBase = std::sqrt(eigLong);
    section.shortSigmaBase = std::sqrt(eigShort);
    section.rmsRadiusBase = std::sqrt(sumSquares / count);
    section.areaBase = count * scale * scale;
    double lowest = std::numeric_limits<double>::max();
    double highest = std::numeric_limits<double>::lowest();
    for (const auto& [du, dv] : blob) {
        const double along =
            (du - meanU) * longUV[0] + (dv - meanV) * longUV[1];
        lowest = std::min(lowest, along);
        highest = std::max(highest, along);
    }
    section.longMinBase = lowest;
    section.longMaxBase = highest;

    return section;
}

std::vector<cv::Vec3d> fiberSkeletonBlendEnds(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, const cv::Vec3d& startBaseXYZ,
    const cv::Vec3d& endBaseXYZ)
{
    auto points = pointsBaseXYZ;
    if (points.empty())
        return points;
    if (points.size() == 1) {
        points.front() = startBaseXYZ;
        return points;
    }
    const cv::Vec3d startShift = startBaseXYZ - points.front();
    const cv::Vec3d endShift = endBaseXYZ - points.back();
    const double total = fiberPolylineLength(points);
    double arc = 0.0;
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (index > 0)
            arc += cv::norm(pointsBaseXYZ[index] - pointsBaseXYZ[index - 1]);
        const double ramp = total > kEpsilon ? arc / total
            : static_cast<double>(index) / static_cast<double>(points.size() - 1);
        points[index] += startShift * (1.0 - ramp) + endShift * ramp;
    }
    return points;
}

cv::Vec3d refineFiberSkeletonPointInPlane(
    const FiberSkeletonCrop& crop, int thresholdU8, const cv::Vec3d& pointBaseXYZ,
    const cv::Vec3d& tangentXYZ, double radiusBase, int iterations)
{
    const double scale = crop.predictionToBaseScale;
    cv::Vec3d tangent = normalizedOrZero(tangentXYZ);
    if (!(scale > 0.0) || !(radiusBase > 0.0) || tangent.dot(tangent) <= 0.0)
        return pointBaseXYZ;
    const cv::Vec3d axisU = orthogonalTo(tangent);
    const cv::Vec3d axisV = tangent.cross(axisU);
    const int radiusSamples =
        std::max(1, static_cast<int>(std::lround(radiusBase / scale)));

    cv::Vec3d centre = pointBaseXYZ;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        cv::Vec3d sum{0.0, 0.0, 0.0};
        double weight = 0.0;
        for (int iv = -radiusSamples; iv <= radiusSamples; ++iv) {
            for (int iu = -radiusSamples; iu <= radiusSamples; ++iu) {
                const double du = static_cast<double>(iu) * scale;
                const double dv = static_cast<double>(iv) * scale;
                if (du * du + dv * dv > radiusBase * radiusBase)
                    continue;
                const cv::Vec3d sample = centre + axisU * du + axisV * dv;
                const int value = presenceAtBase(crop, sample);
                if (value < thresholdU8)
                    continue;
                sum += sample * static_cast<double>(value);
                weight += static_cast<double>(value);
            }
        }
        if (!(weight > 0.0))
            break;
        const cv::Vec3d next = sum / weight;
        const double moved = cv::norm(next - centre);
        centre = next;
        if (moved < 0.05 * scale)
            break;
    }
    return centre;
}

bool fiberSkeletonSearchPath(
    FiberSkeletonFieldProvider& fields, const FiberSkeletonPathSearchConfig& config,
    const std::vector<cv::Vec3d>& corridorBaseXYZ, const cv::Vec3d& startBaseXYZ,
    const cv::Vec3d& endBaseXYZ, std::vector<cv::Vec3d>& pathBaseXYZ)
{
    pathBaseXYZ.clear();
    const auto& crop = fields.crop();
    const double scale = crop.predictionToBaseScale;
    if (!(scale > 0.0) || crop.presence.empty() || corridorBaseXYZ.empty())
        return false;
    const double radius = config.corridorRadiusBase;
    if (!(radius > 0.0))
        return false;

    // Bounding box of the corridor in crop voxels.
    std::array<long long, 3> low{
        std::numeric_limits<long long>::max(),
        std::numeric_limits<long long>::max(),
        std::numeric_limits<long long>::max()};
    std::array<long long, 3> high{
        std::numeric_limits<long long>::lowest(),
        std::numeric_limits<long long>::lowest(),
        std::numeric_limits<long long>::lowest()};
    const auto include = [&](const cv::Vec3d& point) {
        for (int axis = 0; axis < 3; ++axis) {
            // point is XYZ; crop is ZYX
            const double value = point[2 - axis] / scale;
            const auto origin = crop.originZYX[axis];
            low[axis] = std::min(low[axis], static_cast<long long>(
                std::floor((value - radius / scale))) - origin);
            high[axis] = std::max(high[axis], static_cast<long long>(
                std::ceil((value + radius / scale))) - origin);
        }
    };
    for (const auto& point : corridorBaseXYZ)
        include(point);
    include(startBaseXYZ);
    include(endBaseXYZ);
    for (int axis = 0; axis < 3; ++axis) {
        low[axis] = std::max<long long>(low[axis], 0);
        high[axis] = std::min<long long>(
            high[axis], static_cast<long long>(crop.shapeZYX[axis]) - 1);
        if (high[axis] < low[axis])
            return false;
    }
    const std::array<std::size_t, 3> shape{
        static_cast<std::size_t>(high[0] - low[0] + 1),
        static_cast<std::size_t>(high[1] - low[1] + 1),
        static_cast<std::size_t>(high[2] - low[2] + 1)};
    const std::size_t count = shape[0] * shape[1] * shape[2];
    const auto local = [&](std::size_t z, std::size_t y, std::size_t x) {
        return (z * shape[1] + y) * shape[2] + x;
    };
    const auto baseOf = [&](std::size_t z, std::size_t y, std::size_t x) {
        return cv::Vec3d{
            (static_cast<double>(x + low[2]) + static_cast<double>(crop.originZYX[2])) * scale,
            (static_cast<double>(y + low[1]) + static_cast<double>(crop.originZYX[1])) * scale,
            (static_cast<double>(z + low[0]) + static_cast<double>(crop.originZYX[0])) * scale};
    };

    // Passable: above threshold and within the corridor of the skeleton piece.
    const double radiusSquared = radius * radius;
    std::vector<std::uint8_t> passable(count, 0);
    std::vector<std::size_t> cropIndexOf(count, kFiberSkeletonNoIndex);
    for (std::size_t z = 0; z < shape[0]; ++z) {
        for (std::size_t y = 0; y < shape[1]; ++y) {
            for (std::size_t x = 0; x < shape[2]; ++x) {
                const auto value = crop.presence[crop.flatIndex(
                    z + low[0], y + low[1], x + low[2])];
                if (static_cast<int>(value) < config.thresholdU8)
                    continue;
                const cv::Vec3d point = baseOf(z, y, x);
                bool near = false;
                for (std::size_t index = 1; index < corridorBaseXYZ.size() && !near;
                     ++index) {
                    const cv::Vec3d a = corridorBaseXYZ[index - 1];
                    const cv::Vec3d delta = corridorBaseXYZ[index] - a;
                    const double lengthSquared = delta.dot(delta);
                    const double t = lengthSquared > kEpsilon
                        ? std::clamp((point - a).dot(delta) / lengthSquared, 0.0, 1.0)
                        : 0.0;
                    const cv::Vec3d foot = a + delta * t;
                    near = (point - foot).dot(point - foot) <= radiusSquared;
                }
                if (corridorBaseXYZ.size() == 1) {
                    const cv::Vec3d d = point - corridorBaseXYZ.front();
                    near = d.dot(d) <= radiusSquared;
                }
                if (!near)
                    continue;
                passable[local(z, y, x)] = 1;
                cropIndexOf[local(z, y, x)] =
                    crop.flatIndex(z + low[0], y + low[1], x + low[2]);
            }
        }
    }

    // Presence and direction at every passable voxel, loaded once.
    std::vector<std::size_t> flatIndices;
    std::vector<std::size_t> sampleOf(count, kFiberSkeletonNoIndex);
    for (std::size_t flat = 0; flat < count; ++flat) {
        if (passable[flat] == 0)
            continue;
        sampleOf[flat] = flatIndices.size();
        flatIndices.push_back(cropIndexOf[flat]);
    }
    std::vector<FiberSkeletonFieldSample> loaded;
    fields.loadVoxels(flatIndices, loaded);
    std::vector<FiberLocalMetricSample> samples(loaded.size());
    for (std::size_t index = 0; index < loaded.size(); ++index) {
        samples[index].presence = static_cast<float>(loaded[index].presence);
        samples[index].valid = loaded[index].valid;
        samples[index].direction = prepareFiberLocalUnitDirection(cv::Vec3f{
            static_cast<float>(loaded[index].direction[0]),
            static_cast<float>(loaded[index].direction[1]),
            static_cast<float>(loaded[index].direction[2])});
    }
    // The tracing cost without its smoothness terms: a voxel search carries no
    // incoming direction, so the step itself stands in for the previous step
    // and only presence and alignment with the predicted direction remain.
    FiberLocalMetricConfig metric;
    metric.invalidPredictionCostPerVoxel = config.invalidPredictionCostPerVoxel;
    metric.smoothness.isotropicWeight = 0.0f;
    metric.smoothness.normalWeight = 0.0f;
    metric.smoothness.tangentWeight = 0.0f;
    const auto stepCost = [&](std::size_t from, std::size_t to,
                              const cv::Vec3f& stepUnit, float stepLength) {
        const auto& current = samples[sampleOf[from]];
        const auto& next = samples[sampleOf[to]];
        return static_cast<double>(
            fiberLocalMetricCostPrepared(
                &current, next, stepUnit, stepLength, stepUnit, stepLength,
                cv::Vec3f{0.0f, 0.0f, 0.0f}, false, metric)
                .total());
    };

    // Nearest passable voxel to a base point.
    const auto nearestPassable = [&](const cv::Vec3d& point) {
        std::size_t best = kFiberSkeletonNoIndex;
        double bestDistance = std::numeric_limits<double>::max();
        for (std::size_t z = 0; z < shape[0]; ++z) {
            for (std::size_t y = 0; y < shape[1]; ++y) {
                for (std::size_t x = 0; x < shape[2]; ++x) {
                    const auto flat = local(z, y, x);
                    if (passable[flat] == 0)
                        continue;
                    const cv::Vec3d d = baseOf(z, y, x) - point;
                    const double distance = d.dot(d);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        best = flat;
                    }
                }
            }
        }
        return best;
    };
    const std::size_t source = nearestPassable(startBaseXYZ);
    const std::size_t target = nearestPassable(endBaseXYZ);
    if (source == kFiberSkeletonNoIndex || target == kFiberSkeletonNoIndex)
        return false;

    // Dijkstra over 26-neighbours under that cost.
    std::vector<double> distance(count, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> cameFrom(count, kFiberSkeletonNoIndex);
    using Entry = std::pair<double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> queue;
    distance[source] = 0.0;
    queue.emplace(0.0, source);
    while (!queue.empty()) {
        const auto [cost, flat] = queue.top();
        queue.pop();
        if (cost > distance[flat] + kEpsilon)
            continue;
        if (flat == target)
            break;
        const std::size_t x = flat % shape[2];
        const std::size_t y = (flat / shape[2]) % shape[1];
        const std::size_t z = flat / (shape[2] * shape[1]);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0)
                        continue;
                    const long long nz = static_cast<long long>(z) + dz;
                    const long long ny = static_cast<long long>(y) + dy;
                    const long long nx = static_cast<long long>(x) + dx;
                    if (nz < 0 || ny < 0 || nx < 0 ||
                        nz >= static_cast<long long>(shape[0]) ||
                        ny >= static_cast<long long>(shape[1]) ||
                        nx >= static_cast<long long>(shape[2])) {
                        continue;
                    }
                    const auto next = local(
                        static_cast<std::size_t>(nz), static_cast<std::size_t>(ny),
                        static_cast<std::size_t>(nx));
                    if (passable[next] == 0)
                        continue;
                    const float stepLength = static_cast<float>(
                        scale * std::sqrt(static_cast<double>(
                                    dx * dx + dy * dy + dz * dz)));
                    const cv::Vec3f stepUnit = prepareFiberLocalUnitDirection(
                        cv::Vec3f{static_cast<float>(dx), static_cast<float>(dy),
                                  static_cast<float>(dz)});
                    const double step = stepCost(flat, next, stepUnit, stepLength);
                    if (cost + step + kEpsilon < distance[next]) {
                        distance[next] = cost + step;
                        cameFrom[next] = flat;
                        queue.emplace(distance[next], next);
                    }
                }
            }
        }
    }
    if (!std::isfinite(distance[target]))
        return false;
    for (std::size_t flat = target; flat != kFiberSkeletonNoIndex;
         flat = cameFrom[flat]) {
        const std::size_t x = flat % shape[2];
        const std::size_t y = (flat / shape[2]) % shape[1];
        const std::size_t z = flat / (shape[2] * shape[1]);
        pathBaseXYZ.push_back(baseOf(z, y, x));
        if (flat == source)
            break;
    }
    std::reverse(pathBaseXYZ.begin(), pathBaseXYZ.end());
    // The ends are the candidates themselves, not their nearest voxels.
    if (pathBaseXYZ.size() >= 2) {
        pathBaseXYZ.front() = startBaseXYZ;
        pathBaseXYZ.back() = endBaseXYZ;
    } else {
        pathBaseXYZ = {startBaseXYZ, endBaseXYZ};
    }
    return true;
}

FiberSkeletonTopology buildFiberSkeletonTopology(
    const FiberSkeletonGraph& graph, const FiberSkeletonCrop& crop,
    const FiberSkeletonSplitConfig& split, double supportSampleBase)
{
    const double entryOffsetBase = split.entryOffsetBase;
    if (!(entryOffsetBase > 0.0) || !std::isfinite(entryOffsetBase))
        throw std::invalid_argument("entry offset must be finite and positive");
    if (!(supportSampleBase > 0.0) || !std::isfinite(supportSampleBase))
        throw std::invalid_argument("support sampling must be finite and positive");
    if (!(split.sectionRadiusBase > 0.0) || !std::isfinite(split.sectionRadiusBase))
        throw std::invalid_argument("split section radius must be finite and positive");
    if (!(split.candidateMinOffsetBase >= 0.0) ||
        !std::isfinite(split.candidateMinOffsetBase)) {
        throw std::invalid_argument("candidate offset must be finite");
    }
    if (!(split.candidateRefineRadiusBase >= 0.0) ||
        !std::isfinite(split.candidateRefineRadiusBase)) {
        throw std::invalid_argument("candidate refine radius must be finite");
    }

    FiberSkeletonTopology topology;
    // One region per junction group. Grouping is a labelling produced by the
    // reduction, so a region's nodes and interior edges remain ordinary graph
    // elements and a route through it is an ordinary graph path.
    std::map<std::size_t, std::size_t> regionOfGroup;
    for (const auto& node : graph.nodes) {
        if (node.regionGroup == kFiberSkeletonNoGroup)
            continue;
        auto found = regionOfGroup.find(node.regionGroup);
        if (found == regionOfGroup.end()) {
            FiberSkeletonRegion region;
            region.id = topology.regions.size();
            region.group = node.regionGroup;
            found = regionOfGroup.emplace(node.regionGroup, region.id).first;
            topology.regions.push_back(std::move(region));
        }
        topology.regions[found->second].nodeIds.push_back(node.id);
    }
    for (auto& region : topology.regions) {
        cv::Vec3d sum{0.0, 0.0, 0.0};
        for (const auto nodeId : region.nodeIds)
            sum += graph.nodes[nodeId].positionBaseXYZ;
        region.centerBaseXYZ = sum /
            static_cast<double>(std::max<std::size_t>(1, region.nodeIds.size()));
    }

    const auto regionOfNode = [&](std::size_t nodeId) {
        const auto group = graph.nodes[nodeId].regionGroup;
        if (group == kFiberSkeletonNoGroup)
            return kFiberSkeletonNoIndex;
        const auto found = regionOfGroup.find(group);
        return found == regionOfGroup.end() ? kFiberSkeletonNoIndex
                                            : found->second;
    };

    for (const auto& edge : graph.edges) {
        if (edge.pointsBaseXYZ.size() < 2)
            continue;
        const std::size_t startRegion = regionOfNode(edge.nodeIds[0]);
        const std::size_t endRegion = regionOfNode(edge.nodeIds[1]);
        if (startRegion != kFiberSkeletonNoIndex && startRegion == endRegion) {
            // Interior to one region: part of what that region routes over,
            // never a line and never a port.
            topology.regions[startRegion].interiorEdgeIds.push_back(edge.id);
            continue;
        }

        const double length = edge.lengthBase;
        const double span = std::max(length * 0.05, kEpsilon);
        const bool startGuarded = startRegion != kFiberSkeletonNoIndex;
        const bool endGuarded = endRegion != kFiberSkeletonNoIndex;

        // A node is one skeleton voxel, so the region boundary along this edge
        // is the node itself and the offset is measured straight from it.
        double startArc = startGuarded ? std::min(entryOffsetBase, length) : 0.0;
        double endArc =
            endGuarded ? std::max(length - entryOffsetBase, 0.0) : length;
        bool shared = false;
        if (startGuarded && endGuarded && !(endArc > startArc + kEpsilon)) {
            startArc = endArc = length * 0.5;
            shared = true;
        }

        const auto addPort = [&](double arc, std::size_t primaryRegion,
                                 std::size_t primaryNode,
                                 std::size_t secondaryRegion,
                                 std::size_t secondaryNode,
                                 bool awayFromStart) {
            FiberSkeletonPort port;
            port.id = topology.ports.size();
            port.edgeId = edge.id;
            port.regionIds = {primaryRegion, secondaryRegion};
            port.nodeIds = {primaryNode, secondaryNode};
            port.arcFromNodeBase = {awayFromStart ? arc : length - arc,
                                    awayFromStart ? length - arc : arc};
            const cv::Vec3d skeletonPoint =
                fiberPolylinePointAtArc(edge.pointsBaseXYZ, arc);
            const cv::Vec3d forward =
                polylineTangentAtArc(edge.pointsBaseXYZ, arc, span);
            port.tangentXYZ = awayFromStart ? forward : -forward;
            port.shared = secondaryRegion != kFiberSkeletonNoIndex;

            // The port's positions come from the section of the prediction at
            // this thinning voxel, restricted to the piece of the mask the
            // voxel lies in: the centre, and each extreme along the section's
            // long axis that is far enough from the centre to be a different
            // fibre. Which of them the fibre uses is a routing choice. A
            // shared midpoint port belongs to two regions that would each
            // want to pick for it, so it keeps the centre only.
            const auto section = fiberSkeletonSectionAt(
                crop, split.thresholdU8, skeletonPoint, forward,
                split.sectionRadiusBase);
            port.initialPositionBaseXYZ = skeletonPoint;
            if (!section.valid) {
                port.candidatesBaseXYZ = {skeletonPoint};
            } else {
                port.sectionWidthBase = section.longSigmaBase;
                port.candidatesBaseXYZ.push_back(section.centreBaseXYZ);
                if (!port.shared) {
                    for (const double along :
                         {section.longMinBase, section.longMaxBase}) {
                        if (std::abs(along) > split.candidateMinOffsetBase) {
                            port.candidatesBaseXYZ.push_back(
                                section.centreBaseXYZ +
                                section.longAxisXYZ * along);
                        }
                    }
                }
                // An extreme sits on the edge of the mask; a few steps of a
                // small local centroid move it onto the fibre it belongs to.
                for (auto& candidate : port.candidatesBaseXYZ) {
                    candidate = refineFiberSkeletonPointInPlane(
                        crop, split.thresholdU8, candidate, forward,
                        split.candidateRefineRadiusBase,
                        split.candidateRefineIterations);
                }
            }
            port.candidateIsFibre.assign(port.candidatesBaseXYZ.size(), 1);
            if (port.candidatesBaseXYZ.size() > 1)
                port.candidateIsFibre.front() = 0;
            port.positionBaseXYZ = port.candidatesBaseXYZ.front();
            topology.ports.push_back(port);
            topology.regions[primaryRegion].portIds.push_back(port.id);
            if (secondaryRegion != kFiberSkeletonNoIndex &&
                secondaryRegion != primaryRegion) {
                topology.regions[secondaryRegion].portIds.push_back(port.id);
            }
            return port.id;
        };

        std::size_t startPort = kFiberSkeletonNoIndex;
        std::size_t endPort = kFiberSkeletonNoIndex;
        if (shared) {
            startPort = addPort(startArc, startRegion, edge.nodeIds[0],
                                endRegion, edge.nodeIds[1], true);
            endPort = startPort;
            ++topology.sharedMidpointPorts;
        } else {
            if (startGuarded)
                startPort = addPort(startArc, startRegion, edge.nodeIds[0],
                                    kFiberSkeletonNoIndex,
                                    kFiberSkeletonNoIndex, true);
            if (endGuarded)
                endPort = addPort(endArc, endRegion, edge.nodeIds[1],
                                  kFiberSkeletonNoIndex,
                                  kFiberSkeletonNoIndex, false);
        }

        // The stub between a region node and its port is the region's support.
        if (startGuarded) {
            appendSamples(topology.regions[startRegion].supportPointsBaseXYZ,
                          edge.pointsBaseXYZ, 0.0, startArc, supportSampleBase);
        }
        if (endGuarded) {
            appendSamples(topology.regions[endRegion].supportPointsBaseXYZ,
                          edge.pointsBaseXYZ, endArc, length, supportSampleBase);
        }

        if (shared || !(endArc > startArc + kEpsilon)) {
            ++topology.droppedShortEdges;
            continue;
        }

        FiberSkeletonLine line;
        line.id = topology.lines.size();
        line.edgeId = edge.id;
        line.initialPointsBaseXYZ =
            fiberPolylineSlice(edge.pointsBaseXYZ, startArc, endArc);
        line.pointsBaseXYZ = line.initialPointsBaseXYZ;
        line.initialLengthBase = fiberPolylineLength(line.initialPointsBaseXYZ);
        line.lengthBase = line.initialLengthBase;
        line.portIds = {startPort, endPort};
        topology.lines.push_back(std::move(line));
    }

    for (auto& region : topology.regions) {
        for (const auto edgeId : region.interiorEdgeIds) {
            const auto& edge = graph.edges[edgeId];
            appendSamples(region.supportPointsBaseXYZ, edge.pointsBaseXYZ, 0.0,
                          edge.lengthBase, supportSampleBase);
        }
        if (region.supportPointsBaseXYZ.empty())
            region.supportPointsBaseXYZ.push_back(region.centerBaseXYZ);
    }
    return topology;
}

void refineFiberSkeletonLines(
    FiberSkeletonTopology& topology, FiberSkeletonFieldProvider& fields,
    const FiberSkeletonPathSearchConfig& search, bool routeThroughCentre,
    const FiberSkeletonCostWeights& weights,
    const FiberSkeletonSolverConfig& solver, double stepBase)
{
    if (!(stepBase > 0.0) || !std::isfinite(stepBase))
        throw std::invalid_argument("refinement step must be finite and positive");
    topology.searchedInitials = 0;
    topology.unsearchedInitials = 0;

    for (auto& line : topology.lines) {
        line.variants.clear();
        line.entryVariants = {};
        const FiberSkeletonPort* startPort = line.portIds[0] != kFiberSkeletonNoIndex
            ? &topology.ports[line.portIds[0]] : nullptr;
        const FiberSkeletonPort* endPort = line.portIds[1] != kFiberSkeletonNoIndex
            ? &topology.ports[line.portIds[1]] : nullptr;
        const std::size_t startCount =
            startPort ? std::max<std::size_t>(1, startPort->candidatesBaseXYZ.size()) : 1;
        const std::size_t endCount =
            endPort ? std::max<std::size_t>(1, endPort->candidatesBaseXYZ.size()) : 1;

        FiberSkeletonRefineConfig config;
        config.weights = weights;
        config.solver = solver;
        // Both ends stay fixed. A free end is under-constrained along the line:
        // once the geometry is aligned the objective is flat in that direction.
        config.freeStart = false;
        config.freeEnd = false;
        if (startPort) {
            const auto tangent = normalizedOrZero(startPort->tangentXYZ);
            if (tangent.dot(tangent) > 0.0)
                config.startTangent = tangent;
        }
        if (endPort) {
            const auto tangent = normalizedOrZero(-endPort->tangentXYZ);
            if (tangent.dot(tangent) > 0.0)
                config.endTangent = tangent;
        }

        // One solve per pair of end candidates. The ends are the only thing
        // that differs between variants; the interior starts from the same
        // recentred stretch with its ends moved onto the chosen candidates.
        for (std::size_t startIndex = 0; startIndex < startCount; ++startIndex) {
            for (std::size_t endIndex = 0; endIndex < endCount; ++endIndex) {
                FiberSkeletonLineVariant variant;
                variant.candidates = {startIndex, endIndex};
                const cv::Vec3d startPoint = startPort
                    ? startPort->candidatesBaseXYZ[startIndex]
                    : line.initialPointsBaseXYZ.front();
                const cv::Vec3d endPoint = endPort
                    ? endPort->candidatesBaseXYZ[endIndex]
                    : line.initialPointsBaseXYZ.back();
                // The skeleton piece says which stretch of the prediction
                // this is; the shortest path through that stretch between
                // the two candidates is the initial line.
                std::vector<cv::Vec3d> initial;
                if (fiberSkeletonSearchPath(fields, search, line.initialPointsBaseXYZ,
                                            startPoint, endPoint, initial)) {
                    ++topology.searchedInitials;
                } else {
                    ++topology.unsearchedInitials;
                    initial = fiberSkeletonBlendEnds(
                        line.initialPointsBaseXYZ, startPoint, endPoint);
                }
                const double initialLength = fiberPolylineLength(initial);
                if (initial.size() < 2 || !(initialLength > stepBase)) {
                    variant.pointsBaseXYZ = initial;
                    variant.lengthBase = initialLength;
                    variant.refined = false;
                    if (initial.size() >= 2) {
                        variant.cost = evaluateFiberSkeletonPolylineCost(
                            initial, fields, config);
                    }
                } else {
                    const auto result = refineFixedEndpointPolyline(
                        resampleFiberPolyline(initial, stepBase), fields, config);
                    variant.pointsBaseXYZ = result.pointsBaseXYZ;
                    variant.lengthBase = result.lengthBase;
                    variant.cost = result.finalCost;
                    variant.refined = true;
                    variant.iterations = result.iterations;
                    variant.converged = result.converged;
                }
                line.variants.push_back(std::move(variant));
            }
        }
        line.initialLengthBase = fiberPolylineLength(line.initialPointsBaseXYZ);
        // Each sub-port's cheapest variant: the route a fibre takes when it
        // enters the line there, from either side.
        line.entryVariants[0].assign(startCount, kFiberSkeletonNoIndex);
        line.entryVariants[1].assign(endCount, kFiberSkeletonNoIndex);
        const auto isFibre = [&](std::size_t end, std::size_t candidate) {
            const FiberSkeletonPort* port = end == 0 ? startPort : endPort;
            if (port == nullptr || candidate >= port->candidateIsFibre.size())
                return true;
            return port->candidateIsFibre[candidate] != 0;
        };
        for (std::size_t index = 0; index < line.variants.size(); ++index) {
            const auto& variant = line.variants[index];
            if (!routeThroughCentre &&
                (!isFibre(0, variant.candidates[0]) ||
                 !isFibre(1, variant.candidates[1]))) {
                continue;
            }
            for (std::size_t end = 0; end < 2; ++end) {
                auto& entry = line.entryVariants[end][variant.candidates[end]];
                if (entry == kFiberSkeletonNoIndex ||
                    variant.cost.total() <
                        line.variants[entry].cost.total() - kEpsilon) {
                    entry = index;
                }
            }
        }
        applyFiberSkeletonLineVariant(line, 0);
    }
}

void applyFiberSkeletonLineVariant(FiberSkeletonLine& line, std::size_t variant)
{
    if (variant >= line.variants.size())
        return;
    const auto& chosen = line.variants[variant];
    line.pointsBaseXYZ = chosen.pointsBaseXYZ;
    line.lengthBase = chosen.lengthBase;
    line.cost = chosen.cost;
    line.refined = chosen.refined;
    line.iterations = chosen.iterations;
    line.converged = chosen.converged;
}

}  // namespace vc::fiber_tracer
