#include "vc/lasagna/LineViewBuilder.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kRollSmoothness = 4.0;
constexpr int kRollSmoothIterations = 80;

bool finite(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

double norm(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    if (!finite(v)) {
        return {0.0, 0.0, 0.0};
    }
    const double n = norm(v);
    if (n <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

bool validDirection(const cv::Vec3d& v)
{
    return finite(v) && norm(v) > kEpsilon;
}

cv::Vec3d axisFallbackLeastAlignedWith(const cv::Vec3d& reference)
{
    const cv::Vec3d r = normalizedOrZero(reference);
    const cv::Vec3d axes[] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const cv::Vec3d* best = &axes[0];
    double bestAbsDot = std::abs(r.dot(*best));
    for (const auto& axis : axes) {
        const double absDot = std::abs(r.dot(axis));
        if (absDot < bestAbsDot) {
            best = &axis;
            bestAbsDot = absDot;
        }
    }
    return *best;
}

cv::Vec3f toVec3f(const cv::Vec3d& v)
{
    return {static_cast<float>(v[0]),
            static_cast<float>(v[1]),
            static_cast<float>(v[2])};
}

std::vector<double> crossOffsets(double halfWidth, int samples)
{
    if (samples < 2) {
        throw std::invalid_argument("LineViewConfig::crossSamples must be at least 2");
    }
    std::vector<double> offsets;
    offsets.reserve(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples - 1);
        offsets.push_back(-halfWidth + 2.0 * halfWidth * t);
    }
    return offsets;
}

double typicalStepSize(const std::vector<SegmentNormalSample>& samples)
{
    std::vector<double> steps;
    steps.reserve(samples.size());
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        const double step = norm(samples[i + 1].position - samples[i].position);
        if (std::isfinite(step) && step > kEpsilon) {
            steps.push_back(step);
        }
    }
    if (steps.empty()) {
        return 1.0;
    }
    std::sort(steps.begin(), steps.end());
    return steps[steps.size() / 2];
}

double resolvedHalfExtent(double configuredHalfExtent,
                          const std::vector<SegmentNormalSample>& samples,
                          int crossSamples)
{
    if (configuredHalfExtent > 0.0) {
        return configuredHalfExtent;
    }
    return typicalStepSize(samples) * static_cast<double>(crossSamples - 1) * 0.5;
}

std::vector<SegmentNormalSample> controlPointSamples(const LineModel& line)
{
    std::vector<SegmentNormalSample> samples;
    samples.reserve(line.points.size());
    for (const auto& point : line.points) {
        samples.push_back({0.0, point.position, point.sampledNormal});
    }
    return samples;
}

std::vector<SegmentNormalSample> denseSamples(const LineModel& line)
{
    std::vector<SegmentNormalSample> samples;
    for (const auto& segment : line.segmentSamples) {
        for (const auto& sample : segment.samples) {
            if (!samples.empty() && norm(sample.position - samples.back().position) <= kEpsilon) {
                continue;
            }
            samples.push_back(sample);
        }
    }
    return samples;
}

std::vector<cv::Vec3d> resolvedNormals(const std::vector<SegmentNormalSample>& samples)
{
    std::vector<cv::Vec3d> normals(samples.size(), {0.0, 0.0, 0.0});
    std::vector<int> validIndices;
    validIndices.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        const cv::Vec3d normal = normalizedOrZero(samples[i].sampledNormal.normal);
        if (samples[i].sampledNormal.valid && validDirection(normal)) {
            normals[i] = normal;
            validIndices.push_back(static_cast<int>(i));
        }
    }

    if (validIndices.empty()) {
        std::fill(normals.begin(), normals.end(), cv::Vec3d{0.0, 0.0, 1.0});
        return normals;
    }

    for (size_t i = 0; i < samples.size(); ++i) {
        if (validDirection(normals[i])) {
            continue;
        }
        int nearest = validIndices.front();
        int bestDistance = std::abs(static_cast<int>(i) - nearest);
        for (const int index : validIndices) {
            const int distance = std::abs(static_cast<int>(i) - index);
            if (distance < bestDistance) {
                nearest = index;
                bestDistance = distance;
            }
        }
        normals[i] = normals[static_cast<size_t>(nearest)];
    }
    return normals;
}

cv::Vec3d tangentAt(const std::vector<SegmentNormalSample>& samples, size_t row)
{
    if (samples.size() < 2) {
        return {1.0, 0.0, 0.0};
    }
    cv::Vec3d tangent{0.0, 0.0, 0.0};
    if (row == 0) {
        tangent = samples[1].position - samples[0].position;
    } else if (row + 1 == samples.size()) {
        tangent = samples[row].position - samples[row - 1].position;
    } else {
        tangent = samples[row + 1].position - samples[row - 1].position;
    }
    tangent = normalizedOrZero(tangent);
    if (!validDirection(tangent)) {
        return {1.0, 0.0, 0.0};
    }
    return tangent;
}

cv::Vec3d sideDirection(const cv::Vec3d& normal, const cv::Vec3d& tangent)
{
    cv::Vec3d side = normalizedOrZero(normal.cross(tangent));
    if (validDirection(side)) {
        return side;
    }

    side = normalizedOrZero(axisFallbackLeastAlignedWith(tangent).cross(tangent));
    if (validDirection(side)) {
        return side;
    }
    return {0.0, 1.0, 0.0};
}

cv::Vec3d projectToTangentPlane(const cv::Vec3d& vector, const cv::Vec3d& tangent)
{
    const cv::Vec3d projected = vector - tangent * vector.dot(tangent);
    return normalizedOrZero(projected);
}

double clamped(double value, double minValue, double maxValue)
{
    return std::max(minValue, std::min(maxValue, value));
}

cv::Vec3d rotateAroundAxis(const cv::Vec3d& vector, const cv::Vec3d& axis, double angle)
{
    const cv::Vec3d unitAxis = normalizedOrZero(axis);
    if (!validDirection(unitAxis)) {
        return vector;
    }
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return vector * c + unitAxis.cross(vector) * s + unitAxis * (unitAxis.dot(vector) * (1.0 - c));
}

cv::Vec3d transportNormal(const cv::Vec3d& previousNormal,
                          const cv::Vec3d& previousTangent,
                          const cv::Vec3d& tangent)
{
    const cv::Vec3d axis = previousTangent.cross(tangent);
    const double sinAngle = norm(axis);
    const double cosAngle = clamped(previousTangent.dot(tangent), -1.0, 1.0);
    cv::Vec3d transported = previousNormal;
    if (sinAngle > kEpsilon) {
        transported = rotateAroundAxis(previousNormal, axis, std::atan2(sinAngle, cosAngle));
    }
    transported = projectToTangentPlane(transported, tangent);
    if (validDirection(transported)) {
        return transported;
    }

    const cv::Vec3d side = sideDirection(axisFallbackLeastAlignedWith(tangent), tangent);
    transported = normalizedOrZero(tangent.cross(side));
    if (validDirection(transported)) {
        return transported;
    }
    return {0.0, 0.0, 1.0};
}

double unwrapNear(double angle, double reference)
{
    constexpr double twoPi = 2.0 * 3.14159265358979323846;
    while (angle - reference > 3.14159265358979323846) {
        angle -= twoPi;
    }
    while (angle - reference < -3.14159265358979323846) {
        angle += twoPi;
    }
    return angle;
}

std::vector<double> smoothRollAngles(const std::vector<double>& targets)
{
    std::vector<double> angles = targets;
    if (angles.size() < 2) {
        return angles;
    }

    for (int iteration = 0; iteration < kRollSmoothIterations; ++iteration) {
        for (size_t i = 0; i < angles.size(); ++i) {
            double neighborSum = 0.0;
            double neighborCount = 0.0;
            if (i > 0) {
                neighborSum += angles[i - 1];
                neighborCount += 1.0;
            }
            if (i + 1 < angles.size()) {
                neighborSum += angles[i + 1];
                neighborCount += 1.0;
            }
            angles[i] = (targets[i] + kRollSmoothness * neighborSum) /
                        (1.0 + kRollSmoothness * neighborCount);
        }
    }
    return angles;
}

std::vector<cv::Vec3d> alignedTargetNormals(const std::vector<cv::Vec3d>& normals,
                                            const std::vector<cv::Vec3d>& tangents,
                                            const std::vector<cv::Vec3d>& baseNormals)
{
    std::vector<cv::Vec3d> targets(normals.size(), {0.0, 0.0, 0.0});
    cv::Vec3d previous{0.0, 0.0, 0.0};
    for (size_t row = 0; row < normals.size(); ++row) {
        cv::Vec3d target = projectToTangentPlane(normalizedOrZero(normals[row]), tangents[row]);
        if (!validDirection(target)) {
            targets[row] = previous;
            continue;
        }

        const cv::Vec3d reference = validDirection(previous) && row > 0
            ? transportNormal(previous, tangents[row - 1], tangents[row])
            : baseNormals[row];
        if (validDirection(reference) && target.dot(reference) < 0.0) {
            target *= -1.0;
        }
        targets[row] = target;
        previous = target;
    }
    return targets;
}

struct LineFrame {
    cv::Vec3d side;
    cv::Vec3d meshNormal;
};

std::vector<LineFrame> buildFrames(const std::vector<SegmentNormalSample>& samples,
                                   const std::vector<cv::Vec3d>& normals)
{
    std::vector<LineFrame> frames;
    frames.reserve(samples.size());

    std::vector<cv::Vec3d> tangents;
    std::vector<cv::Vec3d> baseNormals;
    tangents.reserve(samples.size());
    baseNormals.reserve(samples.size());

    for (size_t row = 0; row < samples.size(); ++row) {
        tangents.push_back(tangentAt(samples, row));
    }

    cv::Vec3d normal = projectToTangentPlane(normalizedOrZero(normals.front()), tangents.front());
    if (!validDirection(normal)) {
        normal = normalizedOrZero(tangents.front().cross(sideDirection(axisFallbackLeastAlignedWith(tangents.front()),
                                                                       tangents.front())));
    }
    if (!validDirection(normal)) {
        normal = {0.0, 0.0, 1.0};
    }
    baseNormals.push_back(normal);

    for (size_t row = 1; row < samples.size(); ++row) {
        baseNormals.push_back(transportNormal(baseNormals.back(), tangents[row - 1], tangents[row]));
    }

    const std::vector<cv::Vec3d> targetNormals = alignedTargetNormals(normals, tangents, baseNormals);
    std::vector<double> rollTargets(samples.size(), 0.0);
    for (size_t row = 0; row < samples.size(); ++row) {
        const cv::Vec3d targetNormal = targetNormals[row];
        if (!validDirection(targetNormal)) {
            rollTargets[row] = row > 0 ? rollTargets[row - 1] : 0.0;
            continue;
        }

        const cv::Vec3d binormal = normalizedOrZero(tangents[row].cross(baseNormals[row]));
        if (!validDirection(binormal)) {
            rollTargets[row] = row > 0 ? rollTargets[row - 1] : 0.0;
            continue;
        }

        double angle = std::atan2(targetNormal.dot(binormal), targetNormal.dot(baseNormals[row]));
        if (row > 0) {
            angle = unwrapNear(angle, rollTargets[row - 1]);
        }
        rollTargets[row] = angle;
    }

    const std::vector<double> rollAngles = smoothRollAngles(rollTargets);
    cv::Vec3d previousFrameNormal{0.0, 0.0, 0.0};
    for (size_t row = 0; row < samples.size(); ++row) {
        cv::Vec3d meshNormal = rotateAroundAxis(baseNormals[row], tangents[row], rollAngles[row]);
        meshNormal = projectToTangentPlane(meshNormal, tangents[row]);
        if (!validDirection(meshNormal)) {
            meshNormal = baseNormals[row];
        }
        if (validDirection(previousFrameNormal) && row > 0) {
            const cv::Vec3d reference = transportNormal(previousFrameNormal, tangents[row - 1], tangents[row]);
            if (validDirection(reference) && meshNormal.dot(reference) < 0.0) {
                meshNormal *= -1.0;
            }
        }
        cv::Vec3d side = normalizedOrZero(meshNormal.cross(tangents[row]));
        if (!validDirection(side)) {
            side = sideDirection(axisFallbackLeastAlignedWith(tangents[row]), tangents[row]);
            meshNormal = normalizedOrZero(tangents[row].cross(side));
        }
        previousFrameNormal = meshNormal;
        frames.push_back({side, meshNormal});
    }
    return frames;
}

std::shared_ptr<QuadSurface> buildRibbon(const std::vector<SegmentNormalSample>& samples,
                                         const std::vector<double>& offsets,
                                         const std::vector<LineFrame>& frames,
                                         bool useSide)
{
    cv::Mat_<cv::Vec3f> points(static_cast<int>(offsets.size()),
                               static_cast<int>(samples.size()));
    for (int col = 0; col < points.cols; ++col) {
        const auto& frame = frames[static_cast<size_t>(col)];
        const cv::Vec3d direction = useSide ? frame.side : frame.meshNormal;
        for (int row = 0; row < points.rows; ++row) {
            points(row, col) = toVec3f(samples[static_cast<size_t>(col)].position
                                     + direction * offsets[static_cast<size_t>(row)]);
        }
    }
    return std::make_shared<QuadSurface>(points, cv::Vec2f{1.0f, 1.0f});
}

std::vector<LineFrame> framesAtControlPoints(const std::vector<SegmentNormalSample>& controlSamples,
                                             const std::vector<SegmentNormalSample>& frameSamples,
                                             const std::vector<LineFrame>& frameSamplesFrames)
{
    std::vector<LineFrame> frames;
    frames.reserve(controlSamples.size());
    for (const auto& controlSample : controlSamples) {
        size_t bestIndex = 0;
        double bestDistance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < frameSamples.size(); ++i) {
            const double distance = norm(frameSamples[i].position - controlSample.position);
            if (distance < bestDistance) {
                bestIndex = i;
                bestDistance = distance;
            }
        }
        frames.push_back(frameSamplesFrames[bestIndex]);
    }
    return frames;
}

cv::Vec3d pointTangent(const LineModel& line, size_t index)
{
    if (line.points.size() < 2) {
        return {1.0, 0.0, 0.0};
    }
    cv::Vec3d tangent{0.0, 0.0, 0.0};
    if (index == 0) {
        tangent = line.points[1].position - line.points[0].position;
    } else if (index + 1 == line.points.size()) {
        tangent = line.points[index].position - line.points[index - 1].position;
    } else {
        tangent = line.points[index + 1].position - line.points[index - 1].position;
    }
    tangent = normalizedOrZero(tangent);
    if (validDirection(tangent)) {
        return tangent;
    }
    return {1.0, 0.0, 0.0};
}

} // namespace

LineViewSurfaces buildLineViewSurfaces(const LineModel& line, const LineViewConfig& config)
{
    const auto samples = controlPointSamples(line);
    if (samples.empty()) {
        throw std::invalid_argument("Cannot build line annotation views for an empty LineModel");
    }

    auto frameSamples = denseSamples(line);
    if (frameSamples.empty()) {
        frameSamples = samples;
    }
    const auto frameNormals = resolvedNormals(frameSamples);
    const auto denseFrames = buildFrames(frameSamples, frameNormals);
    const auto frames = framesAtControlPoints(samples, frameSamples, denseFrames);
    const double surfaceHalfWidth = resolvedHalfExtent(config.surfaceHalfWidth,
                                                       samples,
                                                       config.crossSamples);
    const double sideSliceHalfDepth = resolvedHalfExtent(config.sideSliceHalfDepth,
                                                        samples,
                                                        config.crossSamples);

    LineViewSurfaces surfaces;
    surfaces.lineSurface = buildRibbon(samples,
                                       crossOffsets(surfaceHalfWidth, config.crossSamples),
                                       frames,
                                       true);
    surfaces.lineSideSlice = buildRibbon(samples,
                                         crossOffsets(sideSliceHalfDepth, config.crossSamples),
                                         frames,
                                         false);

    surfaces.lineZSlices.reserve(line.points.size());
    for (size_t i = 0; i < line.points.size(); ++i) {
        surfaces.lineZSlices.push_back(std::make_shared<PlaneSurface>(
            toVec3f(line.points[i].position),
            toVec3f(pointTangent(line, i))));
    }
    return surfaces;
}

} // namespace vc::lasagna
