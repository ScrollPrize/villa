#include "vc/lasagna/LineViewBuilder.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kRollSmoothness = 4.0;
constexpr int kRollSmoothIterations = 80;
constexpr double kMaxFrameRollDelta = 0.78539816339744830962;

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

double unwrapAxisNear(double angle, double reference)
{
    constexpr double pi = 3.14159265358979323846;
    while (angle - reference > 0.5 * pi) {
        angle -= pi;
    }
    while (angle - reference < -0.5 * pi) {
        angle += pi;
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

cv::Vec3d fallbackMeshNormalForTangent(const cv::Vec3d& tangent)
{
    const cv::Vec3d side = sideDirection(axisFallbackLeastAlignedWith(tangent), tangent);
    cv::Vec3d normal = normalizedOrZero(tangent.cross(side));
    if (validDirection(normal)) {
        return normal;
    }
    normal = projectToTangentPlane({0.0, 0.0, 1.0}, tangent);
    if (validDirection(normal)) {
        return normal;
    }
    return {0.0, 1.0, 0.0};
}

cv::Vec3d clampedFrameNormal(const cv::Vec3d& reference,
                             const cv::Vec3d& target,
                             const cv::Vec3d& tangent)
{
    if (!validDirection(reference)) {
        return target;
    }
    if (!validDirection(target)) {
        return reference;
    }

    const cv::Vec3d binormal = normalizedOrZero(tangent.cross(reference));
    if (!validDirection(binormal)) {
        return target;
    }

    const double angle = std::atan2(target.dot(binormal), target.dot(reference));
    const double clampedAngle = clamped(angle, -kMaxFrameRollDelta, kMaxFrameRollDelta);
    cv::Vec3d normal = rotateAroundAxis(reference, tangent, clampedAngle);
    normal = projectToTangentPlane(normal, tangent);
    return validDirection(normal) ? normal : target;
}

LineFrame frameFromMeshNormal(cv::Vec3d meshNormal, const cv::Vec3d& tangent)
{
    meshNormal = projectToTangentPlane(meshNormal, tangent);
    if (!validDirection(meshNormal)) {
        meshNormal = fallbackMeshNormalForTangent(tangent);
    }
    cv::Vec3d side = normalizedOrZero(meshNormal.cross(tangent));
    if (!validDirection(side)) {
        side = sideDirection(axisFallbackLeastAlignedWith(tangent), tangent);
        meshNormal = normalizedOrZero(tangent.cross(side));
    }
    return {side, meshNormal};
}

std::vector<LineFrame> buildFrames(const std::vector<SegmentNormalSample>& samples,
                                   const std::vector<cv::Vec3d>& normals)
{
    std::vector<LineFrame> frames(samples.size());
    if (samples.empty()) {
        return frames;
    }

    std::vector<cv::Vec3d> tangents;
    tangents.reserve(samples.size());
    for (size_t row = 0; row < samples.size(); ++row) {
        tangents.push_back(tangentAt(samples, row));
    }

    const size_t anchor = samples.size() / 2;
    std::vector<cv::Vec3d> baseNormals(samples.size(), {0.0, 0.0, 0.0});
    baseNormals[anchor] = projectToTangentPlane(normalizedOrZero(normals[anchor]), tangents[anchor]);
    if (!validDirection(baseNormals[anchor])) {
        baseNormals[anchor] = fallbackMeshNormalForTangent(tangents[anchor]);
    }

    for (size_t row = anchor + 1; row < samples.size(); ++row) {
        baseNormals[row] = transportNormal(baseNormals[row - 1], tangents[row - 1], tangents[row]);
    }
    for (size_t row = anchor; row > 0; --row) {
        baseNormals[row - 1] = transportNormal(baseNormals[row], tangents[row], tangents[row - 1]);
    }

    auto targetAxisAngle = [&](size_t row) -> std::optional<double> {
        const cv::Vec3d axis = projectToTangentPlane(normalizedOrZero(normals[row]), tangents[row]);
        if (!validDirection(axis)) {
            return std::nullopt;
        }
        const cv::Vec3d binormal = normalizedOrZero(tangents[row].cross(baseNormals[row]));
        if (!validDirection(binormal)) {
            return std::nullopt;
        }
        return std::atan2(axis.dot(binormal), axis.dot(baseNormals[row]));
    };

    std::vector<double> rollTargets(samples.size(), 0.0);
    if (const auto angle = targetAxisAngle(anchor)) {
        rollTargets[anchor] = unwrapAxisNear(*angle, 0.0);
    }
    for (size_t row = anchor + 1; row < samples.size(); ++row) {
        if (const auto angle = targetAxisAngle(row)) {
            rollTargets[row] = unwrapAxisNear(*angle, rollTargets[row - 1]);
        } else {
            rollTargets[row] = rollTargets[row - 1];
        }
    }
    for (size_t row = anchor; row > 0; --row) {
        if (const auto angle = targetAxisAngle(row - 1)) {
            rollTargets[row - 1] = unwrapAxisNear(*angle, rollTargets[row]);
        } else {
            rollTargets[row - 1] = rollTargets[row];
        }
    }

    const std::vector<double> rollAngles = smoothRollAngles(rollTargets);
    frames[anchor] = frameFromMeshNormal(rotateAroundAxis(baseNormals[anchor],
                                                          tangents[anchor],
                                                          rollAngles[anchor]),
                                         tangents[anchor]);
    for (size_t row = anchor + 1; row < samples.size(); ++row) {
        cv::Vec3d meshNormal = rotateAroundAxis(baseNormals[row], tangents[row], rollAngles[row]);
        const cv::Vec3d transported = transportNormal(frames[row - 1].meshNormal,
                                                      tangents[row - 1],
                                                      tangents[row]);
        if (validDirection(transported) && meshNormal.dot(transported) < 0.0) {
            meshNormal *= -1.0;
        }
        frames[row] = frameFromMeshNormal(meshNormal, tangents[row]);
    }
    for (size_t row = anchor; row > 0; --row) {
        cv::Vec3d meshNormal = rotateAroundAxis(baseNormals[row - 1],
                                                tangents[row - 1],
                                                rollAngles[row - 1]);
        const cv::Vec3d transported = transportNormal(frames[row].meshNormal,
                                                      tangents[row],
                                                      tangents[row - 1]);
        if (validDirection(transported) && meshNormal.dot(transported) < 0.0) {
            meshNormal *= -1.0;
        }
        frames[row - 1] = frameFromMeshNormal(meshNormal, tangents[row - 1]);
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

struct LineViewFrameData {
    std::vector<SegmentNormalSample> samples;
    std::vector<cv::Vec3d> normals;
    std::vector<cv::Vec3d> tangents;
    std::vector<LineFrame> frames;
};

LineViewFrameData buildControlFrameData(const LineModel& line)
{
    LineViewFrameData data;
    data.samples = controlPointSamples(line);
    if (data.samples.empty()) {
        return data;
    }

    data.normals = resolvedNormals(data.samples);
    data.frames = buildFrames(data.samples, data.normals);
    data.tangents.reserve(data.samples.size());
    for (size_t row = 0; row < data.samples.size(); ++row) {
        data.tangents.push_back(tangentAt(data.samples, row));
    }
    return data;
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
    const auto frameData = buildControlFrameData(line);
    const auto& samples = frameData.samples;
    const auto& frames = frameData.frames;
    if (samples.empty()) {
        throw std::invalid_argument("Cannot build line annotation views for an empty LineModel");
    }

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
    surfaces.lineUpVectors.reserve(line.points.size());
    for (size_t i = 0; i < line.points.size(); ++i) {
        const cv::Vec3f origin = toVec3f(line.points[i].position);
        const cv::Vec3f tangent = toVec3f(pointTangent(line, i));
        const cv::Vec3f up = toVec3f(frames[i].meshNormal);
        auto plane = std::make_shared<PlaneSurface>();
        plane->setFromNormalAndUp(origin, tangent, up);
        surfaces.lineZSlices.push_back(std::move(plane));
        surfaces.lineUpVectors.push_back(up);
    }
    return surfaces;
}

LineViewFrameDiagnostics diagnoseLineViewFrames(const LineModel& line, const LineViewConfig& /*config*/)
{
    const auto frameData = buildControlFrameData(line);
    LineViewFrameDiagnostics diagnostics;
    diagnostics.frameCount = frameData.frames.size();
    if (frameData.frames.size() < 2) {
        return diagnostics;
    }

    for (size_t i = 1; i < frameData.frames.size(); ++i) {
        const auto& prevFrame = frameData.frames[i - 1];
        const auto& frame = frameData.frames[i];
        const cv::Vec3d tangent = frameData.tangents[i];
        const cv::Vec3d transportedNormal = transportNormal(prevFrame.meshNormal,
                                                            frameData.tangents[i - 1],
                                                            tangent);
        const cv::Vec3d transportedSide = transportNormal(prevFrame.side,
                                                          frameData.tangents[i - 1],
                                                          tangent);
        const cv::Vec3d prevSampledNormal = projectToTangentPlane(frameData.normals[i - 1],
                                                                  frameData.tangents[i - 1]);
        const cv::Vec3d transportedSampledNormal = transportNormal(prevSampledNormal,
                                                                   frameData.tangents[i - 1],
                                                                   tangent);
        const cv::Vec3d sampledNormal = projectToTangentPlane(frameData.normals[i], tangent);

        const double normalDot = validDirection(transportedNormal)
            ? frame.meshNormal.dot(transportedNormal)
            : 1.0;
        const double sideDot = validDirection(transportedSide)
            ? frame.side.dot(transportedSide)
            : 1.0;
        const double sampledAxisDot = validDirection(transportedSampledNormal) && validDirection(sampledNormal)
            ? std::abs(sampledNormal.dot(transportedSampledNormal))
            : 1.0;
        const double meshToSampledAxisDot = validDirection(sampledNormal)
            ? std::abs(frame.meshNormal.dot(sampledNormal))
            : 1.0;

        double rollDelta = 0.0;
        if (validDirection(transportedNormal)) {
            const cv::Vec3d binormal = normalizedOrZero(tangent.cross(transportedNormal));
            if (validDirection(binormal)) {
                rollDelta = std::atan2(frame.meshNormal.dot(binormal),
                                       frame.meshNormal.dot(transportedNormal));
            }
        }

        diagnostics.maxAbsRollDeltaRadians = std::max(diagnostics.maxAbsRollDeltaRadians,
                                                      std::abs(rollDelta));
        diagnostics.minNormalContinuityDot = std::min(diagnostics.minNormalContinuityDot,
                                                      normalDot);
        diagnostics.minSideContinuityDot = std::min(diagnostics.minSideContinuityDot,
                                                    sideDot);
        diagnostics.minSampledAxisContinuityDot = std::min(diagnostics.minSampledAxisContinuityDot,
                                                           sampledAxisDot);
        diagnostics.minMeshToSampledAxisDot = std::min(diagnostics.minMeshToSampledAxisDot,
                                                       meshToSampledAxisDot);

        std::string reason;
        if (normalDot < 0.0 || sideDot < 0.0) {
            reason = "generated_frame_flip";
        } else if (std::abs(rollDelta) > 1.57079632679489661923) {
            reason = "large_generated_roll_jump";
        }

        if (!reason.empty()) {
            diagnostics.issues.push_back({i,
                                          rollDelta,
                                          normalDot,
                                          sideDot,
                                          sampledAxisDot,
                                          meshToSampledAxisDot,
                                          std::move(reason)});
        }
    }
    return diagnostics;
}

} // namespace vc::lasagna
