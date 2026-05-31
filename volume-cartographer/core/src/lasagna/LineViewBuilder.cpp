#include "vc/lasagna/LineViewBuilder.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;

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

struct LineFrame {
    cv::Vec3d side;
    cv::Vec3d meshNormal;
};

std::vector<LineFrame> buildFrames(const std::vector<SegmentNormalSample>& samples,
                                   const std::vector<cv::Vec3d>& normals)
{
    std::vector<LineFrame> frames;
    frames.reserve(samples.size());

    cv::Vec3d previousSide{0.0, 0.0, 0.0};
    cv::Vec3d previousNormal{0.0, 0.0, 0.0};
    for (size_t row = 0; row < samples.size(); ++row) {
        const cv::Vec3d tangent = tangentAt(samples, row);
        const cv::Vec3d sampledNormal = normalizedOrZero(normals[row]);

        cv::Vec3d side = sideDirection(sampledNormal, tangent);
        if (validDirection(previousSide) && side.dot(previousSide) < 0.0) {
            side *= -1.0;
        }

        cv::Vec3d meshNormal = normalizedOrZero(tangent.cross(side));
        if (!validDirection(meshNormal)) {
            meshNormal = sampledNormal;
        }
        if (!validDirection(meshNormal)) {
            meshNormal = normalizedOrZero(tangent.cross(sideDirection(axisFallbackLeastAlignedWith(tangent),
                                                                       tangent)));
        }
        if (!validDirection(meshNormal)) {
            meshNormal = {0.0, 0.0, 1.0};
        }
        if (validDirection(sampledNormal) && meshNormal.dot(sampledNormal) < 0.0) {
            meshNormal *= -1.0;
            side *= -1.0;
        }
        if (validDirection(previousNormal) && meshNormal.dot(previousNormal) < 0.0) {
            meshNormal *= -1.0;
            side *= -1.0;
        }

        previousSide = side;
        previousNormal = meshNormal;
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

    const auto normals = resolvedNormals(samples);
    const auto frames = buildFrames(samples, normals);
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
