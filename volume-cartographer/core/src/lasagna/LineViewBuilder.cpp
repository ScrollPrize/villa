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

std::shared_ptr<QuadSurface> buildRibbon(const std::vector<SegmentNormalSample>& samples,
                                         const std::vector<cv::Vec3d>& normals,
                                         const std::vector<double>& offsets,
                                         bool useSide)
{
    cv::Mat_<cv::Vec3f> points(static_cast<int>(samples.size()),
                               static_cast<int>(offsets.size()));
    cv::Vec3d previousDirection{0.0, 0.0, 0.0};
    for (int row = 0; row < points.rows; ++row) {
        const cv::Vec3d tangent = tangentAt(samples, static_cast<size_t>(row));
        const cv::Vec3d normal = normalizedOrZero(normals[static_cast<size_t>(row)]);
        cv::Vec3d direction = useSide ? sideDirection(normal, tangent) : normal;
        if (validDirection(previousDirection) && direction.dot(previousDirection) < 0.0) {
            direction *= -1.0;
        }
        previousDirection = direction;
        for (int col = 0; col < points.cols; ++col) {
            points(row, col) = toVec3f(samples[static_cast<size_t>(row)].position
                                     + direction * offsets[static_cast<size_t>(col)]);
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
    LineViewSurfaces surfaces;
    surfaces.lineSurface = buildRibbon(samples,
                                       normals,
                                       crossOffsets(config.surfaceHalfWidth, config.crossSamples),
                                       true);
    surfaces.lineSideSlice = buildRibbon(samples,
                                         normals,
                                         crossOffsets(config.sideSliceHalfDepth, config.crossSamples),
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
