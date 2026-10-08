#include "vc/fiber_tracer/FiberTraceSeed.hpp"

#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kLengthEpsilon = 1.0e-9;

bool finitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) &&
        std::isfinite(point[2]);
}

}  // namespace

FiberTraceSeedGeometryReport measureFiberTraceSeedGeometry(
    const std::vector<FiberletCropTraceLine>& traces,
    const cv::Vec3d& cropMinimumBaseXYZ,
    const cv::Vec3d& cropMaximumBaseXYZ)
{
    if (!finitePoint(cropMinimumBaseXYZ) ||
        !finitePoint(cropMaximumBaseXYZ)) {
        throw std::invalid_argument("Fiber trace crop bounds must be finite");
    }
    const cv::Vec3d extent = cropMaximumBaseXYZ - cropMinimumBaseXYZ;
    if (!(extent[0] > 0.0) || !(extent[1] > 0.0) || !(extent[2] > 0.0)) {
        throw std::invalid_argument(
            "Fiber trace crop bounds must have positive extent");
    }
    const cv::Vec3d center =
        0.5 * (cropMinimumBaseXYZ + cropMaximumBaseXYZ);

    FiberTraceSeedGeometryReport report;
    report.primaryMinimumArcLengthBaseVoxels =
        0.5 * std::min({extent[0], extent[1], extent[2]});
    report.traces.reserve(traces.size());
    for (const auto& trace : traces) {
        FiberTraceSeedGeometry current;
        if (trace.pointsBaseXYZ.size() < 2) {
            report.traces.push_back(current);
            continue;
        }
        for (std::size_t index = 1;
             index < trace.pointsBaseXYZ.size();
             ++index) {
            const double length = cv::norm(
                trace.pointsBaseXYZ[index] -
                trace.pointsBaseXYZ[index - 1]);
            if (!std::isfinite(length)) {
                throw std::invalid_argument(
                    "Fiber trace contains non-finite geometry");
            }
            current.arcLengthBaseVoxels += length;
        }
        if (!(current.arcLengthBaseVoxels > kLengthEpsilon)) {
            report.traces.push_back(current);
            continue;
        }
        const double chord = cv::norm(
            trace.pointsBaseXYZ.back() - trace.pointsBaseXYZ.front());
        if (!std::isfinite(chord)) {
            throw std::invalid_argument(
                "Fiber trace contains non-finite geometry");
        }
        current.straightness = chord / current.arcLengthBaseVoxels;
        const auto arc = makePolylineArcGeometry(trace.pointsBaseXYZ);
        current.centerDistanceBaseVoxels = distanceToPolylineArc(
            arc, center, 0.0, arc.length());
        if (!std::isfinite(current.centerDistanceBaseVoxels)) {
            throw std::invalid_argument(
                "Fiber trace has invalid crop-center distance");
        }
        current.valid = true;
        report.traces.push_back(current);
    }
    return report;
}

std::optional<std::size_t> selectCentralStraightFiberTrace(
    const FiberTraceSeedGeometryReport& geometry,
    std::span<const unsigned char> eligible,
    bool requirePrimaryLength)
{
    if (!eligible.empty() && eligible.size() != geometry.traces.size()) {
        throw std::invalid_argument(
            "Fiber trace seed eligibility size does not match traces");
    }
    std::optional<std::size_t> best;
    for (std::size_t trace = 0; trace < geometry.traces.size(); ++trace) {
        if ((!eligible.empty() && eligible[trace] == 0) ||
            !geometry.traces[trace].valid) {
            continue;
        }
        const auto& current = geometry.traces[trace];
        if (requirePrimaryLength &&
            !(current.arcLengthBaseVoxels >
              geometry.primaryMinimumArcLengthBaseVoxels)) {
            continue;
        }
        if (!best) {
            best = trace;
            continue;
        }
        const auto& selected = geometry.traces[*best];
        if (current.straightness > selected.straightness ||
            (current.straightness == selected.straightness &&
             (current.centerDistanceBaseVoxels <
                  selected.centerDistanceBaseVoxels ||
              (current.centerDistanceBaseVoxels ==
                   selected.centerDistanceBaseVoxels &&
               (current.arcLengthBaseVoxels >
                    selected.arcLengthBaseVoxels ||
                (current.arcLengthBaseVoxels ==
                     selected.arcLengthBaseVoxels &&
                 trace < *best)))))) {
            best = trace;
        }
    }
    return best;
}

}  // namespace vc::fiber_tracer
