#include "vc/fiber_tracer/FiberReplay.hpp"

#include "vc/core/util/AtomicFile.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

double segmentAabbDistanceSquared(
    const cv::Vec3d& start,
    const cv::Vec3d& end,
    const cv::Vec3d& low,
    const cv::Vec3d& high)
{
    const cv::Vec3d delta = end - start;
    std::vector<double> breaks{0.0, 1.0};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(delta[axis]) <= kEpsilon)
            continue;
        for (const double bound : {low[axis], high[axis]}) {
            const double t = (bound - start[axis]) / delta[axis];
            if (t > 0.0 && t < 1.0)
                breaks.push_back(t);
        }
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    double best = std::numeric_limits<double>::infinity();
    const auto evaluate = [&](double t) {
        const cv::Vec3d point = start + delta * t;
        double squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double outside = point[axis] < low[axis]
                ? low[axis] - point[axis]
                : point[axis] > high[axis]
                    ? point[axis] - high[axis]
                    : 0.0;
            squared += outside * outside;
        }
        best = std::min(best, squared);
    };
    for (size_t interval = 0; interval + 1 < breaks.size(); ++interval) {
        const double begin = breaks[interval];
        const double finish = breaks[interval + 1];
        evaluate(begin);
        evaluate(finish);
        const double middle = 0.5 * (begin + finish);
        double quadratic = 0.0;
        double linear = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double point = start[axis] + delta[axis] * middle;
            double offset = 0.0;
            if (point < low[axis])
                offset = start[axis] - low[axis];
            else if (point > high[axis])
                offset = start[axis] - high[axis];
            else
                continue;
            quadratic += delta[axis] * delta[axis];
            linear += delta[axis] * offset;
        }
        if (quadratic > kEpsilon)
            evaluate(std::clamp(-linear / quadratic, begin, finish));
    }
    return best;
}

double polylineAabbDistanceSquared(
    const std::vector<cv::Vec3d>& points,
    const cv::Vec3d& low,
    const cv::Vec3d& high)
{
    double best = std::numeric_limits<double>::infinity();
    if (points.size() == 1)
        return segmentAabbDistanceSquared(points.front(), points.front(), low, high);
    for (size_t index = 1; index < points.size(); ++index) {
        best = std::min(best, segmentAabbDistanceSquared(
            points[index - 1], points[index], low, high));
    }
    return best;
}

nlohmann::json pointJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

nlohmann::json pointsJson(const std::vector<cv::Vec3d>& points)
{
    nlohmann::json result = nlohmann::json::array();
    for (const auto& point : points)
        result.push_back(pointJson(point));
    return result;
}

std::string lineObj(
    const char* header,
    const std::vector<cv::Vec3d>& points,
    bool pointRecord = false)
{
    std::ostringstream output;
    output << header << '\n' << std::setprecision(17);
    for (const auto& point : points)
        output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    if (pointRecord) {
        if (points.size() != 1)
            throw std::invalid_argument("replay point OBJ requires exactly one point");
        output << "p 1\n";
    } else if (!points.empty()) {
        output << "l";
        if (points.size() == 1) {
            output << " 1 1";
        } else {
            for (size_t index = 0; index < points.size(); ++index)
                output << ' ' << index + 1;
        }
        output << '\n';
    }
    return output.str();
}

std::string hashString(const std::string& value)
{
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read replay artifact: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string artifactHash(const std::filesystem::path& path)
{
    return hashString(readFile(path));
}

} // namespace

bool FiberReplayTube::containsBasePoint(const cv::Vec3d& point) const
{
    return distanceToBasePoint(point) <= radiusBaseVoxels + kEpsilon;
}

double FiberReplayTube::distanceToBasePoint(const cv::Vec3d& point) const
{
    return distanceToPolylineArc(reference, point, beginArcBase, endArcBase);
}

bool FiberReplayTube::containsPredictionPoint(
    const cv::Vec3d& pointPredictionXYZ,
    double predictionToBaseScale) const
{
    return containsBasePoint(pointPredictionXYZ * predictionToBaseScale);
}

FiberReplayTube makeFiberReplayTube(
    const std::vector<cv::Vec3d>& referenceLineBase,
    double centerArcBase,
    double alongBaseVoxels,
    double radiusBaseVoxels,
    const FiberPredictionGridInfo& grid,
    int anchorCellSizePredictionVoxels)
{
    if (!(alongBaseVoxels >= 0.0) || !std::isfinite(alongBaseVoxels) ||
        !(radiusBaseVoxels > 0.0) || !std::isfinite(radiusBaseVoxels)) {
        throw std::invalid_argument("fiber replay tube distances are invalid");
    }
    if (!(grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(grid.predictionToBaseScale) ||
        anchorCellSizePredictionVoxels < 1) {
        throw std::invalid_argument("fiber replay tube grid is invalid");
    }
    FiberReplayTube tube;
    tube.reference = makePolylineArcGeometry(referenceLineBase);
    tube.beginArcBase = std::max(0.0, centerArcBase - alongBaseVoxels);
    tube.endArcBase = std::min(tube.reference.length(), centerArcBase + alongBaseVoxels);
    tube.radiusBaseVoxels = radiusBaseVoxels;
    tube.referenceIntervalBase = slicePolylineArc(
        tube.reference, tube.beginArcBase, tube.endArcBase);

    cv::Vec3d low = tube.referenceIntervalBase.front();
    cv::Vec3d high = low;
    for (const auto& point : tube.referenceIntervalBase) {
        for (int axis = 0; axis < 3; ++axis) {
            low[axis] = std::min(low[axis], point[axis]);
            high[axis] = std::max(high[axis], point[axis]);
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double gridHigh = static_cast<double>(grid.shapeZYX[2 - axis]) *
            grid.predictionToBaseScale;
        const double cropLow = std::clamp(
            std::floor(low[axis] - radiusBaseVoxels), 0.0, gridHigh);
        const double cropHigh = std::clamp(
            std::ceil(high[axis] + radiusBaseVoxels), 0.0, gridHigh);
        tube.volumeCropBaseXYZWHD[axis] = static_cast<size_t>(cropLow);
        tube.volumeCropBaseXYZWHD[axis + 3] = static_cast<size_t>(cropHigh - cropLow);
    }

    const size_t cellSize = static_cast<size_t>(anchorCellSizePredictionVoxels);
    const std::array<size_t, 3> cellShape{
        (grid.shapeZYX[0] + cellSize - 1) / cellSize,
        (grid.shapeZYX[1] + cellSize - 1) / cellSize,
        (grid.shapeZYX[2] + cellSize - 1) / cellSize,
    };
    const double scale = grid.predictionToBaseScale;
    const double radiusSquared = radiusBaseVoxels * radiusBaseVoxels;
    std::array<size_t, 3> cellBeginZYX{};
    std::array<size_t, 3> cellEndZYX{};
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t xyz = 2 - axis;
        const double cropBeginPrediction =
            static_cast<double>(tube.volumeCropBaseXYZWHD[xyz]) / scale;
        const double cropEndPrediction =
            static_cast<double>(tube.volumeCropBaseXYZWHD[xyz] +
                                tube.volumeCropBaseXYZWHD[xyz + 3]) /
            scale;
        cellBeginZYX[axis] = std::min(
            cellShape[axis],
            static_cast<size_t>(std::max(0.0, std::floor(cropBeginPrediction /
                static_cast<double>(cellSize)))));
        cellEndZYX[axis] = std::min(
            cellShape[axis],
            static_cast<size_t>(std::max(0.0, std::ceil(cropEndPrediction /
                static_cast<double>(cellSize)))));
    }
    for (size_t cz = cellBeginZYX[0]; cz < cellEndZYX[0]; ++cz) {
        for (size_t cy = cellBeginZYX[1]; cy < cellEndZYX[1]; ++cy) {
            for (size_t cx = cellBeginZYX[2]; cx < cellEndZYX[2]; ++cx) {
                const std::array<size_t, 3> begin{cz * cellSize, cy * cellSize, cx * cellSize};
                const std::array<size_t, 3> end{
                    std::min(grid.shapeZYX[0], begin[0] + cellSize),
                    std::min(grid.shapeZYX[1], begin[1] + cellSize),
                    std::min(grid.shapeZYX[2], begin[2] + cellSize),
                };
                const cv::Vec3d cellLow{
                    (static_cast<double>(begin[2]) - 0.5) * scale,
                    (static_cast<double>(begin[1]) - 0.5) * scale,
                    (static_cast<double>(begin[0]) - 0.5) * scale,
                };
                const cv::Vec3d cellHigh{
                    (static_cast<double>(end[2]) - 0.5) * scale,
                    (static_cast<double>(end[1]) - 0.5) * scale,
                    (static_cast<double>(end[0]) - 0.5) * scale,
                };
                if (polylineAabbDistanceSquared(
                        tube.referenceIntervalBase, cellLow, cellHigh) <=
                    radiusSquared + kEpsilon) {
                    tube.cellsZYX.push_back({cz, cy, cx});
                }
            }
        }
    }
    if (tube.cellsZYX.empty())
        throw std::runtime_error("fiber replay tube selects no prediction cells");
    return tube;
}

nlohmann::json writeFiberReplayBundle(
    const std::filesystem::path& outputDirectory,
    const FiberReplayBundleInput& input)
{
    const bool failed = input.replay.status == FiberReplayStatus::FailureWithPostroll ||
        input.replay.status == FiberReplayStatus::FailureTruncated;
    if (input.referenceGeometryBase.empty() || input.replay.tracePointsBase.empty() ||
        input.replay.cumulativeLosses.size() != input.replay.tracePointsBase.size()) {
        throw std::invalid_argument("fiber replay bundle geometry and losses are incomplete");
    }
    if (failed != (input.replay.failureTracePointIndex.has_value() &&
                   input.replay.failureReferenceArcBase.has_value())) {
        throw std::invalid_argument("fiber replay failure metadata disagrees with status");
    }
    if (input.replay.failureTracePointIndex.has_value() &&
        *input.replay.failureTracePointIndex >= input.replay.tracePointsBase.size()) {
        throw std::invalid_argument("fiber replay failure index is outside the trace");
    }
    if (input.replay.completedPostrollSteps < 0 ||
        input.replay.completedPostrollSteps > input.replay.requestedPostrollSteps ||
        input.replay.requestedPostrollSteps < 0) {
        throw std::invalid_argument("fiber replay postroll metadata is invalid");
    }
    if (failed != input.tube.has_value())
        throw std::invalid_argument("fiber replay failure and tube presence disagree");
    if (failed != (input.anchors.has_value() && input.anchorArtifact.has_value() &&
                   input.paths.has_value() && input.pathArtifact.has_value())) {
        throw std::invalid_argument("fiber replay extraction artifacts disagree with status");
    }
    std::filesystem::create_directories(outputDirectory / "runs");
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path staging = outputDirectory / "runs" /
        (".staging-" + std::to_string(stamp));
    std::filesystem::create_directories(staging / "replay");

    vc::core::util::atomicWriteString(
        staging / "replay/reference.obj",
        lineObj("# vc_fiber_replay_reference version 1", input.referenceGeometryBase));
    vc::core::util::atomicWriteString(
        staging / "replay/trace.obj",
        lineObj("# vc_fiber_replay_trace version 1", input.replay.tracePointsBase));
    if (failed) {
        const cv::Vec3d failure = input.replay.tracePointsBase.at(
            *input.replay.failureTracePointIndex);
        vc::core::util::atomicWriteString(
            staging / "replay/failure.obj",
            lineObj("# vc_fiber_replay_failure version 1", {failure}, true));
        writeFiberAnchorArtifacts(
            staging / "anchors", *input.anchors, *input.anchorArtifact);
        writeFiberletPathArtifacts(
            staging / "paths", *input.paths, *input.pathArtifact);
    }

    std::vector<std::filesystem::path> relativeArtifacts{
        "replay/reference.obj", "replay/trace.obj"};
    if (failed) {
        relativeArtifacts.insert(relativeArtifacts.end(), {
            "replay/failure.obj",
            "anchors/anchors.json",
            "anchors/anchors.obj",
            "anchors/anchors_0.obj",
            "anchors/anchors_1.obj",
            "anchors/anchor_cells.obj",
            "anchors/stages/initialized.json",
            "anchors/stages/refined.json",
            "anchors/stages/support.json",
            "anchors/stages/selection.json",
            "anchors/stages/nms.json",
            "paths/fiberlets.json",
            "paths/fiberlets.obj",
        });
    }
    std::string generationMaterial;
    for (const auto& relative : relativeArtifacts) {
        generationMaterial += relative.generic_string();
        generationMaterial += '\0';
        generationMaterial += readFile(staging / relative);
        generationMaterial += '\0';
    }
    const std::string generationHash = hashString(generationMaterial);
    const std::string generationName = generationHash.substr(generationHash.find(':') + 1);
    const std::filesystem::path finalGeneration = outputDirectory / "runs" / generationName;
    if (std::filesystem::exists(finalGeneration))
        std::filesystem::remove_all(staging);
    else
        std::filesystem::rename(staging, finalGeneration);

    nlohmann::json matches = nlohmann::json::array();
    for (const auto& match : input.replay.matches) {
        matches.push_back({
            {"trace_point_index", match.tracePointIndex},
            {"predicted_reference_arc_base", match.predictedReferenceArcBase},
            {"matched_reference_arc_base", match.matchedReferenceArcBase},
            {"matched_reference_point_base_xyz", pointJson(match.matchedReferencePointBase)},
            {"search_begin_arc_base", match.searchBeginArcBase},
            {"search_end_arc_base", match.searchEndArcBase},
            {"error_base_voxels", match.errorBaseVoxels},
            {"error_ratio", match.errorRatio},
        });
    }
    nlohmann::json bundle = {
        {"format", "vc_fiber_replay"},
        {"version", 1},
        {"coordinates", {{"position_order", "XYZ"}, {"position_space", "base_volume"}, {"distance_unit", "base_voxels"}}},
        {"sources", input.sources},
        {"bindings", {{"trace", input.traceBinding}, {"prediction", input.predictionBinding}}},
        {"trace_config", {{"requested", input.requestedTraceConfig}, {"effective", input.effectiveTraceConfig}}},
        {"status", fiberReplayStatusName(input.replay.status)},
        {"termination_reason", input.replay.terminationReason},
        {"reference_points_base_xyz", pointsJson(input.referenceGeometryBase)},
        {"trace_points_base_xyz", pointsJson(input.replay.tracePointsBase)},
        {"trace_cumulative_losses", input.replay.cumulativeLosses},
        {"matching", {
            {"failure_threshold_base_voxels", input.request.errorThresholdBaseVoxels},
            {"refine_steps", input.request.matchRefineSteps},
            {"records", std::move(matches)},
        }},
        {"postroll", {
            {"requested_steps", input.replay.requestedPostrollSteps},
            {"completed_steps", input.replay.completedPostrollSteps},
            {"maximum_trace_steps", input.replay.maximumTraceSteps},
        }},
        {"failure_trace_point_index", nullptr},
        {"failure_reference_arc_base", nullptr},
        {"tube", nullptr},
        {"volume_crop_base_xyzwhd", nullptr},
        {"artifacts", nlohmann::json::object()},
    };
    if (input.replay.failureTracePointIndex.has_value()) {
        bundle["failure_trace_point_index"] =
            *input.replay.failureTracePointIndex;
    }
    if (input.replay.failureReferenceArcBase.has_value()) {
        bundle["failure_reference_arc_base"] =
            *input.replay.failureReferenceArcBase;
    }
    if (input.tube.has_value()) {
        bundle["tube"] = {
            {"begin_arc_base", input.tube->beginArcBase},
            {"end_arc_base", input.tube->endArcBase},
            {"radius_base_voxels", input.tube->radiusBaseVoxels},
            {"reference_points_base_xyz", pointsJson(input.tube->referenceIntervalBase)},
            {"cells_zyx", input.tube->cellsZYX},
        };
        bundle["volume_crop_base_xyzwhd"] = input.tube->volumeCropBaseXYZWHD;
    }
    const std::filesystem::path generationRelative =
        std::filesystem::path("runs") / generationName;
    for (const auto& relative : relativeArtifacts) {
        const std::string key = relative.generic_string();
        bundle["artifacts"][key] = {
            {"path", (generationRelative / relative).generic_string()},
            {"content_hash", artifactHash(finalGeneration / relative)},
        };
    }
    vc::core::util::atomicWriteString(
        outputDirectory / "fiber_replay.json", bundle.dump(2) + "\n");
    return bundle;
}

} // namespace vc::fiber_tracer
