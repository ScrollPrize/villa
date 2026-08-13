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

bool nearlyEqual(double left, double right)
{
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <= 1.0e-10 * scale;
}

bool samePoints(
    const std::vector<cv::Vec3d>& left,
    const std::vector<cv::Vec3d>& right)
{
    if (left.size() != right.size())
        return false;
    for (size_t index = 0; index < left.size(); ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!nearlyEqual(left[index][axis], right[index][axis]))
                return false;
        }
    }
    return true;
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

    tube.cellsZYX = fiberAnchorCellsNearPolyline(
        tube.referenceIntervalBase,
        radiusBaseVoxels,
        grid,
        anchorCellSizePredictionVoxels);
    return tube;
}

FiberReplayComparisonWindow makeFiberReplayComparisonWindow(
    const PolylineArcGeometry& reference,
    double failureReferenceArcBase,
    const PolylineArcGeometry& trace,
    size_t failureTracePointIndex,
    double requestedHalfExtentBaseVoxels)
{
    if (!(requestedHalfExtentBaseVoxels > 0.0) ||
        !std::isfinite(requestedHalfExtentBaseVoxels) ||
        !std::isfinite(failureReferenceArcBase) ||
        failureReferenceArcBase < 0.0 ||
        failureReferenceArcBase > reference.length() ||
        failureTracePointIndex >= trace.points.size()) {
        throw std::invalid_argument("fiber replay comparison inputs are invalid");
    }
    const double failureTraceArc = trace.vertexArcs.at(failureTracePointIndex);
    const double extent = std::min({
        requestedHalfExtentBaseVoxels,
        failureReferenceArcBase,
        reference.length() - failureReferenceArcBase,
        failureTraceArc,
        trace.length() - failureTraceArc,
    });
    return {
        requestedHalfExtentBaseVoxels,
        extent,
        failureReferenceArcBase - extent,
        failureReferenceArcBase + extent,
        failureTraceArc - extent,
        failureTraceArc,
        failureTraceArc + extent,
    };
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
    if (failed != input.comparison.has_value())
        throw std::invalid_argument("fiber replay failure and comparison window disagree");
    if (failed != (input.anchors.has_value() && input.anchorArtifact.has_value() &&
                   input.paths.has_value() && input.pathArtifact.has_value())) {
        throw std::invalid_argument("fiber replay extraction artifacts disagree with status");
    }
    if (input.graphReplay.has_value() != input.graphReplayConfig.has_value())
        throw std::invalid_argument("fiberlet graph replay result/config disagree");
    if (input.graphReplay.has_value() && !input.graphReplayRequested)
        throw std::invalid_argument("unrequested fiberlet graph replay result");
    if (input.graphReplay.has_value() && !failed)
        throw std::invalid_argument("fiberlet graph replay requires a failed greedy replay tube");

    std::vector<cv::Vec3d> comparisonTraceGeometryBase =
        input.replay.tracePointsBase;
    if (input.comparison.has_value()) {
        const auto& comparison = *input.comparison;
        const std::array<double, 7> values{
            comparison.requestedHalfExtentBaseVoxels,
            comparison.effectiveHalfExtentBaseVoxels,
            comparison.referenceBeginArcBase,
            comparison.referenceEndArcBase,
            comparison.traceBeginArcBase,
            comparison.traceFailureArcBase,
            comparison.traceEndArcBase,
        };
        if (std::any_of(values.begin(), values.end(), [](double value) {
                return !std::isfinite(value);
            }) ||
            !(comparison.requestedHalfExtentBaseVoxels > 0.0) ||
            comparison.effectiveHalfExtentBaseVoxels < 0.0 ||
            comparison.effectiveHalfExtentBaseVoxels >
                comparison.requestedHalfExtentBaseVoxels) {
            throw std::invalid_argument("fiber replay comparison extent is invalid");
        }
        const auto trace = makePolylineArcGeometry(input.replay.tracePointsBase);
        const double failureTraceArc = trace.vertexArcs.at(
            *input.replay.failureTracePointIndex);
        const double extent = comparison.effectiveHalfExtentBaseVoxels;
        if (!nearlyEqual(comparison.referenceBeginArcBase, input.tube->beginArcBase) ||
            !nearlyEqual(comparison.referenceEndArcBase, input.tube->endArcBase) ||
            !samePoints(
                input.referenceGeometryBase,
                input.tube->referenceIntervalBase) ||
            !nearlyEqual(
                comparison.referenceBeginArcBase + extent,
                *input.replay.failureReferenceArcBase) ||
            !nearlyEqual(
                comparison.referenceEndArcBase - extent,
                *input.replay.failureReferenceArcBase) ||
            !nearlyEqual(comparison.traceFailureArcBase, failureTraceArc) ||
            !nearlyEqual(comparison.traceBeginArcBase + extent, failureTraceArc) ||
            !nearlyEqual(comparison.traceEndArcBase - extent, failureTraceArc) ||
            comparison.traceBeginArcBase < 0.0 ||
            comparison.traceEndArcBase > trace.length()) {
            throw std::invalid_argument("fiber replay comparison window is inconsistent");
        }
        comparisonTraceGeometryBase = slicePolylineArc(
            trace,
            comparison.traceBeginArcBase,
            comparison.traceEndArcBase);
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
        lineObj(
            "# vc_fiber_replay_trace version 1",
            comparisonTraceGeometryBase));
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
        if (input.graphReplay.has_value()) {
            vc::core::util::atomicWriteString(
                staging / "replay/fiberlet_trace.json",
                fiberletGraphReplayJson(
                    *input.graphReplay, *input.graphReplayConfig).dump(2) + "\n");
            if (!input.graphReplay->routePointsBaseXYZ.empty()) {
                vc::core::util::atomicWriteString(
                    staging / "replay/fiberlet_trace.obj",
                    fiberletGraphReplayObj(*input.graphReplay));
            }
        }
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
            "paths/fiberlet_graph.json",
        });
        if (input.graphReplay.has_value()) {
            relativeArtifacts.push_back("replay/fiberlet_trace.json");
            if (!input.graphReplay->routePointsBaseXYZ.empty())
                relativeArtifacts.push_back("replay/fiberlet_trace.obj");
        }
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
        {"comparison_trace_points_base_xyz",
         pointsJson(comparisonTraceGeometryBase)},
        {"comparison", nullptr},
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
        {"fiberlet_replay", nullptr},
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
    if (input.comparison.has_value()) {
        const auto& comparison = *input.comparison;
        bundle["comparison"] = {
            {"requested_half_extent_base_voxels",
             comparison.requestedHalfExtentBaseVoxels},
            {"effective_half_extent_base_voxels",
             comparison.effectiveHalfExtentBaseVoxels},
            {"reference_begin_arc_base", comparison.referenceBeginArcBase},
            {"reference_end_arc_base", comparison.referenceEndArcBase},
            {"trace_begin_arc_base", comparison.traceBeginArcBase},
            {"trace_failure_arc_base", comparison.traceFailureArcBase},
            {"trace_end_arc_base", comparison.traceEndArcBase},
        };
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
    if (input.graphReplay.has_value()) {
        bundle["fiberlet_replay"] = fiberletGraphReplayJson(
            *input.graphReplay, *input.graphReplayConfig);
    } else if (input.graphReplayRequested) {
        bundle["fiberlet_replay"] = {
            {"status", "not_run"},
            {"reason", "greedy_replay_did_not_produce_a_failure_tube"},
        };
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
