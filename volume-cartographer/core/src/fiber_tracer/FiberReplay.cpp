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
#include <string_view>
#include <tuple>

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

nlohmann::json optionalPointJson(const std::optional<cv::Vec3d>& point)
{
    return point.has_value() ? nlohmann::json(pointJson(*point)) : nlohmann::json(nullptr);
}

nlohmann::json failureJson(const FiberReplayFailure& failure)
{
    return {
        {"index", failure.index},
        {"segment_index", failure.segmentIndex},
        {"reason", failure.reason},
        {"reference_arc_base", failure.referenceArcBase},
        {"reference_arc_fraction", failure.referenceArcFraction},
        {"reference_point_base_xyz", pointJson(failure.referencePointBase)},
        {"evaluator_point_base_xyz", optionalPointJson(failure.evaluatorPointBase)},
        {"segment_point_index", failure.segmentPointIndex.has_value() ? nlohmann::json(*failure.segmentPointIndex) : nlohmann::json(nullptr)},
        {"candidate_index", failure.candidateIndex.has_value() ? nlohmann::json(*failure.candidateIndex) : nlohmann::json(nullptr)},
        {"arc_index", failure.arcIndex.has_value() ? nlohmann::json(*failure.arcIndex) : nlohmann::json(nullptr)},
        {"candidate_path_point_index",
         failure.candidatePathPointIndex.has_value() ? nlohmann::json(*failure.candidatePathPointIndex) : nlohmann::json(nullptr)},
        {"error_base_voxels", failure.errorBaseVoxels.has_value() ? nlohmann::json(*failure.errorBaseVoxels) : nlohmann::json(nullptr)},
        {"error_ratio", failure.errorRatio.has_value() ? nlohmann::json(*failure.errorRatio) : nlohmann::json(nullptr)},
    };
}

std::string lineObj(const char* header, const std::vector<cv::Vec3d>& points, bool pointRecord = false)
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

std::string segmentedLineObj(const char* header, const std::vector<std::vector<cv::Vec3d>>& segments)
{
    std::ostringstream output;
    output << header << '\n' << std::setprecision(17);
    size_t offset = 0;
    for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
        const auto& points = segments[segmentIndex];
        if (points.empty())
            continue;
        output << "g segment_" << segmentIndex << '\n';
        for (const auto& point : points)
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
        output << "l";
        if (points.size() == 1) {
            output << ' ' << offset + 1 << ' ' << offset + 1;
        } else {
            for (size_t index = 0; index < points.size(); ++index)
                output << ' ' << offset + index + 1;
        }
        output << '\n';
        offset += points.size();
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

bool nearlyEqual(const cv::Vec3d& left, const cv::Vec3d& right)
{
    return nearlyEqual(left[0], right[0]) &&
        nearlyEqual(left[1], right[1]) && nearlyEqual(left[2], right[2]);
}

template <typename Replay>
void validateReplayFailures(
    const Replay& replay, const char* tracer)
{
    const double length =
        replay.referenceEndArcBase - replay.referenceBeginArcBase;
    for (size_t index = 0; index < replay.failures.size(); ++index) {
        const auto& failure = replay.failures[index];
        if (failure.index != index ||
            failure.referenceArcBase < replay.referenceBeginArcBase - kEpsilon ||
            failure.referenceArcBase > replay.referenceEndArcBase + kEpsilon ||
            failure.referenceArcFraction < -kEpsilon ||
            failure.referenceArcFraction > 1.0 + kEpsilon ||
            !nearlyEqual(
                failure.referenceArcFraction,
                (failure.referenceArcBase - replay.referenceBeginArcBase) /
                    length)) {
            throw std::invalid_argument(
                std::string(tracer) +
                " replay failure lies outside the selected interval");
        }
    }
}

std::vector<std::vector<cv::Vec3d>> greedySegments(const FiberReplayTraceResult& replay)
{
    std::vector<std::vector<cv::Vec3d>> result;
    result.reserve(replay.segments.size());
    for (const auto& segment : replay.segments)
        result.push_back(segment.tracePointsBase);
    return result;
}

std::vector<std::vector<cv::Vec3d>> fiberletSegments(const FiberletGraphReplayResult& replay)
{
    std::vector<std::vector<cv::Vec3d>> result;
    result.reserve(replay.segments.size());
    for (const auto& segment : replay.segments)
        result.push_back(segment.routePointsBaseXYZ);
    return result;
}

nlohmann::json greedyReplayJson(const FiberReplayTraceResult& replay)
{
    nlohmann::json root = {
        {"format", "vc_greedy_fiber_replay"},
        {"version", 2},
        {"reference_begin_arc_base", replay.referenceBeginArcBase},
        {"reference_end_arc_base", replay.referenceEndArcBase},
        {"completed_reference_arc_base", replay.completedReferenceArcBase},
        {"segments", nlohmann::json::array()},
        {"failures", nlohmann::json::array()},
    };
    for (const auto& segment : replay.segments) {
        nlohmann::json matches = nlohmann::json::array();
        for (const auto& match : segment.matches) {
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
        root["segments"].push_back({
            {"start_reference_arc_base", segment.startReferenceArcBase},
            {"end_reference_arc_base", segment.endReferenceArcBase},
            {"termination_reason", segment.terminationReason},
            {"trace_points_base_xyz", pointsJson(segment.tracePointsBase)},
            {"trace_cumulative_losses", segment.cumulativeLosses},
            {"matches", std::move(matches)},
        });
    }
    for (const auto& failure : replay.failures)
        root["failures"].push_back(failureJson(failure));
    return root;
}

struct ArcPoint {
    double arc = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
};

std::vector<cv::Vec3d> clipArcPoints(const std::vector<ArcPoint>& input, double beginArc, double endArc)
{
    if (input.empty() || input.back().arc < beginArc - kEpsilon || input.front().arc > endArc + kEpsilon)
        return {};
    const auto sample = [&](double arc) {
        if (arc <= input.front().arc + kEpsilon)
            return input.front().point;
        if (arc >= input.back().arc - kEpsilon)
            return input.back().point;
        const auto upper =
            std::upper_bound(input.begin(), input.end(), arc, [](double value, const ArcPoint& item) { return value < item.arc; });
        const auto& right = *upper;
        const auto& left = *(upper - 1);
        if (right.arc <= left.arc + kEpsilon)
            return right.point;
        const double fraction = (arc - left.arc) / (right.arc - left.arc);
        return left.point + fraction * (right.point - left.point);
    };
    const double clippedBegin = std::max(beginArc, input.front().arc);
    const double clippedEnd = std::min(endArc, input.back().arc);
    std::vector<cv::Vec3d> output{sample(clippedBegin)};
    for (const auto& item : input) {
        if (item.arc > clippedBegin + kEpsilon && item.arc < clippedEnd - kEpsilon)
            output.push_back(item.point);
    }
    const cv::Vec3d end = sample(clippedEnd);
    if (cv::norm(end - output.back()) > kEpsilon)
        output.push_back(end);
    return output;
}

std::vector<std::vector<cv::Vec3d>> clippedGreedySegments(const FiberReplayTraceResult& replay, double beginArc, double endArc)
{
    std::vector<std::vector<cv::Vec3d>> output;
    for (const auto& segment : replay.segments) {
        if (segment.tracePointsBase.empty())
            continue;
        std::vector<ArcPoint> points{{segment.startReferenceArcBase, segment.tracePointsBase.front()}};
        for (const auto& match : segment.matches) {
            if (match.tracePointIndex >= segment.tracePointsBase.size())
                throw std::logic_error("greedy replay match point index is invalid");
            points.push_back({match.matchedReferenceArcBase, segment.tracePointsBase[match.tracePointIndex]});
        }
        auto clipped = clipArcPoints(points, beginArc, endArc);
        if (!clipped.empty())
            output.push_back(std::move(clipped));
    }
    return output;
}

std::vector<std::vector<cv::Vec3d>> clippedFiberletSegments(const FiberletGraphReplayResult& replay, double beginArc, double endArc)
{
    std::vector<std::vector<cv::Vec3d>> output;
    for (const auto& segment : replay.segments) {
        if (segment.routePointsBaseXYZ.empty() || segment.matches.empty())
            continue;
        std::vector<ArcPoint> points;
        for (const auto& match : segment.matches) {
            if (match.routePointIndex >= segment.routePointsBaseXYZ.size())
                throw std::logic_error("fiberlet replay match point index is invalid");
            points.push_back({match.matchedReferenceArcBase, segment.routePointsBaseXYZ[match.routePointIndex]});
        }
        auto clipped = clipArcPoints(points, beginArc, endArc);
        if (!clipped.empty())
            output.push_back(std::move(clipped));
    }
    return output;
}

const FiberReplayFailure& visualizationFailure(const FiberReplayBundleInput& input, const FiberReplayVisualizationInput& visualization)
{
    const auto& failures = visualization.tracer == FiberReplayTracer::Greedy ? input.greedyReplay.failures : input.fiberletReplay.failures;
    if (visualization.tracerFailureIndex >= failures.size())
        throw std::invalid_argument("replay visualization failure index is invalid");
    return failures[visualization.tracerFailureIndex];
}

std::vector<std::filesystem::path> relativeFiles(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file())
            files.push_back(std::filesystem::relative(entry.path(), root));
    }
    std::sort(files.begin(), files.end());
    return files;
}

}  // namespace

bool FiberReplayTube::containsBasePoint(const cv::Vec3d& point) const
{
    return distanceToBasePoint(point) <= radiusBaseVoxels + kEpsilon;
}

double FiberReplayTube::distanceToBasePoint(const cv::Vec3d& point) const
{
    return distanceToPolylineArc(reference, point, beginArcBase, endArcBase);
}

bool FiberReplayTube::containsPredictionPoint(const cv::Vec3d& pointPredictionXYZ, double predictionToBaseScale) const
{
    return containsBasePoint(pointPredictionXYZ * predictionToBaseScale);
}

FiberReplayTube makeFiberReplayTube(
    const std::vector<cv::Vec3d>& referenceLineBase, double centerArcBase, double alongBaseVoxels, double radiusBaseVoxels, const FiberPredictionGridInfo& grid, int anchorCellSizePredictionVoxels)
{
    if (!(alongBaseVoxels >= 0.0) || !std::isfinite(alongBaseVoxels) || !(radiusBaseVoxels > 0.0) || !std::isfinite(radiusBaseVoxels)) {
        throw std::invalid_argument("fiber replay tube distances are invalid");
    }
    if (!(grid.predictionToBaseScale > 0.0) || !std::isfinite(grid.predictionToBaseScale) || anchorCellSizePredictionVoxels < 1) {
        throw std::invalid_argument("fiber replay tube grid is invalid");
    }
    FiberReplayTube tube;
    tube.reference = makePolylineArcGeometry(referenceLineBase);
    tube.beginArcBase = std::max(0.0, centerArcBase - alongBaseVoxels);
    tube.endArcBase = std::min(tube.reference.length(), centerArcBase + alongBaseVoxels);
    tube.radiusBaseVoxels = radiusBaseVoxels;
    tube.referenceIntervalBase = slicePolylineArc(tube.reference, tube.beginArcBase, tube.endArcBase);

    cv::Vec3d low = tube.referenceIntervalBase.front();
    cv::Vec3d high = low;
    for (const auto& point : tube.referenceIntervalBase) {
        for (int axis = 0; axis < 3; ++axis) {
            low[axis] = std::min(low[axis], point[axis]);
            high[axis] = std::max(high[axis], point[axis]);
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double gridHigh = static_cast<double>(grid.shapeZYX[2 - axis]) * grid.predictionToBaseScale;
        const double cropLow = std::clamp(std::floor(low[axis] - radiusBaseVoxels), 0.0, gridHigh);
        const double cropHigh = std::clamp(std::ceil(high[axis] + radiusBaseVoxels), 0.0, gridHigh);
        tube.volumeCropBaseXYZWHD[axis] = static_cast<size_t>(cropLow);
        tube.volumeCropBaseXYZWHD[axis + 3] = static_cast<size_t>(cropHigh - cropLow);
    }

    tube.cellsZYX = fiberAnchorCellsNearPolyline(tube.referenceIntervalBase, radiusBaseVoxels, grid, anchorCellSizePredictionVoxels);
    return tube;
}

const char* fiberReplayTracerName(FiberReplayTracer tracer) noexcept
{
    return tracer == FiberReplayTracer::Greedy ? "greedy" : "fiberlet";
}

nlohmann::json writeFiberReplayBundle(const std::filesystem::path& outputDirectory, const FiberReplayBundleInput& input)
{
    if (input.referenceGeometryBase.size() < 2)
        throw std::invalid_argument("fiber replay reference geometry is incomplete");
    if (!nearlyEqual(input.greedyReplay.referenceBeginArcBase, input.fiberletReplay.referenceBeginArcBase) ||
        !nearlyEqual(input.greedyReplay.referenceEndArcBase, input.fiberletReplay.referenceEndArcBase) ||
        !nearlyEqual(input.greedyReplay.completedReferenceArcBase, input.greedyReplay.referenceEndArcBase) ||
        !nearlyEqual(input.fiberletReplay.completedReferenceArcBase, input.fiberletReplay.referenceEndArcBase)) {
        throw std::invalid_argument("dual replay reference intervals are inconsistent or incomplete");
    }
    const double beginArc = input.greedyReplay.referenceBeginArcBase;
    const double endArc = input.greedyReplay.referenceEndArcBase;
    if (!(endArc > beginArc + kEpsilon) ||
        input.request.startControlPointIndex >=
            input.request.fiber.controlPointLineIndices.size()) {
        throw std::invalid_argument("fiber replay selected interval is invalid");
    }
    const auto fullReference =
        makePolylineArcGeometry(input.request.fiber.linePointsXyzBase);
    const size_t startLineIndex = input.request.fiber.controlPointLineIndices[
        input.request.startControlPointIndex];
    if (startLineIndex >= fullReference.vertexArcs.size() ||
        !nearlyEqual(fullReference.vertexArcs[startLineIndex], beginArc) ||
        endArc > fullReference.length() + kEpsilon) {
        throw std::invalid_argument(
            "fiber replay selected interval differs from its request");
    }
    const auto expectedReference =
        slicePolylineArc(fullReference, beginArc, endArc);
    if (expectedReference.size() != input.referenceGeometryBase.size() ||
        !std::equal(
            expectedReference.begin(), expectedReference.end(),
            input.referenceGeometryBase.begin(),
            [](const auto& left, const auto& right) {
                return nearlyEqual(left, right);
            })) {
        throw std::invalid_argument(
            "fiber replay reference geometry differs from the selected interval");
    }
    if (!nearlyEqual(input.fiberletReplayConfig.referenceBeginArcBase, beginArc) ||
        !input.fiberletReplayConfig.referenceEndArcBase.has_value() ||
        !nearlyEqual(*input.fiberletReplayConfig.referenceEndArcBase, endArc) ||
        !input.request.referenceEndArcBase.has_value() ||
        !nearlyEqual(*input.request.referenceEndArcBase, endArc)) {
        throw std::invalid_argument(
            "fiber replay engine configurations differ from the selected interval");
    }
    validateReplayFailures(input.greedyReplay, "greedy");
    validateReplayFailures(input.fiberletReplay, "fiberlet");

    std::vector<const FiberReplayVisualizationInput*> visualizations;
    visualizations.reserve(input.visualizations.size());
    for (const auto& visualization : input.visualizations)
    {
        const auto& failure = visualizationFailure(input, visualization);
        if (visualization.tube.beginArcBase < beginArc - kEpsilon ||
            visualization.tube.endArcBase > endArc + kEpsilon ||
            failure.referenceArcBase < visualization.tube.beginArcBase - kEpsilon ||
            failure.referenceArcBase > visualization.tube.endArcBase + kEpsilon) {
            throw std::invalid_argument(
                "fiber replay visualization exceeds the selected interval");
        }
        visualizations.push_back(&visualization);
    }
    std::sort(visualizations.begin(), visualizations.end(), [&](const auto* left, const auto* right) {
        const auto& leftFailure = visualizationFailure(input, *left);
        const auto& rightFailure = visualizationFailure(input, *right);
        return std::tuple{leftFailure.referenceArcBase, left->tracer, left->tracerFailureIndex} <
               std::tuple{rightFailure.referenceArcBase, right->tracer, right->tracerFailureIndex};
    });

    std::filesystem::create_directories(outputDirectory / "runs");
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path staging = outputDirectory / "runs" / (".staging-" + std::to_string(stamp));
    std::filesystem::create_directories(staging / "replay");

    const auto greedyJson = greedyReplayJson(input.greedyReplay);
    const auto fiberletJson = fiberletGraphReplayJson(input.fiberletReplay, input.fiberletReplayConfig);
    vc::core::util::atomicWriteString(staging / "replay/reference.obj", lineObj("# vc_fiber_replay_reference version 2", input.referenceGeometryBase));
    vc::core::util::atomicWriteString(staging / "replay/greedy.json", greedyJson.dump(2) + "\n");
    vc::core::util::
        atomicWriteString(staging / "replay/greedy.obj", segmentedLineObj("# vc_greedy_fiber_replay version 2", greedySegments(input.greedyReplay)));
    vc::core::util::atomicWriteString(staging / "replay/fiberlet.json", fiberletJson.dump(2) + "\n");
    vc::core::util::atomicWriteString(staging / "replay/fiberlet.obj", fiberletGraphReplayObj(input.fiberletReplay));

    nlohmann::json visualizationIndex = nlohmann::json::array();
    for (size_t globalIndex = 0; globalIndex < visualizations.size(); ++globalIndex) {
        const auto& visualization = *visualizations[globalIndex];
        const auto& failure = visualizationFailure(input, visualization);
        const std::ostringstream nameBuilder = [&]() {
            std::ostringstream value;
            value << std::setfill('0') << std::setw(6) << globalIndex;
            return value;
        }();
        const std::filesystem::path relativeDirectory = std::filesystem::path("visualizations") / nameBuilder.str();
        const auto directory = staging / relativeDirectory;
        std::filesystem::create_directories(directory / "replay");
        writeFiberAnchorArtifacts(directory / "anchors", visualization.anchors, visualization.anchorArtifact);
        writeFiberletPathArtifacts(directory / "paths", visualization.paths, visualization.pathArtifact);
        const auto greedy = clippedGreedySegments(input.greedyReplay, visualization.tube.beginArcBase, visualization.tube.endArcBase);
        const auto fiberlet = clippedFiberletSegments(input.fiberletReplay, visualization.tube.beginArcBase, visualization.tube.endArcBase);
        const cv::Vec3d marker = failure.evaluatorPointBase.value_or(failure.referencePointBase);
        vc::core::util::
            atomicWriteString(directory / "replay/reference.obj", lineObj("# vc_fiber_replay_reference version 2", visualization.tube.referenceIntervalBase));
        vc::core::util::atomicWriteString(directory / "replay/greedy.obj", segmentedLineObj("# vc_greedy_fiber_replay version 2", greedy));
        vc::core::util::atomicWriteString(directory / "replay/fiberlet.obj", segmentedLineObj("# vc_fiberlet_graph_replay version 2", fiberlet));
        vc::core::util::atomicWriteString(directory / "replay/failure.obj", lineObj("# vc_fiber_replay_failure version 2", {marker}, true));

        const std::vector<std::filesystem::path> localArtifacts{
            "replay/reference.obj",
            "replay/greedy.obj",
            "replay/fiberlet.obj",
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
        };
        nlohmann::json local = {
            {"format", "vc_fiber_replay_visualization"},
            {"version", 1},
            {"identity",
             {
                 {"global_index", globalIndex},
                 {"tracer", fiberReplayTracerName(visualization.tracer)},
                 {"tracer_failure_index", visualization.tracerFailureIndex},
             }},
            {"coordinates", {{"position_order", "XYZ"}, {"position_space", "base_volume"}, {"distance_unit", "base_voxels"}}},
            {"sources", input.sources},
            {"prediction_binding", input.predictionBinding},
            {"failure", failureJson(failure)},
            {"tube",
             {
                 {"begin_arc_base", visualization.tube.beginArcBase},
                 {"end_arc_base", visualization.tube.endArcBase},
                 {"radius_base_voxels", visualization.tube.radiusBaseVoxels},
                 {"reference_points_base_xyz", pointsJson(visualization.tube.referenceIntervalBase)},
                 {"cells_zyx", visualization.tube.cellsZYX},
             }},
            {"volume_crop_base_xyzwhd", visualization.tube.volumeCropBaseXYZWHD},
            {"reference_points_base_xyz", pointsJson(visualization.tube.referenceIntervalBase)},
            {"greedy_trace_segments_base_xyz", nlohmann::json::array()},
            {"fiberlet_trace_segments_base_xyz", nlohmann::json::array()},
            {"artifacts", nlohmann::json::object()},
        };
        for (const auto& segment : greedy)
            local["greedy_trace_segments_base_xyz"].push_back(pointsJson(segment));
        for (const auto& segment : fiberlet)
            local["fiberlet_trace_segments_base_xyz"].push_back(pointsJson(segment));
        for (const auto& relative : localArtifacts) {
            local["artifacts"][relative.generic_string()] = {
                {"path", relative.generic_string()},
                {"content_hash", artifactHash(directory / relative)},
            };
        }
        vc::core::util::atomicWriteString(directory / "manifest.json", local.dump(2) + "\n");
        visualizationIndex.push_back({
            {"global_index", globalIndex},
            {"tracer", fiberReplayTracerName(visualization.tracer)},
            {"tracer_failure_index", visualization.tracerFailureIndex},
            {"reference_arc_base", failure.referenceArcBase},
            {"reference_arc_fraction", failure.referenceArcFraction},
            {"manifest_relative_path", (relativeDirectory / "manifest.json").generic_string()},
        });
    }

    std::string generationMaterial;
    for (const auto& relative : relativeFiles(staging)) {
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

    const std::filesystem::path generationRelative = std::filesystem::path("runs") / generationName;
    std::set<std::string> publishedVisualizationAliases;
    for (auto& entry : visualizationIndex) {
        const auto localRelative = std::filesystem::path(
            entry.at("manifest_relative_path").get<std::string>());
        auto local = nlohmann::json::parse(
            readFile(finalGeneration / localRelative));
        for (auto& [key, descriptor] : local.at("artifacts").items()) {
            const auto artifactRelative = std::filesystem::path(
                descriptor.at("path").get<std::string>());
            descriptor["path"] =
                (generationRelative / localRelative.parent_path() /
                 artifactRelative)
                    .generic_string();
        }
        std::ostringstream aliasName;
        aliasName << "fiber_replay_visualization."
                  << entry.at("tracer").get<std::string>() << '.'
                  << std::setfill('0') << std::setw(6)
                  << entry.at("tracer_failure_index").get<size_t>() << ".json";
        const std::string alias = aliasName.str();
        const std::string content = local.dump(2) + "\n";
        vc::core::util::atomicWriteString(outputDirectory / alias, content);
        publishedVisualizationAliases.insert(alias);
        entry["manifest"] = {
            {"path", alias},
            {"content_hash", hashString(content)},
        };
        entry.erase("manifest_relative_path");
    }
    nlohmann::json root = {
        {"format", "vc_fiber_replay"},
        {"version", 2},
        {"coordinates", {{"position_order", "XYZ"}, {"position_space", "base_volume"}, {"distance_unit", "base_voxels"}}},
        {"sources", input.sources},
        {"bindings", {{"trace", input.traceBinding}, {"prediction", input.predictionBinding}}},
        {"trace_config", {{"requested", input.requestedTraceConfig}, {"effective", input.effectiveTraceConfig}}},
        {"fiberlet_config", fiberletJson.at("config")},
        {"requested_length_base_voxels", input.requestedLengthBaseVoxels.has_value()
             ? nlohmann::json(*input.requestedLengthBaseVoxels)
             : nlohmann::json(nullptr)},
        {"reference_begin_arc_base", beginArc},
        {"reference_end_arc_base", endArc},
        {"reference_length_base_voxels", endArc - beginArc},
        {"reference_points_base_xyz", pointsJson(input.referenceGeometryBase)},
        {"greedy", greedyJson},
        {"fiberlet", fiberletJson},
        {"failure_counts", {{"greedy", input.greedyReplay.failures.size()}, {"fiberlet", input.fiberletReplay.failures.size()}}},
        {"visualizations", visualizationIndex},
        {"artifacts", nlohmann::json::object()},
    };
    for (const auto& relative : std::vector<
             std::filesystem::
                 path>{"replay/reference.obj", "replay/greedy.json", "replay/greedy.obj", "replay/fiberlet.json", "replay/fiberlet.obj"}) {
        root["artifacts"][relative.generic_string()] = {
            {"path", (generationRelative / relative).generic_string()},
            {"content_hash", artifactHash(finalGeneration / relative)},
        };
    }
    vc::core::util::atomicWriteString(outputDirectory / "fiber_replay.json", root.dump(2) + "\n");
    constexpr std::string_view kVisualizationPrefix =
        "fiber_replay_visualization.";
    for (const auto& entry : std::filesystem::directory_iterator(outputDirectory)) {
        if (!entry.is_regular_file())
            continue;
        const std::string name = entry.path().filename().string();
        if (name.starts_with(kVisualizationPrefix) &&
            !publishedVisualizationAliases.contains(name)) {
            std::filesystem::remove(entry.path());
        }
    }
    return root;
}

}  // namespace vc::fiber_tracer
