#include "vc/fiber_tracer/FiberReplay.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceTexture.hpp"
#include "vc/core/util/TexturedMesh.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

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

struct StripAtlasArtifact {
    std::string obj;
    std::string mtl;
    cv::Mat_<uint8_t> texture;
};

struct OmeZarrGroupTransform {
    std::array<double, 3> scaleFromBaseXYZ{};
    std::array<double, 3> offsetFromBaseXYZ{};
};

struct OmeCoordinateTransform {
    std::array<double, 3> scaleZYX{1.0, 1.0, 1.0};
    std::array<double, 3> translationZYX{0.0, 0.0, 0.0};
};

OmeCoordinateTransform omeCoordinateTransform(
    const nlohmann::json& dataset,
    const std::filesystem::path& attrsPath)
{
    OmeCoordinateTransform result;
    if (!dataset.contains("coordinateTransformations"))
        return result;
    const auto& transforms = dataset.at("coordinateTransformations");
    if (!transforms.is_array()) {
        throw std::invalid_argument(
            "OME-Zarr dataset coordinateTransformations must be an array: " +
            attrsPath.string());
    }
    bool sawScale = false;
    bool sawTranslation = false;
    for (const auto& transform : transforms) {
        if (!transform.is_object() || !transform.contains("type") ||
            !transform.at("type").is_string()) {
            throw std::invalid_argument(
                "OME-Zarr dataset has an invalid coordinate transformation: " +
                attrsPath.string());
        }
        const std::string type = transform.at("type").get<std::string>();
        if (type != "scale" && type != "translation")
            continue;
        if (!transform.contains(type) || !transform.at(type).is_array() ||
            transform.at(type).size() != 3 ||
            (type == "scale" && sawScale) ||
            (type == "translation" && sawTranslation)) {
            throw std::invalid_argument(
                "OME-Zarr dataset has an invalid " + type +
                " transformation: " + attrsPath.string());
        }
        auto& target = type == "scale" ? result.scaleZYX : result.translationZYX;
        for (size_t axis = 0; axis < 3; ++axis) {
            if (!transform.at(type).at(axis).is_number()) {
                throw std::invalid_argument(
                    "OME-Zarr dataset transformation values must be numeric: " +
                    attrsPath.string());
            }
            target[axis] = transform.at(type).at(axis).get<double>();
            if (!std::isfinite(target[axis]) ||
                (type == "scale" && !(target[axis] > 0.0))) {
                throw std::invalid_argument(
                    "OME-Zarr dataset transformation values are invalid: " +
                    attrsPath.string());
            }
        }
        sawScale = sawScale || type == "scale";
        sawTranslation = sawTranslation || type == "translation";
    }
    return result;
}

OmeZarrGroupTransform omeZarrGroupTransform(
    const std::filesystem::path& groupPath)
{
    const auto normalizeDirectory = [](const std::filesystem::path& path) {
        auto normalized = std::filesystem::absolute(path).lexically_normal();
        while (normalized != normalized.root_path() &&
               normalized.filename().empty()) {
            normalized = normalized.parent_path();
        }
        return normalized;
    };
    const auto selected = normalizeDirectory(groupPath);
    if (!std::filesystem::exists(selected / ".zarray") &&
        !std::filesystem::exists(selected / "zarr.json")) {
        throw std::invalid_argument(
            "replay strip CT --volume must name a concrete Zarr array/group");
    }

    for (auto root = selected.parent_path(); !root.empty();
         root = root.parent_path()) {
        const auto attrsPath = root / ".zattrs";
        if (!std::filesystem::exists(attrsPath)) {
            if (root == root.root_path())
                break;
            continue;
        }
        std::ifstream input(attrsPath);
        if (!input)
            throw std::runtime_error("cannot read OME-Zarr attributes: " + attrsPath.string());
        nlohmann::json attrs;
        input >> attrs;
        if (!attrs.contains("multiscales") || !attrs.at("multiscales").is_array())
            continue;
        for (const auto& multiscale : attrs.at("multiscales")) {
            if (!multiscale.is_object() || !multiscale.contains("datasets") ||
                !multiscale.at("datasets").is_array() ||
                multiscale.at("datasets").empty()) {
                continue;
            }
            if (multiscale.contains("axes")) {
                const auto& axes = multiscale.at("axes");
                if (!axes.is_array() || axes.size() != 3) {
                    throw std::invalid_argument(
                        "replay strip CT OME-Zarr must have three Z,Y,X axes");
                }
                constexpr std::array<const char*, 3> expected{"z", "y", "x"};
                for (size_t axis = 0; axis < 3; ++axis) {
                    if (!axes.at(axis).is_object() ||
                        axes.at(axis).value("name", std::string{}) != expected[axis]) {
                        throw std::invalid_argument(
                            "replay strip CT OME-Zarr axes must be ordered Z,Y,X");
                    }
                }
            }

            const auto& datasets = multiscale.at("datasets");
            const nlohmann::json* selectedDataset = nullptr;
            for (const auto& dataset : datasets) {
                if (!dataset.is_object() || !dataset.contains("path") ||
                    !dataset.at("path").is_string()) {
                    continue;
                }
                const auto candidate = normalizeDirectory(
                    root / dataset.at("path").get<std::string>());
                if (candidate == selected) {
                    selectedDataset = &dataset;
                    break;
                }
            }
            if (selectedDataset == nullptr)
                continue;
            const auto& baseDataset = datasets.front();
            if (!baseDataset.is_object() || !baseDataset.contains("path") ||
                !baseDataset.at("path").is_string()) {
                throw std::invalid_argument(
                    "OME-Zarr base dataset descriptor is invalid: " +
                    attrsPath.string());
            }
            const auto base = omeCoordinateTransform(baseDataset, attrsPath);
            const auto group = omeCoordinateTransform(*selectedDataset, attrsPath);
            OmeZarrGroupTransform result;
            for (size_t xyzAxis = 0; xyzAxis < 3; ++xyzAxis) {
                const size_t zyxAxis = 2 - xyzAxis;
                result.scaleFromBaseXYZ[xyzAxis] =
                    base.scaleZYX[zyxAxis] / group.scaleZYX[zyxAxis];
                result.offsetFromBaseXYZ[xyzAxis] =
                    (base.translationZYX[zyxAxis] -
                     group.translationZYX[zyxAxis]) /
                    group.scaleZYX[zyxAxis];
            }
            return result;
        }
        if (root == root.root_path())
            break;
    }
    throw std::invalid_argument(
        "replay strip CT --volume group is not advertised by parent OME-Zarr multiscales metadata");
}

cv::Mat_<cv::Vec3f> nativeGroupTextureCoordinates(
    const cv::Mat_<cv::Vec3f>& basePoints,
    const FiberReplayStripTextureSource& source)
{
    cv::Mat_<cv::Vec3f> groupPoints = basePoints.clone();
    for (auto& point : groupPoints) {
        for (size_t axis = 0; axis < 3; ++axis) {
            point[axis] = static_cast<float>(
                static_cast<double>(point[axis]) *
                    source.scaleFromBaseXYZ[axis] +
                source.offsetFromBaseXYZ[axis]);
        }
    }

    const auto distance = [](const cv::Vec3f& left, const cv::Vec3f& right) {
        const cv::Vec3f delta = right - left;
        return std::sqrt(static_cast<double>(delta.dot(delta)));
    };
    double maximumRowArc = 0.0;
    for (int column = 0; column < groupPoints.cols; ++column) {
        double arc = 0.0;
        for (int row = 1; row < groupPoints.rows; ++row)
            arc += distance(groupPoints(row - 1, column), groupPoints(row, column));
        maximumRowArc = std::max(maximumRowArc, arc);
    }
    double maximumColumnArc = 0.0;
    for (int row = 0; row < groupPoints.rows; ++row) {
        double arc = 0.0;
        for (int column = 1; column < groupPoints.cols; ++column)
            arc += distance(groupPoints(row, column - 1), groupPoints(row, column));
        maximumColumnArc = std::max(maximumColumnArc, arc);
    }
    const auto sampleCount = [](double arc) {
        if (!std::isfinite(arc) || arc < 0.0 ||
            arc > static_cast<double>(std::numeric_limits<int>::max() - 1)) {
            throw std::invalid_argument(
                "replay strip source-group extent is invalid");
        }
        return std::max(2, static_cast<int>(std::ceil(arc)) + 1);
    };
    const cv::Size nativeSize(
        sampleCount(maximumColumnArc), sampleCount(maximumRowArc));
    if (nativeSize.width == groupPoints.cols &&
        nativeSize.height == groupPoints.rows) {
        return groupPoints;
    }
    cv::Mat_<cv::Vec3f> nativePoints(nativeSize.height, nativeSize.width);
    for (int row = 0; row < nativePoints.rows; ++row) {
        const double sourceRow = static_cast<double>(row) *
            (groupPoints.rows - 1) / (nativePoints.rows - 1);
        const int row0 = static_cast<int>(std::floor(sourceRow));
        const int row1 = std::min(row0 + 1, groupPoints.rows - 1);
        const float rowWeight = static_cast<float>(sourceRow - row0);
        for (int column = 0; column < nativePoints.cols; ++column) {
            const double sourceColumn = static_cast<double>(column) *
                (groupPoints.cols - 1) / (nativePoints.cols - 1);
            const int column0 = static_cast<int>(std::floor(sourceColumn));
            const int column1 = std::min(column0 + 1, groupPoints.cols - 1);
            const float columnWeight =
                static_cast<float>(sourceColumn - column0);
            const cv::Vec3f top =
                groupPoints(row0, column0) * (1.0F - columnWeight) +
                groupPoints(row0, column1) * columnWeight;
            const cv::Vec3f bottom =
                groupPoints(row1, column0) * (1.0F - columnWeight) +
                groupPoints(row1, column1) * columnWeight;
            nativePoints(row, column) =
                top * (1.0F - rowWeight) + bottom * rowWeight;
        }
    }
    return nativePoints;
}

StripAtlasArtifact stripAtlasArtifact(
    const char* header,
    const std::string& stem,
    const std::vector<FiberReplayStripComponent>& components)
{
    constexpr int padding = 1;
    int atlasRows = 1;
    int atlasColumns = 1;
    if (!components.empty()) {
        atlasRows = 0;
        int64_t columns = 0;
        for (const auto& component : components) {
            if (!component.lineSurface || component.texture.empty()) {
                throw std::invalid_argument(
                    "replay strip component has not been rendered");
            }
            const auto* points = component.lineSurface->rawPointsPtr();
            if (!points || points->empty() ||
                component.texture.rows < 2 || component.texture.cols < 2) {
                throw std::invalid_argument(
                    "replay strip rendered image dimensions are invalid");
            }
            atlasRows = std::max(
                atlasRows, component.texture.rows + 2 * padding);
            columns += static_cast<int64_t>(component.texture.cols) +
                2 * padding;
        }
        if (columns > std::numeric_limits<int>::max()) {
            throw std::overflow_error("replay strip texture atlas is too wide");
        }
        atlasColumns = static_cast<int>(columns);
    }

    cv::Mat_<uint8_t> atlas(atlasRows, atlasColumns, uint8_t(0));
    vc::core::util::TexturedMesh atlasMesh;
    int atlasColumn = 0;
    for (const auto& component : components) {
        const int rows = component.texture.rows;
        const int columns = component.texture.cols;
        component.texture.copyTo(
            atlas(cv::Rect(atlasColumn + padding, padding, columns, rows)));
        component.texture.row(0).copyTo(
            atlas(cv::Rect(atlasColumn + padding, 0, columns, 1)));
        component.texture.row(rows - 1).copyTo(
            atlas(cv::Rect(atlasColumn + padding, rows + padding, columns, 1)));
        component.texture.col(0).copyTo(
            atlas(cv::Rect(atlasColumn, padding, 1, rows)));
        component.texture.col(columns - 1).copyTo(
            atlas(cv::Rect(atlasColumn + columns + padding, padding, 1, rows)));
        atlas(0, atlasColumn) = component.texture(0, 0);
        atlas(rows + padding, atlasColumn) = component.texture(rows - 1, 0);
        atlas(0, atlasColumn + columns + padding) = component.texture(0, columns - 1);
        atlas(rows + padding, atlasColumn + columns + padding) =
            component.texture(rows - 1, columns - 1);

        auto mesh = vc::core::util::texturedSurfaceMesh(*component.lineSurface);
        const size_t vertexOffset = atlasMesh.vertices.size();
        const size_t textureOffset = atlasMesh.textureCoordinates.size();
        atlasMesh.vertices.insert(
            atlasMesh.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        const double left =
            (static_cast<double>(atlasColumn + padding) + 0.5) / atlasColumns;
        const double right =
            (static_cast<double>(atlasColumn + padding + columns) - 0.5) /
            atlasColumns;
        const double bottom =
            1.0 - (static_cast<double>(padding + rows) - 0.5) / atlasRows;
        const double top =
            1.0 - (static_cast<double>(padding) + 0.5) / atlasRows;
        for (const auto& uv : mesh.textureCoordinates) {
            atlasMesh.textureCoordinates.push_back({
                left + uv[0] * (right - left),
                bottom + uv[1] * (top - bottom),
            });
        }
        for (auto quad : mesh.quads) {
            for (size_t corner = 0; corner < 4; ++corner) {
                quad.vertexIndices[corner] += vertexOffset;
                quad.textureCoordinateIndices[corner] += textureOffset;
            }
            atlasMesh.quads.push_back(quad);
        }
        atlasColumn += columns + 2 * padding;
    }
    std::string obj;
    if (components.empty()) {
        obj = std::string("# ") + header + "\nmtllib " + stem +
            ".mtl\nusemtl " + stem + "_texture\n";
    } else {
        obj = vc::core::util::texturedMeshObj(
            atlasMesh,
            header,
            stem + ".mtl",
            stem + "_texture",
            stem);
    }
    return {
        std::move(obj),
        vc::core::util::textureMaterialMtl(
            stem + "_texture", stem + ".tif"),
        std::move(atlas),
    };
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

bool meshNearlyEqual(double left, double right)
{
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <=
        4.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) * scale;
}

bool meshNearlyEqual(const cv::Vec3d& left, const cv::Vec3d& right)
{
    return meshNearlyEqual(left[0], right[0]) &&
        meshNearlyEqual(left[1], right[1]) &&
        meshNearlyEqual(left[2], right[2]);
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

void validateStripComponents(
    const std::vector<FiberReplayStripComponent>& components,
    const std::vector<std::vector<cv::Vec3d>>& source,
    const FiberReplayStripMeshes& meshes)
{
    constexpr int expectedCrossSamples = 21;
    size_t componentIndex = 0;
    for (size_t sourceIndex = 0; sourceIndex < source.size(); ++sourceIndex) {
        const auto& points = source[sourceIndex];
        if (points.size() < 2)
            continue;
        if (componentIndex >= components.size())
            throw std::invalid_argument("replay strip component is missing");
        const auto& component = components[componentIndex++];
        const auto* surfacePoints = component.lineSurface
            ? component.lineSurface->rawPointsPtr()
            : nullptr;
        if (component.sourceSegmentIndex != sourceIndex ||
            !surfacePoints || surfacePoints->empty() ||
            surfacePoints->rows != expectedCrossSamples ||
            static_cast<size_t>(surfacePoints->cols) != points.size() ||
            component.centerlineBaseXYZ != points) {
            throw std::invalid_argument("replay strip component dimensions are invalid");
        }
        if (meshes.textureSource.has_value()) {
            const auto nativePoints = nativeGroupTextureCoordinates(
                *surfacePoints, *meshes.textureSource);
            if (component.texture.empty() ||
                component.texture.rows != nativePoints.rows ||
                component.texture.cols != nativePoints.cols) {
                throw std::invalid_argument(
                    "replay strip rendered CT image is invalid");
            }
        } else if (!component.texture.empty()) {
            throw std::invalid_argument(
                "replay strip rendered CT image has no source metadata");
        }
        for (size_t index = 0; index < points.size(); ++index) {
            if (!meshNearlyEqual(
                    cv::Vec3d((*surfacePoints)(expectedCrossSamples / 2,
                                               static_cast<int>(index))),
                    points[index])) {
                throw std::invalid_argument("replay strip centerline differs from trace geometry");
            }
        }
    }
    if (componentIndex != components.size())
        throw std::invalid_argument("replay strip has an unexpected component");
}

void validateTextureSource(const FiberReplayStripTextureSource& source)
{
    if (source.locator.empty() ||
        std::any_of(
            source.shapeZYX.begin(), source.shapeZYX.end(),
            [](int value) { return value <= 0; }) ||
        std::any_of(
            source.scaleFromBaseXYZ.begin(), source.scaleFromBaseXYZ.end(),
            [](double value) { return !std::isfinite(value) || value <= 0.0; }) ||
        std::any_of(
            source.offsetFromBaseXYZ.begin(), source.offsetFromBaseXYZ.end(),
            [](double value) { return !std::isfinite(value); })) {
        throw std::invalid_argument(
            "replay strip CT source metadata is invalid");
    }
}

void validateStripMeshes(
    const FiberReplayStripMeshes& meshes,
    const FiberReplayTube& tube,
    const std::vector<std::vector<cv::Vec3d>>& greedy,
    const std::vector<std::vector<cv::Vec3d>>& fiberlet)
{
    if (meshes.textureSource.has_value())
        validateTextureSource(*meshes.textureSource);
    validateStripComponents(meshes.reference, {tube.referenceIntervalBase}, meshes);
    validateStripComponents(meshes.greedy, greedy, meshes);
    validateStripComponents(meshes.fiberlet, fiberlet, meshes);
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

FiberReplayStripMeshes makeFiberReplayStripSurfaces(
    const FiberReplayTube& tube,
    const FiberReplayTraceResult& greedyReplay,
    const FiberletGraphReplayResult& fiberletReplay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    int parallelThreads)
{
    if (!(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) || parallelThreads < 1) {
        throw std::invalid_argument("replay strip normal-sampling configuration is invalid");
    }
    FiberReplayStripMeshes result;
    const auto greedy = clippedGreedySegments(
        greedyReplay, tube.beginArcBase, tube.endArcBase);
    const auto fiberlet = clippedFiberletSegments(
        fiberletReplay, tube.beginArcBase, tube.endArcBase);

    struct PendingComponent {
        std::vector<FiberReplayStripComponent>* output = nullptr;
        size_t sourceSegmentIndex = 0;
        std::vector<cv::Vec3d> points;
        size_t normalOffset = 0;
    };
    std::vector<PendingComponent> pending;
    const auto append = [&](std::vector<FiberReplayStripComponent>& output,
                            const std::vector<std::vector<cv::Vec3d>>& source) {
        for (size_t index = 0; index < source.size(); ++index) {
            if (source[index].size() >= 2)
                pending.push_back({&output, index, source[index], 0});
        }
    };
    append(result.reference, {tube.referenceIntervalBase});
    append(result.greedy, greedy);
    append(result.fiberlet, fiberlet);

    std::vector<cv::Vec3d> workingPoints;
    for (auto& component : pending) {
        component.normalOffset = workingPoints.size();
        for (const auto& point : component.points)
            workingPoints.push_back(point * (1.0 / normalWorkingToBaseScale));
    }
    std::vector<vc::lasagna::NormalSampleWithDerivative> normalSamples;
    normalSampler.sampleNormalBatch(
        workingPoints, false, parallelThreads, normalSamples);
    if (normalSamples.size() != workingPoints.size())
        throw std::runtime_error("replay strip normal sampler returned the wrong count");

    for (auto& component : pending) {
        vc::lasagna::LineModel line;
        line.points.reserve(component.points.size());
        for (size_t index = 0; index < component.points.size(); ++index) {
            line.points.push_back({
                component.points[index],
                normalSamples[component.normalOffset + index].sample,
                true,
            });
        }
        auto views = vc::lasagna::buildLineViewSurfaces(line);
        FiberReplayStripComponent output;
        output.sourceSegmentIndex = component.sourceSegmentIndex;
        output.centerlineBaseXYZ = std::move(component.points);
        output.lineSurface = std::move(views.lineSurface);
        component.output->push_back(std::move(output));
    }
    validateStripMeshes(result, tube, greedy, fiberlet);
    return result;
}

FiberReplayStripTextureSource validateFiberReplayStripCtVolume(
    ::Volume& volume,
    const std::string& sourceLocator)
{
    if (sourceLocator.empty())
        throw std::invalid_argument("replay strip CT source locator is empty");
    auto* cache = volume.chunkedCache();
    if (cache == nullptr)
        throw std::runtime_error("replay strip CT volume has no chunk cache");
    if (cache->numLevels() != 1 || !volume.hasScaleLevel(0)) {
        throw std::invalid_argument(
            "replay strip CT --volume must name one concrete Zarr array/group");
    }
    if (volume.dtype() != vc::render::ChunkDtype::UInt8) {
        throw std::invalid_argument(
            "replay strip CT volume must use uint8 voxels");
    }
    const auto transform = omeZarrGroupTransform(sourceLocator);
    FiberReplayStripTextureSource source{
        sourceLocator,
        volume.shape(0),
        transform.scaleFromBaseXYZ,
        transform.offsetFromBaseXYZ,
    };
    validateTextureSource(source);
    return source;
}

void renderFiberReplayStripTextures(
    FiberReplayStripMeshes& meshes,
    ::Volume& volume,
    const std::string& sourceLocator)
{
    if (meshes.textureSource.has_value())
        throw std::invalid_argument("replay strip CT has already been rendered");
    const auto source = validateFiberReplayStripCtVolume(volume, sourceLocator);

    const auto allComponents = [&](auto&& callback) {
        for (auto* collection : {&meshes.reference, &meshes.greedy,
                                 &meshes.fiberlet}) {
            for (auto& component : *collection)
                callback(component);
        }
    };
    allComponents([&](FiberReplayStripComponent& component) {
        if (!component.lineSurface || !component.texture.empty()) {
            throw std::invalid_argument(
                "replay strip component is missing or already rendered");
        }
        const auto* basePoints = component.lineSurface->rawPointsPtr();
        if (!basePoints || basePoints->empty())
            throw std::invalid_argument("replay strip surface has no coordinates");
        const auto groupPoints = nativeGroupTextureCoordinates(
            *basePoints, source);
        component.texture = vc::core::util::renderCoordsTextureFineToCoarse(
            groupPoints, volume, 0, 1, "Strip texture sampling");
    });
    meshes.textureSource = source;
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

        if (visualization.strips.has_value()) {
            if (!visualization.strips->textureSource.has_value()) {
                throw std::invalid_argument(
                    "replay strip visualization has no rendered CT images");
            }
            validateStripMeshes(
                *visualization.strips, visualization.tube, greedy, fiberlet);
            const auto referenceStrip = stripAtlasArtifact(
                "vc_fiber_replay_reference_strip version 4",
                "reference_strip", visualization.strips->reference);
            const auto greedyStrip = stripAtlasArtifact(
                "vc_greedy_fiber_replay_strip version 4",
                "greedy_strip", visualization.strips->greedy);
            const auto fiberletStrip = stripAtlasArtifact(
                "vc_fiberlet_graph_replay_strip version 4",
                "fiberlet_strip", visualization.strips->fiberlet);
            vc::core::util::atomicWriteString(
                directory / "replay/reference_strip.obj",
                referenceStrip.obj);
            vc::core::util::atomicWriteString(
                directory / "replay/reference_strip.mtl",
                referenceStrip.mtl);
            vc::core::util::writeUncompressedTextureTiff(
                directory / "replay/reference_strip.tif",
                referenceStrip.texture);
            vc::core::util::atomicWriteString(
                directory / "replay/greedy_strip.obj",
                greedyStrip.obj);
            vc::core::util::atomicWriteString(
                directory / "replay/greedy_strip.mtl",
                greedyStrip.mtl);
            vc::core::util::writeUncompressedTextureTiff(
                directory / "replay/greedy_strip.tif",
                greedyStrip.texture);
            vc::core::util::atomicWriteString(
                directory / "replay/fiberlet_strip.obj",
                fiberletStrip.obj);
            vc::core::util::atomicWriteString(
                directory / "replay/fiberlet_strip.mtl",
                fiberletStrip.mtl);
            vc::core::util::writeUncompressedTextureTiff(
                directory / "replay/fiberlet_strip.tif",
                fiberletStrip.texture);
        }

        std::vector<std::filesystem::path> localArtifacts{
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
        if (visualization.strips.has_value()) {
            localArtifacts.insert(
                localArtifacts.end(),
                {
                    "replay/reference_strip.obj",
                    "replay/reference_strip.mtl",
                    "replay/reference_strip.tif",
                    "replay/greedy_strip.obj",
                    "replay/greedy_strip.mtl",
                    "replay/greedy_strip.tif",
                    "replay/fiberlet_strip.obj",
                    "replay/fiberlet_strip.mtl",
                    "replay/fiberlet_strip.tif",
                });
        }
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
        if (visualization.strips.has_value()) {
            const auto& texture = *visualization.strips->textureSource;
            local["trace_strips"] = {
                {"orientation", "sheet_aligned_normal_cross_tangent"},
                {"geometry_builder", "buildLineViewSurfaces_default"},
                {"cross_samples", 21},
                {"values",
                 {
                     {"semantic", "ct_intensity"},
                     {"encoding", "obj_uv_grayscale_tiff_u8"},
                     {"renderer", "vc_line_probe_fine_to_coarse"},
                     {"sampling_grid", "source_group_voxel_pitch"},
                     {"atlas_padding_pixels", 1},
                     {"texture_format", "tiff_gray_u8_uncompressed"},
                     {"source_locator", texture.locator},
                     {"source_dtype", "uint8"},
                     {"source_shape_zyx", texture.shapeZYX},
                     {"source_group_scale_from_base_xyz",
                      texture.scaleFromBaseXYZ},
                     {"source_group_offset_from_base_xyz",
                      texture.offsetFromBaseXYZ},
                     {"source_storage_order", "ZYX"},
                     {"vertex_position_order", "XYZ"},
                     {"position_space", "base_volume"},
                     {"scale_xyz", {1.0, 1.0, 1.0}},
                     {"translation_xyz", {0.0, 0.0, 0.0}},
                 }},
            };
        }
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
