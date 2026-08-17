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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

nlohmann::json failureJson(
    const FiberReplayFailure& failure,
    double normalThresholdBaseVoxels)
{
    auto result = fiberReplayOptionalThresholdMeasurementJson(
        failure.thresholdMeasurement, normalThresholdBaseVoxels);
    result.update({
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
    });
    return result;
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

const cv::Vec3b kReferenceColorBgr{0, 255, 255};
const cv::Vec3b kGreedyColorBgr{0, 0, 255};
const cv::Vec3b kFiberletColorBgr{255, 255, 0};
constexpr int kOverviewHeaderRows = 50;
constexpr int kOverviewLabelRows = 20;
constexpr int kOverviewSeparatorRows = 4;
constexpr int kOverviewPanelGapRows = 8;
constexpr int kOverviewMinimumColumns = 90;
constexpr int kOverviewPanelColumns = 32000;
constexpr int kJpegMaximumDimension = 65000;
constexpr int kOverviewRenderScale = 8;
constexpr int kOverviewMarkerWidthPixels = 3;
constexpr int kOverviewFiberletComponentGapColumns = 8;
const cv::Vec3b kOverlapColorBgr{255, 0, 255};

struct MatchedTracePoint {
    cv::Vec3d pointBase;
    cv::Vec3d referencePointBase;
    double referenceArcBase = 0.0;
};

using MatchedTraceSegment = std::vector<MatchedTracePoint>;

struct IndexedMatchedTraceSegment {
    size_t sourceSegmentIndex = 0;
    MatchedTraceSegment points;
};

void validateMatchedPoint(const MatchedTracePoint& point, const PolylineArcGeometry& selectedReference, double referenceBeginArc, double referenceEndArc, const char* tracer)
{
    if (!std::isfinite(point.referenceArcBase) || point.referenceArcBase < referenceBeginArc - kEpsilon ||
        point.referenceArcBase > referenceEndArc + kEpsilon) {
        throw std::invalid_argument(std::string(tracer) + " replay overview match lies outside the selected interval");
    }
    const auto expected =
        samplePolylineArc(selectedReference, std::clamp(point.referenceArcBase - referenceBeginArc, 0.0, selectedReference.length()));
    if (cv::norm(expected.point - point.referencePointBase) > 1.0e-6) {
        throw std::invalid_argument(std::string(tracer) + " replay overview match point differs from its reference arc");
    }
}

std::vector<MatchedTraceSegment> matchedGreedySegments(const FiberReplayTraceResult& replay, const PolylineArcGeometry& selectedReference)
{
    std::vector<MatchedTraceSegment> output;
    for (const auto& segment : replay.segments) {
        if (segment.tracePointsBase.empty())
            continue;
        if (segment.matches.size() + 1 != segment.tracePointsBase.size()) {
            throw std::invalid_argument("greedy replay overview requires one match per non-seed point");
        }
        MatchedTraceSegment points;
        points.reserve(segment.tracePointsBase.size());
        MatchedTracePoint seed{
            segment.tracePointsBase.front(),
            samplePolylineArc(
                selectedReference, std::clamp(segment.startReferenceArcBase - replay.referenceBeginArcBase, 0.0, selectedReference.length()))
                .point,
            segment.startReferenceArcBase,
        };
        validateMatchedPoint(seed, selectedReference, replay.referenceBeginArcBase, replay.referenceEndArcBase, "greedy");
        points.push_back(seed);
        for (size_t index = 0; index < segment.matches.size(); ++index) {
            const auto& match = segment.matches[index];
            if (match.tracePointIndex != index + 1) {
                throw std::invalid_argument("greedy replay overview match coverage is not contiguous");
            }
            MatchedTracePoint point{
                segment.tracePointsBase[match.tracePointIndex],
                match.matchedReferencePointBase,
                match.matchedReferenceArcBase,
            };
            validateMatchedPoint(point, selectedReference, replay.referenceBeginArcBase, replay.referenceEndArcBase, "greedy");
            if (point.referenceArcBase + kEpsilon < points.back().referenceArcBase) {
                throw std::invalid_argument("greedy replay overview match arcs are not monotonic");
            }
            points.push_back(point);
        }
        output.push_back(std::move(points));
    }
    return output;
}

std::vector<IndexedMatchedTraceSegment> matchedFiberletSegments(
    const FiberletGraphReplayResult& replay,
    const PolylineArcGeometry& selectedReference)
{
    std::vector<IndexedMatchedTraceSegment> output;
    for (size_t segmentIndex = 0; segmentIndex < replay.segments.size();
         ++segmentIndex) {
        const auto& segment = replay.segments[segmentIndex];
        if (segment.routePointsBaseXYZ.empty())
            continue;
        if (segment.matches.size() != segment.routePointsBaseXYZ.size()) {
            throw std::invalid_argument("fiberlet replay overview requires one match per route point");
        }
        MatchedTraceSegment points;
        points.reserve(segment.routePointsBaseXYZ.size());
        for (size_t index = 0; index < segment.matches.size(); ++index) {
            const auto& match = segment.matches[index];
            if (match.routePointIndex != index) {
                throw std::invalid_argument("fiberlet replay overview match coverage is not contiguous");
            }
            MatchedTracePoint point{
                segment.routePointsBaseXYZ[index],
                match.matchedReferencePointBaseXYZ,
                match.matchedReferenceArcBase,
            };
            validateMatchedPoint(point, selectedReference, replay.referenceBeginArcBase, replay.referenceEndArcBase, "fiberlet");
            if (!points.empty() && point.referenceArcBase + kEpsilon < points.back().referenceArcBase) {
                throw std::invalid_argument("fiberlet replay overview match arcs are not monotonic");
            }
            points.push_back(point);
        }
        output.push_back({segmentIndex, std::move(points)});
    }
    return output;
}

cv::Vec3d interpolateSurfacePoint(const cv::Mat_<cv::Vec3f>& surface, int row, double sourceColumn)
{
    const int left = std::clamp(static_cast<int>(std::floor(sourceColumn)), 0, surface.cols - 1);
    const int right = std::min(left + 1, surface.cols - 1);
    const double fraction = std::clamp(sourceColumn - left, 0.0, 1.0);
    const cv::Vec3f value = surface(row, left) * static_cast<float>(1.0 - fraction) + surface(row, right) * static_cast<float>(fraction);
    return {value[0], value[1], value[2]};
}

double referenceSourceColumn(
    const PolylineArcGeometry& selectedReference,
    double referenceBeginArc,
    double referenceArc)
{
    const auto referenceSample =
        samplePolylineArc(selectedReference, std::clamp(referenceArc - referenceBeginArc, 0.0, selectedReference.length()));
    const double segmentBegin = selectedReference.vertexArcs[referenceSample.segmentIndex];
    const double segmentEnd = selectedReference.vertexArcs[referenceSample.segmentIndex + 1];
    const double fraction = segmentEnd > segmentBegin + kEpsilon ? (referenceSample.arc - segmentBegin) / (segmentEnd - segmentBegin) : 0.0;
    return static_cast<double>(referenceSample.segmentIndex) + fraction;
}

cv::Point projectPointAtSourceColumn(
    const cv::Vec3d& pointBase, double sourceColumn,
    const cv::Mat_<cv::Vec3f>& surface, const cv::Size& renderedSize)
{
    const cv::Vec3d low = interpolateSurfacePoint(surface, 0, sourceColumn);
    const cv::Vec3d high = interpolateSurfacePoint(surface, surface.rows - 1, sourceColumn);
    const cv::Vec3d cross = high - low;
    const double crossLength = cv::norm(cross);
    if (!(crossLength > kEpsilon) || !std::isfinite(crossLength))
        throw std::runtime_error("replay overview strip has an invalid transverse axis");
    const cv::Vec3d center = interpolateSurfacePoint(
        surface, surface.rows / 2, sourceColumn);
    const double transverse = (pointBase - center).dot(cross / crossLength);
    const double sourceRow = static_cast<double>(surface.rows / 2) + transverse * (surface.rows - 1) / crossLength;
    const double pixelColumn = sourceColumn * (renderedSize.width - 1) / (surface.cols - 1);
    const double pixelRow = sourceRow * (renderedSize.height - 1) / (surface.rows - 1);
    if (!std::isfinite(pixelColumn) || !std::isfinite(pixelRow) || std::abs(pixelRow) > static_cast<double>(std::numeric_limits<int>::max() / 4)) {
        throw std::runtime_error("replay overview projection is not finite");
    }
    return {
        static_cast<int>(std::lround(pixelColumn)),
        static_cast<int>(std::lround(pixelRow)),
    };
}

cv::Point projectMatchedPoint(
    const MatchedTracePoint& point,
    const PolylineArcGeometry& selectedReference, double referenceBeginArc,
    const cv::Mat_<cv::Vec3f>& surface, const cv::Size& renderedSize)
{
    return projectPointAtSourceColumn(
        point.pointBase,
        referenceSourceColumn(
            selectedReference, referenceBeginArc, point.referenceArcBase),
        surface, renderedSize);
}

void drawPixels(
    cv::Mat_<cv::Vec3b>& image, const std::vector<cv::Point>& pixels,
    const cv::Vec3b& color)
{
    if (pixels.size() >= 2) {
        cv::polylines(
            image, pixels, false,
            cv::Scalar(color[0], color[1], color[2]), 2, cv::LINE_AA);
    } else if (pixels.size() == 1) {
        cv::circle(
            image, pixels.front(), 1,
            cv::Scalar(color[0], color[1], color[2]), cv::FILLED,
            cv::LINE_AA);
    }
}

void drawMatchedSegment(
    cv::Mat_<cv::Vec3b>& image, const MatchedTraceSegment& segment,
    const PolylineArcGeometry& selectedReference,
    double referenceBeginArc,
    const cv::Mat_<cv::Vec3f>& surface,
    const cv::Vec3b& color)
{
    std::vector<cv::Point> pixels;
    pixels.reserve(segment.size());
    for (const auto& point : segment) {
        pixels.push_back(projectMatchedPoint(
            point, selectedReference, referenceBeginArc, surface,
            image.size()));
    }
    drawPixels(image, pixels, color);
}

void drawMatchedSegments(
    cv::Mat_<cv::Vec3b>& image,
    const std::vector<MatchedTraceSegment>& segments,
    const PolylineArcGeometry& selectedReference,
    double referenceBeginArc,
    const cv::Mat_<cv::Vec3f>& surface,
    const cv::Vec3b& color)
{
    for (const auto& segment : segments)
        drawMatchedSegment(
            image, segment, selectedReference, referenceBeginArc, surface,
            color);
}

std::array<int, 2> markerBand(int center, int columns)
{
    const int begin = std::max(0, center - kOverviewMarkerWidthPixels / 2);
    const int end = std::min(columns, begin + kOverviewMarkerWidthPixels);
    return {std::max(0, end - kOverviewMarkerWidthPixels), end};
}

template <typename Failure>
std::vector<std::array<int, 2>> failureMarkerBands(
    const std::vector<Failure>& failures,
    const PolylineArcGeometry& selectedReference,
    double referenceBeginArc,
    int sourceColumns,
    int renderedColumns)
{
    std::vector<std::array<int, 2>> bands;
    bands.reserve(failures.size());
    for (const auto& failure : failures) {
        const double sourceColumn = referenceSourceColumn(
            selectedReference, referenceBeginArc, failure.referenceArcBase);
        const int pixelColumn = static_cast<int>(std::lround(
            sourceColumn * (renderedColumns - 1) / (sourceColumns - 1)));
        bands.push_back(markerBand(pixelColumn, renderedColumns));
    }
    return bands;
}

void drawFailureMarkers(
    cv::Mat_<cv::Vec3b>& image,
    const std::vector<std::array<int, 2>>& greedyBands,
    const std::vector<std::array<int, 2>>& fiberletBands)
{
    const auto drawBands = [&](const auto& bands, const cv::Vec3b& color) {
        for (const auto& band : bands) {
            if (band[1] > band[0]) {
                image(cv::Rect(band[0], 0, band[1] - band[0], image.rows)) =
                    color;
            }
        }
    };
    drawBands(greedyBands, kGreedyColorBgr);
    drawBands(fiberletBands, kFiberletColorBgr);
    for (const auto& greedy : greedyBands) {
        for (const auto& fiberlet : fiberletBands) {
            const int begin = std::max(greedy[0], fiberlet[0]);
            const int end = std::min(greedy[1], fiberlet[1]);
            if (end > begin) {
                image(cv::Rect(begin, 0, end - begin, image.rows)) =
                    kOverlapColorBgr;
            }
        }
    }
}

std::optional<double> segmentSourceColumnForReferenceArc(
    const MatchedTraceSegment& segment, double referenceArcBase)
{
    if (segment.empty() ||
        referenceArcBase < segment.front().referenceArcBase - kEpsilon ||
        referenceArcBase > segment.back().referenceArcBase + kEpsilon) {
        return std::nullopt;
    }
    if (referenceArcBase <= segment.front().referenceArcBase + kEpsilon)
        return 0.0;
    if (referenceArcBase >= segment.back().referenceArcBase - kEpsilon)
        return static_cast<double>(segment.size() - 1);
    const auto upper = std::upper_bound(
        segment.begin(), segment.end(), referenceArcBase,
        [](double arc, const MatchedTracePoint& point) {
            return arc < point.referenceArcBase;
        });
    const size_t right = static_cast<size_t>(upper - segment.begin());
    const size_t left = right - 1;
    const double begin = segment[left].referenceArcBase;
    const double end = segment[right].referenceArcBase;
    if (end <= begin + kEpsilon)
        return static_cast<double>(right);
    return static_cast<double>(left) +
        (referenceArcBase - begin) / (end - begin);
}

void drawMappedSegments(
    cv::Mat_<cv::Vec3b>& image,
    const std::vector<MatchedTraceSegment>& sourceSegments,
    const MatchedTraceSegment& fiberletFrame,
    const cv::Mat_<cv::Vec3f>& surface, const cv::Vec3b& color)
{
    for (const auto& source : sourceSegments) {
        std::vector<cv::Point> pixels;
        const auto flush = [&]() {
            drawPixels(image, pixels, color);
            pixels.clear();
        };
        for (const auto& point : source) {
            const auto column = segmentSourceColumnForReferenceArc(
                fiberletFrame, point.referenceArcBase);
            if (!column.has_value()) {
                flush();
                continue;
            }
            pixels.push_back(projectPointAtSourceColumn(
                point.pointBase, *column, surface, image.size()));
        }
        flush();
    }
}

void drawReferenceInFiberletFrame(
    cv::Mat_<cv::Vec3b>& image, const MatchedTraceSegment& fiberletFrame,
    const cv::Mat_<cv::Vec3f>& surface)
{
    std::vector<cv::Point> pixels;
    pixels.reserve(fiberletFrame.size());
    for (size_t index = 0; index < fiberletFrame.size(); ++index) {
        pixels.push_back(projectPointAtSourceColumn(
            fiberletFrame[index].referencePointBase,
            static_cast<double>(index), surface, image.size()));
    }
    drawPixels(image, pixels, kReferenceColorBgr);
}

template <typename Failure>
std::vector<std::array<int, 2>> mappedFailureMarkerBands(
    const std::vector<Failure>& failures,
    const MatchedTraceSegment& fiberletFrame, int sourceColumns,
    int renderedColumns, std::optional<size_t> requiredSegmentIndex)
{
    std::vector<std::array<int, 2>> bands;
    for (const auto& failure : failures) {
        if (requiredSegmentIndex.has_value() &&
            failure.segmentIndex != *requiredSegmentIndex) {
            continue;
        }
        const auto sourceColumn = segmentSourceColumnForReferenceArc(
            fiberletFrame, failure.referenceArcBase);
        if (!sourceColumn.has_value())
            continue;
        const int pixelColumn = static_cast<int>(std::lround(
            *sourceColumn * (renderedColumns - 1) /
            (sourceColumns - 1)));
        bands.push_back(markerBand(pixelColumn, renderedColumns));
    }
    return bands;
}

void drawCenterline(cv::Mat_<cv::Vec3b>& image, const cv::Vec3b& color)
{
    const int row = static_cast<int>(std::lround((image.rows - 1) / 2.0));
    cv::line(
        image, {0, row}, {image.cols - 1, row},
        cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);
}

FiberReplayOverview composeOverview(
    const cv::Mat_<cv::Vec3b>& referenceTop,
    const cv::Mat_<cv::Vec3b>& referenceSide,
    const cv::Mat_<cv::Vec3b>& fiberletTop,
    const cv::Mat_<cv::Vec3b>& fiberletSide,
    int maximumPageRows = kJpegMaximumDimension)
{
    if (referenceTop.empty() || referenceSide.empty() ||
        fiberletTop.empty() || fiberletSide.empty()) {
        throw std::invalid_argument("fiber replay overview strips are empty");
    }
    const int unwrappedColumns = std::max(
        {referenceTop.cols, referenceSide.cols, fiberletTop.cols,
         fiberletSide.cols});
    const int panelCount = std::max(
        1, (unwrappedColumns + kOverviewPanelColumns - 1) /
            kOverviewPanelColumns);
    const int panelBlockRows = 4 * kOverviewLabelRows +
        referenceTop.rows + referenceSide.rows + fiberletTop.rows +
        fiberletSide.rows + 3 * kOverviewSeparatorRows;
    if (maximumPageRows < 1 ||
        panelBlockRows + kOverviewHeaderRows > maximumPageRows) {
        throw std::invalid_argument(
            "one fiber replay overview block exceeds the JPEG height limit");
    }
    const int panelsPerPage = std::max(
        1, (maximumPageRows - kOverviewHeaderRows +
            kOverviewPanelGapRows) /
               (panelBlockRows + kOverviewPanelGapRows));
    const int pageCount =
        (panelCount + panelsPerPage - 1) / panelsPerPage;
    const auto columns = [&](const cv::Mat_<cv::Vec3b>& strip, int index) {
        return std::array<int, 2>{
            static_cast<int>(static_cast<int64_t>(index) * strip.cols /
                             panelCount),
            static_cast<int>(static_cast<int64_t>(index + 1) * strip.cols /
                             panelCount),
        };
    };

    FiberReplayOverview overview;
    overview.referenceTopShapeYX = {referenceTop.rows, referenceTop.cols};
    overview.referenceSideShapeYX = {referenceSide.rows, referenceSide.cols};
    overview.fiberletTopShapeYX = {fiberletTop.rows, fiberletTop.cols};
    overview.fiberletSideShapeYX = {fiberletSide.rows, fiberletSide.cols};
    overview.renderScale = kOverviewRenderScale;
    overview.markerWidthPixels = kOverviewMarkerWidthPixels;
    overview.fiberletComponentGapColumns =
        kOverviewFiberletComponentGapColumns;
    overview.pages.reserve(pageCount);
    for (int pageIndex = 0; pageIndex < pageCount; ++pageIndex) {
        const int firstPanel = pageIndex * panelsPerPage;
        const int endPanel = std::min(panelCount, firstPanel + panelsPerPage);
        const int pagePanels = endPanel - firstPanel;
        const int outputRows = kOverviewHeaderRows +
            pagePanels * panelBlockRows +
            (pagePanels - 1) * kOverviewPanelGapRows;
        int outputColumns = kOverviewMinimumColumns;
        for (int index = firstPanel; index < endPanel; ++index) {
            const auto referenceTopColumns = columns(referenceTop, index);
            const auto referenceSideColumns = columns(referenceSide, index);
            const auto fiberletTopColumns = columns(fiberletTop, index);
            const auto fiberletSideColumns = columns(fiberletSide, index);
            outputColumns = std::max(
                {outputColumns,
                 referenceTopColumns[1] - referenceTopColumns[0],
                 referenceSideColumns[1] - referenceSideColumns[0],
                 fiberletTopColumns[1] - fiberletTopColumns[0],
                 fiberletSideColumns[1] - fiberletSideColumns[0]});
        }
        if (outputColumns > kJpegMaximumDimension ||
            outputRows > maximumPageRows) {
            throw std::invalid_argument(
                "fiber replay overview page exceeds the JPEG dimension limit");
        }
        FiberReplayOverviewPage page;
        page.image = cv::Mat_<cv::Vec3b>(
            outputRows, outputColumns, cv::Vec3b{0, 0, 0});
        page.panels.reserve(pagePanels);
        const auto text = [&](const std::string& value, cv::Point origin,
                              const cv::Vec3b& color) {
            cv::putText(
                page.image, value, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);
        };
        text("Reference", {8, 14}, kReferenceColorBgr);
        text("Greedy", {8, 30}, kGreedyColorBgr);
        text("Fiberlet", {8, 46}, kFiberletColorBgr);
        int panelRow = kOverviewHeaderRows;
        for (int index = firstPanel; index < endPanel; ++index) {
            const auto referenceTopColumns = columns(referenceTop, index);
            const auto referenceSideColumns = columns(referenceSide, index);
            const auto fiberletTopColumns = columns(fiberletTop, index);
            const auto fiberletSideColumns = columns(fiberletSide, index);
            const int referenceTopRow = panelRow + kOverviewLabelRows;
            const int referenceSideRow = referenceTopRow + referenceTop.rows +
                kOverviewSeparatorRows + kOverviewLabelRows;
            const int fiberletTopRow = referenceSideRow + referenceSide.rows +
                kOverviewSeparatorRows + kOverviewLabelRows;
            const int fiberletSideRow = fiberletTopRow + fiberletTop.rows +
                kOverviewSeparatorRows + kOverviewLabelRows;
            const auto copy = [&](const cv::Mat_<cv::Vec3b>& source,
                                  const std::array<int, 2>& range, int row) {
                const int width = range[1] - range[0];
                if (width > 0) {
                    source(cv::Rect(range[0], 0, width, source.rows))
                        .copyTo(page.image(cv::Rect(0, row, width, source.rows)));
                }
            };
            copy(referenceTop, referenceTopColumns, referenceTopRow);
            copy(referenceSide, referenceSideColumns, referenceSideRow);
            copy(fiberletTop, fiberletTopColumns, fiberletTopRow);
            copy(fiberletSide, fiberletSideColumns, fiberletSideRow);
            const std::string suffix = panelCount == 1
                ? std::string()
                : " " + std::to_string(index + 1) + "/" +
                    std::to_string(panelCount);
            text("Reference top" + suffix, {8, referenceTopRow - 5},
                 cv::Vec3b{255, 255, 255});
            text("Reference side" + suffix, {8, referenceSideRow - 5},
                 cv::Vec3b{255, 255, 255});
            text("Fiberlet top" + suffix, {8, fiberletTopRow - 5},
                 cv::Vec3b{255, 255, 255});
            text("Fiberlet side" + suffix, {8, fiberletSideRow - 5},
                 cv::Vec3b{255, 255, 255});
            page.panels.push_back({
                static_cast<double>(index) / panelCount,
                static_cast<double>(index + 1) / panelCount,
                referenceTopColumns,
                referenceSideColumns,
                fiberletTopColumns,
                fiberletSideColumns,
                {referenceTopRow, referenceTopRow + referenceTop.rows},
                {referenceSideRow, referenceSideRow + referenceSide.rows},
                {fiberletTopRow, fiberletTopRow + fiberletTop.rows},
                {fiberletSideRow, fiberletSideRow + fiberletSide.rows},
            });
            panelRow += panelBlockRows + kOverviewPanelGapRows;
        }
        overview.pages.push_back(std::move(page));
    }
    return overview;
}

std::string encodeOverviewJpeg(const cv::Mat_<cv::Vec3b>& image)
{
    if (image.empty())
        throw std::invalid_argument("fiber replay overview image is empty");
    std::vector<uint8_t> encoded;
    if (!cv::imencode(".jpg", image, encoded, {cv::IMWRITE_JPEG_QUALITY, 95, cv::IMWRITE_JPEG_PROGRESSIVE, 0, cv::IMWRITE_JPEG_OPTIMIZE, 0})) {
        throw std::runtime_error("cannot encode fiber replay overview JPEG");
    }
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

std::string indexedOverviewName(std::string_view stem, size_t index)
{
    std::ostringstream output;
    output << stem << '.' << std::setfill('0') << std::setw(6) << index
           << ".jpg";
    return output.str();
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

void validateOverviewLayout(const FiberReplayOverview& overview)
{
    const auto validShape = [](const std::array<int, 2>& shape) {
        return shape[0] >= 2 && shape[1] >= 2;
    };
    if (!validShape(overview.referenceTopShapeYX) ||
        !validShape(overview.referenceSideShapeYX) ||
        !validShape(overview.fiberletTopShapeYX) ||
        !validShape(overview.fiberletSideShapeYX) ||
        overview.renderScale != kOverviewRenderScale ||
        overview.markerWidthPixels != kOverviewMarkerWidthPixels ||
        overview.fiberletComponentGapColumns !=
            kOverviewFiberletComponentGapColumns ||
        overview.pages.empty()) {
        throw std::invalid_argument(
            "fiber replay overview image or rendering metadata is invalid");
    }
    int expectedTopColumn = 0;
    int expectedSideColumn = 0;
    std::set<size_t> componentSegments;
    for (const auto& component : overview.fiberletComponents) {
        if (!std::isfinite(component.referenceArcBeginBase) ||
            !std::isfinite(component.referenceArcEndBase) ||
            component.referenceArcEndBase + kEpsilon <
                component.referenceArcBeginBase ||
            !componentSegments.insert(component.sourceSegmentIndex).second ||
            component.topColumns[0] != expectedTopColumn ||
            component.sideColumns[0] != expectedSideColumn ||
            component.topColumns[1] <= component.topColumns[0] ||
            component.sideColumns[1] <= component.sideColumns[0] ||
            component.topColumns[1] > overview.fiberletTopShapeYX[1] ||
            component.sideColumns[1] > overview.fiberletSideShapeYX[1] ||
            component.topRows[0] < 0 ||
            component.topRows[1] > overview.fiberletTopShapeYX[0] ||
            component.topRows[1] <= component.topRows[0] ||
            component.sideRows[0] < 0 ||
            component.sideRows[1] > overview.fiberletSideShapeYX[0] ||
            component.sideRows[1] <= component.sideRows[0]) {
            throw std::invalid_argument(
                "fiber replay overview component placement is invalid");
        }
        expectedTopColumn = component.topColumns[1] +
            kOverviewFiberletComponentGapColumns;
        expectedSideColumn = component.sideColumns[1] +
            kOverviewFiberletComponentGapColumns;
    }
    if (overview.fiberletComponents.empty()) {
        if (overview.fiberletTopShapeYX != std::array<int, 2>{2, 2} ||
            overview.fiberletSideShapeYX != std::array<int, 2>{2, 2}) {
            throw std::invalid_argument(
                "empty fiberlet overview must use the canonical blank raster");
        }
    } else if (
        expectedTopColumn - kOverviewFiberletComponentGapColumns !=
            overview.fiberletTopShapeYX[1] ||
        expectedSideColumn - kOverviewFiberletComponentGapColumns !=
            overview.fiberletSideShapeYX[1]) {
        throw std::invalid_argument(
            "fiber replay overview component columns are incomplete");
    }
    const int unwrappedColumns = std::max(
        {overview.referenceTopShapeYX[1],
         overview.referenceSideShapeYX[1],
         overview.fiberletTopShapeYX[1],
         overview.fiberletSideShapeYX[1]});
    const int panelCount = std::max(
        1, (unwrappedColumns + kOverviewPanelColumns - 1) /
            kOverviewPanelColumns);
    const int panelBlockRows = 4 * kOverviewLabelRows +
        overview.referenceTopShapeYX[0] +
        overview.referenceSideShapeYX[0] +
        overview.fiberletTopShapeYX[0] +
        overview.fiberletSideShapeYX[0] + 3 * kOverviewSeparatorRows;
    if (panelBlockRows + kOverviewHeaderRows > kJpegMaximumDimension)
        throw std::invalid_argument("fiber replay overview block is too tall");
    const int panelsPerPage = std::max(
        1, (kJpegMaximumDimension - kOverviewHeaderRows +
            kOverviewPanelGapRows) /
               (panelBlockRows + kOverviewPanelGapRows));
    const size_t expectedPages = static_cast<size_t>(
        (panelCount + panelsPerPage - 1) / panelsPerPage);
    if (overview.pages.size() != expectedPages)
        throw std::invalid_argument("fiber replay overview page count is invalid");
    int globalPanel = 0;
    for (size_t pageIndex = 0; pageIndex < overview.pages.size(); ++pageIndex) {
        const auto& page = overview.pages[pageIndex];
        const int pagePanels =
            std::min(panelsPerPage, panelCount - globalPanel);
        const int expectedRows = kOverviewHeaderRows +
            pagePanels * panelBlockRows +
            (pagePanels - 1) * kOverviewPanelGapRows;
        int expectedColumns = kOverviewMinimumColumns;
        int panelRow = kOverviewHeaderRows;
        if (page.image.empty() || page.image.type() != CV_8UC3 ||
            page.image.rows != expectedRows ||
            page.panels.size() != static_cast<size_t>(pagePanels)) {
            throw std::invalid_argument(
                "fiber replay overview page dimensions are invalid");
        }
        for (int localPanel = 0; localPanel < pagePanels;
             ++localPanel, ++globalPanel) {
            const auto range = [&](const std::array<int, 2>& shape) {
                return std::array<int, 2>{
                    static_cast<int>(static_cast<int64_t>(globalPanel) *
                                     shape[1] / panelCount),
                    static_cast<int>(static_cast<int64_t>(globalPanel + 1) *
                                     shape[1] / panelCount),
                };
            };
            const auto referenceTopColumns = range(overview.referenceTopShapeYX);
            const auto referenceSideColumns = range(overview.referenceSideShapeYX);
            const auto fiberletTopColumns = range(overview.fiberletTopShapeYX);
            const auto fiberletSideColumns = range(overview.fiberletSideShapeYX);
            const int referenceTopRow = panelRow + kOverviewLabelRows;
            const int referenceSideRow = referenceTopRow +
                overview.referenceTopShapeYX[0] + kOverviewSeparatorRows +
                kOverviewLabelRows;
            const int fiberletTopRow = referenceSideRow +
                overview.referenceSideShapeYX[0] + kOverviewSeparatorRows +
                kOverviewLabelRows;
            const int fiberletSideRow = fiberletTopRow +
                overview.fiberletTopShapeYX[0] + kOverviewSeparatorRows +
                kOverviewLabelRows;
            const auto& panel = page.panels[static_cast<size_t>(localPanel)];
            if (!nearlyEqual(
                    panel.progressFractionBegin,
                    static_cast<double>(globalPanel) / panelCount) ||
                !nearlyEqual(
                    panel.progressFractionEnd,
                    static_cast<double>(globalPanel + 1) / panelCount) ||
                panel.referenceTopColumns != referenceTopColumns ||
                panel.referenceSideColumns != referenceSideColumns ||
                panel.fiberletTopColumns != fiberletTopColumns ||
                panel.fiberletSideColumns != fiberletSideColumns ||
                panel.referenceTopRows != std::array<int, 2>{
                    referenceTopRow,
                    referenceTopRow + overview.referenceTopShapeYX[0]} ||
                panel.referenceSideRows != std::array<int, 2>{
                    referenceSideRow,
                    referenceSideRow + overview.referenceSideShapeYX[0]} ||
                panel.fiberletTopRows != std::array<int, 2>{
                    fiberletTopRow,
                    fiberletTopRow + overview.fiberletTopShapeYX[0]} ||
                panel.fiberletSideRows != std::array<int, 2>{
                    fiberletSideRow,
                    fiberletSideRow + overview.fiberletSideShapeYX[0]}) {
                throw std::invalid_argument(
                    "fiber replay overview panel geometry is invalid");
            }
            expectedColumns = std::max(
                {expectedColumns,
                 referenceTopColumns[1] - referenceTopColumns[0],
                 referenceSideColumns[1] - referenceSideColumns[0],
                 fiberletTopColumns[1] - fiberletTopColumns[0],
                 fiberletSideColumns[1] - fiberletSideColumns[0]});
            panelRow += panelBlockRows + kOverviewPanelGapRows;
        }
        if (page.image.cols != expectedColumns ||
            page.image.rows > kJpegMaximumDimension ||
            page.image.cols > kJpegMaximumDimension) {
            throw std::invalid_argument(
                "fiber replay overview composed dimensions are invalid");
        }
    }
    if (globalPanel != panelCount)
        throw std::invalid_argument("fiber replay overview panels are incomplete");
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

bool sameThresholdMeasurement(
    const FiberReplayThresholdMeasurement& left,
    const FiberReplayThresholdMeasurement& right)
{
    const auto sameOptional = [](const auto& first, const auto& second) {
        return first.has_value() == second.has_value() &&
            (!first.has_value() || nearlyEqual(*first, *second));
    };
    return nearlyEqual(
               left.euclideanErrorBaseVoxels,
               right.euclideanErrorBaseVoxels) &&
        sameOptional(
            left.normalErrorBaseVoxels,
            right.normalErrorBaseVoxels) &&
        sameOptional(
            left.tangentialErrorBaseVoxels,
            right.tangentialErrorBaseVoxels) &&
        nearlyEqual(
            left.thresholdErrorBaseVoxels,
            right.thresholdErrorBaseVoxels) &&
        nearlyEqual(left.thresholdErrorRatio, right.thresholdErrorRatio) &&
        left.localNormalValid == right.localNormalValid;
}

void validateGreedyThresholdData(
    const FiberReplayTraceResult& replay,
    double normalThresholdBaseVoxels)
{
    for (const auto& segment : replay.segments) {
        for (const auto& match : segment.matches) {
            if (match.tracePointIndex >= segment.tracePointsBase.size()) {
                throw std::invalid_argument(
                    "greedy replay threshold match index is invalid");
            }
            validateFiberReplayThresholdMeasurement(
                match.thresholdMeasurement,
                normalThresholdBaseVoxels);
            if (!nearlyEqual(
                    cv::norm(
                        segment.tracePointsBase[match.tracePointIndex] -
                        match.matchedReferencePointBase),
                    match.thresholdMeasurement.euclideanErrorBaseVoxels)) {
                throw std::invalid_argument(
                    "greedy replay threshold Euclidean error differs from geometry");
            }
        }
    }
    for (const auto& failure : replay.failures) {
        if (failure.thresholdMeasurement.has_value()) {
            validateFiberReplayThresholdMeasurement(
                *failure.thresholdMeasurement,
                normalThresholdBaseVoxels);
        }
        if (failure.reason != "distance_above_threshold")
            continue;
        if (!failure.thresholdMeasurement.has_value() ||
            !failure.segmentPointIndex.has_value() ||
            failure.segmentIndex >= replay.segments.size()) {
            throw std::invalid_argument(
                "greedy replay distance failure lacks threshold diagnostics");
        }
        const auto& matches = replay.segments[failure.segmentIndex].matches;
        const auto found = std::find_if(
            matches.begin(), matches.end(), [&](const auto& match) {
                return match.tracePointIndex == *failure.segmentPointIndex;
            });
        if (found == matches.end() ||
            !sameThresholdMeasurement(
                found->thresholdMeasurement,
                *failure.thresholdMeasurement)) {
            throw std::invalid_argument(
                "greedy replay distance failure differs from its terminal match");
        }
    }
}

void validateFiberletThresholdData(
    const FiberletGraphReplayResult& replay,
    double normalThresholdBaseVoxels)
{
    for (const auto& segment : replay.segments) {
        for (const auto& match : segment.matches) {
            if (match.routePointIndex >= segment.routePointsBaseXYZ.size()) {
                throw std::invalid_argument(
                    "fiberlet replay threshold match index is invalid");
            }
            validateFiberReplayThresholdMeasurement(
                match.thresholdMeasurement,
                normalThresholdBaseVoxels);
            if (!nearlyEqual(
                    cv::norm(
                        segment.routePointsBaseXYZ[match.routePointIndex] -
                        match.matchedReferencePointBaseXYZ),
                    match.thresholdMeasurement.euclideanErrorBaseVoxels)) {
                throw std::invalid_argument(
                    "fiberlet replay threshold Euclidean error differs from geometry");
            }
        }
    }
    for (const auto& failure : replay.failures) {
        if (failure.thresholdMeasurement.has_value()) {
            validateFiberReplayThresholdMeasurement(
                *failure.thresholdMeasurement,
                normalThresholdBaseVoxels);
        }
        if (failure.reason != "distance_above_threshold")
            continue;
        if (!failure.thresholdMeasurement.has_value() ||
            !failure.segmentPointIndex.has_value() ||
            failure.segmentIndex >= replay.segments.size()) {
            throw std::invalid_argument(
                "fiberlet replay distance failure lacks threshold diagnostics");
        }
        const auto& matches = replay.segments[failure.segmentIndex].matches;
        const auto found = std::find_if(
            matches.begin(), matches.end(), [&](const auto& match) {
                return match.routePointIndex == *failure.segmentPointIndex;
            });
        if (found == matches.end() ||
            !sameThresholdMeasurement(
                found->thresholdMeasurement,
                *failure.thresholdMeasurement)) {
            throw std::invalid_argument(
                "fiberlet replay distance failure differs from its terminal match");
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

nlohmann::json greedyReplayJson(
    const FiberReplayTraceResult& replay,
    double normalThresholdBaseVoxels)
{
    nlohmann::json root = {
        {"format", "vc_greedy_fiber_replay"},
        {"version", 2},
        {"reference_begin_arc_base", replay.referenceBeginArcBase},
        {"reference_end_arc_base", replay.referenceEndArcBase},
        {"completed_reference_arc_base", replay.completedReferenceArcBase},
        {"threshold", fiberReplayThresholdDescriptorJson(
             normalThresholdBaseVoxels)},
        {"segments", nlohmann::json::array()},
        {"failures", nlohmann::json::array()},
    };
    for (const auto& segment : replay.segments) {
        nlohmann::json matches = nlohmann::json::array();
        for (const auto& match : segment.matches) {
            validateFiberReplayThresholdMeasurement(
                match.thresholdMeasurement,
                normalThresholdBaseVoxels);
            auto matchJson = fiberReplayThresholdMeasurementJson(
                match.thresholdMeasurement);
            matchJson.update({
                {"trace_point_index", match.tracePointIndex},
                {"predicted_reference_arc_base", match.predictedReferenceArcBase},
                {"matched_reference_arc_base", match.matchedReferenceArcBase},
                {"matched_reference_point_base_xyz", pointJson(match.matchedReferencePointBase)},
                {"search_begin_arc_base", match.searchBeginArcBase},
                {"search_end_arc_base", match.searchEndArcBase},
            });
            matches.push_back(std::move(matchJson));
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
        root["failures"].push_back(failureJson(
            failure, normalThresholdBaseVoxels));
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

struct LineViewComponentRequest {
    size_t sourceSegmentIndex = 0;
    std::vector<cv::Vec3d> pointsBaseXYZ;
};

struct BuiltLineViewComponent {
    size_t sourceSegmentIndex = 0;
    std::vector<cv::Vec3d> pointsBaseXYZ;
    vc::lasagna::LineViewSurfaces surfaces;
};

std::vector<BuiltLineViewComponent> buildLineViewComponents(
    std::vector<LineViewComponentRequest> requests,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, int parallelThreads)
{
    if (!(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) || parallelThreads < 1) {
        throw std::invalid_argument(
            "line-view normal-sampling configuration is invalid");
    }
    std::vector<size_t> normalOffsets;
    normalOffsets.reserve(requests.size());
    std::vector<cv::Vec3d> workingPoints;
    for (const auto& request : requests) {
        if (request.pointsBaseXYZ.size() < 2)
            throw std::invalid_argument(
                "line-view component requires at least two points");
        normalOffsets.push_back(workingPoints.size());
        for (const auto& point : request.pointsBaseXYZ)
            workingPoints.push_back(point * (1.0 / normalWorkingToBaseScale));
    }
    std::vector<vc::lasagna::NormalSampleWithDerivative> normalSamples;
    normalSampler.sampleNormalBatch(
        workingPoints, false, parallelThreads, normalSamples);
    if (normalSamples.size() != workingPoints.size()) {
        throw std::runtime_error(
            "line-view normal sampler returned the wrong count");
    }

    std::vector<BuiltLineViewComponent> output;
    output.reserve(requests.size());
    for (size_t requestIndex = 0; requestIndex < requests.size();
         ++requestIndex) {
        auto& request = requests[requestIndex];
        vc::lasagna::LineModel line;
        line.points.reserve(request.pointsBaseXYZ.size());
        for (size_t pointIndex = 0;
             pointIndex < request.pointsBaseXYZ.size(); ++pointIndex) {
            line.points.push_back({
                request.pointsBaseXYZ[pointIndex],
                normalSamples[normalOffsets[requestIndex] + pointIndex].sample,
                true,
            });
        }
        auto surfaces = vc::lasagna::buildLineViewSurfaces(line);
        if (!surfaces.lineSurface || !surfaces.lineSideSlice) {
            throw std::runtime_error(
                "line view builder did not produce top and side surfaces");
        }
        output.push_back({
            request.sourceSegmentIndex,
            std::move(request.pointsBaseXYZ),
            std::move(surfaces),
        });
    }
    return output;
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

    enum class ComponentKind {
        Reference,
        Greedy,
        Fiberlet,
    };
    std::vector<LineViewComponentRequest> requests;
    std::vector<ComponentKind> kinds;
    const auto append = [&](ComponentKind kind,
                            const std::vector<std::vector<cv::Vec3d>>& source) {
        for (size_t index = 0; index < source.size(); ++index) {
            if (source[index].size() >= 2) {
                requests.push_back({index, source[index]});
                kinds.push_back(kind);
            }
        }
    };
    append(ComponentKind::Reference, {tube.referenceIntervalBase});
    append(ComponentKind::Greedy, greedy);
    append(ComponentKind::Fiberlet, fiberlet);
    auto built = buildLineViewComponents(
        std::move(requests), normalSampler, normalWorkingToBaseScale,
        parallelThreads);
    for (size_t index = 0; index < built.size(); ++index) {
        auto& component = built[index];
        FiberReplayStripComponent output;
        output.sourceSegmentIndex = component.sourceSegmentIndex;
        output.centerlineBaseXYZ = std::move(component.pointsBaseXYZ);
        output.lineSurface = std::move(component.surfaces.lineSurface);
        switch (kinds[index]) {
        case ComponentKind::Reference:
            result.reference.push_back(std::move(output));
            break;
        case ComponentKind::Greedy:
            result.greedy.push_back(std::move(output));
            break;
        case ComponentKind::Fiberlet:
            result.fiberlet.push_back(std::move(output));
            break;
        }
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

FiberReplayOverview renderFiberReplayOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& greedyReplay,
    const FiberletGraphReplayResult& fiberletReplay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    int parallelThreads,
    ::Volume& volume,
    const std::string& sourceLocator)
{
    if (referenceGeometryBase.size() < 2 || !(normalWorkingToBaseScale > 0.0) || !std::isfinite(normalWorkingToBaseScale) || parallelThreads < 1) {
        throw std::invalid_argument("replay overview geometry or normal-sampling configuration is invalid");
    }
    if (!nearlyEqual(greedyReplay.referenceBeginArcBase, fiberletReplay.referenceBeginArcBase) ||
        !nearlyEqual(greedyReplay.referenceEndArcBase, fiberletReplay.referenceEndArcBase)) {
        throw std::invalid_argument("replay overview trace intervals are inconsistent");
    }
    const auto selectedReference = makePolylineArcGeometry(referenceGeometryBase);
    if (!nearlyEqual(selectedReference.length(), greedyReplay.referenceEndArcBase - greedyReplay.referenceBeginArcBase)) {
        throw std::invalid_argument("replay overview reference geometry has the wrong arc extent");
    }
    validateReplayFailures(greedyReplay, "greedy");
    validateReplayFailures(fiberletReplay, "fiberlet");
    const auto greedy = matchedGreedySegments(greedyReplay, selectedReference);
    const auto fiberlet = matchedFiberletSegments(fiberletReplay, selectedReference);
    std::vector<LineViewComponentRequest> requests{
        {std::numeric_limits<size_t>::max(), referenceGeometryBase}};
    for (const auto& segment : fiberlet) {
        if (segment.points.size() < 2)
            continue;
        std::vector<cv::Vec3d> points;
        points.reserve(segment.points.size());
        for (const auto& point : segment.points)
            points.push_back(point.pointBase);
        requests.push_back({segment.sourceSegmentIndex, std::move(points)});
    }
    auto views = buildLineViewComponents(
        std::move(requests), normalSampler, normalWorkingToBaseScale,
        parallelThreads);
    if (views.empty())
        throw std::runtime_error("replay overview has no reference surfaces");
    const auto source = validateFiberReplayStripCtVolume(volume, sourceLocator);
    const auto renderSurface = [&](const QuadSurface& surface) {
        const auto* basePoints = surface.rawPointsPtr();
        if (!basePoints || basePoints->empty()) {
            throw std::invalid_argument("replay overview strip surface has no coordinates");
        }
        const auto groupPoints = nativeGroupTextureCoordinates(*basePoints, source);
        cv::Mat_<uint8_t> grayscale =
            vc::core::util::renderCoordsTextureFineToCoarse(groupPoints, volume, 0, kOverviewRenderScale, "Replay overview texture sampling");
        cv::Mat_<cv::Vec3b> color;
        cv::cvtColor(grayscale, color, cv::COLOR_GRAY2BGR);
        return color;
    };
    auto referenceTop = renderSurface(*views.front().surfaces.lineSurface);
    auto referenceSide = renderSurface(*views.front().surfaces.lineSideSlice);
    const auto draw = [&](cv::Mat_<cv::Vec3b>& image, const QuadSurface& surface) {
        const auto* points = surface.rawPointsPtr();
        drawMatchedSegments(image, greedy, selectedReference, greedyReplay.referenceBeginArcBase, *points, kGreedyColorBgr);
        for (const auto& segment : fiberlet) {
            drawMatchedSegment(
                image, segment.points, selectedReference,
                greedyReplay.referenceBeginArcBase, *points,
                kFiberletColorBgr);
        }
        const auto greedyBands = failureMarkerBands(
            greedyReplay.failures, selectedReference,
            greedyReplay.referenceBeginArcBase, points->cols, image.cols);
        const auto fiberletBands = failureMarkerBands(
            fiberletReplay.failures, selectedReference,
            greedyReplay.referenceBeginArcBase, points->cols, image.cols);
        drawFailureMarkers(image, greedyBands, fiberletBands);
        drawCenterline(image, kReferenceColorBgr);
    };
    draw(referenceTop, *views.front().surfaces.lineSurface);
    draw(referenceSide, *views.front().surfaces.lineSideSlice);

    struct RenderedFiberletComponent {
        size_t sourceSegmentIndex = 0;
        double referenceArcBeginBase = 0.0;
        double referenceArcEndBase = 0.0;
        cv::Mat_<cv::Vec3b> top;
        cv::Mat_<cv::Vec3b> side;
    };
    std::vector<RenderedFiberletComponent> renderedFiberlet;
    renderedFiberlet.reserve(views.size() - 1);
    for (size_t viewIndex = 1; viewIndex < views.size(); ++viewIndex) {
        auto& view = views[viewIndex];
        const auto match = std::find_if(
            fiberlet.begin(), fiberlet.end(), [&](const auto& segment) {
                return segment.sourceSegmentIndex == view.sourceSegmentIndex;
            });
        if (match == fiberlet.end() || match->points.size() < 2)
            throw std::logic_error("fiberlet overview view has no match metadata");
        auto top = renderSurface(*view.surfaces.lineSurface);
        auto side = renderSurface(*view.surfaces.lineSideSlice);
        const auto drawFiberletFrame = [&](cv::Mat_<cv::Vec3b>& image,
                                           const QuadSurface& surface) {
            const auto* points = surface.rawPointsPtr();
            drawMappedSegments(
                image, greedy, match->points, *points, kGreedyColorBgr);
            drawReferenceInFiberletFrame(image, match->points, *points);
            const auto greedyBands = mappedFailureMarkerBands(
                greedyReplay.failures, match->points, points->cols,
                image.cols, std::nullopt);
            const auto fiberletBands = mappedFailureMarkerBands(
                fiberletReplay.failures, match->points, points->cols,
                image.cols, view.sourceSegmentIndex);
            drawFailureMarkers(image, greedyBands, fiberletBands);
            drawCenterline(image, kFiberletColorBgr);
        };
        drawFiberletFrame(top, *view.surfaces.lineSurface);
        drawFiberletFrame(side, *view.surfaces.lineSideSlice);
        renderedFiberlet.push_back({
            view.sourceSegmentIndex,
            match->points.front().referenceArcBase,
            match->points.back().referenceArcBase,
            std::move(top),
            std::move(side),
        });
    }

    const auto assembledShape = [&](bool top) {
        int rows = 2;
        int64_t columns = 0;
        for (const auto& component : renderedFiberlet) {
            const auto& image = top ? component.top : component.side;
            rows = std::max(rows, image.rows);
            columns += image.cols;
        }
        if (!renderedFiberlet.empty()) {
            columns += static_cast<int64_t>(renderedFiberlet.size() - 1) *
                kOverviewFiberletComponentGapColumns;
        }
        if (columns > std::numeric_limits<int>::max())
            throw std::overflow_error("fiberlet overview raster is too wide");
        return cv::Size(
            std::max(2, static_cast<int>(columns)), rows);
    };
    const cv::Size fiberletTopSize = assembledShape(true);
    const cv::Size fiberletSideSize = assembledShape(false);
    cv::Mat_<cv::Vec3b> fiberletTop(
        fiberletTopSize, cv::Vec3b{0, 0, 0});
    cv::Mat_<cv::Vec3b> fiberletSide(
        fiberletSideSize, cv::Vec3b{0, 0, 0});
    std::vector<FiberReplayOverviewFiberletComponent> placements;
    placements.reserve(renderedFiberlet.size());
    int topColumn = 0;
    int sideColumn = 0;
    for (const auto& component : renderedFiberlet) {
        const int topRow = (fiberletTop.rows - component.top.rows) / 2;
        const int sideRow = (fiberletSide.rows - component.side.rows) / 2;
        component.top.copyTo(fiberletTop(cv::Rect(
            topColumn, topRow, component.top.cols, component.top.rows)));
        component.side.copyTo(fiberletSide(cv::Rect(
            sideColumn, sideRow, component.side.cols, component.side.rows)));
        placements.push_back({
            component.sourceSegmentIndex,
            component.referenceArcBeginBase,
            component.referenceArcEndBase,
            {topColumn, topColumn + component.top.cols},
            {sideColumn, sideColumn + component.side.cols},
            {topRow, topRow + component.top.rows},
            {sideRow, sideRow + component.side.rows},
        });
        topColumn += component.top.cols +
            kOverviewFiberletComponentGapColumns;
        sideColumn += component.side.cols +
            kOverviewFiberletComponentGapColumns;
    }

    auto overview = composeOverview(
        referenceTop, referenceSide, fiberletTop, fiberletSide);
    overview.textureSource = source;
    overview.fiberletComponents = std::move(placements);
    return overview;
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
    if (!nearlyEqual(
            input.request.errorThresholdBaseVoxels,
            input.fiberletReplayConfig.errorThresholdBaseVoxels)) {
        throw std::invalid_argument(
            "fiber replay engine thresholds differ");
    }
    validateReplayFailures(input.greedyReplay, "greedy");
    validateReplayFailures(input.fiberletReplay, "fiberlet");
    validateGreedyThresholdData(
        input.greedyReplay, input.request.errorThresholdBaseVoxels);
    validateFiberletThresholdData(
        input.fiberletReplay, input.request.errorThresholdBaseVoxels);

    std::vector<std::string> overviewJpegs;
    if (input.overview.has_value()) {
        const auto& overview = *input.overview;
        validateTextureSource(overview.textureSource);
        validateOverviewLayout(overview);
        size_t componentIndex = 0;
        for (size_t segmentIndex = 0;
             segmentIndex < input.fiberletReplay.segments.size();
             ++segmentIndex) {
            const auto& segment = input.fiberletReplay.segments[segmentIndex];
            if (segment.routePointsBaseXYZ.size() < 2)
                continue;
            if (segment.matches.size() != segment.routePointsBaseXYZ.size() ||
                componentIndex >= overview.fiberletComponents.size()) {
                throw std::invalid_argument(
                    "fiber replay overview component coverage differs from the fiberlet replay");
            }
            const auto& component =
                overview.fiberletComponents[componentIndex++];
            if (component.sourceSegmentIndex != segmentIndex ||
                !nearlyEqual(
                    component.referenceArcBeginBase,
                    segment.matches.front().matchedReferenceArcBase) ||
                !nearlyEqual(
                    component.referenceArcEndBase,
                    segment.matches.back().matchedReferenceArcBase)) {
                throw std::invalid_argument(
                    "fiber replay overview component identity is inconsistent");
            }
        }
        if (componentIndex != overview.fiberletComponents.size()) {
            throw std::invalid_argument(
                "fiber replay overview has unexpected fiberlet components");
        }
        overviewJpegs.reserve(overview.pages.size());
        for (const auto& page : overview.pages)
            overviewJpegs.push_back(encodeOverviewJpeg(page.image));
    }

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

    const auto greedyJson = greedyReplayJson(
        input.greedyReplay, input.request.errorThresholdBaseVoxels);
    const auto fiberletJson = fiberletGraphReplayJson(input.fiberletReplay, input.fiberletReplayConfig);
    vc::core::util::atomicWriteString(staging / "replay/reference.obj", lineObj("# vc_fiber_replay_reference version 2", input.referenceGeometryBase));
    vc::core::util::atomicWriteString(staging / "replay/greedy.json", greedyJson.dump(2) + "\n");
    vc::core::util::
        atomicWriteString(staging / "replay/greedy.obj", segmentedLineObj("# vc_greedy_fiber_replay version 2", greedySegments(input.greedyReplay)));
    vc::core::util::atomicWriteString(staging / "replay/fiberlet.json", fiberletJson.dump(2) + "\n");
    vc::core::util::atomicWriteString(staging / "replay/fiberlet.obj", fiberletGraphReplayObj(input.fiberletReplay));
    for (size_t index = 0; index < overviewJpegs.size(); ++index) {
        vc::core::util::atomicWriteString(
            staging / "replay" /
                indexedOverviewName("full_strip", index),
            overviewJpegs[index]);
    }

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
            {"failure", failureJson(
                 failure, input.request.errorThresholdBaseVoxels)},
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
        {"threshold", fiberReplayThresholdDescriptorJson(
             input.request.errorThresholdBaseVoxels)},
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
    std::vector<std::filesystem::path> rootArtifacts{
        "replay/reference.obj",
        "replay/greedy.json",
        "replay/greedy.obj",
        "replay/fiberlet.json",
        "replay/fiberlet.obj",
    };
    for (size_t index = 0; index < overviewJpegs.size(); ++index) {
        rootArtifacts.emplace_back(
            std::filesystem::path("replay") /
            indexedOverviewName("full_strip", index));
    }
    for (const auto& relative : rootArtifacts) {
        root["artifacts"][relative.generic_string()] = {
            {"path", (generationRelative / relative).generic_string()},
            {"content_hash", artifactHash(finalGeneration / relative)},
        };
    }
    std::set<std::string> publishedOverviewAliases;
    if (input.overview.has_value()) {
        const auto& overview = *input.overview;
        nlohmann::json pages = nlohmann::json::array();
        size_t globalPanelIndex = 0;
        for (size_t pageIndex = 0; pageIndex < overview.pages.size();
             ++pageIndex) {
            const auto& page = overview.pages[pageIndex];
            const auto relative = std::filesystem::path("replay") /
                indexedOverviewName("full_strip", pageIndex);
            const std::string stable =
                indexedOverviewName("fiber_replay", pageIndex);
            nlohmann::json panels = nlohmann::json::array();
            for (const auto& panel : page.panels) {
                panels.push_back({
                    {"index", globalPanelIndex++},
                    {"progress_fraction_begin", panel.progressFractionBegin},
                    {"progress_fraction_end", panel.progressFractionEnd},
                    {"reference_top_columns", panel.referenceTopColumns},
                    {"reference_side_columns", panel.referenceSideColumns},
                    {"fiberlet_top_columns", panel.fiberletTopColumns},
                    {"fiberlet_side_columns", panel.fiberletSideColumns},
                    {"reference_top_rows", panel.referenceTopRows},
                    {"reference_side_rows", panel.referenceSideRows},
                    {"fiberlet_top_rows", panel.fiberletTopRows},
                    {"fiberlet_side_rows", panel.fiberletSideRows},
                });
            }
            pages.push_back({
                {"index", pageIndex},
                {"artifact", root["artifacts"].at(relative.generic_string())},
                {"stable_path", stable},
                {"image_shape_yx", {page.image.rows, page.image.cols}},
                {"panels", std::move(panels)},
            });
            vc::core::util::atomicWriteString(
                outputDirectory / stable, overviewJpegs[pageIndex]);
            publishedOverviewAliases.insert(stable);
        }
        nlohmann::json components = nlohmann::json::array();
        for (const auto& component : overview.fiberletComponents) {
            components.push_back({
                {"source_segment_index", component.sourceSegmentIndex},
                {"reference_arc_begin_base", component.referenceArcBeginBase},
                {"reference_arc_end_base", component.referenceArcEndBase},
                {"top_columns", component.topColumns},
                {"side_columns", component.sideColumns},
                {"top_rows", component.topRows},
                {"side_rows", component.sideRows},
            });
        }
        root["overview"] = {
            {"reference_begin_arc_base", beginArc},
            {"reference_end_arc_base", endArc},
            {"reference_point_count", input.referenceGeometryBase.size()},
            {"ct_source",
             {
                 {"locator", overview.textureSource.locator},
                 {"shape_zyx", overview.textureSource.shapeZYX},
                 {"scale_from_base_xyz", overview.textureSource.scaleFromBaseXYZ},
                 {"offset_from_base_xyz", overview.textureSource.offsetFromBaseXYZ},
             }},
            {"render_scale", overview.renderScale},
            {"reference_top_shape_yx", overview.referenceTopShapeYX},
            {"reference_side_shape_yx", overview.referenceSideShapeYX},
            {"fiberlet_top_shape_yx", overview.fiberletTopShapeYX},
            {"fiberlet_side_shape_yx", overview.fiberletSideShapeYX},
            {"fiberlet_components",
             {
                 {"gap_columns", overview.fiberletComponentGapColumns},
                 {"gap_fill", "black"},
                 {"placements", std::move(components)},
             }},
            {"pages", std::move(pages)},
            {"layout",
             {
                 {"order", {"reference_top", "reference_side", "fiberlet_top", "fiberlet_side"}},
                 {"header_rows", kOverviewHeaderRows},
                 {"label_rows", kOverviewLabelRows},
                 {"separator_rows", kOverviewSeparatorRows},
                 {"panel_gap_rows", kOverviewPanelGapRows},
                 {"maximum_panel_columns", kOverviewPanelColumns},
                 {"maximum_image_dimension", kJpegMaximumDimension},
                 {"panel_count", globalPanelIndex},
                 {"page_count", overview.pages.size()},
                 {"alignment", "left"},
                 {"padding", "black"},
             }},
            {"overlay_colors_rgb",
             {
                 {"reference", {255, 255, 0}},
                 {"greedy", {255, 0, 0}},
                 {"fiberlet", {0, 255, 255}},
             }},
            {"failure_markers",
             {
                 {"semantic", "pre_reset_error"},
                 {"reference_arc_field", "failure_reference_arc"},
                 {"reset_seed_markers", false},
                 {"width_pixels", overview.markerWidthPixels},
                 {"greedy_color_rgb", {255, 0, 0}},
                 {"fiberlet_color_rgb", {0, 255, 255}},
                 {"overlap_color_rgb", {255, 0, 255}},
                 {"greedy_count", input.greedyReplay.failures.size()},
                 {"fiberlet_count", input.fiberletReplay.failures.size()},
             }},
        };
    }
    vc::core::util::atomicWriteString(outputDirectory / "fiber_replay.json", root.dump(2) + "\n");
    constexpr std::string_view kVisualizationPrefix =
        "fiber_replay_visualization.";
    constexpr std::string_view kOverviewPrefix = "fiber_replay.";
    for (const auto& entry : std::filesystem::directory_iterator(outputDirectory)) {
        if (!entry.is_regular_file())
            continue;
        const std::string name = entry.path().filename().string();
        if (name.starts_with(kVisualizationPrefix) &&
            !publishedVisualizationAliases.contains(name)) {
            std::filesystem::remove(entry.path());
        } else if (name.starts_with(kOverviewPrefix) &&
                   name.ends_with(".jpg") &&
                   !publishedOverviewAliases.contains(name)) {
            std::filesystem::remove(entry.path());
        }
    }
    return root;
}

#ifdef VC_TESTING
namespace testing
{
FiberReplayOverview composeFiberReplayOverviewForTesting(
    const cv::Mat_<cv::Vec3b>& referenceTop,
    const cv::Mat_<cv::Vec3b>& referenceSide,
    const cv::Mat_<cv::Vec3b>& fiberletTop,
    const cv::Mat_<cv::Vec3b>& fiberletSide,
    int maximumPageRows)
{
    return composeOverview(
        referenceTop, referenceSide, fiberletTop, fiberletSide,
        maximumPageRows);
}
}  // namespace testing
#endif

}  // namespace vc::fiber_tracer
