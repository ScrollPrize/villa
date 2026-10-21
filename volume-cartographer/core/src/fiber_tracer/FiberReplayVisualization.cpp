#include "vc/fiber_tracer/FiberReplayVisualization.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceTexture.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <opencv2/imgproc.hpp>
namespace vc::fiber_tracer {
namespace {
constexpr double kEpsilon = 1.0e-12;
nlohmann::json pointJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
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
    const FiberReplayStripTextureSource& source, double baseSamplesPerPixel = 0,
    int longitudinalColumns = 0)
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
    const auto& metricPoints = baseSamplesPerPixel > 0 ? basePoints : groupPoints;
    const double spacing = baseSamplesPerPixel > 0 ? baseSamplesPerPixel : 1;
    double maximumRowArc = 0.0;
    for (int column = 0; column < groupPoints.cols; ++column) {
        double arc = 0.0;
        for (int row = 1; row < groupPoints.rows; ++row)
            arc += distance(metricPoints(row - 1, column), metricPoints(row, column)) / spacing;
        maximumRowArc = std::max(maximumRowArc, arc);
    }
    double maximumColumnArc = 0.0;
    for (int row = 0; row < groupPoints.rows; ++row) {
        double arc = 0.0;
        for (int column = 1; column < groupPoints.cols; ++column)
            arc += distance(metricPoints(row, column - 1), metricPoints(row, column)) / spacing;
        maximumColumnArc = std::max(maximumColumnArc, arc);
    }
    if (baseSamplesPerPixel > 0) {
        // Inspection's horizontal axis follows the fiber, not an offset ribbon
        // edge (whose length changes with curvature and viewing plane).
        const int lo = (basePoints.rows - 1) / 2;
        const int hi = basePoints.rows / 2;
        maximumColumnArc = 0;
        for (int column = 1; column < basePoints.cols; ++column)
            maximumColumnArc += distance(
                (basePoints(lo, column - 1) + basePoints(hi, column - 1)) * 0.5F,
                (basePoints(lo, column) + basePoints(hi, column)) * 0.5F) / spacing;
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
        longitudinalColumns > 0 ? longitudinalColumns : sampleCount(maximumColumnArc),
        sampleCount(maximumRowArc));
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
    const vc::lasagna::LineStripPositionMap& positionMap,
    const cv::Mat_<cv::Vec3f>& surface, const cv::Size& renderedSize)
{
    sourceColumn = positionMap.originalPositionToStripGridColumn(sourceColumn);
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
    const vc::lasagna::LineStripPositionMap& positionMap,
    const cv::Mat_<cv::Vec3f>& surface, const cv::Size& renderedSize)
{
    return projectPointAtSourceColumn(
        point.pointBase,
        referenceSourceColumn(
            selectedReference, referenceBeginArc, point.referenceArcBase),
        positionMap, surface, renderedSize);
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
    const vc::lasagna::LineStripPositionMap& positionMap,
    const cv::Mat_<cv::Vec3f>& surface,
    const cv::Vec3b& color)
{
    std::vector<cv::Point> pixels;
    pixels.reserve(segment.size());
    for (const auto& point : segment) {
        pixels.push_back(projectMatchedPoint(
            point, selectedReference, referenceBeginArc, positionMap, surface,
            image.size()));
    }
    drawPixels(image, pixels, color);
}

void drawMatchedSegments(
    cv::Mat_<cv::Vec3b>& image,
    const std::vector<MatchedTraceSegment>& segments,
    const PolylineArcGeometry& selectedReference,
    double referenceBeginArc,
    const vc::lasagna::LineStripPositionMap& positionMap,
    const cv::Mat_<cv::Vec3f>& surface,
    const cv::Vec3b& color)
{
    for (const auto& segment : segments)
        drawMatchedSegment(
            image, segment, selectedReference, referenceBeginArc, positionMap, surface,
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
    const vc::lasagna::LineStripPositionMap& positionMap,
    int sourceColumns,
    int renderedColumns)
{
    std::vector<std::array<int, 2>> bands;
    bands.reserve(failures.size());
    for (const auto& failure : failures) {
        const double sourceColumn = positionMap.originalPositionToStripGridColumn(referenceSourceColumn(
            selectedReference, referenceBeginArc, failure.referenceArcBase));
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
    const vc::lasagna::LineStripPositionMap& positionMap,
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
                point.pointBase, *column, positionMap, surface, image.size()));
        }
        flush();
    }
}

void drawReferenceInFiberletFrame(
    cv::Mat_<cv::Vec3b>& image, const MatchedTraceSegment& fiberletFrame,
    const vc::lasagna::LineStripPositionMap& positionMap,
    const cv::Mat_<cv::Vec3f>& surface)
{
    std::vector<cv::Point> pixels;
    pixels.reserve(fiberletFrame.size());
    for (size_t index = 0; index < fiberletFrame.size(); ++index) {
        pixels.push_back(projectPointAtSourceColumn(
            fiberletFrame[index].referencePointBase,
            static_cast<double>(index), positionMap, surface, image.size()));
    }
    drawPixels(image, pixels, kReferenceColorBgr);
}

template <typename Failure>
std::vector<std::array<int, 2>> mappedFailureMarkerBands(
    const std::vector<Failure>& failures,
    const vc::lasagna::LineStripPositionMap& positionMap,
    const MatchedTraceSegment& fiberletFrame, int sourceColumns,
    int renderedColumns, std::optional<size_t> requiredSegmentIndex,
    bool ownTraceEndpoint = false)
{
    std::vector<std::array<int, 2>> bands;
    for (const auto& failure : failures) {
        if (requiredSegmentIndex.has_value() &&
            failure.segmentIndex != *requiredSegmentIndex) {
            continue;
        }
        const auto sourceColumn = ownTraceEndpoint
            ? std::optional<double>(double(fiberletFrame.size() - 1))
            : segmentSourceColumnForReferenceArc(fiberletFrame, failure.referenceArcBase);
        if (!sourceColumn.has_value())
            continue;
        const int pixelColumn = static_cast<int>(std::lround(
            positionMap.originalPositionToStripGridColumn(*sourceColumn) * (renderedColumns - 1) /
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
    int maximumPageRows = kJpegMaximumDimension,
    bool nativeTraceOnly = false)
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
        if (!nativeTraceOnly) text("Greedy", {8, 30}, kGreedyColorBgr);
        text(nativeTraceOnly ? "Native trace" : "Fiberlet", {8, 46}, kFiberletColorBgr);
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
            text(std::string(nativeTraceOnly ? "Traced top" : "Fiberlet top") + suffix, {8, fiberletTopRow - 5},
                 cv::Vec3b{255, 255, 255});
            text(std::string(nativeTraceOnly ? "Traced side" : "Fiberlet side") + suffix, {8, fiberletSideRow - 5},
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
    // Per-sample validity is consumed by the shared line-view builder below.
    (void)normalSampler.sampleNormalBatch(
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
namespace {
template <typename Failure>
FiberReplayOverview renderReplayOverviewCommon(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    double referenceBeginArcBase,
    const std::vector<MatchedTraceSegment>& greedy,
    const std::vector<IndexedMatchedTraceSegment>& fiberlet,
    const std::vector<FiberReplayFailure>& greedyFailures,
    const std::vector<Failure>& fiberletFailures,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, int parallelThreads,
    const FiberReplayStripTextureSource& source,
    const FiberReplayTextureRenderer& renderTexture,
    bool nativeTraceOnly = false, int renderScale = kOverviewRenderScale,
    bool labelRestarts = false, double baseSamplesPerPixel = 0)
{
    const auto selectedReference = makePolylineArcGeometry(referenceGeometryBase);
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
    validateTextureSource(source);
    const auto renderSurface = [&](const QuadSurface& surface, int longitudinalColumns = 0) {
        const auto* basePoints = surface.rawPointsPtr();
        if (!basePoints || basePoints->empty()) {
            throw std::invalid_argument("replay overview strip surface has no coordinates");
        }
        const auto groupPoints = nativeGroupTextureCoordinates(*basePoints, source, baseSamplesPerPixel, longitudinalColumns);
        cv::Mat_<uint8_t> grayscale = renderTexture(groupPoints, renderScale);
        if (grayscale.rows != groupPoints.rows * renderScale ||
            grayscale.cols != groupPoints.cols * renderScale)
            throw std::runtime_error("overview texture callback returned wrong raster dimensions");
        cv::Mat_<cv::Vec3b> color;
        cv::cvtColor(grayscale, color, cv::COLOR_GRAY2BGR);
        return color;
    };
    auto referenceTop = renderSurface(*views.front().surfaces.lineSurface);
    auto referenceSide = renderSurface(*views.front().surfaces.lineSideSlice,
        baseSamplesPerPixel > 0 ? referenceTop.cols / renderScale : 0);
    const auto draw = [&](cv::Mat_<cv::Vec3b>& image, const QuadSurface& surface) {
        const auto* points = surface.rawPointsPtr();
        const auto& positionMap = views.front().surfaces.stripPositionMap;
        drawMatchedSegments(image, greedy, selectedReference, referenceBeginArcBase, positionMap, *points, kGreedyColorBgr);
        for (const auto& segment : fiberlet) {
            drawMatchedSegment(
                image, segment.points, selectedReference,
                referenceBeginArcBase, positionMap, *points,
                kFiberletColorBgr);
        }
        const auto greedyBands = failureMarkerBands(
            greedyFailures, selectedReference,
            referenceBeginArcBase, positionMap, points->cols, image.cols);
        const auto fiberletBands = failureMarkerBands(
            fiberletFailures, selectedReference,
            referenceBeginArcBase, positionMap, points->cols, image.cols);
        drawFailureMarkers(image, greedyBands, fiberletBands);
        if (labelRestarts) {
            for (size_t i = 0; i < fiberletBands.size(); ++i) {
                const auto label = "R" + std::to_string(fiberletFailures[i].index);
                cv::putText(image, label, {fiberletBands[i][0], 14 + static_cast<int>(i % 3) * 16},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
        }
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
        auto side = renderSurface(*view.surfaces.lineSideSlice,
            baseSamplesPerPixel > 0 ? top.cols / renderScale : 0);
        const auto drawFiberletFrame = [&](cv::Mat_<cv::Vec3b>& image,
                                           const QuadSurface& surface) {
            const auto* points = surface.rawPointsPtr();
            const auto& positionMap = view.surfaces.stripPositionMap;
            drawMappedSegments(
                image, greedy, match->points, positionMap, *points, kGreedyColorBgr);
            drawReferenceInFiberletFrame(image, match->points, positionMap, *points);
            const auto greedyBands = mappedFailureMarkerBands(
                greedyFailures, positionMap, match->points, points->cols,
                image.cols, std::nullopt);
            const auto fiberletBands = mappedFailureMarkerBands(
                fiberletFailures, positionMap, match->points, points->cols,
                image.cols, view.sourceSegmentIndex, nativeTraceOnly);
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
        auto& placement = placements.back();
        const auto view = std::find_if(views.begin(), views.end(), [&](const auto& v) {
            return v.sourceSegmentIndex == component.sourceSegmentIndex;
        });
        const auto match = std::find_if(fiberlet.begin(), fiberlet.end(), [&](const auto& v) {
            return v.sourceSegmentIndex == component.sourceSegmentIndex;
        });
        placement.positionMap = view->surfaces.stripPositionMap;
        placement.topSourceColumns = view->surfaces.lineSurface->rawPointsPtr()->cols;
        for (const auto& point : match->points) {
            placement.pointsBase.push_back(point.pointBase);
            placement.referenceArcsBase.push_back(point.referenceArcBase);
        }
        topColumn += component.top.cols +
            kOverviewFiberletComponentGapColumns;
        sideColumn += component.side.cols +
            kOverviewFiberletComponentGapColumns;
    }

    // Label the assembled raster: individual components can be narrower than
    // the text. Use segment identity, not reference-arc matching across breaks.
    if (labelRestarts) {
        for (const auto& failure : fiberletFailures) {
            for (int next = 0; next < 2; ++next) {
                const auto placement = std::find_if(placements.begin(), placements.end(),
                    [&](const auto& p) { return p.sourceSegmentIndex == failure.segmentIndex + next; });
                if (placement == placements.end()) continue; // No strip for a singleton.
                const auto label = "R" + std::to_string(failure.index) + (next ? "+" : "-");
                int baseline = 0;
                const auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
                for (int side = 0; side < 2; ++side) {
                    auto& image = side ? fiberletSide : fiberletTop;
                    const auto columns = side ? placement->sideColumns : placement->topColumns;
                    const int anchor = next ? columns[0] : columns[1] - 1;
                    const int x = std::clamp(next ? anchor : anchor - size.width,
                                             0, std::max(0, image.cols - size.width));
                    const int y = std::min(image.rows - baseline - 1,
                        14 + static_cast<int>((2 * failure.index + next) % 3) * 16);
                    cv::putText(image, label, {x, y}, cv::FONT_HERSHEY_SIMPLEX,
                        0.4, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
                }
            }
        }
    }
    auto overview = composeOverview(
        referenceTop, referenceSide, fiberletTop, fiberletSide,
        renderScale == kOverviewRenderScale ? kJpegMaximumDimension : 12000, nativeTraceOnly);
    overview.renderScale = renderScale;
    overview.textureSource = source;
    overview.fiberletComponents = std::move(placements);
    return overview;
}

} // namespace

FiberReplayOverview renderFiberReplayOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& greedyReplay,
    const FiberletGraphReplayResult& fiberletReplay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, int parallelThreads,
    ::Volume& volume, const std::string& sourceLocator)
{
    if (referenceGeometryBase.size() < 2 || !(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) || parallelThreads < 1)
        throw std::invalid_argument("replay overview geometry or normal-sampling configuration is invalid");
    if (!nearlyEqual(greedyReplay.referenceBeginArcBase, fiberletReplay.referenceBeginArcBase) ||
        !nearlyEqual(greedyReplay.referenceEndArcBase, fiberletReplay.referenceEndArcBase))
        throw std::invalid_argument("replay overview trace intervals are inconsistent");
    const auto reference = makePolylineArcGeometry(referenceGeometryBase);
    if (!nearlyEqual(reference.length(), greedyReplay.referenceEndArcBase - greedyReplay.referenceBeginArcBase))
        throw std::invalid_argument("replay overview reference geometry has the wrong arc extent");
    validateReplayFailures(greedyReplay, "greedy");
    validateReplayFailures(fiberletReplay, "fiberlet");
    return renderReplayOverviewCommon(referenceGeometryBase, greedyReplay.referenceBeginArcBase,
        matchedGreedySegments(greedyReplay, reference), matchedFiberletSegments(fiberletReplay, reference),
        greedyReplay.failures, fiberletReplay.failures, normalSampler,
        normalWorkingToBaseScale, parallelThreads,
        validateFiberReplayStripCtVolume(volume, sourceLocator),
        [&](const auto& coords, int scale) -> cv::Mat_<uint8_t> {
            return vc::core::util::renderCoordsTextureFineToCoarse(
                coords, volume, 0, scale, "Replay overview CT sampling");
        });
}

FiberReplayOverview renderFiberReplayTraceOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& replay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, int parallelThreads,
    ::Volume& volume, const std::string& sourceLocator)
{
    return renderFiberReplayTraceOverview(referenceGeometryBase, replay, normalSampler,
        normalWorkingToBaseScale, parallelThreads,
        validateFiberReplayStripCtVolume(volume, sourceLocator),
        [&](const auto& coords, int scale) -> cv::Mat_<uint8_t> {
            return vc::core::util::renderCoordsTextureFineToCoarse(
                coords, volume, 0, scale, "Replay overview CT sampling");
        });
}

FiberReplayOverview renderFiberReplayTraceOverview(
    const std::vector<cv::Vec3d>& referenceGeometryBase,
    const FiberReplayTraceResult& replay,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, int parallelThreads,
    const FiberReplayStripTextureSource& source,
    const FiberReplayTextureRenderer& renderTexture, int renderScale, bool labelRestarts, double baseSamplesPerPixel)
{
    if (renderScale < 1 || referenceGeometryBase.size() < 2 || !(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale) || parallelThreads < 1)
        throw std::invalid_argument("replay overview geometry or normal-sampling configuration is invalid");
    const auto reference = makePolylineArcGeometry(referenceGeometryBase);
    if (!nearlyEqual(reference.length(), replay.referenceEndArcBase - replay.referenceBeginArcBase))
        throw std::invalid_argument("replay overview reference geometry has the wrong arc extent");
    validateReplayFailures(replay, "native");
    const auto matches = matchedGreedySegments(replay, reference);
    std::vector<IndexedMatchedTraceSegment> indexed;
    size_t matchIndex = 0;
    for (size_t i = 0; i < replay.segments.size(); ++i)
        if (!replay.segments[i].tracePointsBase.empty())
            indexed.push_back({i, matches.at(matchIndex++)});
    // One native trace occupies the existing traced-frame rows. No fabricated
    // graph result and no duplicate greedy overlay.
    return renderReplayOverviewCommon(referenceGeometryBase, replay.referenceBeginArcBase,
        {}, indexed, {}, replay.failures, normalSampler,
        normalWorkingToBaseScale, parallelThreads, source, renderTexture, true, renderScale, labelRestarts, baseSamplesPerPixel);
}

cv::Mat_<uint8_t> renderFiberReplayPresenceTexture(
    const cv::Mat_<cv::Vec3f>& sourceCoordinates, int renderScale,
    const FiberReplayStripTextureSource& source, double workingToBaseScale,
    const FiberPredictionSource& predictions)
{
    validateTextureSource(source);
    if (sourceCoordinates.empty() || renderScale < 1 ||
        !std::isfinite(workingToBaseScale) || workingToBaseScale <= 0)
        throw std::invalid_argument("invalid presence texture coordinates/scale");
    cv::Mat_<cv::Vec3f> coords;
    cv::resize(sourceCoordinates, coords,
        cv::Size(sourceCoordinates.cols * renderScale, sourceCoordinates.rows * renderScale),
        0, 0, cv::INTER_LINEAR);
    cv::Mat_<uint8_t> image(coords.rows, coords.cols, uint8_t(0));
    // Bounded requests avoid materializing a full strip's interpolation data.
    constexpr size_t batchSize = 4096;
    for (size_t begin = 0; begin < coords.total(); begin += batchSize) {
        const size_t count = std::min(batchSize, coords.total() - begin);
        std::vector<cv::Vec3d> points(count), directions(count, cv::Vec3d{1, 0, 0});
        for (size_t i = 0; i < count; ++i)
            for (int axis = 0; axis < 3; ++axis)
                points[i][axis] = (coords.ptr<cv::Vec3f>()[begin + i][axis] -
                    source.offsetFromBaseXYZ[axis]) / source.scaleFromBaseXYZ[axis] / workingToBaseScale;
        std::vector<FiberPredictionSample> samples;
        predictions.sampleBatch(points, directions, 1, samples);
        if (samples.size() != count) throw std::runtime_error("invalid presence sample count");
        for (size_t i = 0; i < count; ++i) {
            float presence = 0;
            for (const auto& option : samples[i].options)
                presence = std::max(presence, option.presence);
            image.ptr<uint8_t>()[begin + i] = cv::saturate_cast<uint8_t>(255 * presence);
        }
    }
    return image;
}

FiberReplaySectionFrame fiberReplaySectionFrame(
    const cv::Vec3d& center, const cv::Vec3d& inputTangent,
    const vc::lasagna::NormalSampler& normals, double workingToBaseScale, double thresholdBase,
    double tangentialFactor)
{
    (void)fiberReplayThresholdDescriptorJson(thresholdBase, tangentialFactor);
    if (!(cv::norm(inputTangent) > 0) || !std::isfinite(cv::norm(inputTangent)) || !(thresholdBase > 0))
        throw std::invalid_argument("invalid transverse frame tangent or threshold");
    const cv::Vec3d tangent = inputTangent / cv::norm(inputTangent);
    const auto normal = normals.sampleNormal(center / workingToBaseScale);
    cv::Vec3d horizontal = normal.normal - tangent * normal.normal.dot(tangent);
    if (!normal.valid || !std::isfinite(cv::norm(horizontal)) || cv::norm(horizontal) < 1e-10) {
        const cv::Vec3d seed = std::abs(tangent[0]) < 0.9 ? cv::Vec3d{1, 0, 0} : cv::Vec3d{0, 1, 0};
        horizontal = seed - tangent * seed.dot(tangent);
    }
    horizontal /= cv::norm(horizontal);
    cv::Vec3d vertical = tangent.cross(horizontal); vertical /= cv::norm(vertical);
    double rx = 1.0 / measureFiberReplayThreshold(center + horizontal, center,
        normals, workingToBaseScale, thresholdBase, tangentialFactor).thresholdErrorRatio;
    double ry = 1.0 / measureFiberReplayThreshold(center + vertical, center,
        normals, workingToBaseScale, thresholdBase, tangentialFactor).thresholdErrorRatio;
    if (rx > ry) { std::swap(horizontal, vertical); std::swap(rx, ry); }
    return {horizontal, vertical, tangent, rx, ry};
}

static cv::Vec3d inspectionTraceTangent(const std::vector<cv::Vec3d>& points,
    size_t index, const cv::Vec3d& fallback)
{
    for (size_t radius = 0; radius < points.size(); ++radius) {
        const size_t left = index > radius ? index - radius : 0;
        const size_t right = std::min(points.size() - 1, index + radius + 1);
        const auto tangent = points[right] - points[left];
        if (cv::norm(tangent) > 1e-12) return tangent;
    }
    return fallback;
}

FiberReplayStripPoint sampleFiberReplayStripColumn(
    const FiberReplayOverviewFiberletComponent& component, double assembledColumn)
{
    if (component.pointsBase.size() < 2 ||
        component.referenceArcsBase.size() != component.pointsBase.size() ||
        component.topSourceColumns < 2 || !component.positionMap.valid() ||
        !std::isfinite(assembledColumn) || assembledColumn < component.topColumns[0] ||
        assembledColumn > component.topColumns[1] - 1)
        throw std::invalid_argument("invalid traced-strip sample");
    const double grid = (assembledColumn - component.topColumns[0]) *
        (component.topSourceColumns - 1) / std::max(1, component.topColumns[1] - component.topColumns[0] - 1);
    const double position = std::clamp(component.positionMap.stripGridColumnToOriginalPosition(grid),
                                       0.0, double(component.pointsBase.size() - 1));
    const size_t i = std::min(size_t(position), component.pointsBase.size() - 2);
    const double t = position - i;
    return {component.pointsBase[i] * (1 - t) + component.pointsBase[i + 1] * t,
            inspectionTraceTangent(component.pointsBase, i, {1, 0, 0}),
            component.referenceArcsBase[i] * (1 - t) + component.referenceArcsBase[i + 1] * t, position};
}

FiberReplayInspection renderFiberReplayInspection(
    const std::vector<cv::Vec3d>& referenceGeometryBase, const FiberReplayTraceResult& replay,
    const vc::lasagna::NormalSampler& normals, double workingToBaseScale,
    const FiberReplayStripTextureSource& source, const FiberReplayTextureRenderer& ct,
    const FiberReplayTextureRenderer& presence, double thresholdBase, bool reverse, double voxelSizeUm,
    const std::string& ctSamplingDescription, int presenceThresholdU8)
{
    if (!(thresholdBase > 0) || !std::isfinite(thresholdBase) || !(voxelSizeUm > 0))
        throw std::invalid_argument("inspection needs positive threshold and voxel size");
    if (presenceThresholdU8 < 0 || presenceThresholdU8 > 255)
        throw std::invalid_argument("presence threshold must be in [0, 255]");
    const double factor = replay.tangentialThresholdFactor;
    const auto descriptor = fiberReplayThresholdDescriptorJson(thresholdBase, factor);
    constexpr int scale = 1;
    constexpr double baseSpacing = 2;
    auto scan = renderFiberReplayTraceOverview(referenceGeometryBase, replay, normals,
        workingToBaseScale, 1, source, ct, scale, true, baseSpacing);
    auto prediction = renderFiberReplayTraceOverview(referenceGeometryBase, replay, normals,
        workingToBaseScale, 1, source, presence, scale, true, baseSpacing);
    if (scan.pages.size() != prediction.pages.size())
        throw std::runtime_error("unaligned inspection modalities");
    FiberReplayInspection result;
    result.metadata = {{"format", "vc_fiber_inspection"}, {"version", 3},
        {"render_scale", scale}, {"base_samples_per_pixel", baseSpacing},
        {"ct_sampling", ctSamplingDescription},
        {"cross_section_center", "actual traced line; ellipse is trace-centered display tolerance, not GT rejection center"},
        {"modalities", {"ct", "presence"}}, {"reverse", reverse},
        {"threshold", descriptor},
        {"rows", nlohmann::json::array()}, {"restarts", nlohmann::json::array()},
        {"cross_sections", nlohmann::json::array()}, {"components", nlohmann::json::array()}};
    for (const auto& component : scan.fiberletComponents)
        result.metadata["components"].push_back({{"segment", component.sourceSegmentIndex},
            {"reference_arc_begin_base", component.referenceArcBeginBase},
            {"reference_arc_end_base", component.referenceArcEndBase},
            {"top_columns", component.topColumns}, {"side_columns", component.sideColumns},
            {"top_rows", component.topRows}, {"side_rows", component.sideRows}});
    const auto geometry = makePolylineArcGeometry(referenceGeometryBase);
    const auto arcMetadata = [&](double arc) {
        const double local = arc - replay.referenceBeginArcBase;
        const double original = reverse ? geometry.length() - local : local;
        return nlohmann::json{{"reference_arc_base", arc}, {"directional_arc_base", local},
            {"original_fiber_arc_base", original}, {"original_fiber_arc_mm", original * voxelSizeUm / 1000}};
    };
    for (const auto& failure : replay.failures) {
        auto entry = failureJson(failure, thresholdBase);
        entry.update(arcMetadata(failure.referenceArcBase));
        entry["label"] = "R" + std::to_string(failure.index);
        result.metadata["restarts"].push_back(entry);
    }
    const int margin = int(std::ceil(thresholdBase * std::max(1.0, factor) / baseSpacing)) + 4;
    int width = 1600;
    for (const auto& page : scan.pages) width = std::max(width, page.image.cols + 2 * margin);
    std::vector<cv::Mat_<cv::Vec3b>> rows;
    int height = 0;
    const auto flush = [&] {
        if (rows.empty()) return;
        cv::Mat_<cv::Vec3b> page;
        cv::vconcat(rows, page);
        result.pages.push_back(std::move(page));
        rows.clear(); height = 0;
    };
    const auto addRow = [&](const cv::Mat_<cv::Vec3b>& image, const std::string& label,
                            nlohmann::json metadata, int xOffset = 0) {
        const int h = image.rows + 22;
        if (height + h > 16000) flush();
        cv::Mat_<cv::Vec3b> row(h, width, cv::Vec3b{0, 0, 0});
        image.copyTo(row(cv::Rect(xOffset, 22, image.cols, image.rows)));
        cv::putText(row, label, {4, 16}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        metadata.update({{"label", label}, {"page", result.pages.size()},
                         {"rows", {height, height + h}}, {"image_row_begin", height + 22},
                         {"image_column_begin", xOffset}});
        result.metadata["rows"].push_back(metadata);
        rows.push_back(std::move(row)); height += h;
    };
    struct Station {
        FiberReplayStripPoint point;
        size_t segment;
        double column;
        std::string role;
        const FiberReplayFailure* failure = nullptr;
        bool hasStrip = true;
        size_t anchorSegment = 0;
    };
    std::vector<Station> forced;
    for (const auto& failure : replay.failures) {
        for (int next = 0; next < 2; ++next) {
            const size_t segmentIndex = failure.segmentIndex + next;
            if (segmentIndex >= replay.segments.size() || replay.segments[segmentIndex].tracePointsBase.empty()) {
                result.metadata["restarts"][failure.index][next ? "restart_start_available" : "breakoff_available"] = false;
                continue;
            }
            result.metadata["restarts"][failure.index][next ? "restart_start_available" : "breakoff_available"] = true;
            const auto& segment = replay.segments[segmentIndex];
            const auto& points = segment.tracePointsBase;
            const size_t index = next ? 0 : points.size() - 1;
            const double arc = next ? segment.startReferenceArcBase : segment.endReferenceArcBase;
            auto tangent = samplePolylineArc(geometry, arc - replay.referenceBeginArcBase).tangent;
            tangent = inspectionTraceTangent(points, index, tangent);
            auto component = std::find_if(scan.fiberletComponents.begin(), scan.fiberletComponents.end(),
                [&](const auto& c) { return c.sourceSegmentIndex == segmentIndex; });
            double column = 0;
            const bool hasStrip = component != scan.fiberletComponents.end();
            size_t anchorSegment = segmentIndex;
            if (hasStrip) column = next ? component->topColumns[0] : component->topColumns[1] - 1;
            else {
                // A one-point, zero-progress segment has no longitudinal strip.
                // Anchor its exact section beside the nearest real strip endpoint.
                double distance = std::numeric_limits<double>::infinity();
                for (const auto& c : scan.fiberletComponents) {
                    for (int end = 0; end < 2; ++end) {
                        const double a = end ? c.referenceArcEndBase : c.referenceArcBeginBase;
                        if (std::abs(a - arc) < distance) {
                            distance = std::abs(a - arc);
                            column = end ? c.topColumns[1] - 1 : c.topColumns[0];
                            anchorSegment = c.sourceSegmentIndex;
                        }
                    }
                }
            }
            forced.push_back({{points[index], tangent, arc, double(index)}, segmentIndex, column,
                next ? "restart_start" : "breakoff", &failure, hasStrip, anchorSegment});
        }
    }
    for (size_t p = 0; p < scan.pages.size(); ++p) {
        const auto& page = scan.pages[p];
        const auto& pred = prediction.pages[p];
        if (page.image.size() != pred.image.size()) throw std::runtime_error("unaligned inspection rasters");
        for (const auto& panel : page.panels) {
            const std::array<std::array<int, 2>, 4> bands{panel.referenceTopRows,
                panel.referenceSideRows, panel.fiberletTopRows, panel.fiberletSideRows};
            const std::array<const char*, 4> names{"reference top", "reference side", "traced top", "traced side"};
            size_t stripPage = 0;
            int stripCtRow = 0, stripPresenceRow = 0;
            for (size_t b = 0; b < bands.size(); ++b) {
                if (height + 2 * (bands[b][1] - bands[b][0] + 22) > 16000) flush();
                auto metadata = nlohmann::json{{"view", names[b]},
                    {"progress", {panel.progressFractionBegin, panel.progressFractionEnd}},
                    {"traced_top_columns", panel.fiberletTopColumns}};
                metadata["modality"] = "ct";
                if (b == 2) { stripPage = result.pages.size(); stripCtRow = height + 22; }
                addRow(page.image.rowRange(bands[b][0], bands[b][1]),
                    "CT / " + std::string(names[b]) + " / " + ctSamplingDescription + " / 2 base voxels per pixel", metadata, margin);
                metadata["modality"] = "presence";
                if (b == 2) stripPresenceRow = height + 22;
                addRow(pred.image.rowRange(bands[b][0], bands[b][1]), "Presence / " + std::string(names[b]), metadata, margin);
                if (presenceThresholdU8 > 0) {
                    // The presence raster stores 255*presence as grey, so a
                    // grey pixel carries the stored level exactly. Coloured
                    // pixels are the reference/trace overlays and are kept.
                    cv::Mat_<cv::Vec3b> band = pred.image.rowRange(bands[b][0], bands[b][1]).clone();
                    for (int row = 0; row < band.rows; ++row) {
                        auto* pixels = band.ptr<cv::Vec3b>(row);
                        for (int column = 0; column < band.cols; ++column) {
                            auto& pixel = pixels[column];
                            if (pixel[0] != pixel[1] || pixel[1] != pixel[2])
                                continue;
                            pixel = pixel[0] >= presenceThresholdU8
                                ? cv::Vec3b(255, 255, 255)
                                : cv::Vec3b(0, 0, 0);
                        }
                    }
                    if (height + band.rows + 22 > 16000) flush();
                    metadata["modality"] = "presence_threshold";
                    metadata["presence_threshold_u8"] = presenceThresholdU8;
                    addRow(band, "Presence >= " + std::to_string(presenceThresholdU8) +
                        " / " + std::string(names[b]), metadata, margin);
                    metadata.erase("presence_threshold_u8");
                }
            }
            struct Placed {
                Station station;
                FiberReplaySectionFrame frame;
                int x, w, h;
            };
            std::vector<std::vector<Placed>> lanes(1);
            const auto place = [&](Station station, bool mandatory) {
                const auto frame = fiberReplaySectionFrame(station.point.pointBase, station.point.tangentBase,
                    normals, workingToBaseScale, thresholdBase, factor);
                // Odd dimensions place the exact trace point at the exact strip
                // column, not half a pixel between two thumbnail samples.
                const int w = std::max(3, 2 * int(std::ceil(frame.horizontalRadius / baseSpacing)) + 1);
                const int h = std::max(3, 2 * int(std::ceil(frame.verticalRadius / baseSpacing)) + 1);
                const int center = margin + int(std::lround(station.column)) - panel.fiberletTopColumns[0];
                const int x = center - (w - 1) / 2;
                for (size_t lane = 0; ; ++lane) {
                    if (lane == lanes.size()) {
                        if (!mandatory) return;
                        lanes.emplace_back();
                    }
                    const bool overlaps = std::any_of(lanes[lane].begin(), lanes[lane].end(),
                        [&](const auto& a) { return x < a.x + a.w + 2 && x + w + 2 > a.x; });
                    if (!overlaps) {
                        lanes[lane].push_back({std::move(station), frame, x, w, h});
                        return;
                    }
                    if (!mandatory) return; // Dense regular sections fill the primary lane.
                }
            };
            for (const auto& station : forced)
                if (station.column >= panel.fiberletTopColumns[0] && station.column < panel.fiberletTopColumns[1])
                    place(station, true);
            // Layout first, inverse mapping second. Never sample a restart gap.
            for (const auto& c : scan.fiberletComponents) {
                const int begin = std::max(c.topColumns[0], panel.fiberletTopColumns[0]);
                const int end = std::min(c.topColumns[1], panel.fiberletTopColumns[1]);
                for (int column = begin; column < end; ++column)
                {
                    const int center = margin + column - panel.fiberletTopColumns[0];
                    // Even the narrowest valid section contains its center pixel.
                    if (std::any_of(lanes[0].begin(), lanes[0].end(), [&](const auto& a) {
                        return center >= a.x - 2 && center < a.x + a.w + 2;
                    })) continue;
                    place({sampleFiberReplayStripColumn(c, column), c.sourceSegmentIndex, double(column), "dense", nullptr, true, c.sourceSegmentIndex}, false);
                }
            }
            for (size_t lane = 0; lane < lanes.size(); ++lane) {
                if (lanes[lane].empty()) continue;
                int sectionHeight = 0;
                for (const auto& s : lanes[lane]) sectionHeight = std::max(sectionHeight, s.h);
                cv::Mat_<cv::Vec3b> ctRow(sectionHeight + 22, width, cv::Vec3b{0, 0, 0}), predRow = ctRow.clone();
                if (height + 2 * (ctRow.rows + 22) > 16000) flush();
                if (stripPage != result.pages.size()) {
                    stripPage = result.pages.size(); stripCtRow = height + 22;
                    addRow(page.image.rowRange(bands[2][0], bands[2][1]), "CT / traced top (section context) / " + ctSamplingDescription,
                        {{"modality", "ct"}, {"view", "traced top"}}, margin);
                    stripPresenceRow = height + 22;
                    addRow(pred.image.rowRange(bands[2][0], bands[2][1]), "Presence / traced top (section context)",
                        {{"modality", "presence"}, {"view", "traced top"}}, margin);
                }
                for (const auto& placed : lanes[lane]) {
                    const auto& station = placed.station;
                    const auto& frame = placed.frame;
                    const auto& sample = station.point;
                    const int w = placed.w, h = placed.h, x0 = placed.x;
                    if (x0 < 0 || x0 + w > width) throw std::runtime_error("aligned section exceeds page");
                    cv::Mat_<cv::Vec3f> coords(h, w);
                    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                        const auto base = sample.pointBase + frame.horizontal * (frame.horizontalRadius * (2.0 * x / (w - 1) - 1))
                            + frame.vertical * (frame.verticalRadius * (2.0 * y / (h - 1) - 1));
                        for (int a = 0; a < 3; ++a)
                            coords(y, x)[a] = float(base[a] * source.scaleFromBaseXYZ[a] + source.offsetFromBaseXYZ[a]);
                    }
                    const auto label = station.failure
                        ? "R" + std::to_string(station.failure->index) + (station.role == "breakoff" ? "-" : "+")
                        : "s" + std::to_string(result.metadata["cross_sections"].size());
                    const auto decorate = [&](const FiberReplayTextureRenderer& render) {
                        const auto gray = render(coords, 1);
                        if (gray.size() != coords.size()) throw std::runtime_error("cross-section raster mismatch");
                        cv::Mat_<cv::Vec3b> color; cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
                        auto overlay = color.clone();
                        cv::ellipse(overlay, {(w - 1) / 2, (h - 1) / 2}, {(w - 1) / 2, (h - 1) / 2},
                            0, 0, 360, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                        cv::addWeighted(overlay, 0.25, color, 0.75, 0, color);
                        return color;
                    };
                    decorate(ct).copyTo(ctRow(cv::Rect(x0, 22, w, h)));
                    decorate(presence).copyTo(predRow(cv::Rect(x0, 22, w, h)));
                    for (auto* row : {&ctRow, &predRow})
                        cv::putText(*row, label, {x0, 15}, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(220, 220, 220), 1);
                    auto entry = arcMetadata(sample.referenceArcBase);
                    entry.update({{"label", label}, {"role", station.role}, {"segment", station.segment},
                        {"strip_column", station.column}, {"strip_center_column", margin + station.column - panel.fiberletTopColumns[0]},
                        {"original_trace_position", sample.originalPosition}, {"has_longitudinal_strip", station.hasStrip},
                        {"anchor_segment", station.anchorSegment}, {"strip_page", stripPage},
                        {"strip_ct_row_begin", stripCtRow}, {"strip_presence_row_begin", stripPresenceRow},
                        {"center_base_xyz", pointJson(sample.pointBase)}, {"horizontal_axis_xyz", pointJson(frame.horizontal)},
                        {"vertical_axis_xyz", pointJson(frame.vertical)}, {"tangent_xyz", pointJson(frame.tangent)},
                        {"radii_base", {frame.horizontalRadius, frame.verticalRadius}},
                        {"extent_base", {2 * frame.horizontalRadius, 2 * frame.verticalRadius}},
                        {"columns", {x0, x0 + w}}, {"shape_yx", {h, w}}, {"lane", lane},
                        {"page", result.pages.size()}, {"ct_row_begin", height + 44},
                        {"presence_row_begin", height + ctRow.rows + 66}});
                    if (station.failure) entry["restart_index"] = station.failure->index;
                    result.metadata["cross_sections"].push_back(std::move(entry));
                }
                addRow(ctRow, "CT / aligned transverse sections; R- old end, R+ new start / " + ctSamplingDescription,
                    {{"modality", "ct"}, {"view", "cross_sections"}, {"traced_top_columns", panel.fiberletTopColumns}});
                addRow(predRow, "Presence / aligned transverse sections (long axis vertical)",
                    {{"modality", "presence"}, {"view", "cross_sections"}, {"traced_top_columns", panel.fiberletTopColumns}});
            }
        }
    }
    flush();
    result.metadata["pages"] = result.pages.size();
    return result;
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
