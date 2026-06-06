#pragma once

#include <opencv2/core/types.hpp>

#include <QPoint>
#include <QPointF>
#include <QString>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class CChunkedVolumeViewer;
class PlaneSurface;
class QuadSurface;
class QWidget;

namespace vc3d::line_annotation {

enum class GeneratedControlPointContextResult {
    None,
    Handled,
    NewLineAnnotationRequested,
};

struct GeneratedOverlay {
    struct ControlPointMarker {
        cv::Vec3f point{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
        double linePosition = std::numeric_limits<double>::quiet_NaN();
        bool isSeed = false;
    };

    std::vector<cv::Vec3f> linePoints;
    cv::Vec3f seedPoint{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
    cv::Vec3f pointMarker{std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN()};
    int seedLineIndex = -1;
    std::vector<double> markerLinePositions;
    std::vector<ControlPointMarker> controlPoints;
    double currentLinePosition = std::numeric_limits<double>::quiet_NaN();
    bool emphasizedPointMarker = false;
    bool useSurfaceCenterLine = false;
    bool currentLineMarkerAsCross = false;
};

struct GeneratedViews {
    std::string lineSurfaceName;
    QString lineSurfaceTitle;
    std::string lineSideSliceName;
    QString lineSideSliceTitle;
    std::string currentCutName;
    std::shared_ptr<PlaneSurface> currentCutSurface;
    std::vector<std::pair<std::string, std::shared_ptr<PlaneSurface>>> bottomCutSurfaces;
    std::vector<cv::Vec3f> linePoints;
    std::vector<cv::Vec3f> lineUpVectors;
    cv::Vec3f seedPoint{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
    int seedLineIndex = -1;
    int initialCenterIndex = 0;
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints;
};

inline bool finiteGeneratedPoint(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

inline bool validGeneratedLinePosition(double position, size_t pointCount)
{
    return std::isfinite(position) &&
           pointCount > 0 &&
           position >= 0.0 &&
           position <= static_cast<double>(pointCount - 1);
}

inline cv::Vec3f interpolatedGeneratedLinePoint(const std::vector<cv::Vec3f>& linePoints,
                                                double linePosition)
{
    if (linePoints.empty()) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    linePosition = std::clamp(linePosition, 0.0, static_cast<double>(linePoints.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(linePoints.size()) - 1);
    const float t = static_cast<float>(linePosition - static_cast<double>(lower));
    return linePoints[static_cast<size_t>(lower)] * (1.0f - t) +
           linePoints[static_cast<size_t>(upper)] * t;
}

inline std::optional<std::pair<double, double>> generatedControlLinePositionRange(
    const std::vector<GeneratedOverlay::ControlPointMarker>& controlPoints)
{
    double first = std::numeric_limits<double>::infinity();
    double last = -std::numeric_limits<double>::infinity();
    int finiteCount = 0;
    for (const auto& control : controlPoints) {
        if (!std::isfinite(control.linePosition)) {
            continue;
        }
        ++finiteCount;
        first = std::min(first, control.linePosition);
        last = std::max(last, control.linePosition);
    }
    if (finiteCount < 2 || !std::isfinite(first) || !std::isfinite(last) || first >= last) {
        return std::nullopt;
    }
    return std::make_pair(first, last);
}

inline bool generatedLineSegmentIsTail(
    double startPosition,
    double endPosition,
    const std::optional<std::pair<double, double>>& controlRange)
{
    if (!controlRange || !std::isfinite(startPosition) || !std::isfinite(endPosition)) {
        return false;
    }
    const double midpoint = (startPosition + endPosition) * 0.5;
    return midpoint < controlRange->first || midpoint > controlRange->second;
}

inline GeneratedOverlay makeGeneratedStripOverlay(
    const GeneratedViews& views,
    double currentLinePosition,
    const std::vector<double>& markerLinePositions)
{
    GeneratedOverlay overlay;
    overlay.linePoints = views.linePoints;
    overlay.seedPoint = views.seedPoint;
    overlay.seedLineIndex = views.controlPoints.empty() ? views.seedLineIndex : -1;
    overlay.useSurfaceCenterLine = true;
    overlay.currentLinePosition = currentLinePosition;
    overlay.controlPoints = views.controlPoints;
    overlay.markerLinePositions = markerLinePositions;
    return overlay;
}

inline GeneratedOverlay makeGeneratedCrossSliceOverlay(
    const GeneratedViews& views,
    double linePosition,
    bool emphasized,
    std::optional<float> controlDistanceThreshold,
    const std::function<float(const cv::Vec3f&)>& pointDistance)
{
    GeneratedOverlay overlay;
    overlay.pointMarker = interpolatedGeneratedLinePoint(views.linePoints, linePosition);
    overlay.emphasizedPointMarker = emphasized;
    if (!controlDistanceThreshold || !pointDistance) {
        return overlay;
    }
    for (const auto& control : views.controlPoints) {
        if (!finiteGeneratedPoint(control.point)) {
            continue;
        }
        const float distance = pointDistance(control.point);
        if (std::isfinite(distance) && std::abs(distance) <= *controlDistanceThreshold) {
            overlay.controlPoints.push_back(control);
        }
    }
    return overlay;
}

struct GeneratedControlPointContextMenuOptions {
    QWidget* parent = nullptr;
    std::string surfaceName;
    CChunkedVolumeViewer* viewer = nullptr;
    QPointF scenePoint;
    QPoint globalPos;
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints;
    size_t linePointCount = 0;
    double linePosition = std::numeric_limits<double>::quiet_NaN();
    bool stripViewer = false;
    std::function<void(double, cv::Vec3f)> deleteControlPoint;
};

QPointF generatedStripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                          QuadSurface* surface,
                                          double linePosition);
double generatedLinePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                           const QPointF& scenePoint);
std::optional<float> generatedCrossSliceControlPointDistanceThreshold(CChunkedVolumeViewer* viewer);
GeneratedOverlay makeGeneratedCrossSliceOverlayForPlane(const GeneratedViews& views,
                                                        double linePosition,
                                                        bool emphasized,
                                                        CChunkedVolumeViewer* viewer,
                                                        PlaneSurface* plane);
void applyGeneratedOverlay(CChunkedVolumeViewer* viewer,
                           const std::string& surfaceName,
                           const GeneratedOverlay& overlay);
void clearGeneratedControlPointContextPreview(CChunkedVolumeViewer* viewer,
                                              const std::string& surfaceName);
GeneratedControlPointContextResult showGeneratedControlPointContextMenu(
    const GeneratedControlPointContextMenuOptions& options);

} // namespace vc3d::line_annotation
