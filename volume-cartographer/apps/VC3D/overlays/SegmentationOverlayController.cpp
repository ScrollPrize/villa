#include "SegmentationOverlayController.hpp"

#include "../CState.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "../ViewerManager.hpp"
#include "../segmentation/tools/SegmentationEditManager.hpp"

#include <QColor>
#include <QDebug>
#include <QElapsedTimer>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>


#include "vc/core/util/Logging.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "SurfaceCatalog.hpp"
#include <vc_object_store/Uuid.hpp>

namespace
{
constexpr const char* kOverlayGroupKey = "segmentation_vertex_edit";
const QColor kActiveFill = QColor(255, 215, 0, 220);
const QColor kActiveBorder = QColor(255, 180, 0, 255);
const QColor kNeighborFill = QColor(0, 220, 255, 160);
const QColor kNeighborBorder = QColor(255, 255, 255, 210);
const QColor kGrowthFill = QColor(140, 255, 160, 210);
const QColor kGrowthBorder = QColor(80, 190, 90, 230);
constexpr qreal kMarkerRadius = 5.0;
constexpr qreal kMarkerPenWidth = 2.0;
constexpr qreal kActivePenWidth = 2.5;
constexpr qreal kMaskZ = 60.0;
constexpr qreal kSurfaceOverlapZ = 45.0;
constexpr qreal kMarkerZ = 95.0;
constexpr qreal kRadiusCircleZ = 80.0;
constexpr float kManualAddPlaneDistanceTolerance = 10.0f;
constexpr qreal kManualAddIntersectionZ = 121.0;
constexpr float kManualAddIntersectionOpacityScale = 1.2f;
constexpr float kManualAddIntersectionWidthScale = 1.3f;
constexpr float kManualAddIntersectionMinWidthDelta = 0.75f;
QColor manualAddIntersectionColor()
{
    return QColor(0, 220, 255);
}

qreal manualAddIntersectionWidth(qreal baseWidth)
{
    return std::max(baseWidth * kManualAddIntersectionWidthScale,
                    baseWidth + kManualAddIntersectionMinWidthDelta);
}

std::optional<QPointF> markerGridToScene(VolumeViewerBase* viewer,
                                         const SegmentationOverlayController::VertexMarker& marker)
{
    if (!viewer || viewer->surfName() != "segmentation") {
        return std::nullopt;
    }

    auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    if (!quad) {
        return std::nullopt;
    }

    const cv::Vec2f scale = quad->scale();
    if (std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) {
        return std::nullopt;
    }

    const cv::Vec3f center = quad->center();
    const float surfaceX = static_cast<float>(marker.col) / scale[0] - center[0];
    const float surfaceY = static_cast<float>(marker.row) / scale[1] - center[1];
    return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
}
}

bool SegmentationOverlayController::State::operator==(const State& rhs) const
{
    const auto equalMarker = [](const VertexMarker& lhs, const VertexMarker& rhs) {
        return lhs.row == rhs.row &&
               lhs.col == rhs.col &&
               std::fabs(lhs.world[0] - rhs.world[0]) < 1e-4f &&
               std::fabs(lhs.world[1] - rhs.world[1]) < 1e-4f &&
               std::fabs(lhs.world[2] - rhs.world[2]) < 1e-4f &&
               lhs.isActive == rhs.isActive &&
               lhs.isGrowth == rhs.isGrowth;
    };

    const auto optionalEqual = [&](const std::optional<VertexMarker>& lhsOpt,
                                   const std::optional<VertexMarker>& rhsOpt) {
        if (!lhsOpt && !rhsOpt) {
            return true;
        }
        if (static_cast<bool>(lhsOpt) != static_cast<bool>(rhsOpt)) {
            return false;
        }
        return equalMarker(*lhsOpt, *rhsOpt);
    };

    const auto vectorEqual = [&](const std::vector<VertexMarker>& lhsVec,
                                 const std::vector<VertexMarker>& rhsVec) {
        if (lhsVec.size() != rhsVec.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhsVec.size(); ++i) {
            if (!equalMarker(lhsVec[i], rhsVec[i])) {
                return false;
            }
        }
        return true;
    };

    const auto floatEqual = [](float lhs, float rhs) {
        return std::fabs(lhs - rhs) < 1e-4f;
    };

    // Helper for cv::Vec3f comparison
    const auto vec3fEqual = [](const cv::Vec3f& lhs, const cv::Vec3f& rhs) {
        return cv::norm(lhs - rhs) < 1e-4f;
    };

    return optionalEqual(activeMarker, rhs.activeMarker) &&
           vectorEqual(neighbours, rhs.neighbours) &&
           surfaceMaskPoints == rhs.surfaceMaskPoints &&
           pushPullActive == rhs.pushPullActive &&
           floatEqual(gaussianRadiusSteps, rhs.gaussianRadiusSteps) &&
           floatEqual(gaussianSigmaSteps, rhs.gaussianSigmaSteps) &&
           floatEqual(displayRadiusSteps, rhs.displayRadiusSteps) &&
           floatEqual(gridStepWorld, rhs.gridStepWorld) &&
           correctionDragActive == rhs.correctionDragActive &&
           vec3fEqual(correctionDragStart, rhs.correctionDragStart) &&
           vec3fEqual(correctionDragCurrent, rhs.correctionDragCurrent) &&
           manualAddActive == rhs.manualAddActive &&
           manualAddRevision == rhs.manualAddRevision;
}

SegmentationOverlayController::SegmentationOverlayController(CState* state, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroupKey, parent)
    , _state(state)
{
    if (_state) {
        connect(_state, &CState::surfaceChanged,
                this, &SegmentationOverlayController::onSurfaceChanged);
    }

    _overlapWatcher = new QFutureWatcher<QImage>(this);
    connect(_overlapWatcher, &QFutureWatcher<QImage>::finished,
            this, &SegmentationOverlayController::handleOverlapComputeFinished);
}

void SegmentationOverlayController::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;
    refreshAll();
}

void SegmentationOverlayController::applyState(const State& state)
{
    State sanitized = state;
    sanitized.gaussianRadiusSteps = std::max(sanitized.gaussianRadiusSteps, 0.0f);
    sanitized.gaussianSigmaSteps = std::max(sanitized.gaussianSigmaSteps, 0.0f);
    sanitized.displayRadiusSteps = std::max(sanitized.displayRadiusSteps, 0.0f);
    sanitized.gridStepWorld = std::max(sanitized.gridStepWorld, 1e-4f);

    // skip refresh if state hasn't changed
    if (_currentState && *_currentState == sanitized) {
        return;
    }

    // throttle the overlay refresh a bit
    static constexpr auto kMinRefreshInterval = std::chrono::milliseconds(32);
    auto now = std::chrono::steady_clock::now();
    if (now - _lastRefreshTime < kMinRefreshInterval) {
        // state changed but we're throttling - save state for next refresh
        _currentState = std::move(sanitized);
        // ensure a deferred refresh fires after the throttle window
        if (!_deferredRefreshTimer) {
            _deferredRefreshTimer = new QTimer(this);
            _deferredRefreshTimer->setSingleShot(true);
            connect(_deferredRefreshTimer, &QTimer::timeout, this, [this]() {
                _lastRefreshTime = {};  // Reset so next applyState proceeds
                if (_currentState) {
                    refreshAll();
                }
            });
        }
        _deferredRefreshTimer->start(static_cast<int>(kMinRefreshInterval.count()));
        return;
    }
    _lastRefreshTime = now;

    _currentState = std::move(sanitized);
    refreshAll();
}

void SegmentationOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    _overlapCaches.erase(viewer);
    if (_pendingOverlap.viewer == viewer) {
        _pendingOverlap.viewer = nullptr;
        if (_overlapCancel) {
            _overlapCancel->store(true, std::memory_order_relaxed);
        }
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}

bool SegmentationOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    // Enable overlay rendering if editing is enabled or the surface overlap
    // overlay is active on this viewer
    const bool surfaceOverlapActive = viewer && viewer->surfaceOverlayEnabled() &&
                                      !viewer->surfaceOverlays().empty();

    return _editingEnabled || surfaceOverlapActive;
}

void SegmentationOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                                      ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer) {
        return;
    }

    // Render surface overlap overlay (independent of editing mode and state)
    buildSurfaceOverlapOverlay(viewer, builder);

    if (!_currentState) {
        return;
    }

    const State& state = *_currentState;

    if (state.manualAddActive) {
        const bool flattened = viewer->surfName() == "segmentation";
        auto* quadSurface = flattened ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
        auto gridToScene = [&](const QPointF& grid) -> QPointF {
            if (!quadSurface) {
                return {};
            }
            const cv::Vec2f surfScale = quadSurface->scale();
            const cv::Vec3f center = quadSurface->center();
            if (std::abs(surfScale[0]) < 1e-6f || std::abs(surfScale[1]) < 1e-6f) {
                return {};
            }
            const float surfX = static_cast<float>(grid.x()) / surfScale[0] - center[0];
            const float surfY = static_cast<float>(grid.y()) / surfScale[1] - center[1];
            return viewer->surfaceCoordsToScene(surfX, surfY);
        };
        auto drawLine = [&](const State::ManualAddLine& line) {
            std::vector<QPointF> points;
            if (flattened) {
                points.reserve(line.surfacePoints.size());
                for (const QPointF& grid : line.surfacePoints) {
                    const QPointF scene = gridToScene(grid);
                    if (std::isfinite(scene.x()) && std::isfinite(scene.y())) {
                        points.push_back(scene);
                    }
                }
            } else if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
                Q_UNUSED(plane);
                return;
            } else {
                points = volumeToScene(viewer, line.worldPoints);
            }
            if (points.size() < 2) {
                return;
            }
            ViewerOverlayControllerBase::OverlayStyle style;
            style.penColor = line.committed ? QColor(0, 220, 255, 240) : QColor(255, 220, 0, 230);
            style.penWidth = 3.0;
            style.z = 110.0;
            builder.addLineStrip(points, false, style);
        };
        if (flattened) {
            ViewerOverlayControllerBase::OverlayStyle hoverFillStyle;
            hoverFillStyle.penColor = Qt::transparent;
            hoverFillStyle.brushColor = QColor(255, 220, 0, 55);
            hoverFillStyle.penWidth = 0.0;
            hoverFillStyle.z = 104.0;
            for (const auto& span : state.manualAddHoverFillSpans) {
                const QPointF topLeft = gridToScene(QPointF(span.colFirst - 0.5, span.row - 0.5));
                const QPointF bottomRight = gridToScene(QPointF(span.colLast + 0.5, span.row + 0.5));
                if (!std::isfinite(topLeft.x()) || !std::isfinite(topLeft.y()) ||
                    !std::isfinite(bottomRight.x()) || !std::isfinite(bottomRight.y())) {
                    continue;
                }
                builder.addRect(QRectF(topLeft, bottomRight).normalized(), true, hoverFillStyle);
            }
        }
        for (const auto& line : state.manualAddCommittedLines) {
            drawLine(line);
        }
        for (const auto& line : state.manualAddHoverLines) {
            drawLine(line);
        }
        if (flattened && state.manualAddHoverVertex) {
            const QPointF scene = gridToScene(*state.manualAddHoverVertex);
            if (std::isfinite(scene.x()) && std::isfinite(scene.y())) {
                ViewerOverlayControllerBase::OverlayStyle hoverStyle;
                hoverStyle.penColor = QColor(255, 255, 255, 250);
                hoverStyle.brushColor = state.manualAddHoverCrossFill ? Qt::transparent : QColor(255, 220, 0, 220);
                hoverStyle.penWidth = 2.0;
                hoverStyle.z = 112.0;
                builder.addCircle(scene, state.manualAddHoverCrossFill ? 8.0 : 5.0, !state.manualAddHoverCrossFill, hoverStyle);
            }
        }
        if (!flattened) {
            auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface());
            if (plane) {
                ViewerOverlayControllerBase::OverlayStyle previewStyle;
                previewStyle.penColor = manualAddIntersectionColor();
                previewStyle.penColor.setAlphaF(std::clamp(viewer->intersectionOpacity() *
                                                           kManualAddIntersectionOpacityScale,
                                                           0.0f,
                                                           1.0f));
                previewStyle.penWidth = manualAddIntersectionWidth(viewer->intersectionThickness());
                previewStyle.z = kManualAddIntersectionZ;

                auto addUnique = [](std::vector<cv::Vec3f>& points, const cv::Vec3f& candidate) {
                    for (const cv::Vec3f& point : points) {
                        if (cv::norm(point - candidate) < 1e-3f) {
                            return;
                        }
                    }
                    points.push_back(candidate);
                };

                constexpr int kEdges[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
                constexpr float kPlaneEpsilon = 1e-3f;
                std::vector<std::pair<cv::Vec3f, cv::Vec3f>> segments;
                for (const auto& quad : state.manualAddPreviewQuads) {
                    std::vector<cv::Vec3f> intersections;
                    for (const auto& edge : kEdges) {
                        const cv::Vec3f& a = quad[edge[0]];
                        const cv::Vec3f& b = quad[edge[1]];
                        const float da = plane->scalarp(a);
                        const float db = plane->scalarp(b);
                        const bool aOnPlane = std::abs(da) <= kPlaneEpsilon;
                        const bool bOnPlane = std::abs(db) <= kPlaneEpsilon;
                        if (aOnPlane) {
                            addUnique(intersections, a);
                        }
                        if (bOnPlane) {
                            addUnique(intersections, b);
                        }
                        if (aOnPlane || bOnPlane || da * db > 0.0f) {
                            continue;
                        }
                        const float t = da / (da - db);
                        addUnique(intersections, a + (b - a) * t);
                    }
                    if (intersections.size() == 2) {
                        segments.push_back({intersections[0], intersections[1]});
                    } else if (intersections.size() > 2) {
                        std::pair<int, int> farthest{0, 1};
                        float farthestDist = 0.0f;
                        for (int i = 0; i < static_cast<int>(intersections.size()); ++i) {
                            for (int j = i + 1; j < static_cast<int>(intersections.size()); ++j) {
                                const float dist = cv::norm(intersections[i] - intersections[j]);
                                if (dist > farthestDist) {
                                    farthestDist = dist;
                                    farthest = {i, j};
                                }
                            }
                        }
                        segments.push_back({intersections[farthest.first], intersections[farthest.second]});
                    }
                }

                std::vector<std::vector<cv::Vec3f>> polylines;
                constexpr float kMergeEpsilon = 1e-2f;
                auto samePoint = [](const cv::Vec3f& a, const cv::Vec3f& b) {
                    return cv::norm(a - b) < kMergeEpsilon;
                };
                for (const auto& segment : segments) {
                    if (cv::norm(segment.first - segment.second) < kMergeEpsilon) {
                        continue;
                    }
                    bool merged = false;
                    for (auto& line : polylines) {
                        if (samePoint(line.back(), segment.first)) {
                            line.push_back(segment.second);
                            merged = true;
                            break;
                        }
                        if (samePoint(line.back(), segment.second)) {
                            line.push_back(segment.first);
                            merged = true;
                            break;
                        }
                        if (samePoint(line.front(), segment.second)) {
                            line.insert(line.begin(), segment.first);
                            merged = true;
                            break;
                        }
                        if (samePoint(line.front(), segment.first)) {
                            line.insert(line.begin(), segment.second);
                            merged = true;
                            break;
                        }
                    }
                    if (!merged) {
                        polylines.push_back({segment.first, segment.second});
                    }
                }
                for (bool changed = true; changed;) {
                    changed = false;
                    for (std::size_t i = 0; i < polylines.size() && !changed; ++i) {
                        for (std::size_t j = i + 1; j < polylines.size(); ++j) {
                            if (samePoint(polylines[i].back(), polylines[j].front())) {
                                polylines[i].insert(polylines[i].end(), polylines[j].begin() + 1, polylines[j].end());
                            } else if (samePoint(polylines[i].back(), polylines[j].back())) {
                                polylines[i].insert(polylines[i].end(), polylines[j].rbegin() + 1, polylines[j].rend());
                            } else if (samePoint(polylines[i].front(), polylines[j].back())) {
                                polylines[i].insert(polylines[i].begin(), polylines[j].begin(), polylines[j].end() - 1);
                            } else if (samePoint(polylines[i].front(), polylines[j].front())) {
                                polylines[i].insert(polylines[i].begin(), polylines[j].rbegin(), polylines[j].rend() - 1);
                            } else {
                                continue;
                            }
                            polylines.erase(polylines.begin() + static_cast<std::ptrdiff_t>(j));
                            changed = true;
                            break;
                        }
                    }
                }

                auto lineLength = [](const std::vector<cv::Vec3f>& line) {
                    float length = 0.0f;
                    for (std::size_t i = 1; i < line.size(); ++i) {
                        length += cv::norm(line[i] - line[i - 1]);
                    }
                    return length;
                };
                auto bestLine = std::max_element(polylines.begin(), polylines.end(), [&](const auto& lhs, const auto& rhs) {
                    return lineLength(lhs) < lineLength(rhs);
                });
                if (bestLine != polylines.end() && bestLine->size() >= 2) {
                    builder.addLineStrip(volumeToScene(viewer, *bestLine), false, previewStyle);
                }
            }
            ViewerOverlayControllerBase::OverlayStyle constraintStyle;
            constraintStyle.penColor = QColor(255, 255, 120, 255);
            constraintStyle.brushColor = QColor(255, 255, 120, 220);
            constraintStyle.penWidth = 1.0;
            constraintStyle.z = 111.0;
            for (const cv::Vec3f& constraint : state.manualAddPlaneConstraints) {
                if (!plane || plane->pointDist(constraint) <= kManualAddPlaneDistanceTolerance) {
                    builder.addCircle(viewer->volumeToScene(constraint), 5.0, true, constraintStyle);
                }
            }
        }
    }

    // Render correction drag line (regardless of editing mode - corrections are annotations)
    if (state.correctionDragActive) {
        const QPointF startScene = viewer->volumeToScene(state.correctionDragStart);
        const QPointF currentScene = viewer->volumeToScene(state.correctionDragCurrent);

        // Draw line from start to current
        ViewerOverlayControllerBase::OverlayStyle lineStyle;
        lineStyle.penColor = QColor(255, 100, 100);  // Red-ish for correction
        lineStyle.penWidth = 3.0;
        lineStyle.z = 100.0;  // Draw on top
        builder.addLineStrip({startScene, currentScene}, false, lineStyle);

        // Draw a small circle at the start (anchor) position
        ViewerOverlayControllerBase::OverlayStyle anchorStyle;
        anchorStyle.penColor = QColor(255, 200, 100);  // Orange for anchor
        anchorStyle.brushColor = QColor(255, 200, 100, 180);
        anchorStyle.penWidth = 2.0;
        anchorStyle.z = 101.0;
        builder.addCircle(startScene, 8.0, true, anchorStyle);

        // Draw a small circle at the current (target) position
        ViewerOverlayControllerBase::OverlayStyle targetStyle;
        targetStyle.penColor = QColor(100, 255, 100);  // Green for target
        targetStyle.brushColor = QColor(100, 255, 100, 180);
        targetStyle.penWidth = 2.0;
        targetStyle.z = 101.0;
        builder.addCircle(currentScene, 8.0, true, targetStyle);
    }

    // Other overlays require editing to be enabled
    if (!_editingEnabled) {
        return;
    }

    if (!state.surfaceMaskPoints.empty() && viewer->surfName() == "segmentation") {
        std::vector<QPointF> scenePoints;
        scenePoints.reserve(state.surfaceMaskPoints.size());
        for (const QPointF& surfacePoint : state.surfaceMaskPoints) {
            const QPointF scene = viewer->surfaceCoordsToScene(static_cast<float>(surfacePoint.x()),
                                                               static_cast<float>(surfacePoint.y()));
            if (std::isfinite(scene.x()) && std::isfinite(scene.y())) {
                scenePoints.push_back(scene);
            }
        }

        if (!scenePoints.empty()) {
            ViewerOverlayControllerBase::OverlayStyle style;
            style.penColor = QColor(255, 80, 40, 230);
            style.brushColor = QColor(255, 80, 40, 70);
            style.penWidth = 2.0;
            style.penStyle = Qt::DashLine;
            style.dashPattern = {4.0, 4.0};
            style.z = kMaskZ + 5.0;

            const QPointF center = scenePoints.back();
            const QPointF edge = viewer->surfaceCoordsToScene(
                static_cast<float>(state.surfaceMaskPoints.back().x() + state.displayRadiusSteps),
                static_cast<float>(state.surfaceMaskPoints.back().y()));
            const qreal radiusPixels = std::hypot(edge.x() - center.x(), edge.y() - center.y());
            if (radiusPixels > 1.0) {
                builder.addCircle(center, radiusPixels, false, style);
            }

            if (scenePoints.size() >= 2) {
                style.penWidth = std::max<qreal>(2.0, radiusPixels * 0.35);
                style.penStyle = Qt::SolidLine;
                builder.addLineStrip(scenePoints, false, style);
            }
        }
    }

    buildVertexMarkers(state, viewer, builder);
    buildRadiusOverlay(state, viewer, builder);
}

void SegmentationOverlayController::onSurfaceChanged(std::string name, std::shared_ptr<Surface> surface)
{
    Q_UNUSED(surface);

    if (name == "segmentation") {
        refreshAll();
    }
}

void SegmentationOverlayController::buildRadiusOverlay(const State& state,
                                                       VolumeViewerBase* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    if (!state.activeMarker || !state.activeMarker->isActive) {
        return;
    }

    const float radiusWorld = state.gaussianRadiusSteps * state.gridStepWorld;
    if (radiusWorld <= 0.0f) {
        return;
    }

    const cv::Vec3f world = state.activeMarker->world;
    const QPointF sceneCenter = viewer->volumeToScene(world);

    cv::Vec3f offsetWorld = world;
    offsetWorld[0] += radiusWorld;
    const QPointF sceneEdge = viewer->volumeToScene(offsetWorld);
    const qreal radiusPixels = std::hypot(sceneEdge.x() - sceneCenter.x(),
                                          sceneEdge.y() - sceneCenter.y());

    if (radiusPixels <= 1.0e-3) {
        return;
    }

    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 255, 255, 120);
    style.penWidth = 1.5f;
    style.brushColor = QColor(Qt::transparent);
    style.z = kRadiusCircleZ;

    builder.addCircle(sceneCenter, radiusPixels, false, style);
}

void SegmentationOverlayController::buildVertexMarkers(const State& state,
                                                       VolumeViewerBase* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    const auto buildStyle = [](const VertexMarker& marker) {
        ViewerOverlayControllerBase::OverlayStyle style;
        style.penWidth = marker.isActive ? kActivePenWidth : kMarkerPenWidth;
        style.z = kMarkerZ;

        if (marker.isActive) {
            style.penColor = kActiveBorder;
            style.brushColor = kActiveFill;
        } else if (marker.isGrowth) {
            style.penColor = kGrowthBorder;
            style.brushColor = kGrowthFill;
        } else {
            style.penColor = kNeighborBorder;
            style.brushColor = kNeighborFill;
        }

        return style;
    };

    const auto appendMarker = [&](VertexMarker marker) {
        const auto directScene = markerGridToScene(viewer, marker);
        const QPointF scene = directScene ? *directScene : viewer->volumeToScene(marker.world);
        const auto style = buildStyle(marker);
        builder.addCircle(scene, kMarkerRadius, true, style);
    };

    for (const auto& marker : state.neighbours) {
        appendMarker(marker);
    }

    if (state.activeMarker) {
        auto active = *state.activeMarker;
        active.isActive = true;
        appendMarker(active);
    }
}

void SegmentationOverlayController::buildSurfaceOverlapOverlay(
    VolumeViewerBase* viewer,
    ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer || !viewer->surfaceOverlayEnabled()) {
        return;
    }

    const auto& overlays = viewer->surfaceOverlays();
    if (overlays.empty()) {
        return;
    }

    // Get the current surface
    Surface* viewerSurf = viewer->currentSurface();
    auto* currentQuad = dynamic_cast<QuadSurface*>(viewerSurf);
    if (!currentQuad) {
        return;
    }

    const float threshold = viewer->surfaceOverlapThreshold();
    if (threshold <= 0.0f || !_state || !_viewerManager) {
        return;
    }

    const cv::Mat_<cv::Vec3f>* currentPts = currentQuad->rawPointsPtr();
    if (!currentPts || currentPts->empty()) {
        return;
    }
    const int rows = currentPts->rows;
    const int cols = currentPts->cols;
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const auto session = _state->objectStoreSession();
    if (!session || !_state->currentVolume()) {
        return;
    }

    const int sampleRows = rows;
    const int sampleCols = cols;
    if (sampleRows <= 0 || sampleCols <= 0) {
        return;
    }

    // The common interaction path reaches here every overlay refresh. Compare
    // only O(1) revisions before resolving surface IDs or querying the store.
    // The surface map's version deliberately over-invalidates when an
    // unrelated surface changes, which is preferable to walking and hashing
    // every selected surface on each pan/zoom tick.
    const std::uint64_t overlaysRevision = viewer->surfaceOverlaysRevision();
    const std::uint64_t surfacesVersion = _state->surfacesVersion();
    auto cacheIt = _overlapCaches.find(viewer);
    bool needsRebuild = (cacheIt == _overlapCaches.end()) ||
                        (cacheIt->second.surface != currentQuad) ||
                        (cacheIt->second.overlaysRevision != overlaysRevision) ||
                        (cacheIt->second.surfacesVersion != surfacesVersion) ||
                        (std::abs(cacheIt->second.threshold - threshold) > 0.01f) ||
                        (cacheIt->second.image.width() != sampleCols) ||
                        (cacheIt->second.image.height() != sampleRows);

    if (needsRebuild) {
        // Overlay surfaces by id (stored, or drafts) and their colours.
        std::vector<std::string> targets;
        std::unordered_map<std::string, cv::Vec3b> targetColors;
        targets.reserve(overlays.size());
        targetColors.reserve(overlays.size());

        for (const auto& [surfId, color] : overlays) {
            auto surfBase = _state->surface(surfId);
            auto overlaySurf = std::dynamic_pointer_cast<QuadSurface>(surfBase);
            if (!overlaySurf || overlaySurf.get() == currentQuad || overlaySurf->id == currentQuad->id ||
                !vc::objects::util::isUuid(overlaySurf->id)) {
                continue;
            }
            const auto uuid = vc::objects::util::canonicalUuid(overlaySurf->id);
            targets.push_back(uuid);
            targetColors[uuid] = color;
        }

        if (targets.empty()) {
            OverlapCache emptyCache;
            emptyCache.surface = currentQuad;
            emptyCache.overlaysRevision = overlaysRevision;
            emptyCache.surfacesVersion = surfacesVersion;
            emptyCache.threshold = threshold;
            _overlapCaches[viewer] = std::move(emptyCache);
            return;
        }

        // If a compute for *different* parameters is still in flight (e.g. the
        // previously shown surface), cancel it so the fresh one can start as soon
        // as the watcher fires instead of waiting out the stale pass.
        if (_overlapComputeRunning && _overlapCancel) {
            const OverlapCache& pending = _pendingOverlap.cache;
            const bool pendingMatches = _pendingOverlap.viewer == viewer &&
                                        pending.surface == currentQuad &&
                                        pending.overlaysRevision == overlaysRevision &&
                                        pending.surfacesVersion == surfacesVersion &&
                                        std::abs(pending.threshold - threshold) <= 0.01f;
            if (!pendingMatches) {
                _overlapCancel->store(true, std::memory_order_relaxed);
            }
        }

        if (!_overlapComputeRunning) {
            cv::Mat_<cv::Vec3f> sampledPoints(sampleRows, sampleCols);
            for (int sr = 0; sr < sampleRows; ++sr) {
                const int r = sr;
                for (int sc = 0; sc < sampleCols; ++sc) {
                    const int c = sc;
                    sampledPoints(sr, sc) = (*currentPts)(r, c);
                }
            }

            // Store cache metadata for when the result arrives
            _pendingOverlap.viewer = viewer;
            _pendingOverlap.cache = OverlapCache{};
            _pendingOverlap.cache.surface = currentQuad;
            _pendingOverlap.cache.overlaysRevision = overlaysRevision;
            _pendingOverlap.cache.surfacesVersion = surfacesVersion;
            _pendingOverlap.cache.threshold = threshold;

            _overlapComputeRunning = true;

            auto cancel = std::make_shared<std::atomic_bool>(false);
            _overlapCancel = cancel;

            auto future = QtConcurrent::run(
                [sampledPoints = std::move(sampledPoints),
                 targets = std::move(targets),
                 targetColors = std::move(targetColors),
                 threshold,
                 session,
                 sources = _state->surfaceSources(),
                 volume = vc::objects::VolumeVoxels(_state->annotationVolume().value_or(vc::project::AnnotationVolume{})),
                 cancel]() -> QImage {
                try {
                    QElapsedTimer timer;
                    timer.start();

                    QImage sampledImage(sampledPoints.cols, sampledPoints.rows,
                                        QImage::Format_ARGB32);
                    sampledImage.fill(Qt::transparent);
                    uchar* const bits = sampledImage.bits();
                    const auto bytesPerLine = sampledImage.bytesPerLine();

                    // Explicit fan-out: VC3D pins the implicit pools (OpenCV/
                    // OMP) to one thread, so parallelism must be spawned here.
                    // Each row is one store read; rows own their scanlines.
                    const unsigned hw = std::thread::hardware_concurrency();
                    const std::size_t workerCount = std::max<std::size_t>(
                        1, std::min<std::size_t>(hw == 0 ? 4 : hw,
                                                 static_cast<std::size_t>(sampledPoints.rows)));
                    std::atomic_int nextRow{0};
                    std::vector<std::future<void>> workers;
                    workers.reserve(workerCount);
                    for (std::size_t w = 0; w < workerCount; ++w) {
                        workers.emplace_back(std::async(std::launch::async, [&]() {
                            std::vector<std::array<double, 3>> points;
                            std::vector<int> columns;
                            while (true) {
                                const int r = nextRow.fetch_add(1, std::memory_order_relaxed);
                                if (r >= sampledPoints.rows ||
                                    cancel->load(std::memory_order_relaxed)) {
                                    return;
                                }
                                points.clear();
                                columns.clear();
                                for (int c = 0; c < sampledPoints.cols; ++c) {
                                    const cv::Vec3f& pt = sampledPoints(r, c);
                                    if (vc::geometry::validSurfacePoint(pt)) {
                                        points.push_back({pt[0], pt[1], pt[2]});
                                        columns.push_back(c);
                                    }
                                }
                                // The nearest overlay surface within the threshold colours the sample.
                                const auto hits = SurfaceCatalog::nearest(*session, sources, volume,
                                                                          targets, points, threshold);
                                auto* scanline = reinterpret_cast<QRgb*>(bits + r * bytesPerLine);
                                for (size_t i = 0; i < points.size(); ++i) {
                                    if (hits[i].empty()) {
                                        continue;
                                    }
                                    const auto colorIt = targetColors.find(hits[i].front().objectUuid);
                                    if (colorIt == targetColors.end()) {
                                        continue;
                                    }
                                    const cv::Vec3b& hitColor = colorIt->second;
                                    scanline[columns[i]] = qRgba(hitColor[2], hitColor[1], hitColor[0], 200);
                                }
                            }
                        }));
                    }
                    for (auto& worker : workers) {
                        worker.get();
                    }

                    const bool canceled = cancel->load(std::memory_order_relaxed);
                    if (ProfileLoggingEnabled()) {
                        Logger()->info("[vc3d-profile] surfaceOverlapCompute elapsed_ms={} "
                                       "size={}x{} targets={} workers={} canceled={}",
                                       timer.elapsed(), sampledPoints.cols, sampledPoints.rows,
                                       targets.size(), workerCount, canceled);
                    }
                    if (canceled) {
                        return {};  // canceled: result is discarded
                    }
                    return sampledImage;
                } catch (const std::exception& e) {
                    qWarning() << "Segmentation overlap worker failed:" << e.what();
                } catch (...) {
                    qWarning() << "Segmentation overlap worker failed with an unknown exception";
                }
                return {};
            });

            _overlapWatcher->setFuture(future);
        }
    }

    // Render from cache if available (stale data while recomputing is fine)
    if (cacheIt == _overlapCaches.end()) {
        return;
    }

    const OverlapCache& cache = cacheIt->second;
    if (cache.image.isNull()) {
        return;
    }

    // Place the overlap image by mapping surface grid cells to scene coordinates.
    const cv::Vec2f surfScale = currentQuad->scale();
    const cv::Vec3f center = currentQuad->center();

    auto gridToScene = [&](int row, int col) -> QPointF {
        const float surfX = static_cast<float>(col) / surfScale[0] - center[0];
        const float surfY = static_cast<float>(row) / surfScale[1] - center[1];
        return viewer->surfaceCoordsToScene(surfX, surfY);
    };

    QPointF p0 = gridToScene(0, 0);
    QPointF p1 = gridToScene(0, 1);
    qreal gridToSceneScale = std::hypot(p1.x() - p0.x(), p1.y() - p0.y());

    if (gridToSceneScale < 1e-6) {
        return;
    }

    QPointF topLeft = gridToScene(0, 0);
    builder.addImage(cache.image,
                     topLeft,
                     gridToSceneScale,
                     0.6,
                     kSurfaceOverlapZ);
}

void SegmentationOverlayController::handleOverlapComputeFinished()
{
    _overlapComputeRunning = false;

    const bool wasCanceled = _overlapCancel &&
                             _overlapCancel->load(std::memory_order_relaxed);
    _overlapCancel.reset();

    if (!_pendingOverlap.viewer) {
        return;  // Viewer was detached while computing
    }

    QImage result = _overlapWatcher->result();
    if (result.isNull() && wasCanceled) {
        // Stale compute was canceled mid-run: don't cache it. refreshAll()
        // re-enters buildSurfaceOverlapOverlay, which starts the fresh compute.
        _pendingOverlap.viewer = nullptr;
        refreshAll();
        return;
    }
    _pendingOverlap.cache.image = std::move(result);
    _overlapCaches[_pendingOverlap.viewer] = std::move(_pendingOverlap.cache);
    _pendingOverlap.viewer = nullptr;

    refreshAll();
}
