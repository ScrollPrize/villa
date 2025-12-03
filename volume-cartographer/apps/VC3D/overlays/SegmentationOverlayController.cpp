#include "SegmentationOverlayController.hpp"

#include "../CSurfaceCollection.hpp"
#include "../CVolumeViewer.hpp"
#include "../ViewerManager.hpp"

#include <QColor>
#include <QDebug>
#include <QElapsedTimer>
#include <QPainter>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

namespace
{
constexpr const char* kOverlayGroupKey = "segmentation_vertex_edit";
const QColor kActiveFill = QColor(255, 215, 0, 220);
const QColor kActiveBorder = QColor(255, 180, 0, 255);
const QColor kNeighborFill = QColor(0, 220, 255, 160);
const QColor kNeighborBorder = QColor(255, 255, 255, 210);
const QColor kGrowthFill = QColor(140, 255, 160, 210);
const QColor kGrowthBorder = QColor(80, 190, 90, 230);
const QColor kHoverFill = QColor(255, 255, 255, 200);
const QColor kHoverBorder = QColor(255, 255, 255, 240);
constexpr qreal kMarkerRadius = 5.0;
constexpr qreal kMarkerPenWidth = 2.0;
constexpr qreal kActivePenWidth = 2.5;
constexpr qreal kMaskZ = 60.0;
constexpr qreal kApprovalMaskZ = 50.0;  // Below mask path (60) but above most other overlays
constexpr qreal kMarkerZ = 95.0;
constexpr qreal kRadiusCircleZ = 80.0;
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

    const auto maskEqual = [&](const std::vector<cv::Vec3f>& lhsVec,
                               const std::vector<cv::Vec3f>& rhsVec) {
        if (lhsVec.size() != rhsVec.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhsVec.size(); ++i) {
            const cv::Vec3f delta = lhsVec[i] - rhsVec[i];
            if (cv::norm(delta) >= 1e-4f) {
                return false;
            }
        }
        return true;
    };

    const auto floatEqual = [](float lhs, float rhs) {
        return std::fabs(lhs - rhs) < 1e-4f;
    };

    const auto vec3fOptEqual = [&](const std::optional<cv::Vec3f>& lhs, const std::optional<cv::Vec3f>& rhs) {
        if (!lhs && !rhs) return true;
        if (static_cast<bool>(lhs) != static_cast<bool>(rhs)) return false;
        return cv::norm(*lhs - *rhs) < 1e-4f;
    };

    return optionalEqual(activeMarker, rhs.activeMarker) &&
           vectorEqual(neighbours, rhs.neighbours) &&
           maskEqual(maskPoints, rhs.maskPoints) &&
           maskVisible == rhs.maskVisible &&
           brushActive == rhs.brushActive &&
           brushStrokeActive == rhs.brushStrokeActive &&
           lineStrokeActive == rhs.lineStrokeActive &&
           hasLineStroke == rhs.hasLineStroke &&
           pushPullActive == rhs.pushPullActive &&
           falloff == rhs.falloff &&
           floatEqual(gaussianRadiusSteps, rhs.gaussianRadiusSteps) &&
           floatEqual(gaussianSigmaSteps, rhs.gaussianSigmaSteps) &&
           floatEqual(displayRadiusSteps, rhs.displayRadiusSteps) &&
           floatEqual(gridStepWorld, rhs.gridStepWorld) &&
           approvalMaskMode == rhs.approvalMaskMode &&
           approvalStrokeActive == rhs.approvalStrokeActive &&
           approvalStrokeSegments == rhs.approvalStrokeSegments &&
           maskEqual(approvalCurrentStroke, rhs.approvalCurrentStroke) &&
           floatEqual(approvalBrushRadius, rhs.approvalBrushRadius) &&
           paintingApproval == rhs.paintingApproval &&
           surface == rhs.surface &&
           approvalHoverScenePos == rhs.approvalHoverScenePos &&
           floatEqual(approvalHoverViewerScale, rhs.approvalHoverViewerScale) &&
           vec3fOptEqual(approvalHoverPlaneNormal, rhs.approvalHoverPlaneNormal);
}

SegmentationOverlayController::SegmentationOverlayController(CSurfaceCollection* surfaces, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroupKey, parent)
    , _surfaces(surfaces)
{
    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged,
                this, &SegmentationOverlayController::onSurfaceChanged);
    }
}

void SegmentationOverlayController::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;
    refreshAll();
}

void SegmentationOverlayController::setEditManager(SegmentationEditManager* manager)
{
    _editManager = manager;
}

void SegmentationOverlayController::applyState(const State& state)
{
    State sanitized = state;
    sanitized.gaussianRadiusSteps = std::max(sanitized.gaussianRadiusSteps, 0.0f);
    sanitized.gaussianSigmaSteps = std::max(sanitized.gaussianSigmaSteps, 0.0f);
    sanitized.displayRadiusSteps = std::max(sanitized.displayRadiusSteps, 0.0f);
    sanitized.gridStepWorld = std::max(sanitized.gridStepWorld, 1e-4f);

    _currentState = std::move(sanitized);
    refreshAll();
}

void SegmentationOverlayController::loadApprovalMaskImage(QuadSurface* surface)
{
    if (!surface) {
        _savedApprovalMaskImage = QImage();
        _pendingApprovalMaskImage = QImage();
        return;
    }

    cv::Mat approvalMask = surface->channel("approval", SURF_CHANNEL_NORESIZE);
    if (approvalMask.empty()) {
        // Create new empty mask
        const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
        if (!points || points->empty()) {
            _savedApprovalMaskImage = QImage();
            _pendingApprovalMaskImage = QImage();
            return;
        }
        approvalMask = cv::Mat_<uint8_t>(points->size(), static_cast<uint8_t>(0));
    }

    // Convert saved mask to ARGB32_Premultiplied format with DARK GREEN, semi-transparent
    QImage savedImage(approvalMask.cols, approvalMask.rows, QImage::Format_ARGB32_Premultiplied);
    for (int row = 0; row < approvalMask.rows; ++row) {
        const uint8_t* maskRow = approvalMask.ptr<uint8_t>(row);
        QRgb* imageRow = reinterpret_cast<QRgb*>(savedImage.scanLine(row));
        for (int col = 0; col < approvalMask.cols; ++col) {
            uint8_t val = maskRow[col];
            if (val > 0) {
                // GREEN for saved approval: RGB(0, 200, 0) with 60% opacity (alpha = 153)
                // Premultiply: R=0, G=120, B=0, A=153
                imageRow[col] = qRgba(0, 120, 0, 153);
            } else {
                imageRow[col] = qRgba(0, 0, 0, 0);  // Fully transparent
            }
        }
    }

    _savedApprovalMaskImage = savedImage;

    // Initialize empty pending mask with same dimensions
    _pendingApprovalMaskImage = QImage(approvalMask.cols, approvalMask.rows, QImage::Format_ARGB32_Premultiplied);
    _pendingApprovalMaskImage.fill(qRgba(0, 0, 0, 0));  // Fully transparent

    // Invalidate all viewer caches
    ++_savedImageVersion;
    ++_pendingImageVersion;

    // Trigger re-rendering of intersection lines on plane viewers
    invalidatePlaneIntersections();
}

void SegmentationOverlayController::paintApprovalMaskDirect(
    const std::vector<std::pair<int, int>>& gridPositions,
    float radiusSteps,
    uint8_t paintValue,
    bool useRectangle,
    float widthSteps,
    float heightSteps)
{
    if (_pendingApprovalMaskImage.isNull() || gridPositions.empty()) {
        return;
    }

    const int radius = static_cast<int>(std::ceil(radiusSteps));

    // For rectangle mode (flattened view), use explicit width and height
    const int rectHalfWidth = useRectangle && widthSteps > 0
        ? static_cast<int>(std::ceil(widthSteps / 2.0f))
        : radius;
    const int rectHalfHeight = useRectangle && heightSteps > 0
        ? static_cast<int>(std::ceil(heightSteps / 2.0f))
        : radius;

    // Compute bounding box of affected region
    int minRow = std::numeric_limits<int>::max();
    int maxRow = std::numeric_limits<int>::min();
    int minCol = std::numeric_limits<int>::max();
    int maxCol = std::numeric_limits<int>::min();

    for (const auto& [row, col] : gridPositions) {
        minRow = std::min(minRow, row - rectHalfHeight);
        maxRow = std::max(maxRow, row + rectHalfHeight);
        minCol = std::min(minCol, col - rectHalfWidth);
        maxCol = std::max(maxCol, col + rectHalfWidth);
    }

    // Clamp to image bounds
    minRow = std::max(0, minRow);
    maxRow = std::min(_pendingApprovalMaskImage.height() - 1, maxRow);
    minCol = std::max(0, minCol);
    maxCol = std::min(_pendingApprovalMaskImage.width() - 1, maxCol);

    const int regionWidth = maxCol - minCol + 1;
    const int regionHeight = maxRow - minRow + 1;

    // Save the affected region for undo
    if (regionWidth > 0 && regionHeight > 0) {
        ApprovalMaskUndoEntry undoEntry;
        undoEntry.topLeft = QPoint(minCol, minRow);
        undoEntry.savedRegion = _pendingApprovalMaskImage.copy(minCol, minRow, regionWidth, regionHeight);
        _approvalMaskUndoStack.push_back(std::move(undoEntry));

        // Limit undo stack size
        if (_approvalMaskUndoStack.size() > kMaxUndoEntries) {
            _approvalMaskUndoStack.erase(_approvalMaskUndoStack.begin());
        }
    }

    // Paint directly into the PENDING QImage
    for (const auto& [centerRow, centerCol] : gridPositions) {
        // For rectangle: iterate over explicit width/height
        // For circle: iterate over full radius in both dimensions
        const int colRange = useRectangle ? rectHalfWidth : radius;
        const int rowRange = useRectangle ? rectHalfHeight : radius;

        for (int dr = -rowRange; dr <= rowRange; ++dr) {
            for (int dc = -colRange; dc <= colRange; ++dc) {
                const int row = centerRow + dr;
                const int col = centerCol + dc;

                if (row < 0 || row >= _pendingApprovalMaskImage.height() ||
                    col < 0 || col >= _pendingApprovalMaskImage.width()) {
                    continue;
                }

                // For circle mode, skip pixels outside the radius
                if (!useRectangle) {
                    const float distance = std::sqrt(static_cast<float>(dr * dr + dc * dc));
                    if (distance > radiusSteps) {
                        continue;
                    }
                }

                // Update pixel color based on paint mode:
                // - Blue for pending approval (paintValue = 255)
                // - Red for pending unapproval (paintValue = 0)
                if (paintValue > 0) {
                    // Approving: BLUE with 50% opacity
                    _pendingApprovalMaskImage.setPixel(col, row, qRgba(0, 50, 127, 128));
                } else {
                    // Unapproving: RED with 50% opacity
                    _pendingApprovalMaskImage.setPixel(col, row, qRgba(127, 25, 25, 128));
                }
            }
        }
    }

    // Invalidate pending version since we modified the pending image
    ++_pendingImageVersion;

    // Trigger re-rendering of intersection lines on plane viewers
    invalidatePlaneIntersections();
}

void SegmentationOverlayController::saveApprovalMaskToSurface(QuadSurface* surface)
{
    if (!surface || (_savedApprovalMaskImage.isNull() && _pendingApprovalMaskImage.isNull())) {
        return;
    }

    // Merge pending into saved, then convert to cv::Mat
    int width = !_savedApprovalMaskImage.isNull() ? _savedApprovalMaskImage.width() : _pendingApprovalMaskImage.width();
    int height = !_savedApprovalMaskImage.isNull() ? _savedApprovalMaskImage.height() : _pendingApprovalMaskImage.height();

    cv::Mat_<uint8_t> approvalMask(height, width, static_cast<uint8_t>(0));

    for (int row = 0; row < height; ++row) {
        uint8_t* maskRow = approvalMask.ptr<uint8_t>(row);
        for (int col = 0; col < width; ++col) {
            uint8_t savedVal = 0;

            // Get saved value (green channel indicates approval)
            if (!_savedApprovalMaskImage.isNull() && row < _savedApprovalMaskImage.height() && col < _savedApprovalMaskImage.width()) {
                const QRgb* savedRow = reinterpret_cast<const QRgb*>(_savedApprovalMaskImage.constScanLine(row));
                QRgb pixel = savedRow[col];
                if (qAlpha(pixel) > 0 && qGreen(pixel) > 0) {
                    savedVal = 255;  // Was approved
                }
            }

            // Check pending state - can be approval (blue) or unapproval (red)
            bool hasPending = false;
            bool pendingIsApproval = false;
            if (!_pendingApprovalMaskImage.isNull() && row < _pendingApprovalMaskImage.height() && col < _pendingApprovalMaskImage.width()) {
                const QRgb* pendingRow = reinterpret_cast<const QRgb*>(_pendingApprovalMaskImage.constScanLine(row));
                QRgb pixel = pendingRow[col];
                if (qAlpha(pixel) > 0) {
                    hasPending = true;
                    // Blue (approval) has higher blue than red, Red (unapproval) has higher red than blue
                    pendingIsApproval = qBlue(pixel) > qRed(pixel);
                }
            }

            // Apply pending changes
            if (hasPending) {
                if (pendingIsApproval) {
                    maskRow[col] = 255;  // Approve
                } else {
                    maskRow[col] = 0;    // Unapprove (clear)
                }
            } else {
                // No pending change, keep saved value
                maskRow[col] = savedVal;
            }
        }
    }

    // Save to surface
    surface->setChannel("approval", approvalMask);
    surface->saveOverwrite();

    // Merge pending into saved and clear pending
    for (int row = 0; row < height; ++row) {
        QRgb* savedRow = reinterpret_cast<QRgb*>(_savedApprovalMaskImage.scanLine(row));
        for (int col = 0; col < width; ++col) {
            uint8_t mergedVal = approvalMask(row, col);
            if (mergedVal > 0) {
                // Update saved image with dark green
                savedRow[col] = qRgba(0, 60, 0, 153);
            } else {
                savedRow[col] = qRgba(0, 0, 0, 0);
            }
        }
    }

    // Clear pending
    _pendingApprovalMaskImage.fill(qRgba(0, 0, 0, 0));

    // Invalidate both versions
    ++_savedImageVersion;
    ++_pendingImageVersion;

    // Trigger re-rendering of intersection lines on plane viewers
    invalidatePlaneIntersections();

    // Clear undo history since changes are now persisted
    clearApprovalMaskUndoHistory();
}

bool SegmentationOverlayController::undoLastApprovalMaskPaint()
{
    if (_approvalMaskUndoStack.empty() || _pendingApprovalMaskImage.isNull()) {
        return false;
    }

    // Pop the last entry
    ApprovalMaskUndoEntry entry = std::move(_approvalMaskUndoStack.back());
    _approvalMaskUndoStack.pop_back();

    // Restore the saved region
    QPainter painter(&_pendingApprovalMaskImage);
    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.drawImage(entry.topLeft, entry.savedRegion);
    painter.end();

    ++_pendingImageVersion;
    invalidatePlaneIntersections();
    return true;
}

bool SegmentationOverlayController::canUndoApprovalMaskPaint() const
{
    return !_approvalMaskUndoStack.empty();
}

void SegmentationOverlayController::clearApprovalMaskUndoHistory()
{
    _approvalMaskUndoStack.clear();
}

bool SegmentationOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    Q_UNUSED(viewer);
    // Enable overlay rendering if editing is enabled OR if approval mask mode is active
    // (approval mask can be viewed without editing enabled)
    const bool approvalMaskActive = _currentState && _currentState->approvalMaskMode;
    return _editingEnabled || approvalMaskActive;
}

void SegmentationOverlayController::collectPrimitives(CVolumeViewer* viewer,
                                                      ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer || !_currentState) {
        return;
    }

    const State& state = *_currentState;

    // Render approval mask overlays regardless of editing enabled
    // (but painting requires editing to be enabled - handled in SegmentationModule)
    if (state.approvalMaskMode && state.surface) {
        buildApprovalMaskOverlay(state, viewer, builder);
    }

    // Other overlays require editing to be enabled
    if (!_editingEnabled) {
        return;
    }

    if (shouldShowMask(state)) {
        builder.addPath(buildMaskPrimitive(state));
    }

    buildVertexMarkers(state, viewer, builder);
    buildRadiusOverlay(state, viewer, builder);
}

ViewerOverlayControllerBase::PathPrimitive SegmentationOverlayController::buildMaskPrimitive(const State& state) const
{
    ViewerOverlayControllerBase::PathPrimitive maskPath;
    maskPath.points = state.maskPoints;
    maskPath.renderMode = state.hasLineStroke ? ViewerOverlayControllerBase::PathRenderMode::LineStrip
                                              : ViewerOverlayControllerBase::PathRenderMode::Points;
    maskPath.brushShape = ViewerOverlayControllerBase::PathBrushShape::Circle;

    const float brushPixelRadius = std::clamp(state.displayRadiusSteps * 1.5f, 3.0f, 18.0f);
    const bool drawingOverlay = state.brushActive || state.brushStrokeActive || state.lineStrokeActive || state.hasLineStroke;
    const float brushOpacity = drawingOverlay ? 0.6f : 0.45f;

    maskPath.pointRadius = state.hasLineStroke ? std::max(brushPixelRadius * 0.35f, 2.0f) : brushPixelRadius;
    maskPath.lineWidth = state.hasLineStroke ? 3.0f : std::max(brushPixelRadius * 0.5f, 2.0f);
    maskPath.color = state.hasLineStroke ? QColor(80, 170, 255) : QColor(255, 140, 0);
    maskPath.opacity = state.hasLineStroke ? 0.85f : brushOpacity;
    maskPath.z = kMaskZ;

    return maskPath;
}

bool SegmentationOverlayController::shouldShowMask(const State& state) const
{
    return state.maskVisible && !state.maskPoints.empty();
}

void SegmentationOverlayController::onSurfaceChanged(std::string name, Surface* surface)
{
    Q_UNUSED(surface);
    if (name == "segmentation") {
        refreshAll();
    }
}

void SegmentationOverlayController::buildRadiusOverlay(const State& state,
                                                       CVolumeViewer* viewer,
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
    const QPointF sceneCenter = viewer->volumePointToScene(world);

    cv::Vec3f offsetWorld = world;
    offsetWorld[0] += radiusWorld;
    const QPointF sceneEdge = viewer->volumePointToScene(offsetWorld);
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
                                                       CVolumeViewer* viewer,
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
        const QPointF scene = viewer->volumePointToScene(marker.world);
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

void SegmentationOverlayController::rebuildViewerCache(CVolumeViewer* viewer, QuadSurface* surface) const
{
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const int gridRows = points->rows;
    const int gridCols = points->cols;

    const bool hasSaved = !_savedApprovalMaskImage.isNull() &&
                         _savedApprovalMaskImage.width() == gridCols &&
                         _savedApprovalMaskImage.height() == gridRows;
    const bool hasPending = !_pendingApprovalMaskImage.isNull() &&
                           _pendingApprovalMaskImage.width() == gridCols &&
                           _pendingApprovalMaskImage.height() == gridRows;

    if (!hasSaved && !hasPending) {
        return;
    }

    // Simple approach: composite saved and pending images using QPainter (fast!)
    QImage compositeImage(gridCols, gridRows, QImage::Format_ARGB32_Premultiplied);
    compositeImage.fill(Qt::transparent);

    QPainter painter(&compositeImage);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

    // Draw saved (dark green) first
    if (hasSaved) {
        painter.drawImage(0, 0, _savedApprovalMaskImage);
    }

    // Draw pending (light green) on top
    if (hasPending) {
        painter.drawImage(0, 0, _pendingApprovalMaskImage);
    }

    painter.end();

    // Calculate scene-space bounds using direct grid-to-scene formula (no pointTo!)
    // The relationship between scene coords and grid coords:
    // - scene2vol: surfLoc = scenePos / dsScale, then coord() uses (surfLoc + center) * surfScale
    // - So: gridPos = (scenePos/dsScale + center) * surfScale
    // - Inverting: scenePos = (gridPos/surfScale - center) * dsScale
    // BUT the rendering uses viewerScale (_scale), so we need to use that for overlay positioning
    const cv::Vec3f center = surface->center();
    const cv::Vec2f surfScale = surface->scale();
    const float viewerScale = viewer->getCurrentScale();

    // Formula: scenePos = (gridPos/surfScale - center) * viewerScale
    auto gridToScene = [&](int row, int col) -> QPointF {
        const float surfLocalX = static_cast<float>(col) / surfScale[0] - center[0];
        const float surfLocalY = static_cast<float>(row) / surfScale[1] - center[1];
        const float sceneX = surfLocalX * viewerScale;
        const float sceneY = surfLocalY * viewerScale;
        return QPointF(sceneX, sceneY);
    };

    // Calculate corners directly
    QPointF topLeft = gridToScene(0, 0);
    QPointF topRight = gridToScene(0, gridCols - 1);
    QPointF bottomLeft = gridToScene(gridRows - 1, 0);
    QPointF bottomRight = gridToScene(gridRows - 1, gridCols - 1);

    QRectF sceneBounds;
    sceneBounds.setLeft(std::min({topLeft.x(), topRight.x(), bottomLeft.x(), bottomRight.x()}));
    sceneBounds.setRight(std::max({topLeft.x(), topRight.x(), bottomLeft.x(), bottomRight.x()}));
    sceneBounds.setTop(std::min({topLeft.y(), topRight.y(), bottomLeft.y(), bottomRight.y()}));
    sceneBounds.setBottom(std::max({topLeft.y(), topRight.y(), bottomLeft.y(), bottomRight.y()}));

    // Calculate scale to map from grid-space to scene-space
    const qreal sceneWidth = sceneBounds.width();
    const qreal sceneHeight = sceneBounds.height();
    const qreal scaleX = sceneWidth / static_cast<qreal>(gridCols);
    const qreal scaleY = sceneHeight / static_cast<qreal>(gridRows);
    const qreal scale = std::max(scaleX, scaleY);

    // Store in cache
    ViewerImageCache& cache = _viewerCaches[viewer];
    cache.compositeImage = compositeImage;
    cache.topLeft = sceneBounds.topLeft();
    cache.scale = scale;
    cache.surface = surface;
    cache.savedImageVersion = _savedImageVersion;
    cache.pendingImageVersion = _pendingImageVersion;
}

int SegmentationOverlayController::queryApprovalStatus(int row, int col) const
{
    // Check pending first (takes priority for display)
    // Returns: 0 = not approved, 1 = saved approved, 2 = pending approved, 3 = pending unapproved
    if (!_pendingApprovalMaskImage.isNull() &&
        row >= 0 && row < _pendingApprovalMaskImage.height() &&
        col >= 0 && col < _pendingApprovalMaskImage.width()) {
        QRgb pixel = _pendingApprovalMaskImage.pixel(col, row);
        if (qAlpha(pixel) > 0) {
            // Distinguish between pending approve (blue) and pending unapprove (red)
            // Blue has high blue component, red has high red component
            if (qRed(pixel) > qBlue(pixel)) {
                return 3;  // Pending unapproval (red)
            }
            return 2;  // Pending approval (blue)
        }
    }

    // Check saved
    if (!_savedApprovalMaskImage.isNull() &&
        row >= 0 && row < _savedApprovalMaskImage.height() &&
        col >= 0 && col < _savedApprovalMaskImage.width()) {
        QRgb pixel = _savedApprovalMaskImage.pixel(col, row);
        if (qAlpha(pixel) > 0) {
            return 1;  // Saved
        }
    }

    return 0;  // Not approved
}

float SegmentationOverlayController::sampleImageBilinear(const QImage& image, float row, float col)
{
    if (image.isNull()) {
        return 0.0f;
    }

    const int width = image.width();
    const int height = image.height();

    // Get the four surrounding pixel coordinates
    const int col0 = static_cast<int>(std::floor(col));
    const int col1 = col0 + 1;
    const int row0 = static_cast<int>(std::floor(row));
    const int row1 = row0 + 1;

    // Compute interpolation weights
    const float colFrac = col - static_cast<float>(col0);
    const float rowFrac = row - static_cast<float>(row0);

    // Sample the four corners (clamped to image bounds)
    auto sampleAlpha = [&](int r, int c) -> float {
        if (r < 0 || r >= height || c < 0 || c >= width) {
            return 0.0f;
        }
        return static_cast<float>(qAlpha(image.pixel(c, r)));
    };

    const float v00 = sampleAlpha(row0, col0);
    const float v01 = sampleAlpha(row0, col1);
    const float v10 = sampleAlpha(row1, col0);
    const float v11 = sampleAlpha(row1, col1);

    // Bilinear interpolation
    const float v0 = v00 * (1.0f - colFrac) + v01 * colFrac;
    const float v1 = v10 * (1.0f - colFrac) + v11 * colFrac;
    return v0 * (1.0f - rowFrac) + v1 * rowFrac;
}

float SegmentationOverlayController::queryApprovalBilinear(float row, float col, int* outStatus) const
{
    // First check pending mask (takes priority)
    const float pendingAlpha = sampleImageBilinear(_pendingApprovalMaskImage, row, col);
    if (pendingAlpha > 0.5f) {  // Threshold to determine if we're "in" pending region
        // Determine if pending approve or unapprove by checking nearest pixel color
        const int nearestRow = static_cast<int>(std::round(row));
        const int nearestCol = static_cast<int>(std::round(col));
        if (!_pendingApprovalMaskImage.isNull() &&
            nearestRow >= 0 && nearestRow < _pendingApprovalMaskImage.height() &&
            nearestCol >= 0 && nearestCol < _pendingApprovalMaskImage.width()) {
            QRgb pixel = _pendingApprovalMaskImage.pixel(nearestCol, nearestRow);
            if (qAlpha(pixel) > 0) {
                if (outStatus) {
                    *outStatus = (qRed(pixel) > qBlue(pixel)) ? 3 : 2;  // 3 = unapprove, 2 = approve
                }
                return pendingAlpha / 255.0f;
            }
        }
    }

    // Check saved mask with bilinear interpolation
    const float savedAlpha = sampleImageBilinear(_savedApprovalMaskImage, row, col);
    if (savedAlpha > 0.0f) {
        if (outStatus) {
            *outStatus = 1;  // Saved
        }
        return savedAlpha / 255.0f;
    }

    if (outStatus) {
        *outStatus = 0;  // Not approved
    }
    return 0.0f;
}

bool SegmentationOverlayController::hasApprovalMaskData() const
{
    if (!_currentState.has_value() || !_currentState->approvalMaskMode) {
        return false;
    }
    return !_savedApprovalMaskImage.isNull() || !_pendingApprovalMaskImage.isNull();
}

void SegmentationOverlayController::invalidatePlaneIntersections()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
        if (!viewer) {
            return;
        }
        // Only invalidate for plane surface viewers (XY, XZ, YZ)
        if (dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            viewer->renderIntersections();
        }
    });
}

void SegmentationOverlayController::buildApprovalMaskOverlay(const State& state,
                                                              CVolumeViewer* viewer,
                                                              ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    if (!state.surface) {
        return;
    }

    // Check if this viewer is displaying a PlaneSurface (XY/XZ/YZ orthogonal view)
    // For plane viewers, the approval mask is rendered via modified intersection lines
    // in CVolumeViewerIntersections.cpp, not here
    Surface* viewerSurf = viewer->currentSurface();
    const bool isPlaneViewer = dynamic_cast<PlaneSurface*>(viewerSurf) != nullptr;

    // Draw brush reticle at hover position
    // Show when we have a hover position (which only exists in edit approval mask mode)
    // Flat cylinder model: circle in XY/XZ/YZ planes, rectangle in flattened view
    if (state.approvalHoverWorld) {
        const cv::Vec3f& hoverWorld = *state.approvalHoverWorld;
        const float brushRadiusNative = state.approvalBrushRadius;

        if (isPlaneViewer) {
            // For plane viewers: use volumePointToScene which is fast (O(1) for PlaneSurface)
            // Compute center and radius consistently using the same projection
            const QPointF sceneCenter = viewer->volumePointToScene(hoverWorld);

            // Convert brush radius from native voxels to scene pixels
            // Use both X and Y offsets to handle different plane orientations
            const cv::Vec3f offsetPosX = hoverWorld + cv::Vec3f(brushRadiusNative, 0, 0);
            const cv::Vec3f offsetPosY = hoverWorld + cv::Vec3f(0, brushRadiusNative, 0);
            const QPointF sceneOffsetX = viewer->volumePointToScene(offsetPosX);
            const QPointF sceneOffsetY = viewer->volumePointToScene(offsetPosY);
            const qreal radiusPixelsX = std::hypot(sceneOffsetX.x() - sceneCenter.x(),
                                                    sceneOffsetX.y() - sceneCenter.y());
            const qreal radiusPixelsY = std::hypot(sceneOffsetY.x() - sceneCenter.x(),
                                                    sceneOffsetY.y() - sceneCenter.y());
            // Use whichever axis projects into the view (the other will be ~0)
            const qreal radiusPixels = std::max(radiusPixelsX, radiusPixelsY);

            if (radiusPixels > 1.0) {
                ViewerOverlayControllerBase::OverlayStyle style;
                style.penColor = state.paintingApproval ? QColor(0, 100, 255, 200) : QColor(255, 80, 80, 200);
                style.penWidth = 2.0;
                style.brushColor = Qt::transparent;
                style.penStyle = Qt::DashLine;
                style.dashPattern = {4.0, 4.0};  // Dashed pattern
                style.z = kApprovalMaskZ + 10.0;

                // XY/XZ/YZ planes: draw a circle (cylinder cross-section)
                builder.addCircle(sceneCenter, radiusPixels, false, style);
            }
        } else {
            // For segmentation/flattened view: convert world position to scene coordinates
            // Draw a rectangle (cylinder side view): Width = 2 * radius (diameter), Height = depth
            const float brushDepthNative = state.approvalBrushDepth;
            const float thisViewerScale = viewer->getCurrentScale();

            // Convert world position to scene coordinates
            // This uses volumePointToScene which calls pointTo for QuadSurface
            const QPointF sceneCenter = viewer->volumePointToScene(hoverWorld);

            // Convert from native voxels to grid units
            float surfaceScale = 1.0f;
            if (state.surface) {
                const cv::Vec2f scale = state.surface->scale();
                surfaceScale = (scale[0] + scale[1]) * 0.5f;
            }
            const float gridRadius = brushRadiusNative * surfaceScale;
            const float gridDepth = brushDepthNative * surfaceScale;

            // Convert grid units to scene pixels using viewer scale
            const qreal gridToScene = thisViewerScale / surfaceScale;

            // Add a small offset to account for painting extending to cell edges
            constexpr float gridOffset = 0.5f;
            const qreal rectHalfWidth = (gridRadius + gridOffset) * gridToScene;
            const qreal rectHalfHeight = (gridDepth / 2.0f + gridOffset) * gridToScene;

            if (rectHalfWidth > 1.0 && rectHalfHeight > 1.0) {
                ViewerOverlayControllerBase::OverlayStyle style;
                style.penColor = state.paintingApproval ? QColor(0, 100, 255, 200) : QColor(255, 80, 80, 200);
                style.penWidth = 2.0;
                style.brushColor = Qt::transparent;
                style.penStyle = Qt::DashLine;
                style.dashPattern = {4.0, 4.0};  // Dashed pattern
                style.z = kApprovalMaskZ + 10.0;

                // Determine rectangle orientation based on cylinder axis (plane normal)
                qreal rotationDegrees = 0.0;
                if (state.approvalHoverPlaneNormal) {
                    // When hovering in XY/XZ/YZ planes, orient the rectangle along the cylinder axis
                    // Project the cylinder axis (plane normal) into the flattened view
                    const cv::Vec3f& normal = *state.approvalHoverPlaneNormal;
                    const cv::Vec3f axisEndWorld = hoverWorld + normal * brushDepthNative;
                    const QPointF axisEndScene = viewer->volumePointToScene(axisEndWorld);

                    // Compute angle from center to axis end
                    const qreal dx = axisEndScene.x() - sceneCenter.x();
                    const qreal dy = axisEndScene.y() - sceneCenter.y();
                    if (std::abs(dx) > 0.1 || std::abs(dy) > 0.1) {
                        // atan2 gives angle from positive X axis, we want rotation where
                        // the rectangle's height (Y) aligns with the cylinder axis
                        rotationDegrees = std::atan2(dy, dx) * 180.0 / M_PI - 90.0;
                    }
                }

                if (std::abs(rotationDegrees) < 0.1) {
                    // No rotation needed - draw axis-aligned rectangle
                    const QRectF rect(sceneCenter.x() - rectHalfWidth,
                                      sceneCenter.y() - rectHalfHeight,
                                      rectHalfWidth * 2.0,
                                      rectHalfHeight * 2.0);
                    builder.addRect(rect, false, style);
                } else {
                    // Draw rotated rectangle as a closed line strip
                    const qreal angleRad = rotationDegrees * M_PI / 180.0;
                    const qreal cosA = std::cos(angleRad);
                    const qreal sinA = std::sin(angleRad);

                    // Rectangle corners before rotation (centered at origin)
                    // Width along X, Height along Y
                    const std::array<QPointF, 4> corners = {{
                        {-rectHalfWidth, -rectHalfHeight},
                        { rectHalfWidth, -rectHalfHeight},
                        { rectHalfWidth,  rectHalfHeight},
                        {-rectHalfWidth,  rectHalfHeight}
                    }};

                    std::vector<QPointF> points;
                    points.reserve(4);
                    for (const auto& corner : corners) {
                        // Rotate and translate
                        const qreal rx = corner.x() * cosA - corner.y() * sinA + sceneCenter.x();
                        const qreal ry = corner.x() * sinA + corner.y() * cosA + sceneCenter.y();
                        points.emplace_back(rx, ry);
                    }
                    builder.addLineStrip(points, true, style);
                }
            }
        }
    }

    if (isPlaneViewer) {
        return;  // Mask image handled by renderIntersections()
    }

    // For segmentation view (QuadSurface), render as image overlay
    QElapsedTimer buildTimer;
    buildTimer.start();

    // Check if we need to rebuild the COMPOSITE IMAGE (only when masks change, not when view changes)
    auto it = _viewerCaches.find(viewer);
    bool needsRebuild = (it == _viewerCaches.end()) ||
                       (it->second.surface != state.surface) ||
                       (it->second.savedImageVersion != _savedImageVersion) ||
                       (it->second.pendingImageVersion != _pendingImageVersion);

    if (needsRebuild) {
        rebuildViewerCache(viewer, state.surface);
        it = _viewerCaches.find(viewer);
    }

    // If cache still doesn't exist or is empty, nothing to render
    if (it == _viewerCaches.end() || it->second.compositeImage.isNull()) {
        return;
    }

    const ViewerImageCache& cache = it->second;

    // Get surface parameters for direct grid-to-scene coordinate conversion
    // The rendering uses viewerScale, so we need to use that for overlay positioning
    const cv::Vec3f center = state.surface->center();
    const cv::Vec2f surfScale = state.surface->scale();
    const float viewerScale = viewer->getCurrentScale();

    // Find valid reference points and calculate the current grid-to-scene scale dynamically
    const cv::Mat_<cv::Vec3f>* points = state.surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const int gridCols = points->cols;

    // Lambda to convert grid index directly to scene position (no pointTo!)
    // Formula: scenePos = (gridPos/surfScale - center) * viewerScale
    auto gridToScene = [&](int row, int col) -> QPointF {
        const float sceneX = (static_cast<float>(col) / surfScale[0] - center[0]) * viewerScale;
        const float sceneY = (static_cast<float>(row) / surfScale[1] - center[1]) * viewerScale;
        return QPointF(sceneX, sceneY);
    };

    // Calculate grid-to-scene scale from adjacent cells
    QPointF p0 = gridToScene(0, 0);
    QPointF p1 = gridToScene(0, 1);
    qreal gridToSceneScale = std::hypot(p1.x() - p0.x(), p1.y() - p0.y());

    if (gridToSceneScale < 1e-6) {
        return;
    }

    // Render the composite image as a single scaled image overlay
    // This is much faster than rendering individual rectangles
    QPointF topLeft = gridToScene(0, 0);
    builder.addImage(cache.compositeImage, topLeft, gridToSceneScale, 1.0, kApprovalMaskZ);
}

