#include "ApprovalMaskBrushTool.hpp"

#include "SegmentationEditManager.hpp"
#include "SegmentationModule.hpp"
#include "SegmentationWidget.hpp"
#include "../overlays/SegmentationOverlayController.hpp"

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <limits>

#include "vc/core/util/QuadSurface.hpp"

Q_DECLARE_LOGGING_CATEGORY(lcApprovalMask)
Q_LOGGING_CATEGORY(lcApprovalMask, "vc.segmentation.approvalmask")

namespace
{
constexpr float kBrushSampleSpacing = 2.0f;       // For accurate stroke data
constexpr float kOverlayPointSpacing = 20.0f;     // For visual overlay (much sparser)

// Check if a point is invalid (NaN, infinity, or the -1,-1,-1 marker)
bool isInvalidPoint(const cv::Vec3f& value)
{
    return !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
           (value[0] == -1.0f && value[1] == -1.0f && value[2] == -1.0f);
}

// Gaussian falloff function
float gaussianFalloff(float distance, float sigma)
{
    if (sigma <= 0.0f) {
        return distance <= 0.0f ? 1.0f : 0.0f;
    }
    return std::exp(-(distance * distance) / (2.0f * sigma * sigma));
}
}

ApprovalMaskBrushTool::ApprovalMaskBrushTool(SegmentationModule& module,
                                             SegmentationEditManager* editManager,
                                             SegmentationWidget* widget)
    : _module(module)
    , _editManager(editManager)
    , _widget(widget)
{
}

void ApprovalMaskBrushTool::setDependencies(SegmentationWidget* widget)
{
    _widget = widget;
}

void ApprovalMaskBrushTool::setSurface(QuadSurface* surface)
{
    _surface = surface;
    qCInfo(lcApprovalMask) << "Surface set on approval tool:" << (surface ? "valid" : "null");
    if (surface) {
        qCInfo(lcApprovalMask) << "  Surface ID:" << QString::fromStdString(surface->id);
        const auto* points = surface->rawPointsPtr();
        if (points) {
            qCInfo(lcApprovalMask) << "  Surface size:" << points->cols << "x" << points->rows;
        }
    }
}

void ApprovalMaskBrushTool::setActive(bool active)
{
    if (_brushActive == active) {
        return;
    }

    _brushActive = active;
    qCInfo(lcApprovalMask) << "Approval brush active:" << active;
    if (!_brushActive) {
        _hasLastSample = false;
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::startStroke(const cv::Vec3f& worldPos, const QPointF& scenePos, float viewerScale)
{
    qCInfo(lcApprovalMask) << "Starting approval stroke at:" << worldPos[0] << worldPos[1] << worldPos[2]
                           << "scenePos:" << scenePos.x() << scenePos.y() << "viewerScale:" << viewerScale;
    qCInfo(lcApprovalMask) << "  Surface:" << (_surface ? "valid" : "NULL");
    _strokeActive = true;
    _currentStroke.clear();
    _currentStroke.push_back(worldPos);

    // Clear overlay points to start fresh - prevents connecting to previous strokes
    _overlayPoints.clear();
    _overlayPoints.push_back(worldPos);

    _lastSample = worldPos;
    _hasLastSample = true;
    _lastOverlaySample = worldPos;
    _hasLastOverlaySample = true;

    // Initialize throttling timer
    _lastRefreshTimer.start();
    _lastRefreshTime = 0;
    _pendingRefresh = false;

    // Clear accumulated grid positions for real-time painting
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    // Add the starting point for painting - compute grid position from scene coordinates
    auto gridIdx = sceneToGridIndex(scenePos, viewerScale);
    if (gridIdx) {
        qCInfo(lcApprovalMask) << "  Grid index:" << gridIdx->first << gridIdx->second;
        const uint64_t hash = (static_cast<uint64_t>(gridIdx->first) << 32) | static_cast<uint64_t>(gridIdx->second);
        _accumulatedGridPosSet.insert(hash);
        _accumulatedGridPositions.push_back(*gridIdx);
    } else {
        qCInfo(lcApprovalMask) << "  Grid index: OUT OF BOUNDS";
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::extendStroke(const cv::Vec3f& worldPos, const QPointF& scenePos, float viewerScale, bool forceSample)
{
    if (!_strokeActive) {
        return;
    }

    // Check if position is within valid surface bounds using scene coordinates
    auto gridIdx = sceneToGridIndex(scenePos, viewerScale);
    if (!gridIdx) {
        // Outside valid surface area - break the current stroke segment
        // but keep stroke active so we can start a new segment when back in bounds
        if (!_currentStroke.empty()) {
            _pendingStrokes.push_back(_currentStroke);
            _currentStroke.clear();
        }

        // Save current overlay segment and clear for next segment
        if (!_overlayPoints.empty()) {
            _overlayStrokeSegments.push_back(_overlayPoints);
            _overlayPoints.clear();
        }

        // Reset last sample tracking so we don't interpolate across the gap
        _hasLastSample = false;
        _hasLastOverlaySample = false;
        return;
    }

    // Back in bounds - if we were out of bounds, this starts a new stroke segment
    // (no logging needed here)

    const float spacing = kBrushSampleSpacing;
    const float spacingSq = spacing * spacing;

    // Sample stroke data at high resolution (every 2.0 units)
    if (_hasLastSample) {
        const cv::Vec3f delta = worldPos - _lastSample;
        const float distanceSq = delta.dot(delta);
        if (!forceSample && distanceSq < spacingSq) {
            return;
        }

        const float distance = std::sqrt(distanceSq);
        if (distance > spacing) {
            const cv::Vec3f direction = delta / distance;
            float travelled = spacing;
            while (travelled < distance) {
                const cv::Vec3f intermediate = _lastSample + direction * travelled;
                _currentStroke.push_back(intermediate);
                travelled += spacing;
            }
        }
    }

    _currentStroke.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;

    // Accumulate grid position for real-time painting (reuse gridIdx from above)
    // We know gridIdx is valid here because we would have returned early if it was nullopt
    const uint64_t hash = (static_cast<uint64_t>(gridIdx->first) << 32) | static_cast<uint64_t>(gridIdx->second);
    if (_accumulatedGridPosSet.insert(hash).second) {
        _accumulatedGridPositions.push_back(*gridIdx);
    }

    // Paint accumulated points periodically (every 20 points or forceSample)
    constexpr size_t kPaintBatchSize = 20;
    if (forceSample || _accumulatedGridPositions.size() >= kPaintBatchSize) {
        paintAccumulatedPointsToImage();
    }

    // Sample overlay points at much lower resolution (every 20.0 units) for performance
    const float overlaySpacing = kOverlayPointSpacing;
    const float overlaySpacingSq = overlaySpacing * overlaySpacing;

    bool overlayNeedsRefresh = false;
    if (_hasLastOverlaySample) {
        const cv::Vec3f overlayDelta = worldPos - _lastOverlaySample;
        const float overlayDistSq = overlayDelta.dot(overlayDelta);
        if (forceSample || overlayDistSq >= overlaySpacingSq) {
            _overlayPoints.push_back(worldPos);
            _lastOverlaySample = worldPos;
            overlayNeedsRefresh = true;
        }
    } else {
        _overlayPoints.push_back(worldPos);
        _lastOverlaySample = worldPos;
        _hasLastOverlaySample = true;
        overlayNeedsRefresh = true;
    }

    // Only refresh overlay when we actually add a new overlay point (every 20 units)
    // AND throttle to max 20 FPS (50ms minimum interval) to avoid excessive redraws
    if (overlayNeedsRefresh) {
        const qint64 currentTime = _lastRefreshTimer.elapsed();
        const qint64 timeSinceLastRefresh = currentTime - _lastRefreshTime;
        constexpr qint64 kMinRefreshIntervalMs = 50;  // 20 FPS max

        if (timeSinceLastRefresh >= kMinRefreshIntervalMs) {
            _module.refreshOverlay();
            _lastRefreshTime = currentTime;
            _pendingRefresh = false;
        } else {
            // Refresh was skipped due to throttling, mark as pending
            _pendingRefresh = true;
        }
    }
}

void ApprovalMaskBrushTool::finishStroke()
{
    if (!_strokeActive) {
        return;
    }

    // Paint any remaining accumulated points
    if (!_accumulatedGridPositions.empty()) {
        paintAccumulatedPointsToImage();
    }

    _strokeActive = false;
    if (!_currentStroke.empty()) {
        _pendingStrokes.push_back(_currentStroke);
    }
    _currentStroke.clear();

    // Save current overlay segment to keep it visible
    if (!_overlayPoints.empty()) {
        _overlayStrokeSegments.push_back(_overlayPoints);
        _overlayPoints.clear();
    }

    _hasLastSample = false;
    _hasLastOverlaySample = false;

    // Refresh on finish to show final state (even if throttled during drawing)
    if (_pendingRefresh) {
        _pendingRefresh = false;
    }
    _module.refreshOverlay();
}

bool ApprovalMaskBrushTool::applyPending(float /*dragRadiusSteps*/)
{
    QElapsedTimer totalTimer;
    totalTimer.start();

    if (!_surface) {
        qCWarning(lcApprovalMask) << "Cannot apply: no surface";
        return false;
    }

    if (_strokeActive) {
        finishStroke();
    }

    // Since we're painting in real-time, just save the QImage to disk
    auto overlay = _module.overlay();
    if (!overlay) {
        qCWarning(lcApprovalMask) << "Cannot apply: no overlay controller";
        return false;
    }

    // Save the approval mask QImage to disk
    overlay->saveApprovalMaskToSurface(_surface);

    qCInfo(lcApprovalMask) << "Saved approval mask to disk in" << totalTimer.elapsed() << "ms";

    // Clear pending strokes and overlay segments (but keep the painted QImage)
    _strokeActive = false;
    _currentStroke.clear();
    _pendingStrokes.clear();
    _overlayPoints.clear();
    _overlayStrokeSegments.clear();
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    _module.refreshOverlay();

    Q_EMIT _module.statusMessageRequested(
        QCoreApplication::translate("ApprovalMaskBrushTool", "Applied approval mask to surface."),
        2000);

    return true;
}

void ApprovalMaskBrushTool::clear()
{
    _strokeActive = false;
    _currentStroke.clear();
    _pendingStrokes.clear();
    _overlayPoints.clear();
    _overlayStrokeSegments.clear();
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();
    _hasLastSample = false;
    _hasLastOverlaySample = false;

    // Reload approval mask from disk to discard pending changes
    auto overlay = _module.overlay();
    if (overlay && _surface) {
        overlay->loadApprovalMaskImage(_surface);
        qCInfo(lcApprovalMask) << "Reloaded approval mask from disk (discarded pending changes)";
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::paintAccumulatedPointsToImage()
{
    if (_accumulatedGridPositions.empty()) {
        return;
    }

    auto overlay = _module.overlay();
    if (!overlay) {
        qCWarning(lcApprovalMask) << "Cannot paint: no overlay controller";
        _accumulatedGridPositions.clear();
        _accumulatedGridPosSet.clear();
        return;
    }

    const uint8_t paintValue = (_paintMode == PaintMode::Approve) ? 255 : 0;

    // Get the current brush radius from the module
    const float brushRadius = _module.approvalMaskBrushRadius();
    const float clampedRadius = std::clamp(brushRadius, 0.5f, 500.0f);

    // Paint the accumulated points into the QImage
    overlay->paintApprovalMaskDirect(_accumulatedGridPositions, clampedRadius, paintValue);

    // Clear for next batch
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    // Trigger overlay refresh to show the updated image
    _module.refreshOverlay();
}

std::vector<std::pair<int, int>> ApprovalMaskBrushTool::findGridCellsInSphere(const cv::Vec3f& worldPos, float radius) const
{
    std::vector<std::pair<int, int>> result;

    if (!_surface) {
        return result;
    }

    const cv::Mat_<cv::Vec3f>* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        return result;
    }

    const float radiusSq = radius * radius;
    const int rows = points->rows;
    const int cols = points->cols;

    // For efficiency, we use the SurfacePatchIndex if available, but for now
    // we'll do a simple scan of all points. This can be optimized later.
    // Since the brush radius is typically small relative to the surface,
    // we could use a spatial index, but a full scan works for moderate surfaces.

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const cv::Vec3f& pt = (*points)(row, col);

            // Skip invalid points
            if (isInvalidPoint(pt)) {
                continue;
            }

            // Check if point is within sphere
            const cv::Vec3f delta = pt - worldPos;
            const float distSq = delta.dot(delta);
            if (distSq <= radiusSq) {
                result.emplace_back(row, col);
            }
        }
    }

    return result;
}

void ApprovalMaskBrushTool::startStrokeFromWorld(const cv::Vec3f& worldPos, float worldRadius)
{
    qCInfo(lcApprovalMask) << "Starting approval stroke from world pos:" << worldPos[0] << worldPos[1] << worldPos[2]
                           << "radius:" << worldRadius;
    _strokeActive = true;
    _currentStroke.clear();
    _currentStroke.push_back(worldPos);

    _overlayPoints.clear();
    _overlayPoints.push_back(worldPos);

    _lastSample = worldPos;
    _hasLastSample = true;
    _lastOverlaySample = worldPos;
    _hasLastOverlaySample = true;

    _lastRefreshTimer.start();
    _lastRefreshTime = 0;
    _pendingRefresh = false;

    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    // Find all grid cells within the sphere and add them
    auto cells = findGridCellsInSphere(worldPos, worldRadius);
    qCInfo(lcApprovalMask) << "  Found" << cells.size() << "grid cells in sphere";

    for (const auto& cell : cells) {
        const uint64_t hash = (static_cast<uint64_t>(cell.first) << 32) | static_cast<uint64_t>(cell.second);
        if (_accumulatedGridPosSet.insert(hash).second) {
            _accumulatedGridPositions.push_back(cell);
        }
    }

    // Paint immediately
    if (!_accumulatedGridPositions.empty()) {
        paintAccumulatedPointsToImage();
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::extendStrokeFromWorld(const cv::Vec3f& worldPos, float worldRadius, bool forceSample)
{
    if (!_strokeActive) {
        return;
    }

    const float spacing = kBrushSampleSpacing;
    const float spacingSq = spacing * spacing;

    // Check if we've moved enough to sample
    if (_hasLastSample && !forceSample) {
        const cv::Vec3f delta = worldPos - _lastSample;
        const float distanceSq = delta.dot(delta);
        if (distanceSq < spacingSq) {
            return;
        }
    }

    _currentStroke.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;

    // Find cells in sphere and add to accumulated
    auto cells = findGridCellsInSphere(worldPos, worldRadius);
    for (const auto& cell : cells) {
        const uint64_t hash = (static_cast<uint64_t>(cell.first) << 32) | static_cast<uint64_t>(cell.second);
        if (_accumulatedGridPosSet.insert(hash).second) {
            _accumulatedGridPositions.push_back(cell);
        }
    }

    // Paint periodically
    constexpr size_t kPaintBatchSize = 20;
    if (forceSample || _accumulatedGridPositions.size() >= kPaintBatchSize) {
        paintAccumulatedPointsToImage();
    }

    // Update overlay points for visualization
    const float overlaySpacing = kOverlayPointSpacing;
    const float overlaySpacingSq = overlaySpacing * overlaySpacing;

    bool overlayNeedsRefresh = false;
    if (_hasLastOverlaySample) {
        const cv::Vec3f overlayDelta = worldPos - _lastOverlaySample;
        const float overlayDistSq = overlayDelta.dot(overlayDelta);
        if (forceSample || overlayDistSq >= overlaySpacingSq) {
            _overlayPoints.push_back(worldPos);
            _lastOverlaySample = worldPos;
            overlayNeedsRefresh = true;
        }
    } else {
        _overlayPoints.push_back(worldPos);
        _lastOverlaySample = worldPos;
        _hasLastOverlaySample = true;
        overlayNeedsRefresh = true;
    }

    if (overlayNeedsRefresh) {
        const qint64 currentTime = _lastRefreshTimer.elapsed();
        const qint64 timeSinceLastRefresh = currentTime - _lastRefreshTime;
        constexpr qint64 kMinRefreshIntervalMs = 50;

        if (timeSinceLastRefresh >= kMinRefreshIntervalMs) {
            _module.refreshOverlay();
            _lastRefreshTime = currentTime;
            _pendingRefresh = false;
        } else {
            _pendingRefresh = true;
        }
    }
}

void ApprovalMaskBrushTool::finishStrokeFromWorld()
{
    if (!_strokeActive) {
        return;
    }

    // Paint any remaining accumulated points
    if (!_accumulatedGridPositions.empty()) {
        paintAccumulatedPointsToImage();
    }

    _strokeActive = false;
    if (!_currentStroke.empty()) {
        _pendingStrokes.push_back(_currentStroke);
    }
    _currentStroke.clear();

    if (!_overlayPoints.empty()) {
        _overlayStrokeSegments.push_back(_overlayPoints);
        _overlayPoints.clear();
    }

    _hasLastSample = false;
    _hasLastOverlaySample = false;

    if (_pendingRefresh) {
        _pendingRefresh = false;
    }
    _module.refreshOverlay();
}

std::optional<std::pair<int, int>> ApprovalMaskBrushTool::sceneToGridIndex(const QPointF& scenePos, float viewerScale) const
{
    // Convert scene coordinates to grid indices
    // The overlay rendering uses: scenePos = (gridPos/surfScale - center) * viewerScale
    // Inverting: gridPos = (scenePos/viewerScale + center) * surfScale
    if (!_surface) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    // Get surface parameters
    const cv::Vec3f center = _surface->center();
    const cv::Vec2f surfScale = _surface->scale();

    // Compute grid position: (scenePos / viewerScale + center) * surfaceScale
    const float surfLocX = static_cast<float>(scenePos.x()) / viewerScale;
    const float surfLocY = static_cast<float>(scenePos.y()) / viewerScale;
    const float gridX = (surfLocX + center[0]) * surfScale[0];
    const float gridY = (surfLocY + center[1]) * surfScale[1];

    const int col = static_cast<int>(std::round(gridX));
    const int row = static_cast<int>(std::round(gridY));

    // Check bounds
    if (row < 0 || row >= points->rows || col < 0 || col >= points->cols) {
        return std::nullopt;
    }

    // Check if the point at this location is valid
    const cv::Vec3f& point = (*points)(row, col);
    if (isInvalidPoint(point)) {
        // Search nearby for a valid point (small radius since we have precise coordinates)
        constexpr int kSearchRadius = 3;
        for (int dr = -kSearchRadius; dr <= kSearchRadius; ++dr) {
            for (int dc = -kSearchRadius; dc <= kSearchRadius; ++dc) {
                const int r = row + dr;
                const int c = col + dc;
                if (r >= 0 && r < points->rows && c >= 0 && c < points->cols) {
                    if (!isInvalidPoint((*points)(r, c))) {
                        return std::make_pair(r, c);
                    }
                }
            }
        }
        return std::nullopt;
    }

    return std::make_pair(row, col);
}
