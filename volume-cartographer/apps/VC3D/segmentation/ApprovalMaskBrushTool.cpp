#include "ApprovalMaskBrushTool.hpp"

#include "SegmentationEditManager.hpp"
#include "SegmentationModule.hpp"
#include "SegmentationWidget.hpp"
#include "ViewerManager.hpp"
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
    // Clear the search cache when surface changes
    _hasLastSearchCache = false;

    // Build spatial index for fast 3D queries
    _pointIndex.clear();
    _pointIndexCols = 0;
    if (surface) {
        const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
        if (points && !points->empty()) {
            _pointIndex.buildFromMat(*points);
            _pointIndexCols = points->cols;
            qCDebug(lcApprovalMask) << "Built PointIndex with" << _pointIndex.size() << "points";
        }
    }

    qCDebug(lcApprovalMask) << "Surface set on approval tool:" << (surface ? "valid" : "null");
}

void ApprovalMaskBrushTool::setActive(bool active)
{
    if (_brushActive == active) {
        return;
    }

    _brushActive = active;
    qCDebug(lcApprovalMask) << "Approval brush active:" << active;
    if (!_brushActive) {
        _hasLastSample = false;
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::startStroke(const cv::Vec3f& worldPos, const QPointF& scenePos, float viewerScale)
{
    qCDebug(lcApprovalMask) << "Starting approval stroke at:" << worldPos[0] << worldPos[1] << worldPos[2]
                           << "scenePos:" << scenePos.x() << scenePos.y() << "viewerScale:" << viewerScale;
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

    // Disable plane effective radius mode for flattened view
    _usePlaneEffectiveRadius = false;

    // Update hover position for brush circle display
    _hoverWorldPos = worldPos;
    _hoverEffectiveRadius = _module.approvalMaskBrushRadius();

    // Add the starting point for painting - compute grid position from scene coordinates
    auto gridIdx = sceneToGridIndex(scenePos, viewerScale);
    if (gridIdx) {
        qCDebug(lcApprovalMask) << "  Grid index:" << gridIdx->first << gridIdx->second;
        const uint64_t hash = (static_cast<uint64_t>(gridIdx->first) << 32) | static_cast<uint64_t>(gridIdx->second);
        _accumulatedGridPosSet.insert(hash);
        _accumulatedGridPositions.push_back(*gridIdx);

        // Paint immediately for instant feedback on first click
        paintAccumulatedPointsToImage();
    } else {
        qCDebug(lcApprovalMask) << "  Grid index: OUT OF BOUNDS";
    }

    // Note: paintAccumulatedPointsToImage already calls refreshOverlay
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

    // Update hover position for brush circle display during drag
    _hoverWorldPos = worldPos;
    // For flattened view, use full brush radius (no effective radius calculation needed)
    _hoverEffectiveRadius = _module.approvalMaskBrushRadius();

    // Accumulate grid position for real-time painting (reuse gridIdx from above)
    // We know gridIdx is valid here because we would have returned early if it was nullopt
    const uint64_t hash = (static_cast<uint64_t>(gridIdx->first) << 32) | static_cast<uint64_t>(gridIdx->second);
    if (_accumulatedGridPosSet.insert(hash).second) {
        _accumulatedGridPositions.push_back(*gridIdx);
    }

    // Paint with time-based throttling to avoid performance issues
    // Paint every 50ms or when we have accumulated enough points
    constexpr qint64 kPaintIntervalMs = 50;
    constexpr size_t kPaintBatchSize = 10;
    const qint64 elapsed = _lastRefreshTimer.elapsed();
    if (forceSample || _accumulatedGridPositions.size() >= kPaintBatchSize ||
        elapsed - _lastRefreshTime >= kPaintIntervalMs) {
        if (!_accumulatedGridPositions.empty()) {
            paintAccumulatedPointsToImage();
            _lastRefreshTime = elapsed;
        }
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

    qCDebug(lcApprovalMask) << "Saved approval mask to disk in" << totalTimer.elapsed() << "ms";

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
    _hasLastSearchCache = false;

    // Reload approval mask from disk to discard pending changes
    auto overlay = _module.overlay();
    if (overlay && _surface) {
        overlay->loadApprovalMaskImage(_surface);
        qCDebug(lcApprovalMask) << "Reloaded approval mask from disk (discarded pending changes)";
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

    // Cylinder brush model:
    // - Radius = circle in plane views, rectangle width in flattened view (diameter = 2*radius)
    // - Depth = cylinder thickness, rectangle height in flattened view
    const float brushRadiusNative = _module.approvalMaskBrushRadius();
    const float brushDepthNative = _module.approvalBrushDepth();

    // Convert from native voxels to grid units for painting into the QImage
    float avgScale = 1.0f;
    if (_surface) {
        const cv::Vec2f scale = _surface->scale();
        avgScale = (scale[0] + scale[1]) * 0.5f;
    }
    const float gridRadius = brushRadiusNative * avgScale;
    const float gridDepth = brushDepthNative * avgScale;

    float paintRadius;
    float paintWidth = 0.0f;
    float paintHeight = 0.0f;
    bool useRectangle = false;

    if (_usePlaneEffectiveRadius) {
        // For plane viewer strokes: we already found the exact grid cells within the
        // cylinder. Just paint those cells directly with minimal radius.
        paintRadius = 1.0f;
        useRectangle = false;  // Plane views paint individual cells
    } else {
        // For segmentation/flattened view strokes: paint a rectangle (cylinder side view)
        // Width = diameter (2 * radius), Height = depth
        paintWidth = gridRadius * 2.0f;
        paintHeight = gridDepth;
        paintRadius = gridRadius;
        useRectangle = true;
    }
    const float clampedRadius = std::clamp(paintRadius, 0.5f, 500.0f);

    // Paint the accumulated points into the QImage
    overlay->paintApprovalMaskDirect(_accumulatedGridPositions, clampedRadius, paintValue,
                                      useRectangle, paintWidth, paintHeight);

    // Clear for next batch
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    // Trigger overlay refresh to show the updated image
    _module.refreshOverlay();
}

std::vector<std::pair<int, int>> ApprovalMaskBrushTool::findGridCellsInSphere(const cv::Vec3f& worldPos, float radius) const
{
    std::vector<std::pair<int, int>> result;

    if (!_surface || _pointIndex.empty() || _pointIndexCols <= 0) {
        return result;
    }

    // Use PointIndex for O(log n + k) spatial query instead of grid window iteration
    auto queryResults = _pointIndex.queryRadius(worldPos, radius);

    result.reserve(queryResults.size());
    for (const auto& qr : queryResults) {
        // Decode ID back to grid position: id = row * cols + col
        const int row = static_cast<int>(qr.id / _pointIndexCols);
        const int col = static_cast<int>(qr.id % _pointIndexCols);
        result.emplace_back(row, col);
    }

    return result;
}

std::vector<std::pair<int, int>> ApprovalMaskBrushTool::findGridCellsInCylinder(
    const cv::Vec3f& worldPos,
    const cv::Vec3f& planeNormal,
    float radius,
    float depth,
    float* outMinDist) const
{
    std::vector<std::pair<int, int>> result;

    if (outMinDist) {
        *outMinDist = std::numeric_limits<float>::max();
    }

    if (!_surface || _pointIndex.empty() || _pointIndexCols <= 0) {
        return result;
    }

    // Cylinder parameters: axis along planeNormal, radius perpendicular, depth along axis
    const float radiusSq = radius * radius;
    const float halfDepth = depth / 2.0f;

    // Normalize the plane normal for projection calculations
    const float normalLen = std::sqrt(planeNormal.dot(planeNormal));
    const cv::Vec3f normal = (normalLen > 1e-6f) ? planeNormal / normalLen : cv::Vec3f(0, 0, 1);

    // Query bounding sphere that contains the cylinder
    // Bounding radius = sqrt(radius² + halfDepth²)
    const float boundingRadius = std::sqrt(radiusSq + halfDepth * halfDepth);
    auto queryResults = _pointIndex.queryRadius(worldPos, boundingRadius);

    // Filter by cylinder test and track minimum distance
    float minDistSq = std::numeric_limits<float>::max();

    for (const auto& qr : queryResults) {
        // Track minimum distance for outMinDist
        if (qr.distanceSq < minDistSq) {
            minDistSq = qr.distanceSq;
        }

        // Cylinder test: check axial and perpendicular distance
        const cv::Vec3f delta = qr.position - worldPos;

        // Project delta onto the normal (axial distance)
        const float axialDist = std::abs(delta.dot(normal));

        // Perpendicular distance squared = total distance squared - axial distance squared
        const float perpDistSq = qr.distanceSq - axialDist * axialDist;

        // Check if point is within the cylinder
        if (axialDist <= halfDepth && perpDistSq <= radiusSq) {
            // Decode ID back to grid position: id = row * cols + col
            const int row = static_cast<int>(qr.id / _pointIndexCols);
            const int col = static_cast<int>(qr.id % _pointIndexCols);
            result.emplace_back(row, col);
        }
    }

    if (outMinDist && minDistSq < std::numeric_limits<float>::max()) {
        *outMinDist = std::sqrt(minDistSq);
    }

    return result;
}

void ApprovalMaskBrushTool::startStrokeFromPlane(const cv::Vec3f& worldPos, const cv::Vec3f& planeNormal, float worldRadius)
{
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

    // Enable plane effective radius mode
    _usePlaneEffectiveRadius = true;
    _effectivePaintRadiusNative = 0.0f;

    // For flat cylinder model: use full brush radius in plane views
    // The cylinder has full radius in the XY/XZ/YZ planes
    _effectivePaintRadiusNative = worldRadius;

    // Update hover position for brush circle display
    _hoverWorldPos = worldPos;
    _hoverEffectiveRadius = worldRadius;

    // Find cells within the cylinder (radius perpendicular to plane, depth along plane normal)
    const float brushDepth = _module.approvalBrushDepth();
    auto cells = findGridCellsInCylinder(worldPos, planeNormal, worldRadius, brushDepth, nullptr);
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

void ApprovalMaskBrushTool::extendStrokeFromPlane(const cv::Vec3f& worldPos, const cv::Vec3f& planeNormal, float worldRadius, bool forceSample)
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

    // For cylinder model: use full brush radius in plane views
    _effectivePaintRadiusNative = worldRadius;

    // Update hover position for brush circle display during drag
    _hoverWorldPos = worldPos;
    _hoverEffectiveRadius = worldRadius;

    // Find cells within the cylinder (radius perpendicular to plane, depth along plane normal)
    const float brushDepth = _module.approvalBrushDepth();
    auto cells = findGridCellsInCylinder(worldPos, planeNormal, worldRadius, brushDepth, nullptr);
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

void ApprovalMaskBrushTool::finishStrokeFromPlane()
{
    // Reset plane effective radius mode
    _usePlaneEffectiveRadius = false;
    _effectivePaintRadiusNative = 0.0f;

    finishStrokeFromWorld();
}

void ApprovalMaskBrushTool::startStrokeFromWorld(const cv::Vec3f& worldPos, float worldRadius)
{
    qCDebug(lcApprovalMask) << "Starting approval stroke from world pos:" << worldPos[0] << worldPos[1] << worldPos[2]
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
    qCDebug(lcApprovalMask) << "  Found" << cells.size() << "grid cells in sphere";

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

void ApprovalMaskBrushTool::setHoverWorldPos(const cv::Vec3f& pos, float brushRadius, const QPointF& scenePos, float viewerScale,
                                             const std::optional<cv::Vec3f>& planeNormal)
{
    _hoverWorldPos = pos;
    _hoverScenePos = scenePos;
    _hoverViewerScale = viewerScale;
    _hoverPlaneNormal = planeNormal;

    // For flat cylinder model: always use full brush radius
    // The cylinder has full radius in all orthogonal plane views (XY, XZ, YZ)
    _hoverEffectiveRadius = brushRadius;
}
