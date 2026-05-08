#include "SurfaceMaskBrushTool.hpp"

#include "../SegmentationModule.hpp"
#include "SegmentationEditManager.hpp"
#include "../../ViewerManager.hpp"

#include <QCoreApplication>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <exception>

#include "vc/core/util/QuadSurface.hpp"

Q_LOGGING_CATEGORY(lcSurfaceMaskBrush, "vc.segmentation.surfacemask")

namespace
{
uint64_t cellKey(int row, int col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint32_t>(col);
}

struct GridPointF {
    float row;
    float col;
};

GridPointF toPoint(const std::pair<int, int>& p)
{
    return {static_cast<float>(p.first), static_cast<float>(p.second)};
}

float cross(const GridPointF& a, const GridPointF& b, const GridPointF& c)
{
    return (b.col - a.col) * (c.row - a.row) -
           (b.row - a.row) * (c.col - a.col);
}

bool pointInRect(const GridPointF& p, float rowMin, float rowMax, float colMin, float colMax)
{
    return p.row >= rowMin && p.row <= rowMax &&
           p.col >= colMin && p.col <= colMax;
}

bool pointInPolygon(const GridPointF& point, const std::vector<std::pair<int, int>>& polygon)
{
    bool inside = false;
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const GridPointF a = toPoint(polygon[i]);
        const GridPointF b = toPoint(polygon[j]);
        const bool crosses = ((a.row > point.row) != (b.row > point.row)) &&
                             (point.col < (b.col - a.col) * (point.row - a.row) /
                                              (b.row - a.row) + a.col);
        if (crosses) {
            inside = !inside;
        }
    }
    return inside;
}

bool segmentsIntersect(GridPointF a, GridPointF b, GridPointF c, GridPointF d)
{
    constexpr float eps = 1.0e-5f;
    const float c1 = cross(a, b, c);
    const float c2 = cross(a, b, d);
    const float c3 = cross(c, d, a);
    const float c4 = cross(c, d, b);

    auto onSegment = [](const GridPointF& p, const GridPointF& q, const GridPointF& r) {
        constexpr float eps = 1.0e-5f;
        return std::abs(cross(p, q, r)) <= eps &&
               q.row >= std::min(p.row, r.row) - eps &&
               q.row <= std::max(p.row, r.row) + eps &&
               q.col >= std::min(p.col, r.col) - eps &&
               q.col <= std::max(p.col, r.col) + eps;
    };

    if (((c1 > eps && c2 < -eps) || (c1 < -eps && c2 > eps)) &&
        ((c3 > eps && c4 < -eps) || (c3 < -eps && c4 > eps))) {
        return true;
    }

    return onSegment(a, c, b) ||
           onSegment(a, d, b) ||
           onSegment(c, a, d) ||
           onSegment(c, b, d);
}

bool polygonIntersectsCell(const std::vector<std::pair<int, int>>& polygon, int row, int col)
{
    const float rowMin = static_cast<float>(row);
    const float rowMax = static_cast<float>(row + 1);
    const float colMin = static_cast<float>(col);
    const float colMax = static_cast<float>(col + 1);

    const GridPointF center{rowMin + 0.5f, colMin + 0.5f};
    if (pointInPolygon(center, polygon)) {
        return true;
    }

    const GridPointF corners[] = {
        {rowMin, colMin},
        {rowMin, colMax},
        {rowMax, colMax},
        {rowMax, colMin}
    };
    for (const GridPointF& corner : corners) {
        if (pointInPolygon(corner, polygon)) {
            return true;
        }
    }

    for (const auto& p : polygon) {
        if (pointInRect(toPoint(p), rowMin, rowMax, colMin, colMax)) {
            return true;
        }
    }

    const std::pair<GridPointF, GridPointF> edges[] = {
        {corners[0], corners[1]},
        {corners[1], corners[2]},
        {corners[2], corners[3]},
        {corners[3], corners[0]}
    };
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const GridPointF a = toPoint(polygon[j]);
        const GridPointF b = toPoint(polygon[i]);
        for (const auto& edge : edges) {
            if (segmentsIntersect(a, b, edge.first, edge.second)) {
                return true;
            }
        }
    }

    return false;
}
}

SurfaceMaskBrushTool::SurfaceMaskBrushTool(SegmentationModule& module)
    : _module(module)
{
}

void SurfaceMaskBrushTool::setSurface(QuadSurface* surface)
{
    if (_surface == surface) {
        return;
    }
    _surface = surface;
    _mask.release();
    _lastGridPosition.reset();
    _undoSnapshotCaptured = false;
    _paintedCells.clear();
    _pendingCells.clear();
    _strokeGridPoints.clear();
    _overlaySurfacePoints.clear();
}

void SurfaceMaskBrushTool::setActive(bool active)
{
    if (_active == active) {
        return;
    }
    _active = active;
    if (!_active && (_strokeActive || hasPendingStroke())) {
        finishStroke();
    }
}

void SurfaceMaskBrushTool::startStroke(const QPointF& surfacePos)
{
    if (!_active || !_surface) {
        return;
    }

    const auto gridPos = surfaceToGridPosition(surfacePos);
    if (!gridPos) {
        return;
    }
    const int row = static_cast<int>(std::lround(gridPos->first));
    const int col = static_cast<int>(std::lround(gridPos->second));

    ensureMask();
    _undoSnapshotCaptured = _module.captureUndoSnapshot();
    _strokeActive = true;
    _paintedCells.clear();
    _pendingCells.clear();
    _strokeGridPoints.clear();
    _overlaySurfacePoints.clear();
    _lastGridPosition = gridPos;
    appendOverlayPoint(surfacePos);
    paintAt(row, col);
    invalidateOverlay();
}

void SurfaceMaskBrushTool::extendStroke(const QPointF& surfacePos, bool forceSample)
{
    if (!_strokeActive || !_surface) {
        return;
    }

    const auto gridPos = surfaceToGridPosition(surfacePos);
    if (!gridPos) {
        _lastGridPosition.reset();
        return;
    }
    appendOverlayPoint(surfacePos);

    const float avgScale = [&]() {
        const cv::Vec2f scale = _surface->scale();
        const float avg = 0.5f * (std::abs(scale[0]) + std::abs(scale[1]));
        return avg > 1e-4f ? avg : 1.0f;
    }();
    const float radius = std::max(1.0f, _module.approvalMaskBrushRadius() * avgScale);
    const float sampleSpacing = std::max(1.0f, radius / 3.0f);

    if (_lastGridPosition) {
        const float dr = gridPos->first - _lastGridPosition->first;
        const float dc = gridPos->second - _lastGridPosition->second;
        const float distance = std::sqrt(dr * dr + dc * dc);
        if (!forceSample && distance < sampleSpacing) {
            invalidateOverlay();
            return;
        }

        const int steps = std::max(1, static_cast<int>(std::ceil(distance / sampleSpacing)));
        for (int step = 1; step <= steps; ++step) {
            const float t = static_cast<float>(step) / static_cast<float>(steps);
            const int row = static_cast<int>(std::lround(_lastGridPosition->first + dr * t));
            const int col = static_cast<int>(std::lround(_lastGridPosition->second + dc * t));
            paintAt(row, col);
        }
    } else {
        const int row = static_cast<int>(std::lround(gridPos->first));
        const int col = static_cast<int>(std::lround(gridPos->second));
        paintAt(row, col);
    }

    _lastGridPosition = gridPos;
    invalidateOverlay();
}

void SurfaceMaskBrushTool::pauseStroke()
{
    if (!_strokeActive) {
        return;
    }

    _strokeActive = false;
    _lastGridPosition.reset();
    invalidateOverlay();
}

void SurfaceMaskBrushTool::finishStroke()
{
    if (!_strokeActive && _pendingCells.empty()) {
        return;
    }

    _strokeActive = false;
    _lastGridPosition.reset();
    fillEnclosedStrokeArea();
    const cv::Rect changedRegion = applyPendingCells();
    if (changedRegion.empty() && _undoSnapshotCaptured) {
        _module.discardLastUndoSnapshot();
    }
    _undoSnapshotCaptured = false;
    if (_module.hasActiveSession() && _module.activeBaseSurface() == _surface && !changedRegion.empty()) {
        _module._editManager->applyExternalSurfaceUpdate(changedRegion);
    }
    refreshSurfacePatchIndex(changedRegion);
    persistSurface();
    _paintedCells.clear();
    _pendingCells.clear();
    _strokeGridPoints.clear();
    _overlaySurfacePoints.clear();
    invalidateViewers(!changedRegion.empty());
}

void SurfaceMaskBrushTool::cancelStroke()
{
    _strokeActive = false;
    _lastGridPosition.reset();
    if (_undoSnapshotCaptured) {
        _module.discardLastUndoSnapshot();
        _undoSnapshotCaptured = false;
    }
    _paintedCells.clear();
    _pendingCells.clear();
    _strokeGridPoints.clear();
    _overlaySurfacePoints.clear();
    if (_surface) {
        _mask = _surface->validMask();
    } else {
        _mask.release();
    }
    invalidateViewers(false);
}

void SurfaceMaskBrushTool::refreshFromSurface()
{
    if (_strokeActive || hasPendingStroke()) {
        cancelStroke();
    }
    _lastGridPosition.reset();
    _paintedCells.clear();
    _pendingCells.clear();
    _strokeGridPoints.clear();
    _overlaySurfacePoints.clear();
    if (_surface) {
        _surface->invalidateCache();
        _mask = _surface->validMask();
    } else {
        _mask.release();
    }
    invalidateViewers(true);
}

std::optional<std::pair<float, float>> SurfaceMaskBrushTool::surfaceToGridPosition(const QPointF& surfacePos) const
{
    if (!_surface) {
        return std::nullopt;
    }

    const auto* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    const cv::Vec3f center = _surface->center();
    const cv::Vec2f scale = _surface->scale();
    if (std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) {
        return std::nullopt;
    }

    const float col = (static_cast<float>(surfacePos.x()) + center[0]) * scale[0];
    const float row = (static_cast<float>(surfacePos.y()) + center[1]) * scale[1];
    if (row < 0 || row >= points->rows || col < 0 || col >= points->cols) {
        return std::nullopt;
    }

    return std::make_pair(row, col);
}

void SurfaceMaskBrushTool::ensureMask()
{
    if (!_surface) {
        _mask.release();
        return;
    }

    const auto* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        _mask.release();
        return;
    }

    if (_mask.rows == points->rows && _mask.cols == points->cols) {
        return;
    }

    _mask = _surface->validMask();
}

void SurfaceMaskBrushTool::paintAt(int centerRow, int centerCol)
{
    if (!_surface || _mask.empty()) {
        return;
    }

    auto* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const cv::Vec2f scale = _surface->scale();
    if (std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) {
        return;
    }
    const float avgScale = 0.5f * (std::abs(scale[0]) + std::abs(scale[1]));
    const int radius = static_cast<int>(std::ceil(std::max(1.0f, _module.approvalMaskBrushRadius() *
                                                                 (avgScale > 1e-4f ? avgScale : 1.0f))));
    const float radiusSq = static_cast<float>(radius * radius);
    if (_strokeGridPoints.empty() ||
        _strokeGridPoints.back().first != centerRow ||
        _strokeGridPoints.back().second != centerCol) {
        _strokeGridPoints.emplace_back(centerRow, centerCol);
    }

    if (_mask.rows < 2 || _mask.cols < 2) {
        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                if (static_cast<float>(dr * dr + dc * dc) > radiusSq) {
                    continue;
                }
                queueVertex(centerRow + dr, centerCol + dc);
            }
        }
        return;
    }

    const int rowStart = std::max(0, centerRow - radius - 1);
    const int rowEnd = std::min(_mask.rows - 2, centerRow + radius);
    const int colStart = std::max(0, centerCol - radius - 1);
    const int colEnd = std::min(_mask.cols - 2, centerCol + radius);

    for (int cellRow = rowStart; cellRow <= rowEnd; ++cellRow) {
        const float nearestRow = std::clamp(static_cast<float>(centerRow),
                                            static_cast<float>(cellRow),
                                            static_cast<float>(cellRow + 1));
        const float dr = nearestRow - static_cast<float>(centerRow);
        for (int cellCol = colStart; cellCol <= colEnd; ++cellCol) {
            const float nearestCol = std::clamp(static_cast<float>(centerCol),
                                                static_cast<float>(cellCol),
                                                static_cast<float>(cellCol + 1));
            const float dc = nearestCol - static_cast<float>(centerCol);
            if (dr * dr + dc * dc > radiusSq + 1.0e-4f) {
                continue;
            }

            queueVertex(cellRow, cellCol);
            queueVertex(cellRow + 1, cellCol);
            queueVertex(cellRow, cellCol + 1);
            queueVertex(cellRow + 1, cellCol + 1);
        }
    }
}

void SurfaceMaskBrushTool::appendOverlayPoint(const QPointF& surfacePos)
{
    if (!_overlaySurfacePoints.empty()) {
        const QPointF delta = surfacePos - _overlaySurfacePoints.back();
        if (delta.x() * delta.x() + delta.y() * delta.y() < 0.0625) {
            _overlaySurfacePoints.back() = surfacePos;
            return;
        }
    }

    _overlaySurfacePoints.push_back(surfacePos);
}

void SurfaceMaskBrushTool::queueVertex(int row, int col)
{
    if (row < 0 || row >= _mask.rows || col < 0 || col >= _mask.cols) {
        return;
    }

    const uint64_t key = cellKey(row, col);
    if (!_paintedCells.insert(key).second) {
        return;
    }

    _pendingCells.emplace_back(row, col);
}

void SurfaceMaskBrushTool::fillEnclosedStrokeArea()
{
    if (_strokeGridPoints.size() < 3 || _mask.rows < 2 || _mask.cols < 2) {
        return;
    }

    std::vector<std::pair<int, int>> polygon;
    polygon.reserve(_strokeGridPoints.size());
    for (const auto& p : _strokeGridPoints) {
        if (!polygon.empty() && polygon.back() == p) {
            continue;
        }
        polygon.push_back(p);
    }
    if (polygon.size() < 3) {
        return;
    }

    int minRow = polygon.front().first;
    int maxRow = polygon.front().first;
    int minCol = polygon.front().second;
    int maxCol = polygon.front().second;
    for (const auto& p : polygon) {
        minRow = std::min(minRow, p.first);
        maxRow = std::max(maxRow, p.first);
        minCol = std::min(minCol, p.second);
        maxCol = std::max(maxCol, p.second);
    }

    const int rowStart = std::max(0, minRow - 1);
    const int rowEnd = std::min(_mask.rows - 2, maxRow);
    const int colStart = std::max(0, minCol - 1);
    const int colEnd = std::min(_mask.cols - 2, maxCol);

    for (int cellRow = rowStart; cellRow <= rowEnd; ++cellRow) {
        for (int cellCol = colStart; cellCol <= colEnd; ++cellCol) {
            if (!polygonIntersectsCell(polygon, cellRow, cellCol)) {
                continue;
            }

            queueVertex(cellRow, cellCol);
            queueVertex(cellRow + 1, cellCol);
            queueVertex(cellRow, cellCol + 1);
            queueVertex(cellRow + 1, cellCol + 1);
        }
    }
}

void SurfaceMaskBrushTool::persistSurface()
{
    if (!_surface || _mask.empty()) {
        return;
    }

    _surface->invalidateCache();

    if (_surface->path.empty()) {
        Q_EMIT _module.statusMessageRequested(
            QCoreApplication::translate("SurfaceMaskBrushTool", "Cannot save surface: surface has no tifxyz path."),
            kStatusMedium);
        return;
    }

    if (_module.hasActiveSession()) {
        _module.emitPendingChanges();
        _module.markAutosaveNeeded(true);
        return;
    }

    try {
        _surface->saveOverwrite();
        Q_EMIT _module.statusMessageRequested(
            QCoreApplication::translate("SurfaceMaskBrushTool", "Saved surface."),
            kStatusShort);
    } catch (const std::exception& e) {
        qCWarning(lcSurfaceMaskBrush) << "Failed to save surface:" << e.what();
        Q_EMIT _module.statusMessageRequested(
            QCoreApplication::translate("SurfaceMaskBrushTool", "Failed to save surface."),
            kStatusLong);
    }
}

cv::Rect SurfaceMaskBrushTool::applyPendingCells()
{
    if (!_surface || _mask.empty()) {
        return {};
    }

    auto* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        return {};
    }

    int minRow = points->rows;
    int maxRow = -1;
    int minCol = points->cols;
    int maxCol = -1;

    const cv::Vec3f invalid(-1.0f, -1.0f, -1.0f);
    for (const auto& [row, col] : _pendingCells) {
        if (row < 0 || row >= _mask.rows || col < 0 || col >= _mask.cols) {
            continue;
        }
        const cv::Vec3f previousWorld = (*points)(row, col);
        if (previousWorld == invalid) {
            continue;
        }
        _mask(row, col) = 0;
        (*points)(row, col) = invalid;
        minRow = std::min(minRow, row);
        maxRow = std::max(maxRow, row);
        minCol = std::min(minCol, col);
        maxCol = std::max(maxCol, col);
    }

    if (maxRow < minRow || maxCol < minCol || points->rows < 2 || points->cols < 2) {
        return {};
    }

    const int cellRowCount = points->rows - 1;
    const int cellColCount = points->cols - 1;
    const int rowStart = std::max(0, minRow - 1);
    const int rowEnd = std::min(cellRowCount, maxRow + 1);
    const int colStart = std::max(0, minCol - 1);
    const int colEnd = std::min(cellColCount, maxCol + 1);
    if (rowStart >= rowEnd || colStart >= colEnd) {
        return {};
    }

    return cv::Rect(colStart, rowStart, colEnd - colStart, rowEnd - rowStart);
}

void SurfaceMaskBrushTool::refreshSurfacePatchIndex(const cv::Rect& changedRegion)
{
    if (!_surface || changedRegion.empty()) {
        return;
    }

    auto* manager = _module.viewerManager();
    if (!manager) {
        return;
    }

    SurfacePatchIndex::SurfacePtr surface(_surface, [](QuadSurface*) {});
    manager->refreshSurfacePatchIndex(surface, changedRegion);
}

void SurfaceMaskBrushTool::invalidateOverlay()
{
    _module.refreshOverlay();
}

void SurfaceMaskBrushTool::invalidateViewers(bool surfaceChanged)
{
    if (auto* manager = _module.viewerManager()) {
        manager->forEachBaseViewer([surfaceChanged](VolumeViewerBase* viewer) {
            if (viewer) {
                if (surfaceChanged) {
                    viewer->invalidateIntersect();
                    viewer->renderIntersections();
                }
                if (surfaceChanged && viewer->surfName() == "segmentation") {
                    viewer->renderVisible(true);
                }
                viewer->requestRender();
            }
        });
    }
    _module.refreshOverlay();
}
