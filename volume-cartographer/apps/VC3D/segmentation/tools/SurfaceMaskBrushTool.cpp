#include "SurfaceMaskBrushTool.hpp"

#include "../SegmentationModule.hpp"
#include "../../ViewerManager.hpp"

#include <QCoreApplication>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Tiff.hpp"

Q_LOGGING_CATEGORY(lcSurfaceMaskBrush, "vc.segmentation.surfacemask")

namespace
{
uint64_t cellKey(int row, int col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint32_t>(col);
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
    _lastGrid.reset();
    _paintedCells.clear();
    _pendingCells.clear();
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

    const auto grid = surfaceToGridIndex(surfacePos);
    if (!grid) {
        return;
    }

    ensureMask();
    _strokeActive = true;
    _paintedCells.clear();
    _pendingCells.clear();
    _overlaySurfacePoints.clear();
    _lastGrid = grid;
    paintAt(grid->first, grid->second);
    invalidateViewers();
}

void SurfaceMaskBrushTool::extendStroke(const QPointF& surfacePos, bool forceSample)
{
    if (!_strokeActive || !_surface) {
        return;
    }

    const auto grid = surfaceToGridIndex(surfacePos);
    if (!grid) {
        _lastGrid.reset();
        return;
    }

    const float avgScale = [&]() {
        const cv::Vec2f scale = _surface->scale();
        const float avg = 0.5f * (std::abs(scale[0]) + std::abs(scale[1]));
        return avg > 1e-4f ? avg : 1.0f;
    }();
    const float radius = std::max(1.0f, _module.approvalMaskBrushRadius() * avgScale);
    const float sampleSpacing = std::max(1.0f, radius / 3.0f);

    if (_lastGrid) {
        const int dr = grid->first - _lastGrid->first;
        const int dc = grid->second - _lastGrid->second;
        const float distance = std::sqrt(static_cast<float>(dr * dr + dc * dc));
        if (!forceSample && distance < sampleSpacing) {
            return;
        }

        const int steps = std::max(1, static_cast<int>(std::ceil(distance / sampleSpacing)));
        for (int step = 1; step <= steps; ++step) {
            const float t = static_cast<float>(step) / static_cast<float>(steps);
            const int row = static_cast<int>(std::lround(_lastGrid->first + dr * t));
            const int col = static_cast<int>(std::lround(_lastGrid->second + dc * t));
            paintAt(row, col);
        }
    } else {
        paintAt(grid->first, grid->second);
    }

    _lastGrid = grid;
    invalidateViewers();
}

void SurfaceMaskBrushTool::pauseStroke()
{
    if (!_strokeActive) {
        return;
    }

    _strokeActive = false;
    _lastGrid.reset();
    invalidateViewers();
}

void SurfaceMaskBrushTool::finishStroke()
{
    if (!_strokeActive && _pendingCells.empty()) {
        return;
    }

    _strokeActive = false;
    _lastGrid.reset();
    applyPendingCells();
    persistMask();
    _paintedCells.clear();
    _pendingCells.clear();
    _overlaySurfacePoints.clear();
    invalidateViewers();
}

void SurfaceMaskBrushTool::cancelStroke()
{
    _strokeActive = false;
    _lastGrid.reset();
    _paintedCells.clear();
    _pendingCells.clear();
    _overlaySurfacePoints.clear();
    if (_surface) {
        _mask = _surface->validMask();
    } else {
        _mask.release();
    }
    invalidateViewers();
}

std::optional<std::pair<int, int>> SurfaceMaskBrushTool::surfaceToGridIndex(const QPointF& surfacePos) const
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

    const int col = static_cast<int>(std::lround((static_cast<float>(surfacePos.x()) + center[0]) * scale[0]));
    const int row = static_cast<int>(std::lround((static_cast<float>(surfacePos.y()) + center[1]) * scale[1]));
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
    const int radiusSq = radius * radius;
    const cv::Vec3f center = _surface->center();
    _overlaySurfacePoints.emplace_back(static_cast<float>(centerCol) / scale[0] - center[0],
                                       static_cast<float>(centerRow) / scale[1] - center[1]);

    for (int dr = -radius; dr <= radius; ++dr) {
        for (int dc = -radius; dc <= radius; ++dc) {
            if (dr * dr + dc * dc > radiusSq) {
                continue;
            }

            const int row = centerRow + dr;
            const int col = centerCol + dc;
            if (row < 0 || row >= _mask.rows || col < 0 || col >= _mask.cols) {
                continue;
            }

            const uint64_t key = cellKey(row, col);
            if (!_paintedCells.insert(key).second) {
                continue;
            }

            _pendingCells.emplace_back(row, col);
        }
    }
}

void SurfaceMaskBrushTool::persistMask()
{
    if (!_surface || _mask.empty()) {
        return;
    }

    _surface->invalidateCache();

    if (_surface->path.empty()) {
        Q_EMIT _module.statusMessageRequested(
            QCoreApplication::translate("SurfaceMaskBrushTool", "Cannot save mask: surface has no tifxyz path."),
            kStatusMedium);
        return;
    }

    try {
        const std::filesystem::path maskPath = _surface->path / "mask.tif";
        writeTiff(maskPath, _mask, -1, 1024, 1024, -1.0f, COMPRESSION_LZW, _surface->dpi());
        _surface->refreshMaskTimestamp();
        Q_EMIT _module.statusMessageRequested(
            QCoreApplication::translate("SurfaceMaskBrushTool", "Saved mask.tif."),
            kStatusShort);
    } catch (const std::exception& e) {
        qCWarning(lcSurfaceMaskBrush) << "Failed to save mask.tif:" << e.what();
        Q_EMIT _module.statusMessageRequested(
            QCoreApplication::translate("SurfaceMaskBrushTool", "Failed to save mask.tif."),
            kStatusLong);
    }
}

void SurfaceMaskBrushTool::applyPendingCells()
{
    if (!_surface || _mask.empty()) {
        return;
    }

    auto* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const cv::Vec3f invalid(-1.0f, -1.0f, -1.0f);
    for (const auto& [row, col] : _pendingCells) {
        if (row < 0 || row >= _mask.rows || col < 0 || col >= _mask.cols) {
            continue;
        }
        _mask(row, col) = 0;
        (*points)(row, col) = invalid;
    }
}

void SurfaceMaskBrushTool::invalidateViewers()
{
    if (auto* manager = _module.viewerManager()) {
        manager->forEachViewer([](CTiledVolumeViewer* viewer) {
            if (viewer) {
                viewer->invalidateIntersect();
                viewer->renderIntersections();
                viewer->requestRender();
            }
        });
    }
    _module.refreshOverlay();
}
