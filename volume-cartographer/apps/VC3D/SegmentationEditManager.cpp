#include "SegmentationEditManager.hpp"

#include "vc/core/util/Surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
bool isInvalidPoint(const cv::Vec3f& p)
{
    return !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
           (p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f);
}

void ensurePointValid(cv::Mat_<cv::Vec3f>* mat, int row, int col, const cv::Vec3f& worldPos)
{
    if (!mat || row < 0 || col < 0 || row >= mat->rows || col >= mat->cols) {
        return;
    }
    cv::Vec3f& cell = (*mat)(row, col);
    if (isInvalidPoint(cell)) {
        cell = worldPos;
    }
}
}

SegmentationEditManager::SegmentationEditManager(QObject* parent)
    : QObject(parent)
{
}

bool SegmentationEditManager::beginSession(QuadSurface* baseSurface, int downsample)
{
    if (!baseSurface) {
        return false;
    }

    _baseSurface = baseSurface;
    _downsample = std::max(1, downsample);

    _originalPoints = std::make_unique<cv::Mat_<cv::Vec3f>>(baseSurface->rawPoints().clone());

    auto* previewMatrix = new cv::Mat_<cv::Vec3f>(_originalPoints->clone());
    _previewSurface = std::make_unique<QuadSurface>(previewMatrix, baseSurface->scale());
    _previewSurface->meta = baseSurface->meta;
    _previewSurface->path = baseSurface->path;
    _previewSurface->id = baseSurface->id;
    _previewPoints = _previewSurface->rawPointsPtr();

    _handles.clear();
    regenerateHandles();
    _dirty = false;
    return true;
}

void SegmentationEditManager::endSession()
{
    _handles.clear();
    _previewSurface.reset();
    _previewPoints = nullptr;
    _originalPoints.reset();
    _baseSurface = nullptr;
    _dirty = false;
}

void SegmentationEditManager::setDownsample(int value)
{
    _downsample = std::max(1, value);
    if (hasSession()) {
        regenerateHandles();
    }
}

void SegmentationEditManager::setRadius(float radius)
{
    _radius = std::max(1.0f, radius);
    reapplyAllHandles();
}

void SegmentationEditManager::setSigma(float sigma)
{
    _sigma = std::max(0.1f, sigma);
    reapplyAllHandles();
}

void SegmentationEditManager::resetPreview()
{
    if (!hasSession() || !_previewPoints || !_originalPoints) {
        return;
    }

    _originalPoints->copyTo(*_previewPoints);
    syncPreviewFromBase();
    regenerateHandles();
    _dirty = false;
}

void SegmentationEditManager::applyPreview()
{
    if (!hasSession() || !_previewPoints || !_baseSurface) {
        return;
    }

    auto* basePoints = _baseSurface->rawPointsPtr();
    if (!basePoints || basePoints->empty()) {
        return;
    }

    _previewPoints->copyTo(*basePoints);
    _baseSurface->invalidateCache();

    // Update original snapshot to new base state
    *(_originalPoints) = basePoints->clone();
    basePoints->copyTo(*_previewPoints);
    regenerateHandles();
    _dirty = false;
}

constexpr float kMinInfluenceWeight = 1e-3f;

void SegmentationEditManager::updateHandleWorldPosition(int row, int col, const cv::Vec3f& newWorldPos)
{
    if (!hasSession() || !_previewPoints) {
        return;
    }

    Handle* handle = findHandle(row, col);
    if (!handle) {
        return;
    }

    handle->currentWorld = newWorldPos;
    if (_previewPoints && _originalPoints) {
        _originalPoints->copyTo(*_previewPoints);
        for (const auto& h : _handles) {
            applyHandleInfluence(h);
        }
        // Ensure stored current positions match preview values
        for (auto& h : _handles) {
            if (h.row >= 0 && h.row < _previewPoints->rows && h.col >= 0 && h.col < _previewPoints->cols) {
                h.currentWorld = (*_previewPoints)(h.row, h.col);
            }
        }
        _dirty = true;
    }
}

SegmentationEditManager::Handle* SegmentationEditManager::findHandle(int row, int col)
{
    for (auto& handle : _handles) {
        if (handle.row == row && handle.col == col) {
            return &handle;
        }
    }
    return nullptr;
}

SegmentationEditManager::Handle* SegmentationEditManager::findNearestHandle(const cv::Vec3f& world, float tolerance)
{
    if (!hasSession()) {
        return nullptr;
    }

    const bool unlimited = tolerance < 0.0f || !std::isfinite(tolerance);
    const float clamped = unlimited ? 0.0f : std::max(1.0f, tolerance);
    const float limitSq = unlimited ? std::numeric_limits<float>::max() : clamped * clamped;
    float bestSq = std::numeric_limits<float>::max();
    Handle* bestHandle = nullptr;

    for (auto& handle : _handles) {
        const float distSq = static_cast<float>(cv::norm(handle.currentWorld - world));
        if (!unlimited && distSq > limitSq) {
            continue;
        }
        if (distSq < bestSq) {
            bestSq = distSq;
            bestHandle = &handle;
        }
    }

    return bestHandle;
}

std::optional<std::pair<int,int>> SegmentationEditManager::addHandleAtWorld(const cv::Vec3f& worldPos,
                                                                           float tolerance,
                                                                           PlaneSurface* plane,
                                                                           float planeTolerance)
{
    if (!hasSession() || !_previewPoints || !_originalPoints) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>& preview = *_previewPoints;
    const int rows = preview.rows;
    const int cols = preview.cols;
    const float maxDist = tolerance <= 0.0f ? std::numeric_limits<float>::max() : tolerance;
    const float maxDistSq = maxDist * maxDist;
    const float planeMax = plane ? (planeTolerance > 0.0f ? planeTolerance : maxDist) : 0.0f;

    float primarySq = maxDistSq;
    float primaryPlane = std::numeric_limits<float>::max();
    int primaryRow = -1;
    int primaryCol = -1;

    float bestPlaneDist = std::numeric_limits<float>::max();
    int bestPlaneRow = -1;
    int bestPlaneCol = -1;

    float fallbackSq = std::numeric_limits<float>::max();
    int fallbackRow = -1;
    int fallbackCol = -1;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& wp = preview(r, c);
            if (wp[0] == -1.0f && wp[1] == -1.0f && wp[2] == -1.0f) {
                continue;
            }
            const float distSq = static_cast<float>(cv::norm(wp - worldPos));
            const float planeDist = plane ? std::fabs(plane->pointDist(wp)) : 0.0f;

            if (distSq < fallbackSq) {
                fallbackSq = distSq;
                fallbackRow = r;
                fallbackCol = c;
            }

            const bool withinWorld = distSq <= maxDistSq;
            const bool withinPlane = !plane || planeDist <= planeMax;
            if (withinWorld && withinPlane) {
                if (distSq < primarySq || (std::fabs(distSq - primarySq) < 1e-6f && planeDist < primaryPlane)) {
                    primarySq = distSq;
                    primaryPlane = planeDist;
                    primaryRow = r;
                    primaryCol = c;
                }
            }

            if (plane && planeDist < bestPlaneDist) {
                bestPlaneDist = planeDist;
                bestPlaneRow = r;
                bestPlaneCol = c;
            }
        }
    }

    int bestRow = primaryRow;
    int bestCol = primaryCol;

    if (bestRow < 0 || bestCol < 0) {
        if (plane && bestPlaneRow >= 0 && bestPlaneCol >= 0) {
            bestRow = bestPlaneRow;
            bestCol = bestPlaneCol;
        } else {
            bestRow = fallbackRow;
            bestCol = fallbackCol;
        }
    }

    if (bestRow < 0 || bestCol < 0) {
        if (auto gridIdx = worldToGridIndex(worldPos)) {
            bestRow = gridIdx->first;
            bestCol = gridIdx->second;
        }
    }

    if (bestRow < 0 || bestCol < 0) {
        return std::nullopt;
    }

    ensurePointValid(_originalPoints.get(), bestRow, bestCol, worldPos);
    ensurePointValid(_previewPoints, bestRow, bestCol, worldPos);

    if (auto* existing = findHandle(bestRow, bestCol)) {
        if (!existing->isManual) {
            existing->isManual = true;
            existing->originalWorld = (*_originalPoints)(bestRow, bestCol);
            existing->currentWorld = (*_previewPoints)(bestRow, bestCol);
            _dirty = true;
            return std::make_pair(bestRow, bestCol);
        }
        return std::nullopt;
    }

    Handle handle;
    handle.row = bestRow;
    handle.col = bestCol;
    handle.originalWorld = (*_originalPoints)(bestRow, bestCol);
    handle.currentWorld = (*_previewPoints)(bestRow, bestCol);
    handle.isManual = true;
    _handles.push_back(handle);

    _dirty = true;

    return std::make_pair(bestRow, bestCol);
}

std::optional<std::pair<int,int>> SegmentationEditManager::worldToGridIndex(const cv::Vec3f& worldPos, float* outDistance) const
{
    if (!_baseSurface) {
        return std::nullopt;
    }

    cv::Vec3f ptr = _baseSurface->pointer();
    const float dist = _baseSurface->pointTo(ptr, worldPos, std::numeric_limits<float>::max(), 400);
    cv::Vec3f raw = _baseSurface->loc_raw(ptr);
    auto* points = _baseSurface->rawPointsPtr();
    if (!points) {
        return std::nullopt;
    }

    int col = static_cast<int>(std::round(raw[0]));
    int row = static_cast<int>(std::round(raw[1]));

    if (row < 0 || col < 0 || row >= points->rows || col >= points->cols) {
        row = std::clamp(row, 0, points->rows - 1);
        col = std::clamp(col, 0, points->cols - 1);
    }

    if (outDistance) {
        *outDistance = dist;
    }
    return std::make_pair(row, col);
}

bool SegmentationEditManager::removeHandle(int row, int col)
{
    auto it = std::remove_if(_handles.begin(), _handles.end(), [&](const Handle& h) {
        return h.row == row && h.col == col && h.isManual;
    });
    if (it == _handles.end()) {
        return false;
    }
    _handles.erase(it, _handles.end());
    _dirty = true;
    return true;
}

std::optional<cv::Vec3f> SegmentationEditManager::handleWorldPosition(int row, int col) const
{
    if (!hasSession() || !_previewPoints) {
        return std::nullopt;
    }
    for (const auto& handle : _handles) {
        if (handle.row == row && handle.col == col) {
            return handle.currentWorld;
        }
    }
    const cv::Mat_<cv::Vec3f>& preview = *_previewPoints;
    if (row >= 0 && row < preview.rows && col >= 0 && col < preview.cols) {
        const cv::Vec3f& wp = preview(row, col);
        if (!(wp[0] == -1.0f && wp[1] == -1.0f && wp[2] == -1.0f)) {
            return wp;
        }
    }
    return std::nullopt;
}

void SegmentationEditManager::regenerateHandles()
{
    std::vector<Handle> manualHandles;
    manualHandles.reserve(_handles.size());
    for (const auto& handle : _handles) {
        if (handle.isManual) {
            manualHandles.push_back(handle);
        }
    }

    _handles.clear();
    if (!hasSession() || !_previewPoints) {
        _handles = manualHandles;
        return;
    }

    const cv::Mat_<cv::Vec3f>& points = *_previewPoints;
    const int rows = points.rows;
    const int cols = points.cols;

    for (int r = 0; r < rows; r += _downsample) {
        for (int c = 0; c < cols; c += _downsample) {
            const cv::Vec3f& wp = points(r, c);
            if (wp[0] == -1.0f && wp[1] == -1.0f && wp[2] == -1.0f) {
                continue;
            }

            Handle handle;
            handle.row = r;
            handle.col = c;
            handle.originalWorld = wp;
            handle.currentWorld = wp;
            handle.isManual = false;
            _handles.push_back(handle);
        }
    }

    for (auto& handle : manualHandles) {
        if (handle.row < 0 || handle.row >= rows || handle.col < 0 || handle.col >= cols) {
            continue;
        }
        const cv::Vec3f& orig = (*_originalPoints)(handle.row, handle.col);
        if (orig[0] == -1.0f && orig[1] == -1.0f && orig[2] == -1.0f) {
            continue;
        }
        handle.originalWorld = orig;
        handle.currentWorld = (*_previewPoints)(handle.row, handle.col);
        handle.isManual = true;
        if (!findHandle(handle.row, handle.col)) {
            _handles.push_back(handle);
        }
    }
}

void SegmentationEditManager::applyHandleInfluence(const Handle& handle)
{
    if (!_previewPoints || !_originalPoints || !_baseSurface) {
        return;
    }

    const cv::Vec3f delta = handle.currentWorld - handle.originalWorld;
    if (cv::norm(delta) < 1e-4f) {
        return;
    }

    const cv::Mat_<cv::Vec3f>& original = *_originalPoints;
    cv::Mat_<cv::Vec3f>& preview = *_previewPoints;

    const cv::Vec2f scale = _baseSurface->scale();
    const float radius = std::max(1.0f, _radius);
    const float radiusSq = radius * radius;
    const float sigma = std::max(0.1f, _sigma);
    const float sigmaSq = sigma * sigma;

    const int rowRadius = std::max(1, static_cast<int>(std::ceil(radius / std::max(1e-3f, scale[1]))));
    const int colRadius = std::max(1, static_cast<int>(std::ceil(radius / std::max(1e-3f, scale[0]))));

    const int rowStart = std::max(0, handle.row - rowRadius);
    const int rowEnd = std::min(preview.rows - 1, handle.row + rowRadius);
    const int colStart = std::max(0, handle.col - colRadius);
    const int colEnd = std::min(preview.cols - 1, handle.col + colRadius);

    for (int r = rowStart; r <= rowEnd; ++r) {
        for (int c = colStart; c <= colEnd; ++c) {
            const cv::Vec3f& orig = original(r, c);
            if (orig[0] == -1.0f && orig[1] == -1.0f && orig[2] == -1.0f) {
                continue;
            }

            const float distSq = static_cast<float>(cv::norm(handle.originalWorld - orig));
            if (distSq > radiusSq) {
                continue;
            }

            const float weight = std::exp(-distSq / (2.0f * sigmaSq));
            if (weight < kMinInfluenceWeight) {
                continue;
            }

            preview(r, c) = orig + delta * weight;
        }
    }

    // Keep handle point itself in sync
    if (handle.row >= 0 && handle.row < preview.rows && handle.col >= 0 && handle.col < preview.cols) {
        preview(handle.row, handle.col) = handle.currentWorld;
    }
}

void SegmentationEditManager::syncPreviewFromBase()
{
    if (!_baseSurface || !_previewPoints) {
        return;
    }

    const cv::Mat_<cv::Vec3f>& basePoints = _baseSurface->rawPoints();
    if (!_previewPoints->empty() && basePoints.size() == _previewPoints->size()) {
        basePoints.copyTo(*_previewPoints);
    }
}

void SegmentationEditManager::reapplyAllHandles()
{
    if (!hasSession() || !_previewPoints || !_originalPoints) {
        return;
    }

    _originalPoints->copyTo(*_previewPoints);
    for (const auto& handle : _handles) {
        applyHandleInfluence(handle);
    }
    for (auto& handle : _handles) {
        if (handle.row >= 0 && handle.row < _previewPoints->rows &&
            handle.col >= 0 && handle.col < _previewPoints->cols) {
            handle.currentWorld = (*_previewPoints)(handle.row, handle.col);
        }
    }
}
