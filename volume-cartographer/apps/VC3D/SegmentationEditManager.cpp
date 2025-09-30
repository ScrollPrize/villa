#include "SegmentationEditManager.hpp"

#include "vc/core/util/Surface.hpp"

#include <QDebug>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <stdexcept>

#include <nlohmann/json.hpp>

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

void ensureSurfaceMetaObject(QuadSurface* surface)
{
    if (!surface) {
        return;
    }
    if (surface->meta && surface->meta->is_object()) {
        return;
    }
    if (surface->meta) {
        delete surface->meta;
    }
    surface->meta = new nlohmann::json(nlohmann::json::object());
}

bool containsNearby(const std::vector<cv::Vec3f>& haystack,
                    const cv::Vec3f& needle,
                    float toleranceSq)
{
    for (const auto& point : haystack) {
        if (isInvalidPoint(point)) {
            continue;
        }
        const cv::Vec3f diff = point - needle;
        if (diff.dot(diff) <= toleranceSq) {
            return true;
        }
    }
    return false;
}

struct TriangleSample
{
    cv::Vec3f p0;
    cv::Vec3f p1;
    cv::Vec3f p2;
    cv::Vec2f uv0;
    cv::Vec2f uv1;
    cv::Vec2f uv2;
};

void closestPointOnTriangle(const TriangleSample& tri,
                            const cv::Vec3f& point,
                            cv::Vec3f& closest,
                            cv::Vec3f& bary)
{
    const cv::Vec3f& a = tri.p0;
    const cv::Vec3f& b = tri.p1;
    const cv::Vec3f& c = tri.p2;

    cv::Vec3f ab = b - a;
    cv::Vec3f ac = c - a;
    cv::Vec3f ap = point - a;

    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        bary = cv::Vec3f(1.0f, 0.0f, 0.0f);
        return;
    }

    cv::Vec3f bp = point - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        bary = cv::Vec3f(0.0f, 1.0f, 0.0f);
        return;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        closest = a + v * ab;
        bary = cv::Vec3f(1.0f - v, v, 0.0f);
        return;
    }

    cv::Vec3f cp = point - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        bary = cv::Vec3f(0.0f, 0.0f, 1.0f);
        return;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        closest = a + w * ac;
        bary = cv::Vec3f(1.0f - w, 0.0f, w);
        return;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + w * (c - b);
        bary = cv::Vec3f(0.0f, 1.0f - w, w);
        return;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float u = 1.0f - v - w;
    closest = u * a + v * b + w * c;
    bary = cv::Vec3f(u, v, w);
}

const std::array<cv::Vec2i, 4> kNeighbourOffsets = {
    cv::Vec2i{1, 0},
    cv::Vec2i{-1, 0},
    cv::Vec2i{0, 1},
    cv::Vec2i{0, -1}
};
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

    ensureSurfaceMetaObject(baseSurface);

    _baseSurface = baseSurface;
    _downsample = std::max(1, downsample);
    _autoHandleWorldSnapshot.clear();
    _growthHandleWorld.clear();
    _pendingGrowthMarking = false;

    _originalPoints = std::make_unique<cv::Mat_<cv::Vec3f>>(baseSurface->rawPoints().clone());

    auto* previewMatrix = new cv::Mat_<cv::Vec3f>(_originalPoints->clone());
    _previewSurface = std::make_unique<QuadSurface>(previewMatrix, baseSurface->scale());
    if (_previewSurface->meta) {
        delete _previewSurface->meta;
        _previewSurface->meta = nullptr;
    }
    if (baseSurface->meta) {
        _previewSurface->meta = new nlohmann::json(*baseSurface->meta);
    }
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
    _autoHandleWorldSnapshot.clear();
    _growthHandleWorld.clear();
    _pendingGrowthMarking = false;
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
    const int clamped = std::max(1, static_cast<int>(std::lround(radius)));
    const float snapped = static_cast<float>(clamped);
    if (std::fabs(snapped - _radius) < 1e-4f) {
        return;
    }

    _radius = snapped;
    reapplyAllHandles();
}

void SegmentationEditManager::setSigma(float sigma)
{
    const float clamped = std::clamp(sigma, 0.10f, 2.0f);
    if (std::fabs(clamped - _sigma) < 1e-4f) {
        return;
    }

    _sigma = clamped;
    reapplyAllHandles();
}

void SegmentationEditManager::setInfluenceMode(SegmentationInfluenceMode mode)
{
    if (_influenceMode == mode) {
        return;
    }
    _influenceMode = mode;
    reapplyAllHandles();
}

void SegmentationEditManager::setRowColMode(SegmentationRowColMode mode)
{
    if (_rowColMode == mode) {
        return;
    }
    _rowColMode = mode;
    reapplyAllHandles();
}

void SegmentationEditManager::setHoleSearchRadius(int radius)
{
    const int clamped = std::clamp(radius, 1, 64);
    if (clamped == _holeSearchRadius) {
        return;
    }
    _holeSearchRadius = clamped;
}

void SegmentationEditManager::setHoleSmoothIterations(int iterations)
{
    const int clamped = std::clamp(iterations, 1, 200);
    if (clamped == _holeSmoothIterations) {
        return;
    }
    _holeSmoothIterations = clamped;
}

void SegmentationEditManager::setFillInvalidCells(bool enabled)
{
    if (_fillInvalidCells == enabled) {
        return;
    }
    _fillInvalidCells = enabled;
}

const cv::Mat_<cv::Vec3f>& SegmentationEditManager::previewPoints() const
{
    if (!_previewPoints) {
        throw std::runtime_error("SegmentationEditManager::previewPoints() called without an active session");
    }
    return *_previewPoints;
}

bool SegmentationEditManager::setPreviewPoints(const cv::Mat_<cv::Vec3f>& points,
                                               bool dirtyState)
{
    if (!hasSession() || !_previewPoints) {
        return false;
    }
    if (points.rows != _previewPoints->rows || points.cols != _previewPoints->cols) {
        return false;
    }

    points.copyTo(*_previewPoints);
    if (_previewSurface) {
        _previewSurface->invalidateCache();
    }
    regenerateHandles();
    _dirty = dirtyState;
    return true;
}

bool SegmentationEditManager::invalidateRegion(int centerRow, int centerCol, int radius)
{
    if (!_previewPoints) {
        return false;
    }

    const int rows = _previewPoints->rows;
    const int cols = _previewPoints->cols;
    const int boundedRadius = std::max(0, radius);

    bool changed = false;
    const int rowMin = std::max(0, centerRow - boundedRadius);
    const int rowMax = std::min(rows - 1, centerRow + boundedRadius);
    const int colMin = std::max(0, centerCol - boundedRadius);
    const int colMax = std::min(cols - 1, centerCol + boundedRadius);

    for (int row = rowMin; row <= rowMax; ++row) {
        for (int col = colMin; col <= colMax; ++col) {
            const int dr = std::abs(row - centerRow);
            const int dc = std::abs(col - centerCol);
            if (std::max(dr, dc) > boundedRadius) {
                continue;
            }

            cv::Vec3f& cell = (*_previewPoints)(row, col);
            if (!isInvalidPoint(cell)) {
                cell = cv::Vec3f(-1.0f, -1.0f, -1.0f);
                changed = true;
            }
        }
    }

    if (!changed) {
        return false;
    }

    if (_previewSurface) {
        _previewSurface->invalidateCache();
    }
    regenerateHandles();
    _dirty = true;
    return true;
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

    ensureSurfaceMetaObject(_baseSurface);

    // Update original snapshot to new base state
    *(_originalPoints) = basePoints->clone();
    basePoints->copyTo(*_previewPoints);
    regenerateHandles();
    _dirty = false;
}

void SegmentationEditManager::refreshFromBaseSurface()
{
    if (!hasSession() || !_baseSurface) {
        return;
    }

    ensureSurfaceMetaObject(_baseSurface);

    const cv::Mat_<cv::Vec3f>& basePoints = _baseSurface->rawPoints();

    if (!_originalPoints || _originalPoints->size() != basePoints.size()) {
        _originalPoints = std::make_unique<cv::Mat_<cv::Vec3f>>(basePoints.clone());
    } else {
        basePoints.copyTo(*_originalPoints);
    }

    const auto rebuildPreviewSurface = [this, &basePoints]() {
        auto* previewMatrix = new cv::Mat_<cv::Vec3f>(basePoints.clone());
        _previewSurface = std::make_unique<QuadSurface>(previewMatrix, _baseSurface->scale());
        if (_previewSurface->meta) {
            delete _previewSurface->meta;
            _previewSurface->meta = nullptr;
        }
        if (_baseSurface->meta) {
            _previewSurface->meta = new nlohmann::json(*_baseSurface->meta);
        }
        _previewSurface->path = _baseSurface->path;
        _previewSurface->id = _baseSurface->id;
        _previewPoints = _previewSurface->rawPointsPtr();
    };

    if (!_previewPoints || _previewSurface == nullptr) {
        rebuildPreviewSurface();
    } else if (_previewPoints->size() != basePoints.size()) {
        rebuildPreviewSurface();
    } else {
        basePoints.copyTo(*_previewPoints);
        if (_previewSurface) {
            if (_previewSurface->meta) {
                delete _previewSurface->meta;
                _previewSurface->meta = nullptr;
            }
            if (_baseSurface->meta) {
                _previewSurface->meta = new nlohmann::json(*_baseSurface->meta);
            }
            _previewSurface->path = _baseSurface->path;
            _previewSurface->id = _baseSurface->id;
        }
    }

    if (_previewSurface) {
        _previewSurface->invalidateCache();
    }

    regenerateHandles();
    _dirty = false;
}

constexpr float kMinInfluenceWeight = 1e-3f;

bool SegmentationEditManager::updateHandleWorldPosition(int row,
                                                        int col,
                                                        const cv::Vec3f& newWorldPos,
                                                        std::optional<SegmentationRowColAxis> axisHint)
{
    if (!hasSession() || !_previewPoints || !_originalPoints) {
        qWarning() << "SegmentationEditManager: cannot move handle without an active session";
        return false;
    }

    Handle* handle = findHandle(row, col);
    if (!handle) {
        qWarning() << "SegmentationEditManager: no handle found at" << row << col << "for movement";
        return false;
    }

    if (axisHint.has_value()) {
        handle->rowColAxis = *axisHint;
    } else if (_influenceMode == SegmentationInfluenceMode::RowColumn &&
               handle->rowColAxis == SegmentationRowColAxis::Both) {
        handle->rowColAxis = (_rowColMode == SegmentationRowColMode::ColumnOnly)
                                 ? SegmentationRowColAxis::Column
                                 : SegmentationRowColAxis::Row;
    }

    if (!handle->isManual) {
        handle->isManual = true;
        if (_originalPoints &&
            row >= 0 && row < _originalPoints->rows &&
            col >= 0 && col < _originalPoints->cols) {
            handle->originalWorld = (*_originalPoints)(row, col);
        }
    }

    handle->currentWorld = newWorldPos;
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
    if (_previewSurface) {
        _previewSurface->invalidateCache();
    }
    _dirty = true;
    return true;
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
                                                                           float planeTolerance,
                                                                           bool allowCreate,
                                                                           bool allowReuse,
                                                                           std::optional<SegmentationRowColAxis> axisHint)
{
    if (!hasSession() || !_previewPoints || !_originalPoints) {
        qWarning() << "SegmentationEditManager: cannot add handle without an active session";
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

    cv::Vec3f solvedWorld = worldPos;
    bool createdCell = false;
    std::vector<std::pair<int,int>> holeCells;

    const bool allowHoleFill = allowCreate && _fillInvalidCells;

    if (allowHoleFill) {
        int seedRow = bestRow;
        int seedCol = bestCol;
        if (seedRow < 0 || seedCol < 0) {
            if (fallbackRow >= 0 && fallbackCol >= 0) {
                seedRow = fallbackRow;
                seedCol = fallbackCol;
            } else if (bestPlaneRow >= 0 && bestPlaneCol >= 0) {
                seedRow = bestPlaneRow;
                seedCol = bestPlaneCol;
            }
        }
        std::pair<int,int> newCell;
        if (fillInvalidCellWithLocalSolve(worldPos, seedRow, seedCol, newCell, &solvedWorld)) {
            bestRow = newCell.first;
            bestCol = newCell.second;
            createdCell = true;
        }
    }

    if (bestRow < 0 || bestCol < 0) {
        qWarning() << "SegmentationEditManager: failed to locate a suitable cell for new handle near world position"
                   << worldPos[0] << worldPos[1] << worldPos[2];
        return std::nullopt;
    }

    const bool targetCellInvalid = isInvalidPoint(preview(bestRow, bestCol));
    if (targetCellInvalid) {
        if (!allowHoleFill) {
            qWarning() << "SegmentationEditManager: target cell invalid and hole filling disabled";
            return std::nullopt;
        }
        createdCell = true;
    }

    if (createdCell) {
        if (allowHoleFill) {
            holeCells = collectHoleCells(bestRow, bestCol, _holeSearchRadius);
            if (holeCells.empty()) {
                holeCells.emplace_back(bestRow, bestCol);
            }
        }
    } else {
        solvedWorld = preview(bestRow, bestCol);
    }

    ensurePointValid(_originalPoints.get(), bestRow, bestCol, solvedWorld);
    ensurePointValid(_previewPoints, bestRow, bestCol, solvedWorld);

    if (createdCell && _originalPoints && _previewPoints) {
        (*_originalPoints)(bestRow, bestCol) = solvedWorld;
        (*_previewPoints)(bestRow, bestCol) = solvedWorld;

        if (!holeCells.empty() && allowHoleFill) {
            relaxHolePatch(holeCells, std::make_pair(bestRow, bestCol), solvedWorld);
        }
    }

    if (auto* existing = findHandle(bestRow, bestCol)) {
        if (!existing->isManual) {
            existing->isManual = true;
            existing->originalWorld = (*_originalPoints)(bestRow, bestCol);
            existing->currentWorld = (*_previewPoints)(bestRow, bestCol);
            if (axisHint.has_value()) {
                existing->rowColAxis = *axisHint;
            }
            if (_previewSurface) {
                _previewSurface->invalidateCache();
            }
            _dirty = true;
            return std::make_pair(bestRow, bestCol);
        }
        if (allowReuse) {
            if (axisHint.has_value()) {
                existing->rowColAxis = *axisHint;
            }
            return std::make_pair(bestRow, bestCol);
        }
        qWarning() << "SegmentationEditManager: handle at" << bestRow << bestCol
                   << "is already user-managed; skipping add";
        return std::nullopt;
    }

    Handle handle;
    handle.row = bestRow;
    handle.col = bestCol;
    handle.originalWorld = (*_originalPoints)(bestRow, bestCol);
    handle.currentWorld = (*_previewPoints)(bestRow, bestCol);
    handle.isManual = true;
    if (axisHint.has_value()) {
        handle.rowColAxis = *axisHint;
    } else if (_influenceMode == SegmentationInfluenceMode::RowColumn) {
        handle.rowColAxis = (_rowColMode == SegmentationRowColMode::ColumnOnly)
                                ? SegmentationRowColAxis::Column
                                : SegmentationRowColAxis::Row;
    }
    handle.isGrowth = false;
    _handles.push_back(handle);

    if (_previewSurface) {
        _previewSurface->invalidateCache();
    }
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

void SegmentationEditManager::markNextRefreshHandlesAsGrowth()
{
    if (!hasSession()) {
        return;
    }
    _pendingGrowthMarking = true;
}

bool SegmentationEditManager::fillInvalidCellWithLocalSolve(const cv::Vec3f& worldPos,
                                                            int seedRow,
                                                            int seedCol,
                                                            std::pair<int,int>& outCell,
                                                            cv::Vec3f* outWorld)
{
    if (!_previewPoints) {
        return false;
    }

    const cv::Mat_<cv::Vec3f>& preview = *_previewPoints;
    const int rows = preview.rows;
    const int cols = preview.cols;

    if (seedRow < 0 || seedCol < 0 || seedRow >= rows || seedCol >= cols) {
        if (auto idx = worldToGridIndex(worldPos)) {
            seedRow = idx->first;
            seedCol = idx->second;
        } else {
            seedRow = std::clamp(static_cast<int>(rows / 2), 0, rows - 1);
            seedCol = std::clamp(static_cast<int>(cols / 2), 0, cols - 1);
        }
    }

    std::vector<TriangleSample> triangles;
    triangles.reserve(512);

    if (rows < 2 || cols < 2) {
        return false;
    }

    constexpr int kRadius = 8;
    const int rowStart = std::max(0, seedRow - kRadius);
    const int rowEnd = std::min(rows - 2, seedRow + kRadius);
    const int colStart = std::max(0, seedCol - kRadius);
    const int colEnd = std::min(cols - 2, seedCol + kRadius);

    const auto accumulateTriangles = [&](int rStart, int rEnd, int cStart, int cEnd) {
        for (int r = rStart; r <= rEnd; ++r) {
            for (int c = cStart; c <= cEnd; ++c) {
                const cv::Vec3f& p00 = preview(r, c);
                const cv::Vec3f& p10 = preview(r + 1, c);
                const cv::Vec3f& p01 = preview(r, c + 1);
                const cv::Vec3f& p11 = preview(r + 1, c + 1);

                if (!isInvalidPoint(p00) && !isInvalidPoint(p10) && !isInvalidPoint(p01)) {
                    triangles.push_back({p00, p10, p01,
                                         cv::Vec2f(static_cast<float>(c), static_cast<float>(r)),
                                         cv::Vec2f(static_cast<float>(c), static_cast<float>(r + 1)),
                                         cv::Vec2f(static_cast<float>(c + 1), static_cast<float>(r))});
                }

                if (!isInvalidPoint(p11) && !isInvalidPoint(p01) && !isInvalidPoint(p10)) {
                    triangles.push_back({p11, p01, p10,
                                         cv::Vec2f(static_cast<float>(c + 1), static_cast<float>(r + 1)),
                                         cv::Vec2f(static_cast<float>(c + 1), static_cast<float>(r)),
                                         cv::Vec2f(static_cast<float>(c), static_cast<float>(r + 1))});
                }
            }
        }
    };

    if (rowStart <= rowEnd && colStart <= colEnd) {
        accumulateTriangles(rowStart, rowEnd, colStart, colEnd);
    }

    if (triangles.empty()) {
        accumulateTriangles(0, rows - 2, 0, cols - 2);
    }

    if (triangles.empty()) {
        return false;
    }

    bool found = false;
    float bestDistance = std::numeric_limits<float>::max();
    cv::Vec2f bestUV(0.0f, 0.0f);
    cv::Vec3f bestWorld = worldPos;

    for (const TriangleSample& tri : triangles) {
        cv::Vec3f closest;
        cv::Vec3f bary;
        closestPointOnTriangle(tri, worldPos, closest, bary);
        const float distance = cv::norm(worldPos - closest);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestUV = bary[0] * tri.uv0 + bary[1] * tri.uv1 + bary[2] * tri.uv2;
            found = true;
            bestWorld = closest;
        }
    }

    if (!found) {
        return false;
    }

    int targetRow = std::clamp(static_cast<int>(std::lround(bestUV[1])), 0, rows - 1);
    int targetCol = std::clamp(static_cast<int>(std::lround(bestUV[0])), 0, cols - 1);

    constexpr int kSearchRadius = 4;
    float bestScore = std::numeric_limits<float>::max();
    std::optional<std::pair<int,int>> bestCell;
    for (int r = std::max(0, targetRow - kSearchRadius); r <= std::min(rows - 1, targetRow + kSearchRadius); ++r) {
        for (int c = std::max(0, targetCol - kSearchRadius); c <= std::min(cols - 1, targetCol + kSearchRadius); ++c) {
            if (!isInvalidPoint(preview(r, c))) {
                continue;
            }
            const float score = std::hypot(static_cast<float>(r) - bestUV[1],
                                           static_cast<float>(c) - bestUV[0]);
            if (score < bestScore) {
                bestScore = score;
                bestCell = std::make_pair(r, c);
            }
        }
    }

    if (!bestCell) {
        return false;
    }

    outCell = *bestCell;
    if (outWorld) {
        *outWorld = bestWorld;
    }
    return true;
}

std::vector<std::pair<int,int>> SegmentationEditManager::collectHoleCells(int centerRow, int centerCol, int radius) const
{
    std::vector<std::pair<int,int>> result;
    if (!_previewPoints) {
        return result;
    }

    const cv::Mat_<cv::Vec3f>& preview = *_previewPoints;
    const int rows = preview.rows;
    const int cols = preview.cols;

    if (centerRow < 0 || centerCol < 0 || centerRow >= rows || centerCol >= cols) {
        return result;
    }

    const int rowStart = std::max(0, centerRow - radius);
    const int rowEnd = std::min(rows - 1, centerRow + radius);
    const int colStart = std::max(0, centerCol - radius);
    const int colEnd = std::min(cols - 1, centerCol + radius);

    const int width = colEnd - colStart + 1;
    const int height = rowEnd - rowStart + 1;
    if (width <= 0 || height <= 0) {
        return result;
    }

    const auto indexFor = [&](int r, int c) {
        return (r - rowStart) * width + (c - colStart);
    };

    std::vector<uint8_t> visited(static_cast<std::size_t>(width * height), 0);
    std::queue<std::pair<int,int>> frontier;

    if (isInvalidPoint(preview(centerRow, centerCol))) {
        frontier.emplace(centerRow, centerCol);
        visited[indexFor(centerRow, centerCol)] = 1;
    } else {
        result.emplace_back(centerRow, centerCol);
        return result;
    }

    while (!frontier.empty()) {
        auto [row, col] = frontier.front();
        frontier.pop();
        result.emplace_back(row, col);

        for (const auto& offset : kNeighbourOffsets) {
            const int nr = row + offset[0];
            const int nc = col + offset[1];
            if (nr < rowStart || nr > rowEnd || nc < colStart || nc > colEnd) {
                continue;
            }
            const int idx = indexFor(nr, nc);
            if (visited[idx]) {
                continue;
            }
            if (!isInvalidPoint(preview(nr, nc))) {
                continue;
            }
            visited[idx] = 1;
            frontier.emplace(nr, nc);
        }
    }

    if (result.empty()) {
        result.emplace_back(centerRow, centerCol);
    }
    return result;
}

void SegmentationEditManager::relaxHolePatch(const std::vector<std::pair<int,int>>& holeCells,
                                             const std::pair<int,int>& seedCell,
                                             const cv::Vec3f& seedWorld)
{
    if (holeCells.empty() || !_previewPoints || !_originalPoints) {
        return;
    }

    const int cols = _previewPoints->cols;

    std::unordered_map<int, int> cellIndex;
    cellIndex.reserve(holeCells.size());
    for (std::size_t i = 0; i < holeCells.size(); ++i) {
        const auto& cell = holeCells[i];
        cellIndex[cell.first * cols + cell.second] = static_cast<int>(i);
    }

    struct Node
    {
        cv::Vec3f value;
        cv::Vec3f scratch;
        bool fixed{false};
    };

    std::vector<Node> nodes(holeCells.size());
    for (std::size_t i = 0; i < holeCells.size(); ++i) {
        const auto& cell = holeCells[i];
        const bool isSeed = cell == seedCell;
        nodes[i].fixed = isSeed;
        nodes[i].value = isSeed ? seedWorld : seedWorld;
    }

    // Initialise non-seed cells with averages from available neighbours to reduce drift.
    for (std::size_t i = 0; i < holeCells.size(); ++i) {
        if (nodes[i].fixed) {
            continue;
        }
        const auto& [row, col] = holeCells[i];
        cv::Vec3f accum(0.0f, 0.0f, 0.0f);
        int count = 0;
        for (const auto& offset : kNeighbourOffsets) {
            const int nr = row + offset[0];
            const int nc = col + offset[1];
            if (nr < 0 || nc < 0 || nr >= _previewPoints->rows || nc >= _previewPoints->cols) {
                continue;
            }
            const int key = nr * cols + nc;
            auto it = cellIndex.find(key);
            if (it != cellIndex.end()) {
                accum += nodes[it->second].value;
                ++count;
            } else {
                const cv::Vec3f& neighbour = (*_previewPoints)(nr, nc);
                if (!isInvalidPoint(neighbour)) {
                    accum += neighbour;
                    ++count;
                }
            }
        }
        if (count > 0) {
            nodes[i].value = accum * (1.0f / static_cast<float>(count));
        }
    }

    const int iterations = std::max(1, _holeSmoothIterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t i = 0; i < holeCells.size(); ++i) {
            if (nodes[i].fixed) {
                nodes[i].scratch = nodes[i].value;
                continue;
            }

            const auto& [row, col] = holeCells[i];
            cv::Vec3f accum(0.0f, 0.0f, 0.0f);
            int count = 0;
            for (const auto& offset : kNeighbourOffsets) {
                const int nr = row + offset[0];
                const int nc = col + offset[1];
                if (nr < 0 || nc < 0 || nr >= _previewPoints->rows || nc >= _previewPoints->cols) {
                    continue;
                }
                const int key = nr * cols + nc;
                auto it = cellIndex.find(key);
                if (it != cellIndex.end()) {
                    accum += nodes[it->second].value;
                    ++count;
                } else {
                    const cv::Vec3f& neighbour = (*_previewPoints)(nr, nc);
                    if (!isInvalidPoint(neighbour)) {
                        accum += neighbour;
                        ++count;
                    }
                }
            }

            if (count > 0) {
                nodes[i].scratch = accum * (1.0f / static_cast<float>(count));
            } else {
                nodes[i].scratch = nodes[i].value;
            }
        }

        for (std::size_t i = 0; i < holeCells.size(); ++i) {
            if (!nodes[i].fixed) {
                nodes[i].value = nodes[i].scratch;
            }
        }
    }

    for (std::size_t i = 0; i < holeCells.size(); ++i) {
        const auto& cell = holeCells[i];
        const cv::Vec3f& value = nodes[i].value;
        (*_originalPoints)(cell.first, cell.second) = value;
        (*_previewPoints)(cell.first, cell.second) = value;
    }
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

    const std::vector<cv::Vec3f> previousAutoWorld = _autoHandleWorldSnapshot;
    const std::vector<cv::Vec3f> previousGrowthWorld = _growthHandleWorld;

    _handles.clear();
    if (!hasSession() || !_previewPoints) {
        _handles = manualHandles;
        _pendingGrowthMarking = false;
        _autoHandleWorldSnapshot.clear();
        _growthHandleWorld.clear();
        return;
    }

    const cv::Mat_<cv::Vec3f>& points = *_previewPoints;
    const int rows = points.rows;
    const int cols = points.cols;
    const int stride = std::max(1, _downsample);

    const float gridStep = estimateGridStepWorld();
    const float matchTolerance = std::max(gridStep * 0.5f, 1.0f);
    const float matchToleranceSq = matchTolerance * matchTolerance;

    std::vector<cv::Vec3f> newAutoWorld;
    newAutoWorld.reserve((rows / stride + 1) * (cols / stride + 1));
    std::vector<cv::Vec3f> newGrowthWorld;
    newGrowthWorld.reserve(previousGrowthWorld.size());

    for (int r = 0; r < rows; r += stride) {
        for (int c = 0; c < cols; c += stride) {
            const cv::Vec3f& wp = points(r, c);
            if (isInvalidPoint(wp)) {
                continue;
            }

            Handle handle;
            handle.row = r;
            handle.col = c;
            handle.originalWorld = wp;
            handle.currentWorld = wp;
            handle.isManual = false;
            handle.rowColAxis = SegmentationRowColAxis::Both;

            const bool existedBefore = containsNearby(previousAutoWorld, wp, matchToleranceSq);
            const bool wasGrowth = containsNearby(previousGrowthWorld, wp, matchToleranceSq);
            const bool markAsNewGrowth = _pendingGrowthMarking && !existedBefore;

            handle.isGrowth = wasGrowth || markAsNewGrowth;
            if (handle.isGrowth) {
                newGrowthWorld.push_back(wp);
            }

            newAutoWorld.push_back(wp);
            _handles.push_back(handle);
        }
    }

    for (auto& handle : manualHandles) {
        if (handle.row < 0 || handle.row >= rows || handle.col < 0 || handle.col >= cols) {
            continue;
        }
        const cv::Vec3f& orig = (*_originalPoints)(handle.row, handle.col);
        if (isInvalidPoint(orig)) {
            continue;
        }
        handle.originalWorld = orig;
        handle.currentWorld = (*_previewPoints)(handle.row, handle.col);
        handle.isManual = true;
        handle.isGrowth = false;
        if (!findHandle(handle.row, handle.col)) {
            _handles.push_back(handle);
        }
    }

    _autoHandleWorldSnapshot = std::move(newAutoWorld);
    _growthHandleWorld = std::move(newGrowthWorld);
    _pendingGrowthMarking = false;
}

void SegmentationEditManager::applyHandleInfluence(const Handle& handle)
{
    if (!_previewPoints || !_originalPoints || !_baseSurface) {
        return;
    }

    const cv::Vec3f delta = handle.currentWorld - handle.originalWorld;
    const bool hasDelta = cv::norm(delta) >= 1e-4f;

    if (hasDelta) {
        if (_influenceMode == SegmentationInfluenceMode::GeodesicCircular) {
            applyHandleInfluenceGeodesic(handle, delta);
        } else if (_influenceMode == SegmentationInfluenceMode::RowColumn) {
            applyHandleInfluenceRowCol(handle, delta);
        } else {
            applyHandleInfluenceGrid(handle, delta);
        }
    }

    if (handle.row >= 0 && handle.row < _previewPoints->rows &&
        handle.col >= 0 && handle.col < _previewPoints->cols) {
        (*_previewPoints)(handle.row, handle.col) = handle.currentWorld;
    }
}

void SegmentationEditManager::applyHandleInfluenceGrid(const Handle& handle, const cv::Vec3f& delta)
{
    const cv::Mat_<cv::Vec3f>& original = *_originalPoints;
    cv::Mat_<cv::Vec3f>& preview = *_previewPoints;

    const int gridRadius = std::max(1, static_cast<int>(std::lround(_radius)));
    const float strength = std::clamp(_sigma, 0.10f, 2.0f);
    const float maxWeight = std::min(1.5f, std::max(strength, 0.0f));
    const float denom = static_cast<float>(gridRadius + 1);

    const int rowStart = std::max(0, handle.row - gridRadius);
    const int rowEnd = std::min(preview.rows - 1, handle.row + gridRadius);
    const int colStart = std::max(0, handle.col - gridRadius);
    const int colEnd = std::min(preview.cols - 1, handle.col + gridRadius);

    for (int r = rowStart; r <= rowEnd; ++r) {
        for (int c = colStart; c <= colEnd; ++c) {
            if (r == handle.row && c == handle.col) {
                continue;
            }
            const cv::Vec3f& orig = original(r, c);
            if (isInvalidPoint(orig)) {
                continue;
            }

            const int dr = r - handle.row;
            const int dc = c - handle.col;
            const int gridDistance = std::max(std::abs(dr), std::abs(dc));
            if (gridDistance > gridRadius) {
                continue;
            }

            float normalized = static_cast<float>(gridDistance) / denom;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            const float falloff = std::max(0.0f, 1.0f - normalized);
            float weight = strength * falloff;
            weight = std::clamp(weight, 0.0f, maxWeight);

            if (weight < kMinInfluenceWeight) {
                continue;
            }

            preview(r, c) = orig + delta * weight;
        }
    }
}

void SegmentationEditManager::applyHandleInfluenceGeodesic(const Handle& handle, const cv::Vec3f& delta)
{
    const cv::Mat_<cv::Vec3f>& original = *_originalPoints;
    cv::Mat_<cv::Vec3f>& preview = *_previewPoints;

    const int gridRadius = std::max(1, static_cast<int>(std::lround(_radius)));
    const int windowRadius = std::max(gridRadius * 2, gridRadius + 1);
    const int rowStart = std::max(0, handle.row - windowRadius);
    const int rowEnd = std::min(preview.rows - 1, handle.row + windowRadius);
    const int colStart = std::max(0, handle.col - windowRadius);
    const int colEnd = std::min(preview.cols - 1, handle.col + windowRadius);

    if (rowEnd < rowStart || colEnd < colStart) {
        return;
    }

    const int windowRows = rowEnd - rowStart + 1;
    const int windowCols = colEnd - colStart + 1;
    const int windowSize = windowRows * windowCols;

    auto localIndex = [rowStart, colStart, windowCols](int r, int c) {
        return (r - rowStart) * windowCols + (c - colStart);
    };

    auto isValidCell = [&](int r, int c) {
        if (r < rowStart || r > rowEnd || c < colStart || c > colEnd) {
            return false;
        }
        const cv::Vec3f& orig = original(r, c);
        return !isInvalidPoint(orig);
    };

    if (!isValidCell(handle.row, handle.col)) {
        return;
    }

    const float strength = std::clamp(_sigma, 0.10f, 2.0f);
    const float maxWeight = std::min(1.5f, std::max(strength, 0.0f));
    const float stepWorld = estimateGridStepWorld();
    const float radiusWorld = std::max(stepWorld, (static_cast<float>(gridRadius) + 0.5f) * stepWorld);

    struct Node
    {
        float dist;
        int row;
        int col;
    };

    auto cmp = [](const Node& a, const Node& b) {
        return a.dist > b.dist;
    };

    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> queue(cmp);
    std::vector<float> distances(windowSize, std::numeric_limits<float>::max());

    const int startIndex = localIndex(handle.row, handle.col);
    distances[startIndex] = 0.0f;
    queue.push({0.0f, handle.row, handle.col});

    const std::array<cv::Vec2i, 8> neighbourOffsets = {
        cv::Vec2i{1, 0},
        cv::Vec2i{-1, 0},
        cv::Vec2i{0, 1},
        cv::Vec2i{0, -1},
        cv::Vec2i{1, 1},
        cv::Vec2i{1, -1},
        cv::Vec2i{-1, 1},
        cv::Vec2i{-1, -1}
    };

    while (!queue.empty()) {
        Node current = queue.top();
        queue.pop();

        const int currentIndex = localIndex(current.row, current.col);
        if (current.dist > distances[currentIndex]) {
            continue;
        }
        if (current.dist > radiusWorld) {
            continue;
        }

        if (!(current.row == handle.row && current.col == handle.col)) {
            const cv::Vec3f& orig = original(current.row, current.col);
            float normalized = current.dist / radiusWorld;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            const float falloff = std::max(0.0f, 1.0f - normalized);
            float weight = strength * falloff;
            weight = std::clamp(weight, 0.0f, maxWeight);
            if (weight >= kMinInfluenceWeight) {
                preview(current.row, current.col) = orig + delta * weight;
            }
        }

        const cv::Vec3f& currentOrig = original(current.row, current.col);

        for (const auto& offset : neighbourOffsets) {
            const int nr = current.row + offset[0];
            const int nc = current.col + offset[1];
            if (!isValidCell(nr, nc)) {
                continue;
            }
            const float edgeLength = static_cast<float>(cv::norm(original(nr, nc) - currentOrig));
            float edgeCost = edgeLength;
            if (!std::isfinite(edgeCost) || edgeCost < 1e-5f) {
                edgeCost = stepWorld;
            }
            const float newDist = current.dist + edgeCost;
            if (newDist > radiusWorld) {
                continue;
            }
            const int neighbourIndex = localIndex(nr, nc);
            if (newDist + 1e-5f < distances[neighbourIndex]) {
                distances[neighbourIndex] = newDist;
                queue.push({newDist, nr, nc});
            }
        }
    }
}

void SegmentationEditManager::applyHandleInfluenceRowCol(const Handle& handle, const cv::Vec3f& delta)
{
    const cv::Mat_<cv::Vec3f>& original = *_originalPoints;
    cv::Mat_<cv::Vec3f>& preview = *_previewPoints;

    const int radius = std::max(1, static_cast<int>(std::lround(_radius)));
    if (radius <= 0) {
        return;
    }

    if (handle.row < 0 || handle.row >= original.rows ||
        handle.col < 0 || handle.col >= original.cols) {
        return;
    }

    const cv::Vec3f& centerOriginal = handle.originalWorld;
    if (isInvalidPoint(centerOriginal)) {
        return;
    }

    const float strength = std::clamp(_sigma, 0.10f, 2.0f);
    const float baseMaxWeight = std::min(1.5f, std::max(strength * 1.25f, 0.0f));
    const float denom = static_cast<float>(radius + 1);

    const float deltaNorm = static_cast<float>(cv::norm(delta));
    const cv::Vec3f deltaUnit = (deltaNorm > 1e-5f)
                                    ? (delta / deltaNorm)
                                    : cv::Vec3f(0.0f, 0.0f, 0.0f);

    auto directionUnit = [&](int rowOffset, int colOffset) -> std::optional<cv::Vec3f> {
        const int nr = handle.row + rowOffset;
        const int nc = handle.col + colOffset;
        if (nr < 0 || nr >= original.rows || nc < 0 || nc >= original.cols) {
            return std::nullopt;
        }
        const cv::Vec3f& neighbour = original(nr, nc);
        if (isInvalidPoint(neighbour)) {
            return std::nullopt;
        }
        cv::Vec3f vec = neighbour - centerOriginal;
        const float len = static_cast<float>(cv::norm(vec));
        if (len < 1e-5f) {
            return std::nullopt;
        }
        return vec / len;
    };

    auto alignmentFor = [&](int rowOffset, int colOffset) {
        if (deltaNorm <= 1e-5f) {
            return 0.0f;
        }
        if (auto dir = directionUnit(rowOffset, colOffset)) {
            return static_cast<float>(dir->dot(deltaUnit));
        }
        return 0.0f;
    };

    const float aboveAlignment = alignmentFor(-1, 0);
    const float belowAlignment = alignmentFor(1, 0);
    const float leftAlignment = alignmentFor(0, -1);
    const float rightAlignment = alignmentFor(0, 1);

    auto directionalBias = [&](float alignment) {
        const float deltaSigma = strength - 1.0f;
        float bias = 1.0f + deltaSigma * alignment;
        return std::clamp(bias, 0.1f, 3.0f);
    };

    const float aboveBias = directionalBias(aboveAlignment);
    const float belowBias = directionalBias(belowAlignment);
    const float leftBias = directionalBias(leftAlignment);
    const float rightBias = directionalBias(rightAlignment);

    auto baseWeightForDistance = [&](int distance) {
        float normalized = static_cast<float>(distance) / denom;
        normalized = std::clamp(normalized, 0.0f, 1.0f);
        const float falloff = std::max(0.0f, 1.0f - normalized);
        float weight = strength * falloff;
        return std::clamp(weight, 0.0f, baseMaxWeight);
    };

    auto applyRow = [&]() {
        const int r = handle.row;
        const int colStart = std::max(0, handle.col - radius);
        const int colEnd = std::min(original.cols - 1, handle.col + radius);
        for (int c = colStart; c <= colEnd; ++c) {
            if (c == handle.col) {
                continue;
            }
            const cv::Vec3f& orig = original(r, c);
            if (isInvalidPoint(orig)) {
                continue;
            }
            const int distance = std::abs(c - handle.col);
            float weight = baseWeightForDistance(distance);
            if (c < handle.col) {
                weight *= leftBias;
            } else {
                weight *= rightBias;
            }
            weight = std::clamp(weight, 0.0f, baseMaxWeight);
            if (weight < kMinInfluenceWeight) {
                continue;
            }
            preview(r, c) = orig + delta * weight;
        }
    };

    auto applyColumn = [&]() {
        const int c = handle.col;
        const int rowStart = std::max(0, handle.row - radius);
        const int rowEnd = std::min(original.rows - 1, handle.row + radius);
        for (int r = rowStart; r <= rowEnd; ++r) {
            if (r == handle.row) {
                continue;
            }
            const cv::Vec3f& orig = original(r, c);
            if (isInvalidPoint(orig)) {
                continue;
            }
            const int distance = std::abs(r - handle.row);
            float weight = baseWeightForDistance(distance);
            if (r < handle.row) {
                weight *= aboveBias;
            } else {
                weight *= belowBias;
            }
            weight = std::clamp(weight, 0.0f, baseMaxWeight);
            if (weight < kMinInfluenceWeight) {
                continue;
            }
            preview(r, c) = orig + delta * weight;
        }
    };

    SegmentationRowColAxis axis = handle.rowColAxis;
    if (_rowColMode == SegmentationRowColMode::RowOnly) {
        axis = SegmentationRowColAxis::Row;
    } else if (_rowColMode == SegmentationRowColMode::ColumnOnly) {
        axis = SegmentationRowColAxis::Column;
    } else if (axis == SegmentationRowColAxis::Both) {
        axis = SegmentationRowColAxis::Row;
    }

    if (axis == SegmentationRowColAxis::Row) {
        applyRow();
    } else if (axis == SegmentationRowColAxis::Column) {
        applyColumn();
    } else {
        applyRow();
        applyColumn();
    }
}

float SegmentationEditManager::estimateGridStepWorld() const
{
    if (_baseSurface) {
        const cv::Vec2f scale = _baseSurface->scale();
        const float sx = std::fabs(scale[0]);
        const float sy = std::fabs(scale[1]);
        const float step = std::max(sx, sy);
        if (std::isfinite(step) && step > 1e-4f) {
            return step;
        }
    }
    return 1.0f;
}

void SegmentationEditManager::syncPreviewFromBase()
{
    if (!_baseSurface || !_previewPoints) {
        return;
    }

    ensureSurfaceMetaObject(_baseSurface);

    const cv::Mat_<cv::Vec3f>& basePoints = _baseSurface->rawPoints();
    if (!_previewPoints->empty() && basePoints.size() == _previewPoints->size()) {
        basePoints.copyTo(*_previewPoints);
        if (_previewSurface) {
            if (_previewSurface->meta) {
                delete _previewSurface->meta;
                _previewSurface->meta = nullptr;
            }
            if (_baseSurface->meta) {
                _previewSurface->meta = new nlohmann::json(*_baseSurface->meta);
            }
            _previewSurface->path = _baseSurface->path;
            _previewSurface->id = _baseSurface->id;
        }
    }
}

void SegmentationEditManager::bakePreviewToOriginal()
{
    if (!_previewPoints || !_originalPoints) {
        return;
    }

    _previewPoints->copyTo(*_originalPoints);

    for (auto& handle : _handles) {
        if (handle.row < 0 || handle.col < 0) {
            continue;
        }
        if (handle.row >= _previewPoints->rows || handle.col >= _previewPoints->cols) {
            continue;
        }
        const cv::Vec3f& value = (*_previewPoints)(handle.row, handle.col);
        handle.originalWorld = value;
        handle.currentWorld = value;
    }

    auto manualBegin = std::remove_if(_handles.begin(), _handles.end(), [](const Handle& h) {
        return h.isManual;
    });
    if (manualBegin != _handles.end()) {
        _handles.erase(manualBegin, _handles.end());
    }

    if (_previewSurface) {
        _previewSurface->invalidateCache();
    }

    regenerateHandles();
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
