#include "SegmentationEditManager.hpp"

#include "vc/core/util/Surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <nlohmann/json.hpp>

namespace
{
bool isInvalidPoint(const cv::Vec3f& value)
{
    return !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
           (value[0] == -1.0f && value[1] == -1.0f && value[2] == -1.0f);
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
}

SegmentationEditManager::SegmentationEditManager(QObject* parent)
    : QObject(parent)
{
}

bool SegmentationEditManager::beginSession(QuadSurface* baseSurface)
{
    if (!baseSurface) {
        return false;
    }

    ensureSurfaceMetaObject(baseSurface);

    _baseSurface = baseSurface;
    _gridScale = baseSurface->scale();

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

    _editedVertices.clear();
    _recentTouched.clear();
    clearActiveDrag();
    rebuildPreviewFromOriginal();

    _dirty = false;
    _pendingGrowthMarking = false;
    return true;
}

void SegmentationEditManager::endSession()
{
    _editedVertices.clear();
    _recentTouched.clear();
    clearActiveDrag();

    _previewSurface.reset();
    _previewPoints = nullptr;
    _originalPoints.reset();
    _baseSurface = nullptr;
    _dirty = false;
    _pendingGrowthMarking = false;
}

void SegmentationEditManager::setRadius(float radiusSteps)
{
    if (!std::isfinite(radiusSteps)) {
        return;
    }
    _radiusSteps = std::clamp(radiusSteps, 0.25f, 128.0f);
}

void SegmentationEditManager::setSigma(float sigmaSteps)
{
    if (!std::isfinite(sigmaSteps)) {
        return;
    }
    _sigmaSteps = std::clamp(sigmaSteps, 0.05f, 32.0f);
}

const cv::Mat_<cv::Vec3f>& SegmentationEditManager::previewPoints() const
{
    static const cv::Mat_<cv::Vec3f> kEmpty;
    if (_previewPoints) {
        return *_previewPoints;
    }
    return kEmpty;
}

bool SegmentationEditManager::setPreviewPoints(const cv::Mat_<cv::Vec3f>& points, bool dirtyState)
{
    if (!_previewPoints) {
        return false;
    }
    if (points.rows != _previewPoints->rows || points.cols != _previewPoints->cols) {
        return false;
    }

    points.copyTo(*_previewPoints);
    if (_originalPoints) {
        points.copyTo(*_originalPoints);
    }
    _editedVertices.clear();
    _dirty = dirtyState;
    return true;
}

void SegmentationEditManager::resetPreview()
{
    rebuildPreviewFromOriginal();
    _editedVertices.clear();
    _recentTouched.clear();
    clearActiveDrag();
    _dirty = false;
}

void SegmentationEditManager::applyPreview()
{
    if (!_baseSurface || !_previewPoints) {
        return;
    }

    if (auto* basePoints = _baseSurface->rawPointsPtr()) {
        _previewPoints->copyTo(*basePoints);
    }

    if (_originalPoints) {
        _previewPoints->copyTo(*_originalPoints);
    }

    _editedVertices.clear();
    _recentTouched.clear();
    clearActiveDrag();
    _dirty = false;
}

void SegmentationEditManager::refreshFromBaseSurface()
{
    if (!_baseSurface) {
        return;
    }
    _gridScale = _baseSurface->scale();

    auto current = _baseSurface->rawPoints();
    if (!_originalPoints) {
        _originalPoints = std::make_unique<cv::Mat_<cv::Vec3f>>(current.clone());
    } else {
        current.copyTo(*_originalPoints);
    }

    if (!_previewSurface) {
        auto* previewMatrix = new cv::Mat_<cv::Vec3f>(_originalPoints->clone());
        _previewSurface = std::make_unique<QuadSurface>(previewMatrix, _baseSurface->scale());
        _previewPoints = _previewSurface->rawPointsPtr();
    }

    rebuildPreviewFromOriginal();
    _dirty = !_editedVertices.empty();
}

std::optional<std::pair<int, int>> SegmentationEditManager::worldToGridIndex(const cv::Vec3f& worldPos,
                                                                              float* outDistance) const
{
    if (!_baseSurface) {
        return std::nullopt;
    }

    cv::Vec3f ptr = _baseSurface->pointer();
    const float distance = _baseSurface->pointTo(ptr, worldPos, std::numeric_limits<float>::max(), 400);
    cv::Vec3f raw = _baseSurface->loc_raw(ptr);

    auto* points = _baseSurface->rawPointsPtr();
    if (!points) {
        return std::nullopt;
    }

    int col = static_cast<int>(std::round(raw[0]));
    int row = static_cast<int>(std::round(raw[1]));

    if (row < 0 || row >= points->rows || col < 0 || col >= points->cols) {
        row = std::clamp(row, 0, points->rows - 1);
        col = std::clamp(col, 0, points->cols - 1);
    }

    if (outDistance) {
        *outDistance = distance;
    }

    return std::make_pair(row, col);
}

std::optional<cv::Vec3f> SegmentationEditManager::vertexWorldPosition(int row, int col) const
{
    if (!_previewPoints) {
        return std::nullopt;
    }
    if (row < 0 || row >= _previewPoints->rows || col < 0 || col >= _previewPoints->cols) {
        return std::nullopt;
    }
    const cv::Vec3f& world = (*_previewPoints)(row, col);
    if (isInvalidPoint(world)) {
        return std::nullopt;
    }
    return world;
}

bool SegmentationEditManager::beginActiveDrag(const std::pair<int, int>& gridIndex)
{
    if (!_previewPoints) {
        return false;
    }
    clearActiveDrag();
    if (!buildActiveSamples(gridIndex)) {
        return false;
    }
    _activeDrag.active = true;
    _activeDrag.center = GridKey{gridIndex.first, gridIndex.second};
    _activeDrag.baseWorld = (*_previewPoints)(gridIndex.first, gridIndex.second);
    _activeDrag.targetWorld = _activeDrag.baseWorld;
    return true;
}

bool SegmentationEditManager::updateActiveDrag(const cv::Vec3f& newCenterWorld)
{
    if (!_activeDrag.active || !_previewPoints) {
        return false;
    }

    const cv::Vec3f delta = newCenterWorld - _activeDrag.baseWorld;
    _activeDrag.targetWorld = newCenterWorld;
    applyGaussianToSamples(delta);
    return true;
}

void SegmentationEditManager::commitActiveDrag()
{
    if (!_activeDrag.active) {
        return;
    }
    _activeDrag.active = false;
    _activeDrag.samples.clear();
    _recentTouched.clear();
}

void SegmentationEditManager::cancelActiveDrag()
{
    if (!_activeDrag.active || !_previewPoints) {
        return;
    }

    for (const auto& sample : _activeDrag.samples) {
        (*_previewPoints)(sample.row, sample.col) = sample.baseWorld;
        recordVertexEdit(sample.row, sample.col, sample.baseWorld);
    }

    _recentTouched.clear();
    clearActiveDrag();
}

std::vector<SegmentationEditManager::VertexEdit> SegmentationEditManager::editedVertices() const
{
    std::vector<VertexEdit> result;
    result.reserve(_editedVertices.size());
    for (const auto& entry : _editedVertices) {
        result.push_back(entry.second);
    }
    return result;
}

void SegmentationEditManager::markNextEditsAsGrowth()
{
    _pendingGrowthMarking = true;
}

void SegmentationEditManager::bakePreviewToOriginal()
{
    if (!_previewPoints || !_originalPoints) {
        return;
    }

    _previewPoints->copyTo(*_originalPoints);
    _editedVertices.clear();
    _recentTouched.clear();
    clearActiveDrag();
    _dirty = false;
}

bool SegmentationEditManager::invalidateRegion(int centerRow, int centerCol, int radius)
{
    if (!_originalPoints || !_previewPoints) {
        return false;
    }
    if (radius <= 0) {
        return false;
    }

    const int rows = _previewPoints->rows;
    const int cols = _previewPoints->cols;
    const int rowStart = std::max(0, centerRow - radius);
    const int rowEnd = std::min(rows - 1, centerRow + radius);
    const int colStart = std::max(0, centerCol - radius);
    const int colEnd = std::min(cols - 1, centerCol + radius);

    bool changed = false;

    for (int r = rowStart; r <= rowEnd; ++r) {
        for (int c = colStart; c <= colEnd; ++c) {
            const cv::Vec3f& original = (*_originalPoints)(r, c);
            if (isInvalidPoint(original)) {
                continue;
            }
            (*_previewPoints)(r, c) = original;
            recordVertexEdit(r, c, original);
            changed = true;
        }
    }

    if (changed) {
        _dirty = true;
    }

    return changed;
}

bool SegmentationEditManager::markInvalidRegion(int centerRow, int centerCol, float radiusSteps)
{
    if (!_previewPoints) {
        return false;
    }

    const int rows = _previewPoints->rows;
    const int cols = _previewPoints->cols;
    if (rows <= 0 || cols <= 0) {
        return false;
    }

    const float sanitizedRadius = std::max(radiusSteps, 0.0f);
    const float stepNorm = stepNormalization();
    const float stepX = std::abs(_gridScale[0]);
    const float stepY = std::abs(_gridScale[1]);

    const float radiusWorld = std::max(stepNorm * 0.5f, sanitizedRadius * stepNorm);
    const float radiusWorldSq = radiusWorld * radiusWorld;

    const int gridExtent = std::max(1, static_cast<int>(std::ceil(std::max(sanitizedRadius, 0.5f)))) + 1;
    const int rowStart = std::max(0, centerRow - gridExtent);
    const int rowEnd = std::min(rows - 1, centerRow + gridExtent);
    const int colStart = std::max(0, centerCol - gridExtent);
    const int colEnd = std::min(cols - 1, centerCol + gridExtent);

    const cv::Vec3f invalid(-1.0f, -1.0f, -1.0f);

    bool changed = false;
    std::vector<GridKey> touched;
    touched.reserve(static_cast<std::size_t>((rowEnd - rowStart + 1) * (colEnd - colStart + 1)));

    for (int r = rowStart; r <= rowEnd; ++r) {
        const float dy = static_cast<float>(r - centerRow) * stepY;
        for (int c = colStart; c <= colEnd; ++c) {
            const float dx = static_cast<float>(c - centerCol) * stepX;
            if ((dx * dx + dy * dy) > radiusWorldSq) {
                continue;
            }

            cv::Vec3f& preview = (*_previewPoints)(r, c);
            if (isInvalidPoint(preview)) {
                continue;
            }

            preview = invalid;
            recordVertexEdit(r, c, invalid);
            touched.push_back(GridKey{r, c});
            changed = true;
        }
    }

    if (changed) {
        _recentTouched = std::move(touched);
    } else {
        _recentTouched.clear();
    }

    return changed;
}

void SegmentationEditManager::clearInvalidatedEdits()
{
    if (_editedVertices.empty()) {
        return;
    }

    bool removed = false;
    for (auto it = _editedVertices.begin(); it != _editedVertices.end();) {
        if (isInvalidPoint(it->second.currentWorld)) {
            it = _editedVertices.erase(it);
            removed = true;
        } else {
            ++it;
        }
    }

    if (removed) {
        _recentTouched.clear();
    }

    _dirty = !_editedVertices.empty();
}

bool SegmentationEditManager::isInvalidPoint(const cv::Vec3f& value)
{
    return ::isInvalidPoint(value);
}

void SegmentationEditManager::rebuildPreviewFromOriginal()
{
    if (!_originalPoints || !_previewPoints) {
        return;
    }

    _originalPoints->copyTo(*_previewPoints);

    for (const auto& [key, edit] : _editedVertices) {
        if (key.row < 0 || key.col < 0 ||
            key.row >= _previewPoints->rows || key.col >= _previewPoints->cols) {
            continue;
        }
        (*_previewPoints)(key.row, key.col) = edit.currentWorld;
    }
}

void SegmentationEditManager::ensurePreviewAvailable()
{
    if (_previewPoints) {
        return;
    }
    if (!_originalPoints) {
        return;
    }
    auto* previewMatrix = new cv::Mat_<cv::Vec3f>(_originalPoints->clone());
    _previewSurface = std::make_unique<QuadSurface>(previewMatrix, _gridScale);
    _previewPoints = _previewSurface->rawPointsPtr();
}

bool SegmentationEditManager::buildActiveSamples(const std::pair<int, int>& gridIndex)
{
    if (!_previewPoints || !_originalPoints) {
        return false;
    }

    const int rows = _previewPoints->rows;
    const int cols = _previewPoints->cols;
    const int centerRow = gridIndex.first;
    const int centerCol = gridIndex.second;

    if (centerRow < 0 || centerRow >= rows || centerCol < 0 || centerCol >= cols) {
        return false;
    }

    const cv::Vec3f& centerWorld = (*_previewPoints)(centerRow, centerCol);
    if (isInvalidPoint(centerWorld)) {
        return false;
    }

    const float stepNorm = stepNormalization();
    const float maxRadiusWorld = std::max(0.0f, _radiusSteps) * stepNorm;
    if (maxRadiusWorld <= 0.0f) {
        return false;
    }
    const float maxRadiusWorldSq = maxRadiusWorld * maxRadiusWorld;

    const int gridExtent = std::max(1, static_cast<int>(std::ceil(_radiusSteps))) + 1;
    const int rowStart = std::max(0, centerRow - gridExtent);
    const int rowEnd = std::min(rows - 1, centerRow + gridExtent);
    const int colStart = std::max(0, centerCol - gridExtent);
    const int colEnd = std::min(cols - 1, centerCol + gridExtent);

    _activeDrag.samples.clear();
    _activeDrag.samples.reserve(static_cast<size_t>((rowEnd - rowStart + 1) * (colEnd - colStart + 1)));

    const float stepX = _gridScale[0];
    const float stepY = _gridScale[1];

    for (int r = rowStart; r <= rowEnd; ++r) {
        for (int c = colStart; c <= colEnd; ++c) {
            const cv::Vec3f& original = (*_originalPoints)(r, c);
            const cv::Vec3f& baseWorld = (*_previewPoints)(r, c);
            if (isInvalidPoint(original) || isInvalidPoint(baseWorld)) {
                continue;
            }

            const float dx = static_cast<float>(c - centerCol) * stepX;
            const float dy = static_cast<float>(r - centerRow) * stepY;
            const float distanceWorldSq = dx * dx + dy * dy;
            if (distanceWorldSq > maxRadiusWorldSq) {
                continue;
            }

            _activeDrag.samples.push_back({r, c, baseWorld, distanceWorldSq});
        }
    }

    if (_activeDrag.samples.empty()) {
        return false;
    }

    return true;
}

void SegmentationEditManager::applyGaussianToSamples(const cv::Vec3f& delta)
{
    if (!_previewPoints || !_originalPoints) {
        return;
    }
    if (_activeDrag.samples.empty()) {
        return;
    }

    const float stepNorm = stepNormalization();
    const float sigmaWorld = std::max(0.001f, _sigmaSteps * stepNorm);
    const float invTwoSigmaSq = 1.0f / (2.0f * sigmaWorld * sigmaWorld);

    _recentTouched.clear();
    _recentTouched.reserve(_activeDrag.samples.size());

    for (const auto& sample : _activeDrag.samples) {
        float weight = 1.0f;
        if (sample.distanceWorldSq > 0.0f) {
            weight = std::exp(-sample.distanceWorldSq * invTwoSigmaSq);
        }

        cv::Vec3f newWorld = sample.baseWorld + delta * weight;
        (*_previewPoints)(sample.row, sample.col) = newWorld;
        recordVertexEdit(sample.row, sample.col, newWorld);
        _recentTouched.push_back(GridKey{sample.row, sample.col});
    }

    _dirty = true;
    if (_pendingGrowthMarking) {
        _pendingGrowthMarking = false;
    }
}

void SegmentationEditManager::recordVertexEdit(int row, int col, const cv::Vec3f& newWorld)
{
    if (!_originalPoints) {
        return;
    }
    if (row < 0 || row >= _originalPoints->rows || col < 0 || col >= _originalPoints->cols) {
        return;
    }

    const cv::Vec3f& original = (*_originalPoints)(row, col);
    if (isInvalidPoint(original)) {
        return;
    }

    GridKey key{row, col};
    const float delta = static_cast<float>(cv::norm(newWorld - original));
    if (delta < 1e-4f) {
        _editedVertices.erase(key);
        return;
    }

    auto [it, inserted] = _editedVertices.try_emplace(key);
    if (inserted) {
        it->second = VertexEdit{row, col, original, newWorld, _pendingGrowthMarking};
    } else {
        it->second.currentWorld = newWorld;
        if (_pendingGrowthMarking) {
            it->second.isGrowth = true;
        }
    }

    _dirty = true;
}

void SegmentationEditManager::clearActiveDrag()
{
    _activeDrag.active = false;
    _activeDrag.center = GridKey{};
    _activeDrag.baseWorld = cv::Vec3f(0.0f, 0.0f, 0.0f);
    _activeDrag.targetWorld = cv::Vec3f(0.0f, 0.0f, 0.0f);
    _activeDrag.samples.clear();
}

float SegmentationEditManager::stepNormalization() const
{
    const float sx = std::abs(_gridScale[0]);
    const float sy = std::abs(_gridScale[1]);
    const float avg = 0.5f * (sx + sy);
    return (avg > 1e-4f) ? avg : 1.0f;
}
