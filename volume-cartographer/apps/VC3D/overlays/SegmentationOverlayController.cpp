#include "SegmentationOverlayController.hpp"

#include "../CSurfaceCollection.hpp"
#include "../CVolumeViewer.hpp"
#include "../SegmentationEditManager.hpp"

#include <QColor>

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include "vc/core/util/Surface.hpp"

namespace
{
constexpr const char* kOverlayGroup = "segmentation_edit_points";
const QColor kBaseFillColor = QColor(0, 200, 255, 190);
const QColor kHoverFillColor = QColor(255, 255, 255, 225);
const QColor kActiveFillColor = QColor(255, 215, 0, 235);
const QColor kKeyboardFillColor = QColor(0, 255, 160, 225);
constexpr qreal kBaseRadius = 5.0;
constexpr qreal kHoverRadiusMultiplier = 1.35;
constexpr qreal kActiveRadiusMultiplier = 1.55;
constexpr qreal kKeyboardRadiusMultiplier = 1.45;
constexpr qreal kBasePenWidth = 1.5;
constexpr qreal kHoverPenWidth = 2.5;
constexpr qreal kActivePenWidth = 2.6;
constexpr qreal kKeyboardPenWidth = 2.4;
const QColor kBaseBorderColor = QColor(255, 255, 255, 200);
const QColor kHoverBorderColor = QColor(Qt::yellow);
const QColor kActiveBorderColor = QColor(255, 180, 0, 255);
const QColor kKeyboardBorderColor = QColor(60, 255, 180, 240);
constexpr qreal kPointZ = 95.0;
}

SegmentationOverlayController::SegmentationOverlayController(CSurfaceCollection* surfCollection, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
    , _surfCollection(surfCollection)
{
    if (_surfCollection) {
        connect(_surfCollection, &CSurfaceCollection::sendSurfaceChanged,
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

void SegmentationOverlayController::setDownsample(int value)
{
    int clamped = std::max(1, value);
    if (_downsample == clamped) {
        return;
    }
    _downsample = clamped;
    refreshAll();
}

void SegmentationOverlayController::setRadius(float radius)
{
    const int snapped = std::max(1, static_cast<int>(std::lround(radius)));
    const float snappedRadius = static_cast<float>(snapped);
    if (std::fabs(snappedRadius - _radius) < 1e-4f) {
        return;
    }
    _radius = snappedRadius;
    refreshAll();
}

void SegmentationOverlayController::onSurfaceChanged(std::string name, Surface* /*surf*/)
{
    if (name == "segmentation") {
        refreshAll();
    }
}

bool SegmentationOverlayController::isOverlayEnabledFor(CVolumeViewer* /*viewer*/) const
{
    return _editingEnabled;
}

void SegmentationOverlayController::collectPrimitives(CVolumeViewer* viewer,
                                                      ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer || !_surfCollection) {
        return;
    }

    const float planeTolerance = planeToleranceWorld();

    auto* currentSurface = viewer->currentSurface();
    auto* planeSurface = dynamic_cast<PlaneSurface*>(currentSurface);

    auto matches = [](int row, int col, const std::optional<std::pair<int,int>>& opt) {
        return opt && opt->first == row && opt->second == col;
    };

    auto pointVisuals = [](bool isActive, bool isHover, bool isKeyboard) {
        qreal radius = kBaseRadius;
        qreal penWidth = kBasePenWidth;
        QColor border = kBaseBorderColor;
        QColor fill = kBaseFillColor;

        if (isActive) {
            radius *= kActiveRadiusMultiplier;
            penWidth = kActivePenWidth;
            border = kActiveBorderColor;
            fill = kActiveFillColor;
        } else if (isHover) {
            radius *= kHoverRadiusMultiplier;
            penWidth = kHoverPenWidth;
            border = kHoverBorderColor;
            fill = kHoverFillColor;
        } else if (isKeyboard) {
            radius *= kKeyboardRadiusMultiplier;
            penWidth = kKeyboardPenWidth;
            border = kKeyboardBorderColor;
            fill = kKeyboardFillColor;
        }

        OverlayStyle style;
        style.penColor = border;
        style.brushColor = fill;
        style.penWidth = penWidth;
        style.z = kPointZ;

        return std::make_pair(radius, style);
    };

    std::optional<std::pair<int,int>> keyboardHighlight;
    if (_editManager && _editManager->hasSession()) {
        keyboardHighlight = _keyboardHandle;

        struct HandleEntry {
            cv::Vec3f world;
            bool isActive;
            bool isHover;
            bool isKeyboard;
        };

        std::vector<cv::Vec3f> handlePoints;
        std::vector<HandleEntry> handleEntries;
        const auto& handles = _editManager->handles();
        handlePoints.reserve(handles.size());
        handleEntries.reserve(handles.size());

        for (const auto& handle : handles) {
            const bool isActive = matches(handle.row, handle.col, _activeHandle);
            const bool isHover = matches(handle.row, handle.col, _hoverHandle);
            const bool isKeyboard = matches(handle.row, handle.col, keyboardHighlight);

            handlePoints.push_back(handle.currentWorld);
            handleEntries.push_back({handle.currentWorld, isActive, isHover, isKeyboard});
        }

        PointFilterOptions filter;
        filter.clipToSurface = false;
        filter.computeScenePoints = true;
        filter.volumePredicate = [planeSurface, planeTolerance, &handleEntries](const cv::Vec3f&, size_t index) {
            const auto& entry = handleEntries[index];
            if (planeSurface) {
                float dist = planeSurface->pointDist(entry.world);
                if (std::fabs(dist) > planeTolerance && !(entry.isActive || entry.isHover || entry.isKeyboard)) {
                    return false;
                }
            }
            return true;
        };

        auto filtered = filterPoints(viewer, handlePoints, filter);
        for (size_t i = 0; i < filtered.volumePoints.size(); ++i) {
            size_t srcIndex = filtered.sourceIndices.empty() ? i : filtered.sourceIndices[i];
            const auto& entry = handleEntries[srcIndex];
            const QPointF& scenePt = filtered.scenePoints[i];
            auto [radius, style] = pointVisuals(entry.isActive, entry.isHover, entry.isKeyboard);
            builder.addPoint(scenePt, radius, style);
        }
        return;
    }

    auto* surface = dynamic_cast<QuadSurface*>(_surfCollection->surface("segmentation"));
    if (!surface) {
        return;
    }

    auto* pointsPtr = surface->rawPointsPtr();
    if (!pointsPtr || pointsPtr->empty()) {
        return;
    }

    const cv::Mat_<cv::Vec3f>& points = *pointsPtr;
    const int rows = points.rows;
    const int cols = points.cols;
    const int step = std::max(1, _downsample);

    struct GridEntry {
        cv::Vec3f world;
        bool isActive;
        bool isHover;
        bool isKeyboard;
        bool valid;
    };

    std::vector<cv::Vec3f> gridPoints;
    std::vector<GridEntry> gridEntries;
    gridPoints.reserve((rows / step + 1) * (cols / step + 1));
    gridEntries.reserve(gridPoints.capacity());

    for (int y = 0; y < rows; y += step) {
        for (int x = 0; x < cols; x += step) {
            const cv::Vec3f& wp = points(y, x);
            bool valid = !(wp[0] == -1.0f && wp[1] == -1.0f && wp[2] == -1.0f);
            const bool isActive = matches(y, x, _activeHandle);
            const bool isHover = matches(y, x, _hoverHandle);
            const bool isKeyboard = matches(y, x, keyboardHighlight);

            gridPoints.push_back(wp);
            gridEntries.push_back({wp, isActive, isHover, isKeyboard, valid});
        }
    }

    PointFilterOptions filter;
    filter.clipToSurface = false;
    filter.computeScenePoints = true;
    filter.volumePredicate = [planeSurface, planeTolerance, &gridEntries](const cv::Vec3f&, size_t index) {
        const auto& entry = gridEntries[index];
        if (!entry.valid) {
            return false;
        }
        if (planeSurface) {
            float dist = planeSurface->pointDist(entry.world);
            if (std::fabs(dist) > planeTolerance && !(entry.isActive || entry.isHover || entry.isKeyboard)) {
                return false;
            }
        }
        return true;
    };

    auto filtered = filterPoints(viewer, gridPoints, filter);
    for (size_t i = 0; i < filtered.volumePoints.size(); ++i) {
        size_t srcIndex = filtered.sourceIndices.empty() ? i : filtered.sourceIndices[i];
        const auto& entry = gridEntries[srcIndex];
        const QPointF& scenePt = filtered.scenePoints[i];
        auto [radius, style] = pointVisuals(entry.isActive, entry.isHover, entry.isKeyboard);
        builder.addPoint(scenePt, radius, style);
    }
}

float SegmentationOverlayController::gridStepWorld() const
{
    if (_editManager && _editManager->baseSurface()) {
        const cv::Vec2f scale = _editManager->baseSurface()->scale();
        const float sx = std::fabs(scale[0]);
        const float sy = std::fabs(scale[1]);
        const float step = std::max(sx, sy);
        if (std::isfinite(step) && step > 1e-4f) {
            return step;
        }
    }

    if (_surfCollection) {
        if (auto* surface = dynamic_cast<QuadSurface*>(_surfCollection->surface("segmentation"))) {
            const cv::Vec2f scale = surface->scale();
            const float sx = std::fabs(scale[0]);
            const float sy = std::fabs(scale[1]);
            const float step = std::max(sx, sy);
            if (std::isfinite(step) && step > 1e-4f) {
                return step;
            }
        }
    }

    return 1.0f;
}

float SegmentationOverlayController::planeToleranceWorld() const
{
    const float step = gridStepWorld();
    const float cells = std::max(_radius, 1.0f);
    const float baseExtent = (cells + 0.5f) * step;
    const float minExtent = std::max(step, 3.0f);
    return std::max(baseExtent, minExtent);
}

void SegmentationOverlayController::setActiveHandle(std::optional<std::pair<int,int>> key, bool refresh)
{
    if (_activeHandle == key) {
        return;
    }
    _activeHandle = key;
    if (refresh) {
        refreshAll();
    }
}

void SegmentationOverlayController::setHoverHandle(std::optional<std::pair<int,int>> key, bool refresh)
{
    if (_hoverHandle == key) {
        return;
    }
    _hoverHandle = key;
    if (refresh) {
        refreshAll();
    }
}

void SegmentationOverlayController::setKeyboardHandle(std::optional<std::pair<int,int>> key, bool refresh)
{
    if (_keyboardHandle == key) {
        return;
    }
    _keyboardHandle = key;
    if (refresh) {
        refreshAll();
    }
}
