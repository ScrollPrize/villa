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
constexpr QColor kBaseFillColor(0, 200, 255, 190);
constexpr QColor kHoverFillColor(255, 255, 255, 225);
constexpr QColor kActiveFillColor(255, 215, 0, 235);
constexpr QColor kKeyboardFillColor(0, 255, 160, 225);
constexpr qreal kBaseRadius = 5.0;
constexpr qreal kHoverRadiusMultiplier = 1.35;
constexpr qreal kActiveRadiusMultiplier = 1.55;
constexpr qreal kKeyboardRadiusMultiplier = 1.45;
constexpr qreal kBasePenWidth = 1.5;
constexpr qreal kHoverPenWidth = 2.5;
constexpr qreal kActivePenWidth = 2.6;
constexpr qreal kKeyboardPenWidth = 2.4;
constexpr QColor kBaseBorderColor(255, 255, 255, 200);
constexpr QColor kHoverBorderColor(Qt::yellow);
constexpr QColor kActiveBorderColor(255, 180, 0, 255);
constexpr QColor kKeyboardBorderColor(60, 255, 180, 240);
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
    if (std::fabs(radius - _radius) < 1e-4f) {
        return;
    }
    _radius = radius;
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

    const float planeTolerance = std::max(1.0f, _radius);

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
        const auto& handles = _editManager->handles();
        for (const auto& handle : handles) {
            const bool isActive = matches(handle.row, handle.col, _activeHandle);
            const bool isHover = matches(handle.row, handle.col, _hoverHandle);
            const bool isKeyboard = matches(handle.row, handle.col, keyboardHighlight);
            if (planeSurface) {
                float dist = planeSurface->pointDist(handle.currentWorld);
                if (std::fabs(dist) > planeTolerance && !isActive && !isHover && !isKeyboard) {
                    continue;
                }
            }

            QPointF scenePt = viewer->volumePointToScene(handle.currentWorld);
            auto [radius, style] = pointVisuals(isActive, isHover, isKeyboard);
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

    for (int y = 0; y < rows; y += step) {
        for (int x = 0; x < cols; x += step) {
            const cv::Vec3f& wp = points(y, x);
            if (wp[0] == -1.0f && wp[1] == -1.0f && wp[2] == -1.0f) {
                continue;
            }

            const bool isActive = matches(y, x, _activeHandle);
            const bool isHover = matches(y, x, _hoverHandle);
            const bool isKeyboard = matches(y, x, keyboardHighlight);
            if (planeSurface) {
                float dist = planeSurface->pointDist(wp);
                if (std::fabs(dist) > planeTolerance && !isActive && !isHover && !isKeyboard) {
                    continue;
                }
            }

            QPointF scenePt = viewer->volumePointToScene(wp);
            auto [radius, style] = pointVisuals(isActive, isHover, isKeyboard);
            builder.addPoint(scenePt, radius, style);
        }
    }
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
