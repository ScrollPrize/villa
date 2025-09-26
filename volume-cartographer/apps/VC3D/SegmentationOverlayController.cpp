#include "SegmentationOverlayController.hpp"

#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "SegmentationEditManager.hpp"

#include <QColor>
#include <QBrush>
#include <QGraphicsEllipseItem>
#include <QGraphicsScene>
#include <QPen>

#include <algorithm>
#include <cmath>
#include <optional>

#include "vc/core/util/Surface.hpp"

namespace
{
constexpr const char* kOverlayGroup = "segmentation_edit_points";
constexpr QColor kPointFillColor(0, 200, 255, 180);
constexpr float kBasePixelRadius = 4.0f;
constexpr QColor kHoverColor(255, 255, 255, 220);
constexpr QColor kActiveColor(255, 215, 0, 230);
}

SegmentationOverlayController::SegmentationOverlayController(CSurfaceCollection* surfCollection, QObject* parent)
    : QObject(parent)
    , _surfCollection(surfCollection)
{
    if (_surfCollection) {
        connect(_surfCollection, &CSurfaceCollection::sendSurfaceChanged,
                this, &SegmentationOverlayController::onSurfaceChanged);
    }
}

void SegmentationOverlayController::attachViewer(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    auto exists = std::find_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerState& state) {
        return state.viewer == viewer;
    });
    if (exists != _viewers.end()) {
        return;
    }

    ViewerState state;
    state.viewer = viewer;
    state.overlayUpdateConn = connect(viewer, &CVolumeViewer::overlaysUpdated,
                                      this, [this, viewer]() { rebuildViewerOverlay(viewer); });
    state.destroyedConn = connect(viewer, &QObject::destroyed,
                                  this, [this, viewer]() { detachViewer(viewer); });
    _viewers.push_back(state);

    rebuildViewerOverlay(viewer);
}

void SegmentationOverlayController::detachViewer(CVolumeViewer* viewer)
{
    auto it = std::remove_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerState& state) {
        return state.viewer == viewer;
    });
    for (auto iter = it; iter != _viewers.end(); ++iter) {
        QObject::disconnect(iter->overlayUpdateConn);
        QObject::disconnect(iter->destroyedConn);
    }
    _viewers.erase(it, _viewers.end());
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

void SegmentationOverlayController::refreshAll()
{
    for (auto& state : _viewers) {
        rebuildViewerOverlay(state.viewer);
    }
}

void SegmentationOverlayController::refreshViewer(CVolumeViewer* viewer)
{
    rebuildViewerOverlay(viewer);
}

void SegmentationOverlayController::onSurfaceChanged(std::string name, Surface* /*surf*/)
{
    if (name == "segmentation") {
        refreshAll();
    }
}

void SegmentationOverlayController::rebuildViewerOverlay(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    if (!_editingEnabled) {
        viewer->clearOverlayGroup(kOverlayGroup);
        return;
    }

    if (!_surfCollection) {
        viewer->clearOverlayGroup(kOverlayGroup);
        return;
    }

    auto* scene = viewer->fGraphicsView ? viewer->fGraphicsView->scene() : nullptr;
    if (!scene) {
        viewer->clearOverlayGroup(kOverlayGroup);
        return;
    }

    std::vector<QGraphicsItem*> items;
    const float pixelRadius = kBasePixelRadius;
    const float planeTolerance = std::max(1.0f, _radius);

    auto* currentSurface = viewer->currentSurface();
    auto* planeSurface = dynamic_cast<PlaneSurface*>(currentSurface);

    auto matches = [](int row, int col, const std::optional<std::pair<int,int>>& opt) {
        return opt && opt->first == row && opt->second == col;
    };

    std::optional<std::pair<int,int>> keyboardHighlight;
    if (_editManager && _editManager->hasSession()) {
        keyboardHighlight = _keyboardHandle;
        const auto& handles = _editManager->handles();
        items.reserve(handles.size());
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
            float radius = pixelRadius;
            QColor color = kPointFillColor;
            if (isActive) {
                radius *= 1.8f;
                color = kActiveColor;
            } else if (isHover) {
                radius *= 1.4f;
                color = kHoverColor;
            } else if (isKeyboard) {
                radius *= 1.6f;
                color = QColor(0, 255, 127, 220);
            }
            auto* item = new QGraphicsEllipseItem(scenePt.x() - radius,
                                                  scenePt.y() - radius,
                                                  radius * 2.0f,
                                                  radius * 2.0f);
            item->setBrush(QBrush(color));
            item->setPen(QPen(Qt::black, 0.5f));
            item->setZValue(95.0f);
            item->setData(0, handle.row);
            item->setData(1, handle.col);
            scene->addItem(item);
            items.push_back(item);
        }
    } else {
        auto* surface = dynamic_cast<QuadSurface*>(_surfCollection->surface("segmentation"));
        if (!surface) {
            viewer->clearOverlayGroup(kOverlayGroup);
            return;
        }

        auto* pointsPtr = surface->rawPointsPtr();
        if (!pointsPtr || pointsPtr->empty()) {
            viewer->clearOverlayGroup(kOverlayGroup);
            return;
        }

        const cv::Mat_<cv::Vec3f>& points = *pointsPtr;
        const int rows = points.rows;
        const int cols = points.cols;
        const int step = std::max(1, _downsample);
        items.reserve((rows / step + 1) * (cols / step + 1));

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
                float radius = pixelRadius;
                QColor color = kPointFillColor;
                if (isActive) {
                    radius *= 1.8f;
                    color = kActiveColor;
                } else if (isHover) {
                    radius *= 1.4f;
                    color = kHoverColor;
                } else if (isKeyboard) {
                    radius *= 1.6f;
                    color = QColor(0, 255, 127, 220);
                }

                auto* item = new QGraphicsEllipseItem(scenePt.x() - radius,
                                                      scenePt.y() - radius,
                                                      radius * 2.0f,
                                                      radius * 2.0f);
                item->setBrush(QBrush(color));
                item->setPen(QPen(Qt::black, 0.5f));
                item->setZValue(95.0f);
                item->setData(0, y);
                item->setData(1, x);
                scene->addItem(item);
                items.push_back(item);
            }
        }
    }

    viewer->setOverlayGroup(kOverlayGroup, items);
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
