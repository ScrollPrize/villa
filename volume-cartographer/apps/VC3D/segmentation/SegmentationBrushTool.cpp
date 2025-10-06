#include "SegmentationBrushTool.hpp"

#include "SegmentationModule.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "CSurfaceCollection.hpp"

#include <QCoreApplication>
#include <QObject>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace
{
constexpr float kBrushSampleSpacing = 2.0f;
}

SegmentationBrushTool::SegmentationBrushTool(SegmentationModule& module,
                                             SegmentationEditManager* editManager,
                                             SegmentationWidget* widget,
                                             CSurfaceCollection* surfaces)
    : _module(module)
    , _editManager(editManager)
    , _widget(widget)
    , _surfaces(surfaces)
{
}

void SegmentationBrushTool::setDependencies(SegmentationEditManager* editManager,
                                            SegmentationWidget* widget,
                                            CSurfaceCollection* surfaces)
{
    _editManager = editManager;
    _widget = widget;
    _surfaces = surfaces;
}

void SegmentationBrushTool::setActive(bool active)
{
    if (_brushActive == active) {
        if (_widget) {
            _widget->setEraseBrushActive(_brushActive);
        }
        return;
    }

    _brushActive = active;
    if (!_brushActive) {
        _hasLastSample = false;
    } else {
        ensureFalloff();
    }

    if (_widget) {
        _widget->setEraseBrushActive(_brushActive);
    }

    _module.refreshOverlay();
}

void SegmentationBrushTool::startStroke(const cv::Vec3f& worldPos)
{
    ensureFalloff();
    _strokeActive = true;
    _currentStroke.clear();
    _currentStroke.push_back(worldPos);
    _overlayPoints.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;
    _module.refreshOverlay();
}

void SegmentationBrushTool::extendStroke(const cv::Vec3f& worldPos, bool forceSample)
{
    if (!_strokeActive) {
        return;
    }

    const float spacing = kBrushSampleSpacing;
    const float spacingSq = spacing * spacing;

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
                _overlayPoints.push_back(intermediate);
                travelled += spacing;
            }
        }
    }

    _currentStroke.push_back(worldPos);
    _overlayPoints.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;
    _module.refreshOverlay();
}

void SegmentationBrushTool::finishStroke()
{
    if (!_strokeActive) {
        return;
    }

    _strokeActive = false;
    if (!_currentStroke.empty()) {
        _pendingStrokes.push_back(_currentStroke);
    }
    _currentStroke.clear();
    _hasLastSample = false;
    _module.refreshOverlay();
}

bool SegmentationBrushTool::applyPending(float dragRadiusSteps)
{
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    if (_strokeActive) {
        finishStroke();
    }

    if (_pendingStrokes.empty()) {
        return false;
    }

    using GridKey = SegmentationEditManager::GridKey;
    using GridKeyHash = SegmentationEditManager::GridKeyHash;

    std::unordered_set<GridKey, GridKeyHash> targets;
    std::size_t estimate = 0;
    for (const auto& stroke : _pendingStrokes) {
        estimate += stroke.size();
    }
    targets.reserve(estimate);

    const float stepWorld = _module.gridStepWorld();
    const float brushRadius = std::max(dragRadiusSteps, 0.5f);
    const float maxDistance = stepWorld * std::max(brushRadius * 6.0f, 15.0f);

    for (const auto& stroke : _pendingStrokes) {
        for (const auto& world : stroke) {
            float gridDistance = 0.0f;
            auto grid = _editManager->worldToGridIndex(world, &gridDistance);
            if (!grid) {
                continue;
            }
            if (maxDistance > 0.0f && gridDistance > maxDistance) {
                continue;
            }
            targets.insert(GridKey{grid->first, grid->second});
        }
    }

    if (targets.empty()) {
        clear();
        return false;
    }

    bool snapshotCaptured = _module.captureUndoSnapshot();
    bool anyChanged = false;
    for (const auto& key : targets) {
        if (_editManager->markInvalidRegion(key.row, key.col, brushRadius)) {
            anyChanged = true;
        }
    }

    clear();

    if (!anyChanged) {
        if (snapshotCaptured) {
            _module.discardLastUndoSnapshot();
        }
        return false;
    }

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, false);
    }

    _module.emitPendingChanges();
    Q_EMIT _module.statusMessageRequested(QCoreApplication::translate("SegmentationModule",
                                                                     "Invalidated %1 brush target(s).")
                                              .arg(static_cast<int>(targets.size())),
                                          2000);
    return true;
}

void SegmentationBrushTool::clear()
{
    _strokeActive = false;
    _currentStroke.clear();
    _pendingStrokes.clear();
    _overlayPoints.clear();
    _hasLastSample = false;

    if (!_brushActive && _widget) {
        _widget->setEraseBrushActive(false);
    }

    _module.refreshOverlay();
}

void SegmentationBrushTool::ensureFalloff()
{
    _module.useFalloff(SegmentationModule::FalloffTool::Drag);
}
