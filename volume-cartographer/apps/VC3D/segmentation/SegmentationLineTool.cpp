#include "SegmentationLineTool.hpp"

#include "SegmentationModule.hpp"
#include "SegmentationEditManager.hpp"
#include "CSurfaceCollection.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include <QCoreApplication>
#include <QLoggingCategory>
#include <QObject>

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "vc/core/util/QuadSurface.hpp"

Q_DECLARE_LOGGING_CATEGORY(lcSegModule)

namespace
{
constexpr float kBrushSampleSpacing = 2.0f;
}

SegmentationLineTool::SegmentationLineTool(SegmentationModule& module,
                                           SegmentationEditManager* editManager,
                                           CSurfaceCollection* surfaces,
                                           float& smoothStrength,
                                           int& smoothIterations)
    : _module(module)
    , _editManager(editManager)
    , _surfaces(surfaces)
    , _smoothStrength(&smoothStrength)
    , _smoothIterations(&smoothIterations)
{
}

void SegmentationLineTool::setDependencies(SegmentationEditManager* editManager,
                                           CSurfaceCollection* surfaces)
{
    _editManager = editManager;
    _surfaces = surfaces;
}

void SegmentationLineTool::setSmoothing(float& smoothStrength, int& smoothIterations)
{
    _smoothStrength = &smoothStrength;
    _smoothIterations = &smoothIterations;
}

void SegmentationLineTool::startStroke(const cv::Vec3f& worldPos)
{
    _module.useFalloff(SegmentationModule::FalloffTool::Line);
    _strokeActive = true;
    _strokePoints.clear();
    _overlayPoints.clear();
    _strokePoints.push_back(worldPos);
    _overlayPoints.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;
    _module.refreshOverlay();
}

void SegmentationLineTool::extendStroke(const cv::Vec3f& worldPos, bool forceSample)
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
                _strokePoints.push_back(intermediate);
                _overlayPoints.push_back(intermediate);
                travelled += spacing;
            }
        }
    }

    _strokePoints.push_back(worldPos);
    _overlayPoints.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;
    _module.refreshOverlay();
}

void SegmentationLineTool::finishStroke(bool keepLineFalloff)
{
    if (!_strokeActive) {
        return;
    }

    _strokeActive = false;
    const std::vector<cv::Vec3f> strokeCopy = _strokePoints;
    applyStroke(strokeCopy);
    clear();

    if (!keepLineFalloff) {
        _module.useFalloff(SegmentationModule::FalloffTool::Drag);
    }
}

bool SegmentationLineTool::applyStroke(const std::vector<cv::Vec3f>& stroke)
{
    _module.useFalloff(SegmentationModule::FalloffTool::Line);
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }
    if (stroke.size() < 2) {
        return false;
    }

    using GridKey = SegmentationEditManager::GridKey;
    using GridKeyHash = SegmentationEditManager::GridKeyHash;

    std::unordered_set<GridKey, GridKeyHash> visited;
    visited.reserve(stroke.size());

    bool anyMoved = false;

    for (const auto& world : stroke) {
        auto gridIndex = _editManager->worldToGridIndex(world);
        if (!gridIndex) {
            continue;
        }

        GridKey key{gridIndex->first, gridIndex->second};
        if (!visited.insert(key).second) {
            continue;
        }

        if (!_editManager->beginActiveDrag(*gridIndex)) {
            continue;
        }

        if (!_editManager->updateActiveDrag(world)) {
            _editManager->cancelActiveDrag();
            continue;
        }

        const float smoothStrength = (_smoothStrength) ? *_smoothStrength : 0.0f;
        const int smoothIterations = (_smoothIterations) ? *_smoothIterations : 0;
        if (smoothStrength > 0.0f && smoothIterations > 0) {
            _editManager->smoothRecentTouched(smoothStrength, smoothIterations);
        }

        _editManager->commitActiveDrag();
        anyMoved = true;
    }

    if (!anyMoved) {
        return false;
    }

    // Capture delta for undo before applyPreview() clears edited vertices
    _module.captureUndoDelta();

    // Auto-approve edited regions before applyPreview() clears them
    auto* overlay = _module.overlay();
    if (overlay && overlay->hasApprovalMaskData()) {
        const auto editedVerts = _editManager->editedVertices();
        if (!editedVerts.empty()) {
            std::vector<std::pair<int, int>> gridPositions;
            gridPositions.reserve(editedVerts.size());
            for (const auto& edit : editedVerts) {
                gridPositions.emplace_back(edit.row, edit.col);
            }
            constexpr uint8_t kApproved = 255;
            constexpr float kRadius = 1.0f;
            constexpr bool kIsAutoApproval = true;
            const QColor brushColor = _module.approvalBrushColor();
            overlay->paintApprovalMaskDirect(gridPositions, kRadius, kApproved, brushColor, false, 0.0f, 0.0f, kIsAutoApproval);
            overlay->scheduleDebouncedSave(_editManager->baseSurface().get());
            qCInfo(lcSegModule) << "Auto-approved" << gridPositions.size() << "line tool edited vertices";
        }
    }

    _editManager->applyPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }

    _module.refreshOverlay();
    _module.emitPendingChanges();
    _module.markAutosaveNeeded();
    Q_EMIT _module.statusMessageRequested(QCoreApplication::translate("SegmentationModule",
                                                                     "Applied segmentation drag along path."),
                                          1500);
    return true;
}

void SegmentationLineTool::clear()
{
    _strokeActive = false;
    _strokePoints.clear();
    _overlayPoints.clear();
    _hasLastSample = false;
    _module.refreshOverlay();
}
