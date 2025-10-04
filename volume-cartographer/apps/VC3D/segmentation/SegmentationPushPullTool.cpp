#include "SegmentationPushPullTool.hpp"

#include "SegmentationModule.hpp"
#include "CVolumeViewer.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "CSurfaceCollection.hpp"

#include <QCoreApplication>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace
{
constexpr int kPushPullIntervalMs = 30;
}

SegmentationPushPullTool::SegmentationPushPullTool(SegmentationModule& module,
                                                   SegmentationEditManager* editManager,
                                                   SegmentationWidget* widget,
                                                   SegmentationOverlayController* overlay,
                                                   CSurfaceCollection* surfaces)
    : _module(module)
    , _editManager(editManager)
    , _widget(widget)
    , _overlay(overlay)
    , _surfaces(surfaces)
{
    ensureTimer();
}

void SegmentationPushPullTool::setDependencies(SegmentationEditManager* editManager,
                                               SegmentationWidget* widget,
                                               SegmentationOverlayController* overlay,
                                               CSurfaceCollection* surfaces)
{
    _editManager = editManager;
    _widget = widget;
    _overlay = overlay;
    _surfaces = surfaces;
}

void SegmentationPushPullTool::setStepMultiplier(float multiplier)
{
    _stepMultiplier = std::clamp(multiplier, 0.05f, 10.0f);
}

void SegmentationPushPullTool::setAlphaEnabled(bool enabled)
{
    _alphaEnabled = enabled;
}

void SegmentationPushPullTool::setAlphaConfig(const AlphaPushPullConfig& config)
{
    _alphaConfig = config;
}

bool SegmentationPushPullTool::start(int direction)
{
    if (direction == 0) {
        return false;
    }

    ensureTimer();

    if (_state.active && _state.direction == direction) {
        if (_timer && !_timer->isActive()) {
            _timer->start();
        }
        return true;
    }

    if (!_module._hover.valid || !_module._hover.viewer || !_module.isSegmentationViewer(_module._hover.viewer)) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    _state.active = true;
    _state.direction = direction;
    _undoCaptured = false;
    _module.useFalloff(SegmentationModule::FalloffTool::PushPull);

    if (_timer && !_timer->isActive()) {
        _timer->start();
    }

    if (!applyStepInternal()) {
        stopAll();
        return false;
    }

    return true;
}

void SegmentationPushPullTool::stop(int direction)
{
    if (!_state.active) {
        return;
    }
    if (direction != 0 && direction != _state.direction) {
        return;
    }
    stopAll();
}

void SegmentationPushPullTool::stopAll()
{
    _state.active = false;
    _state.direction = 0;
    if (_timer && _timer->isActive()) {
        _timer->stop();
    }
    _undoCaptured = false;
    if (_module._activeFalloff == SegmentationModule::FalloffTool::PushPull) {
        _module.useFalloff(SegmentationModule::FalloffTool::Drag);
    }
}

bool SegmentationPushPullTool::applyStep()
{
    return applyStepInternal();
}

bool SegmentationPushPullTool::applyStepInternal()
{
    if (!_state.active || !_editManager || !_editManager->hasSession()) {
        return false;
    }

    if (!_module._hover.valid || !_module._hover.viewer || !_module.isSegmentationViewer(_module._hover.viewer)) {
        return false;
    }

    const int row = _module._hover.row;
    const int col = _module._hover.col;

    bool snapshotCapturedThisStep = false;
    if (!_undoCaptured) {
        snapshotCapturedThisStep = _module.captureUndoSnapshot();
        if (snapshotCapturedThisStep) {
            _undoCaptured = true;
        }
    }

    if (!_editManager->beginActiveDrag({row, col})) {
        if (snapshotCapturedThisStep) {
            _module.discardLastUndoSnapshot();
            _undoCaptured = false;
        }
        return false;
    }

    auto centerWorldOpt = _editManager->vertexWorldPosition(row, col);
    if (!centerWorldOpt) {
        _editManager->cancelActiveDrag();
        if (snapshotCapturedThisStep) {
            _module.discardLastUndoSnapshot();
            _undoCaptured = false;
        }
        return false;
    }
    const cv::Vec3f centerWorld = *centerWorldOpt;

    QuadSurface* baseSurface = _editManager->baseSurface();
    if (!baseSurface) {
        _editManager->cancelActiveDrag();
        return false;
    }

    cv::Vec3f ptr = baseSurface->pointer();
    baseSurface->pointTo(ptr, centerWorld, std::numeric_limits<float>::max(), 400);
    cv::Vec3f normal = baseSurface->normal(ptr);
    if (std::isnan(normal[0]) || std::isnan(normal[1]) || std::isnan(normal[2])) {
        _editManager->cancelActiveDrag();
        return false;
    }

    const float norm = cv::norm(normal);
    if (norm <= 1e-4f) {
        _editManager->cancelActiveDrag();
        return false;
    }
    normal /= norm;

    cv::Vec3f targetWorld = centerWorld;
    bool usedAlphaPushPull = false;

    if (_alphaEnabled && _alphaConfig.perVertex) {
        std::vector<cv::Vec3f> samples;
        auto perVertex = _editManager->recentTouched();
        samples.reserve(perVertex.size());

        for (const auto& key : perVertex) {
            if (auto vertex = _editManager->vertexWorldPosition(key.row, key.col)) {
                samples.push_back(*vertex);
            }
        }

        if (!samples.empty()) {
            bool alphaUnavailable = false;

            std::vector<cv::Vec3f> perVertexTargets;
            perVertexTargets.reserve(samples.size());
            std::vector<float> perVertexMovements;
            perVertexMovements.reserve(samples.size());
            bool anyMovement = false;
            float minMovement = std::numeric_limits<float>::max();

            for (const auto& baseWorld : samples) {
                cv::Vec3f sampleNormal = normal;
                cv::Vec3f samplePtr = baseSurface->pointer();
                baseSurface->pointTo(samplePtr, baseWorld, std::numeric_limits<float>::max(), 400);
                cv::Vec3f candidateNormal = baseSurface->normal(samplePtr);
                if (std::isfinite(candidateNormal[0]) &&
                    std::isfinite(candidateNormal[1]) &&
                    std::isfinite(candidateNormal[2])) {
                    const float candidateNorm = cv::norm(candidateNormal);
                    if (candidateNorm > 1e-4f) {
                        sampleNormal = candidateNormal / candidateNorm;
                    }
                }

                bool sampleUnavailable = false;
                auto sampleTarget = _module.computeAlphaPushPullTarget(baseWorld,
                                                                       sampleNormal,
                                                                       _state.direction,
                                                                       baseSurface,
                                                                       _module._hover.viewer,
                                                                       &sampleUnavailable);
                if (sampleUnavailable) {
                    alphaUnavailable = true;
                    break;
                }

                cv::Vec3f newWorld = baseWorld;
                float movement = 0.0f;
                if (sampleTarget) {
                    newWorld = *sampleTarget;
                    const cv::Vec3f delta = newWorld - baseWorld;
                    movement = static_cast<float>(cv::norm(delta));
                    if (movement >= 1e-4f) {
                        anyMovement = true;
                    }
                }

                perVertexTargets.push_back(newWorld);
                perVertexMovements.push_back(movement);
                minMovement = std::min(minMovement, movement);
            }

            const float perVertexLimit = std::max(0.0f, _alphaConfig.perVertexLimit);
            if (perVertexLimit > 0.0f && !perVertexTargets.empty() && std::isfinite(minMovement)) {
                const float maxAllowedMovement = minMovement + perVertexLimit;
                for (std::size_t i = 0; i < perVertexTargets.size(); ++i) {
                    if (perVertexMovements[i] > maxAllowedMovement + 1e-4f) {
                        const cv::Vec3f& baseWorld = samples[i];
                        const cv::Vec3f delta = perVertexTargets[i] - baseWorld;
                        const float length = perVertexMovements[i];
                        if (length > 1e-6f) {
                            const float scale = maxAllowedMovement / length;
                            perVertexTargets[i] = baseWorld + delta * scale;
                            perVertexMovements[i] = maxAllowedMovement;
                            if (maxAllowedMovement >= 1e-4f) {
                                anyMovement = true;
                            }
                        }
                    }
                }
            }

            if (alphaUnavailable) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    _module.discardLastUndoSnapshot();
                    _undoCaptured = false;
                }
                return false;
            }

            if (!anyMovement) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    _module.discardLastUndoSnapshot();
                    _undoCaptured = false;
                }
                return false;
            }

            if (!_editManager->updateActiveDragTargets(perVertexTargets)) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    _module.discardLastUndoSnapshot();
                    _undoCaptured = false;
                }
                return false;
            }

            usedAlphaPushPull = true;
        }
    } else if (_alphaEnabled) {
        bool alphaUnavailable = false;
        auto alphaTarget = _module.computeAlphaPushPullTarget(centerWorld,
                                                             normal,
                                                             _state.direction,
                                                             baseSurface,
                                                             _module._hover.viewer,
                                                             &alphaUnavailable);
        if (alphaTarget) {
            targetWorld = *alphaTarget;
            usedAlphaPushPull = true;
        } else if (!alphaUnavailable) {
            _editManager->cancelActiveDrag();
            if (snapshotCapturedThisStep) {
                _module.discardLastUndoSnapshot();
                _undoCaptured = false;
            }
            return false;
        }
    }

    if (!usedAlphaPushPull) {
        const float stepWorld = _module.gridStepWorld() * _stepMultiplier;
        if (stepWorld <= 0.0f) {
            _editManager->cancelActiveDrag();
            return false;
        }
        targetWorld = centerWorld + normal * (static_cast<float>(_state.direction) * stepWorld);
    }

    if (!usedAlphaPushPull && !_editManager->updateActiveDrag(targetWorld)) {
        _editManager->cancelActiveDrag();
        if (snapshotCapturedThisStep) {
            _module.discardLastUndoSnapshot();
            _undoCaptured = false;
        }
        return false;
    }

    _editManager->commitActiveDrag();
    _editManager->applyPreview();

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    _module.refreshOverlay();
    _module.emitPendingChanges();
    return true;
}

void SegmentationPushPullTool::ensureTimer()
{
    if (_timer) {
        return;
    }

    _timer = new QTimer(&_module);
    _timer->setInterval(kPushPullIntervalMs);
    QObject::connect(_timer, &QTimer::timeout, &_module, [this]() {
        if (!applyStepInternal()) {
            stopAll();
        }
    });
}
