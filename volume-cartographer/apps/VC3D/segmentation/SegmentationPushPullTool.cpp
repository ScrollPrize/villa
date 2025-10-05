#include "SegmentationPushPullTool.hpp"

#include "SegmentationModule.hpp"
#include "CVolumeViewer.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "CSurfaceCollection.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <QCoreApplication>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{
constexpr int kPushPullIntervalMs = 30;
constexpr float kAlphaMinStep = 0.05f;
constexpr float kAlphaMaxStep = 20.0f;
constexpr float kAlphaMinRange = 0.01f;
constexpr float kAlphaDefaultHighDelta = 0.05f;
constexpr float kAlphaBorderLimit = 20.0f;
constexpr int kAlphaBlurRadiusMax = 15;
constexpr float kAlphaPerVertexLimitMax = 128.0f;

bool nearlyEqual(float lhs, float rhs)
{
    return std::fabs(lhs - rhs) < 1e-4f;
}
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

AlphaPushPullConfig SegmentationPushPullTool::sanitizeConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = config;

    sanitized.start = std::clamp(sanitized.start, -128.0f, 128.0f);
    sanitized.stop = std::clamp(sanitized.stop, -128.0f, 128.0f);
    if (sanitized.start > sanitized.stop) {
        std::swap(sanitized.start, sanitized.stop);
    }

    const float magnitude = std::clamp(std::fabs(sanitized.step), kAlphaMinStep, kAlphaMaxStep);
    sanitized.step = (sanitized.step < 0.0f) ? -magnitude : magnitude;

    sanitized.low = std::clamp(sanitized.low, 0.0f, 1.0f);
    sanitized.high = std::clamp(sanitized.high, 0.0f, 1.0f);
    if (sanitized.high <= sanitized.low + kAlphaMinRange) {
        sanitized.high = std::min(1.0f, sanitized.low + kAlphaDefaultHighDelta);
    }

    sanitized.borderOffset = std::clamp(sanitized.borderOffset, -kAlphaBorderLimit, kAlphaBorderLimit);
    sanitized.blurRadius = std::clamp(sanitized.blurRadius, 0, kAlphaBlurRadiusMax);
    sanitized.perVertexLimit = std::clamp(sanitized.perVertexLimit, 0.0f, kAlphaPerVertexLimitMax);

    return sanitized;
}

bool SegmentationPushPullTool::configsEqual(const AlphaPushPullConfig& lhs, const AlphaPushPullConfig& rhs)
{
    return nearlyEqual(lhs.start, rhs.start) &&
           nearlyEqual(lhs.stop, rhs.stop) &&
           nearlyEqual(lhs.step, rhs.step) &&
           nearlyEqual(lhs.low, rhs.low) &&
           nearlyEqual(lhs.high, rhs.high) &&
           nearlyEqual(lhs.borderOffset, rhs.borderOffset) &&
           lhs.blurRadius == rhs.blurRadius &&
           nearlyEqual(lhs.perVertexLimit, rhs.perVertexLimit) &&
           lhs.perVertex == rhs.perVertex;
}

void SegmentationPushPullTool::setAlphaConfig(const AlphaPushPullConfig& config)
{
    _alphaConfig = sanitizeConfig(config);
}

bool SegmentationPushPullTool::start(int direction, std::optional<bool> alphaOverride)
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

    const auto hover = _module.hoverInfo();
    if (!hover.valid || !hover.viewer || !_module.isSegmentationViewer(hover.viewer)) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    _activeAlphaEnabled = alphaOverride.value_or(false);
    _alphaOverrideActive = alphaOverride.has_value();

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
    _alphaOverrideActive = false;
    _activeAlphaEnabled = false;
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

    const auto hover = _module.hoverInfo();
    if (!hover.valid || !hover.viewer || !_module.isSegmentationViewer(hover.viewer)) {
        return false;
    }

    const int row = hover.row;
    const int col = hover.col;

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
    bool usedAlphaPushPullPerVertex = false;

    if (_activeAlphaEnabled && _alphaConfig.perVertex) {
        const auto& activeSamples = _editManager->activeDrag().samples;
        if (!activeSamples.empty()) {
            bool alphaUnavailable = false;

            std::vector<cv::Vec3f> perVertexTargets;
            perVertexTargets.reserve(activeSamples.size());
            std::vector<float> perVertexMovements;
            perVertexMovements.reserve(activeSamples.size());
            bool anyMovement = false;
            float minMovement = std::numeric_limits<float>::max();

            for (const auto& sample : activeSamples) {
                const cv::Vec3f& baseWorld = sample.baseWorld;
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
                auto sampleTarget = computeAlphaTarget(baseWorld,
                                 sampleNormal,
                                 _state.direction,
                                 baseSurface,
                                 hover.viewer,
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
                        const cv::Vec3f& baseWorld = activeSamples[i].baseWorld;
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
            usedAlphaPushPullPerVertex = true;
        }
    } else if (_activeAlphaEnabled) {
        bool alphaUnavailable = false;
        auto alphaTarget = computeAlphaTarget(centerWorld,
                          normal,
                          _state.direction,
                          baseSurface,
                          hover.viewer,
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

    if (!usedAlphaPushPullPerVertex) {
        if (!_editManager->updateActiveDrag(targetWorld)) {
            _editManager->cancelActiveDrag();
            if (snapshotCapturedThisStep) {
                _module.discardLastUndoSnapshot();
                _undoCaptured = false;
            }
            return false;
        }
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

std::optional<cv::Vec3f> SegmentationPushPullTool::computeAlphaTarget(const cv::Vec3f& centerWorld,
                                                const cv::Vec3f& normal,
                                                int direction,
                                                QuadSurface* surface,
                                                CVolumeViewer* viewer,
                                                bool* outUnavailable) const
{
    if (outUnavailable) {
        *outUnavailable = false;
    }

    if (!_activeAlphaEnabled || !viewer || !surface) {
        return std::nullopt;
    }

    std::shared_ptr<Volume> volume = viewer->currentVolume();
    if (!volume) {
        if (outUnavailable) {
            *outUnavailable = true;
        }
        return std::nullopt;
    }

    const size_t scaleCount = volume->numScales();
    int datasetIndex = viewer->datasetScaleIndex();
    if (scaleCount == 0) {
        datasetIndex = 0;
    } else {
        datasetIndex = std::clamp(datasetIndex, 0, static_cast<int>(scaleCount) - 1);
    }

    z5::Dataset* dataset = volume->zarrDataset(datasetIndex);
    if (!dataset) {
        dataset = volume->zarrDataset(0);
    }
    if (!dataset) {
        if (outUnavailable) {
            *outUnavailable = true;
        }
        return std::nullopt;
    }

    float scale = viewer->datasetScaleFactor();
    if (!std::isfinite(scale) || scale <= 0.0f) {
        scale = 1.0f;
    }

    ChunkCache* cache = viewer->chunkCachePtr();

    AlphaPushPullConfig cfg = sanitizeConfig(_alphaConfig);

    cv::Vec3f orientedNormal = normal * static_cast<float>(direction);
    const float norm = cv::norm(orientedNormal);
    if (norm <= 1e-4f) {
        return std::nullopt;
    }
    orientedNormal /= norm;

    const int radius = std::max(cfg.blurRadius, 0);
    const int kernel = radius * 2 + 1;
    const cv::Size patchSize(kernel, kernel);

    PlaneSurface plane(centerWorld, orientedNormal);
    cv::Mat_<cv::Vec3f> coords;
    plane.gen(&coords, nullptr, patchSize, cv::Vec3f(0, 0, 0), scale, cv::Vec3f(0, 0, 0));
    coords *= scale;

    const cv::Point2i centerIndex(radius, radius);
    const float range = std::max(cfg.high - cfg.low, kAlphaMinRange);

    float transparent = 1.0f;
    float integ = 0.0f;

    const float start = cfg.start;
    const float stop = cfg.stop;
    const float step = std::fabs(cfg.step);

    for (float offset = start; offset <= stop + 1e-4f; offset += step) {
        cv::Mat_<uint8_t> slice;
        cv::Mat_<cv::Vec3f> offsetMat(patchSize, orientedNormal * (offset * scale));
        readInterpolated3D(slice, dataset, coords + offsetMat, cache);
        if (slice.empty()) {
            continue;
        }

        cv::Mat sliceFloat;
        slice.convertTo(sliceFloat, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(sliceFloat, sliceFloat, cv::Size(kernel, kernel), 0);

        cv::Mat_<float> opaq = sliceFloat;
        opaq = (opaq - cfg.low) / range;
        cv::min(opaq, 1.0f, opaq);
        cv::max(opaq, 0.0f, opaq);

        const float centerOpacity = opaq(centerIndex);
        const float joint = transparent * centerOpacity;
        integ += joint * offset;
        transparent -= joint;

        if (transparent <= 1e-3f) {
            break;
        }
    }

    if (transparent >= 1.0f) {
        return std::nullopt;
    }

    const float denom = 1.0f - transparent;
    if (denom < 1e-5f) {
        return std::nullopt;
    }

    const float expected = integ / denom;
    if (!std::isfinite(expected)) {
        return std::nullopt;
    }

    const float totalOffset = expected + cfg.borderOffset;
    if (!std::isfinite(totalOffset) || totalOffset <= 0.0f) {
        return std::nullopt;
    }

    const cv::Vec3f targetWorld = centerWorld + orientedNormal * totalOffset;
    if (!std::isfinite(targetWorld[0]) || !std::isfinite(targetWorld[1]) || !std::isfinite(targetWorld[2])) {
        return std::nullopt;
    }

    const cv::Vec3f delta = targetWorld - centerWorld;
    if (cv::norm(delta) < 1e-4f) {
        return std::nullopt;
    }

    return targetWorld;
}
