#include "SegmentationPushPullTool.hpp"

#include "../SegmentationModule.hpp"
#include "../../ViewerManager.hpp"
#include "../../CVolumeViewer.hpp"
#include "SegmentationEditManager.hpp"
#include "../SegmentationWidget.hpp"
#include "../../overlays/SegmentationOverlayController.hpp"
#include "../../CSurfaceCollection.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <QCoreApplication>
#include <QTimer>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <numeric>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

Q_LOGGING_CATEGORY(lcSegPushPull, "vc.segmentation.pushpull")

namespace
{
constexpr int kPushPullIntervalMs = 16;       // ~60fps for smooth feedback
constexpr int kPushPullIntervalMsFast = 16;   // Non-alpha mode: faster feedback
constexpr int kPushPullIntervalMsSlow = 100;  // Alpha mode: more time for computation
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

bool isFiniteVec3(const cv::Vec3f& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool isValidNormal(const cv::Vec3f& normal)
{
    if (!isFiniteVec3(normal)) {
        return false;
    }
    const float norm = static_cast<float>(cv::norm(normal));
    return norm > 1e-4f;
}

cv::Vec3f normalizeVec(const cv::Vec3f& value)
{
    const float norm = static_cast<float>(cv::norm(value));
    if (!std::isfinite(norm) || norm <= 1e-6f) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }
    return value / norm;
}

std::optional<cv::Vec3f> averageNormals(const std::vector<cv::Vec3f>& normals)
{
    if (normals.empty()) {
        return std::nullopt;
    }
    cv::Vec3f sum(0.0f, 0.0f, 0.0f);
    for (const auto& normal : normals) {
        sum += normal;
    }
    sum = normalizeVec(sum);
    if (!isValidNormal(sum)) {
        return std::nullopt;
    }
    return sum;
}

std::optional<cv::Vec3f> sampleSurfaceNormalsNearCenter(QuadSurface* surface,
                                                        const cv::Vec3f& basePtr,
                                                        const SegmentationEditManager::ActiveDrag& drag,
                                                        SurfacePatchIndex* patchIndex = nullptr)
{
    if (!surface || !drag.active) {
        return std::nullopt;
    }

    std::vector<cv::Vec3f> normals;
    normals.reserve(8);

    // Try the center first, then nearby samples in order of proximity.
    const auto collectNormalAt = [&](const cv::Vec3f& worldPoint) {
        cv::Vec3f ptrCandidate = basePtr;
        surface->pointTo(ptrCandidate, worldPoint, std::numeric_limits<float>::max(), 200, patchIndex);
        const cv::Vec3f candidateNormal = surface->normal(ptrCandidate);
        if (isValidNormal(candidateNormal)) {
            normals.push_back(candidateNormal);
        }
    };

    collectNormalAt(drag.baseWorld);
    if (normals.empty()) {
        // Evaluate additional nearby samples, prioritising the ones closest to the center.
        auto samples = drag.samples;
        std::sort(samples.begin(),
                  samples.end(),
                  [](const SegmentationEditManager::DragSample& lhs,
                     const SegmentationEditManager::DragSample& rhs) {
                      return lhs.distanceWorldSq < rhs.distanceWorldSq;
                  });
        for (const auto& sample : samples) {
            if (sample.row == drag.center.row && sample.col == drag.center.col) {
                continue;
            }
            collectNormalAt(sample.baseWorld);
            if (normals.size() >= 4) {
                break;
            }
        }
    }

    return averageNormals(normals);
}

enum class AxisDirection
{
    Row,
    Column
};

std::optional<cv::Vec3f> axisVectorFromSamples(AxisDirection axis,
                                               const SegmentationEditManager::ActiveDrag& drag)
{
    if (!drag.active || drag.samples.empty()) {
        return std::nullopt;
    }

    const auto& center = drag.center;
    const cv::Vec3f& centerWorld = drag.baseWorld;

    const SegmentationEditManager::DragSample* posSample = nullptr;
    const SegmentationEditManager::DragSample* negSample = nullptr;

    int bestPosPrimary = std::numeric_limits<int>::max();
    int bestPosSecondary = std::numeric_limits<int>::max();
    float bestPosDist = std::numeric_limits<float>::max();

    int bestNegPrimary = std::numeric_limits<int>::max();
    int bestNegSecondary = std::numeric_limits<int>::max();
    float bestNegDist = std::numeric_limits<float>::max();

    for (const auto& sample : drag.samples) {
        int primaryDelta = 0;
        int secondaryDelta = 0;
        if (axis == AxisDirection::Row) {
            primaryDelta = sample.row - center.row;
            secondaryDelta = std::abs(sample.col - center.col);
        } else {
            primaryDelta = sample.col - center.col;
            secondaryDelta = std::abs(sample.row - center.row);
        }

        if (primaryDelta == 0) {
            continue;
        }

        const int absPrimary = std::abs(primaryDelta);
        const float distSq = std::max(sample.distanceWorldSq, 0.0f);

        auto updateCandidate = [&](const SegmentationEditManager::DragSample*& currentSample,
                                   int& bestPrimary,
                                   int& bestSecondary,
                                   float& bestDist) {
            if (absPrimary < bestPrimary ||
                (absPrimary == bestPrimary &&
                 (secondaryDelta < bestSecondary ||
                  (secondaryDelta == bestSecondary && distSq < bestDist)))) {
                currentSample = &sample;
                bestPrimary = absPrimary;
                bestSecondary = secondaryDelta;
                bestDist = distSq;
            }
        };

        if (primaryDelta > 0) {
            updateCandidate(posSample, bestPosPrimary, bestPosSecondary, bestPosDist);
        } else {
            updateCandidate(negSample, bestNegPrimary, bestNegSecondary, bestNegDist);
        }
    }

    cv::Vec3f axisVec(0.0f, 0.0f, 0.0f);
    if (posSample && negSample) {
        axisVec = posSample->baseWorld - negSample->baseWorld;
    } else if (posSample) {
        axisVec = posSample->baseWorld - centerWorld;
    } else if (negSample) {
        axisVec = centerWorld - negSample->baseWorld;
    } else {
        return std::nullopt;
    }

    axisVec = normalizeVec(axisVec);
    if (!isValidNormal(axisVec)) {
        return std::nullopt;
    }
    return axisVec;
}

std::optional<cv::Vec3f> fitPlaneNormal(const SegmentationEditManager::ActiveDrag& drag,
                                        const cv::Vec3f& centerWorld,
                                        const std::optional<cv::Vec3f>& rowVec,
                                        const std::optional<cv::Vec3f>& colVec,
                                        const std::optional<cv::Vec3f>& orientationHint)
{
    if (!drag.active) {
        return std::nullopt;
    }

    if (drag.samples.size() < 3) {
        return std::nullopt;
    }

    std::vector<cv::Vec3d> points;
    std::vector<double> weights;
    points.reserve(drag.samples.size());
    weights.reserve(drag.samples.size());

    double weightSum = 0.0;
    cv::Vec3d centroid(0.0, 0.0, 0.0);

    for (const auto& sample : drag.samples) {
        if (!isFiniteVec3(sample.baseWorld)) {
            continue;
        }
        const double dist = std::sqrt(std::max(sample.distanceWorldSq, 0.0f));
        const double weight = 1.0 / (1.0 + dist);
        const cv::Vec3d point(sample.baseWorld[0], sample.baseWorld[1], sample.baseWorld[2]);
        points.push_back(point);
        weights.push_back(weight);
        centroid += point * weight;
        weightSum += weight;
    }

    if (weightSum <= 0.0 || points.size() < 3) {
        return std::nullopt;
    }

    centroid /= weightSum;

    cv::Matx33d covariance = cv::Matx33d::zeros();
    for (std::size_t i = 0; i < points.size(); ++i) {
        const cv::Vec3d diff = points[i] - centroid;
        const double w = weights[i];
        covariance(0, 0) += w * diff[0] * diff[0];
        covariance(0, 1) += w * diff[0] * diff[1];
        covariance(0, 2) += w * diff[0] * diff[2];
        covariance(1, 0) += w * diff[1] * diff[0];
        covariance(1, 1) += w * diff[1] * diff[1];
        covariance(1, 2) += w * diff[1] * diff[2];
        covariance(2, 0) += w * diff[2] * diff[0];
        covariance(2, 1) += w * diff[2] * diff[1];
        covariance(2, 2) += w * diff[2] * diff[2];
    }

    cv::Mat eigenValues, eigenVectors;
    cv::eigen(covariance, eigenValues, eigenVectors);
    if (eigenVectors.rows != 3 || eigenVectors.cols != 3) {
        return std::nullopt;
    }

    cv::Vec3d normal(eigenVectors.at<double>(2, 0),
                     eigenVectors.at<double>(2, 1),
                     eigenVectors.at<double>(2, 2));
    double normalNorm = cv::norm(normal);
    if (!std::isfinite(normalNorm) || normalNorm <= 1e-6) {
        return std::nullopt;
    }
    normal /= normalNorm;

    cv::Vec3d orientation(0.0, 0.0, 0.0);
    bool orientationValid = false;
    if (rowVec && colVec) {
        const cv::Vec3f crossHint = rowVec->cross(*colVec);
        const double hintNorm = cv::norm(crossHint);
        if (hintNorm > 1e-6) {
            orientation = cv::Vec3d(crossHint[0] / hintNorm,
                                    crossHint[1] / hintNorm,
                                    crossHint[2] / hintNorm);
            orientationValid = true;
        }
    }

    if (!orientationValid && orientationHint) {
        const cv::Vec3f hintVec = normalizeVec(*orientationHint);
        const double hintNorm = cv::norm(cv::Vec3d(hintVec[0], hintVec[1], hintVec[2]));
        if (hintNorm > 1e-6) {
            orientation = cv::Vec3d(hintVec[0], hintVec[1], hintVec[2]);
            orientationValid = true;
        }
    }

    if (!orientationValid) {
        cv::Vec3d toCentroid = centroid - cv::Vec3d(centerWorld[0], centerWorld[1], centerWorld[2]);
        const double hintNorm = cv::norm(toCentroid);
        if (hintNorm > 1e-6) {
            orientation = toCentroid / hintNorm;
            orientationValid = true;
        }
    }

    if (orientationValid && normal.dot(orientation) < 0.0) {
        normal = -normal;
    }

    return cv::Vec3f(static_cast<float>(normal[0]),
                     static_cast<float>(normal[1]),
                     static_cast<float>(normal[2]));
}

std::optional<cv::Vec3f> computeRobustNormal(QuadSurface* surface,
                                             const cv::Vec3f& centerPtr,
                                             const cv::Vec3f& centerWorld,
                                             const SegmentationEditManager::ActiveDrag& drag,
                                             SurfacePatchIndex* patchIndex = nullptr)
{
    if (!surface || !drag.active) {
        return std::nullopt;
    }

    const auto surfaceNormal = sampleSurfaceNormalsNearCenter(surface, centerPtr, drag, patchIndex);
    const auto rowVec = axisVectorFromSamples(AxisDirection::Row, drag);
    const auto colVec = axisVectorFromSamples(AxisDirection::Column, drag);

    std::optional<cv::Vec3f> crossNormal;
    if (rowVec && colVec) {
        cv::Vec3f candidate = rowVec->cross(*colVec);
        candidate = normalizeVec(candidate);
        if (isValidNormal(candidate)) {
            crossNormal = candidate;
        }
    }

    const std::optional<cv::Vec3f> orientationHint = crossNormal ? crossNormal : surfaceNormal;
    if (auto planeNormal = fitPlaneNormal(drag, centerWorld, rowVec, colVec, orientationHint)) {
        return planeNormal;
    }

    if (surfaceNormal) {
        return surfaceNormal;
    }
    if (crossNormal) {
        return crossNormal;
    }

    return std::nullopt;
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
    _stepMultiplier = std::clamp(multiplier, 0.05f, 40.0f);
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

    if (!_module.ensureHoverTarget()) {
        return false;
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

    // Reset cached position for new operation
    _cachedRow = -1;
    _cachedCol = -1;
    _samplesValid = false;

    _module.useFalloff(SegmentationModule::FalloffTool::PushPull);

    // Set adaptive timer interval based on alpha mode
    if (_timer) {
        const int interval = _activeAlphaEnabled ? kPushPullIntervalMsSlow : kPushPullIntervalMsFast;
        _timer->setInterval(interval);
        if (!_timer->isActive()) {
            _timer->start();
        }
    }

    // Let the timer handle the first step asynchronously to avoid blocking the UI
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
    const bool wasActive = _state.active;
    _state.active = false;
    _state.direction = 0;
    if (_timer && _timer->isActive()) {
        _timer->stop();
    }
    _alphaOverrideActive = false;
    _activeAlphaEnabled = false;

    // Clear cached position
    _cachedRow = -1;
    _cachedCol = -1;
    _samplesValid = false;

    if (_module._activeFalloff == SegmentationModule::FalloffTool::PushPull) {
        _module.useFalloff(SegmentationModule::FalloffTool::Drag);
    }

    // Finalize the edits and trigger final surface update
    if (wasActive && _editManager && _editManager->hasSession() && _surfaces) {
        // Capture delta for undo before applyPreview() clears edited vertices
        _module.captureUndoDelta();

        // Auto-approve edited regions before applyPreview() clears them
        if (_module.autoApproveEdits() && _overlay && _overlay->hasApprovalMaskData()) {
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
                _overlay->paintApprovalMaskDirect(gridPositions, kRadius, kApproved, brushColor, false, 0.0f, 0.0f, kIsAutoApproval);
                _overlay->scheduleDebouncedSave(_editManager->baseSurface().get());
                qCInfo(lcSegPushPull) << "Auto-approved" << gridPositions.size() << "push/pull edited vertices";
            }
        }

        _editManager->applyPreview();
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
        _module.emitPendingChanges();
    }

    _undoCaptured = false;
}

bool SegmentationPushPullTool::applyStep()
{
    return applyStepInternal();
}

bool SegmentationPushPullTool::applyStepInternal()
{
    if (!_state.active || !_editManager || !_editManager->hasSession()) {
        qCWarning(lcSegPushPull) << "Push/pull aborted: tool inactive or no active editing session.";
        return false;
    }

    if (!_module.ensureHoverTarget()) {
        return false;
    }
    const auto hover = _module.hoverInfo();
    if (!hover.valid || !hover.viewer || !_module.isSegmentationViewer(hover.viewer)) {
        qCWarning(lcSegPushPull) << "Push/pull aborted: hover info invalid or viewer not ready.";
        return false;
    }

    const int row = hover.row;
    const int col = hover.col;
    const auto logFailure = [&](const char* reason) {
        qCWarning(lcSegPushPull) << reason << "(row" << row << ", col" << col << ")";
    };

    // Check if we can reuse existing samples (position unchanged and samples still valid)
    const bool positionChanged = (row != _cachedRow || col != _cachedCol);
    const bool needRebuild = positionChanged || !_samplesValid || !_editManager->activeDrag().active;

    if (needRebuild) {
        if (!_editManager->beginActiveDrag({row, col})) {
            logFailure("Push/pull aborted: beginActiveDrag failed");
            return false;
        }
        _cachedRow = row;
        _cachedCol = col;
        _samplesValid = true;
    }

    auto centerWorldOpt = _editManager->vertexWorldPosition(row, col);
    if (!centerWorldOpt) {
        _editManager->cancelActiveDrag();
        _samplesValid = false;
        logFailure("Push/pull aborted: vertex world position unavailable");
        return false;
    }
    const cv::Vec3f centerWorld = *centerWorldOpt;

    auto baseSurface = _editManager->baseSurface();
    if (!baseSurface) {
        _editManager->cancelActiveDrag();
        _samplesValid = false;
        logFailure("Push/pull aborted: base surface missing");
        return false;
    }

    auto* patchIndex = _module.viewerManager() ? _module.viewerManager()->surfacePatchIndex() : nullptr;

    // Get normal directly from grid position (avoids expensive pointTo lookup)
    cv::Vec3f normal = baseSurface->gridNormal(row, col);
    if (!isValidNormal(normal)) {
        // Fallback to robust normal computation if direct lookup fails
        cv::Vec3f ptr = baseSurface->pointer();
        baseSurface->pointTo(ptr, centerWorld, std::numeric_limits<float>::max(), 400, patchIndex);
        if (const auto fallbackNormal = computeRobustNormal(baseSurface.get(), ptr, centerWorld, _editManager->activeDrag(), patchIndex)) {
            normal = *fallbackNormal;
        } else {
            _editManager->cancelActiveDrag();
            _samplesValid = false;
            logFailure("Push/pull aborted: surface normal lookup failed");
            return false;
        }
    }

    const float norm = cv::norm(normal);
    if (norm <= 1e-4f) {
        _editManager->cancelActiveDrag();
        _samplesValid = false;
        logFailure("Push/pull aborted: surface normal magnitude too small");
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

                // Get normal directly from grid position (fast, no pointTo needed)
                cv::Vec3f sampleNormal = baseSurface->gridNormal(sample.row, sample.col);
                if (!isValidNormal(sampleNormal)) {
                    sampleNormal = normal;  // Fallback to center normal
                } else {
                    const float sampleNorm = cv::norm(sampleNormal);
                    if (sampleNorm > 1e-4f) {
                        sampleNormal /= sampleNorm;
                    } else {
                        sampleNormal = normal;
                    }
                }

                bool sampleUnavailable = false;
                auto sampleTarget = computeAlphaTarget(baseWorld,
                                 sampleNormal,
                                 _state.direction,
                                 baseSurface.get(),
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
                _samplesValid = false;
                logFailure("Push/pull aborted: alpha push/pull unavailable for per-vertex samples");
                return false;
            }

            if (!anyMovement) {
                _editManager->cancelActiveDrag();
                _samplesValid = false;
                logFailure("Push/pull aborted: alpha push/pull produced no movement for per-vertex samples");
                return false;
            }

            if (!_editManager->updateActiveDragTargets(perVertexTargets)) {
                _editManager->cancelActiveDrag();
                _samplesValid = false;
                logFailure("Push/pull aborted: failed to update per-vertex drag targets");
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
                          baseSurface.get(),
                          hover.viewer,
                          &alphaUnavailable);
        if (alphaTarget) {
            targetWorld = *alphaTarget;
            usedAlphaPushPull = true;
        } else if (!alphaUnavailable) {
            _editManager->cancelActiveDrag();
            _samplesValid = false;
            logFailure("Push/pull aborted: alpha push/pull target unavailable");
            return false;
        }
    }

    if (!usedAlphaPushPull) {
        const float stepWorld = _module.gridStepWorld() * _stepMultiplier;
        if (stepWorld <= 0.0f) {
            _editManager->cancelActiveDrag();
            _samplesValid = false;
            logFailure("Push/pull aborted: computed step size non-positive");
            return false;
        }
        targetWorld = centerWorld + normal * (static_cast<float>(_state.direction) * stepWorld);
    }

    if (!usedAlphaPushPullPerVertex) {
        if (!_editManager->updateActiveDrag(targetWorld)) {
            _editManager->cancelActiveDrag();
            _samplesValid = false;
            logFailure("Push/pull aborted: failed to update drag target");
            return false;
        }
    }

    // Update sample base positions for next tick (allows reusing samples)
    // Skip commitActiveDrag() and applyPreview() during continuous operation
    // - they clear samples, causing expensive rebuilds every tick
    // Final cleanup happens in stopAll()
    _editManager->refreshActiveDragBasePositions();

    // Trigger visual refresh
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }

    _module.refreshOverlay();
    // Note: emitPendingChanges() removed here for performance - called in stopAll() instead
    _module.markAutosaveNeeded();
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

    ChunkCache<uint8_t>* cache = viewer->chunkCachePtr();

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
