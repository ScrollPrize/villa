#include "SegmentationPushPullTool.hpp"

#include "../SegmentationModule.hpp"
#include "../../ViewerManager.hpp"
#include "../../tiled/CTiledVolumeViewer.hpp"
#include "SegmentationEditManager.hpp"
#include "../SegmentationWidget.hpp"
#include "../../overlays/SegmentationOverlayController.hpp"
#include "../../CState.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <QCoreApplication>
#include <QTimer>
#include <QLoggingCategory>
#include <QtConcurrent/QtConcurrent>

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
constexpr int kPushPullIntervalMsSlow = 33;   // Alpha mode: ~30fps for responsiveness
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
                                                   CState* state)
    : _module(module)
    , _editManager(editManager)
    , _widget(widget)
    , _overlay(overlay)
    , _state(state)
{
    ensureTimer();

    QObject::connect(&_alphaWatcher, &QFutureWatcher<AlphaResult>::finished,
                     &_module, [this]() { applyAlphaResult(); });
}

void SegmentationPushPullTool::setDependencies(SegmentationEditManager* editManager,
                                               SegmentationWidget* widget,
                                               SegmentationOverlayController* overlay,
                                               CState* state)
{
    _editManager = editManager;
    _widget = widget;
    _overlay = overlay;
    _state = state;
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

    if (_ppState.active && _ppState.direction == direction) {
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

    _ppState.active = true;
    _ppState.direction = direction;
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
    if (!_ppState.active) {
        return;
    }
    if (direction != 0 && direction != _ppState.direction) {
        return;
    }
    stopAll();
}

void SegmentationPushPullTool::stopAll()
{
    const bool wasActive = _ppState.active;
    _ppState.active = false;
    _ppState.direction = 0;
    if (_timer && _timer->isActive()) {
        _timer->stop();
    }
    _alphaOverrideActive = false;
    _activeAlphaEnabled = false;
    _alphaComputePending = false;

    // If an async computation is running, wait for it so we can apply final results
    if (_alphaComputeRunning) {
        _alphaWatcher.waitForFinished();
        _alphaComputeRunning = false;
    }

    // Clear cached position
    _cachedRow = -1;
    _cachedCol = -1;
    _samplesValid = false;

    if (_module._activeFalloff == SegmentationModule::FalloffTool::PushPull) {
        _module.useFalloff(SegmentationModule::FalloffTool::Drag);
    }

    // Finalize the edits and trigger final surface update
    if (wasActive && _editManager && _editManager->hasSession() && _state) {
        // Capture delta for undo before applyPreview() clears edited vertices
        _module.captureUndoDelta();

        // Auto-approve edited regions before applyPreview() clears them
        if (_module.autoApprovalEnabled() && _overlay && _overlay->hasApprovalMaskData()) {
            const auto editedVerts = _editManager->editedVertices();
            if (!editedVerts.empty()) {
                // Get drag center from the cached row/col if available
                std::optional<std::pair<int, int>> dragCenter;
                if (_cachedRow >= 0 && _cachedCol >= 0) {
                    dragCenter = std::make_pair(_cachedRow, _cachedCol);
                }
                const auto filteredVerts = _module.filterVerticesForAutoApproval(editedVerts, dragCenter);
                _module.performAutoApproval(filteredVerts);
            }
        }

        _editManager->applyPreview();
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
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
    if (!_ppState.active || !_editManager || !_editManager->hasSession()) {
        qCWarning(lcSegPushPull) << "Push/pull aborted: tool inactive or no active editing session.";
        return false;
    }

    // If an async alpha computation is already running, note that another tick
    // is needed and return — the watcher callback will re-launch.
    if (_activeAlphaEnabled && _alphaComputeRunning) {
        _alphaComputePending = true;
        return true;
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
        cv::Vec3f ptr(0, 0, 0);
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

    // --- Alpha mode: launch expensive computation on background thread ---
    if (_activeAlphaEnabled) {
        launchAlphaCompute();
        return true;
    }

    // --- Non-alpha mode: synchronous step (cheap) ---
    cv::Vec3f targetWorld = centerWorld;
    const float stepWorld = _module.gridStepWorld() * _stepMultiplier;
    if (stepWorld <= 0.0f) {
        _editManager->cancelActiveDrag();
        _samplesValid = false;
        logFailure("Push/pull aborted: computed step size non-positive");
        return false;
    }
    targetWorld = centerWorld + normal * (static_cast<float>(_ppState.direction) * stepWorld);

    if (!_editManager->updateActiveDrag(targetWorld)) {
        _editManager->cancelActiveDrag();
        _samplesValid = false;
        logFailure("Push/pull aborted: failed to update drag target");
        return false;
    }

    // Update sample base positions for next tick (allows reusing samples)
    // Skip commitActiveDrag() and applyPreview() during continuous operation
    // - they clear samples, causing expensive rebuilds every tick
    // Final cleanup happens in stopAll()
    _editManager->refreshActiveDragBasePositions();

    // Trigger visual refresh
    if (_state) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }

    _module.refreshOverlay();
    _module.markAutosaveNeeded();
    return true;
}

void SegmentationPushPullTool::launchAlphaCompute()
{
    if (_alphaComputeRunning) {
        _alphaComputePending = true;
        return;
    }

    if (!_editManager || !_editManager->hasSession() || !_ppState.active) {
        return;
    }

    if (!_module.ensureHoverTarget()) {
        return;
    }
    const auto hover = _module.hoverInfo();
    if (!hover.valid || !hover.viewer) {
        return;
    }

    auto baseSurface = _editManager->baseSurface();
    if (!baseSurface) {
        return;
    }

    // Capture all inputs needed by the background thread on the main thread
    std::shared_ptr<Volume> volume = hover.viewer->currentVolume();
    if (!volume) {
        return;
    }

    const size_t scaleCount = volume->numScales();
    int datasetIndex = hover.viewer->datasetScaleIndex();
    if (scaleCount == 0) {
        datasetIndex = 0;
    } else {
        datasetIndex = std::clamp(datasetIndex, 0, static_cast<int>(scaleCount) - 1);
    }

    float scale = hover.viewer->datasetScaleFactor();
    if (!std::isfinite(scale) || scale <= 0.0f) {
        scale = 1.0f;
    }
    scale = std::max(scale, 0.25f);

    const int direction = _ppState.direction;
    const AlphaPushPullConfig config = _alphaConfig;
    const bool perVertex = config.perVertex;

    // Snapshot per-vertex sample data from the active drag
    struct SampleInput {
        cv::Vec3f baseWorld;
        cv::Vec3f normal;
    };

    std::vector<SampleInput> sampleInputs;
    cv::Vec3f centerWorld(0, 0, 0);
    cv::Vec3f centerNormal(0, 0, 0);

    {
        const int row = _cachedRow;
        const int col = _cachedCol;

        auto centerWorldOpt = _editManager->vertexWorldPosition(row, col);
        if (!centerWorldOpt) {
            return;
        }
        centerWorld = *centerWorldOpt;

        centerNormal = baseSurface->gridNormal(row, col);
        if (!isValidNormal(centerNormal)) {
            auto* patchIndex = _module.viewerManager() ? _module.viewerManager()->surfacePatchIndex() : nullptr;
            cv::Vec3f ptr(0, 0, 0);
            baseSurface->pointTo(ptr, centerWorld, std::numeric_limits<float>::max(), 400, patchIndex);
            if (const auto fallback = computeRobustNormal(baseSurface.get(), ptr, centerWorld, _editManager->activeDrag(), patchIndex)) {
                centerNormal = *fallback;
            } else {
                return;
            }
        }
        const float n = cv::norm(centerNormal);
        if (n <= 1e-4f) {
            return;
        }
        centerNormal /= n;

        if (perVertex) {
            const auto& activeSamples = _editManager->activeDrag().samples;
            sampleInputs.reserve(activeSamples.size());
            for (const auto& sample : activeSamples) {
                cv::Vec3f sampleNormal = baseSurface->gridNormal(sample.row, sample.col);
                if (!isValidNormal(sampleNormal)) {
                    sampleNormal = centerNormal;
                } else {
                    const float sn = cv::norm(sampleNormal);
                    if (sn > 1e-4f) {
                        sampleNormal /= sn;
                    } else {
                        sampleNormal = centerNormal;
                    }
                }
                sampleInputs.push_back({sample.baseWorld, sampleNormal});
            }
        }
    }

    _alphaComputeRunning = true;
    _alphaComputePending = false;

    auto future = QtConcurrent::run(
        [volume, datasetIndex, scale, direction, config, perVertex,
         centerWorld, centerNormal, sampleInputs]() -> AlphaResult {

        AlphaResult result;

        if (perVertex && !sampleInputs.empty()) {
            result.perVertex = true;

            std::vector<cv::Vec3f> targets;
            targets.reserve(sampleInputs.size());
            std::vector<float> movements;
            movements.reserve(sampleInputs.size());
            bool anyMovement = false;
            float minMovement = std::numeric_limits<float>::max();

            for (const auto& si : sampleInputs) {
                bool unavailable = false;
                auto target = computeAlphaTargetStatic(
                    si.baseWorld, si.normal, direction, config,
                    volume, datasetIndex, scale, &unavailable);

                if (unavailable) {
                    return result;  // success = false
                }

                cv::Vec3f newWorld = si.baseWorld;
                float movement = 0.0f;
                if (target) {
                    newWorld = *target;
                    const cv::Vec3f delta = newWorld - si.baseWorld;
                    movement = static_cast<float>(cv::norm(delta));
                    if (movement >= 1e-4f) {
                        anyMovement = true;
                    }
                }

                targets.push_back(newWorld);
                movements.push_back(movement);
                minMovement = std::min(minMovement, movement);
            }

            // Apply per-vertex limit clamping
            const float perVertexLimit = std::max(0.0f, config.perVertexLimit);
            if (perVertexLimit > 0.0f && !targets.empty() && std::isfinite(minMovement)) {
                const float maxAllowed = minMovement + perVertexLimit;
                for (std::size_t i = 0; i < targets.size(); ++i) {
                    if (movements[i] > maxAllowed + 1e-4f) {
                        const cv::Vec3f delta = targets[i] - sampleInputs[i].baseWorld;
                        const float length = movements[i];
                        if (length > 1e-6f) {
                            const float s = maxAllowed / length;
                            targets[i] = sampleInputs[i].baseWorld + delta * s;
                            movements[i] = maxAllowed;
                            if (maxAllowed >= 1e-4f) {
                                anyMovement = true;
                            }
                        }
                    }
                }
            }

            if (!anyMovement) {
                return result;  // success = false
            }

            result.perVertexTargets = std::move(targets);
            result.success = true;
        } else {
            // Single-target alpha mode
            bool unavailable = false;
            auto target = computeAlphaTargetStatic(
                centerWorld, centerNormal, direction, config,
                volume, datasetIndex, scale, &unavailable);

            if (target) {
                result.singleTarget = *target;
                result.success = true;
            } else if (unavailable) {
                // Volume unavailable — not an error, just skip
                result.success = false;
            }
        }

        return result;
    });

    _alphaWatcher.setFuture(future);
}

void SegmentationPushPullTool::applyAlphaResult()
{
    _alphaComputeRunning = false;

    if (!_ppState.active || !_editManager || !_editManager->hasSession()) {
        return;
    }

    const auto result = _alphaWatcher.result();

    if (result.success) {
        if (result.perVertex) {
            if (!_editManager->updateActiveDragTargets(result.perVertexTargets)) {
                qCWarning(lcSegPushPull) << "Alpha async: failed to update per-vertex drag targets";
                _editManager->cancelActiveDrag();
                _samplesValid = false;
                return;
            }
        } else if (result.singleTarget) {
            if (!_editManager->updateActiveDrag(*result.singleTarget)) {
                qCWarning(lcSegPushPull) << "Alpha async: failed to update drag target";
                _editManager->cancelActiveDrag();
                _samplesValid = false;
                return;
            }
        }

        // Update sample base positions for next tick
        _editManager->refreshActiveDragBasePositions();

        // Trigger visual refresh on the main thread
        if (_state) {
            _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
        }

        _module.refreshOverlay();
        _module.markAutosaveNeeded();
    }

    // If another tick came in while we were computing, launch again
    if (_alphaComputePending && _ppState.active) {
        _alphaComputePending = false;
        launchAlphaCompute();
    }
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

std::optional<cv::Vec3f> SegmentationPushPullTool::computeAlphaTargetStatic(
    const cv::Vec3f& centerWorld,
    const cv::Vec3f& normal,
    int direction,
    const AlphaPushPullConfig& config,
    const std::shared_ptr<Volume>& volume,
    int datasetIndex,
    float scale,
    bool* outUnavailable)
{
    if (outUnavailable) {
        *outUnavailable = false;
    }

    if (!volume) {
        if (outUnavailable) {
            *outUnavailable = true;
        }
        return std::nullopt;
    }

    AlphaPushPullConfig cfg = sanitizeConfig(config);

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

    // Pre-allocate mats reused across loop iterations to avoid per-tick allocations
    cv::Mat_<cv::Vec3f> offsetMat(patchSize);
    cv::Mat_<cv::Vec3f> offsetCoords(patchSize);
    cv::Mat_<uint8_t> slice(patchSize);
    cv::Mat sliceFloat(patchSize, CV_32F);

    for (float offset = start; offset <= stop + 1e-4f; offset += step) {
        offsetMat.setTo(orientedNormal * (offset * scale));
        cv::add(coords, offsetMat, offsetCoords);
        vc::SampleParams sp;
        sp.level = datasetIndex;
        volume->sample(slice, offsetCoords, sp);
        if (slice.empty()) {
            continue;
        }

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
