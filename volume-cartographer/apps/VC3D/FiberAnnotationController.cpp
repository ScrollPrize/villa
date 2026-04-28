#include "FiberAnnotationController.hpp"

#include "CState.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "adaptive/CAdaptiveVolumeViewer.hpp"

#include <QMdiArea>
#include <QMdiSubWindow>
#include <cmath>

FiberAnnotationController::FiberAnnotationController(CState* state,
                                                     VCCollection* collection,
                                                     QObject* parent)
    : QObject(parent), _cstate(state), _collection(collection)
{
}

FiberAnnotationController::~FiberAnnotationController()
{
    closeAnnotationViewer();
}

void FiberAnnotationController::beginNewFiber()
{
    closeAnnotationViewer();
    _state = State::WaitingForFirstClick;
    _currentFiberId = 0;
    _recentPoints.clear();
    _fiberCollectionName.clear();
    emit crosshairModeChanged(true);
}

bool FiberAnnotationController::handleVolumeClick(const cv::Vec3f& vol_loc,
                                                   const cv::Vec3f& normal,
                                                   Surface* /*surf*/,
                                                   Qt::MouseButton button,
                                                   Qt::KeyboardModifiers /*modifiers*/)
{
    if (_state != State::WaitingForFirstClick)
        return false;
    if (button != Qt::LeftButton)
        return false;

    // Create the fiber collection
    _fiberCollectionName = _collection->generateNewCollectionName("fiber");
    uint64_t colId = _collection->addCollection(_fiberCollectionName);
    _currentFiberId = colId;

    // Set fiber tag
    _collection->setCollectionTag(colId, "fiber", "true");

    // Set winding metadata: not absolute
    CollectionMetadata meta;
    meta.absolute_winding_number = false;
    _collection->setCollectionMetadata(colId, meta);

    // Store initial normal and first point
    _initialNormal = normal;
    addFiberPoint(vol_loc);

    // Transition to annotation state
    _state = State::Annotating;
    emit crosshairModeChanged(false);

    // Open annotation viewer at predicted next position
    advanceToNextPrediction();
    emit requestReferenceViewer(kFiberReferenceSurface, tr("Fiber Ref"));
    emit requestAnnotationViewer(kFiberAnnotationSurface,
                                 QString("Fiber: %1").arg(QString::fromStdString(_fiberCollectionName)));

    return true;
}

bool FiberAnnotationController::handleEscape()
{
    if (_state == State::WaitingForFirstClick) {
        _state = State::Idle;
        emit crosshairModeChanged(false);
        return true;
    }
    if (_state == State::Annotating) {
        uint64_t id = _currentFiberId;
        closeAnnotationViewer();
        _state = State::Idle;
        _currentFiberId = 0;
        _recentPoints.clear();
        _fiberCollectionName.clear();
        emit annotationFinished(id);
        return true;
    }
    return false;
}

void FiberAnnotationController::onAnnotationViewerClicked(cv::Vec3f vol_loc,
                                                           cv::Vec3f /*normal*/,
                                                           Surface* /*surf*/,
                                                           Qt::MouseButton button,
                                                           Qt::KeyboardModifiers /*modifiers*/)
{
    if (_state != State::Annotating) return;
    if (button != Qt::LeftButton) return;

    addFiberPoint(vol_loc);
    advanceToNextPrediction();
}

void FiberAnnotationController::addFiberPoint(const cv::Vec3f& position)
{
    ColPoint pt = _collection->addPoint(_fiberCollectionName, position);

    // Set winding to 0
    ColPoint updated = pt;
    updated.winding_annotation = 0.0f;
    _collection->updatePoint(updated);

    // Compute arrival direction
    FiberPoint fp;
    fp.position = position;
    if (!_recentPoints.empty()) {
        cv::Vec3f diff = position - _recentPoints.back().position;
        float len = static_cast<float>(cv::norm(diff));
        fp.arrivalDirection = len > 1e-6f ? diff / len : _recentPoints.back().arrivalDirection;
    } else {
        fp.arrivalDirection = _initialNormal;
    }

    _recentPoints.push_back(fp);
    // Keep only last 3
    while (_recentPoints.size() > 3) {
        _recentPoints.erase(_recentPoints.begin());
    }
}

void FiberAnnotationController::onStepChanged(int step)
{
    _fiberStep = step;
    if (_state == State::Annotating && !_recentPoints.empty()) {
        advanceToNextPrediction();
    }
}

void FiberAnnotationController::advanceToNextPrediction()
{
    auto [nextPos, nextDir] = [this]() -> std::pair<cv::Vec3f, cv::Vec3f> {
        if (_recentPoints.size() == 1) return predictFromOnePoint();
        if (_recentPoints.size() == 2) return predictFromTwoPoints();
        return predictFromThreeOrMore();
    }();

    // 1. Get old annotation plane
    auto oldPlane = std::dynamic_pointer_cast<PlaneSurface>(_cstate->surface(kFiberAnnotationSurface));
    cv::Vec3f oldVy = oldPlane ? oldPlane->basisY() : cv::Vec3f(0, 1, 0);

    // Clamp normal rotation to 5°/voxel relative to previous slice normal
    if (oldPlane) {
        cv::Vec3f oldNormal = oldPlane->normal({});
        float cosA = std::clamp(oldNormal.dot(nextDir), -1.0f, 1.0f);
        float angle = std::acos(cosA);
        float maxAngle = static_cast<float>(_fiberStep) * 5.0f * static_cast<float>(M_PI / 180.0);
        if (angle > maxAngle) {
            cv::Vec3f axis = oldNormal.cross(nextDir);
            float axisLen = static_cast<float>(cv::norm(axis));
            if (axisLen > 1e-6f) {
                axis /= axisLen;
                // Rodrigues: rotate oldNormal towards nextDir by maxAngle
                // axis ⊥ oldNormal so the (axis·v)(1-cos) term is zero
                nextDir = oldNormal * std::cos(maxAngle)
                        + axis.cross(oldNormal) * std::sin(maxAngle);
            }
        }
        // Recompute position along clamped direction so prediction doesn't wander
        nextPos = _recentPoints.back().position + nextDir * static_cast<float>(_fiberStep);
    }

    // 2. Copy old annotation plane as reference (must be a separate object —
    //    reusing the same pointer causes surfaceWillBeDeleted to nuke the
    //    reference when the annotation surface is overwritten).
    if (oldPlane) {
        auto refPlane = std::make_shared<PlaneSurface>();
        refPlane->setFromNormalAndUp(oldPlane->origin(), oldPlane->normal({}), oldPlane->basisY());
        _cstate->setSurface(kFiberReferenceSurface, refPlane);
    }

    // 3. Create new annotation plane with stable rotation.
    //    Uses oldVy as up hint — setFromNormalAndUp computes the basis directly
    //    (bypasses vxy_from_normal, no discontinuous sign flips).
    auto newPlane = std::make_shared<PlaneSurface>();
    newPlane->setFromNormalAndUp(nextPos, nextDir, oldVy);

    // 6. Register and center viewers
    _cstate->setSurface(kFiberAnnotationSurface, newPlane);
    if (_annotationViewer)
        _annotationViewer->centerOnVolumePoint(nextPos);
    if (_referenceViewer && !_recentPoints.empty())
        _referenceViewer->centerOnVolumePoint(_recentPoints.back().position);
}

std::pair<cv::Vec3f, cv::Vec3f> FiberAnnotationController::predictFromOnePoint() const
{
    const cv::Vec3f& p0 = _recentPoints[0].position;
    cv::Vec3f dir = _initialNormal;
    float len = static_cast<float>(cv::norm(dir));
    if (len > 1e-6f) dir /= len;
    else dir = cv::Vec3f(0, 0, 1);

    cv::Vec3f predicted = p0 + dir * static_cast<float>(_fiberStep);
    return {predicted, dir};
}

std::pair<cv::Vec3f, cv::Vec3f> FiberAnnotationController::predictFromTwoPoints() const
{
    const cv::Vec3f& p0 = _recentPoints[0].position;
    const cv::Vec3f& p1 = _recentPoints[1].position;
    cv::Vec3f diff = p1 - p0;
    float len = static_cast<float>(cv::norm(diff));
    cv::Vec3f dir = len > 1e-6f ? diff / len : cv::Vec3f(0, 0, 1);

    cv::Vec3f predicted = p1 + dir * static_cast<float>(_fiberStep);
    return {predicted, dir};
}

std::pair<cv::Vec3f, cv::Vec3f> FiberAnnotationController::predictFromThreeOrMore() const
{
    size_t n = _recentPoints.size();
    const cv::Vec3f& p0 = _recentPoints[n - 3].position;
    const cv::Vec3f& p1 = _recentPoints[n - 2].position;
    const cv::Vec3f& p2 = _recentPoints[n - 1].position;

    // Rotation-based extrapolation: mirror v1 through v2
    cv::Vec3f d1 = p1 - p0;
    cv::Vec3f d2 = p2 - p1;
    float len1 = static_cast<float>(cv::norm(d1));
    float len2 = static_cast<float>(cv::norm(d2));

    if (len1 < 1e-6f || len2 < 1e-6f) {
        // Degenerate — fall back to linear
        cv::Vec3f dir = len2 > 1e-6f ? d2 / len2 : cv::Vec3f(0, 0, 1);
        return {p2 + dir * static_cast<float>(_fiberStep), dir};
    }

    cv::Vec3f v1 = d1 / len1;
    cv::Vec3f v2 = d2 / len2;

    // v3 = 2*(v1·v2)*v2 - v1: same angular change from v1→v2 applied past v2.
    // Clamp: if v1·v2 < 0 (angle > 90°), fall back to linear to prevent reversal.
    float cosAngle = v1.dot(v2);
    cv::Vec3f v3;
    if (cosAngle < 0.0f) {
        v3 = v2;
    } else {
        v3 = 2.0f * cosAngle * v2 - v1;
    }
    float v3Len = static_cast<float>(cv::norm(v3));
    cv::Vec3f dir = v3Len > 1e-6f ? v3 / v3Len : v2;

    return {p2 + dir * static_cast<float>(_fiberStep), dir};
}

void FiberAnnotationController::closeAnnotationViewer()
{
    if (_annotationViewer) {
        auto* subWindow = qobject_cast<QMdiSubWindow*>(_annotationViewer->parentWidget());
        if (subWindow) {
            subWindow->close();
        }
        _annotationViewer = nullptr;
    }
    if (_referenceViewer) {
        auto* subWindow = qobject_cast<QMdiSubWindow*>(_referenceViewer->parentWidget());
        if (subWindow) {
            subWindow->close();
        }
        _referenceViewer = nullptr;
    }
    _cstate->setSurface(kFiberAnnotationSurface, nullptr);
    _cstate->setSurface(kFiberReferenceSurface, nullptr);
}
