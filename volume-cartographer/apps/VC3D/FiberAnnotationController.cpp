#include "FiberAnnotationController.hpp"

#include "CState.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "adaptive/CAdaptiveVolumeViewer.hpp"

#include <QMdiArea>
#include <QMdiSubWindow>
#include <cmath>
#include <algorithm>
#include <iostream>

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

    // Initialize stable up: pick a vector perpendicular to the fiber direction.
    // The fiber direction is along the clicked slice's normal, so pick a world
    // axis that's not parallel to it.
    cv::Vec3f dir = normal;
    float dirLen = static_cast<float>(cv::norm(dir));
    if (dirLen > 1e-6f) dir /= dirLen;
    cv::Vec3f candidate = (std::abs(dir[1]) < 0.9f) ? cv::Vec3f(0,1,0) : cv::Vec3f(1,0,0);
    _stableUp = candidate - candidate.dot(dir) * dir;
    float upLen = static_cast<float>(cv::norm(_stableUp));
    if (upLen > 1e-6f) _stableUp /= upLen;

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

    updateSlicePlane(nextPos, nextDir);

    // Update reference plane: same orientation as annotation, centered on last point
    if (!_recentPoints.empty()) {
        cv::Vec3f refPos = _recentPoints.back().position;
        updateReferencePlane(refPos, nextDir);
        centerViewers(nextPos, refPos);
    }
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

    // Quadratic fit: P(t) = a*t^2 + b*t + c where t=0->p0, t=1->p1, t=2->p2
    cv::Vec3f a = 0.5f * (p2 - 2.0f * p1 + p0);
    cv::Vec3f b = (p1 - p0) - a;
    // c = p0

    // Predict at t=3
    float t = 3.0f;
    cv::Vec3f predicted_raw = a * t * t + b * t + p0;

    // Direction = P'(t) = 2*a*t + b
    cv::Vec3f dir_raw = 2.0f * a * t + b;
    float dirLen = static_cast<float>(cv::norm(dir_raw));
    cv::Vec3f dir = dirLen > 1e-6f ? dir_raw / dirLen : cv::Vec3f(0, 0, 1);

    // Normalize distance from last point to fiberStep
    cv::Vec3f fromLast = predicted_raw - p2;
    float fromLastLen = static_cast<float>(cv::norm(fromLast));
    cv::Vec3f predicted;
    if (fromLastLen > 1e-6f) {
        predicted = p2 + (fromLast / fromLastLen) * static_cast<float>(_fiberStep);
    } else {
        predicted = p2 + dir * static_cast<float>(_fiberStep);
    }

    return {predicted, dir};
}

void FiberAnnotationController::updateSlicePlane(const cv::Vec3f& center,
                                                  const cv::Vec3f& direction)
{
    if (!_annotationPlane) {
        _annotationPlane = std::make_shared<PlaneSurface>();
    }

    // Project _stableUp onto the plane orthogonal to the new direction
    // to maintain a consistent "up" across steps and prevent rotation jumps.
    float dot = _stableUp.dot(direction);
    cv::Vec3f upProj = _stableUp - dot * direction;
    float upLen = static_cast<float>(cv::norm(upProj));
    if (upLen > 1e-6f) {
        upProj /= upLen;
        _stableUp = upProj;
    } else {
        // _stableUp is (anti-)parallel to direction — pick a new one
        cv::Vec3f fallback = (std::abs(direction[1]) < 0.9f)
                             ? cv::Vec3f(0, 1, 0) : cv::Vec3f(1, 0, 0);
        upProj = fallback - fallback.dot(direction) * direction;
        cv::normalize(upProj, upProj);
        _stableUp = upProj;
    }

    // Reset rotation before setting normal so update() computes the true
    // default basis vectors (otherwise the old rotation contaminates them).
    _annotationPlane->setInPlaneRotation(0);
    _annotationPlane->setOrigin(center);
    _annotationPlane->setNormal(direction);

    // Signed angle from default basisY to _stableUp around the normal.
    // Uses cross product to get correct sign regardless of basis handedness.
    cv::Vec3f defaultVy = _annotationPlane->basisY();
    cv::Vec3f cross = defaultVy.cross(_stableUp);
    float sinA = cross.dot(direction);  // project onto normal for signed sin
    float cosA = defaultVy.dot(_stableUp);
    float angle = std::atan2(sinA, cosA);

    _annotationPlane->setInPlaneRotation(angle);
    _cstate->setSurface(kFiberAnnotationSurface, _annotationPlane);
}

void FiberAnnotationController::updateReferencePlane(const cv::Vec3f& center,
                                                      const cv::Vec3f& direction)
{
    if (!_referencePlane) {
        _referencePlane = std::make_shared<PlaneSurface>();
    }
    _referencePlane->setInPlaneRotation(0);
    _referencePlane->setOrigin(center);
    _referencePlane->setNormal(direction);

    cv::Vec3f defaultVy = _referencePlane->basisY();
    cv::Vec3f cross = defaultVy.cross(_stableUp);
    float sinA = cross.dot(direction);
    float cosA = defaultVy.dot(_stableUp);
    float angle = std::atan2(sinA, cosA);
    _referencePlane->setInPlaneRotation(angle);

    _cstate->setSurface(kFiberReferenceSurface, _referencePlane);
}

void FiberAnnotationController::centerViewers(const cv::Vec3f& annotationCenter,
                                               const cv::Vec3f& referenceCenter)
{
    if (_annotationViewer)
        _annotationViewer->centerOnVolumePoint(annotationCenter);
    if (_referenceViewer)
        _referenceViewer->centerOnVolumePoint(referenceCenter);
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
    _annotationPlane.reset();
    _referencePlane.reset();
}
