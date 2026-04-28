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

    // 1. Get old annotation plane's basisY (the "up" the user was seeing)
    auto oldPlane = std::dynamic_pointer_cast<PlaneSurface>(_cstate->surface(kFiberAnnotationSurface));
    cv::Vec3f oldVy = oldPlane ? oldPlane->basisY() : cv::Vec3f(0, 1, 0);

    // 2. Copy old annotation plane as reference (must be a separate object —
    //    reusing the same pointer causes surfaceWillBeDeleted to nuke the
    //    reference when the annotation surface is overwritten).
    if (oldPlane) {
        auto refPlane = std::make_shared<PlaneSurface>(oldPlane->origin(), oldPlane->normal({}));
        refPlane->setInPlaneRotation(oldPlane->inPlaneRotation());
        _cstate->setSurface(kFiberReferenceSurface, refPlane);
    }

    // 3. Create new annotation plane (gets default basis from vxy_from_normal)
    auto newPlane = std::make_shared<PlaneSurface>(nextPos, nextDir);

    // 4. Project old basisY onto new plane to get target "up"
    float dot = oldVy.dot(nextDir);
    cv::Vec3f projUp = oldVy - dot * nextDir;
    float projLen = static_cast<float>(cv::norm(projUp));
    if (projLen > 1e-6f) {
        projUp /= projLen;

        // 5. Signed angle from new default basisY to projected old basisY
        cv::Vec3f defaultVy = newPlane->basisY();
        cv::Vec3f cross = defaultVy.cross(projUp);
        float sinA = cross.dot(nextDir);
        float cosA = defaultVy.dot(projUp);
        newPlane->setInPlaneRotation(std::atan2(sinA, cosA));
    }

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
