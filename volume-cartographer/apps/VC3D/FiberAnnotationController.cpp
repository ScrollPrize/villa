#include "FiberAnnotationController.hpp"

#include "CState.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "adaptive/CAdaptiveVolumeViewer.hpp"

#include <QMdiArea>
#include <QMdiSubWindow>
#include <QElapsedTimer>
#include <QKeyEvent>
#include <QTimer>
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

std::string FiberAnnotationController::fiberSurfaceName(int index)
{
    return "fiber_slice_" + std::to_string(index);
}

void FiberAnnotationController::setFiberViewer(int index, CTiledVolumeViewer* viewer)
{
    if (index >= 0 && index < kNumViews)
        _fiberViewers[index] = viewer;
}

CTiledVolumeViewer* FiberAnnotationController::fiberViewer(int index) const
{
    return (index >= 0 && index < kNumViews) ? _fiberViewers[index] : nullptr;
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

    _fiberCollectionName = _collection->generateNewCollectionName("fiber");
    uint64_t colId = _collection->addCollection(_fiberCollectionName);
    _currentFiberId = colId;

    _collection->setCollectionTag(colId, "fiber", "true");

    CollectionMetadata meta;
    meta.absolute_winding_number = false;
    _collection->setCollectionMetadata(colId, meta);

    _initialNormal = normal;
    addFiberPoint(vol_loc);

    _state = State::Annotating;
    emit crosshairModeChanged(false);

    advanceToNextPrediction();
    emit requestFiberViewers();

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

    ColPoint updated = pt;
    updated.winding_annotation = 0.0f;
    _collection->updatePoint(updated);

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

    // Get old annotation plane (index 5) for basisY and normal
    auto oldPlane = std::dynamic_pointer_cast<PlaneSurface>(
        _cstate->surface(fiberSurfaceName(kNumViews - 1)));
    cv::Vec3f oldVy = oldPlane ? oldPlane->basisY() : cv::Vec3f(0, 1, 0);
    cv::Vec3f oldNormal = oldPlane ? oldPlane->normal({}) : _initialNormal;

    // Clamp normal rotation to 5°/voxel
    if (oldPlane) {
        float cosA = std::clamp(oldNormal.dot(nextDir), -1.0f, 1.0f);
        float angle = std::acos(cosA);
        float maxAngle = static_cast<float>(_fiberStep) * 5.0f * static_cast<float>(M_PI / 180.0);
        if (angle > maxAngle) {
            cv::Vec3f axis = oldNormal.cross(nextDir);
            float axisLen = static_cast<float>(cv::norm(axis));
            if (axisLen > 1e-6f) {
                axis /= axisLen;
                nextDir = oldNormal * std::cos(maxAngle)
                        + axis.cross(oldNormal) * std::sin(maxAngle);
            }
        }
        nextPos = _recentPoints.back().position + nextDir * static_cast<float>(_fiberStep);
    }

    // Ref view (index 0): exact clone of previous annotation view
    if (oldPlane) {
        auto refPlane = std::make_shared<PlaneSurface>();
        refPlane->setFromNormalAndUp(oldPlane->origin(), oldNormal, oldVy);
        _cstate->setSurface(fiberSurfaceName(0), refPlane);
    }

    // Annotation state
    cv::Vec3f annotVy = oldVy - oldVy.dot(nextDir) * nextDir;
    float annotVyLen = static_cast<float>(cv::norm(annotVy));
    if (annotVyLen > 1e-6f) annotVy /= annotVyLen;
    else annotVy = oldVy;

    cv::Vec3f refPos = oldPlane ? oldPlane->origin() : _recentPoints.back().position;

    // Interpolated views (index 1..4) and annotation view (index 5)
    for (int i = 1; i < kNumViews; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(kNumViews - 1);

        cv::Vec3f pos = refPos * (1.0f - t) + nextPos * t;

        cv::Vec3f n = oldNormal * (1.0f - t) + nextDir * t;
        float nLen = static_cast<float>(cv::norm(n));
        if (nLen > 1e-6f) n /= nLen;

        cv::Vec3f up = oldVy * (1.0f - t) + annotVy * t;
        float upLen = static_cast<float>(cv::norm(up));
        if (upLen > 1e-6f) up /= upLen;

        auto plane = std::make_shared<PlaneSurface>();
        plane->setFromNormalAndUp(pos, n, up);
        _cstate->setSurface(fiberSurfaceName(i), plane);

        if (_fiberViewers[i])
            _fiberViewers[i]->centerOnVolumePoint(pos);
    }
}

std::pair<cv::Vec3f, cv::Vec3f> FiberAnnotationController::predictFromOnePoint() const
{
    const cv::Vec3f& p0 = _recentPoints[0].position;
    cv::Vec3f dir = _initialNormal;
    float len = static_cast<float>(cv::norm(dir));
    if (len > 1e-6f) dir /= len;
    else dir = cv::Vec3f(0, 0, 1);

    return {p0 + dir * static_cast<float>(_fiberStep), dir};
}

std::pair<cv::Vec3f, cv::Vec3f> FiberAnnotationController::predictFromTwoPoints() const
{
    const cv::Vec3f& p0 = _recentPoints[0].position;
    const cv::Vec3f& p1 = _recentPoints[1].position;
    cv::Vec3f diff = p1 - p0;
    float len = static_cast<float>(cv::norm(diff));
    cv::Vec3f dir = len > 1e-6f ? diff / len : cv::Vec3f(0, 0, 1);

    return {p1 + dir * static_cast<float>(_fiberStep), dir};
}

std::pair<cv::Vec3f, cv::Vec3f> FiberAnnotationController::predictFromThreeOrMore() const
{
    size_t n = _recentPoints.size();
    const cv::Vec3f& p0 = _recentPoints[n - 3].position;
    const cv::Vec3f& p1 = _recentPoints[n - 2].position;
    const cv::Vec3f& p2 = _recentPoints[n - 1].position;

    cv::Vec3f d1 = p1 - p0;
    cv::Vec3f d2 = p2 - p1;
    float len1 = static_cast<float>(cv::norm(d1));
    float len2 = static_cast<float>(cv::norm(d2));

    if (len1 < 1e-6f || len2 < 1e-6f) {
        cv::Vec3f dir = len2 > 1e-6f ? d2 / len2 : cv::Vec3f(0, 0, 1);
        return {p2 + dir * static_cast<float>(_fiberStep), dir};
    }

    cv::Vec3f v1 = d1 / len1;
    cv::Vec3f v2 = d2 / len2;

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

bool FiberAnnotationController::handleKeyPress(QKeyEvent* event)
{
    if (event->key() != Qt::Key_J) return false;
    if (_state != State::Annotating) return false;
    if (_animating) return false;

    // Snapshot ref (index 0) and annotation (index 5) endpoints
    auto refSurf = std::dynamic_pointer_cast<PlaneSurface>(
        _cstate->surface(fiberSurfaceName(0)));
    auto annotSurf = std::dynamic_pointer_cast<PlaneSurface>(
        _cstate->surface(fiberSurfaceName(kNumViews - 1)));
    if (!refSurf || !annotSurf) return false;

    _animRefPos = refSurf->origin();
    _animRefNormal = refSurf->normal({});
    _animRefVy = refSurf->basisY();
    _animAnnotPos = annotSurf->origin();
    _animAnnotNormal = annotSurf->normal({});
    _animAnnotVy = annotSurf->basisY();

    // Save the original annotation plane to restore after animation
    _animSavedAnnotPlane = annotSurf;

    _animating = true;
    if (!_animClock) _animClock = new QElapsedTimer();
    _animClock->start();

    onAnimTick();
    event->accept();
    return true;
}

void FiberAnnotationController::onAnimTick()
{
    if (!_animating || !_animClock) return;

    qint64 elapsed = _animClock->elapsed();
    constexpr qint64 kDurationMs = 1000;

    auto* viewer = _fiberViewers[kNumViews - 1];
    if (!viewer) { _animating = false; return; }

    if (elapsed >= kDurationMs) {
        // Done — restore original annotation plane
        _animating = false;
        if (_animSavedAnnotPlane) {
            _cstate->setSurface(fiberSurfaceName(kNumViews - 1), _animSavedAnnotPlane);
            viewer->centerOnVolumePoint(_animSavedAnnotPlane->origin());
            _animSavedAnnotPlane.reset();
        }
        return;
    }

    // Triangle wave: 0→1→0 over kDurationMs
    float frac = static_cast<float>(elapsed) / static_cast<float>(kDurationMs);
    float t = 1.0f - std::abs(2.0f * frac - 1.0f);

    // Interpolate
    cv::Vec3f pos = _animRefPos * (1.0f - t) + _animAnnotPos * t;

    cv::Vec3f n = _animRefNormal * (1.0f - t) + _animAnnotNormal * t;
    float nLen = static_cast<float>(cv::norm(n));
    if (nLen > 1e-6f) n /= nLen;

    cv::Vec3f up = _animRefVy * (1.0f - t) + _animAnnotVy * t;
    float upLen = static_cast<float>(cv::norm(up));
    if (upLen > 1e-6f) up /= upLen;

    auto plane = std::make_shared<PlaneSurface>();
    plane->setFromNormalAndUp(pos, n, up);
    _cstate->setSurface(fiberSurfaceName(kNumViews - 1), plane);
    viewer->centerOnVolumePoint(pos);

    // Schedule next frame on next event loop tick
    QTimer::singleShot(0, this, &FiberAnnotationController::onAnimTick);
}

void FiberAnnotationController::closeAnnotationViewer()
{
    _animating = false;
    delete _animClock;
    _animClock = nullptr;

    for (int i = 0; i < kNumViews; ++i) {
        if (_fiberViewers[i]) {
            auto* subWindow = qobject_cast<QMdiSubWindow*>(_fiberViewers[i]->parentWidget());
            if (subWindow) {
                subWindow->close();
            }
            _fiberViewers[i] = nullptr;
        }
        _cstate->setSurface(fiberSurfaceName(i), nullptr);
    }
}
