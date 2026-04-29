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
#include <algorithm>
#include <cmath>
#include <limits>

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
    _invertMode = false;
    _hasAnchor = false;
    addFiberPoint(vol_loc);

    _state = State::Annotating;
    emit crosshairModeChanged(false);

    commitClickAndAdvance();
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
        _invertMode = false;
        _hasAnchor = false;
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
    commitClickAndAdvance();
}

void FiberAnnotationController::addFiberPoint(const cv::Vec3f& position)
{
    // When inverted, prepend by giving the new point a creation_time earlier
    // than every existing point in the chain.
    int64_t adjustedTime = -1;
    if (_invertMode) {
        auto chain = _collection->getPoints(_fiberCollectionName);
        if (!chain.empty()) {
            int64_t minTime = std::numeric_limits<int64_t>::max();
            for (const auto& p : chain) minTime = std::min(minTime, p.creation_time);
            adjustedTime = minTime - 1;
        }
    }

    ColPoint pt = _collection->addPoint(_fiberCollectionName, position);
    if (adjustedTime != -1) pt.creation_time = adjustedTime;
    pt.winding_annotation = 0.0f;
    _collection->updatePoint(pt);

    rebuildRecentPointsFromChain();
}

void FiberAnnotationController::rebuildRecentPointsFromChain()
{
    auto pts = _collection->getPoints(_fiberCollectionName);
    std::sort(pts.begin(), pts.end(),
              [](const ColPoint& a, const ColPoint& b) {
                  return a.creation_time < b.creation_time;
              });
    if (_invertMode) std::reverse(pts.begin(), pts.end());

    _recentPoints.clear();
    if (pts.empty()) return;

    int total = static_cast<int>(pts.size());
    int n = std::min(3, total);
    int start = total - n;
    for (int i = start; i < total; ++i) {
        FiberPoint fp;
        fp.position = pts[i].p;
        if (i == 0) {
            fp.arrivalDirection = _initialNormal;
        } else {
            cv::Vec3f d = pts[i].p - pts[i - 1].p;
            float len = static_cast<float>(cv::norm(d));
            fp.arrivalDirection = len > 1e-6f ? d / len : _initialNormal;
        }
        _recentPoints.push_back(fp);
    }
}

void FiberAnnotationController::onStepChanged(int step)
{
    _fiberStep = step;
    if (_state == State::Annotating && _hasAnchor && !_recentPoints.empty()) {
        // Keep the ref point fixed; rescale the prediction.
        updatePrediction();
    }
}

void FiberAnnotationController::invertDirection()
{
    if (_state != State::Annotating) return;
    if (_recentPoints.empty()) return;

    _invertMode = !_invertMode;
    _initialNormal = -_initialNormal;
    rebuildRecentPointsFromChain();
    if (_recentPoints.empty()) return;

    // Anchor the slice with the (now-flipped) initial normal — the same plane
    // orientation the user saw when they first started the fiber. The chain-
    // geometry extrapolation `predictDir` would otherwise drift from the
    // initial slice on curving chains; using `_initialNormal` keeps the ref
    // view recognisable. From the next click onward the rotation clamp lets
    // the normal evolve toward chain geometry as usual.
    cv::Vec3f anchorNormal = _initialNormal;
    float anchorNormalLen = static_cast<float>(cv::norm(anchorNormal));
    if (anchorNormalLen > 1e-6f) anchorNormal /= anchorNormalLen;
    else anchorNormal = cv::Vec3f(0, 0, 1);

    auto oldAnnot = std::dynamic_pointer_cast<PlaneSurface>(
        _cstate->surface(fiberSurfaceName(kNumViews - 1)));
    cv::Vec3f baseVy = oldAnnot ? oldAnnot->basisY() : cv::Vec3f(0, 1, 0);

    cv::Vec3f anchorVy = baseVy - baseVy.dot(anchorNormal) * anchorNormal;
    float vlen = static_cast<float>(cv::norm(anchorVy));
    if (vlen > 1e-6f) {
        anchorVy /= vlen;
    } else {
        cv::Vec3f hint = std::abs(anchorNormal.dot(cv::Vec3f(0, 1, 0))) < 0.9f
                            ? cv::Vec3f(0, 1, 0) : cv::Vec3f(1, 0, 0);
        PlaneSurface tmp;
        tmp.setFromNormalAndUp({0, 0, 0}, anchorNormal, hint);
        anchorVy = tmp.basisY();
    }

    _anchorPos = _recentPoints.back().position;
    _anchorNormal = anchorNormal;
    _anchorVy = anchorVy;
    _hasAnchor = true;

    auto refPlane = std::make_shared<PlaneSurface>();
    refPlane->setFromNormalAndUp(_anchorPos, _anchorNormal, _anchorVy);
    _cstate->setSurface(fiberSurfaceName(0), refPlane);

    // Invert teleports to the other end of the chain — the ref viewer's
    // camera must follow, otherwise it stays parked over the old endpoint.
    if (_fiberViewers[0])
        _fiberViewers[0]->centerOnVolumePoint(_anchorPos);

    updatePrediction();
}

void FiberAnnotationController::commitClickAndAdvance()
{
    // Snapshot current annotation plane as the new anchor (this is the pose
    // the user just clicked on). On first click no annotation plane exists yet,
    // so seed from the click point + initial normal.
    auto oldPlane = std::dynamic_pointer_cast<PlaneSurface>(
        _cstate->surface(fiberSurfaceName(kNumViews - 1)));
    if (oldPlane) {
        _anchorPos = oldPlane->origin();
        _anchorNormal = oldPlane->normal({});
        _anchorVy = oldPlane->basisY();
    } else {
        _anchorPos = _recentPoints.empty() ? cv::Vec3f(0, 0, 0)
                                           : _recentPoints.back().position;
        _anchorNormal = _initialNormal;
        cv::Vec3f hint = std::abs(_initialNormal.dot(cv::Vec3f(0, 1, 0))) < 0.9f
                            ? cv::Vec3f(0, 1, 0) : cv::Vec3f(1, 0, 0);
        PlaneSurface tmp;
        tmp.setFromNormalAndUp(_anchorPos, _anchorNormal, hint);
        _anchorVy = tmp.basisY();
    }
    _hasAnchor = true;

    auto refPlane = std::make_shared<PlaneSurface>();
    refPlane->setFromNormalAndUp(_anchorPos, _anchorNormal, _anchorVy);
    _cstate->setSurface(fiberSurfaceName(0), refPlane);

    updatePrediction();
}

void FiberAnnotationController::updatePrediction()
{
    if (!_hasAnchor || _recentPoints.empty()) return;

    cv::Vec3f nextDir = predictDirection();

    // Clamp normal rotation to 5°/voxel of the current step.
    float cosA = std::clamp(_anchorNormal.dot(nextDir), -1.0f, 1.0f);
    float angle = std::acos(cosA);
    float maxAngle = static_cast<float>(_fiberStep) * 5.0f * static_cast<float>(M_PI / 180.0);
    if (angle > maxAngle) {
        cv::Vec3f axis = _anchorNormal.cross(nextDir);
        float axisLen = static_cast<float>(cv::norm(axis));
        if (axisLen > 1e-6f) {
            axis /= axisLen;
            nextDir = _anchorNormal * std::cos(maxAngle)
                    + axis.cross(_anchorNormal) * std::sin(maxAngle);
        } else {
            nextDir = _anchorNormal;
        }
    }

    cv::Vec3f nextPos = _recentPoints.back().position
                      + nextDir * static_cast<float>(_fiberStep);

    cv::Vec3f annotVy = _anchorVy - _anchorVy.dot(nextDir) * nextDir;
    float vlen = static_cast<float>(cv::norm(annotVy));
    if (vlen > 1e-6f) annotVy /= vlen;
    else annotVy = _anchorVy;

    auto plane = std::make_shared<PlaneSurface>();
    plane->setFromNormalAndUp(nextPos, nextDir, annotVy);
    _cstate->setSurface(fiberSurfaceName(kNumViews - 1), plane);

    if (_fiberViewers[kNumViews - 1])
        _fiberViewers[kNumViews - 1]->centerOnVolumePoint(nextPos);
}

cv::Vec3f FiberAnnotationController::predictDirection() const
{
    if (_recentPoints.size() == 1) return predictFromOnePoint().second;
    if (_recentPoints.size() == 2) return predictFromTwoPoints().second;
    return predictFromThreeOrMore().second;
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

    // Triangle wave: 1→0→1 over kDurationMs (start at current, go to ref, back)
    float frac = static_cast<float>(elapsed) / static_cast<float>(kDurationMs);
    float t = std::abs(2.0f * frac - 1.0f);

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
    _invertMode = false;
    _hasAnchor = false;

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
