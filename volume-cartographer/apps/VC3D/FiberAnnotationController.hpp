#pragma once

#include <QObject>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

class CState;
class VCCollection;
class CAdaptiveVolumeViewer;
#ifndef CTiledVolumeViewer
#define CTiledVolumeViewer CAdaptiveVolumeViewer
#endif
class QMdiArea;
class QMdiSubWindow;
class QElapsedTimer;
class QKeyEvent;
class PlaneSurface;
class Surface;

class FiberAnnotationController : public QObject
{
    Q_OBJECT

public:
    static constexpr int kNumViews = 2;  // 0 = ref, 1 = annotation

    enum class State {
        Idle,
        WaitingForFirstClick,
        Annotating
    };

    explicit FiberAnnotationController(CState* state,
                                       VCCollection* collection,
                                       QObject* parent = nullptr);
    ~FiberAnnotationController();

    State currentState() const { return _state; }

    void beginNewFiber();

    bool handleVolumeClick(const cv::Vec3f& vol_loc, const cv::Vec3f& normal,
                           Surface* surf, Qt::MouseButton button,
                           Qt::KeyboardModifiers modifiers);

    bool handleEscape();
    bool handleKeyPress(QKeyEvent* event);

    void setMdiArea(QMdiArea* mdiArea) { _mdiArea = mdiArea; }
    void setFiberViewer(int index, CTiledVolumeViewer* viewer);
    CTiledVolumeViewer* fiberViewer(int index) const;
    int fiberStep() const { return _fiberStep; }

    static std::string fiberSurfaceName(int index);

signals:
    void crosshairModeChanged(bool active);
    void annotationFinished(uint64_t fiberId);
    void requestFiberViewers();

public slots:
    void onAnnotationViewerClicked(cv::Vec3f vol_loc, cv::Vec3f normal,
                                   Surface* surf, Qt::MouseButton button,
                                   Qt::KeyboardModifiers modifiers);
    void onStepChanged(int step);
    void invertDirection();

private slots:
    void onAnimTick();

private:
    struct FiberPoint {
        cv::Vec3f position;
        cv::Vec3f arrivalDirection;
    };

    void addFiberPoint(const cv::Vec3f& position);
    void rebuildRecentPointsFromChain();
    void commitClickAndAdvance();
    void updatePrediction();
    void closeAnnotationViewer();

    cv::Vec3f predictDirection() const;
    std::pair<cv::Vec3f, cv::Vec3f> predictFromOnePoint() const;
    std::pair<cv::Vec3f, cv::Vec3f> predictFromTwoPoints() const;
    std::pair<cv::Vec3f, cv::Vec3f> predictFromThreeOrMore() const;

    CState* _cstate;
    VCCollection* _collection;
    QMdiArea* _mdiArea = nullptr;
    CTiledVolumeViewer* _fiberViewers[kNumViews] = {};

    State _state = State::Idle;
    uint64_t _currentFiberId = 0;
    std::string _fiberCollectionName;
    cv::Vec3f _initialNormal = {0, 0, 1};
    int _fiberStep = 50;
    bool _invertMode = false;

    // Anchor pose (the "ref" view); set on click, preserved across step changes
    bool _hasAnchor = false;
    cv::Vec3f _anchorPos = {0, 0, 0};
    cv::Vec3f _anchorNormal = {0, 0, 1};
    cv::Vec3f _anchorVy = {0, 1, 0};

    std::vector<FiberPoint> _recentPoints;

    // Animation state
    bool _animating = false;
    QElapsedTimer* _animClock = nullptr;
    std::shared_ptr<PlaneSurface> _animSavedAnnotPlane;  // restored when animation ends
    cv::Vec3f _animRefPos, _animRefNormal, _animRefVy;
    cv::Vec3f _animAnnotPos, _animAnnotNormal, _animAnnotVy;
};
