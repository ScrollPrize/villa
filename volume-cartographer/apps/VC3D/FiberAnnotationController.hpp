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
class PlaneSurface;
class Surface;

class FiberAnnotationController : public QObject
{
    Q_OBJECT

public:
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

    // Called by CWindow::onVolumeClicked for first-point pick.
    // Returns true if this controller consumed the click.
    bool handleVolumeClick(const cv::Vec3f& vol_loc, const cv::Vec3f& normal,
                           Surface* surf, Qt::MouseButton button,
                           Qt::KeyboardModifiers modifiers);

    // Called when Escape is pressed. Returns true if consumed.
    bool handleEscape();

    void setMdiArea(QMdiArea* mdiArea) { _mdiArea = mdiArea; }
    void setAnnotationViewer(CTiledVolumeViewer* viewer) { _annotationViewer = viewer; }
    void setReferenceViewer(CTiledVolumeViewer* viewer) { _referenceViewer = viewer; }
    int fiberStep() const { return _fiberStep; }

signals:
    void crosshairModeChanged(bool active);
    void annotationFinished(uint64_t fiberId);
    void requestAnnotationViewer(const std::string& surfaceName, const QString& title);
    void requestReferenceViewer(const std::string& surfaceName, const QString& title);

public slots:
    void onAnnotationViewerClicked(cv::Vec3f vol_loc, cv::Vec3f normal,
                                   Surface* surf, Qt::MouseButton button,
                                   Qt::KeyboardModifiers modifiers);
    void onStepChanged(int step);

private:
    struct FiberPoint {
        cv::Vec3f position;
        cv::Vec3f arrivalDirection;
    };

    void addFiberPoint(const cv::Vec3f& position);
    void advanceToNextPrediction();
    void closeAnnotationViewer();

    std::pair<cv::Vec3f, cv::Vec3f> predictFromOnePoint() const;
    std::pair<cv::Vec3f, cv::Vec3f> predictFromTwoPoints() const;
    std::pair<cv::Vec3f, cv::Vec3f> predictFromThreeOrMore() const;

    void updateSlicePlane(const cv::Vec3f& center, const cv::Vec3f& direction);

    CState* _cstate;
    VCCollection* _collection;
    QMdiArea* _mdiArea = nullptr;
    CTiledVolumeViewer* _annotationViewer = nullptr;
    CTiledVolumeViewer* _referenceViewer = nullptr;
    std::shared_ptr<PlaneSurface> _annotationPlane;
    std::shared_ptr<PlaneSurface> _referencePlane;

    State _state = State::Idle;
    uint64_t _currentFiberId = 0;
    std::string _fiberCollectionName;
    cv::Vec3f _initialNormal = {0, 0, 1};
    int _fiberStep = 50;

    std::vector<FiberPoint> _recentPoints;
    cv::Vec3f _stableUp = {0, 1, 0};  // maintained across steps to prevent in-plane rotation jumps

    void updateReferencePlane(const cv::Vec3f& center, const cv::Vec3f& direction);
    void centerViewers(const cv::Vec3f& annotationCenter, const cv::Vec3f& referenceCenter);

    static constexpr const char* kFiberAnnotationSurface = "fiber_annotation_plane";
    static constexpr const char* kFiberReferenceSurface = "fiber_reference_plane";
};
