#pragma once

#include <QMainWindow>
#include <QMetaObject>
#include <QPointer>

#include <memory>
#include <functional>
#include <map>
#include <limits>
#include <string>
#include <vector>
#include <utility>

#include "LineAnnotationGeneratedViews.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include <opencv2/core/mat.hpp>

class CState;
class QComboBox;
class QGraphicsPathItem;
class QLabel;
class QMdiArea;
class QMdiSubWindow;
class QPoint;
class QPushButton;
class QResizeEvent;
class QVariantAnimation;
class QVBoxLayout;
class QWheelEvent;
class ViewerManager;
class PlaneSurface;
class QuadSurface;

class LineAnnotationDialog : public QMainWindow
{
    Q_OBJECT

public:
    enum class InitialDirectionMode {
        Sideways,
        ZInOut,
    };
    enum class ReoptimizationMode {
        AutoReoptimize,
        NoOptimization,
    };
    enum class ShiftScrollMode {
        AlongLine,
        StraightNormal,
    };
    using GeneratedControlPointContextResult =
        vc3d::line_annotation::GeneratedControlPointContextResult;

    struct Pane {
        std::string surfaceName;
        QPointer<CChunkedVolumeViewer> viewer;
        QPointer<QMdiSubWindow> subWindow;
    };

    struct FastStripOverlayItems {
        QPointer<CChunkedVolumeViewer> viewer;
        std::string surfaceName;
        QGraphicsPathItem* currentLine = nullptr;
    };

    struct FastCurrentCutOverlayItems {
        QPointer<CChunkedVolumeViewer> viewer;
        QGraphicsPathItem* centerPoint = nullptr;
        QGraphicsPathItem* controlPoints = nullptr;
        QGraphicsPathItem* seedPoints = nullptr;
    };

    using GeneratedOverlay = vc3d::line_annotation::GeneratedOverlay;
    using GeneratedViews = vc3d::line_annotation::GeneratedViews;
    using VolumeSelectorFactory = std::function<QWidget*(QWidget*)>;

    explicit LineAnnotationDialog(ViewerManager* viewerManager,
                                  VolumeSelectorFactory volumeSelectorFactory = {},
                                  QWidget* parent = nullptr);

    CChunkedVolumeViewer* addPane(const std::string& surfaceName,
                                  const QString& title,
                                  const CChunkedVolumeViewer::CameraState& camera);
    bool setGeneratedRows(
        const std::vector<std::vector<std::pair<std::string, QString>>>& rows,
        const CChunkedVolumeViewer::CameraState& camera,
        const std::map<std::string, GeneratedOverlay>& overlays = {});
    bool setGeneratedLineViews(const GeneratedViews& views,
                               const CChunkedVolumeViewer::CameraState& camera);
    GeneratedControlPointContextResult showGeneratedControlPointContextMenu(
        const std::string& surfaceName,
        CChunkedVolumeViewer* viewer,
        const QPointF& scenePoint,
        const QPoint& globalPos);
    const std::vector<Pane>& panes() const { return _panes; }
    InitialDirectionMode initialDirectionMode() const;
    ReoptimizationMode reoptimizationMode() const;
    ShiftScrollMode shiftScrollMode() const;
    void setGeneratedControlPoints(std::vector<GeneratedOverlay::ControlPointMarker> controlPoints);
    void setGeneratedPredSnapPoints(std::vector<GeneratedOverlay::PredSnapMarker> predSnapPoints);
    void setOptimizationBusy(bool busy);

signals:
    void paneClosed(const std::string& surfaceName);
    void lineSeedRequested(const std::string& surfaceName, cv::Vec3f volumePoint, QPointF scenePoint);
    void generatedControlPointRequested(const std::string& surfaceName,
                                        cv::Vec3f volumePoint,
                                        double linePosition);
    void generatedControlPointDeleteRequested(const std::string& surfaceName,
                                              double linePosition,
                                              cv::Vec3f volumePoint);
    void generatedPredSnapPointRequested(const std::string& surfaceName,
                                         cv::Vec3f volumePoint);
    void showAsMeshRequested();
    void fullOptimizationRequested();
    void reoptimizationModeChanged(LineAnnotationDialog::ReoptimizationMode mode);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void bindPaneInteractions(const std::string& surfaceName,
                              CChunkedVolumeViewer* viewer,
                              bool seedPlacementEnabled);
    void connectGeneratedOverlayRefresh(CChunkedVolumeViewer* viewer);
    void clearGeneratedOverlayRefreshConnections();
    void setGeneratedOverlay(const std::string& surfaceName,
                             CChunkedVolumeViewer* viewer,
                             const GeneratedOverlay& overlay);
    void applyGeneratedOverlay(const std::string& surfaceName,
                               CChunkedVolumeViewer* viewer,
                               const GeneratedOverlay& overlay);
    double linePositionFromStripScene(CChunkedVolumeViewer* viewer, const QPointF& scenePoint) const;
    void setCurrentLinePosition(double position,
                                bool updateBottomCuts = true,
                                bool updateCurrentCutOverlay = true);
    void cancelControlPointPreviewAnimation();
    void jumpToPreviousControlPoint();
    void jumpToNextControlPoint();
    void previewClosestControlPoint();
    bool shiftCurrentLinePositionByScrollSteps(int steps);
    bool shiftCurrentCutPlaneStraightByScrollSteps(int steps);
    bool shiftBottomSlicesByScrollSteps(int steps);
    bool scaleBottomSliceLineStepByScrollSteps(int steps);
    bool handleBottomSliceStepWheel(QWheelEvent* event);
    void handleShiftScrollModeChanged();
    void setCurrentCutFollowsStripMouse(bool follows);
    double snappedControlPointPosition(double position) const;
    void rebuildGeneratedStaticStripOverlays();
    void rebuildGeneratedDynamicOverlays(bool updateCurrentCutOverlay = true);
    void updateGeneratedDynamicOverlaysFast(bool updateCurrentCutOverlay);
    void clearFastGeneratedOverlayItemRefs();
    void rebuildGeneratedOverlays();
    void installGeneratedViewShortcuts();
    void resetGeneratedViews();
    bool toggleCurrentCutFollowFromKeyboard();
    bool rotateCurrentCut(vc3d::line_annotation::GeneratedCutRotationAxis axis, float radians);
    cv::Vec3f currentCutViewerCenterVolumePoint() const;
    void captureInitialGeneratedViewState();
    void restoreInitialGeneratedViewerCameras();
    void applyOverlayForViewer(const std::string& overlayKey,
                               CChunkedVolumeViewer* viewer,
                               const GeneratedOverlay& overlay);
    void clearControlPointContextPreview(const std::string& surfaceName,
                                         CChunkedVolumeViewer* viewer);
    GeneratedOverlay staticStripOverlay() const;
    GeneratedOverlay zSliceOverlay(double linePosition,
                                   bool emphasized,
                                   CChunkedVolumeViewer* viewer,
                                   PlaneSurface* plane) const;
    cv::Vec3f interpolatedLinePoint(double linePosition) const;
    cv::Vec3f interpolatedLineTangent(double linePosition) const;
    cv::Vec3f interpolatedLineUp(double linePosition, const cv::Vec3f& tangent) const;
    bool updatePlaneSurface(PlaneSurface* plane, double linePosition) const;
    void updateBottomSliceStepLabel();
    QPointF stripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                     QuadSurface* surface,
                                     double linePosition) const;
    bool handleKeyPress(QKeyEvent* event);
    void updateOptimizationOverlayGeometry();

    ViewerManager* _viewerManager = nullptr;
    QVBoxLayout* _layout = nullptr;
    QComboBox* _initialDirectionCombo = nullptr;
    QComboBox* _reoptimizationCombo = nullptr;
    QComboBox* _shiftScrollCombo = nullptr;
    QLabel* _sliceStepLabel = nullptr;
    QLabel* _bottomSliceStepLabel = nullptr;
    QPushButton* _showAsMeshButton = nullptr;
    QPushButton* _fullOptimizationButton = nullptr;
    QPushButton* _resetViewsButton = nullptr;
    QPointer<QWidget> _optimizationOverlay;
    QMdiArea* _mdiArea = nullptr;
    std::vector<Pane> _panes;
    bool _suppressPaneClosed = false;

    QWidget* _generatedTopWidget = nullptr;
    std::vector<QPointer<QWidget>> _generatedContainers;
    std::vector<QMetaObject::Connection> _generatedOverlayRefreshConnections;
    std::vector<FastStripOverlayItems> _fastStripOverlayItems;
    FastCurrentCutOverlayItems _fastCurrentCutOverlayItems;
    QPointer<CChunkedVolumeViewer> _currentCutViewer;
    std::vector<QPointer<CChunkedVolumeViewer>> _stripViewers;
    GeneratedViews _generatedViews;
    bool _hasGeneratedViews = false;
    double _currentLinePosition = 0.0;
    double _bottomCenterPosition = 0.0;
    double _bottomSliceLineStep = 10.0;
    double _initialCurrentLinePosition = 0.0;
    double _initialBottomCenterPosition = 0.0;
    double _initialBottomSliceLineStep = 10.0;
    int _bottomSliceStepWheelAccum = 0;
    bool _currentCutFollowsStripMouse = true;
    cv::Matx33f _currentCutManualRotation = cv::Matx33f::eye();
    bool _currentCutManualRotationActive = false;
    bool _currentCutStraightOffsetActive = false;
    bool _generatedOverlayRefreshQueued = false;
    vc3d::line_annotation::GeneratedControlPointLinePositionIndex _generatedControlIndex;
    QPointer<QVariantAnimation> _controlPointPreviewAnimation;
    bool _haveInitialCurrentCutCamera = false;
    CChunkedVolumeViewer::CameraState _initialCurrentCutCamera;
    std::vector<CChunkedVolumeViewer::CameraState> _initialStripCameras;
};
