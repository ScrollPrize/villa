#pragma once

#include <QMainWindow>
#include <QMetaObject>
#include <QPointer>

#include <memory>
#include <map>
#include <limits>
#include <string>
#include <vector>
#include <utility>

#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include <opencv2/core/mat.hpp>

class CState;
class QComboBox;
class QLabel;
class QMdiArea;
class QMdiSubWindow;
class QPoint;
class QPushButton;
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
    enum class GeneratedControlPointContextResult {
        None,
        Handled,
        NewLineAnnotationRequested,
    };

    struct Pane {
        std::string surfaceName;
        QPointer<CChunkedVolumeViewer> viewer;
        QPointer<QMdiSubWindow> subWindow;
    };

    struct GeneratedOverlay {
        struct ControlPointMarker {
            cv::Vec3f point{std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN()};
            double linePosition = std::numeric_limits<double>::quiet_NaN();
            bool isSeed = false;
        };

        std::vector<cv::Vec3f> linePoints;
        cv::Vec3f seedPoint{std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN()};
        cv::Vec3f pointMarker{std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN()};
        int seedLineIndex = -1;
        std::vector<double> markerLinePositions;
        std::vector<ControlPointMarker> controlPoints;
        double currentLinePosition = std::numeric_limits<double>::quiet_NaN();
        bool emphasizedPointMarker = false;
        bool useSurfaceCenterLine = false;
    };

    struct GeneratedViews {
        std::string lineSurfaceName;
        QString lineSurfaceTitle;
        std::string lineSideSliceName;
        QString lineSideSliceTitle;
        std::string currentCutName;
        std::shared_ptr<PlaneSurface> currentCutSurface;
        std::vector<std::pair<std::string, std::shared_ptr<PlaneSurface>>> bottomCutSurfaces;
        std::vector<cv::Vec3f> linePoints;
        std::vector<cv::Vec3f> lineUpVectors;
        cv::Vec3f seedPoint{std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN()};
        int seedLineIndex = -1;
        int initialCenterIndex = 0;
        std::vector<GeneratedOverlay::ControlPointMarker> controlPoints;
    };

    explicit LineAnnotationDialog(ViewerManager* viewerManager, QWidget* parent = nullptr);

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

signals:
    void paneClosed(const std::string& surfaceName);
    void lineSeedRequested(const std::string& surfaceName, cv::Vec3f volumePoint, QPointF scenePoint);
    void generatedControlPointRequested(const std::string& surfaceName,
                                        cv::Vec3f volumePoint,
                                        double linePosition);
    void generatedControlPointDeleteRequested(const std::string& surfaceName,
                                              double linePosition,
                                              cv::Vec3f volumePoint);
    void showAsMeshRequested();
    void fullOptimizationRequested();

protected:
    void keyPressEvent(QKeyEvent* event) override;
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
    void setCurrentLinePosition(double position);
    bool shiftCurrentLinePositionByScrollSteps(int steps);
    bool shiftBottomSlicesByScrollSteps(int steps);
    bool scaleBottomSliceLineStepByScrollSteps(int steps);
    bool handleBottomSliceStepWheel(QWheelEvent* event);
    void setCurrentCutFollowsStripMouse(bool follows);
    void recenterBottomSlicesOnCurrentPosition();
    double snappedControlPointPosition(double position) const;
    void rebuildGeneratedOverlays();
    void applyOverlayForViewer(const std::string& overlayKey,
                               CChunkedVolumeViewer* viewer,
                               const GeneratedOverlay& overlay);
    void clearControlPointContextPreview(const std::string& surfaceName,
                                         CChunkedVolumeViewer* viewer);
    GeneratedOverlay stripOverlay() const;
    GeneratedOverlay zSliceOverlay(double linePosition,
                                   bool emphasized,
                                   CChunkedVolumeViewer* viewer,
                                   PlaneSurface* plane) const;
    cv::Vec3f interpolatedLinePoint(double linePosition) const;
    cv::Vec3f interpolatedLineTangent(double linePosition) const;
    cv::Vec3f interpolatedLineUp(double linePosition, const cv::Vec3f& tangent) const;
    bool updatePlaneSurface(PlaneSurface* plane, double linePosition) const;
    double bottomSliceLinePosition(int slot, int bottomCount) const;
    void updateBottomSliceStepLabel();
    QPointF stripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                     QuadSurface* surface,
                                     double linePosition) const;
    bool handleKeyPress(QKeyEvent* event);
    void renderBottomSlicePlanes(const char* reason);

    ViewerManager* _viewerManager = nullptr;
    QVBoxLayout* _layout = nullptr;
    QComboBox* _initialDirectionCombo = nullptr;
    QLabel* _sliceStepLabel = nullptr;
    QLabel* _bottomSliceStepLabel = nullptr;
    QPushButton* _showAsMeshButton = nullptr;
    QPushButton* _fullOptimizationButton = nullptr;
    QMdiArea* _mdiArea = nullptr;
    std::vector<Pane> _panes;
    bool _suppressPaneClosed = false;

    QWidget* _generatedTopWidget = nullptr;
    std::vector<QPointer<QWidget>> _generatedContainers;
    std::vector<QMetaObject::Connection> _generatedOverlayRefreshConnections;
    QPointer<CChunkedVolumeViewer> _currentCutViewer;
    std::vector<QPointer<CChunkedVolumeViewer>> _stripViewers;
    std::vector<QPointer<CChunkedVolumeViewer>> _bottomSliceViewers;
    GeneratedViews _generatedViews;
    bool _hasGeneratedViews = false;
    double _currentLinePosition = 0.0;
    double _bottomCenterPosition = 0.0;
    double _bottomSliceLineStep = 10.0;
    int _bottomSliceStepWheelAccum = 0;
    bool _currentCutFollowsStripMouse = true;
};
