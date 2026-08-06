#pragma once

#include <QMainWindow>
#include <QList>
#include <QMetaObject>
#include <QPointer>

#include <cstdint>
#include <memory>
#include <functional>
#include <map>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <utility>

#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationFiberSegments.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include <opencv2/core/mat.hpp>

class CState;
class QAction;
class QComboBox;
class QGraphicsPathItem;
class QGraphicsRectItem;
class QGraphicsSimpleTextItem;
class QLabel;
class QMdiArea;
class QMdiSubWindow;
class QPoint;
class QProgressBar;
class QPushButton;
class QCloseEvent;
class QHBoxLayout;
class QMenu;
class QResizeEvent;
class QTimer;
class QToolButton;
class QVariantAnimation;
class QVBoxLayout;
class QSplitter;
class QSpinBox;
class ViewerManager;
class PlaneSurface;
class QuadSurface;

class LineAnnotationDialog : public QMainWindow
{
    Q_OBJECT

public:
    enum class ReoptimizationMode {
        AutoReoptimize,
        NoOptimization,
    };
    using GeneratedControlPointContextResult =
        vc3d::line_annotation::GeneratedControlPointContextResult;

    struct Pane {
        std::string surfaceName;
        QPointer<CChunkedVolumeViewer> viewer;
        QPointer<QMdiSubWindow> subWindow;
    };

    struct FastStripOverlayItems {
        struct SpanLabelItems {
            QGraphicsRectItem* background = nullptr;
            QGraphicsSimpleTextItem* text = nullptr;
        };

        QPointer<CChunkedVolumeViewer> viewer;
        std::string surfaceName;
        QGraphicsPathItem* currentLine = nullptr;
        std::vector<SpanLabelItems> spanLabels;
    };

    struct FastCurrentCutOverlayItems {
        QPointer<CChunkedVolumeViewer> viewer;
        QGraphicsPathItem* centerPoint = nullptr;
        QGraphicsPathItem* controlPoints = nullptr;
        QGraphicsPathItem* seedPoints = nullptr;
        QGraphicsPathItem* linkCandidatePoints = nullptr;
        QGraphicsPathItem* splitCandidatePoints = nullptr;
        QGraphicsPathItem* branchControlPoints = nullptr;
        QGraphicsPathItem* pendingBranchControlPoints = nullptr;
        QGraphicsPathItem* sameHvBranchControlPoints = nullptr;
        QGraphicsPathItem* sameHvPendingBranchControlPoints = nullptr;
        QGraphicsPathItem* fiberIntersections = nullptr;
        QGraphicsPathItem* linkCandidateFiberIntersections = nullptr;
        QGraphicsPathItem* branchLinkFiberIntersections = nullptr;
        QGraphicsPathItem* pendingBranchLinkFiberIntersections = nullptr;
        QGraphicsPathItem* fiberIntersectionConnectors = nullptr;
    };

    using GeneratedOverlay = vc3d::line_annotation::GeneratedOverlay;
    using GeneratedSpanAlignmentMetric = vc3d::line_annotation::GeneratedSpanAlignmentMetric;
    using GeneratedViews = vc3d::line_annotation::GeneratedViews;
    using VolumeSelectorFactory = std::function<QWidget*(QWidget*)>;

    explicit LineAnnotationDialog(ViewerManager* viewerManager,
                                  VolumeSelectorFactory volumeSelectorFactory = {},
                                  QWidget* parent = nullptr);

    void showWithSavedGeometry();
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
        const QPoint& globalPos,
        const vc3d::line_annotation::GeneratedLinkCandidateMenuState& linkCandidateState = {},
        const vc3d::line_annotation::GeneratedLinkCandidateMenuState& splitCandidateState = {},
        const vc3d::line_annotation::GeneratedLinkCandidateMenuState& splitAndLinkCandidateState = {},
        const vc3d::line_annotation::GeneratedLinkCandidateMenuState& mergeCandidateState = {});
    const std::vector<Pane>& panes() const { return _panes; }
    ReoptimizationMode reoptimizationMode() const;
    int initialCenterlineLengthVx() const;
    int extrapolationDistanceVx() const;
    int maxControlPointDistanceVx() const;
    vc3d::line_annotation::FiberOptimizationMode fiberOptimizationMode() const;
    void setFiberOptimizationMode(vc3d::line_annotation::FiberOptimizationMode mode);
    void setLasagnaDatasetOptions(
        std::vector<std::pair<std::string, std::string>> options,
        const std::string& selectedLocation);
    void setFiberInferenceDatasetOptions(
        std::vector<std::pair<std::string, std::string>> options,
        const std::string& selectedLocation);
    void setGeneratedControlPoints(std::vector<GeneratedOverlay::ControlPointMarker> controlPoints);
    void setGeneratedBranchLinePoints(std::vector<std::vector<cv::Vec3f>> branchLinePoints);
    void setGeneratedBranchLinks(std::vector<GeneratedOverlay::BranchLinkMarker> branchLinks);
    void setGeneratedBranchOverlayData(
        std::vector<GeneratedOverlay::ControlPointMarker> controlPoints,
        std::vector<std::vector<cv::Vec3f>> branchLinePoints,
        std::vector<GeneratedOverlay::BranchLinkMarker> branchLinks,
        bool requestSideStripIntersections = true,
        std::vector<GeneratedSpanAlignmentMetric> spanAlignmentMetrics = {});
    void setGeneratedFiberIntersectionMarkers(
        std::vector<GeneratedOverlay::FiberIntersectionMarker> markers);
    void setGeneratedSideStripIntersectionBusy(bool busy);
    void setGeneratedSideStripIntersectionProgress(const QString& stage,
                                                   size_t completed,
                                                   size_t total);
    void setGeneratedSideStripIntersectionResult(size_t markerCount);
    void setGeneratedSideStripIntersectionError();
    void setGeneratedPredSnapPoints(std::vector<GeneratedOverlay::PredSnapMarker> predSnapPoints);
    void setGeneratedSpanAlignmentMetrics(
        std::vector<GeneratedSpanAlignmentMetric> spanAlignmentMetrics);
    void setOptimizationBusy(bool busy);
    void setOptimizationStatus(bool optimized);
    void setFiberDisplayName(const QString& name);
    // "H"/"V" (or empty) shown next to the fiber name in the top-right label.
    void setFiberHvTag(const QString& tag);
    // Rebuilds the clickable tag buttons in the top bar. enabled=false while the
    // fiber hasn't been saved yet (tag edits need a stored fiber).
    void setFiberTags(const std::vector<std::string>& knownTags,
                      const std::vector<std::string>& activeTags,
                      bool enabled);
    void setCloseAfterFinalizationAllowed(bool allowed);
    void setWorkspaceEmbedded(bool embedded);
    bool workspaceEmbedded() const { return _workspaceEmbedded; }
    // Programmatic twin of the "current cut follows strip mouse" toggle.
    void setCutFollowEnabled(bool enabled);
    bool cutFollowEnabled() const { return _currentCutFollowsStripMouse; }

signals:
    void paneClosed(const std::string& surfaceName);
    void lineSeedRequested(const std::string& surfaceName, cv::Vec3f volumePoint, QPointF scenePoint);
    void generatedControlPointRequested(const std::string& surfaceName,
                                        cv::Vec3f volumePoint,
                                        double linePosition);
    void generatedControlPointDeleteRequested(const std::string& surfaceName,
                                              double linePosition,
                                              cv::Vec3f volumePoint);
    void generatedControlPointBranchRequested(const std::string& surfaceName,
                                              size_t controlPointIndex,
                                              cv::Vec3f linkedControlPoint,
                                              bool openAfterCreate,
                                              cv::Vec3f linkDirection);
    void generatedControlPointBranchOpenRequested(uint64_t branchFiberId,
                                                   int branchControlPointIndex);
    void generatedControlPointLinkCandidateRequested(const std::string& surfaceName,
                                                     size_t controlPointIndex,
                                                     cv::Vec3f volumePoint);
    void generatedControlPointLinkWithCandidateRequested(const std::string& surfaceName,
                                                         size_t controlPointIndex,
                                                         cv::Vec3f volumePoint);
    void generatedControlPointMergeWithCandidateRequested(const std::string& surfaceName,
                                                          size_t controlPointIndex,
                                                          cv::Vec3f volumePoint);
    void generatedControlPointSplitCandidateRequested(const std::string& surfaceName,
                                                      size_t controlPointIndex,
                                                      cv::Vec3f volumePoint);
    void generatedControlPointSplitFromCandidateRequested(const std::string& surfaceName,
                                                          size_t controlPointIndex,
                                                          cv::Vec3f volumePoint);
    void generatedControlPointSplitAndLinkFromCandidateRequested(const std::string& surfaceName,
                                                                 size_t controlPointIndex,
                                                                 cv::Vec3f volumePoint);
    void generatedNearbyAnnotationOpenRequested(uint64_t fiberId, cv::Vec3f volumePoint);
    void generatedControlPointUnlinkRequested(const std::string& surfaceName,
                                              size_t controlPointIndex,
                                              uint64_t branchFiberId,
                                              int branchControlPointIndex);
    void generatedControlPointLinkPendingChangeRequested(const std::string& surfaceName,
                                                         size_t controlPointIndex,
                                                         uint64_t branchFiberId,
                                                         int branchControlPointIndex,
                                                         bool pending);
    void generatedSegmentInterpolationGoalRequested(const std::string& surfaceName,
                                                    size_t firstControlPointIndex,
                                                    size_t secondControlPointIndex,
                                                    const std::string& goal);
    void generatedPredSnapPointRequested(const std::string& surfaceName,
                                         cv::Vec3f volumePoint);
    void generatedSideStripIntersectionQueryRequested(const std::string& surfaceName);
    void showAsMeshRequested();
    void fullOptimizationRequested();
    void fiberTagChangeRequested(const QString& tag, bool enabled);
    void closeFinalizationRequested(QCloseEvent* event);
    void reoptimizationModeChanged(LineAnnotationDialog::ReoptimizationMode mode);
    void fiberOptimizationModeChanged(
        vc3d::line_annotation::FiberOptimizationMode mode);
    void lasagnaDatasetSelectionChanged(const std::string& location);
    void fiberInferenceDatasetSelectionChanged(const std::string& location);
    void extrapolationDistanceChanged(int distanceVx);

protected:
    void closeEvent(QCloseEvent* event) override;
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
    // Coalesced entry point for the mouse-follow path: stores the latest line position and
    // applies it at most once per ~render tick via _lineUpdateTimer, so a burst of mouse-move
    // events collapses into a single (potentially O(N)) plane/overlay rebuild instead of one
    // per event. Discrete callers (keyboard jumps, clicks, scroll) keep calling
    // setCurrentLinePosition directly for immediate response.
    void requestCurrentLinePosition(double position);
    void setCurrentLinePosition(double position, bool updateCurrentCutOverlay = true);
    void cancelControlPointPreviewAnimation();
    void jumpToPreviousControlPoint();
    void jumpToNextControlPoint();
    void previewClosestControlPoint();
    bool shiftCurrentLinePositionByScrollSteps(int steps);
    bool shiftSideCutPlaneNormalOffsetByScrollSteps(int steps);
    bool shiftCutPlaneNormalOffsetByScrollSteps(PlaneSurface* plane,
                                                CChunkedVolumeViewer* viewer,
                                                int steps,
                                                double& offsetVx,
                                                const char* renderReason);
    bool applyCutPlaneNormalOffset(PlaneSurface* plane, double offsetVx) const;
    void resetGeneratedCutNormalOffsets(bool forceRender);
    void setCurrentCutFollowsStripMouse(bool follows);
    void requestGeneratedSideStripIntersections();
    cv::Vec3f branchLinkDirectionForViewer(CChunkedVolumeViewer* viewer,
                                           double linePosition) const;
    bool controlPointPlacementAllowedAt(double linePosition) const;
    vc3d::line_annotation::GeneratedCurrentLineMarkerState currentLineMarkerState() const;
    double snappedControlPointPosition(double position) const;
    void rebuildGeneratedStaticStripOverlays();
    void rebuildGeneratedDynamicOverlays(bool updateCurrentCutOverlay = true,
                                         bool updateSpanLabels = true);
    void updateGeneratedDynamicOverlaysFast(bool updateCurrentCutOverlay,
                                            bool updateSpanLabels);
    void clearFastGeneratedOverlayItemRefs();
    void rebuildGeneratedOverlays(bool requestSideStripIntersections = true);
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
    GeneratedOverlay zSliceOverlay(const GeneratedViews& views,
                                   const vc3d::line_annotation::GeneratedControlPointLinePositionIndex& controlIndex,
                                   double linePosition,
                                   bool emphasized,
                                   CChunkedVolumeViewer* viewer,
                                   PlaneSurface* plane) const;
    cv::Vec3f interpolatedLinePoint(double linePosition) const;
    cv::Vec3f interpolatedLineTangent(double linePosition) const;
    cv::Vec3f interpolatedLineUp(double linePosition, const cv::Vec3f& tangent) const;
    bool updatePlaneSurface(PlaneSurface* plane, double linePosition) const;
    bool updateSidePlaneSurface(PlaneSurface* plane, double linePosition);
    // Least-squares fit of the side-view plane orientation for the window centered on the given
    // (integer) line index. Pure/cacheable: depends only on the static line geometry.
    bool computeSideFit(int center, cv::Vec3f& normal, cv::Vec3f& upHint) const;
    QPointF stripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                     QuadSurface* surface,
                                     double linePosition) const;
    bool handleKeyPress(QKeyEvent* event);
    // Pushes line length, control dots, and the current-position marker to the
    // schematic overview bar.
    void updateOverviewBar();
    // Ctrl+right-click on an overview-bar control point: synthesize the matching
    // bottom-strip scene point and route through its context-menu signal so the
    // controller-supplied menu behaves exactly like an in-viewer click.
    void forwardOverviewControlContextMenu(double linePosition, QPoint globalPos);
    // "R": one-shot jump of the other panes to the cursor's line position on the
    // overview bar (works regardless of follow mode; leaves it unchanged).
    void snapPanesToOverviewCursor();
    // Pause badge on the bottom strip while mouse-follow is toggled off (Space).
    void updatePauseIndicator();
    // "optimized"/"not optimized" badge in the bottom strip's top-right corner.
    void updateOptimizationStatusIndicator();
    void updateOptimizationOverlayGeometry();
    void updateFiberNameLabel();
    void rebuildDatasetMenus();
    void restoreWindowGeometry();
    void saveWindowGeometry() const;
    void restoreGeneratedViewStateSettings();
    void saveGeneratedViewStateSettings();

    ViewerManager* _viewerManager = nullptr;
    QVBoxLayout* _layout = nullptr;
    QComboBox* _fiberOptimizationCombo = nullptr;
    QToolButton* _datasetMenuButton = nullptr;
    QMenu* _lasagnaDatasetMenu = nullptr;
    QMenu* _fiberInferenceDatasetMenu = nullptr;
    std::vector<std::pair<std::string, std::string>> _lasagnaDatasetOptions;
    std::vector<std::pair<std::string, std::string>> _fiberInferenceDatasetOptions;
    std::string _selectedLasagnaDatasetLocation;
    std::string _selectedFiberInferenceDatasetLocation;
    // Checked = auto-reoptimize after each edit; unchecked = no optimization.
    QAction* _autoReoptimizeAction = nullptr;
    QAction* _showAsMeshAction = nullptr;
    QAction* _fullOptimizationAction = nullptr;
    QSpinBox* _initialCenterlineLengthSpin = nullptr;
    QSpinBox* _extrapolationDistanceSpin = nullptr;
    QSpinBox* _maxControlPointDistanceSpin = nullptr;
    QLabel* _fiberNameLabel = nullptr;
    QPointer<QLabel> _optimizationStatusLabel;
    bool _optimizationStatusOptimized = false;
    QWidget* _tagRowWidget = nullptr;
    QHBoxLayout* _tagRowLayout = nullptr;
    QProgressBar* _sideStripIntersectionProgress = nullptr;
    QPushButton* _resetViewsButton = nullptr;
    QPointer<QWidget> _optimizationOverlay;
    QMdiArea* _mdiArea = nullptr;
    std::vector<Pane> _panes;
    bool _suppressPaneClosed = false;
    bool _closeAfterFinalizationAllowed = false;
    bool _closing = false;
    bool _workspaceEmbedded = false;
    QString _fiberDisplayName;
    QString _fiberHvTag;

    QWidget* _generatedTopWidget = nullptr;
    std::vector<QPointer<QWidget>> _generatedContainers;
    QPointer<QSplitter> _generatedOuterSplitter;
    QPointer<QSplitter> _generatedTopSplitter;
    QPointer<QSplitter> _generatedStripSplitter;
    // Persisted splitter sizes so resizing survives the teardown/rebuild that happens on
    // every point placement (mirrors the camera-state preservation in setGeneratedLineViews).
    QList<int> _savedOuterSplitterSizes;
    QList<int> _savedTopSplitterSizes;
    QList<int> _savedStripSplitterSizes;
    bool _haveSavedCurrentCutZoom = false;
    float _savedCurrentCutZoom = 1.0f;
    bool _haveSavedSideCutZoom = false;
    float _savedSideCutZoom = 1.0f;
    std::vector<float> _savedStripZooms;
    std::vector<QMetaObject::Connection> _generatedOverlayRefreshConnections;
    std::vector<FastStripOverlayItems> _fastStripOverlayItems;
    FastCurrentCutOverlayItems _fastCurrentCutOverlayItems;
    QPointer<CChunkedVolumeViewer> _currentCutViewer;
    QPointer<CChunkedVolumeViewer> _sideCutViewer;
    // In-place updates: keep drawing each pane's overlays from the pre-update
    // views until THAT pane adopts its first rendered frame of the re-optimized
    // surfaces (renderFrameCompleted), so a newly placed control point appears
    // together with the revised image instead of a beat earlier on the stale one.
    GeneratedViews _heldGeneratedViews;
    vc3d::line_annotation::GeneratedControlPointLinePositionIndex _heldControlIndex;
    // Line position the held overlays were drawn at; panes with a pending swap
    // keep their position markers here until their new frame lands.
    double _heldLinePosition = 0.0;
    bool _currentCutOverlaySwapPending = false;
    bool _sideCutOverlaySwapPending = false;
    std::vector<bool> _stripOverlaySwapPending;
    // Volume point of the most recent control-point placement click; the next
    // in-place update moves the current line position onto the control point
    // nearest to it, so the marker lands on the new point with the new image.
    std::optional<cv::Vec3f> _pendingPlacementFocus;
    std::vector<QPointer<CChunkedVolumeViewer>> _stripViewers;
    // Schematic fixed-height bar above the cut views: a straight line with the
    // control points (LineAnnotationOverviewBar, file-local in the .cpp).
    QPointer<QWidget> _overviewBar;
    QPointer<QLabel> _pauseIndicator;
    GeneratedViews _generatedViews;
    // Double-precision copy of _generatedViews.linePoints, built once when views are
    // generated so the per-cursor-move side plane fit doesn't reconvert the whole polyline.
    std::vector<cv::Vec3d> _linePointsd;
    // Cached side-view best-fit plane orientations for the two integer window centers that
    // straddle the current fractional position. The fit depends only on the (static) line
    // geometry, so we recompute a center only when the straddling bracket shifts; between the
    // two cached fits we interpolate by the fractional position so the side view re-orients
    // continuously instead of snapping at discrete window centers.
    struct SideFit {
        int center = std::numeric_limits<int>::min();
        cv::Vec3f normal{0.0f, 0.0f, 0.0f};
        cv::Vec3f upHint{0.0f, 0.0f, 0.0f};
        bool valid = false;
    };
    SideFit _sideFitBracket[2];
    bool _hasGeneratedViews = false;
    // Coalescing of the mouse-follow line-position updates onto a ~render-tick cadence.
    // requestCurrentLinePosition() stashes the latest position here and (re)arms the timer;
    // its timeout applies the most recent value once, so N moves between ticks collapse to one.
    QTimer* _lineUpdateTimer = nullptr;
    double _pendingLinePosition = 0.0;
    bool _lineUpdatePending = false;
    double _currentLinePosition = 0.0;
    double _initialCurrentLinePosition = 0.0;
    bool _currentCutFollowsStripMouse = true;
    cv::Matx33f _currentCutManualRotation = cv::Matx33f::eye();
    bool _currentCutManualRotationActive = false;
    double _currentCutNormalOffsetVx = 0.0;
    double _sideCutNormalOffsetVx = 0.0;
    bool _generatedOverlayRefreshQueued = false;
    vc3d::line_annotation::GeneratedControlPointLinePositionIndex _generatedControlIndex;
    QPointer<QVariantAnimation> _controlPointPreviewAnimation;
    bool _restoredWindowGeometry = false;
    bool _haveInitialCurrentCutCamera = false;
    CChunkedVolumeViewer::CameraState _initialCurrentCutCamera;
    bool _haveInitialSideCutCamera = false;
    CChunkedVolumeViewer::CameraState _initialSideCutCamera;
    std::vector<CChunkedVolumeViewer::CameraState> _initialStripCameras;
};
