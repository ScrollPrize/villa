#pragma once

#include <QMainWindow>
#include <QElapsedTimer>
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
class QCloseEvent;
class QHBoxLayout;
class QMenu;
class QResizeEvent;
class QTimer;
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
        QGraphicsPathItem* ghostControlPointPrev = nullptr;
        QGraphicsPathItem* ghostControlPointNext = nullptr;
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
    // Takes the views by value: the caller hands its (large - all line
    // points, normals, branch polylines, strip map) struct over instead of
    // this dialog copying it a second time.
    bool setGeneratedLineViews(GeneratedViews views,
                               const CChunkedVolumeViewer::CameraState& camera);
    // Shift freshly rebuilt strip surfaces so the fiber spot under the live
    // strip cameras keeps its surface coordinate across the update. The
    // controller calls this BEFORE re-registering the surfaces: the strip
    // viewers keep their raw camera coordinates when they adopt replacements,
    // so anchoring the new parameterization instead of moving the cameras
    // keeps every frame — old, new, and the held overlays in between —
    // pixel-stable through the swap. No-op without live strip views.
    void anchorGeneratedStripSurfacesForUpdate(
        QuadSurface* newLineSurface,
        QuadSurface* newLineSideSlice,
        const vc::lasagna::LineStripPositionMap& newPositionMap,
        const std::vector<cv::Vec3f>& newLinePoints) const;
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
    int maxControlPointExtrapolationDistanceVx() const;
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
    // blockInput=true covers the panes with the input-swallowing overlay
    // (initial seed solves: there is no line to edit yet). blockInput=false
    // shows a passive top-center badge instead and leaves the panes
    // interactive, so control points can keep being placed while a
    // re-optimization runs (edits coalesce in the controller).
    void setOptimizationBusy(bool busy, bool blockInput = true);
    void setOptimizationStatus(bool optimized);
    // Empty retracts the notice; see updateUmbilicusNotice().
    void setUmbilicusNotice(const QString& notice);
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
    // lineAnchor: the 3D point of linePosition on the DISPLAYED line, so the
    // controller can resolve the position on its own (possibly renumbered)
    // line without consulting the clicked point.
    void generatedControlPointRequested(const std::string& surfaceName,
                                        cv::Vec3f volumePoint,
                                        double linePosition,
                                        cv::Vec3f lineAnchor);
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
    void keyReleaseEvent(QKeyEvent* event) override;
    void changeEvent(QEvent* event) override;
    bool event(QEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void bindPaneInteractions(const std::string& surfaceName,
                              CChunkedVolumeViewer* viewer,
                              bool seedPlacementEnabled);
    // One shared cursor cross across the generated panes: the hovered pane
    // broadcasts its cursor volume point to the others. Dialog-local —
    // independent of the global "Sync cursor across views" toggle.
    void connectLinkedCursorMirroring(
        std::vector<QPointer<CChunkedVolumeViewer>> panes);
    // Coalesces mirror updates onto a ~render-tick cadence (same pattern as
    // requestCurrentLinePosition): a burst of mouse moves collapses into one
    // projection + crosshair update per non-hovered pane per tick.
    void requestLinkedCursorMirror(CChunkedVolumeViewer* source,
                                   const std::optional<cv::Vec3f>& point);
    // Pushes the "Mirror cursor across panes" state onto the panes. The block
    // has to sit on the receiving side: the panes belong to the same
    // ViewerManager as the main window, so the global cursor sync would keep
    // feeding them even with this dialog's own broadcast silenced.
    void applyLinkedCursorMirroringToPanes();
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
    // forceApply bypasses the sub-1e-3 no-op shortcut so a keyboard-pan landing
    // moves the cut planes onto the exact control-point position.
    void setCurrentLinePosition(double position,
                                bool updateCurrentCutOverlay = true,
                                bool forceApply = false);
    void cancelControlPointPreviewAnimation();
    // Left/Right arrow panning between control points. One velocity integrator
    // (generatedArrowPanStep) drives it: a tap brakes into the first control
    // point ahead, a hold cruises through the intermediate ones and lands on the
    // next one after the key comes up, and the opposite arrow reverses mid-pan.
    void startArrowPan(int direction);
    // Control-point line positions plus (when there is room) one boundary
    // target beyond each outer control point: the maximum base-voxel
    // arclength extrapolation distance or the line end, whichever is shorter.
    std::vector<double> arrowPanTargetPositions() const;
    void releaseArrowPanKey(int direction);
    void updateArrowPanStopTarget();
    // Re-validates the pan against an edited control-point set: re-promises
    // the minimum target if the edit removed it, then re-selects the stop
    // target. Cancels when nothing remains in the travel direction.
    void rebaseArrowPanTargets();
    // Cancels the pan and clears the physical key flags when focus leaves the
    // window/app (the key-up is delivered elsewhere; don't render unattended).
    void stopArrowPanForFocusLoss();
    void tickArrowPan();
    void finishArrowPan(double position);
    void cancelArrowPan();
    // Up/Down: scale the cruise speed (persisted) and flash the badge.
    void adjustArrowPanCruiseSpeed(double factor);
    void updateArrowPanSpeedIndicator();
    // Keeps the current-position line centered in the strips while the keyboard
    // pan scrolls them underneath it. Vertical only on the initial snap. Takes
    // the position explicitly so each tick can move the camera BEFORE the
    // overlay rebuild bakes it into the drawn line position.
    void centerStripsOnLinePosition(double linePosition, bool includeVertical);
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
    // "B": zero every accumulated normal offset — the side cut plane's and
    // both strips' surface offsets. The current cut cannot accumulate one
    // (Shift-scroll steps along the line there) but is reset with the side
    // cut for symmetry.
    void resetGeneratedNormalOffsets();
    void setCurrentCutFollowsStripMouse(bool follows);
    void requestGeneratedSideStripIntersections();
    cv::Vec3f branchLinkDirectionForViewer(CChunkedVolumeViewer* viewer,
                                           double linePosition) const;
    bool controlPointPlacementAllowedAt(double linePosition) const;
    vc3d::line_annotation::GeneratedCurrentLineMarkerState currentLineMarkerState() const;
    double snappedControlPointPosition(double position) const;
    // Cumulative base-voxel arclengths of the displayed line (one per point),
    // or an empty vector when no usable map exists; along-line motion (wheel,
    // arrow pan, Space snap) is measured in these units.
    const std::vector<double>& currentLineArclengths() const;
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
    bool placeControlPointAtCurrentLinePosition();
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
    // Interpolated sampled sheet normal (oriented away from the scroll
    // center, see GeneratedViews::lineNormals). NaN when samples are missing.
    cv::Vec3f interpolatedOrientedNormal(double linePosition) const;
    // The same normal projected perpendicular to the tangent. NaN when the
    // projection is unstable (normal nearly parallel to the tangent, i.e.
    // extreme bends).
    cv::Vec3f interpolatedLineNormal(double linePosition, const cv::Vec3f& tangent) const;
    bool updatePlaneSurface(PlaneSurface* plane, double linePosition) const;
    bool updateSidePlaneSurface(PlaneSurface* plane, double linePosition) const;
    QPointF stripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                     QuadSurface* surface,
                                     double linePosition) const;
    bool handleKeyPress(QKeyEvent* event);
    bool handleKeyRelease(QKeyEvent* event);
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
    // Mirrors the along-line position and zoom from one strip viewer to the
    // other; vertical offset stays per-strip.
    void syncLinkedStripCamera(CChunkedVolumeViewer* source);
    // Pause badge on the bottom strip while mouse-follow is toggled off (Space).
    void updatePauseIndicator();
    // "optimized"/"not optimized" badge in the bottom strip's top-right corner.
    void updateOptimizationStatusIndicator();
    // Red notice at the bottom strip's top-left when the package's umbilicus
    // could not be used, so that falling back to the volume centre is visible
    // rather than only logged.
    void updateUmbilicusNotice();
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
    // Values committed via the menu rows' Apply buttons; the spinboxes hold
    // uncommitted edits until then (and revert when the menu reopens).
    int _appliedInitialCenterlineLengthVx = 0;
    int _appliedExtrapolationDistanceVx = 0;
    QSpinBox* _maxControlPointExtrapolationDistanceSpin = nullptr;
    QLabel* _fiberNameLabel = nullptr;
    QPointer<QLabel> _optimizationStatusLabel;
    bool _optimizationStatusOptimized = false;
    // The overlay only blocks the mouse, so keyboard-driven edits have to test
    // this themselves before they queue any deferred state.
    // True only while the blocking overlay is up (busy && blockInput):
    // passive-badge solves keep editing live, and key handlers must match
    // what a mouse click can reach.
    bool _optimizationInputBlocked = false;
    QWidget* _tagRowWidget = nullptr;
    QHBoxLayout* _tagRowLayout = nullptr;
    QProgressBar* _sideStripIntersectionProgress = nullptr;
    QAction* _mirrorCursorAction = nullptr;
    QAction* _resetViewsAction = nullptr;
    QPointer<QWidget> _optimizationOverlay;
    QPointer<QWidget> _optimizationBadge;
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
    std::vector<QPointer<CChunkedVolumeViewer>> _stripViewers;
    // Schematic fixed-height bar above the cut views: a straight line with the
    // control points (LineAnnotationOverviewBar, file-local in the .cpp).
    QPointer<QWidget> _overviewBar;
    QPointer<QLabel> _pauseIndicator;
    QPointer<QLabel> _umbilicusNoticeLabel;
    QString _umbilicusNotice;
    GeneratedViews _generatedViews;
    // Sign applied to the displayed line tangent so the current cut's screen
    // left/right and the side cut's vertical do not depend on the arbitrary
    // stored point order. Recomputed once per materialization.
    float _displayTangentSign = 1.0f;
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
    // Generation-based deduplication of the coalesced overlay refresh: every
    // overlaysUpdated bumps the generation; a landing's full rebuild records
    // the generation it covered, and the queued callback skips only when no
    // newer update arrived in between.
    uint64_t _generatedOverlayRefreshGeneration = 0;
    uint64_t _generatedOverlayRefreshCoveredGeneration = 0;
    bool _syncingStripCameras = false;
    std::vector<QPointer<CChunkedVolumeViewer>> _linkedCursorPanes;
    QPointer<CChunkedVolumeViewer> _linkedCursorSource;
    std::optional<cv::Vec3f> _pendingLinkedCursorPoint;
    // Owned single-shot coalescing timer (like _lineUpdateTimer); stopped on
    // pane teardown so a pending mirror can't stamp a pre-rebuild point onto
    // freshly built panes.
    QTimer* _linkedCursorMirrorTimer = nullptr;
    vc3d::line_annotation::GeneratedControlPointLinePositionIndex _generatedControlIndex;
    QPointer<QVariantAnimation> _controlPointPreviewAnimation;
    // Arrow-key pan integrator. _arrowPanDirection is the travel direction and
    // stays set while a released tap coasts into its target; _arrowPanKeyHeld
    // only tracks the key. _arrowPanMinimumTarget is the first control point the
    // gesture promised at press time (NaN when idle), so a hold can never land
    // short of what the same tap would have reached.
    int _arrowPanDirection = 0;
    bool _arrowPanKeyHeld = false;
    // Physical key state of the two horizontal arrows, so releasing a reversal
    // key can hand the pan back to the key that is still held down.
    bool _arrowKeyLeftDown = false;
    bool _arrowKeyRightDown = false;
    // Distinguishes a pan that ended by landing from one that was cancelled
    // (space, edits): only a landed pan may hand back to a still-held key.
    bool _arrowPanEndedByLanding = false;
    double _arrowPanVelocity = 0.0;
    // Unit the running pan's velocity is in (true: base-voxel arclength,
    // false: line positions); a change mid-gesture cancels the pan.
    std::optional<bool> _arrowPanArclengthUnits;
    std::optional<double> _arrowPanStopTarget;
    double _arrowPanMinimumTarget = std::numeric_limits<double>::quiet_NaN();
    double _arrowPanCruiseSpeed =
        vc3d::line_annotation::kGeneratedArrowPanDefaultSpeed;
    QTimer* _arrowPanTimer = nullptr;
    QElapsedTimer _arrowPanClock;
    QPointer<QLabel> _arrowPanSpeedLabel;
    QTimer* _arrowPanSpeedLabelTimer = nullptr;
    bool _restoredWindowGeometry = false;
    bool _haveInitialCurrentCutCamera = false;
    CChunkedVolumeViewer::CameraState _initialCurrentCutCamera;
    bool _haveInitialSideCutCamera = false;
    CChunkedVolumeViewer::CameraState _initialSideCutCamera;
    std::vector<CChunkedVolumeViewer::CameraState> _initialStripCameras;
};
