#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>

#include <opencv2/core/mat.hpp>
#include <QComboBox>
#include <QCheckBox>
#include <QColor>
#include <QPointer>
#include <QString>
#include <QStringList>
#include <memory>
#include <optional>
#include <vector>
#include "ui_VCMain.h"

#include "vc/ui/VCCollection.hpp"
#include "vc/atlas/FiberIntersections.hpp"

#include <filesystem>
#include <QShortcut>
#include <set>
#include <unordered_map>
#include <map>

#include <functional>

#include "CPointCollectionWidget.hpp"
#include "CFiberWidget.hpp"
#include "CState.hpp"
#include "OpenDataManifest.hpp"
#include "LineAnnotationFiberClassification.hpp"
#include "segmentation/tools/SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/RawPointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "overlays/PlaneSlicingOverlayController.hpp"
#include "overlays/SurfaceRotationOverlayController.hpp"
#include "overlays/AtlasOverlayController.hpp"
#include "overlays/AtlasControlPointsOverlayController.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "SurfaceAffineTransformController.hpp"

class CChunkedVolumeViewer;
#include "ViewerManager.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "SeedingWidget.hpp"

class Volume;
class VolumePkg;
class Surface;
class QuadSurface;
class RenderBenchRecorder;
class RenderBenchReplay;
class QTreeWidget;
class QStandardItemModel;

// Render-bench profiling modes (see RenderBenchRecorder/RenderBenchReplay).
struct RenderBenchOptions {
    QString recordPath;   // non-empty: record camera-state timeline to this file
    QString replayPath;   // non-empty: replay a recorded timeline then quit
    bool replayWarm = false;
    bool replayOffscreen4k = false;
    bool replaySkipChunkComplete = false;
    bool replaySkipFastRender = false;
    bool replayTimedProfile = false;
    int replayTimedProfilePeriodMs = 200;
    int replayLimit = 0;
};

struct AtlasSearchFiberSnapshot {
    std::filesystem::path fiberPath;
    vc::atlas::FiberPolyline fiber;
    uint64_t storedFiberId = 0;
    vc3d::line_annotation::FiberHvClassification hvClassification;
    std::string manualHvTag;
    std::vector<std::string> tags;
};

enum class VolumeAttachResult {
    Attached,
    AlreadyAttached,
    VolumeIdConflict,
};

// Parameters for an atlas fiber-intersection search. The GUI slot builds the
// same value from its widgets.
struct AtlasFiberSearchParams {
    int searchMode{0};                 // ATLAS_SEARCH_MODE_* (CWindow.cpp)
    QStringList requiredTags;
    QStringList excludedTags;
    std::optional<double> maxDistance; // -> FiberIntersectionBroadPhaseOptions::maxDistance
};

#define MAX_RECENT_VOLPKG 10

// Project JSON schema version required by this app.
static constexpr int VOLPKG_MIN_VERSION = 1;


//forward declaration to avoid circular inclusion as CommandLineToolRunner needs CWindow.hpp
class CommandLineToolRunner;
class FiberAnnotationController;
class SegmentationModule;
class SurfacePanelController;
class MenuActionController;
class SegmentationGrower;
class ViewerControlsPanel;
class QLabel;
class QMainWindow;
class QMenu;
class QSpinBox;
class QStandardItemModel;
class QTabWidget;
class QFrame;
class FileWatcherService;
class AxisAlignedSliceController;
class SegmentationCommandHandler;
class ViewerTransformsPanel;
class LineAnnotationController;
class FiberOverlayController;
class WrapAnnotationWidget;
class AtlasControlPointsDock;
class StatusDockPanelHost;
class ViewerCompositePanel;
class LineAnnotationDialog;
class SpiralWorkspace;

class CWindow : public QMainWindow
{

    Q_OBJECT

    friend class MenuActionController;
    friend class RenderBenchReplay;
    friend class AgentBridgeServer;

public:
    // Starts an atlas fiber-intersection search without opening a dialog.
    // Progress and completion use the atlasSearch signals below.
    // Distinct name (not an overload): the slot is used as a member-function
    // pointer in new-style connect(), which an overload would make ambiguous.
    bool startAtlasFiberIntersectionSearchHeadless(const AtlasFiberSearchParams& params,
                                                   QString* errorMessage = nullptr);

    // Renders <segment>/mask.tif on a QtConcurrent worker. Presentation belongs
    // to the caller; onFinished runs on the GUI thread.
    bool startMaskRender(const QString& segmentId, bool append,
                         std::function<void(bool, QString)> onFinished,
                         QString* errorMessage = nullptr);

signals:
    // Atlas search progress. phase is in [1, ATLAS_SEARCH_PHASE_COUNT] and
    // fraction is in [0, 1] within that phase.
    void atlasSearchProgressChanged(int phase, double fraction);
    // Terminal atlas-search notification. success is false for cancellation
    // and error; resultCount equals _atlasSearchResults.size().
    void atlasSearchFinished(bool success, int resultCount);

public slots:
    void onShowStatusMessage(QString text, int timeout);
    void onVisLasagnaObj(const std::string& segmentId);
    void onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                   SegmentationGrowthDirection direction,
                                   int steps,
                                   bool inpaintOnly);
    void onFocusPOIChanged(std::string name, POI* poi);
    void onPointDoubleClicked(uint64_t pointId);
    void onCopyWithNtRequested();
    void onFocusViewsRequested(uint64_t collectionId, uint64_t pointId);

public:
    enum class VolumeOpenError {
        None,
        PackageLoadFailed,
        VolumeNotFound,
    };

    explicit CWindow(size_t cacheSizeGB = CHUNK_CACHE_SIZE_GB,
                     RenderBenchOptions benchOptions = {});
    ~CWindow(void);

    bool openVolumePackage(const QString& path,
                           bool interactive = true,
                           QString* errorMessage = nullptr,
                           const QString& preferredVolumeId = {},
                           VolumeOpenError* openError = nullptr);

    // Helper method to get the current volume path
    QString getCurrentVolumePath() const;
    VCCollection* pointCollection() { return _state->pointCollection(); }

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void CreateWidgets(void);
    QMainWindow* segmentWorkspaceWindow() const { return _segmentWorkspaceWindow; }
    ViewerManager* activeWorkspaceViewerManager() const;
    void updateActiveWorkspaceViewerControls();
    void populateDockToggleMenu(QMenu* menu) const;
    void createAtlasWorkspace();
    void displayAtlasFromDirectory(const std::filesystem::path& atlasDir);
    // Dialog-free core of displayAtlasFromDirectory: loads/displays the atlas,
    // letting std::exception propagate (no QMessageBox). Shared by the interactive
    // method and the headless variant below.
    void loadAndDisplayAtlas(const std::filesystem::path& atlasDir);
    // Opens an atlas without dialogs or the rebuild prompt. The distinct name
    // avoids ambiguity where displayAtlasFromDirectory is used in connect().
    bool displayAtlasFromDirectoryHeadless(const std::filesystem::path& atlasDir,
                                           QString* errorMessage = nullptr);
    void refreshAtlasOverviewDocks();
    void updateAtlasFiberDocks();
    void updateAtlasSearchDocks();
    void remapCurrentAtlas();
    // Starts atlas remapping without dialogs. The interactive caller can add
    // its completion UI through onFinished.
    bool startAtlasRemapHeadless(QString* errorMessage = nullptr,
                                 std::function<void(bool success, const QString& detail)> onFinished = {});
    void optimizeAtlasSnapCandidates();
    // Starts snap-candidate optimization without dialogs. Async failures go to
    // the status bar and stderr, plus onAsyncError when provided.
    bool optimizeAtlasSnapCandidatesHeadless(QString* errorMessage = nullptr,
                                             std::function<void(const QString& detail)> onAsyncError = {});
    void startAtlasFiberIntersectionSearch();
    void cancelAtlasFiberIntersectionSearch();
    void updateAtlasSearchProgress(vc::atlas::AtlasSearchProgressPhase phase,
                                   std::size_t completed,
                                   std::size_t total);
    void populateAtlasSearchResults(const std::vector<vc::atlas::FiberIntersectionResult>& results,
                                    std::vector<double> signedWindings = {});
    void openAtlasSearchResult(int sortedResultIndex);
    void clearAtlasSearchPreviewState();
    void updateAtlasSearchPreviewCandidates();
    void setAtlasSearchHoverResult(std::optional<int> sortedResultIndex);
    void updateAtlasSearchSelectionFromTree(QTreeWidget* sourceTree);
    void syncAtlasSearchTreeSelection(QTreeWidget* sourceTree);
    void updateAtlasSearchPreviewRequests();
    void requestAtlasSearchPreviewLine(int sortedResultIndex);
    void switchToLasagnaWorkspace();
    void switchToMainWorkspace();
    void switchToFiberSliceWorkspace();
    void openLineAnnotationWorkspace(LineAnnotationDialog* dialog, const QString& title);
    void repeatLastLasagnaAction();
    void selectLasagnaOutputSegment(const QString& outputName);

    void UpdateView(void);
    void UpdateVolpkgLabel(int filterCounter);
    void updateVolumePackageEmptyState();
    void showStatusBarMessage(const QString& text, int timeout = 0);
    void clearStatusBarMessage();


    // Helper method for command line tools
    bool initializeCommandLineRunner(void);

    VolumeViewerBase *newConnectedViewer(std::string surfaceName, QString title, QMdiArea *mdiArea);
    VolumeViewerBase* newConnectedViewerInWidget(std::string surfaceName, QString title, QWidget* parent);
    void closeEvent(QCloseEvent* event) override;

    void setWidgetsEnabled(bool state);

    bool OpenVolume(const QString& path,
                    bool interactive = true,
                    QString* errorMessage = nullptr,
                    const QString& preferredVolumeId = {},
                    VolumeOpenError* openError = nullptr);
    void CloseVolume(void);


    void setVolume(std::shared_ptr<Volume> newvol);
    VolumeAttachResult attachVolumeToCurrentPackage(
        const std::shared_ptr<Volume>& volume,
        const QString& location,
        std::vector<std::string> tags = {},
        const QString& remoteCacheRoot = {},
        const QString& preferredVolumeId = {});
    void refreshCurrentVolumePackageUi(const QString& preferredVolumeId = QString(),
                                       bool reloadSurfaces = true);
    void syncVolumeSelectionControls(const QString& activeVolumeId = QString());
    QWidget* createAnnotationVolumeSelector(QWidget* parent);
    void updateNormalGridAvailability();
    void toggleVolumeOverlayVisibility();
    bool centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId);
    bool centerFocusOnCursor();
    void recenterPlaneViewersOn(const cv::Vec3f& position);
    void recenterSegmentationViewerNear(const cv::Vec3f& position);
    bool recenterViewersOnCurrentFocus();
    void setSegmentationCursorMirroring(bool enabled);
    bool segmentationCursorMirroringEnabled() const { return _mirrorCursorToSegmentation; }
    void updateSurfaceOverlayDropdown();
    void onSurfaceOverlaySelectionChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void showSurfaceOverlaySelectionDialog();
    void applySurfaceOverlaySelection();
    void updateSurfaceOverlayButtonText();
    QColor getOverlayColor(size_t index) const;
    cv::Vec3b getOverlayColorBGR(size_t index) const;

private slots:
    void onSegmentationDirChanged(int index);
    void onEditMaskPressed(const QString& segmentId);
    void onAppendMaskPressed(const QString& segmentId);
    void onManualLocationChanged();
    void onZoomIn();
    void onZoomOut();
    void onCopyCoordinates();
    void onResetAxisAlignedRotations();
    void onAxisAlignedSlicesToggled(bool enabled);
    void onAxisOverlayVisibilityToggled(bool enabled);
    void onAxisOverlayOpacityChanged(int value);
    void onMoveOnSurfaceChangedToggled(bool enabled);
    void onPlaneIntersectionLinesToggled(bool enabled);
    void onSegmentationEditingModeChanged(bool enabled);
    void onSegmentationStopToolsRequested();
    void configureChunkedViewerConnections(CChunkedVolumeViewer* viewer);

    CChunkedVolumeViewer* segmentationViewer() const;
    VolumeViewerBase* segmentationBaseViewer() const;
    VolumeViewerBase* activeBaseViewer() const;
    std::vector<QComboBox*> volumeSelectionControls() const;
    void connectVolumeSelector(QComboBox* selector);
    void clearSurfaceSelection();
    QString lastVolumeSettingKeyForCurrentPackage() const;
    QString rememberedVolumeIdForCurrentPackage() const;
    void rememberCurrentVolumeForPackage(const QString& volumeId) const;
    const vc3d::opendata::OpenDataManifest* cachedOpenDataManifest() const;
    std::string openDataVolumeIdForLoadedVolumeId(const std::string& volumeId) const;
    std::optional<cv::Matx44d> openDataVolumeTransformForSwitch(
        const std::string& fromLoadedVolumeId,
        const std::string& toLoadedVolumeId) const;
    void updateOpenDataSegmentTransformState(bool showDialog = true);
    void resetSegmentationViews(bool persistLayout = true);
    void onSurfaceActivated(const QString& surfaceId, QuadSurface* surface);
    void onSurfaceActivatedPreserveEditing(const QString& surfaceId, QuadSurface* surface);
    bool restoreActiveSurfaceAfterSurfaceReload(const std::string& surfaceId);
    // Attaches the render-bench recorder once a volume+segment are active (no-op
    // unless --record was passed and the recorder isn't already attached).
    void maybeAttachBenchRecorder();
    void onSegmentationGrowthStatusChanged(bool running);
    void onZScrollSensitivityChanged(double sensitivity);
    void onSharedCacheStatsChanged(const QStringList& items);
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void onConvertPointToAnchor(uint64_t pointId, uint64_t collectionId);
    void onNewFiberRequested();
    void onFiberCrosshairModeChanged(bool active);
    void onFiberViewersRequested();
    void onFiberAnnotationFinished(uint64_t fiberId);
    void refreshVolumeSelectionUi(const QString& preferredVolumeId = QString());
    void refreshSegmentationDirectoryDropdown();
    void applySegmentFolderSelection(bool reloadSurfaces);
    void showSegmentFolderPaletteMenu(int row);
    QColor defaultSegmentFolderColor(const QString& dirName) const;
    QString effectiveDefaultSegmentFolderDir() const;

private:
    CState* _state;

    QComboBox* volSelect{nullptr};
    std::vector<QPointer<QComboBox>> _annotationVolumeSelects;
    QComboBox* cmbSegmentationDir;
    QStandardItemModel* _segmentDirModel{nullptr};
    bool _updatingSegmentDirUi{false};
    std::map<QString, QColor> _segmentFolderSolidColors;
    QString _segmentFolderDefaultPaletteDir;
    mutable bool _openDataManifestLoadAttempted{false};
    mutable std::optional<vc3d::opendata::OpenDataManifest> _openDataManifestCache;


    SeedingWidget* _seedingWidget;
    SegmentationWidget* _segmentationWidget{nullptr};
    QDockWidget* _lasagnaDock{nullptr};
    CPointCollectionWidget* _point_collection_widget;
    WrapAnnotationWidget* _wrapAnnotationWidget{nullptr};
    CFiberWidget* _fiberWidget{nullptr};
    CFiberWidget* _fiberSliceWidget{nullptr};
    std::unique_ptr<FiberAnnotationController> _fiberController;

    SurfaceTreeWidget *treeWidgetSurfaces;
    QPushButton *btnReloadSurfaces;

    //TODO abstract these into separate QWidget class?
    QLineEdit* lblLocFocus;
    QCheckBox* chkAxisAlignedSlices;
    QLabel* _segmentationGrowthWarning{nullptr};
    QLabel* _segmentTransformWarning{nullptr};
    QLabel* _statusMessageLabel{nullptr};
    QLabel* _sharedCacheStatsLabel{nullptr};
    QLabel* _persistentCacheLowSpaceLabel{nullptr};
    QLabel* _persistentCacheWarningText{nullptr};
    QFrame* _persistentCacheWarningBanner{nullptr};
    QLabel* _sliceStepLabel{nullptr};
    QTimer* _statusMessageTimer{nullptr};
    QTimer* _persistentCacheSpaceTimer{nullptr};
    bool _persistentCacheBannerShownThisSession{false};
    QString _segmentationGrowthStatusText;
    QString _lastSegmentTransformWarningVolumeId;
    bool _relayingNativeStatusMessage{false};


    Ui_VCMainWindow ui;
    bool _destroyingWindow{false};
    QTabWidget* _workspaceTabs{nullptr};
    QMainWindow* _segmentWorkspaceWindow{nullptr};
    StatusDockPanelHost* _statusDockPanelHost{nullptr};
    QMainWindow* _lasagnaWorkspaceWindow{nullptr};
    QMainWindow* _atlasWorkspaceWindow{nullptr};
    QMainWindow* _fiberSliceWorkspaceWindow{nullptr};
    QMainWindow* _intersectionsWorkspaceWindow{nullptr};
    QMainWindow* _spiralWorkspaceWindow{nullptr};
    SpiralWorkspace* _spiralWorkspace{nullptr};
    QDockWidget* _atlasOverviewDock{nullptr};
    QDockWidget* _atlasSearchDock{nullptr};
    QDockWidget* _inkDetectionDock{nullptr};
    QDockWidget* _transformsDock{nullptr};
    AtlasControlPointsDock* _atlasControlDock{nullptr};
    QDockWidget* _atlasWorkspaceOverviewDock{nullptr};
    QDockWidget* _atlasWorkspaceFiberDock{nullptr};
    QDockWidget* _atlasWorkspaceSearchDock{nullptr};
    VolumeViewerBase* _atlasViewer{nullptr};
    std::optional<std::filesystem::path> _currentAtlasDir;
    std::string _currentAtlasName;
    vc::atlas::FiberIntersectionCache _fiberIntersectionCache;
    std::vector<vc::atlas::FiberIntersectionResult> _atlasSearchResults;
    std::vector<double> _atlasSearchSignedWindings;
    std::unordered_map<uint64_t, AtlasSearchFiberSnapshot> _atlasSearchFiberSnapshotsByRuntimeId;
    std::optional<std::filesystem::path> _atlasSearchLasagnaManifestPath;
    double _atlasSearchLasagnaWorkingToBaseScale = 1.0;
    int _atlasSearchPreviewGeneration{0};
    std::optional<int> _atlasSearchHoveredResult;
    std::set<int> _atlasSearchSelectedResults;
    std::set<int> _atlasSearchPreviewRequestedResults;
    bool _atlasSearchCancelRequested{false};
    std::shared_ptr<std::atomic_bool> _atlasSearchCancelFlag;
    vc::atlas::AtlasSearchProgressPhase _atlasSearchProgressPhase{
        vc::atlas::AtlasSearchProgressPhase::PrepareInputs};
    std::size_t _atlasSearchPhaseCompleted{0};
    std::size_t _atlasSearchPhaseTotal{0};
    QMdiArea *mdiArea;
    QMdiArea* _fiberSliceMdiArea{nullptr};
    QMdiArea* _intersectionsMdiArea{nullptr};

    bool can_change_volume_();

    size_t _cacheSizeBytes = 0;

    std::unique_ptr<VolumeOverlayController> _volumeOverlay;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<ViewerControlsPanel> _viewerControlsPanel;
    ViewerCompositePanel* _viewerCompositePanel{nullptr};
    bool _mirrorCursorToSegmentation{false};
    std::unique_ptr<SegmentationGrower> _segmentationGrower;

    // Surface overlay multi-select state
    std::map<std::string, size_t> _surfaceOverlayColorAssignments;
    size_t _nextSurfaceOverlayColorIndex{0};
    QStandardItemModel* _surfaceOverlayModel{nullptr};

    std::unique_ptr<SegmentationEditManager> _segmentationEdit;
    std::unique_ptr<SegmentationOverlayController> _segmentationOverlay;
    std::unique_ptr<PointsOverlayController> _pointsOverlay;
    std::unique_ptr<RawPointsOverlayController> _rawPointsOverlay;
    std::unique_ptr<PathsOverlayController> _pathsOverlay;
    std::unique_ptr<BBoxOverlayController> _bboxOverlay;
    std::unique_ptr<VectorOverlayController> _vectorOverlay;
    std::unique_ptr<InkDetectionOverlayController> _inkDetectionOverlay;
    std::unique_ptr<PlaneSlicingOverlayController> _planeSlicingOverlay;
    std::unique_ptr<SurfaceRotationOverlayController> _surfaceRotationOverlay;
    std::unique_ptr<AtlasOverlayController> _atlasOverlay;
    std::unique_ptr<AtlasControlPointsOverlayController> _atlasControlOverlay;
    std::unique_ptr<FiberOverlayController> _fiberOverlay;
    std::unique_ptr<SegmentationModule> _segmentationModule;
    std::unique_ptr<SurfacePanelController> _surfacePanel;
    std::unique_ptr<MenuActionController> _menuController;
    std::unique_ptr<SurfaceAffineTransformController> _surfaceAffineTransforms;
    // runner for command line tools
    CommandLineToolRunner* _cmdRunner;
    // Render-bench profiling harness (record/replay navigation timelines).
    RenderBenchOptions _benchOptions;
    std::unique_ptr<RenderBenchRecorder> _benchRecorder;
    std::unique_ptr<RenderBenchReplay> _benchReplay;
    bool _normalGridAvailable{false};
    QString _normalGridPath;

    std::unique_ptr<FileWatcherService> _fileWatcher;
    std::unique_ptr<AxisAlignedSliceController> _axisAlignedSliceController;
    bool _maskRenderInProgress{false};
    std::unique_ptr<SegmentationCommandHandler> _segmentationCommandHandler;
    std::unique_ptr<LineAnnotationController> _lineAnnotationController;
    // Keyboard shortcuts
    QShortcut* fCompositeViewShortcut;
    QShortcut* fDirectionHintsShortcut;
    QShortcut* fSurfaceNormalsShortcut;
    QShortcut* fAxisAlignedSlicesShortcut;
    QShortcut* fZoomInShortcut;
    QShortcut* fZoomOutShortcut;
    QShortcut* fResetViewShortcut;
    QShortcut* fOpenLasagnaWorkspaceShortcut{nullptr};
    QShortcut* fRepeatLasagnaActionShortcut{nullptr};

    // Z offset shortcuts (Ctrl+,/. for normal direction)
    QShortcut* fWorldOffsetZPosShortcut;  // Ctrl+. (further/deeper)
    QShortcut* fWorldOffsetZNegShortcut;  // Ctrl+, (closer)

    // Segment cycling shortcuts
    QShortcut* fCycleNextSegmentShortcut;
    QShortcut* fCyclePrevSegmentShortcut;
    QShortcut* fApplyApprovedTagShortcut{nullptr};
    QShortcut* fApplyDefectiveTagShortcut{nullptr};

    QShortcut* fFocusedViewShortcut;
    bool _focusedViewActive{false};
    struct SavedDockState {
        bool visible;
        bool floating;
        bool wasRaised;
    };
    std::map<QDockWidget*, SavedDockState> _savedDockStates;
    void toggleFocusedView();

    // Timer for debounced window state saving
    QTimer* _windowStateSaveTimer{nullptr};
    void scheduleWindowStateSave();
    void saveWindowState();

};  // class CWindow
