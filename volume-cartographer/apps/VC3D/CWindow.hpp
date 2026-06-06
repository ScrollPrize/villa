#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core/mat.hpp>
#include <QComboBox>
#include <QCheckBox>
#include <QString>
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

#include "CPointCollectionWidget.hpp"
#include "CFiberWidget.hpp"
#include "CState.hpp"
#include "segmentation/tools/SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/RawPointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/PlaneSlicingOverlayController.hpp"
#include "overlays/SurfaceRotationOverlayController.hpp"
#include "overlays/AtlasOverlayController.hpp"
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

// Render-bench profiling modes (see RenderBenchRecorder/RenderBenchReplay).
struct RenderBenchOptions {
    QString recordPath;   // non-empty: record camera-state timeline to this file
    QString replayPath;   // non-empty: replay a recorded timeline then quit
    bool replayWarm = false;
};

struct AtlasSearchFiberSnapshot {
    std::filesystem::path fiberPath;
    vc::atlas::FiberPolyline fiber;
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
class QTableWidget;
class FileWatcherService;
class AxisAlignedSliceController;
class SegmentationCommandHandler;
class ViewerTransformsPanel;
class LineAnnotationController;

class CWindow : public QMainWindow
{

    Q_OBJECT

    friend class MenuActionController;
    friend class RenderBenchReplay;

public:
signals:

public slots:
    void onShowStatusMessage(QString text, int timeout);
    void onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
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
    explicit CWindow(size_t cacheSizeGB = CHUNK_CACHE_SIZE_GB,
                     RenderBenchOptions benchOptions = {});
    ~CWindow(void);

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
    void populateDockToggleMenu(QMenu* menu) const;
    void createAtlasWorkspace();
    void displayAtlasFromDirectory(const std::filesystem::path& atlasDir);
    void refreshAtlasOverviewDocks();
    void updateAtlasSearchDocks();
    void startAtlasFiberIntersectionSearch();
    void cancelAtlasFiberIntersectionSearch();
    void populateAtlasSearchResults(const std::vector<vc::atlas::FiberIntersectionResult>& results);
    void openAtlasSearchResult(int sortedResultIndex);
    void clearAtlasSearchPreviewState();
    void updateAtlasSearchPreviewCandidates();
    void setAtlasSearchHoverResult(std::optional<int> sortedResultIndex);
    void updateAtlasSearchSelectionFromTable(QTableWidget* sourceTable);
    void syncAtlasSearchTableSelection(QTableWidget* sourceTable);
    void updateAtlasSearchPreviewRequests();
    void requestAtlasSearchPreviewLine(int sortedResultIndex);
    void switchToLasagnaWorkspace();
    void switchToMainWorkspace();
    void switchToFiberSliceWorkspace();
    void repeatLastLasagnaAction();
    void selectLasagnaOutputSegment(const QString& outputName);

    void UpdateView(void);
    void UpdateVolpkgLabel(int filterCounter);


    // Helper method for command line tools
    bool initializeCommandLineRunner(void);

    VolumeViewerBase *newConnectedViewer(std::string surfaceName, QString title, QMdiArea *mdiArea);
    void closeEvent(QCloseEvent* event) override;

    void setWidgetsEnabled(bool state);

    bool InitializeVolumePkg(const std::string& nVpkgPath);

    void OpenVolume(const QString& path);
    void CloseVolume(void);


    void setVolume(std::shared_ptr<Volume> newvol);
    bool attachVolumeToCurrentPackage(const std::shared_ptr<Volume>& volume,
                                      const QString& preferredVolumeId = QString());
    void refreshCurrentVolumePackageUi(const QString& preferredVolumeId = QString(),
                                       bool reloadSurfaces = true);
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
    QColor getOverlayColor(size_t index) const;
    cv::Vec3b getOverlayColorBGR(size_t index) const;

private slots:
    void onSegmentationDirChanged(int index);
    void onEditMaskPressed();
    void onAppendMaskPressed();
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
    void clearSurfaceSelection();
    void onSurfaceActivated(const QString& surfaceId, QuadSurface* surface);
    void onSurfaceActivatedPreserveEditing(const QString& surfaceId, QuadSurface* surface);
    // Attaches the render-bench recorder once a volume+segment are active (no-op
    // unless --record was passed and the recorder isn't already attached).
    void maybeAttachBenchRecorder();
    void onSegmentationGrowthStatusChanged(bool running);
    void onSliceStepSizeChanged(int newSize);
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void onConvertPointToAnchor(uint64_t pointId, uint64_t collectionId);
    void onNewFiberRequested();
    void onFiberCrosshairModeChanged(bool active);
    void onFiberViewersRequested();
    void onFiberAnnotationFinished(uint64_t fiberId);
    void refreshVolumeSelectionUi(const QString& preferredVolumeId = QString());

private:
    CState* _state;

    QComboBox* volSelect{nullptr};
    QComboBox* cmbSegmentationDir;


    SeedingWidget* _seedingWidget;
    SegmentationWidget* _segmentationWidget{nullptr};
    QDockWidget* _lasagnaDock{nullptr};
    CPointCollectionWidget* _point_collection_widget;
    CFiberWidget* _fiberWidget{nullptr};
    CFiberWidget* _fiberSliceWidget{nullptr};
    std::unique_ptr<FiberAnnotationController> _fiberController;

    SurfaceTreeWidget *treeWidgetSurfaces;
    QPushButton *btnReloadSurfaces;

    //TODO abstract these into separate QWidget class?
    QLineEdit* lblLocFocus;
    QCheckBox* chkAxisAlignedSlices;
    QLabel* _segmentationGrowthWarning{nullptr};
    QLabel* _sliceStepLabel{nullptr};
    QString _segmentationGrowthStatusText;


    Ui_VCMainWindow ui;
    QTabWidget* _workspaceTabs{nullptr};
    QMainWindow* _segmentWorkspaceWindow{nullptr};
    QMainWindow* _lasagnaWorkspaceWindow{nullptr};
    QMainWindow* _atlasWorkspaceWindow{nullptr};
    QMainWindow* _fiberSliceWorkspaceWindow{nullptr};
    QMainWindow* _intersectionsWorkspaceWindow{nullptr};
    QDockWidget* _atlasOverviewDock{nullptr};
    QDockWidget* _atlasSearchDock{nullptr};
    QDockWidget* _atlasWorkspaceOverviewDock{nullptr};
    QDockWidget* _atlasWorkspaceSearchDock{nullptr};
    VolumeViewerBase* _atlasViewer{nullptr};
    std::optional<std::filesystem::path> _currentAtlasDir;
    std::string _currentAtlasName;
    vc::atlas::FiberSpatialIndex _fiberIntersectionIndex;
    vc::atlas::FiberIntersectionCache _fiberIntersectionCache;
    std::vector<vc::atlas::FiberIntersectionResult> _atlasSearchResults;
    std::unordered_map<uint64_t, AtlasSearchFiberSnapshot> _atlasSearchFiberSnapshotsByRuntimeId;
    std::optional<std::filesystem::path> _atlasSearchLasagnaManifestPath;
    int _atlasSearchPreviewGeneration{0};
    std::optional<int> _atlasSearchHoveredResult;
    std::set<int> _atlasSearchSelectedResults;
    std::set<int> _atlasSearchPreviewRequestedResults;
    bool _atlasSearchCancelRequested{false};
    QMdiArea *mdiArea;
    QMdiArea* _fiberSliceMdiArea{nullptr};
    QMdiArea* _intersectionsMdiArea{nullptr};

    bool can_change_volume_();

    size_t _cacheSizeBytes = 0;

    std::unique_ptr<VolumeOverlayController> _volumeOverlay;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<ViewerControlsPanel> _viewerControlsPanel;
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
    std::unique_ptr<PlaneSlicingOverlayController> _planeSlicingOverlay;
    std::unique_ptr<SurfaceRotationOverlayController> _surfaceRotationOverlay;
    std::unique_ptr<AtlasOverlayController> _atlasOverlay;
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
