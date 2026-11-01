#pragma once
#include <vc_reader/catalog.hpp>

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
#include "overlays/VolumeOverlayController.hpp"

class CChunkedVolumeViewer;
#include "ViewerManager.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "SeedingWidget.hpp"

class Volume;
class VolumePkg;
class Surface;
class QuadSurface;
class QTreeWidget;
class QStandardItemModel;

#define MAX_RECENT_VOLPKG 10

// Project JSON schema version required by this app.


//forward declaration to avoid circular inclusion as CommandLineToolRunner needs CWindow.hpp
class CommandLineToolRunner;
class SegmentationModule;
class SurfacePanelController;
class SurfaceCatalog;
class StoreSurfaceOverlayController;
class MenuActionController;
class CommandCatalog;
class VolumeAttachmentController;
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
class LineAnnotationController;
class FiberCollectionController;
class WrapAnnotationWidget;
class StatusDockPanelHost;
class ViewerCompositePanel;
class LineAnnotationDialog;
class SpiralWorkspace;
class ProjectWorkspace;
class FiberMapWorkspace;
namespace vc3d::fibers3d { class Fiber3DPanel; }

class CWindow : public QMainWindow
{

    Q_OBJECT

    friend class MenuActionController;
    friend class VolumeAttachmentController;
    friend class AgentBridgeServer;

public:
    // Attach a store beside the open project (reported in the status bar on failure).
    void attachStore(const QString& path);

public slots:
    void onShowStatusMessage(QString text, int timeout);
    void onVisLasagnaObj(const std::string& segmentId);
    void onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                   SegmentationGrowthDirection direction,
                                   int steps,
                                   bool inpaintOnly);
    void onFocusPOIChanged(std::string name, POI* poi);
    void onPointDoubleClicked(vc::PointRef point);
    void onCopyWithNtRequested();
    void onFocusViewsRequested(uint64_t collectionId, std::optional<vc::PointRef> point);

public:
    // Status bar and conflict prompt for an attached hosted replica.
    void updateHubStatus();
    void promptHubConflicts();
    enum class VolumeOpenError {
        None,
        PackageLoadFailed,
        VolumeNotFound,
    };

    explicit CWindow(size_t cacheSizeGB = CHUNK_CACHE_SIZE_GB,
                     bool debugDownloadQueue = false);
    ~CWindow(void);

    bool openVolumePackage(const QString& path,
                           bool interactive = true,
                           QString* errorMessage = nullptr,
                           const QString& preferredVolumeId = {},
                           VolumeOpenError* openError = nullptr);

    // Helper method to get the current volume path
    QString getCurrentVolumePath() const;
    // Menu actions, keyboard shortcuts and status bar panels, by stable id.
    CommandCatalog* commandCatalog() const { return _commandCatalog; }
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
    void updateProjectNameLabel();
    // Pushes the annotation frame's voxel size to the fiber panels so their
    // lengths read in centimetres where the package can say, then refreshes
    // the status bar's fiber-length readout.
    void updateFiberLengthUnits();
    // The status bar's fiber-length readout: the line length of the fiber
    // open in the active Line Annotation tab; hidden for any other tab.
    void updateLineAnnotationLengthLabel();
    void populateDockToggleMenu(QMenu* menu) const;
    void switchToLasagnaWorkspace();
    void switchToMainWorkspace();
    void openLineAnnotationWorkspace(LineAnnotationDialog* dialog, const QString& title);
    void repeatLastLasagnaAction();
    void selectLasagnaOutputSurface(const QString& surfaceUuid);
    // List the current Open Data volume's catalog surfaces beside the user's.
    void updateCatalogSurfaceSources();
    // The active surface's Lasagna results (attached to it in the store).
    void loadLasagnaResults(Surface* surface);

    void UpdateView(void);
    void UpdateVolpkgLabel(int filterCounter);
    void updateVolumePackageEmptyState();
    void showStatusBarMessage(const QString& text, int timeout = 0);
    void clearStatusBarMessage();


    // Helper method for command line tools
    bool initializeCommandLineRunner(void);

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
    enum class VolumeAttachResult {
        Attached,
        AlreadyAttached,
        VolumeIdConflict,
    };
    VolumeAttachResult attachVolumeToCurrentPackage(
        const std::shared_ptr<Volume>& volume,
        const QString& location,
        std::vector<std::string> tags = {},
        const QString& preferredVolumeId = {});
    void refreshCurrentVolumePackageUi(const QString& preferredVolumeId = QString());
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

private slots:
    void onManualLocationChanged();
    void onFocusBoundsEdited();
    void onFocusBoundsToggled(bool enabled);
    void onZoomIn();
    void onZoomOut();
    void onCopyCoordinates();
    void onResetAxisAlignedRotations();
    void onAxisAlignedSlicesToggled(bool enabled);
    void onAxisOverlayVisibilityToggled(bool enabled);
    void onAxisOverlayOpacityChanged(int value);
    void onMoveOnSurfaceChangedToggled(bool enabled);
    void onSegmentationEditingModeChanged(bool enabled);
    void onSegmentationStopToolsRequested();
    void configureChunkedViewerConnections(CChunkedVolumeViewer* viewer);
    void refreshFocusBoundsUi();

    CChunkedVolumeViewer* segmentationViewer() const;
    VolumeViewerBase* segmentationBaseViewer() const;
    VolumeViewerBase* activeBaseViewer() const;
    std::vector<QComboBox*> volumeSelectionControls() const;
    void connectVolumeSelector(QComboBox* selector);
    void clearSurfaceSelection();
    QString lastVolumeSettingKeyForCurrentPackage() const;
    QString rememberedVolumeIdForCurrentPackage() const;
    void rememberCurrentVolumeForPackage(const QString& volumeId) const;
    // Volume coordinates of `from` in `to` (the two volumes' store voxels) when
    // switching volumes or working levels, when the two are related.
    std::optional<vcr::catalog::Affine> volumeTransformForSwitch(const vc::objects::VolumeVoxels& from,
                                                                 const vc::objects::VolumeVoxels& to) const;
    // Reopen the current volume at working level `level` (the same volume, its
    // voxels 2^level of its own); refused while a surface is being edited.
    void setVolumeLevel(int level);
    void syncVolumeLevelControl();
    void resetSegmentationViews(bool persistLayout = true);
    void onSurfaceActivated(const QString& surfaceId, const std::shared_ptr<QuadSurface>& surface);
    void onSurfaceActivatedPreserveEditing(const QString& surfaceId, const std::shared_ptr<QuadSurface>& surface);
    void onSegmentationGrowthStatusChanged(bool running);
    void updateSharedStatusLabel();
    void onZScrollSensitivityChanged(double sensitivity);
    void onSharedCacheStatsChanged(const QStringList& items);
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void onConvertPointToAnchor(vc::PointRef point);
    void refreshVolumeSelectionUi(const QString& preferredVolumeId = QString());

private:
    CState* _state;

    QComboBox* volSelect{nullptr};
    QSpinBox* _volumeLevelSpin{nullptr};
    std::vector<QPointer<QComboBox>> _annotationVolumeSelects;


    SeedingWidget* _seedingWidget;
    SegmentationWidget* _segmentationWidget{nullptr};
    QDockWidget* _lasagnaDock{nullptr};
    CPointCollectionWidget* _point_collection_widget;
    WrapAnnotationWidget* _wrapAnnotationWidget{nullptr};
    CFiberWidget* _fiberWidget{nullptr};
    QDockWidget* _fibers3DDock{nullptr};
    vc3d::fibers3d::Fiber3DPanel* _fibers3DPanel{nullptr};

    SurfaceTreeWidget *treeWidgetSurfaces;

    //TODO abstract these into separate QWidget class?
    QLineEdit* lblLocFocus;
    QCheckBox* chkAxisAlignedSlices;
    QLabel* _segmentationGrowthWarning{nullptr};
    QLabel* _statusMessageLabel{nullptr};
    // Owned by the status bar, which the destructor detaches (and Qt then
    // deletes) before the volume closes: a QPointer, so the late refreshes
    // that close triggers see it gone.
    QPointer<QLabel> _lineAnnotationLengthLabel;
    QLabel* _sharedCacheStatsLabel{nullptr};
    QLabel* _persistentCacheLowSpaceLabel{nullptr};
    QLabel* _hubStatusLabel{nullptr};
    std::vector<std::string> _hubPromptedConflicts;
    QLabel* _persistentCacheWarningText{nullptr};
    QFrame* _persistentCacheWarningBanner{nullptr};
    QTimer* _statusMessageTimer{nullptr};
    QTimer* _persistentCacheSpaceTimer{nullptr};
    bool _persistentCacheBannerShownThisSession{false};
    QStringList _sharedCacheStatsItems;
    QString _segmentationGrowthStatusText;
    bool _relayingNativeStatusMessage{false};


    Ui_VCMainWindow ui;
    bool _destroyingWindow{false};
    bool _spiralCloseGuardBypass{false};
    QTabWidget* _workspaceTabs{nullptr};
    QLabel* _projectNameLabel{nullptr};
    QMainWindow* _segmentWorkspaceWindow{nullptr};
    StatusDockPanelHost* _statusDockPanelHost{nullptr};
    QMainWindow* _lasagnaWorkspaceWindow{nullptr};
    QMainWindow* _spiralWorkspaceWindow{nullptr};
    SpiralWorkspace* _spiralWorkspace{nullptr};
    FiberMapWorkspace* _fiberMapWorkspace{nullptr};
    ProjectWorkspace* _projectWorkspace{nullptr};
    QDockWidget* _inkDetectionDock{nullptr};
    QDockWidget* _transformsDock{nullptr};
    QMdiArea *mdiArea;

    bool can_change_volume_();

    size_t _cacheSizeBytes = 0;

    std::unique_ptr<VolumeOverlayController> _volumeOverlay;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<ViewerControlsPanel> _viewerControlsPanel;
    ViewerCompositePanel* _viewerCompositePanel{nullptr};
    bool _mirrorCursorToSegmentation{false};
    std::unique_ptr<SegmentationGrower> _segmentationGrower;

    // Surface overlay multi-select state

    std::unique_ptr<SegmentationEditManager> _segmentationEdit;
    std::unique_ptr<SegmentationOverlayController> _segmentationOverlay;
    std::unique_ptr<PointsOverlayController> _pointsOverlay;
    std::unique_ptr<RawPointsOverlayController> _rawPointsOverlay;
    std::unique_ptr<PathsOverlayController> _pathsOverlay;
    std::unique_ptr<BBoxOverlayController> _bboxOverlay;
    std::unique_ptr<VectorOverlayController> _vectorOverlay;
    std::unique_ptr<InkDetectionOverlayController> _inkDetectionOverlay;
    std::unique_ptr<PlaneSlicingOverlayController> _planeSlicingOverlay;
    std::unique_ptr<StoreSurfaceOverlayController> _storeSurfaceOverlay;
    std::unique_ptr<SurfaceRotationOverlayController> _surfaceRotationOverlay;
    std::unique_ptr<FiberCollectionController> _fiberCollection;
    std::unique_ptr<SegmentationModule> _segmentationModule;
    std::unique_ptr<SurfaceCatalog> _surfaceCatalog;
    std::unique_ptr<SurfacePanelController> _surfacePanel;
    std::unique_ptr<VolumeAttachmentController> _volumeAttachmentController;
    std::unique_ptr<MenuActionController> _menuController;
    // A QObject child of this window.
    CommandCatalog* _commandCatalog{nullptr};
    // runner for command line tools
    CommandLineToolRunner* _cmdRunner;
    bool _debugDownloadQueue{false};
    bool _normalGridAvailable{false};
    QString _normalGridPath;

    std::unique_ptr<FileWatcherService> _fileWatcher;
    std::unique_ptr<AxisAlignedSliceController> _axisAlignedSliceController;
    std::unique_ptr<SegmentationCommandHandler> _segmentationCommandHandler;
    std::unique_ptr<LineAnnotationController> _lineAnnotationController;
    // Keyboard shortcuts
    QAction* fCompositeViewShortcut;
    QAction* fSurfaceNormalsShortcut;
    QAction* fAxisAlignedSlicesShortcut;
    QAction* fZoomInShortcut;
    QAction* fZoomOutShortcut;
    QAction* fResetViewShortcut;
    QAction* fZoomToFitShortcut;
    QAction* fOpenLasagnaWorkspaceShortcut{nullptr};
    QAction* fRepeatLasagnaActionShortcut{nullptr};

    // Z offset shortcuts (Ctrl+,/. for normal direction)
    QAction* fWorldOffsetZPosShortcut;  // Ctrl+. (further/deeper)
    QAction* fWorldOffsetZNegShortcut;  // Ctrl+, (closer)

    // Segment cycling shortcuts
    QAction* fCycleNextSegmentShortcut;
    QAction* fCyclePrevSegmentShortcut;
    QAction* fApplyApprovedTagShortcut{nullptr};
    QAction* fApplyDefectiveTagShortcut{nullptr};

    QAction* fFocusedViewShortcut;
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
