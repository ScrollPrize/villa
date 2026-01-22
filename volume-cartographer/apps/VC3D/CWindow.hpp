#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <QComboBox>
#include <QCheckBox>
#include <QFutureWatcher>
#include <QPointF>
#include <QElapsedTimer>
#include <memory>
#include <vector>
#include <deque>
#include <optional>
#include <chrono>
#include "ui_VCMain.h"

#include "vc/ui/VCCollection.hpp"

#include <QShortcut>
#include <QSet>
#include <unordered_map>
#include <map>

#include "CPointCollectionWidget.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "DrawingWidget.hpp"
#include "segmentation/SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/RawPointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/PlaneSlicingOverlayController.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/SegmentationGrowth.hpp"
#include "SeedingWidget.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <sys/inotify.h>
#include <QSocketNotifier>

#define MAX_RECENT_VOLPKG 10

// Volpkg version required by this app
static constexpr int VOLPKG_MIN_VERSION = 6;
static constexpr int VOLPKG_SLICE_MIN_INDEX = 0;


//forward declaration to avoid circular inclusion as CommandLineToolRunner needs CWindow.hpp
class CommandLineToolRunner;
class SegmentationModule;
class SurfacePanelController;
class MenuActionController;
class SegmentationGrower;
class WindowRangeWidget;
class QLabel;
class QTemporaryFile;
class QStandardItemModel;

class CWindow : public QMainWindow
{

    Q_OBJECT

    friend class MenuActionController;

public:
    enum SaveResponse : bool { Cancelled, Continue };


signals:
    void sendVolumeChanged(std::shared_ptr<Volume> vol, const std::string& volumeId);
    void sendSurfacesLoaded();
    void sendVolumeClosing(); // Signal to notify viewers before closing volume

public slots:
    void onShowStatusMessage(QString text, int timeout);
    void onLocChanged(void);
    void onManualPlaneChanged(void);
    void onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onRenderSegment(const std::string& segmentId);
    void onGrowSegmentFromSegment(const std::string& segmentId);
    void onAddOverlap(const std::string& segmentId);
    void onConvertToObj(const std::string& segmentId);
    void onCropSurfaceToValidRegion(const std::string& segmentId);
    void onFlipSurface(const std::string& segmentId, bool flipU);
    void onAlphaCompRefine(const std::string& segmentId);
    void onSlimFlatten(const std::string& segmentId);
    void onABFFlatten(const std::string& segmentId);
    void onAWSUpload(const std::string& segmentId);
    void onExportWidthChunks(const std::string& segmentId);
    void onGrowSeeds(const std::string& segmentId, bool isExpand, bool isRandomSeed = false);
    void onNeighborCopyRequested(const QString& segmentId, bool copyOut);
    void onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                   SegmentationGrowthDirection direction,
                                   int steps,
                                   bool inpaintOnly);
    void onFocusPOIChanged(std::string name, POI* poi);
    void onPointDoubleClicked(uint64_t pointId);
    void onMoveSegmentToPaths(const QString& segmentId);

public:
    CWindow();
    ~CWindow(void);
    
    // Helper method to get the current volume path
    QString getCurrentVolumePath() const;
    VCCollection* pointCollection() { return _point_collection; }

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void CreateWidgets(void);

    void UpdateView(void);
    void UpdateVolpkgLabel(int filterCounter);


    // Helper method for command line tools
    bool initializeCommandLineRunner(void);

    CVolumeViewer *newConnectedCVolumeViewer(std::string surfaceName, QString title, QMdiArea *mdiArea);
    void closeEvent(QCloseEvent* event);

    void setWidgetsEnabled(bool state);

    bool InitializeVolumePkg(const std::string& nVpkgPath);

    void OpenVolume(const QString& path);
    void CloseVolume(void);


    void setVolume(std::shared_ptr<Volume> newvol);
    void updateNormalGridAvailability();
    void toggleVolumeOverlayVisibility();
    bool centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId, bool addToHistory = false);
    bool centerFocusOnCursor();
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
    void onSegmentationEditingModeChanged(bool enabled);
    void onSegmentationStopToolsRequested();
    void configureViewerConnections(CVolumeViewer* viewer);
    CVolumeViewer* segmentationViewer() const;
    void clearSurfaceSelection();
    void onSurfaceActivated(const QString& surfaceId, QuadSurface* surface);
    void onSurfaceActivatedPreserveEditing(const QString& surfaceId, QuadSurface* surface);
    void onAxisAlignedSliceMousePress(CVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onAxisAlignedSliceMouseMove(CVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onAxisAlignedSliceMouseRelease(CVolumeViewer* viewer, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onSegmentationGrowthStatusChanged(bool running);
    void processPendingInotifyEvents();
    void onSliceStepSizeChanged(int newSize);
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void onConvertPointToAnchor(uint64_t pointId, uint64_t collectionId);

private:
    void recalcAreaForSegments(const std::vector<std::string>& ids);
    std::shared_ptr<VolumePkg> fVpkg;
    QString fVpkgPath;

    std::shared_ptr<Volume> currentVolume;
    std::string currentVolumeId;
    std::string _segmentationGrowthVolumeId;

    QComboBox* volSelect;
    QComboBox* cmbSegmentationDir;
    std::weak_ptr<QuadSurface> _surf_weak;  // Non-owning reference to active surface
    std::string _surfID;
    
  
    SeedingWidget* _seedingWidget;
    SegmentationWidget* _segmentationWidget{nullptr};
    DrawingWidget* _drawingWidget;
    CPointCollectionWidget* _point_collection_widget;

    VCCollection* _point_collection;

    SurfaceTreeWidget *treeWidgetSurfaces;
    QPushButton *btnReloadSurfaces;
    
    //TODO abstract these into separate QWidget class?
    QLineEdit* lblLocFocus;
    QDoubleSpinBox* spNorm[3];
    QPushButton* btnZoomIn;
   QPushButton* btnZoomOut;
   QCheckBox* chkAxisAlignedSlices;
    WindowRangeWidget* _volumeWindowWidget{nullptr};
    WindowRangeWidget* _overlayWindowWidget{nullptr};
    QLabel* _segmentationGrowthWarning{nullptr};
    QLabel* _sliceStepLabel{nullptr};
    QString _segmentationGrowthStatusText;


    Ui_VCMainWindow ui;
    QMdiArea *mdiArea;

    bool can_change_volume_();

    ChunkCache<uint8_t> *chunk_cache;

    std::unique_ptr<VolumeOverlayController> _volumeOverlay;
    std::unique_ptr<ViewerManager> _viewerManager;
    CSurfaceCollection *_surf_col;
    bool _useAxisAlignedSlices{false};
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
    std::unique_ptr<SegmentationModule> _segmentationModule;
    std::unique_ptr<SurfacePanelController> _surfacePanel;
    std::unique_ptr<MenuActionController> _menuController;
    // runner for command line tools 
    CommandLineToolRunner* _cmdRunner;
    bool _normalGridAvailable{false};
    QString _normalGridPath;

    struct FocusHistoryEntry {
        cv::Vec3f position;
        cv::Vec3f normal;
        std::string surfaceId;  // Store ID instead of raw pointer
    };
    std::deque<FocusHistoryEntry> _focusHistory;
    int _focusHistoryIndex{-1};
    bool _navigatingFocusHistory{false};

    void recordFocusHistory(const POI& poi);
    bool stepFocusHistory(int direction);
    
    // Keyboard shortcuts
    QShortcut* fDrawingModeShortcut;
    QShortcut* fCompositeViewShortcut;
    QShortcut* fDirectionHintsShortcut;
    QShortcut* fSurfaceNormalsShortcut;
    QShortcut* fAxisAlignedSlicesShortcut;
    QShortcut* fZoomInShortcut;
    QShortcut* fZoomOutShortcut;
    QShortcut* fResetViewShortcut;

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

    void applySlicePlaneOrientation(Surface* sourceOverride = nullptr);
    void updateAxisAlignedSliceInteraction();
    float currentAxisAlignedRotationDegrees(const std::string& surfaceName) const;
    void setAxisAlignedRotationDegrees(const std::string& surfaceName, float degrees);
    void scheduleAxisAlignedOrientationUpdate();
    void flushAxisAlignedOrientationUpdate();
    void processAxisAlignedOrientationUpdate();
    void cancelAxisAlignedOrientationTimer();
    static float normalizeDegrees(float degrees);

    struct AxisAlignedSliceDragState {
        bool active = false;
        QPointF startScenePos;
        float startRotationDegrees = 0.0f;
    };
    std::unordered_map<const CVolumeViewer*, AxisAlignedSliceDragState> _axisAlignedSliceDrags;
    float _axisAlignedSegXZRotationDeg = 0.0f;
    float _axisAlignedSegYZRotationDeg = 0.0f;

    QTimer* _axisAlignedRotationTimer{nullptr};
    bool _axisAlignedOrientationDirty{false};

    int _inotifyFd;
    QSocketNotifier* _inotifyNotifier;
    std::map<int, std::string> _watchDescriptors; // wd -> directory name
    std::map<uint32_t, std::string> _pendingMoves; // cookie -> segment ID for rename tracking

    void startWatchingWithInotify();
    void stopWatchingWithInotify();
    void onInotifyEvent();
    void processInotifySegmentAddition(const std::string& dirName, const std::string& segmentId);
    void processInotifySegmentRemoval(const std::string& dirName, const std::string& segmentId);
    void processInotifySegmentRename(const std::string& dirName, const std::string& oldId, const std::string& newId);
    void processInotifySegmentUpdate(const std::string& dirName, const std::string& segmentName);
    void scheduleInotifyProcessing();
    bool shouldSkipInotifyForSegment(const std::string& segmentId, const char* eventCategory);
    void markSegmentRecentlyEdited(const std::string& segmentId);
    void pruneExpiredRecentlyEdited();

    // Periodic timer for inotify events
    QTimer* _inotifyProcessTimer;

    // Timer for debounced window state saving
    QTimer* _windowStateSaveTimer{nullptr};
    void scheduleWindowStateSave();
    void saveWindowState();

    struct InotifyEvent {
        enum Type { Addition, Removal, Rename, Update };
        Type type;
        std::string dirName;
        std::string segmentId;
        std::string newId; // Only used for rename events
    };

    // Queue of pending inotify events
    std::vector<InotifyEvent> _pendingInotifyEvents;

    // Set to track unique segments that need updating (to avoid duplicates)
    std::set<std::pair<std::string, std::string>> _pendingSegmentUpdates; // (dirName, segmentId)
    QElapsedTimer _lastInotifyProcessTime;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> _recentlyEditedSegments;
    static constexpr int INOTIFY_THROTTLE_MS = 100;
    static constexpr int RECENT_EDIT_GRACE_SECONDS = 30;

    struct NeighborCopyJob {
        enum class Stage { None, FirstPass, SecondPass };
        Stage stage{Stage::None};
        QString segmentId;
        QString volumePath;
        QString resumeSurfacePath;
        QString outputDir;
        QString generatedSurfacePath;
        QString pass1JsonPath;
        QString pass2JsonPath;
        QString directoryPrefix;
        QString resumeOptMode{QStringLiteral("local")};
        bool copyOut{true};
        QSet<QString> baselineEntries;
        std::unique_ptr<QTemporaryFile> pass1JsonFile;
        std::unique_ptr<QTemporaryFile> pass2JsonFile;
    };

    std::optional<NeighborCopyJob> _neighborCopyJob;
    void handleNeighborCopyToolFinished(bool success);
    QString findNewNeighborSurface(const NeighborCopyJob& job) const;
    bool startNeighborCopyPass(const QString& paramsPath,
                               const QString& resumeSurface,
                               const QString& resumeOpt,
                               int ompThreads);
    void launchNeighborCopySecondPass();


};  // class CWindow
