#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <QComboBox>
#include <QCheckBox>
#include <QPointF>
#include <QString>
#include <memory>
#include <vector>
#include "ui_VCMain.h"

#include "vc/ui/VCCollection.hpp"

#include <QShortcut>
#include <unordered_map>
#include <map>

#include "CPointCollectionWidget.hpp"
#include "CState.hpp"
#include "tiled/CTiledVolumeViewer.hpp"
#include "DrawingWidget.hpp"
#include "segmentation/tools/SegmentationEditManager.hpp"
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
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "SeedingWidget.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/RemoteScroll.hpp"
#include "FocusHistoryManager.hpp"

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
class QSpinBox;
class QTemporaryFile;
class QTemporaryDir;
class QStandardItemModel;
class FileWatcherService;
class AxisAlignedSliceController;
class FocusHistoryManager;
class SegmentationCommandHandler;

class CWindow : public QMainWindow
{

    Q_OBJECT

    friend class MenuActionController;

public:
    enum SaveResponse : bool { Cancelled, Continue };


signals:

public slots:
    void onShowStatusMessage(QString text, int timeout);
    void onLocChanged(void);
    void onManualPlaneChanged(void);
    void onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                   SegmentationGrowthDirection direction,
                                   int steps,
                                   bool inpaintOnly);
    void onFocusPOIChanged(std::string name, POI* poi);
    void onPointDoubleClicked(uint64_t pointId);
    void onCopyWithNtRequested();
    void onFocusViewsRequested(uint64_t collectionId, uint64_t pointId);

public:
    explicit CWindow(size_t cacheSizeGB = CHUNK_CACHE_SIZE_GB);
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

    void UpdateView(void);
    void UpdateVolpkgLabel(int filterCounter);


    // Helper method for command line tools
    bool initializeCommandLineRunner(void);

    CTiledVolumeViewer *newConnectedViewer(std::string surfaceName, QString title, QMdiArea *mdiArea);
    void closeEvent(QCloseEvent* event);

    void setWidgetsEnabled(bool state);

    bool InitializeVolumePkg(const std::string& nVpkgPath);

    void OpenVolume(const QString& path);
    void CloseVolume(void);


    void setVolume(std::shared_ptr<Volume> newvol);
    bool attachVolumeToCurrentPackage(const std::shared_ptr<Volume>& volume,
                                      const QString& preferredVolumeId = QString());
    void setRemoteSurfaces(const std::vector<std::pair<std::string, std::shared_ptr<Surface>>>& surfaces);
    void refreshCurrentVolumePackageUi(const QString& preferredVolumeId = QString(),
                                       bool reloadSurfaces = true);
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
    void configureViewerConnections(CTiledVolumeViewer* viewer);

    CTiledVolumeViewer* segmentationViewer() const;
    void clearSurfaceSelection();
    void onSurfaceActivated(const QString& surfaceId, QuadSurface* surface);
    void onSurfaceActivatedPreserveEditing(const QString& surfaceId, QuadSurface* surface);
    void onSegmentationGrowthStatusChanged(bool running);
    void onSliceStepSizeChanged(int newSize);
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void onConvertPointToAnchor(uint64_t pointId, uint64_t collectionId);
    void refreshVolumeSelectionUi(const QString& preferredVolumeId = QString());
    void onPreviewTransformToggled(bool enabled);
    void onSaveTransformedRequested();
    void onLoadAffineRequested();

private:
    CState* _state;

    QComboBox* volSelect;
    QComboBox* cmbSegmentationDir;


    SeedingWidget* _seedingWidget;
    SegmentationWidget* _segmentationWidget{nullptr};
    QDockWidget* _lasagnaDock{nullptr};
    DrawingWidget* _drawingWidget;
    CPointCollectionWidget* _point_collection_widget;

    SurfaceTreeWidget *treeWidgetSurfaces;
    QPushButton *btnReloadSurfaces;

    //TODO abstract these into separate QWidget class?
    QLineEdit* lblLocFocus;
    QDoubleSpinBox* spNorm[3];
    QPushButton* btnZoomIn;
   QPushButton* btnZoomOut;
    QCheckBox* chkAxisAlignedSlices;
    QCheckBox* _previewTransformCheck{nullptr};
    QCheckBox* _invertTransformCheck{nullptr};
    QSpinBox* _transformScaleSpin{nullptr};
    QPushButton* _loadAffineButton{nullptr};
    QPushButton* _saveTransformedButton{nullptr};
    QLabel* _transformStatusLabel{nullptr};
    enum class RemoteTransformFetchState { Unknown, Pending, Available, Missing };
    std::unordered_map<std::string, RemoteTransformFetchState> _remoteTransformFetchStates;
    WindowRangeWidget* _volumeWindowWidget{nullptr};
    WindowRangeWidget* _overlayWindowWidget{nullptr};
    QLabel* _segmentationGrowthWarning{nullptr};
    QLabel* _sliceStepLabel{nullptr};
    QString _segmentationGrowthStatusText;


    Ui_VCMainWindow ui;
    QMdiArea *mdiArea;

    bool can_change_volume_();

    size_t _cacheSizeBytes = 0;

    std::unique_ptr<VolumeOverlayController> _volumeOverlay;
    std::unique_ptr<ViewerManager> _viewerManager;
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

    std::unique_ptr<FileWatcherService> _fileWatcher;
    std::unique_ptr<AxisAlignedSliceController> _axisAlignedSliceController;
    FocusHistoryManager _focusHistory;
    std::unique_ptr<SegmentationCommandHandler> _segmentationCommandHandler;
    std::shared_ptr<QuadSurface> _transformPreviewSourceSurface;
    std::shared_ptr<QuadSurface> _transformPreviewSurface;
    QString _customTransformSource;
    std::filesystem::path _customTransformLocalPath;
    std::unique_ptr<QTemporaryDir> _customTransformTempDir;

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

    // Timer for debounced window state saving
    QTimer* _windowStateSaveTimer{nullptr};
    void scheduleWindowStateSave();
    void saveWindowState();
    void refreshTransformsPanelState();
    void ensureCurrentRemoteTransformJsonAsync();
    void clearTransformPreview(bool restoreDisplayedSurface = true);
    bool applyTransformPreview(bool allowRemoteFetch = true);
    std::shared_ptr<QuadSurface> currentTransformSourceSurface() const;
    QString currentTransformSourceDescription() const;
    bool setCustomTransformSource(const QString& source, QString* errorMessage = nullptr);
    std::filesystem::path localCurrentTransformJsonPath() const;
    std::string currentRemoteTransformJsonUrl() const;
    std::filesystem::path currentTransformJsonPath(bool allowRemoteFetch = true);


};  // class CWindow
