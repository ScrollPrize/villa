#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <QComboBox>
#include <QCheckBox>
#include <QFutureWatcher>
#include <memory>
#include "ui_VCMain.h"

#include "vc/ui/VCCollection.hpp"

#include <QShortcut>
#include <unordered_map>

#include "CPointCollectionWidget.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "DrawingWidget.hpp"
#include "SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "ViewerManager.hpp"
#include "SegmentationWidget.hpp"
#include "SegmentationGrowth.hpp"
#include "OpChain.hpp"
#include "OpsList.hpp"
#include "OpsSettings.hpp"
#include "SeedingWidget.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"

#define MAX_RECENT_VOLPKG 10

// Volpkg version required by this app
static constexpr int VOLPKG_MIN_VERSION = 6;
static constexpr int VOLPKG_SLICE_MIN_INDEX = 0;


//forward declaration to avoid circular inclusion as CommandLineToolRunner needs CWindow.hpp
class CommandLineToolRunner;
class SegmentationModule;
class SurfacePanelController;
class MenuActionController;

class CWindow : public QMainWindow
{

    Q_OBJECT

    friend class MenuActionController;

public:
    enum SaveResponse : bool { Cancelled, Continue };


signals:
    void sendLocChanged(int x, int y, int z);
    void sendVolumeChanged(std::shared_ptr<Volume> vol, const std::string& volumeId);
    void sendSliceChanged(std::string,Surface*);
    void sendOpChainSelected(OpChain*);
    void sendSurfacesLoaded();
    void sendVolumeClosing(); // Signal to notify viewers before closing volume

public slots:
    void onShowStatusMessage(QString text, int timeout);
    void onLocChanged(void);
    void onManualPlaneChanged(void);
    void onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onOpChainChanged(OpChain *chain);
    void onRenderSegment(const std::string& segmentId);
    void onGrowSegmentFromSegment(const std::string& segmentId);
    void onAddOverlap(const std::string& segmentId);
    void onConvertToObj(const std::string& segmentId);
    void onSlimFlatten(const std::string& segmentId);
    void onAWSUpload(const std::string& segmentId);
    void onGrowSeeds(const std::string& segmentId, bool isExpand, bool isRandomSeed = false);
    void onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                   SegmentationGrowthDirection direction,
                                   int steps);
   void onFocusPOIChanged(std::string name, POI* poi);
    void onPointDoubleClicked(uint64_t pointId);

public:
    CWindow();
    ~CWindow(void);
    
    // Helper method to get the current volume path
    QString getCurrentVolumePath() const;
    VCCollection* pointCollection() { return _point_collection; }

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    void CreateWidgets(void);
    void CreateMenus(void);
    void CreateActions(void);

    void FillSurfaceTree(void);
    void UpdateView(void);
    void UpdateVolpkgLabel(int filterCounter);

    void UpdateRecentVolpkgActions(void);
    void UpdateRecentVolpkgList(const QString& path);
    void RemoveEntryFromRecentVolpkg(const QString& path);

    // Helper method for command line tools
    bool initializeCommandLineRunner(void);

    CVolumeViewer *newConnectedCVolumeViewer(std::string surfaceName, QString title, QMdiArea *mdiArea);
    void closeEvent(QCloseEvent* event);

    void setWidgetsEnabled(bool state);

    bool InitializeVolumePkg(const std::string& nVpkgPath);
    void setDefaultWindowWidth(std::shared_ptr<Volume> volume);

    void OpenVolume(const QString& path);
    void CloseVolume(void);

    static void audio_callback(void *user_data, uint8_t *raw_buffer, int bytes);
    void playPing();

    void setVolume(std::shared_ptr<Volume> newvol);
    void updateNormalGridAvailability();

private slots:
    void onSegmentationDirChanged(int index);
    void onEditMaskPressed();
    void onAppendMaskPressed();
    void onManualLocationChanged();
    void onZoomIn();
    void onZoomOut();
    void onCopyCoordinates();
    void onAxisAlignedSlicesToggled(bool enabled);
    void onSegmentationEditingModeChanged(bool enabled);
    void onSegmentationStopToolsRequested();
    void configureViewerConnections(CVolumeViewer* viewer);
    CVolumeViewer* segmentationViewer() const;
    void clearSurfaceSelection();
    void onSurfaceActivated(const QString& surfaceId, QuadSurface* surface, OpChain* chain);

private:
    bool appInitComplete{false};
    std::shared_ptr<VolumePkg> fVpkg;
    Surface *_seg_surf;
    QString fVpkgPath;
    std::string fVpkgName;

    std::shared_ptr<Volume> currentVolume;
    std::string currentVolumeId;
    std::string _segmentationGrowthVolumeId;
    int loc[3] = {0,0,0};

    static const int AMPLITUDE = 28000;
    static const int FREQUENCY = 44100;

    // Selection dock
    QDockWidget* _dockSelection = nullptr;
    QPushButton* _btnSurfaceFromSelection = nullptr;

    QComboBox* volSelect;
    QComboBox* cmbSegmentationDir;
    QuadSurface *_surf;
    std::string _surfID;
    
  
    SeedingWidget* _seedingWidget;
    SegmentationWidget* _segmentationWidget{nullptr};
    DrawingWidget* _drawingWidget;
    CPointCollectionWidget* _point_collection_widget;

    VCCollection* _point_collection;
    
    SurfaceTreeWidget *treeWidgetSurfaces;
    OpsList *wOpsList;
    OpsSettings *wOpsSettings;
    QPushButton *btnReloadSurfaces;
    
    //TODO abstract these into separate QWidget class?
    QLineEdit* lblLocFocus;
    QDoubleSpinBox* spNorm[3];
    QPushButton* btnZoomIn;
    QPushButton* btnZoomOut;
    QCheckBox* chkAxisAlignedSlices;


    Ui_VCMainWindow ui;
    QMdiArea *mdiArea;

    bool can_change_volume_();
    
    ChunkCache *chunk_cache;
    std::unique_ptr<ViewerManager> _viewerManager;
    CSurfaceCollection *_surf_col;
    bool _useAxisAlignedSlices{false};
    bool _segmentationGrowthRunning{false};
    std::unique_ptr<QFutureWatcher<TracerGrowthResult>> _tracerGrowthWatcher;

    std::unordered_map<std::string, OpChain*> _opchains;

    std::unique_ptr<SegmentationEditManager> _segmentationEdit;
    std::unique_ptr<SegmentationOverlayController> _segmentationOverlay;
    std::unique_ptr<PointsOverlayController> _pointsOverlay;
    std::unique_ptr<PathsOverlayController> _pathsOverlay;
    std::unique_ptr<BBoxOverlayController> _bboxOverlay;
    std::unique_ptr<VectorOverlayController> _vectorOverlay;
    std::unique_ptr<SegmentationModule> _segmentationModule;
    std::unique_ptr<SurfacePanelController> _surfacePanel;
    std::unique_ptr<MenuActionController> _menuController;
    // runner for command line tools 
    CommandLineToolRunner* _cmdRunner;
    bool _normalGridAvailable{false};
    QString _normalGridPath;
    
    // Keyboard shortcuts
    QShortcut* fReviewedShortcut;
    QShortcut* fRevisitShortcut;
    QShortcut* fDefectiveShortcut;
    QShortcut* fDrawingModeShortcut;
    QShortcut* fCompositeViewShortcut;
    QShortcut* fDirectionHintsShortcut;
    QShortcut* fAxisAlignedSlicesShortcut;

    void applySlicePlaneOrientation(Surface* sourceOverride = nullptr);

    QAction* fImportObjAct;
};  // class CWindow
