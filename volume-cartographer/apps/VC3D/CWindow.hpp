#pragma once

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <QComboBox>
#include "ui_VCMain.h"

#include "vc/ui/VCCollection.hpp"

#include <QShortcut>

#include "CPointCollectionWidget.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "DrawingWidget.hpp"
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

class CWindow : public QMainWindow
{

    Q_OBJECT

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
    void onTagChanged(void);
    void onSurfaceContextMenuRequested(const QPoint& pos);
    void onRenderSegment(const std::string& segmentId);
    void onGrowSegmentFromSegment(const std::string& segmentId);
    void onAddOverlap(const std::string& segmentId);
    void onConvertToObj(const std::string& segmentId);
    void onSlimFlattenAndRender(const std::string& segmentId);
    void onGrowSeeds(const std::string& segmentId, bool isExpand, bool isRandomSeed = false);
    void onToggleConsoleOutput();
    void onDeleteSegments(const std::vector<std::string>& segmentIds);
    void onVoxelizePaths();
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
    void UpdateSurfaceTreeIcon(SurfaceTreeWidgetItem *item);

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
    void LoadSurfaces(bool reload = false);
    
    // Incremental surface loading methods
    struct SurfaceChanges {
        std::vector<std::string> toAdd;
        std::vector<std::string> toRemove;
    };
    SurfaceChanges DetectSurfaceChanges();
    void AddSingleSegmentation(const std::string& segId);
    void RemoveSingleSegmentation(const std::string& segId);
    void LoadSurfacesIncremental();

    static void audio_callback(void *user_data, uint8_t *raw_buffer, int bytes);
    void playPing();

    void setVolume(std::shared_ptr<Volume> newvol);

private slots:
    void Open(void);
    void Open(const QString& path);
    void OpenRecent();
    void Keybindings(void);
    void About(void);
    void ShowSettings();
    void ResetSegmentationViews();
    void onSurfaceSelected();
    void onSegFilterChanged(int index);
    void onSegmentationDirChanged(int index);
    void onEditMaskPressed();
    void onInpaintTeleaSelected();
    void onRefreshSurfaces();
    void onGenerateReviewReport();
    void onDrawBBoxToggled(bool enabled);
    void onSurfaceFromSelection();
    void onSelectionClear();
    void onManualLocationChanged();
    void onZoomIn();
    void onZoomOut();
    void onCopyCoordinates();
    void onImportObjAsPatches();

private:
    bool appInitComplete{false};
    std::shared_ptr<VolumePkg> fVpkg;
    Surface *_seg_surf;
    QString fVpkgPath;
    std::string fVpkgName;

    std::shared_ptr<Volume> currentVolume;
    std::string currentVolumeId;
    int loc[3] = {0,0,0};

    static const int AMPLITUDE = 28000;
    static const int FREQUENCY = 44100;

    // window components
    QMenu* fFileMenu;
    QMenu* fEditMenu;
    QMenu* fViewMenu;
    QMenu* fActionsMenu;
    QMenu* fSelectionMenu;
    QMenu* fHelpMenu;
    QMenu* fRecentVolpkgMenu{};

    QAction* fOpenVolAct;
    QAction* fOpenRecentVolpkg[MAX_RECENT_VOLPKG]{};
    QAction* fSettingsAct;
    QAction* fExitAct;
    QAction* fKeybinds;
    QAction* fAboutAct;
    QAction* fResetMdiView;
    QAction* fShowConsoleOutputAct;
    QAction* fReportingAct;
    QAction* fVoxelizePathsAct;
    QAction* fDrawBBoxAct;
    QAction* fSelectionSurfaceFromAct;
    QAction* fSelectionClearAct;
    QAction* fInpaintTeleaAct = nullptr;

    // Selection dock
    QDockWidget* _dockSelection = nullptr;
    QPushButton* _btnSurfaceFromSelection = nullptr;

    QComboBox* volSelect;
    QCheckBox* chkFilterFocusPoints;
   QComboBox* cmbPointSetFilter;
   QPushButton* btnPointSetFilterAll;
   QPushButton* btnPointSetFilterNone;
   QComboBox* cmbPointSetFilterMode;
   QCheckBox* chkFilterUnreviewed;
    QCheckBox* chkFilterRevisit;
    QCheckBox* chkFilterNoExpansion;
    QCheckBox* chkFilterNoDefective;
    QCheckBox* chkFilterPartialReview;
    QCheckBox* chkFilterCurrentOnly;
    QCheckBox* chkFilterHideUnapproved;
    QCheckBox* chkFilterInspectOnly;
    QComboBox* cmbSegmentationDir;
    
    QCheckBox* _chkApproved;
    QCheckBox* _chkDefective;
    QCheckBox* _chkReviewed;
    QCheckBox* _chkRevisit;
    QCheckBox* _chkInspect;
    QuadSurface *_surf;
    std::string _surfID;
    
  
    SeedingWidget* _seedingWidget;
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


    Ui_VCMainWindow ui;
    QMdiArea *mdiArea;

    bool can_change_volume_();
    
    ChunkCache *chunk_cache;
    std::vector<CVolumeViewer*> _viewers;
    CSurfaceCollection *_surf_col;

    std::unordered_map<std::string, OpChain*> _opchains;

    // runner for command line tools 
    CommandLineToolRunner* _cmdRunner;
    
    // Keyboard shortcuts
    QShortcut* fReviewedShortcut;
    QShortcut* fRevisitShortcut;
    QShortcut* fDefectiveShortcut;
    QShortcut* fDrawingModeShortcut;
    QShortcut* fCompositeViewShortcut;
    QShortcut* fDirectionHintsShortcut;

    QAction* fImportObjAct;
};  // class CWindow

