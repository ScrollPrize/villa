#include "CWindow.hpp"

#include <QKeySequence>
#include <QKeyEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMenu>
#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QDateTime>
#include <QFileDialog>
#include <QTextStream>
#include <QFileInfo>
#include <QProgressDialog>
#include <QMessageBox>
#include <QThread>
#include <QStandardItemModel>
#include <QtConcurrent/QtConcurrent>
#include <QComboBox>
#include <QFutureWatcher>
#include <QRegularExpressionValidator>
#include <QDockWidget>
#include <QProcess>
#include <QTemporaryDir>
#include <QToolBar>
#include <QFileInfo>

#include <atomic>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"
#include "SettingsDialog.hpp"
#include "CSurfaceCollection.hpp"
#include "CPointCollectionWidget.hpp"
#include "OpChain.hpp"
#include "OpsList.hpp"
#include "OpsSettings.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "DrawingWidget.hpp"
#include "CommandLineToolRunner.hpp"

#include "vc/core/types/Exceptions.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/SurfaceVoxelizer.hpp"





using qga = QGuiApplication;


// Constructor
CWindow::CWindow() :
    fVpkg(nullptr),
    _cmdRunner(nullptr),
    _seedingWidget(nullptr),
    _drawingWidget(nullptr),
    _point_collection_widget(nullptr)
{
    _point_collection = new VCCollection(this);
    const QSettings settings("VC.ini", QSettings::IniFormat);
    setWindowIcon(QPixmap(":/images/logo.png"));
    ui.setupUi(this);
    // setAttribute(Qt::WA_DeleteOnClose);

    chunk_cache = new ChunkCache(CHUNK_CACHE_SIZE_GB*1024ULL*1024ULL*1024ULL);
    std::cout << "chunk cache size is " << CHUNK_CACHE_SIZE_GB << " gigabytes " << std::endl;
    
    _surf_col = new CSurfaceCollection();
    
    //_surf_col->setSurface("manual plane", new PlaneSurface({2000,2000,2000},{1,1,1}));
    _surf_col->setSurface("xy plane", new PlaneSurface({2000,2000,2000},{0,0,1}));
    _surf_col->setSurface("xz plane", new PlaneSurface({2000,2000,2000},{0,1,0}));
    _surf_col->setSurface("yz plane", new PlaneSurface({2000,2000,2000},{1,0,0}));
    
    // create UI widgets
    CreateWidgets();

    // create menu
    CreateActions();
    CreateMenus();
    UpdateRecentVolpkgActions();

#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    if (QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark) {
        // stylesheet
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(55, 80, 170), stop:0.8 rgb(225, 90, 80), stop:1 rgb(225, 150, 0)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(235, 180, 30); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(55, 55, 55); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(60, 60, 75); }"
            "QTabBar::tab { background: rgb(60, 60, 75); }"
            "QWidget#tabSegment { background: rgb(55, 55, 55); }";
        setStyleSheet(style);
    } else
#endif
    {
        // stylesheet
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(85, 110, 200), stop:0.8 rgb(255, 120, 110), stop:1 rgb(255, 180, 30)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(255, 200, 50); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(245, 245, 255); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(205, 210, 240); }"
            "QTabBar::tab { background: rgb(205, 210, 240); }"
            "QWidget#tabSegment { background: rgb(245, 245, 255); }"
            "QRadioButton:disabled { color: gray; }";
        setStyleSheet(style);
    }

    // Restore geometry / sizes
    const QSettings geometry;
    if (geometry.contains("mainWin/geometry")) {
        restoreGeometry(geometry.value("mainWin/geometry").toByteArray());
    }
    if (geometry.contains("mainWin/state")) {
        restoreState(geometry.value("mainWin/state").toByteArray());
    }

    // If enabled, auto open the last used volpkg
    if (settings.value("volpkg/auto_open", false).toInt() != 0) {

        QStringList files = settings.value("volpkg/recent").toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            Open(files[0]);
        }
    }

    // Create application-wide keyboard shortcuts
    fReviewedShortcut = new QShortcut(QKeySequence("R"), this);
    fReviewedShortcut->setContext(Qt::ApplicationShortcut);
    connect(fReviewedShortcut, &QShortcut::activated, [this]() {
        if (_chkReviewed && _surf) {
            _chkReviewed->setCheckState(_chkReviewed->checkState() == Qt::Unchecked ? Qt::Checked : Qt::Unchecked);
        }
    });
    
    fRevisitShortcut = new QShortcut(QKeySequence("Shift+R"), this);
    fRevisitShortcut->setContext(Qt::ApplicationShortcut);
    connect(fRevisitShortcut, &QShortcut::activated, [this]() {
        if (_chkRevisit && _surf) {
            _chkRevisit->setCheckState(_chkRevisit->checkState() == Qt::Unchecked ? Qt::Checked : Qt::Unchecked);
        }
    });
    
    fDefectiveShortcut = new QShortcut(QKeySequence("Shift+D"), this);
    fDefectiveShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDefectiveShortcut, &QShortcut::activated, [this]() {
        if (_chkDefective && _surf) {
            _chkDefective->setCheckState(_chkDefective->checkState() == Qt::Unchecked ? Qt::Checked : Qt::Unchecked);
        }
    });
    
    fDrawingModeShortcut = new QShortcut(QKeySequence("D"), this);
    fDrawingModeShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDrawingModeShortcut, &QShortcut::activated, [this]() {
        if (_drawingWidget) {
            _drawingWidget->toggleDrawingMode();
        }
    });
    
    fCompositeViewShortcut = new QShortcut(QKeySequence("C"), this);
    fCompositeViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCompositeViewShortcut, &QShortcut::activated, [this]() {
        // Find the segmentation viewer and toggle its composite view
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeEnabled(!viewer->isCompositeEnabled());
                break;
            }
        }
    });

    // Toggle direction hints overlay (Ctrl+T)
    fDirectionHintsShortcut = new QShortcut(QKeySequence("Ctrl+T"), this);
    fDirectionHintsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDirectionHintsShortcut, &QShortcut::activated, [this]() {
        QSettings settings("VC.ini", QSettings::IniFormat);
        bool current = settings.value("viewer/show_direction_hints", true).toBool();
        bool next = !current;
        settings.setValue("viewer/show_direction_hints", next ? "1" : "0");
        for (auto& viewer : _viewers) {
            viewer->setShowDirectionHints(next);
        }
    });

    appInitComplete = true;
}

// Destructor
CWindow::~CWindow(void)
{
    setStatusBar(nullptr);

    CloseVolume();
    delete chunk_cache;
    delete _surf_col;
    delete _point_collection;
}

CVolumeViewer *CWindow::newConnectedCVolumeViewer(std::string surfaceName, QString title, QMdiArea *mdiArea)
{
    auto volView = new CVolumeViewer(_surf_col, mdiArea);
    QMdiSubWindow *win = mdiArea->addSubWindow(volView);
    win->setWindowTitle(title);
    win->setWindowFlags(Qt::WindowTitleHint | Qt::WindowMinMaxButtonsHint);
    volView->setCache(chunk_cache);
    connect(this, &CWindow::sendVolumeChanged, volView, &CVolumeViewer::OnVolumeChanged);
    volView->setPointCollection(_point_collection);
    connect(_point_collection, &VCCollection::pointAdded, volView, &CVolumeViewer::onPointAdded);
    connect(_point_collection, &VCCollection::pointChanged, volView, &CVolumeViewer::onPointChanged);
    connect(_point_collection, &VCCollection::pointRemoved, volView, &CVolumeViewer::onPointRemoved);
    connect(_surf_col, &CSurfaceCollection::sendSurfaceChanged, volView, &CVolumeViewer::onSurfaceChanged);
    connect(_surf_col, &CSurfaceCollection::sendPOIChanged, volView, &CVolumeViewer::onPOIChanged);
    connect(_surf_col, &CSurfaceCollection::sendPOIChanged, this, &CWindow::onFocusPOIChanged);
    connect(_surf_col, &CSurfaceCollection::sendIntersectionChanged, volView, &CVolumeViewer::onIntersectionChanged);

    // Initialize viewer settings from persisted configuration
    {
        QSettings settings("VC.ini", QSettings::IniFormat);
        bool showDirHints = settings.value("viewer/show_direction_hints", true).toBool();
        volView->setShowDirectionHints(showDirHints);
    }
    connect(volView, &CVolumeViewer::sendVolumeClicked, this, &CWindow::onVolumeClicked);
    connect(this, &CWindow::sendVolumeClosing, volView, &CVolumeViewer::onVolumeClosing);

    QSettings settings("VC.ini", QSettings::IniFormat);
    bool resetViewOnSurfaceChange = settings.value("viewer/reset_view_on_surface_change", true).toBool();
    volView->setResetViewOnSurfaceChange(resetViewOnSurfaceChange);


    volView->setSurface(surfaceName);
    
    _viewers.push_back(volView);
    
    return volView;
}

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    bool keep_poi = false;
    if (currentVolume && currentVolume->sliceWidth() == newvol->sliceWidth() && currentVolume->sliceHeight() == newvol->sliceHeight() && currentVolume->numSlices() == newvol->numSlices()) {
        keep_poi = true;
    }
    currentVolume = newvol;
    
    // Find the volume ID for the current volume
    currentVolumeId.clear();
    if (fVpkg && currentVolume) {
        for (const auto& id : fVpkg->volumeIDs()) {
            if (fVpkg->volume(id) == currentVolume) {
                currentVolumeId = id;
                break;
            }
        }
    }

    sendVolumeChanged(currentVolume, currentVolumeId);

    if (currentVolume->numScales() >= 2)
        wOpsList->setDataset(currentVolume->zarrDataset(1), chunk_cache, 0.5);
    else
        wOpsList->setDataset(currentVolume->zarrDataset(0), chunk_cache, 1.0);
    
    int w = currentVolume->sliceWidth();
    int h = currentVolume->sliceHeight();
    int d = currentVolume->numSlices();
    

    if (!keep_poi) {
        // Set default focus at middle of volume
        POI *poi = _surf_col->poi("focus");
        if (!poi) {
            poi = new POI;
        }
        poi->p = cv::Vec3f(w/2, h/2, d/2);
        poi->n = cv::Vec3f(0, 0, 1); // Default normal for XY plane
        _surf_col->setPOI("focus", poi);
    }

    onManualPlaneChanged();
}

// Create widgets
void CWindow::CreateWidgets(void)
{
    QSettings settings("VC.ini", QSettings::IniFormat);

    // add volume viewer
    auto aWidgetLayout = new QVBoxLayout;
    ui.tabSegment->setLayout(aWidgetLayout);
    
    mdiArea = new QMdiArea(ui.tabSegment);
    aWidgetLayout->addWidget(mdiArea);
    
    // newConnectedCVolumeViewer("manual plane", tr("Manual Plane"), mdiArea);
    newConnectedCVolumeViewer("seg xz", tr("Segmentation XZ"), mdiArea)->setIntersects({"segmentation"});
    newConnectedCVolumeViewer("seg yz", tr("Segmentation YZ"), mdiArea)->setIntersects({"segmentation"});
    newConnectedCVolumeViewer("xy plane", tr("XY / Slices"), mdiArea)->setIntersects({"segmentation"});
    newConnectedCVolumeViewer("segmentation", tr("Surface"), mdiArea)->setIntersects({"seg xz","seg yz"});
    mdiArea->tileSubWindows();

    treeWidgetSurfaces = ui.treeWidgetSurfaces;
    treeWidgetSurfaces->setContextMenuPolicy(Qt::CustomContextMenu);
    treeWidgetSurfaces->setSelectionMode(QAbstractItemView::ExtendedSelection);
    connect(treeWidgetSurfaces, &QWidget::customContextMenuRequested, this, &CWindow::onSurfaceContextMenuRequested);
    btnReloadSurfaces = ui.btnReloadSurfaces;

    wOpsList = new OpsList(ui.dockWidgetOpList);
    ui.dockWidgetOpList->setWidget(wOpsList);
    wOpsSettings = new OpsSettings(ui.dockWidgetOpSettings);
    ui.dockWidgetOpSettings->setWidget(wOpsSettings);
    
    // i recognize that having both a seeding widget and a drawing widget that both handle mouse events and paths is redundant, 
    // but i can't find an easy way yet to merge them and maintain the path iteration that the seeding widget currently uses
    // so for now we have both. i suppose i could probably add a 'mode' , but for now i will just hate this section :(


    // Create Drawing widget
    _drawingWidget = new DrawingWidget(ui.dockWidgetDrawing);
    ui.dockWidgetDrawing->setWidget(_drawingWidget);

    connect(this, &CWindow::sendVolumeChanged, _drawingWidget, 
            static_cast<void (DrawingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&DrawingWidget::onVolumeChanged));
    connect(_drawingWidget, &DrawingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(this, &CWindow::sendSurfacesLoaded, _drawingWidget, &DrawingWidget::onSurfacesLoaded);

    _drawingWidget->setCache(chunk_cache);
    
    // Create Seeding widget
    _seedingWidget = new SeedingWidget(_point_collection, _surf_col, ui.dockWidgetDistanceTransform);
    ui.dockWidgetDistanceTransform->setWidget(_seedingWidget);
    
    connect(this, &CWindow::sendVolumeChanged, _seedingWidget, 
            static_cast<void (SeedingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&SeedingWidget::onVolumeChanged));
    connect(_seedingWidget, &SeedingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(this, &CWindow::sendSurfacesLoaded, _seedingWidget, &SeedingWidget::onSurfacesLoaded);
    
    _seedingWidget->setCache(chunk_cache);
    
    // Connect seeding and drawing widgets to the volume viewers
    for (auto& viewer : _viewers) {

        connect(_drawingWidget, &DrawingWidget::sendPathsChanged,
                viewer, &CVolumeViewer::onPathsChanged);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMousePress,
                viewer, &CVolumeViewer::onMousePress);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseMove,
                viewer, &CVolumeViewer::onMouseMove);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseRelease,
                viewer, &CVolumeViewer::onMouseRelease);
        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                _drawingWidget, &DrawingWidget::onMousePress);
        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                _drawingWidget, &DrawingWidget::onMouseMove);
        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                _drawingWidget, &DrawingWidget::onMouseRelease);
        connect(viewer, &CVolumeViewer::sendZSliceChanged,
                _drawingWidget, &DrawingWidget::updateCurrentZSlice);

        // Selection is stored in the viewer; actions are triggered from Selection dock
        
        // Connect drawing mode signal to update cursor
        connect(_drawingWidget, &DrawingWidget::sendDrawingModeActive,
                [viewer, this](bool active) {
                    viewer->onDrawingModeActive(active, 
                        _drawingWidget->getBrushSize(),
                        _drawingWidget->getBrushShape() == PathData::BrushShape::SQUARE);
                });
    }
    
    for (auto& viewer : _viewers) {
        connect(_seedingWidget, &SeedingWidget::sendPathsChanged,
                viewer, &CVolumeViewer::onPathsChanged);
        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                _seedingWidget, &SeedingWidget::onMousePress);
        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                _seedingWidget, &SeedingWidget::onMouseMove);
        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                _seedingWidget, &SeedingWidget::onMouseRelease);
        connect(viewer, &CVolumeViewer::sendZSliceChanged,
                _seedingWidget, &SeedingWidget::updateCurrentZSlice);
    }
    
    // Create and add the point collection widget
    _point_collection_widget = new CPointCollectionWidget(_point_collection, this);
    _point_collection_widget->setObjectName("pointCollectionDock");
    addDockWidget(Qt::RightDockWidgetArea, _point_collection_widget);

    // Selection dock (removed per request; selection actions remain in the menu)

    for (auto& viewer : _viewers) {
        connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected, viewer, &CVolumeViewer::onCollectionSelected);
        connect(viewer, &CVolumeViewer::sendCollectionSelected, _point_collection_widget, &CPointCollectionWidget::selectCollection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelected, viewer, &CVolumeViewer::onPointSelected);
        connect(viewer, &CVolumeViewer::pointSelected, _point_collection_widget, &CPointCollectionWidget::selectPoint);
        connect(viewer, &CVolumeViewer::pointClicked, _point_collection_widget, &CPointCollectionWidget::selectPoint);
    }
    connect(_point_collection_widget, &CPointCollectionWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);

   connect(_point_collection, &VCCollection::collectionAdded, this, [this](uint64_t id) {
       const auto& collections = _point_collection->getAllCollections();
       auto it = collections.find(id);
       if (it != collections.end()) {
           const auto& collection = it->second;
           QStandardItem* item = new QStandardItem(QString::fromStdString(collection.name));
           item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
           item->setData(Qt::Unchecked, Qt::CheckStateRole);
           qobject_cast<QStandardItemModel*>(cmbPointSetFilter->model())->appendRow(item);
       }
       onSegFilterChanged(0);
   });
   connect(_point_collection, &VCCollection::collectionRemoved, this, [this](uint64_t id) {
       // This is inefficient, but simple. A better way would be to store the mapping from id to combobox index.
       const auto& collections = _point_collection->getAllCollections();
       cmbPointSetFilter->clear();
       for (const auto& pair : collections) {
           QStandardItem* item = new QStandardItem(QString::fromStdString(pair.second.name));
           item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
           item->setData(Qt::Unchecked, Qt::CheckStateRole);
           qobject_cast<QStandardItemModel*>(cmbPointSetFilter->model())->appendRow(item);
       }
       onSegFilterChanged(0);
   });
   connect(_point_collection, &VCCollection::collectionChanged, this, [this](uint64_t id) {
       // This is inefficient, but simple. A better way would be to store the mapping from id to combobox index.
       const auto& collections = _point_collection->getAllCollections();
       cmbPointSetFilter->clear();
       for (const auto& pair : collections) {
           QStandardItem* item = new QStandardItem(QString::fromStdString(pair.second.name));
           item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
           item->setData(Qt::Unchecked, Qt::CheckStateRole);
           qobject_cast<QStandardItemModel*>(cmbPointSetFilter->model())->appendRow(item);
       }
       onSegFilterChanged(0);
   });

   connect(_point_collection, &VCCollection::pointAdded, this, [this](const ColPoint&) { onSegFilterChanged(0); });
   connect(_point_collection, &VCCollection::pointChanged, this, [this](const ColPoint&) { onSegFilterChanged(0); });
   connect(_point_collection, &VCCollection::pointRemoved, this, [this](uint64_t) { onSegFilterChanged(0); });

    // Tab the docks - Drawing first, then Seeding, then Tools
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDistanceTransform);
    tabifyDockWidget(ui.dockWidgetDistanceTransform, ui.dockWidgetDrawing);
    
    // Make Drawing dock the active tab by default
    ui.dockWidgetDrawing->raise();
    
    // Tab the composite widget with the Volume Package widget on the left dock
    tabifyDockWidget(ui.dockWidgetVolumes, ui.dockWidgetComposite);
    
    // Make Volume Package dock the active tab by default
    ui.dockWidgetVolumes->show();
    ui.dockWidgetVolumes->raise();

    connect(treeWidgetSurfaces, &QTreeWidget::itemSelectionChanged, this, &CWindow::onSurfaceSelected);
    connect(btnReloadSurfaces, &QPushButton::clicked, this, &CWindow::onRefreshSurfaces);
    connect(this, &CWindow::sendOpChainSelected, wOpsList, &OpsList::onOpChainSelected);
    connect(wOpsList, &OpsList::sendOpSelected, wOpsSettings, &OpsSettings::onOpSelected);

    connect(wOpsList, &OpsList::sendOpChainChanged, this, &CWindow::onOpChainChanged);
    connect(wOpsSettings, &OpsSettings::sendOpChainChanged, this, &CWindow::onOpChainChanged);

    // new and remove path buttons
    // connect(ui.btnNewPath, SIGNAL(clicked()), this, SLOT(OnNewPathClicked()));
    // connect(ui.btnRemovePath, SIGNAL(clicked()), this, SLOT(OnRemovePathClicked()));

    // TODO CHANGE VOLUME LOADING; FIRST CHECK FOR OTHER VOLUMES IN THE STRUCTS
    volSelect = ui.volSelect;
    connect(
        volSelect, &QComboBox::currentIndexChanged, [this](const int& index) {
            std::shared_ptr<Volume> newVolume;
            try {
                newVolume = fVpkg->volume(volSelect->currentData().toString().toStdString());
            } catch (const std::out_of_range& e) {
                QMessageBox::warning(this, "Error", "Could not load volume.");
                return;
            }
            setVolume(newVolume);
        });

    chkFilterFocusPoints = ui.chkFilterFocusPoints;
   cmbPointSetFilter = ui.cmbPointSetFilter;
   btnPointSetFilterAll = ui.btnPointSetFilterAll;
   btnPointSetFilterNone = ui.btnPointSetFilterNone;
    cmbPointSetFilterMode = new QComboBox();
    cmbPointSetFilterMode->addItem("Any (OR)");
    cmbPointSetFilterMode->addItem("All (AND)");
    ui.pointSetFilterLayout->insertWidget(1, cmbPointSetFilterMode);
    chkFilterUnreviewed = ui.chkFilterUnreviewed;
    chkFilterRevisit = ui.chkFilterRevisit;
    chkFilterNoExpansion = ui.chkFilterNoExpansion;
    chkFilterNoDefective = ui.chkFilterNoDefective;
    chkFilterPartialReview = ui.chkFilterPartialReview;
    
    connect(chkFilterFocusPoints, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });
   connect(btnPointSetFilterAll, &QPushButton::clicked, [this]() {
       for (int i = 0; i < cmbPointSetFilter->count(); ++i) {
           cmbPointSetFilter->model()->setData(cmbPointSetFilter->model()->index(i, 0), Qt::Checked, Qt::CheckStateRole);
       }
       onSegFilterChanged(0);
   });
   connect(btnPointSetFilterNone, &QPushButton::clicked, [this]() {
       for (int i = 0; i < cmbPointSetFilter->count(); ++i) {
           cmbPointSetFilter->model()->setData(cmbPointSetFilter->model()->index(i, 0), Qt::Unchecked, Qt::CheckStateRole);
       }
       onSegFilterChanged(0);
   });
    connect(chkFilterUnreviewed, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });
    connect(cmbPointSetFilterMode, &QComboBox::currentIndexChanged, this, [this]() { onSegFilterChanged(0); });
    connect(chkFilterRevisit, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });
    connect(chkFilterNoExpansion, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });
    connect(chkFilterNoDefective, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });
    connect(chkFilterPartialReview, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });

    chkFilterHideUnapproved = ui.chkFilterHideUnapproved;
    connect(chkFilterHideUnapproved, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });


    chkFilterInspectOnly = ui.chkFilterInspectOnly;
    connect(chkFilterInspectOnly, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });

    cmbSegmentationDir = ui.cmbSegmentationDir;
    connect(cmbSegmentationDir, &QComboBox::currentIndexChanged, this, &CWindow::onSegmentationDirChanged);

    // Location input element (single QLineEdit for comma-separated values)
    lblLocFocus = ui.sliceFocus;

    // Set up validator for location input (accepts digits, commas, and spaces)
    QRegularExpressionValidator* validator = new QRegularExpressionValidator(
        QRegularExpression("^\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*$"), this);
    lblLocFocus->setValidator(validator);
    connect(lblLocFocus, &QLineEdit::editingFinished, this, &CWindow::onManualLocationChanged);

    QPushButton* btnCopyCoords = ui.btnCopyCoords;
    connect(btnCopyCoords, &QPushButton::clicked, this, &CWindow::onCopyCoordinates);

    // Zoom buttons
    btnZoomIn = ui.btnZoomIn;
    btnZoomOut = ui.btnZoomOut;
    
    connect(btnZoomIn, &QPushButton::clicked, this, &CWindow::onZoomIn);
    connect(btnZoomOut, &QPushButton::clicked, this, &CWindow::onZoomOut);
    
    spNorm[0] = ui.dspNX;
    spNorm[1] = ui.dspNY;
    spNorm[2] = ui.dspNZ;
    
    _chkApproved = ui.chkApproved;
    _chkDefective = ui.chkDefective;
    _chkReviewed = ui.chkReviewed;
    _chkRevisit = ui.chkRevisit;
    _chkInspect = ui.chkInspect;
    
    for(int i=0;i<3;i++)
        spNorm[i]->setRange(-10,10);
    
    connect(spNorm[0], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[1], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[2], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    
#if (QT_VERSION < QT_VERSION_CHECK(6, 8, 0))
    connect(_chkApproved, &QCheckBox::stateChanged, this, &CWindow::onTagChanged);
    connect(_chkDefective, &QCheckBox::stateChanged, this, &CWindow::onTagChanged);
    connect(_chkReviewed, &QCheckBox::stateChanged, this, &CWindow::onTagChanged);
    connect(_chkRevisit, &QCheckBox::stateChanged, this, &CWindow::onTagChanged);
    connect(_chkInspect, &QCheckBox::stateChanged, this, &CWindow::onTagChanged);
#else
    connect(_chkApproved, &QCheckBox::checkStateChanged, this, &CWindow::onTagChanged);
    connect(_chkDefective, &QCheckBox::checkStateChanged, this, &CWindow::onTagChanged);
    connect(_chkReviewed, &QCheckBox::checkStateChanged, this, &CWindow::onTagChanged);
    connect(_chkRevisit, &QCheckBox::checkStateChanged, this, &CWindow::onTagChanged);
    connect(_chkInspect, &QCheckBox::checkStateChanged, this, &CWindow::onTagChanged);
#endif

    connect(ui.btnEditMask, &QPushButton::pressed, this, &CWindow::onEditMaskPressed);
    
    // Connect composite view controls
    connect(ui.chkCompositeEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        // Find the segmentation viewer and update its composite setting
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeEnabled(checked);
                break;
            }
        }
    });
    
    connect(ui.cmbCompositeMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        // Find the segmentation viewer and update its composite method
        std::string method = "max";
        switch (index) {
            case 0: method = "max"; break;
            case 1: method = "mean"; break;
            case 2: method = "min"; break;
            case 3: method = "alpha"; break;
        }
        
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeMethod(method);
                break;
            }
        }
    });
    
    // Connect Layers In Front controls
    connect(ui.spinLayersInFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeLayersInFront(value);
                break;
            }
        }
    });
    
    // Connect Layers Behind controls
    connect(ui.spinLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeLayersBehind(value);
                break;
            }
        }
    });
    
    // Connect Alpha Min controls
    connect(ui.spinAlphaMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeAlphaMin(value);
                break;
            }
        }
    });
    
    // Connect Alpha Max controls
    connect(ui.spinAlphaMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeAlphaMax(value);
                break;
            }
        }
    });
    
    // Connect Alpha Threshold controls
    connect(ui.spinAlphaThreshold, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeAlphaThreshold(value);
                break;
            }
        }
    });
    
    // Connect Material controls
    connect(ui.spinMaterial, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeMaterial(value);
                break;
            }
        }
    });
    
    // Connect Reverse Direction control
    connect(ui.chkReverseDirection, &QCheckBox::toggled, this, [this](bool checked) {
        for (auto& viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeReverseDirection(checked);
                break;
            }
        }
    });
    bool resetViewOnSurfaceChange = settings.value("viewer/reset_view_on_surface_change", true).toBool();
    for (auto& viewer : _viewers) {
        viewer->setResetViewOnSurfaceChange(resetViewOnSurfaceChange);
    }

    chkFilterCurrentOnly = ui.chkFilterCurrentOnly;
    connect(chkFilterCurrentOnly, &QCheckBox::toggled, [this]() { onSegFilterChanged(0); });

    // Connect Stop tools button from Tools dock
    if (ui.btnStopTools) {
        connect(ui.btnStopTools, &QPushButton::clicked, this, [this]() {
            if (!initializeCommandLineRunner()) return;
            if (_cmdRunner) {
                _cmdRunner->cancel();
                statusBar()->showMessage(tr("Cancelling running tools..."), 3000);
            }
        });
    }

}

void CWindow::onDrawBBoxToggled(bool enabled)
{
    // Toggle bbox mode on the segmentation viewer
    for (auto* viewer : _viewers) {
        if (viewer->surfName() == "segmentation") {
            viewer->setBBoxMode(enabled);
            statusBar()->showMessage(enabled ? tr("BBox mode active: drag on Surface view")
                                             : tr("BBox mode off"), 3000);
            break;
        }
    }
}
void CWindow::onSurfaceFromSelection()
{
    // Use the segmentation viewer's stored selections to create surfaces
    CVolumeViewer* segViewer = nullptr;
    for (auto* v : _viewers) if (v->surfName() == "segmentation") { segViewer = v; break; }
    if (!segViewer) { statusBar()->showMessage(tr("No Surface viewer found"), 3000); return; }

    auto sels = segViewer->selections();
    if (sels.empty()) {
        statusBar()->showMessage(tr("No selections to convert"), 3000);
        return;
    }

    if (_surfID.empty() || !fVpkg || !fVpkg->getSurface(_surfID)) {
        statusBar()->showMessage(tr("Select a segmentation first"), 3000);
        return;
    }

    auto surfMeta = fVpkg->getSurface(_surfID);
    std::filesystem::path baseSegPath = surfMeta->path; // .../paths/<uuid>
    std::filesystem::path parentDir = baseSegPath.parent_path();

    int idx = 1;
    int created = 0;
    QString ts = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    for (const auto& pr : sels) {
        const QRectF& rect = pr.first;
        std::unique_ptr<QuadSurface> filtered(segViewer->makeBBoxFilteredSurfaceFromSceneRect(rect));
        if (!filtered) continue;
        std::string newId = _surfID + std::string("_sel_") + ts.toStdString() + std::string("_") + std::to_string(idx++);
        std::filesystem::path outDir = parentDir / newId;
        try {
            filtered->save(outDir.string(), newId);
            created++;
        } catch (const std::exception& e) {
            statusBar()->showMessage(tr("Failed to save selection: ") + e.what(), 5000);
        }
    }

    if (created > 0) {
        try {
            fVpkg->refreshSegmentations();
            LoadSurfacesIncremental();
            statusBar()->showMessage(tr("Created ") + QString::number(created) + tr(" surface(s) from selection"), 5000);
        } catch (...) {
            statusBar()->showMessage(tr("Created surfaces but failed to refresh"), 5000);
        }
    } else {
        statusBar()->showMessage(tr("No surfaces created from selection"), 3000);
    }
}

void CWindow::onSelectionClear()
{
    // Clear all stored selections on the segmentation (Surface) viewer
    CVolumeViewer* segViewer = nullptr;
    for (auto* v : _viewers) if (v->surfName() == "segmentation") { segViewer = v; break; }
    if (!segViewer) { statusBar()->showMessage(tr("No Surface viewer found"), 3000); return; }
    segViewer->clearSelections();
    statusBar()->showMessage(tr("Selections cleared"), 2000);
}

// Create menus
void CWindow::CreateMenus(void)
{
    // "Recent Volpkg" menu
    fRecentVolpkgMenu = new QMenu(tr("Open &recent volpkg"), this);
    fRecentVolpkgMenu->setEnabled(false);
    for (auto& action : fOpenRecentVolpkg)
    {
        fRecentVolpkgMenu->addAction(action);
    }

    fFileMenu = new QMenu(tr("&File"), this);
    fFileMenu->addAction(fOpenVolAct);
    fFileMenu->addMenu(fRecentVolpkgMenu);
    fFileMenu->addSeparator();
    fFileMenu->addAction(fReportingAct);
    fFileMenu->addSeparator();
    fFileMenu->addAction(fSettingsAct);
    fFileMenu->addSeparator();
    fFileMenu->addAction(fExitAct);

    fEditMenu = new QMenu(tr("&Edit"), this);

    fViewMenu = new QMenu(tr("&View"), this);
    fViewMenu->addAction(ui.dockWidgetVolumes->toggleViewAction());
    fViewMenu->addAction(ui.dockWidgetSegmentation->toggleViewAction());
    fViewMenu->addAction(ui.dockWidgetDistanceTransform->toggleViewAction());
    fViewMenu->addAction(ui.dockWidgetOpList->toggleViewAction());
    fViewMenu->addAction(ui.dockWidgetDrawing->toggleViewAction());
    fViewMenu->addAction(ui.dockWidgetOpSettings->toggleViewAction());
    fViewMenu->addAction(ui.dockWidgetComposite->toggleViewAction());
    fViewMenu->addAction(ui.dockWidgetLocation->toggleViewAction());

    if (_point_collection_widget) {
        fViewMenu->addAction(_point_collection_widget->toggleViewAction());
    }

    fViewMenu->addSeparator();
    fViewMenu->addAction(fResetMdiView);
    fViewMenu->addSeparator();
    fViewMenu->addAction(fShowConsoleOutputAct);

    fActionsMenu = new QMenu(tr("&Actions"), this);
    fActionsMenu->addAction(fVoxelizePathsAct);
    fActionsMenu->addAction(fDrawBBoxAct);

    fSelectionMenu = new QMenu(tr("&Selection"), this);
    fSelectionMenu->addAction(fSelectionSurfaceFromAct);
    fSelectionMenu->addAction(fSelectionClearAct);

    // Add Telea pipeline to menus
    fActionsMenu->addSeparator();
    fActionsMenu->addAction(fInpaintTeleaAct);
    fSelectionMenu->addSeparator();
    fSelectionMenu->addAction(fInpaintTeleaAct);

    fHelpMenu = new QMenu(tr("&Help"), this);
    fHelpMenu->addAction(fKeybinds);
    fFileMenu->addSeparator();

    fHelpMenu->addAction(fAboutAct);

    menuBar()->addMenu(fFileMenu);
    menuBar()->addMenu(fEditMenu);
    menuBar()->addMenu(fViewMenu);
    menuBar()->addMenu(fActionsMenu);
    menuBar()->addMenu(fSelectionMenu);
    menuBar()->addMenu(fHelpMenu);
}

// Create actions
void CWindow::keyPressEvent(QKeyEvent* event)
{
    // Key handling moved to QShortcut objects for application-wide access
    QMainWindow::keyPressEvent(event);
}

void CWindow::CreateActions(void)
{
    fOpenVolAct = new QAction(style()->standardIcon(QStyle::SP_DialogOpenButton), tr("&Open volpkg..."), this);
    connect(fOpenVolAct, SIGNAL(triggered()), this, SLOT(Open()));
    fOpenVolAct->setShortcut(QKeySequence::Open);

    for (auto& action : fOpenRecentVolpkg)
    {
        action = new QAction(this);
        action->setVisible(false);
        connect(action, &QAction::triggered, this, &CWindow::OpenRecent);
    }

    fSettingsAct = new QAction(tr("Settings"), this);
    connect(fSettingsAct, SIGNAL(triggered()), this, SLOT(ShowSettings()));

    fExitAct = new QAction(style()->standardIcon(QStyle::SP_DialogCloseButton), tr("E&xit..."), this);
    connect(fExitAct, SIGNAL(triggered()), this, SLOT(close()));

    fKeybinds = new QAction(tr("&Keybinds"), this);
    connect(fKeybinds, SIGNAL(triggered()), this, SLOT(Keybindings()));

    fAboutAct = new QAction(tr("&About..."), this);
    connect(fAboutAct, SIGNAL(triggered()), this, SLOT(About()));

    fResetMdiView = new QAction(tr("Reset Segmentation Views"), this);
    connect(fResetMdiView, SIGNAL(triggered()), this, SLOT(ResetSegmentationViews()));
    
    fShowConsoleOutputAct = new QAction(tr("Show Console Output"), this);
    connect(fShowConsoleOutputAct, &QAction::triggered, this, &CWindow::onToggleConsoleOutput);
    
    fReportingAct = new QAction(tr("Generate Review Report..."), this);
    connect(fReportingAct, &QAction::triggered, this, &CWindow::onGenerateReviewReport);
    
    fVoxelizePathsAct = new QAction(tr("&Voxelize Paths..."), this);
    connect(fVoxelizePathsAct, &QAction::triggered, this, &CWindow::onVoxelizePaths);

    fDrawBBoxAct = new QAction(tr("Draw BBox"), this);
    fDrawBBoxAct->setCheckable(true);
    connect(fDrawBBoxAct, &QAction::toggled, this, &CWindow::onDrawBBoxToggled);

    // Selection menu actions
    fSelectionSurfaceFromAct = new QAction(tr("Surface from Selection"), this);
    connect(fSelectionSurfaceFromAct, &QAction::triggered, this, &CWindow::onSurfaceFromSelection);
    fSelectionClearAct = new QAction(tr("Clear"), this);
    connect(fSelectionClearAct, &QAction::triggered, this, &CWindow::onSelectionClear);

    // Inpaint (Telea) -> rebuild segment
    fInpaintTeleaAct = new QAction(tr("Inpaint (Telea) && Rebuild Segment"), this);
    fInpaintTeleaAct->setToolTip(tr("Generate RGB, Telea-inpaint it, then convert back to tifxyz into a new segment"));
    #if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
        fInpaintTeleaAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_I));
    #endif
    connect(fInpaintTeleaAct, &QAction::triggered, this, &CWindow::onInpaintTeleaSelected);
}

void CWindow::UpdateRecentVolpkgActions()
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    QStringList files = settings.value("volpkg/recent").toStringList();
    if (files.isEmpty()) {
        return;
    }

    // The automatic conversion to string list from the settings, (always?) adds an
    // empty entry at the end. Remove it if present.
    if (files.last().isEmpty()) {
        files.removeLast();
    }

    const int numRecentFiles = qMin(files.size(), static_cast<int>(MAX_RECENT_VOLPKG));

    for (int i = 0; i < numRecentFiles; ++i) {
        // Replace "&" with "&&" since otherwise they will be hidden and interpreted
        // as mnemonics
        QString fileName = QFileInfo(files[i]).fileName();
        fileName.replace("&", "&&");
        QString path = QFileInfo(files[i]).canonicalPath();

        if (path == "."){
            path = tr("Directory not available!");
        } else {
            path.replace("&", "&&");
        }

        QString text = tr("&%1 | %2 (%3)").arg(i + 1).arg(fileName).arg(path);
        fOpenRecentVolpkg[i]->setText(text);
        fOpenRecentVolpkg[i]->setData(files[i]);
        fOpenRecentVolpkg[i]->setVisible(true);
    }

    for (int j = numRecentFiles; j < MAX_RECENT_VOLPKG; ++j) {
        fOpenRecentVolpkg[j]->setVisible(false);
    }

    fRecentVolpkgMenu->setEnabled(numRecentFiles > 0);
}

void CWindow::UpdateRecentVolpkgList(const QString& path)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    QStringList files = settings.value("volpkg/recent").toStringList();
    const QString pathCanonical = QFileInfo(path).absoluteFilePath();
    files.removeAll(pathCanonical);
    files.prepend(pathCanonical);

    while(files.size() > MAX_RECENT_VOLPKG) {
        files.removeLast();
    }

    settings.setValue("volpkg/recent", files);

    UpdateRecentVolpkgActions();
}

void CWindow::RemoveEntryFromRecentVolpkg(const QString& path)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    QStringList files = settings.value("volpkg/recent").toStringList();
    files.removeAll(path);
    settings.setValue("volpkg/recent", files);

    UpdateRecentVolpkgActions();
}

// Asks User to Save Data Prior to VC.app Exit
void CWindow::closeEvent(QCloseEvent* event)
{
    QSettings settings;
    settings.setValue("mainWin/geometry", saveGeometry());
    settings.setValue("mainWin/state", saveState());

    QMainWindow::closeEvent(event);
}

void CWindow::setWidgetsEnabled(bool state)
{
    ui.grpVolManager->setEnabled(state);
}

auto CWindow::InitializeVolumePkg(const std::string& nVpkgPath) -> bool
{
    fVpkg = nullptr;

    try {
        fVpkg = VolumePkg::New(nVpkgPath);
    } catch (const std::exception& e) {
        Logger()->error("Failed to initialize volpkg: {}", e.what());
    }

    if (fVpkg == nullptr) {
        Logger()->error("Cannot open .volpkg: {}", nVpkgPath);
        QMessageBox::warning(
            this, "Error",
            "Volume package failed to load. Package might be corrupt.");
        return false;
    }
    return true;
}

// Update the widgets
void CWindow::UpdateView(void)
{
    if (fVpkg == nullptr) {
        setWidgetsEnabled(false);  // Disable Widgets for User
        ui.lblVpkgName->setText("[ No Volume Package Loaded ]");
        return;
    }

    setWidgetsEnabled(true);  // Enable Widgets for User

    // show volume package name
    UpdateVolpkgLabel(0);    

    volSelect->setEnabled(can_change_volume_());

    update();
}

void CWindow::UpdateVolpkgLabel(int filterCounter)
{
    if (!fVpkg) {
        return;
    }
    QString label = tr("%1").arg(QString::fromStdString(fVpkg->name()));
    ui.lblVpkgName->setText(label);
}

void CWindow::onShowStatusMessage(QString text, int timeout)
{
    statusBar()->showMessage(text, timeout);
}

std::filesystem::path seg_path_name(const std::filesystem::path &path)
{
    std::string name;
    bool store = false;
    for(auto elm : path) {
        if (store)
            name += "/"+elm.string();
        else if (elm == "paths")
            store = true;
    }
    name.erase(0,1);
    return name;
}

// Open volume package
void CWindow::OpenVolume(const QString& path)
{
    QString aVpkgPath = path;
    QSettings settings("VC.ini", QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        aVpkgPath = QFileDialog::getExistingDirectory(
            this, tr("Open Directory"), settings.value("volpkg/default_path").toString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);
        // Dialog box cancelled
        if (aVpkgPath.length() == 0) {
            Logger()->info("Open .volpkg canceled");
            return;
        }
    }

    // Checks the folder path for .volpkg extension
    auto const extension = aVpkgPath.toStdString().substr(
        aVpkgPath.toStdString().length() - 7, aVpkgPath.toStdString().length());
    if (extension != ".volpkg") {
        QMessageBox::warning(
            this, tr("ERROR"),
            "The selected file is not of the correct type: \".volpkg\"");
        Logger()->error(
            "Selected file is not .volpkg: {}", aVpkgPath.toStdString());
        fVpkg = nullptr;  // Is needed for User Experience, clears screen.
        return;
    }

    // Open volume package
    if (!InitializeVolumePkg(aVpkgPath.toStdString() + "/")) {
        return;
    }

    // Check version number
    if (fVpkg->version() < VOLPKG_MIN_VERSION) {
        const auto msg = "Volume package is version " +
                         std::to_string(fVpkg->version()) +
                         " but this program requires version " +
                         std::to_string(VOLPKG_MIN_VERSION) + "+.";
        Logger()->error(msg);
        QMessageBox::warning(this, tr("ERROR"), QString(msg.c_str()));
        fVpkg = nullptr;
        return;
    }

    fVpkgPath = aVpkgPath;
    setVolume(fVpkg->volume());
    {
        const QSignalBlocker blocker{volSelect};
        volSelect->clear();
    }
    QStringList volIds;
    for (const auto& id : fVpkg->volumeIDs()) {
        volSelect->addItem(
            QString("%1 (%2)").arg(QString::fromStdString(id)).arg(QString::fromStdString(fVpkg->volume(id)->name())),
            QVariant(QString::fromStdString(id)));
    }

    // Populate the segmentation directory dropdown
    {
        const QSignalBlocker blocker{cmbSegmentationDir};
        cmbSegmentationDir->clear();
        
        auto availableDirs = fVpkg->getAvailableSegmentationDirectories();
        for (const auto& dirName : availableDirs) {
            cmbSegmentationDir->addItem(QString::fromStdString(dirName));
        }
        
        // Select the current directory (default is "paths")
        int currentIndex = cmbSegmentationDir->findText(QString::fromStdString(fVpkg->getSegmentationDirectory()));
        if (currentIndex >= 0) {
            cmbSegmentationDir->setCurrentIndex(currentIndex);
        }
    }

    LoadSurfaces();
    UpdateRecentVolpkgList(aVpkgPath);
    
    // Set volume package in Seeding widget
   if (_seedingWidget) {
       _seedingWidget->setVolumePkg(fVpkg);
   }

   // Populate point set filter
   cmbPointSetFilter->clear();
   cmbPointSetFilter->setModel(new QStandardItemModel(this));
   connect(cmbPointSetFilter->model(), &QStandardItemModel::dataChanged, this, [this](const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles) {
        if (roles.contains(Qt::CheckStateRole)) {
            onSegFilterChanged(0);
        }
   });
   for (const auto& pair : _point_collection->getAllCollections()) {
       QStandardItem* item = new QStandardItem(QString::fromStdString(pair.second.name));
       item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
       item->setData(Qt::Unchecked, Qt::CheckStateRole);
       qobject_cast<QStandardItemModel*>(cmbPointSetFilter->model())->appendRow(item);
   }
}

void CWindow::CloseVolume(void)
{
    // Notify viewers to clear their surface pointers before we delete them
    emit sendVolumeClosing();

    // Clear surface collection first
    _surf_col->setSurface("segmentation", nullptr, true);

    // Clear all surfaces from the surface collection
    if (fVpkg) {
        for (const auto& id : fVpkg->getLoadedSurfaceIDs()) {
            _surf_col->setSurface(id, nullptr, true);
        }
        // Tell VolumePkg to unload all surfaces
        fVpkg->unloadAllSurfaces();
    }

    // Clean up OpChains (still owned by CWindow)
    for (auto& pair : _opchains) {
        delete pair.second;
    }
    _opchains.clear();

    // Clear the volume package
    fVpkg = nullptr;
    currentVolume = nullptr;

    // Update UI
    UpdateView();
    treeWidgetSurfaces->clear();
    
    // Clear points
    _point_collection->clearAll();
}

// Handle open request
void CWindow::Open(void)
{
    Open(QString());
}

// Handle open request
void CWindow::Open(const QString& path)
{
    CloseVolume();
    OpenVolume(path);
    UpdateView();  // update the panel when volume package is loaded
}

void CWindow::OpenRecent()
{
    auto action = qobject_cast<QAction*>(sender());
    if (action)
        Open(action->data().toString());
}

void CWindow::LoadSurfaces(bool reload)
{
    if (!fVpkg) return;

    if (reload) {
        // Clear OpChains
        for (auto& pair : _opchains) {
            delete pair.second;
        }
        _opchains.clear();

        // Force reload of surfaces
        fVpkg->unloadAllSurfaces();
    }

    // Get all segmentation IDs for current directory
    auto segIds = fVpkg->segmentationIDs();

    // Batch load surfaces
    fVpkg->loadSurfacesBatch(segIds);

    // Update surface collection
    for (const auto& id : segIds) {
        auto surfMeta = fVpkg->getSurface(id);
        if (surfMeta) {
            _surf_col->setSurface(id, surfMeta->surface(), true);
        }
    }

    FillSurfaceTree();
    onSegFilterChanged(0);
}

// Pop up about dialog
void CWindow::Keybindings(void)
{
    QMessageBox::information(
        this, tr("Keybindings for Volume Cartographer"),
        tr("Keyboard: \n"
        "------------------- \n"
        "FIXME FIXME FIXME \n"
        "------------------- \n"
        "Ctrl+O: Open Volume Package \n"
        "Ctrl+S: Save Volume Package \n"
        "A,D: Impact Range down/up \n"
        "[, ]: Alternative Impact Range down/up \n"
        "Q,E: Slice scan range down/up (mouse wheel scanning) \n"
        "Arrow Left/Right: Slice down/up by 1 \n"
        "1,2: Slice down/up by 1 \n"
        "3,4: Slice down/up by 5 \n"
        "5,6: Slice down/up by 10 \n"
        "7,8: Slice down/up by 50 \n"
        "9,0: Slice down/up by 100 \n"
        "Ctrl+G: Go to slice (opens dialog to insert slice index) \n"
        "Ctrl+T: Toggle direction hints (flip_x arrows) \n"
        "T: Segmentation Tool \n"
        "P: Pen Tool \n"
        "Space: Toggle Curve Visibility \n"
        "C: Alternate Toggle Curve Visibility \n"
        "J: Highlight Next Curve that is selected for computation \n"
        "K: Highlight Previous Curve that is selected for computation \n"
        "F: Return to slice that the currently active tool was started on \n"
        "L: Mark/unmark current slice as anchor (only in Segmentation Tool) \n"
        "Y/Z/V: Evenly space Points on Curve (only in Segmentation Tool) \n"
        "U: Rotate view counterclockwise \n"
        "O: Rotate view clockwise \n"
        "X/I: Reset view rotation back to zero \n"
        "\n"
        "Mouse: \n"
        "------------------- \n"
        "Mouse Wheel: Scroll up/down \n"
        "Mouse Wheel + Alt: Scroll left/right \n"
        "Mouse Wheel + Ctrl: Zoom in/out \n"
        "Mouse Wheel + Shift: Next/previous slice \n"
        "Mouse Wheel + W Key Hold: Change impact range \n"
        "Mouse Wheel + R Key Hold: Follow Highlighted Curve \n"
        "Mouse Wheel + S Key Hold: Rotate view \n"
        "Mouse Left Click: Add Points to Curve in Pen Tool. Snap Closest Point to Cursor in Segmentation Tool. \n"
        "Mouse Left Drag: Drag Point / Curve after Mouse Left Click \n"
        "Mouse Right Drag: Pan slice image\n"
        "Mouse Back/Forward Button: Follow Highlighted Curve \n"
        "Highlighting Segment ID: Shift/(Alt as well as Ctrl) Modifier to jump to Segment start/end."));
}

// Pop up about dialog
void CWindow::About(void)
{
    QMessageBox::information(
        this, tr("About Volume Cartographer"),
        tr("Vis Center, University of Kentucky\n\n"
        "Fork: https://github.com/spacegaier/volume-cartographer"));
}

void CWindow::ShowSettings()
{
    auto pDlg = new SettingsDialog(this);
    
    pDlg->exec();
    // Apply updated settings immediately to viewers
    {
        QSettings settings("VC.ini", QSettings::IniFormat);
        bool showDirHints = settings.value("viewer/show_direction_hints", true).toBool();
        for (auto &viewer : _viewers) {
            viewer->setShowDirectionHints(showDirHints);
        }
    }
    delete pDlg;
}

void CWindow::ResetSegmentationViews()
{
    for(auto sub : mdiArea->subWindowList()) {
        sub->showNormal();
    }
    mdiArea->tileSubWindows();
}

auto CWindow::can_change_volume_() -> bool
{
    bool canChange = fVpkg != nullptr && fVpkg->numberOfVolumes() > 1;
    return canChange;
}

// Handle request to step impact range down
void CWindow::onLocChanged(void)
{
    // std::cout << "loc changed!" << "\n";
    
    // sendLocChanged(spinLoc[0]->value(),spinLoc[1]->value(),spinLoc[2]->value());
}

void CWindow::onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    if (modifiers & Qt::ShiftModifier) {
        return;
    }
    else if (modifiers & Qt::ControlModifier) {
        std::cout << "clicked on vol loc " << vol_loc << std::endl;
        //NOTE this comes before the focus poi, so focus is applied by views using these slices
        //FIXME this assumes a single segmentation ... make configurable and cleaner ...
        QuadSurface *segment = dynamic_cast<QuadSurface*>(surf);
        if (segment) {
            PlaneSurface *segXZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg xz"));
            PlaneSurface *segYZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg yz"));
            
            if (!segXZ)
                segXZ = new PlaneSurface();
            if (!segYZ)
                segYZ = new PlaneSurface();

            //FIXME actually properly use ptr
            auto ptr = segment->pointer();
            segment->pointTo(ptr, vol_loc, 1.0);
            
            cv::Vec3f p2;
            p2 = segment->coord(ptr, {1,0,0});
            
            segXZ->setOrigin(vol_loc);
            segXZ->setNormal(p2-vol_loc);
            
            p2 = segment->coord(ptr, {0,1,0});
            
            segYZ->setOrigin(vol_loc);
            segYZ->setNormal(p2-vol_loc);
            
            _surf_col->setSurface("seg xz", segXZ);
            _surf_col->setSurface("seg yz", segYZ);
        }
        
        POI *poi = _surf_col->poi("focus");
        
        if (!poi)
            poi = new POI;

        poi->src = surf;
        poi->p = vol_loc;
        poi->n = normal;
        
        _surf_col->setPOI("focus", poi);

    }
    else {
    }
}

void CWindow::onManualPlaneChanged(void)
{    
    cv::Vec3f normal;
    
    for(int i=0;i<3;i++) {
        normal[i] = spNorm[i]->value();
    }
 
    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf_col->surface("manual plane"));
 
    if (!plane)
        return;
 
    plane->setNormal(normal);
    _surf_col->setSurface("manual plane", plane);
}

void CWindow::onOpChainChanged(OpChain *chain)
{
    _surf_col->setSurface("segmentation", chain);
}

static void sync_tag(nlohmann::json &dict, bool checked, std::string name, const std::string& username = "")
{
    if (checked && !dict.count(name)) {
        dict[name] = nlohmann::json::object();
        if (!username.empty()) {
            dict[name]["user"] = username;
        }
        dict[name]["date"] = QDateTime::currentDateTime().toString(Qt::ISODate).toStdString();
        if (name == "approved")
            dict["date_last_modified"] = get_surface_time_str();
    }
    if (!checked && dict.count(name)) {
        dict.erase(name);
        if (name == "approved")
            dict["date_last_modified"] = get_surface_time_str();
    }
}

void CWindow::onTagChanged(void)
{
    // Get username from settings
    QSettings settings("VC.ini", QSettings::IniFormat);
    std::string username = settings.value("viewer/username", "").toString().toStdString();
    
    // Get all selected items
    QList<QTreeWidgetItem*> selectedItems = treeWidgetSurfaces->selectedItems();
    
    // Apply tags to all selected surfaces
    for (QTreeWidgetItem* item : selectedItems) {
        std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        
        // Get the surface for this item
        QuadSurface* surf = nullptr;
        if (_opchains.count(id) && _opchains[id]) {
            surf = _opchains[id]->src();
        } else if (fVpkg) {
            auto surfMeta = fVpkg->getSurface(id);
            if (surfMeta) {
                surf = surfMeta->surface();
            }
        }

        if (!surf || !surf->meta) {
            continue;
        }

        // Track if reviewed status changed from unchecked to checked
        bool wasReviewed = surf->meta->contains("tags") &&
                          surf->meta->at("tags").contains("reviewed");
        bool isNowReviewed = _chkReviewed->checkState() == Qt::Checked;
        bool reviewedJustAdded = !wasReviewed && isNowReviewed;

        if (surf->meta->contains("tags")) {
            sync_tag(surf->meta->at("tags"), _chkApproved->checkState() == Qt::Checked, "approved", username);
            sync_tag(surf->meta->at("tags"), _chkDefective->checkState() == Qt::Checked, "defective", username);
            sync_tag(surf->meta->at("tags"), _chkReviewed->checkState() == Qt::Checked, "reviewed", username);
            sync_tag(surf->meta->at("tags"), _chkRevisit->checkState() == Qt::Checked, "revisit", username);
            sync_tag(surf->meta->at("tags"), _chkInspect->checkState() == Qt::Checked, "inspect", username);
            surf->save_meta();
        }
        else if (_chkApproved->checkState() || _chkDefective->checkState() || _chkReviewed->checkState() || _chkRevisit->checkState() || _chkInspect->checkState()) {
            surf->meta->push_back({"tags", nlohmann::json::object()});
            if (_chkApproved->checkState()) {
                if (!username.empty()) {
                    surf->meta->at("tags")["approved"] = nlohmann::json::object();
                    surf->meta->at("tags")["approved"]["user"] = username;
                } else {
                    surf->meta->at("tags")["approved"] = nullptr;
                }
            }
            if (_chkDefective->checkState()) {
                if (!username.empty()) {
                    surf->meta->at("tags")["defective"] = nlohmann::json::object();
                    surf->meta->at("tags")["defective"]["user"] = username;
                } else {
                    surf->meta->at("tags")["defective"] = nullptr;
                }
            }
            if (_chkReviewed->checkState()) {
                if (!username.empty()) {
                    surf->meta->at("tags")["reviewed"] = nlohmann::json::object();
                    surf->meta->at("tags")["reviewed"]["user"] = username;
                } else {
                    surf->meta->at("tags")["reviewed"] = nullptr;
                }
            }
            if (_chkRevisit->checkState()) {
                if (!username.empty()) {
                    surf->meta->at("tags")["revisit"] = nlohmann::json::object();
                    surf->meta->at("tags")["revisit"]["user"] = username;
                } else {
                    surf->meta->at("tags")["revisit"] = nullptr;
                }
            }
            if (_chkInspect->checkState()) {
                if (!username.empty()) {
                    surf->meta->at("tags")["inspect"] = nlohmann::json::object();
                    surf->meta->at("tags")["inspect"]["user"] = username;
                } else {
                    surf->meta->at("tags")["inspect"] = nullptr;
                }
            }
            surf->save_meta();
        }

        // If reviewed was just added, mark overlapping segmentations with partial_review
        if (reviewedJustAdded && fVpkg) {
            auto surfMeta = fVpkg->getSurface(id);
            if (surfMeta) {
                std::cout << "Marking partial review for overlaps of " << id << ", found " << surfMeta->overlapping_str.size() << " overlaps" << std::endl;

                // Iterate through overlapping surfaces
                for (const std::string& overlapId : surfMeta->overlapping_str) {
                    auto overlapMeta = fVpkg->getSurface(overlapId);
                    if (overlapMeta) {
                        QuadSurface* overlapSurf = overlapMeta->surface();

                        if (overlapSurf && overlapSurf->meta) {
                            // Don't mark as partial_review if it's already reviewed
                            bool alreadyReviewed = overlapSurf->meta->contains("tags") &&
                                                 overlapSurf->meta->at("tags").contains("reviewed");

                            if (!alreadyReviewed) {
                                // Ensure tags object exists
                                if (!overlapSurf->meta->contains("tags")) {
                                    (*overlapSurf->meta)["tags"] = nlohmann::json::object();
                                }

                                // Add partial_review tag
                                if (!username.empty()) {
                                    (*overlapSurf->meta)["tags"]["partial_review"] = nlohmann::json::object();
                                    (*overlapSurf->meta)["tags"]["partial_review"]["user"] = username;
                                    (*overlapSurf->meta)["tags"]["partial_review"]["source"] = id;
                                } else {
                                    (*overlapSurf->meta)["tags"]["partial_review"] = nlohmann::json::object();
                                    (*overlapSurf->meta)["tags"]["partial_review"]["source"] = id;
                                }

                                // Save the metadata
                                overlapSurf->save_meta();

                                std::cout << "Added partial_review tag to " << overlapId << std::endl;
                            }
                        }
                    }
                }
            }
        }
        
        // Update the tree icon for this item
        UpdateSurfaceTreeIcon(static_cast<SurfaceTreeWidgetItem*>(item));
    }
    
    // Update filters to reflect the tag changes
    onSegFilterChanged(0);
}

void CWindow::onSurfaceSelected()
{
    // Get the first selected item for single-segment operations
    QList<QTreeWidgetItem*> selectedItems = treeWidgetSurfaces->selectedItems();
    if (selectedItems.isEmpty()) {
        // Reset sub window title
        for (auto &viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setWindowTitle(tr("Surface"));
                break;
            }
        }

        return;
    }
    
    // Use the first selected item for all existing functionality
    QTreeWidgetItem* firstSelected = selectedItems.first();
    _surfID = firstSelected->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();

    // Update sub window title with surface ID
    for (auto &viewer : _viewers) {
        if (viewer->surfName() == "segmentation") {
            viewer->setWindowTitle(tr("Surface %1").arg(QString::fromStdString(_surfID)));
            break;
        }
    }

    if (!_opchains.count(_surfID)) {
        auto surfMeta = fVpkg->getSurface(_surfID);
        if (surfMeta) {
            _opchains[_surfID] = new OpChain(surfMeta->surface());
        }
    }

    if (_opchains[_surfID]) {
        _surf_col->setSurface("segmentation", _opchains[_surfID]->src());
        sendOpChainSelected(_opchains[_surfID]);
        _surf = _opchains[_surfID]->src();
        {
            const QSignalBlocker b1{_chkApproved};
            const QSignalBlocker b2{_chkDefective};
            const QSignalBlocker b3{_chkReviewed};
            const QSignalBlocker b4{_chkRevisit};
            const QSignalBlocker b5{_chkInspect};
            
            std::cout << "surf " << _surf->path << _surfID <<  _surf->meta << std::endl;
            
            _chkApproved->setEnabled(true);
            _chkDefective->setEnabled(true);
            _chkReviewed->setEnabled(true);
            _chkRevisit->setEnabled(true);
            _chkInspect->setEnabled(true);
            
            _chkApproved->setCheckState(Qt::Unchecked);
            _chkDefective->setCheckState(Qt::Unchecked);
            _chkReviewed->setCheckState(Qt::Unchecked);
            _chkRevisit->setCheckState(Qt::Unchecked);
            _chkInspect->setCheckState(Qt::Unchecked);
            if (_surf->meta) {
                if (_surf->meta->value("tags", nlohmann::json::object_t()).count("approved"))
                    _chkApproved->setCheckState(Qt::Checked);
                if (_surf->meta->value("tags", nlohmann::json::object_t()).count("defective"))
                    _chkDefective->setCheckState(Qt::Checked);
                if (_surf->meta->value("tags", nlohmann::json::object_t()).count("reviewed"))
                    _chkReviewed->setCheckState(Qt::Checked);
                if (_surf->meta->value("tags", nlohmann::json::object_t()).count("revisit"))
                    _chkRevisit->setCheckState(Qt::Checked);
                if (_surf->meta->value("tags", nlohmann::json::object_t()).count("inspect"))
                    _chkInspect->setCheckState(Qt::Checked);
            }
            else {
                _chkApproved->setEnabled(false);
                _chkDefective->setEnabled(false);
                _chkReviewed->setEnabled(true);
                _chkRevisit->setEnabled(true);
                _chkInspect->setEnabled(true);
            }
        }
    }
    else
        std::cout << "ERROR loading " << _surfID << std::endl;

    // If "Current Segment Only" is checked, refresh the filter to update intersections
    if (chkFilterCurrentOnly && chkFilterCurrentOnly->isChecked()) {
        onSegFilterChanged(0);
    }
}

void CWindow::FillSurfaceTree()
{
    const QSignalBlocker blocker{treeWidgetSurfaces};
    treeWidgetSurfaces->clear();

    // VolumePkg now only returns surfaces from the current directory
    for (const auto& id : fVpkg->segmentationIDs()) {
        auto surfMeta = fVpkg->getSurface(id);
        if (!surfMeta) continue;

        auto* item = new SurfaceTreeWidgetItem(treeWidgetSurfaces);
        item->setText(SURFACE_ID_COLUMN, QString(id.c_str()));
        item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QVariant(id.c_str()));

        double size = surfMeta->meta->value("area_cm2", -1.f);
        item->setText(2, QString::number(size, 'f', 3));
        double cost = surfMeta->meta->value("avg_cost", -1.f);
        item->setText(3, QString::number(cost, 'f', 3));
        item->setText(4, QString::number(surfMeta->overlapping_str.size()));
        QString timestamp;
        if (surfMeta->meta && surfMeta->meta->contains("date_last_modified")) {
            timestamp = QString::fromStdString((*surfMeta->meta)["date_last_modified"].get<std::string>());
        }
        item->setText(5, timestamp);
        UpdateSurfaceTreeIcon(item);
    }

    treeWidgetSurfaces->resizeColumnToContents(0);
    treeWidgetSurfaces->resizeColumnToContents(1);
    treeWidgetSurfaces->resizeColumnToContents(2);
    treeWidgetSurfaces->resizeColumnToContents(3);

    if (!appInitComplete) {
        // Apply initial sorting during apps tartup, but afterwards keep
        // whatever the user chose
        treeWidgetSurfaces->sortByColumn(SURFACE_ID_COLUMN, Qt::AscendingOrder);
    }
}

void CWindow::UpdateSurfaceTreeIcon(SurfaceTreeWidgetItem *item)
{
    std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();

    if (fVpkg) {
        auto surfMeta = fVpkg->getSurface(id);
        if (surfMeta && surfMeta->surface() && surfMeta->surface()->meta) {
            item->updateItemIcon(
                surfMeta->surface()->meta->value("tags", nlohmann::json::object_t()).count("approved"),
                surfMeta->surface()->meta->value("tags", nlohmann::json::object_t()).count("defective"));
        }
    }
}

void CWindow::onSegFilterChanged(int index)
{
    if (!fVpkg) {
        return;
    }

    // Check if ANY filters are actually active
    bool hasActiveFilters = chkFilterFocusPoints->isChecked() ||
                           chkFilterUnreviewed->isChecked() ||
                           chkFilterRevisit->isChecked() ||
                           chkFilterNoExpansion->isChecked() ||
                           chkFilterNoDefective->isChecked() ||
                           chkFilterPartialReview->isChecked() ||
                           chkFilterCurrentOnly->isChecked() ||
                               chkFilterHideUnapproved->isChecked() ||
                               chkFilterInspectOnly->isChecked();

    // Check if point set filter has any checked items
    if (!hasActiveFilters && cmbPointSetFilter->count() > 0) {
        for (int i = 0; i < cmbPointSetFilter->count(); ++i) {
            if (cmbPointSetFilter->itemData(i, Qt::CheckStateRole) == Qt::Checked) {
                hasActiveFilters = true;
                break;
            }
        }
    }

    // If no filters are active, show everything and exit early
    if (!hasActiveFilters) {
        // Show all items
        QTreeWidgetItemIterator it(treeWidgetSurfaces);
        while (*it) {
            (*it)->setHidden(false);
            ++it;
        }

        // Add all surfaces to intersection set
        std::set<std::string> all_intersects = {"segmentation"};
        for (const auto& id : fVpkg->getLoadedSurfaceIDs()) {
            all_intersects.insert(id);
        }

        // Apply to viewers
        for (auto &viewer : _viewers) {
            if (viewer->surfName() != "segmentation") {
                viewer->setIntersects(all_intersects);
            }
        }

        UpdateVolpkgLabel(0);
        return;
    }

    std::set<std::string> dbg_intersects = {"segmentation"};
    POI *poi = _surf_col->poi("focus");
    int filterCounter = 0;

    // Check if "Current Segment Only" is checked
    bool currentOnly = chkFilterCurrentOnly->isChecked();

    if (currentOnly) {
        // Only add the currently selected segment if it exists
        if (!_surfID.empty() && fVpkg->getSurface(_surfID)) {
            dbg_intersects.insert(_surfID);
        }
    }

    QTreeWidgetItemIterator it(treeWidgetSurfaces);
    while (*it) {
        std::string id = (*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();

        bool show = true;
        auto surfMeta = fVpkg->getSurface(id);
        if (!surfMeta) {
            show = true;
        } else {
            // Start with show = true and apply filters as AND conditions
            show = true;

            // Filter by focus points
            if (chkFilterFocusPoints->isChecked() && poi) {
                show = show && contains(*surfMeta, poi->p);
            }

            // Filter by point sets
            bool any_checked = false;
            for (int i = 0; i < cmbPointSetFilter->count(); ++i) {
                if (cmbPointSetFilter->itemData(i, Qt::CheckStateRole) == Qt::Checked) {
                    any_checked = true;
                    break;
                }
            }

            if (any_checked) {
                bool match = false;
                bool all_match = true;
                for (int i = 0; i < cmbPointSetFilter->count(); ++i) {
                    if (cmbPointSetFilter->itemData(i, Qt::CheckStateRole) == Qt::Checked) {
                        std::vector<cv::Vec3f> points;
                        auto collection = _point_collection->getPoints(cmbPointSetFilter->itemText(i).toStdString());
                        points.reserve(collection.size());
                        for (const auto& p : collection) {
                            points.push_back(p.p);
                        }
                        if (all_match && !contains(*surfMeta, points))
                            all_match = false;
                        if (!match && contains_any(*surfMeta, points))
                            match = true;
                    }
                }
                if (cmbPointSetFilterMode->currentIndex() == 0) { // Any (OR)
                    show = show && match;
                } else { // All (AND)
                    show = show && all_match;
                }
            }

            // Filter by unreviewed
            if (chkFilterUnreviewed->isChecked()) {
                auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    auto tags = surface->meta->value("tags", nlohmann::json::object_t());
                    show = show && !tags.count("reviewed");
                } else {
                    show = show && true;
                }
            }

            // Filter by revisit
            if (chkFilterRevisit->isChecked()) {
                auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    auto tags = surface->meta->value("tags", nlohmann::json::object_t());
                    show = show && (tags.count("revisit") > 0);
                } else {
                    show = show && false;
                }
            }

            // Filter out expansion
            if (chkFilterNoExpansion->isChecked()) {
                auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    if (surface->meta->contains("vc_gsfs_mode")) {
                        std::string mode = surface->meta->value("vc_gsfs_mode", "");
                        show = show && (mode != "expansion");
                    } else {
                        show = show && true;
                    }
                } else {
                    show = show && true;
                }
            }

            // Filter out defective
            if (chkFilterNoDefective->isChecked()) {
                auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    auto tags = surface->meta->value("tags", nlohmann::json::object_t());
                    show = show && !tags.count("defective");
                } else {
                    show = show && true;
                }
            }

            // Filter out partial review
            if (chkFilterPartialReview->isChecked()) {
                auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    auto tags = surface->meta->value("tags", nlohmann::json::object_t());
                    show = show && !tags.count("partial_review");
                } else {
                    show = show && true;
                }
            }

            if (chkFilterHideUnapproved->isChecked()) {
                auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    auto tags = surface->meta->value("tags", nlohmann::json::object_t());
                    show = show && (tags.count("approved") > 0);
                } else {
                    show = show && false;  // Hide segments without metadata when filter is active
                }
            }
            if (chkFilterInspectOnly->isChecked()) {
                auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    auto tags = surface->meta->value("tags", nlohmann::json::object_t());
                    show = show && (tags.count("inspect") > 0);
                } else {
                    show = show && false;  // Hide segments without metadata when filter is active
                }
            }
        }

        (*it)->setHidden(!show);

        if (show && !currentOnly) {
            if (surfMeta)
                dbg_intersects.insert(id);
        } else if (!show) {
            filterCounter++;
        }

        ++it;
    }

    UpdateVolpkgLabel(filterCounter);

    // Apply the intersection set to all non-segmentation viewers
    for (auto &viewer : _viewers) {
        if (viewer->surfName() != "segmentation") {
            viewer->setIntersects(dbg_intersects);
        }
    }
}

void CWindow::onEditMaskPressed(void)
{
    if (!_surf)
        return;
    std::cout << "oneditmaskpressed" << std::endl;
    std::filesystem::path path = _surf->path/"mask.tif";

    if (!std::filesystem::exists(path)) {
        cv::Mat_<uint8_t> img;
        cv::Mat_<uint8_t> mask;

        // Use generate_mask function instead of duplicating logic
        z5::Dataset* ds_high = currentVolume ? currentVolume->zarrDataset(0) : nullptr;
        z5::Dataset* ds_low = (currentVolume && currentVolume->numScales() >= 3) ?
                              currentVolume->zarrDataset(2) : nullptr;

        generate_mask(_surf, mask, img, ds_high, ds_low, chunk_cache);

        // Save the generated mask and image
        std::vector<cv::Mat> layers = {mask, img};
        imwritemulti(path, layers);
    }
    
    QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));
}

void CWindow::onRefreshSurfaces()
{
    LoadSurfacesIncremental();
}

QString CWindow::getCurrentVolumePath() const
{
    if (currentVolume == nullptr) {
        return QString();
    }
    return QString::fromStdString(currentVolume->path().string());
}

void CWindow::onToggleConsoleOutput()
{
    if (_cmdRunner) {
        _cmdRunner->showConsoleOutput();
    } else {
        QMessageBox::information(this, tr("Console Output"), 
                                tr("No command line tool has been run yet. The console will be available after running a tool."));
    }
}
void CWindow::onSurfaceContextMenuRequested(const QPoint& pos)
{
    QTreeWidgetItem* item = treeWidgetSurfaces->itemAt(pos);
    if (!item) {
        return;
    }

    // Get all selected segments
    QList<QTreeWidgetItem*> selectedItems = treeWidgetSurfaces->selectedItems();
    std::vector<std::string> selectedSegmentIds;
    for (auto* selectedItem : selectedItems) {
        selectedSegmentIds.push_back(selectedItem->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString());
    }

    // Use the first selected segment for single-segment operations
    std::string segmentId = selectedSegmentIds.empty() ?
        item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() :
        selectedSegmentIds.front();

    QMenu contextMenu(tr("Context Menu"), this);

    // Copy segment path action
    QAction* copyPathAction = new QAction(tr("Copy Segment Path"), this);
    connect(copyPathAction, &QAction::triggered, [this, segmentId]() {
        if (fVpkg) {
            auto surfMeta = fVpkg->getSurface(segmentId);
            if (surfMeta) {
                QString path = QString::fromStdString(surfMeta->path.string());
                QApplication::clipboard()->setText(path);
                statusBar()->showMessage(tr("Copied segment path to clipboard: %1").arg(path), 3000);
            }
        }
    });
    
    // Delete segment(s) action
    QString deleteText = selectedSegmentIds.size() > 1 ? 
        tr("Delete %1 Segments").arg(selectedSegmentIds.size()) : 
        tr("Delete Segment");
    QAction* deleteAction = new QAction(deleteText, this);
    deleteAction->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));
    connect(deleteAction, &QAction::triggered, [this, selectedSegmentIds]() {
        onDeleteSegments(selectedSegmentIds);
    });
    
    // Render segment action
    QAction* renderAction = new QAction(tr("Render segment"), this);
    connect(renderAction, &QAction::triggered, [this, segmentId]() {
        onRenderSegment(segmentId);
    });

    // SLIM-flatten and render (calls slot implemented in CWindowContextMenu.cpp)
    QAction* slimFlattenAction = new QAction(tr("SLIM-flatten and render"), this);
    connect(slimFlattenAction, &QAction::triggered, [this, segmentId]() {
        onSlimFlattenAndRender(segmentId);
    });

    // Grow segment from segment action
    QAction* growSegmentAction = new QAction(tr("Run Trace"), this);
    connect(growSegmentAction, &QAction::triggered, [this, segmentId]() {
        onGrowSegmentFromSegment(segmentId);
    });
    
    // Add overlap action
    QAction* addOverlapAction = new QAction(tr("Add overlap"), this);
    connect(addOverlapAction, &QAction::triggered, [this, segmentId]() {
        onAddOverlap(segmentId);
    });
    
    // Convert to OBJ action
    QAction* convertToObjAction = new QAction(tr("Convert to OBJ"), this);
    connect(convertToObjAction, &QAction::triggered, [this, segmentId]() {
        onConvertToObj(segmentId);
    });
    
    // Seed submenu with options
    QMenu* seedMenu = new QMenu(tr("Run Seed"), &contextMenu);
    QAction* seedWithSeedAction = new QAction(tr("Seed from Focus Point"), seedMenu);
    connect(seedWithSeedAction, &QAction::triggered, [this, segmentId]() {
        onGrowSeeds(segmentId, false, false);
    });
    QAction* seedWithRandomAction = new QAction(tr("Random Seed"), seedMenu);
    connect(seedWithRandomAction, &QAction::triggered, [this, segmentId]() {
        onGrowSeeds(segmentId, false, true);
    });
    QAction* seedWithExpandAction = new QAction(tr("Expand Seed"), seedMenu);
    connect(seedWithExpandAction, &QAction::triggered, [this, segmentId]() {
        onGrowSeeds(segmentId, true, false);
    });
    seedMenu->addAction(seedWithSeedAction);
    seedMenu->addAction(seedWithRandomAction);
    seedMenu->addAction(seedWithExpandAction);
    
    // Build menu
    contextMenu.addAction(copyPathAction);
    contextMenu.addSeparator();
    contextMenu.addMenu(seedMenu);
    contextMenu.addAction(growSegmentAction);
    contextMenu.addAction(addOverlapAction);
    contextMenu.addSeparator();
    contextMenu.addAction(renderAction);
    contextMenu.addAction(convertToObjAction);
    contextMenu.addAction(slimFlattenAction);
    contextMenu.addSeparator();
    // Telea pipeline (RGB -> inpaint -> back to tifxyz)
    QAction* inpaintTeleaAction = new QAction(tr("Inpaint (Telea) && Rebuild Segment"), this);
    connect(inpaintTeleaAction, &QAction::triggered, [this]() {
        onInpaintTeleaSelected();
    });
    contextMenu.addAction(inpaintTeleaAction);
    contextMenu.addAction(deleteAction);
    
    contextMenu.exec(treeWidgetSurfaces->mapToGlobal(pos));
}

void CWindow::onSegmentationDirChanged(int index)
{
    if (!fVpkg || index < 0 || !cmbSegmentationDir) {
        return;
    }
    
    std::string newDir = cmbSegmentationDir->itemText(index).toStdString();
    
    // Only reload if the directory actually changed
    if (newDir != fVpkg->getSegmentationDirectory()) {
        // Clear the current segmentation surface first to ensure viewers update
        _surf_col->setSurface("segmentation", nullptr, true);
        
        // Clear current surface selection
        _surf = nullptr;
        _surfID.clear();
        treeWidgetSurfaces->clearSelection();
        wOpsList->onOpChainSelected(nullptr);
        
        // Clear checkboxes
        {
            const QSignalBlocker b1{_chkApproved};
            const QSignalBlocker b2{_chkDefective};
            _chkApproved->setCheckState(Qt::Unchecked);
            _chkDefective->setCheckState(Qt::Unchecked);
            _chkApproved->setEnabled(false);
            _chkDefective->setEnabled(false);
            _chkReviewed->setEnabled(false);
            _chkRevisit->setEnabled(false);
            _chkInspect->setEnabled(false);
        }
        
        // Set the new directory in the VolumePkg
        fVpkg->setSegmentationDirectory(newDir);
        
        // Reload surfaces from the new directory
        LoadSurfaces(false);
        
        // Update the status bar to show the change
        statusBar()->showMessage(tr("Switched to %1 directory").arg(QString::fromStdString(newDir)), 3000);
    }
}

// ===== Telea inpaint pipeline implementation =====
namespace {

// Run a CLI tool and collect its merged stdout/stderr; show error box on failure
static bool run_cli(QWidget* parent, const QString& program, const QStringList& args, QString* outLog = nullptr) {
    QProcess p;
    p.setProcessChannelMode(QProcess::MergedChannels);
    p.start(program, args);
    if (!p.waitForStarted()) {
        QMessageBox::critical(parent, QObject::tr("Error"),
                              QObject::tr("Failed to start %1").arg(program));
        return false;
    }
    p.waitForFinished(-1);
    const QString log = p.readAll();
    if (outLog) *outLog = log;
    if (p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0) {
        QMessageBox::critical(parent, QObject::tr("Command Failed"),
                              QObject::tr("%1 exited with code %2.\n\n%3")
                              .arg(program).arg(p.exitCode()).arg(log));
        return false;
    }
    return true;
}

// Resolve a tool located next to the application; otherwise fall back to PATH
static QString find_tool(const char* baseName) {
#ifdef _WIN32
    const QString exe = QString::fromLatin1(baseName) + ".exe";
#else
    const QString exe = QString::fromLatin1(baseName);
#endif
    const QString appDir = QCoreApplication::applicationDirPath();
    const QString local  = appDir + QDir::separator() + exe;
    if (QFileInfo::exists(local)) return local;
    return exe; // rely on PATH
}
} // namespace

void CWindow::onInpaintTeleaSelected()
{
    if (!fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("No volume package loaded."));
        return;
    }

    // Use all selected segments (patches/traces)
    QList<QTreeWidgetItem*> selectedItems = treeWidgetSurfaces->selectedItems();
    if (selectedItems.isEmpty()) {
        QMessageBox::information(this, tr("Info"), tr("Select a patch/trace first in the Surfaces list."));
        return;
    }

    // Locate tools (next to app or PATH)
    const QString vc_tifxyz2rgb    = find_tool("vc_tifxyz2rgb");
    const QString vc_telea_inpaint = find_tool("vc_telea_inpaint");
    const QString vc_rgb2tifxyz    = find_tool("vc_rgb2tifxyz");

    int successCount = 0, failCount = 0;

    for (QTreeWidgetItem* item : selectedItems) {
        const std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        auto surfMeta = fVpkg->getSurface(id);
        if (!surfMeta) { ++failCount; continue; }

        const std::filesystem::path segDir    = surfMeta->path;              // .../paths/<id> or .../traces/<id>
        const std::filesystem::path parentDir = segDir.parent_path();        // .../paths or .../traces
        const std::filesystem::path metaJson  = segDir / "meta.json";

        if (!std::filesystem::exists(metaJson)) {
            QMessageBox::warning(this, tr("Error"),
                                 tr("Missing meta.json for %1").arg(QString::fromStdString(id)));
            ++failCount; continue;
        }

        // Time-stamped names and temp dirs
        const QString stamp  = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmsszzz");
        const QString rgbPngName = QString::fromStdString(id) + "_xyz_rgb_" + stamp + ".png";
        const QString newSegName = QString::fromStdString(id) + "_telea_" + stamp;

        QTemporaryDir tmpInDir;   // for RGB input to inpaint
        QTemporaryDir tmpOutDir;  // for inpainted output
        if (!tmpInDir.isValid() || !tmpOutDir.isValid()) {
            QMessageBox::warning(this, tr("Error"), tr("Failed to create temporary directories."));
            ++failCount; continue;
        }

        // 1) tifxyz -> RGB (explicit path in temp)
        const QString rgbPng = QDir(tmpInDir.path()).filePath(rgbPngName);
        {
            QStringList args;
            // Pass the segment directory (contains x.tif, y.tif, z.tif, meta.json)
            args << QString::fromStdString(segDir.string())
                 << rgbPng;
            QString log;
            if (!run_cli(this, vc_tifxyz2rgb, args, &log)) { ++failCount; continue; }
        }

        // 2) Telea inpaint (non-recursive, single file)
        QString inpaintedPng;
        {
            QStringList args;
            args << tmpInDir.path()     // input dir
                 << tmpOutDir.path()    // output dir
                 << QString::number(3)  // radius
                 << QString::number(8)  // black_threshold
                 << QString::number(1)  // min_area
                 << QString::number(0); // recursive
            QString log;
            if (!run_cli(this, vc_telea_inpaint, args, &log)) { ++failCount; continue; }

            // Expect exactly one PNG out
            QDir d(tmpOutDir.path());
            const QStringList outs = d.entryList(QStringList() << "*.png", QDir::Files);
            if (outs.isEmpty()) {
                QMessageBox::warning(this, tr("Error"), tr("Telea inpaint produced no PNG."));
                ++failCount; continue;
            }
            inpaintedPng = d.absoluteFilePath(outs.first());
        }

        // 3) RGB -> tifxyz (new segment beside the original)
        {
            QStringList args;
            args << inpaintedPng
                 << QString::fromStdString(metaJson.string())   // bounds/scale source
                 << QString::fromStdString(parentDir.string())  // out_dir
                 << newSegName
                 << "--invalid-black";
            QString log;
            if (!run_cli(this, vc_rgb2tifxyz, args, &log)) { ++failCount; continue; }
        }

        ++successCount;
    }

    // Reload surfaces so new segments appear
    if (successCount > 0) {
        try {
            fVpkg->refreshSegmentations();
            LoadSurfacesIncremental();
        } catch (...) {
            // best effort
        }
    }

    statusBar()->showMessage(tr("Telea inpaint pipeline complete. Success: %1, Failed: %2")
                             .arg(successCount).arg(failCount), 6000);
}

CWindow::SurfaceChanges CWindow::DetectSurfaceChanges()
{
    SurfaceChanges changes;

    if (!fVpkg) {
        return changes;
    }

    // Get IDs from disk for current directory only
    std::set<std::string> diskIds;
    auto segIds = fVpkg->segmentationIDs();
    for (const auto& id : segIds) {
        diskIds.insert(id);
    }

    // Get currently loaded surface IDs
    std::set<std::string> loadedIds;
    auto loadedIdVec = fVpkg->getLoadedSurfaceIDs();
    for (const auto& id : loadedIdVec) {
        loadedIds.insert(id);
    }

    // Get the current directory
    std::string currentDir = fVpkg->getSegmentationDirectory();

    // Find additions (in disk but not loaded)
    for (const auto& id : diskIds) {
        if (loadedIds.find(id) == loadedIds.end()) {
            changes.toAdd.push_back(id);
        }
    }

    // Find removals - iterate through loaded surfaces to see if they still exist on disk
    for (const auto& loadedId : loadedIds) {
        // Check if this loaded surface is no longer on disk for current directory
        if (diskIds.find(loadedId) == diskIds.end()) {
            // This surface was loaded but is no longer on disk for the current directory
            // Check if it still exists in VolumePkg (might be in another directory)

            try {
                auto seg = fVpkg->segmentation(loadedId);
                // If we can still get it from VolumePkg, it exists in another directory
                // So we shouldn't remove it from our display - it's just not in current dir
            } catch (const std::out_of_range& e) {
                // Can't find it in VolumePkg anymore - it was truly deleted
                changes.toRemove.push_back(loadedId);
            }
        }
    }
    
    return changes;
}
void CWindow::AddSingleSegmentation(const std::string& segId)
{
    if (!fVpkg) {
        return;
    }

    std::cout << "Adding segmentation: " << segId << std::endl;

    try {
        // Load the surface through VolumePkg
        auto sm = fVpkg->loadSurface(segId);
        if (!sm) {
            std::cout << "Failed to load surface " << segId << " (wrong format or other issue)" << std::endl;
            return;
        }

        // Add to surface collection
        _surf_col->setSurface(segId, sm->surface(), true);

        // Add to tree widget
        auto* item = new SurfaceTreeWidgetItem(treeWidgetSurfaces);
        item->setText(SURFACE_ID_COLUMN, QString(segId.c_str()));
        item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QVariant(segId.c_str()));
        double size = sm->meta->value("area_cm2", -1.f);
        item->setText(2, QString::number(size, 'f', 3));
        double cost = sm->meta->value("avg_cost", -1.f);
        item->setText(3, QString::number(cost, 'f', 3));
        item->setText(4, QString::number(sm->overlapping_str.size()));
        QString timestamp;
        if (sm->meta && sm->meta->contains("date_last_modified")) {
            timestamp = QString::fromStdString((*sm->meta)["date_last_modified"].get<std::string>());
        }
        item->setText(5, timestamp);
        UpdateSurfaceTreeIcon(item);

    } catch (const std::exception& e) {
        std::cout << "Failed to add segmentation " << segId << ": " << e.what() << std::endl;
    }
}
void CWindow::RemoveSingleSegmentation(const std::string& segId)
{
    std::cout << "Removing segmentation: " << segId << std::endl;

    // Check if this is the currently selected segmentation
    bool wasSelected = (_surfID == segId);

    // Remove from surface collection - send signal to notify viewers
    _surf_col->setSurface(segId, nullptr, false);

    // If this was the selected segmentation, clear the segmentation surface
    if (wasSelected) {
        _surf_col->setSurface("segmentation", nullptr, false);  // Send signal to clear viewer pointers
        _surf = nullptr;
        _surfID.clear();

        // Clear checkboxes
        const QSignalBlocker b1{_chkApproved};
        const QSignalBlocker b2{_chkDefective};
        const QSignalBlocker b3{_chkReviewed};
        const QSignalBlocker b4{_chkRevisit};
        const QSignalBlocker b5{_chkInspect};
        _chkApproved->setCheckState(Qt::Unchecked);
        _chkDefective->setCheckState(Qt::Unchecked);
        _chkReviewed->setCheckState(Qt::Unchecked);
        _chkRevisit->setCheckState(Qt::Unchecked);
        _chkInspect->setCheckState(Qt::Unchecked);
        _chkApproved->setEnabled(false);
        _chkDefective->setEnabled(false);
        _chkReviewed->setEnabled(false);
        _chkRevisit->setEnabled(false);
        _chkInspect->setEnabled(false);

        // Reset window title
        for (auto &viewer : _viewers) {
            if (viewer->surfName() == "segmentation") {
                viewer->setWindowTitle(tr("Surface"));
                break;
            }
        }
    }

    // Remove from tree widget
    QTreeWidgetItemIterator it(treeWidgetSurfaces);
    while (*it) {
        if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == segId) {
            delete *it;
            break;
        }
        ++it;
    }

    // Unload surface from VolumePkg
    if (fVpkg) {
        fVpkg->unloadSurface(segId);
    }

    // Clean up OpChain if it exists (still owned by CWindow)
    if (_opchains.count(segId)) {
        delete _opchains[segId];
        _opchains.erase(segId);
    }
}

void CWindow::LoadSurfacesIncremental()
{
    std::cout << "Starting incremental surface load..." << std::endl;
    
    if (!fVpkg) {
        return;
    }
    
    // Refresh the VolumePkg's segmentation cache
    fVpkg->refreshSegmentations();
    
    // Detect changes
    auto changes = DetectSurfaceChanges();
    
    std::cout << "Found " << changes.toAdd.size() << " surfaces to add and " 
              << changes.toRemove.size() << " surfaces to remove" << std::endl;
    
    // Apply removals first
    for (const auto& id : changes.toRemove) {
        RemoveSingleSegmentation(id);
    }
    
    // Then apply additions
    for (const auto& id : changes.toAdd) {
        AddSingleSegmentation(id);
    }
    
    // Re-apply filter to update views
    onSegFilterChanged(0);
    
    // Update the volpkg label
    UpdateVolpkgLabel(0);
    
    // Emit signal to notify that surfaces have been updated
    emit sendSurfacesLoaded();
    
    std::cout << "Incremental surface load completed." << std::endl;
}
void CWindow::onGenerateReviewReport()
{
    if (!fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("No volume package loaded."));
        return;
    }

    // Let user choose save location
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("Save Review Report"),
        "review_report.csv",
        tr("CSV Files (*.csv)"));

    if (fileName.isEmpty()) {
        return;
    }

    // Data structure to aggregate by date and username
    struct UserStats {
        double totalArea = 0.0;
        int surfaceCount = 0;
    };
    std::map<QString, std::map<QString, UserStats>> dailyStats; // date -> username -> stats

    int totalReviewedCount = 0;
    double grandTotalArea = 0.0;

    // Iterate through all loaded surfaces
    for (const auto& id : fVpkg->getLoadedSurfaceIDs()) {
        auto surfMeta = fVpkg->getSurface(id);
        
        if (!surfMeta || !surfMeta->surface() || !surfMeta->surface()->meta) {
            continue;
        }
        
        nlohmann::json* meta = surfMeta->surface()->meta;
        
        // Check if surface has reviewed tag
        if (!meta->contains("tags") || !meta->at("tags").contains("reviewed")) {
            continue;
        }
        
        // Get review date
        QString reviewDate = "Unknown";
        if (meta->at("tags").at("reviewed").contains("date")) {
            QString fullDate = QString::fromStdString(meta->at("tags").at("reviewed").at("date").get<std::string>());
            // Extract just the date portion (YYYY-MM-DD) from ISO date string
            reviewDate = fullDate.left(10);
        } else {
            // Fallback to file modification time
            QFileInfo metaFile(QString::fromStdString(surfMeta->path.string()) + "/meta.json");
            if (metaFile.exists()) {
                reviewDate = metaFile.lastModified().toString("yyyy-MM-dd");
            }
        }
        
        // Get username
        QString username = "Unknown";
        if (meta->at("tags").at("reviewed").contains("user")) {
            username = QString::fromStdString(meta->at("tags").at("reviewed").at("user").get<std::string>());
        }
        
        // Get area
        double area = meta->value("area_cm2", 0.0);
        
        // Aggregate data
        dailyStats[reviewDate][username].totalArea += area;
        dailyStats[reviewDate][username].surfaceCount++;
        
        totalReviewedCount++;
        grandTotalArea += area;
    }
    
    // Open file for writing
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("Error"), tr("Could not open file for writing."));
        return;
    }
    
    QTextStream stream(&file);
    
    // Write header
    stream << "Date,Username,CM² Reviewed,Surface Count\n";
    
    // Write aggregated data sorted by date and username
    for (const auto& dateEntry : dailyStats) {
        const QString& date = dateEntry.first;
        for (const auto& userEntry : dateEntry.second) {
            const QString& username = userEntry.first;
            const UserStats& stats = userEntry.second;
            
            stream << date << ","
                   << username << ","
                   << QString::number(stats.totalArea, 'f', 3) << ","
                   << stats.surfaceCount << "\n";
        }
    }
    
    file.close();
    
    // Show summary message
    QString message = tr("Review report saved successfully.\n\n"
                        "Total reviewed surfaces: %1\n"
                        "Total area reviewed: %2 cm²\n"
                        "Days covered: %3")
                        .arg(totalReviewedCount)
                        .arg(grandTotalArea, 0, 'f', 3)
                        .arg(dailyStats.size());
    
    QMessageBox::information(this, tr("Report Generated"), message);
}
void CWindow::onVoxelizePaths()
{
    // Check if volume is loaded
    if (!fVpkg || !currentVolume) {
        QMessageBox::warning(this, tr("Error"),
                           tr("Please load a volume package first."));
        return;
    }

    // Get output path
    QString outputPath = QFileDialog::getSaveFileName(
        this,
        tr("Save Voxelized Surfaces"),
        fVpkgPath + "/voxelized_paths.zarr",
        tr("Zarr Files (*.zarr)")
    );

    if (outputPath.isEmpty()) return;

    // Create progress dialog (non-modal)
    QProgressDialog* progress = new QProgressDialog(tr("Voxelizing surfaces..."),
                                                   tr("Cancel"), 0, 100, this);
    progress->setWindowModality(Qt::NonModal);
    progress->setAttribute(Qt::WA_DeleteOnClose);

    // Gather surfaces from current paths directory
    std::map<std::string, QuadSurface*> surfacesToVoxelize;
    for (const auto& id : fVpkg->getLoadedSurfaceIDs()) {
        auto surfMeta = fVpkg->getSurface(id);
        // Only include surfaces from current segmentation directory
        if (surfMeta && surfMeta->surface()) {
            surfacesToVoxelize[id] = surfMeta->surface();
        }
    }
    
    if (surfacesToVoxelize.empty()) {
        QMessageBox::warning(this, tr("Error"), 
                           tr("No surfaces found in current paths directory."));
        return;
    }
    
    // Set up volume info from current volume
    SurfaceVoxelizer::VolumeInfo volumeInfo;
    volumeInfo.width = currentVolume->sliceWidth();
    volumeInfo.height = currentVolume->sliceHeight();
    volumeInfo.depth = currentVolume->numSlices();
    // Get voxel size from volume metadata if available
    float voxelSize = 1.0f;
    try {
        if (currentVolume->metadata().hasKey("voxelsize")) {
            voxelSize = currentVolume->metadata().get<float>("voxelsize");
        }
    } catch (...) {
        // Default to 1.0 if not found
        voxelSize = 1.0f;
    }
    volumeInfo.voxelSize = voxelSize;
    
    // Set up parameters
    SurfaceVoxelizer::VoxelizationParams params;
    params.voxelSize = volumeInfo.voxelSize; // Match volume voxel size
    params.samplingDensity = 0.5f; // Sample every 0.5 surface units
    params.fillGaps = true;
    params.chunkSize = 64;
    
    // Set OpenMP thread count based on system
    int numThreads = std::max(1, QThread::idealThreadCount() - 1); // Leave one core free
    omp_set_num_threads(numThreads);
    
    // Run voxelization in separate thread
    QFutureWatcher<void>* watcher = new QFutureWatcher<void>(this);
    connect(watcher, &QFutureWatcher<void>::finished, [progress, watcher]() {
        progress->close();
        watcher->deleteLater();
    });
    
    // Progress tracking  
    std::atomic<bool> cancelled(false);
    connect(progress, &QProgressDialog::canceled, [&cancelled]() {
        cancelled = true;
    });
    
    auto surfaces = surfacesToVoxelize;  // Copy for lambda capture
    auto outputStr = outputPath.toStdString();
    
    QFuture<void> future = QtConcurrent::run([this, outputStr, surfaces, volumeInfo, params, progress, &cancelled]() {
        try {
            SurfaceVoxelizer::voxelizeSurfaces(
                outputStr,
                surfaces,
                volumeInfo,
                params,
                [progress, &cancelled](int value) {
                    if (!cancelled) {
                        QMetaObject::invokeMethod(progress, [progress, value]() {
                            progress->setValue(value);
                        }, Qt::QueuedConnection);
                    }
                }
            );
        } catch (const std::exception& e) {
            QString errorMsg = QString::fromStdString(e.what());
            QMetaObject::invokeMethod(this, [this, errorMsg]() {
                QMessageBox::critical(this, tr("Error"), 
                    tr("Voxelization failed: %1").arg(errorMsg));
            }, Qt::QueuedConnection);
        }
    });
    
    watcher->setFuture(future);
    progress->show();  // Show progress dialog non-blocking
    
    // When voxelization completes and dialog closes, show success message
    connect(watcher, &QFutureWatcher<void>::finished, [this, outputPath, surfacesToVoxelize, volumeInfo, progress, &cancelled]() {
        if (!progress->wasCanceled() && !cancelled) {
            QMessageBox::information(this, tr("Success"),
                tr("Surfaces voxelized successfully!\n\n"
                   "Output saved to: %1\n"
                   "Surfaces processed: %2\n"
                   "Volume dimensions: %3x%4x%5")
                .arg(outputPath)
                .arg(surfacesToVoxelize.size())
                .arg(volumeInfo.width)
                .arg(volumeInfo.height)
                .arg(volumeInfo.depth));
        }
    });
}

void CWindow::onManualLocationChanged()
{
    // Check if we have a valid volume loaded
    if (!currentVolume) {
        return;
    }
    
    // Parse the comma-separated values
    QString text = lblLocFocus->text().trimmed();
    QStringList parts = text.split(',');

    // Validate we have exactly 3 parts
    if (parts.size() != 3) {
        // Invalid input - restore the previous values
        POI* poi = _surf_col->poi("focus");
        if (poi) {
            lblLocFocus->setText(QString("%1, %2, %3")
                .arg(static_cast<int>(poi->p[0]))
                .arg(static_cast<int>(poi->p[1]))
                .arg(static_cast<int>(poi->p[2])));
        }
        return;
    }

    // Parse each coordinate
    bool ok[3];
    int x = parts[0].trimmed().toInt(&ok[0]);
    int y = parts[1].trimmed().toInt(&ok[1]);
    int z = parts[2].trimmed().toInt(&ok[2]);

    // Validate the input
    if (!ok[0] || !ok[1] || !ok[2]) {
        // Invalid input - restore the previous values
        POI* poi = _surf_col->poi("focus");
        if (poi) {
            lblLocFocus->setText(QString("%1, %2, %3")
                .arg(static_cast<int>(poi->p[0]))
                .arg(static_cast<int>(poi->p[1]))
                .arg(static_cast<int>(poi->p[2])));
        }
        return;
    }

    // Clamp values to volume bounds
    int w = currentVolume->sliceWidth();
    int h = currentVolume->sliceHeight();
    int d = currentVolume->numSlices();

    x = std::max(0, std::min(x, w - 1));
    y = std::max(0, std::min(y, h - 1));
    z = std::max(0, std::min(z, d - 1));

    // Update the line edit with clamped values
    lblLocFocus->setText(QString("%1, %2, %3").arg(x).arg(y).arg(z));
    
    // Update the focus POI
    POI* poi = _surf_col->poi("focus");
    if (!poi) {
        poi = new POI;
    }
    
    poi->p = cv::Vec3f(x, y, z);
    poi->n = cv::Vec3f(0, 0, 1); // Default normal for XY plane
    
    _surf_col->setPOI("focus", poi);
    
    // Force an update of the filter
    onSegFilterChanged(0);
}

void CWindow::onZoomIn()
{
    // Get the active sub-window
    QMdiSubWindow* activeWindow = mdiArea->activeSubWindow();
    if (!activeWindow) return;
    
    // Get the viewer from the active window
    CVolumeViewer* viewer = qobject_cast<CVolumeViewer*>(activeWindow->widget());
    if (!viewer) return;
    
    // Get the center of the current view as the zoom point
    QPointF center = viewer->fGraphicsView->mapToScene(
        viewer->fGraphicsView->viewport()->rect().center());
    
    // Trigger zoom in (positive steps)
    viewer->onZoom(1, center, Qt::NoModifier);
}

void CWindow::onFocusPOIChanged(std::string name, POI* poi)
{
    if (name == "focus" && poi) {
        lblLocFocus->setText(QString("%1, %2, %3")
            .arg(static_cast<int>(poi->p[0]))
            .arg(static_cast<int>(poi->p[1]))
            .arg(static_cast<int>(poi->p[2])));

        // Force an update of the filter
        onSegFilterChanged(0);
    }
}

void CWindow::onPointDoubleClicked(uint64_t pointId)
{
    auto point_opt = _point_collection->getPoint(pointId);
    if (point_opt) {
        POI *poi = _surf_col->poi("focus");
        if (!poi) {
            poi = new POI;
        }
        poi->p = point_opt->p;

        // Find the closest normal on the segmentation surface
        Surface* seg_surface = _surf_col->surface("segmentation");
        if (auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface)) {
            auto ptr = quad_surface->pointer();
            quad_surface->pointTo(ptr, point_opt->p, 4.0, 100);
            poi->n = quad_surface->normal(ptr, quad_surface->loc(ptr));
        } else {
            poi->n = cv::Vec3f(0, 0, 1); // Default normal if no surface
        }
        
        _surf_col->setPOI("focus", poi);
    }
}

void CWindow::onZoomOut()
{
    // Get the active sub-window
    QMdiSubWindow* activeWindow = mdiArea->activeSubWindow();
    if (!activeWindow) return;
    
    // Get the viewer from the active window
    CVolumeViewer* viewer = qobject_cast<CVolumeViewer*>(activeWindow->widget());
    if (!viewer) return;
    
    // Get the center of the current view as the zoom point
    QPointF center = viewer->fGraphicsView->mapToScene(
        viewer->fGraphicsView->viewport()->rect().center());
    
    // Trigger zoom out (negative steps)
    viewer->onZoom(-1, center, Qt::NoModifier);
}

void CWindow::onCopyCoordinates()
{
    QString coords = lblLocFocus->text().trimmed();
    if (!coords.isEmpty()) {
        QApplication::clipboard()->setText(coords);
        statusBar()->showMessage(tr("Coordinates copied to clipboard: %1").arg(coords), 2000);
    }
}
