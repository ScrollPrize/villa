#include "CWindow.hpp"

#include <QKeySequence>
#include <QKeyEvent>
#include <QSettings>
#include <QMdiArea>
#include <QApplication>
#include <QClipboard>
#include <QDateTime>
#include <QFileDialog>
#include <QTextStream>
#include <QFileInfo>
#include <QDir>
#include <QProgressDialog>
#include <QMessageBox>
#include <QThread>
#include <QtConcurrent/QtConcurrent>
#include <QComboBox>
#include <QFutureWatcher>
#include <QRegularExpressionValidator>
#include <QDockWidget>
#include <QProcess>
#include <QTemporaryDir>
#include <QToolBar>
#include <QFileInfo>
#include <QTimer>
#include <QSize>
#include <QVector>
#include <QLoggingCategory>
#include <QScrollArea>
#include <QSignalBlocker>
#include <nlohmann/json.hpp>
#include <QGraphicsSimpleTextItem>
#include <QPointer>
#include <QPen>
#include <QFont>
#include <QPainter>
#include <atomic>
#include <cmath>
#include <optional>
#include <cctype>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <vector>
#include <initializer_list>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QStringList>

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
#include "SegmentationModule.hpp"
#include "SegmentationGrowth.hpp"
#include "SurfacePanelController.hpp"
#include "MenuActionController.hpp"

#include "vc/core/types/Exceptions.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Render.hpp"





Q_LOGGING_CATEGORY(lcSegGrowth, "vc.segmentation.growth");

using qga = QGuiApplication;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;

namespace
{
void ensureDockWidgetFeatures(QDockWidget* dock)
{
    if (!dock) {
        return;
    }

    auto features = dock->features();
    features |= QDockWidget::DockWidgetMovable;
    features |= QDockWidget::DockWidgetFloatable;
    features |= QDockWidget::DockWidgetClosable;
    dock->setFeatures(features);
}

void ensureGenerationsChannel(QuadSurface* surface)
{
    if (!surface) {
        return;
    }
    cv::Mat generations = surface->channel("generations");
    if (!generations.empty()) {
        return;
    }
    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }
    cv::Mat_<uint16_t> seeded(points->rows, points->cols, static_cast<uint16_t>(1));
    surface->setChannel("generations", seeded);
}

QString cacheRootForVolumePkg(const std::shared_ptr<VolumePkg>& pkg)
{
    if (!pkg) {
        return QString();
    }
    const QString base = QString::fromStdString(pkg->getVolpkgDirectory());
    return QDir(base).filePath(QStringLiteral("cache"));
}

QString normalGridDirectoryForVolumePkg(const std::shared_ptr<VolumePkg>& pkg,
                                        QString* checkedPath)
{
    if (checkedPath) {
        *checkedPath = QString();
    }

    if (!pkg) {
        qCInfo(lcSegGrowth) << "Normal grid lookup skipped (no volume package loaded)";
        return QString();
    }

    std::filesystem::path rootPath(pkg->getVolpkgDirectory());
    if (rootPath.empty()) {
        qCInfo(lcSegGrowth) << "Normal grid lookup skipped (volume package path empty)";
        return QString();
    }

    const std::filesystem::path candidate = rootPath / "normal_grids";
    const QString candidateStr = QString::fromStdString(candidate.string());
    if (checkedPath) {
        *checkedPath = candidateStr;
    }

    if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
        qCInfo(lcSegGrowth) << "Normal grid lookup at" << candidateStr << ": found";
        return candidateStr;
    }

    qCInfo(lcSegGrowth) << "Normal grid lookup at" << candidateStr << ": missing";
    return QString();
}

void ensureSurfaceMetaObject(QuadSurface* surface)
{
    if (!surface) {
        return;
    }
    if (surface->meta && surface->meta->is_object()) {
        return;
    }
    if (surface->meta) {
        delete surface->meta;
    }
    surface->meta = new nlohmann::json(nlohmann::json::object());
}

void synchronizeSurfaceMeta(const std::shared_ptr<VolumePkg>& pkg,
                            QuadSurface* surface,
                            SurfacePanelController* panel)
{
    if (!pkg || !surface) {
        return;
    }

    const auto loadedIds = pkg->getLoadedSurfaceIDs();
    for (const auto& id : loadedIds) {
        auto surfMeta = pkg->getSurface(id);
        if (!surfMeta) {
            continue;
        }
        if (surfMeta->path == surface->path) {
            if (!surfMeta->meta) {
                surfMeta->meta = new nlohmann::json(nlohmann::json::object());
            }
            if (surface->meta) {
                *surfMeta->meta = *surface->meta;
            } else {
                surfMeta->meta->clear();
            }
            surfMeta->bbox = surface->bbox();
            surfMeta->setSurface(surface);
            if (panel) {
                panel->refreshSurfaceMetrics(id);
            }
        }
    }
}

void refreshSegmentationViewers(ViewerManager* manager)
{
    if (!manager) {
        return;
    }
    manager->forEachViewer([](CVolumeViewer* viewer) {
        if (!viewer) {
            return;
        }
        if (viewer->surfName() == "segmentation") {
            viewer->invalidateVis();
            viewer->renderVisible(true);
        }
    });
}
}


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

    connect(_surf_col, &CSurfaceCollection::sendPOIChanged, this, &CWindow::onFocusPOIChanged);

    _viewerManager = std::make_unique<ViewerManager>(_surf_col, _point_collection, chunk_cache, this);
    connect(_viewerManager.get(), &ViewerManager::viewerCreated, this, [this](CVolumeViewer* viewer) {
        configureViewerConnections(viewer);
    });

    _pointsOverlay = std::make_unique<PointsOverlayController>(_point_collection, this);
    _viewerManager->setPointsOverlay(_pointsOverlay.get());

    _pathsOverlay = std::make_unique<PathsOverlayController>(this);
    _viewerManager->setPathsOverlay(_pathsOverlay.get());

    _bboxOverlay = std::make_unique<BBoxOverlayController>(this);
    _viewerManager->setBBoxOverlay(_bboxOverlay.get());

    _vectorOverlay = std::make_unique<VectorOverlayController>(_surf_col, this);
    _viewerManager->setVectorOverlay(_vectorOverlay.get());

    _volumeOverlay = std::make_unique<VolumeOverlayController>(_viewerManager.get(), this);
    connect(_volumeOverlay.get(), &VolumeOverlayController::requestStatusMessage, this,
            [this](const QString& message, int timeout) {
                if (statusBar()) {
                    statusBar()->showMessage(message, timeout);
                }
            });

    // create UI widgets
    CreateWidgets();

    // create menus/actions controller
    _menuController = std::make_unique<MenuActionController>(this);
    _menuController->populateMenus(menuBar());

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
    const QByteArray savedGeometry = geometry.value("mainWin/geometry").toByteArray();
    if (!savedGeometry.isEmpty()) {
        restoreGeometry(savedGeometry);
    }
    const QByteArray savedState = geometry.value("mainWin/state").toByteArray();
    if (!savedState.isEmpty()) {
        restoreState(savedState);
    }

    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetDrawing,
                               ui.dockWidgetOpList,
                               ui.dockWidgetOpSettings,
                               ui.dockWidgetComposite,
                               ui.dockWidgetVolumes,
                               ui.dockWidgetLocation }) {
        ensureDockWidgetFeatures(dock);
    }
    ensureDockWidgetFeatures(_point_collection_widget);

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }

    // If enabled, auto open the last used volpkg
    if (settings.value("volpkg/auto_open", false).toInt() != 0) {

        QStringList files = settings.value("volpkg/recent").toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            if (_menuController) {
                _menuController->openVolpkgAt(files[0]);
            }
        }
    }

    // Create application-wide keyboard shortcuts
    fReviewedShortcut = new QShortcut(QKeySequence("R"), this);
    fReviewedShortcut->setContext(Qt::ApplicationShortcut);
    connect(fReviewedShortcut, &QShortcut::activated, [this]() {
        if (_surfacePanel) {
            _surfacePanel->toggleTag(SurfacePanelController::Tag::Reviewed);
        }
    });
    
    fRevisitShortcut = new QShortcut(QKeySequence("Shift+R"), this);
    fRevisitShortcut->setContext(Qt::ApplicationShortcut);
    connect(fRevisitShortcut, &QShortcut::activated, [this]() {
        if (_surfacePanel) {
            _surfacePanel->toggleTag(SurfacePanelController::Tag::Revisit);
        }
    });
    
    fDefectiveShortcut = new QShortcut(QKeySequence("Shift+D"), this);
    fDefectiveShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDefectiveShortcut, &QShortcut::activated, [this]() {
        if (_surfacePanel) {
            _surfacePanel->toggleTag(SurfacePanelController::Tag::Defective);
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
        if (!_viewerManager) {
            return;
        }
        _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
            if (viewer && viewer->surfName() == "segmentation") {
                viewer->setCompositeEnabled(!viewer->isCompositeEnabled());
            }
        });
    });

    // Toggle direction hints overlay (Ctrl+T)
    fDirectionHintsShortcut = new QShortcut(QKeySequence("Ctrl+T"), this);
    fDirectionHintsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDirectionHintsShortcut, &QShortcut::activated, [this]() {
        QSettings settings("VC.ini", QSettings::IniFormat);
        bool current = settings.value("viewer/show_direction_hints", true).toBool();
        bool next = !current;
        settings.setValue("viewer/show_direction_hints", next ? "1" : "0");
        if (_viewerManager) {
            _viewerManager->forEachViewer([next](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->setShowDirectionHints(next);
                }
            });
        }
    });

    fAxisAlignedSlicesShortcut = new QShortcut(QKeySequence("Ctrl+J"), this);
    fAxisAlignedSlicesShortcut->setContext(Qt::ApplicationShortcut);
    connect(fAxisAlignedSlicesShortcut, &QShortcut::activated, [this]() {
        if (chkAxisAlignedSlices) {
            chkAxisAlignedSlices->toggle();
        }
    });

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
    if (!_viewerManager) {
        return nullptr;
    }

    CVolumeViewer* viewer = _viewerManager->createViewer(surfaceName, title, mdiArea);
    if (!viewer) {
        return nullptr;
    }

    return viewer;
}

void CWindow::configureViewerConnections(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    connect(this, &CWindow::sendVolumeChanged, viewer, &CVolumeViewer::OnVolumeChanged, Qt::UniqueConnection);
    connect(this, &CWindow::sendVolumeClosing, viewer, &CVolumeViewer::onVolumeClosing, Qt::UniqueConnection);
    connect(viewer, &CVolumeViewer::sendVolumeClicked, this, &CWindow::onVolumeClicked, Qt::UniqueConnection);

    if (viewer->fGraphicsView) {
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMousePress,
                viewer, &CVolumeViewer::onMousePress, Qt::UniqueConnection);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseMove,
                viewer, &CVolumeViewer::onMouseMove, Qt::UniqueConnection);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseRelease,
                viewer, &CVolumeViewer::onMouseRelease, Qt::UniqueConnection);
    }

    if (_drawingWidget && !viewer->property("vc_drawing_bound").toBool()) {
        connect(_drawingWidget, &DrawingWidget::sendPathsChanged,
                viewer, &CVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                _drawingWidget, &DrawingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                _drawingWidget, &DrawingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                _drawingWidget, &DrawingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendZSliceChanged,
                _drawingWidget, &DrawingWidget::updateCurrentZSlice, Qt::UniqueConnection);
    connect(_drawingWidget, &DrawingWidget::sendDrawingModeActive,
            this, [this, viewer](bool active) {
                viewer->onDrawingModeActive(active,
                    _drawingWidget->getBrushSize(),
                    _drawingWidget->getBrushShape() == PathBrushShape::Square);
            });
        viewer->setProperty("vc_drawing_bound", true);
    }

    if (_seedingWidget && !viewer->property("vc_seeding_bound").toBool()) {
        connect(_seedingWidget, &SeedingWidget::sendPathsChanged,
                viewer, &CVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                _seedingWidget, &SeedingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                _seedingWidget, &SeedingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                _seedingWidget, &SeedingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendZSliceChanged,
                _seedingWidget, &SeedingWidget::updateCurrentZSlice, Qt::UniqueConnection);
        viewer->setProperty("vc_seeding_bound", true);
    }

    if (_segmentationModule) {
        _segmentationModule->attachViewer(viewer);
    }

    if (_point_collection_widget && !viewer->property("vc_points_bound").toBool()) {
        connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
                viewer, &CVolumeViewer::onCollectionSelected, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendCollectionSelected,
                _point_collection_widget, &CPointCollectionWidget::selectCollection, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelected,
                viewer, &CVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::pointSelected,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::pointClicked,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        viewer->setProperty("vc_points_bound", true);
    }
}

CVolumeViewer* CWindow::segmentationViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }
    for (auto* viewer : _viewerManager->viewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            return viewer;
        }
    }
    return nullptr;
}

void CWindow::clearSurfaceSelection()
{
    _surf = nullptr;
    _surfID.clear();

    if (_surfacePanel) {
        _surfacePanel->resetTagUi();
    }

    if (auto* viewer = segmentationViewer()) {
        viewer->setWindowTitle(tr("Surface"));
    }

    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clearSelection();
    }

    sendOpChainSelected(nullptr);
}

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    const bool hadVolume = static_cast<bool>(currentVolume);
    POI* existingFocusPoi = _surf_col ? _surf_col->poi("focus") : nullptr;
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

    const bool growthVolumeValid = fVpkg && !_segmentationGrowthVolumeId.empty() &&
                                   fVpkg->hasVolume(_segmentationGrowthVolumeId);
    if (!growthVolumeValid) {
        _segmentationGrowthVolumeId = currentVolumeId;
        if (_segmentationWidget) {
            _segmentationWidget->setActiveVolume(QString::fromStdString(currentVolumeId));
        }
    }

    updateNormalGridAvailability();

    sendVolumeChanged(currentVolume, currentVolumeId);

    if (currentVolume && currentVolume->numScales() >= 2) {
        wOpsList->setDataset(currentVolume->zarrDataset(1), chunk_cache, 0.5);
    } else if (currentVolume) {
        wOpsList->setDataset(currentVolume->zarrDataset(0), chunk_cache, 1.0);
    }

    if (currentVolume && _surf_col) {
        const int w = currentVolume->sliceWidth();
        const int h = currentVolume->sliceHeight();
        const int d = currentVolume->numSlices();

        POI* poi = existingFocusPoi;
        const bool createdPoi = (poi == nullptr);
        if (!poi) {
            poi = new POI;
            poi->n = cv::Vec3f(0, 0, 1);
        }

        const auto clampCoord = [](float value, int maxDim) {
            if (maxDim <= 0) {
                return 0.0f;
            }
            const float maxValue = static_cast<float>(maxDim - 1);
            return std::clamp(value, 0.0f, maxValue);
        };

        if (createdPoi || !hadVolume) {
            poi->p = cv::Vec3f(w / 2.0f, h / 2.0f, d / 2.0f);
        } else {
            poi->p[0] = clampCoord(poi->p[0], w);
            poi->p[1] = clampCoord(poi->p[1], h);
            poi->p[2] = clampCoord(poi->p[2], d);
        }

        _surf_col->setPOI("focus", poi);
    }

    onManualPlaneChanged();
    applySlicePlaneOrientation(_surf_col ? _surf_col->surface("segmentation") : nullptr);
}

void CWindow::updateNormalGridAvailability()
{
    QString checkedPath;
    const QString path = normalGridDirectoryForVolumePkg(fVpkg, &checkedPath);
    const bool available = !path.isEmpty();

    _normalGridAvailable = available;
    _normalGridPath = path;

    if (_segmentationWidget) {
        _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
        QString hint;
        if (_normalGridAvailable) {
            hint = tr("Normal grids directory: %1").arg(_normalGridPath);
        } else if (!checkedPath.isEmpty()) {
            hint = tr("Checked: %1").arg(checkedPath);
        } else {
            hint = tr("No volume package loaded.");
        }
        _segmentationWidget->setNormalGridPathHint(hint);
    }
}

void CWindow::toggleVolumeOverlayVisibility()
{
    if (_volumeOverlay) {
        _volumeOverlay->toggleVisibility();
    }
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
    treeWidgetSurfaces->setSelectionMode(QAbstractItemView::ExtendedSelection);
    btnReloadSurfaces = ui.btnReloadSurfaces;

    SurfacePanelController::UiRefs surfaceUi{
        .treeWidget = treeWidgetSurfaces,
        .reloadButton = btnReloadSurfaces,
    };
    _surfacePanel = std::make_unique<SurfacePanelController>(
        surfaceUi,
        _surf_col,
        _viewerManager.get(),
        &_opchains,
        [this]() { return segmentationViewer(); },
        std::function<void()>{},
        this);
    connect(_surfacePanel.get(), &SurfacePanelController::surfacesLoaded, this, [this]() {
        emit sendSurfacesLoaded();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceSelectionCleared, this, [this]() {
        clearSurfaceSelection();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::filtersApplied, this, [this](int filterCount) {
        UpdateVolpkgLabel(filterCount);
    });
    connect(_surfacePanel.get(), &SurfacePanelController::copySegmentPathRequested,
            this, [this](const QString& segmentId) {
                if (!fVpkg) {
                    return;
                }
                auto surfMeta = fVpkg->getSurface(segmentId.toStdString());
                if (!surfMeta) {
                    return;
                }
                const QString path = QString::fromStdString(surfMeta->path.string());
                QApplication::clipboard()->setText(path);
                statusBar()->showMessage(tr("Copied segment path to clipboard: %1").arg(path), 3000);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::renderSegmentRequested,
            this, [this](const QString& segmentId) {
                onRenderSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::growSegmentRequested,
            this, [this](const QString& segmentId) {
                onGrowSegmentFromSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::addOverlapRequested,
            this, [this](const QString& segmentId) {
                onAddOverlap(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::convertToObjRequested,
            this, [this](const QString& segmentId) {
                onConvertToObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::slimFlattenRequested,
            this, [this](const QString& segmentId) {
                onSlimFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::awsUploadRequested,
            this, [this](const QString& segmentId) {
                onAWSUpload(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::growSeedsRequested,
            this, [this](const QString& segmentId, bool isExpand, bool isRandomSeed) {
                onGrowSeeds(segmentId.toStdString(), isExpand, isRandomSeed);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::teleaInpaintRequested,
            this, [this]() {
                if (_menuController) {
                    _menuController->triggerTeleaInpaint();
                }
            });
    connect(_surfacePanel.get(), &SurfacePanelController::recalcAreaRequested,
            this, [this](const QStringList& segmentIds) {
                if (segmentIds.isEmpty()) {
                    return;
                }
                std::vector<std::string> ids;
                ids.reserve(segmentIds.size());
                for (const auto& id : segmentIds) {
                    ids.push_back(id.toStdString());
                }
                recalcAreaForSegments(ids);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::statusMessageRequested,
            this, [this](const QString& message, int timeoutMs) {
                statusBar()->showMessage(message, timeoutMs);
            });

    wOpsList = new OpsList(ui.dockWidgetOpList);
    ui.dockWidgetOpList->setWidget(wOpsList);
    wOpsSettings = new OpsSettings(ui.dockWidgetOpSettings);
    ui.dockWidgetOpSettings->setWidget(wOpsSettings);

    // i recognize that having both a seeding widget and a drawing widget that both handle mouse events and paths is redundant,
    // but i can't find an easy way yet to merge them and maintain the path iteration that the seeding widget currently uses
    // so for now we have both. i suppose i could probably add a 'mode' , but for now i will just hate this section :(

    const auto attachScrollAreaToDock = [](QDockWidget* dock, QWidget* content, const QString& objectName) {
        if (!dock || !content) {
            return;
        }

        auto* container = new QWidget(dock);
        container->setObjectName(objectName);
        auto* layout = new QVBoxLayout(container);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        layout->addWidget(content);
        layout->addStretch(1);

        auto* scrollArea = new QScrollArea(dock);
        scrollArea->setWidgetResizable(true);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setWidget(container);

        dock->setWidget(scrollArea);
    };


    // Create Segmentation widget
    _segmentationWidget = new SegmentationWidget();
    _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
    const QString initialHint = _normalGridAvailable
        ? tr("Normal grids directory: %1").arg(_normalGridPath)
        : tr("No volume package loaded.");
    _segmentationWidget->setNormalGridPathHint(initialHint);
    attachScrollAreaToDock(ui.dockWidgetSegmentation, _segmentationWidget, QStringLiteral("dockWidgetSegmentationContent"));

    _segmentationEdit = std::make_unique<SegmentationEditManager>(this);
    _segmentationOverlay = std::make_unique<SegmentationOverlayController>(_surf_col, this);
    _segmentationOverlay->setEditManager(_segmentationEdit.get());

    _segmentationModule = std::make_unique<SegmentationModule>(
        _segmentationWidget,
        _segmentationEdit.get(),
        _segmentationOverlay.get(),
        _viewerManager.get(),
        _surf_col,
        _point_collection,
        _segmentationWidget->isEditingEnabled(),
        _segmentationWidget->radius(),
        _segmentationWidget->sigma(),
        this);

    connect(_segmentationModule.get(), &SegmentationModule::editingEnabledChanged,
            this, &CWindow::onSegmentationEditingModeChanged);
    connect(_segmentationModule.get(), &SegmentationModule::statusMessageRequested,
            this, &CWindow::onShowStatusMessage);
    connect(_segmentationModule.get(), &SegmentationModule::stopToolsRequested,
            this, &CWindow::onSegmentationStopToolsRequested);
    connect(_segmentationModule.get(), &SegmentationModule::focusPoiRequested,
            this, [this](const cv::Vec3f& position, QuadSurface* base) {
                Q_UNUSED(position);
                applySlicePlaneOrientation(base);
            });
    connect(_segmentationModule.get(), &SegmentationModule::growSurfaceRequested,
            this, &CWindow::onGrowSegmentationSurface);
    connect(_segmentationWidget, &SegmentationWidget::volumeSelectionChanged, this, [this](const QString& volumeId) {
        if (!fVpkg) {
            statusBar()->showMessage(tr("No volume package loaded."), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!_segmentationGrowthVolumeId.empty()
                                                                   ? _segmentationGrowthVolumeId
                                                                   : currentVolumeId);
                _segmentationWidget->setActiveVolume(fallbackId);
            }
            return;
        }

        const std::string requestedId = volumeId.toStdString();
        try {
            (void)fVpkg->volume(requestedId);
            _segmentationGrowthVolumeId = requestedId;
            statusBar()->showMessage(tr("Using volume '%1' for surface growth.").arg(volumeId), 2500);
        } catch (const std::out_of_range&) {
            statusBar()->showMessage(tr("Volume '%1' not found in this package.").arg(volumeId), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!currentVolumeId.empty()
                                                                   ? currentVolumeId
                                                                   : std::string{});
                _segmentationWidget->setActiveVolume(fallbackId);
                _segmentationGrowthVolumeId = currentVolumeId;
            }
        }
    });

    // Create Drawing widget
    _drawingWidget = new DrawingWidget();
    attachScrollAreaToDock(ui.dockWidgetDrawing, _drawingWidget, QStringLiteral("dockWidgetDrawingContent"));

    connect(this, &CWindow::sendVolumeChanged, _drawingWidget, 
            static_cast<void (DrawingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&DrawingWidget::onVolumeChanged));
    connect(_drawingWidget, &DrawingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(this, &CWindow::sendSurfacesLoaded, _drawingWidget, &DrawingWidget::onSurfacesLoaded);

    _drawingWidget->setCache(chunk_cache);
    
    // Create Seeding widget
    _seedingWidget = new SeedingWidget(_point_collection, _surf_col);
    attachScrollAreaToDock(ui.dockWidgetDistanceTransform, _seedingWidget, QStringLiteral("dockWidgetDistanceTransformContent"));
    
    connect(this, &CWindow::sendVolumeChanged, _seedingWidget, 
            static_cast<void (SeedingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&SeedingWidget::onVolumeChanged));
    connect(_seedingWidget, &SeedingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(this, &CWindow::sendSurfacesLoaded, _seedingWidget, &SeedingWidget::onSurfacesLoaded);
    
    _seedingWidget->setCache(chunk_cache);
    
    // Create and add the point collection widget
    _point_collection_widget = new CPointCollectionWidget(_point_collection, this);
    _point_collection_widget->setObjectName("pointCollectionDock");
    addDockWidget(Qt::RightDockWidgetArea, _point_collection_widget);

    // Selection dock (removed per request; selection actions remain in the menu)
    if (_viewerManager) {
        _viewerManager->forEachViewer([this](CVolumeViewer* viewer) {
            configureViewerConnections(viewer);
        });
    }
    connect(_point_collection_widget, &CPointCollectionWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);

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

    connect(this, &CWindow::sendOpChainSelected, wOpsList, &OpsList::onOpChainSelected);
    connect(wOpsList, &OpsList::sendOpSelected, wOpsSettings, &OpsSettings::onOpSelected);

    connect(wOpsList, &OpsList::sendOpChainChanged, this, &CWindow::onOpChainChanged);
    connect(wOpsSettings, &OpsSettings::sendOpChainChanged, this, &CWindow::onOpChainChanged);

    connect(_surfacePanel.get(), &SurfacePanelController::surfaceActivated,
            this, &CWindow::onSurfaceActivated);

    // new and remove path buttons
    // connect(ui.btnNewPath, SIGNAL(clicked()), this, SLOT(OnNewPathClicked()));
    // connect(ui.btnRemovePath, SIGNAL(clicked()), this, SLOT(OnRemovePathClicked()));

    // TODO CHANGE VOLUME LOADING; FIRST CHECK FOR OTHER VOLUMES IN THE STRUCTS
    volSelect = ui.volSelect;

    if (_volumeOverlay) {
        VolumeOverlayController::UiRefs overlayUi{
            .volumeSelect = ui.overlayVolumeSelect,
            .colormapSelect = ui.overlayColormapSelect,
            .opacitySlider = ui.overlayOpacitySlider,
            .thresholdSpin = ui.overlayThresholdSpin,
        };
        _volumeOverlay->setUi(overlayUi);
    }

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

    auto* chkFilterFocusPoints = ui.chkFilterFocusPoints;
    auto* cmbPointSetFilter = ui.cmbPointSetFilter;
    auto* btnPointSetFilterAll = ui.btnPointSetFilterAll;
    auto* btnPointSetFilterNone = ui.btnPointSetFilterNone;
    auto* cmbPointSetFilterMode = new QComboBox();
    cmbPointSetFilterMode->addItem("Any (OR)");
    cmbPointSetFilterMode->addItem("All (AND)");
    ui.pointSetFilterLayout->insertWidget(1, cmbPointSetFilterMode);

    SurfacePanelController::FilterUiRefs filterUi{
        .focusPoints = chkFilterFocusPoints,
        .pointSet = cmbPointSetFilter,
        .pointSetAll = btnPointSetFilterAll,
        .pointSetNone = btnPointSetFilterNone,
        .pointSetMode = cmbPointSetFilterMode,
        .unreviewed = ui.chkFilterUnreviewed,
        .revisit = ui.chkFilterRevisit,
        .noExpansion = ui.chkFilterNoExpansion,
        .noDefective = ui.chkFilterNoDefective,
        .partialReview = ui.chkFilterPartialReview,
        .hideUnapproved = ui.chkFilterHideUnapproved,
        .inspectOnly = ui.chkFilterInspectOnly,
        .currentOnly = ui.chkFilterCurrentOnly,
    };
    _surfacePanel->configureFilters(filterUi, _point_collection);

    SurfacePanelController::TagUiRefs tagUi{
        .approved = ui.chkApproved,
        .defective = ui.chkDefective,
        .reviewed = ui.chkReviewed,
        .revisit = ui.chkRevisit,
        .inspect = ui.chkInspect,
    };
    _surfacePanel->configureTags(tagUi);

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

    auto* spinIntersectionOpacity = ui.spinIntersectionOpacity;
    const int savedIntersectionOpacity = settings.value("viewer/intersection_opacity",
                                                        spinIntersectionOpacity->value()).toInt();
    const int boundedIntersectionOpacity = std::clamp(savedIntersectionOpacity,
                                                      spinIntersectionOpacity->minimum(),
                                                      spinIntersectionOpacity->maximum());
    spinIntersectionOpacity->setValue(boundedIntersectionOpacity);

    connect(spinIntersectionOpacity, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!_viewerManager) {
            return;
        }
        const float normalized = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
        _viewerManager->setIntersectionOpacity(normalized);
    });
    if (_viewerManager) {
        _viewerManager->setIntersectionOpacity(spinIntersectionOpacity->value() / 100.0f);
    }

    chkAxisAlignedSlices = ui.chkAxisAlignedSlices;
    connect(chkAxisAlignedSlices, &QCheckBox::toggled, this, &CWindow::onAxisAlignedSlicesToggled);

    spNorm[0] = ui.dspNX;
    spNorm[1] = ui.dspNY;
    spNorm[2] = ui.dspNZ;

    for (int i = 0; i < 3; i++) {
        spNorm[i]->setRange(-10, 10);
    }

    connect(spNorm[0], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[1], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[2], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);

    connect(ui.btnEditMask, &QPushButton::pressed, this, &CWindow::onEditMaskPressed);
    connect(ui.btnAppendMask, &QPushButton::pressed, this, &CWindow::onAppendMaskPressed);  // Add this
    // Connect composite view controls
    connect(ui.chkCompositeEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeEnabled(checked);
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
        
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeMethod(method);
        }
    });
    
    // Connect Layers In Front controls
    connect(ui.spinLayersInFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeLayersInFront(value);
        }
    });
    
    // Connect Layers Behind controls
    connect(ui.spinLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeLayersBehind(value);
        }
    });
    
    // Connect Alpha Min controls
    connect(ui.spinAlphaMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeAlphaMin(value);
        }
    });
    
    // Connect Alpha Max controls
    connect(ui.spinAlphaMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeAlphaMax(value);
        }
    });
    
    // Connect Alpha Threshold controls
    connect(ui.spinAlphaThreshold, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!_viewerManager) {
            return;
        }
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeAlphaThreshold(value);
                break;
            }
        }
    });
    
    // Connect Material controls
    connect(ui.spinMaterial, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!_viewerManager) {
            return;
        }
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeMaterial(value);
                break;
            }
        }
    });
    
    // Connect Reverse Direction control
    connect(ui.chkReverseDirection, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) {
            return;
        }
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeReverseDirection(checked);
                break;
            }
        }
    });
    bool resetViewOnSurfaceChange = settings.value("viewer/reset_view_on_surface_change", true).toBool();
    if (_viewerManager) {
        for (auto* viewer : _viewerManager->viewers()) {
            viewer->setResetViewOnSurfaceChange(resetViewOnSurfaceChange);
            _viewerManager->setResetDefaultFor(viewer, resetViewOnSurfaceChange);
        }
    }

}

// Create menus
// Create actions
void CWindow::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Space && event->modifiers() == Qt::NoModifier) {
        toggleVolumeOverlayVisibility();
        event->accept();
        return;
    }

    if (_segmentationModule && _segmentationModule->handleKeyPress(event)) {
        return;
    }

    QMainWindow::keyPressEvent(event);
}

void CWindow::keyReleaseEvent(QKeyEvent* event)
{
    if (_segmentationModule && _segmentationModule->handleKeyRelease(event)) {
        return;
    }

    QMainWindow::keyReleaseEvent(event);
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
    updateNormalGridAvailability();
    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        _segmentationModule->setEditingEnabled(false);
    }
    if (_segmentationWidget) {
        if (!_segmentationModule || _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

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
        updateNormalGridAvailability();
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
        updateNormalGridAvailability();
        return;
    }

    fVpkgPath = aVpkgPath;
    if (_segmentationWidget) {
        _segmentationWidget->setVolumePackagePath(aVpkgPath);
    }
    setVolume(fVpkg->volume());
    {
        const QSignalBlocker blocker{volSelect};
        volSelect->clear();
    }
    QVector<QPair<QString, QString>> volumeEntries;
    QString bestGrowthVolumeId = QString::fromStdString(currentVolumeId);
    bool preferredVolumeFound = false;
    for (const auto& id : fVpkg->volumeIDs()) {
        auto vol = fVpkg->volume(id);
        const QString idStr = QString::fromStdString(id);
        const QString nameStr = QString::fromStdString(vol->name());
        const QString label = nameStr.isEmpty() ? idStr : QStringLiteral("%1 (%2)").arg(nameStr, idStr);
        volSelect->addItem(label, QVariant(idStr));
        volumeEntries.append({idStr, label});

        const QString loweredName = nameStr.toLower();
        const QString loweredId = idStr.toLower();
        const bool matchesPreferred = loweredName.contains(QStringLiteral("surface")) ||
                                      loweredName.contains(QStringLiteral("surf")) ||
                                      loweredId.contains(QStringLiteral("surface")) ||
                                      loweredId.contains(QStringLiteral("surf"));

        if (!preferredVolumeFound && matchesPreferred) {
            bestGrowthVolumeId = idStr;
            preferredVolumeFound = true;
        }
    }

    if (bestGrowthVolumeId.isEmpty() && !volumeEntries.isEmpty()) {
        bestGrowthVolumeId = volumeEntries.front().first;
    }
    _segmentationGrowthVolumeId = bestGrowthVolumeId.toStdString();

    if (_segmentationWidget) {
        _segmentationWidget->setAvailableVolumes(volumeEntries, bestGrowthVolumeId);
    }

    if (_volumeOverlay) {
        _volumeOverlay->setVolumePkg(fVpkg, aVpkgPath);
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

    if (_surfacePanel) {
        _surfacePanel->setVolumePkg(fVpkg);
        _surfacePanel->loadSurfaces(false);
    }
    if (_menuController) {
        _menuController->updateRecentVolpkgList(aVpkgPath);
    }
    
    // Set volume package in Seeding widget
   if (_seedingWidget) {
       _seedingWidget->setVolumePkg(fVpkg);
   }

   if (_surfacePanel) {
       _surfacePanel->refreshPointSetFilterOptions();
   }
}

void CWindow::CloseVolume(void)
{
    // Notify viewers to clear their surface pointers before we delete them
    emit sendVolumeClosing();

    // Tear down active segmentation editing before surfaces disappear to avoid
    // dangling pointers inside the edit manager when the underlying surfaces
    // are unloaded (reloading with editing enabled previously triggered a
    // use-after-free crash).
    if (_segmentationModule) {
        if (_segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationModule->hasActiveSession()) {
            _segmentationModule->endEditingSession();
        }
    }

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
    _segmentationGrowthVolumeId.clear();
    updateNormalGridAvailability();
    if (_segmentationWidget) {
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

    if (_surfacePanel) {
        _surfacePanel->clear();
        _surfacePanel->setVolumePkg(nullptr);
        _surfacePanel->resetTagUi();
    }

    // Update UI
    UpdateView();
    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clear();
    }
    
    // Clear points
    _point_collection->clearAll();

    if (_volumeOverlay) {
        _volumeOverlay->clearVolumePkg();
    }
}

// Handle open request
auto CWindow::can_change_volume_() -> bool
{
    bool canChange = fVpkg != nullptr && fVpkg->numberOfVolumes() > 1;
    return canChange;
}

// Handle request to step impact range down
void CWindow::onLocChanged(void)
{
    // std::cout << "loc changed!" << "\n";
    
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
        POI *poi = _surf_col->poi("focus");
        
        if (!poi)
            poi = new POI;

        poi->src = surf;
        poi->p = vol_loc;
        poi->n = normal;
        
        _surf_col->setPOI("focus", poi);

        applySlicePlaneOrientation(segment);

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

void CWindow::onSurfaceActivated(const QString& surfaceId, QuadSurface* surface, OpChain* chain)
{
    const std::string previousSurfId = _surfID;
    _surfID = surfaceId.toStdString();
    _surf = surface;

    if (_surfID != previousSurfId) {
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }
    }

    if (chain) {
        sendOpChainSelected(chain);
    } else {
        sendOpChainSelected(nullptr);
    }

    if (_surf) {
        applySlicePlaneOrientation(_surf);
    } else {
        applySlicePlaneOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::onEditMaskPressed(void)
{
    if (!_surf)
        return;

    std::filesystem::path path = _surf->path/"mask.tif";

    if (!std::filesystem::exists(path)) {
        cv::Mat_<uint8_t> mask;
        cv::Mat_<cv::Vec3f> coords; // Not used after generation

        // Generate only the binary mask
        render_binary_mask(_surf, mask, coords);

        // Save just the mask as single layer
        cv::imwrite(path.string(), mask);

        // Update metadata
        (*_surf->meta)["date_last_modified"] = get_surface_time_str();
        _surf->save_meta();
    }

    QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));
}

void CWindow::onAppendMaskPressed(void)
{
    if (!_surf || !currentVolume) {
        if (!_surf) {
            QMessageBox::warning(this, tr("Error"), tr("No surface selected."));
        } else {
            QMessageBox::warning(this, tr("Error"), tr("No volume loaded."));
        }
        return;
    }

    std::filesystem::path path = _surf->path/"mask.tif";

    cv::Mat_<uint8_t> mask;
    cv::Mat_<uint8_t> img;
    std::vector<cv::Mat> existing_layers;

    z5::Dataset* ds = currentVolume->zarrDataset(0);

    try {
        // Find the segmentation viewer and check if composite is enabled
        CVolumeViewer* segViewer = segmentationViewer();
        bool useComposite = segViewer && segViewer->isCompositeEnabled();

        // Check if mask.tif exists
        if (std::filesystem::exists(path)) {
            // Load existing mask
            cv::imreadmulti(path.string(), existing_layers, cv::IMREAD_UNCHANGED);

            if (existing_layers.empty()) {
                QMessageBox::warning(this, tr("Error"), tr("Could not read existing mask file."));
                return;
            }

            // Use the first layer as the mask
            mask = existing_layers[0];
            cv::Size maskSize = mask.size();

            if (useComposite) {
                // Use composite rendering from the segmentation viewer
                img = segViewer->renderCompositeForSurface(_surf, maskSize);
            } else {
                // Original single-layer rendering
                cv::Vec3f ptr = _surf->pointer();
                cv::Vec3f offset(-maskSize.width/2.0f, -maskSize.height/2.0f, 0);

                cv::Mat_<cv::Vec3f> coords;
                _surf->gen(&coords, nullptr, maskSize, ptr, 1.0f, offset);

                render_image_from_coords(coords, img, ds, chunk_cache);
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            // Append the new image layer to existing layers
            existing_layers.push_back(img);

            // Save all layers
            imwritemulti(path.string(), existing_layers);

            QString message = useComposite ?
                tr("Appended composite surface image to existing mask (now %1 layers)").arg(existing_layers.size()) :
                tr("Appended surface image to existing mask (now %1 layers)").arg(existing_layers.size());
            statusBar()->showMessage(message, 3000);

        } else {
            // No existing mask, generate both mask and image
            cv::Mat_<cv::Vec3f> coords;
            render_binary_mask(_surf, mask, coords);
            cv::Size maskSize = mask.size();

            if (useComposite) {
                // Use composite rendering for image
                img = segViewer->renderCompositeForSurface(_surf, maskSize);
            } else {
                // Original rendering
                render_surface_image(_surf, mask, img, ds, chunk_cache);
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            // Save as new multi-layer TIFF
            std::vector<cv::Mat> layers = {mask, img};
            imwritemulti(path.string(), layers);

            QString message = useComposite ?
                tr("Created new surface mask with composite image data") :
                tr("Created new surface mask with image data");
            statusBar()->showMessage(message, 3000);
        }

        // Update metadata
        (*_surf->meta)["date_last_modified"] = get_surface_time_str();
        _surf->save_meta();

        QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));

    } catch (const std::exception& e) {
        QMessageBox::critical(this, tr("Error"),
                            tr("Failed to render surface: %1").arg(e.what()));
    }
}

QString CWindow::getCurrentVolumePath() const
{
    if (currentVolume == nullptr) {
        return QString();
    }
    return QString::fromStdString(currentVolume->path().string());
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
        
        if (_surfacePanel) {
            _surfacePanel->resetTagUi();
        }

        // Set the new directory in the VolumePkg
        fVpkg->setSegmentationDirectory(newDir);
        
        if (_surfacePanel) {
            _surfacePanel->loadSurfaces(false);
        }
        
        // Update the status bar to show the change
        statusBar()->showMessage(tr("Switched to %1 directory").arg(QString::fromStdString(newDir)), 3000);
    }
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
    
    if (_surfacePanel) {
        _surfacePanel->refreshFiltersOnly();
    }
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

        if (_surfacePanel) {
            _surfacePanel->refreshFiltersOnly();
        }

        applySlicePlaneOrientation();
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

void CWindow::recalcAreaForSegments(const std::vector<std::string>& ids)
{
    if (!fVpkg || ids.empty()) {
        return;
    }

    float voxelsize = 1.0f;
    try {
        if (currentVolume && currentVolume->metadata().hasKey("voxelsize")) {
            voxelsize = currentVolume->metadata().get<float>("voxelsize");
        }
    } catch (...) {
        voxelsize = 1.0f;
    }
    if (!std::isfinite(voxelsize) || voxelsize <= 0.f) {
        voxelsize = 1.0f;
    }

    int okCount = 0;
    int failCount = 0;
    QStringList skippedIds;

    auto triAreaVox2 = [](const cv::Vec3d& a,
                          const cv::Vec3d& b,
                          const cv::Vec3d& c) -> double {
        const cv::Vec3d u = b - a;
        const cv::Vec3d v = c - a;
        const cv::Vec3d w(u[1] * v[2] - u[2] * v[1],
                          u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0]);
        return 0.5 * std::sqrt(w.dot(w));
    };

    auto quadAreaVox2 = [&triAreaVox2](const cv::Vec3d& p00,
                                       const cv::Vec3d& p10,
                                       const cv::Vec3d& p01,
                                       const cv::Vec3d& p11) -> double {
        return triAreaVox2(p00, p10, p11) + triAreaVox2(p00, p11, p01);
    };

    auto isFiniteVec3 = [](const cv::Vec3d& p) {
        return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
    };

    constexpr bool INSIDE_IS_NONZERO = true;

    auto median = [](std::vector<double>& values) -> double {
        if (values.empty()) {
            return 0.0;
        }
        const auto mid = values.begin() + values.size() / 2;
        std::nth_element(values.begin(), mid, values.end());
        double m = *mid;
        if ((values.size() % 2) == 0 && mid != values.begin()) {
            const double partner = *std::max_element(values.begin(), mid);
            m = 0.5 * (m + partner);
        }
        return m;
    };

    for (const auto& id : ids) {
        auto surfMeta = fVpkg->getSurface(id);
        if (!surfMeta || !surfMeta->surface()) {
            skippedIds << QString::fromStdString(id) + tr(" (missing surface)");
            ++failCount;
            continue;
        }

        auto* surface = surfMeta->surface();
        const std::filesystem::path maskPath = surfMeta->path / "mask.tif";
        if (!std::filesystem::exists(maskPath)) {
            skippedIds << QString::fromStdString(id) + tr(" (no mask.tif)");
            ++failCount;
            continue;
        }

        std::vector<cv::Mat> layers;
        if (!cv::imreadmulti(maskPath.string(), layers, cv::IMREAD_UNCHANGED) || layers.empty()) {
            skippedIds << QString::fromStdString(id) + tr(" (mask read error)");
            ++failCount;
            continue;
        }

        cv::Mat mask = layers[0];
        if (mask.empty()) {
            skippedIds << QString::fromStdString(id) + tr(" (mask empty)");
            ++failCount;
            continue;
        }
        if (mask.channels() != 1) {
            cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
        }
        if (mask.type() != CV_8U) {
            mask.convertTo(mask, CV_8U);
        }

        cv::Mat_<cv::Vec3f> coords;
        try {
            const cv::Vec3f pointer = surface->pointer();
            const cv::Vec3f offset(-mask.cols / 2.0f, -mask.rows / 2.0f, 0.0f);
            surface->gen(&coords, nullptr, mask.size(), pointer, 1.0f, offset);
        } catch (...) {
            skippedIds << QString::fromStdString(id) + tr(" (coord gen failed)");
            ++failCount;
            continue;
        }

        if (coords.empty() || coords.rows != mask.rows || coords.cols != mask.cols) {
            skippedIds << QString::fromStdString(id) + tr(" (coord/mask size mismatch)");
            ++failCount;
            continue;
        }

        const int height = mask.rows;
        const int width = mask.cols;

        auto quadAreaFromMask = [&](const cv::Mat& maskImage,
                                    double& outDu,
                                    double& outDv) -> double {
            std::vector<double> stepsU;
            std::vector<double> stepsV;
            stepsU.reserve(static_cast<size_t>(height) * static_cast<size_t>(std::max(0, width - 1)));
            stepsV.reserve(static_cast<size_t>(std::max(0, height - 1)) * static_cast<size_t>(width));
            double areaVox2 = 0.0;

            for (int y = 0; y < height - 1; ++y) {
                const uint8_t* row0 = maskImage.ptr<uint8_t>(y);
                const uint8_t* row1 = maskImage.ptr<uint8_t>(y + 1);
                for (int x = 0; x < width - 1; ++x) {
                    const bool allBlack = (row0[x] == 0) && (row0[x + 1] == 0) &&
                                          (row1[x] == 0) && (row1[x + 1] == 0);
                    if (allBlack) {
                        continue;
                    }

                    const cv::Vec3d p00 = coords(y, x);
                    const cv::Vec3d p10 = coords(y, x + 1);
                    const cv::Vec3d p01 = coords(y + 1, x);
                    const cv::Vec3d p11 = coords(y + 1, x + 1);

                    if (!isFiniteVec3(p00) || !isFiniteVec3(p10) ||
                        !isFiniteVec3(p01) || !isFiniteVec3(p11)) {
                        continue;
                    }

                    stepsU.push_back(cv::norm(p10 - p00));
                    stepsV.push_back(cv::norm(p01 - p00));
                    areaVox2 += quadAreaVox2(p00, p10, p01, p11);
                }
            }

            outDu = median(stepsU);
            outDv = median(stepsV);
            return areaVox2;
        };

        double du = 0.0;
        double dv = 0.0;
        double areaVox2 = quadAreaFromMask(mask, du, dv);

        if (areaVox2 <= 0.0) {
            std::vector<double> globalStepsU;
            std::vector<double> globalStepsV;
            globalStepsU.reserve(static_cast<size_t>(height) * static_cast<size_t>(std::max(0, width - 1)));
            globalStepsV.reserve(static_cast<size_t>(std::max(0, height - 1)) * static_cast<size_t>(width));

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width - 1; ++x) {
                    globalStepsU.push_back(cv::norm(coords(y, x + 1) - coords(y, x)));
                }
            }
            for (int y = 0; y < height - 1; ++y) {
                for (int x = 0; x < width; ++x) {
                    globalStepsV.push_back(cv::norm(coords(y + 1, x) - coords(y, x)));
                }
            }

            const double duGlobal = median(globalStepsU);
            const double dvGlobal = median(globalStepsV);
            if (du <= 0.0) {
                du = duGlobal;
            }
            if (dv <= 0.0) {
                dv = dvGlobal;
            }

            long long pixelCount = 0;
            for (int y = 0; y < height; ++y) {
                const uint8_t* row = mask.ptr<uint8_t>(y);
                for (int x = 0; x < width; ++x) {
                    const bool inside = INSIDE_IS_NONZERO ? (row[x] != 0) : (row[x] == 0);
                    if (inside) {
                        ++pixelCount;
                    }
                }
            }

            if (pixelCount > 0 && du > 0.0 && dv > 0.0) {
                areaVox2 = static_cast<double>(pixelCount) * (du * dv);
            }
        }

        if (!std::isfinite(areaVox2)) {
            skippedIds << QString::fromStdString(id) + tr(" (non-finite area)");
            ++failCount;
            continue;
        }

        const double areaCm2 = areaVox2 * static_cast<double>(voxelsize) * static_cast<double>(voxelsize) / 1e8;
        if (!std::isfinite(areaCm2)) {
            skippedIds << QString::fromStdString(id) + tr(" (non-finite cm^2)");
            ++failCount;
            continue;
        }

        try {
            if (!surface->meta) {
                surface->meta = new nlohmann::json();
            }
            (*surface->meta)["area_vx2"] = areaVox2;
            (*surface->meta)["area_cm2"] = areaCm2;
            (*surface->meta)["date_last_modified"] = get_surface_time_str();
            surface->save_meta();
        } catch (...) {
            skippedIds << QString::fromStdString(id) + tr(" (meta save failed)");
            ++failCount;
            continue;
        }

        if (_surfacePanel) {
            _surfacePanel->refreshSurfaceMetrics(id);
        }

        ++okCount;
    }

    if (okCount > 0) {
        statusBar()->showMessage(tr("Recalculated area for %1 segment(s).").arg(okCount), 5000);
    }
    if (failCount > 0) {
        QMessageBox::warning(
            this,
            tr("Area Recalculation"),
            tr("Updated: %1\nSkipped: %2\n\n%3")
                .arg(okCount)
                .arg(failCount)
                .arg(skippedIds.join("\n")));
    }
}

void CWindow::onAxisAlignedSlicesToggled(bool enabled)
{
    _useAxisAlignedSlices = enabled;
    applySlicePlaneOrientation();
}

void CWindow::onSegmentationEditingModeChanged(bool enabled)
{
    if (!_segmentationModule) {
        return;
    }

    const bool already = _segmentationModule->editingEnabled();
    if (already != enabled) {
        // Update widget to reflect actual module state to avoid drift.
        if (_segmentationWidget && _segmentationWidget->isEditingEnabled() != already) {
            _segmentationWidget->setEditingEnabled(already);
        }
        enabled = already;
    }

    if (enabled) {
        QuadSurface* activeSurface = dynamic_cast<QuadSurface*>(_surf_col->surface("segmentation"));
        if (!activeSurface && _opchains.count(_surfID) && _opchains[_surfID]) {
            activeSurface = _opchains[_surfID]->src();
        }

        if (!_segmentationModule->beginEditingSession(activeSurface)) {
            statusBar()->showMessage(tr("Unable to start segmentation editing"), 3000);
            if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
                QSignalBlocker blocker(_segmentationWidget);
                _segmentationWidget->setEditingEnabled(false);
            }
            _segmentationModule->setEditingEnabled(false);
            return;
        }

        if (_viewerManager) {
            _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->clearOverlayGroup("segmentation_radius_indicator");
                }
            });
        }

        if (fReviewedShortcut) {
            fReviewedShortcut->setEnabled(false);
        }
    } else {
        _segmentationModule->endEditingSession();

        if (fReviewedShortcut) {
            fReviewedShortcut->setEnabled(true);
        }
    }

    const QString message = enabled
        ? tr("Segmentation editing enabled")
        : tr("Segmentation editing disabled");
    statusBar()->showMessage(message, 2000);

    if (_viewerManager) {
        _viewerManager->forEachViewer([this, enabled](CVolumeViewer* viewer) {
            if (!viewer) {
                return;
            }
            if (viewer->surfName() == "segmentation") {
                bool defaultReset = _viewerManager->resetDefaultFor(viewer);
                if (enabled) {
                    viewer->setResetViewOnSurfaceChange(false);
                } else {
                    viewer->setResetViewOnSurfaceChange(defaultReset);
                }
            }
        });
    }
}

void CWindow::onSegmentationStopToolsRequested()
{
    if (!initializeCommandLineRunner()) {
        return;
    }
    if (_cmdRunner) {
        _cmdRunner->cancel();
        statusBar()->showMessage(tr("Cancelling running tools..."), 3000);
    }
}

void CWindow::onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                        SegmentationGrowthDirection direction,
                                        int steps)
{
    qCInfo(lcSegGrowth) << "Segmentation growth requested"
                        << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << steps;
    if (_segmentationGrowthRunning) {
        qCInfo(lcSegGrowth) << "Rejecting growth because another operation is running";
        statusBar()->showMessage(tr("A surface growth operation is already running."), 4000);
        return;
    }

    auto* segmentationSurface = dynamic_cast<QuadSurface*>(_surf_col->surface("segmentation"));
    if (!segmentationSurface) {
        qCInfo(lcSegGrowth) << "Rejecting growth because segmentation surface is missing";
        statusBar()->showMessage(tr("Segmentation surface is not available."), 4000);
        return;
    }

    ensureGenerationsChannel(segmentationSurface);

    std::string growthVolumeId = !_segmentationGrowthVolumeId.empty()
                                     ? _segmentationGrowthVolumeId
                                     : currentVolumeId;
    std::shared_ptr<Volume> growthVolume;

    if (fVpkg && !growthVolumeId.empty()) {
        try {
            growthVolume = fVpkg->volume(growthVolumeId);
        } catch (const std::out_of_range&) {
            growthVolume.reset();
        }
    }

    if (!growthVolume) {
        growthVolume = currentVolume;
        growthVolumeId = currentVolumeId;
    }

    if (!growthVolume) {
        qCInfo(lcSegGrowth) << "Rejecting growth because no usable volume is available";
        statusBar()->showMessage(tr("No volume available for growth."), 4000);
        return;
    }

    if (!_segmentationGrowthVolumeId.empty() && _segmentationGrowthVolumeId != growthVolumeId) {
        statusBar()->showMessage(tr("Selected growth volume unavailable; using the active volume instead."), 4000);
    }

    _segmentationGrowthRunning = true;
    if (_segmentationModule) {
        _segmentationModule->setGrowthInProgress(true);
    }
    const auto finalize = [this]() {
        _segmentationGrowthRunning = false;
        if (_segmentationModule) {
            _segmentationModule->setGrowthInProgress(false);
        }
        qCInfo(lcSegGrowth) << "Segmentation growth finalize called";
    };

    SegmentationCorrectionsPayload corrections;
    if (_segmentationModule) {
        corrections = _segmentationModule->buildCorrectionsPayload();
    }

    const bool hasCorrections = !corrections.empty();
    const bool usingCorrections = method == SegmentationGrowthMethod::Corrections && hasCorrections;

    if (method == SegmentationGrowthMethod::Corrections && !hasCorrections) {
        qCInfo(lcSegGrowth) << "Corrections growth requested without correction points; continuing with tracer behavior.";
    }

    if (usingCorrections) {
        qCInfo(lcSegGrowth) << "Including" << corrections.collections.size() << "correction set(s)";
    }

    qCInfo(lcSegGrowth) << "Growth volume ID" << QString::fromStdString(growthVolumeId);

    qCInfo(lcSegGrowth) << "Starting tracer growth";
    if (method == SegmentationGrowthMethod::Corrections) {
        if (usingCorrections) {
            statusBar()->showMessage(tr("Applying correction-guided tracer growth..."), 2000);
        } else {
            statusBar()->showMessage(tr("No correction points provided; running tracer growth..."), 2000);
        }
    } else {
        statusBar()->showMessage(tr("Running tracer-based surface growth..."), 2000);
    }

    SegmentationGrowthRequest request;
    request.method = method;
    request.direction = direction;
    request.steps = steps;
    if (_segmentationModule) {
        if (auto overrideDirs = _segmentationModule->takeShortcutDirectionOverride()) {
            request.allowedDirections = std::move(*overrideDirs);
        }
    }
    if (request.allowedDirections.empty()) {
        if (_segmentationWidget) {
            request.allowedDirections = _segmentationWidget->allowedGrowthDirections();
        } else {
            request.allowedDirections = {
                SegmentationGrowthDirection::Up,
                SegmentationGrowthDirection::Down,
                SegmentationGrowthDirection::Left,
                SegmentationGrowthDirection::Right
            };
        }
    }
    if (_segmentationWidget) {
        request.directionFields = _segmentationWidget->directionFieldConfigs();
        if (!_segmentationWidget->customParamsValid()) {
            const QString errorText = _segmentationWidget->customParamsError();
            const QString message = errorText.isEmpty()
                ? tr("Custom params JSON is invalid. Fix the contents and try again.")
                : tr("Custom params JSON is invalid: %1").arg(errorText);
            statusBar()->showMessage(message, 5000);
            finalize();
            return;
        }
        if (auto customParams = _segmentationWidget->customParamsJson()) {
            request.customParams = std::move(*customParams);
        }
    }
    request.corrections = corrections;
    if (method == SegmentationGrowthMethod::Corrections && _segmentationModule) {
        if (auto zRange = _segmentationModule->correctionsZRange()) {
            request.correctionsZRange = zRange;
        }
    }

    TracerGrowthContext ctx;
    ctx.resumeSurface = segmentationSurface;
    ctx.volume = growthVolume.get();
    ctx.cache = chunk_cache;
    ctx.cacheRoot = cacheRootForVolumePkg(fVpkg);
    ctx.voxelSize = growthVolume->voxelSize();
    ctx.normalGridPath = _normalGridPath;

    if (ctx.cacheRoot.isEmpty()) {
        const auto volumePath = growthVolume->path();
        ctx.cacheRoot = QDir(QString::fromStdString(volumePath.parent_path().string())).filePath(QStringLiteral("cache"));
    }

    if (ctx.cacheRoot.isEmpty()) {
        qCInfo(lcSegGrowth) << "Tracer growth aborted because cache root is empty";
        statusBar()->showMessage(tr("Cache root unavailable for tracer growth."), 5000);
        finalize();
        return;
    }

    const double growthVoxelSize = growthVolume ? growthVolume->voxelSize() : 0.0;

    qCInfo(lcSegGrowth) << "Launching tracer future" << ctx.cacheRoot;
    auto future = QtConcurrent::run(runTracerGrowth, request, ctx);
    auto* watcher = new QFutureWatcher<TracerGrowthResult>(this);
    _tracerGrowthWatcher.reset(watcher);

    connect(watcher, &QFutureWatcher<TracerGrowthResult>::finished, this, [this, segmentationSurface, finalize, usingCorrections, growthVoxelSize, growthVolume]() {
        Q_UNUSED(growthVolume);
        if (!_tracerGrowthWatcher) {
            qCInfo(lcSegGrowth) << "Tracer watcher finished but watcher reset early";
            finalize();
            return;
        }

        const TracerGrowthResult result = _tracerGrowthWatcher->result();
        _tracerGrowthWatcher.reset();

        qCInfo(lcSegGrowth) << "Tracer growth finished" << (result.error.isEmpty() ? "success" : "error");

        if (!result.error.isEmpty()) {
            qCInfo(lcSegGrowth) << "Tracer growth error" << result.error;
            statusBar()->showMessage(result.error, 6000);
            finalize();
            return;
        }

        if (!result.surface) {
            qCInfo(lcSegGrowth) << "Tracer growth returned null surface";
            statusBar()->showMessage(tr("Tracer growth did not return a surface."), 5000);
            finalize();
            return;
        }

        const double voxelSize = growthVoxelSize;
        cv::Mat generations = result.surface->channel("generations");

        std::vector<QuadSurface*> surfacesToUpdate;
        if (_segmentationModule && _segmentationModule->hasActiveSession()) {
            if (auto* baseSurface = _segmentationModule->activeBaseSurface()) {
                surfacesToUpdate.push_back(baseSurface);
            }
        }
        if (std::find(surfacesToUpdate.begin(), surfacesToUpdate.end(), segmentationSurface) == surfacesToUpdate.end()) {
            surfacesToUpdate.push_back(segmentationSurface);
        }

        for (QuadSurface* targetSurface : surfacesToUpdate) {
            if (!targetSurface) {
                continue;
            }

            if (auto* destPoints = targetSurface->rawPointsPtr()) {
                result.surface->rawPoints().copyTo(*destPoints);
            }

            if (!generations.empty()) {
                targetSurface->setChannel("generations", generations);
            }

            targetSurface->invalidateCache();

            if (result.surface->meta) {
                if (targetSurface->meta) {
                    delete targetSurface->meta;
                    targetSurface->meta = nullptr;
                }
                targetSurface->meta = new nlohmann::json(*result.surface->meta);
            } else {
                ensureSurfaceMetaObject(targetSurface);
            }

            updateSegmentationSurfaceMetadata(targetSurface, voxelSize);
        }

        QuadSurface* surfaceToPersist = nullptr;
        if (_segmentationModule && _segmentationModule->hasActiveSession()) {
            surfaceToPersist = _segmentationModule->activeBaseSurface();
        }
        if (!surfaceToPersist) {
            surfaceToPersist = segmentationSurface;
        }

        try {
            if (surfaceToPersist) {
                ensureSurfaceMetaObject(surfaceToPersist);
                surfaceToPersist->saveOverwrite();
            }
        } catch (const std::exception& ex) {
            qCInfo(lcSegGrowth) << "Failed to save tracer result" << ex.what();
            statusBar()->showMessage(tr("Failed to save segmentation: %1").arg(ex.what()), 5000);
        }

        std::vector<std::pair<CVolumeViewer*, bool>> resetDefaults;
        if (_viewerManager) {
            _viewerManager->forEachViewer([this, &resetDefaults](CVolumeViewer* viewer) {
                if (!viewer || viewer->surfName() != "segmentation") {
                    return;
                }
                const bool defaultReset = _viewerManager->resetDefaultFor(viewer);
                resetDefaults.emplace_back(viewer, defaultReset);
                viewer->setResetViewOnSurfaceChange(false);
            });
        }

        _surf_col->setSurface("segmentation", segmentationSurface);

        if (!resetDefaults.empty()) {
            const bool editingActive = _segmentationModule && _segmentationModule->editingEnabled();
            for (auto& entry : resetDefaults) {
                auto* viewer = entry.first;
                if (!viewer) {
                    continue;
                }
                if (editingActive) {
                    viewer->setResetViewOnSurfaceChange(false);
                } else {
                    viewer->setResetViewOnSurfaceChange(entry.second);
                }
            }
        }

        if (_segmentationModule && _segmentationModule->hasActiveSession()) {
            _segmentationModule->markNextHandlesFromGrowth();
            qCInfo(lcSegGrowth) << "Refreshing active segmentation session after tracer growth";
            _segmentationModule->refreshSessionFromSurface(surfaceToPersist);
        }

        QuadSurface* currentSegSurface = dynamic_cast<QuadSurface*>(_surf_col->surface("segmentation"));
        if (!currentSegSurface) {
            currentSegSurface = segmentationSurface;
        }

        synchronizeSurfaceMeta(fVpkg, currentSegSurface, _surfacePanel ? _surfacePanel.get() : nullptr);
        applySlicePlaneOrientation(currentSegSurface);
        refreshSegmentationViewers(_viewerManager.get());

        if (usingCorrections && _segmentationModule) {
            _segmentationModule->clearPendingCorrections();
        }

        qCInfo(lcSegGrowth) << "Tracer growth completed successfully";
        delete result.surface;

        QString message = result.statusMessage.isEmpty() ? tr("Tracer growth complete.") : result.statusMessage;
        if (usingCorrections) {
            message = tr("Corrections applied; tracer growth complete.");
        }
        statusBar()->showMessage(message, 4000);
        finalize();
    });

    watcher->setFuture(future);
}

void CWindow::applySlicePlaneOrientation(Surface* sourceOverride)
{
    if (!_surf_col) {
        return;
    }

    POI *focus = _surf_col->poi("focus");
    cv::Vec3f origin = focus ? focus->p : cv::Vec3f(0, 0, 0);

    if (_useAxisAlignedSlices) {
        PlaneSurface *segXZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg xz"));
        PlaneSurface *segYZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg yz"));

        if (!segXZ) {
            segXZ = new PlaneSurface();
        }
        if (!segYZ) {
            segYZ = new PlaneSurface();
        }

        segXZ->setOrigin(origin);
        segYZ->setOrigin(origin);

        segXZ->setNormal(cv::Vec3f(0, 1, 0));
        segYZ->setNormal(cv::Vec3f(1, 0, 0));

        _surf_col->setSurface("seg xz", segXZ);
        _surf_col->setSurface("seg yz", segYZ);
        return;
    } else {
        auto* segment = dynamic_cast<QuadSurface*>(sourceOverride ? sourceOverride : _surf_col->surface("segmentation"));
        if (!segment) {
            return;
        }

        PlaneSurface *segXZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg xz"));
        PlaneSurface *segYZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg yz"));

        if (!segXZ) {
            segXZ = new PlaneSurface();
        }
        if (!segYZ) {
            segYZ = new PlaneSurface();
        }

        segXZ->setOrigin(origin);
        segYZ->setOrigin(origin);

        auto ptr = segment->pointer();
        segment->pointTo(ptr, origin, 1.0f);

        cv::Vec3f xDir = segment->coord(ptr, {1, 0, 0});
        cv::Vec3f yDir = segment->coord(ptr, {0, 1, 0});
        segXZ->setNormal(xDir - origin);
        segYZ->setNormal(yDir - origin);

        _surf_col->setSurface("seg xz", segXZ);
        _surf_col->setSurface("seg yz", segYZ);
        return;
    }
}
