#include "CWindow.hpp"
#include <iostream>

#include <functional>

#include "VCSettings.hpp"
#include "Keybinds.hpp"
#include <QGridLayout>
#include <QCursor>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <QGuiApplication>
#include <QStyleHints>
#include <QWindow>
#include <QScreen>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QFileDialog>
#include <QFileInfo>
#include <QPointF>
#include <QMessageBox>
#include <QtConcurrent/QtConcurrent>
#include <QComboBox>
#include <QFutureWatcher>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QDockWidget>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QSizePolicy>
#include <QTimer>
#include <QSize>
#include <QVector>
#include <QLoggingCategory>
#include <QDebug>
#include <QScrollArea>
#include <QSignalBlocker>
#include "utils/Json.hpp"
#include <QPointer>
#include <QListView>
#include <algorithm>
#include <cmath>
#include "vc/core/types/Segmentation.hpp"
#include <limits>
#include <optional>
#include <cctype>
#include <utility>
#include <filesystem>
#include <system_error>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QStringList>

#include "volume_viewers/CVolumeViewerView.hpp"
#include "viewer_controls/ViewerControlsPanel.hpp"
#include "viewer_controls/panels/ViewerTransformsPanel.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"
#include "SettingsDialog.hpp"
#include "elements/VolumeSelector.hpp"
#include "CPointCollectionWidget.hpp"
#include "CFiberWidget.hpp"
#include "FiberAnnotationController.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "CommandLineToolRunner.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "SurfacePanelController.hpp"
#include "MenuActionController.hpp"
#include "FileWatcherService.hpp"
#include "AxisAlignedSliceController.hpp"
#include "SurfaceAreaCalculator.hpp"
#include "SegmentationCommandHandler.hpp"
#include "LasagnaServiceManager.hpp"
#include "segmentation/panels/SegmentationLasagnaPanel.hpp"
#include "vc/core/Version.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Render.hpp"
#include "vc/core/util/Tiff.hpp"
#include <utils/zarr.hpp>





Q_LOGGING_CATEGORY(lcSegGrowth, "vc.segmentation.growth");

using qga = QGuiApplication;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;
namespace
{

VolumeViewerBase* baseViewerFromWidget(QWidget* widget)
{
    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(widget)) {
        return chunkedViewer;
    }
    return nullptr;
}

bool isChunkedViewer(VolumeViewerBase* viewer)
{
    return viewer && qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
}

void centerViewerOnVolumePointForNavigation(VolumeViewerBase* viewer, const cv::Vec3f& position)
{
    if (!viewer) {
        return;
    }
    viewer->centerOnVolumePoint(position, !isChunkedViewer(viewer));
}

void centerViewerOnSurfacePointForNavigation(VolumeViewerBase* viewer, const cv::Vec2f& position)
{
    if (!viewer) {
        return;
    }
    viewer->centerOnSurfacePoint(position, !isChunkedViewer(viewer));
}

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

    auto paths = pkg->normalGridPaths();
    if (paths.empty()) {
        qCInfo(lcSegGrowth) << "Normal grid lookup: no normal_grids entries in project";
        return QString();
    }
    const QString candidateStr = QString::fromStdString(paths.front().string());
    if (checkedPath) *checkedPath = candidateStr;
    qCInfo(lcSegGrowth) << "Normal grid resolved to" << candidateStr;
    return candidateStr;
}

QStringList normal3dZarrCandidatesForVolumePkg(const std::shared_ptr<VolumePkg>& pkg,
                                               QString* hint)
{
    if (hint) {
        *hint = QString();
    }
    if (!pkg) {
        if (hint) {
            *hint = QObject::tr("Normal3D lookup skipped (no volume package loaded)");
        }
        return {};
    }
    auto paths = pkg->normal3dZarrPaths();
    QStringList candidates;
    for (const auto& p : paths) candidates.push_back(QString::fromStdString(p.string()));
    candidates.sort();
    if (hint) {
        *hint = candidates.isEmpty()
            ? QObject::tr("No volumes tagged 'normal3d' in project")
            : QObject::tr("%1 normal3d zarr(s) tagged").arg(candidates.size());
    }
    return candidates;
}

QString absoluteSegmentPathForClipboard(const std::filesystem::path& segmentPath,
                                        const std::shared_ptr<VolumePkg>& pkg)
{
    auto path = segmentPath;
    if (!path.is_absolute() && pkg) {
        const auto projectPath = pkg->path();
        const auto projectDir = projectPath.has_parent_path()
            ? projectPath.parent_path()
            : std::filesystem::current_path();
        path = projectDir / path;
    }

    std::error_code ec;
    const auto absolutePath = std::filesystem::absolute(path, ec);
    if (!ec) {
        path = absolutePath;
    }
    return QString::fromStdString(path.lexically_normal().string());
}

constexpr float kEpsilon = 1e-6f;

cv::Vec3f projectVectorOntoPlane(const cv::Vec3f& v, const cv::Vec3f& normal)
{
    const float dot = v.dot(normal);
    return v - normal * dot;
}

cv::Vec3f normalizeOrZero(const cv::Vec3f& v)
{
    const float magnitude = cv::norm(v);
    if (magnitude <= kEpsilon) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }
    return v * (1.0f / magnitude);
}

cv::Vec3f crossProduct(const cv::Vec3f& a, const cv::Vec3f& b)
{
    return cv::Vec3f(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]);
}

float signedAngleBetween(const cv::Vec3f& from, const cv::Vec3f& to, const cv::Vec3f& axis)
{
    cv::Vec3f fromNorm = normalizeOrZero(from);
    cv::Vec3f toNorm = normalizeOrZero(to);
    if (cv::norm(fromNorm) <= kEpsilon || cv::norm(toNorm) <= kEpsilon) {
        return 0.0f;
    }

    float dot = fromNorm.dot(toNorm);
    dot = std::clamp(dot, -1.0f, 1.0f);
    cv::Vec3f cross = crossProduct(fromNorm, toNorm);
    float angle = std::atan2(cv::norm(cross), dot);
    float sign = cross.dot(axis) >= 0.0f ? 1.0f : -1.0f;
    return angle * sign;
}

} // namespace

// Dark mode detection - works on all Qt 6.x versions
static bool isDarkMode() {
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    if (QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark)
        return true;
#endif
    // Fallback: check system palette brightness
    const auto windowColor = QGuiApplication::palette().color(QPalette::Window);
    return windowColor.lightness() < 128;
}

// Apply a consistent dark palette application-wide
static void applyDarkPalette() {
    QPalette p;
    p.setColor(QPalette::Window, QColor(53, 53, 53));
    p.setColor(QPalette::WindowText, Qt::white);
    p.setColor(QPalette::Base, QColor(42, 42, 42));
    p.setColor(QPalette::AlternateBase, QColor(66, 66, 66));
    p.setColor(QPalette::ToolTipBase, QColor(53, 53, 53));
    p.setColor(QPalette::ToolTipText, Qt::white);
    p.setColor(QPalette::Text, Qt::white);
    p.setColor(QPalette::Button, QColor(53, 53, 53));
    p.setColor(QPalette::ButtonText, Qt::white);
    p.setColor(QPalette::BrightText, Qt::red);
    p.setColor(QPalette::Link, QColor(42, 130, 218));
    p.setColor(QPalette::Highlight, QColor(42, 130, 218));
    p.setColor(QPalette::HighlightedText, Qt::black);
    p.setColor(QPalette::Disabled, QPalette::Text, QColor(127, 127, 127));
    p.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(127, 127, 127));
    QApplication::setPalette(p);
}

static QString windowStateScreenSignature()
{
    QStringList parts;
    parts << qga::platformName();
    const auto screens = qga::screens();
    parts << QString::number(screens.size());
    for (const QScreen* screen : screens) {
        if (!screen) {
            continue;
        }
        const QRect geom = screen->geometry();
        const qreal dpr = screen->devicePixelRatio();
        const QString name = screen->name().isEmpty() ? QStringLiteral("screen") : screen->name();
        parts << QString("%1:%2x%3+%4+%5@%6")
                     .arg(name)
                     .arg(geom.width())
                     .arg(geom.height())
                     .arg(geom.x())
                     .arg(geom.y())
                     .arg(dpr, 0, 'f', 2);
    }
    return parts.join("|");
}

static QString windowStateQtVersion()
{
    return QString::fromUtf8(qVersion());
}

static QString windowStateAppVersion()
{
    return QString::fromStdString(ProjectInfo::VersionString());
}

static void writeWindowStateMeta(QSettings& settings,
                                 const QString& screenSignature,
                                 const QString& qtVersion,
                                 const QString& appVersion)
{
    settings.setValue(vc3d::settings::window::STATE_META_SCREEN_SIGNATURE, screenSignature);
    settings.setValue(vc3d::settings::window::STATE_META_QT_VERSION, qtVersion);
    settings.setValue(vc3d::settings::window::STATE_META_APP_VERSION, appVersion);
}

static bool windowStateMetaMatches(const QSettings& settings,
                                   const QString& screenSignature,
                                   const QString& qtVersion,
                                   const QString& appVersion)
{
    const QString savedSignature =
        settings.value(vc3d::settings::window::STATE_META_SCREEN_SIGNATURE).toString();
    const QString savedQtVersion =
        settings.value(vc3d::settings::window::STATE_META_QT_VERSION).toString();
    const QString savedAppVersion =
        settings.value(vc3d::settings::window::STATE_META_APP_VERSION).toString();

    if (savedSignature.isEmpty() || savedQtVersion.isEmpty() || savedAppVersion.isEmpty()) {
        return false;
    }

    return savedSignature == screenSignature
        && savedQtVersion == qtVersion
        && savedAppVersion == appVersion;
}

// Constructor
CWindow::CWindow(size_t cacheSizeGB) :
    _cmdRunner(nullptr),
    _seedingWidget(nullptr),
    _point_collection_widget(nullptr)
{
    // Initialize timer for debounced window state saving (500ms delay)
    _windowStateSaveTimer = new QTimer(this);
    _windowStateSaveTimer->setSingleShot(true);
    _windowStateSaveTimer->setInterval(500);
    connect(_windowStateSaveTimer, &QTimer::timeout, this, &CWindow::saveWindowState);

    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _mirrorCursorToSegmentation = settings.value(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION,
                                                  vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION_DEFAULT).toBool();
    setWindowIcon(QPixmap(":/images/logo.png"));
    ui.setupUi(this);
    const QString baseTitle = windowTitle();
    const QString repoShortHash = QString::fromStdString(ProjectInfo::RepositoryShortHash()).trimmed();
    if (!repoShortHash.isEmpty() && !repoShortHash.startsWith('@')
        && repoShortHash.compare("Untracked", Qt::CaseInsensitive) != 0) {
        setWindowTitle(QString("%1 %2").arg(baseTitle, repoShortHash));
    }
    // setAttribute(Qt::WA_DeleteOnClose);

    _cacheSizeBytes = cacheSizeGB * 1024ULL * 1024ULL * 1024ULL;
    std::cout << "chunk cache budget is " << cacheSizeGB << " gigabytes" << std::endl;

    _state = new CState(_cacheSizeBytes, this);
    connect(_state, &CState::poiChanged, this, &CWindow::onFocusPOIChanged);
    connect(_state, &CState::surfaceWillBeDeleted, this, &CWindow::onSurfaceWillBeDeleted);
    connect(_state, &CState::vpkgChanged, this,
            [this](std::shared_ptr<VolumePkg> pkg) {
                if (!pkg) return;
                pkg->setSegmentsChangedCallback(
                    [self = QPointer<CWindow>(this)]() {
                        QMetaObject::invokeMethod(self.data(), [self]() {
                            auto* w = self.data();
                            if (!w || !w->_surfacePanel || !w->_state || !w->_state->vpkg()) return;
                            w->_surfacePanel->setVolumePkg(w->_state->vpkg());
                            w->_surfacePanel->refreshSurfaceList();
                        }, Qt::QueuedConnection);
                    });
            });

    _fileWatcher = std::make_unique<FileWatcherService>(_state, this);
    connect(_fileWatcher.get(), &FileWatcherService::statusMessage,
            this, &CWindow::onShowStatusMessage);
    connect(_fileWatcher.get(), &FileWatcherService::volumeCatalogChanged,
            this, [this](const QString& preferredVolumeId) {
                refreshCurrentVolumePackageUi(preferredVolumeId, false);
            });

    _axisAlignedSliceController = std::make_unique<AxisAlignedSliceController>(_state, this);

    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
    _viewerManager->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);
    connect(_viewerManager.get(), &ViewerManager::baseViewerCreated, this, [this](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }
        if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
            configureChunkedViewerConnections(chunkedViewer);
        }
    });

    // Slice step size label in status bar
    _sliceStepLabel = new QLabel(this);
    _sliceStepLabel->setContentsMargins(4, 0, 4, 0);
    int initialStepSize = _viewerManager->sliceStepSize();
    _sliceStepLabel->setText(tr("Step: %1").arg(initialStepSize));
    _sliceStepLabel->setToolTip(tr("Slice step size: use Shift+G / Shift+H to adjust"));
    statusBar()->addPermanentWidget(_sliceStepLabel);

    _pointsOverlay = std::make_unique<PointsOverlayController>(_state->pointCollection(), this);
    _viewerManager->setPointsOverlay(_pointsOverlay.get());

    _rawPointsOverlay = std::make_unique<RawPointsOverlayController>(_state, this);
    _viewerManager->setRawPointsOverlay(_rawPointsOverlay.get());

    _pathsOverlay = std::make_unique<PathsOverlayController>(this);
    _viewerManager->setPathsOverlay(_pathsOverlay.get());

    _bboxOverlay = std::make_unique<BBoxOverlayController>(this);
    _viewerManager->setBBoxOverlay(_bboxOverlay.get());

    _vectorOverlay = std::make_unique<VectorOverlayController>(_state, this);
    _viewerManager->setVectorOverlay(_vectorOverlay.get());

    _planeSlicingOverlay = std::make_unique<PlaneSlicingOverlayController>(_state, this);
    _planeSlicingOverlay->bindToViewerManager(_viewerManager.get());
    _planeSlicingOverlay->setRotationSetter([this](const std::string& planeName, float degrees) {
        _axisAlignedSliceController->setRotationDegrees(planeName, degrees);
        _axisAlignedSliceController->scheduleOrientationUpdate();
    });
    _planeSlicingOverlay->setRotationFinishedCallback([this]() {
        _axisAlignedSliceController->flushOrientationUpdate();
    });
    _planeSlicingOverlay->setAxisAlignedEnabled(_axisAlignedSliceController && _axisAlignedSliceController->isEnabled());

    _axisAlignedSliceController->setPlaneSlicingOverlay(_planeSlicingOverlay.get());
    _axisAlignedSliceController->setViewerManager(_viewerManager.get());

    _volumeOverlay = std::make_unique<VolumeOverlayController>(_viewerManager.get(), this);
    connect(_volumeOverlay.get(), &VolumeOverlayController::requestStatusMessage, this,
            [this](const QString& message, int timeout) {
                if (statusBar()) {
                    statusBar()->showMessage(message, timeout);
                }
            });
    _viewerManager->setVolumeOverlay(_volumeOverlay.get());

    // create UI widgets
    CreateWidgets();

    // create menus/actions controller
    _menuController = std::make_unique<MenuActionController>(this);
    _menuController->populateMenus(menuBar());
    // Wire the Actions -> Merge tifxyz... menu entry to the handler.
    // Has to happen here (not inside CreateWidgets) because
    // _menuController is created after CreateWidgets() returns.
    connect(_menuController.get(), &MenuActionController::mergeTifxyzFromMenuRequested,
            this, [this]() {
                _segmentationCommandHandler->onMergeTifxyz(QStringList{});
            });

    if (isDarkMode()) {
        applyDarkPalette();
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
    } else {
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
    QSettings geometry(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString currentScreenSignature = windowStateScreenSignature();
    const QString currentQtVersion = windowStateQtVersion();
    const QString currentAppVersion = windowStateAppVersion();

    const bool restoreDisabled =
        geometry.value(vc3d::settings::window::RESTORE_DISABLED, false).toBool();
    const bool restoreInProgress =
        geometry.value(vc3d::settings::window::RESTORE_IN_PROGRESS, false).toBool();

    auto clearSavedWindowState = [&geometry]() {
        geometry.remove(vc3d::settings::window::GEOMETRY);
        geometry.remove(vc3d::settings::window::STATE);
    };

    bool allowRestore = !restoreDisabled && !restoreInProgress;
    if (restoreInProgress) {
        Logger()->warn("Previous window-state restore did not complete; clearing saved state");
        clearSavedWindowState();
        geometry.setValue(vc3d::settings::window::RESTORE_DISABLED, true);
        geometry.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
        geometry.sync();
        allowRestore = false;
    }
    const bool hasStateMeta =
        geometry.contains(vc3d::settings::window::STATE_META_SCREEN_SIGNATURE)
        && geometry.contains(vc3d::settings::window::STATE_META_QT_VERSION)
        && geometry.contains(vc3d::settings::window::STATE_META_APP_VERSION);
    if (allowRestore && hasStateMeta
        && !windowStateMetaMatches(geometry,
                                   currentScreenSignature,
                                   currentQtVersion,
                                   currentAppVersion)) {
        Logger()->warn("Window state metadata mismatch; skipping restore");
        clearSavedWindowState();
        writeWindowStateMeta(geometry, currentScreenSignature, currentQtVersion, currentAppVersion);
        geometry.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
        geometry.sync();
        allowRestore = false;
    }

    if (allowRestore) {
        geometry.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, true);
        geometry.sync();
    }

    bool restoredGeometry = false;
    bool restoredState = false;
    if (allowRestore) {
        const QByteArray savedGeometry = geometry.value(vc3d::settings::window::GEOMETRY).toByteArray();
        if (!savedGeometry.isEmpty()) {
            restoredGeometry = restoreGeometry(savedGeometry);
            if (!restoredGeometry) {
                Logger()->warn("Failed to restore main window geometry; clearing saved geometry");
                geometry.remove(vc3d::settings::window::GEOMETRY);
                geometry.sync();
            }
        }
        const QByteArray savedState = geometry.value(vc3d::settings::window::STATE).toByteArray();
        if (!savedState.isEmpty()) {
            restoredState = restoreState(savedState);
            if (!restoredState) {
                Logger()->warn("Failed to restore main window state; clearing saved state");
                geometry.remove(vc3d::settings::window::STATE);
                geometry.sync();
            }
        }
    }
    if (allowRestore) {
        QTimer::singleShot(1500, this, [currentScreenSignature, currentQtVersion, currentAppVersion]() {
            QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
            if (settings.value(vc3d::settings::window::RESTORE_DISABLED, false).toBool()) {
                settings.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
                settings.sync();
                return;
            }
            writeWindowStateMeta(settings, currentScreenSignature, currentQtVersion, currentAppVersion);
            settings.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
            settings.sync();
        });
    }
    // Ensure right-side tabified docks have a usable minimum size
    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               _lasagnaDock,
                               ui.dockWidgetDistanceTransform }) {
        if (dock) {
            dock->setMinimumWidth(250);
            dock->setMinimumHeight(120);
        }
    }
    if (!restoredState) {
        // No saved state - set sensible default sizes for dock widgets
        resizeDocks({ui.dockWidgetVolumes}, {300}, Qt::Horizontal);
        resizeDocks({ui.dockWidgetVolumes}, {400}, Qt::Vertical);
        resizeDocks({ui.dockWidgetSegmentation}, {350}, Qt::Horizontal);
    }

    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               _lasagnaDock,
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetVolumes,
                               ui.dockWidgetViewerControls  }) {
        ensureDockWidgetFeatures(dock);
        // Connect dock widget signals to trigger state saving
        connect(dock, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
        connect(dock, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);
    }
    ensureDockWidgetFeatures(_point_collection_widget);
    connect(_point_collection_widget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
    connect(_point_collection_widget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);

    // Wayland workaround: dock drags trigger grabMouse() which fails,
    // leaving Qt's internal state stuck so all mouse events stop.
    // Synthesize a mouse release to clear the stuck button state.
    if (QGuiApplication::platformName() == QLatin1String("wayland")) {
        auto fixGrab = [this](){
            // Defer to after Qt finishes its internal dock drag processing.
            QTimer::singleShot(100, this, [](){
                if (auto* g = QWidget::mouseGrabber())
                    g->releaseMouse();
                for (auto* w : QGuiApplication::topLevelWindows())
                    w->setMouseGrabEnabled(false);
            });
        };
        for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                                   _lasagnaDock,
                                   ui.dockWidgetDistanceTransform,
                                   ui.dockWidgetVolumes,
                                   ui.dockWidgetViewerControls,
                                   static_cast<QDockWidget*>(_point_collection_widget) }) {
            if (!dock) continue;
            connect(dock, &QDockWidget::topLevelChanged, this, fixGrab);
            connect(dock, &QDockWidget::dockLocationChanged, this, fixGrab);
        }
    }

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }

    // If enabled, auto open the last used local volume package.
    if (settings.value(vc3d::settings::project::AUTO_OPEN, vc3d::settings::project::AUTO_OPEN_DEFAULT).toInt() != 0) {

        QStringList files = settings.value(vc3d::settings::project::RECENT).toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            QString path = files[0];
            QTimer::singleShot(0, this, [this, path]() {
                if (_menuController) {
                    _menuController->openVolpkgAt(path);
                }
            });
        }
    }

    // Create application-wide keyboard shortcuts
    fCompositeViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CompositeView), this);
    fCompositeViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCompositeViewShortcut, &QShortcut::activated, [this]() {
        if (_viewerControlsPanel) {
            _viewerControlsPanel->toggleSegmentationComposite();
        }
    });

    // Toggle direction hints overlay (Ctrl+T)
    fDirectionHintsShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::DirectionHints), this);
    fDirectionHintsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDirectionHintsShortcut, &QShortcut::activated, [this]() {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool current = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        bool next = !current;
        settings.setValue(viewer::SHOW_DIRECTION_HINTS, next ? "1" : "0");
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([next](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setShowDirectionHints(next);
                }
            });
        }
    });

    // Toggle surface normals visualization (Ctrl+N)
    fSurfaceNormalsShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::SurfaceNormals), this);
    fSurfaceNormalsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fSurfaceNormalsShortcut, &QShortcut::activated, [this]() {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool current = settings.value(viewer::SHOW_SURFACE_NORMALS, viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        bool next = !current;
        settings.setValue(viewer::SHOW_SURFACE_NORMALS, next ? "1" : "0");
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([next](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setShowSurfaceNormals(next);
                }
            });
        }
        statusBar()->showMessage(next ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
    });

    fAxisAlignedSlicesShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::AxisAlignedSlices), this);
    fAxisAlignedSlicesShortcut->setContext(Qt::ApplicationShortcut);
    connect(fAxisAlignedSlicesShortcut, &QShortcut::activated, [this]() {
        if (chkAxisAlignedSlices) {
            chkAxisAlignedSlices->toggle();
        }
    });

    // Raw points overlay shortcut (P key)
    auto* rawPointsShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::RawPointsOverlay), this);
    rawPointsShortcut->setContext(Qt::ApplicationShortcut);
    connect(rawPointsShortcut, &QShortcut::activated, [this]() {
        if (_rawPointsOverlay) {
            bool newEnabled = !_rawPointsOverlay->isEnabled();
            _rawPointsOverlay->setEnabled(newEnabled);
            statusBar()->showMessage(
                newEnabled ? tr("Raw points overlay enabled") : tr("Raw points overlay disabled"),
                2000);
        }
    });

    // Zoom shortcuts (Shift+= for zoom in, Shift+- for zoom out)
    // Use 15% steps for smooth, proportional zooming - only affects active viewer
    constexpr float ZOOM_FACTOR = 1.15f;
    fZoomInShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ZoomIn), this);
    fZoomInShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomInShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustZoomByFactor(ZOOM_FACTOR);
        }
    });

    fZoomOutShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ZoomOut), this);
    fZoomOutShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomOutShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustZoomByFactor(1.0f / ZOOM_FACTOR);
        }
    });

    // Reset view shortcut (m to fit surface in view and reset all offsets)
    fResetViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ResetView), this);
    fResetViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fResetViewShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->resetSurfaceOffsets();
            viewer->fitSurfaceInView();
            viewer->renderVisible(true);
        }
    });

    // Z offset: Ctrl+. = +Z (further/deeper), Ctrl+, = -Z (closer)
    fWorldOffsetZPosShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::WorldOffsetZPos), this);
    fWorldOffsetZPosShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZPosShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustSurfaceOffset(1.0f);
        }
    });

    fWorldOffsetZNegShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::WorldOffsetZNeg), this);
    fWorldOffsetZNegShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZNegShortcut, &QShortcut::activated, [this]() {
        if (auto* viewer = activeBaseViewer()) {
            viewer->adjustSurfaceOffset(-1.0f);
        }
    });

    // Segment cycling shortcuts (] for next, [ for previous)
    fCycleNextSegmentShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CycleNextSegment), this);
    fCycleNextSegmentShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCycleNextSegmentShortcut, &QShortcut::activated, [this]() {
        if (!_surfacePanel) {
            return;
        }

        const bool preserveEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        bool previousIgnore = false;
        if (preserveEditing && _segmentationModule) {
            previousIgnore = _segmentationModule->ignoreSegSurfaceChange();
            _segmentationModule->setIgnoreSegSurfaceChange(true);
        }

        _surfacePanel->cycleToNextVisibleSegment();

        if (preserveEditing && _segmentationModule) {
            _segmentationModule->setIgnoreSegSurfaceChange(previousIgnore);
        }
    });

    fCyclePrevSegmentShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CyclePrevSegment), this);
    fCyclePrevSegmentShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCyclePrevSegmentShortcut, &QShortcut::activated, [this]() {
        if (!_surfacePanel) {
            return;
        }

        const bool preserveEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        bool previousIgnore = false;
        if (preserveEditing && _segmentationModule) {
            previousIgnore = _segmentationModule->ignoreSegSurfaceChange();
            _segmentationModule->setIgnoreSegSurfaceChange(true);
        }

        _surfacePanel->cycleToPreviousVisibleSegment();

        if (preserveEditing && _segmentationModule) {
            _segmentationModule->setIgnoreSegSurfaceChange(previousIgnore);
        }
    });

    // Focused view toggle (Shift+Ctrl+F) - hides dock widgets, keeps all viewers
    fFocusedViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::FocusedView), this);
    fFocusedViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fFocusedViewShortcut, &QShortcut::activated, this, &CWindow::toggleFocusedView);

    connect(_surfacePanel.get(), &SurfacePanelController::moveToPathsRequested,
            _segmentationCommandHandler.get(), &SegmentationCommandHandler::onMoveSegmentToPaths);
    connect(_surfacePanel.get(), &SurfacePanelController::renameSurfaceRequested,
            _segmentationCommandHandler.get(), &SegmentationCommandHandler::onRenameSurface);
    connect(_surfacePanel.get(), &SurfacePanelController::copySurfaceRequested,
            _segmentationCommandHandler.get(), &SegmentationCommandHandler::onCopySurfaceRequested);
}

// Destructor
CWindow::~CWindow()
{
    if (qApp) {
        qApp->removeEventFilter(this);
    }

    // Backstop in case ~CWindow is reached without closeEvent firing (e.g.
    // if the app is torn down programmatically). Same rationale as the
    // closeEvent hook — skip SurfacePatchIndex removal during teardown.
    if (_viewerManager) {
        _viewerManager->beginShutdown();
    }
    if (_fileWatcher) {
        _fileWatcher->stopWatching();
    }
    setStatusBar(nullptr);

    CloseVolume();
}

VolumeViewerBase *CWindow::newConnectedViewer(std::string surfaceName, QString title, QMdiArea *mdiArea)
{
    if (!_viewerManager) {
        return nullptr;
    }

    VolumeViewerBase* viewer = _viewerManager->createViewer(surfaceName, title, mdiArea);
    if (!viewer) {
        return nullptr;
    }

    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
        configureChunkedViewerConnections(chunkedViewer);
    }
    return viewer;
}

void CWindow::configureChunkedViewerConnections(CChunkedVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    connect(_state, &CState::volumeChanged, viewer, &CChunkedVolumeViewer::OnVolumeChanged, Qt::UniqueConnection);
    connect(_state, &CState::volumeClosing, viewer, &CChunkedVolumeViewer::onVolumeClosing, Qt::UniqueConnection);
    connect(viewer, &CChunkedVolumeViewer::sendVolumeClicked, this, &CWindow::onVolumeClicked, Qt::UniqueConnection);

    if (auto* graphicsView = viewer->graphicsView()) {
        connect(graphicsView, &CVolumeViewerView::sendMousePress,
                viewer, &CChunkedVolumeViewer::onMousePress, Qt::UniqueConnection);
        connect(graphicsView, &CVolumeViewerView::sendMouseMove,
                viewer, &CChunkedVolumeViewer::onMouseMove, Qt::UniqueConnection);
        connect(graphicsView, &CVolumeViewerView::sendMouseRelease,
                viewer, &CChunkedVolumeViewer::onMouseRelease, Qt::UniqueConnection);
    }

    if (_seedingWidget && !viewer->property("vc_seeding_bound").toBool()) {
        connect(_seedingWidget, &SeedingWidget::sendPathsChanged,
                viewer, &CChunkedVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendMousePressVolume,
                _seedingWidget, &SeedingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendMouseMoveVolume,
                _seedingWidget, &SeedingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendMouseReleaseVolume,
                _seedingWidget, &SeedingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendZSliceChanged,
                _seedingWidget, &SeedingWidget::updateCurrentZSlice, Qt::UniqueConnection);
        viewer->setProperty("vc_seeding_bound", true);
    }

    if (_point_collection_widget && !viewer->property("vc_points_bound").toBool()) {
        connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
                viewer, &CChunkedVolumeViewer::onCollectionSelected, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendCollectionSelected,
                _point_collection_widget, &CPointCollectionWidget::selectCollection, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelected,
                viewer, &CChunkedVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointSelected,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointClicked,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        viewer->setProperty("vc_points_bound", true);
    }

    const std::string& surfName = viewer->surfName();
    if ((surfName == "seg xz" || surfName == "seg yz") && !viewer->property("vc_axisaligned_bound").toBool()) {
        if (auto* graphicsView = viewer->graphicsView()) {
            graphicsView->setMiddleButtonPanEnabled(!_axisAlignedSliceController->isEnabled());
        }

        connect(viewer, &CChunkedVolumeViewer::sendMousePressVolume,
                this, [this, viewer](cv::Vec3f volLoc, cv::Vec3f /*normal*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMousePress(viewer, volLoc, button, modifiers);
                });

        connect(viewer, &CChunkedVolumeViewer::sendMouseMoveVolume,
                this, [this, viewer](cv::Vec3f volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMouseMove(viewer, volLoc, buttons, modifiers);
                });

        connect(viewer, &CChunkedVolumeViewer::sendMouseReleaseVolume,
                this, [this, viewer](cv::Vec3f /*volLoc*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMouseRelease(viewer, button, modifiers);
                });

        viewer->setProperty("vc_axisaligned_bound", true);
    }
}

CChunkedVolumeViewer* CWindow::segmentationViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            return qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
        }
    }
    return nullptr;
}

VolumeViewerBase* CWindow::segmentationBaseViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            return viewer;
        }
    }
    return nullptr;
}

VolumeViewerBase* CWindow::activeBaseViewer() const
{
    if (!mdiArea) {
        return nullptr;
    }
    auto* subWindow = mdiArea->activeSubWindow();
    if (!subWindow) {
        return nullptr;
    }
    return baseViewerFromWidget(subWindow->widget());
}

void CWindow::clearSurfaceSelection()
{
    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->clearPreview(true);
    }
    _state->clearActiveSurface();

    if (_surfacePanel) {
        _surfacePanel->resetTagUi();
    }

    if (auto* viewer = segmentationViewer()) {
        viewer->setWindowTitle(tr("Surface"));
    }

    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clearSelection();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    const bool hadVolume = static_cast<bool>(_state->currentVolume());
    POI* existingFocusPoi = _state ? _state->poi("focus") : nullptr;

    // CState handles cache budget and volume ID resolution, and emits volumeChanged
    _state->setCurrentVolume(newvol);

    const bool growthVolumeValid = _state->hasVpkg() && !_state->segmentationGrowthVolumeId().empty() &&
                                   _state->vpkg()->hasVolume(_state->segmentationGrowthVolumeId());
    if (!growthVolumeValid) {
        _state->setSegmentationGrowthVolumeId(_state->currentVolumeId());
        if (_segmentationWidget) {
            _segmentationWidget->setActiveVolume(QString::fromStdString(_state->currentVolumeId()));
        }
    }

    updateNormalGridAvailability();

    if (_state->currentVolume() && _state) {
        auto [w, h, d] = _state->currentVolume()->shapeXyz();
        float x0 = 0, y0 = 0, z0 = 0;
        float x1 = static_cast<float>(w - 1), y1 = static_cast<float>(h - 1), z1 = static_cast<float>(d - 1);

        POI* poi = existingFocusPoi;
        const bool createdPoi = (poi == nullptr);
        if (!poi) {
            poi = new POI;
            poi->n = cv::Vec3f(0, 0, 1);
        }

        if (createdPoi || !hadVolume) {
            poi->p = cv::Vec3f((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, (z0 + z1) * 0.5f);
        } else {
            poi->p[0] = std::clamp(poi->p[0], x0, x1);
            poi->p[1] = std::clamp(poi->p[1], y0, y1);
            poi->p[2] = std::clamp(poi->p[2], z0, z1);
        }

        _state->setPOI("focus", poi);
    }

    _axisAlignedSliceController->applyOrientation(_state ? _state->surface("segmentation").get() : nullptr);
}

bool CWindow::attachVolumeToCurrentPackage(const std::shared_ptr<Volume>& volume,
                                           const QString& preferredVolumeId)
{
    if (!_state || !_state->vpkg() || !volume) {
        return false;
    }

    if (!_state->vpkg()->addVolume(volume)) {
        return false;
    }

    const bool needSurfaceLoad = _surfacePanel && !_surfacePanel->hasSurfaces();
    refreshCurrentVolumePackageUi(preferredVolumeId.isEmpty()
                                      ? QString::fromStdString(volume->id())
                                      : preferredVolumeId,
                                  needSurfaceLoad);
    UpdateView();
    return true;
}

void CWindow::refreshCurrentVolumePackageUi(const QString& preferredVolumeId,
                                            bool reloadSurfaces)
{
    if (!_state || !_state->vpkg()) {
        return;
    }

    if (_segmentationWidget) {
        _segmentationWidget->setVolumePackagePath(_state->vpkgPath());
    }

    updateNormalGridAvailability();

    refreshVolumeSelectionUi(preferredVolumeId);
    if (!_state->vpkg()->hasVolumes()) {
        Logger()->info("Opened volpkg '{}' with no volumes", _state->vpkgPath().toStdString());
        statusBar()->showMessage(tr("Opened volume package with no volumes."), 5000);
    }

    if (_volumeOverlay) {
        _volumeOverlay->setVolumePkg(_state->vpkg(), _state->vpkgPath());
    }

    {
        const QSignalBlocker blocker{cmbSegmentationDir};
        cmbSegmentationDir->clear();

        auto availableDirs = _state->vpkg()->getAvailableSegmentationDirectories();
        for (const auto& dirName : availableDirs) {
            cmbSegmentationDir->addItem(QString::fromStdString(dirName));
        }

        int currentIndex = cmbSegmentationDir->findText(
            QString::fromStdString(_state->vpkg()->getSegmentationDirectory()));
        if (currentIndex >= 0) {
            cmbSegmentationDir->setCurrentIndex(currentIndex);
        }
    }

    if (_surfacePanel) {
        _surfacePanel->setVolumePkg(_state->vpkg());
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
        if (reloadSurfaces) {
            _surfacePanel->loadSurfaces(false);
            _surfacePanel->refreshPointSetFilterOptions();
        }
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

void CWindow::updateNormalGridAvailability()
{
    QString checkedPath;
    const QString path = normalGridDirectoryForVolumePkg(_state->vpkg(), &checkedPath);
    const bool available = !path.isEmpty();

    _normalGridAvailable = available;
    _normalGridPath = path;

    if (_segmentationWidget) {
        _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
        _segmentationWidget->setNormalGridPath(_normalGridPath);
        QString hint;
        if (_normalGridAvailable) {
        } else if (!checkedPath.isEmpty()) {
            hint = tr("Checked: %1").arg(checkedPath);
        } else {
            hint = tr("No volume package loaded.");
        }
        _segmentationWidget->setNormalGridPathHint(hint);

        QString normal3dHint;
        const QStringList normal3d = normal3dZarrCandidatesForVolumePkg(_state->vpkg(), &normal3dHint);
        _segmentationWidget->setNormal3dZarrCandidates(normal3d, normal3dHint);
    }
}

void CWindow::toggleVolumeOverlayVisibility()
{
    if (_volumeOverlay) {
        _volumeOverlay->toggleVisibility();
    }
}

void CWindow::toggleFocusedView()
{
    if (_focusedViewActive) {
        for (const auto& [dock, state] : _savedDockStates) {
            if (dock) {
                dock->setVisible(state.visible);
            }
        }
        for (const auto& [dock, state] : _savedDockStates) {
            if (dock && state.wasRaised) {
                dock->raise();
            }
        }
        _savedDockStates.clear();
        _focusedViewActive = false;
        statusBar()->showMessage(tr("Restored full view"), 2000);
    } else {
        _savedDockStates.clear();
        const QList<QDockWidget*> docks = findChildren<QDockWidget*>();
        for (QDockWidget* dock : docks) {
            bool wasRaised = false;
            if (dock->isVisible() && !dock->isFloating()) {
                if (QWidget* content = dock->widget()) {
                    wasRaised = !content->visibleRegion().isEmpty();
                }
            }
            _savedDockStates[dock] = {dock->isVisible(), dock->isFloating(), wasRaised};
            dock->hide();
        }
        _focusedViewActive = true;
        statusBar()->showMessage(tr("Focused view (Shift+Ctrl+F to restore)"), 2000);
    }
}

bool CWindow::centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId)
{
    if (!_state) {
        return false;
    }

    POI* focus = _state->poi("focus");
    if (!focus) {
        focus = new POI;
    }

    focus->p = position;
    if (cv::norm(normal) > 0.0) {
        focus->n = normal;
    }
    if (!sourceId.empty()) {
        focus->surfaceId = sourceId;
    } else if (focus->surfaceId.empty()) {
        focus->surfaceId = "segmentation";
    }

    focus->suppressTransientPlaneIntersections = true;
    _state->setPOI("focus", focus);
    recenterSegmentationViewerNear(position);

    // Get surface for orientation - look up by ID
    Surface* orientationSource = _state->surfaceRaw(focus->surfaceId);
    if (!orientationSource) {
        orientationSource = _state->surfaceRaw("segmentation");
    }
    _axisAlignedSliceController->applyOrientation(orientationSource);

    return true;
}

void CWindow::recenterPlaneViewersOn(const cv::Vec3f& position)
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachBaseViewer([&position](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }

        const std::string name = viewer->surfName();
        if (name == "xy plane" || name == "seg xz" || name == "seg yz") {
            centerViewerOnVolumePointForNavigation(viewer, position);
        }
    });
}

void CWindow::recenterSegmentationViewerNear(const cv::Vec3f& position)
{
    static constexpr float kMaxDistanceVoxels = 100.0f;

    if (!_viewerManager) {
        return;
    }

    auto* viewer = segmentationViewer();
    if (!viewer) {
        return;
    }

    auto activeSurface = _segmentationModule ? _segmentationModule->activeBaseSurfaceShared() : nullptr;
    if (!activeSurface) {
        activeSurface = std::dynamic_pointer_cast<QuadSurface>(_state ? _state->surface("segmentation") : nullptr);
    }
    if (!activeSurface) {
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || !patchIndex->containsSurface(activeSurface)) {
        return;
    }

    SurfacePatchIndex::PointQuery query;
    query.worldPoint = position;
    query.tolerance = kMaxDistanceVoxels;
    query.surfaces.only = activeSurface;
    auto hit = patchIndex->locate(query);
    if (hit && hit->distance <= kMaxDistanceVoxels) {
        const cv::Vec3f loc = activeSurface->loc(hit->ptr);
        centerViewerOnSurfacePointForNavigation(viewer, {loc[0], loc[1]});
    }
}

bool CWindow::recenterViewersOnCurrentFocus()
{
    if (!_state || !_viewerManager) {
        return false;
    }

    POI* focus = _state->poi("focus");
    if (!focus) {
        return false;
    }

    const cv::Vec3f position = focus->p;
    _viewerManager->forEachBaseViewer([&position](VolumeViewerBase* viewer) {
        if (viewer) {
            centerViewerOnVolumePointForNavigation(viewer, position);
        }
    });

    return true;
}

bool CWindow::centerFocusOnCursor()
{
    if (!_state || !mdiArea) {
        return false;
    }

    const QPoint globalPos = QCursor::pos();
    auto tryCenterFromViewer = [&](VolumeViewerBase* viewer) -> bool {
        if (!viewer) {
            return false;
        }

        auto* viewerObject = viewer->asQObject();
        auto* viewerWidget = qobject_cast<QWidget*>(viewerObject);
        if (viewerWidget && !viewerWidget->isVisible()) {
            return false;
        }

        auto* gv = viewer->graphicsView();
        auto* viewport = gv ? gv->viewport() : nullptr;
        if (!viewport) {
            return false;
        }

        const QPoint viewportPos = viewport->mapFromGlobal(globalPos);
        if (!viewport->rect().contains(viewportPos)) {
            return false;
        }

        const QPointF scenePos = gv->mapToScene(viewportPos);
        cv::Vec3f p = viewer->sceneToVolume(scenePos);
        cv::Vec3f n(0, 0, 1);
        if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            n = plane->normal(cv::Vec3f(0, 0, 0), {});
        }

        return centerFocusAt(p, n, viewer->surfName());
    };

    // Prefer the viewer actually under the mouse cursor. With tiled MDI
    // windows, the active subwindow can lag behind the hovered viewer, which
    // makes the focus jump use the wrong scene transform.
    if (QWidget* hoveredWidget = QApplication::widgetAt(globalPos)) {
        for (QWidget* widget = hoveredWidget; widget; widget = widget->parentWidget()) {
            if (auto* viewer = baseViewerFromWidget(widget)) {
                if (tryCenterFromViewer(viewer)) {
                    return true;
                }
                break;
            }
        }
    }

    if (_viewerManager) {
        for (auto* viewer : _viewerManager->baseViewers()) {
            if (tryCenterFromViewer(viewer)) {
                return true;
            }
        }
    }

    // Fall back to the active viewer if the cursor isn't currently over any
    // tiled viewport.
    if (auto* subWindow = mdiArea->activeSubWindow()) {
        if (auto* viewer = baseViewerFromWidget(subWindow->widget())) {
            if (tryCenterFromViewer(viewer)) {
                return true;
            }
        }
    }

    // Fallback to stored cursor POI if no active viewer or cursor is outside
    POI* cursor = _state->poi("cursor");
    if (!cursor) {
        return false;
    }

    return centerFocusAt(cursor->p, cursor->n, cursor->surfaceId);
}

void CWindow::setSegmentationCursorMirroring(bool enabled)
{
    if (_mirrorCursorToSegmentation == enabled) {
        return;
    }

    _mirrorCursorToSegmentation = enabled;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION, enabled ? "1" : "0");

    if (_viewerManager) {
        _viewerManager->setSegmentationCursorMirroring(enabled);
    }

    if (statusBar()) {
        statusBar()->showMessage(enabled ? tr("Mirroring cursor to Surface view enabled")
                                         : tr("Mirroring cursor to Surface view disabled"),
                                  2000);
    }
}

// Create widgets
void CWindow::CreateWidgets(void)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    // add volume viewer
    auto aWidgetLayout = new QVBoxLayout;
    ui.tabSegment->setLayout(aWidgetLayout);

    mdiArea = new QMdiArea(ui.tabSegment);
    aWidgetLayout->addWidget(mdiArea);

    // Ensure the viewer's graphics view gets focus when subwindow is activated
    connect(mdiArea, &QMdiArea::subWindowActivated, [](QMdiSubWindow* subWindow) {
        if (subWindow) {
            if (auto* viewer = dynamic_cast<VolumeViewerBase*>(subWindow->widget())) {
                if (auto* graphicsView = viewer->graphicsView()) {
                    graphicsView->setFocus();
                }
            }
        }
    });

    {
        newConnectedViewer("seg xz", tr("Segmentation XZ"), mdiArea)->setIntersects({"segmentation"});
        newConnectedViewer("seg yz", tr("Segmentation YZ"), mdiArea)->setIntersects({"segmentation"});
        newConnectedViewer("xy plane", tr("XY / Slices"), mdiArea)->setIntersects({"segmentation"});
        newConnectedViewer("segmentation", tr("Surface"), mdiArea)->setIntersects({"seg xz","seg yz"});
    }
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
        _state,
        _viewerManager.get(),
        [this]() { return segmentationViewer(); },
        std::function<void()>{},
        this);
    if (_segmentationGrower) {
        _segmentationGrower->setSurfacePanel(_surfacePanel.get());
    }
    connect(_surfacePanel.get(), &SurfacePanelController::surfacesLoaded, this, [this]() {
        emit _state->surfacesLoaded();
        // Update surface overlay dropdown when surfaces are loaded
        updateSurfaceOverlayDropdown();
        if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
    });
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceSelectionCleared, this, [this]() {
        clearSurfaceSelection();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::filtersApplied, this, [this](int filterCount) {
        UpdateVolpkgLabel(filterCount);
    });
    connect(_surfacePanel.get(), &SurfacePanelController::copySegmentPathRequested,
            this, [this](const QString& segmentId) {
                if (!_state->vpkg()) {
                    return;
                }
                auto surf = _state->vpkg()->getSurface(segmentId.toStdString());
                if (!surf) {
                    return;
                }
                const QString path = absoluteSegmentPathForClipboard(surf->path, _state->vpkg());
                QApplication::clipboard()->setText(path);
                statusBar()->showMessage(tr("Copied segment path to clipboard: %1").arg(path), 3000);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::renderSegmentRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onRenderSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::growSegmentRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onGrowSegmentFromSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::neighborCopyRequested,
            this, [this](const QString& segmentId, bool copyOut) {
                _segmentationCommandHandler->onNeighborCopyRequested(segmentId, copyOut);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::resumeLocalGrowPatchRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onResumeLocalGrowPatchRequested(segmentId);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::reloadFromBackupRequested,
            this, [this](const QString& segmentId, int backupIndex) {
                _segmentationCommandHandler->onReloadFromBackup(segmentId, backupIndex);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::convertToObjRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onConvertToObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::mergeTifxyzRequested,
            this, [this](const QStringList& segmentIds) {
                _segmentationCommandHandler->onMergeTifxyz(segmentIds);
            });
    // Note: the Actions -> Merge tifxyz... menu wiring lives in the
    // constructor after _menuController is initialized -- _menuController
    // is null inside CreateWidgets().
    connect(_surfacePanel.get(), &SurfacePanelController::visLasagnaObjRequested,
            this, [this](const QString& segmentId) {
                onVisLasagnaObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::cropBoundsRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onCropSurfaceToValidRegion(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::flipURequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onFlipSurface(segmentId.toStdString(), true);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::flipVRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onFlipSurface(segmentId.toStdString(), false);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::rotateSurfaceRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onRotateSurface(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::focusSurfaceRequested,
            this, [this](const QString& segmentId) {
                if (!_state || !_state->vpkg()) return;
                auto surf = _state->vpkg()->getSurface(segmentId.toStdString());
                auto* quad = dynamic_cast<QuadSurface*>(surf.get());
                if (!quad) return;
                quad->ensureLoaded();
                const cv::Vec3f worldCenter = quad->coord({0, 0, 0}, {0, 0, 0});
                if (!std::isfinite(worldCenter[0]) || worldCenter[0] < 0.0f) return;
                cv::Vec3f normal = quad->normal({0, 0, 0}, {0, 0, 0});
                if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
                    normal = cv::Vec3f(0, 0, 1);
                }
                if (auto vol = _state->currentVolume()) {
                    auto [w, h, d] = vol->shapeXyz();
                    cv::Vec3f clamped = worldCenter;
                    clamped[0] = std::clamp(clamped[0], 0.0f, static_cast<float>(w - 1));
                    clamped[1] = std::clamp(clamped[1], 0.0f, static_cast<float>(h - 1));
                    clamped[2] = std::clamp(clamped[2], 0.0f, static_cast<float>(d - 1));
                    POI* poi = new POI;
                    poi->p = clamped;
                    poi->n = normal;
                    poi->surfaceId = segmentId.toStdString();
                    _state->setPOI("focus", poi);
                }
            });
    connect(_surfacePanel.get(), &SurfacePanelController::alphaCompRefineRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onAlphaCompRefine(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::slimFlattenRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onSlimFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::abfFlattenRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onABFFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::exportTifxyzChunksRequested,
        this, [this](const QString& segmentId) {
            _segmentationCommandHandler->onExportWidthChunks(segmentId.toStdString());
        });
    connect(_surfacePanel.get(), &SurfacePanelController::rasterizeSegmentsRequested,
        this, [this](const QStringList& segmentIds) {
            _segmentationCommandHandler->onRasterizeSegments(segmentIds);
        });
    connect(_surfacePanel.get(), &SurfacePanelController::addIgnoreLabelRequested,
        this, [this]() {
            _segmentationCommandHandler->onAddIgnoreLabel();
        });
    connect(_surfacePanel.get(), &SurfacePanelController::recalcAreaRequested,
            this, [this](const QStringList& segmentIds) {
                if (segmentIds.isEmpty()) return;
                std::vector<std::string> ids;
                ids.reserve(segmentIds.size());
                for (const auto& id : segmentIds) {
                    ids.push_back(id.toStdString());
                }
                auto results = SurfaceAreaCalculator::calculateAreas(_state->vpkg(), _state->currentVolume(), ids);
                int okCount = 0, failCount = 0;
                QStringList skippedIds;
                for (const auto& r : results) {
                    if (r.success) {
                        ++okCount;
                        // Update tree widget
                        QTreeWidgetItemIterator it(treeWidgetSurfaces);
                        while (*it) {
                            if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == r.segmentId) {
                                (*it)->setText(2, QString::number(r.areaCm2, 'f', 3));
                                break;
                            }
                            ++it;
                        }
                    } else {
                        ++failCount;
                        skippedIds << QString::fromStdString(r.segmentId + " (" + r.errorReason + ")");
                    }
                }
                if (okCount > 0) {
                    statusBar()->showMessage(
                        tr("Recalculated area for %1 segment(s).").arg(okCount), 5000);
                }
                if (failCount > 0) {
                    QMessageBox::warning(this, tr("Area Recalculation"),
                        tr("Updated: %1\nSkipped: %2\n\n%3").arg(okCount).arg(failCount).arg(skippedIds.join("\n")));
                }
            });
    connect(_surfacePanel.get(), &SurfacePanelController::statusMessageRequested,
            this, [this](const QString& message, int timeoutMs) {
                statusBar()->showMessage(message, timeoutMs);
            });

    const auto attachScrollAreaToDock = [](QDockWidget* dock, QWidget* content, const QString& objectName) {
        if (!dock || !content) {
            return;
        }

        // Delete any existing widget from the .ui file to prevent ghosting
        if (auto* oldWidget = dock->widget()) {
            delete oldWidget;
        }

        auto* container = new QWidget(dock);
        container->setObjectName(objectName);
        auto* layout = new QVBoxLayout(container);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        layout->addWidget(content);
        layout->addStretch(1);

        auto* scrollArea = new QScrollArea(dock);
        scrollArea->setFrameShape(QFrame::NoFrame);
        scrollArea->setWidgetResizable(true);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setWidget(container);

        dock->setWidget(scrollArea);
    };


    // Create Segmentation widget
    _segmentationWidget = new SegmentationWidget();
    _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
    _segmentationWidget->setNormalGridPath(_normalGridPath);
    const QString initialHint = _normalGridAvailable
        ? tr("Normal grids directory found.")
        : tr("No volume package loaded.");
    _segmentationWidget->setNormalGridPathHint(initialHint);
    attachScrollAreaToDock(ui.dockWidgetSegmentation, _segmentationWidget, QStringLiteral("dockWidgetSegmentationContent"));

    // Create Lasagna dock from the panel already constructed by SegmentationWidget
    {
        auto* panel = _segmentationWidget->lasagnaPanel();
        panel->setVisible(true);
        _lasagnaDock = new QDockWidget(tr("Lasagna Model"), this);
        _lasagnaDock->setObjectName(QStringLiteral("dockWidgetLasagna"));
        attachScrollAreaToDock(_lasagnaDock, panel, QStringLiteral("dockWidgetLasagnaContent"));
        addDockWidget(Qt::RightDockWidgetArea, _lasagnaDock);
    }

    _segmentationEdit = std::make_unique<SegmentationEditManager>(this);
    _segmentationEdit->setViewerManager(_viewerManager.get());
    _segmentationOverlay = std::make_unique<SegmentationOverlayController>(_state, this);
    _segmentationOverlay->setEditManager(_segmentationEdit.get());
    _segmentationOverlay->setViewerManager(_viewerManager.get());
    _surfaceRotationOverlay = std::make_unique<SurfaceRotationOverlayController>(_state, this);
    _surfaceRotationOverlay->setViewerManager(_viewerManager.get());

    _segmentationModule = std::make_unique<SegmentationModule>(
        _segmentationWidget,
        _segmentationEdit.get(),
        _segmentationOverlay.get(),
        _viewerManager.get(),
        _state,
        _state->pointCollection(),
        _segmentationWidget->isEditingEnabled(),
        this);

    if (_segmentationModule && _planeSlicingOverlay) {
        QPointer<PlaneSlicingOverlayController> overlayPtr(_planeSlicingOverlay.get());
        _segmentationModule->setRotationHandleHitTester(
            [overlayPtr](VolumeViewerBase* viewer, const cv::Vec3f& worldPos) {
                if (!overlayPtr) {
                    return false;
                }
                return overlayPtr->isVolumePointNearRotationHandle(viewer, worldPos, 1.5);
            });
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationOverlay(_segmentationOverlay.get());
    }

    // Wire annotate mode: module -> header row (via SegmentationWidget)
    // NOTE: connections involving _point_collection_widget are below, after it is created.
    connect(_segmentationModule.get(), &SegmentationModule::annotateModeChanged,
            _segmentationWidget, &SegmentationWidget::setAnnotateChecked);
    connect(_segmentationModule.get(), &SegmentationModule::annotationPointFocused,
            this, &CWindow::onPointDoubleClicked);

    connect(_segmentationModule.get(), &SegmentationModule::editingEnabledChanged,
            this, &CWindow::onSegmentationEditingModeChanged);
    connect(_segmentationModule.get(), &SegmentationModule::statusMessageRequested,
            this, &CWindow::onShowStatusMessage);
    connect(_segmentationModule.get(), &SegmentationModule::stopToolsRequested,
            this, &CWindow::onSegmentationStopToolsRequested);
    connect(_segmentationModule.get(), &SegmentationModule::growthInProgressChanged,
            this, &CWindow::onSegmentationGrowthStatusChanged);
    connect(_segmentationModule.get(), &SegmentationModule::focusPoiRequested,
            this, [this](const cv::Vec3f& position, QuadSurface* base) {
                Q_UNUSED(position);
                _axisAlignedSliceController->applyOrientation(base);
            });
    connect(_segmentationModule.get(), &SegmentationModule::growSurfaceRequested,
            this, &CWindow::onGrowSegmentationSurface);
    connect(_segmentationModule.get(), &SegmentationModule::approvalMaskSaved,
            _fileWatcher.get(), &FileWatcherService::markSegmentRecentlyEdited);

    SegmentationGrower::Context growerContext{
        _segmentationModule.get(),
        _segmentationWidget,
        _state,
        _viewerManager.get()
    };
    SegmentationGrower::UiCallbacks growerCallbacks{
        [this](const QString& text, int timeout) {
            if (statusBar()) {
                statusBar()->showMessage(text, timeout);
            }
        },
        [this](QuadSurface* surface) {
            _axisAlignedSliceController->applyOrientation(surface);
        }
    };
    _segmentationGrower = std::make_unique<SegmentationGrower>(growerContext, growerCallbacks, this);

    _segmentationCommandHandler = std::make_unique<SegmentationCommandHandler>(this, _state, this);
    _segmentationCommandHandler->setCmdRunner(_cmdRunner);
    _segmentationCommandHandler->setSurfacePanel(_surfacePanel.get());
    _segmentationCommandHandler->setSegmentationGrower(_segmentationGrower.get());
    initializeCommandLineRunner();
    _segmentationCommandHandler->setIsEditingCheck([this]() -> bool {
        return _segmentationModule && _segmentationModule->isEditingApprovalMask();
    });
    _segmentationCommandHandler->setClearSelectionCallback([this]() {
        clearSurfaceSelection();
    });
    _segmentationCommandHandler->setRestoreSelectionCallback([this](const std::string& id) {
        if (treeWidgetSurfaces) {
            QTreeWidgetItemIterator it(treeWidgetSurfaces);
            while (*it) {
                if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == id) {
                    treeWidgetSurfaces->setCurrentItem(*it);
                    break;
                }
                ++it;
            }
        }
    });
    _segmentationCommandHandler->setNormal3dZarrPathGetter([this]() -> QString {
        return _segmentationWidget ? _segmentationWidget->normal3dZarrPath() : QString();
    });
    connect(_segmentationCommandHandler.get(), &SegmentationCommandHandler::statusMessage,
            this, &CWindow::onShowStatusMessage);

    _fileWatcher->setSurfacePanel(_surfacePanel.get());
    _fileWatcher->setSegmentationModule(_segmentationModule.get());
    _fileWatcher->setTreeWidget(treeWidgetSurfaces);

    connect(_segmentationWidget, &SegmentationWidget::copyWithNtRequested,
            this, &CWindow::onCopyWithNtRequested);
    connect(_segmentationWidget, &SegmentationWidget::volumeSelectionChanged, this, [this](const QString& volumeId) {
        if (!_state->vpkg()) {
            statusBar()->showMessage(tr("No volume package loaded."), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!_state->segmentationGrowthVolumeId().empty()
                                                                   ? _state->segmentationGrowthVolumeId()
                                                                   : _state->currentVolumeId());
                _segmentationWidget->setActiveVolume(fallbackId);
            }
            return;
        }

        const std::string requestedId = volumeId.toStdString();
        try {
            auto vol = _state->vpkg()->volume(requestedId);
            _state->setSegmentationGrowthVolumeId(requestedId);
            // Set volume zarr path for neural tracing
            if (_segmentationWidget && vol) {
                _segmentationWidget->setVolumeZarrPath(QString::fromStdString(vol->path().string()));
            }
            statusBar()->showMessage(tr("Using volume '%1' for surface growth.").arg(volumeId), 2500);
        } catch (const std::out_of_range&) {
            statusBar()->showMessage(tr("Volume '%1' not found in this package.").arg(volumeId), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!_state->currentVolumeId().empty()
                                                                   ? _state->currentVolumeId()
                                                                   : std::string{});
                _segmentationWidget->setActiveVolume(fallbackId);
                _state->setSegmentationGrowthVolumeId(_state->currentVolumeId());
            }
        }
    });

    // -- Lasagna connections --
    connect(_segmentationWidget, &SegmentationWidget::seedFromFocusRequested, this, [this]() {
        POI* focus = _state ? _state->poi("focus") : nullptr;
        if (focus)
            _segmentationWidget->setSeedFromFocus(
                static_cast<int>(focus->p[0]),
                static_cast<int>(focus->p[1]),
                static_cast<int>(focus->p[2]));
    });

    connect(_segmentationWidget, &SegmentationWidget::lasagnaOptimizeRequested, this, [this]() {
        if (auto* panel = _segmentationWidget->lasagnaPanel()) {
            panel->startOptimization(_state, statusBar());
        }
    });

    connect(_segmentationWidget, &SegmentationWidget::lasagnaStopRequested, this, [this]() {
        LasagnaServiceManager::instance().stopOptimization();
        statusBar()->showMessage(tr("Lasagna optimization stop requested."), 3000);
    });

    // Auto-reload segments when fit optimization finishes
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::optimizationFinished,
            this, [this](const QString& outputDir) {
        statusBar()->showMessage(
            tr("Lasagna optimization finished. Reloading segments from %1").arg(outputDir), 5000);
        if (_surfacePanel) {
            _surfacePanel->loadSurfacesIncremental();
        }
        // corr_points_results will be loaded when the new segment is activated
    });

    // Create Seeding widget
    _seedingWidget = new SeedingWidget(_state->pointCollection(), _state);
    attachScrollAreaToDock(ui.dockWidgetDistanceTransform, _seedingWidget, QStringLiteral("dockWidgetDistanceTransformContent"));

    _seedingWidget->setState(_state);
    connect(_state, &CState::volumeChanged, _seedingWidget,
            static_cast<void (SeedingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&SeedingWidget::onVolumeChanged));
    connect(_state, &CState::volumeChanged, this,
            [this](std::shared_ptr<Volume>, const std::string&) {
                if (_surfaceAffineTransforms) {
                    _surfaceAffineTransforms->refresh();
                }
            });
    connect(_seedingWidget, &SeedingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(_state, &CState::surfacesLoaded, _seedingWidget, &SeedingWidget::onSurfacesLoaded);

    // Create and add the point collection widget
    _point_collection_widget = new CPointCollectionWidget(_state->pointCollection(), this);
    _point_collection_widget->setObjectName("pointCollectionDock");
    addDockWidget(Qt::RightDockWidgetArea, _point_collection_widget);

    // Selection dock (removed per request; selection actions remain in the menu)
    if (_viewerManager) {
        for (auto* viewer : _viewerManager->baseViewers()) {
            if (viewer) {
                if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
                    configureChunkedViewerConnections(chunkedViewer);
                }
            }
        }
    }
    connect(_point_collection_widget, &CPointCollectionWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);
    connect(_point_collection_widget, &CPointCollectionWidget::convertPointToAnchorRequested, this, &CWindow::onConvertPointToAnchor);
    connect(_point_collection_widget, &CPointCollectionWidget::focusViewsRequested, this, &CWindow::onFocusViewsRequested);

    // Tab the docks - keep Segmentation, Lasagna, Seeding, and Point Collections together
    // Wire annotate mode & annotation selection: dock widget <-> segmentation module
    // (must be after _point_collection_widget creation)
    connect(_point_collection_widget, &CPointCollectionWidget::annotateToggled,
            _segmentationModule.get(), &SegmentationModule::setAnnotateMode);
    connect(_segmentationModule.get(), &SegmentationModule::annotateModeChanged,
            _point_collection_widget, &CPointCollectionWidget::setAnnotateChecked);
    connect(_segmentationModule.get(), &SegmentationModule::annotationPointSelected,
            _point_collection_widget, &CPointCollectionWidget::selectPoint);
    connect(_segmentationModule.get(), &SegmentationModule::annotationCollectionSelected,
            _point_collection_widget, &CPointCollectionWidget::selectCollection);
    connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
            _segmentationModule.get(), &SegmentationModule::setSelectedAnnotationCollection);

    // Create fiber annotation controller and dock
    _fiberController = std::make_unique<FiberAnnotationController>(
        _state, _state->pointCollection(), this);
    _fiberController->setMdiArea(mdiArea);

    _fiberWidget = new CFiberWidget(_state->pointCollection(), this);
    _fiberWidget->setObjectName("fiberDock");
    addDockWidget(Qt::RightDockWidgetArea, _fiberWidget);

    connect(_fiberWidget, &CFiberWidget::newFiberRequested,
            this, &CWindow::onNewFiberRequested);
    connect(_fiberWidget, &CFiberWidget::stepChanged,
            _fiberController.get(), &FiberAnnotationController::onStepChanged);
    connect(_fiberWidget, &CFiberWidget::invertDirectionRequested,
            _fiberController.get(), &FiberAnnotationController::invertDirection);
    connect(_fiberController.get(), &FiberAnnotationController::crosshairModeChanged,
            this, &CWindow::onFiberCrosshairModeChanged);
    connect(_fiberController.get(), &FiberAnnotationController::requestFiberViewers,
            this, &CWindow::onFiberViewersRequested);
    connect(_fiberController.get(), &FiberAnnotationController::annotationFinished,
            this, &CWindow::onFiberAnnotationFinished);

    ensureDockWidgetFeatures(_fiberWidget);
    connect(_fiberWidget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
    connect(_fiberWidget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);

    // Tab the docks - keep Segmentation, Lasagna, Seeding, Point Collections, and Fibers together
    tabifyDockWidget(ui.dockWidgetSegmentation, _lasagnaDock);
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDistanceTransform);
    tabifyDockWidget(ui.dockWidgetSegmentation, _point_collection_widget);
    tabifyDockWidget(ui.dockWidgetSegmentation, _fiberWidget);

    // Make Segmentation dock the active tab by default
    ui.dockWidgetSegmentation->raise();

    ViewerControlsPanel::UiRefs viewerControlsUi{
        .contents = ui.dockWidgetViewerControlsContents,
        .viewScrollArea = ui.scrollAreaView,
        .viewContents = ui.dockWidgetViewContents,
        .overlayScrollArea = ui.scrollAreaOverlay,
        .overlayContents = ui.dockWidgetOverlayContents,
        .compositeScrollArea = ui.scrollAreaComposite,
        .compositeContents = ui.dockWidgetCompositeContents,
        .compositeEnabled = ui.chkCompositeEnabled,
        .compositeMode = ui.cmbCompositeMode,
        .layersInFront = ui.spinLayersInFront,
        .layersBehind = ui.spinLayersBehind,
        .alphaMinLabel = ui.lblAlphaMin,
        .alphaMin = ui.spinAlphaMin,
        .alphaMaxLabel = ui.lblAlphaMax,
        .alphaMax = ui.spinAlphaMax,
        .alphaThresholdLabel = ui.lblAlphaThreshold,
        .alphaThreshold = ui.spinAlphaThreshold,
        .materialLabel = ui.lblMaterial,
        .material = ui.spinMaterial,
        .reverseDirection = ui.chkReverseDirection,
        .methodScaleLabel = ui.lblMethodScale,
        .methodScale = ui.sliderMethodScale,
        .methodScaleValue = ui.lblMethodScaleValue,
        .methodParamLabel = ui.lblMethodParam,
        .methodParam = ui.sliderMethodParam,
        .methodParamValue = ui.lblMethodParamValue,
        .blExtinctionLabel = ui.lblBLExtinction,
        .blExtinction = ui.spinBLExtinction,
        .blEmissionLabel = ui.lblBLEmission,
        .blEmission = ui.spinBLEmission,
        .blAmbientLabel = ui.lblBLAmbient,
        .blAmbient = ui.spinBLAmbient,
        .lightingEnabled = ui.chkLightingEnabled,
        .lightAzimuthLabel = ui.lblLightAzimuth,
        .lightAzimuth = ui.spinLightAzimuth,
        .lightElevationLabel = ui.lblLightElevation,
        .lightElevation = ui.spinLightElevation,
        .lightDiffuseLabel = ui.lblLightDiffuse,
        .lightDiffuse = ui.spinLightDiffuse,
        .lightAmbientLabel = ui.lblLightAmbient,
        .lightAmbient = ui.spinLightAmbient,
        .useVolumeGradients = ui.chkUseVolumeGradients,
        .shadowStepsLabel = ui.lblShadowSteps,
        .shadowSteps = ui.spinShadowSteps,
        .rakingEnabled = ui.chkRakingEnabled,
        .rakingAzimuthLabel = ui.lblRakingAzimuth,
        .rakingAzimuth = ui.spinRakingAzimuth,
        .rakingElevationLabel = ui.lblRakingElevation,
        .rakingElevation = ui.spinRakingElevation,
        .rakingStrengthLabel = ui.lblRakingStrength,
        .rakingStrength = ui.spinRakingStrength,
        .rakingDepthLabel = ui.lblRakingDepth,
        .rakingDepthScale = ui.spinRakingDepthScale,
        .preNormalizeLayers = ui.chkPreNormalizeLayers,
        .preHistEqLayers = ui.chkPreHistEqLayers,
        .preTfEnabled = ui.chkPreTfEnabled,
        .preTfX1 = ui.spinPreTfX1,
        .preTfY1 = ui.spinPreTfY1,
        .preTfKnot2Label = ui.lblPreTfKnot2,
        .preTfX2 = ui.spinPreTfX2,
        .preTfY2 = ui.spinPreTfY2,
        .postTfEnabled = ui.chkPostTfEnabled,
        .postTfX1 = ui.spinPostTfX1,
        .postTfY1 = ui.spinPostTfY1,
        .postTfKnot2Label = ui.lblPostTfKnot2,
        .postTfX2 = ui.spinPostTfX2,
        .postTfY2 = ui.spinPostTfY2,
        .dvrAmbientLabel = ui.lblDvrAmbient,
        .dvrAmbient = ui.spinDvrAmbient,
        .pbrRoughnessLabel = ui.lblPbrRoughness,
        .pbrRoughness = ui.spinPbrRoughness,
        .pbrMetallicLabel = ui.lblPbrMetallic,
        .pbrMetallic = ui.spinPbrMetallic,
        .planeCompositeXY = ui.chkPlaneCompositeXY,
        .planeCompositeXZ = ui.chkPlaneCompositeXZ,
        .planeCompositeYZ = ui.chkPlaneCompositeYZ,
        .planeLayersFront = ui.spinPlaneLayersFront,
        .planeLayersBehind = ui.spinPlaneLayersBehind,
        .renderSettingsScrollArea = ui.scrollAreaRenderSettings,
        .renderSettingsContents = ui.dockWidgetRenderSettingsContents,
        .normalVisualizationContents = ui.dockWidgetNormalVisContents,
        .showSurfaceNormals = ui.chkShowSurfaceNormals,
        .normalArrowLengthLabel = ui.labelNormalArrowLength,
        .normalArrowLengthSlider = ui.sliderNormalArrowLength,
        .normalArrowLengthValueLabel = ui.labelNormalArrowLengthValue,
        .normalMaxArrowsLabel = ui.labelNormalMaxArrows,
        .normalMaxArrowsSlider = ui.sliderNormalMaxArrows,
        .normalMaxArrowsValueLabel = ui.labelNormalMaxArrowsValue,
        .preprocessingScrollArea = ui.scrollAreaPreprocessing,
        .preprocessingContents = ui.dockWidgetPreprocessingContents,
        .isoCutoff = ui.sliderIsoCutoff,
        .isoCutoffValue = ui.lblIsoCutoffValue,
        .postprocessingScrollArea = ui.scrollAreaPostprocessing,
        .postprocessingContents = ui.dockWidgetPostprocessingContents,
        .baseColormap = ui.baseColormapSelect,
        .stretchValuesPost = ui.chkStretchValuesPost,
        .removeSmallComponents = ui.chkRemoveSmallComponents,
        .minComponentSizeLabel = ui.lblMinComponentSize,
        .minComponentSize = ui.spinMinComponentSize,
        .claheEnabled = ui.chkClaheEnabled,
        .claheClipLimitLabel = ui.lblClaheClipLimit,
        .claheClipLimit = ui.spinClaheClipLimit,
        .claheTileSizeLabel = ui.lblClaheTileSize,
        .claheTileSize = ui.spinClaheTileSize,
        .zoomInButton = ui.btnZoomIn,
        .zoomOutButton = ui.btnZoomOut,
        .sliceStepSizeSpin = ui.spinSliceStepSize,
        .volumeWindowContainer = ui.volumeWindowContainer,
        .overlayWindowContainer = ui.overlayWindowContainer,
        .intersectionOpacitySpin = ui.spinIntersectionOpacity,
        .intersectionThicknessSpin = ui.doubleSpinIntersectionThickness,
    };
    _viewerControlsPanel = std::make_unique<ViewerControlsPanel>(viewerControlsUi,
                                                                 _viewerManager.get(),
                                                                 ui.dockWidgetViewerControlsContents);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomInRequested,
            this, &CWindow::onZoomIn);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomOutRequested,
            this, &CWindow::onZoomOut);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::sliceStepSizeChanged,
            this, &CWindow::onSliceStepSizeChanged);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::statusMessageRequested,
            this, &CWindow::onShowStatusMessage);
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewControlsEnabled(!ui.grpVolManager || ui.grpVolManager->isEnabled());
    }

    _surfaceAffineTransforms = std::make_unique<SurfaceAffineTransformController>(
        SurfaceAffineTransformController::Deps{
            .state = _state,
            .viewerControlsPanel = _viewerControlsPanel.get(),
            .viewerManager = _viewerManager.get(),
            .segmentationModule = _segmentationModule.get(),
            .surfacePanel = _surfacePanel.get(),
            .axisAlignedSliceController = _axisAlignedSliceController.get(),
            .dialogParent = this,
            .showStatus = [this](const QString& text, int timeoutMs) {
                onShowStatusMessage(text, timeoutMs);
            },
        },
        this);

    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetViewerControls);
    splitDockWidget(ui.dockWidgetVolumes, ui.dockWidgetViewerControls, Qt::Vertical);

    auto hideLegacyViewerDocks = [this]() {
        for (QDockWidget* dock : { ui.dockWidgetPreprocessing,
                                   ui.dockWidgetNormalVis,
                                   ui.dockWidgetView,
                                   ui.dockWidgetOverlay,
                                   ui.dockWidgetRenderSettings,
                                   ui.dockWidgetComposite,
                                   ui.dockWidgetPostprocessing }) {
            if (!dock) {
                continue;
            }
            removeDockWidget(dock);
            dock->setVisible(false);
        }
    };
    hideLegacyViewerDocks();
    QTimer::singleShot(0, this, [hideLegacyViewerDocks]() {
        hideLegacyViewerDocks();
    });

    connect(_surfacePanel.get(), &SurfacePanelController::surfaceActivated,
            this, &CWindow::onSurfaceActivated);
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceActivatedPreserveEditing,
            this, &CWindow::onSurfaceActivatedPreserveEditing);
    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }

    // new and remove path buttons
    // connect(ui.btnNewPath, SIGNAL(clicked()), this, SLOT(OnNewPathClicked()));
    // connect(ui.btnRemovePath, SIGNAL(clicked()), this, SLOT(OnRemovePathClicked()));

    // TODO CHANGE VOLUME LOADING; FIRST CHECK FOR OTHER VOLUMES IN THE STRUCTS
    if (ui.volSelect) {
        ui.volSelect->setLabelVisible(false);
        volSelect = ui.volSelect->comboBox();
    } else {
        volSelect = nullptr;
    }

    QComboBox* overlayVolumeSelect = nullptr;
    if (ui.overlayVolumeSelect) {
        ui.overlayVolumeSelect->setLabelVisible(false);
        overlayVolumeSelect = ui.overlayVolumeSelect->comboBox();
    }

    if (_volumeOverlay) {
        VolumeOverlayController::UiRefs overlayUi{
            .volumeSelect = overlayVolumeSelect,
            .colormapSelect = ui.overlayColormapSelect,
            .opacitySpin = ui.overlayOpacitySpin,
            .thresholdSpin = ui.overlayThresholdSpin,
        };
        _volumeOverlay->setUi(overlayUi);
        if (_viewerControlsPanel) {
            _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay->hasOverlaySelection());
        }
    }

    // Setup surface overlay controls
    connect(ui.chkSurfaceOverlay, &QCheckBox::toggled, [this](bool checked) {
        if (!_viewerManager) return;
        _viewerManager->forEachBaseViewer([checked](VolumeViewerBase* viewer) {
            viewer->setSurfaceOverlayEnabled(checked);
        });
        ui.surfaceOverlaySelect->setEnabled(checked);
        ui.spinOverlapThreshold->setEnabled(checked);
    });

    connect(ui.spinOverlapThreshold, qOverload<double>(&QDoubleSpinBox::valueChanged), [this](double value) {
        if (!_viewerManager) return;
        _viewerManager->forEachBaseViewer([value](VolumeViewerBase* viewer) {
            viewer->setSurfaceOverlapThreshold(static_cast<float>(value));
        });
    });

    // Initially disable surface overlay controls
    ui.surfaceOverlaySelect->setEnabled(false);
    ui.spinOverlapThreshold->setEnabled(false);

    // Initialize surface overlay dropdown (will be populated when surfaces load)
    updateSurfaceOverlayDropdown();



    connect(
        volSelect, &QComboBox::currentIndexChanged, [this](const int& index) {
            auto vpkg = _state->vpkg();
            if (vpkg && index >= 0) {
                std::shared_ptr<Volume> newVolume;
                try {
                    newVolume = vpkg->volume(volSelect->currentData().toString().toStdString());
                } catch (const std::out_of_range& e) {
                    QMessageBox::warning(this, "Error", "Could not load volume.");
                    return;
                }
                setVolume(newVolume);
            }
        });

    auto* filterDropdown = ui.btnFilterDropdown;
    auto* cmbPointSetFilter = ui.cmbPointSetFilter;
    auto* btnPointSetFilterAll = ui.btnPointSetFilterAll;
    auto* btnPointSetFilterNone = ui.btnPointSetFilterNone;
    auto* cmbPointSetFilterMode = new QComboBox();
    cmbPointSetFilterMode->addItem("Any (OR)");
    cmbPointSetFilterMode->addItem("All (AND)");
    ui.pointSetFilterLayout->insertWidget(1, cmbPointSetFilterMode);

    SurfacePanelController::FilterUiRefs filterUi;
    filterUi.dropdown = filterDropdown;
    filterUi.pointSet = cmbPointSetFilter;
    filterUi.pointSetAll = btnPointSetFilterAll;
    filterUi.pointSetNone = btnPointSetFilterNone;
    filterUi.pointSetMode = cmbPointSetFilterMode;
    filterUi.surfaceIdFilter = ui.lineEditSurfaceFilter;
    _surfacePanel->configureFilters(filterUi, _state->pointCollection());

    SurfacePanelController::TagUiRefs tagUi{
        .approved = ui.chkApproved,
        .defective = ui.chkDefective,
        .reviewed = ui.chkReviewed,
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

    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        bool showOverlays = settings.value(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS,
                                           vc3d::settings::viewer::SHOW_AXIS_OVERLAYS_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisOverlays);
        chkAxisOverlays->setChecked(showOverlays);
        connect(chkAxisOverlays, &QCheckBox::toggled, this, &CWindow::onAxisOverlayVisibilityToggled);
    }
    if (auto* chkMoveOnSurfaceChanged = ui.chkMoveOnSurfaceChanged) {
        bool moveOnSurfaceChanged = settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                                                   vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
        QSignalBlocker blocker(chkMoveOnSurfaceChanged);
        chkMoveOnSurfaceChanged->setChecked(moveOnSurfaceChanged);
        connect(chkMoveOnSurfaceChanged, &QCheckBox::toggled, this, &CWindow::onMoveOnSurfaceChangedToggled);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        int storedOpacity = settings.value(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY,
                                           spinAxisOverlayOpacity->value()).toInt();
        storedOpacity = std::clamp(storedOpacity, spinAxisOverlayOpacity->minimum(), spinAxisOverlayOpacity->maximum());
        QSignalBlocker blocker(spinAxisOverlayOpacity);
        spinAxisOverlayOpacity->setValue(storedOpacity);
        connect(spinAxisOverlayOpacity, qOverload<int>(&QSpinBox::valueChanged), this, &CWindow::onAxisOverlayOpacityChanged);
    }

    if (auto* btnResetRot = ui.btnResetAxisRotations) {
        connect(btnResetRot, &QPushButton::clicked, this, &CWindow::onResetAxisAlignedRotations);
    }

    chkAxisAlignedSlices = ui.chkAxisAlignedSlices;
    if (chkAxisAlignedSlices) {
        bool useAxisAligned = settings.value(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES,
                                             vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisAlignedSlices);
        chkAxisAlignedSlices->setChecked(useAxisAligned);
        connect(chkAxisAlignedSlices, &QCheckBox::toggled, this, &CWindow::onAxisAlignedSlicesToggled);
    }

    connect(ui.btnEditMask, &QPushButton::pressed, this, &CWindow::onEditMaskPressed);
    connect(ui.btnAppendMask, &QPushButton::pressed, this, &CWindow::onAppendMaskPressed);  // Add this

    if (chkAxisAlignedSlices) {
        onAxisAlignedSlicesToggled(chkAxisAlignedSlices->isChecked());
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        onAxisOverlayOpacityChanged(spinAxisOverlayOpacity->value());
    }
    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        onAxisOverlayVisibilityToggled(chkAxisOverlays->isChecked());
    }
    if (auto* chkMoveOnSurfaceChanged = ui.chkMoveOnSurfaceChanged) {
        onMoveOnSurfaceChangedToggled(chkMoveOnSurfaceChanged->isChecked());
    }

}

// Create menus
// Create actions
void CWindow::keyPressEvent(QKeyEvent* event)
{
    // Let fiber controller handle Escape first
    if (event->key() == Qt::Key_Escape && _fiberController && _fiberController->handleEscape()) {
        event->accept();
        return;
    }

    // Fiber animation (P key)
    if (_fiberController && _fiberController->handleKeyPress(event))
        return;

    if (event->key() == vc3d::keybinds::keypress::ToggleVolumeOverlay.key &&
        event->modifiers() == vc3d::keybinds::keypress::ToggleVolumeOverlay.modifiers) {
        toggleVolumeOverlayVisibility();
        event->accept();
        return;
    }

    if (event->key() == vc3d::keybinds::keypress::CenterFocusOnCursor.key &&
        event->modifiers() == vc3d::keybinds::keypress::CenterFocusOnCursor.modifiers) {
        if (centerFocusOnCursor()) {
            event->accept();
            return;
        }
    }

    if (event->key() == vc3d::keybinds::keypress::RecenterFocus.key &&
        event->modifiers() == vc3d::keybinds::keypress::RecenterFocus.modifiers) {
        if (recenterViewersOnCurrentFocus()) {
            event->accept();
            return;
        }
    }

    // Shift+G decreases slice step size, Shift+H increases it
    if (event->modifiers() == vc3d::keybinds::keypress::SliceStepDecrease.modifiers && _viewerManager) {
        if (event->key() == vc3d::keybinds::keypress::SliceStepDecrease.key) {
            int currentStep = _viewerManager->sliceStepSize();
            int newStep = std::max(1, currentStep - 1);
            _viewerManager->setSliceStepSize(newStep);
            onSliceStepSizeChanged(newStep);
            event->accept();
            return;
        } else if (event->key() == vc3d::keybinds::keypress::SliceStepIncrease.key) {
            int currentStep = _viewerManager->sliceStepSize();
            int newStep = std::min(100, currentStep + 1);
            _viewerManager->setSliceStepSize(newStep);
            onSliceStepSizeChanged(newStep);
            event->accept();
            return;
        }
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

void CWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    scheduleWindowStateSave();
}

void CWindow::scheduleWindowStateSave()
{
    // Restart the timer - this debounces rapid changes
    if (_windowStateSaveTimer) {
        _windowStateSaveTimer->start();
    }
}

void CWindow::saveWindowState()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::window::GEOMETRY, saveGeometry());
    settings.setValue(vc3d::settings::window::STATE, saveState());
    writeWindowStateMeta(settings,
                         windowStateScreenSignature(),
                         windowStateQtVersion(),
                         windowStateAppVersion());
    settings.sync();
}

void CWindow::closeEvent(QCloseEvent* event)
{
    // Tell ViewerManager to stop maintaining the SurfacePatchIndex. The
    // CState teardown below iterates every tracked surface and sets it to
    // nullptr, which would otherwise trigger an O(N) rtree->remove() per
    // surface — easily 10+ seconds on a flattened segment with millions
    // of cells.
    if (_viewerManager) {
        _viewerManager->beginShutdown();
    }
    if (_state && _state->vpkg()) {
        try { _state->vpkg()->saveAutosave(); } catch (...) {}
    }
    saveWindowState();
    event->accept();
}

void CWindow::setWidgetsEnabled(bool state)
{
    ui.grpVolManager->setEnabled(state);
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewControlsEnabled(state);
        _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay && _volumeOverlay->hasOverlaySelection());
    }
}

auto CWindow::InitializeVolumePkg(const std::string& nVpkgPath) -> bool
{
    _state->setVpkg(nullptr);
    updateNormalGridAvailability();
    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        _segmentationModule->setEditingEnabled(false);
    }
    if (_segmentationModule) {
        _segmentationModule->setAnnotateMode(false);
    }
    if (_segmentationWidget) {
        if (!_segmentationModule || _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

    try {
        _state->setVpkg(VolumePkg::load(nVpkgPath));
    } catch (const std::exception& e) {
        Logger()->error("Failed to initialize volpkg: {}", e.what());
    }

    if (_state->vpkg() == nullptr) {
        Logger()->error("Cannot open project: {}", nVpkgPath);
        QMessageBox::warning(
            this, "Error",
            "Project failed to load. Falling back to a new blank project.");
        _state->setVpkg(VolumePkg::newEmpty());
        return false;
    }
    return true;
}

// Update the widgets
void CWindow::UpdateView(void)
{
    if (!_state->hasVpkg() && _state->currentVolume() == nullptr) {
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
    if (_state->vpkg()) {
        QString label = tr("%1").arg(QString::fromStdString(_state->vpkg()->name()));
        ui.lblVpkgName->setText(label);
    } else if (_state->currentVolume()) {
        QString label = tr("Remote: %1").arg(QString::fromStdString(_state->currentVolumeId()));
        ui.lblVpkgName->setText(label);
    }
}

void CWindow::onShowStatusMessage(QString text, int timeout)
{
    statusBar()->showMessage(text, timeout);
}

void CWindow::onSegmentationGrowthStatusChanged(bool running)
{
    if (!statusBar()) {
        return;
    }

    if (_surfacePanel) {
        _surfacePanel->setSelectionLocked(running);
    }

    if (running) {
        if (!_segmentationGrowthWarning) {
            _segmentationGrowthWarning = new QLabel(statusBar());
            _segmentationGrowthWarning->setObjectName(QStringLiteral("segmentationGrowthWarning"));
            _segmentationGrowthWarning->setStyleSheet(QStringLiteral("color: #c62828; font-weight: 600;"));
            _segmentationGrowthWarning->setContentsMargins(8, 0, 8, 0);
            _segmentationGrowthWarning->setAlignment(Qt::AlignCenter);
            _segmentationGrowthWarning->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
            _segmentationGrowthWarning->setMinimumWidth(260);
            _segmentationGrowthWarning->hide();
            statusBar()->addPermanentWidget(_segmentationGrowthWarning, 1);
        }
        _segmentationGrowthStatusText = tr("Surface growth in progress - surface selection locked");
        _segmentationGrowthWarning->setText(_segmentationGrowthStatusText);
        _segmentationGrowthWarning->setVisible(true);
        statusBar()->showMessage(_segmentationGrowthStatusText, 0);
    } else if (_segmentationGrowthWarning) {
        _segmentationGrowthWarning->clear();
        _segmentationGrowthWarning->setVisible(false);
        if (statusBar()->currentMessage() == _segmentationGrowthStatusText) {
            statusBar()->clearMessage();
        }
        _segmentationGrowthStatusText.clear();
    }
}

void CWindow::onSliceStepSizeChanged(int newSize)
{
    // Update status bar label
    if (_sliceStepLabel) {
        _sliceStepLabel->setText(tr("Step: %1").arg(newSize));
    }

    // Save to settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SLICE_STEP_SIZE, newSize);

    if (_viewerControlsPanel) {
        _viewerControlsPanel->setSliceStepSize(newSize);
    }
}

// Open volume package
void CWindow::OpenVolume(const QString& path)
{
    QString aVpkgPath = path;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        aVpkgPath = QFileDialog::getOpenFileName(
            this, tr("Open Project"), settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
            tr("Project (*.volpkg.json);;All files (*.*)"),
            nullptr, QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);
        if (aVpkgPath.isEmpty()) {
            Logger()->info("Open project canceled");
            return;
        }
    }

    if (!InitializeVolumePkg(aVpkgPath.toStdString())) {
        return;
    }

    // Check version number
    if (_state->vpkg()->version() < VOLPKG_MIN_VERSION) {
        const auto msg = "Volume package is version " +
                         std::to_string(_state->vpkg()->version()) +
                         " but this program requires version " +
                         std::to_string(VOLPKG_MIN_VERSION) + "+.";
        Logger()->error(msg);
        QMessageBox::warning(this, tr("ERROR"), QString(msg.c_str()));
        _state->setVpkg(nullptr);
        updateNormalGridAvailability();
        return;
    }

    refreshCurrentVolumePackageUi(QString(), true);
    if (_menuController) {
        _menuController->updateRecentVolpkgList(aVpkgPath);
    }

    if (_fileWatcher) {
        _fileWatcher->startWatching();
    }
}

void CWindow::refreshVolumeSelectionUi(const QString& preferredVolumeId)
{
    if (!volSelect || !_state || !_state->vpkg()) {
        return;
    }

    QVector<QPair<QString, QString>> volumeEntries;
    std::vector<QString> orderedIds;
    QString activeCandidate = preferredVolumeId;
    const QString currentComboId = volSelect->currentData().toString();
    const QString currentVolumeId = QString::fromStdString(_state->currentVolumeId());

    auto hasVolume = [&](const QString& volumeId) {
        for (const auto& id : orderedIds) {
            if (id == volumeId) {
                return true;
            }
        }
        return false;
    };

    QString bestGrowthVolumeId;
    bool preferredVolumeFound = false;
    const auto volumeIds = _state->vpkg()->volumeIDs();
    for (const auto& id : volumeIds) {
        try {
            auto vol = _state->vpkg()->volume(id);
            const QString idStr = QString::fromStdString(id);
            const QString nameStr = QString::fromStdString(vol->name());
            const QString label = nameStr.isEmpty() ? idStr : QStringLiteral("%1 (%2)").arg(nameStr, idStr);

            orderedIds.push_back(idStr);
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
        } catch (...) {
            continue;
        }
    }

    if (bestGrowthVolumeId.isEmpty() && !volumeEntries.isEmpty()) {
        bestGrowthVolumeId = orderedIds.front();
    }

    if (!activeCandidate.isEmpty() && !hasVolume(activeCandidate)) {
        activeCandidate.clear();
    }
    if (activeCandidate.isEmpty() && !currentComboId.isEmpty() && hasVolume(currentComboId)) {
        activeCandidate = currentComboId;
    }
    if (activeCandidate.isEmpty() && !currentVolumeId.isEmpty() && hasVolume(currentVolumeId)) {
        activeCandidate = currentVolumeId;
    }
    if (activeCandidate.isEmpty() && !volumeEntries.isEmpty()) {
        activeCandidate = orderedIds.front();
    }

    {
        const QSignalBlocker blocker{volSelect};
        volSelect->clear();
        for (const auto& [id, label] : volumeEntries) {
            volSelect->addItem(label, QVariant(id));
        }
        if (activeCandidate.isEmpty()) {
            if (volSelect->count() > 0) {
                volSelect->setCurrentIndex(0);
            }
        } else {
            volSelect->setCurrentIndex(volSelect->findData(activeCandidate));
        }
    }

    QString activeId = volSelect->count() > 0 ? volSelect->currentData().toString() : QString();

    QString growthVolumeId = QString::fromStdString(_state->segmentationGrowthVolumeId());
    if (!growthVolumeId.isEmpty() && !hasVolume(growthVolumeId)) {
        growthVolumeId.clear();
    }
    if (growthVolumeId.isEmpty()) {
        growthVolumeId = bestGrowthVolumeId;
    }
    if (growthVolumeId.isEmpty()) {
        growthVolumeId = activeId;
    }

    if (!activeId.isEmpty()) {
        if (!_state->currentVolume() || _state->currentVolumeId() != activeId.toStdString()) {
            try {
                auto newVolume = _state->vpkg()->volume(activeId.toStdString());
                setVolume(newVolume);
            } catch (...) {
                // Ignore errors - keep existing volume selection if invalid.
            }
        }

        _state->setSegmentationGrowthVolumeId(growthVolumeId.toStdString());

        if (_segmentationWidget) {
            _segmentationWidget->setAvailableVolumes(volumeEntries, growthVolumeId);
            if (!growthVolumeId.isEmpty()) {
                _segmentationWidget->setActiveVolume(growthVolumeId);
            }
            try {
                auto growthVolume = _state->vpkg()->volume(growthVolumeId.toStdString());
                if (growthVolume) {
                    _segmentationWidget->setVolumeZarrPath(QString::fromStdString(growthVolume->path().string()));
                }
            } catch (...) {
                // Ignore errors - neural growth path update is non-critical.
            }
        }
    } else {
        setVolume(nullptr);
        _state->setSegmentationGrowthVolumeId({});
        if (_segmentationWidget) {
            _segmentationWidget->setAvailableVolumes(QVector<QPair<QString, QString>>{}, {});
            _segmentationWidget->setActiveVolume({});
            _segmentationWidget->setVolumeZarrPath({});
        }
    }
}

void CWindow::CloseVolume(void)
{
    if (_fileWatcher) {
        _fileWatcher->stopWatching();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->clearPreview(false);
    }

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

    // CState::closeAll emits volumeClosing, clears surfaces, vpkg, volume, points
    _state->closeAll();

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

    if (_volumeOverlay) {
        _volumeOverlay->clearVolumePkg();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

// Handle open request
auto CWindow::can_change_volume_() -> bool
{
    if (_state->hasVpkg() && _state->vpkg()->numberOfVolumes() > 1) {
        return true;
    }
    // Also allow switching when volSelect has multiple remote volumes
    if (volSelect && volSelect->count() > 1) {
        return true;
    }
    return false;
}

void CWindow::onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    // Let fiber annotation controller consume the click if in WaitingForFirstClick mode
    if (_fiberController && _fiberController->handleVolumeClick(vol_loc, normal, surf, buttons, modifiers))
        return;

    if (modifiers & Qt::ShiftModifier) {
        return;
    }
    else if (modifiers & Qt::ControlModifier) {
        std::cout << "clicked on vol loc " << vol_loc << std::endl;
        // Get the surface ID from the surface collection
        std::string surfId;
        if (_state && surf) {
            surfId = _state->findSurfaceId(surf);
        }
        centerFocusAt(vol_loc, normal, surfId);
    }
    else {
    }
}

void CWindow::onSurfaceActivated(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _state->activeSurfaceId();
    const std::string newSurfId = surfaceId.toStdString();

    // Look up the shared_ptr by ID
    if (_state->vpkg() && !newSurfId.empty()) {
        _state->setActiveSurface(newSurfId, _state->vpkg()->getSurface(newSurfId));
    } else {
        _state->clearActiveSurface();
    }

    auto surf = _state->activeSurface().lock();

    if (newSurfId != previousSurfId) {
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }

        if (_segmentationModule) {
            try {
                _segmentationModule->onActiveSegmentChanged(surf.get());
            } catch (const std::exception& e) {
                qWarning() << "Failed to activate surface"
                           << surfaceId
                           << "while it may still be writing:"
                           << e.what();
                _state->clearActiveSurface();
                _state->setSurface("segmentation", nullptr, false, false);
                surf.reset();
            }
        }

        // Load corr_points_results for the new segment
        if (_point_collection_widget) {
            auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf);
            if (quadSurf && !quadSurf->path.empty()) {
                _point_collection_widget->loadCorrPointsResults(
                    quadSurf->path / "corr_points_results.json");
            } else {
                _point_collection_widget->clearCorrPointsResults();
            }
        }
    }

    const bool activatingAxisAlignedPlane =
        newSurfId == "xy plane" || newSurfId == "seg xz" || newSurfId == "seg yz";
    if (_axisAlignedSliceController && !activatingAxisAlignedPlane) {
        _axisAlignedSliceController->resetAll();
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const bool moveOnSurfaceChange =
        settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                       vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();

    if (moveOnSurfaceChange) {
        if (auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf)) {
            try {
                quadSurf->ensureLoaded();
                const cv::Vec3f worldCenter = quadSurf->coord({0, 0, 0}, {0, 0, 0});
                const bool centerValid = std::isfinite(worldCenter[0])
                    && std::isfinite(worldCenter[1])
                    && std::isfinite(worldCenter[2])
                    && worldCenter[0] >= 0.0f;
                if (centerValid) {
                    if (auto vol = _state->currentVolume()) {
                        auto [w, h, d] = vol->shapeXyz();
                        cv::Vec3f clamped = worldCenter;
                        clamped[0] = std::clamp(clamped[0], 0.0f, static_cast<float>(w - 1));
                        clamped[1] = std::clamp(clamped[1], 0.0f, static_cast<float>(h - 1));
                        clamped[2] = std::clamp(clamped[2], 0.0f, static_cast<float>(d - 1));
                        POI* poi = new POI;
                        poi->p = clamped;
                        poi->n = cv::Vec3f(0, 0, 0);
                        poi->surfaceId = newSurfId;
                        _state->setPOI("focus", poi);
                    }
                }
            } catch (const std::exception& e) {
                qWarning() << "Could not compute world center for"
                           << surfaceId << ":" << e.what();
            }
        }
    }

    try {
        if (surf) {
            _axisAlignedSliceController->applyOrientation(surf.get());
        } else {
            _axisAlignedSliceController->applyOrientation();
        }
    } catch (const std::exception& e) {
        qWarning() << "Failed to apply surface orientation for"
                   << surfaceId
                   << "while it may still be writing:"
                   << e.what();
        _state->clearActiveSurface();
        _state->setSurface("segmentation", nullptr, false, false);
        _axisAlignedSliceController->applyOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

void CWindow::onSurfaceActivatedPreserveEditing(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _state->activeSurfaceId();
    const std::string newSurfId = surfaceId.toStdString();

    if (_state->vpkg() && !newSurfId.empty()) {
        _state->setActiveSurface(newSurfId, _state->vpkg()->getSurface(newSurfId));
    } else {
        _state->clearActiveSurface();
    }

    auto surf = _state->activeSurface().lock();

    if (newSurfId != previousSurfId && _segmentationModule) {
        try {
            _segmentationModule->onActiveSegmentChanged(surf.get());
        } catch (const std::exception& e) {
            qWarning() << "Failed to activate surface"
                       << surfaceId
                       << "while it may still be writing:"
                       << e.what();
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, false, false);
            surf.reset();
        }

        // Load corr_points_results for the new segment
        if (_point_collection_widget) {
            auto quadSurf = std::dynamic_pointer_cast<QuadSurface>(surf);
            if (quadSurf && !quadSurf->path.empty()) {
                _point_collection_widget->loadCorrPointsResults(
                    quadSurf->path / "corr_points_results.json");
            } else {
                _point_collection_widget->clearCorrPointsResults();
            }
        }

        const bool wantsEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        if (wantsEditing) {
            if (!_segmentationModule->editingEnabled()) {
                _segmentationModule->setEditingEnabled(true);
            } else if (_state) {
                auto targetSurface = surf;
                if (!targetSurface) {
                    targetSurface = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
                }

                if (targetSurface) {
                    _segmentationModule->endEditingSession();
                    if (_segmentationModule->beginEditingSession(targetSurface) && _viewerManager) {
                        _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                            if (viewer) {
                                viewer->clearOverlayGroup("segmentation_radius_indicator");
                            }
                        });
                    }
                }
            }
        }
    }

    try {
        if (surf) {
            _axisAlignedSliceController->applyOrientation(surf.get());
        } else {
            _axisAlignedSliceController->applyOrientation();
        }
    } catch (const std::exception& e) {
        qWarning() << "Failed to apply surface orientation for"
                   << surfaceId
                   << "while it may still be writing:"
                   << e.what();
        _state->clearActiveSurface();
        _state->setSurface("segmentation", nullptr, false, false);
        _axisAlignedSliceController->applyOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }

    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
    }
}

void CWindow::onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf)
{
    // Called BEFORE surface deletion - clear all references to prevent use-after-free

    // Clear if this is our current active surface
    auto currentSurf = _state->activeSurface().lock();
    if (currentSurf && currentSurf == surf) {
        _state->clearActiveSurface();
    }

    // Focus history uses string IDs now, so no cleanup needed for surface pointers
    // (the ID remains valid for lookup - will just return nullptr if surface is gone)
}

void CWindow::onEditMaskPressed(void)
{
    auto surf = _state->activeSurface().lock();
    if (!surf)
        return;

    std::filesystem::path path = surf->path/"mask.tif";

    // If mask already exists, just open it
    if (std::filesystem::exists(path)) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));
        return;
    }

    if (_maskRenderInProgress)
        return;
    _maskRenderInProgress = true;
    ui.btnEditMask->setEnabled(false);
    ui.btnAppendMask->setEnabled(false);
    statusBar()->showMessage(tr("Rendering mask..."));

    auto* watcher = new QFutureWatcher<void>(this);
    connect(watcher, &QFutureWatcher<void>::finished, this,
            [this, watcher, surf, path]() {
                watcher->deleteLater();
                _maskRenderInProgress = false;
                ui.btnEditMask->setEnabled(true);
                ui.btnAppendMask->setEnabled(true);

                statusBar()->showMessage(tr("Mask saved"), 3000);
                QDesktopServices::openUrl(QUrl::fromLocalFile(
                    QString::fromStdString(path.string())));
            });

    watcher->setFuture(QtConcurrent::run([surf, path]() {
        cv::Mat_<uint8_t> mask;
        cv::Mat_<cv::Vec3f> coords;
        render_binary_mask(surf.get(), mask, coords, 1.0f);
        cv::imwrite(path.string(), mask);

        surf->meta["date_last_modified"] = get_surface_time_str();
        surf->save_meta();
    }));
}

void CWindow::onAppendMaskPressed(void)
{
    auto surf = _state->activeSurface().lock();
    if (!surf || !_state->currentVolume()) {
        if (!surf) {
            QMessageBox::warning(this, tr("Error"), tr("No surface selected."));
        } else {
            QMessageBox::warning(this, tr("Error"), tr("No volume loaded."));
        }
        return;
    }

    if (_maskRenderInProgress)
        return;
    _maskRenderInProgress = true;
    ui.btnEditMask->setEnabled(false);
    ui.btnAppendMask->setEnabled(false);
    statusBar()->showMessage(tr("Rendering mask..."));

    std::filesystem::path path = surf->path/"mask.tif";
    auto volume = _state->currentVolume();

    auto* watcher = new QFutureWatcher<QString>(this);
    connect(watcher, &QFutureWatcher<QString>::finished, this,
            [this, watcher, path]() {
                watcher->deleteLater();
                _maskRenderInProgress = false;
                ui.btnEditMask->setEnabled(true);
                ui.btnAppendMask->setEnabled(true);

                try {
                    QString msg = watcher->result();
                    statusBar()->showMessage(msg, 3000);
                    QDesktopServices::openUrl(QUrl::fromLocalFile(
                        QString::fromStdString(path.string())));
                } catch (const std::exception& e) {
                    QMessageBox::critical(this, tr("Error"),
                                         tr("Failed to render surface: %1").arg(e.what()));
                    statusBar()->clearMessage();
                }
            });

    watcher->setFuture(QtConcurrent::run([surf, volume, path]() -> QString {
        cv::Mat_<uint8_t> mask;
        cv::Mat_<uint8_t> img;
        std::vector<cv::Mat> existing_layers;

        if (std::filesystem::exists(path)) {
            cv::imreadmulti(path.string(), existing_layers, cv::IMREAD_UNCHANGED);

            if (existing_layers.empty())
                throw std::runtime_error("Could not read existing mask file.");

            mask = existing_layers[0];
            cv::Size maskSize = mask.size();

            {
                cv::Size rawSize = surf->rawPointsPtr()->size();
                cv::Vec3f ptr(0, 0, 0);
                cv::Vec3f offset(-rawSize.width/2.0f, -rawSize.height/2.0f, 0);
                float surfScale = surf->scale()[0];
                cv::Mat_<cv::Vec3f> coords;
                surf->gen(&coords, nullptr, maskSize, ptr, surfScale, offset);
                img.create(coords.size());
                render_image_from_coords(coords, img, volume.get());
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            existing_layers.push_back(img);
            atomicImwriteMulti(path, existing_layers);

            QString msg = QString("Appended surface image to existing mask (now %1 layers)")
                              .arg(existing_layers.size());

            surf->meta["date_last_modified"] = get_surface_time_str();
            surf->save_meta();
            return msg;

        } else {
            cv::Mat_<cv::Vec3f> coords;
            render_binary_mask(surf.get(), mask, coords, 1.0f);
            render_surface_image(surf.get(), mask, img, volume.get(), 0, 1.0f);
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            std::vector<cv::Mat> layers = {mask, img};
            atomicImwriteMulti(path, layers);

            surf->meta["date_last_modified"] = get_surface_time_str();
            surf->save_meta();
            return QString("Created new surface mask with image data");
        }
    }));
}

QString CWindow::getCurrentVolumePath() const
{
    auto volume = _state->currentVolume();
    if (volume == nullptr) {
        return QString();
    }
    if (volume->isRemote()) {
        return QString::fromStdString(volume->remoteUrl());
    }
    return QString::fromStdString(volume->path().string());
}

void CWindow::onSegmentationDirChanged(int index)
{
    if (!_state->vpkg() || index < 0 || !cmbSegmentationDir) {
        return;
    }

    std::string newDir = cmbSegmentationDir->itemText(index).toStdString();

    // Only reload if the directory actually changed
    if (newDir != _state->vpkg()->getSegmentationDirectory()) {
        // Clear the current segmentation surface first to ensure viewers update
        _state->setSurface("segmentation", nullptr, true);

        // Clear current surface selection
        _state->clearActiveSurface();
        treeWidgetSurfaces->clearSelection();

        if (_surfacePanel) {
            _surfacePanel->resetTagUi();
        }

        // Set the new directory in the VolumePkg
        _state->vpkg()->setSegmentationDirectory(newDir);

        // Reset stride user override for the new directory.
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
        if (_surfacePanel) {
            _surfacePanel->loadSurfaces(true);
        }

        // Update the status bar to show the change
        statusBar()->showMessage(tr("Switched to %1 directory").arg(QString::fromStdString(newDir)), 3000);
    }
}


void CWindow::onManualLocationChanged()
{
    // Check if we have a valid volume loaded
    if (!_state->currentVolume()) {
        return;
    }

    // Parse the comma-separated values
    QString text = lblLocFocus->text().trimmed();
    QStringList parts = text.split(',');

    // Validate we have exactly 3 parts
    if (parts.size() != 3) {
        // Invalid input - restore the previous values
        POI* poi = _state->poi("focus");
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
        POI* poi = _state->poi("focus");
        if (poi) {
            lblLocFocus->setText(QString("%1, %2, %3")
                .arg(static_cast<int>(poi->p[0]))
                .arg(static_cast<int>(poi->p[1]))
                .arg(static_cast<int>(poi->p[2])));
        }
        return;
    }

    // Clamp values to physical volume bounds
    auto [w, h, d] = _state->currentVolume()->shapeXyz();
    int cx0 = 0, cy0 = 0, cz0 = 0;
    int cx1 = w - 1, cy1 = h - 1, cz1 = d - 1;

    x = std::max(cx0, std::min(x, cx1));
    y = std::max(cy0, std::min(y, cy1));
    z = std::max(cz0, std::min(z, cz1));

    // Update the line edit with clamped values
    lblLocFocus->setText(QString("%1, %2, %3").arg(x).arg(y).arg(z));

    // Route through centerFocusAt so slice planes reorient (canonical fallback
    // when no segment is loaded, segment-tangent otherwise) — same behaviour
    // as ctrl-click in a viewer.
    centerFocusAt(cv::Vec3f(x, y, z), cv::Vec3f(0, 0, 1), std::string());

    if (_surfacePanel) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::onZoomIn()
{
    if (auto* viewer = activeBaseViewer()) {
        viewer->adjustZoomByFactor(1.15f);
    }
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

        _axisAlignedSliceController->applyOrientation();

        if (!poi->suppressViewerRecenter) {
            const cv::Vec3f focusPosition = poi->p;
            QTimer::singleShot(0, this, [this, focusPosition]() {
                recenterPlaneViewersOn(focusPosition);
            });
        }
    }
}

void CWindow::onPointDoubleClicked(uint64_t pointId)
{
    auto point_opt = _state->pointCollection()->getPoint(pointId);
    if (point_opt) {
        centerFocusAt(point_opt->p, cv::Vec3f(0, 0, 0), "");
    }
}

void CWindow::onConvertPointToAnchor(uint64_t pointId, uint64_t collectionId)
{
    auto point_opt = _state->pointCollection()->getPoint(pointId);
    if (!point_opt) {
        statusBar()->showMessage(tr("Point not found"), 2000);
        return;
    }

    // Get the segmentation surface to project the point onto
    auto seg_surface = _state->surface("segmentation");
    auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get());
    if (!quad_surface) {
        statusBar()->showMessage(tr("No active segmentation surface for anchor conversion"), 3000);
        return;
    }

    // Find the 2D grid location of this point on the surface
    cv::Vec3f ptr(0, 0, 0);
    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    float dist = quad_surface->pointTo(ptr, point_opt->p, 4.0, 1000, patchIndex);

    if (dist > 10.0) {
        statusBar()->showMessage(tr("Point is too far from surface (distance: %1)").arg(dist), 3000);
        return;
    }

    // Get the raw grid location (internal coordinates)
    cv::Vec3f loc_3d = quad_surface->loc_raw(ptr);
    cv::Vec2f anchor2d(loc_3d[0], loc_3d[1]);

    // Set the anchor2d on the collection
    _state->pointCollection()->setCollectionAnchor2d(collectionId, anchor2d);

    // Remove the point (it's now represented by the anchor)
    _state->pointCollection()->removePoint(pointId);

    statusBar()->showMessage(tr("Converted point to anchor at grid position (%1, %2)").arg(anchor2d[0]).arg(anchor2d[1]), 3000);
}

void CWindow::onZoomOut()
{
    if (auto* viewer = activeBaseViewer()) {
        viewer->adjustZoomByFactor(1.0f / 1.15f);
    }
}

void CWindow::onCopyCoordinates()
{
    QString coords = lblLocFocus->text().trimmed();
    if (!coords.isEmpty()) {
        QApplication::clipboard()->setText(coords);
        statusBar()->showMessage(tr("Coordinates copied to clipboard: %1").arg(coords), 2000);
    }
}

void CWindow::onResetAxisAlignedRotations()
{
    _axisAlignedSliceController->resetRotations();
    _axisAlignedSliceController->applyOrientation();
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }
    statusBar()->showMessage(tr("All plane rotations reset"), 2000);
}

void CWindow::onAxisOverlayVisibilityToggled(bool enabled)
{
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && _axisAlignedSliceController->isEnabled());
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        spinAxisOverlayOpacity->setEnabled(_axisAlignedSliceController->isEnabled() && enabled);
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS, enabled ? "1" : "0");
}

void CWindow::onAxisOverlayOpacityChanged(int value)
{
    float normalized = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedOverlayOpacity(normalized);
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY, value);
}

void CWindow::onAxisAlignedSlicesToggled(bool enabled)
{
    _axisAlignedSliceController->setEnabled(enabled, ui.chkAxisOverlays, ui.spinAxisOverlayOpacity);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES, enabled ? "1" : "0");
}

void CWindow::onMoveOnSurfaceChangedToggled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE, enabled ? "1" : "0");

    if (!_viewerManager) {
        return;
    }

    const bool editingActive = _segmentationModule && _segmentationModule->editingEnabled();
    _viewerManager->forEachBaseViewer([this, enabled, editingActive](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }
        _viewerManager->setResetDefaultFor(viewer, enabled);
        if (editingActive && viewer->surfName() == "segmentation") {
            viewer->setResetViewOnSurfaceChange(false);
            return;
        }
        viewer->setResetViewOnSurfaceChange(enabled);
    });
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

    std::optional<std::string> recentlyEditedId;
    if (!enabled) {
        if (auto* activeSurface = _segmentationModule->activeBaseSurface()) {
            recentlyEditedId = activeSurface->id;
        }
    }

    // Set flag BEFORE beginEditingSession so the surface change doesn't reset view
    if (_viewerManager) {
        _viewerManager->forEachBaseViewer([this, enabled](VolumeViewerBase* viewer) {
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

    if (enabled) {
        auto activeSurfaceShared = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));

        if (!_segmentationModule->beginEditingSession(activeSurfaceShared)) {
            statusBar()->showMessage(tr("Unable to start segmentation editing"), 3000);
            if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
                QSignalBlocker blocker(_segmentationWidget);
                _segmentationWidget->setEditingEnabled(false);
            }
            _segmentationModule->setEditingEnabled(false);
            return;
        }

        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->clearOverlayGroup("segmentation_radius_indicator");
                }
            });
        }
    } else {
        _segmentationModule->endEditingSession();

        if (recentlyEditedId && !recentlyEditedId->empty()) {
            _fileWatcher->markSegmentRecentlyEdited(*recentlyEditedId);
        }
    }

    const QString message = enabled
        ? tr("Segmentation editing enabled")
        : tr("Segmentation editing disabled");
    statusBar()->showMessage(message, 2000);
    if (_surfaceAffineTransforms) {
        _surfaceAffineTransforms->refresh();
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
                                        int steps,
                                        bool inpaintOnly)
{
    if (!_segmentationGrower) {
        statusBar()->showMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

    SegmentationGrower::Context context{
        _segmentationModule.get(),
        _segmentationWidget,
        _state,
        _viewerManager.get(),
    };
    _segmentationGrower->updateContext(context);

    SegmentationGrower::VolumeContext volumeContext{
        _state->vpkg(),
        _state->currentVolume(),
        _state->currentVolumeId(),
        _state->segmentationGrowthVolumeId().empty() ? _state->currentVolumeId() : _state->segmentationGrowthVolumeId(),
        _normalGridPath,
        _segmentationWidget ? _segmentationWidget->normal3dZarrPath() : QString()
    };

    if (!_segmentationGrower->start(volumeContext, method, direction, steps, inpaintOnly)) {
        return;
    }
}

void CWindow::updateSurfaceOverlayDropdown()
{
    if (!ui.surfaceOverlaySelect) {
        return;
    }

    // Disconnect previous model's signals if any
    if (_surfaceOverlayModel) {
        disconnect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
                   this, &CWindow::onSurfaceOverlaySelectionChanged);
    }

    // Create new model
    _surfaceOverlayModel = new QStandardItemModel(ui.surfaceOverlaySelect);
    ui.surfaceOverlaySelect->setModel(_surfaceOverlayModel);

    // Use a QListView to properly show checkboxes
    auto* listView = new QListView(ui.surfaceOverlaySelect);
    ui.surfaceOverlaySelect->setView(listView);

    // Get current segmentation directory for filtering
    std::string currentDir;
    if (_state->vpkg()) {
        currentDir = _state->vpkg()->getSegmentationDirectory();
    }

    // Add "All" item at the top
    auto* allItem = new QStandardItem(tr("All"));
    allItem->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    allItem->setData(Qt::Unchecked, Qt::CheckStateRole);
    allItem->setData(QStringLiteral("__all__"), Qt::UserRole);
    _surfaceOverlayModel->appendRow(allItem);

    if (_state) {
        const auto names = _state->surfaceNames();
        for (const auto& name : names) {
            // Only add QuadSurfaces (actual segmentations), skip PlaneSurfaces
            auto surf = _state->surface(name);
            auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get());
            if (!quadSurf) {
                continue;
            }

            // Filter by current segmentation directory
            if (!currentDir.empty() && !surf->path.empty()) {
                std::string surfDir = surf->path.parent_path().filename().string();
                if (surfDir != currentDir) {
                    continue;
                }
            }

            auto* item = new QStandardItem(QString::fromStdString(name));
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
            item->setData(Qt::Unchecked, Qt::CheckStateRole);
            item->setData(QString::fromStdString(name), Qt::UserRole);

            // Assign persistent color if not already assigned
            if (_surfaceOverlayColorAssignments.find(name) == _surfaceOverlayColorAssignments.end()) {
                _surfaceOverlayColorAssignments[name] = _nextSurfaceOverlayColorIndex++;
            }
            size_t colorIdx = _surfaceOverlayColorAssignments[name];

            // Create color swatch icon (16x16 colored square)
            QPixmap swatch(16, 16);
            swatch.fill(getOverlayColor(colorIdx));
            item->setIcon(QIcon(swatch));

            _surfaceOverlayModel->appendRow(item);
        }
    }

    // Connect model's dataChanged signal for checkbox state changes
    connect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
            this, &CWindow::onSurfaceOverlaySelectionChanged);
}

void CWindow::onSurfaceOverlaySelectionChanged(const QModelIndex& topLeft,
                                                const QModelIndex& /*bottomRight*/,
                                                const QVector<int>& roles)
{
    if (!roles.contains(Qt::CheckStateRole) || !_surfaceOverlayModel || !_viewerManager) {
        return;
    }

    // Check if "All" was toggled (row 0)
    QStandardItem* changedItem = _surfaceOverlayModel->itemFromIndex(topLeft);
    if (changedItem && changedItem->data(Qt::UserRole).toString() == QStringLiteral("__all__")) {
        bool allChecked = changedItem->checkState() == Qt::Checked;

        // Block signals while updating all items
        {
            QSignalBlocker blocker(_surfaceOverlayModel);
            for (int row = 1; row < _surfaceOverlayModel->rowCount(); ++row) {
                QStandardItem* item = _surfaceOverlayModel->item(row);
                if (item) {
                    item->setCheckState(allChecked ? Qt::Checked : Qt::Unchecked);
                }
            }
        }
    }

    // Build map of selected surfaces with colors
    std::map<std::string, cv::Vec3b> selectedSurfaces;
    int checkedCount = 0;
    int totalSurfaces = 0;

    for (int row = 1; row < _surfaceOverlayModel->rowCount(); ++row) {
        QStandardItem* item = _surfaceOverlayModel->item(row);
        if (!item) continue;

        totalSurfaces++;
        if (item->checkState() == Qt::Checked) {
            checkedCount++;
            std::string name = item->data(Qt::UserRole).toString().toStdString();
            size_t colorIdx = _surfaceOverlayColorAssignments[name];
            selectedSurfaces[name] = getOverlayColorBGR(colorIdx);
        }
    }

    // Update "All" checkbox state (partial/full/none) without triggering recursion
    {
        QSignalBlocker blocker(_surfaceOverlayModel);
        QStandardItem* allItem = _surfaceOverlayModel->item(0);
        if (allItem) {
            if (checkedCount == 0) {
                allItem->setCheckState(Qt::Unchecked);
            } else if (checkedCount == totalSurfaces && totalSurfaces > 0) {
                allItem->setCheckState(Qt::Checked);
            } else {
                allItem->setCheckState(Qt::PartiallyChecked);
            }
        }
    }

    // Propagate to all viewers
    _viewerManager->forEachBaseViewer([&selectedSurfaces](VolumeViewerBase* viewer) {
        viewer->setSurfaceOverlays(selectedSurfaces);
    });
}

QColor CWindow::getOverlayColor(size_t index) const
{
    static const std::vector<QColor> palette = {
        QColor(80, 180, 255),   // sky blue
        QColor(180, 80, 220),   // violet
        QColor(80, 220, 200),   // aqua/teal
        QColor(220, 80, 180),   // magenta
        QColor(80, 130, 255),   // medium blue
        QColor(160, 80, 255),   // purple
        QColor(80, 255, 220),   // cyan
        QColor(255, 80, 200),   // hot pink
        QColor(120, 220, 80),   // lime green
        QColor(80, 180, 120),   // spring green
        QColor(150, 200, 255),  // light sky blue
        QColor(200, 150, 230),  // light violet
    };
    return palette[index % palette.size()];
}

cv::Vec3b CWindow::getOverlayColorBGR(size_t index) const
{
    QColor c = getOverlayColor(index);
    return cv::Vec3b(c.blue(), c.green(), c.red());
}

void CWindow::onCopyWithNtRequested()
{
    if (!_segmentationGrower) {
        statusBar()->showMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

    SegmentationGrower::Context context{
        _segmentationModule.get(),
        _segmentationWidget,
        _state,
        _viewerManager.get(),
    };
    _segmentationGrower->updateContext(context);

    SegmentationGrower::VolumeContext volumeContext{
        _state->vpkg(),
        _state->currentVolume(),
        _state->currentVolumeId(),
        _state->segmentationGrowthVolumeId().empty() ? _state->currentVolumeId() : _state->segmentationGrowthVolumeId(),
        _normalGridPath,
        _segmentationWidget ? _segmentationWidget->normal3dZarrPath() : QString()
    };

    if (!_segmentationGrower->startCopyWithNt(volumeContext)) {
        return;
    }
}

void CWindow::onFocusViewsRequested(uint64_t collectionId, uint64_t pointId)
{
    if (!_state) return;
    auto* pointCollection = _state->pointCollection();
    if (!pointCollection) return;

    const auto& collections = pointCollection->getAllCollections();
    auto it = collections.find(collectionId);
    if (it == collections.end()) return;

    const auto& collection = it->second;
    if (collection.points.empty()) return;

    // Gather all 3D points
    std::vector<cv::Vec3f> pts;
    pts.reserve(collection.points.size());
    for (const auto& pair : collection.points) {
        pts.push_back(pair.second.p);
    }

    // Compute centroid
    cv::Vec3f centroid(0, 0, 0);
    for (const auto& p : pts) centroid += p;
    centroid *= 1.0f / pts.size();

    // Determine focus position
    cv::Vec3f focusPos = centroid;
    if (pointId != 0) {
        auto point_opt = pointCollection->getPoint(pointId);
        if (point_opt) focusPos = point_opt->p;
    }

    // Compute plane normal via PCA (only if >= 3 points)
    cv::Vec3f N(0, 0, 1); // default
    if (pts.size() >= 3) {
        // Build 3x3 covariance matrix from centered points
        cv::Matx33f cov = cv::Matx33f::zeros();
        for (const auto& p : pts) {
            cv::Vec3f d = p - centroid;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    cov(r, c) += d[r] * d[c];
        }
        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(cv::Mat(cov), eigenvalues, eigenvectors);
        // Eigenvectors are sorted descending by eigenvalue.
        // Smallest eigenvalue's eigenvector (row 2) = plane normal.
        N = cv::Vec3f(eigenvectors.at<float>(2, 0),
                      eigenvectors.at<float>(2, 1),
                      eigenvectors.at<float>(2, 2));
        N = normalizeOrZero(N);
        if (cv::norm(N) < kEpsilon) N = cv::Vec3f(0, 0, 1);
    } else if (pts.size() == 2) {
        cv::Vec3f d = normalizeOrZero(pts[1] - pts[0]);
        if (cv::norm(d) > kEpsilon) {
            // Pick N perpendicular to d and closest to a canonical axis
            cv::Vec3f candidates[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
            float bestDot = 1.0f;
            cv::Vec3f bestN(0, 0, 1);
            for (auto& axis : candidates) {
                float absDot = std::abs(d.dot(axis));
                if (absDot < bestDot) {
                    bestDot = absDot;
                    cv::Vec3f proj = normalizeOrZero(axis - d * d.dot(axis));
                    if (cv::norm(proj) > kEpsilon) bestN = proj;
                }
            }
            N = bestN;
        }
    } else {
        // 1 point: just center, don't change orientation
        centerFocusAt(focusPos, cv::Vec3f(0, 0, 1), "");
        return;
    }

    // Choose which viewer gets the primary plane
    const cv::Vec3f segYZCanonical(1, 0, 0);
    const cv::Vec3f segXZCanonical(0, 1, 0);

    std::string primaryName, secondaryName;
    cv::Vec3f secondaryCanonical;

    if (std::abs(N.dot(segYZCanonical)) >= std::abs(N.dot(segXZCanonical))) {
        primaryName = "seg yz";
        secondaryName = "seg xz";
        secondaryCanonical = segXZCanonical;
    } else {
        primaryName = "seg xz";
        secondaryName = "seg yz";
        secondaryCanonical = segYZCanonical;
    }

    // Helper to configure a plane with Z-up in-plane rotation
    const auto configureFocusPlane = [&](const std::string& planeName,
                                         const cv::Vec3f& normal) {
        auto planeShared = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(planeName));
        if (!planeShared) {
            planeShared = std::make_shared<PlaneSurface>();
        }
        planeShared->setOrigin(focusPos);
        planeShared->setNormal(normal);
        planeShared->setInPlaneRotation(0.0f);

        // Adjust in-plane rotation so Z projects "up"
        const cv::Vec3f upAxis(0.0f, 0.0f, 1.0f);
        const cv::Vec3f projectedUp = projectVectorOntoPlane(upAxis, normal);
        const cv::Vec3f desiredUp = normalizeOrZero(projectedUp);
        if (cv::norm(desiredUp) > kEpsilon) {
            const cv::Vec3f currentUp = planeShared->basisY();
            const float delta = signedAngleBetween(currentUp, desiredUp, normal);
            if (std::abs(delta) > kEpsilon) {
                planeShared->setInPlaneRotation(delta);
            }
        }

        _state->setSurface(planeName, planeShared);
    };

    // Set focus POI first — this triggers applySlicePlaneOrientation() which
    // overwrites slice planes. We set our custom planes after.
    POI* focus = _state->poi("focus");
    if (!focus) {
        focus = new POI;
    }
    focus->p = focusPos;
    focus->n = N;
    _state->setPOI("focus", focus);

    // Now set our PCA-derived planes (overriding what applySlicePlaneOrientation set)
    configureFocusPlane(primaryName, N);

    // Set secondary plane: component of other canonical axis orthogonal to N
    cv::Vec3f secNormal = normalizeOrZero(secondaryCanonical - N * N.dot(secondaryCanonical));
    if (cv::norm(secNormal) < kEpsilon) {
        // Fallback: use cross product
        secNormal = normalizeOrZero(crossProduct(N, cv::Vec3f(0, 0, 1)));
        if (cv::norm(secNormal) < kEpsilon) {
            secNormal = normalizeOrZero(crossProduct(N, cv::Vec3f(0, 1, 0)));
        }
    }
    configureFocusPlane(secondaryName, secNormal);

    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }

    statusBar()->showMessage(tr("Focused & aligned view to %1 points").arg(pts.size()), 3000);
}

// ---- Fiber annotation slots -------------------------------------------------

void CWindow::onNewFiberRequested()
{
    if (_fiberController) {
        _fiberController->beginNewFiber();
    }
}

void CWindow::onFiberCrosshairModeChanged(bool active)
{
    if (!_viewerManager) return;
    _viewerManager->forEachBaseViewer([active](VolumeViewerBase* v) {
        if (v->graphicsView()) {
            v->graphicsView()->setCursor(active ? Qt::CrossCursor : Qt::ArrowCursor);
        }
    });
}

void CWindow::onFiberViewersRequested()
{
    if (!_fiberController) return;

    constexpr int N = FiberAnnotationController::kNumViews;

    QMdiSubWindow* subWindows[N] = {};

    for (int i = 0; i < N; ++i) {
        QString title = (i == 0) ? tr("Fiber Ref") : tr("Fiber Annotate");

        auto* baseViewer = newConnectedViewer(
            FiberAnnotationController::fiberSurfaceName(i), title, mdiArea);
        auto* viewer = baseViewer
            ? qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())
            : nullptr;
        if (!viewer) continue;

        _fiberController->setFiberViewer(i, viewer);

        // Isolate from segmentation module and global focus POI
        disconnect(_state, &CState::poiChanged, viewer, &CTiledVolumeViewer::onPOIChanged);
        if (_segmentationModule)
            disconnect(viewer, nullptr, _segmentationModule.get(), nullptr);

        // Only the last view (annotation) handles clicks
        if (i == N - 1) {
            connect(viewer, &CTiledVolumeViewer::sendVolumeClicked,
                    _fiberController.get(), &FiberAnnotationController::onAnnotationViewerClicked);
        }

        if (_state->currentVolume())
            viewer->OnVolumeChanged(_state->currentVolume());

        subWindows[i] = qobject_cast<QMdiSubWindow*>(viewer->parentWidget());
        if (subWindows[i])
            subWindows[i]->show();
    }

    // Layout: 2 columns × 1 row — ref on the left, annotate on the right.
    QRect area = mdiArea->contentsRect();
    int colW = area.width() / 2;
    int rowH = area.height();

    for (int i = 0; i < N; ++i) {
        if (!subWindows[i]) continue;
        int x = area.x() + i * colW;
        int y = area.y();
        subWindows[i]->setGeometry(x, y, colW, rowH);
    }
}

void CWindow::onFiberAnnotationFinished(uint64_t fiberId)
{
    if (_fiberWidget) {
        _fiberWidget->selectFiber(fiberId);
    }
}
