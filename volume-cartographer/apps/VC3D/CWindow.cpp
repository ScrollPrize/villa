#include "CWindow.hpp"
#include "VolumeComboTooltip.hpp"

#include <QDateTime>
#include "FiberLengthDisplay.hpp"
#include "VolumeDisplayNames.hpp"
#include <vc_object_store/objects/VolumeTransforms.hpp>

#include "StatusDockPanelHost.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <iostream>

#include <array>
#include <functional>
#include <optional>

#include "VCSettings.hpp"
#include "Keybinds.hpp"
#include "OpenDataNormalGrids.hpp"
#include "viewer_controls/panels/ViewerCompositePanel.hpp"
#include "viewer_controls/panels/ViewerInkDetectionPanel.hpp"
#include <QGridLayout>
#include <QAction>
#include <QVBoxLayout>
#include <QCursor>
#include <QEvent>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <QGuiApplication>
#include <QBrush>
#include <QDialog>
#include <QDialogButtonBox>
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
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QStatusBar>
#include <QSizePolicy>
#include <QTimer>
#include <QSize>
#include <QVector>
#include <QVector3D>
#include <QLoggingCategory>
#include <QDebug>
#include <QFrame>
#include <QMouseEvent>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QStandardPaths>
#include <QMenu>
#include <QMainWindow>
#include <QTabBar>
#include <QTabWidget>
#include <QTreeWidget>
#include <QHeaderView>
#include <QFormLayout>
#include <QProgressBar>
#include <QColorDialog>
#include <QListView>
#include <QTreeView>
#include <QPainter>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStyledItemDelegate>
#include <QStyleOptionButton>
#include <QPersistentModelIndex>
#include <QSet>
#include <QPushButton>
#include <QItemSelectionModel>
#include <QItemSelection>
#include "utils/Json.hpp"
#include <QPointer>
#include <QListView>
#include <algorithm>
#include <set>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <cctype>
#include <string_view>
#include <utility>
#include <filesystem>
#include <system_error>
#include <unordered_map>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QStringList>
#include <array>

#include "volume_viewers/CVolumeViewerView.hpp"
#include "viewer_controls/ViewerControlsPanel.hpp"
#include "viewer_controls/panels/VolumeRegistrationsPanel.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "SettingsDialog.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/ViewerSplitGrid.hpp"
#include "CPointCollectionWidget.hpp"
#include "WrapAnnotationWidget.hpp"
#include "CFiberWidget.hpp"
#include "FiberCollectionController.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "CommandLineToolRunner.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "SurfaceCatalog.hpp"
#include "overlays/StoreSurfaceOverlayController.hpp"
#include "SurfacePanelController.hpp"
#include "elements/DropdownChecklistButton.hpp"
#include "elements/GrowOnlyLabel.hpp"
#include "MenuActionController.hpp"
#include "VolumeAttachmentController.hpp"
#include "FileWatcherService.hpp"
#include "AxisAlignedSliceController.hpp"
#include "SegmentationCommandHandler.hpp"
#include "LasagnaServiceManager.hpp"
#include <vc_object_store/objects/AttachedDocument.hpp>
#include "FiberMapWorkspace.hpp"
#include "SpiralWorkspace.hpp"
#include "ProjectWorkspace.hpp"
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
#include "vc/core/util/Tiff.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <utils/zarr.hpp>





Q_LOGGING_CATEGORY(lcSegGrowth, "vc.segmentation.growth");

using qga = QGuiApplication;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;
namespace
{
constexpr auto WORKSPACE_TAB_SETTING = "mainWin/workspace_tab";
constexpr auto WORKSPACE_ID_SETTING = "mainWin/workspace_id";
constexpr auto MAIN_VIEWER_SPLIT_X_SETTING = "mainWin/main_viewer_split_x";
constexpr auto MAIN_VIEWER_SPLIT_Y_SETTING = "mainWin/main_viewer_split_y";
constexpr auto MAIN_VIEWER_LAYOUT_SURFACES_SETTING = "mainWin/main_viewer_layout_surfaces";
constexpr auto MAIN_VIEWER_LAYOUT_HIDDEN_SETTING = "mainWin/main_viewer_layout_hidden";

VolumeViewerBase* baseViewerFromWidget(QWidget* widget);

class DockMenuMainWindow : public QMainWindow
{
public:
    using DockMenuBuilder = std::function<void(QMenu*)>;

    explicit DockMenuMainWindow(QWidget* parent = nullptr)
        : QMainWindow(parent)
    {
    }

    void setDockMenuBuilder(DockMenuBuilder builder)
    {
        _dockMenuBuilder = std::move(builder);
    }

protected:
    QMenu* createPopupMenu() override
    {
        auto* menu = new QMenu(this);
        if (_dockMenuBuilder) {
            _dockMenuBuilder(menu);
        }
        return menu;
    }

private:
    DockMenuBuilder _dockMenuBuilder;
};

struct MainViewerSpec {
    std::string surfaceName;
    QString title;
    std::set<std::string> intersects;
};

const std::array<const char*, 4> kDefaultMainViewerSurfaces{
    "segmentation",
    "xy plane",
    "seg xz",
    "seg yz",
};

const std::array<bool, 4> kDefaultMainViewerHidden{
    false,
    false,
    true,
    false,
};

const std::array<const char*, 4> kMainViewerPaneLabels{
    "Top left",
    "Top right",
    "Bottom left",
    "Bottom right",
};

std::optional<MainViewerSpec> mainViewerSpecForSurface(const std::string& surfaceName)
{
    if (surfaceName == "segmentation") {
        return MainViewerSpec{surfaceName, QObject::tr("Surface"), {"seg xz", "seg yz"}};
    }
    if (surfaceName == "xy plane") {
        return MainViewerSpec{surfaceName, QObject::tr("XY"), {"segmentation"}};
    }
    if (surfaceName == "seg xz") {
        return MainViewerSpec{surfaceName, QObject::tr("XZ"), {"segmentation"}};
    }
    if (surfaceName == "seg yz") {
        return MainViewerSpec{surfaceName, QObject::tr("YZ"), {"segmentation"}};
    }
    return std::nullopt;
}

QString mainViewerDisplayName(const std::string& surfaceName)
{
    if (surfaceName == "segmentation") {
        return QObject::tr("Surface");
    }
    if (surfaceName == "xy plane") {
        return QObject::tr("XY");
    }
    if (surfaceName == "seg xz") {
        return QObject::tr("XZ");
    }
    if (surfaceName == "seg yz") {
        return QObject::tr("YZ");
    }
    return QString::fromStdString(surfaceName);
}

std::array<std::string, 4> readMainViewerSurfaceLayout()
{
    std::array<std::string, 4> layout{};
    for (int i = 0; i < 4; ++i) {
        layout[i] = kDefaultMainViewerSurfaces[i];
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QStringList saved = settings.value(MAIN_VIEWER_LAYOUT_SURFACES_SETTING).toStringList();
    if (saved.size() != 4) {
        return layout;
    }

    std::set<std::string> seen;
    std::array<std::string, 4> candidate{};
    for (int i = 0; i < 4; ++i) {
        const std::string name = saved[i].toStdString();
        if (!mainViewerSpecForSurface(name) || !seen.insert(name).second) {
            return layout;
        }
        candidate[i] = name;
    }
    return candidate;
}

std::array<bool, 4> readMainViewerHiddenLayout()
{
    std::array<bool, 4> hidden = kDefaultMainViewerHidden;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QStringList saved = settings.value(MAIN_VIEWER_LAYOUT_HIDDEN_SETTING).toStringList();
    if (saved.size() != 4) {
        return hidden;
    }

    int visibleCount = 0;
    for (int i = 0; i < 4; ++i) {
        hidden[i] = saved[i].toInt() != 0;
        if (!hidden[i]) {
            ++visibleCount;
        }
    }
    if (visibleCount == 0) {
        hidden[0] = false;
    }
    return hidden;
}

void persistMainViewerLayout(ViewerSplitGrid* grid)
{
    if (!grid) {
        return;
    }

    QStringList surfaces;
    QStringList hidden;
    for (int i = 0; i < 4; ++i) {
        std::string surfaceName = kDefaultMainViewerSurfaces[i];
        if (auto* viewer = baseViewerFromWidget(grid->viewer(i))) {
            surfaceName = viewer->surfName();
        }
        surfaces.push_back(QString::fromStdString(surfaceName));
        hidden.push_back(grid->paneHidden(i) ? QStringLiteral("1") : QStringLiteral("0"));
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(MAIN_VIEWER_LAYOUT_SURFACES_SETTING, surfaces);
    settings.setValue(MAIN_VIEWER_LAYOUT_HIDDEN_SETTING, hidden);
}

ViewerSplitGrid* mainViewerSplitGrid(QWidget* parent)
{
    auto* widget = parent
        ? parent->findChild<QWidget*>(QStringLiteral("mainViewerSplitGrid"))
        : nullptr;
    return dynamic_cast<ViewerSplitGrid*>(widget);
}

void applyMainViewerLayout(ViewerSplitGrid* grid, ViewerManager* manager)
{
    if (!grid || !manager) {
        return;
    }

    const auto surfaces = readMainViewerSurfaceLayout();
    const auto hidden = readMainViewerHiddenLayout();
    for (int pane = 0; pane < 4; ++pane) {
        QWidget* widget = nullptr;
        for (auto* viewer : manager->baseViewers()) {
            if (!viewer || viewer->surfName() != surfaces[pane]) {
                continue;
            }
            widget = qobject_cast<QWidget*>(viewer->asQObject());
            if (widget && widget->parentWidget() == grid) {
                break;
            }
            widget = nullptr;
        }
        grid->setViewer(pane, widget);
        grid->setPaneHidden(pane, hidden[pane]);
    }
}

class FunctionEventFilter final : public QObject
{
public:
    using Handler = std::function<bool(QObject*, QEvent*)>;

    FunctionEventFilter(Handler handler, QObject* parent)
        : QObject(parent)
        , handler_(std::move(handler))
    {
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        return handler_ ? handler_(watched, event) : false;
    }

private:
    Handler handler_;
};

VolumeViewerBase* baseViewerFromWidget(QWidget* widget)
{
    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(widget)) {
        return chunkedViewer;
    }
    return nullptr;
}

bool moveOnSurfaceChangeEnabled()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                          vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
}

struct SurfaceFocusPoint {
    cv::Vec3f world{0, 0, 0};
    cv::Vec3f ptr{0, 0, 0};
    int row = -1;
    int col = -1;
};

std::optional<SurfaceFocusPoint> focusPointAtGrid(QuadSurface& surface, int row, int col)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || row < 0 || row >= points->rows || col < 0 || col >= points->cols) {
        return std::nullopt;
    }
    if (row <= 0 || col <= 0 || row >= points->rows - 1 || col >= points->cols - 1) {
        return std::nullopt;
    }
    if (!surface.isQuadValid(row, col)) {
        return std::nullopt;
    }
    const cv::Vec3f normal = surface.gridNormal(row, col);
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
        return std::nullopt;
    }

    const cv::Vec3f& point = (*points)(row, col);
    if (!vc::geometry::validSurfacePoint(point)) {
        return std::nullopt;
    }

    const cv::Vec3f center = surface.center();
    const cv::Vec2f scale = surface.scale();
    return SurfaceFocusPoint{
        point,
        cv::Vec3f(static_cast<float>(col) - center[0] * scale[0],
                  static_cast<float>(row) - center[1] * scale[1],
                  0.0f),
        row,
        col,
    };
}

std::optional<SurfaceFocusPoint> findSegmentFocusPoint(QuadSurface& surface)
{
    surface.ensureLoaded();
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    const cv::Vec2f centerGrid = surface.ptrToGrid({0, 0, 0});
    const int centerRow = std::clamp(static_cast<int>(std::lround(centerGrid[1])), 0, points->rows - 1);
    const int centerCol = std::clamp(static_cast<int>(std::lround(centerGrid[0])), 0, points->cols - 1);

    if (auto focus = focusPointAtGrid(surface, centerRow, centerCol)) {
        return focus;
    }

    const int maxHorizontalRadius = std::max(centerCol, points->cols - 1 - centerCol);
    for (int radius = 1; radius <= maxHorizontalRadius; ++radius) {
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol - radius)) {
            return focus;
        }
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol + radius)) {
            return focus;
        }
    }

    const int maxRadius = std::max({centerRow, centerCol,
                                    points->rows - 1 - centerRow,
                                    points->cols - 1 - centerCol});
    for (int radius = 1; radius <= maxRadius; ++radius) {
        const int rowMin = std::max(0, centerRow - radius);
        const int rowMax = std::min(points->rows - 1, centerRow + radius);
        const int colMin = std::max(0, centerCol - radius);
        const int colMax = std::min(points->cols - 1, centerCol + radius);

        for (int col = colMin; col <= colMax; ++col) {
            if (auto focus = focusPointAtGrid(surface, rowMin, col)) {
                return focus;
            }
            if (rowMax != rowMin) {
                if (auto focus = focusPointAtGrid(surface, rowMax, col)) {
                    return focus;
                }
            }
        }
        for (int row = rowMin + 1; row < rowMax; ++row) {
            if (auto focus = focusPointAtGrid(surface, row, colMin)) {
                return focus;
            }
            if (colMax != colMin) {
                if (auto focus = focusPointAtGrid(surface, row, colMax)) {
                    return focus;
                }
            }
        }
    }

    return std::nullopt;
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
                                        const std::string& loadedVolumeId,
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

    // A catalog volume pairs with the normal grids the catalog publishes for
    // its frame (volume and level); other volumes with the first store the
    // catalog does not publish.
    const auto view = pkg->catalogView(loadedVolumeId);
    QString unpublishedFallback;
    for (const auto& path : paths) {
        const auto pathStr = path.string();
        const QString candidateStr = QString::fromStdString(pathStr);
        const vc::project::Entry* owningEntry = nullptr;
        for (const auto& entry : pkg->normalGridEntries()) {
            if (pathStr == entry.location ||
                pathStr.rfind(entry.location + "/", 0) == 0) {
                owningEntry = &entry;
                break;
            }
        }
        const auto published = owningEntry
            ? vc3d::opendata::normalGridsCatalogView(*pkg, *owningEntry)
            : std::nullopt;
        if (view) {
            if (published && published->frameId() == view->frameId()) {
                if (checkedPath) *checkedPath = candidateStr;
                qCInfo(lcSegGrowth) << "Normal grid resolved by catalog frame to" << candidateStr;
                return candidateStr;
            }
            continue;
        }
        if (!published && unpublishedFallback.isEmpty()) {
            unpublishedFallback = candidateStr;
        }
    }

    if (!view && !unpublishedFallback.isEmpty()) {
        if (checkedPath) *checkedPath = unpublishedFallback;
        qCInfo(lcSegGrowth) << "Normal grid resolved to" << unpublishedFallback;
        return unpublishedFallback;
    }

    qCInfo(lcSegGrowth) << "Normal grid lookup: no store paired with volume"
                        << QString::fromStdString(loadedVolumeId);
    return QString();
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

    if (savedSignature.isEmpty() || savedQtVersion.isEmpty()) {
        return false;
    }

    Q_UNUSED(savedAppVersion);
    Q_UNUSED(appVersion);
    return savedSignature == screenSignature
        && savedQtVersion == qtVersion;
}

// Constructor
CWindow::CWindow(size_t cacheSizeGB, bool debugDownloadQueue) :
    _cmdRunner(nullptr),
    _debugDownloadQueue(debugDownloadQueue),
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
    QWidget* segmentCentralWidget = takeCentralWidget();
    auto* segmentWorkspaceWindow = new DockMenuMainWindow(this);
    segmentWorkspaceWindow->setDockMenuBuilder([this](QMenu* menu) {
        populateDockToggleMenu(menu);
    });
    _segmentWorkspaceWindow = segmentWorkspaceWindow;
    _segmentWorkspaceWindow->setObjectName(QStringLiteral("segmentWorkspaceWindow"));
    _segmentWorkspaceWindow->setDockOptions(dockOptions());
    _statusDockPanelHost = new StatusDockPanelHost(segmentCentralWidget, _segmentWorkspaceWindow);
    _segmentWorkspaceWindow->setCentralWidget(_statusDockPanelHost);

    auto moveExistingDockToSegment = [this](QDockWidget* dock, Qt::DockWidgetArea area) {
        if (!dock || !_segmentWorkspaceWindow) {
            return;
        }
        removeDockWidget(dock);
        _segmentWorkspaceWindow->addDockWidget(area, dock);
    };
    moveExistingDockToSegment(ui.dockWidgetVolumes, Qt::LeftDockWidgetArea);
    moveExistingDockToSegment(ui.dockWidgetSegmentation, Qt::RightDockWidgetArea);
    moveExistingDockToSegment(ui.dockWidgetDistanceTransform, Qt::RightDockWidgetArea);
    moveExistingDockToSegment(ui.dockWidgetViewerControls, Qt::LeftDockWidgetArea);
    for (QDockWidget* dock : {ui.dockWidgetNormalVis,
                              ui.dockWidgetView,
                              ui.dockWidgetOverlay,
                              ui.dockWidgetComposite}) {
        moveExistingDockToSegment(dock, Qt::LeftDockWidgetArea);
    }

    _lasagnaWorkspaceWindow = new QMainWindow(this);
    _lasagnaWorkspaceWindow->setObjectName(QStringLiteral("lasagnaWorkspaceWindow"));

    // The real Spiral workspace is installed after Main's CState exists. Keep
    // an inert shell here so it can be appended without shifting legacy tabs.
    _spiralWorkspaceWindow = new QMainWindow(this);
    _spiralWorkspaceWindow->setObjectName(QStringLiteral("spiralWorkspaceShell"));

    _workspaceTabs = new QTabWidget(this);
    _workspaceTabs->setObjectName(QStringLiteral("workspaceTabs"));
    _projectNameLabel = new QLabel(_workspaceTabs);
    _projectNameLabel->setObjectName(QStringLiteral("currentProjectName"));
    _projectNameLabel->setTextFormat(Qt::PlainText);
    _projectNameLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    _projectNameLabel->setContentsMargins(8, 0, 8, 0);
    _workspaceTabs->setCornerWidget(_projectNameLabel, Qt::TopRightCorner);
    _workspaceTabs->setTabsClosable(true);
    _workspaceTabs->addTab(_segmentWorkspaceWindow, tr("main"));
    _workspaceTabs->addTab(_lasagnaWorkspaceWindow, tr("Lasagna"));
    _workspaceTabs->addTab(_spiralWorkspaceWindow, tr("Spiral"));
    _segmentWorkspaceWindow->setProperty("workspaceId", QStringLiteral("main"));
    _lasagnaWorkspaceWindow->setProperty("workspaceId", QStringLiteral("lasagna"));
    _spiralWorkspaceWindow->setProperty("workspaceId", QStringLiteral("spiral"));
    if (auto* tabBar = _workspaceTabs->tabBar()) {
        for (int i = 0; i < _workspaceTabs->count(); ++i) {
            tabBar->setTabButton(i, QTabBar::RightSide, nullptr);
        }
    }
    auto* workspaceContainer = new QWidget(this);
    workspaceContainer->setObjectName(QStringLiteral("workspaceContainer"));
    auto* workspaceLayout = new QVBoxLayout(workspaceContainer);
    workspaceLayout->setContentsMargins(0, 0, 0, 0);
    workspaceLayout->setSpacing(0);
    _persistentCacheWarningBanner = new QFrame(workspaceContainer);
    _persistentCacheWarningBanner->setObjectName(QStringLiteral("persistentCacheWarningBanner"));
    _persistentCacheWarningBanner->setStyleSheet(
        QStringLiteral("QFrame#persistentCacheWarningBanner { background: #fff3cd; color: #664d03; border-bottom: 1px solid #ffda6a; }"));
    auto* warningLayout = new QHBoxLayout(_persistentCacheWarningBanner);
    warningLayout->setContentsMargins(10, 5, 8, 5);
    _persistentCacheWarningText = new QLabel(_persistentCacheWarningBanner);
    _persistentCacheWarningText->setObjectName(QStringLiteral("persistentCacheWarningText"));
    _persistentCacheWarningText->setWordWrap(true);
    warningLayout->addWidget(_persistentCacheWarningText, 1);
    auto* dismissCacheWarning = new QPushButton(tr("Dismiss"), _persistentCacheWarningBanner);
    dismissCacheWarning->setObjectName(QStringLiteral("dismissPersistentCacheWarning"));
    warningLayout->addWidget(dismissCacheWarning);
    connect(dismissCacheWarning, &QPushButton::clicked,
            _persistentCacheWarningBanner, &QWidget::hide);
    _persistentCacheWarningBanner->hide();
    workspaceLayout->addWidget(_persistentCacheWarningBanner);
    workspaceLayout->addWidget(_workspaceTabs, 1);
    setCentralWidget(workspaceContainer);
    connect(_workspaceTabs, &QTabWidget::currentChanged, this, [this]() {
        scheduleWindowStateSave();
        updateActiveWorkspaceViewerControls();
        updateLineAnnotationLengthLabel();
        QTimer::singleShot(0, this, &CWindow::updateProjectNameLabel);
    });
    connect(_workspaceTabs, &QTabWidget::tabCloseRequested, this, [this](int index) {
        if (!_workspaceTabs) {
            return;
        }
        auto* dialog = qobject_cast<LineAnnotationDialog*>(_workspaceTabs->widget(index));
        if (dialog) {
            dialog->close();
        }
    });
    auto* lasagnaEscapeShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), _lasagnaWorkspaceWindow);
    lasagnaEscapeShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(lasagnaEscapeShortcut, &QShortcut::activated, this, &CWindow::switchToMainWorkspace);

    const QString baseTitle = windowTitle();
    const QString repoShortHash = QString::fromStdString(ProjectInfo::RepositoryShortHash()).trimmed();
    if (!repoShortHash.isEmpty() && !repoShortHash.startsWith('@')
        && repoShortHash.compare("Untracked", Qt::CaseInsensitive) != 0) {
        setWindowTitle(QString("%1 %2").arg(baseTitle, repoShortHash));
    }
    // setAttribute(Qt::WA_DeleteOnClose);

    _cacheSizeBytes = cacheSizeGB * 1024ULL * 1024ULL * 1024ULL;
    std::cout << "chunk cache budget is " << cacheSizeGB << " gigabytes" << std::endl;

    vc::render::processChunkCacheService()->configureDecodedByteCapacity(
        _cacheSizeBytes);
    _state = new CState(this, _debugDownloadQueue);
    connect(_state, &CState::vpkgChanged, this, &CWindow::updateProjectNameLabel);
    updateProjectNameLabel();
    connect(_state, &CState::poiChanged, this, &CWindow::onFocusPOIChanged);
    connect(_state, &CState::surfaceWillBeDeleted, this, &CWindow::onSurfaceWillBeDeleted);

    _fileWatcher = std::make_unique<FileWatcherService>(_state, this);
    connect(_fileWatcher.get(), &FileWatcherService::volumeCatalogChanged,
            this, [this](const QString& preferredVolumeId) {
                refreshCurrentVolumePackageUi(preferredVolumeId);
            });

    _axisAlignedSliceController = std::make_unique<AxisAlignedSliceController>(_state, this);

    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
    _viewerManager->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);
    // Let SurfaceCache tile fills serve the flattened view instead of queueing
    // behind raw-path chunk demand for frames that never sample it, as Spiral
    // does.
    _viewerManager->setPreferSurfaceTileFills(true);
    if (_workspaceTabs && _spiralWorkspaceWindow) {
        const int spiralIndex = _workspaceTabs->indexOf(_spiralWorkspaceWindow);
        auto* shell = _spiralWorkspaceWindow;
        _workspaceTabs->removeTab(spiralIndex);
        _spiralWorkspace = new SpiralWorkspace(_state, this);
        _spiralWorkspace->setProperty("workspaceId", QStringLiteral("spiral"));
        _spiralWorkspaceWindow = _spiralWorkspace;
        _workspaceTabs->insertTab(spiralIndex, _spiralWorkspace, tr("Spiral"));
        if (auto* tabBar = _workspaceTabs->tabBar()) tabBar->setTabButton(spiralIndex, QTabBar::RightSide, nullptr);
        shell->deleteLater();
        connectVolumeSelector(_spiralWorkspace->volumeSelectionControl());
        connect(_spiralWorkspace, &SpiralWorkspace::patchEditorRequested, this,
                [this](const QString& id, const QString& path) {
                    try {
                        auto surface = std::make_shared<QuadSurface>(path.toStdString());
                        surface->ensureLoaded();
                        _state->setSurface(id.toStdString(), surface);
                        onSurfaceActivatedPreserveEditing(id, surface);
                        if (_segmentationModule) {
                            _segmentationModule->setEditingEnabled(true);
                            _segmentationModule->beginEditingSession(surface);
                        }
                    } catch (const std::exception& error) {
                        showStatusBarMessage(QString::fromUtf8(error.what()), 15000);
                    }
                });
        connect(_spiralWorkspace->viewerManager(), &ViewerManager::baseViewerCreated,
                this, [this](VolumeViewerBase* viewer) {
                    if (auto* chunked = viewer
                            ? qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())
                            : nullptr)
                        configureChunkedViewerConnections(chunked);
                    if (_fiberCollection && viewer) {
                        _fiberCollection->attachViewer(
                            viewer, _spiralWorkspace->viewerManager());
                    }
                });
        for (auto* viewer : _spiralWorkspace->viewerManager()->baseViewers()) {
            if (auto* chunked = viewer
                    ? qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())
                    : nullptr)
                configureChunkedViewerConnections(chunked);
        }
        // The "Add to current spiral fit" context actions are greyed out
        // unless a Spiral session is active on the connected service.
        connect(_spiralWorkspace, &SpiralWorkspace::spiralSessionActiveChanged, this, [this](bool active) {
            if (_surfacePanel) _surfacePanel->setSpiralFitAvailable(active);
            if (_fiberWidget) _fiberWidget->setSpiralFitAvailable(active);
        });

    }
    _lineAnnotationController = std::make_unique<LineAnnotationController>(_state,
                                                                           _viewerManager.get(),
                                                                           this,
                                                                           this);
    if (_spiralWorkspace)
        _spiralWorkspace->setLineAnnotationController(
            _lineAnnotationController.get());
    _lineAnnotationController->setVolumeSelectorFactory(
        [this](QWidget* parent) { return createAnnotationVolumeSelector(parent); });
    connect(_lineAnnotationController.get(), &LineAnnotationController::volumeOverlayToggleRequested,
            this, &CWindow::toggleVolumeOverlayVisibility);
    connect(_lineAnnotationController.get(),
            &LineAnnotationController::lineAnnotationWorkspaceRequested,
            this,
            &CWindow::openLineAnnotationWorkspace);
    connect(_lineAnnotationController.get(), &LineAnnotationController::fiberRetired,
            this, [this](const QString& path) {
                if (_spiralWorkspace) _spiralWorkspace->noteFiberRetired(path);
            });
    _fiberMapWorkspace = new FiberMapWorkspace(_state, _lineAnnotationController.get(), this);
    _fiberMapWorkspace->setProperty("workspaceId", QStringLiteral("fiber-map"));
    const int fiberMapIndex = _workspaceTabs->addTab(_fiberMapWorkspace, tr("Fiber Map"));
    if (auto* tabBar = _workspaceTabs->tabBar()) {
        tabBar->setTabButton(fiberMapIndex, QTabBar::RightSide, nullptr);
    }
    connect(_fiberMapWorkspace, &FiberMapWorkspace::openFiberAtControlPointRequested,
            this, [this](uint64_t fiberId, int controlPointIndex) {
                if (_fiberWidget) {
                    _fiberWidget->selectFiber(fiberId);
                }
                if (_lineAnnotationController) {
                    _lineAnnotationController->openFiberAtControlPoint(fiberId,
                                                                      controlPointIndex);
                }
            });

    connect(_viewerManager.get(), &ViewerManager::baseViewerCreated, this, [this](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }
        if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
            configureChunkedViewerConnections(chunkedViewer);
        }
    });
    // Automated Fiber Volume picking precedes the shared focus policy.
    _viewerManager->setVolumeClickInterceptor(
        [this](const cv::Vec3f&, const cv::Vec3f&, Surface*,
               Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
            return (_fiberCollection && !(_segmentationModule && _segmentationModule->editingEnabled()) &&
                    _fiberCollection->handleVolumeClick(button, modifiers));
        });
    connect(_viewerManager.get(), &ViewerManager::surfaceActivationRequested,
            this, [this](const std::string& surfaceId) {
                if (_surfacePanel) {
                    _surfacePanel->activateSurfaceById(surfaceId);
                }
            });
    connect(_viewerManager.get(), &ViewerManager::sharedCacheStatsChanged,
            this, &CWindow::onSharedCacheStatsChanged);

    // Left of the cache figures: the length of the fiber being annotated,
    // while a Line Annotation tab is the active workspace.
    _lineAnnotationLengthLabel = new QLabel(this);
    _lineAnnotationLengthLabel->setObjectName(QStringLiteral("lineAnnotationLengthLabel"));
    _lineAnnotationLengthLabel->setContentsMargins(8, 0, 8, 0);
    _lineAnnotationLengthLabel->setToolTip(
        tr("Line length of the fiber open in this Line Annotation tab. "
           "Centimetres when the annotation frame's voxel size is known, "
           "otherwise voxels."));
    _lineAnnotationLengthLabel->hide();
    statusBar()->addPermanentWidget(_lineAnnotationLengthLabel);

    _sharedCacheStatsLabel = new GrowOnlyLabel(this);
    _sharedCacheStatsLabel->setContentsMargins(8, 0, 8, 0);
    _sharedCacheStatsLabel->setToolTip(
        tr("Z-scroll sensitivity: use Shift+G / Shift+H to adjust"));
    statusBar()->addPermanentWidget(_sharedCacheStatsLabel);
    updateSharedStatusLabel();

    _persistentCacheLowSpaceLabel = new QLabel(this);
    _persistentCacheLowSpaceLabel->setObjectName(QStringLiteral("persistentCacheLowSpaceLabel"));
    _persistentCacheLowSpaceLabel->setStyleSheet(QStringLiteral("color: #d28b00; font-weight: 600;"));
    _persistentCacheLowSpaceLabel->hide();
    statusBar()->addPermanentWidget(_persistentCacheLowSpaceLabel);

    _hubStatusLabel = new QLabel(this);
    _hubStatusLabel->setContentsMargins(8, 0, 8, 0);
    _hubStatusLabel->setToolTip(tr("Sync state of the attached hosted object store replica"));
    _hubStatusLabel->hide();
    statusBar()->addPermanentWidget(_hubStatusLabel);
    connect(_state, &CState::storeRoutesChanged, this, &CWindow::updateHubStatus);
    connect(_state, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
        using Kind = vc::objects::StoreSessionEvent::Kind;
        if (event.kind != Kind::Committed && event.kind != Kind::Draft) updateHubStatus();
    });

    _persistentCacheSpaceTimer = new QTimer(this);
    _persistentCacheSpaceTimer->setInterval(5000);
    auto updatePersistentCacheSpace = [this]() {
        auto budget = vc::render::PersistentZarrCacheBudget::findForPath(
            vc3d::remoteCachePathFs());
        if (!budget)
            return;
        const auto stats = budget->stats();
        if (stats.lowSpace) {
            const double freeGiB = static_cast<double>(stats.freeBytes) /
                                   (1024.0 * 1024.0 * 1024.0);
            const QString text = tr("Low disk space: %1 GiB free. Cache cleanup runs asynchronously; downloads continue and cache writes may fail if the disk fills.")
                                     .arg(freeGiB, 0, 'f', 1);
            _persistentCacheLowSpaceLabel->setText(QStringLiteral("⚠ ") + text);
            _persistentCacheLowSpaceLabel->show();
            if (!_persistentCacheBannerShownThisSession) {
                _persistentCacheBannerShownThisSession = true;
                _persistentCacheWarningText->setText(text);
                _persistentCacheWarningBanner->show();
            }
        } else {
            _persistentCacheLowSpaceLabel->hide();
            _persistentCacheWarningBanner->hide();
        }
    };
    connect(_persistentCacheSpaceTimer, &QTimer::timeout, this,
            updatePersistentCacheSpace);
    _persistentCacheSpaceTimer->start();
    updatePersistentCacheSpace();

    _pointsOverlay = std::make_unique<PointsOverlayController>(_state->pointCollection(), this);
    _viewerManager->setPointsOverlay(_pointsOverlay.get());

    _fiberCollection = std::make_unique<FiberCollectionController>(_state, _viewerManager.get(), this, _lineAnnotationController.get());
    updateActiveWorkspaceViewerControls();
    if (_spiralWorkspace && _spiralWorkspace->viewerManager()) {
        _spiralWorkspace->viewerManager()->forEachBaseViewer(
            [this](VolumeViewerBase* viewer) {
                _fiberCollection->attachViewer(
                    viewer, _spiralWorkspace->viewerManager());
            });
    }

    _rawPointsOverlay = std::make_unique<RawPointsOverlayController>(_state, this);
    _viewerManager->setRawPointsOverlay(_rawPointsOverlay.get());

    _pathsOverlay = std::make_unique<PathsOverlayController>(this);
    _viewerManager->setPathsOverlay(_pathsOverlay.get());

    _bboxOverlay = std::make_unique<BBoxOverlayController>(this);
    _viewerManager->setBBoxOverlay(_bboxOverlay.get());

    _vectorOverlay = std::make_unique<VectorOverlayController>(_state, this);
    _viewerManager->setVectorOverlay(_vectorOverlay.get());

    _inkDetectionOverlay = std::make_unique<InkDetectionOverlayController>(_state, this);
    _viewerManager->setInkDetectionOverlay(_inkDetectionOverlay.get());

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
    _volumeOverlay->setViewerManagers(
        {_viewerManager.get(), _spiralWorkspace ? _spiralWorkspace->viewerManager() : nullptr});
    connect(_volumeOverlay.get(), &VolumeOverlayController::requestStatusMessage, this,
            [this](const QString& message, int timeout) {
                if (statusBar()) {
                    showStatusBarMessage(message, timeout);
                }
            });

    if (_statusDockPanelHost) {
        if (auto* statusDockBar = _statusDockPanelHost->takeBarWidget()) {
            statusDockBar->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
            statusBar()->insertWidget(0, statusDockBar, 0);
        }
    }

    _statusMessageLabel = new QLabel(this);
    _statusMessageLabel->setObjectName(QStringLiteral("statusMessageLabel"));
    _statusMessageLabel->setContentsMargins(8, 0, 8, 0);
    _statusMessageLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _statusMessageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    _statusMessageLabel->setMinimumWidth(160);
    _statusMessageLabel->setWordWrap(false);
    statusBar()->addWidget(_statusMessageLabel, 1);

    _statusMessageTimer = new QTimer(this);
    _statusMessageTimer->setSingleShot(true);
    connect(_statusMessageTimer, &QTimer::timeout, this, &CWindow::clearStatusBarMessage);
    connect(statusBar(), &QStatusBar::messageChanged, this, [this](const QString& message) {
        if (_relayingNativeStatusMessage || message.isEmpty()) {
            return;
        }

        _relayingNativeStatusMessage = true;
        showStatusBarMessage(message, 0);
        statusBar()->clearMessage();
        _relayingNativeStatusMessage = false;
    });

    // create UI widgets
    CreateWidgets();

    _volumeAttachmentController =
        std::make_unique<VolumeAttachmentController>(this);

    // create menus/actions controller
    _menuController = std::make_unique<MenuActionController>(this);
    _menuController->populateMenus(menuBar());
    // Wire the Actions -> Merge Surfaces... menu entry to the handler.
    // Has to happen here (not inside CreateWidgets) because
    // _menuController is created after CreateWidgets() returns.
    connect(_menuController.get(), &MenuActionController::mergeSurfacesFromMenuRequested,
            this, [this]() {
                _segmentationCommandHandler->onMergeSurfaces(QStringList{});
            });
    connect(_menuController.get(), &MenuActionController::mergePatchFromMenuRequested,
            this, [this]() {
                _segmentationCommandHandler->onMergePatch(QStringList{});
            });
    connect(_menuController.get(), &MenuActionController::growTrackPatchesFromMenuRequested,
            this, [this]() {
                _segmentationCommandHandler->onGrowTrackPatches();
            });
    connect(_menuController.get(), &MenuActionController::openDataCatalogVisibilityChanged,
            this, [this](bool) {
                updateVolumePackageEmptyState();
            });

    // The Project tab drives the File menu's project actions.
    _state->prepareStoreSwitch = [this]() {
        try { _state->savePointCollections(); }
        catch (const std::exception&) { return false; }
        if (_segmentationModule) {
            if (_segmentationModule->growthInProgress()) return false;
            const auto status = _segmentationModule->autosaveStatus();
            if (status.pending || status.saveInProgress || status.dirtyAfterSave) {
                const auto active = _state->activeSurface().lock();
                if (!active || !_segmentationModule->saveSurface(active.get())) return false;
            }
        }
        return !_lineAnnotationController || _lineAnnotationController->prepareForPackageSwitch();
    };
    _projectWorkspace = new ProjectWorkspace(_state, _menuController.get(), this);
    _projectWorkspace->setProperty("workspaceId", QStringLiteral("project"));
    const int projectIndex = _workspaceTabs->addTab(_projectWorkspace, tr("Project"));
    if (auto* tabBar = _workspaceTabs->tabBar()) tabBar->setTabButton(projectIndex, QTabBar::RightSide, nullptr);
    connect(_projectWorkspace, &ProjectWorkspace::projectRenamed, this, &CWindow::updateProjectNameLabel);
    connect(_projectWorkspace, &ProjectWorkspace::hubSyncRequested, this, &CWindow::promptHubConflicts);
    connect(_projectWorkspace, &ProjectWorkspace::volumeActivationRequested, this, [this](const QString& volumeId) {
        const auto vpkg = _state->vpkg();
        std::shared_ptr<Volume> volume;
        try {
            volume = vpkg ? vpkg->volume(volumeId.toStdString()) : nullptr;
        } catch (const std::out_of_range&) {
        }
        if (!volume) {
            QMessageBox::warning(this, tr("Error"), tr("Could not load volume."));
            return;
        }
        if (_state->currentVolumeId() != volumeId.toStdString()) {
            setVolume(volume);
            syncVolumeSelectionControls(volumeId);
        }
        switchToMainWorkspace();
    });
    connect(ui.btnOpenDataCatalog, &QPushButton::clicked, this, [this]() {
        if (_menuController) {
            _menuController->showOpenDataCatalog();
            updateVolumePackageEmptyState();
        }
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
            restoredState = _segmentWorkspaceWindow && _segmentWorkspaceWindow->restoreState(savedState);
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
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetOverlay,
                               _inkDetectionDock,
                               _transformsDock }) {
        if (dock) {
            dock->setMinimumWidth(250);
            dock->setMinimumHeight(120);
        }
    }
    if (!restoredState) {
        // No saved state - set sensible default sizes for dock widgets
        _segmentWorkspaceWindow->resizeDocks({ui.dockWidgetVolumes}, {300}, Qt::Horizontal);
        _segmentWorkspaceWindow->resizeDocks({ui.dockWidgetVolumes}, {400}, Qt::Vertical);
        _segmentWorkspaceWindow->resizeDocks({ui.dockWidgetSegmentation}, {350}, Qt::Horizontal);
    }

    if (_workspaceTabs) {
        int workspaceIndex = -1;
        const QString workspaceId = geometry.value(WORKSPACE_ID_SETTING).toString();
        if (!workspaceId.isEmpty()) {
            for (int i = 0; i < _workspaceTabs->count(); ++i) {
                if (_workspaceTabs->widget(i)->property("workspaceId").toString() == workspaceId) {
                    workspaceIndex = i;
                    break;
                }
            }
        }
        // An id from a workspace that no longer exists opens main; settings
        // saved before workspace ids fall back to the tab index.
        if (workspaceIndex < 0) {
            workspaceIndex = workspaceId.isEmpty() ? geometry.value(WORKSPACE_TAB_SETTING, 0).toInt() : 0;
        }
        if (workspaceIndex >= 0 && workspaceIndex < _workspaceTabs->count()) {
            _workspaceTabs->setCurrentIndex(workspaceIndex);
        }
    }

    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               _lasagnaDock,
                               ui.dockWidgetDistanceTransform,
                               static_cast<QDockWidget*>(_wrapAnnotationWidget),
                               ui.dockWidgetVolumes,
                               ui.dockWidgetViewerControls,
                               ui.dockWidgetOverlay,
                                _inkDetectionDock,
                               _transformsDock }) {
        ensureDockWidgetFeatures(dock);
        // Connect dock widget signals to trigger state saving
        if (!dock) continue;
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
                                   static_cast<QDockWidget*>(_wrapAnnotationWidget),
                                   ui.dockWidgetVolumes,
                                   ui.dockWidgetViewerControls,
                                   ui.dockWidgetOverlay,
                                   ui.dockWidgetComposite,
                                        _inkDetectionDock,
                                   _transformsDock,
                                   static_cast<QDockWidget*>(_point_collection_widget) }) {
            if (!dock) continue;
            connect(dock, &QDockWidget::topLevelChanged, this, fixGrab);
            connect(dock, &QDockWidget::dockLocationChanged, this, fixGrab);
        }
    }

    if (_statusDockPanelHost) {
        for (QDockWidget* dock : {ui.dockWidgetVolumes,
                                  ui.dockWidgetViewerControls,
                                  ui.dockWidgetComposite,
                                      ui.dockWidgetOverlay,
                                  _inkDetectionDock,
                                  _transformsDock,
                                  ui.dockWidgetSegmentation,
                                  _lasagnaDock,
                                  ui.dockWidgetDistanceTransform,
                                  static_cast<QDockWidget*>(_point_collection_widget),
                                  static_cast<QDockWidget*>(_wrapAnnotationWidget),
                                  static_cast<QDockWidget*>(_fiberWidget)}) {
            _statusDockPanelHost->addDock(dock);
        }
    }

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }
    if (!restoredGeometry) {
        setWindowState(windowState() | Qt::WindowMaximized);
    }

    bool scheduledStartupAction = false;

    // If enabled, auto open the last used local volume package.
    if (settings.value(vc3d::settings::project::AUTO_OPEN, vc3d::settings::project::AUTO_OPEN_DEFAULT).toInt() != 0) {

        QStringList files = settings.value(vc3d::settings::project::RECENT).toStringList();
        files.removeIf([](const QString& path) { return !path.endsWith(".vcproj", Qt::CaseInsensitive); });

        if (!files.empty() && !files.at(0).isEmpty()) {
            QString path = files[0];
            scheduledStartupAction = true;
            QTimer::singleShot(0, this, [this, path]() {
                if (_menuController) {
                    _menuController->openVolpkgAt(path);
                }
            });
        }
    }

    if (!scheduledStartupAction &&
        settings.value(vc3d::settings::project::SHOW_OPEN_DATA_CATALOG_ON_STARTUP,
                       vc3d::settings::project::SHOW_OPEN_DATA_CATALOG_ON_STARTUP_DEFAULT).toBool()) {
        QTimer::singleShot(0, this, [this]() {
            if (_menuController && !_state->hasVpkg()) {
                _menuController->showOpenDataCatalog();
                updateVolumePackageEmptyState();
            }
        });
    }

    // Create application-wide keyboard shortcuts
    fCompositeViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CompositeView), this);
    fCompositeViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCompositeViewShortcut, &QShortcut::activated, [this]() {
        if (_viewerCompositePanel) {
            _viewerCompositePanel->toggleSegmentationComposite();
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
        for (auto* manager : ViewerManager::allManagers()) {
            manager->setShowSurfaceNormals(next);
        }
        showStatusBarMessage(next ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
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
            showStatusBarMessage(
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

    // Zoom-to-fit shortcut (Ctrl+0): fit the current segment in the flattened
    // segmentation viewer, adjusting pan and zoom to its full extent.
    fZoomToFitShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ZoomToFit), this);
    fZoomToFitShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomToFitShortcut, &QShortcut::activated, [this]() {
        VolumeViewerBase* viewer = _viewerManager ? _viewerManager->segmentationViewer() : nullptr;
        if (viewer) {
            viewer->resetViewForCurrentContent(true);
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

    fApplyApprovedTagShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ApplyApprovedTag), this);
    fApplyApprovedTagShortcut->setContext(Qt::ApplicationShortcut);
    connect(fApplyApprovedTagShortcut, &QShortcut::activated, [this]() {
        if (_surfacePanel &&
            _surfacePanel->setTagChecked(SurfacePanelController::Tag::Approved, true)) {
            showStatusBarMessage(tr("Applied Approved tag"), 2000);
        }
    });

    fApplyDefectiveTagShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ApplyDefectiveTag), this);
    fApplyDefectiveTagShortcut->setContext(Qt::ApplicationShortcut);
    connect(fApplyDefectiveTagShortcut, &QShortcut::activated, [this]() {
        if (_surfacePanel &&
            _surfacePanel->setTagChecked(SurfacePanelController::Tag::Defective, true)) {
            showStatusBarMessage(tr("Applied Defective tag"), 2000);
        }
    });

    // Focused view toggle (Shift+Ctrl+F) - hides dock widgets, keeps all viewers
    fFocusedViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::FocusedView), this);
    fFocusedViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fFocusedViewShortcut, &QShortcut::activated, this, &CWindow::toggleFocusedView);

    fOpenLasagnaWorkspaceShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+L")), this);
    fOpenLasagnaWorkspaceShortcut->setContext(Qt::ApplicationShortcut);
    connect(fOpenLasagnaWorkspaceShortcut, &QShortcut::activated,
            this, &CWindow::switchToLasagnaWorkspace);

    fRepeatLasagnaActionShortcut = new QShortcut(QKeySequence(QStringLiteral("Ctrl+Shift+L")), this);
    fRepeatLasagnaActionShortcut->setContext(Qt::ApplicationShortcut);
    connect(fRepeatLasagnaActionShortcut, &QShortcut::activated,
            this, &CWindow::repeatLastLasagnaAction);
}

// Destructor
CWindow::~CWindow()
{
    _destroyingWindow = true;
    if (qApp) {
        qApp->removeEventFilter(this);
    }

    if (_fileWatcher) {
        _fileWatcher->stopWatching();
    }
    setStatusBar(nullptr);

    if (_lineAnnotationController) {
        _lineAnnotationController->saveOpenFibers();
    }
    CloseVolume();
}

void CWindow::populateDockToggleMenu(QMenu* menu) const
{
    if (!menu) {
        return;
    }

    auto addDock = [](QMenu* targetMenu, QDockWidget* dock) {
        if (dock) {
            targetMenu->addAction(dock->toggleViewAction());
        }
    };

    addDock(menu, ui.dockWidgetVolumes);
    addDock(menu, ui.dockWidgetSegmentation);
    addDock(menu, ui.dockWidgetDistanceTransform);
    addDock(menu, ui.dockWidgetViewerControls);
    addDock(menu, ui.dockWidgetOverlay);
    addDock(menu, _inkDetectionDock);
    addDock(menu, _transformsDock);
    addDock(menu, ui.dockWidgetComposite);
    addDock(menu, _lasagnaDock);

    addDock(menu, _point_collection_widget);
    addDock(menu, _wrapAnnotationWidget);

    if (_fiberWidget) {
        auto* fiberMenu = menu->addMenu(tr("Fibers"));
        addDock(fiberMenu, _fiberWidget);
    }
}

VolumeViewerBase* CWindow::newConnectedViewerInWidget(std::string surfaceName, QString title, QWidget* parent)
{
    if (!_viewerManager || !parent) {
        return nullptr;
    }

    VolumeViewerBase* viewer = _viewerManager->createViewerInWidget(surfaceName, parent);
    if (!viewer) {
        return nullptr;
    }

    if (auto* viewerObject = viewer->asQObject())
        viewerObject->setProperty("vc_viewer_label", title);

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

    // Volume-change propagation, cache-stats aggregation, the default
    // volume-click policy and active-viewer tracking are wired by
    // ViewerManager::initializeChunkedViewer for every workspace.
    const bool annotationViewer = viewer->property("vc_viewer_role").toString() == QStringLiteral("annotation");
    const bool mainViewer = _viewerManager &&
        std::find(_viewerManager->baseViewers().begin(),
                  _viewerManager->baseViewers().end(), viewer) !=
            _viewerManager->baseViewers().end();

    if (annotationViewer && !viewer->property("vc_annotation_focus_bound").toBool()) {
        const std::string surfaceName = viewer->surfName();
        const bool lineFocusViewer = surfaceName.rfind("line-", 0) == 0 ||
                                     surfaceName.rfind("line_annotation_slice_", 0) == 0;
        if (lineFocusViewer) {
            connect(viewer,
                    &CChunkedVolumeViewer::sendVolumeClicked,
                    this,
                    [this](cv::Vec3f volLoc,
                           cv::Vec3f normal,
                           Surface*,
                           Qt::MouseButton button,
                           Qt::KeyboardModifiers modifiers) {
                        if (button != Qt::LeftButton || !modifiers.testFlag(Qt::ControlModifier)) {
                            return;
                        }
                        if (centerFocusAt(volLoc, normal, "segmentation")) {
                            switchToMainWorkspace();
                            raise();
                            activateWindow();
                        }
                    });
            viewer->setProperty("vc_annotation_focus_bound", true);
        }
    }

    if (auto* graphicsView = viewer->graphicsView()) {
        if (!viewer->property("vc_annotation_context_bound").toBool()) {
            connect(graphicsView,
                    &CVolumeViewerView::sendAnnotationContextMenuRequested,
                    this,
                    [this, viewer](QPointF scenePoint, QPoint globalPos, Qt::KeyboardModifiers) {
                        if (_lineAnnotationController &&
                            _lineAnnotationController->showGeneratedControlPointContextMenu(
                                viewer,
                                scenePoint,
                                globalPos)) {
                            return;
                        }

                        const cv::Vec3f volumePoint = viewer->sceneToVolume(scenePoint);
                        const bool validVolumePoint =
                            std::isfinite(volumePoint[0]) &&
                            std::isfinite(volumePoint[1]) &&
                            std::isfinite(volumePoint[2]);
                        const int seedX = validVolumePoint
                            ? static_cast<int>(std::lround(volumePoint[0]))
                            : 0;
                        const int seedY = validVolumePoint
                            ? static_cast<int>(std::lround(volumePoint[1]))
                            : 0;
                        const int seedZ = validVolumePoint
                            ? static_cast<int>(std::lround(volumePoint[2]))
                            : 0;

                        QPointer<SegmentationLasagnaPanel> lasagnaPanel = _segmentationWidget
                            ? _segmentationWidget->lasagnaPanel()
                            : nullptr;

                        const auto activeSurface = _state ? _state->surface("segmentation") : nullptr;
                        const bool activeSegmentHasLasagnaModel =
                            activeSurface && LasagnaServiceManager::instance().surfaceModel(activeSurface->id);

                        QMenu menu(this);

                        if (_fiberCollection && _lineAnnotationController) {
                            if (const auto hit = _fiberCollection->hitTestControlPoint(
                                    viewer, scenePoint, 12.0);
                                hit && hit->fiberId != 0) {
                                const uint64_t fiberId = hit->fiberId;
                                const int controlIndex = hit->controlPointIndex;
                                QAction* gotoControlPointAction = menu.addAction(
                                    tr("Go to control point %1 in %2")
                                        .arg(controlIndex)
                                        .arg(_lineAnnotationController->fiberDisplayName(fiberId)));
                                connect(gotoControlPointAction, &QAction::triggered, this,
                                        [this, fiberId, controlIndex]() {
                                            if (_fiberWidget) {
                                                _fiberWidget->selectFiber(fiberId);
                                            }
                                            if (_lineAnnotationController) {
                                                _lineAnnotationController->openFiberAtControlPoint(
                                                    fiberId, controlIndex);
                                            }
                                        });
                                menu.addSeparator();
                            }
                        }

                        if (_spiralWorkspace &&
                            _spiralWorkspace->isFlattenedViewer(viewer)) {
                            QAction* spiralLineAction =
                                menu.addAction(tr("2D line annotation"));
                            spiralLineAction->setObjectName(
                                QStringLiteral("spiral2dLineAnnotationAction"));
                            const QString reason = _spiralWorkspace
                                                       ->lineAnnotationDraftUnavailableReason();
                            spiralLineAction->setEnabled(reason.isEmpty());
                            spiralLineAction->setToolTip(reason);
                            connect(spiralLineAction, &QAction::triggered, this,
                                    [this]() {
                                        if (_spiralWorkspace && _menuController &&
                                            _menuController->requireProject(tr("Creating a fiber")))
                                            _spiralWorkspace->startLineAnnotationDraft();
                                    });
                            menu.addSeparator();
                        }

                        // Enabled without a store: choosing it offers to create or attach one.
                        QAction* newLineAnnotationAction = menu.addAction(tr("New line annotation"));
                        newLineAnnotationAction->setEnabled(
                            _lineAnnotationController &&
                            _lineAnnotationController->viewerSupportsLaunch(viewer) &&
                            viewer->sampleSceneVolume(scenePoint).has_value());

                        QAction* modelLineAnnotationAction = menu.addAction(tr("New 2D line annotation (model)"));
                        modelLineAnnotationAction->setObjectName(QStringLiteral("model2dLineAnnotationAction"));
                        modelLineAnnotationAction->setEnabled(newLineAnnotationAction->isEnabled());

                        QAction* createGrowPatchAction = menu.addAction(tr("Create Segment (GrowPatch)"));
                        createGrowPatchAction->setEnabled(validVolumePoint &&
                                                          _segmentationCommandHandler &&
                                                          _state &&
                                                          _state->vpkg() &&
                                                          _state->currentVolume());
                        connect(createGrowPatchAction, &QAction::triggered, this,
                                [this, volumePoint]() {
                                    if (!_segmentationCommandHandler) {
                                        return;
                                    }
                                    _segmentationCommandHandler->onCreateSegmentGrowPatchFromSeed(
                                        QVector3D(volumePoint[0], volumePoint[1], volumePoint[2]));
                                });

                        menu.addSeparator();

                        auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject());
                        auto* viewerGrid = mainViewerSplitGrid(ui.tabSegment);
                        bool mainGrid = viewerGrid && viewerGrid->indexOf(viewerWidget) >= 0;
                        if (!mainGrid && _spiralWorkspace && _spiralWorkspace->viewerGrid())
                            viewerGrid = _spiralWorkspace->viewerGrid();
                        const int viewerPane = viewerGrid ? viewerGrid->indexOf(viewerWidget) : -1;
                        if (viewerGrid && viewerPane >= 0) {
                            const bool fullSizeActive = viewerGrid->fullSizeActive();
                            QAction* fullSizeAction = menu.addAction(
                                viewerGrid->fullSizeActiveForPane(viewerPane)
                                    ? tr("Exit full size")
                                    : tr("Full size viewer"));
                            connect(fullSizeAction, &QAction::triggered, this,
                                    [viewerGrid, viewerPane]() {
                                        if (viewerGrid->fullSizeActiveForPane(viewerPane)) {
                                            viewerGrid->exitFullSize();
                                        } else {
                                            viewerGrid->setFullSizePane(viewerPane);
                                        }
                                    });

                            QAction* closeAction = menu.addAction(tr("Close viewer"));
                            closeAction->setEnabled(!fullSizeActive && !viewerGrid->paneHidden(viewerPane));
                            connect(closeAction, &QAction::triggered, this, [viewerGrid, viewerPane, mainGrid]() {
                                viewerGrid->setPaneHidden(viewerPane, true);
                                if (mainGrid) persistMainViewerLayout(viewerGrid);
                            });

                            auto* moveMenu = menu.addMenu(tr("Move viewer to"));
                            for (int pane = 0; pane < 4; ++pane) {
                                const QString paneLabel = QObject::tr(kMainViewerPaneLabels[pane]);
                                QAction* paneAction = moveMenu->addAction(paneLabel);
                                paneAction->setEnabled(!fullSizeActive && pane != viewerPane);
                                connect(paneAction, &QAction::triggered, this,
                                        [viewerGrid, viewerPane, pane, mainGrid]() {
                                            const bool targetWasHidden = viewerGrid->paneHidden(pane);
                                            viewerGrid->swapViewers(viewerPane, pane);
                                            if (targetWasHidden) {
                                                viewerGrid->setPaneHidden(pane, false);
                                                viewerGrid->setPaneHidden(viewerPane, true);
                                            }
                                            if (mainGrid) persistMainViewerLayout(viewerGrid);
                                        });
                            }

                            auto* showMenu = menu.addMenu(tr("Show closed viewer here"));
                            bool hasClosedViewer = false;
                            for (int pane = 0; pane < 4; ++pane) {
                                if (!viewerGrid->paneHidden(pane) || !viewerGrid->viewer(pane)) {
                                    continue;
                                }
                                auto* hiddenViewer = baseViewerFromWidget(viewerGrid->viewer(pane));
                                const QString label = hiddenViewer
                                    ? mainViewerDisplayName(hiddenViewer->surfName())
                                    : QObject::tr(kMainViewerPaneLabels[pane]);
                                QAction* showAction = showMenu->addAction(label);
                                hasClosedViewer = true;
                                connect(showAction, &QAction::triggered, this,
                                        [viewerGrid, viewerPane, pane, mainGrid]() {
                                            viewerGrid->swapViewers(viewerPane, pane);
                                            viewerGrid->setPaneHidden(viewerPane, false);
                                            viewerGrid->setPaneHidden(pane, true);
                                            if (mainGrid) persistMainViewerLayout(viewerGrid);
                                        });
                            }
                            showMenu->setEnabled(!fullSizeActive && hasClosedViewer);
                            menu.addSeparator();
                        }

                        QPointer<QMdiSubWindow> subWindow = viewerWidget
                            ? qobject_cast<QMdiSubWindow*>(viewerWidget->parentWidget())
                            : nullptr;
                        if (subWindow) {
                            QAction* maximizeAction = menu.addAction(
                                subWindow->isMaximized() ? tr("Restore viewer") : tr("Maximize viewer"));
                            connect(maximizeAction, &QAction::triggered, this, [subWindow]() {
                                if (!subWindow)
                                    return;
                                if (subWindow->isMaximized())
                                    subWindow->showNormal();
                                else
                                    subWindow->showMaximized();
                            });

                            QAction* closeAction = menu.addAction(tr("Close viewer"));
                            connect(closeAction, &QAction::triggered, this, [subWindow]() {
                                if (subWindow)
                                    subWindow->close();
                            });
                            menu.addSeparator();
                        }

                        if (viewer->measurementSupported()) {
                            QAction* measureAction = menu.addAction(tr("Measure"));
                            connect(measureAction, &QAction::triggered, this, [viewer]() {
                                viewer->startMeasurementMode();
                            });
                            menu.addSeparator();
                        }

                        auto addLasagnaLaunchAction =
                            [this, lasagnaPanel, &menu, seedX, seedY, seedZ, validVolumePoint,
                             activeSegmentHasLasagnaModel](
                                const QString& verb,
                                SegmentationLasagnaPanel::LasagnaMode mode,
                                bool requiresLasagnaModel,
                                bool showSeedInLabel) {
                                const QString configPath = lasagnaPanel
                                    ? lasagnaPanel->selectedLasagnaConfigPathForMode(mode)
                                    : QString();
                                const QString configName = QFileInfo(configPath).fileName();
                                const QString seedText = tr("%1,%2,%3").arg(seedX).arg(seedY).arg(seedZ);
                                QString label;
                                if (configName.isEmpty()) {
                                    label = showSeedInLabel
                                        ? tr("%1 l3d (%2)").arg(verb, seedText)
                                        : tr("%1 l3d").arg(verb);
                                } else {
                                    label = showSeedInLabel
                                        ? tr("%1 l3d (%2 %3)").arg(verb, configName, seedText)
                                        : tr("%1 l3d (%2)").arg(verb, configName);
                                }
                                QAction* launchAction = menu.addAction(label);
                                launchAction->setEnabled(
                                    validVolumePoint &&
                                    !configPath.isEmpty() &&
                                    (!requiresLasagnaModel || activeSegmentHasLasagnaModel));
                                launchAction->setToolTip(configPath);
                                connect(launchAction, &QAction::triggered, this,
                                        [this, lasagnaPanel, mode, configPath, seedX, seedY, seedZ]() {
                                            if (!lasagnaPanel) {
                                                return;
                                            }
                                            lasagnaPanel->startOptimizationAtSeed(
                                                _state,
                                                statusBar(),
                                                mode,
                                                configPath,
                                                seedX,
                                                seedY,
                                                seedZ);
                                        });
                            };

                        addLasagnaLaunchAction(
                            tr("new"), SegmentationLasagnaPanel::LasagnaMode::NewModel, false, true);
                        addLasagnaLaunchAction(
                            tr("reopt"), SegmentationLasagnaPanel::LasagnaMode::ReOptimize, true, false);

                        auto* configMenu = menu.addMenu(
                            tr("l3d config (%1,%2,%3)").arg(seedX).arg(seedY).arg(seedZ));
                        configMenu->setEnabled(validVolumePoint && lasagnaPanel);
                        if (lasagnaPanel) {
                            QStringList configs = lasagnaPanel->lasagnaConfigPathsForMode(
                                SegmentationLasagnaPanel::LasagnaMode::NewModel);
                            const QStringList reoptConfigs = lasagnaPanel->lasagnaConfigPathsForMode(
                                SegmentationLasagnaPanel::LasagnaMode::ReOptimize);
                            for (const QString& configPath : reoptConfigs) {
                                if (!configs.contains(configPath)) {
                                    configs.append(configPath);
                                }
                            }
                            if (configs.isEmpty()) {
                                QAction* none = configMenu->addAction(tr("No config selected"));
                                none->setEnabled(false);
                            }
                            for (const QString& configPath : configs) {
                                QAction* configAction =
                                    configMenu->addAction(QFileInfo(configPath).fileName());
                                configAction->setToolTip(configPath);
                                connect(configAction, &QAction::triggered, this,
                                        [this, lasagnaPanel, configPath, seedX, seedY, seedZ]() {
                                            if (!lasagnaPanel) {
                                                return;
                                            }
                                            lasagnaPanel->startOptimizationAtSeed(
                                                _state,
                                                statusBar(),
                                                SegmentationLasagnaPanel::LasagnaMode::NewModel,
                                                configPath,
                                                seedX,
                                                seedY,
                                                seedZ);
                                        });
                            }
                        }

                        QAction* selected = menu.exec(globalPos);
                        const bool launch = (selected == newLineAnnotationAction ||
                                             selected == modelLineAnnotationAction) &&
                                            selected->isEnabled() && _lineAnnotationController &&
                                            _menuController &&
                                            _menuController->requireProject(tr("Creating a fiber"));
                        if (launch) {
                            _lineAnnotationController->launchFromViewerAtPoint(
                                viewer, scenePoint, true, selected == modelLineAnnotationAction);
                        }
                    });
            viewer->setProperty("vc_annotation_context_bound", true);
        }

        if (!annotationViewer) {
            connect(graphicsView, &CVolumeViewerView::sendMousePress,
                    viewer, &CChunkedVolumeViewer::onMousePress, Qt::UniqueConnection);
            connect(graphicsView, &CVolumeViewerView::sendMouseMove,
                    viewer, &CChunkedVolumeViewer::onMouseMove, Qt::UniqueConnection);
            connect(graphicsView, &CVolumeViewerView::sendMouseRelease,
                    viewer, &CChunkedVolumeViewer::onMouseRelease, Qt::UniqueConnection);
        }
    }

    if (annotationViewer) {
        return;
    }
    if (!mainViewer) {
        // The menu and direct viewer input above are shared. Editing widgets
        // below intentionally remain bound only to Main's state and manager.
        return;
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
        connect(_point_collection_widget, &CPointCollectionWidget::collectionSelectionCleared,
                viewer, &CChunkedVolumeViewer::clearCollectionSelection, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendCollectionSelected,
                _point_collection_widget, &CPointCollectionWidget::selectCollection, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelected,
                viewer, &CChunkedVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelectionCleared,
                viewer, &CChunkedVolumeViewer::clearPointSelection, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointSelected,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointClicked,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        viewer->setProperty("vc_points_bound", true);
    }

    if (_wrapAnnotationWidget && !viewer->property("vc_wrap_annotation_bound").toBool()) {
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::collectionSelected,
                viewer, &CChunkedVolumeViewer::onCollectionSelected, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::collectionSelectionCleared,
                viewer, &CChunkedVolumeViewer::clearCollectionSelection, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::sendCollectionSelected,
                _wrapAnnotationWidget, &WrapAnnotationWidget::selectCollection, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::pointSelected,
                viewer, &CChunkedVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::pointSelectionCleared,
                viewer, &CChunkedVolumeViewer::clearPointSelection, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointSelected,
                _wrapAnnotationWidget, &WrapAnnotationWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CChunkedVolumeViewer::pointClicked,
                _wrapAnnotationWidget, &WrapAnnotationWidget::selectPoint, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationToggled,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationMode, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationSpacingChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationSpacing, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationPolylineOpacityChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationPolylineOpacity, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationMergeToggled,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationMergeExisting, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationPathTypeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationPathType, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationFilterTypeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationFilterType, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationFilterKernelSizeChanged,
                viewer, &CChunkedVolumeViewer::setSameWrapAnnotationFilterKernelSize, Qt::UniqueConnection);
        connect(_wrapAnnotationWidget, &WrapAnnotationWidget::sameWrapAnnotationClearRequested,
                viewer, &CChunkedVolumeViewer::clearSameWrapAnnotationPreview, Qt::UniqueConnection);
        viewer->setSameWrapAnnotationSpacing(_wrapAnnotationWidget->sameWrapAnnotationSpacing());
        viewer->setSameWrapAnnotationPolylineOpacity(_wrapAnnotationWidget->sameWrapAnnotationPolylineOpacity());
        viewer->setSameWrapAnnotationMergeExisting(_wrapAnnotationWidget->sameWrapAnnotationMergeEnabled());
        viewer->setSameWrapAnnotationPathType(_wrapAnnotationWidget->sameWrapAnnotationPathType());
        viewer->setSameWrapAnnotationFilterKernelSize(_wrapAnnotationWidget->sameWrapAnnotationFilterKernelSize());
        viewer->setSameWrapAnnotationFilterType(_wrapAnnotationWidget->sameWrapAnnotationFilterType());
        viewer->setSameWrapAnnotationMode(_wrapAnnotationWidget->sameWrapAnnotationEnabled());
        viewer->setProperty("vc_wrap_annotation_bound", true);
    }

    const std::string& surfName = viewer->surfName();
    if ((surfName == "xy plane" || surfName == "seg xz" || surfName == "seg yz") &&
        !viewer->property("vc_volcam_bound").toBool()) {
        // Slice views fold their volumetric-camera azimuth into the slice
        // plane itself (so sampling, intersections, focus and handles all
        // agree); reconfigure the planes whenever the per-view camera (or
        // anything that implicitly zeroes it) changes.
        viewer->setVolumetricAzimuthInSurface(true);
        connect(viewer, &CChunkedVolumeViewer::compositeCameraChanged, this, [this]() {
            if (_axisAlignedSliceController) {
                _axisAlignedSliceController->syncVolumetricAzimuths();
            }
        });
        viewer->setProperty("vc_volcam_bound", true);
    }
    // Axis-aligned rotation/tilt wiring is attached per viewer by
    // AxisAlignedSliceController via ViewerManager::baseViewerCreated.
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
    auto* manager = activeWorkspaceViewerManager();
    if (!manager) {
        return nullptr;
    }

    if (auto* viewer = manager->activeViewer()) {
        return viewer;
    }

    if (manager != _viewerManager.get()) {
        QWidget* focus = QApplication::focusWidget();
        for (auto* viewer : manager->baseViewers()) {
            auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
            if (widget && focus && (widget == focus || widget->isAncestorOf(focus))) {
                return viewer;
            }
        }
        for (auto* viewer : manager->baseViewers()) {
            auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
            if (widget && widget->isVisible() && widget->underMouse()) {
                return viewer;
            }
        }
        for (auto* viewer : manager->baseViewers()) {
            auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
            if (widget && widget->isVisible()) {
                return viewer;
            }
        }
        return nullptr;
    }

    if (!mdiArea) {
        return nullptr;
    }
    auto* subWindow = mdiArea->activeSubWindow();
    if (!subWindow) {
        return nullptr;
    }
    return baseViewerFromWidget(subWindow->widget());
}

ViewerManager* CWindow::activeWorkspaceViewerManager() const
{
    if (_workspaceTabs && _spiralWorkspace &&
        _workspaceTabs->currentWidget() == _spiralWorkspace) {
        return _spiralWorkspace->viewerManager();
    }
    return _viewerManager.get();
}

void CWindow::updateActiveWorkspaceViewerControls()
{
    const bool lineAnnotationActive =
        _workspaceTabs && qobject_cast<LineAnnotationDialog*>(_workspaceTabs->currentWidget());
    if (_fiberCollection) {
        _fiberCollection->setLineAnnotationActive(lineAnnotationActive);
    }
    if (_fiberWidget) {
        _fiberWidget->setLineAnnotationActive(lineAnnotationActive);
    }
    auto* manager = activeWorkspaceViewerManager();
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewerManager(manager);
    }
    // Composite and volume-overlay controls remain bound to both managers.
    // Only navigation and other workspace-local controls follow the active tab.
}

void CWindow::updateProjectNameLabel()
{
    if (!_projectNameLabel || !_workspaceTabs) return;
    const auto project = _state ? _state->vpkg() : nullptr;
    if (!project) {
        _projectNameLabel->clear();
        _projectNameLabel->setToolTip({});
        _projectNameLabel->setFixedWidth(0);
        return;
    }
    const QString name = QString::fromStdString(project->name());
    _projectNameLabel->setToolTip(QString::fromStdString(project->path().string()));
    const int available = std::max(0, _workspaceTabs->width() -
        _workspaceTabs->tabBar()->sizeHint().width() - 8);
    const auto metrics = _projectNameLabel->fontMetrics();
    _projectNameLabel->setFixedWidth(std::min(available, metrics.horizontalAdvance(name) + 16));
    _projectNameLabel->setText(metrics.elidedText(
        name, Qt::ElideMiddle, std::max(0, _projectNameLabel->width() - 16)));
}

void CWindow::resetSegmentationViews(bool persistLayout)
{
    auto* viewerGrid = mainViewerSplitGrid(ui.tabSegment);
    if (!viewerGrid) {
        return;
    }

    auto ensureDefaultViewer = [this, viewerGrid](const std::string& surfaceName,
                                                  const QString& title,
                                                  const std::set<std::string>& intersects) -> VolumeViewerBase* {
        if (_viewerManager) {
            for (auto* viewer : _viewerManager->baseViewers()) {
                auto* viewerWidget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr;
                if (viewer && viewerWidget && viewer->surfName() == surfaceName && viewerWidget->parentWidget() == viewerGrid) {
                    return viewer;
                }
            }
        }

        auto* viewer = newConnectedViewerInWidget(surfaceName, title, viewerGrid);
        if (viewer) {
            viewer->setIntersects(intersects);
        }
        return viewer;
    };

    auto* surfaceViewer = ensureDefaultViewer("segmentation", tr("Surface"), {"seg xz", "seg yz"});
    auto* xyViewer = ensureDefaultViewer("xy plane", tr("XY"), {"segmentation"});
    auto* xzViewer = ensureDefaultViewer("seg xz", tr("XZ"), {"segmentation"});
    auto* yzViewer = ensureDefaultViewer("seg yz", tr("YZ"), {"segmentation"});

    viewerGrid->setViewer(0, surfaceViewer ? qobject_cast<QWidget*>(surfaceViewer->asQObject()) : nullptr);
    viewerGrid->setViewer(1, xyViewer ? qobject_cast<QWidget*>(xyViewer->asQObject()) : nullptr);
    viewerGrid->setViewer(2, xzViewer ? qobject_cast<QWidget*>(xzViewer->asQObject()) : nullptr);
    viewerGrid->setViewer(3, yzViewer ? qobject_cast<QWidget*>(yzViewer->asQObject()) : nullptr);

    for (auto* viewer : {surfaceViewer, xzViewer, xyViewer, yzViewer}) {
        if (auto* widget = viewer ? qobject_cast<QWidget*>(viewer->asQObject()) : nullptr) {
            widget->show();
        }
    }

    viewerGrid->setPaneHidden(0, kDefaultMainViewerHidden[0]);
    viewerGrid->setPaneHidden(1, kDefaultMainViewerHidden[1]);
    viewerGrid->setPaneHidden(2, kDefaultMainViewerHidden[2]);
    viewerGrid->setPaneHidden(3, kDefaultMainViewerHidden[3]);
    viewerGrid->resetSplits();
    if (persistLayout) {
        persistMainViewerLayout(viewerGrid);
    }
}

void CWindow::clearSurfaceSelection()
{
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
}

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    const bool hadVolume = static_cast<bool>(_state->currentVolume());
    const std::string previousVolumeId = _state ? _state->currentVolumeId() : std::string{};
    std::string targetVolumeId;
    if (_state && _state->vpkg() && newvol) {
        for (const auto& id : _state->vpkg()->volumeIDs()) {
            if (_state->vpkg()->volume(id) == newvol) {
                targetVolumeId = id;
                break;
            }
        }
    }
    if (targetVolumeId.empty() && newvol) {
        targetVolumeId = newvol->id();
    }

    const auto navigationTransform =
        (hadVolume && newvol && previousVolumeId != targetVolumeId)
            ? volumeTransformForSwitch(previousVolumeId, targetVolumeId)
            : std::optional<vcr::catalog::Affine>{};

    // Switch both independent states before revalidating the canonical overlay.
    // This prevents an incompatible base-volume transition from clearing only
    // one viewer fleet while the other still reports the previous coordinates.
    if (_volumeOverlay) _volumeOverlay->beginBaseVolumeUpdate();
    if (_viewerManager) {
        _viewerManager->switchVolume(newvol, navigationTransform);
    } else {
        _state->setCurrentVolume(newvol);
    }
    if (_spiralWorkspace) {
        _spiralWorkspace->synchronizeVolume(newvol, navigationTransform);
    }
    if (_volumeOverlay) _volumeOverlay->endBaseVolumeUpdate();
    if (_state->currentVolume() && !_state->currentVolumeId().empty()) {
        rememberCurrentVolumeForPackage(QString::fromStdString(_state->currentVolumeId()));
    }

    const bool growthVolumeValid = _state->hasVpkg() && !_state->segmentationGrowthVolumeId().empty() &&
                                   _state->vpkg()->hasVolume(_state->segmentationGrowthVolumeId());
    if (!growthVolumeValid) {
        _state->setSegmentationGrowthVolumeId(_state->currentVolumeId());
        if (_segmentationWidget) {
            _segmentationWidget->setActiveVolume(QString::fromStdString(_state->currentVolumeId()));
        }
    }

    updateNormalGridAvailability();

    _axisAlignedSliceController->applyOrientation(_state ? _state->surface("segmentation").get() : nullptr);
    syncVolumeSelectionControls();
}

CWindow::VolumeAttachResult CWindow::attachVolumeToCurrentPackage(
    const std::shared_ptr<Volume>& volume,
    const QString& location,
    std::vector<std::string> tags,
    const QString& preferredVolumeId)
{
    if (!_state || !_state->vpkg() || !volume) {
        throw std::logic_error("cannot attach a volume without an open project");
    }

    const auto result = _state->vpkg()->attachPreparedVolume(
        location.toStdString(),
        std::move(tags),
        volume);
    if (result == VolumePkg::AttachVolumeResult::VolumeIdConflict)
        return VolumeAttachResult::VolumeIdConflict;

    refreshCurrentVolumePackageUi(preferredVolumeId);
    UpdateView();
    return result == VolumePkg::AttachVolumeResult::Attached
        ? VolumeAttachResult::Attached
        : VolumeAttachResult::AlreadyAttached;
}

void CWindow::refreshCurrentVolumePackageUi(const QString& preferredVolumeId)
{
    if (!_state || !_state->vpkg()) {
        return;
    }

    if (_segmentationWidget) {
        _segmentationWidget->setVolumePackagePath(_state->projectDirectory());
    }

    updateNormalGridAvailability();

    refreshVolumeSelectionUi(preferredVolumeId);
    if (!_state->vpkg()->hasVolumes()) {
        Logger()->info("Opened volpkg '{}' with no volumes", _state->projectDirectory().toStdString());
        showStatusBarMessage(tr("Opened volume package with no volumes."), 5000);
    }

    if (_volumeOverlay) {
        _volumeOverlay->setVolumePkg(_state->vpkg(), _state->projectDirectory());
    }

    if (_surfacePanel) {
        _surfacePanel->refreshPointSetFilterOptions();
    }

    if (_projectWorkspace) {
        _projectWorkspace->refresh();
    }
}

void CWindow::updateNormalGridAvailability()
{
    QString checkedPath;
    const QString path = normalGridDirectoryForVolumePkg(
        _state->vpkg(), _state->currentVolumeId(), &checkedPath);
    const bool available = !path.isEmpty();

    _normalGridAvailable = available;
    _normalGridPath = path;

    if (_segmentationWidget) {
        _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
        _segmentationWidget->setNormalGridPath(_normalGridPath);
        QString hint;
        if (!_normalGridAvailable && _state->vpkg() &&
            _state->vpkg()->catalogView(_state->currentVolumeId())) {
            hint = tr("No normal grids are published for this open-data volume.");
        }
        if (_normalGridAvailable) {
        } else if (!hint.isEmpty()) {
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
    if (_inkDetectionOverlay && _inkDetectionOverlay->hasLoadedSelection()) {
        _inkDetectionOverlay->toggleVisibility();
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
        showStatusBarMessage(tr("Restored full view"), 2000);
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
        showStatusBarMessage(tr("Focused view (Shift+Ctrl+F to restore)"), 2000);
    }
}

void CWindow::switchToLasagnaWorkspace()
{
    if (!_workspaceTabs || !_lasagnaWorkspaceWindow) {
        return;
    }
    const int index = _workspaceTabs->indexOf(_lasagnaWorkspaceWindow);
    if (index >= 0) {
        _workspaceTabs->setCurrentIndex(index);
        _lasagnaWorkspaceWindow->raise();
        _lasagnaWorkspaceWindow->setFocus(Qt::ShortcutFocusReason);
    }
}

void CWindow::switchToMainWorkspace()
{
    if (!_workspaceTabs || !_segmentWorkspaceWindow) {
        return;
    }
    const int index = _workspaceTabs->indexOf(_segmentWorkspaceWindow);
    if (index >= 0) {
        _workspaceTabs->setCurrentIndex(index);
        _segmentWorkspaceWindow->raise();
        _segmentWorkspaceWindow->setFocus(Qt::ShortcutFocusReason);
    }
}

void CWindow::openLineAnnotationWorkspace(LineAnnotationDialog* dialog, const QString& title)
{
    if (!_workspaceTabs || !dialog) {
        return;
    }

    dialog->setWorkspaceEmbedded(true);

    int index = _workspaceTabs->indexOf(dialog);
    if (index < 0) {
        index = _workspaceTabs->addTab(dialog, title.isEmpty() ? tr("Line Annotation") : title);
        auto* escapeShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), dialog);
        escapeShortcut->setContext(Qt::WidgetWithChildrenShortcut);
        connect(escapeShortcut, &QShortcut::activated, dialog, &QWidget::close);
        QWidget* tabWidget = dialog;
        connect(dialog, &QObject::destroyed, this, [this, tabWidget]() {
            if (_destroyingWindow) {
                return;
            }
            if (!_workspaceTabs) {
                return;
            }
            for (int i = 0; i < _workspaceTabs->count(); ++i) {
                if (_workspaceTabs->widget(i) == tabWidget) {
                    const bool wasCurrent = _workspaceTabs->currentIndex() == i;
                    _workspaceTabs->removeTab(i);
                    if (wasCurrent) {
                        switchToMainWorkspace();
                    }
                    break;
                }
            }
        });
    }

    _workspaceTabs->setCurrentIndex(index);
    dialog->show();
    dialog->raise();
    dialog->setFocus(Qt::ShortcutFocusReason);
}

void CWindow::repeatLastLasagnaAction()
{
    if (_segmentationWidget && _segmentationWidget->lasagnaPanel()) {
        _segmentationWidget->lasagnaPanel()->repeatLastLasagnaAction();
    }
}

void CWindow::selectLasagnaOutputSurface(const QString& surfaceUuid)
{
    switchToMainWorkspace();
    if (!treeWidgetSurfaces || surfaceUuid.isEmpty()) {
        return;
    }
    for (QTreeWidgetItemIterator it(treeWidgetSurfaces); *it; ++it) {
        if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() == surfaceUuid) {
            treeWidgetSurfaces->setCurrentItem(*it);
            treeWidgetSurfaces->scrollToItem(*it);
            ui.dockWidgetVolumes->raise();
            return;
        }
    }
}

void CWindow::loadLasagnaResults(Surface* surface)
{
    std::optional<nlohmann::json> corr;
    if (surface && _surfaceCatalog && _surfaceCatalog->attachment()) {
        try {
            auto& session = *_surfaceCatalog->session();
            corr = vc::objects::attachedDocument(session, *_surfaceCatalog->attachment(), surface->id,
                                                 LasagnaServiceManager::kCorrResultsRole);
        } catch (const std::exception& error) {
            qWarning() << "Cannot read Lasagna results for" << QString::fromStdString(surface->id) << error.what();
        }
    }
    if (_point_collection_widget) {
        if (corr) {
            _point_collection_widget->loadCorrPointsResults(utils::Json::parse(corr->dump()));
        } else {
            _point_collection_widget->clearCorrPointsResults();
        }
    }
}

bool CWindow::centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId)
{
    return _viewerManager && _viewerManager->centerFocusAt(position, normal, sourceId);
}

void CWindow::recenterPlaneViewersOn(const cv::Vec3f& position)
{
    if (_viewerManager) {
        _viewerManager->recenterPlaneViewersOn(position);
    }
}

void CWindow::recenterSegmentationViewerNear(const cv::Vec3f& position)
{
    if (_viewerManager) {
        _viewerManager->recenterSegmentationViewerNear(position);
    }
}

bool CWindow::recenterViewersOnCurrentFocus()
{
    auto* manager = activeWorkspaceViewerManager();
    return manager && manager->recenterViewersOnCurrentFocus();
}

bool CWindow::centerFocusOnCursor()
{
    auto* manager = activeWorkspaceViewerManager();
    return manager && manager->centerFocusOnCursor();
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
        showStatusBarMessage(enabled ? tr("Syncing cursor across views enabled")
                                         : tr("Syncing cursor across views disabled"),
                                  2000);
    }
}

// Create widgets
void CWindow::CreateWidgets(void)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    // add volume viewer
    auto aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->setContentsMargins(0, 0, 0, 0);
    aWidgetLayout->setSpacing(0);
    ui.tabSegment->setLayout(aWidgetLayout);

    mdiArea = nullptr;

    auto* viewerGrid = new ViewerSplitGrid(ui.tabSegment);
    viewerGrid->setObjectName(QStringLiteral("mainViewerSplitGrid"));
    aWidgetLayout->addWidget(viewerGrid, 1);

    {
        resetSegmentationViews(false);
        applyMainViewerLayout(viewerGrid, _viewerManager.get());
        viewerGrid->setSplits(settings.value(MAIN_VIEWER_SPLIT_X_SETTING, 0.5).toDouble(),
                              settings.value(MAIN_VIEWER_SPLIT_Y_SETTING, 0.5).toDouble());
    }

    viewerGrid->onSplitChanged = [viewerGrid]() {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(MAIN_VIEWER_SPLIT_X_SETTING, viewerGrid->splitX());
        settings.setValue(MAIN_VIEWER_SPLIT_Y_SETTING, viewerGrid->splitY());
    };

    treeWidgetSurfaces = ui.treeWidgetSurfaces;
    treeWidgetSurfaces->setSelectionMode(QAbstractItemView::ExtendedSelection);

    // The surface list reads the surfaces' store for the current volume.
    _surfaceCatalog = std::make_unique<SurfaceCatalog>(_state->objectStoreSession(), this);
    if (_inkDetectionOverlay) {
        _inkDetectionOverlay->setSurfaceCatalog(_surfaceCatalog.get());
    }
    auto updateSurfaceSource = [this]() {
        _surfaceCatalog->setSource(_state->storeFor("surface"), _state->currentVolume());
        LasagnaServiceManager::instance().setObjectStore(_state->objectStoreSession(), _state->storeFor("surface"));
    };
    connect(_state, &CState::objectStoreChanged, _surfaceCatalog.get(),
            [this, updateSurfaceSource](const vc::objects::StoreSessionEvent& event) {
                using Kind = vc::objects::StoreSessionEvent::Kind;
                if (event.kind == Kind::Attached || event.kind == Kind::Detached) updateSurfaceSource();
                else _surfaceCatalog->handle(event);
            });
    connect(_state, &CState::volumeChanged, _surfaceCatalog.get(), [updateSurfaceSource]() { updateSurfaceSource(); });
    connect(_state, &CState::storeRoutesChanged, _surfaceCatalog.get(), [updateSurfaceSource]() { updateSurfaceSource(); });

    SurfacePanelController::UiRefs surfaceUi{
        .treeWidget = treeWidgetSurfaces,
    };
    _surfacePanel = std::make_unique<SurfacePanelController>(
        surfaceUi,
        _state,
        _surfaceCatalog.get(),
        _viewerManager.get(),
        [this]() { return segmentationViewer(); },
        this);
    _storeSurfaceOverlay = std::make_unique<StoreSurfaceOverlayController>(_state, _viewerManager.get(), this);
    connect(_surfacePanel.get(), &SurfacePanelController::visibleSurfacesChanged,
            _storeSurfaceOverlay.get(), &StoreSurfaceOverlayController::setSurfaces);
    connect(_surfacePanel.get(), &SurfacePanelController::highlightedSurfacesChanged,
            _storeSurfaceOverlay.get(), &StoreSurfaceOverlayController::setHighlighted);
    // Another replica, tool or service changed the open surface: reopen it, unless
    // it is being edited (its edits continue from their own revisions).
    connect(_surfaceCatalog.get(), &SurfaceCatalog::surfacesCommitted, this, [this](const std::vector<std::string>& uuids) {
        const auto active = _state->activeSurface().lock();
        const auto* row = active ? _surfaceCatalog->row(active->id) : nullptr;
        if (!row || std::find(uuids.begin(), uuids.end(), active->id) == uuids.end()) {
            return;
        }
        auto known = _segmentationModule ? _segmentationModule->storePin(active.get()) : std::nullopt;
        if (!known && _surfacePanel->opened()) known = _surfacePanel->opened()->pin;
        std::optional<std::string> head;
        _surfaceCatalog->session()->read(row->source, [&](const auto& db, const auto& txn, const auto&) {
            if (const auto revision = db.currentObjectRevision(txn, active->id)) head = revision->uuid;
        });
        if (!head || (known && known->objectRevision == *head)) {
            return;
        }
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            showStatusBarMessage(tr("This surface changed in the object store while you edit it; your saves will be refused until you reopen it."), 15000);
            return;
        }
        _surfacePanel->activateSurfaceById(active->id);
    });
    if (_viewerManager) {
        _viewerManager->setSurfacePicker([this](const cv::Vec3f& point, float tolerance) -> std::optional<std::string> {
            const auto active = _state->surface("segmentation");
            if (!_surfaceCatalog->volume()) return std::nullopt;
            vc::objects::SurfaceProximityQuery query;
            query.point = {point[0], point[1], point[2]};
            query.tolerance = tolerance;
            query.uuids = _surfacePanel->visibleSurfaceIds();
            query.volume = _surfaceCatalog->volume()->id();
            for (const auto& [attachment, hit] : _surfaceCatalog->session()->nearestSurfaces(_surfaceCatalog->sources(), query))
                if (!active || hit.objectUuid != active->id) return hit.objectUuid;
            return std::nullopt;
        });
    }
    if (_segmentationGrower) {
        _segmentationGrower->setSurfacePanel(_surfacePanel.get());
    }
    if (_lineAnnotationController) {
        _lineAnnotationController->setSurfacePanel(_surfacePanel.get());
    }
    connect(_surfaceCatalog.get(), &SurfaceCatalog::rowsChanged, this, [this]() {
        emit _state->surfacesLoaded();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceSelectionCleared, this, [this]() {
        clearSurfaceSelection();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::filtersApplied, this, [this](int filterCount) {
        UpdateVolpkgLabel(filterCount);
    });
    connect(_surfacePanel.get(), &SurfacePanelController::addSurfaceToSpiralFitRequested,
            this, [this](const QString& segmentId) {
                if (!_spiralWorkspace) {
                    return;
                }
                std::shared_ptr<QuadSurface> surf;
                try {
                    surf = _surfaceCatalog->open(segmentId.toStdString()).surface;
                } catch (const std::exception&) {
                }
                if (!surf || surf->path.empty()) {
                    showStatusBarMessage(tr("Cannot resolve an on-disk TIFXYZ directory for %1").arg(segmentId), 5000);
                    return;
                }
                _spiralWorkspace->addPatchToCurrentFit(
                    QString::fromStdString(surf->path.string()), surf);
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
    connect(_surfacePanel.get(), &SurfacePanelController::convertToObjRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onConvertToObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::mergeSurfacesRequested,
            this, [this](const QStringList& segmentIds) {
                _segmentationCommandHandler->onMergeSurfaces(segmentIds);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::mergePatchRequested,
            this, [this](const QStringList& segmentIds) {
                _segmentationCommandHandler->onMergePatch(segmentIds);
            });
    // Note: the Actions -> Merge Surfaces... menu wiring lives in the
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
                if (!_state) return;
                const std::string id = segmentId.toStdString();
                auto surf = std::dynamic_pointer_cast<QuadSurface>(_state->surface(id));
                if (const auto& opened = _surfacePanel->opened(); !surf && opened && opened->surface->id == id) {
                    surf = opened->surface;
                }
                if (!surf && _surfaceCatalog) {
                    try {
                        surf = _surfaceCatalog->open(id).surface;
                    } catch (const std::exception& error) {
                        showStatusBarMessage(tr("Cannot focus %1: %2").arg(segmentId, QString::fromUtf8(error.what())), 5000);
                        return;
                    }
                }
                auto* quad = dynamic_cast<QuadSurface*>(surf.get());
                if (!quad) return;
                const auto focusPoint = findSegmentFocusPoint(*quad);
                if (!focusPoint) return;
                cv::Vec3f normal = quad->normal(focusPoint->ptr, {0, 0, 0});
                if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
                    normal = cv::Vec3f(0, 0, 1);
                }
                if (auto vol = _state->currentVolume()) {
                    auto [w, h, d] = vol->shapeXyz();
                    cv::Vec3f clamped = focusPoint->world;
                    clamped[0] = std::clamp(clamped[0], 0.0f, static_cast<float>(w - 1));
                    clamped[1] = std::clamp(clamped[1], 0.0f, static_cast<float>(h - 1));
                    clamped[2] = std::clamp(clamped[2], 0.0f, static_cast<float>(d - 1));
                    POI* poi = new POI;
                    poi->p = clamped;
                    poi->n = normal;
                    poi->surfaceId = segmentId.toStdString();
                    poi->surfacePtr = focusPoint->ptr;
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
    connect(_surfacePanel.get(), &SurfacePanelController::straightenRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onStraighten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::rasterizeSegmentsRequested,
        this, [this](const QStringList& segmentIds) {
            _segmentationCommandHandler->onRasterizeSegments(segmentIds);
        });
    connect(_surfacePanel.get(), &SurfacePanelController::addIgnoreLabelRequested,
        this, [this]() {
            _segmentationCommandHandler->onAddIgnoreLabel();
        });
    connect(_surfacePanel.get(), &SurfacePanelController::statusMessageRequested,
            this, [this](const QString& message, int timeoutMs) {
                showStatusBarMessage(message, timeoutMs);
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
        panel->setState(_state);
        panel->setSurfaceCatalog(_surfaceCatalog.get());
        panel->setVisible(true);
        _lasagnaWorkspaceWindow->setCentralWidget(panel);

        _lasagnaDock = new QDockWidget(tr("Lasagna"), this);
        _lasagnaDock->setObjectName(QStringLiteral("dockWidgetLasagna"));
        attachScrollAreaToDock(_lasagnaDock,
                               panel->createCompactView(_lasagnaDock),
                               QStringLiteral("dockWidgetLasagnaContent"));
        _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _lasagnaDock);
        connect(panel, &SegmentationLasagnaPanel::openLasagnaWorkspaceRequested,
                this, &CWindow::switchToLasagnaWorkspace);
        connect(panel, &SegmentationLasagnaPanel::lasagnaOutputActivated,
                this, &CWindow::selectLasagnaOutputSurface);
    }

    _segmentationEdit = std::make_unique<SegmentationEditManager>(this);
    _segmentationEdit->setViewerManager(_viewerManager.get());
    _segmentationOverlay = std::make_unique<SegmentationOverlayController>(_state, this);
    _segmentationOverlay->setEditManager(_segmentationEdit.get());
    _segmentationOverlay->setViewerManager(_viewerManager.get());
    _surfaceRotationOverlay = std::make_unique<SurfaceRotationOverlayController>(_state, this);
    _surfaceRotationOverlay->setSaver([this](QuadSurface* surface) {
        return _segmentationModule && _segmentationModule->saveSurface(surface);
    });
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
    _segmentationModule->setEditingDestinationResolver([this](const std::shared_ptr<QuadSurface>& surface) {
        // Continue editing in the store the surface was opened from.
        QString error;
        if (surface && _surfacePanel && !_surfacePanel->prepareSurfaceEditing(surface->id, &error)) {
            showStatusBarMessage(error, 15000);
            return false;
        }
        return !_spiralWorkspace || _spiralWorkspace->prepareManagedPatch(surface);
    });
    // Edits of the active surface commit to the store object it was opened from.
    _segmentationModule->setStorePinResolver([this](const QuadSurface* surface) -> std::optional<SegmentationModule::StoreOrigin> {
        const auto& opened = _surfacePanel ? _surfacePanel->opened() : std::nullopt;
        // By id: transforms and edit previews publish copies of the opened surface.
        if (opened && surface && opened->surface->id == surface->id && opened->editable)
            return SegmentationModule::StoreOrigin{opened->pin, opened->view};
        return std::nullopt;
    });


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
    connect(_segmentationModule.get(), &SegmentationModule::growSurfaceRequested,
            this, &CWindow::onGrowSegmentationSurface);

    SegmentationGrower::Context growerContext{
        _segmentationModule.get(),
        _segmentationWidget,
        _state,
        _viewerManager.get()
    };
    SegmentationGrower::UiCallbacks growerCallbacks{
        [this](const QString& text, int timeout) {
            if (statusBar()) {
                showStatusBarMessage(text, timeout);
            }
        },
        [this](QuadSurface* surface) {
            _axisAlignedSliceController->applyOrientation(surface);
        }
    };
    _segmentationGrower = std::make_unique<SegmentationGrower>(growerContext, growerCallbacks, this);

    _segmentationCommandHandler = std::make_unique<SegmentationCommandHandler>(this, _state, this);
    _segmentationCommandHandler->setEditingDestinationResolver([this](const std::shared_ptr<QuadSurface>& surface) {
        return _segmentationModule->prepareEditingDestination(surface);
    });
    _segmentationCommandHandler->setSurfaceSaver([this](QuadSurface* surface) {
        return _segmentationModule && _segmentationModule->saveSurface(surface);
    });
    _segmentationCommandHandler->setCmdRunner(_cmdRunner);
    _segmentationCommandHandler->setSurfacePanel(_surfacePanel.get());
    _segmentationCommandHandler->setSegmentationGrower(_segmentationGrower.get());
    initializeCommandLineRunner();
    _segmentationCommandHandler->setNormalGridPathGetter([this]() -> QString {
        updateNormalGridAvailability();
        return _normalGridPath;
    });
    connect(_segmentationCommandHandler.get(), &SegmentationCommandHandler::statusMessage,
            this, &CWindow::onShowStatusMessage);


    connect(_segmentationWidget, &SegmentationWidget::copyWithNtRequested,
            this, &CWindow::onCopyWithNtRequested);
    connect(_segmentationWidget, &SegmentationWidget::volumeSelectionChanged, this, [this](const QString& volumeId) {
        if (!_state->vpkg()) {
            showStatusBarMessage(tr("No volume package loaded."), 4000);
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
            showStatusBarMessage(tr("Using volume '%1' for surface growth.").arg(volumeId), 2500);
        } catch (const std::out_of_range&) {
            showStatusBarMessage(tr("Volume '%1' not found in this package.").arg(volumeId), 4000);
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
        showStatusBarMessage(tr("Lasagna optimization stop requested."), 3000);
    });

    // The new surfaces reach the surface list through the store's commit events.
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::optimizationFinished,
            this, [this](const QStringList& surfaceUuids) {
        showStatusBarMessage(tr("Lasagna optimization finished: %1 new surface(s)").arg(surfaceUuids.size()), 5000);
    });

    // Create Seeding widget
    _seedingWidget = new SeedingWidget(_state->pointCollection(), _state);
    _seedingWidget->setSurfaceCatalog(_surfaceCatalog.get());
    _seedingWidget->setViewerManager(_viewerManager.get());
    attachScrollAreaToDock(ui.dockWidgetDistanceTransform, _seedingWidget, QStringLiteral("dockWidgetDistanceTransformContent"));

    _seedingWidget->setState(_state);
    connect(_state, &CState::volumeChanged, _seedingWidget,
            static_cast<void (SeedingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&SeedingWidget::onVolumeChanged));
    connect(_seedingWidget, &SeedingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(_state, &CState::surfacesLoaded, _seedingWidget, &SeedingWidget::onSurfacesLoaded);

    _wrapAnnotationWidget = new WrapAnnotationWidget(_state->pointCollection(), this);
    _wrapAnnotationWidget->setObjectName("wrapAnnotationDock");
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _wrapAnnotationWidget);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::relWindingAnnotationToggled,
            _seedingWidget, &SeedingWidget::setRelWindingAnnotationMode);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::relWindingIntersectionSourceChanged,
            _seedingWidget, &SeedingWidget::setRelWindingIntersectionSource);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::relWindingPatchToleranceChanged,
            _seedingWidget, &SeedingWidget::setRelWindingPatchTolerance);
    connect(_seedingWidget, &SeedingWidget::relWindingAnnotationModeChanged,
            _wrapAnnotationWidget, &WrapAnnotationWidget::setRelWindingAnnotationChecked);
    _seedingWidget->setRelWindingIntersectionSource(_wrapAnnotationWidget->relWindingIntersectionSource());
    _seedingWidget->setRelWindingPatchTolerance(_wrapAnnotationWidget->relWindingPatchTolerance());

    // Create and add the point collection widget
    _point_collection_widget = new CPointCollectionWidget(_state->pointCollection(), this);
    _point_collection_widget->setObjectName("pointCollectionDock");
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _point_collection_widget);

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
    connect(_point_collection_widget, &CPointCollectionWidget::focusCollectionRequested,
            this, [this](uint64_t collectionId) {
                onFocusViewsRequested(collectionId, std::nullopt);
            });
    connect(_point_collection_widget, &CPointCollectionWidget::focusPointRequested,
            this, [this](vc::PointRef point) {
                onFocusViewsRequested(point.collectionId, point);
            });
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::focusCollectionRequested,
            this, [this](uint64_t collectionId) {
                onFocusViewsRequested(collectionId, std::nullopt);
            });
    connect(_wrapAnnotationWidget, &WrapAnnotationWidget::focusPointRequested,
            this, [this](vc::PointRef point) {
                onFocusViewsRequested(point.collectionId, point);
            });
    if (_pointsOverlay) {
        _pointsOverlay->setViewTolerance(_point_collection_widget->pointViewTolerance());
        connect(_point_collection_widget, &CPointCollectionWidget::pointViewToleranceChanged,
                _pointsOverlay.get(), &PointsOverlayController::setViewTolerance);
    }

    // Tab the docks - keep Segmentation, Lasagna, Seeding, and Point Collections together
    // Wire annotate mode & annotation selection: dock widget <-> segmentation module
    // (must be after _point_collection_widget creation)
    connect(_point_collection_widget, &CPointCollectionWidget::annotateToggled,
            _segmentationModule.get(), &SegmentationModule::setAnnotateMode);
    connect(_segmentationModule.get(), &SegmentationModule::annotateModeChanged,
            _point_collection_widget, &CPointCollectionWidget::setAnnotateChecked);
    connect(_segmentationModule.get(), &SegmentationModule::annotationPointSelected,
            _point_collection_widget, &CPointCollectionWidget::selectPoint);
    connect(_segmentationModule.get(), &SegmentationModule::annotationSelectionCleared,
            _point_collection_widget, &CPointCollectionWidget::clearSelection);
    connect(_segmentationModule.get(), &SegmentationModule::annotationCollectionSelected,
            _point_collection_widget, &CPointCollectionWidget::selectCollection);
    connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
            _segmentationModule.get(), &SegmentationModule::setSelectedAnnotationCollection);
    connect(_point_collection_widget, &CPointCollectionWidget::collectionSelectionCleared,
            _segmentationModule.get(), &SegmentationModule::clearSelectedAnnotationCollection);

    _fiberWidget = new CFiberWidget(this);
    _fiberWidget->setObjectName("fiberDock");
    _segmentWorkspaceWindow->addDockWidget(Qt::RightDockWidgetArea, _fiberWidget);

    auto withFiberWidget = [this](auto&& apply) {
        if (_fiberWidget) apply(_fiberWidget);
    };

    if (_lineAnnotationController && _fiberCollection) {
        auto* collection = _fiberCollection.get();
        auto toFiberWidgetAlignment =
            [](const LineAnnotationController::FiberSummary::AlignmentMetrics& source) {
                CFiberWidget::FiberEntry::AlignmentMetrics alignment;
                alignment.available = source.available;
                alignment.pending = source.pending;
                alignment.sampleCount = source.sampleCount;
                alignment.meanErrorDegrees = source.meanErrorDegrees;
                alignment.maxErrorDegrees = source.maxErrorDegrees;
                alignment.error = source.error;
                return alignment;
            };
        // A store catalog row as the panels list it; details load on selection.
        auto catalogEntry = [this](const FiberCollectionController::CatalogRow& row) {
            CFiberWidget::FiberEntry entry;
            entry.id = _lineAnnotationController->fiberIdForUuid(row.record.uuid);
            entry.fileName = row.record.name + ".json";
            entry.tags = row.record.tags;
            if (const auto& fiber = row.record.fiber) {
                entry.controlPointCount = static_cast<int>(fiber->controlPointCount);
                entry.linePointCount = static_cast<int>(fiber->linePointCount);
                // The volume length the store ordered a bound row by; stored units otherwise.
                entry.lengthVx = row.record.volumeLength.value_or(fiber->length);
                (fiber->manualFamily ? entry.manualHvTag : entry.automaticHvTag) = fiber->family;
            }
            entry.linkedFiberCount = row.componentSize >= 2 ? row.componentSize : 0;
            entry.pendingLinkCount = row.pendingLinks;
            entry.detailsLoaded = false;
            return entry;
        };
        auto detailEntry = [toFiberWidgetAlignment](const LineAnnotationController::FiberSummary& fiber) {
            std::vector<CFiberWidget::FiberEntry::SpanEntry> spans;
            spans.reserve(fiber.spans.size());
            for (const auto& span : fiber.spans) {
                spans.push_back(CFiberWidget::FiberEntry::SpanEntry{
                    span.spanIndex,
                    span.firstControlIndex,
                    span.secondControlIndex,
                    span.controlPointCount,
                    span.linePointCount,
                    span.lengthVx,
                    toFiberWidgetAlignment(span.alignment),
                    span.interpMarker,
                    span.fiberManifest,
                });
            }
            CFiberWidget::FiberEntry::TraceState traceState =
                CFiberWidget::FiberEntry::TraceState::Legacy;
            switch (fiber.traceState) {
            case vc3d::line_annotation::FiberTraceState::Predictions:
                traceState = CFiberWidget::FiberEntry::TraceState::Predictions;
                break;
            case vc3d::line_annotation::FiberTraceState::Mixed:
                traceState = CFiberWidget::FiberEntry::TraceState::Mixed;
                break;
            case vc3d::line_annotation::FiberTraceState::Legacy:
                break;
            }
            return CFiberWidget::FiberEntry{
                fiber.id,
                fiber.name,
                fiber.controlPointCount,
                fiber.linePointCount,
                fiber.annotatedLengthVx,
                toFiberWidgetAlignment(fiber.alignment),
                std::move(spans),
                fiber.hvZDistance,
                fiber.hvFiberLength,
                fiber.horizontalScore,
                fiber.verticalScore,
                fiber.automaticCertainty,
                fiber.automaticHvTag,
                fiber.manualHvTag,
                fiber.tags,
                fiber.linkedFiberCount,
                fiber.pendingLinkCount,
                traceState,
                true,
            };
        };

        connect(collection, &FiberCollectionController::catalogChanged, this,
                [this, withFiberWidget, catalogEntry](const std::vector<FiberCollectionController::CatalogRow>& rows, bool morePages) {
                    std::vector<CFiberWidget::FiberEntry> entries;
                    entries.reserve(rows.size());
                    for (const auto& row : rows) {
                        entries.push_back(catalogEntry(row));
                    }
                    // Unit first, so the rebuilt rows below are texted once in it.
                    updateFiberLengthUnits();
                    const auto tags = _lineAnnotationController->knownFiberTags();
                    withFiberWidget([&](CFiberWidget* widget) {
                        widget->setFibers(entries);
                        widget->setKnownTags(tags);
                        widget->setNextPageAvailable(morePages);
                    });
                });
        connect(collection, &FiberCollectionController::fiberRevealed, this,
                [withFiberWidget, catalogEntry](const FiberCollectionController::CatalogRow& row) {
                    const auto entry = catalogEntry(row);
                    withFiberWidget([&](CFiberWidget* widget) { widget->revealFiber(entry); });
                });
        connect(collection, &FiberCollectionController::statusChanged, this,
                [withFiberWidget](bool attached, const QString& text) {
                    withFiberWidget([&](CFiberWidget* widget) { widget->setStoreStatus(attached, text); });
                });
        connect(collection, &FiberCollectionController::selectionChanged, this,
                [withFiberWidget](const FiberCollectionController::Selection& selection) {
                    withFiberWidget([&](CFiberWidget* widget) {
                        widget->setTraceSelection(selection.text, selection.toolTip, selection.pointCount,
                                                  selection.index, selection.openable);
                    });
                });
        connect(collection, &FiberCollectionController::annotationStatusChanged, this,
                [withFiberWidget](const QString& text) {
                    withFiberWidget([&](CFiberWidget* widget) { widget->setAnnotationStatus(text); });
                });
        connect(collection, &FiberCollectionController::panelRequested, this, [this]() {
            if (_statusDockPanelHost) {
                _statusDockPanelHost->showDock(_fiberWidget);
            } else {
                _fiberWidget->show();
                _fiberWidget->raise();
            }
        });
        connect(_lineAnnotationController.get(), &LineAnnotationController::fibersChanged, this,
                [this, withFiberWidget]() {
                    const auto tags = _lineAnnotationController->knownFiberTags();
                    withFiberWidget([&](CFiberWidget* widget) { widget->setKnownTags(tags); });
                    updateLineAnnotationLengthLabel();
                });

        withFiberWidget([&](CFiberWidget* widget) {
            widget->setDisplayOptions(collection->options());
            connect(widget, &CFiberWidget::fiberDetailsRequested, this,
                    [this, withFiberWidget, detailEntry](uint64_t fiberId) {
                        const auto summary = _lineAnnotationController->fiberSummary(fiberId);
                        if (!summary) return;
                        const auto entry = detailEntry(*summary);
                        withFiberWidget([&](CFiberWidget* listed) { listed->updateFiberDetails(entry); });
                    },
                    Qt::QueuedConnection);
            connect(widget, &CFiberWidget::displayOptionsChanged, this,
                    [collection, withFiberWidget](const FiberDisplayOptions& options) {
                        collection->setOptions(options);
                        withFiberWidget([&](CFiberWidget* listed) { listed->setDisplayOptions(options); });
                    });
            // The widget's store buttons choose where fibers are kept.
            connect(widget, &CFiberWidget::attachStoreRequested, this, [this]() {
                if (_menuController) _menuController->attachStore(false, "fiber");
            });
            connect(widget, &CFiberWidget::detachStoreRequested, this, [this]() {
                try { _state->setStoreFor("fiber", std::nullopt); }
                catch (const std::exception& error) { showStatusBarMessage(QString::fromUtf8(error.what()), 8000); }
            });
            connect(widget, &CFiberWidget::firstPageRequested, collection, &FiberCollectionController::firstPage);
            connect(widget, &CFiberWidget::nextPageRequested, collection, &FiberCollectionController::nextPage);
            connect(widget, &CFiberWidget::fiberActivated, collection,
                    [collection](uint64_t fiberId) { collection->activateFiber(int64_t(fiberId)); });
            connect(widget, &CFiberWidget::traceIndexChanged, collection, &FiberCollectionController::navigate);
            connect(widget, &CFiberWidget::openInLineAnnotationRequested, collection,
                    [collection](uint64_t fiberId) { collection->showInLineAnnotation(int64_t(fiberId)); });
        });
        auto updateFiberMetricRows =
            [this, toFiberWidgetAlignment](
                uint64_t fiberId,
                LineAnnotationController::FiberSummary::AlignmentMetrics alignment,
                const std::vector<LineAnnotationController::FiberSummary::AlignmentMetrics>& spanAlignments) {
                std::vector<CFiberWidget::FiberEntry::AlignmentMetrics> widgetSpanAlignments;
                widgetSpanAlignments.reserve(spanAlignments.size());
                for (const auto& spanAlignment : spanAlignments) {
                    widgetSpanAlignments.push_back(toFiberWidgetAlignment(spanAlignment));
                }
                const CFiberWidget::FiberEntry::AlignmentMetrics widgetAlignment =
                    toFiberWidgetAlignment(alignment);
                if (_fiberWidget) {
                    _fiberWidget->updateAlignmentMetrics(fiberId,
                                                         widgetAlignment,
                                                         widgetSpanAlignments);
                }
            };
        auto connectFiberWidget = [this](CFiberWidget* widget) {
            if (!widget || !_lineAnnotationController) {
                return;
            }
            connect(widget,
                    &CFiberWidget::fiberOpenRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::openFiber);
            connect(widget,
                    &CFiberWidget::fiberSpanOpenRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::openFiberSpan);
            connect(widget,
                    &CFiberWidget::deleteFibersRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::deleteFibers);
            connect(widget,
                    &CFiberWidget::renameFiberFileRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::renameFiberFile);
            connect(widget,
                    &CFiberWidget::importFibersRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::importFibers);
            connect(widget,
                    &CFiberWidget::exportFibersRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::exportFibers);
            connect(widget,
                    &CFiberWidget::metricsCalculationRequested,
                    _lineAnnotationController.get(),
                    [this](std::vector<uint64_t> orderedFiberIds) {
                        if (_lineAnnotationController) {
                            _lineAnnotationController->calculateFiberAlignmentMetrics(
                                std::move(orderedFiberIds));
                        }
                    });
            connect(widget,
                    &CFiberWidget::manualHvTagChanged,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::setFiberManualHvTag);
            connect(widget,
                    &CFiberWidget::fiberTagChanged,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::setFiberTag);
            connect(widget,
                    &CFiberWidget::hvScoreRecalculationRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::recalculateFiberHvClassification);
            connect(widget,
                    &CFiberWidget::addFibersToPointCollectionsRequested,
                    _lineAnnotationController.get(),
                    &LineAnnotationController::addFibersToPointCollections);
            connect(widget,
                    &CFiberWidget::addFibersToSpiralFitRequested,
                    this,
                    [this](std::vector<uint64_t> fiberIds) {
                        if (!_spiralWorkspace || !_lineAnnotationController) {
                            return;
                        }
                        _spiralWorkspace->addFibersToCurrentFit(fiberIds);
                    });
        };
        connect(_lineAnnotationController.get(),
                &LineAnnotationController::fiberAlignmentMetricsUpdated,
                this,
                updateFiberMetricRows);
        connectFiberWidget(_fiberWidget);
        // The annotation frame (and so the cm conversion) follows the
        // current volume; the lengths themselves do not change.
        connect(_state, &CState::volumeChanged, this,
                [this](const std::shared_ptr<Volume>&, const std::string&) {
                    updateFiberLengthUnits();
                });
    }

    ensureDockWidgetFeatures(_fiberWidget);
    connect(_fiberWidget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
    connect(_fiberWidget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);

    // Tab the docks - keep Segmentation, Lasagna, Seeding, Point Collections, and Fibers together
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, _lasagnaDock);
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDistanceTransform);
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, _point_collection_widget);
    _segmentWorkspaceWindow->tabifyDockWidget(ui.dockWidgetSegmentation, _fiberWidget);

    // Make Segmentation dock the active tab by default
    ui.dockWidgetSegmentation->raise();

    ViewerControlsPanel::UiRefs viewerControlsUi{
        .contents = ui.dockWidgetViewerControlsContents,
        .viewScrollArea = ui.scrollAreaView,
        .viewContents = ui.dockWidgetViewContents,
        .overlayScrollArea = ui.scrollAreaOverlay,
        .overlayContents = ui.dockWidgetOverlayContents,
        .normalVisualizationContents = ui.dockWidgetNormalVisContents,
        .showSurfaceNormals = ui.chkShowSurfaceNormals,
        .normalArrowLengthLabel = ui.labelNormalArrowLength,
        .normalArrowLengthSlider = ui.sliderNormalArrowLength,
        .normalArrowLengthValueLabel = ui.labelNormalArrowLengthValue,
        .normalMaxArrowsLabel = ui.labelNormalMaxArrows,
        .normalMaxArrowsSlider = ui.sliderNormalMaxArrows,
        .normalMaxArrowsValueLabel = ui.labelNormalMaxArrowsValue,
        .zoomInButton = ui.btnZoomIn,
        .zoomOutButton = ui.btnZoomOut,
        .volumeWindowContainer = ui.volumeWindowContainer,
        .overlayWindowContainer = ui.overlayWindowContainer,
        .intersectionThicknessSpin = ui.doubleSpinIntersectionThickness,
    };
    _viewerControlsPanel = std::make_unique<ViewerControlsPanel>(viewerControlsUi,
                                                                 _viewerManager.get(),
                                                                 ui.dockWidgetViewerControlsContents);
    ViewerCompositePanel::UiRefs compositeUi{
        .scrollArea = ui.scrollAreaComposite,
        .contents = ui.dockWidgetCompositeContents,
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
        .planeCompositeXY = ui.chkPlaneCompositeXY,
        .planeCompositeXZ = ui.chkPlaneCompositeXZ,
        .planeCompositeYZ = ui.chkPlaneCompositeYZ,
        .planeLayersFront = ui.spinPlaneLayersFront,
        .planeLayersBehind = ui.spinPlaneLayersBehind,
        .planeReverseDirection = ui.chkPlaneReverseDirection,
    };
    _viewerCompositePanel = new ViewerCompositePanel(compositeUi, _viewerManager.get(), ui.dockWidgetComposite);
    _viewerCompositePanel->setViewerManagers(
        {_viewerManager.get(), _spiralWorkspace ? _spiralWorkspace->viewerManager() : nullptr});
    attachScrollAreaToDock(ui.dockWidgetComposite,
                           _viewerCompositePanel,
                           QStringLiteral("dockWidgetCompositeContent"));

    _inkDetectionDock = new QDockWidget(tr("Ink Detection"), this);
    _inkDetectionDock->setObjectName(QStringLiteral("dockWidgetInkDetection"));
    attachScrollAreaToDock(_inkDetectionDock,
                           new ViewerInkDetectionPanel(_viewerManager.get(), _inkDetectionDock),
                           QStringLiteral("dockWidgetInkDetectionContent"));

    _transformsDock = new QDockWidget(tr("Transforms"), this);
    _transformsDock->setObjectName(QStringLiteral("dockWidgetTransforms"));
    auto* transformsPanel = new VolumeRegistrationsPanel(_state, _transformsDock);
    attachScrollAreaToDock(_transformsDock,
                           transformsPanel,
                           QStringLiteral("dockWidgetTransformsContent"));

    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomInRequested,
            this, &CWindow::onZoomIn);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::zoomOutRequested,
            this, &CWindow::onZoomOut);
    connect(_viewerManager.get(), &ViewerManager::zScrollSensitivityChanged,
            this, &CWindow::onZScrollSensitivityChanged);
    connect(_viewerControlsPanel.get(), &ViewerControlsPanel::statusMessageRequested,
            this, &CWindow::onShowStatusMessage);
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewControlsEnabled(!ui.grpVolManager || ui.grpVolManager->isEnabled());
    }


    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetViewerControls);
    _segmentWorkspaceWindow->splitDockWidget(ui.dockWidgetVolumes, ui.dockWidgetViewerControls, Qt::Vertical);
    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetOverlay);
    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _inkDetectionDock);
    _segmentWorkspaceWindow->addDockWidget(Qt::LeftDockWidgetArea, _transformsDock);
    _segmentWorkspaceWindow->splitDockWidget(ui.dockWidgetViewerControls, ui.dockWidgetOverlay, Qt::Vertical);
    _segmentWorkspaceWindow->splitDockWidget(ui.dockWidgetOverlay, _inkDetectionDock, Qt::Vertical);
    _segmentWorkspaceWindow->splitDockWidget(_inkDetectionDock, _transformsDock, Qt::Vertical);

    auto hideLegacyViewerDocks = [this]() {
        for (QDockWidget* dock : { ui.dockWidgetNormalVis,
                                   ui.dockWidgetView,
                                        ui.dockWidgetComposite }) {
            if (!dock) {
                continue;
            }
            _segmentWorkspaceWindow->removeDockWidget(dock);
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
            .samplingMethodSelect = ui.overlayInterpolationSelect,
            .opacitySpin = ui.overlayOpacitySpin,
            .intensityAsAlphaCheck = ui.overlayIntensityAsAlphaCheck,
            .maxDisplayedResolutionSpin = ui.overlayMaxDisplayedResolutionSpin,
            .compositeEnabledCheck = ui.chkOverlayComposite,
            .compositeMethodSelect = ui.cmbOverlayCompositeMethod,
            .compositeLayersFrontSpin = ui.spinOverlayCompositeLayersFront,
            .compositeLayersBehindSpin = ui.spinOverlayCompositeLayersBehind,
        };
        _volumeOverlay->setUi(overlayUi);
        if (_viewerControlsPanel) {
            _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay->hasOverlaySelection());
        }
        updateActiveWorkspaceViewerControls();
    }

    connectVolumeSelector(volSelect);

    auto* filterDropdown = ui.btnFilterDropdown;
    auto* cmbPointSetFilter = new QComboBox(this);
    auto* btnPointSetFilterAll = new QPushButton(tr("All"), this);
    auto* btnPointSetFilterNone = new QPushButton(tr("None"), this);
    auto* cmbPointSetFilterMode = new QComboBox(this);
    auto* spinFocusPointFilterDistance = new QDoubleSpinBox(this);
    auto* spinSurfaceZLowerBound = new QDoubleSpinBox(this);
    auto* spinSurfaceZUpperBound = new QDoubleSpinBox(this);
    cmbPointSetFilterMode->addItem("Any (OR)");
    cmbPointSetFilterMode->addItem("All (AND)");

    SurfacePanelController::FilterUiRefs filterUi;
    filterUi.dropdown = filterDropdown;
    filterUi.currentOnly = ui.chkFilterCurrentOnly;
    filterUi.pointSet = cmbPointSetFilter;
    filterUi.pointSetAll = btnPointSetFilterAll;
    filterUi.pointSetNone = btnPointSetFilterNone;
    filterUi.pointSetMode = cmbPointSetFilterMode;
    filterUi.surfaceIdFilter = ui.lineEditSurfaceFilter;
    filterUi.focusPointDistance = spinFocusPointFilterDistance;
    filterUi.zLowerBound = spinSurfaceZLowerBound;
    filterUi.zUpperBound = spinSurfaceZUpperBound;
    _surfacePanel->configureFilters(filterUi, _state->pointCollection());

    auto* tagDropdown = ui.btnTagDropdown;
    SurfacePanelController::TagUiRefs tagUi{
        .dropdown = tagDropdown,
        .approved = tagDropdown->addOption(tr("Approved"), QStringLiteral("chkApproved")),
        .defective = tagDropdown->addOption(tr("Defective"), QStringLiteral("chkDefective")),
        .reviewed = tagDropdown->addOption(tr("Reviewed"), QStringLiteral("chkReviewed")),
        .inspect = tagDropdown->addOption(tr("Inspect"), QStringLiteral("chkInspect")),
    };
    _surfacePanel->configureTags(tagUi);


    // Location input element (single QLineEdit for comma-separated values)
    lblLocFocus = ui.sliceFocus;

    // Set up validator for location input (accepts digits, commas, and spaces)
    QRegularExpressionValidator* validator = new QRegularExpressionValidator(
        QRegularExpression("^\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*$"), this);
    lblLocFocus->setValidator(validator);
    connect(lblLocFocus, &QLineEdit::editingFinished, this, &CWindow::onManualLocationChanged);

    QPushButton* btnCopyCoords = ui.btnCopyCoords;
    connect(btnCopyCoords, &QPushButton::clicked, this, &CWindow::onCopyCoordinates);

    const QRegularExpression coordinateTriple(
        "^\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*$");
    ui.focusBBoxMin->setValidator(
        new QRegularExpressionValidator(coordinateTriple, ui.focusBBoxMin));
    ui.focusBBoxMax->setValidator(
        new QRegularExpressionValidator(coordinateTriple, ui.focusBBoxMax));
    connect(ui.focusBBoxMin, &QLineEdit::editingFinished,
            this, &CWindow::onFocusBoundsEdited);
    connect(ui.focusBBoxMax, &QLineEdit::editingFinished,
            this, &CWindow::onFocusBoundsEdited);
    connect(ui.chkFocusBBox, &QCheckBox::toggled,
            this, &CWindow::onFocusBoundsToggled);
    connect(_state, &CState::focusBoundsChanged,
            this, &CWindow::refreshFocusBoundsUi);
    connect(_state, &CState::volumeChanged, this,
            [this](std::shared_ptr<Volume>, const std::string&) {
                refreshFocusBoundsUi();
            });
    refreshFocusBoundsUi();

    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        bool showOverlays = settings.value(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS,
                                           vc3d::settings::viewer::SHOW_AXIS_OVERLAYS_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisOverlays);
        chkAxisOverlays->setChecked(showOverlays);
        connect(chkAxisOverlays, &QCheckBox::toggled, this, &CWindow::onAxisOverlayVisibilityToggled);
    }
    // Deliberately not persisted: the coordinate frame gizmo is on by default
    // every session, matching the checkbox's designer state.
    if (auto* chkCoordinateFrames = ui.chkCoordinateFrames) {
        connect(chkCoordinateFrames, &QCheckBox::toggled, this, [](bool show) {
            for (auto* manager : ViewerManager::allManagers()) {
                if (manager) {
                    manager->setShowCoordinateFrames(show);
                }
            }
        });
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

    if (chkAxisAlignedSlices) {
        onAxisAlignedSlicesToggled(chkAxisAlignedSlices->isChecked());
    }
    onAxisOverlayOpacityChanged(settings.value(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY,
                                               vc3d::settings::viewer::AXIS_OVERLAY_OPACITY_DEFAULT).toInt());
    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        onAxisOverlayVisibilityToggled(chkAxisOverlays->isChecked());
    }
    onMoveOnSurfaceChangedToggled(moveOnSurfaceChangeEnabled());

}

// Create menus
// Create actions
void CWindow::keyPressEvent(QKeyEvent* event)
{
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

    if (_viewerManager && _wrapAnnotationWidget && _wrapAnnotationWidget->sameWrapAnnotationEnabled()) {
        if (event->key() == Qt::Key_E && event->modifiers() == Qt::ShiftModifier) {
            bool committed = false;
            _viewerManager->forEachBaseViewer([&committed](VolumeViewerBase* baseViewer) {
                if (committed || !baseViewer) {
                    return;
                }
                auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject());
                if (!viewer) {
                    return;
                }
                committed = viewer->commitSameWrapAnnotationPreview();
            });
            if (committed) {
                event->accept();
                return;
            }
        }
        if (event->key() == Qt::Key_Z && event->modifiers() == Qt::ControlModifier) {
            bool clearedPreview = false;
            _viewerManager->forEachBaseViewer([&clearedPreview](VolumeViewerBase* baseViewer) {
                if (!baseViewer) {
                    return;
                }
                if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                    if (viewer->hasSameWrapAnnotationPreview()) {
                        viewer->clearSameWrapAnnotationPreview();
                        clearedPreview = true;
                    }
                }
            });
            if (clearedPreview) {
                event->accept();
                return;
            }

            bool undone = false;
            _viewerManager->forEachBaseViewer([&undone](VolumeViewerBase* baseViewer) {
                if (undone || !baseViewer) {
                    return;
                }
                if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                    undone = viewer->undoSameWrapAnnotation();
                }
            });
            if (undone) {
                event->accept();
                return;
            }
        }
    }

    // Shift+G decreases Z-scroll sensitivity, Shift+H increases it. The
    // preference is global, so keep every workspace's viewers in sync.
    if (event->modifiers() == vc3d::keybinds::keypress::SliceStepDecrease.modifiers && _viewerManager) {
        if (event->key() == vc3d::keybinds::keypress::SliceStepDecrease.key) {
            const double sensitivity = _viewerManager->zScrollSensitivity() - 0.1;
            for (auto* manager : ViewerManager::allManagers()) {
                manager->setZScrollSensitivity(sensitivity);
            }
            event->accept();
            return;
        } else if (event->key() == vc3d::keybinds::keypress::SliceStepIncrease.key) {
            const double sensitivity = _viewerManager->zScrollSensitivity() + 0.1;
            for (auto* manager : ViewerManager::allManagers()) {
                manager->setZScrollSensitivity(sensitivity);
            }
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
    if (_viewerManager && _wrapAnnotationWidget &&
        _wrapAnnotationWidget->sameWrapAnnotationEnabled() &&
        event->key() == Qt::Key_Shift) {
        _viewerManager->forEachBaseViewer([event](VolumeViewerBase* baseViewer) {
            if (!baseViewer) {
                return;
            }
            if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject())) {
                viewer->onKeyRelease(event->key(), event->modifiers());
            }
        });
    }

    if (_segmentationModule && _segmentationModule->handleKeyRelease(event)) {
        return;
    }

    QMainWindow::keyReleaseEvent(event);
}

void CWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    QTimer::singleShot(0, this, &CWindow::updateProjectNameLabel);
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
    if (_segmentWorkspaceWindow) {
        settings.setValue(vc3d::settings::window::STATE, _segmentWorkspaceWindow->saveState());
    }
    if (_workspaceTabs) {
        settings.setValue(WORKSPACE_TAB_SETTING, _workspaceTabs->currentIndex());
        if (auto* workspace = _workspaceTabs->currentWidget())
            settings.setValue(WORKSPACE_ID_SETTING, workspace->property("workspaceId").toString());
    }
    settings.setValue(vc3d::settings::window::RESTORE_DISABLED, false);
    settings.setValue(vc3d::settings::window::RESTORE_IN_PROGRESS, false);
    writeWindowStateMeta(settings,
                         windowStateScreenSignature(),
                         windowStateQtVersion(),
                         windowStateAppVersion());
    settings.sync();
}

void CWindow::closeEvent(QCloseEvent* event)
{
    if (_spiralWorkspace && !_spiralCloseGuardBypass
        && _spiralWorkspace->hasPendingBrushWork()) {
        event->ignore();
        _spiralWorkspace->requestSessionExit([this]() {
            _spiralCloseGuardBypass = true;
            close();
        });
        return;
    }
    _destroyingWindow = true;
    if (_lineAnnotationController) {
        _lineAnnotationController->saveOpenFibers();
    }
    saveWindowState();
    event->accept();
}

void CWindow::setWidgetsEnabled(bool state)
{
    ui.grpVolManager->setEnabled(state);
    ui.btnOpenDataCatalog->setEnabled(!state);
    if (_viewerControlsPanel) {
        _viewerControlsPanel->setViewControlsEnabled(state);
        _viewerControlsPanel->setOverlayWindowAvailable(_volumeOverlay && _volumeOverlay->hasOverlaySelection());
    }
}

// Update the widgets
void CWindow::UpdateView(void)
{
    if (!_state->hasVpkg() && _state->currentVolume() == nullptr) {
        setWidgetsEnabled(false);  // Disable Widgets for User
        ui.lblVpkgName->setText("[ No Volume Package Loaded ]");
        updateVolumePackageEmptyState();
        return;
    }

    setWidgetsEnabled(true);  // Enable Widgets for User
    updateVolumePackageEmptyState();

    // show volume package name
    UpdateVolpkgLabel(0);

    syncVolumeSelectionControls();

    update();
}

void CWindow::updateVolumePackageEmptyState()
{
    const bool hasOpenPackage = _state && (_state->hasVpkg() || _state->currentVolume() != nullptr);
    const bool catalogVisible = _menuController && _menuController->isOpenDataCatalogVisible();
    ui.btnOpenDataCatalog->setVisible(!hasOpenPackage && !catalogVisible);
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
    showStatusBarMessage(text, timeout);
}

void CWindow::showStatusBarMessage(const QString& text, int timeout)
{
    if (!_statusMessageLabel) {
        if (statusBar()) {
            statusBar()->showMessage(text, timeout);
        }
        return;
    }

    _statusMessageLabel->setText(text);
    _statusMessageLabel->setToolTip(text);

    if (_statusMessageTimer) {
        _statusMessageTimer->stop();
        if (timeout > 0) {
            _statusMessageTimer->start(timeout);
        }
    }
}

void CWindow::clearStatusBarMessage()
{
    if (_statusMessageTimer) {
        _statusMessageTimer->stop();
    }
    if (_statusMessageLabel) {
        _statusMessageLabel->clear();
        _statusMessageLabel->setToolTip(QString());
    } else if (statusBar()) {
        statusBar()->clearMessage();
    }
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
        showStatusBarMessage(_segmentationGrowthStatusText, 0);
    } else if (_segmentationGrowthWarning) {
        _segmentationGrowthWarning->clear();
        _segmentationGrowthWarning->setVisible(false);
        if (_statusMessageLabel && _statusMessageLabel->text() == _segmentationGrowthStatusText) {
            clearStatusBarMessage();
        } else if (statusBar()->currentMessage() == _segmentationGrowthStatusText) {
            clearStatusBarMessage();
        }
        _segmentationGrowthStatusText.clear();
    }
}

void CWindow::onZScrollSensitivityChanged(double sensitivity)
{
    Q_UNUSED(sensitivity);
    updateSharedStatusLabel();
}

void CWindow::onSharedCacheStatsChanged(const QStringList& items)
{
    if (!items.isEmpty())
        _sharedCacheStatsItems = items;
    updateSharedStatusLabel();
}

void CWindow::updateFiberLengthUnits()
{
    if (_destroyingWindow) {
        return;
    }
    // Fiber lengths are in the open volume's voxels.
    std::optional<double> voxelSizeUm;
    if (const auto volume = _state ? _state->currentVolume() : nullptr; volume && volume->voxelSize() > 0) {
        voxelSizeUm = volume->voxelSize();
    }
    if (_fiberWidget) {
        _fiberWidget->setAnnotationVoxelSizeUm(voxelSizeUm);
    }
    updateLineAnnotationLengthLabel();
}

void CWindow::updateLineAnnotationLengthLabel()
{
    if (_destroyingWindow || !_lineAnnotationLengthLabel) {
        return;
    }
    auto* dialog = _workspaceTabs
        ? qobject_cast<LineAnnotationDialog*>(_workspaceTabs->currentWidget())
        : nullptr;
    const uint64_t fiberId = (dialog && _lineAnnotationController)
        ? _lineAnnotationController->fiberIdForDialog(dialog)
        : 0;
    const auto summary = fiberId != 0 ? _lineAnnotationController->fiberSummary(fiberId) : std::nullopt;
    if (!summary) {
        _lineAnnotationLengthLabel->clear();
        _lineAnnotationLengthLabel->hide();
        return;
    }
    // Fiber lengths are in the open volume's voxels, as in the Fibers dock.
    const auto volume = _state->currentVolume();
    const std::optional<double> voxelSizeUm =
        volume && volume->voxelSize() > 0 ? std::optional(volume->voxelSize()) : std::nullopt;
    _lineAnnotationLengthLabel->setText(
        tr("Fiber length %1").arg(vc3d::fiber_length::formatLength(summary->annotatedLengthVx, voxelSizeUm)));
    _lineAnnotationLengthLabel->show();
}

void CWindow::updateSharedStatusLabel()
{
    if (!_sharedCacheStatsLabel)
        return;

    QStringList items = _sharedCacheStatsItems;
    if (items.isEmpty())
        items << tr("RAM --  disk -- GiB  net --");
    if (_viewerManager) {
        items << tr("Z sens: %1")
            .arg(_viewerManager->zScrollSensitivity(), 0, 'f', 1);
    }
    _sharedCacheStatsLabel->setText(items.join(QStringLiteral("  ")));
}

// Open volume package
bool CWindow::OpenVolume(const QString& path,
                         bool interactive,
                         QString* errorMessage,
                         const QString& preferredVolumeId,
                         VolumeOpenError* openError)
{
    if (errorMessage) {
        errorMessage->clear();
    }
    if (openError) {
        *openError = VolumeOpenError::None;
    }
    QString aVpkgPath = path;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        if (!interactive) {
            if (errorMessage) {
                *errorMessage = tr("Project path is empty.");
            }
            if (openError) {
                *openError = VolumeOpenError::PackageLoadFailed;
            }
            return false;
        }
        aVpkgPath = QFileDialog::getOpenFileName(
            this, tr("Open Project"), settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
            tr("Project (*.vcproj)"),
            nullptr, QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);
        if (aVpkgPath.isEmpty()) {
            Logger()->info("Open project canceled");
            return false;
        }
    }

    // Projects are written as the user; a headless open uses the settings' name.
    const QString author = interactive && _menuController ? _menuController->objectStoreAuthor()
                                                          : CState::storeAuthor();
    if (author.isEmpty()) {
        return false;
    }
    std::shared_ptr<VolumePkg> package;
    QString loadError;
    try {
        vc::project::LoadOptions options;
        options.catalog = vc3d::opendata::cachedCatalog();
        package = VolumePkg::open(aVpkgPath.toStdString(), options);
    } catch (const std::exception& e) {
        Logger()->error("Failed to open project: {}", e.what());
        loadError = QString::fromUtf8(e.what());
    }
    if (!package) {
        Logger()->error("Cannot open project: {}", aVpkgPath.toStdString());
        const QString message = loadError.isEmpty()
            ? tr("Project failed to load.")
            : tr("Project failed to load: %1").arg(loadError);
        if (errorMessage) {
            *errorMessage = message;
        }
        if (openError) {
            *openError = VolumeOpenError::PackageLoadFailed;
        }
        if (interactive) {
            QMessageBox::warning(this, tr("Error"), message);
        }
        return false;
    }

    if (!preferredVolumeId.isEmpty()) {
        const auto ids = package->volumeIDs();
        const auto id = preferredVolumeId.toStdString();
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            if (errorMessage) {
                *errorMessage = tr("Unknown volume id: %1").arg(preferredVolumeId);
            }
            if (openError) {
                *openError = VolumeOpenError::VolumeNotFound;
            }
            return false;
        }
        try {
            (void)package->volume(id);
        } catch (const std::exception& e) {
            if (errorMessage) {
                *errorMessage = tr("Failed to load volume %1: %2")
                                    .arg(preferredVolumeId, QString::fromUtf8(e.what()));
            }
            if (openError) {
                *openError = VolumeOpenError::PackageLoadFailed;
            }
            return false;
        }
    }

    // Line-annotation sessions must end — finalized and saved — while the
    // package they belong to is still the active one; after the swap, their
    // close path would save into the new package. A refusal (a running
    // optimization, a failed finalization) aborts the switch with the
    // workspace intact.
    if (_state->prepareStoreSwitch && !_state->prepareStoreSwitch()) {
        if (errorMessage) {
            *errorMessage = tr("Pending edits could not be saved; "
                               "the project was not switched.");
        }
        if (openError) {
            *openError = VolumeOpenError::PackageLoadFailed;
        }
        return false;
    }

    CloseVolume();
    // Without a choice, the volume the user last viewed in the project.
    const auto ids = package->volumeIDs();
    const auto remembered = QString::fromStdString(package->currentVolumeId());
    const QString volumeId = !preferredVolumeId.isEmpty() ? preferredVolumeId
        : std::find(ids.begin(), ids.end(), remembered.toStdString()) != ids.end() ? remembered : QString();
    _state->setVpkg(std::move(package));
    refreshCurrentVolumePackageUi(volumeId);
    if (_menuController) {
        _menuController->updateRecentVolpkgList(aVpkgPath);
    }

    if (_fileWatcher) {
        _fileWatcher->startWatching();
    }
    return true;
}

void CWindow::attachStore(const QString& path)
{
    try {
        if (!_state->hasVpkg()) throw std::runtime_error("no project is open");
        _state->attachStore(path.toStdString(), CState::storeAuthor());
    } catch (const std::exception& error) {
        const auto message = tr("Cannot attach store %1: %2").arg(path, QString::fromUtf8(error.what()));
        qWarning().noquote() << message;
        statusBar()->showMessage(message, 10000);
    }
}

bool CWindow::openVolumePackage(const QString& path,
                                bool interactive,
                                QString* errorMessage,
                                const QString& preferredVolumeId,
                                VolumeOpenError* openError)
{
    const bool opened = OpenVolume(
        path, interactive, errorMessage, preferredVolumeId, openError);
    UpdateView();
    return opened;
}

std::vector<QComboBox*> CWindow::volumeSelectionControls() const
{
    std::vector<QComboBox*> selectors;
    if (volSelect) {
        selectors.push_back(volSelect);
    }
    for (const auto& selector : _annotationVolumeSelects) {
        if (selector) {
            selectors.push_back(selector.data());
        }
    }
    if (_spiralWorkspace && _spiralWorkspace->volumeSelectionControl()) {
        selectors.push_back(_spiralWorkspace->volumeSelectionControl());
    }
    return selectors;
}

void CWindow::connectVolumeSelector(QComboBox* selector)
{
    if (!selector) {
        return;
    }
    connect(selector,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            [this, selector](const int& index) {
                auto vpkg = _state->vpkg();
                if (vpkg && index >= 0) {
                    const QString volumeId = selector->currentData().toString();
                    std::shared_ptr<Volume> newVolume;
                    try {
                        newVolume = vpkg->volume(volumeId.toStdString());
                    } catch (const std::out_of_range&) {
                        QMessageBox::warning(this, "Error", "Could not load volume.");
                        syncVolumeSelectionControls();
                        return;
                    }
                    setVolume(newVolume);
                    syncVolumeSelectionControls(volumeId);
                }
            });
}

QString CWindow::lastVolumeSettingKeyForCurrentPackage() const
{
    if (!_state || !_state->vpkg()) {
        return {};
    }

    QString projectPath = QString::fromStdString(_state->vpkg()->path().string());
    if (projectPath.isEmpty()) {
        return {};
    }

    const QFileInfo info(projectPath);
    QString stablePath = info.canonicalFilePath();
    if (stablePath.isEmpty()) {
        stablePath = info.absoluteFilePath();
    }
    if (stablePath.isEmpty()) {
        stablePath = projectPath;
    }

    return QStringLiteral("%1%2").arg(
        QString::fromLatin1(vc3d::settings::project::LAST_VOLUME_PREFIX),
        QString::fromLatin1(QUrl::toPercentEncoding(stablePath)));
}

QString CWindow::rememberedVolumeIdForCurrentPackage() const
{
    const QString key = lastVolumeSettingKeyForCurrentPackage();
    if (key.isEmpty()) {
        return {};
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(key).toString();
}

void CWindow::rememberCurrentVolumeForPackage(const QString& volumeId) const
{
    if (volumeId.isEmpty() || !_state || !_state->vpkg() ||
        !_state->vpkg()->hasVolume(volumeId.toStdString())) {
        return;
    }

    const QString key = lastVolumeSettingKeyForCurrentPackage();
    if (key.isEmpty()) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(key, volumeId);
}

std::optional<vcr::catalog::Affine> CWindow::volumeTransformForSwitch(const std::string& from, const std::string& to) const
{
    // The store's resolution (catalog registrations, its own and its route
    // choices); without a store, the cached catalog's registrations.
    vc::objects::VolumeTransforms transforms;
    if (const auto attachment = _state->storeFor("volume_transform")) {
        _state->objectStoreSession()->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
            transforms = db.volumeTransforms(txn);
        });
    } else if (const auto catalog = vc3d::opendata::cachedCatalog()) {
        transforms.graph.addCatalog(*catalog);
    }
    const auto resolution = transforms.resolve(from, to);
    if (resolution.status != vcr::catalog::TransformStatus::Resolved) return std::nullopt;
    return resolution.resolved().forward;
}

QWidget* CWindow::createAnnotationVolumeSelector(QWidget* parent)
{
    auto* volumeSelector = new VolumeSelector(parent);
    volumeSelector->setLabelVisible(false);
    if (auto* combo = volumeSelector->comboBox()) {
        combo->setObjectName(QStringLiteral("annotationVolumeSelect"));
        combo->setMinimumWidth(130);
        combo->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        _annotationVolumeSelects.push_back(QPointer<QComboBox>(combo));
        connectVolumeSelector(combo);
    }
    refreshVolumeSelectionUi();
    return volumeSelector;
}

void CWindow::syncVolumeSelectionControls(const QString& activeVolumeId)
{
    QString targetId = activeVolumeId;
    if (targetId.isEmpty() && _state) {
        targetId = QString::fromStdString(_state->currentVolumeId());
    }

    for (QComboBox* selector : volumeSelectionControls()) {
        if (!selector) {
            continue;
        }
        const int index = targetId.isEmpty() ? -1 : selector->findData(targetId);
        if (index >= 0 && selector->currentIndex() != index) {
            const QSignalBlocker blocker(selector);
            selector->setCurrentIndex(index);
        }
        selector->setEnabled(can_change_volume_());
    }
}

void CWindow::refreshVolumeSelectionUi(const QString& preferredVolumeId)
{
    const auto selectors = volumeSelectionControls();
    if (selectors.empty() || !_state || !_state->vpkg()) {
        return;
    }

    QVector<QPair<QString, QString>> volumeEntries;
    QVector<QPair<QString, QString>> openDataVolumeIdMap;
    std::set<QString> preferredOpenDataSourceIds;
    std::vector<QString> orderedIds;
    QString activeCandidate = preferredVolumeId;
    const bool hasExplicitPreferredVolume = !activeCandidate.isEmpty();
    QString currentComboId;
    for (QComboBox* selector : selectors) {
        if (selector && selector->currentIndex() >= 0) {
            currentComboId = selector->currentData().toString();
            if (!currentComboId.isEmpty()) {
                break;
            }
        }
    }
    const QString currentVolumeId = QString::fromStdString(_state->currentVolumeId());

    auto hasVolume = [&](const QString& volumeId) {
        for (const auto& id : orderedIds) {
            if (id == volumeId) {
                return true;
            }
        }
        return false;
    };

    auto openDataVolumeIdMappedToLoadedId = [&](const QString& catalogVolumeId) {
        for (const auto& mapping : openDataVolumeIdMap) {
            if (mapping.first == catalogVolumeId) {
                return mapping.second;
            }
        }
        return QString{};
    };

    QString bestGrowthVolumeId;
    bool preferredVolumeFound = false;
    const auto volumeIds = vc3d::orderedDisplayVolumeIds(*_state->vpkg());
    const auto volumeSuffixes = vc3d::volumeDisplaySuffixes(*_state->vpkg());
    for (const auto& id : volumeIds) {
        try {
            auto vol = _state->vpkg()->volume(id);
            const QString idStr = QString::fromStdString(id);
            const QString nameStr = QString::fromStdString(vol->name());
            const QString label = vc3d::volumeDisplayLabel(*_state->vpkg(), id,
                nameStr.isEmpty() ? idStr : QStringLiteral("%1 (%2)").arg(nameStr, idStr),
                volumeSuffixes.value(idStr));

            orderedIds.push_back(idStr);
            volumeEntries.append({idStr, label});
            // A catalog volume id maps to a loaded volume showing it, its
            // own zarr at level 0 first.
            if (const auto view = _state->vpkg()->catalogView(id)) {
                const QString catalogVolumeId = QString::fromStdString(view->volumeId);
                const bool preferredOpenDataSource = view->source && view->level == 0;
                const auto existing = std::find_if(
                    openDataVolumeIdMap.begin(), openDataVolumeIdMap.end(),
                    [&](const auto& mapping) { return mapping.first == catalogVolumeId; });
                if (existing == openDataVolumeIdMap.end()) {
                    openDataVolumeIdMap.append({catalogVolumeId, idStr});
                } else if (preferredOpenDataSource &&
                           !preferredOpenDataSourceIds.contains(catalogVolumeId)) {
                    existing->second = idStr;
                }
                if (preferredOpenDataSource)
                    preferredOpenDataSourceIds.insert(catalogVolumeId);
            }

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
        activeCandidate = openDataVolumeIdMappedToLoadedId(activeCandidate);
    }
    if (activeCandidate.isEmpty() && !hasExplicitPreferredVolume) {
        const QString rememberedVolumeId = rememberedVolumeIdForCurrentPackage();
        if (!rememberedVolumeId.isEmpty() && hasVolume(rememberedVolumeId)) {
            activeCandidate = rememberedVolumeId;
        }
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

    for (QComboBox* selector : selectors) {
        if (!selector) {
            continue;
        }
        const QSignalBlocker blocker{selector};
        selector->clear();
        for (const auto& [id, label] : volumeEntries) {
            selector->addItem(label, QVariant(id));
            const int row = selector->count() - 1;
            const auto sourcePath = vc3d::volumeDisplayPath(*_state->vpkg(), id.toStdString());
            selector->setItemData(row, sourcePath, Qt::ToolTipRole);
        }
        if (activeCandidate.isEmpty()) {
            if (selector->count() > 0) {
                selector->setCurrentIndex(0);
            }
        } else {
            const int activeIndex = selector->findData(activeCandidate);
            selector->setCurrentIndex(activeIndex >= 0 ? activeIndex : 0);
        }
        vc3d::refreshVolumeComboTooltip(selector);
    }

    QString activeId;
    for (QComboBox* selector : selectors) {
        if (selector && selector->count() > 0) {
            activeId = selector->currentData().toString();
            break;
        }
    }
    syncVolumeSelectionControls(activeId);

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
    if (_spiralWorkspace) {
        _spiralWorkspace->synchronizeVolume(nullptr);
    }

    updateNormalGridAvailability();
    if (_segmentationWidget) {
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

    if (_surfacePanel) {
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
    for (QComboBox* selector : volumeSelectionControls()) {
        if (!selector) continue;
        const QSignalBlocker blocker(selector);
        selector->clear();
        selector->setEnabled(false);
    }
    if (_viewerManager) {
        _viewerManager->setOverlayVolume(nullptr, {});
    }
    if (_spiralWorkspace && _spiralWorkspace->viewerManager()) {
        _spiralWorkspace->viewerManager()->setOverlayVolume(nullptr, {});
    }

}

// Handle open request
auto CWindow::can_change_volume_() -> bool
{
    if (_state->hasVpkg() && _state->vpkg()->numberOfVolumes() > 1) {
        return true;
    }
    // Also allow switching when volSelect has multiple remote volumes
    if (_state && _state->currentVolume()) {
        for (QComboBox* selector : volumeSelectionControls()) {
            if (selector && selector->count() > 1) {
                return true;
            }
        }
    }
    return false;
}

void CWindow::onSurfaceActivated(const QString& surfaceId, const std::shared_ptr<QuadSurface>& surface)
{
    const std::string previousSurfId = _state->activeSurfaceId();
    const std::string newSurfId = surfaceId.toStdString();

    if (_state && !newSurfId.empty() && surface) {
        _state->setActiveSurface(newSurfId, surface);
    } else {
        _state->clearActiveSurface();
    }

    auto surf = _state->activeSurface().lock();

    _state->setSurface("segmentation", surf, false, false);

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
                           << ":"
                           << e.what();
                _state->clearActiveSurface();
                _state->setSurface("segmentation", nullptr, false, false);
                surf.reset();
            }
        }

        loadLasagnaResults(surf.get());
    }

    const bool moveOnSurfaceChange = moveOnSurfaceChangeEnabled();
    const bool activatingAxisAlignedPlane =
        newSurfId == "xy plane" || newSurfId == "seg xz" || newSurfId == "seg yz";
    if (moveOnSurfaceChange && _axisAlignedSliceController && !activatingAxisAlignedPlane) {
        _axisAlignedSliceController->resetAll();
    }

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

    if (moveOnSurfaceChange || activatingAxisAlignedPlane) {
        try {
            if (surf) {
                _axisAlignedSliceController->applyOrientation(surf.get());
            } else {
                _axisAlignedSliceController->applyOrientation();
            }
        } catch (const std::exception& e) {
            qWarning() << "Failed to apply surface orientation for"
                       << surfaceId
                       << ":"
                       << e.what();
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, false, false);
            _axisAlignedSliceController->applyOrientation();
        }
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->applyFilters();
    }

}

void CWindow::onSurfaceActivatedPreserveEditing(const QString& surfaceId, const std::shared_ptr<QuadSurface>& surface)
{
    const std::string previousSurfId = _state->activeSurfaceId();
    const std::string newSurfId = surfaceId.toStdString();

    if (_state && !newSurfId.empty() && surface) {
        _state->setActiveSurface(newSurfId, surface);
    } else {
        _state->clearActiveSurface();
    }

    auto surf = _state->activeSurface().lock();

    _state->setSurface("segmentation", surf, false, false);

    if (newSurfId != previousSurfId && _segmentationModule) {
        try {
            _segmentationModule->onActiveSegmentChanged(surf.get());
        } catch (const std::exception& e) {
            qWarning() << "Failed to activate surface"
                       << surfaceId
                       << ":"
                       << e.what();
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, false, false);
            surf.reset();
        }

        loadLasagnaResults(surf.get());

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

    if (moveOnSurfaceChangeEnabled()) {
        try {
            if (surf) {
                _axisAlignedSliceController->applyOrientation(surf.get());
            } else {
                _axisAlignedSliceController->applyOrientation();
            }
        } catch (const std::exception& e) {
            qWarning() << "Failed to apply surface orientation for"
                       << surfaceId
                       << ":"
                       << e.what();
            _state->clearActiveSurface();
            _state->setSurface("segmentation", nullptr, false, false);
            _axisAlignedSliceController->applyOrientation();
        }
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->applyFilters();
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

QString CWindow::getCurrentVolumePath() const
{
    auto volume = _state->currentVolume();
    if (volume == nullptr) {
        return QString();
    }
    if (volume->isRemote()) {
        return QString::fromStdString(volume->remoteLocator());
    }
    return QString::fromStdString(volume->path().string());
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
        _surfacePanel->applyFilters();
    }
}

void CWindow::refreshFocusBoundsUi()
{
    if (_fiberMapWorkspace) {
        // Map geometry indexes the annotation volume, as stored bounds do.
        _fiberMapWorkspace->setFocusBounds(_state ? _state->activeAnnotationFocusBounds() : std::nullopt);
    }
    if (!ui.chkFocusBBox || !ui.focusBBoxMin || !ui.focusBBoxMax) {
        return;
    }

    std::optional<Rect3D> bounds = _state ? _state->focusBounds() : std::nullopt;
    if (!bounds && _state && _state->currentVolume()) {
        const auto [width, height, depth] = _state->currentVolume()->shapeXyz();
        bounds = Rect3D{{0.0f, 0.0f, 0.0f},
                        {static_cast<float>(std::max(0, width - 1)),
                         static_cast<float>(std::max(0, height - 1)),
                         static_cast<float>(std::max(0, depth - 1))}};
    }

    const QSignalBlocker checkBlocker(ui.chkFocusBBox);
    const QSignalBlocker minBlocker(ui.focusBBoxMin);
    const QSignalBlocker maxBlocker(ui.focusBBoxMax);
    ui.chkFocusBBox->setChecked(_state && _state->focusBoundsEnabled());
    const bool haveVolume = _state && _state->currentVolume();
    ui.chkFocusBBox->setEnabled(haveVolume);
    ui.focusBBoxMin->setEnabled(haveVolume);
    ui.focusBBoxMax->setEnabled(haveVolume);
    if (!bounds) {
        ui.focusBBoxMin->clear();
        ui.focusBBoxMax->clear();
        return;
    }
    const auto formatPoint = [](const cv::Vec3f& point) {
        return QStringLiteral("%1, %2, %3")
            .arg(static_cast<int>(std::lround(point[0])))
            .arg(static_cast<int>(std::lround(point[1])))
            .arg(static_cast<int>(std::lround(point[2])));
    };
    ui.focusBBoxMin->setText(formatPoint(bounds->low));
    ui.focusBBoxMax->setText(formatPoint(bounds->high));
}

void CWindow::onFocusBoundsEdited()
{
    if (!_state || !_state->currentVolume()) {
        refreshFocusBoundsUi();
        return;
    }

    const auto parsePoint = [](const QString& text) -> std::optional<cv::Vec3f> {
        const QStringList parts = text.trimmed().split(',');
        if (parts.size() != 3) {
            return std::nullopt;
        }
        cv::Vec3f point;
        for (int axis = 0; axis < 3; ++axis) {
            bool ok = false;
            const int value = parts[axis].trimmed().toInt(&ok);
            if (!ok) {
                return std::nullopt;
            }
            point[axis] = static_cast<float>(value);
        }
        return point;
    };

    auto low = parsePoint(ui.focusBBoxMin->text());
    auto high = parsePoint(ui.focusBBoxMax->text());
    if (!low || !high) {
        refreshFocusBoundsUi();
        return;
    }

    const auto [width, height, depth] = _state->currentVolume()->shapeXyz();
    const cv::Vec3f volumeHigh{
        static_cast<float>(std::max(0, width - 1)),
        static_cast<float>(std::max(0, height - 1)),
        static_cast<float>(std::max(0, depth - 1))};
    for (int axis = 0; axis < 3; ++axis) {
        (*low)[axis] = std::clamp((*low)[axis], 0.0f, volumeHigh[axis]);
        (*high)[axis] = std::clamp((*high)[axis], 0.0f, volumeHigh[axis]);
    }
    _state->setFocusBounds(Rect3D{*low, *high});
    refreshFocusBoundsUi();
}

void CWindow::onFocusBoundsToggled(bool enabled)
{
    if (!_state || !_state->currentVolume()) {
        refreshFocusBoundsUi();
        return;
    }
    if (enabled && !_state->focusBounds()) {
        onFocusBoundsEdited();
        if (!_state->focusBounds()) {
            refreshFocusBoundsUi();
            return;
        }
    }
    _state->setFocusBoundsEnabled(enabled);
}

void CWindow::onZoomIn()
{
    if (auto* viewer = activeBaseViewer()) {
        viewer->adjustZoomByFactor(1.15f);
    }
}

void CWindow::onFocusPOIChanged(std::string name, POI* poi)
{
    // Slice-plane reorientation and viewer recentering are handled centrally
    // by ViewerManager::handleFocusPoiChanged; only Main-specific UI here.
    if (name == "focus" && poi) {
        lblLocFocus->setText(QString("%1, %2, %3")
            .arg(static_cast<int>(poi->p[0]))
            .arg(static_cast<int>(poi->p[1]))
            .arg(static_cast<int>(poi->p[2])));

        if (_surfacePanel) {
            _surfacePanel->applyFilters();
        }
    }
}

void CWindow::onPointDoubleClicked(vc::PointRef point)
{
    auto point_opt = _state->pointCollection()->getPoint(point);
    if (point_opt) {
        centerFocusAt(point_opt->p, cv::Vec3f(0, 0, 0), "");
    }
}

void CWindow::onConvertPointToAnchor(vc::PointRef point)
{
    auto point_opt = _state->pointCollection()->getPoint(point);
    if (!point_opt) {
        showStatusBarMessage(tr("Point not found"), 2000);
        return;
    }

    // Get the segmentation surface to project the point onto
    auto seg_surface = _state->surface("segmentation");
    auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get());
    if (!quad_surface) {
        showStatusBarMessage(tr("No active segmentation surface for anchor conversion"), 3000);
        return;
    }

    // Find the 2D grid location of this point on the surface
    constexpr float kMaxDistance = 10.0f;
    const auto hit = _viewerManager ? _viewerManager->locate(*quad_surface, point_opt->p, kMaxDistance) : std::nullopt;
    if (!hit) {
        showStatusBarMessage(tr("Point is farther than %1 voxels from the surface").arg(kMaxDistance), 3000);
        return;
    }
    const cv::Vec3f ptr = hit->ptr;

    // Get the raw grid location (internal coordinates)
    cv::Vec3f loc_3d = quad_surface->loc_raw(ptr);
    cv::Vec2f anchor2d(loc_3d[0], loc_3d[1]);

    // Set the anchor2d on the collection
    _state->pointCollection()->setCollectionAnchor2d(point.collectionId, anchor2d);

    // Remove the point (it's now represented by the anchor)
    _state->pointCollection()->removePoint(point);

    showStatusBarMessage(tr("Converted point to anchor at grid position (%1, %2)").arg(anchor2d[0]).arg(anchor2d[1]), 3000);
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
        showStatusBarMessage(tr("Coordinates copied to clipboard: %1").arg(coords), 2000);
    }
}

void CWindow::onResetAxisAlignedRotations()
{
    _axisAlignedSliceController->resetRotations();
    _axisAlignedSliceController->applyOrientation();
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }
    showStatusBarMessage(tr("All plane rotations reset"), 2000);
}

void CWindow::onAxisOverlayVisibilityToggled(bool enabled)
{
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && _axisAlignedSliceController->isEnabled());
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
    _axisAlignedSliceController->setEnabled(enabled, ui.chkAxisOverlays);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES, enabled ? "1" : "0");
}

void CWindow::onMoveOnSurfaceChangedToggled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE, enabled ? "1" : "0");

    for (auto* manager : ViewerManager::allManagers()) {
        manager->setResetViewOnSurfaceChangeDefault(enabled);
    }
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

    // Set flag BEFORE beginEditingSession so the surface change doesn't reset view
    if (_viewerManager) {
        _viewerManager->setSegmentationResetViewSuppressed(enabled);
    }

    if (enabled) {
        auto activeSurfaceShared = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));

        if (!_segmentationModule->beginEditingSession(activeSurfaceShared)) {
            showStatusBarMessage(tr("Unable to start segmentation editing"), 3000);
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
    }

    const QString message = enabled
        ? tr("Segmentation editing enabled")
        : tr("Segmentation editing disabled");
    showStatusBarMessage(message, 2000);
}

void CWindow::onSegmentationStopToolsRequested()
{
    if (!initializeCommandLineRunner()) {
        return;
    }
    if (_cmdRunner) {
        _cmdRunner->cancel();
        showStatusBarMessage(tr("Cancelling running tools..."), 3000);
    }
}

void CWindow::onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                        SegmentationGrowthDirection direction,
                                        int steps,
                                        bool inpaintOnly)
{
    if (!_segmentationGrower) {
        showStatusBarMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

    updateNormalGridAvailability();

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
        _normalGridPath
    };

    if (!_segmentationGrower->start(volumeContext, method, direction, steps, inpaintOnly)) {
        return;
    }
}

void CWindow::onCopyWithNtRequested()
{
    if (!_segmentationGrower) {
        showStatusBarMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

    updateNormalGridAvailability();

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
        _normalGridPath
    };

    if (!_segmentationGrower->startCopyWithNt(volumeContext)) {
        return;
    }
}

void CWindow::onFocusViewsRequested(uint64_t collectionId, std::optional<vc::PointRef> point)
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
    if (point) {
        auto point_opt = pointCollection->getPoint(*point);
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

        // Keep the volumetric-camera azimuth folded into the plane (see
        // AxisAlignedSliceController::applyOrientation).
        float inPlaneRot = AxisAlignedSliceController::azimuthInPlaneRotation(
            *planeShared,
            _axisAlignedSliceController
                ? _axisAlignedSliceController->volumetricAzimuthDeg(planeName)
                : 0.0f);

        // Adjust in-plane rotation so Z projects "up"
        const cv::Vec3f upAxis(0.0f, 0.0f, 1.0f);
        const cv::Vec3f projectedUp = projectVectorOntoPlane(upAxis, normal);
        const cv::Vec3f desiredUp = normalizeOrZero(projectedUp);
        if (cv::norm(desiredUp) > kEpsilon) {
            const cv::Vec3f currentUp = planeShared->basisY();
            inPlaneRot += signedAngleBetween(currentUp, desiredUp, normal);
        }
        if (std::abs(inPlaneRot) > kEpsilon) {
            planeShared->setInPlaneRotation(inPlaneRot);
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
    focus->surfacePtr.reset();
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

    showStatusBarMessage(tr("Focused & aligned view to %1 points").arg(pts.size()), 3000);
}

namespace {
// The project's store, then the stores beside it.
std::vector<vc::objects::AttachmentIdentity> openStores(const CState& state)
{
    std::vector<vc::objects::AttachmentIdentity> stores;
    for (const auto& attached : state.attachedStores()) stores.push_back(attached);
    return stores;
}
}

void CWindow::updateHubStatus()
{
    const auto session = _state->objectStoreSession();
    QStringList texts;
    std::vector<std::string> conflicts;
    for (const auto& attachment : openStores(*_state)) {
        const auto status = session->hubStatus(attachment);
        if (!status) continue;
        texts << tr("%1 Hub: %2").arg(QString::fromStdString(session->path(attachment).stem().string()),
                                       hubStatusSummary(*status));
        for (const auto& conflict : status->conflicts) conflicts.push_back(conflict.uuid);
    }
    if (texts.isEmpty()) {
        _hubStatusLabel->hide();
        _hubPromptedConflicts.clear();
        return;
    }
    _hubStatusLabel->setText(texts.join(QStringLiteral("  ")));
    _hubStatusLabel->show();
    // Ask about a set of conflicts once; Sync Now asks again.
    if (conflicts.empty() || conflicts == _hubPromptedConflicts) return;
    _hubPromptedConflicts = conflicts;
    promptHubConflicts();
}

void CWindow::promptHubConflicts()
{
    const auto session = _state->objectStoreSession();
    for (const auto& attachment : openStores(*_state)) {
        const auto status = session->hubStatus(attachment);
        if (!status || status->conflicts.empty()) continue;
        QStringList uuids;
        for (const auto& conflict : status->conflicts) uuids << QString::fromStdString(conflict.uuid);
        QMessageBox box(QMessageBox::Warning, tr("Hosted object store conflict"),
            tr("%n record(s) of %1 changed both here and on the hub:", nullptr, int(uuids.size()))
                .arg(QString::fromStdString(session->path(attachment).filename().string())) +
                QStringLiteral("\n") + uuids.mid(0, 10).join(QStringLiteral("\n")) +
                (uuids.size() > 10 ? QStringLiteral("\n...") : QString()) +
                tr("\n\nUse the hub's versions (discarding your changes to them), or keep yours and publish them?"),
            QMessageBox::NoButton, this);
        auto* useHub = box.addButton(tr("Use Hub Versions"), QMessageBox::DestructiveRole);
        auto* keepMine = box.addButton(tr("Keep Mine"), QMessageBox::AcceptRole);
        box.addButton(tr("Later"), QMessageBox::RejectRole);
        box.exec();
        if (box.clickedButton() == useHub) session->resolveHubConflicts(attachment, "hub");
        else if (box.clickedButton() == keepMine) session->resolveHubConflicts(attachment, "local");
    }
}
