#include "CWindow.hpp"

#include <cstdlib>

#include "WindowRangeWidget.hpp"
#include "VCSettings.hpp"
#include <QKeySequence>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <QGuiApplication>
#include <QStyleHints>
#include <QDesktopServices>
#include <QUrl>
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
#include <QLabel>
#include <QSizePolicy>
#include <QProcess>
#include <QTemporaryDir>
#include <QToolBar>
#include <QFileInfo>
#include <QTimer>
#include <QSize>
#include <QVector>
#include <QLoggingCategory>
#include <QDebug>
#include <QScrollArea>
#include <QSignalBlocker>
#include <nlohmann/json.hpp>
#include <QGraphicsSimpleTextItem>
#include <QPointer>
#include <QPen>
#include <QFont>
#include <QPainter>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cmath>
#include <limits>
#include <optional>
#include <cctype>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <vector>
#include <initializer_list>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QStringList>

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"
#include "SettingsDialog.hpp"
#include "CSurfaceCollection.hpp"
#include "CPointCollectionWidget.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "DrawingWidget.hpp"
#include "CommandLineToolRunner.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationGrowth.hpp"
#include "segmentation/SegmentationGrower.hpp"
#include "SurfacePanelController.hpp"
#include "MenuActionController.hpp"

#include "vc/core/util/Logging.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Render.hpp"





Q_LOGGING_CATEGORY(lcSegGrowth, "vc.segmentation.growth");
Q_LOGGING_CATEGORY(lcAxisSlices, "vc.axis_aligned");

using qga = QGuiApplication;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;

// ---- Area recompute helpers (robust) ---------------------------------------
namespace {

// --- Small config knobs (can be lifted to QSettings later) ------------------
static constexpr bool   kDeactivateWhenZero   = true;   // mask 0 => deactivate; flip if workflow differs
static constexpr double kTauDeactivate        = 0.50;   // fraction of deactivating pixels needed to drop a quad
static constexpr bool   kBackfaceCullFolds    = false;   // reduce double-count in folds by culling backfaces
static constexpr double kCullDotEps           = 1e-12;  // tolerance for backface culling
static constexpr int    kNormalDecimateMax    = 128;    // sampling grid for global normal estimation

// --- Utilities ---------------------------------------------------------------
static inline bool isFinite3(const cv::Vec3d& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Triangle area (standard “notorious” cross-product formula)
static inline double tri_area3D(const cv::Vec3d& a,
                                const cv::Vec3d& b,
                                const cv::Vec3d& c)
{
    return 0.5 * cv::norm((b - a).cross(c - a));
}

// Triangle area with simple backface culling vs. a reference normal
static inline double tri_area3D_culled(const cv::Vec3d& a,
                                       const cv::Vec3d& b,
                                       const cv::Vec3d& c,
                                       const cv::Vec3d& refN,
                                       double dot_eps)
{
    const cv::Vec3d n = (b - a).cross(c - a);
    const double dot  = n.dot(refN);
    if (dot <= dot_eps) return 0.0;       // backfacing or near parallel -> culled
    return 0.5 * cv::norm(n);
}

// Choose largest image (by pixel count) among multi-page TIFFs
static int choose_largest_page(const std::vector<cv::Mat>& pages) {
    int bestIdx = -1;
    size_t bestPix = 0;
    for (int i = 0; i < (int)pages.size(); ++i) {
        const size_t pix = (size_t)pages[i].rows * (size_t)pages[i].cols;
        if (pix > bestPix) { bestPix = pix; bestIdx = i; }
    }
    return bestIdx;
}

// Robustly binarize an 8/16/32-bit single-channel mask to {0,1}
//  - fast path if already {0,255} (or {0,1})
//  - else Otsu
static void binarize_mask(const cv::Mat& srcAnyDepth, cv::Mat1b& mask01)
{
    cv::Mat m;
    if (srcAnyDepth.channels() != 1) {
        cv::Mat gray; cv::cvtColor(srcAnyDepth, gray, cv::COLOR_BGR2GRAY);
        m = gray;
    } else {
        m = srcAnyDepth;
    }

    // Convert to 8U (preserving dynamic range)
    if (m.type() != CV_8U) {
        double minv, maxv;
        cv::minMaxLoc(m, &minv, &maxv);
        if (std::abs(maxv - minv) < 1e-12) {
            mask01 = cv::Mat1b(m.size(), 0);
            return;
        }
        cv::Mat m8;
        m.convertTo(m8, CV_8U, 255.0 / (maxv - minv), (-minv) * 255.0 / (maxv - minv));
        m = m8;
    }

    // Fast path: already binary?
    int nz = cv::countNonZero(m);
    if (nz == 0) { mask01 = cv::Mat1b(m.size(), 0); return; }
    if (nz == m.rows * m.cols) { mask01 = cv::Mat1b(m.size(), 1); return; }

    // Check if unique values are (0,255) or (0,1)
    // (cheap test using bitwise ops)
    cv::Mat1b tmp;
    cv::threshold(m, tmp, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(m != tmp) == 0) {
        // values are {0, something}; normalize to {0,1}
        mask01 = (tmp > 0) / 255;
        return;
    }

    // Otsu threshold to {0,1}
    cv::Mat1b otsu;
    cv::threshold(m, otsu, 0, 1, cv::THRESH_BINARY | cv::THRESH_OTSU);
    mask01 = otsu;
}

// Load single-channel TIFF -> CV_32F
static bool load_tif_as_float(const std::filesystem::path& file, cv::Mat1f& out)
{
    cv::Mat raw = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty() || raw.channels() != 1) return false;

    switch (raw.type()) {
        case CV_32FC1: out = raw; return true;
        case CV_64FC1: raw.convertTo(out, CV_32F); return true;
        default:       raw.convertTo(out, CV_32F); return true;
    }
}

// 64-bit (double) integral image for 0/1 maps.
// ii has size (H+1, W+1), type CV_64F
static inline double sumRect01d(const cv::Mat1d& ii, int x0, int y0, int x1, int y1)
{
    // rectangle is [x0,x1) × [y0,y1)
    return ii(y1, x1) - ii(y0, x1) - ii(y1, x0) + ii(y0, x0);
}

// Estimate a global reference normal from sparse samples of the grid
static cv::Vec3d estimate_global_normal(const cv::Mat1f& X,
                                        const cv::Mat1f& Y,
                                        const cv::Mat1f& Z)
{
    const int H = X.rows, W = X.cols;
    const int sy = std::max(1, H / kNormalDecimateMax);
    const int sx = std::max(1, W / kNormalDecimateMax);

    cv::Vec3d acc(0,0,0);
    for (int y = 0; y + sy < H; y += sy) {
        for (int x = 0; x + sx < W; x += sx) {
            const cv::Vec3d A(X(y, x),         Y(y, x),         Z(y, x));
            const cv::Vec3d B(X(y, x+sx),      Y(y, x+sx),      Z(y, x+sx));
            const cv::Vec3d C(X(y+sy, x),      Y(y+sy, x),      Z(y+sy, x));
            if (!isFinite3(A) || !isFinite3(B) || !isFinite3(C)) continue;
            acc += (B - A).cross(C - A);
        }
    }
    const double nrm = cv::norm(acc);
    if (nrm < 1e-20) return cv::Vec3d(0,0,1); // fallback (rare)
    return acc / nrm;
}

// Core: area from kept quads using original X/Y/Z grids, fractional mask rule, 64-bit integral,
// and optional backface culling against a global normal to reduce fold double-counting.
static double area_from_mesh_and_mask(const cv::Mat1f& X,
                                      const cv::Mat1f& Y,
                                      const cv::Mat1f& Z,
                                      const cv::Mat1b& mask01)
{
    const int Hq = X.rows, Wq = X.cols;
    if (Hq < 2 || Wq < 2) return 0.0;

    const int Hm = mask01.rows, Wm = mask01.cols;
    if (Hm <= 0 || Wm <= 0) return 0.0;

    // Build "deactivation" map: 1 when a pixel should deactivate, 0 otherwise
    cv::Mat1b deact;
    if (kDeactivateWhenZero) deact = (mask01 == 0);
    else                     deact = (mask01 != 0);

    // 64-bit integral image (double) -> no overflow for huge images
    cv::Mat1d ii; cv::integral(deact, ii, CV_64F);

    // Linear mapping from quad cells to mask pixels
    const double sx = static_cast<double>(Wm) / static_cast<double>(Wq - 1);
    const double sy = static_cast<double>(Hm) / static_cast<double>(Hq - 1);

    // Optional global normal for backface culling
    const cv::Vec3d refN = kBackfaceCullFolds ? estimate_global_normal(X, Y, Z) : cv::Vec3d(0,0,0);

    double total = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:total) schedule(static)
    #endif
    for (int qy = 0; qy < Hq - 1; ++qy) {
        for (int qx = 0; qx < Wq - 1; ++qx) {
            // Map UV cell [qx,qx+1)×[qy,qy+1) → mask rect [x0,x1)×[y0,y1)
            int x0 = (int)std::floor(qx * sx);
            int y0 = (int)std::floor(qy * sy);
            int x1 = (int)std::ceil ((qx + 1) * sx);  // A3 fix: ceil end
            int y1 = (int)std::ceil ((qy + 1) * sy);

            // Clamp and ensure ≥1 pixel extent
            x0 = std::clamp(x0, 0, Wm - 1);
            y0 = std::clamp(y0, 0, Hm - 1);
            x1 = std::clamp(x1, x0 + 1, Wm);
            y1 = std::clamp(y1, y0 + 1, Hm);

            const int rectPix = (x1 - x0) * (y1 - y0);
            if (rectPix <= 0) continue;

            const double deactCount = sumRect01d(ii, x0, y0, x1, y1);
            const double fracDeact  = deactCount / (double)rectPix;

            // Fractional rule (Brittle ANY-pixel fixed) -> robust fraction rule
            if (fracDeact >= kTauDeactivate) continue;  // drop quad

            // 3D corners
            const cv::Vec3d A(X(qy,   qx),   Y(qy,   qx),   Z(qy,   qx));
            const cv::Vec3d B(X(qy,   qx+1), Y(qy,   qx+1), Z(qy,   qx+1));
            const cv::Vec3d C(X(qy+1, qx),   Y(qy+1, qx),   Z(qy+1, qx));
            const cv::Vec3d D(X(qy+1, qx+1), Y(qy+1, qx+1), Z(qy+1, qx+1));
            if (!isFinite3(A) || !isFinite3(B) || !isFinite3(C) || !isFinite3(D))
                continue;

            if (kBackfaceCullFolds) {
                // Count only front-facing triangles vs. global refN (C4 mitigation)
                total += tri_area3D_culled(A, B, D, refN, kCullDotEps);
                total += tri_area3D_culled(A, D, C, refN, kCullDotEps);
            } else {
                // No culling: fixed diagonal (deterministic) is fine for area
                total += tri_area3D(A, B, D) + tri_area3D(A, D, C);
            }
        }
    }

    return total;
}

} // namespace

namespace
{
constexpr float kAxisRotationDegreesPerScenePixel = 0.25f;
constexpr float kEpsilon = 1e-6f;
constexpr float kDegToRad = static_cast<float>(CV_PI / 180.0);
constexpr int kAxisAlignedRotationApplyDelayMs = 25;

int axisAlignedRotationCacheKey(float degrees)
{
    int key = static_cast<int>(std::lround(degrees));
    key %= 360;
    if (key < 0) {
        key += 360;
    }
    return key;
}

cv::Vec3f rotateAroundZ(const cv::Vec3f& v, float radians)
{
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return {
        v[0] * c - v[1] * s,
        v[0] * s + v[1] * c,
        v[2]
    };
}

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

// Constructor
CWindow::CWindow() :
    fVpkg(nullptr),
    _cmdRunner(nullptr),
    _seedingWidget(nullptr),
    _drawingWidget(nullptr),
    _point_collection_widget(nullptr),
    _inotifyFd(-1),
    _inotifyNotifier(nullptr)
{
    // Initialize periodic timer for inotify events
    _inotifyProcessTimer = new QTimer(this);
    connect(_inotifyProcessTimer, &QTimer::timeout, this, &CWindow::processPendingInotifyEvents);

    // Initialize timer for debounced window state saving (500ms delay)
    _windowStateSaveTimer = new QTimer(this);
    _windowStateSaveTimer->setSingleShot(true);
    _windowStateSaveTimer->setInterval(500);
    connect(_windowStateSaveTimer, &QTimer::timeout, this, &CWindow::saveWindowState);

    _point_collection = new VCCollection(this);
    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _mirrorCursorToSegmentation = settings.value(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION,
                                                  vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION_DEFAULT).toBool();
    setWindowIcon(QPixmap(":/images/logo.png"));
    ui.setupUi(this);
    // setAttribute(Qt::WA_DeleteOnClose);

    chunk_cache = new ChunkCache<uint8_t>(CHUNK_CACHE_SIZE_GB*1024ULL*1024ULL*1024ULL);
    std::cout << "chunk cache size is " << CHUNK_CACHE_SIZE_GB << " gigabytes " << std::endl;

    _surf_col = new CSurfaceCollection();

    //_surf_col->setSurface("manual plane", new PlaneSurface({2000,2000,2000},{1,1,1}));
    _surf_col->setSurface("xy plane", std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,0,1}));
    _surf_col->setSurface("xz plane", std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,1,0}));
    _surf_col->setSurface("yz plane", std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{1,0,0}));

    connect(_surf_col, &CSurfaceCollection::sendPOIChanged, this, &CWindow::onFocusPOIChanged);
    connect(_surf_col, &CSurfaceCollection::sendSurfaceWillBeDeleted, this, &CWindow::onSurfaceWillBeDeleted);

    _viewerManager = std::make_unique<ViewerManager>(_surf_col, _point_collection, chunk_cache, this);
    _viewerManager->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);
    connect(_viewerManager.get(), &ViewerManager::viewerCreated, this, [this](CVolumeViewer* viewer) {
        configureViewerConnections(viewer);
    });

    // Slice step size label in status bar
    _sliceStepLabel = new QLabel(this);
    _sliceStepLabel->setContentsMargins(4, 0, 4, 0);
    int initialStepSize = _viewerManager->sliceStepSize();
    _sliceStepLabel->setText(tr("Step: %1").arg(initialStepSize));
    _sliceStepLabel->setToolTip(tr("Slice step size: use Shift+G / Shift+H to adjust"));
    statusBar()->addPermanentWidget(_sliceStepLabel);

    _pointsOverlay = std::make_unique<PointsOverlayController>(_point_collection, this);
    _viewerManager->setPointsOverlay(_pointsOverlay.get());

    _pathsOverlay = std::make_unique<PathsOverlayController>(this);
    _viewerManager->setPathsOverlay(_pathsOverlay.get());

    _bboxOverlay = std::make_unique<BBoxOverlayController>(this);
    _viewerManager->setBBoxOverlay(_bboxOverlay.get());

    _vectorOverlay = std::make_unique<VectorOverlayController>(_surf_col, this);
    _viewerManager->setVectorOverlay(_vectorOverlay.get());

    _planeSlicingOverlay = std::make_unique<PlaneSlicingOverlayController>(_surf_col, this);
    _planeSlicingOverlay->bindToViewerManager(_viewerManager.get());
    _planeSlicingOverlay->setRotationSetter([this](const std::string& planeName, float degrees) {
        setAxisAlignedRotationDegrees(planeName, degrees);
        scheduleAxisAlignedOrientationUpdate();
    });
    _planeSlicingOverlay->setRotationFinishedCallback([this]() {
        flushAxisAlignedOrientationUpdate();
    });
    _planeSlicingOverlay->setAxisAlignedEnabled(_useAxisAlignedSlices);

    _axisAlignedRotationTimer = new QTimer(this);
    _axisAlignedRotationTimer->setSingleShot(true);
    _axisAlignedRotationTimer->setInterval(kAxisAlignedRotationApplyDelayMs);
    connect(_axisAlignedRotationTimer, &QTimer::timeout,
            this, &CWindow::processAxisAlignedOrientationUpdate);

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
    const QSettings geometry(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QByteArray savedGeometry = geometry.value(vc3d::settings::window::GEOMETRY).toByteArray();
    if (!savedGeometry.isEmpty()) {
        restoreGeometry(savedGeometry);
    }
    const QByteArray savedState = geometry.value(vc3d::settings::window::STATE).toByteArray();
    if (!savedState.isEmpty()) {
        restoreState(savedState);
    } else {
        // No saved state - set sensible default sizes for dock widgets
        // The Volume Package dock (left side) should have a reasonable width and height
        resizeDocks({ui.dockWidgetVolumes}, {300}, Qt::Horizontal);
        resizeDocks({ui.dockWidgetVolumes}, {400}, Qt::Vertical);
    }

    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetDrawing,
                               ui.dockWidgetComposite,
                               ui.dockWidgetVolumes,
                               ui.dockWidgetView,
                               ui.dockWidgetOverlay,
                               ui.dockWidgetRenderSettings  }) {
        ensureDockWidgetFeatures(dock);
        // Connect dock widget signals to trigger state saving
        connect(dock, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
        connect(dock, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);
    }
    ensureDockWidgetFeatures(_point_collection_widget);
    connect(_point_collection_widget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
    connect(_point_collection_widget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }

    // If enabled, auto open the last used volpkg
    if (settings.value(vc3d::settings::volpkg::AUTO_OPEN, vc3d::settings::volpkg::AUTO_OPEN_DEFAULT).toInt() != 0) {

        QStringList files = settings.value(vc3d::settings::volpkg::RECENT).toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            if (_menuController) {
                _menuController->openVolpkgAt(files[0]);
            }
        }
    }

    // Create application-wide keyboard shortcuts
    fDrawingModeShortcut = new QShortcut(QKeySequence("Ctrl+Shift+D"), this);
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
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool current = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        bool next = !current;
        settings.setValue(viewer::SHOW_DIRECTION_HINTS, next ? "1" : "0");
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
    connect(_surfacePanel.get(), &SurfacePanelController::moveToPathsRequested, this, &CWindow::onMoveSegmentToPaths);
}

// Destructor
CWindow::~CWindow()
{

    if (_inotifyProcessTimer) {
        _inotifyProcessTimer->stop();
        delete _inotifyProcessTimer;
    }

    stopWatchingWithInotify();
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

    const std::string& surfName = viewer->surfName();
    if ((surfName == "seg xz" || surfName == "seg yz") && !viewer->property("vc_axisaligned_bound").toBool()) {
        if (viewer->fGraphicsView) {
            viewer->fGraphicsView->setMiddleButtonPanEnabled(!_useAxisAlignedSlices);
        }

        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                this, [this, viewer](cv::Vec3f volLoc, cv::Vec3f /*normal*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    onAxisAlignedSliceMousePress(viewer, volLoc, button, modifiers);
                });

        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                this, [this, viewer](cv::Vec3f volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                    onAxisAlignedSliceMouseMove(viewer, volLoc, buttons, modifiers);
                });

        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                this, [this, viewer](cv::Vec3f /*volLoc*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    onAxisAlignedSliceMouseRelease(viewer, button, modifiers);
                });

        viewer->setProperty("vc_axisaligned_bound", true);
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
    _surf_weak.reset();
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
}

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    const bool hadVolume = static_cast<bool>(currentVolume);
    auto previousVolume = currentVolume;
    POI* existingFocusPoi = _surf_col ? _surf_col->poi("focus") : nullptr;
    currentVolume = newvol;

    if (previousVolume != newvol) {
        _focusHistory.clear();
        _focusHistoryIndex = -1;
        _navigatingFocusHistory = false;
    }

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

    if (currentVolume && _surf_col) {
        auto [w, h, d] = currentVolume->shape();

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
    applySlicePlaneOrientation(_surf_col ? _surf_col->surface("segmentation").get() : nullptr);
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

void CWindow::recordFocusHistory(const POI& poi)
{
    if (_navigatingFocusHistory) {
        return;
    }

    FocusHistoryEntry entry;
    entry.position = poi.p;
    entry.normal = poi.n;
    entry.surfaceId = poi.surfaceId;

    if (_focusHistoryIndex >= 0 &&
        _focusHistoryIndex < static_cast<int>(_focusHistory.size())) {
        const auto& current = _focusHistory[_focusHistoryIndex];
        const float positionDelta = cv::norm(current.position - entry.position);
        const float normalDelta = cv::norm(current.normal - entry.normal);
        if (positionDelta < 1e-4f && normalDelta < 1e-4f && current.surfaceId == entry.surfaceId) {
            return;
        }
    }

    if (_focusHistoryIndex >= 0 &&
        _focusHistoryIndex + 1 < static_cast<int>(_focusHistory.size())) {
        _focusHistory.erase(_focusHistory.begin() + _focusHistoryIndex + 1,
                            _focusHistory.end());
    }

    _focusHistory.push_back(entry);

    if (_focusHistory.size() > 10) {
        _focusHistory.pop_front();
        if (_focusHistoryIndex > 0) {
            --_focusHistoryIndex;
        }
    }

    _focusHistoryIndex = static_cast<int>(_focusHistory.size()) - 1;
}

bool CWindow::stepFocusHistory(int direction)
{
    if (_focusHistory.empty() || direction == 0 || _focusHistoryIndex < 0) {
        return false;
    }

    const int lastIndex = static_cast<int>(_focusHistory.size()) - 1;
    int targetIndex = _focusHistoryIndex + direction;
    targetIndex = std::max(0, std::min(targetIndex, lastIndex));

    if (targetIndex == _focusHistoryIndex) {
        return false;
    }

    _focusHistoryIndex = targetIndex;
    _navigatingFocusHistory = true;
    const auto& entry = _focusHistory[_focusHistoryIndex];
    centerFocusAt(entry.position, entry.normal, entry.surfaceId, false);
    _navigatingFocusHistory = false;
    return true;
}

bool CWindow::centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId, bool addToHistory)
{
    if (!_surf_col) {
        return false;
    }

    POI* focus = _surf_col->poi("focus");
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

    _surf_col->setPOI("focus", focus);

    if (addToHistory) {
        recordFocusHistory(*focus);
    }

    // Get surface for orientation - look up by ID
    Surface* orientationSource = _surf_col->surfaceRaw(focus->surfaceId);
    if (!orientationSource) {
        orientationSource = _surf_col->surfaceRaw("segmentation");
    }
    applySlicePlaneOrientation(orientationSource);

    return true;
}

bool CWindow::centerFocusOnCursor()
{
    if (!_surf_col) {
        return false;
    }

    POI* cursor = _surf_col->poi("cursor");
    if (!cursor) {
        return false;
    }

    return centerFocusAt(cursor->p, cursor->n, cursor->surfaceId, true);
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
        [this]() { return segmentationViewer(); },
        std::function<void()>{},
        this);
    if (_segmentationGrower) {
        _segmentationGrower->setSurfacePanel(_surfacePanel.get());
    }
    connect(_surfacePanel.get(), &SurfacePanelController::surfacesLoaded, this, [this]() {
        emit sendSurfacesLoaded();
        // Update surface overlay dropdown when surfaces are loaded
        updateSurfaceOverlayDropdown();
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
                auto surf = fVpkg->getSurface(segmentId.toStdString());
                if (!surf) {
                    return;
                }
                const QString path = QString::fromStdString(surf->path.string());
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
    connect(_surfacePanel.get(), &SurfacePanelController::neighborCopyRequested,
            this, [this](const QString& segmentId, bool copyOut) {
                onNeighborCopyRequested(segmentId, copyOut);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::convertToObjRequested,
            this, [this](const QString& segmentId) {
                onConvertToObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::cropBoundsRequested,
            this, [this](const QString& segmentId) {
                onCropSurfaceToValidRegion(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::alphaCompRefineRequested,
            this, [this](const QString& segmentId) {
                onAlphaCompRefine(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::slimFlattenRequested,
            this, [this](const QString& segmentId) {
                onSlimFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::abfFlattenRequested,
            this, [this](const QString& segmentId) {
                onABFFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::awsUploadRequested,
            this, [this](const QString& segmentId) {
                onAWSUpload(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::exportTifxyzChunksRequested,
        this, [this](const QString& segmentId) {
            onExportWidthChunks(segmentId.toStdString());
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
    _segmentationEdit->setViewerManager(_viewerManager.get());
    _segmentationOverlay = std::make_unique<SegmentationOverlayController>(_surf_col, this);
    _segmentationOverlay->setEditManager(_segmentationEdit.get());
    _segmentationOverlay->setViewerManager(_viewerManager.get());

    _segmentationModule = std::make_unique<SegmentationModule>(
        _segmentationWidget,
        _segmentationEdit.get(),
        _segmentationOverlay.get(),
        _viewerManager.get(),
        _surf_col,
        _point_collection,
        _segmentationWidget->isEditingEnabled(),
        this);

    if (_segmentationModule && _planeSlicingOverlay) {
        QPointer<PlaneSlicingOverlayController> overlayPtr(_planeSlicingOverlay.get());
        _segmentationModule->setRotationHandleHitTester(
            [overlayPtr](CVolumeViewer* viewer, const cv::Vec3f& worldPos) {
                if (!overlayPtr) {
                    return false;
                }
                return overlayPtr->isVolumePointNearRotationHandle(viewer, worldPos, 1.5);
            });
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationOverlay(_segmentationOverlay.get());
    }

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
                applySlicePlaneOrientation(base);
            });
    connect(_segmentationModule.get(), &SegmentationModule::growSurfaceRequested,
            this, &CWindow::onGrowSegmentationSurface);
    connect(_segmentationModule.get(), &SegmentationModule::approvalMaskSaved,
            this, &CWindow::markSegmentRecentlyEdited);

    SegmentationGrower::Context growerContext{
        _segmentationModule.get(),
        _segmentationWidget,
        _surf_col,
        _viewerManager.get(),
        chunk_cache
    };
    SegmentationGrower::UiCallbacks growerCallbacks{
        [this](const QString& text, int timeout) {
            if (statusBar()) {
                statusBar()->showMessage(text, timeout);
            }
        },
        [this](QuadSurface* surface) {
            applySlicePlaneOrientation(surface);
        }
    };
    _segmentationGrower = std::make_unique<SegmentationGrower>(growerContext, growerCallbacks, this);

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

    // Tab the docks - keep Segmentation, Seeding, Point Collections, and Drawing together
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDistanceTransform);
    tabifyDockWidget(ui.dockWidgetSegmentation, _point_collection_widget);
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDrawing);

    // Make Drawing dock the active tab by default
    ui.dockWidgetDrawing->raise();

    // Keep the view-related docks on the left and grouped together as tabs
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetView);
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetOverlay);
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetRenderSettings);
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetComposite);

    auto ensureTabified = [this](QDockWidget* primary, QDockWidget* candidate) {
        const auto currentTabs = tabifiedDockWidgets(primary);
        const bool alreadyTabified = std::find(currentTabs.cbegin(), currentTabs.cend(), candidate) != currentTabs.cend();
        if (!alreadyTabified) {
            tabifyDockWidget(primary, candidate);
        }
    };

    ensureTabified(ui.dockWidgetView, ui.dockWidgetOverlay);
    ensureTabified(ui.dockWidgetView, ui.dockWidgetRenderSettings);
    ensureTabified(ui.dockWidgetView, ui.dockWidgetComposite);

    const auto tabOrder = tabifiedDockWidgets(ui.dockWidgetView);
    for (QDockWidget* dock : tabOrder) {
        tabifyDockWidget(ui.dockWidgetView, dock);
    }

    ui.dockWidgetView->show();
    ui.dockWidgetView->raise();
    QTimer::singleShot(0, this, [this]() {
        if (ui.dockWidgetView) {
            ui.dockWidgetView->raise();
        }
    });

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
            .opacitySpin = ui.overlayOpacitySpin,
            .thresholdSpin = ui.overlayThresholdSpin,
        };
        _volumeOverlay->setUi(overlayUi);
    }

        // Setup base colormap selector
    {
        const auto& entries = CVolumeViewer::overlayColormapEntries();
        ui.baseColormapSelect->clear();
        ui.baseColormapSelect->addItem(tr("None (Grayscale)"), QString());
        for (const auto& entry : entries) {
            ui.baseColormapSelect->addItem(entry.label, QString::fromStdString(entry.id));
        }
        ui.baseColormapSelect->setCurrentIndex(0);
    }

    connect(ui.baseColormapSelect, qOverload<int>(&QComboBox::currentIndexChanged), [this](int index) {
        if (index < 0 || !_viewerManager) return;
        const QString id = ui.baseColormapSelect->currentData().toString();
        _viewerManager->forEachViewer([&id](CVolumeViewer* viewer) {
            viewer->setBaseColormap(id.toStdString());
        });
    });

    connect(ui.chkStretchValues, &QCheckBox::toggled, [this](bool checked) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([checked](CVolumeViewer* viewer) {
            viewer->setStretchValues(checked);
        });
    });

    // Setup surface overlay controls
    connect(ui.chkSurfaceOverlay, &QCheckBox::toggled, [this](bool checked) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([checked](CVolumeViewer* viewer) {
            viewer->setSurfaceOverlayEnabled(checked);
        });
        ui.surfaceOverlaySelect->setEnabled(checked);
        ui.spinOverlapThreshold->setEnabled(checked);
    });

    connect(ui.surfaceOverlaySelect, qOverload<int>(&QComboBox::currentIndexChanged), [this](int index) {
        if (index < 0 || !_viewerManager) return;
        const QString surfaceName = ui.surfaceOverlaySelect->currentData().toString();
        _viewerManager->forEachViewer([&surfaceName](CVolumeViewer* viewer) {
            viewer->setSurfaceOverlay(surfaceName.toStdString());
        });
    });

    connect(ui.spinOverlapThreshold, qOverload<double>(&QDoubleSpinBox::valueChanged), [this](double value) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([value](CVolumeViewer* viewer) {
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
            std::shared_ptr<Volume> newVolume;
            try {
                newVolume = fVpkg->volume(volSelect->currentData().toString().toStdString());
            } catch (const std::out_of_range& e) {
                QMessageBox::warning(this, "Error", "Could not load volume.");
                return;
            }
            setVolume(newVolume);
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

    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        bool showOverlays = settings.value(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS,
                                           vc3d::settings::viewer::SHOW_AXIS_OVERLAYS_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisOverlays);
        chkAxisOverlays->setChecked(showOverlays);
        connect(chkAxisOverlays, &QCheckBox::toggled, this, &CWindow::onAxisOverlayVisibilityToggled);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        int storedOpacity = settings.value(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY,
                                           spinAxisOverlayOpacity->value()).toInt();
        storedOpacity = std::clamp(storedOpacity, spinAxisOverlayOpacity->minimum(), spinAxisOverlayOpacity->maximum());
        QSignalBlocker blocker(spinAxisOverlayOpacity);
        spinAxisOverlayOpacity->setValue(storedOpacity);
        connect(spinAxisOverlayOpacity, qOverload<int>(&QSpinBox::valueChanged), this, &CWindow::onAxisOverlayOpacityChanged);
    }

    if (auto* spinSliceStep = ui.spinSliceStepSize) {
        int savedStep = settings.value(vc3d::settings::viewer::SLICE_STEP_SIZE,
                                       vc3d::settings::viewer::SLICE_STEP_SIZE_DEFAULT).toInt();
        savedStep = std::clamp(savedStep, spinSliceStep->minimum(), spinSliceStep->maximum());
        QSignalBlocker blocker(spinSliceStep);
        spinSliceStep->setValue(savedStep);
        if (_viewerManager) {
            _viewerManager->setSliceStepSize(savedStep);
        }
        connect(spinSliceStep, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
            if (_viewerManager) {
                _viewerManager->setSliceStepSize(value);
            }
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(vc3d::settings::viewer::SLICE_STEP_SIZE, value);
            if (_sliceStepLabel) {
                _sliceStepLabel->setText(tr("Step: %1").arg(value));
            }
        });
    }

    if (auto* btnResetRot = ui.btnResetAxisRotations) {
        connect(btnResetRot, &QPushButton::clicked, this, &CWindow::onResetAxisAlignedRotations);
    }

    // Zoom buttons
    btnZoomIn = ui.btnZoomIn;
    btnZoomOut = ui.btnZoomOut;

    connect(btnZoomIn, &QPushButton::clicked, this, &CWindow::onZoomIn);
    connect(btnZoomOut, &QPushButton::clicked, this, &CWindow::onZoomOut);

    if (auto* volumeContainer = ui.volumeWindowContainer) {
        auto* layout = new QHBoxLayout(volumeContainer);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(6);

        _volumeWindowWidget = new WindowRangeWidget(volumeContainer);
        _volumeWindowWidget->setRange(0, 255);
        _volumeWindowWidget->setMinimumSeparation(1);
        _volumeWindowWidget->setControlsEnabled(false);
        layout->addWidget(_volumeWindowWidget);

        connect(_volumeWindowWidget, &WindowRangeWidget::windowValuesChanged,
                this, [this](int low, int high) {
                    if (_viewerManager) {
                        _viewerManager->setVolumeWindow(static_cast<float>(low),
                                                        static_cast<float>(high));
                    }
                });

        if (_viewerManager) {
            connect(_viewerManager.get(), &ViewerManager::volumeWindowChanged,
                    this, [this](float low, float high) {
                        if (!_volumeWindowWidget) {
                            return;
                        }
                        const int lowInt = static_cast<int>(std::lround(low));
                        const int highInt = static_cast<int>(std::lround(high));
                        _volumeWindowWidget->setWindowValues(lowInt, highInt);
                    });

            _volumeWindowWidget->setWindowValues(
                static_cast<int>(std::lround(_viewerManager->volumeWindowLow())),
                static_cast<int>(std::lround(_viewerManager->volumeWindowHigh())));
        }

        const bool viewEnabled = !ui.grpVolManager || ui.grpVolManager->isEnabled();
        _volumeWindowWidget->setControlsEnabled(viewEnabled);
    }

    if (auto* container = ui.overlayWindowContainer) {
        auto* layout = new QHBoxLayout(container);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(6);

        _overlayWindowWidget = new WindowRangeWidget(container);
        _overlayWindowWidget->setRange(0, 255);
        _overlayWindowWidget->setMinimumSeparation(1);
        _overlayWindowWidget->setControlsEnabled(false);
        layout->addWidget(_overlayWindowWidget);

        connect(_overlayWindowWidget, &WindowRangeWidget::windowValuesChanged,
                this, [this](int low, int high) {
                    if (_viewerManager) {
                        _viewerManager->setOverlayWindow(static_cast<float>(low),
                                                         static_cast<float>(high));
                    }
                });

        if (_viewerManager) {
            connect(_viewerManager.get(), &ViewerManager::overlayWindowChanged,
                    this, [this](float low, float high) {
                        if (!_overlayWindowWidget) {
                            return;
                        }
                        const int lowInt = static_cast<int>(std::lround(low));
                        const int highInt = static_cast<int>(std::lround(high));
                        _overlayWindowWidget->setWindowValues(lowInt, highInt);
                    });

            _overlayWindowWidget->setWindowValues(
                static_cast<int>(std::lround(_viewerManager->overlayWindowLow())),
                static_cast<int>(std::lround(_viewerManager->overlayWindowHigh())));
        }
    }

    if (_viewerManager && _overlayWindowWidget) {
        connect(_viewerManager.get(), &ViewerManager::overlayVolumeAvailabilityChanged,
                this, [this](bool hasOverlay) {
                    if (!_overlayWindowWidget) {
                        return;
                    }
                    const bool viewEnabled = !ui.grpVolManager || ui.grpVolManager->isEnabled();
                    _overlayWindowWidget->setControlsEnabled(hasOverlay && viewEnabled);
                });
    }

    if (_overlayWindowWidget) {
        const bool hasOverlay = _volumeOverlay && _volumeOverlay->hasOverlaySelection();
        const bool viewEnabled = !ui.grpVolManager || ui.grpVolManager->isEnabled();
        _overlayWindowWidget->setControlsEnabled(hasOverlay && viewEnabled);
    }

    auto* spinIntersectionOpacity = ui.spinIntersectionOpacity;
    const int savedIntersectionOpacity = settings.value(vc3d::settings::viewer::INTERSECTION_OPACITY,
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

    if (auto* spinIntersectionThickness = ui.doubleSpinIntersectionThickness) {
        const double savedThickness = settings.value(vc3d::settings::viewer::INTERSECTION_THICKNESS,
                                                     spinIntersectionThickness->value()).toDouble();
        const double boundedThickness = std::clamp(savedThickness,
                                                   static_cast<double>(spinIntersectionThickness->minimum()),
                                                   static_cast<double>(spinIntersectionThickness->maximum()));
        spinIntersectionThickness->setValue(boundedThickness);
        connect(spinIntersectionThickness,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                [this](double value) {
                    if (!_viewerManager) {
                        return;
                    }
                    _viewerManager->setIntersectionThickness(static_cast<float>(value));
                });
        if (_viewerManager) {
            _viewerManager->setIntersectionThickness(static_cast<float>(spinIntersectionThickness->value()));
        }
    }

    auto* comboIntersectionSampling = ui.comboIntersectionSampling;
    if (comboIntersectionSampling) {
        struct SamplingOption {
            const char* label;
            int stride;
        };
        const SamplingOption options[] = {
            {"Full (1x)", 1},
            {"2x", 2},
            {"4x", 4},
            {"8x", 8},
        };
        comboIntersectionSampling->clear();
        for (const auto& opt : options) {
            comboIntersectionSampling->addItem(tr(opt.label), opt.stride);
        }

        const int savedStride = settings.value(vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE,
                                              vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE_DEFAULT).toInt();
        int selectedIndex = comboIntersectionSampling->findData(savedStride);
        if (selectedIndex < 0) {
            selectedIndex = comboIntersectionSampling->findData(1);
        }
        if (selectedIndex >= 0) {
            comboIntersectionSampling->setCurrentIndex(selectedIndex);
        }

        connect(comboIntersectionSampling,
                QOverload<int>::of(&QComboBox::currentIndexChanged),
                this,
                [this, comboIntersectionSampling](int) {
                    if (!_viewerManager) {
                        return;
                    }
                    const int stride = std::max(1, comboIntersectionSampling->currentData().toInt());
                    _viewerManager->setSurfacePatchSamplingStride(stride);
                });

        // Update combobox when stride changes programmatically (e.g., tiered defaults)
        if (_viewerManager) {
            connect(_viewerManager.get(),
                    &ViewerManager::samplingStrideChanged,
                    this,
                    [comboIntersectionSampling](int stride) {
                        const int index = comboIntersectionSampling->findData(stride);
                        if (index >= 0 && index != comboIntersectionSampling->currentIndex()) {
                            QSignalBlocker blocker(comboIntersectionSampling);
                            comboIntersectionSampling->setCurrentIndex(index);
                        }
                    });
        }
    }

    chkAxisAlignedSlices = ui.chkAxisAlignedSlices;
    if (chkAxisAlignedSlices) {
        bool useAxisAligned = settings.value(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES,
                                             vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisAlignedSlices);
        chkAxisAlignedSlices->setChecked(useAxisAligned);
        connect(chkAxisAlignedSlices, &QCheckBox::toggled, this, &CWindow::onAxisAlignedSlicesToggled);
    }

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

    if (chkAxisAlignedSlices) {
        onAxisAlignedSlicesToggled(chkAxisAlignedSlices->isChecked());
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        onAxisOverlayOpacityChanged(spinAxisOverlayOpacity->value());
    }
    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        onAxisOverlayVisibilityToggled(chkAxisOverlays->isChecked());
    }

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
    bool resetViewOnSurfaceChange = settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                                                   vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
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

    if (event->key() == Qt::Key_R && event->modifiers() == Qt::NoModifier) {
        if (centerFocusOnCursor()) {
            event->accept();
            return;
        }
    }

    if (event->key() == Qt::Key_F) {
        if (event->modifiers() == Qt::NoModifier) {
            stepFocusHistory(-1);
            event->accept();
            return;
        } else if (event->modifiers() == Qt::ControlModifier) {
            stepFocusHistory(1);
            event->accept();
            return;
        }
    }

    // Shift+G decreases slice step size, Shift+H increases it
    if (event->modifiers() == Qt::ShiftModifier && _viewerManager) {
        if (event->key() == Qt::Key_G) {
            int currentStep = _viewerManager->sliceStepSize();
            int newStep = std::max(1, currentStep - 1);
            _viewerManager->setSliceStepSize(newStep);
            onSliceStepSizeChanged(newStep);
            event->accept();
            return;
        } else if (event->key() == Qt::Key_H) {
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
    settings.sync();
}

// Asks User to Save Data Prior to VC.app Exit
void CWindow::closeEvent(QCloseEvent* event)
{
    saveWindowState();
    std::quick_exit(0);
}

void CWindow::setWidgetsEnabled(bool state)
{
    ui.grpVolManager->setEnabled(state);
    if (_volumeWindowWidget) {
        _volumeWindowWidget->setControlsEnabled(state);
    }
    if (_overlayWindowWidget) {
        const bool hasOverlay = _volumeOverlay && _volumeOverlay->hasOverlaySelection();
        _overlayWindowWidget->setControlsEnabled(state && hasOverlay);
    }
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
            "Volume package failed to load. Package might be corrupt. Check the console log for a detailed error message.");
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

    // Update View dock widget spinbox
    if (auto* spinSliceStep = ui.spinSliceStepSize) {
        QSignalBlocker blocker(spinSliceStep);
        spinSliceStep->setValue(newSize);
    }
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
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        aVpkgPath = QFileDialog::getExistingDirectory(
            this, tr("Open Directory"), settings.value(vc3d::settings::volpkg::DEFAULT_PATH).toString(),
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
        // Reset stride user override so tiered defaults apply to new volume
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
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

    startWatchingWithInotify();
}

void CWindow::CloseVolume(void)
{
    stopWatchingWithInotify();
    if (_inotifyProcessTimer) {
        _inotifyProcessTimer->stop();
    }

    // Clear any pending inotify events
    _pendingInotifyEvents.clear();
    _pendingSegmentUpdates.clear();
    _pendingMoves.clear();

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

    // Clear the volume package
    fVpkg = nullptr;
    currentVolume = nullptr;
    _focusHistory.clear();
    _focusHistoryIndex = -1;
    _navigatingFocusHistory = false;
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
        // Get the surface ID from the surface collection
        std::string surfId;
        if (_surf_col && surf) {
            surfId = _surf_col->findSurfaceId(surf);
        }
        centerFocusAt(vol_loc, normal, surfId, true);
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

    auto planeShared = _surf_col->surface("manual plane");
    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(planeShared.get());

    if (!plane)
        return;

    plane->setNormal(normal);
    _surf_col->setSurface("manual plane", planeShared);
}

void CWindow::onSurfaceActivated(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _surfID;
    _surfID = surfaceId.toStdString();

    // Look up the shared_ptr by ID
    if (fVpkg && !_surfID.empty()) {
        _surf_weak = fVpkg->getSurface(_surfID);
    } else {
        _surf_weak.reset();
    }

    auto surf = _surf_weak.lock();

    if (_surfID != previousSurfId) {
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }

        // Handle approval mask when switching segments
        if (_segmentationModule) {
            _segmentationModule->onActiveSegmentChanged(surf.get());
        }
    }

    if (surf) {
        applySlicePlaneOrientation(surf.get());
    } else {
        applySlicePlaneOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf)
{
    // Called BEFORE surface deletion - clear all references to prevent use-after-free

    // Clear if this is our current active surface
    auto currentSurf = _surf_weak.lock();
    if (currentSurf && currentSurf == surf) {
        _surf_weak.reset();
        _surfID.clear();
    }

    // Focus history uses string IDs now, so no cleanup needed for surface pointers
    // (the ID remains valid for lookup - will just return nullptr if surface is gone)
}

void CWindow::onEditMaskPressed(void)
{
    auto surf = _surf_weak.lock();
    if (!surf)
        return;

    std::filesystem::path path = surf->path/"mask.tif";

    if (!std::filesystem::exists(path)) {
        cv::Mat_<uint8_t> mask;
        cv::Mat_<cv::Vec3f> coords; // Not used after generation

        // Generate the binary mask at raw points resolution
        render_binary_mask(surf.get(), mask, coords, 1.0f);

        // Save just the mask as single layer
        cv::imwrite(path.string(), mask);

        // Update metadata
        (*surf->meta)["date_last_modified"] = get_surface_time_str();
        surf->save_meta();
    }

    QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));
}

void CWindow::onAppendMaskPressed(void)
{
    auto surf = _surf_weak.lock();
    if (!surf || !currentVolume) {
        if (!surf) {
            QMessageBox::warning(this, tr("Error"), tr("No surface selected."));
        } else {
            QMessageBox::warning(this, tr("Error"), tr("No volume loaded."));
        }
        return;
    }

    std::filesystem::path path = surf->path/"mask.tif";

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
                img = segViewer->renderCompositeForSurface(surf, maskSize);
            } else {
                // Original single-layer rendering - use same approach as render_binary_mask
                cv::Size rawSize = surf->rawPointsPtr()->size();
                cv::Vec3f ptr = surf->pointer();
                cv::Vec3f offset(-rawSize.width/2.0f, -rawSize.height/2.0f, 0);

                // Use surface's scale so sx = _scale/_scale = 1.0, sampling 1:1 from raw points
                float surfScale = surf->scale()[0];
                cv::Mat_<cv::Vec3f> coords;
                surf->gen(&coords, nullptr, maskSize, ptr, surfScale, offset);

                std::cout << "[AppendMask non-composite] rawSize: " << rawSize.width << "x" << rawSize.height
                          << ", maskSize: " << maskSize.width << "x" << maskSize.height
                          << ", coords size: " << coords.cols << "x" << coords.rows
                          << ", surface._scale: " << surf->scale()[0] << std::endl;

                // Sample a few coords to verify they're in native voxel space
                if (coords.rows > 4 && coords.cols > 4) {
                    std::cout << "[AppendMask non-composite] coords[0,0]: " << coords(4,4)
                              << ", coords[center]: " << coords(coords.rows/2, coords.cols/2)
                              << ", coords[end]: " << coords(coords.rows-5, coords.cols-5) << std::endl;
                }

                render_image_from_coords(coords, img, ds, chunk_cache);
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            std::cout << "[AppendMask] maskSize: " << maskSize.width << "x" << maskSize.height
                      << ", img size: " << img.cols << "x" << img.rows
                      << ", useComposite: " << useComposite << std::endl;

            // Append the new image layer to existing layers
            existing_layers.push_back(img);

            // Save all layers
            imwritemulti(path.string(), existing_layers);

            QString message = useComposite ?
                tr("Appended composite surface image to existing mask (now %1 layers)").arg(existing_layers.size()) :
                tr("Appended surface image to existing mask (now %1 layers)").arg(existing_layers.size());
            statusBar()->showMessage(message, 3000);

        } else {
            // No existing mask, generate both mask and image at raw points resolution
            cv::Mat_<cv::Vec3f> coords;
            render_binary_mask(surf.get(), mask, coords, 1.0f);
            cv::Size maskSize = mask.size();

            if (useComposite) {
                // Use composite rendering for image
                img = segViewer->renderCompositeForSurface(surf, maskSize);
            } else {
                // Original rendering
                render_surface_image(surf.get(), mask, img, ds, chunk_cache, 1.0f);
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
        (*surf->meta)["date_last_modified"] = get_surface_time_str();
        surf->save_meta();

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
        _surf_weak.reset();
        _surfID.clear();
        treeWidgetSurfaces->clearSelection();

        if (_surfacePanel) {
            _surfacePanel->resetTagUi();
        }

        // Set the new directory in the VolumePkg
        fVpkg->setSegmentationDirectory(newDir);

        // Reset stride user override so tiered defaults apply to new directory
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
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
    auto [w, h, d] = currentVolume->shape();

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
        auto seg_surface = _surf_col->surface("segmentation");
        if (auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get())) {
            auto ptr = quad_surface->pointer();
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            quad_surface->pointTo(ptr, point_opt->p, 4.0, 100, patchIndex);
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

void CWindow::onResetAxisAlignedRotations()
{
    _axisAlignedSegXZRotationDeg = 0.0f;
    _axisAlignedSegYZRotationDeg = 0.0f;
    _axisAlignedSliceDrags.clear();
    applySlicePlaneOrientation();
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }
    statusBar()->showMessage(tr("Axis-aligned rotations reset"), 2000);
}

void CWindow::onAxisOverlayVisibilityToggled(bool enabled)
{
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && _useAxisAlignedSlices);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        spinAxisOverlayOpacity->setEnabled(_useAxisAlignedSlices && enabled);
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

void CWindow::recalcAreaForSegments(const std::vector<std::string>& ids)
{
    if (!fVpkg) return;

    // Linear voxel size (µm/voxel) for cm² conversion
    float voxelsize = 1.0f;
    try {
        if (currentVolume) {
            voxelsize = static_cast<float>(currentVolume->voxelSize());
        }
    } catch (...) { voxelsize = 1.0f; }
    if (!std::isfinite(voxelsize) || voxelsize <= 0.f) voxelsize = 1.0f;

    int okCount = 0, failCount = 0;
    QStringList updatedIds, skippedIds;

    for (const auto& id : ids) {
        auto sm = fVpkg->getSurface(id);
        if (!sm) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (missing surface)";
            continue;
        }
        auto* surf = sm.get(); // QuadSurface* - sm is already shared_ptr<QuadSurface>

        // --- Load mask (robust multi-page handling) ----------------------
        const std::filesystem::path maskPath = sm->path / "mask.tif";
        if (!std::filesystem::exists(maskPath)) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (no mask.tif)";
            continue;
        }

        cv::Mat1b mask01;
        {
            std::vector<cv::Mat> pages;
            if (cv::imreadmulti(maskPath.string(), pages, cv::IMREAD_UNCHANGED) && !pages.empty()) {
                int best = choose_largest_page(pages);
                if (best < 0) { ++failCount; skippedIds << QString::fromStdString(id) + " (mask pages invalid)"; continue; }
                binarize_mask(pages[best], mask01);
            } else {
                // Fallback: single-page read
                cv::Mat m = cv::imread(maskPath.string(), cv::IMREAD_UNCHANGED);
                if (m.empty()) { ++failCount; skippedIds << QString::fromStdString(id) + " (mask read error)"; continue; }
                binarize_mask(m, mask01);
            }
        }
        if (mask01.empty()) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (empty mask)";
            continue;
        }

        // --- Load ORIGINAL quadmesh (no resampling; lower memory) --------
        cv::Mat1f X, Y, Z;
        if (!load_tif_as_float(sm->path / "x.tif", X) ||
            !load_tif_as_float(sm->path / "y.tif", Y) ||
            !load_tif_as_float(sm->path / "z.tif", Z)) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (bad or missing x/y/z.tif)";
            continue;
        }
        if (X.size() != Y.size() || X.size() != Z.size()
            || X.rows < 2 || X.cols < 2) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (xyz size mismatch)";
            continue;
        }

        // --- Area from kept quads --------------
        double area_vx2 = 0.0;
        try {
            area_vx2 = area_from_mesh_and_mask(X, Y, Z, mask01);
        } catch (...) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (area compute error)";
            continue;
        }
        if (!std::isfinite(area_vx2)) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (non-finite area)";
            continue;
        }

        // --- Convert voxel^2 → cm^2 -----------------------------------------
        const double area_cm2 = area_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
        if (!std::isfinite(area_cm2)) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (non-finite cm²)";
            continue;
        }

        // --- Persist & UI update --------------------------------------------
        try {
            if (!surf->meta) surf->meta = std::make_unique<nlohmann::json>();
            (*surf->meta)["area_vx2"] = area_vx2;
            (*surf->meta)["area_cm2"] = area_cm2;
            (*surf->meta)["date_last_modified"] = get_surface_time_str();
            surf->save_meta();
            okCount++;
            updatedIds << QString::fromStdString(id);
        } catch (...) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (meta save failed)";
            continue;
        }

        // Update the Surfaces tree (Area column)
        QTreeWidgetItemIterator it(treeWidgetSurfaces);
        while (*it) {
            if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == id) {
                (*it)->setText(2, QString::number(area_cm2, 'f', 3));
                break;
            }
            ++it;
        }
    }

    if (okCount > 0) {
        statusBar()->showMessage(
            tr("Recalculated area (triangulated kept quads) for %1 segment(s).").arg(okCount), 5000);
    }
    if (failCount > 0) {
        QMessageBox::warning(this, tr("Area Recalculation"),
                             tr("Updated: %1\nSkipped: %2\n\n%3")
                                .arg(okCount)
                                .arg(failCount)
                                .arg(skippedIds.join("\n")));
    }
}

void CWindow::onAxisAlignedSlicesToggled(bool enabled)
{
    _useAxisAlignedSlices = enabled;
    if (enabled) {
        _axisAlignedSegXZRotationDeg = 0.0f;
        _axisAlignedSegYZRotationDeg = 0.0f;
    }
    _axisAlignedSliceDrags.clear();
    qCDebug(lcAxisSlices) << "Axis-aligned slices" << (enabled ? "enabled" : "disabled");
    if (_planeSlicingOverlay) {
        bool overlaysVisible = !ui.chkAxisOverlays || ui.chkAxisOverlays->isChecked();
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && overlaysVisible);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        spinAxisOverlayOpacity->setEnabled(enabled && (!ui.chkAxisOverlays || ui.chkAxisOverlays->isChecked()));
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES, enabled ? "1" : "0");
    updateAxisAlignedSliceInteraction();
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

    std::optional<std::string> recentlyEditedId;
    if (!enabled) {
        if (auto* activeSurface = _segmentationModule->activeBaseSurface()) {
            recentlyEditedId = activeSurface->id;
        }
    }

    // Set flag BEFORE beginEditingSession so the surface change doesn't reset view
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

    if (enabled) {
        auto activeSurfaceShared = std::dynamic_pointer_cast<QuadSurface>(_surf_col->surface("segmentation"));

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
            _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->clearOverlayGroup("segmentation_radius_indicator");
                }
            });
        }
    } else {
        _segmentationModule->endEditingSession();

        if (recentlyEditedId && !recentlyEditedId->empty()) {
            markSegmentRecentlyEdited(*recentlyEditedId);
        }
    }

    const QString message = enabled
        ? tr("Segmentation editing enabled")
        : tr("Segmentation editing disabled");
    statusBar()->showMessage(message, 2000);
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
        _surf_col,
        _viewerManager.get(),
        chunk_cache
    };
    _segmentationGrower->updateContext(context);

    SegmentationGrower::VolumeContext volumeContext{
        fVpkg,
        currentVolume,
        currentVolumeId,
        _segmentationGrowthVolumeId.empty() ? currentVolumeId : _segmentationGrowthVolumeId,
        _normalGridPath
    };

    if (!_segmentationGrower->start(volumeContext, method, direction, steps, inpaintOnly)) {
        return;
    }
}

float CWindow::normalizeDegrees(float degrees)
{
    while (degrees > 180.0f) {
        degrees -= 360.0f;
    }
    while (degrees <= -180.0f) {
        degrees += 360.0f;
    }
    return degrees;
}

float CWindow::currentAxisAlignedRotationDegrees(const std::string& surfaceName) const
{
    if (surfaceName == "seg xz") {
        return _axisAlignedSegXZRotationDeg;
    }
    if (surfaceName == "seg yz") {
        return _axisAlignedSegYZRotationDeg;
    }
    return 0.0f;
}

void CWindow::setAxisAlignedRotationDegrees(const std::string& surfaceName, float degrees)
{
    const float normalized = normalizeDegrees(degrees);
    if (surfaceName == "seg xz") {
        _axisAlignedSegXZRotationDeg = normalized;
    } else if (surfaceName == "seg yz") {
        _axisAlignedSegYZRotationDeg = normalized;
    }
}

void CWindow::scheduleAxisAlignedOrientationUpdate()
{
    if (!_useAxisAlignedSlices) {
        applySlicePlaneOrientation();
        return;
    }
    _axisAlignedOrientationDirty = true;
    if (!_axisAlignedRotationTimer) {
        applySlicePlaneOrientation();
        return;
    }
    if (!_axisAlignedRotationTimer->isActive()) {
        _axisAlignedRotationTimer->start(kAxisAlignedRotationApplyDelayMs);
    }
}

void CWindow::flushAxisAlignedOrientationUpdate()
{
    if (!_axisAlignedOrientationDirty) {
        return;
    }
    cancelAxisAlignedOrientationTimer();
    applySlicePlaneOrientation();
}

void CWindow::processAxisAlignedOrientationUpdate()
{
    if (!_axisAlignedOrientationDirty) {
        return;
    }
    _axisAlignedOrientationDirty = false;
    applySlicePlaneOrientation();
}

void CWindow::cancelAxisAlignedOrientationTimer()
{
    if (_axisAlignedRotationTimer && _axisAlignedRotationTimer->isActive()) {
        _axisAlignedRotationTimer->stop();
    }
    _axisAlignedOrientationDirty = false;
}

void CWindow::updateAxisAlignedSliceInteraction()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachViewer([this](CVolumeViewer* viewer) {
        if (!viewer || !viewer->fGraphicsView) {
            return;
        }
        const std::string& name = viewer->surfName();
        if (name == "seg xz" || name == "seg yz") {
            viewer->fGraphicsView->setMiddleButtonPanEnabled(!_useAxisAlignedSlices);
            qCDebug(lcAxisSlices) << "Middle-button pan set" << QString::fromStdString(name)
                                 << "enabled" << viewer->fGraphicsView->middleButtonPanEnabled();
        }
    });
}

void CWindow::onAxisAlignedSliceMousePress(CVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButton button, Qt::KeyboardModifiers)
{
    if (!_useAxisAlignedSlices || button != Qt::MiddleButton || !viewer) {
        return;
    }

    const std::string surfaceName = viewer->surfName();
    if (surfaceName != "seg xz" && surfaceName != "seg yz") {
        return;
    }

    AxisAlignedSliceDragState& state = _axisAlignedSliceDrags[viewer];
    state.active = true;
    state.startScenePos = viewer->volumePointToScene(volLoc);
    state.startRotationDegrees = currentAxisAlignedRotationDegrees(surfaceName);

}

void CWindow::onAxisAlignedSliceMouseMove(CVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers)
{
    if (!_useAxisAlignedSlices || !viewer || !(buttons & Qt::MiddleButton)) {
        return;
    }

    const std::string surfaceName = viewer->surfName();
    if (surfaceName != "seg xz" && surfaceName != "seg yz") {
        return;
    }

    auto it = _axisAlignedSliceDrags.find(viewer);
    if (it == _axisAlignedSliceDrags.end() || !it->second.active) {
        return;
    }

    AxisAlignedSliceDragState& state = it->second;
    QPointF currentScenePos = viewer->volumePointToScene(volLoc);
    const float dragPixels = static_cast<float>(currentScenePos.y() - state.startScenePos.y());
    const float candidate = normalizeDegrees(state.startRotationDegrees - dragPixels * kAxisRotationDegreesPerScenePixel);
    const float currentRotation = currentAxisAlignedRotationDegrees(surfaceName);

    if (std::abs(candidate - currentRotation) < 0.01f) {
        return;
    }

    setAxisAlignedRotationDegrees(surfaceName, candidate);
    scheduleAxisAlignedOrientationUpdate();

}

void CWindow::onAxisAlignedSliceMouseRelease(CVolumeViewer* viewer, Qt::MouseButton button, Qt::KeyboardModifiers)
{
    if (button != Qt::MiddleButton) {
        return;
    }

    auto it = _axisAlignedSliceDrags.find(viewer);
    if (it != _axisAlignedSliceDrags.end()) {
        it->second.active = false;
    }
    flushAxisAlignedOrientationUpdate();
}

void CWindow::applySlicePlaneOrientation(Surface* sourceOverride)
{
    if (!_surf_col) {
        return;
    }

    cancelAxisAlignedOrientationTimer();

    POI *focus = _surf_col->poi("focus");
    cv::Vec3f origin = focus ? focus->p : cv::Vec3f(0, 0, 0);

    if (_useAxisAlignedSlices) {
        auto segXZShared = std::dynamic_pointer_cast<PlaneSurface>(_surf_col->surface("seg xz"));
        auto segYZShared = std::dynamic_pointer_cast<PlaneSurface>(_surf_col->surface("seg yz"));

        if (!segXZShared) {
            segXZShared = std::make_shared<PlaneSurface>();
        }
        if (!segYZShared) {
            segYZShared = std::make_shared<PlaneSurface>();
        }


        const auto configurePlane = [&](PlaneSurface* plane,
                                        float degrees,
                                        const cv::Vec3f& baseNormal) {
            if (!plane) {
                return;
            }

            plane->setOrigin(origin);
            plane->setInPlaneRotation(0.0f);

            const float radians = degrees * kDegToRad;
            const cv::Vec3f rotatedNormal = rotateAroundZ(baseNormal, radians);
            plane->setNormal(rotatedNormal);

            const cv::Vec3f upAxis(0.0f, 0.0f, 1.0f);
            const cv::Vec3f projectedUp = projectVectorOntoPlane(upAxis, rotatedNormal);
            const cv::Vec3f desiredUp = normalizeOrZero(projectedUp);

            if (cv::norm(desiredUp) > kEpsilon) {
                const cv::Vec3f currentUp = plane->basisY();
                const float delta = signedAngleBetween(currentUp, desiredUp, rotatedNormal);
                if (std::abs(delta) > kEpsilon) {
                    plane->setInPlaneRotation(delta);
                }
            } else {
                plane->setInPlaneRotation(0.0f);
            }
        };

        configurePlane(segXZShared.get(), _axisAlignedSegXZRotationDeg, cv::Vec3f(0.0f, 1.0f, 0.0f));
        configurePlane(segYZShared.get(), _axisAlignedSegYZRotationDeg, cv::Vec3f(1.0f, 0.0f, 0.0f));

        if (segXZShared) {
            segXZShared->setAxisAlignedRotationKey(axisAlignedRotationCacheKey(_axisAlignedSegXZRotationDeg));
        }
        if (segYZShared) {
            segYZShared->setAxisAlignedRotationKey(axisAlignedRotationCacheKey(_axisAlignedSegYZRotationDeg));
        }

        _surf_col->setSurface("seg xz", segXZShared);
        _surf_col->setSurface("seg yz", segYZShared);
        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
        return;
    } else {
        QuadSurface* segment = nullptr;
        std::shared_ptr<Surface> segmentHolder;  // Keep surface alive during this scope
        if (sourceOverride) {
            segment = dynamic_cast<QuadSurface*>(sourceOverride);
        } else {
            segmentHolder = _surf_col->surface("segmentation");
            segment = dynamic_cast<QuadSurface*>(segmentHolder.get());
        }
        if (!segment) {
            return;
        }

        auto segXZShared = std::dynamic_pointer_cast<PlaneSurface>(_surf_col->surface("seg xz"));
        auto segYZShared = std::dynamic_pointer_cast<PlaneSurface>(_surf_col->surface("seg yz"));

        if (!segXZShared) {
            segXZShared = std::make_shared<PlaneSurface>();
        }
        if (!segYZShared) {
            segYZShared = std::make_shared<PlaneSurface>();
        }

        segXZShared->setOrigin(origin);
        segYZShared->setOrigin(origin);

        auto ptr = segment->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        segment->pointTo(ptr, origin, 1.0f, 1000, patchIndex);

        cv::Vec3f xDir = segment->coord(ptr, {1, 0, 0});
        cv::Vec3f yDir = segment->coord(ptr, {0, 1, 0});
        segXZShared->setNormal(xDir - origin);
        segYZShared->setNormal(yDir - origin);
        segXZShared->setInPlaneRotation(0.0f);
        segYZShared->setInPlaneRotation(0.0f);
        segXZShared->setAxisAlignedRotationKey(-1);
        segYZShared->setAxisAlignedRotationKey(-1);

        _surf_col->setSurface("seg xz", segXZShared);
        _surf_col->setSurface("seg yz", segYZShared);
        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
        return;
    }
}


void CWindow::startWatchingWithInotify()
{
    if (!fVpkg) {
        return;
    }

    // Check if file watching is enabled in settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (!settings.value(vc3d::settings::perf::ENABLE_FILE_WATCHING, vc3d::settings::perf::ENABLE_FILE_WATCHING_DEFAULT).toBool()) {
        Logger()->info("File watching is disabled in settings");
        return;
    }

    // Stop any existing watches
    stopWatchingWithInotify();

    // Initialize inotify
    _inotifyFd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (_inotifyFd < 0) {
        Logger()->error("Failed to initialize inotify: {}", strerror(errno));
        return;
    }

    // Watch both paths and traces directories
    auto availableDirs = fVpkg->getAvailableSegmentationDirectories();
    for (const auto& dirName : availableDirs) {
        std::filesystem::path dirPath = std::filesystem::path(fVpkg->getVolpkgDirectory()) / dirName;

        if (!std::filesystem::exists(dirPath)) {
            Logger()->debug("Directory {} does not exist, skipping watch", dirPath.string());
            continue;
        }

        // Watch for directory create, delete, and move events
        int wd = inotify_add_watch(_inotifyFd, dirPath.c_str(),
                                  IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_ONLYDIR);

        if (wd < 0) {
            Logger()->error("Failed to add inotify watch for {}: {}", dirPath.string(), strerror(errno));
            continue;
        }

        _watchDescriptors[wd] = dirName;
        Logger()->info("Started inotify watch for {} directory (wd={})", dirName, wd);
    }

    // Set up Qt socket notifier to integrate with event loop
    _inotifyNotifier = new QSocketNotifier(_inotifyFd, QSocketNotifier::Read, this);
    connect(_inotifyNotifier, &QSocketNotifier::activated, this, &CWindow::onInotifyEvent);;
}

void CWindow::stopWatchingWithInotify()
{
    if (_inotifyProcessTimer) {if (_inotifyProcessTimer) {
        _inotifyProcessTimer->stop();
    }
        _inotifyProcessTimer->stop();
    }

    if (_inotifyNotifier) {
        delete _inotifyNotifier;
        _inotifyNotifier = nullptr;
    }

    if (_inotifyFd >= 0) {
        // Remove all watches
        for (const auto& [wd, dirName] : _watchDescriptors) {
            inotify_rm_watch(_inotifyFd, wd);
        }
        _watchDescriptors.clear();

        ::close(_inotifyFd);
        _inotifyFd = -1;
    }

    _pendingMoves.clear();
}

void CWindow::onInotifyEvent()
{
    char buffer[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
    ssize_t length = read(_inotifyFd, buffer, sizeof(buffer));

    if (length < 0) {
        if (errno != EAGAIN) {
            std::cerr << "Error reading inotify events: " << strerror(errno) << std::endl;
        }
        return;
    }

    ssize_t i = 0;
    while (i < length) {
        struct inotify_event* event = reinterpret_cast<struct inotify_event*>(&buffer[i]);

        if (event->len > 0) {
            std::string fileName(event->name);

            // Skip hidden files and temporary files
            if (fileName.empty() || fileName[0] == '.' || fileName.find("~") != std::string::npos) {
                i += sizeof(struct inotify_event) + event->len;
                continue;
            }

            // Find the directory name for this watch descriptor
            auto it = _watchDescriptors.find(event->wd);
            if (it != _watchDescriptors.end()) {
                std::string dirName = it->second;

                // Handle different event types
                if (event->mask & IN_CREATE) {
                    // New segment created
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Addition;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingInotifyEvents.push_back(evt);

                } else if (event->mask & IN_DELETE) {
                    // Segment deleted
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Removal;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingInotifyEvents.push_back(evt);

                } else if (event->mask & IN_MOVED_FROM) {
                    // First part of move/rename - store with cookie
                    // Store both the filename and directory for orphaned move cleanup
                    _pendingMoves[event->cookie] = dirName + "/" + fileName;

                } else if (event->mask & IN_MOVED_TO) {
                    // Second part of move/rename
                    auto moveIt = _pendingMoves.find(event->cookie);
                    if (moveIt != _pendingMoves.end()) {
                        // This is a rename within watched directories
                        // Extract the old filename from the stored path
                        std::string oldPath = moveIt->second;
                        size_t lastSlash = oldPath.rfind('/');
                        std::string oldName = (lastSlash != std::string::npos) ?
                                               oldPath.substr(lastSlash + 1) : oldPath;
                        _pendingMoves.erase(moveIt);

                        InotifyEvent evt;
                        evt.type = InotifyEvent::Rename;
                        evt.dirName = dirName;
                        evt.segmentId = oldName;  // old segment ID
                        evt.newId = fileName;      // new segment ID
                        _pendingInotifyEvents.push_back(evt);

                    } else {
                        // File moved from outside watched directory - treat as new addition
                        InotifyEvent evt;
                        evt.type = InotifyEvent::Addition;
                        evt.dirName = dirName;
                        evt.segmentId = fileName;
                        _pendingInotifyEvents.push_back(evt);
                    }

                } else if (event->mask & (IN_MODIFY | IN_CLOSE_WRITE)) {
                    // Segment modified or closed after writing
                    // Use set to avoid duplicate updates for the same segment
                    _pendingSegmentUpdates.insert({dirName, fileName});
                }

                // Handle overflow
                if (event->mask & IN_Q_OVERFLOW) {
                    std::cerr << "Inotify queue overflow - some events may have been lost" << std::endl;
                    // Could trigger a full reload here if needed
                }
            }
        }

        i += sizeof(struct inotify_event) + event->len;
    }

    // Clean up old pending moves that never got their MOVED_TO pair
    if (!_pendingMoves.empty()) {
        static auto lastCleanup = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();

        // Clean up orphaned moves every 5 seconds
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastCleanup).count() > 5) {
            for (auto it = _pendingMoves.begin(); it != _pendingMoves.end(); ) {
                // Extract directory and filename from stored path
                std::string fullPath = it->second;
                size_t lastSlash = fullPath.rfind('/');
                if (lastSlash != std::string::npos) {
                    std::string dirName = fullPath.substr(0, lastSlash);
                    std::string fileName = fullPath.substr(lastSlash + 1);

                    // Treat orphaned MOVED_FROM as deletions
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Removal;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingInotifyEvents.push_back(evt);
                }

                it = _pendingMoves.erase(it);
            }
            lastCleanup = now;
        }
    }

    scheduleInotifyProcessing();
}

void CWindow::processInotifySegmentUpdate(const std::string& dirName, const std::string& segmentName)
{
    if (!fVpkg) return;

    Logger()->info("Processing update of {} in {}", segmentName, dirName);

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());
    if (!isCurrentDir) {
        Logger()->debug("Update in non-current directory {}, skipping UI update", dirName);
        return;
    }

    std::string segmentId = segmentName; // UUID = directory name

    // Check if the segment exists
    auto seg = fVpkg->segmentation(segmentId);
    if (!seg) {
        Logger()->warn("Segment {} not found for update, treating as addition", segmentId);
        processInotifySegmentAddition(dirName, segmentName);
        return;
    }

    // Skip reloads triggered right after editing sessions end
    if (shouldSkipInotifyForSegment(segmentId, "reload")) {
        return;
    }

    bool wasSelected = (_surfID == segmentId);

    // Reload the segmentation
    if (fVpkg->reloadSingleSegmentation(segmentId)) {
        try {
            auto reloadedSurf = fVpkg->loadSurface(segmentId);
            if (reloadedSurf) {
                if (_surf_col) {
                    _surf_col->setSurface(segmentId, reloadedSurf, false, false);
                }

                if (_surfacePanel) {
                    _surfacePanel->refreshSurfaceMetrics(segmentId);
                }

                statusBar()->showMessage(tr("Updated: %1").arg(QString::fromStdString(segmentName)), 2000);

                if (wasSelected) {
                    _surfID = segmentId;
                    _surf_weak = reloadedSurf;

                    if (_surf_col) {
                        _surf_col->setSurface("segmentation", reloadedSurf, false, false);
                    }

                    if (_surfacePanel) {
                        _surfacePanel->syncSelectionUi(segmentId, reloadedSurf.get());
                    }
                }
            }
        } catch (const std::exception& e) {
            Logger()->error("Failed to reload segment {}: {}", segmentId, e.what());
        }
    }
}

void CWindow::processInotifySegmentRename(const std::string& dirName,
                                          const std::string& oldDirName,
                                          const std::string& newDirName)
{
    if (!fVpkg) return;

    Logger()->info("Processing rename in {}: {} -> {}", dirName, oldDirName, newDirName);

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());

    // The old UUID would have been the old directory name
    std::string oldId = oldDirName;
    std::string newId = newDirName;

    // Check if the old segment exists
    if (!fVpkg->segmentation(oldId)) {
        Logger()->warn("Old segment {} not found, treating as new addition", oldId);
        processInotifySegmentAddition(dirName, newDirName);
        return;
    }

    // Remove the old entry
    bool wasSelected = isCurrentDir && (_surfID == oldId);
    fVpkg->removeSingleSegmentation(oldId);

    if (isCurrentDir && _surfacePanel) {
        _surfacePanel->removeSingleSegmentation(oldId);
    }

    // Add with new name (which will read the meta.json and update the UUID)
    std::string previousDir;
    if (!isCurrentDir) {
        previousDir = fVpkg->getSegmentationDirectory();
        fVpkg->setSegmentationDirectory(dirName);
    }

    if (fVpkg->addSingleSegmentation(newDirName)) {
        // The UUID in meta.json will be updated when the segment is saved/loaded
        try {
            auto loadedSurf = fVpkg->loadSurface(newId);

            if (loadedSurf && isCurrentDir) {
                _surf_col->setSurface(newId, loadedSurf, true);
                if (_surfacePanel) {
                    _surfacePanel->addSingleSegmentation(newId);
                }

                statusBar()->showMessage(tr("Renamed: %1 → %2")
                                       .arg(QString::fromStdString(oldDirName),
                                            QString::fromStdString(newDirName)), 3000);

                // Reselect if it was selected
                if (wasSelected) {
                    _surfID = newId;
                    auto surf = _surf_col->surface(newId);
                    if (surf && _surfacePanel) {
                        _surfacePanel->syncSelectionUi(newId, dynamic_cast<QuadSurface*>(surf.get()));
                    }
                }

                if (_surfacePanel) {
                    //_surfacePanel->refreshFiltersOnly();
                }
            }
        } catch (const std::exception& e) {
            Logger()->error("Failed to load renamed segment {}: {}", newId, e.what());
        }
    }

    if (!isCurrentDir && !previousDir.empty()) {
        fVpkg->setSegmentationDirectory(previousDir);
    }
}

void CWindow::processInotifySegmentAddition(const std::string& dirName, const std::string& segmentName)
{
    if (!fVpkg) return;

    Logger()->info("Processing addition of {} to {}", segmentName, dirName);

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());

    // The UUID will be the directory name (or will be updated to match)
    std::string segmentId = segmentName;

    // Skip addition if editing just stopped recently to avoid thrashing the active surface
    if (shouldSkipInotifyForSegment(segmentId, "addition")) {
        return;
    }

    // Switch directory if needed
    std::string previousDir;
    if (!isCurrentDir) {
        previousDir = fVpkg->getSegmentationDirectory();
        fVpkg->setSegmentationDirectory(dirName);
    }

    // Add the segment
    if (fVpkg->addSingleSegmentation(segmentName)) {
        if (isCurrentDir) {
            try {
                auto loadedSurf = fVpkg->loadSurface(segmentId);
                if (loadedSurf) {
                    _surf_col->setSurface(segmentId, loadedSurf, true);
                    if (_surfacePanel) {
                        _surfacePanel->addSingleSegmentation(segmentId);
                    }
                    statusBar()->showMessage(tr("Added: %1").arg(QString::fromStdString(segmentName)), 2000);
                    if (_surfacePanel) {
                        //_surfacePanel->refreshFiltersOnly();
                    }
                }
            } catch (const std::exception& e) {
                Logger()->error("Failed to load segment {}: {}", segmentId, e.what());
            }
        }
    }

    if (!isCurrentDir && !previousDir.empty()) {
        fVpkg->setSegmentationDirectory(previousDir);
    }
}

void CWindow::processInotifySegmentRemoval(const std::string& dirName, const std::string& segmentName)
{
    if (!fVpkg) return;

    std::string segmentId = segmentName;

    Logger()->info("Processing removal of {} from {}", segmentId, dirName);

    // First check if this segment even exists and belongs to this directory
    auto seg = fVpkg->segmentation(segmentId);
    if (!seg) {
        Logger()->debug("Segment {} not found, ignoring removal event from {}", segmentId, dirName);
        return;
    }

    // Verify the segment is actually in the directory that reported the removal
    if (seg->path().parent_path().filename() != dirName) {
        Logger()->warn("Removal event for {} from {}, but segment is actually in {}",
                      segmentId, dirName, seg->path().parent_path().filename().string());
        return;
    }

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());

    // Skip removal if editing just ended or is still running
    if (shouldSkipInotifyForSegment(segmentId, "removal")) {
        return;
    }

    // Remove from VolumePkg
    if (fVpkg->removeSingleSegmentation(segmentId)) {
        if (isCurrentDir && _surfacePanel) {
            _surfacePanel->removeSingleSegmentation(segmentId);
            statusBar()->showMessage(tr("Removed: %1").arg(QString::fromStdString(segmentName)), 2000);
            //_surfacePanel->refreshFiltersOnly();
        }
    }
}

void CWindow::processPendingInotifyEvents()
{
    if (_pendingInotifyEvents.empty() && _pendingSegmentUpdates.empty()) {
        return;
    }

    // Store current selection to restore later
    std::string previousSelection = _surfID;
    auto previousSurface = _surf_weak.lock();

    // Track if the previously selected segment gets removed
    bool previousSelectionRemoved = false;

    // Sort events to process removals first, then renames, then additions
    std::vector<InotifyEvent> removals, renames, additions, updates;
    for (const auto& evt : _pendingInotifyEvents) {
        switch (evt.type) {
            case InotifyEvent::Removal:
                removals.push_back(evt);
                break;
            case InotifyEvent::Rename:
                renames.push_back(evt);
                break;
            case InotifyEvent::Addition:
                additions.push_back(evt);
                break;
            case InotifyEvent::Update:
                updates.push_back(evt);
                break;
        }
    }

    if (!removals.empty() && !additions.empty()) {
        using EventKey = std::pair<std::string, std::string>;
        auto makeKey = [](const InotifyEvent& evt) -> EventKey {
            return {evt.dirName, evt.segmentId};
        };

        std::map<EventKey, int> additionCounts;
        for (const auto& addition : additions) {
            additionCounts[makeKey(addition)]++;
        }

        std::map<EventKey, int> pairedCounts;
        std::vector<InotifyEvent> filteredRemovals;
        filteredRemovals.reserve(removals.size());

        for (const auto& removal : removals) {
            EventKey key = makeKey(removal);
            auto availableIt = additionCounts.find(key);
            const int available = (availableIt != additionCounts.end()) ? availableIt->second : 0;
            auto existingPairIt = pairedCounts.find(key);
            const int alreadyPaired = (existingPairIt != pairedCounts.end()) ? existingPairIt->second : 0;

            if (available > alreadyPaired) {
                pairedCounts[key] = alreadyPaired + 1;

                InotifyEvent updateEvt;
                updateEvt.type = InotifyEvent::Update;
                updateEvt.dirName = removal.dirName;
                updateEvt.segmentId = removal.segmentId;
                updates.push_back(updateEvt);
            } else {
                filteredRemovals.push_back(removal);
            }
        }

        std::vector<InotifyEvent> filteredAdditions;
        filteredAdditions.reserve(additions.size());
        for (const auto& addition : additions) {
            EventKey key = makeKey(addition);
            auto pairIt = pairedCounts.find(key);
            if (pairIt != pairedCounts.end() && pairIt->second > 0) {
                pairIt->second--;
                continue;
            }
            filteredAdditions.push_back(addition);
        }

        removals.swap(filteredRemovals);
        additions.swap(filteredAdditions);
    }

    for (const auto& evt : removals) {
        if (evt.segmentId == previousSelection) {
            previousSelectionRemoved = true;
            break;
        }
    }

    // Process in order: removals, renames, additions, updates
    for (const auto& evt : removals) {
        processInotifySegmentRemoval(evt.dirName, evt.segmentId);
    }

    for (const auto& evt : renames) {
        processInotifySegmentRename(evt.dirName, evt.segmentId, evt.newId);
    }

    for (const auto& evt : additions) {
        processInotifySegmentAddition(evt.dirName, evt.segmentId);
    }

    for (const auto& evt : updates) {
        processInotifySegmentUpdate(evt.dirName, evt.segmentId);
    }

    // Process unique segment updates
    for (const auto& [dirName, segmentId] : _pendingSegmentUpdates) {
        processInotifySegmentUpdate(dirName, segmentId);
    }

    // Clear the queues
    _pendingInotifyEvents.clear();
    _pendingSegmentUpdates.clear();

    // Restore selection if it still exists (might have been renamed or re-added)
    if (!previousSelection.empty() && previousSelectionRemoved) {
        // If editing is active for this segment, skip re-emitting the segmentation surface change
        // because autosave-triggered inotify events were intentionally skipped.
        bool skipRestoreForActiveEdit = false;
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            auto* activeBaseSurface = _segmentationModule->activeBaseSurface();
            if (activeBaseSurface && activeBaseSurface->id == previousSelection) {
                Logger()->info("Skipping selection restore of {} - currently being edited", previousSelection);
                skipRestoreForActiveEdit = true;
            }
        }

        if (!skipRestoreForActiveEdit) {
            // Check if the segment was re-added in this batch
            bool wasReAdded = false;
            for (const auto& evt : additions) {
                if (evt.segmentId == previousSelection) {
                    wasReAdded = true;
                    break;
                }
            }

            if (wasReAdded) {
                // The segment was removed and re-added - restore selection
                auto seg = fVpkg ? fVpkg->segmentation(previousSelection) : nullptr;
                if (seg) {
                    _surfID = previousSelection;
                    auto surf = fVpkg->getSurface(previousSelection);
                    if (surf) {
                        _surf_weak = surf;

                        if (_surf_col) {
                            _surf_col->setSurface("segmentation", surf, false, false);
                        }

                        if (treeWidgetSurfaces) {
                            QTreeWidgetItemIterator it(treeWidgetSurfaces);
                            while (*it) {
                                if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == previousSelection) {
                                    const QSignalBlocker blocker{treeWidgetSurfaces};
                                    treeWidgetSurfaces->clearSelection();
                                    (*it)->setSelected(true);
                                    treeWidgetSurfaces->scrollToItem(*it);
                                    break;
                                }
                                ++it;
                            }
                        }

                        if (_surfacePanel) {
                            _surfacePanel->syncSelectionUi(previousSelection, surf.get());
                        }

                        if (auto* viewer = segmentationViewer()) {
                            viewer->setWindowTitle(tr("Surface %1").arg(QString::fromStdString(previousSelection)));
                        }
                    }
                }
            }
        }
    } else if (!previousSelection.empty()) {
        // Original logic for non-removed segments
        auto seg = fVpkg ? fVpkg->segmentation(previousSelection) : nullptr;
        if (seg) {
            _surfID = previousSelection;
            _surf_weak = previousSurface;

            if (_surfacePanel) {
                auto surface = _surf_col->surface(previousSelection);
                if (surface) {
                    _surfacePanel->syncSelectionUi(previousSelection, dynamic_cast<QuadSurface*>(surface.get()));
                }
            }
        }
    }

    // Refresh filters once at the end instead of multiple times
    if (_surfacePanel) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::scheduleInotifyProcessing()
{
    if (!_inotifyProcessTimer) {
        return;
    }

    // Stop any existing timer
    _inotifyProcessTimer->stop();

    // Use single-shot timer with short delay
    _inotifyProcessTimer->setSingleShot(true);
    _inotifyProcessTimer->setInterval(INOTIFY_THROTTLE_MS);
    _inotifyProcessTimer->start();
}

bool CWindow::shouldSkipInotifyForSegment(const std::string& segmentId, const char* eventCategory)
{
    if (segmentId.empty()) {
        return false;
    }

    const char* category = eventCategory ? eventCategory : "inotify event";

    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        if (auto* activeBaseSurface = _segmentationModule->activeBaseSurface()) {
            if (activeBaseSurface->id == segmentId) {
                Logger()->info("Skipping {} for {} - currently being edited", category, segmentId);
                return true;
            }
        }
    }

    // Also skip if approval mask editing is active for this segment
    if (_segmentationModule && _segmentationModule->isEditingApprovalMask()) {
        // Get the segment being used for approval mask (the active segmentation surface)
        if (_surf_col) {
            auto segSurface = _surf_col->surface("segmentation");
            if (segSurface && segSurface->id == segmentId) {
                Logger()->info("Skipping {} for {} - approval mask being edited", category, segmentId);
                return true;
            }
        }
    }

    pruneExpiredRecentlyEdited();

    if (_recentlyEditedSegments.empty()) {
        return false;
    }

    auto it = _recentlyEditedSegments.find(segmentId);
    if (it == _recentlyEditedSegments.end()) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - it->second;
    const auto grace = std::chrono::seconds(RECENT_EDIT_GRACE_SECONDS);

    if (elapsed < grace) {
        const double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
        const double remainingSeconds = std::max(
            0.0, static_cast<double>(RECENT_EDIT_GRACE_SECONDS) - elapsedSeconds);
        Logger()->info("Skipping {} for {} - edits ended {:.1f}s ago ({}s grace remaining)",
                       category,
                       segmentId,
                       elapsedSeconds,
                       static_cast<int>(std::ceil(remainingSeconds)));
        return true;
    }

    _recentlyEditedSegments.erase(it);
    return false;
}

void CWindow::markSegmentRecentlyEdited(const std::string& segmentId)
{
    if (segmentId.empty()) {
        return;
    }

    pruneExpiredRecentlyEdited();
    _recentlyEditedSegments[segmentId] = std::chrono::steady_clock::now();
}

void CWindow::pruneExpiredRecentlyEdited()
{
    if (_recentlyEditedSegments.empty()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto grace = std::chrono::seconds(RECENT_EDIT_GRACE_SECONDS);

    for (auto it = _recentlyEditedSegments.begin(); it != _recentlyEditedSegments.end(); ) {
        if (now - it->second >= grace) {
            it = _recentlyEditedSegments.erase(it);
        } else {
            ++it;
        }
    }
}


void CWindow::onMoveSegmentToPaths(const QString& segmentId)
{
    if (!fVpkg) {
        statusBar()->showMessage(tr("No volume package loaded"), 3000);
        return;
    }

    // Verify we're in traces directory
    if (fVpkg->getSegmentationDirectory() != "traces") {
        statusBar()->showMessage(tr("Can only move segments from traces directory"), 3000);
        return;
    }

    // Get the segment
    auto seg = fVpkg->segmentation(segmentId.toStdString());
    if (!seg) {
        statusBar()->showMessage(tr("Segment not found: %1").arg(segmentId), 3000);
        return;
    }

    // Build paths
    std::filesystem::path volpkgPath(fVpkg->getVolpkgDirectory());
    std::filesystem::path currentPath = seg->path();
    std::filesystem::path newPath = volpkgPath / "paths" / currentPath.filename();

    // Check if destination exists
    if (std::filesystem::exists(newPath)) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            tr("Destination Exists"),
            tr("Segment '%1' already exists in paths/.\nDo you want to replace it?").arg(segmentId),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No
        );

        if (reply != QMessageBox::Yes) {
            return;
        }

        // Remove the existing one
        try {
            std::filesystem::remove_all(newPath);
        } catch (const std::exception& e) {
            QMessageBox::critical(this, tr("Error"),
                tr("Failed to remove existing segment: %1").arg(e.what()));
            return;
        }
    }

    // Confirm the move
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        tr("Move to Paths"),
        tr("Move segment '%1' from traces/ to paths/?\n\n"
           "Note: The segment will be closed if currently open.").arg(segmentId),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes
    );

    if (reply != QMessageBox::Yes) {
        return;
    }

    // === CRITICAL: Clean up the segment before moving ===
    std::string idStd = segmentId.toStdString();

    // Check if this is the currently selected segment
    bool wasSelected = (_surfID == idStd);

    // Clear from surface collection (including "segmentation" if it matches)
    if (_surf_col) {
        auto currentSurface = _surf_col->surface(idStd);
        auto segmentationSurface = _surf_col->surface("segmentation");

        // If this surface is currently shown as "segmentation", clear it
        if (currentSurface && segmentationSurface && currentSurface == segmentationSurface) {
            _surf_col->setSurface("segmentation", nullptr, false, false);
        }

        // Clear the surface from the collection
        _surf_col->setSurface(idStd, nullptr, false, false);
    }

    // Unload the surface from VolumePkg
    fVpkg->unloadSurface(idStd);

    // Clear selection if this was selected
    if (wasSelected) {
        clearSurfaceSelection();

        // Clear tree selection
        if (treeWidgetSurfaces) {
            treeWidgetSurfaces->clearSelection();
        }
    }

    // Perform the move
    try {
        std::filesystem::rename(currentPath, newPath);

        // Remove from VolumePkg's internal tracking for traces
        fVpkg->removeSingleSegmentation(idStd);

        // The inotify system will pick up the IN_MOVED_TO in paths/
        // and handle adding it there if the user switches to that directory

        if (_surfacePanel) {
            _surfacePanel->removeSingleSegmentation(idStd);
        }

        statusBar()->showMessage(
            tr("Moved %1 from traces/ to paths/. Switch to paths directory to see it.").arg(segmentId), 5000);

    } catch (const std::exception& e) {
        // If move failed, we might want to reload the segment
        // but it's probably safer to leave it unloaded
        QMessageBox::critical(this, tr("Error"),
            tr("Failed to move segment: %1\n\n"
               "The segment has been unloaded from the viewer.").arg(e.what()));
    }
}

void CWindow::updateSurfaceOverlayDropdown()
{
    if (!ui.surfaceOverlaySelect) {
        return;
    }

    const QSignalBlocker blocker(ui.surfaceOverlaySelect);
    QString currentSelection = ui.surfaceOverlaySelect->currentData().toString();

    ui.surfaceOverlaySelect->clear();
    ui.surfaceOverlaySelect->addItem(tr("(None)"), QString());

    if (_surf_col) {
        const auto names = _surf_col->surfaceNames();
        int selectedIndex = 0;
        int currentIndex = 1; // Start at 1 since "(None)" is at 0

        for (const auto& name : names) {
            // Only add QuadSurfaces (actual segmentations), skip PlaneSurfaces (xy, xz, yz, etc.)
            auto surf = _surf_col->surface(name);
            if (surf && dynamic_cast<QuadSurface*>(surf.get())) {
                ui.surfaceOverlaySelect->addItem(QString::fromStdString(name), QString::fromStdString(name));

                // Restore previous selection if it still exists
                if (QString::fromStdString(name) == currentSelection) {
                    selectedIndex = currentIndex;
                }
                currentIndex++;
            }
        }

        ui.surfaceOverlaySelect->setCurrentIndex(selectedIndex);
    }
}