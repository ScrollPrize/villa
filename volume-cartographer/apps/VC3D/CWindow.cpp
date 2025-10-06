#include "CWindow.hpp"

#include "WindowRangeWidget.hpp"
#include <QKeySequence>
#include <QHBoxLayout>
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
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationGrowth.hpp"
#include "segmentation/SegmentationGrower.hpp"
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

    _planeSlicingOverlay = std::make_unique<PlaneSlicingOverlayController>(_surf_col, this);
    _planeSlicingOverlay->bindToViewerManager(_viewerManager.get());
    _planeSlicingOverlay->setRotationSetter([this](const std::string& planeName, float degrees) {
        setAxisAlignedRotationDegrees(planeName, degrees);
        applySlicePlaneOrientation();
    });
    _planeSlicingOverlay->setAxisAlignedEnabled(_useAxisAlignedSlices);

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
                               ui.dockWidgetView,
                               ui.dockWidgetOverlay }) {
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

void CWindow::recordFocusHistory(const POI& poi)
{
    if (_navigatingFocusHistory) {
        return;
    }

    FocusHistoryEntry entry;
    entry.position = poi.p;
    entry.normal = poi.n;
    entry.source = poi.src;

    if (_focusHistoryIndex >= 0 &&
        _focusHistoryIndex < static_cast<int>(_focusHistory.size())) {
        const auto& current = _focusHistory[_focusHistoryIndex];
        const float positionDelta = cv::norm(current.position - entry.position);
        const float normalDelta = cv::norm(current.normal - entry.normal);
        if (positionDelta < 1e-4f && normalDelta < 1e-4f && current.source == entry.source) {
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
    centerFocusAt(entry.position, entry.normal, entry.source, false);
    _navigatingFocusHistory = false;
    return true;
}

bool CWindow::centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, Surface* source, bool addToHistory)
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
    if (source) {
        focus->src = source;
    } else if (!focus->src) {
        focus->src = _surf_col->surface("segmentation");
    }

    _surf_col->setPOI("focus", focus);

    if (addToHistory) {
        recordFocusHistory(*focus);
    }

    Surface* orientationSource = focus->src ? focus->src : _surf_col->surface("segmentation");
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

    return centerFocusAt(cursor->p, cursor->n, cursor->src, true);
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
    if (_segmentationGrower) {
        _segmentationGrower->setSurfacePanel(_surfacePanel.get());
    }
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
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetComposite);

    auto ensureTabified = [this](QDockWidget* primary, QDockWidget* candidate) {
        const auto currentTabs = tabifiedDockWidgets(primary);
        const bool alreadyTabified = std::find(currentTabs.cbegin(), currentTabs.cend(), candidate) != currentTabs.cend();
        if (!alreadyTabified) {
            tabifyDockWidget(primary, candidate);
        }
    };

    ensureTabified(ui.dockWidgetView, ui.dockWidgetOverlay);
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
            .opacitySpin = ui.overlayOpacitySpin,
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
        bool showOverlays = settings.value("viewer/show_axis_overlays", true).toBool();
        QSignalBlocker blocker(chkAxisOverlays);
        chkAxisOverlays->setChecked(showOverlays);
        connect(chkAxisOverlays, &QCheckBox::toggled, this, &CWindow::onAxisOverlayVisibilityToggled);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        int storedOpacity = settings.value("viewer/axis_overlay_opacity", spinAxisOverlayOpacity->value()).toInt();
        storedOpacity = std::clamp(storedOpacity, spinAxisOverlayOpacity->minimum(), spinAxisOverlayOpacity->maximum());
        QSignalBlocker blocker(spinAxisOverlayOpacity);
        spinAxisOverlayOpacity->setValue(storedOpacity);
        connect(spinAxisOverlayOpacity, qOverload<int>(&QSpinBox::valueChanged), this, &CWindow::onAxisOverlayOpacityChanged);
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
    if (chkAxisAlignedSlices) {
        bool useAxisAligned = settings.value("viewer/use_axis_aligned_slices", false).toBool();
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
        centerFocusAt(vol_loc, normal, surf, true);
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
    _surf_col->setSurface("segmentation", chain, false, false);
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
    QSettings settings("VC.ini", QSettings::IniFormat);
    settings.setValue("viewer/show_axis_overlays", enabled ? "1" : "0");
}

void CWindow::onAxisOverlayOpacityChanged(int value)
{
    float normalized = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedOverlayOpacity(normalized);
    }
    QSettings settings("VC.ini", QSettings::IniFormat);
    settings.setValue("viewer/axis_overlay_opacity", value);
}

void CWindow::recalcAreaForSegments(const std::vector<std::string>& ids)
{
    if (!fVpkg) return;

    // Linear voxel size (µm/voxel) for cm² conversion
    float voxelsize = 1.0f;
    try {
        if (currentVolume && currentVolume->metadata().hasKey("voxelsize")) {
            voxelsize = currentVolume->metadata().get<float>("voxelsize");
        }
    } catch (...) { voxelsize = 1.0f; }
    if (!std::isfinite(voxelsize) || voxelsize <= 0.f) voxelsize = 1.0f;

    int okCount = 0, failCount = 0;
    QStringList updatedIds, skippedIds;

    for (const auto& id : ids) {
        auto sm = fVpkg->getSurface(id);
        if (!sm || !sm->surface()) {
            ++failCount; skippedIds << QString::fromStdString(id) + " (missing surface)";
            continue;
        }
        auto* surf = sm->surface(); // QuadSurface*

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
            if (!surf->meta) surf->meta = new nlohmann::json();
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
    QSettings settings("VC.ini", QSettings::IniFormat);
    settings.setValue("viewer/use_axis_aligned_slices", enabled ? "1" : "0");
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
    } else {
        _segmentationModule->endEditingSession();
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
    applySlicePlaneOrientation();

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

        const auto configurePlane = [origin](PlaneSurface* plane,
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

        configurePlane(segXZ, _axisAlignedSegXZRotationDeg, cv::Vec3f(0.0f, 1.0f, 0.0f));
        configurePlane(segYZ, _axisAlignedSegYZRotationDeg, cv::Vec3f(1.0f, 0.0f, 0.0f));

        if (segXZ) {
            segXZ->setAxisAlignedRotationKey(axisAlignedRotationCacheKey(_axisAlignedSegXZRotationDeg));
        }
        if (segYZ) {
            segYZ->setAxisAlignedRotationKey(axisAlignedRotationCacheKey(_axisAlignedSegYZRotationDeg));
        }

        _surf_col->setSurface("seg xz", segXZ);
        _surf_col->setSurface("seg yz", segYZ);
        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
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
        segXZ->setInPlaneRotation(0.0f);
        segYZ->setInPlaneRotation(0.0f);
        segXZ->setAxisAlignedRotationKey(-1);
        segYZ->setAxisAlignedRotationKey(-1);

        _surf_col->setSurface("seg xz", segXZ);
        _surf_col->setSurface("seg yz", segYZ);
        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
        return;
    }
}
