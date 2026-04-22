#include "CWindow.hpp"
#include <iostream>
#include "RamStats.hpp"

#include <cstdlib>
#include <functional>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#if defined(VC_HAVE_MIMALLOC)
#include <mimalloc.h>
#endif

#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "WindowRangeWidget.hpp"
#include "VCSettings.hpp"
#include "Keybinds.hpp"
#include <QKeySequence>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QCursor>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <QGuiApplication>
#include <QScreen>
#include <QStyleHints>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QDateTime>
#include <QFileDialog>
#include <QTextStream>
#include <QFileInfo>
#include <QDir>
#include <QEventLoop>
#include <QProgressDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QThread>
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
#include "utils/Json.hpp"
#include <QGraphicsSimpleTextItem>
#include <QPointer>
#include <QPen>
#include <QListView>
#include <QFont>
#include <QPainter>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cmath>
#include "vc/core/types/Segmentation.hpp"
#include <limits>
#include <optional>
#include <cctype>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <fstream>
#include <vector>
#include <initializer_list>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QStringList>

#include "CVolumeViewerView.hpp"
#include "VolumeViewerCmaps.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"
#include "SettingsDialog.hpp"
#include "elements/VolumeSelector.hpp"
#include "CPointCollectionWidget.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "DrawingWidget.hpp"
#include "CommandLineToolRunner.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "SurfacePanelController.hpp"
#include "MenuActionController.hpp"
#include "FileWatcherService.hpp"
#include "AxisAlignedSliceController.hpp"
#include "SurfaceAreaCalculator.hpp"
#include "SegmentationCommandHandler.hpp"
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
#include "vc/core/util/NetworkFilesystem.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include <utils/zarr.hpp>





Q_LOGGING_CATEGORY(lcSegGrowth, "vc.segmentation.growth");

using qga = QGuiApplication;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;

namespace
{

std::string compositeMethodForModeIndex(int index)
{
    switch (index) {
        case 0:  return "max";
        case 1:  return "mean";
        case 2:  return "min";
        case 3:  return "alpha";
        case 4:  return "beerLambert";
        case 5:  return "volumetric";
        case 6:  return "dvr";
        case 7:  return "firstHitIso";
        case 8:  return "devFromMean";
        case 9:  return "emissionDvr";
        case 10: return "maxAboveIso";
        case 11: return "gammaWeighted";
        case 12: return "gradientMag";
        case 13: return "pbrIso";
        case 14: return "shadedDvr";
        default: return "mean";
    }
}

int compositeModeIndexForMethod(const std::string& method)
{
    if (method == "max") return 0;
    if (method == "mean") return 1;
    if (method == "min") return 2;
    if (method == "alpha") return 3;
    if (method == "beerLambert") return 4;
    if (method == "volumetric") return 5;
    if (method == "dvr") return 6;
    if (method == "firstHitIso") return 7;
    if (method == "devFromMean") return 8;
    if (method == "emissionDvr") return 9;
    if (method == "maxAboveIso") return 10;
    if (method == "gammaWeighted") return 11;
    if (method == "gradientMag") return 12;
    if (method == "pbrIso") return 13;
    if (method == "shadedDvr") return 14;
    return 1;
}

bool isRemoteTransformSource(const QString& source)
{
    const QString trimmed = source.trimmed();
    return trimmed.startsWith("http://", Qt::CaseInsensitive) ||
           trimmed.startsWith("https://", Qt::CaseInsensitive) ||
           trimmed.startsWith("s3://", Qt::CaseInsensitive) ||
           trimmed.startsWith("s3+", Qt::CaseInsensitive);
}

std::filesystem::path expandLocalTransformPath(const QString& source)
{
    QString path = source.trimmed();
    if (path.startsWith("~/")) {
        path.replace(0, 1, QDir::homePath());
    } else if (path == "~") {
        path = QDir::homePath();
    }

    std::filesystem::path fsPath = path.toStdString();
    if (fsPath.is_relative()) {
        fsPath = std::filesystem::absolute(fsPath);
    }
    return fsPath;
}

vc::cache::HttpAuth authForRemoteTransformSource(const QString& source)
{
    vc::cache::HttpAuth auth;
    const auto resolved = vc::resolveRemoteUrl(source.trimmed().toStdString());
    if (!resolved.useAwsSigv4) {
        return auth;
    }

    auth = vc::cache::loadAwsCredentials();
    if (auth.region.empty())
        auth.region = resolved.awsRegion;
    // Fall back to saved QSettings if ~/.aws/ files had nothing
    if (auth.access_key.empty() || auth.secret_key.empty()) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const auto savedAccess = settings.value(vc3d::settings::aws::ACCESS_KEY).toString();
        const auto savedSecret = settings.value(vc3d::settings::aws::SECRET_KEY).toString();
        const auto savedToken = settings.value(vc3d::settings::aws::SESSION_TOKEN).toString();

        if (!savedAccess.isEmpty() && !savedSecret.isEmpty()) {
            auth.access_key = savedAccess.toStdString();
            auth.secret_key = savedSecret.toStdString();
            auth.session_token = savedToken.toStdString();
        }
    }

    return auth;
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
    std::filesystem::path rootPath(pkg->getVolpkgDirectory());
    if (rootPath.empty()) {
        if (hint) {
            *hint = QObject::tr("Normal3D lookup skipped (volume package path empty)");
        }
        return {};
    }
    const std::filesystem::path base = rootPath / "normal3d";
    const QString baseStr = QString::fromStdString(base.string());
    if (hint) {
        *hint = QObject::tr("Checked: %1").arg(baseStr);
    }
    std::error_code ec;
    if (!std::filesystem::exists(base, ec) || !std::filesystem::is_directory(base, ec)) {
        return {};
    }

    QStringList candidates;
    for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_directory(ec) || ec) {
            continue;
        }
        const std::filesystem::path p = entry.path();
        // Heuristic: treat as zarr if it contains x/0, y/0, z/0.
        if (std::filesystem::is_directory(p / "x" / "0") &&
            std::filesystem::is_directory(p / "y" / "0") &&
            std::filesystem::is_directory(p / "z" / "0")) {
            candidates.push_back(QString::fromStdString(p.string()));
        }
    }

    candidates.sort();
    return candidates;
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

cv::Matx44d loadAffineTransformMatrix(const std::filesystem::path& path)
{
    if (path.empty()) {
        throw std::runtime_error("transform path is empty");
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("transform.json not found");
    }

    utils::Json json = utils::Json::parse_file(path);

    if (!json.contains("transformation_matrix")) {
        throw std::runtime_error("transform.json is missing transformation_matrix");
    }

    const auto& matrixJson = json.at("transformation_matrix");
    if (!matrixJson.is_array() || (matrixJson.size() != 3 && matrixJson.size() != 4)) {
        throw std::runtime_error("transformation_matrix must be 3x4 or 4x4");
    }

    cv::Matx44d matrix = cv::Matx44d::eye();
    for (int row = 0; row < static_cast<int>(matrixJson.size()); ++row) {
        const auto& rowJson = matrixJson.at(row);
        if (!rowJson.is_array() || rowJson.size() != 4) {
            throw std::runtime_error("each transformation_matrix row must have 4 values");
        }
        for (int col = 0; col < 4; ++col) {
            matrix(row, col) = rowJson.at(col).get_double();
        }
    }

    if (matrixJson.size() == 4) {
        if (std::abs(matrix(3, 0)) > 1e-12 ||
            std::abs(matrix(3, 1)) > 1e-12 ||
            std::abs(matrix(3, 2)) > 1e-12 ||
            std::abs(matrix(3, 3) - 1.0) > 1e-12) {
            throw std::runtime_error("transform.json bottom row must be [0, 0, 0, 1]");
        }
    }

    return matrix;
}

cv::Matx44d invertAffineTransformMatrix(const cv::Matx44d& matrix)
{
    const cv::Matx33d linear(matrix(0, 0), matrix(0, 1), matrix(0, 2),
                             matrix(1, 0), matrix(1, 1), matrix(1, 2),
                             matrix(2, 0), matrix(2, 1), matrix(2, 2));
    const double determinant = cv::determinant(linear);
    if (!std::isfinite(determinant) || std::abs(determinant) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("transform is not invertible");
    }

    cv::Mat linearMat(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            linearMat.at<double>(row, col) = matrix(row, col);
        }
    }
    cv::Mat linearInvMat;
    if (cv::invert(linearMat, linearInvMat, cv::DECOMP_SVD) <= 0.0) {
        throw std::runtime_error("transform is not invertible");
    }

    cv::Matx33d linearInv;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            linearInv(row, col) = linearInvMat.at<double>(row, col);
        }
    }

    const cv::Vec3d translation(matrix(0, 3), matrix(1, 3), matrix(2, 3));
    const cv::Vec3d inverseTranslation = -(linearInv * translation);

    cv::Matx44d inverted = cv::Matx44d::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            inverted(row, col) = linearInv(row, col);
        }
    }
    inverted(0, 3) = inverseTranslation[0];
    inverted(1, 3) = inverseTranslation[1];
    inverted(2, 3) = inverseTranslation[2];
    return inverted;
}

cv::Vec3f applyAffineTransform(const cv::Vec3f& point, const cv::Matx44d& matrix)
{
    if (point[0] == -1.0f) {
        return point;
    }

    const cv::Vec4d homogeneous(point[0], point[1], point[2], 1.0);
    const cv::Vec4d transformed = matrix * homogeneous;
    return cv::Vec3f(static_cast<float>(transformed[0]),
                     static_cast<float>(transformed[1]),
                     static_cast<float>(transformed[2]));
}

cv::Vec3f applyPreAffineScale(const cv::Vec3f& point, int scale)
{
    if (point[0] == -1.0f || scale == 1) {
        return point;
    }

    return point * static_cast<float>(scale);
}

void transformSurfacePoints(QuadSurface* surface, int scale, const std::optional<cv::Matx44d>& matrix)
{
    if (!surface) {
        return;
    }

    if (auto* points = surface->rawPointsPtr()) {
        for (int row = 0; row < points->rows; ++row) {
            for (int col = 0; col < points->cols; ++col) {
                auto& point = (*points)(row, col);
                if (point[0] == -1.0f) {
                    continue;
                }

                point = applyPreAffineScale(point, scale);
                if (matrix) {
                    point = applyAffineTransform(point, *matrix);
                }
            }
        }
    }
}

void refreshTransformedSurfaceState(QuadSurface* surface)
{
    if (!surface) {
        return;
    }

    surface->invalidateCache();

    if (surface->meta.is_null() || !surface->meta.is_object()) {
        surface->meta = utils::Json::object();
    }

    const auto bbox = surface->bbox();
    {
        auto lo = utils::Json::array();
        lo.push_back(bbox.low[0]); lo.push_back(bbox.low[1]); lo.push_back(bbox.low[2]);
        auto hi = utils::Json::array();
        hi.push_back(bbox.high[0]); hi.push_back(bbox.high[1]); hi.push_back(bbox.high[2]);
        auto bb = utils::Json::array();
        bb.push_back(std::move(lo)); bb.push_back(std::move(hi));
        surface->meta["bbox"] = std::move(bb);
    }
    {
        auto sc = utils::Json::array();
        sc.push_back(surface->scale()[0]); sc.push_back(surface->scale()[1]);
        surface->meta["scale"] = std::move(sc);
    }
}

std::shared_ptr<QuadSurface> cloneSurfaceForTransform(const std::shared_ptr<QuadSurface>& source)
{
    if (!source) {
        return nullptr;
    }

    auto clone = std::make_shared<QuadSurface>(source->rawPoints(), source->scale());
    clone->meta = source->meta.is_null() ? utils::Json::object() : source->meta;
    clone->id = source->id;
    clone->path = source->path;
    clone->setOverlappingIds(source->overlappingIds());

    for (const auto& channelName : source->channelNames()) {
        clone->setChannel(channelName, source->channel(channelName, SURF_CHANNEL_NORESIZE).clone());
    }

    return clone;
}

void primeRemoteLevel5WithDialog(CWindow* window, const std::shared_ptr<Volume>& volume)
{
    if (!window || !volume) return;
    // Level-5 is loaded automatically into the BlockCache resident region
    // when the BlockPipeline is first created. Trigger it asynchronously.
    auto* watcher = new QFutureWatcher<void>(window);
    QObject::connect(watcher, &QFutureWatcher<void>::finished, watcher, &QObject::deleteLater);
    watcher->setFuture(QtConcurrent::run([volume]() {
        (void)volume->tieredCache();
    }));
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
    _drawingWidget(nullptr),
    _point_collection_widget(nullptr)
{
    // Initialize timer for debounced window state saving (500ms delay)
    _windowStateSaveTimer = new QTimer(this);
    _windowStateSaveTimer->setSingleShot(true);
    _windowStateSaveTimer->setInterval(500);
    connect(_windowStateSaveTimer, &QTimer::timeout, this, &CWindow::saveWindowState);

    // Periodic heap trim: under mimalloc, mi_collect asks it to purge
    // thread / segment caches and return freed pages to the OS. Under
    // glibc, malloc_trim returns sbrk-grown segments. Also dumps a RAM
    // stats line for live monitoring.
    auto* trimTimer = new QTimer(this);
    trimTimer->setInterval(1000);
    connect(trimTimer, &QTimer::timeout, this, [this]() {
#if defined(VC_HAVE_MIMALLOC)
        mi_collect(false);
#elif defined(__GLIBC__)
        ::malloc_trim(0);
#endif
        vc3d::ramstats::dumpOnce(_viewerManager.get(), _state);
    });
    trimTimer->start();

    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _mirrorCursorToSegmentation = settings.value(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION,
                                                  vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION_DEFAULT).toBool();
    setWindowIcon(QPixmap(":/images/logo.png"));
    ui.setupUi(this);
    ui.cmbCompositeMode->setCurrentIndex(compositeModeIndexForMethod("max"));
    const QString baseTitle = windowTitle();
    const QString repoShortHash = QString::fromStdString(ProjectInfo::RepositoryShortHash()).trimmed();
    if (!repoShortHash.isEmpty() && !repoShortHash.startsWith('@')
        && repoShortHash.compare("Untracked", Qt::CaseInsensitive) != 0) {
        setWindowTitle(QString("%1 %2").arg(baseTitle, repoShortHash));
    }
    // setAttribute(Qt::WA_DeleteOnClose);

    _cacheSizeBytes = cacheSizeGB * 1024ULL * 1024ULL * 1024ULL;
    std::cout << "chunk cache budget is " << cacheSizeGB << " gigabytes" << std::endl;

    _tickCoordinator = std::make_unique<vc::cache::TickCoordinator>();

    _state = new CState(_cacheSizeBytes, this);
    connect(_state, &CState::poiChanged, this, &CWindow::onFocusPOIChanged);
    connect(_state, &CState::surfaceWillBeDeleted, this, &CWindow::onSurfaceWillBeDeleted);

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
    connect(_viewerManager.get(), &ViewerManager::viewerCreated, this, [this](CTiledVolumeViewer* viewer) {
        configureViewerConnections(viewer);
        if (!viewer) {
            return;
        }
        auto s = viewer->compositeRenderSettings();
        s.params.method = compositeMethodForModeIndex(ui.cmbCompositeMode->currentIndex());
        viewer->setCompositeRenderSettings(s);
        if (viewer->surfName() == "segmentation") {
            QSignalBlocker blocker(ui.chkCompositeEnabled);
            ui.chkCompositeEnabled->setChecked(s.enabled);
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
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetDrawing }) {
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
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetDrawing,
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

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }

    // If enabled, auto open the last used volume (local or remote, deferred so window shows first)
    if (settings.value(vc3d::settings::volpkg::AUTO_OPEN, vc3d::settings::volpkg::AUTO_OPEN_DEFAULT).toInt() != 0) {

        QStringList files = settings.value(vc3d::settings::volpkg::RECENT).toStringList();
        QStringList remoteUrls = settings.value(vc3d::settings::viewer::REMOTE_RECENT_URLS).toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            // Local volpkg available — open it
            QString path = files[0];
            QTimer::singleShot(0, this, [this, path]() {
                if (_menuController) {
                    _menuController->openVolpkgAt(path);
                }
            });
        } else if (!remoteUrls.empty() && !remoteUrls.at(0).isEmpty()) {
            // No local volpkg but have a recent remote URL — open it
            QString url = remoteUrls[0];
            QTimer::singleShot(0, this, [this, url]() {
                if (_menuController) {
                    _menuController->openRemoteUrl(url, false);
                }
            });
        }
    }

    // Create application-wide keyboard shortcuts
    fDrawingModeShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::DrawingMode), this);
    fDrawingModeShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDrawingModeShortcut, &QShortcut::activated, [this]() {
        if (_drawingWidget) {
            _drawingWidget->toggleDrawingMode();
        }
    });

    fCompositeViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CompositeView), this);
    fCompositeViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCompositeViewShortcut, &QShortcut::activated, [this]() {
        auto* viewer = segmentationViewer();
        if (!viewer) {
            return;
        }
        auto s = viewer->compositeRenderSettings();
        s.enabled = !s.enabled;
        viewer->setCompositeRenderSettings(s);
        QSignalBlocker blocker(ui.chkCompositeEnabled);
        ui.chkCompositeEnabled->setChecked(s.enabled);
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
            _viewerManager->forEachViewer([next](CTiledVolumeViewer* viewer) {
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
            _viewerManager->forEachViewer([next](CTiledVolumeViewer* viewer) {
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
        if (!mdiArea) return;
        if (auto* subWindow = mdiArea->activeSubWindow()) {
            if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(subWindow->widget())) {
                viewer->adjustZoomByFactor(ZOOM_FACTOR);
            }
        }
    });

    fZoomOutShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ZoomOut), this);
    fZoomOutShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomOutShortcut, &QShortcut::activated, [this]() {
        if (!mdiArea) return;
        if (auto* subWindow = mdiArea->activeSubWindow()) {
            if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(subWindow->widget())) {
                viewer->adjustZoomByFactor(1.0f / ZOOM_FACTOR);
            }
        }
    });

    // Reset view shortcut (m to fit surface in view and reset all offsets)
    fResetViewShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::ResetView), this);
    fResetViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fResetViewShortcut, &QShortcut::activated, [this]() {
        if (!mdiArea) return;
        if (auto* subWindow = mdiArea->activeSubWindow()) {
            if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(subWindow->widget())) {
                viewer->resetSurfaceOffsets();
                viewer->fitSurfaceInView();
                viewer->renderVisible(true);
            }
        }
    });

    // Z offset: Ctrl+. = +Z (further/deeper), Ctrl+, = -Z (closer)
    fWorldOffsetZPosShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::WorldOffsetZPos), this);
    fWorldOffsetZPosShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZPosShortcut, &QShortcut::activated, [this]() {
        if (!mdiArea) return;
        if (auto* subWindow = mdiArea->activeSubWindow()) {
            if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(subWindow->widget())) {
                viewer->adjustSurfaceOffset(1.0f);
            }
        }
    });

    fWorldOffsetZNegShortcut = new QShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::WorldOffsetZNeg), this);
    fWorldOffsetZNegShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZNegShortcut, &QShortcut::activated, [this]() {
        if (!mdiArea) return;
        if (auto* subWindow = mdiArea->activeSubWindow()) {
            if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(subWindow->widget())) {
                viewer->adjustSurfaceOffset(-1.0f);
            }
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

CTiledVolumeViewer *CWindow::newConnectedViewer(std::string surfaceName, QString title, QMdiArea *mdiArea)
{
    if (!_viewerManager) {
        return nullptr;
    }

    CTiledVolumeViewer* viewer = _viewerManager->createViewer(surfaceName, title, mdiArea);
    if (!viewer) {
        return nullptr;
    }

    configureViewerConnections(viewer);
    return viewer;
}

void CWindow::configureViewerConnections(CTiledVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    connect(_state, &CState::volumeChanged, viewer, &CTiledVolumeViewer::OnVolumeChanged, Qt::UniqueConnection);
    connect(_state, &CState::volumeClosing, viewer, &CTiledVolumeViewer::onVolumeClosing, Qt::UniqueConnection);
    connect(viewer, &CTiledVolumeViewer::sendVolumeClicked, this, &CWindow::onVolumeClicked, Qt::UniqueConnection);

    if (viewer->fGraphicsView) {
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMousePress,
                viewer, &CTiledVolumeViewer::onMousePress, Qt::UniqueConnection);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseMove,
                viewer, &CTiledVolumeViewer::onMouseMove, Qt::UniqueConnection);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseRelease,
                viewer, &CTiledVolumeViewer::onMouseRelease, Qt::UniqueConnection);
    }

    if (_drawingWidget && !viewer->property("vc_drawing_bound").toBool()) {
        connect(_drawingWidget, &DrawingWidget::sendPathsChanged,
                viewer, &CTiledVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendMousePressVolume,
                _drawingWidget, &DrawingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendMouseMoveVolume,
                _drawingWidget, &DrawingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendMouseReleaseVolume,
                _drawingWidget, &DrawingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendZSliceChanged,
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
                viewer, &CTiledVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendMousePressVolume,
                _seedingWidget, &SeedingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendMouseMoveVolume,
                _seedingWidget, &SeedingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendMouseReleaseVolume,
                _seedingWidget, &SeedingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendZSliceChanged,
                _seedingWidget, &SeedingWidget::updateCurrentZSlice, Qt::UniqueConnection);
        viewer->setProperty("vc_seeding_bound", true);
    }

    if (_point_collection_widget && !viewer->property("vc_points_bound").toBool()) {
        connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
                viewer, &CTiledVolumeViewer::onCollectionSelected, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::sendCollectionSelected,
                _point_collection_widget, &CPointCollectionWidget::selectCollection, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelected,
                viewer, &CTiledVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::pointSelected,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CTiledVolumeViewer::pointClicked,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        viewer->setProperty("vc_points_bound", true);
    }

    const std::string& surfName = viewer->surfName();
    if ((surfName == "seg xz" || surfName == "seg yz") && !viewer->property("vc_axisaligned_bound").toBool()) {
        if (viewer->fGraphicsView) {
            viewer->fGraphicsView->setMiddleButtonPanEnabled(!_axisAlignedSliceController->isEnabled());
        }

        connect(viewer, &CTiledVolumeViewer::sendMousePressVolume,
                this, [this, viewer](cv::Vec3f volLoc, cv::Vec3f /*normal*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMousePress(viewer, volLoc, button, modifiers);
                });

        connect(viewer, &CTiledVolumeViewer::sendMouseMoveVolume,
                this, [this, viewer](cv::Vec3f volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMouseMove(viewer, volLoc, buttons, modifiers);
                });

        connect(viewer, &CTiledVolumeViewer::sendMouseReleaseVolume,
                this, [this, viewer](cv::Vec3f /*volLoc*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    _axisAlignedSliceController->onMouseRelease(viewer, button, modifiers);
                });

        viewer->setProperty("vc_axisaligned_bound", true);
    }
}

CTiledVolumeViewer* CWindow::segmentationViewer() const
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
    clearTransformPreview(true);
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

    refreshTransformsPanelState();
}

std::shared_ptr<QuadSurface> CWindow::currentTransformSourceSurface() const
{
    if (!_state) {
        return nullptr;
    }

    if (auto active = _state->activeSurface().lock()) {
        return active;
    }

    if (_state->vpkg()) {
        const std::string activeId = _state->activeSurfaceId();
        if (!activeId.empty()) {
            if (auto surface = _state->vpkg()->getSurface(activeId)) {
                return surface;
            }
        }
    }

    auto segmentationSurface = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    if (segmentationSurface && segmentationSurface != _transformPreviewSurface) {
        return segmentationSurface;
    }

    return _transformPreviewSourceSurface;
}

QString CWindow::currentTransformSourceDescription() const
{
    if (!_customTransformSource.trimmed().isEmpty()) {
        return _customTransformSource.trimmed();
    }

    const auto currentVolume = _state ? _state->currentVolume() : nullptr;
    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    if (currentVolume && currentVolume->isRemote() && !remoteTransformUrl.empty()) {
        return QString::fromStdString(remoteTransformUrl);
    }

    if (currentVolume && !currentVolume->path().empty()) {
        return QString::fromStdString((currentVolume->path() / "transform.json").string());
    }

    const auto localPath = localCurrentTransformJsonPath();
    if (!localPath.empty()) {
        return QString::fromStdString(localPath.string());
    }

    return {};
}

std::filesystem::path CWindow::localCurrentTransformJsonPath() const
{
    if (!_customTransformSource.trimmed().isEmpty()) {
        return _customTransformLocalPath;
    }

    if (!_state) {
        return {};
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume) {
        return {};
    }

    const auto volumePath = currentVolume->path();
    if (volumePath.empty()) {
        return {};
    }

    const auto localTransformPath = volumePath / "transform.json";
    if (std::filesystem::exists(localTransformPath)) {
        return localTransformPath;
    }

    return {};
}

std::string CWindow::currentRemoteTransformJsonUrl() const
{
    if (!_state) {
        return {};
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume || !currentVolume->isRemote() || currentVolume->remoteUrl().empty()) {
        return {};
    }

    std::string remoteTransformUrl = currentVolume->remoteUrl();
    while (!remoteTransformUrl.empty() && remoteTransformUrl.back() == '/') {
        remoteTransformUrl.pop_back();
    }
    remoteTransformUrl += "/transform.json";
    return remoteTransformUrl;
}

void CWindow::ensureCurrentRemoteTransformJsonAsync()
{
    if (!_state) {
        return;
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume || !currentVolume->isRemote() || currentVolume->remoteUrl().empty()) {
        return;
    }

    const auto volumePath = currentVolume->path();
    if (volumePath.empty()) {
        return;
    }

    const auto localTransformPath = volumePath / "transform.json";
    if (std::filesystem::exists(localTransformPath)) {
        const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
        if (!remoteTransformUrl.empty()) {
            _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
        }
        return;
    }

    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    if (remoteTransformUrl.empty()) {
        return;
    }

    auto& fetchState = _remoteTransformFetchStates[remoteTransformUrl];
    if (fetchState == RemoteTransformFetchState::Available &&
        !std::filesystem::exists(localTransformPath)) {
        fetchState = RemoteTransformFetchState::Unknown;
    }
    if (fetchState != RemoteTransformFetchState::Unknown) {
        return;
    }

    fetchState = RemoteTransformFetchState::Pending;
    const auto auth = currentVolume->remoteAuth();
    auto* watcher = new QFutureWatcher<bool>(this);
    connect(watcher, &QFutureWatcher<bool>::finished, this,
            [this, watcher, remoteTransformUrl, localTransformPath]() {
                watcher->deleteLater();

                bool downloaded = false;
                try {
                    downloaded = watcher->result();
                } catch (const std::exception&) {
                    downloaded = false;
                }

                _remoteTransformFetchStates[remoteTransformUrl] =
                    (downloaded && std::filesystem::exists(localTransformPath))
                        ? RemoteTransformFetchState::Available
                        : RemoteTransformFetchState::Missing;

                if (currentRemoteTransformJsonUrl() == remoteTransformUrl) {
                    refreshTransformsPanelState();
                }
            });
    watcher->setFuture(QtConcurrent::run(
        [remoteTransformUrl, localTransformPath, auth]() {
            return vc::cache::httpDownloadFile(remoteTransformUrl, localTransformPath, auth);
        }));
}

bool CWindow::setCustomTransformSource(const QString& source, QString* errorMessage)
{
    const QString trimmed = source.trimmed();
    if (trimmed.isEmpty()) {
        _customTransformSource.clear();
        _customTransformLocalPath.clear();
        _customTransformTempDir.reset();
        return true;
    }

    try {
        std::filesystem::path resolvedPath;
        if (isRemoteTransformSource(trimmed)) {
            if (!_customTransformTempDir) {
                _customTransformTempDir = std::make_unique<QTemporaryDir>();
            }
            if (!_customTransformTempDir || !_customTransformTempDir->isValid()) {
                throw std::runtime_error("failed to create temporary directory for affine download");
            }

            const auto resolved = vc::resolveRemoteUrl(trimmed.toStdString());
            const auto auth = authForRemoteTransformSource(trimmed);
            const auto tempRoot = std::filesystem::path(_customTransformTempDir->path().toStdString());
            resolvedPath = tempRoot / "custom_transform.json";
            if (!vc::cache::httpDownloadFile(resolved.httpsUrl, resolvedPath, auth)) {
                throw std::runtime_error("failed to download affine from the provided path");
            }
        } else {
            resolvedPath = expandLocalTransformPath(trimmed);
        }

        loadAffineTransformMatrix(resolvedPath);
        _customTransformSource = trimmed;
        _customTransformLocalPath = std::move(resolvedPath);
        return true;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QString::fromUtf8(ex.what());
        }
        return false;
    }
}

std::filesystem::path CWindow::currentTransformJsonPath(bool allowRemoteFetch)
{
    if (!_customTransformSource.trimmed().isEmpty()) {
        return _customTransformLocalPath;
    }

    if (const auto localTransformPath = localCurrentTransformJsonPath();
        !localTransformPath.empty()) {
        const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
        if (!remoteTransformUrl.empty()) {
            _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
        }
        return localTransformPath;
    }

    if (!_state) {
        return {};
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume) {
        return {};
    }

    const auto volumePath = currentVolume->path();
    if (volumePath.empty()) {
        return {};
    }

    if (!currentVolume->isRemote() || currentVolume->remoteUrl().empty()) {
        return {};
    }

    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    const auto localTransformPath = volumePath / "transform.json";

    if (!allowRemoteFetch) {
        return {};
    }

    if (vc::cache::httpDownloadFile(remoteTransformUrl,
                                    localTransformPath,
                                    currentVolume->remoteAuth())) {
        _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
        return localTransformPath;
    }

    _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Missing;
    return {};
}

void CWindow::clearTransformPreview(bool restoreDisplayedSurface)
{
    if (!_state) {
        _transformPreviewSurface.reset();
        _transformPreviewSourceSurface.reset();
        return;
    }

    auto currentDisplayed = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    const bool showingPreview = (_transformPreviewSurface && currentDisplayed == _transformPreviewSurface);

    if (restoreDisplayedSurface && showingPreview) {
        auto restoreSurface = currentTransformSourceSurface();
        if (!restoreSurface) {
            restoreSurface = _transformPreviewSourceSurface;
        }
        _state->setSurface("segmentation", restoreSurface, false, false);
        if (_axisAlignedSliceController) {
            _axisAlignedSliceController->applyOrientation(restoreSurface.get());
        }
    }

    _transformPreviewSurface.reset();
    _transformPreviewSourceSurface.reset();
}

bool CWindow::applyTransformPreview(bool allowRemoteFetch)
{
    if (!_state || (_segmentationModule && _segmentationModule->editingEnabled())) {
        return false;
    }

    auto sourceSurface = currentTransformSourceSurface();
    if (!sourceSurface) {
        return false;
    }

    const int scale = _transformScaleSpin ? _transformScaleSpin->value() : 1;
    const bool scaleOnly = _scaleOnlyTransformCheck && _scaleOnlyTransformCheck->isChecked();
    std::optional<cv::Matx44d> matrix;
    if (!scaleOnly) {
        const auto transformPath = currentTransformJsonPath(allowRemoteFetch);
        if (!transformPath.empty() && std::filesystem::exists(transformPath)) {
            matrix = loadAffineTransformMatrix(transformPath);
            if (_invertTransformCheck && _invertTransformCheck->isChecked()) {
                matrix = invertAffineTransformMatrix(*matrix);
            }
        } else if (scale == 1) {
            return false;
        }
    } else if (scale == 1) {
        return false;
    }

    auto previewSurface = cloneSurfaceForTransform(sourceSurface);
    if (!previewSurface) {
        return false;
    }

    previewSurface->path.clear();
    previewSurface->id.clear();

    transformSurfacePoints(previewSurface.get(), scale, matrix);
    refreshTransformedSurfaceState(previewSurface.get());
    if (_viewerManager) {
        _viewerManager->refreshSurfacePatchIndex(previewSurface);
    }

    clearTransformPreview(false);
    _transformPreviewSourceSurface = sourceSurface;
    _transformPreviewSurface = previewSurface;
    _state->setSurface("segmentation", previewSurface, false, false);
    if (_axisAlignedSliceController) {
        _axisAlignedSliceController->applyOrientation(previewSurface.get());
    }
    return true;
}

void CWindow::refreshTransformsPanelState()
{
    if (!_previewTransformCheck || !_scaleOnlyTransformCheck || !_invertTransformCheck || !_transformScaleSpin ||
        !_loadAffineButton || !_saveTransformedButton || !_transformStatusLabel) {
        return;
    }

    const bool editingEnabled = _segmentationModule && _segmentationModule->editingEnabled();
    const auto sourceSurface = currentTransformSourceSurface();
    const auto currentVolume = _state ? _state->currentVolume() : nullptr;
    const bool scaleOnly = _scaleOnlyTransformCheck->isChecked();
    const auto transformPath = localCurrentTransformJsonPath();
    const bool hasTransform = !transformPath.empty() && std::filesystem::exists(transformPath);
    const int scale = _transformScaleSpin->value();
    const bool hasScaleOnlyTransform = scale != 1;
    const bool previewEnabled =
        sourceSurface && !editingEnabled &&
        (hasScaleOnlyTransform || (!scaleOnly && hasTransform));
    const bool saveEnabled = previewEnabled && sourceSurface && !sourceSurface->path.empty();
    const bool hasCustomTransform = !_customTransformSource.trimmed().isEmpty();
    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    RemoteTransformFetchState remoteFetchState = RemoteTransformFetchState::Unknown;
    if (!scaleOnly && !hasCustomTransform && currentVolume && currentVolume->isRemote() &&
        !remoteTransformUrl.empty()) {
        if (hasTransform) {
            _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
        } else {
            ensureCurrentRemoteTransformJsonAsync();
            auto it = _remoteTransformFetchStates.find(remoteTransformUrl);
            if (it != _remoteTransformFetchStates.end()) {
                remoteFetchState = it->second;
            }
        }
    }

    const QString transformLocation = currentTransformSourceDescription();

    QString statusText;
    if (!_state || !_state->vpkg()) {
        statusText = tr("Open a volume package to use transforms.");
    } else if (!_state->currentVolume()) {
        statusText = tr("Select a volume to load transform.json.");
    } else if (!sourceSurface) {
        statusText = tr("Select a segmentation to preview or save its transform.");
    } else if (editingEnabled) {
        statusText = tr("Transform preview is unavailable while segmentation editing is enabled.");
    } else if (scaleOnly) {
        if (hasScaleOnlyTransform) {
            statusText = hasTransform
                ? tr("Scaling points by %1 only. Affine from %2 is ignored.")
                      .arg(scale)
                      .arg(transformLocation)
                : tr("Scaling points by %1 only. No affine will be applied.")
                      .arg(scale);
        } else {
            statusText = hasTransform
                ? tr("Scale only is enabled. Affine from %1 will be ignored until scale is greater than 1.")
                      .arg(transformLocation)
                : tr("Scale only is enabled. Increase scale above 1 to preview or save.");
        }
    } else if (!hasTransform && remoteFetchState == RemoteTransformFetchState::Pending) {
        if (hasScaleOnlyTransform) {
            statusText = tr("Scaling points by %1 while checking %2 for transform.json.")
                .arg(scale)
                .arg(transformLocation);
        } else {
            statusText = tr("Checking %1 for transform.json.")
                .arg(transformLocation);
        }
    } else if (!hasTransform) {
        if (hasScaleOnlyTransform) {
            statusText = hasCustomTransform
                ? tr("Scaling points by %1. No affine was loaded from %2.")
                      .arg(scale)
                      .arg(transformLocation)
                : tr("Scaling points by %1. No affine transform was found at %2.")
                      .arg(scale)
                      .arg(transformLocation);
        } else {
            statusText = hasCustomTransform
                ? tr("No affine was loaded from %1")
                      .arg(transformLocation)
                : tr("No transform.json found at %1")
                      .arg(transformLocation);
        }
    } else {
        statusText = hasCustomTransform
            ? tr("Using custom affine %1%2 with scale %3")
                  .arg(transformLocation,
                       (_invertTransformCheck->isChecked() ? tr(" (inverted)") : QString()))
                  .arg(scale)
            : tr("Using %1%2 with scale %3")
                  .arg(transformLocation,
                       (_invertTransformCheck->isChecked() ? tr(" (inverted)") : QString()))
                  .arg(scale);
    }

    if (!previewEnabled && _previewTransformCheck->isChecked()) {
        const QSignalBlocker blocker(_previewTransformCheck);
        _previewTransformCheck->setChecked(false);
        clearTransformPreview(true);
    } else if (previewEnabled && _previewTransformCheck->isChecked()) {
        try {
            if (!applyTransformPreview(false)) {
                clearTransformPreview(true);
            }
        } catch (const std::exception& ex) {
            clearTransformPreview(true);
            const QSignalBlocker blocker(_previewTransformCheck);
            _previewTransformCheck->setChecked(false);
            statusText = tr("Failed to load transform.json: %1")
                .arg(QString::fromUtf8(ex.what()));
        }
    } else if (!_previewTransformCheck->isChecked()) {
        clearTransformPreview(true);
    }

    _previewTransformCheck->setEnabled(previewEnabled);
    _scaleOnlyTransformCheck->setEnabled(sourceSurface && !editingEnabled);
    _invertTransformCheck->setEnabled(!editingEnabled && hasTransform && !scaleOnly);
    _transformScaleSpin->setEnabled(sourceSurface && !editingEnabled);
    _loadAffineButton->setEnabled(!editingEnabled);
    _saveTransformedButton->setEnabled(saveEnabled);
    _transformStatusLabel->setText(statusText);
}

void CWindow::onPreviewTransformToggled(bool enabled)
{
    if (!_previewTransformCheck) {
        return;
    }

    if (!enabled) {
        clearTransformPreview(true);
        refreshTransformsPanelState();
        return;
    }

    try {
        if (!applyTransformPreview()) {
            throw std::runtime_error("transform preview is unavailable for the current selection");
        }
    } catch (const std::exception& ex) {
        clearTransformPreview(true);
        {
            const QSignalBlocker blocker(_previewTransformCheck);
            _previewTransformCheck->setChecked(false);
        }
        statusBar()->showMessage(tr("Failed to preview transform: %1")
                                     .arg(QString::fromUtf8(ex.what())),
                                 5000);
    }

    refreshTransformsPanelState();
}

void CWindow::onSaveTransformedRequested()
{
    if (!_state || !_state->vpkg()) {
        return;
    }

    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        QMessageBox::warning(this, tr("Editing Active"),
                             tr("Disable segmentation editing before saving a transformed surface."));
        return;
    }

    auto sourceSurface = currentTransformSourceSurface();
    if (!sourceSurface || sourceSurface->path.empty()) {
        QMessageBox::warning(this, tr("No Segmentation"),
                             tr("Select a segmentation with files on disk first."));
        return;
    }

    const int scale = _transformScaleSpin ? _transformScaleSpin->value() : 1;
    const bool scaleOnly = _scaleOnlyTransformCheck && _scaleOnlyTransformCheck->isChecked();
    const auto transformPath = scaleOnly ? std::filesystem::path{} : currentTransformJsonPath();
    const bool hasTransform = !transformPath.empty() && std::filesystem::exists(transformPath);
    if (!hasTransform && scale == 1) {
        QMessageBox::warning(this, tr("Missing Transform"),
                             scaleOnly
                                 ? tr("Scale only is enabled, and scale is set to 1.")
                                 : (_customTransformSource.trimmed().isEmpty()
                                        ? tr("No transform.json was found for the current volume, and scale is set to 1.")
                                        : tr("The selected affine could not be loaded, and scale is set to 1.")));
        return;
    }

    const QString defaultName = QString::fromStdString(sourceSurface->id.empty()
        ? sourceSurface->path.filename().string() + "_transformed"
        : sourceSurface->id + "_transformed");
    bool ok = false;
    QString newName = QInputDialog::getText(this,
                                            tr("Save Transformed"),
                                            tr("New surface name:"),
                                            QLineEdit::Normal,
                                            defaultName,
                                            &ok).trimmed();
    if (!ok || newName.isEmpty()) {
        return;
    }

    static const QRegularExpression validNameRegex(QStringLiteral("^[a-zA-Z0-9_-]+$"));
    if (!validNameRegex.match(newName).hasMatch()) {
        QMessageBox::warning(this, tr("Invalid Name"),
                             tr("Surface name can only contain letters, numbers, underscores, and hyphens."));
        return;
    }

    const std::string newId = newName.toStdString();
    const std::filesystem::path sourcePath = sourceSurface->path;
    const std::filesystem::path parentDir = sourcePath.parent_path();
    const std::filesystem::path targetPath = parentDir / newId;
    if (std::filesystem::exists(targetPath)) {
        QMessageBox::warning(this, tr("Name Exists"),
                             tr("A surface with the name '%1' already exists.").arg(newName));
        return;
    }

    QTemporaryDir stagingRoot;
    if (!stagingRoot.isValid()) {
        QMessageBox::critical(this, tr("Temporary Directory Error"),
                              tr("Failed to create a temporary staging directory."));
        return;
    }

    try {
        std::optional<cv::Matx44d> matrix;
        if (hasTransform) {
            matrix = loadAffineTransformMatrix(transformPath);
            if (_invertTransformCheck && _invertTransformCheck->isChecked()) {
                matrix = invertAffineTransformMatrix(*matrix);
            }
        }
        auto transformedSurface = cloneSurfaceForTransform(sourceSurface);
        if (!transformedSurface) {
            throw std::runtime_error("failed to clone source surface");
        }

        transformSurfacePoints(transformedSurface.get(), scale, matrix);
        refreshTransformedSurfaceState(transformedSurface.get());

        const std::filesystem::path stagingPath =
            std::filesystem::path(stagingRoot.path().toStdString()) / newId;
        transformedSurface->save(stagingPath.string(), newId, false);

        std::filesystem::copy(sourcePath,
                              targetPath,
                              std::filesystem::copy_options::recursive);
        std::filesystem::copy(stagingPath,
                              targetPath,
                              std::filesystem::copy_options::recursive |
                                  std::filesystem::copy_options::overwrite_existing);
    } catch (const std::exception& ex) {
        std::error_code cleanupError;
        std::filesystem::remove_all(targetPath, cleanupError);
        QMessageBox::critical(this, tr("Save Failed"),
                              tr("Failed to save transformed surface: %1")
                                  .arg(QString::fromUtf8(ex.what())));
        return;
    }

    if (_state->vpkg()->addSingleSegmentation(newId)) {
        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(newId);
        }
    } else {
        _state->vpkg()->refreshSegmentations();
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
    }

    statusBar()->showMessage(tr("Saved transformed surface as '%1'.").arg(newName), 5000);
    refreshTransformsPanelState();
}

void CWindow::onLoadAffineRequested()
{
    const QString promptText = tr("Enter a local path or URL for an affine JSON (http://, https://, s3://).\n"
                                  "Leave blank to use the current volume transform.json.");
    bool accepted = false;
    const QString source = QInputDialog::getText(this,
                                                 tr("Load Affine"),
                                                 promptText,
                                                 QLineEdit::Normal,
                                                 _customTransformSource,
                                                 &accepted).trimmed();
    if (!accepted) {
        return;
    }

    QString errorMessage;
    if (!setCustomTransformSource(source, &errorMessage)) {
        QMessageBox::warning(this,
                             tr("Load Affine Failed"),
                             tr("Failed to load affine from %1: %2")
                                 .arg(source.isEmpty() ? tr("(empty path)") : source, errorMessage));
        return;
    }

    if (source.isEmpty() && _previewTransformCheck && _previewTransformCheck->isChecked()) {
        currentTransformJsonPath();
    }

    if (source.isEmpty()) {
        statusBar()->showMessage(tr("Using the current volume transform.json."), 5000);
    } else {
        statusBar()->showMessage(tr("Loaded affine from %1").arg(source), 5000);
    }

    refreshTransformsPanelState();
}

void CWindow::setVolume(std::shared_ptr<Volume> newvol)
{
    const bool hadVolume = static_cast<bool>(_state->currentVolume());
    POI* existingFocusPoi = _state ? _state->poi("focus") : nullptr;

    // CState handles cache budget and volume ID resolution, and emits volumeChanged
    _state->setCurrentVolume(newvol);

    if (newvol) {
        primeRemoteLevel5WithDialog(this, newvol);
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

    if (_state->currentVolume() && _state) {
        auto [w, h, d] = _state->currentVolume()->shape();
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

    onManualPlaneChanged();
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

void CWindow::setRemoteSurfaces(const std::vector<std::pair<std::string, std::shared_ptr<Surface>>>& surfaces)
{
    if (surfaces.empty()) return;

    if (_surfacePanel) {
        _surfacePanel->loadRemoteSurfaces(surfaces);
    }

    // Set the first surface as the active segmentation
    if (_state) {
        const auto& [firstId, firstSurf] = surfaces.front();
        _state->setSurface("segmentation", firstSurf);
        _state->setActiveSurface(firstId, std::dynamic_pointer_cast<QuadSurface>(firstSurf));
        _state->emitSurfacesChanged();
    }

    emit _state->surfacesLoaded();
    refreshTransformsPanelState();
}

void CWindow::setRemoteStubs(
    const std::vector<std::string>& segmentIds,
    const std::vector<std::pair<std::string, std::shared_ptr<Surface>>>& cachedSurfaces)
{
    if (_surfacePanel) {
        _surfacePanel->loadRemoteStubs(segmentIds, cachedSurfaces);
    }

    // Activate the first cached surface if available
    if (_state && !cachedSurfaces.empty()) {
        const auto& [firstId, firstSurf] = cachedSurfaces.front();
        _state->setSurface("segmentation", firstSurf);
        _state->setActiveSurface(firstId, std::dynamic_pointer_cast<QuadSurface>(firstSurf));
        _state->emitSurfacesChanged();
    }

    emit _state->surfacesLoaded();
    refreshTransformsPanelState();
}

void CWindow::downloadRemoteSegmentOnDemand(const QString& segmentId)
{
    const std::string segId = segmentId.toStdString();
    const std::string dlBase = (_remoteScroll.segSource == vc::RemoteSegmentSource::Direct)
        ? _remoteScroll.segmentsBaseUrl : _remoteScroll.baseUrl;
    const std::string cachePath = _remoteScroll.cachePath;
    const auto auth = _remoteScroll.auth;
    const auto segSource = _remoteScroll.segSource;

    if (statusBar()) {
        statusBar()->showMessage(
            tr("Downloading segment %1...").arg(segmentId));
    }

    auto* watcher = new QFutureWatcher<std::shared_ptr<QuadSurface>>(this);
    // Capture the current session URL so the completion handler can detect
    // a stale result if the user closed/switched volumes during download.
    const std::string expectedBaseUrl = _remoteScroll.baseUrl;
    connect(watcher, &QFutureWatcher<std::shared_ptr<QuadSurface>>::finished, this,
        [this, watcher, segId, expectedBaseUrl]() {
            watcher->deleteLater();
            // If the remote scroll session changed, silently discard the
            // result — applying a segment from a previous dataset to the
            // current CState would corrupt surface state.
            if (_remoteScroll.baseUrl != expectedBaseUrl) {
                return;
            }
            std::shared_ptr<QuadSurface> surf;
            try {
                surf = watcher->result();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[RemoteScroll] Download failed for %s: %s\n",
                    segId.c_str(), e.what());
            }

            if (surf) {
                if (statusBar()) {
                    statusBar()->showMessage(
                        tr("Downloaded segment %1").arg(QString::fromStdString(segId)), 3000);
                }
                if (_surfacePanel) {
                    _surfacePanel->replaceStubWithSurface(segId, surf);
                }
            } else {
                if (statusBar()) {
                    statusBar()->showMessage(
                        tr("Failed to download segment %1").arg(QString::fromStdString(segId)), 5000);
                }
                // Reset the stub state so user can retry
                if (_surfacePanel) {
                    _surfacePanel->replaceStubWithSurface(segId, nullptr);
                }
            }
        });

    const std::string baseUrl = _remoteScroll.baseUrl;

    auto future = QtConcurrent::run(
        [dlBase, baseUrl, segId, cachePath, auth, segSource]() -> std::shared_ptr<QuadSurface> {
            // Cache-root layout MUST match MenuActionController::promptAndLoadRemoteSegments
            // so previously-preloaded segments are reused on demand instead
            // of re-downloaded into a second location.
            //   Direct sources: flat → cachePath/paths/<segId>
            //   Segments/Paths (full volpkg): nested → cachePath/<volpkgName>/{paths|segments}/<segId>
            std::filesystem::path segmentRoot = cachePath;
            if (segSource != vc::RemoteSegmentSource::Direct) {
                std::string volpkgName = baseUrl;
                while (!volpkgName.empty() && volpkgName.back() == '/') volpkgName.pop_back();
                auto slash = volpkgName.rfind('/');
                if (slash != std::string::npos) volpkgName = volpkgName.substr(slash + 1);
                segmentRoot = std::filesystem::path(cachePath) / volpkgName;
            }

            auto localDir = vc::downloadRemoteSegment(
                dlBase, segId, segmentRoot, auth, segSource);

            if (!std::filesystem::exists(localDir / "meta.json")) {
                return nullptr;
            }

            auto seg = Segmentation::New(localDir);
            if (seg && seg->canLoadSurface()) {
                return seg->loadSurface();
            }
            return nullptr;
        });
    watcher->setFuture(future);
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

    refreshTransformsPanelState();
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

    _state->setPOI("focus", focus);

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

    _viewerManager->forEachViewer([&position](CTiledVolumeViewer* viewer) {
        if (!viewer) {
            return;
        }

        const std::string name = viewer->surfName();
        if (name == "xy plane" || name == "seg xz" || name == "seg yz") {
            viewer->centerOnVolumePoint(position, true);
        }
    });
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
    _viewerManager->forEachViewer([&position](CTiledVolumeViewer* viewer) {
        if (viewer) {
            viewer->centerOnVolumePoint(position, true);
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
    auto tryCenterFromViewer = [&](CTiledVolumeViewer* viewer) -> bool {
        if (!viewer || !viewer->isVisible()) {
            return false;
        }

        auto* gv = viewer->fGraphicsView;
        auto* viewport = gv ? gv->viewport() : nullptr;
        if (!viewport) {
            return false;
        }

        const QPoint viewportPos = viewport->mapFromGlobal(globalPos);
        if (!viewport->rect().contains(viewportPos)) {
            return false;
        }

        cv::Vec3f p, n;
        const QPointF scenePos = gv->mapToScene(viewportPos);
        if (!viewer->sceneToVolumePN(p, n, scenePos)) {
            return false;
        }

        return centerFocusAt(p, n, viewer->surfName());
    };

    // Prefer the viewer actually under the mouse cursor. With tiled MDI
    // windows, the active subwindow can lag behind the hovered viewer, which
    // makes the focus jump use the wrong scene transform.
    if (QWidget* hoveredWidget = QApplication::widgetAt(globalPos)) {
        for (QWidget* widget = hoveredWidget; widget; widget = widget->parentWidget()) {
            if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(widget)) {
                if (tryCenterFromViewer(viewer)) {
                    return true;
                }
                break;
            }
        }
    }

    if (_viewerManager) {
        for (auto* viewer : _viewerManager->viewers()) {
            if (tryCenterFromViewer(viewer)) {
                return true;
            }
        }
    }

    // Fall back to the active viewer if the cursor isn't currently over any
    // tiled viewport.
    if (auto* subWindow = mdiArea->activeSubWindow()) {
        if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(subWindow->widget())) {
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
            if (auto* viewer = qobject_cast<CTiledVolumeViewer*>(subWindow->widget())) {
                viewer->fGraphicsView->setFocus();
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
        refreshTransformsPanelState();
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
                const QString path = QString::fromStdString(surf->path.string());
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
    connect(_surfacePanel.get(), &SurfacePanelController::addOverlapRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onAddOverlap(segmentId.toStdString());
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
    connect(_surfacePanel.get(), &SurfacePanelController::awsUploadRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onAWSUpload(segmentId.toStdString());
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
    connect(_surfacePanel.get(), &SurfacePanelController::fetchRemoteChunksRequested,
            this, [this](const QString& segmentId) {
                _segmentationCommandHandler->onFetchRemoteChunks(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::remoteSegmentDownloadRequested,
            this, &CWindow::downloadRemoteSegmentOnDemand);

    connect(_surfacePanel.get(), &SurfacePanelController::growSeedsRequested,
            this, [this](const QString& segmentId, bool isExpand, bool isRandomSeed) {
                _segmentationCommandHandler->onGrowSeeds(segmentId.toStdString(), isExpand, isRandomSeed);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::teleaInpaintRequested,
            this, [this]() {
                if (_menuController) {
                    _menuController->triggerTeleaInpaint();
                }
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

    // i recognize that having both a seeding widget and a drawing widget that both handle mouse events and paths is redundant,
    // but i can't find an easy way yet to merge them and maintain the path iteration that the seeding widget currently uses
    // so for now we have both. i suppose i could probably add a 'mode' , but for now i will just hate this section :(

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


    _segmentationEdit = std::make_unique<SegmentationEditManager>(this);
    _segmentationEdit->setViewerManager(_viewerManager.get());
    _segmentationOverlay = std::make_unique<SegmentationOverlayController>(_state, this);
    _segmentationOverlay->setEditManager(_segmentationEdit.get());
    _segmentationOverlay->setViewerManager(_viewerManager.get());

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
            [overlayPtr](CTiledVolumeViewer* viewer, const cv::Vec3f& worldPos) {
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

    // Create Drawing widget
    _drawingWidget = new DrawingWidget();
    _drawingWidget->setState(_state);
    attachScrollAreaToDock(ui.dockWidgetDrawing, _drawingWidget, QStringLiteral("dockWidgetDrawingContent"));

    connect(_state, &CState::volumeChanged, _drawingWidget,
            static_cast<void (DrawingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&DrawingWidget::onVolumeChanged));
    connect(_drawingWidget, &DrawingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(_state, &CState::surfacesLoaded, _drawingWidget, &DrawingWidget::onSurfacesLoaded);

    // Cache is now obtained from volume->tieredCache()

    // Create Seeding widget
    _seedingWidget = new SeedingWidget(_state->pointCollection(), _state);
    attachScrollAreaToDock(ui.dockWidgetDistanceTransform, _seedingWidget, QStringLiteral("dockWidgetDistanceTransformContent"));

    _seedingWidget->setState(_state);
    connect(_state, &CState::volumeChanged, _seedingWidget,
            static_cast<void (SeedingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&SeedingWidget::onVolumeChanged));
    connect(_state, &CState::volumeChanged, this,
            [this](std::shared_ptr<Volume>, const std::string&) { refreshTransformsPanelState(); });
    connect(_seedingWidget, &SeedingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(_state, &CState::surfacesLoaded, _seedingWidget, &SeedingWidget::onSurfacesLoaded);

    // Cache is now obtained from volume->tieredCache()

    // Create and add the point collection widget
    _point_collection_widget = new CPointCollectionWidget(_state->pointCollection(), this);
    _point_collection_widget->setObjectName("pointCollectionDock");
    addDockWidget(Qt::RightDockWidgetArea, _point_collection_widget);

    // Selection dock (removed per request; selection actions remain in the menu)
    if (_viewerManager) {
        _viewerManager->forEachViewer([this](CTiledVolumeViewer* viewer) {
            configureViewerConnections(viewer);
        });
    }
    connect(_point_collection_widget, &CPointCollectionWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);
    connect(_point_collection_widget, &CPointCollectionWidget::convertPointToAnchorRequested, this, &CWindow::onConvertPointToAnchor);
    connect(_point_collection_widget, &CPointCollectionWidget::focusViewsRequested, this, &CWindow::onFocusViewsRequested);

    // Tab the docks - keep Segmentation, Seeding, Point Collections, and Drawing together
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDistanceTransform);
    tabifyDockWidget(ui.dockWidgetSegmentation, _point_collection_widget);
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDrawing);

    // Make Drawing dock the active tab by default
    ui.dockWidgetDrawing->raise();

    // Build Viewer Controls dock from the existing view-related panels.
    auto* viewerControlsLayout = qobject_cast<QVBoxLayout*>(ui.dockWidgetViewerControlsContents->layout());
    if (!viewerControlsLayout) {
        viewerControlsLayout = new QVBoxLayout(ui.dockWidgetViewerControlsContents);
        viewerControlsLayout->setContentsMargins(4, 4, 4, 4);
        viewerControlsLayout->setSpacing(8);
    }

    auto detachScrollContents = [](QScrollArea* scrollArea, QWidget* contents) -> QWidget* {
        if (!contents) {
            return nullptr;
        }
        if (scrollArea && scrollArea->widget() == contents) {
            scrollArea->takeWidget();
        }
        contents->setParent(nullptr);
        return contents;
    };

    auto moveGridLayoutItems = [](QGridLayout* from, QGridLayout* to, QWidget* newParent) {
        if (!from || !to) {
            return;
        }
        to->setContentsMargins(from->contentsMargins());
        to->setHorizontalSpacing(from->horizontalSpacing());
        to->setVerticalSpacing(from->verticalSpacing());
        for (int column = 0; column < from->columnCount(); ++column) {
            to->setColumnStretch(column, from->columnStretch(column));
            to->setColumnMinimumWidth(column, from->columnMinimumWidth(column));
        }
        for (int row = 0; row < from->rowCount(); ++row) {
            to->setRowStretch(row, from->rowStretch(row));
            to->setRowMinimumHeight(row, from->rowMinimumHeight(row));
        }
        for (int index = from->count() - 1; index >= 0; --index) {
            int row = 0;
            int column = 0;
            int rowSpan = 1;
            int columnSpan = 1;
            from->getItemPosition(index, &row, &column, &rowSpan, &columnSpan);
            if (auto* item = from->takeAt(index)) {
                if (newParent) {
                    if (auto* widget = item->widget()) {
                        widget->setParent(newParent);
                    } else if (auto* layout = item->layout()) {
                        layout->setParent(newParent);
                    }
                }
                to->addItem(item, row, column, rowSpan, columnSpan, item->alignment());
            }
        }
    };

    auto* normalVisContainer = new QWidget(ui.dockWidgetViewerControlsContents);
    auto* normalVisLayout = new QGridLayout(normalVisContainer);
    moveGridLayoutItems(qobject_cast<QGridLayout*>(ui.dockWidgetNormalVisContents->layout()),
                        normalVisLayout,
                        normalVisContainer);

    auto* transformsContainer = new QWidget(ui.dockWidgetViewerControlsContents);
    auto* transformsLayout = new QVBoxLayout(transformsContainer);
    transformsLayout->setContentsMargins(0, 0, 0, 0);
    transformsLayout->setSpacing(8);

    _previewTransformCheck = new QCheckBox(tr("Preview Result"), transformsContainer);
    _previewTransformCheck->setToolTip(
        tr("Preview the scaled and/or affine-transformed segmentation."));
    transformsLayout->addWidget(_previewTransformCheck);

    _scaleOnlyTransformCheck = new QCheckBox(tr("Scale Only"), transformsContainer);
    _scaleOnlyTransformCheck->setToolTip(
        tr("Ignore any loaded or volume affine and apply scale only."));
    transformsLayout->addWidget(_scaleOnlyTransformCheck);

    _invertTransformCheck = new QCheckBox(tr("Invert Affine"), transformsContainer);
    _invertTransformCheck->setToolTip(
        tr("Invert the loaded affine. Ignored when no affine transform is loaded."));
    transformsLayout->addWidget(_invertTransformCheck);

    auto* transformScaleRow = new QHBoxLayout();
    transformScaleRow->setContentsMargins(0, 0, 0, 0);
    transformScaleRow->setSpacing(8);
    transformScaleRow->addWidget(new QLabel(tr("Scale"), transformsContainer));
    _transformScaleSpin = new QSpinBox(transformsContainer);
    _transformScaleSpin->setMinimum(1);
    _transformScaleSpin->setMaximum(1000);
    _transformScaleSpin->setValue(1);
    _transformScaleSpin->setToolTip(
        tr("Multiply segmentation points by this integer. Works with or without an affine transform."));
    transformScaleRow->addWidget(_transformScaleSpin);
    transformsLayout->addLayout(transformScaleRow);

    _loadAffineButton = new QPushButton(tr("Load Affine (Optional)"), transformsContainer);
    _loadAffineButton->setToolTip(
        tr("Load an affine JSON from a local path or URL. Leave the dialog blank to return to the current volume transform."));
    transformsLayout->addWidget(_loadAffineButton);

    _saveTransformedButton = new QPushButton(tr("Save Transformed"), transformsContainer);
    _saveTransformedButton->setToolTip(
        tr("Save a new surface using the current scale and optional affine transform."));
    transformsLayout->addWidget(_saveTransformedButton);

    _transformStatusLabel = new QLabel(transformsContainer);
    _transformStatusLabel->setWordWrap(true);
    _transformStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    transformsLayout->addWidget(_transformStatusLabel);

    connect(_previewTransformCheck, &QCheckBox::toggled,
            this, &CWindow::onPreviewTransformToggled);
    connect(_scaleOnlyTransformCheck, &QCheckBox::toggled,
            this, [this](bool) { refreshTransformsPanelState(); });
    connect(_invertTransformCheck, &QCheckBox::toggled,
            this, [this](bool) { refreshTransformsPanelState(); });
    connect(_transformScaleSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int) { refreshTransformsPanelState(); });
    connect(_loadAffineButton, &QPushButton::clicked,
            this, &CWindow::onLoadAffineRequested);
    connect(_saveTransformedButton, &QPushButton::clicked,
            this, &CWindow::onSaveTransformedRequested);

    auto rememberGroupState = [this](CollapsibleSettingsGroup* group, const char* key) {
        if (!group) {
            return;
        }
        connect(group, &CollapsibleSettingsGroup::toggled, this, [key](bool expanded) {
            QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
            settings.setValue(key, expanded);
        });
    };

    auto addViewerGroup = [this, &settings, viewerControlsLayout, &rememberGroupState](
                              const QString& title, QWidget* contents, const char* key, bool defaultExpanded) {
        if (!viewerControlsLayout || !contents) {
            return static_cast<CollapsibleSettingsGroup*>(nullptr);
        }
        auto* group = new CollapsibleSettingsGroup(title, ui.dockWidgetViewerControlsContents);
        group->contentLayout()->addWidget(contents);
        viewerControlsLayout->addWidget(group);
        group->setExpanded(settings.value(key, defaultExpanded).toBool());
        rememberGroupState(group, key);
        return group;
    };

    using namespace vc3d::settings;
    auto* viewGroup = addViewerGroup(tr("View"),
                   detachScrollContents(ui.scrollAreaView, ui.dockWidgetViewContents),
                   viewer::GROUP_VIEW_EXPANDED,
                   viewer::GROUP_VIEW_EXPANDED_DEFAULT);

    // Interpolation + highlight-downscaled live inside the View group so
    // all display-affecting toggles sit together.
    auto* viewExtrasLayout = viewGroup ? viewGroup->contentLayout() : viewerControlsLayout;

    // Interpolation method selector
    {
        auto* interpWidget = new QWidget;
        auto* interpLayout = new QHBoxLayout(interpWidget);
        interpLayout->setContentsMargins(2, 2, 2, 2);
        interpLayout->addWidget(new QLabel(tr("Interpolation")));
        auto* cmbInterp = new QComboBox;
        cmbInterp->addItem(tr("Nearest"));
        cmbInterp->addItem(tr("Trilinear"));
        cmbInterp->addItem(tr("Tricubic"));
        cmbInterp->addItem(tr("Lanczos"));
        cmbInterp->setCurrentIndex(settings.value(perf::INTERPOLATION_METHOD, 1).toInt());
        interpLayout->addWidget(cmbInterp);
        connect(cmbInterp, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int idx) {
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(vc3d::settings::perf::INTERPOLATION_METHOD, idx);
            _viewerManager->forEachViewer([](CTiledVolumeViewer* v) {
                v->reloadPerfSettings();
                v->update();
            });
        });
        viewExtrasLayout->addWidget(interpWidget);
    }

    // Highlight downscaled chunks — tints pixels that rendered against a
    // coarser pyramid level than the zoom-level target (green → red).
    {
        auto* chkHighlight = new QCheckBox(tr("Highlight downscaled chunks"));
        chkHighlight->setToolTip(
            tr("Tint pixels sourced from a coarser pyramid level than the current zoom "
               "target. Green = 1 level coarser; red = 5+ levels coarser. Untinted pixels "
               "rendered at the requested resolution."));
        chkHighlight->setChecked(
            settings.value("viewer_controls/highlight_downscaled", false).toBool());
        connect(chkHighlight, &QCheckBox::toggled, this, [this](bool on) {
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue("viewer_controls/highlight_downscaled", on);
            if (_viewerManager) {
                _viewerManager->forEachViewer([](CTiledVolumeViewer* v) {
                    v->reloadPerfSettings();
                    v->renderVisible(true);
                });
            }
        });
        viewExtrasLayout->addWidget(chkHighlight);
    }

    // Navigation sensitivity controls
    {
        auto* navWidget = new QWidget;
        auto* navLayout = new QGridLayout(navWidget);
        navLayout->setContentsMargins(2, 2, 2, 2);
        navLayout->setVerticalSpacing(2);

        auto addSpin = [&](int row, const QString& label, const char* settingsKey, float defaultVal) {
            navLayout->addWidget(new QLabel(label), row, 0);
            auto* spin = new QDoubleSpinBox;
            spin->setRange(0.1, 100.0);
            spin->setSingleStep(0.1);
            spin->setDecimals(1);
            spin->setValue(settings.value(settingsKey, defaultVal).toDouble());
            navLayout->addWidget(spin, row, 1);
            connect(spin, &QDoubleSpinBox::valueChanged, this, [this, settingsKey](double v) {
                QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
                s.setValue(settingsKey, v);
                if (_viewerManager) {
                    _viewerManager->forEachViewer([](CTiledVolumeViewer* v) {
                        v->reloadPerfSettings();
                    });
                }
            });
        };
        addSpin(0, tr("Pan sensitivity"), viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT);
        addSpin(1, tr("Zoom sensitivity"), viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT);
        addSpin(2, tr("Z-scroll sensitivity"), viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT);

        addViewerGroup(tr("Navigation"),
                       navWidget,
                       "viewer_controls/group_navigation_expanded",
                       true);
    }

    addViewerGroup(tr("Overlay"),
                   detachScrollContents(ui.scrollAreaOverlay, ui.dockWidgetOverlayContents),
                   viewer::GROUP_OVERLAY_EXPANDED,
                   viewer::GROUP_OVERLAY_EXPANDED_DEFAULT);
    addViewerGroup(tr("Composite View"),
                   detachScrollContents(ui.scrollAreaComposite, ui.dockWidgetCompositeContents),
                   viewer::GROUP_COMPOSITE_EXPANDED,
                   viewer::GROUP_COMPOSITE_EXPANDED_DEFAULT);
    addViewerGroup(tr("Render Settings"),
                   detachScrollContents(ui.scrollAreaRenderSettings, ui.dockWidgetRenderSettingsContents),
                   viewer::GROUP_RENDER_SETTINGS_EXPANDED,
                   viewer::GROUP_RENDER_SETTINGS_EXPANDED_DEFAULT);
    addViewerGroup(tr("Normal Visualization"),
                   normalVisContainer,
                   viewer::GROUP_NORMAL_VIS_EXPANDED,
                   viewer::GROUP_NORMAL_VIS_EXPANDED_DEFAULT);
    addViewerGroup(tr("Preprocessing"),
                   detachScrollContents(ui.scrollAreaPreprocessing, ui.dockWidgetPreprocessingContents),
                   viewer::GROUP_PREPROCESSING_EXPANDED,
                   viewer::GROUP_PREPROCESSING_EXPANDED_DEFAULT);
    addViewerGroup(tr("Postprocessing"),
                   detachScrollContents(ui.scrollAreaPostprocessing, ui.dockWidgetPostprocessingContents),
                   viewer::GROUP_POSTPROCESSING_EXPANDED,
                   viewer::GROUP_POSTPROCESSING_EXPANDED_DEFAULT);
    addViewerGroup(tr("Transforms"),
                   transformsContainer,
                   viewer::GROUP_TRANSFORMS_EXPANDED,
                   viewer::GROUP_TRANSFORMS_EXPANDED_DEFAULT);
    viewerControlsLayout->addStretch(1);

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
    refreshTransformsPanelState();

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
    }

        // Setup base colormap selector
    {
        const auto& entries = volume_viewer_cmaps::entries(volume_viewer_cmaps::EntryScope::SharedOnly);
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
        _viewerManager->forEachViewer([&id](CTiledVolumeViewer* viewer) {
            viewer->setBaseColormap(id.toStdString());
        });
    });

    // Setup surface overlay controls
    connect(ui.chkSurfaceOverlay, &QCheckBox::toggled, [this](bool checked) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([checked](CTiledVolumeViewer* viewer) {
            viewer->setSurfaceOverlayEnabled(checked);
        });
        ui.surfaceOverlaySelect->setEnabled(checked);
        ui.spinOverlapThreshold->setEnabled(checked);
    });

    connect(ui.spinOverlapThreshold, qOverload<double>(&QDoubleSpinBox::valueChanged), [this](double value) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([value](CTiledVolumeViewer* viewer) {
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

    // Surface normals visualization controls
    if (auto* chkShowNormals = ui.chkShowSurfaceNormals) {
        bool showNormals = settings.value(vc3d::settings::viewer::SHOW_SURFACE_NORMALS,
                                          vc3d::settings::viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        QSignalBlocker blocker(chkShowNormals);
        chkShowNormals->setChecked(showNormals);

        // Enable/disable the arrow length and max arrows controls based on checkbox state
        if (auto* lblArrowLength = ui.labelNormalArrowLength) {
            lblArrowLength->setEnabled(showNormals);
        }
        if (auto* sliderArrowLength = ui.sliderNormalArrowLength) {
            sliderArrowLength->setEnabled(showNormals);
        }
        if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
            lblArrowLengthValue->setEnabled(showNormals);
        }
        if (auto* lblMaxArrows = ui.labelNormalMaxArrows) {
            lblMaxArrows->setEnabled(showNormals);
        }
        if (auto* sliderMaxArrows = ui.sliderNormalMaxArrows) {
            sliderMaxArrows->setEnabled(showNormals);
        }
        if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
            lblMaxArrowsValue->setEnabled(showNormals);
        }

        connect(chkShowNormals, &QCheckBox::toggled, this, [this](bool checked) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::SHOW_SURFACE_NORMALS, checked ? "1" : "0");
            if (_viewerManager) {
                _viewerManager->forEachViewer([checked](CTiledVolumeViewer* viewer) {
                    if (viewer) {
                        viewer->setShowSurfaceNormals(checked);
                    }
                });
            }
            // Enable/disable arrow length and max arrows controls
            if (auto* lblArrowLength = ui.labelNormalArrowLength) {
                lblArrowLength->setEnabled(checked);
            }
            if (auto* sliderArrowLength = ui.sliderNormalArrowLength) {
                sliderArrowLength->setEnabled(checked);
            }
            if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
                lblArrowLengthValue->setEnabled(checked);
            }
            if (auto* lblMaxArrows = ui.labelNormalMaxArrows) {
                lblMaxArrows->setEnabled(checked);
            }
            if (auto* sliderMaxArrows = ui.sliderNormalMaxArrows) {
                sliderMaxArrows->setEnabled(checked);
            }
            if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
                lblMaxArrowsValue->setEnabled(checked);
            }
            statusBar()->showMessage(checked ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
        });
    }

    if (auto* sliderArrowLength = ui.sliderNormalArrowLength) {
        int savedScale = settings.value(vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE,
                                        vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE_DEFAULT).toInt();
        savedScale = std::clamp(savedScale, sliderArrowLength->minimum(), sliderArrowLength->maximum());
        QSignalBlocker blocker(sliderArrowLength);
        sliderArrowLength->setValue(savedScale);

        if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
            lblArrowLengthValue->setText(tr("%1%").arg(savedScale));
        }

        float scaleFloat = static_cast<float>(savedScale) / 100.0f;
        if (_viewerManager) {
            _viewerManager->forEachViewer([scaleFloat](CTiledVolumeViewer* viewer) {
                if (viewer) {
                    viewer->setNormalArrowLengthScale(scaleFloat);
                }
            });
        }

        connect(sliderArrowLength, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_ARROW_LENGTH_SCALE, value);

            if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
                lblArrowLengthValue->setText(tr("%1%").arg(value));
            }

            float scaleFloat = static_cast<float>(value) / 100.0f;
            if (_viewerManager) {
                _viewerManager->forEachViewer([scaleFloat](CTiledVolumeViewer* viewer) {
                    if (viewer) {
                        viewer->setNormalArrowLengthScale(scaleFloat);
                    }
                });
            }
        });
    }

    if (auto* sliderMaxArrows = ui.sliderNormalMaxArrows) {
        int savedMaxArrows = settings.value(vc3d::settings::viewer::NORMAL_MAX_ARROWS,
                                            vc3d::settings::viewer::NORMAL_MAX_ARROWS_DEFAULT).toInt();
        savedMaxArrows = std::clamp(savedMaxArrows, sliderMaxArrows->minimum(), sliderMaxArrows->maximum());
        QSignalBlocker blocker(sliderMaxArrows);
        sliderMaxArrows->setValue(savedMaxArrows);

        if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
            lblMaxArrowsValue->setText(QString::number(savedMaxArrows));
        }

        if (_viewerManager) {
            _viewerManager->forEachViewer([savedMaxArrows](CTiledVolumeViewer* viewer) {
                if (viewer) {
                    viewer->setNormalMaxArrows(savedMaxArrows);
                }
            });
        }

        connect(sliderMaxArrows, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_MAX_ARROWS, value);

            if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
                lblMaxArrowsValue->setText(QString::number(value));
            }

            if (_viewerManager) {
                _viewerManager->forEachViewer([value](CTiledVolumeViewer* viewer) {
                    if (viewer) {
                        viewer->setNormalMaxArrows(value);
                    }
                });
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
            {"16x", 16},
            {"32x", 32},
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

    // Max intersection surfaces spinbox
    if (auto* spinMaxSurfaces = ui.spinIntersectionMaxSurfaces) {
        const int savedMax =
            settings.value(vc3d::settings::viewer::INTERSECTION_MAX_SURFACES, vc3d::settings::viewer::INTERSECTION_MAX_SURFACES_DEFAULT).toInt();
        spinMaxSurfaces->setValue(std::max(0, savedMax));
        connect(spinMaxSurfaces, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            if (_viewerManager) {
                _viewerManager->setIntersectionMaxSurfaces(value);
            }
        });
        if (_viewerManager) {
            _viewerManager->setIntersectionMaxSurfaces(savedMax);
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
            auto s = viewer->compositeRenderSettings();
            s.enabled = checked;
            viewer->setCompositeRenderSettings(s);
        }
    });

    connect(ui.cmbCompositeMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (!_viewerManager) {
            return;
        }
        const std::string method = compositeMethodForModeIndex(index);
        _viewerManager->forEachViewer([&method](CTiledVolumeViewer* viewer) {
            if (!viewer) {
                return;
            }
            auto s = viewer->compositeRenderSettings();
            s.params.method = method;
            viewer->setCompositeRenderSettings(s);
        });
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
            auto s = viewer->compositeRenderSettings();
            s.layersFront = value;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Layers Behind controls
    connect(ui.spinLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.layersBehind = value;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Alpha Min controls
    connect(ui.spinAlphaMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.alphaMin = value / 255.0f;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Alpha Max controls
    connect(ui.spinAlphaMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.alphaMax = value / 255.0f;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Alpha Threshold controls
    connect(ui.spinAlphaThreshold, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.alphaCutoff = value / 10000.0f;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Material controls
    connect(ui.spinMaterial, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.alphaOpacity = value / 255.0f;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Reverse Direction control
    connect(ui.chkReverseDirection, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.reverseDirection = checked;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Beer-Lambert Extinction control
    connect(ui.spinBLExtinction, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.blExtinction = static_cast<float>(value);
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Beer-Lambert Emission control
    connect(ui.spinBLEmission, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.blEmission = static_cast<float>(value);
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Beer-Lambert Ambient control
    connect(ui.spinBLAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.blAmbient = static_cast<float>(value);
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Lighting Enable control
    connect(ui.chkLightingEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.lightingEnabled = checked;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Light Azimuth control
    connect(ui.spinLightAzimuth, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.lightAzimuth = static_cast<float>(value);
            s.params.updateLightDir();
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Light Elevation control
    connect(ui.spinLightElevation, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.lightElevation = static_cast<float>(value);
            s.params.updateLightDir();
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Light Diffuse control
    connect(ui.spinLightDiffuse, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.lightDiffuse = static_cast<float>(value);
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Light Ambient control
    connect(ui.spinLightAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.lightAmbient = static_cast<float>(value);
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Volume Gradients checkbox — switches the lighting normal
    // source between mesh-interpolated (0) and per-sample volume gradient (1).
    connect(ui.chkUseVolumeGradients, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.useVolumeGradients = checked;
            s.params.lightNormalSource = checked ? 1 : 0;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Connect Shadow Steps spinbox (Volumetric method)
    connect(ui.spinShadowSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.shadowSteps = std::clamp(value, 1, 64);
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Per-ray layer preprocess (applied to N composite samples before composite method)
    connect(ui.chkPreNormalizeLayers, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.preNormalizeLayers = checked;
            viewer->setCompositeRenderSettings(s);
        }
    });
    connect(ui.chkPreHistEqLayers, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.params.preHistEqLayers = checked;
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Pre-TF / Post-TF: 4-knot piecewise-linear LUTs. Endpoints (0,0) and
    // (255,255) are fixed; only the two middle knots are editable.
    auto applyTfParam = [this](auto&& mutate) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            mutate(s.params);
            viewer->setCompositeRenderSettings(s);
        }
    };
    connect(ui.chkPreTfEnabled, &QCheckBox::toggled, this, [applyTfParam](bool v) {
        applyTfParam([v](CompositeParams& p) { p.preTfEnabled = v; });
    });
    connect(ui.spinPreTfX1, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.preTfX1 = uint8_t(v); }); });
    connect(ui.spinPreTfY1, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.preTfY1 = uint8_t(v); }); });
    connect(ui.spinPreTfX2, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.preTfX2 = uint8_t(v); }); });
    connect(ui.spinPreTfY2, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.preTfY2 = uint8_t(v); }); });
    connect(ui.chkPostTfEnabled, &QCheckBox::toggled, this, [applyTfParam](bool v) {
        applyTfParam([v](CompositeParams& p) { p.postTfEnabled = v; });
    });
    connect(ui.spinPostTfX1, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.postTfX1 = uint8_t(v); }); });
    connect(ui.spinPostTfY1, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.postTfY1 = uint8_t(v); }); });
    connect(ui.spinPostTfX2, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.postTfX2 = uint8_t(v); }); });
    connect(ui.spinPostTfY2, QOverload<int>::of(&QSpinBox::valueChanged), this,
        [applyTfParam](int v) { applyTfParam([v](CompositeParams& p) { p.postTfY2 = uint8_t(v); }); });
    connect(ui.spinDvrAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
        [applyTfParam](double v) { applyTfParam([v](CompositeParams& p) { p.dvrAmbient = float(v); }); });
    connect(ui.spinPbrRoughness, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
        [applyTfParam](double v) { applyTfParam([v](CompositeParams& p) { p.pbrRoughness = float(v); }); });
    connect(ui.spinPbrMetallic, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
        [applyTfParam](double v) { applyTfParam([v](CompositeParams& p) { p.pbrMetallic = float(v); }); });

    // Connect ISO Cutoff slider - applies to all viewers (segmentation, XY, XZ, YZ)
    connect(ui.sliderIsoCutoff, &QSlider::valueChanged, this, [this](int value) {
        ui.lblIsoCutoffValue->setText(QString::number(value));
        if (!_viewerManager) {
            return;
        }
        _viewerManager->forEachViewer([value](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.params.isoCutoff = static_cast<uint8_t>(std::clamp(value, 0, 255));
            viewer->setCompositeRenderSettings(s);
        });
    });

    // Connect Method Scale slider (for methods with scale parameters)
    connect(ui.sliderMethodScale, &QSlider::valueChanged, this, [this](int value) {
        // Convert slider value (1-100) to scale (0.1-10.0)
        float scale = value / 10.0f;
        ui.lblMethodScaleValue->setText(QString::number(scale, 'f', 1));

        if (!_viewerManager) {
            return;
        }

        // Currently no methods use the scale parameter
        (void)scale;
    });

    // Connect Method Param slider (for methods with threshold/percentile parameters)
    connect(ui.sliderMethodParam, &QSlider::valueChanged, this, [this](int value) {
        // Currently no methods use this parameter
        (void)value;
    });

    // Helper lambda to update visibility of method-specific parameters
    auto updateCompositeParamsVisibility = [this]() {
        const int methodIndex = ui.cmbCompositeMode->currentIndex();
        const bool lightingOn = ui.chkLightingEnabled->isChecked();
        const bool preTfOn = ui.chkPreTfEnabled->isChecked();
        const bool postTfOn = ui.chkPostTfEnabled->isChecked();

        // Method-family flags.
        const bool isAlpha    = (methodIndex == 3);
        const bool isBL       = (methodIndex == 4);
        const bool isVolum    = (methodIndex == 5);
        const bool isDvr      = (methodIndex == 6);
        const bool isFirstHit = (methodIndex == 7);
        const bool isPbr      = (methodIndex == 13);
        const bool isShadedDvr = (methodIndex == 14);

        // Alpha knobs: only for the Alpha method.
        ui.lblAlphaMin->setVisible(isAlpha);
        ui.spinAlphaMin->setVisible(isAlpha);
        ui.lblAlphaMax->setVisible(isAlpha);
        ui.spinAlphaMax->setVisible(isAlpha);
        ui.lblAlphaThreshold->setVisible(isAlpha);
        ui.spinAlphaThreshold->setVisible(isAlpha);
        ui.lblMaterial->setVisible(isAlpha);
        ui.spinMaterial->setVisible(isAlpha);

        // Beer-Lambert knobs: shared by Beer-Lambert and Volumetric modes.
        const bool showBL = isBL || isVolum;
        ui.lblBLExtinction->setVisible(showBL);
        ui.spinBLExtinction->setVisible(showBL);
        ui.lblBLEmission->setVisible(showBL);
        ui.spinBLEmission->setVisible(showBL);
        ui.lblBLAmbient->setVisible(showBL);
        ui.spinBLAmbient->setVisible(showBL);

        // Shadow-ray steps: only Volumetric uses the secondary shadow ray.
        ui.lblShadowSteps->setVisible(isVolum);
        ui.spinShadowSteps->setVisible(isVolum);

        // DVR ambient: DVR and shaded-DVR methods.
        const bool showDvrAmbient = isDvr || isShadedDvr;
        ui.lblDvrAmbient->setVisible(showDvrAmbient);
        ui.spinDvrAmbient->setVisible(showDvrAmbient);

        // PBR roughness/metallic knobs: only the PBR method.
        ui.lblPbrRoughness->setVisible(isPbr);
        ui.spinPbrRoughness->setVisible(isPbr);
        ui.lblPbrMetallic->setVisible(isPbr);
        ui.spinPbrMetallic->setVisible(isPbr);

        // Lighting: check always visible (user toggles on/off); the
        // direction/diffuse/ambient knobs appear only when lighting is
        // actually on. First-Hit Iso is a shading-heavy method so we
        // gently enforce its need for lighting by showing the chk always.
        ui.chkLightingEnabled->setVisible(true);
        ui.lblLightAzimuth->setVisible(lightingOn);
        ui.spinLightAzimuth->setVisible(lightingOn);
        ui.lblLightElevation->setVisible(lightingOn);
        ui.spinLightElevation->setVisible(lightingOn);
        ui.lblLightDiffuse->setVisible(lightingOn);
        ui.spinLightDiffuse->setVisible(lightingOn);
        ui.lblLightAmbient->setVisible(lightingOn);
        ui.spinLightAmbient->setVisible(lightingOn);
        ui.chkUseVolumeGradients->setVisible(lightingOn);

        // Pre/Post TF knots: spinboxes + knot-2 labels appear only when
        // the corresponding enable checkbox is ticked.
        ui.spinPreTfX1->setVisible(preTfOn);
        ui.spinPreTfY1->setVisible(preTfOn);
        ui.spinPreTfX2->setVisible(preTfOn);
        ui.spinPreTfY2->setVisible(preTfOn);
        ui.lblPreTfKnot2->setVisible(preTfOn);
        ui.spinPostTfX1->setVisible(postTfOn);
        ui.spinPostTfY1->setVisible(postTfOn);
        ui.spinPostTfX2->setVisible(postTfOn);
        ui.spinPostTfY2->setVisible(postTfOn);
        ui.lblPostTfKnot2->setVisible(postTfOn);

        // No methods currently use scale or param sliders.
        ui.lblMethodScale->setVisible(false);
        ui.sliderMethodScale->setVisible(false);
        ui.lblMethodScaleValue->setVisible(false);
        ui.lblMethodParam->setVisible(false);
        ui.sliderMethodParam->setVisible(false);
        ui.lblMethodParamValue->setVisible(false);

        (void)isFirstHit;  // reserved for future First-Hit-specific knobs
        (void)isShadedDvr; (void)isPbr; // already consumed above
    };

    // Re-run visibility logic whenever any of the inputs that gate widgets
    // change — composite method, or any of the three enable checkboxes
    // that each control a sub-group of knobs.
    connect(ui.cmbCompositeMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [updateCompositeParamsVisibility](int) { updateCompositeParamsVisibility(); });
    connect(ui.chkLightingEnabled, &QCheckBox::toggled,
            this, [updateCompositeParamsVisibility](bool) { updateCompositeParamsVisibility(); });
    connect(ui.chkPreTfEnabled, &QCheckBox::toggled,
            this, [updateCompositeParamsVisibility](bool) { updateCompositeParamsVisibility(); });
    connect(ui.chkPostTfEnabled, &QCheckBox::toggled,
            this, [updateCompositeParamsVisibility](bool) { updateCompositeParamsVisibility(); });

    // Initialize visibility from current UI state.
    updateCompositeParamsVisibility();

    // Connect Plane Composite controls (separate enable for XY/XZ/YZ, shared layer counts)
    connect(ui.chkPlaneCompositeXY, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) return;
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "xy plane") {
                auto s = viewer->compositeRenderSettings();
                s.planeEnabled = checked;
                viewer->setCompositeRenderSettings(s);
            }
        }
    });

    connect(ui.chkPlaneCompositeXZ, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) return;
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "seg xz") {
                auto s = viewer->compositeRenderSettings();
                s.planeEnabled = checked;
                viewer->setCompositeRenderSettings(s);
            }
        }
    });

    connect(ui.chkPlaneCompositeYZ, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) return;
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "seg yz") {
                auto s = viewer->compositeRenderSettings();
                s.planeEnabled = checked;
                viewer->setCompositeRenderSettings(s);
            }
        }
    });

    auto isPlaneViewer = [](const std::string& name) {
        return name == "seg xz" || name == "seg yz" || name == "xy plane";
    };

    connect(ui.spinPlaneLayersFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, isPlaneViewer](int value) {
        if (!_viewerManager) return;
        int behind = ui.spinPlaneLayersBehind->value();
        for (auto* viewer : _viewerManager->viewers()) {
            if (isPlaneViewer(viewer->surfName())) {
                auto s = viewer->compositeRenderSettings();
                s.planeLayersFront = std::max(0, value);
                s.planeLayersBehind = std::max(0, behind);
                viewer->setCompositeRenderSettings(s);
            }
        }
    });

    connect(ui.spinPlaneLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, isPlaneViewer](int value) {
        if (!_viewerManager) return;
        int front = ui.spinPlaneLayersFront->value();
        for (auto* viewer : _viewerManager->viewers()) {
            if (isPlaneViewer(viewer->surfName())) {
                auto s = viewer->compositeRenderSettings();
                s.planeLayersFront = std::max(0, front);
                s.planeLayersBehind = std::max(0, value);
                viewer->setCompositeRenderSettings(s);
            }
        }
    });

    // Connect Postprocessing controls
    connect(ui.chkStretchValuesPost, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.postStretchValues = checked;
            viewer->setCompositeRenderSettings(s);
        }
    });

    connect(ui.chkRemoveSmallComponents, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.postRemoveSmallComponents = checked;
            viewer->setCompositeRenderSettings(s);
        }
        // Enable/disable the min component size spinbox based on checkbox state
        ui.spinMinComponentSize->setEnabled(checked);
        ui.lblMinComponentSize->setEnabled(checked);
    });

    connect(ui.spinMinComponentSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            auto s = viewer->compositeRenderSettings();
            s.postMinComponentSize = std::clamp(value, 1, 100000);
            viewer->setCompositeRenderSettings(s);
        }
    });

    // Initialize min component size controls based on checkbox state
    ui.spinMinComponentSize->setEnabled(ui.chkRemoveSmallComponents->isChecked());
    ui.lblMinComponentSize->setEnabled(ui.chkRemoveSmallComponents->isChecked());

    // CLAHE postprocessing — applied to every viewer
    auto setClaheEnabled = [this](bool on) {
        ui.spinClaheClipLimit->setEnabled(on);
        ui.spinClaheTileSize->setEnabled(on);
        ui.lblClaheClipLimit->setEnabled(on);
        ui.lblClaheTileSize->setEnabled(on);
    };
    setClaheEnabled(ui.chkClaheEnabled->isChecked());

    connect(ui.chkClaheEnabled, &QCheckBox::toggled, this, [this, setClaheEnabled](bool checked) {
        setClaheEnabled(checked);
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([checked](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postClaheEnabled = checked;
            viewer->setCompositeRenderSettings(s);
        });
    });

    connect(ui.spinClaheClipLimit, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([value](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postClaheClipLimit = static_cast<float>(value);
            viewer->setCompositeRenderSettings(s);
        });
    });

    connect(ui.spinClaheTileSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([value](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postClaheTileSize = std::clamp(value, 2, 64);
            viewer->setCompositeRenderSettings(s);
        });
    });

    // Raking light — heightfield post-process
    auto setRakingEnabled = [this](bool on) {
        ui.spinRakingAzimuth->setEnabled(on);
        ui.spinRakingElevation->setEnabled(on);
        ui.spinRakingStrength->setEnabled(on);
        ui.spinRakingDepthScale->setEnabled(on);
        ui.lblRakingAzimuth->setEnabled(on);
        ui.lblRakingElevation->setEnabled(on);
        ui.lblRakingStrength->setEnabled(on);
        ui.lblRakingDepth->setEnabled(on);
    };
    setRakingEnabled(ui.chkRakingEnabled->isChecked());

    connect(ui.chkRakingEnabled, &QCheckBox::toggled, this, [this, setRakingEnabled](bool checked) {
        setRakingEnabled(checked);
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([checked](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postRakingEnabled = checked;
            viewer->setCompositeRenderSettings(s);
        });
    });
    connect(ui.spinRakingAzimuth, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([v](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postRakingAzimuth = float(v);
            viewer->setCompositeRenderSettings(s);
        });
    });
    connect(ui.spinRakingElevation, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([v](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postRakingElevation = float(v);
            viewer->setCompositeRenderSettings(s);
        });
    });
    connect(ui.spinRakingStrength, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([v](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postRakingStrength = std::clamp(float(v), 0.0f, 1.0f);
            viewer->setCompositeRenderSettings(s);
        });
    });
    connect(ui.spinRakingDepthScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([v](CTiledVolumeViewer* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.postRakingDepthScale = std::max(0.01f, float(v));
            viewer->setCompositeRenderSettings(s);
        });
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
    saveWindowState();
    event->accept();
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
    _state->setVpkg(nullptr);
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
        _state->setVpkg(VolumePkg::New(nVpkgPath));
    } catch (const std::exception& e) {
        Logger()->error("Failed to initialize volpkg: {}", e.what());
    }

    if (_state->vpkg() == nullptr) {
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
        _state->setVpkg(nullptr);  // Is needed for User Experience, clears screen.
        updateNormalGridAvailability();
        return;
    }

    // Open volume package
    if (!InitializeVolumePkg(aVpkgPath.toStdString() + "/")) {
        return;
    }

    // Detect network-mounted volpkg and inform user about auto-caching
    {
        namespace fs = std::filesystem;
        auto mountInfo = vc::detectNetworkMount(fs::path(aVpkgPath.toStdString()));
        if (mountInfo.type == vc::FilesystemType::NetworkMount) {
            auto label = mountInfo.label;
            if (!mountInfo.cacheDir.empty()) {
                statusBar()->showMessage(
                    tr("Detected %1 mount (use_cache active) \u2014 using s3fs disk cache")
                        .arg(QString::fromStdString(label)), 8000);
                Logger()->info("Detected {} mount with use_cache={}; using s3fs disk cache",
                               label, mountInfo.cacheDir);
            } else {
                statusBar()->showMessage(
                    tr("Detected %1 mount").arg(QString::fromStdString(label)), 8000);
                Logger()->info("Detected network filesystem ({})", label);
            }
        }
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

    clearTransformPreview(false);

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

    _remoteScroll.active = false;

    // Update UI
    UpdateView();
    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clear();
    }

    if (_volumeOverlay) {
        _volumeOverlay->clearVolumePkg();
    }

    refreshTransformsPanelState();
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
        if (_state && surf) {
            surfId = _state->findSurfaceId(surf);
        }
        centerFocusAt(vol_loc, normal, surfId);
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

    auto planeShared = _state->surface("manual plane");
    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(planeShared.get());

    if (!plane)
        return;

    plane->setNormal(normal);
    _state->setSurface("manual plane", planeShared);
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

        // Handle approval mask when switching segments
        if (_segmentationModule) {
            _segmentationModule->onActiveSegmentChanged(surf.get());
        }
    }

    if (surf) {
        _axisAlignedSliceController->applyOrientation(surf.get());
    } else {
        _axisAlignedSliceController->applyOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }

    refreshTransformsPanelState();
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
        _segmentationModule->onActiveSegmentChanged(surf.get());

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
                        _viewerManager->forEachViewer([](CTiledVolumeViewer* viewer) {
                            if (viewer) {
                                viewer->clearOverlayGroup("segmentation_radius_indicator");
                            }
                        });
                    }
                }
            }
        }
    }

    if (surf) {
        _axisAlignedSliceController->applyOrientation(surf.get());
    } else {
        _axisAlignedSliceController->applyOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }

    refreshTransformsPanelState();
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
                render_image_from_coords(coords, img, volume.get());
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            existing_layers.push_back(img);
            imwritemulti(path.string(), existing_layers);

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
            imwritemulti(path.string(), layers);

            surf->meta["date_last_modified"] = get_surface_time_str();
            surf->save_meta();
            return QString("Created new surface mask with image data");
        }
    }));
}

QString CWindow::getCurrentVolumePath() const
{
    if (_state->currentVolume() == nullptr) {
        return QString();
    }
    return QString::fromStdString(_state->currentVolume()->path().string());
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
    auto [w, h, d] = _state->currentVolume()->shape();
    int cx0 = 0, cy0 = 0, cz0 = 0;
    int cx1 = w - 1, cy1 = h - 1, cz1 = d - 1;

    x = std::max(cx0, std::min(x, cx1));
    y = std::max(cy0, std::min(y, cy1));
    z = std::max(cz0, std::min(z, cz1));

    // Update the line edit with clamped values
    lblLocFocus->setText(QString("%1, %2, %3").arg(x).arg(y).arg(z));

    // Update the focus POI
    POI* poi = _state->poi("focus");
    if (!poi) {
        poi = new POI;
    }

    poi->p = cv::Vec3f(x, y, z);
    poi->n = cv::Vec3f(0, 0, 1); // Default normal for XY plane

    _state->setPOI("focus", poi);

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
    CTiledVolumeViewer* viewer = qobject_cast<CTiledVolumeViewer*>(activeWindow->widget());
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

        _axisAlignedSliceController->applyOrientation();

        const cv::Vec3f focusPosition = poi->p;
        QTimer::singleShot(0, this, [this, focusPosition]() {
            recenterPlaneViewersOn(focusPosition);
        });
    }
}

void CWindow::onPointDoubleClicked(uint64_t pointId)
{
    auto point_opt = _state->pointCollection()->getPoint(pointId);
    if (point_opt) {
        POI *poi = _state->poi("focus");
        if (!poi) {
            poi = new POI;
        }
        poi->p = point_opt->p;

        // Find the closest normal on the segmentation surface
        auto seg_surface = _state->surface("segmentation");
        if (auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get())) {
            cv::Vec3f ptr(0, 0, 0);
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            quad_surface->pointTo(ptr, point_opt->p, 4.0, 100, patchIndex);
            poi->n = quad_surface->normal(ptr, quad_surface->loc(ptr));
        } else {
            poi->n = cv::Vec3f(0, 0, 1); // Default normal if no surface
        }

        _state->setPOI("focus", poi);
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
    // Get the active sub-window
    QMdiSubWindow* activeWindow = mdiArea->activeSubWindow();
    if (!activeWindow) return;

    // Get the viewer from the active window
    CTiledVolumeViewer* viewer = qobject_cast<CTiledVolumeViewer*>(activeWindow->widget());
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
        _viewerManager->forEachViewer([this, enabled](CTiledVolumeViewer* viewer) {
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
            _viewerManager->forEachViewer([](CTiledVolumeViewer* viewer) {
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
    refreshTransformsPanelState();
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
    _viewerManager->forEachViewer([&selectedSurfaces](CTiledVolumeViewer* viewer) {
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
