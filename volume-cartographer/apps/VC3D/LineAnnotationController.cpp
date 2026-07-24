#include "LineAnnotationController.hpp"

#include "CState.hpp"
#include "FiberNameDisplay.hpp"
#include "OpenDataCoordinateIdentity.hpp"
#include "OpenDataLasagna.hpp"
#include "FiberSliceGeometry.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationFiberSaveJob.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "LineAnnotationDialog.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/FiberSliceOverlayController.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/atlas/Atlas.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QFileDialog>
#include <QFutureWatcher>
#include <QButtonGroup>
#include <QColor>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QEventLoop>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QMessageBox>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPoint>
#include <QPointF>
#include <QPushButton>
#include <QRadioButton>
#include <QShortcut>
#include <QDateTime>
#include <QSettings>
#include <QStringList>
#include <QThread>
#include <QThreadPool>
#include <QtConcurrent/QtConcurrent>
#include <QVariant>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct LineAnnotationController::LineAnnotationSession {
    enum class TaskState {
        Idle,
        Running,
        Succeeded,
        Failed,
    };

    std::string surfaceName;
    std::string selectedDatasetLocation;
    fs::path selectedManifestPath;
    double workingToBaseScale = 1.0;
    std::shared_ptr<vc::lasagna::LasagnaDataset> dataset;
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> normalSampler;
    TaskState taskState = TaskState::Idle;
    cv::Vec3d seedPoint{0.0, 0.0, 0.0};
    std::string sourceAnnotationSurfaceName;
    vc::lasagna::LineOptimizationReport optimizationReport;
    vc::lasagna::LineModel optimizedLine;
    // These vectors are intentionally private to this translation unit. Branch
    // refs store live control-point indices plus reciprocal metadata in linked
    // fibers; after mutating either vector, call
    // syncLinkedBranchMetadataAfterFiberModification() before refreshing overlays
    // or saving.
    std::vector<vc::lasagna::LineControlPoint> controlPoints;
    std::vector<LineAnnotationController::FiberBranchRef> branches;
    bool showLinkedLineOverlays = false;
    double focusedLinePosition = 0.0;
    std::optional<cv::Vec3d> focusedControlPoint;
    cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
    LineAnnotationController::InitialDirectionMode initialDirectionMode =
        LineAnnotationController::InitialDirectionMode::Sideways;
    std::vector<std::string> generatedSurfaceNames;
    std::string generatedLineSurfaceName;
    std::string generatedLineSideSliceName;
    std::string error;
    QPointer<QFutureWatcher<OptimizationTaskResult>> watcher;
    bool deferShowUntilGenerated = false;
    uint64_t fiberId = 0;
    std::string fiberUsername;
    std::string fiberStartedAt;
    uint64_t fiberSequence = 0;
    std::string fiberFileName;
    std::string fiberManualHvTag;
    std::vector<std::string> fiberTags;
    std::optional<fs::path> atlasDir;
    fs::path atlasFiberPath;
    vc::atlas::AtlasPredSnapSet predSnapSet;
    bool suppressFiberSave = false;
    bool suppressGeneratedViews = false;
    LineAnnotationController::SessionOptimizationState optimizationState =
        LineAnnotationController::SessionOptimizationState::Unoptimized;
    LineAnnotationController::SessionOptimizationState pendingOptimizationState =
        LineAnnotationController::SessionOptimizationState::Optimized;
    std::optional<std::pair<double, double>> initialStripLinePositionRange;
    bool disableInitialGeneratedHoverFollow = false;
    std::function<void(LineAnnotationSession&)> optimizationSucceededCallback;
    bool fiberMetricsMatchStoredFiber = false;
};

struct LineAnnotationController::FiberMetricsTaskResult {
    bool ok = false;
    uint64_t generation = 0;
    fs::path manifestPath;
    std::string error;
    std::vector<uint64_t> requestedFiberIds;
    std::unordered_map<uint64_t, uint64_t> requestTokens;
    std::unordered_map<uint64_t, CachedFiberAlignmentMetrics> metrics;
};

struct LineAnnotationController::IntersectionInspectionSession {
    struct FollowSlice {
        bool valid = false;
        bool followsMouse = true;
        bool sourceSide = true;
        std::string surfaceName;
        QPointer<CChunkedVolumeViewer> viewer;
        QPointer<QLabel> controlPointInfoLabel;
        std::vector<cv::Vec3d> linePoints;
        std::vector<cv::Vec3f> lineUpVectors;
        std::vector<vc::lasagna::LineControlPoint> controlPoints;
        double linePosition = 0.0;
    };

    struct GeneratedSurfaceContext {
        bool valid = false;
        bool sourceSide = true;
        bool strip = false;
        bool follow = false;
        double linePosition = 0.0;
    };

    QPointer<QMdiArea> targetArea;
    vc::atlas::FiberIntersectionResult result;
    std::optional<fs::path> atlasDir;
    double sourceFocusLinePosition = 0.0;
    double targetFocusLinePosition = 0.0;
    std::vector<std::string> surfaceNames;
    std::shared_ptr<LineAnnotationSession> sourceLineSession;
    std::shared_ptr<LineAnnotationSession> targetLineSession;
    std::string sourceSessionSurfaceName;
    std::string targetSessionSurfaceName;
    FollowSlice sourceFollow;
    FollowSlice targetFollow;
    QPointer<QShortcut> followShortcut;
    std::optional<bool> activeFollowSourceSide;
    std::map<std::string, GeneratedSurfaceContext> generatedSurfaceContexts;
};

namespace {

std::optional<vc3d::opendata::CoordinateIdentity> coordinateIdentityForState(
    const CState* state)
{
    if (!state || !state->vpkg() || state->currentVolumeId().empty())
        return std::nullopt;
    return vc3d::opendata::coordinateIdentityForVolume(
        *state->vpkg(), state->currentVolumeId());
}

void copyCoordinateIdentityToJson(
    nlohmann::json& target,
    const std::optional<vc3d::opendata::CoordinateIdentity>& identity)
{
    if (!identity)
        return;
    target["vc_open_data_coordinate_space"] = identity->coordinateSpace;
    target["vc_open_data_source_path"] = identity->sourcePath;
    target["vc_open_data_source_coordinate_level"] =
        identity->sourceCoordinateLevel;
    target["vc_open_data_source_coordinate_scale_factor"] =
        identity->sourceCoordinateScaleFactor;
    target["vc_open_data_source_original_resolution"] =
        identity->sourceOriginalResolution;
}

constexpr double kEpsilon = 1.0e-12;
constexpr double kLineSegmentLength = 32.0;
constexpr double kControlPointLabelLinePositionTolerance = 1.0e-3;
using Clock = std::chrono::steady_clock;

struct InitialLineDiscretization {
    int segmentsPerSide = 1;
    double segmentLength = kLineSegmentLength;
};

InitialLineDiscretization initialLineDiscretization(int totalLengthVx)
{
    const double halfLength = std::max(1, totalLengthVx) * 0.5;
    const int segmentsPerSide = std::max(
        1, static_cast<int>(std::ceil(halfLength / kLineSegmentLength)));
    return {segmentsPerSide, halfLength / static_cast<double>(segmentsPerSide)};
}

void closeLineAnnotationDialogAfterFinalization(LineAnnotationDialog* dialog)
{
    if (!dialog) {
        return;
    }
    dialog->setCloseAfterFinalizationAllowed(true);
    QMetaObject::invokeMethod(dialog,
                              [dialog]() {
                                  dialog->close();
                              },
                              Qt::QueuedConnection);
}

struct FiberJsonPathOptions {
    fs::path path;
    double scale = 1.0;
};

QString fiberDisplayNameFromFileName(const std::string& fileName)
{
    const QString stem = vc3d::displayStemForFiberFile(QString::fromStdString(fileName));
    return stem.isEmpty() ? QObject::tr("unsaved fiber") : stem;
}

std::optional<FiberJsonPathOptions> showFiberJsonPathDialog(QWidget* parent,
                                                            bool importMode,
                                                            const fs::path& defaultDir)
{
    QDialog dialog(parent);
    dialog.setWindowTitle(importMode
                              ? QObject::tr("Import Fibers")
                              : QObject::tr("Export Fibers"));

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();

    auto* pathRow = new QWidget(&dialog);
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    auto* pathEdit = new QLineEdit(pathRow);
    pathEdit->setPlaceholderText(importMode
                                     ? QObject::tr("Folder or .json path")
                                     : QObject::tr(".json path"));
    pathLayout->addWidget(pathEdit, 1);

    auto defaultDirectory = QString::fromStdString(defaultDir.string());
    if (importMode) {
        auto* browseFileButton = new QPushButton(QObject::tr("File"), pathRow);
        auto* browseFolderButton = new QPushButton(QObject::tr("Folder"), pathRow);
        pathLayout->addWidget(browseFileButton);
        pathLayout->addWidget(browseFolderButton);
        QObject::connect(browseFileButton, &QPushButton::clicked, &dialog, [&]() {
            const QString path = QFileDialog::getOpenFileName(
                &dialog,
                QObject::tr("Import fiber JSON"),
                defaultDirectory,
                QObject::tr("JSON files (*.json);;All files (*)"));
            if (!path.isEmpty()) {
                pathEdit->setText(path);
            }
        });
        QObject::connect(browseFolderButton, &QPushButton::clicked, &dialog, [&]() {
            const QString path = QFileDialog::getExistingDirectory(
                &dialog,
                QObject::tr("Import fiber JSON folder"),
                defaultDirectory);
            if (!path.isEmpty()) {
                pathEdit->setText(path);
            }
        });
    } else {
        auto* browseButton = new QPushButton(QObject::tr("Browse"), pathRow);
        pathLayout->addWidget(browseButton);
        QObject::connect(browseButton, &QPushButton::clicked, &dialog, [&]() {
            QString path = QFileDialog::getSaveFileName(
                &dialog,
                QObject::tr("Export fiber JSON"),
                defaultDirectory,
                QObject::tr("JSON files (*.json);;All files (*)"));
            if (!path.isEmpty()) {
                if (!path.endsWith(QStringLiteral(".json"), Qt::CaseInsensitive)) {
                    path += QStringLiteral(".json");
                }
                pathEdit->setText(path);
            }
        });
    }

    auto* scaleSpin = new QDoubleSpinBox(&dialog);
    scaleSpin->setRange(-1000000000.0, 1000000000.0);
    scaleSpin->setDecimals(6);
    scaleSpin->setSingleStep(0.25);
    scaleSpin->setValue(1.0);

    form->addRow(QObject::tr("Path:"), pathRow);
    form->addRow(QObject::tr("Scale:"), scaleSpin);
    layout->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                                         &dialog);
    layout->addWidget(buttons);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, [&]() {
        const QString text = pathEdit->text().trimmed();
        if (text.isEmpty()) {
            QMessageBox::warning(&dialog,
                                 dialog.windowTitle(),
                                 QObject::tr("Enter a path."));
            return;
        }

        const fs::path path(text.toStdString());
        std::error_code ec;
        const bool isDirectory = fs::is_directory(path, ec);
        if (importMode) {
            ec.clear();
            const bool isRegularFile = fs::is_regular_file(path, ec);
            if (!isDirectory && !isRegularFile) {
                QMessageBox::warning(&dialog,
                                     dialog.windowTitle(),
                                     QObject::tr("Import path must be an existing folder or JSON file."));
                return;
            }
            if (!isDirectory && path.extension() != ".json") {
                QMessageBox::warning(&dialog,
                                     dialog.windowTitle(),
                                     QObject::tr("Import file must end in .json."));
                return;
            }
        } else if (path.extension() != ".json") {
            QMessageBox::warning(&dialog,
                                 dialog.windowTitle(),
                                 QObject::tr("Export path must end in .json."));
            return;
        }

        if (!std::isfinite(scaleSpin->value())) {
            QMessageBox::warning(&dialog,
                                 dialog.windowTitle(),
                                 QObject::tr("Scale must be a finite number."));
            return;
        }
        dialog.accept();
    });

    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }

    return FiberJsonPathOptions{fs::path(pathEdit->text().trimmed().toStdString()),
                                scaleSpin->value()};
}

void writeJsonAtomic(const fs::path& finalPath, const nlohmann::json& root)
{
    std::error_code ec;
    const fs::path parent = finalPath.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) {
            throw std::runtime_error("Failed to create " + parent.string() + ": " + ec.message());
        }
    }

    const fs::path tempPath = finalPath.string() + ".tmp";
    {
        std::ofstream out(tempPath);
        if (!out) {
            throw std::runtime_error("Failed to open " + tempPath.string());
        }
        out << root.dump(2) << '\n';
    }
    fs::rename(tempPath, finalPath, ec);
    if (ec) {
        fs::remove(tempPath);
        throw std::runtime_error("Failed to replace " + finalPath.string() + ": " + ec.message());
    }
}

bool atlasDebugEnabled()
{
    const char* value = std::getenv("VC_ATLAS_DEBUG");
    return value && *value != '\0' && std::string_view(value) != "0";
}

void atlasDebug(const std::string& message)
{
    if (atlasDebugEnabled()) {
        Logger()->info("[atlas] {}", message);
    }
}

bool finitePoint(const cv::Vec3d& v);
cv::Vec3f toVec3f(const cv::Vec3d& v);

const char* sideStripProgressPhaseName(vc::atlas::FiberSideStripProgressPhase phase)
{
    switch (phase) {
    case vc::atlas::FiberSideStripProgressPhase::BuildStripTriangles:
        return "build strip triangles";
    case vc::atlas::FiberSideStripProgressPhase::BuildTriangleIndex:
        return "build triangle index";
    case vc::atlas::FiberSideStripProgressPhase::BranchLinks:
        return "branch links";
    case vc::atlas::FiberSideStripProgressPhase::FiberSegments:
        return "fiber segments";
    case vc::atlas::FiberSideStripProgressPhase::Deduplicate:
        return "deduplicate";
    default:
        return "unknown";
    }
}

void combineSideStripHash(uint64_t& seed, uint64_t value)
{
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

uint64_t hashSideStripDouble(double value)
{
    if (!std::isfinite(value)) {
        return 0x7ff8000000000000ULL;
    }
    return std::hash<long long>{}(
        static_cast<long long>(std::llround(value * 1000.0)));
}

void combineSideStripPointHash(uint64_t& seed, const cv::Vec3d& point)
{
    combineSideStripHash(seed, hashSideStripDouble(point[0]));
    combineSideStripHash(seed, hashSideStripDouble(point[1]));
    combineSideStripHash(seed, hashSideStripDouble(point[2]));
}

void combineSideStripPointHash(uint64_t& seed, const cv::Vec3f& point)
{
    combineSideStripHash(seed, hashSideStripDouble(point[0]));
    combineSideStripHash(seed, hashSideStripDouble(point[1]));
    combineSideStripHash(seed, hashSideStripDouble(point[2]));
}

uint64_t sideStripRequestCacheKey(
    const cv::Mat_<cv::Vec3f>& stripPoints,
    const std::vector<vc::atlas::FiberPolyline>& fibers,
    const std::vector<uint64_t>& excludedFiberIds,
    const std::vector<vc::atlas::FiberSideStripLineQuery>& branchLinks)
{
    uint64_t seed = 0x5349444553545249ULL;
    combineSideStripHash(seed, static_cast<uint64_t>(stripPoints.rows));
    combineSideStripHash(seed, static_cast<uint64_t>(stripPoints.cols));
    for (int row = 0; row < stripPoints.rows; ++row) {
        for (int col = 0; col < stripPoints.cols; ++col) {
            combineSideStripPointHash(seed, stripPoints(row, col));
        }
    }
    combineSideStripHash(seed, static_cast<uint64_t>(excludedFiberIds.size()));
    for (const uint64_t fiberId : excludedFiberIds) {
        combineSideStripHash(seed, fiberId);
    }
    combineSideStripHash(seed, static_cast<uint64_t>(branchLinks.size()));
    for (const auto& link : branchLinks) {
        combineSideStripHash(seed, link.fiberId);
        combineSideStripPointHash(seed, link.point);
        combineSideStripPointHash(seed, link.direction);
        combineSideStripPointHash(seed, link.connectorStart);
    }
    combineSideStripHash(seed, static_cast<uint64_t>(fibers.size()));
    for (const auto& fiber : fibers) {
        combineSideStripHash(seed, fiber.id);
        combineSideStripHash(seed, fiber.generation);
        combineSideStripHash(seed, static_cast<uint64_t>(fiber.points.size()));
        for (const auto& point : fiber.points) {
            combineSideStripPointHash(seed, point.position);
        }
        combineSideStripHash(seed, static_cast<uint64_t>(fiber.controlPoints.size()));
        for (const auto& point : fiber.controlPoints) {
            combineSideStripPointHash(seed, point);
        }
    }
    return seed;
}

std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker>
sideStripMarkersFromIntersections(
    const std::vector<vc::atlas::FiberSideStripIntersection>& intersections)
{
    std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker> markers;
    markers.reserve(intersections.size());
    for (const auto& intersection : intersections) {
        vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker marker;
        marker.point = toVec3f(intersection.point);
        marker.fiberId = intersection.fiberId;
        marker.segmentIndex = intersection.segmentIndex;
        marker.arclength = intersection.arclength;
        marker.distance = intersection.distance;
        marker.projectedBranchLink =
            intersection.source == vc::atlas::FiberSideStripIntersectionSource::BranchLink;
        if (marker.projectedBranchLink && finitePoint(intersection.connectorStart)) {
            marker.connectorStart = toVec3f(intersection.connectorStart);
        }
        markers.push_back(marker);
    }
    return markers;
}

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool finiteDirection(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
           std::sqrt(v.dot(v)) > kEpsilon;
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        return {0.0, 0.0, 0.0};
    }
    const double n = std::sqrt(v.dot(v));
    if (n <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

bool finitePoint(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

bool finitePoint(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

double polylineLengthRange(const std::vector<cv::Vec3d>& points,
                           size_t firstIndex,
                           size_t lastIndex)
{
    if (points.size() < 2 || firstIndex >= points.size() || lastIndex >= points.size() ||
        lastIndex <= firstIndex) {
        return 0.0;
    }

    double length = 0.0;
    for (size_t i = firstIndex + 1; i <= lastIndex; ++i) {
        if (!finitePoint(points[i - 1]) || !finitePoint(points[i])) {
            continue;
        }
        const cv::Vec3d delta = points[i] - points[i - 1];
        const double step = std::sqrt(delta.dot(delta));
        if (std::isfinite(step)) {
            length += step;
        }
    }
    return length;
}

double normalAlignmentErrorDegrees(const cv::Vec3d& tangent,
                                   const cv::Vec3d& normal)
{
    const cv::Vec3d unitTangent = normalizedOrZero(tangent);
    const cv::Vec3d unitNormal = normalizedOrZero(normal);
    if (!finiteDirection(unitTangent) || !finiteDirection(unitNormal)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double alignment = std::clamp(std::abs(unitTangent.dot(unitNormal)), 0.0, 1.0);
    return std::asin(alignment) * 180.0 / M_PI;
}

cv::Vec3f toVec3f(const cv::Vec3d& v)
{
    return {static_cast<float>(v[0]),
            static_cast<float>(v[1]),
            static_cast<float>(v[2])};
}

cv::Vec3d toVec3d(const cv::Vec3f& v)
{
    return {static_cast<double>(v[0]),
            static_cast<double>(v[1]),
            static_cast<double>(v[2])};
}

bool approximatelyEqual(double a, double b)
{
    return std::abs(a - b) <= 1.0e-9;
}

bool pointsApproximatelyEqual(const cv::Vec3d& a,
                              const cv::Vec3d& b,
                              double tolerance = 1.0e-6)
{
    if (!finitePoint(a) || !finitePoint(b)) {
        return false;
    }
    const cv::Vec3d delta = a - b;
    return delta.dot(delta) <= tolerance * tolerance;
}

std::optional<int> storedControlPointIndexByPosition(
    const std::vector<cv::Vec3d>& controlPoints,
    const cv::Vec3d& point)
{
    if (!finitePoint(point)) {
        return std::nullopt;
    }
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        if (pointsApproximatelyEqual(controlPoints[i], point)) {
            return static_cast<int>(i);
        }
    }
    return std::nullopt;
}

std::optional<int> matchingStoredControlPointIndex(
    const std::vector<cv::Vec3d>& controlPoints,
    int fallbackIndex,
    const cv::Vec3d& point)
{
    if (auto index = storedControlPointIndexByPosition(controlPoints, point)) {
        return index;
    }
    if (fallbackIndex < 0 ||
        static_cast<size_t>(fallbackIndex) >= controlPoints.size() ||
        !pointsApproximatelyEqual(controlPoints[static_cast<size_t>(fallbackIndex)], point)) {
        return std::nullopt;
    }
    return fallbackIndex;
}

std::optional<int> sessionControlPointIndexByPosition(
    const std::vector<vc::lasagna::LineControlPoint>& controlPoints,
    const cv::Vec3d& point)
{
    if (!finitePoint(point)) {
        return std::nullopt;
    }
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        if (pointsApproximatelyEqual(controlPoints[i].volumePoint, point)) {
            return static_cast<int>(i);
        }
    }
    return std::nullopt;
}

std::vector<int> storedIndexMapForSessionControls(
    const std::vector<vc::lasagna::LineControlPoint>& controlPoints)
{
    std::vector<size_t> order(controlPoints.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(),
                     order.end(),
                     [&controlPoints](size_t lhs, size_t rhs) {
                         return controlPoints[lhs].linePosition <
                                controlPoints[rhs].linePosition;
                     });

    std::vector<int> storedIndexForSessionIndex(controlPoints.size(), -1);
    for (size_t storedIndex = 0; storedIndex < order.size(); ++storedIndex) {
        storedIndexForSessionIndex[order[storedIndex]] = static_cast<int>(storedIndex);
    }
    return storedIndexForSessionIndex;
}

std::optional<int> storedControlPointIndexForSessionPosition(
    const std::vector<vc::lasagna::LineControlPoint>& controlPoints,
    const cv::Vec3d& point)
{
    const auto sessionIndex = sessionControlPointIndexByPosition(controlPoints, point);
    if (!sessionIndex) {
        return std::nullopt;
    }
    const auto storedIndexForSessionIndex =
        storedIndexMapForSessionControls(controlPoints);
    const int storedIndex = storedIndexForSessionIndex[static_cast<size_t>(*sessionIndex)];
    if (storedIndex < 0) {
        return std::nullopt;
    }
    return storedIndex;
}

std::optional<int> matchingSessionControlPointIndex(
    const std::vector<vc::lasagna::LineControlPoint>& controlPoints,
    int fallbackIndex,
    const cv::Vec3d& point)
{
    if (auto index = sessionControlPointIndexByPosition(controlPoints, point)) {
        return index;
    }
    if (fallbackIndex < 0 ||
        static_cast<size_t>(fallbackIndex) >= controlPoints.size() ||
        !pointsApproximatelyEqual(
            controlPoints[static_cast<size_t>(fallbackIndex)].volumePoint,
            point)) {
        return std::nullopt;
    }
    return fallbackIndex;
}

std::string fiberErrorName(const std::string& fileName)
{
    const std::string baseName = fs::path(fileName).filename().string();
    return baseName.empty() ? std::string{"<unknown>"} : baseName;
}

bool branchDirectionsCompatible(const cv::Vec3d& a,
                                const cv::Vec3d& b,
                                double tolerance = 1.0e-5)
{
    const cv::Vec3d na = normalizedOrZero(a);
    const cv::Vec3d nb = normalizedOrZero(b);
    if (!finiteDirection(na) || !finiteDirection(nb)) {
        return false;
    }
    return std::abs(std::abs(na.dot(nb)) - 1.0) <= tolerance;
}

void addUniqueFiberId(std::vector<uint64_t>& ids, uint64_t fiberId)
{
    if (fiberId == 0 ||
        std::find(ids.begin(), ids.end(), fiberId) != ids.end()) {
        return;
    }
    ids.push_back(fiberId);
}

bool branchReferencesFiber(const LineAnnotationController::FiberBranchRef& branch,
                           uint64_t fiberId,
                           const std::string& fileName)
{
    if (fiberId != 0 && branch.branchFiberId == fiberId) {
        return true;
    }
    return !fileName.empty() && branch.branchFileName == fileName;
}

bool controlPointHasBranchLink(
    const std::vector<LineAnnotationController::FiberBranchRef>& branches,
    size_t controlPointIndex)
{
    return std::any_of(
        branches.begin(),
        branches.end(),
        [controlPointIndex](const LineAnnotationController::FiberBranchRef& branch) {
            return branch.controlPointIndex == static_cast<int>(controlPointIndex);
        });
}

bool branchLinkedEndpointMatches(
    const LineAnnotationController::FiberBranchRef& lhs,
    const LineAnnotationController::FiberBranchRef& rhs)
{
    const bool sameTargetId =
        lhs.branchFiberId != 0 &&
        rhs.branchFiberId != 0 &&
        lhs.branchFiberId == rhs.branchFiberId;
    const bool sameTargetFile =
        !lhs.branchFileName.empty() &&
        lhs.branchFileName == rhs.branchFileName;
    return (sameTargetId || sameTargetFile) &&
           lhs.branchControlPointIndex == rhs.branchControlPointIndex;
}

std::optional<size_t> controlPointIndexAtLinePosition(
    const std::vector<vc::lasagna::LineControlPoint>& controls,
    double linePosition)
{
    if (!std::isfinite(linePosition)) {
        return std::nullopt;
    }
    std::optional<size_t> bestIndex;
    double bestDistance = kControlPointLabelLinePositionTolerance;
    for (size_t i = 0; i < controls.size(); ++i) {
        const double controlPosition = controls[i].linePosition;
        if (!std::isfinite(controlPosition)) {
            continue;
        }
        const double distance = std::abs(controlPosition - linePosition);
        if (distance <= bestDistance) {
            bestDistance = distance;
            bestIndex = i;
        }
    }
    return bestIndex;
}

int controlLineIndex(const vc::lasagna::LineControlPoint& control, int maxIndex)
{
    if (control.optimizedIndex >= 0) {
        return std::clamp(control.optimizedIndex, 0, maxIndex);
    }
    if (std::isfinite(control.linePosition)) {
        return std::clamp(static_cast<int>(std::llround(control.linePosition)), 0, maxIndex);
    }
    return 0;
}

std::pair<int, int> activeRangeAroundDeletedControl(
    const std::vector<vc::lasagna::LineControlPoint>& controls,
    double deletedLinePosition,
    int linePointCount,
    int spanRadius)
{
    if (linePointCount <= 0) {
        return {-1, -1};
    }
    const int maxIndex = linePointCount - 1;
    if (controls.empty() || !std::isfinite(deletedLinePosition)) {
        return {0, maxIndex};
    }

    std::vector<vc::lasagna::LineControlPoint> sortedControls;
    sortedControls.reserve(controls.size());
    for (const auto& control : controls) {
        if (std::isfinite(control.linePosition)) {
            sortedControls.push_back(control);
        }
    }
    if (sortedControls.empty()) {
        return {0, maxIndex};
    }
    std::stable_sort(sortedControls.begin(),
                     sortedControls.end(),
                     [](const auto& a, const auto& b) {
                         return a.linePosition < b.linePosition;
                     });

    const auto insertion = std::lower_bound(
        sortedControls.begin(),
        sortedControls.end(),
        deletedLinePosition,
        [](const vc::lasagna::LineControlPoint& control, double position) {
            return control.linePosition < position;
        });
    const int insertionIndex = static_cast<int>(std::distance(sortedControls.begin(), insertion));
    int leftControl = insertionIndex - 1;
    int rightControl = insertionIndex;

    bool includeLeftOpenEnd = leftControl < 0;
    bool includeRightOpenEnd = rightControl >= static_cast<int>(sortedControls.size());
    if (!includeLeftOpenEnd) {
        for (int span = 0; span < spanRadius && leftControl > 0; ++span) {
            --leftControl;
        }
    }
    if (!includeRightOpenEnd) {
        for (int span = 0;
             span < spanRadius && rightControl + 1 < static_cast<int>(sortedControls.size());
             ++span) {
            ++rightControl;
        }
    }

    int activeStart = includeLeftOpenEnd
        ? 0
        : controlLineIndex(sortedControls[static_cast<size_t>(leftControl)], maxIndex);
    int activeEnd = includeRightOpenEnd
        ? maxIndex
        : controlLineIndex(sortedControls[static_cast<size_t>(rightControl)], maxIndex);
    if (activeEnd < activeStart) {
        std::swap(activeStart, activeEnd);
    }
    return {activeStart, activeEnd};
}

QString controlPointInfoText(const std::vector<vc::lasagna::LineControlPoint>& controls,
                             double linePosition)
{
    const auto index = controlPointIndexAtLinePosition(controls, linePosition);
    if (!index) {
        return {};
    }
    const auto& point = controls[*index].volumePoint;
    return QStringLiteral("CP %1  pos [%2, %3, %4]")
        .arg(static_cast<qulonglong>(*index))
        .arg(point[0], 0, 'f', 2)
        .arg(point[1], 0, 'f', 2)
        .arg(point[2], 0, 'f', 2);
}

void positionControlPointInfoLabel(QLabel* label)
{
    if (!label) {
        return;
    }
    auto* parent = qobject_cast<QWidget*>(label->parentWidget());
    if (!parent) {
        return;
    }
    constexpr int margin = 8;
    label->adjustSize();
    label->move(margin, std::max(margin, parent->height() - label->height() - margin));
    label->raise();
}

QLabel* createControlPointInfoLabel(
    QWidget* parent,
    const std::vector<vc::lasagna::LineControlPoint>& controls,
    double linePosition)
{
    if (!parent) {
        return nullptr;
    }
    auto* label = new QLabel(parent);
    label->setObjectName(QStringLiteral("intersectionControlPointInfo"));
    label->setAttribute(Qt::WA_TransparentForMouseEvents);
    label->setStyleSheet(QStringLiteral(
        "QLabel#intersectionControlPointInfo {"
        " color: white;"
        " background: rgba(0, 0, 0, 180);"
        " padding: 3px 6px;"
        " border-radius: 3px;"
        " font-weight: 600;"
        "}"));
    const QString text = controlPointInfoText(controls, linePosition);
    label->setText(text);
    label->setVisible(!text.isEmpty());
    positionControlPointInfoLabel(label);
    return label;
}

void updateControlPointInfoLabel(QLabel* label,
                                 const std::vector<vc::lasagna::LineControlPoint>& controls,
                                 double linePosition)
{
    if (!label) {
        return;
    }
    const QString text = controlPointInfoText(controls, linePosition);
    label->setText(text);
    label->setVisible(!text.isEmpty());
    positionControlPointInfoLabel(label);
}

std::optional<std::string> normalizedFiberJsonFileNameInput(const QString& input, QString* error)
{
    QString fileName = input.trimmed();
    if (fileName.isEmpty()) {
        if (error) {
            *error = QObject::tr("Enter a JSON file name.");
        }
        return std::nullopt;
    }
    if (fileName.contains(QChar('/')) || fileName.contains(QChar('\\'))) {
        if (error) {
            *error = QObject::tr("The file name cannot contain folders or path separators.");
        }
        return std::nullopt;
    }

    constexpr int kJsonSuffixLength = 5;
    if (fileName.endsWith(QStringLiteral(".json"), Qt::CaseInsensitive)) {
        fileName = fileName.left(fileName.size() - kJsonSuffixLength) + QStringLiteral(".json");
    } else {
        fileName += QStringLiteral(".json");
    }

    const QString stem = fileName.left(fileName.size() - kJsonSuffixLength).trimmed();
    if (stem.isEmpty() || stem == QStringLiteral(".") || stem == QStringLiteral("..")) {
        if (error) {
            *error = QObject::tr("Enter a file name before the .json extension.");
        }
        return std::nullopt;
    }
    if (fileName == QStringLiteral(".") || fileName == QStringLiteral("..")) {
        if (error) {
            *error = QObject::tr("Enter a valid JSON file name.");
        }
        return std::nullopt;
    }
    return fileName.toStdString();
}

std::optional<std::string> normalizedFiberTagInput(const QString& input, QString* error)
{
    const QString tag = input.trimmed();
    if (tag.isEmpty()) {
        if (error) {
            *error = QObject::tr("Enter a tag name.");
        }
        return std::nullopt;
    }
    return tag.toStdString();
}

void addUniqueSorted(std::vector<std::string>& values, const std::string& value)
{
    if (value.empty()) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) != values.end()) {
        return;
    }
    values.push_back(value);
    std::sort(values.begin(), values.end());
}

std::vector<std::string> normalizedFiberTagsFromJson(const nlohmann::json& tags)
{
    std::vector<std::string> result;
    if (!tags.is_array()) {
        throw std::runtime_error("tags must be an array");
    }
    for (const auto& tag : tags) {
        if (!tag.is_string()) {
            throw std::runtime_error("tags entries must be strings");
        }
        addUniqueSorted(result, QString::fromStdString(tag.get<std::string>()).trimmed().toStdString());
    }
    return result;
}

cv::Vec3d interpolatedPointAtLinePosition(const std::vector<cv::Vec3d>& points,
                                          double linePosition)
{
    if (points.empty() || !std::isfinite(linePosition)) {
        return {0.0, 0.0, 0.0};
    }
    linePosition = std::clamp(linePosition, 0.0, static_cast<double>(points.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(points.size()) - 1);
    const double t = linePosition - static_cast<double>(lower);
    return points[static_cast<size_t>(lower)] * (1.0 - t) +
           points[static_cast<size_t>(upper)] * t;
}

cv::Vec3d tangentAtLinePosition(const std::vector<cv::Vec3d>& points,
                                double linePosition)
{
    if (points.size() < 2 || !std::isfinite(linePosition)) {
        return {1.0, 0.0, 0.0};
    }
    linePosition = std::clamp(linePosition, 0.0, static_cast<double>(points.size() - 1));
    int lower = static_cast<int>(std::floor(linePosition));
    int upper = std::min<int>(lower + 1, static_cast<int>(points.size()) - 1);
    if (lower == upper && lower > 0) {
        --lower;
    }
    cv::Vec3d tangent = points[static_cast<size_t>(upper)] - points[static_cast<size_t>(lower)];
    tangent = normalizedOrZero(tangent);
    return finiteDirection(tangent) ? tangent : cv::Vec3d{1.0, 0.0, 0.0};
}

cv::Vec3d endpointTangentFromLinePoints(const std::vector<cv::Vec3d>& linePoints,
                                        const cv::Vec3d& controlPoint,
                                        const cv::Vec3d& fallback = {0.0, 0.0, 0.0})
{
    if (linePoints.size() >= 2 && finitePoint(controlPoint)) {
        const size_t index =
            vc3d::fiber_slice::nearestLinePointIndex(linePoints, controlPoint);
        const cv::Vec3d tangent =
            tangentAtLinePosition(linePoints, static_cast<double>(index));
        if (finiteDirection(tangent)) {
            return normalizedOrZero(tangent);
        }
    }
    return finiteDirection(fallback) ? normalizedOrZero(fallback) : cv::Vec3d{0.0, 0.0, 0.0};
}

std::vector<cv::Vec3d> linePointPositions(const vc::lasagna::LineModel& line)
{
    std::vector<cv::Vec3d> points;
    points.reserve(line.points.size());
    for (const auto& point : line.points) {
        points.push_back(point.position);
    }
    return points;
}

cv::Vec3f interpolatedUpAtLinePosition(const std::vector<cv::Vec3f>& upVectors,
                                       double linePosition,
                                       const cv::Vec3d& tangent)
{
    if (upVectors.empty() || !std::isfinite(linePosition)) {
        return {0.0f, 1.0f, 0.0f};
    }
    linePosition = std::clamp(linePosition, 0.0, static_cast<double>(upVectors.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(upVectors.size()) - 1);
    cv::Vec3f lowerUp = upVectors[static_cast<size_t>(lower)];
    cv::Vec3f upperUp = upVectors[static_cast<size_t>(upper)];
    if (!finitePoint(lowerUp) || !finitePoint(upperUp)) {
        return {0.0f, 1.0f, 0.0f};
    }
    if (lowerUp.dot(upperUp) < 0.0f) {
        upperUp *= -1.0f;
    }
    const float t = static_cast<float>(linePosition - static_cast<double>(lower));
    cv::Vec3d up = toVec3d(lowerUp * (1.0f - t) + upperUp * t);
    up -= tangent * up.dot(tangent);
    up = normalizedOrZero(up);
    return finiteDirection(up) ? toVec3f(up) : cv::Vec3f{0.0f, 1.0f, 0.0f};
}

std::optional<cv::Vec2f> stripLinePositionToSurfacePoint(QuadSurface* surface,
                                                         double linePosition)
{
    if (!surface || !std::isfinite(linePosition)) {
        return std::nullopt;
    }
    const auto* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }
    const cv::Vec2f scale = surface->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f) {
        return std::nullopt;
    }
    const float surfaceX = (static_cast<float>(linePosition) -
                            static_cast<float>(points->cols) / 2.0f) / scale[0];
    const float centerRow = static_cast<float>(points->rows / 2);
    const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
    return cv::Vec2f{surfaceX, surfaceY};
}

void frameStripLineSpan(CChunkedVolumeViewer* viewer,
                        QuadSurface* surface,
                        double firstLinePosition,
                        double secondLinePosition)
{
    if (!viewer || !surface ||
        !std::isfinite(firstLinePosition) ||
        !std::isfinite(secondLinePosition)) {
        return;
    }
    const auto first = stripLinePositionToSurfacePoint(surface, firstLinePosition);
    const auto second = stripLinePositionToSurfacePoint(surface, secondLinePosition);
    if (!first || !second) {
        return;
    }
    const double span = std::abs(static_cast<double>((*second)[0] - (*first)[0]));
    if (!std::isfinite(span) || span <= 1.0e-6) {
        return;
    }
    auto* view = viewer->graphicsView();
    const int viewportWidth = view && view->viewport() ? view->viewport()->width() : 0;
    if (viewportWidth <= 0) {
        return;
    }
    constexpr double kViewportFill = 0.86;
    auto camera = viewer->cameraState();
    camera.surfacePtrX = static_cast<float>((static_cast<double>((*first)[0]) +
                                             static_cast<double>((*second)[0])) * 0.5);
    camera.surfacePtrY = static_cast<float>((static_cast<double>((*first)[1]) +
                                             static_cast<double>((*second)[1])) * 0.5);
    camera.scale = static_cast<float>(std::clamp(kViewportFill * static_cast<double>(viewportWidth) / span,
                                                 0.01,
                                                 100000.0));
    viewer->applyCameraState(camera, true);
}

std::vector<vc3d::line_annotation::GeneratedOverlay::ControlPointMarker>
generatedControlMarkers(
    const std::vector<vc::lasagna::LineControlPoint>& controls,
    const std::vector<LineAnnotationController::FiberBranchRef>& branches = {})
{
    struct BranchLinkTarget {
        uint64_t fiberId = 0;
        int controlPointIndex = -1;
        bool pending = false;
    };
    std::unordered_map<size_t, std::vector<BranchLinkTarget>> branchesByControl;
    branchesByControl.reserve(branches.size());
    for (const auto& branch : branches) {
        if (branch.controlPointIndex < 0 || branch.branchFiberId == 0) {
            continue;
        }
        const size_t controlIndex = static_cast<size_t>(branch.controlPointIndex);
        if (controlIndex >= controls.size()) {
            continue;
        }
        auto& targets = branchesByControl[controlIndex];
        const auto duplicate = std::find_if(
            targets.begin(),
            targets.end(),
            [&branch](const BranchLinkTarget& target) {
                return target.fiberId == branch.branchFiberId &&
                       target.controlPointIndex == branch.branchControlPointIndex;
            });
        if (duplicate == targets.end()) {
            targets.push_back({branch.branchFiberId,
                               branch.branchControlPointIndex,
                               branch.pending});
        } else {
            duplicate->pending = duplicate->pending || branch.pending;
        }
    }

    std::vector<vc3d::line_annotation::GeneratedOverlay::ControlPointMarker> markers;
    markers.reserve(controls.size());
    for (size_t i = 0; i < controls.size(); ++i) {
        const auto& control = controls[i];
        vc3d::line_annotation::GeneratedOverlay::ControlPointMarker marker;
        marker.point = toVec3f(control.volumePoint);
        marker.linePosition = control.linePosition;
        marker.controlIndex = i;
        marker.isSeed = control.isSeed;
        if (auto it = branchesByControl.find(i); it != branchesByControl.end()) {
            std::sort(it->second.begin(),
                      it->second.end(),
                      [](const BranchLinkTarget& lhs, const BranchLinkTarget& rhs) {
                          if (lhs.fiberId != rhs.fiberId) {
                              return lhs.fiberId < rhs.fiberId;
                          }
                          return lhs.controlPointIndex < rhs.controlPointIndex;
                      });
            for (const auto& target : it->second) {
                marker.branchIds.push_back(target.fiberId);
                marker.branchLinks.push_back(
                    {target.fiberId, target.controlPointIndex, target.pending});
                marker.hasPendingLinks = marker.hasPendingLinks || target.pending;
            }
            marker.branchIds.erase(std::unique(marker.branchIds.begin(), marker.branchIds.end()),
                                   marker.branchIds.end());
            marker.hasBranches = !marker.branchLinks.empty();
        }
        markers.push_back(std::move(marker));
    }
    return markers;
}

std::vector<vc3d::line_annotation::GeneratedOverlay::BranchLinkMarker>
generatedBranchLinkMarkers(const std::vector<LineAnnotationController::FiberBranchRef>& branches)
{
    std::vector<vc3d::line_annotation::GeneratedOverlay::BranchLinkMarker> markers;
    markers.reserve(branches.size());
    for (const auto& branch : branches) {
        if (!finitePoint(branch.controlPointPosition) ||
            !finitePoint(branch.branchControlPointPosition) ||
            !finiteDirection(branch.controlPointDirection) ||
            !finiteDirection(branch.branchControlPointDirection)) {
            continue;
        }
        vc3d::line_annotation::GeneratedOverlay::BranchLinkMarker marker;
        marker.linkedFiberId = branch.branchFiberId;
        marker.localControlPoint = toVec3f(branch.controlPointPosition);
        marker.linkedControlPoint = toVec3f(branch.branchControlPointPosition);
        marker.localDirection = toVec3f(normalizedOrZero(branch.controlPointDirection));
        marker.linkedDirection =
            toVec3f(normalizedOrZero(branch.branchControlPointDirection));
        marker.planePoint = marker.linkedControlPoint;
        marker.estimated = false;
        markers.push_back(marker);
    }
    return markers;
}

void remapBranchControlPointIndices(
    const std::vector<vc::lasagna::LineControlPoint>& oldControls,
    const std::vector<vc::lasagna::LineControlPoint>& newControls,
    std::vector<LineAnnotationController::FiberBranchRef>& branches)
{
    if (branches.empty()) {
        return;
    }

    constexpr double kMaxMatchDistanceSq = 1.0e-6;
    for (auto& branch : branches) {
        if (branch.controlPointIndex < 0 ||
            static_cast<size_t>(branch.controlPointIndex) >= oldControls.size()) {
            branch.controlPointIndex = -1;
            continue;
        }

        const cv::Vec3d oldPoint = oldControls[static_cast<size_t>(branch.controlPointIndex)].volumePoint;
        int bestIndex = -1;
        double bestDistanceSq = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < newControls.size(); ++i) {
            const cv::Vec3d delta = newControls[i].volumePoint - oldPoint;
            const double distanceSq = delta.dot(delta);
            if (distanceSq < bestDistanceSq) {
                bestDistanceSq = distanceSq;
                bestIndex = static_cast<int>(i);
            }
        }
        branch.controlPointIndex = bestDistanceSq <= kMaxMatchDistanceSq ? bestIndex : -1;
    }
    branches.erase(std::remove_if(branches.begin(),
                                  branches.end(),
                                  [](const LineAnnotationController::FiberBranchRef& branch) {
                                      return branch.controlPointIndex < 0;
                                  }),
                   branches.end());
}

std::vector<vc3d::line_annotation::GeneratedOverlay::PredSnapMarker>
generatedPredSnapMarkers(const std::vector<vc::lasagna::LineControlPoint>& controls,
                         const vc::atlas::AtlasPredSnapSet& predSnapSet)
{
    std::unordered_map<std::string, const vc::atlas::AtlasPredSnapPoint*> snapsByControl;
    snapsByControl.reserve(predSnapSet.points.size());
    for (const auto& point : predSnapSet.points) {
        if (point.predSnapPoint &&
            (point.source == vc::atlas::AtlasPredSnapSource::Manual ||
             point.source == vc::atlas::AtlasPredSnapSource::Optimized)) {
            snapsByControl[vc::atlas::atlasPredSnapControlPointKey(point.controlPoint)] = &point;
        }
    }

    std::vector<vc3d::line_annotation::GeneratedOverlay::PredSnapMarker> markers;
    markers.reserve(controls.size());
    for (size_t i = 0; i < controls.size(); ++i) {
        const auto& control = controls[i];
        const auto it = snapsByControl.find(
            vc::atlas::atlasPredSnapControlPointKey(control.volumePoint));
        if (it == snapsByControl.end() || !it->second || !it->second->predSnapPoint) {
            continue;
        }
        vc3d::line_annotation::GeneratedOverlay::PredSnapMarker marker;
        marker.controlPoint = toVec3f(control.volumePoint);
        marker.snapPoint = toVec3f(*it->second->predSnapPoint);
        marker.linePosition = control.linePosition;
        marker.controlIndex = i;
        marker.manual = it->second->source == vc::atlas::AtlasPredSnapSource::Manual;
        markers.push_back(marker);
    }
    return markers;
}

std::vector<cv::Vec3f> generatedLinePoints(const std::vector<cv::Vec3d>& points)
{
    std::vector<cv::Vec3f> converted;
    converted.reserve(points.size());
    for (const auto& point : points) {
        converted.push_back(toVec3f(point));
    }
    return converted;
}

nlohmann::json pointToJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

nlohmann::json controlsToJson(const std::vector<vc::lasagna::LineControlPoint>& controls)
{
    nlohmann::json array = nlohmann::json::array();
    for (const auto& control : controls) {
        array.push_back({
            {"line_position", control.linePosition},
            {"optimized_index", control.optimizedIndex},
            {"is_seed", control.isSeed},
            {"xyz", pointToJson(control.volumePoint)},
        });
    }
    return array;
}

nlohmann::json linePointsToJson(const std::vector<cv::Vec3d>& points)
{
    nlohmann::json array = nlohmann::json::array();
    for (const auto& point : points) {
        array.push_back(pointToJson(point));
    }
    return array;
}

nlohmann::json linePointsToJson(const vc::lasagna::LineModel& line)
{
    nlohmann::json array = nlohmann::json::array();
    for (const auto& point : line.points) {
        array.push_back(pointToJson(point.position));
    }
    return array;
}

std::string sanitizedEventName(std::string event)
{
    for (char& ch : event) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
            ch = '_';
        }
    }
    return event.empty() ? "event" : event;
}

std::string sanitizedProjectFiberDirName(const fs::path& projectPath,
                                         const fs::path& volpkgRoot)
{
    std::string name = projectPath.empty()
        ? volpkgRoot.filename().string()
        : projectPath.filename().string();
    for (char& ch : name) {
        const auto c = static_cast<unsigned char>(ch);
        if (!std::isalnum(c) && ch != '.' && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    while (!name.empty() && name.front() == '_') {
        name.erase(name.begin());
    }
    while (!name.empty() && name.back() == '_') {
        name.pop_back();
    }
    return name.empty() ? "project" : name;
}

void writeLineDebugJson(const std::string& eventName,
                        const std::vector<vc::lasagna::LineControlPoint>& controls,
                        const nlohmann::json& linePoints,
                        const vc::lasagna::LineOptimizationReport* report = nullptr)
{
    const char* debugDir = std::getenv("VC3D_LINE_DEBUG_DIR");
    if (!debugDir || *debugDir == '\0') {
        return;
    }

    static std::atomic<int> sequence{0};
    const int id = sequence.fetch_add(1, std::memory_order_relaxed) + 1;

    std::error_code ec;
    fs::create_directories(debugDir, ec);
    if (ec) {
        Logger()->warn("Could not create VC3D_LINE_DEBUG_DIR {}: {}", debugDir, ec.message());
        return;
    }

    nlohmann::json root;
    root["event"] = eventName;
    root["control_points"] = controlsToJson(controls);
    root["line_points"] = linePoints;
    if (report) {
        root["optimization_report"] = {
            {"initial_cost", report->initialCost},
            {"final_cost", report->finalCost},
            {"iterations", report->iterations},
            {"valid_normal_samples", report->validNormalSamples},
            {"invalid_normal_samples", report->invalidNormalSamples},
            {"converged", report->converged},
            {"normal_prefetch_calls", report->normalPrefetchCalls},
            {"ceres_solve_ms", report->ceresSolveMs},
            {"normal_chunk_prefetch_ms", report->normalChunkPrefetchMs},
            {"normal_materialize_ms", report->normalMaterializeMs},
            {"total_ms", report->totalMs},
            {"message", report->message},
        };
        root["optimization_report"]["losses"] = nlohmann::json::array();
        for (const auto& loss : report->finalLosses) {
            root["optimization_report"]["losses"].push_back({
                {"name", loss.name},
                {"weight", loss.weight},
                {"residuals", loss.residuals},
                {"raw_cost", loss.rawCost},
                {"weighted_cost", loss.weightedCost},
            });
        }
    }

    std::ostringstream fileName;
    fileName.imbue(std::locale::classic());
    fileName << "line_edit_" << std::setw(4) << std::setfill('0') << id
             << '_' << sanitizedEventName(eventName) << ".json";
    const fs::path path = fs::path(debugDir) / fileName.str();
    std::ofstream output(path);
    if (!output.good()) {
        Logger()->warn("Could not write line debug JSON {}", path.string());
        return;
    }
    output << root.dump(2) << '\n';
}

cv::Vec3d pointFromJson(const nlohmann::json& value)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("Point must be a [x, y, z] array");
    }
    cv::Vec3d point{
        value.at(0).get<double>(),
        value.at(1).get<double>(),
        value.at(2).get<double>(),
    };
    if (!finitePoint(point)) {
        throw std::runtime_error("Point contains non-finite coordinates");
    }
    return point;
}

cv::Vec3d initialTangentForMode(
    LineAnnotationController::InitialDirectionMode mode,
    const cv::Vec3d& sourceSliceNormal,
    const vc::lasagna::NormalSample& seedNormal)
{
    const cv::Vec3d sliceNormal = normalizedOrZero(sourceSliceNormal);
    const cv::Vec3d gtNormal = seedNormal.valid
        ? normalizedOrZero(seedNormal.normal)
        : cv::Vec3d{0.0, 0.0, 0.0};

    if (!finiteDirection(sliceNormal) || !finiteDirection(gtNormal)) {
        return {0.0, 0.0, 0.0};
    }

    if (mode == LineAnnotationController::InitialDirectionMode::ZInOut) {
        return normalizedOrZero(sliceNormal - gtNormal * sliceNormal.dot(gtNormal));
    }
    return normalizedOrZero(sliceNormal.cross(gtNormal));
}

void validateLasagnaManifest(const fs::path& manifestPath)
{
    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
}

LineAnnotationController::OptimizationTaskResult optimizeLineWithSampler(
    fs::path manifestPath,
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    int initialCenterlineLengthVx,
    bool forceFullOptimization,
    int activeStart,
    int activeEnd,
    const vc::lasagna::NormalSampler& sampler);

LineAnnotationController::OptimizationTaskResult optimizeLineFromManifest(
    fs::path manifestPath,
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    int initialCenterlineLengthVx,
    bool forceFullOptimization,
    int activeStart,
    int activeEnd)
{
    vc::lasagna::LasagnaDataset dataset =
        vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    return optimizeLineWithSampler(std::move(manifestPath),
                                   std::move(controlPoints),
                                   std::move(initialLinePoints),
                                   sourceSliceNormal,
                                   directionMode,
                                   initialCenterlineLengthVx,
                                   forceFullOptimization,
                                   activeStart,
                                   activeEnd,
                                   sampler);
}

LineAnnotationController::OptimizationTaskResult optimizeLineWithSampler(
    fs::path manifestPath,
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    int initialCenterlineLengthVx,
    bool forceFullOptimization,
    int activeStart,
    int activeEnd,
    const vc::lasagna::NormalSampler& sampler)
{
    LineAnnotationController::OptimizationTaskResult task;
    task.manifestPath = std::move(manifestPath);
    task.controlPoints = std::move(controlPoints);
    if (!task.controlPoints.empty()) {
        const auto seedIt = std::find_if(task.controlPoints.begin(),
                                         task.controlPoints.end(),
                                         [](const vc::lasagna::LineControlPoint& control) {
                                             return control.isSeed;
                                         });
        task.seedPoint = (seedIt == task.controlPoints.end()
            ? task.controlPoints.front()
            : *seedIt).volumePoint;
    }
    task.sourceSliceNormal = sourceSliceNormal;
    task.initialDirectionMode = directionMode;
    task.eventName = initialLinePoints.empty()
        ? "seed"
        : (forceFullOptimization ? "full_reinit_reopt" : "control_optimization");
    try {
        vc::lasagna::LineOptimizer optimizer(sampler);
        vc::lasagna::LineOptimizationConfig config;
        const auto discretization = initialLineDiscretization(initialCenterlineLengthVx);
        config.segmentsPerSide = discretization.segmentsPerSide;
        config.segmentLength = discretization.segmentLength;
        config.straightnessWeight = 0.1;
        config.tangentStraightnessWeight = 5.0;
        config.samplesPerSegment = 1;
        config.maxIterations = 1000;
        config.differentiableNormalSampling = true;
        config.printSolverProgress = false;
        (void)sampler.prefetchNormalSamples({task.seedPoint}, false);
        config.initialTangent = initialTangentForMode(
            directionMode,
            sourceSliceNormal,
            sampler.sampleNormal(task.seedPoint));
        config.useInitialTangent = finiteDirection(config.initialTangent);
        config.tangentGuideVector = normalizedOrZero(sourceSliceNormal);
        config.tangentGuideWeight = 1.0;
        config.tangentGuideMode = directionMode == LineAnnotationController::InitialDirectionMode::ZInOut
            ? vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane
            : vc::lasagna::LineOptimizationConfig::TangentGuideMode::CrossVectorWithNormal;
        if (forceFullOptimization && initialLinePoints.size() >= 2 && task.controlPoints.size() >= 2) {
            std::vector<int> fixedIndices;
            fixedIndices.reserve(task.controlPoints.size());
            int displayFrameAnchorIndex = static_cast<int>(initialLinePoints.size() / 2);
            for (const auto& control : task.controlPoints) {
                if (!std::isfinite(control.linePosition)) {
                    continue;
                }
                const int index = std::clamp(static_cast<int>(std::llround(control.linePosition)),
                                             0,
                                             static_cast<int>(initialLinePoints.size()) - 1);
                fixedIndices.push_back(index);
                if (control.isSeed) {
                    displayFrameAnchorIndex = index;
                }
            }
            auto reinitialized =
                optimizer.reinitializeAndOptimizeExistingLine(std::move(initialLinePoints),
                                                              task.controlPoints,
                                                              std::move(fixedIndices),
                                                              displayFrameAnchorIndex,
                                                              config);
            if (reinitialized.failed) {
                task.ok = false;
                task.error = reinitialized.failureReason;
                return task;
            }
            task.result = std::move(reinitialized.optimization);
        } else if (!forceFullOptimization && initialLinePoints.size() >= 2) {
            std::vector<int> fixedIndices;
            fixedIndices.reserve(task.controlPoints.size());
            int displayFrameAnchorIndex = static_cast<int>(initialLinePoints.size() / 2);
            for (const auto& control : task.controlPoints) {
                if (!std::isfinite(control.linePosition)) {
                    continue;
                }
                const int index = std::clamp(static_cast<int>(std::llround(control.linePosition)),
                                             0,
                                             static_cast<int>(initialLinePoints.size()) - 1);
                initialLinePoints[static_cast<size_t>(index)] = control.volumePoint;
                fixedIndices.push_back(index);
                if (control.isSeed) {
                    displayFrameAnchorIndex = index;
                }
            }
            const bool hasLocalRange = activeStart >= 0 && activeEnd >= activeStart;
            const std::string candidateName = !hasLocalRange
                ? "existing-line+global"
                : "existing-line+local";
            task.result = optimizer.optimizeExistingLine(std::move(initialLinePoints),
                                                         std::move(fixedIndices),
                                                         displayFrameAnchorIndex,
                                                         config,
                                                         activeStart,
                                                         activeEnd,
                                                         candidateName);
        } else {
            task.result = optimizer.optimizeFromControlPoints(task.controlPoints, config);
        }
        task.ok = true;
    } catch (const std::exception& ex) {
        task.ok = false;
        task.error = ex.what();
    } catch (...) {
        task.ok = false;
        task.error = "Unknown Lasagna line optimization error.";
    }
    return task;
}

} // namespace

LineAnnotationController::LineAnnotationController(CState* state,
                                                   ViewerManager* viewerManager,
                                                   QWidget* parentWidget,
                                                   QObject* parent)
    : QObject(parent)
    , _state(state)
    , _viewerManager(viewerManager)
    , _parentWidget(parentWidget)
    , _fiberSliceOverlay(std::make_unique<FiberSliceOverlayController>())
    , _datasetPicker([this](QWidget* parent, const fs::path& startDir) {
        return pickDataset(parent, startDir);
    })
    , _optimizationTaskFactory([](fs::path manifestPath,
                                  std::vector<vc::lasagna::LineControlPoint> controlPoints,
                                  std::vector<cv::Vec3d> initialLinePoints,
                                  cv::Vec3d sourceSliceNormal,
                                  InitialDirectionMode directionMode,
                                  int initialCenterlineLengthVx,
                                  bool forceFullOptimization,
                                  int activeStart,
                                  int activeEnd) {
        return optimizeLineFromManifest(std::move(manifestPath),
                                        std::move(controlPoints),
                                        std::move(initialLinePoints),
                                        sourceSliceNormal,
                                        directionMode,
                                        initialCenterlineLengthVx,
                                        forceFullOptimization,
                                        activeStart,
                                        activeEnd);
    })
{
    if (_state) {
        connect(_state,
                &CState::surfaceChanged,
                this,
                &LineAnnotationController::onSurfaceChanged);
        connect(_state,
                &CState::vpkgChanged,
                this,
                &LineAnnotationController::onVolumePackageChanged);
        if (_state->vpkg()) {
            loadFibersForCurrentPackage();
        }
    }
}

LineAnnotationController::~LineAnnotationController()
{
    waitForFiberSaves();
}

void LineAnnotationController::setDatasetPickerForTesting(DatasetPicker picker)
{
    _datasetPicker = std::move(picker);
}

void LineAnnotationController::setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory)
{
    _optimizationTaskFactory = std::move(factory);
}

void LineAnnotationController::setVolumeSelectorFactory(VolumeSelectorFactory factory)
{
    _volumeSelectorFactory = std::move(factory);
}

void LineAnnotationController::setSurfacePanel(SurfacePanelController* panel)
{
    _surfacePanel = panel;
}

void LineAnnotationController::setCurrentAtlasDirectory(std::optional<fs::path> atlasDir)
{
    _currentAtlasDir = std::move(atlasDir);
}

bool LineAnnotationController::canLaunchFromViewer(const CChunkedVolumeViewer* viewer) const
{
    if (!viewer || !_state || !_viewerManager) {
        return false;
    }
    auto* surface = viewer->currentSurface();
    if (dynamic_cast<PlaneSurface*>(surface)) {
        return true;
    }
    if (!dynamic_cast<QuadSurface*>(surface)) {
        return false;
    }
    const std::string surfaceName = viewer->surfName();
    if (surfaceName == "segmentation" || surfaceName.rfind("line-", 0) == 0) {
        return true;
    }
    return std::any_of(_panes.begin(),
                       _panes.end(),
                       [&surfaceName](const PaneRecord& pane) {
                           return pane.session &&
                                  std::find(pane.session->generatedSurfaceNames.begin(),
                                            pane.session->generatedSurfaceNames.end(),
                                            surfaceName) !=
                                      pane.session->generatedSurfaceNames.end();
                       });
}

void LineAnnotationController::launchFromViewer(CChunkedVolumeViewer* viewer, const QPointF& /*scenePoint*/)
{
    if (!canLaunchFromViewer(viewer)) {
        return;
    }

    auto camera = viewer->cameraState();
    SourceKind sourceKind = SourceKind::Plane;
    std::shared_ptr<Surface> sourceSurface;
    cv::Vec3d sourceSliceNormal{
        camera.zOffsetWorldDir[0],
        camera.zOffsetWorldDir[1],
        camera.zOffsetWorldDir[2],
    };

    if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
        auto clone = std::make_shared<PlaneSurface>(*plane);
        const cv::Vec3f normal = plane->normal({0, 0, 0});
        sourceSliceNormal = {normal[0], normal[1], normal[2]};
        if (std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
            std::isfinite(normal[2]) && cv::norm(normal) > 0.0f) {
            clone->setOrigin(plane->origin() + normal * viewer->normalOffset());
        }
        camera.zOffset = 0.0f;
        camera.zOffsetWorldDir = {0, 0, 0};
        sourceSurface = clone;
    } else {
        sourceKind = SourceKind::Segmentation;
        sourceSurface = _state->surface("segmentation");
    }

    auto session = std::make_shared<LineAnnotationSession>();
    const std::string surfaceName = nextSurfaceName();
    (void)launchSession(sourceKind,
                        surfaceName,
                        std::move(sourceSurface),
                        camera,
                        sourceSliceNormal,
                        std::move(session));
}

void LineAnnotationController::launchFromViewerAtPoint(CChunkedVolumeViewer* viewer,
                                                       const QPointF& scenePoint,
                                                       bool replaceOwningAnnotation)
{
    if (!canLaunchFromViewer(viewer)) {
        return;
    }

    const auto sample = viewer->sampleSceneVolume(scenePoint);
    if (!sample) {
        return;
    }
    cv::Vec3f normal = sample->normal;
    if (!std::isfinite(normal[0]) ||
        !std::isfinite(normal[1]) ||
        !std::isfinite(normal[2]) ||
        cv::norm(normal) <= 0.0f) {
        normal = {0.0f, 0.0f, 1.0f};
    }
    normal *= 1.0f / cv::norm(normal);

    (void)replaceOwningAnnotation;

    auto surfaceName = nextSurfaceName();
    CChunkedVolumeViewer::CameraState camera;
    auto sourceSurface = std::make_shared<PlaneSurface>(sample->position, normal);
    auto session = std::make_shared<LineAnnotationSession>();
    if (!launchSession(SourceKind::Plane,
                       surfaceName,
                       std::move(sourceSurface),
                       camera,
                       cv::Vec3d{normal[0], normal[1], normal[2]},
                       session,
                       true)) {
        return;
    }
    handleLineSeed(surfaceName, sample->position, InitialDirectionMode::ZInOut);
}

bool LineAnnotationController::prepareForUserFacingLineAnnotationOpen()
{
    struct DialogRecord {
        std::string surfaceName;
        QPointer<LineAnnotationDialog> dialog;
        std::shared_ptr<LineAnnotationSession> session;
    };

    std::vector<DialogRecord> dialogs;
    dialogs.reserve(_panes.size());
    for (const auto& pane : _panes) {
        if (!pane.dialog) {
            continue;
        }
        dialogs.push_back({pane.surfaceName, pane.dialog, pane.session});
    }
    if (dialogs.empty()) {
        return true;
    }

    for (const auto& record : dialogs) {
        if (record.session &&
            record.session->taskState == LineAnnotationSession::TaskState::Running) {
            showError(tr("Line optimization is already running."));
            return false;
        }
    }

    for (const auto& record : dialogs) {
        if (!record.session) {
            continue;
        }
        auto& session = *record.session;
        const bool suppressBeforeFinalize = session.suppressFiberSave;
        if (!suppressBeforeFinalize && needsFinalOptimization(session)) {
            session.suppressFiberSave = true;
        }
        if (!finalizeSessionOptimizationSynchronously(session, false)) {
            session.suppressFiberSave = suppressBeforeFinalize;
            return false;
        }
        session.suppressFiberSave = suppressBeforeFinalize;
        if (!session.suppressFiberSave &&
            session.taskState == LineAnnotationSession::TaskState::Succeeded &&
            !session.optimizedLine.points.empty() &&
            !session.controlPoints.empty()) {
            saveSessionAsFiber(session);
            session.suppressFiberSave = true;
        }
    }

    for (const auto& record : dialogs) {
        if (record.dialog) {
            record.dialog->setCloseAfterFinalizationAllowed(true);
            record.dialog->close();
        }
        cleanupSurfaceName(record.surfaceName);
    }
    return true;
}

bool LineAnnotationController::launchSession(LineAnnotationController::SourceKind sourceKind,
                                             const std::string& surfaceName,
                                             std::shared_ptr<Surface> sourceSurface,
                                             const CChunkedVolumeViewer::CameraState& camera,
                                             cv::Vec3d sourceSliceNormal,
                                             std::shared_ptr<LineAnnotationController::LineAnnotationSession> session,
                                             bool deferShowUntilGenerated)
{
    if (!_state || !session) {
        return false;
    }
    if (!prepareForUserFacingLineAnnotationOpen()) {
        return false;
    }

    session->deferShowUntilGenerated = deferShowUntilGenerated;
    _state->setSurface(surfaceName, std::move(sourceSurface));
    auto* dialog = new LineAnnotationDialog(_viewerManager, _volumeSelectorFactory, nullptr);
    dialog->setFiberDisplayName(fiberDisplayNameFromFileName(session->fiberFileName));
    if (!dialog->addPane(surfaceName, tr("Line Annotation Slice"), camera)) {
        dialog->deleteLater();
        _state->setSurface(surfaceName, nullptr);
        return false;
    }
    session->surfaceName = surfaceName;
    session->sourceAnnotationSurfaceName = surfaceName;
    session->sourceSliceNormal = finiteDirection(sourceSliceNormal)
        ? normalizedOrZero(sourceSliceNormal)
        : cv::Vec3d{0.0, 0.0, 1.0};

    _panes.push_back(PaneRecord{_nextPaneId - 1, sourceKind, surfaceName, dialog, session});
    connect(dialog, &LineAnnotationDialog::paneClosed, this, [this](const std::string& name) {
        cleanupSurfaceName(name);
    });
    connect(dialog,
            &LineAnnotationDialog::closeFinalizationRequested,
            this,
            [this, surfaceName](QCloseEvent*) {
                requestFinalizedClose(surfaceName);
            });
    connect(dialog,
            &LineAnnotationDialog::lineSeedRequested,
            this,
            [this, dialog](const std::string& name, cv::Vec3f volumePoint, QPointF) {
                InitialDirectionMode mode = InitialDirectionMode::Sideways;
                if (dialog) {
                    mode = dialog->initialDirectionMode() == LineAnnotationDialog::InitialDirectionMode::ZInOut
                        ? InitialDirectionMode::ZInOut
                        : InitialDirectionMode::Sideways;
                }
                handleLineSeed(name, volumePoint, mode);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointRequested,
            this,
            [this](const std::string& name, cv::Vec3f volumePoint, double linePosition) {
                handleGeneratedControlPoint(name, volumePoint, linePosition);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointDeleteRequested,
            this,
            [this](const std::string& name, double linePosition, cv::Vec3f volumePoint) {
                handleGeneratedControlPointDelete(name, linePosition, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointBranchRequested,
            this,
            [this](const std::string& name,
                   size_t controlPointIndex,
                   cv::Vec3f linkedControlPoint,
                   bool openAfterCreate,
                   cv::Vec3f linkDirection) {
                handleGeneratedControlPointBranch(name,
                                                  controlPointIndex,
                                                  linkedControlPoint,
                                                  openAfterCreate,
                                                  linkDirection);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointBranchOpenRequested,
            this,
            [this](uint64_t branchFiberId, int branchControlPointIndex) {
                openFiberAtControlPoint(branchFiberId, branchControlPointIndex);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointLinkCandidateRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, cv::Vec3f volumePoint) {
                handleGeneratedControlPointLinkCandidate(name, controlPointIndex, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointLinkWithCandidateRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, cv::Vec3f volumePoint) {
                handleGeneratedControlPointLinkWithCandidate(name, controlPointIndex, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedNearbyAnnotationOpenRequested,
            this,
            [this](uint64_t fiberId, cv::Vec3f volumePoint) {
                handleGeneratedOpenNearbyAnnotation(fiberId, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointUnlinkRequested,
            this,
            [this](const std::string& name,
                   size_t controlPointIndex,
                   uint64_t branchFiberId,
                   int branchControlPointIndex) {
                handleGeneratedControlPointUnlink(name,
                                                  controlPointIndex,
                                                  branchFiberId,
                                                  branchControlPointIndex);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointLinkPendingChangeRequested,
            this,
            [this](const std::string& name,
                   size_t controlPointIndex,
                   uint64_t branchFiberId,
                   int branchControlPointIndex,
                   bool pending) {
                handleGeneratedControlPointSetLinkPending(name,
                                                          controlPointIndex,
                                                          branchFiberId,
                                                          branchControlPointIndex,
                                                          pending);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedPredSnapPointRequested,
            this,
            [this](const std::string& name, cv::Vec3f volumePoint) {
                handleGeneratedPredSnapPoint(name, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedSideStripIntersectionQueryRequested,
            this,
            [this](const std::string& name) {
                handleGeneratedSideStripIntersectionQuery(name);
            });
    connect(dialog, &LineAnnotationDialog::showAsMeshRequested, this, [this, surfaceName]() {
        handleShowAsMesh(surfaceName);
    });
    connect(dialog, &LineAnnotationDialog::fullOptimizationRequested, this, [this, surfaceName]() {
        auto* pane = paneForSurface(surfaceName);
        if (!pane || !pane->session) {
            return;
        }
        auto& session = *pane->session;
        if (session.taskState == LineAnnotationSession::TaskState::Running) {
            showError(tr("Line optimization is already running."));
            return;
        }
        if (session.optimizedLine.points.empty() || session.controlPoints.empty()) {
            return;
        }
        if (!ensureDatasetForSession(session)) {
            return;
        }
        startOptimization(session, true);
    });
    connect(dialog,
            &LineAnnotationDialog::reoptimizationModeChanged,
            this,
            [this, surfaceName](LineAnnotationDialog::ReoptimizationMode mode) {
                if (mode != LineAnnotationDialog::ReoptimizationMode::AutoReoptimize) {
                    return;
                }
                auto* pane = paneForSurface(surfaceName);
                if (!pane || !pane->session) {
                    return;
                }
                auto& session = *pane->session;
                if (session.optimizationState == SessionOptimizationState::Optimized) {
                    return;
                }
                if (session.taskState == LineAnnotationSession::TaskState::Running) {
                    showError(tr("Line optimization is already running."));
                    return;
                }
                if (session.optimizedLine.points.empty() || session.controlPoints.empty()) {
                    return;
                }
                if (!ensureDatasetForSession(session)) {
                    return;
                }
                startOptimization(session, false);
            });
    connect(dialog, &QObject::destroyed, this, [this, surfaceName]() {
        cleanupSurfaceName(surfaceName);
    });

    refreshSessionOptimizationStatus(*session);
    if (!session->optimizedLine.points.empty()) {
        session->taskState = LineAnnotationSession::TaskState::Succeeded;
        setSessionOptimizationState(*session, SessionOptimizationState::Optimized);
        materializeGeneratedViews(*session);
    }
    if (!session->deferShowUntilGenerated || !session->optimizedLine.points.empty()) {
        emit lineAnnotationWorkspaceRequested(dialog, tr("Line Annotation"));
        dialog->showWithSavedGeometry();
        dialog->raise();
        dialog->activateWindow();
    }
    return true;
}

void LineAnnotationController::openFiber(uint64_t fiberId)
{
    openFiberWithControlPoint(fiberId, std::nullopt, std::nullopt);
}

void LineAnnotationController::openFiberAtControlPoint(uint64_t fiberId, int controlPointIndex)
{
    openFiberWithControlPoint(fiberId, controlPointIndex, std::nullopt);
}

void LineAnnotationController::openFiberAtLinePointIndex(uint64_t fiberId, int linePointIndex)
{
    openFiberWithControlPoint(fiberId, std::nullopt, linePointIndex);
}

void LineAnnotationController::openFiberSpan(uint64_t fiberId,
                                             int firstControlIndex,
                                             int secondControlIndex)
{
    openFiberWithControlPoint(fiberId,
                              std::nullopt,
                              std::nullopt,
                              std::make_pair(firstControlIndex, secondControlIndex));
}

void LineAnnotationController::openFiberWithControlPoint(uint64_t fiberId,
                                                         std::optional<int> controlPointIndex,
                                                         std::optional<int> linePointIndex,
                                                         std::optional<std::pair<int, int>> spanControlIndices)
{
    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
        showError(tr("Fiber %1 is not loaded.").arg(fiberId));
        return;
    }
    const std::optional<cv::Vec3d> seedOnlyPoint =
        vc3d::line_annotation::storedSinglePointFiberSeed(it->controlPoints, it->linePoints);
    if (!seedOnlyPoint && it->linePoints.empty()) {
        showError(tr("Fiber %1 has no line points.").arg(fiberId));
        return;
    }

    auto session = std::make_shared<LineAnnotationSession>();
    session->fiberId = it->id;
    session->fiberUsername = it->username;
    session->fiberStartedAt = it->startedAt;
    session->fiberSequence = it->sequence;
    session->fiberFileName = it->fileName;
    session->fiberManualHvTag = it->manualHvTag;
    session->fiberTags = it->tags;
    session->branches = it->branches;
    session->disableInitialGeneratedHoverFollow =
        controlPointIndex.has_value() || linePointIndex.has_value() ||
        spanControlIndices.has_value();

    if (seedOnlyPoint) {
        const cv::Vec3d seedPoint = *seedOnlyPoint;
        session->seedPoint = seedPoint;
        session->focusedLinePosition = 0.0;
        session->focusedControlPoint = seedPoint;
        session->initialDirectionMode = InitialDirectionMode::ZInOut;
        const cv::Vec3d sourceSliceNormal =
            seedTraceSourceNormalForStoredFiber(*it, controlPointIndex, seedPoint);

        if (!ensureDatasetForSession(*session)) {
            return;
        }

        CChunkedVolumeViewer::CameraState camera;
        camera.scale = 1.0f;
        auto sourcePlane = std::make_shared<PlaneSurface>(
            toVec3f(seedPoint),
            toVec3f(sourceSliceNormal));
        const std::string surfaceName = nextSurfaceName();
        if (!launchSession(SourceKind::Plane,
                           surfaceName,
                           std::move(sourcePlane),
                           camera,
                           sourceSliceNormal,
                           session,
                           true)) {
            return;
        }
        handleLineSeed(surfaceName, toVec3f(seedPoint), InitialDirectionMode::ZInOut);
        return;
    }

    session->focusedLinePosition = static_cast<double>(it->linePoints.size() / 2);
    session->focusedControlPoint = it->controlPoints.empty()
        ? std::optional<cv::Vec3d>{}
        : std::optional<cv::Vec3d>{it->controlPoints[it->controlPoints.size() / 2]};

    if (!ensureDatasetForSession(*session)) {
        return;
    }

    try {
        session->optimizedLine = lineModelFromPoints(it->linePoints, session->normalSampler.get());
        session->fiberMetricsMatchStoredFiber = true;
    } catch (const std::exception& ex) {
        showError(tr("Could not reopen fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    session->controlPoints.clear();
    session->controlPoints.reserve(it->controlPoints.size());
    double seedDistance = std::numeric_limits<double>::infinity();
    int seedControl = -1;
    for (size_t i = 0; i < it->controlPoints.size(); ++i) {
        const cv::Vec3d& controlPoint = it->controlPoints[i];
        int bestIndex = 0;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (size_t lineIndex = 0; lineIndex < it->linePoints.size(); ++lineIndex) {
            const cv::Vec3d delta = it->linePoints[lineIndex] - controlPoint;
            const double distance = std::sqrt(delta.dot(delta));
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = static_cast<int>(lineIndex);
            }
        }
        vc::lasagna::LineControlPoint control;
        control.linePosition = static_cast<double>(bestIndex);
        control.volumePoint = controlPoint;
        control.optimizedIndex = bestIndex;
        session->controlPoints.push_back(control);

        const double centerDistance = std::abs(control.linePosition -
            static_cast<double>(it->linePoints.size() - 1) * 0.5);
        if (centerDistance < seedDistance) {
            seedDistance = centerDistance;
            seedControl = static_cast<int>(i);
        }
    }
    if (!session->controlPoints.empty()) {
        std::optional<std::pair<double, double>> spanLineRange;
        if (linePointIndex &&
            *linePointIndex >= 0 &&
            *linePointIndex < static_cast<int>(it->linePoints.size())) {
            double bestLineDistance = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < session->controlPoints.size(); ++i) {
                const auto& control = session->controlPoints[i];
                if (!std::isfinite(control.linePosition)) {
                    continue;
                }
                const double distance = std::abs(
                    control.linePosition - static_cast<double>(*linePointIndex));
                if (distance < bestLineDistance) {
                    bestLineDistance = distance;
                    seedControl = static_cast<int>(i);
                }
            }
        } else if (controlPointIndex &&
            *controlPointIndex >= 0 &&
            *controlPointIndex < static_cast<int>(session->controlPoints.size())) {
            seedControl = *controlPointIndex;
        } else if (spanControlIndices) {
            const int first = spanControlIndices->first;
            const int second = spanControlIndices->second;
            if (first >= 0 &&
                second >= 0 &&
                first < static_cast<int>(session->controlPoints.size()) &&
                second < static_cast<int>(session->controlPoints.size()) &&
                first != second) {
                const auto& firstControl = session->controlPoints[static_cast<size_t>(first)];
                const auto& secondControl = session->controlPoints[static_cast<size_t>(second)];
                if (std::isfinite(firstControl.linePosition) &&
                    std::isfinite(secondControl.linePosition)) {
                    const double start = std::min(firstControl.linePosition,
                                                  secondControl.linePosition);
                    const double end = std::max(firstControl.linePosition,
                                                secondControl.linePosition);
                    spanLineRange = std::make_pair(start, end);
                    const double center = (start + end) * 0.5;
                    seedControl = std::abs(firstControl.linePosition - center) <=
                                      std::abs(secondControl.linePosition - center)
                        ? first
                        : second;
                }
            }
        }
        if (seedControl < 0) {
            seedControl = 0;
        }
        session->controlPoints[static_cast<size_t>(seedControl)].isSeed = true;
        session->seedPoint = session->controlPoints[static_cast<size_t>(seedControl)].volumePoint;
        session->optimizedLine.displayFrameAnchorIndex =
            session->controlPoints[static_cast<size_t>(seedControl)].optimizedIndex;
        session->focusedLinePosition =
            session->controlPoints[static_cast<size_t>(seedControl)].linePosition;
        session->focusedControlPoint =
            session->controlPoints[static_cast<size_t>(seedControl)].volumePoint;
        if (spanLineRange) {
            const double center = (spanLineRange->first + spanLineRange->second) * 0.5;
            session->focusedLinePosition = std::clamp(
                center,
                0.0,
                static_cast<double>(it->linePoints.size() - 1));
            session->focusedControlPoint =
                interpolatedPointAtLinePosition(it->linePoints, session->focusedLinePosition);
            session->initialStripLinePositionRange = spanLineRange;
        }
    }
    if (_currentAtlasDir) {
        attachAtlasPredSnaps(*it, *session, *_currentAtlasDir);
    }
    setSessionOptimizationState(*session, SessionOptimizationState::Optimized);

    CChunkedVolumeViewer::CameraState camera;
    camera.scale = 1.0f;
    const cv::Vec3d origin = session->focusedControlPoint.value_or(
        it->linePoints.empty()
            ? cv::Vec3d{0.0, 0.0, 0.0}
            : it->linePoints[static_cast<size_t>(std::clamp(
                  static_cast<int>(std::llround(session->focusedLinePosition)),
                  0,
                  static_cast<int>(it->linePoints.size() - 1)))]
    );
    auto sourcePlane = std::make_shared<PlaneSurface>(
        cv::Vec3f{static_cast<float>(origin[0]),
                  static_cast<float>(origin[1]),
                  static_cast<float>(origin[2])},
        cv::Vec3f{0.0f, 0.0f, 1.0f});
    (void)launchSession(SourceKind::Plane,
                        nextSurfaceName(),
                        sourcePlane,
                        camera,
                        {0.0, 0.0, 1.0},
                        std::move(session));
}

void LineAnnotationController::deleteFiber(uint64_t fiberId)
{
    deleteFibers({fiberId});
}

void LineAnnotationController::deleteFibers(std::vector<uint64_t> fiberIds)
{
    std::sort(fiberIds.begin(), fiberIds.end());
    fiberIds.erase(std::unique(fiberIds.begin(), fiberIds.end()), fiberIds.end());
    fiberIds.erase(std::remove(fiberIds.begin(), fiberIds.end(), uint64_t{0}), fiberIds.end());
    if (fiberIds.empty()) {
        return;
    }

    std::vector<std::pair<uint64_t, std::string>> deletedFibers;
    deletedFibers.reserve(fiberIds.size());
    std::vector<uint64_t> deletedIds;
    deletedIds.reserve(fiberIds.size());
    for (uint64_t fiberId : fiberIds) {
        const auto path = fiberPath(fiberId);
        std::error_code ec;
        fs::remove(path, ec);
        if (ec) {
            showError(tr("Could not delete fiber %1: %2")
                          .arg(fiberId)
                          .arg(QString::fromStdString(ec.message())));
            continue;
        }
        deletedIds.push_back(fiberId);
        auto fiberIt = std::find_if(_fibers.begin(),
                                    _fibers.end(),
                                    [fiberId](const StoredFiber& fiber) {
                                        return fiber.id == fiberId;
                                    });
        deletedFibers.push_back({fiberId,
                                 fiberIt == _fibers.end()
                                     ? std::string{}
                                     : fiberIt->fileName});
    }
    if (deletedIds.empty()) {
        return;
    }

    if (_linkCandidate && std::binary_search(deletedIds.begin(),
                                             deletedIds.end(),
                                             _linkCandidate->fiberId)) {
        _linkCandidate.reset();
    }

    _fibers.erase(std::remove_if(_fibers.begin(),
                                 _fibers.end(),
                                 [&deletedIds](const StoredFiber& fiber) {
                                     return std::binary_search(deletedIds.begin(),
                                                               deletedIds.end(),
                                                               fiber.id);
                                 }),
                  _fibers.end());
    for (uint64_t deletedId : deletedIds) {
        invalidateFiberAlignmentMetrics(deletedId, false);
    }
    for (const auto& pane : _panes) {
        if (pane.session && std::binary_search(deletedIds.begin(),
                                               deletedIds.end(),
                                               pane.session->fiberId)) {
            pane.session->suppressFiberSave = true;
        }
    }
    for (const auto& [deletedId, deletedFileName] : deletedFibers) {
        removeBranchLinksToFiber(deletedId, deletedFileName);
    }
    emitFiberSummaries();
    emit fibersDeleted(deletedIds);
}

void LineAnnotationController::renameFiberFile(uint64_t fiberId)
{
    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
        showError(tr("Fiber %1 is not loaded.").arg(fiberId));
        return;
    }

    const QString currentName = QString::fromStdString(
        it->fileName.empty() ? fiberPath(*it).filename().string() : it->fileName);
    bool accepted = false;
    const QString input = QInputDialog::getText(_parentWidget.data(),
                                                tr("Rename Line JSON"),
                                                tr("File name:"),
                                                QLineEdit::Normal,
                                                currentName,
                                                &accepted);
    if (!accepted) {
        return;
    }

    QString validationError;
    const auto newFileName = normalizedFiberJsonFileNameInput(input, &validationError);
    if (!newFileName) {
        showError(validationError);
        return;
    }
    if (*newFileName == it->fileName) {
        return;
    }

    const fs::path dir = fibersDir();
    if (dir.empty()) {
        showError(tr("No volume package is loaded."));
        return;
    }

    const fs::path oldPath = fiberPath(*it);
    const fs::path newPath = dir / *newFileName;
    std::error_code ec;
    if (!fs::exists(oldPath, ec)) {
        showError(tr("Could not rename fiber %1: %2 does not exist.")
                      .arg(fiberId)
                      .arg(QString::fromStdString(oldPath.string())));
        return;
    }
    ec.clear();
    if (fs::exists(newPath, ec)) {
        showError(tr("Could not rename fiber %1: %2 already exists.")
                      .arg(fiberId)
                      .arg(QString::fromStdString(newPath.filename().string())));
        return;
    }

    const std::string oldFileName = it->fileName;
    StoredFiber renamed = *it;
    renamed.fileName = *newFileName;
    try {
        saveFiberNow(renamed);
        ec.clear();
        fs::remove(oldPath, ec);
        if (ec) {
            fs::remove(newPath);
            throw std::runtime_error("Failed to remove old file " +
                                     oldPath.string() + ": " + ec.message());
        }
    } catch (const std::exception& ex) {
        showError(tr("Could not rename fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    *it = std::move(renamed);
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            pane.session->fiberFileName = it->fileName;
            if (pane.dialog) {
                pane.dialog->setFiberDisplayName(
                    fiberDisplayNameFromFileName(pane.session->fiberFileName));
            }
        }
    }
    syncBranchFiberFileRename(fiberId, oldFileName, it->fileName);
    emitFiberSummaries();
}

void LineAnnotationController::importFibers()
{
    const fs::path dir = fibersDir();
    if (dir.empty()) {
        showError(tr("No volume package is loaded."));
        return;
    }

    const auto options = showFiberJsonPathDialog(_parentWidget.data(), true, dir);
    if (!options) {
        return;
    }

    std::vector<StoredFiber> importedFibers;
    int skipped = 0;

    auto tryAddFiber = [&](std::optional<StoredFiber> fiber) {
        if (!fiber) {
            ++skipped;
            return;
        }
        scaleStoredFiber(*fiber, options->scale);
        importedFibers.push_back(std::move(*fiber));
    };
    auto bundleEntryPath = [&](const nlohmann::json& item, size_t index) {
        if (item.is_object()) {
            const std::string fileName = fs::path(item.value("filename", std::string{}))
                                             .filename()
                                             .string();
            if (!fileName.empty()) {
                return options->path.parent_path() / fileName;
            }
        }
        return options->path.parent_path() /
               (options->path.stem().string() + "_" + std::to_string(index) + ".json");
    };

    try {
        std::error_code ec;
        if (fs::is_directory(options->path, ec)) {
            std::vector<fs::path> fiberFiles;
            for (const auto& entry : fs::directory_iterator(options->path, ec)) {
                if (ec) {
                    break;
                }
                if (entry.is_regular_file() && entry.path().extension() == ".json") {
                    fiberFiles.push_back(entry.path());
                }
            }
            std::sort(fiberFiles.begin(), fiberFiles.end());
            for (const auto& path : fiberFiles) {
                try {
                    tryAddFiber(loadFiberFile(path));
                } catch (const std::exception& ex) {
                    ++skipped;
                    Logger()->warn("Skipping invalid imported fiber JSON {}: {}",
                                   path.string(),
                                   ex.what());
                }
            }
        } else {
            std::ifstream in(options->path);
            if (!in) {
                throw std::runtime_error("Failed to open " + options->path.string());
            }
            const nlohmann::json root = nlohmann::json::parse(in);
            if (root.is_array()) {
                size_t index = 0;
                for (const auto& item : root) {
                    try {
                        tryAddFiber(loadFiberJson(item, bundleEntryPath(item, index++)));
                    } catch (const std::exception& ex) {
                        ++skipped;
                        Logger()->warn("Skipping invalid imported fiber entry in {}: {}",
                                       options->path.string(),
                                       ex.what());
                    }
                }
            } else if (root.is_object() && root.value("type", std::string{}) == "vc3d_fiber") {
                tryAddFiber(loadFiberJson(root, options->path));
            } else {
                const nlohmann::json* entries = nullptr;
                if (root.is_object() && root.contains("point_collections")) {
                    entries = &root.at("point_collections");
                } else if (root.is_object() && root.contains("fibers")) {
                    entries = &root.at("fibers");
                }
                if (!entries || !entries->is_array()) {
                    throw std::runtime_error(
                        "Import JSON must be a vc3d_fiber, an array of vc3d_fiber objects, "
                        "or a bundle with point_collections/fibers");
                }
                size_t index = 0;
                for (const auto& item : *entries) {
                    try {
                        tryAddFiber(loadFiberJson(item, bundleEntryPath(item, index++)));
                    } catch (const std::exception& ex) {
                        ++skipped;
                        Logger()->warn("Skipping invalid imported fiber entry in {}: {}",
                                       options->path.string(),
                                       ex.what());
                    }
                }
            }
        }

        if (importedFibers.empty()) {
            showError(skipped > 0
                          ? tr("No valid fibers were found. Skipped %1 invalid JSON item(s).")
                                .arg(skipped)
                          : tr("No fibers were found."));
            return;
        }

        uint64_t nextSequence = nextFiberSequenceForUsername(currentFiberUsername());
        std::unordered_set<std::string> reservedNames;
        reservedNames.reserve(importedFibers.size());
        for (auto& fiber : importedFibers) {
            fiber.id = 0;
            fiber.generation = std::max<uint64_t>(uint64_t{1}, fiber.generation);
            if (fiber.username.empty()) {
                fiber.username = currentFiberUsername();
            }
            if (fiber.startedAt.empty()) {
                fiber.startedAt = currentFiberDateTimeString();
            }
            if (fiber.sequence == 0) {
                fiber.sequence = nextSequence++;
            }
            fiber.fileName = uniqueImportedFiberFileName(fiber, reservedNames, nextSequence);
            fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(fiber.controlPoints);
            saveFiberNow(fiber);
        }

        loadFibersForCurrentPackage();
        QMessageBox::information(_parentWidget.data(),
                                 tr("Import Fibers"),
                                 skipped > 0
                                     ? tr("Imported %1 fiber(s). Skipped %2 invalid JSON item(s).")
                                           .arg(importedFibers.size())
                                           .arg(skipped)
                                     : tr("Imported %1 fiber(s).").arg(importedFibers.size()));
    } catch (const std::exception& ex) {
        showError(tr("Could not import fibers: %1").arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::exportFibers()
{
    if (_fibers.empty()) {
        showError(tr("There are no fibers to export."));
        return;
    }

    const fs::path defaultDir = currentVolpkgRoot().empty() ? fibersDir() : currentVolpkgRoot();
    const auto options = showFiberJsonPathDialog(_parentWidget.data(), false, defaultDir);
    if (!options) {
        return;
    }

    try {
        nlohmann::json root = nlohmann::json::object();
        root["type"] = "vc3d_fiber_collection";
        root["version"] = 1;
        root["scale"] = options->scale;
        copyCoordinateIdentityToJson(root, coordinateIdentityForState(_state));
        root["point_collections"] = nlohmann::json::array();
        for (const auto& fiber : _fibers) {
            root["point_collections"].push_back(fiberToJson(fiber, options->scale));
        }

        writeJsonAtomic(options->path, root);
        QMessageBox::information(_parentWidget.data(),
                                 tr("Export Fibers"),
                                 tr("Exported %1 fiber(s) to %2.")
                                     .arg(_fibers.size())
                                     .arg(QString::fromStdString(options->path.string())));
    } catch (const std::exception& ex) {
        showError(tr("Could not export fibers: %1").arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::setFiberManualHvTag(uint64_t fiberId, const QString& tag)
{
    const auto normalizedTag = vc3d::line_annotation::fiberHvTagToString(
        vc3d::line_annotation::fiberHvTagFromString(tag.toStdString()));
    const std::string manualTag = normalizedTag == "unknown" ? std::string{} : normalizedTag;

    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
        showError(tr("Fiber %1 is not loaded.").arg(fiberId));
        return;
    }

    const std::string previousManualTag = it->manualHvTag;
    it->manualHvTag = manualTag;
    it->needsSave = false;
    try {
        scheduleFiberSave(*it);
    } catch (const std::exception& ex) {
        it->manualHvTag = previousManualTag;
        showError(tr("Could not save fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            pane.session->fiberManualHvTag = manualTag;
        }
    }
    emitFiberSummaries();
}

void LineAnnotationController::setFiberTag(uint64_t fiberId, const QString& tag, bool enabled)
{
    QString validationError;
    const auto normalizedTag = normalizedFiberTagInput(tag, &validationError);
    if (!normalizedTag) {
        showError(validationError);
        return;
    }

    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
        showError(tr("Fiber %1 is not loaded.").arg(fiberId));
        return;
    }

    addKnownFiberTags({*normalizedTag});
    const std::vector<std::string> previousTags = it->tags;
    if (enabled) {
        addUniqueSorted(it->tags, *normalizedTag);
    } else {
        it->tags.erase(std::remove(it->tags.begin(), it->tags.end(), *normalizedTag),
                       it->tags.end());
    }
    if (it->tags == previousTags) {
        emitFiberSummaries();
        return;
    }

    it->needsSave = false;
    try {
        scheduleFiberSave(*it);
    } catch (const std::exception& ex) {
        it->tags = previousTags;
        showError(tr("Could not save fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            pane.session->fiberTags = it->tags;
        }
    }
    emitFiberSummaries();
}

void LineAnnotationController::recalculateFiberHvClassification(uint64_t fiberId)
{
    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
        showError(tr("Fiber %1 is not loaded.").arg(fiberId));
        return;
    }

    const auto previousClassification = it->hvClassification;
    it->hvClassification = vc3d::line_annotation::classifyFiberHv(it->controlPoints);
    it->needsSave = false;
    try {
        scheduleFiberSave(*it);
    } catch (const std::exception& ex) {
        it->hvClassification = previousClassification;
        showError(tr("Could not save fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }
    emitFiberSummaries();
}

void LineAnnotationController::recalculateAllFiberHvClassifications()
{
    bool changed = false;
    for (auto& fiber : _fibers) {
        const auto previousClassification = fiber.hvClassification;
        fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(fiber.controlPoints);
        fiber.needsSave = false;
        try {
            scheduleFiberSave(fiber);
            changed = true;
        } catch (const std::exception& ex) {
            fiber.hvClassification = previousClassification;
            fiber.needsSave = true;
            Logger()->warn("Could not save recalculated VC3D fiber {}: {}",
                           fiberPath(fiber).string(),
                           ex.what());
        }
    }
    if (changed) {
        emitFiberSummaries();
    }
}

void LineAnnotationController::calculateFiberAlignmentMetrics()
{
    if (_fibers.empty()) {
        return;
    }
    std::vector<uint64_t> fiberIds;
    fiberIds.reserve(_fibers.size());
    for (const auto& fiber : _fibers) {
        fiberIds.push_back(fiber.id);
    }
    requestFiberAlignmentMetricsForFibers(std::move(fiberIds));
}

void LineAnnotationController::calculateFiberAlignmentMetrics(std::vector<uint64_t> orderedFiberIds)
{
    if (orderedFiberIds.empty()) {
        calculateFiberAlignmentMetrics();
        return;
    }
    requestFiberAlignmentMetricsForFibers(std::move(orderedFiberIds));
}

void LineAnnotationController::requestFiberAlignmentMetrics(uint64_t fiberId)
{
    if (fiberId == 0) {
        return;
    }
    requestFiberAlignmentMetricsForFibers({fiberId});
}

void LineAnnotationController::createAtlasFromFiber(uint64_t fiberId)
{
    try {
        auto vpkg = _state ? _state->vpkg() : nullptr;
        if (!vpkg) {
            throw std::runtime_error("No volume package is loaded");
        }
        const fs::path volpkgRoot = vpkg->path().empty()
            ? fs::path(vpkg->getVolpkgDirectory())
            : vpkg->path().parent_path();
        if (volpkgRoot.empty()) {
            throw std::runtime_error("The current volume package has no root directory");
        }

        auto fiberIt = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
            return fiber.id == fiberId;
        });
        if (fiberIt == _fibers.end()) {
            throw std::runtime_error("Selected fiber is not available");
        }
        if (fiberIt->linePoints.empty()) {
            throw std::runtime_error("Selected fiber has no line points");
        }

        const auto resolvedLasagna = resolveAlignmentMetricsManifestPath();
        if (!resolvedLasagna)
            throw std::runtime_error("No Lasagna dataset selected");
        const fs::path manifestPath = resolvedLasagna->first;
        if (manifestPath.empty() || !fs::exists(manifestPath)) {
            throw std::runtime_error("Selected Lasagna dataset does not exist");
        }
        vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(
            manifestPath, {resolvedLasagna->second});
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        const fs::path initShellDir =
            vc::atlas::initShellDirectoryFromManifest(dataset.manifest());
        atlasDebug("selected_manifest=" + manifestPath.string());
        atlasDebug("resolved_init_shell_dir=" + initShellDir.string());

        std::vector<vc::atlas::SurfaceCandidate> candidates =
            vc::atlas::loadInitShellCandidates(initShellDir);
        if (atlasDebugEnabled()) {
            for (const auto& candidate : candidates) {
                const auto* points = candidate.surface ? candidate.surface->rawPointsPtr() : nullptr;
                atlasDebug("candidate_shell path=" + candidate.path.string() +
                           " grid=" + (points
                               ? std::to_string(points->cols) + "x" + std::to_string(points->rows)
                               : std::string("invalid")));
            }
        }

        vc::atlas::FiberInput input;
        std::error_code relativeEc;
        input.fiberPath = fs::relative(fiberPath(*fiberIt), volpkgRoot, relativeEc);
        if (relativeEc || input.fiberPath.empty()) {
            input.fiberPath = relativeFiberPath(*fiberIt);
        }
        input.controlPoints = fiberIt->controlPoints;
        input.linePoints = fiberIt->linePoints;
        vc::atlas::validateFiberInputControlPoints(input);
        atlasDebug("fiber line_points=" + std::to_string(input.linePoints.size()) +
                   " control_points=" + std::to_string(input.controlPoints.size()));

        SurfacePatchIndex shellIndex;
        std::vector<SurfacePatchIndex::SurfacePtr> candidateSurfaces;
        candidateSurfaces.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            if (candidate.surface) {
                candidateSurfaces.push_back(candidate.surface);
            }
        }
        shellIndex.rebuild(candidateSurfaces);
        const auto selection = vc::atlas::selectBaseSurfaceBySeedRay(
            input, candidates, shellIndex, sampler);
        auto& selected = candidates.at(static_cast<size_t>(selection.surfaceIndex));
        const int zeroWindingColumn = vc::atlas::computeZeroWindingColumn(*selected.surface);
        atlasDebug("zero_winding_column=" + std::to_string(zeroWindingColumn));

        SurfacePatchIndex baseIndex;
        baseIndex.rebuild({selected.surface});
        auto mapping = vc::atlas::mapFiberToBaseSurface(input, *selected.surface, baseIndex, sampler);

        const std::string atlasName = "fiber_" + std::to_string(fiberId);
        const fs::path atlasDir = vc::atlas::uniqueAtlasDirectory(volpkgRoot, atlasName);
        auto atlas = vc::atlas::createSingleFiberAtlas(volpkgRoot,
                                                       atlasDir.filename().string(),
                                                       input,
                                                       selected,
                                                       zeroWindingColumn,
                                                       std::move(mapping));
        const auto coordinateIdentity = coordinateIdentityForState(_state);
        copyCoordinateIdentityToJson(
            atlas.metadata.coordinateMetadata, coordinateIdentity);
        vc3d::opendata::copyCoordinateIdentityToSurface(
            *selected.surface, coordinateIdentity);
        vc::atlas::saveAtlasBaseMeshCopy(*selected.surface,
                                         atlasDir / atlas.metadata.baseMeshPath);
        atlas.save(atlasDir);
        if (sampler.hasPredDtChannel() && !atlas.fibers.empty()) {
            (void)vc::atlas::ensureAtlasPredSnapSet(atlasDir,
                                                    input,
                                                    atlas.fibers.front(),
                                                    *selected.surface,
                                                    sampler);
        }
        emit atlasCreated(atlasDir);
    } catch (const std::exception& ex) {
        showError(tr("Could not create atlas: %1").arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::addFiberToPointCollection(uint64_t fiberId)
{
    auto fiberIt = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (fiberIt == _fibers.end()) {
        showError(tr("Selected fiber is not available"));
        return;
    }
    if (fiberIt->controlPoints.empty()) {
        showError(tr("Selected fiber has no control points"));
        return;
    }

    auto* col = _state ? _state->pointCollection() : nullptr;
    if (!col) {
        showError(tr("No point collection is available"));
        return;
    }

    const std::string name = col->generateNewCollectionName("fiber");
    const uint64_t collectionId = col->addCollection(name);

    CollectionMetadata meta;
    meta.absolute_winding_number = false;
    col->setCollectionMetadata(collectionId, meta);

    std::vector<cv::Vec3f> points;
    points.reserve(fiberIt->controlPoints.size());
    for (const auto& cp : fiberIt->controlPoints) {
        points.emplace_back(static_cast<float>(cp[0]),
                            static_cast<float>(cp[1]),
                            static_cast<float>(cp[2]));
    }
    col->addPoints(name, points);
}

void LineAnnotationController::addFibersToPointCollections(std::vector<uint64_t> fiberIds)
{
    for (uint64_t fiberId : fiberIds) {
        addFiberToPointCollection(fiberId);
    }
}

void LineAnnotationController::showFiberSlice(uint64_t fiberId, QMdiArea* targetArea)
{
    namespace fslice = vc3d::fiber_slice;

    try {
        if (!_state || !_viewerManager || !targetArea) {
            throw std::runtime_error("Fiber slice workspace is not available");
        }
        auto fiberIt = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
            return fiber.id == fiberId;
        });
        if (fiberIt == _fibers.end()) {
            throw std::runtime_error("Selected fiber is not loaded");
        }
        if (fiberIt->linePoints.empty()) {
            throw std::runtime_error("Selected fiber has no line points");
        }

        const fslice::ControlSpanSelection span =
            fslice::selectControlSpan(fiberIt->linePoints, fiberIt->controlPoints);
        if (!span.valid) {
            throw std::runtime_error(span.error);
        }
        const fslice::PlaneFit fit = fslice::fitLeastSquaresPlane(span, fiberIt->linePoints);
        if (!fit.valid) {
            throw std::runtime_error(fit.error);
        }

        if (_fiberSliceOverlay) {
            _fiberSliceOverlay->clearSlice();
        }
        for (QMdiSubWindow* subWindow : targetArea->subWindowList()) {
            if (!subWindow) {
                continue;
            }
            const QString oldSurface = subWindow->property("vc_fiber_slice_surface").toString();
            if (!oldSurface.isEmpty()) {
                _state->setSurface(oldSurface.toStdString(), nullptr);
            }
        }
        targetArea->closeAllSubWindows();

        const std::string surfaceName =
            "fiber_slice_" + std::to_string(fiberId) + "_" + std::to_string(_nextPaneId++);
        auto surface = std::make_shared<PlaneSurface>();
        surface->setFromNormalAndUp(
            cv::Vec3f{static_cast<float>(fit.origin[0]),
                      static_cast<float>(fit.origin[1]),
                      static_cast<float>(fit.origin[2])},
            cv::Vec3f{static_cast<float>(fit.normal[0]),
                      static_cast<float>(fit.normal[1]),
                      static_cast<float>(fit.normal[2])},
            cv::Vec3f{static_cast<float>(fit.upHint[0]),
                      static_cast<float>(fit.upHint[1]),
                      static_cast<float>(fit.upHint[2])});
        surface->id = surfaceName;
        _state->setSurface(surfaceName, surface);

        VolumeViewerBase* viewer = _viewerManager->createViewer(
            surfaceName,
            tr("Fiber %1 Slice").arg(fiberId),
            targetArea,
            ViewerManager::ViewerRole::Annotation);
        if (!viewer) {
            _state->setSurface(surfaceName, nullptr);
            throw std::runtime_error("Could not create fiber slice viewer");
        }
        if (auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject())) {
            if (auto* subWindow = qobject_cast<QMdiSubWindow*>(viewerWidget->parentWidget())) {
                subWindow->setProperty("vc_fiber_slice_surface", QString::fromStdString(surfaceName));
                subWindow->showMaximized();
            } else {
                viewerWidget->show();
            }
            connect(viewerWidget, &QObject::destroyed, this, [this, surfaceName]() {
                if (_state) {
                    _state->setSurface(surfaceName, nullptr);
                }
            });
        }

        const cv::Vec3f center{
            static_cast<float>(span.centroid[0]),
            static_cast<float>(span.centroid[1]),
            static_cast<float>(span.centroid[2]),
        };
        viewer->fitSurfaceInView();
        viewer->centerOnVolumePoint(center, true);

        if (_fiberSliceOverlay) {
            FiberSliceOverlayController::SliceData overlayData;
            overlayData.surfaceName = surfaceName;
            overlayData.selectedFiberId = fiberId;
            overlayData.plane = fslice::Plane{fit.origin, fit.normal};
            overlayData.fitSamples = span.samples;
            overlayData.fibers.reserve(_fibers.size());
            for (const StoredFiber& fiber : _fibers) {
                overlayData.fibers.push_back(FiberSliceOverlayController::FiberData{
                    fiber.id,
                    fiber.linePoints,
                    fiber.controlPoints,
                    FiberSliceOverlayController::sourceFiberStyle(),
                });
            }
            _fiberSliceOverlay->setSlice(viewer, std::move(overlayData));
        }

        if (auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject())) {
            viewerWidget->raise();
            viewerWidget->setFocus(Qt::OtherFocusReason);
        }
    } catch (const std::exception& ex) {
        showError(tr("Could not show fiber slice: %1")
                      .arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::showIntersectionInspection(
    const vc::atlas::FiberIntersectionResult& result,
    QMdiArea* targetArea,
    std::optional<fs::path> atlasDir)
{
    try {
        if (!_state || !_viewerManager || !targetArea) {
            throw std::runtime_error("Intersections workspace is not available");
        }
        cleanupIntersectionInspectionSurfaces();
        _intersectionInspection = std::make_unique<IntersectionInspectionSession>();
        _intersectionInspection->targetArea = targetArea;
        _intersectionInspection->result = result;
        _intersectionInspection->atlasDir = std::move(atlasDir);
        targetArea->installEventFilter(this);
        if (auto* viewport = targetArea->viewport()) {
            viewport->installEventFilter(this);
        }
        auto* followShortcut = new QShortcut(QKeySequence(Qt::Key_Space), targetArea);
        followShortcut->setContext(Qt::WidgetWithChildrenShortcut);
        _intersectionInspection->followShortcut = followShortcut;
        connect(followShortcut, &QShortcut::activated, this, [this]() {
            (void)handleIntersectionFollowKeyPress(Qt::Key_Space, Qt::NoModifier);
        });
        rebuildIntersectionInspection();
    } catch (const std::exception& ex) {
        showError(tr("Could not show intersection inspection: %1")
                      .arg(QString::fromStdString(ex.what())));
    }
}

bool LineAnnotationController::acceptIntersectionSameWindingChoice()
{
    try {
        if (!_intersectionInspection) {
            throw std::runtime_error("No intersection inspection is active");
        }
        if (!_intersectionInspection->atlasDir || _intersectionInspection->atlasDir->empty()) {
            throw std::runtime_error("Select an atlas before accepting an intersection link");
        }
        auto vpkg = _state ? _state->vpkg() : nullptr;
        if (!vpkg) {
            throw std::runtime_error("No volume package is loaded");
        }
        const fs::path volpkgRoot = vpkg->path().empty()
            ? fs::path(vpkg->getVolpkgDirectory())
            : vpkg->path().parent_path();
        if (volpkgRoot.empty()) {
            throw std::runtime_error("The current volume package has no root directory");
        }

        const auto result = _intersectionInspection->result;
        auto sourceIt = std::find_if(_fibers.begin(), _fibers.end(),
                                     [&result](const StoredFiber& fiber) {
                                         return fiber.id == result.sourceFiberId;
                                     });
        auto targetIt = std::find_if(_fibers.begin(), _fibers.end(),
                                     [&result](const StoredFiber& fiber) {
                                         return fiber.id == result.targetFiberId;
                                     });
        if (sourceIt == _fibers.end() || targetIt == _fibers.end()) {
            throw std::runtime_error("One or both intersection fibers are not loaded");
        }

        auto makeInput = [this, &volpkgRoot](const StoredFiber& fiber) {
            vc::atlas::FiberInput input;
            std::error_code relativeEc;
            input.fiberPath = fs::relative(fiberPath(fiber), volpkgRoot, relativeEc);
            if (relativeEc || input.fiberPath.empty()) {
                input.fiberPath = fs::path("fibers") / fiber.fileName;
            }
            input.controlPoints = fiber.controlPoints;
            input.linePoints = fiber.linePoints;
            vc::atlas::validateFiberInputControlPoints(input);
            return input;
        };
        const vc::atlas::FiberInput sourceInput = makeInput(*sourceIt);
        const vc::atlas::FiberInput targetInput = makeInput(*targetIt);

        vc::atlas::Atlas atlas = vc::atlas::Atlas::load(*_intersectionInspection->atlasDir);
        auto findMapping = [](vc::atlas::Atlas& atlas, const fs::path& fiberPath) {
            const std::string key = vc::atlas::atlasFiberPathKey(fiberPath);
            return std::find_if(atlas.fibers.begin(),
                                atlas.fibers.end(),
                                [&key](const vc::atlas::FiberMapping& mapping) {
                                    return vc::atlas::atlasFiberPathKey(mapping.fiberPath) == key;
                                });
        };
        auto sourceMappingIt = findMapping(atlas, sourceInput.fiberPath);
        auto targetMappingIt = findMapping(atlas, targetInput.fiberPath);
        const bool sourceMapped = sourceMappingIt != atlas.fibers.end();
        const bool targetMapped = targetMappingIt != atlas.fibers.end();
        if (!sourceMapped && !targetMapped) {
            throw std::runtime_error(
                "An atlas must be seeded from one inspected object before linked objects can be added");
        }

        const auto resolvedLasagna = resolveAlignmentMetricsManifestPath();
        if (!resolvedLasagna)
            throw std::runtime_error("No Lasagna dataset selected");
        const fs::path manifestPath = resolvedLasagna->first;
        if (!fs::exists(manifestPath)) {
            throw std::runtime_error("Selected Lasagna dataset does not exist");
        }
        vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(
            manifestPath, {resolvedLasagna->second});
        vc::lasagna::LasagnaNormalSampler sampler(dataset);

        const fs::path basePath = *_intersectionInspection->atlasDir / atlas.metadata.baseMeshPath;
        auto baseSurface = std::make_shared<QuadSurface>(basePath);
        SurfacePatchIndex baseIndex;
        baseIndex.rebuild({baseSurface});
        const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(*baseSurface);

        if (!sourceMapped) {
            auto mapping = vc::atlas::mapFiberToBaseSurface(
                sourceInput, *baseSurface, baseIndex, sampler);
            atlas.fibers.push_back(std::move(mapping));
        }
        if (!targetMapped) {
            auto mapping = vc::atlas::mapFiberToBaseSurface(
                targetInput, *baseSurface, baseIndex, sampler);
            atlas.fibers.push_back(std::move(mapping));
        }

        sourceMappingIt = findMapping(atlas, sourceInput.fiberPath);
        targetMappingIt = findMapping(atlas, targetInput.fiberPath);
        if (sourceMappingIt == atlas.fibers.end() || targetMappingIt == atlas.fibers.end()) {
            throw std::runtime_error("Could not map both inspected fibers into the selected atlas");
        }

        const auto sourceSample =
            vc3d::fiber_slice::samplePolylineAtArclength(sourceIt->linePoints,
                                                         result.sourceArclength);
        const auto targetSample =
            vc3d::fiber_slice::samplePolylineAtArclength(targetIt->linePoints,
                                                         result.targetArclength);
        if (!sourceSample.valid || !targetSample.valid) {
            throw std::runtime_error("Could not sample inspected arclengths on the loaded fibers");
        }

        auto endpointFor = [](const vc::atlas::FiberMapping& mapping,
                              double arclength,
                              double linePosition) {
            const vc::atlas::AtlasAnchor* best = nullptr;
            double bestDelta = std::numeric_limits<double>::infinity();
            for (const auto& anchor : mapping.lineAnchors) {
                const double delta = std::abs(static_cast<double>(anchor.sourceIndex) - linePosition);
                if (delta < bestDelta) {
                    best = &anchor;
                    bestDelta = delta;
                }
            }
            if (!best) {
                throw std::runtime_error("Mapped fiber has no line anchors for the inspected link");
            }
            vc::atlas::AtlasLinkEndpoint endpoint;
            endpoint.fiberPath = mapping.fiberPath;
            endpoint.sourceIndex = best->sourceIndex;
            endpoint.arclength = arclength;
            endpoint.atlasU = best->atlasU;
            endpoint.atlasV = best->atlasV;
            return endpoint;
        };

        vc::atlas::AtlasLink link;
        link.first = endpointFor(*sourceMappingIt,
                                 result.sourceArclength,
                                 sourceSample.linePosition);
        link.second = endpointFor(*targetMappingIt,
                                  result.targetArclength,
                                  targetSample.linePosition);
        link.desiredWindingDelta = 0;
        atlas.links.push_back(std::move(link));
        vc::atlas::layoutAtlasObjects(atlas, periodColumns);
        if (sampler.hasPredDtChannel()) {
            sourceMappingIt = findMapping(atlas, sourceInput.fiberPath);
            targetMappingIt = findMapping(atlas, targetInput.fiberPath);
            if (sourceMappingIt != atlas.fibers.end()) {
                (void)vc::atlas::ensureAtlasPredSnapSet(*_intersectionInspection->atlasDir,
                                                        sourceInput,
                                                        *sourceMappingIt,
                                                        *baseSurface,
                                                        sampler);
            }
            if (targetMappingIt != atlas.fibers.end()) {
                (void)vc::atlas::ensureAtlasPredSnapSet(*_intersectionInspection->atlasDir,
                                                        targetInput,
                                                        *targetMappingIt,
                                                        *baseSurface,
                                                        sampler);
            }
        }
        atlas.save(*_intersectionInspection->atlasDir);

        emit atlasCreated(*_intersectionInspection->atlasDir);
        return true;
    } catch (const std::exception& ex) {
        showError(tr("Could not accept intersection link: %1")
                      .arg(QString::fromStdString(ex.what())));
        return false;
    }
}

void LineAnnotationController::cleanupIntersectionInspectionSurfaces()
{
    if (!_intersectionInspection) {
        return;
    }
    if (_fiberSliceOverlay) {
        _fiberSliceOverlay->clearSlice();
    }
    if (auto* area = _intersectionInspection->targetArea.data()) {
        for (QMdiSubWindow* subWindow : area->subWindowList()) {
            if (!subWindow) {
                continue;
            }
            const QString oldSurface = subWindow->property("vc_intersection_slice_surface").toString();
            if (!oldSurface.isEmpty() && _state) {
                _state->setSurface(oldSurface.toStdString(), nullptr);
            }
        }
        area->closeAllSubWindows();
    }
    if (_state) {
        for (const auto& name : _intersectionInspection->surfaceNames) {
            _state->setSurface(name, nullptr);
        }
    }
    const std::string sourceSession = _intersectionInspection->sourceSessionSurfaceName;
    const std::string targetSession = _intersectionInspection->targetSessionSurfaceName;
    _panes.erase(std::remove_if(_panes.begin(),
                                _panes.end(),
                                [&sourceSession, &targetSession](const PaneRecord& pane) {
                                    return (!sourceSession.empty() && pane.surfaceName == sourceSession) ||
                                           (!targetSession.empty() && pane.surfaceName == targetSession);
                                }),
                 _panes.end());
    _intersectionInspection->surfaceNames.clear();
    _intersectionInspection->sourceLineSession.reset();
    _intersectionInspection->targetLineSession.reset();
    _intersectionInspection->sourceSessionSurfaceName.clear();
    _intersectionInspection->targetSessionSurfaceName.clear();
    _intersectionInspection->sourceFollow = {};
    _intersectionInspection->targetFollow = {};
    if (_intersectionInspection->followShortcut) {
        delete _intersectionInspection->followShortcut.data();
        _intersectionInspection->followShortcut = nullptr;
    }
    _intersectionInspection->activeFollowSourceSide.reset();
    _intersectionInspection->generatedSurfaceContexts.clear();
}

bool LineAnnotationController::updateIntersectionFollowSlice(bool sourceSideFlag,
                                                             double linePosition,
                                                             const char* reason)
{
    if (!_intersectionInspection || !_state) {
        return false;
    }
    auto& follow = sourceSideFlag ? _intersectionInspection->sourceFollow
                                  : _intersectionInspection->targetFollow;
    if (!follow.valid || follow.linePoints.empty()) {
        return false;
    }
    linePosition = std::clamp(linePosition,
                              0.0,
                              static_cast<double>(follow.linePoints.size() - 1));
    const cv::Vec3d origin = interpolatedPointAtLinePosition(follow.linePoints, linePosition);
    const cv::Vec3d tangent = tangentAtLinePosition(follow.linePoints, linePosition);
    const cv::Vec3f upHint = interpolatedUpAtLinePosition(follow.lineUpVectors,
                                                           linePosition,
                                                           tangent);
    const auto fit = vc3d::fiber_slice::planeFromNormalAndTangent(origin,
                                                                  tangent,
                                                                  toVec3d(upHint));
    if (!fit.valid) {
        return false;
    }
    auto surface = std::make_shared<PlaneSurface>();
    surface->setFromNormalAndUp(toVec3f(fit.origin),
                                toVec3f(fit.normal),
                                toVec3f(fit.upHint));
    surface->id = follow.surfaceName;
    _state->setSurface(follow.surfaceName, surface, false, true);
    follow.linePosition = linePosition;
    updateControlPointInfoLabel(follow.controlPointInfoLabel,
                                follow.controlPoints,
                                follow.linePosition);
    if (auto* viewer = follow.viewer.data()) {
        viewer->centerOnVolumePoint(toVec3f(origin), false);
        viewer->renderVisible(true, reason);
        if (_fiberSliceOverlay) {
            _fiberSliceOverlay->refreshViewer(viewer);
        }
    }
    return true;
}

void LineAnnotationController::toggleIntersectionFollowSlice(bool sourceSideFlag)
{
    if (!_intersectionInspection) {
        return;
    }
    auto& follow = sourceSideFlag ? _intersectionInspection->sourceFollow
                                  : _intersectionInspection->targetFollow;
    if (!follow.valid) {
        return;
    }
    if (follow.followsMouse) {
        std::vector<double> controlLinePositions;
        controlLinePositions.reserve(follow.controlPoints.size());
        for (const auto& control : follow.controlPoints) {
            if (std::isfinite(control.linePosition)) {
                controlLinePositions.push_back(control.linePosition);
            }
        }
        const double snapped = vc3d::line_annotation::snappedControlPointLinePosition(
            follow.linePosition,
            controlLinePositions);
        (void)updateIntersectionFollowSlice(sourceSideFlag,
                                            snapped,
                                            "intersection follow slice frozen");
        follow.followsMouse = false;
    } else {
        follow.followsMouse = true;
        updateControlPointInfoLabel(follow.controlPointInfoLabel,
                                    follow.controlPoints,
                                    std::numeric_limits<double>::quiet_NaN());
    }
}

bool LineAnnotationController::handleIntersectionFollowKeyPress(int key,
                                                                Qt::KeyboardModifiers modifiers)
{
    if (!_intersectionInspection || key != Qt::Key_Space || modifiers != Qt::NoModifier) {
        return false;
    }

    const bool anyFollowing =
        (_intersectionInspection->sourceFollow.valid &&
         _intersectionInspection->sourceFollow.followsMouse) ||
        (_intersectionInspection->targetFollow.valid &&
         _intersectionInspection->targetFollow.followsMouse);
    if (!anyFollowing) {
        if (_intersectionInspection->sourceFollow.valid) {
            _intersectionInspection->sourceFollow.followsMouse = true;
        }
        if (_intersectionInspection->targetFollow.valid) {
            _intersectionInspection->targetFollow.followsMouse = true;
        }
        return true;
    }

    if (_intersectionInspection->sourceFollow.valid &&
        _intersectionInspection->sourceFollow.followsMouse) {
        toggleIntersectionFollowSlice(true);
    } else if (_intersectionInspection->sourceFollow.valid) {
        _intersectionInspection->sourceFollow.followsMouse = false;
    }
    if (_intersectionInspection->targetFollow.valid &&
        _intersectionInspection->targetFollow.followsMouse) {
        toggleIntersectionFollowSlice(false);
    } else if (_intersectionInspection->targetFollow.valid) {
        _intersectionInspection->targetFollow.followsMouse = false;
    }
    return true;
}

bool LineAnnotationController::eventFilter(QObject* watched, QEvent* event)
{
    if (_intersectionInspection && event && event->type() == QEvent::Resize) {
        if (auto* widget = qobject_cast<QWidget*>(watched)) {
            if (auto* label = widget->findChild<QLabel*>(
                    QStringLiteral("intersectionControlPointInfo"))) {
                positionControlPointInfoLabel(label);
            }
        }
    }
    if (_intersectionInspection && event && event->type() == QEvent::KeyPress) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (watched) {
            const QVariant side = watched->property("vc_intersection_follow_source_side");
            if (side.isValid()) {
                _intersectionInspection->activeFollowSourceSide = side.toBool();
            }
        }
        if (handleIntersectionFollowKeyPress(keyEvent->key(), keyEvent->modifiers())) {
            keyEvent->accept();
            return true;
        }
    }
    return QObject::eventFilter(watched, event);
}

void LineAnnotationController::rebuildIntersectionInspection()
{
    namespace fslice = vc3d::fiber_slice;

    struct SliceSpec {
        QString title;
        std::string surfaceName;
        uint64_t selectedFiberId = 0;
        fslice::PlaneFit fit;
        cv::Vec3d center{0.0, 0.0, 0.0};
        std::vector<uint64_t> fullLineFiberIds;
        std::vector<FiberSliceOverlayController::FocusMarker> focusMarkers;
        bool editableCurrentCross = false;
        std::shared_ptr<LineAnnotationSession> editSession;
        double editLinePosition = 0.0;
        bool editUsesFollowLinePosition = false;
        bool shiftScroll = false;
        bool showGenericCrossings = true;
        bool showConnectionSegment = true;
        bool followCross = false;
        bool hasFollowSide = false;
        bool sourceFollow = true;
    };

    struct SideBuild {
        const StoredFiber* fiber = nullptr;
        uint64_t otherFiberId = 0;
        bool sourceSide = true;
        double focusLinePosition = 0.0;
        cv::Vec3d focusPoint{0.0, 0.0, 0.0};
        cv::Vec3d tangent{1.0, 0.0, 0.0};
        vc::lasagna::LineViewSurfaces lineViews;
        fslice::ControlTripletSelection triplet;
        std::shared_ptr<LineAnnotationSession> editSession;
        std::string editSessionName;
        std::string stripSurfaceName;
        QString displayTitle;
        std::string displayPrefix;
        FiberSliceOverlayController::FiberStyle style;
    };

    try {
        if (!_intersectionInspection || !_state || !_viewerManager) {
            return;
        }
        auto* targetArea = _intersectionInspection->targetArea.data();
        if (!targetArea) {
            throw std::runtime_error("Intersections workspace is not available");
        }

        const auto result = _intersectionInspection->result;
        auto sourceIt = std::find_if(_fibers.begin(), _fibers.end(),
                                     [&result](const StoredFiber& fiber) {
                                         return fiber.id == result.sourceFiberId;
                                     });
        auto targetIt = std::find_if(_fibers.begin(), _fibers.end(),
                                     [&result](const StoredFiber& fiber) {
                                         return fiber.id == result.targetFiberId;
                                     });
        if (sourceIt == _fibers.end() || targetIt == _fibers.end()) {
            throw std::runtime_error("One or both intersection fibers are not loaded");
        }
        auto finitePointCount = [](const std::vector<cv::Vec3d>& points) {
            return std::count_if(points.begin(), points.end(), fslice::isFinitePoint);
        };
        if (finitePointCount(sourceIt->linePoints) < 2 ||
            finitePointCount(targetIt->linePoints) < 2) {
            throw std::runtime_error("One or both intersection fibers have too few finite line points");
        }

        const fslice::ArclengthSample sourceSample =
            fslice::samplePolylineAtArclength(sourceIt->linePoints, result.sourceArclength);
        const fslice::ArclengthSample targetSample =
            fslice::samplePolylineAtArclength(targetIt->linePoints, result.targetArclength);
        if (!sourceSample.valid || !targetSample.valid) {
            throw std::runtime_error("Could not sample intersection arclengths on the loaded fibers");
        }

        cleanupIntersectionInspectionSurfaces();
        _intersectionInspection->sourceFocusLinePosition = sourceSample.linePosition;
        _intersectionInspection->targetFocusLinePosition = targetSample.linePosition;

        const cv::Vec3d connector = result.targetPoint - result.sourcePoint;
        const cv::Vec3d midpoint = (result.sourcePoint + result.targetPoint) * 0.5;
        const double connectorDistance = std::max(cv::norm(connector), 1.0e-6);
        if (!fslice::isFinitePoint(connector) || cv::norm(connector) <= 1.0e-10) {
            throw std::runtime_error("The refined connector segment is too short to define an inspection plane");
        }

        const double oldSourceArclength = result.sourceArclength;
        const double oldTargetArclength = result.targetArclength;
        auto sourceCallback = [this, oldSourceArclength, oldTargetArclength]() {
            if (_intersectionInspection && _intersectionInspection->sourceLineSession) {
                saveSessionAsFiber(*_intersectionInspection->sourceLineSession);
            }
            refreshIntersectionInspectionAfterEdit(
                _intersectionInspection ? _intersectionInspection->result.sourceFiberId : 0,
                oldSourceArclength,
                oldTargetArclength);
        };
        auto targetCallback = [this, oldSourceArclength, oldTargetArclength]() {
            if (_intersectionInspection && _intersectionInspection->targetLineSession) {
                saveSessionAsFiber(*_intersectionInspection->targetLineSession);
            }
            refreshIntersectionInspectionAfterEdit(
                _intersectionInspection ? _intersectionInspection->result.targetFiberId : 0,
                oldSourceArclength,
                oldTargetArclength);
        };

        SideBuild sourceSide;
        sourceSide.fiber = &*sourceIt;
        sourceSide.otherFiberId = targetIt->id;
        sourceSide.sourceSide = true;
        sourceSide.focusLinePosition = sourceSample.linePosition;
        sourceSide.focusPoint = result.sourcePoint;
        sourceSide.tangent = sourceSample.tangent;
        sourceSide.editSessionName = "intersection_edit_source_" + std::to_string(_nextPaneId++);
        sourceSide.editSession = makeIntersectionLineSession(*sourceIt,
                                                             sourceSample.linePosition,
                                                             sourceSample.tangent,
                                                             sourceSide.editSessionName,
                                                             sourceCallback);

        SideBuild targetSide;
        targetSide.fiber = &*targetIt;
        targetSide.otherFiberId = sourceIt->id;
        targetSide.sourceSide = false;
        targetSide.focusLinePosition = targetSample.linePosition;
        targetSide.focusPoint = result.targetPoint;
        targetSide.tangent = targetSample.tangent;
        targetSide.editSessionName = "intersection_edit_target_" + std::to_string(_nextPaneId++);
        targetSide.editSession = makeIntersectionLineSession(*targetIt,
                                                             targetSample.linePosition,
                                                             targetSample.tangent,
                                                             targetSide.editSessionName,
                                                             targetCallback);

        auto attachAtlasPredSnapsForInspection = [this](const StoredFiber& fiber,
                                                        const std::shared_ptr<LineAnnotationSession>& session) {
            if (!_intersectionInspection || !_intersectionInspection->atlasDir || !session) {
                return;
            }
            attachAtlasPredSnaps(fiber, *session, *_intersectionInspection->atlasDir);
        };
        attachAtlasPredSnapsForInspection(*sourceIt, sourceSide.editSession);
        attachAtlasPredSnapsForInspection(*targetIt, targetSide.editSession);

        const bool sourceIsH = vc3d::line_annotation::firstFiberDisplaysAsH(
            sourceIt->hvClassification,
            sourceIt->manualHvTag,
            targetIt->hvClassification,
            targetIt->manualHvTag,
            sourceIt->id < targetIt->id);
        SideBuild& hSide = sourceIsH ? sourceSide : targetSide;
        SideBuild& vSide = sourceIsH ? targetSide : sourceSide;
        hSide.displayTitle = tr("h");
        hSide.displayPrefix = "h";
        hSide.style = FiberSliceOverlayController::sourceFiberStyle();
        vSide.displayTitle = tr("v");
        vSide.displayPrefix = "v";
        vSide.style = FiberSliceOverlayController::targetFiberStyle();

        auto prepareSide = [this](SideBuild& side) {
            if (!side.editSession) {
                throw std::runtime_error("Intersection side has no editable line session");
            }
            if (!ensureDatasetForSession(*side.editSession)) {
                throw std::runtime_error(
                    "Intersection side strips require a selected Lasagna dataset; "
                    "not showing a synthetic strip");
            }
            side.editSession->optimizedLine =
                lineModelFromPoints(side.fiber->linePoints,
                                    side.editSession->normalSampler.get());
            side.lineViews = vc::lasagna::buildLineViewSurfaces(side.editSession->optimizedLine);
            Logger()->info("Intersection {} strip built from sampled normals: fiber={} points={} surface={} side_slice={}",
                           side.displayPrefix,
                           side.fiber ? side.fiber->id : 0,
                           side.fiber ? side.fiber->linePoints.size() : 0,
                           static_cast<const void*>(side.lineViews.lineSurface.get()),
                           static_cast<const void*>(side.lineViews.lineSideSlice.get()));
            side.triplet = vc3d::fiber_slice::selectControlTriplet(
                side.fiber->linePoints,
                side.fiber->controlPoints,
                side.focusLinePosition,
                side.focusPoint);
            if (!side.triplet.valid) {
                throw std::runtime_error("Could not select intersection cross-slice control points");
            }
            side.stripSurfaceName = std::string("intersection_strip_") +
                                    side.displayPrefix + "_" +
                                    std::to_string(_nextPaneId++);
            if (!side.lineViews.lineSideSlice) {
                throw std::runtime_error("Could not build intersection side line strip");
            }
            side.lineViews.lineSideSlice->id = side.stripSurfaceName;
            _state->setSurface(side.stripSurfaceName, side.lineViews.lineSideSlice);
            _intersectionInspection->surfaceNames.push_back(side.stripSurfaceName);
            side.editSession->generatedSurfaceNames.push_back(side.stripSurfaceName);
        };
        prepareSide(sourceSide);
        prepareSide(targetSide);

        _intersectionInspection->sourceLineSession = sourceSide.editSession;
        _intersectionInspection->targetLineSession = targetSide.editSession;
        _intersectionInspection->sourceSessionSurfaceName = sourceSide.editSessionName;
        _intersectionInspection->targetSessionSurfaceName = targetSide.editSessionName;
        _panes.push_back(PaneRecord{_nextPaneId++,
                                    SourceKind::Plane,
                                    sourceSide.editSessionName,
                                    {},
                                    sourceSide.editSession});
        _panes.push_back(PaneRecord{_nextPaneId++,
                                    SourceKind::Plane,
                                    targetSide.editSessionName,
                                    {},
                                    targetSide.editSession});

        std::vector<FiberSliceOverlayController::FiberData> overlayFibers{
            {sourceIt->id, sourceIt->linePoints, sourceIt->controlPoints, sourceSide.style},
            {targetIt->id, targetIt->linePoints, targetIt->controlPoints, targetSide.style},
        };

        auto makeCrossFit = [](const SideBuild& side, double position, const cv::Vec3d& origin) {
            const cv::Vec3d tangent = tangentAtLinePosition(side.fiber->linePoints, position);
            const cv::Vec3f upHint = interpolatedUpAtLinePosition(
                side.lineViews.lineUpVectors,
                position,
                tangent);
            return vc3d::fiber_slice::planeFromNormalAndTangent(origin,
                                                                 tangent,
                                                                 toVec3d(upHint));
        };

        std::vector<SliceSpec> planeSpecs;
        planeSpecs.reserve(9);
        auto appendSideSpecs = [&](const SideBuild& side) {
            const std::array<std::pair<QString, std::pair<double, cv::Vec3d>>, 3> crosses{{
                {tr("previous"), {side.triplet.previousLinePosition, side.triplet.previousPoint}},
                {tr("current"), {side.triplet.currentLinePosition, side.triplet.currentPoint}},
                {tr("next"), {side.triplet.nextLinePosition, side.triplet.nextPoint}},
            }};
            for (const auto& cross : crosses) {
                const std::string surfaceName = "intersection_" + side.displayPrefix + "_cross_" +
                    cross.first.toStdString() + "_" + std::to_string(_nextPaneId++);
                SliceSpec spec;
                spec.title = side.displayTitle + QStringLiteral(" ") + cross.first;
                spec.surfaceName = surfaceName;
                spec.selectedFiberId = 0;
                spec.fit = makeCrossFit(side, cross.second.first, cross.second.second);
                spec.center = cross.second.second;
                spec.focusMarkers = {
                    FiberSliceOverlayController::FocusMarker{side.fiber->id,
                                                             side.focusPoint,
                                                             4.5,
                                                             true},
                };
                spec.editableCurrentCross = true;
                spec.editSession = side.editSession;
                spec.editLinePosition = cross.second.first;
                spec.showGenericCrossings = false;
                spec.showConnectionSegment = false;
                spec.hasFollowSide = true;
                spec.sourceFollow = side.sourceSide;
                planeSpecs.push_back(std::move(spec));
                side.editSession->generatedSurfaceNames.push_back(surfaceName);
                _intersectionInspection->generatedSurfaceContexts[surfaceName] =
                    IntersectionInspectionSession::GeneratedSurfaceContext{
                        true,
                        side.sourceSide,
                        false,
                        false,
                        cross.second.first,
                    };
            }
            SliceSpec connection;
            connection.title = side.displayTitle + tr(" connection");
            connection.surfaceName = "intersection_" + side.displayPrefix + "_connection_" +
                                     std::to_string(_nextPaneId++);
            connection.selectedFiberId = side.fiber->id;
            connection.fit = fslice::planeFromDirections(midpoint, connector, side.tangent);
            connection.center = midpoint;
            connection.fullLineFiberIds = {side.fiber->id};
            connection.shiftScroll = true;
            connection.showGenericCrossings = true;
            connection.showConnectionSegment = true;
            connection.hasFollowSide = true;
            connection.sourceFollow = side.sourceSide;
            planeSpecs.push_back(std::move(connection));
        };
        appendSideSpecs(hSide);
        appendSideSpecs(vSide);

        SliceSpec normalSpec;
        normalSpec.title = tr("normal");
        normalSpec.surfaceName = "intersection_normal_" + std::to_string(_nextPaneId++);
        normalSpec.selectedFiberId = hSide.fiber->id;
        normalSpec.fit = fslice::planeFromDirections(midpoint,
                                                     sourceSample.tangent,
                                                     targetSample.tangent);
        normalSpec.center = midpoint;
        normalSpec.fullLineFiberIds = {hSide.fiber->id, vSide.fiber->id};
        normalSpec.shiftScroll = true;
        normalSpec.showGenericCrossings = true;
        normalSpec.showConnectionSegment = true;

        auto makeFollowSpec = [&](const SideBuild& side) {
            SliceSpec spec;
            spec.title = side.displayTitle + tr(" follow");
            spec.surfaceName = "intersection_" + side.displayPrefix + "_follow_" +
                               std::to_string(_nextPaneId++);
            spec.selectedFiberId = 0;
            spec.fit = makeCrossFit(side, side.focusLinePosition, side.focusPoint);
            spec.center = side.focusPoint;
            spec.focusMarkers = {
                FiberSliceOverlayController::FocusMarker{side.fiber->id,
                                                         side.focusPoint,
                                                         4.5,
                                                         true},
            };
            spec.showGenericCrossings = false;
            spec.showConnectionSegment = false;
            spec.followCross = true;
            spec.hasFollowSide = true;
            spec.sourceFollow = side.sourceSide;
            spec.editableCurrentCross = true;
            spec.editSession = side.editSession;
            spec.editLinePosition = side.focusLinePosition;
            spec.editUsesFollowLinePosition = true;
            side.editSession->generatedSurfaceNames.push_back(spec.surfaceName);
            _intersectionInspection->generatedSurfaceContexts[spec.surfaceName] =
                IntersectionInspectionSession::GeneratedSurfaceContext{
                    true,
                    side.sourceSide,
                    false,
                    true,
                    side.focusLinePosition,
                };
            return spec;
        };
        SliceSpec hFollowSpec = makeFollowSpec(hSide);
        SliceSpec vFollowSpec = makeFollowSpec(vSide);

        auto addPlaneViewer = [&](const SliceSpec& spec, const QRect& geometry) -> VolumeViewerBase* {
            if (!spec.fit.valid) {
                throw std::runtime_error(spec.fit.error);
            }
            auto surface = std::make_shared<PlaneSurface>();
            surface->setFromNormalAndUp(toVec3f(spec.fit.origin),
                                        toVec3f(spec.fit.normal),
                                        toVec3f(spec.fit.upHint));
            surface->id = spec.surfaceName;
            _state->setSurface(spec.surfaceName, surface);
            _intersectionInspection->surfaceNames.push_back(spec.surfaceName);

            VolumeViewerBase* viewer = _viewerManager->createViewer(
                spec.surfaceName,
                spec.title,
                targetArea,
                ViewerManager::ViewerRole::Annotation);
            if (!viewer) {
                throw std::runtime_error("Could not create intersection slice viewer");
            }
            QPointer<QLabel> controlPointInfoLabel;
            if (auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject())) {
                viewerWidget->installEventFilter(this);
                if (spec.editableCurrentCross && spec.editSession) {
                    controlPointInfoLabel = createControlPointInfoLabel(
                        viewerWidget,
                        spec.editSession->controlPoints,
                        spec.editLinePosition);
                }
                if (spec.hasFollowSide) {
                    viewerWidget->setProperty("vc_intersection_follow_source_side", spec.sourceFollow);
                }
                if (auto* subWindow = qobject_cast<QMdiSubWindow*>(viewerWidget->parentWidget())) {
                    subWindow->installEventFilter(this);
                    if (spec.hasFollowSide) {
                        subWindow->setProperty("vc_intersection_follow_source_side", spec.sourceFollow);
                    }
                    subWindow->setProperty("vc_intersection_slice_surface",
                                           QString::fromStdString(spec.surfaceName));
                    subWindow->setGeometry(geometry);
                    subWindow->show();
                } else {
                    viewerWidget->show();
                }
                connect(viewerWidget, &QObject::destroyed, this, [this, name = spec.surfaceName]() {
                    if (_state) {
                        _state->setSurface(name, nullptr);
                    }
                });
            }
            if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject())) {
                if (auto* view = chunkedViewer->graphicsView()) {
                    view->installEventFilter(this);
                    if (spec.hasFollowSide) {
                        view->setProperty("vc_intersection_follow_source_side", spec.sourceFollow);
                    }
                    if (auto* viewport = view->viewport()) {
                        viewport->installEventFilter(this);
                        if (spec.hasFollowSide) {
                            viewport->setProperty("vc_intersection_follow_source_side", spec.sourceFollow);
                        }
                    }
                }
                if (spec.followCross && _intersectionInspection) {
                    auto& follow = spec.sourceFollow ? _intersectionInspection->sourceFollow
                                                     : _intersectionInspection->targetFollow;
                    const SideBuild& followSide = spec.sourceFollow ? sourceSide : targetSide;
                    follow.valid = true;
                    follow.followsMouse = true;
                    follow.sourceSide = spec.sourceFollow;
                    follow.surfaceName = spec.surfaceName;
                    follow.viewer = chunkedViewer;
                    follow.linePoints = followSide.fiber->linePoints;
                    follow.lineUpVectors = followSide.lineViews.lineUpVectors;
                    follow.controlPoints = followSide.editSession->controlPoints;
                    follow.linePosition = followSide.focusLinePosition;
                    follow.controlPointInfoLabel = controlPointInfoLabel;
                    updateControlPointInfoLabel(follow.controlPointInfoLabel,
                                                follow.controlPoints,
                                                std::numeric_limits<double>::quiet_NaN());
                }
                if (spec.shiftScroll) {
                    chunkedViewer->setProperty("vc_show_custom_normal_offset", true);
                    chunkedViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
                    const cv::Vec3f stableUpHint = toVec3f(spec.fit.upHint);
                    auto offsetVx = std::make_shared<double>(0.0);
                    chunkedViewer->setShiftScrollOverride(
                        [this, surfaceName = spec.surfaceName, stableUpHint, offsetVx, chunkedViewer](
                            int steps,
                            QPointF,
                            Qt::KeyboardModifiers) {
                            if (!_state || steps == 0) {
                                return true;
                            }
                            auto planeShared =
                                std::dynamic_pointer_cast<PlaneSurface>(_state->surface(surfaceName));
                            if (!planeShared) {
                                return false;
                            }
                            const cv::Vec3f normal = planeShared->normal({0, 0, 0});
                            if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) ||
                                !std::isfinite(normal[2]) || cv::norm(normal) <= 0.0f) {
                                return true;
                            }

                            QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                            const double sensitivity = std::max(
                                0.01,
                                settings.value(vc3d::settings::viewer::ZSCROLL_SENSITIVITY,
                                               vc3d::settings::viewer::ZSCROLL_SENSITIVITY_DEFAULT)
                                    .toDouble());
                            const double delta = static_cast<double>(steps) * sensitivity;
                            auto shiftedPlane = std::make_shared<PlaneSurface>();
                            shiftedPlane->setFromNormalAndUp(
                                planeShared->origin() + normal * static_cast<float>(delta),
                                normal,
                                stableUpHint);
                            shiftedPlane->id = surfaceName;
                            *offsetVx += delta;
                            chunkedViewer->setProperty("vc_custom_normal_offset_vx", *offsetVx);
                            _state->setSurface(surfaceName, shiftedPlane, false, true);
                            return true;
                        });
                }
                if (spec.editableCurrentCross && spec.editSession) {
                    connect(chunkedViewer,
                            &CChunkedVolumeViewer::sendMousePressVolume,
                            this,
                            [this,
                             surfaceName = spec.surfaceName,
                             linePosition = spec.editLinePosition,
                             useFollowLinePosition = spec.editUsesFollowLinePosition,
                             sourceFollow = spec.sourceFollow](
                                cv::Vec3f volumePoint,
                                cv::Vec3f,
                                Qt::MouseButton button,
                                Qt::KeyboardModifiers modifiers,
                                QPointF) {
                                double effectiveLinePosition = linePosition;
                                if (useFollowLinePosition && _intersectionInspection) {
                                    const auto& follow = sourceFollow
                                        ? _intersectionInspection->sourceFollow
                                        : _intersectionInspection->targetFollow;
                                    if (follow.valid && std::isfinite(follow.linePosition)) {
                                        effectiveLinePosition = follow.linePosition;
                                    }
                                }
                                if (button == Qt::LeftButton && modifiers == Qt::ShiftModifier) {
                                    handleGeneratedPredSnapPoint(surfaceName, volumePoint);
                                } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                                    handleGeneratedControlPoint(surfaceName,
                                                                volumePoint,
                                                                effectiveLinePosition);
                                }
                            });
                    const SideBuild& generatedSide = spec.sourceFollow ? sourceSide : targetSide;
                    const std::vector<cv::Vec3d> linePoints = generatedSide.fiber->linePoints;
                    const std::vector<cv::Vec3f> lineUpVectors = generatedSide.lineViews.lineUpVectors;
                    const auto session = spec.editSession;
                    const bool sourceFollowFlag = spec.sourceFollow;
                    const bool useFollowPosition = spec.editUsesFollowLinePosition;
                    const double fixedLinePosition = spec.editLinePosition;
                    auto applyGeneratedCrossOverlay =
                        [this,
                         chunkedViewer,
                         surfaceName = spec.surfaceName,
                         linePoints,
                         lineUpVectors,
                         session,
                         sourceFollowFlag,
                         useFollowPosition,
                         fixedLinePosition]() {
                            if (!chunkedViewer || !_state || !session) {
                                return;
                            }
                            double linePosition = fixedLinePosition;
                            if (useFollowPosition && _intersectionInspection) {
                                const auto& follow = sourceFollowFlag
                                    ? _intersectionInspection->sourceFollow
                                    : _intersectionInspection->targetFollow;
                                if (follow.valid && std::isfinite(follow.linePosition)) {
                                    linePosition = follow.linePosition;
                                }
                            }
                            auto planeShared =
                                std::dynamic_pointer_cast<PlaneSurface>(_state->surface(surfaceName));
                            vc3d::line_annotation::GeneratedViews views;
                            views.linePoints = generatedLinePoints(linePoints);
                            views.lineUpVectors = lineUpVectors;
                            views.controlPoints = controlMarkersForSession(*session);
                            views.branchLinePoints = generatedBranchLinePointsForSession(*session);
                            views.branchLinks = generatedBranchLinkMarkers(session->branches);
                            views.predSnapPoints =
                                generatedPredSnapMarkers(session->controlPoints,
                                                         session->predSnapSet);
                            vc3d::line_annotation::applyGeneratedOverlay(
                                chunkedViewer,
                                surfaceName,
                                vc3d::line_annotation::makeGeneratedCrossSliceOverlayForPlane(
                                    views,
                                    linePosition,
                                    true,
                                    chunkedViewer,
                                    planeShared.get()));
                        };
                    chunkedViewer->renderVisible(true, "intersection generated cross overlay");
                    applyGeneratedCrossOverlay();
                    chunkedViewer->connectOverlaysUpdated(this, applyGeneratedCrossOverlay);
                }
            }
            viewer->fitSurfaceInView();
            viewer->centerOnVolumePoint(toVec3f(spec.center), true);
            if (_fiberSliceOverlay) {
                FiberSliceOverlayController::SliceData overlayData;
                overlayData.surfaceName = spec.surfaceName;
                overlayData.selectedFiberId = spec.selectedFiberId;
                overlayData.fullLineFiberIds = spec.fullLineFiberIds;
                overlayData.plane = fslice::Plane{spec.fit.origin, spec.fit.normal};
                overlayData.fitSamples = {result.sourcePoint, result.targetPoint, spec.center};
                overlayData.fibers = overlayFibers;
                overlayData.focusMarkers = spec.focusMarkers;
                overlayData.showGenericCrossings = spec.showGenericCrossings;
                if (spec.showConnectionSegment) {
                    overlayData.connectionSegment = FiberSliceOverlayController::ConnectionSegment{
                        sourceIt->id,
                        targetIt->id,
                        result.sourcePoint,
                        result.targetPoint,
                        connectorDistance,
                    };
                }
                _fiberSliceOverlay->setSlice(viewer, std::move(overlayData));
            }
            return viewer;
        };

        auto addStripViewer = [&](const SideBuild& side, const QRect& geometry) {
            VolumeViewerBase* viewer = _viewerManager->createViewer(
                side.stripSurfaceName,
                side.displayTitle + tr(" line"),
                targetArea,
                ViewerManager::ViewerRole::Annotation);
            if (!viewer) {
                throw std::runtime_error("Could not create intersection line strip viewer");
            }
            if (auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject())) {
                viewerWidget->installEventFilter(this);
                viewerWidget->setProperty("vc_intersection_follow_source_side", side.sourceSide);
                if (auto* subWindow = qobject_cast<QMdiSubWindow*>(viewerWidget->parentWidget())) {
                    subWindow->installEventFilter(this);
                    subWindow->setProperty("vc_intersection_follow_source_side", side.sourceSide);
                    subWindow->setProperty("vc_intersection_slice_surface",
                                           QString::fromStdString(side.stripSurfaceName));
                    subWindow->setGeometry(geometry);
                    subWindow->show();
                } else {
                    viewerWidget->show();
                }
            }
            auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
            if (!chunkedViewer) {
                return;
            }
            if (auto* view = chunkedViewer->graphicsView()) {
                view->installEventFilter(this);
                view->setProperty("vc_intersection_follow_source_side", side.sourceSide);
                if (auto* viewport = view->viewport()) {
                    viewport->installEventFilter(this);
                    viewport->setProperty("vc_intersection_follow_source_side", side.sourceSide);
                }
            }
            chunkedViewer->fitSurfaceInView();
            if (auto* quad = dynamic_cast<QuadSurface*>(chunkedViewer->currentSurface())) {
                frameStripLineSpan(chunkedViewer,
                                   quad,
                                   side.triplet.previousLinePosition,
                                   side.triplet.nextLinePosition);
            }
            if (_intersectionInspection) {
                _intersectionInspection->generatedSurfaceContexts[side.stripSurfaceName] =
                    IntersectionInspectionSession::GeneratedSurfaceContext{
                        true,
                        side.sourceSide,
                        true,
                        false,
                        side.focusLinePosition,
                    };
            }
            const std::vector<cv::Vec3d> linePoints = side.fiber->linePoints;
            const std::vector<cv::Vec3f> lineUpVectors = side.lineViews.lineUpVectors;
            const auto session = side.editSession;
            const double focus = side.focusLinePosition;
            const std::vector<double> markerLinePositions{
                side.triplet.previousLinePosition,
                side.triplet.currentLinePosition,
                side.triplet.nextLinePosition,
            };
            auto applyGeneratedStripOverlay =
                [this,
                 chunkedViewer,
                 key = side.stripSurfaceName,
                 linePoints,
                 lineUpVectors,
                 session,
                 focus,
                 markerLinePositions]() {
                    if (!chunkedViewer || !session) {
                        return;
                    }
                    vc3d::line_annotation::GeneratedViews views;
                    views.linePoints = generatedLinePoints(linePoints);
                    views.lineUpVectors = lineUpVectors;
                    views.controlPoints = controlMarkersForSession(*session);
                    views.branchLinePoints = generatedBranchLinePointsForSession(*session);
                    views.branchLinks = generatedBranchLinkMarkers(session->branches);
                    views.predSnapPoints =
                        generatedPredSnapMarkers(session->controlPoints,
                                                 session->predSnapSet);
                    auto overlay = vc3d::line_annotation::makeGeneratedStripOverlay(
                        views,
                        focus,
                        markerLinePositions);
                    overlay.currentLineMarkerAsCross = true;
                    vc3d::line_annotation::applyGeneratedOverlay(chunkedViewer, key, overlay);
                };
            chunkedViewer->renderVisible(true, "intersection generated strip overlay");
            applyGeneratedStripOverlay();
            chunkedViewer->connectOverlaysUpdated(this, applyGeneratedStripOverlay);
            connect(chunkedViewer,
                    &CChunkedVolumeViewer::sendMouseMoveVolume,
                    this,
                    [this,
                     chunkedViewer,
                     sourceSideFlag = side.sourceSide](cv::Vec3f,
                                                       Qt::MouseButtons,
                                                       Qt::KeyboardModifiers,
                                                       QPointF scenePoint) {
                        if (!_intersectionInspection) {
                            return;
                        }
                        auto& follow = sourceSideFlag ? _intersectionInspection->sourceFollow
                                                      : _intersectionInspection->targetFollow;
                        _intersectionInspection->activeFollowSourceSide = sourceSideFlag;
                        if (!follow.valid || !follow.followsMouse) {
                            return;
                        }
                        const double linePosition =
                            vc3d::line_annotation::generatedLinePositionFromStripScene(
                                chunkedViewer,
                                scenePoint);
                        if (std::isfinite(linePosition)) {
                            (void)updateIntersectionFollowSlice(
                                sourceSideFlag,
                                linePosition,
                                "intersection follow slice hover");
                        }
                    });
            connect(chunkedViewer,
                    &CChunkedVolumeViewer::sendMousePressVolume,
                    this,
                    [this,
                     surfaceName = side.stripSurfaceName,
                     sourceSideFlag = side.sourceSide](
                        cv::Vec3f volumePoint,
                        cv::Vec3f,
                        Qt::MouseButton button,
                        Qt::KeyboardModifiers modifiers,
                        QPointF scenePoint) {
                        if (modifiers != Qt::NoModifier && modifiers != Qt::ShiftModifier) {
                            return;
                        }
                        const double linePosition =
                            vc3d::line_annotation::generatedLinePositionFromStripScene(
                                qobject_cast<CChunkedVolumeViewer*>(sender()),
                                scenePoint);
                        if (!std::isfinite(linePosition)) {
                            return;
                        }
                        if (_intersectionInspection) {
                            _intersectionInspection->activeFollowSourceSide = sourceSideFlag;
                        }
                        if (button == Qt::LeftButton) {
                            if (modifiers == Qt::ShiftModifier) {
                                handleGeneratedPredSnapPoint(surfaceName, volumePoint);
                            } else {
                                handleGeneratedControlPoint(surfaceName, volumePoint, linePosition);
                            }
                        }
                    });
        };

        auto fiberDisplayName = [](const StoredFiber& fiber) {
            if (!fiber.fileName.empty()) {
                return QString::fromStdString(fs::path(fiber.fileName).stem().string());
            }
            return QStringLiteral("fiber %1").arg(fiber.id);
        };
        auto hvSummary = [](const StoredFiber& fiber) {
            const QString manual = fiber.manualHvTag.empty()
                ? QStringLiteral("unknown")
                : QString::fromStdString(fiber.manualHvTag);
            const auto& c = fiber.hvClassification;
            return QStringLiteral("manual %1; auto %2 h=%3 v=%4 cert=%5")
                .arg(manual,
                     QString::fromStdString(vc3d::line_annotation::fiberHvTagToString(c.automaticTag)))
                .arg(c.horizontalScore, 0, 'f', 2)
                .arg(c.verticalScore, 0, 'f', 2)
                .arg(c.automaticCertainty, 0, 'f', 2);
        };
        auto addDecisionPane = [&](const QRect& geometry) {
            auto* pane = new QWidget;
            pane->setObjectName(QStringLiteral("intersectionDecisionPane"));
            auto* layout = new QVBoxLayout(pane);
            layout->setContentsMargins(8, 8, 8, 8);
            layout->setSpacing(4);

            auto* title = new QLabel(tr("Intersection decision"), pane);
            title->setObjectName(QStringLiteral("intersectionDecisionTitle"));
            layout->addWidget(title);

            if (_volumeSelectorFactory) {
                if (auto* volumeSelector = _volumeSelectorFactory(pane)) {
                    volumeSelector->setObjectName(QStringLiteral("intersectionDecisionVolumeSelector"));
                    layout->addWidget(volumeSelector);
                }
            }

            auto* hLabel = new QLabel(
                tr("H: %1 - %2").arg(fiberDisplayName(*hSide.fiber), hvSummary(*hSide.fiber)),
                pane);
            hLabel->setObjectName(QStringLiteral("intersectionDecisionHLabel"));
            hLabel->setWordWrap(true);
            layout->addWidget(hLabel);

            auto* vLabel = new QLabel(
                tr("V: %1 - %2").arg(fiberDisplayName(*vSide.fiber), hvSummary(*vSide.fiber)),
                pane);
            vLabel->setObjectName(QStringLiteral("intersectionDecisionVLabel"));
            vLabel->setWordWrap(true);
            layout->addWidget(vLabel);

            auto* choices = new QButtonGroup(pane);
            choices->setExclusive(true);
            auto* same = new QRadioButton(tr("same winding (h inside v)"), pane);
            same->setObjectName(QStringLiteral("intersectionDecisionSameWinding"));
            auto* different = new QRadioButton(tr("different winding"), pane);
            different->setObjectName(QStringLiteral("intersectionDecisionDifferentWinding"));
            auto* hard = new QRadioButton(tr("hard to say"), pane);
            hard->setObjectName(QStringLiteral("intersectionDecisionHardToSay"));
            hard->setChecked(true);
            choices->addButton(same, 0);
            choices->addButton(different, 1);
            choices->addButton(hard, 2);
            layout->addWidget(same);
            layout->addWidget(different);
            layout->addWidget(hard);

            auto* bottomRow = new QHBoxLayout;
            auto* status = new QLabel(pane);
            status->setObjectName(QStringLiteral("intersectionDecisionStatus"));
            status->setWordWrap(true);
            status->setText(_intersectionInspection && _intersectionInspection->atlasDir
                ? tr("Ready")
                : tr("No atlas selected"));
            bottomRow->addWidget(status, 1);

            auto* accept = new QPushButton(tr("Accept choice"), pane);
            accept->setObjectName(QStringLiteral("intersectionDecisionAccept"));
            accept->setEnabled(_intersectionInspection &&
                               _intersectionInspection->atlasDir.has_value() &&
                               hSide.fiber && vSide.fiber &&
                               hSide.fiber->linePoints.size() >= 2 &&
                               vSide.fiber->linePoints.size() >= 2);
            bottomRow->addWidget(accept);
            layout->addLayout(bottomRow);

            connect(accept, &QPushButton::clicked, this, [this, choices, status]() {
                switch (choices->checkedId()) {
                case 0:
                    if (acceptIntersectionSameWindingChoice() && status) {
                        status->setText(tr("Saved same-winding link"));
                    }
                    break;
                case 1:
                    if (status) {
                        status->setText(tr("No atlas change recorded for different winding"));
                    }
                    break;
                case 2:
                default:
                    if (status) {
                        status->setText(tr("No atlas change recorded"));
                    }
                    break;
                }
            });

            auto* subWindow = targetArea->addSubWindow(pane);
            subWindow->setObjectName(QStringLiteral("intersectionDecisionSubWindow"));
            subWindow->setWindowTitle(tr("Decision"));
            subWindow->setGeometry(geometry);
            subWindow->show();
        };

        const QSize size = targetArea->viewport() ? targetArea->viewport()->size()
                                                  : targetArea->size();
        const int width = std::max(3, size.width());
        const int height = std::max(4, size.height());
        const int colW = width / 3;
        const int leftX = 0;
        const int centerX = colW;
        const int rightX = colW * 2;
        const int rightW = width - rightX;
        const int topH = height / 4;
        const int midH = height / 2;
        const int bottomY = topH + midH;
        const int bottomH = height - bottomY;
        const int centerTopH = std::min(std::max(1, height / 3),
                                        std::max(1, colW / 2));
        const int centerFollowW = std::max(1, colW / 2);
        auto topRect = [&](int x, int colWidth, int slot) {
            const int slotW = colWidth / 3;
            const int slotX = x + slot * slotW;
            const int w = slot == 2 ? colWidth - slotW * 2 : slotW;
            return QRect(slotX, 0, w, topH);
        };
        const QRect leftMid(leftX, topH, colW, midH);
        const QRect leftBottom(leftX, bottomY, colW, bottomH);
        const QRect hFollowRect(centerX, 0, centerFollowW, centerTopH);
        const QRect vFollowRect(centerX + centerFollowW,
                                0,
                                colW - centerFollowW,
                                centerTopH);
        const QRect centerRect(centerX, centerTopH, colW, std::max(1, bottomY - centerTopH));
        const QRect decisionRect(centerX, bottomY, colW, bottomH);
        const QRect rightMid(rightX, topH, rightW, midH);
        const QRect rightBottom(rightX, bottomY, rightW, bottomH);

        addPlaneViewer(planeSpecs[0], topRect(leftX, colW, 0));
        addPlaneViewer(planeSpecs[1], topRect(leftX, colW, 1));
        addPlaneViewer(planeSpecs[2], topRect(leftX, colW, 2));
        addPlaneViewer(planeSpecs[3], leftMid);
        addStripViewer(hSide, leftBottom);
        addPlaneViewer(hFollowSpec, hFollowRect);
        addPlaneViewer(vFollowSpec, vFollowRect);
        addPlaneViewer(normalSpec, centerRect);
        addDecisionPane(decisionRect);
        addPlaneViewer(planeSpecs[4], topRect(rightX, rightW, 0));
        addPlaneViewer(planeSpecs[5], topRect(rightX, rightW, 1));
        addPlaneViewer(planeSpecs[6], topRect(rightX, rightW, 2));
        addPlaneViewer(planeSpecs[7], rightMid);
        addStripViewer(vSide, rightBottom);

        if (auto* active = targetArea->activeSubWindow()) {
            if (auto* viewer = dynamic_cast<VolumeViewerBase*>(active->widget())) {
                if (auto* graphicsView = viewer->graphicsView()) {
                    graphicsView->setFocus();
                }
            }
        }
    } catch (const std::exception& ex) {
        showError(tr("Could not show intersection inspection: %1")
                      .arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::refreshIntersectionInspectionAfterEdit(uint64_t editedFiberId,
                                                                      double oldSourceArclength,
                                                                      double oldTargetArclength)
{
    if (!_intersectionInspection || editedFiberId == 0) {
        return;
    }
    const uint64_t sourceId = _intersectionInspection->result.sourceFiberId;
    const uint64_t targetId = _intersectionInspection->result.targetFiberId;
    try {
        std::vector<vc::atlas::FiberPolyline> fibers;
        fibers.reserve(2);
        for (const auto& fiber : _fibers) {
            if (fiber.id != sourceId && fiber.id != targetId) {
                continue;
            }
            vc::atlas::FiberPolyline polyline;
            polyline.id = fiber.id;
            polyline.generation = fiber.generation;
            polyline.controlPoints = fiber.controlPoints;
            polyline.points.reserve(fiber.linePoints.size());
            for (const auto& point : fiber.linePoints) {
                polyline.points.push_back(vc::atlas::FiberPoint{point, std::nullopt});
            }
            fibers.push_back(std::move(polyline));
        }
        if (fibers.size() != 2) {
            throw std::runtime_error("The edited intersection fibers are no longer both loaded");
        }
        vc::atlas::FiberIntersectionCache cache;
        vc::atlas::FiberIntersectionBroadPhaseOptions broad;
        vc::atlas::FiberIntersectionCeresOptions ceres;
        const auto results = vc::atlas::searchFiberIntersections(
            fibers,
            {sourceId},
            {targetId},
            &cache,
            broad,
            ceres);
        const auto nearest = vc::atlas::nearestIntersectionResultByArclength(
            results,
            oldSourceArclength,
            oldTargetArclength);
        if (!nearest) {
            cleanupIntersectionInspectionSurfaces();
            _intersectionInspection.reset();
            QMessageBox::warning(_parentWidget,
                                 tr("Intersections"),
                                 tr("The edited fiber pair no longer has an intersection result."));
            return;
        }
        _intersectionInspection->result = results[*nearest];
        rebuildIntersectionInspection();
    } catch (const std::exception& ex) {
        showError(tr("Could not refresh intersection inspection: %1")
                      .arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::saveOpenFibers()
{
    for (const auto& pane : _panes) {
        if (!pane.session || pane.session->suppressFiberSave) {
            continue;
        }
        auto& session = *pane.session;
        if (session.taskState != LineAnnotationSession::TaskState::Succeeded ||
            session.optimizedLine.points.empty() ||
            session.controlPoints.empty()) {
            continue;
        }
        if (!finalizeSessionOptimizationSynchronously(session, false)) {
            continue;
        }
        saveSessionAsFiber(session);
        session.suppressFiberSave = true;
    }
    waitForFiberSaves();
}

void LineAnnotationController::closeFiberWindowForSurface(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (pane && pane->dialog) {
        pane->dialog->close();
    }
}

std::vector<vc3d::line_annotation::GeneratedOverlay::ControlPointMarker>
LineAnnotationController::controlMarkersForSession(const LineAnnotationSession& session) const
{
    auto markers = generatedControlMarkers(session.controlPoints, session.branches);
    if (_linkCandidate && _linkCandidate->fiberId != 0 &&
        session.fiberId == _linkCandidate->fiberId) {
        for (size_t i = 0; i < session.controlPoints.size() && i < markers.size(); ++i) {
            if (pointsApproximatelyEqual(session.controlPoints[i].volumePoint,
                                         _linkCandidate->position)) {
                markers[i].isLinkCandidate = true;
                break;
            }
        }
    }
    return markers;
}

std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker>
LineAnnotationController::markLinkCandidateFiberIntersections(
    std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker> markers,
    const std::vector<FiberBranchRef>& branches) const
{
    const uint64_t candidateFiberId =
        _linkCandidate ? _linkCandidate->fiberId : uint64_t{0};
    for (auto& marker : markers) {
        marker.isLinkCandidateFiber =
            candidateFiberId != 0 && marker.fiberId == candidateFiberId;
        marker.pendingBranchLink = false;
        if (!marker.projectedBranchLink || !marker.connectorStart) {
            continue;
        }
        for (const auto& branch : branches) {
            // connectorStart carries the local endpoint position through a
            // double->float round trip; match with a coarse tolerance.
            if (branch.pending && branch.branchFiberId == marker.fiberId &&
                cv::norm(toVec3f(branch.controlPointPosition) -
                         *marker.connectorStart) <= 1.0e-3) {
                marker.pendingBranchLink = true;
                break;
            }
        }
    }
    return markers;
}

vc3d::line_annotation::GeneratedLinkCandidateMenuState
LineAnnotationController::linkCandidateMenuState(const LineAnnotationSession& session) const
{
    vc3d::line_annotation::GeneratedLinkCandidateMenuState state;
    if (!_linkCandidate || _linkCandidate->fiberId == 0) {
        return state;
    }
    if (session.fiberId != 0 && session.fiberId == _linkCandidate->fiberId) {
        state.enabled = false;
        state.label = tr("Link with candidate (same fiber)");
    } else {
        state.enabled = true;
        state.label = tr("Link with candidate (Fiber %1)")
                          .arg(static_cast<qulonglong>(_linkCandidate->fiberId));
    }
    return state;
}

bool LineAnnotationController::showGeneratedControlPointContextMenu(CChunkedVolumeViewer* viewer,
                                                                    const QPointF& scenePoint,
                                                                    const QPoint& globalPos)
{
    if (!viewer) {
        return false;
    }
    auto* pane = paneForSurface(viewer->surfName());
    if (!pane || !pane->session) {
        return false;
    }
    if (std::find(pane->session->generatedSurfaceNames.begin(),
                  pane->session->generatedSurfaceNames.end(),
                  viewer->surfName()) == pane->session->generatedSurfaceNames.end()) {
        return false;
    }
    vc3d::line_annotation::GeneratedControlPointContextResult result =
        vc3d::line_annotation::GeneratedControlPointContextResult::None;
    if (pane->dialog) {
        result = pane->dialog->showGeneratedControlPointContextMenu(
            viewer->surfName(),
            viewer,
            scenePoint,
            globalPos,
            linkCandidateMenuState(*pane->session));
    } else if (_intersectionInspection) {
        const auto contextIt =
            _intersectionInspection->generatedSurfaceContexts.find(viewer->surfName());
        if (contextIt == _intersectionInspection->generatedSurfaceContexts.end() ||
            !contextIt->second.valid) {
            return false;
        }
        const auto& context = contextIt->second;
        double linePosition = context.linePosition;
        if (context.strip) {
            linePosition = vc3d::line_annotation::generatedLinePositionFromStripScene(
                viewer,
                scenePoint);
        } else if (context.follow) {
            const auto& follow = context.sourceSide
                ? _intersectionInspection->sourceFollow
                : _intersectionInspection->targetFollow;
            if (follow.valid && std::isfinite(follow.linePosition)) {
                linePosition = follow.linePosition;
            }
        }
        vc3d::line_annotation::GeneratedControlPointContextMenuOptions options;
        options.parent = viewer;
        options.surfaceName = viewer->surfName();
        options.viewer = viewer;
        options.scenePoint = scenePoint;
        options.globalPos = globalPos;
        options.controlPoints = controlMarkersForSession(*pane->session);
        options.linePointCount = pane->session->optimizedLine.points.empty()
            ? pane->session->controlPoints.size()
            : pane->session->optimizedLine.points.size();
        options.linePosition = linePosition;
        options.stripViewer = context.strip;
        const auto candidateState = linkCandidateMenuState(*pane->session);
        options.linkWithCandidateEnabled = candidateState.enabled;
        options.linkWithCandidateLabel = candidateState.label;
        if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            options.branchLinkDirection = plane->normal({0.0f, 0.0f, 0.0f});
        }
        options.deleteControlPoint = [this, surfaceName = viewer->surfName()](
                                         double selectedLinePosition,
                                         cv::Vec3f selectedPoint) {
            handleGeneratedControlPointDelete(surfaceName,
                                              selectedLinePosition,
                                              selectedPoint);
        };
        options.addBranch = [this, surfaceName = viewer->surfName()](
                                size_t controlPointIndex,
                                cv::Vec3f linkedControlPoint,
                                bool openAfterCreate,
                                cv::Vec3f linkDirection) {
            handleGeneratedControlPointBranch(surfaceName,
                                              controlPointIndex,
                                              linkedControlPoint,
                                              openAfterCreate,
                                              linkDirection);
        };
        options.designateLinkCandidate = [this, surfaceName = viewer->surfName()](
                                             size_t controlPointIndex,
                                             cv::Vec3f volumePoint) {
            handleGeneratedControlPointLinkCandidate(surfaceName,
                                                     controlPointIndex,
                                                     volumePoint);
        };
        options.linkWithCandidate = [this, surfaceName = viewer->surfName()](
                                        size_t controlPointIndex,
                                        cv::Vec3f volumePoint) {
            handleGeneratedControlPointLinkWithCandidate(surfaceName,
                                                         controlPointIndex,
                                                         volumePoint);
        };
        options.unlinkBranch = [this, surfaceName = viewer->surfName()](
                                   size_t controlPointIndex,
                                   uint64_t branchFiberId,
                                   int branchControlPointIndex) {
            handleGeneratedControlPointUnlink(surfaceName,
                                              controlPointIndex,
                                              branchFiberId,
                                              branchControlPointIndex);
        };
        options.setBranchLinkPending = [this, surfaceName = viewer->surfName()](
                                           size_t controlPointIndex,
                                           uint64_t branchFiberId,
                                           int branchControlPointIndex,
                                           bool pending) {
            handleGeneratedControlPointSetLinkPending(surfaceName,
                                                      controlPointIndex,
                                                      branchFiberId,
                                                      branchControlPointIndex,
                                                      pending);
        };
        result = vc3d::line_annotation::showGeneratedControlPointContextMenu(options);
    }
    if (result == LineAnnotationDialog::GeneratedControlPointContextResult::NewLineAnnotationRequested) {
        launchFromViewerAtPoint(viewer, scenePoint, false);
    }
    return result != LineAnnotationDialog::GeneratedControlPointContextResult::None;
}

std::vector<LineAnnotationController::ControlSpanRecord>
LineAnnotationController::controlSpansForFiber(const StoredFiber& fiber)
{
    std::vector<ControlSpanRecord> spans;
    if (fiber.controlPoints.size() < 2 || fiber.linePoints.size() < 2) {
        return spans;
    }

    struct ControlIndex {
        int controlIndex = 0;
        size_t lineIndex = 0;
    };
    std::vector<ControlIndex> controls;
    controls.reserve(fiber.controlPoints.size());
    for (size_t i = 0; i < fiber.controlPoints.size(); ++i) {
        const cv::Vec3d& control = fiber.controlPoints[i];
        if (!finitePoint(control)) {
            continue;
        }
        controls.push_back({
            static_cast<int>(i),
            vc3d::fiber_slice::nearestLinePointIndex(fiber.linePoints, control),
        });
    }
    std::sort(controls.begin(),
              controls.end(),
              [](const ControlIndex& a, const ControlIndex& b) {
                  if (a.lineIndex != b.lineIndex) {
                      return a.lineIndex < b.lineIndex;
                  }
                  return a.controlIndex < b.controlIndex;
              });

    int spanIndex = 0;
    for (size_t i = 1; i < controls.size(); ++i) {
        const auto& left = controls[i - 1];
        const auto& right = controls[i];
        if (right.lineIndex <= left.lineIndex) {
            continue;
        }
        ControlSpanRecord span;
        span.spanIndex = spanIndex++;
        span.firstControlIndex = left.controlIndex;
        span.secondControlIndex = right.controlIndex;
        span.firstLineIndex = left.lineIndex;
        span.lastLineIndex = right.lineIndex;
        span.linePointCount = static_cast<int>(right.lineIndex - left.lineIndex + 1);
        span.lengthVx = polylineLengthRange(fiber.linePoints,
                                            span.firstLineIndex,
                                            span.lastLineIndex);
        spans.push_back(span);
    }
    return spans;
}

LineAnnotationController::FiberSummary::AlignmentMetrics
LineAnnotationController::cachedAlignmentForFiber(uint64_t fiberId) const
{
    auto metricIt = _fiberAlignmentMetrics.find(fiberId);
    if (metricIt != _fiberAlignmentMetrics.end()) {
        return metricIt->second.fiber;
    }
    FiberSummary::AlignmentMetrics metric;
    metric.pending = isAlignmentPendingForFiber(fiberId);
    return metric;
}

LineAnnotationController::FiberSummary::AlignmentMetrics
LineAnnotationController::cachedAlignmentForSpan(uint64_t fiberId, int spanIndex) const
{
    auto metricIt = _fiberAlignmentMetrics.find(fiberId);
    if (metricIt != _fiberAlignmentMetrics.end() &&
        spanIndex >= 0 &&
        static_cast<size_t>(spanIndex) < metricIt->second.spans.size()) {
        return metricIt->second.spans[static_cast<size_t>(spanIndex)];
    }
    FiberSummary::AlignmentMetrics metric;
    metric.pending = isAlignmentPendingForFiber(fiberId);
    return metric;
}

bool LineAnnotationController::hasCachedAlignmentForFiber(uint64_t fiberId) const
{
    return _fiberAlignmentMetrics.find(fiberId) != _fiberAlignmentMetrics.end();
}

bool LineAnnotationController::isAlignmentPendingForFiber(uint64_t fiberId) const
{
    return _pendingFiberAlignmentMetrics.find(fiberId) !=
           _pendingFiberAlignmentMetrics.end();
}

bool LineAnnotationController::isAlignmentPendingForFiber(uint64_t fiberId,
                                                          uint64_t requestToken) const
{
    const auto tokenIt = _pendingFiberAlignmentMetricTokens.find(fiberId);
    return tokenIt != _pendingFiberAlignmentMetricTokens.end() &&
           tokenIt->second == requestToken &&
           isAlignmentPendingForFiber(fiberId);
}

std::optional<std::pair<fs::path, double>>
LineAnnotationController::resolveAlignmentMetricsManifestPath()
{
    if (!_state || !_state->vpkg()) {
        showError(tr("No volume package loaded."));
        return std::nullopt;
    }

    auto vpkg = _state->vpkg();
    try {
        if (const auto resolved = vc3d::opendata::resolveLasagnaForVolume(
                *vpkg, _state->currentVolumeId())) {
            return std::pair{resolved->manifestPath, resolved->workingToBaseScale};
        }
    } catch (const std::exception& ex) {
        showError(tr("Cannot resolve Lasagna for the active volume: %1")
                      .arg(QString::fromStdString(ex.what())));
        return std::nullopt;
    }
    std::string selected = vpkg->selectedLasagnaDataset();
    fs::path manifestPath = vpkg->selectedLasagnaDatasetPath();
    if (!selected.empty() && !manifestPath.empty()) {
        return std::pair{manifestPath, 1.0};
    }

    const fs::path startDir = vpkg->path().empty()
        ? fs::path{}
        : vpkg->path().parent_path();
    auto picked = _datasetPicker ? _datasetPicker(_parentWidget, startDir)
                                 : std::optional<std::string>{};
    if (!picked || picked->empty()) {
        return std::nullopt;
    }
    selected = *picked;
    manifestPath = vc::project::resolveLocalPath(selected, vpkg->path().parent_path());
    vpkg->setSelectedLasagnaDataset(selected);
    return std::pair{manifestPath, 1.0};
}

void LineAnnotationController::requestFiberAlignmentMetricsForFibers(std::vector<uint64_t> fiberIds)
{
    std::vector<uint64_t> orderedFiberIds;
    orderedFiberIds.reserve(fiberIds.size());
    std::unordered_set<uint64_t> seenFiberIds;
    seenFiberIds.reserve(fiberIds.size());
    for (const uint64_t fiberId : fiberIds) {
        if (fiberId == 0 || !seenFiberIds.insert(fiberId).second) {
            continue;
        }
        orderedFiberIds.push_back(fiberId);
    }
    fiberIds = std::move(orderedFiberIds);
    if (fiberIds.empty()) {
        return;
    }

    std::vector<StoredFiber> fibers;
    std::vector<uint64_t> requestTokens;
    fibers.reserve(fiberIds.size());
    requestTokens.reserve(fiberIds.size());
    for (const uint64_t fiberId : fiberIds) {
        if (hasCachedAlignmentForFiber(fiberId)) {
            updateGeneratedViewMetricsForFiber(fiberId);
            continue;
        }
        if (isAlignmentPendingForFiber(fiberId)) {
            updateGeneratedViewMetricsForFiber(fiberId);
            continue;
        }
        auto fiberIt = std::find_if(_fibers.begin(),
                                    _fibers.end(),
                                    [fiberId](const StoredFiber& fiber) {
                                        return fiber.id == fiberId;
                                    });
        if (fiberIt == _fibers.end()) {
            continue;
        }
        _pendingFiberAlignmentMetrics.insert(fiberId);
        const uint64_t requestToken = ++_nextFiberAlignmentMetricToken;
        _pendingFiberAlignmentMetricTokens[fiberId] = requestToken;
        fibers.push_back(*fiberIt);
        requestTokens.push_back(requestToken);
        publishPendingFiberAlignmentMetrics(*fiberIt);
    }
    _fiberMetricsPending = !_pendingFiberAlignmentMetrics.empty();
    if (fibers.empty()) {
        return;
    }

    const auto manifestPath = resolveAlignmentMetricsManifestPath();
    if (!manifestPath) {
        for (const auto& fiber : fibers) {
            _pendingFiberAlignmentMetrics.erase(fiber.id);
            _pendingFiberAlignmentMetricTokens.erase(fiber.id);
            publishUnavailableFiberAlignmentMetrics(fiber.id);
        }
        _fiberMetricsPending = !_pendingFiberAlignmentMetrics.empty();
        return;
    }

    const uint64_t generation = _fiberMetricsGeneration;
    auto* watcher = new QFutureWatcher<FiberMetricsTaskResult>(this);
    _fiberMetricsWatchers.push_back(watcher);
    connect(watcher,
            &QFutureWatcher<FiberMetricsTaskResult>::finished,
            this,
            [this, watcher]() {
                finishFiberAlignmentMetrics(watcher);
                watcher->deleteLater();
            });
    QPointer<LineAnnotationController> self(this);
    watcher->setFuture(QtConcurrent::run([generation,
                                           self,
                                           resolvedManifestPath = manifestPath->first,
                                           workingToBaseScale = manifestPath->second,
                                           fibers = std::move(fibers),
                                           requestTokens = std::move(requestTokens)]() mutable {
        FiberMetricsTaskResult result;
        result.ok = true;
        result.generation = generation;
        result.manifestPath = resolvedManifestPath;
        result.requestedFiberIds.reserve(fibers.size());
        for (size_t i = 0; i < fibers.size(); ++i) {
            const uint64_t fiberId = fibers[i].id;
            result.requestedFiberIds.push_back(fiberId);
            if (i < requestTokens.size()) {
                result.requestTokens.emplace(fiberId, requestTokens[i]);
            }
        }

        auto errorMetricsForFiber = [](const StoredFiber& fiber, const std::string& error) {
            CachedFiberAlignmentMetrics metrics;
            metrics.fiber.error = error;
            metrics.spans.resize(LineAnnotationController::controlSpansForFiber(fiber).size());
            for (auto& spanMetric : metrics.spans) {
                spanMetric.error = error;
            }
            return metrics;
        };

        try {
            vc::lasagna::LasagnaDataset dataset =
                vc::lasagna::LasagnaDataset::open(
                    resolvedManifestPath, {workingToBaseScale});
            vc::lasagna::LasagnaNormalSampler sampler(dataset);
            result.metrics.reserve(fibers.size());
            for (size_t i = 0; i < fibers.size(); ++i) {
                const auto& fiber = fibers[i];
                const std::vector<ControlSpanRecord> spans =
                    LineAnnotationController::controlSpansForFiber(fiber);
                CachedFiberAlignmentMetrics metrics;
                try {
                    metrics = LineAnnotationController::calculateAlignmentMetricsForFiber(
                        fiber,
                        spans,
                        sampler);
                } catch (const std::exception& ex) {
                    metrics = errorMetricsForFiber(fiber, ex.what());
                } catch (...) {
                    metrics = errorMetricsForFiber(
                        fiber,
                        "Unknown Lasagna normal metric calculation error.");
                }
                result.metrics.emplace(fiber.id, metrics);
                if (self) {
                    const uint64_t requestToken =
                        i < requestTokens.size() ? requestTokens[i] : uint64_t{0};
                    QMetaObject::invokeMethod(
                        self.data(),
                        [self, generation, fiberId = fiber.id, requestToken, metrics]() mutable {
                            if (!self ||
                                generation != self->_fiberMetricsGeneration ||
                                !self->isAlignmentPendingForFiber(fiberId, requestToken)) {
                                return;
                            }
                            self->publishFiberAlignmentMetrics(fiberId, std::move(metrics));
                        },
                        Qt::QueuedConnection);
                }
            }
        } catch (const std::exception& ex) {
            result.ok = false;
            result.error = ex.what();
            for (const auto& fiber : fibers) {
                result.metrics.emplace(fiber.id, errorMetricsForFiber(fiber, result.error));
            }
        } catch (...) {
            result.ok = false;
            result.error = "Unknown Lasagna normal metric calculation error.";
            for (const auto& fiber : fibers) {
                result.metrics.emplace(fiber.id, errorMetricsForFiber(fiber, result.error));
            }
        }
        return result;
    }));
}

void LineAnnotationController::publishFiberAlignmentMetrics(
    uint64_t fiberId,
    CachedFiberAlignmentMetrics metrics)
{
    metrics.fiber.pending = false;
    for (auto& spanMetric : metrics.spans) {
        spanMetric.pending = false;
    }
    _pendingFiberAlignmentMetrics.erase(fiberId);
    _pendingFiberAlignmentMetricTokens.erase(fiberId);
    _fiberMetricsPending = !_pendingFiberAlignmentMetrics.empty();
    _fiberAlignmentMetrics[fiberId] = std::move(metrics);
    const auto& cached = _fiberAlignmentMetrics.at(fiberId);
    emit fiberAlignmentMetricsUpdated(fiberId, cached.fiber, cached.spans);
    updateGeneratedViewMetricsForFiber(fiberId);
}

void LineAnnotationController::publishPendingFiberAlignmentMetrics(const StoredFiber& fiber)
{
    CachedFiberAlignmentMetrics metrics;
    metrics.fiber.pending = true;
    metrics.spans.resize(controlSpansForFiber(fiber).size());
    for (auto& spanMetric : metrics.spans) {
        spanMetric.pending = true;
    }
    emit fiberAlignmentMetricsUpdated(fiber.id, metrics.fiber, metrics.spans);
    updateGeneratedViewMetricsForFiber(fiber.id);
}

void LineAnnotationController::publishUnavailableFiberAlignmentMetrics(uint64_t fiberId)
{
    std::vector<FiberSummary::AlignmentMetrics> spanMetrics;
    auto fiberIt = std::find_if(_fibers.begin(),
                                _fibers.end(),
                                [fiberId](const StoredFiber& fiber) {
                                    return fiber.id == fiberId;
                                });
    if (fiberIt != _fibers.end()) {
        spanMetrics.resize(controlSpansForFiber(*fiberIt).size());
    }
    emit fiberAlignmentMetricsUpdated(fiberId, FiberSummary::AlignmentMetrics{}, spanMetrics);
    updateGeneratedViewMetricsForFiber(fiberId);
}

void LineAnnotationController::invalidateFiberAlignmentMetrics(uint64_t fiberId, bool notify)
{
    if (fiberId == 0) {
        return;
    }
    _fiberAlignmentMetrics.erase(fiberId);
    _pendingFiberAlignmentMetrics.erase(fiberId);
    _pendingFiberAlignmentMetricTokens.erase(fiberId);
    _fiberMetricsPending = !_pendingFiberAlignmentMetrics.empty();
    if (notify) {
        publishUnavailableFiberAlignmentMetrics(fiberId);
    }
}

std::vector<LineAnnotationController::FiberSummary> LineAnnotationController::fiberSummaries() const
{
    // Connected-component sizes over the branch-link graph (union-find), so the
    // panel can show how many fibers are chained together through links.
    std::unordered_map<uint64_t, size_t> indexById;
    indexById.reserve(_fibers.size());
    for (size_t i = 0; i < _fibers.size(); ++i) {
        indexById.emplace(_fibers[i].id, i);
    }
    std::vector<size_t> componentParent(_fibers.size());
    for (size_t i = 0; i < componentParent.size(); ++i) {
        componentParent[i] = i;
    }
    const auto findRoot = [&componentParent](size_t index) {
        while (componentParent[index] != index) {
            componentParent[index] = componentParent[componentParent[index]];
            index = componentParent[index];
        }
        return index;
    };
    for (size_t i = 0; i < _fibers.size(); ++i) {
        for (const auto& branch : _fibers[i].branches) {
            const auto targetIt = indexById.find(branch.branchFiberId);
            if (targetIt == indexById.end()) {
                continue;
            }
            componentParent[findRoot(i)] = findRoot(targetIt->second);
        }
    }
    std::unordered_map<size_t, int> componentSizes;
    for (size_t i = 0; i < _fibers.size(); ++i) {
        ++componentSizes[findRoot(i)];
    }

    std::vector<FiberSummary> summaries;
    summaries.reserve(_fibers.size());
    for (const auto& fiber : _fibers) {
        std::vector<FiberSummary::SpanSummary> spanSummaries;
        const std::vector<ControlSpanRecord> spans = controlSpansForFiber(fiber);
        spanSummaries.reserve(spans.size());
        for (const auto& span : spans) {
            FiberSummary::SpanSummary summary;
            summary.spanIndex = span.spanIndex;
            summary.firstControlIndex = span.firstControlIndex;
            summary.secondControlIndex = span.secondControlIndex;
            summary.controlPointCount = 2;
            summary.linePointCount = span.linePointCount;
            summary.lengthVx = span.lengthVx;
            summary.alignment = cachedAlignmentForSpan(fiber.id, span.spanIndex);
            spanSummaries.push_back(std::move(summary));
        }
        const int componentSize = componentSizes[findRoot(indexById.at(fiber.id))];
        const int pendingLinkCount = static_cast<int>(
            std::count_if(fiber.branches.begin(),
                          fiber.branches.end(),
                          [](const FiberBranchRef& branch) { return branch.pending; }));
        summaries.push_back(FiberSummary{
            fiber.id,
            fiber.fileName,
            static_cast<int>(fiber.controlPoints.size()),
            static_cast<int>(fiber.linePoints.size()),
            lineLengthVx(fiber.linePoints),
            cachedAlignmentForFiber(fiber.id),
            std::move(spanSummaries),
            fiber.hvClassification.zDistance,
            fiber.hvClassification.fiberLength,
            fiber.hvClassification.horizontalScore,
            fiber.hvClassification.verticalScore,
            fiber.hvClassification.automaticCertainty,
            vc3d::line_annotation::fiberHvTagToString(fiber.hvClassification.automaticTag),
            fiber.manualHvTag,
            fiber.tags,
            componentSize >= 2 ? componentSize : 0,
            pendingLinkCount,
        });
    }
    std::sort(summaries.begin(), summaries.end(), [](const FiberSummary& a, const FiberSummary& b) {
        return a.id < b.id;
    });
    return summaries;
}

std::vector<LineAnnotationController::FiberLinkOverlayInfo>
LineAnnotationController::fiberLinkOverlayInfos() const
{
    // Union-find mirrors fiberSummaries(): all branches, pending included,
    // unknown targets skipped.
    std::unordered_map<uint64_t, size_t> indexById;
    indexById.reserve(_fibers.size());
    for (size_t i = 0; i < _fibers.size(); ++i) {
        indexById.emplace(_fibers[i].id, i);
    }
    std::vector<size_t> parent(_fibers.size());
    for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = i;
    }
    const auto findRoot = [&parent](size_t index) {
        while (parent[index] != index) {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        return index;
    };
    for (size_t i = 0; i < _fibers.size(); ++i) {
        for (const auto& branch : _fibers[i].branches) {
            const auto targetIt = indexById.find(branch.branchFiberId);
            if (targetIt == indexById.end()) {
                continue;
            }
            parent[findRoot(i)] = findRoot(targetIt->second);
        }
    }
    std::unordered_map<size_t, uint64_t> minIdByRoot;
    for (size_t i = 0; i < _fibers.size(); ++i) {
        auto [it, inserted] = minIdByRoot.emplace(findRoot(i), _fibers[i].id);
        if (!inserted) {
            it->second = std::min(it->second, _fibers[i].id);
        }
    }

    std::vector<FiberLinkOverlayInfo> infos;
    for (size_t i = 0; i < _fibers.size(); ++i) {
        const StoredFiber& fiber = _fibers[i];
        // Pending wins per control point, matching the annotation GUI's
        // marker precedence.
        std::map<int, bool> pendingByControlPoint;
        for (const auto& branch : fiber.branches) {
            if (indexById.find(branch.branchFiberId) == indexById.end()) {
                continue;
            }
            if (branch.controlPointIndex < 0 ||
                branch.controlPointIndex >=
                    static_cast<int>(fiber.controlPoints.size())) {
                continue;
            }
            auto [it, inserted] =
                pendingByControlPoint.emplace(branch.controlPointIndex, branch.pending);
            if (!inserted) {
                it->second = it->second || branch.pending;
            }
        }
        if (pendingByControlPoint.empty()) {
            continue;
        }
        FiberLinkOverlayInfo info;
        info.fiberId = fiber.id;
        info.linkGroupId = minIdByRoot.at(findRoot(i));
        info.linkedControlPoints.assign(pendingByControlPoint.begin(),
                                        pendingByControlPoint.end());
        infos.push_back(std::move(info));
    }
    return infos;
}

QString LineAnnotationController::fiberDisplayName(uint64_t fiberId) const
{
    for (const StoredFiber& fiber : _fibers) {
        if (fiber.id == fiberId) {
            const QString stem =
                vc3d::displayStemForFiberFile(QString::fromStdString(fiber.fileName));
            return stem.isEmpty() ? tr("unnamed") : stem;
        }
    }
    return tr("unnamed");
}

std::vector<std::string> LineAnnotationController::knownFiberTags() const
{
    return _knownFiberTags;
}

std::vector<vc::atlas::FiberPolyline> LineAnnotationController::fiberSnapshots() const
{
    std::vector<vc::atlas::FiberPolyline> snapshots;
    snapshots.reserve(_fibers.size());
    for (const auto& fiber : _fibers) {
        vc::atlas::FiberPolyline snapshot;
        snapshot.id = fiber.id;
        snapshot.generation = fiber.generation;
        snapshot.controlPoints = fiber.controlPoints;
        snapshot.points.reserve(fiber.linePoints.size());
        for (const auto& point : fiber.linePoints) {
            snapshot.points.push_back(vc::atlas::FiberPoint{point, std::nullopt});
        }
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

std::vector<vc::atlas::FiberPolyline> LineAnnotationController::fiberSnapshotsFromStorage() const
{
    std::vector<vc::atlas::FiberPolyline> snapshots;
    const auto snapshotsWithPaths = fiberSnapshotsFromStorageWithPaths();
    snapshots.reserve(snapshotsWithPaths.size());
    for (const auto& snapshot : snapshotsWithPaths) {
        snapshots.push_back(snapshot.fiber);
    }
    return snapshots;
}

std::vector<LineAnnotationController::FiberSnapshotWithPath>
LineAnnotationController::fiberSnapshotsFromStorageWithPaths() const
{
    auto snapshotForFiber = [](const StoredFiber& fiber) {
        vc::atlas::FiberPolyline snapshot;
        snapshot.id = 0;
        snapshot.generation = fiber.generation;
        snapshot.controlPoints = fiber.controlPoints;
        snapshot.points.reserve(fiber.linePoints.size());
        for (const auto& point : fiber.linePoints) {
            snapshot.points.push_back(vc::atlas::FiberPoint{point, std::nullopt});
        }
        return snapshot;
    };
    std::map<fs::path, FiberSnapshotWithPath> byPath;
    const fs::path dir = fibersDir();
    std::error_code ec;
    if (!dir.empty() && fs::exists(dir, ec)) {
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_regular_file() || entry.path().extension() != ".json") {
                continue;
            }
            try {
                if (auto fiber = loadFiberFile(entry.path())) {
                    const fs::path path = relativeFiberPath(*fiber);
                    byPath[path] = FiberSnapshotWithPath{
                        path,
                        snapshotForFiber(*fiber),
                        fiber->id,
                        fiber->hvClassification,
                        fiber->manualHvTag,
                        fiber->tags};
                }
            } catch (const std::exception& ex) {
                Logger()->warn("Skipping invalid VC3D fiber file {} during atlas search: {}",
                               entry.path().string(),
                               ex.what());
            }
        }
    }

    for (const auto& fiber : _fibers) {
        const fs::path path = relativeFiberPath(fiber);
        byPath[path] = FiberSnapshotWithPath{
            path,
            snapshotForFiber(fiber),
            fiber.id,
            fiber.hvClassification,
            fiber.manualHvTag,
            fiber.tags};
    }

    std::vector<fs::path> orderedPaths;
    orderedPaths.reserve(byPath.size());
    for (const auto& [path, snapshot] : byPath) {
        (void)snapshot;
        orderedPaths.push_back(path);
    }
    const auto runtimeIds = vc::atlas::makeFiberRuntimeIdentityMap(orderedPaths);

    std::vector<FiberSnapshotWithPath> snapshots;
    snapshots.reserve(byPath.size());
    for (auto& [path, snapshot] : byPath) {
        snapshot.fiber.id = runtimeIds.idForPath(path);
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

std::optional<uint64_t> LineAnnotationController::fiberIdForAtlasPath(
    const fs::path& atlasFiberPath) const
{
    const std::string targetKey = vc::atlas::atlasFiberPathKey(atlasFiberPath);
    for (const auto& fiber : _fibers) {
        const auto keys = atlasPathKeysForFiber(fiber);
        if (std::find(keys.begin(), keys.end(), targetKey) != keys.end()) {
            return fiber.id;
        }
    }
    return std::nullopt;
}

void LineAnnotationController::onSurfaceChanged(std::string name,
                                                std::shared_ptr<Surface> surf,
                                                bool /*isEditUpdate*/)
{
    if (name != "segmentation" || !_state) {
        return;
    }
    for (const auto& pane : _panes) {
        if (pane.sourceKind == SourceKind::Segmentation) {
            _state->setSurface(pane.surfaceName, surf);
        }
    }
}

void LineAnnotationController::onVolumePackageChanged(std::shared_ptr<VolumePkg> /*pkg*/)
{
    loadFibersForCurrentPackage();
}

void LineAnnotationController::handleLineSeed(const std::string& surfaceName,
                                              cv::Vec3f volumePoint,
                                              InitialDirectionMode directionMode)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }

    if (!ensureDatasetForSession(session)) {
        return;
    }

    session.initialDirectionMode = directionMode;
    ensureSessionFiberIdentity(session);
    const cv::Vec3d seedPoint(volumePoint[0], volumePoint[1], volumePoint[2]);
    session.seedPoint = seedPoint;
    session.focusedLinePosition = 0.0;
    session.focusedControlPoint = seedPoint;
    session.controlPoints = {{0.0, seedPoint, true, -1}};
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    syncLinkedBranchMetadataAfterFiberModification(session);
    startOptimization(session);
}

void LineAnnotationController::handleGeneratedControlPoint(const std::string& surfaceName,
                                                          cv::Vec3f volumePoint,
                                                          double linePosition)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (session.optimizedLine.points.empty() || session.controlPoints.empty()) {
        return;
    }
    if (!ensureDatasetForSession(session)) {
        return;
    }

    if (session.fiberId != 0 && session.fiberMetricsMatchStoredFiber) {
        session.fiberMetricsMatchStoredFiber = false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
    }

    const double maxPosition = static_cast<double>(session.optimizedLine.points.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
    if (pane->dialog && pane->dialog->maxControlPointDistanceVx() > 0) {
        std::vector<double> controlLinePositions;
        controlLinePositions.reserve(session.controlPoints.size());
        for (const auto& control : session.controlPoints) {
            controlLinePositions.push_back(control.linePosition);
        }
        if (!vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
                linePosition,
                controlLinePositions,
                static_cast<double>(pane->dialog->maxControlPointDistanceVx()))) {
            return;
        }
    }
    const cv::Vec3d clicked(volumePoint[0], volumePoint[1], volumePoint[2]);

    auto nearest = session.controlPoints.end();
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (auto it = session.controlPoints.begin(); it != session.controlPoints.end(); ++it) {
        if (!std::isfinite(it->linePosition)) {
            continue;
        }
        const double distance = std::abs(it->linePosition - linePosition);
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearest = it;
        }
    }

    size_t changedControlIndex = 0;
    bool editedExistingControl = false;
    if (nearest != session.controlPoints.end() && nearestDistance <= 0.5) {
        editedExistingControl = true;
        changedControlIndex = static_cast<size_t>(std::distance(session.controlPoints.begin(), nearest));
        if (!confirmLinkedControlPointEdit(session,
                                           static_cast<int>(changedControlIndex),
                                           tr("Moving it"))) {
            return;
        }
        nearest->volumePoint = clicked;
        nearest->optimizedIndex = -1;
        linePosition = nearest->linePosition;
        if (nearest->isSeed) {
            session.seedPoint = clicked;
        }
    } else {
        session.controlPoints.push_back({linePosition, clicked, false, -1});
        changedControlIndex = session.controlPoints.size() - 1;
    }

    session.focusedLinePosition = linePosition;
    session.focusedControlPoint = clicked;
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    const bool autoReoptimize =
        !pane->dialog ||
        pane->dialog->reoptimizationMode() ==
            LineAnnotationDialog::ReoptimizationMode::AutoReoptimize;
    if (!autoReoptimize) {
        const std::string noReoptEventName = editedExistingControl
            ? "control_edit_no_reopt"
            : "control_add_no_reopt";
        writeLineDebugJson(noReoptEventName,
                           session.controlPoints,
                           linePointsToJson(session.optimizedLine));
        syncLinkedBranchMetadataAfterFiberModification(session);
        if (pane->dialog) {
            pane->dialog->setGeneratedBranchOverlayData(
                controlMarkersForSession(session),
                generatedBranchLinePointsForSession(session),
                generatedBranchLinkMarkers(session.branches));
            pane->dialog->setGeneratedPredSnapPoints(
                generatedPredSnapMarkers(session.controlPoints, session.predSnapSet));
        }
        return;
    }

    std::vector<cv::Vec3d> currentLinePoints;
    currentLinePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        currentLinePoints.push_back(point.position);
    }

    const std::vector<vc::lasagna::LineControlPoint> branchRemapControls = session.controlPoints;
    const std::vector<FiberBranchRef> branchRemapBranches = session.branches;

    vc::lasagna::LineControlPointUpdateResult update;
    const std::string updateEventName = editedExistingControl
        ? "control_edit_span_update"
        : "control_add_span_update";
    const auto updateStart = Clock::now();
    try {
        vc::lasagna::LineOptimizationConfig updateConfig;
        const int initialCenterlineLengthVx = pane->dialog
            ? pane->dialog->initialCenterlineLengthVx()
            : vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX_DEFAULT;
        const auto discretization = initialLineDiscretization(initialCenterlineLengthVx);
        updateConfig.segmentsPerSide = discretization.segmentsPerSide;
        updateConfig.segmentLength = discretization.segmentLength;
        update = vc::lasagna::updateExistingLineControlPoint(std::move(currentLinePoints),
                                                             std::move(session.controlPoints),
                                                             changedControlIndex,
                                                             *session.normalSampler,
                                                             updateConfig);
    } catch (const std::exception& ex) {
        showError(tr("Could not update line control point: %1").arg(QString::fromStdString(ex.what())));
        return;
    }
    session.optimizedLine = lineModelFromPoints(update.linePoints, session.normalSampler.get());
    session.controlPoints = update.controlPoints;
    syncLinkedBranchMetadataAfterFiberModification(
        session,
        &branchRemapControls,
        &branchRemapBranches);
    if (update.changedControlIndex >= 0 &&
        update.changedControlIndex < static_cast<int>(session.controlPoints.size())) {
        const auto& changed = session.controlPoints[static_cast<size_t>(update.changedControlIndex)];
        session.focusedLinePosition = changed.linePosition;
        session.focusedControlPoint = changed.volumePoint;
        if (changed.isSeed) {
            session.seedPoint = changed.volumePoint;
        }
    }
    const double updateMs = elapsedMs(updateStart, Clock::now());
    Logger()->info("Line annotation Lasagna stage timing: event={} overall_ms={:.3f} points={}",
                   updateEventName,
                   updateMs,
                   session.optimizedLine.points.size());
    writeLineDebugJson(updateEventName,
                       session.controlPoints,
                       linePointsToJson(session.optimizedLine));
    startOptimization(session, false, update.activeStart, update.activeEnd);
}

void LineAnnotationController::handleGeneratedControlPointBranch(const std::string& surfaceName,
                                                                size_t controlPointIndex,
                                                                cv::Vec3f linkedControlPoint,
                                                                bool openAfterCreate,
                                                                cv::Vec3f requestedLinkDirection)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& parentSession = *pane->session;
    if (parentSession.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (controlPointIndex >= parentSession.controlPoints.size()) {
        return;
    }
    if (controlPointHasBranchLink(parentSession.branches, controlPointIndex)) {
        showError(tr("This control point is already linked; unlink it first."));
        return;
    }
    if (openAfterCreate && !ensureDatasetForSession(parentSession)) {
        return;
    }

    ensureSessionFiberIdentity(parentSession);
    if (parentSession.fiberId == 0) {
        parentSession.fiberId = nextFiberId();
    }
    const uint64_t parentFiberId = parentSession.fiberId;
    const std::string parentFileName = parentSession.fiberFileName;
    const cv::Vec3d branchPoint = parentSession.controlPoints[controlPointIndex].volumePoint;
    if (!finitePoint(branchPoint)) {
        return;
    }
    const cv::Vec3d linkedPoint = toVec3d(linkedControlPoint);
    if (!finitePoint(linkedPoint)) {
        showError(tr("Could not determine a finite linked-fiber point from the clicked location."));
        return;
    }
    cv::Vec3d linkDirection = toVec3d(requestedLinkDirection);
    if (!finiteDirection(linkDirection)) {
        showError(tr("Could not determine a finite linked-fiber direction from the current view."));
        return;
    }
    linkDirection = normalizedOrZero(linkDirection);
    const std::vector<cv::Vec3d> parentLinePoints =
        linePointPositions(parentSession.optimizedLine);
    const cv::Vec3d parentEndpointDirection =
        endpointTangentFromLinePoints(parentLinePoints, branchPoint, linkDirection);
    const cv::Vec3d linkedInitialDirection = linkDirection;

    auto linkedSeedRecord = std::make_shared<LineAnnotationSession>();
    linkedSeedRecord->fiberId = std::max(nextFiberId(), parentFiberId + 1);
    linkedSeedRecord->sourceSliceNormal = linkedInitialDirection;
    linkedSeedRecord->initialDirectionMode = InitialDirectionMode::ZInOut;
    linkedSeedRecord->selectedDatasetLocation = parentSession.selectedDatasetLocation;
    linkedSeedRecord->selectedManifestPath = parentSession.selectedManifestPath;
    linkedSeedRecord->workingToBaseScale = parentSession.workingToBaseScale;
    linkedSeedRecord->dataset = parentSession.dataset;
    linkedSeedRecord->normalSampler = parentSession.normalSampler;
    ensureSessionFiberIdentity(*linkedSeedRecord);

    FiberBranchRef parentToLinked;
    parentToLinked.controlPointIndex = static_cast<int>(controlPointIndex);
    parentToLinked.branchFiberId = linkedSeedRecord->fiberId;
    parentToLinked.branchControlPointIndex = 0;
    parentToLinked.branchFileName = linkedSeedRecord->fiberFileName;
    parentToLinked.controlPointDirection = parentEndpointDirection;
    parentToLinked.branchControlPointDirection = linkedInitialDirection;
    parentToLinked.controlPointPosition = branchPoint;
    parentToLinked.branchControlPointPosition = linkedPoint;
    parentToLinked.pending = true;

    const auto duplicateParentLink = std::find_if(
        parentSession.branches.begin(),
        parentSession.branches.end(),
        [&parentToLinked](const FiberBranchRef& branch) {
            return branch.controlPointIndex == parentToLinked.controlPointIndex &&
                   branch.branchFiberId == parentToLinked.branchFiberId &&
                   branch.branchControlPointIndex == parentToLinked.branchControlPointIndex;
        });
    if (duplicateParentLink == parentSession.branches.end()) {
        parentSession.branches.push_back(parentToLinked);
    }

    FiberBranchRef linkedToParent;
    linkedToParent.controlPointIndex = 0;
    linkedToParent.branchFiberId = parentFiberId;
    linkedToParent.branchControlPointIndex = static_cast<int>(controlPointIndex);
    linkedToParent.branchFileName = parentFileName;
    linkedToParent.controlPointDirection = linkedInitialDirection;
    linkedToParent.branchControlPointDirection = parentEndpointDirection;
    linkedToParent.controlPointPosition = linkedPoint;
    linkedToParent.branchControlPointPosition = branchPoint;
    linkedToParent.pending = true;
    linkedSeedRecord->branches.push_back(linkedToParent);
    linkedSeedRecord->showLinkedLineOverlays = false;
    linkedSeedRecord->surfaceName = "linked_fiber_create_only";
    linkedSeedRecord->sourceAnnotationSurfaceName = linkedSeedRecord->surfaceName;
    linkedSeedRecord->seedPoint = linkedPoint;
    linkedSeedRecord->focusedLinePosition = 0.0;
    linkedSeedRecord->focusedControlPoint = linkedPoint;
    linkedSeedRecord->controlPoints = {{0.0, linkedPoint, true, -1}};
    linkedSeedRecord->optimizedLine = syntheticLineModelFromPoints({linkedPoint});
    linkedSeedRecord->taskState = LineAnnotationSession::TaskState::Succeeded;
    linkedSeedRecord->optimizationState = SessionOptimizationState::Optimized;

    uint64_t linkedFiberId = linkedSeedRecord->fiberId;
    try {
        StoredFiber parentFiber = storedFiberFromSession(parentSession);
        StoredFiber linkedFiber = storedFiberFromSession(*linkedSeedRecord);
        linkedFiberId = linkedFiber.id;
        scheduleFiberPairSave(parentFiber, linkedFiber);

        auto upsertFiber = [this](StoredFiber fiber) {
            auto it = std::find_if(
                _fibers.begin(),
                _fibers.end(),
                [&fiber](const StoredFiber& existing) {
                    return (!fiber.fileName.empty() && existing.fileName == fiber.fileName) ||
                           existing.id == fiber.id;
                });
            if (it == _fibers.end()) {
                _fibers.push_back(std::move(fiber));
            } else {
                *it = std::move(fiber);
            }
        };
        upsertFiber(std::move(parentFiber));
        upsertFiber(std::move(linkedFiber));
    } catch (const std::exception& ex) {
        parentSession.branches.erase(
            std::remove_if(parentSession.branches.begin(),
                           parentSession.branches.end(),
                           [&parentToLinked](const FiberBranchRef& branch) {
                               return branch.branchFiberId == parentToLinked.branchFiberId &&
                                      branch.controlPointIndex ==
                                          parentToLinked.controlPointIndex &&
                                      branch.branchControlPointIndex ==
                                          parentToLinked.branchControlPointIndex;
                           }),
            parentSession.branches.end());
        if (pane->dialog) {
            pane->dialog->setGeneratedBranchOverlayData(
                controlMarkersForSession(parentSession),
                generatedBranchLinePointsForSession(parentSession),
                generatedBranchLinkMarkers(parentSession.branches),
                true);
        }
        showError(tr("Could not save linked fiber: %1")
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    parentSession.fiberMetricsMatchStoredFiber = true;
    invalidateFiberAlignmentMetrics(parentFiberId, true);
    emitFiberSummaries();
    if (pane->dialog) {
        auto controlMarkers = controlMarkersForSession(parentSession);
        auto branchLinePoints = generatedBranchLinePointsForSession(parentSession);
        auto branchLinks = generatedBranchLinkMarkers(parentSession.branches);
        pane->dialog->setGeneratedBranchOverlayData(
            std::move(controlMarkers),
            std::move(branchLinePoints),
            std::move(branchLinks),
            true);
    }
    if (openAfterCreate) {
        openFiberAtControlPoint(linkedFiberId, 0);
        return;
    }
    refreshBranchLineViews(parentFiberId);
}

void LineAnnotationController::handleGeneratedControlPointLinkCandidate(
    const std::string& surfaceName,
    size_t controlPointIndex,
    cv::Vec3f volumePoint)
{
    (void)volumePoint;
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (controlPointIndex >= session.controlPoints.size()) {
        return;
    }
    if (controlPointHasBranchLink(session.branches, controlPointIndex)) {
        showError(tr("This control point is already linked; unlink it first."));
        return;
    }
    const cv::Vec3d candidatePosition = session.controlPoints[controlPointIndex].volumePoint;
    if (!finitePoint(candidatePosition)) {
        showError(tr("Could not determine a finite position for the link candidate."));
        return;
    }
    ensureSessionFiberIdentity(session);
    if (session.fiberId == 0) {
        session.fiberId = nextFiberId();
    }

    LinkCandidate candidate;
    candidate.fiberId = session.fiberId;
    candidate.fiberFileName = session.fiberFileName;
    candidate.position = candidatePosition;
    const auto storedIndexMap = storedIndexMapForSessionControls(session.controlPoints);
    if (controlPointIndex < storedIndexMap.size()) {
        candidate.storedControlPointIndexHint = storedIndexMap[controlPointIndex];
    }
    _linkCandidate = candidate;

    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            false);
    }
}

void LineAnnotationController::handleGeneratedOpenNearbyAnnotation(uint64_t fiberId,
                                                                   cv::Vec3f volumePoint)
{
    if (fiberId == 0) {
        return;
    }
    const auto fiberIt = std::find_if(_fibers.begin(),
                                      _fibers.end(),
                                      [fiberId](const StoredFiber& fiber) {
                                          return fiber.id == fiberId;
                                      });
    if (fiberIt == _fibers.end()) {
        showError(tr("Could not find fiber %1 for the nearby annotation.")
                      .arg(static_cast<qulonglong>(fiberId)));
        return;
    }
    if (fiberIt->linePoints.empty() || !finitePoint(toVec3d(volumePoint))) {
        openFiber(fiberId);
        return;
    }
    const size_t linePointIndex =
        vc3d::fiber_slice::nearestLinePointIndex(fiberIt->linePoints, toVec3d(volumePoint));
    openFiberAtLinePointIndex(fiberId, static_cast<int>(linePointIndex));
}

void LineAnnotationController::handleGeneratedControlPointLinkWithCandidate(
    const std::string& surfaceName,
    size_t controlPointIndex,
    cv::Vec3f volumePoint)
{
    (void)volumePoint;
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (controlPointIndex >= session.controlPoints.size()) {
        return;
    }
    if (controlPointHasBranchLink(session.branches, controlPointIndex)) {
        showError(tr("This control point is already linked; unlink it first."));
        return;
    }
    if (!_linkCandidate || _linkCandidate->fiberId == 0) {
        showError(tr("No link candidate is designated."));
        return;
    }

    ensureSessionFiberIdentity(session);
    if (session.fiberId == 0) {
        session.fiberId = nextFiberId();
    }
    if (session.fiberId == _linkCandidate->fiberId) {
        showError(tr("The link candidate is on this fiber; designate a candidate on a "
                     "different fiber."));
        return;
    }

    const LinkCandidate candidate = *_linkCandidate;
    const auto findFarFiber = [this, &candidate]() {
        return std::find_if(_fibers.begin(),
                            _fibers.end(),
                            [&candidate](const StoredFiber& fiber) {
                                return fiber.id == candidate.fiberId;
                            });
    };
    auto farIt = findFarFiber();
    if (farIt == _fibers.end()) {
        _linkCandidate.reset();
        showError(tr("The link candidate's fiber no longer exists."));
        return;
    }
    const auto farControlIndex = matchingStoredControlPointIndex(
        farIt->controlPoints,
        candidate.storedControlPointIndexHint,
        candidate.position);
    if (!farControlIndex) {
        _linkCandidate.reset();
        showError(tr("The link candidate control point no longer exists."));
        return;
    }
    if (controlPointHasBranchLink(farIt->branches,
                                  static_cast<size_t>(*farControlIndex))) {
        showError(tr("The link candidate is already linked; unlink it first."));
        return;
    }
    const cv::Vec3d farPoint = farIt->controlPoints[static_cast<size_t>(*farControlIndex)];
    const cv::Vec3d localPoint = session.controlPoints[controlPointIndex].volumePoint;
    if (!finitePoint(localPoint) || !finitePoint(farPoint)) {
        showError(tr("Could not determine finite endpoints for the link."));
        return;
    }
    const cv::Vec3d fallbackDirection = normalizedOrZero(farPoint - localPoint);
    const cv::Vec3d localDirection = endpointTangentFromLinePoints(
        linePointPositions(session.optimizedLine),
        localPoint,
        fallbackDirection);
    const cv::Vec3d farDirection = endpointTangentFromLinePoints(
        farIt->linePoints,
        farPoint,
        -fallbackDirection);
    if (!finiteDirection(localDirection) || !finiteDirection(farDirection)) {
        showError(tr("Could not determine finite link directions between the control points."));
        return;
    }

    FiberBranchRef localRef;
    localRef.controlPointIndex = static_cast<int>(controlPointIndex);
    localRef.branchFiberId = farIt->id;
    localRef.branchControlPointIndex = *farControlIndex;
    localRef.branchFileName = farIt->fileName;
    localRef.controlPointDirection = localDirection;
    localRef.branchControlPointDirection = farDirection;
    localRef.controlPointPosition = localPoint;
    localRef.branchControlPointPosition = farPoint;
    localRef.pending = true;

    const auto duplicateLocal = std::find_if(
        session.branches.begin(),
        session.branches.end(),
        [&localRef](const FiberBranchRef& branch) {
            return branch.branchFiberId == localRef.branchFiberId &&
                   branch.controlPointIndex == localRef.controlPointIndex &&
                   branch.branchControlPointIndex == localRef.branchControlPointIndex;
        });
    if (duplicateLocal != session.branches.end()) {
        showError(tr("These control points are already linked."));
        return;
    }
    session.branches.push_back(localRef);

    const uint64_t localFiberId = session.fiberId;
    const uint64_t farFiberId = farIt->id;
    try {
        StoredFiber localFiber = storedFiberFromSession(session);
        const auto storedLocalBranch = std::find_if(
            localFiber.branches.begin(),
            localFiber.branches.end(),
            [farFiberId, &localPoint, &farPoint](const FiberBranchRef& branch) {
                return branch.branchFiberId == farFiberId &&
                       pointsApproximatelyEqual(branch.controlPointPosition, localPoint) &&
                       pointsApproximatelyEqual(branch.branchControlPointPosition, farPoint);
            });
        if (storedLocalBranch == localFiber.branches.end()) {
            throw std::runtime_error("link entry missing after serialization");
        }

        FiberBranchRef reciprocal;
        reciprocal.controlPointIndex = storedLocalBranch->branchControlPointIndex;
        reciprocal.branchFiberId = localFiber.id;
        reciprocal.branchControlPointIndex = storedLocalBranch->controlPointIndex;
        reciprocal.branchFileName = localFiber.fileName;
        reciprocal.controlPointDirection = storedLocalBranch->branchControlPointDirection;
        reciprocal.branchControlPointDirection = storedLocalBranch->controlPointDirection;
        reciprocal.controlPointPosition = storedLocalBranch->branchControlPointPosition;
        reciprocal.branchControlPointPosition = storedLocalBranch->controlPointPosition;
        reciprocal.pending = storedLocalBranch->pending;

        // storedFiberFromSession runs the branch metadata sync hook, which may
        // touch _fibers; re-find the far fiber before mutating it.
        farIt = findFarFiber();
        if (farIt == _fibers.end()) {
            throw std::runtime_error("linked fiber disappeared while saving");
        }
        const auto duplicateFar = std::find_if(
            farIt->branches.begin(),
            farIt->branches.end(),
            [&reciprocal](const FiberBranchRef& branch) {
                return branch.branchFiberId == reciprocal.branchFiberId &&
                       branch.controlPointIndex == reciprocal.controlPointIndex &&
                       branch.branchControlPointIndex == reciprocal.branchControlPointIndex;
            });
        if (duplicateFar == farIt->branches.end()) {
            farIt->branches.push_back(reciprocal);
        }
        const StoredFiber farFiber = *farIt;
        scheduleFiberPairSave(localFiber, farFiber);

        auto localIt = std::find_if(
            _fibers.begin(),
            _fibers.end(),
            [&localFiber](const StoredFiber& existing) {
                return (!localFiber.fileName.empty() &&
                        existing.fileName == localFiber.fileName) ||
                       existing.id == localFiber.id;
            });
        if (localIt == _fibers.end()) {
            _fibers.push_back(std::move(localFiber));
        } else {
            *localIt = std::move(localFiber);
        }
    } catch (const std::exception& ex) {
        session.branches.erase(
            std::remove_if(session.branches.begin(),
                           session.branches.end(),
                           [&localRef](const FiberBranchRef& branch) {
                               return branch.branchFiberId == localRef.branchFiberId &&
                                      branch.controlPointIndex == localRef.controlPointIndex &&
                                      branch.branchControlPointIndex ==
                                          localRef.branchControlPointIndex;
                           }),
            session.branches.end());
        if (pane->dialog) {
            pane->dialog->setGeneratedBranchOverlayData(
                controlMarkersForSession(session),
                generatedBranchLinePointsForSession(session),
                generatedBranchLinkMarkers(session.branches),
                true);
        }
        showError(tr("Could not save linked fibers: %1")
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    _linkCandidate.reset();
    session.fiberMetricsMatchStoredFiber = true;
    invalidateFiberAlignmentMetrics(localFiberId, true);
    invalidateFiberAlignmentMetrics(farFiberId, true);
    emitFiberSummaries();
    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true);
    }
    refreshBranchLineViews(localFiberId);
}

void LineAnnotationController::handleGeneratedControlPointUnlink(
    const std::string& surfaceName,
    size_t controlPointIndex,
    uint64_t branchFiberId,
    int branchControlPointIndex)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (controlPointIndex >= session.controlPoints.size() || branchFiberId == 0) {
        return;
    }

    const std::vector<FiberBranchRef> previousBranches = session.branches;
    session.branches.erase(
        std::remove_if(session.branches.begin(),
                       session.branches.end(),
                       [&](const FiberBranchRef& branch) {
                           return branch.controlPointIndex ==
                                      static_cast<int>(controlPointIndex) &&
                                  branch.branchFiberId == branchFiberId &&
                                  (branchControlPointIndex < 0 ||
                                   branch.branchControlPointIndex ==
                                       branchControlPointIndex);
                       }),
        session.branches.end());
    if (session.branches.size() == previousBranches.size()) {
        showError(tr("Could not find the link to remove."));
        return;
    }

    const BranchMetadataSyncResult branchSync =
        syncLinkedBranchMetadataAfterFiberModification(session, nullptr, &previousBranches);
    scheduleBranchMetadataSaves(branchSync.affectedFiberIds, session.fiberId);

    try {
        StoredFiber localFiber = storedFiberFromSession(session);
        scheduleFiberSave(localFiber);
        auto localIt = std::find_if(
            _fibers.begin(),
            _fibers.end(),
            [&localFiber](const StoredFiber& existing) {
                return (!localFiber.fileName.empty() &&
                        existing.fileName == localFiber.fileName) ||
                       existing.id == localFiber.id;
            });
        if (localIt == _fibers.end()) {
            _fibers.push_back(std::move(localFiber));
        } else {
            *localIt = std::move(localFiber);
        }
        session.fiberMetricsMatchStoredFiber = true;
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber after unlinking: %1")
                      .arg(QString::fromStdString(ex.what())));
    }

    emitFiberSummaries();
    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true);
    }
    refreshBranchLineViews(session.fiberId);
    refreshBranchLineViews(branchFiberId);
}

void LineAnnotationController::handleGeneratedControlPointSetLinkPending(
    const std::string& surfaceName,
    size_t controlPointIndex,
    uint64_t branchFiberId,
    int branchControlPointIndex,
    bool pending)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (controlPointIndex >= session.controlPoints.size() || branchFiberId == 0) {
        return;
    }

    std::vector<FiberBranchRef> updatedBranches;
    for (auto& branch : session.branches) {
        if (branch.controlPointIndex == static_cast<int>(controlPointIndex) &&
            branch.branchFiberId == branchFiberId &&
            (branchControlPointIndex < 0 ||
             branch.branchControlPointIndex == branchControlPointIndex)) {
            branch.pending = pending;
            updatedBranches.push_back(branch);
        }
    }
    if (updatedBranches.empty()) {
        showError(tr("Could not find the link to update."));
        return;
    }

    // Flip the reciprocal refs in the linked fiber, whether it is open in a
    // pane or only loaded from storage. If it is neither, only the local side
    // is updated/saved (same semantics as unlinking).
    std::vector<uint64_t> affectedFiberIds;
    auto updateReciprocals = [&session, pending](std::vector<FiberBranchRef>& targetBranches,
                                                 const FiberBranchRef& updatedBranch) {
        bool changed = false;
        for (auto& candidate : targetBranches) {
            const bool pointsToSession =
                candidate.branchFiberId == session.fiberId ||
                (!session.fiberFileName.empty() &&
                 candidate.branchFileName == session.fiberFileName);
            const bool sameLocalEndpoint =
                candidate.branchControlPointIndex == updatedBranch.controlPointIndex ||
                pointsApproximatelyEqual(candidate.branchControlPointPosition,
                                         updatedBranch.controlPointPosition);
            if (pointsToSession &&
                candidate.controlPointIndex == updatedBranch.branchControlPointIndex &&
                sameLocalEndpoint) {
                candidate.pending = pending;
                changed = true;
            }
        }
        return changed;
    };
    for (const auto& updatedBranch : updatedBranches) {
        for (const auto& otherPane : _panes) {
            if (!otherPane.session ||
                !branchReferencesFiber(updatedBranch,
                                       otherPane.session->fiberId,
                                       otherPane.session->fiberFileName)) {
                continue;
            }
            if (updateReciprocals(otherPane.session->branches, updatedBranch)) {
                addUniqueFiberId(affectedFiberIds, otherPane.session->fiberId);
            }
        }
        for (auto& fiber : _fibers) {
            if (!branchReferencesFiber(updatedBranch, fiber.id, fiber.fileName)) {
                continue;
            }
            if (updateReciprocals(fiber.branches, updatedBranch)) {
                addUniqueFiberId(affectedFiberIds, fiber.id);
            }
        }
    }
    scheduleBranchMetadataSaves(affectedFiberIds, session.fiberId);

    try {
        StoredFiber localFiber = storedFiberFromSession(session);
        scheduleFiberSave(localFiber);
        auto localIt = std::find_if(
            _fibers.begin(),
            _fibers.end(),
            [&localFiber](const StoredFiber& existing) {
                return (!localFiber.fileName.empty() &&
                        existing.fileName == localFiber.fileName) ||
                       existing.id == localFiber.id;
            });
        if (localIt == _fibers.end()) {
            _fibers.push_back(std::move(localFiber));
        } else {
            *localIt = std::move(localFiber);
        }
        session.fiberMetricsMatchStoredFiber = true;
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber after updating the link: %1")
                      .arg(QString::fromStdString(ex.what())));
    }

    emitFiberSummaries();
    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true);
    }
    refreshBranchLineViews(session.fiberId);
    refreshBranchLineViews(branchFiberId);
}

void LineAnnotationController::handleGeneratedPredSnapPoint(const std::string& surfaceName,
                                                           cv::Vec3f volumePoint)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (!session.atlasDir || session.atlasDir->empty() || session.controlPoints.empty()) {
        return;
    }

    const cv::Vec3d clicked(volumePoint[0], volumePoint[1], volumePoint[2]);
    if (!finitePoint(clicked)) {
        return;
    }

    size_t nearestIndex = 0;
    double nearestDistanceSq = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < session.controlPoints.size(); ++i) {
        const cv::Vec3d delta = session.controlPoints[i].volumePoint - clicked;
        const double distanceSq = delta.dot(delta);
        if (distanceSq < nearestDistanceSq) {
            nearestIndex = i;
            nearestDistanceSq = distanceSq;
        }
    }

    fs::path atlasFiberPath = session.atlasFiberPath;
    if (atlasFiberPath.empty()) {
        return;
    }

    std::optional<double> predDtValue;
    if (session.normalSampler) {
        predDtValue = session.normalSampler->samplePredDt(clicked);
    }

    try {
        session.predSnapSet = vc::atlas::setManualAtlasPredSnapPoint(
            *session.atlasDir,
            atlasFiberPath,
            session.controlPoints[nearestIndex].volumePoint,
            clicked,
            predDtValue);
        if (pane->dialog) {
            pane->dialog->setGeneratedPredSnapPoints(
                generatedPredSnapMarkers(session.controlPoints, session.predSnapSet));
        }
    } catch (const std::exception& ex) {
        showError(tr("Could not save pred-snap point: %1")
                      .arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::handleGeneratedControlPointDelete(const std::string& surfaceName,
                                                                double linePosition,
                                                                cv::Vec3f volumePoint)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (session.optimizedLine.points.empty() || session.controlPoints.size() <= 1) {
        return;
    }
    if (!ensureDatasetForSession(session)) {
        return;
    }

    if (session.fiberId != 0 && session.fiberMetricsMatchStoredFiber) {
        session.fiberMetricsMatchStoredFiber = false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
    }

    const double maxPosition = static_cast<double>(session.optimizedLine.points.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
    const cv::Vec3d selectedPoint(volumePoint[0], volumePoint[1], volumePoint[2]);

    auto selected = session.controlPoints.end();
    double bestScore = std::numeric_limits<double>::infinity();
    for (auto it = session.controlPoints.begin(); it != session.controlPoints.end(); ++it) {
        if (!std::isfinite(it->linePosition)) {
            continue;
        }
        const cv::Vec3d delta = it->volumePoint - selectedPoint;
        const double pointDistanceSq = delta.dot(delta);
        const double lineDistance = std::abs(it->linePosition - linePosition);
        const double score = pointDistanceSq + lineDistance * 1.0e-6;
        if (score < bestScore) {
            bestScore = score;
            selected = it;
        }
    }
    if (selected == session.controlPoints.end()) {
        return;
    }

    const int deletedControlIndex =
        static_cast<int>(std::distance(session.controlPoints.begin(), selected));
    if (!confirmLinkedControlPointEdit(session,
                                       deletedControlIndex,
                                       tr("Deleting it"))) {
        return;
    }

    const bool deletedSeed = selected->isSeed;
    const int linePointCount = static_cast<int>(session.optimizedLine.points.size());
    const std::vector<vc::lasagna::LineControlPoint> branchRemapControls = session.controlPoints;
    const std::vector<FiberBranchRef> branchRemapBranches = session.branches;
    session.controlPoints.erase(selected);
    const BranchMetadataSyncResult branchSync =
        syncLinkedBranchMetadataAfterFiberModification(
            session,
            &branchRemapControls,
            &branchRemapBranches);
    scheduleBranchMetadataSaves(branchSync.affectedFiberIds, session.fiberId);
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    if (session.controlPoints.empty()) {
        return;
    }

    const bool hasSeed = std::any_of(session.controlPoints.begin(),
                                     session.controlPoints.end(),
                                     [](const vc::lasagna::LineControlPoint& control) {
                                         return control.isSeed;
                                     });
    if (deletedSeed || !hasSeed) {
        auto replacementSeed = session.controlPoints.begin();
        double replacementDistance = std::numeric_limits<double>::infinity();
        for (auto it = session.controlPoints.begin(); it != session.controlPoints.end(); ++it) {
            it->isSeed = false;
            const double distance = std::isfinite(it->linePosition)
                ? std::abs(it->linePosition - linePosition)
                : std::numeric_limits<double>::infinity();
            if (distance < replacementDistance) {
                replacementDistance = distance;
                replacementSeed = it;
            }
        }
        replacementSeed->isSeed = true;
        session.seedPoint = replacementSeed->volumePoint;
    }

    auto focus = session.controlPoints.begin();
    double focusDistance = std::numeric_limits<double>::infinity();
    for (auto it = session.controlPoints.begin(); it != session.controlPoints.end(); ++it) {
        const double distance = std::isfinite(it->linePosition)
            ? std::abs(it->linePosition - linePosition)
            : std::numeric_limits<double>::infinity();
        if (distance < focusDistance) {
            focusDistance = distance;
            focus = it;
        }
    }
    session.focusedLinePosition = std::isfinite(focus->linePosition)
        ? focus->linePosition
        : linePosition;
    session.focusedControlPoint = focus->volumePoint;

    writeLineDebugJson("control_delete",
                       session.controlPoints,
                       linePointsToJson(session.optimizedLine));
    const bool autoReoptimize =
        !pane->dialog ||
        pane->dialog->reoptimizationMode() ==
            LineAnnotationDialog::ReoptimizationMode::AutoReoptimize;
    if (autoReoptimize) {
        const auto activeRange = activeRangeAroundDeletedControl(session.controlPoints,
                                                                linePosition,
                                                                linePointCount,
                                                                3);
        startOptimization(session, false, activeRange.first, activeRange.second);
        return;
    }

    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches));
        pane->dialog->setGeneratedPredSnapPoints(
            generatedPredSnapMarkers(session.controlPoints, session.predSnapSet));
    }
}

bool LineAnnotationController::ensureDatasetForSession(LineAnnotationSession& session)
{
    if (!_state || !_state->vpkg()) {
        showError(tr("No volume package loaded."));
        return false;
    }

    auto vpkg = _state->vpkg();
    std::string selected;
    fs::path manifestPath;
    double workingToBaseScale = 1.0;
    try {
        if (const auto resolved = vc3d::opendata::resolveLasagnaForVolume(
                *vpkg, _state->currentVolumeId())) {
            manifestPath = resolved->manifestPath;
            selected = resolved->manifestBacked
                ? manifestPath.string()
                : vpkg->selectedLasagnaDataset();
            workingToBaseScale = resolved->workingToBaseScale;
        }
    } catch (const std::exception& ex) {
        showError(tr("Cannot resolve Lasagna for the active volume: %1")
                      .arg(QString::fromStdString(ex.what())));
        return false;
    }

    if (selected.empty()) {
        const fs::path startDir = vpkg->path().empty()
            ? fs::path{}
            : vpkg->path().parent_path();
        auto picked = _datasetPicker ? _datasetPicker(_parentWidget, startDir)
                                     : std::optional<std::string>{};
        if (!picked || picked->empty()) {
            return false;
        }
        selected = *picked;
        manifestPath = vc::project::resolveLocalPath(selected, vpkg->path().parent_path());
        try {
            auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                vc::lasagna::LasagnaDataset::open(
                    manifestPath, {workingToBaseScale}));
            auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
            session.dataset = std::move(dataset);
            session.normalSampler = std::move(sampler);
        } catch (const std::exception& ex) {
            showError(tr("Invalid Lasagna dataset: %1").arg(QString::fromStdString(ex.what())));
            return false;
        }
        vpkg->setSelectedLasagnaDataset(selected);
    } else {
        if (!session.normalSampler || session.selectedManifestPath != manifestPath ||
            session.workingToBaseScale != workingToBaseScale) {
            try {
                auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                    vc::lasagna::LasagnaDataset::open(
                        manifestPath, {workingToBaseScale}));
                auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
                session.dataset = std::move(dataset);
                session.normalSampler = std::move(sampler);
            } catch (const std::exception& ex) {
                showError(tr("Invalid selected Lasagna dataset: %1")
                              .arg(QString::fromStdString(ex.what())));
                return false;
            }
        }
    }

    session.selectedDatasetLocation = selected;
    session.selectedManifestPath = manifestPath;
    session.workingToBaseScale = workingToBaseScale;
    return true;
}

bool LineAnnotationController::needsFinalOptimization(const LineAnnotationSession& session) const
{
    return session.optimizationState != SessionOptimizationState::Optimized &&
           !session.optimizedLine.points.empty() &&
           !session.controlPoints.empty();
}

void LineAnnotationController::refreshSessionOptimizationStatus(
    const LineAnnotationSession& session)
{
    auto* pane = paneForSurface(session.surfaceName);
    if (pane && pane->dialog) {
        pane->dialog->setOptimizationStatus(
            session.optimizationState == SessionOptimizationState::Optimized);
    }
}

void LineAnnotationController::setSessionOptimizationState(
    LineAnnotationSession& session,
    SessionOptimizationState state)
{
    session.optimizationState = state;
    refreshSessionOptimizationStatus(session);
}

bool LineAnnotationController::applyOptimizationTaskResult(LineAnnotationSession& session,
                                                           OptimizationTaskResult task,
                                                           bool updateGeneratedViews,
                                                           SessionOptimizationState resultOptimizationState,
                                                           const std::string& eventOverride,
                                                           bool fireSuccessCallback)
{
    if (!task.ok) {
        session.taskState = LineAnnotationSession::TaskState::Failed;
        session.error = task.error;
        showError(tr("Lasagna line optimization failed: %1")
                      .arg(QString::fromStdString(task.error)));
        return false;
    }

    auto* pane = paneForSurface(session.surfaceName);
    session.taskState = LineAnnotationSession::TaskState::Succeeded;
    session.seedPoint = task.seedPoint;
    session.selectedManifestPath = task.manifestPath;
    session.optimizationReport = task.result.report;
    session.optimizedLine = std::move(task.result.line);
    const std::vector<vc::lasagna::LineControlPoint> branchRemapControls = session.controlPoints;
    const std::vector<FiberBranchRef> branchRemapBranches = session.branches;
    session.controlPoints = std::move(task.controlPoints);
    syncLinkedBranchMetadataAfterFiberModification(
        session,
        &branchRemapControls,
        &branchRemapBranches);
    for (auto& control : session.controlPoints) {
        double bestDistance = std::numeric_limits<double>::infinity();
        int bestIndex = -1;
        for (size_t i = 0; i < session.optimizedLine.points.size(); ++i) {
            const cv::Vec3d delta = session.optimizedLine.points[i].position - control.volumePoint;
            const double distance = std::sqrt(delta.dot(delta));
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = static_cast<int>(i);
            }
        }
        if (bestIndex >= 0) {
            const cv::Vec3d controlVolumePoint = control.volumePoint;
            control.optimizedIndex = bestIndex;
            control.linePosition = static_cast<double>(bestIndex);
            const bool matchesFocusedControl = session.focusedControlPoint.has_value() &&
                std::sqrt((controlVolumePoint - *session.focusedControlPoint).dot(
                    controlVolumePoint - *session.focusedControlPoint)) <= 1.0e-6;
            if (std::abs(session.focusedLinePosition - control.linePosition) <= 0.5 ||
                matchesFocusedControl) {
                session.focusedLinePosition = control.linePosition;
            }
        }
        if (control.isSeed) {
            session.seedPoint = control.volumePoint;
        }
    }

    const std::string eventName = eventOverride.empty() ? task.eventName : eventOverride;
    setSessionOptimizationState(session, resultOptimizationState);

    const std::string resultEvent = eventName.empty()
        ? "optimization_result"
        : eventName + "_result";
    writeLineDebugJson(resultEvent,
                       session.controlPoints,
                       linePointsToJson(session.optimizedLine),
                       &session.optimizationReport);
    if (updateGeneratedViews && !session.suppressGeneratedViews) {
        if (!materializeGeneratedViews(session)) {
            session.taskState = LineAnnotationSession::TaskState::Failed;
            return false;
        }
    }
    refreshBranchLineViews(session.fiberId);

    if (session.deferShowUntilGenerated && pane && pane->dialog && !pane->dialog->isVisible()) {
        emit lineAnnotationWorkspaceRequested(pane->dialog, tr("Line Annotation"));
        pane->dialog->showWithSavedGeometry();
        pane->dialog->raise();
        pane->dialog->activateWindow();
    }

    const double prefetchPrepMs = session.optimizationReport.normalChunkPrefetchMs +
                                  session.optimizationReport.normalMaterializeMs;
    Logger()->info("Line annotation Lasagna stage timing: event={} prefetch_prep_ms={:.3f} ceres_solve_ms={:.3f} overall_ms={:.3f} points={}",
                   resultEvent,
                   prefetchPrepMs,
                   session.optimizationReport.ceresSolveMs,
                   session.optimizationReport.totalMs,
                   session.optimizedLine.points.size());
    if (pane && !session.suppressFiberSave &&
        session.taskState == LineAnnotationSession::TaskState::Succeeded &&
        !session.optimizedLine.points.empty() &&
        !session.controlPoints.empty()) {
        saveSessionAsFiber(session);
    }
    auto callback = fireSuccessCallback ? session.optimizationSucceededCallback : nullptr;
    if (callback) {
        callback(session);
    }
    return true;
}

bool LineAnnotationController::finalizeSessionOptimizationSynchronously(
    LineAnnotationSession& session,
    bool fireSuccessCallback)
{
    if (!needsFinalOptimization(session)) {
        return true;
    }
    if (!ensureDatasetForSession(session)) {
        return false;
    }
    if (!session.normalSampler) {
        showError(tr("Could not run final line optimization: no Lasagna dataset is loaded."));
        return false;
    }

    std::vector<cv::Vec3d> initialLinePoints;
    initialLinePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        initialLinePoints.push_back(point.position);
    }
    const auto* pane = paneForSurface(session.surfaceName);
    const int initialCenterlineLengthVx = pane && pane->dialog
        ? pane->dialog->initialCenterlineLengthVx()
        : vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX_DEFAULT;
    OptimizationTaskResult task = optimizeLineWithSampler(session.selectedManifestPath,
                                                          session.controlPoints,
                                                          std::move(initialLinePoints),
                                                          session.sourceSliceNormal,
                                                          session.initialDirectionMode,
                                                          initialCenterlineLengthVx,
                                                          false,
                                                          -1,
                                                          -1,
                                                          *session.normalSampler);
    return applyOptimizationTaskResult(session,
                                       std::move(task),
                                       false,
                                       SessionOptimizationState::Optimized,
                                       "final_full_line_opt",
                                       fireSuccessCallback);
}

void LineAnnotationController::requestFinalizedClose(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->dialog) {
        cleanupSurfaceName(surfaceName);
        return;
    }
    if (!pane->session) {
        closeLineAnnotationDialogAfterFinalization(pane->dialog);
        return;
    }

    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (!needsFinalOptimization(session)) {
        closeLineAnnotationDialogAfterFinalization(pane->dialog);
        return;
    }
    if (!finalizeSessionOptimizationSynchronously(session, false)) {
        return;
    }
    saveSessionAsFiber(session);
    session.suppressFiberSave = true;
    closeLineAnnotationDialogAfterFinalization(pane->dialog);
}

void LineAnnotationController::startOptimization(LineAnnotationSession& session,
                                                 bool forceFullOptimization,
                                                 int activeStart,
                                                 int activeEnd)
{
    if (session.controlPoints.empty()) {
        return;
    }
    if (session.fiberId != 0 && session.fiberMetricsMatchStoredFiber) {
        session.fiberMetricsMatchStoredFiber = false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
    }
    session.taskState = LineAnnotationSession::TaskState::Running;
    session.error.clear();
    auto seedIt = std::find_if(session.controlPoints.begin(),
                               session.controlPoints.end(),
                               [](const vc::lasagna::LineControlPoint& control) {
                                   return control.isSeed;
                               });
    if (seedIt == session.controlPoints.end()) {
        session.controlPoints.front().isSeed = true;
        seedIt = session.controlPoints.begin();
    }
    session.seedPoint = seedIt->volumePoint;

    auto* watcher = new QFutureWatcher<OptimizationTaskResult>(this);
    session.watcher = watcher;
    const std::string surfaceName = session.surfaceName;
    if (auto* pane = paneForSurface(surfaceName); pane && pane->dialog) {
        pane->dialog->setOptimizationBusy(true);
    }
    const bool localOptimization = !forceFullOptimization &&
        activeStart >= 0 &&
        activeEnd >= activeStart;
    session.pendingOptimizationState = localOptimization
        ? SessionOptimizationState::Incremental
        : SessionOptimizationState::Optimized;
    connect(watcher,
            &QFutureWatcher<OptimizationTaskResult>::finished,
            this,
            [this, surfaceName, watcher]() {
                finishOptimization(surfaceName);
                watcher->deleteLater();
            });

    const auto manifestPath = session.selectedManifestPath;
    auto factory = _optimizationTaskFactory;
    auto controlPoints = session.controlPoints;
    std::vector<cv::Vec3d> initialLinePoints;
    initialLinePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        initialLinePoints.push_back(point.position);
    }
    const cv::Vec3d sourceSliceNormal = session.sourceSliceNormal;
    const InitialDirectionMode directionMode = session.initialDirectionMode;
    const auto* pane = paneForSurface(surfaceName);
    const int initialCenterlineLengthVx = pane && pane->dialog
        ? pane->dialog->initialCenterlineLengthVx()
        : vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX_DEFAULT;
    auto dataset = session.dataset;
    auto normalSampler = session.normalSampler;
    watcher->setFuture(QtConcurrent::run([factory,
                                           manifestPath,
                                           controlPoints,
                                           initialLinePoints,
                                           sourceSliceNormal,
                                           directionMode,
                                           initialCenterlineLengthVx,
                                           forceFullOptimization,
                                           activeStart,
                                           activeEnd,
                                           dataset,
                                           normalSampler]() mutable {
        if (factory) {
            return factory(manifestPath,
                           std::move(controlPoints),
                           std::move(initialLinePoints),
                           sourceSliceNormal,
                           directionMode,
                           initialCenterlineLengthVx,
                           forceFullOptimization,
                           activeStart,
                           activeEnd);
        }
        if (normalSampler) {
            (void)dataset;
            return optimizeLineWithSampler(manifestPath,
                                           std::move(controlPoints),
                                           std::move(initialLinePoints),
                                           sourceSliceNormal,
                                           directionMode,
                                           initialCenterlineLengthVx,
                                           forceFullOptimization,
                                           activeStart,
                                           activeEnd,
                                           *normalSampler);
        }
        return optimizeLineFromManifest(manifestPath,
                                        std::move(controlPoints),
                                        std::move(initialLinePoints),
                                        sourceSliceNormal,
                                        directionMode,
                                        initialCenterlineLengthVx,
                                        forceFullOptimization,
                                        activeStart,
                                        activeEnd);
    }));
}

void LineAnnotationController::finishOptimization(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session || !pane->session->watcher) {
        return;
    }

    auto& session = *pane->session;
    auto* watcher = session.watcher.data();
    if (!watcher) {
        return;
    }

    OptimizationTaskResult task = watcher->result();
    session.watcher = nullptr;
    const bool ok = applyOptimizationTaskResult(session,
                                               std::move(task),
                                               true,
                                               session.pendingOptimizationState,
                                               {},
                                               true);
    if (pane->dialog) {
        pane->dialog->setOptimizationBusy(false);
    }
    if (!ok) {
        setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    }
}

void LineAnnotationController::finishFiberAlignmentMetrics(
    QFutureWatcher<FiberMetricsTaskResult>* watcher)
{
    if (!watcher) {
        return;
    }
    _fiberMetricsWatchers.erase(
        std::remove_if(_fiberMetricsWatchers.begin(),
                       _fiberMetricsWatchers.end(),
                       [watcher](const QPointer<QFutureWatcher<FiberMetricsTaskResult>>& item) {
                           return item.data() == watcher || item.isNull();
                       }),
        _fiberMetricsWatchers.end());

    FiberMetricsTaskResult result = watcher->future().result();
    if (result.generation != _fiberMetricsGeneration) {
        return;
    }

    if (!result.ok) {
        showError(tr("Could not calculate fiber alignment metrics: %1")
                      .arg(QString::fromStdString(result.error)));
    }

    for (uint64_t fiberId : result.requestedFiberIds) {
        const auto tokenIt = result.requestTokens.find(fiberId);
        if (tokenIt == result.requestTokens.end() ||
            !isAlignmentPendingForFiber(fiberId, tokenIt->second)) {
            continue;
        }
        auto metricIt = result.metrics.find(fiberId);
        if (metricIt != result.metrics.end()) {
            publishFiberAlignmentMetrics(fiberId, std::move(metricIt->second));
        } else {
            _pendingFiberAlignmentMetrics.erase(fiberId);
            _pendingFiberAlignmentMetricTokens.erase(fiberId);
            publishUnavailableFiberAlignmentMetrics(fiberId);
        }
    }
    _fiberMetricsPending = !_pendingFiberAlignmentMetrics.empty();
}

std::vector<vc3d::line_annotation::GeneratedSpanAlignmentMetric>
LineAnnotationController::generatedSpanAlignmentMetricsForSession(
    const LineAnnotationSession& session) const
{
    std::vector<vc3d::line_annotation::GeneratedSpanAlignmentMetric> metrics;
    if (session.fiberId == 0 ||
        !session.fiberMetricsMatchStoredFiber ||
        session.controlPoints.size() < 2) {
        return metrics;
    }

    const auto controls = generatedControlMarkers(session.controlPoints);
    struct ControlRef {
        size_t index = 0;
        double linePosition = std::numeric_limits<double>::quiet_NaN();
    };
    std::vector<ControlRef> sortedControls;
    sortedControls.reserve(controls.size());
    for (size_t i = 0; i < controls.size(); ++i) {
        if (std::isfinite(controls[i].linePosition)) {
            sortedControls.push_back({i, controls[i].linePosition});
        }
    }
    std::sort(sortedControls.begin(),
              sortedControls.end(),
              [](const ControlRef& lhs, const ControlRef& rhs) {
                  if (lhs.linePosition == rhs.linePosition) {
                      return lhs.index < rhs.index;
                  }
                  return lhs.linePosition < rhs.linePosition;
              });
    if (sortedControls.size() < 2) {
        return metrics;
    }

    metrics.reserve(sortedControls.size() - 1);
    int spanIndex = 0;
    for (size_t i = 1; i < sortedControls.size(); ++i) {
        const auto& first = sortedControls[i - 1];
        const auto& second = sortedControls[i];
        if (second.linePosition <= first.linePosition) {
            continue;
        }
        auto metric = vc3d::line_annotation::makeGeneratedSpanAlignmentMetric(
            spanIndex,
            static_cast<int>(first.index),
            static_cast<int>(second.index),
            controls);
        const auto alignment = cachedAlignmentForSpan(session.fiberId, spanIndex);
        metric.available = alignment.available;
        metric.pending = alignment.pending;
        metric.maxErrorDegrees = alignment.maxErrorDegrees;
        metric.error = alignment.error;
        metrics.push_back(std::move(metric));
        ++spanIndex;
    }
    return metrics;
}

void LineAnnotationController::updateGeneratedViewMetricsForFiber(uint64_t fiberId)
{
    for (const auto& pane : _panes) {
        if (!pane.session || !pane.dialog || pane.session->fiberId != fiberId) {
            continue;
        }
        pane.dialog->setGeneratedSpanAlignmentMetrics(
            generatedSpanAlignmentMetricsForSession(*pane.session));
    }
}

bool LineAnnotationController::materializeGeneratedViews(LineAnnotationSession& session)
{
    if (!_state) {
        session.error = "No active application state.";
        showError(tr("Could not create line annotation views: no active application state."));
        return false;
    }

    vc::lasagna::LineViewSurfaces views;
    try {
        views = vc::lasagna::buildLineViewSurfaces(session.optimizedLine);
    } catch (const std::exception& ex) {
        session.error = ex.what();
        showError(tr("Could not create line annotation views: %1")
                      .arg(QString::fromStdString(session.error)));
        return false;
    }

    for (const auto& name : session.generatedSurfaceNames) {
        _state->setSurface(name, nullptr);
    }
    session.generatedSurfaceNames.clear();
    const std::string generatedPrefix = session.surfaceName.empty()
        ? nextSurfaceName()
        : session.surfaceName;
    session.generatedLineSurfaceName = generatedPrefix + "_line_surface";
    session.generatedLineSideSliceName = generatedPrefix + "_line_side_slice";

    _state->setSurface(session.generatedLineSurfaceName, views.lineSurface);
    _state->setSurface(session.generatedLineSideSliceName, views.lineSideSlice);
    session.generatedSurfaceNames.push_back(session.generatedLineSurfaceName);
    session.generatedSurfaceNames.push_back(session.generatedLineSideSliceName);

    std::vector<cv::Vec3f> linePoints;
    linePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        linePoints.push_back({static_cast<float>(point.position[0]),
                              static_cast<float>(point.position[1]),
                              static_cast<float>(point.position[2])});
    }

    const cv::Vec3f seedPoint{static_cast<float>(session.seedPoint[0]),
                              static_cast<float>(session.seedPoint[1]),
                              static_cast<float>(session.seedPoint[2])};
    const cv::Vec3f focusPoint = session.focusedControlPoint
        ? cv::Vec3f{static_cast<float>((*session.focusedControlPoint)[0]),
                    static_cast<float>((*session.focusedControlPoint)[1]),
                    static_cast<float>((*session.focusedControlPoint)[2])}
        : seedPoint;

    LineAnnotationDialog::GeneratedViews generatedViews;
    generatedViews.lineSurfaceName = session.generatedLineSurfaceName;
    generatedViews.lineSurfaceTitle = tr("Line Surface");
    generatedViews.lineSideSliceName = session.generatedLineSideSliceName;
    generatedViews.lineSideSliceTitle = tr("Line Side Slice");
    generatedViews.lineSideSlice = views.lineSideSlice;
    generatedViews.linePoints = std::move(linePoints);
    generatedViews.lineUpVectors = views.lineUpVectors;
    generatedViews.branchLinePoints = generatedBranchLinePointsForSession(session);
    generatedViews.branchLinks = generatedBranchLinkMarkers(session.branches);
    generatedViews.seedPoint = seedPoint;
    generatedViews.focusPoint = focusPoint;
    generatedViews.seedLineIndex = static_cast<int>(session.optimizedLine.points.size() / 2);
    generatedViews.initialCurrentCutFollowsStripMouse =
        !session.disableInitialGeneratedHoverFollow;
    generatedViews.controlPoints = controlMarkersForSession(session);
    for (const auto& marker : generatedViews.controlPoints) {
        if (marker.isSeed && std::isfinite(marker.linePosition)) {
            generatedViews.seedLineIndex = static_cast<int>(std::llround(marker.linePosition));
        }
    }
    const auto controlLinePositionRange =
        vc3d::line_annotation::generatedControlLinePositionRange(
            generatedViews.controlPoints);
    generatedViews.predSnapPoints =
        generatedPredSnapMarkers(session.controlPoints, session.predSnapSet);
    const double maxLinePosition =
        static_cast<double>(std::max<size_t>(1, session.optimizedLine.points.size()) - 1);
    double initialCenterPosition = std::clamp(session.focusedLinePosition,
                                             0.0,
                                             maxLinePosition);
    if (controlLinePositionRange) {
        initialCenterPosition = std::clamp(initialCenterPosition,
                                           controlLinePositionRange->first,
                                           controlLinePositionRange->second);
    }
    generatedViews.initialCenterIndex =
        static_cast<int>(std::llround(initialCenterPosition));
    generatedViews.initialStripLinePositionRange =
        session.initialStripLinePositionRange
            ? session.initialStripLinePositionRange
            : controlLinePositionRange;
    generatedViews.spanAlignmentMetrics =
        generatedSpanAlignmentMetricsForSession(session);

    generatedViews.currentCutName = generatedPrefix + "_line_current_cut";
    generatedViews.currentCutSurface = std::make_shared<PlaneSurface>(
        seedPoint,
        cv::Vec3f{1.0f, 0.0f, 0.0f});
    _state->setSurface(generatedViews.currentCutName, generatedViews.currentCutSurface);
    session.generatedSurfaceNames.push_back(generatedViews.currentCutName);

    generatedViews.sideCutName = generatedPrefix + "_line_side_cut";
    generatedViews.sideCutSurface = std::make_shared<PlaneSurface>(
        seedPoint,
        cv::Vec3f{1.0f, 0.0f, 0.0f});
    _state->setSurface(generatedViews.sideCutName, generatedViews.sideCutSurface);
    session.generatedSurfaceNames.push_back(generatedViews.sideCutName);

    auto* pane = paneForSurface(session.surfaceName);
    if (!pane || !pane->dialog) {
        return true;
    }

    CChunkedVolumeViewer::CameraState camera;
    camera.scale = 1.0f;
    if (!pane->dialog->panes().empty() && pane->dialog->panes().front().viewer) {
        camera = pane->dialog->panes().front().viewer->cameraState();
        camera.zOffset = 0.0f;
        camera.zOffsetWorldDir = {0, 0, 0};
    }

    if (!pane->dialog->setGeneratedLineViews(generatedViews, camera)) {
        for (const auto& name : session.generatedSurfaceNames) {
            _state->setSurface(name, nullptr);
        }
        session.generatedSurfaceNames.clear();
        session.error = "Failed to create generated annotation viewers.";
        showError(tr("Could not create generated line annotation viewers."));
        return false;
    }
    if (session.fiberId != 0 &&
        session.fiberMetricsMatchStoredFiber &&
        !hasCachedAlignmentForFiber(session.fiberId) &&
        !isAlignmentPendingForFiber(session.fiberId)) {
        requestFiberAlignmentMetrics(session.fiberId);
    }
    return true;
}

void LineAnnotationController::handleShowAsMesh(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (session.taskState != LineAnnotationSession::TaskState::Succeeded) {
        showError(tr("Run line optimization before exporting generated meshes."));
        return;
    }
    if (!finalizeSessionOptimizationSynchronously(session, false)) {
        return;
    }
    if (!session.suppressGeneratedViews && !materializeGeneratedViews(session)) {
        return;
    }

    try {
        const auto savedPaths = saveGeneratedQuadMeshes(session);
        if (savedPaths.empty()) {
            showError(tr("No generated line quad meshes are available to export."));
            return;
        }

        QStringList labels;
        labels.reserve(static_cast<int>(savedPaths.size()));
        for (const auto& path : savedPaths) {
            labels.push_back(QString::fromStdString(path.filename().string()));
        }
        QMessageBox::information(_parentWidget,
                                 tr("Line Annotation"),
                                 tr("Saved generated mesh surfaces in paths:\n%1")
                                     .arg(labels.join(QStringLiteral("\n"))));
    } catch (const std::exception& ex) {
        showError(tr("Could not save generated line meshes: %1")
                      .arg(QString::fromStdString(ex.what())));
    }
}

fs::path LineAnnotationController::resolveMeshExportPathsDir() const
{
    if (!_state || !_state->vpkg()) {
        throw std::runtime_error("No volume package loaded.");
    }

    auto vpkg = _state->vpkg();
    fs::path pathsDir = vpkg->outputSegmentsPath();
    const fs::path volpkgRoot = vpkg->path().empty()
        ? fs::path(vpkg->getVolpkgDirectory())
        : vpkg->path().parent_path();

    if (pathsDir.empty()) {
        if (volpkgRoot.empty()) {
            throw std::runtime_error("Volume package path is unavailable.");
        }
        pathsDir = volpkgRoot / "paths";
    }

    std::error_code ec;
    fs::create_directories(pathsDir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create paths directory " +
                                 pathsDir.string() + ": " + ec.message());
    }

    bool hasEntry = false;
    const fs::path canonicalPaths = fs::weakly_canonical(pathsDir, ec);
    for (const auto& entryPath : vpkg->availableSegmentPaths()) {
        std::error_code entryEc;
        if (fs::weakly_canonical(entryPath, entryEc) == canonicalPaths && !entryEc) {
            hasEntry = true;
            break;
        }
    }
    if (!hasEntry && !volpkgRoot.empty() && pathsDir == volpkgRoot / "paths") {
        vpkg->addSegmentsEntry("paths");
    }

    return pathsDir;
}

fs::path LineAnnotationController::nextMeshExportPath(const fs::path& pathsDir,
                                                      const std::string& stem) const
{
    const std::string timestamp =
        QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss")).toStdString();
    std::string base = "line_annotation_" + timestamp + "_" + stem;
    fs::path candidate = pathsDir / base;
    int suffix = 1;
    while (fs::exists(candidate)) {
        candidate = pathsDir / (base + "_" + std::to_string(suffix++));
    }
    return candidate;
}

std::vector<fs::path> LineAnnotationController::saveGeneratedQuadMeshes(LineAnnotationSession& session)
{
    if (!_state) {
        throw std::runtime_error("No active application state.");
    }

    const fs::path pathsDir = resolveMeshExportPathsDir();
    const std::vector<std::pair<std::string, std::string>> exports = {
        {session.generatedLineSurfaceName, "surface"},
        {session.generatedLineSideSliceName, "side_slice"},
    };

    std::vector<fs::path> savedPaths;
    for (const auto& [surfaceName, stem] : exports) {
        if (surfaceName.empty()) {
            continue;
        }
        auto surface = std::dynamic_pointer_cast<QuadSurface>(_state->surface(surfaceName));
        if (!surface) {
            continue;
        }

        auto clone = std::make_shared<QuadSurface>(surface->rawPoints().clone(), surface->scale());
        clone->meta = surface->meta;
        vc3d::opendata::copyCoordinateIdentityToSurface(
            *clone, coordinateIdentityForState(_state));

        const fs::path outputPath = nextMeshExportPath(pathsDir, stem);
        const std::string outputName = outputPath.filename().string();
        clone->save(outputPath.string(), outputName, false);

        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(outputName);
        } else if (_state->vpkg()) {
            (void)_state->vpkg()->addSingleSegmentation(outputName);
        }
        savedPaths.push_back(outputPath);
    }

    if (!savedPaths.empty() && _state->vpkg()) {
        _state->emitSurfacesChanged();
    }

    Logger()->info("Line annotation saved {} generated mesh surface(s) to {}",
                   savedPaths.size(), pathsDir.string());
    return savedPaths;
}

std::string LineAnnotationController::nextSurfaceName()
{
    return "line_annotation_slice_" + std::to_string(_nextPaneId++);
}

void LineAnnotationController::cleanupSurfaceName(const std::string& surfaceName)
{
    if (surfaceName.empty()) {
        return;
    }

    const auto before = _panes.size();
    std::vector<std::string> generatedSurfaceNames;
    std::shared_ptr<LineAnnotationSession> sessionToSave;
    for (const auto& pane : _panes) {
        if (pane.surfaceName == surfaceName && pane.session && pane.session->watcher) {
            auto* watcher = pane.session->watcher.data();
            disconnect(watcher, nullptr, this, nullptr);
            connect(watcher,
                    &QFutureWatcher<OptimizationTaskResult>::finished,
                    watcher,
                    &QObject::deleteLater);
            pane.session->watcher = nullptr;
        }
        if (pane.surfaceName == surfaceName && pane.session) {
            generatedSurfaceNames = pane.session->generatedSurfaceNames;
            if (!pane.session->suppressFiberSave &&
                pane.session->taskState == LineAnnotationSession::TaskState::Succeeded &&
                !pane.session->optimizedLine.points.empty() &&
                !pane.session->controlPoints.empty()) {
                sessionToSave = pane.session;
            }
        }
    }
    _panes.erase(std::remove_if(_panes.begin(),
                                _panes.end(),
                                [&surfaceName](const PaneRecord& pane) {
                                    return pane.surfaceName == surfaceName;
                                }),
                 _panes.end());
    if (before == _panes.size()) {
        return;
    }

    auto matchesClosingSurface = [&surfaceName, &generatedSurfaceNames](const std::string& name) {
        return name == surfaceName ||
               std::find(generatedSurfaceNames.begin(),
                         generatedSurfaceNames.end(),
                         name) != generatedSurfaceNames.end();
    };
    if (matchesClosingSurface(_runningSideStripIntersectionSurfaceName)) {
        const uint64_t cancelToken = ++_nextSideStripIntersectionToken;
        _latestSideStripIntersectionToken = cancelToken;
        if (_latestSideStripIntersectionTokenAtomic) {
            _latestSideStripIntersectionTokenAtomic->store(cancelToken,
                                                           std::memory_order_relaxed);
        }
    }
    if (_pendingSideStripIntersectionRequest &&
        matchesClosingSurface(_pendingSideStripIntersectionRequest->surfaceName)) {
        _pendingSideStripIntersectionRequest.reset();
    }
    if (matchesClosingSurface(_lastSideStripIntersectionSurfaceName)) {
        _lastSideStripIntersectionKey = 0;
        _lastSideStripIntersectionSurfaceName.clear();
        _lastSideStripIntersectionMarkers.clear();
    }

    if (sessionToSave) {
        saveSessionAsFiber(*sessionToSave);
    }

    if (_state) {
        _state->setSurface(surfaceName, nullptr);
        for (const auto& name : generatedSurfaceNames) {
            _state->setSurface(name, nullptr);
        }
    }
}

LineAnnotationController::PaneRecord*
LineAnnotationController::paneForSurface(const std::string& surfaceName)
{
    auto it = std::find_if(_panes.begin(), _panes.end(), [&surfaceName](const PaneRecord& pane) {
        if (pane.surfaceName == surfaceName) {
            return true;
        }
        return pane.session &&
               std::find(pane.session->generatedSurfaceNames.begin(),
                         pane.session->generatedSurfaceNames.end(),
                         surfaceName) != pane.session->generatedSurfaceNames.end();
    });
    return it == _panes.end() ? nullptr : &*it;
}

const LineAnnotationController::PaneRecord*
LineAnnotationController::paneForSurface(const std::string& surfaceName) const
{
    auto it = std::find_if(_panes.begin(), _panes.end(), [&surfaceName](const PaneRecord& pane) {
        if (pane.surfaceName == surfaceName) {
            return true;
        }
        return pane.session &&
               std::find(pane.session->generatedSurfaceNames.begin(),
                         pane.session->generatedSurfaceNames.end(),
                         surfaceName) != pane.session->generatedSurfaceNames.end();
    });
    return it == _panes.end() ? nullptr : &*it;
}

std::optional<std::string> LineAnnotationController::pickDataset(
    QWidget* parent,
    const fs::path& startDir) const
{
    const QString picked = QFileDialog::getOpenFileName(
        parent,
        tr("Select Lasagna Dataset"),
        QString::fromStdString(startDir.string()),
        tr("Lasagna datasets (*.lasagna.json);;JSON files (*.json);;All files (*)"));
    if (picked.isEmpty()) {
        return std::nullopt;
    }
    return picked.toStdString();
}

LineAnnotationController::OptimizationTaskResult LineAnnotationController::runOptimizationTask(
    fs::path manifestPath,
    std::vector<vc::lasagna::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    InitialDirectionMode directionMode,
    int initialCenterlineLengthVx,
    bool forceFullOptimization,
    int activeStart,
    int activeEnd) const
{
    if (_optimizationTaskFactory) {
        return _optimizationTaskFactory(std::move(manifestPath),
                                        std::move(controlPoints),
                                        std::move(initialLinePoints),
                                        sourceSliceNormal,
                                        directionMode,
                                        initialCenterlineLengthVx,
                                        forceFullOptimization,
                                        activeStart,
                                        activeEnd);
    }
    return optimizeLineFromManifest(std::move(manifestPath),
                                    std::move(controlPoints),
                                    std::move(initialLinePoints),
                                    sourceSliceNormal,
                                    directionMode,
                                    initialCenterlineLengthVx,
                                    forceFullOptimization,
                                    activeStart,
                                    activeEnd);
}

void LineAnnotationController::loadFibersForCurrentPackage()
{
    // Runtime fiber ids are reassigned per package load; a surviving candidate
    // could silently point at an unrelated fiber with the same id.
    _linkCandidate.reset();
    _fibers.clear();
    _fiberAlignmentMetrics.clear();
    _pendingFiberAlignmentMetrics.clear();
    _pendingFiberAlignmentMetricTokens.clear();
    ++_nextFiberAlignmentMetricToken;
    ++_fiberMetricsGeneration;
    _fiberMetricsPending = false;
    _knownFiberTags.clear();
    if (!_state || !_state->vpkg()) {
        emitFiberSummaries();
        return;
    }

    const fs::path dir = fibersDir();
    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        emitFiberSummaries();
        return;
    }

    std::vector<fs::path> fiberFiles;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }
        fiberFiles.push_back(entry.path());
    }
    std::sort(fiberFiles.begin(), fiberFiles.end());

    auto sortLoadedFibers = [](std::vector<StoredFiber>& fibers) {
        std::sort(fibers.begin(),
                  fibers.end(),
                  [](const StoredFiber& a, const StoredFiber& b) {
                      return a.fileName < b.fileName;
                  });
    };

    auto loadStrictFibers = [&]() {
        std::vector<StoredFiber> strictFibers;
        std::vector<std::string> strictErrors;
        for (const auto& path : fiberFiles) {
            try {
                if (auto fiber = loadFiberFile(path)) {
                    strictFibers.push_back(std::move(*fiber));
                }
            } catch (const std::exception& ex) {
                strictErrors.push_back(
                    fiberErrorName(path.filename().string()) + ": " + ex.what());
            }
        }
        sortLoadedFibers(strictFibers);
        (void)validateLoadedFiberLinks(strictFibers, strictErrors);
        return std::pair<std::vector<StoredFiber>, std::vector<std::string>>{
            std::move(strictFibers),
            std::move(strictErrors)};
    };

    std::vector<StoredFiber> loadedFibers;
    std::vector<std::string> fatalLoadErrors;
    std::vector<std::string> branchLoadErrors;
    std::unordered_set<std::string> fibersWithRemovedBranchEntries;
    for (const auto& path : fiberFiles) {
        try {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("Failed to open fiber file");
            }
            const nlohmann::json root = nlohmann::json::parse(in);
            std::vector<std::string> branchErrors;
            if (auto fiber = loadFiberJson(root, path, &branchErrors)) {
                if (!branchErrors.empty()) {
                    fibersWithRemovedBranchEntries.insert(fiber->fileName);
                    branchLoadErrors.insert(branchLoadErrors.end(),
                                            branchErrors.begin(),
                                            branchErrors.end());
                }
                loadedFibers.push_back(std::move(*fiber));
            }
        } catch (const std::exception& ex) {
            fatalLoadErrors.push_back(
                fiberErrorName(path.filename().string()) + ": " + ex.what());
        }
    }
    sortLoadedFibers(loadedFibers);
    {
        std::unordered_map<std::string, uint64_t> fiberIdByFileName;
        fiberIdByFileName.reserve(loadedFibers.size());
        uint64_t runtimeId = 1;
        for (auto& fiber : loadedFibers) {
            fiber.id = runtimeId++;
            if (!fiber.fileName.empty()) {
                fiberIdByFileName[fiber.fileName] = fiber.id;
            }
        }
        for (auto& fiber : loadedFibers) {
            for (auto& branch : fiber.branches) {
                if (auto it = fiberIdByFileName.find(branch.branchFileName);
                    it != fiberIdByFileName.end()) {
                    branch.branchFiberId = it->second;
                }
            }
        }
    }

    const auto branchLinkIssues = collectLoadedFiberBranchIssues(loadedFibers);
    std::vector<std::string> branchErrors = branchLoadErrors;
    branchErrors.reserve(branchErrors.size() + branchLinkIssues.size());
    for (const auto& issue : branchLinkIssues) {
        if (issue.fiberIndex >= loadedFibers.size()) {
            continue;
        }
        branchErrors.push_back(fiberErrorName(loadedFibers[issue.fiberIndex].fileName) +
                               ": " + issue.reason);
    }

    if (!branchErrors.empty()) {
        QString details;
        const size_t shown = std::min<size_t>(branchErrors.size(), 8);
        for (size_t i = 0; i < shown; ++i) {
            if (!details.isEmpty()) {
                details += QStringLiteral("\n");
            }
            details += QString::fromStdString(branchErrors[i]);
        }
        if (branchErrors.size() > shown) {
            details += tr("\n... %1 more branch/link error(s)")
                           .arg(static_cast<int>(branchErrors.size() - shown));
        }

        QMessageBox prompt(_parentWidget.data());
        prompt.setIcon(QMessageBox::Warning);
        prompt.setWindowTitle(tr("Broken branch links"));
        prompt.setText(tr("Some saved fiber branch links are obsolete or inconsistent."));
        prompt.setInformativeText(details);
        auto* repairButton = prompt.addButton(
            tr("Remove broken branch links and reload"),
            QMessageBox::AcceptRole);
        prompt.addButton(tr("Keep files unchanged"), QMessageBox::RejectRole);
        prompt.setDefaultButton(qobject_cast<QPushButton*>(repairButton));
        prompt.exec();

        if (prompt.clickedButton() == repairButton) {
            std::vector<std::string> repairErrors;
            if (repairLoadedFiberBranchLinks(loadedFibers,
                                             fibersWithRemovedBranchEntries,
                                             branchLinkIssues,
                                             repairErrors)) {
                loadFibersForCurrentPackage();
                return;
            }
            for (const auto& error : repairErrors) {
                Logger()->warn("{}", error);
            }
            showError(tr("Could not repair broken branch links:\n%1")
                          .arg(QString::fromStdString(
                              repairErrors.empty() ? std::string{"unknown error"}
                                                   : repairErrors.front())));
        }

        auto strict = loadStrictFibers();
        loadedFibers = std::move(strict.first);
        fatalLoadErrors = std::move(strict.second);
    }

    std::vector<std::string> loadErrors = std::move(fatalLoadErrors);
    if (branchErrors.empty()) {
        (void)validateLoadedFiberLinks(loadedFibers, loadErrors);
    }

    for (auto& fiber : loadedFibers) {
        addKnownFiberTags(fiber.tags);
        if (fiber.needsSave) {
            try {
                fiber.needsSave = false;
                saveFiberNow(fiber);
            } catch (const std::exception& ex) {
                fiber.needsSave = true;
                Logger()->warn("Could not update VC3D fiber metadata {}: {}",
                               fiberPath(fiber).string(),
                               ex.what());
            }
        }
    }

    _fibers = std::move(loadedFibers);
    if (!loadErrors.empty()) {
        for (const auto& error : loadErrors) {
            Logger()->warn("{}", error);
        }
        QString message;
        const size_t shown = std::min<size_t>(loadErrors.size(), 8);
        for (size_t i = 0; i < shown; ++i) {
            if (!message.isEmpty()) {
                message += QStringLiteral("\n");
            }
            message += QString::fromStdString(loadErrors[i]);
        }
        if (loadErrors.size() > shown) {
            message += tr("\n... %1 more error(s)")
                           .arg(static_cast<int>(loadErrors.size() - shown));
        }
        showError(message);
    }
    emitFiberSummaries();
}

void LineAnnotationController::emitFiberSummaries()
{
    emit fibersChanged(fiberSummaries());
}

void LineAnnotationController::addKnownFiberTags(const std::vector<std::string>& tags)
{
    for (const auto& tag : tags) {
        addUniqueSorted(_knownFiberTags, tag);
    }
}

fs::path LineAnnotationController::fibersRootDir() const
{
    const fs::path root = currentVolpkgRoot();
    return root.empty() ? fs::path{} : root / "fibers";
}

fs::path LineAnnotationController::fibersDir() const
{
    if (!_state || !_state->vpkg()) {
        return {};
    }
    const fs::path root = currentVolpkgRoot();
    if (root.empty()) {
        return {};
    }
    const auto vpkg = _state->vpkg();
    return root / "fibers" / sanitizedProjectFiberDirName(vpkg->path(), root);
}

fs::path LineAnnotationController::relativeFiberPath(const StoredFiber& fiber) const
{
    const fs::path absolutePath = fiberPath(fiber);
    const fs::path root = currentVolpkgRoot();
    if (!absolutePath.empty() && !root.empty()) {
        std::error_code ec;
        const fs::path relativePath = fs::relative(absolutePath, root, ec);
        if (!ec && !relativePath.empty()) {
            return relativePath;
        }
    }

    const fs::path dir = fibersDir();
    const fs::path rootDir = fibersRootDir();
    if (!dir.empty() && !rootDir.empty()) {
        std::error_code ec;
        const fs::path relativeDir = fs::relative(dir, rootDir.parent_path(), ec);
        if (!ec && !relativeDir.empty()) {
            if (!fiber.fileName.empty()) {
                return relativeDir / fiber.fileName;
            }
            if (!fiber.username.empty() && !fiber.startedAt.empty() && fiber.sequence > 0) {
                return relativeDir / vc3d::line_annotation::fiberFileName(
                    fiber.username, fiber.startedAt, fiber.sequence);
            }
        }
    }

    if (!fiber.fileName.empty()) {
        return fs::path("fibers") / fiber.fileName;
    }
    return fs::path("fibers") / (std::to_string(fiber.id) + ".json");
}

fs::path LineAnnotationController::fiberFilePath(uint64_t fiberId) const
{
    const fs::path path = fiberPath(fiberId);
    return fs::exists(path) ? path : fs::path{};
}

fs::path LineAnnotationController::fiberPath(uint64_t fiberId) const
{
    const auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it != _fibers.end()) {
        return fiberPath(*it);
    }
    return fibersDir() / (std::to_string(fiberId) + ".json");
}

fs::path LineAnnotationController::fiberPath(const StoredFiber& fiber) const
{
    if (!fiber.fileName.empty()) {
        return fibersDir() / fiber.fileName;
    }
    if (!fiber.username.empty() && !fiber.startedAt.empty() && fiber.sequence > 0) {
        return fibersDir() / vc3d::line_annotation::fiberFileName(
            fiber.username, fiber.startedAt, fiber.sequence);
    }
    return fibersDir() / (std::to_string(fiber.id) + ".json");
}

fs::path LineAnnotationController::currentVolpkgRoot() const
{
    if (!_state || !_state->vpkg()) {
        return {};
    }
    const auto vpkg = _state->vpkg();
    const fs::path projectPath = vpkg->path();
    if (!projectPath.empty()) {
        return projectPath.parent_path();
    }
    return fs::path(vpkg->getVolpkgDirectory());
}

std::vector<std::string> LineAnnotationController::atlasPathKeysForFiber(
    const StoredFiber& fiber) const
{
    std::vector<std::string> keys;
    auto addKey = [&keys](const fs::path& path) {
        if (path.empty()) {
            return;
        }
        const std::string key = vc::atlas::atlasFiberPathKey(path);
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
            keys.push_back(key);
        }
    };

    if (!fiber.fileName.empty()) {
        addKey(relativeFiberPath(fiber));
        addKey(fs::path("fibers") / fiber.fileName);
        addKey(fs::path(fiber.fileName));
    }

    const fs::path absoluteFiberPath = fiberPath(fiber);
    addKey(absoluteFiberPath);

    const fs::path volpkgRoot = currentVolpkgRoot();
    if (!volpkgRoot.empty()) {
        std::error_code ec;
        const fs::path relativePath = fs::relative(absoluteFiberPath, volpkgRoot, ec);
        if (!ec && !relativePath.empty()) {
            addKey(relativePath);
        }
    }
    return keys;
}

std::optional<fs::path> LineAnnotationController::resolveAtlasFiberPath(
    const StoredFiber& fiber,
    const fs::path& atlasDir) const
{
    if (atlasDir.empty()) {
        return std::nullopt;
    }

    const fs::path volpkgRoot = currentVolpkgRoot();
    vc::atlas::Atlas atlas = volpkgRoot.empty()
        ? vc::atlas::Atlas::load(atlasDir)
        : vc::atlas::Atlas::load(atlasDir, volpkgRoot);
    const auto keys = atlasPathKeysForFiber(fiber);
    for (const auto& mapping : atlas.fibers) {
        const std::string mappingKey = vc::atlas::atlasFiberPathKey(mapping.fiberPath);
        if (std::find(keys.begin(), keys.end(), mappingKey) != keys.end()) {
            return mapping.fiberPath;
        }
    }
    return std::nullopt;
}

void LineAnnotationController::attachAtlasPredSnaps(
    const StoredFiber& fiber,
    LineAnnotationSession& session,
    const fs::path& atlasDir)
{
    session.atlasDir = atlasDir;
    session.atlasFiberPath.clear();
    session.predSnapSet = {};

    std::optional<fs::path> atlasFiberPath;
    try {
        atlasFiberPath = resolveAtlasFiberPath(fiber, atlasDir);
    } catch (const std::exception& ex) {
        Logger()->warn("Could not resolve atlas fiber mapping for {}: {}",
                       fiberPath(fiber).string(),
                       ex.what());
        return;
    }

    if (!atlasFiberPath) {
        Logger()->warn("Could not find atlas mapping for fiber {}", fiberPath(fiber).string());
        return;
    }

    session.atlasFiberPath = *atlasFiberPath;
    try {
        session.predSnapSet = vc::atlas::loadAtlasPredSnapSet(
            vc::atlas::atlasPredSnapAttachmentPath(atlasDir, *atlasFiberPath));
    } catch (const std::exception& ex) {
        Logger()->warn("Could not load pred-snap attachment for {}: {}",
                       atlasFiberPath->string(),
                       ex.what());
    }
}

uint64_t LineAnnotationController::nextFiberId() const
{
    uint64_t id = 1;
    for (const auto& fiber : _fibers) {
        id = std::max(id, fiber.id + 1);
    }
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId != 0) {
            id = std::max(id, pane.session->fiberId + 1);
        }
    }
    return id;
}

uint64_t LineAnnotationController::nextFiberSequenceForUsername(const std::string& username) const
{
    const std::string normalized = vc3d::line_annotation::normalizedFiberUsername(username);
    uint64_t sequence = 1;
    for (const auto& fiber : _fibers) {
        if (vc3d::line_annotation::normalizedFiberUsername(fiber.username) == normalized) {
            sequence = std::max(sequence, fiber.sequence + 1);
        }
    }
    for (const auto& pane : _panes) {
        if (!pane.session || pane.session->fiberSequence == 0) {
            continue;
        }
        if (vc3d::line_annotation::normalizedFiberUsername(pane.session->fiberUsername) == normalized) {
            sequence = std::max(sequence, pane.session->fiberSequence + 1);
        }
    }
    return sequence;
}

std::string LineAnnotationController::currentFiberUsername() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return vc3d::line_annotation::normalizedFiberUsername(
        settings.value(vc3d::settings::viewer::USERNAME,
                       vc3d::settings::viewer::USERNAME_DEFAULT).toString().toStdString());
}

std::string LineAnnotationController::currentFiberDateTimeString()
{
    return QDateTime::currentDateTimeUtc()
        .toString(QStringLiteral("yyyyMMddTHHmmsszzz"))
        .toStdString();
}

void LineAnnotationController::ensureSessionFiberIdentity(LineAnnotationSession& session)
{
    if (!session.fiberFileName.empty()) {
        if (session.fiberUsername.empty()) {
            session.fiberUsername = "anon";
        }
        return;
    }

    session.fiberUsername = currentFiberUsername();
    session.fiberStartedAt = currentFiberDateTimeString();
    session.fiberSequence = nextFiberSequenceForUsername(session.fiberUsername);
    session.fiberFileName = vc3d::line_annotation::fiberFileName(
        session.fiberUsername, session.fiberStartedAt, session.fiberSequence);
}

std::vector<std::vector<cv::Vec3f>>
LineAnnotationController::generatedBranchLinePointsForSession(
    const LineAnnotationSession& session) const
{
    std::vector<std::vector<cv::Vec3f>> branches;
    if (!session.showLinkedLineOverlays) {
        return branches;
    }
    std::unordered_set<uint64_t> seenFiberIds;

    auto linePointsForFiber = [this](uint64_t fiberId) -> std::vector<cv::Vec3d> {
        for (const auto& pane : _panes) {
            if (!pane.session || pane.session->fiberId != fiberId ||
                pane.session->optimizedLine.points.empty()) {
                continue;
            }
            std::vector<cv::Vec3d> points;
            points.reserve(pane.session->optimizedLine.points.size());
            for (const auto& point : pane.session->optimizedLine.points) {
                points.push_back(point.position);
            }
            return points;
        }
        auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
            return fiber.id == fiberId;
        });
        return it == _fibers.end() ? std::vector<cv::Vec3d>{} : it->linePoints;
    };

    for (const auto& branch : session.branches) {
        if (branch.branchFiberId == 0 || !seenFiberIds.insert(branch.branchFiberId).second) {
            continue;
        }
        std::vector<cv::Vec3d> linePoints = linePointsForFiber(branch.branchFiberId);
        if (linePoints.size() >= 2) {
            branches.push_back(generatedLinePoints(linePoints));
        }
    }
    return branches;
}

void LineAnnotationController::refreshBranchLineViews(uint64_t changedFiberId)
{
    for (const auto& pane : _panes) {
        if (!pane.session || !pane.dialog) {
            continue;
        }
        bool relevant = changedFiberId == 0 || pane.session->fiberId == changedFiberId;
        if (!relevant) {
            relevant = std::any_of(pane.session->branches.begin(),
                                   pane.session->branches.end(),
                                   [changedFiberId](const FiberBranchRef& branch) {
                                       return branch.branchFiberId == changedFiberId;
                                   });
        }
        if (!relevant) {
            continue;
        }
        pane.dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(*pane.session),
            generatedBranchLinePointsForSession(*pane.session),
            generatedBranchLinkMarkers(pane.session->branches));
    }
}

std::vector<vc::atlas::FiberPolyline>
LineAnnotationController::fiberSnapshotsForSideStripQuery() const
{
    std::map<uint64_t, vc::atlas::FiberPolyline> byId;

    auto makePolyline = [](uint64_t id,
                           uint64_t generation,
                           const std::vector<cv::Vec3d>& linePoints,
                           const std::vector<cv::Vec3d>& controlPoints) {
        vc::atlas::FiberPolyline polyline;
        polyline.id = id;
        polyline.generation = std::max<uint64_t>(uint64_t{1}, generation);
        polyline.controlPoints = controlPoints;
        polyline.points.reserve(linePoints.size());
        for (const auto& point : linePoints) {
            polyline.points.push_back(vc::atlas::FiberPoint{point, std::nullopt});
        }
        return polyline;
    };

    for (const auto& fiber : _fibers) {
        if (fiber.id == 0 || fiber.linePoints.size() < 2) {
            continue;
        }
        byId[fiber.id] = makePolyline(fiber.id,
                                      fiber.generation,
                                      fiber.linePoints,
                                      fiber.controlPoints);
    }

    for (const auto& pane : _panes) {
        const auto& session = pane.session;
        if (!session || session->fiberId == 0 || session->optimizedLine.points.size() < 2) {
            continue;
        }
        uint64_t generation = 1;
        if (auto it = std::find_if(_fibers.begin(),
                                   _fibers.end(),
                                   [id = session->fiberId](const StoredFiber& fiber) {
                                       return fiber.id == id;
                                   });
            it != _fibers.end()) {
            generation = std::max<uint64_t>(uint64_t{1}, it->generation + 1);
        }

        std::vector<cv::Vec3d> linePoints;
        linePoints.reserve(session->optimizedLine.points.size());
        for (const auto& point : session->optimizedLine.points) {
            linePoints.push_back(point.position);
        }
        std::vector<cv::Vec3d> controlPoints;
        controlPoints.reserve(session->controlPoints.size());
        for (const auto& control : session->controlPoints) {
            controlPoints.push_back(control.volumePoint);
        }
        byId[session->fiberId] = makePolyline(session->fiberId,
                                             generation,
                                             linePoints,
                                             controlPoints);
    }

    std::vector<vc::atlas::FiberPolyline> snapshots;
    snapshots.reserve(byId.size());
    for (auto& [id, polyline] : byId) {
        (void)id;
        snapshots.push_back(std::move(polyline));
    }
    return snapshots;
}

void LineAnnotationController::handleGeneratedSideStripIntersectionQuery(
    const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session || !pane->dialog) {
        return;
    }

    auto stripSurface = _state
        ? std::dynamic_pointer_cast<QuadSurface>(_state->surface(surfaceName))
        : nullptr;
    const auto* stripPointsPtr = stripSurface ? stripSurface->rawPointsPtr() : nullptr;
    if (!stripPointsPtr || stripPointsPtr->rows < 2 || stripPointsPtr->cols < 2) {
        pane->dialog->setGeneratedSideStripIntersectionResult(0);
        pane->dialog->setGeneratedFiberIntersectionMarkers({});
        return;
    }

    SideStripIntersectionRequest request;
    request.surfaceName = surfaceName;
    request.sourceFiberId = pane->session->fiberId;
    request.stripPoints = stripPointsPtr->clone();
    if (request.sourceFiberId != 0) {
        request.excludedFiberIds.push_back(request.sourceFiberId);
    }
    std::sort(request.excludedFiberIds.begin(), request.excludedFiberIds.end());
    request.excludedFiberIds.erase(std::unique(request.excludedFiberIds.begin(),
                                               request.excludedFiberIds.end()),
                                   request.excludedFiberIds.end());
    request.fibers = fiberSnapshotsForSideStripQuery();
    request.branchLinks.reserve(pane->session->branches.size());
    for (const auto& branch : pane->session->branches) {
        if (branch.branchFiberId == 0 ||
            !finitePoint(branch.controlPointPosition) ||
            !finitePoint(branch.branchControlPointPosition) ||
            !finiteDirection(branch.branchControlPointDirection)) {
            continue;
        }
        vc::atlas::FiberSideStripLineQuery query;
        query.fiberId = branch.branchFiberId;
        query.point = branch.branchControlPointPosition;
        query.direction = normalizedOrZero(branch.branchControlPointDirection);
        query.connectorStart = branch.controlPointPosition;
        request.branchLinks.push_back(query);
    }
    request.cacheKey = sideStripRequestCacheKey(request.stripPoints,
                                               request.fibers,
                                               request.excludedFiberIds,
                                               request.branchLinks);

    if (request.fibers.empty() && request.branchLinks.empty()) {
        pane->dialog->setGeneratedSideStripIntersectionResult(0);
        pane->dialog->setGeneratedFiberIntersectionMarkers({});
        return;
    }

    if (_sideStripIntersectionRunning &&
        _runningSideStripIntersectionToken == _latestSideStripIntersectionToken &&
        request.cacheKey == _runningSideStripIntersectionKey &&
        request.surfaceName == _runningSideStripIntersectionSurfaceName) {
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("already running"), 0, 0);
        return;
    }

    if (_sideStripIntersectionRunning &&
        _pendingSideStripIntersectionRequest &&
        _pendingSideStripIntersectionRequest->cacheKey == request.cacheKey &&
        _pendingSideStripIntersectionRequest->surfaceName == request.surfaceName) {
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("already queued"), 0, 0);
        return;
    }

    if (!_sideStripIntersectionRunning &&
        _lastSideStripIntersectionKey != 0 &&
        request.cacheKey == _lastSideStripIntersectionKey &&
        request.surfaceName == _lastSideStripIntersectionSurfaceName) {
        pane->dialog->setGeneratedFiberIntersectionMarkers(
            markLinkCandidateFiberIntersections(_lastSideStripIntersectionMarkers,
                                                pane->session->branches));
        pane->dialog->setGeneratedSideStripIntersectionResult(
            _lastSideStripIntersectionMarkers.size());
        return;
    }

    request.token = ++_nextSideStripIntersectionToken;
    _latestSideStripIntersectionToken = request.token;
    if (_latestSideStripIntersectionTokenAtomic) {
        _latestSideStripIntersectionTokenAtomic->store(request.token, std::memory_order_relaxed);
    }

    pane->dialog->setGeneratedSideStripIntersectionBusy(true);
    pane->dialog->setGeneratedSideStripIntersectionProgress(tr("queued"), 0, 0);
    pane->dialog->setGeneratedFiberIntersectionMarkers({});

    if (!request.branchLinks.empty()) {
        vc::atlas::FiberSpatialIndex previewIndex;
        vc::atlas::FiberSideStripQueryOptions previewOptions;
        previewOptions.stripPoints = request.stripPoints;
        previewOptions.deduplicateStripDistance = 1.0e-3;
        previewOptions.aabbPadding = 1.0e-6;
        previewOptions.maxResults = 0;
        const unsigned previewStdThreads = std::thread::hardware_concurrency();
        const int previewQtThreads = QThread::idealThreadCount();
        const size_t previewHardwareThreads =
            std::max(static_cast<size_t>(previewStdThreads),
                     previewQtThreads > 0 ? static_cast<size_t>(previewQtThreads)
                                          : size_t{0});
        previewOptions.workerThreads =
            std::max<size_t>(
                1,
                previewHardwareThreads > 1 ? (previewHardwareThreads + 1) / 2 : 1);
        previewOptions.branchLinks = request.branchLinks;
        auto previewMarkers = sideStripMarkersFromIntersections(
            previewIndex.sideStripIntersections(previewOptions));
        pane->dialog->setGeneratedFiberIntersectionMarkers(
            markLinkCandidateFiberIntersections(std::move(previewMarkers),
                                                pane->session->branches));
    }

    if (_sideStripIntersectionRunning) {
        _pendingSideStripIntersectionRequest = std::move(request);
        return;
    }
    startSideStripIntersectionQuery(std::move(request));
}

LineAnnotationController::SideStripIntersectionTaskResult
LineAnnotationController::runSideStripIntersectionQuery(
    const SideStripIntersectionRequest& request,
    SideStripProgressCallback progressCallback,
    SideStripPartialResultCallback partialResultCallback,
    SideStripCancelCallback cancelCallback)
{
    SideStripIntersectionTaskResult result;
    result.token = request.token;
    result.cacheKey = request.cacheKey;
    result.surfaceName = request.surfaceName;
    try {
        const size_t stripTriangleBudget =
            request.stripPoints.rows > 1 && request.stripPoints.cols > 1
                ? static_cast<size_t>(request.stripPoints.rows - 1) *
                      static_cast<size_t>(request.stripPoints.cols - 1) * 2
                : size_t{0};
        std::unordered_set<uint64_t> excludedFiberIds(request.excludedFiberIds.begin(),
                                                      request.excludedFiberIds.end());
        size_t segmentBudget = 0;
        for (const auto& fiber : request.fibers) {
            if (excludedFiberIds.find(fiber.id) != excludedFiberIds.end() ||
                fiber.points.size() < 2) {
                continue;
            }
            segmentBudget += fiber.points.size() - 1;
        }
        const size_t branchBudget = request.branchLinks.size() * stripTriangleBudget;
        const size_t triangleIndexBudget = segmentBudget > 0 ? stripTriangleBudget : 0;
        const size_t finishBudget = 1;
        const size_t totalProgressBudget = std::max<size_t>(
            1,
            branchBudget +
                triangleIndexBudget +
                segmentBudget +
                finishBudget);
        const size_t branchOffset = 0;
        const size_t triangleIndexOffset = branchOffset + branchBudget;
        const size_t segmentOffset = triangleIndexOffset + triangleIndexBudget;
        const size_t finishOffset = segmentOffset + segmentBudget;
        const unsigned stdHardwareThreads = std::thread::hardware_concurrency();
        const int qtHardwareThreads = QThread::idealThreadCount();
        const size_t hardwareThreads =
            std::max(static_cast<size_t>(stdHardwareThreads),
                     qtHardwareThreads > 0 ? static_cast<size_t>(qtHardwareThreads)
                                           : size_t{0});
        const size_t workerThreadBudget =
            std::max<size_t>(
                1,
                hardwareThreads > 1 ? (hardwareThreads + 1) / 2 : 1);
        size_t lastAggregateProgress = 0;

        auto publishProgress = [&progressCallback](const std::string& stage,
                                                   size_t completed,
                                                   size_t total) {
            if (progressCallback) {
                progressCallback(stage, completed, total);
            }
        };
        auto scaledProgress = [](size_t completed,
                                 size_t total,
                                 size_t budget) -> size_t {
            if (budget == 0) {
                return 0;
            }
            if (total == 0) {
                return completed > 0 ? budget : 0;
            }
            completed = std::min(completed, total);
            return static_cast<size_t>(
                (static_cast<unsigned long long>(completed) * budget) / total);
        };
        auto publishAggregateProgress =
            [&publishProgress,
             &lastAggregateProgress,
             totalProgressBudget](const std::string& stage,
                                  size_t aggregateCompleted) {
                aggregateCompleted =
                    std::min(aggregateCompleted, totalProgressBudget);
                aggregateCompleted =
                    std::max(aggregateCompleted, lastAggregateProgress);
                lastAggregateProgress = aggregateCompleted;
                publishProgress(stage, aggregateCompleted, totalProgressBudget);
            };
        auto convertIntersections =
            [](const std::vector<vc::atlas::FiberSideStripIntersection>& intersections) {
                std::vector<SideStripMarker> markers;
                markers.reserve(intersections.size());
                for (const auto& intersection : intersections) {
                    SideStripMarker marker;
                    marker.point = toVec3f(intersection.point);
                    marker.fiberId = intersection.fiberId;
                    marker.segmentIndex = intersection.segmentIndex;
                    marker.arclength = intersection.arclength;
                    marker.distance = intersection.distance;
                    marker.projectedBranchLink =
                        intersection.source == vc::atlas::FiberSideStripIntersectionSource::BranchLink;
                    if (marker.projectedBranchLink && finitePoint(intersection.connectorStart)) {
                        marker.connectorStart = toVec3f(intersection.connectorStart);
                    }
                    markers.push_back(marker);
                }
                return markers;
            };
        auto branchCoreProgress =
            [&publishAggregateProgress,
             &scaledProgress,
             branchOffset,
             branchBudget](vc::atlas::FiberSideStripProgressPhase phase,
                           size_t completed,
                           size_t total) {
                if (phase != vc::atlas::FiberSideStripProgressPhase::BranchLinks) {
                    return;
                }
                publishAggregateProgress(
                    sideStripProgressPhaseName(phase),
                    branchOffset + scaledProgress(completed, total, branchBudget));
            };
        auto fiberCoreProgress =
            [&publishAggregateProgress,
             &scaledProgress,
             triangleIndexOffset,
             triangleIndexBudget,
             segmentOffset,
             segmentBudget](vc::atlas::FiberSideStripProgressPhase phase,
                            size_t completed,
                            size_t total) {
                switch (phase) {
                case vc::atlas::FiberSideStripProgressPhase::BuildTriangleIndex:
                    publishAggregateProgress(
                        sideStripProgressPhaseName(phase),
                        triangleIndexOffset +
                            scaledProgress(completed, total, triangleIndexBudget));
                    break;
                case vc::atlas::FiberSideStripProgressPhase::FiberSegments:
                    publishAggregateProgress(
                        sideStripProgressPhaseName(phase),
                        segmentOffset +
                            scaledProgress(completed, total, segmentBudget));
                    break;
                default:
                    break;
                }
            };

        if (!request.branchLinks.empty()) {
            if (cancelCallback && cancelCallback()) {
                result.error = "cancelled";
                return result;
            }
            publishAggregateProgress("branch links", branchOffset);
            vc::atlas::FiberSpatialIndex branchIndex;
            vc::atlas::FiberSideStripQueryOptions branchOptions;
            branchOptions.stripPoints = request.stripPoints;
            branchOptions.deduplicateStripDistance = 1.0e-3;
            branchOptions.aabbPadding = 1.0e-6;
            branchOptions.maxResults = 0;
            branchOptions.workerThreads = workerThreadBudget;
            branchOptions.branchLinks = request.branchLinks;

            const auto branchIntersections =
                branchIndex.sideStripIntersections(branchOptions,
                                                   branchCoreProgress,
                                                   cancelCallback);
            if (cancelCallback && cancelCallback()) {
                result.error = "cancelled";
                return result;
            }
            auto branchMarkers = convertIntersections(branchIntersections);
            result.markers.insert(result.markers.end(),
                                  branchMarkers.begin(),
                                  branchMarkers.end());
            if (partialResultCallback && !branchMarkers.empty()) {
                partialResultCallback(std::move(branchMarkers));
            }
            publishAggregateProgress("branch links", triangleIndexOffset);
        }

        if (segmentBudget > 0) {
            vc::atlas::FiberSideStripQueryOptions options;
            options.stripPoints = request.stripPoints;
            options.deduplicateStripDistance = 1.0e-3;
            options.aabbPadding = 1.0e-6;
            options.maxResults = 0;
            options.excludedFiberIds = request.excludedFiberIds;
            options.workerThreads = workerThreadBudget;
            options.queryFibers.reserve(request.fibers.size());
            for (const auto& fiber : request.fibers) {
                options.queryFibers.push_back(&fiber);
            }
            publishAggregateProgress("fiber geometry", triangleIndexOffset);

            const auto intersections =
                vc::atlas::FiberSpatialIndex{}.sideStripIntersections(options,
                                                                       fiberCoreProgress,
                                                                       cancelCallback);
            if (cancelCallback && cancelCallback()) {
                result.error = "cancelled";
                return result;
            }
            auto fiberMarkers = convertIntersections(intersections);
            result.markers.insert(result.markers.end(),
                                  fiberMarkers.begin(),
                                  fiberMarkers.end());
        }
        publishAggregateProgress("finished", finishOffset + finishBudget);
        result.ok = true;
    } catch (const std::exception& ex) {
        result.error = ex.what();
    } catch (...) {
        result.error = "unknown error";
    }
    return result;
}

void LineAnnotationController::startSideStripIntersectionQuery(
    SideStripIntersectionRequest request)
{
    _sideStripIntersectionRunning = true;
    _runningSideStripIntersectionToken = request.token;
    _runningSideStripIntersectionKey = request.cacheKey;
    _runningSideStripIntersectionSurfaceName = request.surfaceName;
    if (auto* pane = paneForSurface(request.surfaceName); pane && pane->dialog) {
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("starting"), 0, 0);
    }
    auto requestPtr = std::make_shared<SideStripIntersectionRequest>(std::move(request));
    const auto latestToken = _latestSideStripIntersectionTokenAtomic;
    QPointer<LineAnnotationController> self(this);
    QThreadPool::globalInstance()->start(
        [self, requestPtr, latestToken]() {
            std::string lastStage;
            size_t lastCompleted = 0;
            auto shouldPublish = [&lastStage, &lastCompleted](const std::string& stage,
                                                              size_t completed,
                                                              size_t total) {
                if (stage != lastStage || completed == 0 || completed == total) {
                    lastStage = stage;
                    lastCompleted = completed;
                    return true;
                }
                const size_t stride = total > 1000 ? 128 : 16;
                if (completed >= lastCompleted + stride) {
                    lastCompleted = completed;
                    return true;
                }
                return false;
            };
            auto progressCallback =
                [self, requestPtr, shouldPublish](const std::string& stage,
                                                  size_t completed,
                                                  size_t total) mutable {
                    if (!shouldPublish(stage, completed, total)) {
                        return;
                    }
                    if (!self) {
                        return;
                    }
                    const std::string surfaceName = requestPtr->surfaceName;
                    const uint64_t token = requestPtr->token;
                    QMetaObject::invokeMethod(
                        self.data(),
                        [self, token, surfaceName, stage, completed, total]() {
                            if (self) {
                                self->updateSideStripIntersectionProgress(token,
                                                                          surfaceName,
                                                                          stage,
                                                                          completed,
                                                                          total);
                            }
                        },
                        Qt::QueuedConnection);
                };
            auto partialCallback =
                [self, requestPtr](std::vector<SideStripMarker> markers) mutable {
                    if (!self) {
                        return;
                    }
                    const std::string surfaceName = requestPtr->surfaceName;
                    const uint64_t token = requestPtr->token;
                    QMetaObject::invokeMethod(
                        self.data(),
                        [self, token, surfaceName, markers = std::move(markers)]() mutable {
                            if (self) {
                                self->applyPartialSideStripIntersectionMarkers(
                                    token,
                                    surfaceName,
                                    std::move(markers));
                            }
                        },
                        Qt::QueuedConnection);
                };
            auto cancelCallback = [latestToken, token = requestPtr->token]() {
                return latestToken &&
                       latestToken->load(std::memory_order_relaxed) != token;
            };
            auto resultPtr = std::make_shared<SideStripIntersectionTaskResult>(
                LineAnnotationController::runSideStripIntersectionQuery(*requestPtr,
                                                                        progressCallback,
                                                                        partialCallback,
                                                                        cancelCallback));
            if (!self) {
                return;
            }
            QMetaObject::invokeMethod(
                self.data(),
                [self, resultPtr]() mutable {
                    if (self) {
                        self->finishSideStripIntersectionQuery(std::move(*resultPtr));
                    }
                },
                Qt::QueuedConnection);
        },
        -1);
}

void LineAnnotationController::updateSideStripIntersectionProgress(
    uint64_t token,
    const std::string& surfaceName,
    const std::string& stage,
    size_t completed,
    size_t total)
{
    if (token != _latestSideStripIntersectionToken) {
        return;
    }
    if (auto* pane = paneForSurface(surfaceName); pane && pane->dialog) {
        pane->dialog->setGeneratedSideStripIntersectionProgress(
            QString::fromStdString(stage),
            completed,
            total);
    }
}

void LineAnnotationController::applyPartialSideStripIntersectionMarkers(
    uint64_t token,
    const std::string& surfaceName,
    std::vector<SideStripMarker> markers)
{
    if (token != _latestSideStripIntersectionToken) {
        return;
    }
    if (auto* pane = paneForSurface(surfaceName); pane && pane->dialog) {
        pane->dialog->setGeneratedFiberIntersectionMarkers(
            markLinkCandidateFiberIntersections(
                std::move(markers),
                pane->session ? pane->session->branches
                              : std::vector<FiberBranchRef>{}));
    }
}

void LineAnnotationController::finishSideStripIntersectionQuery(
    SideStripIntersectionTaskResult result)
{
    _sideStripIntersectionRunning = false;
    _runningSideStripIntersectionToken = 0;
    _runningSideStripIntersectionKey = 0;
    _runningSideStripIntersectionSurfaceName.clear();
    const bool hasPendingRequest = _pendingSideStripIntersectionRequest.has_value();
    if (result.token == _latestSideStripIntersectionToken) {
        if (auto* pane = paneForSurface(result.surfaceName); pane && pane->dialog) {
            pane->dialog->setGeneratedSideStripIntersectionBusy(hasPendingRequest);
            if (result.ok) {
                const size_t markerCount = result.markers.size();
                _lastSideStripIntersectionKey = result.cacheKey;
                _lastSideStripIntersectionSurfaceName = result.surfaceName;
                _lastSideStripIntersectionMarkers = result.markers;
                pane->dialog->setGeneratedFiberIntersectionMarkers(
                    markLinkCandidateFiberIntersections(
                        std::move(result.markers),
                        pane->session ? pane->session->branches
                                      : std::vector<FiberBranchRef>{}));
                if (!hasPendingRequest) {
                    pane->dialog->setGeneratedSideStripIntersectionResult(markerCount);
                }
            } else {
                pane->dialog->setGeneratedFiberIntersectionMarkers({});
                if (!hasPendingRequest) {
                    pane->dialog->setGeneratedSideStripIntersectionError();
                }
                showError(tr("Could not query strip fiber intersections: %1")
                              .arg(QString::fromStdString(result.error)));
            }
        }
    } else if (!hasPendingRequest) {
        if (auto* pane = paneForSurface(result.surfaceName); pane && pane->dialog) {
            pane->dialog->setGeneratedSideStripIntersectionBusy(false);
        }
    }

    if (_pendingSideStripIntersectionRequest) {
        SideStripIntersectionRequest pending =
            std::move(*_pendingSideStripIntersectionRequest);
        _pendingSideStripIntersectionRequest.reset();
        startSideStripIntersectionQuery(std::move(pending));
    }
}

LineAnnotationController::BranchMetadataSyncResult
LineAnnotationController::syncLinkedBranchMetadataAfterFiberModification(
    LineAnnotationSession& session,
    const std::vector<vc::lasagna::LineControlPoint>* previousControlPoints,
    const std::vector<FiberBranchRef>* previousBranches)
{
    // This is the only synchronization entry point for live session branch
    // metadata. Keep remapping, reciprocal cleanup, and endpoint refreshes
    // together so future mutation paths cannot fix one side while missing
    // another linked fiber.
    BranchMetadataSyncResult result;
    if (session.branches.empty() && (!previousBranches || previousBranches->empty())) {
        return result;
    }
    if (session.fiberId == 0) {
        ensureSessionFiberIdentity(session);
    }

    if (previousControlPoints) {
        remapBranchControlPointIndices(
            *previousControlPoints,
            session.controlPoints,
            session.branches);
    }

    if (previousBranches && session.fiberId != 0) {
        auto removeReciprocal = [&session](std::vector<FiberBranchRef>& targetBranches,
                                           const FiberBranchRef& removedBranch) {
            const auto before = targetBranches.size();
            targetBranches.erase(
                std::remove_if(targetBranches.begin(),
                               targetBranches.end(),
                               [&session, &removedBranch](const FiberBranchRef& candidate) {
                                   const bool pointsToSession =
                                       candidate.branchFiberId == session.fiberId ||
                                       (!session.fiberFileName.empty() &&
                                        candidate.branchFileName == session.fiberFileName);
                                   const bool sameLocalEndpoint =
                                       candidate.branchControlPointIndex ==
                                           removedBranch.controlPointIndex ||
                                       pointsApproximatelyEqual(
                                           candidate.branchControlPointPosition,
                                           removedBranch.controlPointPosition);
                                   return pointsToSession &&
                                          candidate.controlPointIndex ==
                                              removedBranch.branchControlPointIndex &&
                                          sameLocalEndpoint;
                               }),
                targetBranches.end());
            return targetBranches.size() != before;
        };

        for (const auto& oldBranch : *previousBranches) {
            if (oldBranch.controlPointIndex < 0 ||
                oldBranch.branchControlPointIndex < 0) {
                continue;
            }
            const bool stillPresent = std::any_of(
                session.branches.begin(),
                session.branches.end(),
                [&oldBranch](const FiberBranchRef& branch) {
                    return branchLinkedEndpointMatches(oldBranch, branch);
                });
            if (stillPresent) {
                continue;
            }
            for (const auto& pane : _panes) {
                if (!pane.session ||
                    !branchReferencesFiber(oldBranch,
                                           pane.session->fiberId,
                                           pane.session->fiberFileName)) {
                    continue;
                }
                if (removeReciprocal(pane.session->branches, oldBranch)) {
                    addUniqueFiberId(result.affectedFiberIds, pane.session->fiberId);
                }
            }
            for (auto& fiber : _fibers) {
                if (!branchReferencesFiber(oldBranch, fiber.id, fiber.fileName)) {
                    continue;
                }
                if (removeReciprocal(fiber.branches, oldBranch)) {
                    addUniqueFiberId(result.affectedFiberIds, fiber.id);
                }
            }
        }
    }

    syncReciprocalBranchControlPointReferences(session);
    for (const uint64_t fiberId : syncBranchEndpointPositions(session)) {
        addUniqueFiberId(result.affectedFiberIds, fiberId);
    }
    std::sort(result.affectedFiberIds.begin(), result.affectedFiberIds.end());
    return result;
}

void LineAnnotationController::scheduleBranchMetadataSaves(
    const std::vector<uint64_t>& fiberIds,
    uint64_t excludedFiberId)
{
    std::vector<uint64_t> uniqueFiberIds;
    uniqueFiberIds.reserve(fiberIds.size());
    for (const uint64_t fiberId : fiberIds) {
        if (fiberId != excludedFiberId) {
            addUniqueFiberId(uniqueFiberIds, fiberId);
        }
    }
    if (uniqueFiberIds.empty()) {
        return;
    }
    std::sort(uniqueFiberIds.begin(), uniqueFiberIds.end());

    std::vector<FiberSaveSnapshot> snapshots;
    snapshots.reserve(uniqueFiberIds.size());
    for (const uint64_t fiberId : uniqueFiberIds) {
        bool addedOpenPaneSnapshot = false;
        for (const auto& pane : _panes) {
            if (!pane.session ||
                pane.session->fiberId != fiberId ||
                pane.session->suppressFiberSave ||
                pane.session->taskState != LineAnnotationSession::TaskState::Succeeded ||
                pane.session->optimizedLine.points.empty() ||
                pane.session->controlPoints.empty()) {
                continue;
            }
            StoredFiber linkedFiber = storedFiberFromSession(*pane.session);
            auto linkedIt = std::find_if(
                _fibers.begin(),
                _fibers.end(),
                [fiberId, &linkedFiber](const StoredFiber& candidate) {
                    return candidate.id == fiberId ||
                           (!linkedFiber.fileName.empty() &&
                            candidate.fileName == linkedFiber.fileName);
                });
            if (linkedIt == _fibers.end()) {
                _fibers.push_back(std::move(linkedFiber));
                linkedIt = std::prev(_fibers.end());
            } else {
                *linkedIt = std::move(linkedFiber);
            }
            snapshots.push_back(makeFiberSaveSnapshot(*linkedIt));
            addedOpenPaneSnapshot = true;
            break;
        }
        if (addedOpenPaneSnapshot) {
            continue;
        }
        auto linkedIt = std::find_if(
            _fibers.begin(),
            _fibers.end(),
            [fiberId](const StoredFiber& candidate) {
                return candidate.id == fiberId;
            });
        if (linkedIt != _fibers.end()) {
            snapshots.push_back(makeFiberSaveSnapshot(*linkedIt));
        }
    }
    if (!snapshots.empty()) {
        scheduleFiberSaveSnapshots(std::move(snapshots));
    }
}

void LineAnnotationController::syncBranchFiberFileRename(
    uint64_t fiberId,
    const std::string& oldFileName,
    const std::string& newFileName)
{
    if (fiberId == 0 || oldFileName == newFileName) {
        return;
    }
    std::vector<uint64_t> affectedFiberIds;
    auto updateBranches = [&](std::vector<FiberBranchRef>& branches, uint64_t ownerFiberId) {
        bool changed = false;
        for (auto& branch : branches) {
            if (branchReferencesFiber(branch, fiberId, oldFileName)) {
                branch.branchFiberId = fiberId;
                branch.branchFileName = newFileName;
                changed = true;
            }
        }
        if (changed) {
            addUniqueFiberId(affectedFiberIds, ownerFiberId);
        }
    };

    for (const auto& pane : _panes) {
        if (pane.session) {
            updateBranches(pane.session->branches, pane.session->fiberId);
        }
    }
    for (auto& fiber : _fibers) {
        updateBranches(fiber.branches, fiber.id);
    }

    scheduleBranchMetadataSaves(affectedFiberIds, fiberId);
    for (const uint64_t affectedFiberId : affectedFiberIds) {
        refreshBranchLineViews(affectedFiberId);
    }
}

void LineAnnotationController::removeBranchLinksToFiber(uint64_t fiberId,
                                                        const std::string& fileName)
{
    if (fiberId == 0 && fileName.empty()) {
        return;
    }
    std::vector<uint64_t> affectedFiberIds;
    auto removeBranches = [&](std::vector<FiberBranchRef>& branches, uint64_t ownerFiberId) {
        const auto before = branches.size();
        branches.erase(
            std::remove_if(branches.begin(),
                           branches.end(),
                           [fiberId, &fileName](const FiberBranchRef& branch) {
                               return branchReferencesFiber(branch, fiberId, fileName);
                           }),
            branches.end());
        if (branches.size() != before) {
            addUniqueFiberId(affectedFiberIds, ownerFiberId);
        }
    };

    for (const auto& pane : _panes) {
        if (pane.session) {
            removeBranches(pane.session->branches, pane.session->fiberId);
        }
    }
    for (auto& fiber : _fibers) {
        removeBranches(fiber.branches, fiber.id);
    }

    scheduleBranchMetadataSaves(affectedFiberIds, fiberId);
    for (const uint64_t affectedFiberId : affectedFiberIds) {
        refreshBranchLineViews(affectedFiberId);
    }
}

void LineAnnotationController::syncReciprocalBranchControlPointReferences(
    const LineAnnotationSession& session)
{
    if (session.fiberId == 0) {
        return;
    }

    auto updateBranches = [&session](std::vector<FiberBranchRef>& targetBranches) {
        for (auto& targetBranch : targetBranches) {
            if (targetBranch.branchFiberId != session.fiberId &&
                (session.fiberFileName.empty() ||
                 targetBranch.branchFileName != session.fiberFileName)) {
                continue;
            }
            for (const auto& sourceBranch : session.branches) {
                if (sourceBranch.branchFiberId != 0 &&
                    targetBranch.controlPointIndex == sourceBranch.branchControlPointIndex) {
                    targetBranch.branchFiberId = session.fiberId;
                    targetBranch.branchFileName = session.fiberFileName;
                    if (targetBranch.branchControlPointIndex !=
                        sourceBranch.controlPointIndex) {
                        targetBranch.branchControlPointIndex = sourceBranch.controlPointIndex;
                    }
                }
            }
        }
    };

    for (const auto& pane : _panes) {
        if (pane.session && pane.session.get() != &session) {
            updateBranches(pane.session->branches);
        }
    }
    for (auto& fiber : _fibers) {
        updateBranches(fiber.branches);
    }
}

bool LineAnnotationController::controlPointHasBranch(const LineAnnotationSession& session,
                                                     int controlPointIndex) const
{
    return std::any_of(session.branches.begin(),
                       session.branches.end(),
                       [controlPointIndex](const FiberBranchRef& branch) {
                           return branch.controlPointIndex == controlPointIndex;
                       });
}

bool LineAnnotationController::confirmLinkedControlPointEdit(
    const LineAnnotationSession& session,
    int controlPointIndex,
    const QString& action) const
{
    if (!controlPointHasBranch(session, controlPointIndex)) {
        return true;
    }
    const auto response = QMessageBox::question(
        _parentWidget.data(),
        tr("Linked control point"),
        tr("This control point is linked to another fiber. %1 will update linked "
           "branch metadata on both fibers. Continue?")
            .arg(action),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    return response == QMessageBox::Yes;
}

std::vector<uint64_t> LineAnnotationController::syncBranchEndpointPositions(
    LineAnnotationSession& session)
{
    std::vector<uint64_t> affectedFiberIds;
    if (session.fiberId != 0) {
        affectedFiberIds.push_back(session.fiberId);
    }

    auto addAffected = [&affectedFiberIds](uint64_t fiberId) {
        if (fiberId == 0 ||
            std::find(affectedFiberIds.begin(), affectedFiberIds.end(), fiberId) !=
                affectedFiberIds.end()) {
            return;
        }
        affectedFiberIds.push_back(fiberId);
    };

    auto updateReciprocal = [&session](
                                std::vector<FiberBranchRef>& targetBranches,
                                const FiberBranchRef& sourceBranch,
                                int targetControlPointIndex,
                                const cv::Vec3d& targetPoint,
                                const cv::Vec3d& sourcePoint) {
        for (auto& reciprocal : targetBranches) {
            const bool pointsToSession =
                reciprocal.branchFiberId == session.fiberId ||
                (!session.fiberFileName.empty() &&
                 reciprocal.branchFileName == session.fiberFileName);
            const bool sameLocalEndpoint =
                reciprocal.controlPointIndex == targetControlPointIndex ||
                pointsApproximatelyEqual(reciprocal.controlPointPosition, targetPoint);
            const bool sameLinkedEndpoint =
                reciprocal.branchControlPointIndex == sourceBranch.controlPointIndex ||
                pointsApproximatelyEqual(reciprocal.branchControlPointPosition, sourcePoint);
            if (!pointsToSession || !sameLocalEndpoint || !sameLinkedEndpoint) {
                continue;
            }
            reciprocal.controlPointIndex = targetControlPointIndex;
            reciprocal.branchFiberId = session.fiberId;
            reciprocal.branchFileName = session.fiberFileName;
            reciprocal.branchControlPointIndex = sourceBranch.controlPointIndex;
            reciprocal.controlPointPosition = targetPoint;
            reciprocal.branchControlPointPosition = sourcePoint;
            reciprocal.controlPointDirection = sourceBranch.branchControlPointDirection;
            reciprocal.branchControlPointDirection = sourceBranch.controlPointDirection;
        }
    };

    const std::vector<cv::Vec3d> sessionLinePoints =
        linePointPositions(session.optimizedLine);
    for (auto& branch : session.branches) {
        if (branch.controlPointIndex < 0 ||
            static_cast<size_t>(branch.controlPointIndex) >= session.controlPoints.size()) {
            continue;
        }
        const cv::Vec3d sourcePoint =
            session.controlPoints[static_cast<size_t>(branch.controlPointIndex)].volumePoint;
        branch.controlPointPosition = sourcePoint;
        branch.controlPointDirection =
            endpointTangentFromLinePoints(sessionLinePoints,
                                          sourcePoint,
                                          branch.controlPointDirection);
        if (finiteDirection(branch.branchControlPointDirection)) {
            branch.branchControlPointDirection =
                normalizedOrZero(branch.branchControlPointDirection);
        }

        bool resolvedFromOpenPane = false;
        for (const auto& pane : _panes) {
            if (!pane.session || pane.session.get() == &session) {
                continue;
            }
            if (pane.session->fiberId != branch.branchFiberId &&
                (branch.branchFileName.empty() ||
                pane.session->fiberFileName != branch.branchFileName)) {
                continue;
            }
            if (auto targetIndex = matchingSessionControlPointIndex(
                    pane.session->controlPoints,
                    branch.branchControlPointIndex,
                    branch.branchControlPointPosition)) {
                const cv::Vec3d targetPoint =
                    pane.session
                        ->controlPoints[static_cast<size_t>(*targetIndex)]
                        .volumePoint;
                const std::vector<cv::Vec3d> targetLinePoints =
                    linePointPositions(pane.session->optimizedLine);
                branch.branchControlPointIndex = *targetIndex;
                branch.branchControlPointPosition = targetPoint;
                branch.branchControlPointDirection =
                    endpointTangentFromLinePoints(targetLinePoints,
                                                  targetPoint,
                                                  branch.branchControlPointDirection);
                updateReciprocal(pane.session->branches,
                                  branch,
                                  *targetIndex,
                                  targetPoint,
                                  sourcePoint);
                addAffected(pane.session->fiberId);
                resolvedFromOpenPane = true;
            }
        }
        if (resolvedFromOpenPane) {
            continue;
        }
        for (auto& fiber : _fibers) {
            if (fiber.id != branch.branchFiberId &&
                (branch.branchFileName.empty() || fiber.fileName != branch.branchFileName)) {
                continue;
            }
            if (auto targetIndex = matchingStoredControlPointIndex(
                    fiber.controlPoints,
                    branch.branchControlPointIndex,
                    branch.branchControlPointPosition)) {
                const cv::Vec3d targetPoint =
                    fiber.controlPoints[static_cast<size_t>(*targetIndex)];
                branch.branchControlPointIndex = *targetIndex;
                branch.branchControlPointPosition = targetPoint;
                branch.branchControlPointDirection =
                    endpointTangentFromLinePoints(fiber.linePoints,
                                                  targetPoint,
                                                  branch.branchControlPointDirection);
                updateReciprocal(fiber.branches,
                                  branch,
                                  *targetIndex,
                                  targetPoint,
                                  sourcePoint);
                addAffected(fiber.id);
            }
        }
    }
    return affectedFiberIds;
}

double LineAnnotationController::lineLengthVx(const std::vector<cv::Vec3d>& points)
{
    return vc3d::line_annotation::fiberLineLengthVx(points);
}

void LineAnnotationController::scaleStoredFiber(StoredFiber& fiber, double scale)
{
    if (!std::isfinite(scale) || approximatelyEqual(scale, 1.0)) {
        return;
    }
    for (auto& point : fiber.controlPoints) {
        point = point * scale;
    }
    for (auto& point : fiber.linePoints) {
        point = point * scale;
    }
    for (auto& branch : fiber.branches) {
        branch.controlPointPosition = branch.controlPointPosition * scale;
        branch.branchControlPointPosition = branch.branchControlPointPosition * scale;
    }
    fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(fiber.controlPoints);
}

LineAnnotationController::CachedFiberAlignmentMetrics
LineAnnotationController::calculateAlignmentMetricsForFiber(
    const StoredFiber& fiber,
    const std::vector<ControlSpanRecord>& spans,
    const vc::lasagna::NormalSampler& sampler)
{
    struct Accumulator {
        double sum = 0.0;
        double max = 0.0;
        int count = 0;

        void add(double value)
        {
            if (!std::isfinite(value)) {
                return;
            }
            sum += value;
            max = count == 0 ? value : std::max(max, value);
            ++count;
        }

        void add(const Accumulator& other)
        {
            if (other.count <= 0) {
                return;
            }
            sum += other.sum;
            max = count == 0 ? other.max : std::max(max, other.max);
            count += other.count;
        }

        FiberSummary::AlignmentMetrics toMetrics(const std::string& error = {}) const
        {
            FiberSummary::AlignmentMetrics metric;
            metric.sampleCount = count;
            if (count > 0) {
                metric.available = true;
                metric.meanErrorDegrees = sum / static_cast<double>(count);
                metric.maxErrorDegrees = max;
            } else {
                metric.error = error.empty()
                    ? "No valid Lasagna normal samples were available."
                    : error;
            }
            return metric;
        }
    };

    CachedFiberAlignmentMetrics result;
    result.spans.resize(spans.size());
    if (fiber.linePoints.size() < 2) {
        result.fiber.error = "Fiber has fewer than two line points.";
        for (auto& span : result.spans) {
            span.error = result.fiber.error;
        }
        return result;
    }

    std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
    const vc::lasagna::NormalBatchReport batchReport =
        sampler.sampleNormalBatch(fiber.linePoints, false, samples);
    (void)batchReport;
    if (samples.size() != fiber.linePoints.size()) {
        result.fiber.error = "Lasagna normal sampler returned the wrong number of samples.";
        for (auto& span : result.spans) {
            span.error = result.fiber.error;
        }
        return result;
    }

    auto accumulateRange = [&](size_t firstLineIndex, size_t lastLineIndex) {
        Accumulator accumulator;
        if (firstLineIndex >= fiber.linePoints.size() ||
            lastLineIndex >= fiber.linePoints.size() ||
            lastLineIndex <= firstLineIndex) {
            return accumulator;
        }
        for (size_t segment = firstLineIndex; segment < lastLineIndex; ++segment) {
            const cv::Vec3d& a = fiber.linePoints[segment];
            const cv::Vec3d& b = fiber.linePoints[segment + 1];
            if (!finitePoint(a) || !finitePoint(b)) {
                continue;
            }
            const cv::Vec3d tangent = b - a;
            if (!finiteDirection(tangent)) {
                continue;
            }
            for (size_t sampleIndex : {segment, segment + 1}) {
                const auto& sample = samples[sampleIndex].sample;
                if (!sample.valid) {
                    continue;
                }
                accumulator.add(normalAlignmentErrorDegrees(tangent, sample.normal));
            }
        }
        return accumulator;
    };

    Accumulator fiberAccumulator;
    for (size_t i = 0; i < spans.size(); ++i) {
        const auto& span = spans[i];
        const Accumulator spanAccumulator =
            accumulateRange(span.firstLineIndex, span.lastLineIndex);
        result.spans[i] = spanAccumulator.toMetrics();
        fiberAccumulator.add(spanAccumulator);
    }
    result.fiber = spans.empty()
        ? accumulateRange(0, fiber.linePoints.size() - 1).toMetrics()
        : fiberAccumulator.toMetrics();
    return result;
}

vc::lasagna::LineModel LineAnnotationController::lineModelFromPoints(
    const std::vector<cv::Vec3d>& points,
    const vc::lasagna::NormalSampler* normalSampler)
{
    if (points.empty()) {
        throw std::runtime_error("Fiber has no line points");
    }
    if (!normalSampler) {
        throw std::runtime_error("No Lasagna normal sampler is available for this fiber");
    }

    std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
    const vc::lasagna::NormalBatchReport batchReport =
        normalSampler->sampleNormalBatch(points, false, samples);
    (void)batchReport;
    if (samples.size() != points.size()) {
        throw std::runtime_error("Normal sampler returned the wrong number of samples");
    }

    vc::lasagna::LineModel model;
    model.points.reserve(points.size());
    int bestAnchor = -1;
    double bestAnchorDistance = std::numeric_limits<double>::infinity();
    const double center = static_cast<double>(points.size() - 1) * 0.5;
    for (size_t i = 0; i < points.size(); ++i) {
        vc::lasagna::LinePoint linePoint;
        linePoint.position = points[i];
        linePoint.sampledNormal = samples[i].sample;
        linePoint.sampledNormal.normal = normalizedOrZero(linePoint.sampledNormal.normal);
        linePoint.sampledNormal.valid =
            linePoint.sampledNormal.valid &&
            finiteDirection(linePoint.sampledNormal.normal);
        linePoint.valid = linePoint.sampledNormal.valid;
        if (linePoint.valid) {
            const double distance = std::abs(static_cast<double>(i) - center);
            if (distance < bestAnchorDistance) {
                bestAnchorDistance = distance;
                bestAnchor = static_cast<int>(i);
            }
        }
        model.points.push_back(std::move(linePoint));
    }
    if (bestAnchor < 0) {
        throw std::runtime_error("Fiber line points have no valid sampled normals");
    }
    model.displayFrameAnchorIndex = bestAnchor;
    return model;
}

vc::lasagna::LineModel LineAnnotationController::syntheticLineModelFromPoints(
    const std::vector<cv::Vec3d>& points)
{
    if (points.empty()) {
        throw std::runtime_error("Fiber has no line points");
    }

    vc::lasagna::LineModel model;
    model.points.reserve(points.size());
    for (const cv::Vec3d& point : points) {
        vc::lasagna::LinePoint linePoint;
        linePoint.position = point;
        linePoint.sampledNormal = vc::lasagna::NormalSample{{0.0, 0.0, 1.0}, true, {}};
        model.points.push_back(linePoint);
    }
    model.displayFrameAnchorIndex = static_cast<int>(points.size() / 2);
    if (points.size() >= 2) {
        model.segmentSamples.reserve(points.size() - 1);
        for (size_t i = 1; i < points.size(); ++i) {
            vc::lasagna::LineSegmentSamples segment;
            segment.samples.push_back({0.0, points[i - 1], model.points[i - 1].sampledNormal});
            segment.samples.push_back({1.0, points[i], model.points[i].sampledNormal});
            model.segmentSamples.push_back(std::move(segment));
        }
    }
    return model;
}

cv::Vec3d LineAnnotationController::seedTraceSourceNormalForStoredFiber(
    const StoredFiber& fiber,
    std::optional<int> controlPointIndex,
    const cv::Vec3d& seedPoint)
{
    std::optional<int> preferredControlIndex = controlPointIndex;
    if (!preferredControlIndex) {
        int seedControlIndex = -1;
        for (size_t i = 0; i < fiber.controlPoints.size(); ++i) {
            if (pointsApproximatelyEqual(fiber.controlPoints[i], seedPoint)) {
                seedControlIndex = static_cast<int>(i);
                break;
            }
        }
        if (seedControlIndex >= 0) {
            preferredControlIndex = seedControlIndex;
        }
    }

    auto branchMatchesSeed = [&](const FiberBranchRef& branch) {
        if (preferredControlIndex && branch.controlPointIndex == *preferredControlIndex) {
            return true;
        }
        return pointsApproximatelyEqual(branch.controlPointPosition, seedPoint);
    };

    for (const auto& branch : fiber.branches) {
        if (branchMatchesSeed(branch) && finiteDirection(branch.controlPointDirection)) {
            return normalizedOrZero(branch.controlPointDirection);
        }
    }
    for (const auto& branch : fiber.branches) {
        if (finiteDirection(branch.controlPointDirection)) {
            return normalizedOrZero(branch.controlPointDirection);
        }
    }
    return {0.0, 0.0, 1.0};
}

std::shared_ptr<LineAnnotationController::LineAnnotationSession>
LineAnnotationController::makeIntersectionLineSession(
    const StoredFiber& fiber,
    double focusLinePosition,
    const cv::Vec3d& sourceSliceNormal,
    const std::string& surfaceName,
    std::function<void()> onOptimizationSucceeded)
{
    auto session = std::make_shared<LineAnnotationSession>();
    session->surfaceName = surfaceName;
    session->sourceAnnotationSurfaceName = surfaceName;
    session->fiberId = fiber.id;
    session->fiberUsername = fiber.username;
    session->fiberStartedAt = fiber.startedAt;
    session->fiberSequence = fiber.sequence;
    session->fiberFileName = fiber.fileName;
    session->fiberManualHvTag = fiber.manualHvTag;
    session->fiberTags = fiber.tags;
    session->branches = fiber.branches;
    session->focusedLinePosition = std::clamp(
        focusLinePosition,
        0.0,
        static_cast<double>(std::max<size_t>(1, fiber.linePoints.size()) - 1));
    session->focusedControlPoint = interpolatedPointAtLinePosition(
        fiber.linePoints,
        session->focusedLinePosition);
    session->sourceSliceNormal = finiteDirection(sourceSliceNormal)
        ? normalizedOrZero(sourceSliceNormal)
        : cv::Vec3d{0.0, 0.0, 1.0};
    session->optimizedLine = syntheticLineModelFromPoints(fiber.linePoints);
    session->taskState = LineAnnotationSession::TaskState::Succeeded;
    session->optimizationState = SessionOptimizationState::Optimized;
    session->suppressGeneratedViews = true;
    session->optimizationSucceededCallback =
        [callback = std::move(onOptimizationSucceeded)](LineAnnotationSession&) {
            if (callback) {
                callback();
            }
        };

    session->controlPoints.reserve(fiber.controlPoints.size());
    int seedControl = -1;
    double seedDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < fiber.controlPoints.size(); ++i) {
        const cv::Vec3d& controlPoint = fiber.controlPoints[i];
        const int index = static_cast<int>(
            vc3d::fiber_slice::nearestLinePointIndex(fiber.linePoints, controlPoint));
        vc::lasagna::LineControlPoint control;
        control.linePosition = static_cast<double>(index);
        control.volumePoint = controlPoint;
        control.optimizedIndex = index;
        session->controlPoints.push_back(control);
        const double distance = std::abs(control.linePosition - session->focusedLinePosition);
        if (distance < seedDistance) {
            seedDistance = distance;
            seedControl = static_cast<int>(i);
        }
    }
    if (session->controlPoints.empty()) {
        vc::lasagna::LineControlPoint control;
        control.linePosition = session->focusedLinePosition;
        control.volumePoint = *session->focusedControlPoint;
        control.optimizedIndex = static_cast<int>(std::llround(session->focusedLinePosition));
        control.isSeed = true;
        session->controlPoints.push_back(control);
        seedControl = 0;
    }
    if (seedControl < 0) {
        seedControl = 0;
    }
    session->controlPoints[static_cast<size_t>(seedControl)].isSeed = true;
    session->seedPoint = session->controlPoints[static_cast<size_t>(seedControl)].volumePoint;
    session->optimizedLine.displayFrameAnchorIndex =
        session->controlPoints[static_cast<size_t>(seedControl)].optimizedIndex;
    return session;
}

std::optional<int> LineAnnotationController::storedBranchTargetControlPointIndex(
    const FiberBranchRef& branch) const
{
    if (branch.branchControlPointIndex < 0) {
        return std::nullopt;
    }

    for (const auto& pane : _panes) {
        if (!pane.session ||
            !branchReferencesFiber(branch,
                                   pane.session->fiberId,
                                   pane.session->fiberFileName)) {
            continue;
        }
        if (auto index = storedControlPointIndexForSessionPosition(
                pane.session->controlPoints,
                branch.branchControlPointPosition)) {
            return index;
        }
        const auto sessionIndex = matchingSessionControlPointIndex(
            pane.session->controlPoints,
            branch.branchControlPointIndex,
            branch.branchControlPointPosition);
        if (!sessionIndex) {
            return std::nullopt;
        }
        const auto storedIndexForSessionIndex =
            storedIndexMapForSessionControls(pane.session->controlPoints);
        const int storedIndex =
            storedIndexForSessionIndex[static_cast<size_t>(*sessionIndex)];
        return storedIndex < 0 ? std::nullopt : std::optional<int>{storedIndex};
    }

    for (const auto& fiber : _fibers) {
        if (!branchReferencesFiber(branch, fiber.id, fiber.fileName)) {
            continue;
        }
        return matchingStoredControlPointIndex(fiber.controlPoints,
                                               branch.branchControlPointIndex,
                                               branch.branchControlPointPosition);
    }
    return std::nullopt;
}

LineAnnotationController::StoredFiberSessionSnapshot
LineAnnotationController::makeStoredFiberSessionSnapshot(LineAnnotationSession& session)
{
    ensureSessionFiberIdentity(session);
    StoredFiberSessionSnapshot snapshot;
    StoredFiber& fiber = snapshot.fiber;
    fiber.username = session.fiberUsername;
    fiber.startedAt = session.fiberStartedAt;
    fiber.sequence = session.fiberSequence;
    fiber.fileName = session.fiberFileName;
    auto existingIt = std::find_if(_fibers.begin(),
                                   _fibers.end(),
                                   [&fiber](const StoredFiber& existing) {
                                       return !fiber.fileName.empty() &&
                                              existing.fileName == fiber.fileName;
                                   });
    if (existingIt == _fibers.end() && session.fiberId != 0) {
        existingIt = std::find_if(_fibers.begin(),
                                  _fibers.end(),
                                  [&session](const StoredFiber& existing) {
                                      return existing.id == session.fiberId;
                                  });
    }
    fiber.id = existingIt == _fibers.end()
        ? (session.fiberId == 0 ? nextFiberId() : session.fiberId)
        : existingIt->id;
    fiber.generation = existingIt == _fibers.end()
        ? uint64_t{1}
        : std::max<uint64_t>(uint64_t{1}, existingIt->generation + 1);

    std::vector<size_t> order(session.controlPoints.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(),
                     order.end(),
                     [&session](size_t lhs, size_t rhs) {
                         return session.controlPoints[lhs].linePosition <
                                session.controlPoints[rhs].linePosition;
                     });
    snapshot.storedIndexForSessionIndex.assign(session.controlPoints.size(), -1);
    fiber.controlPoints.reserve(order.size());
    for (size_t storedIndex = 0; storedIndex < order.size(); ++storedIndex) {
        const size_t sessionIndex = order[storedIndex];
        snapshot.storedIndexForSessionIndex[sessionIndex] = static_cast<int>(storedIndex);
        fiber.controlPoints.push_back(session.controlPoints[sessionIndex].volumePoint);
    }

    fiber.linePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        fiber.linePoints.push_back(point.position);
    }

    fiber.branches.reserve(session.branches.size());
    for (const auto& branch : session.branches) {
        if (branch.controlPointIndex < 0 ||
            static_cast<size_t>(branch.controlPointIndex) >=
                snapshot.storedIndexForSessionIndex.size()) {
            continue;
        }
        FiberBranchRef storedBranch = branch;
        storedBranch.controlPointIndex =
            snapshot.storedIndexForSessionIndex[static_cast<size_t>(branch.controlPointIndex)];
        if (auto targetIndex = storedBranchTargetControlPointIndex(branch)) {
            storedBranch.branchControlPointIndex = *targetIndex;
        }
        if (storedBranch.controlPointIndex >= 0 &&
            static_cast<size_t>(storedBranch.controlPointIndex) < fiber.controlPoints.size()) {
            storedBranch.controlPointPosition =
                fiber.controlPoints[static_cast<size_t>(storedBranch.controlPointIndex)];
        }
        storedBranch.controlPointDirection =
            endpointTangentFromLinePoints(fiber.linePoints,
                                          storedBranch.controlPointPosition,
                                          storedBranch.controlPointDirection);
        if (finiteDirection(storedBranch.branchControlPointDirection)) {
            storedBranch.branchControlPointDirection =
                normalizedOrZero(storedBranch.branchControlPointDirection);
        }
        if (storedBranch.controlPointIndex >= 0 && storedBranch.branchFiberId != 0) {
            fiber.branches.push_back(std::move(storedBranch));
        }
    }

    fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(fiber.controlPoints);
    fiber.manualHvTag = session.fiberManualHvTag;
    fiber.tags = session.fiberTags;
    return snapshot;
}

LineAnnotationController::StoredFiber
LineAnnotationController::storedFiberFromSession(LineAnnotationSession& session)
{
    ensureSessionFiberIdentity(session);
    syncLinkedBranchMetadataAfterFiberModification(session);
    return makeStoredFiberSessionSnapshot(session).fiber;
}

void LineAnnotationController::saveSessionAsFiber(LineAnnotationSession& session)
{
    try {
        if (!finalizeSessionOptimizationSynchronously(session, false)) {
            return;
        }
        ensureSessionFiberIdentity(session);
        const BranchMetadataSyncResult branchSync =
            syncLinkedBranchMetadataAfterFiberModification(session);
        StoredFiber fiber = makeStoredFiberSessionSnapshot(session).fiber;
        session.fiberId = fiber.id;
        session.fiberUsername = fiber.username;
        session.fiberStartedAt = fiber.startedAt;
        session.fiberSequence = fiber.sequence;
        session.fiberFileName = fiber.fileName;
        session.fiberTags = fiber.tags;
        session.fiberMetricsMatchStoredFiber = true;
        for (const auto& pane : _panes) {
            if (pane.session.get() == &session && pane.dialog) {
                pane.dialog->setFiberDisplayName(
                    fiberDisplayNameFromFileName(session.fiberFileName));
            }
        }
        const uint64_t savedFiberId = fiber.id;

        auto it = std::find_if(_fibers.begin(), _fibers.end(), [&fiber](const StoredFiber& existing) {
            return !fiber.fileName.empty() && existing.fileName == fiber.fileName;
        });
        if (it == _fibers.end()) {
            it = std::find_if(_fibers.begin(), _fibers.end(), [&fiber](const StoredFiber& existing) {
                return existing.id == fiber.id;
            });
        }
        StoredFiber* savedFiber = nullptr;
        if (it == _fibers.end()) {
            _fibers.push_back(std::move(fiber));
            savedFiber = &_fibers.back();
        } else {
            *it = std::move(fiber);
            savedFiber = &*it;
        }
        if (savedFiber) {
            std::vector<FiberSaveSnapshot> snapshots;
            snapshots.push_back(makeFiberSaveSnapshot(*savedFiber));
            for (const uint64_t linkedFiberId : branchSync.affectedFiberIds) {
                if (linkedFiberId == savedFiberId) {
                    continue;
                }
                bool addedOpenPaneSnapshot = false;
                for (const auto& pane : _panes) {
                    if (!pane.session || pane.session.get() == &session ||
                        pane.session->fiberId != linkedFiberId ||
                        pane.session->taskState != LineAnnotationSession::TaskState::Succeeded ||
                        pane.session->optimizedLine.points.empty() ||
                        pane.session->controlPoints.empty()) {
                        continue;
                    }
                    StoredFiber linkedFiber = storedFiberFromSession(*pane.session);
                    auto linkedIt = std::find_if(
                        _fibers.begin(),
                        _fibers.end(),
                        [linkedFiberId, &linkedFiber](const StoredFiber& candidate) {
                            return candidate.id == linkedFiberId ||
                                   (!linkedFiber.fileName.empty() &&
                                    candidate.fileName == linkedFiber.fileName);
                        });
                    if (linkedIt == _fibers.end()) {
                        _fibers.push_back(std::move(linkedFiber));
                        linkedIt = std::prev(_fibers.end());
                    } else {
                        *linkedIt = std::move(linkedFiber);
                    }
                    snapshots.push_back(makeFiberSaveSnapshot(*linkedIt));
                    addedOpenPaneSnapshot = true;
                    break;
                }
                if (addedOpenPaneSnapshot) {
                    continue;
                }
                auto linkedIt = std::find_if(
                    _fibers.begin(),
                    _fibers.end(),
                    [linkedFiberId](const StoredFiber& candidate) {
                        return candidate.id == linkedFiberId;
                    });
                if (linkedIt != _fibers.end()) {
                    snapshots.push_back(makeFiberSaveSnapshot(*linkedIt));
                }
            }
            scheduleFiberSaveSnapshots(std::move(snapshots));
        }
        invalidateFiberAlignmentMetrics(savedFiberId, true);
        addKnownFiberTags(session.fiberTags);
        emitFiberSummaries();
        refreshBranchLineViews(savedFiberId);
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber: %1").arg(QString::fromStdString(ex.what())));
    }
}

nlohmann::json LineAnnotationController::fiberToJson(const StoredFiber& fiber, double scale) const
{
    StoredFiber serialized = fiber;
    auto branchFileNameForId = [this](uint64_t fiberId) -> std::string {
        for (const auto& storedFiber : _fibers) {
            if (storedFiber.id == fiberId) {
                return storedFiber.fileName;
            }
        }
        for (const auto& pane : _panes) {
            if (pane.session && pane.session->fiberId == fiberId) {
                return pane.session->fiberFileName;
            }
        }
        return {};
    };
    for (auto& branch : serialized.branches) {
        if (branch.branchFileName.empty()) {
            branch.branchFileName = branchFileNameForId(branch.branchFiberId);
        }
    }

    FiberSaveSnapshot snapshot;
    snapshot.fiber = std::move(serialized);
    copyCoordinateIdentityToJson(snapshot.coordinateIdentity, coordinateIdentityForState(_state));
    return fiberSaveSnapshotToJson(snapshot, scale);
}

nlohmann::json LineAnnotationController::fiberSaveSnapshotToJson(
    const FiberSaveSnapshot& snapshot,
    double scale)
{
    StoredFiber serialized = snapshot.fiber;
    scaleStoredFiber(serialized, scale);

    nlohmann::json root = nlohmann::json::object();
    root["type"] = "vc3d_fiber";
    root["version"] = 1;
    root["username"] = serialized.username;
    root["started_at"] = serialized.startedAt;
    root["sequence"] = serialized.sequence;
    root["filename"] = serialized.fileName;
    root["generation"] = serialized.generation;
    root["tags"] = serialized.tags;
    root["hv_classification"] = {
        {"z_distance", serialized.hvClassification.zDistance},
        {"control_point_length", serialized.hvClassification.fiberLength},
        {"horizontal_score", serialized.hvClassification.horizontalScore},
        {"vertical_score", serialized.hvClassification.verticalScore},
        {"automatic_tag", vc3d::line_annotation::fiberHvTagToString(serialized.hvClassification.automaticTag)},
        {"automatic_certainty", serialized.hvClassification.automaticCertainty},
        {"manual_tag", serialized.manualHvTag},
    };
    root["branches"] = nlohmann::json::array();
    for (const auto& branch : serialized.branches) {
        if (branch.controlPointIndex < 0 || branch.branchFiberId == 0) {
            continue;
        }
        if (branch.branchControlPointIndex < 0) {
            throw std::runtime_error("Branch link is missing a target control point index");
        }
        if (!finiteDirection(branch.controlPointDirection) ||
            !finiteDirection(branch.branchControlPointDirection)) {
            throw std::runtime_error(
                "Branch link is missing finite endpoint directions");
        }
        if (!finitePoint(branch.controlPointPosition) ||
            !finitePoint(branch.branchControlPointPosition)) {
            throw std::runtime_error("Branch link is missing endpoint control point positions");
        }
        nlohmann::json branchJson = {
            {"control_point_index", branch.controlPointIndex},
            {"branch_fiber_id", branch.branchFiberId},
            {"branch_control_point_index", branch.branchControlPointIndex},
            {"control_point_direction", pointToJson(normalizedOrZero(
                                            branch.controlPointDirection))},
            {"branch_control_point_direction", pointToJson(normalizedOrZero(
                                                   branch.branchControlPointDirection))},
            {"control_point_position", pointToJson(branch.controlPointPosition)},
            {"branch_control_point_position", pointToJson(branch.branchControlPointPosition)},
        };
        const std::string branchFileName = fs::path(branch.branchFileName).filename().string();
        if (branchFileName.empty()) {
            throw std::runtime_error("Branch link is missing branch_file");
        }
        branchJson["branch_file"] = branchFileName;
        if (branch.pending) {
            branchJson["pending"] = true;
        }
        root["branches"].push_back(std::move(branchJson));
    }
    root["control_points"] = nlohmann::json::array();
    root["line_points"] = nlohmann::json::array();
    for (const auto& point : serialized.controlPoints) {
        root["control_points"].push_back(pointToJson(point));
    }
    for (const auto& point : serialized.linePoints) {
        root["line_points"].push_back(pointToJson(point));
    }
    if (snapshot.coordinateIdentity.is_object()) {
        root.update(snapshot.coordinateIdentity);
    }
    return root;
}

void LineAnnotationController::saveFiberNow(const StoredFiber& fiber) const
{
    const fs::path dir = fibersDir();
    if (dir.empty()) {
        throw std::runtime_error("No volume package is loaded");
    }

    writeJsonAtomic(fiberPath(fiber), fiberToJson(fiber));
}

LineAnnotationController::FiberSaveSnapshot
LineAnnotationController::makeFiberSaveSnapshot(const StoredFiber& fiber) const
{
    FiberSaveSnapshot snapshot;
    snapshot.fiberId = fiber.id;
    snapshot.generation = fiber.generation;
    snapshot.path = fiberPath(fiber);
    snapshot.fiber = fiber;
    auto branchFileNameForId = [this](uint64_t fiberId) -> std::string {
        for (const auto& storedFiber : _fibers) {
            if (storedFiber.id == fiberId) {
                return storedFiber.fileName;
            }
        }
        for (const auto& pane : _panes) {
            if (pane.session && pane.session->fiberId == fiberId) {
                return pane.session->fiberFileName;
            }
        }
        return {};
    };
    for (auto& branch : snapshot.fiber.branches) {
        if (branch.branchFileName.empty()) {
            branch.branchFileName = branchFileNameForId(branch.branchFiberId);
        }
    }
    copyCoordinateIdentityToJson(snapshot.coordinateIdentity, coordinateIdentityForState(_state));
    if (snapshot.path.empty()) {
        throw std::runtime_error("No volume package is loaded");
    }
    return snapshot;
}

void LineAnnotationController::canonicalizeFiberSaveSnapshots(
    std::vector<FiberSaveSnapshot>& snapshots) const
{
    auto findSnapshotForBranch =
        [&snapshots](const FiberBranchRef& branch) -> FiberSaveSnapshot* {
        for (auto& snapshot : snapshots) {
            if (branchReferencesFiber(branch,
                                      snapshot.fiber.id,
                                      snapshot.fiber.fileName)) {
                return &snapshot;
            }
        }
        return nullptr;
    };

    for (auto& snapshot : snapshots) {
        for (auto& branch : snapshot.fiber.branches) {
            if (auto localIndex = matchingStoredControlPointIndex(
                    snapshot.fiber.controlPoints,
                    branch.controlPointIndex,
                    branch.controlPointPosition)) {
                branch.controlPointIndex = *localIndex;
                branch.controlPointPosition =
                    snapshot.fiber.controlPoints[static_cast<size_t>(*localIndex)];
            }
            branch.controlPointDirection =
                endpointTangentFromLinePoints(snapshot.fiber.linePoints,
                                              branch.controlPointPosition,
                                              branch.controlPointDirection);
            if (finiteDirection(branch.branchControlPointDirection)) {
                branch.branchControlPointDirection =
                    normalizedOrZero(branch.branchControlPointDirection);
            }

            auto* target = findSnapshotForBranch(branch);
            if (!target) {
                continue;
            }
            branch.branchFiberId = target->fiber.id;
            branch.branchFileName = target->fiber.fileName;
            if (auto targetIndex = matchingStoredControlPointIndex(
                    target->fiber.controlPoints,
                    branch.branchControlPointIndex,
                    branch.branchControlPointPosition)) {
                branch.branchControlPointIndex = *targetIndex;
                branch.branchControlPointPosition =
                    target->fiber.controlPoints[static_cast<size_t>(*targetIndex)];
            }
            branch.branchControlPointDirection =
                endpointTangentFromLinePoints(target->fiber.linePoints,
                                              branch.branchControlPointPosition,
                                              branch.branchControlPointDirection);
        }
    }

    for (auto& snapshot : snapshots) {
        for (auto& branch : snapshot.fiber.branches) {
            if (branch.controlPointIndex < 0 ||
                branch.branchControlPointIndex < 0 ||
                static_cast<size_t>(branch.controlPointIndex) >=
                    snapshot.fiber.controlPoints.size()) {
                continue;
            }
            auto* target = findSnapshotForBranch(branch);
            if (!target ||
                static_cast<size_t>(branch.branchControlPointIndex) >=
                    target->fiber.controlPoints.size()) {
                continue;
            }
            auto reciprocal = std::find_if(
                target->fiber.branches.begin(),
                target->fiber.branches.end(),
                [&snapshot, &branch](const FiberBranchRef& candidate) {
                    return branchReferencesFiber(candidate,
                                                 snapshot.fiber.id,
                                                 snapshot.fiber.fileName) &&
                           (candidate.controlPointIndex ==
                                branch.branchControlPointIndex ||
                            pointsApproximatelyEqual(candidate.controlPointPosition,
                                                     branch.branchControlPointPosition)) &&
                           (candidate.branchControlPointIndex ==
                                branch.controlPointIndex ||
                            pointsApproximatelyEqual(candidate.branchControlPointPosition,
                                                     branch.controlPointPosition));
                });
            if (reciprocal == target->fiber.branches.end()) {
                continue;
            }

            branch.branchFiberId = target->fiber.id;
            branch.branchFileName = target->fiber.fileName;
            branch.controlPointPosition =
                snapshot.fiber.controlPoints[static_cast<size_t>(branch.controlPointIndex)];
            branch.branchControlPointPosition =
                target->fiber.controlPoints[static_cast<size_t>(branch.branchControlPointIndex)];

            reciprocal->controlPointIndex = branch.branchControlPointIndex;
            reciprocal->branchFiberId = snapshot.fiber.id;
            reciprocal->branchControlPointIndex = branch.controlPointIndex;
            reciprocal->branchFileName = snapshot.fiber.fileName;
            reciprocal->controlPointPosition = branch.branchControlPointPosition;
            reciprocal->branchControlPointPosition = branch.controlPointPosition;
            reciprocal->controlPointDirection = branch.branchControlPointDirection;
            reciprocal->branchControlPointDirection = branch.controlPointDirection;
            reciprocal->pending = branch.pending;
        }
    }
}

void LineAnnotationController::validateFiberSaveSnapshots(
    const std::vector<FiberSaveSnapshot>& snapshots) const
{
    if (snapshots.size() < 2) {
        return;
    }

    auto findSnapshotForBranch =
        [&snapshots](const FiberBranchRef& branch) -> const FiberSaveSnapshot* {
        for (const auto& snapshot : snapshots) {
            if (branchReferencesFiber(branch,
                                      snapshot.fiber.id,
                                      snapshot.fiber.fileName)) {
                return &snapshot;
            }
        }
        return nullptr;
    };
    auto fail = [](const FiberSaveSnapshot& snapshot, const std::string& reason) {
        throw std::runtime_error(fiberErrorName(snapshot.fiber.fileName) + ": " + reason);
    };

    for (const auto& snapshot : snapshots) {
        for (const auto& branch : snapshot.fiber.branches) {
            const auto localIndex = matchingStoredControlPointIndex(
                snapshot.fiber.controlPoints,
                branch.controlPointIndex,
                branch.controlPointPosition);
            if (!localIndex) {
                fail(snapshot, "local CP position mismatch");
            }
            if (branch.branchFileName.empty()) {
                fail(snapshot, "missing branch_file");
            }

            const FiberSaveSnapshot* target = findSnapshotForBranch(branch);
            if (!target) {
                continue;
            }
            const auto targetIndex = matchingStoredControlPointIndex(
                target->fiber.controlPoints,
                branch.branchControlPointIndex,
                branch.branchControlPointPosition);
            if (!targetIndex) {
                fail(snapshot, "linked CP position mismatch");
            }

            const auto reciprocal = std::find_if(
                target->fiber.branches.begin(),
                target->fiber.branches.end(),
                [&snapshot, &branch](const FiberBranchRef& candidate) {
                    return branchReferencesFiber(candidate,
                                                 snapshot.fiber.id,
                                                 snapshot.fiber.fileName) &&
                           candidate.controlPointIndex == branch.branchControlPointIndex &&
                           candidate.branchControlPointIndex == branch.controlPointIndex &&
                           pointsApproximatelyEqual(candidate.controlPointPosition,
                                                    branch.branchControlPointPosition) &&
                           pointsApproximatelyEqual(candidate.branchControlPointPosition,
                                                    branch.controlPointPosition) &&
                           branchDirectionsCompatible(candidate.controlPointDirection,
                                                      branch.branchControlPointDirection) &&
                           branchDirectionsCompatible(candidate.branchControlPointDirection,
                                                      branch.controlPointDirection);
                });
            if (reciprocal == target->fiber.branches.end()) {
                fail(snapshot, "missing reciprocal branch");
            }
        }
    }
}

void LineAnnotationController::scheduleFiberSave(const StoredFiber& fiber)
{
    scheduleFiberSaveSnapshots({makeFiberSaveSnapshot(fiber)});
}

void LineAnnotationController::scheduleFiberPairSave(const StoredFiber& first,
                                                     const StoredFiber& second)
{
    if (first.id == second.id || fiberPath(first) == fiberPath(second)) {
        scheduleFiberSave(first);
        return;
    }
    std::vector<FiberSaveSnapshot> snapshots;
    snapshots.reserve(2);
    snapshots.push_back(makeFiberSaveSnapshot(first));
    snapshots.push_back(makeFiberSaveSnapshot(second));
    scheduleFiberSaveSnapshots(std::move(snapshots));
}

void LineAnnotationController::scheduleFiberSaveSnapshots(std::vector<FiberSaveSnapshot> snapshots)
{
    if (snapshots.empty()) {
        return;
    }
    canonicalizeFiberSaveSnapshots(snapshots);
    validateFiberSaveSnapshots(snapshots);

    auto jobKey = [](const FiberSaveJob& job) {
        std::vector<std::string> key;
        key.reserve(job.snapshots.size());
        for (const auto& snapshot : job.snapshots) {
            key.push_back(snapshot.path.lexically_normal().string());
        }
        std::sort(key.begin(), key.end());
        return key;
    };

    FiberSaveJob job;
    job.sequence = ++_nextFiberSaveSequence;
    job.snapshots = std::move(snapshots);
    FiberSaveJob probe = job;
    const auto key = jobKey(probe);
    for (auto& pending : _pendingFiberSaveJobs) {
        if (jobKey(pending) == key) {
            pending = std::move(job);
            startNextFiberSaveJob();
            return;
        }
    }
    _pendingFiberSaveJobs.push_back(std::move(job));
    startNextFiberSaveJob();
}

void LineAnnotationController::startNextFiberSaveJob()
{
    if (_fiberSaveRunning || _pendingFiberSaveJobs.empty()) {
        return;
    }

    FiberSaveJob job = std::move(_pendingFiberSaveJobs.front());
    _pendingFiberSaveJobs.pop_front();
    _fiberSaveRunning = true;

    auto* watcher = new QFutureWatcher<FiberSaveTaskResult>(this);
    _fiberSaveWatcher = watcher;
    connect(watcher,
            &QFutureWatcher<FiberSaveTaskResult>::finished,
            this,
            [this, watcher]() {
                finishFiberSaveJob(watcher);
            });

    watcher->setFuture(QtConcurrent::run([job = std::move(job)]() mutable {
        FiberSaveTaskResult result;
        result.ok = false;
        result.fiberIds.reserve(job.snapshots.size());
        result.generations.reserve(job.snapshots.size());
        for (const auto& snapshot : job.snapshots) {
            result.fiberIds.push_back(snapshot.fiberId);
            result.generations.push_back(snapshot.generation);
        }
        try {
            std::vector<vc3d::line_annotation::FiberSavePayload> payloads;
            payloads.reserve(job.snapshots.size());
            for (const auto& snapshot : job.snapshots) {
                payloads.push_back({snapshot.fiberId,
                                    snapshot.generation,
                                    snapshot.path,
                                    LineAnnotationController::fiberSaveSnapshotToJson(snapshot)});
            }
            const auto saveResult =
                vc3d::line_annotation::runFiberSaveJob(job.sequence, std::move(payloads));
            result.ok = saveResult.ok;
            result.fiberIds = saveResult.fiberIds;
            result.generations = saveResult.generations;
            result.recoveryFiles = saveResult.recoveryFiles;
            result.error = saveResult.error;
        } catch (const std::exception& ex) {
            result.error = ex.what();
        }
        return result;
    }));
}

void LineAnnotationController::finishFiberSaveJob(QFutureWatcher<FiberSaveTaskResult>* watcher)
{
    FiberSaveTaskResult result;
    try {
        result = watcher->result();
    } catch (const std::exception& ex) {
        result.ok = false;
        result.error = ex.what();
    }

    _fiberSaveRunning = false;
    if (_fiberSaveWatcher == watcher) {
        _fiberSaveWatcher = nullptr;
    }
    watcher->deleteLater();

    if (result.ok) {
        for (size_t i = 0; i < result.fiberIds.size(); ++i) {
            const uint64_t generation =
                i < result.generations.size() ? result.generations[i] : 0;
            emit fiberSaved(result.fiberIds[i], generation);
        }
    } else {
        QString message = tr("Could not save fiber data: %1")
                              .arg(QString::fromStdString(result.error));
        if (!result.recoveryFiles.empty()) {
            message += tr("\nRecovery backups were kept:");
            for (const auto& path : result.recoveryFiles) {
                message += QStringLiteral("\n") + QString::fromStdString(path.string());
            }
        }
        showError(message);
    }

    startNextFiberSaveJob();
}

void LineAnnotationController::waitForFiberSaves()
{
    while (_fiberSaveRunning || !_pendingFiberSaveJobs.empty()) {
        if (!_fiberSaveRunning) {
            startNextFiberSaveJob();
        }
        if (!_fiberSaveWatcher) {
            break;
        }
        QEventLoop loop;
        connect(_fiberSaveWatcher,
                &QFutureWatcher<FiberSaveTaskResult>::finished,
                &loop,
                &QEventLoop::quit);
        loop.exec();
    }
}

std::optional<LineAnnotationController::StoredFiber> LineAnnotationController::loadFiberJson(
    const nlohmann::json& root,
    const fs::path& path,
    std::vector<std::string>* branchErrors) const
{
    std::string stem = path.stem().string();
    const std::string originalStem = stem;
    if (stem.rfind("fiber_", 0) == 0) {
        stem = stem.substr(6);
    }
    const bool hasLegacyNumericStem = !stem.empty() &&
        std::all_of(stem.begin(), stem.end(), [](char ch) {
            return ch >= '0' && ch <= '9';
        });

    const std::string type = root.value("type", std::string{});
    if (type != "vc3d_fiber") {
        return std::nullopt;
    }
    if (root.value("version", 0) != 1) {
        throw std::runtime_error("Unsupported vc3d_fiber version");
    }

    StoredFiber fiber;
    fiber.generation = std::max<uint64_t>(uint64_t{1}, root.value("generation", uint64_t{1}));
    fiber.username = vc3d::line_annotation::normalizedFiberUsername(
        root.value("username", std::string{"anon"}));
    fiber.startedAt = root.value("started_at", std::string{});
    if (root.contains("sequence")) {
        fiber.sequence = root.at("sequence").get<uint64_t>();
    } else if (hasLegacyNumericStem) {
        fiber.sequence = std::stoull(stem);
    }
    fiber.fileName = path.filename().string();
    if (fiber.fileName.empty() && !fiber.startedAt.empty() && fiber.sequence > 0) {
        fiber.fileName = vc3d::line_annotation::fiberFileName(
            fiber.username, fiber.startedAt, fiber.sequence);
    }
    if (fiber.startedAt.empty() && !hasLegacyNumericStem) {
        Logger()->warn("VC3D fiber file {} has no started_at metadata", path.string());
    }
    if (fiber.sequence == 0 && !hasLegacyNumericStem) {
        Logger()->warn("VC3D fiber file {} has no sequence metadata", path.string());
    }
    if (fiber.fileName.empty()) {
        fiber.fileName = originalStem + ".json";
    }

    const auto& controls = root.at("control_points");
    const auto& linePoints = root.at("line_points");
    if (!controls.is_array() || !linePoints.is_array()) {
        throw std::runtime_error("control_points and line_points must be arrays");
    }

    fiber.controlPoints.reserve(controls.size());
    for (const auto& point : controls) {
        fiber.controlPoints.push_back(pointFromJson(point));
    }
    fiber.linePoints.reserve(linePoints.size());
    for (const auto& point : linePoints) {
        fiber.linePoints.push_back(pointFromJson(point));
    }
    vc::atlas::FiberInput atlasFiberInput;
    atlasFiberInput.controlPoints = fiber.controlPoints;
    atlasFiberInput.linePoints = fiber.linePoints;
    if (!(fiber.linePoints.empty() && fiber.controlPoints.size() == 1)) {
        vc::atlas::validateFiberInputControlPoints(atlasFiberInput);
    }

    if (root.contains("tags")) {
        fiber.tags = normalizedFiberTagsFromJson(root.at("tags"));
    }

    if (root.contains("branches")) {
        auto recordBranchError = [&](const std::string& reason) {
            if (branchErrors) {
                branchErrors->push_back(fiberErrorName(fiber.fileName) + ": " + reason);
                return true;
            }
            throw std::runtime_error(reason);
        };
        const auto& branches = root.at("branches");
        if (!branches.is_array()) {
            if (recordBranchError("branches must be an array")) {
                fiber.needsSave = true;
            }
        } else {
            for (const auto& branchJson : branches) {
                try {
                    if (!branchJson.is_object()) {
                        if (recordBranchError("branch entries must be objects")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    if (branchJson.contains("link_direction")) {
                        if (recordBranchError("obsolete branch metadata: link_direction")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    FiberBranchRef branch;
                    if (!branchJson.contains("control_point_index") ||
                        !branchJson.contains("branch_control_point_index") ||
                        !branchJson.contains("control_point_direction") ||
                        !branchJson.contains("branch_control_point_direction") ||
                        !branchJson.contains("control_point_position") ||
                        !branchJson.contains("branch_control_point_position")) {
                        if (recordBranchError("invalid branch metadata")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    if (!branchJson.contains("branch_file")) {
                        if (recordBranchError("missing branch_file")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    branch.controlPointIndex =
                        branchJson.at("control_point_index").get<int>();
                    branch.branchFiberId =
                        branchJson.value("branch_fiber_id", uint64_t{0});
                    branch.branchControlPointIndex =
                        branchJson.at("branch_control_point_index").get<int>();
                    branch.branchFileName =
                        fs::path(branchJson.at("branch_file").get<std::string>())
                            .filename()
                            .string();
                    branch.controlPointDirection =
                        pointFromJson(branchJson.at("control_point_direction"));
                    branch.branchControlPointDirection =
                        pointFromJson(branchJson.at("branch_control_point_direction"));
                    branch.controlPointPosition =
                        pointFromJson(branchJson.at("control_point_position"));
                    branch.branchControlPointPosition =
                        pointFromJson(branchJson.at("branch_control_point_position"));
                    branch.pending = branchJson.value("pending", false);
                    if (branch.controlPointIndex < 0 ||
                        static_cast<size_t>(branch.controlPointIndex) >=
                            fiber.controlPoints.size()) {
                        if (recordBranchError("local CP index out of range")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    if (branch.branchControlPointIndex < 0) {
                        if (recordBranchError("linked CP index out of range")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    if (branch.branchFileName.empty()) {
                        if (recordBranchError("missing branch_file")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    if (!finiteDirection(branch.controlPointDirection) ||
                        !finiteDirection(branch.branchControlPointDirection)) {
                        if (recordBranchError("invalid branch directions")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    if (!pointsApproximatelyEqual(
                            fiber.controlPoints[static_cast<size_t>(branch.controlPointIndex)],
                            branch.controlPointPosition)) {
                        if (recordBranchError("local CP position mismatch")) {
                            fiber.needsSave = true;
                            continue;
                        }
                    }
                    branch.controlPointDirection =
                        normalizedOrZero(branch.controlPointDirection);
                    branch.branchControlPointDirection =
                        normalizedOrZero(branch.branchControlPointDirection);
                    fiber.branches.push_back(std::move(branch));
                } catch (const std::exception& ex) {
                    if (recordBranchError(ex.what())) {
                        fiber.needsSave = true;
                        continue;
                    }
                }
            }
        }
    }

    fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(fiber.controlPoints);
    fiber.manualHvTag.clear();
    bool hasHvClassification = false;
    if (root.contains("hv_classification") && root.at("hv_classification").is_object()) {
        const auto& hv = root.at("hv_classification");
        hasHvClassification =
            hv.contains("z_distance") &&
            hv.contains("control_point_length") &&
            hv.contains("horizontal_score") &&
            hv.contains("vertical_score") &&
            hv.contains("automatic_tag") &&
            hv.contains("automatic_certainty") &&
            hv.contains("manual_tag");
        if (hv.contains("manual_tag")) {
            const std::string manualTag = vc3d::line_annotation::fiberHvTagToString(
                vc3d::line_annotation::fiberHvTagFromString(hv.value("manual_tag", std::string{})));
            fiber.manualHvTag = manualTag == "unknown" ? std::string{} : manualTag;
        }
        if (hasHvClassification) {
            const auto storedAutoTag = vc3d::line_annotation::fiberHvTagFromString(
                hv.value("automatic_tag", std::string{}));
            hasHvClassification =
                approximatelyEqual(hv.value("z_distance", -1.0),
                                   fiber.hvClassification.zDistance) &&
                approximatelyEqual(hv.value("control_point_length", -1.0),
                                   fiber.hvClassification.fiberLength) &&
                approximatelyEqual(hv.value("horizontal_score", -1.0),
                                   fiber.hvClassification.horizontalScore) &&
                approximatelyEqual(hv.value("vertical_score", -1.0),
                                   fiber.hvClassification.verticalScore) &&
                approximatelyEqual(hv.value("automatic_certainty", -1.0),
                                   fiber.hvClassification.automaticCertainty) &&
                storedAutoTag == fiber.hvClassification.automaticTag;
        }
    }
    if (!hasHvClassification) {
        fiber.needsSave = true;
    }
    return fiber;
}

std::optional<LineAnnotationController::StoredFiber> LineAnnotationController::loadFiberFile(
    const fs::path& path) const
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open fiber file");
    }
    const nlohmann::json root = nlohmann::json::parse(in);
    return loadFiberJson(root, path);
}

std::vector<LineAnnotationController::BranchLinkValidationIssue>
LineAnnotationController::collectLoadedFiberBranchIssues(
    const std::vector<StoredFiber>& fibers) const
{
    std::vector<BranchLinkValidationIssue> issues;
    std::unordered_map<std::string, size_t> indexByFileName;
    indexByFileName.reserve(fibers.size());
    for (size_t i = 0; i < fibers.size(); ++i) {
        if (!fibers[i].fileName.empty()) {
            indexByFileName[fibers[i].fileName] = i;
        }
    }

    for (size_t fiberIndex = 0; fiberIndex < fibers.size(); ++fiberIndex) {
        const StoredFiber& fiber = fibers[fiberIndex];
        for (size_t branchIndex = 0; branchIndex < fiber.branches.size(); ++branchIndex) {
            const FiberBranchRef& branch = fiber.branches[branchIndex];
            auto addIssue = [&](const std::string& reason) {
                issues.push_back({fiberIndex, branchIndex, reason});
            };

            if (branch.controlPointIndex < 0 ||
                static_cast<size_t>(branch.controlPointIndex) >= fiber.controlPoints.size()) {
                addIssue("local CP index out of range");
                continue;
            }
            if (!pointsApproximatelyEqual(
                    fiber.controlPoints[static_cast<size_t>(branch.controlPointIndex)],
                    branch.controlPointPosition)) {
                addIssue("local CP position mismatch");
                continue;
            }
            if (!finiteDirection(branch.controlPointDirection) ||
                !finiteDirection(branch.branchControlPointDirection)) {
                addIssue("invalid branch directions");
                continue;
            }
            if (fiber.linePoints.size() >= 2) {
                const cv::Vec3d expectedLocal =
                    endpointTangentFromLinePoints(fiber.linePoints,
                                                  branch.controlPointPosition);
                if (!branchDirectionsCompatible(branch.controlPointDirection,
                                                expectedLocal)) {
                    addIssue("branch endpoint direction mismatch");
                    continue;
                }
            }
            if (branch.branchFileName.empty()) {
                addIssue("missing branch_file");
                continue;
            }
            const auto targetIndex = indexByFileName.find(branch.branchFileName);
            if (targetIndex == indexByFileName.end()) {
                addIssue("missing linked fiber");
                continue;
            }
            const StoredFiber& target = fibers[targetIndex->second];
            if (branch.branchControlPointIndex < 0 ||
                static_cast<size_t>(branch.branchControlPointIndex) >=
                    target.controlPoints.size()) {
                addIssue("linked CP index out of range");
                continue;
            }
            if (!pointsApproximatelyEqual(
                    target.controlPoints[static_cast<size_t>(branch.branchControlPointIndex)],
                    branch.branchControlPointPosition)) {
                addIssue("linked CP position mismatch");
                continue;
            }
            if (target.linePoints.size() >= 2) {
                const cv::Vec3d expectedLinked =
                    endpointTangentFromLinePoints(target.linePoints,
                                                  branch.branchControlPointPosition);
                if (!branchDirectionsCompatible(branch.branchControlPointDirection,
                                                expectedLinked)) {
                    addIssue("branch endpoint direction mismatch");
                    continue;
                }
            }

            const auto reciprocal = std::find_if(
                target.branches.begin(),
                target.branches.end(),
                [&fiber, &branch](const FiberBranchRef& candidate) {
                    return candidate.branchFileName == fiber.fileName &&
                           candidate.controlPointIndex == branch.branchControlPointIndex &&
                           candidate.branchControlPointIndex == branch.controlPointIndex &&
                           pointsApproximatelyEqual(candidate.controlPointPosition,
                                                    branch.branchControlPointPosition) &&
                           pointsApproximatelyEqual(candidate.branchControlPointPosition,
                                                    branch.controlPointPosition) &&
                           branchDirectionsCompatible(candidate.controlPointDirection,
                                                      branch.branchControlPointDirection) &&
                           branchDirectionsCompatible(candidate.branchControlPointDirection,
                                                      branch.controlPointDirection);
                });
            if (reciprocal == target.branches.end()) {
                addIssue("missing reciprocal branch");
            }
        }
    }
    return issues;
}

bool LineAnnotationController::repairLoadedFiberBranchLinks(
    std::vector<StoredFiber>& fibers,
    const std::unordered_set<std::string>& fibersWithRemovedBranchEntries,
    const std::vector<BranchLinkValidationIssue>& initialIssues,
    std::vector<std::string>& errors) const
{
    std::unordered_set<std::string> changedFiles = fibersWithRemovedBranchEntries;

    auto removeIssues = [&](const std::vector<BranchLinkValidationIssue>& issues) {
        std::unordered_map<size_t, std::vector<size_t>> branchIndicesByFiber;
        for (const auto& issue : issues) {
            if (issue.fiberIndex >= fibers.size()) {
                continue;
            }
            if (issue.branchIndex >= fibers[issue.fiberIndex].branches.size()) {
                continue;
            }
            branchIndicesByFiber[issue.fiberIndex].push_back(issue.branchIndex);
        }

        bool changed = false;
        for (auto& [fiberIndex, branchIndices] : branchIndicesByFiber) {
            auto& fiber = fibers[fiberIndex];
            std::sort(branchIndices.begin(), branchIndices.end());
            branchIndices.erase(std::unique(branchIndices.begin(), branchIndices.end()),
                                branchIndices.end());
            for (auto it = branchIndices.rbegin(); it != branchIndices.rend(); ++it) {
                if (*it >= fiber.branches.size()) {
                    continue;
                }
                fiber.branches.erase(fiber.branches.begin() +
                                     static_cast<std::ptrdiff_t>(*it));
                fiber.needsSave = true;
                changedFiles.insert(fiber.fileName);
                changed = true;
            }
        }
        return changed;
    };

    (void)removeIssues(initialIssues);
    for (;;) {
        const auto issues = collectLoadedFiberBranchIssues(fibers);
        if (issues.empty()) {
            break;
        }
        if (!removeIssues(issues)) {
            break;
        }
    }

    for (auto& fiber : fibers) {
        if (changedFiles.find(fiber.fileName) == changedFiles.end() && !fiber.needsSave) {
            continue;
        }
        try {
            fiber.needsSave = false;
            saveFiberNow(fiber);
        } catch (const std::exception& ex) {
            fiber.needsSave = true;
            errors.push_back(fiberErrorName(fiber.fileName) + ": " + ex.what());
        }
    }

    return errors.empty();
}

bool LineAnnotationController::validateLoadedFiberLinks(std::vector<StoredFiber>& fibers,
                                                        std::vector<std::string>& errors) const
{
    bool removedInvalidFibers = false;
    for (;;) {
        const auto issues = collectLoadedFiberBranchIssues(fibers);
        if (issues.empty()) {
            break;
        }

        std::unordered_set<std::string> invalidFiles;
        for (const auto& issue : issues) {
            if (issue.fiberIndex >= fibers.size()) {
                continue;
            }
            const auto& fiber = fibers[issue.fiberIndex];
            invalidFiles.insert(fiber.fileName);
            errors.push_back(fiberErrorName(fiber.fileName) + ": " + issue.reason);
        }
        if (invalidFiles.empty()) {
            break;
        }
        removedInvalidFibers = true;
        fibers.erase(std::remove_if(fibers.begin(),
                                    fibers.end(),
                                    [&invalidFiles](const StoredFiber& fiber) {
                                        return invalidFiles.find(fiber.fileName) !=
                                               invalidFiles.end();
                                    }),
                     fibers.end());
    }

    std::unordered_map<std::string, uint64_t> fiberIdByFileName;
    fiberIdByFileName.reserve(fibers.size());
    uint64_t runtimeId = 1;
    for (auto& fiber : fibers) {
        fiber.id = runtimeId++;
        if (!fiber.fileName.empty()) {
            fiberIdByFileName[fiber.fileName] = fiber.id;
        }
    }
    for (auto& fiber : fibers) {
        for (auto& branch : fiber.branches) {
            if (auto it = fiberIdByFileName.find(branch.branchFileName);
                it != fiberIdByFileName.end()) {
                branch.branchFiberId = it->second;
            }
        }
    }
    return !removedInvalidFibers;
}

std::string LineAnnotationController::uniqueImportedFiberFileName(
    const StoredFiber& fiber,
    std::unordered_set<std::string>& reserved,
    uint64_t& nextSequence) const
{
    std::string requested = fs::path(fiber.fileName).filename().string();
    if (requested.empty()) {
        const std::string username = fiber.username.empty() ? currentFiberUsername() : fiber.username;
        const std::string startedAt = fiber.startedAt.empty()
            ? currentFiberDateTimeString()
            : fiber.startedAt;
        const uint64_t sequence = fiber.sequence == 0 ? nextSequence++ : fiber.sequence;
        requested = vc3d::line_annotation::fiberFileName(username, startedAt, sequence);
    }

    if (fs::path(requested).extension() != ".json") {
        requested += ".json";
    }

    const fs::path dir = fibersDir();
    const std::string stem = fs::path(requested).stem().string();
    const std::string extension = fs::path(requested).extension().string().empty()
        ? ".json"
        : fs::path(requested).extension().string();

    auto available = [&](const std::string& candidate) {
        if (candidate.empty() || reserved.count(candidate) != 0) {
            return false;
        }
        std::error_code ec;
        return !fs::exists(dir / candidate, ec);
    };

    if (available(requested)) {
        reserved.insert(requested);
        return requested;
    }

    for (uint64_t suffix = 1; suffix < std::numeric_limits<uint64_t>::max(); ++suffix) {
        const std::string candidate = stem + "_import" + std::to_string(suffix) + extension;
        if (available(candidate)) {
            reserved.insert(candidate);
            return candidate;
        }
    }

    throw std::runtime_error("Could not find an available imported fiber file name");
}

void LineAnnotationController::showError(const QString& message) const
{
    if (_parentWidget) {
        QMessageBox::warning(_parentWidget, tr("Line Annotation"), message);
    } else {
        Logger()->warn("Line Annotation: {}", message.toStdString());
    }
}
