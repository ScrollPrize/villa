#include "LineAnnotationController.hpp"

#include "CState.hpp"
#include "FiberSliceGeometry.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "LineAnnotationDialog.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/FiberSliceOverlayController.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
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
#include <QEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMessageBox>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPoint>
#include <QPointF>
#include <QPushButton>
#include <QRadioButton>
#include <QDateTime>
#include <QSettings>
#include <QStringList>
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
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <map>
#include <sstream>
#include <string_view>
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
    std::shared_ptr<vc::lasagna::LasagnaDataset> dataset;
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> normalSampler;
    TaskState taskState = TaskState::Idle;
    cv::Vec3d seedPoint{0.0, 0.0, 0.0};
    std::string sourceAnnotationSurfaceName;
    vc::lasagna::LineOptimizationReport optimizationReport;
    vc::lasagna::LineModel optimizedLine;
    std::vector<vc::lasagna::LineControlPoint> controlPoints;
    double focusedLinePosition = 0.0;
    std::optional<cv::Vec3d> focusedControlPoint;
    cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
    LineAnnotationController::InitialDirectionMode initialDirectionMode =
        LineAnnotationController::InitialDirectionMode::Sideways;
    std::vector<std::string> generatedSurfaceNames;
    std::string error;
    QPointer<QFutureWatcher<OptimizationTaskResult>> watcher;
    bool deferShowUntilGenerated = false;
    uint64_t fiberId = 0;
    std::string fiberUsername;
    std::string fiberStartedAt;
    uint64_t fiberSequence = 0;
    std::string fiberFileName;
    std::string fiberManualHvTag;
    bool suppressFiberSave = false;
    bool suppressGeneratedViews = false;
    std::function<void(LineAnnotationSession&)> optimizationSucceededCallback;
};

struct LineAnnotationController::IntersectionInspectionSession {
    struct FollowSlice {
        bool valid = false;
        bool followsMouse = true;
        bool sourceSide = true;
        std::string surfaceName;
        QPointer<CChunkedVolumeViewer> viewer;
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
    std::optional<bool> activeFollowSourceSide;
    std::map<std::string, GeneratedSurfaceContext> generatedSurfaceContexts;
};

namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kLineSegmentLength = 32.0;
using Clock = std::chrono::steady_clock;

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
generatedControlMarkers(const std::vector<vc::lasagna::LineControlPoint>& controls)
{
    std::vector<vc3d::line_annotation::GeneratedOverlay::ControlPointMarker> markers;
    markers.reserve(controls.size());
    for (const auto& control : controls) {
        vc3d::line_annotation::GeneratedOverlay::ControlPointMarker marker;
        marker.point = toVec3f(control.volumePoint);
        marker.linePosition = control.linePosition;
        marker.isSeed = control.isSeed;
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
        : (forceFullOptimization ? "full_optimization" : "control_optimization");
    try {
        vc::lasagna::LineOptimizer optimizer(sampler);
        vc::lasagna::LineOptimizationConfig config;
        config.segmentsPerSide = 200;
        config.segmentLength = kLineSegmentLength;
        config.straightnessWeight = 0.1;
        config.tangentStraightnessWeight = 5.0;
        config.normalStraightnessWeight = 0.05;
        config.samplesPerSegment = 1;
        config.maxIterations = 1000;
        config.differentiableNormalSampling = true;
        config.printSolverProgress = false;
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
        if (initialLinePoints.size() >= 2) {
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
            const bool hasLocalRange = activeStart >= 0 && activeEnd >= activeStart;
            const std::string candidateName = forceFullOptimization || !hasLocalRange
                ? "existing-line+global"
                : "existing-line+local";
            task.result = optimizer.optimizeExistingLine(std::move(initialLinePoints),
                                                         std::move(fixedIndices),
                                                         displayFrameAnchorIndex,
                                                         config,
                                                         forceFullOptimization ? -1 : activeStart,
                                                         forceFullOptimization ? -1 : activeEnd,
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
                                  bool forceFullOptimization,
                                  int activeStart,
                                  int activeEnd) {
        return optimizeLineFromManifest(std::move(manifestPath),
                                        std::move(controlPoints),
                                        std::move(initialLinePoints),
                                        sourceSliceNormal,
                                        directionMode,
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
}

void LineAnnotationController::setDatasetPickerForTesting(DatasetPicker picker)
{
    _datasetPicker = std::move(picker);
}

void LineAnnotationController::setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory)
{
    _optimizationTaskFactory = std::move(factory);
}

void LineAnnotationController::setSurfacePanel(SurfacePanelController* panel)
{
    _surfacePanel = panel;
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
    return surfaceName == "segmentation" ||
           surfaceName.rfind("line-", 0) == 0;
}

void LineAnnotationController::launchFromViewer(CChunkedVolumeViewer* viewer, const QPointF& /*scenePoint*/)
{
    if (!canLaunchFromViewer(viewer)) {
        return;
    }

    auto surfaceName = nextSurfaceName();
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
    launchSession(sourceKind,
                  surfaceName,
                  std::move(sourceSurface),
                  camera,
                  sourceSliceNormal,
                  std::move(session));
}

void LineAnnotationController::launchFromViewerAtPoint(CChunkedVolumeViewer* viewer, const QPointF& scenePoint)
{
    if (!canLaunchFromViewer(viewer)) {
        return;
    }

    const auto sample = viewer->sampleSceneVolume(scenePoint);
    if (!sample) {
        return;
    }
    const std::string clickedSurfaceName = viewer->surfName();

    cv::Vec3f normal = sample->normal;
    if (!std::isfinite(normal[0]) ||
        !std::isfinite(normal[1]) ||
        !std::isfinite(normal[2]) ||
        cv::norm(normal) <= 0.0f) {
        normal = {0.0f, 0.0f, 1.0f};
    }
    normal *= 1.0f / cv::norm(normal);

    std::string owningSurfaceName;
    QPointer<LineAnnotationDialog> owningDialog;
    if (auto* owner = paneForSurface(clickedSurfaceName)) {
        owningSurfaceName = owner->surfaceName;
        owningDialog = owner->dialog;
    }
    if (!owningSurfaceName.empty()) {
        cleanupSurfaceName(owningSurfaceName);
        if (owningDialog) {
            owningDialog->close();
        }
    }

    auto surfaceName = nextSurfaceName();
    CChunkedVolumeViewer::CameraState camera;
    auto sourceSurface = std::make_shared<PlaneSurface>(sample->position, normal);
    auto session = std::make_shared<LineAnnotationSession>();
    launchSession(SourceKind::Plane,
                  surfaceName,
                  std::move(sourceSurface),
                  camera,
                  cv::Vec3d{normal[0], normal[1], normal[2]},
                  session,
                  true);
    handleLineSeed(surfaceName, sample->position, InitialDirectionMode::ZInOut);
}

void LineAnnotationController::launchSession(LineAnnotationController::SourceKind sourceKind,
                                             const std::string& surfaceName,
                                             std::shared_ptr<Surface> sourceSurface,
                                             const CChunkedVolumeViewer::CameraState& camera,
                                             cv::Vec3d sourceSliceNormal,
                                             std::shared_ptr<LineAnnotationController::LineAnnotationSession> session,
                                             bool deferShowUntilGenerated)
{
    if (!_state || !session) {
        return;
    }

    session->deferShowUntilGenerated = deferShowUntilGenerated;
    _state->setSurface(surfaceName, std::move(sourceSurface));
    auto* dialog = new LineAnnotationDialog(_viewerManager, nullptr);
    if (!dialog->addPane(surfaceName, tr("Line Annotation Slice"), camera)) {
        dialog->deleteLater();
        _state->setSurface(surfaceName, nullptr);
        return;
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
    connect(dialog, &QObject::destroyed, this, [this, surfaceName]() {
        cleanupSurfaceName(surfaceName);
    });

    if (!session->optimizedLine.points.empty()) {
        session->taskState = LineAnnotationSession::TaskState::Succeeded;
        materializeGeneratedViews(*session);
    }
    if (!session->deferShowUntilGenerated || !session->optimizedLine.points.empty()) {
        dialog->showMaximized();
        dialog->raise();
        dialog->activateWindow();
    }
}

void LineAnnotationController::openFiber(uint64_t fiberId)
{
    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const StoredFiber& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
        showError(tr("Fiber %1 is not loaded.").arg(fiberId));
        return;
    }
    if (it->linePoints.empty()) {
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
    session->focusedLinePosition = static_cast<double>(it->linePoints.size() / 2);
    session->focusedControlPoint = it->controlPoints.empty()
        ? std::optional<cv::Vec3d>{}
        : std::optional<cv::Vec3d>{it->controlPoints[it->controlPoints.size() / 2]};

    if (!ensureDatasetForSession(*session)) {
        return;
    }

    try {
        session->optimizedLine = lineModelFromPoints(it->linePoints, session->normalSampler.get());
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
        if (seedControl < 0) {
            seedControl = 0;
        }
        session->controlPoints[static_cast<size_t>(seedControl)].isSeed = true;
        session->seedPoint = session->controlPoints[static_cast<size_t>(seedControl)].volumePoint;
        session->optimizedLine.displayFrameAnchorIndex =
            session->controlPoints[static_cast<size_t>(seedControl)].optimizedIndex;
        session->focusedLinePosition =
            session->controlPoints[static_cast<size_t>(seedControl)].linePosition;
    }

    CChunkedVolumeViewer::CameraState camera;
    camera.scale = 1.0f;
    const cv::Vec3d origin = it->linePoints.empty()
        ? cv::Vec3d{0.0, 0.0, 0.0}
        : it->linePoints[it->linePoints.size() / 2];
    auto sourcePlane = std::make_shared<PlaneSurface>(
        cv::Vec3f{static_cast<float>(origin[0]),
                  static_cast<float>(origin[1]),
                  static_cast<float>(origin[2])},
        cv::Vec3f{0.0f, 0.0f, 1.0f});
    launchSession(SourceKind::Plane,
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
    }
    if (deletedIds.empty()) {
        return;
    }

    _fibers.erase(std::remove_if(_fibers.begin(),
                                 _fibers.end(),
                                 [&deletedIds](const StoredFiber& fiber) {
                                     return std::binary_search(deletedIds.begin(),
                                                               deletedIds.end(),
                                                               fiber.id);
                                 }),
                  _fibers.end());
    for (const auto& pane : _panes) {
        if (pane.session && std::binary_search(deletedIds.begin(),
                                               deletedIds.end(),
                                               pane.session->fiberId)) {
            pane.session->suppressFiberSave = true;
        }
    }
    emitFiberSummaries();
    emit fibersDeleted(deletedIds);
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
        saveFiber(*it);
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
        saveFiber(*it);
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
            saveFiber(fiber);
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

        fs::path manifestPath = vpkg->selectedLasagnaDatasetPath();
        if (manifestPath.empty()) {
            const auto picked = pickDataset(_parentWidget.data(), volpkgRoot);
            if (!picked) {
                throw std::runtime_error("No Lasagna normal dataset selected");
            }
            vpkg->setSelectedLasagnaDataset(*picked);
            manifestPath = vpkg->selectedLasagnaDatasetPath();
        }
        if (manifestPath.empty() || !fs::exists(manifestPath)) {
            throw std::runtime_error("Selected Lasagna normal dataset does not exist");
        }
        vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
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
        input.fiberPath = fs::relative(fiberPath(fiberId), volpkgRoot, relativeEc);
        if (relativeEc || input.fiberPath.empty()) {
            input.fiberPath = fs::path("fibers") / (std::to_string(fiberId) + ".json");
        }
        input.controlPoints = fiberIt->controlPoints;
        input.linePoints = fiberIt->linePoints;
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
        vc::atlas::saveAtlasBaseMeshCopy(*selected.surface,
                                         atlasDir / atlas.metadata.baseMeshPath);
        atlas.save(atlasDir);
        emit atlasCreated(atlasDir);
    } catch (const std::exception& ex) {
        showError(tr("Could not create atlas: %1").arg(QString::fromStdString(ex.what())));
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
                input.fiberPath = fs::path("fibers") / (std::to_string(fiber.id) + ".json");
            }
            input.controlPoints = fiber.controlPoints;
            input.linePoints = fiber.linePoints;
            return input;
        };
        const vc::atlas::FiberInput sourceInput = makeInput(*sourceIt);
        const vc::atlas::FiberInput targetInput = makeInput(*targetIt);

        vc::atlas::Atlas atlas = vc::atlas::Atlas::load(*_intersectionInspection->atlasDir);
        auto findMapping = [](vc::atlas::Atlas& atlas, const fs::path& fiberPath) {
            return std::find_if(atlas.fibers.begin(),
                                atlas.fibers.end(),
                                [&fiberPath](const vc::atlas::FiberMapping& mapping) {
                                    return mapping.fiberPath == fiberPath;
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

        fs::path manifestPath = vpkg->selectedLasagnaDatasetPath();
        if (manifestPath.empty()) {
            throw std::runtime_error("No Lasagna normal dataset selected");
        }
        if (!fs::exists(manifestPath)) {
            throw std::runtime_error("Selected Lasagna normal dataset does not exist");
        }
        vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
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
    }
}

bool LineAnnotationController::handleIntersectionFollowKeyPress(int key,
                                                                Qt::KeyboardModifiers modifiers)
{
    if (!_intersectionInspection || key != Qt::Key_Space || modifiers != Qt::NoModifier) {
        return false;
    }
    if (_intersectionInspection->activeFollowSourceSide) {
        toggleIntersectionFollowSlice(*_intersectionInspection->activeFollowSourceSide);
    } else {
        toggleIntersectionFollowSlice(true);
        toggleIntersectionFollowSlice(false);
    }
    return true;
}

bool LineAnnotationController::eventFilter(QObject* watched, QEvent* event)
{
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
            side.lineViews = vc::lasagna::buildLineViewSurfaces(
                syntheticLineModelFromPoints(side.fiber->linePoints));
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
            if (auto* viewerWidget = qobject_cast<QWidget*>(viewer->asQObject())) {
                viewerWidget->installEventFilter(this);
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
                                if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
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
                            views.controlPoints = generatedControlMarkers(session->controlPoints);
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
                [chunkedViewer,
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
                    views.controlPoints = generatedControlMarkers(session->controlPoints);
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
                        if (modifiers != Qt::NoModifier) {
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
                            handleGeneratedControlPoint(surfaceName, volumePoint, linePosition);
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
        vc::atlas::FiberSpatialIndex index;
        vc::atlas::FiberIntersectionCache cache;
        vc::atlas::FiberIntersectionBroadPhaseOptions broad;
        vc::atlas::FiberIntersectionCeresOptions ceres;
        const auto results = vc::atlas::searchFiberIntersections(
            fibers,
            {sourceId},
            {targetId},
            index,
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
        saveSessionAsFiber(session);
    }
}

void LineAnnotationController::closeFiberWindowForSurface(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (pane && pane->dialog) {
        pane->dialog->close();
    }
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
            globalPos);
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
        options.controlPoints = generatedControlMarkers(pane->session->controlPoints);
        options.linePointCount = pane->session->optimizedLine.points.empty()
            ? pane->session->controlPoints.size()
            : pane->session->optimizedLine.points.size();
        options.linePosition = linePosition;
        options.stripViewer = context.strip;
        options.deleteControlPoint = [this, surfaceName = viewer->surfName()](
                                         double selectedLinePosition,
                                         cv::Vec3f selectedPoint) {
            handleGeneratedControlPointDelete(surfaceName,
                                              selectedLinePosition,
                                              selectedPoint);
        };
        result = vc3d::line_annotation::showGeneratedControlPointContextMenu(options);
    }
    if (result == LineAnnotationDialog::GeneratedControlPointContextResult::NewLineAnnotationRequested) {
        launchFromViewerAtPoint(viewer, scenePoint);
    }
    return result != LineAnnotationDialog::GeneratedControlPointContextResult::None;
}

std::vector<LineAnnotationController::FiberSummary> LineAnnotationController::fiberSummaries() const
{
    std::vector<FiberSummary> summaries;
    summaries.reserve(_fibers.size());
    for (const auto& fiber : _fibers) {
        summaries.push_back(FiberSummary{
            fiber.id,
            fiber.fileName,
            static_cast<int>(fiber.controlPoints.size()),
            static_cast<int>(fiber.linePoints.size()),
            lineLengthVx(fiber.linePoints),
            fiber.hvClassification.zDistance,
            fiber.hvClassification.fiberLength,
            fiber.hvClassification.horizontalScore,
            fiber.hvClassification.verticalScore,
            fiber.hvClassification.automaticCertainty,
            vc3d::line_annotation::fiberHvTagToString(fiber.hvClassification.automaticTag),
            fiber.manualHvTag,
        });
    }
    std::sort(summaries.begin(), summaries.end(), [](const FiberSummary& a, const FiberSummary& b) {
        return a.id < b.id;
    });
    return summaries;
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

    const double maxPosition = static_cast<double>(session.optimizedLine.points.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
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
    std::vector<cv::Vec3d> currentLinePoints;
    currentLinePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        currentLinePoints.push_back(point.position);
    }

    vc::lasagna::LineControlPointUpdateResult update;
    const std::string updateEventName = editedExistingControl
        ? "control_edit_span_update"
        : "control_add_span_update";
    const auto updateStart = Clock::now();
    try {
        vc::lasagna::LineOptimizationConfig updateConfig;
        updateConfig.segmentsPerSide = 200;
        updateConfig.segmentLength = kLineSegmentLength;
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

    const bool deletedSeed = selected->isSeed;
    session.controlPoints.erase(selected);
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
    startOptimization(session, true);
}

bool LineAnnotationController::ensureDatasetForSession(LineAnnotationSession& session)
{
    if (!_state || !_state->vpkg()) {
        showError(tr("No volume package loaded."));
        return false;
    }

    auto vpkg = _state->vpkg();
    std::string selected = vpkg->selectedLasagnaDataset();
    fs::path manifestPath = vpkg->selectedLasagnaDatasetPath();

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
                vc::lasagna::LasagnaDataset::open(manifestPath));
            auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
            session.dataset = std::move(dataset);
            session.normalSampler = std::move(sampler);
        } catch (const std::exception& ex) {
            showError(tr("Invalid Lasagna dataset: %1").arg(QString::fromStdString(ex.what())));
            return false;
        }
        vpkg->setSelectedLasagnaDataset(selected);
    } else {
        if (!session.normalSampler || session.selectedManifestPath != manifestPath) {
            try {
                auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                    vc::lasagna::LasagnaDataset::open(manifestPath));
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
    return true;
}

void LineAnnotationController::startOptimization(LineAnnotationSession& session,
                                                 bool forceFullOptimization,
                                                 int activeStart,
                                                 int activeEnd)
{
    if (session.controlPoints.empty()) {
        return;
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
    auto dataset = session.dataset;
    auto normalSampler = session.normalSampler;
    watcher->setFuture(QtConcurrent::run([factory,
                                           manifestPath,
                                           controlPoints,
                                           initialLinePoints,
                                           sourceSliceNormal,
                                           directionMode,
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
    if (task.ok) {
        session.taskState = LineAnnotationSession::TaskState::Succeeded;
        session.seedPoint = task.seedPoint;
        session.selectedManifestPath = task.manifestPath;
        session.optimizationReport = task.result.report;
        session.optimizedLine = std::move(task.result.line);
        session.controlPoints = std::move(task.controlPoints);
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
                control.optimizedIndex = bestIndex;
                control.linePosition = static_cast<double>(bestIndex);
                control.volumePoint = session.optimizedLine.points[static_cast<size_t>(bestIndex)].position;
                const bool matchesFocusedControl = session.focusedControlPoint.has_value() &&
                    std::sqrt((control.volumePoint - *session.focusedControlPoint).dot(
                        control.volumePoint - *session.focusedControlPoint)) <= 1.0e-6;
                if (std::abs(session.focusedLinePosition - control.linePosition) <= 0.5 ||
                    matchesFocusedControl) {
                    session.focusedLinePosition = control.linePosition;
                }
            }
            if (control.isSeed) {
                session.seedPoint = control.volumePoint;
            }
        }
        const std::string resultEvent = task.eventName.empty()
            ? "optimization_result"
            : task.eventName + "_result";
        writeLineDebugJson(resultEvent,
                           session.controlPoints,
                           linePointsToJson(session.optimizedLine),
                           &session.optimizationReport);
        if (!session.suppressGeneratedViews) {
            if (!materializeGeneratedViews(session)) {
                session.taskState = LineAnnotationSession::TaskState::Failed;
                return;
            }
        }
        if (session.deferShowUntilGenerated && pane->dialog && !pane->dialog->isVisible()) {
            pane->dialog->showMaximized();
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
        auto callback = session.optimizationSucceededCallback;
        if (callback) {
            callback(session);
        }
        return;
    }

    session.taskState = LineAnnotationSession::TaskState::Failed;
    session.error = task.error;
    showError(tr("Lasagna line optimization failed: %1")
                  .arg(QString::fromStdString(task.error)));
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

    _state->setSurface("line-surface", views.lineSurface);
    _state->setSurface("line-side-slice", views.lineSideSlice);
    session.generatedSurfaceNames.push_back("line-surface");
    session.generatedSurfaceNames.push_back("line-side-slice");

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

    LineAnnotationDialog::GeneratedViews generatedViews;
    generatedViews.lineSurfaceName = "line-surface";
    generatedViews.lineSurfaceTitle = tr("Line Surface");
    generatedViews.lineSideSliceName = "line-side-slice";
    generatedViews.lineSideSliceTitle = tr("Line Side Slice");
    generatedViews.linePoints = std::move(linePoints);
    generatedViews.lineUpVectors = views.lineUpVectors;
    generatedViews.seedPoint = seedPoint;
    generatedViews.seedLineIndex = static_cast<int>(session.optimizedLine.points.size() / 2);
    for (const auto& control : session.controlPoints) {
        LineAnnotationDialog::GeneratedOverlay::ControlPointMarker marker;
        marker.point = {static_cast<float>(control.volumePoint[0]),
                        static_cast<float>(control.volumePoint[1]),
                        static_cast<float>(control.volumePoint[2])};
        marker.linePosition = std::isfinite(control.linePosition)
            ? control.linePosition
            : static_cast<double>(control.optimizedIndex);
        marker.isSeed = control.isSeed;
        generatedViews.controlPoints.push_back(marker);
        if (control.isSeed && std::isfinite(marker.linePosition)) {
            generatedViews.seedLineIndex = static_cast<int>(std::llround(marker.linePosition));
        }
    }
    generatedViews.initialCenterIndex = static_cast<int>(std::llround(std::clamp(
        session.focusedLinePosition,
        0.0,
        static_cast<double>(std::max<size_t>(1, session.optimizedLine.points.size()) - 1))));

    generatedViews.currentCutName = "line-current-cut";
    generatedViews.currentCutSurface = std::make_shared<PlaneSurface>(
        seedPoint,
        cv::Vec3f{1.0f, 0.0f, 0.0f});
    _state->setSurface(generatedViews.currentCutName, generatedViews.currentCutSurface);
    session.generatedSurfaceNames.push_back(generatedViews.currentCutName);

    generatedViews.bottomCutSurfaces.reserve(7);
    for (int i = 0; i < 7; ++i) {
        const std::string surfaceName = "line-bottom-cut-" + std::to_string(i);
        auto plane = std::make_shared<PlaneSurface>(seedPoint, cv::Vec3f{1.0f, 0.0f, 0.0f});
        _state->setSurface(surfaceName, plane);
        session.generatedSurfaceNames.push_back(surfaceName);
        generatedViews.bottomCutSurfaces.push_back({surfaceName, std::move(plane)});
    }

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
        {"line-surface", "surface"},
        {"line-side-slice", "side_slice"},
    };

    std::vector<fs::path> savedPaths;
    for (const auto& [surfaceName, stem] : exports) {
        auto surface = std::dynamic_pointer_cast<QuadSurface>(_state->surface(surfaceName));
        if (!surface) {
            continue;
        }

        auto clone = std::make_shared<QuadSurface>(surface->rawPoints().clone(), surface->scale());
        clone->meta = surface->meta;

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
                saveSessionAsFiber(*pane.session);
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
                                        forceFullOptimization,
                                        activeStart,
                                        activeEnd);
    }
    return optimizeLineFromManifest(std::move(manifestPath),
                                    std::move(controlPoints),
                                    std::move(initialLinePoints),
                                    sourceSliceNormal,
                                    directionMode,
                                    forceFullOptimization,
                                    activeStart,
                                    activeEnd);
}

void LineAnnotationController::loadFibersForCurrentPackage()
{
    _fibers.clear();
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

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }
        try {
            if (auto fiber = loadFiberFile(entry.path())) {
                if (fiber->needsSave) {
                    try {
                        fiber->needsSave = false;
                        saveFiber(*fiber);
                    } catch (const std::exception& ex) {
                        fiber->needsSave = true;
                        Logger()->warn("Could not update VC3D fiber metadata {}: {}",
                                       entry.path().string(),
                                       ex.what());
                    }
                }
                _fibers.push_back(std::move(*fiber));
            }
        } catch (const std::exception& ex) {
            Logger()->warn("Skipping invalid VC3D fiber file {}: {}",
                           entry.path().string(),
                           ex.what());
        }
    }
    std::sort(_fibers.begin(), _fibers.end(), [](const StoredFiber& a, const StoredFiber& b) {
        return a.id < b.id;
    });
    emitFiberSummaries();
}

void LineAnnotationController::emitFiberSummaries()
{
    emit fibersChanged(fiberSummaries());
}

fs::path LineAnnotationController::fibersDir() const
{
    if (!_state || !_state->vpkg()) {
        return {};
    }
    const auto vpkg = _state->vpkg();
    const fs::path projectPath = vpkg->path();
    const fs::path root = projectPath.empty()
        ? fs::path(vpkg->getVolpkgDirectory())
        : projectPath.parent_path();
    return root / "fibers";
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

uint64_t LineAnnotationController::nextFiberId() const
{
    uint64_t id = 1;
    for (const auto& fiber : _fibers) {
        id = std::max(id, fiber.id + 1);
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
        if (session.fiberSequence == 0 && session.fiberId != 0) {
            session.fiberSequence = session.fiberId;
        }
        return;
    }

    session.fiberUsername = currentFiberUsername();
    session.fiberStartedAt = currentFiberDateTimeString();
    session.fiberSequence = nextFiberSequenceForUsername(session.fiberUsername);
    session.fiberFileName = vc3d::line_annotation::fiberFileName(
        session.fiberUsername, session.fiberStartedAt, session.fiberSequence);
}

double LineAnnotationController::lineLengthVx(const std::vector<cv::Vec3d>& points)
{
    return vc3d::line_annotation::fiberLineLengthVx(points);
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

void LineAnnotationController::saveSessionAsFiber(LineAnnotationSession& session)
{
    try {
        ensureSessionFiberIdentity(session);
        StoredFiber fiber;
        fiber.id = session.fiberId == 0 ? nextFiberId() : session.fiberId;
        fiber.username = session.fiberUsername;
        fiber.startedAt = session.fiberStartedAt;
        fiber.sequence = session.fiberSequence;
        fiber.fileName = session.fiberFileName;
        auto existingIt = std::find_if(_fibers.begin(),
                                       _fibers.end(),
                                       [&fiber](const StoredFiber& existing) {
                                           return existing.id == fiber.id;
                                       });
        fiber.generation = existingIt == _fibers.end()
            ? uint64_t{1}
            : std::max<uint64_t>(uint64_t{1}, existingIt->generation + 1);
        fiber.controlPoints.reserve(session.controlPoints.size());
        auto controls = session.controlPoints;
        std::stable_sort(controls.begin(),
                         controls.end(),
                         [](const vc::lasagna::LineControlPoint& a,
                            const vc::lasagna::LineControlPoint& b) {
                             return a.linePosition < b.linePosition;
                         });
        for (const auto& control : controls) {
            fiber.controlPoints.push_back(control.volumePoint);
        }
        fiber.linePoints.reserve(session.optimizedLine.points.size());
        for (const auto& point : session.optimizedLine.points) {
            fiber.linePoints.push_back(point.position);
        }
        fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(fiber.controlPoints);
        fiber.manualHvTag = session.fiberManualHvTag;
        saveFiber(fiber);
        session.fiberId = fiber.id;
        session.fiberUsername = fiber.username;
        session.fiberStartedAt = fiber.startedAt;
        session.fiberSequence = fiber.sequence;
        session.fiberFileName = fiber.fileName;
        const uint64_t savedFiberId = fiber.id;
        const uint64_t savedGeneration = fiber.generation;

        auto it = std::find_if(_fibers.begin(), _fibers.end(), [&fiber](const StoredFiber& existing) {
            return existing.id == fiber.id;
        });
        if (it == _fibers.end()) {
            _fibers.push_back(std::move(fiber));
        } else {
            *it = std::move(fiber);
        }
        emitFiberSummaries();
        emit fiberSaved(savedFiberId, savedGeneration);
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber: %1").arg(QString::fromStdString(ex.what())));
    }
}

void LineAnnotationController::saveFiber(const StoredFiber& fiber) const
{
    const fs::path dir = fibersDir();
    if (dir.empty()) {
        throw std::runtime_error("No volume package is loaded");
    }

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create fibers directory " +
                                 dir.string() + ": " + ec.message());
    }

    nlohmann::json root = nlohmann::json::object();
    root["type"] = "vc3d_fiber";
    root["version"] = 1;
    root["id"] = fiber.id;
    root["username"] = fiber.username;
    root["started_at"] = fiber.startedAt;
    root["sequence"] = fiber.sequence;
    root["filename"] = fiber.fileName;
    root["generation"] = fiber.generation;
    root["hv_classification"] = {
        {"z_distance", fiber.hvClassification.zDistance},
        {"control_point_length", fiber.hvClassification.fiberLength},
        {"horizontal_score", fiber.hvClassification.horizontalScore},
        {"vertical_score", fiber.hvClassification.verticalScore},
        {"automatic_tag", vc3d::line_annotation::fiberHvTagToString(fiber.hvClassification.automaticTag)},
        {"automatic_certainty", fiber.hvClassification.automaticCertainty},
        {"manual_tag", fiber.manualHvTag},
    };
    root["control_points"] = nlohmann::json::array();
    root["line_points"] = nlohmann::json::array();
    for (const auto& point : fiber.controlPoints) {
        root["control_points"].push_back(pointToJson(point));
    }
    for (const auto& point : fiber.linePoints) {
        root["line_points"].push_back(pointToJson(point));
    }

    const fs::path finalPath = fiberPath(fiber);
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

std::optional<LineAnnotationController::StoredFiber> LineAnnotationController::loadFiberFile(
    const fs::path& path) const
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

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open fiber file");
    }
    const nlohmann::json root = nlohmann::json::parse(in);
    const std::string type = root.value("type", std::string{});
    if (type != "vc3d_fiber") {
        return std::nullopt;
    }
    if (root.value("version", 0) != 1) {
        throw std::runtime_error("Unsupported vc3d_fiber version");
    }

    StoredFiber fiber;
    fiber.generation = std::max<uint64_t>(uint64_t{1}, root.value("generation", uint64_t{1}));
    if (root.contains("id")) {
        fiber.id = root.at("id").get<uint64_t>();
    } else if (hasLegacyNumericStem) {
        fiber.id = std::stoull(stem);
    } else {
        return std::nullopt;
    }
    fiber.username = vc3d::line_annotation::normalizedFiberUsername(
        root.value("username", std::string{"anon"}));
    fiber.startedAt = root.value("started_at", std::string{});
    if (root.contains("sequence")) {
        fiber.sequence = root.at("sequence").get<uint64_t>();
    } else if (hasLegacyNumericStem) {
        fiber.sequence = fiber.id;
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

void LineAnnotationController::showError(const QString& message) const
{
    if (_parentWidget) {
        QMessageBox::warning(_parentWidget, tr("Line Annotation"), message);
    } else {
        Logger()->warn("Line Annotation: {}", message.toStdString());
    }
}
