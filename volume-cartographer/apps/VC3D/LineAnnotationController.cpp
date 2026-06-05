#include "LineAnnotationController.hpp"

#include "CState.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationDialog.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
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

#include <QFileDialog>
#include <QFutureWatcher>
#include <QMessageBox>
#include <QPointF>
#include <QDateTime>
#include <QSettings>
#include <QStringList>
#include <QtConcurrent/QtConcurrent>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <locale>
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
    uint64_t fiberId = 0;
    std::string fiberUsername;
    std::string fiberStartedAt;
    uint64_t fiberSequence = 0;
    std::string fiberFileName;
    std::string fiberManualHvTag;
    bool suppressFiberSave = false;
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

bool approximatelyEqual(double a, double b)
{
    return std::abs(a - b) <= 1.0e-9;
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
    return viewer->surfName() == "segmentation" &&
           dynamic_cast<QuadSurface*>(surface) != nullptr;
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

void LineAnnotationController::launchSession(LineAnnotationController::SourceKind sourceKind,
                                             const std::string& surfaceName,
                                             std::shared_ptr<Surface> sourceSurface,
                                             const CChunkedVolumeViewer::CameraState& camera,
                                             cv::Vec3d sourceSliceNormal,
                                             std::shared_ptr<LineAnnotationController::LineAnnotationSession> session)
{
    if (!_state || !session) {
        return;
    }

    _state->setSurface(surfaceName, std::move(sourceSurface));
    auto* dialog = new LineAnnotationDialog(_viewerManager, nullptr);
    if (!dialog->addPane(surfaceName, tr("Line Annotation Slice"), camera)) {
        dialog->deleteLater();
        _state->setSurface(surfaceName, nullptr);
        return;
    }
    dialog->showMaximized();
    dialog->raise();
    dialog->activateWindow();

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

std::vector<LineAnnotationController::FiberSummary> LineAnnotationController::fiberSummaries() const
{
    std::vector<FiberSummary> summaries;
    summaries.reserve(_fibers.size());
    for (const auto& fiber : _fibers) {
        summaries.push_back(FiberSummary{
            fiber.id,
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
        if (!materializeGeneratedViews(session)) {
            session.taskState = LineAnnotationSession::TaskState::Failed;
            return;
        }
        const double prefetchPrepMs = session.optimizationReport.normalChunkPrefetchMs +
                                      session.optimizationReport.normalMaterializeMs;
        Logger()->info("Line annotation Lasagna stage timing: event={} prefetch_prep_ms={:.3f} ceres_solve_ms={:.3f} overall_ms={:.3f} points={}",
                       resultEvent,
                       prefetchPrepMs,
                       session.optimizationReport.ceresSolveMs,
                       session.optimizationReport.totalMs,
                       session.optimizedLine.points.size());
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
