#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/FiberJson.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/GeometryEdits.hpp>
#include <vc_object_store/objects/GeometryIngest.hpp>
#include "ObjectStoreCoordinates.hpp"
#include "ObjectStoreEvents.hpp"
#include "ObjectStoreRegions.hpp"
#include "ObjectStoreArtifacts.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include "LineAnnotationController.hpp"
#include <QTemporaryDir>
#include <QSaveFile>
#include "vc/fiber_tracer/FiberDisplay.hpp"

#include "CState.hpp"
#include "FiberNameDisplay.hpp"
#include "FiberSaveBatchTracker.hpp"
#include "OpenDataCoordinateIdentity.hpp"
#include "OpenDataLasagna.hpp"
#include "FiberSliceGeometry.hpp"
#include "FiberMapAutoLink.hpp"
#include "LineAnnotationFiberLinks.hpp"
#include "LineAnnotationStructuralEdits.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "LineAnnotationDialog.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "FiberFollowService.hpp"
#include "ViewerManager.hpp"
#include "overlays/FiberSliceOverlayController.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/AffineTransform.hpp"
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

#include "vc/lasagna/NormalAlignment.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
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
#include <QFileInfo>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QMessageBox>
#include <QScopedValueRollback>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPoint>
#include <QTimer>
#include <QPointF>
#include <QPushButton>
#include <QRadioButton>
#include <QScopeGuard>
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
#include <cstring>
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

#include "LineAnnotationCoordinateScale.hpp"
#include "LineAnnotationOptimizationQueue.hpp"

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
    std::string selectedLasagnaManifestIdentity;
    fs::path selectedManifestPath;
    std::string selectedFiberInferenceDatasetLocation;
    std::string selectedFiberManifestIdentity;
    fs::path selectedFiberInferenceManifestPath;
    // Session geometry is in the open volume's voxels: stored fibers are read
    // through the store's mapping into it (StoredFiber::frame).
    std::optional<std::array<std::size_t, 3>> coordinateBaseShapeZYX;
    double workingToBaseScale = 1.0;
    std::shared_ptr<vc::lasagna::LasagnaDataset> dataset;
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> normalSampler;
    bool inspectionFrameOnly = false;
    bool modelOnly = false;
    // A stored fiber opens view only: the editing signals of its dialog are
    // not connected, and every save and optimization path skips it, until
    // enableSessionEditing(). Opening therefore never re-solves or rewrites it.
    bool readOnly = false;
    // The store is writable and the fiber decodes, so editing can be enabled.
    bool editingAvailable = false;
    std::shared_ptr<vc::lasagna::LasagnaDataset> fiberInferenceDataset;
    std::shared_ptr<vc::fiber_tracer::FiberPredictionField> fiberPredictionField;
    std::string traceNormalDatasetLocation;
    double fiberTraceToBaseScale = 1.0;
    std::shared_ptr<vc::lasagna::LasagnaDataset> traceNormalDataset;
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> traceNormalSampler;
    TaskState taskState = TaskState::Idle;
    // The controller's orientation epoch when the current generated views were
    // materialized. Behind the controller's value, the views were built before
    // the umbilicus or active volume last changed and are rebuilt by the next
    // refreshStaleGeneratedViews() pass.
    int orientationEpoch = -1;
    cv::Vec3d seedPoint{0.0, 0.0, 0.0};
    std::string sourceAnnotationSurfaceName;
    vc::lasagna::LineOptimizationReport optimizationReport;
    vc::lasagna::LineModel optimizedLine;
    // These vectors are intentionally private to this translation unit. Branch
    // refs store live control-point indices plus reciprocal metadata in linked
    // fibers; after mutating either vector, call
    // syncLinkedBranchMetadataAfterFiberModification() before refreshing overlays
    // or saving.
    std::vector<vc3d::line_annotation::LineControlPoint> controlPoints;
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
    // Cooperative cancellation flag of the solve currently in flight (shared
    // with the worker); replaced per launch. Cancellation means INVALIDATION
    // only — teardown, package switch, pane close, quit-save supersede.
    // Edits never set it: the solve runs to completion and its result is
    // span-merged at landing (render-job model).
    std::shared_ptr<std::atomic<bool>> runningSolveCancel;
    // Solve lifecycle: single-flight, epoch-refused publication after a
    // session mutation, coalesced pending dirty spans. Pure decision object;
    // the controller drives it at edit, launch, and finish.
    vc3d::line_annotation::OptimizationCoalescingQueue solveQueue;
    // The queue epoch the in-flight solve was started in.
    uint64_t runningSolveEpoch = 0;
    // The in-flight solve covers the whole line (retrace-all or no dirty
    // set). A superseded whole-line solve must be re-requested as whole-line,
    // or a mode switch's retrace would silently shrink to the edit's spans.
    bool runningSolveFullLine = false;
    // The dirty spans the in-flight solve was launched over, kept in the
    // CURRENT control numbering (edits that renumber controls remap this
    // alongside the queue's pending set). A superseded solve re-adds them to
    // the pending union - discarding a solve must never silently drop the
    // spans it was going to fix, or their provisional geometry survives
    // until a save forces a full pass.
    std::vector<size_t> runningSolveDirtySegments;
    // Merge bookkeeping for publishing a superseded solve (render-job model:
    // edits no longer cancel the in-flight solve; its result is span-merged
    // when it lands). Maintained ONLY by handleGeneratedControlPoint; every
    // other mutation path resets the map, which downgrades the landing to
    // the discard-and-redispatch fallback.
    // - runningSolveControlMap: solve-start control index -> current index,
    //   composed with each edit's oldToNewIndices.
    // - runningSolveEditedSpans: current-numbering spans touched by edits
    //   since the solve started (independent of the queue's fullLine
    //   escalation, so multi-collapses do not disable merging).
    // - runningSolveConfigChanged: a solver-input change (extrapolation
    //   distance) landed mid-solve; the merge then keeps the current tails.
    std::optional<std::vector<size_t>> runningSolveControlMap;
    std::vector<size_t> runningSolveEditedSpans;
    bool runningSolveConfigChanged = false;
    // An armed rollback snapshot (controlPointCollapseRollback /
    // controlPointsBeforeModeChange) no longer describes an optimized state:
    // a tag edit mirrored from another pane changed its tail policy or goals,
    // or a tail distance changed. A restore then lands Unoptimized.
    bool rollbackInvalidatesOptimization = false;
    // A debounced dispatch of the queue's pending solve is scheduled.
    bool solveDispatchScheduled = false;
    // Reentrancy guard for immediate interaction publication.
    bool autoSaveScheduled = false;
    // Bumped on every optimizedLine replacement; part of the side-strip
    // query fingerprint so a landed solve (or rollback) that happens to
    // reuse the previous vector's allocation is still seen as a change.
    uint64_t lineRevision = 0;
    // A landing's epilogue holds the queue's pending request in a local
    // while it publishes and then re-dispatches it: in that window the
    // queue reads empty although work is still owed. Reported as pending.
    bool detachedPendingWork = false;
    bool deferShowUntilGenerated = false;
    uint64_t fiberId = 0;
    std::string fiberUuid = vc::objects::util::newUuid();
    std::optional<vc::objects::PinnedObject> storePin;
    std::optional<vc::objects::PinnedObject> runningStorePin;
    // A hosted replica keeps the fiber leased while it is open for editing.
    vc::objects::LeaseHold editingLeases;
    std::string fiberUsername;
    std::string fiberStartedAt;
    uint64_t fiberSequence = 0;
    std::string fiberFileName;
    std::string fiberManualHvTag;
    std::vector<std::string> fiberTags;
    double fiberWidth = 0.0;
    double fiberWidthGapFraction = vc::fiber_tracer::kDefaultFiberWidthGapFraction;
    vc::fiber_tracer::FiberDisplayField displayField;
    vc3d::line_annotation::FiberOptimizationMode fiberOptimizationMode =
        vc3d::line_annotation::kDefaultNewFiberOptimizationMode;
    vc3d::line_annotation::FiberOptimizationMode fiberOptimizationModeBeforeTask =
        vc3d::line_annotation::kDefaultNewFiberOptimizationMode;
    bool restoreFiberOptimizationModeOnFailure = false;
    // Set when a toolbar interpolation-mode switch successfully re-optimized
    // the whole line; the next saveSessionAsFiber drops the 'reviewed' tag.
    bool stripReviewedTagOnSave = false;
    bool nativeSeedTracePending = false;
    std::optional<std::vector<vc3d::line_annotation::LineControlPoint>>
        controlPointsBeforeModeChange;
    // Paired with controlPointsBeforeModeChange: applyOptimizationTaskResult
    // installs the new line and re-anchors branches before its last failure
    // exit (generated-view materialization), so a failed whole-line task must
    // restore these too or a later save writes the rolled-back control points
    // against the new line — a pair no single optimizer run produced.
    // Peer-fiber sync is deferred until past that failure exit instead of
    // being rolled back (see the sync call in applyOptimizationTaskResult).
    std::optional<vc::lasagna::LineModel> optimizedLineBeforeModeChange;
    std::optional<std::vector<LineAnnotationController::FiberBranchRef>>
        branchesBeforeModeChange;
    struct ControlPointCollapseRollback {
        std::vector<vc3d::line_annotation::LineControlPoint> controlPoints;
        vc::lasagna::LineModel optimizedLine;
        vc::lasagna::LineOptimizationReport optimizationReport;
        std::vector<LineAnnotationController::FiberBranchRef> branches;
        fs::path selectedManifestPath;
        cv::Vec3d seedPoint{0.0, 0.0, 0.0};
        double focusedLinePosition = 0.0;
        std::optional<cv::Vec3d> focusedControlPoint;
        bool lineWasOptimized = false;
        bool fiberMetricsMatchStoredFiber = false;
        SessionOptimizationState optimizationState =
            SessionOptimizationState::Unoptimized;
    };
    std::optional<ControlPointCollapseRollback> controlPointCollapseRollback;
    std::optional<fs::path> atlasDir;
    fs::path atlasFiberPath;
    vc::atlas::AtlasPredSnapSet predSnapSet;
    bool suppressFiberSave = false;
    // The fiber this session edits was deleted through the app while the
    // session stayed open. Unlike suppressFiberSave - which is also set
    // transiently (a headless save, the merge/split retirement window) -
    // this never clears: the structural-edit handlers refuse the session,
    // and the session save and the linked-peer snapshot loop skip it.
    bool fiberDeleted = false;
    bool suppressGeneratedViews = false;
    bool suppressErrorDialogs = false;
    // True once the optimizer actually ran in this session (async or the
    // synchronous save-time finalize). Gates the needs_reoptimization tag
    // strip on save: a session whose line is still the loaded placeholder
    // must keep the tag.
    bool lineWasOptimized = false;
    LineAnnotationController::SessionOptimizationState optimizationState =
        LineAnnotationController::SessionOptimizationState::Unoptimized;
    LineAnnotationController::SessionOptimizationState pendingOptimizationState =
        LineAnnotationController::SessionOptimizationState::Optimized;
    LineAnnotationController::SessionOptimizationState optimizationStateBeforeTask =
        LineAnnotationController::SessionOptimizationState::Unoptimized;
    std::optional<std::pair<double, double>> initialStripLinePositionRange;
    bool initialFitWholeLine = false;
    bool disableInitialGeneratedHoverFollow = false;
    std::function<void(LineAnnotationSession&)> optimizationSucceededCallback;
    bool fiberMetricsMatchStoredFiber = false;
};

struct LineAnnotationController::FiberMetricsTaskResult {
    bool ok = false;
    bool suppressErrorDialogs = false;
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
        vc::lasagna::LineStripPositionMap stripPositionMap;
    };

    QPointer<QMdiArea> targetArea;
    bool suppressErrorDialogs = false;
    vc::atlas::FiberIntersectionResult result;
    std::optional<fs::path> atlasDir;
    double sourceFocusLinePosition = 0.0;
    double targetFocusLinePosition = 0.0;
    std::vector<std::string> surfaceNames;
    // The cross-section and strip viewers whose generated overlays are drawn
    // from the side sessions' controls (via connectOverlaysUpdated), so an
    // edit made to those controls from elsewhere can ask them to redraw.
    std::vector<QPointer<CChunkedVolumeViewer>> generatedOverlayViewers;
    std::shared_ptr<LineAnnotationSession> sourceLineSession;
    std::shared_ptr<LineAnnotationSession> targetLineSession;
    std::string sourceSessionSurfaceName;
    std::string targetSessionSurfaceName;
    FollowSlice sourceFollow;
    FollowSlice targetFollow;
    QPointer<QShortcut> followShortcut;
    std::optional<bool> activeFollowSourceSide;
    std::map<std::string, GeneratedSurfaceContext> generatedSurfaceContexts;
    // The controller's orientation epoch when these strips were built. The
    // inspection's sessions suppress ordinary generated views, so the pane walk
    // in refreshStaleGeneratedViews() never reaches them; this is how the
    // inspection itself is noticed as stale instead.
    int orientationEpoch = -1;
};

namespace {

// A fiber's sampled line in the frame its controls were read in: the open
// volume's voxels through the store's view, else stored coordinates.
std::vector<vc::geometry::Point> storedFiberLinePoints(
    const vc::objects::FiberView& view, const std::optional<vc3d::line_annotation::FiberVolumeFrame>& frame)
{
    if (!frame) return view.linePoints(0, view.linePointCount());
    const auto shown = view.viewTransformedAt(frame->volume);
    if (shown.mapping() != frame->storedToVolume)
        throw std::runtime_error("The fiber's volume registration changed; reopen it");
    return shown.points("line_points", 0, view.linePointCount());
}

// Link predicates shared with the save/validation helpers and the
// structural-edit planners (LineAnnotationFiberLinks.hpp).
namespace {
// Axis-aligned box (minx,miny,minz,maxx,maxy,maxz) around finite points,
// grown by margin on every side.
std::array<double, 6> boxAroundPoints(const std::vector<cv::Vec3d>& points, double margin)
{
    std::array<double, 6> box{INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (const auto& point : points) {
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) continue;
        for (int axis = 0; axis < 3; ++axis) {
            box[axis] = std::min(box[axis], point[axis]);
            box[axis + 3] = std::max(box[axis + 3], point[axis]);
        }
    }
    if (!(box[0] <= box[3])) return {0, 0, 0, -1, -1, -1};
    for (int axis = 0; axis < 3; ++axis) {
        box[axis] -= margin;
        box[axis + 3] += margin;
    }
    return box;
}
}  // namespace

using vc3d::line_annotation::branchDirectionsCompatible;
using vc3d::line_annotation::branchReferencesFiber;
using vc3d::line_annotation::endpointTangentFromLinePoints;
using vc3d::line_annotation::fiberErrorName;
using vc3d::line_annotation::finiteDirection;
using vc3d::line_annotation::finitePoint;
using vc3d::line_annotation::matchingStoredControlPointIndex;
using vc3d::line_annotation::normalizedOrZero;
using vc3d::line_annotation::pointsApproximatelyEqual;
using vc3d::line_annotation::nearestControlAtPoint;
using vc3d::line_annotation::storedControlPointIndexByPosition;
using vc3d::line_annotation::tangentAtLinePosition;
using vc3d::coordinateBindingForVolume;
using vc3d::line_annotation::kNeedsReoptimizationTag;

// Arclength radius within which a placement click replaces the existing
// control(s) instead of adding one. Tied to the strip's along-line sampling
// so no control span can be created shorter than one strip column (#1484:
// such a span would be drawn stretched to a full column).
constexpr double kMinimumControlPointSpacingBaseVoxels =
    vc::lasagna::kLineViewAlongSamplingDistanceBaseVoxels;

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

constexpr double kLineSegmentLength = 32.0;
// Quiet window between a control-point edit and the solve it dispatches;
// rapid placement coalesces into one solve over the union of dirty spans.
constexpr int kSolveDispatchDebounceMs = 150;
// Quiet window coalescing the several overlay-rebuild-triggered side-strip
// intersection requests of one placement into a single dispatch.
constexpr int kSideStripQueryDebounceMs = 150;
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
};

QString fiberDisplayNameFromFileName(const std::string& fileName)
{
    const QString stem = vc3d::displayStemForFiberFile(QString::fromStdString(fileName));
    return stem.isEmpty() ? QObject::tr("unsaved fiber") : stem;
}

std::string datasetEntryMenuLabel(const vc::project::Entry& entry)
{
    QString label;
    if (!vc::project::isLocationRemote(entry.location)) {
        label = QFileInfo(QString::fromStdString(entry.location)).fileName();
    }
    if (label.isEmpty()) {
        label = QString::fromStdString(entry.location);
    }
    for (const auto& tag : entry.tags) {
        if (tag.rfind("model:", 0) == 0 ||
            tag.rfind("source:", 0) == 0 ||
            tag.rfind("name:", 0) == 0) {
            label += QStringLiteral(" (%1)").arg(QString::fromStdString(tag));
            break;
        }
    }
    return label.toStdString();
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

    form->addRow(QObject::tr("Path:"), pathRow);
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

        dialog.accept();
    });

    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }

    return FiberJsonPathOptions{fs::path(pathEdit->text().trimmed().toStdString())};
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

    const auto text = root.dump(2) + '\n';
    QSaveFile output(QString::fromStdString(finalPath.string()));
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(text.data(),text.size()) != qint64(text.size()) || !output.commit())
        throw std::runtime_error("Failed to export " + finalPath.string());
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

// Hash of one fiber's query-relevant geometry, memoized alongside its
// snapshot so an unchanged fiber is never re-hashed point by point.
uint64_t sideStripFiberGeometryHash(const vc::atlas::FiberPolyline& fiber)
{
    uint64_t seed = 0x4649424552484153ULL;
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
    return seed;
}

uint64_t sideStripRequestCacheKey(
    const cv::Mat_<cv::Vec3f>& stripPoints,
    const std::vector<LineAnnotationController::SideStripFiberSnapshot>& fibers,
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
        combineSideStripHash(seed, fiber.geometryHash);
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

std::optional<int> sessionControlPointIndexByPosition(
    const std::vector<vc3d::line_annotation::LineControlPoint>& controlPoints,
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
    const std::vector<vc3d::line_annotation::LineControlPoint>& controlPoints)
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
    const std::vector<vc3d::line_annotation::LineControlPoint>& controlPoints,
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
    const std::vector<vc3d::line_annotation::LineControlPoint>& controlPoints,
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

void addUniqueFiberId(std::vector<uint64_t>& ids, uint64_t fiberId)
{
    if (fiberId == 0 ||
        std::find(ids.begin(), ids.end(), fiberId) != ids.end()) {
        return;
    }
    ids.push_back(fiberId);
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
    return vc3d::line_annotation::sameFiberIdentity(lhs.branchFiberId, rhs.branchFiberId) &&
           lhs.branchControlPointIndex == rhs.branchControlPointIndex;
}

bool deduplicateBranchLinks(
    std::vector<LineAnnotationController::FiberBranchRef>& branches,
    uint64_t targetFiberId = 0,
    const std::string& targetFileName = {})
{
    const bool targetScoped = targetFiberId != 0 || !targetFileName.empty();
    std::vector<LineAnnotationController::FiberBranchRef> deduplicated;
    deduplicated.reserve(branches.size());
    for (auto& branch : branches) {
        if (targetScoped &&
            !branchReferencesFiber(branch, targetFiberId)) {
            deduplicated.push_back(std::move(branch));
            continue;
        }
        const auto duplicate = std::find_if(
            deduplicated.begin(), deduplicated.end(), [&branch](const auto& candidate) {
                return candidate.controlPointIndex == branch.controlPointIndex &&
                       branchLinkedEndpointMatches(candidate, branch);
            });
        if (duplicate == deduplicated.end()) {
            deduplicated.push_back(std::move(branch));
        } else {
            // Same OR as the fiber map's dedup: an adjacent ref never quietly
            // becomes an ordinary one because an ordinary duplicate came first.
            const bool adjacent = duplicate->adjacent || branch.adjacent;
            const auto combined = vc3d::line_annotation::combineReviewStates(
                duplicate->pending, duplicate->autoLink, branch.pending, branch.autoLink, adjacent);
            duplicate->pending = combined.pending;
            duplicate->adjacent = adjacent;
            duplicate->autoLink = combined.autoLink;
            vc3d::line_annotation::normalizeBranchRefReviewState(*duplicate);
        }
    }
    const bool changed = deduplicated.size() != branches.size();
    branches = std::move(deduplicated);
    return changed;
}

template <typename ControlPoint>
std::optional<size_t> controlPointIndexAtLinePosition(
    const std::vector<ControlPoint>& controls,
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

std::vector<size_t> controlPointOrder(
    const std::vector<vc3d::line_annotation::LineControlPoint>& controls)
{
    std::vector<size_t> order(controls.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&controls](size_t lhs, size_t rhs) {
        return controls[lhs].linePosition < controls[rhs].linePosition;
    });
    return order;
}

std::vector<std::pair<int, int>> protectedLinePointRanges(
    const std::vector<vc3d::line_annotation::LineControlPoint>& controls,
    int linePointCount,
    std::optional<size_t> excludedOwner = std::nullopt)
{
    std::vector<std::pair<int, int>> ranges;
    if (linePointCount < 2) {
        return ranges;
    }
    const auto order = controlPointOrder(controls);
    for (size_t sortedIndex = 0; sortedIndex + 1 < order.size(); ++sortedIndex) {
        const size_t owner = order[sortedIndex];
        if (!vc3d::line_annotation::isAcceptedNativeTrace(
                controls[owner].segmentToNext) ||
            owner == excludedOwner) {
            continue;
        }
        int first = controlLineIndex(controls[owner], linePointCount - 1);
        int second = controlLineIndex(controls[order[sortedIndex + 1]], linePointCount - 1);
        if (second < first) {
            std::swap(first, second);
        }
        ranges.emplace_back(first, second);
    }
    return ranges;
}

std::vector<std::pair<int, int>> protectedControlPointSpans(
    const std::vector<vc3d::line_annotation::LineControlPoint>& controls)
{
    std::vector<std::pair<int, int>> spans;
    const auto order = controlPointOrder(controls);
    for (size_t sortedIndex = 0; sortedIndex + 1 < order.size(); ++sortedIndex) {
        const size_t owner = order[sortedIndex];
        if (vc3d::line_annotation::isAcceptedNativeTrace(
                controls[owner].segmentToNext)) {
            spans.emplace_back(static_cast<int>(owner),
                               static_cast<int>(order[sortedIndex + 1]));
        }
    }
    return spans;
}

template <typename ControlPoint>
QString controlPointInfoText(const std::vector<ControlPoint>& controls,
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

template <typename ControlPoint>
QLabel* createControlPointInfoLabel(
    QWidget* parent,
    const std::vector<ControlPoint>& controls,
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

template <typename ControlPoint>
void updateControlPointInfoLabel(QLabel* label,
                                 const std::vector<ControlPoint>& controls,
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
                                                         double linePosition,
                                                         const vc::lasagna::LineStripPositionMap* positionMap)
{
    if (!surface || !std::isfinite(linePosition)) {
        return std::nullopt;
    }
    const auto* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }
    const double gridColumn = positionMap && positionMap->valid()
        ? positionMap->originalPositionToStripGridColumn(linePosition)
        : linePosition;
    if (!std::isfinite(gridColumn)) {
        return std::nullopt;
    }
    const cv::Vec2d surfacePoint = surface->gridToSurface(
        {gridColumn, static_cast<double>(points->rows / 2)});
    return cv::Vec2f{static_cast<float>(surfacePoint[0]),
                     static_cast<float>(surfacePoint[1])};
}

void frameStripLineSpan(CChunkedVolumeViewer* viewer,
                        QuadSurface* surface,
                        double firstLinePosition,
                        double secondLinePosition,
                        const vc::lasagna::LineStripPositionMap* positionMap)
{
    if (!viewer || !surface ||
        !std::isfinite(firstLinePosition) ||
        !std::isfinite(secondLinePosition)) {
        return;
    }
    const auto first = stripLinePositionToSurfacePoint(surface, firstLinePosition, positionMap);
    const auto second = stripLinePositionToSurfacePoint(surface, secondLinePosition, positionMap);
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
    const std::vector<vc3d::line_annotation::LineControlPoint>& controls,
    const std::vector<LineAnnotationController::FiberBranchRef>& branches = {})
{
    struct BranchLinkTarget {
        uint64_t fiberId = 0;
        int controlPointIndex = -1;
        bool pending = false;
        bool adjacent = false;
        bool autoLink = false;
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
                               branch.pending,
                               branch.adjacent,
                               branch.autoLink && branch.pending && !branch.adjacent});
        } else {
            const bool adjacent = duplicate->adjacent || branch.adjacent;
            const auto combined = vc3d::line_annotation::combineReviewStates(
                duplicate->pending, duplicate->autoLink, branch.pending,
                branch.autoLink && branch.pending && !branch.adjacent, adjacent);
            duplicate->pending = combined.pending;
            duplicate->adjacent = adjacent;
            duplicate->autoLink = combined.autoLink;
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
        marker.identity = control.identity;
        marker.direction = control.direction;
        marker.isSeed = control.isSeed;
        marker.isKollesisTermination = vc3d::line_annotation::hasControlPointTag(
            control.tags, vc3d::line_annotation::kKollesisTerminationTag);
        marker.isBreak = vc3d::line_annotation::hasControlPointTag(
            control.tags, vc3d::line_annotation::kBreakTag);
        marker.hasGapToNext = vc3d::line_annotation::spanIsGap(control.segmentToNext);
        marker.hasDamagedToNext = vc3d::line_annotation::spanIsDamaged(control.segmentToNext);
        marker.hasTracedSegmentToNext =
            vc3d::line_annotation::isAcceptedNativeTrace(control.segmentToNext);
        if (control.segmentToNext) {
            marker.interpolationGoal =
                vc3d::line_annotation::segmentInterpolationGoalToString(
                    control.segmentToNext->interpGoal);
            marker.interpolationModeMarker =
                vc3d::line_annotation::segmentInterpolationModeMarker(
                    control.segmentToNext->interpMode);
        }
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
                marker.branchLinks.push_back({target.fiberId,
                                              target.controlPointIndex,
                                              target.pending,
                                              target.adjacent,
                                              target.autoLink});
                marker.hasPendingLinks = marker.hasPendingLinks || target.pending;
                marker.hasAdjacentLinks = marker.hasAdjacentLinks || target.adjacent;
            }
            marker.hasAutoPendingLinks =
                vc3d::line_annotation::autoPendingMarker(marker.branchLinks);
            marker.branchIds.erase(std::unique(marker.branchIds.begin(), marker.branchIds.end()),
                                   marker.branchIds.end());
            marker.hasBranches = !marker.branchLinks.empty();
        }
        markers.push_back(std::move(marker));
    }
    return markers;
}

void remapBranchControlPointIndices(
    const std::vector<vc3d::line_annotation::LineControlPoint>& oldControls,
    const std::vector<vc3d::line_annotation::LineControlPoint>& newControls,
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

void remapCollapsedBranchControlPointIndices(
    const std::vector<size_t>& oldToNewIndices,
    std::vector<LineAnnotationController::FiberBranchRef>& branches)
{
    for (auto& branch : branches) {
        if (branch.controlPointIndex < 0 ||
            static_cast<size_t>(branch.controlPointIndex) >= oldToNewIndices.size()) {
            branch.controlPointIndex = -1;
            continue;
        }
        branch.controlPointIndex = static_cast<int>(
            oldToNewIndices[static_cast<size_t>(branch.controlPointIndex)]);
    }
    branches.erase(
        std::remove_if(branches.begin(), branches.end(), [](const auto& branch) {
            return branch.controlPointIndex < 0;
        }),
        branches.end());

    deduplicateBranchLinks(branches);
}

std::vector<vc3d::line_annotation::GeneratedOverlay::PredSnapMarker>
generatedPredSnapMarkers(const std::vector<vc3d::line_annotation::LineControlPoint>& controls,
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

nlohmann::json controlsToJson(
    const std::vector<vc3d::line_annotation::LineControlPoint>& controls)
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
                        const std::vector<vc3d::line_annotation::LineControlPoint>& controls,
                        const vc::lasagna::LineModel& line,
                        const vc::lasagna::LineOptimizationReport* report = nullptr)
{
    // Gate before any serialization: with the debug dir unset (the default),
    // this must cost nothing - the callers sit on the interactive edit path.
    const char* debugDir = std::getenv("VC3D_LINE_DEBUG_DIR");
    if (!debugDir || *debugDir == '\0') {
        return;
    }
    const nlohmann::json linePoints = linePointsToJson(line);

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
    std::vector<vc3d::line_annotation::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    int initialCenterlineLengthVx,
    bool forceFullOptimization,
    int activeStart,
    int activeEnd,
    const vc::lasagna::NormalSampler& sampler,
    double rayTailDistanceBaseVoxels,
    const std::atomic<bool>* cancelFlag = nullptr);

// rayTailDistanceBaseVoxels: the straight-ray length for open tails past a
// lone break or a kollesis end (OpenTailPolicy), applied to the result.
LineAnnotationController::OptimizationTaskResult optimizeLineFromManifest(
    fs::path manifestPath,
    double workingToBaseScale,
    std::vector<vc3d::line_annotation::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    int initialCenterlineLengthVx,
    bool forceFullOptimization,
    int activeStart,
    int activeEnd,
    double rayTailDistanceBaseVoxels,
    const std::atomic<bool>* cancelFlag = nullptr)
{
    vc::lasagna::LasagnaDataset dataset =
        vc::lasagna::LasagnaDataset::open(
            manifestPath, {workingToBaseScale});
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
                                   sampler,
                                   rayTailDistanceBaseVoxels,
                                   cancelFlag);
}

LineAnnotationController::OptimizationTaskResult optimizeLineWithSampler(
    fs::path manifestPath,
    std::vector<vc3d::line_annotation::LineControlPoint> controlPoints,
    std::vector<cv::Vec3d> initialLinePoints,
    cv::Vec3d sourceSliceNormal,
    LineAnnotationController::InitialDirectionMode directionMode,
    int initialCenterlineLengthVx,
    bool forceFullOptimization,
    int activeStart,
    int activeEnd,
    const vc::lasagna::NormalSampler& sampler,
    double rayTailDistanceBaseVoxels,
    const std::atomic<bool>* cancelFlag)
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
        config.cancelFlag = cancelFlag;
        const auto throwIfCancelled = [cancelFlag] {
            if (cancelFlag && cancelFlag->load(std::memory_order_relaxed)) {
                throw vc::lasagna::LineOptimizationCancelled();
            }
        };
        throwIfCancelled();
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
        // The optimizer's own control index map, where a branch reports one
        // (the reinit does); otherwise the outer vertices are resolved on
        // the result with a direction-aware nearest scan.
        std::optional<std::vector<int>> optimizerControlIndices;
        bool optimizerIndicesInLineOrder = false;
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
                                                              vc3d::line_annotation::optimizerControlPoints(
                                                                  task.controlPoints),
                                                              std::move(fixedIndices),
                                                              displayFrameAnchorIndex,
                                                              config,
                                                              protectedControlPointSpans(
                                                                  task.controlPoints));
            if (reinitialized.failed) {
                task.ok = false;
                task.error = reinitialized.failureReason;
                return task;
            }
            if (reinitialized.fixedPointIndices.size() == task.controlPoints.size()) {
                // Reported in LINE order (the reinit sorts its controls);
                // mapped back to the task's control slots below.
                optimizerControlIndices = reinitialized.fixedPointIndices;
                optimizerIndicesInLineOrder = true;
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
            const int initialPointCount = static_cast<int>(initialLinePoints.size());
            // The existing-line optimizer keeps these indices (the control
            // positions were written onto the line at them above).
            if (fixedIndices.size() == task.controlPoints.size()) {
                optimizerControlIndices = fixedIndices;
            }
            task.result = optimizer.optimizeExistingLine(std::move(initialLinePoints),
                                                         std::move(fixedIndices),
                                                         displayFrameAnchorIndex,
                                                         config,
                                                         activeStart,
                                                         activeEnd,
                                                         candidateName,
                                                         protectedLinePointRanges(
                                                             task.controlPoints,
                                                             initialPointCount));
        } else {
            task.result = optimizer.optimizeFromControlPoints(
                vc3d::line_annotation::optimizerControlPoints(task.controlPoints), config);
        }
        // Open tails past a lone break or a kollesis end are straight rays
        // whatever produced the line. The controls' vertices come from the
        // optimizer's index map when it reports one, else from a nearest
        // scan that resolves the head to the FIRST and the tail to the LAST
        // candidate (a line that revisits its own coordinates must not lose
        // controlled vertices to the wrong visit); the landing re-resolves
        // every control the same way.
        const auto tailPolicy =
            vc3d::line_annotation::openTailPolicyForControls(task.controlPoints);
        if (tailPolicy.any() && task.result.line.points.size() >= 2) {
            throwIfCancelled();
            std::vector<size_t> order(task.controlPoints.size());
            std::iota(order.begin(), order.end(), size_t{0});
            std::stable_sort(order.begin(), order.end(), [&task](size_t a, size_t b) {
                return task.controlPoints[a].linePosition < task.controlPoints[b].linePosition;
            });
            std::vector<cv::Vec3d> resultPoints;
            resultPoints.reserve(task.result.line.points.size());
            for (const auto& point : task.result.line.points) {
                resultPoints.push_back(point.position);
            }
            std::vector<int> resolved(task.controlPoints.size(), 0);
            if (optimizerControlIndices && optimizerIndicesInLineOrder) {
                for (size_t rank = 0; rank < order.size(); ++rank) {
                    resolved[order[rank]] = (*optimizerControlIndices)[rank];
                }
            } else if (optimizerControlIndices) {
                resolved = *optimizerControlIndices;
            } else {
                // Only the seed branch (one fresh control): the optimizer
                // anchors its line at the seed, authoritative when that
                // index names the seed vertex; else the first match.
                const int anchor = task.result.line.displayFrameAnchorIndex;
                for (size_t rank = 0; rank < order.size(); ++rank) {
                    const size_t i = order[rank];
                    const bool anchorIsControl = order.size() == 1 && anchor >= 0 &&
                        anchor < static_cast<int>(resultPoints.size()) &&
                        pointsApproximatelyEqual(resultPoints[static_cast<size_t>(anchor)],
                                                 task.controlPoints[i].volumePoint);
                    resolved[i] = anchorIsControl
                        ? anchor
                        : static_cast<int>(vc3d::line_annotation::nearestLineIndex(
                              resultPoints, task.controlPoints[i].volumePoint,
                              order.size() > 1 && rank + 1 == order.size()
                                  ? vc3d::line_annotation::NearestTie::Last
                                  : vc3d::line_annotation::NearestTie::First));
                }
            }
            const auto& head = task.controlPoints[order.front()];
            const auto& tail = task.controlPoints[order.back()];
            const int shift = vc3d::line_annotation::replaceRayTails(
                task.result.line,
                resolved[order.front()],
                resolved[order.back()],
                tailPolicy,
                head.direction,
                tail.direction,
                rayTailDistanceBaseVoxels,
                std::max(0.1, config.segmentLength),
                &sampler);
            for (size_t i = 0; i < task.controlPoints.size(); ++i) {
                task.controlPoints[i].optimizedIndex = resolved[i] + shift;
                task.controlPoints[i].linePosition = static_cast<double>(resolved[i] + shift);
            }
            task.result.report.message += "\nray_extrapolations=" +
                std::to_string(int(tailPolicy.headRay) + int(tailPolicy.tailRay));
        }
        // Unconditional: the optimizer's own post-processing (final normal
        // sampling, loss evaluation) has no checkpoint of its own.
        throwIfCancelled();
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

// The persisted ray length for paths without a dialog (headless solves, the
// legacy wrappers); the dialog's applied value wins when there is one.
int breakExtrapolationDistanceVxFromSettings()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return vc3d::settings::line_annotation::breakExtrapolationDistanceVx(settings);
}

void applyFocusBoundsToOptimizationTask(
    LineAnnotationController::OptimizationTaskResult& task)
{
    if (!task.ok || !task.focusBoundsBase || task.focusBoundsApplied) {
        return;
    }
    try {
        vc3d::line_annotation::constrainLineOpenTailsToBounds(
            task.result.line, task.controlPoints, *task.focusBoundsBase);
        task.focusBoundsApplied = true;
    } catch (const std::exception& ex) {
        task.ok = false;
        task.error = std::string("Could not apply focus bounding box: ") + ex.what();
    }
}

} // namespace

std::vector<vc3d::line_annotation::GeneratedOverlay::BranchLinkMarker>
LineAnnotationController::generatedBranchLinkMarkers(const std::vector<LineAnnotationController::FiberBranchRef>& branches) const
{
    std::vector<vc3d::line_annotation::GeneratedOverlay::BranchLinkMarker> markers;
    markers.reserve(branches.size());
    for (const auto& branch : branches) {
        // A peer outside the volume has its position in another frame.
        if (!fiberBoundToVolume(branch.branchFiberId, _state ? _state->currentVolume() : nullptr)) continue;
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
{
    _lineSolvePool.setMaxThreadCount(2);
    // Background work must not compete with interactive rendering at normal
    // OS priority (review guidance: run it low, publish late results).
    _lineSolvePool.setThreadPriority(QThread::LowPriority);
    if (_state) {
        connect(_state, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
            // Hub status alone changes no fiber revisions or geometry. Actual
            // pulled edits are observed separately as Committed events.
            if (event.kind == vc::objects::StoreSessionEvent::Kind::Hub) return;
            if (event.kind == vc::objects::StoreSessionEvent::Kind::Attached || event.kind == vc::objects::StoreSessionEvent::Kind::Detached) {
                onVolumePackageChanged({});
                return;
            }
            if (event.attachment != _state->objectStoreAttachment() ||
                !vc3d::store::touches(event, {"fiber", "fiber_branch", vc::objects::kAttachmentEdgeKind, "volume_transform", "volume_transform_choice"})) return;
            ++_fiberDataGeneration;
            _sideStripStoredSnapshotCache.clear();
            refreshChangedStoreFibers();
            emit fibersChanged();
        });
        connect(_state,
                &CState::surfaceChanged,
                this,
                &LineAnnotationController::onSurfaceChanged);
        connect(_state,
                &CState::vpkgChanged,
                this,
                &LineAnnotationController::onVolumePackageChanged);
        // Both handlers only bump counters and queue one coalesced refresh; the
        // actual reload happens at the cache choke point when a stale view is
        // rebuilt, so neither handler touches the filesystem itself.
        connect(_state,
                &CState::volumeChanged,
                this,
                [this](const std::shared_ptr<Volume>&,
                       const std::string& volumeId) {
                    // CState re-emits for a reselection of the current volume
                    // (package UI refreshes do this); only an actual switch
                    // changes what views were built from. Guarded here rather
                    // than in CState, whose other consumers may rely on the
                    // re-emission. Cleared on package change, since two
                    // projects can name a volume identically.
                    if (volumeId == _lastVolumeChangedId) {
                        return;
                    }
                    _lastVolumeChangedId = volumeId;
                    onActiveVolumeChanged();
                });
        // Without this an attach or detach would not reach normal orientation
        // until the project was reopened, since the resolved umbilicus is cached.
        // Dropping the cache alone is not enough either: nothing re-applies
        // normals to surfaces already built, so the attach would report success
        // while every open strip kept its old orientation — told it worked, and
        // nothing visibly changes.
        connect(_state, &CState::umbilicusChanged, this, [this]() {
            ++_umbilicusGeneration;
            ++_orientationEpoch;
            invalidateScrollUmbilicus();
            publishUmbilicusNotice();
            scheduleStaleViewRefresh();
        });
    }
}

LineAnnotationController::~LineAnnotationController()
{
    flushAllPendingSessionAutoSaves();
    waitForFiberSaves();
    // Bounded teardown of in-flight line solves: request cooperative
    // cancellation on every session, then drain the private pool. The Ceres
    // iteration callback honors the flag within one iteration; the native
    // tracer has no cancellation point, so this can still block for one
    // trace segment at worst.
    for (auto& pane : _panes) {
        if (!pane.session) {
            continue;
        }
        pane.session->solveQueue.shutdown();
        if (pane.session->runningSolveCancel) {
            pane.session->runningSolveCancel->store(true);
        }
    }
    _lineSolvePool.waitForDone();
}

void LineAnnotationController::setDatasetPickerForTesting(DatasetPicker picker)
{
    _datasetPicker = std::move(picker);
}

void LineAnnotationController::setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory)
{
    _optimizationTaskFactory = std::move(factory);
}

void LineAnnotationController::setModeChangeConfirmationForTesting(
    std::function<bool(vc3d::line_annotation::FiberOptimizationMode)> confirmer)
{
    _modeChangeConfirmation = std::move(confirmer);
}

bool LineAnnotationController::confirmFiberOptimizationModeChange(
    const LineAnnotationSession& session,
    vc3d::line_annotation::FiberOptimizationMode requestedMode)
{
    if (_modeChangeConfirmation) {
        return _modeChangeConfirmation(requestedMode);
    }
    if (session.suppressErrorDialogs || _errorDialogsSuppressed) {
        // Agent-driven flows are deliberate scripted actions; the 'reviewed'
        // tag is still stripped on the save that follows, so the new
        // geometry does not inherit the old human verdict.
        return true;
    }
    const bool toNative = requestedMode ==
        vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d;
    QMessageBox prompt(_parentWidget.data());
    prompt.setIcon(QMessageBox::Warning);
    prompt.setWindowTitle(tr("Change interpolation mode"));
    if (toNative) {
        prompt.setText(
            tr("Switching to the Fiber model re-traces every global-goal "
               "span with model predictions and overwrites the current "
               "line."));
    } else {
        prompt.setText(
            tr("Switching to Lasagna re-fits every global-goal span and "
               "overwrites the traced line."));
    }
    prompt.setInformativeText(
        tr("If the fiber is tagged 'reviewed', the tag is removed when the "
           "re-optimized line is saved; re-tag it after inspecting the new "
           "geometry."));
    auto* proceedButton =
        prompt.addButton(tr("Re-optimize"), QMessageBox::AcceptRole);
    prompt.addButton(QMessageBox::Cancel);
    prompt.setDefaultButton(QMessageBox::Cancel);
    prompt.exec();
    return prompt.clickedButton() == proceedButton;
}

void LineAnnotationController::setMergeModePickerForTesting(
    std::function<std::optional<vc3d::line_annotation::FiberOptimizationMode>(
        vc3d::line_annotation::FiberOptimizationMode,
        vc3d::line_annotation::FiberOptimizationMode)> picker)
{
    _mergeModePicker = std::move(picker);
}

std::optional<vc3d::line_annotation::FiberOptimizationMode>
LineAnnotationController::pickMergeOptimizationMode(
    const LineAnnotationSession& session,
    vc3d::line_annotation::FiberOptimizationMode clickedMode,
    vc3d::line_annotation::FiberOptimizationMode candidateMode)
{
    if (clickedMode == candidateMode) {
        return clickedMode;
    }
    if (_mergeModePicker) {
        return _mergeModePicker(clickedMode, candidateMode);
    }
    if (session.suppressErrorDialogs || _errorDialogsSuppressed) {
        // Agent-driven flows cannot answer modals; the clicked fiber's mode
        // is the deterministic default.
        return clickedMode;
    }
    const auto modeLabel =
        [](vc3d::line_annotation::FiberOptimizationMode mode) {
            return mode ==
                    vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d
                ? tr("Fiber model")
                : tr("Lasagna");
        };
    QMessageBox prompt(_parentWidget.data());
    prompt.setIcon(QMessageBox::Question);
    prompt.setWindowTitle(tr("Merge with different modes"));
    prompt.setText(tr("The two fibers use different interpolation modes. "
                      "The merged fiber will be re-optimized under the "
                      "chosen mode."));
    auto* clickedButton =
        prompt.addButton(tr("Keep %1 (this fiber)").arg(modeLabel(clickedMode)),
                         QMessageBox::AcceptRole);
    auto* candidateButton =
        prompt.addButton(tr("Keep %1 (candidate)").arg(modeLabel(candidateMode)),
                         QMessageBox::AcceptRole);
    prompt.addButton(QMessageBox::Cancel);
    prompt.setDefaultButton(clickedButton);
    prompt.exec();
    if (prompt.clickedButton() == clickedButton) {
        return clickedMode;
    }
    if (prompt.clickedButton() == candidateButton) {
        return candidateMode;
    }
    return std::nullopt;
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
    if (!viewer || !_state || !_viewerManager || !_state->objectStoreAttachment()) {
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

void LineAnnotationController::launchFromViewerAtPoint(CChunkedVolumeViewer* viewer,
                                                       const QPointF& scenePoint,
                                                       bool replaceOwningAnnotation,
                                                       bool modelOnly)
{
    if (!canLaunchFromViewer(viewer)) {
        return;
    }

    const auto sample = viewer->sampleSceneVolume(scenePoint);
    if (!sample) {
        return;
    }
    if (!placementAllowedByFocusBounds(toVec3d(sample->position), false)) {
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
    session->modelOnly = modelOnly;
    const auto volume = _state ? _state->currentVolume() : nullptr;
    if (!volume) {
        showError(tr("Creating a fiber requires an active volume."));
        return;
    }
    session->coordinateBaseShapeZYX =
        vc3d::line_annotation::fiberCoordinateShapeFromVolume(volume->shape());
    if (!launchSession(SourceKind::Plane,
                       surfaceName,
                       std::move(sourceSurface),
                       camera,
                       cv::Vec3d{normal[0], normal[1], normal[2]},
                       session,
                       !modelOnly)) {
        return;
    }
    handleLineSeed(surfaceName, sample->position, InitialDirectionMode::ZInOut);
}

bool LineAnnotationController::prepareForUserFacingLineAnnotationOpen(
    const std::function<bool()>& stillValid,
    const std::function<bool(const LineAnnotationSession&)>& only)
{
    const QPointer<LineAnnotationController> alive(this);
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
        // With a filter, a dialog without a session has nothing to decide
        // on and stays; without one every dialog goes, as before.
        if (only && (!pane.session || !only(*pane.session))) {
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
        // The finalization and the save below yield: the pane must still
        // hold this session afterwards (the record owns it meanwhile).
        const auto paneIntact = [this, &record]() {
            auto* pane = paneForSurface(record.surfaceName);
            return pane && pane->session == record.session;
        };
        const auto sessionValid = [&alive, &stillValid, &paneIntact]() {
            return alive && (!stillValid || stillValid()) && paneIntact();
        };
        if (!finalizeSessionOptimizationSynchronously(session, false, sessionValid)) {
            if (alive && paneIntact()) {
                session.suppressFiberSave = suppressBeforeFinalize;
            }
            return false;
        }
        if (!alive || (stillValid && !stillValid()) || !paneIntact()) {
            return false;
        }
        session.suppressFiberSave = suppressBeforeFinalize;
        if (!session.suppressFiberSave &&
            session.taskState == LineAnnotationSession::TaskState::Succeeded &&
            !session.optimizedLine.points.empty() &&
            !session.controlPoints.empty()) {
            // A save that fails synchronously — a serialization or validation
            // exception — never queues a job, and it can fail before the fiber
            // list was updated, so this session may hold the only copy of the
            // work. Refuse before anything closes, and leave the session
            // saveable rather than suppressed, so retrying is possible.
            // Asynchronous write failures are different: their job holds a
            // full snapshot and the fiber list was already updated, so they
            // are handled by the flush-and-compare gate a caller runs before
            // acting destructively.
            const uint64_t saveFailuresBefore = _fiberSaveFailureCount;
            saveSessionAsFiber(session);
            if (!alive || (stillValid && !stillValid()) || !paneIntact()) {
                return false;
            }
            if (_fiberSaveFailureCount != saveFailuresBefore) {
                session.suppressFiberSave = suppressBeforeFinalize;
                return false;
            }
            session.suppressFiberSave = true;
        }
    }

    for (const auto& record : dialogs) {
        if (record.dialog) {
            record.dialog->setCloseAfterFinalizationAllowed(true);
            record.dialog->close();
            // The close runs the dialog's teardown slots synchronously.
            if (!alive) {
                return false;
            }
        }
        cleanupSurfaceName(record.surfaceName);
        if (!alive) {
            return false;
        }
    }
    return true;
}

bool LineAnnotationController::prepareForPackageSwitch()
{
    // Ends every line-annotation session while its package is still the active
    // one, so the existing finalize/save/close path writes into the paths the
    // work belongs to. The project-open flows call this *before* replacing the
    // package; the vpkgChanged handler is only the defensive net behind it,
    // and that net must never save.
    //
    // Captured before any save this function can trigger — synchronous
    // serialization failures bump the count during the finalize walk itself,
    // not only during the flush at the end.
    const uint64_t saveFailuresBefore = _fiberSaveFailureCount;
    // Debounced autosaves must land in this package's files, not the next's.
    flushAllPendingSessionAutoSaves();
    // Refused outright while any optimization runs, whichever pane owns it:
    // switching mid-run would either discard the result or apply it to the
    // wrong package, and a refusal the user can retry beats both. Checked
    // before anything is torn down, so a refusal leaves the workspace intact.
    for (const auto& pane : _panes) {
        if (pane.session &&
            pane.session->taskState == LineAnnotationSession::TaskState::Running) {
            showError(tr("Finish or cancel the running line optimization before "
                         "opening another project."));
            return false;
        }
    }
    // Finalizes and saves idle sessions, then closes their dialogs
    // (WA_DeleteOnClose deletes them) and cleans up their panes. Fails before
    // closing anything if a finalization cannot complete — which is why it
    // runs before any teardown below: every way this function can refuse must
    // come before anything it destroys, or a refused switch would still have
    // cost the user work.
    if (!prepareForUserFacingLineAnnotationOpen()) {
        return false;
    }
    // Past every refusal point. The inspection holds dialog-less sessions the
    // finalize walk above never sees; its own cleanup removes its panes and
    // surfaces, and it has nothing to save.
    if (_intersectionInspection) {
        cleanupIntersectionInspectionSurfaces();
        _intersectionInspection.reset();
    }
    // Whatever remains has no dialog; the normal close path still saves a
    // succeeded session — into this, still-current, package.
    std::vector<std::string> remaining;
    remaining.reserve(_panes.size());
    for (const auto& pane : _panes) {
        remaining.push_back(pane.surfaceName);
    }
    for (const auto& name : remaining) {
        cleanupSurfaceName(name);
    }
    // The close path above only *queues* fiber writes; the switch must not
    // proceed until they have actually reached this package's disk. The flush
    // and the failure-count comparison are the same gate the header documents
    // for destructive follow-ups (and saveOpenFibers() makes the same flush on
    // shutdown); the count also covers a save that failed before it could
    // queue a job. The fiber list keeps the failed fiber in memory, so
    // refusing here leaves the user able to save it again.
    waitForFiberSaves();
    if (_fiberSaveFailureCount != saveFailuresBefore) {
        showError(tr("A fiber save failed while closing sessions; the project "
                     "was not switched."));
        return false;
    }
    return _panes.empty();
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
    session->suppressErrorDialogs =
        session->suppressErrorDialogs || _errorDialogsSuppressed;
    if (!prepareForUserFacingLineAnnotationOpen()) {
        return false;
    }

    session->deferShowUntilGenerated = deferShowUntilGenerated;
    _state->setSurface(surfaceName, std::move(sourceSurface));
    auto* dialog = new LineAnnotationDialog(_viewerManager, _volumeSelectorFactory, nullptr);
    dialog->setFiberDisplayName(fiberDisplayNameFromFileName(session->fiberFileName));
    dialog->setFiberOptimizationMode(session->fiberOptimizationMode);
    dialog->setModelOnlySession(session->modelOnly);
    refreshLineAnnotationDatasetMenu(dialog);
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
    pushFiberUiState(_panes.back());
    // A pane opened after the umbilicus was resolved has to be told too; the
    // notice describes the package, not this pane's fiber.
    publishUmbilicusNotice();
    connect(dialog, &LineAnnotationDialog::volumeOverlayToggleRequested,
            this, &LineAnnotationController::volumeOverlayToggleRequested);
    connect(dialog, &LineAnnotationDialog::paneClosed, this, [this](const std::string& name) {
        cleanupSurfaceName(name);
    });
    connect(dialog,
            &LineAnnotationDialog::closeFinalizationRequested,
            this,
            [this, surfaceName](QCloseEvent*) {
                requestFinalizedClose(surfaceName);
            });
    // Viewing and navigation work in every pane, read-only included.
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointBranchOpenRequested,
            this,
            [this](uint64_t branchFiberId, int branchControlPointIndex) {
                openFiberAtControlPoint(branchFiberId, branchControlPointIndex);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedNearbyAnnotationOpenRequested,
            this,
            [this, surfaceName](uint64_t fiberId, cv::Vec3f volumePoint) {
                handleGeneratedOpenNearbyAnnotation(
                    fiberId,
                    volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedSideStripIntersectionQueryRequested,
            this,
            [this](const std::string& name) {
                handleGeneratedSideStripIntersectionQuery(name);
            });
    if (session->readOnly) {
        // View only until the user enables editing: none of the editing,
        // linking or optimization requests are connected, and the window
        // closes without finalization.
        dialog->setCloseAfterFinalizationAllowed(true);
        dialog->setFiberDisplayName(fiberDisplayNameFromFileName(session->fiberFileName) + tr(" · read only"));
        dialog->setReadOnly(true, session->editingAvailable);
        connect(dialog, &LineAnnotationDialog::editingRequested, this, [this, surfaceName]() {
            enableSessionEditing(surfaceName);
        });
    } else {
        connectSessionEditing(dialog, surfaceName);
    }
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

bool LineAnnotationController::enableFiberEditing(uint64_t fiberId)
{
    for (auto it = _panes.rbegin(); it != _panes.rend(); ++it) {
        if (it->dialog && it->session && it->session->fiberId == fiberId) {
            const std::string surfaceName = it->surfaceName;
            return enableSessionEditing(surfaceName);
        }
    }
    return false;
}

bool LineAnnotationController::enableSessionEditing(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session || !pane->dialog) {
        return false;
    }
    auto session = pane->session;
    if (!session->readOnly) {
        return true;
    }
    if (!session->editingAvailable) {
        showError(tr("This fiber cannot be edited: the object store is attached read-only or the fiber does not decode."),
                  session->suppressErrorDialogs);
        return false;
    }
    // A hosted replica edits only fibers it leases.
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    if (connection && session->storePin && !session->storePin->objectRevision.empty()) {
        try {
            session->editingLeases = vc3d::store::holdForEditing(*connection, *session->storePin);
        } catch (const std::exception& error) {
            showError(tr("This fiber cannot be edited now: %1").arg(QString::fromUtf8(error.what())),
                      session->suppressErrorDialogs);
            return false;
        }
    }
    // Editing starts from the revision on screen; a newer one reopens the view.
    const auto stored = readStoredFiber(session->fiberId);
    if (!stored || stored->readOnly || stored->storePin != session->storePin) {
        showError(tr("The fiber changed in the object store since it was opened; reopen it to edit."),
                  session->suppressErrorDialogs);
        return false;
    }
    // Solves need the normals; the dataset picker can process events.
    if (!session->modelOnly && !ensureDatasetForSession(*session)) {
        return false;
    }
    pane = paneForSurface(surfaceName);
    if (!pane || pane->session != session || !pane->dialog) {
        return false;
    }
    if (session->inspectionFrameOnly && session->normalSampler && !session->optimizedLine.points.empty()) {
        // The view was opened without normals: rebuild the same line in the sheet frame.
        try {
            session->optimizedLine = lineModelFromPoints(linePointPositions(session->optimizedLine),
                                                         session->normalSampler.get());
            session->inspectionFrameOnly = false;
            ++session->lineRevision;
            materializeGeneratedViews(*session);
        } catch (const std::exception& error) {
            showError(tr("Could not prepare the fiber for editing: %1").arg(QString::fromUtf8(error.what())),
                      session->suppressErrorDialogs);
            return false;
        }
    }
    session->readOnly = false;
    session->suppressFiberSave = false;
    pane->dialog->setReadOnly(false);
    pane->dialog->setCloseAfterFinalizationAllowed(false);
    pane->dialog->setFiberDisplayName(fiberDisplayNameFromFileName(session->fiberFileName));
    connectSessionEditing(pane->dialog, surfaceName);
    pushFiberUiState(*pane);
    publishUmbilicusNotice();
    if (session->optimizedLine.points.empty() && session->controlPoints.empty()) {
        // A single-point fiber: tracing its line is the first edit.
        handleLineSeed(surfaceName,
                       toVec3f(session->seedPoint),
                       InitialDirectionMode::ZInOut,
                       SeedOrigin::StoredFiber,
                       stored->controlPoints.empty() ? std::vector<std::string>{}
                                                     : stored->controlPoints.front().tags);
    }
    return true;
}

void LineAnnotationController::connectSessionEditing(LineAnnotationDialog* dialog, const std::string& surfaceName)
{
    connect(dialog, &LineAnnotationDialog::fiberFollowConfidenceChanged, this, [this]() {
        for (const auto& pane : _panes)
            if (pane.dialog) pane.dialog->updateFiberFollowMenu();
    });
    connect(dialog, &LineAnnotationDialog::fiberFollowSettingsChanged, this, [this, surfaceName]() {
        for (const auto& pane : _panes) {
            if (pane.dialog) {
                pane.dialog->updateFiberFollowMenu();
            }
        }
        // Load the model now rather than at the first trace.
        auto* pane = paneForSurface(surfaceName);
        if (fiberFollowEnabled() || (pane && pane->session && pane->session->modelOnly)) {
            ensureFiberFollowService(pane && pane->session && pane->session->suppressErrorDialogs,
                                     pane ? pane->dialog.data() : nullptr);
        }
    });
    connect(dialog, &LineAnnotationDialog::fiberWidthChanged, this, [this, surfaceName](double width) {
        auto* pane = paneForSurface(surfaceName);
        if (!pane || !pane->session) return;
        auto& session = *pane->session;
        setSessionFiberWidth(session, width);
        saveFiberDisplayAnnotations(session);
    });
    connect(dialog, &LineAnnotationDialog::controlDisplayAngleChanged, this,
            [this, surfaceName](size_t index, double degrees) {
        auto* pane = paneForSurface(surfaceName);
        if (!pane || !pane->session) return;
        auto& session = *pane->session;
        if (index >= session.controlPoints.size() || index >= session.displayField.controlBaselines.size()) return;
        const auto baseline = vc::fiber_tracer::displayUnit(session.displayField.controlBaselines[index]);
        const auto tangent = vc::fiber_tracer::displayUnit(session.displayField.controlTangents[index]);
        if (!baseline || !tangent) return;
        const std::optional<cv::Vec3d> normal = degrees == 0 ? std::nullopt :
            std::optional<cv::Vec3d>(vc::fiber_tracer::rotateDisplayNormal(*baseline, *tangent, degrees * std::acos(-1.0) / 180.0));
        const cv::Vec3d position = session.controlPoints[index].volumePoint;
        for (auto& other : _panes) {
            if (!other.session || (other.session != pane->session &&
                (!session.fiberId || other.session->fiberId != session.fiberId))) continue;
            for (auto& cp : other.session->controlPoints)
                if (pointsApproximatelyEqual(cp.volumePoint, position)) {
                    cp.displayNormal = normal;
                    cp.displayNormalSource = normal ? "manual" : "unknown";
                }
            materializeGeneratedViews(*other.session);
        }
        saveFiberDisplayAnnotations(session);
    });
    const auto clearCorrections = [this, surfaceName](std::optional<size_t> index) {
        auto* pane=paneForSurface(surfaceName);
        if (!pane || !pane->session) return;
        auto& session=*pane->session;
        if (index && *index >= session.controlPoints.size()) return;
        const std::optional<cv::Vec3d> point = index
            ? std::make_optional(session.controlPoints[*index].volumePoint) : std::nullopt;
        bool directionsChanged=false;
        std::vector<size_t> dirty;
        for (size_t i=0; i<session.controlPoints.size(); ++i) {
            if (index && i != *index) continue;
            if (!session.controlPoints[i].direction) continue;
            directionsChanged=true;
            if (i>0) dirty.push_back(i-1);
            if (i+1<session.controlPoints.size()) dirty.push_back(i);
        }
        std::sort(dirty.begin(), dirty.end());
        dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());
        for (auto& other : _panes) {
            if (!other.session || (other.session!=pane->session &&
                (!session.fiberId || other.session->fiberId!=session.fiberId))) continue;
            for (auto& cp : other.session->controlPoints) {
                if (point && !pointsApproximatelyEqual(cp.volumePoint,*point)) continue;
                vc3d::line_annotation::clearControlPointCorrections(cp);
            }
            ++other.session->lineRevision;
            materializeGeneratedViews(*other.session);
        }
        saveFiberDisplayAnnotations(session);
        if (directionsChanged) {
            session.fiberMetricsMatchStoredFiber=false;
            invalidateFiberAlignmentMetrics(session.fiberId,true);
            session.stripReviewedTagOnSave=true;
            setSessionOptimizationState(session,SessionOptimizationState::Unoptimized);
            session.solveQueue.addPending(dirty,false);
            scheduleSolveDispatch(session);
        }
    };
    connect(dialog, &LineAnnotationDialog::clearFiberCorrectionsRequested, this,
            [clearCorrections]() { clearCorrections(std::nullopt); });
    connect(dialog, &LineAnnotationDialog::clearControlCorrectionsRequested, this,
            [clearCorrections](size_t index) { clearCorrections(index); });
    connect(dialog, &LineAnnotationDialog::controlDirectionCreated, this,
            [this, surfaceName](cv::Vec3f point, double position,
                                cv::Vec3f anchor, cv::Vec3f direction) {
        handleGeneratedControlPoint(surfaceName,
            point, position,
            anchor, std::nullopt,
            vc::fiber_tracer::displayUnit(cv::Vec3d(direction)));
    });
    connect(dialog, &LineAnnotationDialog::controlDirectionChanged, this,
            [this, surfaceName](size_t index, cv::Vec3f value) {
        auto* pane = paneForSurface(surfaceName);
        if (!pane || !pane->session) return;
        auto& session=*pane->session;
        const auto axis=vc::fiber_tracer::displayUnit(cv::Vec3d(value));
        if (!axis || index>=session.controlPoints.size()) return;
        const auto point=session.controlPoints[index].volumePoint;
        for (auto& other : _panes) {
            if (!other.session || (other.session!=pane->session &&
                (!session.fiberId || other.session->fiberId!=session.fiberId))) continue;
            for (auto& cp : other.session->controlPoints)
                if (pointsApproximatelyEqual(cp.volumePoint,point)) cp.direction=*axis;
            materializeGeneratedViews(*other.session);
        }
        ++session.lineRevision;
        session.fiberMetricsMatchStoredFiber=false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
        session.stripReviewedTagOnSave=true;
        setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
        std::vector<size_t> dirty;
        if (index>0) dirty.push_back(index-1);
        if (index+1<session.controlPoints.size()) dirty.push_back(index);
        session.solveQueue.addPending(dirty,false);
        scheduleSessionAutoSave(session);
        scheduleSolveDispatch(session);
    });
    connect(dialog,
            &LineAnnotationDialog::lineSeedRequested,
            this,
            [this](const std::string& name, cv::Vec3f volumePoint, QPointF) {
                handleLineSeed(name,
                               volumePoint,
                               InitialDirectionMode::ZInOut);
            });
    connect(dialog,
            &LineAnnotationDialog::fiberTagChangeRequested,
            this,
            [this, surfaceName](const QString& tag, bool enabled) {
                auto* pane = paneForSurface(surfaceName);
                if (!pane || !pane->session || pane->session->fiberId == 0) {
                    return;
                }
                setFiberTag(pane->session->fiberId, tag, enabled);
            });
    connect(dialog,
            &LineAnnotationDialog::lasagnaDatasetSelectionChanged,
            this,
            [this](const std::string& location) {
                handleLasagnaDatasetSelectionChanged(location);
            });
    connect(dialog,
            &LineAnnotationDialog::fiberInferenceDatasetSelectionChanged,
            this,
            [this](const std::string& location) {
                handleFiberInferenceDatasetSelectionChanged(location);
            });
    connect(dialog,
            &LineAnnotationDialog::crossSectionDragFinished,
            this,
            [this](const std::string& name, cv::Vec3f point, double position,
                   cv::Vec3f anchor, cv::Vec3f normal, bool edge) {
                handleGeneratedControlPoint(name, point,
                    position, anchor,
                    edge ? std::optional<cv::Vec3d>(toVec3d(normal)) : std::nullopt);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointRequested,
            this,
            [this](const std::string& name,
                   cv::Vec3f volumePoint,
                   double linePosition,
                   cv::Vec3f lineAnchor) {
                handleGeneratedControlPoint(
                    name,
                    volumePoint,
                    linePosition,
                    lineAnchor);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointDeleteRequested,
            this,
            [this](const std::string& name, double linePosition, cv::Vec3f volumePoint) {
                handleGeneratedControlPointDelete(
                    name, linePosition, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedNewLineAnnotationLinkedToCandidateRequested,
            this,
            [this](const std::string& name, cv::Vec3f volumePoint, cv::Vec3f linkDirection) {
                handleGeneratedNewLineAnnotationLinkedToCandidate(name,
                                                                  volumePoint,
                                                                  linkDirection);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointLinkCandidateRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, cv::Vec3f volumePoint) {
                handleGeneratedControlPointLinkCandidate(
                    name, controlPointIndex, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointAdjacentLinkCandidateRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, cv::Vec3f volumePoint) {
                handleGeneratedControlPointLinkCandidate(
                    name, controlPointIndex, volumePoint,
                    /*adjacent=*/true);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointLinkWithCandidateRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, cv::Vec3f volumePoint) {
                handleGeneratedControlPointLinkWithCandidate(
                    name, controlPointIndex, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointMergeWithCandidateRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, cv::Vec3f volumePoint) {
                handleGeneratedControlPointMergeWithCandidate(
                    name, controlPointIndex, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedSpanSplitRequested,
            this,
            [this](const std::string& name, size_t first, size_t second, bool linkHalves) {
                handleGeneratedSpanSplit(name, first, second, linkHalves);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedSpanGapChangeRequested,
            this,
            [this](const std::string& name, size_t first, size_t second, bool enabled) {
                handleGeneratedSpanSetGap(name, first, second, enabled);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedSpanDamagedChangeRequested,
            this,
            [this](const std::string& name, size_t first, size_t second, bool enabled) {
                handleGeneratedSpanSetDamaged(name, first, second, enabled);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedNearbyFiberLinkRequested,
            this,
            [this](const std::string& name, uint64_t fiberId, cv::Vec3f intersectionPoint,
                   double anchorLinePosition, uint64_t anchorLineRevision) {
                handleGeneratedLinkNearbyFiber(name, fiberId, intersectionPoint,
                                               anchorLinePosition, anchorLineRevision);
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
                handleGeneratedPredSnapPoint(
                    name, volumePoint);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedSegmentInterpolationGoalRequested,
            this,
            [this](const std::string& name,
                   size_t firstControlPointIndex,
                   size_t secondControlPointIndex,
                   const std::string& goal) {
                handleGeneratedSegmentInterpolationGoal(
                    name, firstControlPointIndex, secondControlPointIndex, goal);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointKollesisTerminationChangeRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, bool enabled) {
                handleGeneratedControlPointSetKollesisTermination(
                    name, controlPointIndex, enabled);
            });
    connect(dialog,
            &LineAnnotationDialog::generatedControlPointBreakChangeRequested,
            this,
            [this](const std::string& name, size_t controlPointIndex, bool enabled) {
                handleGeneratedControlPointSetBreak(name, controlPointIndex, enabled);
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
            showError(tr("Line optimization is already running."),
                      session.suppressErrorDialogs);
            return;
        }
        if (session.optimizedLine.points.empty() || session.controlPoints.empty()) {
            return;
        }
        if (!ensureDatasetForSession(session)) {
            return;
        }
        startFiberModeOptimization(session, true);
    });
    connect(dialog,
            &LineAnnotationDialog::fiberOptimizationModeChanged,
            this,
            [this, surfaceName](vc3d::line_annotation::FiberOptimizationMode mode) {
                auto* pane = paneForSurface(surfaceName);
                if (!pane || !pane->session) {
                    return;
                }
                auto& session = *pane->session;
                if (session.modelOnly && mode == vc3d::line_annotation::FiberOptimizationMode::Lasagna) {
                    pane->dialog->setFiberOptimizationMode(session.fiberOptimizationMode);
                    showError(tr("This session uses the fiber-follow model. Start a normal line annotation to use Lasagna."),
                              session.suppressErrorDialogs);
                    return;
                }
                if (mode == session.fiberOptimizationMode) {
                    return;
                }
                if (session.taskState == LineAnnotationSession::TaskState::Running) {
                    showError(tr("Line optimization is already running."),
                              session.suppressErrorDialogs);
                    if (pane->dialog) {
                        pane->dialog->setFiberOptimizationMode(
                            session.fiberOptimizationMode);
                    }
                    return;
                }
                if (session.optimizedLine.points.size() < 2 ||
                    session.controlPoints.empty()) {
                    session.fiberOptimizationMode = mode;
                    return;
                }
                if (!confirmFiberOptimizationModeChange(session, mode)) {
                    if (pane->dialog) {
                        pane->dialog->setFiberOptimizationMode(
                            session.fiberOptimizationMode);
                    }
                    return;
                }

                session.fiberOptimizationModeBeforeTask =
                    session.fiberOptimizationMode;
                session.restoreFiberOptimizationModeOnFailure = true;
                session.controlPointsBeforeModeChange = session.controlPoints;
                session.optimizedLineBeforeModeChange = session.optimizedLine;
                session.branchesBeforeModeChange = session.branches;
                session.fiberOptimizationMode = mode;
                if (ensureDatasetForSession(session))
                    startFiberModeOptimization(session, true, std::nullopt, true);
                if (session.taskState != LineAnnotationSession::TaskState::Running) {
                    session.fiberOptimizationMode =
                        session.fiberOptimizationModeBeforeTask;
                    session.restoreFiberOptimizationModeOnFailure = false;
                    if (session.controlPointsBeforeModeChange) {
                        session.controlPoints =
                            std::move(*session.controlPointsBeforeModeChange);
                        session.controlPointsBeforeModeChange.reset();
                    }
                    consumeRollbackInvalidation(session);
                    if (session.optimizedLineBeforeModeChange) {
                        session.optimizedLine =
                            std::move(*session.optimizedLineBeforeModeChange);
                        session.optimizedLineBeforeModeChange.reset();
                    }
                    if (session.branchesBeforeModeChange) {
                        session.branches =
                            std::move(*session.branchesBeforeModeChange);
                        session.branchesBeforeModeChange.reset();
                    }
                    if (pane->dialog) {
                        pane->dialog->setFiberOptimizationMode(
                            session.fiberOptimizationMode);
                    }
                }
            });
    connect(dialog,
            &LineAnnotationDialog::extrapolationDistanceChanged,
            this,
            [this, surfaceName](int) { handleTailConfigChanged(surfaceName); });
    connect(dialog,
            &LineAnnotationDialog::breakExtrapolationDistanceChanged,
            this,
            [this, surfaceName](int) { handleTailConfigChanged(surfaceName); });
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
                    showError(tr("Line optimization is already running."),
                              session.suppressErrorDialogs);
                    return;
                }
                if (session.optimizedLine.points.empty() || session.controlPoints.empty()) {
                    return;
                }
                if (!ensureDatasetForSession(session)) {
                    return;
                }
                startFiberModeOptimization(session, false);
            });
}

void LineAnnotationController::openFiber(uint64_t fiberId)
{
    openFiberWithControlPoint(fiberId, std::nullopt, std::nullopt);
}

LineAnnotationDialog* LineAnnotationController::openCollectionFiber(uint64_t fiberId, QString* error)
{
    // The caller reports failures in its own status area.
    QScopedValueRollback<bool> quiet(_errorDialogsSuppressed, true);
    QScopedValueRollback<QString> previousError(_lastSuppressedError, QString{});
    const QPointer<LineAnnotationDialog> previous = mostRecentLineAnnotationDialog();
    openFiberWithControlPoint(fiberId, std::nullopt, std::nullopt, std::nullopt, true);
    auto* dialog = mostRecentLineAnnotationDialog();
    if (!_lastSuppressedError.isEmpty() || !dialog || dialog == previous ||
        _panes.empty() || !_panes.back().session ||
        _panes.back().session->fiberId != fiberId ||
        _panes.back().session->generatedLineSurfaceName.empty()) {
        if (error)
            *error = _lastSuppressedError.isEmpty()
                ? tr("Could not open the flattened fiber. The previous annotation may still be busy.")
                : _lastSuppressedError;
        return nullptr;
    }
    return dialog;
}



std::optional<std::vector<cv::Vec3d>> LineAnnotationController::fiberMapLinePoints(uint64_t fiberId) const
{
    // Exactly the points fiberMapSnapshot draws, so the map can invert a
    // drawn point to a position on the stored line (positions are indices,
    // shared by every frame of one line).
    const auto fiber = readStoredFiber(fiberId, true);
    if (!fiber || !fiber->editorError.empty() || !fiberInVolume(*fiber)) {
        return std::nullopt;
    }
    auto points = fiberInVolumeCoordinates(*fiber).linePoints;
    const double factor = annotationFrame().factor;
    for (auto& point : points) {
        point *= factor;
    }
    return points;
}

LineAnnotationController::FiberMapAutoLinkOutcome
LineAnnotationController::prepareFiberMapAutoLink(uint64_t fiberIdA, uint64_t fiberIdB)
{
    if (fiberIdA == fiberIdB) {
        FiberMapAutoLinkOutcome outcome;
        outcome.ok = false;
        outcome.message = tr("Select two different fibers.");
        return outcome;
    }
    return prepareAutoLinkParticipants({fiberIdA, fiberIdB}, std::nullopt);
}

LineAnnotationController::FiberMapAutoLinkOutcome
LineAnnotationController::prepareAutoLinkParticipants(
    const std::vector<uint64_t>& participantIds,
    const std::optional<std::string>& keepSurfaceName)
{
    FiberMapAutoLinkOutcome outcome;
    const auto refuse = [&outcome](const QString& message) {
        outcome.ok = false;
        outcome.message = message;
        return outcome;
    };
    const QPointer<LineAnnotationController> alive(this);
    const uint64_t packageGeneration = _packageGeneration;
    const uint64_t attachmentSequence = _attachmentSequence;
    for (const uint64_t id : participantIds) {
        const auto fiber = readStoredFiber(id);
        if (!fiber) {
            return refuse(tr("One of the fibers is no longer in the store; press Update."));
        }
        if (fiber->readOnly) {
            return refuse(tr("%1 is read-only.").arg(fiberDisplayName(id)));
        }
    }
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->taskState == LineAnnotationSession::TaskState::Running) {
            return refuse(tr("Line optimization is already running."));
        }
    }
    const auto isParticipant = [&participantIds](const LineAnnotationSession& session) {
        return std::find(participantIds.begin(), participantIds.end(), session.fiberId) !=
               participantIds.end();
    };
    // The one pane that stays (the dialog the link is made from): never
    // finalized, retired or counted as still open.
    const auto kept = [&keepSurfaceName](const std::string& surfaceName) {
        return keepSurfaceName && *keepSurfaceName == surfaceName;
    };

    // The active volume is an input of every scale a finalization resolves
    // (model-only sessions resolve theirs before the service start yields):
    // a switch during preparation invalidates it.
    const auto volume = _state ? _state->currentVolume() : nullptr;
    const auto stillValid = [&alive, this, packageGeneration, attachmentSequence, volume]() {
        return alive && _packageGeneration == packageGeneration &&
               _attachmentSequence == attachmentSequence && _state &&
               _state->currentVolume() == volume;
    };
    // Dialogs of the participants are finalized, saved and closed exactly as
    // opening another fiber does it - but ONLY those: an unrelated dialog
    // stays open and unsaved, so it neither bumps the fiber generation
    // (which would make the map stale for nothing) nor loses its place.
    const auto finalizeFilter = [&isParticipant, &kept](const LineAnnotationSession& session) {
        return isParticipant(session) && !kept(session.surfaceName);
    };
    if (!prepareForUserFacingLineAnnotationOpen(stillValid, finalizeFilter)) {
        return refuse(tr("Could not finalize the open line annotation."));
    }
    if (!alive) {
        return refuse(tr("The annotation controller went away."));
    }
    if (!stillValid()) {
        return refuse(tr("The fibers changed meanwhile."));
    }

    // The dialog-less copies the finalize walk never sees: every remaining
    // pane of a participant, plus both sessions of the intersection
    // inspection when it involves one (its teardown erases both). Each is
    // finalized and saved BEFORE any pane is erased; a failure retains
    // everything.
    std::vector<std::string> retire;
    const auto addRetire = [&retire](const std::string& surfaceName) {
        if (!surfaceName.empty() &&
            std::find(retire.begin(), retire.end(), surfaceName) == retire.end()) {
            retire.push_back(surfaceName);
        }
    };
    for (const auto& pane : _panes) {
        if (pane.session && isParticipant(*pane.session) && !kept(pane.surfaceName)) {
            addRetire(pane.surfaceName);
        }
    }
    if (_intersectionInspection) {
        const bool involved =
            (_intersectionInspection->sourceLineSession &&
             isParticipant(*_intersectionInspection->sourceLineSession)) ||
            (_intersectionInspection->targetLineSession &&
             isParticipant(*_intersectionInspection->targetLineSession));
        if (involved) {
            addRetire(_intersectionInspection->sourceSessionSurfaceName);
            addRetire(_intersectionInspection->targetSessionSurfaceName);
        }
    }
    const auto finished = [](const LineAnnotationSession& session) {
        return session.taskState == LineAnnotationSession::TaskState::Succeeded &&
               !session.optimizedLine.points.empty() && !session.controlPoints.empty();
    };
    // Every retiring session is checked before any is touched: a refusal
    // on the second must not leave the first saved-and-suppressed.
    for (const std::string& surfaceName : retire) {
        const auto* pane = paneForSurface(surfaceName);
        if (pane && pane->session && !finished(*pane->session)) {
            return refuse(tr("A session of %1 did not finish its last optimization; close it first.")
                              .arg(fiberDisplayNameFromFileName(pane->session->fiberFileName)));
        }
    }
    // Saved sessions are marked suppressFiberSave so their close below does
    // not save again. A refusal anywhere before the closes are through
    // restores every flag: a session left open with its saves suppressed
    // would drop later edits.
    std::vector<std::pair<std::shared_ptr<LineAnnotationSession>, bool>> suppressed;
    bool retired = false;
    const auto restoreSuppression = qScopeGuard([&suppressed, &retired]() {
        if (retired) {
            return;
        }
        for (auto& [session, before] : suppressed) {
            session->suppressFiberSave = before;
        }
    });
    for (const std::string& surfaceName : retire) {
        auto* pane = paneForSurface(surfaceName);
        if (!pane || !pane->session) {
            continue;
        }
        // Owned for the duration: the pane can be erased by what the yields
        // run; after each one the pane must still exist with this session.
        const std::shared_ptr<LineAnnotationSession> session = pane->session;
        const auto sessionValid = [this, &stillValid, &surfaceName, &session]() {
            if (!stillValid()) {
                return false;
            }
            auto* current = paneForSurface(surfaceName);
            return current && current->session == session;
        };
        const QString name = fiberDisplayNameFromFileName(session->fiberFileName);
        if (!finished(*session)) {
            // Moved on by an earlier session's finalization or save.
            return refuse(tr("A session of %1 did not finish its last optimization; close it first.")
                              .arg(name));
        }
        if (!finalizeSessionOptimizationSynchronously(*session, false, sessionValid)) {
            return refuse(tr("Could not finalize a session of %1.").arg(name));
        }
        if (!sessionValid()) {
            return refuse(tr("The fibers changed meanwhile."));
        }
        const uint64_t failuresBefore = _fiberSaveFailureCount;
        saveSessionAsFiber(*session);
        if (!sessionValid()) {
            return refuse(tr("The fibers changed meanwhile."));
        }
        if (_fiberSaveFailureCount != failuresBefore) {
            return refuse(tr("Could not save a session of %1.").arg(name));
        }
        suppressed.emplace_back(session, session->suppressFiberSave);
        session->suppressFiberSave = true;
    }
    for (const std::string& surfaceName : retire) {
        cleanupSurfaceName(surfaceName);
        if (!stillValid()) {
            return refuse(tr("The fibers changed meanwhile."));
        }
    }
    // Every retiring pane is closed: the flags of the (now unowned)
    // sessions stay as the closes read them.
    retired = true;
    closeIntersectionInspectionForRetiredFibers(participantIds);
    if (!alive) {
        return refuse(tr("The annotation controller went away."));
    }

    std::string error;
    if (!structuralEditPreflight(packageGeneration, attachmentSequence, &error)) {
        return refuse(error.empty() ? tr("The fibers changed meanwhile.")
                                    : QString::fromStdString(error));
    }
    for (const auto& pane : _panes) {
        if (pane.session && isParticipant(*pane.session) && !kept(pane.surfaceName)) {
            return refuse(tr("A session of %1 is still open.")
                              .arg(fiberDisplayNameFromFileName(pane.session->fiberFileName)));
        }
    }
    outcome.ok = true;
    return outcome;
}

std::optional<QString> LineAnnotationController::linkPairEligibilityError(
    const StoredFiber& local, const StoredFiber& far) const
{
    if (local.id == far.id) {
        return tr("Select two different fibers.");
    }
    if (local.readOnly || far.readOnly) {
        return tr("%1 is read-only.").arg(fiberDisplayName(local.readOnly ? local.id : far.id));
    }
    return std::nullopt;
}

std::optional<QString> LineAnnotationController::linkStoredControlPoints(StoredFiber& local,
                                                                         int localControlIndex,
                                                                         StoredFiber& far,
                                                                         int farControlIndex,
                                                                         bool autoLink)
{
    namespace la = vc3d::line_annotation;
    if (localControlIndex < 0 ||
        static_cast<size_t>(localControlIndex) >= local.controlPoints.size() ||
        farControlIndex < 0 || static_cast<size_t>(farControlIndex) >= far.controlPoints.size()) {
        return tr("A control point index is out of range.");
    }
    if (const auto error = linkPairEligibilityError(local, far)) {
        return error;
    }
    const QString localName = fiberDisplayNameFromFileName(local.fileName);
    const QString farName = fiberDisplayNameFromFileName(far.fileName);
    if (controlPointHasBranchLink(local.branches, static_cast<size_t>(localControlIndex))) {
        return tr("%1 / CP %2 is already linked.").arg(localName).arg(localControlIndex);
    }
    if (controlPointHasBranchLink(far.branches, static_cast<size_t>(farControlIndex))) {
        return tr("%1 / CP %2 is already linked.").arg(farName).arg(farControlIndex);
    }
    const cv::Vec3d localPoint = local.controlPoints[static_cast<size_t>(localControlIndex)];
    const cv::Vec3d farPoint = far.controlPoints[static_cast<size_t>(farControlIndex)];
    if (!finitePoint(localPoint) || !finitePoint(farPoint)) {
        return tr("Could not determine finite endpoints for the link.");
    }
    // The two fibers may be stored in different frames: the fallback
    // direction (used only where a line gives no tangent) is taken in the
    // volume's.
    if (!fiberInVolume(local) || !fiberInVolume(far)) {
        return tr("Bind both fibers to the active volume before linking.");
    }
    const cv::Vec3d fallbackDirection = normalizedOrZero(farPoint - localPoint);
    const cv::Vec3d localDirection =
        endpointTangentFromLinePoints(local.linePoints, localPoint, fallbackDirection);
    const cv::Vec3d farDirection =
        endpointTangentFromLinePoints(far.linePoints, farPoint, -fallbackDirection);
    if (!finiteDirection(localDirection) || !finiteDirection(farDirection)) {
        return tr("Could not determine finite link directions between the control points.");
    }
    const FiberBranchRef localRef = la::makeBranchRef(
        localControlIndex, localPoint, localDirection, far.id, far.fileName, farControlIndex,
        farPoint, farDirection, /*pending=*/true, /*adjacent=*/false, autoLink);
    local.branches.push_back(localRef);
    far.branches.push_back(la::reciprocalBranchRef(localRef, local.id, local.fileName));
    try {
        commitFibers({&local, &far});
    } catch (const std::exception& ex) {
        local.branches.pop_back();
        far.branches.pop_back();
        return tr("Could not save linked fibers: %1").arg(QString::fromStdString(ex.what()));
    }
    invalidateFiberAlignmentMetrics(local.id, true);
    invalidateFiberAlignmentMetrics(far.id, true);
    refreshBranchLineViews(local.id);
    refreshBranchLineViews(far.id);
    return std::nullopt;
}

std::optional<QString> LineAnnotationController::placePreparedControl(
    AutoLinkPlacement& placement,
    const AutoLinkPlacementContext& context)
{
    // The automatic link's placement of ONE new control on a session that
    // the caller prepared (makeIntersectionLineSession, the solve backends,
    // the real line model): the geometric prepare + splice, the pre-solve
    // identity validation, the synchronous span re-optimization validated
    // before it is applied, the checked commit and the read-back of the
    // stored control. Every yield is followed by the caller's check.
    // placement.placed records a committed control (a refusal after it must
    // say the control was kept), placement.storedIndex the control's stored
    // index on success. Returns the refusal message.
    namespace la = vc3d::line_annotation;
    const QPointer<LineAnnotationController> alive(this);
    LineAnnotationSession& session = *placement.session;
    // An earlier step of this operation (the other fiber's placement) may
    // have revised this fiber's record through a shared link. Its geometry
    // must be the one the session was prepared from; the revision is then
    // adopted, as open panes adopt such receipts.
    {
        const auto current = readStoredFiber(session.fiberId);
        if (!current || !current->storePin || !session.storePin ||
            current->storePin->geometryRevision != session.storePin->geometryRevision) {
            return context.changedMessage;
        }
        session.storePin = current->storePin;
    }
    const size_t countBefore = session.controlPoints.size();
    const std::vector<cv::Vec3d> currentLinePoints = linePointPositions(session.optimizedLine);
    const auto cumulativeArclengths =
        vc3d::fiber_slice::cumulativePolylineArclengths(currentLinePoints);
    std::vector<double> controlLinePositions;
    controlLinePositions.reserve(session.controlPoints.size());
    for (const auto& control : session.controlPoints) {
        controlLinePositions.push_back(control.linePosition);
    }
    la::PreparedControlPointEdit prepared;
    try {
        GeometricControlPointEdit edit;
        edit.linePosition = placement.choice.linePosition;
        edit.point = placement.choice.point;
        edit.segmentLength = initialLineDiscretization(
            std::max(2, vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX_DEFAULT * 2))
            .segmentLength;
        prepared = applyGeometricControlPointEdit(session, currentLinePoints,
                                                  cumulativeArclengths,
                                                  controlLinePositions, edit);
    } catch (const std::exception& ex) {
        return tr("Could not place a control point on %1: %2")
            .arg(placement.name, QString::fromStdString(ex.what()));
    }
    // Pre-solve identity validation on the spliced session: nothing has
    // been published yet, so a violation simply discards the session.
    const auto validateControls =
        [&](const std::vector<la::LineControlPoint>& controls,
            const std::vector<cv::Vec3d>& line) -> std::optional<QString> {
        if (controls.size() != countBefore + 1) {
            return tr("placing the control point changed the control count of %1")
                .arg(placement.name);
        }
        std::vector<cv::Vec3d> positions;
        positions.reserve(controls.size());
        for (const auto& control : controls) {
            positions.push_back(control.volumePoint);
        }
        const auto ordered = la::orderedControlPointLineIndices(positions, line);
        if (!ordered || ordered->size() != controls.size()) {
            return tr("the control points of %1 no longer resolve in line order")
                .arg(placement.name);
        }
        for (size_t i = 0; i + 1 < ordered->size(); ++i) {
            if ((*ordered)[i] >= (*ordered)[i + 1]) {
                return tr("the control points of %1 no longer resolve in line order")
                    .arg(placement.name);
            }
        }
        // The nearest-vertex rule every consumer of this geometry uses (the
        // splice, the span optimizer, the next session) must agree with the
        // ordered scan for every control, on the spliced line AND on the
        // solved line, and so must the solver's own indices.
        for (size_t i = 0; i < controls.size(); ++i) {
            if (vc3d::fiber_slice::nearestLinePointIndex(line, positions[i]) != (*ordered)[i] ||
                controls[i].optimizedIndex < 0 ||
                static_cast<size_t>(controls[i].optimizedIndex) != (*ordered)[i]) {
                return tr("%1 passes through one of its control points twice")
                    .arg(placement.name);
            }
        }
        const auto placedCount = std::count_if(positions.begin(), positions.end(),
            [&placement](const cv::Vec3d& p) {
                return pointsApproximatelyEqual(p, placement.choice.point);
            });
        if (placedCount != 1) {
            return tr("the new control point of %1 is not unique").arg(placement.name);
        }
        return std::nullopt;
    };
    if (const auto error = validateControls(session.controlPoints,
                                            linePointPositions(session.optimizedLine))) {
        return tr("Could not place a control point: %1.").arg(*error);
    }
    if (prepared.replacementIndex >= session.controlPoints.size() ||
        !pointsApproximatelyEqual(session.controlPoints[prepared.replacementIndex].volumePoint,
                                  placement.choice.point)) {
        return tr("Could not place a control point on %1.").arg(placement.name);
    }
    // Backends again (a fiber-follow service can have stopped meanwhile),
    // then the solve, result only.
    if (!ensureSessionSolveBackends(session, [&]() { return context.check(true); })) {
        return tr("Could not prepare the solve for %1.").arg(placement.name);
    }
    if (!context.check(true)) {
        return context.changedMessage;
    }
    OptimizationTaskResult task = runFiberModeSolveSynchronously(
        session, prepared.dirtySegmentIndices, "fiber_auto_link_span_opt");
    if (!task.ok) {
        return tr("Re-optimization of %1 failed: %2")
            .arg(placement.name, QString::fromStdString(task.error));
    }
    {
        std::vector<cv::Vec3d> solvedLine;
        solvedLine.reserve(task.result.line.points.size());
        for (const auto& point : task.result.line.points) {
            solvedLine.push_back(point.position);
        }
        if (const auto error = validateControls(task.controlPoints, solvedLine)) {
            return tr("Re-optimization of %1 was rejected: %2").arg(placement.name, *error);
        }
    }
    if (!applyOptimizationTaskResult(session, std::move(task), false,
                                     SessionOptimizationState::Optimized,
                                     "fiber_auto_link_span_opt", false)) {
        return tr("Could not apply the re-optimization of %1.").arg(placement.name);
    }
    if (!alive) {
        return context.changedMessage;
    }
    const auto snapshot = makeStoredFiberSessionSnapshot(session);
    if (prepared.replacementIndex >= snapshot.storedIndexForSessionIndex.size()) {
        return tr("Could not place a control point on %1.").arg(placement.name);
    }
    const int storedIndex = snapshot.storedIndexForSessionIndex[prepared.replacementIndex];
    const uint64_t failuresBefore = _fiberSaveFailureCount;
    saveSessionAsFiber(session);
    if (!alive) {
        return context.changedMessage;
    }
    // The commit, read back before any verdict: a refusal from here on must
    // say whether this control is retained.
    if (const auto own = readStoredFiber(session.fiberId)) {
        placement.placed = storedIndex >= 0 &&
            static_cast<size_t>(storedIndex) < own->controlPoints.size() &&
            pointsApproximatelyEqual(own->controlPoints[static_cast<size_t>(storedIndex)],
                                     placement.choice.point);
    }
    if (_fiberSaveFailureCount != failuresBefore) {
        return tr("Could not save %1.").arg(placement.name);
    }
    // The commit's own generation bump is this operation's.
    *context.expectedGeneration = fiberDataGeneration();
    if (!context.check(true)) {
        return context.changedMessage;
    }
    if (!placement.placed) {
        return tr("The new control point of %1 was not found after saving.").arg(placement.name);
    }
    placement.storedIndex = storedIndex;
    return std::nullopt;
}

LineAnnotationController::AutoLinkChoiceSource
LineAnnotationController::choiceSourceFromStoredFiber(const StoredFiber& fiber) const
{
    AutoLinkChoiceSource source;
    source.linePoints = &fiber.linePoints;
    source.facts = vc3d::line_annotation::autoLinkChoiceFactsFromStoredFiber(fiber);
    return source;
}

std::optional<QString> LineAnnotationController::planAutoLinkChoices(
    std::array<AutoLinkChoiceSide, 2>& sides,
    double* passDistance) const
{
    // The automatic link's choices over plain data (line points, control
    // positions, linked flags, tags, gap spans - base voxels), so a stored
    // fiber and a live session feed the same rule: the ordered-scan /
    // nearest-vertex agreement, the closest 3D pass inside the windows, a
    // control per side (reused within reuseVx when unlinked, else placed).
    namespace la = vc3d::line_annotation;
    namespace al = vc3d::fiber_map::autolink;
    for (AutoLinkChoiceSide& side : sides) {
        const std::vector<cv::Vec3d>& linePoints = *side.source.linePoints;
        const std::vector<cv::Vec3d>& controls = side.source.facts.controls;
        if (linePoints.size() < 2) {
            return tr("%1 has no line to link on.").arg(side.name);
        }
        const auto ordered = la::orderedControlPointLineIndices(controls, linePoints);
        if (!ordered || ordered->size() != controls.size()) {
            return tr("The control points of %1 do not resolve in line order.").arg(side.name);
        }
        // The shared placement machinery (splice, span optimizer) resolves
        // controls by nearest vertex; it is only correct where that agrees
        // with the loader's ordered scan.
        for (size_t c = 0; c < controls.size(); ++c) {
            if (vc3d::fiber_slice::nearestLinePointIndex(linePoints, controls[c]) !=
                (*ordered)[c]) {
                return tr("%1 passes through one of its control points twice.").arg(side.name);
            }
        }
        side.orderedIndex = *ordered;
        side.cumulative = vc3d::fiber_slice::cumulativePolylineArclengths(linePoints);
    }
    al::FiberSide sideA{sides[0].source.linePoints, &sides[0].cumulative,
                        sides[0].anchorPosition, sides[0].windowVx};
    al::FiberSide sideB{sides[1].source.linePoints, &sides[1].cumulative,
                        sides[1].anchorPosition, sides[1].windowVx};
    const auto pass = al::findClosestPass(sideA, sideB);
    if (!pass) {
        return tr("Could not find where the fibers pass each other.");
    }
    const auto focusBounds = _state ? _state->activeFocusBounds() : std::nullopt;
    for (size_t s = 0; s < 2; ++s) {
        AutoLinkChoiceSide& side = sides[s];
        al::ControlChoiceInput input;
        input.linePoints = side.source.linePoints;
        input.cumulative = &side.cumulative;
        input.controlLineIndex = side.orderedIndex;
        input.linked = side.source.facts.linked;
        input.kollesisTerminations = side.source.facts.kollesisTerminations;
        input.gapSpans = side.source.facts.gapSpans;
        if (focusBounds) {
            input.inFocusBounds = [bounds = *focusBounds](const cv::Vec3d& p) {
                return contains_point(bounds, p);
            };
            input.focusBox = std::make_pair(
                cv::Vec3d(focusBounds->low[0], focusBounds->low[1], focusBounds->low[2]),
                cv::Vec3d(focusBounds->high[0], focusBounds->high[1], focusBounds->high[2]));
        }
        input.passPosition = s == 0 ? pass->positionA : pass->positionB;
        input.windowLo = s == 0 ? pass->windowLoA : pass->windowLoB;
        input.windowHi = s == 0 ? pass->windowHiA : pass->windowHiB;
        input.reuseVx = side.reuseVx;
        input.minSpacingVx = kMinimumControlPointSpacingBaseVoxels;
        side.choice = al::chooseControlPoint(input);
        if (side.choice.kind == al::ControlChoice::Kind::Refused) {
            return tr("No place for a control point on %1: %2")
                .arg(side.name, QString::fromStdString(side.choice.reason));
        }
    }
    if (passDistance) {
        *passDistance = pass->distance;
    }
    return std::nullopt;
}

LineAnnotationController::FiberMapAutoLinkOutcome
LineAnnotationController::autoLinkFibers(const FiberMapAutoLinkEndpoint& a,
                                         const FiberMapAutoLinkEndpoint& b,
                                         uint64_t mapFiberDataGeneration,
                                         std::function<bool()> mapStillFresh,
                                         const FiberMapAutoLinkParams& params)
{
    namespace al = vc3d::fiber_map::autolink;
    FiberMapAutoLinkOutcome outcome;
    const auto refuse = [&outcome](const QString& message) {
        outcome.ok = false;
        outcome.message = message;
        return outcome;
    };
    if (fiberDataGeneration() != mapFiberDataGeneration) {
        return refuse(tr("The Fiber Map is stale; press Update."));
    }
    if (a.fiberId == b.fiberId) {
        return refuse(tr("Select two different fibers."));
    }

    // The guard: everything this operation's correctness rests on, checked
    // after every yield. The expected fiber generation is re-captured after
    // this operation's own commits.
    const QPointer<LineAnnotationController> alive(this);
    const uint64_t packageGeneration = _packageGeneration;
    const uint64_t attachmentSequence = _attachmentSequence;
    uint64_t expectedGeneration = fiberDataGeneration();
    const uint64_t saveFailures = _fiberSaveFailureCount;
    const uint64_t focusRevision = _state ? _state->focusBoundsRevision() : 0;
    const auto volume = _state ? _state->currentVolume() : nullptr;
    if (!volume) {
        return refuse(tr("No active volume is loaded."));
    }
    // A structural edit for its whole duration: the same guard a merge or
    // split holds, so neither can start inside one of the yields below.
    if (_structuralEditInProgress) {
        return refuse(tr("A merge, split or link is in progress; try again once it has finished."));
    }
    _structuralEditInProgress = true;
    const auto releaseStructuralEdit = qScopeGuard([this, &alive]() {
        if (alive) {
            _structuralEditInProgress = false;
        }
    });
    const auto check = [&](bool includeGeneration) -> bool {
        if (!alive || _packageGeneration != packageGeneration ||
            _attachmentSequence != attachmentSequence) {
            return false;
        }
        // The active volume (its voxel size, the scale each fiber resolves
        // against) is an input of every physical threshold and of the solve.
        if (!_state || _state->currentVolume() != volume) {
            return false;
        }
        if (includeGeneration && fiberDataGeneration() != expectedGeneration) {
            return false;
        }
        if (_fiberSaveFailureCount != saveFailures ||
            _state->focusBoundsRevision() != focusRevision) {
            return false;
        }
        if (mapStillFresh && !mapStillFresh()) {
            return false;
        }
        for (const auto& pane : _panes) {
            if (pane.session && (pane.session->fiberId == a.fiberId ||
                                 pane.session->fiberId == b.fiberId ||
                                 pane.session->taskState == LineAnnotationSession::TaskState::Running)) {
                return false;
            }
        }
        return true;
    };
    const QString changedMessage = tr("The fibers changed meanwhile; nothing was linked.");
    if (!check(true)) {
        return refuse(changedMessage);
    }

    // Per-fiber working state.
    struct Side {
        const FiberMapAutoLinkEndpoint* endpoint = nullptr;
        std::optional<StoredFiber> fiber;
        QString name;
        double windowVx = 0.0;
        double reuseVx = 0.0;
        al::ControlChoice choice;
        std::shared_ptr<LineAnnotationSession> session;
        int linkedControlIndex = -1;
        bool placed = false;
    };
    std::array<Side, 2> sides;
    sides[0].endpoint = &a;
    sides[1].endpoint = &b;
    // After a committed placement, a refusal says that control stays (a
    // plain, unlinked control point).
    const auto refuseKeeping = [&](const QString& message) {
        QString kept;
        for (const Side& side : sides) {
            if (side.placed) {
                kept += tr(" The new control point placed on %1 was kept (unlinked).")
                            .arg(side.name);
            }
        }
        return refuse(message + kept);
    };
    for (Side& side : sides) {
        side.fiber = readStoredFiber(side.endpoint->fiberId, true);
        if (!side.fiber) {
            return refuse(tr("One of the fibers is no longer in the store; press Update."));
        }
    }
    // Pair eligibility BEFORE anything is mutated.
    if (const auto error = linkPairEligibilityError(*sides[0].fiber, *sides[1].fiber)) {
        return refuse(*error);
    }
    for (Side& side : sides) {
        side.name = fiberDisplayNameFromFileName(side.fiber->fileName);
        if (!fiberInVolume(*side.fiber)) {
            return refuse(tr("%1 is not bound to the active volume.").arg(side.name));
        }
        const double baseVoxelSizeUm = volume->voxelSize();
        if (!(baseVoxelSizeUm > 0.0) || !std::isfinite(baseVoxelSizeUm)) {
            return refuse(tr("The active volume has no usable voxel size."));
        }
        side.windowVx = params.searchWindowUm / baseVoxelSizeUm;
        side.reuseVx = params.reuseUm / baseVoxelSizeUm;
    }

    // Step 1: choices first (a reuse-only link needs no backend).
    {
        std::array<AutoLinkChoiceSide, 2> choiceSides;
        for (size_t s = 0; s < 2; ++s) {
            choiceSides[s].name = sides[s].name;
            choiceSides[s].source = choiceSourceFromStoredFiber(*sides[s].fiber);
            choiceSides[s].anchorPosition = sides[s].endpoint->linePosition;
            choiceSides[s].windowVx = sides[s].windowVx;
            choiceSides[s].reuseVx = sides[s].reuseVx;
        }
        double passDistance = 0.0;
        if (const auto error = planAutoLinkChoices(choiceSides, &passDistance)) {
            return refuse(*error);
        }
        for (size_t s = 0; s < 2; ++s) {
            sides[s].choice = choiceSides[s].choice;
        }
        // Two reused controls already linked to each other: refuse now.
        if (sides[0].choice.kind == al::ControlChoice::Kind::Reuse &&
            sides[1].choice.kind == al::ControlChoice::Kind::Reuse) {
            for (const FiberBranchRef& branch : sides[0].fiber->branches) {
                if (branch.branchFiberId == sides[1].fiber->id &&
                    branch.controlPointIndex == sides[0].choice.controlIndex &&
                    branch.branchControlPointIndex == sides[1].choice.controlIndex) {
                    return refuse(tr("%1 / CP %2 and %3 / CP %4 are already linked.")
                                      .arg(sides[0].name).arg(sides[0].choice.controlIndex)
                                      .arg(sides[1].name).arg(sides[1].choice.controlIndex));
                }
            }
        }
        outcome.message = tr("3D gap %1 vx").arg(passDistance, 0, 'f', 1);
    }
    const QString gapText = outcome.message;

    // Steps 2-3: backends and the real line model (sampled normals), only
    // for fibers that get a new control, before any edit. The samplers
    // throw on unusable normals; a throw is a refusal, not a crash.
    for (Side& side : sides) {
        if (side.choice.kind != al::ControlChoice::Kind::Place) {
            continue;
        }
        side.session = makeIntersectionLineSession(
            *side.fiber, side.choice.linePosition, cv::Vec3d{0.0, 0.0, 1.0},
            "fiber_auto_link_" + side.fiber->uuid, nullptr);
        side.session->suppressErrorDialogs = false;
        if (!ensureSessionSolveBackends(*side.session, [&]() { return check(true); })) {
            return refuse(tr("Could not prepare the solve for %1.").arg(side.name));
        }
        if (!check(true)) {
            return refuse(changedMessage);
        }
        try {
            if (!prepareSessionLineModel(*side.session, side.fiber->linePoints,
                                         [&]() { return check(true); })) {
                return refuse(tr("Could not prepare the line model of %1.").arg(side.name));
            }
        } catch (const std::exception& ex) {
            return refuse(tr("Could not prepare the line model of %1: %2")
                              .arg(side.name, QString::fromStdString(ex.what())));
        }
        if (!check(true)) {
            return refuse(changedMessage);
        }
    }

    // Step 4: placement, A then B, each committed on its own.
    for (Side& side : sides) {
        if (side.choice.kind == al::ControlChoice::Kind::Reuse) {
            side.linkedControlIndex = side.choice.controlIndex;
            continue;
        }
        AutoLinkPlacement placement;
        placement.session = side.session;
        placement.choice = side.choice;
        placement.name = side.name;
        AutoLinkPlacementContext context;
        context.check = check;
        context.expectedGeneration = &expectedGeneration;
        context.changedMessage = changedMessage;
        const auto error = placePreparedControl(placement, context);
        side.placed = placement.placed;
        if (!alive) {
            return refuse(changedMessage);
        }
        if (error) {
            return refuseKeeping(*error);
        }
        side.linkedControlIndex = placement.storedIndex;
    }

    // Step 5: the link, on the current records, in one checked commit.
    for (Side& side : sides) {
        side.fiber = readStoredFiber(side.endpoint->fiberId, true);
        if (!side.fiber) {
            return refuseKeeping(changedMessage);
        }
    }
    if (const auto error = linkStoredControlPoints(*sides[0].fiber, sides[0].linkedControlIndex,
                                                   *sides[1].fiber, sides[1].linkedControlIndex,
                                                   /*autoLink=*/true)) {
        return refuseKeeping(*error);
    }
    outcome.ok = true;
    outcome.message = tr("Linked %1 / CP %2 and %3 / CP %4 (%5; %6 on %1, %7 on %3)")
                          .arg(sides[0].name)
                          .arg(sides[0].linkedControlIndex)
                          .arg(sides[1].name)
                          .arg(sides[1].linkedControlIndex)
                          .arg(gapText,
                               sides[0].placed ? tr("new control point") : tr("existing control point"),
                               sides[1].placed ? tr("new control point") : tr("existing control point"));
    return outcome;
}

LineAnnotationController::AutoLinkChoiceSource
LineAnnotationController::choiceSourceFromSession(const LineAnnotationSession& session,
                                                  const std::vector<cv::Vec3d>* linePoints) const
{
    namespace la = vc3d::line_annotation;
    // The generic shape the shared builder takes: positions (base voxels),
    // the session's refs, each control's tags and each span's gap flag.
    std::vector<cv::Vec3d> controls;
    std::vector<std::vector<std::string>> tags;
    std::vector<bool> gaps;
    controls.reserve(session.controlPoints.size());
    for (const auto& control : session.controlPoints) {
        controls.push_back(control.volumePoint);
        tags.push_back(control.tags);
        gaps.push_back(la::spanIsGap(control.segmentToNext));
    }
    AutoLinkChoiceSource source;
    source.linePoints = linePoints;
    source.facts = la::autoLinkChoiceFacts(controls, session.branches, tags, gaps);
    return source;
}

void LineAnnotationController::handleGeneratedLinkNearbyFiber(const std::string& surfaceName,
                                                              uint64_t fiberIdB,
                                                              cv::Vec3f viewerPoint,
                                                              double anchorLinePosition,
                                                              uint64_t anchorLineRevision)
{
    // Phase A, synchronous in the slot: nothing is mutated here.
    const ContextMenuIdentity menuIdentity = _contextMenuIdentity;
    _contextMenuIdentity = ContextMenuIdentity{};
    if (!menuIdentity.valid || menuIdentity.surfaceName != surfaceName) {
        showError(tr("The menu's context is gone; choose Link again."));
        return;
    }
    if (menuIdentity.packageGeneration != _packageGeneration ||
        menuIdentity.attachmentSequence != _attachmentSequence ||
        menuIdentity.fiberGeneration != fiberDataGeneration() ||
        !_state || _state->currentVolume() != menuIdentity.volume) {
        showError(tr("The fibers or the view changed meanwhile; choose Link again."));
        return;
    }
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    const std::shared_ptr<LineAnnotationSession> owned = pane->session;
    LineAnnotationSession& session = *owned;
    if (session.fiberDeleted) {
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return;
    }
    for (const auto& other : _panes) {
        if (other.session && other.session->taskState == LineAnnotationSession::TaskState::Running) {
            showError(tr("Line optimization is already running."));
            return;
        }
    }
    if (_structuralEditInProgress) {
        showError(tr("A merge, split or link is in progress; try again once it has finished."));
        return;
    }
    if (session.lineRevision != anchorLineRevision) {
        showError(tr("The annotation changed; choose Link again."));
        return;
    }
    if (session.fiberId == 0 || session.fiberId == fiberIdB) {
        showError(tr("Select a marker of a different fiber."));
        return;
    }
    const auto a = readStoredFiber(session.fiberId);
    const auto b = readStoredFiber(fiberIdB);
    if (!a) {
        showError(tr("This annotation has not been saved yet; save it before linking."));
        return;
    }
    if (!b) {
        showError(tr("Could not find fiber %1 for the link.").arg(static_cast<qulonglong>(fiberIdB)));
        return;
    }
    if (const auto error = linkPairEligibilityError(*a, *b)) {
        showError(*error);
        return;
    }
    // The identity verified, the X's point is converted into the session's
    // canonical base voxels (the scale is part of that identity).
    const cv::Vec3f pointB = viewerPoint;
    if (!finitePoint(pointB)) {
        showError(tr("The marker has no finite position."));
        return;
    }
    DialogAutoLinkCapture capture;
    capture.packageGeneration = _packageGeneration;
    capture.attachmentSequence = _attachmentSequence;
    capture.volume = _state->currentVolume();
    capture.surfaceName = surfaceName;
    capture.session = owned;
    capture.fiberIdA = a->id;
    capture.lineRevision = session.lineRevision;
    capture.anchorLinePosition = anchorLinePosition;
    capture.extrapolationLimitVx =
        pane->dialog ? pane->dialog->maxControlPointExtrapolationDistanceVx() : 0;
    capture.fiberIdB = b->id;
    capture.pointB = toVec3d(pointB);
    // The structural guard is reserved here and handed to the callback (as
    // the merge does), so nothing structural can slip in between the enqueue
    // and the execution; the callback's scope guard releases it.
    _structuralEditInProgress = true;
    const QPointer<LineAnnotationController> self(this);
    QMetaObject::invokeMethod(
        this,
        [self, capture]() {
            if (!self) {
                return;
            }
            const auto release = qScopeGuard([self]() {
                if (self) {
                    self->_structuralEditInProgress = false;
                }
            });
            self->runDialogAutoLink(capture);
        },
        Qt::QueuedConnection);
}

void LineAnnotationController::runDialogAutoLink(const DialogAutoLinkCapture& c)
{
    namespace al = vc3d::fiber_map::autolink;
    const QPointer<LineAnnotationController> alive(this);
    const auto refuse = [&](const QString& message) {
        Logger()->warn("Line annotation auto-link refused: {}", message.toStdString());
        if (alive) {
            showError(message);
        }
    };
    const QString changedMessage = tr("The fibers changed meanwhile; nothing was linked.");
    const auto identityOnly = [&]() -> bool {
        return alive && _packageGeneration == c.packageGeneration &&
               _attachmentSequence == c.attachmentSequence && _state &&
               _state->currentVolume() == c.volume;
    };
    const auto paneIntact = [&]() {
        auto* pane = paneForSurface(c.surfaceName);
        return pane && pane->session == c.session;
    };
    if (!identityOnly()) {
        return refuse(changedMessage);
    }
    if (!paneIntact()) {
        return refuse(tr("The annotation was closed; nothing was linked."));
    }
    LineAnnotationSession& sessionA = *c.session;
    if (sessionA.lineRevision != c.lineRevision) {
        return refuse(tr("The annotation changed meanwhile; choose Link again."));
    }
    const QString nameA = fiberDisplayName(c.fiberIdA);
    const QString nameB = fiberDisplayName(c.fiberIdB);

    // The OTHER fiber's dialogs and dialog-less sessions are finalized,
    // saved and closed; this pane stays. Then the structural preflight.
    {
        const FiberMapAutoLinkOutcome prepared =
            prepareAutoLinkParticipants({c.fiberIdB}, c.surfaceName);
        if (!alive) {
            return;
        }
        if (!prepared.ok) {
            return refuse(prepared.message);
        }
    }

    // The guard, checked after every yield from here on. The expected
    // fiber generation is captured after the preparation and advanced only
    // over this operation's own commits.
    uint64_t expectedGeneration = fiberDataGeneration();
    const uint64_t saveFailures = _fiberSaveFailureCount;
    const uint64_t focusRevision = _state ? _state->focusBoundsRevision() : 0;
    // A's predicate by stage: the captured revision until A's geometric
    // mutation; while the placement tail runs either that or the control
    // already present and unlinked (the tail bumps the revision before its
    // manual-mode materialization can yield); afterwards the placed control
    // present and unlinked.
    enum class Stage { BeforePlacement, Placing, PlacedUnlinked };
    Stage stage = Stage::BeforePlacement;
    std::optional<cv::Vec3d> placedPointA;
    const auto placedControlUnlinked = [&]() {
        if (!placedPointA) {
            return false;
        }
        for (size_t i = 0; i < sessionA.controlPoints.size(); ++i) {
            if (pointsApproximatelyEqual(sessionA.controlPoints[i].volumePoint, *placedPointA)) {
                return !controlPointHasBranchLink(sessionA.branches, i);
            }
        }
        return false;
    };
    const auto check = [&](bool includeGeneration) -> bool {
        if (!identityOnly() || !paneIntact()) {
            return false;
        }
        if (includeGeneration && fiberDataGeneration() != expectedGeneration) {
            return false;
        }
        if (_fiberSaveFailureCount != saveFailures ||
            _state->focusBoundsRevision() != focusRevision) {
            return false;
        }
        auto* pane = paneForSurface(c.surfaceName);
        if (pane->dialog &&
            pane->dialog->maxControlPointExtrapolationDistanceVx() != c.extrapolationLimitVx) {
            return false;
        }
        switch (stage) {
        case Stage::BeforePlacement:
            if (sessionA.lineRevision != c.lineRevision ||
                sessionA.taskState == LineAnnotationSession::TaskState::Running) {
                return false;
            }
            break;
        case Stage::Placing:
            if (sessionA.lineRevision != c.lineRevision && !placedControlUnlinked()) {
                return false;
            }
            break;
        case Stage::PlacedUnlinked:
            if (!placedControlUnlinked()) {
                return false;
            }
            break;
        }
        for (const auto& other : _panes) {
            if (other.session && other.session != c.session &&
                (other.session->fiberId == c.fiberIdB ||
                 other.session->taskState == LineAnnotationSession::TaskState::Running)) {
                return false;
            }
        }
        return true;
    };
    if (!check(true)) {
        return refuse(changedMessage);
    }
    // B as it is now, base voxels; A over its live session.
    auto fiberB = readStoredFiber(c.fiberIdB, true);
    if (!fiberB || !readStoredFiber(c.fiberIdA)) {
        return refuse(changedMessage);
    }
    FiberMapAutoLinkParams params;
    if (!fiberInVolume(*fiberB)) {
        return refuse(tr("%1 is not bound to the active volume.").arg(nameB));
    }
    const double voxel = c.volume->voxelSize();
    if (!(voxel > 0.0) || !std::isfinite(voxel)) {
        return refuse(tr("The active volume has no usable voxel size."));
    }
    const std::vector<cv::Vec3d> lineA = linePointPositions(sessionA.optimizedLine);
    const auto anchorB = al::nearestLinePosition(fiberB->linePoints, c.pointB);
    if (!anchorB) {
        return refuse(tr("%1 has no line to link on.").arg(nameB));
    }
    std::array<AutoLinkChoiceSide, 2> choiceSides;
    choiceSides[0].name = nameA;
    choiceSides[0].source = choiceSourceFromSession(sessionA, &lineA);
    choiceSides[0].anchorPosition =
        std::clamp(c.anchorLinePosition, 0.0, static_cast<double>(std::max<size_t>(lineA.size(), 1) - 1));
    choiceSides[0].windowVx = params.searchWindowUm / voxel;
    choiceSides[0].reuseVx = params.reuseUm / voxel;
    choiceSides[1].name = nameB;
    choiceSides[1].source = choiceSourceFromStoredFiber(*fiberB);
    choiceSides[1].anchorPosition = *anchorB;
    choiceSides[1].windowVx = params.searchWindowUm / voxel;
    choiceSides[1].reuseVx = params.reuseUm / voxel;
    double passDistance = 0.0;
    if (const auto error = planAutoLinkChoices(choiceSides, &passDistance)) {
        return refuse(*error);
    }
    AutoLinkChoiceSide& sideA = choiceSides[0];
    AutoLinkChoiceSide& sideB = choiceSides[1];
    if (sideA.choice.kind == al::ControlChoice::Kind::Reuse &&
        sideB.choice.kind == al::ControlChoice::Kind::Reuse) {
        for (const FiberBranchRef& branch : sessionA.branches) {
            if (branch.branchFiberId == c.fiberIdB &&
                branch.controlPointIndex == sideA.choice.controlIndex &&
                branch.branchControlPointIndex == sideB.choice.controlIndex) {
                return refuse(tr("%1 / CP %2 and %3 / CP %4 are already linked.")
                                  .arg(nameA).arg(sideA.choice.controlIndex)
                                  .arg(nameB).arg(sideB.choice.controlIndex));
            }
        }
    }
    // A's placement limit (the manual handler's rule): checked on the chosen
    // position now and again right before A's mutation.
    const auto withinLimitA = [&]() {
        if (sideA.choice.kind != al::ControlChoice::Kind::Place || c.extrapolationLimitVx <= 0) {
            return true;
        }
        std::vector<double> controlLinePositions;
        for (const auto& control : sessionA.controlPoints) {
            controlLinePositions.push_back(control.linePosition);
        }
        return vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
            sideA.cumulative, sideA.choice.linePosition, controlLinePositions,
            static_cast<double>(c.extrapolationLimitVx));
    };
    if (!withinLimitA()) {
        return refuse(tr("The pass lies beyond the placement limit of this annotation."));
    }

    // Refusals after a control was committed say so.
    AutoLinkPlacement placementB;
    placementB.name = nameB;
    bool placedA = false;
    const auto refuseKeeping = [&](const QString& message) {
        QString kept;
        if (placementB.placed) {
            kept += tr(" The new control point placed on %1 was kept (unlinked).").arg(nameB);
        }
        if (placedA) {
            kept += tr(" The new control point placed on %1 was kept (unlinked).").arg(nameA);
        }
        refuse(message + kept);
    };

    // B: preparation then placement, before anything touches A.
    int linkedIndexB = -1;
    if (sideB.choice.kind == al::ControlChoice::Kind::Reuse) {
        linkedIndexB = sideB.choice.controlIndex;
    } else {
        placementB.session = makeIntersectionLineSession(
            *fiberB, sideB.choice.linePosition, cv::Vec3d{0.0, 0.0, 1.0},
            "fiber_auto_link_" + fiberB->uuid, nullptr);
        placementB.session->suppressErrorDialogs = false;
        placementB.choice = sideB.choice;
        if (!ensureSessionSolveBackends(*placementB.session, [&]() { return check(true); })) {
            return refuse(tr("Could not prepare the solve for %1.").arg(nameB));
        }
        if (!check(true)) {
            return refuse(changedMessage);
        }
        try {
            if (!prepareSessionLineModel(*placementB.session, fiberB->linePoints,
                                         [&]() { return check(true); })) {
                return refuse(tr("Could not prepare the line model of %1.").arg(nameB));
            }
        } catch (const std::exception& ex) {
            return refuse(tr("Could not prepare the line model of %1: %2")
                              .arg(nameB, QString::fromStdString(ex.what())));
        }
        if (!check(true)) {
            return refuse(changedMessage);
        }
        AutoLinkPlacementContext context;
        context.check = check;
        context.expectedGeneration = &expectedGeneration;
        context.changedMessage = changedMessage;
        const auto error = placePreparedControl(placementB, context);
        if (!alive) {
            return;
        }
        if (error) {
            return refuseKeeping(*error);
        }
        linkedIndexB = placementB.storedIndex;
    }

    // A: the interactive placement tail with an explicitly empty collapse
    // set, after the preparation yields and the limit re-check.
    size_t linkedIndexA = 0;
    if (sideA.choice.kind == al::ControlChoice::Kind::Reuse) {
        linkedIndexA = static_cast<size_t>(sideA.choice.controlIndex);
    } else {
        if (!prepareSessionForControlPlacement(c.surfaceName, c.session,
                                               [&]() { return check(true); })) {
            return refuseKeeping(tr("Could not prepare %1 for the placement.").arg(nameA));
        }
        if (!check(true)) {
            return refuseKeeping(changedMessage);
        }
        if (!withinLimitA()) {
            return refuseKeeping(tr("The pass lies beyond the placement limit of this annotation."));
        }
        const std::vector<cv::Vec3d> currentLinePoints = linePointPositions(sessionA.optimizedLine);
        const auto cumulative = vc3d::fiber_slice::cumulativePolylineArclengths(currentLinePoints);
        std::vector<double> controlLinePositions;
        for (const auto& control : sessionA.controlPoints) {
            controlLinePositions.push_back(control.linePosition);
        }
        GeometricControlPointEdit edit;
        edit.linePosition = sideA.choice.linePosition;
        edit.point = sideA.choice.point;
        placedPointA = sideA.choice.point;
        stage = Stage::Placing;
        const auto index = placeControlInSession(
            c.surfaceName, c.session, currentLinePoints, cumulative, controlLinePositions,
            std::move(edit), [&]() { return check(true); },
            [&]() {
                // The tail's own mutation and commits are done: the control
                // is placed, the expectation advances over them.
                stage = Stage::PlacedUnlinked;
                expectedGeneration = fiberDataGeneration();
            });
        if (!alive) {
            return;
        }
        if (paneIntact()) {
            for (size_t i = 0; i < sessionA.controlPoints.size(); ++i) {
                if (pointsApproximatelyEqual(sessionA.controlPoints[i].volumePoint,
                                             sideA.choice.point)) {
                    placedA = true;
                    linkedIndexA = i;
                    break;
                }
            }
        }
        if (!placedA || !index || *index != linkedIndexA) {
            return refuseKeeping(tr("Could not place a control point on %1.").arg(nameA));
        }
        stage = Stage::PlacedUnlinked;
        expectedGeneration = fiberDataGeneration();
        if (!check(true)) {
            return refuseKeeping(changedMessage);
        }
    }

    // The link: one checked commit of A's session and B's current record.
    if (linkedIndexB < 0 || linkedIndexA >= sessionA.controlPoints.size()) {
        return refuseKeeping(changedMessage);
    }
    if (const auto error = linkSessionControlToStoredControl(
            c.surfaceName, c.session, linkedIndexA, c.fiberIdB, linkedIndexB,
            /*adjacent=*/false, /*autoLink=*/true)) {
        return refuseKeeping(*error);
    }
    Logger()->info("Line annotation auto-link: linked {} / CP {} and {} / CP {} (3D gap {:.1f} vx; {} on {}, {} on {})",
                   nameA.toStdString(), linkedIndexA, nameB.toStdString(), linkedIndexB, passDistance,
                   placedA ? "new control point" : "existing control point", nameA.toStdString(),
                   placementB.placed ? "new control point" : "existing control point",
                   nameB.toStdString());
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
                                                         std::optional<std::pair<int, int>> spanControlIndices,
                                                         bool collectionInspection)
{
    // A debounced autosave of this fiber's still-open session must reach the
    // store first, or the replacement session would carry (and later re-save)
    // the pre-solve geometry over the newer state.
    flushAllPendingSessionAutoSaves();
    const auto readFiber = [&]() -> std::optional<StoredFiber> {
        try {
            auto fiber = readStoredFiber(fiberId, true);
            if (!fiber) showError(tr("Fiber %1 is not in the attached object store.").arg(fiberId));
            else if (!fiber->editorError.empty()) { showError(QString::fromStdString(fiber->editorError)); return std::nullopt; }
            return fiber;
        } catch (const std::exception& error) {
            showError(QString::fromUtf8(error.what()));
            return std::nullopt;
        }
    };
    if (!readFiber()) return;
    if (!ensureFiberInVolume(fiberId, true)) return;
    // Binding the coordinates publishes a new revision; open the current one.
    auto stored = readFiber();
    if (!stored) return;
    const StoredFiber& fiber = *stored;
    const std::optional<cv::Vec3d> seedOnlyPoint =
        vc3d::line_annotation::storedSinglePointFiberSeed(
            vc3d::line_annotation::storedControlPointPositions(fiber.controlPoints),
            fiber.linePoints);
    if (!seedOnlyPoint && fiber.linePoints.empty()) {
        showError(tr("Fiber %1 has no line points.").arg(fiberId));
        return;
    }
    const std::optional<std::array<std::size_t, 3>> coordinateBaseShapeZYX =
        fiber.coordinateBaseShapeZYX;

    auto session = std::make_shared<LineAnnotationSession>();
    session->suppressErrorDialogs = _errorDialogsSuppressed;
    session->fiberId = fiber.id;
    session->fiberUuid = fiber.uuid;
    session->storePin = fiber.storePin;
    session->fiberUsername = fiber.username;
    session->fiberStartedAt = fiber.startedAt;
    session->fiberSequence = fiber.sequence;
    session->fiberFileName = fiber.fileName;
    session->fiberManualHvTag = fiber.manualHvTag;
    session->fiberTags = fiber.tags;
    session->fiberWidth = fiber.width;
    session->fiberWidthGapFraction = fiber.widthGapFraction;
    session->fiberOptimizationMode = fiber.optimizationMode;
    session->modelOnly = fiber.modelOnly;
    session->coordinateBaseShapeZYX = coordinateBaseShapeZYX;
    session->branches = fiber.branches;
    session->disableInitialGeneratedHoverFollow =
        controlPointIndex.has_value() || linePointIndex.has_value() ||
        spanControlIndices.has_value();
    session->initialFitWholeLine = collectionInspection;
    session->readOnly = true;
    session->suppressFiberSave = true;
    session->editingAvailable = !fiber.readOnly;

    if (seedOnlyPoint) {
        // A single-point fiber has no line yet; enabling editing traces it.
        const cv::Vec3d seedPoint = *seedOnlyPoint;
        session->seedPoint = seedPoint;
        session->focusedLinePosition = 0.0;
        session->focusedControlPoint = seedPoint;
        session->initialDirectionMode = InitialDirectionMode::ZInOut;
        const cv::Vec3d sourceSliceNormal =
            seedTraceSourceNormalForStoredFiber(fiber, controlPointIndex, seedPoint);

        if (!ensureDatasetForSession(*session, false)) {
            return;
        }

        CChunkedVolumeViewer::CameraState camera;
        camera.scale = 1.0f;
        auto sourcePlane = std::make_shared<PlaneSurface>(
            toVec3f(seedPoint),
            toVec3f(sourceSliceNormal));
        (void)launchSession(SourceKind::Plane,
                            nextSurfaceName(),
                            std::move(sourcePlane),
                            camera,
                            sourceSliceNormal,
                            session);
        return;
    }

    const StoredFiber& reopeningFiber = fiber;
    session->focusedLinePosition = static_cast<double>(reopeningFiber.linePoints.size() / 2);
    session->focusedControlPoint = reopeningFiber.controlPoints.empty()
        ? std::optional<cv::Vec3d>{}
        : std::optional<cv::Vec3d>{reopeningFiber.controlPoints[reopeningFiber.controlPoints.size() / 2]};

    // Viewing never needs a dataset; without normals the strips use an
    // inspection frame until editing is enabled.
    if (!ensureDatasetForSession(*session, false)) {
        return;
    }

    try {
        if (session->modelOnly && !ensureFiberFollowService(session->suppressErrorDialogs)) return;
        session->inspectionFrameOnly = !session->normalSampler && !session->modelOnly;
        session->optimizedLine = session->modelOnly
            ? FiberFollowService::instance().lineModel(reopeningFiber.linePoints,
                _state->currentVolume()->voxelSize())
            : session->inspectionFrameOnly
            ? vc::lasagna::lineModelForInspection(reopeningFiber.linePoints)
            : lineModelFromPoints(reopeningFiber.linePoints, session->normalSampler.get());
        ++session->lineRevision;
        session->fiberMetricsMatchStoredFiber = true;
    } catch (const std::exception& ex) {
        showError(tr("Could not reopen fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())),
                  session->suppressErrorDialogs);
        return;
    }

    session->controlPoints.clear();
    session->controlPoints.reserve(reopeningFiber.controlPoints.size());
    double seedDistance = std::numeric_limits<double>::infinity();
    int seedControl = -1;
    for (size_t i = 0; i < reopeningFiber.controlPoints.size(); ++i) {
        const cv::Vec3d& controlPoint = reopeningFiber.controlPoints[i];
        int bestIndex = 0;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (size_t lineIndex = 0; lineIndex < reopeningFiber.linePoints.size(); ++lineIndex) {
            const cv::Vec3d delta = reopeningFiber.linePoints[lineIndex] - controlPoint;
            const double distance = std::sqrt(delta.dot(delta));
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = static_cast<int>(lineIndex);
            }
        }
        vc3d::line_annotation::LineControlPoint control;
        control.linePosition = static_cast<double>(bestIndex);
        control.volumePoint = controlPoint;
        control.optimizedIndex = bestIndex;
        control.segmentToNext = reopeningFiber.controlPoints[i].segmentToNext;
        control.tags = reopeningFiber.controlPoints[i].tags;
        control.displayNormal = reopeningFiber.controlPoints[i].displayNormal;
        control.direction = reopeningFiber.controlPoints[i].direction;
        control.storeId = reopeningFiber.controlPoints[i].storeId;
        control.displayNormalSource = reopeningFiber.controlPoints[i].displayNormalSource;
        session->controlPoints.push_back(control);

        const double centerDistance = std::abs(control.linePosition -
            static_cast<double>(reopeningFiber.linePoints.size() - 1) * 0.5);
        if (centerDistance < seedDistance) {
            seedDistance = centerDistance;
            seedControl = static_cast<int>(i);
        }
    }
    if (!session->controlPoints.empty()) {
        std::optional<std::pair<double, double>> spanLineRange;
        if (linePointIndex &&
            *linePointIndex >= 0 &&
            *linePointIndex < static_cast<int>(reopeningFiber.linePoints.size())) {
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
                static_cast<double>(reopeningFiber.linePoints.size() - 1));
            session->focusedControlPoint =
                interpolatedPointAtLinePosition(reopeningFiber.linePoints, session->focusedLinePosition);
            session->initialStripLinePositionRange = spanLineRange;
        }
    }
    if (_currentAtlasDir) {
        attachAtlasPredSnaps(reopeningFiber, *session, *_currentAtlasDir);
    }
    setSessionOptimizationState(*session, SessionOptimizationState::Optimized);

    CChunkedVolumeViewer::CameraState camera;
    camera.scale = 1.0f;
    const cv::Vec3d origin = session->focusedControlPoint.value_or(
        reopeningFiber.linePoints.empty()
            ? cv::Vec3d{0.0, 0.0, 0.0}
            : reopeningFiber.linePoints[static_cast<size_t>(std::clamp(
                  static_cast<int>(std::llround(session->focusedLinePosition)),
                  0,
                  static_cast<int>(reopeningFiber.linePoints.size() - 1)))]
    );
    const cv::Vec3d displayOrigin = origin;
    auto sourcePlane = std::make_shared<PlaneSurface>(
        cv::Vec3f{static_cast<float>(displayOrigin[0]),
                  static_cast<float>(displayOrigin[1]),
                  static_cast<float>(displayOrigin[2])},
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

vc3d::line_annotation::FiberDeleteOutcome LineAnnotationController::deleteFibers(
    std::vector<uint64_t> fiberIds)
{
    vc3d::line_annotation::FiberDeleteOutcome outcome;
    std::vector<std::string> uuids;
    for (const auto id : fiberIds) if (auto uuid = fiberUuidForId(id); !uuid.empty()) uuids.push_back(std::move(uuid));
    std::vector<StoredFiber> retired;
    try { retired = readStoredFibers(uuids); }
    catch (const std::exception& error) {
        outcome.error = error.what(); outcome.aborted = true;
        showError(tr("Could not retire fibers: %1").arg(QString::fromUtf8(error.what())));
        return outcome;
    }
    for (const auto& fiber : retired) outcome.requested.push_back({fiber.id,fiber.uuid});
    if (retired.empty()) return outcome;
    try {
        // Retirement removes every incident link in the same checked action.
        std::vector<FiberSaveSnapshot> noEdits;
        commitFiberStoreAction(noEdits,retired);
        for (const auto& fiber : retired) {
            outcome.deleted.push_back({fiber.id,fiber.uuid});
            outcome.deletedIds.push_back(fiber.id);
        }
        // Retired UUIDs remain in history; old workers cannot recreate them.
        refreshChangedStoreFibers();
        for (const auto& fiber : retired) emit fiberRetired(QString::fromStdString(fiber.uuid));
        emit fibersDeleted(outcome.deletedIds);
    } catch (const std::exception& error) {
        outcome.error = error.what(); outcome.aborted = true;
        refreshChangedStoreFibers();
        showError(tr("Could not retire fibers: %1").arg(QString::fromUtf8(error.what())));
    }
    return outcome;
}

void LineAnnotationController::renameFiberFile(uint64_t fiberId)
{
    const auto original = readStoredFiber(fiberId);
    if (!original) return;
    bool accepted = false;
    const auto name = QInputDialog::getText(_parentWidget.data(),tr("Rename Fiber"),tr("Name:"),
        QLineEdit::Normal,QString::fromStdString(fs::path(original->fileName).stem().string()),&accepted).trimmed();
    if (!accepted || name.isEmpty()) return;
    try {
        StoredFiber renamed = *original; renamed.fileName = name.toStdString()+".json";
        commitFibers({&renamed});
        // Names are labels; UUIDs and runtime handles survive a rename.
        for (const auto& pane : _panes) if (pane.session && pane.session->fiberId == fiberId) {
            pane.session->fiberFileName = renamed.fileName;
            if (pane.dialog) pane.dialog->setFiberDisplayName(fiberDisplayNameFromFileName(renamed.fileName));
        }
    } catch (const std::exception& error) {
        showError(tr("Could not rename fiber: %1").arg(QString::fromUtf8(error.what())));
    }
}

void LineAnnotationController::importFibers()
{
    if (!_state || !_state->objectStoreAttachment()) {
        showError(tr("Attach an object store before importing fibers."));
        return;
    }
    const auto options = showFiberJsonPathDialog(_parentWidget.data(), true, currentVolpkgRoot());
    if (!options) {
        return;
    }

    QString errorMessage;
    int imported = 0;
    int skipped = 0;
    if (!importFibersFromPath(options->path, &errorMessage,
                              &imported, &skipped)) {
        showError(errorMessage);
        return;
    }
    QMessageBox::information(_parentWidget.data(),
                             tr("Import Fibers"),
                             skipped > 0
                                 ? tr("Imported %1 fiber(s). Skipped %2 invalid JSON item(s).")
                                       .arg(imported)
                                       .arg(skipped)
                                 : tr("Imported %1 fiber(s).").arg(imported));
}

bool LineAnnotationController::importFibersFromPath(const fs::path& importPath,
                                                    QString* errorMessage,
                                                    int* importedCount,
                                                    int* skippedCount)
{
    if (importedCount) {
        *importedCount = 0;
    }
    if (skippedCount) {
        *skippedCount = 0;
    }
    if (_structuralEditInProgress) {
        if (errorMessage) {
            *errorMessage = tr("A merge, split or link is in progress; import once it has finished.");
        }
        return false;
    }
    if (!_state || !_state->objectStoreAttachment()) {
        if(errorMessage) *errorMessage=tr("Attach an object store before importing fibers.");
        return false;
    }
    try {
        using namespace vc::objects;
        const auto connection = _state->objectStoreSession();
        const auto attachment = *_state->objectStoreAttachment();
        std::vector<std::string> existing;
        connection->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            for (const auto& object : db.objectsByKind(txn, "fiber")) existing.push_back(object.uuid);
        });
        auto batch = prepareGeometrySource("fiber", importPath, existing);
        // Pin only the existing fibers the imported ones link to, so unrelated
        // edits in the meantime do not fail the import.
        RevisionSelection expected;
        connection->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            for (const auto& uuid : fiberBatchEndpoints(batch))
                if (const auto head = db.currentObjectRevision(txn, uuid)) expected[uuid] = head->uuid;
        });
        for (const auto& source : batch) expected[source.object.uuid] = std::nullopt;
        const auto count = batch.size();
        connection->act(attachment, expected, [&](auto& db, auto& txn) { ingestGeometryBatch(db, txn, std::move(batch)); });
        if (importedCount) *importedCount = int(count);
        return true;
    } catch (const std::exception& ex) {
        if (errorMessage) *errorMessage = tr("Could not import fibers: %1").arg(QString::fromUtf8(ex.what()));
        return false;
    }
}

uint64_t LineAnnotationController::importFiberJson(const nlohmann::json& root,
                                                   const std::string& fileName,
                                                   QString* errorMessage)
{
    try {
        if (_structuralEditInProgress) throw std::runtime_error("A merge or split is in progress");
        if (!_state || !_state->objectStoreAttachment())
            throw std::runtime_error("Attach an object store before importing fibers");
        auto decoded = vc::objects::decodeFiberJson(root.dump());
        const auto& source = decoded.source;
        if (!source.value("branches", nlohmann::json::array()).empty() ||
            !source.value("adjacent_branches", nlohmann::json::array()).empty())
            throw std::runtime_error("Linked fibers can only be imported with Import Fibers");
        const auto uuid = decoded.object.uuid;
        if (decoded.object.name.empty()) decoded.object.name = fs::path(fileName).stem().string();
        const auto connection = _state->objectStoreSession();
        connection->act(*_state->objectStoreAttachment(), {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
            vc::objects::ingestGeometryBatch(db, txn, {std::move(decoded)});
        });
        return fiberIdForUuid(uuid);
    } catch (const std::exception& ex) {
        if (errorMessage) *errorMessage = tr("Could not import fibers: %1").arg(QString::fromUtf8(ex.what()));
        return 0;
    }
}

void LineAnnotationController::exportFibers()
{
    // Publish pending session edits before exporting the attached store.
    flushAllPendingSessionAutoSaves();

    const fs::path defaultDir = currentVolpkgRoot();
    const auto options = showFiberJsonPathDialog(_parentWidget.data(), false, defaultDir);
    if (!options) {
        return;
    }

    QString errorMessage;
    int exported = 0;
    if (!exportFibersToPath(options->path, &errorMessage, &exported)) {
        showError(errorMessage);
        return;
    }
    QMessageBox::information(_parentWidget.data(),
                             tr("Export Fibers"),
                             tr("Exported %1 fiber(s) to %2.")
                                 .arg(exported)
                                 .arg(QString::fromStdString(options->path.string())));
}

bool LineAnnotationController::exportFibersToPath(const fs::path& exportPath,
                                                  QString* errorMessage,
                                                  int* exportedCount)
{
    // Publish pending edits before reading a consistent store snapshot.
    flushAllPendingSessionAutoSaves();
    if (exportedCount) {
        *exportedCount = 0;
    }

    try {
        const auto connection = _state->objectStoreSession();
        const auto attachment = _state->objectStoreAttachment();
        if (!attachment) throw std::runtime_error("Attach an object store before exporting fibers");
        nlohmann::json root{{"type", "vc3d_fiber_collection"}, {"version", 1},
                            {"point_collections", nlohmann::json::array()}};
        connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
            vc::objects::ObjectSelection objects; objects.kind = "fiber"; objects.includeRetired = true;
            for (const auto& [uuid, document] : vc::objects::fiberNetworkJson(db, txn, db.currentRevisions(txn, objects)))
                root["point_collections"].push_back(nlohmann::json::parse(document));
        });

        if (root["point_collections"].empty()) {
            if (errorMessage) *errorMessage = tr("There are no fibers to export.");
            return false;
        }
        writeJsonAtomic(exportPath, root);
        if (exportedCount) {
            *exportedCount = static_cast<int>(root["point_collections"].size());
        }
        return true;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage =
                tr("Could not export fibers: %1").arg(QString::fromStdString(ex.what()));
        }
        return false;
    }
}

void LineAnnotationController::setFiberManualHvTag(uint64_t fiberId, const QString& tag)
{
    const auto normalizedTag = vc3d::line_annotation::fiberHvTagToString(
        vc3d::line_annotation::fiberHvTagFromString(tag.toStdString()));
    const std::string manualTag = normalizedTag == "unknown" ? std::string{} : normalizedTag;

    auto fiber = readStoredFiber(fiberId);
    if (!fiber) {
        showError(tr("Fiber %1 is not in the attached object store.").arg(fiberId));
        return;
    }
    fiber->manualHvTag = manualTag;
    try {
        commitFibers({&*fiber});
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            pane.session->fiberManualHvTag = manualTag;
            pushFiberUiState(pane);
        }
    }
}

void LineAnnotationController::setFiberTag(uint64_t fiberId, const QString& tag, bool enabled)
{
    QString validationError;
    const auto normalizedTag = normalizedFiberTagInput(tag, &validationError);
    if (!normalizedTag) {
        showError(validationError);
        return;
    }
    auto fiber = readStoredFiber(fiberId);
    if (!fiber) {
        showError(tr("Fiber %1 is not in the attached object store.").arg(fiberId));
        return;
    }

    const std::vector<std::string> previousTags = fiber->tags;
    if (enabled) {
        addUniqueSorted(fiber->tags, *normalizedTag);
    } else {
        fiber->tags.erase(std::remove(fiber->tags.begin(), fiber->tags.end(), *normalizedTag),
                          fiber->tags.end());
    }
    if (fiber->tags == previousTags) {
        return;
    }

    try {
        commitFibers({&*fiber});
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }

    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            pane.session->fiberTags = fiber->tags;
            pushFiberUiState(pane);
        }
    }
}

QString LineAnnotationController::fiberHvDirectionTag(uint64_t fiberId) const
{
    const auto fiber = readStoredFiber(fiberId);
    if (!fiber) {
        return {};
    }
    if (fiber->manualHvTag == "H" || fiber->manualHvTag == "V") {
        return QString::fromStdString(fiber->manualHvTag);
    }
    const std::string automaticTag =
        vc3d::line_annotation::fiberHvTagToString(fiber->hvClassification.automaticTag);
    if (automaticTag == "H" || automaticTag == "V") {
        return QString::fromStdString(automaticTag);
    }
    return {};
}

void LineAnnotationController::pushFiberUiState(const PaneRecord& pane) const
{
    if (!pane.dialog || !pane.session) {
        return;
    }
    const uint64_t fiberId = pane.session->fiberId;
    pane.dialog->setFiberWidth(pane.session->fiberWidth, pane.session->fiberWidthGapFraction);
    pane.dialog->setFiberHvTag(fiberHvDirectionTag(fiberId));
    const auto stored = fiberId != 0 ? readStoredFiber(fiberId) : std::nullopt;
    pane.dialog->setFiberTags(knownFiberTags(),
                              stored ? stored->tags : pane.session->fiberTags,
                              stored.has_value() && !pane.session->readOnly);
}

void LineAnnotationController::recalculateFiberHvClassification(uint64_t fiberId)
{
    auto fiber = readStoredFiber(fiberId);
    if (!fiber) {
        showError(tr("Fiber %1 is not in the attached object store.").arg(fiberId));
        return;
    }
    fiber->hvClassification = vc3d::line_annotation::classifyFiberHv(
        vc3d::line_annotation::storedControlPointPositions(fiber->controlPoints));
    try {
        commitFibers({&*fiber});
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber %1: %2")
                      .arg(fiberId)
                      .arg(QString::fromStdString(ex.what())));
        return;
    }
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            pushFiberUiState(pane);
        }
    }
}

void LineAnnotationController::calculateFiberAlignmentMetrics(std::vector<uint64_t> orderedFiberIds)
{
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
        const fs::path atlasDir = createAtlasFromFiberCore(fiberId);
        emit atlasCreated(atlasDir);
    } catch (const std::exception& ex) {
        showError(tr("Could not create atlas: %1").arg(QString::fromStdString(ex.what())));
    }
}

bool LineAnnotationController::createAtlasFromFiberHeadless(uint64_t fiberId,
                                                            QString* errorMessage,
                                                            fs::path* atlasDirOut)
{
    // Do not emit atlasCreated: it is connected to the interactive display
    // path. Direct callers display the returned directory themselves.
    try {
        const fs::path atlasDir = createAtlasFromFiberCore(fiberId);
        if (atlasDirOut) {
            *atlasDirOut = atlasDir;
        }
        return true;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QString::fromStdString(ex.what());
        }
        return false;
    }
}

fs::path LineAnnotationController::createAtlasFromFiberCore(uint64_t fiberId)
{
    // Finish any reentrant publication before pinning the atlas inputs.
    flushAllPendingSessionAutoSaves();
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

    // Atlas coordinates are the open volume's voxels; the fiber is read in them.
    if (!ensureFiberInVolume(fiberId, true))
        throw std::runtime_error("Bind the fiber to the active volume before creating an atlas");
    const auto stored = readStoredFiber(fiberId, true);
    if (!stored) {
        throw std::runtime_error("Selected fiber is not available");
    }
    if (stored->linePoints.empty()) {
        throw std::runtime_error("Selected fiber has no line points");
    }

    const auto resolvedLasagna = resolveAlignmentMetricsDataset();
    if (!resolvedLasagna)
        throw std::runtime_error("No Lasagna dataset selected");
    const auto& dataset = *resolvedLasagna;
    const fs::path manifestPath = dataset.manifest().manifestPath;
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
    input.storePin=stored->storePin;
    input.fiberPath=relativeFiberPath(*stored);
    input.controlPoints =
        vc3d::line_annotation::storedControlPointPositions(stored->controlPoints);
    input.linePoints = stored->linePoints;
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
    const auto coordinateIdentity = coordinateIdentityForState(_state);
    vc3d::opendata::copyCoordinateIdentityToSurface(
        *selected.surface, coordinateIdentity);
    // The chosen init shell becomes a stored surface; the atlas pins the revision it maps onto.
    const auto attachment = _state->objectStoreAttachment();
    const auto volume = _state->currentVolume();
    if (!attachment || !volume) {
        throw std::runtime_error("Creating an atlas needs an attached object store and an open volume");
    }
    auto baseMeta = vc3d::coordinateBindingForVolume(*volume, 1);
    baseMeta["provenance"] = {{"tool", "vc3d_atlas"},
                              {"parameters", {{"init_shell", selected.path.string()}}},
                              {"sources", nlohmann::json::array()}};
    const auto basePin = vc::atlas::storeAtlasBaseSurface(
        *_state->objectStoreSession(), *attachment, *selected.surface,
        atlasDir.filename().string() + "_base", baseMeta.dump());
    auto atlas = vc::atlas::createSingleFiberAtlas(atlasDir.filename().string(),
                                                   volume->id(),
                                                   input,
                                                   basePin,
                                                   zeroWindingColumn,
                                                   std::move(mapping));
    copyCoordinateIdentityToJson(
        atlas.metadata.coordinateMetadata, coordinateIdentity);
    atlas.save(atlasDir);
    if (sampler.hasPredDtChannel() && !atlas.fibers.empty()) {
        (void)vc::atlas::ensureAtlasPredSnapSet(atlasDir,
                                                input,
                                                atlas.fibers.front(),
                                                *selected.surface,
                                                sampler);
    }
    return atlasDir;
}

void LineAnnotationController::addFiberToPointCollection(uint64_t fiberId)
{
    // A debounced autosave may not have reached the store yet.
    flushAllPendingSessionAutoSaves();
    const auto stored = readStoredFiber(fiberId, true);
    if (!stored) {
        showError(tr("Selected fiber is not available"));
        return;
    }
    if (stored->controlPoints.empty()) {
        showError(tr("Selected fiber has no control points"));
        return;
    }

    auto* col = _state ? _state->pointCollection() : nullptr;
    if (!col) {
        showError(tr("No point collection is available"));
        return;
    }

    StoredFiber display;
    try { display = fiberInVolumeCoordinates(*stored); }
    catch (const std::exception& error) { showError(QString::fromUtf8(error.what())); return; }
    const std::string name = col->generateNewCollectionName("fiber");
    const uint64_t collectionId = col->addCollection(name);

    CollectionMetadata meta;
    meta.absolute_winding_number = false;
    col->setCollectionMetadata(collectionId, meta);

    std::vector<cv::Vec3f> points;
    points.reserve(stored->controlPoints.size());
    for (const auto& cp : display.controlPoints) {
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

LineAnnotationController::StoredFiber LineAnnotationController::fiberInVolumeCoordinates(
    const StoredFiber& fiber) const
{
    ensureFiberLines(fiber);
    if (!fiberInVolume(fiber)) throw std::runtime_error("Bind the fiber to the active volume before using its geometry");
    return fiber;
}

void LineAnnotationController::showFiberSlice(uint64_t fiberId, QMdiArea* targetArea)
{
    namespace fslice = vc3d::fiber_slice;

    try {
        if (!_state || !_viewerManager || !targetArea) {
            throw std::runtime_error("Fiber slice workspace is not available");
        }
        const auto stored = readStoredFiber(fiberId, true);
        if (!stored) {
            throw std::runtime_error("Selected fiber is not in the attached object store");
        }
        if (stored->linePoints.empty()) {
            throw std::runtime_error("Selected fiber has no line points");
        }

        const auto displayFiber = fiberInVolumeCoordinates(*stored);
        const fslice::ControlSpanSelection span =
            fslice::selectControlSpan(
                displayFiber.linePoints,
                vc3d::line_annotation::storedControlPointPositions(displayFiber.controlPoints));
        if (!span.valid) {
            throw std::runtime_error(span.error);
        }
        const fslice::PlaneFit fit = fslice::fitLeastSquaresPlane(span, displayFiber.linePoints);
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
            // The slice shows the selected fiber's neighbourhood: every bound
            // store fiber within one fiber extent of it.
            const auto lineBox = boxAroundPoints(displayFiber.linePoints, 0.0);
            const double extent = std::max({lineBox[3] - lineBox[0], lineBox[4] - lineBox[1], lineBox[5] - lineBox[2], 0.0});
            const auto nearby = readStoredFibersInVolumeBox(boxAroundPoints(displayFiber.linePoints, extent), true);
            overlayData.fibers.reserve(nearby.size());
            for (const StoredFiber& fiber : nearby) {
                if (!fiber.editorError.empty() || !fiberInVolume(fiber)) continue;
                const auto display = fiberInVolumeCoordinates(fiber);
                overlayData.fibers.push_back(FiberSliceOverlayController::FiberData{
                    fiber.id,
                    display.linePoints,
                    vc3d::line_annotation::storedControlPointPositions(display.controlPoints),
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
    QString errorMessage;
    if (!showIntersectionInspectionHeadless(result, targetArea, std::move(atlasDir),
                                            &errorMessage)) {
        showError(tr("Could not show intersection inspection: %1").arg(errorMessage));
    }
}

// Dialog-free intersection inspection.
bool LineAnnotationController::showIntersectionInspectionHeadless(
    const vc::atlas::FiberIntersectionResult& result,
    QMdiArea* targetArea,
    std::optional<fs::path> atlasDir,
    QString* errorMessage)
{
    try {
        if (!_state || !_viewerManager || !targetArea) {
            throw std::runtime_error("Intersections workspace is not available");
        }
        cleanupIntersectionInspectionSurfaces();
        _intersectionInspection = std::make_unique<IntersectionInspectionSession>();
        _intersectionInspection->targetArea = targetArea;
        _intersectionInspection->suppressErrorDialogs = _errorDialogsSuppressed;
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
        return rebuildIntersectionInspection(errorMessage);
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QString::fromStdString(ex.what());
        }
        return false;
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
        const auto sourceFiber = readStoredFiber(result.sourceFiberId, true);
        const auto targetFiber = readStoredFiber(result.targetFiberId, true);
        if (!sourceFiber || !targetFiber) {
            throw std::runtime_error("One or both intersection fibers are not in the attached object store");
        }

        auto makeInput = [this, &volpkgRoot](const StoredFiber& fiber) {
            const auto display = fiberInVolumeCoordinates(fiber);
            vc::atlas::FiberInput input;
            input.storePin=fiber.storePin;
            input.fiberPath=relativeFiberPath(fiber);
            input.controlPoints =
                vc3d::line_annotation::storedControlPointPositions(display.controlPoints);
            input.linePoints = display.linePoints;
            vc::atlas::validateFiberInputControlPoints(input);
            return input;
        };
        const vc::atlas::FiberInput sourceInput = makeInput(*sourceFiber);
        const vc::atlas::FiberInput targetInput = makeInput(*targetFiber);

        vc::atlas::Atlas atlas = vc::atlas::Atlas::load(*_intersectionInspection->atlasDir, _state->objectStoreSession());
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

        const auto resolvedLasagna = resolveAlignmentMetricsDataset();
        if (!resolvedLasagna)
            throw std::runtime_error("No Lasagna dataset selected");
        const auto& dataset = *resolvedLasagna;
        vc::lasagna::LasagnaNormalSampler sampler(dataset);

        auto baseSurface = vc::atlas::loadAtlasBaseSurface(atlas, _state->objectStoreSession());
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
            vc3d::fiber_slice::samplePolylineAtArclength(sourceFiber->linePoints,
                                                         result.sourceArclength);
        const auto targetSample =
            vc3d::fiber_slice::samplePolylineAtArclength(targetFiber->linePoints,
                                                         result.targetArclength);
        if (!sourceSample.valid || !targetSample.valid) {
            throw std::runtime_error("Could not sample inspected arclengths on the stored fibers");
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
    // Surface notifications and window closes run arbitrary slots: the
    // controller and the inspection are re-checked after each, the area is
    // reached through its QPointer and the names are copied first.
    const QPointer<LineAnnotationController> alive(this);
    const QPointer<QMdiArea> area = _intersectionInspection->targetArea;
    if (area) {
        std::vector<std::string> oldSurfaces;
        for (QMdiSubWindow* subWindow : area->subWindowList()) {
            if (!subWindow) {
                continue;
            }
            const QString oldSurface = subWindow->property("vc_intersection_slice_surface").toString();
            if (!oldSurface.isEmpty()) {
                oldSurfaces.push_back(oldSurface.toStdString());
            }
        }
        for (const std::string& oldSurface : oldSurfaces) {
            if (!alive || !_state) {
                return;
            }
            _state->setSurface(oldSurface, nullptr);
        }
        if (!alive || !area) {
            return;
        }
        area->closeAllSubWindows();
        if (!alive || !_intersectionInspection) {
            return;
        }
    }
    if (_state) {
        const std::vector<std::string> names = _intersectionInspection->surfaceNames;
        for (const auto& name : names) {
            if (!alive || !_state || !_intersectionInspection) {
                return;
            }
            _state->setSurface(name, nullptr);
        }
    }
    if (!alive || !_intersectionInspection) {
        return;
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

void LineAnnotationController::closeIntersectionInspectionForRetiredFibers(
    const std::vector<uint64_t>& fiberIds)
{
    if (!_intersectionInspection) {
        return;
    }
    const auto sessionHoldsRetiredFiber =
        [&fiberIds](const std::shared_ptr<LineAnnotationSession>& session) {
            return session &&
                std::find(fiberIds.begin(), fiberIds.end(), session->fiberId) !=
                fiberIds.end();
        };
    if (!sessionHoldsRetiredFiber(_intersectionInspection->sourceLineSession) &&
        !sessionHoldsRetiredFiber(_intersectionInspection->targetLineSession)) {
        return;
    }
    cleanupIntersectionInspectionSurfaces();
    _intersectionInspection.reset();
}

void LineAnnotationController::closeDialogPanesForFibers(
    const std::vector<uint64_t>& fiberIds)
{
    // Collect first: closing mutates pane state through queued events.
    std::vector<std::string> surfaceNames;
    for (const auto& pane : _panes) {
        if (pane.session && pane.dialog &&
            std::find(fiberIds.begin(), fiberIds.end(), pane.session->fiberId) !=
                fiberIds.end()) {
            surfaceNames.push_back(pane.surfaceName);
        }
    }
    for (const auto& surfaceName : surfaceNames) {
        closeFiberWindowForSurface(surfaceName);
    }
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

bool LineAnnotationController::rebuildIntersectionInspection(QString* errorMessage)
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
            return true;  // nothing to rebuild; not a failure
        }
        auto* targetArea = _intersectionInspection->targetArea.data();
        if (!targetArea) {
            throw std::runtime_error("Intersections workspace is not available");
        }

        const auto result = _intersectionInspection->result;
        const auto sourceFiber = readStoredFiber(result.sourceFiberId, true);
        const auto targetFiber = readStoredFiber(result.targetFiberId, true);
        if (!sourceFiber || !targetFiber) {
            throw std::runtime_error("One or both intersection fibers are not in the attached object store");
        }
        const auto sourceDisplay = fiberInVolumeCoordinates(*sourceFiber);
        const auto targetDisplay = fiberInVolumeCoordinates(*targetFiber);
        auto finitePointCount = [](const std::vector<cv::Vec3d>& points) {
            return std::count_if(points.begin(), points.end(), fslice::isFinitePoint);
        };
        if (finitePointCount(sourceFiber->linePoints) < 2 ||
            finitePointCount(targetFiber->linePoints) < 2) {
            throw std::runtime_error("One or both intersection fibers have too few finite line points");
        }

        const fslice::ArclengthSample sourceSample =
            fslice::samplePolylineAtArclength(sourceDisplay.linePoints, result.sourceArclength);
        const fslice::ArclengthSample targetSample =
            fslice::samplePolylineAtArclength(targetDisplay.linePoints, result.targetArclength);
        if (!sourceSample.valid || !targetSample.valid) {
            throw std::runtime_error("Could not sample intersection arclengths on the stored fibers");
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
        sourceSide.fiber = &sourceDisplay;
        sourceSide.otherFiberId = targetFiber->id;
        sourceSide.sourceSide = true;
        sourceSide.focusLinePosition = sourceSample.linePosition;
        sourceSide.focusPoint = result.sourcePoint;
        sourceSide.tangent = sourceSample.tangent;
        sourceSide.editSessionName = "intersection_edit_source_" + std::to_string(_nextPaneId++);
        sourceSide.editSession = makeIntersectionLineSession(*sourceFiber,
                                                             sourceSample.linePosition,
                                                             sourceSample.tangent,
                                                             sourceSide.editSessionName,
                                                             sourceCallback);

        SideBuild targetSide;
        targetSide.fiber = &targetDisplay;
        targetSide.otherFiberId = sourceFiber->id;
        targetSide.sourceSide = false;
        targetSide.focusLinePosition = targetSample.linePosition;
        targetSide.focusPoint = result.targetPoint;
        targetSide.tangent = targetSample.tangent;
        targetSide.editSessionName = "intersection_edit_target_" + std::to_string(_nextPaneId++);
        targetSide.editSession = makeIntersectionLineSession(*targetFiber,
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
        attachAtlasPredSnapsForInspection(*sourceFiber, sourceSide.editSession);
        attachAtlasPredSnapsForInspection(*targetFiber, targetSide.editSession);

        const bool sourceIsH = vc3d::line_annotation::firstFiberDisplaysAsH(
            sourceFiber->hvClassification,
            sourceFiber->manualHvTag,
            targetFiber->hvClassification,
            targetFiber->manualHvTag,
            sourceFiber->id < targetFiber->id);
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
            std::vector<cv::Vec3d> nativePoints;
            for (const auto& point : side.editSession->optimizedLine.points) nativePoints.push_back(point.position);
            if (!prepareSessionLineModel(*side.editSession, nativePoints)) {
                throw std::runtime_error("Could not start the fiber-follow model for the intersection view.");
            }
            vc::lasagna::LineViewConfig sideConfig;
            sideConfig.controlPointLinePositions.reserve(
                side.editSession->controlPoints.size());
            for (const auto& control : side.editSession->controlPoints) {
                sideConfig.controlPointLinePositions.push_back(control.linePosition);
            }
            // Same orientation pin as the annotation views, so the inspection
            // strip's vertical does not flip between fibers or rebuilds.
            sideConfig.orientedPointNormals =
                orientedLineNormalsForSession(*side.editSession);
            side.lineViews = vc::lasagna::buildLineViewSurfaces(
                side.editSession->optimizedLine,
                sideConfig);
            Logger()->info("Intersection {} strip built from sampled normals: fiber={} points={} surface={} side_slice={}",
                           side.displayPrefix,
                           side.fiber ? side.fiber->id : 0,
                           side.fiber ? side.fiber->linePoints.size() : 0,
                           static_cast<const void*>(side.lineViews.lineSurface.get()),
                           static_cast<const void*>(side.lineViews.lineSideSlice.get()));
            side.triplet = vc3d::fiber_slice::selectControlTriplet(
                side.fiber->linePoints,
                vc3d::line_annotation::storedControlPointPositions(
                    side.fiber->controlPoints),
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
            {sourceFiber->id,
             sourceDisplay.linePoints,
             vc3d::line_annotation::storedControlPointPositions(sourceDisplay.controlPoints),
             sourceSide.style},
            {targetFiber->id,
             targetDisplay.linePoints,
             vc3d::line_annotation::storedControlPointPositions(targetDisplay.controlPoints),
             targetSide.style},
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
                    follow.controlPoints = vc3d::line_annotation::optimizerControlPoints(
                        followSide.editSession->controlPoints);
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
                                generatedPredSnapMarkers(session->controlPoints, session->predSnapSet);
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
                    if (_intersectionInspection) {
                        _intersectionInspection->generatedOverlayViewers.push_back(chunkedViewer);
                    }
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
                        sourceFiber->id,
                        targetFiber->id,
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
                                   side.triplet.nextLinePosition,
                                   &side.lineViews.stripPositionMap);
            }
            if (_intersectionInspection) {
                _intersectionInspection->generatedSurfaceContexts[side.stripSurfaceName] =
                    IntersectionInspectionSession::GeneratedSurfaceContext{
                        true,
                        side.sourceSide,
                        true,
                        false,
                        side.focusLinePosition,
                        side.lineViews.stripPositionMap,
                    };
            }
            const std::vector<cv::Vec3d> linePoints = side.fiber->linePoints;
            const std::vector<cv::Vec3f> lineUpVectors = side.lineViews.lineUpVectors;
            const vc::lasagna::LineStripPositionMap stripPositionMap =
                side.lineViews.stripPositionMap;
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
                 stripPositionMap,
                 session,
                 focus,
                 markerLinePositions]() {
                    if (!chunkedViewer || !session) {
                        return;
                    }
                    vc3d::line_annotation::GeneratedViews views;
                    views.linePoints = generatedLinePoints(linePoints);
                    views.lineUpVectors = lineUpVectors;
                    views.stripPositionMap = stripPositionMap;
                    views.controlPoints = controlMarkersForSession(*session);
                    views.branchLinePoints = generatedBranchLinePointsForSession(*session);
                    views.branchLinks = generatedBranchLinkMarkers(session->branches);
                    views.predSnapPoints =
                        generatedPredSnapMarkers(session->controlPoints, session->predSnapSet);
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
            if (_intersectionInspection) {
                _intersectionInspection->generatedOverlayViewers.push_back(chunkedViewer);
            }
            connect(chunkedViewer,
                    &CChunkedVolumeViewer::sendMouseMoveVolume,
                    this,
                    [this,
                     chunkedViewer,
                     stripPositionMap,
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
                                scenePoint,
                                &stripPositionMap);
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
                     stripPositionMap,
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
                                scenePoint,
                                &stripPositionMap);
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
        // Only a completed build may claim the current orientation; a throw above
        // leaves the previous epoch in place so the next refresh retries.
        _intersectionInspection->orientationEpoch = _orientationEpoch;
        return true;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QString::fromStdString(ex.what());
        } else {
            showError(tr("Could not show intersection inspection: %1")
                          .arg(QString::fromStdString(ex.what())));
        }
        return false;
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
        for (const auto& fiber : readStoredFibers(std::vector<std::string>{fiberUuidForId(sourceId), fiberUuidForId(targetId)}, true)) {
            const auto display = fiberInVolumeCoordinates(fiber);
            vc::atlas::FiberPolyline polyline;
            polyline.id = fiber.id;
            polyline.generation = fiber.generation;
            polyline.controlPoints =
                vc3d::line_annotation::storedControlPointPositions(display.controlPoints);
            polyline.points.reserve(fiber.linePoints.size());
            for (const auto& point : display.linePoints) {
                polyline.points.push_back(vc::atlas::FiberPoint{point, std::nullopt});
            }
            fibers.push_back(std::move(polyline));
        }
        if (fibers.size() != 2) {
            throw std::runtime_error("The edited intersection fibers are no longer both in the object store");
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
            const bool suppressErrorDialogs =
                _intersectionInspection->suppressErrorDialogs || _errorDialogsSuppressed;
            cleanupIntersectionInspectionSurfaces();
            _intersectionInspection.reset();
            if (suppressErrorDialogs) {
                showError(tr("The edited fiber pair no longer has an intersection result."),
                          true);
            } else {
                QMessageBox::warning(_parentWidget,
                                     tr("Intersections"),
                                     tr("The edited fiber pair no longer has an intersection result."));
            }
            return;
        }
        _intersectionInspection->result = results[*nearest];
        rebuildIntersectionInspection();
    } catch (const std::exception& ex) {
        showError(tr("Could not refresh intersection inspection: %1")
                      .arg(QString::fromStdString(ex.what())),
                  _intersectionInspection && _intersectionInspection->suppressErrorDialogs);
    }
}

void LineAnnotationController::saveOpenFibersCore()
{
    std::vector<std::string> surfaces;
    for (const auto& pane : _panes) if (pane.session) surfaces.push_back(pane.surfaceName);
    for (const auto& surface : surfaces) {
        auto* pane = paneForSurface(surface);
        while (pane && pane->session && pane->session->watcher) {
            pane->session->watcher->waitForFinished();
            finishOptimization(surface);
            pane = paneForSurface(surface);
        }
        if (pane && pane->session && !pane->session->suppressFiberSave)
            saveSessionAsFiber(*pane->session, false);
    }
    if (_state) _state->objectStoreSession()->finalize();
}

void LineAnnotationController::saveOpenFibers()
{
    saveOpenFibersCore();
    waitForFiberSaves();
}

void LineAnnotationController::saveOpenFibersHeadless(FiberSaveCompletion onFinished)
{
    auto batch = std::make_shared<FiberSaveBatchTracker>(std::move(onFinished));
    _activeFiberSaveBatch = batch;
    try {
        saveOpenFibersCore();
    } catch (const std::exception& ex) {
        batch->addError(QString::fromUtf8(ex.what()));
    }
    _activeFiberSaveBatch.reset();
    batch->finish();
}

LineAnnotationDialog* LineAnnotationController::mostRecentLineAnnotationDialog() const
{
    for (auto it = _panes.rbegin(); it != _panes.rend(); ++it) {
        if (it->dialog) {
            return it->dialog.data();
        }
    }
    return nullptr;
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
    // Every control gets its session-lifetime identity before it is first
    // published; the struct carries it through later edits.
    for (const auto& control : session.controlPoints) {
        if (control.identity == 0) {
            control.identity = _nextControlIdentity++;
        }
    }
    auto markers = generatedControlMarkers(session.controlPoints, session.branches);
    // Same-orientation links (H-H / V-V) get the orange warning palette; an
    // unclassified fiber on either side keeps the default H-V colors.
    const QString localHvTag = fiberHvDirectionTag(session.fiberId);
    if (!localHvTag.isEmpty()) {
        for (const auto& branch : session.branches) {
            if (branch.controlPointIndex < 0 ||
                static_cast<size_t>(branch.controlPointIndex) >= markers.size()) {
                continue;
            }
            if (fiberHvDirectionTag(branch.branchFiberId) != localHvTag) {
                continue;
            }
            auto& marker = markers[static_cast<size_t>(branch.controlPointIndex)];
            if (branch.pending) {
                marker.hasSameHvPendingLinks = true;
            } else {
                marker.hasSameHvBranches = true;
            }
        }
    }
    if (_linkCandidate && _linkCandidate->fiberId != 0 &&
        session.fiberId == _linkCandidate->fiberId) {
        for (size_t i = 0; i < session.controlPoints.size() && i < markers.size(); ++i) {
            if (pointsApproximatelyEqual(session.controlPoints[i].volumePoint,
                                         _linkCandidate->position)) {
                markers[i].isLinkCandidate = true;
                markers[i].isAdjacentLinkCandidate = _linkCandidate->adjacent;
                break;
            }
        }
    }
    // Arc lengths on the session's live line, the line the markers' positions
    // index, in DISPLAY units (the strip map and the dialog's line are built
    // from the display-scaled line, like marker.point above). The dialog may
    // still hold the previous line (a placement publishes its spliced
    // controls before any landing), so the overview bar takes distances from
    // here, never from the positions.
    {
        const auto& points = session.optimizedLine.points;
        std::vector<double> cumulative(points.size(), 0.0);
        for (size_t i = 1; i < points.size(); ++i) {
            const auto& a = points[i - 1].position;
            const auto& b = points[i].position;
            const bool finite = std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]) &&
                                std::isfinite(b[0]) && std::isfinite(b[1]) && std::isfinite(b[2]);
            cumulative[i] = cumulative[i - 1] + (finite ? cv::norm(b - a) : 0.0);
        }
        const double total = cumulative.empty() ? 0.0 : cumulative.back();
        for (size_t index = 0; index < markers.size(); ++index) {
            auto& marker = markers[index];
            if (cumulative.empty() || !std::isfinite(marker.linePosition)) {
                continue;
            }
            const double last = static_cast<double>(cumulative.size() - 1);
            const double p = std::clamp(marker.linePosition, 0.0, last);
            const size_t lower = static_cast<size_t>(std::floor(p));
            const size_t upper = std::min(lower + 1, cumulative.size() - 1);
            const double t = p - static_cast<double>(lower);
            marker.arcLength = cumulative[lower] * (1.0 - t) + cumulative[upper] * t;
            marker.lineArcLength = total;
            marker.lineRevision = session.lineRevision;
            // On the line: the control's own point (session space, before the
            // display scaling above) is where the line is at its position.
            if (index < session.controlPoints.size()) {
                const cv::Vec3d linePoint =
                    points[lower].position * (1.0 - t) + points[upper].position * t;
                const cv::Vec3d delta = session.controlPoints[index].volumePoint - linePoint;
                // The line passes through its controls as vertices; this
                // only has to absorb float rounding of the stored points.
                constexpr double kOnLineToleranceVx = 1.0e-2;
                marker.onLine = std::isfinite(delta[0]) && std::isfinite(delta[1]) &&
                                std::isfinite(delta[2]) &&
                                delta.dot(delta) <= kOnLineToleranceVx * kOnLineToleranceVx;
            }
        }
    }
    return markers;
}

bool LineAnnotationController::fibersShareHvDirection(uint64_t localFiberId,
                                                      uint64_t linkedFiberId) const
{
    // Same-orientation links (H-H / V-V) get the orange warning palette; an
    // unclassified fiber on either side keeps the default H-V colors.
    const QString localTag = fiberHvDirectionTag(localFiberId);
    return !localTag.isEmpty() && fiberHvDirectionTag(linkedFiberId) == localTag;
}

std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker>
LineAnnotationController::markLinkCandidateFiberIntersections(
    std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker> markers,
    const LineAnnotationSession* session) const
{
    static const std::vector<FiberBranchRef> kNoBranches;
    const std::vector<FiberBranchRef>& branches = session ? session->branches : kNoBranches;
    const uint64_t localFiberId = session ? session->fiberId : uint64_t{0};
    const uint64_t candidateFiberId =
        _linkCandidate ? _linkCandidate->fiberId : uint64_t{0};
    for (auto& marker : markers) {
        marker.isLinkCandidateFiber =
            candidateFiberId != 0 && marker.fiberId == candidateFiberId;
        marker.pendingBranchLink = false;
        marker.autoPendingBranchLink = false;
        marker.sameHvBranchLink = false;
        if (!marker.projectedBranchLink || !marker.connectorStart) {
            continue;
        }
        marker.sameHvBranchLink = fibersShareHvDirection(localFiberId, marker.fiberId);
        for (const auto& branch : branches) {
            // connectorStart carries the local endpoint position through a
            // double->float round trip; match with a coarse tolerance.
            if (branch.pending && branch.branchFiberId == marker.fiberId &&
                cv::norm(toVec3f(branch.controlPointPosition) -
                         *marker.connectorStart) <= 1.0e-3) {
                marker.pendingBranchLink = true;
                marker.autoPendingBranchLink = branch.autoLink;
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
    if (_linkCandidate->adjacent) {
        // Any pair may be linked: a tag can change and a new fiber has none
        // yet. The fiber map flags an adjacent link whose fibers are not one
        // H and one V.
        if (session.fiberId != 0 && session.fiberId == _linkCandidate->fiberId) {
            state.enabled = false;
            state.label = tr("Link with adjacent candidate (same fiber)");
        } else {
            state.enabled = true;
            state.label = tr("Link with adjacent candidate (%1)").arg(linkCandidateMenuName());
        }
        return state;
    }
    if (session.fiberId != 0 && session.fiberId == _linkCandidate->fiberId) {
        state.enabled = false;
        state.label = tr("Link with candidate (same fiber)");
    } else {
        state.enabled = true;
        state.label = tr("Link with candidate (%1)").arg(linkCandidateMenuName());
    }
    return state;
}

vc3d::line_annotation::GeneratedLinkCandidateMenuState
LineAnnotationController::newLinkedToCandidateMenuState() const
{
    vc3d::line_annotation::GeneratedLinkCandidateMenuState state;
    if (!_linkCandidate || _linkCandidate->fiberId == 0) {
        return state;
    }
    state.enabled = true;
    state.label = (_linkCandidate->adjacent
                       ? tr("New line annotation - linked to adjacent candidate (%1)")
                       : tr("New line annotation - linked to candidate (%1)"))
                      .arg(linkCandidateMenuName());
    return state;
}

QString LineAnnotationController::fiberDisplayNameForId(uint64_t fiberId) const
{
    // Resolved when a menu opens, so renames are reflected without touching
    // any cached marker. Live sessions first: an unsaved fiber exists only there.
    std::string fileName;
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId) {
            fileName = pane.session->fiberFileName;
            break;
        }
    }
    if (fileName.empty()) {
        fileName = storedFiberFileName(fiberId);
    }
    return fiberDisplayNameFromFileName(fileName);
}

QString LineAnnotationController::linkCandidateMenuName() const
{
    if (!_linkCandidate || _linkCandidate->fiberId == 0) {
        return {};
    }
    const QString name = fiberDisplayNameForId(_linkCandidate->fiberId);
    const auto resolved = resolvedLinkCandidateControlIndex();
    if (!resolved) {
        return name;
    }
    // Both substitutions at once: a '%' inside the file stem must not be
    // re-interpreted by the second .arg().
    return tr("%1 / CP %2").arg(name, QString::number(resolved->storedControlIndex));
}

std::optional<LineAnnotationController::ResolvedLinkCandidate>
LineAnnotationController::resolvedLinkCandidateControlIndex() const
{
    if (!_linkCandidate || _linkCandidate->fiberId == 0) {
        return std::nullopt;
    }
    // Prefer a live editing session's geometry so enablement agrees with the
    // handlers, which bake such a session before acting on the candidate.
    for (const auto& pane : _panes) {
        if (!pane.session || pane.session->fiberId != _linkCandidate->fiberId) {
            continue;
        }
        const auto sessionIndex = sessionControlPointIndexByPosition(
            pane.session->controlPoints, _linkCandidate->position);
        if (!sessionIndex) {
            return std::nullopt;
        }
        const auto storedIndexMap =
            storedIndexMapForSessionControls(pane.session->controlPoints);
        if (static_cast<size_t>(*sessionIndex) >= storedIndexMap.size()) {
            return std::nullopt;
        }
        return ResolvedLinkCandidate{storedIndexMap[*sessionIndex],
                                     pane.session->controlPoints.size()};
    }
    const auto fiber = readStoredFiber(_linkCandidate->fiberId);
    if (!fiber) {
        return std::nullopt;
    }
    const auto candidateIndex = matchingStoredControlPointIndex(
        fiber->controlPoints,
        _linkCandidate->storedControlPointIndexHint,
        _linkCandidate->position);
    if (!candidateIndex) {
        return std::nullopt;
    }
    return ResolvedLinkCandidate{*candidateIndex, fiber->controlPoints.size()};
}

vc3d::line_annotation::GeneratedLinkCandidateMenuState
LineAnnotationController::mergeCandidateMenuState(const LineAnnotationSession& session) const
{
    vc3d::line_annotation::GeneratedLinkCandidateMenuState state;
    if (!_linkCandidate || _linkCandidate->fiberId == 0) {
        return state;
    }
    if (session.fiberId != 0 && session.fiberId == _linkCandidate->fiberId) {
        // Same-fiber refusal already surfaces on the link action.
        return state;
    }
    // Merging joins fiber endpoints; gate on the candidate side here, the
    // clicked side is validated in the handler.
    const auto resolved = resolvedLinkCandidateControlIndex();
    if (!resolved) {
        return state;
    }
    if (resolved->storedControlIndex != 0 &&
        static_cast<size_t>(resolved->storedControlIndex) + 1 != resolved->controlCount) {
        state.enabled = false;
        state.label = tr("Merge with candidate (not an endpoint)");
        return state;
    }
    if (_linkCandidate->adjacent) {
        // Merging joins two pieces of ONE fiber; an adjacent candidate is by
        // definition on another winding.
        state.enabled = false;
        state.label = tr("Merge with candidate (adjacent candidate)");
        return state;
    }
    state.enabled = true;
    state.label = tr("Merge with candidate (%1)").arg(linkCandidateMenuName());
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
        // What the deferred automatic link must still see (the menu runs a
        // nested loop; a volume switch inside it would otherwise go unseen).
        _contextMenuIdentity = ContextMenuIdentity{};
        _contextMenuIdentity.valid = true;
        _contextMenuIdentity.surfaceName = viewer->surfName();
        _contextMenuIdentity.packageGeneration = _packageGeneration;
        _contextMenuIdentity.attachmentSequence = _attachmentSequence;
        _contextMenuIdentity.fiberGeneration = fiberDataGeneration();
        _contextMenuIdentity.volume = _state ? _state->currentVolume() : nullptr;
        result = pane->dialog->showGeneratedControlPointContextMenu(
            viewer->surfName(),
            viewer,
            scenePoint,
            globalPos,
            linkCandidateMenuState(*pane->session),
            mergeCandidateMenuState(*pane->session),
            newLinkedToCandidateMenuState(),
            [this](uint64_t fiberId) { return fiberDisplayNameForId(fiberId); },
            pane->session->fiberId);
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
                scenePoint,
                &context.stripPositionMap);
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
        options.stripPositionMap = context.stripPositionMap;
        const auto candidateState = linkCandidateMenuState(*pane->session);
        options.linkWithCandidateEnabled = candidateState.enabled;
        options.linkWithCandidateLabel = candidateState.label;
        const auto mergeState = mergeCandidateMenuState(*pane->session);
        options.mergeWithCandidateEnabled = mergeState.enabled;
        options.mergeWithCandidateLabel = mergeState.label;
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
        options.newLinkedToCandidateLabel = newLinkedToCandidateMenuState().label;
        options.fiberDisplayNameForId = [this](uint64_t fiberId) {
            return fiberDisplayNameForId(fiberId);
        };
        options.newLineAnnotationLinkedToCandidate =
            [this, surfaceName = viewer->surfName()](cv::Vec3f volumePoint,
                                                     cv::Vec3f linkDirection) {
                handleGeneratedNewLineAnnotationLinkedToCandidate(surfaceName,
                                                                  volumePoint,
                                                                  linkDirection);
            };
        options.designateLinkCandidate = [this, surfaceName = viewer->surfName()](
                                             size_t controlPointIndex,
                                             cv::Vec3f volumePoint) {
            handleGeneratedControlPointLinkCandidate(surfaceName,
                                                     controlPointIndex,
                                                     volumePoint);
        };
        options.designateAdjacentLinkCandidate = [this, surfaceName = viewer->surfName()](
                                                     size_t controlPointIndex,
                                                     cv::Vec3f volumePoint) {
            handleGeneratedControlPointLinkCandidate(surfaceName,
                                                     controlPointIndex,
                                                     volumePoint,
                                                     /*adjacent=*/true);
        };
        options.linkWithCandidate = [this, surfaceName = viewer->surfName()](
                                        size_t controlPointIndex,
                                        cv::Vec3f volumePoint) {
            handleGeneratedControlPointLinkWithCandidate(surfaceName,
                                                         controlPointIndex,
                                                         volumePoint);
        };
        options.mergeWithCandidate = [this, surfaceName = viewer->surfName()](
                                         size_t controlPointIndex,
                                         cv::Vec3f volumePoint) {
            handleGeneratedControlPointMergeWithCandidate(surfaceName,
                                                          controlPointIndex,
                                                          volumePoint);
        };
        options.setSegmentInterpolationGoal = [this, surfaceName = viewer->surfName()](
                                                  size_t first, size_t second, std::string goal) {
            handleGeneratedSegmentInterpolationGoal(surfaceName, first, second, goal);
        };
        options.splitSpan = [this, surfaceName = viewer->surfName()](
                                size_t first, size_t second, bool linkHalves) {
            handleGeneratedSpanSplit(surfaceName, first, second, linkHalves);
        };
        options.setSpanGap = [this, surfaceName = viewer->surfName()](
                                 size_t first, size_t second, bool enabled) {
            handleGeneratedSpanSetGap(surfaceName, first, second, enabled);
        };
        options.setSpanDamaged = [this, surfaceName = viewer->surfName()](
                                     size_t first, size_t second, bool enabled) {
            handleGeneratedSpanSetDamaged(surfaceName, first, second, enabled);
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
        options.setKollesisTermination = [this, surfaceName = viewer->surfName()](
                                             size_t controlPointIndex, bool enabled) {
            handleGeneratedControlPointSetKollesisTermination(surfaceName,
                                                              controlPointIndex,
                                                              enabled);
        };
        options.setBreak = [this, surfaceName = viewer->surfName()](size_t controlPointIndex,
                                                                    bool enabled) {
            handleGeneratedControlPointSetBreak(surfaceName, controlPointIndex, enabled);
        };
        result = vc3d::line_annotation::showGeneratedControlPointContextMenu(options);
    }
    if (result == LineAnnotationDialog::GeneratedControlPointContextResult::NewLineAnnotationRequested) {
        launchFromViewerAtPoint(viewer, scenePoint, false);
    } else if (result == LineAnnotationDialog::GeneratedControlPointContextResult::NewModelLineAnnotationRequested) {
        launchFromViewerAtPoint(viewer, scenePoint, false, true);
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
        span.lengthVx = vc3d::fiber_slice::polylineLengthRange(fiber.linePoints,
                                                               span.firstLineIndex,
                                                               span.lastLineIndex);
        spans.push_back(span);
    }
    return spans;
}

const std::vector<LineAnnotationController::ControlSpanRecord>&
LineAnnotationController::cachedControlSpansForFiber(const StoredFiber& fiber) const
{
    auto cached = _controlSpanCache.find(fiber.id);
    if (cached == _controlSpanCache.end() ||
        cached->second.generation != fiber.generation ||
        cached->second.packageGeneration != _packageGeneration) {
        ControlSpanCacheEntry entry;
        entry.generation = fiber.generation;
        entry.packageGeneration = _packageGeneration;
        entry.spans = controlSpansForFiber(fiber);
        cached = _controlSpanCache.insert_or_assign(fiber.id, std::move(entry)).first;
    }
    return cached->second.spans;
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

std::shared_ptr<vc::lasagna::LasagnaDataset>
LineAnnotationController::resolveAlignmentMetricsDataset()
{
    if (!_state || !_state->vpkg()) {
        showError(tr("No volume package loaded."));
        return {};
    }

    auto vpkg = _state->vpkg();
    const auto openDataset = [&](const std::string& location, double scale) {
        vc::lasagna::LasagnaDatasetOpenOptions options;
        options.workingToBaseScale = scale;
        options.remoteCacheRoot = vc3d::remoteCachePathFs();
        const auto resolved = vc::project::isLocationRemote(location)
            ? location
            : vc::project::resolveLocalPath(location, vpkg->path().parent_path()).string();
        return std::make_shared<vc::lasagna::LasagnaDataset>(
            vc::lasagna::LasagnaDataset::openLocation(resolved, options));
    };
    try {
        if (const auto resolved = vc3d::opendata::resolveLasagnaForVolume(
                *vpkg, _state->currentVolumeId())) {
            // Marker-backed catalog entries retain their origin/auth policy on
            // local open; direct remote attachments must retain the source URL.
            return openDataset(resolved->manifestBacked
                                   ? resolved->manifestPath.string()
                                   : resolved->sourceManifestLocation,
                               resolved->workingToBaseScale);
        }
    } catch (const std::exception& ex) {
        showError(tr("Cannot resolve Lasagna for the active volume: %1")
                      .arg(QString::fromStdString(ex.what())));
        return {};
    }
    std::string selected = vpkg->selectedLasagnaDataset();

    const fs::path startDir = vpkg->path().empty()
        ? fs::path{}
        : vpkg->path().parent_path();
    // Suppressed callers cannot open the dataset picker.
    auto picked = (_datasetPicker && !_errorDialogsSuppressed)
        ? _datasetPicker(_parentWidget, startDir)
        : std::optional<std::string>{};
    if (!picked || picked->empty()) {
        if (_errorDialogsSuppressed) {
            showError(tr("No Lasagna dataset is selected for the active volume."));
        }
        return {};
    }
    selected = *picked;
    try {
        auto dataset = openDataset(selected, 1.0);
        vpkg->setSelectedLasagnaDataset(selected);
        return dataset;
    } catch (const std::exception& ex) {
        showError(tr("Cannot resolve Lasagna for the active volume: %1")
                      .arg(QString::fromUtf8(ex.what())));
        return {};
    }
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

    bool suppressErrorDialogs = _errorDialogsSuppressed;
    if (!suppressErrorDialogs) {
        suppressErrorDialogs = std::any_of(
            _panes.begin(), _panes.end(), [&fiberIds](const PaneRecord& pane) {
                return pane.session && pane.session->suppressErrorDialogs &&
                       std::find(fiberIds.begin(), fiberIds.end(), pane.session->fiberId) !=
                           fiberIds.end();
            });
    }

    std::vector<std::string> uuids;
    for (const uint64_t fiberId : fiberIds) {
        if (hasCachedAlignmentForFiber(fiberId) || isAlignmentPendingForFiber(fiberId)) {
            updateGeneratedViewMetricsForFiber(fiberId);
            continue;
        }
        if (auto uuid = fiberUuidForId(fiberId); !uuid.empty()) uuids.push_back(std::move(uuid));
    }
    std::vector<StoredFiber> fibers;
    try {
        fibers = readStoredFibers(uuids, true);
    } catch (const std::exception& error) {
        showError(tr("Could not read fibers for metrics: %1").arg(QString::fromUtf8(error.what())),
                  suppressErrorDialogs);
        return;
    }
    std::erase_if(fibers, [](const StoredFiber& fiber) { return !fiber.editorError.empty(); });
    std::vector<uint64_t> requestTokens;
    requestTokens.reserve(fibers.size());
    for (const auto& fiber : fibers) {
        _pendingFiberAlignmentMetrics.insert(fiber.id);
        const uint64_t requestToken = ++_nextFiberAlignmentMetricToken;
        _pendingFiberAlignmentMetricTokens[fiber.id] = requestToken;
        requestTokens.push_back(requestToken);
        publishPendingFiberAlignmentMetrics(fiber);
    }
    _fiberMetricsPending = !_pendingFiberAlignmentMetrics.empty();
    if (fibers.empty()) {
        return;
    }

    const auto dataset = resolveAlignmentMetricsDataset();
    if (!dataset) {
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
                                           suppressErrorDialogs,
                                           self,
                                           dataset,
                                           fibers = std::move(fibers),
                                           requestTokens = std::move(requestTokens)]() mutable {
        FiberMetricsTaskResult result;
        result.ok = true;
        result.suppressErrorDialogs = suppressErrorDialogs;
        result.generation = generation;
        result.manifestPath = dataset->manifest().manifestPath;
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
            vc::lasagna::LasagnaNormalSampler sampler(*dataset);
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
    const auto applyLegacyLasagnaMetrics = [&cached](auto& controls) {
        const size_t spanCount = std::min(
            cached.spans.size(), controls.empty() ? size_t{0} : controls.size() - 1);
        for (size_t index = 0; index < spanCount; ++index) {
            auto& segment = controls[index].segmentToNext;
            if (segment &&
                segment->interpMode ==
                    vc3d::line_annotation::SegmentInterpolationMode::Lasagna &&
                !segment->metric && cached.spans[index].available &&
                std::isfinite(cached.spans[index].maxErrorDegrees)) {
                segment->metric = cached.spans[index].maxErrorDegrees;
            }
        }
    };
    for (auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiberId)
            applyLegacyLasagnaMetrics(pane.session->controlPoints);
    }
    emit fiberAlignmentMetricsUpdated(fiberId, cached.fiber, cached.spans);
    updateGeneratedViewMetricsForFiber(fiberId);
}

void LineAnnotationController::publishPendingFiberAlignmentMetrics(const StoredFiber& fiber)
{
    CachedFiberAlignmentMetrics metrics;
    metrics.fiber.pending = true;
    metrics.spans.resize(cachedControlSpansForFiber(fiber).size());
    for (auto& spanMetric : metrics.spans) {
        spanMetric.pending = true;
    }
    emit fiberAlignmentMetricsUpdated(fiber.id, metrics.fiber, metrics.spans);
    updateGeneratedViewMetricsForFiber(fiber.id);
}

void LineAnnotationController::publishUnavailableFiberAlignmentMetrics(uint64_t fiberId)
{
    std::vector<FiberSummary::AlignmentMetrics> spanMetrics;
    if (const auto fiber = readStoredFiber(fiberId, true)) {
        spanMetrics.resize(cachedControlSpansForFiber(*fiber).size());
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

std::optional<LineAnnotationController::FiberSummary> LineAnnotationController::fiberSummary(uint64_t fiberId) const
{
    const auto uuid = fiberUuidForId(fiberId);
    if (uuid.empty()) return std::nullopt;
    auto summaries = fiberSummaries(std::vector<std::string>{uuid});
    if (summaries.empty()) return std::nullopt;
    return std::move(summaries.front());
}

std::vector<LineAnnotationController::FiberSummary> LineAnnotationController::fiberSummaries(
    const std::optional<std::vector<std::string>>& uuids) const
{
    const auto fibers = readStoredFibers(uuids, true);
    std::vector<FiberSummary> summaries;
    if (fibers.empty()) return summaries;
    // Fibers chained together through branch links, each one included.
    std::unordered_map<uint64_t, int> linkedFiberCounts;
    _state->objectStoreSession()->read(fibers.front().storePin->attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& fiber : fibers)
            if (!fiber.branches.empty())
                linkedFiberCounts[fiber.id] = static_cast<int>(db.componentOf(txn, fiber.uuid, {"fiber_branch"}).size());
    });
    summaries.reserve(fibers.size());
    for (const auto& fiber : fibers)
        summaries.push_back(summarizeStoredFiber(fiber, linkedFiberCounts[fiber.id]));
    return summaries;
}

LineAnnotationController::FiberSummary LineAnnotationController::summarizeStoredFiber(
    const StoredFiber& fiber, int linkedFiberCount) const
{
    std::vector<FiberSummary::SpanSummary> spanSummaries;
    const std::vector<ControlSpanRecord>& spans = cachedControlSpansForFiber(fiber);
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
        if (span.firstControlIndex >= 0 &&
            static_cast<size_t>(span.firstControlIndex) < fiber.controlPoints.size() &&
            fiber.controlPoints[span.firstControlIndex].segmentToNext) {
            const auto& segment =
                *fiber.controlPoints[span.firstControlIndex].segmentToNext;
            summary.interpMarker =
                vc3d::line_annotation::segmentInterpolationModeMarker(
                    segment.interpMode);
            if (vc3d::line_annotation::isAcceptedNativeTrace(segment)) {
                summary.fiberManifest = segment.fiberManifestLocation;
            }
        }
        spanSummaries.push_back(std::move(summary));
    }
    // The spans are cached per fiber generation; summing them is the
    // annotated length without another control-to-line scan. The same
    // rule as fiber_slice::annotatedLineLengthVx: without two finite
    // controls the whole line, otherwise the line between the outermost
    // ones - which is nothing when they all sit on one line point.
    const double lineLength = vc3d::line_annotation::fiberLineLengthVx(fiber.linePoints);
    double annotatedLength = 0.0;
    if (!spans.empty()) {
        for (const auto& span : spans) {
            annotatedLength += span.lengthVx;
        }
    } else if (std::count_if(fiber.controlPoints.begin(),
                             fiber.controlPoints.end(),
                             [](const cv::Vec3d& control) { return finitePoint(control); }) < 2) {
        annotatedLength = lineLength;
    }
    const int pendingLinkCount = static_cast<int>(
        std::count_if(fiber.branches.begin(),
                      fiber.branches.end(),
                      [](const FiberBranchRef& branch) { return branch.pending; }));
    return FiberSummary{
        fiber.id,
        fiber.fileName,
        static_cast<int>(fiber.controlPoints.size()),
        static_cast<int>(fiber.linePoints.size()),
        lineLength,
        annotatedLength,
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
        linkedFiberCount >= 2 ? linkedFiberCount : 0,
        pendingLinkCount,
        vc3d::line_annotation::deriveTraceState(fiber.optimizationMode,
                                                fiber.controlPoints),
    };
}

QString LineAnnotationController::umbilicusFingerprint() const
{
    return _state ? QString::fromStdString(_state->storedUmbilicus().token) : QString{};
}

LineAnnotationController::UmbilicusStatus
LineAnnotationController::umbilicusStatus() const
{
    if (!_state) {
        return {};
    }
    const auto umbilicus = _state->storedUmbilicus();
    if (umbilicus.error.empty()) {
        return {true, QString::fromStdString(umbilicus.name.empty() ? umbilicus.uuid : umbilicus.name)};
    }
    return {false, QString::fromStdString(umbilicus.error)};
}

std::string LineAnnotationController::fiberMapCoordinateSpace() const
{
    // Read off the tags directly, like annotationFrame(): the coordinate
    // space is a name, and coordinateIdentityForVolume() would withhold it
    // over absent physical metadata.
    if (!_state || !_state->vpkg() || _state->currentVolumeId().empty()) {
        return {};
    }
    try {
        if (const auto identity = vc3d::opendata::coordinateIdentityFromTags(
                _state->vpkg()->volumeTags(_state->currentVolumeId()))) {
            return identity->coordinateSpace;
        }
    } catch (...) {
    }
    return {};
}

LineAnnotationController::FiberMapSnapshot LineAnnotationController::fiberMapSnapshot() const
{
    FiberMapSnapshot snapshot;
    snapshot.generation = _fiberDataGeneration;

    // Fibers are annotated in a source frame the current volume may only
    // describe indirectly, so the resolution and the voxel counts both come from
    // one derivation — deriving them separately is what let a rebased store pair
    // a level-0 resolution with level-N counts.
    const auto frame = annotationFrame();
    snapshot.frame = frame;
    snapshot.voxelSizeUm = frame.voxelSizeUm;
    if (snapshot.voxelSizeUm) {
        Logger()->info(
            "Fiber map: annotation voxel size {:g} um, volume grid scaled by "
            "x{:g} into the annotation frame",
            *snapshot.voxelSizeUm,
            frame.factor);
    } else {
        Logger()->info("Fiber map: no voxel size available; the map will report "
                       "lengths in voxels");
    }

    // The scroll's z extent in that same frame, so it pairs with voxelSizeUm.
    if (frame.extentXyz[2] > 0.0) {
        snapshot.annotationZSlices =
            static_cast<int>(std::llround(frame.extentXyz[2]));
    }

    // Which catalog volume this is, for the map's winding sense.
    snapshot.coordinateSpace = fiberMapCoordinateSpace();

    // The map covers every store fiber bound to the current volume.
    std::vector<StoredFiber> fibers = readStoredFibers(std::nullopt, true);
    std::erase_if(fibers, [&](const StoredFiber& fiber) { return !fiber.editorError.empty() || !fiberInVolume(fiber); });
    std::unordered_set<uint64_t> boundIds;
    boundIds.reserve(fibers.size());
    for (const StoredFiber& fiber : fibers) boundIds.insert(fiber.id);

    snapshot.fibers.reserve(fibers.size());
    for (const StoredFiber& fiber : fibers) {
        const auto display = fiberInVolumeCoordinates(fiber);
        FiberMapFiber entry;
        entry.id = fiber.id;
        // The user part of the label comes from the file name, not the
        // username field: for some fibers the two disagree and the file name
        // is what the fiber panel shows.
        entry.fileName = fiber.fileName;
        const QString stem =
            vc3d::displayStemForFiberFile(QString::fromStdString(fiber.fileName));
        const int separator = stem.indexOf(QLatin1Char('_'));
        entry.label = QStringLiteral("%1-%2")
                          .arg(separator >= 0 ? stem.left(separator) : stem)
                          .arg(fiber.sequence);
        switch (vc::atlas::effectiveFiberHvTag(fiber.hvClassification, fiber.manualHvTag)) {
        case vc3d::line_annotation::FiberHvTag::H:
            entry.hvTag = 'H';
            break;
        case vc3d::line_annotation::FiberHvTag::V:
            entry.hvTag = 'V';
            break;
        case vc3d::line_annotation::FiberHvTag::Unknown:
        default:
            entry.hvTag = '?';
            break;
        }
        entry.controlPoints.assign(display.controlPoints.begin(), display.controlPoints.end());
        entry.linePoints = display.linePoints;
        for (auto& point : entry.controlPoints) point *= frame.factor;
        for (auto& point : entry.linePoints) point *= frame.factor;
        const size_t spanCount =
            fiber.controlPoints.empty() ? 0 : fiber.controlPoints.size() - 1;
        entry.tracedSegments.reserve(spanCount);
        for (size_t i = 0; i < spanCount; ++i) {
            entry.tracedSegments.push_back(vc3d::line_annotation::isAcceptedNativeTrace(
                fiber.controlPoints[i].segmentToNext));
        }
        entry.kollesisTerminations.reserve(fiber.controlPoints.size());
        for (const auto& control : fiber.controlPoints) {
            entry.kollesisTerminations.push_back(vc3d::line_annotation::hasControlPointTag(
                control.tags, vc3d::line_annotation::kKollesisTerminationTag));
        }
        entry.breaks.reserve(fiber.controlPoints.size());
        for (const auto& control : fiber.controlPoints) {
            entry.breaks.push_back(vc3d::line_annotation::hasControlPointTag(
                control.tags, vc3d::line_annotation::kBreakTag));
        }
        entry.gapSegments.reserve(spanCount);
        entry.damagedSegments.reserve(spanCount);
        for (size_t i = 0; i < spanCount; ++i) {
            entry.gapSegments.push_back(
                vc3d::line_annotation::spanIsGap(fiber.controlPoints[i].segmentToNext));
            entry.damagedSegments.push_back(
                vc3d::line_annotation::spanIsDamaged(fiber.controlPoints[i].segmentToNext));
        }
        for (const FiberBranchRef& branch : fiber.branches) {
            if (boundIds.count(branch.branchFiberId) == 0) {
                continue;
            }
            entry.links.push_back(FiberMapLink{branch.controlPointIndex,
                                               branch.branchFiberId,
                                               branch.branchControlPointIndex,
                                               branch.pending,
                                               branch.adjacent,
                                               branch.autoLink});
        }
        snapshot.fibers.push_back(std::move(entry));
    }

    // Which umbilicus resolves, which frame it is in, and what to say about it
    // are one question with several exits, answered by umbilicusForSnapshot()
    // so this function stays the data export it reads as.
    SnapshotUmbilicus umbilicus = umbilicusForSnapshot(frame);
    snapshot.umbilicusMessage = std::move(umbilicus.message);
    snapshot.umbilicusLabel = std::move(umbilicus.label);
    snapshot.umbilicusCenters = std::move(umbilicus.centers);
    return snapshot;
}

LineAnnotationController::SnapshotUmbilicus
LineAnnotationController::umbilicusForSnapshot(
    const vc3d::annotation::AnnotationFrame& frame) const
{
    SnapshotUmbilicus result;
    if (!_state) {
        return result;
    }
    auto umbilicus = _state->storedUmbilicus();
    if (!umbilicus.error.empty()) {
        result.message = QString::fromStdString(umbilicus.error);
        Logger()->warn("Fiber map: no usable umbilicus: {}", umbilicus.error);
        return result;
    }
    if (!vc3d::annotation::hasCoordinateFrame(frame)) {
        result.message = tr("The current volume has no annotation frame.");
        return result;
    }
    // Stored points are in the volume's voxels; fibers live in the annotation frame.
    for (auto& point : umbilicus.points) {
        point *= static_cast<float>(frame.factor);
    }
    std::sort(umbilicus.points.begin(), umbilicus.points.end(),
              [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] < b[2]; });
    result.centers = std::move(umbilicus.points);
    result.label = tr("umbilicus %1").arg(QString::fromStdString(umbilicus.name));
    return result;
}

std::string LineAnnotationController::storedFiberFileName(uint64_t fiberId) const
{
    const auto uuid = fiberUuidForId(fiberId);
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    std::string fileName;
    if (!uuid.empty() && attachment) {
        connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
            if (const auto object = db.objectByUuid(txn, uuid)) fileName = object->name + ".json";
        });
    }
    return fileName;
}

QString LineAnnotationController::fiberDisplayName(uint64_t fiberId) const
{
    const QString stem = vc3d::displayStemForFiberFile(QString::fromStdString(storedFiberFileName(fiberId)));
    return stem.isEmpty() ? tr("unnamed") : stem;
}

std::vector<std::string> LineAnnotationController::knownFiberTags() const
{
    // 'reviewed' leads the list (dialog pills follow the given order); the
    // store's fiber tags follow in sorted order.
    const std::string reviewedTag{vc3d::line_annotation::kReviewedTag};
    std::vector<std::string> tags{reviewedTag};
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment) return tags;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (auto& tag : db.objectTags(txn, std::string{"fiber"}))
            if (tag != reviewedTag) tags.push_back(std::move(tag));
    });
    return tags;
}

std::vector<LineAnnotationController::FiberSnapshotWithPath>
LineAnnotationController::fiberSnapshotsFromStorageWithPaths()
{
    std::vector<FiberSnapshotWithPath> snapshots;
    for (const auto& fiber : readStoredFibers(std::nullopt, true)) {
        if (!fiber.editorError.empty()) continue;
        vc::atlas::FiberPolyline geometry;
        geometry.id = fiber.id;
        geometry.generation = fiber.generation;
        geometry.storePin = fiber.storePin;
        geometry.controlPoints = vc3d::line_annotation::storedControlPointPositions(fiber.controlPoints);
        geometry.points.reserve(fiber.linePoints.size());
        for (const auto& point : fiber.linePoints)
            geometry.points.push_back(vc::atlas::FiberPoint{point, std::nullopt});
        // Bound fibers are read in the volume's voxels.
        geometry.coordinateKnown = fiberInVolume(fiber);
        snapshots.push_back({relativeFiberPath(fiber),std::move(geometry),fiber.id,fiber.hvClassification,fiber.manualHvTag,fiber.tags});
    }
    return snapshots;
}

std::optional<uint64_t> LineAnnotationController::fiberIdForAtlasPath(
    const fs::path& atlasFiberPath) const
{
    // Atlas fiber paths are <store UUID>/<object UUID>.
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment) return std::nullopt;
    const auto uuid = atlasFiberPath.filename().string();
    if (vc::atlas::atlasFiberPathKey(fs::path(attachment->storeUuid) / uuid) !=
        vc::atlas::atlasFiberPathKey(atlasFiberPath)) return std::nullopt;
    bool active = false;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto object = db.objectByUuid(txn, uuid);
        active = object && object->kind == "fiber" && !object->retired;
    });
    if (!active) return std::nullopt;
    return fiberIdForUuid(uuid);
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

void LineAnnotationController::refreshChangedStoreFibers()
{
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment) return;
    std::vector<uint64_t> changed;
    struct Reopen {
        bool editing = false;
        bool retry = false;
    };
    std::map<std::string, Reopen> reopen;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        auto stale = [&](const auto& pin) {
            if (pin->attachment != *attachment) return true;
            const auto head = db.currentObjectRevision(txn, pin->object.objectUuid);
            return !head || head->uuid != pin->objectRevision || head->record.retired;
        };
        // A fiber not yet committed (no revision) has nothing to be stale against.
        for (const auto& pane : _panes) if (pane.session && pane.session->storePin &&
                                           !pane.session->storePin->objectRevision.empty() &&
                                           stale(pane.session->storePin)) {
            changed.push_back(pane.session->fiberId);
            reopen[pane.session->fiberUuid] = {!pane.session->readOnly,
                                               pane.session->watcher || pane.session->solveQueue.hasPending()};
        }
    });
    if (!changed.empty()) discardFiberPanes(changed);
    for (const auto& [uuid, state] : reopen) {
        const auto id = fiberIdForUuid(uuid);
        if (!id || !ensureFiberInVolume(id, false)) continue;
        openFiber(id);
        // A pane being edited stays editable on the new revision.
        if (!state.editing || !enableFiberEditing(id) || !state.retry) continue;
        for (const auto& pane : _panes) if (pane.session && pane.session->fiberUuid == uuid) {
            startFiberModeOptimization(*pane.session, false);
            break;
        }
    }
    refreshBranchLineViews();
}

void LineAnnotationController::discardFiberPanes(const std::vector<uint64_t>& fiberIds)
{
    closeIntersectionInspectionForRetiredFibers(fiberIds);
    std::vector<PaneRecord> removed, kept;
    // Remove records before closing dialogs: close and surface signals re-enter.
    for (auto& pane : _panes) {
        const auto id = pane.session ? pane.session->fiberId : uint64_t(0);
        (std::find(fiberIds.begin(), fiberIds.end(), id) != fiberIds.end() ? removed : kept)
            .push_back(std::move(pane));
    }
    _panes = std::move(kept);
    for (auto& pane : removed) {
        if (pane.session) {
            if (pane.session->runningSolveCancel) pane.session->runningSolveCancel->store(true);
            pane.session->solveQueue.shutdown();
            if (pane.session->watcher) {
                auto* watcher = pane.session->watcher.data();
                disconnect(watcher, nullptr, this, nullptr);
                connect(watcher, &QFutureWatcher<OptimizationTaskResult>::finished, watcher, &QObject::deleteLater);
                pane.session->watcher = nullptr;
            }
        }
        if (pane.dialog) {
            pane.dialog->setCloseAfterFinalizationAllowed(true);
            pane.dialog->close();
        }
        if (_state) {
            _state->setSurface(pane.surfaceName, nullptr);
            if (pane.session) for (const auto& name : pane.session->generatedSurfaceNames)
                _state->setSurface(name, nullptr);
        }
    }
    invalidateSideStripQueries();
    _pendingSideStripIntersectionRequests.clear();
    _sideStripReuseCache.clear();
    _sideStripSessionSnapshotCache.clear();
}

void LineAnnotationController::onVolumePackageChanged(std::shared_ptr<VolumePkg> /*pkg*/)
{
    if (_intersectionInspection) {
        cleanupIntersectionInspectionSurfaces();
        _intersectionInspection.reset();
    }
    std::vector<uint64_t> ids{0};
    for (const auto& pane : _panes) if (pane.session) ids.push_back(pane.session->fiberId);
    discardFiberPanes(ids);
    _lastVolumeChangedId.clear();
    ++_packageGeneration;
    invalidateScrollUmbilicus();
    resetStoreDerivedState();
}

bool LineAnnotationController::placementAllowedByFocusBounds(
    const cv::Vec3d& point,
    bool suppressErrorDialogs) const
{
    const auto bounds = _state ? _state->activeFocusBounds() : std::nullopt;
    if (!bounds || contains_point(*bounds, point)) {
        return true;
    }
    showError(tr("The point is outside the active focus bounding box."),
              suppressErrorDialogs);
    return false;
}

void LineAnnotationController::handleLineSeed(const std::string& surfaceName,
                                              cv::Vec3f volumePoint,
                                              InitialDirectionMode directionMode,
                                              SeedOrigin seedOrigin,
                                              std::vector<std::string> seedTags)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."),
                  session.suppressErrorDialogs);
        return;
    }

    if (seedOrigin == SeedOrigin::NewPlacement &&
        !placementAllowedByFocusBounds(
            toVec3d(volumePoint), session.suppressErrorDialogs)) {
        return;
    }

    if (!ensureDatasetForSession(session)) {
        return;
    }

    session.nativeSeedTracePending = false;
    if (session.fiberOptimizationMode ==
        vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d) {
        const auto vpkg = _state ? _state->vpkg() : nullptr;
        const size_t fiberInferenceEntryCount = vpkg
            ? vpkg->fiberInferenceDatasetEntries().size()
            : 0;
        if (session.fiberOptimizationMode ==
                vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d &&
            (session.modelOnly || fiberFollowEnabled())) {
            if (!ensureFiberFollowService(session.suppressErrorDialogs, pane->dialog)) {
                return;
            }
            session.nativeSeedTracePending = true;
            Logger()->info("Line annotation new fiber will be traced by the fiber-follow model "
                           "after seed initialization");
        } else if (vc3d::line_annotation::shouldRunNativeSeedTrace(
                session.fiberOptimizationMode,
                vpkg && !vpkg->selectedFiberInferenceDataset().empty(),
                fiberInferenceEntryCount)) {
            if (!ensureFiberInferenceDatasetForSession(session)) {
                return;
            }
            session.nativeSeedTracePending = true;
            Logger()->info(
                "Line annotation new fiber will run native tracing after seed "
                "initialization");
        } else {
            Logger()->info(
                "Line annotation new fiber uses Lasagna seed initialization: "
                "no selected or unique fiber inference dataset is configured");
        }
    }

    session.initialDirectionMode = directionMode;
    ensureSessionFiberIdentity(session);
    const cv::Vec3d seedPoint(volumePoint[0], volumePoint[1], volumePoint[2]);
    session.seedPoint = seedPoint;
    session.focusedLinePosition = 0.0;
    session.focusedControlPoint = seedPoint;
    session.controlPoints = {{0.0, seedPoint, true, -1}};
    // Before startOptimization copies the controls into its task: the task's
    // result replaces session.controlPoints wholesale.
    session.controlPoints.front().tags = std::move(seedTags);
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    syncLinkedBranchMetadataAfterFiberModification(session);
    if (session.modelOnly) {
        session.nativeSeedTracePending = false;
        auto axis = normalizedOrZero(session.sourceSliceNormal);
        if (!finiteDirection(axis)) axis = {0.0, 0.0, 1.0};
        if (directionMode == InitialDirectionMode::Sideways) {
            axis = normalizedOrZero(axis.cross(cv::Vec3d{0.0, 0.0, 1.0}));
            if (!finiteDirection(axis)) axis = {1.0, 0.0, 0.0};
        }
        session.optimizedLine = {};
        for (const auto& p : {seedPoint - axis, seedPoint, seedPoint + axis}) {
            vc::lasagna::LinePoint point;
            point.position = p;
            session.optimizedLine.points.push_back(point);
        }
        session.controlPoints.front().linePosition = 1.0;
        session.controlPoints.front().optimizedIndex = 1;
        startFiberModeOptimization(session, true);
    } else {
        startOptimization(session);
    }
}

vc3d::line_annotation::PreparedControlPointEdit
LineAnnotationController::applyGeometricControlPointEdit(
    LineAnnotationSession& session,
    const std::vector<cv::Vec3d>& currentLinePoints,
    const std::vector<double>& cumulativeArclengths,
    const std::vector<double>& controlLinePositions,
    GeometricControlPointEdit edit)
{
    // Display-normal inheritance: a new control inside a region of corrected
    // normals takes the interpolated normal, with its provenance recorded.
    // Captured before inserting a zero-offset CP or splicing the line, on
    // the current session geometry (not a display cache from before recent
    // edits).
    if (!edit.displayNormal && edit.nearbyControlIndices.empty() &&
        std::any_of(session.controlPoints.begin(), session.controlPoints.end(),
                    [](const auto& cp) { return cp.displayNormal.has_value(); })) {
        const auto field = displayFieldForSession(session, orientedLineNormalsForSession(session));
        std::vector<double> controlArcs;
        controlArcs.reserve(controlLinePositions.size());
        for (const double position : controlLinePositions)
            controlArcs.push_back(vc3d::fiber_slice::arclengthAtLinePosition(cumulativeArclengths, position));
        edit.displayNormal = vc::fiber_tracer::inheritedFiberDisplayNormal(field, controlArcs,
            vc3d::fiber_slice::arclengthAtLinePosition(cumulativeArclengths, edit.linePosition),
            edit.linePosition);
        edit.manualDisplayNormal = false;
    }

    // Purely geometric prepare: provisional display geometry only, in
    // microseconds and with NO GUI-thread chunk sampling - the freeze this
    // used to cause was up to four synchronous Ceres solves, and a residual
    // one was the splice resampling normals through the chunk cache.
    // Provisional normals are re-indexed/interpolated from the previous model
    // instead; the fiber-mode re-optimization the caller runs afterwards
    // rebuilds the dirty spans properly (per-span rollouts, native traces,
    // authoritative normals) and replaces this geometry when it lands.
    // Throws (std::runtime_error) before touching the session on a failed
    // prepare or splice.
    vc3d::line_annotation::PreparedControlPointEdit prepared =
        vc3d::line_annotation::prepareGeometricControlPointEdit(
            currentLinePoints,
            session.controlPoints,
            edit.nearbyControlIndices,
            edit.linePosition,
            edit.point,
            edit.segmentLength);
    vc::lasagna::LineModel preparedLine = prepared.lineReconstructed
        ? vc3d::line_annotation::spliceLineModelWithInterpolatedNormals(
              session.optimizedLine,
              prepared.linePoints,
              prepared.replacedStart,
              prepared.replacedCount)
        : session.optimizedLine;
    if (const auto focusBounds = _state ? _state->activeFocusBounds()
                                        : std::nullopt) {
        vc3d::line_annotation::constrainLineOpenTailsToBounds(
            preparedLine, prepared.controlPoints, *focusBounds);
    }

    // Apply display metadata before publishing or scheduling the geometric
    // edit, so solve snapshots and autosaves observe one complete operation.
    if (edit.displayNormal) {
        prepared.controlPoints[prepared.replacementIndex].displayNormal = edit.displayNormal;
        prepared.controlPoints[prepared.replacementIndex].displayNormalSource =
            edit.manualDisplayNormal ? "manual" : "interpolated";
    }
    if (edit.direction)
        prepared.controlPoints[prepared.replacementIndex].direction = edit.direction;
    // Placing beyond a lone end break: the new control is a break too, so
    // the span between them is a gap (closed to placement, bridged by a
    // cubic spline by the policy below) and the damaged papyrus past the
    // break is never traced. session.controlPoints is still the pre-edit
    // list here (the prepared snapshot already holds the inserted control).
    if (const auto tagIndex = vc3d::line_annotation::appendedControlToTagAsBreak(
            session.controlPoints, prepared)) {
        vc3d::line_annotation::setControlPointTag(
            prepared.controlPoints[*tagIndex].tags, vc3d::line_annotation::kBreakTag, true);
    }
    session.controlPoints = prepared.controlPoints;
    // A collapse can make two break points neighbours (or part them): the
    // span tags and the gap goal policy follow before any solve runs.
    vc3d::line_annotation::applyGapSpanPolicy(session.controlPoints);
    remapCollapsedBranchControlPointIndices(prepared.oldToNewIndices,
                                            session.branches);
    session.optimizedLine = std::move(preparedLine);
    ++session.lineRevision;
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    return prepared;
}

void LineAnnotationController::handleGeneratedControlPoint(const std::string& surfaceName,
                                                          cv::Vec3f volumePoint,
                                                          double linePosition,
                                                          std::optional<cv::Vec3f> lineAnchor,
                                                          std::optional<cv::Vec3d> displayNormal,
                                                          std::optional<cv::Vec3d> direction)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }

    // Owned for the duration: the preparation below yields, and the pane
    // is re-resolved by name afterwards (it can be erased meanwhile).
    const std::shared_ptr<LineAnnotationSession> owned = pane->session;
    auto& session = *owned;
    // Control-point edits deliberately do NOT refuse while a solve is in
    // flight: the session is mutated immediately, the in-flight solve is
    // superseded (epoch bump + cooperative cancel), and one solve over the
    // coalesced dirty spans follows. Only a seed solve still excludes edits,
    // via the empty-line check below.
    if (session.optimizedLine.points.empty() || session.controlPoints.empty()) {
        return;
    }
    if (!placementAllowedByFocusBounds(
            toVec3d(volumePoint), session.suppressErrorDialogs)) {
        return;
    }
    if (!prepareSessionForControlPlacement(surfaceName, owned, {})) {
        return;
    }
    pane = paneForSurface(surfaceName);
    if (!pane || pane->session != owned) {
        return;
    }

    const double maxPosition = static_cast<double>(session.optimizedLine.points.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
    std::vector<cv::Vec3d> currentLinePoints;
    currentLinePoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        currentLinePoints.push_back(point.position);
    }
    if (lineAnchor && std::isfinite((*lineAnchor)[0]) && std::isfinite((*lineAnchor)[1]) &&
        std::isfinite((*lineAnchor)[2])) {
        // A rapid second click arrives while the previous edit's geometric
        // splice is still waiting for its landing: its line position was
        // measured on the previously DISPLAYED line, and the splice may have
        // renumbered since. Resolve it on the session line through the
        // position's own 3D line point (index-continuous nearest match, the
        // same remap the landing applies to the pane position). Where the
        // frames agree the anchor sits exactly on the line and the position
        // is unchanged. The CLICKED point must not drive this: a click is
        // deliberately off the line, and where another pass of the same fiber
        // runs through the cut plane the click can be nearer to that pass,
        // which sent the control point thousands of vertices away -- or
        // collapsed it into that pass's control point -- with nothing
        // visible happening at the cut the user was looking at.
        linePosition = vc3d::line_annotation::remappedGeneratedLinePositionFromAnchor(
            currentLinePoints, toVec3d(*lineAnchor), linePosition);
    }
    const auto cumulativeArclengths =
        vc3d::fiber_slice::cumulativePolylineArclengths(currentLinePoints);
    std::vector<double> controlLinePositions;
    std::vector<bool> kollesisTerminations;
    controlLinePositions.reserve(session.controlPoints.size());
    kollesisTerminations.reserve(session.controlPoints.size());
    for (const auto& control : session.controlPoints) {
        controlLinePositions.push_back(control.linePosition);
        kollesisTerminations.push_back(vc3d::line_annotation::hasControlPointTag(
            control.tags, vc3d::line_annotation::kKollesisTerminationTag));
    }
    // A kollesis termination declares the fiber's end. The dialog refuses
    // such clicks itself (red marker), but the intersection-inspection strip
    // and cross-section clicks reach this handler directly, so the rule is
    // enforced here for every caller.
    if (vc3d::line_annotation::generatedLinePositionBeyondTaggedEnd(
            controlLinePositions, kollesisTerminations, linePosition)) {
        return;
    }
    // Nothing is placed inside a gap span (two consecutive break points)
    // either; the breaks have to be removed first. Same reason for checking
    // here: the intersection-inspection clicks bypass the dialog's gate.
    {
        std::vector<std::pair<double, double>> gapLineRanges;
        for (const auto& [lower, upper] :
             vc3d::line_annotation::gapSpansForControls(session.controlPoints)) {
            gapLineRanges.emplace_back(session.controlPoints[lower].linePosition,
                                       session.controlPoints[upper].linePosition);
        }
        if (vc3d::line_annotation::generatedLinePositionInsideGap(gapLineRanges, linePosition)) {
            return;
        }
    }
    if (pane->dialog &&
        pane->dialog->maxControlPointExtrapolationDistanceVx() > 0) {
        if (!vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
                cumulativeArclengths,
                linePosition,
                controlLinePositions,
                static_cast<double>(
                    pane->dialog->maxControlPointExtrapolationDistanceVx()))) {
            return;
        }
    }
    const cv::Vec3d clicked(volumePoint[0], volumePoint[1], volumePoint[2]);

    const std::vector<size_t> nearbyControlIndices =
        vc3d::fiber_slice::linePositionIndicesWithinArclengthRadius(
            cumulativeArclengths,
            linePosition,
            controlLinePositions,
            kMinimumControlPointSpacingBaseVoxels);
    if (!confirmLinkedControlPointEdits(session,
                                        nearbyControlIndices,
                                        tr("Replacing the nearby control points"))) {
        return;
    }
    // The replacement point takes the union of the collapsed points' tags; a
    // break and a kollesis termination may not meet on one point, so a click
    // that would collapse one of each is refused rather than dropping a tag.
    {
        std::vector<std::string> unionTags;
        for (const size_t index : nearbyControlIndices) {
            if (index < session.controlPoints.size()) {
                unionTags = vc3d::line_annotation::mergedControlPointTags(
                    unionTags, session.controlPoints[index].tags);
            }
        }
        if (vc3d::line_annotation::controlPointTagsConflict(unionTags)) {
            showError(tr("Cannot collapse a break point with a kollesis termination; "
                         "remove one of the tags first."),
                      session.suppressErrorDialogs);
            return;
        }
    }

    GeometricControlPointEdit edit;
    edit.linePosition = linePosition;
    edit.point = clicked;
    edit.nearbyControlIndices = nearbyControlIndices;
    edit.displayNormal = displayNormal;
    edit.manualDisplayNormal = displayNormal.has_value();
    edit.direction = direction;
    (void)placeControlInSession(surfaceName, owned, currentLinePoints, cumulativeArclengths,
                                controlLinePositions, std::move(edit), {});
}

bool LineAnnotationController::prepareSessionForControlPlacement(
    const std::string& surfaceName,
    const std::shared_ptr<LineAnnotationSession>& owned,
    const std::function<bool()>& stillValid)
{
    // The yields that precede a placement's geometry capture, in the order
    // the interactive handler always ran them: a debounced autosave still
    // holds the LAST landing's geometry and could never be saved once the
    // edit mutates the session (the flush refuses unoptimized sessions), so
    // it is persisted first; then the Lasagna dataset (a picker). After
    // each, lifetime, validity and the pane (by name) are re-checked: the
    // caller captures its geometry only after this returns true.
    const QPointer<LineAnnotationController> alive(this);
    LineAnnotationSession& session = *owned;
    const auto paneIntact = [this, &surfaceName, &owned]() {
        auto* pane = paneForSurface(surfaceName);
        return pane && pane->session == owned;
    };
    if (session.autoSaveScheduled) {
        flushSessionAutoSave(session.surfaceName);
        if (!alive || (stillValid && !stillValid()) || !paneIntact()) {
            return false;
        }
    }
    if (!ensureDatasetForSession(session, true, stillValid)) {
        return false;
    }
    if (!alive || (stillValid && !stillValid()) || !paneIntact()) {
        return false;
    }
    return true;
}

std::optional<size_t> LineAnnotationController::placeControlInSession(
    const std::string& surfaceName,
    const std::shared_ptr<LineAnnotationSession>& owned,
    const std::vector<cv::Vec3d>& currentLinePoints,
    const std::vector<double>& cumulativeArclengths,
    const std::vector<double>& controlLinePositions,
    GeometricControlPointEdit edit,
    const std::function<bool()>& stillValid,
    const std::function<void()>& mutated)
{
    // The placement tail shared by the interactive handler (which computed
    // edit.nearbyControlIndices from its click) and the automatic link
    // (which passes an explicitly empty collapse set): rollback capture,
    // the geometric prepare + splice, branch-metadata sync and saves, the
    // dialog overlays, the span-solve dispatch (auto mode) or the view
    // materialization (manual mode), debug json. Returns the index of the
    // placed (or replacing) control in the session; nullopt means either
    // that nothing was placed (the error was shown here) or that a validity
    // check failed AFTER the geometric mutation committed (a caller that
    // must know reads the control back by its position).
    const QPointer<LineAnnotationController> alive(this);
    auto* pane = paneForSurface(surfaceName);
    if (!pane || pane->session != owned) {
        return std::nullopt;
    }
    LineAnnotationSession& session = *owned;

    const std::vector<vc3d::line_annotation::LineControlPoint> previousControls =
        session.controlPoints;
    const std::vector<FiberBranchRef> previousBranches = session.branches;
    const cv::Vec3d previousSeedPoint = session.seedPoint;
    const double previousFocusedLinePosition = session.focusedLinePosition;
    const std::optional<cv::Vec3d> previousFocusedControlPoint =
        session.focusedControlPoint;
    const SessionOptimizationState previousOptimizationState =
        session.optimizationState;
    // The collapse rollback needs the pre-edit line; a multi-collapse can
    // only come from the nearby set, so the copy is taken only when one is
    // possible.
    std::optional<vc::lasagna::LineModel> previousLine;
    if (edit.nearbyControlIndices.size() > 1) {
        previousLine = session.optimizedLine;
    }
    const bool autoReoptimize =
        !pane->dialog ||
        pane->dialog->reoptimizationMode() ==
            LineAnnotationDialog::ReoptimizationMode::AutoReoptimize;
    // Both modes share the geometric prepare + splice (applyGeometricControlPointEdit):
    // the manual (no-reoptimize) branch used to update only the control
    // list, leaving the line without the moved control's vertex — an
    // in-memory state that violates the loader's exact-subset contract and
    // made superseded-solve merging impossible. Only the solve bookkeeping
    // at the end differs by mode.
    const auto updateStart = Clock::now();
    vc3d::line_annotation::PreparedControlPointEdit prepared;
    try {
        const int initialCenterlineLengthVx = pane->dialog
            ? std::max(2, pane->dialog->extrapolationDistanceVx() * 2)
            : vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX_DEFAULT * 2;
        edit.segmentLength = initialLineDiscretization(initialCenterlineLengthVx).segmentLength;
        prepared = applyGeometricControlPointEdit(session, currentLinePoints,
                                                  cumulativeArclengths,
                                                  controlLinePositions, edit);
    } catch (const std::exception& ex) {
        showError(tr("Could not update line control point: %1")
                      .arg(QString::fromStdString(ex.what())),
                  session.suppressErrorDialogs);
        return std::nullopt;
    }
    const bool editedExistingControl = !prepared.collapsedOldIndices.empty();
    const size_t collapsedControlCount = prepared.collapsedOldIndices.size();
    const std::vector<vc3d::line_annotation::LineControlPoint> branchRemapControls =
        prepared.controlPointsBeforeLineUpdate;
    const std::vector<FiberBranchRef> branchRemapBranches = session.branches;
    // A rollback captured by an earlier edit or mode/goal change restores a
    // state this edit is about to build on top of; applying it to a later
    // solve failure would silently discard this edit too. Each edit owns at
    // most its own.
    disarmRollbackSnapshots(session);
    session.optimizedLineBeforeModeChange.reset();
    session.branchesBeforeModeChange.reset();
    session.restoreFiberOptimizationModeOnFailure = false;
    if (collapsedControlCount > 1) {
        session.controlPointCollapseRollback =
            LineAnnotationSession::ControlPointCollapseRollback{
                previousControls,
                previousLine ? *previousLine : session.optimizedLine,
                session.optimizationReport,
                previousBranches,
                session.selectedManifestPath,
                previousSeedPoint,
                previousFocusedLinePosition,
                previousFocusedControlPoint,
                session.lineWasOptimized,
                session.fiberMetricsMatchStoredFiber,
                previousOptimizationState};
    }

    const auto& changed = session.controlPoints[prepared.replacementIndex];
    session.focusedLinePosition = changed.linePosition;
    session.focusedControlPoint = changed.volumePoint;
    if (changed.isSeed) {
        session.seedPoint = changed.volumePoint;
    }
    if (collapsedControlCount <= 1 && session.fiberId != 0 &&
        session.fiberMetricsMatchStoredFiber) {
        session.fiberMetricsMatchStoredFiber = false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
    }

    if (collapsedControlCount <= 1) {
        const BranchMetadataSyncResult branchSync =
            syncLinkedBranchMetadataAfterFiberModification(
                session,
                &branchRemapControls,
                &branchRemapBranches);
        scheduleBranchMetadataSaves(branchSync.affectedFiberIds, session.fiberId);
    } else if (!autoReoptimize) {
        // Auto mode defers the multi-collapse's reciprocal cleanup to the
        // solve landing; in manual mode no landing follows, so run the sync
        // now (the pre-unification manual path always did). The sync mutates
        // and saves LINKED fibers, which no local rollback can restore, so
        // the collapse rollback is disarmed with it.
        const BranchMetadataSyncResult branchSync =
            syncLinkedBranchMetadataAfterFiberModification(
                session, nullptr, &previousBranches);
        scheduleBranchMetadataSaves(branchSync.affectedFiberIds, session.fiberId);
        disarmRollbackSnapshots(session);
    }

    // The operation's own mutations (the splice, the metadata saves and
    // their generation bumps) are done: a caller advances its expectations
    // here, before the materialization below can yield.
    if (mutated) {
        mutated();
    }
    // Immediate feedback: the new control point appears in the overlays now;
    // the strip geometry itself follows when the asynchronous solve lands.
    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true,
            generatedSpanAlignmentMetricsForSession(session));
        pane->dialog->setGeneratedPredSnapPoints(
            generatedPredSnapMarkers(session.controlPoints, session.predSnapSet));
    }

    const std::string updateEventName = std::string(collapsedControlCount > 1
        ? "control_collapse_span_update"
        : (editedExistingControl ? "control_edit_span_update"
                                 : "control_add_span_update")) +
        (autoReoptimize ? "" : "_no_reopt");
    const double updateMs = elapsedMs(updateStart, Clock::now());
    Logger()->info("Line annotation Lasagna stage timing: event={} overall_ms={:.3f} points={}",
                   updateEventName,
                   updateMs,
                   session.optimizedLine.points.size());
    writeLineDebugJson(updateEventName,
                       session.controlPoints,
                       session.optimizedLine);
    saveSessionAsFiber(session, false);
    // Discard the superseded solve and requeue its work against this revision.
    if (session.runningSolveCancel) session.runningSolveCancel->store(true);
    session.solveQueue.noteSessionMutated();
    {
        const size_t spanCount = session.controlPoints.size() > 1
            ? session.controlPoints.size() - 1
            : 0;
        session.solveQueue.remapPendingDirty(prepared.oldToNewIndices,
                                             spanCount);
        if (!vc3d::line_annotation::OptimizationCoalescingQueue::remapDirtySpans(
                session.runningSolveDirtySegments,
                prepared.oldToNewIndices,
                spanCount)) {
            session.runningSolveFullLine = true;
            session.runningSolveDirtySegments.clear();
        }
        // Merge bookkeeping: compose the solve-start -> current control map
        // with this edit's renumbering, and record the edit's spans so the
        // landing keeps their provisional geometry.
        if (session.runningSolveControlMap) {
            auto& map = *session.runningSolveControlMap;
            bool valid = true;
            for (auto& entry : map) {
                if (entry >= prepared.oldToNewIndices.size()) {
                    valid = false;
                    break;
                }
                entry = prepared.oldToNewIndices[entry];
            }
            for (size_t k = 1; valid && k < map.size(); ++k) {
                valid = map[k - 1] <= map[k];
            }
            if (valid &&
                !vc3d::line_annotation::OptimizationCoalescingQueue::
                    remapDirtySpans(session.runningSolveEditedSpans,
                                    prepared.oldToNewIndices,
                                    spanCount)) {
                valid = false;
            }
            if (valid) {
                session.runningSolveEditedSpans.insert(
                    session.runningSolveEditedSpans.end(),
                    prepared.dirtySegmentIndices.begin(),
                    prepared.dirtySegmentIndices.end());
                std::sort(session.runningSolveEditedSpans.begin(),
                          session.runningSolveEditedSpans.end());
                session.runningSolveEditedSpans.erase(
                    std::unique(session.runningSolveEditedSpans.begin(),
                                session.runningSolveEditedSpans.end()),
                    session.runningSolveEditedSpans.end());
            } else {
                session.runningSolveControlMap.reset();
                session.runningSolveEditedSpans.clear();
            }
        }
    }
    session.solveQueue.addPending(prepared.dirtySegmentIndices, false);
    if (autoReoptimize) {
        scheduleSolveDispatch(session);
    } else if (!session.suppressGeneratedViews &&
               session.taskState != LineAnnotationSession::TaskState::Running) {
        // Manual mode has no landing to publish the spliced line into the
        // strip surfaces; rebuild the views now or every later click and
        // intersection query works against a display whose geometry no
        // longer matches the session. (While a superseded solve is still
        // running, its landing refreshes the views instead.)
        materializeGeneratedViews(session, stillValid);
        // The materialization can show a modal (a normal reset): lifetime
        // and validity first, then the pane by name.
        if (!alive || (stillValid && !stillValid())) {
            return std::nullopt;
        }
        pane = paneForSurface(surfaceName);
        if (!pane || pane->session != owned) {
            return std::nullopt;
        }
    }
    syncDialogLineSolveActivity(session);
    return prepared.replacementIndex;
}
std::optional<uint64_t> LineAnnotationController::createLinkedSeedFiber(
    const LinkedSeedParent& parent,
    const cv::Vec3d& seedPoint,
    const cv::Vec3d& requestedLinkDirection,
    const LineAnnotationSession& templateSession)
{
    const cv::Vec3d linkDirection = normalizedOrZero(requestedLinkDirection);
    const cv::Vec3d parentEndpointDirection =
        endpointTangentFromLinePoints(parent.linePoints, parent.point, linkDirection);

    // A one-control fiber at the clicked point; opening it runs the seed
    // initialization with the link direction as the slice normal.
    auto linkedSeedRecord = std::make_shared<LineAnnotationSession>();
    linkedSeedRecord->suppressErrorDialogs = templateSession.suppressErrorDialogs;
    linkedSeedRecord->fiberId = std::max(nextFiberId(), parent.fiberId + 1);
    linkedSeedRecord->sourceSliceNormal = linkDirection;
    linkedSeedRecord->initialDirectionMode = InitialDirectionMode::ZInOut;
    linkedSeedRecord->selectedDatasetLocation = templateSession.selectedDatasetLocation;
    linkedSeedRecord->selectedLasagnaManifestIdentity =
        templateSession.selectedLasagnaManifestIdentity;
    linkedSeedRecord->selectedManifestPath = templateSession.selectedManifestPath;
    linkedSeedRecord->workingToBaseScale = templateSession.workingToBaseScale;
    linkedSeedRecord->dataset = templateSession.dataset;
    linkedSeedRecord->normalSampler = templateSession.normalSampler;
    linkedSeedRecord->modelOnly = templateSession.modelOnly;
    linkedSeedRecord->coordinateBaseShapeZYX = templateSession.coordinateBaseShapeZYX;
    ensureSessionFiberIdentity(*linkedSeedRecord);

    // The seed keeps the view's requested link direction as its own
    // endpoint direction (a one-point child has no tangent).
    const FiberBranchRef parentToLinked = vc3d::line_annotation::makeBranchRef(
        parent.controlPointIndex, parent.point, parentEndpointDirection,
        linkedSeedRecord->fiberId, linkedSeedRecord->fiberFileName, 0, seedPoint, linkDirection,
        /*pending=*/true, parent.adjacent, /*autoLink=*/false);

    try {
        // Inside the try: with several live copies of the parent a partial
        // insertion must be rolled back like every later failure.
        parent.addRef(parentToLinked);
        StoredFiber parentFiber = parent.storedFiber();
        // The reciprocal ref must carry the parent's STORED control index; a
        // live session's index may differ, so read it back from the snapshot.
        const auto storedParentBranch = std::find_if(
            parentFiber.branches.begin(),
            parentFiber.branches.end(),
            [&parentToLinked](const FiberBranchRef& branch) {
                return branch.branchFiberId == parentToLinked.branchFiberId &&
                       branch.branchControlPointIndex == parentToLinked.branchControlPointIndex &&
                       pointsApproximatelyEqual(branch.controlPointPosition,
                                                parentToLinked.controlPointPosition);
            });
        if (storedParentBranch == parentFiber.branches.end()) {
            throw std::runtime_error("link entry missing after serialization");
        }

        const FiberBranchRef linkedToParent = vc3d::line_annotation::reciprocalBranchRef(
            *storedParentBranch, parentFiber.id, parentFiber.fileName);
        linkedSeedRecord->branches.push_back(linkedToParent);
        linkedSeedRecord->showLinkedLineOverlays = false;
        linkedSeedRecord->surfaceName = "linked_fiber_create_only";
        linkedSeedRecord->sourceAnnotationSurfaceName = linkedSeedRecord->surfaceName;
        linkedSeedRecord->seedPoint = seedPoint;
        linkedSeedRecord->focusedLinePosition = 0.0;
        linkedSeedRecord->focusedControlPoint = seedPoint;
        linkedSeedRecord->controlPoints = {{0.0, seedPoint, true, -1}};
        linkedSeedRecord->optimizedLine = syntheticLineModelFromPoints({seedPoint});
        linkedSeedRecord->taskState = LineAnnotationSession::TaskState::Succeeded;
        linkedSeedRecord->optimizationState = SessionOptimizationState::Optimized;

        StoredFiber linkedFiber = storedFiberFromSession(*linkedSeedRecord);
        const uint64_t linkedFiberId = linkedFiber.id;
        commitFibers({&parentFiber, &linkedFiber});
        return linkedFiberId;
    } catch (const std::exception& ex) {
        parent.rollback(parentToLinked);
        showError(tr("Could not save linked fiber: %1").arg(QString::fromStdString(ex.what())),
                  templateSession.suppressErrorDialogs);
        return std::nullopt;
    }
}

void LineAnnotationController::handleGeneratedNewLineAnnotationLinkedToCandidate(
    const std::string& surfaceName,
    cv::Vec3f volumePoint,
    cv::Vec3f requestedLinkDirection)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."), session.suppressErrorDialogs);
        return;
    }
    const cv::Vec3d seedPoint = toVec3d(volumePoint);
    if (!finitePoint(seedPoint)) {
        showError(tr("Could not determine a finite point for the new line annotation from the "
                     "clicked location."),
                  session.suppressErrorDialogs);
        return;
    }
    if (!placementAllowedByFocusBounds(seedPoint, session.suppressErrorDialogs)) {
        return;
    }
    const cv::Vec3d linkDirection = toVec3d(requestedLinkDirection);
    if (!finiteDirection(linkDirection)) {
        showError(tr("Could not determine a finite linked-fiber direction from the current view."),
                  session.suppressErrorDialogs);
        return;
    }
    if (!_linkCandidate || _linkCandidate->fiberId == 0) {
        showError(tr("No link candidate is designated."), session.suppressErrorDialogs);
        return;
    }
    // Cross-fiber operation: make sure no participating fiber's newest
    // geometry is still session-only behind a debounced autosave.
    flushAllPendingSessionAutoSaves();
    // The new fiber inherits this session's dataset and is opened right away.
    if (!ensureDatasetForSession(session)) {
        return;
    }
    const LinkCandidate candidate = *_linkCandidate;

    // The candidate's fiber may be live in several panes at once (the clicked
    // dialog session and a dialog-less inspection session, say) and only one
    // of them is saved by the open below: every live copy gets the ref.
    struct LiveParent {
        LineAnnotationSession* session = nullptr;
        bool hasDialog = false;
        int controlIndex = -1;
    };
    std::vector<LiveParent> liveParents;
    for (const auto& record : _panes) {
        if (record.session && record.session->fiberId == candidate.fiberId) {
            liveParents.push_back({record.session.get(), record.dialog != nullptr, -1});
        }
    }

    LinkedSeedParent parent;
    parent.adjacent = candidate.adjacent;
    if (!liveParents.empty()) {
        for (auto& live : liveParents) {
            if (live.session->taskState == LineAnnotationSession::TaskState::Running) {
                showError(tr("Line optimization is already running."),
                          session.suppressErrorDialogs);
                return;
            }
            const auto index = sessionControlPointIndexByPosition(
                live.session->controlPoints, candidate.position);
            if (!index) {
                _linkCandidate.reset();
                showError(tr("The link candidate control point no longer exists."),
                          session.suppressErrorDialogs);
                return;
            }
            if (controlPointHasBranchLink(live.session->branches,
                                          static_cast<size_t>(*index))) {
                showError(tr("The link candidate is already linked; unlink it first."),
                          session.suppressErrorDialogs);
                return;
            }
            live.controlIndex = *index;
        }
        const auto authoritativeIt = std::find_if(
            liveParents.begin(), liveParents.end(),
            [](const LiveParent& live) { return live.hasDialog; });
        const LiveParent authoritative =
            authoritativeIt != liveParents.end() ? *authoritativeIt : liveParents.front();
        parent.fiberId = authoritative.session->fiberId;
        parent.controlPointIndex = authoritative.controlIndex;
        parent.point = authoritative.session
                           ->controlPoints[static_cast<size_t>(authoritative.controlIndex)]
                           .volumePoint;
        parent.linePoints = linePointPositions(authoritative.session->optimizedLine);
        // The candidate CP was verified unlinked in every copy and the new
        // fiber id is fresh, so the ref cannot already be present.
        parent.addRef = [liveParents](const FiberBranchRef& ref) {
            for (const auto& live : liveParents) {
                FiberBranchRef sessionRef = ref;
                sessionRef.controlPointIndex = live.controlIndex;
                live.session->branches.push_back(sessionRef);
            }
        };
        parent.rollback = [liveParents](const FiberBranchRef& ref) {
            for (const auto& live : liveParents) {
                auto& branches = live.session->branches;
                branches.erase(
                    std::remove_if(branches.begin(), branches.end(),
                                   [&ref](const FiberBranchRef& branch) {
                                       return branch.branchFiberId == ref.branchFiberId &&
                                              branch.branchControlPointIndex ==
                                                  ref.branchControlPointIndex;
                                   }),
                    branches.end());
            }
        };
        parent.storedFiber = [this, authoritativeSession = authoritative.session]() {
            return storedFiberFromSession(*authoritativeSession);
        };
    } else {
        auto stored = readStoredFiber(candidate.fiberId, true);
        if (!stored) {
            _linkCandidate.reset();
            showError(tr("The link candidate's fiber no longer exists."),
                      session.suppressErrorDialogs);
            return;
        }
        // The parent's working copy: it is published together with the new
        // fiber, so the ref only needs to exist on this copy.
        const auto parentFiber = std::make_shared<StoredFiber>(std::move(*stored));
        const auto parentIndex = matchingStoredControlPointIndex(
            parentFiber->controlPoints,
            candidate.storedControlPointIndexHint,
            candidate.position);
        if (!parentIndex) {
            _linkCandidate.reset();
            showError(tr("The link candidate control point no longer exists."),
                      session.suppressErrorDialogs);
            return;
        }
        if (controlPointHasBranchLink(parentFiber->branches, static_cast<size_t>(*parentIndex))) {
            showError(tr("The link candidate is already linked; unlink it first."),
                      session.suppressErrorDialogs);
            return;
        }
        parent.fiberId = parentFiber->id;
        parent.controlPointIndex = *parentIndex;
        parent.point = parentFiber->controlPoints[static_cast<size_t>(*parentIndex)];
        parent.linePoints = parentFiber->linePoints;
        parent.addRef = [parentFiber](const FiberBranchRef& ref) {
            parentFiber->branches.push_back(ref);
        };
        parent.rollback = [parentFiber](const FiberBranchRef& ref) {
            parentFiber->branches.erase(
                std::remove_if(parentFiber->branches.begin(), parentFiber->branches.end(),
                               [&ref](const FiberBranchRef& branch) {
                                   return branch.branchFiberId == ref.branchFiberId &&
                                          branch.branchControlPointIndex ==
                                              ref.branchControlPointIndex;
                               }),
                parentFiber->branches.end());
        };
        parent.storedFiber = [parentFiber]() { return *parentFiber; };
    }
    if (!finitePoint(parent.point)) {
        showError(tr("Could not determine a finite position for the link candidate."),
                  session.suppressErrorDialogs);
        return;
    }

    const auto newFiberId = createLinkedSeedFiber(parent, seedPoint, linkDirection, session);
    if (!newFiberId) {
        refreshBranchLineViews(parent.fiberId);
        return;
    }
    _linkCandidate.reset();
    for (const auto& live : liveParents) {
        live.session->fiberMetricsMatchStoredFiber = true;
    }
    invalidateFiberAlignmentMetrics(parent.fiberId, true);
    refreshBranchLineViews(parent.fiberId);
    // Deferred: this runs inside the context-menu callback of the current
    // dialog, and opening finalizes, saves and closes that dialog. Value
    // capture only; the controller is the context object. A package switch
    // that lands in between must not open a fiber of the new package.
    QTimer::singleShot(0, this,
                       [this, fiberId = *newFiberId, packageGeneration = _packageGeneration]() {
                           if (_packageGeneration != packageGeneration) {
                               return;
                           }
                           // Just created to be traced: open it for editing.
                           openFiberAtControlPoint(fiberId, 0);
                           enableFiberEditing(fiberId);
                       });
}

void LineAnnotationController::handleGeneratedControlPointLinkCandidate(
    const std::string& surfaceName,
    size_t controlPointIndex,
    cv::Vec3f volumePoint,
    bool adjacent)
{
    (void)volumePoint;
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.fiberDeleted) {
        // The fiber was deleted underneath this session; a structural edit
        // would be persisted as a new file under the deleted name.
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return;
    }
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
    candidate.adjacent = adjacent;
    const auto storedIndexMap = storedIndexMapForSessionControls(session.controlPoints);
    if (controlPointIndex < storedIndexMap.size()) {
        candidate.storedControlPointIndexHint = storedIndexMap[controlPointIndex];
    }
    // The newest designation of a point wins: a split candidate on this
    // same point would otherwise keep drawing and the point would look stuck.
    _linkCandidate = candidate;

    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            false,
            generatedSpanAlignmentMetricsForSession(session));
    }
}

void LineAnnotationController::handleGeneratedOpenNearbyAnnotation(uint64_t fiberId,
                                                                   cv::Vec3f volumePoint)
{
    if (fiberId == 0) {
        return;
    }
    const auto fiber = readStoredFiber(fiberId, true);
    if (!fiber) {
        showError(tr("Could not find fiber %1 for the nearby annotation.")
                      .arg(static_cast<qulonglong>(fiberId)));
        return;
    }
    if (fiber->linePoints.empty() || !finitePoint(toVec3d(volumePoint))) {
        openFiber(fiberId);
        return;
    }
    const size_t linePointIndex =
        vc3d::fiber_slice::nearestLinePointIndex(fiber->linePoints, toVec3d(volumePoint));
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
    // Owned for the duration: the autosave flush below yields, and the pane
    // is re-resolved by name afterwards.
    const std::shared_ptr<LineAnnotationSession> owned = pane->session;
    auto& session = *owned;
    if (session.fiberDeleted) {
        // The fiber was deleted underneath this session; a structural edit
        // would be persisted as a new file under the deleted name.
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return;
    }
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    // Cross-fiber operation: make sure no participating fiber's newest
    // geometry is still session-only behind a debounced autosave.
    flushAllPendingSessionAutoSaves();
    pane = paneForSurface(surfaceName);
    if (!pane || pane->session != owned) {
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
    const auto farFiber = readStoredFiber(candidate.fiberId);
    if (!farFiber) {
        _linkCandidate.reset();
        showError(tr("The link candidate's fiber no longer exists."));
        return;
    }
    const auto farControlIndex = matchingStoredControlPointIndex(
        farFiber->controlPoints,
        candidate.storedControlPointIndexHint,
        candidate.position);
    if (!farControlIndex) {
        _linkCandidate.reset();
        showError(tr("The link candidate control point no longer exists."));
        return;
    }
    if (controlPointHasBranchLink(farFiber->branches,
                                  static_cast<size_t>(*farControlIndex))) {
        showError(tr("The link candidate is already linked; unlink it first."));
        return;
    }
    if (const auto error = linkSessionControlToStoredControl(
            surfaceName, owned, controlPointIndex, farFiber->id, *farControlIndex,
            candidate.adjacent, /*autoLink=*/false)) {
        showError(*error);
        return;
    }
    _linkCandidate.reset();
}

std::optional<QString> LineAnnotationController::linkSessionControlToStoredControl(
    const std::string& surfaceName,
    const std::shared_ptr<LineAnnotationSession>& owned,
    size_t controlPointIndex,
    uint64_t farFiberId,
    int farControlIndex,
    bool adjacent,
    bool autoLink)
{
    // The link of a SESSION control point to a stored fiber's control point,
    // shared by the link-candidate menu (adjacent or ordinary, human pending)
    // and the automatic link made from a fiber-intersection marker
    // (ordinary, automatic). The session-side ref carries the session index
    // (the stored index is read back from the serialized fiber); both
    // endpoints and their shared edge commit in one checked action, and on a
    // failure the session ref is removed again.
    LineAnnotationSession& session = *owned;
    auto farFiber = readStoredFiber(farFiberId, true);
    if (!farFiber) {
        return tr("The linked fiber no longer exists.");
    }
    if (controlPointIndex >= session.controlPoints.size() || farControlIndex < 0 ||
        static_cast<size_t>(farControlIndex) >= farFiber->controlPoints.size()) {
        return tr("A control point index is out of range.");
    }
    if (controlPointHasBranchLink(session.branches, controlPointIndex)) {
        return tr("This control point is already linked; unlink it first.");
    }
    if (controlPointHasBranchLink(farFiber->branches, static_cast<size_t>(farControlIndex))) {
        return tr("The other control point is already linked; unlink it first.");
    }
    const cv::Vec3d farPoint = farFiber->controlPoints[static_cast<size_t>(farControlIndex)];
    const cv::Vec3d localPoint = session.controlPoints[controlPointIndex].volumePoint;
    if (!finitePoint(localPoint) || !finitePoint(farPoint)) {
        return tr("Could not determine finite endpoints for the link.");
    }
    if (!fiberBoundToVolume(farFiber->id, _state ? _state->currentVolume() : nullptr)) {
        return tr("Bind the linked fiber to the active volume before linking.");
    }
    const cv::Vec3d fallbackDirection = normalizedOrZero(farPoint - localPoint);
    const cv::Vec3d localDirection = endpointTangentFromLinePoints(
        linePointPositions(session.optimizedLine),
        localPoint,
        fallbackDirection);
    const cv::Vec3d farDirection = endpointTangentFromLinePoints(
        farFiber->linePoints,
        farPoint,
        -fallbackDirection);
    if (!finiteDirection(localDirection) || !finiteDirection(farDirection)) {
        return tr("Could not determine finite link directions between the control points.");
    }

    // Session-indexed on this side; the stored index is read back from the
    // serialized fiber below.
    const FiberBranchRef localRef = vc3d::line_annotation::makeBranchRef(
        static_cast<int>(controlPointIndex), localPoint, localDirection,
        farFiber->id, farFiber->fileName, farControlIndex, farPoint, farDirection,
        /*pending=*/true, adjacent, autoLink);

    const auto sameLocalRef = [&localRef](const FiberBranchRef& branch) {
        return branch.branchFiberId == localRef.branchFiberId &&
               branch.controlPointIndex == localRef.controlPointIndex &&
               branch.branchControlPointIndex == localRef.branchControlPointIndex;
    };
    if (std::any_of(session.branches.begin(), session.branches.end(), sameLocalRef)) {
        return tr("These control points are already linked.");
    }
    session.branches.push_back(localRef);

    const auto refreshOverlays = [this, &surfaceName, &owned, &session]() {
        if (auto* pane = paneForSurface(surfaceName); pane && pane->session == owned && pane->dialog) {
            pane->dialog->setGeneratedBranchOverlayData(
                controlMarkersForSession(session),
                generatedBranchLinePointsForSession(session),
                generatedBranchLinkMarkers(session.branches),
                true,
                generatedSpanAlignmentMetricsForSession(session));
        }
    };
    const uint64_t localFiberId = session.fiberId;
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
        const FiberBranchRef reciprocal = vc3d::line_annotation::reciprocalBranchRef(
            *storedLocalBranch, localFiber.id, localFiber.fileName);
        const auto duplicateFar = std::find_if(
            farFiber->branches.begin(),
            farFiber->branches.end(),
            [&reciprocal](const FiberBranchRef& branch) {
                return branch.adjacent == reciprocal.adjacent &&
                       branch.branchFiberId == reciprocal.branchFiberId &&
                       branch.controlPointIndex == reciprocal.controlPointIndex &&
                       branch.branchControlPointIndex == reciprocal.branchControlPointIndex;
            });
        if (duplicateFar == farFiber->branches.end()) {
            farFiber->branches.push_back(reciprocal);
        }
        commitFibers({&localFiber, &*farFiber});
    } catch (const std::exception& ex) {
        std::erase_if(session.branches, sameLocalRef);
        refreshOverlays();
        return tr("Could not save linked fibers: %1").arg(QString::fromStdString(ex.what()));
    }

    session.fiberMetricsMatchStoredFiber = true;
    invalidateFiberAlignmentMetrics(localFiberId, true);
    invalidateFiberAlignmentMetrics(farFiberId, true);
    refreshOverlays();
    refreshBranchLineViews(localFiberId);
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Structural edits: merge two fibers into one, split one into two.
//
// Phase A (the menu callback) validates, captures the participants by
// identity and position, runs the one modal it needs, and schedules phase B
// on a clean stack. Phase B drains the save queue, re-resolves everything,
// plans on stored snapshots (LineAnnotationStructuralEdits.hpp), redirects
// the peers' links, canonicalizes and validates the whole batch, checks the
// graph the next load will see, and commits new objects, redirected peers and
// source retirements in one checked object-store action. Only after that commit
// is the prebuilt working state installed. A failed action leaves the database
// intact and refreshes the working views. The subsequent re-fit is a separate
// checked edit.
// ---------------------------------------------------------------------------

bool LineAnnotationController::objectStoreEditingAvailable(std::string* error) const
{
    if (_state && _state->objectStoreAttachment()) return true;
    if (error) *error = "Attach an object store before editing fibers";
    return false;
}

bool LineAnnotationController::structuralEditPreflight(uint64_t packageGeneration,
                                                       uint64_t attachmentSequence,
                                                       std::string* error)
{
    if (_packageGeneration != packageGeneration) {
        Logger()->warn("Structural edit abandoned: the package changed");
        return false;
    }
    if (_fiberSaveFailureCount != _fiberSaveFailureCountAtAttachment) {
        if (error) {
            *error = "a fiber save failed earlier in this session, so an open fiber may "
                     "differ from the store; reattach the object store first";
        }
        return false;
    }
    // Finalize accepted work before planning against the revision-backed views.
    flushAllPendingSessionAutoSaves();
    waitForFiberSaves();
    if (_packageGeneration != packageGeneration) {
        Logger()->warn("Structural edit abandoned: the package changed while saves drained");
        return false;
    }
    if (_attachmentSequence != attachmentSequence) {
        if (error) {
            *error = "the object store was reattached meanwhile";
        }
        return false;
    }
    if (_fiberSaveFailureCount != _fiberSaveFailureCountAtAttachment) {
        if (error) {
            *error = "a pending fiber save failed";
        }
        return false;
    }
    return true;
}

LineAnnotationController::StructuralEditOutcome
LineAnnotationController::commitStructuralEdit(StructuralEditRequest request, bool* committed)
{
    namespace la = vc3d::line_annotation;
    StructuralEditOutcome outcome;
    if (committed) {
        *committed = false;
    }
    const auto fail = [&outcome](std::string message) {
        outcome.error = std::move(message);
        return outcome;
    };
    const auto isOriginal = [&request](uint64_t id) {
        return std::any_of(request.originals.begin(), request.originals.end(),
                           [id](const StoredFiber& original) {
                               return la::sameFiberIdentity(id, original.id);
                           });
    };
    // The redirect sources point at the originals' snapshots held by this
    // request; bind them here so no caller keeps a pointer into a vector it
    // then moves.
    for (auto& source : request.sources) {
        const auto it = std::find_if(
            request.originals.begin(), request.originals.end(),
            [&source](const StoredFiber& original) {
                return la::sameFiberIdentity(source.id, original.id);
            });
        if (it == request.originals.end()) {
            return fail("internal: a redirect source is not one of the originals");
        }
        source.stored = &*it;
    }

    // 2. Peers: every stored fiber linking to an original gets a redirected
    //    copy; every open pane session of a non-participant likewise (its
    //    target index may be a session index, resolved by position).
    const auto readLinkedFibers = [this, &isOriginal](const auto& fibers, const auto& excluded) {
        std::vector<std::string> uuids;
        for (const auto& fiber : fibers)
            for (const auto& branch : fiber.branches)
                if (!isOriginal(branch.branchFiberId) && !excluded(branch.branchFiberId))
                    uuids.push_back(fiberUuidForId(branch.branchFiberId));
        return readStoredFibers(uuids, true);
    };
    std::vector<StoredFiber> peers;
    for (const auto& fiber : readLinkedFibers(request.originals, [](uint64_t) { return false; })) {
        const bool linksToOriginal = std::any_of(
            fiber.branches.begin(), fiber.branches.end(), [&](const FiberBranchRef& branch) {
                return std::any_of(request.originals.begin(), request.originals.end(),
                                   [&branch](const StoredFiber& original) {
                                       return branchReferencesFiber(branch, original.id);
                                   });
            });
        if (!linksToOriginal) {
            continue;
        }
        StoredFiber copy = fiber;
        std::string error;
        if (!la::redirectBranchRefs(copy.branches, request.sources, &error)) {
            return fail(fiberErrorName(fiber.fileName) + ": " + error);
        }
        copy.generation = std::max<uint64_t>(1, copy.generation + 1);
        peers.push_back(std::move(copy));
    }
    // Every session of a retired fiber is consumed, not only the ones the
    // caller planned from: an inspection session or a second pane of the
    // same fiber would otherwise stay saveable and recreate the file.
    std::vector<std::shared_ptr<LineAnnotationSession>> consumed;
    for (const auto& weak : request.consumedSessions) {
        if (auto session = weak.lock()) {
            consumed.push_back(std::move(session));
        }
    }
    for (const auto& pane : _panes) {
        if (pane.session && !pane.session->fiberDeleted &&
            isOriginal(pane.session->fiberId) &&
            std::find(consumed.begin(), consumed.end(), pane.session) == consumed.end()) {
            consumed.push_back(pane.session);
        }
    }
    for (const auto& session : consumed) {
        if (session->taskState == LineAnnotationSession::TaskState::Running) {
            return fail("a session of " + fiberErrorName(session->fiberFileName) +
                        " is still optimizing");
        }
    }
    // Two sessions of the same original (a dialog and an inspection pane,
    // say) may hold different unsaved versions; the plan used one of them,
    // so the other's edits would be silently discarded. Refuse unless they
    // agree on geometry, links and tags.
    for (std::size_t i = 0; i < consumed.size(); ++i) {
        for (std::size_t j = i + 1; j < consumed.size(); ++j) {
            auto& a = *consumed[i];
            auto& b = *consumed[j];
            if (!la::sameFiberIdentity(a.fiberId, b.fiberId)) {
                continue;
            }
            const StoredFiber sa = makeStoredFiberSessionSnapshot(a).fiber;
            const StoredFiber sb = makeStoredFiberSessionSnapshot(b).fiber;
            const auto sameLinks = [](std::vector<FiberBranchRef> x, std::vector<FiberBranchRef> y) {
                if (x.size() != y.size()) {
                    return false;
                }
                const auto order = [](const FiberBranchRef& l, const FiberBranchRef& r) {
                    return std::tie(l.controlPointIndex, l.branchFileName, l.branchControlPointIndex) <
                           std::tie(r.controlPointIndex, r.branchFileName, r.branchControlPointIndex);
                };
                std::sort(x.begin(), x.end(), order);
                std::sort(y.begin(), y.end(), order);
                for (std::size_t k = 0; k < x.size(); ++k) {
                    const auto& l = x[k];
                    const auto& r = y[k];
                    if (l.controlPointIndex != r.controlPointIndex ||
                        l.branchControlPointIndex != r.branchControlPointIndex ||
                        fs::path(l.branchFileName).filename() != fs::path(r.branchFileName).filename() ||
                        l.pending != r.pending || l.adjacent != r.adjacent ||
                        l.autoLink != r.autoLink ||
                        !pointsApproximatelyEqual(l.controlPointPosition, r.controlPointPosition) ||
                        !pointsApproximatelyEqual(l.branchControlPointPosition,
                                                  r.branchControlPointPosition)) {
                        return false;
                    }
                }
                return true;
            };
            const bool same = sa.linePoints == sb.linePoints && sa.tags == sb.tags &&
                la::storedControlPointPositions(sa.controlPoints) ==
                    la::storedControlPointPositions(sb.controlPoints) &&
                sameLinks(sa.branches, sb.branches);
            if (!same) {
                return fail("two windows hold different unsaved versions of " +
                            fiberErrorName(a.fiberFileName) + "; close one of them first");
            }
        }
    }
    // A running peer session could later publish or roll back branches that
    // still name an original; refuse it now. Stored peers are matched to
    // their sessions by identity, whether or not the session's live branches
    // currently hold the link.
    for (const auto& pane : _panes) {
        if (!pane.session || pane.session->fiberDeleted ||
            pane.session->taskState != LineAnnotationSession::TaskState::Running) {
            continue;
        }
        const bool isPeer = std::any_of(
            peers.begin(), peers.end(), [&pane](const StoredFiber& peer) {
                return la::sameFiberIdentity(pane.session->fiberId, peer.id);
            }) ||
            std::any_of(pane.session->branches.begin(), pane.session->branches.end(),
                        [&](const FiberBranchRef& branch) {
                            return std::any_of(request.originals.begin(), request.originals.end(),
                                               [&branch](const StoredFiber& original) {
                                                   return branchReferencesFiber(branch, original.id);
                                               });
                        });
        if (isPeer) {
            return fail("the linked fiber " + fiberErrorName(pane.session->fiberFileName) +
                        " is still optimizing; try again when it finishes");
        }
    }
    struct LiveRedirect {
        std::shared_ptr<LineAnnotationSession> session;
        std::vector<FiberBranchRef> branches;
        // Armed rollbacks hold branch vectors too; a later failing solve
        // would reinstate them, so they are redirected alongside.
        std::optional<std::vector<FiberBranchRef>> branchesBeforeModeChange;
        std::optional<std::vector<FiberBranchRef>> collapseRollbackBranches;
    };
    std::vector<LiveRedirect> live;
    const auto linksToAnyOriginal = [&](const std::vector<FiberBranchRef>& branches) {
        return std::any_of(branches.begin(), branches.end(), [&](const FiberBranchRef& branch) {
            return std::any_of(request.originals.begin(), request.originals.end(),
                               [&branch](const StoredFiber& original) {
                                   return branchReferencesFiber(branch, original.id);
                               });
        });
    };
    for (const auto& pane : _panes) {
        if (!pane.session ||
            std::find(consumed.begin(), consumed.end(), pane.session) != consumed.end()) {
            continue;
        }
        auto& session = *pane.session;
        const bool touched = linksToAnyOriginal(session.branches) ||
            (session.branchesBeforeModeChange &&
             linksToAnyOriginal(*session.branchesBeforeModeChange)) ||
            (session.controlPointCollapseRollback &&
             linksToAnyOriginal(session.controlPointCollapseRollback->branches));
        if (!touched) {
            continue;
        }
        LiveRedirect redirect;
        redirect.session = pane.session;
        redirect.branches = session.branches;
        std::string error;
        if (!la::redirectBranchRefs(redirect.branches, request.sources, &error)) {
            return fail("the open session of " + fiberErrorName(session.fiberFileName) + ": " +
                        error);
        }
        if (session.branchesBeforeModeChange) {
            redirect.branchesBeforeModeChange = *session.branchesBeforeModeChange;
            if (!la::redirectBranchRefs(*redirect.branchesBeforeModeChange, request.sources,
                                        &error)) {
                return fail("the open session of " + fiberErrorName(session.fiberFileName) +
                            ": " + error);
            }
        }
        if (session.controlPointCollapseRollback) {
            redirect.collapseRollbackBranches = session.controlPointCollapseRollback->branches;
            if (!la::redirectBranchRefs(*redirect.collapseRollbackBranches, request.sources,
                                        &error)) {
                return fail("the open session of " + fiberErrorName(session.fiberFileName) +
                            ": " + error);
            }
        }
        live.push_back(std::move(redirect));
    }

    // 3. One batch: the new fibers and every redirected peer. Canonicalizing
    //    them together recomputes every link direction from the FINAL lines
    //    on both sides (the incident: directions copied onto a reversed
    //    line), and the validator checks reciprocity with the loader's own
    //    predicate.
    std::vector<FiberSaveSnapshot> batch;
    batch.reserve(request.newFibers.size() + peers.size());
    for (const auto& fiber : request.newFibers) {
        batch.push_back(makeFiberSaveSnapshot(fiber));
    }
    for (const auto& peer : peers) {
        batch.push_back(makeFiberSaveSnapshot(peer));
    }
    try {
        vc3d::line_annotation::canonicalizeFiberSaveSnapshots(batch);
        vc3d::line_annotation::validateFiberSaveSnapshots(batch);
    } catch (const std::exception& ex) {
        return fail(std::string{"the result would not load back: "} + ex.what());
    }

    // 4. The store after the edit, around the batch: the batch and every
    //    untouched fiber it links to, the originals gone. Any issue touching
    //    the batch or a retired original aborts; defects between untouched
    //    fibers are left alone.
    std::unordered_set<uint64_t> batchKeys, retiredKeys;
    for (const auto& snapshot : batch) batchKeys.insert(snapshot.fiber.id);
    for (const auto& original : request.originals) retiredKeys.insert(original.id);
    std::vector<StoredFiber> graph;
    for (const auto& snapshot : batch) {
        graph.push_back(snapshot.fiber);
    }
    for (auto& fiber : readLinkedFibers(graph, [&batchKeys](uint64_t id) { return batchKeys.count(id) != 0; })) {
        graph.push_back(std::move(fiber));
    }
    const auto check = la::checkStructuralEditGraph(
        graph, batchKeys, retiredKeys,
        [&](const std::vector<StoredFiber>& current) {
            return la::collectFiberBranchIssues(current, isReciprocalBranchRef);
        });
    for (const auto& issue : check.unrelated) {
        Logger()->warn("Structural edit: pre-existing link issue left alone: {}: {}",
                       fiberErrorName(graph[issue.fiberIndex].fileName), issue.reason);
    }
    if (!check.blocking.empty()) {
        std::string reasons = "the result would not load back:";
        std::size_t shown = 0;
        for (const auto& issue : check.blocking) {
            if (shown++ == 4) {
                reasons += " ...";
                break;
            }
            reasons += "\n" + fiberErrorName(graph[issue.fiberIndex].fileName) + ": " +
                       issue.reason;
        }
        return fail(reasons);
    }

    // 5. Prebuild the notifications: nothing below the commit allocates.
    std::vector<std::string> retiredUuids;
    std::vector<uint64_t> retiredIds;
    for (const auto& original : request.originals) {
        retiredUuids.push_back(original.uuid);
        retiredIds.push_back(original.id);
    }
    const bool candidateOnOriginal =
        _linkCandidate && std::any_of(request.originals.begin(), request.originals.end(),
                                      [this](const StoredFiber& original) {
                                          return original.id == _linkCandidate->fiberId;
                                      });

    // Geometry, topology, summaries and retirement publish in one checked action.
    try {
        commitFiberStoreAction(batch, request.originals);
    } catch (const std::exception& error) {
        return fail(error.what());
    }

    // 7. Install: swaps and flag writes only.
    if (committed) {
        *committed = true;
    }
    for (auto& redirect : live) {
        redirect.session->branches.swap(redirect.branches);
        if (redirect.branchesBeforeModeChange && redirect.session->branchesBeforeModeChange) {
            redirect.session->branchesBeforeModeChange->swap(*redirect.branchesBeforeModeChange);
        }
        if (redirect.collapseRollbackBranches && redirect.session->controlPointCollapseRollback) {
            redirect.session->controlPointCollapseRollback->branches.swap(
                *redirect.collapseRollbackBranches);
        }
    }
    for (const auto& session : consumed) {
        session->fiberDeleted = true;
        session->suppressFiberSave = true;
        session->autoSaveScheduled = false;
        if (session->runningSolveCancel) {
            session->runningSolveCancel->store(true);
        }
        session->solveQueue.shutdown();
    }
    ++_fiberDataGeneration;
    if (candidateOnOriginal) {
        _linkCandidate.reset();
    }
    outcome.ok = true;

    // 8. Notify. Memory is consistent from here on; a throw below leaves the
    //    edit committed and only the views stale, which the caller reports
    //    as such (never as "nothing was changed").
    try {
        for (std::size_t i = 0; i < request.newFibers.size(); ++i) {
            outcome.writtenFileNames.push_back(batch[i].fiber.fileName);
        }
        applyCommittedFibers(batch);
        for (const auto& uuid : retiredUuids) emit fiberRetired(QString::fromStdString(uuid));
        for (const uint64_t id : retiredIds) {
            invalidateFiberAlignmentMetrics(id, false);
        }
        for (std::size_t i = 0; i < request.newFibers.size(); ++i) {
            invalidateFiberAlignmentMetrics(batch[i].fiber.id, true);
        }
        emit fibersDeleted(retiredIds);
        refreshBranchLineViews();

        // 9. The originals' panes close without a finalize or a close-save
        //    (a deleted session cannot save); when they actually go away is
        //    irrelevant to the committed state.
        for (const auto& pane : _panes) {
            if (pane.session && pane.dialog &&
                std::find(consumed.begin(), consumed.end(), pane.session) != consumed.end()) {
                closeLineAnnotationDialogAfterFinalization(pane.dialog);
            }
        }
        closeIntersectionInspectionForRetiredFibers(retiredIds);
    } catch (const std::exception& ex) {
        // Best effort: the diagnostics allocate; if that fails too the
        // caller still sees `committed` and reports a stale-view warning.
        try {
            Logger()->warn("Structural edit committed, but refreshing the views failed: {}",
                           ex.what());
            outcome.warning = ex.what();
        } catch (...) {
        }
    }
    return outcome;
}

void LineAnnotationController::handleGeneratedControlPointMergeWithCandidate(
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
    if (session.fiberDeleted) {
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return;
    }
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (controlPointIndex >= session.controlPoints.size()) {
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
        showError(tr("The merge candidate is on this fiber; designate a candidate on a "
                     "different fiber."));
        return;
    }
    const bool suppressErrors = session.suppressErrorDialogs;
    if (_structuralEditInProgress) {
        showError(tr("Another merge, split or link is still in progress."), suppressErrors);
        return;
    }
    _structuralEditInProgress = true;
    auto release = qScopeGuard([this]() { _structuralEditInProgress = false; });
    try {
        const LinkCandidate candidate = *_linkCandidate;
        for (const auto& otherPane : _panes) {
            if (otherPane.session && otherPane.session->fiberId == candidate.fiberId &&
                otherPane.session->taskState == LineAnnotationSession::TaskState::Running) {
                showError(tr("The merge candidate's fiber is still optimizing; "
                             "try again when it finishes."),
                          suppressErrors);
                return;
            }
        }
        const auto candidateFiber = readStoredFiber(candidate.fiberId);
        if (!candidateFiber) {
            _linkCandidate.reset();
            showError(tr("The merge candidate's fiber no longer exists."), suppressErrors);
            return;
        }
        std::string reason;
        if (!objectStoreEditingAvailable(&reason)) {
            showError(tr("Cannot merge: %1").arg(QString::fromStdString(reason)), suppressErrors);
            return;
        }

        StructuralEditCapture capture;
        capture.packageGeneration = _packageGeneration;
        capture.attachmentSequence = _attachmentSequence;
        capture.session = pane->session;
        capture.surfaceName = surfaceName;
        capture.sessionFiberId = session.fiberId;
        capture.sessionFileName = session.fiberFileName;
        capture.firstPoint = session.controlPoints[controlPointIndex].volumePoint;
        {
            const auto storedIndexMap = storedIndexMapForSessionControls(session.controlPoints);
            capture.firstHint =
                controlPointIndex < storedIndexMap.size() ? storedIndexMap[controlPointIndex] : -1;
        }
        capture.farId = candidateFiber->id;
        capture.farFileName = candidateFiber->fileName;
        capture.farPoint = candidate.position;
        capture.farHint = candidate.storedControlPointIndexHint;
        capture.suppressErrors = suppressErrors;
        const auto candidateMode = candidateFiber->optimizationMode;

        // The only modal of phase A. Nothing captured above is read from
        // live state again: a package switch or a pane close during the
        // prompt is caught by phase B's re-resolution.
        const auto pickedMode =
            pickMergeOptimizationMode(session, session.fiberOptimizationMode, candidateMode);
        if (!pickedMode || _packageGeneration != capture.packageGeneration) {
            return;
        }
        capture.mode = *pickedMode;
        QTimer::singleShot(0, this, [this, capture = std::move(capture)]() mutable {
            commitFiberMerge(std::move(capture));
        });
        release.dismiss();
    } catch (const std::exception& ex) {
        Logger()->warn("Merge aborted: {}", ex.what());
        showError(tr("The merge could not be started: %1").arg(QString::fromUtf8(ex.what())),
                  suppressErrors);
    }
}

void LineAnnotationController::commitFiberMerge(StructuralEditCapture capture)
{
    namespace la = vc3d::line_annotation;
    auto release = qScopeGuard([this]() { _structuralEditInProgress = false; });
    const auto abort = [this, &capture](const QString& reason) {
        Logger()->warn("Merge aborted: {}", reason.toStdString());
        showError(tr("The merge could not be completed: %1\nNothing was changed.").arg(reason),
                  capture.suppressErrors);
    };
    bool committed = false;
    try {
        std::string reason;
        if (!structuralEditPreflight(capture.packageGeneration, capture.attachmentSequence, &reason)) {
            if (!reason.empty()) {
                abort(QString::fromStdString(reason));
            }
            return;
        }
        const auto clickedSession = capture.session.lock();
        if (!clickedSession || clickedSession->fiberDeleted ||
            clickedSession->fiberId != capture.sessionFiberId ||
            clickedSession->fiberFileName != capture.sessionFileName) {
            abort(tr("the fiber's annotation session changed meanwhile"));
            return;
        }
        if (clickedSession->taskState == LineAnnotationSession::TaskState::Running) {
            abort(tr("line optimization is running on the fiber"));
            return;
        }
        std::shared_ptr<LineAnnotationSession> farSession;
        for (const auto& pane : _panes) {
            if (pane.session && !pane.session->fiberDeleted &&
                pane.session->fiberId == capture.farId &&
                pane.session->fiberFileName == capture.farFileName) {
                if (pane.session->taskState == LineAnnotationSession::TaskState::Running) {
                    abort(tr("the merge candidate's fiber is still optimizing"));
                    return;
                }
                farSession = pane.session;
                break;
            }
        }
        // Planning snapshots: a session's current state where one is open
        // (its unsaved edits included), the stored record otherwise.
        const StoredFiber clicked = makeStoredFiberSessionSnapshot(*clickedSession).fiber;
        StoredFiber far;
        if (farSession) {
            far = makeStoredFiberSessionSnapshot(*farSession).fiber;
        } else {
            auto stored = readStoredFiber(capture.farId, true);
            if (!stored || stored->fileName != capture.farFileName) {
                abort(tr("the merge candidate's fiber no longer exists"));
                return;
            }
            far = std::move(*stored);
        }
        const auto clickedIndex = matchingStoredControlPointIndex(
            clicked.controlPoints, capture.firstHint, capture.firstPoint);
        if (!clickedIndex) {
            abort(tr("the clicked control point could not be resolved"));
            return;
        }
        const auto candidateIndex =
            matchingStoredControlPointIndex(far.controlPoints, capture.farHint, capture.farPoint);
        if (!candidateIndex) {
            abort(tr("the merge candidate control point no longer exists"));
            return;
        }
        if (!objectStoreEditingAvailable(&reason)) {
            abort(QString::fromStdString(reason));
            return;
        }
        la::NewFiberIdentity identity;
        identity.username = currentFiberUsername();
        identity.startedAt = currentFiberDateTimeString();
        identity.sequence = nextFiberSequenceForUsername(identity.username);
        identity.fileName =
            la::fiberFileName(identity.username, identity.startedAt, identity.sequence);
        identity.id = std::max(nextFiberId(), std::max(clicked.id, far.id) + 1);

        auto plan = std::make_shared<la::MergePlan>();
        {
            auto planned = la::planFiberMerge(clicked, *clickedIndex, far, *candidateIndex,
                                              capture.mode, identity, &reason);
            if (!planned) {
                abort(QString::fromStdString(reason));
                return;
            }
            *plan = std::move(*planned);
        }
        const uint64_t mergedId = plan->merged.id;
        const std::string mergedFileName = plan->merged.fileName;
        const auto joinControlIndex = static_cast<int>(plan->joinControlIndex);

        StructuralEditRequest request;
        request.verb = "merge";
        request.newFibers.push_back(plan->merged);
        request.originals = {clicked, far};
        request.sources.push_back({clicked.id, clicked.fileName, nullptr,
                                   [plan, mergedId, mergedFileName](int index) {
                                       return la::RedirectTarget{mergedId, mergedFileName,
                                                                 plan->clickedRemap(index)};
                                   }});
        request.sources.push_back({far.id, far.fileName, nullptr,
                                   [plan, mergedId, mergedFileName](int index) {
                                       return la::RedirectTarget{mergedId, mergedFileName,
                                                                 plan->farRemap(index)};
                                   }});
        request.consumedSessions.push_back(clickedSession);
        if (farSession) {
            request.consumedSessions.push_back(farSession);
        }
        const auto outcome = commitStructuralEdit(std::move(request), &committed);
        if (!outcome.ok) {
            abort(QString::fromStdString(outcome.error));
            return;
        }
        if (!outcome.warning.empty()) {
            showError(tr("The merge was saved, but refreshing the views failed: %1\n"
                         "Reload the project to see the result.")
                          .arg(QString::fromStdString(outcome.warning)),
                      capture.suppressErrors);
        }
        Logger()->info("Merged {} and {} into {}", clicked.fileName, far.fileName, mergedFileName);

        // The re-fit yields (dataset picker, save drain): release the guard
        // first. Its outcome does not change the committed graph.
        _structuralEditInProgress = false;
        release.dismiss();
        reoptimizeMergedFibers({mergedId});
        if (_packageGeneration != capture.packageGeneration) {
            return;
        }
        // The user's own merge: continue editing at the join.
        openFiberAtControlPoint(mergedId, joinControlIndex);
        enableFiberEditing(mergedId);
    } catch (const std::exception& ex) {
        if (committed) {
            Logger()->warn("Merge committed, but a later step failed: {}", ex.what());
            showError(tr("The merge was saved, but refreshing the views failed: %1\n"
                         "Reload the project to see the result.")
                          .arg(QString::fromUtf8(ex.what())),
                      capture.suppressErrors);
        } else {
            abort(QString::fromUtf8(ex.what()));
        }
    }
}

void LineAnnotationController::handleGeneratedSpanSplit(
    const std::string& surfaceName,
    size_t firstControlPointIndex,
    size_t secondControlPointIndex,
    bool linkHalves)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.fiberDeleted) {
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return;
    }
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    if (firstControlPointIndex >= session.controlPoints.size() ||
        secondControlPointIndex >= session.controlPoints.size() ||
        firstControlPointIndex == secondControlPointIndex) {
        return;
    }
    if (!spanControlsStillAdjacent(session, firstControlPointIndex, secondControlPointIndex)) {
        showError(tr("The span changed while the menu was open; open it again."),
                  session.suppressErrorDialogs);
        return;
    }
    ensureSessionFiberIdentity(session);
    if (session.fiberId == 0) {
        session.fiberId = nextFiberId();
    }
    const bool suppressErrors = session.suppressErrorDialogs;
    if (_structuralEditInProgress) {
        showError(tr("Another merge, split or link is still in progress."), suppressErrors);
        return;
    }
    std::string reason;
    if (!objectStoreEditingAvailable(&reason)) {
        showError(tr("Cannot split: %1").arg(QString::fromStdString(reason)), suppressErrors);
        return;
    }
    _structuralEditInProgress = true;
    auto release = qScopeGuard([this]() { _structuralEditInProgress = false; });
    try {
    StructuralEditCapture capture;
    capture.packageGeneration = _packageGeneration;
    capture.attachmentSequence = _attachmentSequence;
    capture.session = pane->session;
    capture.surfaceName = surfaceName;
    capture.sessionFiberId = session.fiberId;
    capture.sessionFileName = session.fiberFileName;
    capture.firstPoint = session.controlPoints[firstControlPointIndex].volumePoint;
    capture.secondPoint = session.controlPoints[secondControlPointIndex].volumePoint;
    {
        const auto storedIndexMap = storedIndexMapForSessionControls(session.controlPoints);
        capture.firstHint = firstControlPointIndex < storedIndexMap.size()
            ? storedIndexMap[firstControlPointIndex]
            : -1;
        capture.secondHint = secondControlPointIndex < storedIndexMap.size()
            ? storedIndexMap[secondControlPointIndex]
            : -1;
    }
    capture.linkHalves = linkHalves;
    capture.suppressErrors = suppressErrors;
    QTimer::singleShot(0, this, [this, capture = std::move(capture)]() mutable {
        commitFiberSplit(std::move(capture));
    });
    release.dismiss();
    } catch (const std::exception& ex) {
        Logger()->warn("Split aborted: {}", ex.what());
        showError(tr("The split could not be started: %1").arg(QString::fromUtf8(ex.what())),
                  suppressErrors);
    }
}

void LineAnnotationController::commitFiberSplit(StructuralEditCapture capture)
{
    namespace la = vc3d::line_annotation;
    auto release = qScopeGuard([this]() { _structuralEditInProgress = false; });
    const auto abort = [this, &capture](const QString& reason) {
        Logger()->warn("Split aborted: {}", reason.toStdString());
        showError(tr("The split could not be completed: %1\nNothing was changed.").arg(reason),
                  capture.suppressErrors);
    };
    bool committed = false;
    try {
        std::string reason;
        if (!structuralEditPreflight(capture.packageGeneration, capture.attachmentSequence, &reason)) {
            if (!reason.empty()) {
                abort(QString::fromStdString(reason));
            }
            return;
        }
        const auto parentSession = capture.session.lock();
        if (!parentSession || parentSession->fiberDeleted ||
            parentSession->fiberId != capture.sessionFiberId ||
            parentSession->fiberFileName != capture.sessionFileName) {
            abort(tr("the fiber's annotation session changed meanwhile"));
            return;
        }
        if (parentSession->taskState == LineAnnotationSession::TaskState::Running) {
            abort(tr("line optimization is running on the fiber"));
            return;
        }
        const StoredFiber parent = makeStoredFiberSessionSnapshot(*parentSession).fiber;
        const auto firstIndex = matchingStoredControlPointIndex(
            parent.controlPoints, capture.firstHint, capture.firstPoint);
        const auto secondIndex = matchingStoredControlPointIndex(
            parent.controlPoints, capture.secondHint, capture.secondPoint);
        if (!firstIndex || !secondIndex) {
            abort(tr("the span's control points could not be resolved"));
            return;
        }
        if (std::abs(*firstIndex - *secondIndex) != 1) {
            abort(tr("the span's control points are not adjacent on the stored line"));
            return;
        }
        const auto splitAfter = static_cast<std::size_t>(std::min(*firstIndex, *secondIndex));
        if (!objectStoreEditingAvailable(&reason)) {
            abort(QString::fromStdString(reason));
            return;
        }

        const std::string username = currentFiberUsername();
        const std::string startedAt = currentFiberDateTimeString();
        const uint64_t prefixSequence = nextFiberSequenceForUsername(username);
        la::NewFiberIdentity prefixIdentity;
        prefixIdentity.username = username;
        prefixIdentity.startedAt = startedAt;
        prefixIdentity.sequence = prefixSequence;
        prefixIdentity.fileName = la::fiberFileName(username, startedAt, prefixSequence);
        prefixIdentity.id = std::max(nextFiberId(), parent.id + 1);
        la::NewFiberIdentity suffixIdentity = prefixIdentity;
        suffixIdentity.sequence = prefixSequence + 1;
        suffixIdentity.fileName = la::fiberFileName(username, startedAt, suffixIdentity.sequence);
        // Allocated, not derived: the registry never hands an allocated id
        // out again, a computed one it does not know about.
        suffixIdentity.id = std::max(nextFiberId(), prefixIdentity.id + 1);

        auto plan = std::make_shared<la::SplitPlanResult>();
        {
            auto planned = la::planFiberSplit(parent, splitAfter, prefixIdentity, suffixIdentity,
                                              capture.linkHalves, &reason);
            if (!planned) {
                abort(QString::fromStdString(reason));
                return;
            }
            *plan = std::move(*planned);
        }
        const std::string prefixFileName = plan->prefix.fileName;
        const std::string suffixFileName = plan->suffix.fileName;

        StructuralEditRequest request;
        request.verb = "split";
        request.newFibers = {plan->prefix, plan->suffix};
        request.originals = {parent};
        request.sources.push_back(
            {parent.id, parent.fileName, nullptr,
             [plan](int index) -> std::optional<la::RedirectTarget> {
                 const auto remapped = la::remappedSplitControlPointIndex(plan->plan, index);
                 if (!remapped) {
                     return std::nullopt;
                 }
                 const StoredFiber& half = remapped->first ? plan->suffix : plan->prefix;
                 return la::RedirectTarget{half.id, half.fileName, remapped->second};
             }});
        request.consumedSessions.push_back(parentSession);
        const auto outcome = commitStructuralEdit(std::move(request), &committed);
        if (!outcome.ok) {
            abort(QString::fromStdString(outcome.error));
            return;
        }
        if (!outcome.warning.empty()) {
            showError(tr("The split was saved, but refreshing the views failed: %1\n"
                         "Reload the project to see the result.")
                          .arg(QString::fromStdString(outcome.warning)),
                      capture.suppressErrors);
        }
        Logger()->info("Split {} into {} and {}", parent.fileName, prefixFileName, suffixFileName);

        // Neither half is reopened: the workspace closes with the original.
        // Re-fit both halves so their lines regrow extrapolation tails past
        // the split CPs; the guard is released first because the re-fit yields.
        _structuralEditInProgress = false;
        release.dismiss();
        reoptimizeMergedFibers({plan->prefix.id, plan->suffix.id});
    } catch (const std::exception& ex) {
        if (committed) {
            Logger()->warn("Split committed, but a later step failed: {}", ex.what());
            showError(tr("The split was saved, but refreshing the views failed: %1\n"
                         "Reload the project to see the result.")
                          .arg(QString::fromUtf8(ex.what())),
                      capture.suppressErrors);
        } else {
            abort(QString::fromUtf8(ex.what()));
        }
    }
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
    if (session.fiberDeleted) {
        // The fiber was deleted underneath this session; a structural edit
        // would be persisted as a new file under the deleted name.
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return;
    }
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    // Cross-fiber operation: make sure no participating fiber's newest
    // geometry is still session-only behind a debounced autosave.
    flushAllPendingSessionAutoSaves();
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
        commitFibers({&localFiber});
        session.fiberMetricsMatchStoredFiber = true;
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber after unlinking: %1")
                      .arg(QString::fromStdString(ex.what())));
    }

    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true,
            generatedSpanAlignmentMetricsForSession(session));
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
    if (session.fiberDeleted) {
        // The fiber was deleted underneath this session; a structural edit
        // would be persisted as a new file under the deleted name.
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return;
    }
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."));
        return;
    }
    // Cross-fiber operation: make sure no participating fiber's newest
    // geometry is still session-only behind a debounced autosave.
    flushAllPendingSessionAutoSaves();
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
            // Approving clears the automatic mark; re-flagging an approved
            // link makes a plain (human) pending link.
            branch.autoLink = false;
            updatedBranches.push_back(branch);
        }
    }
    if (updatedBranches.empty()) {
        showError(tr("Could not find the link to update."));
        return;
    }

    // A link is one shared store edge: saving the local side publishes the
    // approval for both. Open panes of the linked fiber mirror it.
    std::vector<uint64_t> affectedFiberIds;
    auto updateReciprocals = [&session, pending](std::vector<FiberBranchRef>& targetBranches,
                                                 const FiberBranchRef& updatedBranch) {
        bool changed = false;
        for (auto& candidate : targetBranches) {
            const bool pointsToSession = vc3d::line_annotation::sameFiberIdentity(candidate.branchFiberId, session.fiberId);
            const bool sameLocalEndpoint =
                candidate.branchControlPointIndex == updatedBranch.controlPointIndex ||
                pointsApproximatelyEqual(candidate.branchControlPointPosition,
                                         updatedBranch.controlPointPosition);
            if (pointsToSession &&
                candidate.controlPointIndex == updatedBranch.branchControlPointIndex &&
                sameLocalEndpoint) {
                candidate.pending = pending;
                candidate.autoLink = false;
                changed = true;
            }
        }
        return changed;
    };
    for (const auto& updatedBranch : updatedBranches) {
        for (const auto& otherPane : _panes) {
            if (!otherPane.session ||
                !branchReferencesFiber(updatedBranch, otherPane.session->fiberId)) {
                continue;
            }
            if (updateReciprocals(otherPane.session->branches, updatedBranch)) {
                addUniqueFiberId(affectedFiberIds, otherPane.session->fiberId);
            }
        }
    }
    scheduleBranchMetadataSaves(affectedFiberIds, session.fiberId);

    try {
        StoredFiber localFiber = storedFiberFromSession(session);
        commitFibers({&localFiber});
        session.fiberMetricsMatchStoredFiber = true;
    } catch (const std::exception& ex) {
        showError(tr("Could not save fiber after updating the link: %1")
                      .arg(QString::fromStdString(ex.what())));
    }

    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true,
            generatedSpanAlignmentMetricsForSession(session));
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
                      .arg(QString::fromStdString(ex.what())),
                  session.suppressErrorDialogs);
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
        showError(tr("Line optimization is already running."),
                  session.suppressErrorDialogs);
        return;
    }
    if (session.optimizedLine.points.empty() || session.controlPoints.size() <= 1) {
        return;
    }
    // See handleGeneratedControlPoint: persist the last landing before this
    // mutation makes its debounced autosave unsaveable.
    if (session.autoSaveScheduled) {
        flushSessionAutoSave(session.surfaceName);
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
    const auto [minControl, maxControl] = std::minmax_element(
        session.controlPoints.begin(),
        session.controlPoints.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.linePosition < rhs.linePosition;
        });
    const bool deletedOuterControl =
        selected == minControl || selected == maxControl;
    const std::vector<vc3d::line_annotation::LineControlPoint> branchRemapControls = session.controlPoints;
    const std::vector<FiberBranchRef> branchRemapBranches = session.branches;
    vc3d::line_annotation::invalidateSegmentsAdjacentToControl(
        session.controlPoints, static_cast<size_t>(deletedControlIndex));
    session.controlPoints.erase(selected);
    // Deleting a point can part two breaks or make two others neighbours.
    vc3d::line_annotation::applyGapSpanPolicy(session.controlPoints);
    // The session's geometry changed (no solve is in flight - this handler
    // refuses while one runs): bump the epoch so the side-strip fingerprint
    // and the session snapshot cache see the deletion immediately instead of
    // serving the pre-delete control points until the follow-up solve lands
    // (or indefinitely in no-reoptimize mode).
    session.solveQueue.noteSessionMutated();
    // Deletion carries no old-to-new map: a superseded solve cannot be
    // span-merged over it (delete refuses while one runs, but hygiene).
    session.runningSolveControlMap.reset();
    session.runningSolveEditedSpans.clear();
    if (session.solveQueue.hasPending()) {
        // A queued (still-debouncing) request is stamped in the pre-delete
        // control numbering, and deletion has no old-to-new index map to
        // remap it through: escalate to a whole-line pass rather than solve
        // the wrong spans under the new numbering.
        session.solveQueue.addPending({}, true);
    }
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
                       session.optimizedLine);
    saveSessionAsFiber(session, false);
    const bool autoReoptimize =
        !pane->dialog ||
        pane->dialog->reoptimizationMode() ==
            LineAnnotationDialog::ReoptimizationMode::AutoReoptimize;
    if (autoReoptimize) {
        std::vector<size_t> dirtySegments;
        if (!deletedOuterControl && deletedControlIndex > 0) {
            dirtySegments.push_back(
                static_cast<size_t>(deletedControlIndex - 1));
        }
        startFiberModeOptimization(session, false, std::move(dirtySegments));
        return;
    }

    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true,
            generatedSpanAlignmentMetricsForSession(session));
        pane->dialog->setGeneratedPredSnapPoints(
            generatedPredSnapMarkers(session.controlPoints, session.predSnapSet));
    }
}

bool LineAnnotationController::ensureDatasetForSession(LineAnnotationSession& session,
                                                       bool required,
                                                       const std::function<bool()>& stillValid)
{
    if (session.modelOnly) {
        try {
            if (!_state || !_state->currentVolume())
                throw std::runtime_error("No active volume is loaded.");
            if(session.storePin && !session.storePin->objectRevision.empty() && !ensureFiberInVolume(session.fiberId,false))
                throw std::runtime_error("Fiber coordinate binding is missing");
            return true;
        } catch (const std::exception& ex) {
            showError(QString::fromUtf8(ex.what()), session.suppressErrorDialogs);
            return false;
        }
    }
    const bool headless = session.suppressErrorDialogs || _errorDialogsSuppressed;
    if (!_state || !_state->vpkg()) {
        showError(tr("No volume package loaded."), headless);
        return false;
    }

    auto vpkg = _state->vpkg();
    std::string selected;
    std::string selectedIdentity;
    fs::path manifestPath;
    double workingToBaseScale = 1.0;
    if (!vpkg->selectedLasagnaDataset().empty()) {
        selected = vpkg->selectedLasagnaDataset();
        selectedIdentity = selected;
        const auto entries = vpkg->lasagnaDatasetEntries();
        const auto selectedEntry = std::find_if(
            entries.begin(), entries.end(), [&](const auto& entry) {
                return entry.location == selected;
            });
        if (selectedEntry != entries.end()) {
            selectedIdentity =
                vc3d::opendata::lasagnaSourceManifestLocation(*selectedEntry);
        }
        if (!vc::project::isLocationRemote(selected)) {
            manifestPath = vc::project::resolveLocalPath(selected, vpkg->path().parent_path());
        }
    } else {
        try {
            if (const auto resolved = vc3d::opendata::resolveLasagnaForVolume(
                    *vpkg, _state->currentVolumeId())) {
                manifestPath = resolved->manifestPath;
                selected = resolved->manifestBacked
                    ? manifestPath.string()
                    : vpkg->selectedLasagnaDataset();
                selectedIdentity = resolved->sourceManifestLocation;
                workingToBaseScale = resolved->workingToBaseScale;
            }
        } catch (const std::exception& ex) {
            showError(tr("Cannot resolve Lasagna for the active volume: %1")
                          .arg(QString::fromStdString(ex.what())),
                      headless);
            return false;
        }
    }
    auto openSelectedDataset = [&](const std::string& location) {
        vc::lasagna::LasagnaDatasetOpenOptions options;
        options.workingToBaseScale = workingToBaseScale;
        options.remoteCacheRoot = vc3d::remoteCachePathFs();
        const std::string resolved =
            vc::project::isLocationRemote(location) ? location : vc::project::resolveLocalPath(location, vpkg->path().parent_path()).string();
        return vc::lasagna::LasagnaDataset::openLocation(resolved, options);
    };

    // Without a dataset the session falls back to a display-only frame
    // (lineModelForInspection); tracing and optimization still require one.
    if (selected.empty() && !required)
        return true;
    if (selected.empty()) {
        const fs::path startDir = vpkg->path().empty()
            ? fs::path{}
            : vpkg->path().parent_path();
        // Suppressed callers cannot open the dataset picker.
        auto picked = (_datasetPicker && !headless)
            ? _datasetPicker(_parentWidget, startDir)
            : std::optional<std::string>{};
        // The picker ran the event loop: an operation that must not continue
        // past a package switch or a teardown says so here, before anything
        // is attached to the session or the package.
        if (stillValid && !stillValid()) {
            return false;
        }
        if (!picked || picked->empty()) {
            if (headless) {
                showError(tr("No Lasagna dataset is selected for the active volume."), true);
            }
            return false;
        }
        selected = *picked;
        selectedIdentity = selected;
        try {
            auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                openSelectedDataset(selected));
            manifestPath = dataset->manifest().manifestPath;
            auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
            session.dataset = std::move(dataset);
            session.normalSampler = std::move(sampler);
        } catch (const std::exception& ex) {
            showError(tr("Invalid Lasagna dataset: %1").arg(QString::fromStdString(ex.what())),
                      headless);
            return false;
        }
        vpkg->setSelectedLasagnaDataset(selected);
    } else {
        if (!session.normalSampler || session.selectedDatasetLocation != selected ||
            session.workingToBaseScale != workingToBaseScale) {
            try {
                auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                    openSelectedDataset(selected));
                manifestPath = dataset->manifest().manifestPath;
                auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
                session.dataset = std::move(dataset);
                session.normalSampler = std::move(sampler);
            } catch (const std::exception& ex) {
                showError(tr("Invalid selected Lasagna dataset: %1")
                              .arg(QString::fromStdString(ex.what())),
                          headless);
                return false;
            }
        }
    }

    session.selectedDatasetLocation = selected;
    session.selectedLasagnaManifestIdentity = selectedIdentity.empty()
        ? selected
        : selectedIdentity;
    session.selectedManifestPath = manifestPath;
    session.workingToBaseScale = workingToBaseScale;
    return true;
}

void LineAnnotationController::refreshLineAnnotationDatasetMenus() const
{
    for (const auto& pane : _panes) {
        if (pane.dialog) {
            refreshLineAnnotationDatasetMenu(pane.dialog);
        }
    }
}

void LineAnnotationController::refreshLineAnnotationDatasetMenu(
    LineAnnotationDialog* dialog) const
{
    if (!dialog || !_state || !_state->vpkg()) {
        return;
    }
    auto vpkg = _state->vpkg();
    std::vector<std::pair<std::string, std::string>> lasagnaOptions;
    for (const auto& entry : vpkg->lasagnaDatasetEntries()) {
        lasagnaOptions.emplace_back(entry.location, datasetEntryMenuLabel(entry));
    }
    std::vector<std::pair<std::string, std::string>> fiberOptions;
    for (const auto& entry : vpkg->fiberInferenceDatasetEntries()) {
        fiberOptions.emplace_back(entry.location, datasetEntryMenuLabel(entry));
    }
    dialog->setLasagnaDatasetOptions(
        std::move(lasagnaOptions),
        vpkg->selectedLasagnaDataset());
    dialog->setFiberInferenceDatasetOptions(
        std::move(fiberOptions),
        vpkg->selectedFiberInferenceDataset());
}

void LineAnnotationController::handleLasagnaDatasetSelectionChanged(
    const std::string& location)
{
    if (!_state || !_state->vpkg() || location.empty()) {
        return;
    }
    for (const auto& pane : _panes) {
        if (pane.session &&
            pane.session->taskState == LineAnnotationSession::TaskState::Running) {
            showError(tr("Line optimization is already running."),
                      pane.session->suppressErrorDialogs);
            refreshLineAnnotationDatasetMenus();
            return;
        }
    }

    auto vpkg = _state->vpkg();
    if (vpkg->selectedLasagnaDataset() == location) {
        return;
    }
    vpkg->setSelectedLasagnaDataset(location);
    for (const auto& pane : _panes) {
        if (!pane.session) {
            continue;
        }
        auto& session = *pane.session;
        if (session.modelOnly) continue;
        session.dataset.reset();
        session.normalSampler.reset();
        session.traceNormalDataset.reset();
        session.traceNormalSampler.reset();
        session.selectedDatasetLocation.clear();
        session.traceNormalDatasetLocation.clear();
        if (!session.optimizedLine.points.empty() && !session.controlPoints.empty()) {
            setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
        }
    }
    refreshLineAnnotationDatasetMenus();
}

void LineAnnotationController::handleFiberInferenceDatasetSelectionChanged(
    const std::string& location)
{
    if (!_state || !_state->vpkg() || location.empty()) {
        return;
    }
    for (const auto& pane : _panes) {
        if (pane.session &&
            pane.session->taskState == LineAnnotationSession::TaskState::Running) {
            showError(tr("Line optimization is already running."),
                      pane.session->suppressErrorDialogs);
            refreshLineAnnotationDatasetMenus();
            return;
        }
    }

    auto vpkg = _state->vpkg();
    if (vpkg->selectedFiberInferenceDataset() == location) {
        return;
    }
    vpkg->setSelectedFiberInferenceDataset(location);
    for (const auto& pane : _panes) {
        if (!pane.session) {
            continue;
        }
        auto& session = *pane.session;
        session.fiberInferenceDataset.reset();
        session.fiberPredictionField.reset();
        session.selectedFiberInferenceDatasetLocation.clear();
        if (!session.optimizedLine.points.empty() && !session.controlPoints.empty()) {
            setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
        }
    }
    refreshLineAnnotationDatasetMenus();
}

bool LineAnnotationController::ensureFiberInferenceDatasetForSession(
    LineAnnotationSession& session,
    const std::function<bool()>& stillValid)
{
    const bool headless = session.suppressErrorDialogs || _errorDialogsSuppressed;
    if (!_state || !_state->vpkg()) {
        showError(tr("No volume package loaded."), headless);
        return false;
    }

    auto vpkg = _state->vpkg();
    std::string selected = vpkg->selectedFiberInferenceDataset();
    const auto fiberEntries = vpkg->fiberInferenceDatasetEntries();
    if (selected.empty() && fiberEntries.size() == 1) {
        selected = fiberEntries.front().location;
        vpkg->setSelectedFiberInferenceDataset(selected);
    }
    std::string selectedIdentity = selected;
    const auto selectedEntry = std::find_if(
        fiberEntries.begin(), fiberEntries.end(), [&](const auto& entry) {
            return entry.location == selected;
        });
    if (selectedEntry != fiberEntries.end()) {
        selectedIdentity =
            vc3d::opendata::lasagnaSourceManifestLocation(*selectedEntry);
    }

    fs::path manifestPath;
    if (!selected.empty() && !vc::project::isLocationRemote(selected)) {
        manifestPath = vc::project::resolveLocalPath(selected, vpkg->path().parent_path());
    }
    if (selected.empty()) {
        const fs::path startDir = vpkg->path().empty()
            ? fs::path{}
            : vpkg->path().parent_path();
        auto picked = (_datasetPicker && !headless)
            ? _datasetPicker(_parentWidget, startDir)
            : std::optional<std::string>{};
        // Same rule as ensureDatasetForSession: the picker yielded.
        if (stillValid && !stillValid()) {
            return false;
        }
        if (!picked || picked->empty()) {
            if (headless) {
                showError(tr("No fiber inference dataset is selected for the active project."),
                          true);
            }
            return false;
        }
        selected = *picked;
        selectedIdentity = selected;
        if (!vc::project::isLocationRemote(selected))
            manifestPath = vc::project::resolveLocalPath(selected, vpkg->path().parent_path());
        try {
            vc::lasagna::LasagnaDatasetOpenOptions options;
            options.remoteCacheRoot = vc3d::remoteCachePathFs();
            const std::string resolved = vc::project::isLocationRemote(selected)
                ? selected
                : vc::project::resolveLocalPath(
                      selected, vpkg->path().parent_path()).string();
            const auto openedDataset = vc::lasagna::LasagnaDataset::openLocation(
                resolved, options);
            (void)vc::fiber_tracer::resolveFiberPredictionTraceScales(
                openedDataset.manifest());
            std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
            for (auto& prepared : vc::lasagna::prepareLasagnaProjectVolumes(
                     openedDataset, selected)) {
                volumes.push_back({
                    std::move(prepared.location), std::move(prepared.tags),
                    std::move(prepared.volume)});
            }
            const auto result = vpkg->attachPreparedLasagnaDataset(
                selected, {}, true, volumes);
            if (result == VolumePkg::AttachLasagnaResult::VolumeIdConflict) {
                showError(
                    tr("A Lasagna volume conflicts with an existing volume id."),
                    headless);
                return false;
            }
            manifestPath = openedDataset.manifest().manifestPath;
            refreshLineAnnotationDatasetMenus();
        } catch (const std::exception& error) {
            showError(QString::fromUtf8(error.what()), headless);
            return false;
        }
    }

    if (!session.fiberPredictionField ||
        session.selectedFiberInferenceDatasetLocation != selected) {
        try {
            vc::lasagna::LasagnaDatasetOpenOptions options;
            options.remoteCacheRoot = vc3d::remoteCachePathFs();
            const std::string resolved = vc::project::isLocationRemote(selected)
                ? selected
                : vc::project::resolveLocalPath(
                      selected, vpkg->path().parent_path()).string();
            auto openedDataset = vc::lasagna::LasagnaDataset::openLocation(
                resolved, options);
            manifestPath = openedDataset.manifest().manifestPath;
            const auto traceScales =
                vc::fiber_tracer::resolveFiberPredictionTraceScales(
                    openedDataset.manifest());
            auto predictionManifest = openedDataset.manifest();
            predictionManifest.workingToBaseScale = traceScales.traceToBaseScale;
            auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                std::move(predictionManifest));
            auto field = std::make_shared<vc::fiber_tracer::FiberPredictionField>(*dataset);
            session.fiberInferenceDataset = std::move(dataset);
            session.fiberPredictionField = std::move(field);
            session.fiberTraceToBaseScale = traceScales.traceToBaseScale;
        } catch (const std::exception& ex) {
            showError(tr("Invalid selected fiber inference dataset: %1")
                          .arg(QString::fromStdString(ex.what())),
                      headless);
            return false;
        }
    }

    if (!session.traceNormalDataset || !session.traceNormalSampler ||
        session.traceNormalDatasetLocation != session.selectedDatasetLocation ||
        !approximatelyEqual(
            session.traceNormalDataset->manifest().workingToBaseScale,
            session.fiberTraceToBaseScale)) {
        try {
            vc::lasagna::LasagnaDatasetOpenOptions options;
            options.workingToBaseScale = session.fiberTraceToBaseScale;
            options.remoteCacheRoot = vc3d::remoteCachePathFs();
            const std::string normalLocation =
                vc::project::isLocationRemote(session.selectedDatasetLocation)
                    ? session.selectedDatasetLocation
                    : vc::project::resolveLocalPath(
                          session.selectedDatasetLocation,
                          vpkg->path().parent_path()).string();
            auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                vc::lasagna::LasagnaDataset::openLocation(normalLocation, options));
            auto sampler =
                std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
            session.traceNormalDataset = std::move(dataset);
            session.traceNormalSampler = std::move(sampler);
            session.traceNormalDatasetLocation = session.selectedDatasetLocation;
        } catch (const std::exception& ex) {
            showError(tr("Invalid trace-scale Lasagna normal dataset: %1")
                          .arg(QString::fromStdString(ex.what())),
                      headless);
            return false;
        }
    }

    session.selectedFiberInferenceDatasetLocation = selected;
    session.selectedFiberManifestIdentity = selectedIdentity.empty()
        ? selected
        : selectedIdentity;
    session.selectedFiberInferenceManifestPath = manifestPath;
    return true;
}

bool LineAnnotationController::applyControlPointEditAndPersist(
    const std::string& surfaceName,
    size_t controlPointIndex,
    const QString& pendingSolveMessage,
    const ControlPointEdit& edit,
    std::vector<size_t>* changedGoalOwners)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return false;
    }
    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        // storedFiberFromSession would read geometry the running task is
        // about to replace.
        showError(tr("Line optimization is already running."),
                  session.suppressErrorDialogs);
        return false;
    }
    if (session.controlPointCollapseRollback || session.controlPointsBeforeModeChange) {
        // A collapse or mode change is still awaiting its solve. Its rollback
        // restores the pre-edit controls wholesale, and a collapse can have
        // moved the point, so a toggle made now could be undone by a failed
        // solve. Refuse rather than race the debounce.
        showError(pendingSolveMessage, session.suppressErrorDialogs);
        return false;
    }
    if (session.fiberDeleted) {
        // The fiber was deleted underneath this session; saving would
        // recreate the file under the deleted name.
        showError(tr("This fiber was deleted; it can no longer be edited."),
                  session.suppressErrorDialogs);
        return false;
    }
    if (controlPointIndex >= session.controlPoints.size()) {
        return false;
    }
    const cv::Vec3d editedPoint = session.controlPoints[controlPointIndex].volumePoint;
    // Whole-list snapshot: the gap sync below may change neighbouring spans
    // too (a gap forming clears a damaged tag), so a failed save restores
    // everything, not just the requested edit.
    const std::vector<vc3d::line_annotation::LineControlPoint> controlsBeforeEdit =
        session.controlPoints;
    if (!edit(session.controlPoints[controlPointIndex])) {
        return false;
    }
    // The span tags AND the gap goal policy follow the point tags at once,
    // so the file written below, the overlays rebuilt afterwards and the
    // peers mirrored below all read the gap, and its cubic-spline goal, from
    // the span; the goal is part of the committed edit, not of the solve
    // that follows (a refused or failed solve rolls back geometry only).
    const auto goalsOf = [](const std::vector<vc3d::line_annotation::LineControlPoint>& controls) {
        std::vector<vc3d::line_annotation::SegmentInterpolationGoal> goals;
        goals.reserve(controls.size());
        for (const auto& control : controls) {
            goals.push_back(control.segmentToNext
                                ? control.segmentToNext->interpGoal
                                : vc3d::line_annotation::SegmentInterpolationGoal::Global);
        }
        return goals;
    };
    const auto goalsBefore = goalsOf(session.controlPoints);
    (void)vc3d::line_annotation::applyGapSpanPolicy(session.controlPoints);
    // Handed to the caller only once the edit is persisted: a failed save
    // below restores controlsBeforeEdit, and the caller must then see no
    // changed goals (nothing to re-solve against a rolled-back state).
    std::vector<size_t> goalOwnersChanged;
    {
        const auto goalsAfter = goalsOf(session.controlPoints);
        for (size_t i = 0; i < goalsAfter.size(); ++i) {
            if (goalsAfter[i] != goalsBefore[i]) {
                goalOwnersChanged.push_back(i);
            }
        }
    }
    if (changedGoalOwners) {
        changedGoalOwners->clear();
    }

    // The tag is fiber content: publish it to the store.
    try {
        StoredFiber localFiber = storedFiberFromSession(session);
        commitFibers({&localFiber});
        session.fiberMetricsMatchStoredFiber = true;
    } catch (const std::exception& ex) {
        // Nothing persisted: put the session back so what is shown and
        // enforced matches the stored fiber, and so the same toggle can be
        // retried (setControlPointTag would otherwise report no change).
        session.controlPoints = controlsBeforeEdit;
        showError(tr("Could not save fiber after tagging the control point: %1")
                      .arg(QString::fromStdString(ex.what())),
                  session.suppressErrorDialogs);
        return false;
    }

    if (changedGoalOwners) {
        *changedGoalOwners = std::move(goalOwnersChanged);
    }

    // The same fiber can be open in another pane (an intersection-inspection
    // side, or a second dialog). Those sessions hold their own control
    // copies and enforce the placement rule from them, so the tag is applied
    // there too: matched by position, since their indices may differ.
    // One control per vector: the toggle named a single point, so a
    // coincident duplicate (should the geometry ever carry one) is left alone;
    // the nearest match is the one the preflight above judged.
    // Panes without a dialog are intersection-inspection sides; their
    // generated overlays redraw only on request.
    bool dialogLessPaneTouched = !pane->dialog;
    for (auto& otherPane : _panes) {
        if (!otherPane.session || otherPane.session.get() == &session) {
            continue;
        }
        auto& other = *otherPane.session;
        if (!vc3d::line_annotation::sameFiberIdentity(other.fiberId, session.fiberId)) {
            continue;
        }
        // A running or rolled-back task in that pane restores these
        // snapshots wholesale; they must carry the tag too or the rollback
        // would undo it. The span tags and the gap goal policy follow in
        // every list the edit changed, exactly as in the initiating pane's
        // persist, so no later save from that pane can regress a goal.
        const auto outcome = vc3d::line_annotation::applyMirroredControlPointEdit(
            other.controlPoints,
            other.controlPointsBeforeModeChange ? &*other.controlPointsBeforeModeChange : nullptr,
            other.controlPointCollapseRollback ? &other.controlPointCollapseRollback->controlPoints
                                               : nullptr,
            editedPoint,
            edit);
        if (!outcome.changed) {
            continue;
        }
        if (outcome.livePolicyChanged || outcome.liveSync.changed()) {
            // The peer's tails or goals no longer match its geometry. No new
            // solve is started in a peer: it is marked Unoptimized (save/close
            // finalizes it) and owed a whole-line pass; a solve already
            // running there is superseded outright (its merge map dropped,
            // so nothing stale is adopted at landing) and its landing then
            // dispatches or retains that pass, mode-gated.
            setSessionOptimizationState(other, SessionOptimizationState::Unoptimized);
            if (other.taskState == LineAnnotationSession::TaskState::Running) {
                other.runningSolveControlMap.reset();
                other.solveQueue.noteSessionMutated();
                other.runningSolveConfigChanged = true;
            }
            other.solveQueue.addPending({}, true);
        }
        if (outcome.snapshotChanged) {
            other.rollbackInvalidatesOptimization = true;
        }
        if (otherPane.dialog) {
            otherPane.dialog->setGeneratedBranchOverlayData(
                controlMarkersForSession(other),
                generatedBranchLinePointsForSession(other),
                generatedBranchLinkMarkers(other.branches),
                true,
                generatedSpanAlignmentMetricsForSession(other));
        } else {
            dialogLessPaneTouched = true;
        }
    }

    if (pane->dialog) {
        pane->dialog->setGeneratedBranchOverlayData(
            controlMarkersForSession(session),
            generatedBranchLinePointsForSession(session),
            generatedBranchLinkMarkers(session.branches),
            true,
            generatedSpanAlignmentMetricsForSession(session));
    }
    if (dialogLessPaneTouched && _intersectionInspection) {
        for (const auto& viewer : _intersectionInspection->generatedOverlayViewers) {
            if (viewer) {
                viewer->refreshOverlays();
            }
        }
    }
    return true;
}


bool LineAnnotationController::setControlPointTagAndPersist(
    const std::string& surfaceName,
    size_t controlPointIndex,
    const char* tag,
    bool enabled,
    const QString& pendingSolveMessage,
    std::vector<size_t>* changedGoalOwners)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return false;
    }
    auto& session = *pane->session;
    if (controlPointIndex >= session.controlPoints.size()) {
        return false;
    }
    if (enabled && controlPointTagConflictsAcrossPanes(session, controlPointIndex, tag)) {
        showError(tr("Cannot tag this control point: it already carries the other tag "
                     "(kollesis termination or break), possibly in another view of this "
                     "fiber. Remove that tag first."),
                  session.suppressErrorDialogs);
        return false;
    }
    return applyControlPointEditAndPersist(
        surfaceName,
        controlPointIndex,
        pendingSolveMessage,
        [tag, enabled](vc3d::line_annotation::LineControlPoint& control) {
            return vc3d::line_annotation::setControlPointTag(control.tags, tag, enabled);
        },
        changedGoalOwners);
}

bool LineAnnotationController::controlPointTagConflictsAcrossPanes(
    const LineAnnotationSession& session,
    size_t controlPointIndex,
    const char* tag) const
{
    // One tag or the other, in every pane: a peer pane's control at this
    // position may already carry the other tag (a collapse there kept a
    // termination's tag while this pane still shows the untagged point), and
    // the mirroring would give it both. Checked before anything is written,
    // the peers' rollback snapshots included since those are mirrored too.
    if (controlPointIndex >= session.controlPoints.size()) {
        return false;
    }
    const cv::Vec3d point = session.controlPoints[controlPointIndex].volumePoint;
    const auto wouldConflict =
        [&point, tag](const std::vector<vc3d::line_annotation::LineControlPoint>& controls) {
            const auto* control = nearestControlAtPoint(controls, point);
            if (!control) {
                return false;
            }
            std::vector<std::string> tags = control->tags;
            vc3d::line_annotation::setControlPointTag(tags, tag, true);
            return vc3d::line_annotation::controlPointTagsConflict(tags);
        };
    if (wouldConflict(session.controlPoints)) {
        return true;
    }
    for (const auto& otherPane : _panes) {
        if (!otherPane.session || otherPane.session.get() == &session) {
            continue;
        }
        const auto& other = *otherPane.session;
        if (!vc3d::line_annotation::sameFiberIdentity(other.fiberId, session.fiberId)) {
            continue;
        }
        if (wouldConflict(other.controlPoints) ||
            (other.controlPointsBeforeModeChange &&
             wouldConflict(*other.controlPointsBeforeModeChange)) ||
            (other.controlPointCollapseRollback &&
             wouldConflict(other.controlPointCollapseRollback->controlPoints))) {
            return true;
        }
    }
    return false;
}

bool LineAnnotationController::spanControlsStillAdjacent(
    const LineAnnotationSession& session,
    size_t firstControlPointIndex,
    size_t secondControlPointIndex) const
{
    const size_t count = session.controlPoints.size();
    if (firstControlPointIndex >= count || secondControlPointIndex >= count ||
        firstControlPointIndex == secondControlPointIndex) {
        return false;
    }
    std::vector<size_t> order;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(session.controlPoints[i].linePosition)) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&session](size_t a, size_t b) {
        return session.controlPoints[a].linePosition < session.controlPoints[b].linePosition;
    });
    for (size_t rank = 1; rank < order.size(); ++rank) {
        const bool pair = (order[rank - 1] == firstControlPointIndex &&
                           order[rank] == secondControlPointIndex) ||
                          (order[rank - 1] == secondControlPointIndex &&
                           order[rank] == firstControlPointIndex);
        if (pair) {
            return true;
        }
    }
    return false;
}

bool LineAnnotationController::breakRefusedNearKollesis(const LineAnnotationSession& session,
                                                        size_t controlPointIndex) const
{
    if (controlPointIndex >= session.controlPoints.size()) {
        return true;
    }
    const auto isKollesis = [&session](size_t index) {
        return vc3d::line_annotation::hasControlPointTag(
            session.controlPoints[index].tags, vc3d::line_annotation::kKollesisTerminationTag);
    };
    if (isKollesis(controlPointIndex)) {
        return true;
    }
    // Neighbours in line-position order.
    std::vector<size_t> order;
    for (size_t i = 0; i < session.controlPoints.size(); ++i) {
        if (std::isfinite(session.controlPoints[i].linePosition)) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&session](size_t a, size_t b) {
        return session.controlPoints[a].linePosition < session.controlPoints[b].linePosition;
    });
    for (size_t rank = 0; rank < order.size(); ++rank) {
        if (order[rank] != controlPointIndex) {
            continue;
        }
        return (rank > 0 && isKollesis(order[rank - 1])) ||
               (rank + 1 < order.size() && isKollesis(order[rank + 1]));
    }
    return false;
}

bool LineAnnotationController::startCommittedGoalSolve(
    LineAnnotationSession& session,
    const std::vector<size_t>& owners)
{
    if (owners.empty() || session.taskState == LineAnnotationSession::TaskState::Running) {
        return false;
    }
    // Read before the state changes below: an optimized interior with
    // nothing pending may be solved span by span.
    const bool interiorOptimized =
        session.optimizationState == SessionOptimizationState::Optimized &&
        !session.solveQueue.hasPending();
    // The goals changed with the tag and are saved: whatever happens below,
    // the geometry no longer matches them until a solve lands, and the
    // save/close finalization covers an Unoptimized session.
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    if (!ensureDatasetForSession(session)) {
        // The goal is saved but its span is unsolved: owe a whole-line pass
        // so no later partial or global-goals-only solve (a mode switch)
        // certifies the span with its old geometry.
        session.solveQueue.addPending({}, true);
        return false;
    }
    // Nothing to roll back but geometry: the controls snapshot is the
    // current list, goals included.
    session.controlPointsBeforeModeChange = session.controlPoints;
    return launchGoalSolve(session,
                           vc3d::line_annotation::lineOrderSpanRanks(session.controlPoints, owners),
                           interiorOptimized,
                           true);
}

void LineAnnotationController::refreshTailsAfterTailPolicyChange(
    LineAnnotationSession& session,
    bool wholeLine)
{
    if (session.optimizedLine.points.size() < 2 || session.controlPoints.empty()) {
        return;
    }
    // A tails-only solve certifies the interior as it stands: allowed only
    // when that interior was optimized and nothing is owed. Otherwise (a
    // successor failed after an incomplete landing, provisional spans on
    // screen, queued work) the whole line is solved.
    const bool interiorOptimized =
        session.optimizationState == SessionOptimizationState::Optimized &&
        !session.solveQueue.hasPending();
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        // The running solve computes tails with the OLD policy or distance:
        // refuse its wholesale publication (the epoch routes its landing
        // through the merge, where the config flag keeps the current tails)
        // and queue a whole-line pass instead of silently dropping the
        // change. Dispatch stays mode-gated.
        session.solveQueue.noteSessionMutated();
        session.runningSolveConfigChanged = true;
        session.solveQueue.addPending({}, true);
        return;
    }
    auto* pane = paneForSurface(session.surfaceName);
    if (!pane || !pane->dialog ||
        pane->dialog->reoptimizationMode() !=
            LineAnnotationDialog::ReoptimizationMode::AutoReoptimize) {
        return;
    }
    if (!ensureDatasetForSession(session)) {
        return;
    }
    // Tails only (every span fixed) when the caller allows it and the
    // interior was optimized with nothing pending; a whole-line solve
    // otherwise (it subsumes a pending request at dispatch instead of
    // landing a tails-only result with the successor still queued).
    if (wholeLine || !interiorOptimized) {
        startFiberModeOptimization(session, false);
    } else {
        startFiberModeOptimization(session, false, std::vector<size_t>{});
    }
}

void LineAnnotationController::refreshTailsIfPolicyChanged(
    const std::string& surfaceName,
    vc3d::line_annotation::OpenTailPolicy before)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        // A solve this edit started re-traces the tails anyway.
        return;
    }
    if (vc3d::line_annotation::openTailPolicyForControls(session.controlPoints) != before) {
        refreshTailsAfterTailPolicyChange(session, false);
    }
}

void LineAnnotationController::finishBreakEdit(
    const std::string& surfaceName,
    const std::vector<size_t>& changedGapOwners,
    vc3d::line_annotation::OpenTailPolicy before)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        // Nothing was committed (the persist refuses while a solve runs).
        return;
    }
    if (startCommittedGoalSolve(session, changedGapOwners)) {
        // The goal re-solve re-traces the tails as well.
        return;
    }
    if (vc3d::line_annotation::openTailPolicyForControls(session.controlPoints) != before) {
        refreshTailsAfterTailPolicyChange(session, false);
    }
}

void LineAnnotationController::handleTailConfigChanged(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    // An armed rollback snapshot was solved with the old distance: a later
    // restore must not land as optimized.
    if (session.controlPointCollapseRollback || session.controlPointsBeforeModeChange) {
        session.rollbackInvalidatesOptimization = true;
    }
    refreshTailsAfterTailPolicyChange(session, true);
}

void LineAnnotationController::disarmRollbackSnapshots(LineAnnotationSession& session)
{
    session.controlPointCollapseRollback.reset();
    session.controlPointsBeforeModeChange.reset();
    session.rollbackInvalidatesOptimization = false;
}

void LineAnnotationController::consumeRollbackInvalidation(LineAnnotationSession& session)
{
    if (!session.rollbackInvalidatesOptimization) {
        return;
    }
    session.rollbackInvalidatesOptimization = false;
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
}

void LineAnnotationController::handleGeneratedSpanSetGap(
    const std::string& surfaceName,
    size_t firstControlPointIndex,
    size_t secondControlPointIndex,
    bool enabled)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (firstControlPointIndex >= session.controlPoints.size() ||
        secondControlPointIndex >= session.controlPoints.size() ||
        firstControlPointIndex == secondControlPointIndex) {
        return;
    }
    if (!spanControlsStillAdjacent(session, firstControlPointIndex, secondControlPointIndex)) {
        showError(tr("The span changed while the menu was open; open it again."),
                  session.suppressErrorDialogs);
        return;
    }
    const std::array<size_t, 2> ends{firstControlPointIndex, secondControlPointIndex};
    const auto ownersOf = [](const std::vector<std::pair<size_t, size_t>>& spans) {
        std::vector<size_t> owners;
        for (const auto& span : spans) {
            owners.push_back(span.first);
        }
        return owners;
    };
    const auto spansBefore = vc3d::line_annotation::gapSpansForControls(session.controlPoints);
    const std::vector<size_t> gapsBefore = ownersOf(spansBefore);
    // Whatever this menu action commits is re-solved on EVERY exit path:
    // the first end's tag can form a gap with a neighbour (its goal is saved
    // with the tag) even when the second end's save is then refused, and
    // the tail policy can change without any goal change (the far end is an
    // end control).
    std::vector<size_t> changedGapOwners;
    const auto collect = [&changedGapOwners](const std::vector<size_t>& owners) {
        changedGapOwners.insert(changedGapOwners.end(), owners.begin(), owners.end());
    };
    const auto policyBefore = vc3d::line_annotation::openTailPolicyForControls(session.controlPoints);
    const auto finish = qScopeGuard([this, surfaceName, &changedGapOwners, policyBefore]() {
        finishBreakEdit(surfaceName, changedGapOwners, policyBefore);
    });
    const QString pendingMessage = tr("A line optimization is pending; wait for it to finish "
                                      "before changing the gap.");
    if (enabled) {
        // Both ends become breaks. A break at or immediately next to a
        // kollesis termination is refused altogether.
        // Every refusal is decided for BOTH ends before either is tagged and
        // saved, so a refused gap never persists half of itself.
        for (const size_t end : ends) {
            if (breakRefusedNearKollesis(session, end)) {
                showError(tr("Cannot make this span a gap: a break at or next to a kollesis "
                             "termination is not allowed."),
                          session.suppressErrorDialogs);
                return;
            }
            if (controlPointTagConflictsAcrossPanes(session, end, vc3d::line_annotation::kBreakTag)) {
                showError(tr("Cannot make this span a gap: an end already carries the other tag "
                             "(kollesis termination), possibly in another view of this fiber."),
                          session.suppressErrorDialogs);
                return;
            }
        }
        for (const size_t end : ends) {
            if (vc3d::line_annotation::hasControlPointTag(session.controlPoints[end].tags,
                                                          vc3d::line_annotation::kBreakTag)) {
                continue;
            }
            std::vector<size_t> owners;
            if (!setControlPointTagAndPersist(surfaceName, end,
                                              vc3d::line_annotation::kBreakTag, true,
                                              pendingMessage, &owners)) {
                return;
            }
            collect(owners);
        }
    } else {
        // Remove the break only from ends no other gap span depends on.
        for (const size_t end : ends) {
            bool needed = false;
            for (const auto& span : spansBefore) {
                const bool thisSpan =
                    (span.first == firstControlPointIndex && span.second == secondControlPointIndex) ||
                    (span.first == secondControlPointIndex && span.second == firstControlPointIndex);
                if (!thisSpan && (span.first == end || span.second == end)) {
                    needed = true;
                }
            }
            if (needed ||
                !vc3d::line_annotation::hasControlPointTag(session.controlPoints[end].tags,
                                                           vc3d::line_annotation::kBreakTag)) {
                continue;
            }
            std::vector<size_t> owners;
            if (!setControlPointTagAndPersist(surfaceName, end,
                                              vc3d::line_annotation::kBreakTag, false,
                                              pendingMessage, &owners)) {
                return;
            }
            collect(owners);
        }
    }
}

void LineAnnotationController::handleGeneratedSpanSetDamaged(
    const std::string& surfaceName,
    size_t firstControlPointIndex,
    size_t secondControlPointIndex,
    bool enabled)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (firstControlPointIndex >= session.controlPoints.size() ||
        secondControlPointIndex >= session.controlPoints.size() ||
        firstControlPointIndex == secondControlPointIndex) {
        return;
    }
    if (!spanControlsStillAdjacent(session, firstControlPointIndex, secondControlPointIndex)) {
        showError(tr("The span changed while the menu was open; open it again."),
                  session.suppressErrorDialogs);
        return;
    }
    // The span's descriptor lives on its lower-position control.
    const size_t owner =
        session.controlPoints[firstControlPointIndex].linePosition <=
                session.controlPoints[secondControlPointIndex].linePosition
            ? firstControlPointIndex
            : secondControlPointIndex;
    if (enabled && vc3d::line_annotation::spanIsGap(session.controlPoints[owner].segmentToNext)) {
        showError(tr("A gap span cannot be marked damaged."), session.suppressErrorDialogs);
        return;
    }
    // A manual-mode edit can leave a new span without a descriptor until the
    // next save back-fills it; this owner does own a span (it is the lower
    // end of one), so give it the default descriptor here. The peer mirror
    // stays a no-op on controls without one (a peer's final control).
    if (enabled && !session.controlPoints[owner].segmentToNext) {
        auto& metadata = session.controlPoints[owner].segmentToNext;
        metadata.emplace();
        metadata->interpMode = vc3d::line_annotation::SegmentInterpolationMode::Lasagna;
        metadata->message = "lasagna";
    }
    applyControlPointEditAndPersist(
        surfaceName,
        owner,
        tr("A line optimization is pending; wait for it to finish before changing the "
           "damaged tag."),
        [enabled](vc3d::line_annotation::LineControlPoint& control) {
            return vc3d::line_annotation::setSpanTag(
                control.segmentToNext, vc3d::line_annotation::kDamagedSpanTag, enabled);
        });
}

void LineAnnotationController::handleGeneratedControlPointSetKollesisTermination(
    const std::string& surfaceName,
    size_t controlPointIndex,
    bool enabled)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    // A kollesis end's tail is a short ray: refreshed on every exit path.
    const auto policyBefore = vc3d::line_annotation::openTailPolicyForControls(session.controlPoints);
    const auto tailRefresh = qScopeGuard([this, surfaceName, policyBefore]() {
        refreshTailsIfPolicyChanged(surfaceName, policyBefore);
    });
    // One or the other: a break point cannot also be a termination. Removing
    // the tag is always allowed, so a point that somehow carries both (an
    // edited file) can be repaired from either menu item.
    if (enabled && controlPointIndex < session.controlPoints.size() &&
        vc3d::line_annotation::hasControlPointTag(
            session.controlPoints[controlPointIndex].tags, vc3d::line_annotation::kBreakTag)) {
        showError(tr("Cannot tag a break point as a kollesis termination; remove the break first."),
                  session.suppressErrorDialogs);
        return;
    }
    setControlPointTagAndPersist(
        surfaceName,
        controlPointIndex,
        vc3d::line_annotation::kKollesisTerminationTag,
        enabled,
        tr("A line optimization is pending; wait for it to finish before "
           "changing the kollesis termination."));
}

void LineAnnotationController::handleGeneratedControlPointSetBreak(
    const std::string& surfaceName,
    size_t controlPointIndex,
    bool enabled)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    // The gap spans this toggle forms or dissolves (both endpoints,
    // neighbours in line order, breaks) take their goal with the tag, in the
    // persist; they are re-solved, else a lone end break's ray tail is
    // refreshed, on every exit path.
    std::vector<size_t> changedGapOwners;
    const auto policyBefore = vc3d::line_annotation::openTailPolicyForControls(session.controlPoints);
    const auto finish = qScopeGuard([this, surfaceName, &changedGapOwners, policyBefore]() {
        finishBreakEdit(surfaceName, changedGapOwners, policyBefore);
    });
    if (enabled && breakRefusedNearKollesis(session, controlPointIndex)) {
        showError(tr("Cannot tag this point as a break: a break at or next to a kollesis "
                     "termination is not allowed."),
                  session.suppressErrorDialogs);
        return;
    }
    (void)setControlPointTagAndPersist(
        surfaceName,
        controlPointIndex,
        vc3d::line_annotation::kBreakTag,
        enabled,
        tr("A line optimization is pending; wait for it to finish before "
           "changing the break."),
        &changedGapOwners);
}

void LineAnnotationController::handleGeneratedSegmentInterpolationGoal(
    const std::string& surfaceName,
    size_t firstControlPointIndex,
    size_t secondControlPointIndex,
    const std::string& goalValue)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session)
        return;
    auto& session = *pane->session;
    if (firstControlPointIndex >= session.controlPoints.size() ||
        secondControlPointIndex >= session.controlPoints.size() ||
        firstControlPointIndex == secondControlPointIndex ||
        session.optimizedLine.points.size() < 2) {
        showError(tr("Interpolation goal received an invalid CP span."),
                  session.suppressErrorDialogs);
        return;
    }

    vc3d::line_annotation::SegmentInterpolationGoal goal;
    try {
        goal = vc3d::line_annotation::segmentInterpolationGoalFromString(goalValue);
    } catch (const std::exception& ex) {
        showError(tr("Invalid interpolation goal: %1")
                      .arg(QString::fromStdString(ex.what())),
                  session.suppressErrorDialogs);
        return;
    }
    const size_t owner =
        session.controlPoints[firstControlPointIndex].linePosition <=
                session.controlPoints[secondControlPointIndex].linePosition
            ? firstControlPointIndex
            : secondControlPointIndex;
    applySegmentInterpolationGoals(session, std::vector<size_t>{owner}, goal);
}

bool LineAnnotationController::applySegmentInterpolationGoals(
    LineAnnotationSession& session,
    const std::vector<size_t>& owners,
    vc3d::line_annotation::SegmentInterpolationGoal goal)
{
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."),
                  session.suppressErrorDialogs);
        return false;
    }
    // Owners are indices into session.controlPoints, whose descriptors hold
    // the goals; the solver's dirty spans are their ranks in line order
    // (lineOrderSpanRanks), and an owner counts only with a successor.
    std::vector<size_t> changed;
    for (const size_t owner : owners) {
        if (owner >= session.controlPoints.size() ||
            vc3d::line_annotation::lineOrderSpanRanks(session.controlPoints, {owner}).empty()) {
            continue;
        }
        const auto& metadata = session.controlPoints[owner].segmentToNext;
        if (!metadata || metadata->interpGoal != goal) {
            changed.push_back(owner);
        }
    }
    if (changed.empty()) {
        return false;
    }
    if (!ensureDatasetForSession(session))
        return false;

    const bool interiorOptimized =
        session.optimizationState == SessionOptimizationState::Optimized &&
        !session.solveQueue.hasPending();
    // A menu choice: rolled back with the geometry when the solve fails.
    session.controlPointsBeforeModeChange = session.controlPoints;
    for (const size_t owner : changed) {
        auto& metadata = session.controlPoints[owner].segmentToNext;
        if (!metadata) {
            metadata.emplace();
            metadata->interpMode =
                vc3d::line_annotation::SegmentInterpolationMode::Lasagna;
            metadata->message = "lasagna";
        }
        metadata->interpGoal = goal;
    }
    return launchGoalSolve(session,
                           vc3d::line_annotation::lineOrderSpanRanks(session.controlPoints, changed),
                           interiorOptimized,
                           false);
}

bool LineAnnotationController::launchGoalSolve(
    LineAnnotationSession& session,
    const std::vector<size_t>& dirtyRanks,
    bool interiorOptimized,
    bool owedOnFailure)
{
    session.optimizedLineBeforeModeChange = session.optimizedLine;
    session.branchesBeforeModeChange = session.branches;
    // A dirty-span solve certifies the rest of the interior as it stands:
    // only from an optimized interior with nothing owed (interiorOptimized,
    // read by the caller before any state change). Otherwise (an incomplete
    // landing whose successor failed, provisional spans, queued work) the
    // goal change solves the whole line.
    setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    startFiberModeOptimization(session, true,
                               interiorOptimized ? std::optional(dirtyRanks) : std::nullopt);
    if (session.taskState != LineAnnotationSession::TaskState::Running) {
        session.controlPoints = std::move(*session.controlPointsBeforeModeChange);
        session.controlPointsBeforeModeChange.reset();
        if (session.optimizedLineBeforeModeChange) {
            session.optimizedLine = std::move(*session.optimizedLineBeforeModeChange);
            ++session.lineRevision;
            session.optimizedLineBeforeModeChange.reset();
        }
        if (session.branchesBeforeModeChange) {
            session.branches = std::move(*session.branchesBeforeModeChange);
            session.branchesBeforeModeChange.reset();
        }
        consumeRollbackInvalidation(session);
        if (owedOnFailure) {
            // Committed goals with unsolved spans: see startCommittedGoalSolve.
            session.solveQueue.addPending({}, true);
        }
        return false;
    }
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

void LineAnnotationController::syncDialogLineSolveActivity(const LineAnnotationSession& session)
{
    auto* pane = paneForSurface(session.surfaceName);
    if (pane && pane->dialog) {
        pane->dialog->setLineSolveActivity(
            session.taskState == LineAnnotationSession::TaskState::Running,
            session.solveQueue.hasPending() || session.detachedPendingWork);
    }
}

bool LineAnnotationController::applyOptimizationTaskResult(LineAnnotationSession& session,
                                                           OptimizationTaskResult task,
                                                           bool updateGeneratedViews,
                                                           SessionOptimizationState resultOptimizationState,
                                                           const std::string& eventOverride,
                                                           bool fireSuccessCallback,
                                                           bool allowFiberSave)
{
    applyFocusBoundsToOptimizationTask(task);
    if (!task.ok) {
        session.taskState = LineAnnotationSession::TaskState::Failed;
        session.error = task.error;
        showError(tr("Line optimization failed: %1")
                      .arg(QString::fromStdString(task.error)),
                  session.suppressErrorDialogs);
        return false;
    }
    auto* pane = paneForSurface(session.surfaceName);
    const cv::Vec3d previousSeedPoint = session.seedPoint;
    const fs::path previousManifestPath = session.selectedManifestPath;
    const bool previousLineWasOptimized = session.lineWasOptimized;
    const double previousFocusedLinePosition = session.focusedLinePosition;
    vc::lasagna::LineModel previousLine = session.optimizedLine;
    session.taskState = LineAnnotationSession::TaskState::Succeeded;
    session.lineWasOptimized = true;
    session.seedPoint = task.seedPoint;
    session.selectedManifestPath = task.manifestPath;
    session.optimizationReport = task.result.report;
    session.optimizedLine = std::move(task.result.line);
    ++session.lineRevision;
    const std::vector<vc3d::line_annotation::LineControlPoint> branchRemapControls = session.controlPoints;
    const std::vector<FiberBranchRef> branchRemapBranches = session.branches;
    // Per-point tags are session state the optimizer never touches, and a
    // toggle made on this fiber from another pane while the task ran was
    // propagated into session.controlPoints, not into the task's snapshot.
    // The live session is the authority: re-apply its tags by point.
    // One-to-one by position: each live control donates its tags to at most
    // one result control, so coincident points (should the geometry ever
    // carry any) cannot all inherit the first match. The donor is the NEAREST
    // undonated control within tolerance, not the first: two distinct
    // controls within tolerance of each other (a line revisiting its own
    // coordinates) must each keep their own tags, or a break could land on
    // its neighbour after the re-solve the toggle itself starts.
    {
        std::vector<bool> donated(branchRemapControls.size(), false);
        for (auto& control : task.controlPoints) {
            size_t best = std::numeric_limits<size_t>::max();
            double bestDistance = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < branchRemapControls.size(); ++i) {
                if (donated[i] ||
                    !pointsApproximatelyEqual(branchRemapControls[i].volumePoint,
                                              control.volumePoint)) {
                    continue;
                }
                const double distance =
                    cv::norm(branchRemapControls[i].volumePoint - control.volumePoint);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    best = i;
                }
            }
            if (best != std::numeric_limits<size_t>::max()) {
                control.tags = branchRemapControls[best].tags;
                // Span tags likewise: a damaged toggle mirrored in during the
                // solve exists only on the live control's descriptor.
                if (control.segmentToNext && branchRemapControls[best].segmentToNext) {
                    control.segmentToNext->tags = branchRemapControls[best].segmentToNext->tags;
                }
                control.displayNormal = branchRemapControls[best].displayNormal;
                control.direction = branchRemapControls[best].direction;
                control.displayNormalSource = branchRemapControls[best].displayNormalSource;
                donated[best] = true;
            }
        }
    }
    session.controlPoints = std::move(task.controlPoints);
    // Tags were re-attached from the live session; the span tags follow them
    // (a toggle mirrored in during the solve is not in the task's snapshot).
    vc3d::line_annotation::syncGapSpanTags(session.controlPoints);
    // Only the session-local remap before the generated views are rebuilt:
    // materializeGeneratedViews reads session.branches for the link markers,
    // but it can still fail and roll this session back, so the peer-fiber
    // phases of the sync must wait until it has succeeded — they mutate
    // other sessions, which the rollback cannot restore.
    remapBranchControlPointIndices(
        branchRemapControls, session.controlPoints, session.branches);
    // The solver reports each control's exact line index; trust it when the
    // named vertex really is the control's position (it always should be -
    // spans are pinned through the control points). Only a mismatch falls
    // back to the old O(controls x linePoints) nearest-point scan, which on
    // long fibers was a measurable slice of every landing.
    const auto solverIndexTrusted =
        [&session](const vc3d::line_annotation::LineControlPoint& control) {
            if (control.optimizedIndex < 0 ||
                control.optimizedIndex >=
                    static_cast<int>(session.optimizedLine.points.size())) {
                return false;
            }
            const cv::Vec3d delta =
                session.optimizedLine.points[static_cast<size_t>(
                                                 control.optimizedIndex)]
                    .position -
                control.volumePoint;
            // Same tolerance as the loader's exact-membership scan
            // (kControlPointMatchEpsilon = 1e-8, squared here): an index may
            // only be trusted if the fiber it produces would reload.
            return delta.dot(delta) <= 1.0e-16;
        };
    const bool solverIndicesTrusted =
        std::all_of(session.controlPoints.begin(),
                    session.controlPoints.end(),
                    solverIndexTrusted);
    bool membershipViolated = false;
    for (auto& control : session.controlPoints) {
        double bestDistance = std::numeric_limits<double>::infinity();
        int bestIndex = -1;
        if (solverIndicesTrusted) {
            bestIndex = control.optimizedIndex;
        } else {
            for (size_t i = 0; i < session.optimizedLine.points.size(); ++i) {
                const cv::Vec3d delta = session.optimizedLine.points[i].position - control.volumePoint;
                const double distance = std::sqrt(delta.dot(delta));
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = static_cast<int>(i);
                }
            }
            // The nearest vertex must actually BE the control (the loader's
            // exact-membership tolerance): a solved line whose vertices only
            // pass near a control would otherwise be marked optimized and
            // then rejected by the save guard, wedging the session in a
            // state that can never save. Reject the result instead.
            if (bestIndex >= 0 && bestDistance > 1.0e-8) {
                membershipViolated = true;
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

    // The nearest-point remap above can in principle land two controls on the
    // same line index or out of order when the solved line moved far from the
    // controls. Every consumer of control indices (span metadata, split/merge,
    // the loader's ordered-subset contract) assumes strict ordering, so a
    // violation must fail the task instead of being applied silently.
    for (size_t index = 1; !membershipViolated && index < session.controlPoints.size(); ++index) {
        if (session.controlPoints[index].optimizedIndex <=
            session.controlPoints[index - 1].optimizedIndex) {
            membershipViolated = true;
        }
    }
    {
        if (membershipViolated) {
            session.controlPoints = branchRemapControls;
            session.branches = branchRemapBranches;
            session.optimizedLine = std::move(previousLine);
            ++session.lineRevision;
            session.seedPoint = previousSeedPoint;
            session.selectedManifestPath = previousManifestPath;
            session.lineWasOptimized = previousLineWasOptimized;
            // The remap loop above may have moved the focus onto the line
            // being rejected; restore it with the rest of the session.
            session.focusedLinePosition = previousFocusedLinePosition;
            session.taskState = LineAnnotationSession::TaskState::Failed;
            session.error =
                "optimized line no longer passes exactly through the control points in order";
            showError(tr("Line optimization failed: control points no longer "
                         "resolve exactly on the line in order."),
                      session.suppressErrorDialogs);
            return false;
        }
    }

    const std::string eventName = eventOverride.empty() ? task.eventName : eventOverride;
    setSessionOptimizationState(session, resultOptimizationState);

    const std::string resultEvent = eventName.empty()
        ? "optimization_result"
        : eventName + "_result";
    writeLineDebugJson(resultEvent,
                       session.controlPoints,
                       session.optimizedLine,
                       &session.optimizationReport);
    if (updateGeneratedViews && !session.suppressGeneratedViews) {
        const bool materialized = materializeGeneratedViews(session);
        if (!materialized) {
            session.taskState = LineAnnotationSession::TaskState::Failed;
            return false;
        }
    }
    if (session.controlPointCollapseRollback && session.fiberId != 0 &&
        session.fiberMetricsMatchStoredFiber) {
        session.fiberMetricsMatchStoredFiber = false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
    }
    // Past the last failure exit: complete the deferred reciprocal cleanup
    // and endpoint refresh on linked fibers (the session-local remap already
    // ran before the views were rebuilt).
    const BranchMetadataSyncResult branchSync =
        syncLinkedBranchMetadataAfterFiberModification(
            session, nullptr, &branchRemapBranches);
    scheduleBranchMetadataSaves(branchSync.affectedFiberIds, session.fiberId);
    refreshBranchLineViews(session.fiberId);

    if (session.restoreFiberOptimizationModeOnFailure) {
        // Only the toolbar interpolation-mode switch sets that flag, and it is
        // still true while this (successful) result is applied: the whole line
        // was re-optimized under the new mode, so the human 'reviewed' verdict
        // no longer covers the geometry. Placed after the last failure exit so
        // rollback paths never see the strip flag set.
        session.stripReviewedTagOnSave = true;
    }

    if (updateGeneratedViews && session.deferShowUntilGenerated && pane && pane->dialog &&
        !pane->dialog->isVisible()) {
        emit lineAnnotationWorkspaceRequested(pane->dialog, tr("Line Annotation"));
        pane->dialog->showWithSavedGeometry();
        pane->dialog->raise();
        pane->dialog->activateWindow();
    }

    const double prefetchPrepMs = session.optimizationReport.normalChunkPrefetchMs +
                                  session.optimizationReport.normalMaterializeMs;
    Logger()->info("Line annotation Lasagna stage timing: event={} prefetch_prep_ms={:.3f} ceres_solve_ms={:.3f} overall_ms={:.3f} points={} iterations={} residuals={} converged={} prefetch_calls={}",
                   resultEvent,
                   prefetchPrepMs,
                   session.optimizationReport.ceresSolveMs,
                   session.optimizationReport.totalMs,
                   session.optimizedLine.points.size(),
                   session.optimizationReport.iterations,
                   session.optimizationReport.residuals,
                   session.optimizationReport.converged,
                   session.optimizationReport.normalPrefetchCalls);
    if (allowFiberSave && pane && !session.suppressFiberSave &&
        session.taskState == LineAnnotationSession::TaskState::Succeeded &&
        !session.optimizedLine.points.empty() &&
        !session.controlPoints.empty()) {
        // Debounced: consecutive landings coalesce into one save (with its
        // no-op probe, all-fiber summary rebuild, and linked-fiber sync)
        // instead of paying that tail on the GUI thread per landing.
        scheduleSessionAutoSave(session);
    }
    auto callback = fireSuccessCallback ? session.optimizationSucceededCallback : nullptr;
    if (callback) {
        callback(session);
    }
    return true;
}

bool LineAnnotationController::ensureSessionSolveBackends(
    LineAnnotationSession& session,
    const std::function<bool()>& stillValid,
    bool* solveAsLasagna)
{
    if (solveAsLasagna) {
        *solveAsLasagna = false;
    }
    // Every backend a fiber-mode solve of this session can need, resolved
    // before the solve: the Lasagna dataset (normals), and for a fiber that
    // traces, the fiber-follow service or the inference dataset. Each of
    // these can yield (a picker, the service start); `stillValid` is checked
    // by the helpers right after their yield and here after each returns,
    // before the next one touches controller state.
    if (!ensureDatasetForSession(session, true, stillValid)) {
        return false;
    }
    if (stillValid && !stillValid()) {
        return false;
    }
    if (!session.normalSampler && !session.modelOnly) {
        showError(tr("Could not run final line optimization: no Lasagna dataset is loaded."),
                  session.suppressErrorDialogs);
        return false;
    }
    // Every session solves through the fiber-mode solver, a single control
    // too (its branch regenerates the line from the control, so the open
    // tails follow the current tail policy).
    if (session.controlPoints.empty()) {
        return true;
    }
    const bool needsTrace = session.fiberOptimizationMode ==
            vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d ||
        std::any_of(session.controlPoints.begin(), session.controlPoints.end(),
                    [](const auto& control) {
                        return control.segmentToNext &&
                            control.segmentToNext->interpGoal ==
                                vc3d::line_annotation::SegmentInterpolationGoal::Trace;
                    });
    // A single control whose both tails are rays traces nothing, so the
    // trace inputs are not needed for it. The model service is always
    // ensured for a model session: its request installs the callbacks the
    // solver requires whatever the tails do.
    const bool rayOnly = vc3d::line_annotation::singleControlRayOnly(session.controlPoints);
    if (needsTrace && (session.modelOnly || (fiberFollowEnabled() && !rayOnly))) {
        auto* pane = paneForSurface(session.surfaceName);
        if (!ensureFiberFollowService(session.suppressErrorDialogs,
                                      pane ? pane->dialog.data() : nullptr, stillValid))
            return false;
        if (stillValid && !stillValid()) {
            return false;
        }
    } else if (needsTrace && !rayOnly && solveAsLasagna && session.controlPoints.size() == 1 &&
               (!session.fiberPredictionField || !session.traceNormalSampler)) {
        // A seed placed without an inference dataset (VC3D allows that; the
        // seed itself was a Lasagna line). The synchronous save/close path
        // never opened a dataset picker for one control and must not start
        // to: this one solve runs as Lasagna (tails grown or rays); the
        // session's persisted optimization mode is untouched. Only a caller
        // that can take that decision (the finalizer, with the
        // out-parameter) gets it; any other (the auto-link's pre-placement
        // check of a one-control fiber) ensures the trace inputs below, as
        // it always did.
        *solveAsLasagna = true;
        Logger()->info(
            "Line annotation finalize: single control without trace inputs, "
            "solving its tails as Lasagna");
    } else if (needsTrace && !rayOnly) {
        if (!ensureFiberInferenceDatasetForSession(session, stillValid)) {
            return false;
        }
        if (stillValid && !stillValid()) {
            return false;
        }
        if (!session.fiberPredictionField || !session.traceNormalSampler) {
            return false;
        }
    }
    return true;
}

LineAnnotationController::OptimizationTaskResult
LineAnnotationController::runFiberModeSolveSynchronously(
    const LineAnnotationSession& session,
    std::optional<std::vector<size_t>> dirtySegments,
    const std::string& eventName,
    bool solveAsLasagna) const
{
    // The synchronous fiber-mode solve, result only: nothing is applied to
    // the session here (applyOptimizationTaskResult publishes, including the
    // reciprocal sync of linked peers), so a caller can validate the result
    // first. Backends must be resolved (ensureSessionSolveBackends); the
    // solve itself does not process events.
    OptimizationTaskResult task;
    task.manifestPath = session.selectedManifestPath;
    task.eventName = eventName;
    task.seedPoint = session.seedPoint;
    task.focusBoundsBase = _state ? _state->activeFocusBounds() : std::nullopt;
    try {
        auto request = makeFiberModeOptimizationRequest(session, false, std::move(dirtySegments));
        if (solveAsLasagna) {
            request.globalMode = vc3d::line_annotation::FiberOptimizationMode::Lasagna;
        }
        auto optimized =
            vc3d::line_annotation::optimizeFiberWithNativeFallback(std::move(request));
        task.controlPoints = std::move(optimized.controlPoints);
        std::vector<cv::Vec3d> points;
        points.reserve(optimized.optimization.line.points.size());
        for (const auto& point : optimized.optimization.line.points) {
            points.push_back(point.position);
        }
        task.result = std::move(optimized.optimization);
        if (!session.modelOnly)
            task.result.line = lineModelFromPoints(points, session.normalSampler.get());
        task.ok = true;
    } catch (const std::exception& ex) {
        task.error = ex.what();
    } catch (...) {
        task.error = "Unknown fiber-mode finalization error.";
    }
    return task;
}

bool LineAnnotationController::finalizeSessionOptimizationSynchronously(
    LineAnnotationSession& session,
    bool fireSuccessCallback,
    const std::function<bool()>& stillValid)
{
    if (!needsFinalOptimization(session)) {
        return true;
    }
    bool solveAsLasagna = false;
    if (!ensureSessionSolveBackends(session, stillValid, &solveAsLasagna)) {
        return false;
    }

    // Every session finalizes through the fiber-mode solver, a single
    // control too (its branch regenerates the line from the control, so the
    // open tails follow the current tail policy: rays past a lone break or
    // a kollesis end, the ordinary tails once such a tag is removed); the
    // asynchronous solves have always used it for one control as well.
    if (!session.controlPoints.empty()) {
        OptimizationTaskResult task = runFiberModeSolveSynchronously(
            session, std::nullopt, "segment_interpolation_final_full_line_opt", solveAsLasagna);
        if (stillValid && !stillValid()) {
            return false;
        }
        const bool applied = applyOptimizationTaskResult(
            session,
            std::move(task),
            false,
            SessionOptimizationState::Optimized,
            "segment_interpolation_final_full_line_opt",
            fireSuccessCallback);
        if (applied) {
            // This synchronous whole-line solve subsumes any queued request
            // and any armed rollback: leaving one armed would let a LATER
            // failing solve restore state from before work this pass just
            // finalized (the async success path clears the same fields).
            (void)session.solveQueue.takePending();
            disarmRollbackSnapshots(session);
            session.optimizedLineBeforeModeChange.reset();
            session.branchesBeforeModeChange.reset();
            session.restoreFiberOptimizationModeOnFailure = false;
        }
        return applied;
    }
    // Unreachable: needsFinalOptimization requires a control.
    return true;
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
    if (session.readOnly) {
        closeLineAnnotationDialogAfterFinalization(pane->dialog);
        return;
    }
    if (session.taskState == LineAnnotationSession::TaskState::Running) {
        showError(tr("Line optimization is already running."),
                  session.suppressErrorDialogs);
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
    if (session.readOnly || session.controlPoints.empty()) {
        return;
    }
    const auto failures = _fiberSaveFailureCount;
    saveSessionAsFiber(session, false);
    if (_fiberSaveFailureCount != failures) return;

    if (session.fiberId != 0 && session.fiberMetricsMatchStoredFiber) {
        session.fiberMetricsMatchStoredFiber = false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
    }
    uint64_t solveEpoch = 0;
    if (!session.solveQueue.beginSolve(solveEpoch)) {
        showError(tr("Line optimization is already running."),
                  session.suppressErrorDialogs);
        return;
    }
    session.runningSolveEpoch = solveEpoch;
    session.runningSolveFullLine = true;
    session.runningSolveDirtySegments.clear();
    // Seed solves are never span-merged (no edit can supersede one).
    session.runningSolveControlMap.reset();
    session.runningSolveEditedSpans.clear();
    session.runningSolveConfigChanged = false;
    // No edit can supersede a seed solve (handleGeneratedControlPoint
    // requires a non-empty line), but teardown and package switches still
    // need a handle to stop it: without one the destructor's waitForDone
    // blocks for the whole solve.
    session.runningSolveCancel = std::make_shared<std::atomic<bool>>(false);
    session.optimizationStateBeforeTask = session.optimizationState;
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
    std::weak_ptr<LineAnnotationSession> weakSession;
    if (auto* pane = paneForSurface(surfaceName)) {
        weakSession = pane->session;
        if (pane->dialog) {
            pane->dialog->setOptimizationBusy(true);
        }
        syncDialogLineSolveActivity(session);
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
            [this, surfaceName, watcher, weakSession]() {
                if (paneForSurface(surfaceName)) {
                    finishOptimization(surfaceName);
                } else if (auto survivor = weakSession.lock()) {
                    // See startFiberModeOptimization: a session that
                    // outlives its pane must not strand its queue Running.
                    (void)watcher->result();
                    survivor->watcher = nullptr;
                    survivor->runningSolveCancel.reset();
                    survivor->taskState = LineAnnotationSession::TaskState::Idle;
                    (void)survivor->solveQueue.finishSolve();
                }
                watcher->deleteLater();
            });

    const auto manifestPath = session.selectedManifestPath;
    const double workingToBaseScale = session.workingToBaseScale;
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
    const int initialCenterlineLengthVx = forceFullOptimization && pane && pane->dialog
        ? std::max(2, pane->dialog->extrapolationDistanceVx() * 2)
        : (pane && pane->dialog
               ? pane->dialog->initialCenterlineLengthVx()
               : vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX_DEFAULT);
    auto dataset = session.dataset;
    auto normalSampler = session.normalSampler;
    auto cancelFlag = session.runningSolveCancel;
    const auto focusBoundsBase = _state ? _state->activeFocusBounds()
                                        : std::nullopt;
    const double rayTailDistanceBaseVoxels = static_cast<double>(
        breakExtrapolationDistanceVxForPane(paneForSurface(session.surfaceName)));
    session.runningStorePin = session.storePin;
    watcher->setFuture(QtConcurrent::run(&_lineSolvePool,
                                          [factory,
                                           manifestPath,
                                           controlPoints,
                                           initialLinePoints,
                                           sourceSliceNormal,
                                           directionMode,
                                           initialCenterlineLengthVx,
                                           forceFullOptimization,
                                           activeStart,
                                           activeEnd,
                                           workingToBaseScale,
                                           dataset,
                                           normalSampler,
                                           cancelFlag,
                                           focusBoundsBase,
                                           rayTailDistanceBaseVoxels]() mutable {
        OptimizationTaskResult task;
        if (factory) {
            task = factory(manifestPath,
                           std::move(controlPoints),
                           std::move(initialLinePoints),
                           sourceSliceNormal,
                           directionMode,
                           initialCenterlineLengthVx,
                           forceFullOptimization,
                           activeStart,
                           activeEnd);
        } else if (normalSampler) {
            (void)dataset;
            task = optimizeLineWithSampler(manifestPath,
                                           std::move(controlPoints),
                                           std::move(initialLinePoints),
                                           sourceSliceNormal,
                                           directionMode,
                                           initialCenterlineLengthVx,
                                           forceFullOptimization,
                                           activeStart,
                                           activeEnd,
                                           *normalSampler,
                                           rayTailDistanceBaseVoxels,
                                           cancelFlag.get());
        } else {
            task = optimizeLineFromManifest(manifestPath,
                                            workingToBaseScale,
                                            std::move(controlPoints),
                                            std::move(initialLinePoints),
                                            sourceSliceNormal,
                                            directionMode,
                                            initialCenterlineLengthVx,
                                            forceFullOptimization,
                                            activeStart,
                                            activeEnd,
                                            rayTailDistanceBaseVoxels,
                                            cancelFlag.get());
        }
        task.focusBoundsBase = focusBoundsBase;
        return task;
    }));
}

vc3d::line_annotation::FiberModeOptimizationRequest
LineAnnotationController::makeFiberModeOptimizationRequest(
    const LineAnnotationSession& session,
    bool retraceAll,
    std::optional<std::vector<size_t>> dirtySegments,
    bool globalGoalsOnly) const
{
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = session.controlPoints;
    request.linePointsBase.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points) {
        request.linePointsBase.push_back(point.position);
    }
    request.predictions = session.fiberPredictionField.get();
    request.baseNormalSampler = session.normalSampler.get();
    request.traceNormalSampler = session.traceNormalSampler.get();
    request.traceToBaseScale = session.fiberTraceToBaseScale;
    request.normalManifestLocation = session.selectedLasagnaManifestIdentity;
    request.fiberManifestLocation = session.selectedFiberManifestIdentity;
    request.globalMode = session.fiberOptimizationMode;
    request.retraceAll = retraceAll;
    request.dirtySegments = std::move(dirtySegments);
    request.globalGoalsOnly = globalGoalsOnly;
    request.traceConfig.traceToBaseScale = session.fiberTraceToBaseScale;
    try {
        if (_state && _state->currentVolume()) {
            const double voxelSizeUm = _state->currentVolume()->voxelSize();
            if (voxelSizeUm > 0.0 && std::isfinite(voxelSizeUm)) {
                request.traceConfig.baseVoxelSizeUm = voxelSizeUm;
            }
        }
    } catch (...) {
        request.traceConfig.baseVoxelSizeUm.reset();
    }

    const auto* pane = paneForSurface(session.surfaceName);
    const int extrapolationDistanceVx = pane && pane->dialog
        ? pane->dialog->extrapolationDistanceVx()
        : vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX_DEFAULT;
    request.extrapolationDistanceBaseVoxels =
        static_cast<double>(extrapolationDistanceVx);
    request.breakExtrapolationDistanceBaseVoxels =
        static_cast<double>(breakExtrapolationDistanceVxForPane(pane));
    request.extrapolationFallbackCallback =
        [modelOnly = session.modelOnly](const vc3d::line_annotation::FiberExtrapolationFallbackDiagnostic& diagnostic) {
            const char* side = diagnostic.side ==
                    vc3d::line_annotation::FiberExtrapolationFallbackDiagnostic::Side::Left
                ? "left"
                : "right";
            Logger()->warn(
                "Line annotation fiber extrapolation: side={} reason={} trace_points={} source={} fallback={}",
                side,
                diagnostic.reason,
                diagnostic.tracePointCount,
                diagnostic.fromException ? "exception" : "trace_result",
                modelOnly ? "none" : "Lasagna");
        };
    const auto discretization = initialLineDiscretization(
        std::max(2, extrapolationDistanceVx * 2));
    request.lasagnaConfig.segmentsPerSide = discretization.segmentsPerSide;
    request.lasagnaConfig.segmentLength = discretization.segmentLength;
    request.lasagnaConfig.straightnessWeight = 0.1;
    request.lasagnaConfig.tangentStraightnessWeight = 5.0;
    request.lasagnaConfig.samplesPerSegment = 1;
    request.lasagnaConfig.maxIterations = 1000;
    request.lasagnaConfig.differentiableNormalSampling = true;
    request.lasagnaConfig.printSolverProgress = false;
    if ((session.modelOnly || fiberFollowEnabled()) && FiberFollowService::instance().isRunning()) {
        installFiberFollowTracer(request, session.fiberId != 0 ? fiberHvDirectionTag(session.fiberId)
                                                               : QString::fromStdString(session.fiberManualHvTag));
    }
    if (session.modelOnly) {
        const double baseUm = request.traceConfig.baseVoxelSizeUm.value_or(0.0);
        request.modelLineBuilder = [baseUm](const auto& points) {
            return FiberFollowService::instance().lineModel(points, baseUm);
        };
    }
    return request;
}

bool LineAnnotationController::fiberFollowEnabled()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(vc3d::settings::line_annotation::FIBER_FOLLOW_ENABLED,
                          vc3d::settings::line_annotation::FIBER_FOLLOW_ENABLED_DEFAULT).toBool();
}

bool LineAnnotationController::ensureFiberFollowService(bool headless,
                                                        LineAnnotationDialog* dialog,
                                                        const std::function<bool()>& stillValid)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QMetaObject::Connection progress;
    if (dialog) {
        dialog->setFiberTraceStatus(tr("Fiber follow: preparing model…"), QStringLiteral(""), true);
        progress = connect(&FiberFollowService::instance(), &FiberFollowService::statusMessage,
                           dialog, &LineAnnotationDialog::appendFiberTraceDetail);
    }
    const auto disconnectProgress = qScopeGuard([&]() { disconnect(progress); });
    const auto volume = _state ? _state->currentVolume() : nullptr;
    const QString volumeSource = volume && volume->isRemote()
        ? QString::fromStdString(volume->remoteUrl()) : QString();
    const QString volumeLocator = volume && volume->isRemote()
        ? QString::fromStdString(volume->remoteLocator()) : QString();
    // The service start processes events: the dialog is only reached again
    // through a QPointer.
    const QPointer<LineAnnotationDialog> dialogGuard(dialog);
    QString error;
    const bool running = FiberFollowService::instance().ensureRunning(
        settings.value(vc3d::settings::line_annotation::FIBER_FOLLOW_CHECKPOINT).toString(),
        settings.value(vc3d::settings::line_annotation::FIBER_FOLLOW_PYTHON).toString(),
        &error, volumeSource, volumeLocator);
    // The service start processes events while the model boots: an
    // operation that must not continue past a package switch or a teardown
    // says so here, before the dialog or the error path is touched.
    if (stillValid && !stillValid()) {
        return false;
    }
    if (running) {
        if (dialogGuard)
            dialogGuard->setFiberTraceStatus(tr("Fiber follow: model ready"), {}, false);
        return true;
    }
    if (dialogGuard)
        dialogGuard->setFiberTraceStatus(tr("Fiber follow: model could not start — %1").arg(error), error, false, true);
    showError(tr("Fiber-follow model: %1").arg(error), headless);
    return false;
}

void LineAnnotationController::installFiberFollowTracer(
    vc3d::line_annotation::FiberModeOptimizationRequest& request,
    const QString& familyHint) const
{
    // The service works in VC3D base voxels with this volume's voxel size;
    // the solve hands over trace coordinates (base / traceToBaseScale).
    double baseUm = 0.0;
    try {
        if (_state && _state->currentVolume()) {
            baseUm = _state->currentVolume()->voxelSize();
        }
    } catch (...) {
    }
    baseUm = request.traceConfig.baseVoxelSizeUm.value_or(baseUm);
    const double toBase = request.traceToBaseScale > 0.0 ? request.traceToBaseScale : 1.0;
    const nlohmann::json family = familyHint == "H" || familyHint == "V"
        ? nlohmann::json(familyHint.toStdString()) : nlohmann::json(nullptr);
    const auto points = [toBase](const std::vector<cv::Vec3d>& trace) {
        auto out = nlohmann::json::array();
        for (const auto& p : trace) {
            out.push_back({p[0] * toBase, p[1] * toBase, p[2] * toBase});
        }
        return out;
    };
    const auto vector = [](const cv::Vec3d& v) { return nlohmann::json::array({v[0], v[1], v[2]}); };
    const auto fromBase = [toBase](const nlohmann::json& list) {
        std::vector<cv::Vec3d> out;
        out.reserve(list.size());
        for (const auto& p : list) {
            out.emplace_back(p.at(0).get<double>() / toBase, p.at(1).get<double>() / toBase,
                             p.at(2).get<double>() / toBase);
        }
        return out;
    };
    const auto checkVoxelSize = [baseUm]() {
        if (!(baseUm > 0.0) || !std::isfinite(baseUm))
            throw std::runtime_error("Fiber follow needs a valid volume voxel size to map coordinates.");
    };
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const double confidence = settings.value(vc3d::settings::line_annotation::FIBER_FOLLOW_CONFIDENCE,
        vc3d::settings::line_annotation::FIBER_FOLLOW_CONFIDENCE_DEFAULT).toDouble();
    request.externalExtrapolation = [=](const std::vector<cv::Vec3d>& history, const cv::Vec3d& direction,
                                        double distance) {
        checkVoxelSize();
        const auto reply = FiberFollowService::instance().request({
            {"op", "extrapolate"}, {"history", points(history)}, {"direction", vector(direction)},
            {"distance", distance * toBase}, {"base_um", baseUm}, {"family", family}, {"confidence", confidence}});
        vc::fiber_tracer::FiberTraceOneWayResult result;
        result.points = fromBase(reply.at("points"));
        result.reachedTraceLength = reply.value("reached_length", false);
        result.reason = "fiber_follow: " + reply.value("reason", std::string());
        result.steps = reply.value("decisions", 0);
        return result;
    };
    request.externalSegment = [=](const vc::fiber_tracer::FiberTraceSegmentRequest& span) {
        checkVoxelSize();
        std::vector<cv::Vec3d> history(span.referenceLine.begin(),
                                       span.referenceLine.begin() + static_cast<std::ptrdiff_t>(span.startIndex) + 1);
        nlohmann::json body{{"op", "segment"}, {"history", points(history)},
                            {"target", vector(span.referenceLine.at(span.targetIndex) * toBase)},
                            {"base_um", baseUm}, {"family", family}, {"confidence", confidence}};
        if (span.startDirection) {
            body["direction"] = vector(*span.startDirection);
        }
        const auto reply = FiberFollowService::instance().request(body);
        vc::fiber_tracer::FiberTraceSegmentResult result;
        result.fusedLine = fromBase(reply.at("points"));
        result.forward.points = result.fusedLine;
        result.forward.reason = "fiber_follow: " + reply.value("reason", std::string());
        result.forward.steps = reply.value("decisions", 0);
        result.accepted = reply.value("accepted", false) && result.fusedLine.size() >= 2;
        result.meetingErrorBaseVoxels = reply.value("meeting_error", std::numeric_limits<double>::infinity());
        result.meetingErrorTraceVoxels = result.meetingErrorBaseVoxels / toBase;
        result.maxEndpointErrorBaseVoxels = result.meetingErrorBaseVoxels;
        result.maxEndpointErrorTraceVoxels = result.meetingErrorTraceVoxels;
        result.meetingSource = "fiber_follow";
        result.reason = "fiber_follow: " + reply.value("reason", std::string());
        return result;
    };
}

void LineAnnotationController::scheduleSolveDispatch(LineAnnotationSession& session)
{
    if (session.readOnly) {
        return;
    }
    saveSessionAsFiber(session, false);
    using Queue = vc3d::line_annotation::OptimizationCoalescingQueue;
    syncDialogLineSolveActivity(session);
    if (session.solveQueue.state() != Queue::State::Idle) {
        // A solve is in flight (or shutdown): finishOptimization's epilogue
        // dispatches the pending request.
        return;
    }
    if (session.solveDispatchScheduled) {
        return;
    }
    session.solveDispatchScheduled = true;
    const std::string surfaceName = session.surfaceName;
    QTimer::singleShot(kSolveDispatchDebounceMs, this, [this, surfaceName]() {
        dispatchPendingSolve(surfaceName);
    });
}

void LineAnnotationController::dispatchPendingSolve(const std::string& surfaceName)
{
    using Queue = vc3d::line_annotation::OptimizationCoalescingQueue;
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    session.solveDispatchScheduled = false;
    if (session.solveQueue.state() != Queue::State::Idle ||
        !session.solveQueue.hasPending()) {
        // Another path started a solve meanwhile (its epilogue will dispatch
        // what is pending), or the pending request was consumed by a
        // synchronous finalize.
        return;
    }
    if (pane->dialog &&
        pane->dialog->reoptimizationMode() !=
            LineAnnotationDialog::ReoptimizationMode::AutoReoptimize) {
        // The mode changed inside the debounce window: keep the request
        // queued for a later explicit or auto pass instead of launching a
        // solve the mode says must not run.
        return;
    }
    auto pending = session.solveQueue.takePending();
    const std::vector<size_t> pendingSpansForRetry = pending.dirtySegments;
    const bool pendingWasFullLine = pending.fullLine;
    startFiberModeOptimization(session,
                               false,
                               pending.fullLine
                                   ? std::nullopt
                                   : std::optional<std::vector<size_t>>(
                                         std::move(pending.dirtySegments)));
    if (session.taskState != LineAnnotationSession::TaskState::Running &&
        !session.controlPointCollapseRollback) {
        // The solve did not start (dataset or trace prerequisites vanished)
        // and there is no rollback to restore: keep the request pending so
        // the next dispatch (or the close/save finalize, which solves the
        // whole line regardless) picks it up instead of silently dropping
        // the provisional spans. Deliberately not rescheduled here - a
        // permanently missing prerequisite must not become a retry storm.
        session.solveQueue.addPending(pendingSpansForRetry, pendingWasFullLine);
    }
    if (session.taskState != LineAnnotationSession::TaskState::Running &&
        session.controlPointCollapseRollback) {
        // The solve did not start (dataset or trace prerequisites vanished):
        // restore the multi-control collapse the way the synchronous path
        // used to when its start failed.
        auto rollback = std::move(*session.controlPointCollapseRollback);
        session.controlPointCollapseRollback.reset();
        session.controlPoints = std::move(rollback.controlPoints);
        session.optimizedLine = std::move(rollback.optimizedLine);
        ++session.lineRevision;
        session.optimizationReport = std::move(rollback.optimizationReport);
        session.branches = std::move(rollback.branches);
        session.selectedManifestPath = std::move(rollback.selectedManifestPath);
        session.seedPoint = rollback.seedPoint;
        session.focusedLinePosition = rollback.focusedLinePosition;
        session.focusedControlPoint = std::move(rollback.focusedControlPoint);
        session.lineWasOptimized = rollback.lineWasOptimized;
        session.fiberMetricsMatchStoredFiber =
            rollback.fiberMetricsMatchStoredFiber;
        setSessionOptimizationState(session, rollback.optimizationState);
        consumeRollbackInvalidation(session);
        // Whole-line work is queued below: a session that owes queued work
        // is never Optimized, whatever the snapshot said.
        setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
        // The dispatched request may have carried EARLIER edits' coalesced
        // spans too; the rollback restored a numbering they no longer fit,
        // so escalate to a whole-line request instead of dropping them (the
        // close/save finalize consumes it if no dispatch ever succeeds).
        session.solveQueue.addPending({}, true);
    }
}

void LineAnnotationController::startFiberModeOptimization(
    LineAnnotationSession& session,
    bool retraceAll,
    std::optional<std::vector<size_t>> dirtySegments,
    bool globalGoalsOnly)
{
    if (session.readOnly || session.controlPoints.empty() || session.optimizedLine.points.size() < 2) {
        return;
    }
    const auto failures = _fiberSaveFailureCount;
    saveSessionAsFiber(session, false);
    if (_fiberSaveFailureCount != failures) return;

    if (!ensureDatasetForSession(session) || (!session.normalSampler && !session.modelOnly)) {
        return;
    }
    const bool needsTrace = session.fiberOptimizationMode ==
            vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d ||
        std::any_of(session.controlPoints.begin(), session.controlPoints.end(),
                    [](const auto& control) {
                        return control.segmentToNext &&
                            control.segmentToNext->interpGoal ==
                                vc3d::line_annotation::SegmentInterpolationGoal::Trace;
                    });
    // A single control whose both tails are rays (lone break, kollesis end)
    // traces nothing: no inference inputs are needed for it. A model session
    // always ensures its service (the request installs the callbacks the
    // solver requires).
    const bool rayOnly = vc3d::line_annotation::singleControlRayOnly(session.controlPoints);
    if (needsTrace && (session.modelOnly || (fiberFollowEnabled() && !rayOnly))) {
        auto* tracePane = paneForSurface(session.surfaceName);
        if (!ensureFiberFollowService(session.suppressErrorDialogs,
                                      tracePane ? tracePane->dialog.data() : nullptr)) {
            return;
        }
    } else if (needsTrace && !rayOnly &&
        (!ensureFiberInferenceDatasetForSession(session) ||
         !session.traceNormalSampler || !session.fiberPredictionField)) {
        return;
    }
    if (!session.controlPointCollapseRollback && session.fiberId != 0 &&
        session.fiberMetricsMatchStoredFiber) {
        session.fiberMetricsMatchStoredFiber = false;
        invalidateFiberAlignmentMetrics(session.fiberId, true);
    }

    uint64_t solveEpoch = 0;
    if (!session.solveQueue.beginSolve(solveEpoch)) {
        // A solve is already in flight (a caller slipped past its own gate)
        // or the session is shutting down. Coalesce instead of double-
        // starting; the numbering behind an in-flight caller's dirty set
        // cannot be trusted, so escalate to a whole-line pass.
        session.solveQueue.addPending({}, true);
        return;
    }
    session.runningSolveEpoch = solveEpoch;
    session.runningSolveFullLine = retraceAll || !dirtySegments;
    session.runningSolveDirtySegments =
        dirtySegments ? *dirtySegments : std::vector<size_t>{};
    // Merge bookkeeping (see the session fields): identity map at launch;
    // each edit composes its renumbering in, any other mutation resets it.
    session.runningSolveControlMap.emplace(session.controlPoints.size());
    std::iota(session.runningSolveControlMap->begin(),
              session.runningSolveControlMap->end(),
              size_t{0});
    session.runningSolveEditedSpans.clear();
    session.runningSolveConfigChanged = false;
    session.runningSolveCancel = std::make_shared<std::atomic<bool>>(false);
    if (!dirtySegments && !globalGoalsOnly) {
        // A whole-line solve subsumes whatever partial request was pending.
        // (A global-goals-only pass does not: it leaves explicit-goal spans
        // alone, so the request it would swallow is re-queued below.)
        (void)session.solveQueue.takePending();
    } else if (session.solveQueue.hasPending()) {
        // A pending request recorded before this caller's own session
        // mutation may reference stale control numbering; re-solve the whole
        // line afterwards rather than trust it.
        (void)session.solveQueue.takePending();
        session.solveQueue.addPending({}, true);
    }

    session.optimizationStateBeforeTask = session.optimizationState;
    session.taskState = LineAnnotationSession::TaskState::Running;
    session.error.clear();
    session.pendingOptimizationState = SessionOptimizationState::Optimized;
    auto* watcher = new QFutureWatcher<OptimizationTaskResult>(this);
    session.watcher = watcher;
    const std::string surfaceName = session.surfaceName;
    auto* pane = paneForSurface(surfaceName);
    if (pane && pane->dialog) {
        // Passive badge, not the input-swallowing overlay: control points
        // can keep being placed while this solve runs; edits coalesce.
        pane->dialog->setOptimizationBusy(true, false);
    }
    syncDialogLineSolveActivity(session);
    std::weak_ptr<LineAnnotationSession> weakSession;
    if (pane) {
        weakSession = pane->session;
    }
    connect(watcher,
            &QFutureWatcher<OptimizationTaskResult>::finished,
            this,
            [this, surfaceName, watcher, weakSession]() {
                if (paneForSurface(surfaceName)) {
                    finishOptimization(surfaceName);
                } else if (auto survivor = weakSession.lock()) {
                    // The pane closed mid-solve but the session outlives it
                    // (the intersection inspection holds its sessions). The
                    // result has nowhere to land: discard it and return the
                    // queue to Idle so the surviving session can solve again.
                    (void)watcher->result();
                    survivor->watcher = nullptr;
                    survivor->runningSolveCancel.reset();
                    survivor->taskState = LineAnnotationSession::TaskState::Idle;
                    (void)survivor->solveQueue.finishSolve();
                }
                watcher->deleteLater();
            });

    auto request = makeFiberModeOptimizationRequest(
        session, retraceAll, std::move(dirtySegments), globalGoalsOnly);
    request.cancelFlag = session.runningSolveCancel.get();
    const bool usingFiberFollow = bool(request.externalExtrapolation);
    if (usingFiberFollow && pane && pane->dialog)
        pane->dialog->setFiberTraceStatus(tr("Fiber follow: tracing…"), QStringLiteral(""), true);
    const auto traceProgress = [this, surfaceName, watcher](const QString& message) {
        QMetaObject::invokeMethod(this, [this, surfaceName, watcher, message]() {
            auto* current = paneForSurface(surfaceName);
            if (current && current->session && current->session->watcher == watcher && current->dialog) {
                current->dialog->setFiberTraceProgress(message);
                current->dialog->appendFiberTraceDetail(message);
            }
        }, Qt::QueuedConnection);
    };

    const fs::path manifestPath = session.selectedManifestPath;
    auto predictionField = session.fiberPredictionField;
    auto normalSampler = session.normalSampler;
    auto traceNormalSampler = session.traceNormalSampler;
    const auto focusBoundsBase = _state ? _state->activeFocusBounds()
                                        : std::nullopt;
    // cancelToken keeps the flag the request points at alive for the whole
    // solve, even after the session replaces runningSolveCancel for a
    // successor solve.
    session.runningStorePin = session.storePin;
    watcher->setFuture(QtConcurrent::run(
        &_lineSolvePool,
        [request = std::move(request),
         manifestPath,
         predictionField,
         normalSampler,
         traceNormalSampler,
         focusBoundsBase,
         usingFiberFollow,
         modelOnly = session.modelOnly,
         traceProgress,
         cancelToken = session.runningSolveCancel]() mutable {
            OptimizationTaskResult task;
            task.manifestPath = manifestPath;
            task.eventName = "native_fiber_trace3d_fiber_mode";
            for (const auto& control : request.controlPoints)
                if (control.isSeed) task.seedPoint = control.volumePoint;
            task.focusBoundsBase = focusBoundsBase;
            QStringList traceDetails;
            QStringList traceWarnings;
            int traceCalls = 0;
            int tailCalls = 0;
            QElapsedTimer traceElapsed;
            traceElapsed.start();
            if (usingFiberFollow) {
                if (request.modelLineBuilder) {
                    auto buildLine = request.modelLineBuilder;
                    request.modelLineBuilder = [buildLine, traceProgress](const auto& points) {
                        traceProgress(QObject::tr("Fiber follow: building sheet normals…"));
                        return buildLine(points);
                    };
                }
                auto extrapolate = request.externalExtrapolation;
                // A ray side (lone break / kollesis end) never calls the
                // model: with a ray head the only call is the last control's.
                const bool headIsRay =
                    vc3d::line_annotation::openTailPolicyForControls(request.controlPoints).headRay;
                request.externalExtrapolation = [&, extrapolate, headIsRay](const auto& history, const auto& direction,
                                                                           double distance) {
                    ++traceCalls;
                    const QString end = ++tailCalls == 1 && !headIsRay ? QObject::tr("first") : QObject::tr("last");
                    const QString operation = QObject::tr("Fiber follow: extending outward from %1 control…").arg(end);
                    traceProgress(operation);
                    try {
                        auto result = extrapolate(history, direction, distance);
                        const QString reason = QString::fromStdString(result.reason);
                        const QString detail = QObject::tr("%1 end: %2 points; %3")
                            .arg(end).arg(result.points.size()).arg(reason);
                        traceDetails << detail;
                        traceProgress(detail);
                        if (!result.reachedTraceLength) {
                            traceWarnings << (result.points.size() >= 2
                                ? QObject::tr("%1 end stopped early; kept the traced tail: %2").arg(end, reason)
                                : (modelOnly ? QObject::tr("%1 end made no progress: %2")
                                             : QObject::tr("%1 end made no progress; using Lasagna fallback: %2")).arg(end, reason));
                        }
                        return result;
                    } catch (const std::exception& error) {
                        const QString detail = (modelOnly ? QObject::tr("Model extension failed: %1")
                            : QObject::tr("Model extension failed; using Lasagna fallback: %1"))
                            .arg(QString::fromUtf8(error.what()));
                        traceWarnings << detail;
                        traceDetails << detail;
                        traceProgress(detail);
                        throw;
                    }
                };
                auto segment = request.externalSegment;
                request.externalSegment = [&, segment](const auto& span) {
                    ++traceCalls;
                    traceProgress(QObject::tr("Fiber follow: tracing between control points…"));
                    try {
                        auto result = segment(span);
                        const QString detail = QObject::tr("Span: %1; %2")
                            .arg(result.accepted ? QObject::tr("accepted")
                                : modelOnly ? QObject::tr("failed") : QObject::tr("fallback"))
                            .arg(QString::fromStdString(result.reason));
                        traceDetails << detail;
                        traceProgress(detail);
                        if (!result.accepted)
                            traceWarnings << detail;
                        return result;
                    } catch (const std::exception& error) {
                        const QString detail = (modelOnly ? QObject::tr("Model span failed: %1")
                            : QObject::tr("Model span failed; using fallback: %1"))
                            .arg(QString::fromUtf8(error.what()));
                        traceWarnings << detail;
                        traceDetails << detail;
                        traceProgress(detail);
                        throw;
                    }
                };
            }
            try {
                (void)predictionField;
                (void)normalSampler;
                (void)traceNormalSampler;
                (void)cancelToken;
                auto optimized =
                    vc3d::line_annotation::optimizeFiberWithNativeFallback(
                        std::move(request));
                task.controlPoints = std::move(optimized.controlPoints);
                std::vector<cv::Vec3d> points;
                points.reserve(optimized.optimization.line.points.size());
                for (const auto& point : optimized.optimization.line.points) {
                    points.push_back(point.position);
                }
                task.result = std::move(optimized.optimization);
                if (!modelOnly)
                    task.result.line = LineAnnotationController::lineModelFromPoints(
                        points, normalSampler.get());
                task.ok = true;
            } catch (const std::exception& ex) {
                task.error = ex.what();
            } catch (...) {
                task.error = "Unknown fiber-mode optimization error.";
            }
            if (usingFiberFollow) {
                task.fiberTraceWarning = !task.ok || !traceWarnings.empty();
                task.fiberTraceSummary = !task.ok
                    ? QObject::tr("Fiber follow failed: %1").arg(QString::fromStdString(task.error))
                    : traceWarnings.empty()
                        ? (traceCalls ? QObject::tr("Fiber follow: tracing complete (%1 s)")
                                           .arg(traceElapsed.elapsed() / 1000.0, 0, 'f', 1)
                                      : QObject::tr("Fiber follow: no model trace needed for these spans"))
                        : QObject::tr("Fiber follow: %1").arg(traceWarnings.first());
                for (const auto& warning : traceWarnings) {
                    if (!traceDetails.contains(warning))
                        traceDetails << warning;
                }
                if (!task.ok)
                    traceDetails << QString::fromStdString(task.error);
                task.fiberTraceDetails = traceDetails.join('\n');
            }
            return task;
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
    bool currentStoreRevision = true;
    if (session.runningStorePin) {
        const auto pin = *session.runningStorePin;
        const auto store = _state->objectStoreSession();
        currentStoreRevision = store->current(pin.attachment);
        if (currentStoreRevision) store->read(pin.attachment,[&](const auto& db,const auto& txn,const auto&) {
            const auto head=db.currentObjectRevision(txn,pin.object.objectUuid);
            // A local completed edit already advanced the session pin and epoch;
            // let the superseded-result path requeue its work without closing it.
            const auto expected = session.storePin ? session.storePin->objectRevision : pin.objectRevision;
            currentStoreRevision = head ? head->uuid==expected : expected.empty();
        });
    }
    if (!currentStoreRevision) {
        session.watcher = nullptr; watcher->deleteLater();
        session.suppressFiberSave = true;
        session.solveQueue.addPending({},true);
        QTimer::singleShot(0,this,[this] { refreshChangedStoreFibers(); });
        showError(tr("The fiber changed while optimization was running. Its result was discarded; outstanding work will restart from the current revision."),session.suppressErrorDialogs);
        return;
    }
    session.watcher = nullptr;
    session.runningSolveCancel.reset();
    applyFocusBoundsToOptimizationTask(task);

    // Keep the outcome visible after the busy badge disappears, but never
    // overwrite a successor solve's progress with the previous result.
    const QString traceSummary = task.fiberTraceSummary;
    const QString traceDetails = task.fiberTraceDetails;
    const bool traceWarning = task.fiberTraceWarning;
    const bool traceSuperseded = !session.solveQueue.mayPublish(session.runningSolveEpoch);
    const auto showTraceOutcome = qScopeGuard([&]() {
        if (traceSummary.isEmpty() || !pane->dialog ||
            session.taskState == LineAnnotationSession::TaskState::Running)
            return;
        const bool failed = session.taskState == LineAnnotationSession::TaskState::Failed;
        pane->dialog->setFiberTraceStatus(failed
            ? tr("Fiber follow failed: %1").arg(QString::fromStdString(session.error))
            : traceSuperseded ? tr("Fiber follow: edits changed during tracing; reoptimization pending")
                              : traceSummary,
            traceDetails, false, failed || traceWarning);
    });

    if (!session.solveQueue.mayPublish(session.runningSolveEpoch)) {
        // The session was mutated (or began shutdown) while this solve ran:
        // its result describes a line that no longer exists. Discard it
        // wholesale - no apply, no rollback - and dispatch one solve over
        // the coalesced pending spans instead. A superseded whole-line solve
        // re-requests the whole line so a retrace is never silently shrunk
        // to the superseding edit's spans.
        Logger()->info(
            "Line annotation solve superseded: event={} ok={} solve_epoch={} "
            "current_epoch={}",
            task.eventName,
            task.ok,
            session.runningSolveEpoch,
            session.solveQueue.epoch());
        if (session.suppressFiberSave) {
            // saveOpenFibersCore superseded this solve, finalized the session
            // synchronously, and saved it; nothing is left for a successor to
            // do, and launching one would mutate a session whose saves are
            // now suppressed (the post-save geometry could never persist).
            session.runningSolveDirtySegments.clear();
            session.runningSolveControlMap.reset();
            session.runningSolveEditedSpans.clear();
            session.runningSolveConfigChanged = false;
            session.nativeSeedTracePending = false;
            (void)session.solveQueue.finishSolve();
            if (pane->dialog) {
                pane->dialog->setOptimizationBusy(false, false);
            }
            syncDialogLineSolveActivity(session);
            return;
        }
        // Never publish any portion of a result based on a superseded revision.
        session.runningSolveControlMap.reset();
        session.runningSolveEditedSpans.clear();
        session.runningSolveConfigChanged = false;
        if (session.runningSolveFullLine) session.solveQueue.addPending({}, true);
        else session.solveQueue.addPending(session.runningSolveDirtySegments, false);
        session.runningSolveDirtySegments.clear();
        session.nativeSeedTracePending = false;
        session.taskState = LineAnnotationSession::TaskState::Idle;
        auto pending = session.solveQueue.finishSolve();
        session.detachedPendingWork = pending.requested;
        if (pending.requested) {
            // Work is still owed: not an optimized state, whatever the
            // successor does (dispatched, retained in manual mode, failed).
            setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
        }
        const bool autoReoptimize = !pane->dialog ||
            pane->dialog->reoptimizationMode() ==
                LineAnnotationDialog::ReoptimizationMode::AutoReoptimize;
        if (pending.requested && !autoReoptimize) {
            // The user chose manual reoptimization: the superseding edit
            // deliberately recorded no solve request, so the coalesced spans
            // stay queued for a later explicit or auto pass instead of
            // launching a solve the mode says must not run.
            session.solveQueue.addPending(pending.dirtySegments,
                                          pending.fullLine);
        } else if (pending.requested) {
            const std::vector<size_t> pendingSpansForRetry =
                pending.dirtySegments;
            startFiberModeOptimization(
                session,
                false,
                pending.fullLine ? std::nullopt
                                 : std::optional<std::vector<size_t>>(
                                       std::move(pending.dirtySegments)));
            if (session.taskState !=
                LineAnnotationSession::TaskState::Running) {
                // See dispatchPendingSolve: never silently drop the spans.
                session.solveQueue.addPending(pendingSpansForRetry,
                                              pending.fullLine);
            }
        }
        session.detachedPendingWork = false;
        if (pane->dialog) {
            pane->dialog->setOptimizationBusy(
                session.taskState == LineAnnotationSession::TaskState::Running,
                false);
        }
        syncDialogLineSolveActivity(session);
        return;
    }
    // Publishable: return the queue to Idle before applying, so an apply
    // epilogue that starts a follow-up solve (native seed chaining, a
    // pending request from a caller that coalesced without mutating) can
    // begin a new solve cleanly.
    auto pendingAfterPublish = session.solveQueue.finishSolve();
    session.detachedPendingWork = pendingAfterPublish.requested;
    session.runningSolveDirtySegments.clear();
    session.runningSolveControlMap.reset();
    session.runningSolveEditedSpans.clear();
    session.runningSolveConfigChanged = false;

    const bool chainNativeSeedTrace = session.nativeSeedTracePending &&
        task.eventName == "seed" && task.ok;
    // Captured before the move below: a failed apply despite an ok task
    // means the generated-view rebuild failed after tearing down the
    // previous surfaces, so the rollback must rematerialize.
    const bool taskProducedResult = task.ok;
    session.nativeSeedTracePending = false;
    // A landing that still owes queued work (a successor it will dispatch or
    // retain below) is not an optimized state: it publishes as Unoptimized,
    // so a successor that fails restores Unoptimized, a retained request
    // keeps the autosave away, and save/close finalizes.
    const SessionOptimizationState resultState = pendingAfterPublish.requested
        ? SessionOptimizationState::Unoptimized
        : session.pendingOptimizationState;
    const bool ok = applyOptimizationTaskResult(session,
                                               std::move(task),
                                               !chainNativeSeedTrace,
                                               resultState,
                                               {},
                                               !chainNativeSeedTrace,
                                               !chainNativeSeedTrace);
    if (!ok) {
        setSessionOptimizationState(session, session.optimizationStateBeforeTask);
        bool rollbackRestoredModel = false;
        if (session.controlPointCollapseRollback) {
            auto rollback = std::move(*session.controlPointCollapseRollback);
            session.controlPointCollapseRollback.reset();
            session.controlPoints = std::move(rollback.controlPoints);
            session.optimizedLine = std::move(rollback.optimizedLine);
        ++session.lineRevision;
            session.optimizationReport = std::move(rollback.optimizationReport);
            session.branches = std::move(rollback.branches);
            session.selectedManifestPath = std::move(rollback.selectedManifestPath);
            session.seedPoint = rollback.seedPoint;
            session.focusedLinePosition = rollback.focusedLinePosition;
            session.focusedControlPoint = std::move(rollback.focusedControlPoint);
            session.lineWasOptimized = rollback.lineWasOptimized;
            session.fiberMetricsMatchStoredFiber =
                rollback.fiberMetricsMatchStoredFiber;
            setSessionOptimizationState(session, rollback.optimizationState);
            rollbackRestoredModel = true;
        }
        if (session.controlPointsBeforeModeChange) {
            session.controlPoints = std::move(*session.controlPointsBeforeModeChange);
            session.controlPointsBeforeModeChange.reset();
        }
        if (session.optimizedLineBeforeModeChange) {
            session.optimizedLine = std::move(*session.optimizedLineBeforeModeChange);
            ++session.lineRevision;
            session.optimizedLineBeforeModeChange.reset();
            rollbackRestoredModel = true;
        }
        if (session.branchesBeforeModeChange) {
            session.branches = std::move(*session.branchesBeforeModeChange);
            session.branchesBeforeModeChange.reset();
        }
        if (session.restoreFiberOptimizationModeOnFailure) {
            session.fiberOptimizationMode =
                session.fiberOptimizationModeBeforeTask;
            session.restoreFiberOptimizationModeOnFailure = false;
            if (pane->dialog) {
                pane->dialog->setFiberOptimizationMode(
                    session.fiberOptimizationMode);
            }
        }
        consumeRollbackInvalidation(session);
        if (session.optimizationState != SessionOptimizationState::Optimized) {
            // The failed solve owed this interior its certification (an
            // incomplete landing's successor, or a solve over provisional
            // spans): owe it again as a whole-line pass for the NEXT
            // dispatch. Not retried from this epilogue (pendingAfterPublish
            // was taken before the apply), so a persistent failure cannot
            // storm; the invariant "pending work => Unoptimized" holds.
            session.solveQueue.addPending({}, true);
        }
        if ((taskProducedResult || rollbackRestoredModel) &&
            !session.suppressGeneratedViews &&
            !session.optimizedLine.points.empty() &&
            !session.controlPoints.empty()) {
            // Two reasons to rebuild: the optimizer succeeded but
            // materializeGeneratedViews failed after tearing the panes down
            // (they are empty), or a rollback just replaced the session
            // model while the panes still display what an earlier landing —
            // possibly a merged superseded publish — materialized. Either
            // way the views must track the restored model, best-effort.
            materializeGeneratedViews(session);
        }
    } else {
        disarmRollbackSnapshots(session);
        session.optimizedLineBeforeModeChange.reset();
        session.branchesBeforeModeChange.reset();
        session.restoreFiberOptimizationModeOnFailure = false;
        if (chainNativeSeedTrace) {
            startFiberModeOptimization(session, true);
            if (session.taskState != LineAnnotationSession::TaskState::Running) {
                session.taskState = LineAnnotationSession::TaskState::Failed;
                session.error = "native seed tracing did not start";
                showError(tr("Could not start native tracing for the new fiber."),
                          session.suppressErrorDialogs);
            }
        }
    }
    if (pendingAfterPublish.requested) {
        // Re-asserted after a failed apply's rollback restore, which set
        // the snapshot's own state.
        setSessionOptimizationState(session, SessionOptimizationState::Unoptimized);
    }
    if (pendingAfterPublish.requested &&
        session.taskState != LineAnnotationSession::TaskState::Running &&
        pane->dialog &&
        pane->dialog->reoptimizationMode() !=
            LineAnnotationDialog::ReoptimizationMode::AutoReoptimize) {
        // The mode changed to manual while this solve ran: keep the request
        // queued (the superseded epilogue and the debounce dispatcher apply
        // the same gate) instead of launching a solve the mode forbids.
        session.solveQueue.addPending(pendingAfterPublish.dirtySegments,
                                      pendingAfterPublish.fullLine);
    } else if (pendingAfterPublish.requested &&
        session.taskState != LineAnnotationSession::TaskState::Running) {
        // A request coalesced without a session mutation (a gated caller
        // slipped past its own gate mid-flight); honor it now.
        const std::vector<size_t> pendingSpansForRetry =
            pendingAfterPublish.dirtySegments;
        startFiberModeOptimization(
            session,
            false,
            pendingAfterPublish.fullLine
                ? std::nullopt
                : std::optional<std::vector<size_t>>(
                      std::move(pendingAfterPublish.dirtySegments)));
        if (session.taskState != LineAnnotationSession::TaskState::Running) {
            // See dispatchPendingSolve: never silently drop the spans.
            session.solveQueue.addPending(pendingSpansForRetry,
                                          pendingAfterPublish.fullLine);
        }
    }
    session.detachedPendingWork = false;
    if (pane->dialog) {
        pane->dialog->setOptimizationBusy(
            session.taskState == LineAnnotationSession::TaskState::Running,
            false);
    }
    syncDialogLineSolveActivity(session);
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
                      .arg(QString::fromStdString(result.error)),
                  result.suppressErrorDialogs);
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
    if (session.inspectionFrameOnly || session.modelOnly || session.controlPoints.size() < 2) {
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
        const auto& segment = session.controlPoints[first.index].segmentToNext;
        if (segment) {
            metric.modeMarker = vc3d::line_annotation::segmentInterpolationModeMarker(
                segment->interpMode);
            metric.message = segment->message;
            metric.gap = vc3d::line_annotation::spanIsGap(segment);
            metric.damaged = vc3d::line_annotation::spanIsDamaged(segment);
            if (segment->interpMode ==
                vc3d::line_annotation::SegmentInterpolationMode::Trace) {
                metric.kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind::NativeMeetingError;
            } else if (segment->interpMode ==
                       vc3d::line_annotation::SegmentInterpolationMode::Cspline) {
                metric.kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind::Cspline;
            } else {
                metric.kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind::LasagnaNormalAlignment;
            }
            if (segment->metric) {
                metric.available = true;
                metric.maxErrorDegrees = *segment->metric;
            }
            if (segment->meetingErrorBaseVoxels) {
                metric.meetingErrorBaseVoxels =
                    *segment->meetingErrorBaseVoxels;
            }
            if (segment->meetingErrorRatio)
                metric.meetingErrorRatio = *segment->meetingErrorRatio;
            metric.meetingSource = segment->meetingSource;
            metric.failureCode = segment->failureCode;
            metric.failureDetail = segment->failureDetail;
        } else if (session.fiberId != 0 &&
                   session.fiberMetricsMatchStoredFiber) {
            const auto alignment = cachedAlignmentForSpan(
                session.fiberId, spanIndex);
            metric.available = alignment.available;
            metric.pending = alignment.pending;
            metric.maxErrorDegrees = alignment.maxErrorDegrees;
            metric.error = alignment.error;
        }
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

void LineAnnotationController::invalidateScrollUmbilicus()
{
    _scrollUmbilicusLoadAttempted = false;
    _scrollUmbilicusToken.clear();
    _scrollUmbilicusVolumeId.clear();
    _scrollUmbilicus.reset();
    _umbilicusNotice.clear();
}

void LineAnnotationController::onActiveVolumeChanged()
{
    // Deliberately conservative: any volume switch marks built views stale,
    // including a downsample-level switch of one scan, where the rebuild will
    // reproduce the same orientation. The alternative — proving the previous
    // views equivalent without re-resolving — needs a record of every input the
    // orientation came from (raw shape, umbilicus reading, scale source, the
    // registration transform's identity), and each review of that record found
    // another input it was missing. A wasted rebuild after a user-initiated
    // switch is bounded and visible; a stale strip is neither.
    //
    // Deliberately no ++_umbilicusGeneration: that counter means "the attachment
    // changed", which holders such as the Fiber Map read to decide staleness,
    // and a volume switch is not an attachment change — the Fiber Map covers it
    // through its own frame comparison.
    ++_orientationEpoch;
    scheduleStaleViewRefresh();
}

void LineAnnotationController::scheduleStaleViewRefresh()
{
    if (_staleViewRefreshQueued) {
        return;
    }
    _staleViewRefreshQueued = true;
    // Next event-loop turn: CState emits volumeChanged from inside
    // ViewerManager::switchVolume(), before it restores focus and navigation,
    // and materialization reads pane camera state. Deferring also collapses
    // rapid switching into one rebuild.
    QMetaObject::invokeMethod(
        this,
        [this]() {
            _staleViewRefreshQueued = false;
            refreshStaleGeneratedViews();
        },
        Qt::QueuedConnection);
}

void LineAnnotationController::refreshStaleGeneratedViews()
{
    // Surfaces already built carry the normals they were built with, so an
    // invalidated umbilicus only reaches the screen by rebuilding them.
    //
    // Snapshotted first, and by shared_ptr: materializeGeneratedViews() calls
    // _state->setSurface(), whose surfaceChanged signal reaches this controller's
    // own slot, and a pane closing there would invalidate an iterator over _panes
    // and destroy the session that the rollback below holds a reference into.
    struct Target {
        std::shared_ptr<LineAnnotationSession> session;
        std::string surfaceName;
    };
    std::vector<Target> targets;
    targets.reserve(_panes.size());
    for (const auto& pane : _panes) {
        // The selection rule lives in UmbilicusOrientationFreshness.hpp so it
        // is asserted rather than read; the skips it encodes are the ones this
        // path has been burned by (an empty model turning a successful attach
        // into a modal complaint, intersection sides double-rebuilt).
        const vc3d::annotation::GeneratedViewsPaneState state{
            pane.session != nullptr,
            pane.session && pane.session->suppressGeneratedViews,
            pane.session && !pane.session->generatedSurfaceNames.empty(),
            pane.session && !pane.session->optimizedLine.points.empty(),
            pane.session ? pane.session->orientationEpoch : -1};
        if (!vc3d::annotation::paneNeedsOrientationRefresh(state,
                                                           _orientationEpoch)) {
            continue;
        }
        targets.push_back({pane.session, pane.surfaceName});
    }

    // One pane failing must not strand the rest: each is an independent fiber, and
    // the alternative -- the first failure leaving later panes on geometry from the
    // old umbilicus -- is the inconsistency this exists to remove.
    for (const auto& target : targets) {
        // A reaction to someone else's action must not interrupt with a dialog about
        // a pane they were not looking at; the warning below is the report. Scoped
        // rather than saved and restored by hand: materializeGeneratedViews can
        // throw past its own catch, which would leave this session silent for good.
        QScopedValueRollback<bool> quiet(target.session->suppressErrorDialogs, true);
        try {
            if (!materializeGeneratedViews(*target.session)) {
                Logger()->warn(
                    "Line annotation: could not rebuild generated views for {} "
                    "after the umbilicus changed; its orientation is unchanged",
                    target.surfaceName);
            }
        } catch (const std::exception& e) {
            Logger()->warn(
                "Line annotation: rebuilding generated views for {} after the "
                "umbilicus changed threw: {}",
                target.surfaceName,
                e.what());
        }
    }

    // The intersection inspection's strips consume the same oriented normals
    // but through sessions that suppress ordinary generated views, so the pane
    // walk above never reaches them; the inspection records its own epoch and
    // is rebuilt whole.
    if (_intersectionInspection &&
        _intersectionInspection->orientationEpoch != _orientationEpoch) {
        QString error;
        if (!rebuildIntersectionInspection(&error)) {
            Logger()->warn(
                "Line annotation: could not rebuild the intersection inspection "
                "after the umbilicus or volume changed: {}",
                error.toStdString());
        }
    }
}

void LineAnnotationController::publishUmbilicusNotice()
{
    // Every open pane, because the umbilicus is a property of the package rather
    // than of any one fiber. Idempotent: the dialog ignores an unchanged notice.
    for (const auto& pane : _panes) {
        if (pane.dialog) {
            QString notice = _umbilicusNotice;
            if (pane.session && pane.session->inspectionFrameOnly) {
                if (!notice.isEmpty()) notice += "\n";
                notice += tr("No Lasagna normals: cut orientation follows the curve, not the sheet.");
            }
            pane.dialog->setUmbilicusNotice(notice);
        }
    }
}

vc3d::annotation::AnnotationFrame LineAnnotationController::annotationFrame() const
{
    if (!_state) {
        return {};
    }
    try {
        return _state->annotationFrame();
    } catch (...) {
        return {};
    }
}

const std::optional<vc::core::util::Umbilicus>&
LineAnnotationController::ensureScrollUmbilicusLoaded()
{
    // Keyed by the volume (points in its voxels, centres sized to it) and the
    // stored umbilicus revision.
    const std::string currentVolumeId =
        _state ? _state->currentVolumeId() : std::string{};
    const QString currentUmbilicusToken = umbilicusFingerprint();
    const vc3d::annotation::UmbilicusCacheInputs cachedInputs{
        _scrollUmbilicusVolumeId,
        _scrollUmbilicusToken.toStdString()};
    const vc3d::annotation::UmbilicusCacheInputs currentInputs{
        currentVolumeId,
        currentUmbilicusToken.toStdString()};
    if (vc3d::annotation::umbilicusReloadNeeded(
            _scrollUmbilicusLoadAttempted, cachedInputs, currentInputs)) {
        _scrollUmbilicusLoadAttempted = true;
        _scrollUmbilicusToken = currentUmbilicusToken;
        _scrollUmbilicusVolumeId = currentVolumeId;
        // Cleared before the attempt: whatever is set below describes this
        // attempt, and success has to be able to retract an earlier complaint.
        _scrollUmbilicus.reset();
        _umbilicusNotice.clear();
        try {
            // Line points and the stored umbilicus are both in the volume's voxels.
            auto umbilicus = _state ? _state->storedUmbilicus() : CState::StoredUmbilicus{};
            const auto volume = _state ? _state->currentVolume() : nullptr;
            if (!umbilicus.error.empty()) {
                if (_state && _state->objectStoreAttachment()) {
                    Logger()->warn("Line annotation: no usable umbilicus: {}", umbilicus.error);
                    _umbilicusNotice = tr("no usable umbilicus: %1")
                                           .arg(QString::fromStdString(umbilicus.error));
                }
            } else if (volume) {
                const auto shape = volume->shape();
                _scrollUmbilicus = vc::core::util::Umbilicus::FromPoints(
                    std::move(umbilicus.points), cv::Vec3i{shape[0], shape[1], shape[2]});
            }
        } catch (...) {
            _scrollUmbilicus.reset();
        }
        publishUmbilicusNotice();
    }
    return _scrollUmbilicus;
}

std::vector<cv::Vec3f> LineAnnotationController::orientedLineNormalsForSession(
    const LineAnnotationSession& session)
{
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    std::vector<cv::Vec3f> normals;
    normals.reserve(session.optimizedLine.points.size());

    ensureScrollUmbilicusLoaded();

    std::shared_ptr<Volume> volume;
    try {
        volume = _state ? _state->currentVolume() : nullptr;
    } catch (...) {
        volume.reset();
    }

    cv::Vec2f volumeCenterXY{kNan, kNan};
    if (volume) {
        volumeCenterXY = {
            static_cast<float>(volume->sliceWidth()) * 0.5f,
            static_cast<float>(volume->sliceHeight()) * 0.5f};
    }

    // Pass 1: chain signs along the line. Each valid normal is aligned to
    // the previous valid one, so a local sign flip -- whether from the field
    // itself or from any per-point orientation test -- is structurally
    // impossible, and the displayed frame cannot mirror mid-line.
    cv::Vec3f previousNormal{kNan, kNan, kNan};
    for (const auto& point : session.optimizedLine.points) {
        const cv::Vec3d sampled = normalizedOrZero(point.sampledNormal.normal);
        if (!point.sampledNormal.valid || !finiteDirection(sampled)) {
            normals.push_back({kNan, kNan, kNan});
            continue;
        }
        cv::Vec3f normal = toVec3f(sampled);
        if (std::isfinite(previousNormal[0]) &&
            normal.dot(previousNormal) < 0.0f) {
            normal = -normal;
        }
        previousNormal = normal;
        normals.push_back(normal);
    }

    // Pass 2: one global majority vote decides the whole line's sign, so a
    // noisy or imperfect center reference cannot flip individual stretches.
    // The reference only needs to get the hemisphere right on balance; its z
    // component is zeroed because orientation is radial-only (z is the
    // scroll axis).
    // Sign convention: normals point away from the scroll center. The dialog
    // uses them directly as the cut view's up hint, and the viewer's scene y
    // renders downward, so this puts the fiber's center-facing surface at
    // the top of the view.
    double awayScore = 0.0;
    for (size_t index = 0; index < normals.size(); ++index) {
        if (!std::isfinite(normals[index][0])) {
            continue;
        }
        const cv::Vec3f position =
            toVec3f(session.optimizedLine.points[index].position);
        cv::Vec3f towardCenter{kNan, kNan, kNan};
        if (_scrollUmbilicus) {
            try {
                towardCenter = _scrollUmbilicus->vector_to_umbilicus(position);
            } catch (...) {
            }
        }
        if (!std::isfinite(towardCenter[0]) && std::isfinite(volumeCenterXY[0])) {
            towardCenter = {volumeCenterXY[0] - position[0],
                            volumeCenterXY[1] - position[1],
                            0.0f};
        }
        if (!std::isfinite(towardCenter[0])) {
            continue;
        }
        // Cosine-weighted vote: normalize the reference so each point's
        // influence reflects only how well its normal aligns with the radial
        // direction, not its distance from the center -- a few large-radius
        // points must not outvote the rest of the line.
        towardCenter[2] = 0.0f;
        const float towardNorm = cv::norm(towardCenter);
        if (!(towardNorm > 1.0e-3f)) {
            continue;
        }
        awayScore -= static_cast<double>(
            normals[index].dot(towardCenter) / towardNorm);
    }
    if (awayScore < 0.0) {
        for (cv::Vec3f& normal : normals) {
            if (std::isfinite(normal[0])) {
                normal = -normal;
            }
        }
    }
    return normals;
}

vc::fiber_tracer::FiberDisplayField LineAnnotationController::displayFieldForSession(
    const LineAnnotationSession& session, const std::vector<cv::Vec3f>& orientedNormals) const
{
    std::vector<cv::Vec3f> displayPoints;
    std::vector<double> displayControlPositions;
    std::vector<std::optional<cv::Vec3d>> manualNormals;
    std::vector<std::optional<cv::Vec3d>> controlDirections;
    displayPoints.reserve(session.optimizedLine.points.size());
    for (const auto& point : session.optimizedLine.points)
        displayPoints.push_back(cv::Vec3f(point.position));
    for (const auto& control : session.controlPoints) {
        displayControlPositions.push_back(control.linePosition);
        manualNormals.push_back(control.displayNormal);
        controlDirections.push_back(control.direction);
    }
    return vc::fiber_tracer::fiberDisplayField(
        displayPoints, orientedNormals, displayControlPositions, manualNormals, controlDirections);
}

bool LineAnnotationController::materializeGeneratedViews(LineAnnotationSession& session,
                                                         const std::function<bool()>& stillValid)
{
    const QPointer<LineAnnotationController> alive(this);
    session.inspectionFrameOnly = std::any_of(
        session.optimizedLine.points.begin(), session.optimizedLine.points.end(),
        [](const auto& point) { return point.sampledNormal.reason == "display frame only"; });
    publishUmbilicusNotice();
    if (!_state) {
        session.error = "No active application state.";
        showError(tr("Could not create line annotation views: no active application state."),
                  session.suppressErrorDialogs);
        return false;
    }

    if (session.coordinateBaseShapeZYX) {
        try {
            const auto volume = _state->currentVolume();
            if (!volume) {
                throw std::runtime_error("no active volume is loaded");
            }
            if(session.storePin && !session.storePin->objectRevision.empty() && !ensureFiberInVolume(session.fiberId,false))
                throw std::runtime_error("Fiber coordinate binding is missing");
        } catch (const std::exception& ex) {
            session.error = ex.what();
            showError(tr("Could not map the fiber into the active volume: %1")
                          .arg(QString::fromStdString(session.error)),
                      session.suppressErrorDialogs);
            return false;
        }
    }

    // The builder's frame signs are inherited from an unoriented manifest
    // normal at an anchor that moves as the line grows; hand it the sheet
    // normals oriented away from the scroll center so the bottom strip's
    // vertical, the display ups, and the cut views all agree per fiber and
    // across rebuilds.
    std::vector<cv::Vec3f> orientedNormals = orientedLineNormalsForSession(session);
    session.displayField = displayFieldForSession(session, orientedNormals);
    for (const size_t index : session.displayField.resetControls)
        session.controlPoints[index].displayNormal.reset();
    if (!session.displayField.resetControls.empty()) {
        for (auto& other : _panes) {
            if (!other.session || other.session.get() == &session || !session.fiberId ||
                other.session->fiberId != session.fiberId) continue;
            for (const size_t index : session.displayField.resetControls)
                for (auto& cp : other.session->controlPoints)
                    if (pointsApproximatelyEqual(cp.volumePoint, session.controlPoints[index].volumePoint))
                        cp.displayNormal.reset();
        }
        saveFiberDisplayAnnotations(session, stillValid);
        // The metadata save can show a save-error modal of its own.
        if (!alive || (stillValid && !stillValid())) {
            return false;
        }
        showError(tr("Normal offset reset to 0 for %1 control points because their stored normals could not be projected into the cross-section plane.")
                      .arg(session.displayField.resetControls.size()), session.suppressErrorDialogs);
        // The warning is modal: nothing of the controller or the session is
        // touched, and nothing published, unless both are still valid.
        if (!alive || (stillValid && !stillValid())) {
            return false;
        }
    }
    const bool hasManualNormals = std::any_of(session.controlPoints.begin(), session.controlPoints.end(),
        [](const auto& control) { return control.displayNormal.has_value(); });

    vc::lasagna::LineViewSurfaces views;
    vc::lasagna::LineModel displayLine;
    try {
        displayLine = session.optimizedLine;
        vc::lasagna::LineViewConfig viewConfig;
        viewConfig.controlPointLinePositions.reserve(session.controlPoints.size());
        for (const auto& control : session.controlPoints) {
            viewConfig.controlPointLinePositions.push_back(control.linePosition);
        }
        viewConfig.orientedPointNormals = orientedNormals;
        if (hasManualNormals)
            viewConfig.displayPointNormals = session.displayField.normals;
        // Only lineUpVectors is consumed below; skip the per-line-point
        // PlaneSurface allocations.
        viewConfig.buildLineZSlices = false;
        views = vc::lasagna::buildLineViewSurfaces(displayLine, viewConfig);
    } catch (const std::exception& ex) {
        session.error = ex.what();
        showError(tr("Could not create line annotation views: %1")
                      .arg(QString::fromStdString(session.error)),
                  session.suppressErrorDialogs);
        return false;
    }

    std::vector<cv::Vec3f> linePoints;
    linePoints.reserve(displayLine.points.size());
    for (const auto& point : displayLine.points) {
        linePoints.push_back({static_cast<float>(point.position[0]),
                              static_cast<float>(point.position[1]),
                              static_cast<float>(point.position[2])});
    }

    // Anchor the new strips' parameterization on what the live strip cameras
    // are viewing BEFORE any registration: the strip viewers keep their raw
    // camera coordinates when they adopt the replacements, so shifting the
    // new surfaces (instead of correcting the cameras afterwards) is the only
    // ordering where no frame — including the adoption-triggered render — is
    // ever rendered with the drifted arc-length mapping.
    // The markers are built here, before the anchoring, because the anchoring
    // carries the strip cameras' positions relative to the controls; the
    // views below adopt this same list.
    std::vector<LineAnnotationDialog::GeneratedOverlay::ControlPointMarker>
        controlMarkers = controlMarkersForSession(session);
    if (auto* anchorPane = paneForSurface(session.surfaceName);
        anchorPane && anchorPane->dialog) {
        anchorPane->dialog->anchorGeneratedStripSurfacesForUpdate(
            views.lineSurface.get(),
            views.lineSideSlice.get(),
            views.stripPositionMap,
            linePoints,
            controlMarkers,
            session.lineRevision);
    }

    // Everything the session currently shows, retained so a failed install can
    // put it back. The registered surfaces are replaced under the same names
    // before the dialog has accepted their replacements — live viewers render
    // the new planes immediately, and the pose carry-over below deliberately
    // reads what is registered at replacement time — so a failure must restore
    // rather than blank. That mattered less when only an edit reached this
    // path; it now also runs after umbilicus and volume changes, where the
    // user changed nothing and losing the on-screen views is a pure loss.
    const std::vector<std::string> previousSurfaceNames =
        session.generatedSurfaceNames;
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>>
        previousSurfaces;
    previousSurfaces.reserve(previousSurfaceNames.size());
    for (const auto& name : previousSurfaceNames) {
        previousSurfaces.emplace_back(name, _state->surface(name));
    }
    const std::string previousLineSurfaceName = session.generatedLineSurfaceName;
    const std::string previousLineSideSliceName =
        session.generatedLineSideSliceName;

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

    const cv::Vec3d displaySeed = session.seedPoint;
    const cv::Vec3f seedPoint{static_cast<float>(displaySeed[0]),
                              static_cast<float>(displaySeed[1]),
                              static_cast<float>(displaySeed[2])};
    const cv::Vec3f focusPoint = session.focusedControlPoint
        ? toVec3f(*session.focusedControlPoint)
        : seedPoint;

    LineAnnotationDialog::GeneratedViews generatedViews;
    generatedViews.fiberWidth = session.fiberWidth;
    generatedViews.fiberWidthGapFraction = session.fiberWidthGapFraction;
    generatedViews.hasManualDisplayNormals = hasManualNormals;
    for (size_t i = 0; i < session.displayField.controlOffsets.size(); ++i)
        generatedViews.controlAngleOffsetsDegrees.push_back(
            cv::norm(session.displayField.controlBaselines[i]) > 0
                ? session.displayField.controlOffsets[i] * 180.0 / std::acos(-1.0)
                : std::numeric_limits<double>::quiet_NaN());
    generatedViews.lineSurfaceName = session.generatedLineSurfaceName;
    generatedViews.lineSurfaceTitle = tr("Line Surface");
    generatedViews.lineSurface = views.lineSurface;
    generatedViews.lineSideSliceName = session.generatedLineSideSliceName;
    generatedViews.lineSideSliceTitle = tr("Line Side Slice");
    generatedViews.lineSideSlice = views.lineSideSlice;
    generatedViews.linePoints = std::move(linePoints);
    generatedViews.lineUpVectors = views.lineUpVectors;
    generatedViews.stripPositionMap = views.stripPositionMap;
    generatedViews.stripPositionMap.lineRevision = session.lineRevision;
    generatedViews.lineNormals = std::move(orientedNormals);
    if (hasManualNormals) generatedViews.displayLineNormals = session.displayField.normals;
    {
        // Winding angles for the side cut's half-wrap window (see the side
        // overlay in LineAnnotationDialog). Same center reference chain as
        // orientedLineNormalsForSession, which already loaded the umbilicus:
        // umbilicus, else the volume center; with neither the angles stay NaN
        // and the side cut shows the whole line as before.
        constexpr float kNanF = std::numeric_limits<float>::quiet_NaN();
        cv::Vec2f volumeCenterXY{kNanF, kNanF};
        try {
            if (const auto volume = _state->currentVolume()) {
                volumeCenterXY = {static_cast<float>(volume->sliceWidth()) * 0.5f,
                                  static_cast<float>(volume->sliceHeight()) * 0.5f};
            }
        } catch (...) {
        }
        generatedViews.lineWindingAngles =
            vc3d::line_annotation::unwrappedGeneratedWindingAngles(
                generatedViews.linePoints,
                [this, volumeCenterXY](const cv::Vec3f& point) -> cv::Vec3f {
                    if (_scrollUmbilicus) {
                        try {
                            return _scrollUmbilicus->vector_to_umbilicus(point);
                        } catch (...) {
                        }
                    }
                    if (std::isfinite(volumeCenterXY[0])) {
                        return {volumeCenterXY[0] - point[0],
                                volumeCenterXY[1] - point[1],
                                0.0f};
                    }
                    return {kNanF, kNanF, kNanF};
                });
    }
    generatedViews.branchLinePoints = generatedBranchLinePointsForSession(session);
    generatedViews.branchLinks = generatedBranchLinkMarkers(session.branches);
    generatedViews.seedPoint = seedPoint;
    generatedViews.focusPoint = focusPoint;
    generatedViews.lineRevision = session.lineRevision;
    generatedViews.seedLineIndex = static_cast<int>(session.optimizedLine.points.size() / 2);
    generatedViews.initialCurrentCutFollowsStripMouse =
        !session.disableInitialGeneratedHoverFollow;
    generatedViews.controlPoints = std::move(controlMarkers);
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
    generatedViews.initialFitWholeLine = session.initialFitWholeLine;
    generatedViews.spanAlignmentMetrics =
        generatedSpanAlignmentMetricsForSession(session);

    // Re-registration replaces planes that live viewers render immediately, and
    // the dialog only re-poses them to the current line position afterwards.
    // Seed each new plane with the previous plane's pose (when one exists) so
    // the surfaceChanged frame is visually continuous instead of flashing the
    // seed-point view on every control-point placement.
    const auto carryOverPreviousPose = [this](const std::string& name,
                                              const std::shared_ptr<PlaneSurface>& plane) {
        const auto previous =
            std::dynamic_pointer_cast<PlaneSurface>(_state->surface(name));
        if (!previous) {
            return;
        }
        plane->setFromNormalAndUp(previous->origin(),
                                  previous->normal({0.0f, 0.0f, 0.0f}),
                                  previous->basisY());
    };

    generatedViews.currentCutName = generatedPrefix + "_line_current_cut";
    generatedViews.currentCutSurface = std::make_shared<PlaneSurface>(
        seedPoint,
        cv::Vec3f{1.0f, 0.0f, 0.0f});
    carryOverPreviousPose(generatedViews.currentCutName, generatedViews.currentCutSurface);
    _state->setSurface(generatedViews.currentCutName, generatedViews.currentCutSurface);
    session.generatedSurfaceNames.push_back(generatedViews.currentCutName);

    generatedViews.sideCutName = generatedPrefix + "_line_side_cut";
    generatedViews.sideCutSurface = std::make_shared<PlaneSurface>(
        seedPoint,
        cv::Vec3f{1.0f, 0.0f, 0.0f});
    carryOverPreviousPose(generatedViews.sideCutName, generatedViews.sideCutSurface);
    _state->setSurface(generatedViews.sideCutName, generatedViews.sideCutSurface);
    session.generatedSurfaceNames.push_back(generatedViews.sideCutName);

    auto* pane = paneForSurface(session.surfaceName);
    if (!pane || !pane->dialog) {
        // Recorded only on success, and only once every surface is in place: an
        // epoch written earlier would claim the new orientation while a later
        // throw or failure had left the old, partial or cleared views on screen,
        // and the next refresh would then skip the rebuild that would fix it.
        session.orientationEpoch = _orientationEpoch;
        return true;
    }

    CChunkedVolumeViewer::CameraState camera;
    camera.scale = 1.0f;
    if (!pane->dialog->panes().empty() && pane->dialog->panes().front().viewer) {
        camera = pane->dialog->panes().front().viewer->cameraState();
        camera.zOffset = 0.0f;
        camera.zOffsetWorldDir = {0, 0, 0};
    }

    if (!pane->dialog->setGeneratedLineViews(std::move(generatedViews), camera)) {
        // Restore what was on screen: drop the rejected replacements, then
        // re-register the retained surfaces under their old names and hand the
        // session its previous metadata back. The epoch is deliberately not
        // advanced (success-only, above), so the next umbilicus or volume
        // change retries this rebuild instead of skipping it.
        for (const auto& name : session.generatedSurfaceNames) {
            _state->setSurface(name, nullptr);
        }
        for (const auto& [name, surface] : previousSurfaces) {
            if (surface) {
                _state->setSurface(name, surface);
            }
        }
        session.generatedSurfaceNames = previousSurfaceNames;
        session.generatedLineSurfaceName = previousLineSurfaceName;
        session.generatedLineSideSliceName = previousLineSideSliceName;
        session.error = "Failed to create generated annotation viewers.";
        showError(tr("Could not create generated line annotation viewers."),
                  session.suppressErrorDialogs);
        return false;
    }
    if (!session.inspectionFrameOnly && !session.modelOnly && session.fiberId != 0 &&
        session.fiberMetricsMatchStoredFiber &&
        !hasCachedAlignmentForFiber(session.fiberId) &&
        !isAlignmentPendingForFiber(session.fiberId)) {
        requestFiberAlignmentMetrics(session.fiberId);
    }
    // Every line publish is followed by a solve-activity report, so the
    // dialog's overview bar can confirm the geometry whatever path published
    // it (the synchronous finalizations included).
    syncDialogLineSolveActivity(session);
    // See the note at the other success return: only a completed build may claim an
    // orientation.
    session.orientationEpoch = _orientationEpoch;
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
        showError(tr("Run line optimization before exporting generated meshes."),
                  session.suppressErrorDialogs);
        return;
    }
    if (!finalizeSessionOptimizationSynchronously(session, false)) {
        return;
    }
    if (!session.suppressGeneratedViews && !materializeGeneratedViews(session)) {
        return;
    }

    try {
        const auto savedNames = saveGeneratedQuadMeshes(session);
        if (savedNames.empty()) {
            showError(tr("No generated line quad meshes are available to export."),
                      session.suppressErrorDialogs);
            return;
        }

        QStringList labels;
        labels.reserve(static_cast<int>(savedNames.size()));
        for (const auto& name : savedNames) {
            labels.push_back(QString::fromStdString(name));
        }
        if (!session.suppressErrorDialogs && !_errorDialogsSuppressed) {
            QMessageBox::information(_parentWidget,
                                     tr("Line Annotation"),
                                     tr("Saved generated mesh surfaces:\n%1")
                                         .arg(labels.join(QStringLiteral("\n"))));
        }
    } catch (const std::exception& ex) {
        showError(tr("Could not save generated line meshes: %1")
                      .arg(QString::fromStdString(ex.what())),
                  session.suppressErrorDialogs);
    }
}

std::vector<std::string> LineAnnotationController::saveGeneratedQuadMeshes(LineAnnotationSession& session)
{
    auto* catalog = _surfacePanel ? _surfacePanel->catalog() : nullptr;
    if (!_state || !catalog) {
        throw std::runtime_error("No active application state.");
    }

    const std::vector<std::pair<std::string, std::string>> exports = {
        {session.generatedLineSurfaceName, "surface"},
        {session.generatedLineSideSliceName, "side_slice"},
    };
    const std::string timestamp =
        QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss")).toStdString();

    std::vector<std::string> names;
    for (const auto& [surfaceName, stem] : exports) {
        if (surfaceName.empty()) {
            continue;
        }
        auto surface = std::dynamic_pointer_cast<QuadSurface>(_state->surface(surfaceName));
        if (!surface) {
            continue;
        }

        QuadSurface clone(surface->rawPoints().clone(), surface->scale());
        clone.meta = surface->meta;
        const std::string name = "line_annotation_" + timestamp + "_" + stem;
        const nlohmann::json provenance{{"tool", "vc3d-line-annotation"},
                                        {"parameters", {{"view", stem}}},
                                        {"sources", session.fiberUuid.empty() ? nlohmann::json::array()
                                                                              : nlohmann::json::array({session.fiberUuid})}};
        catalog->create(clone, name, provenance);
        names.push_back(name);
    }

    Logger()->info("Line annotation saved {} generated mesh surface(s) to the object store", names.size());
    return names;
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
            // The orphaned solve's result has no pane to land in: ask it to
            // stop so it frees its pool slot, and shut the queue so nothing
            // re-dispatches. Once the pane record is erased below, the
            // destructor's teardown loop can no longer reach this session.
            if (pane.session->runningSolveCancel) {
                pane.session->runningSolveCancel->store(true);
            }
            pane.session->solveQueue.shutdown();
        }
        if (pane.surfaceName == surfaceName && pane.session) {
            generatedSurfaceNames = pane.session->generatedSurfaceNames;
            // Session-cache entries are keyed by fiberId and validated by
            // (lineRevision, epoch), both of which restart when the fiber is
            // reopened in a fresh session -- a surviving entry could then
            // pass validation while holding the previous session's geometry.
            _sideStripSessionSnapshotCache.erase(pane.session->fiberId);
            if (!pane.session->readOnly &&
                !pane.session->suppressFiberSave &&
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
        invalidateSideStripQueries();
    }
    std::erase_if(_pendingSideStripIntersectionRequests,
                  [&matchesClosingSurface](const auto& item) {
                      return matchesClosingSurface(item.first);
                  });
    refreshLatestSideStripIntersectionToken();
    std::erase_if(_sideStripReuseCache,
                  [&matchesClosingSurface](const auto& item) {
                      return matchesClosingSurface(item.first);
                  });

    // The save and every surface notification below can run arbitrary
    // slots (a modal, a teardown): nothing of `this` is touched afterwards
    // unless it is alive. The names are held by value.
    const QPointer<LineAnnotationController> alive(this);
    if (sessionToSave) {
        saveSessionAsFiber(*sessionToSave);
        if (!alive) {
            return;
        }
    }

    if (_state) {
        const QPointer<CState> state(_state);
        state->setSurface(surfaceName, nullptr);
        for (const auto& name : generatedSurfaceNames) {
            if (!alive || !state) {
                return;
            }
            state->setSurface(name, nullptr);
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
    std::vector<vc3d::line_annotation::LineControlPoint> controlPoints,
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
                                    1.0,
                                    std::move(controlPoints),
                                    std::move(initialLinePoints),
                                    sourceSliceNormal,
                                    directionMode,
                                    initialCenterlineLengthVx,
                                    forceFullOptimization,
                                    activeStart,
                                    activeEnd,
                                    breakExtrapolationDistanceVxFromSettings());
}

int LineAnnotationController::breakExtrapolationDistanceVxForPane(const PaneRecord* pane) const
{
    if (pane && pane->dialog) {
        return pane->dialog->breakExtrapolationDistanceVx();
    }
    return breakExtrapolationDistanceVxFromSettings();
}


void LineAnnotationController::reoptimizeMergedFibers(
    const std::vector<uint64_t>& fiberIds)
{
    auto stripReoptimizationTag = [](std::vector<std::string>& tags) {
        tags.erase(std::remove(tags.begin(), tags.end(),
                               std::string{kNeedsReoptimizationTag}),
                   tags.end());
    };

    // One dataset/sampler serves the whole batch; failing to resolve it
    // (or cancelling the picker) aborts the batch instead of prompting
    // once per fiber.
    std::shared_ptr<vc::lasagna::LasagnaDataset> dataset;
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
    fs::path manifestPath;
    std::string datasetLocation;
    double workingToBaseScale = 1.0;

    const QPointer<LineAnnotationController> alive(this);
    // The dataset picker and each save can yield. Stop if the package changes.
    const uint64_t packageGeneration = _packageGeneration;
    int reoptimized = 0;
    bool scheduledTagOnlySave = false;
    for (const uint64_t fiberId : fiberIds) {
        if (_packageGeneration != packageGeneration) {
            Logger()->warn("Re-optimization abandoned: the package changed");
            return;
        }
        auto stored = readStoredFiber(fiberId, true);
        if (!stored || !stored->editorError.empty()) {
            continue;
        }
        // Collected before a modal prompt; another writer may have changed
        // the fiber since - only still-tagged fibers are ours to touch.
        if (std::find(stored->tags.begin(), stored->tags.end(),
                      kNeedsReoptimizationTag) == stored->tags.end()) {
            continue;
        }
        const std::string fileName = stored->fileName;
        if (stored->controlPoints.size() < 2 || stored->linePoints.size() < 2) {
            // No line to fit; just clear the tag.
            stripReoptimizationTag(stored->tags);
            try {
                commitFibers({&*stored});
                scheduledTagOnlySave = true;
            } catch (const std::exception& ex) {
                Logger()->warn("Could not save {} after clearing the {} "
                               "tag: {}",
                               fileName, kNeedsReoptimizationTag, ex.what());
            }
            continue;
        }
        // Pane-less session: the surface name is never registered, so no
        // dialog, pane, or generated view is created. saveSessionAsFiber
        // finalizes the optimization synchronously, refreshes branch
        // endpoint metadata on this fiber and its linked peers, bumps the
        // generation, and schedules the writes.
        auto session = makeIntersectionLineSession(
            *stored,
            static_cast<double>(stored->linePoints.size() / 2),
            cv::Vec3d{0.0, 0.0, 0.0},
            "fiber_reoptimization_" + fileName,
            nullptr);
        stripReoptimizationTag(session->fiberTags);
        // Arms finalizeSessionOptimizationSynchronously inside the save.
        session->optimizationState = SessionOptimizationState::Unoptimized;
        if (!dataset) {
            if (!ensureDatasetForSession(*session) || !alive) {
                if (alive) {
                    Logger()->warn("Re-optimization aborted: no usable "
                                   "Lasagna dataset; the {} tag(s) are kept "
                                   "for the next load",
                                   kNeedsReoptimizationTag);
                }
                return;
            }
            if (_packageGeneration != packageGeneration) {
                Logger()->warn("Re-optimization abandoned: the package changed "
                               "while the dataset was being chosen");
                return;
            }
            dataset = session->dataset;
            sampler = session->normalSampler;
            manifestPath = session->selectedManifestPath;
            datasetLocation = session->selectedDatasetLocation;
            workingToBaseScale = session->workingToBaseScale;
        } else {
            // Pre-seeded so ensureDatasetForSession (inside the finalize)
            // reuses it instead of reopening per fiber.
            session->dataset = dataset;
            session->normalSampler = sampler;
            session->selectedManifestPath = manifestPath;
            session->selectedDatasetLocation = datasetLocation;
            session->workingToBaseScale = workingToBaseScale;
        }
        saveSessionAsFiber(*session);
        if (!alive) {
            return;
        }
        if (session->optimizationState == SessionOptimizationState::Optimized) {
            ++reoptimized;
        } else {
            // saveSessionAsFiber already surfaced the error dialog; a
            // failure here is almost certainly systematic (dataset,
            // volume), so abort the batch instead of stacking one modal
            // per remaining fiber. All untouched tags are kept for a
            // retry on the next load.
            Logger()->warn("Re-optimization failed for fiber {}; aborting "
                           "the batch — the {} tag(s) are kept for a retry "
                           "on next load",
                           fileName,
                           kNeedsReoptimizationTag);
            break;
        }
    }
    if (reoptimized > 0 || scheduledTagOnlySave) {
        if (reoptimized > 0) {
            Logger()->info("Re-optimized {} merged fiber(s)", reoptimized);
        }
        // Wait for ALL scheduled writes (including tag-strip-only saves)
        // before returning: the caller clears the prompt-dedup flag right
        // after, and a later read must see the tag-free fibers.
        waitForFiberSaves();
        if (!alive) {
            return;
        }
    }
}

uint64_t LineAnnotationController::fiberIdForFilePath(const fs::path& path) const
{
    std::string uuid;
    try {
        std::ifstream input(path);
        const auto document=nlohmann::json::parse(input);
        uuid = vc::objects::util::canonicalUuid(document.at("uuid").get<std::string>());
    } catch(...) { return 0; }
    // Only a fiber the attached store holds has a runtime id.
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment) return 0;
    bool active = false;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto object = db.objectByUuid(txn, uuid);
        active = object && object->kind == "fiber" && !object->retired;
    });
    return active ? fiberIdForUuid(uuid) : 0;
}

// Check the runtime identity against the store's current display label.
bool LineAnnotationController::storedFiberHasFileName(uint64_t fiberId, const std::string& fileName) const
{
    const auto uuid = fiberUuidForId(fiberId);
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (uuid.empty() || !attachment) return false;
    bool matches = false;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto object = db.objectByUuid(txn, uuid);
        matches = object && !object->retired && object->name + ".json" == fileName;
    });
    return matches;
}



uint64_t LineAnnotationController::fiberIdForDialog(const LineAnnotationDialog* dialog) const
{
    if (!dialog) {
        return 0;
    }
    for (const auto& pane : _panes) {
        if (pane.dialog.data() == dialog && pane.session) {
            // The session's own runtime id: a basename lookup could land on
            // another working copy's fiber of the same name.
            return pane.session->fiberId;
        }
    }
    return 0;
}

void LineAnnotationController::retainStoredFiberCoordinateBaseShape(StoredFiber& fiber) const
{
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->fiberId == fiber.id &&
            pane.session->fiberUuid == fiber.uuid &&
            !pane.session->coordinateBaseShapeZYX)
            pane.session->coordinateBaseShapeZYX = fiber.coordinateBaseShapeZYX;
    }
}

fs::path LineAnnotationController::relativeFiberPath(const StoredFiber& fiber) const
{
    if (!fiber.storePin) throw std::runtime_error("Fiber has not been committed to an object store");
    return fs::path(fiber.storePin->object.storeUuid)/fiber.uuid;
}

std::map<std::string, fs::path> LineAnnotationController::fiberTransportPaths(
    const std::vector<uint64_t>& fiberIds) const
{
    if (fiberIds.empty()) return {};
    const auto connection = _state->objectStoreSession();
    const auto attachment = _state->objectStoreAttachment();
    if (!attachment) throw std::runtime_error("Attach an object store before exporting fibers");
    std::vector<std::string> roots;
    for (const auto id : fiberIds) {
        const auto pin = fiberPin(id);
        if (!pin || pin->attachment != *attachment)
            throw std::runtime_error("Fiber is not in the attached object store");
        roots.push_back(pin->object.objectUuid);
    }
    std::map<std::string, std::string> documents;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        vc::objects::ObjectSelection objects; objects.kind = "fiber"; objects.includeRetired = true;
        documents = vc::objects::fiberNetworkJson(db, txn, db.currentRevisions(txn, objects), roots);
    });
    return vc3d::materializeJsonDocuments(_fiberTransportDirectory, *attachment, documents);
}

std::optional<LineAnnotationController::ResolvedFiberOptimizationInputs>
LineAnnotationController::resolveFiberOptimizationInputs(
    const std::string& fallbackNormalLocation,
    const std::string& fallbackFiberLocation,
    QString* errorMessage) const
{
    if (!_state || !_state->vpkg()) {
        if (errorMessage) *errorMessage = tr("No volume package is loaded.");
        return std::nullopt;
    }
    auto vpkg = _state->vpkg();
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = vc3d::remoteCachePathFs();
    const auto resolveLocation = [vpkg](const std::string& location) {
        return vc::project::isLocationRemote(location)
            ? location
            : vc::project::resolveLocalPath(
                  location, vpkg->path().parent_path()).string();
    };

    ResolvedFiberOptimizationInputs result;
    QStringList failures;
    const std::vector<std::string> fiberCandidates = {
        vpkg->selectedFiberInferenceDataset(), fallbackFiberLocation};
    for (const auto& candidate : fiberCandidates) {
        if (candidate.empty() || result.fiberDataset) continue;
        try {
            auto opened = vc::lasagna::LasagnaDataset::openLocation(
                resolveLocation(candidate), options);
            const auto scales =
                vc::fiber_tracer::resolveFiberPredictionTraceScales(opened.manifest());
            auto manifest = opened.manifest();
            manifest.workingToBaseScale = scales.traceToBaseScale;
            auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                std::move(manifest));
            auto predictions =
                std::make_shared<vc::fiber_tracer::FiberPredictionField>(*dataset);
            result.fiberManifestLocation = candidate;
            result.traceToBaseScale = scales.traceToBaseScale;
            result.fiberDataset = std::move(dataset);
            result.predictions = std::move(predictions);
        } catch (const std::exception& ex) {
            failures.push_back(tr("Fiber-inference dataset %1: %2")
                                   .arg(QString::fromStdString(candidate),
                                        QString::fromUtf8(ex.what())));
        }
    }

    const std::vector<std::string> normalCandidates = {
        vpkg->selectedLasagnaDataset(), fallbackNormalLocation};
    for (const auto& candidate : normalCandidates) {
        if (candidate.empty() || result.normalDataset || !result.fiberDataset)
            continue;
        try {
            auto opened = vc::lasagna::LasagnaDataset::openLocation(
                resolveLocation(candidate), options);
            const auto coordinateScales =
                vc3d::line_annotation::resolveFiberNormalCoordinateScales(
                    opened.manifest().baseShapeZYX,
                    result.fiberDataset->manifest().baseShapeZYX,
                    result.traceToBaseScale);

            auto baseManifest = opened.manifest();
            baseManifest.workingToBaseScale =
                coordinateScales.fiberBaseToNormalBase;
            auto baseDataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                std::move(baseManifest));
            auto baseSampler =
                std::make_shared<vc::lasagna::LasagnaNormalSampler>(*baseDataset);

            auto traceManifest = opened.manifest();
            traceManifest.workingToBaseScale =
                coordinateScales.traceToNormalBase;
            auto traceDataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                std::move(traceManifest));
            auto traceSampler =
                std::make_shared<vc::lasagna::LasagnaNormalSampler>(*traceDataset);

            result.normalManifestLocation = candidate;
            result.normalDataset = std::move(baseDataset);
            result.baseNormalSampler = std::move(baseSampler);
            result.traceNormalDataset = std::move(traceDataset);
            result.traceNormalSampler = std::move(traceSampler);
        } catch (const std::exception& ex) {
            failures.push_back(tr("Normal dataset %1: %2")
                                   .arg(QString::fromStdString(candidate),
                                        QString::fromUtf8(ex.what())));
        }
    }
    if (!result.normalDataset || !result.fiberDataset) {
        if (errorMessage) {
            QString missing;
            if (!result.normalDataset) missing += tr("No usable Lasagna normal dataset. ");
            if (!result.fiberDataset) missing += tr("No usable fiber-inference dataset. ");
            *errorMessage = missing + failures.join(QStringLiteral("; "));
        }
        return std::nullopt;
    }
    return result;
}

void LineAnnotationController::optimizeAndSaveFiberHeadless(
    HeadlessFiberOptimizationRequest request,
    HeadlessFiberCompletion completion)
{
    auto fail = [completion](const QString& message) {
        if (completion) completion(false, message, 0);
    };
    if (request.controlPointsL0.size() < 2) {
        fail(tr("A 2D line annotation needs at least two control points."));
        return;
    }
    for (size_t i = 0; i < request.controlPointsL0.size(); ++i) {
        if (!finitePoint(request.controlPointsL0[i])) {
            fail(tr("A line control point contains non-finite coordinates."));
            return;
        }
        for (size_t j = 0; j < i; ++j) {
            if (pointsApproximatelyEqual(request.controlPointsL0[i],
                                         request.controlPointsL0[j])) {
                fail(tr("A 2D line annotation needs at least two distinct points and "
                        "cannot contain duplicate controls."));
                return;
            }
        }
    }
    // With the fiber-follow model the spans are traced by it and both ends are
    // extended by the Line Annotation extrapolation distance; otherwise the
    // native tracer keeps the drawn controls as the fiber's bounds.
    const bool fiberFollow = fiberFollowEnabled();
    if (fiberFollow && !ensureFiberFollowService(true)) {
        fail(tr("The fiber-follow model could not start; see the log."));
        return;
    }
    if (!request.inputs.baseNormalSampler ||
        (!fiberFollow && (!request.inputs.traceNormalSampler || !request.inputs.predictions))) {
        fail(tr("Fiber optimization inputs are incomplete."));
        return;
    }
    vc3d::line_annotation::FiberModeOptimizationRequest tracerTemplate;
    tracerTemplate.traceToBaseScale = request.inputs.traceToBaseScale;
    double fiberFollowExtrapolation = 0.0;
    if (fiberFollow) {
        installFiberFollowTracer(tracerTemplate, {});
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        fiberFollowExtrapolation = settings.value(
            vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX,
            vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX_DEFAULT).toDouble();
    }
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment) { fail(tr("Attach an object store before creating fibers.")); return; }
    auto shouldSave = std::move(request.shouldSave);
    const auto coordinateBaseShapeZYX = request.inputs.fiberDataset
        ? request.inputs.fiberDataset->manifest().baseShapeZYX
        : request.inputs.normalDataset->manifest().baseShapeZYX;
    if (!coordinateBaseShapeZYX) {
        fail(tr("Inference manifest must declare base_shape_zyx before creating fibers."));
        return;
    }

    nlohmann::json creationBinding;
    try {
        const auto volume=_state->currentVolume();
        if(!volume) throw std::runtime_error("An active volume is required to bind a new fiber");
        const auto scale=vc3d::line_annotation::resolveFiberBaseToVolumeScale(coordinateBaseShapeZYX,volume->shape());
        creationBinding=coordinateBindingForVolume(*volume,1.0/scale);
    } catch(const std::exception& error) {fail(QString::fromUtf8(error.what()));return;}

    struct HeadlessResult {
        bool ok = false;
        std::vector<vc3d::line_annotation::LineControlPoint> controls;
        std::vector<cv::Vec3d> line;
        QString error;
    };
    auto* watcher = new QFutureWatcher<HeadlessResult>(this);
    connect(watcher, &QFutureWatcher<HeadlessResult>::finished, this,
            [this, watcher, completion, coordinateBaseShapeZYX, attachment, creationBinding,
             shouldSave = std::move(shouldSave)]() mutable {
                HeadlessResult result = watcher->result();
                watcher->deleteLater();
                if (!result.ok) {
                    if (completion) completion(false, result.error, 0);
                    return;
                }
                if (!_state->objectStoreSession()->current(*attachment) || (shouldSave && !shouldSave())) {
                    if (completion) {
                        completion(false,
                                   tr("The Spiral preview or session changed before "
                                      "the annotation could be saved."),
                                   0);
                    }
                    return;
                }
                try {
                    StoredFiber fiber;
                    fiber.id = nextFiberId();
                    fiber.storePin = vc::objects::PinnedObject{*attachment,{attachment->storeUuid,fiber.uuid},"",{}};
                    fiber.username = currentFiberUsername();
                    fiber.startedAt = currentFiberDateTimeString();
                    fiber.sequence = nextFiberSequenceForUsername(fiber.username);
                    fiber.fileName=vc3d::line_annotation::fiberFileName(fiber.username,fiber.startedAt,fiber.sequence);
                    fiber.generation = 1;
                    fiber.optimizationMode =
                        vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d;
                    fiber.coordinateBaseShapeZYX = coordinateBaseShapeZYX;
                    fiber.linePoints = std::move(result.line);
                    fiber.controlPoints.reserve(result.controls.size());
                    for (auto& control : result.controls) {
                        vc3d::line_annotation::StoredControlPoint stored{
                            control.volumePoint};
                        stored.segmentToNext = std::move(control.segmentToNext);
                        stored.tags = std::move(control.tags);
                        stored.displayNormal = control.displayNormal;
                        stored.direction = control.direction;
                        stored.displayNormalSource = control.displayNormalSource;
                        fiber.controlPoints.push_back(std::move(stored));
                    }
                    fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(
                        vc3d::line_annotation::storedControlPointPositions(
                            fiber.controlPoints));
                    const uint64_t id = fiber.id;
                    std::vector<FiberSaveSnapshot> snapshots{makeFiberSaveSnapshot(fiber)};
                    snapshots.front().fiber.coordinateIdentity.update(creationBinding);
                    commitFiberStoreAction(snapshots);
                    applyCommittedFibers(snapshots);
                    refreshBranchLineViews(id);
                    if (completion) completion(true, {}, id);
                } catch (const std::exception& ex) {
                    if (completion) {
                        completion(false,
                                   tr("Could not save Spiral line annotation: %1")
                                       .arg(QString::fromUtf8(ex.what())),
                                   0);
                    }
                }
            });
    watcher->setFuture(QtConcurrent::run(
        [request = std::move(request),
         externalExtrapolation = std::move(tracerTemplate.externalExtrapolation),
         externalSegment = std::move(tracerTemplate.externalSegment),
         fiberFollowExtrapolation,
         breakExtrapolation = static_cast<double>(breakExtrapolationDistanceVxFromSettings())]() mutable {
            HeadlessResult output;
            try {
                vc::lasagna::LineOptimizationConfig config;
                const auto discretization = initialLineDiscretization(400);
                config.segmentsPerSide = discretization.segmentsPerSide;
                config.segmentLength = discretization.segmentLength;
                config.straightnessWeight = 0.1;
                config.tangentStraightnessWeight = 5.0;
                config.samplesPerSegment = 1;
                config.maxIterations = 1000;
                config.differentiableNormalSampling = true;
                config.printSolverProgress = false;

                std::vector<vc3d::line_annotation::LineControlPoint> controls;
                controls.reserve(request.controlPointsL0.size());
                for (size_t i = 0; i < request.controlPointsL0.size(); ++i) {
                    controls.emplace_back(static_cast<double>(i),
                                          request.controlPointsL0[i], i == 0, -1);
                }
                vc::lasagna::LineOptimizer initializer(
                    *request.inputs.baseNormalSampler);
                auto initialized = initializer.optimizeFromControlPoints(
                    vc3d::line_annotation::optimizerControlPoints(controls), config);
                std::vector<cv::Vec3d> initialLine;
                initialLine.reserve(initialized.line.points.size());
                for (const auto& point : initialized.line.points)
                    initialLine.push_back(point.position);
                size_t previous = 0;
                for (size_t i = 0; i < controls.size(); ++i) {
                    const size_t index = vc3d::fiber_slice::nearestLinePointIndex(
                        initialLine, controls[i].volumePoint);
                    if (i > 0 && index <= previous) {
                        throw std::runtime_error(
                            "Lasagna initialization did not preserve control-point order");
                    }
                    previous = index;
                    controls[i].linePosition = static_cast<double>(index);
                    controls[i].optimizedIndex = static_cast<int>(index);
                }

                vc3d::line_annotation::FiberModeOptimizationRequest fiberRequest;
                fiberRequest.controlPoints = std::move(controls);
                fiberRequest.linePointsBase = std::move(initialLine);
                fiberRequest.predictions = request.inputs.predictions.get();
                fiberRequest.baseNormalSampler = request.inputs.baseNormalSampler.get();
                fiberRequest.traceNormalSampler = request.inputs.traceNormalSampler.get();
                fiberRequest.lasagnaConfig = config;
                fiberRequest.normalManifestLocation =
                    request.inputs.normalManifestLocation;
                fiberRequest.fiberManifestLocation =
                    request.inputs.fiberManifestLocation;
                fiberRequest.traceToBaseScale = request.inputs.traceToBaseScale;
                fiberRequest.globalMode =
                    vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d;
                fiberRequest.retainOpenTails = false;
                fiberRequest.extrapolationDistanceBaseVoxels = 0.0;
                fiberRequest.breakExtrapolationDistanceBaseVoxels = breakExtrapolation;
                if (externalSegment) {
                    fiberRequest.externalExtrapolation = std::move(externalExtrapolation);
                    fiberRequest.externalSegment = std::move(externalSegment);
                    fiberRequest.retainOpenTails = true;
                    fiberRequest.extrapolationDistanceBaseVoxels = fiberFollowExtrapolation;
                }
                auto optimized = vc3d::line_annotation::
                    optimizeFiberWithNativeFallback(std::move(fiberRequest));
                output.controls = std::move(optimized.controlPoints);
                output.line.reserve(optimized.optimization.line.points.size());
                for (const auto& point : optimized.optimization.line.points)
                    output.line.push_back(point.position);
                output.ok = true;
            } catch (const std::exception& ex) {
                output.error = QString::fromUtf8(ex.what());
            }
            return output;
        }));
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
    return {vc::atlas::atlasFiberPathKey(relativeFiberPath(fiber))};
}

std::optional<fs::path> LineAnnotationController::resolveAtlasFiberPath(
    const StoredFiber& fiber,
    const fs::path& atlasDir) const
{
    if (atlasDir.empty()) {
        return std::nullopt;
    }

    const auto atlas = vc::atlas::Atlas::load(atlasDir, _state->objectStoreSession());
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
                       fiber.uuid,
                       ex.what());
        return;
    }

    if (!atlasFiberPath) {
        Logger()->warn("Could not find atlas mapping for fiber {}", fiber.uuid);
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
    return ++_nextFiberRuntimeId;
}

uint64_t LineAnnotationController::nextFiberSequenceForUsername(const std::string& username) const
{
    const std::string normalized = vc3d::line_annotation::normalizedFiberUsername(username);
    uint64_t sequence = 1;
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (attachment) {
        // Retired fibers keep their sequence: a label is never reissued.
        connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
            vc::objects::ObjectSelection selection;
            selection.kind = "fiber";
            selection.includeRetired = true;
            for (const auto& object : db.selectObjects(txn, selection, std::vector<std::string>{"meta"})) {
                const auto source = nlohmann::json::parse(object.meta).value("source", nlohmann::json::object());
                if (vc3d::line_annotation::normalizedFiberUsername(source.value("username", std::string{"anon"})) == normalized)
                    sequence = std::max(sequence, source.value("sequence", uint64_t{0}) + 1);
            }
        });
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
    if (!session.storePin && _state) {
        const auto attachment = _state->objectStoreAttachment();
        if (!attachment) throw std::runtime_error("Attach an object store before editing fibers");
        session.storePin = vc::objects::PinnedObject{*attachment, {attachment->storeUuid, session.fiberUuid}, "", {}};
    }
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

    // Linked fibers are drawn between their outer control points only; the
    // extrapolated tails stay hidden like their intersection markers.
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
            std::vector<cv::Vec3d> controls;
            controls.reserve(pane.session->controlPoints.size());
            for (const auto& control : pane.session->controlPoints) {
                controls.push_back(control.volumePoint);
            }
            return vc3d::line_annotation::linePointsBetweenOuterControlPoints(points, controls);
        }
        const auto fiber = readStoredFiber(fiberId, true);
        if (!fiber) {
            return {};
        }
        return vc3d::line_annotation::linePointsBetweenOuterControlPoints(
            fiber->linePoints,
            vc3d::line_annotation::storedControlPointPositions(fiber->controlPoints));
    };

    for (const auto& branch : session.branches) {
        if (branch.branchFiberId == 0 || !seenFiberIds.insert(branch.branchFiberId).second) {
            continue;
        }
        if (!fiberBoundToVolume(branch.branchFiberId, _state ? _state->currentVolume() : nullptr)) continue;
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
            generatedBranchLinkMarkers(pane.session->branches),
            true,
            generatedSpanAlignmentMetricsForSession(*pane.session));
    }
}

std::vector<LineAnnotationController::SideStripFiberSnapshot>
LineAnnotationController::fiberSnapshotsForSideStripQuery(const std::vector<StoredFiber>& stored) const
{
    // Per-fiber memoization: a snapshot (and its geometry hash) is rebuilt
    // only when the fiber's geometry actually changed - the stored copy by
    // its save generation (plus the package generation, since a package
    // switch brings another fiber list), an open session by its line revision and
    // edit epoch. Unchanged fibers cost two integer compares per query
    // instead of a full polyline deep copy plus per-point hashing. The
    // caches are GUI-thread-only (like all controller state); workers only
    // ever hold the immutable shared snapshots.
    std::map<uint64_t, SideStripFiberSnapshot> byId;

    const auto makePolyline = [](uint64_t id,
                                 uint64_t generation,
                                 const std::vector<cv::Vec3d>& linePoints,
                                 std::vector<cv::Vec3d> controlPoints) {
        // Other fibers take part only between their outer control points:
        // extrapolated tails must not produce intersection markers.
        const std::vector<cv::Vec3d> controlledSpan =
            vc3d::line_annotation::linePointsBetweenOuterControlPoints(linePoints,
                                                                       controlPoints);
        auto polyline = std::make_shared<vc::atlas::FiberPolyline>();
        polyline->id = id;
        polyline->generation = std::max<uint64_t>(uint64_t{1}, generation);
        polyline->controlPoints = std::move(controlPoints);
        polyline->points.reserve(controlledSpan.size());
        for (const auto& point : controlledSpan) {
            polyline->points.push_back(vc::atlas::FiberPoint{point, std::nullopt});
        }
        SideStripFiberSnapshot snapshot;
        snapshot.geometryHash = sideStripFiberGeometryHash(*polyline);
        snapshot.polyline = std::move(polyline);
        return snapshot;
    };

    std::unordered_set<uint64_t> liveStoredIds;
    std::unordered_map<uint64_t, uint64_t> storedGenerations;
    for (const auto& fiber : stored) {
        storedGenerations[fiber.id] = fiber.generation;
        if (fiber.id == 0 || fiber.linePoints.size() < 2) {
            continue;
        }
        liveStoredIds.insert(fiber.id);
        auto cached = _sideStripStoredSnapshotCache.find(fiber.id);
        if (cached == _sideStripStoredSnapshotCache.end() ||
            cached->second.validityA != fiber.generation ||
            cached->second.validityB != _packageGeneration) {
            SideStripSnapshotCacheEntry entry;
            entry.validityA = fiber.generation;
            entry.validityB = _packageGeneration;
            entry.snapshot = makePolyline(
                fiber.id,
                fiber.generation,
                fiber.linePoints,
                vc3d::line_annotation::storedControlPointPositions(
                    fiber.controlPoints));
            cached = _sideStripStoredSnapshotCache
                         .insert_or_assign(fiber.id, std::move(entry))
                         .first;
        }
        byId[fiber.id] = cached->second.snapshot;
    }

    std::unordered_set<uint64_t> liveSessionIds;
    for (const auto& pane : _panes) {
        const auto& session = pane.session;
        if (!session || session->fiberId == 0 || session->optimizedLine.points.size() < 2) {
            continue;
        }
        liveSessionIds.insert(session->fiberId);
        auto cached = _sideStripSessionSnapshotCache.find(session->fiberId);
        if (cached == _sideStripSessionSnapshotCache.end() ||
            cached->second.validityA != session->lineRevision ||
            cached->second.validityB != session->solveQueue.epoch()) {
            uint64_t generation = 1;
            if (const auto it = storedGenerations.find(session->fiberId); it != storedGenerations.end()) {
                generation = std::max<uint64_t>(uint64_t{1}, it->second + 1);
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
            SideStripSnapshotCacheEntry entry;
            entry.validityA = session->lineRevision;
            entry.validityB = session->solveQueue.epoch();
            entry.snapshot = makePolyline(session->fiberId,
                                          generation,
                                          linePoints,
                                          std::move(controlPoints));
            cached = _sideStripSessionSnapshotCache
                         .insert_or_assign(session->fiberId, std::move(entry))
                         .first;
        }
        byId[session->fiberId] = cached->second.snapshot;
    }

    // Sweep entries whose fibers/sessions vanished so the caches cannot grow
    // unbounded across loads.
    std::erase_if(_sideStripStoredSnapshotCache, [&](const auto& item) {
        return liveStoredIds.find(item.first) == liveStoredIds.end();
    });
    std::erase_if(_sideStripSessionSnapshotCache, [&](const auto& item) {
        return liveSessionIds.find(item.first) == liveSessionIds.end();
    });

    std::vector<SideStripFiberSnapshot> snapshots;
    snapshots.reserve(byId.size());
    for (auto& [id, snapshot] : byId) {
        (void)id;
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

void LineAnnotationController::handleGeneratedSideStripIntersectionQuery(
    const std::string& surfaceName)
{
    if (!_pendingSideStripQuerySurfaces.insert(surfaceName).second) {
        return;
    }
    QTimer::singleShot(kSideStripQueryDebounceMs, this, [this, surfaceName]() {
        _pendingSideStripQuerySurfaces.erase(surfaceName);
        dispatchSideStripIntersectionQuery(surfaceName);
    });
}

QString LineAnnotationController::sideStripQueryFingerprint(
    const std::string& surfaceName,
    const cv::Mat_<cv::Vec3f>* stripPoints) const
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << surfaceName << '|'
        << reinterpret_cast<std::uintptr_t>(stripPoints) << '|'
        << (stripPoints ? stripPoints->rows : 0) << 'x'
        << (stripPoints ? stripPoints->cols : 0) << '|'
        << _fiberDataGeneration;
    for (const auto& pane : _panes) {
        if (!pane.session) {
            continue;
        }
        const auto& session = *pane.session;
        uint64_t branchDigest = 0x4252414e43484653ULL;
        for (const auto& branch : session.branches) {
            combineSideStripHash(branchDigest,
                                 static_cast<uint64_t>(branch.controlPointIndex));
            combineSideStripHash(branchDigest, branch.branchFiberId);
            combineSideStripHash(branchDigest,
                                 static_cast<uint64_t>(branch.branchControlPointIndex));
            combineSideStripHash(branchDigest, branch.pending ? 1u : 0u);
            combineSideStripPointHash(branchDigest, cv::Vec3f(branch.controlPointPosition));
            combineSideStripPointHash(branchDigest, cv::Vec3f(branch.branchControlPointPosition));
            combineSideStripPointHash(branchDigest, cv::Vec3f(branch.branchControlPointDirection));
        }
        out << '|' << session.fiberId << ':'
            << session.solveQueue.epoch() << ':'
            << session.lineRevision << ':'
            << session.optimizedLine.points.size() << ':'
            << reinterpret_cast<std::uintptr_t>(session.optimizedLine.points.data())
            << ':' << branchDigest;
    }
    return QString::fromStdString(out.str());
}

void LineAnnotationController::invalidateSideStripQueries()
{
    // Requests stamped before this watermark are invalid: the running query
    // cancels cooperatively and its result is dropped. This is the ONLY
    // cancellation in the side-strip scheduler; a merely superseded query
    // runs to completion and publishes (render-job model). A surviving
    // pending request for an unrelated surface is protected by the fresh
    // re-stamp it receives when it starts.
    _sideStripInvalidationWatermark = ++_nextSideStripIntersectionToken;
    _latestSideStripIntersectionToken = _sideStripInvalidationWatermark;
    if (_sideStripInvalidationWatermarkAtomic) {
        _sideStripInvalidationWatermarkAtomic->store(
            _sideStripInvalidationWatermark, std::memory_order_relaxed);
    }
}

void LineAnnotationController::refreshLatestSideStripIntersectionToken()
{
    uint64_t latest = _sideStripIntersectionRunning
        ? _runningSideStripIntersectionToken
        : uint64_t{0};
    for (const auto& [surface, request] : _pendingSideStripIntersectionRequests) {
        (void)surface;
        latest = std::max(latest, request.token);
    }
    if (latest != 0) {
        _latestSideStripIntersectionToken = latest;
    }
}

void LineAnnotationController::dispatchSideStripIntersectionQuery(
    const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session || !pane->dialog) {
        return;
    }

    auto stripSurface = _state
        ? std::dynamic_pointer_cast<QuadSurface>(_state->surface(surfaceName))
        : nullptr;
    // An intentional clear must also invalidate: a query still running (or
    // queued) would otherwise pass the latest-token check when it finishes
    // and repaint the very markers the clear removed.
    const auto invalidateInFlightQueries = [this, &surfaceName]() {
        // Scoped to THIS surface: bump the watermark only when the running
        // query would deliver into the pane being cleared — an unrelated
        // pane's active query keeps running (its pending peers get
        // re-stamped above the watermark when they start).
        if (_sideStripIntersectionRunning &&
            _runningSideStripIntersectionSurfaceName == surfaceName) {
            invalidateSideStripQueries();
        }
        _pendingSideStripIntersectionRequests.erase(surfaceName);
        // The reuse entry must die with the clear: a later failed query
        // would otherwise restore markers this clear intentionally removed.
        _sideStripReuseCache.erase(surfaceName);
        refreshLatestSideStripIntersectionToken();
    };

    const auto* stripPointsPtr = stripSurface ? stripSurface->rawPointsPtr() : nullptr;
    if (!stripPointsPtr || stripPointsPtr->rows < 2 || stripPointsPtr->cols < 2) {
        invalidateInFlightQueries();
        pane->dialog->setGeneratedSideStripIntersectionResult(0);
        pane->dialog->setGeneratedFiberIntersectionMarkers({});
        return;
    }

    // Cheap pre-snapshot staleness gate: the precise cache checks below need
    // the request's hash, and the hash needs a deep copy of every loaded
    // fiber's polyline - the single most expensive step of this handler. Skip
    // both when nothing the query reads can have changed.
    const QString fingerprint =
        sideStripQueryFingerprint(surfaceName, stripPointsPtr);
    if (_sideStripIntersectionRunning &&
        // Not invalidated: an invalidated worker may still be draining, and
        // its fingerprint must not swallow a fresh dispatch. (The latest
        // intent is per surface now — another pane queueing does not make
        // this pane's identical running query stale.)
        _runningSideStripIntersectionToken > _sideStripInvalidationWatermark &&
        _runningSideStripIntersectionSurfaceName == surfaceName &&
        !_runningSideStripFingerprint.isEmpty() &&
        fingerprint == _runningSideStripFingerprint) {
        // The running query IS this surface's latest intent again; a queued
        // older intent (A -> B -> A) would otherwise still run and display
        // last.
        _pendingSideStripIntersectionRequests.erase(surfaceName);
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("already running"), 0, 0);
        return;
    }
    if (const auto pendingIt =
            _pendingSideStripIntersectionRequests.find(surfaceName);
        pendingIt != _pendingSideStripIntersectionRequests.end() &&
        !pendingIt->second.fingerprint.isEmpty() &&
        fingerprint == pendingIt->second.fingerprint) {
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("already queued"), 0, 0);
        return;
    }
    if (const auto reuseIt = _sideStripReuseCache.find(surfaceName);
        reuseIt != _sideStripReuseCache.end() &&
        reuseIt->second.cacheKey != 0 &&
        !reuseIt->second.fingerprint.isEmpty() &&
        fingerprint == reuseIt->second.fingerprint) {
        // The cache satisfies this surface's LATEST intent: a queued older
        // request would otherwise still run, publish obsolete work, and only
        // then be caught up (visible A -> B -> pending -> A oscillation).
        _pendingSideStripIntersectionRequests.erase(surfaceName);
        refreshLatestSideStripIntersectionToken();
        const bool runningHere = _sideStripIntersectionRunning &&
            _runningSideStripIntersectionSurfaceName == surfaceName;
        pane->dialog->setGeneratedSideStripIntersectionBusy(runningHere);
        pane->dialog->setGeneratedFiberIntersectionMarkers(
            markLinkCandidateFiberIntersections(reuseIt->second.markers,
                                                pane->session.get()));
        if (!runningHere) {
            // While this surface's own query is still in flight, the busy
            // indicator tells the truth; a completed-count would show 100%
            // mid-run and progress cannot go backwards.
            pane->dialog->setGeneratedSideStripIntersectionResult(
                reuseIt->second.markers.size(),reuseIt->second.unboundFibers);
        }
        return;
    }

    SideStripIntersectionRequest request;
    request.fingerprint = fingerprint;
    request.suppressErrorDialogs =
        pane->session->suppressErrorDialogs || _errorDialogsSuppressed;
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
    // Select conservative candidates through the store's spatial regions. All
    // acceptance, segment/triangle tests and ordering remain in the shared core.
    try {
    const auto connection=_state->objectStoreSession();
    const auto attachment=_state->objectStoreAttachment();
    const auto volume=_state->currentVolume();
    if(!attachment || !volume) return;
    // Strip points and candidates are in the volume's voxels.
    std::array<double,6> stripBounds{INFINITY,INFINITY,INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for(int row=0;row<request.stripPoints.rows;++row) for(int col=0;col<request.stripPoints.cols;++col) {
        const auto point=request.stripPoints(row,col);
        if(!finitePoint(cv::Vec3d{point[0],point[1],point[2]})) continue;
        for(int axis=0;axis<3;++axis) {
            stripBounds[axis]=std::min(stripBounds[axis],double(point[axis])-1.0e-6);
            stripBounds[axis+3]=std::max(stripBounds[axis+3],double(point[axis])+1.0e-6);
        }
    }
    std::vector<std::string> selected;
    connection->read(*attachment,[&](const auto& db,const auto& txn,const auto&) {
        vc::objects::ObjectSelection selection;selection.kind="fiber";selection.volume=volume->id();
        selection.bbox=stripBounds;
        for(const auto& region:db.spatialRegions(txn,selection)) selected.push_back(region.objectUuid);
        request.unboundFibers=static_cast<size_t>(db.unboundObjectCount(txn,"fiber",volume->id()));
    });
    std::sort(selected.begin(),selected.end());
    selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
    const auto stored=readStoredFibers(selected,true);
    std::unordered_set<uint64_t> inVolume;
    for(const auto& fiber:stored) if(fiberInVolume(fiber)) inVolume.insert(fiber.id);
    for(const auto& open:_panes) if(open.session && open.session->fiberId!=0) inVolume.insert(open.session->fiberId);
    request.fibers = fiberSnapshotsForSideStripQuery(stored);
    std::erase_if(request.fibers,[&](const auto& candidate) { return !inVolume.contains(candidate.polyline->id); });

    } catch(const std::exception& error) {
        showError(QString::fromUtf8(error.what()),request.suppressErrorDialogs);
        return;
    }
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
        if (!fiberBoundToVolume(branch.branchFiberId, _state->currentVolume())) continue;
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
        invalidateInFlightQueries();
        pane->dialog->setGeneratedSideStripIntersectionResult(0);
        pane->dialog->setGeneratedFiberIntersectionMarkers({});
        return;
    }

    if (_sideStripIntersectionRunning &&
        _runningSideStripIntersectionToken > _sideStripInvalidationWatermark &&
        request.cacheKey == _runningSideStripIntersectionKey &&
        request.surfaceName == _runningSideStripIntersectionSurfaceName) {
        _pendingSideStripIntersectionRequests.erase(request.surfaceName);
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("already running"), 0, 0);
        return;
    }

    if (const auto pendingIt =
            _pendingSideStripIntersectionRequests.find(request.surfaceName);
        _sideStripIntersectionRunning &&
        pendingIt != _pendingSideStripIntersectionRequests.end() &&
        pendingIt->second.cacheKey == request.cacheKey) {
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("already queued"), 0, 0);
        return;
    }

    if (const auto reuseIt = _sideStripReuseCache.find(request.surfaceName);
        (!_sideStripIntersectionRunning ||
         _runningSideStripIntersectionSurfaceName != request.surfaceName) &&
        reuseIt != _sideStripReuseCache.end() &&
        reuseIt->second.cacheKey != 0 &&
        request.cacheKey == reuseIt->second.cacheKey) {
        // Another surface's running query must not force this surface into a
        // redundant full solve when its exact result is cached; retire any
        // queued older intent along the way.
        _pendingSideStripIntersectionRequests.erase(request.surfaceName);
        refreshLatestSideStripIntersectionToken();
        pane->dialog->setGeneratedSideStripIntersectionBusy(false);
        pane->dialog->setGeneratedFiberIntersectionMarkers(
            markLinkCandidateFiberIntersections(reuseIt->second.markers,
                                                pane->session.get()));
        pane->dialog->setGeneratedSideStripIntersectionResult(
            reuseIt->second.markers.size(),reuseIt->second.unboundFibers);
        return;
    }

    request.token = ++_nextSideStripIntersectionToken;
    _latestSideStripIntersectionToken = request.token;

    pane->dialog->setGeneratedSideStripIntersectionBusy(true);
    pane->dialog->setGeneratedSideStripIntersectionProgress(tr("queued"), 0, 0);
    pane->dialog->setGeneratedFiberIntersectionMarkers({});

    if (_sideStripIntersectionRunning) {
        const std::string pendingSurface = request.surfaceName;
        _pendingSideStripIntersectionRequests.insert_or_assign(
            pendingSurface, std::move(request));
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
    result.unboundFibers = request.unboundFibers;
    result.suppressErrorDialogs = request.suppressErrorDialogs;
    result.token = request.token;
    result.cacheKey = request.cacheKey;
    result.fingerprint = request.fingerprint;
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
            if (!fiber.polyline ||
                excludedFiberIds.find(fiber.polyline->id) != excludedFiberIds.end() ||
                fiber.polyline->points.size() < 2) {
                continue;
            }
            segmentBudget += fiber.polyline->points.size() - 1;
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
            vc::atlas::FiberSideStripQueryOptions branchOptions;
            branchOptions.stripPoints = request.stripPoints;
            branchOptions.deduplicateStripDistance = 1.0e-3;
            branchOptions.aabbPadding = 1.0e-6;
            branchOptions.maxResults = 0;
            branchOptions.workerThreads = workerThreadBudget;
            branchOptions.branchLinks = request.branchLinks;

            const auto branchIntersections =
                vc::atlas::fiberSideStripIntersections(branchOptions,
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
                result.publishedPartial = true;
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
                if (fiber.polyline) {
                    options.queryFibers.push_back(fiber.polyline.get());
                }
            }
            publishAggregateProgress("fiber geometry", triangleIndexOffset);

            const auto intersections =
                vc::atlas::fiberSideStripIntersections(options,
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
    _runningSideStripFingerprint = request.fingerprint;
    if (auto* pane = paneForSurface(request.surfaceName); pane && pane->dialog) {
        pane->dialog->setGeneratedSideStripIntersectionBusy(true);
        pane->dialog->setGeneratedSideStripIntersectionProgress(tr("starting"), 0, 0);
    }
    auto requestPtr = std::make_shared<SideStripIntersectionRequest>(std::move(request));
    const auto invalidationWatermark = _sideStripInvalidationWatermarkAtomic;
    QPointer<LineAnnotationController> self(this);
    QThreadPool::globalInstance()->start(
        [self, requestPtr, invalidationWatermark]() {
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
                    // Queued on the application object: the GUI thread can
                    // destroy the controller between the QPointer check above
                    // and invokeMethod consuming a raw receiver. The lambda
                    // re-checks `self` on the GUI thread.
                    QMetaObject::invokeMethod(
                        QCoreApplication::instance(),
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
                    // Same application-object queuing as the progress path.
                    QMetaObject::invokeMethod(
                        QCoreApplication::instance(),
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
            // Cancellation means INVALIDATION only (pane closed, package
            // switched, intentional clear): a query merely superseded by a
            // newer request runs to completion and publishes -- its result
            // is still fresher than what is displayed (render-job model).
            auto cancelCallback = [invalidationWatermark, token = requestPtr->token]() {
                return invalidationWatermark &&
                       invalidationWatermark->load(std::memory_order_relaxed) > token;
            };
            auto resultPtr = std::make_shared<SideStripIntersectionTaskResult>(
                LineAnnotationController::runSideStripIntersectionQuery(*requestPtr,
                                                                        progressCallback,
                                                                        partialCallback,
                                                                        cancelCallback));
            if (!self) {
                return;
            }
            // Same application-object queuing as the progress path.
            QMetaObject::invokeMethod(
                QCoreApplication::instance(),
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
    if (token < _sideStripInvalidationWatermark) {
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
    if (token < _sideStripInvalidationWatermark) {
        return;
    }
    if (auto* pane = paneForSurface(surfaceName); pane && pane->dialog) {
        pane->dialog->setGeneratedFiberIntersectionMarkers(
            markLinkCandidateFiberIntersections(
                std::move(markers),
                pane->session.get()));
    }
}

void LineAnnotationController::finishSideStripIntersectionQuery(
    SideStripIntersectionTaskResult result)
{
    _sideStripIntersectionRunning = false;
    _runningSideStripIntersectionToken = 0;
    _runningSideStripIntersectionKey = 0;
    _runningSideStripIntersectionSurfaceName.clear();
    _runningSideStripFingerprint.clear();
    // Render-job model: a completed query PUBLISHES even when a newer request
    // is already pending -- its markers are still fresher than what is on
    // screen -- and only an invalidated (pane closed / package switched /
    // intentionally cleared) result is dropped. The monotonic published-token
    // guard is belt-and-braces: one query runs at a time, so results finish
    // in start order.
    const bool invalidated = result.token < _sideStripInvalidationWatermark ||
                             result.token <= _lastPublishedSideStripToken;
    QString deferredError;
    // Busy is per surface: the completed pane stays busy only when the
    // pending request targets the SAME surface; a pending query for another
    // pane sets its own busy state when it starts below.
    const bool samePanePending =
        _pendingSideStripIntersectionRequests.count(result.surfaceName) > 0;
    if (!invalidated) {
        _lastPublishedSideStripToken = result.token;
        if (auto* pane = paneForSurface(result.surfaceName); pane && pane->dialog) {
            pane->dialog->setGeneratedSideStripIntersectionBusy(samePanePending);
            if (result.ok) {
                const size_t markerCount = result.markers.size();
                _sideStripReuseCache.insert_or_assign(
                    result.surfaceName,
                    SideStripReuseEntry{result.unboundFibers,result.cacheKey, result.markers,
                                        result.fingerprint});
                pane->dialog->setGeneratedFiberIntersectionMarkers(
                    markLinkCandidateFiberIntersections(
                        std::move(result.markers),
                        pane->session.get()));
                if (!samePanePending) {
                    pane->dialog->setGeneratedSideStripIntersectionResult(markerCount,result.unboundFibers);
                }
            } else {
                // A failed refresh must not blank a valid display — and it
                // must not REPAINT either, unless the failed run itself
                // corrupted the display with its partial branch-link markers:
                // the reuse entry is by definition from an older fingerprint,
                // so restoring it unconditionally would render stale
                // intersections over newer strip geometry. The error state
                // still shows either way.
                if (const auto reuseIt =
                        _sideStripReuseCache.find(result.surfaceName);
                    result.publishedPartial &&
                    reuseIt != _sideStripReuseCache.end() &&
                    reuseIt->second.cacheKey != 0) {
                    pane->dialog->setGeneratedFiberIntersectionMarkers(
                        markLinkCandidateFiberIntersections(
                            reuseIt->second.markers,
                            pane->session.get()));
                }
                if (!samePanePending) {
                    pane->dialog->setGeneratedSideStripIntersectionError();
                }
                // Shown AFTER the pending start below: showError spins a
                // nested event loop, and a debounced dispatch firing inside
                // it would see running == false and start a second worker
                // beside the one the epilogue is about to launch.
                deferredError = tr("Could not query strip fiber intersections: %1")
                                    .arg(QString::fromStdString(result.error));
            }
        }
    } else if (!samePanePending) {
        if (auto* pane = paneForSurface(result.surfaceName); pane && pane->dialog) {
            pane->dialog->setGeneratedSideStripIntersectionBusy(false);
        }
    }

    if (!_pendingSideStripIntersectionRequests.empty()) {
        // FIFO across surfaces: start the oldest queued request so a busy
        // pane's steady refreshes cannot starve another pane's single query.
        auto oldest = _pendingSideStripIntersectionRequests.begin();
        for (auto it = _pendingSideStripIntersectionRequests.begin();
             it != _pendingSideStripIntersectionRequests.end(); ++it) {
            if (it->second.token < oldest->second.token) {
                oldest = it;
            }
        }
        SideStripIntersectionRequest pending = std::move(oldest->second);
        _pendingSideStripIntersectionRequests.erase(oldest);
        // Re-stamp: the pending request may predate an invalidation aimed at
        // a DIFFERENT pane (the watermark is global); a fresh token keeps a
        // legitimately surviving request valid. Same-surface pending was
        // dropped explicitly at the invalidation site.
        pending.token = ++_nextSideStripIntersectionToken;
        _latestSideStripIntersectionToken = pending.token;
        startSideStripIntersectionQuery(std::move(pending));
    }
    if (!invalidated && result.ok && !result.surfaceName.empty()) {
        // The render scheduler's post-adoption catch-up: the displayed state
        // may still lag the latest inputs (A -> B -> A orderings dedupe the
        // third request against the active first). Unconditional on every
        // successful publish — starting a queued OTHER surface above must
        // not suppress this surface's convergence. The debounced re-dispatch
        // is a fingerprint/reuse-gated no-op whenever the published result
        // already matches. Failed or invalidated results deliberately do NOT
        // catch up: a persistent failure would otherwise re-run (and re-show
        // its error) forever.
        handleGeneratedSideStripIntersectionQuery(result.surfaceName);
    }
    if (!deferredError.isEmpty()) {
        showError(deferredError, result.suppressErrorDialogs);
    }
}

LineAnnotationController::BranchMetadataSyncResult
LineAnnotationController::syncLinkedBranchMetadataAfterFiberModification(
    LineAnnotationSession& session,
    const std::vector<vc3d::line_annotation::LineControlPoint>* previousControlPoints,
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
                                       vc3d::line_annotation::sameFiberIdentity(candidate.branchFiberId, session.fiberId);
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
                    !branchReferencesFiber(oldBranch, pane.session->fiberId)) {
                    continue;
                }
                if (removeReciprocal(pane.session->branches, oldBranch)) {
                    addUniqueFiberId(result.affectedFiberIds, pane.session->fiberId);
                }
            }
        }
    }

    syncReciprocalBranchControlPointReferences(session);
    for (const uint64_t fiberId : syncBranchEndpointPositions(session)) {
        addUniqueFiberId(result.affectedFiberIds, fiberId);
    }
    for (const auto& pane : _panes) {
        if (pane.session && pane.session.get() != &session &&
            deduplicateBranchLinks(pane.session->branches,
                                   session.fiberId,
                                   session.fiberFileName)) {
            addUniqueFiberId(result.affectedFiberIds, pane.session->fiberId);
        }
    }
    std::sort(result.affectedFiberIds.begin(), result.affectedFiberIds.end());
    return result;
}

void LineAnnotationController::scheduleBranchMetadataSaves(
    const std::vector<uint64_t>& fiberIds, uint64_t primaryFiberId)
{
    if (fiberIds.empty()) return;
    std::set<uint64_t> participants(fiberIds.begin(),fiberIds.end());
    if (primaryFiberId) participants.insert(primaryFiberId);
    std::vector<FiberSaveSnapshot> snapshots;
    for (const auto id : participants) {
        const auto pane=std::find_if(_panes.begin(),_panes.end(),[&](const auto& p) {
            return p.session && p.session->fiberId==id && !p.session->suppressFiberSave &&
                   !p.session->controlPoints.empty();
        });
        if (pane!=_panes.end()) snapshots.push_back(makeFiberSaveSnapshot(makeStoredFiberSessionSnapshot(*pane->session).fiber));
    }
    // Open sessions of linked fibers publish with the edit. A stored fiber
    // needs no write: each link is one shared edge, saved with either side.
    if (!snapshots.empty()) {
        commitFiberStoreAction(snapshots);
        applyCommittedFibers(snapshots);
    }
}

void LineAnnotationController::syncReciprocalBranchControlPointReferences(
    const LineAnnotationSession& session)
{
    if (session.fiberId == 0) {
        return;
    }

    // A target's branch is updated only when it points at this session AND
    // the session's branch it is mirrored from points back at the target's
    // owner: a stale reciprocal on some other fiber must not be rewritten
    // from a branch that links elsewhere.
    auto updateBranches = [&session](std::vector<FiberBranchRef>& targetBranches,
                                     uint64_t ownerFiberId,
                                     const std::string& ownerFileName) {
        for (auto& targetBranch : targetBranches) {
            if (!vc3d::line_annotation::sameFiberIdentity(targetBranch.branchFiberId, session.fiberId)) {
                continue;
            }
            for (const auto& sourceBranch : session.branches) {
                if (sourceBranch.branchFiberId != 0 &&
                    vc3d::line_annotation::sameFiberIdentity(sourceBranch.branchFiberId, ownerFiberId) &&
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
            updateBranches(pane.session->branches, pane.session->fiberId,
                           pane.session->fiberFileName);
        }
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
    if (session.suppressErrorDialogs || _errorDialogsSuppressed) {
        // Without confirmation UI, conservatively refuse the linked edit.
        showError(
            tr("%1 was rejected: the control point is linked to another "
               "fiber and confirmation prompts are disabled in headless mode.")
                .arg(action),
            true);
        return false;
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

bool LineAnnotationController::confirmLinkedControlPointEdits(
    const LineAnnotationSession& session,
    const std::vector<size_t>& controlPointIndices,
    const QString& action) const
{
    const size_t linkedCount = static_cast<size_t>(std::count_if(
        controlPointIndices.begin(),
        controlPointIndices.end(),
        [this, &session](size_t index) {
            return index <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   controlPointHasBranch(session, static_cast<int>(index));
        }));
    if (linkedCount == 0) {
        return true;
    }
    if (session.suppressErrorDialogs || _errorDialogsSuppressed) {
        showError(
            tr("%1 was rejected: %2 affected control point(s) are linked to "
               "other fibers and confirmation prompts are disabled in headless mode.")
                .arg(action)
                .arg(linkedCount),
            true);
        return false;
    }
    const auto response = QMessageBox::question(
        _parentWidget.data(),
        tr("Linked control points"),
        tr("%1 will affect %2 control point(s) linked to other fibers and "
           "update branch metadata on both sides. Continue?")
            .arg(action)
            .arg(linkedCount),
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
            const bool pointsToSession = vc3d::line_annotation::sameFiberIdentity(reciprocal.branchFiberId, session.fiberId);
            const bool sameLocalEndpoint =
                reciprocal.controlPointIndex == targetControlPointIndex ||
                pointsApproximatelyEqual(reciprocal.controlPointPosition, targetPoint);
            const bool sameLinkedEndpoint =
                reciprocal.branchControlPointIndex == sourceBranch.controlPointIndex ||
                pointsApproximatelyEqual(reciprocal.branchControlPointPosition, sourcePoint);
            if (reciprocal.adjacent != sourceBranch.adjacent ||
                !pointsToSession || !sameLocalEndpoint || !sameLinkedEndpoint) {
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
            if (!vc3d::line_annotation::sameFiberIdentity(pane.session->fiberId, branch.branchFiberId)) {
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
        // A stored target only refreshes this side's view of the shared edge.
        const auto fiber = readStoredFiber(branch.branchFiberId, true);
        if (!fiber) {
            continue;
        }
        if (auto targetIndex = matchingStoredControlPointIndex(
                fiber->controlPoints,
                branch.branchControlPointIndex,
                branch.branchControlPointPosition)) {
            const cv::Vec3d targetPoint =
                fiber->controlPoints[static_cast<size_t>(*targetIndex)];
            branch.branchControlPointIndex = *targetIndex;
            branch.branchControlPointPosition = targetPoint;
            branch.branchControlPointDirection =
                endpointTangentFromLinePoints(fiber->linePoints,
                                              targetPoint,
                                              branch.branchControlPointDirection);
        }
    }
    return affectedFiberIds;
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
                accumulator.add(vc::lasagna::normalAlignmentErrorDegrees(
                    tangent, sample.normal));
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

bool LineAnnotationController::prepareSessionLineModel(LineAnnotationSession& session,
                                                       const std::vector<cv::Vec3d>& linePoints,
                                                       const std::function<bool()>& stillValid)
{
    // A pane-less session starts on a synthetic line model (normals all
    // +z); the real model - sampled normals, or the fiber-follow model when
    // the fiber is model-only - is what display-normal inheritance and the
    // solve read. The dataset (normalSampler) must already be resolved.
    if (session.modelOnly) {
        if (!ensureFiberFollowService(session.suppressErrorDialogs, nullptr, stillValid)) {
            return false;
        }
        if (stillValid && !stillValid()) {
            return false;
        }
        session.optimizedLine = FiberFollowService::instance().lineModel(
            linePoints, _state->currentVolume()->voxelSize());
    } else {
        session.optimizedLine = lineModelFromPoints(linePoints, session.normalSampler.get());
    }
    return true;
}

std::shared_ptr<LineAnnotationController::LineAnnotationSession>
LineAnnotationController::makeIntersectionLineSession(
    const StoredFiber& fiber,
    double focusLinePosition,
    const cv::Vec3d& sourceSliceNormal,
    const std::string& surfaceName,
    std::function<void()> onOptimizationSucceeded)
{
    ensureFiberLines(fiber);
    auto session = std::make_shared<LineAnnotationSession>();
    session->suppressErrorDialogs =
        (_intersectionInspection && _intersectionInspection->suppressErrorDialogs) ||
        _errorDialogsSuppressed;
    session->surfaceName = surfaceName;
    session->sourceAnnotationSurfaceName = surfaceName;
    session->fiberId = fiber.id;
    session->fiberUuid = fiber.uuid;
    session->storePin = fiber.storePin;
    session->readOnly = fiber.readOnly;
    session->suppressFiberSave = fiber.readOnly;
    session->fiberUsername = fiber.username;
    session->fiberStartedAt = fiber.startedAt;
    session->fiberSequence = fiber.sequence;
    session->fiberFileName = fiber.fileName;
    session->fiberManualHvTag = fiber.manualHvTag;
    session->fiberTags = fiber.tags;
    session->fiberWidth = fiber.width;
    session->fiberWidthGapFraction = fiber.widthGapFraction;
    session->fiberOptimizationMode = fiber.optimizationMode;
    session->modelOnly = fiber.modelOnly;
    if (_state->currentVolume() && !fiberInVolume(fiber)) throw std::runtime_error("Fiber coordinate binding is missing");
    // Carried through to the saved fiber; without it a re-save from this
    // pane-less session (re-optimization, intersection edits) drops the
    // fiber's coordinate_base_shape_zyx.
    session->coordinateBaseShapeZYX = fiber.coordinateBaseShapeZYX;
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
    ++session->lineRevision;
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
        vc3d::line_annotation::LineControlPoint control;
        control.linePosition = static_cast<double>(index);
        control.volumePoint = controlPoint;
        control.optimizedIndex = index;
        control.segmentToNext = fiber.controlPoints[i].segmentToNext;
        control.tags = fiber.controlPoints[i].tags;
        control.displayNormal = fiber.controlPoints[i].displayNormal;
        control.direction = fiber.controlPoints[i].direction;
        control.storeId = fiber.controlPoints[i].storeId;
        control.displayNormalSource = fiber.controlPoints[i].displayNormalSource;
        session->controlPoints.push_back(control);
        const double distance = std::abs(control.linePosition - session->focusedLinePosition);
        if (distance < seedDistance) {
            seedDistance = distance;
            seedControl = static_cast<int>(i);
        }
    }
    if (session->controlPoints.empty()) {
        vc3d::line_annotation::LineControlPoint control;
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
            !branchReferencesFiber(branch, pane.session->fiberId)) {
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

    if (const auto fiber = readStoredFiber(branch.branchFiberId)) {
        return matchingStoredControlPointIndex(fiber->controlPoints,
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
    // The store assigns the generation and keeps the coordinate frame of an
    // existing object when this snapshot is committed.
    fiber.id = session.fiberId == 0 ? nextFiberId() : session.fiberId;
    fiber.uuid = session.fiberUuid;
    registerFiberId(fiber.uuid, fiber.id);
    fiber.storePin = session.storePin;
    fiber.coordinateBaseShapeZYX = session.coordinateBaseShapeZYX;
    if (!fiber.coordinateBaseShapeZYX && (!session.storePin || session.storePin->objectRevision.empty())) {
        const auto volume = _state ? _state->currentVolume() : nullptr;
        if (!volume)
            throw std::runtime_error("Cannot save a fiber without a coordinate base shape or active volume");
        fiber.coordinateBaseShapeZYX =
            vc3d::line_annotation::fiberCoordinateShapeFromVolume(volume->shape());
        session.coordinateBaseShapeZYX = fiber.coordinateBaseShapeZYX;
    }

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
        vc3d::line_annotation::StoredControlPoint stored{
            session.controlPoints[sessionIndex].volumePoint};
        stored.segmentToNext = session.controlPoints[sessionIndex].segmentToNext;
        stored.tags = session.controlPoints[sessionIndex].tags;
        stored.displayNormal = session.controlPoints[sessionIndex].displayNormal;
        stored.direction = session.controlPoints[sessionIndex].direction;
        stored.storeId = session.controlPoints[sessionIndex].storeId;
        stored.displayNormalSource = session.controlPoints[sessionIndex].displayNormalSource;
        fiber.controlPoints.push_back(std::move(stored));
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

    vc3d::line_annotation::validateStoredControlPoints(fiber.controlPoints);
    fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(
        vc3d::line_annotation::storedControlPointPositions(fiber.controlPoints));
    fiber.manualHvTag = session.fiberManualHvTag;
    fiber.tags = session.fiberTags;
    fiber.width = session.fiberWidth;
    fiber.widthGapFraction = session.fiberWidthGapFraction;
    fiber.optimizationMode = session.fiberOptimizationMode;
    fiber.modelOnly = session.modelOnly;
    return snapshot;
}

LineAnnotationController::StoredFiber
LineAnnotationController::storedFiberFromSession(LineAnnotationSession& session)
{
    ensureSessionFiberIdentity(session);
    syncLinkedBranchMetadataAfterFiberModification(session);
    return makeStoredFiberSessionSnapshot(session).fiber;
}

void LineAnnotationController::setSessionFiberWidth(LineAnnotationSession& session, double width)
{
    session.fiberWidth = width;
    for (auto& other : _panes) {
        if (other.session && (other.session.get() == &session ||
            (session.fiberId && other.session->fiberId == session.fiberId))) {
            other.session->fiberWidth = width;
            other.session->fiberWidthGapFraction = session.fiberWidthGapFraction;
            if (other.dialog) other.dialog->setFiberWidth(width, session.fiberWidthGapFraction);
        }
    }
}

void LineAnnotationController::saveFiberDisplayAnnotations(LineAnnotationSession& session,
                                                           const std::function<bool()>& stillValid)
{
    const QPointer<LineAnnotationController> alive(this);
    // Save metadata without invoking a solve or replacing stored geometry.
    // Newly moved/inserted CPs are persisted with the next geometry save.
    auto stored = session.suppressFiberSave ? std::nullopt : readStoredFiber(session.fiberId, true);
    if (stored) {
        StoredFiber updated = std::move(*stored);
        updated.width = session.fiberWidth;
        updated.widthGapFraction = session.fiberWidthGapFraction;
        for (auto& stored : updated.controlPoints) {
            const auto cp = std::find_if(session.controlPoints.begin(), session.controlPoints.end(),
                [&](const auto& point) { return pointsApproximatelyEqual(point.volumePoint, stored); });
            if (cp != session.controlPoints.end()) {
                stored.displayNormal = cp->displayNormal;
                stored.direction = cp->direction;
                stored.displayNormalSource = cp->displayNormalSource;
            }
        }
        try {
            commitFibers({&updated});
        } catch (const std::exception& ex) {
            showError(tr("Could not save fiber display annotations: %1").arg(QString::fromUtf8(ex.what())), session.suppressErrorDialogs);
            // The modal: nothing scheduled on a controller or session that
            // is no longer valid.
            if (!alive || (stillValid && !stillValid())) {
                return;
            }
        }
    }
    scheduleSessionAutoSave(session);
}

void LineAnnotationController::scheduleSessionAutoSave(LineAnnotationSession& session)
{
    if (session.readOnly || session.autoSaveScheduled) return;
    session.autoSaveScheduled = true;
    flushSessionAutoSave(session.surfaceName);
}

void LineAnnotationController::flushSessionAutoSave(const std::string& surfaceName)
{
    auto* pane = paneForSurface(surfaceName);
    if (!pane || !pane->session) {
        return;
    }
    auto& session = *pane->session;
    if (!session.autoSaveScheduled) {
        return;
    }
    session.autoSaveScheduled = false;
    if (session.suppressFiberSave || session.controlPoints.empty()) return;
    saveSessionAsFiber(session, false);
}

void LineAnnotationController::flushAllPendingSessionAutoSaves(
    const std::function<bool()>& stillValid)
{
    std::vector<std::string> surfaces;
    surfaces.reserve(_panes.size());
    for (const auto& pane : _panes) {
        if (pane.session && pane.session->autoSaveScheduled) {
            surfaces.push_back(pane.session->surfaceName);
        }
    }
    // A flushed save can show a modal (serialization error) and the
    // controller can be destroyed inside it; the next flush must not run on
    // a destroyed object.
    const QPointer<LineAnnotationController> alive(this);
    for (const auto& surfaceName : surfaces) {
        flushSessionAutoSave(surfaceName);
        if (!alive || (stillValid && !stillValid())) {
            return;
        }
    }
}

void LineAnnotationController::saveSessionAsFiber(LineAnnotationSession& session, bool finalize)
{
    // Clear the immediate-publication reentrancy guard.
    session.autoSaveScheduled = false;
    if (session.readOnly) {
        return;
    }
    // A session whose fiber was deleted underneath it is not saved through
    // here - the close with a final optimization included - or it would
    // recreate the deleted file (or overwrite an import under that name).
    // Checked again after the finalize below, which can yield (its dataset
    // picker) and so let a delete land meanwhile. The transient
    // suppressFiberSave (headless save, retirement window) is deliberately
    // not consulted here: such a session remains eligible for this save
    // path (its close-save in particular).
    if (session.fiberDeleted) {
        return;
    }
    try {
        if (finalize && !finalizeSessionOptimizationSynchronously(session, false)) {
            return;
        }
        if (session.fiberDeleted) {
            return;
        }
        if (session.lineWasOptimized && session.optimizationState == SessionOptimizationState::Optimized) {
            // The line being saved was produced by the optimizer in this
            // session, so any sync-applied needs_reoptimization tag is
            // satisfied. Without this, an ordinary pane save would write
            // the tag back and re-trigger the load-time prompt forever.
            session.fiberTags.erase(
                std::remove(session.fiberTags.begin(), session.fiberTags.end(),
                            std::string{kNeedsReoptimizationTag}),
                session.fiberTags.end());
        }
        if (session.stripReviewedTagOnSave) {
            // An interpolation-mode switch overwrote the geometry; any prior
            // human review verdict is stale. Ordinary edits/merges/splits keep it.
            session.fiberTags.erase(
                std::remove(session.fiberTags.begin(), session.fiberTags.end(),
                            std::string{vc3d::line_annotation::kReviewedTag}),
                session.fiberTags.end());
            session.stripReviewedTagOnSave = false;
        }
        ensureSessionFiberIdentity(session);
        const auto candidate = makeStoredFiberSessionSnapshot(session).fiber;
        const auto storedFibers = readStoredFibers(std::vector<std::string>{candidate.uuid}, true);
        if (!storedFibers.empty()) {
            const auto stored = storedFibers.begin();
            const auto sameBranch = [](const auto& a, const auto& b) {
                return a.branchFiberId == b.branchFiberId && a.controlPointIndex == b.controlPointIndex &&
                    a.branchControlPointIndex == b.branchControlPointIndex && a.pending == b.pending && a.adjacent == b.adjacent &&
                    a.autoLink == b.autoLink &&
                    a.controlPointPosition == b.controlPointPosition && a.branchControlPointPosition == b.branchControlPointPosition &&
                    a.controlPointDirection == b.controlPointDirection && a.branchControlPointDirection == b.branchControlPointDirection;
            };
            if (stored->fileName == candidate.fileName && stored->tags == candidate.tags &&
                stored->width == candidate.width && stored->widthGapFraction == candidate.widthGapFraction &&
                stored->optimizationMode == candidate.optimizationMode && stored->manualHvTag == candidate.manualHvTag &&
                stored->controlPoints == candidate.controlPoints && stored->linePoints == candidate.linePoints &&
                std::equal(stored->branches.begin(), stored->branches.end(), candidate.branches.begin(), candidate.branches.end(), sameBranch) &&
                fiberSourceMetadata(*stored) == fiberSourceMetadata(candidate)) return;
        }
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

        // Open sessions of linked fibers publish with this one; a stored
        // linked fiber needs no write (each link is one shared edge).
        std::vector<FiberSaveSnapshot> snapshots;
        snapshots.push_back(makeFiberSaveSnapshot(fiber));
        for (const uint64_t linkedFiberId : branchSync.affectedFiberIds) {
            if (linkedFiberId == savedFiberId) {
                continue;
            }
            for (const auto& pane : _panes) {
                if (!pane.session || pane.session.get() == &session ||
                    pane.session->fiberId != linkedFiberId ||
                    pane.session->fiberDeleted ||
                    pane.session->taskState != LineAnnotationSession::TaskState::Succeeded ||
                    pane.session->optimizedLine.points.empty() ||
                    pane.session->controlPoints.empty()) {
                    continue;
                }
                StoredFiber linkedFiber = storedFiberFromSession(*pane.session);
                snapshots.push_back(makeFiberSaveSnapshot(linkedFiber));
                break;
            }
        }
        commitFiberStoreAction(snapshots);
        applyCommittedFibers(snapshots);
        invalidateFiberAlignmentMetrics(savedFiberId, true);
        for (const auto& pane : _panes) {
            if (pane.session && pane.session->fiberId == savedFiberId) {
                pushFiberUiState(pane);
            }
        }
        refreshBranchLineViews(savedFiberId);
    } catch (const std::exception& ex) {
        // Destructive follow-ups check this count before proceeding.
        ++_fiberSaveFailureCount;
        showError(tr("Could not save fiber: %1").arg(QString::fromStdString(ex.what())),
                  session.suppressErrorDialogs);
    }
}

nlohmann::json LineAnnotationController::fiberSourceMetadata(const StoredFiber& fiber)
{
    nlohmann::json source = {{"username", fiber.username}, {"started_at", fiber.startedAt},
        {"sequence", fiber.sequence}, {"model_only", fiber.modelOnly},
        {"hv_classification", {
            {"z_distance", fiber.hvClassification.zDistance},
            {"control_point_length", fiber.hvClassification.fiberLength},
            {"horizontal_score", fiber.hvClassification.horizontalScore},
            {"vertical_score", fiber.hvClassification.verticalScore},
            {"automatic_tag", vc3d::line_annotation::fiberHvTagToString(fiber.hvClassification.automaticTag)},
            {"automatic_certainty", fiber.hvClassification.automaticCertainty}}}};
    if (fiber.coordinateBaseShapeZYX) source["coordinate_base_shape_zyx"] = *fiber.coordinateBaseShapeZYX;
    auto controls = fiber.controlPoints;
    if (!controls.empty()) controls.back().segmentToNext.reset();
    vc3d::line_annotation::syncGapSpanTags(controls);
    auto& annotations = source["control_points_attributes"] = nlohmann::json::array();
    for (const auto& point : controls) {
        auto annotation = vc3d::line_annotation::storedControlPointToJson(point);
        annotation.erase("position");
        annotations.push_back(std::move(annotation));
    }
    return source;
}

LineAnnotationController::FiberSaveSnapshot
LineAnnotationController::makeFiberSaveSnapshot(const StoredFiber& fiber)
{
    ensureFiberLines(fiber);
    FiberSaveSnapshot snapshot;
    snapshot.fiber = fiber;
    retainStoredFiberCoordinateBaseShape(snapshot.fiber);
    if ((!snapshot.fiber.storePin || snapshot.fiber.storePin->objectRevision.empty()) && snapshot.fiber.coordinateIdentity.empty())
        copyCoordinateIdentityToJson(snapshot.fiber.coordinateIdentity, coordinateIdentityForState(_state));
    if ((!fiber.storePin || fiber.storePin->objectRevision.empty()) && _state && _state->currentVolume()) {
        // A fiber created in an annotation pane is in the volume's voxels.
        for (const auto& pane : _panes) if (pane.session && pane.session->fiberUuid == fiber.uuid) {
            snapshot.fiber.coordinateIdentity.update(coordinateBindingForVolume(*_state->currentVolume(),1.0));
            break;
        }
    }
    if (!_state || !_state->objectStoreAttachment())
        throw std::runtime_error("Attach an object store before editing fibers");
    return snapshot;
}

void LineAnnotationController::commitFibers(std::initializer_list<StoredFiber*> fibers)
{
    std::vector<FiberSaveSnapshot> snapshots;
    for (const auto* fiber : fibers)
        if (std::none_of(snapshots.begin(), snapshots.end(), [&](const auto& saved) { return saved.fiber.uuid == fiber->uuid; }))
            snapshots.push_back(makeFiberSaveSnapshot(*fiber));
    try {
        commitFiberStoreAction(snapshots);
        for (auto* fiber : fibers) for (const auto& saved : snapshots)
            if (fiber->uuid == saved.fiber.uuid) *fiber = saved.fiber;
        applyCommittedFibers(snapshots);
    } catch (...) {
        ++_fiberSaveFailureCount;
        throw;
    }
}

void LineAnnotationController::applyCommittedFibers(const std::vector<FiberSaveSnapshot>& snapshots)
{
    // Holders of derived views see the commit before its store event arrives.
    ++_fiberDataGeneration;
    for (auto& pane : _panes) if (pane.session) {
        auto& session = *pane.session;
        const auto snapshot = std::find_if(snapshots.begin(), snapshots.end(), [&](const auto& candidate) { return candidate.fiber.uuid == session.fiberUuid; });
        if (snapshot == snapshots.end()) continue;
        const StoredFiber* saved = &snapshot->fiber;
        if (session.watcher && session.storePin != saved->storePin) session.solveQueue.noteSessionMutated();
        session.storePin = saved->storePin;
        const auto ordering = storedIndexMapForSessionControls(session.controlPoints);
        for (size_t i = 0; i < ordering.size(); ++i)
            if (ordering[i] >= 0 && size_t(ordering[i]) < saved->controlPoints.size() &&
                session.controlPoints[i].volumePoint == saved->controlPoints[ordering[i]])
                session.controlPoints[i].storeId = saved->controlPoints[ordering[i]].storeId;
    }
    for (const auto& snapshot : snapshots) emit fiberSaved(snapshot.fiber.id, snapshot.fiber.generation);
}

void LineAnnotationController::waitForFiberSaves()
{
    if (_state) _state->objectStoreSession()->finalize();
}

bool LineAnnotationController::isReciprocalBranchRef(const StoredFiber& fiber,
                                                     const FiberBranchRef& branch,
                                                     const FiberBranchRef& candidate)
{
    return vc3d::line_annotation::reciprocalBranchRefMatches(
        fiber, branch, candidate,
        [](const cv::Vec3d& a, const cv::Vec3d& b) {
            return pointsApproximatelyEqual(a, b);
        },
        [](const cv::Vec3d& a, const cv::Vec3d& b) {
            return branchDirectionsCompatible(a, b);
        });
}

void LineAnnotationController::showError(const QString& message, bool suppressDialog) const
{
    if (_activeFiberSaveBatch) {
        _activeFiberSaveBatch->addError(message);
        Logger()->warn("Line Annotation (headless save): {}", message.toStdString());
        return;
    }
    if (_errorDialogsSuppressed || suppressDialog) {
        if (_errorDialogsSuppressed) {
            _lastSuppressedError = message;
        }
        Logger()->warn("Line Annotation (suppressed dialog): {}", message.toStdString());
        return;
    }
    if (_parentWidget) {
        QMessageBox::warning(_parentWidget, tr("Line Annotation"), message);
    } else {
        Logger()->warn("Line Annotation: {}", message.toStdString());
    }
}

void LineAnnotationController::setErrorDialogsSuppressed(bool suppressed)
{
    _errorDialogsSuppressed = suppressed;
    if (!suppressed) {
        _lastSuppressedError.clear();
    }
}

bool LineAnnotationController::errorDialogsSuppressed() const
{
    return _errorDialogsSuppressed;
}

QString LineAnnotationController::takeLastSuppressedError()
{
    QString message = _lastSuppressedError;
    _lastSuppressedError.clear();
    return message;
}

void LineAnnotationController::ensureFiberLines(const StoredFiber& fiber) const
{
    if (!fiber.editorError.empty()) throw std::runtime_error(fiber.editorError);
    if (fiber.linePointsLoaded) return;
    if (!fiber.storePin || !fiber.storePin->geometryRevision) throw std::runtime_error("Fiber geometry pin is missing");
    const auto& pin = *fiber.storePin;
    std::vector<cv::Vec3d> points;
    _state->objectStoreSession()->read(pin.attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto view = vc::objects::GeometrySnapshot(db, txn).fiber(fiber.uuid, *pin.geometryRevision);
        const auto line = storedFiberLinePoints(view, fiber.frame);
        points.reserve(line.size());
        for (const auto& point : line) points.emplace_back(point[0],point[1],point[2]);
    });
    fiber.linePoints = std::move(points);
    fiber.linePointsLoaded = true;
}

std::vector<LineAnnotationController::StoredFiber> LineAnnotationController::readStoredFibers(
    const std::optional<std::vector<std::string>>& uuids, bool lines) const
{
    using namespace vc::objects;
    std::vector<StoredFiber> result;
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment || (uuids && uuids->empty())) return result;
    const bool readOnly = !connection->writable(*attachment);
    const auto volume = _state->currentVolume();
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto& cancel) {
        // Fibers bound to the open volume (by scale or a registration) are read in its voxels.
        const auto transforms = volume ? std::optional(db.volumeTransforms(txn)) : std::nullopt;
        auto decode = [&](const ObjectRecord& object, const ObjectRevision& head, bool withLines) {
            StoredFiber fiber;
            fiber.uuid = object.uuid;
            fiber.readOnly = readOnly;
            fiber.linePointsLoaded = withLines; fiber.catalogDetail = object.fiber;
            fiber.fileName = object.name + ".json";
            fiber.generation = object.generation;
            fiber.tags = object.tags;
            fiber.storePin = PinnedObject{*attachment, {attachment->storeUuid, object.uuid}, head.uuid, head.geometryRevision};
            fiber.id = fiberIdForUuid(object.uuid);
            try {
                if (!head.geometryRevision) throw std::runtime_error("Fiber has no geometry");
                // Controls always; the sampled line only when asked for.
                const auto view = GeometrySnapshot(db, txn).fiber(object.uuid, *head.geometryRevision);
                const auto metadata = nlohmann::json::parse(object.meta);
                if (transforms)
                    if (const auto mapping = transforms->toVolume(metadata, volume->id()))
                        fiber.frame = vc3d::line_annotation::FiberVolumeFrame{volume->id(), *mapping};
                vc::geometry::Fiber geometry;
                geometry.controlPointIds = view.controlPointIds(0, view.controlPointCount());
                geometry.controlPoints = fiber.frame
                    ? view.viewTransformedAt(volume->id()).points("control_points", 0, view.controlPointCount())
                    : view.controlPoints(0, view.controlPointCount());
                if (withLines) geometry.linePoints = storedFiberLinePoints(view, fiber.frame);
                const auto source = metadata.value("source", nlohmann::json::object());
                fiber.username = source.value("username", std::string{"anon"});
                fiber.startedAt = source.value("started_at", std::string{});
                fiber.sequence = source.value("sequence", uint64_t{0});
                fiber.modelOnly = source.value("model_only", false);
                if (fiber.frame) {
                    const auto shape = volume->shape();
                    fiber.coordinateBaseShapeZYX = std::array<size_t, 3>{size_t(shape[0]), size_t(shape[1]), size_t(shape[2])};
                } else if (source.contains("coordinate_base_shape_zyx"))
                    fiber.coordinateBaseShapeZYX = source.at("coordinate_base_shape_zyx").template get<std::array<size_t,3>>();
                for (const auto* key : {"coordinates", "coordinate_binding"})
                    if (metadata.contains(key)) fiber.coordinateIdentity[key] = metadata.at(key);
                if (object.fiber) {
                    fiber.width = fiber.frame ? vc::fibers::mapFiberWidth(fiber.frame->storedToVolume, object.fiber->width)
                                              : object.fiber->width;
                    fiber.widthGapFraction = object.fiber->widthGapFraction;
                    if (!object.fiber->optimizationMode.empty())
                        fiber.optimizationMode = vc3d::line_annotation::fiberOptimizationModeFromString(object.fiber->optimizationMode);
                    if (object.fiber->manualFamily) fiber.manualHvTag = object.fiber->family;
                }
                const auto attributes = nlohmann::json::parse(db.geometryPointAnnotations(txn, *head.geometryRevision, "control_points"));
                for (size_t i = 0; i < geometry.controlPoints.size(); ++i) {
                    auto entry = attributes.at(i);
                    if (entry.is_null()) entry = nlohmann::json::object();
                    entry["position"] = geometry.controlPoints[i];
                    vc3d::line_annotation::StoredControlPoint cp{cv::Vec3d{geometry.controlPoints[i][0],geometry.controlPoints[i][1],geometry.controlPoints[i][2]}};
                    try {
                        cp = vc3d::line_annotation::storedControlPointFromJson(entry, vc3d::line_annotation::kFiberFormatVersion);
                    } catch (const std::exception& error) {
                        fiber.readOnly = true;
                        fiber.editorError = error.what();
                    }
                    cp.storeId = geometry.controlPointIds.at(i);
                    if (fiber.frame) {
                        const auto& m = fiber.frame->storedToVolume;
                        if (cp.direction) cp.direction = vc::fibers::mapFiberDirection(m, *cp.direction);
                        if (cp.displayNormal) cp.displayNormal = vc::fibers::mapFiberNormal(m, *cp.displayNormal);
                    }
                    fiber.controlPoints.push_back(std::move(cp));
                }
                for (const auto& point : geometry.linePoints)
                    fiber.linePoints.emplace_back(point[0], point[1], point[2]);
                fiber.hvClassification = vc3d::line_annotation::classifyFiberHv(
                    vc3d::line_annotation::storedControlPointPositions(fiber.controlPoints));
            } catch (const std::exception& error) {
                fiber.readOnly = true; fiber.editorError = error.what();
                fiber.controlPoints.clear(); fiber.linePoints.clear();
            }
            if (!fiber.editorError.empty())
                Logger()->warn("Fiber {} cannot be edited: {}", fiber.uuid, fiber.editorError);
            return fiber;
        };
        // Branch endpoints outside the request are decoded (controls only) so
        // every reference resolves in this snapshot; only requested fibers return.
        std::vector<ObjectRecord> objects;
        std::set<std::string> requested;
        if (uuids) {
            for (const auto& uuid : *uuids) if (auto object = db.objectByUuid(txn, uuid)) {
                if (object->kind == "fiber" && !object->retired && requested.insert(object->uuid).second)
                    objects.push_back(std::move(*object));
            }
            for (const auto& uuid : std::vector<std::string>(requested.begin(), requested.end()))
                for (const auto& peer : db.neighbours(txn, uuid, std::string{"fiber_branch"}, std::string{"fiber"}))
                    if (!requested.contains(peer) && std::none_of(objects.begin(), objects.end(), [&](const auto& o) { return o.uuid == peer; }))
                        if (auto object = db.objectByUuid(txn, peer); object && !object->retired) objects.push_back(std::move(*object));
        } else {
            ObjectSelection selection; selection.kind = "fiber";
            objects = db.selectObjects(txn, selection);
            for (const auto& object : objects) requested.insert(object.uuid);
        }
        RevisionSelection revisions;
        std::vector<StoredFiber> decoded;
        std::unordered_map<std::string, size_t> byUuid;
        for (const auto& object : objects) {
            if (cancel.load()) throw std::runtime_error("Object store read cancelled");
            const auto head = db.currentObjectRevision(txn, object.uuid);
            if (!head) throw std::runtime_error("Missing object revision");
            revisions[object.uuid] = head->uuid;
            byUuid[object.uuid] = decoded.size();
            decoded.push_back(decode(object, *head, lines && requested.contains(object.uuid)));
        }
        for (const auto& relationship : db.relationshipsAt(txn, revisions)) {
            const auto& edge = relationship.relationship.record;
            if (!relationship.applicable || edge.kind != "fiber_branch") continue;
            if (!requested.contains(edge.aUuid) && !requested.contains(edge.bUuid)) continue;
            auto& a = decoded.at(byUuid.at(edge.aUuid));
            auto& b = decoded.at(byUuid.at(edge.bUuid));
            try {
                const auto index = [](const StoredFiber& fiber, const Anchor& anchor) {
                    std::vector<int64_t> ids;
                    for (const auto& cp : fiber.controlPoints) ids.push_back(cp.storeId);
                    return fiberBranchControlIndex(ids, anchor);
                };
                const auto ai = index(a, edge.aAnchor), bi = index(b, edge.bAnchor);
                const auto append = [&](StoredFiber& local, const StoredFiber& target, size_t i, size_t j,
                                        const Anchor& own, const Anchor& other) {
                    if (!own.direction || !other.direction) throw std::runtime_error("Missing fiber branch direction");
                    FiberBranchRef branch;
                    branch.edgeUuid = edge.uuid;
                    branch.controlPointIndex = i; branch.branchControlPointIndex = j;
                    branch.branchFiberId = target.id; branch.branchFileName = target.fileName;
                    branch.controlPointPosition = local.controlPoints.at(i);
                    branch.branchControlPointPosition = target.controlPoints.at(j);
                    // Anchor directions are in each fiber's stored frame.
                    const auto direction = [](const StoredFiber& fiber, const Anchor& anchor) {
                        const cv::Vec3d d{(*anchor.direction)[0], (*anchor.direction)[1], (*anchor.direction)[2]};
                        return fiber.frame ? vc::fibers::mapFiberDirection(fiber.frame->storedToVolume, d) : d;
                    };
                    branch.controlPointDirection = direction(local, own);
                    branch.branchControlPointDirection = direction(target, other);
                    const auto meta = nlohmann::json::parse(edge.meta);
                    branch.pending = edge.approvalState != "approved";
                    branch.adjacent = meta.value("adjacent", false);
                    branch.autoLink = meta.value("auto", false);
                    vc3d::line_annotation::normalizeBranchRefReviewState(branch);
                    local.branches.push_back(std::move(branch));
                };
                if (requested.contains(a.uuid)) append(a,b,ai,bi,edge.aAnchor,edge.bAnchor);
                if (requested.contains(b.uuid)) append(b,a,bi,ai,edge.bAnchor,edge.aAnchor);
            } catch (const std::exception& error) {
                a.readOnly = b.readOnly = true;
                a.editorError = b.editorError = error.what();
            }
        }
        for (auto& fiber : decoded) if (requested.contains(fiber.uuid)) result.push_back(std::move(fiber));
    });
    return result;
}

std::vector<LineAnnotationController::StoredFiber> LineAnnotationController::readStoredFibersInVolumeBox(
    const std::array<double, 6>& volumeBox, bool lines) const
{
    const auto volume = _state ? _state->currentVolume() : nullptr;
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!volume || !attachment) return {};
    std::vector<std::string> uuids;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        vc::objects::ObjectSelection selection;
        selection.kind = "fiber";
        selection.volume = volume->id();
        selection.bbox = volumeBox;
        for (auto& object : db.selectObjects(txn, selection, std::vector<std::string>{}))
            uuids.push_back(std::move(object.uuid));
    });
    return readStoredFibers(uuids, lines);
}

std::optional<LineAnnotationController::StoredFiber> LineAnnotationController::readStoredFiber(
    uint64_t id, bool lines) const
{
    const auto uuid = fiberUuidForId(id);
    if (uuid.empty()) return std::nullopt;
    auto fibers = readStoredFibers(std::vector<std::string>{uuid}, lines);
    if (fibers.empty()) return std::nullopt;
    return std::move(fibers.front());
}

void LineAnnotationController::resetStoreDerivedState()
{
    ++_attachmentSequence;
    _linkCandidate.reset();
    _controlSpanCache.clear();
    _sideStripStoredSnapshotCache.clear(); _sideStripSessionSnapshotCache.clear();
    _fiberAlignmentMetrics.clear(); _pendingFiberAlignmentMetrics.clear();
    _pendingFiberAlignmentMetricTokens.clear(); ++_nextFiberAlignmentMetricToken;
    ++_fiberDataGeneration; ++_fiberMetricsGeneration; _fiberMetricsPending = false;
    _fiberIdsByUuid.clear(); _fiberUuidsById.clear();
    _fiberSaveFailureCountAtAttachment = _fiberSaveFailureCount;
    emit fibersChanged();
}

void LineAnnotationController::commitFiberStoreAction(std::vector<FiberSaveSnapshot>& snapshots,
                                                       const std::vector<StoredFiber>& retired)
{
    using namespace vc::objects;
    using Json = nlohmann::json;
    const int exceptions = std::uncaught_exceptions();
    const auto refreshOnFailure = qScopeGuard([this, exceptions] {
        if (std::uncaught_exceptions() > exceptions)
            QTimer::singleShot(0, this, [this] { refreshChangedStoreFibers(); });
    });
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment) throw std::runtime_error("Attach an object store before editing fibers");
    // Working copies receive allocated IDs only after the database commits.
    auto edits = snapshots;
    vc3d::line_annotation::canonicalizeFiberSaveSnapshots(edits);
    vc3d::line_annotation::validateFiberSaveSnapshots(edits);
    RevisionSelection expected;
    auto expect = [&](const StoredFiber& fiber) {
        if (fiber.storePin && fiber.storePin->attachment != *attachment)
            throw std::runtime_error("The fiber belongs to a previous object store attachment");
        const auto revision = fiber.storePin && !fiber.storePin->objectRevision.empty()
            ? std::optional(fiber.storePin->objectRevision) : std::nullopt;
        const auto [it, inserted] = expected.emplace(fiber.uuid, revision);
        if (!inserted && it->second != revision) throw std::runtime_error("Conflicting fiber revision expectations");
    };
    // Branch endpoints outside the edit are read from the store's current heads.
    std::vector<std::string> targetUuids;
    for (const auto& edit : edits)
        for (const auto& branch : edit.fiber.branches)
            if (std::none_of(edits.begin(), edits.end(), [&](const auto& other) { return other.fiber.id == branch.branchFiberId; }))
                targetUuids.push_back(fiberUuidForId(branch.branchFiberId));
    const auto targets = readStoredFibers(targetUuids);
    auto findFiber = [&](uint64_t id) -> const StoredFiber& {
        for (const auto& edit : edits) if (edit.fiber.id == id) return edit.fiber;
        for (const auto& fiber : targets) if (fiber.id == id) return fiber;
        throw std::runtime_error("Branch endpoint no longer exists");
    };
    for (const auto& edit : edits) {
        if (edit.fiber.readOnly) throw std::runtime_error(edit.fiber.editorError.empty() ? "Object store is attached read-only" : edit.fiber.editorError);
        expect(edit.fiber);
        for (const auto& branch : edit.fiber.branches) expect(findFiber(branch.branchFiberId));
    }
    for (const auto& fiber : retired) expect(fiber);
    // The action changes the edited and retired fibers; branch targets only gain links.
    RevisionSelection pins, related;
    for (const auto& [uuid, revision] : expected) {
        const bool changed = std::any_of(edits.begin(), edits.end(), [&](const auto& edit) { return edit.fiber.uuid == uuid; }) ||
            std::any_of(retired.begin(), retired.end(), [&](const auto& fiber) { return fiber.uuid == uuid; });
        (changed ? pins : related).emplace(uuid, revision);
    }
    std::map<std::string, std::string> geometryHeads;
    const auto committed = connection->editKeepingRelationships(*attachment, pins, related, [&](auto& db, auto& txn, const auto& decisions) {
        // A fiber read in an open volume, back in its stored frame: positions
        // through the store's view (unchanged points keep their exact values),
        // vectors and width through the inverse; classification in that frame.
        std::map<std::string, StoredFiber> storedFrames;
        auto inStoredFrame = [&](const StoredFiber& fiber) -> const StoredFiber& {
            if (const auto found = storedFrames.find(fiber.uuid); found != storedFrames.end()) return found->second;
            auto stored = fiber;
            if (fiber.frame) {
                const auto shown = GeometrySnapshot(db, txn).viewTransformedAt(fiber.uuid, fiber.frame->volume);
                if (shown.mapping() != fiber.frame->storedToVolume)
                    throw std::runtime_error("The fiber's volume registration changed; reopen it");
                vc::geometry::Fiber edited;
                for (const auto& cp : fiber.controlPoints) {
                    edited.controlPoints.push_back({cp[0], cp[1], cp[2]});
                    edited.controlPointIds.push_back(cp.storeId);
                }
                for (const auto& point : fiber.linePoints) edited.linePoints.push_back({point[0], point[1], point[2]});
                const auto canonical = shown.canonicalFiber(edited);
                const auto inverse = fiber.frame->storedToVolume.inverse();
                for (size_t i = 0; i < stored.controlPoints.size(); ++i) {
                    auto& cp = stored.controlPoints[i];
                    cp[0] = canonical.controlPoints[i][0]; cp[1] = canonical.controlPoints[i][1]; cp[2] = canonical.controlPoints[i][2];
                    if (cp.direction) cp.direction = vc::fibers::mapFiberDirection(inverse, *cp.direction);
                    if (cp.displayNormal) cp.displayNormal = vc::fibers::mapFiberNormal(inverse, *cp.displayNormal);
                }
                for (size_t i = 0; i < stored.linePoints.size(); ++i)
                    stored.linePoints[i] = {canonical.linePoints[i][0], canonical.linePoints[i][1], canonical.linePoints[i][2]};
                stored.width = vc::fibers::mapFiberWidth(inverse, fiber.width);
                stored.hvClassification = vc3d::line_annotation::classifyFiberHv(
                    vc3d::line_annotation::storedControlPointPositions(stored.controlPoints));
                // The object keeps the frame it was created in.
                stored.coordinateBaseShapeZYX.reset();
                stored.frame.reset();
            }
            return storedFrames.emplace(fiber.uuid, std::move(stored)).first->second;
        };
        std::map<std::string, EdgeRecord> incident;
        for (const auto& [uuid, revision] : expected)
            if (db.objectByUuid(txn, uuid))
                for (const auto& edge : db.edgesOf(txn, uuid, true)) incident[edge.uuid] = edge;
        for (auto& snapshot : edits) {
            auto& fiber = snapshot.fiber;
            const auto current = db.objectByUuid(txn, fiber.uuid);
            auto record = current.value_or(ObjectRecord{});
            record.uuid = fiber.uuid; record.kind = "fiber";
            record.name = fs::path(fiber.fileName).stem().string(); record.tags = fiber.tags;
            record.formatVersion = vc3d::line_annotation::kFiberFormatVersion;
            auto meta = Json::parse(record.meta);
            if (!current) for (const auto* key : {"coordinates", "coordinate_binding"})
                if (snapshot.fiber.coordinateIdentity.contains(key)) meta[key] = snapshot.fiber.coordinateIdentity.at(key);
            fiber.coordinateIdentity = Json::object();
            for (const auto* key : {"coordinates", "coordinate_binding"})
                if (meta.contains(key)) fiber.coordinateIdentity[key] = meta.at(key);
            auto source = meta.value("source", Json::object());
            const auto& stored = inStoredFrame(fiber);
            auto incoming = fiberSourceMetadata(stored);
            // An existing object keeps the coordinate frame it was created in.
            if (source.contains("coordinate_base_shape_zyx")) incoming.erase("coordinate_base_shape_zyx");
            source.update(incoming, true);
            if (source.contains("coordinate_base_shape_zyx"))
                fiber.coordinateBaseShapeZYX = source.at("coordinate_base_shape_zyx").template get<std::array<size_t,3>>();
            int64_t nextId = source.value("next_control_point_id", int64_t(0));
            for (const auto& cp : fiber.controlPoints) nextId = std::max(nextId, cp.storeId + 1);
            std::set<int64_t> ids;
            vc::geometry::Fiber geometry;
            // IDs are allocated on the working copy; positions are the stored frame's.
            for (auto& cp : fiber.controlPoints) {
                if (cp.storeId < 0 || ids.contains(cp.storeId)) cp.storeId = nextId++;
                ids.insert(cp.storeId);
                geometry.controlPointIds.push_back(cp.storeId);
            }
            for (const auto& cp : stored.controlPoints) geometry.controlPoints.push_back({cp[0], cp[1], cp[2]});
            for (const auto& point : stored.linePoints) geometry.linePoints.push_back({point[0], point[1], point[2]});
            source["next_control_point_id"] = nextId;
            FiberDetail detail;
            detail.controlPointCount = geometry.controlPoints.size(); detail.linePointCount = geometry.linePoints.size();
            detail.length = geometry.length(); detail.width = stored.width; detail.widthGapFraction = fiber.widthGapFraction;
            detail.optimizationMode = vc3d::line_annotation::fiberOptimizationModeToString(fiber.optimizationMode);
            detail.manualFamily = !fiber.manualHvTag.empty();
            detail.family = detail.manualFamily ? fiber.manualHvTag : vc3d::line_annotation::fiberHvTagToString(stored.hvClassification.automaticTag);
            record.fiber = detail;
            const auto dataToSave = vc::geometry::encode(geometry);
            auto head = current ? db.currentGeometry(txn, fiber.uuid) : std::nullopt;
            if (!head) {
                meta["source"] = source; record.meta = meta.dump();
                db.upsertObject(txn, record);
                geometryHeads[fiber.uuid] = db.putGeometry(txn, fiber.uuid, dataToSave);
            } else {
                const auto revision = updateGeometryArrays(db, txn, fiber.uuid, dataToSave, head->uuid,
                    {{"control_points", source.at("control_points_attributes").dump()}}, decisions);
                geometryHeads[fiber.uuid] = revision;
                // Sequence annotations are immutable and live with geometry.
                source.erase("control_points_attributes");
                meta["source"] = source; record.meta = meta.dump(); db.upsertObject(txn, record);
            }
        }
        std::set<std::string> retiredUuids;
        for (const auto& fiber : retired) retiredUuids.insert(fiber.uuid);
        std::set<std::string> retainedBranches;
        for (auto& snapshot : edits) {
            auto& local = snapshot.fiber;
            for (auto& branch : local.branches) {
                const auto& target = findFiber(branch.branchFiberId);
                if (retiredUuids.contains(target.uuid)) throw std::runtime_error("Branch still references a retired fiber");
                // Anchors are in each fiber's stored frame; IDs come from the working copy.
                auto anchor = [&](const StoredFiber& fiber, int index, const cv::Vec3d& shown) {
                    const auto& stored = inStoredFrame(fiber);
                    const auto direction = fiber.frame
                        ? vc::fibers::mapFiberDirection(fiber.frame->storedToVolume.inverse(), shown) : shown;
                    Anchor a;
                    const auto& cp = stored.controlPoints.at(index);
                    a.position = {cp[0],cp[1],cp[2]}; a.direction = {direction[0],direction[1],direction[2]};
                    a.points = {{"control_point_ids", {fiber.controlPoints.at(index).storeId}}}; return a;
                };
                EdgeRecord edge; edge.uuid = branch.edgeUuid; edge.kind = "fiber_branch";
                edge.aUuid = local.uuid; edge.bUuid = target.uuid;
                edge.aAnchor = anchor(local,branch.controlPointIndex,branch.controlPointDirection);
                edge.bAnchor = anchor(target,branch.branchControlPointIndex,branch.branchControlPointDirection);
                // The automatic mark lives only while the link is pending and ordinary.
                const auto reviewMeta = [&branch](Json metadata) {
                    metadata["adjacent"] = branch.adjacent;
                    if (branch.autoLink && branch.pending && !branch.adjacent) metadata["auto"] = true;
                    else metadata.erase("auto");
                    return metadata.dump();
                };
                edge.meta = reviewMeta(Json::object());
                // Reciprocal UI references describe one shared edge.
                for (const auto& existing : db.edgesOf(txn,local.uuid)) {
                    const bool forward = existing.aUuid == edge.aUuid && existing.bUuid == edge.bUuid;
                    const bool reverse = existing.bUuid == edge.aUuid && existing.aUuid == edge.bUuid;
                    if (existing.kind != "fiber_branch" || (!forward && !reverse)) continue;
                    if ((!edge.uuid.empty() && edge.uuid == existing.uuid) ||
                        ((forward ? existing.aAnchor.points == edge.aAnchor.points && existing.bAnchor.points == edge.bAnchor.points : existing.bAnchor.points == edge.aAnchor.points && existing.aAnchor.points == edge.bAnchor.points) &&
                         Json::parse(existing.meta).value("adjacent",false) == branch.adjacent)) {
                        edge.uuid = existing.uuid; edge.tags = existing.tags;
                        edge.meta = reviewMeta(Json::parse(existing.meta)); break;
                    }
                }
                if (!edge.uuid.empty()) {
                    const auto old = db.edgeByUuid(txn, edge.uuid);
                    if (old && !((old->aUuid == edge.aUuid && old->bUuid == edge.bUuid) ||
                                 (old->bUuid == edge.aUuid && old->aUuid == edge.bUuid))) edge.uuid.clear();
                }
                if (!edge.uuid.empty() && db.edgeByUuid(txn,edge.uuid)) db.updateEdge(txn,edge);
                else edge.uuid = db.addEdge(txn,edge);
                db.retainRelationship(txn,edge.uuid);
                const auto approval = branch.pending ? "pending" : "approved";
                if (db.edgeByUuid(txn,edge.uuid)->approvalState != approval) db.setApproval(txn,edge.uuid,approval);
                retainedBranches.insert(edge.uuid); branch.edgeUuid = edge.uuid;
            }
        }
        std::set<std::string> edited;
        for (const auto& snapshot : edits) edited.insert(snapshot.fiber.uuid);
        for (const auto& [uuid,edge] : incident) {
            if (edge.kind != "fiber_branch") { db.keepRecordedRelationship(txn,uuid); continue; }
            if ((edited.contains(edge.aUuid) || edited.contains(edge.bUuid) || retiredUuids.contains(edge.aUuid) || retiredUuids.contains(edge.bUuid)) && !retainedBranches.contains(uuid)) db.removeEdge(txn,uuid);
        }
        for (const auto& fiber : retired) db.retireObject(txn,fiber.uuid);
        for (auto& snapshot : edits) snapshot.fiber.generation = db.objectByUuid(txn, snapshot.fiber.uuid)->generation;
    });
    for (auto& snapshot : edits) {
        auto& fiber = snapshot.fiber;
        fiber.storePin = PinnedObject{*attachment, {attachment->storeUuid, fiber.uuid},
                                     *committed.at(fiber.uuid), geometryHeads.at(fiber.uuid)};
        registerFiberId(fiber.uuid, fiber.id);
        retainStoredFiberCoordinateBaseShape(fiber);
    }
    // Shared links also revise the other endpoint: keep its open sessions
    // current rather than treating the receipt as an external change.
    for (auto& pane : _panes) {
        if (!pane.session || !pane.session->storePin) continue;
        auto& session = *pane.session;
        if (std::any_of(edits.begin(), edits.end(), [&](const auto& edit) { return edit.fiber.uuid == session.fiberUuid; })) continue;
        const auto receipt = committed.find(session.fiberUuid);
        if (receipt == committed.end() || !receipt->second || session.storePin->objectRevision == *receipt->second) continue;
        if (session.watcher) session.solveQueue.noteSessionMutated();
        session.storePin->objectRevision = *receipt->second;
    }
    snapshots = std::move(edits);
}

#ifdef VC_TESTING
#include "LineAnnotationObjectStoreTest.inc"
#endif

bool LineAnnotationController::ensureFiberInVolume(uint64_t id, bool prompt)
{
    const auto volume=_state ? _state->currentVolume() : nullptr;
    if (fiberBoundToVolume(id, volume)) return true;
    if (!prompt || _errorDialogsSuppressed || !volume) return false;
    const auto stored=readStoredFiber(id, true);
    if (!stored || !stored->storePin) return false;
    const auto& fiber=*stored; const auto pin=*fiber.storePin;
    using Json=nlohmann::json;
    Json metadata;
    const auto connection=_state->objectStoreSession();
    connection->read(pin.attachment,[&](const auto& db,const auto& txn,const auto&) {
        const auto record=db.objectRevision(txn,pin.objectRevision);
        if (!record) throw std::runtime_error("Fiber revision missing");
        metadata=Json::parse(record->record.meta);
    });
    const auto coordinates=metadata.value("coordinates",Json::object());
    const auto shape=volume->shape();
    const auto source=metadata.value("source",Json::object());
    Json known=coordinates;
    for(const auto* key:{"coordinate_base_shape_zyx","vc_open_data_coordinate_space","vc_open_data_source_coordinate_scale_factor","vc_open_data_resolution_um"})
        if(source.contains(key)) known[key]=source[key];
    cv::Vec3d lower{INFINITY,INFINITY,INFINITY},upper{-INFINITY,-INFINITY,-INFINITY};
    for (const auto& point : fiber.linePoints) for(int axis=0;axis<3;++axis) {
        lower[axis]=std::min(lower[axis],point[axis]);upper[axis]=std::max(upper[axis],point[axis]);
    }
    const auto message=tr("Assign this fiber to the active volume?\n\nVolume: %1\nDimensions Z/Y/X: %2 / %3 / %4\nResolution: %5 µm\nFiber metadata: %6\nFiber bounds XYZ: (%7, %8, %9) – (%10, %11, %12)\n\nEnter stored fiber units per active-volume voxel. No scale is assumed.")
        .arg(QString::fromStdString(volume->id())).arg(shape[0]).arg(shape[1]).arg(shape[2]).arg(volume->voxelSize())
        .arg(QString::fromStdString(known.dump())).arg(lower[0]).arg(lower[1]).arg(lower[2]).arg(upper[0]).arg(upper[1]).arg(upper[2]);
    bool accepted=false;
    const auto answer=QInputDialog::getText(_parentWidget.data(),tr("Bind Fiber Coordinates"),message,QLineEdit::Normal,{},&accepted);
    if (!accepted) return false;
    bool valid=false;const double scale=answer.toDouble(&valid);
    if (!valid || !(scale>0) || !std::isfinite(scale)) { showError(tr("Enter a positive finite coordinate scale."));return false; }
    vc3d::bindCoordinates(*connection,pin.attachment,fiber.uuid,pin.objectRevision,*volume,scale);
    return true;
}

bool LineAnnotationController::fiberBoundToVolume(uint64_t id, const std::shared_ptr<Volume>& volume) const
{
    const auto uuid = fiberUuidForId(id);
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!volume || uuid.empty() || !attachment) return false;
    bool bound = false;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto object = db.objectByUuid(txn, uuid);
        bound = object && !object->retired &&
                db.volumeTransforms(txn).toVolume(nlohmann::json::parse(object->meta), volume->id()).has_value();
    });
    return bound;
}

bool LineAnnotationController::fiberInVolume(const StoredFiber& fiber) const
{
    const auto volume = _state ? _state->currentVolume() : nullptr;
    return volume && fiber.storePin && fiber.frame && fiber.frame->volume == volume->id();
}

uint64_t LineAnnotationController::fiberIdForUuid(const std::string& uuid) const
{
    if (uuid.empty()) return 0;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment) return 0;
    auto [found, inserted] = _fiberIdsByUuid.try_emplace(attachment->storeUuid + "/" + uuid, 0);
    if (inserted) {
        found->second = nextFiberId();
        _fiberUuidsById[found->second] = uuid;
    }
    return found->second;
}

std::string LineAnnotationController::fiberUuidForId(uint64_t id) const
{
    const auto found = _fiberUuidsById.find(id);
    return found == _fiberUuidsById.end() ? std::string{} : found->second;
}

void LineAnnotationController::registerFiberId(const std::string& uuid, uint64_t id) const
{
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!attachment || uuid.empty() || id == 0) return;
    _fiberIdsByUuid[attachment->storeUuid + "/" + uuid] = id;
    _fiberUuidsById[id] = uuid;
}

std::optional<vc::objects::PinnedObject> LineAnnotationController::fiberPin(uint64_t id) const
{
    const auto uuid = fiberUuidForId(id);
    const auto connection = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (uuid.empty() || !attachment) return std::nullopt;
    std::optional<vc::objects::PinnedObject> pin;
    connection->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto head = db.currentObjectRevision(txn, uuid);
        if (head && !head->record.retired)
            pin = vc::objects::PinnedObject{*attachment, {attachment->storeUuid, uuid}, head->uuid, head->geometryRevision};
    });
    return pin;
}
