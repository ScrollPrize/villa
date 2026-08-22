#pragma once

#include <QObject>
#include <QPointF>
#include <QPointer>
#include <QString>
#include <QFutureWatcher>

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

#include "AnnotationFrame.hpp"
#include "UmbilicusOrientationFreshness.hpp"
#include "LineAnnotationFiberClassification.hpp"
#include "LineAnnotationFiberSegments.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "vc/atlas/FiberIntersections.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include "vc/core/util/ScrollUmbilicus.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

class CState;
class FiberSaveBatchTracker;
class FiberSliceOverlayController;
class LineAnnotationDialog;
class QMdiArea;
class QEvent;
class QPoint;
class Surface;
class SurfacePanelController;
class ViewerManager;
class VolumePkg;
class QWidget;

class LineAnnotationController : public QObject
{
    Q_OBJECT

public:
    enum class InitialDirectionMode {
        Sideways,
        ZInOut,
    };

    struct OptimizationTaskResult {
        bool ok = false;
        std::filesystem::path manifestPath;
        cv::Vec3d seedPoint{0.0, 0.0, 0.0};
        std::vector<vc3d::line_annotation::LineControlPoint> controlPoints;
        cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
        InitialDirectionMode initialDirectionMode = InitialDirectionMode::Sideways;
        vc::lasagna::LineOptimizationResult result;
        std::string error;
        std::string eventName;
    };

    struct FiberSummary {
        struct AlignmentMetrics {
            bool available = false;
            bool pending = false;
            int sampleCount = 0;
            double meanErrorDegrees = 0.0;
            double maxErrorDegrees = 0.0;
            std::string error;
        };

        struct SpanSummary {
            int spanIndex = 0;
            int firstControlIndex = 0;
            int secondControlIndex = 0;
            int controlPointCount = 0;
            int linePointCount = 0;
            double lengthVx = 0.0;
            AlignmentMetrics alignment;
            // Actual producer of the stored span geometry: 'C' (cspline),
            // 'L' (lasagna), or 'T' (prediction trace).
            char interpMarker = 'L';
            // Predictions provenance: the fiber-inference manifest the
            // trace ran with (segment_to_next.fiber_manifest); empty for
            // non-trace spans.
            std::string fiberManifest;
        };

        uint64_t id = 0;
        std::string name;
        int controlPointCount = 0;
        int linePointCount = 0;
        double lengthVx = 0.0;
        AlignmentMetrics alignment;
        std::vector<SpanSummary> spans;
        double hvZDistance = 0.0;
        double hvFiberLength = 0.0;
        double horizontalScore = 0.0;
        double verticalScore = 0.0;
        double automaticCertainty = 0.0;
        std::string automaticHvTag;
        std::string manualHvTag;
        std::vector<std::string> tags;
        // Number of fibers in this fiber's branch-link connected component
        // (including itself); 0 when the fiber has no links.
        int linkedFiberCount = 0;
        // Number of branch links on this fiber still awaiting review approval.
        int pendingLinkCount = 0;
        // Interpolation provenance of the stored geometry (see deriveTraceState).
        vc3d::line_annotation::FiberTraceState traceState =
            vc3d::line_annotation::FiberTraceState::Legacy;
    };

    struct FiberSnapshotWithPath {
        std::filesystem::path fiberPath;
        vc::atlas::FiberPolyline fiber;
        uint64_t storedFiberId = 0;
        vc3d::line_annotation::FiberHvClassification hvClassification;
        std::string manualHvTag;
        std::vector<std::string> tags;
    };

    // Persisted branch-link metadata. Live branch refs are coupled to
    // LineAnnotationSession::controlPoints, reciprocal refs in linked fibers, and
    // saved-fiber control-point ordering. Any live mutation of control points or
    // branches must go through the private session paths that call
    // syncLinkedBranchMetadataAfterFiberModification().
    struct FiberBranchRef {
        int controlPointIndex = -1;
        uint64_t branchFiberId = 0;
        int branchControlPointIndex = -1;
        std::string branchFileName;
        cv::Vec3d controlPointDirection{0.0, 0.0, 0.0};
        cv::Vec3d branchControlPointDirection{0.0, 0.0, 0.0};
        cv::Vec3d controlPointPosition{0.0, 0.0, 0.0};
        cv::Vec3d branchControlPointPosition{0.0, 0.0, 0.0};
        // Link awaits reviewer approval; kept in sync on both reciprocal refs.
        bool pending = false;
    };

    // Per-fiber data for the fiber overlay's "Show linked" mode. Only fibers
    // with at least one valid cross-fiber link are returned. linkGroupId is
    // the smallest fiber id in the fiber's connected component over all
    // branch links, pending included — same union-find semantics as
    // fiberSummaries().
    struct FiberLinkOverlayInfo {
        uint64_t fiberId = 0;
        uint64_t linkGroupId = 0;
        // (local control point index, pending); one entry per linked control
        // point, pending winning when a point carries both link states.
        std::vector<std::pair<int, bool>> linkedControlPoints;
    };

    using DatasetPicker =
        std::function<std::optional<std::string>(QWidget*, const std::filesystem::path&)>;
    using VolumeSelectorFactory = std::function<QWidget*(QWidget*)>;
    using OptimizationTaskFactory =
        std::function<OptimizationTaskResult(std::filesystem::path,
                                             std::vector<vc3d::line_annotation::LineControlPoint>,
                                             std::vector<cv::Vec3d>,
                                             cv::Vec3d,
                                             InitialDirectionMode,
                                             int,
                                             bool,
                                             int,
                                             int)>;

    LineAnnotationController(CState* state,
                             ViewerManager* viewerManager,
                             QWidget* parentWidget,
                             QObject* parent = nullptr);
    ~LineAnnotationController() override;

    bool canLaunchFromViewer(const CChunkedVolumeViewer* viewer) const;
    void launchFromViewerAtPoint(CChunkedVolumeViewer* viewer,
                                 const QPointF& scenePoint,
                                 bool replaceOwningAnnotation = true);
    void openFiber(uint64_t fiberId);
    void openFiberAtControlPoint(uint64_t fiberId, int controlPointIndex);
    void openFiberAtLinePointIndex(uint64_t fiberId, int linePointIndex);
    void openFiberSpan(uint64_t fiberId, int firstControlIndex, int secondControlIndex);
    void deleteFiber(uint64_t fiberId);
    void deleteFibers(std::vector<uint64_t> fiberIds);
    void renameFiberFile(uint64_t fiberId);
    void importFibers();
    void exportFibers();
    void setFiberManualHvTag(uint64_t fiberId, const QString& tag);
    void setFiberTag(uint64_t fiberId, const QString& tag, bool enabled);
    void recalculateFiberHvClassification(uint64_t fiberId);
    void recalculateAllFiberHvClassifications();
    void calculateFiberAlignmentMetrics();
    void calculateFiberAlignmentMetrics(std::vector<uint64_t> orderedFiberIds);
    void requestFiberAlignmentMetrics(uint64_t fiberId);
    void createAtlasFromFiber(uint64_t fiberId);
    void addFiberToPointCollection(uint64_t fiberId);
    void addFibersToPointCollections(std::vector<uint64_t> fiberIds);
    void showFiberSlice(uint64_t fiberId, QMdiArea* targetArea);
    void showIntersectionInspection(const vc::atlas::FiberIntersectionResult& result,
                                    QMdiArea* targetArea,
                                    std::optional<std::filesystem::path> atlasDir = std::nullopt);
    // Shows an intersection inspection without a dialog, reporting failures
    // through errorMessage.
    bool showIntersectionInspectionHeadless(const vc::atlas::FiberIntersectionResult& result,
                                            QMdiArea* targetArea,
                                            std::optional<std::filesystem::path> atlasDir,
                                            QString* errorMessage = nullptr);
    void saveOpenFibers();
    using FiberSaveCompletion = std::function<void(bool, const QString&)>;
    void saveOpenFibersHeadless(FiberSaveCompletion onFinished);
    void closeFiberWindowForSurface(const std::string& surfaceName);
    bool showGeneratedControlPointContextMenu(CChunkedVolumeViewer* viewer,
                                              const QPointF& scenePoint,
                                              const QPoint& globalPos);
    [[nodiscard]] std::vector<FiberSummary> fiberSummaries() const;
    [[nodiscard]] std::vector<FiberLinkOverlayInfo> fiberLinkOverlayInfos() const;
    // Display name as shown in the fiber panel (file stem, "unnamed" fallback).
    [[nodiscard]] QString fiberDisplayName(uint64_t fiberId) const;
    [[nodiscard]] std::vector<std::string> knownFiberTags() const;
    [[nodiscard]] std::vector<vc::atlas::FiberPolyline> fiberSnapshots() const;
    [[nodiscard]] std::vector<vc::atlas::FiberPolyline> fiberSnapshotsFromStorage() const;
    [[nodiscard]] std::vector<FiberSnapshotWithPath> fiberSnapshotsFromStorageWithPaths() const;
    [[nodiscard]] std::optional<uint64_t> fiberIdForAtlasPath(
        const std::filesystem::path& atlasFiberPath) const;

    // Dialog-free operation entry points. Distinct names avoid ambiguity where
    // their interactive counterparts are used in connect().

    // Writes a vc3d_fiber_collection bundle without opening a dialog.
    bool exportFibersToPath(const std::filesystem::path& path, double scale,
                            QString* errorMessage = nullptr, int* exportedCount = nullptr);
    // Imports a fiber JSON, bundle, or directory without opening a dialog.
    bool importFibersFromPath(const std::filesystem::path& path, double scale,
                              QString* errorMessage = nullptr,
                              int* importedCount = nullptr, int* skippedCount = nullptr);
    // Creates an atlas without dialogs. It does not emit atlasCreated because
    // that signal is connected to the interactive display path.
    bool createAtlasFromFiberHeadless(uint64_t fiberId, QString* errorMessage = nullptr,
                                      std::filesystem::path* atlasDirOut = nullptr);
    // Most recently opened live line-annotation workspace, or nullptr.
    [[nodiscard]] LineAnnotationDialog* mostRecentLineAnnotationDialog() const;
    // Short-lived presentation guard for direct operations. Sessions created
    // while it is set retain the same error policy.
    void setErrorDialogsSuppressed(bool suppressed);
    [[nodiscard]] bool errorDialogsSuppressed() const;
    [[nodiscard]] QString takeLastSuppressedError();

    void setDatasetPickerForTesting(DatasetPicker picker);
    void setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory);
    // Replaces the modal QMessageBox that guards fiber optimization-mode
    // switches; the callback receives the requested mode and returns
    // whether to proceed.
    void setModeChangeConfirmationForTesting(
        std::function<bool(vc3d::line_annotation::FiberOptimizationMode)> confirmer);
    // Replaces the modal QMessageBox shown when the two fibers of a merge
    // carry different optimization modes; the callback receives (clicked,
    // candidate) modes and returns the mode to keep, or nullopt to cancel.
    void setMergeModePickerForTesting(
        std::function<std::optional<vc3d::line_annotation::FiberOptimizationMode>(
            vc3d::line_annotation::FiberOptimizationMode,
            vc3d::line_annotation::FiberOptimizationMode)> picker);
    void setVolumeSelectorFactory(VolumeSelectorFactory factory);
    void setSurfacePanel(SurfacePanelController* panel);
    void setCurrentAtlasDirectory(std::optional<std::filesystem::path> atlasDir);

    // On-disk JSON path of a stored fiber (empty when the fiber is unknown or
    // not yet saved). Used by cross-panel actions such as adding a fiber to a
    // running Spiral fit.
    [[nodiscard]] std::filesystem::path fiberFilePath(uint64_t fiberId) const;

    // Bumped whenever the project's umbilicus attachment changes. Cheap to
    // read, so holders of geometry placed relative to the umbilicus can compare
    // it lazily instead of being signalled.
    [[nodiscard]] uint64_t umbilicusGeneration() const
    {
        return _umbilicusGeneration;
    }

    // Ends every session — saving through the normal close path — while the
    // current package is still the one the work belongs to. The project-open
    // flows call this before replacing the package, and must not replace it
    // when this returns false: an optimization is running, or a finalization
    // failed, and the workspace is left intact for the user to resolve.
    [[nodiscard]] bool prepareForPackageSwitch();

signals:
    void lineAnnotationWorkspaceRequested(LineAnnotationDialog* dialog, const QString& title);
    void fibersChanged(std::vector<LineAnnotationController::FiberSummary> fibers);
    void fiberAlignmentMetricsReset(bool pending);
    void fiberAlignmentMetricsUpdated(
        uint64_t fiberId,
        LineAnnotationController::FiberSummary::AlignmentMetrics alignment,
        std::vector<LineAnnotationController::FiberSummary::AlignmentMetrics> spanAlignments);
    void fiberSaved(uint64_t fiberId, uint64_t generation);
    void fibersDeleted(std::vector<uint64_t> fiberIds);
    void atlasCreated(std::filesystem::path atlasDir);

private slots:
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void onVolumePackageChanged(std::shared_ptr<VolumePkg> pkg);

private:
    enum class SourceKind {
        Plane,
        Segmentation,
    };

    enum class SessionOptimizationState {
        Unoptimized,
        Incremental,
        Optimized,
    };

    // Intentionally opaque outside LineAnnotationController.cpp. Keeping session
    // state private prevents external code from mutating controlPoints/branches
    // without the branch metadata synchronization hook.
    struct LineAnnotationSession;
    struct IntersectionInspectionSession;
    struct FiberMetricsTaskResult;
    struct ControlSpanRecord {
        int spanIndex = 0;
        int firstControlIndex = 0;
        int secondControlIndex = 0;
        size_t firstLineIndex = 0;
        size_t lastLineIndex = 0;
        double lengthVx = 0.0;
        int linePointCount = 0;
    };
    struct CachedFiberAlignmentMetrics {
        FiberSummary::AlignmentMetrics fiber;
        std::vector<FiberSummary::AlignmentMetrics> spans;
    };
    struct StoredFiber {
        uint64_t id = 0;
        std::string username;
        std::string startedAt;
        uint64_t sequence = 0;
        std::string fileName;
        uint64_t generation = 1;
        std::vector<vc3d::line_annotation::StoredControlPoint> controlPoints;
        std::vector<cv::Vec3d> linePoints;
        // Stored snapshots only. Live-session branch metadata must be converted
        // through storedFiberFromSession()/saveSessionAsFiber() so the central
        // hook can remap linked control-point indices before serialization.
        std::vector<FiberBranchRef> branches;
        vc3d::line_annotation::FiberHvClassification hvClassification;
        std::string manualHvTag;
        std::vector<std::string> tags;
        vc3d::line_annotation::FiberOptimizationMode optimizationMode =
            vc3d::line_annotation::FiberOptimizationMode::Lasagna;
        bool needsSave = false;
    };

    struct StoredFiberSessionSnapshot {
        StoredFiber fiber;
        std::vector<int> storedIndexForSessionIndex;
    };

    struct FiberSaveSnapshot {
        uint64_t fiberId = 0;
        uint64_t generation = 0;
        std::filesystem::path path;
        StoredFiber fiber;
        nlohmann::json coordinateIdentity = nlohmann::json::object();
    };

    struct FiberSaveJob {
        uint64_t sequence = 0;
        std::vector<FiberSaveSnapshot> snapshots;
        bool showErrors = true;
        std::vector<std::shared_ptr<FiberSaveBatchTracker>> batches;
    };

    struct BranchLinkValidationIssue {
        size_t fiberIndex = 0;
        size_t branchIndex = 0;
        std::string reason;
    };

    struct FiberSaveTaskResult {
        bool ok = false;
        std::vector<uint64_t> fiberIds;
        std::vector<uint64_t> generations;
        std::vector<std::filesystem::path> recoveryFiles;
        std::string error;
    };

    struct BranchMetadataSyncResult {
        std::vector<uint64_t> affectedFiberIds;
    };

    using SideStripMarker =
        vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker;
    using SideStripProgressCallback =
        std::function<void(const std::string& stage, size_t completed, size_t total)>;
    using SideStripPartialResultCallback =
        std::function<void(std::vector<SideStripMarker> markers)>;
    using SideStripCancelCallback = std::function<bool()>;

    struct SideStripIntersectionRequest {
        bool suppressErrorDialogs = false;
        uint64_t token = 0;
        uint64_t cacheKey = 0;
        std::string surfaceName;
        uint64_t sourceFiberId = 0;
        std::vector<uint64_t> excludedFiberIds;
        cv::Mat_<cv::Vec3f> stripPoints;
        std::vector<vc::atlas::FiberPolyline> fibers;
        std::vector<vc::atlas::FiberSideStripLineQuery> branchLinks;
    };

    struct SideStripIntersectionTaskResult {
        bool ok = false;
        bool suppressErrorDialogs = false;
        uint64_t token = 0;
        uint64_t cacheKey = 0;
        std::string surfaceName;
        std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker> markers;
        std::string error;
    };

    struct PaneRecord {
        int id = 0;
        SourceKind sourceKind = SourceKind::Plane;
        std::string surfaceName;
        QPointer<LineAnnotationDialog> dialog;
        std::shared_ptr<LineAnnotationSession> session;
    };

    VolumeSelectorFactory _volumeSelectorFactory;

    std::string nextSurfaceName();
    void cleanupSurfaceName(const std::string& surfaceName);
    bool prepareForUserFacingLineAnnotationOpen();
    bool launchSession(SourceKind sourceKind,
                       const std::string& surfaceName,
                       std::shared_ptr<Surface> sourceSurface,
                       const CChunkedVolumeViewer::CameraState& camera,
                       cv::Vec3d sourceSliceNormal,
                       std::shared_ptr<LineAnnotationSession> session,
                       bool deferShowUntilGenerated = false);
    void openFiberWithControlPoint(uint64_t fiberId,
                                   std::optional<int> controlPointIndex,
                                   std::optional<int> linePointIndex = std::nullopt,
                                   std::optional<std::pair<int, int>> spanControlIndices = std::nullopt);
    void handleLineSeed(const std::string& surfaceName,
                        cv::Vec3f volumePoint,
                        InitialDirectionMode directionMode);
    void handleGeneratedControlPoint(const std::string& surfaceName,
                                     cv::Vec3f volumePoint,
                                     double linePosition);
    void handleGeneratedControlPointDelete(const std::string& surfaceName,
                                           double linePosition,
                                           cv::Vec3f volumePoint);
    void handleGeneratedControlPointBranch(const std::string& surfaceName,
                                           size_t controlPointIndex,
                                           cv::Vec3f linkedControlPoint,
                                           bool openAfterCreate,
                                           cv::Vec3f requestedLinkDirection);
    void handleGeneratedPredSnapPoint(const std::string& surfaceName,
                                      cv::Vec3f volumePoint);
    void handleGeneratedSideStripIntersectionQuery(const std::string& surfaceName);
    void handleGeneratedSegmentInterpolationGoal(const std::string& surfaceName,
                                                 size_t firstControlPointIndex,
                                                 size_t secondControlPointIndex,
                                                 const std::string& goal);
    void handleGeneratedControlPointLinkCandidate(const std::string& surfaceName,
                                                  size_t controlPointIndex,
                                                  cv::Vec3f volumePoint);
    void handleGeneratedControlPointLinkWithCandidate(const std::string& surfaceName,
                                                      size_t controlPointIndex,
                                                      cv::Vec3f volumePoint);
    // Concatenates the link candidate's fiber onto the session's fiber
    // end-to-end (both control points must be endpoints) into one brand-new
    // fiber: tags unioned, third-party links remapped, the pair link between
    // the merge endpoints consumed, both originals deleted, the merged line
    // re-optimized and reopened at the join.
    void handleGeneratedControlPointMergeWithCandidate(const std::string& surfaceName,
                                                       size_t controlPointIndex,
                                                       cv::Vec3f volumePoint);
    void handleGeneratedControlPointSplitCandidate(const std::string& surfaceName,
                                                   size_t controlPointIndex,
                                                   cv::Vec3f volumePoint);
    // Splits the session's fiber between the split candidate and the clicked
    // adjacent control point into two brand-new fibers (fresh identities,
    // tags/mode/span metadata inherited, branch links remapped onto the
    // halves), deletes the original, and reopens the candidate's half.
    // linkHalves additionally records a reciprocal branch link between the
    // two boundary control points ("Split from candidate and link").
    void handleGeneratedControlPointSplitFromCandidate(const std::string& surfaceName,
                                                       size_t controlPointIndex,
                                                       cv::Vec3f volumePoint,
                                                       bool linkHalves);
    void handleGeneratedOpenNearbyAnnotation(uint64_t fiberId, cv::Vec3f volumePoint);
    void handleGeneratedControlPointUnlink(const std::string& surfaceName,
                                           size_t controlPointIndex,
                                           uint64_t branchFiberId,
                                           int branchControlPointIndex);
    void handleGeneratedControlPointSetLinkPending(const std::string& surfaceName,
                                                   size_t controlPointIndex,
                                                   uint64_t branchFiberId,
                                                   int branchControlPointIndex,
                                                   bool pending);
    [[nodiscard]] std::vector<vc3d::line_annotation::GeneratedOverlay::ControlPointMarker>
        controlMarkersForSession(const LineAnnotationSession& session) const;
    [[nodiscard]] vc3d::line_annotation::GeneratedLinkCandidateMenuState
        linkCandidateMenuState(const LineAnnotationSession& session) const;
    [[nodiscard]] vc3d::line_annotation::GeneratedLinkCandidateMenuState
        splitCandidateMenuState(const LineAnnotationSession& session) const;
    [[nodiscard]] vc3d::line_annotation::GeneratedLinkCandidateMenuState
        splitAndLinkCandidateMenuState(const LineAnnotationSession& session) const;
    [[nodiscard]] vc3d::line_annotation::GeneratedLinkCandidateMenuState
        mergeCandidateMenuState(const LineAnnotationSession& session) const;
    [[nodiscard]] std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker>
        markLinkCandidateFiberIntersections(
            std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker> markers,
            const std::vector<FiberBranchRef>& branches) const;
    bool ensureDatasetForSession(LineAnnotationSession& session);
    bool ensureFiberInferenceDatasetForSession(LineAnnotationSession& session);
    void refreshLineAnnotationDatasetMenus() const;
    void refreshLineAnnotationDatasetMenu(LineAnnotationDialog* dialog) const;
    void handleLasagnaDatasetSelectionChanged(const std::string& location);
    void handleFiberInferenceDatasetSelectionChanged(const std::string& location);
    bool needsFinalOptimization(const LineAnnotationSession& session) const;
    bool finalizeSessionOptimizationSynchronously(LineAnnotationSession& session,
                                                  bool fireSuccessCallback);
    void setSessionOptimizationState(LineAnnotationSession& session,
                                     SessionOptimizationState state);
    void refreshSessionOptimizationStatus(const LineAnnotationSession& session);
    bool applyOptimizationTaskResult(LineAnnotationSession& session,
                                     OptimizationTaskResult task,
                                     bool updateGeneratedViews,
                                     SessionOptimizationState resultOptimizationState,
                                     const std::string& eventOverride = {},
                                     bool fireSuccessCallback = true,
                                     bool allowFiberSave = true);
    void requestFinalizedClose(const std::string& surfaceName);
    void startOptimization(LineAnnotationSession& session,
                           bool fullOptimization = false,
                           int activeStart = -1,
                           int activeEnd = -1);
    void startFiberModeOptimization(LineAnnotationSession& session,
                                    bool retraceAll,
                                    std::optional<std::vector<size_t>> dirtySegments = std::nullopt,
                                    bool globalGoalsOnly = false);
    [[nodiscard]] vc3d::line_annotation::FiberModeOptimizationRequest
        makeFiberModeOptimizationRequest(const LineAnnotationSession& session,
                                         bool retraceAll,
                                         std::optional<std::vector<size_t>> dirtySegments = std::nullopt,
                                         bool globalGoalsOnly = false) const;
    // The frame line points and control points are expressed in: the current
    // volume's grid carried to the resolution the fibers were annotated at.
    // Default-constructed (no voxel size, zero extent) when no volume is loaded.
    [[nodiscard]] vc3d::annotation::AnnotationFrame annotationFrame() const;
    // Drops the cached scroll umbilicus and everything describing it, so the next
    // use resolves again.
    void invalidateScrollUmbilicus();
    // Rebuilds the generated views of panes whose recorded orientation epoch is
    // behind the controller's — which is what actually re-applies sheet normals
    // after the umbilicus or the active volume changed — and the intersection
    // inspection's strips, which have no generated-view sessions of their own.
    // Failures are logged per pane and do not stop the others.
    void refreshStaleGeneratedViews();
    // Coalesces refreshStaleGeneratedViews() onto the next event-loop turn:
    // CState emits volumeChanged from inside ViewerManager::switchVolume(),
    // before focus and navigation are restored, and materialization reads pane
    // camera state. Also collapses rapid switching into one rebuild.
    void scheduleStaleViewRefresh();
    // Cheap fingerprint over everything the resolved umbilicus was read from: a
    // stat() of every resolver candidate (the attached file when the project
    // field is set, the discovery candidates otherwise) plus the registration
    // transform the legacy reading would consult — no JSON parse. Part of the
    // cached umbilicus's key, so fixing a refused file, editing a transform in
    // place, or a new candidate appearing all reach the views.
    [[nodiscard]] QString umbilicusCacheToken() const;
    void onActiveVolumeChanged();
    // Pushes _umbilicusNotice to every open pane's dialog.
    void publishUmbilicusNotice();
    void finishOptimization(const std::string& surfaceName);
    // Per-line-point sampled sheet normals, sign-oriented away from the
    // scroll center (umbilicus when available, volume XY center otherwise);
    // NaN entries mark invalid samples.
    [[nodiscard]] std::vector<cv::Vec3f> orientedLineNormalsForSession(
        const LineAnnotationSession& session);
    bool materializeGeneratedViews(LineAnnotationSession& session);
    bool materializeGeneratedViews(LineAnnotationSession& session,
                                   const std::string& surfacePrefix);
    void handleShowAsMesh(const std::string& surfaceName);
    [[nodiscard]] std::filesystem::path resolveMeshExportPathsDir() const;
    [[nodiscard]] std::filesystem::path nextMeshExportPath(const std::filesystem::path& pathsDir,
                                                           const std::string& stem) const;
    [[nodiscard]] std::vector<std::filesystem::path> saveGeneratedQuadMeshes(LineAnnotationSession& session);
    [[nodiscard]] PaneRecord* paneForSurface(const std::string& surfaceName);
    [[nodiscard]] const PaneRecord* paneForSurface(const std::string& surfaceName) const;
    // "H"/"V" from the manual tag, falling back to the automatic classification;
    // empty when unknown or the fiber isn't loaded.
    [[nodiscard]] QString fiberHvDirectionTag(uint64_t fiberId) const;
    // Pushes the H/V tag and the clickable tag buttons to the pane's dialog.
    void pushFiberUiState(const PaneRecord& pane) const;
    [[nodiscard]] std::optional<std::string> pickDataset(QWidget* parent,
                                                          const std::filesystem::path& startDir) const;
    [[nodiscard]] OptimizationTaskResult runOptimizationTask(std::filesystem::path manifestPath,
                                                             std::vector<vc3d::line_annotation::LineControlPoint> controlPoints,
                                                             std::vector<cv::Vec3d> initialLinePoints,
                                                             cv::Vec3d sourceSliceNormal,
                                                             InitialDirectionMode directionMode,
                                                             int initialCenterlineLengthVx,
                                                             bool fullOptimization = false,
                                                             int activeStart = -1,
                                                             int activeEnd = -1) const;
    void loadFibersForCurrentPackage();
    [[nodiscard]] bool validateLoadedFiberLinks(std::vector<StoredFiber>& fibers,
                                                std::vector<std::string>& errors) const;
    // Fibers merged by the sync tool (scripts/fiber_merge.py) carry a
    // needs_reoptimization tag; on load VC3D offers to re-fit their lines.
    // Declining keeps the tag so the next load asks again.
    void promptReoptimizationForMergedFibers();
    // Modal guard before a fiber optimization-mode switch re-optimizes the
    // line; returns false when the user cancels. Suppressed (agent-driven)
    // sessions proceed without prompting.
    [[nodiscard]] bool confirmFiberOptimizationModeChange(
        const LineAnnotationSession& session,
        vc3d::line_annotation::FiberOptimizationMode requestedMode);
    // Modal picker when the two fibers of a merge carry different
    // optimization modes; nullopt cancels the merge. Suppressed
    // (agent-driven) sessions take the clicked fiber's mode.
    [[nodiscard]] std::optional<vc3d::line_annotation::FiberOptimizationMode>
        pickMergeOptimizationMode(
            const LineAnnotationSession& session,
            vc3d::line_annotation::FiberOptimizationMode clickedMode,
            vc3d::line_annotation::FiberOptimizationMode candidateMode);
    // fileNames, not runtime ids: ids are densely reassigned on reloads,
    // which can happen while the prompt's modal spins.
    void reoptimizeMergedFibers(const std::vector<std::string>& fiberFileNames);
    void emitFiberSummaries();
    void addKnownFiberTags(const std::vector<std::string>& tags);
    [[nodiscard]] std::filesystem::path fibersRootDir() const;
    [[nodiscard]] std::filesystem::path fibersDir() const;
    [[nodiscard]] std::filesystem::path relativeFiberPath(const StoredFiber& fiber) const;
    [[nodiscard]] std::filesystem::path fiberPath(uint64_t fiberId) const;
    [[nodiscard]] std::filesystem::path fiberPath(const StoredFiber& fiber) const;
    [[nodiscard]] std::filesystem::path currentVolpkgRoot() const;
    [[nodiscard]] std::vector<std::string> atlasPathKeysForFiber(const StoredFiber& fiber) const;
    [[nodiscard]] std::optional<std::filesystem::path> resolveAtlasFiberPath(
        const StoredFiber& fiber,
        const std::filesystem::path& atlasDir) const;
    void attachAtlasPredSnaps(const StoredFiber& fiber,
                              LineAnnotationSession& session,
                              const std::filesystem::path& atlasDir);
    [[nodiscard]] uint64_t nextFiberId() const;
    [[nodiscard]] uint64_t nextFiberSequenceForUsername(const std::string& username) const;
    [[nodiscard]] std::string currentFiberUsername() const;
    [[nodiscard]] static std::string currentFiberDateTimeString();
    void ensureSessionFiberIdentity(LineAnnotationSession& session);
    [[nodiscard]] std::vector<std::vector<cv::Vec3f>> generatedBranchLinePointsForSession(
        const LineAnnotationSession& session) const;
    void refreshBranchLineViews(uint64_t changedFiberId = 0);
    [[nodiscard]] std::vector<vc::atlas::FiberPolyline> fiberSnapshotsForSideStripQuery() const;
    void startSideStripIntersectionQuery(SideStripIntersectionRequest request);
    void updateSideStripIntersectionProgress(uint64_t token,
                                             const std::string& surfaceName,
                                             const std::string& stage,
                                             size_t completed,
                                             size_t total);
    void applyPartialSideStripIntersectionMarkers(
        uint64_t token,
        const std::string& surfaceName,
        std::vector<SideStripMarker> markers);
    void finishSideStripIntersectionQuery(SideStripIntersectionTaskResult result);
    [[nodiscard]] static SideStripIntersectionTaskResult runSideStripIntersectionQuery(
        const SideStripIntersectionRequest& request,
        SideStripProgressCallback progressCallback = {},
        SideStripPartialResultCallback partialResultCallback = {},
        SideStripCancelCallback cancelCallback = {});
    // Central hook after any live LineAnnotationSession control-point or branch
    // mutation. Pass previous controls/branches when indices or links may have
    // changed, then schedule saves for returned linked fibers as needed.
    BranchMetadataSyncResult syncLinkedBranchMetadataAfterFiberModification(
        LineAnnotationSession& session,
        const std::vector<vc3d::line_annotation::LineControlPoint>* previousControlPoints = nullptr,
        const std::vector<FiberBranchRef>* previousBranches = nullptr);
    void scheduleBranchMetadataSaves(const std::vector<uint64_t>& fiberIds,
                                     uint64_t excludedFiberId = 0);
    void syncBranchFiberFileRename(uint64_t fiberId,
                                   const std::string& oldFileName,
                                   const std::string& newFileName);
    void removeBranchLinksToFiber(uint64_t fiberId, const std::string& fileName);
    // Hook internals; do not call directly from mutation sites.
    void syncReciprocalBranchControlPointReferences(const LineAnnotationSession& session);
    [[nodiscard]] bool confirmLinkedControlPointEdit(const LineAnnotationSession& session,
                                                     int controlPointIndex,
                                                     const QString& action) const;
    [[nodiscard]] bool confirmLinkedControlPointEdits(
        const LineAnnotationSession& session,
        const std::vector<size_t>& controlPointIndices,
        const QString& action) const;
    [[nodiscard]] bool controlPointHasBranch(const LineAnnotationSession& session,
                                             int controlPointIndex) const;
    std::vector<uint64_t> syncBranchEndpointPositions(LineAnnotationSession& session);
    [[nodiscard]] static double lineLengthVx(const std::vector<cv::Vec3d>& points);
    static void scaleStoredFiber(StoredFiber& fiber, double scale);
    [[nodiscard]] static vc::lasagna::LineModel lineModelFromPoints(
        const std::vector<cv::Vec3d>& points,
        const vc::lasagna::NormalSampler* normalSampler);
    [[nodiscard]] static vc::lasagna::LineModel syntheticLineModelFromPoints(
        const std::vector<cv::Vec3d>& points);
    [[nodiscard]] static cv::Vec3d seedTraceSourceNormalForStoredFiber(
        const StoredFiber& fiber,
        std::optional<int> controlPointIndex,
        const cv::Vec3d& seedPoint);
    [[nodiscard]] std::optional<int> storedBranchTargetControlPointIndex(
        const FiberBranchRef& branch) const;
    [[nodiscard]] StoredFiberSessionSnapshot makeStoredFiberSessionSnapshot(
        LineAnnotationSession& session);
    [[nodiscard]] StoredFiber storedFiberFromSession(LineAnnotationSession& session);
    void saveSessionAsFiber(LineAnnotationSession& session);
    [[nodiscard]] nlohmann::json fiberToJson(const StoredFiber& fiber, double scale = 1.0) const;
    void saveFiberNow(const StoredFiber& fiber) const;
    void scheduleFiberSave(const StoredFiber& fiber);
    void scheduleFiberPairSave(const StoredFiber& first, const StoredFiber& second);
    void scheduleFiberSaveSnapshots(std::vector<FiberSaveSnapshot> snapshots,
                                    bool showErrors = true);
    void canonicalizeFiberSaveSnapshots(std::vector<FiberSaveSnapshot>& snapshots) const;
    void validateFiberSaveSnapshots(const std::vector<FiberSaveSnapshot>& snapshots) const;
    void startNextFiberSaveJob();
    void finishFiberSaveJob(QFutureWatcher<FiberSaveTaskResult>* watcher,
                            bool showErrors,
                            std::vector<std::shared_ptr<FiberSaveBatchTracker>> batches);
    void waitForFiberSaves();
    [[nodiscard]] FiberSaveSnapshot makeFiberSaveSnapshot(const StoredFiber& fiber) const;
    [[nodiscard]] static nlohmann::json fiberSaveSnapshotToJson(
        const FiberSaveSnapshot& snapshot,
        double scale = 1.0);
    [[nodiscard]] std::optional<StoredFiber> loadFiberJson(const nlohmann::json& root,
                                                           const std::filesystem::path& path,
                                                           std::vector<std::string>* branchErrors = nullptr) const;
    [[nodiscard]] std::optional<StoredFiber> loadFiberFile(const std::filesystem::path& path) const;
    [[nodiscard]] std::vector<BranchLinkValidationIssue> collectLoadedFiberBranchIssues(
        const std::vector<StoredFiber>& fibers) const;
    [[nodiscard]] bool repairLoadedFiberBranchLinks(
        std::vector<StoredFiber>& fibers,
        const std::unordered_set<std::string>& fibersWithRemovedBranchEntries,
        const std::vector<BranchLinkValidationIssue>& initialIssues,
        std::vector<std::string>& errors) const;
    [[nodiscard]] std::string uniqueImportedFiberFileName(const StoredFiber& fiber,
                                                          std::unordered_set<std::string>& reserved,
                                                          uint64_t& nextSequence) const;
    [[nodiscard]] static std::vector<ControlSpanRecord> controlSpansForFiber(
        const StoredFiber& fiber);
    [[nodiscard]] FiberSummary::AlignmentMetrics cachedAlignmentForFiber(
        uint64_t fiberId) const;
    [[nodiscard]] FiberSummary::AlignmentMetrics cachedAlignmentForSpan(
        uint64_t fiberId,
        int spanIndex) const;
    [[nodiscard]] bool hasCachedAlignmentForFiber(uint64_t fiberId) const;
    [[nodiscard]] bool isAlignmentPendingForFiber(uint64_t fiberId) const;
    [[nodiscard]] bool isAlignmentPendingForFiber(uint64_t fiberId,
                                                  uint64_t requestToken) const;
    [[nodiscard]] std::optional<std::pair<std::filesystem::path, double>>
        resolveAlignmentMetricsManifestPath();
    void requestFiberAlignmentMetricsForFibers(std::vector<uint64_t> fiberIds);
    void publishFiberAlignmentMetrics(uint64_t fiberId,
                                      CachedFiberAlignmentMetrics metrics);
    void publishPendingFiberAlignmentMetrics(const StoredFiber& fiber);
    void publishUnavailableFiberAlignmentMetrics(uint64_t fiberId);
    void invalidateFiberAlignmentMetrics(uint64_t fiberId, bool notify);
    [[nodiscard]] std::vector<vc3d::line_annotation::GeneratedSpanAlignmentMetric>
        generatedSpanAlignmentMetricsForSession(const LineAnnotationSession& session) const;
    void updateGeneratedViewMetricsForFiber(uint64_t fiberId);
    [[nodiscard]] static CachedFiberAlignmentMetrics calculateAlignmentMetricsForFiber(
        const StoredFiber& fiber,
        const std::vector<ControlSpanRecord>& spans,
        const vc::lasagna::NormalSampler& sampler);
    void finishFiberAlignmentMetrics(QFutureWatcher<FiberMetricsTaskResult>* watcher);
    void showError(const QString& message, bool suppressDialog = false) const;
    // Shared core of createAtlasFromFiber / createAtlasFromFiberHeadless.
    // Returns the created atlas directory; throws std::exception on failure.
    // Does not emit atlasCreated (callers decide).
    std::filesystem::path createAtlasFromFiberCore(uint64_t fiberId);
    // Shared per-pane finalize+save loop of saveOpenFibers /
    // saveOpenFibersHeadless (no waiting).
    void saveOpenFibersCore();
    void cleanupIntersectionInspectionSurfaces();
    // Tears down the intersection-inspection workspace when one of its
    // editing sessions holds a fiber that was just retired (split/merge);
    // otherwise the pane would stay open and editable with all saves
    // suppressed.
    void closeIntersectionInspectionForRetiredFibers(
        const std::vector<uint64_t>& fiberIds);
    // Closes every dialog pane whose session holds one of the fibers. The
    // single-dialog invariant means at most the invoking pane matches
    // today; sweeping by fiber id keeps split/merge retirement correct by
    // construction rather than by that invariant.
    void closeDialogPanesForFibers(const std::vector<uint64_t>& fiberIds);
    // Returns false on failure; with a non-null `errorMessage` the failure is
    // reported there (dialog-free), otherwise via showError (interactive).
    bool rebuildIntersectionInspection(QString* errorMessage = nullptr);
    bool updateIntersectionFollowSlice(bool sourceSideFlag,
                                       double linePosition,
                                       const char* reason);
    void toggleIntersectionFollowSlice(bool sourceSideFlag);
    bool handleIntersectionFollowKeyPress(int key, Qt::KeyboardModifiers modifiers);
    bool eventFilter(QObject* watched, QEvent* event) override;
    void refreshIntersectionInspectionAfterEdit(uint64_t editedFiberId,
                                                double oldSourceArclength,
                                                double oldTargetArclength);
    bool acceptIntersectionSameWindingChoice();
    [[nodiscard]] std::shared_ptr<LineAnnotationSession> makeIntersectionLineSession(
        const StoredFiber& fiber,
        double focusLinePosition,
        const cv::Vec3d& sourceSliceNormal,
        const std::string& surfaceName,
        std::function<void()> onOptimizationSucceeded);

    CState* _state = nullptr;
    ViewerManager* _viewerManager = nullptr;
    SurfacePanelController* _surfacePanel = nullptr;
    QPointer<QWidget> _parentWidget;
    int _nextPaneId = 1;
    std::vector<PaneRecord> _panes;
    std::vector<StoredFiber> _fibers;
    std::vector<std::string> _knownFiberTags;
    std::unordered_map<uint64_t, CachedFiberAlignmentMetrics> _fiberAlignmentMetrics;
    std::unordered_set<uint64_t> _pendingFiberAlignmentMetrics;
    std::unordered_map<uint64_t, uint64_t> _pendingFiberAlignmentMetricTokens;
    std::vector<QPointer<QFutureWatcher<FiberMetricsTaskResult>>> _fiberMetricsWatchers;
    uint64_t _nextFiberAlignmentMetricToken = 0;
    uint64_t _fiberMetricsGeneration = 0;
    bool _fiberMetricsPending = false;
    std::unique_ptr<IntersectionInspectionSession> _intersectionInspection;
    std::unique_ptr<FiberSliceOverlayController> _fiberSliceOverlay;
    // Scroll-center reference for orienting generated-view normals; loaded
    // lazily from the volpkg's umbilicus file and re-attempted when the
    // volpkg root changes. nullopt after a failed attempt (volume-center
    // fallback is used instead).
    std::optional<vc::core::util::Umbilicus> _scrollUmbilicus;
    std::filesystem::path _scrollUmbilicusRoot;
    // The resolver's dependencies as of the cached load — a stat of every
    // candidate plus the legacy transform — so any of them changing underneath
    // VC3D (fixed, replaced, removed, or newly appearing) is noticed.
    QString _scrollUmbilicusToken;
    // The annotation frame _scrollUmbilicus was scaled into. Part of the cache
    // key because the cached value is not the file's contents: its points are
    // already multiplied by a frame-dependent factor and its per-slice centres
    // sized to that frame's extent. Keyed on the project directory alone, a
    // volume switch handed the orientation vote geometry from the previous frame.
    vc3d::annotation::AnnotationFrame _scrollUmbilicusFrame;
    // The volume the cached umbilicus was read for. Deliberately conservative:
    // the volume-centre fallback and the legacy reading depend on the volume's
    // raw shape and its registration transform, not only on the annotation
    // frame, so any volume switch re-resolves rather than proving the previous
    // geometry equivalent. Re-resolution is a few stats and one JSON parse.
    std::string _scrollUmbilicusVolumeId;
    bool _scrollUmbilicusLoadAttempted = false;
    // Bumped whenever the orientation inputs change out from under built views
    // (umbilicus attach/detach, active volume switch). Sessions record the
    // value their generated views were built at; refreshStaleGeneratedViews()
    // rebuilds the ones that are behind.
    int _orientationEpoch = 0;
    bool _staleViewRefreshQueued = false;
    // Bumped on attach/detach only. Holders such as the Fiber Map read it to
    // decide staleness of derived geometry; a volume switch is deliberately not
    // an attachment change.
    uint64_t _umbilicusGeneration = 0;
    // The volumeId of the last volumeChanged this controller acted on, so a
    // reselection of the current volume is not treated as a switch. Cleared on
    // package change.
    std::string _lastVolumeChangedId;
    // Why the package's umbilicus could not be used, for the strip notice.
    // Empty when one was applied, and when none exists to complain about.
    // Orienting off the volume centre instead is exactly the silent degradation
    // that hid a frame mismatch for a whole scroll, so it is said out loud.
    QString _umbilicusNotice;
    std::deque<FiberSaveJob> _pendingFiberSaveJobs;
    QPointer<QFutureWatcher<FiberSaveTaskResult>> _fiberSaveWatcher;
    uint64_t _nextFiberSaveSequence = 0;
    bool _fiberSaveRunning = false;
    // Total failed save jobs; callers compare before/after a
    // waitForFiberSaves() flush to gate destructive follow-ups (fiber
    // retirement) on the flushed saves having actually succeeded.
    uint64_t _fiberSaveFailureCount = 0;
    mutable std::shared_ptr<FiberSaveBatchTracker> _activeFiberSaveBatch;
    uint64_t _nextSideStripIntersectionToken = 0;
    uint64_t _latestSideStripIntersectionToken = 0;
    std::shared_ptr<std::atomic<uint64_t>> _latestSideStripIntersectionTokenAtomic =
        std::make_shared<std::atomic<uint64_t>>(0);
    uint64_t _runningSideStripIntersectionToken = 0;
    uint64_t _runningSideStripIntersectionKey = 0;
    std::string _runningSideStripIntersectionSurfaceName;
    uint64_t _lastSideStripIntersectionKey = 0;
    std::string _lastSideStripIntersectionSurfaceName;
    std::vector<SideStripMarker> _lastSideStripIntersectionMarkers;
    bool _sideStripIntersectionRunning = false;
    std::optional<SideStripIntersectionRequest> _pendingSideStripIntersectionRequest;
    std::optional<std::filesystem::path> _currentAtlasDir;
    DatasetPicker _datasetPicker;
    OptimizationTaskFactory _optimizationTaskFactory;
    std::function<bool(vc3d::line_annotation::FiberOptimizationMode)>
        _modeChangeConfirmation;
    std::function<std::optional<vc3d::line_annotation::FiberOptimizationMode>(
        vc3d::line_annotation::FiberOptimizationMode,
        vc3d::line_annotation::FiberOptimizationMode)>
        _mergeModePicker;
    bool _errorDialogsSuppressed = false;
    // Deduplicates the deferred re-optimization prompt across reentrant
    // fiber (re)loads.
    bool _reoptimizationPromptPending = false;
    mutable QString _lastSuppressedError;

    // Transient (in-memory only) staging state for a designated control
    // point: linking two CPs across fibers (_linkCandidate) or splitting a
    // fiber between adjacent CPs (_splitCandidate). Position is the primary
    // key; the stored index is a hint re-resolved at use time because
    // indices are remapped on save.
    struct LinkCandidate {
        uint64_t fiberId = 0;
        std::string fiberFileName;
        cv::Vec3d position{0.0, 0.0, 0.0};
        int storedControlPointIndexHint = -1;
    };
    std::optional<LinkCandidate> _linkCandidate;
    std::optional<LinkCandidate> _splitCandidate;
};
