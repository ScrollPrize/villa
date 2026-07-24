#pragma once

#include <QObject>
#include <QPointF>
#include <QPointer>
#include <QString>
#include <QFutureWatcher>

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

#include "LineAnnotationFiberClassification.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "vc/atlas/FiberIntersections.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

class CState;
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
        std::vector<vc::lasagna::LineControlPoint> controlPoints;
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
                                             std::vector<vc::lasagna::LineControlPoint>,
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
    void launchFromViewer(CChunkedVolumeViewer* viewer, const QPointF& scenePoint);
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
    void saveOpenFibers();
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

    void setDatasetPickerForTesting(DatasetPicker picker);
    void setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory);
    void setVolumeSelectorFactory(VolumeSelectorFactory factory);
    void setSurfacePanel(SurfacePanelController* panel);
    void setCurrentAtlasDirectory(std::optional<std::filesystem::path> atlasDir);

    // On-disk JSON path of a stored fiber (empty when the fiber is unknown or
    // not yet saved). Used by cross-panel actions such as adding a fiber to a
    // running Spiral fit.
    [[nodiscard]] std::filesystem::path fiberFilePath(uint64_t fiberId) const;

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
        std::vector<cv::Vec3d> controlPoints;
        std::vector<cv::Vec3d> linePoints;
        // Stored snapshots only. Live-session branch metadata must be converted
        // through storedFiberFromSession()/saveSessionAsFiber() so the central
        // hook can remap linked control-point indices before serialization.
        std::vector<FiberBranchRef> branches;
        vc3d::line_annotation::FiberHvClassification hvClassification;
        std::string manualHvTag;
        std::vector<std::string> tags;
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
    void handleGeneratedControlPointLinkCandidate(const std::string& surfaceName,
                                                  size_t controlPointIndex,
                                                  cv::Vec3f volumePoint);
    void handleGeneratedControlPointLinkWithCandidate(const std::string& surfaceName,
                                                      size_t controlPointIndex,
                                                      cv::Vec3f volumePoint);
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
    [[nodiscard]] std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker>
        markLinkCandidateFiberIntersections(
            std::vector<vc3d::line_annotation::GeneratedOverlay::FiberIntersectionMarker> markers,
            const std::vector<FiberBranchRef>& branches) const;
    bool ensureDatasetForSession(LineAnnotationSession& session);
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
                                     bool fireSuccessCallback = true);
    void requestFinalizedClose(const std::string& surfaceName);
    void startOptimization(LineAnnotationSession& session,
                           bool fullOptimization = false,
                           int activeStart = -1,
                           int activeEnd = -1);
    void finishOptimization(const std::string& surfaceName);
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
    [[nodiscard]] std::optional<std::string> pickDataset(QWidget* parent,
                                                          const std::filesystem::path& startDir) const;
    [[nodiscard]] OptimizationTaskResult runOptimizationTask(std::filesystem::path manifestPath,
                                                             std::vector<vc::lasagna::LineControlPoint> controlPoints,
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
        const std::vector<vc::lasagna::LineControlPoint>* previousControlPoints = nullptr,
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
    void scheduleFiberSaveSnapshots(std::vector<FiberSaveSnapshot> snapshots);
    void canonicalizeFiberSaveSnapshots(std::vector<FiberSaveSnapshot>& snapshots) const;
    void validateFiberSaveSnapshots(const std::vector<FiberSaveSnapshot>& snapshots) const;
    void startNextFiberSaveJob();
    void finishFiberSaveJob(QFutureWatcher<FiberSaveTaskResult>* watcher);
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
    void showError(const QString& message) const;
    void cleanupIntersectionInspectionSurfaces();
    void rebuildIntersectionInspection();
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
    std::deque<FiberSaveJob> _pendingFiberSaveJobs;
    QPointer<QFutureWatcher<FiberSaveTaskResult>> _fiberSaveWatcher;
    uint64_t _nextFiberSaveSequence = 0;
    bool _fiberSaveRunning = false;
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

    // Transient (in-memory only) staging state for linking two existing control
    // points across fibers. Position is the primary key; the stored index is a
    // hint re-resolved at link time because indices are remapped on save.
    struct LinkCandidate {
        uint64_t fiberId = 0;
        std::string fiberFileName;
        cv::Vec3d position{0.0, 0.0, 0.0};
        int storedControlPointIndexHint = -1;
    };
    std::optional<LinkCandidate> _linkCandidate;
};
