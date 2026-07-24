#pragma once

// An in-process JSON-RPC 2.0 server that listens on a QLocalServer socket and
// lets an out-of-process agent drive and inspect VC3D.
//
// Everything here runs on the Qt GUI thread: QLocalSocket delivers readyRead on
// the main thread, so no extra thread is required and handlers may touch the UI
// directly. Handlers must never spin a nested event loop (no dialog exec()).

#include <QByteArray>
#include <QHash>
#include <QJsonObject>
#include <QJsonValue>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QStringList>

#include <deque>
#include <functional>
#include <optional>
#include <vector>

#include "MenuActionController.hpp"

#include "agent_bridge/AgentBridgeError.hpp"  // AgentBridgeError
#include "agent_bridge/AgentBridgeMethod.hpp"

class QLocalServer;
class QLocalSocket;
class QTimer;
class CWindow;
class VolumeViewerBase;
struct CommandLaunchError;

namespace vc3d::opendata {
struct OpenDataManifest;
}

// Thrown by a handler that has deferred its response: the handler
// has stashed (socket, id) via beginDeferred() and will write the response
// later from a signal/timeout. dispatch() catches this and sends nothing now.
struct AgentBridgeDeferred {};

class AgentBridgeServer : public QObject
{
    Q_OBJECT

public:
    static constexpr int kProtocolVersion = 1;

    explicit AgentBridgeServer(CWindow* window, QObject* parent = nullptr);
    ~AgentBridgeServer() override;

    // Starts listening on the given QLocalServer name. On an initial failure the
    // stale socket is removed once and listen retried. Returns true
    // on success. Does not print the handshake line — the caller does that so it
    // owns the exit-code policy.
    bool listen(const QString& serverName);

    QString serverName() const;
    QString fullServerName() const;

private slots:
    void onNewConnection();
    void onSocketReadyRead();
    void onSocketDisconnected();

private:
    using Handler = std::function<QJsonObject(const QJsonValue& params)>;

    void registerHandlers();
    void registerSessionHandlers();
    void registerCanvasHandlers();
    void registerEditingHandlers();
    void registerFlattenHandlers();
    void registerViewerHandlers();
    void registerWrapHandlers();
    void registerJobHandlers();
    void registerLasagnaHandlers();
    void registerAtlasHandlers();
    void registerFiberHandlers();
    void registerPointsHandlers();
    void registerSeedingHandlers();
    void registerSurfaceOpsHandlers();
    void registerMethod(AgentBridgeMethod method, Handler handler);

    // --- Discovery registry file (mirrors LasagnaServiceManager's
    // ~/.fit_services convention) ---
    // On a successful listen(), writes ~/.vc3d/agent_bridge/<pid>.json
    // ({pid, name, path, startedAt}) so an out-of-process MCP can auto-discover a
    // running bridge. Removed on clean shutdown; an orphan from a hard kill is
    // reaped by the reader's stale-PID check. Pure QFile/QDir I/O, never the UI.
    void writeRegistryFile();
    void removeRegistryFile();

    void processLine(QLocalSocket* socket, const QByteArray& line);
    void dispatch(QLocalSocket* socket, const QJsonObject& request);
    void sendResponse(QLocalSocket* socket, const QJsonValue& id, const QJsonObject& result);
    void sendError(QLocalSocket* socket, const QJsonValue& id, int code,
                   const QString& message, const QJsonObject& data = QJsonObject());
    void writeMessage(QLocalSocket* socket, const QJsonObject& message);
    // Server -> client notification (no "id") broadcast to every connected
    // client.
    void broadcastNotification(const QString& method, const QJsonObject& params);
    [[noreturn]] void throwCommandLaunchError(const CommandLaunchError& error,
                                              const QString& fallbackMessage,
                                              const QString& segmentId,
                                              const QString& source);

    // --- Viewer registry ---
    struct ViewerEntry {
        QString id;                 // stable "v<N>", never reused within a process
        VolumeViewerBase* viewer;   // live viewer; removed on baseViewerClosing
    };
    void seedViewerRegistry();
    void registerViewer(VolumeViewerBase* viewer);
    void unregisterViewer(VolumeViewerBase* viewer);
    QString viewerIdFor(VolumeViewerBase* viewer) const;
    QString viewerTitle(VolumeViewerBase* viewer) const;
    // Resolves a "viewer" param (id, slot name, or empty->defaultSlot). Throws
    // AgentBridgeError (-32002) on ambiguity or no match.
    VolumeViewerBase* resolveViewer(const QJsonValue& ref,
                                    const QString& defaultSlot = QStringLiteral("segmentation")) const;

    // --- Handlers ---
    QJsonObject handlePing(const QJsonValue& params);
    QJsonObject handleRpcDescribe(const QJsonValue& params);
    QJsonObject handleStateGet(const QJsonValue& params);
    QJsonObject handleSegmentsList(const QJsonValue& params);
    QJsonObject handleSegmentsActivate(const QJsonValue& params);
    QJsonObject handleSegmentsFetch(const QJsonValue& params);
    // Destructive on-disk operations use the same dialog-free cores as the UI.
    QJsonObject handleSegmentsDelete(const QJsonValue& params);
    QJsonObject handleSegmentsRename(const QJsonValue& params);
    // Review-state-aware segment listing with optional server-side tag filters
    // (AgentBridgeHandlers_review.cpp); reads meta["tags"] without forcing a
    // heavy surface load.
    QJsonObject handleSegmentsReview(const QJsonValue& params);
    QJsonObject handleScreenshotCapture(const QJsonValue& params);
    QJsonObject handleCursorVolumePoint(const QJsonValue& params);
    QJsonObject handleCanvasClick(const QJsonValue& params, bool addShift);
    QJsonObject handleCanvasDrag(const QJsonValue& params);
    QJsonObject handleViewerCenterOnPoint(const QJsonValue& params);
    QJsonObject handleViewerZoom(const QJsonValue& params);
    QJsonObject handleViewerRotate(const QJsonValue& params);
    QJsonObject handleViewerSetAxisAlignedSlices(const QJsonValue& params);
    // viewerRenderSettingsJson() builds the shared get/set reply body.
    QJsonObject handleViewerGetRenderSettings(const QJsonValue& params);
    QJsonObject handleViewerSetRenderSettings(const QJsonValue& params);
    QJsonObject viewerRenderSettingsJson() const;
    // Overlay-volume controls and per-viewer intersection sets.
    QJsonObject handleViewerGetOverlay(const QJsonValue& params);
    QJsonObject handleViewerSetOverlay(const QJsonValue& params);
    QJsonObject handleViewerListOverlayVolumes(const QJsonValue& params);
    QJsonObject handleViewerSetIntersects(const QJsonValue& params);
    QJsonObject handleSegmentationEnableEditing(const QJsonValue& params);
    QJsonObject handleSegmentationGrow(const QJsonValue& params);
    QJsonObject handleSegmentationGrowPatchFromSeed(const QJsonValue& params);
    // Force an explicit segment save/flush, reported as an "autosave"-source job.
    // Return idle when nothing is pending; otherwise return a running job
    // closed by handleAutosaveCompleted().
    QJsonObject handleSegmentationSave(const QJsonValue& params);
    // Manual-add (hole-fill) and corrections point authoring.
    QJsonObject handleManualAddBegin(const QJsonValue& params);
    QJsonObject handleManualAddFinish(const QJsonValue& params);
    QJsonObject handleManualAddSetLineMode(const QJsonValue& params);
    QJsonObject handleManualAddSetInterpolation(const QJsonValue& params);
    QJsonObject handleManualAddUndoConstraint(const QJsonValue& params);
    QJsonObject handleCorrectionsSetPointMode(const QJsonValue& params);
    // Same-winding wrap annotation (tutorial's shift+E). set_mode drives the
    // WrapAnnotationWidget checkbox; commit/undo mirror the shift+E / Ctrl+Z
    // key handlers over the chunked viewers.
    QJsonObject handleWrapAnnotationSetMode(const QJsonValue& params);
    QJsonObject handleWrapAnnotationCommit(const QJsonValue& params);
    QJsonObject handleWrapAnnotationUndo(const QJsonValue& params);
    QJsonObject handlePointsCommit(const QJsonValue& params);
    QJsonObject handlePointsList(const QJsonValue& params);
    // Full point-collection editing surface (points.*, AgentBridgeHandlers_points.cpp):
    // collection lifecycle, point mutation, attributes, winding fills, and IO.
    QJsonObject handlePointsAddCollection(const QJsonValue& params);
    QJsonObject handlePointsUpdatePoint(const QJsonValue& params);
    QJsonObject handlePointsRemovePoint(const QJsonValue& params);
    QJsonObject handlePointsClearCollection(const QJsonValue& params);
    QJsonObject handlePointsClearAll(const QJsonValue& params);
    QJsonObject handlePointsRenameCollection(const QJsonValue& params);
    QJsonObject handlePointsSetCollectionColor(const QJsonValue& params);
    QJsonObject handlePointsSetCollectionMetadata(const QJsonValue& params);
    QJsonObject handlePointsSetCollectionTag(const QJsonValue& params);
    QJsonObject handlePointsRemoveCollectionTag(const QJsonValue& params);
    QJsonObject handlePointsSetWindingsLinked(const QJsonValue& params);
    QJsonObject handlePointsAutoFillWindings(const QJsonValue& params);
    QJsonObject handlePointsSetAutoFillMode(const QJsonValue& params);
    QJsonObject handlePointsResetWindings(const QJsonValue& params);
    QJsonObject handlePointsApplyAnchorOffset(const QJsonValue& params);
    QJsonObject handlePointsSaveJson(const QJsonValue& params);
    QJsonObject handlePointsLoadJson(const QJsonValue& params);
    QJsonObject handlePointsSaveSegmentPath(const QJsonValue& params);
    QJsonObject handlePointsLoadSegmentPath(const QJsonValue& params);
    QJsonObject handleProjectCreate(const QJsonValue& params);
    QJsonObject handleVolumeOpen(const QJsonValue& params);
    QJsonObject handleVolumeAttach(const QJsonValue& params);
    QJsonObject handleVolumeSelect(const QJsonValue& params);
    QJsonObject handleVolumeList(const QJsonValue& params);
    QJsonObject handleCatalogOpenSample(const QJsonValue& params);
    // Remote catalog resource selection.
    QJsonObject handleCatalogListSamples(const QJsonValue& params);
    QJsonObject handleCatalogDescribeSample(const QJsonValue& params);
    QJsonObject handleJobStatus(const QJsonValue& params);
    // Resolves a running job by id or source, then invokes its cancel authority.
    QJsonObject handleJobCancel(const QJsonValue& params);
    // Lasagna RPCs + workspace switching.
    QJsonObject handleLasagnaServiceStatus(const QJsonValue& params);
    QJsonObject handleLasagnaEnsureService(const QJsonValue& params);
    QJsonObject handleLasagnaListDatasets(const QJsonValue& params);
    QJsonObject handleLasagnaStartOptimization(const QJsonValue& params);
    QJsonObject handleLasagnaJobs(const QJsonValue& params);
    QJsonObject handleLasagnaCancel(const QJsonValue& params);
    QJsonObject handleLasagnaSelectOutputSegment(const QJsonValue& params);
    QJsonObject handleLasagnaRepeatLast(const QJsonValue& params);
    QJsonObject handleWorkspaceSwitch(const QJsonValue& params);
    // Atlas RPCs.
    QJsonObject handleAtlasOpen(const QJsonValue& params);
    QJsonObject handleAtlasStatus(const QJsonValue& params);
    QJsonObject handleAtlasSearchStart(const QJsonValue& params);
    QJsonObject handleAtlasSearchCancel(const QJsonValue& params);
    QJsonObject handleAtlasSearchResults(const QJsonValue& params);
    QJsonObject handleAtlasOpenResult(const QJsonValue& params);
    QJsonObject handleAtlasRemap(const QJsonValue& params);
    QJsonObject handleAtlasOptimizeSnapCandidates(const QJsonValue& params);
    // Line-annotation / fiber RPCs. fiberController() requires an
    // open volume package (-32000) and a live LineAnnotationController
    // (-32010); requireKnownFiber throws -32007 kind:"fiber" for an unknown
    // id.
    class LineAnnotationController* fiberController() const;
    void requireKnownFiber(class LineAnnotationController* ctrl, quint64 fiberId) const;
    QJsonObject handleFiberLaunch(const QJsonValue& params);
    QJsonObject handleFiberList(const QJsonValue& params);
    QJsonObject handleFiberOpen(const QJsonValue& params);
    QJsonObject handleFiberSetFollow(const QJsonValue& params);
    QJsonObject handleFiberSave(const QJsonValue& params);
    QJsonObject handleFiberDelete(const QJsonValue& params);
    QJsonObject handleFiberSetTag(const QJsonValue& params);
    QJsonObject handleFiberCreateAtlas(const QJsonValue& params);
    QJsonObject handleFiberExport(const QJsonValue& params);
    QJsonObject handleFiberImport(const QJsonValue& params);
    // Tags, seeding, push/pull, and tracing.
    QJsonObject handleTagsSet(const QJsonValue& params);
    QJsonObject handleSeedingSetWindingAnnotationMode(const QJsonValue& params);
    QJsonObject handleSeedingPreviewRays(const QJsonValue& params);
    QJsonObject handleSeedingCastRays(const QJsonValue& params);
    QJsonObject handleSeedingResetPoints(const QJsonValue& params);
    // Batch seeding: run/expand spawn vc_grow_seg_from_seed child processes and
    // resolve through the seedingBatch* signals as source:"seeding" jobs.
    // analyze_paths is synchronous; cancel performs bounded teardown.
    QJsonObject handleSeedingRun(const QJsonValue& params);
    QJsonObject handleSeedingExpand(const QJsonValue& params);
    QJsonObject handleSeedingCancel(const QJsonValue& params);
    QJsonObject handleSeedingAnalyzePaths(const QJsonValue& params);
    // Shared body for run/expand: validate vpkg/volume/widget, requireSourceIdle,
    // invoke the headless launcher, map its typed failure to a bridge error,
    // and register the source:"seeding" job.
    QJsonObject launchSeedingBatch(
        const QString& kind, const QString& label,
        const std::function<bool(QString* err)>& launch);
    QJsonObject handlePushPullSetConfig(const QJsonValue& params);
    QJsonObject handlePushPullStart(const QJsonValue& params);
    QJsonObject handlePushPullStop(const QJsonValue& params);
    QJsonObject handleTracerRunTrace(const QJsonValue& params);
    QJsonObject handleRenderTifxyz(const QJsonValue& params);
    // Flattening RPCs: SLIM/flatboi, ABF++, straighten. Each drives a
    // headless SegmentationCommandHandler::start* launcher (dialogs suppressed)
    // and tracks the result as a source:"flatten" job.
    QJsonObject handleFlattenSlim(const QJsonValue& params);
    QJsonObject handleFlattenAbf(const QJsonValue& params);
    QJsonObject handleFlattenStraighten(const QJsonValue& params);
    // Shared body for the three flatten handlers.
    QJsonObject launchFlattenJob(
        const QString& kind, const QString& label, const QString& segmentId,
        const std::function<bool(CommandLaunchError* error,
                                 QString* outDir)>& launch);

    // Per-segment mesh operations live in AgentBridgeHandlers_surfaceops.cpp.
    // Sync: crop_bounds / recalc_area. Async source:"tool": reoptimize /
    // refine_alpha_comp (mirror render.tifxyz). In-process deferred:
    // generate_mask / append_mask (shared handleSegmentMask helper).
    QJsonObject handleSegmentCropBounds(const QJsonValue& params);
    QJsonObject handleSegmentRecalcArea(const QJsonValue& params);
    QJsonObject handleSegmentReoptimize(const QJsonValue& params);
    QJsonObject handleSegmentRefineAlphaComp(const QJsonValue& params);
    QJsonObject handleSegmentGenerateMask(const QJsonValue& params);
    QJsonObject handleSegmentAppendMask(const QJsonValue& params);
    QJsonObject handleSegmentMask(const QJsonValue& params, bool append);

    // --- Open Data manifest acquisition ---
    // Serves the cached manifest synchronously when available and `refresh` is
    // false; otherwise starts an async re-fetch with a 30-second deadline and
    // throws AgentBridgeDeferred, later invoking `build` to produce the result.
    // `build` may throw AgentBridgeError to surface a -3200x error.
    QJsonObject withOpenDataManifest(
        bool refresh,
        const std::function<QJsonObject(const vc3d::opendata::OpenDataManifest&)>& build);
    void startManifestFetch(
        int token,
        const std::function<QJsonObject(const vc3d::opendata::OpenDataManifest&)>& build);

    // Job tracking permits one active job per lifecycle source. Independent
    // sources may run concurrently; job ids share one monotonic counter.
    struct JobRecord {
        QString id;          // "job-<n>"
        int     num = 0;     // monotonic order key (parsed-free "most recent" sort)
        QString source;      // fixed lifecycle source reported over the wire
        QString kind;        // "segmentation.grow", "segmentation.grow_patch_from_seed", ...
        QString label;
        QString state;       // "running" | "succeeded" | "failed"
        QString message;
        QString outputPath;
        QString externalId;  // service job id when known, else empty (-> null)
        QStringList consoleTail;
        qint64 nextProgressSeq = 1;
        std::deque<QJsonObject> progressHistory;
        qint64 startedAtMs = 0;
        qint64 finishedAtMs = 0;  // 0 => null
        // Additive terminal result body. Empty => "result": null on
        // the wire; catalog jobs carry the catalog.open_sample result body.
        QJsonObject resultJson;
    };
    void subscribeJobSignals();
    // Per-source running check: true when a bridge JobRecord for `source` is
    // active, or the underlying lifecycle authority reports it busy.
    bool jobIsRunning(const QString& source) const;
    QString activeJobId(const QString& source) const;
    QString beginJob(const QString& source, const QString& kind, const QString& label,
                     bool broadcastStart);
    void finishJob(const QString& source, bool success, const QString& message,
                   const QString& outputPath);
    // Most recently started job (active preferred, else recent), optionally
    // filtered to a single source. Returns nullptr if none match.
    const JobRecord* mostRecentJob(const QString& sourceFilter = QString()) const;
    // Looks up any job (active or recent) by id across all sources.
    const JobRecord* jobById(const QString& jobId) const;
    void broadcastJobProgress(JobRecord& job, const QString& phase,
                              const QString& messageOverride = QString(),
                              std::optional<bool> success = std::nullopt);
    QJsonObject jobStatusJson(const JobRecord& job) const;
    // Throws -32004 (with data.source) when a job of `source` is already active.
    void requireSourceIdle(const QString& source) const;
    // Reactions to job-lifecycle signals (called from lambdas so the header
    // stays free of CommandLineToolRunner's enum).
    void handleToolStarted(const QString& message);
    void handleToolFinished(bool success, const QString& message, const QString& outputPath);
    void handleConsoleOutput(const QString& output);
    void handleGrowthStatusChanged(bool running);
    // Closes the "autosave"-source job when an explicit segment save
    // finishes. Wired from SegmentationModule::autosaveCompleted; a no-op unless a
    // bridge-initiated autosave job is active.
    void handleAutosaveCompleted(bool success);
    // Flattening lifecycle -> source:"flatten" JobRecord. Wired
    // from SegmentationCommandHandler::flattenJobStarted / flattenJobFinished.
    // Registers human-initiated flattens as external jobs too.
    void handleFlattenStarted(const QString& kind, const QString& label);
    void handleFlattenFinished(bool success, const QString& message,
                               const QString& outputPath);
    // Lasagna optimization lifecycle -> source:"lasagna" JobRecord. Wired from
    // LasagnaServiceManager signals in subscribeJobSignals().
    void handleLasagnaStarted();
    void handleLasagnaJobStarted(const QString& externalId);
    void handleLasagnaProgress(const QString& stageName, int step, int totalSteps,
                               double loss, double overallProgress);
    void handleLasagnaFinished(bool success, const QString& message,
                               const QString& outputPath);
    void handleLasagnaResultsPlaced(const QString& outputDir,
                                    const QStringList& segmentNames);
    // Atlas fiber-search lifecycle -> source:"atlas" JobRecord.
    // Only bridge-initiated searches are tracked as jobs: progress for
    // an untracked (human-initiated) search is deliberately NOT auto-registered,
    // because FinishResults progress also fires on pure UI re-population (the
    // group-by-fiber checkbox), which would leak a never-finishing job.
    void handleAtlasSearchProgress(int phase, double fraction);
    void handleAtlasSearchFinished(bool success, int resultCount);
    // Batch seeding lifecycle -> source:"seeding" JobRecord. Wired
    // from SeedingWidget::seedingBatchProgressChanged / seedingBatchFinished. Like
    // the atlas search, only bridge-initiated batches are tracked (progress/
    // finished for a human-initiated Run/Expand click are not auto-registered).
    void handleSeedingBatchProgress(const QString& kind, int completed, int total);
    void handleSeedingBatchFinished(const QString& kind, bool success, bool canceled,
                                    int completed, int total, const QString& message);
    // Resolves the active "catalog" job when an async catalog.open_sample
    // finishes: stores the result body in the job record and calls finishJob.
    void completeCatalogOpenJob(const MenuActionController::OpenDataSampleOpenOutcome& outcome);
    void completeVolumeAttachmentJob(const VolumeAttachmentOutcome& outcome);

    // --- Deferred responses ---
    // A handler calls beginDeferred() to stash the current (socket, id), arm a
    // timeout timer, and get a token; it then throws AgentBridgeDeferred{}. When
    // the awaited signal fires it calls completeDeferredResult/Error with the
    // token. Timeout fires -32005 automatically. At most one deferred call per
    // (connection, method) may be in flight.
    struct PendingDeferred {
        QPointer<QLocalSocket> socket;
        QJsonValue id;
        QString method;
        QString signalDesc;
        QTimer* timer = nullptr;
    };
    int beginDeferred(int timeoutMs, const QString& signalDesc);
    void completeDeferredResult(int token, const QJsonObject& result);
    void completeDeferredError(int token, int code, const QString& message,
                               const QJsonObject& data = QJsonObject());

    CWindow* _window = nullptr;
    QLocalServer* _server = nullptr;
    // Absolute path of the discovery registry file this process wrote (empty
    // until a successful listen()); removed in the destructor.
    QString _registryFilePath;
    QHash<QString, Handler> _handlers;
    QHash<QString, AgentBridgeMethod> _methodDescriptions;
    QHash<QLocalSocket*, QByteArray> _buffers;

    std::vector<ViewerEntry> _viewers;
    int _nextViewerNum = 1;

    // At most one active job per source, with the last eight completed records
    // retained per source. Job numbers are shared across sources.
    QHash<QString, JobRecord> _activeJobs;
    QHash<QString, std::deque<JobRecord>> _recentJobs;
    int _nextJobNum = 1;
    qint64 _lastConsoleBroadcastMs = 0;  // rate-limit for job.progress "output"
    // Sample id of the in-flight async catalog.open_sample; read by
    // completeCatalogOpenJob to build the result body. Single-threaded GUI, at
    // most one "catalog" job, so a scalar is race-free.
    QString _catalogOpenSampleId;

    // Deferred-response bookkeeping and the per-request context dispatch() sets
    // before invoking a handler (so a handler can stash the caller).
    QHash<int, PendingDeferred> _pendingDeferred;
    int _nextDeferredToken = 1;
    QLocalSocket* _currentSocket = nullptr;
    QJsonValue _currentRequestId;
    QString _currentMethod;
};
