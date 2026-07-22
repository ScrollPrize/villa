#pragma once

// VC3D Agent Bridge — Phase 1 server.
//
// An in-process JSON-RPC 2.0 server that listens on a QLocalServer socket and
// lets an out-of-process agent (via the Phase 3 MCP server) drive/inspect VC3D.
// See apps/VC3D/agent_bridge/SPEC.md for the binding wire contract.
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

class QLocalServer;
class QLocalSocket;
class QTimer;
class CWindow;
class VolumeViewerBase;

namespace vc3d::opendata {
struct OpenDataManifest;
}

// Thrown by a handler that has deferred its response (SPEC §8.4): the handler
// has stashed (socket, id) via beginDeferred() and will write the response
// later from a signal/timeout. dispatch() catches this and sends nothing now.
struct AgentBridgeDeferred {};

class AgentBridgeServer : public QObject
{
    Q_OBJECT

public:
    explicit AgentBridgeServer(CWindow* window, QObject* parent = nullptr);
    ~AgentBridgeServer() override;

    // Starts listening on the given QLocalServer name. On an initial failure the
    // stale socket is removed once and listen retried (SPEC §1.1). Returns true
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

    // --- Discovery registry file (mirrors LasagnaServiceManager's
    // ~/.fit_services convention) ---
    // On a successful listen(), a small JSON file named "<pid>.json" is written
    // under ~/.vc3d/agent_bridge/ containing {pid, name, path, startedAt} so an
    // out-of-process MCP server can auto-discover a running bridge without the
    // human relaying the stdout handshake line. Removed on clean shutdown (the
    // destructor); an orphaned file from a hard kill is reaped by the reader's
    // stale-PID check (dead pid -> file removed), exactly as discoverServices()
    // does. Pure QFile/QDir I/O -- never touches the UI.
    void writeRegistryFile();
    void removeRegistryFile();

    void processLine(QLocalSocket* socket, const QByteArray& line);
    void dispatch(QLocalSocket* socket, const QJsonObject& request);
    void sendResponse(QLocalSocket* socket, const QJsonValue& id, const QJsonObject& result);
    void sendError(QLocalSocket* socket, const QJsonValue& id, int code,
                   const QString& message, const QJsonObject& data = QJsonObject());
    void writeMessage(QLocalSocket* socket, const QJsonObject& message);
    // Server -> client notification (no "id") broadcast to every connected
    // client (SPEC §1.2).
    void broadcastNotification(const QString& method, const QJsonObject& params);

    // --- Viewer registry (SPEC §2.2) ---
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
    QJsonObject handleStateGet(const QJsonValue& params);
    QJsonObject handleSegmentsList(const QJsonValue& params);
    QJsonObject handleSegmentsActivate(const QJsonValue& params);
    QJsonObject handleSegmentsFetch(const QJsonValue& params);
    // Destructive on-disk segment ops (ADDITIONS_SPEC item 5). Both resolve the
    // id like handleSegmentsActivate and drive dialog-free cores extracted from
    // the interactive SurfacePanelController / SegmentationCommandHandler slots.
    QJsonObject handleSegmentsDelete(const QJsonValue& params);
    QJsonObject handleSegmentsRename(const QJsonValue& params);
    QJsonObject handleScreenshotCapture(const QJsonValue& params);
    QJsonObject handleCursorVolumePoint(const QJsonValue& params);
    // Phase 2: canvas + mutating action handlers.
    QJsonObject handleCanvasClick(const QJsonValue& params, bool addShift);
    QJsonObject handleCanvasDrag(const QJsonValue& params);
    QJsonObject handleViewerCenterOnPoint(const QJsonValue& params);
    QJsonObject handleViewerZoom(const QJsonValue& params);
    QJsonObject handleViewerRotate(const QJsonValue& params);
    QJsonObject handleViewerSetAxisAlignedSlices(const QJsonValue& params);
    // Global render-settings get/set (ADDITIONS_SPEC item 6). Reads current
    // values from ViewerManager getters and the first chunked viewer; the set
    // handler applies each present field via ViewerManager setters / broadcast
    // and echoes the resulting full settings. viewerRenderSettingsJson() builds
    // the shared reply body.
    QJsonObject handleViewerGetRenderSettings(const QJsonValue& params);
    QJsonObject handleViewerSetRenderSettings(const QJsonValue& params);
    QJsonObject viewerRenderSettingsJson() const;
    QJsonObject handleSegmentationEnableEditing(const QJsonValue& params);
    QJsonObject handleSegmentationGrow(const QJsonValue& params);
    QJsonObject handleSegmentationGrowPatchFromSeed(const QJsonValue& params);
    // Force an explicit segment save/flush, reported as a "autosave"-source job
    // (SPEC §3.11c). Idle response when nothing is pending; else a running job
    // closed by handleAutosaveCompleted().
    QJsonObject handleSegmentationSave(const QJsonValue& params);
    // Manual-add (hole-fill) + corrections point authoring (SPEC §9.2–9.7).
    QJsonObject handleManualAddBegin(const QJsonValue& params);
    QJsonObject handleManualAddFinish(const QJsonValue& params);
    QJsonObject handleManualAddSetLineMode(const QJsonValue& params);
    QJsonObject handleManualAddSetInterpolation(const QJsonValue& params);
    QJsonObject handleManualAddUndoConstraint(const QJsonValue& params);
    QJsonObject handleCorrectionsSetPointMode(const QJsonValue& params);
    // Same-winding wrap annotation (tutorial's shift+E). set_mode drives the
    // WrapAnnotationWidget checkbox; commit/undo mirror the shift+E / Ctrl+Z
    // key handlers over the chunked viewers (SPEC §3.9d).
    QJsonObject handleWrapAnnotationSetMode(const QJsonValue& params);
    QJsonObject handleWrapAnnotationCommit(const QJsonValue& params);
    QJsonObject handleWrapAnnotationUndo(const QJsonValue& params);
    QJsonObject handlePointsCommit(const QJsonValue& params);
    QJsonObject handlePointsList(const QJsonValue& params);
    QJsonObject handleVolumeOpen(const QJsonValue& params);
    QJsonObject handleVolumeSelect(const QJsonValue& params);
    // Lists every volume id in the open package (ADDITIONS_SPEC item 4).
    QJsonObject handleVolumeList(const QJsonValue& params);
    QJsonObject handleCatalogOpenSample(const QJsonValue& params);
    // Remote catalog resource selection (SPEC §10.1-10.2).
    QJsonObject handleCatalogListSamples(const QJsonValue& params);
    QJsonObject handleCatalogDescribeSample(const QJsonValue& params);
    QJsonObject handleJobStatus(const QJsonValue& params);
    // Generic cancel that resolves a running job (by id or source) and dispatches
    // to its per-source cancel authority (ADDITIONS_SPEC item 3).
    QJsonObject handleJobCancel(const QJsonValue& params);
    // Lasagna RPCs (SPEC §11) + workspace switching (SPEC §11.9).
    QJsonObject handleLasagnaServiceStatus(const QJsonValue& params);
    QJsonObject handleLasagnaEnsureService(const QJsonValue& params);
    QJsonObject handleLasagnaListDatasets(const QJsonValue& params);
    QJsonObject handleLasagnaStartOptimization(const QJsonValue& params);
    QJsonObject handleLasagnaJobs(const QJsonValue& params);
    QJsonObject handleLasagnaCancel(const QJsonValue& params);
    QJsonObject handleLasagnaSelectOutputSegment(const QJsonValue& params);
    QJsonObject handleLasagnaRepeatLast(const QJsonValue& params);
    QJsonObject handleWorkspaceSwitch(const QJsonValue& params);
    // Atlas RPCs (SPEC §12).
    QJsonObject handleAtlasOpen(const QJsonValue& params);
    QJsonObject handleAtlasStatus(const QJsonValue& params);
    QJsonObject handleAtlasSearchStart(const QJsonValue& params);
    QJsonObject handleAtlasSearchCancel(const QJsonValue& params);
    QJsonObject handleAtlasSearchResults(const QJsonValue& params);
    QJsonObject handleAtlasOpenResult(const QJsonValue& params);
    QJsonObject handleAtlasRemap(const QJsonValue& params);
    QJsonObject handleAtlasOptimizeSnapCandidates(const QJsonValue& params);
    // Line-annotation / fiber RPCs (SPEC §13). fiberController() requires an
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
    // Stage 6 backlog surface (SPEC §15): tags, seeding, push/pull, run-trace.
    QJsonObject handleTagsSet(const QJsonValue& params);
    QJsonObject handleSeedingSetWindingAnnotationMode(const QJsonValue& params);
    QJsonObject handleSeedingPreviewRays(const QJsonValue& params);
    QJsonObject handleSeedingCastRays(const QJsonValue& params);
    QJsonObject handleSeedingResetPoints(const QJsonValue& params);
    // Batch seeding: run/expand spawn vc_grow_seg_from_seed child processes and
    // resolve through the seedingBatch* signals as source:"seeding" jobs (SPEC
    // §15.2). analyze_paths is pure synchronous compute (not a job). cancel is a
    // bounded synchronous teardown.
    QJsonObject handleSeedingRun(const QJsonValue& params);
    QJsonObject handleSeedingExpand(const QJsonValue& params);
    QJsonObject handleSeedingCancel(const QJsonValue& params);
    QJsonObject handleSeedingAnalyzePaths(const QJsonValue& params);
    // Shared body for run/expand: validate vpkg/volume/widget, requireSourceIdle,
    // invoke the headless launcher, map its distinct failure sentences to codes,
    // and register the source:"seeding" job.
    QJsonObject launchSeedingBatch(
        const QString& kind, const QString& label,
        const std::function<bool(QString* err)>& launch);
    QJsonObject handlePushPullSetConfig(const QJsonValue& params);
    QJsonObject handlePushPullStart(const QJsonValue& params);
    QJsonObject handlePushPullStop(const QJsonValue& params);
    QJsonObject handleTracerRunTrace(const QJsonValue& params);
    QJsonObject handleRenderTifxyz(const QJsonValue& params);
    // Flattening RPCs (SPEC §20): SLIM/flatboi, ABF++, straighten. Each drives a
    // headless SegmentationCommandHandler::start* launcher (dialogs suppressed)
    // and tracks the result as a source:"flatten" job (§8.3).
    QJsonObject handleFlattenSlim(const QJsonValue& params);
    QJsonObject handleFlattenAbf(const QJsonValue& params);
    QJsonObject handleFlattenStraighten(const QJsonValue& params);
    // Shared body for the three flatten handlers: validated params + a callback
    // that invokes the specific start* launcher. Registers the "flatten" job,
    // maps the launcher's distinct failure sentences to JSON-RPC codes.
    QJsonObject launchFlattenJob(
        const QString& kind, const QString& label, const QString& segmentId,
        const std::function<bool(QString* err, QString* outDir)>& launch);

    // --- Open Data manifest acquisition (SPEC §10.1) ---
    // Serves the cached manifest synchronously when available and `refresh` is
    // false; otherwise starts an async re-fetch (SPEC §8.4 deferred, 30 s) and
    // throws AgentBridgeDeferred, later invoking `build` to produce the result.
    // `build` may throw AgentBridgeError to surface a -3200x error.
    QJsonObject withOpenDataManifest(
        bool refresh,
        const std::function<QJsonObject(const vc3d::opendata::OpenDataManifest&)>& build);
    void startManifestFetch(
        int token,
        const std::function<QJsonObject(const vc3d::opendata::OpenDataManifest&)>& build);

    // --- Job tracking (SPEC §2.4 as amended by §8.3) ---
    // The generalized model tracks at most one active job **per source**
    // ("tool" | "growth" | "lasagna" | "atlas"): a lasagna optimization and an
    // external tool can run concurrently, but two jobs of the same source
    // cannot. Job ids share one monotonic counter across all sources.
    struct JobRecord {
        QString id;          // "job-<n>"
        int     num = 0;     // monotonic order key (parsed-free "most recent" sort)
        QString source;      // "tool" | "growth" | "lasagna" | "atlas" | "catalog" | "seeding"
        QString kind;        // "segmentation.grow", "segmentation.grow_patch_from_seed", ...
        QString label;
        QString state;       // "running" | "succeeded" | "failed"
        QString message;
        QString outputPath;
        QString externalId;  // service job id when known, else empty (-> null)
        QStringList consoleTail;
        qint64 startedAtMs = 0;
        qint64 finishedAtMs = 0;  // 0 => null
        // Additive terminal result body (SPEC §18.4). Empty => "result": null on
        // the wire; carries the v1 catalog.open_sample body for "catalog" jobs.
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
    void broadcastJobProgress(const JobRecord& job, const QString& phase,
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
    // Closes the "autosave"-source job when an explicit segment save (§3.11c)
    // finishes. Wired from SegmentationModule::autosaveCompleted; a no-op unless a
    // bridge-initiated autosave job is active.
    void handleAutosaveCompleted(bool success);
    // Flattening lifecycle -> source:"flatten" JobRecord (SPEC §8.3, §20). Wired
    // from SegmentationCommandHandler::flattenJobStarted / flattenJobFinished.
    // Registers human-initiated flattens as external jobs too.
    void handleFlattenStarted(const QString& kind, const QString& label);
    void handleFlattenFinished(bool success, const QString& message,
                               const QString& outputPath);
    // Lasagna optimization lifecycle -> source:"lasagna" JobRecord (SPEC §8.3,
    // §11.4). Wired from LasagnaServiceManager signals in subscribeJobSignals().
    void handleLasagnaStarted();
    void handleLasagnaJobStarted(const QString& externalId);
    void handleLasagnaProgress(const QString& stageName, int step, int totalSteps,
                               double loss, double overallProgress);
    void handleLasagnaFinished(bool success, const QString& message,
                               const QString& outputPath);
    void handleLasagnaResultsPlaced(const QString& outputDir,
                                    const QStringList& segmentNames);
    // Atlas fiber-search lifecycle -> source:"atlas" JobRecord (SPEC §12.9).
    // Only bridge-initiated searches are tracked as jobs (§12.2): progress for
    // an untracked (human-initiated) search is deliberately NOT auto-registered,
    // because FinishResults progress also fires on pure UI re-population (the
    // group-by-fiber checkbox), which would leak a never-finishing job.
    void handleAtlasSearchProgress(int phase, double fraction);
    void handleAtlasSearchFinished(bool success, int resultCount);
    // Batch seeding lifecycle -> source:"seeding" JobRecord (SPEC §15.2). Wired
    // from SeedingWidget::seedingBatchProgressChanged / seedingBatchFinished. Like
    // the atlas search, only bridge-initiated batches are tracked (progress/
    // finished for a human-initiated Run/Expand click are not auto-registered).
    void handleSeedingBatchProgress(const QString& kind, int completed, int total);
    void handleSeedingBatchFinished(const QString& kind, bool success, bool canceled,
                                    int completed, int total, const QString& message);
    // Resolves the active "catalog" job when an async catalog.open_sample
    // finishes: builds the v1/§10.3 result body into the job record and calls
    // finishJob (SPEC §18.4). Kept out of the handler lambda.
    void completeCatalogOpenJob(const MenuActionController::OpenDataSampleOpenOutcome& outcome);

    // --- Deferred responses (SPEC §8.4) ---
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
    QHash<QLocalSocket*, QByteArray> _buffers;

    std::vector<ViewerEntry> _viewers;
    int _nextViewerNum = 1;

    // Active jobs keyed by source (0-4 entries) and per-source completed history
    // (last <=8 each, SPEC §8.3). One job-number counter across all sources.
    QHash<QString, JobRecord> _activeJobs;
    QHash<QString, std::deque<JobRecord>> _recentJobs;
    int _nextJobNum = 1;
    qint64 _lastConsoleBroadcastMs = 0;  // rate-limit for job.progress "output"
    // Sample id of the in-flight async catalog.open_sample (SPEC §18.4); read by
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
