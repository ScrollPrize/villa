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

class QLocalServer;
class QLocalSocket;
class CWindow;
class VolumeViewerBase;

// Thrown by handlers to produce a JSON-RPC error response. Codes follow SPEC §2.5.
struct AgentBridgeError {
    int code;
    QString message;
    QJsonObject data;
};

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
    QJsonObject handleScreenshotCapture(const QJsonValue& params);
    QJsonObject handleCursorVolumePoint(const QJsonValue& params);
    // Phase 2: canvas + mutating action handlers.
    QJsonObject handleCanvasClick(const QJsonValue& params, bool addShift);
    QJsonObject handleViewerCenterOnPoint(const QJsonValue& params);
    QJsonObject handleViewerZoom(const QJsonValue& params);
    QJsonObject handleSegmentationEnableEditing(const QJsonValue& params);
    QJsonObject handleSegmentationGrow(const QJsonValue& params);
    QJsonObject handleSegmentationGrowPatchFromSeed(const QJsonValue& params);
    QJsonObject handlePointsCommit(const QJsonValue& params);
    QJsonObject handlePointsList(const QJsonValue& params);
    QJsonObject handleVolumeOpen(const QJsonValue& params);
    QJsonObject handleCatalogOpenSample(const QJsonValue& params);
    QJsonObject handleJobStatus(const QJsonValue& params);

    // --- Job tracking (SPEC §2.4, §3.17-3.18) ---
    // The bridge tracks at most one active job, matching reality (a single
    // QProcess in CommandLineToolRunner and a single SegmentationGrower flag).
    struct JobRecord {
        QString id;          // "job-<n>"
        QString kind;        // "segmentation.grow", "segmentation.grow_patch_from_seed", "tool"
        QString label;
        QString state;       // "running" | "succeeded" | "failed"
        QString message;
        QString outputPath;
        QStringList consoleTail;
    };
    void subscribeJobSignals();
    bool jobIsRunning() const;
    QString activeJobId() const;
    QString beginJob(const QString& kind, const QString& label, bool broadcastStart);
    void finishActiveJob(bool success, const QString& message, const QString& outputPath);
    void broadcastJobProgress(const JobRecord& job, const QString& phase,
                              const QString& messageOverride = QString(),
                              std::optional<bool> success = std::nullopt);
    QJsonObject jobStatusJson(const JobRecord& job) const;
    // Reactions to job-lifecycle signals (called from lambdas so the header
    // stays free of CommandLineToolRunner's enum).
    void handleToolStarted(const QString& message);
    void handleToolFinished(bool success, const QString& message, const QString& outputPath);
    void handleConsoleOutput(const QString& output);
    void handleGrowthStatusChanged(bool running);

    CWindow* _window = nullptr;
    QLocalServer* _server = nullptr;
    QHash<QString, Handler> _handlers;
    QHash<QLocalSocket*, QByteArray> _buffers;

    std::vector<ViewerEntry> _viewers;
    int _nextViewerNum = 1;

    std::optional<JobRecord> _activeJob;
    std::deque<JobRecord> _recentJobs;   // last <=8 completed jobs for late polling
    int _nextJobNum = 1;
    qint64 _lastConsoleBroadcastMs = 0;  // rate-limit for job.progress "output"
};
