#pragma once

#include "SpiralServiceProfile.hpp"

#include <QJsonObject>
#include <QElapsedTimer>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QStringList>
#include <QTimer>

#include <functional>

class QNetworkAccessManager;
class QNetworkReply;
class QNetworkRequest;
class SpiralArtifactCache;
class SpiralSshTunnel;

// One connection state machine for every Spiral service. A local service is a
// service reached through a loopback URL; VC3D may optionally launch and own
// that process, but local and remote connections share the same
// authentication, status, and artifact-transfer code.
//
//   Disconnected -> Starting (optional) -> Connecting -> Ready
//         ^                                          |
//         +------------- Reconnecting <--------------+
class SpiralServiceManager : public QObject
{
    Q_OBJECT
public:
    using FetchPreviewFileCallback =
        std::function<void(const QString& localPath, const QString& error)>;

    enum class ConnectionState { Disconnected, Starting, Connecting, Ready,
                                 Reconnecting, Failed };
    Q_ENUM(ConnectionState)

    // The one service API version this build speaks; the handshake refuses
    // anything else. Reported to the user so a mismatch is self-explanatory.
    static constexpr int kApiVersion = 28;

    explicit SpiralServiceManager(QObject* parent = nullptr);
    ~SpiralServiceManager() override;

    void connectToService(const SpiralServiceProfile& profile);
    void disconnectFromService();
    void reconnect();

    // Convenience for the built-in local profile (compatibility with callers
    // that only ever used the auto-launched loopback service).
    void ensureStarted();
    void stopService();

    ConnectionState connectionState() const { return _connectionState; }
    bool isReady() const { return _connectionState == ConnectionState::Ready; }
    bool hasActiveSession() const { return _hasActiveSession; }
    QJsonObject advertisedDataset() const { return _advertisedDataset; }
    const SpiralServiceProfile& profile() const { return _profile; }
    bool ownsProcess() const;

    // Rebuild the always-loaded session: the service holds one from
    // startup, so this replaces it rather than creating it. It is also the
    // only verb that may change the model domain or structural config.
    void rebuildSession(QJsonObject request);
    // Rebuild from the service's own launch defaults, ignoring any autosave.
    // This is how a service stuck in Error recovers.
    void rebuildWithDefaults();
    void runIterations(int iterations, const QJsonObject& influenceConfig,
                       const QJsonObject& runConfig,
                       const QJsonObject& inputs = {});
    void stopAfterIteration();
    // Save on service: writes to a service-host path.
    void saveCheckpoint(const QString& name);
    // Download checkpoint: creates a checkpoint on the service, registers it
    // as an artifact, and streams it to a VC3D-local path.
    void downloadCheckpoint(const QString& localPath);
    // Every checkpoint the service says it can load: GET /dataset's
    // session_checkpoints (newest first) followed by detected_checkpoints.
    QStringList serviceCheckpoints() const;
    // Load a checkpoint into the fit. Exactly one of hostPath (a checkpoint
    // the service advertised) and localPath (a file on this machine, uploaded
    // first) is set; the service, which owns the filesystem, resolves it
    // either way. Without allowRebuild the service refuses anything that is
    // not an exact match for the live model, reporting the refusal through
    // checkpointLoadRefused with the rebuild that would accept it; passing
    // allowRebuild has the service perform that rebuild.
    void loadCheckpoint(const QString& hostPath, const QString& localPath,
                        bool allowRebuild = false);
    // Ask the session to export and publish one preview generation. Previews
    // are no longer a side effect of pausing or of resuming a checkpoint, so
    // this is what keeps VC3D's "see the fit after it stops" behaviour.
    void requestPreview();
    // Whether preview exports should also compute the loss overlays. They
    // roughly double the cost of a preview and arrive as a second artifact
    // after the surface, so this follows what the panel is displaying rather
    // than being on by default.
    void setPreviewDiagnostics(bool enabled) { _previewDiagnosticsWanted = enabled; }
    void commitInputs();
    void uploadPatch(const QString& directory, const QString& inputId);
    void uploadJsonInput(const QString& kind, const QString& filePath,
                         const QString& inputId, const QString& role = {});
    // Remove an added input that has not joined the resident fit yet.
    void removeEphemeralInput(const QString& kind, const QString& inputId);
    // Fetch a file intentionally omitted from the initial preview transfer.
    // Only files declared by the currently installed diagnostics artifact are
    // accepted by the cache.
    void fetchPreviewFile(const QString& relativeName,
                          FetchPreviewFileCallback done);

signals:
    void connectionStateChanged(SpiralServiceManager::ConnectionState state,
                                const QString& message);
    void serviceStateChanged(const QString& state);
    void datasetResolved(const QJsonObject& resolution);
    void configurationCatalogChanged(const QJsonObject& catalog);
    void configurationReviewRequested();
    // Emitted once when this connection first observes a resident session,
    // whether VC3D loaded it or attached after another client did.
    void sessionSynchronized(const QJsonObject& sessionRequest,
                             const QJsonObject& status);
    void sessionStatusChanged(const QJsonObject& status);
    void sessionActiveChanged(bool active);
    // Local (cache) filesystem paths: artifact transfers already happened.
    void previewAvailable(const QString& manifestPath, qint64 generation);
    // The loss overlays for an already-installed preview, published by the
    // service as a second artifact once the surface was on its way.
    void previewDiagnosticsAvailable(const QString& manifestPath,
                                     qint64 generation);
    void previewTransferProgress(const QString& phase, const QString& fileName,
                                 int filesComplete, int totalFiles,
                                 qint64 bytesReceived, qint64 totalBytes);
    void checkpointDownloadProgress(const QString& phase,
                                    qint64 bytesReceived, qint64 totalBytes);
    void checkpointDownloadFinished(const QString& localPath, const QString& error);
    void checkpointUploadProgress(qint64 sentBytes, qint64 totalBytes);
    // A checkpoint was loaded into the live session at the given iteration.
    void checkpointLoaded(const QString& hostPath, qint64 restoredIteration);
    // The service refused a checkpoint. ``stage`` is the rebuild that would
    // accept it ("model" keeps the loaded inputs, "all" replaces everything);
    // it is empty when no rebuild would help, which the service reports and
    // the client must not offer to escalate.
    void checkpointLoadRefused(const QString& hostPath, const QString& localPath,
                               const QStringList& reasons, const QString& stage,
                               const QString& message);
    void inputUploadFinished(const QString& inputId, const QString& error);
    void commitInputsFinished(const QStringList& committedIds, const QString& error);
    void logMessage(const QString& message);
    void errorOccurred(const QString& message);

private:
    // Per-operation-class request timeouts: a single global timeout is wrong.
    enum class Timeout : int {
        Quick = 5000,          // health checks and status polls
        Command = 30000,       // run/stop and small mutations
        LongCommand = 240000,  // save-checkpoint blocks up to two minutes
        Load = 600000,         // session load tears down and validates datasets
    };

    // Failure callback that also receives the parsed error body, for
    // refusals whose structured fields the caller acts on rather than only
    // displays. When set it replaces the plain failure callback.
    using DetailedFailure = std::function<void(const QString& message,
                                               const QJsonObject& body)>;

    QString findPython() const;
    QString findService() const;
    void setConnectionState(ConnectionState state, const QString& message = {});
    void startLocalProcess();
    void startTunnel();
    void beginHandshake();
    void handleHealth(const QJsonObject& health);
    QNetworkRequest makeRequest(const QString& path, int timeoutMs) const;
    void post(const QString& path, QJsonObject body, Timeout timeout,
              std::function<void(const QJsonObject&)> success = {},
              std::function<void(const QString&)> failure = {});
    void postWithRetry(const QString& path, QJsonObject body, Timeout timeout,
                       int retriesLeft,
                       std::function<void(const QJsonObject&)> success,
                       std::function<void(const QString&)> failure = {},
                       DetailedFailure detailedFailure = {});
    void get(const QString& path, Timeout timeout,
             std::function<void(const QJsonObject&)> success,
             std::function<void(const QString&)> failure = {});
    void del(const QString& path, Timeout timeout,
             std::function<void(const QJsonObject&)> success = {},
             std::function<void(const QString&)> failure = {});
    void handleReply(QNetworkReply* reply, quint64 generation,
                     std::function<void(const QJsonObject&)> success,
                     std::function<void(const QString&)> failure,
                     DetailedFailure detailedFailure = {});
    void pollStatus();
    // One structured event subscriber for every connection: GET /events with
    // a persisted cursor; the panel interleaves all record kinds and popups
    // are reserved for error severity.
    void pollEvents();
    void handleStatus(const QJsonObject& status);
    void syncArtifacts(const QJsonObject& status);
    void fetchAdvertisedDataset();
    QString commandId();
    QString endpointFingerprint() const;
    void continueUpload(const QString& uploadId, const QString& inputId,
                        const QString& baseDir, QStringList pendingFiles);
    void sendRebuildRequest(QJsonObject request);
    void sendLoadCheckpoint(QJsonObject body, const QString& hostPath,
                            const QString& localPath);
    // Streams a client-local resume checkpoint into the service's
    // uploaded-checkpoints directory and reports the resulting host path.
    void uploadCheckpointForResume(const QString& localPath,
                                   std::function<void(const QString& hostPath,
                                                      const QString& error,
                                                      bool reused)> done);

    SpiralServiceProfile _profile;
    QProcess* _process = nullptr;       // owned local service process, if any
    QStringList _ownedLaunchBinding;    // --dataset/--output/--cache of _process
    QNetworkAccessManager* _network = nullptr;
    SpiralSshTunnel* _tunnel = nullptr;
    SpiralArtifactCache* _artifactCache = nullptr;
    QTimer* _poll = nullptr;
    QTimer* _eventPoll = nullptr;
    QUrl _baseUrl;
    QString _credential;
    QString _clientId;
    ConnectionState _connectionState = ConnectionState::Disconnected;
    quint64 _connectionGeneration = 0;  // stale replies are ignored
    bool _statusInFlight = false;
    int _statusFailures = 0;
    bool _hasActiveSession = false;
    bool _eventsInFlight = false;
    int _eventFailures = 0;
    qint64 _lastEventCursor = 0;
    QJsonObject _advertisedDataset;
    QJsonObject _configurationDefaults;
    QJsonObject _appliedConfiguration;
    qint64 _sessionRevision = 0;
    quint64 _commandCounter = 0;
    qint64 _lastStatusGeneration = -1;
    // True once a run has been observed; the following Idle is the pause the
    // panel wants a preview of.
    bool _sawRunningSinceIdle = false;
    bool _previewRequestInFlight = false;
    QString _installedPreviewArtifact;
    QString _installedPreviewSession;
    QString _fetchingPreviewArtifact;
    QString _installedDiagnosticsArtifact;
    QString _fetchingDiagnosticsArtifact;
    bool _previewDiagnosticsWanted = false;
    QString _fetchingCheckpointArtifact;
    qint64 _previewSequence = 0;
    QString _lastPreviewLocalPath;
    QString _lastDiagnosticsLocalPath;
    QString _synchronizedSessionId;
};
