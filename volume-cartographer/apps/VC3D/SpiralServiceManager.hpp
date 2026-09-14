#pragma once

#include "SpiralPclRole.hpp"
#include "SpiralServiceProfile.hpp"
#include "SpiralInputDraft.hpp"
#include "SpiralInputCopy.hpp"
#include <QFutureWatcher>
#include <QJsonArray>
#include <QTemporaryDir>
#include <memory>

#include <QJsonObject>
#include <QElapsedTimer>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QStringList>
#include <QTimer>

#include <array>
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
    static constexpr int kApiVersion = 33;

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

    // Create the first resident session. The service exposes dataset and
    // checkpoint discovery before this without importing the fit runtime.
    void initializeSession(QJsonObject request);
    // Replace an initialized session. This is also the only later verb that
    // may change the model domain or structural configuration.
    void rebuildSession(QJsonObject request);
    // Rebuild from the service's own launch defaults, ignoring any autosave.
    // This is how a service stuck in Error recovers.
    void rebuildWithDefaults();
    void runIterations(int iterations, const QJsonObject& influenceConfig,
                       const QJsonObject& runConfig);
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
    void applyInputDrafts(bool commit = false, const QStringList& selection = {});
    void refreshInputCatalog();
    QJsonArray inputDraftStatus() const;
    bool hasInputDrafts() const;
    bool ownsInputWorkspace() const { return _inputOwner; }
    void restoreInputDraft(const QString& id);
    void editInputDraft(const QString& id);
    void resolveInputConflict(const QJsonObject& conflict, const QString& action);
    void discardInputDraft(const QString& id);
    void discardInputWorkspace(std::function<void()> done);
    void releaseInputWorkspace(std::function<void()> done);
    void invalidateWorkingCopy(const QString& source);
    void workingCopyAsync(const QString& source, FetchPreviewFileCallback done);
    QString workingCopy(const QString& source, QString* error = nullptr);
    QString inputWorkspaceId() const { return _inputWorkspaceId; }
    void setInputSelection(const QStringList& ids) { _inputSelection = ids; _inputSelectionExplicit = true; }

    void stagePatch(const QString& directory, const QString& inputId);
    void stageJsonInput(const QString& kind, const QString& filePath,
                         const QString& inputId, const QString& role = {});
    // Stage a revision of an existing editable point collection.
    void stagePclReplacement(vc3d::spiral::PclRole role,
                              const QString& filePath,
                              const QString& inputId,
                              const QString& operation,
                              const QString& targetCollectionId);
    // Stage a deletion; Apply removes supervision and Commit persists it.
    void removeInputDraft(const QString& inputId);
    // Fetch a file intentionally omitted from the initial preview transfer.
    // Only files declared by the currently installed diagnostics artifact are
    // accepted by the cache.
    void fetchPreviewFile(const QString& relativeName,
                          FetchPreviewFileCallback done);

signals:
    void inputPreparationProgress(const QString& message);
    void inputCopyProgress(int activeCopies, const QString& message);
    void inputDraftsChanged();
    void inputWorkspaceReleased();
    void inputDraftStaged(const QString& alias);
    void inputDraftDiscarded(const QString& alias);
    void inputConflict(const QJsonObject& conflict);
    void inputEditorRequested(const QJsonObject& input, const QString& workingPath);
    void inputBatchFinished(const QString& error);
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
    // Immutable display-only PCL snapshot of one editable role resolved by
    // the service. The descriptor carries the source coordinate domain.
    void pclArtifactAvailable(vc3d::spiral::PclRole role,
                              const QString& manifestPath,
                              const QJsonObject& artifactRef);
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
    // On a CAS conflict, revision is the service's current revision and error
    // is non-empty so tracked-fiber clients can update their base and retry.
    void fiberRevisionUploadFinished(const QString& inputId,
                                     const QString& revision,
                                     const QString& error);
    void pclReplacementUploadFinished(const QString& inputId,
                                      const QString& currentRevision,
                                      const QString& error);
    void pclCommitConflict(const QString& currentRevision,
                           const QString& error);
    void commitInputsFinished(const QStringList& committedIds, const QString& error);
    void logMessage(const QString& message);
    void errorOccurred(const QString& message);

private:
    void copyInputAsync(const QString& source, FetchPreviewFileCallback done, bool reuseWorkingCopy);
    void fetchInputContent(const QString& id, quint64 revision, const QString& kind,
                           std::function<void(const QString&)> done);
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
    struct DraftTransfer {
        QString id, directory, uploadId;
        QJsonObject manifest;
    };
    struct DraftCommand {
        vc3d::spiral::InputDraftBatch batch;
        QVector<DraftTransfer> transfers;
        QJsonObject request;
        QJsonArray revisions;
        QStringList localDeletions;
        QString commitId;
        bool commit = false;
        bool applied = false;
        bool preparing = true;
        QString preparationError;
        std::shared_ptr<QTemporaryDir> directory;
    };
    void stageInput(const QString& kind, const QString& path, const QString& alias,
                    const QString& role = {}, const QString& targetCollection = {}, bool deleted = false);
    void resumeInputCommand();
    void transferInput(int index);
    void sendInputChanges();
    void finishInputChanges(const QJsonObject& response);
    void persistInputCommand();
    void failInputCommand(const QString& error, const QJsonObject& body = {});
    void finishInputCommand();
    void installInputCatalog(const QJsonArray& inputs);
    void claimInputWorkspace();
    void clearInputWorkspace();
    QString logicalInputId(const QString& kind, const QString& alias,
                           const QString& role, const QString& targetCollection);
    QMap<QString, std::shared_ptr<vc3d::spiral::InputDraft>> _inputDrafts;
    QMap<QString, QJsonObject> _inputCatalog;
    QStringList _inputOrder;
    mutable QJsonArray _inputRowsCache;
    mutable bool _inputRowsDirty = true;
    QMap<QString, QString> _inputAliases;
    QMap<QString, QString> _inputErrors;
    QMap<QString, QString> _workingCopies;
    QMap<QString, std::shared_ptr<QTemporaryDir>> _workingCopyDirectories;
    QMap<QString, QFutureWatcher<vc3d::spiral::InputCopyResult>*> _workingCopyJobs;
    quint64 _workingCopyGeneration = 0;
    void cancelWorkingCopies();
    void reportInputPreparation(const QString& message);
    QStringList _inputSelection;
    bool _inputSelectionExplicit = false;
    QString _inputWorkspaceId;
    bool _inputOwner = false;
    bool _inputCommandBusy = false;
    std::shared_ptr<DraftCommand> _inputCommand;
    std::function<void()> _afterInputCommand;
    vc3d::spiral::InputDraftSubmission _inputSubmission;
    QTemporaryDir _inputCopies;
    void sendRebuildRequest(QJsonObject request);
    void sendInitializeRequest(QJsonObject request);
    void prepareSessionRequest(QJsonObject request, bool initialize);
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
    // Per editable PCL role, indexed by vc3d::spiral::pclRoleIndex.
    std::array<QString, vc3d::spiral::kEditablePclRoles.size()> _installedPclArtifact;
    std::array<QString, vc3d::spiral::kEditablePclRoles.size()> _fetchingPclArtifact;
    bool _previewDiagnosticsWanted = false;
    QString _fetchingCheckpointArtifact;
    std::array<quint64, vc3d::spiral::kEditablePclRoles.size()> _pclSequence{};
    qint64 _previewSequence = 0;
    QString _lastPreviewLocalPath;
    QString _lastDiagnosticsLocalPath;
    std::array<QString, vc3d::spiral::kEditablePclRoles.size()> _lastPclLocalPath;
    QString _synchronizedSessionId;

    QStringList pclArtifactCachePins() const;
};
