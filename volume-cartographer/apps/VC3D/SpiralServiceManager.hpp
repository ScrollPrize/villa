#pragma once

#include "SpiralProjectInputs.hpp"
#include "SpiralServiceProfile.hpp"
#include <QSet>
#include <QJsonArray>
#include <memory>

#include <QJsonObject>
#include <QElapsedTimer>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QStringList>
#include <QTimer>

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>

namespace vc::objects {
class ObjectStoreSession;
class PushHold;
struct AttachmentIdentity;
}

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
    static constexpr int kApiVersion = 46;

    // The object store a service reads its fit inputs from, beside the one
    // attached in this VC3D. Inputs are displayed as fitted only when both
    // are the same store (the same UUID: a hub replica shares it). A service
    // bound to a hub replica fits hub positions; it matches only a replica
    // of the same hub (the hub identity), whose positions are comparable.
    // A local service on a plain store shares the file with this VC3D and
    // fits positions of the store itself.
    struct StoreBinding {
        QString serviceUuid;   // /bindings object_store.uuid
        QString servicePath;   // where the service host opened it
        QString serviceHub;    // the hub address on the service host
        QString serviceHubIdentity;
        qint64 servicePosition = -1;  // the service replica's last pull (a shared store's head)
        QString servicePositionError;  // why it could not be read
        // Whose positions the service fits: its hub's, or (a plain store
        // shared in place) the store's own; empty when it fits none.
        QString servicePositions;
        // The attachment is the service's own store file: a local service
        // on the same path, which saves reach without a hub.
        bool sameFile = false;
        QString localUuid;     // the attachment's store UUID
        QString localPath;
        QString localHub;      // the attachment's hub, when it is a replica
        QString localHubIdentity;
        qint64 localPosition = -1;    // the hub position the last sync reached
        int localPending = 0;         // saved edits not yet pushed
        int localConflicts = 0;
        bool advertised() const { return !serviceUuid.isEmpty(); }
        bool hosted() const { return !serviceHubIdentity.isEmpty(); }
        bool localHosted() const { return !localHubIdentity.isEmpty(); }
        bool matches() const
        {
            return advertised() && serviceUuid == localUuid
                && (!hosted() || serviceHubIdentity == localHubIdentity);
        }
        // The service fits positions; a client can name one (its saved
        // edits) when it replicates the service's hub or shares its file.
        bool positioned() const { return !servicePositions.isEmpty(); }
        bool shared() const { return positioned() && !hosted() && sameFile && matches(); }
        bool canUpdate() const { return matches() && (hosted() || shared()); }
        // Which stores are bound, without sync progress (positions, pending
        // edits, conflicts), which every hub sync updates.
        QStringList identity() const
        {
            return {serviceUuid, serviceHubIdentity, servicePositions, localUuid, localPath,
                    localHubIdentity, matches() ? QStringLiteral("matches") : QString(),
                    shared() ? QStringLiteral("shared") : QString()};
        }
    };

    explicit SpiralServiceManager(QObject* parent = nullptr);
    ~SpiralServiceManager() override;

    // spiral_service.py as a local launch would run it (SPIRAL_SERVICE_PATH,
    // a source checkout, or the installed copy); empty when none is found.
    static QString findService();

    void connectToService(const SpiralServiceProfile& profile);
    void disconnectFromService();
    void reconnect();

    void stopService();

    bool isReady() const { return _connectionState == ConnectionState::Ready; }
    bool hasActiveSession() const { return _hasActiveSession; }
    // GET /bindings: the service's host-owned output/cache roots, its
    // winding-inference binding, capabilities, store binding and checkpoints.
    QJsonObject advertisedBindings() const { return _advertisedBindings; }
    // Whether the service can run winding-model supervision (a usable
    // --winding-inference artifact is bound on its host), and if not, why.
    bool windingInferenceAvailable(QString* reason = nullptr) const;
    // The "project" block every initialize and rebuild carries
    // (SpiralProjectInputs), made from the project's Lasagna dataset that
    // `lasagna` gives: the service reads the project's facts and umbilicus
    // from it, not from the project file. Either throws when the project
    // cannot be described; the request is then not sent.
    void setProjectSource(
        std::function<std::optional<vc3d::spiral::ProjectLasagna>()> lasagna,
        std::function<QJsonObject(const std::optional<vc3d::spiral::ProjectLasagna>&)> block)
    {
        _projectLasagna = std::move(lasagna);
        _projectSource = std::move(block);
    }
    const SpiralServiceProfile& profile() const { return _profile; }
    bool ownsProcess() const;
    qint64 displayedPreviewSourceIteration() const
    {
        return _displayedPreviewSourceIteration;
    }
    QString displayedPreviewInputDigest() const { return _displayedPreviewInputDigest; }
    qint64 displayedPreviewInputPosition() const { return _displayedPreviewInputPosition; }

    // Create the first resident session. The service exposes its bindings and
    // checkpoint discovery before this without importing the fit runtime.
    void initializeSession(QJsonObject request);
    // Replace an initialized session. This is also the only later verb that
    // may change the model domain or structural configuration.
    // ``reloadInputs`` asks for the whole build, which rereads the service's
    // object store, even when the request changes nothing else.
    void rebuildSession(QJsonObject request, bool reloadInputs = false);
    // Rebuild from the service's own launch defaults, ignoring any autosave.
    // This is how a service stuck in Error recovers.
    void rebuildWithDefaults();

    // One "Update fit inputs": the resident fit's inputs replaced in place
    // (or a rebuild) at a position. Needs a service that fits positions and
    // either a replica of its hub or its own store file attached here
    // (storeBinding().canUpdate()).
    struct InputUpdate {
        qint64 requested = -1;  // M: where this store's saved (pushed) edits reached
        qint64 head = -1;       // H: the head the service linked at
        qint64 fitted = -1;     // N: H plus the service's links; the fit's inputs
        QString error;          // empty on success
        QStringList problems;   // the service's reasons, or the hub conflicts
        bool conflicts = false; // hub conflicts blocked the push
        // Commits the position includes beyond M by other writers, and
        // commits after N (excluded: they show as stale).
        QStringList includedBeyondRequested;
        QStringList afterFitted;
    };
    // Hold this replica's background pushes, push its saved edits and wait
    // (stopping on hub conflicts), then initialize or rebuild the fit from a
    // hub position including them; finally release the hold and pull to the
    // position the service fitted. ``request`` is the panel's session request.
    // With an idle resident fit this updates its inputs in place
    // (POST /session/update-inputs: model and optimiser state are kept);
    // without a session it initializes one, and a failed session is rebuilt.
    void updateFitInputs(QJsonObject request);
    bool inputUpdateActive() const { return _inputUpdateActive; }
    void runIterations(int iterations, const QJsonObject& runConfig,
                       const QJsonObject& dtLossSchedule,
                       const QJsonObject& previewSchedule = {});
    void stopAfterIteration();
    // Save on service: writes to a service-host path.
    void saveCheckpoint(const QString& name);
    // Download checkpoint: creates a checkpoint on the service, registers it
    // as an artifact, and streams it to a VC3D-local path.
    void downloadCheckpoint(const QString& localPath);
    // Every checkpoint the service says it can load: GET /bindings'
    // session_checkpoints (newest first).
    QStringList serviceCheckpoints() const;
    // The store this VC3D has attached. A local service is launched bound to
    // it (and relaunched when it changes); any service is compared with it.
    void setObjectStore(std::shared_ptr<vc::objects::ObjectStoreSession> session,
                        std::optional<vc::objects::AttachmentIdentity> attachment,
                        const QString& unavailableReason = {});
    const StoreBinding& storeBinding() const { return _storeBinding; }
    // Re-read the attachment's hub state (its sync position), e.g. after a
    // hub session event.
    void refreshLocalHub();
    // Where this machine reaches the service's hub: the profile's hub
    // address, else for an SSH profile the advertised path on its host
    // (sftp://destination/path), else the advertised address itself.
    QString serviceHubAddress() const;
    // The local service's own replica of the attachment's hub (under the
    // output root, by hub identity): client and service never share one.
    static QString localServiceReplica(const QString& outputRoot, const QString& hubIdentity);
    // Configuration keys the service's launch binding owns
    // (schema.fields[*].service_owned): requests never choose them.
    const QSet<QString>& serviceOwnedKeys() const { return _serviceOwnedKeys; }
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
    // How input changes reach a fit: saved in the store, then an input update
    // (or, without a hub, a rebuild with current inputs).
    static QString inputChangesMessage();
    // Whether this client holds the service's session lease, which every
    // mutating request needs.
    bool ownsSessionLease() const { return _leaseOwner; }
    void releaseSessionLease(std::function<void()> done);
    QString sessionLeaseId() const { return _sessionLeaseId; }

    // One commit of a published preview to the service's object store
    // (POST /session/commit), as status "commit" reports it.
    struct CommitResult {
        QString artifact;          // the preview artifact committed
        QString state;             // running, done or failed
        QString stage;             // while running
        QString surfaceUuid;
        QString geometryRevision;
        bool reused = false;       // the same fit was committed before
        // Whether the surface reached the service's hub; nullopt for a
        // service sharing its store in place (no hub).
        std::optional<bool> pushed;
        QString pushError;
        QString error;             // state failed
        // Status "commit", or nullopt when it carries none.
        static std::optional<CommitResult> fromStatus(const QJsonObject& status);
    };
    // Whether the service can commit previews (capabilities.commits).
    bool commitsAvailable() const;
    // The preview artifact on display, which a commit names.
    QString displayedPreviewArtifact() const { return _installedPreviewArtifact; }
    bool commitActive() const { return !_commitArtifact.isEmpty(); }
    // Commit the preview artifact `artifact` (default: the one on display),
    // or repeat a commit to retry its push; the outcome arrives through
    // commitFinished.
    void commitPreview(const QString& artifact = {});
    // Committed fits the service's hub replica holds unpushed (status
    // "pending_commits").
    QStringList pendingCommits() const { return _pendingCommits; }

    // Fetch a file intentionally omitted from the initial preview transfer.
    // Only files declared by the currently installed diagnostics artifact are
    // accepted by the cache.
    void fetchPreviewFile(const QString& relativeName,
                          FetchPreviewFileCallback done);

signals:
    void sessionLeaseReleased();
    void storeBindingChanged();
    void connectionStateChanged(SpiralServiceManager::ConnectionState state,
                                const QString& message);
    void bindingsAdvertised(const QJsonObject& bindings);
    void configurationCatalogChanged(const QJsonObject& catalog);
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
    // The current session's input manifest (object_store_inputs.json: the
    // digest and every object and relationship the fit froze, with revision
    // pins), downloaded from the service's input manifest artifact.
    void inputManifestAvailable(const QString& manifestPath, const QString& digest);
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
    void logMessage(const QString& message);
    void errorOccurred(const QString& message);
    // Update fit inputs: a phase ("pushing", "linking", "pulling") as it
    // starts, then the outcome.
    void inputUpdateProgress(const QString& phase, const QString& text);
    void inputUpdateFinished(const SpiralServiceManager::InputUpdate& update);
    void commitProgress(const QString& stage);
    void commitFinished(const SpiralServiceManager::CommitResult& result);

private:
    // Per-operation-class request timeouts: a single global timeout is wrong.
    enum class Timeout : int {
        Quick = 5000,          // health checks and status polls
        Command = 30000,       // run/stop and small mutations
        LongCommand = 240000,  // save-checkpoint blocks up to two minutes
        Load = 600000,         // session load tears down and validates inputs
    };

    // Failure callback that also receives the parsed error body, for
    // refusals whose structured fields the caller acts on rather than only
    // displays. When set it replaces the plain failure callback.
    using DetailedFailure = std::function<void(const QString& message,
                                               const QJsonObject& body)>;

    QString findPython() const;
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
    void fetchAdvertisedBindings();
    QString commandId();
    QString endpointFingerprint() const;
    void claimSessionLease();
    void clearSessionLease();
    void updateStoreBinding();
    void updateSameFile();
    // A store's journal head: the position a shared store's saves reached.
    static qint64 storeHead(vc::objects::ObjectStoreSession&, const vc::objects::AttachmentIdentity&);
    QString _sessionLeaseId;
    bool _leaseOwner = false;
    // Update fit inputs in flight: its push hold, released when it finishes.
    bool _inputUpdateActive = false;
    // The resident session's state and input generation, as last reported.
    QString _sessionState;
    qint64 _inputGeneration = 0;
    qint64 _forwardSyncPosition = -1;  // the fitted position a sync was last asked to reach
    std::shared_ptr<vc::objects::PushHold> _inputUpdateHold;
    // Each update's callbacks carry its serial; abandoning one (the
    // connection changed under it) bumps the serial, so the late callbacks
    // of the abandoned update are ignored.
    quint64 _inputUpdateSerial = 0;
    void finishInputUpdate(InputUpdate update, quint64 serial);
    void sendInputUpdate(QJsonObject request, qint64 requested, quint64 serial);
    // The attachment an update holds pushes on: its hold ends with it.
    std::unique_ptr<vc::objects::AttachmentIdentity> _inputUpdateAttachment;
    void abandonInputUpdate(const QString& reason);
    void sendRebuildRequest(QJsonObject request);
    void sendInitializeRequest(QJsonObject request);
    // Adds the project block to `body`; the error when it cannot be read.
    // Add the project block to `body`, then `send` it. A service on this host
    // that builds missing Lasagna packs is given the project's remote arrays
    // as their read-through cache directories, filled first when a pack it
    // lacks needs them; `failed` says why the request was not sent.
    void withProject(QJsonObject body, std::function<void(QJsonObject)> send,
                     std::function<void(const QString&)> failed);
    // Cache `groups` (remote Lasagna arrays `lasagna` packs from), then
    // finish(body, cached); fails through `failed`.
    void cacheLasagnaThenSend(
        QJsonObject body, const vc3d::spiral::ProjectLasagna& lasagna,
        const vc3d::spiral::ProjectLasagna& cached,
        std::vector<vc::lasagna::LasagnaChannelGroup> groups,
        std::function<void(QJsonObject, const std::optional<vc3d::spiral::ProjectLasagna>&)> finish,
        std::function<void(const QString&)> failed);
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
    QStringList _ownedLaunchBinding;    // --output/--cache/--object-store/--winding-inference of _process
    bool _waitingForLocalStore = false;
    // Set while stopService() stops _process: its exit is then the one asked
    // for (a disconnect, restart or profile switch), not a failure.
    bool _stoppingProcess = false;
    std::weak_ptr<vc::objects::ObjectStoreSession> _storeSession;
    std::unique_ptr<vc::objects::AttachmentIdentity> _storeAttachment;
    QString _localStorePath;
    QString _localStoreUnavailableReason;
    StoreBinding _storeBinding;
    QSet<QString> _serviceOwnedKeys;
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
    QJsonObject _advertisedBindings;
    std::function<std::optional<vc3d::spiral::ProjectLasagna>()> _projectLasagna;
    std::function<QJsonObject(const std::optional<vc3d::spiral::ProjectLasagna>&)> _projectSource;
    // Set to stop the Lasagna arrays being cached for a request; one at a time.
    std::shared_ptr<std::atomic<bool>> _lasagnaCacheCancel;
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
    qint64 _displayedPreviewSourceIteration = -1;
    QString _displayedPreviewInputDigest;
    qint64 _displayedPreviewInputPosition = -1;
    QString _fetchingPreviewArtifact;
    QString _installedDiagnosticsArtifact;
    QString _fetchingDiagnosticsArtifact;
    QString _installedInputManifest;
    QString _fetchingInputManifest;
    bool _previewDiagnosticsWanted = false;
    QString _fetchingCheckpointArtifact;
    qint64 _previewSequence = 0;
    QString _lastPreviewLocalPath;
    QString _lastDiagnosticsLocalPath;
    QString _lastInputManifestLocalPath;
    QString _synchronizedSessionId;
    // The artifact a commit in flight names, and the status generation its
    // request was answered at: only later statuses report its outcome.
    QString _commitArtifact;
    qint64 _commitStatusGeneration = -1;
    QStringList _pendingCommits;
    void finishCommit(const CommitResult& result);

    QStringList artifactCachePins() const;
};

Q_DECLARE_METATYPE(SpiralServiceManager::InputUpdate)
Q_DECLARE_METATYPE(SpiralServiceManager::CommitResult)
