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
    static constexpr int kApiVersion = 47;

    // The object store a service reads its fit inputs from, beside the one
    // attached in this VC3D. Inputs are displayed as fitted, and saved edits
    // reach the fit, only when both are the same store: the same UUID and
    // history generation. A service binds a store file on its host or a
    // served store's endpoint; this VC3D attaches a file or an endpoint.
    struct StoreBinding {
        QString serviceUuid;        // /bindings object_store.uuid
        QString serviceGeneration;  // its history generation
        QString serviceLocation;    // the path or endpoint the service opened
        bool serviceServed = false; // the service binds a vc_object_server endpoint
        qint64 serviceCommitSequence = -1;  // the service's store head when advertised
        QString servicePositionError;       // why it could not be read
        // The attachment is the service's own store file: a local service
        // on the same path.
        bool sameFile = false;
        QString localUuid;          // the attachment's store UUID
        QString localGeneration;
        QString localLocation;      // its path, or the endpoint it is attached through
        bool localHosted = false;   // attached through vc_object_server
        qint64 localCommitSequence = -1;  // the attachment's head at the last refresh
        bool advertised() const { return !serviceUuid.isEmpty(); }
        bool matches() const
        {
            return advertised() && serviceUuid == localUuid && serviceGeneration == localGeneration;
        }
        // The service's store is a file local to its host and this VC3D is
        // not attached to it: edits and commits need it served and attached.
        bool localToService() const { return advertised() && !serviceServed && !matches(); }
        // A client can name its saved edits (an input position) when it is
        // attached to the service's store.
        bool canUpdate() const { return matches(); }
        // Which stores are bound, without the heads, which every commit moves.
        QStringList identity() const
        {
            return {serviceUuid, serviceGeneration, serviceLocation, serviceServed ? QStringLiteral("served") : QString(),
                    localUuid, localGeneration, localLocation, matches() ? QStringLiteral("matches") : QString(),
                    sameFile ? QStringLiteral("same-file") : QString()};
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
    // (or a rebuild) at a position. Needs the service's store attached here
    // (storeBinding().canUpdate()).
    struct InputUpdate {
        qint64 requested = -1;  // M: the store's head (commit sequence) holding this VC3D's saved edits
        qint64 head = -1;       // H: the head the service linked at
        qint64 fitted = -1;     // N: H plus the service's links; the fit's inputs
        QString error;          // empty on success
        QStringList problems;   // the service's reasons
        // Commits the position includes beyond M by other writers, and
        // commits after N (excluded: they show as stale).
        QStringList includedBeyondRequested;
        QStringList afterFitted;
    };
    // Initialize or rebuild the fit from a position of the service's store
    // including this VC3D's saved edits (its head and the receipt of its
    // last save), or with an idle resident fit update its inputs in place
    // (POST /session/update-inputs: model and optimiser state are kept); a
    // failed session is rebuilt. ``request`` is the panel's session request.
    void updateFitInputs(QJsonObject request);
    // The request UUID of the last save this VC3D committed to the attached
    // store: an input update names it, so the service can verify its store
    // holds the edits (and is not a copy with the same identity).
    void setLastSaveRequest(const QString& requestUuid) { _lastSaveRequest = requestUuid; }
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
    // Re-read the attachment's status (its generation and head), e.g. after
    // a commit event.
    void refreshLocalStore();
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
    // (or a rebuild with current inputs).
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
        qint64 version = 0;        // the surface's object version in the service's store
        bool reused = false;       // the same fit was committed before
        QString error;             // state failed
        // Status "commit", or nullopt when it carries none.
        static std::optional<CommitResult> fromStatus(const QJsonObject& status);
    };
    // Whether the service can commit previews (capabilities.commits).
    bool commitsAvailable() const;
    // The preview artifact on display, which a commit names.
    QString displayedPreviewArtifact() const { return _installedPreviewArtifact; }
    bool commitActive() const { return !_commitArtifact.isEmpty(); }
    // Commit the preview artifact `artifact` (default: the one on display);
    // the outcome arrives through commitFinished.
    void commitPreview(const QString& artifact = {});

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
    // A store's head: the commit sequence its saves reached.
    static qint64 storeHead(vc::objects::ObjectStoreSession&, const vc::objects::AttachmentIdentity&);
    QString _sessionLeaseId;
    bool _leaseOwner = false;
    // Update fit inputs in flight.
    bool _inputUpdateActive = false;
    QString _lastSaveRequest;
    // The resident session's state and input generation, as last reported.
    QString _sessionState;
    qint64 _inputGeneration = 0;
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
    void finishCommit(const CommitResult& result);

    QStringList artifactCachePins() const;
};

Q_DECLARE_METATYPE(SpiralServiceManager::InputUpdate)
Q_DECLARE_METATYPE(SpiralServiceManager::CommitResult)
