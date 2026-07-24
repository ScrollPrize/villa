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
    enum class ConnectionState { Disconnected, Starting, Connecting, Ready,
                                 Reconnecting, Failed };
    Q_ENUM(ConnectionState)

    explicit SpiralServiceManager(QObject* parent = nullptr);
    ~SpiralServiceManager() override;

    void connectToService(const SpiralServiceProfile& profile);
    void disconnectFromService();
    void reconnect();
    void restartRemoteService();

    // Convenience for the built-in local profile (compatibility with callers
    // that only ever used the auto-launched loopback service).
    void ensureStarted();
    void stopService();

    ConnectionState connectionState() const { return _connectionState; }
    bool isReady() const { return _connectionState == ConnectionState::Ready; }
    bool hasActiveSession() const { return _hasActiveSession; }
    bool serviceOwnsDataset() const { return _serviceOwnsDataset; }
    QJsonObject advertisedDataset() const { return _advertisedDataset; }
    const SpiralServiceProfile& profile() const { return _profile; }
    bool ownsProcess() const;

    void resolveDataset(const QString& root);
    void loadSession(QJsonObject request);
    void runIterations(int iterations, const QJsonObject& influenceConfig,
                       const QJsonObject& runConfig);
    void stopAfterIteration();
    // Save on service: writes to a service-host path.
    void saveCheckpoint(const QString& path);
    // Download checkpoint: creates a checkpoint on the service, registers it
    // as an artifact, and streams it to a VC3D-local path.
    void downloadCheckpoint(const QString& localPath);
    void deleteSession();
    void commitInputs();
    void uploadPatch(const QString& directory, const QString& inputId);
    void uploadJsonInput(const QString& kind, const QString& filePath,
                         const QString& inputId, const QString& role = {});
    // Remove an added input that has not joined the resident fit yet.
    void removeEphemeralInput(const QString& kind, const QString& inputId);

signals:
    void connectionStateChanged(SpiralServiceManager::ConnectionState state,
                                const QString& message);
    void serviceStateChanged(const QString& state);
    void datasetResolved(const QJsonObject& resolution);
    void sessionAccepted(const QJsonObject& inputPaths, qint64 sessionGeneration);
    void sessionStatusChanged(const QJsonObject& status);
    void sessionActiveChanged(bool active);
    // Local (cache) filesystem paths: artifact transfers already happened.
    void previewAvailable(const QString& manifestPath, qint64 generation);
    void geometryAvailable(const QString& manifestPath, quint64 generation);
    void checkpointDownloadFinished(const QString& localPath, const QString& error);
    void checkpointUploadProgress(qint64 sentBytes, qint64 totalBytes);
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

    QString findPython() const;
    QString findService() const;
    void setConnectionState(ConnectionState state, const QString& message = {});
    void startLocalProcess();
    void startTunnel();
    void beginHandshake();
    void probeRestartedService();
    void handleHealth(const QJsonObject& health);
    QNetworkRequest makeRequest(const QString& path, int timeoutMs) const;
    void post(const QString& path, QJsonObject body, Timeout timeout,
              std::function<void(const QJsonObject&)> success = {},
              std::function<void(const QString&)> failure = {});
    void postWithRetry(const QString& path, QJsonObject body, Timeout timeout,
                       int retriesLeft,
                       std::function<void(const QJsonObject&)> success,
                       std::function<void(const QString&)> failure = {});
    void get(const QString& path, Timeout timeout,
             std::function<void(const QJsonObject&)> success,
             std::function<void(const QString&)> failure = {});
    void handleReply(QNetworkReply* reply, quint64 generation,
                     std::function<void(const QJsonObject&)> success,
                     std::function<void(const QString&)> failure);
    void pollStatus();
    void pollRemoteLogs();
    void handleStatus(const QJsonObject& status);
    void syncArtifacts(const QJsonObject& status);
    void fetchAdvertisedDataset();
    QString commandId();
    QString endpointFingerprint() const;
    QJsonObject remoteInputPaths() const;
    void continueUpload(const QString& uploadId, const QString& inputId,
                        const QString& baseDir, QStringList pendingFiles);
    void sendLoadRequest(QJsonObject request, const QJsonObject& inputPaths);
    // Streams a client-local resume checkpoint into the service's
    // uploaded-checkpoints directory and reports the resulting host path.
    void uploadCheckpointForResume(const QString& localPath,
                                   std::function<void(const QString& hostPath,
                                                      const QString& error)> done);

    SpiralServiceProfile _profile;
    QProcess* _process = nullptr;       // owned local service process, if any
    QNetworkAccessManager* _network = nullptr;
    SpiralSshTunnel* _tunnel = nullptr;
    SpiralArtifactCache* _artifactCache = nullptr;
    QTimer* _poll = nullptr;
    QTimer* _remoteLogPoll = nullptr;
    QUrl _baseUrl;
    QString _credential;
    QString _clientId;
    ConnectionState _connectionState = ConnectionState::Disconnected;
    quint64 _connectionGeneration = 0;  // stale replies are ignored
    bool _statusInFlight = false;
    int _statusFailures = 0;
    bool _hasActiveSession = false;
    bool _serviceOwnsDataset = false;
    bool _remoteLogsInFlight = false;
    bool _restartInProgress = false;
    QElapsedTimer _restartElapsed;
    int _remoteLogFailures = 0;
    qint64 _lastRemoteLogSequence = 0;
    QJsonObject _advertisedDataset;
    quint64 _commandCounter = 0;
    qint64 _lastStatusGeneration = -1;
    QString _installedPreviewArtifact;
    QString _fetchingPreviewArtifact;
    QString _installedGeometryArtifact;
    QString _fetchingGeometryArtifact;
    qint64 _previewSequence = 0;
    QString _lastPreviewLocalPath;
    QString _lastGeometryLocalPath;
};
