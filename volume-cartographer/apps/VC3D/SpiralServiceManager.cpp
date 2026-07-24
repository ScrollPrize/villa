#include "SpiralServiceManager.hpp"

#include "SpiralArtifactCache.hpp"
#include "SpiralSshTunnel.hpp"
#include "VCSettings.hpp"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QSettings>
#include <QStandardPaths>
#include <QUrl>
#include <QUuid>
#include <QtConcurrent/QtConcurrent>

#include <memory>

namespace {
constexpr int kPollMs = 500;
constexpr int kPollBackoffMs = 2000;
constexpr int kPollReconnectMs = 5000;
constexpr int kRemoteLogPollMs = 10000;
constexpr int kRestartProbeMs = 500;
constexpr int kRestartTimeoutMs = 60000;
constexpr int kMutationRetries = 2;
constexpr int kSupportedApiVersion = 6;
constexpr int kPreviewCacheKept = 3;

QString stateName(SpiralServiceManager::ConnectionState state)
{
    switch (state) {
    case SpiralServiceManager::ConnectionState::Disconnected: return QStringLiteral("Disconnected");
    case SpiralServiceManager::ConnectionState::Starting: return QStringLiteral("Starting");
    case SpiralServiceManager::ConnectionState::Connecting: return QStringLiteral("Connecting");
    case SpiralServiceManager::ConnectionState::Ready: return QStringLiteral("Ready");
    case SpiralServiceManager::ConnectionState::Reconnecting: return QStringLiteral("Reconnecting");
    case SpiralServiceManager::ConnectionState::Failed: return QStringLiteral("Failed");
    }
    return QStringLiteral("Unknown");
}
} // namespace

SpiralServiceManager::SpiralServiceManager(QObject* parent) : QObject(parent)
{
    _network = new QNetworkAccessManager(this);
    _artifactCache = new SpiralArtifactCache(this);
    _tunnel = new SpiralSshTunnel(this);
    _clientId = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    _poll = new QTimer(this);
    _poll->setInterval(kPollMs);
    connect(_poll, &QTimer::timeout, this, &SpiralServiceManager::pollStatus);
    _remoteLogPoll = new QTimer(this);
    _remoteLogPoll->setInterval(kRemoteLogPollMs);
    connect(_remoteLogPoll, &QTimer::timeout, this,
            &SpiralServiceManager::pollRemoteLogs);

    connect(_tunnel, &SpiralSshTunnel::logMessage, this, &SpiralServiceManager::logMessage);
    connect(_tunnel, &SpiralSshTunnel::ready, this, [this](int localPort) {
        if (_connectionState != ConnectionState::Starting
            && _connectionState != ConnectionState::Reconnecting) return;
        _baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(localPort));
        beginHandshake();
    });
    connect(_tunnel, &SpiralSshTunnel::failed, this,
            [this](SpiralSshTunnel::FailureKind, const QString& message) {
                setConnectionState(ConnectionState::Failed, message);
                emit errorOccurred(message);
            });
    connect(_tunnel, &SpiralSshTunnel::collapsed, this, [this](const QString& message) {
        emit logMessage(message);
        // A collapsed tunnel moves the connection to the reconnecting state.
        if (_connectionState == ConnectionState::Ready
            || _connectionState == ConnectionState::Connecting) {
            setConnectionState(ConnectionState::Reconnecting, message);
            startTunnel();
        }
    });
}

SpiralServiceManager::~SpiralServiceManager()
{
    disconnectFromService();
}

bool SpiralServiceManager::ownsProcess() const
{
    return _process && _process->state() != QProcess::NotRunning;
}

QString SpiralServiceManager::findPython() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QStringList candidates{
        settings.value(QStringLiteral("spiral/python")).toString(),
        qEnvironmentVariable("SPIRAL_PYTHON"),
        qEnvironmentVariable("PYTHON_EXECUTABLE"),
        QDir(qEnvironmentVariable("CONDA_PREFIX")).filePath(QStringLiteral("bin/python")),
        QStandardPaths::findExecutable(QStringLiteral("python3")),
        QStandardPaths::findExecutable(QStringLiteral("python")),
    };
    for (const QString& candidate : candidates)
        if (!candidate.isEmpty() && QFileInfo(candidate).isExecutable()) return QFileInfo(candidate).absoluteFilePath();
    return {};
}

QString SpiralServiceManager::findService() const
{
    const QString app = QCoreApplication::applicationDirPath();
    const QStringList candidates{
        qEnvironmentVariable("SPIRAL_SERVICE_PATH"),
        QDir::current().filePath(QStringLiteral("scripts/spiral/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../../scripts/spiral/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../../../scripts/spiral/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../share/volume-cartographer/spiral/spiral_service.py")),
    };
    for (const QString& candidate : candidates)
        if (!candidate.isEmpty() && QFileInfo(candidate).isFile()) return QFileInfo(candidate).absoluteFilePath();
    return {};
}

void SpiralServiceManager::setConnectionState(ConnectionState state, const QString& message)
{
    if (_connectionState == state && message.isEmpty()) return;
    _connectionState = state;
    emit connectionStateChanged(state, message);
    emit serviceStateChanged(message.isEmpty() ? stateName(state)
                                               : stateName(state) + QStringLiteral(": ") + message);
}

QString SpiralServiceManager::endpointFingerprint() const
{
    // Identifies the endpoint in the on-disk cache; must not contain the key.
    QString identity = _profile.id + QLatin1Char('|');
    if (_profile.transport == SpiralServiceProfile::Transport::SshTunnel)
        identity += QStringLiteral("ssh:%1:%2").arg(_profile.sshDestination)
                        .arg(_profile.remoteServicePort);
    else if (_profile.isLocalhost())
        identity += QStringLiteral("localhost");
    else
        identity += _profile.baseUrl.toString(QUrl::RemoveUserInfo);
    return QString::fromLatin1(
        QCryptographicHash::hash(identity.toUtf8(), QCryptographicHash::Sha256)
            .toHex().left(16));
}

void SpiralServiceManager::connectToService(const SpiralServiceProfile& profile)
{
    disconnectFromService();
    _profile = profile;
    ++_connectionGeneration;
    // A different or restarted service starts its generation counters over, so
    // every new connection resets the status/preview high-water marks.
    _lastStatusGeneration = -1;
    _installedPreviewArtifact.clear();
    _fetchingPreviewArtifact.clear();
    _installedGeometryArtifact.clear();
    _fetchingGeometryArtifact.clear();
    _statusFailures = 0;
    _hasActiveSession = false;
    _serviceOwnsDataset = false;
    _remoteLogsInFlight = false;
    _restartInProgress = false;
    _remoteLogFailures = 0;
    _lastRemoteLogSequence = 0;
    _advertisedDataset = {};

    _credential = profile.apiKey;
    if (_credential.isEmpty())
        _credential = qEnvironmentVariable("SPIRAL_API_KEY");

    if (profile.autoLaunch) {
        startLocalProcess();
        return;
    }
    if (profile.transport == SpiralServiceProfile::Transport::SshTunnel) {
        setConnectionState(ConnectionState::Starting, tr("Opening SSH tunnel to %1").arg(profile.sshDestination));
        startTunnel();
        return;
    }
    if (!profile.baseUrl.isValid() || profile.baseUrl.host().isEmpty()) {
        setConnectionState(ConnectionState::Failed, tr("The service URL is not valid"));
        return;
    }
    _baseUrl = profile.baseUrl;
    beginHandshake();
}

void SpiralServiceManager::startTunnel()
{
    const quint64 generation = _connectionGeneration;
    auto startForward = [this, generation]() {
        if (generation != _connectionGeneration) return;
        _tunnel->start(_profile.sshDestination, _profile.remoteServicePort);
    };
    if (!_credential.isEmpty()) {
        startForward();
        return;
    }
    // Read the service's auto-generated key file over SSH so the user never
    // copies a credential.
    _tunnel->readRemoteFile(
        _profile.sshDestination,
        QStringLiteral("${XDG_CONFIG_HOME:-$HOME/.config}/vc3d/spiral_api_key"),
        [this, generation, startForward](const QString& contents, const QString& error) {
            if (generation != _connectionGeneration) return;
            if (contents.isEmpty()) {
                const QString message =
                    tr("Could not read the Spiral API key on %1: %2\nStart the service "
                       "on the host once so it generates its key file.")
                        .arg(_profile.sshDestination, error);
                setConnectionState(ConnectionState::Failed, message);
                emit errorOccurred(message);
                return;
            }
            _credential = contents.split('\n').first().trimmed();
            startForward();
        });
}

void SpiralServiceManager::startLocalProcess()
{
    if (ownsProcess()) {
        // Reuse the already-running owned service.
        beginHandshake();
        return;
    }
    const QString python = findPython();
    const QString service = findService();
    if (python.isEmpty() || service.isEmpty()) {
        const QString message = tr("Cannot find the Spiral Python interpreter or spiral_service.py. Set SPIRAL_PYTHON and SPIRAL_SERVICE_PATH.");
        setConnectionState(ConnectionState::Failed, message);
        emit errorOccurred(message);
        return;
    }
    if (!_process) {
        _process = new QProcess(this);
        _process->setProcessChannelMode(QProcess::SeparateChannels);
        connect(_process, &QProcess::readyReadStandardOutput, this, [this]() {
            const QString output = QString::fromUtf8(_process->readAllStandardOutput());
            for (const QString& line : output.split('\n', Qt::SkipEmptyParts)) {
                emit logMessage(line);
                // The ready line carries only the port; API compatibility is
                // validated through the authenticated /health handshake.
                const QRegularExpressionMatch match = QRegularExpression(
                    QStringLiteral("^SPIRAL_SERVICE_READY port=(\\d+)\\b")).match(line.trimmed());
                if (match.hasMatch() && _connectionState == ConnectionState::Starting) {
                    _baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(match.captured(1)));
                    beginHandshake();
                }
            }
        });
        connect(_process, &QProcess::readyReadStandardError, this, [this]() {
            const QString output = QString::fromUtf8(_process->readAllStandardError());
            for (const QString& line : output.split('\n', Qt::SkipEmptyParts)) emit logMessage(line);
        });
        connect(_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
            emit errorOccurred(_process->errorString());
        });
        connect(_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
                [this](int code, QProcess::ExitStatus) {
                    _poll->stop();
                    if (_profile.autoLaunch
                        && _connectionState != ConnectionState::Disconnected) {
                        setConnectionState(ConnectionState::Failed, tr("The local Spiral service stopped"));
                    }
                    if (code != 0) emit errorOccurred(tr("Spiral service exited with code %1").arg(code));
                });
    }
    _credential = QString::number(QRandomGenerator::global()->generate64(), 16)
        + QString::number(QRandomGenerator::global()->generate64(), 16);
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    const QString moduleDir = QFileInfo(service).absolutePath();
    const QString oldPythonPath = environment.value(QStringLiteral("PYTHONPATH"));
    environment.insert(QStringLiteral("PYTHONPATH"), oldPythonPath.isEmpty() ? moduleDir : moduleDir + QDir::listSeparator() + oldPythonPath);
    environment.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    _process->setProcessEnvironment(environment);
    setConnectionState(ConnectionState::Starting);
    _process->start(python, {service, QStringLiteral("--nonce"), _credential,
                             QStringLiteral("--parent-pid"), QString::number(QCoreApplication::applicationPid())});
}

void SpiralServiceManager::beginHandshake()
{
    setConnectionState(ConnectionState::Connecting);
    const quint64 generation = _connectionGeneration;
    get(QStringLiteral("/health"), Timeout::Quick,
        [this, generation](const QJsonObject& health) {
            if (generation != _connectionGeneration) return;
            handleHealth(health);
        },
        [this, generation](const QString& error) {
            if (generation != _connectionGeneration) return;
            if (_connectionState == ConnectionState::Starting
                || _connectionState == ConnectionState::Connecting
                || _connectionState == ConnectionState::Reconnecting) {
                setConnectionState(ConnectionState::Failed, error);
                emit errorOccurred(tr("Spiral service handshake failed: %1").arg(error));
            }
        });
}

void SpiralServiceManager::handleHealth(const QJsonObject& health)
{
    const int apiVersion = health.value(QStringLiteral("api_version")).toInt(-1);
    if (apiVersion != kSupportedApiVersion) {
        const QString message = tr("Incompatible Spiral service: expected API version %1, received %2. "
                                   "Update the service and VC3D together.")
                                    .arg(kSupportedApiVersion).arg(apiVersion);
        setConnectionState(ConnectionState::Failed, message);
        emit errorOccurred(message);
        if (ownsProcess()) stopService();
        return;
    }
    _serviceOwnsDataset = health.value(QStringLiteral("dataset_owned")).toBool();
    _artifactCache->setEndpoint(endpointFingerprint(), _network,
                                [this](const QString& path, int timeoutMs) {
                                    return makeRequest(path, timeoutMs);
                                });
    const QString serviceName = health.value(QStringLiteral("service_name")).toString();
    setConnectionState(ConnectionState::Ready,
                       serviceName.isEmpty() ? QString() : tr("service %1").arg(serviceName));
    _statusFailures = 0;
    _poll->setInterval(kPollMs);
    _poll->start();
    if (!ownsProcess()) {
        _remoteLogPoll->start();
        pollRemoteLogs();
    } else {
        _remoteLogPoll->stop();
    }
    if (_serviceOwnsDataset) fetchAdvertisedDataset();
    pollStatus();
}

void SpiralServiceManager::fetchAdvertisedDataset()
{
    const quint64 generation = _connectionGeneration;
    get(QStringLiteral("/dataset"), Timeout::Command,
        [this, generation](const QJsonObject& dataset) {
            if (generation != _connectionGeneration) return;
            _advertisedDataset = dataset;
            emit datasetResolved(dataset);
        });
}

void SpiralServiceManager::disconnectFromService()
{
    ++_connectionGeneration;
    _poll->stop();
    _remoteLogPoll->stop();
    _statusInFlight = false;
    _remoteLogsInFlight = false;
    _restartInProgress = false;
    _artifactCache->clearEndpoint();
    _tunnel->stop();
    // Disconnecting from an independently started service never shuts the
    // service down; only a process this manager launched is terminated.
    if (ownsProcess()) stopService();
    if (_hasActiveSession) {
        _hasActiveSession = false;
        emit sessionActiveChanged(false);
    }
    setConnectionState(ConnectionState::Disconnected);
}

void SpiralServiceManager::reconnect()
{
    if (_profile.id.isEmpty()) return;
    connectToService(_profile);
}

void SpiralServiceManager::restartRemoteService()
{
    if (!_profile.isRemote()) {
        emit errorOccurred(tr("Only a remote Spiral service can be restarted"));
        return;
    }
    if (!isReady() || _restartInProgress) {
        emit errorOccurred(tr("Spiral service is not connected"));
        return;
    }

    post(QStringLiteral("/service/restart"),
         {{QStringLiteral("command_id"), commandId()}},
         Timeout::Command,
         [this](const QJsonObject&) {
             // Invalidate every request from the old process before probing
             // the replacement. The SSH tunnel itself remains alive.
             ++_connectionGeneration;
             _poll->stop();
             _remoteLogPoll->stop();
             _statusInFlight = false;
             _remoteLogsInFlight = false;
             _statusFailures = 0;
             _remoteLogFailures = 0;
             _lastStatusGeneration = -1;
             _installedPreviewArtifact.clear();
             _fetchingPreviewArtifact.clear();
             _installedGeometryArtifact.clear();
             _fetchingGeometryArtifact.clear();
             _lastRemoteLogSequence = 0;
             _serviceOwnsDataset = false;
             _advertisedDataset = {};
             _artifactCache->clearEndpoint();
             if (_hasActiveSession) {
                 _hasActiveSession = false;
                 emit sessionActiveChanged(false);
             }
             _restartInProgress = true;
             _restartElapsed.start();
             setConnectionState(ConnectionState::Reconnecting,
                                tr("Restarting remote service…"));
             emit logMessage(tr("Remote Spiral service accepted the restart request"));
             QTimer::singleShot(kRestartProbeMs, this,
                                &SpiralServiceManager::probeRestartedService);
         },
         [this](const QString& error) {
             emit errorOccurred(tr("Could not restart the Spiral service: %1").arg(error));
         });
}

void SpiralServiceManager::probeRestartedService()
{
    if (!_restartInProgress
        || _connectionState != ConnectionState::Reconnecting)
        return;

    const quint64 generation = _connectionGeneration;
    get(QStringLiteral("/health"), Timeout::Quick,
        [this, generation](const QJsonObject& health) {
            if (generation != _connectionGeneration || !_restartInProgress) return;
            _restartInProgress = false;
            emit logMessage(tr("Remote Spiral service restarted successfully"));
            handleHealth(health);
        },
        [this, generation](const QString& error) {
            if (generation != _connectionGeneration || !_restartInProgress) return;
            if (_restartElapsed.elapsed() >= kRestartTimeoutMs) {
                _restartInProgress = false;
                const QString message =
                    tr("The Spiral service did not return after restarting: %1").arg(error);
                setConnectionState(ConnectionState::Failed, message);
                emit errorOccurred(message);
                return;
            }
            QTimer::singleShot(kRestartProbeMs, this,
                               &SpiralServiceManager::probeRestartedService);
        });
}

void SpiralServiceManager::ensureStarted()
{
    if (_connectionState == ConnectionState::Ready
        || _connectionState == ConnectionState::Starting
        || _connectionState == ConnectionState::Connecting) return;
    if (_profile.id.isEmpty()) _profile = SpiralServiceProfile::localhostProfile();
    connectToService(_profile);
}

void SpiralServiceManager::stopService()
{
    _poll->stop();
    if (!_process || _process->state() == QProcess::NotRunning) return;
    _process->terminate();
    if (!_process->waitForFinished(2000)) {
        _process->kill();
        _process->waitForFinished(1000);
    }
}

QString SpiralServiceManager::commandId()
{
    // The random client id keeps commands from different computers from
    // colliding in the service's deduplication window.
    return QStringLiteral("vc3d-%1-%2").arg(_clientId).arg(++_commandCounter);
}

QNetworkRequest SpiralServiceManager::makeRequest(const QString& path, int timeoutMs) const
{
    QUrl url = _baseUrl;
    const QUrl relative(path);
    url.setPath(relative.path());
    url.setQuery(relative.query());
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    if (!_credential.isEmpty())
        request.setRawHeader("Authorization", QStringLiteral("Bearer %1").arg(_credential).toUtf8());
    request.setRawHeader("X-Spiral-Client", _clientId.toUtf8());
    request.setTransferTimeout(timeoutMs);
    return request;
}

void SpiralServiceManager::resolveDataset(const QString& root)
{
    if (!isReady()) { ensureStarted(); emit errorOccurred(tr("Spiral service is not connected; retry dataset resolution when Ready.")); return; }
    post(QStringLiteral("/dataset/resolve"), {{QStringLiteral("dataset_root"), root}},
         Timeout::Command,
         [this](const QJsonObject& value) { emit datasetResolved(value); });
}

void SpiralServiceManager::loadSession(QJsonObject request)
{
    QJsonObject inputPaths = request.value(QStringLiteral("paths")).toObject();
    if (!_serviceOwnsDataset) {
        request[QStringLiteral("command_id")] = commandId();
        sendLoadRequest(request, inputPaths);
        return;
    }
    // The service owns its base inputs; a remote load request carries run
    // parameters plus the client-selectable checkpoint/tracks values only.
    const QJsonObject requested = inputPaths;
    inputPaths = remoteInputPaths();
    QJsonObject selectable;
    const QString tracks = requested.value(QStringLiteral("tracks_dbm")).toString().trimmed();
    if (!tracks.isEmpty()) selectable[QStringLiteral("tracks_dbm")] = tracks;
    const QString checkpoint = requested.value(QStringLiteral("checkpoint")).toString().trimmed();

    auto finish = [this, request, inputPaths, selectable](const QString& checkpointHostPath) mutable {
        if (!checkpointHostPath.isEmpty())
            selectable[QStringLiteral("checkpoint")] = checkpointHostPath;
        QJsonObject load = request;
        if (selectable.isEmpty()) load.remove(QStringLiteral("paths"));
        else load[QStringLiteral("paths")] = selectable;
        load[QStringLiteral("command_id")] = commandId();
        sendLoadRequest(load, inputPaths);
    };

    if (checkpoint.isEmpty()) {
        finish({});
        return;
    }
    // Service-advertised checkpoints and paths under the service's output
    // directory pass through unchanged; an existing client-local file is
    // uploaded first and the load resumes from the returned host path.
    bool serviceSide = false;
    for (const QJsonValue& value :
         _advertisedDataset.value(QStringLiteral("detected_checkpoints")).toArray())
        if (value.toString() == checkpoint) serviceSide = true;
    const QString outputDir = _advertisedDataset.value(QStringLiteral("resolved")).toObject()
                                  .value(QStringLiteral("output_directory")).toString();
    if (!outputDir.isEmpty() && checkpoint.startsWith(outputDir)) serviceSide = true;
    if (serviceSide || !QFileInfo(checkpoint).isFile()) {
        finish(checkpoint);
        return;
    }
    emit logMessage(tr("Uploading resume checkpoint %1 to the service…").arg(checkpoint));
    uploadCheckpointForResume(checkpoint,
                              [this, finish](const QString& hostPath, const QString& error) mutable {
                                  if (hostPath.isEmpty()) {
                                      emit errorOccurred(tr("Resume checkpoint upload failed: %1").arg(error));
                                      return;
                                  }
                                  emit logMessage(tr("Checkpoint uploaded to service path %1").arg(hostPath));
                                  finish(hostPath);
                              });
}

void SpiralServiceManager::sendLoadRequest(QJsonObject request, const QJsonObject& inputPaths)
{
    postWithRetry(QStringLiteral("/session/load"), request, Timeout::Load, kMutationRetries,
                  [this, inputPaths](const QJsonObject& response) {
                      emit sessionAccepted(inputPaths,
                                           response.value(QStringLiteral("session_generation")).toInteger());
                  });
}

void SpiralServiceManager::uploadCheckpointForResume(
    const QString& localPath,
    std::function<void(const QString&, const QString&)> done)
{
    const quint64 generation = _connectionGeneration;
    auto* watcher = new QFutureWatcher<QJsonObject>(this);
    connect(watcher, &QFutureWatcher<QJsonObject>::finished, this,
            [this, watcher, localPath, generation, done]() {
                const QJsonObject digest = watcher->result();
                watcher->deleteLater();
                if (generation != _connectionGeneration) return;
                if (digest.contains(QStringLiteral("error"))) {
                    done({}, digest.value(QStringLiteral("error")).toString());
                    return;
                }
                QString inputId = QFileInfo(localPath).fileName();
                inputId.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]")),
                                QStringLiteral("-"));
                while (!inputId.isEmpty()
                       && !QRegularExpression(QStringLiteral("^[A-Za-z0-9]")).match(inputId).hasMatch())
                    inputId.remove(0, 1);
                if (inputId.isEmpty()) inputId = QStringLiteral("uploaded.ckpt");
                inputId.truncate(120);
                const QJsonObject begin{
                    {QStringLiteral("kind"), QStringLiteral("checkpoint")},
                    {QStringLiteral("id"), inputId},
                    {QStringLiteral("files"), QJsonArray{QJsonObject{
                        {QStringLiteral("name"), inputId},
                        {QStringLiteral("size"), digest.value(QStringLiteral("size"))},
                        {QStringLiteral("sha256"), digest.value(QStringLiteral("sha256"))},
                    }}},
                };
                post(QStringLiteral("/session/inputs"), begin, Timeout::Command,
                     [this, localPath, inputId, done](const QJsonObject& response) {
                         const QString uploadId =
                             response.value(QStringLiteral("upload_id")).toString();
                         if (uploadId.isEmpty()) {
                             done({}, tr("The service did not return an upload id"));
                             return;
                         }
                         auto file = std::make_unique<QFile>(localPath);
                         if (!file->open(QIODevice::ReadOnly)) {
                             done({}, tr("Cannot read %1").arg(localPath));
                             return;
                         }
                         // Checkpoints can be multiple gigabytes: no total
                         // transfer timeout; a dead transport surfaces as a
                         // socket error (the SSH tunnel keepalives bound it).
                         QNetworkRequest request = makeRequest(
                             QStringLiteral("/session/inputs/%1/files/%2").arg(uploadId, inputId), 0);
                         request.setHeader(QNetworkRequest::ContentTypeHeader,
                                           QStringLiteral("application/octet-stream"));
                         QFile* fileRaw = file.release();
                         auto* reply = _network->put(request, fileRaw);
                         fileRaw->setParent(reply);
                         connect(reply, &QNetworkReply::uploadProgress, this,
                                 [this](qint64 sent, qint64 total) {
                                     emit checkpointUploadProgress(sent, total);
                                 });
                         const quint64 putGeneration = _connectionGeneration;
                         connect(reply, &QNetworkReply::finished, this,
                                 [this, reply, putGeneration, uploadId, done]() {
                                     handleReply(reply, putGeneration,
                                                 [this, uploadId, done](const QJsonObject&) {
                                                     post(QStringLiteral("/session/inputs/%1/finalize").arg(uploadId),
                                                          {}, Timeout::LongCommand,
                                                          [done](const QJsonObject& response) {
                                                              const QString hostPath =
                                                                  response.value(QStringLiteral("input")).toObject()
                                                                      .value(QStringLiteral("path")).toString();
                                                              done(hostPath,
                                                                   hostPath.isEmpty()
                                                                       ? QObject::tr("The service did not return the checkpoint path")
                                                                       : QString());
                                                          },
                                                          [done](const QString& error) { done({}, error); });
                                                 },
                                                 [done](const QString& error) { done({}, error); });
                                 });
                     },
                     [done](const QString& error) { done({}, error); });
            });
    watcher->setFuture(QtConcurrent::run([localPath]() -> QJsonObject {
        QFile file(localPath);
        if (!file.open(QIODevice::ReadOnly))
            return {{QStringLiteral("error"), tr("Cannot read %1").arg(localPath)}};
        QCryptographicHash hash(QCryptographicHash::Sha256);
        if (!hash.addData(&file))
            return {{QStringLiteral("error"), tr("Cannot hash %1").arg(localPath)}};
        return {{QStringLiteral("size"), file.size()},
                {QStringLiteral("sha256"), QString::fromLatin1(hash.result().toHex())}};
    }));
}

QJsonObject SpiralServiceManager::remoteInputPaths() const
{
    QJsonObject paths;
    const QJsonObject resolved = _advertisedDataset.value(QStringLiteral("resolved")).toObject();
    for (auto it = resolved.begin(); it != resolved.end(); ++it)
        paths[it.key()] = it.value();
    paths[QStringLiteral("dataset_root")] = _advertisedDataset.value(QStringLiteral("root"));
    paths[QStringLiteral("pcls")] = _advertisedDataset.value(QStringLiteral("pcl_inputs"));
    return paths;
}

void SpiralServiceManager::runIterations(int iterations,
                                         const QJsonObject& influenceConfig,
                                         const QJsonObject& runConfig)
{
    postWithRetry(QStringLiteral("/session/run"),
                  {{QStringLiteral("command_id"), commandId()},
                   {QStringLiteral("iterations"), iterations},
                   {QStringLiteral("influence_config"), influenceConfig},
                   {QStringLiteral("run_config"), runConfig}},
                  Timeout::Command, kMutationRetries, {});
}

void SpiralServiceManager::stopAfterIteration()
{
    postWithRetry(QStringLiteral("/session/stop"),
                  {{QStringLiteral("command_id"), commandId()}},
                  Timeout::Command, kMutationRetries, {});
}

void SpiralServiceManager::saveCheckpoint(const QString& path)
{
    postWithRetry(QStringLiteral("/session/save-checkpoint"),
                  {{QStringLiteral("command_id"), commandId()},
                   {QStringLiteral("path"), path}},
                  Timeout::LongCommand, kMutationRetries, {});
}

void SpiralServiceManager::downloadCheckpoint(const QString& localPath)
{
    postWithRetry(
        QStringLiteral("/session/download-checkpoint"),
        {{QStringLiteral("command_id"), commandId()}},
        Timeout::LongCommand, kMutationRetries,
        [this, localPath](const QJsonObject& response) {
            const QJsonObject ref = response.value(QStringLiteral("checkpoint_artifact")).toObject();
            const QString artifactId = ref.value(QStringLiteral("id")).toString();
            const QString sessionId = response.value(QStringLiteral("session_id")).toString();
            if (artifactId.isEmpty()) {
                emit checkpointDownloadFinished(localPath, tr("The service did not return a checkpoint artifact"));
                return;
            }
            _artifactCache->fetchArtifact(
                sessionId, artifactId,
                [this, localPath](const QString& entryPath, const QString& error, bool) {
                    if (entryPath.isEmpty()) {
                        emit checkpointDownloadFinished(localPath, error);
                        return;
                    }
                    // Atomic replacement: a failed transfer cannot leave a
                    // partial file at the selected destination.
                    const QString temporary = localPath + QStringLiteral(".part");
                    QFile::remove(temporary);
                    if (!QFile::copy(entryPath, temporary)) {
                        emit checkpointDownloadFinished(localPath, tr("Could not write %1").arg(temporary));
                        return;
                    }
                    QFile::remove(localPath);
                    if (!QFile::rename(temporary, localPath)) {
                        QFile::remove(temporary);
                        emit checkpointDownloadFinished(localPath, tr("Could not replace %1").arg(localPath));
                        return;
                    }
                    emit checkpointDownloadFinished(localPath, {});
                });
        },
        [this, localPath](const QString& error) {
            emit checkpointDownloadFinished(localPath, error);
        });
}

void SpiralServiceManager::deleteSession()
{
    if (!isReady()) return;
    QNetworkRequest request = makeRequest(QStringLiteral("/session"),
                                          static_cast<int>(Timeout::Command));
    const QJsonObject body{{QStringLiteral("command_id"), commandId()}};
    auto* reply = _network->sendCustomRequest(request, "DELETE",
                                              QJsonDocument(body).toJson(QJsonDocument::Compact));
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        handleReply(reply, generation, {}, {});
    });
}

void SpiralServiceManager::commitInputs()
{
    postWithRetry(QStringLiteral("/session/commit-inputs"),
                  {{QStringLiteral("command_id"), commandId()}},
                  Timeout::LongCommand, kMutationRetries,
                  [this](const QJsonObject& response) {
                      if (_serviceOwnsDataset) fetchAdvertisedDataset();
                      handleStatus(response);
                      QStringList committed;
                      for (const QJsonValue& value : response.value(QStringLiteral("committed")).toArray())
                          committed.push_back(value.toString());
                      emit commitInputsFinished(committed, {});
                  },
                  [this](const QString& error) {
                      emit commitInputsFinished({}, error);
                      emit errorOccurred(error);
                  });
}

void SpiralServiceManager::removeEphemeralInput(const QString& kind, const QString& inputId)
{
    if (!isReady()) return;
    QNetworkRequest request = makeRequest(QStringLiteral("/session/ephemeral-inputs"),
                                          static_cast<int>(Timeout::Command));
    const QJsonObject body{{QStringLiteral("command_id"), commandId()},
                           {QStringLiteral("kind"), kind},
                           {QStringLiteral("id"), inputId}};
    auto* reply = _network->sendCustomRequest(request, "DELETE",
                                              QJsonDocument(body).toJson(QJsonDocument::Compact));
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        handleReply(reply, generation, {}, {});
    });
}

void SpiralServiceManager::uploadPatch(const QString& directory, const QString& inputId)
{
    if (!isReady()) { emit inputUploadFinished(inputId, tr("Spiral service is not connected")); return; }
    const quint64 generation = _connectionGeneration;
    auto* watcher = new QFutureWatcher<QJsonObject>(this);
    connect(watcher, &QFutureWatcher<QJsonObject>::finished, this,
            [this, watcher, directory, inputId, generation]() {
                const QJsonObject begin = watcher->result();
                watcher->deleteLater();
                if (generation != _connectionGeneration) return;
                if (begin.contains(QStringLiteral("error"))) {
                    emit inputUploadFinished(inputId, begin.value(QStringLiteral("error")).toString());
                    return;
                }
                QStringList names;
                for (const QJsonValue& value : begin.value(QStringLiteral("files")).toArray())
                    names.push_back(value.toObject().value(QStringLiteral("name")).toString());
                post(QStringLiteral("/session/inputs"), begin, Timeout::Command,
                     [this, directory, inputId, names](const QJsonObject& response) {
                         continueUpload(response.value(QStringLiteral("upload_id")).toString(),
                                        inputId, directory, names);
                     },
                     [this, inputId](const QString& error) {
                         emit inputUploadFinished(inputId, error);
                     });
            });
    watcher->setFuture(QtConcurrent::run([directory, inputId]() -> QJsonObject {
        QJsonArray files;
        QDirIterator it(directory, QDir::Files, QDirIterator::Subdirectories);
        const QDir base(directory);
        while (it.hasNext()) {
            const QString path = it.next();
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly))
                return {{QStringLiteral("error"), tr("Cannot read %1").arg(path)}};
            QCryptographicHash hash(QCryptographicHash::Sha256);
            hash.addData(&file);
            files.append(QJsonObject{
                {QStringLiteral("name"), base.relativeFilePath(path)},
                {QStringLiteral("size"), file.size()},
                {QStringLiteral("sha256"), QString::fromLatin1(hash.result().toHex())},
            });
        }
        if (files.isEmpty())
            return {{QStringLiteral("error"), tr("The patch directory %1 is empty").arg(directory)}};
        return {{QStringLiteral("kind"), QStringLiteral("patch")},
                {QStringLiteral("id"), inputId},
                {QStringLiteral("files"), files}};
    }));
}

void SpiralServiceManager::uploadJsonInput(const QString& kind, const QString& filePath,
                                           const QString& inputId, const QString& role)
{
    if (!isReady()) { emit inputUploadFinished(inputId, tr("Spiral service is not connected")); return; }
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        emit inputUploadFinished(inputId, tr("Cannot read %1").arg(filePath));
        return;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(&file);
    const QString name = QFileInfo(filePath).fileName();
    QJsonObject begin{
        {QStringLiteral("kind"), kind},
        {QStringLiteral("id"), inputId},
        {QStringLiteral("files"), QJsonArray{QJsonObject{
            {QStringLiteral("name"), name},
            {QStringLiteral("size"), file.size()},
            {QStringLiteral("sha256"), QString::fromLatin1(hash.result().toHex())},
        }}},
    };
    if (!role.isEmpty()) begin[QStringLiteral("role")] = role;
    const QString baseDir = QFileInfo(filePath).absolutePath();
    post(QStringLiteral("/session/inputs"), begin, Timeout::Command,
         [this, baseDir, inputId, name](const QJsonObject& response) {
             continueUpload(response.value(QStringLiteral("upload_id")).toString(),
                            inputId, baseDir, {name});
         },
         [this, inputId](const QString& error) { emit inputUploadFinished(inputId, error); });
}

void SpiralServiceManager::continueUpload(const QString& uploadId, const QString& inputId,
                                          const QString& baseDir, QStringList pendingFiles)
{
    if (uploadId.isEmpty()) {
        emit inputUploadFinished(inputId, tr("The service did not return an upload id"));
        return;
    }
    if (pendingFiles.isEmpty()) {
        post(QStringLiteral("/session/inputs/%1/finalize").arg(uploadId), {}, Timeout::Command,
             [this, inputId](const QJsonObject&) { emit inputUploadFinished(inputId, {}); },
             [this, inputId](const QString& error) { emit inputUploadFinished(inputId, error); });
        return;
    }
    const QString name = pendingFiles.takeFirst();
    auto file = std::make_unique<QFile>(QDir(baseDir).filePath(name));
    if (!file->open(QIODevice::ReadOnly)) {
        emit inputUploadFinished(inputId, tr("Cannot read %1").arg(file->fileName()));
        return;
    }
    QNetworkRequest request = makeRequest(
        QStringLiteral("/session/inputs/%1/files/%2").arg(uploadId, name),
        static_cast<int>(Timeout::LongCommand));
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/octet-stream"));
    QFile* fileRaw = file.release();
    auto* reply = _network->put(request, fileRaw);
    fileRaw->setParent(reply);
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, generation, uploadId, inputId, baseDir, pendingFiles]() {
                handleReply(reply, generation,
                            [this, uploadId, inputId, baseDir, pendingFiles](const QJsonObject&) {
                                continueUpload(uploadId, inputId, baseDir, pendingFiles);
                            },
                            [this, inputId](const QString& error) {
                                emit inputUploadFinished(inputId, error);
                            });
            });
}

void SpiralServiceManager::post(const QString& path, QJsonObject body, Timeout timeout,
                                std::function<void(const QJsonObject&)> success,
                                std::function<void(const QString&)> failure)
{
    if (!isReady() && _connectionState != ConnectionState::Connecting) {
        const QString message = tr("Spiral service is not connected");
        if (failure) failure(message); else emit errorOccurred(message);
        return;
    }
    QNetworkRequest request = makeRequest(path, static_cast<int>(timeout));
    auto* reply = _network->post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation, success, failure]() {
        handleReply(reply, generation, success, failure);
    });
}

void SpiralServiceManager::postWithRetry(const QString& path, QJsonObject body, Timeout timeout,
                                         int retriesLeft,
                                         std::function<void(const QJsonObject&)> success,
                                         std::function<void(const QString&)> failure)
{
    if (!isReady()) {
        const QString message = tr("Spiral service is not connected");
        if (failure) failure(message); else emit errorOccurred(message);
        return;
    }
    QNetworkRequest request = makeRequest(path, static_cast<int>(timeout));
    auto* reply = _network->post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, generation, path, body, timeout, retriesLeft, success, failure]() {
                const QNetworkReply::NetworkError networkError = reply->error();
                const int http = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
                // When a mutating request times out or the transport drops, it
                // is retried with the SAME command id: the service's
                // deduplication waits for the original or returns its cached
                // response, and never re-executes the operation.
                const bool transportFailure = networkError != QNetworkReply::NoError && http == 0;
                if (transportFailure && retriesLeft > 0 && generation == _connectionGeneration) {
                    reply->deleteLater();
                    emit logMessage(tr("Retrying %1 with the same command id (%2 retries left)")
                                        .arg(path).arg(retriesLeft));
                    QTimer::singleShot(1000, this, [this, path, body, timeout, retriesLeft, success, failure]() {
                        postWithRetry(path, body, timeout, retriesLeft - 1, success, failure);
                    });
                    return;
                }
                handleReply(reply, generation, success, failure);
            });
}

void SpiralServiceManager::get(const QString& path, Timeout timeout,
                               std::function<void(const QJsonObject&)> success,
                               std::function<void(const QString&)> failure)
{
    QNetworkRequest request = makeRequest(path, static_cast<int>(timeout));
    auto* reply = _network->get(request);
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation, success, failure]() {
        handleReply(reply, generation, success, failure);
    });
}

void SpiralServiceManager::handleReply(QNetworkReply* reply, quint64 generation,
                                       std::function<void(const QJsonObject&)> success,
                                       std::function<void(const QString&)> failure)
{
    const QByteArray bytes = reply->readAll();
    const int http = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    const QNetworkReply::NetworkError networkError = reply->error();
    const QString networkErrorString = reply->errorString();
    const QJsonDocument document = QJsonDocument::fromJson(bytes);
    reply->deleteLater();
    // Replies belonging to an obsolete connection generation are ignored.
    if (generation != _connectionGeneration) return;
    if (networkError != QNetworkReply::NoError && http == 0) {
        // Unreachable / timeout: distinguish from service-reported errors.
        const QString message = tr("Spiral service is unreachable: %1").arg(networkErrorString);
        if (failure) failure(message); else emit errorOccurred(message);
        return;
    }
    if (!document.isObject() || http >= 400) {
        QString message = document.object().value(QStringLiteral("error")).toString(
            networkErrorString.isEmpty() ? tr("Invalid Spiral service response") : networkErrorString);
        if (http == 401)
            message = tr("Unauthorized: the Spiral service rejected the API key. %1").arg(message);
        const QJsonArray details = document.object().value(QStringLiteral("details")).toArray();
        QStringList detailLines;
        for (const QJsonValue& value : details) {
            const QJsonObject detail = value.toObject();
            const QString field = detail.value(QStringLiteral("field")).toString();
            const QString description = detail.value(QStringLiteral("message")).toString();
            if (!description.isEmpty())
                detailLines.push_back(field.isEmpty() ? description : QStringLiteral("%1: %2").arg(field, description));
        }
        if (!detailLines.isEmpty()) message += QStringLiteral("\n") + detailLines.join(QStringLiteral("\n"));
        if (failure) failure(message); else emit errorOccurred(message);
        return;
    }
    if (success) success(document.object());
}

void SpiralServiceManager::pollStatus()
{
    // No more than one status poll in flight.
    if (_statusInFlight) return;
    if (_connectionState != ConnectionState::Ready
        && _connectionState != ConnectionState::Reconnecting) return;
    _statusInFlight = true;
    const quint64 generation = _connectionGeneration;
    get(QStringLiteral("/session/status"), Timeout::Quick,
        [this, generation](const QJsonObject& status) {
            _statusInFlight = false;
            if (generation != _connectionGeneration) return;
            if (_connectionState == ConnectionState::Reconnecting) {
                // Transport recovered: resume normal cadence and refresh the
                // generation trackers via the normal handling below.
                setConnectionState(ConnectionState::Ready);
                _lastStatusGeneration = -1;
            }
            _statusFailures = 0;
            _poll->setInterval(kPollMs);
            handleStatus(status);
        },
        [this, generation](const QString& error) {
            _statusInFlight = false;
            if (generation != _connectionGeneration) return;
            ++_statusFailures;
            if (_statusFailures == 1) emit logMessage(error);
            // Slow down while the endpoint is unreachable instead of hammering it.
            _poll->setInterval(_statusFailures >= 3 ? kPollReconnectMs : kPollBackoffMs);
            if (_statusFailures >= 3 && _connectionState == ConnectionState::Ready)
                setConnectionState(ConnectionState::Reconnecting, error);
        });
}

void SpiralServiceManager::pollRemoteLogs()
{
    if (_remoteLogsInFlight || ownsProcess()
        || _connectionState != ConnectionState::Ready)
        return;
    _remoteLogsInFlight = true;
    const quint64 generation = _connectionGeneration;
    get(QStringLiteral("/logs?after=%1").arg(_lastRemoteLogSequence), Timeout::Quick,
        [this, generation](const QJsonObject& response) {
            _remoteLogsInFlight = false;
            if (generation != _connectionGeneration) return;
            _remoteLogFailures = 0;
            if (response.value(QStringLiteral("cursor_reset")).toBool())
                _lastRemoteLogSequence = 0;
            const qint64 dropped = response.value(QStringLiteral("dropped")).toInteger();
            if (dropped > 0)
                emit logMessage(tr("Remote Python log buffer dropped %1 older line(s).")
                                    .arg(dropped));
            const QJsonArray entries = response.value(QStringLiteral("entries")).toArray();
            for (const QJsonValue& value : entries) {
                const QJsonObject entry = value.toObject();
                const qint64 sequence = entry.value(QStringLiteral("sequence")).toInteger();
                const QString message = entry.value(QStringLiteral("text")).toString();
                if (sequence <= _lastRemoteLogSequence || message.isEmpty()) continue;
                _lastRemoteLogSequence = sequence;
                emit logMessage(message);
            }
            _lastRemoteLogSequence = response.value(QStringLiteral("next_sequence"))
                                         .toInteger(_lastRemoteLogSequence);
        },
        [this, generation](const QString& error) {
            _remoteLogsInFlight = false;
            if (generation != _connectionGeneration) return;
            if (++_remoteLogFailures == 1)
                emit logMessage(tr("Remote Python log polling failed: %1").arg(error));
        });
}

void SpiralServiceManager::handleStatus(const QJsonObject& status)
{
    const qint64 generation = status.value(QStringLiteral("generation")).toInteger(-1);
    if (generation < _lastStatusGeneration) return;
    _lastStatusGeneration = generation;
    emit sessionStatusChanged(status);
    const bool active = !status.value(QStringLiteral("session_id")).toString().isEmpty()
        && status.value(QStringLiteral("state")).toString() != QStringLiteral("Empty");
    if (active != _hasActiveSession) {
        _hasActiveSession = active;
        emit sessionActiveChanged(active);
    }
    syncArtifacts(status);
}

void SpiralServiceManager::syncArtifacts(const QJsonObject& status)
{
    const QString sessionId = status.value(QStringLiteral("session_id")).toString();
    if (sessionId.isEmpty()) return;

    const QJsonObject previewRef = status.value(QStringLiteral("preview_artifact")).toObject();
    const QString previewId = previewRef.value(QStringLiteral("id")).toString();
    if (!previewId.isEmpty() && previewId != _installedPreviewArtifact
        && previewId != _fetchingPreviewArtifact) {
        _fetchingPreviewArtifact = previewId;
        const qint64 sequence = ++_previewSequence;
        const quint64 generation = _connectionGeneration;
        _artifactCache->fetchArtifact(
            sessionId, previewId,
            [this, previewId, sequence, sessionId, generation](const QString& entryPath,
                                                               const QString& error, bool gone) {
                if (generation != _connectionGeneration) return;
                if (_fetchingPreviewArtifact == previewId) _fetchingPreviewArtifact.clear();
                if (entryPath.isEmpty()) {
                    // 410 Gone means a newer preview exists; the next status
                    // poll carries its reference. Anything else is reported.
                    if (!gone) emit errorOccurred(error);
                    return;
                }
                // Previews are installed in order; a stale download is ignored.
                if (sequence < _previewSequence) return;
                _installedPreviewArtifact = previewId;
                _lastPreviewLocalPath = entryPath;
                emit previewAvailable(entryPath, sequence);
                _artifactCache->pruneSession(sessionId, kPreviewCacheKept,
                                             {_lastPreviewLocalPath, _lastGeometryLocalPath});
            });
    }

    const QJsonObject geometryRef = status.value(QStringLiteral("geometry_artifact")).toObject();
    const QString geometryId = geometryRef.value(QStringLiteral("id")).toString();
    if (!geometryId.isEmpty() && geometryId != _installedGeometryArtifact
        && geometryId != _fetchingGeometryArtifact) {
        _fetchingGeometryArtifact = geometryId;
        const quint64 generation = _connectionGeneration;
        const qint64 sessionGeneration = status.value(QStringLiteral("session_generation")).toInteger();
        _artifactCache->fetchArtifact(
            sessionId, geometryId,
            [this, geometryId, sessionGeneration, generation](const QString& entryPath,
                                                              const QString& error, bool gone) {
                if (generation != _connectionGeneration) return;
                if (_fetchingGeometryArtifact == geometryId) _fetchingGeometryArtifact.clear();
                if (entryPath.isEmpty()) {
                    if (!gone) emit errorOccurred(error);
                    return;
                }
                _installedGeometryArtifact = geometryId;
                _lastGeometryLocalPath = entryPath;
                emit geometryAvailable(entryPath, static_cast<quint64>(sessionGeneration));
            });
    }
}
