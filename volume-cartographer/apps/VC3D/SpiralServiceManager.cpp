#include "SpiralServiceManager.hpp"

#include "SpiralArtifactCache.hpp"
#include "SpiralSshTunnel.hpp"
#include "VCSettings.hpp"

#include <QCoreApplication>
#include <QApplication>
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
#include <QMessageBox>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QSettings>
#include <QStandardPaths>
#include <QUrl>
#include <QUuid>
#include <QtConcurrent/QtConcurrent>

#include <cmath>
#include <memory>

namespace {
constexpr int kPollMs = 500;
constexpr int kPollBackoffMs = 2000;
constexpr int kPollReconnectMs = 5000;
// One structured event subscriber covers every connection (local and
// remote). The service coalesces high-frequency records server-side, so a
// short poll interval keeps the panel live without flooding it.
constexpr int kEventPollMs = 500;
constexpr int kRestartProbeMs = 500;
constexpr int kRestartTimeoutMs = 60000;
constexpr int kMutationRetries = 2;
constexpr int kSupportedApiVersion = 18;
constexpr int kPreviewCacheKept = 3;

// Deterministic per-dataset default output root, outside the dataset: the
// service requires --output and rejects a location under --dataset.
QString defaultLocalOutputRoot(const QString& datasetRoot)
{
    const QString canonical = QFileInfo(datasetRoot).absoluteFilePath();
    const QString digest = QString::fromLatin1(
        QCryptographicHash::hash(canonical.toUtf8(), QCryptographicHash::Sha256)
            .toHex().left(8));
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation)
        + QStringLiteral("/spiral-output/%1-%2")
              .arg(QFileInfo(canonical).fileName(), digest);
}

// One panel line per structured event record. Log-kind records stay as raw
// text (the panel keeps its old /logs relay look); the other kinds carry a
// kind prefix, and child-rank records name their rank.
QString formatEventRecord(const QJsonObject& event)
{
    const QString kind = event.value(QStringLiteral("kind")).toString();
    const QString text = event.value(QStringLiteral("text")).toString();
    const QJsonValue rankValue = event.value(QStringLiteral("rank"));
    const QString rankSuffix = (rankValue.isDouble() && rankValue.toInt() > 0)
        ? QStringLiteral(" [rank %1]").arg(rankValue.toInt()) : QString();
    if (kind == QLatin1String("log")) return text + rankSuffix;
    const QJsonObject payload = event.value(QStringLiteral("payload")).toObject();
    if (kind == QLatin1String("progress")) {
        QString line = QStringLiteral("[progress] ") + text;
        const qint64 step = payload.value(QStringLiteral("step")).toInteger(-1);
        const qint64 total = payload.value(QStringLiteral("total_steps")).toInteger(-1);
        if (step >= 0 && total > 0) {
            line += QStringLiteral(" — %1/%2").arg(step).arg(total);
            const QString unit = payload.value(QStringLiteral("unit")).toString();
            if (!unit.isEmpty()) line += QLatin1Char(' ') + unit;
        }
        const QString detail = payload.value(QStringLiteral("detail")).toString();
        if (!detail.isEmpty()) line += QStringLiteral(" — ") + detail;
        return line + rankSuffix;
    }
    if (kind == QLatin1String("metric")) {
        QString line = QStringLiteral("[metric] ") + text;
        const QJsonValue loss = payload.value(QStringLiteral("total_loss"));
        if (loss.isDouble())
            line += QStringLiteral(" — loss %1").arg(loss.toDouble(), 0, 'g', 6);
        const QJsonValue lr = payload.value(QStringLiteral("learning_rate"));
        if (lr.isDouble())
            line += QStringLiteral(" — lr %1").arg(lr.toDouble(), 0, 'g', 4);
        return line + rankSuffix;
    }
    if (kind == QLatin1String("error"))
        return QStringLiteral("[error] ") + text + rankSuffix;
    return text + rankSuffix;
}

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
    _eventPoll = new QTimer(this);
    _eventPoll->setInterval(kEventPollMs);
    connect(_eventPoll, &QTimer::timeout, this,
            &SpiralServiceManager::pollEvents);
    connect(_artifactCache, &SpiralArtifactCache::fetchProgress, this,
            [this](const QString& artifactId, const QString& phase,
                   const QString& fileName, int filesComplete, int totalFiles,
                   qint64 bytesReceived, qint64 totalBytes) {
                if (artifactId != _fetchingPreviewArtifact) return;
                emit previewTransferProgress(
                    phase, fileName, filesComplete, totalFiles,
                    bytesReceived, totalBytes);
            });
    connect(_artifactCache, &SpiralArtifactCache::fetchProgress, this,
            [this](const QString& artifactId, const QString& phase,
                   const QString&, int, int,
                   qint64 bytesReceived, qint64 totalBytes) {
                if (artifactId != _fetchingCheckpointArtifact) return;
                emit checkpointDownloadProgress(phase, bytesReceived, totalBytes);
            });

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
    _installedPreviewSession.clear();
    _fetchingPreviewArtifact.clear();
    _synchronizedSessionId.clear();
    _statusFailures = 0;
    _hasActiveSession = false;
    _serviceOwnsDataset = false;
    _eventsInFlight = false;
    _restartInProgress = false;
    _eventFailures = 0;
    _lastEventCursor = 0;
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
    // The owned service is bound to one dataset/output/cache triple at
    // startup; a different selection is a different service instance.
    if (_profile.datasetRoot.trimmed().isEmpty()) {
        const QString message = tr("The local Spiral service needs a dataset root. "
                                   "Set it in the Spiral Service connection panel.");
        setConnectionState(ConnectionState::Failed, message);
        emit errorOccurred(message);
        return;
    }
    QString outputRoot = _profile.outputRoot.trimmed();
    if (outputRoot.isEmpty())
        outputRoot = defaultLocalOutputRoot(_profile.datasetRoot.trimmed());
    QStringList binding{QStringLiteral("--dataset"), _profile.datasetRoot.trimmed(),
                        QStringLiteral("--output"), outputRoot};
    if (!_profile.cacheRoot.trimmed().isEmpty())
        binding << QStringLiteral("--cache") << _profile.cacheRoot.trimmed();

    if (ownsProcess()) {
        if (binding == _ownedLaunchBinding) {
            // Reuse the already-running owned service: same bound instance.
            beginHandshake();
            return;
        }
        // A different dataset/output/cache selection restarts the owned
        // process with the new binding.
        emit logMessage(tr("Restarting the local Spiral service for a different "
                           "dataset binding"));
        stopService();
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
                // Once the connection is Ready the structured event stream
                // delivers the service's console lines; relaying stdout as
                // well would duplicate them. Before that, stdout is the only
                // startup diagnostic channel.
                if (_connectionState != ConnectionState::Ready) emit logMessage(line);
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
            for (const QString& line : output.split('\n', Qt::SkipEmptyParts))
                if (_connectionState != ConnectionState::Ready) emit logMessage(line);
        });
        connect(_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
            emit errorOccurred(_process->errorString());
        });
        connect(_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
                [this](int code, QProcess::ExitStatus) {
                    _poll->stop();
                    _eventPoll->stop();
                    // A dead service can no longer serve /events; surface any
                    // console tail (crash tracebacks) that was suppressed
                    // while the event stream was authoritative.
                    const QString tail =
                        QString::fromUtf8(_process->readAllStandardOutput())
                        + QString::fromUtf8(_process->readAllStandardError());
                    for (const QString& line : tail.split('\n', Qt::SkipEmptyParts))
                        emit logMessage(line);
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
    _ownedLaunchBinding = binding;
    _process->start(python, QStringList{service, QStringLiteral("--nonce"), _credential,
                                        QStringLiteral("--parent-pid"),
                                        QString::number(QCoreApplication::applicationPid())}
                                + binding);
}

void SpiralServiceManager::beginHandshake()
{
    // Keep a tunnel recovery distinguishable from an explicit new
    // connection. Consumers retain the synchronized session and preview while
    // the same endpoint is briefly unreachable.
    if (_connectionState != ConnectionState::Reconnecting)
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
    const QString sessionName = health.value(QStringLiteral("session_name")).toString();
    QString identity;
    if (!serviceName.isEmpty())
        identity = tr("service %1").arg(serviceName);
    if (!sessionName.isEmpty())
        identity += identity.isEmpty()
            ? tr("session %1").arg(sessionName)
            : tr(" / session %1").arg(sessionName);
    setConnectionState(ConnectionState::Ready, identity);
    _statusFailures = 0;
    _poll->setInterval(kPollMs);
    _poll->start();
    if (_serviceOwnsDataset) fetchAdvertisedDataset();
    get(QStringLiteral("/configuration"), Timeout::Quick,
        [this](const QJsonObject& catalog) {
            _configurationDefaults =
                catalog.value(QStringLiteral("defaults")).toObject();
            emit configurationCatalogChanged(catalog);
        });
    // Reconnect protocol: read the durable status snapshot first, then
    // subscribe to the bounded event stream from the persisted cursor.
    pollStatus();
    _eventPoll->start();
    pollEvents();
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
    _eventPoll->stop();
    _statusInFlight = false;
    _eventsInFlight = false;
    _restartInProgress = false;
    _artifactCache->clearEndpoint();
    _synchronizedSessionId.clear();
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
             _eventPoll->stop();
             _statusInFlight = false;
             _eventsInFlight = false;
             _statusFailures = 0;
             _eventFailures = 0;
             _lastStatusGeneration = -1;
             _installedPreviewArtifact.clear();
             _installedPreviewSession.clear();
             _fetchingPreviewArtifact.clear();
             _synchronizedSessionId.clear();
             _lastEventCursor = 0;
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
    if (_profile.id.isEmpty()) {
        // Pick up the persisted local launch binding (dataset/output/cache).
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _profile = SpiralServiceProfile::localhostProfile(&settings);
    }
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

void SpiralServiceManager::loadSession(QJsonObject request)
{
    if (!_serviceOwnsDataset) {
        request[QStringLiteral("command_id")] = commandId();
        sendLoadRequest(request);
        return;
    }
    // The service owns its base inputs; a remote load request carries run
    // parameters plus the client-selectable checkpoint/tracks values only.
    const QJsonObject requested =
        request.value(QStringLiteral("paths")).toObject();
    QJsonObject selectable;
    const QString tracks = requested.value(QStringLiteral("tracks_dbm")).toString().trimmed();
    if (!tracks.isEmpty()) selectable[QStringLiteral("tracks_dbm")] = tracks;
    const QString checkpoint = requested.value(QStringLiteral("checkpoint")).toString().trimmed();

    auto finish = [this, request, selectable](const QString& checkpointHostPath) mutable {
        if (!checkpointHostPath.isEmpty())
            selectable[QStringLiteral("checkpoint")] = checkpointHostPath;
        QJsonObject load = request;
        if (selectable.isEmpty()) load.remove(QStringLiteral("paths"));
        else load[QStringLiteral("paths")] = selectable;
        load[QStringLiteral("command_id")] = commandId();
        sendLoadRequest(load);
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
                              [this, finish](const QString& hostPath, const QString& error,
                                             bool reused) mutable {
                                  if (hostPath.isEmpty()) {
                                      emit errorOccurred(tr("Resume checkpoint upload failed: %1").arg(error));
                                      return;
                                  }
                                  emit logMessage(
                                      reused
                                          ? tr("Reusing checkpoint already on the service at %1").arg(hostPath)
                                          : tr("Checkpoint uploaded to service path %1").arg(hostPath));
                                  finish(hostPath);
                              });
}

void SpiralServiceManager::sendLoadRequest(QJsonObject request)
{
    postWithRetry(QStringLiteral("/session/load"), request, Timeout::Load, kMutationRetries,
                  [this](const QJsonObject& response) {
                      handleStatus(response);
                  });
}

void SpiralServiceManager::uploadCheckpointForResume(
    const QString& localPath,
    std::function<void(const QString&, const QString&, bool)> done)
{
    const quint64 generation = _connectionGeneration;
    auto* watcher = new QFutureWatcher<QJsonObject>(this);
    connect(watcher, &QFutureWatcher<QJsonObject>::finished, this,
            [this, watcher, localPath, generation, done]() {
                const QJsonObject digest = watcher->result();
                watcher->deleteLater();
                if (generation != _connectionGeneration) return;
                if (digest.contains(QStringLiteral("error"))) {
                    done({}, digest.value(QStringLiteral("error")).toString(), false);
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
                         if (response.value(QStringLiteral("deduplicated")).toBool()) {
                             const QString hostPath =
                                 response.value(QStringLiteral("input")).toObject()
                                     .value(QStringLiteral("path")).toString();
                             done(hostPath,
                                  hostPath.isEmpty()
                                      ? tr("The service did not return the cached checkpoint path")
                                      : QString(),
                                  true);
                             return;
                         }
                         const QString uploadId =
                             response.value(QStringLiteral("upload_id")).toString();
                         if (uploadId.isEmpty()) {
                             done({}, tr("The service did not return an upload id"), false);
                             return;
                         }
                         auto file = std::make_unique<QFile>(localPath);
                         if (!file->open(QIODevice::ReadOnly)) {
                             done({}, tr("Cannot read %1").arg(localPath), false);
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
                                                                       : QString(),
                                                                   false);
                                                          },
                                                          [done](const QString& error) {
                                                              done({}, error, false);
                                                          });
                                                 },
                                                 [done](const QString& error) {
                                                     done({}, error, false);
                                                 });
                                 });
                     },
                     [done](const QString& error) { done({}, error, false); });
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

void SpiralServiceManager::runIterations(int iterations,
                                         const QJsonObject& influenceConfig,
                                         const QJsonObject& runConfig,
                                         const QJsonObject& inputs)
{
    QJsonObject configuration = _configurationDefaults;
    for (auto it = _appliedConfiguration.begin();
         it != _appliedConfiguration.end(); ++it)
        configuration[it.key()] = it.value();
    for (auto it = runConfig.begin(); it != runConfig.end(); ++it)
        configuration[it.key()] = it.value();
    post(QStringLiteral("/session/run/plan"),
         {{QStringLiteral("configuration"), configuration},
          {QStringLiteral("iterations"), iterations},
          {QStringLiteral("influence"), influenceConfig},
          {QStringLiteral("inputs"), inputs},
          {QStringLiteral("expected_session_revision"), _sessionRevision}},
         Timeout::Command,
         [this](const QJsonObject& plan) {
             if (plan.value(QStringLiteral("new_fit_required")).toBool()) {
                 QMessageBox::information(
                     QApplication::activeWindow(), tr("Start New Fit required"),
                     tr("These changes are incompatible with the resident model. "
                        "Use New Fit to apply them."));
                 emit configurationReviewRequested();
                 return;
             }
             if (plan.value(QStringLiteral("session_reload_required")).toBool()) {
                 QMessageBox::information(
                     QApplication::activeWindow(), tr("Reload fit inputs required"),
                     tr("These changes require reloading the resident fit inputs. "
                        "Use New Fit to apply them."));
                 emit configurationReviewRequested();
                 return;
             }
             const bool changed =
                 !plan.value(QStringLiteral("changes")).toArray().isEmpty()
                 || plan.value(QStringLiteral("input_changed")).toBool();
             if (changed) {
                 QMessageBox box(QMessageBox::Question,
                                 tr("Spiral configuration changed"),
                                 tr("Configuration changed since last run: continue?"),
                                 QMessageBox::Cancel,
                                 QApplication::activeWindow());
                 auto* proceed = box.addButton(tr("Continue Run"),
                                               QMessageBox::AcceptRole);
                 auto* review = box.addButton(tr("Review"),
                                              QMessageBox::ActionRole);
                 box.exec();
                 if (box.clickedButton() == review) {
                     emit configurationReviewRequested();
                     return;
                 }
                 if (box.clickedButton() != proceed) return;
             }
             postWithRetry(
                 QStringLiteral("/session/run"),
                 {{QStringLiteral("command_id"), commandId()},
                  {QStringLiteral("plan_token"),
                   plan.value(QStringLiteral("plan_token"))}},
                 Timeout::Command, kMutationRetries, {});
         });
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
    emit checkpointDownloadProgress(QStringLiteral("creating"), 0, 0);
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
            _fetchingCheckpointArtifact = artifactId;
            _artifactCache->fetchArtifact(
                sessionId, artifactId,
                [this, localPath](const QString& entryPath, const QString& error, bool) {
                    _fetchingCheckpointArtifact.clear();
                    if (entryPath.isEmpty()) {
                        emit checkpointDownloadFinished(localPath, error);
                        return;
                    }
                    emit checkpointDownloadProgress(QStringLiteral("copying"), 0, 0);
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
        handleReply(reply, generation,
                    [this](const QJsonObject& response) {
                        handleStatus(response);
                    }, {});
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

void SpiralServiceManager::pollEvents()
{
    if (_eventsInFlight || _connectionState != ConnectionState::Ready) return;
    _eventsInFlight = true;
    const quint64 generation = _connectionGeneration;
    get(QStringLiteral("/events?cursor=%1").arg(_lastEventCursor), Timeout::Quick,
        [this, generation](const QJsonObject& response) {
            _eventsInFlight = false;
            if (generation != _connectionGeneration) return;
            _eventFailures = 0;
            if (response.value(QStringLiteral("cursor_reset")).toBool())
                _lastEventCursor = 0;
            const qint64 dropped = response.value(QStringLiteral("dropped")).toInteger();
            if (dropped > 0 && _lastEventCursor > 0) {
                // The bounded event ring overran this client's cursor: the
                // stream is history, not reconnect state, so refresh the
                // durable view from the status snapshot and continue from
                // the cursor the service reports.
                emit logMessage(tr("Spiral event stream dropped %1 older record(s); "
                                   "refreshing session status.").arg(dropped));
                pollStatus();
            }
            const QJsonArray events = response.value(QStringLiteral("events")).toArray();
            for (const QJsonValue& value : events) {
                const QJsonObject event = value.toObject();
                const qint64 sequence = event.value(QStringLiteral("sequence")).toInteger();
                if (sequence <= _lastEventCursor) continue;
                _lastEventCursor = sequence;
                if (event.value(QStringLiteral("severity")).toString()
                        == QLatin1String("error")) {
                    // Popups are reserved for error severity; the handler
                    // also appends the message to the panel.
                    emit errorOccurred(event.value(QStringLiteral("text")).toString());
                    continue;
                }
                const QString line = formatEventRecord(event);
                if (!line.isEmpty()) emit logMessage(line);
            }
            _lastEventCursor = response.value(QStringLiteral("next_cursor"))
                                   .toInteger(_lastEventCursor);
        },
        [this, generation](const QString& error) {
            _eventsInFlight = false;
            if (generation != _connectionGeneration) return;
            if (++_eventFailures == 1)
                emit logMessage(tr("Spiral event polling failed: %1").arg(error));
        });
}

void SpiralServiceManager::handleStatus(const QJsonObject& status)
{
    const qint64 generation = status.value(QStringLiteral("generation")).toInteger(-1);
    if (generation < _lastStatusGeneration) return;
    _lastStatusGeneration = generation;
    _sessionRevision =
        status.value(QStringLiteral("session_revision")).toInteger();
    const QJsonObject applied =
        status.value(QStringLiteral("applied_config")).toObject();
    if (!applied.isEmpty()) _appliedConfiguration = applied;
    const QString sessionId =
        status.value(QStringLiteral("session_id")).toString();
    const bool active = !sessionId.isEmpty()
        && status.value(QStringLiteral("state")).toString() != QStringLiteral("Empty");
    if (active != _hasActiveSession) {
        _hasActiveSession = active;
        emit sessionActiveChanged(active);
    }
    if (!active) {
        _synchronizedSessionId.clear();
        _installedPreviewSession.clear();
    } else if (sessionId != _synchronizedSessionId) {
        const QJsonObject request =
            status.value(QStringLiteral("session_request")).toObject();
        if (!request.isEmpty()) {
            _synchronizedSessionId = sessionId;
            emit sessionSynchronized(request, status);
        }
    }
    emit sessionStatusChanged(status);
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
                _installedPreviewSession = sessionId;
                _lastPreviewLocalPath = entryPath;
                emit previewAvailable(entryPath, sequence);
                _artifactCache->pruneSession(
                    sessionId, kPreviewCacheKept,
                    {_lastPreviewLocalPath});
            });
    }
}

void SpiralServiceManager::fetchPreviewFile(const QString& relativeName,
                                            FetchPreviewFileCallback done)
{
    if (_installedPreviewArtifact.isEmpty() || _installedPreviewSession.isEmpty()) {
        done({}, tr("No Spiral preview artifact is installed"));
        return;
    }
    const QString artifactId = _installedPreviewArtifact;
    const QString sessionId = _installedPreviewSession;
    const quint64 generation = _connectionGeneration;
    _artifactCache->fetchFile(
        sessionId, artifactId, relativeName,
        [this, artifactId, generation, done = std::move(done)](
            const QString& localPath, const QString& error, bool gone) {
            if (generation != _connectionGeneration
                || artifactId != _installedPreviewArtifact)
                return;
            done(localPath,
                 gone ? tr("The Spiral preview was pruned before the file was downloaded")
                      : error);
        });
}
