#include "SpiralServiceManager.hpp"

#include "SpiralArtifactCache.hpp"
#include "SpiralArtifactPins.hpp"
#include "SpiralPreviewProvenance.hpp"
#include "SpiralSessionSync.hpp"
#include "SpiralSshTunnel.hpp"
#include "VCSettings.hpp"

#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <QCoreApplication>
#include <QApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHostAddress>
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
#include <QSignalBlocker>
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
constexpr int kMutationRetries = 2;
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
// text (the panel keeps its plain console-line look); the other kinds carry a
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
    // The owner's destructor has already disconnected while its widgets were
    // alive; by now receivers may hold dangling pointers, so release resources
    // without notifying anyone.
    const QSignalBlocker blocker(this);
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

QString SpiralServiceManager::findService()
{
    const QString app = QCoreApplication::applicationDirPath();
    const QStringList candidates{
        qEnvironmentVariable("SPIRAL_SERVICE_PATH"),
        QDir::current().filePath(QStringLiteral("spiral-fitting/spiral_service.py")),
        QDir::current().filePath(QStringLiteral("../spiral-fitting/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../../../spiral-fitting/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../../../../spiral-fitting/spiral_service.py")),
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
    _displayedPreviewSourceIteration = -1;
    _fetchingPreviewArtifact.clear();
    _installedDiagnosticsArtifact.clear();
    _fetchingDiagnosticsArtifact.clear();
    _installedInputManifest.clear();
    _fetchingInputManifest.clear();
    _lastInputManifestLocalPath.clear();
    _lastPreviewLocalPath.clear();
    _lastDiagnosticsLocalPath.clear();
    _synchronizedSessionId.clear();
    _statusFailures = 0;
    _hasActiveSession = false;
    _eventsInFlight = false;
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
    // Patches, PCLs and fibers come from an object store: the local service
    // is launched bound to the one attached here.
    if (_localStorePath.isEmpty()) {
        const QString message = tr("The local Spiral service reads its patches, point "
                                   "collections and fibers from one object store. Route "
                                   "all three to one of the project's stores (File > Stores) "
                                   "and connect again.");
        setConnectionState(ConnectionState::Failed, message);
        emit errorOccurred(message);
        return;
    }
    QStringList binding{QStringLiteral("--dataset"), _profile.datasetRoot.trimmed(),
                        QStringLiteral("--output"), outputRoot};
    // A hosted replica's service works on its own replica of the same hub
    // (cloned at first launch, reattached later), never on this one.
    if (_storeBinding.localHosted())
        binding << QStringLiteral("--object-store")
                << localServiceReplica(outputRoot, _storeBinding.localHubIdentity)
                << QStringLiteral("--object-store-hub") << _storeBinding.localHub;
    else
        binding << QStringLiteral("--object-store") << _localStorePath;
    if (!_profile.cacheRoot.trimmed().isEmpty())
        binding << QStringLiteral("--cache") << _profile.cacheRoot.trimmed();

    if (ownsProcess()) {
        if (binding == _ownedLaunchBinding) {
            // Reuse the already-running owned service: same bound instance.
            beginHandshake();
            return;
        }
        // A different dataset/output/cache/store selection restarts the
        // owned process with the new binding.
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
    if (apiVersion != SpiralServiceManager::kApiVersion) {
        const QString message = tr("Incompatible Spiral service: expected API version %1, received %2. "
                                   "Update the service and VC3D together.")
                                    .arg(SpiralServiceManager::kApiVersion).arg(apiVersion);
        setConnectionState(ConnectionState::Failed, message);
        emit errorOccurred(message);
        if (ownsProcess()) stopService();
        return;
    }
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
    claimSessionLease();
    _statusFailures = 0;
    _poll->setInterval(kPollMs);
    _poll->start();
    fetchAdvertisedDataset();
    get(QStringLiteral("/configuration"), Timeout::Quick,
        [this](const QJsonObject& catalog) {
            _configurationDefaults =
                catalog.value(QStringLiteral("defaults")).toObject();
            _serviceOwnedKeys.clear();
            const QJsonObject fields = catalog.value(QStringLiteral("schema")).toObject()
                                           .value(QStringLiteral("fields")).toObject();
            for (auto it = fields.begin(); it != fields.end(); ++it)
                if (it.value().toObject().value(QStringLiteral("service_owned")).toBool())
                    _serviceOwnedKeys.insert(it.key());
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
            updateStoreBinding();
            emit datasetResolved(dataset);
        });
}

void SpiralServiceManager::setObjectStore(
    std::shared_ptr<vc::objects::ObjectStoreSession> session,
    std::optional<vc::objects::AttachmentIdentity> attachment)
{
    if (_inputUpdateActive && _inputUpdateAttachment
        && (session != _storeSession.lock() || !attachment || *attachment != *_inputUpdateAttachment))
        abandonInputUpdate(tr("The attached object store changed"));
    _storeSession = session;
    _storeAttachment = attachment ? std::make_unique<vc::objects::AttachmentIdentity>(*attachment) : nullptr;
    QString path;
    QString uuid;
    if (session && attachment) {
        path = QString::fromStdString(session->path(*attachment).string());
        uuid = QString::fromStdString(attachment->storeUuid);
    }
    const QString hub = _storeBinding.localHub;
    const bool changed = path != _localStorePath;
    _localStorePath = path;
    _storeBinding.localPath = path;
    _storeBinding.localUuid = uuid;
    updateSameFile();
    refreshLocalHub();
    // A local service is bound to the store at launch: follow the attachment.
    if ((changed || hub != _storeBinding.localHub) && ownsProcess() && _profile.autoLaunch
        && _connectionState != ConnectionState::Disconnected) {
        emit logMessage(tr("The attached object store changed; restarting the local Spiral service"));
        connectToService(_profile);
    }
}

void SpiralServiceManager::refreshLocalHub()
{
    const auto session = _storeSession.lock();
    std::optional<vc::objects::HubSessionStatus> status;
    if (session && _storeAttachment) status = session->hubStatus(*_storeAttachment);
    _storeBinding.localHub = status ? QString::fromStdString(status->hub) : QString();
    _storeBinding.localHubIdentity = status ? QString::fromStdString(status->hubReplica) : QString();
    _storeBinding.localPosition = status && status->syncedAtUs ? status->position : -1;
    _storeBinding.localPending = status ? int(status->pending.size()) : 0;
    _storeBinding.localConflicts = status ? int(status->conflicts.size()) : 0;
    emit storeBindingChanged();
}

QString SpiralServiceManager::localServiceReplica(const QString& outputRoot,
                                                  const QString& hubIdentity)
{
    return QDir(outputRoot).filePath(QStringLiteral("object-store-replicas/%1.vcobj").arg(hubIdentity));
}

QString SpiralServiceManager::serviceHubAddress() const
{
    if (!_profile.hubAddress.trimmed().isEmpty()) return _profile.hubAddress.trimmed();
    const QString advertised = _storeBinding.serviceHub;
    if (advertised.isEmpty() || advertised.contains(QStringLiteral("://"))) return advertised;
    if (_profile.transport == SpiralServiceProfile::Transport::SshTunnel && !_profile.sshDestination.isEmpty())
        return QStringLiteral("sftp://%1%2").arg(_profile.sshDestination, advertised);
    return advertised;
}

void SpiralServiceManager::updateStoreBinding()
{
    const QJsonObject store = _advertisedDataset.value(QStringLiteral("object_store")).toObject();
    _storeBinding.serviceUuid = store.value(QStringLiteral("uuid")).toString();
    _storeBinding.servicePath = store.value(QStringLiteral("path")).toString();
    const QJsonObject hub = store.value(QStringLiteral("hub")).toObject();
    _storeBinding.serviceHub = hub.value(QStringLiteral("address")).toString();
    _storeBinding.serviceHubIdentity = hub.value(QStringLiteral("identity")).toString();
    _storeBinding.servicePosition = hub.value(QStringLiteral("position")).toInteger(-1);
    _storeBinding.servicePositionError = hub.value(QStringLiteral("position_error")).toString();
    const QJsonObject positions = store.value(QStringLiteral("positions")).toObject();
    _storeBinding.servicePositions = _storeBinding.hosted()
        ? _storeBinding.serviceHubIdentity : positions.value(QStringLiteral("identity")).toString();
    if (!_storeBinding.hosted() && !positions.isEmpty()) {
        _storeBinding.servicePosition = positions.value(QStringLiteral("position")).toInteger(-1);
        _storeBinding.servicePositionError = positions.value(QStringLiteral("position_error")).toString();
    }
    updateSameFile();
    if (_storeBinding.advertised() && !_storeBinding.matches())
        emit logMessage(
            _storeBinding.localUuid.isEmpty()
                ? tr("The Spiral service reads object store %1, and the project routes Spiral's "
                     "inputs to no one store here: its fit inputs cannot be displayed")
                      .arg(_storeBinding.serviceUuid)
            : _storeBinding.serviceUuid == _storeBinding.localUuid
                ? tr("The Spiral service fits positions of hub %1 (%2); the attached store is "
                     "not a replica of it: clone that hub to edit and update the fit's inputs")
                      .arg(_storeBinding.serviceHubIdentity, serviceHubAddress())
                : tr("The Spiral service reads object store %1, not the attached store %2: "
                     "its fit inputs are not displayed as fitted")
                      .arg(_storeBinding.serviceUuid, _storeBinding.localUuid));
    emit storeBindingChanged();
}

qint64 SpiralServiceManager::storeHead(vc::objects::ObjectStoreSession& session,
                                      const vc::objects::AttachmentIdentity& attachment)
{
    qint64 head = -1;
    session.read(attachment, [&](const vc::objects::ObjectStore& store, const vc::objects::ReadTxn& txn,
                                 const std::atomic_bool&) { head = store.journalPosition(txn); });
    return head;
}

void SpiralServiceManager::updateSameFile()
{
    // Only a service on this machine opens the path it advertises here: a
    // launched one, or one reached directly at a loopback address (an SSH
    // tunnel's local end is not).
    const QString host = _profile.baseUrl.host();
    const bool here = _profile.isLocalhost()
        || (_profile.transport == SpiralServiceProfile::Transport::Direct
            && (host == QStringLiteral("localhost") || QHostAddress(host).isLoopback()));
    const QFileInfo local(_storeBinding.localPath), service(_storeBinding.servicePath);
    _storeBinding.sameFile = here && !_storeBinding.localPath.isEmpty()
        && local.exists() && local.canonicalFilePath() == service.canonicalFilePath();
}

void SpiralServiceManager::abandonInputUpdate(const QString& reason)
{
    // The replies of the old connection are dropped (or its hold ended with
    // the attachment), so an update waiting on them would never finish: end
    // it here. Its outcome on the service is unknown until the next status
    // names the fit's position.
    if (!_inputUpdateActive) return;
    ++_inputUpdateSerial;
    _inputUpdateHold.reset();
    _inputUpdateAttachment.reset();
    _inputUpdateActive = false;
    InputUpdate abandoned;
    abandoned.error = tr("%1 during the update; its outcome is unknown: check the fit's input position")
                          .arg(reason);
    emit inputUpdateFinished(abandoned);
}

void SpiralServiceManager::disconnectFromService()
{
    abandonInputUpdate(tr("The connection to the Spiral service changed"));
    ++_connectionGeneration;
    _leaseOwner = false;
    _poll->stop();
    _eventPoll->stop();
    _statusInFlight = false;
    _eventsInFlight = false;
    _artifactCache->clearEndpoint();
    _synchronizedSessionId.clear();
    _displayedPreviewSourceIteration = -1;
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
    if (ownsProcess()) {
        abandonInputUpdate(tr("The connection to the Spiral service changed"));
        ++_connectionGeneration;
        _leaseOwner = false;
        _statusInFlight = false;
        _eventsInFlight = false;
        _synchronizedSessionId.clear();
        beginHandshake();
        return;
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
    request.setRawHeader("X-Spiral-Session-Token", _clientId.toUtf8());
    request.setTransferTimeout(timeoutMs);
    return request;
}

void SpiralServiceManager::initializeSession(QJsonObject request)
{
    prepareSessionRequest(std::move(request), true);
}

void SpiralServiceManager::rebuildSession(QJsonObject request, bool reloadInputs)
{
    if (reloadInputs) request[QStringLiteral("inputs")] = QStringLiteral("reload");
    prepareSessionRequest(std::move(request), false);
}

void SpiralServiceManager::prepareSessionRequest(QJsonObject request,
                                                 bool initialize)
{
    // The service owns its base inputs (it is launched with --dataset), so a
    // session construction request carries run
    // parameters plus the client-selectable checkpoint only.
    const QJsonObject requested =
        request.value(QStringLiteral("paths")).toObject();
    QJsonObject selectable;
    const QString checkpoint = requested.value(QStringLiteral("checkpoint")).toString().trimmed();
    // The service's launch binding owns these (its object store and the
    // author of its link refreshes); a request never chooses them.
    QJsonObject run = request.value(QStringLiteral("run")).toObject();
    QJsonObject config = run.value(QStringLiteral("config")).toObject();
    for (const QString& key : _serviceOwnedKeys) config.remove(key);
    run[QStringLiteral("config")] = config;
    request[QStringLiteral("run")] = run;

    auto finish = [this, request, selectable, initialize](
                      const QString& checkpointHostPath) mutable {
        if (!checkpointHostPath.isEmpty())
            selectable[QStringLiteral("checkpoint")] = checkpointHostPath;
        QJsonObject load = request;
        if (selectable.isEmpty()) load.remove(QStringLiteral("paths"));
        else load[QStringLiteral("paths")] = selectable;
        load[QStringLiteral("command_id")] = commandId();
        if (initialize) sendInitializeRequest(load);
        else sendRebuildRequest(load);
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

void SpiralServiceManager::updateFitInputs(QJsonObject request)
{
    // A second request while one runs is dropped, not reported as that
    // update's outcome: inputUpdateFinished belongs to the running one.
    if (_inputUpdateActive) {
        emit logMessage(tr("An input update is already in progress"));
        return;
    }
    InputUpdate refused;
    const auto session = _storeSession.lock();
    if (!isReady() || !_leaseOwner)
        refused.error = tr("Connect to the Spiral service and own its session lease first");
    else if (!_storeBinding.positioned())
        refused.error = tr("The Spiral service fits no store positions: rebuild with current inputs instead");
    else if (_storeBinding.hosted() && (!_storeBinding.matches() || !session || !_storeAttachment))
        refused.error = tr("Attach a replica of the service's hub (%1) to update the fit's inputs")
                            .arg(_storeBinding.serviceHubIdentity);
    else if (!_storeBinding.hosted() && (!_storeBinding.shared() || !session || !_storeAttachment))
        refused.error = tr("Spiral's store here is not the service's store file (%1): saved edits "
                           "cannot reach the fit").arg(_storeBinding.servicePath);
    if (!refused.error.isEmpty()) {
        emit inputUpdateFinished(refused);
        return;
    }
    _inputUpdateActive = true;
    const quint64 serial = ++_inputUpdateSerial;
    const vc::objects::AttachmentIdentity attachment = *_storeAttachment;
    _inputUpdateAttachment = std::make_unique<vc::objects::AttachmentIdentity>(attachment);
    if (!_storeBinding.hosted()) {
        // The service's own file: saves are already there. The update
        // includes everything saved so far, at the store's head now.
        try {
            sendInputUpdate(request, storeHead(*session, attachment), serial);
        } catch (const std::exception& error) {
            refused.error = QString::fromUtf8(error.what());
            finishInputUpdate(refused, serial);
        }
        return;
    }
    // Held before pushing: the waited sync below still pushes everything
    // saved so far, and nothing pushed meanwhile can slip in between it and
    // the service's link-then-pin loop (which would only make it relink).
    try {
        _inputUpdateHold = std::make_shared<vc::objects::PushHold>(session->holdPushes(attachment));
    } catch (const std::exception& error) {
        _inputUpdateActive = false;
        _inputUpdateAttachment.reset();
        refused.error = QString::fromUtf8(error.what());
        emit inputUpdateFinished(refused);
        return;
    }
    emit inputUpdateProgress(QStringLiteral("pushing"), tr("Pushing saved edits to the hub…"));
    using Pushed = std::pair<std::optional<vc::objects::HubSessionStatus>, QString>;
    auto* watcher = new QFutureWatcher<Pushed>(this);
    connect(watcher, &QFutureWatcher<Pushed>::finished, this, [this, watcher, request, serial]() {
        const auto [status, exception] = watcher->result();
        watcher->deleteLater();
        if (serial != _inputUpdateSerial) return;  // abandoned
        InputUpdate update;
        if (!exception.isEmpty() || !status) {
            update.error = tr("The hub could not be reached: %1").arg(exception.isEmpty() ? tr("no hub") : exception);
        } else if (!status->conflicts.empty()) {
            update.conflicts = true;
            update.error = tr("Hub conflicts block the update: resolve them (File > Stores > "
                              "Sync Hosted Stores Now), then update again");
            for (const auto& conflict : status->conflicts)
                update.problems << QString::fromStdString(conflict.uuid);
        } else if (!status->error.empty()) {
            update.error = tr("Saved edits could not be pushed: %1").arg(QString::fromStdString(status->error));
        } else if (!status->pending.empty()) {
            update.error = tr("%n saved edit(s) were not pushed", nullptr, int(status->pending.size()));
        }
        if (!update.error.isEmpty()) {
            finishInputUpdate(update, serial);
            return;
        }
        sendInputUpdate(request, status->position, serial);
    });
    watcher->setFuture(QtConcurrent::run([session, attachment]() -> Pushed {
        try {
            return {session->syncHubAndWait(attachment, std::chrono::seconds(120)), QString()};
        } catch (const std::exception& error) {
            return {std::nullopt, QString::fromUtf8(error.what())};
        }
    }));
}

void SpiralServiceManager::sendInputUpdate(QJsonObject request, qint64 requested, quint64 serial)
{
    emit inputUpdateProgress(QStringLiteral("linking"),
                             tr("Linking and pinning a position including %1…").arg(requested));
    QJsonObject run = request.value(QStringLiteral("run")).toObject();
    QJsonObject config = run.value(QStringLiteral("config")).toObject();
    for (const QString& key : _serviceOwnedKeys) config.remove(key);
    run[QStringLiteral("config")] = config;
    QJsonObject body{{QStringLiteral("run"), run},
                     {QStringLiteral("inputs_position"), requested},
                     {QStringLiteral("command_id"), commandId()}};
    const bool initialize = !_hasActiveSession;
    // An idle fit keeps training across the update; a failed one is rebuilt.
    // Any other state (a run, a build) is the user's to stop first: a
    // rebuild would discard the model the update means to keep.
    const bool update = !initialize && _sessionState == QStringLiteral("Idle");
    if (!initialize && !update && _sessionState != QStringLiteral("Error")) {
        InputUpdate busy;
        busy.requested = requested;
        busy.error = tr("The fit is %1: stop it, then update its inputs").arg(_sessionState);
        finishInputUpdate(busy, serial);
        return;
    }
    QString route = QStringLiteral("/session/initialize");
    if (update) {
        route = QStringLiteral("/session/update-inputs");
        body = {{QStringLiteral("inputs_position"), requested},
                {QStringLiteral("expected_input_generation"), _inputGeneration},
                {QStringLiteral("command_id"), commandId()}};
    } else if (!initialize) {
        route = QStringLiteral("/session/rebuild");
        body[QStringLiteral("inputs")] = QStringLiteral("reload");
    }
    auto failed = [this, requested, serial](const QString& message, const QJsonObject& body) {
        InputUpdate update;
        update.requested = requested;
        update.error = message;
        const QJsonObject failure = body.value(QStringLiteral("input_preparation_error")).toObject();
        for (const QJsonValue& problem : failure.value(QStringLiteral("problems")).toArray())
            update.problems << problem.toString();
        if (!body.isEmpty()) handleStatus(body);
        finishInputUpdate(update, serial);
    };
    // A build (not an update of the resident fit) reads the project again.
    if (!update) {
        if (const QString error = addProject(body); !error.isEmpty()) {
            failed(error, {});
            return;
        }
    }
    postWithRetry(route, body, Timeout::Load, kMutationRetries,
                  [this, requested, serial](const QJsonObject& response) {
                      handleStatus(response);
                      const QJsonObject position = response.value(QStringLiteral("input_position")).toObject();
                      InputUpdate update;
                      update.requested = requested;
                      update.head = position.value(QStringLiteral("head")).toInteger(-1);
                      update.fitted = position.value(QStringLiteral("sequence")).toInteger(-1);
                      if (update.fitted < 0) update.error = tr("The service reported no fitted position");
                      finishInputUpdate(update, serial);
                  },
                  {}, failed);
}

void SpiralServiceManager::finishInputUpdate(InputUpdate update, quint64 serial)
{
    if (serial != _inputUpdateSerial) return;  // abandoned
    // Pushes resume; then pull to (at least) the fitted position, so the
    // service's links and any edits it included are here. A shared store
    // holds them already.
    _inputUpdateHold.reset();
    _inputUpdateAttachment.reset();
    const auto session = _storeSession.lock();
    if (!update.error.isEmpty() || !session || !_storeAttachment) {
        _inputUpdateActive = false;
        emit inputUpdateFinished(update);
        return;
    }
    const bool hub = _storeBinding.hosted();
    if (hub)
        emit inputUpdateProgress(QStringLiteral("pulling"), tr("Pulling the fitted position %1…").arg(update.fitted));
    const vc::objects::AttachmentIdentity attachment = *_storeAttachment;
    auto* watcher = new QFutureWatcher<InputUpdate>(this);
    connect(watcher, &QFutureWatcher<InputUpdate>::finished, this, [this, watcher, serial]() {
        const InputUpdate result = watcher->result();
        watcher->deleteLater();
        if (serial != _inputUpdateSerial) return;  // abandoned
        _inputUpdateActive = false;
        refreshLocalHub();
        emit inputUpdateFinished(result);
    });
    watcher->setFuture(QtConcurrent::run([session, attachment, update, hub]() mutable {
        auto describe = [](const vc::objects::JournalEntry& entry) {
            return QStringLiteral("%1 %2 %3 (%4)").arg(QString::fromStdString(entry.changeKind),
                QString::fromStdString(entry.kind), QString::fromStdString(entry.recordUuid),
                QString::fromStdString(entry.author));
        };
        // The hub's journal, or a shared store's own.
        auto journal = [&](int64_t after, int64_t upTo) {
            if (hub) return session->hubJournal(attachment, after, upTo);
            std::vector<vc::objects::JournalEntry> entries;
            session->read(attachment, [&](const vc::objects::ObjectStore& store, const vc::objects::ReadTxn& txn,
                                          const std::atomic_bool&) { entries = store.journalSince(txn, after); });
            std::erase_if(entries, [&](const auto& entry) { return entry.seq > upTo; });
            return entries;
        };
        try {
            qint64 position = -1;
            if (hub) {
                if (const auto status = session->syncHubAndWait(attachment, std::chrono::seconds(120)))
                    position = status->position;
            } else {
                position = storeHead(*session, attachment);
            }
            if (update.head > update.requested)
                for (const auto& entry : journal(update.requested, update.head))
                    update.includedBeyondRequested << describe(entry);
            if (position > update.fitted)
                for (const auto& entry : journal(update.fitted, position))
                    update.afterFitted << describe(entry);
        } catch (const std::exception& error) {
            update.problems << QObject::tr("Pulling the fitted position failed: %1").arg(QString::fromUtf8(error.what()));
        }
        return update;
    }));
}

void SpiralServiceManager::rebuildWithDefaults()
{
    if (!isReady()) return;
    sendRebuildRequest({{QStringLiteral("defaults"), true},
                        {QStringLiteral("command_id"), commandId()}});
}

QString SpiralServiceManager::addProject(QJsonObject& body) const
{
    try {
        if (!_projectSource) return tr("No project is open");
        body[QStringLiteral("project")] = _projectSource();
        return {};
    } catch (const std::exception& error) {
        return tr("The project's Spiral inputs could not be read: %1").arg(QString::fromUtf8(error.what()));
    }
}

void SpiralServiceManager::sendRebuildRequest(QJsonObject request)
{
    if (const QString error = addProject(request); !error.isEmpty()) {
        emit errorOccurred(error);
        return;
    }
    postWithRetry(QStringLiteral("/session/rebuild"), request, Timeout::Load, kMutationRetries,
                  [this](const QJsonObject& response) {
                      handleStatus(response);
                  });
}

void SpiralServiceManager::sendInitializeRequest(QJsonObject request)
{
    if (const QString error = addProject(request); !error.isEmpty()) {
        emit errorOccurred(error);
        return;
    }
    postWithRetry(QStringLiteral("/session/initialize"), request, Timeout::Load,
                  kMutationRetries, [this](const QJsonObject& response) {
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
                                         const QJsonObject& runConfig,
                                         const QJsonObject& dtLossSchedule,
                                         const QJsonObject& previewSchedule)
{
    const QJsonObject configuration =
        vc3d::completeSpiralRunConfiguration(
            _configurationDefaults, _appliedConfiguration, runConfig);
    // Base inputs are owned and canonicalized by the service.  A Run changes
    // neither their paths nor their contents, so restating the panel's
    // necessarily partial path view here can only create a false mismatch
    // (for example, winding_inference has no editable panel row).
    QJsonObject body = vc3d::spiralRunRequest(
        configuration, iterations, dtLossSchedule,
        _sessionRevision, previewSchedule);
    body[QStringLiteral("command_id")] = commandId();
    postWithRetry(
        QStringLiteral("/session/run"),
        body,
        Timeout::Command, kMutationRetries, {});
}

void SpiralServiceManager::stopAfterIteration()
{
    postWithRetry(QStringLiteral("/session/stop"),
                  {{QStringLiteral("command_id"), commandId()}},
                  Timeout::Command, kMutationRetries, {});
}

void SpiralServiceManager::saveCheckpoint(const QString& name)
{
    // The client names the checkpoint; the service places it under the
    // session output directory and reports the resulting host path back
    // through the status snapshot.
    postWithRetry(QStringLiteral("/session/save-checkpoint"),
                  {{QStringLiteral("command_id"), commandId()},
                   {QStringLiteral("name"), name}},
                  Timeout::LongCommand, kMutationRetries,
                  [this](const QJsonObject& response) {
                      handleStatus(response);
                      // The saved file joins the checkpoints the service
                      // advertises, which is what a load offers to choose from.
                      fetchAdvertisedDataset();
                  });
}

void SpiralServiceManager::requestPreview()
{
    if (!isReady() || _previewRequestInFlight) return;
    _previewRequestInFlight = true;
    // The service accepts the export and returns; the work itself costs
    // minutes and is followed through the status snapshot this manager
    // already polls (preview_exporting, then preview_publish, then
    // preview_artifact or preview_publish_error, and for the overlays
    // preview_diagnostics_artifact after that).
    postWithRetry(QStringLiteral("/session/export-preview"),
                  {{QStringLiteral("command_id"), commandId()},
                   {QStringLiteral("diagnostics"), _previewDiagnosticsWanted}},
                  Timeout::Command, kMutationRetries,
                  [this](const QJsonObject& response) {
                      _previewRequestInFlight = false;
                      handleStatus(response);
                  },
                  [this](const QString& error) {
                      _previewRequestInFlight = false;
                      emit logMessage(tr("Preview export failed: %1").arg(error));
                  });
}

QStringList SpiralServiceManager::serviceCheckpoints() const
{
    QStringList paths;
    for (const QString& key : {QStringLiteral("session_checkpoints"),
                               QStringLiteral("detected_checkpoints")})
        for (const QJsonValue& value : _advertisedDataset.value(key).toArray())
            if (!value.toString().isEmpty()) paths.push_back(value.toString());
    return paths;
}

void SpiralServiceManager::loadCheckpoint(const QString& hostPath,
                                          const QString& localPath,
                                          bool allowRebuild)
{
    QJsonObject body;
    if (allowRebuild) body[QStringLiteral("allow_rebuild")] = true;
    if (localPath.isEmpty()) {
        body[QStringLiteral("host_checkpoint")] = hostPath;
        sendLoadCheckpoint(body, hostPath, localPath);
        return;
    }
    // A local file has to exist on the service before the service can resolve
    // it. The upload reuses an identical checkpoint already on the host, so
    // escalating a refusal to a rebuild re-sends nothing.
    emit logMessage(tr("Uploading checkpoint %1 to the service…").arg(localPath));
    uploadCheckpointForResume(
        localPath,
        [this, body, localPath](const QString& uploaded, const QString& error,
                                bool reused) mutable {
            if (uploaded.isEmpty()) {
                emit errorOccurred(tr("Checkpoint upload failed: %1").arg(error));
                return;
            }
            emit logMessage(reused
                                ? tr("Reusing checkpoint already on the service at %1").arg(uploaded)
                                : tr("Checkpoint uploaded to service path %1").arg(uploaded));
            body[QStringLiteral("uploaded_checkpoint")] = uploaded;
            sendLoadCheckpoint(body, {}, localPath);
        });
}

void SpiralServiceManager::sendLoadCheckpoint(QJsonObject body,
                                              const QString& hostPath,
                                              const QString& localPath)
{
    // Without allow_rebuild the resident model stays and only its parameters,
    // optimiser, scheduler and RNG state are replaced, and only if the
    // service's preflight finds the checkpoint an exact match. A refusal
    // carries the reasons and the rebuild that would accept it, which the
    // panel offers; with allow_rebuild the service performs that rebuild.
    // Which checkpoint this is the service resolves: the body only says
    // whether the name came from its own advertisement or from an upload.
    body[QStringLiteral("command_id")] = commandId();
    postWithRetry(QStringLiteral("/session/load-checkpoint"), body,
                  Timeout::Load, kMutationRetries,
                  [this](const QJsonObject& response) {
                      handleStatus(response);
                      // Preserve the behaviour the automatic
                      // checkpoint-resume preview used to give: after a load,
                      // show the restored model.
                      requestPreview();
                      emit checkpointLoaded(
                          response.value(QStringLiteral("checkpoint_path")).toString(),
                          response.value(QStringLiteral("restored_iteration"))
                              .toInteger(-1));
                  },
                  {},
                  [this, hostPath, localPath](const QString& message,
                                              const QJsonObject& refusal) {
                      QStringList reasons;
                      for (const QJsonValue& value :
                               refusal.value(QStringLiteral("reasons")).toArray())
                          reasons.push_back(value.toString());
                      emit checkpointLoadRefused(
                          hostPath, localPath, reasons,
                          refusal.value(QStringLiteral("stage")).toString(),
                          message);
                  });
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
                                         std::function<void(const QString&)> failure,
                                         DetailedFailure detailedFailure)
{
    if (!isReady()) {
        const QString message = tr("Spiral service is not connected");
        if (detailedFailure) detailedFailure(message, {});
        else if (failure) failure(message);
        else emit errorOccurred(message);
        return;
    }
    QNetworkRequest request = makeRequest(path, static_cast<int>(timeout));
    auto* reply = _network->post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, generation, path, body, timeout, retriesLeft, success,
             failure, detailedFailure]() {
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
                    QTimer::singleShot(1000, this, [this, generation, path, body, timeout, retriesLeft,
                                                    success, failure, detailedFailure]() {
                        if (generation != _connectionGeneration) return;
                        postWithRetry(path, body, timeout, retriesLeft - 1, success,
                                      failure, detailedFailure);
                    });
                    return;
                }
                handleReply(reply, generation, success, failure, detailedFailure);
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

void SpiralServiceManager::del(const QString& path, Timeout timeout,
                               std::function<void(const QJsonObject&)> success,
                               std::function<void(const QString&)> failure)
{
    QNetworkRequest request = makeRequest(path, static_cast<int>(timeout));
    auto* reply = _network->sendCustomRequest(request, "DELETE");
    const quint64 generation = _connectionGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation, success, failure]() {
        handleReply(reply, generation, success, failure);
    });
}

void SpiralServiceManager::handleReply(QNetworkReply* reply, quint64 generation,
                                       std::function<void(const QJsonObject&)> success,
                                       std::function<void(const QString&)> failure,
                                       DetailedFailure detailedFailure)
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
        if (detailedFailure) detailedFailure(message, {});
        else if (failure) failure(message);
        else emit errorOccurred(message);
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
        if (detailedFailure) detailedFailure(message, document.object());
        else if (failure) failure(message);
        else emit errorOccurred(message);
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
    // A session identity appears when explicit initialization starts. Until
    // then the service remains connected for dataset/checkpoint discovery.
    const bool active = !sessionId.isEmpty();
    if (active != _hasActiveSession) {
        _hasActiveSession = active;
        emit sessionActiveChanged(active);
    }
    _sessionState = status.value(QStringLiteral("state")).toString();
    _inputGeneration = status.value(QStringLiteral("input_generation")).toInteger();
    // The display follows the fitted position: a replica behind it pulls
    // forward (bringing the service's links); one ahead shows later edits
    // as stale through the input freshness. One sync is asked for per fitted
    // position; the session's periodic sync retries one that failed (an
    // unreachable hub), rather than every status poll.
    const qint64 fitted = status.value(QStringLiteral("input_position")).toObject()
                              .value(QStringLiteral("sequence")).toInteger(-1);
    if (_storeBinding.hosted() && fitted > _storeBinding.localPosition && fitted != _forwardSyncPosition
        && _storeBinding.matches() && !_inputUpdateActive)
        if (const auto session = _storeSession.lock(); session && _storeAttachment) {
            _forwardSyncPosition = fitted;
            session->syncHub(*_storeAttachment);
        }
    if (!active) {
        _synchronizedSessionId.clear();
        _installedPreviewSession.clear();
        _displayedPreviewSourceIteration = -1;
    } else if (sessionId != _synchronizedSessionId) {
        _displayedPreviewSourceIteration = -1;
        const QJsonObject request =
            status.value(QStringLiteral("session_request")).toObject();
        if (!request.isEmpty()) {
            _synchronizedSessionId = sessionId;
            emit sessionSynchronized(request, status);
        }
    }
    // Pausing no longer exports a preview by itself. Ask for one exactly
    // where the automatic pause preview used to appear: the first Idle after
    // a run.
    const QString state = status.value(QStringLiteral("state")).toString();
    if (state == QStringLiteral("Running")) {
        _sawRunningSinceIdle = true;
    } else if (state == QStringLiteral("Idle") && _sawRunningSinceIdle) {
        _sawRunningSinceIdle = false;
        // A service-owned schedule captures the final Run boundary itself.
        // Do not enqueue the legacy client-triggered duplicate.
        if (status.value(QStringLiteral("preview_schedule")).toObject().isEmpty())
            requestPreview();
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
                QFile manifestFile(entryPath);
                if (manifestFile.open(QIODevice::ReadOnly)) {
                    const QJsonObject manifest =
                        QJsonDocument::fromJson(manifestFile.readAll()).object();
                    const auto provenance =
                        vc3d::spiralPreviewProvenance(manifest);
                    _displayedPreviewSourceIteration = provenance.sourceIteration;
                    _displayedPreviewInputDigest = provenance.inputDigest;
                    _displayedPreviewInputPosition = provenance.inputPosition;
                } else {
                    _displayedPreviewSourceIteration = -1;
                    _displayedPreviewInputDigest.clear();
                    _displayedPreviewInputPosition = -1;
                }
                // A newly installed surface has no overlays until its own
                // diagnostics artifact arrives; the previous generation's
                // must not be drawn over it.
                _installedDiagnosticsArtifact.clear();
                emit previewAvailable(entryPath, sequence);
                _artifactCache->pruneSession(
                    sessionId, kPreviewCacheKept, artifactCachePins());
            });
    }

    // The overlays are published as a second artifact once the surface is on
    // its way, so they are fetched separately and only reach the workspace
    // when they belong to the preview it has installed.
    const QString diagnosticsId =
        status.value(QStringLiteral("preview_diagnostics_artifact")).toObject()
            .value(QStringLiteral("id")).toString();
    if (!diagnosticsId.isEmpty() && diagnosticsId != _installedDiagnosticsArtifact
        && diagnosticsId != _fetchingDiagnosticsArtifact) {
        _fetchingDiagnosticsArtifact = diagnosticsId;
        const qint64 sequence = _previewSequence;
        const quint64 generation = _connectionGeneration;
        _artifactCache->fetchArtifact(
            sessionId, diagnosticsId,
            [this, diagnosticsId, sequence, sessionId, generation](
                const QString& entryPath, const QString& error, bool gone) {
                if (generation != _connectionGeneration) return;
                if (_fetchingDiagnosticsArtifact == diagnosticsId)
                    _fetchingDiagnosticsArtifact.clear();
                if (entryPath.isEmpty()) {
                    if (!gone) emit errorOccurred(error);
                    return;
                }
                // The surface these overlays were mapped onto has already
                // been replaced; the next generation publishes its own.
                if (sequence < _previewSequence) return;
                _installedDiagnosticsArtifact = diagnosticsId;
                _lastDiagnosticsLocalPath = entryPath;
                emit previewDiagnosticsAvailable(entryPath, sequence);
                _artifactCache->pruneSession(
                    sessionId, kPreviewCacheKept, artifactCachePins());
            });
    }

    // The frozen inputs' manifest: one immutable artifact per session, from
    // which the workspace reads the fitted inputs at their pinned revisions.
    const QJsonObject manifestRef =
        status.value(QStringLiteral("input_manifest_artifact")).toObject();
    const QString manifestId = manifestRef.value(QStringLiteral("id")).toString();
    if (!manifestId.isEmpty() && manifestId != _installedInputManifest
        && manifestId != _fetchingInputManifest) {
        _fetchingInputManifest = manifestId;
        const QString digest = manifestRef.value(QStringLiteral("input_digest")).toString();
        const quint64 generation = _connectionGeneration;
        _artifactCache->fetchArtifact(
            sessionId, manifestId,
            [this, sessionId, manifestId, digest, generation](
                const QString& entryPath, const QString& error, bool gone) {
                if (generation != _connectionGeneration) return;
                if (_fetchingInputManifest == manifestId) _fetchingInputManifest.clear();
                if (entryPath.isEmpty()) {
                    if (!gone) emit errorOccurred(error);
                    return;
                }
                _installedInputManifest = manifestId;
                _lastInputManifestLocalPath = entryPath;
                emit inputManifestAvailable(entryPath, digest);
                _artifactCache->pruneSession(
                    sessionId, kPreviewCacheKept, artifactCachePins());
            });
    }
}

QStringList SpiralServiceManager::artifactCachePins() const
{
    return vc3d::spiralArtifactCachePins(
        _lastPreviewLocalPath, _lastDiagnosticsLocalPath,
        {_lastInputManifestLocalPath});
}

void SpiralServiceManager::fetchPreviewFile(const QString& relativeName,
                                            FetchPreviewFileCallback done)
{
    // Deferred preview files are the loss overlays, and those live in the
    // diagnostics artifact the service publishes after the surface.
    if (_installedDiagnosticsArtifact.isEmpty() || _installedPreviewSession.isEmpty()) {
        done({}, tr("No Spiral preview diagnostics are installed"));
        return;
    }
    const QString artifactId = _installedDiagnosticsArtifact;
    const QString sessionId = _installedPreviewSession;
    const quint64 generation = _connectionGeneration;
    _artifactCache->fetchFile(
        sessionId, artifactId, relativeName,
        [this, artifactId, generation, done = std::move(done)](
            const QString& localPath, const QString& error, bool gone) {
            if (generation != _connectionGeneration
                || artifactId != _installedDiagnosticsArtifact)
                return;
            done(localPath,
                 gone ? tr("The Spiral preview was pruned before the file was downloaded")
                      : error);
        });
}
