#include "agent_bridge/AgentBridgeServer.hpp"

#include "CWindow.hpp"
#include "SpiralServiceManager.hpp"
#include "SpiralServiceProfile.hpp"
#include "SpiralWorkspace.hpp"
#include "VCSettings.hpp"

#include <QFileInfo>
#include <QJsonArray>
#include <QRegularExpression>
#include <QSettings>
#include <QUrl>

#include <memory>

namespace {

QString connectionStateName(SpiralServiceManager::ConnectionState state)
{
    using State = SpiralServiceManager::ConnectionState;
    switch (state) {
    case State::Disconnected: return QStringLiteral("disconnected");
    case State::Starting: return QStringLiteral("starting");
    case State::Connecting: return QStringLiteral("connecting");
    case State::Ready: return QStringLiteral("ready");
    case State::Reconnecting: return QStringLiteral("reconnecting");
    case State::Failed: return QStringLiteral("failed");
    }
    return QStringLiteral("unknown");
}

bool safeIdentifier(const QString& value)
{
    static const QRegularExpression pattern(
        QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"));
    return pattern.match(value).hasMatch();
}

bool safeDirectEndpoint(const QUrl& endpoint)
{
    const QString scheme = endpoint.scheme().toLower();
    return endpoint.isValid()
        && (scheme == QLatin1String("http") || scheme == QLatin1String("https"))
        && !endpoint.host().isEmpty()
        && endpoint.userInfo().isEmpty()
        && endpoint.query().isEmpty()
        && endpoint.fragment().isEmpty();
}

SpiralServiceProfile loadProfile(const QString& profileId)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (profileId == QLatin1String("localhost"))
        return SpiralServiceProfile::localhostProfile(&settings);
    const QStringList ids =
        settings.value(QStringLiteral("spiral/profile_ids")).toStringList();
    if (!ids.contains(profileId))
        return {};
    return SpiralServiceProfile::load(settings, profileId);
}

QJsonObject profileJson(const SpiralServiceProfile& profile)
{
    const bool ssh =
        profile.transport == SpiralServiceProfile::Transport::SshTunnel;
    QJsonObject value{
        {QStringLiteral("profileId"), profile.id},
        {QStringLiteral("name"), profile.name},
        {QStringLiteral("mode"),
         profile.autoLaunch ? QStringLiteral("local")
                            : ssh ? QStringLiteral("ssh")
                                  : QStringLiteral("direct")},
        {QStringLiteral("credentialSource"),
         profile.autoLaunch ? QStringLiteral("generated")
                            : ssh ? QStringLiteral("ssh_auto")
                                  : QStringLiteral("environment")},
    };
    if (!profile.autoLaunch && !ssh)
        value[QStringLiteral("endpoint")] = profile.baseUrl.toString();
    if (ssh) {
        value[QStringLiteral("sshDestination")] = profile.sshDestination;
        value[QStringLiteral("remoteServicePort")] = profile.remoteServicePort;
    }
    return value;
}

void requireReady(SpiralServiceManager* service)
{
    if (!service->isReady()) {
        throw AgentBridgeError{
            -32005, "Spiral service is not connected",
            {{QStringLiteral("connectionState"),
              connectionStateName(service->connectionState())}}};
    }
}

} // namespace

SpiralServiceManager* AgentBridgeServer::spiralService() const
{
    if (!_window || !_window->_spiralWorkspace
        || !_window->_spiralWorkspace->serviceManager()) {
        throw AgentBridgeError{
            -32005,
            "Spiral workspace is unavailable; open a project first",
            {}};
    }
    return _window->_spiralWorkspace->serviceManager();
}

QJsonObject AgentBridgeServer::handleSpiralStatus(const QJsonValue&)
{
    SpiralServiceManager* service = spiralService();
    const SpiralServiceProfile profile = service->profile();
    QJsonObject result{
        {QStringLiteral("available"), true},
        {QStringLiteral("apiVersion"), SpiralServiceManager::kApiVersion},
        {QStringLiteral("connectionState"),
         connectionStateName(service->connectionState())},
        {QStringLiteral("ready"), service->isReady()},
        {QStringLiteral("hasActiveSession"), service->hasActiveSession()},
        {QStringLiteral("ownedLocalService"), service->ownsProcess()},
    };
    if (!profile.id.isEmpty())
        result[QStringLiteral("endpoint")] = profileJson(profile);
    return result;
}

QJsonObject AgentBridgeServer::handleSpiralProfiles(const QJsonValue&)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QStringList ids =
        settings.value(QStringLiteral("spiral/profile_ids")).toStringList();
    if (!ids.contains(QStringLiteral("localhost")))
        ids.prepend(QStringLiteral("localhost"));
    QJsonArray profiles;
    for (const QString& id : ids)
        profiles.append(profileJson(loadProfile(id)));
    return {
        {QStringLiteral("profiles"), profiles},
        {QStringLiteral("selectedProfileId"),
         settings.value(QStringLiteral("spiral/selected_profile"),
                        QStringLiteral("localhost")).toString()},
    };
}

QJsonObject AgentBridgeServer::handleSpiralConnect(const QJsonValue& params)
{
    const QString profileId =
        params.toObject().value(QStringLiteral("profileId")).toString();
    SpiralServiceProfile profile = loadProfile(profileId);
    if (profile.id.isEmpty())
        throw AgentBridgeError{-32007, "Unknown Spiral profile",
                               {{QStringLiteral("profileId"), profileId}}};
    if (!profile.autoLaunch
        && profile.transport == SpiralServiceProfile::Transport::Direct) {
        if (!safeDirectEndpoint(profile.baseUrl))
            throw AgentBridgeError{-32602, "Saved Spiral endpoint is invalid", {}};
        profile.apiKey = qEnvironmentVariable("SPIRAL_API_KEY");
        if (profile.apiKey.isEmpty())
            throw AgentBridgeError{
                -32005, "SPIRAL_API_KEY is required for a direct profile", {}};
    }
    if (profile.transport == SpiralServiceProfile::Transport::SshTunnel
        && (profile.sshDestination.trimmed().isEmpty()
            || profile.remoteServicePort <= 0
            || profile.remoteServicePort > 65535)) {
        throw AgentBridgeError{-32602, "Saved Spiral SSH profile is invalid", {}};
    }
    spiralService()->connectToService(profile);
    return {{QStringLiteral("accepted"), true},
            {QStringLiteral("profileId"), profile.id},
            {QStringLiteral("connectionState"), QStringLiteral("starting")}};
}

QJsonObject AgentBridgeServer::handleSpiralDisconnect(const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    const bool force = params.toObject().value(QStringLiteral("force")).toBool();
    if (service->ownsProcess() && !force) {
        throw AgentBridgeError{
            -32010,
            "Disconnecting an owned local service requires force=true",
            {}};
    }
    service->disconnectFromService();
    return {{QStringLiteral("disconnected"), true}};
}

QJsonObject AgentBridgeServer::handleSpiralReconnect(const QJsonValue&)
{
    SpiralServiceManager* service = spiralService();
    if (service->profile().id.isEmpty())
        throw AgentBridgeError{-32004, "No previous Spiral profile", {}};
    service->reconnect();
    return {{QStringLiteral("accepted"), true},
            {QStringLiteral("connectionState"), QStringLiteral("reconnecting")}};
}

QJsonObject AgentBridgeServer::handleSpiralDataset(const QJsonValue&)
{
    const QJsonObject dataset = spiralService()->advertisedDataset();
    return {{QStringLiteral("available"), !dataset.isEmpty()},
            {QStringLiteral("dataset"), dataset}};
}

QJsonObject AgentBridgeServer::handleSpiralRebuild(const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const QJsonObject request = params.toObject();
    if (!request.value(QStringLiteral("confirm")).toBool())
        throw AgentBridgeError{-32602, "rebuild requires confirm=true", {}};
    const bool defaults = request.value(QStringLiteral("defaults")).toBool();
    const bool hasRequest = request.contains(QStringLiteral("request"));
    if (defaults == hasRequest)
        throw AgentBridgeError{
            -32602, "Specify exactly one of defaults=true or request", {}};
    if (defaults)
        service->rebuildWithDefaults();
    else
        service->rebuildSession(request.value(QStringLiteral("request")).toObject());
    return {{QStringLiteral("accepted"), true}};
}

QJsonObject AgentBridgeServer::handleSpiralRun(const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const QJsonObject request = params.toObject();
    const int iterations = request.value(QStringLiteral("iterations")).toInt();
    if (iterations <= 0)
        throw AgentBridgeError{-32602, "iterations must be positive", {}};
    service->runIterations(
        iterations,
        request.value(QStringLiteral("influenceConfig")).toObject(),
        request.value(QStringLiteral("runConfig")).toObject(),
        request.value(QStringLiteral("inputs")).toObject());
    return {{QStringLiteral("accepted"), true},
            {QStringLiteral("iterations"), iterations}};
}

QJsonObject AgentBridgeServer::handleSpiralStop(const QJsonValue&)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    if (!service->hasActiveSession())
        throw AgentBridgeError{-32004, "No active Spiral session", {}};
    service->stopAfterIteration();
    return {{QStringLiteral("accepted"), true}};
}

QJsonObject AgentBridgeServer::handleSpiralPreviewExport(const QJsonValue&)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    if (!service->hasActiveSession())
        throw AgentBridgeError{-32004, "No active Spiral session", {}};
    service->requestPreview();
    return {{QStringLiteral("accepted"), true}};
}

QJsonObject AgentBridgeServer::handleSpiralCheckpointSave(const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const QString name =
        params.toObject().value(QStringLiteral("name")).toString();
    if (!safeIdentifier(name))
        throw AgentBridgeError{-32602, "Checkpoint name is not a safe identifier", {}};
    service->saveCheckpoint(name);
    return {{QStringLiteral("accepted"), true},
            {QStringLiteral("name"), name}};
}

QJsonObject AgentBridgeServer::handleSpiralCheckpointDownload(
    const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const QFileInfo destination(
        params.toObject().value(QStringLiteral("localPath")).toString());
    if (!destination.isAbsolute() || !QFileInfo(destination.absolutePath()).isDir())
        throw AgentBridgeError{
            -32007,
            "localPath must be absolute and its parent directory must exist",
            {}};
    const int token = beginDeferred(600000, "Spiral checkpoint download");
    connect(service, &SpiralServiceManager::checkpointDownloadFinished, this,
            [this, token](const QString& path, const QString& error) {
                if (!error.isEmpty()) {
                    completeDeferredError(token, -32005,
                                          "Checkpoint download failed",
                                          {{QStringLiteral("detail"), error}});
                    return;
                }
                completeDeferredResult(
                    token, {{QStringLiteral("localPath"), path}});
            },
            Qt::SingleShotConnection);
    service->downloadCheckpoint(destination.absoluteFilePath());
    throw AgentBridgeDeferred{};
}

QJsonObject AgentBridgeServer::handleSpiralCheckpointLoad(
    const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const QJsonObject request = params.toObject();
    const QString hostPath = request.value(QStringLiteral("hostPath")).toString();
    const QString localPath = request.value(QStringLiteral("localPath")).toString();
    if (hostPath.isEmpty() == localPath.isEmpty())
        throw AgentBridgeError{
            -32602, "Specify exactly one of hostPath or localPath", {}};
    if (!hostPath.isEmpty() && !service->serviceCheckpoints().contains(hostPath))
        throw AgentBridgeError{-32007, "hostPath is not service-advertised", {}};
    if (!localPath.isEmpty()
        && (!QFileInfo(localPath).isAbsolute() || !QFileInfo(localPath).isFile()))
        throw AgentBridgeError{-32007, "localPath must be an existing absolute file", {}};

    const int token = beginDeferred(600000, "Spiral checkpoint load");
    auto completed = std::make_shared<bool>(false);
    connect(service, &SpiralServiceManager::checkpointLoaded, this,
            [this, token, completed](const QString& path, qint64 iteration) {
                if (*completed) return;
                *completed = true;
                completeDeferredResult(
                    token,
                    {{QStringLiteral("hostPath"), path},
                     {QStringLiteral("restoredIteration"), iteration}});
            });
    connect(service, &SpiralServiceManager::checkpointLoadRefused, this,
            [this, token, completed](const QString&, const QString&,
                                     const QStringList& reasons,
                                     const QString& stage,
                                     const QString& message) {
                if (*completed) return;
                *completed = true;
                completeDeferredError(
                    token, -32010, "Checkpoint load refused",
                    {{QStringLiteral("detail"), message},
                     {QStringLiteral("reasons"), QJsonArray::fromStringList(reasons)},
                     {QStringLiteral("rebuildStage"), stage}});
            });
    service->loadCheckpoint(
        hostPath, localPath,
        request.value(QStringLiteral("allowRebuild")).toBool(false));
    throw AgentBridgeDeferred{};
}

QJsonObject AgentBridgeServer::handleSpiralInputUpload(const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const QJsonObject request = params.toObject();
    const QString kind = request.value(QStringLiteral("kind")).toString();
    const QString path = request.value(QStringLiteral("localPath")).toString();
    const QString inputId = request.value(QStringLiteral("inputId")).toString();
    const QString role = request.value(QStringLiteral("role")).toString();
    if (!safeIdentifier(inputId))
        throw AgentBridgeError{-32602, "inputId is not a safe identifier", {}};
    const QFileInfo source(path);
    const bool validSource = source.isAbsolute()
        && (kind == QLatin1String("patch") ? source.isDir() : source.isFile());
    if (!validSource)
        throw AgentBridgeError{-32007, "localPath is not a compatible absolute input", {}};
    if (kind != QLatin1String("pcl") && !role.isEmpty())
        throw AgentBridgeError{-32602, "role is valid only for PCL input", {}};

    const int token = beginDeferred(600000, "Spiral input upload");
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(
        service, &SpiralServiceManager::inputUploadFinished, this,
        [this, token, inputId, connection](const QString& finishedId,
                                           const QString& error) {
            if (finishedId != inputId) return;
            disconnect(*connection);
            if (!error.isEmpty()) {
                completeDeferredError(token, -32005, "Input upload failed",
                                      {{QStringLiteral("detail"), error}});
                return;
            }
            completeDeferredResult(
                token, {{QStringLiteral("inputId"), inputId}});
        });
    if (kind == QLatin1String("patch"))
        service->uploadPatch(source.absoluteFilePath(), inputId);
    else
        service->uploadJsonInput(kind, source.absoluteFilePath(), inputId, role);
    throw AgentBridgeDeferred{};
}

QJsonObject AgentBridgeServer::handleSpiralInputRemove(const QJsonValue& params)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const QJsonObject request = params.toObject();
    const QString inputId = request.value(QStringLiteral("inputId")).toString();
    if (!safeIdentifier(inputId))
        throw AgentBridgeError{-32602, "inputId is not a safe identifier", {}};
    service->removeEphemeralInput(
        request.value(QStringLiteral("kind")).toString(), inputId);
    return {{QStringLiteral("accepted"), true},
            {QStringLiteral("inputId"), inputId}};
}

QJsonObject AgentBridgeServer::handleSpiralInputsCommit(const QJsonValue&)
{
    SpiralServiceManager* service = spiralService();
    requireReady(service);
    const int token = beginDeferred(240000, "Spiral input commit");
    connect(service, &SpiralServiceManager::commitInputsFinished, this,
            [this, token](const QStringList& committed, const QString& error) {
                if (!error.isEmpty()) {
                    completeDeferredError(token, -32005, "Input commit failed",
                                          {{QStringLiteral("detail"), error}});
                    return;
                }
                completeDeferredResult(
                    token,
                    {{QStringLiteral("committed"),
                      QJsonArray::fromStringList(committed)}});
            },
            Qt::SingleShotConnection);
    service->commitInputs();
    throw AgentBridgeDeferred{};
}
