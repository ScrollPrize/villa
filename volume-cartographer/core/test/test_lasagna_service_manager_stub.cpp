#include "LasagnaServiceManager.hpp"

#include <QJsonArray>
#include <QJsonObject>

QJsonObject g_lastLasagnaOptimizationRequest;

LasagnaServiceManager& LasagnaServiceManager::instance()
{
    static LasagnaServiceManager manager;
    return manager;
}

LasagnaServiceManager* LasagnaServiceManager::createTransient(QObject* parent)
{
    return new LasagnaServiceManager(parent, true);
}

LasagnaServiceManager::LasagnaServiceManager(QObject* parent, bool containProcessTree)
    : QObject(parent)
    , _containProcessTree(containProcessTree)
{
}

LasagnaServiceManager::~LasagnaServiceManager() = default;

bool LasagnaServiceManager::ensureServiceRunning(const QString&, const QString&)
{
    _serviceReady = true;
    return true;
}

void LasagnaServiceManager::connectToExternal(const QString&, int)
{
}

void LasagnaServiceManager::stopService()
{
    _serviceReady = false;
    emit serviceStopped();
}

bool LasagnaServiceManager::isRunning() const
{
    return _serviceReady;
}

void LasagnaServiceManager::startOptimization(const QJsonObject& request)
{
    g_lastLasagnaOptimizationRequest = request;
    emit optimizationStarted();
}

void LasagnaServiceManager::setObjectStore(std::shared_ptr<vc::objects::ObjectStoreSession> session,
                                           std::optional<vc::objects::AttachmentIdentity> attachment)
{
    _storeSession = std::move(session);
    _storeAttachment = attachment;
}

std::optional<std::filesystem::path> LasagnaServiceManager::surfaceModel(const std::string& surfaceUuid) const
{
    if (!_storeSession || !_storeAttachment) {
        return std::nullopt;
    }
    const auto model = _storeSession->artifactDirectory(*_storeAttachment, surfaceUuid, false) / "model.pt";
    return std::filesystem::is_regular_file(model) ? std::optional(model) : std::nullopt;
}

void LasagnaServiceManager::stopOptimization()
{
}

void LasagnaServiceManager::cancelJob(const QString&)
{
}

void LasagnaServiceManager::moveJobBefore(const QString&, const QString&)
{
}

void LasagnaServiceManager::moveJobToEnd(const QString&)
{
}

void LasagnaServiceManager::fetchJobs()
{
    emit jobsUpdated(QJsonArray{});
}

void LasagnaServiceManager::exportLasagnaVis(const QJsonObject&)
{
}

QJsonArray LasagnaServiceManager::discoverServices()
{
    return {};
}

QString LasagnaServiceManager::findConfigFile(const QString&)
{
    return {};
}

void LasagnaServiceManager::fetchDatasets()
{
    emit datasetsReceived(QJsonArray{});
}

void LasagnaServiceManager::rankLaplaceSnapPairs(
    const QJsonObject&,
    std::function<void(const QJsonObject&)> onSuccess,
    std::function<void(const QString&)>,
    std::function<void(int, const QJsonObject&)>)
{
    if (onSuccess) {
        onSuccess(QJsonObject{{QStringLiteral("results"), QJsonArray{}}});
    }
}
