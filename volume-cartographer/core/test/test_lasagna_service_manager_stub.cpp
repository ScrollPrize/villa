#include "LasagnaServiceManager.hpp"

#include <QJsonArray>
#include <QJsonObject>

QJsonObject g_lastLasagnaOptimizationRequest;
QString g_lastLasagnaOptimizationOutputDir;

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

void LasagnaServiceManager::startOptimization(const QJsonObject& request, const QString& outputDir)
{
    g_lastLasagnaOptimizationRequest = request;
    g_lastLasagnaOptimizationOutputDir = outputDir;
    emit optimizationStarted();
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

QJsonObject LasagnaServiceManager::makeTifxyzArtifactUpload(const QString&)
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
