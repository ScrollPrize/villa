#include "LasagnaServiceManager.hpp"

#include <QJsonArray>
#include <QJsonObject>

QJsonObject g_lastLasagnaOptimizationRequest;
QString g_lastLasagnaOptimizationOutputDir;
double g_lastLasagnaOptimizationOutputScale = 1.0;

LasagnaServiceManager& LasagnaServiceManager::instance()
{
    static LasagnaServiceManager manager;
    return manager;
}

LasagnaServiceManager::LasagnaServiceManager(QObject* parent)
    : QObject(parent)
{
}

LasagnaServiceManager::~LasagnaServiceManager() = default;

bool LasagnaServiceManager::ensureServiceRunning(const QString&)
{
    _serviceReady = true;
    return true;
}

void LasagnaServiceManager::connectToExternal(const QString&, int)
{
}

void LasagnaServiceManager::stopService()
{
}

bool LasagnaServiceManager::isRunning() const
{
    return _serviceReady;
}

void LasagnaServiceManager::startOptimization(const QJsonObject& request,
                                               const QString& outputDir,
                                               double outputScaleFactor)
{
    g_lastLasagnaOptimizationRequest = request;
    g_lastLasagnaOptimizationOutputDir = outputDir;
    g_lastLasagnaOptimizationOutputScale = outputScaleFactor;
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

void LasagnaServiceManager::fetchDatasets()
{
    emit datasetsReceived(QJsonArray{});
}
