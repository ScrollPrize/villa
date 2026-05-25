#include "LasagnaServiceManager.hpp"

#include <QJsonArray>

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

void LasagnaServiceManager::startOptimization(const QJsonObject&, const QString&)
{
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
