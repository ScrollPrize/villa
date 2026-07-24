#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QBuffer>
#include <QByteArray>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFutureWatcher>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMdiSubWindow>
#include <QPixmap>
#include <QPointF>
#include <QTabWidget>
#include <QTimer>
#include <QVector3D>
#include <QWidget>
#include <QtConcurrent>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <unordered_set>

#include "CWindow.hpp"
#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "LasagnaServiceManager.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "MenuActionController.hpp"
#include "OpenDataManifest.hpp"
#include "OpenDataSampleProject.hpp"
#include "OpenDataSegmentCache.hpp"
#include "SeedingWidget.hpp"
#include "SegmentationCommandHandler.hpp"
#include "SurfacePanelController.hpp"
#include "CommandLineToolRunner.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationPushPullConfig.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/tools/SegmentationPushPullTool.hpp"
#include "segmentation/panels/SegmentationLasagnaPanel.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"


// --- Lasagna RPCs + workspace switching ---

QJsonObject AgentBridgeServer::handleLasagnaServiceStatus(const QJsonValue&)
{
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();
    QJsonObject result;
    result["running"] = mgr.isRunning();
    result["external"] = mgr.isExternal();
    result["host"] = mgr.host();
    result["port"] = mgr.port();
    const QString err = mgr.lastError();
    result["lastError"] = err.isEmpty() ? QJsonValue() : QJsonValue(err);
    return result;
}

QJsonObject AgentBridgeServer::handleLasagnaEnsureService(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const bool hasHost = p.contains("host");
    const bool hasPort = p.contains("port");
    if (hasHost != hasPort) {
        QJsonObject data;
        data["detail"] = "host and port must be given together";
        throw AgentBridgeError{-32602, "host/port mismatch", data};
    }

    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();

    if (hasHost && hasPort) {
        // External mode: connectToExternal pings GET /health asynchronously;
        // completion is deferred on serviceStarted/serviceError.
        const QString host = p.value("host").toString();
        const int port = p.value("port").toInt();
        const int token = beginDeferred(15000, "Lasagna external service connect");
        connect(&mgr, &LasagnaServiceManager::serviceStarted, this,
                [this, token]() {
                    LasagnaServiceManager& m = LasagnaServiceManager::instance();
                    QJsonObject result;
                    result["running"] = true;
                    result["external"] = m.isExternal();
                    result["host"] = m.host();
                    result["port"] = m.port();
                    completeDeferredResult(token, result);
                }, Qt::SingleShotConnection);
        connect(&mgr, &LasagnaServiceManager::serviceError, this,
                [this, token](const QString& message) {
                    QJsonObject data;
                    data["detail"] = message;
                    completeDeferredError(token, -32005,
                                          "Lasagna service connection failed", data);
                }, Qt::SingleShotConnection);
        mgr.connectToExternal(host, port);
        throw AgentBridgeDeferred{};
    }

    // Internal mode: ensureServiceRunning() blocks until the process is up (or
    // fails) and returns synchronously -- no deferral needed.
    const QString pythonPath = p.value("pythonPath").toString();
    if (!mgr.ensureServiceRunning(pythonPath)) {
        QJsonObject data;
        data["detail"] = mgr.lastError();
        throw AgentBridgeError{-32005, "Failed to start lasagna service", data};
    }
    QJsonObject result;
    result["running"] = true;
    result["external"] = mgr.isExternal();
    result["host"] = mgr.host();
    result["port"] = mgr.port();
    return result;
}


QJsonObject AgentBridgeServer::handleLasagnaListDatasets(const QJsonValue&)
{
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();
    if (!mgr.isRunning()) {
        QJsonObject data;
        data["detail"] = "lasagna service is not running";
        throw AgentBridgeError{-32005, "Lasagna service not running", data};
    }
    const int token = beginDeferred(10000, "Lasagna datasets fetch");
    connect(&mgr, &LasagnaServiceManager::datasetsReceived, this,
            [this, token](const QJsonArray& datasets) {
                QJsonObject result;
                result["datasets"] = datasets;
                completeDeferredResult(token, result);
            }, Qt::SingleShotConnection);
    mgr.fetchDatasets();
    throw AgentBridgeDeferred{};
}


QJsonObject AgentBridgeServer::handleLasagnaStartOptimization(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString modeStr = p.value("mode").toString();
    SegmentationLasagnaPanel::LasagnaMode mode;
    if (modeStr == QLatin1String("reoptimize"))
        mode = SegmentationLasagnaPanel::ReOptimize;
    else if (modeStr == QLatin1String("new_model"))
        mode = SegmentationLasagnaPanel::NewModel;
    else if (modeStr == QLatin1String("offset"))
        mode = SegmentationLasagnaPanel::Offset;
    else
        mode = SegmentationLasagnaPanel::Atlas;

    requireSourceIdle(QStringLiteral("lasagna"));

    SegmentationWidget* widget = _window->_segmentationWidget;
    SegmentationLasagnaPanel* panel = widget ? widget->lasagnaPanel() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "lasagna panel is not available";
        throw AgentBridgeError{-32009, "Lasagna panel unavailable", data};
    }

    const QString configPath = p.value("configPath").toString();
    const QString atlasPath = p.value("atlasPath").toString();
    std::optional<cv::Vec3i> seed;
    if (p.contains("seed")) {
        const cv::Vec3f v = jsonToVec3(p.value("seed"), "seed");
        seed = cv::Vec3i(qRound(v[0]), qRound(v[1]), qRound(v[2]));
    }

    QString errorMessage;
    const bool started =
        panel->startOptimizationHeadless(state, mode, configPath, seed, atlasPath, &errorMessage);
    if (!started) {
        QJsonObject data;
        data["detail"] = errorMessage;
        if (errorMessage.contains(QLatin1String("config"), Qt::CaseInsensitive)) {
            data["kind"] = "config";
            throw AgentBridgeError{-32007, "Lasagna config not found", data};
        }
        if (errorMessage.contains(QLatin1String("atlas"), Qt::CaseInsensitive)) {
            data["kind"] = "atlas";
            throw AgentBridgeError{-32007, "No atlas selected", data};
        }
        throw AgentBridgeError{-32005, "Lasagna optimization failed to start", data};
    }

    // The optimizationStarted/jobStarted signals may already have fired
    // synchronously (direct connection, same thread) and registered this as
    // an external job via handleLasagnaStarted/handleLasagnaJobStarted --
    // reuse that record rather than double-registering.
    QString jobId;
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it != _activeJobs.end()) {
        it.value().kind = QStringLiteral("lasagna.optimize");
        jobId = it.value().id;
    } else {
        jobId = beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.optimize"),
                          QStringLiteral("Lasagna optimization started"),
                          /*broadcastStart=*/true);
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "lasagna.optimize";
    result["source"] = "lasagna";
    return result;
}


QJsonObject AgentBridgeServer::handleLasagnaJobs(const QJsonValue&)
{
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();
    if (!mgr.isRunning()) {
        QJsonObject data;
        data["detail"] = "lasagna service is not running";
        throw AgentBridgeError{-32005, "Lasagna service not running", data};
    }
    const int token = beginDeferred(10000, "Lasagna jobs fetch");
    connect(&mgr, &LasagnaServiceManager::jobsUpdated, this,
            [this, token](const QJsonArray& jobs) {
                QJsonObject result;
                result["jobs"] = jobs;
                completeDeferredResult(token, result);
            }, Qt::SingleShotConnection);
    mgr.fetchJobs();
    throw AgentBridgeDeferred{};
}


QJsonObject AgentBridgeServer::handleLasagnaCancel(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();

    QString serviceJobId;
    if (!p.contains("jobId")) {
        if (!jobIsRunning(QStringLiteral("lasagna"))) {
            QJsonObject data;
            data["kind"] = "job";
            throw AgentBridgeError{-32007, "No active lasagna job", data};
        }
        if (!mgr.isRunning()) {
            QJsonObject data;
            data["detail"] = "lasagna service is not running";
            throw AgentBridgeError{-32005, "Lasagna service not running", data};
        }
        mgr.stopOptimization();
    } else {
        const QString jobId = p.value("jobId").toString();
        if (jobId.startsWith(QLatin1String("job-"))) {
            const JobRecord* job = jobById(jobId);
            if (!job) {
                QJsonObject data;
                data["kind"] = "job";
                data["id"] = jobId;
                throw AgentBridgeError{-32007, QStringLiteral("Unknown job id: %1").arg(jobId),
                                       data};
            }
            serviceJobId = job->externalId;
        } else {
            serviceJobId = jobId;  // raw service job id passthrough.
        }
        if (!mgr.isRunning()) {
            QJsonObject data;
            data["detail"] = "lasagna service is not running";
            throw AgentBridgeError{-32005, "Lasagna service not running", data};
        }
        mgr.cancelJob(serviceJobId);
    }

    QJsonObject result;
    result["cancelRequested"] = true;
    result["serviceJobId"] = serviceJobId.isEmpty() ? QJsonValue() : QJsonValue(serviceJobId);
    return result;
}


QJsonObject AgentBridgeServer::handleLasagnaSelectOutputSegment(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString name = p.value("name").toString();
    if (name.isEmpty()) {
        QJsonObject data;
        data["param"] = "name";
        throw AgentBridgeError{-32602, "name is required", data};
    }
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    SurfacePanelController* panel = _window->_surfacePanel.get();
    if (!panel || !panel->selectSurfaceById(name.toStdString())) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = name;
        throw AgentBridgeError{-32007,
                               QStringLiteral("Unknown or unselectable segment: %1").arg(name),
                               data};
    }

    QJsonObject result;
    result["selected"] = true;
    result["name"] = name;
    return result;
}


QJsonObject AgentBridgeServer::handleLasagnaRepeatLast(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    requireSourceIdle(QStringLiteral("lasagna"));

    SegmentationWidget* widget = _window->_segmentationWidget;
    SegmentationLasagnaPanel* panel = widget ? widget->lasagnaPanel() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "lasagna panel is not available";
        throw AgentBridgeError{-32009, "Lasagna panel unavailable", data};
    }

    // repeatLastLasagnaAction() re-emits lasagnaOptimizeRequested, which
    // CWindow routes to the *interactive* startOptimization(state,
    // statusBar()) overload -- unsafe here. Use the headless
    // twin, which calls startOptimizationHeadless directly instead.
    QString errorMessage;
    const bool started = panel->repeatLastLasagnaActionHeadless(state, &errorMessage);
    if (!started) {
        QJsonObject data;
        data["detail"] = errorMessage;
        throw AgentBridgeError{-32005, "Nothing to repeat", data};
    }

    // Same synchronous-registration caveat as handleLasagnaStartOptimization.
    QString jobId;
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it != _activeJobs.end()) {
        it.value().kind = QStringLiteral("lasagna.optimize");
        jobId = it.value().id;
    } else {
        jobId = beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.optimize"),
                          QStringLiteral("Lasagna optimization started"),
                          /*broadcastStart=*/true);
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "lasagna.optimize";
    result["source"] = "lasagna";
    return result;
}


QJsonObject AgentBridgeServer::handleWorkspaceSwitch(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString name = p.value("name").toString();

    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    if (name == QLatin1String("main")) {
        _window->switchToMainWorkspace();
    } else if (name == QLatin1String("lasagna")) {
        _window->switchToLasagnaWorkspace();
    } else {
        _window->switchToFiberSliceWorkspace();
    }

    QJsonObject result;
    result["workspace"] = name;
    return result;
}
