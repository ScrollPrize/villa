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


void AgentBridgeServer::completeCatalogOpenJob(
    const MenuActionController::OpenDataSampleOpenOutcome& outcome)
{
    auto it = _activeJobs.find(QStringLiteral("catalog"));
    if (it == _activeJobs.end())
        return;  // job already reaped (e.g. shutdown); nothing to resolve.

    QString message;
    QString vpkgPath;
    if (outcome.success) {
        CState* state = _window ? _window->_state : nullptr;
        vpkgPath = (state && state->hasVpkg()) ? state->vpkgPath() : QString();

        QJsonObject body;
        body["opened"] = true;
        body["sampleId"] = _catalogOpenSampleId;
        body["vpkgPath"] = vpkgPath.isEmpty() ? QJsonValue(QString())
                                              : QJsonValue(vpkgPath);
        QJsonArray idArr;
        if (state && state->vpkg()) {
            for (const auto& id : state->vpkg()->volumeIDs())
                idArr.append(QString::fromStdString(id));
        }
        body["volumeIds"] = idArr;

        QJsonObject attached;
        attached["volumes"] = outcome.result.attachedVolumeEntries;
        attached["segments"] = outcome.result.attachedSegmentEntries;
        attached["normalGrids"] = outcome.result.attachedNormalGrids;
        attached["lasagnaDatasets"] = outcome.result.attachedLasagnaDatasets;
        body["attached"] = attached;

        QJsonArray messages;
        for (const auto& m : outcome.result.messages)
            messages.append(QString::fromStdString(m));
        body["messages"] = messages;

        it->resultJson = body;  // carried by finishJob's copy into history/wire.
        message = QStringLiteral("Opened sample %1").arg(_catalogOpenSampleId);
    } else {
        message = outcome.error.isEmpty()
            ? QStringLiteral("Open Data sample open failed")
            : outcome.error;
        // resultJson stays empty -> "result": null on the wire.
    }

    _catalogOpenSampleId.clear();
    finishJob(QStringLiteral("catalog"), outcome.success, message, vpkgPath);
}

void AgentBridgeServer::completeVolumeAttachmentJob(
    const VolumeAttachmentOutcome& outcome)
{
    auto it = _activeJobs.find(QStringLiteral("volume"));
    if (it == _activeJobs.end())
        return;

    if (outcome.success) {
        it->resultJson = {
            {"attached", !outcome.alreadyAttached},
            {"alreadyAttached", outcome.alreadyAttached},
            {"volumeId", outcome.volumeId},
            {"location", outcome.location},
            {"projectPath", outcome.projectPath},
        };
    }
    const QString message = outcome.success
        ? outcome.alreadyAttached
            ? QStringLiteral("Volume %1 is already attached").arg(outcome.volumeId)
            : QStringLiteral("Attached volume %1").arg(outcome.volumeId)
        : outcome.error.isEmpty()
            ? QStringLiteral("Volume attachment failed")
            : outcome.error;
    finishJob(
        QStringLiteral("volume"),
        outcome.success,
        message,
        outcome.projectPath);
}

// ---------------------------------------------------------------------------
// Job tracking
// ---------------------------------------------------------------------------

void AgentBridgeServer::subscribeJobSignals()
{
    if (!_window)
        return;

    if (_window->_cmdRunner) {
        CommandLineToolRunner* runner = _window->_cmdRunner;
        connect(runner, &CommandLineToolRunner::toolStarted, this,
                [this](CommandLineToolRunner::Tool, const QString& message) {
                    handleToolStarted(message);
                });
        connect(runner, &CommandLineToolRunner::toolFinished, this,
                [this](CommandLineToolRunner::Tool, bool success, const QString& message,
                       const QString& outputPath, bool) {
                    handleToolFinished(success, message, outputPath);
                });
        connect(runner, &CommandLineToolRunner::consoleOutputReceived, this,
                [this](const QString& output) { handleConsoleOutput(output); });
    }

    if (_window->_segmentationModule) {
        connect(_window->_segmentationModule.get(), &SegmentationModule::growthInProgressChanged,
                this, [this](bool running) { handleGrowthStatusChanged(running); });
        // Explicit segment save/flush lifecycle -> source:"autosave" jobs
        // Closes the job opened by handleSegmentationSave.
        connect(_window->_segmentationModule.get(), &SegmentationModule::autosaveCompleted,
                this, [this](bool success) { handleAutosaveCompleted(success); });
    }

    // Flattening lifecycle -> source:"flatten" jobs. Emitted by
    // the SlimJob / ABFJob / StraightenJob classes from BOTH the interactive
    // slots and the headless start* launchers, so human-initiated flattens are
    // registered as external jobs too.
    if (_window->_segmentationCommandHandler) {
        SegmentationCommandHandler* sch = _window->_segmentationCommandHandler.get();
        connect(sch, &SegmentationCommandHandler::flattenJobStarted, this,
                [this](const QString& kind, const QString& label) {
                    handleFlattenStarted(kind, label);
                });
        connect(sch, &SegmentationCommandHandler::flattenJobFinished, this,
                [this](bool success, const QString& message, const QString& outputPath) {
                    handleFlattenFinished(success, message, outputPath);
                });
    }

    // Lasagna optimization lifecycle -> source:"lasagna" jobs.
    // The singleton outlives the bridge, so the connections are lifetime-safe.
    LasagnaServiceManager* lasagna = &LasagnaServiceManager::instance();
    connect(lasagna, &LasagnaServiceManager::optimizationStarted, this,
            [this]() { handleLasagnaStarted(); });
    connect(lasagna, &LasagnaServiceManager::jobStarted, this,
            [this](const QString& jobId) { handleLasagnaJobStarted(jobId); });
    connect(lasagna, &LasagnaServiceManager::optimizationProgress, this,
            [this](const QString& stage, int step, int totalSteps, double loss,
                   double /*stageProgress*/, double overallProgress,
                   const QString& stageName) {
                handleLasagnaProgress(stageName.isEmpty() ? stage : stageName,
                                      step, totalSteps, loss, overallProgress);
            });
    connect(lasagna, &LasagnaServiceManager::optimizationFinished, this,
            [this](const QString& outputDir) {
                handleLasagnaFinished(true, QStringLiteral("Optimization finished"), outputDir);
            });
    connect(lasagna, &LasagnaServiceManager::optimizationError, this,
            [this](const QString& message) {
                handleLasagnaFinished(false, message, QString());
            });
    connect(lasagna, &LasagnaServiceManager::jobFinished, this,
            [this](const QString& /*jobId*/, const QString& outputDir) {
                handleLasagnaFinished(true, QStringLiteral("Job finished"), outputDir);
            });
    connect(lasagna, &LasagnaServiceManager::jobError, this,
            [this](const QString& /*jobId*/, const QString& message) {
                handleLasagnaFinished(false, message, QString());
            });
    connect(lasagna, &LasagnaServiceManager::resultsPlaced, this,
            [this](const QString& outputDir, const QStringList& segmentNames) {
                handleLasagnaResultsPlaced(outputDir, segmentNames);
            });

    // Atlas fiber-search lifecycle -> source:"atlas" jobs.
    if (_window) {
        connect(_window, &CWindow::atlasSearchProgressChanged, this,
                [this](int phase, double fraction) {
                    handleAtlasSearchProgress(phase, fraction);
                });
        connect(_window, &CWindow::atlasSearchFinished, this,
                [this](bool success, int resultCount) {
                    handleAtlasSearchFinished(success, resultCount);
                });
    }

    // Batch seeding lifecycle -> source:"seeding" jobs. Emitted from
    // the SeedingWidget QProcess finished callbacks (run/expand). Only
    // bridge-initiated batches become tracked jobs (see handleSeedingBatch*).
    if (_window && _window->_seedingWidget) {
        connect(_window->_seedingWidget, &SeedingWidget::seedingBatchProgressChanged, this,
                [this](const QString& kind, int completed, int total) {
                    handleSeedingBatchProgress(kind, completed, total);
                });
        connect(_window->_seedingWidget, &SeedingWidget::seedingBatchFinished, this,
                [this](const QString& kind, bool success, bool canceled, int completed,
                       int total, const QString& message) {
                    handleSeedingBatchFinished(kind, success, canceled, completed, total, message);
                });
    }
}


bool AgentBridgeServer::jobIsRunning(const QString& source) const
{
    if (auto it = _activeJobs.constFind(source);
        it != _activeJobs.constEnd() && it->state == QLatin1String("running"))
        return true;
    if (_window) {
        if (source == QLatin1String("tool") &&
            _window->_cmdRunner && _window->_cmdRunner->isRunning())
            return true;
        if (source == QLatin1String("growth") &&
            _window->_segmentationGrower && _window->_segmentationGrower->running())
            return true;
        // Lifecycle authority for atlas searches: the cancel flag is created
        // when a search launches and reset in its finished handler, so a
        // non-null flag means a search (bridge- or human-initiated) is live.
        if (source == QLatin1String("atlas") && _window->_atlasSearchCancelFlag)
            return true;
        // Lifecycle authority for seeding batches: seedingBatchActive() is true
        // only while a run/expand batch is draining (not a neural trace, which
        // shares the widget's jobsRunning flag) — so a human-initiated Run/Expand
        // click is also seen as busy here.
        if (source == QLatin1String("seeding") && _window->_seedingWidget &&
            _window->_seedingWidget->seedingBatchActive())
            return true;
    }
    return false;
}


QString AgentBridgeServer::activeJobId(const QString& source) const
{
    if (auto it = _activeJobs.constFind(source);
        it != _activeJobs.constEnd() && it->state == QLatin1String("running"))
        return it->id;
    return QString();
}


void AgentBridgeServer::requireSourceIdle(const QString& source) const
{
    if (!jobIsRunning(source))
        return;
    QJsonObject data;
    const QString jid = activeJobId(source);
    if (!jid.isEmpty())
        data["jobId"] = jid;
    data["source"] = source;
    throw AgentBridgeError{-32004,
        QStringLiteral("A %1 job is already running").arg(source), data};
}


QString AgentBridgeServer::beginJob(const QString& source, const QString& kind,
                                    const QString& label, bool broadcastStart)
{
    JobRecord job;
    job.num = _nextJobNum++;
    job.id = QStringLiteral("job-%1").arg(job.num);
    job.source = source;
    job.kind = kind;
    job.label = label;
    job.state = QStringLiteral("running");
    job.message = label;
    job.startedAtMs = QDateTime::currentMSecsSinceEpoch();
    _activeJobs.insert(source, job);
    if (broadcastStart) {
        auto it = _activeJobs.find(source);
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }
    return job.id;
}


void AgentBridgeServer::finishJob(const QString& source, bool success,
                                  const QString& message, const QString& outputPath)
{
    auto it = _activeJobs.find(source);
    if (it == _activeJobs.end())
        return;
    JobRecord job = it.value();
    _activeJobs.erase(it);

    job.state = success ? QStringLiteral("succeeded") : QStringLiteral("failed");
    if (!message.isEmpty())
        job.message = message;
    if (!outputPath.isEmpty())
        job.outputPath = outputPath;
    job.finishedAtMs = QDateTime::currentMSecsSinceEpoch();

    broadcastJobProgress(job, QStringLiteral("finished"), QString(), success);

    // Retain the last <=8 completed jobs per source.
    std::deque<JobRecord>& hist = _recentJobs[source];
    hist.push_back(job);
    while (hist.size() > 8)
        hist.pop_front();
}


const AgentBridgeServer::JobRecord*
AgentBridgeServer::mostRecentJob(const QString& sourceFilter) const
{
    const JobRecord* best = nullptr;
    auto consider = [&](const JobRecord& rec) {
        if (!sourceFilter.isEmpty() && rec.source != sourceFilter)
            return;
        if (!best || rec.num > best->num)
            best = &rec;
    };
    for (const auto& rec : _activeJobs)
        consider(rec);
    for (const auto& hist : _recentJobs)
        for (const auto& rec : hist)
            consider(rec);
    return best;
}


const AgentBridgeServer::JobRecord*
AgentBridgeServer::jobById(const QString& jobId) const
{
    for (const auto& rec : _activeJobs)
        if (rec.id == jobId)
            return &rec;
    for (const auto& hist : _recentJobs)
        for (const auto& rec : hist)
            if (rec.id == jobId)
                return &rec;
    return nullptr;
}


void AgentBridgeServer::broadcastJobProgress(JobRecord& job, const QString& phase,
                                             const QString& messageOverride,
                                             std::optional<bool> success)
{
    QJsonObject params;
    params["jobId"] = job.id;
    params["seq"] = static_cast<double>(job.nextProgressSeq++);
    params["source"] = job.source;
    params["kind"] = job.kind;
    params["phase"] = phase;
    const QString msg = messageOverride.isEmpty() ? job.message : messageOverride;
    if (!msg.isEmpty())
        params["message"] = msg;
    if (success.has_value()) {
        params["success"] = *success;
        if (!job.outputPath.isEmpty())
            params["outputPath"] = job.outputPath;
        // Terminal job.progress carries the result body too.
        params["result"] = job.resultJson.isEmpty() ? QJsonValue(QJsonValue::Null)
                                                     : QJsonValue(job.resultJson);
    }
    job.progressHistory.push_back(params);
    while (job.progressHistory.size() > 64)
        job.progressHistory.pop_front();
    broadcastNotification(QStringLiteral("job.progress"), params);
}


QJsonObject AgentBridgeServer::jobStatusJson(const JobRecord& job) const
{
    QJsonObject o;
    o["jobId"] = job.id;
    o["source"] = job.source;
    o["kind"] = job.kind;
    o["label"] = job.label;
    o["state"] = job.state;
    o["message"] = job.message;
    o["outputPath"] = job.outputPath.isEmpty() ? QJsonValue(QJsonValue::Null)
                                               : QJsonValue(job.outputPath);
    o["externalId"] = job.externalId.isEmpty() ? QJsonValue(QJsonValue::Null)
                                               : QJsonValue(job.externalId);
    QJsonArray tail;
    for (const QString& line : job.consoleTail)
        tail.append(line);
    o["consoleTail"] = tail;
    QJsonArray progressHistory;
    for (const QJsonObject& update : job.progressHistory)
        progressHistory.append(update);
    o["progressHistory"] = progressHistory;
    o["startedAtMs"] = static_cast<double>(job.startedAtMs);
    o["finishedAtMs"] = job.finishedAtMs == 0
                            ? QJsonValue(QJsonValue::Null)
                            : QJsonValue(static_cast<double>(job.finishedAtMs));
    // Catalog and volume jobs may expose a structured terminal result.
    o["result"] = job.resultJson.isEmpty() ? QJsonValue(QJsonValue::Null)
                                           : QJsonValue(job.resultJson);
    return o;
}


void AgentBridgeServer::handleToolStarted(const QString& message)
{
    auto it = _activeJobs.find(QStringLiteral("tool"));
    if (it == _activeJobs.end()) {
        // Tool run initiated outside the bridge (e.g. a menu action): track it
        // as an externally-initiated job.
        beginJob(QStringLiteral("tool"), QStringLiteral("tool.external"), message,
                 /*broadcastStart=*/true);
    } else {
        // The RPC that launched this already created + broadcast the job.
        it.value().message = message;
    }
}


void AgentBridgeServer::handleToolFinished(bool success, const QString& message,
                                           const QString& outputPath)
{
    finishJob(QStringLiteral("tool"), success, message, outputPath);
}


void AgentBridgeServer::handleConsoleOutput(const QString& output)
{
    auto it = _activeJobs.find(QStringLiteral("tool"));
    if (it == _activeJobs.end())
        return;
    JobRecord& job = it.value();

    const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    for (const QString& line : lines)
        job.consoleTail.append(line);
    while (job.consoleTail.size() > 50)
        job.consoleTail.removeFirst();

    // Rate-limit job.progress "output" to <=10/sec, coalescing.
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(job, QStringLiteral("output"), output.trimmed());
}


void AgentBridgeServer::handleGrowthStatusChanged(bool running)
{
    if (running) {
        if (!_activeJobs.contains(QStringLiteral("growth")))
            beginJob(QStringLiteral("growth"), QStringLiteral("growth.external"),
                     QStringLiteral("Segmentation growth started"), /*broadcastStart=*/true);
    } else if (_activeJobs.contains(QStringLiteral("growth"))) {
        finishJob(QStringLiteral("growth"), true,
                  QStringLiteral("Segmentation growth finished"), QString());
    }
}


void AgentBridgeServer::handleAutosaveCompleted(bool success)
{
    // Only close a bridge-initiated explicit save. Autosaves that
    // fire from the periodic timer / edit hooks are not tracked as jobs, so
    // finishJob is a no-op unless handleSegmentationSave opened one.
    if (!_activeJobs.contains(QStringLiteral("autosave")))
        return;
    finishJob(QStringLiteral("autosave"), success,
              success ? QStringLiteral("Segment saved")
                      : QStringLiteral("Segment save failed"),
              QString());
}


// --- Flattening lifecycle ---

void AgentBridgeServer::handleFlattenStarted(const QString& kind, const QString& label)
{
    auto it = _activeJobs.find(QStringLiteral("flatten"));
    if (it == _activeJobs.end()) {
        // A flatten launched outside the bridge (a human ran SLIM/ABF/Straighten
        // from the context menu): register it as an externally-initiated job so
        // -32004 / state.get reflect true app state. The kind carries the
        // real flatten type (flatten.slim / flatten.abf / flatten.straighten).
        beginJob(QStringLiteral("flatten"), kind,
                 label.isEmpty() ? QStringLiteral("Flatten") : label,
                 /*broadcastStart=*/true);
    } else {
        // The RPC that launched this already created + broadcast the job; adopt
        // the concrete kind/label the job reported.
        it.value().kind = kind;
        if (!label.isEmpty())
            it.value().label = label;
    }
}


void AgentBridgeServer::handleFlattenFinished(bool success, const QString& message,
                                              const QString& outputPath)
{
    // An empty message on failure denotes a user cancel; surface a
    // stable message so job.status is not blank.
    const QString msg = (!success && message.isEmpty())
                            ? QStringLiteral("Flatten cancelled")
                            : message;
    finishJob(QStringLiteral("flatten"), success, msg, outputPath);
}


// --- Lasagna optimization lifecycle ---

void AgentBridgeServer::handleLasagnaStarted()
{
    // An optimization launched outside the bridge (a human clicked the panel):
    // register it as an externally-initiated job so -32004 / state.get reflect
    // true app state. A bridge-submitted job already exists here.
    if (!_activeJobs.contains(QStringLiteral("lasagna"))) {
        beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.external"),
                 QStringLiteral("Lasagna optimization started"), /*broadcastStart=*/true);
    } else {
        auto it = _activeJobs.find(QStringLiteral("lasagna"));
        it.value().message = QStringLiteral("Lasagna optimization started");
    }
}


void AgentBridgeServer::handleLasagnaJobStarted(const QString& externalId)
{
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it == _activeJobs.end()) {
        // Started outside the bridge: register it, carrying the service job id.
        const QString jobId = beginJob(QStringLiteral("lasagna"),
                                       QStringLiteral("lasagna.external"),
                                       QStringLiteral("Lasagna optimization started"),
                                       /*broadcastStart=*/true);
        (void)jobId;
        it = _activeJobs.find(QStringLiteral("lasagna"));
    }
    if (it != _activeJobs.end())
        it.value().externalId = externalId;
}


void AgentBridgeServer::handleLasagnaProgress(const QString& stageName, int step,
                                              int totalSteps, double loss,
                                              double overallProgress)
{
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it == _activeJobs.end()) {
        // Progress before any registered start: adopt it as an external job.
        beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.external"),
                 QStringLiteral("Lasagna optimization running"), /*broadcastStart=*/true);
        it = _activeJobs.find(QStringLiteral("lasagna"));
        if (it == _activeJobs.end())
            return;
    }
    JobRecord& job = it.value();

    QString label = stageName.isEmpty() ? QStringLiteral("optimizing") : stageName;
    if (totalSteps > 0)
        label += QStringLiteral(" step %1/%2").arg(step).arg(totalSteps);
    if (std::isfinite(loss))
        label += QStringLiteral(" loss=%1").arg(loss, 0, 'g', 6);
    if (std::isfinite(overallProgress) && overallProgress > 0.0)
        label += QStringLiteral(" (%1%)").arg(overallProgress * 100.0, 0, 'f', 1);
    job.message = label;

    job.consoleTail.append(label);
    while (job.consoleTail.size() > 50)
        job.consoleTail.removeFirst();

    // Rate-limit job.progress "output" to <=10/sec.
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(job, QStringLiteral("output"), label);
}


void AgentBridgeServer::handleLasagnaFinished(bool success, const QString& message,
                                              const QString& outputPath)
{
    // finishJob is a no-op if no lasagna job is active, so the redundant
    // optimizationFinished + jobFinished pair resolves the job exactly once.
    if (_activeJobs.contains(QStringLiteral("lasagna")))
        finishJob(QStringLiteral("lasagna"), success, message, outputPath);
}


void AgentBridgeServer::handleLasagnaResultsPlaced(const QString& outputDir,
                                                   const QStringList& segmentNames)
{
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it == _activeJobs.end())
        return;
    JobRecord& job = it.value();
    if (!outputDir.isEmpty())
        job.outputPath = outputDir;
    const QString label = segmentNames.isEmpty()
        ? QStringLiteral("Results placed: %1").arg(outputDir)
        : QStringLiteral("Results placed: %1 (%2)")
              .arg(outputDir, segmentNames.join(QStringLiteral(", ")));
    job.message = label;
    broadcastJobProgress(job, QStringLiteral("output"), label);
}


// --- Atlas fiber-search lifecycle ---

void AgentBridgeServer::handleAtlasSearchProgress(int phase, double fraction)
{
    // Only bridge-initiated searches are tracked: do NOT
    // auto-register an external job here — FinishResults progress also fires
    // on pure UI re-population (group-by-fiber checkbox), which would leak a
    // job that never finishes.
    auto it = _activeJobs.find(QStringLiteral("atlas"));
    if (it == _activeJobs.end())
        return;
    JobRecord& job = it.value();

    const double clamped = std::clamp(fraction, 0.0, 1.0);
    const QString label = QStringLiteral("phase %1/5 (%2%)")
                              .arg(phase)
                              .arg(qRound(clamped * 100.0));
    job.message = label;

    // Rate-limit job.progress "output" to <=10/sec.
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(job, QStringLiteral("output"), label);
}


void AgentBridgeServer::handleAtlasSearchFinished(bool success, int resultCount)
{
    // finishJob is a no-op when no atlas job is active (e.g. a human-initiated
    // search, or an early-return failure before the bridge registered a job).
    if (!_activeJobs.contains(QStringLiteral("atlas")))
        return;
    finishJob(QStringLiteral("atlas"), success,
              success
                  ? QStringLiteral("Atlas fiber search finished: %1 result(s)").arg(resultCount)
                  : QStringLiteral("Atlas fiber search canceled or failed"),
              QString());
}


// --- Batch seeding lifecycle ---

void AgentBridgeServer::handleSeedingBatchProgress(const QString& kind, int completed,
                                                   int total)
{
    // Only bridge-initiated batches are tracked (mirroring the atlas search): a
    // human-clicked Run/Expand is deliberately NOT auto-registered as a job.
    auto it = _activeJobs.find(QStringLiteral("seeding"));
    if (it == _activeJobs.end())
        return;
    JobRecord& job = it.value();

    const QString label = QStringLiteral("%1 %2/%3").arg(kind).arg(completed).arg(total);
    job.message = label;

    job.consoleTail.append(label);
    while (job.consoleTail.size() > 50)
        job.consoleTail.removeFirst();

    // Rate-limit job.progress "output" to <=10/sec.
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(job, QStringLiteral("output"), label);
}


void AgentBridgeServer::handleSeedingBatchFinished(const QString& kind, bool success,
                                                   bool canceled, int completed, int total,
                                                   const QString& message)
{
    // finishJob is a no-op when no seeding job is active (a human-initiated
    // batch, or the neural-trace cancel path that emits an empty kind).
    if (!_activeJobs.contains(QStringLiteral("seeding")))
        return;
    // Both an execution failure and a user cancel map to a failed job; the widget
    // supplies meaningful terminal text distinguishing them. Fall back
    // to a synthesized label only if the widget passed none.
    QString label = message;
    if (label.isEmpty()) {
        if (success)
            label = QStringLiteral("Seeding %1 finished: %2/%3").arg(kind).arg(completed).arg(total);
        else if (canceled)
            label = QStringLiteral("Seeding %1 canceled after %2/%3").arg(kind).arg(completed).arg(total);
        else
            label = QStringLiteral("Seeding %1 failed").arg(kind);
    }
    finishJob(QStringLiteral("seeding"), success, label, QString());
}


QJsonObject AgentBridgeServer::handleJobStatus(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString jobId = p.value("jobId").toString();
    const QString source = p.value("source").toString();

    const JobRecord* rec = jobId.isEmpty() ? mostRecentJob(source) : jobById(jobId);

    if (!rec) {
        QJsonObject data;
        data["kind"] = "job";
        if (!jobId.isEmpty())
            data["id"] = jobId;
        throw AgentBridgeError{-32007, "No such job", data};
    }

    return jobStatusJson(*rec);
}


QJsonObject AgentBridgeServer::handleJobCancel(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString jobId = p.value("jobId").toString();
    const QString source = p.value("source").toString();
    if (jobId.isEmpty() && source.isEmpty()) {
        QJsonObject data;
        data["param"] = "jobId";
        throw AgentBridgeError{-32602,
            "at least one of jobId or source is required", data};
    }

    // Prefer jobId; if only source given, resolve the source's active job.
    const QString resolvedId = jobId.isEmpty() ? activeJobId(source) : jobId;
    const JobRecord* rec = resolvedId.isEmpty() ? nullptr : jobById(resolvedId);
    if (!rec || rec->state != QLatin1String("running")) {
        QJsonObject data;
        data["kind"] = "job";
        if (!resolvedId.isEmpty())
            data["id"] = resolvedId;
        else if (!source.isEmpty())
            data["source"] = source;
        throw AgentBridgeError{-32007, "No such running job", data};
    }

    // Snapshot the identifying fields BEFORE dispatching: a synchronous cancel
    // authority (e.g. tool/seeding) may call finishJob, which erases the active
    // JobRecord and invalidates `rec`.
    const QString outId = rec->id;
    const QString outSource = rec->source;
    const QString outKind = rec->kind;
    const QString externalId = rec->externalId;

    // Dispatch on the job's source to its per-source cancel authority.
    if (outSource == QLatin1String("atlas")) {
        if (!_window)
            throw AgentBridgeError{-32010, "Window unavailable", {}};
        // Dialog-free: sets cancel flags; the job terminates via atlasSearchFinished.
        _window->cancelAtlasFiberIntersectionSearch();
    } else if (outSource == QLatin1String("seeding")) {
        SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
        if (!widget) {
            QJsonObject data;
            data["detail"] = "seeding widget is not available";
            throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
        }
        // Bounded synchronous teardown; emits seedingBatchFinished -> finishJob.
        widget->cancelSeedingBatchHeadless();
    } else if (outSource == QLatin1String("lasagna")) {
        LasagnaServiceManager& mgr = LasagnaServiceManager::instance();
        if (!mgr.isRunning()) {
            QJsonObject data;
            data["detail"] = "lasagna service is not running";
            throw AgentBridgeError{-32005, "Lasagna service not running", data};
        }
        if (externalId.isEmpty())
            mgr.stopOptimization();
        else
            mgr.cancelJob(externalId);
    } else if (outSource == QLatin1String("tool")) {
        // grow/trace/render tools run as CommandLineToolRunner child QProcesses:
        // terminate cleanly; toolFinished then resolves the job via finishJob.
        if (!_window || !_window->_cmdRunner || !_window->_cmdRunner->isRunning()) {
            QJsonObject data;
            data["detail"] = "no running tool process to cancel";
            throw AgentBridgeError{-32005, "Tool process not running", data};
        }
        _window->_cmdRunner->cancel();
    } else {
        // growth (QtConcurrent future, no cancel), flatten (self-owned job with no
        // reachable cancel handle), catalog, volume, autosave: no safe cancel
        // authority.
        QJsonObject data;
        data["kind"] = "job";
        data["reason"] = "not cancellable";
        data["source"] = outSource;
        throw AgentBridgeError{-32010,
            QStringLiteral("A %1 job cannot be cancelled").arg(outSource), data};
    }

    QJsonObject result;
    result["cancelRequested"] = true;
    result["jobId"] = outId;
    result["source"] = outSource;
    result["kind"] = outKind;
    return result;
}
