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


namespace {
QJsonObject pushPullConfigToJson(const AlphaPushPullConfig& c)
{
    QJsonObject o;
    o["start"] = static_cast<double>(c.start);
    o["stop"] = static_cast<double>(c.stop);
    o["step"] = static_cast<double>(c.step);
    o["low"] = static_cast<double>(c.low);
    o["high"] = static_cast<double>(c.high);
    o["blurRadius"] = c.blurRadius;
    o["computeScale"] = c.computeScale;
    o["perVertexLimit"] = static_cast<double>(c.perVertexLimit);
    o["perVertex"] = c.perVertex;
    return o;
}
}  // namespace


// ---------------------------------------------------------------------------
// Stage 6 backlog surface (SPEC §15): tags, seeding, push/pull, run-trace
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleTagsSet(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }
    const QString tagStr = p.value("tag").toString();
    if (!p.contains("enabled") || !p.value("enabled").isBool()) {
        QJsonObject data;
        data["param"] = "enabled";
        throw AgentBridgeError{-32602, "enabled (bool) is required", data};
    }
    const bool enabled = p.value("enabled").toBool();

    // §15.1: the four-value Tag enum; "revisit" does NOT exist -> -32602.
    SurfacePanelController::Tag tag;
    if (tagStr == QLatin1String("approved"))
        tag = SurfacePanelController::Tag::Approved;
    else if (tagStr == QLatin1String("defective"))
        tag = SurfacePanelController::Tag::Defective;
    else if (tagStr == QLatin1String("reviewed"))
        tag = SurfacePanelController::Tag::Reviewed;
    else if (tagStr == QLatin1String("inspect"))
        tag = SurfacePanelController::Tag::Inspect;
    else {
        QJsonObject data;
        data["param"] = "tag";
        data["value"] = tagStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid tag: %1").arg(tagStr), data};
    }

    SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "surface panel is not available";
        throw AgentBridgeError{-32010, "Surface panel unavailable", data};
    }

    // §15.1: select first (documented side effect: leaves segmentId selected,
    // which enables the tag checkboxes), then setTagChecked. selectSurfaceById
    // returns false for an unknown/unloaded id -> -32007.
    const std::string segmentId = segmentIdQ.toStdString();
    if (!panel->selectSurfaceById(segmentId)) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentIdQ;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }
    if (!panel->setTagChecked(tag, enabled)) {
        // Selection succeeded but the tag checkbox is unavailable/disabled.
        QJsonObject data;
        data["detail"] = "tag checkbox is not available for this segment";
        throw AgentBridgeError{-32010, "Tag could not be set", data};
    }

    QJsonObject result;
    result["segmentId"] = segmentIdQ;
    result["tag"] = tagStr;
    result["enabled"] = enabled;
    return result;
}

QJsonObject AgentBridgeServer::handleSeedingSetWindingAnnotationMode(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    const QJsonObject p = paramsObject(params);
    if (!p.contains("active") || !p.value("active").isBool()) {
        QJsonObject data;
        data["param"] = "active";
        throw AgentBridgeError{-32602, "active (bool) is required", data};
    }
    const bool active = p.value("active").toBool();
    widget->setRelWindingAnnotationMode(active);
    QJsonObject result;
    result["active"] = active;
    return result;
}


QJsonObject AgentBridgeServer::handleSeedingPreviewRays(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    widget->runPreviewRays();
    QJsonObject result;
    result["requested"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handleSeedingCastRays(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    widget->runCastRays();
    QJsonObject result;
    result["requested"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handleSeedingResetPoints(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    widget->runResetPoints();
    QJsonObject result;
    result["reset"] = true;
    return result;
}


QJsonObject AgentBridgeServer::launchSeedingBatch(
    const QString& kind, const QString& label,
    const std::function<bool(QString* err)>& launch)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }

    // run and expand are mutually exclusive with each other (shared jobsRunning
    // flag); both map to source:"seeding", so the standard per-source guard
    // rejects a second batch of either kind (SPEC §8.3).
    requireSourceIdle(QStringLiteral("seeding"));

    QString err;
    if (!launch(&err)) {
        QJsonObject data;
        data["detail"] = err;
        // vc_grow_seg_from_seed missing -> -32006.
        if (err.contains(QLatin1String("executable not found"), Qt::CaseInsensitive))
            throw AgentBridgeError{-32006, "vc_grow_seg_from_seed not found", data};
        // Missing config/paths inputs -> -32007 kind:"file" (mirrors
        // tracer.run_trace's treatment of a missing trace_params.json).
        if (err.contains(QLatin1String("seed.json"), Qt::CaseInsensitive) ||
            err.contains(QLatin1String("expand.json"), Qt::CaseInsensitive) ||
            err.contains(QLatin1String("paths directory"), Qt::CaseInsensitive)) {
            data["kind"] = "file";
            throw AgentBridgeError{-32007, err, data};
        }
        // Everything else (no source collection, no points, no volume) is a
        // precondition/launch failure.
        throw AgentBridgeError{-32005, QStringLiteral("Failed to start seeding %1").arg(kind), data};
    }

    const QString rpcKind = QStringLiteral("seeding.%1").arg(kind);
    const QString jobId = beginJob(QStringLiteral("seeding"), rpcKind, label,
                                   /*broadcastStart=*/true);

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = rpcKind;
    result["source"] = "seeding";
    result["total"] = widget->seedingBatchTotal();
    return result;
}


QJsonObject AgentBridgeServer::handleSeedingRun(const QJsonValue&)
{
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    QJsonObject result = launchSeedingBatch(
        QStringLiteral("run"), QStringLiteral("Seeding run started"),
        [widget](QString* err) { return widget->runSegmentationHeadless(err); });
    // For run, "total" is the source-collection point count.
    result["points"] = result.value("total");
    return result;
}


QJsonObject AgentBridgeServer::handleSeedingExpand(const QJsonValue&)
{
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    QJsonObject result = launchSeedingBatch(
        QStringLiteral("expand"), QStringLiteral("Seeding expand started"),
        [widget](QString* err) { return widget->runExpandSeedsHeadless(err); });
    // For expand, "total" is the number of expansion iterations.
    result["iterations"] = result.value("total");
    return result;
}


QJsonObject AgentBridgeServer::handleSeedingCancel(const QJsonValue&)
{
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    if (!jobIsRunning(QStringLiteral("seeding"))) {
        QJsonObject data;
        data["kind"] = "job";
        throw AgentBridgeError{-32007, "No seeding batch running", data};
    }
    // Bounded synchronous teardown: terminate() + waitForFinished(1000) then
    // kill() per running child (numProcesses is small — the "Processes" spin box).
    // This synchronously emits seedingBatchFinished(kind,false,...), so the
    // source:"seeding" job is already resolved (a job.progress "finished"
    // notification has gone out) by the time this response is written.
    widget->cancelSeedingBatchHeadless();
    QJsonObject result;
    result["cancelRequested"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handleSeedingAnalyzePaths(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    // Synchronous, in-process compute (no QProcess, no nested event loop): runs
    // to completion before returning, so this is a plain synchronous RPC rather
    // than a job (SPEC §15.2). Requires paths drawn in the widget's Draw mode.
    QString err;
    int pathsAnalyzed = 0;
    int peaksFound = 0;
    if (!widget->runAnalyzePathsHeadless(&err, &pathsAnalyzed, &peaksFound)) {
        QJsonObject data;
        data["detail"] = err;
        // No drawn paths is a precondition failure, not a bad-params error.
        data["kind"] = "path";
        throw AgentBridgeError{-32007, err, data};
    }
    QJsonObject result;
    result["analyzed"] = true;
    result["paths"] = pathsAnalyzed;
    result["peaks"] = peaksFound;
    return result;
}


QJsonObject AgentBridgeServer::handlePushPullSetConfig(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!mod || !widget) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32010, "Segmentation module unavailable", data};
    }

    const QJsonObject p = paramsObject(params);

    // Read-modify-write over the current effective config (from the panel, which
    // always exists even before a session opens), then sanitize + apply.
    AlphaPushPullConfig cfg = widget->alphaPushPullConfig();

    auto readFloat = [&](const char* key, float& dst) {
        if (!p.contains(key)) return;
        const QJsonValue v = p.value(key);
        const double d = v.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!v.isDouble() || !std::isfinite(d)) {
            QJsonObject data;
            data["param"] = key;
            throw AgentBridgeError{-32602, QStringLiteral("%1 must be a finite number").arg(key), data};
        }
        dst = static_cast<float>(d);
    };
    auto readInt = [&](const char* key, int& dst) {
        if (!p.contains(key)) return;
        const QJsonValue v = p.value(key);
        if (!v.isDouble()) {
            QJsonObject data;
            data["param"] = key;
            throw AgentBridgeError{-32602, QStringLiteral("%1 must be an integer").arg(key), data};
        }
        dst = v.toInt();
    };
    auto readBool = [&](const char* key, bool& dst) {
        if (!p.contains(key)) return;
        const QJsonValue v = p.value(key);
        if (!v.isBool()) {
            QJsonObject data;
            data["param"] = key;
            throw AgentBridgeError{-32602, QStringLiteral("%1 must be a boolean").arg(key), data};
        }
        dst = v.toBool();
    };

    readFloat("start", cfg.start);
    readFloat("stop", cfg.stop);
    readFloat("step", cfg.step);
    readFloat("low", cfg.low);
    readFloat("high", cfg.high);
    readInt("blurRadius", cfg.blurRadius);
    readInt("computeScale", cfg.computeScale);
    readFloat("perVertexLimit", cfg.perVertexLimit);
    readBool("perVertex", cfg.perVertex);

    const AlphaPushPullConfig sanitized = SegmentationPushPullTool::sanitizeConfig(cfg);
    mod->setAlphaPushPullConfig(sanitized);  // sanitizes + updates tool + panel UI

    // Report the effective config as sanitized (matches what the panel now holds).
    return pushPullConfigToJson(sanitized);
}


QJsonObject AgentBridgeServer::handlePushPullStart(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!mod) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32010, "Segmentation module unavailable", data};
    }

    const QJsonObject p = paramsObject(params);
    const QString dirStr = p.value("direction").toString();
    int direction = 0;
    if (dirStr == QLatin1String("push"))
        direction = 1;
    else if (dirStr == QLatin1String("pull"))
        direction = -1;
    else {
        QJsonObject data;
        data["param"] = "direction";
        data["value"] = dirStr;
        throw AgentBridgeError{-32602, "direction must be \"push\" or \"pull\"", data};
    }

    std::optional<bool> alphaOverride;
    if (p.contains("alpha")) {
        if (!p.value("alpha").isBool()) {
            QJsonObject data;
            data["param"] = "alpha";
            throw AgentBridgeError{-32602, "alpha must be a boolean", data};
        }
        alphaOverride = p.value("alpha").toBool();
    }

    // §15.3 preconditions: editing enabled + active edit session.
    if (!widget || !widget->isEditingEnabled())
        throw AgentBridgeError{-32008, "Segmentation editing is not enabled", {}};
    if (!mod->hasActiveSession()) {
        QJsonObject data;
        data["kind"] = "session";
        data["detail"] = "no active segmentation edit session";
        throw AgentBridgeError{-32007, "No active edit session", data};
    }

    // startPushPull returns false when there is no valid hover target (the agent
    // must position the cursor first with a buttonless canvas.drag, §15.3) — that
    // is reported as active:false, not an error.
    const bool active = mod->startPushPullMode(direction, alphaOverride);
    QJsonObject result;
    result["active"] = active;
    return result;
}


QJsonObject AgentBridgeServer::handlePushPullStop(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    if (mod)
        mod->stopPushPullAll();
    QJsonObject result;
    result["stopped"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handleTracerRunTrace(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    // Run Trace runs vc_grow_seg_from_segments: the "tool" source (§8.3).
    requireSourceIdle(QStringLiteral("tool"));

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Run Trace unavailable", data};
    }

    SegmentationCommandHandler::RunTraceParams rt;
    const QJsonValue overrides = p.value("paramOverrides");
    if (overrides.isObject())
        rt.paramOverrides = overrides.toObject();
    else if (!overrides.isUndefined() && !overrides.isNull()) {
        QJsonObject data;
        data["param"] = "paramOverrides";
        throw AgentBridgeError{-32602, "paramOverrides must be an object", data};
    }
    if (p.contains("ompThreads")) {
        // Strict: reject wrong-typed and fractional values (SPEC §5).
        rt.ompThreads = jsonRequireInt(p.value("ompThreads"), "ompThreads");
    }
    rt.tgtDir = p.value("outputDir").toString();

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("tracer.run_trace"),
                                   QStringLiteral("Run Trace"),
                                   /*broadcastStart=*/false);

    // Suppress the runner's interactive "Operation Complete" QMessageBox for this
    // headless run so the modal dialog cannot starve the toolFinished slots that
    // transition the job out of "running" (auto-cleared on toolFinished; also
    // cleared on the synchronous-failure path below).
    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    QString err;
    QString outputDir;
    if (!handler->startRunTrace(segmentIdQ.toStdString(), rt, &err, &outputDir)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJobs.remove(QStringLiteral("tool"));
        // Map the distinct failure sentences from startRunTrace to codes (§15.4).
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentIdQ;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("trace_params.json not found"))) {
            QJsonObject data;
            data["kind"] = "file";
            data["detail"] = err;
            throw AgentBridgeError{-32007, "trace_params.json not found", data};
        }
        if (err.contains(QLatin1String("remote"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32009, "Remote volumes are unsupported by Run Trace", data};
        }
        if (err.contains(QLatin1String("Command line tools not available"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "vc_grow_seg_from_segments unavailable", data};
        }
        if (err.contains(QLatin1String("already running"))) {
            QJsonObject data;
            data["source"] = "tool";
            data["detail"] = err;
            throw AgentBridgeError{-32004, "A tool job is already running", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start Run Trace", data};
    }

    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "tracer.run_trace";
    result["source"] = "tool";
    result["outputDir"] = outputDir;
    return result;
}


// ---------------------------------------------------------------------------
// render.tifxyz (SPEC §19)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleRenderTifxyz(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    SegmentationCommandHandler::RenderSegmentParams rp;

    // outputFormat: required; the headline new capability over the GUI (which is
    // hardcoded to a per-slice TIFF stack).
    const QString outputFormat = p.value("outputFormat").toString();
    if (outputFormat == QLatin1String("zarr")) {
        rp.outputFormat = CommandLineToolRunner::RenderOutputFormat::Zarr;
    } else if (outputFormat == QLatin1String("tif_stack")) {
        rp.outputFormat = CommandLineToolRunner::RenderOutputFormat::TifStack;
    } else {
        QJsonObject data;
        data["param"] = "outputFormat";
        data["detail"] = "outputFormat must be \"zarr\" or \"tif_stack\"";
        throw AgentBridgeError{-32602, "Invalid outputFormat", data};
    }

    // volumeId: optional; validated here so a bad id is a clean -32007 before the
    // job is registered (mirrors segmentation.grow_patch_from_seed).
    rp.volumeId = p.value("volumeId").toString();
    if (!rp.volumeId.isEmpty()) {
        const auto ids = state->vpkg()->volumeIDs();
        if (std::find(ids.begin(), ids.end(), rp.volumeId.toStdString()) == ids.end()) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = rp.volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(rp.volumeId), data};
        }
    }

    // scale: optional, default 1.0; must be finite and > 0.
    if (p.contains("scale")) {
        if (!p.value("scale").isDouble()) {
            QJsonObject data;
            data["param"] = "scale";
            throw AgentBridgeError{-32602, "scale must be a number", data};
        }
        rp.scale = static_cast<float>(p.value("scale").toDouble(1.0));
    }
    if (!std::isfinite(rp.scale) || rp.scale <= 0.0f) {
        QJsonObject data;
        data["param"] = "scale";
        throw AgentBridgeError{-32602, "scale must be a finite value > 0", data};
    }

    // groupIdx: optional, default 0; OME-Zarr group index (>= 0).
    if (p.contains("groupIdx")) {
        if (!p.value("groupIdx").isDouble()) {
            QJsonObject data;
            data["param"] = "groupIdx";
            throw AgentBridgeError{-32602, "groupIdx must be an integer", data};
        }
        rp.groupIdx = p.value("groupIdx").toInt(0);
    }
    if (rp.groupIdx < 0) {
        QJsonObject data;
        data["param"] = "groupIdx";
        throw AgentBridgeError{-32602, "groupIdx must be >= 0", data};
    }

    // numSlices: optional, default 1; must be >= 1.
    if (p.contains("numSlices")) {
        if (!p.value("numSlices").isDouble()) {
            QJsonObject data;
            data["param"] = "numSlices";
            throw AgentBridgeError{-32602, "numSlices must be an integer", data};
        }
        rp.numSlices = p.value("numSlices").toInt(1);
    }
    if (rp.numSlices < 1) {
        QJsonObject data;
        data["param"] = "numSlices";
        throw AgentBridgeError{-32602, "numSlices must be >= 1", data};
    }

    // voxelSize: optional override; when omitted the tool derives it from volume
    // metadata (matching the interactive render path).
    if (p.contains("voxelSize") && !p.value("voxelSize").isNull()) {
        if (!p.value("voxelSize").isDouble()) {
            QJsonObject data;
            data["param"] = "voxelSize";
            throw AgentBridgeError{-32602, "voxelSize must be a number", data};
        }
        const double vs = p.value("voxelSize").toDouble();
        if (!std::isfinite(vs) || vs <= 0.0) {
            QJsonObject data;
            data["param"] = "voxelSize";
            throw AgentBridgeError{-32602, "voxelSize must be a finite value > 0", data};
        }
        rp.hasVoxelSize = true;
        rp.voxelSizeUm = vs;
    }

    rp.outputDir = p.value("outputDir").toString();

    // Render runs the external vc_render_tifxyz process: the "tool" source (§8.3).
    requireSourceIdle(QStringLiteral("tool"));

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Render unavailable", data};
    }

    const QString effectiveVolumeId = rp.volumeId.isEmpty()
        ? QString::fromStdString(state->currentVolumeId())
        : rp.volumeId;

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("render.tifxyz"),
                                   QStringLiteral("Render segment"),
                                   /*broadcastStart=*/false);

    // Suppress the runner's interactive "Operation Complete" QMessageBox for this
    // headless run so the modal dialog cannot starve the toolFinished slots that
    // transition the job out of "running" (auto-cleared on toolFinished; also
    // cleared on the synchronous-failure path below). Same proven pattern as
    // segmentation.grow_patch_from_seed / tracer.run_trace.
    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    QString err;
    QString outputDir;
    if (!handler->startRenderSegment(segmentIdQ.toStdString(), rp, &err, &outputDir)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJobs.remove(QStringLiteral("tool"));
        // Map the distinct failure sentences from startRenderSegment (§19).
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentIdQ;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("Unknown volume id"))) {
            QJsonObject data;
            data["kind"] = "volume";
            data["detail"] = err;
            throw AgentBridgeError{-32007, err, data};
        }
        if (err.contains(QLatin1String("not found or not executable"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "vc_render_tifxyz unavailable", data};
        }
        if (err.contains(QLatin1String("already running"))) {
            QJsonObject data;
            data["source"] = "tool";
            data["detail"] = err;
            throw AgentBridgeError{-32004, "A tool job is already running", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start render", data};
    }

    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "render.tifxyz";
    result["source"] = "tool";
    result["outputDir"] = outputDir;
    result["outputFormat"] = outputFormat;
    result["volumeId"] = effectiveVolumeId;
    return result;
}
