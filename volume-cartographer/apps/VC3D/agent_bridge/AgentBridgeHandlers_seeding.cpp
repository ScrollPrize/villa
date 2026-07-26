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
#include <filesystem>
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
// Tags, seeding, push/pull, and tracing
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleTagsSet(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = params.toObject();
    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }
    const QString tagStr = p.value("tag").toString();
    const bool enabled = p.value("enabled").toBool();

    // Keep this mapping aligned with SurfacePanelController::Tag.
    SurfacePanelController::Tag tag;
    if (tagStr == QLatin1String("approved"))
        tag = SurfacePanelController::Tag::Approved;
    else if (tagStr == QLatin1String("defective"))
        tag = SurfacePanelController::Tag::Defective;
    else if (tagStr == QLatin1String("reviewed"))
        tag = SurfacePanelController::Tag::Reviewed;
    else if (tagStr == QLatin1String("inspect"))
        tag = SurfacePanelController::Tag::Inspect;
    else
        tag = SurfacePanelController::Tag::Inspect;

    SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "surface panel is not available";
        throw AgentBridgeError{-32010, "Surface panel unavailable", data};
    }

    // Select first, leaving segmentId active so the tag checkboxes are enabled,
    // then set the tag. selectSurfaceById returns false for an unknown or
    // unloaded id.
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
    const QJsonObject p = params.toObject();
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
    QString errorMessage;
    if (!widget->previewRaysHeadless(&errorMessage)) {
        QJsonObject data;
        data["detail"] = errorMessage;
        throw AgentBridgeError{-32007, "Cannot preview rays", data};
    }
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
    QString errorMessage;
    if (!widget->castRaysHeadless(&errorMessage)) {
        QJsonObject data;
        data["detail"] = errorMessage;
        throw AgentBridgeError{-32007, "Cannot cast rays", data};
    }
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
    // rejects a second batch of either kind.
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
    // than a job. Requires paths drawn in the widget's Draw mode.
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

    const QJsonObject p = params.toObject();

    // Read-modify-write over the current effective config (from the panel, which
    // always exists even before a session opens), then sanitize + apply.
    AlphaPushPullConfig cfg = widget->alphaPushPullConfig();

    auto readFloat = [&](const char* key, float& dst) {
        if (!p.contains(key)) return;
        dst = static_cast<float>(p.value(key).toDouble());
    };
    auto readInt = [&](const char* key, int& dst) {
        if (!p.contains(key)) return;
        dst = p.value(key).toInt();
    };
    auto readBool = [&](const char* key, bool& dst) {
        if (!p.contains(key)) return;
        dst = p.value(key).toBool();
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

    const QJsonObject p = params.toObject();
    const int direction =
        p.value("direction").toString() == QLatin1String("push") ? 1 : -1;

    std::optional<bool> alphaOverride;
    if (p.contains("alpha"))
        alphaOverride = p.value("alpha").toBool();

    // Push/pull requires editing to be enabled with an active session.
    if (!widget || !widget->isEditingEnabled())
        throw AgentBridgeError{-32008, "Segmentation editing is not enabled", {}};
    if (!mod->hasActiveSession()) {
        QJsonObject data;
        data["kind"] = "session";
        data["detail"] = "no active segmentation edit session";
        throw AgentBridgeError{-32007, "No active edit session", data};
    }

    // startPushPull returns false when there is no valid hover target (the agent
    // must position the cursor first with a buttonless canvas.drag — that
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
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    // Run Trace runs vc_grow_seg_from_segments: the "tool" source.
    requireSourceIdle(QStringLiteral("tool"));

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Run Trace unavailable", data};
    }

    SegmentationCommandHandler::RunTraceParams rt;
    const QJsonValue overrides = p.value("paramOverrides");
    if (!overrides.isUndefined() && !overrides.isNull())
        rt.paramOverrides = overrides.toObject();
    if (p.contains("ompThreads"))
        rt.ompThreads = p.value("ompThreads").toInt();
    rt.tgtDir = p.value("outputDir").toString();

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("tracer.run_trace"),
                                   QStringLiteral("Run Trace"),
                                   /*broadcastStart=*/false);

    CommandLaunchError error;
    QString outputDir;
    if (!handler->startRunTrace(
            segmentIdQ.toStdString(), rt, &error, &outputDir)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, "Failed to start Run Trace",
                                segmentIdQ, "tool");
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
// render.tifxyz
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleRenderTifxyz(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
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
    } else
        rp.outputFormat = CommandLineToolRunner::RenderOutputFormat::TifStack;

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

    rp.scale = static_cast<float>(p.value("scale").toDouble(1.0));
    rp.groupIdx = p.value("groupIdx").toInt(0);
    rp.numSlices = p.value("numSlices").toInt(1);

    // voxelSize: optional override; when omitted the tool derives it from volume
    // metadata (matching the interactive render path).
    if (p.contains("voxelSize") && !p.value("voxelSize").isNull()) {
        rp.hasVoxelSize = true;
        rp.voxelSizeUm = p.value("voxelSize").toDouble();
    }

    rp.outputDir = p.value("outputDir").toString();

    // Render runs the external vc_render_tifxyz process: the "tool" source.
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

    CommandLaunchError error;
    QString outputDir;
    if (!handler->startRenderSegment(
            segmentIdQ.toStdString(), rp, &error, &outputDir)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, "Failed to start render",
                                segmentIdQ, "tool");
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
