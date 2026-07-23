#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"
#include "SegmentationCommandHandler.hpp"
#include "SurfaceAreaCalculator.hpp"
#include "CommandLineToolRunner.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Per-segment mesh operations. Sync dialog-free ops (crop, recalc_area) call a
// SegmentationCommandHandler / SurfaceAreaCalculator entry directly; async ops
// drive a headless start* launcher tracked as a job -- external-tool ones share
// the source:"tool" slot (§8.3, like render.tifxyz), the in-process mask render
// resolves via the §8.4 deferred mechanism.
// ===========================================================================


// ---------------------------------------------------------------------------
// segment.crop_bounds (Stage 1, synchronous)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentCropBounds(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    // The crop core requires a current volume (dialog-free mirror of
    // requireSurfaceAndRunner); reject up front for a clean precondition error.
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = jsonRequireString(p, "segmentId");
    if (!state->vpkg()->getSurface(segmentId.toStdString())) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentId;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Crop unavailable", data};
    }

    // A mask render mutates a surface's meta on a background thread; crop rewrites
    // that same in-memory QuadSurface (points/channels/meta) on this GUI thread.
    // Refuse to overlap the two -- same coarse guard the mask handler applies to
    // itself -- so a bridge caller cannot drive them into a data race.
    if (_window->_maskRenderInProgress) {
        QJsonObject data;
        data["detail"] = "a mask render is in progress";
        throw AgentBridgeError{-32004, "A mask render is in progress", data};
    }

    // Synchronous dialog-free core. Failures report through `err` (not a
    // QMessageBox) so they become a real -32005 instead of a false cropped:true;
    // a no-op (already tightest) still reports success.
    QString err;
    if (!handler->cropSurfaceToValidRegion(segmentId.toStdString(), &err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Crop failed", data};
    }

    QJsonObject result;
    result["cropped"] = true;
    result["segmentId"] = segmentId;
    return result;
}


// ---------------------------------------------------------------------------
// segment.recalc_area (Stage 1, synchronous)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentRecalcArea(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QJsonValue idsVal = p.value("segmentIds");
    if (!idsVal.isArray()) {
        QJsonObject data;
        data["param"] = "segmentIds";
        throw AgentBridgeError{-32602, "segmentIds must be an array of strings", data};
    }
    const QJsonArray idsArr = idsVal.toArray();
    if (idsArr.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentIds";
        throw AgentBridgeError{-32602, "segmentIds must not be empty", data};
    }

    std::vector<std::string> ids;
    ids.reserve(static_cast<size_t>(idsArr.size()));
    for (const QJsonValue& v : idsArr) {
        if (!v.isString()) {
            QJsonObject data;
            data["param"] = "segmentIds";
            throw AgentBridgeError{-32602, "segmentIds must be an array of strings", data};
        }
        ids.push_back(v.toString().toStdString());
    }

    // Pure computation, no UI: per-segment results carry their own success flag,
    // so an unknown/invalid id is reported in-band (success:false) rather than
    // failing the whole call.
    const std::vector<AreaResult> results =
        SurfaceAreaCalculator::calculateAreas(state->vpkg(), state->currentVolume(), ids);

    QJsonArray out;
    for (const AreaResult& r : results) {
        QJsonObject o;
        o["segmentId"] = QString::fromStdString(r.segmentId);
        o["areaVx2"] = r.areaVx2;
        o["areaCm2"] = r.areaCm2;
        o["success"] = r.success;
        o["errorReason"] = r.errorReason.empty()
            ? QJsonValue(QJsonValue::Null)
            : QJsonValue(QString::fromStdString(r.errorReason));
        out.append(o);
    }

    QJsonObject result;
    result["results"] = out;
    return result;
}


// ---------------------------------------------------------------------------
// segment.reoptimize (Stage 2, async, source:"tool")
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentReoptimize(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = jsonRequireString(p, "segmentId");
    if (!state->vpkg()->getSurface(segmentId.toStdString())) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentId;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }

    SegmentationCommandHandler::ResumeLocalGrowParams rp;
    rp.volumeId = jsonOptionalString(p, "volumeId");
    // Validate volumeId up front (clean -32007 before the job is registered).
    if (!rp.volumeId.isEmpty()) {
        const auto vids = state->vpkg()->volumeIDs();
        if (std::find(vids.begin(), vids.end(), rp.volumeId.toStdString()) == vids.end()) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = rp.volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(rp.volumeId), data};
        }
    }
    rp.ompThreads = jsonOptionalInt(p, "ompThreads", rp.ompThreads);
    if (rp.ompThreads < 0) {
        QJsonObject data;
        data["param"] = "ompThreads";
        throw AgentBridgeError{-32602, "ompThreads must be >= 0", data};
    }
    if (p.contains("paramOverrides")) {
        const QJsonValue ov = p.value("paramOverrides");
        if (!ov.isObject()) {
            QJsonObject data;
            data["param"] = "paramOverrides";
            throw AgentBridgeError{-32602, "paramOverrides must be an object", data};
        }
        rp.paramOverrides = ov.toObject();
    }

    // Reoptimize runs vc_grow_seg_from_segments via the runner: the "tool" source.
    requireSourceIdle(QStringLiteral("tool"));

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Reoptimize unavailable", data};
    }

    const QString effectiveVolumeId = rp.volumeId.isEmpty()
        ? QString::fromStdString(state->currentVolumeId())
        : rp.volumeId;

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("segment.reoptimize"),
                                   QStringLiteral("Reoptimize segment"),
                                   /*broadcastStart=*/false);

    // Suppress the runner's interactive completion dialog for this headless run
    // (auto-cleared on toolFinished; cleared on the sync-failure path below), so
    // a modal cannot starve the toolFinished slots that resolve the job.
    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    QString err;
    QString outputDir;
    if (!handler->startResumeLocalGrowPatch(segmentId.toStdString(), rp, &err, &outputDir)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJobs.remove(QStringLiteral("tool"));
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentId;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("Unknown volume id"))) {
            QJsonObject data;
            data["kind"] = "volume";
            data["detail"] = err;
            throw AgentBridgeError{-32007, err, data};
        }
        if (err.contains(QLatin1String("already running")) ||
            err.contains(QLatin1String("already active"))) {
            QJsonObject data;
            data["source"] = "tool";
            data["detail"] = err;
            throw AgentBridgeError{-32004, "A tool job is already running", data};
        }
        if (err.contains(QLatin1String("Command line tools are not available")) ||
            err.contains(QLatin1String("not found or not executable"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "Command line tools unavailable", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start reoptimize", data};
    }

    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.reoptimize";
    result["source"] = "tool";
    result["outputDir"] = outputDir;
    result["volumeId"] = effectiveVolumeId;
    return result;
}


// ---------------------------------------------------------------------------
// segment.refine_alpha_comp (Stage 2, async, source:"tool")
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentRefineAlphaComp(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = jsonRequireString(p, "segmentId");
    if (!state->vpkg()->getSurface(segmentId.toStdString())) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentId;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }

    SegmentationCommandHandler::AlphaCompRefineParams rp;
    // Optional overrides; each present value is strictly typed (SPEC §5). Absent
    // values keep the dialog-default already in the struct.
    auto optFinite = [&](const char* key, double& target) {
        if (p.contains(key)) target = jsonRequireFinite(p.value(key), key);
    };
    rp.refine = jsonOptionalBool(p, "refine", rp.refine);
    optFinite("start", rp.start);
    optFinite("stop", rp.stop);
    optFinite("step", rp.step);
    rp.low = jsonOptionalInt(p, "low", rp.low);
    rp.high = jsonOptionalInt(p, "high", rp.high);
    optFinite("borderOff", rp.borderOff);
    rp.radius = jsonOptionalInt(p, "radius", rp.radius);
    rp.genVertexColor = jsonOptionalBool(p, "genVertexColor", rp.genVertexColor);
    rp.overwrite = jsonOptionalBool(p, "overwrite", rp.overwrite);
    optFinite("readerScale", rp.readerScale);
    rp.scaleGroup = jsonOptionalString(p, "scaleGroup", rp.scaleGroup);
    rp.ompThreads = jsonOptionalInt(p, "ompThreads", rp.ompThreads);
    rp.outputDir = jsonOptionalString(p, "outputDir");

    // Refinement runs vc_objrefine via the runner: the "tool" source.
    requireSourceIdle(QStringLiteral("tool"));

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Refine unavailable", data};
    }

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("segment.refine_alpha_comp"),
                                   QStringLiteral("Refine segment (alpha-comp)"),
                                   /*broadcastStart=*/false);

    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    QString err;
    QString outputDir;
    if (!handler->startAlphaCompRefine(segmentId.toStdString(), rp, &err, &outputDir)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJobs.remove(QStringLiteral("tool"));
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentId;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("only local volumes"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32009, "Remote volume not supported", data};
        }
        if (err.contains(QLatin1String("already running"))) {
            QJsonObject data;
            data["source"] = "tool";
            data["detail"] = err;
            throw AgentBridgeError{-32004, "A tool job is already running", data};
        }
        if (err.contains(QLatin1String("Command line tools are not available")) ||
            err.contains(QLatin1String("not found or not executable"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "Command line tools unavailable", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start refinement", data};
    }

    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.refine_alpha_comp";
    result["source"] = "tool";
    result["outputDir"] = outputDir;
    result["segmentId"] = segmentId;
    return result;
}


// ---------------------------------------------------------------------------
// segment.generate_mask / segment.append_mask (Stage 3, async in-process,
// deferred response — SPEC §8.4)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentGenerateMask(const QJsonValue& params)
{
    return handleSegmentMask(params, /*append=*/false);
}

QJsonObject AgentBridgeServer::handleSegmentAppendMask(const QJsonValue& params)
{
    return handleSegmentMask(params, /*append=*/true);
}


QJsonObject AgentBridgeServer::handleSegmentMask(const QJsonValue& params, bool append)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentId = jsonRequireString(p, "segmentId");
    auto surf = state->vpkg()->getSurface(segmentId.toStdString());
    if (!surf) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentId;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }
    // Appending renders a volume-image layer, so a current volume is required
    // (mirrors onAppendMaskPressed); the binary-mask generate path does not.
    if (append && !state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const std::filesystem::path maskPath = surf->path / "mask.tif";
    const QString maskPathStr = QString::fromStdString(maskPath.string());

    // generate_mask mirrors onEditMaskPressed: an existing mask.tif is NOT
    // regenerated (the GUI simply opens it). Report that synchronously instead of
    // launching a render.
    if (!append && std::filesystem::exists(maskPath)) {
        QJsonObject result;
        result["generated"] = false;
        result["alreadyExists"] = true;
        result["maskPath"] = maskPathStr;
        result["segmentId"] = segmentId;
        return result;
    }

    if (_window->_maskRenderInProgress) {
        QJsonObject data;
        data["detail"] = "a mask render is already in progress";
        throw AgentBridgeError{-32004, "A mask render is already in progress", data};
    }

    // The render runs on a QtConcurrent worker with no bridge-visible completion
    // signal today, so resolve the RPC via the deferred mechanism (§8.4): the
    // reply is written from the worker's finished callback.
    const int token = beginDeferred(
        120000, append ? QStringLiteral("append mask render")
                        : QStringLiteral("generate mask render"));
    const QString failTitle = append ? QStringLiteral("Append mask failed")
                                      : QStringLiteral("Generate mask failed");

    // Once beginDeferred() has armed the token, this handler must resolve it
    // exactly once and then throw AgentBridgeDeferred. Any other exception
    // escaping here would let dispatch send a second reply for an id the pending
    // deferred entry (and its timeout timer) still owns -- a double response.
    // startMaskRenderHeadless can throw during setup, so catch and convert to a
    // single deferred error. (Its own scope guard has already cleared
    // _maskRenderInProgress on that path.)
    try {
        QString err;
        const bool launched = _window->startMaskRenderHeadless(
            segmentId, append,
            [this, token, maskPathStr, segmentId, append](bool success, QString message) {
                if (success) {
                    QJsonObject result;
                    result["generated"] = true;
                    result["appended"] = append;
                    result["maskPath"] = maskPathStr;
                    result["segmentId"] = segmentId;
                    result["message"] = message;
                    completeDeferredResult(token, result);
                } else {
                    QJsonObject data;
                    data["detail"] = message;
                    completeDeferredError(token, -32005,
                                          append ? QStringLiteral("Append mask failed")
                                                 : QStringLiteral("Generate mask failed"),
                                          data);
                }
            },
            &err);

        if (!launched) {
            // Preconditions were re-checked inside the launcher; surface its sentence.
            QJsonObject data;
            data["detail"] = err;
            completeDeferredError(token, -32005, failTitle, data);
        }
    } catch (const std::exception& e) {
        QJsonObject data;
        data["detail"] = QString::fromUtf8(e.what());
        completeDeferredError(token, -32005, failTitle, data);
    }

    throw AgentBridgeDeferred{};
}
