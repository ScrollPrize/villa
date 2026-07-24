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
// the source:"tool" slot like render.tifxyz; the in-process mask render resolves
// via the deferred-response mechanism.
// ===========================================================================


// ---------------------------------------------------------------------------
// segment.crop_bounds
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentCropBounds(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    // The crop core requires a current volume (dialog-free mirror of
    // requireSurfaceAndRunner); reject up front for a clean precondition error.
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
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
// segment.recalc_area
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentRecalcArea(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QJsonArray idsArr = p.value("segmentIds").toArray();
    if (idsArr.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentIds";
        throw AgentBridgeError{-32602, "segmentIds must not be empty", data};
    }

    std::vector<std::string> ids;
    ids.reserve(static_cast<size_t>(idsArr.size()));
    for (const QJsonValue& v : idsArr)
        ids.push_back(v.toString().toStdString());

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
// segment.reoptimize
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentReoptimize(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (!state->vpkg()->getSurface(segmentId.toStdString())) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentId;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }

    SegmentationCommandHandler::ResumeLocalGrowParams rp;
    rp.volumeId = p.value("volumeId").toString();
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
    rp.ompThreads = p.value("ompThreads").toInt(rp.ompThreads);
    rp.paramOverrides = p.value("paramOverrides").toObject();

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

    CommandLaunchError error;
    QString outputDir;
    if (!handler->startResumeLocalGrowPatch(
            segmentId.toStdString(), rp, &error, &outputDir)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, "Failed to start reoptimize",
                                segmentId, "tool");
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
// segment.refine_alpha_comp
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentRefineAlphaComp(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (!state->vpkg()->getSurface(segmentId.toStdString())) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentId;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }

    SegmentationCommandHandler::AlphaCompRefineParams rp;
    rp.refine = p.value("refine").toBool(rp.refine);
    rp.start = p.value("start").toDouble(rp.start);
    rp.stop = p.value("stop").toDouble(rp.stop);
    rp.step = p.value("step").toDouble(rp.step);
    rp.low = p.value("low").toInt(rp.low);
    rp.high = p.value("high").toInt(rp.high);
    rp.borderOff = p.value("borderOff").toDouble(rp.borderOff);
    rp.radius = p.value("radius").toInt(rp.radius);
    rp.genVertexColor =
        p.value("genVertexColor").toBool(rp.genVertexColor);
    rp.overwrite = p.value("overwrite").toBool(rp.overwrite);
    rp.readerScale = p.value("readerScale").toDouble(rp.readerScale);
    rp.scaleGroup = p.value("scaleGroup").toString(rp.scaleGroup);
    if (p.contains("ompThreads"))
        rp.ompThreads = p.value("ompThreads").toInt();
    rp.outputDir = p.value("outputDir").toString();

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

    CommandLaunchError error;
    QString outputDir;
    if (!handler->startAlphaCompRefine(
            segmentId.toStdString(), rp, &error, &outputDir)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, "Failed to start refinement",
                                segmentId, "tool");
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
// segment.generate_mask / segment.append_mask (deferred in-process work)
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
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
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
    // signal today, so resolve the RPC via the deferred mechanism: the
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
    // startMaskRender can throw during setup, so catch and convert to a
    // single deferred error. (Its own scope guard has already cleared
    // _maskRenderInProgress on that path.)
    try {
        QString err;
        const bool launched = _window->startMaskRender(
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
