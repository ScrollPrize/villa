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
#include "CommandLineToolRunner.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Per-segment mesh operations. Sync dialog-free ops (crop) call a
// SegmentationCommandHandler / SurfaceAreaCalculator entry directly; async ops
// drive a headless start* launcher tracked as a job; external-tool ones share
// the source:"tool" slot like render.tifxyz.
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

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Crop unavailable", data};
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

    // Reoptimize runs vc_grow_seg_from_seed (a local resume) via the runner: the "tool" source.
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
    if (!handler->startResumeLocalGrowPatch(segmentId.toStdString(), rp, &error)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, "Failed to start reoptimize",
                                segmentId, "tool");
    }

    // The resumed surface's UUID arrives with the job's completion.
    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end())
        broadcastJobProgress(it.value(), QStringLiteral("started"));

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.reoptimize";
    result["source"] = "tool";
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
    rp.readerScale = p.value("readerScale").toDouble(rp.readerScale);
    rp.scaleGroup = p.value("scaleGroup").toString(rp.scaleGroup);
    if (p.contains("ompThreads"))
        rp.ompThreads = p.value("ompThreads").toInt();
    rp.outputName = p.value("outputName").toString();
    rp.inPlace = p.value("inPlace").toBool(rp.inPlace);

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
    if (!handler->startAlphaCompRefine(segmentId.toStdString(), rp, &error)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, "Failed to start refinement",
                                segmentId, "tool");
    }

    // The refined surface's UUID arrives with the job's completion.
    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end())
        broadcastJobProgress(it.value(), QStringLiteral("started"));

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.refine_alpha_comp";
    result["source"] = "tool";
    result["segmentId"] = segmentId;
    return result;
}

