#include "agent_bridge/AgentBridgeServer.hpp"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <QRegularExpression>
#include <QStringList>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"
#include "CommandLaunchError.hpp"
#include "SegmentationCommandHandler.hpp"
#include "SurfaceCatalog.hpp"
#include "SurfacePanelController.hpp"
#include "segmentation/SegmentationWidget.hpp"

#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Segment operations over SegmentationCommandHandler's dialog-free cores.
// The context-menu and Actions-menu slots run the same cores with their
// dialogs, so both callers share validation, store writes and UI refresh.
// Segment ids are the object store's surface UUIDs.
// ===========================================================================

namespace {

CState* requireProject(CState* state)
{
    if (!state || !state->hasVpkg() || !state->vpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    return state;
}

// Checks that the surface catalog lists the surface without opening it; the
// operation cores check what they need of it and report that themselves.
void requireKnownSegment(SurfacePanelController* panel, const QString& segmentId)
{
    SurfaceCatalog* catalog = panel ? panel->catalog() : nullptr;
    if (!catalog || !catalog->row(segmentId.toStdString())) {
        throw AgentBridgeError{
            -32007, QStringLiteral("Segment not found: %1").arg(segmentId),
            QJsonObject{{"kind", "segment"}, {"id", segmentId}}};
    }
}

SegmentationCommandHandler* requireCommandHandler(SegmentationCommandHandler* handler,
                                                  const QString& operation)
{
    if (!handler) {
        throw AgentBridgeError{
            -32009, QStringLiteral("%1 unavailable").arg(operation),
            QJsonObject{{"detail", "segmentation command handler is not available"}}};
    }
    return handler;
}

// Destructive or lossy calls need an explicit confirm:true, like
// segments.delete.
void requireConfirm(const QJsonObject& p, const QString& reason)
{
    if (!p.value("confirm").toBool(false)) {
        throw AgentBridgeError{
            -32602, QStringLiteral("confirm must be true (%1)").arg(reason),
            QJsonObject{{"param", "confirm"},
                        {"reason", "destructive; pass confirm=true"}}};
    }
}

}  // namespace


// Refuses an in-place rewrite of a surface while something else may be
// reading or writing it: an editing session on that surface, or a
// command-line tool run.
void AgentBridgeServer::requireSegmentFilesIdle(const QString& segmentId,
                                                const QString& operation) const
{
    CState* state = _window ? _window->_state : nullptr;
    if (_window && _window->_segmentationWidget &&
        _window->_segmentationWidget->isEditingEnabled() && state &&
        (segmentId.isEmpty() || state->activeSurfaceId() == segmentId.toStdString())) {
        throw AgentBridgeError{
            -32004, QStringLiteral("Cannot %1 the segment being edited").arg(operation),
            QJsonObject{{"detail", "disable segmentation editing first"}}};
    }
    requireSourceIdle(QStringLiteral("tool"));
}


// ---------------------------------------------------------------------------
// segment.transform
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentTransform(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    requireKnownSegment(_window->_surfacePanel.get(), segmentId);

    // The descriptor restricts op to these spellings.
    const QString op = p.value("op").toString();
    using Transform = SegmentationCommandHandler::SurfaceTransform;
    const Transform transform = op == QLatin1String("flip_u")   ? Transform::FlipU
                              : op == QLatin1String("rotate90") ? Transform::Rotate90
                                                                : Transform::FlipV;
    // Flips are exact and their own inverse; a rotation resamples the grid.
    if (transform == Transform::Rotate90)
        requireConfirm(p, QStringLiteral("rotate90 resamples the grid in place"));

    // Transforming the grid under an open editing session would leave the
    // session's preview points indexed against the old layout.
    requireSegmentFilesIdle(segmentId, QStringLiteral("transform"));

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Transform"));

    QString err;
    if (!handler->transformSurface(segmentId.toStdString(), transform, &err)) {
        throw AgentBridgeError{-32005, "Transform failed", QJsonObject{{"detail", err}}};
    }

    QJsonObject result;
    result["transformed"] = true;
    result["segmentId"] = segmentId;
    result["op"] = op;
    return result;
}


// ---------------------------------------------------------------------------
// Tool jobs for segment operations
// ---------------------------------------------------------------------------

QString AgentBridgeServer::launchSegmentToolJob(
    const QString& kind,
    const QString& label,
    const QString& segmentId,
    const QString& closingOperation,
    const std::function<bool(CommandLaunchError*, QString*)>& launch)
{
    requireSourceIdle(QStringLiteral("tool"));

    const QString jobId = beginJob(QStringLiteral("tool"), kind, label,
                                   /*broadcastStart=*/false);
    _activeJobs[QStringLiteral("tool")].closingOperation = closingOperation;

    CommandLaunchError error;
    QString outputPath;
    if (!launch(&error, &outputPath)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, QStringLiteral("Failed to start %1").arg(label),
                                segmentId, QStringLiteral("tool"));
    }

    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputPath;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }
    return jobId;
}


// ---------------------------------------------------------------------------
// segment.export_obj
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentExportObj(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    requireKnownSegment(_window->_surfacePanel.get(), segmentId);

    SegmentationCommandHandler::ConvertToObjParams objParams;
    if (p.contains("objName")) {
        // The tool writes into the surface's artifact directory.
        objParams.objName = p.value("objName").toString().trimmed();
        if (objParams.objName.isEmpty() || objParams.objName.contains(QLatin1Char('/')) ||
            objParams.objName.contains(QLatin1Char('\\')) ||
            !objParams.objName.endsWith(QStringLiteral(".obj"), Qt::CaseInsensitive)) {
            throw AgentBridgeError{
                -32602, "objName must be a .obj file name without directories",
                QJsonObject{{"param", "objName"}}};
        }
    }
    objParams.normalizeUV = p.value("normalizeUv").toBool(objParams.normalizeUV);
    objParams.alignGrid = p.value("alignGrid").toBool(objParams.alignGrid);
    objParams.ompThreads = p.value("ompThreads").toInt(objParams.ompThreads);

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("OBJ export"));

    // The OBJ's path arrives with the job's completion.
    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segment.export_obj"), QStringLiteral("Convert segment to OBJ"),
        segmentId, QString(),
        [&](CommandLaunchError* error, QString*) {
            return handler->startConvertToObj(segmentId.toStdString(), objParams, error);
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.export_obj";
    result["source"] = "tool";
    result["segmentId"] = segmentId;
    return result;
}


// ---------------------------------------------------------------------------
// segment.neighbor_copy
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentNeighborCopy(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);

    const QString segmentId = p.value("segmentId").toString();
    requireKnownSegment(_window->_surfacePanel.get(), segmentId);
    const bool copyOut = p.value("direction").toString() == QLatin1String("out");

    SegmentationCommandHandler::NeighborCopyParams copy;
    copy.volumeId = p.value("volumeId").toString();
    if (!copy.volumeId.isEmpty()) {
        const auto ids = state->vpkg()->volumeIDs();
        if (std::find(ids.begin(), ids.end(), copy.volumeId.toStdString()) == ids.end()) {
            throw AgentBridgeError{
                -32007, QStringLiteral("Unknown volume id: %1").arg(copy.volumeId),
                QJsonObject{{"kind", "volume"}, {"id", copy.volumeId}}};
        }
    }
    copy.outputTag = p.value("outputTag").toString(copy.outputTag).trimmed();
    if (copy.outputTag.isEmpty()) {
        throw AgentBridgeError{-32602, "outputTag must not be empty",
                               QJsonObject{{"param", "outputTag"}}};
    }
    copy.neighborMaxDistance = p.value("neighborMaxDistance").toInt(copy.neighborMaxDistance);
    copy.neighborMinClearance = p.value("neighborMinClearance").toInt(copy.neighborMinClearance);
    copy.neighborFill = p.value("neighborFill").toBool(copy.neighborFill);
    copy.neighborInterpWindow = p.value("neighborInterpWindow").toInt(copy.neighborInterpWindow);
    copy.generations = p.value("generations").toInt(copy.generations);
    copy.neighborSpikeWindow = p.value("neighborSpikeWindow").toInt(copy.neighborSpikeWindow);
    copy.resumeLocalOptStep = p.value("resumeLocalOptStep").toInt(copy.resumeLocalOptStep);
    // The dialog's radius default follows the step (twice the step).
    copy.resumeLocalOptRadius = p.value("resumeLocalOptRadius").toInt(copy.resumeLocalOptStep * 2);
    copy.resumeLocalMaxIters = p.value("resumeLocalMaxIters").toInt(copy.resumeLocalMaxIters);
    copy.resumeLocalDenseQr = p.value("resumeLocalDenseQr").toBool(copy.resumeLocalDenseQr);
    copy.refineOmpThreads = p.value("refineOmpThreads").toInt(copy.refineOmpThreads);
    copy.refineParamOverrides = p.value("refineParamOverrides").toObject();

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Neighbor copy"));

    // One tool run projects and refines; the new surface's UUID arrives with
    // the job's completion.
    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segment.neighbor_copy"),
        copyOut ? QStringLiteral("Copy segment out") : QStringLiteral("Copy segment in"),
        segmentId, QString(),
        [&](CommandLaunchError* error, QString*) {
            return handler->startNeighborCopy(segmentId, copyOut, copy, error);
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.neighbor_copy";
    result["source"] = "tool";
    result["segmentId"] = segmentId;
    result["direction"] = copyOut ? "out" : "in";
    result["outputTag"] = copy.outputTag;
    return result;
}
