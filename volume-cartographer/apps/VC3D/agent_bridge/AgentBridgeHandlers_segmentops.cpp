#include "agent_bridge/AgentBridgeServer.hpp"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <QStringList>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"
#include "CommandLaunchError.hpp"
#include "SegmentationCommandHandler.hpp"
#include "segmentation/SegmentationWidget.hpp"

#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Segment file operations over SegmentationCommandHandler's dialog-free
// cores. The context-menu and Actions-menu slots run the same cores with
// their dialogs, so both callers share validation, persistence and UI
// refresh.
// ===========================================================================

namespace {

CState* requireProject(CState* state)
{
    if (!state || !state->hasVpkg() || !state->vpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    return state;
}

// Checks the package's segment ids without loading any surface; the operation
// cores load what they need and report load failures themselves.
void requireKnownSegment(CState* state, const QString& segmentId)
{
    const std::vector<std::string> ids = state->vpkg()->segmentationIDs();
    if (std::find(ids.begin(), ids.end(), segmentId.toStdString()) == ids.end()) {
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

QJsonArray toJsonArray(const QStringList& values)
{
    QJsonArray out;
    for (const QString& value : values)
        out.append(value);
    return out;
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


// Refuses an in-place rewrite of a segment's files while something else may
// be reading or writing them: an editing session on that segment, a mask
// render, or a command-line tool run.
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
    if (_window && _window->_maskRenderInProgress) {
        throw AgentBridgeError{
            -32004, "A mask render is in progress",
            QJsonObject{{"detail", "a mask render is in progress"}}};
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
    requireKnownSegment(state, segmentId);

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
    requireKnownSegment(state, segmentId);

    SegmentationCommandHandler::ConvertToObjParams objParams;
    if (p.contains("outputPath")) {
        objParams.objPath = p.value("outputPath").toString().trimmed();
        if (objParams.objPath.isEmpty() || QDir::isRelativePath(objParams.objPath) ||
            !objParams.objPath.endsWith(QStringLiteral(".obj"), Qt::CaseInsensitive)) {
            throw AgentBridgeError{
                -32602, "outputPath must be an absolute path to a .obj file",
                QJsonObject{{"param", "outputPath"}}};
        }
        objParams.objPath = QDir::cleanPath(objParams.objPath);
    }
    objParams.overwrite = p.value("overwrite").toBool(false);
    objParams.normalizeUV = p.value("normalizeUv").toBool(objParams.normalizeUV);
    objParams.alignGrid = p.value("alignGrid").toBool(objParams.alignGrid);
    objParams.ompThreads = p.value("ompThreads").toInt(objParams.ompThreads);

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("OBJ export"));

    QString objPath;
    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segment.export_obj"), QStringLiteral("Convert segment to OBJ"),
        segmentId, QString(),
        [&](CommandLaunchError* error, QString* outputPath) {
            const bool ok = handler->startConvertToObj(segmentId.toStdString(), objParams,
                                                       error, &objPath);
            *outputPath = objPath;
            return ok;
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.export_obj";
    result["source"] = "tool";
    result["segmentId"] = segmentId;
    result["outputPath"] = objPath;
    return result;
}


// ---------------------------------------------------------------------------
// segment.rasterize
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentRasterize(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);

    QStringList segmentIds;
    for (const QJsonValue& value : p.value("segmentIds").toArray()) {
        const QString id = value.toString();
        if (id.trimmed().isEmpty()) {
            throw AgentBridgeError{-32602, "segmentIds must not contain blank ids",
                                   QJsonObject{{"param", "segmentIds"}}};
        }
        if (!segmentIds.contains(id))
            segmentIds.append(id);
    }
    if (segmentIds.isEmpty()) {
        throw AgentBridgeError{-32602, "segmentIds must not be empty",
                               QJsonObject{{"param", "segmentIds"}}};
    }
    for (const QString& id : segmentIds)
        requireKnownSegment(state, id);

    SegmentationCommandHandler::RasterizeParams rasterParams;
    rasterParams.chunkSize = p.value("chunkSize").toInt(rasterParams.chunkSize);
    rasterParams.zMin = p.value("zMin").toInt(rasterParams.zMin);
    rasterParams.zMax = p.value("zMax").toInt(rasterParams.zMax);
    if (rasterParams.zMax >= 0 && rasterParams.zMin >= rasterParams.zMax) {
        throw AgentBridgeError{-32602, "zMin must be smaller than zMax unless zMax is -1",
                               QJsonObject{{"param", "zMax"}}};
    }

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Rasterize"));

    SegmentationCommandHandler::RasterizeLaunch launch;
    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segment.rasterize"), QStringLiteral("Rasterize segments"),
        segmentIds.front(), QStringLiteral("rasterize"),
        [&](CommandLaunchError* error, QString* outputPath) {
            const bool ok = handler->startRasterizeSegments(segmentIds, rasterParams,
                                                            error, &launch);
            *outputPath = launch.outputPath;
            return ok;
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.rasterize";
    result["source"] = "tool";
    result["segmentIds"] = toJsonArray(launch.segmentIds);
    result["skippedIds"] = toJsonArray(launch.skippedIds);
    result["outputPath"] = launch.outputPath;
    return result;
}
