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
// segment.rasterize
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentRasterize(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireProject(_window ? _window->_state : nullptr);

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
        requireKnownSegment(_window->_surfacePanel.get(), id);

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


// ---------------------------------------------------------------------------
// segments.merge / segments.patch
// ---------------------------------------------------------------------------

namespace {

// RANSAC options shared by merge and patch; min must stay below max, as the
// dialogs require.
template <class Params>
void readRansacParams(const QJsonObject& p, Params& out)
{
    out.ransacIters = p.value("ransacIters").toInt(out.ransacIters);
    out.ransacMinThresh = p.value("ransacMinThresh").toDouble(out.ransacMinThresh);
    out.ransacMaxThresh = p.value("ransacMaxThresh").toDouble(out.ransacMaxThresh);
    out.ransacMadK = p.value("ransacMadK").toDouble(out.ransacMadK);
    out.ransacSeed = p.value("ransacSeed").toInt(out.ransacSeed);
    out.anchorCap = p.value("anchorCap").toInt(out.anchorCap);
    out.ompThreads = p.value("ompThreads").toInt(out.ompThreads);
    if (out.ransacMinThresh >= out.ransacMaxThresh) {
        throw AgentBridgeError{-32602, "ransacMinThresh must be smaller than ransacMaxThresh",
                               QJsonObject{{"param", "ransacMinThresh"}}};
    }
}

QStringList requireSegmentIdList(const QJsonObject& p)
{
    QStringList ids;
    for (const QJsonValue& value : p.value("segmentIds").toArray())
        ids.append(value.toString());
    return ids;
}

}  // namespace

QJsonObject AgentBridgeServer::handleSegmentsMerge(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireProject(_window ? _window->_state : nullptr);

    SegmentationCommandHandler::MergeTifxyzParams merge;
    if (p.contains("grid") == p.contains("segmentIds")) {
        throw AgentBridgeError{-32602, "Pass either grid or segmentIds",
                               QJsonObject{{"param", "grid"}}};
    }
    QString param = QStringLiteral("segmentIds");
    if (p.contains("grid")) {
        param = QStringLiteral("grid");
        for (const QJsonValue& rowValue : p.value("grid").toArray()) {
            QStringList row;
            for (const QJsonValue& cell : rowValue.toArray())
                row.append(cell.isNull() ? QString() : cell.toString().trimmed());
            merge.grid.append(row);
        }
    } else {
        // Without a grid the segments form one row, left to right.
        merge.grid.append(requireSegmentIdList(p));
    }

    // The grid rules of the Merge Surfaces dialog, checked here so a bad grid
    // is a parameter error and an unknown surface a lookup error.
    QStringList names;
    for (const QStringList& row : merge.grid) {
        for (const QString& name : row) {
            if (!name.isEmpty() && !names.contains(name))
                names.append(name);
        }
    }
    if (names.size() < 2) {
        throw AgentBridgeError{-32602, "merge needs at least two distinct segments",
                               QJsonObject{{"param", param}}};
    }
    for (const QString& name : names) {
        if (name.contains(QRegularExpression(QStringLiteral("\\s")))) {
            throw AgentBridgeError{
                -32602,
                QStringLiteral("Segment id '%1' contains whitespace, which merge.json "
                               "cannot express").arg(name),
                QJsonObject{{"param", param}}};
        }
    }
    merge.refSurface = p.value("refSegmentId").toString();
    if (!merge.refSurface.isEmpty() && !names.contains(merge.refSurface)) {
        throw AgentBridgeError{-32602, "refSegmentId must be one of the merged segments",
                               QJsonObject{{"param", "refSegmentId"}}};
    }
    merge.outputTag = p.value("outputTag").toString(merge.outputTag).trimmed();
    if (merge.outputTag.isEmpty()) {
        throw AgentBridgeError{-32602, "outputTag must not be empty",
                               QJsonObject{{"param", "outputTag"}}};
    }
    merge.stripCols = p.value("stripCols").toInt(merge.stripCols);
    readRansacParams(p, merge);
    for (const QString& name : names)
        requireKnownSegment(_window->_surfacePanel.get(), name);

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Merge"));

    // The merged surface's UUID arrives with the job's completion.
    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segments.merge"), QStringLiteral("Merge surfaces"),
        names.front(), QString(),
        [&](CommandLaunchError* error, QString*) {
            return handler->startMergeTifxyz(merge, error);
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segments.merge";
    result["source"] = "tool";
    result["segmentIds"] = toJsonArray(names);
    result["outputTag"] = merge.outputTag;
    return result;
}

QJsonObject AgentBridgeServer::handleSegmentsPatch(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireProject(_window ? _window->_state : nullptr);

    const QStringList segmentIds = requireSegmentIdList(p);
    if (segmentIds.size() != 2 || segmentIds[0] == segmentIds[1]) {
        throw AgentBridgeError{-32602, "patch needs exactly two different segmentIds",
                               QJsonObject{{"param", "segmentIds"}}};
    }
    const QString parentId = p.value("parentId").toString();
    if (!parentId.isEmpty() && !segmentIds.contains(parentId)) {
        throw AgentBridgeError{-32602, "parentId must be one of segmentIds",
                               QJsonObject{{"param", "parentId"}}};
    }

    SegmentationCommandHandler::MergePatchParams patch;
    patch.explicitRoles = !parentId.isEmpty();
    patch.parentId = patch.explicitRoles ? parentId : segmentIds[0];
    patch.childId = patch.parentId == segmentIds[0] ? segmentIds[1] : segmentIds[0];
    patch.borderCells = p.value("borderCells").toInt(patch.borderCells);
    patch.blendCells = p.value("blendCells").toInt(patch.blendCells);
    patch.idwK = p.value("idwK").toInt(patch.idwK);
    readRansacParams(p, patch);
    requireConfirm(p, QStringLiteral("patch rewrites the parent segment in place"));
    for (const QString& id : segmentIds)
        requireKnownSegment(_window->_surfacePanel.get(), id);

    // Either segment may become the parent, which gets a new revision.
    for (const QString& id : segmentIds)
        requireSegmentFilesIdle(id, QStringLiteral("patch"));

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Patch"));

    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segments.patch"), QStringLiteral("Patch surface"),
        patch.parentId, QString(),
        [&](CommandLaunchError* error, QString*) {
            return handler->startMergePatch(patch, error);
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segments.patch";
    result["source"] = "tool";
    result["segmentIds"] = toJsonArray(segmentIds);
    result["parentId"] = patch.explicitRoles ? QJsonValue(patch.parentId)
                                             : QJsonValue(QJsonValue::Null);
    return result;
}
