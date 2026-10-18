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


// ---------------------------------------------------------------------------
// segment.neighbor_copy
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentNeighborCopy(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);

    const QString segmentId = p.value("segmentId").toString();
    requireKnownSegment(state, segmentId);
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
    copy.outputDir = p.value("outputDir").toString();
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
    copy.pass2OmpThreads = p.value("pass2OmpThreads").toInt(copy.pass2OmpThreads);
    copy.pass2ParamOverrides = p.value("pass2ParamOverrides").toObject();

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Neighbor copy"));

    QString outputDir;
    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segment.neighbor_copy"),
        copyOut ? QStringLiteral("Copy segment out") : QStringLiteral("Copy segment in"),
        segmentId, QStringLiteral("neighbor_copy"),
        [&](CommandLaunchError* error, QString* jobOutput) {
            const bool ok = handler->startNeighborCopy(segmentId, copyOut, copy, error, &outputDir);
            *jobOutput = outputDir;
            return ok;
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segment.neighbor_copy";
    result["source"] = "tool";
    result["segmentId"] = segmentId;
    result["direction"] = copyOut ? "out" : "in";
    result["outputDir"] = outputDir;
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
    CState* state = requireProject(_window ? _window->_state : nullptr);

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

    // The grid rules of the Merge tifxyz dialog, checked here so a bad grid is
    // a parameter error and an unknown name a lookup error.
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
                QStringLiteral("Segment name '%1' contains whitespace, which merge.json "
                               "cannot express").arg(name),
                QJsonObject{{"param", param}}};
        }
    }
    merge.refSurface = p.value("refSegmentId").toString();
    if (!merge.refSurface.isEmpty() && !names.contains(merge.refSurface)) {
        throw AgentBridgeError{-32602, "refSegmentId must be one of the merged segments",
                               QJsonObject{{"param", "refSegmentId"}}};
    }
    merge.stripCols = p.value("stripCols").toInt(merge.stripCols);
    readRansacParams(p, merge);
    for (const QString& name : names)
        requireKnownSegment(state, name);

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Merge"));

    QString mergeJsonPath;
    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segments.merge"), QStringLiteral("Merge tifxyz"),
        names.front(), QStringLiteral("merge"),
        [&](CommandLaunchError* error, QString*) {
            return handler->startMergeTifxyz(merge, error, &mergeJsonPath);
        });

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segments.merge";
    result["source"] = "tool";
    result["segmentIds"] = toJsonArray(names);
    result["mergeJsonPath"] = mergeJsonPath;
    return result;
}

QJsonObject AgentBridgeServer::handleSegmentsPatch(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);

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
    requireConfirm(p, QStringLiteral("patch overwrites the parent segment in place"));
    for (const QString& id : segmentIds)
        requireKnownSegment(state, id);

    // Either segment may become the parent, which is rewritten on disk.
    for (const QString& id : segmentIds)
        requireSegmentFilesIdle(id, QStringLiteral("patch"));

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Patch"));

    const QString jobId = launchSegmentToolJob(
        QStringLiteral("segments.patch"), QStringLiteral("Patch tifxyz"),
        patch.parentId, QStringLiteral("patch"),
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


// ---------------------------------------------------------------------------
// segment.restore_backup
// ---------------------------------------------------------------------------

namespace {

QJsonArray backupsJson(const std::vector<vc3d::segment_files::Backup>& backups)
{
    QJsonArray out;
    for (const auto& backup : backups) {
        out.append(QJsonObject{{"index", backup.index},
                               {"timestamp", backup.timestamp.isEmpty()
                                                 ? QJsonValue(QJsonValue::Null)
                                                 : QJsonValue(backup.timestamp)}});
    }
    return out;
}

}  // namespace

QJsonObject AgentBridgeServer::handleSegmentRestoreBackup(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);
    const QString segmentId = p.value("segmentId").toString();
    const int index = p.value("index").toInt();
    requireKnownSegment(state, segmentId);

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Restore"));

    // The listing comes before the confirm check, so a call without confirm
    // reports the backups and changes nothing.
    const auto backups = handler->listSurfaceBackups(segmentId);
    const QJsonArray available = backupsJson(backups);
    const bool known = std::any_of(backups.begin(), backups.end(),
                                   [index](const auto& backup) { return backup.index == index; });
    if (!known) {
        throw AgentBridgeError{
            -32007, QStringLiteral("Backup %1 not found for %2").arg(index).arg(segmentId),
            QJsonObject{{"kind", "backup"}, {"id", index}, {"available", available}}};
    }
    if (!p.value("confirm").toBool(false)) {
        throw AgentBridgeError{
            -32602, "confirm must be true (restoring a backup overwrites the segment)",
            QJsonObject{{"param", "confirm"},
                        {"reason", "destructive; pass confirm=true"},
                        {"available", available}}};
    }
    requireSegmentFilesIdle(segmentId, QStringLiteral("restore"));

    QString err;
    int copiedFiles = 0;
    if (!handler->restoreSurfaceFromBackup(segmentId, index, &err, &copiedFiles)) {
        throw AgentBridgeError{-32005, "Restore failed", QJsonObject{{"detail", err}}};
    }

    QJsonObject result;
    result["restored"] = true;
    result["segmentId"] = segmentId;
    result["index"] = index;
    result["copiedFiles"] = copiedFiles;
    // The replaced state is now the newest backup.
    result["backups"] = backupsJson(handler->listSurfaceBackups(segmentId));
    return result;
}


// ---------------------------------------------------------------------------
// segments.move_to_paths
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentsMoveToPaths(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);
    const QString segmentId = p.value("segmentId").toString();
    requireKnownSegment(state, segmentId);
    if (state->vpkg()->getSegmentationDirectory() != "traces") {
        throw AgentBridgeError{
            -32009, "Only segments in the traces directory can move to paths",
            QJsonObject{{"detail", QString::fromStdString(
                                       state->vpkg()->getSegmentationDirectory())}}};
    }
    requireConfirm(p, QStringLiteral("moving unloads the segment and changes its folder"));
    // Moving unloads the segment, which an editing session on any segment
    // may still reference through the shared surface state.
    requireSegmentFilesIdle(QString(), QStringLiteral("move"));

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Move"));

    const bool replaceExisting = p.value("replaceExisting").toBool(false);
    bool destinationExists = false;
    QString err;
    QString newPath;
    if (!handler->moveSegmentToPaths(segmentId, replaceExisting, &err, &newPath,
                                     &destinationExists)) {
        if (destinationExists && !replaceExisting) {
            throw AgentBridgeError{
                -32602, "A segment of this name already exists in paths/",
                QJsonObject{{"param", "replaceExisting"},
                            {"detail", "pass replaceExisting=true to replace it"}}};
        }
        throw AgentBridgeError{-32005, "Move failed", QJsonObject{{"detail", err}}};
    }

    QJsonObject result;
    result["moved"] = true;
    result["segmentId"] = segmentId;
    result["path"] = newPath;
    result["replaced"] = destinationExists;
    return result;
}
