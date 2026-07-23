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


// ---------------------------------------------------------------------------
// Line-annotation / fiber RPCs
// ---------------------------------------------------------------------------

namespace {

template <typename Operation>
QString captureFiberError(LineAnnotationController* controller, Operation&& operation)
{
    const bool wasSuppressed = controller->errorDialogsSuppressed();
    controller->setErrorDialogsSuppressed(true);
    (void)controller->takeLastSuppressedError();
    try {
        operation();
    } catch (...) {
        controller->setErrorDialogsSuppressed(wasSuppressed);
        throw;
    }
    const QString error = controller->takeLastSuppressedError();
    controller->setErrorDialogsSuppressed(wasSuppressed);
    return error;
}

// Parses a fiber id param: a decimal string (the canonical wire form because
// uint64 ids serialize as strings) or a non-negative integer number.
uint64_t jsonToFiberId(const QJsonValue& value, const char* paramName)
{
    QJsonObject data;
    data["param"] = QString::fromLatin1(paramName);
    if (value.isString()) {
        bool ok = false;
        const uint64_t id = value.toString().toULongLong(&ok);
        if (ok && id != 0)
            return id;
        data["value"] = value.toString();
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 must be a positive decimal fiber id")
                .arg(QLatin1String(paramName)), data};
    }
    if (value.isDouble()) {
        const double d = value.toDouble();
        // Reject a finite double that would overflow uint64 on cast (e.g. 1e300):
        // 2^64 is exactly representable as double, so require strictly below it.
        if (std::isfinite(d) && d > 0 && std::floor(d) == d && d < std::ldexp(1.0, 64))
            return static_cast<uint64_t>(d);
        data["value"] = value;
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 must be a positive integer fiber id")
                .arg(QLatin1String(paramName)), data};
    }
    throw AgentBridgeError{-32602,
        QStringLiteral("%1 is required (fiber id as a string)")
            .arg(QLatin1String(paramName)), data};
}
} // namespace


// Shared preamble: requires an open volume package and a live controller.
LineAnnotationController* AgentBridgeServer::fiberController() const
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    LineAnnotationController* ctrl =
        _window->_lineAnnotationController ? _window->_lineAnnotationController.get() : nullptr;
    if (!ctrl)
        throw AgentBridgeError{-32010, "Line annotation controller unavailable", {}};
    return ctrl;
}


// Throws -32007 kind:"fiber" unless `fiberId` is a currently-loaded fiber.
void AgentBridgeServer::requireKnownFiber(LineAnnotationController* ctrl, quint64 fiberId) const
{
    for (const auto& summary : ctrl->fiberSummaries()) {
        if (summary.id == fiberId)
            return;
    }
    QJsonObject data;
    data["kind"] = "fiber";
    data["id"] = QString::number(fiberId);
    throw AgentBridgeError{-32007, "Unknown fiber id", data};
}


QJsonObject AgentBridgeServer::handleFiberLaunch(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();

    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    auto* chunked = dynamic_cast<CChunkedVolumeViewer*>(viewer);
    if (!chunked) {
        QJsonObject data;
        data["detail"] = "viewer is not a chunked volume viewer";
        throw AgentBridgeError{-32009, "Unsupported viewer for fiber.launch", data};
    }
    if (!chunked->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    // Use the same position round-trip rule as canvas.click.
    const QString space =
        p.value("space").toString(QStringLiteral("volume"));
    QPointF scenePos;
    if (space == QLatin1String("scene")) {
        const QJsonObject po = p.value("position").toObject();
        scenePos = QPointF(jsonRequireFiniteFloat(po.value("x"), "x"),
                           jsonRequireFiniteFloat(po.value("y"), "y"));
    } else {
        const cv::Vec3f vol = jsonToVec3(p.value("position"), "position");
        scenePos = chunked->volumeToScene(vol);
        const cv::Vec3f back = chunked->sceneToVolume(scenePos);
        const double dist = cv::norm(back - vol);
        if (!std::isfinite(dist) || dist > 2.0) {
            QJsonObject data;
            data["point"] = vec3ToJson(vol);
            data["detail"] = QStringLiteral(
                "point is not on this viewer's view (round-trip %1 voxels)")
                    .arg(dist, 0, 'f', 3);
            throw AgentBridgeError{-32003, "Invalid coordinates", data};
        }
    }

    if (!ctrl->canLaunchFromViewer(chunked)) {
        QJsonObject data;
        data["detail"] = "this viewer's surface does not support launching the "
                         "line-annotation workspace";
        throw AgentBridgeError{-32009, "Cannot launch line annotation from this viewer", data};
    }
    if (!chunked->sampleSceneVolume(scenePos)) {
        QJsonObject data;
        data["detail"] = "the position does not sample a volume point on this viewer";
        throw AgentBridgeError{-32003, "Invalid coordinates", data};
    }

    const bool replaceOwning = p.value("replaceOwning").toBool(true);
    const QString err = captureFiberError(ctrl, [&] {
        ctrl->launchFromViewerAtPoint(chunked, scenePos, replaceOwning);
    });
    if (!err.isEmpty()) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "fiber.launch failed", data};
    }

    QJsonObject result;
    result["launched"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handleFiberList(const QJsonValue&)
{
    LineAnnotationController* ctrl = fiberController();

    QJsonArray fibers;
    for (const auto& s : ctrl->fiberSummaries()) {
        QJsonObject f;
        f["fiberId"] = QString::number(s.id);
        f["name"] = QString::fromStdString(s.name);
        f["controlPointCount"] = s.controlPointCount;
        f["linePointCount"] = s.linePointCount;
        f["lengthVx"] = s.lengthVx;
        f["automaticHvTag"] = QString::fromStdString(s.automaticHvTag);
        f["manualHvTag"] = QString::fromStdString(s.manualHvTag);
        f["automaticCertainty"] = s.automaticCertainty;
        QJsonArray tags;
        for (const auto& tag : s.tags)
            tags.push_back(QString::fromStdString(tag));
        f["tags"] = tags;
        QJsonArray spans;
        for (const auto& sp : s.spans) {
            QJsonObject span;
            span["spanIndex"] = sp.spanIndex;
            span["firstControlIndex"] = sp.firstControlIndex;
            span["secondControlIndex"] = sp.secondControlIndex;
            span["controlPointCount"] = sp.controlPointCount;
            span["linePointCount"] = sp.linePointCount;
            span["lengthVx"] = sp.lengthVx;
            spans.push_back(span);
        }
        f["spans"] = spans;
        fibers.push_back(f);
    }

    QJsonArray knownTags;
    for (const auto& tag : ctrl->knownFiberTags())
        knownTags.push_back(QString::fromStdString(tag));

    QJsonObject result;
    result["fibers"] = fibers;
    result["knownTags"] = knownTags;
    return result;
}


QJsonObject AgentBridgeServer::handleFiberOpen(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();

    const uint64_t fiberId = jsonToFiberId(p.value("fiberId"), "fiberId");
    requireKnownFiber(ctrl, fiberId);

    // At most one selector.
    int selectors = 0;
    if (p.contains("controlPointIndex")) ++selectors;
    if (p.contains("linePointIndex")) ++selectors;
    if (p.contains("span")) ++selectors;
    if (selectors > 1) {
        QJsonObject data;
        data["detail"] = "pass at most one of controlPointIndex, linePointIndex, span";
        throw AgentBridgeError{-32602, "Conflicting fiber.open selectors", data};
    }

    const QString err = captureFiberError(ctrl, [&] {
        if (p.contains("controlPointIndex")) {
            ctrl->openFiberAtControlPoint(
                fiberId, p.value("controlPointIndex").toInt());
        } else if (p.contains("linePointIndex")) {
            ctrl->openFiberAtLinePointIndex(
                fiberId, p.value("linePointIndex").toInt());
        } else if (p.contains("span")) {
            const QJsonValue sv = p.value("span");
            if (sv.toArray().size() != 2) {
                QJsonObject data;
                data["param"] = "span";
                throw AgentBridgeError{
                    -32602, "span must be [firstControlIndex, secondControlIndex]", data};
            }
            const QJsonArray sa = sv.toArray();
            ctrl->openFiberSpan(fiberId,
                                sa.at(0).toInt(),
                                sa.at(1).toInt());
        } else {
            ctrl->openFiber(fiberId);
        }
    });
    if (!err.isEmpty()) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "fiber.open failed", data};
    }

    QJsonObject result;
    result["opened"] = true;
    result["fiberId"] = QString::number(fiberId);
    return result;
}


QJsonObject AgentBridgeServer::handleFiberSetFollow(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();
    const bool enabled = p.value("enabled").toBool();

    LineAnnotationDialog* dialog = ctrl->mostRecentLineAnnotationDialog();
    if (!dialog) {
        QJsonObject data;
        data["kind"] = "fiber_workspace";
        data["detail"] = "no line-annotation workspace is open";
        throw AgentBridgeError{-32007, "No line-annotation workspace open", data};
    }

    dialog->setCutFollowEnabled(enabled);

    QJsonObject result;
    result["enabled"] = dialog->cutFollowEnabled();
    return result;
}


QJsonObject AgentBridgeServer::handleFiberSave(const QJsonValue&)
{
    LineAnnotationController* ctrl = fiberController();
    const int token = beginDeferred(120000, "fiber saves");
    QPointer<AgentBridgeServer> self(this);
    ctrl->saveOpenFibersHeadless(
        [self, token](bool success, const QString& error) {
            if (!self) {
                return;
            }
            if (success) {
                QJsonObject result;
                result["saved"] = true;
                self->completeDeferredResult(token, result);
                return;
            }
            QJsonObject data;
            data["detail"] = error;
            self->completeDeferredError(token, -32005, "fiber.save failed", data);
        });
    throw AgentBridgeDeferred{};
}


QJsonObject AgentBridgeServer::handleFiberDelete(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();

    const QJsonValue idsv = p.value("fiberIds");
    if (idsv.toArray().isEmpty()) {
        QJsonObject data;
        data["param"] = "fiberIds";
        throw AgentBridgeError{-32602, "fiberIds must be a non-empty array of fiber ids", data};
    }

    std::vector<uint64_t> ids;
    for (const QJsonValue& v : idsv.toArray())
        ids.push_back(jsonToFiberId(v, "fiberIds"));

    // All-or-nothing validation: any unknown id fails the call.
    for (uint64_t id : ids)
        requireKnownFiber(ctrl, id);

    const QString err = captureFiberError(ctrl, [&] { ctrl->deleteFibers(ids); });

    // Determine what actually got removed (deleteFibers continues past
    // per-file failures).
    std::unordered_set<uint64_t> remaining;
    for (const auto& s : ctrl->fiberSummaries())
        remaining.insert(s.id);
    QJsonArray deleted;
    bool allDeleted = true;
    for (uint64_t id : ids) {
        if (remaining.count(id)) {
            allDeleted = false;
        } else {
            deleted.push_back(QString::number(id));
        }
    }
    if (!allDeleted) {
        QJsonObject data;
        data["detail"] = err.isEmpty()
            ? QStringLiteral("some fibers could not be deleted") : err;
        data["deleted"] = deleted;
        throw AgentBridgeError{-32005, "fiber.delete partially failed", data};
    }

    QJsonObject result;
    result["deleted"] = deleted;
    return result;
}


QJsonObject AgentBridgeServer::handleFiberSetTag(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();

    const uint64_t fiberId = jsonToFiberId(p.value("fiberId"), "fiberId");
    const QString tag = p.value("tag").toString().trimmed();
    if (tag.isEmpty()) {
        QJsonObject data;
        data["param"] = "tag";
        throw AgentBridgeError{-32602, "tag must be a non-empty string", data};
    }
    const bool enabled = p.value("enabled").toBool();

    requireKnownFiber(ctrl, fiberId);

    const QString err = captureFiberError(
        ctrl, [&] { ctrl->setFiberTag(fiberId, tag, enabled); });
    if (!err.isEmpty()) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "fiber.set_tag failed", data};
    }

    QJsonObject result;
    result["fiberId"] = QString::number(fiberId);
    result["tag"] = tag;
    result["enabled"] = enabled;
    return result;
}


QJsonObject AgentBridgeServer::handleFiberCreateAtlas(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();

    const uint64_t fiberId = jsonToFiberId(p.value("fiberId"), "fiberId");
    requireKnownFiber(ctrl, fiberId);

    // createAtlasFromFiber is synchronous, but its error and rebuild dialogs are
    // unsafe for remote calls. Run the dialog-free split and display the result
    // through displayAtlasFromDirectoryHeadless.
    QString err;
    std::filesystem::path atlasDir;
    if (!ctrl->createAtlasFromFiberHeadless(fiberId, &err, &atlasDir)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Atlas creation failed", data};
    }

    QString displayErr;
    const bool displayed = _window->displayAtlasFromDirectoryHeadless(atlasDir, &displayErr);

    QJsonObject result;
    result["atlasDir"] = QString::fromStdString(atlasDir.string());
    result["displayed"] = displayed;
    if (!displayed && !displayErr.isEmpty())
        result["displayDetail"] = displayErr;
    return result;
}


QJsonObject AgentBridgeServer::handleFiberExport(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();

    const QString pathStr = p.value("path").toString();
    if (pathStr.isEmpty()) {
        QJsonObject data;
        data["param"] = "path";
        throw AgentBridgeError{-32602, "path is required", data};
    }
    const double scale = p.value("scale").toDouble(1.0);

    QString err;
    int exported = 0;
    if (!ctrl->exportFibersToPath(std::filesystem::path(pathStr.toStdString()),
                                  scale, &err, &exported)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "fiber.export failed", data};
    }

    QJsonObject result;
    result["exported"] = exported;
    result["path"] = pathStr;
    return result;
}


QJsonObject AgentBridgeServer::handleFiberImport(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = params.toObject();

    const QString pathStr = p.value("path").toString();
    if (pathStr.isEmpty()) {
        QJsonObject data;
        data["param"] = "path";
        throw AgentBridgeError{-32602, "path is required", data};
    }
    const double scale = p.value("scale").toDouble(1.0);

    const std::filesystem::path importPath(pathStr.toStdString());
    std::error_code ec;
    if (!std::filesystem::exists(importPath, ec)) {
        QJsonObject data;
        data["kind"] = "path";
        data["path"] = pathStr;
        throw AgentBridgeError{-32007, "Import path does not exist", data};
    }

    QString err;
    int imported = 0;
    int skipped = 0;
    if (!ctrl->importFibersFromPath(importPath, scale, &err, &imported, &skipped)) {
        QJsonObject data;
        data["detail"] = err;
        data["skipped"] = skipped;
        throw AgentBridgeError{-32005, "fiber.import failed", data};
    }

    QJsonObject result;
    result["imported"] = imported;
    result["skipped"] = skipped;
    return result;
}
