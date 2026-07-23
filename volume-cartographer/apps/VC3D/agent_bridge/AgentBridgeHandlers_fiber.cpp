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
// Line-annotation / fiber RPCs (SPEC §13)
// ---------------------------------------------------------------------------

namespace {

template <typename Operation>
QString captureFiberError(LineAnnotationController* controller, Operation&& operation)
{
    controller->setErrorDialogsSuppressed(true);
    (void)controller->takeLastSuppressedError();
    try {
        operation();
    } catch (...) {
        controller->setErrorDialogsSuppressed(false);
        throw;
    }
    const QString error = controller->takeLastSuppressedError();
    controller->setErrorDialogsSuppressed(false);
    return error;
}

// Parses a fiber id param: a decimal string (the canonical wire form — uint64
// ids serialize as strings, SPEC §13.2) or a non-negative integer number.
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
        // 2^64 is exactly representable as double, so require strictly below it
        // (FINDING 11a).
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
    const QJsonObject p = paramsObject(params);

    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    auto* chunked = dynamic_cast<CChunkedVolumeViewer*>(viewer);
    if (!chunked) {
        QJsonObject data;
        data["detail"] = "viewer is not a chunked volume viewer";
        throw AgentBridgeError{-32009, "Unsupported viewer for fiber.launch", data};
    }
    if (!chunked->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    // Position conversion per §3.6 conventions (same round-trip rule as
    // canvas.click).
    const QString space = jsonOptionalString(p, "space", QStringLiteral("volume"));
    QPointF scenePos;
    if (space == QLatin1String("scene")) {
        const QJsonValue posv = p.value("position");
        if (!posv.isObject()) {
            QJsonObject data;
            data["param"] = "position";
            throw AgentBridgeError{-32602, "scene-space position must be an object {x, y}", data};
        }
        const QJsonObject po = posv.toObject();
        scenePos = QPointF(jsonRequireFiniteFloat(po.value("x"), "x"),
                           jsonRequireFiniteFloat(po.value("y"), "y"));
    } else if (space == QLatin1String("volume")) {
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
    } else {
        QJsonObject data;
        data["param"] = "space";
        data["value"] = space;
        throw AgentBridgeError{-32602, "space must be \"volume\" or \"scene\"", data};
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

    const bool replaceOwning = jsonOptionalBool(p, "replaceOwning", true);
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
    const QJsonObject p = paramsObject(params);

    const uint64_t fiberId = jsonToFiberId(p.value("fiberId"), "fiberId");
    requireKnownFiber(ctrl, fiberId);

    // At most one selector (SPEC §13.3).
    int selectors = 0;
    if (p.contains("controlPointIndex")) ++selectors;
    if (p.contains("linePointIndex")) ++selectors;
    if (p.contains("span")) ++selectors;
    if (selectors > 1) {
        QJsonObject data;
        data["detail"] = "pass at most one of controlPointIndex, linePointIndex, span";
        throw AgentBridgeError{-32602, "Conflicting fiber.open selectors", data};
    }

    auto readIndex = [&](const char* key) -> int {
        const QJsonValue v = p.value(QLatin1String(key));
        const double d = v.toDouble(std::numeric_limits<double>::quiet_NaN());
        // The int upper-bound guard rejects a finite double (e.g. 1e300) that would
        // overflow the int cast below (FINDING 11a).
        if (!v.isDouble() || !std::isfinite(d) || std::floor(d) != d || d < 0 ||
            d > static_cast<double>(std::numeric_limits<int>::max())) {
            QJsonObject data;
            data["param"] = QString::fromLatin1(key);
            throw AgentBridgeError{-32602,
                QStringLiteral("%1 must be a non-negative integer").arg(QLatin1String(key)),
                data};
        }
        return static_cast<int>(d);
    };

    const QString err = captureFiberError(ctrl, [&] {
        if (p.contains("controlPointIndex")) {
            ctrl->openFiberAtControlPoint(fiberId, readIndex("controlPointIndex"));
        } else if (p.contains("linePointIndex")) {
            ctrl->openFiberAtLinePointIndex(fiberId, readIndex("linePointIndex"));
        } else if (p.contains("span")) {
            const QJsonValue sv = p.value("span");
            if (!sv.isArray() || sv.toArray().size() != 2) {
                QJsonObject data;
                data["param"] = "span";
                throw AgentBridgeError{
                    -32602, "span must be [firstControlIndex, secondControlIndex]", data};
            }
            const QJsonArray sa = sv.toArray();
            auto readSpanIndex = [&](const QJsonValue& v, const char* which) -> int {
                const double d = v.toDouble(std::numeric_limits<double>::quiet_NaN());
                if (!v.isDouble() || !std::isfinite(d) || std::floor(d) != d || d < 0 ||
                    d > static_cast<double>(std::numeric_limits<int>::max())) {
                    QJsonObject data;
                    data["param"] = "span";
                    data["detail"] =
                        QStringLiteral("%1 span index must be a non-negative integer")
                            .arg(QLatin1String(which));
                    throw AgentBridgeError{-32602, "Invalid span", data};
                }
                return static_cast<int>(d);
            };
            ctrl->openFiberSpan(fiberId,
                                readSpanIndex(sa.at(0), "first"),
                                readSpanIndex(sa.at(1), "second"));
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
    const QJsonObject p = paramsObject(params);

    const QJsonValue ev = p.value("enabled");
    if (!ev.isBool()) {
        QJsonObject data;
        data["param"] = "enabled";
        throw AgentBridgeError{-32602, "enabled must be a boolean", data};
    }

    LineAnnotationDialog* dialog = ctrl->mostRecentLineAnnotationDialog();
    if (!dialog) {
        QJsonObject data;
        data["kind"] = "fiber_workspace";
        data["detail"] = "no line-annotation workspace is open";
        throw AgentBridgeError{-32007, "No line-annotation workspace open", data};
    }

    dialog->setCutFollowEnabled(ev.toBool());

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
    const QJsonObject p = paramsObject(params);

    const QJsonValue idsv = p.value("fiberIds");
    if (!idsv.isArray() || idsv.toArray().isEmpty()) {
        QJsonObject data;
        data["param"] = "fiberIds";
        throw AgentBridgeError{-32602, "fiberIds must be a non-empty array of fiber ids", data};
    }

    std::vector<uint64_t> ids;
    for (const QJsonValue& v : idsv.toArray())
        ids.push_back(jsonToFiberId(v, "fiberIds"));

    // All-or-nothing validation (SPEC §13.6): any unknown id fails the call.
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
    const QJsonObject p = paramsObject(params);

    const uint64_t fiberId = jsonToFiberId(p.value("fiberId"), "fiberId");
    const QString tag = jsonRequireString(p, "tag").trimmed();
    if (tag.isEmpty()) {
        QJsonObject data;
        data["param"] = "tag";
        throw AgentBridgeError{-32602, "tag must be a non-empty string", data};
    }
    const QJsonValue ev = p.value("enabled");
    if (!ev.isBool()) {
        QJsonObject data;
        data["param"] = "enabled";
        throw AgentBridgeError{-32602, "enabled must be a boolean", data};
    }

    requireKnownFiber(ctrl, fiberId);

    const QString err = captureFiberError(
        ctrl, [&] { ctrl->setFiberTag(fiberId, tag, ev.toBool()); });
    if (!err.isEmpty()) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "fiber.set_tag failed", data};
    }

    QJsonObject result;
    result["fiberId"] = QString::number(fiberId);
    result["tag"] = tag;
    result["enabled"] = ev.toBool();
    return result;
}


QJsonObject AgentBridgeServer::handleFiberCreateAtlas(const QJsonValue& params)
{
    LineAnnotationController* ctrl = fiberController();
    const QJsonObject p = paramsObject(params);

    const uint64_t fiberId = jsonToFiberId(p.value("fiberId"), "fiberId");
    requireKnownFiber(ctrl, fiberId);

    // Deviates from §13.8's deferred design: createAtlasFromFiber is fully
    // synchronous (atlasCreated fires before return) and its dialogs (showError
    // / rebuild QMessageBox::question) violate §1.3, so we run the headless
    // split and display via the proven displayAtlasFromDirectoryHeadless (§12.1).
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
    const QJsonObject p = paramsObject(params);

    const QString pathStr = jsonRequireString(p, "path");
    if (pathStr.isEmpty()) {
        QJsonObject data;
        data["param"] = "path";
        throw AgentBridgeError{-32602, "path is required", data};
    }
    double scale = 1.0;
    if (p.contains("scale")) {
        const QJsonValue sv = p.value("scale");
        scale = sv.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!sv.isDouble() || !std::isfinite(scale) || scale <= 0.0) {
            QJsonObject data;
            data["param"] = "scale";
            throw AgentBridgeError{-32602, "scale must be a positive finite number", data};
        }
    }

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
    const QJsonObject p = paramsObject(params);

    const QString pathStr = jsonRequireString(p, "path");
    if (pathStr.isEmpty()) {
        QJsonObject data;
        data["param"] = "path";
        throw AgentBridgeError{-32602, "path is required", data};
    }
    double scale = 1.0;
    if (p.contains("scale")) {
        const QJsonValue sv = p.value("scale");
        scale = sv.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!sv.isDouble() || !std::isfinite(scale) || scale <= 0.0) {
            QJsonObject data;
            data["param"] = "scale";
            throw AgentBridgeError{-32602, "scale must be a positive finite number", data};
        }
    }

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
