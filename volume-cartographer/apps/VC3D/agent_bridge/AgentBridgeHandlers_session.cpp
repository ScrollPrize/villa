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
#include <QRegularExpression>
#include <QSettings>
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
#include "VCSettings.hpp"
#include "WrapAnnotationWidget.hpp"
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

QString representationKindToJson(vc3d::opendata::OpenDataRepresentationKind kind)
{
    switch (kind) {
    case vc3d::opendata::OpenDataRepresentationKind::NormalGrids: return QStringLiteral("normal_grids");
    case vc3d::opendata::OpenDataRepresentationKind::Lasagna:     return QStringLiteral("lasagna");
    case vc3d::opendata::OpenDataRepresentationKind::Prediction:  return QStringLiteral("prediction");
    }
    return QStringLiteral("prediction");
}
std::optional<vc3d::opendata::OpenDataRepresentationKind>
representationKindFromJson(const QString& s)
{
    if (s == QLatin1String("normal_grids"))
        return vc3d::opendata::OpenDataRepresentationKind::NormalGrids;
    if (s == QLatin1String("lasagna"))
        return vc3d::opendata::OpenDataRepresentationKind::Lasagna;
    if (s == QLatin1String("prediction"))
        return vc3d::opendata::OpenDataRepresentationKind::Prediction;
    return std::nullopt;
}

} // namespace


QJsonObject AgentBridgeServer::handlePing(const QJsonValue&)
{
    QJsonObject result;
    result["pong"] = true;
    result["pid"] = static_cast<double>(QCoreApplication::applicationPid());
    result["version"] = QCoreApplication::applicationVersion();
    result["protocolVersion"] = kProtocolVersion;
    return result;
}


QJsonObject AgentBridgeServer::handleStateGet(const QJsonValue&)
{
    QJsonObject result;
    CState* state = _window ? _window->_state : nullptr;

    if (!state) {
        // Degenerate case: report an empty snapshot rather than erroring.
        result["vpkg"] = QJsonValue::Null;
        result["volume"] = QJsonValue::Null;
        result["activeSurface"] = QJsonValue::Null;
        result["segmentationGrowthVolumeId"] = QString();
        result["segmentationEditingEnabled"] = false;
        result["manualAddMode"] = false;
        result["manualAddLineMode"] = QJsonValue::Null;
        result["manualAddInterpolation"] = QJsonValue::Null;
        result["correctionsPointMode"] = false;
        result["axisAlignedSlices"] = QJsonValue::Null;
        result["sameWrapAnnotation"] = QJsonValue::Null;
        result["autosave"] = QJsonValue::Null;
        result["viewers"] = QJsonArray();
        result["job"] = QJsonValue::Null;
        result["focusPoi"] = QJsonValue::Null;
        return result;
    }

    // vpkg
    if (state->hasVpkg()) {
        QJsonObject vpkg;
        vpkg["path"] = state->vpkgPath();
        result["vpkg"] = vpkg;
    } else {
        result["vpkg"] = QJsonValue::Null;
    }

    // volume
    if (auto vol = state->currentVolume()) {
        QJsonObject volume;
        volume["id"] = QString::fromStdString(state->currentVolumeId());
        volume["path"] = QString::fromStdString(vol->path().string());
        volume["voxelSize"] = vol->voxelSize();
        // Additive: every volume id in the open package (ADDITIONS_SPEC item 4).
        QJsonArray volIds;
        if (state->hasVpkg() && state->vpkg()) {
            for (const auto& id : state->vpkg()->volumeIDs())
                volIds.append(QString::fromStdString(id));
        }
        volume["volumeIds"] = volIds;
        result["volume"] = volume;
    } else {
        result["volume"] = QJsonValue::Null;
    }

    // active surface
    const std::string activeId = state->activeSurfaceId();
    if (!activeId.empty()) {
        QJsonObject active;
        active["id"] = QString::fromStdString(activeId);
        result["activeSurface"] = active;
    } else {
        result["activeSurface"] = QJsonValue::Null;
    }

    result["segmentationGrowthVolumeId"] =
        QString::fromStdString(state->segmentationGrowthVolumeId());

    result["segmentationEditingEnabled"] =
        (_window->_segmentationWidget != nullptr)
            ? _window->_segmentationWidget->isEditingEnabled()
            : false;

    // Manual-add (hole-fill) + corrections point-authoring state (SPEC §9.2-9.7;
    // reported here per Stage 2). Line/interpolation modes come from the panel
    // config (they persist whether or not manual-add is currently active).
    if (SegmentationModule* mod = _window->_segmentationModule.get()) {
        result["manualAddMode"] = mod->manualAddMode();
        result["correctionsPointMode"] = mod->correctionPointMode();
        // Explicit save/flush bookkeeping (SPEC §3.11c, §9.8).
        const SegmentationModule::AutosaveStatus save = mod->autosaveStatus();
        QJsonObject autosave;
        autosave["pending"] = save.pending;
        autosave["saveInProgress"] = save.saveInProgress;
        autosave["dirtyAfterSave"] = save.dirtyAfterSave;
        result["autosave"] = autosave;
    } else {
        result["manualAddMode"] = false;
        result["correctionsPointMode"] = false;
        result["autosave"] = QJsonValue::Null;
    }

    // Axis-aligned slice mode (SPEC §3.9c): when enabled, "seg xz"/"seg yz" are the
    // rotatable canonical planes. viewer.rotate requires this on (else -32002), and
    // viewer.set_axis_aligned_slices toggles it.
    if (AxisAlignedSliceController* slices = _window->_axisAlignedSliceController.get()) {
        QJsonObject axis;
        axis["enabled"] = slices->isEnabled();
        axis["segXZRotationDeg"] = slices->currentRotationDegrees("seg xz");
        axis["segYZRotationDeg"] = slices->currentRotationDegrees("seg yz");
        result["axisAlignedSlices"] = axis;
    } else {
        result["axisAlignedSlices"] = QJsonValue::Null;
    }
    if (SegmentationWidget* widget = _window->_segmentationWidget) {
        const ManualAddTool::Config cfg = widget->manualAddConfig();
        result["manualAddLineMode"] = linePreviewModeToString(cfg.linePreviewMode);
        result["manualAddInterpolation"] = interpolationModeToString(cfg.interpolationMode);
    } else {
        result["manualAddLineMode"] = QJsonValue::Null;
        result["manualAddInterpolation"] = QJsonValue::Null;
    }

    // Same-winding wrap annotation (SPEC §3.9d). `enabled` is the widget's
    // checkbox state; `hasPreview` is true when any chunked viewer currently
    // holds an uncommitted preview (seeded by shift-click, committed by shift+E).
    WrapAnnotationWidget* wrapWidget = _window->_wrapAnnotationWidget;
    ViewerManager* viewerMgr = _window->_viewerManager.get();
    if (wrapWidget && viewerMgr) {
        bool hasPreview = false;
        viewerMgr->forEachBaseViewer([&hasPreview](VolumeViewerBase* baseViewer) {
            if (hasPreview || !baseViewer)
                return;
            if (auto* viewer = qobject_cast<CChunkedVolumeViewer*>(baseViewer->asQObject()))
                hasPreview = viewer->hasSameWrapAnnotationPreview();
        });
        QJsonObject wrap;
        wrap["enabled"] = wrapWidget->sameWrapAnnotationEnabled();
        wrap["hasPreview"] = hasPreview;
        result["sameWrapAnnotation"] = wrap;
    } else {
        result["sameWrapAnnotation"] = QJsonValue::Null;
    }

    // viewers
    QJsonArray viewers;
    for (const ViewerEntry& e : _viewers) {
        QJsonObject v;
        v["viewerId"] = e.id;
        v["surfName"] = QString::fromStdString(e.viewer->surfName());
        v["title"] = viewerTitle(e.viewer);
        v["kind"] = (dynamic_cast<CChunkedVolumeViewer*>(e.viewer) != nullptr)
                        ? QStringLiteral("chunked") : QStringLiteral("other");
        v["scale"] = e.viewer->getCurrentScale();
        viewers.push_back(v);
    }
    result["viewers"] = viewers;

    // Active job(s). "job" keeps its v1 meaning (most recently started active
    // job, else null); "jobs" lists every currently active job (§8.3).
    QJsonObject mostRecentActive;
    int bestNum = -1;
    QJsonArray jobs;
    for (const auto& rec : _activeJobs) {
        if (rec.state != QLatin1String("running"))
            continue;
        jobs.push_back(jobStatusJson(rec));
        if (rec.num > bestNum) {
            bestNum = rec.num;
            mostRecentActive = QJsonObject();
            mostRecentActive["jobId"] = rec.id;
            mostRecentActive["kind"] = rec.kind;
            mostRecentActive["label"] = rec.label;
            mostRecentActive["source"] = rec.source;
            mostRecentActive["running"] = true;
        }
    }
    result["job"] = (bestNum >= 0) ? QJsonValue(mostRecentActive) : QJsonValue::Null;
    result["jobs"] = jobs;

    // focus POI
    if (POI* focus = state->poi("focus")) {
        QJsonObject poi;
        poi["position"] = vec3ToJson(focus->p);
        poi["normal"] = vec3ToJson(focus->n);
        poi["surfaceId"] = QString::fromStdString(focus->surfaceId);
        result["focusPoi"] = poi;
    } else {
        result["focusPoi"] = QJsonValue::Null;
    }

    return result;
}


QJsonObject AgentBridgeServer::handleSegmentsList(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const bool onlyLoaded = jsonOptionalBool(p, "onlyLoaded", false);

    // Loaded surface names live in CState; the on-disk segment ids come from the
    // package. A segment is "loaded" when its id appears among CState surfaces.
    const std::vector<std::string> loadedNames = state->surfaceNames();
    const std::unordered_set<std::string> loadedSet(loadedNames.begin(), loadedNames.end());
    const std::string activeId = state->activeSurfaceId();

    QJsonArray segments;
    for (const std::string& id : vpkg->segmentationIDs()) {
        const bool loaded = loadedSet.count(id) > 0;
        if (onlyLoaded && !loaded)
            continue;

        QJsonObject seg;
        seg["id"] = QString::fromStdString(id);

        QString path;
        try {
            if (auto s = vpkg->segmentation(id))
                path = QString::fromStdString(s->path().string());
        } catch (...) {
            // Metadata resolution may fail for a partially written segment; the
            // id is still reportable without a path.
        }
        seg["path"] = path;
        seg["loaded"] = loaded;
        seg["active"] = (id == activeId);
        segments.push_back(seg);
    }

    QJsonObject result;
    result["segments"] = segments;
    return result;
}


QJsonObject AgentBridgeServer::handleSegmentsActivate(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const QString segmentIdQ = jsonRequireString(p, "segmentId");
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }
    const std::string segmentId = segmentIdQ.toStdString();

    SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "surface panel is not available";
        throw AgentBridgeError{-32010, "Surface panel unavailable", data};
    }

    // Resolve the path for the result's segment entry (SPEC §3.3 shape). Prefer the
    // loaded CState surface (covers multi-folder display ids), fall back to the vpkg.
    auto resolveSegPath = [&]() -> QString {
        if (auto surf = state->surface(segmentId))
            return QString::fromStdString(surf->path.string());
        try {
            if (auto s = vpkg->segmentation(segmentId))
                return QString::fromStdString(s->path().string());
        } catch (...) {
        }
        return QString();
    };

    const std::string prevActive = state->activeSurfaceId();
    const QJsonValue previousSegmentId = prevActive.empty()
        ? QJsonValue(QJsonValue::Null)
        : QJsonValue(QString::fromStdString(prevActive));

    // Activating the already-active id is a no-op success (SPEC §17.3): no re-emit,
    // no side effects, mirroring the tree where re-clicking the current row is inert.
    const bool alreadyActive = !prevActive.empty() && prevActive == segmentId;

    if (!alreadyActive) {
        QString err;
        if (!panel->activateSurfaceById(segmentId, &err)) {
            // Classify via the distinct sentences activateSurfaceById produces
            // (SPEC §17.2 contract).
            if (err.contains(QLatin1String("locked"))) {
                QJsonObject data;
                data["source"] = "growth";
                data["detail"] = err;
                throw AgentBridgeError{-32004, "Surface selection is locked", data};
            }
            if (err.startsWith(QLatin1String("unknown segment"))) {
                QJsonObject data;
                data["kind"] = "segment";
                data["id"] = segmentIdQ;
                data["detail"] = err;
                throw AgentBridgeError{-32007, "Segment not found", data};
            }
            // Placeholder / load failure / could-not-select.
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32005, "Segment could not be activated", data};
        }

        // Post-verify: onSurfaceActivated clears the active surface when the surface
        // throws while loading (CWindow.cpp:9614-9625) -> map to -32005 (SPEC §17.3).
        if (state->activeSurfaceId() != segmentId) {
            QJsonObject data;
            data["detail"] = "active surface was cleared during activation "
                             "(surface failed to load)";
            throw AgentBridgeError{-32005, "Segment could not be activated", data};
        }
    }

    QJsonObject segment;
    segment["id"] = segmentIdQ;
    segment["path"] = resolveSegPath();
    segment["loaded"] = true;
    segment["active"] = true;

    QJsonObject result;
    result["activated"] = true;
    result["segment"] = segment;
    result["previousSegmentId"] = previousSegmentId;
    result["alreadyActive"] = alreadyActive;
    return result;
}


QJsonObject AgentBridgeServer::handleSegmentsFetch(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const QString segmentIdQ = jsonRequireString(p, "segmentId");
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }
    const std::string segmentId = segmentIdQ.toStdString();

    SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "surface panel is not available";
        throw AgentBridgeError{-32010, "Surface panel unavailable", data};
    }

    // Resolve the segment path (prefer the loaded CState surface for multi-folder
    // display ids, fall back to the vpkg) so the result carries a stable path.
    auto resolveSegPath = [&]() -> QString {
        if (auto surf = state->surface(segmentId))
            return QString::fromStdString(surf->path.string());
        try {
            if (auto s = vpkg->segmentation(segmentId))
                return QString::fromStdString(s->path().string());
        } catch (...) {
        }
        return QString();
    };
    const QString segPath = resolveSegPath();

    auto segmentEntry = [&](bool placeholder) {
        QJsonObject seg;
        seg["id"] = segmentIdQ;
        seg["path"] = segPath;
        seg["placeholder"] = placeholder;
        return seg;
    };

    // A concurrent Open Data operation (catalog open or another segment fetch)
    // shares the same download subsystem: reject up front (SPEC §1.3 / §18.4).
    requireSourceIdle(QStringLiteral("catalog"));

    // Kick off (or short-circuit) the materialize. fetchOpenDataSegmentAsync
    // only calls onDone for the `Started` outcome; the others are terminal here.
    // beginJob runs synchronously below before we return to the event loop, so
    // the "catalog" job is registered before onDone (which finishes it) can fire.
    const auto outcome = panel->fetchOpenDataSegmentAsync(
        segmentId,
        [this, segmentId](bool success, const QString& message) {
            finishJob(QStringLiteral("catalog"), success,
                      success
                          ? QStringLiteral("segment %1 fetched")
                                .arg(QString::fromStdString(segmentId))
                          : message,
                      QString());
        });

    switch (outcome) {
    case SurfacePanelController::OpenDataFetchOutcome::NotFound: {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentIdQ;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }
    case SurfacePanelController::OpenDataFetchOutcome::Busy: {
        QJsonObject data;
        data["source"] = "catalog";
        data["detail"] = "another open-data segment fetch is already in progress";
        throw AgentBridgeError{-32004, "A segment fetch is already in progress", data};
    }
    case SurfacePanelController::OpenDataFetchOutcome::AlreadyMaterialized: {
        QJsonObject result;
        result["fetched"] = true;
        result["alreadyMaterialized"] = true;
        result["segment"] = segmentEntry(false);
        return result;
    }
    case SurfacePanelController::OpenDataFetchOutcome::Started:
        break;
    }

    const QString jobId = beginJob(QStringLiteral("catalog"),
                                   QStringLiteral("segments.fetch"),
                                   QStringLiteral("Fetching segment %1").arg(segmentIdQ),
                                   /*broadcastStart=*/true);

    QJsonObject result;
    result["jobId"] = jobId;
    result["source"] = "catalog";
    result["fetched"] = false;
    result["alreadyMaterialized"] = false;
    result["segment"] = segmentEntry(true);
    return result;
}


QJsonObject AgentBridgeServer::handleScreenshotCapture(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString target = jsonOptionalString(p, "target", QStringLiteral("window"));

    QWidget* widget = nullptr;
    if (target.isEmpty() || target == QLatin1String("window")) {
        widget = _window;
    } else {
        VolumeViewerBase* viewer = resolveViewer(QJsonValue(target));
        widget = dynamic_cast<QWidget*>(viewer);
        if (!widget) {
            QJsonObject data;
            data["detail"] = "resolved viewer is not a widget";
            throw AgentBridgeError{-32002, "Invalid viewer target", data};
        }
    }

    if (!widget)
        throw AgentBridgeError{-32002, "No capture target available", {}};

    if (!widget->isVisible()) {
        QJsonObject data;
        data["detail"] = "target widget is not currently visible (e.g. it's on a "
                          "non-frontmost tab or a hidden pane) -- a grab() of a "
                          "hidden widget returns a meaningless, often near-zero-size "
                          "image; switch to it (e.g. via workspace.switch) before "
                          "capturing";
        throw AgentBridgeError{-32009, "Capture target not visible", data};
    }

    QPixmap pixmap = widget->grab();

    static constexpr int kMinCaptureDim = 8;
    if (pixmap.width() < kMinCaptureDim || pixmap.height() < kMinCaptureDim) {
        QJsonObject data;
        data["detail"] = QStringLiteral(
                              "captured pixmap is degenerate (%1x%2) -- the target "
                              "widget has not been laid out to a meaningful size")
                              .arg(pixmap.width())
                              .arg(pixmap.height());
        throw AgentBridgeError{-32009, "Capture target has degenerate size", data};
    }

    if (p.contains("maxDim")) {
        const int maxDim = jsonOptionalInt(p, "maxDim", 0);
        if (maxDim > 0) {
            const int longest = std::max(pixmap.width(), pixmap.height());
            if (longest > maxDim)
                pixmap = pixmap.scaled(maxDim, maxDim, Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation);
        }
    }

    QImage image = pixmap.toImage();

    QJsonObject result;
    result["width"] = image.width();
    result["height"] = image.height();
    result["format"] = "png";

    const QString filePath = jsonOptionalString(p, "filePath");
    if (!filePath.isEmpty()) {
        if (!image.save(filePath, "PNG")) {
            QJsonObject data;
            data["detail"] = QStringLiteral("failed to write PNG to %1").arg(filePath);
            throw AgentBridgeError{-32005, "Screenshot write failed", data};
        }
        result["filePath"] = filePath;
        result["base64"] = QJsonValue::Null;
    } else {
        QByteArray bytes;
        QBuffer buffer(&bytes);
        buffer.open(QIODevice::WriteOnly);
        if (!image.save(&buffer, "PNG")) {
            QJsonObject data;
            data["detail"] = "PNG encode failed";
            throw AgentBridgeError{-32005, "Screenshot encode failed", data};
        }
        result["filePath"] = QJsonValue::Null;
        result["base64"] = QString::fromLatin1(bytes.toBase64());
    }

    return result;
}


QJsonObject AgentBridgeServer::handleCursorVolumePoint(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);

    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    auto* chunked = dynamic_cast<CChunkedVolumeViewer*>(viewer);
    if (!chunked) {
        QJsonObject data;
        data["detail"] = "viewer is not a chunked volume viewer";
        throw AgentBridgeError{-32009, "Unsupported viewer for canvas operation", data};
    }

    if (!chunked->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    QPointF scenePos;
    if (p.contains("scene") && !p.value("scene").isNull()) {
        if (!p.value("scene").isObject())
            throwParamError("scene", QStringLiteral("must be an object or null"));
        const QJsonObject scene = p.value("scene").toObject();
        scenePos = QPointF(jsonRequireFiniteFloat(scene.value("x"), "x"),
                           jsonRequireFiniteFloat(scene.value("y"), "y"));
    } else {
        scenePos = chunked->lastScenePosition();
    }

    const auto sample = chunked->sampleSceneVolume(scenePos);
    if (!sample) {
        QJsonObject point;
        point["x"] = scenePos.x();
        point["y"] = scenePos.y();
        QJsonObject data;
        data["point"] = point;
        data["detail"] = "scene position does not hit the surface/volume";
        throw AgentBridgeError{-32003, "Invalid coordinates", data};
    }

    QJsonObject scene;
    scene["x"] = scenePos.x();
    scene["y"] = scenePos.y();

    QJsonObject result;
    result["volumePoint"] = vec3ToJson(sample->position);
    result["normal"] = vec3ToJson(sample->normal);
    result["scene"] = scene;
    result["surfName"] = QString::fromStdString(chunked->surfName());
    return result;
}


// ---------------------------------------------------------------------------
// Project/catalog opening (SPEC §3.15-3.16)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleVolumeOpen(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString path = jsonRequireString(p, "path");
    if (path.isEmpty()) {
        QJsonObject data;
        data["detail"] = "path is required";
        throw AgentBridgeError{-32005, "No volume package path", data};
    }

    if (!_window) {
        QJsonObject data;
        data["detail"] = "main window is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    const QString volumeId = jsonOptionalString(p, "volumeId");
    QString errorMessage;
    CWindow::VolumeOpenError openError = CWindow::VolumeOpenError::None;
    if (!_window->openVolumePackage(path, false, &errorMessage, volumeId, &openError)) {
        QJsonObject data;
        data["detail"] = errorMessage.isEmpty()
            ? QStringLiteral("failed to open volume package at %1").arg(path)
            : errorMessage;
        if (openError == CWindow::VolumeOpenError::VolumeNotFound) {
            data["kind"] = "volume";
            data["id"] = volumeId;
            throw AgentBridgeError{
                -32007, QStringLiteral("Unknown volume id: %1").arg(volumeId), data};
        }
        throw AgentBridgeError{-32005, "Volume package load failed", data};
    }

    CState* state = _window->_state;
    const auto ids = state->vpkg()->volumeIDs();
    QJsonObject result;
    result["opened"] = true;
    result["vpkgPath"] = state->vpkgPath();
    result["volumeId"] = QString::fromStdString(state->currentVolumeId());
    QJsonArray idArr;
    for (const auto& id : ids)
        idArr.append(QString::fromStdString(id));
    result["volumeIds"] = idArr;
    return result;
}


QJsonObject AgentBridgeServer::handleVolumeList(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    QJsonArray volumeIds;
    for (const auto& id : vpkg->volumeIDs())
        volumeIds.append(QString::fromStdString(id));

    const std::string current = state->currentVolumeId();

    QJsonObject result;
    result["volumeIds"] = volumeIds;
    result["currentVolumeId"] = current.empty() ? QJsonValue(QJsonValue::Null)
                                                : QJsonValue(QString::fromStdString(current));
    // Per-volume {id, path, voxelSize} objects are intentionally omitted: reading
    // voxelSize would force-load every (possibly remote) volume, which is not
    // "cheap" per ADDITIONS_SPEC item 4. The id list plus state.get's per-volume
    // block cover the affordable data.
    return result;
}


QJsonObject AgentBridgeServer::handleSegmentsDelete(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const QString segmentIdQ = jsonRequireString(p, "segmentId");
    const bool confirm = jsonOptionalBool(p, "confirm", false);
    if (!confirm) {
        QJsonObject data;
        data["param"] = "confirm";
        data["reason"] = "destructive; pass confirm=true";
        throw AgentBridgeError{-32602,
            "confirm must be true (on-disk deletion is irreversible)", data};
    }

    // Refuse while a segmentation edit is in progress (would race the delete
    // against unsaved edits / active-surface teardown).
    if (_window->_segmentationWidget && _window->_segmentationWidget->isEditingEnabled()) {
        QJsonObject data;
        data["detail"] = "cannot delete while editing";
        throw AgentBridgeError{-32004, "cannot delete while editing", data};
    }

    // Unknown id -> -32007. Match against the package's known segment ids.
    const std::string segmentId = segmentIdQ.toStdString();
    const std::vector<std::string> ids = vpkg->segmentationIDs();
    if (std::find(ids.begin(), ids.end(), segmentId) == ids.end()) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentIdQ;
        throw AgentBridgeError{-32007, QStringLiteral("Unknown segment id: %1").arg(segmentIdQ), data};
    }

    SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "surface panel is not available";
        throw AgentBridgeError{-32010, "Surface panel unavailable", data};
    }

    // Dialog-free core clears the active slot first, deletes on disk, and refreshes.
    QString err;
    if (!panel->deleteSegmentsHeadless(QStringList{segmentIdQ}, &err)) {
        QJsonObject data;
        data["detail"] = err.isEmpty() ? QStringLiteral("delete failed") : err;
        throw AgentBridgeError{-32010, QStringLiteral("Failed to delete segment: %1").arg(err), data};
    }

    QJsonArray deleted;
    deleted.append(segmentIdQ);
    QJsonObject result;
    result["deleted"] = deleted;
    return result;
}


QJsonObject AgentBridgeServer::handleSegmentsRename(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const QString segmentIdQ = jsonRequireString(p, "segmentId");
    const QString newName = jsonRequireString(p, "newName");

    // Validate the new name up front so a bad name is -32602 (not -32010).
    static const QRegularExpression validNameRegex(QStringLiteral("^[a-zA-Z0-9_-]+$"));
    if (!validNameRegex.match(newName).hasMatch()) {
        QJsonObject data;
        data["param"] = "newName";
        throw AgentBridgeError{-32602,
            "newName must match ^[a-zA-Z0-9_-]+$", data};
    }

    // Refuse while editing (mirrors segments.delete and the interactive guard).
    if (_window->_segmentationWidget && _window->_segmentationWidget->isEditingEnabled()) {
        QJsonObject data;
        data["detail"] = "cannot rename while editing";
        throw AgentBridgeError{-32004, "cannot rename while editing", data};
    }

    // Unknown id -> -32007.
    const std::string segmentId = segmentIdQ.toStdString();
    const std::vector<std::string> ids = vpkg->segmentationIDs();
    if (std::find(ids.begin(), ids.end(), segmentId) == ids.end()) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentIdQ;
        throw AgentBridgeError{-32007, QStringLiteral("Unknown segment id: %1").arg(segmentIdQ), data};
    }

    SegmentationCommandHandler* sch = _window ? _window->_segmentationCommandHandler.get() : nullptr;
    if (!sch) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32010, "Segmentation command handler unavailable", data};
    }

    QString err;
    if (!sch->renameSurfaceHeadless(segmentIdQ, newName, &err)) {
        // Map the core's classifiable sentences to JSON-RPC codes.
        if (err == QLatin1String("name exists")) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = newName;
            data["reason"] = "target name already exists";
            throw AgentBridgeError{-32010, QStringLiteral("A segment named %1 already exists").arg(newName), data};
        }
        if (err == QLatin1String("invalid name")) {
            QJsonObject data;
            data["param"] = "newName";
            throw AgentBridgeError{-32602, "newName must match ^[a-zA-Z0-9_-]+$", data};
        }
        if (err == QLatin1String("name unchanged")) {
            QJsonObject data;
            data["param"] = "newName";
            throw AgentBridgeError{-32602, "newName is unchanged", data};
        }
        if (err == QLatin1String("editing in progress")) {
            QJsonObject data;
            data["detail"] = "cannot rename while editing";
            throw AgentBridgeError{-32004, "cannot rename while editing", data};
        }
        if (err == QLatin1String("segment not found")) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentIdQ;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown segment id: %1").arg(segmentIdQ), data};
        }
        QJsonObject data;
        data["detail"] = err.isEmpty() ? QStringLiteral("rename failed") : err;
        throw AgentBridgeError{-32010, QStringLiteral("Failed to rename segment: %1").arg(err), data};
    }

    QJsonObject result;
    result["oldId"] = segmentIdQ;
    result["newId"] = newName;
    return result;
}


QJsonObject AgentBridgeServer::viewerRenderSettingsJson() const
{
    ViewerManager* mgr = _window ? _window->_viewerManager.get() : nullptr;
    if (!mgr) {
        QJsonObject data;
        data["detail"] = "viewer manager is not available";
        throw AgentBridgeError{-32010, "Viewer manager unavailable", data};
    }

    // First live chunked viewer carries the per-viewer (uniformly-driven) toggles;
    // fall back to QSettings-backed defaults when no viewer exists yet.
    CChunkedVolumeViewer* firstChunked = nullptr;
    mgr->forEachBaseViewer([&firstChunked](VolumeViewerBase* v) {
        if (firstChunked || !v)
            return;
        if (auto* c = qobject_cast<CChunkedVolumeViewer*>(v->asQObject()))
            firstChunked = c;
    });

    QJsonObject o;
    o["intersectionOpacity"] = mgr->intersectionOpacity();
    o["intersectionThickness"] = mgr->intersectionThickness();
    o["overlayOpacity"] = mgr->overlayOpacity();
    o["intersectionMaxSurfaces"] = mgr->intersectionMaxSurfaces();
    // ViewerManager-backed, so meaningful with zero viewers instantiated too.
    QJsonObject volumeWindow;
    volumeWindow["low"] = mgr->volumeWindowLow();
    volumeWindow["high"] = mgr->volumeWindowHigh();
    o["volumeWindow"] = volumeWindow;
    o["samplingStride"] = mgr->surfacePatchSamplingStride();
    o["zScrollSensitivity"] = mgr->zScrollSensitivity();
    o["segmentationCursorMirroring"] = _window->segmentationCursorMirroringEnabled();

    if (firstChunked) {
        o["planeIntersectionLinesVisible"] = firstChunked->isPlaneIntersectionLinesVisible();
        o["showSurfaceNormals"] = firstChunked->isShowSurfaceNormals();
        o["showDirectionHints"] = firstChunked->isShowDirectionHints();
        o["surfaceOverlayEnabled"] = firstChunked->surfaceOverlayEnabled();
        o["normalArrowLengthScale"] = firstChunked->normalArrowLengthScale();
        o["normalMaxArrows"] = firstChunked->normalMaxArrows();
    } else {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        o["planeIntersectionLinesVisible"] = settings.value(
            vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES,
            vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES_DEFAULT).toBool();
        o["showSurfaceNormals"] = settings.value(
            vc3d::settings::viewer::SHOW_SURFACE_NORMALS,
            vc3d::settings::viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        o["showDirectionHints"] = settings.value(
            vc3d::settings::viewer::SHOW_DIRECTION_HINTS,
            vc3d::settings::viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        o["surfaceOverlayEnabled"] = false;
        // Stored as a percentage int (100 = 1.0x) by ViewerNormalVisualizationPanel;
        // convert to the same float scale setNormalArrowLengthScale() takes.
        o["normalArrowLengthScale"] = settings.value(
            vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE,
            vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE_DEFAULT).toInt() / 100.0;
        o["normalMaxArrows"] = settings.value(
            vc3d::settings::viewer::NORMAL_MAX_ARROWS,
            vc3d::settings::viewer::NORMAL_MAX_ARROWS_DEFAULT).toInt();
    }

    // Highlighted surface ids are sourced from the surface panel (the source of
    // truth behind the context-menu checkmarks), so get/set stay consistent and
    // the value is reported even when no viewer is instantiated. Fall back to the
    // first live viewer only if the panel is unavailable.
    QJsonArray highlightIds;
    if (SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr) {
        for (const std::string& id : panel->highlightedSurfaceIds())
            highlightIds.append(QString::fromStdString(id));
    } else if (firstChunked) {
        for (const std::string& id : firstChunked->highlightedSurfaceIds())
            highlightIds.append(QString::fromStdString(id));
    }
    o["highlightedSurfaceIds"] = highlightIds;
    return o;
}


QJsonObject AgentBridgeServer::handleViewerGetRenderSettings(const QJsonValue&)
{
    return viewerRenderSettingsJson();
}


QJsonObject AgentBridgeServer::handleViewerSetRenderSettings(const QJsonValue& params)
{
    ViewerManager* mgr = _window ? _window->_viewerManager.get() : nullptr;
    if (!mgr) {
        QJsonObject data;
        data["detail"] = "viewer manager is not available";
        throw AgentBridgeError{-32010, "Viewer manager unavailable", data};
    }

    const QJsonObject p = paramsObject(params);

    // --- Phase 1: parse/validate/clamp into locals (opacities to 0..1). Doing
    // this before any setter or QSettings write means a malformed field rejects
    // the whole request rather than leaving render settings half-applied. ---
    const bool hasIntersectionOpacity = p.contains("intersectionOpacity");
    float intersectionOpacity = 0.0f;
    if (hasIntersectionOpacity) {
        const double v = jsonRequireFinite(p.value("intersectionOpacity"), "intersectionOpacity");
        intersectionOpacity = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }

    const bool hasIntersectionThickness = p.contains("intersectionThickness");
    float intersectionThickness = 0.0f;
    if (hasIntersectionThickness) {
        const double v = jsonRequireFinite(p.value("intersectionThickness"), "intersectionThickness");
        intersectionThickness = static_cast<float>(std::max(0.0, v));
    }

    const bool hasOverlayOpacity = p.contains("overlayOpacity");
    float overlayOpacity = 0.0f;
    if (hasOverlayOpacity) {
        const double v = jsonRequireFinite(p.value("overlayOpacity"), "overlayOpacity");
        overlayOpacity = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }

    const bool hasIntersectionMaxSurfaces = p.contains("intersectionMaxSurfaces");
    int intersectionMaxSurfaces = 0;
    if (hasIntersectionMaxSurfaces) {
        const int v = jsonRequireInt(p.value("intersectionMaxSurfaces"), "intersectionMaxSurfaces");
        intersectionMaxSurfaces = std::max(0, v);
    }

    const bool hasHighlightedSurfaceIds = p.contains("highlightedSurfaceIds");
    std::vector<std::string> highlightedSurfaceIds;
    if (hasHighlightedSurfaceIds) {
        const QJsonValue hv = p.value("highlightedSurfaceIds");
        if (!hv.isArray())
            throwParamError("highlightedSurfaceIds", QStringLiteral("must be an array of strings"));
        for (const QJsonValue& e : hv.toArray()) {
            if (!e.isString())
                throwParamError("highlightedSurfaceIds", QStringLiteral("must be an array of strings"));
            highlightedSurfaceIds.push_back(e.toString().toStdString());
        }
    }

    const bool hasVolumeWindow = p.contains("volumeWindow");
    float volumeWindowLow = 0.0f;
    float volumeWindowHigh = 0.0f;
    if (hasVolumeWindow) {
        const QJsonValue wv = p.value("volumeWindow");
        if (!wv.isObject())
            throwParamError("volumeWindow", QStringLiteral("must be an object {low, high}"));
        const QJsonObject wo = wv.toObject();
        if (!wo.contains("low") || !wo.contains("high"))
            throwParamError("volumeWindow", QStringLiteral("requires low and high"));
        volumeWindowLow = static_cast<float>(jsonRequireFinite(wo.value("low"), "volumeWindow.low"));
        volumeWindowHigh = static_cast<float>(jsonRequireFinite(wo.value("high"), "volumeWindow.high"));
    }

    const bool hasSamplingStride = p.contains("samplingStride");
    const int samplingStride = hasSamplingStride
        ? jsonRequireInt(p.value("samplingStride"), "samplingStride") : 0;

    const bool hasZScrollSensitivity = p.contains("zScrollSensitivity");
    const double zScrollSensitivity = hasZScrollSensitivity
        ? jsonRequireFinite(p.value("zScrollSensitivity"), "zScrollSensitivity") : 0.0;

    const bool hasSegmentationCursorMirroring = p.contains("segmentationCursorMirroring");
    const bool segmentationCursorMirroring = hasSegmentationCursorMirroring
        ? jsonRequireBool(p.value("segmentationCursorMirroring"), "segmentationCursorMirroring") : false;

    const bool hasPlaneIntersectionLinesVisible = p.contains("planeIntersectionLinesVisible");
    const bool planeIntersectionLinesVisible = hasPlaneIntersectionLinesVisible
        ? jsonRequireBool(p.value("planeIntersectionLinesVisible"), "planeIntersectionLinesVisible") : false;

    const bool hasShowSurfaceNormals = p.contains("showSurfaceNormals");
    const bool showSurfaceNormals = hasShowSurfaceNormals
        ? jsonRequireBool(p.value("showSurfaceNormals"), "showSurfaceNormals") : false;

    const bool hasShowDirectionHints = p.contains("showDirectionHints");
    const bool showDirectionHints = hasShowDirectionHints
        ? jsonRequireBool(p.value("showDirectionHints"), "showDirectionHints") : false;

    const bool hasSurfaceOverlayEnabled = p.contains("surfaceOverlayEnabled");
    const bool surfaceOverlayEnabled = hasSurfaceOverlayEnabled
        ? jsonRequireBool(p.value("surfaceOverlayEnabled"), "surfaceOverlayEnabled") : false;

    const bool hasNormalArrowLengthScale = p.contains("normalArrowLengthScale");
    float normalArrowLengthScale = 0.0f;
    if (hasNormalArrowLengthScale) {
        const double v = jsonRequireFinite(p.value("normalArrowLengthScale"), "normalArrowLengthScale");
        // Clamp to the GUI slider's range (sliderNormalArrowLength: 10-200%) rather
        // than passing a negative/huge scale straight to the renderer.
        normalArrowLengthScale = static_cast<float>(std::clamp(v, 0.1, 2.0));
    }

    const bool hasNormalMaxArrows = p.contains("normalMaxArrows");
    int normalMaxArrows = 0;
    if (hasNormalMaxArrows) {
        const int v = jsonRequireInt(p.value("normalMaxArrows"), "normalMaxArrows");
        // Clamp to the GUI slider's range (sliderNormalMaxArrows: 4-100).
        normalMaxArrows = std::clamp(v, 4, 100);
    }

    // --- Phase 2: apply. Fully validated above, so the setters, QSettings
    // writes, and broadcasts run as a group. Global controls go via ViewerManager
    // (broadcast + QSettings-persisted). ---
    if (hasIntersectionOpacity)
        mgr->setIntersectionOpacity(intersectionOpacity);
    if (hasIntersectionThickness)
        mgr->setIntersectionThickness(intersectionThickness);
    if (hasOverlayOpacity)
        mgr->setOverlayOpacity(overlayOpacity);
    if (hasIntersectionMaxSurfaces)
        mgr->setIntersectionMaxSurfaces(intersectionMaxSurfaces);
    if (hasHighlightedSurfaceIds) {
        // Route through the surface panel so its _highlightedSurfaceIds (the source
        // of truth behind the context-menu checkmarks) stays in sync; otherwise the
        // next GUI highlight toggle would rebuild from the stale panel set and
        // clobber these. Fall back to ViewerManager if the panel is unavailable.
        if (SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr)
            panel->setHighlightedSurfaceIds(highlightedSurfaceIds);
        else
            mgr->setHighlightedSurfaceIds(highlightedSurfaceIds);
    }
    if (hasVolumeWindow)
        mgr->setVolumeWindow(volumeWindowLow, volumeWindowHigh);
    if (hasSamplingStride)
        mgr->setSurfacePatchSamplingStride(samplingStride);
    if (hasZScrollSensitivity)
        mgr->setZScrollSensitivity(zScrollSensitivity);
    if (hasSegmentationCursorMirroring)
        _window->setSegmentationCursorMirroring(segmentationCursorMirroring);

    // Per-viewer toggles are driven uniformly across every base viewer.
    auto broadcast = [mgr](const std::function<void(VolumeViewerBase*)>& fn) {
        mgr->forEachBaseViewer([&fn](VolumeViewerBase* v) { if (v) fn(v); });
    };
    if (hasPlaneIntersectionLinesVisible)
        broadcast([planeIntersectionLinesVisible](VolumeViewerBase* v) {
            v->setPlaneIntersectionLinesVisible(planeIntersectionLinesVisible);
        });
    if (hasShowSurfaceNormals)
        broadcast([showSurfaceNormals](VolumeViewerBase* v) { v->setShowSurfaceNormals(showSurfaceNormals); });
    if (hasShowDirectionHints)
        broadcast([showDirectionHints](VolumeViewerBase* v) { v->setShowDirectionHints(showDirectionHints); });
    if (hasSurfaceOverlayEnabled)
        broadcast([surfaceOverlayEnabled](VolumeViewerBase* v) { v->setSurfaceOverlayEnabled(surfaceOverlayEnabled); });
    if (hasNormalArrowLengthScale) {
        // Persist under the same key + storage format (int percent) that
        // ViewerNormalVisualizationPanel writes, so a later GUI open/RPC
        // read-back agree instead of the RPC being a no-op with zero viewers.
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE,
                           static_cast<int>(std::lround(normalArrowLengthScale * 100.0f)));
        broadcast([normalArrowLengthScale](VolumeViewerBase* base) {
            base->setNormalArrowLengthScale(normalArrowLengthScale);
        });
    }
    if (hasNormalMaxArrows) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::viewer::NORMAL_MAX_ARROWS, normalMaxArrows);
        broadcast([normalMaxArrows](VolumeViewerBase* base) { base->setNormalMaxArrows(normalMaxArrows); });
    }

    // Echo the resulting full settings (unknown keys are ignored).
    return viewerRenderSettingsJson();
}


QJsonObject AgentBridgeServer::handleCatalogOpenSample(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString sampleId = jsonRequireString(p, "sampleId");
    if (sampleId.isEmpty()) {
        QJsonObject data;
        data["param"] = "sampleId";
        throw AgentBridgeError{-32602, "sampleId is required", data};
    }
    if (p.contains("resources") && !p.value("resources").isObject()) {
        QJsonObject data;
        data["param"] = "resources";
        throw AgentBridgeError{-32602, "resources must be an object", data};
    }

    if (!_window) {
        QJsonObject data;
        data["detail"] = "window is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    // Resolve the sample first so we can distinguish "unknown sample" (-32007)
    // from "manifest unavailable" / "open failed" (-32005) (SPEC §3.16).
    const vc3d::opendata::OpenDataManifest* manifest = _window->cachedOpenDataManifest();
    if (!manifest) {
        QJsonObject data;
        data["detail"] = "Open Data manifest is unavailable";
        throw AgentBridgeError{-32005, "Manifest fetch failed", data};
    }
    const vc3d::opendata::OpenDataSample* sample =
        manifest->findSample(sampleId.toStdString());
    if (!sample) {
        QJsonObject data;
        data["kind"] = "sample";
        data["id"] = sampleId;
        throw AgentBridgeError{-32007, QStringLiteral("Unknown sample: %1").arg(sampleId), data};
    }

    MenuActionController* mc = _window->_menuController.get();
    if (!mc) {
        QJsonObject data;
        data["detail"] = "menu controller is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    // --- Optional resource selection (SPEC §10.3). Validated in full against
    // the resolved sample *before* any download or project mutation begins. ---
    OpenDataSampleOpenOptions options;
    options.interactive = false;  // SPEC §8.2: explicit call = consent to replace.

    if (p.contains("resources")) {
        const QJsonObject res = p.value("resources").toObject();
        auto& selection = options.selection;

        // Set of the sample's real volume ids (validation + zero-volume check).
        std::vector<std::string> sampleVolumeIds;
        for (const auto& v : sample->volumes)
            sampleVolumeIds.push_back(v.id);

        // volumeIds axis.
        if (res.contains("volumeIds")) {
            const QJsonValue vidsv = res.value("volumeIds");
            if (!vidsv.isArray()) {
                QJsonObject data;
                data["param"] = "resources.volumeIds";
                throw AgentBridgeError{-32602, "resources.volumeIds must be an array", data};
            }
            std::vector<std::string> vids;
            for (const QJsonValue& vv : vidsv.toArray()) {
                const std::string vid =
                    jsonRequireString(vv, "resources.volumeIds").toStdString();
                if (std::find(sampleVolumeIds.begin(), sampleVolumeIds.end(), vid) ==
                    sampleVolumeIds.end()) {
                    QJsonObject data;
                    data["kind"] = "resource";
                    data["id"] = QString::fromStdString(vid);
                    throw AgentBridgeError{-32007,
                        QStringLiteral("Unknown volumeId: %1").arg(QString::fromStdString(vid)),
                        data};
                }
                vids.push_back(vid);
            }
            // A selection that leaves zero volumes is rejected (SPEC §10.3).
            if (vids.empty()) {
                QJsonObject data;
                data["param"] = "resources.volumeIds";
                throw AgentBridgeError{-32602,
                    "resources.volumeIds selects zero volumes", data};
            }
            selection.volumeIds = std::move(vids);
        }

        // representationRefs axis. Each "vi:ai" must name a real derived
        // representation of the sample (from derivedRepresentations()).
        if (res.contains("representationRefs")) {
            const QJsonValue refsv = res.value("representationRefs");
            if (!refsv.isArray()) {
                QJsonObject data;
                data["param"] = "resources.representationRefs";
                throw AgentBridgeError{-32602,
                    "resources.representationRefs must be an array", data};
            }
            const auto derived = vc3d::opendata::derivedRepresentations(*sample);
            std::vector<vc3d::opendata::OpenDataRepresentationRef> refs;
            for (const QJsonValue& rv : refsv.toArray()) {
                const QString refStr =
                    jsonRequireString(rv, "resources.representationRefs");
                const auto reject = [&]() {
                    QJsonObject data;
                    data["kind"] = "resource";
                    data["id"] = refStr;
                    throw AgentBridgeError{-32007,
                        QStringLiteral("Invalid representation ref: %1").arg(refStr), data};
                };
                const int colon = refStr.indexOf(QLatin1Char(':'));
                if (colon <= 0 || colon == refStr.size() - 1)
                    reject();
                bool okVi = false, okAi = false;
                const qulonglong vi = refStr.left(colon).toULongLong(&okVi);
                const qulonglong ai = refStr.mid(colon + 1).toULongLong(&okAi);
                if (!okVi || !okAi)
                    reject();
                const auto match = std::find_if(derived.begin(), derived.end(),
                    [&](const vc3d::opendata::OpenDataRepresentationRef& r) {
                        return r.volumeIndex == vi && r.artifactIndex == ai;
                    });
                if (match == derived.end())
                    reject();
                refs.push_back(*match);
            }
            selection.representations = std::move(refs);
        }

        // kinds axis.
        if (res.contains("kinds")) {
            const QJsonValue kindsv = res.value("kinds");
            if (!kindsv.isArray()) {
                QJsonObject data;
                data["param"] = "resources.kinds";
                throw AgentBridgeError{-32602, "resources.kinds must be an array", data};
            }
            std::vector<vc3d::opendata::OpenDataRepresentationKind> kinds;
            for (const QJsonValue& kv : kindsv.toArray()) {
                const QString ks = jsonRequireString(kv, "resources.kinds");
                const auto kind = representationKindFromJson(ks);
                if (!kind) {
                    QJsonObject data;
                    data["kind"] = "resource";
                    data["id"] = ks;
                    throw AgentBridgeError{-32007,
                        QStringLiteral("Unknown resource kind: %1").arg(ks), data};
                }
                kinds.push_back(*kind);
            }
            selection.kinds = std::move(kinds);
        }
    }

    // Async open (SPEC §18.4): the RPC no longer blocks on the network. Reject a
    // second catalog open (bridge-started or human-initiated) up front, then
    // start the §1.3-safe asynchronous core and return a jobId immediately.
    requireSourceIdle(QStringLiteral("catalog"));
    if (mc->openDataSampleOpenInFlight()) {
        QJsonObject data;
        data["source"] = "catalog";
        data["detail"] = "an interactive Open Data open is in progress";
        throw AgentBridgeError{-32004,
            "An Open Data sample open is already in progress", data};
    }

    // Progress: forward the download/transform stream as job.progress "output"
    // notifications, rate-limited to <=10/s (SPEC §3.18).
    auto onProgress =
        [this](const vc3d::opendata::OpenDataSampleDownloadProgress& progress) {
            auto it = _activeJobs.find(QStringLiteral("catalog"));
            if (it == _activeJobs.end())
                return;
            const qint64 now = QDateTime::currentMSecsSinceEpoch();
            if (now - _lastConsoleBroadcastMs < 100)
                return;
            _lastConsoleBroadcastMs = now;
            const int done = progress.completedSegments + progress.failedSegments;
            QString label;
            const QString status = QString::fromStdString(progress.status);
            if (status == QLatin1String("resolving-volumes")) {
                label = QStringLiteral("Opening remote volumes in parallel...");
            } else if (status == QLatin1String("project-ready")) {
                label = QStringLiteral("Open-data project is ready.");
            } else if (status.startsWith(QLatin1String("placeholder"))) {
                label = QStringLiteral("Preparing segment metadata: %1/%2 representations.")
                            .arg(progress.completedSegments).arg(progress.totalSegments);
            } else if (status.startsWith(QLatin1String("transform-"))) {
                label = QStringLiteral("Transforming segments: %1/%2 transforms.")
                            .arg(done).arg(progress.totalSegments);
            } else {
                label = QStringLiteral("Downloading segments: %1/%2 segments, %3/%4 files.")
                            .arg(done).arg(progress.totalSegments)
                            .arg(progress.completedFiles).arg(progress.totalFiles);
            }
            broadcastJobProgress(it.value(), QStringLiteral("output"), label);
        };

    auto onFinished =
        [this](const MenuActionController::OpenDataSampleOpenOutcome& outcome) {
            completeCatalogOpenJob(outcome);
        };

    _catalogOpenSampleId = sampleId;
    QString err;
    if (!mc->startOpenDataSampleOpen(sampleId, options, onFinished, onProgress, &err)) {
        _catalogOpenSampleId.clear();
        QJsonObject data;
        data["detail"] = err.isEmpty()
            ? QStringLiteral("failed to open open-data sample %1").arg(sampleId)
            : err;
        throw AgentBridgeError{-32005, "Open Data sample open failed", data};
    }

    const QString jobId = beginJob(QStringLiteral("catalog"),
                                   QStringLiteral("catalog.open_sample"),
                                   QStringLiteral("open sample %1").arg(sampleId),
                                   /*broadcastStart=*/true);

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "catalog.open_sample";
    result["source"] = "catalog";
    result["sampleId"] = sampleId;
    return result;
}


// ---------------------------------------------------------------------------
// Remote catalog resource selection (SPEC §10.1-10.4)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::withOpenDataManifest(
    bool refresh,
    const std::function<QJsonObject(const vc3d::opendata::OpenDataManifest&)>& build)
{
    if (!_window) {
        QJsonObject data;
        data["detail"] = "window is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    if (!refresh) {
        if (const vc3d::opendata::OpenDataManifest* cached =
                _window->cachedOpenDataManifest()) {
            return build(*cached);  // AgentBridgeError from build propagates to dispatch.
        }
    }

    // No cached manifest (or a forced refresh): fetch off-thread and reply via
    // the deferred mechanism (SPEC §8.4, 30 s cap per §10.1).
    const int token = beginDeferred(30000, "Open Data manifest fetch");
    startManifestFetch(token, build);
    throw AgentBridgeDeferred{};
}


void AgentBridgeServer::startManifestFetch(
    int token,
    const std::function<QJsonObject(const vc3d::opendata::OpenDataManifest&)>& build)
{
    struct FetchResult {
        std::optional<vc3d::opendata::OpenDataManifest> manifest;
        QString error;
    };

    auto* watcher = new QFutureWatcher<FetchResult>(this);
    connect(watcher, &QFutureWatcher<FetchResult>::finished, this,
            [this, token, build, watcher]() {
                FetchResult fr = watcher->result();
                watcher->deleteLater();
                if (!fr.manifest) {
                    QJsonObject data;
                    data["detail"] = fr.error.isEmpty()
                        ? QStringLiteral("Open Data manifest fetch failed")
                        : fr.error;
                    completeDeferredError(token, -32005, "Manifest fetch failed", data);
                    return;
                }
                // Cache for subsequent list/describe/open calls (the bridge is a
                // CWindow friend). cachedOpenDataManifest() checks this first.
                _window->_openDataManifestCache = *fr.manifest;
                _window->_openDataManifestLoadAttempted = true;
                try {
                    completeDeferredResult(token, build(*fr.manifest));
                } catch (const AgentBridgeError& e) {
                    completeDeferredError(token, e.code, e.message, e.data);
                } catch (const std::exception& e) {
                    QJsonObject data;
                    data["detail"] = QString::fromUtf8(e.what());
                    completeDeferredError(token, -32010, "Internal error", data);
                }
            });

    watcher->setFuture(QtConcurrent::run([]() {
        FetchResult fr;
        try {
            fr.manifest = vc3d::opendata::fetchOpenDataManifest();
        } catch (const std::exception& e) {
            fr.error = QString::fromUtf8(e.what());
        } catch (...) {
            fr.error = QStringLiteral("unknown error fetching Open Data manifest");
        }
        return fr;
    }));
}


QJsonObject AgentBridgeServer::handleCatalogListSamples(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const bool refresh = jsonOptionalBool(p, "refresh", false);

    auto build = [](const vc3d::opendata::OpenDataManifest& manifest) -> QJsonObject {
        QJsonObject result;
        result["manifestUrl"] = QString::fromStdString(manifest.manifestUrl);
        QJsonArray samples;
        for (const auto& s : manifest.samples) {
            QJsonObject o;
            o["id"] = QString::fromStdString(s.id);
            o["type"] = QString::fromStdString(s.type);
            o["description"] = QString::fromStdString(s.description);
            o["volumeCount"] = static_cast<int>(s.volumeCount());
            o["segmentCount"] = static_cast<int>(s.segmentCount());
            o["scanCount"] = static_cast<int>(s.scanCount());
            samples.append(o);
        }
        result["samples"] = samples;
        return result;
    };

    return withOpenDataManifest(refresh, build);
}


QJsonObject AgentBridgeServer::handleCatalogDescribeSample(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString sampleId = jsonRequireString(p, "sampleId");
    if (sampleId.isEmpty()) {
        QJsonObject data;
        data["param"] = "sampleId";
        throw AgentBridgeError{-32602, "sampleId is required", data};
    }
    const bool refresh = jsonOptionalBool(p, "refresh", false);
    const std::string sid = sampleId.toStdString();

    auto build = [sampleId, sid](const vc3d::opendata::OpenDataManifest& manifest) -> QJsonObject {
        const vc3d::opendata::OpenDataSample* sample = manifest.findSample(sid);
        if (!sample) {
            QJsonObject data;
            data["kind"] = "sample";
            data["id"] = sampleId;
            throw AgentBridgeError{-32007,
                QStringLiteral("Unknown sample: %1").arg(sampleId), data};
        }

        QJsonObject result;
        result["sampleId"] = QString::fromStdString(sample->id);
        result["type"] = QString::fromStdString(sample->type);
        result["description"] = QString::fromStdString(sample->description);
        result["segmentCount"] = static_cast<int>(sample->segmentCount());

        QJsonArray volumes;
        for (const auto& v : sample->volumes) {
            QJsonObject o;
            o["id"] = QString::fromStdString(v.id);
            o["scanId"] = QString::fromStdString(v.scanId);
            if (v.shapeZYX) {
                QJsonArray shape;
                shape.append(static_cast<double>((*v.shapeZYX)[0]));
                shape.append(static_cast<double>((*v.shapeZYX)[1]));
                shape.append(static_cast<double>((*v.shapeZYX)[2]));
                o["shapeZYX"] = shape;
            } else {
                o["shapeZYX"] = QJsonValue::Null;
            }
            o["pixelSizeUm"] = v.pixelSizeUm ? QJsonValue(*v.pixelSizeUm)
                                             : QJsonValue(QJsonValue::Null);
            o["dataFormat"] = QString::fromStdString(v.dataFormat);
            volumes.append(o);
        }
        result["volumes"] = volumes;

        QJsonArray representations;
        for (const auto& ref : vc3d::opendata::derivedRepresentations(*sample)) {
            const auto& volume = sample->volumes[ref.volumeIndex];
            const auto& artifact = volume.artifacts[ref.artifactIndex];
            QJsonObject o;
            o["ref"] = QStringLiteral("%1:%2")
                           .arg(static_cast<qulonglong>(ref.volumeIndex))
                           .arg(static_cast<qulonglong>(ref.artifactIndex));
            o["volumeId"] = QString::fromStdString(volume.id);
            o["artifactType"] = QString::fromStdString(artifact.type);
            o["kind"] = representationKindToJson(ref.kind);
            o["url"] = artifact.resolvedUrl.empty()
                           ? QJsonValue(QJsonValue::Null)
                           : QJsonValue(QString::fromStdString(artifact.resolvedUrl));
            o["targetVolumeId"] = artifact.targetVolumeId
                ? QJsonValue(QString::fromStdString(*artifact.targetVolumeId))
                : QJsonValue(QJsonValue::Null);
            o["modelId"] = artifact.modelId
                ? QJsonValue(QString::fromStdString(*artifact.modelId))
                : QJsonValue(QJsonValue::Null);
            representations.append(o);
        }
        result["representations"] = representations;

        return result;
    };

    return withOpenDataManifest(refresh, build);
}


QJsonObject AgentBridgeServer::handleVolumeSelect(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString volumeIdQ = jsonRequireString(p, "volumeId");
    if (volumeIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "volumeId";
        throw AgentBridgeError{-32602, "volumeId is required", data};
    }
    const std::string volumeId = volumeIdQ.toStdString();

    const auto ids = vpkg->volumeIDs();
    if (std::find(ids.begin(), ids.end(), volumeId) == ids.end()) {
        QJsonObject data;
        data["kind"] = "volume";
        data["id"] = volumeIdQ;
        throw AgentBridgeError{-32007,
            QStringLiteral("Unknown volume id: %1").arg(volumeIdQ), data};
    }

    const QString previousVolumeId = QString::fromStdString(state->currentVolumeId());

    // Selecting the already-current volume is a no-op success (SPEC §10.4).
    if (previousVolumeId == volumeIdQ) {
        QJsonObject result;
        result["volumeId"] = volumeIdQ;
        result["previousVolumeId"] = previousVolumeId;
        return result;
    }

    // Switch through the same path the volume combo uses (setVolume ->
    // CState::setCurrentVolume + volumeChanged, then reconcile the selector UI),
    // so all viewer/UI updates fire exactly as for a human combo change.
    try {
        _window->setVolume(vpkg->volume(volumeId));
        _window->syncVolumeSelectionControls(volumeIdQ);
    } catch (const std::exception& e) {
        QJsonObject data;
        data["detail"] = QString::fromUtf8(e.what());
        throw AgentBridgeError{-32005, "Failed to switch volume", data};
    }

    QJsonObject result;
    result["volumeId"] = QString::fromStdString(state->currentVolumeId());
    result["previousVolumeId"] = previousVolumeId;
    return result;
}
