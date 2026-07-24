#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QBuffer>
#include <QByteArray>
#include <QCoreApplication>
#include <QImage>
#include <QJsonArray>
#include <QPixmap>
#include <QPointF>
#include <QRegularExpression>
#include <QWidget>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "CWindow.hpp"
#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "SegmentationCommandHandler.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "WrapAnnotationWidget.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"

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

    // Line and interpolation modes persist when manual-add is inactive.
    if (SegmentationModule* mod = _window->_segmentationModule.get()) {
        result["manualAddMode"] = mod->manualAddMode();
        result["correctionsPointMode"] = mod->correctionPointMode();
        // Explicit save/flush bookkeeping.
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

    // Axis-aligned slice mode: when enabled, "seg xz"/"seg yz" are the
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

    // Same-winding wrap annotation. `enabled` is the widget's
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

    // "job" is the most recently started active job; "jobs" lists all of them.
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

    const QJsonObject p = params.toObject();
    const bool onlyLoaded = p.value("onlyLoaded").toBool(false);

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

    const QJsonObject p = params.toObject();
    const QString segmentIdQ = p.value("segmentId").toString();
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

    // Resolve the path for the result's segment entry. Prefer the
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

    // Activating the already-active id is a no-op success: no re-emit,
    // no side effects, mirroring the tree where re-clicking the current row is inert.
    const bool alreadyActive = !prevActive.empty() && prevActive == segmentId;

    if (!alreadyActive) {
        QString err;
        if (!panel->activateSurfaceById(segmentId, &err)) {
            // Classify via the distinct sentences activateSurfaceById produces.
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
        // A load exception is an operation failure, not an invalid identifier.
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

    const QJsonObject p = params.toObject();
    const QString segmentIdQ = p.value("segmentId").toString();
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
    // shares the same download subsystem, so reject it up front.
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
    const QJsonObject p = params.toObject();
    const QString target =
        p.value("target").toString(QStringLiteral("window"));

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
        const int maxDim = p.value("maxDim").toInt();
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

    const QString filePath = p.value("filePath").toString();
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
    const QJsonObject p = params.toObject();

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
// Project/catalog opening
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleProjectCreate(const QJsonValue& params)
{
    namespace fs = std::filesystem;

    const QJsonObject p = params.toObject();
    QString output = p.value("path").toString();
    const QString volume = p.value("volume").toString();
    if (output.trimmed().isEmpty()) {
        QJsonObject data{{"param", "path"}};
        throw AgentBridgeError{-32602, "path must not be blank", data};
    }
    if (volume.trimmed().isEmpty()) {
        QJsonObject data{{"param", "volume"}};
        throw AgentBridgeError{-32602, "volume must not be blank", data};
    }
    if (!output.endsWith(QStringLiteral(".volpkg.json"), Qt::CaseInsensitive))
        output += QStringLiteral(".volpkg.json");

    fs::path outputPath = fs::path(output.toStdString()).lexically_normal();
    if (!outputPath.is_absolute()) {
        QJsonObject data{{"param", "path"}};
        throw AgentBridgeError{-32602, "path must be absolute", data};
    }

    const std::string volumeLocation = volume.toStdString();
    if (!vc::project::isLocationRemote(volumeLocation) &&
        !vc::project::resolveLocalPath(volumeLocation).is_absolute()) {
        QJsonObject data{{"param", "volume"}};
        throw AgentBridgeError{-32602, "local volume path must be absolute", data};
    }

    std::error_code existsError;
    const bool outputExists = fs::exists(outputPath, existsError);
    if (existsError) {
        QJsonObject data{{"detail", QString::fromStdString(existsError.message())}};
        throw AgentBridgeError{-32005, "Could not inspect output path", data};
    }
    if (outputExists && !p.value("overwrite").toBool(false)) {
        QJsonObject data{{"path", QString::fromStdString(outputPath.string())}};
        throw AgentBridgeError{
            -32005,
            "output file already exists; pass overwrite=true to replace",
            data,
        };
    }

    const std::string validationError =
        vc::project::validateSingleVolumeLocation(volumeLocation);
    if (!validationError.empty()) {
        QJsonObject data{
            {"detail", QString::fromStdString(validationError)},
            {"kind", "volume"},
        };
        throw AgentBridgeError{-32007, "Invalid volume location", data};
    }

    QString projectName;
    if (p.contains("name")) {
        projectName = p.value("name").toString();
    } else {
        projectName = QString::fromStdString(outputPath.filename().string());
        projectName.chop(QStringLiteral(".volpkg.json").size());
        if (projectName.isEmpty())
            projectName = QStringLiteral("Untitled");
    }

    std::vector<std::string> tags;
    const QJsonArray tagValues = p.value("tags").toArray();
    tags.reserve(tagValues.size());
    for (const QJsonValue& tag : tagValues)
        tags.push_back(tag.toString().toStdString());

    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto package = VolumePkg::newDetached(options);
    package->setName(projectName.toStdString());
    if (!package->addVolumeEntry(volumeLocation, std::move(tags)))
        throw AgentBridgeError{-32010, "Could not add volume to project", {}};

    try {
        package->save(outputPath);
    } catch (const std::exception& error) {
        QJsonObject data{{"detail", QString::fromUtf8(error.what())}};
        throw AgentBridgeError{-32005, "Project write failed", data};
    }

    QJsonObject result;
    result["path"] = QString::fromStdString(outputPath.string());
    result["name"] = projectName;
    result["volume"] =
        QString::fromStdString(package->volumeEntries().front().location);
    return result;
}

QJsonObject AgentBridgeServer::handleVolumeAttach(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString location = p.value("location").toString();
    if (location.trimmed().isEmpty()) {
        throw AgentBridgeError{
            -32602,
            "location must not be blank",
            QJsonObject{{"param", "location"}},
        };
    }
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        throw AgentBridgeError{
            -32000,
            "No volume package loaded",
            QJsonObject{},
        };
    }

    const std::string input = location.toStdString();
    if (!vc::project::isLocationRemote(input) &&
        !vc::project::resolveLocalPath(input).is_absolute()) {
        throw AgentBridgeError{
            -32602,
            "local volume path must be absolute",
            QJsonObject{{"param", "location"}},
        };
    }
    VolumeAttachmentController* attachment =
        _window->_volumeAttachmentController.get();
    if (!attachment) {
        throw AgentBridgeError{
            -32010,
            "Internal error",
            QJsonObject{{"detail", "volume attachment controller is not available"}},
        };
    }

    requireSourceIdle(QStringLiteral("volume"));
    if (attachment->inFlight()) {
        throw AgentBridgeError{
            -32004,
            "A volume attachment is already in progress",
            QJsonObject{
                {"source", "volume"},
                {"detail", "an interactive volume attachment is in progress"},
            },
        };
    }

    std::vector<std::string> tags;
    const QJsonArray tagValues = p.value("tags").toArray();
    tags.reserve(tagValues.size());
    for (const QJsonValue& tag : tagValues)
        tags.push_back(tag.toString().toStdString());

    VolumeAttachmentRequest request;
    VolumeAttachmentPreparationFailure preparationFailure =
        VolumeAttachmentPreparationFailure::None;
    QString error;
    if (!attachment->prepare(
            location,
            std::move(tags),
            VolumeAttachmentPresentation::Silent,
            &request,
            &error,
            &preparationFailure)) {
        if (preparationFailure ==
            VolumeAttachmentPreparationFailure::InvalidLocation) {
            throw AgentBridgeError{
                -32007,
                "Invalid volume location",
                QJsonObject{{"detail", error}, {"kind", "volume"}},
            };
        }
        if (preparationFailure ==
            VolumeAttachmentPreparationFailure::NoProject) {
            throw AgentBridgeError{
                -32000,
                "No volume package loaded",
                QJsonObject{},
            };
        }
        throw AgentBridgeError{
            -32005,
            "Could not prepare volume attachment",
            QJsonObject{{"detail", error}},
        };
    }

    request.selection = VolumeAttachmentSelection::PreserveCurrent;
    QPointer<AgentBridgeServer> self(this);
    if (!attachment->start(
            std::move(request),
            [self](const VolumeAttachmentOutcome& outcome) {
                if (self)
                    self->completeVolumeAttachmentJob(outcome);
            },
            &error)) {
        throw AgentBridgeError{
            -32005,
            "Could not start volume attachment",
            QJsonObject{{"detail", error}},
        };
    }

    // The watcher completes through the GUI event loop, after this handler has
    // installed the corresponding job record.
    const QString jobId = beginJob(
        QStringLiteral("volume"),
        QStringLiteral("volume.attach"),
        QStringLiteral("attach volume %1").arg(location),
        /*broadcastStart=*/true);

    return {
        {"jobId", jobId},
        {"kind", "volume.attach"},
        {"source", "volume"},
        {"location", location},
    };
}


QJsonObject AgentBridgeServer::handleVolumeOpen(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString path = p.value("path").toString();
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

    const QString volumeId = p.value("volumeId").toString();
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
    // Reading voxel sizes here would force-load every possibly remote volume.
    return result;
}


QJsonObject AgentBridgeServer::handleSegmentsDelete(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = params.toObject();
    const QString segmentIdQ = p.value("segmentId").toString();
    const bool confirm = p.value("confirm").toBool(false);
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

    const QJsonObject p = params.toObject();
    const QString segmentIdQ = p.value("segmentId").toString();
    const QString newName = p.value("newName").toString();

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


QJsonObject AgentBridgeServer::handleVolumeSelect(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString volumeIdQ = p.value("volumeId").toString();
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

    // Selecting the already-current volume is a no-op success.
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
