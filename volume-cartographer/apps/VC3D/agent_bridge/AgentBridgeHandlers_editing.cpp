#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QBuffer>
#include <QCheckBox>
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

namespace {

Qt::MouseButton jsonToMouseButton(const QJsonValue& value)
{
    const QString s = value.toString(QStringLiteral("left"));
    if (s == QLatin1String("right"))
        return Qt::RightButton;
    if (s == QLatin1String("middle"))
        return Qt::MiddleButton;
    return Qt::LeftButton;
}
QString mouseButtonToJson(Qt::MouseButton button)
{
    switch (button) {
    case Qt::RightButton:  return QStringLiteral("right");
    case Qt::MiddleButton: return QStringLiteral("middle");
    case Qt::LeftButton:
    default:               return QStringLiteral("left");
    }
}

Qt::KeyboardModifiers jsonToModifiers(const QJsonValue& value)
{
    Qt::KeyboardModifiers mods = Qt::NoModifier;
    if (value.isUndefined() || value.isNull())
        return mods;
    for (const QJsonValue& mv : value.toArray()) {
        const QString s = mv.toString();
        if (s == QLatin1String("shift"))
            mods |= Qt::ShiftModifier;
        else if (s == QLatin1String("ctrl"))
            mods |= Qt::ControlModifier;
        else if (s == QLatin1String("alt"))
            mods |= Qt::AltModifier;
        else if (s == QLatin1String("meta"))
            mods |= Qt::MetaModifier;
        else if (s == QLatin1String("keypad"))
            mods |= Qt::KeypadModifier;
    }
    return mods;
}

QJsonArray modifiersToJson(Qt::KeyboardModifiers mods)
{
    QJsonArray arr;
    if (mods & Qt::ShiftModifier)   arr.append(QStringLiteral("shift"));
    if (mods & Qt::ControlModifier) arr.append(QStringLiteral("ctrl"));
    if (mods & Qt::AltModifier)     arr.append(QStringLiteral("alt"));
    if (mods & Qt::MetaModifier)    arr.append(QStringLiteral("meta"));
    if (mods & Qt::KeypadModifier)  arr.append(QStringLiteral("keypad"));
    return arr;
}

} // namespace


// ---------------------------------------------------------------------------
// Canvas + viewer control
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleCanvasClick(const QJsonValue& params, bool addShift)
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

    Qt::MouseButton button = jsonToMouseButton(p.value("button"));
    Qt::KeyboardModifiers modifiers = jsonToModifiers(p.value("modifiers"));
    if (addShift)
        modifiers |= Qt::ShiftModifier;

    const QString space =
        p.value("space").toString(QStringLiteral("volume"));
    QPointF scenePos;
    if (space == QLatin1String("scene")) {
        const QJsonValue posv = p.value("position");
        const QJsonObject po = posv.toObject();
        scenePos = QPointF(jsonRequireFiniteFloat(po.value("x"), "x"),
                           jsonRequireFiniteFloat(po.value("y"), "y"));
    } else if (space == QLatin1String("volume")) {
        const cv::Vec3f vol = jsonToVec3(p.value("position"), "position");
        scenePos = chunked->volumeToScene(vol);
        // Verify the round-trip lands within 2.0 voxels: otherwise the point is
        // not on this viewer's current slice/surface view.
        const cv::Vec3f back = chunked->sceneToVolume(scenePos);
        const double dist = cv::norm(back - vol);
        if (!std::isfinite(dist) || dist > 2.0) {
            QJsonObject data;
            data["point"] = vec3ToJson(vol);
            data["detail"] = QStringLiteral("point is not on this viewer's view (round-trip %1 voxels)")
                                 .arg(dist, 0, 'f', 3);
            throw AgentBridgeError{-32003, "Invalid coordinates", data};
        }
    }

    // Synthesize the full click through the real mouse slots so all signal
    // wiring (sendVolumeClicked -> CWindow::onVolumeClicked, point placement,
    // tools) fires exactly as for a human click.
    chunked->onMousePress(scenePos, button, modifiers);
    chunked->onMouseRelease(scenePos, button, modifiers);
    chunked->onVolumeClicked(scenePos, button, modifiers);

    QJsonValue volumePointJson = QJsonValue::Null;
    if (const auto sample = chunked->sampleSceneVolume(scenePos))
        volumePointJson = vec3ToJson(sample->position);

    QJsonObject scene;
    scene["x"] = scenePos.x();
    scene["y"] = scenePos.y();

    QJsonObject result;
    result["clicked"] = true;
    result["scene"] = scene;
    result["volumePoint"] = volumePointJson;
    result["button"] = mouseButtonToJson(button);
    result["modifiers"] = modifiersToJson(modifiers);
    return result;
}


QJsonObject AgentBridgeServer::handleViewerCenterOnPoint(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    const auto volume = viewer->currentVolume();
    if (!volume)
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const cv::Vec3f point = jsonToVec3(p.value("point"), "point");
    if (!volumePointInBounds(point, volume->shapeXyz())) {
        QJsonObject data;
        data["point"] = vec3ToJson(point);
        throw AgentBridgeError{-32003, "Point is outside volume bounds", data};
    }
    const bool forceRender = p.value("forceRender").toBool(true);
    viewer->centerOnVolumePoint(point, forceRender);

    QJsonObject result;
    result["centered"] = true;
    result["viewerId"] = viewerIdFor(viewer);
    return result;
}


QJsonObject AgentBridgeServer::handleViewerZoom(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    const double factor = p.value("factor").toDouble();

    viewer->adjustZoomByFactor(static_cast<float>(factor));

    QJsonObject result;
    result["scale"] = viewer->getCurrentScale();
    return result;
}


QJsonObject AgentBridgeServer::handleViewerRotate(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();

    AxisAlignedSliceController* slices =
        _window ? _window->_axisAlignedSliceController.get() : nullptr;
    if (!slices)
        throw AgentBridgeError{-32000, "Axis-aligned slice controller unavailable", {}};

    // Accept "seg xz"/"seg yz" and the "xz"/"yz" shorthands.
    QString planeRaw = p.value("plane").toString().trimmed().toLower();
    std::string plane;
    if (planeRaw == "seg xz" || planeRaw == "xz")
        plane = "seg xz";
    else if (planeRaw == "seg yz" || planeRaw == "yz")
        plane = "seg yz";
    else {
        QJsonObject data;
        data["param"] = "plane";
        data["value"] = planeRaw;
        data["allowed"] = QJsonArray{"seg xz", "seg yz"};
        throw AgentBridgeError{-32602,
            "plane must be one of the rotatable axis-aligned planes: \"seg xz\", \"seg yz\"",
            data};
    }

    const double degrees = p.value("degrees").toDouble();

    // Default to relative (delta) rotation — matches the middle-drag interaction.
    const bool relative = p.value("relative").toBool(true);

    if (!slices->isEnabled())
        throw AgentBridgeError{-32002,
            "Axis-aligned slice mode is not active; enable it before rotating the seg xz/yz planes",
            {}};

    const float previous = slices->currentRotationDegrees(plane);
    const float target = relative ? static_cast<float>(previous + degrees)
                                  : static_cast<float>(degrees);
    slices->setRotationDegrees(plane, target);
    // scheduleOrientationUpdate() marks the orientation dirty (exactly as the
    // human middle-drag does); flushOrientationUpdate() early-returns unless the
    // dirty flag is set, so without this the plane would never be reconfigured or
    // repainted even though _segXZ/YZRotationDeg was updated.
    slices->scheduleOrientationUpdate();
    slices->flushOrientationUpdate();

    QJsonObject result;
    result["plane"] = QString::fromStdString(plane);
    result["degrees"] = slices->currentRotationDegrees(plane);
    result["previousDegrees"] = previous;
    result["relative"] = relative;
    return result;
}


QJsonObject AgentBridgeServer::handleViewerSetAxisAlignedSlices(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();

    AxisAlignedSliceController* slices =
        _window ? _window->_axisAlignedSliceController.get() : nullptr;
    if (!slices)
        throw AgentBridgeError{-32000, "Axis-aligned slice controller unavailable", {}};

    const bool enabled = p.value(QStringLiteral("enabled")).toBool();

    // Drive the checkbox so the toggle takes the human/shortcut path
    // (CWindow::onAxisAlignedSlicesToggled); setChecked emits toggled
    // synchronously, so slices->isEnabled() is up to date on return.
    QCheckBox* chk = _window ? _window->chkAxisAlignedSlices : nullptr;
    if (chk) {
        if (chk->isChecked() != enabled)
            chk->setChecked(enabled);
    } else {
        // No UI checkbox available (not expected in a normal bridge session):
        // drive the controller directly as a fallback.
        slices->setEnabled(enabled);
    }

    QJsonObject result;
    result["enabled"] = slices->isEnabled();
    return result;
}


// ---------------------------------------------------------------------------
// Segmentation editing + growth
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentationEnableEditing(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const bool enabled = p.value("enabled").toBool();

    SegmentationWidget* widget = _window->_segmentationWidget;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "segmentation widget is not available";
        throw AgentBridgeError{-32009, "Segmentation editing unavailable", data};
    }

    // Enabling requires an active surface to edit.
    if (enabled && state->activeSurfaceId().empty()) {
        QJsonObject data;
        data["kind"] = "segment";
        data["detail"] = "no active surface to edit";
        throw AgentBridgeError{-32007, "No active segmentation surface", data};
    }

    widget->setEditingEnabled(enabled);

    // widget->setEditingEnabled() is silent (never emits editingModeChanged), so
    // on its own it never reaches SegmentationModule's signal cascade into
    // beginEditingSession(); drive the module directly too (as CWindow's
    // onSurfaceActivated does) so a real edit session is established.
    if (SegmentationModule* mod = _window->_segmentationModule.get()) {
        if (mod->editingEnabled() != enabled) {
            mod->setEditingEnabled(enabled);
        }
    }

    QJsonObject result;
    result["enabled"] = widget->isEditingEnabled();
    return result;
}


QJsonObject AgentBridgeServer::handleSegmentationGrow(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    // Growth method accepted by the wire contract.
    const QString methodStr =
        p.value("method").toString(QStringLiteral("tracer"));
    // Manual-add is an interactive editing mode, not a grow invocation.
    // Feeding SegmentationGrowthMethod::ManualAdd through
    // onGrowSegmentationSurface from the bridge would bypass the mode's session
    // state, so the RPC rejects it and points at the future manual_add RPCs.
    // The C++ enum value itself is untouched (the in-app apply path still uses it).
    if (methodStr == QLatin1String("manual_add")) {
        QJsonObject data;
        data["param"] = "method";
        data["value"] = methodStr;
        data["detail"] = "manual_add is not a growth method; use "
                         "segmentation.manual_add.begin/finish";
        throw AgentBridgeError{-32009,
            "manual_add is not a growth method", data};
    }
    SegmentationGrowthMethod method;
    if (methodStr == QLatin1String("tracer"))
        method = SegmentationGrowthMethod::Tracer;
    else if (methodStr == QLatin1String("corrections"))
        method = SegmentationGrowthMethod::Corrections;
    else if (methodStr == QLatin1String("patch_tracer"))
        method = SegmentationGrowthMethod::PatchTracer;
    else {
        QJsonObject data;
        data["param"] = "method";
        data["value"] = methodStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid method: %1").arg(methodStr), data};
    }

    // Direction enum.
    const QString dirStr =
        p.value("direction").toString(QStringLiteral("all"));
    SegmentationGrowthDirection direction;
    if (dirStr == QLatin1String("all"))
        direction = SegmentationGrowthDirection::All;
    else if (dirStr == QLatin1String("up"))
        direction = SegmentationGrowthDirection::Up;
    else if (dirStr == QLatin1String("down"))
        direction = SegmentationGrowthDirection::Down;
    else if (dirStr == QLatin1String("left"))
        direction = SegmentationGrowthDirection::Left;
    else if (dirStr == QLatin1String("right"))
        direction = SegmentationGrowthDirection::Right;
    else
        direction = SegmentationGrowthDirection::Fill;

    const int steps = p.value("steps").toInt();
    const bool inpaintOnly = p.value("inpaintOnly").toBool(false);

    // Growth has its own source, so other job types may run concurrently but a
    // second growth job may not.
    requireSourceIdle(QStringLiteral("growth"));

    // The grower operates on the surface registered under the "segmentation"
    // slot; without it there is nothing to grow.
    if (!state->surface("segmentation")) {
        QJsonObject data;
        data["kind"] = "segment";
        data["detail"] = "no active segmentation surface";
        throw AgentBridgeError{-32007, "No active segmentation surface", data};
    }

    if (!_window->_segmentationWidget || !_window->_segmentationWidget->isEditingEnabled())
        throw AgentBridgeError{-32008, "Segmentation editing is not enabled", {}};

    // Create the job record up front so the growth-status signal (fired
    // synchronously by SegmentationGrower::start) does not create a duplicate,
    // then confirm the grower actually started before broadcasting.
    const QString jobId = beginJob(QStringLiteral("growth"),
                                   QStringLiteral("segmentation.grow"),
                                   QStringLiteral("Grow %1 (%2, %3 steps)")
                                       .arg(methodStr, dirStr).arg(steps),
                                   /*broadcastStart=*/false);

    _window->onGrowSegmentationSurface(method, direction, steps, inpaintOnly);

    const bool started = _window->_segmentationGrower && _window->_segmentationGrower->running();
    if (!started) {
        // Growth was rejected synchronously (e.g. invalid custom params). The
        // false status signal may already have cleared the active growth job.
        _activeJobs.remove(QStringLiteral("growth"));
        QJsonObject data;
        data["detail"] = "segmentation growth did not start";
        throw AgentBridgeError{-32005, "Failed to start segmentation growth", data};
    }

    if (auto it = _activeJobs.find(QStringLiteral("growth")); it != _activeJobs.end())
        broadcastJobProgress(it.value(), QStringLiteral("started"));

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segmentation.grow";
    return result;
}


QJsonObject AgentBridgeServer::handleSegmentationSave(const QJsonValue&)
{
    // Force the pending autosave to disk and report the flush as an async
    // "autosave"-source job. Params are ignored ({} accepted).
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    if (!mod) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32000, "Segmentation module unavailable", data};
    }

    const SegmentationModule::AutosaveStatus before = mod->autosaveStatus();

    // Nothing dirty and no save in flight -> nothing to flush. Return an idle
    // body (jobId:null) so callers can no-op without polling a job.
    if (!before.pending && !before.saveInProgress) {
        QJsonObject result;
        result["jobId"] = QJsonValue::Null;
        result["kind"] = "segmentation.save";
        result["state"] = "idle";
        result["pending"] = false;
        result["saveInProgress"] = false;
        result["dirtyAfterSave"] = false;
        return result;
    }

    // A save is pending or already running: model it as a job. A second explicit
    // save while one is running is rejected (-32004).
    requireSourceIdle(QStringLiteral("autosave"));

    const QString jobId = beginJob(QStringLiteral("autosave"),
                                   QStringLiteral("segmentation.save"),
                                   QStringLiteral("Save segment"),
                                   /*broadcastStart=*/false);

    // flushAutosave() -> markAutosaveNeeded(true) -> performAutosave():
    // saveInProgress flips synchronously when a save launches (or stays true if
    // one was already running). If it's still false afterward, performAutosave
    // early-returned (no resolvable surface / missing metadata) and no
    // autosaveCompleted signal will ever fire, so resolve the job here.
    mod->flushAutosave();
    const SegmentationModule::AutosaveStatus after = mod->autosaveStatus();

    if (!after.saveInProgress) {
        // The flush did not start (nor continue) a disk write: drop the job we
        // optimistically opened and report the resulting idle state.
        _activeJobs.remove(QStringLiteral("autosave"));
        QJsonObject result;
        result["jobId"] = QJsonValue::Null;
        result["kind"] = "segmentation.save";
        result["state"] = "idle";
        result["pending"] = after.pending;
        result["saveInProgress"] = false;
        result["dirtyAfterSave"] = after.dirtyAfterSave;
        return result;
    }

    if (auto it = _activeJobs.find(QStringLiteral("autosave")); it != _activeJobs.end())
        broadcastJobProgress(it.value(), QStringLiteral("started"));

    const JobRecord* rec = jobById(jobId);
    return rec ? jobStatusJson(*rec) : QJsonObject{};
}


QJsonObject AgentBridgeServer::handleSegmentationGrowPatchFromSeed(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    // GrowPatch runs the external vc_grow_seg_from_seed process: the "tool"
    // source.
    requireSourceIdle(QStringLiteral("tool"));

    const cv::Vec3f seed = jsonToVec3(p.value("seed"), "seed");

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "GrowPatch unavailable", data};
    }

    SegmentationCommandHandler::GrowPatchSeedParams gp;
    gp.volumeId = p.value("volumeId").toString();
    if (!gp.volumeId.isEmpty()) {
        const auto ids = state->vpkg()->volumeIDs();
        if (std::find(ids.begin(), ids.end(), gp.volumeId.toStdString()) == ids.end()) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = gp.volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(gp.volumeId), data};
        }
    }

    gp.iterations = p.value("iterations").toInt(200);
    gp.minAreaCm = p.value("minAreaCm").toDouble(0.002);
    gp.outputDir = p.value("outputDir").toString();

    const QString effectiveVolumeId = gp.volumeId.isEmpty()
        ? QString::fromStdString(state->currentVolumeId())
        : gp.volumeId;

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("segmentation.grow_patch_from_seed"),
                                   QStringLiteral("GrowPatch from seed"),
                                   /*broadcastStart=*/false);

    const QVector3D seedQ(seed[0], seed[1], seed[2]);
    CommandLaunchError error;
    if (!handler->startGrowPatchFromSeed(seedQ, gp, &error)) {
        _activeJobs.remove(QStringLiteral("tool"));
        throwCommandLaunchError(error, "Failed to start GrowPatch from seed",
                                {}, "tool");
    }

    const QString outputDir = handler->activeGrowPatchOutputDir();
    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segmentation.grow_patch_from_seed";
    result["outputDir"] = outputDir;
    result["volumeId"] = effectiveVolumeId;
    return result;
}


// ---------------------------------------------------------------------------
// Manual-add (hole-fill) + corrections point authoring
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleManualAddBegin(const QJsonValue&)
{
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!mod || !widget) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32000, "Segmentation module unavailable", data};
    }
    // Point authoring requires an active edit session with no growth in flight.
    if (!widget->isEditingEnabled())
        throw AgentBridgeError{-32008, "Segmentation editing is not enabled", {}};
    if (!mod->hasActiveSession()) {
        QJsonObject data;
        data["kind"] = "session";
        data["detail"] = "no active segmentation edit session";
        throw AgentBridgeError{-32007, "No active edit session", data};
    }
    requireSourceIdle(QStringLiteral("growth"));

    // Idempotent: already active -> report active without re-entering.
    if (mod->manualAddMode()) {
        QJsonObject result;
        result["active"] = true;
        return result;
    }
    if (!mod->setManualAddModeActive(true)) {
        // beginManualAdd rejected for a residual reason not covered by the
        // precondition checks above (pending edits, save in progress, or the
        // active surface could not be read).
        QJsonObject data;
        data["detail"] = "manual-add mode could not start (pending edits, save "
                         "in progress, or unreadable surface)";
        throw AgentBridgeError{-32005, "Manual add did not start", data};
    }
    QJsonObject result;
    result["active"] = mod->manualAddMode();
    return result;
}


QJsonObject AgentBridgeServer::handleManualAddFinish(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    if (!mod) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32000, "Segmentation module unavailable", data};
    }
    if (!mod->manualAddMode()) {
        QJsonObject data;
        data["kind"] = "manual_add_session";
        data["detail"] = "manual-add mode is not active";
        throw AgentBridgeError{-32007, "Manual add mode not active", data};
    }
    const bool apply = p.value("apply").toBool(true);
    const bool applied = mod->setManualAddModeActive(false, apply);
    QJsonObject result;
    result["applied"] = applied;
    return result;
}


QJsonObject AgentBridgeServer::handleManualAddSetLineMode(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "segmentation widget is not available";
        throw AgentBridgeError{-32000, "Segmentation widget unavailable", data};
    }
    const QString modeStr = p.value("mode").toString();
    ManualAddTool::LinePreviewMode mode;
    if (modeStr == QLatin1String("vertical"))
        mode = ManualAddTool::LinePreviewMode::VerticalOnly;
    else if (modeStr == QLatin1String("horizontal"))
        mode = ManualAddTool::LinePreviewMode::HorizontalOnly;
    else if (modeStr == QLatin1String("cross"))
        mode = ManualAddTool::LinePreviewMode::Cross;
    else if (modeStr == QLatin1String("cross_fill"))
        mode = ManualAddTool::LinePreviewMode::CrossFill;
    else
        mode = ManualAddTool::LinePreviewMode::CrossFill;
    const ManualAddTool::LinePreviewMode effective = widget->setManualAddLinePreviewMode(mode);
    QJsonObject result;
    result["mode"] = linePreviewModeToString(effective);
    return result;
}


QJsonObject AgentBridgeServer::handleManualAddSetInterpolation(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "segmentation widget is not available";
        throw AgentBridgeError{-32000, "Segmentation widget unavailable", data};
    }
    const QString modeStr = p.value("mode").toString();
    ManualAddTool::InterpolationMode mode;
    if (modeStr == QLatin1String("thin_plate_spline"))
        mode = ManualAddTool::InterpolationMode::ThinPlateSpline;
    else if (modeStr == QLatin1String("tracer_restricted_to_fill"))
        mode = ManualAddTool::InterpolationMode::TracerRestrictedToFill;
    else
        mode = ManualAddTool::InterpolationMode::TracerRestrictedToFill;
    const ManualAddTool::InterpolationMode effective = widget->setManualAddInterpolationMode(mode);
    QJsonObject result;
    result["mode"] = interpolationModeToString(effective);
    return result;
}


QJsonObject AgentBridgeServer::handleManualAddUndoConstraint(const QJsonValue&)
{
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    if (!mod) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32000, "Segmentation module unavailable", data};
    }
    if (!mod->manualAddMode()) {
        QJsonObject data;
        data["kind"] = "manual_add_session";
        data["detail"] = "manual-add mode is not active";
        throw AgentBridgeError{-32007, "Manual add mode not active", data};
    }
    const bool undone = mod->undoManualAddConstraint();
    QJsonObject result;
    result["undone"] = undone;
    return result;
}


QJsonObject AgentBridgeServer::handleCorrectionsSetPointMode(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const bool active = p.value("active").toBool();

    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!mod) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32000, "Segmentation module unavailable", data};
    }

    // Enabling enforces the same preconditions as the G-key handler;
    // disabling always succeeds. Pre-check here so failures map to the
    // documented JSON-RPC error codes rather than a generic false.
    if (active) {
        if (!widget || !widget->isEditingEnabled())
            throw AgentBridgeError{-32008, "Segmentation editing is not enabled", {}};
        if (!mod->hasActiveSession()) {
            QJsonObject data;
            data["kind"] = "session";
            data["detail"] = "no active segmentation edit session";
            throw AgentBridgeError{-32007, "No active edit session", data};
        }
        requireSourceIdle(QStringLiteral("growth"));
    }

    mod->setCorrectionPointMode(active);
    QJsonObject result;
    result["active"] = mod->correctionPointMode();
    return result;
}


// ---------------------------------------------------------------------------
// canvas.drag
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleCanvasDrag(const QJsonValue& params)
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

    // button, incl. the "none" hover-only variant.
    const QJsonValue btnv = p.value("button");
    const QString btnStr = btnv.toString(QStringLiteral("left"));
    const bool buttonNone = (btnStr == QLatin1String("none"));
    Qt::MouseButton button = Qt::NoButton;
    if (!buttonNone)
        button = jsonToMouseButton(btnv);
    const Qt::KeyboardModifiers modifiers = jsonToModifiers(p.value("modifiers"));

    const int steps = std::min(p.value("steps").toInt(8), 256);

    const QString space =
        p.value("space").toString(QStringLiteral("volume"));

    // Convert one endpoint (Vec3 volume, or {x,y} scene) to a scene point,
    // reusing the click path's round-trip validation. `name` is "from" / "to".
    auto convertEndpoint = [&](const QJsonValue& v, const char* name) -> QPointF {
        if (space == QLatin1String("scene")) {
            const QJsonObject o = v.toObject();
            return QPointF(jsonRequireFiniteFloat(o.value("x"), "x"),
                           jsonRequireFiniteFloat(o.value("y"), "y"));
        }
        const cv::Vec3f vol = jsonToVec3(v, name);
        const QPointF sc = chunked->volumeToScene(vol);
        const cv::Vec3f back = chunked->sceneToVolume(sc);
        const double dist = cv::norm(back - vol);
        if (!std::isfinite(dist) || dist > 2.0) {
            QJsonObject data;
            data["point"] = QString::fromLatin1(name);
            data["detail"] =
                QStringLiteral("%1 is not on this viewer's view (round-trip %2 voxels)")
                    .arg(QLatin1String(name)).arg(dist, 0, 'f', 3);
            throw AgentBridgeError{-32003, "Invalid coordinates", data};
        }
        return sc;
    };

    const QPointF sceneFrom = convertEndpoint(p.value("from"), "from");
    const QPointF sceneTo = convertEndpoint(p.value("to"), "to");

    // Dispatch press -> steps x move -> release through the real mouse slots so
    // all signal wiring fires exactly as for a human drag. For
    // button:"none" the press/release are skipped (hover-only positioning).
    if (!buttonNone)
        chunked->onMousePress(sceneFrom, button, modifiers);
    const Qt::MouseButtons moveButtons =
        buttonNone ? Qt::MouseButtons(Qt::NoButton) : Qt::MouseButtons(button);
    for (int i = 1; i <= steps; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(steps);
        const QPointF interp(sceneFrom.x() + (sceneTo.x() - sceneFrom.x()) * t,
                             sceneFrom.y() + (sceneTo.y() - sceneFrom.y()) * t);
        chunked->onMouseMove(interp, moveButtons, modifiers);
    }
    if (!buttonNone)
        chunked->onMouseRelease(sceneTo, button, modifiers);

    auto endpointJson = [&](const QPointF& sc) -> QJsonObject {
        QJsonObject o;
        QJsonObject scene;
        scene["x"] = sc.x();
        scene["y"] = sc.y();
        o["scene"] = scene;
        if (const auto sample = chunked->sampleSceneVolume(sc))
            o["volumePoint"] = vec3ToJson(sample->position);
        else
            o["volumePoint"] = QJsonValue::Null;
        return o;
    };

    QJsonObject result;
    result["dragged"] = true;
    result["from"] = endpointJson(sceneFrom);
    result["to"] = endpointJson(sceneTo);
    result["steps"] = steps;
    result["button"] = buttonNone ? QStringLiteral("none") : mouseButtonToJson(button);
    result["modifiers"] = modifiersToJson(modifiers);
    return result;
}
