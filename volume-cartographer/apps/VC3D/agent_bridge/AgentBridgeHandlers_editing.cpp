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
    // Absent/null -> default "left"; a value that IS present must be a valid button
    // string. A wrong-typed value (e.g. button:123) is rejected, not silently
    // coerced to "left" (SPEC §5).
    if (value.isUndefined() || value.isNull())
        return Qt::LeftButton;
    if (!value.isString()) {
        QJsonObject data;
        data["param"] = QStringLiteral("button");
        throw AgentBridgeError{-32602, QStringLiteral("button must be a string"), data};
    }
    const QString s = value.toString();
    if (s == QLatin1String("left"))
        return Qt::LeftButton;
    if (s == QLatin1String("right"))
        return Qt::RightButton;
    if (s == QLatin1String("middle"))
        return Qt::MiddleButton;
    QJsonObject data;
    data["param"] = QStringLiteral("button");
    data["value"] = s;
    throw AgentBridgeError{-32602, QStringLiteral("Invalid button: %1").arg(s), data};
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
    if (!value.isArray()) {
        QJsonObject data;
        data["param"] = QStringLiteral("modifiers");
        throw AgentBridgeError{-32602, "modifiers must be an array of strings", data};
    }
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
        else {
            QJsonObject data;
            data["param"] = QStringLiteral("modifiers");
            data["value"] = s;
            throw AgentBridgeError{-32602, QStringLiteral("Invalid modifier: %1").arg(s), data};
        }
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
// Canvas + viewer control (SPEC §3.6-3.9)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleCanvasClick(const QJsonValue& params, bool addShift)
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

    Qt::MouseButton button = jsonToMouseButton(p.value("button"));
    Qt::KeyboardModifiers modifiers = jsonToModifiers(p.value("modifiers"));
    if (addShift)
        modifiers |= Qt::ShiftModifier;

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
        // Verify the round-trip lands within 2.0 voxels: otherwise the point is
        // not on this viewer's current slice/surface view (SPEC §3.6).
        const cv::Vec3f back = chunked->sceneToVolume(scenePos);
        const double dist = cv::norm(back - vol);
        if (!std::isfinite(dist) || dist > 2.0) {
            QJsonObject data;
            data["point"] = vec3ToJson(vol);
            data["detail"] = QStringLiteral("point is not on this viewer's view (round-trip %1 voxels)")
                                 .arg(dist, 0, 'f', 3);
            throw AgentBridgeError{-32003, "Invalid coordinates", data};
        }
    } else {
        QJsonObject data;
        data["param"] = "space";
        data["value"] = space;
        throw AgentBridgeError{-32602, "space must be \"volume\" or \"scene\"", data};
    }

    // Synthesize the full click through the real mouse slots so all signal
    // wiring (sendVolumeClicked -> CWindow::onVolumeClicked, point placement,
    // tools) fires exactly as for a human click (SPEC §3.6).
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
    const QJsonObject p = paramsObject(params);
    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    if (!viewer->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const cv::Vec3f point = jsonToVec3(p.value("point"), "point");
    const bool forceRender = jsonOptionalBool(p, "forceRender", true);
    viewer->centerOnVolumePoint(point, forceRender);

    QJsonObject result;
    result["centered"] = true;
    result["viewerId"] = viewerIdFor(viewer);
    return result;
}


QJsonObject AgentBridgeServer::handleViewerZoom(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));

    if (!p.contains("factor")) {
        QJsonObject data;
        data["param"] = "factor";
        throw AgentBridgeError{-32602, "factor is required", data};
    }
    const double factor = jsonRequireNumber(p.value("factor"), "factor");
    if (!std::isfinite(factor) || factor <= 0.0) {
        QJsonObject data;
        data["param"] = "factor";
        data["value"] = factor;
        throw AgentBridgeError{-32602, "factor must be a positive finite number", data};
    }

    viewer->adjustZoomByFactor(static_cast<float>(factor));

    QJsonObject result;
    result["scale"] = viewer->getCurrentScale();
    return result;
}


QJsonObject AgentBridgeServer::handleViewerRotate(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);

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
        data["value"] = p.value("plane").toString();
        data["allowed"] = QJsonArray{"seg xz", "seg yz"};
        throw AgentBridgeError{-32602,
            "plane must be one of the rotatable axis-aligned planes: \"seg xz\", \"seg yz\"",
            data};
    }

    if (!p.contains("degrees")) {
        QJsonObject data;
        data["param"] = "degrees";
        throw AgentBridgeError{-32602, "degrees is required", data};
    }
    const double degrees = jsonRequireNumber(p.value("degrees"), "degrees");
    if (!std::isfinite(degrees)) {
        QJsonObject data;
        data["param"] = "degrees";
        data["value"] = degrees;
        throw AgentBridgeError{-32602, "degrees must be a finite number", data};
    }

    // Default to relative (delta) rotation — matches the middle-drag interaction.
    const bool relative = jsonOptionalBool(p, "relative", true);

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
    const QJsonObject p = paramsObject(params);

    AxisAlignedSliceController* slices =
        _window ? _window->_axisAlignedSliceController.get() : nullptr;
    if (!slices)
        throw AgentBridgeError{-32000, "Axis-aligned slice controller unavailable", {}};

    if (!p.contains(QStringLiteral("enabled")) || !p.value(QStringLiteral("enabled")).isBool()) {
        QJsonObject data;
        data["param"] = "enabled";
        throw AgentBridgeError{-32602, "enabled (bool) is required", data};
    }
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
// Segmentation editing + growth (SPEC §3.10-3.12)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentationEnableEditing(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    if (!p.contains("enabled")) {
        QJsonObject data;
        data["param"] = "enabled";
        throw AgentBridgeError{-32602, "enabled is required", data};
    }
    const bool enabled = jsonRequireBool(p.value("enabled"), "enabled");

    SegmentationWidget* widget = _window->_segmentationWidget;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "segmentation widget is not available";
        throw AgentBridgeError{-32009, "Segmentation editing unavailable", data};
    }

    // Enabling requires an active surface to edit (SPEC §3.10).
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
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    // Method enum (SPEC §3.11 as amended by §8.1).
    const QString methodStr = jsonOptionalString(p, "method", QStringLiteral("tracer"));
    // Footgun fix (§8.1): manual-add is an interactive editing mode, not a grow
    // invocation. Feeding SegmentationGrowthMethod::ManualAdd through
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
    const QString dirStr = jsonOptionalString(p, "direction", QStringLiteral("all"));
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
    else if (dirStr == QLatin1String("fill"))
        direction = SegmentationGrowthDirection::Fill;
    else {
        QJsonObject data;
        data["param"] = "direction";
        data["value"] = dirStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid direction: %1").arg(dirStr), data};
    }

    if (!p.contains("steps")) {
        QJsonObject data;
        data["param"] = "steps";
        throw AgentBridgeError{-32602, "steps is required", data};
    }
    const int steps = jsonRequireInt(p.value("steps"), "steps");
    if (steps < 1) {
        QJsonObject data;
        data["param"] = "steps";
        data["value"] = steps;
        throw AgentBridgeError{-32602, "steps must be >= 1", data};
    }

    const bool inpaintOnly = jsonOptionalBool(p, "inpaintOnly", false);

    // Growth is the "growth" source; a concurrent tool/lasagna/atlas job is fine
    // (§8.3), but a second growth job is not.
    requireSourceIdle(QStringLiteral("growth"));

    // The grower operates on the surface registered under the "segmentation"
    // slot; without it there is nothing to grow (SPEC §3.11 -> -32007).
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
    // "autosave"-source job (SPEC §3.11c). Params are ignored ({} accepted).
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
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    // GrowPatch runs the external vc_grow_seg_from_seed process: the "tool"
    // source (§8.3).
    requireSourceIdle(QStringLiteral("tool"));

    const cv::Vec3f seed = jsonToVec3(p.value("seed"), "seed");

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "GrowPatch unavailable", data};
    }

    SegmentationCommandHandler::GrowPatchSeedParams gp;
    gp.volumeId = jsonOptionalString(p, "volumeId");
    if (!gp.volumeId.isEmpty()) {
        const auto ids = state->vpkg()->volumeIDs();
        if (std::find(ids.begin(), ids.end(), gp.volumeId.toStdString()) == ids.end()) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = gp.volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(gp.volumeId), data};
        }
    }

    gp.iterations = jsonOptionalInt(p, "iterations", 200);
    if (gp.iterations < 1 || gp.iterations > 100000) {
        QJsonObject data;
        data["param"] = "iterations";
        data["value"] = gp.iterations;
        throw AgentBridgeError{-32602, "iterations must be in [1, 100000]", data};
    }
    // Present-but-malformed minAreaCm must reject, not silently fall back to the
    // 0.002 default (which .toDouble(0.002) would do for a wrong-typed value).
    gp.minAreaCm = p.contains("minAreaCm")
        ? jsonRequireFinite(p.value("minAreaCm"), "minAreaCm") : 0.002;
    if (gp.minAreaCm < 0.0) {
        QJsonObject data;
        data["param"] = "minAreaCm";
        throw AgentBridgeError{-32602, "minAreaCm must be a finite value >= 0", data};
    }
    gp.outputDir = jsonOptionalString(p, "outputDir");

    const QString effectiveVolumeId = gp.volumeId.isEmpty()
        ? QString::fromStdString(state->currentVolumeId())
        : gp.volumeId;

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("segmentation.grow_patch_from_seed"),
                                   QStringLiteral("GrowPatch from seed"),
                                   /*broadcastStart=*/false);

    // Suppress the interactive "Operation Complete" QMessageBox for this run so
    // the modal dialog cannot starve the toolFinished slots that transition the
    // job out of "running"; auto-cleared on toolFinished, also cleared below on
    // the synchronous-failure path.
    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    const QVector3D seedQ(seed[0], seed[1], seed[2]);
    QString err;
    if (!handler->startGrowPatchFromSeed(seedQ, gp, &err)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJobs.remove(QStringLiteral("tool"));
        // Map the distinct failure sentences to codes (SPEC §3.12, §4).
        if (err.contains(QLatin1String("Unknown volume id"))) {
            QJsonObject data;
            data["kind"] = "volume";
            data["detail"] = err;
            throw AgentBridgeError{-32007, err, data};
        }
        if (err.contains(QLatin1String("Could not find")) ||
            err.contains(QLatin1String("executable"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "vc_grow_seg_from_seed not found", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start GrowPatch from seed", data};
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
// Manual-add (hole-fill) + corrections point authoring (SPEC §9.2-9.7)
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
    // §9.2 preconditions: editing enabled, active edit session, no growth.
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
    const QJsonObject p = paramsObject(params);
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
    const bool apply = jsonOptionalBool(p, "apply", true);
    const bool applied = mod->setManualAddModeActive(false, apply);
    QJsonObject result;
    result["applied"] = applied;
    return result;
}


QJsonObject AgentBridgeServer::handleManualAddSetLineMode(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
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
    else {
        QJsonObject data;
        data["param"] = "mode";
        data["value"] = modeStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid mode: %1").arg(modeStr), data};
    }
    const ManualAddTool::LinePreviewMode effective = widget->setManualAddLinePreviewMode(mode);
    QJsonObject result;
    result["mode"] = linePreviewModeToString(effective);
    return result;
}


QJsonObject AgentBridgeServer::handleManualAddSetInterpolation(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
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
    else {
        QJsonObject data;
        data["param"] = "mode";
        data["value"] = modeStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid mode: %1").arg(modeStr), data};
    }
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
    const QJsonObject p = paramsObject(params);
    if (!p.contains("active")) {
        QJsonObject data;
        data["param"] = "active";
        throw AgentBridgeError{-32602, "active is required", data};
    }
    const bool active = jsonRequireBool(p.value("active"), "active");

    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!mod) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32000, "Segmentation module unavailable", data};
    }

    // Enabling enforces the same preconditions as the G-key handler (§9.7);
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


// NOTE: the deferred-response mechanism (beginDeferred / completeDeferredResult
// / completeDeferredError, plus the AgentBridgeDeferred dispatch path) is
// bridge-core infrastructure per SPEC §8.4, not yet wired to a shipping method
// (lasagna.* / atlas.* / fiber.create_atlas arrive later); timeout -> -32005.

// ---------------------------------------------------------------------------
// canvas.drag (SPEC §9.1)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleCanvasDrag(const QJsonValue& params)
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

    // button, incl. the "none" hover-only variant (§9.1).
    const QJsonValue btnv = p.value("button");
    const QString btnStr = btnv.isString() ? btnv.toString() : QStringLiteral("left");
    const bool buttonNone = (btnStr == QLatin1String("none"));
    Qt::MouseButton button = Qt::NoButton;
    if (!buttonNone)
        button = jsonToMouseButton(btnv);  // validates / throws -32602
    const Qt::KeyboardModifiers modifiers = jsonToModifiers(p.value("modifiers"));

    // steps: default 8; non-integer / < 1 -> -32602; > 256 clamped silently.
    // jsonRequireInt rejects wrong-typed/fractional/overflowing values (the last
    // guarding against a finite double like 1e300 overflowing the int cast, §11a).
    int steps = 8;
    if (p.contains("steps")) {
        steps = jsonRequireInt(p.value("steps"), "steps");
        if (steps < 1) {
            QJsonObject data;
            data["param"] = "steps";
            data["value"] = steps;
            throw AgentBridgeError{-32602, "steps must be >= 1", data};
        }
        if (steps > 256)
            steps = 256;
    }

    const QString space = jsonOptionalString(p, "space", QStringLiteral("volume"));

    // Convert one endpoint (Vec3 volume, or {x,y} scene) to a scene point,
    // reusing the §3.6 round-trip validation. `name` is "from" / "to".
    auto convertEndpoint = [&](const QJsonValue& v, const char* name) -> QPointF {
        if (space == QLatin1String("scene")) {
            if (!v.isObject()) {
                QJsonObject data;
                data["param"] = QString::fromLatin1(name);
                throw AgentBridgeError{-32602,
                    QStringLiteral("%1 must be a scene object {x, y}").arg(QLatin1String(name)),
                    data};
            }
            const QJsonObject o = v.toObject();
            return QPointF(jsonRequireFiniteFloat(o.value("x"), "x"),
                           jsonRequireFiniteFloat(o.value("y"), "y"));
        }
        if (space == QLatin1String("volume")) {
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
        }
        QJsonObject data;
        data["param"] = "space";
        data["value"] = space;
        throw AgentBridgeError{-32602, "space must be \"volume\" or \"scene\"", data};
    };

    const QPointF sceneFrom = convertEndpoint(p.value("from"), "from");
    const QPointF sceneTo = convertEndpoint(p.value("to"), "to");

    // Dispatch press -> steps x move -> release through the real mouse slots so
    // all signal wiring fires exactly as for a human drag (§9.1). For
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
