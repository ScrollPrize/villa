#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QSettings>
#include <QString>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "CState.hpp"
#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Compositing.hpp"

// ---------------------------------------------------------------------------
// ViewerManager-backed overlay controls and per-viewer intersection sets.
// ---------------------------------------------------------------------------

namespace {

// Centralizes the null-check -> -32010 throw for the viewer manager.
ViewerManager* requireViewerManager(ViewerManager* mgr)
{
    if (!mgr) {
        QJsonObject data;
        data["detail"] = "viewer manager is not available";
        throw AgentBridgeError{-32010, "Viewer manager unavailable", data};
    }
    return mgr;
}

QJsonObject overlaySettingsJson(ViewerManager* mgr)
{
    QJsonObject o;
    o["volumeId"] = QString::fromStdString(mgr->overlayVolumeId());
    o["colormap"] = QString::fromStdString(mgr->overlayColormap());
    o["opacity"] = mgr->overlayOpacity();
    o["threshold"] = mgr->overlayThreshold();
    o["windowLow"] = mgr->overlayWindowLow();
    o["windowHigh"] = mgr->overlayWindowHigh();
    o["maxDisplayedResolution"] = mgr->overlayMaxDisplayedResolution();

    const OverlayCompositeSettings& c = mgr->overlayComposite();
    QJsonObject composite;
    composite["enabled"] = c.enabled;
    composite["method"] = QString::fromStdString(c.method);
    composite["layersFront"] = c.layersFront;
    composite["layersBehind"] = c.layersBehind;
    o["composite"] = composite;
    return o;
}

} // namespace


QJsonObject AgentBridgeServer::handleViewerGetOverlay(const QJsonValue&)
{
    ViewerManager* mgr = requireViewerManager(_window ? _window->_viewerManager.get() : nullptr);
    return overlaySettingsJson(mgr);
}


QJsonObject AgentBridgeServer::handleViewerSetOverlay(const QJsonValue& params)
{
    ViewerManager* mgr = requireViewerManager(_window ? _window->_viewerManager.get() : nullptr);
    const QJsonObject p = params.toObject();

    // Mechanical input validation has already run from the method descriptor.
    // Resolve the remaining semantic preconditions into locals before mutating
    // anything, so a rejected request cannot leave the overlay half-updated.

    // Overlay volume selection: "clear" or an explicit empty "volumeId" both
    // mean "no overlay volume"; a non-empty "volumeId" resolves and sets it.
    // Absent entirely -> leave the current overlay volume untouched.
    const bool clear = p.value("clear").toBool(false);
    const bool hasVolumeId = p.contains("volumeId");
    const QString volumeId = hasVolumeId ? p.value("volumeId").toString() : QString();
    const bool clearVolume = clear || (hasVolumeId && volumeId.isEmpty());
    const bool setVolume = !clearVolume && hasVolumeId;
    // Resolve the overlay Volume shared_ptr up front so the -32007 unknown-volume
    // check happens in this validation phase, before any setter runs.
    std::shared_ptr<Volume> overlayVolume;
    if (setVolume) {
        CState* state = _window->_state;
        std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
        if (!state || !state->hasVpkg() || !vpkg)
            throw AgentBridgeError{-32000, "No volume package loaded", {}};
        overlayVolume = vpkg->volume(volumeId.toStdString());
        if (!overlayVolume) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(volumeId), data};
        }
    }

    const bool hasColormap = p.contains("colormap");
    const QString colormap =
        hasColormap ? p.value("colormap").toString() : QString();

    const bool hasOpacity = p.contains("opacity");
    const float opacity = hasOpacity
        ? static_cast<float>(p.value("opacity").toDouble()) : 0.0f;

    const bool hasThreshold = p.contains("threshold");
    const float threshold = hasThreshold
        ? static_cast<float>(p.value("threshold").toDouble()) : 0.0f;

    const bool hasWindow = p.contains("window");
    float windowLow = 0.0f;
    float windowHigh = 0.0f;
    if (hasWindow) {
        const QJsonObject window = p.value("window").toObject();
        windowLow = static_cast<float>(window.value("low").toDouble());
        windowHigh = static_cast<float>(window.value("high").toDouble());
    }

    const bool hasMaxRes = p.contains("maxDisplayedResolution");
    const int maxRes = hasMaxRes
        ? p.value("maxDisplayedResolution").toInt() : 0;

    const bool hasComposite = p.contains("composite");
    OverlayCompositeSettings composite;
    if (hasComposite) {
        const QJsonObject object = p.value("composite").toObject();

        // Merge over current settings so a caller can tweak one sub-key without
        // restating the rest (a read, so it stays in the validation phase).
        composite = mgr->overlayComposite();
        if (object.contains("enabled"))
            composite.enabled = object.value("enabled").toBool();
        if (object.contains("method"))
            composite.method = object.value("method").toString().toStdString();
        if (object.contains("layersFront"))
            composite.layersFront = object.value("layersFront").toInt();
        if (object.contains("layersBehind"))
            composite.layersBehind = object.value("layersBehind").toInt();
    }

    // Apply only after every semantic precondition has passed.
    if (clearVolume) {
        mgr->setOverlayVolume(nullptr, "");
    } else if (setVolume) {
        // ViewerManager re-validates the coordinate space against the base
        // volume and may silently null a mismatched selection; the echoed
        // overlayVolumeId reflects whatever actually stuck.
        mgr->setOverlayVolume(overlayVolume, volumeId.toStdString());
    }
    if (hasColormap)
        mgr->setOverlayColormap(colormap.toStdString());
    if (hasOpacity)
        mgr->setOverlayOpacity(opacity);
    if (hasThreshold)
        mgr->setOverlayThreshold(threshold);
    if (hasWindow)
        mgr->setOverlayWindow(windowLow, windowHigh);
    if (hasMaxRes)
        mgr->setOverlayMaxDisplayedResolution(maxRes);
    if (hasComposite)
        mgr->setOverlayComposite(composite);

    // Echo the resulting full overlay settings (unknown keys are ignored).
    return overlaySettingsJson(mgr);
}


QJsonObject AgentBridgeServer::handleViewerListOverlayVolumes(const QJsonValue&)
{
    ViewerManager* mgr = requireViewerManager(_window ? _window->_viewerManager.get() : nullptr);
    CState* state = _window->_state;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    // Not filtered by coordinate-space compatibility: setOverlayVolume()
    // re-validates and silently nulls a mismatched pick, so listing every
    // volume lets the agent see (and diagnose) that rejection itself.
    const std::string currentId = state->currentVolumeId();
    QJsonArray volumes;
    for (const auto& id : vpkg->volumeIDs()) {
        QJsonObject v;
        v["id"] = QString::fromStdString(id);
        v["current"] = (id == currentId);
        volumes.append(v);
    }

    QJsonObject result;
    result["volumes"] = volumes;
    result["overlayVolumeId"] = QString::fromStdString(mgr->overlayVolumeId());
    return result;
}


QJsonObject AgentBridgeServer::handleViewerSetIntersects(const QJsonValue& params)
{
    ViewerManager* mgr = requireViewerManager(_window ? _window->_viewerManager.get() : nullptr);
    const QJsonObject p = params.toObject();

    // The GUI's own filter application (SurfacePanelController::applyFilters)
    // always seeds the drawn set with "segmentation"; mirror that invariant
    // here rather than trusting the caller to include it.
    std::set<std::string> ids{"segmentation"};
    for (const QJsonValue& e : p.value("surfaceIds").toArray())
        ids.insert(e.toString().toStdString());

    QJsonArray appliedToViewers;
    if (p.contains("viewer") && !p.value("viewer").isNull()) {
        VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
        if (viewer->surfName() == "segmentation") {
            QJsonObject data;
            data["detail"] = "the segmentation viewer does not draw intersections against itself";
            throw AgentBridgeError{-32009, "Cannot set intersects on the segmentation viewer", data};
        }
        viewer->setIntersects(ids);
        appliedToViewers.append(viewerIdFor(viewer));
    } else {
        // No target: broadcast to every base viewer except "segmentation"
        // itself (SurfacePanelController::applyFilters' no-filter path).
        mgr->forEachBaseViewer([&ids, &appliedToViewers, this](VolumeViewerBase* v) {
            if (!v || v->surfName() == "segmentation")
                return;
            v->setIntersects(ids);
            appliedToViewers.append(viewerIdFor(v));
        });
    }

    QJsonArray surfaceIds;
    for (const auto& id : ids)
        surfaceIds.append(QString::fromStdString(id));

    QJsonObject result;
    result["surfaceIds"] = surfaceIds;
    result["appliedToViewers"] = appliedToViewers;
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

    const QJsonObject p = params.toObject();

    // Mechanical input validation has already run from the method descriptor.
    // Normalize into locals before any setter or QSettings write so the update
    // remains atomic.
    const bool hasIntersectionOpacity = p.contains("intersectionOpacity");
    float intersectionOpacity = 0.0f;
    if (hasIntersectionOpacity) {
        const double v = p.value("intersectionOpacity").toDouble();
        intersectionOpacity = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }

    const bool hasIntersectionThickness = p.contains("intersectionThickness");
    float intersectionThickness = 0.0f;
    if (hasIntersectionThickness) {
        const double v = p.value("intersectionThickness").toDouble();
        intersectionThickness = static_cast<float>(std::max(0.0, v));
    }

    const bool hasOverlayOpacity = p.contains("overlayOpacity");
    float overlayOpacity = 0.0f;
    if (hasOverlayOpacity) {
        const double v = p.value("overlayOpacity").toDouble();
        overlayOpacity = static_cast<float>(std::clamp(v, 0.0, 1.0));
    }

    const bool hasIntersectionMaxSurfaces = p.contains("intersectionMaxSurfaces");
    int intersectionMaxSurfaces = 0;
    if (hasIntersectionMaxSurfaces) {
        const int v = p.value("intersectionMaxSurfaces").toInt();
        intersectionMaxSurfaces = std::max(0, v);
    }

    const bool hasHighlightedSurfaceIds = p.contains("highlightedSurfaceIds");
    std::vector<std::string> highlightedSurfaceIds;
    if (hasHighlightedSurfaceIds) {
        for (const QJsonValue& e : p.value("highlightedSurfaceIds").toArray())
            highlightedSurfaceIds.push_back(e.toString().toStdString());
    }

    const bool hasVolumeWindow = p.contains("volumeWindow");
    float volumeWindowLow = 0.0f;
    float volumeWindowHigh = 0.0f;
    if (hasVolumeWindow) {
        const QJsonObject window = p.value("volumeWindow").toObject();
        volumeWindowLow = static_cast<float>(window.value("low").toDouble());
        volumeWindowHigh = static_cast<float>(window.value("high").toDouble());
    }

    const bool hasSamplingStride = p.contains("samplingStride");
    const int samplingStride = hasSamplingStride
        ? p.value("samplingStride").toInt() : 0;

    const bool hasZScrollSensitivity = p.contains("zScrollSensitivity");
    const double zScrollSensitivity = hasZScrollSensitivity
        ? p.value("zScrollSensitivity").toDouble() : 0.0;

    const bool hasSegmentationCursorMirroring = p.contains("segmentationCursorMirroring");
    const bool segmentationCursorMirroring = hasSegmentationCursorMirroring
        ? p.value("segmentationCursorMirroring").toBool() : false;

    const bool hasPlaneIntersectionLinesVisible = p.contains("planeIntersectionLinesVisible");
    const bool planeIntersectionLinesVisible = hasPlaneIntersectionLinesVisible
        ? p.value("planeIntersectionLinesVisible").toBool() : false;

    const bool hasShowSurfaceNormals = p.contains("showSurfaceNormals");
    const bool showSurfaceNormals = hasShowSurfaceNormals
        ? p.value("showSurfaceNormals").toBool() : false;

    const bool hasShowDirectionHints = p.contains("showDirectionHints");
    const bool showDirectionHints = hasShowDirectionHints
        ? p.value("showDirectionHints").toBool() : false;

    const bool hasSurfaceOverlayEnabled = p.contains("surfaceOverlayEnabled");
    const bool surfaceOverlayEnabled = hasSurfaceOverlayEnabled
        ? p.value("surfaceOverlayEnabled").toBool() : false;

    const bool hasNormalArrowLengthScale = p.contains("normalArrowLengthScale");
    float normalArrowLengthScale = 0.0f;
    if (hasNormalArrowLengthScale) {
        const double v = p.value("normalArrowLengthScale").toDouble();
        // Clamp to the GUI slider's range (sliderNormalArrowLength: 10-200%) rather
        // than passing a negative/huge scale straight to the renderer.
        normalArrowLengthScale = static_cast<float>(std::clamp(v, 0.1, 2.0));
    }

    const bool hasNormalMaxArrows = p.contains("normalMaxArrows");
    int normalMaxArrows = 0;
    if (hasNormalMaxArrows) {
        const int v = p.value("normalMaxArrows").toInt();
        // Clamp to the GUI slider's range (sliderNormalMaxArrows: 4-100).
        normalMaxArrows = std::clamp(v, 4, 100);
    }

    // Apply as a group. Global controls go via ViewerManager (broadcast +
    // QSettings-persisted).
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
