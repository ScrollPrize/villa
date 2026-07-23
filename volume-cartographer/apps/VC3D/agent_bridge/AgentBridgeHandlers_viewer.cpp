#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "CState.hpp"
#include "CWindow.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/render/Colormaps.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Compositing.hpp"

// ---------------------------------------------------------------------------
// Viewer round-2: overlay-volume controls and per-viewer intersection sets.
//
// `viewer.get_overlay` / `viewer.set_overlay` expose the ViewerManager-backed
// overlay volume (colormap, opacity, window, composite) that the GUI's
// overlay-volume combo box + sliders drive; `viewer.list_overlay_volumes`
// lists candidate volumes; `viewer.set_intersects` mirrors
// SurfacePanelController's per-viewer `setIntersects()` calls (the surfaces
// whose intersection lines a viewer draws).
// ---------------------------------------------------------------------------

namespace {

// The caller passes _window->_viewerManager.get() (a private member reachable
// only from AgentBridgeServer members, which are friends of CWindow); this
// helper just centralizes the null-check -> -32010 throw.
ViewerManager* requireViewerManager(ViewerManager* mgr)
{
    if (!mgr) {
        QJsonObject data;
        data["detail"] = "viewer manager is not available";
        throw AgentBridgeError{-32010, "Viewer manager unavailable", data};
    }
    return mgr;
}

// vc::resolve() (Colormaps.cpp) has no notion of "unknown" -- an unrecognized
// id silently falls back to the first spec ("fire") at render time, so a typo
// would be accepted and echoed back while quietly not doing what was asked.
// Validate against the same vc::specs() table the overlay panel populates its
// combo box from, rather than a hand-maintained guess. Empty string is the
// sentinel VolumeOverlayController uses for "no explicit choice yet" (see its
// resetToDefaults()), so it stays valid here too.
bool isKnownOverlayColormap(const QString& id)
{
    if (id.isEmpty())
        return true;
    const auto& specs = vc::specs();
    return std::any_of(specs.begin(), specs.end(), [&id](const vc::OverlayColormapSpec& s) {
        return QString::fromStdString(s.id) == id;
    });
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
    const QJsonObject p = paramsObject(params);

    // --- Phase 1: parse, validate, and resolve every supplied field into
    // locals. Nothing here mutates viewer state, so a malformed field rejects
    // the whole request (-32602 / -32007 / -32000) rather than switching the
    // overlay volume and then throwing on a later bad field, which would leave
    // the overlay half-updated. ---

    // Overlay volume selection: "clear" or an explicit empty "volumeId" both
    // mean "no overlay volume"; a non-empty "volumeId" resolves and sets it.
    // Absent entirely -> leave the current overlay volume untouched.
    const bool clear = jsonOptionalBool(p, "clear", false);
    const bool hasVolumeId = p.contains("volumeId");
    const QString volumeId = hasVolumeId ? jsonRequireString(p, "volumeId") : QString();
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
        try {
            overlayVolume = vpkg->volume(volumeId.toStdString());
        } catch (const std::out_of_range&) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(volumeId), data};
        }
    }

    const bool hasColormap = p.contains("colormap");
    QString colormap;
    if (hasColormap) {
        colormap = jsonRequireString(p, "colormap");
        if (!isKnownOverlayColormap(colormap))
            throwParamError("colormap", QStringLiteral("must be one of the known overlay colormap ids"));
    }

    const bool hasOpacity = p.contains("opacity");
    const float opacity = hasOpacity
        ? static_cast<float>(jsonRequireFinite(p.value("opacity"), "opacity")) : 0.0f;

    const bool hasThreshold = p.contains("threshold");
    const float threshold = hasThreshold
        ? static_cast<float>(jsonRequireFinite(p.value("threshold"), "threshold")) : 0.0f;

    const bool hasWindow = p.contains("window");
    float windowLow = 0.0f;
    float windowHigh = 0.0f;
    if (hasWindow) {
        const QJsonValue wv = p.value("window");
        if (!wv.isObject())
            throwParamError("window", QStringLiteral("must be an object {low, high}"));
        const QJsonObject wo = wv.toObject();
        if (!wo.contains("low") || !wo.contains("high"))
            throwParamError("window", QStringLiteral("requires low and high"));
        windowLow = static_cast<float>(jsonRequireFinite(wo.value("low"), "window.low"));
        windowHigh = static_cast<float>(jsonRequireFinite(wo.value("high"), "window.high"));
    }

    const bool hasMaxRes = p.contains("maxDisplayedResolution");
    const int maxRes = hasMaxRes
        ? jsonRequireInt(p.value("maxDisplayedResolution"), "maxDisplayedResolution") : 0;

    const bool hasComposite = p.contains("composite");
    OverlayCompositeSettings composite;
    if (hasComposite) {
        const QJsonValue cv = p.value("composite");
        if (!cv.isObject())
            throwParamError("composite", QStringLiteral("must be an object"));
        const QJsonObject co = cv.toObject();

        // Merge over the current settings so an agent can tweak a single
        // sub-key (e.g. just "enabled") without restating the rest. Reading the
        // current settings is not a mutation, so it stays in the validation phase.
        composite = mgr->overlayComposite();
        if (co.contains("enabled"))
            composite.enabled = jsonRequireBool(co.value("enabled"), "composite.enabled");
        if (co.contains("method")) {
            const QString method = jsonRequireString(co, "method");
            if (method != "max" && method != "mean" && method != "min")
                throwParamError("composite.method", QStringLiteral("must be one of max, mean, min"));
            composite.method = method.toStdString();
        }
        if (co.contains("layersFront"))
            composite.layersFront = jsonRequireInt(co.value("layersFront"), "composite.layersFront");
        if (co.contains("layersBehind"))
            composite.layersBehind = jsonRequireInt(co.value("layersBehind"), "composite.layersBehind");
    }

    // --- Phase 2: apply. The request is fully validated, so these setters run
    // as a group -- a rejected request above mutated nothing. ---
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
    const QJsonObject p = paramsObject(params);

    if (!p.value("surfaceIds").isArray())
        throwParamError("surfaceIds", QStringLiteral("must be an array of strings"));

    // The GUI's own filter application (SurfacePanelController::applyFilters)
    // always seeds the drawn set with "segmentation"; mirror that invariant
    // here rather than trusting the caller to include it.
    std::set<std::string> ids{"segmentation"};
    for (const QJsonValue& e : p.value("surfaceIds").toArray()) {
        if (!e.isString())
            throwParamError("surfaceIds", QStringLiteral("must be an array of strings"));
        ids.insert(e.toString().toStdString());
    }

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
