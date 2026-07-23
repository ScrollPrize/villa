#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <memory>
#include <set>
#include <string>

#include "CState.hpp"
#include "CWindow.hpp"
#include "ViewerManager.hpp"
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
