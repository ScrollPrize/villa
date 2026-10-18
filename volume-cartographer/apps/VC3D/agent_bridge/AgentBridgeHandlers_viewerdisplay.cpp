#include "agent_bridge/AgentBridgeServer.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

#include "CWindow.hpp"
#include "VolumeViewerCmaps.hpp"
#include "overlays/InkDetectionOverlayController.hpp"

// ===========================================================================
// Viewer display state beyond the shared render settings: the ink detection
// overlay shown over flattened segment views, whose setters drive the same
// controller as the Ink Detection panel.
// ===========================================================================

namespace {

InkDetectionOverlayController* requireInkOverlay(InkDetectionOverlayController* overlay)
{
    if (!overlay) {
        throw AgentBridgeError{
            -32010, "Ink detection overlay unavailable",
            QJsonObject{{"detail", "ink detection overlay controller is not available"}}};
    }
    return overlay;
}

// The colormaps the Ink Detection panel offers; "" is grayscale.
QStringList inkColormapIds()
{
    QStringList ids{QString()};
    for (const auto& entry : volume_viewer_cmaps::entries(volume_viewer_cmaps::EntryScope::SharedOnly))
        ids.append(QString::fromStdString(entry.id));
    return ids;
}

}  // namespace


// ---------------------------------------------------------------------------
// viewer.get_ink_overlay / viewer.set_ink_overlay
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::inkOverlayJson() const
{
    const InkDetectionOverlayController* overlay =
        requireInkOverlay(_window ? _window->_inkDetectionOverlay.get() : nullptr);

    QJsonArray available;
    for (const auto& option : overlay->options()) {
        available.append(QJsonObject{
            {"label", option.label},
            {"path", QString::fromStdString(option.localPath.string())},
            {"sampleId", QString::fromStdString(option.sampleId)},
            {"segmentId", QString::fromStdString(option.segmentId)},
            {"segmentLongId", QString::fromStdString(option.segmentLongId)},
        });
    }
    const QString selected = QString::fromStdString(overlay->selectedPath().string());
    const bool loaded = overlay->hasLoadedSelection();

    QJsonObject o;
    o["available"] = available;
    o["path"] = selected.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(selected);
    o["loaded"] = loaded;
    o["visible"] = loaded && overlay->opacity() > 0;
    o["opacity"] = overlay->opacity() / 100.0;
    o["colormap"] = QString::fromStdString(overlay->colormapId());
    o["colormaps"] = QJsonArray::fromStringList(inkColormapIds());
    o["singleChannel"] = loaded && overlay->selectedIsSingleChannel();
    o["flipHorizontal"] = overlay->horizontalFlip();
    o["flipVertical"] = overlay->verticalFlip();
    return o;
}

QJsonObject AgentBridgeServer::handleViewerGetInkOverlay(const QJsonValue& params)
{
    InkDetectionOverlayController* overlay =
        requireInkOverlay(_window ? _window->_inkDetectionOverlay.get() : nullptr);
    if (params.toObject().value("refresh").toBool(false))
        overlay->refreshAvailableDetections();
    return inkOverlayJson();
}

QJsonObject AgentBridgeServer::handleViewerSetInkOverlay(const QJsonValue& params)
{
    InkDetectionOverlayController* overlay =
        requireInkOverlay(_window ? _window->_inkDetectionOverlay.get() : nullptr);
    const QJsonObject p = params.toObject();

    // Validate every field before applying any of them.
    const bool hasPath = p.contains("path");
    std::filesystem::path path;
    if (hasPath && !p.value("path").isNull() && !p.value("path").toString().isEmpty()) {
        const QString requested = p.value("path").toString();
        const auto& options = overlay->options();
        const auto match = std::find_if(options.begin(), options.end(), [&](const auto& option) {
            return QString::fromStdString(option.localPath.string()) == requested;
        });
        if (match == options.end()) {
            QJsonArray available;
            for (const auto& option : options)
                available.append(QString::fromStdString(option.localPath.string()));
            throw AgentBridgeError{
                -32007, "Ink detection not available for the active segment",
                QJsonObject{{"kind", "ink_detection"}, {"id", requested},
                            {"available", available}}};
        }
        path = match->localPath;
    }
    const bool hasColormap = p.contains("colormap");
    const QString colormap = p.value("colormap").toString();
    if (hasColormap) {
        const QStringList allowed = inkColormapIds();
        if (!allowed.contains(colormap)) {
            throw AgentBridgeError{
                -32602, QStringLiteral("Unknown colormap: %1").arg(colormap),
                QJsonObject{{"param", "colormap"},
                            {"allowed", QJsonArray::fromStringList(allowed)}}};
        }
    }

    if (hasPath) {
        if (path.empty())
            overlay->clearSelection();
        else
            overlay->setSelectedPath(path);
    }
    if (hasColormap)
        overlay->setColormapId(colormap.toStdString());
    if (p.contains("opacity"))
        overlay->setOpacity(static_cast<int>(std::lround(p.value("opacity").toDouble() * 100.0)));
    if (p.contains("flipHorizontal"))
        overlay->setHorizontalFlip(p.value("flipHorizontal").toBool());
    if (p.contains("flipVertical"))
        overlay->setVerticalFlip(p.value("flipVertical").toBool());
    return inkOverlayJson();
}
