#include "agent_bridge/AgentBridgeServer.hpp"

#include <QBuffer>
#include <QByteArray>
#include <QCoreApplication>
#include <QDateTime>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMdiSubWindow>
#include <QPixmap>
#include <QPointF>
#include <QVector3D>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <unordered_set>

#include "CWindow.hpp"
#include "CState.hpp"
#include "MenuActionController.hpp"
#include "OpenDataManifest.hpp"
#include "SegmentationCommandHandler.hpp"
#include "CommandLineToolRunner.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"

namespace {

QJsonObject vec3ToJson(const cv::Vec3f& v)
{
    QJsonObject o;
    o["x"] = static_cast<double>(v[0]);
    o["y"] = static_cast<double>(v[1]);
    o["z"] = static_cast<double>(v[2]);
    return o;
}

// Extracts an object from a JSON-RPC params value that may be absent or null.
QJsonObject paramsObject(const QJsonValue& params)
{
    if (params.isObject())
        return params.toObject();
    return QJsonObject();
}

// Parses a {x,y,z} object into a cv::Vec3f, throwing -32602 on a missing field
// or a non-finite component. `paramName` names the offending param in data.
cv::Vec3f jsonToVec3(const QJsonValue& value, const char* paramName)
{
    if (!value.isObject()) {
        QJsonObject data;
        data["param"] = QString::fromLatin1(paramName);
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 must be an object {x, y, z}").arg(QLatin1String(paramName)), data};
    }
    const QJsonObject o = value.toObject();
    if (!o.contains("x") || !o.contains("y") || !o.contains("z")) {
        QJsonObject data;
        data["param"] = QString::fromLatin1(paramName);
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 requires x, y and z").arg(QLatin1String(paramName)), data};
    }
    const double x = o.value("x").toDouble();
    const double y = o.value("y").toDouble();
    const double z = o.value("z").toDouble();
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        QJsonObject data;
        data["param"] = QString::fromLatin1(paramName);
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 has non-finite coordinates").arg(QLatin1String(paramName)), data};
    }
    return cv::Vec3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

// JSON button string -> Qt::MouseButton (SPEC §2.3). Default "left".
Qt::MouseButton jsonToMouseButton(const QJsonValue& value)
{
    const QString s = value.isString() ? value.toString() : QStringLiteral("left");
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

// JSON modifier-string array -> ORed Qt::KeyboardModifiers (SPEC §2.3). An
// absent/null value means no modifiers; anything else must be an array.
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

AgentBridgeServer::AgentBridgeServer(CWindow* window, QObject* parent)
    : QObject(parent), _window(window)
{
    registerHandlers();
    seedViewerRegistry();

    if (_window && _window->_viewerManager) {
        ViewerManager* vm = _window->_viewerManager.get();
        connect(vm, &ViewerManager::baseViewerCreated,
                this, [this](VolumeViewerBase* v) { registerViewer(v); });
        connect(vm, &ViewerManager::baseViewerClosing,
                this, [this](VolumeViewerBase* v) { unregisterViewer(v); });
    }

    subscribeJobSignals();
}

AgentBridgeServer::~AgentBridgeServer() = default;

bool AgentBridgeServer::listen(const QString& serverName)
{
    if (!_server) {
        _server = new QLocalServer(this);
        connect(_server, &QLocalServer::newConnection,
                this, &AgentBridgeServer::onNewConnection);
    }

    if (_server->listen(serverName))
        return true;

    // Stale socket file from a crashed run: remove once and retry (SPEC §1.1).
    QLocalServer::removeServer(serverName);
    return _server->listen(serverName);
}

QString AgentBridgeServer::serverName() const
{
    return _server ? _server->serverName() : QString();
}

QString AgentBridgeServer::fullServerName() const
{
    return _server ? _server->fullServerName() : QString();
}

// ---------------------------------------------------------------------------
// Connection + framing
// ---------------------------------------------------------------------------

void AgentBridgeServer::onNewConnection()
{
    while (_server && _server->hasPendingConnections()) {
        QLocalSocket* socket = _server->nextPendingConnection();
        if (!socket)
            break;
        _buffers.insert(socket, QByteArray());
        connect(socket, &QLocalSocket::readyRead,
                this, &AgentBridgeServer::onSocketReadyRead);
        connect(socket, &QLocalSocket::disconnected,
                this, &AgentBridgeServer::onSocketDisconnected);
    }
}

void AgentBridgeServer::onSocketReadyRead()
{
    auto* socket = qobject_cast<QLocalSocket*>(sender());
    if (!socket)
        return;

    QByteArray& buffer = _buffers[socket];
    buffer.append(socket->readAll());

    int newlineIdx;
    while ((newlineIdx = buffer.indexOf('\n')) != -1) {
        QByteArray line = buffer.left(newlineIdx);
        buffer.remove(0, newlineIdx + 1);
        if (line.endsWith('\r'))
            line.chop(1);
        if (line.trimmed().isEmpty())
            continue;
        processLine(socket, line);
    }
}

void AgentBridgeServer::onSocketDisconnected()
{
    auto* socket = qobject_cast<QLocalSocket*>(sender());
    if (!socket)
        return;
    _buffers.remove(socket);
    socket->deleteLater();
}

void AgentBridgeServer::processLine(QLocalSocket* socket, const QByteArray& line)
{
    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(line, &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        sendError(socket, QJsonValue(QJsonValue::Null), -32700, "Parse error");
        return;
    }

    if (doc.isArray()) {
        // Batch requests are not supported (SPEC §1.2).
        sendError(socket, QJsonValue(QJsonValue::Null), -32600,
                  "Invalid Request: batch requests are not supported");
        return;
    }

    if (!doc.isObject()) {
        sendError(socket, QJsonValue(QJsonValue::Null), -32600,
                  "Invalid Request: expected a JSON-RPC 2.0 object");
        return;
    }

    dispatch(socket, doc.object());
}

void AgentBridgeServer::dispatch(QLocalSocket* socket, const QJsonObject& request)
{
    const QJsonValue id = request.value("id");
    const bool isNotification = !request.contains("id");

    if (request.value("jsonrpc").toString() != QLatin1String("2.0")) {
        if (!isNotification)
            sendError(socket, id, -32600, "Invalid Request: jsonrpc must be \"2.0\"");
        return;
    }

    const QString method = request.value("method").toString();
    if (method.isEmpty()) {
        if (!isNotification)
            sendError(socket, id, -32600, "Invalid Request: missing method");
        return;
    }

    auto it = _handlers.find(method);
    if (it == _handlers.end()) {
        if (!isNotification)
            sendError(socket, id, -32601, QStringLiteral("Method not found: %1").arg(method));
        return;
    }

    QJsonObject result;
    try {
        result = it.value()(request.value("params"));
    } catch (const AgentBridgeError& e) {
        if (!isNotification)
            sendError(socket, id, e.code, e.message, e.data);
        return;
    } catch (const std::exception& e) {
        if (!isNotification) {
            QJsonObject data;
            data["detail"] = QString::fromUtf8(e.what());
            sendError(socket, id, -32010, "Internal error", data);
        }
        return;
    } catch (...) {
        if (!isNotification)
            sendError(socket, id, -32010, "Internal error");
        return;
    }

    if (!isNotification)
        sendResponse(socket, id, result);
}

void AgentBridgeServer::sendResponse(QLocalSocket* socket, const QJsonValue& id,
                                     const QJsonObject& result)
{
    QJsonObject message;
    message["jsonrpc"] = "2.0";
    message["id"] = id;
    message["result"] = result;
    writeMessage(socket, message);
}

void AgentBridgeServer::sendError(QLocalSocket* socket, const QJsonValue& id, int code,
                                  const QString& message, const QJsonObject& data)
{
    QJsonObject error;
    error["code"] = code;
    error["message"] = message;
    if (!data.isEmpty())
        error["data"] = data;

    QJsonObject envelope;
    envelope["jsonrpc"] = "2.0";
    envelope["id"] = id;
    envelope["error"] = error;
    writeMessage(socket, envelope);
}

void AgentBridgeServer::writeMessage(QLocalSocket* socket, const QJsonObject& message)
{
    if (!socket || socket->state() != QLocalSocket::ConnectedState)
        return;
    QByteArray payload = QJsonDocument(message).toJson(QJsonDocument::Compact);
    payload.append('\n');
    socket->write(payload);
    socket->flush();
}

void AgentBridgeServer::broadcastNotification(const QString& method, const QJsonObject& params)
{
    QJsonObject message;
    message["jsonrpc"] = "2.0";
    message["method"] = method;
    message["params"] = params;
    for (auto it = _buffers.begin(); it != _buffers.end(); ++it)
        writeMessage(it.key(), message);
}

// ---------------------------------------------------------------------------
// Viewer registry (SPEC §2.2)
// ---------------------------------------------------------------------------

void AgentBridgeServer::seedViewerRegistry()
{
    if (!_window || !_window->_viewerManager)
        return;
    for (VolumeViewerBase* v : _window->_viewerManager->baseViewers())
        registerViewer(v);
}

void AgentBridgeServer::registerViewer(VolumeViewerBase* viewer)
{
    if (!viewer)
        return;
    for (const ViewerEntry& e : _viewers) {
        if (e.viewer == viewer)
            return; // already registered
    }
    _viewers.push_back({QStringLiteral("v%1").arg(_nextViewerNum++), viewer});
}

void AgentBridgeServer::unregisterViewer(VolumeViewerBase* viewer)
{
    for (auto it = _viewers.begin(); it != _viewers.end(); ++it) {
        if (it->viewer == viewer) {
            _viewers.erase(it);
            return;
        }
    }
}

QString AgentBridgeServer::viewerIdFor(VolumeViewerBase* viewer) const
{
    for (const ViewerEntry& e : _viewers) {
        if (e.viewer == viewer)
            return e.id;
    }
    return QString();
}

QString AgentBridgeServer::viewerTitle(VolumeViewerBase* viewer) const
{
    // Best effort: viewers live inside a QMdiSubWindow whose title is the human
    // label. Walk up the widget parent chain to find it; fall back to the slot
    // name. (VolumeViewerBase is not a QWidget, so cross-cast to reach one.)
    auto* widget = dynamic_cast<QWidget*>(viewer);
    for (QWidget* w = widget; w; w = w->parentWidget()) {
        if (auto* sub = qobject_cast<QMdiSubWindow*>(w)) {
            const QString title = sub->windowTitle();
            if (!title.isEmpty())
                return title;
        }
    }
    return QString::fromStdString(viewer->surfName());
}

VolumeViewerBase* AgentBridgeServer::resolveViewer(const QJsonValue& ref,
                                                   const QString& defaultSlot) const
{
    QString key = ref.isString() ? ref.toString() : QString();
    if (key.isEmpty())
        key = defaultSlot;

    // Rule 1: exact registry id.
    for (const ViewerEntry& e : _viewers) {
        if (e.id == key)
            return e.viewer;
    }

    // Rule 2: match against surface-slot name.
    std::vector<const ViewerEntry*> matches;
    for (const ViewerEntry& e : _viewers) {
        if (QString::fromStdString(e.viewer->surfName()) == key)
            matches.push_back(&e);
    }

    if (matches.size() == 1)
        return matches.front()->viewer;

    QJsonObject data;
    data["viewer"] = key;
    if (matches.size() > 1) {
        QJsonArray candidates;
        for (const ViewerEntry* e : matches) {
            QJsonObject c;
            c["viewerId"] = e->id;
            c["surfName"] = QString::fromStdString(e->viewer->surfName());
            c["title"] = viewerTitle(e->viewer);
            candidates.push_back(c);
        }
        data["candidates"] = candidates;
        throw AgentBridgeError{-32002,
            QStringLiteral("Ambiguous viewer: %1 live viewers share slot \"%2\"")
                .arg(matches.size()).arg(key),
            data};
    }

    data["detail"] = QStringLiteral("no viewer matches id or slot \"%1\"").arg(key);
    throw AgentBridgeError{-32002,
        QStringLiteral("Invalid viewer: %1").arg(key), data};
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

void AgentBridgeServer::registerHandlers()
{
    _handlers.insert("ping",
        [this](const QJsonValue& p) { return handlePing(p); });
    _handlers.insert("state.get",
        [this](const QJsonValue& p) { return handleStateGet(p); });
    _handlers.insert("segments.list",
        [this](const QJsonValue& p) { return handleSegmentsList(p); });
    _handlers.insert("screenshot.capture",
        [this](const QJsonValue& p) { return handleScreenshotCapture(p); });
    _handlers.insert("canvas.get_cursor_volume_point",
        [this](const QJsonValue& p) { return handleCursorVolumePoint(p); });

    // --- Phase 2: canvas + mutating actions ---
    _handlers.insert("canvas.click",
        [this](const QJsonValue& p) { return handleCanvasClick(p, /*addShift=*/false); });
    _handlers.insert("canvas.shift_click",
        [this](const QJsonValue& p) { return handleCanvasClick(p, /*addShift=*/true); });
    _handlers.insert("viewer.center_on_point",
        [this](const QJsonValue& p) { return handleViewerCenterOnPoint(p); });
    _handlers.insert("viewer.zoom",
        [this](const QJsonValue& p) { return handleViewerZoom(p); });
    _handlers.insert("segmentation.enable_editing",
        [this](const QJsonValue& p) { return handleSegmentationEnableEditing(p); });
    _handlers.insert("segmentation.grow",
        [this](const QJsonValue& p) { return handleSegmentationGrow(p); });
    _handlers.insert("segmentation.grow_patch_from_seed",
        [this](const QJsonValue& p) { return handleSegmentationGrowPatchFromSeed(p); });
    _handlers.insert("points.commit",
        [this](const QJsonValue& p) { return handlePointsCommit(p); });
    _handlers.insert("points.list",
        [this](const QJsonValue& p) { return handlePointsList(p); });
    _handlers.insert("volume.open",
        [this](const QJsonValue& p) { return handleVolumeOpen(p); });
    _handlers.insert("catalog.open_sample",
        [this](const QJsonValue& p) { return handleCatalogOpenSample(p); });
    _handlers.insert("job.status",
        [this](const QJsonValue& p) { return handleJobStatus(p); });
}

QJsonObject AgentBridgeServer::handlePing(const QJsonValue&)
{
    QJsonObject result;
    result["pong"] = true;
    result["pid"] = static_cast<double>(QCoreApplication::applicationPid());
    result["version"] = QCoreApplication::applicationVersion();
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

    // Active job (SPEC §3.2). At most one job runs at a time (§2.4).
    if (_activeJob && _activeJob->state == QLatin1String("running")) {
        QJsonObject job;
        job["jobId"] = _activeJob->id;
        job["kind"] = _activeJob->kind;
        job["label"] = _activeJob->label;
        job["running"] = true;
        result["job"] = job;
    } else {
        result["job"] = QJsonValue::Null;
    }

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

QJsonObject AgentBridgeServer::handleScreenshotCapture(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString target = p.value("target").toString(QStringLiteral("window"));

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

    QPixmap pixmap = widget->grab();

    if (p.contains("maxDim")) {
        const int maxDim = p.value("maxDim").toInt(0);
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
    if (p.contains("scene") && p.value("scene").isObject()) {
        const QJsonObject s = p.value("scene").toObject();
        scenePos = QPointF(s.value("x").toDouble(), s.value("y").toDouble());
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

    const QString space = p.value("space").toString(QStringLiteral("volume"));
    QPointF scenePos;
    if (space == QLatin1String("scene")) {
        const QJsonValue posv = p.value("position");
        if (!posv.isObject()) {
            QJsonObject data;
            data["param"] = "position";
            throw AgentBridgeError{-32602, "scene-space position must be an object {x, y}", data};
        }
        const QJsonObject po = posv.toObject();
        scenePos = QPointF(po.value("x").toDouble(), po.value("y").toDouble());
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
    const bool forceRender = p.value("forceRender").toBool(true);
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
    const double factor = p.value("factor").toDouble();
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
    const bool enabled = p.value("enabled").toBool();

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

    // Method enum (SPEC §3.11).
    const QString methodStr = p.value("method").toString(QStringLiteral("tracer"));
    SegmentationGrowthMethod method;
    if (methodStr == QLatin1String("tracer"))
        method = SegmentationGrowthMethod::Tracer;
    else if (methodStr == QLatin1String("corrections"))
        method = SegmentationGrowthMethod::Corrections;
    else if (methodStr == QLatin1String("patch_tracer"))
        method = SegmentationGrowthMethod::PatchTracer;
    else if (methodStr == QLatin1String("manual_add"))
        method = SegmentationGrowthMethod::ManualAdd;
    else {
        QJsonObject data;
        data["param"] = "method";
        data["value"] = methodStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid method: %1").arg(methodStr), data};
    }

    // Direction enum.
    const QString dirStr = p.value("direction").toString(QStringLiteral("all"));
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
    const int steps = p.value("steps").toInt(0);
    if (steps < 1) {
        QJsonObject data;
        data["param"] = "steps";
        data["value"] = steps;
        throw AgentBridgeError{-32602, "steps must be >= 1", data};
    }

    const bool inpaintOnly = p.value("inpaintOnly").toBool(false);

    if (jobIsRunning()) {
        QJsonObject data;
        const QString jid = activeJobId();
        if (!jid.isEmpty())
            data["jobId"] = jid;
        throw AgentBridgeError{-32004, "A job is already running", data};
    }

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
    const QString jobId = beginJob(QStringLiteral("segmentation.grow"),
                                   QStringLiteral("Grow %1 (%2, %3 steps)")
                                       .arg(methodStr, dirStr).arg(steps),
                                   /*broadcastStart=*/false);

    _window->onGrowSegmentationSurface(method, direction, steps, inpaintOnly);

    const bool started = _window->_segmentationGrower && _window->_segmentationGrower->running();
    if (!started) {
        // Growth was rejected synchronously (e.g. invalid custom params). The
        // false status signal may already have cleared _activeJob.
        _activeJob.reset();
        QJsonObject data;
        data["detail"] = "segmentation growth did not start";
        throw AgentBridgeError{-32005, "Failed to start segmentation growth", data};
    }

    if (_activeJob)
        broadcastJobProgress(*_activeJob, QStringLiteral("started"));

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segmentation.grow";
    return result;
}

QJsonObject AgentBridgeServer::handleSegmentationGrowPatchFromSeed(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    if (jobIsRunning()) {
        QJsonObject data;
        const QString jid = activeJobId();
        if (!jid.isEmpty())
            data["jobId"] = jid;
        throw AgentBridgeError{-32004, "A job is already running", data};
    }

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

    gp.iterations = p.contains("iterations") ? p.value("iterations").toInt(200) : 200;
    if (gp.iterations < 1 || gp.iterations > 100000) {
        QJsonObject data;
        data["param"] = "iterations";
        data["value"] = gp.iterations;
        throw AgentBridgeError{-32602, "iterations must be in [1, 100000]", data};
    }
    gp.minAreaCm = p.contains("minAreaCm") ? p.value("minAreaCm").toDouble(0.002) : 0.002;
    if (!std::isfinite(gp.minAreaCm) || gp.minAreaCm < 0.0) {
        QJsonObject data;
        data["param"] = "minAreaCm";
        throw AgentBridgeError{-32602, "minAreaCm must be a finite value >= 0", data};
    }
    gp.outputDir = p.value("outputDir").toString();

    const QString effectiveVolumeId = gp.volumeId.isEmpty()
        ? QString::fromStdString(state->currentVolumeId())
        : gp.volumeId;

    const QString jobId = beginJob(QStringLiteral("segmentation.grow_patch_from_seed"),
                                   QStringLiteral("GrowPatch from seed"),
                                   /*broadcastStart=*/false);

    // This job is bridge-tracked and typically runs headless/offscreen. Tell
    // the runner to suppress the interactive "Operation Complete" QMessageBox
    // for this run so the modal dialog cannot starve the toolFinished slots
    // that transition the job out of "running" (see CommandLineToolRunner
    // ::setSuppressCompletionDialogs). The flag is auto-cleared once
    // toolFinished fires; we also clear it on the synchronous-failure path.
    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    const QVector3D seedQ(seed[0], seed[1], seed[2]);
    QString err;
    if (!handler->startGrowPatchFromSeed(seedQ, gp, &err)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJob.reset();
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
    if (_activeJob) {
        _activeJob->outputPath = outputDir;
        broadcastJobProgress(*_activeJob, QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "segmentation.grow_patch_from_seed";
    result["outputDir"] = outputDir;
    result["volumeId"] = effectiveVolumeId;
    return result;
}

// ---------------------------------------------------------------------------
// Point collections (SPEC §3.13-3.14)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handlePointsCommit(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString collection = p.value("collection").toString();
    if (collection.isEmpty()) {
        QJsonObject data;
        data["param"] = "collection";
        throw AgentBridgeError{-32602, "collection name is required", data};
    }

    const QJsonValue pointsv = p.value("points");
    if (!pointsv.isArray() || pointsv.toArray().isEmpty()) {
        QJsonObject data;
        data["param"] = "points";
        throw AgentBridgeError{-32602, "points must be a non-empty array", data};
    }

    std::vector<cv::Vec3f> pts;
    for (const QJsonValue& pv : pointsv.toArray())
        pts.push_back(jsonToVec3(pv, "points"));  // validates finiteness

    std::optional<double> winding;
    if (p.contains("winding")) {
        const double w = p.value("winding").toDouble();
        if (!std::isfinite(w)) {
            QJsonObject data;
            data["param"] = "winding";
            throw AgentBridgeError{-32602, "winding must be finite", data};
        }
        winding = w;
    }

    VCCollection* pc = state->pointCollection();
    if (!pc) {
        QJsonObject data;
        data["detail"] = "point collection store is unavailable";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    const std::string col = collection.toStdString();
    QJsonArray pointIds;
    for (const cv::Vec3f& v : pts) {
        ColPoint cp = pc->addPoint(col, v);
        if (winding) {
            cp.winding_annotation = static_cast<float>(*winding);
            pc->updatePoint(cp);
        }
        pointIds.append(static_cast<double>(cp.id));
    }

    QJsonObject result;
    result["collectionId"] = static_cast<double>(pc->getCollectionId(col));
    result["pointIds"] = pointIds;
    return result;
}

QJsonObject AgentBridgeServer::handlePointsList(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = state ? state->pointCollection() : nullptr;

    const QString filter = p.value("collection").toString();

    QJsonArray collections;
    if (pc) {
        if (!filter.isEmpty() && pc->getCollectionId(filter.toStdString()) == 0) {
            QJsonObject data;
            data["kind"] = "collection";
            data["id"] = filter;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown collection: %1").arg(filter), data};
        }

        for (const auto& [id, coll] : pc->getAllCollections()) {
            if (!filter.isEmpty() && coll.name != filter.toStdString())
                continue;

            QJsonObject c;
            c["id"] = static_cast<double>(id);
            c["name"] = QString::fromStdString(coll.name);
            QJsonArray color;
            color.append(coll.color[0]);
            color.append(coll.color[1]);
            color.append(coll.color[2]);
            c["color"] = color;

            QJsonArray pointsArr;
            for (const auto& [pid, cp] : coll.points) {
                QJsonObject po;
                po["id"] = static_cast<double>(pid);
                po["position"] = vec3ToJson(cp.p);
                po["winding"] = std::isnan(cp.winding_annotation)
                                    ? QJsonValue(QJsonValue::Null)
                                    : QJsonValue(static_cast<double>(cp.winding_annotation));
                pointsArr.append(po);
            }
            c["points"] = pointsArr;
            collections.append(c);
        }
    } else if (!filter.isEmpty()) {
        QJsonObject data;
        data["kind"] = "collection";
        data["id"] = filter;
        throw AgentBridgeError{-32007, QStringLiteral("Unknown collection: %1").arg(filter), data};
    }

    QJsonObject result;
    result["collections"] = collections;
    return result;
}

// ---------------------------------------------------------------------------
// Project/catalog opening (SPEC §3.15-3.16)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleVolumeOpen(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString path = p.value("path").toString();
    if (path.isEmpty()) {
        QJsonObject data;
        data["detail"] = "path is required";
        throw AgentBridgeError{-32005, "No volume package path", data};
    }

    MenuActionController* mc = _window ? _window->_menuController.get() : nullptr;
    if (!mc) {
        QJsonObject data;
        data["detail"] = "menu controller is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    mc->openVolpkgAt(path);

    CState* state = _window->_state;
    if (!state || !state->hasVpkg()) {
        QJsonObject data;
        data["detail"] = QStringLiteral("failed to open volume package at %1").arg(path);
        throw AgentBridgeError{-32005, "Volume package load failed", data};
    }

    auto vpkg = state->vpkg();
    const auto ids = vpkg->volumeIDs();

    const QString volumeId = p.value("volumeId").toString();
    if (!volumeId.isEmpty()) {
        if (std::find(ids.begin(), ids.end(), volumeId.toStdString()) == ids.end()) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(volumeId), data};
        }
        try {
            _window->setVolume(vpkg->volume(volumeId.toStdString()));
            _window->syncVolumeSelectionControls(volumeId);
        } catch (const AgentBridgeError&) {
            throw;
        } catch (const std::exception& e) {
            QJsonObject data;
            data["detail"] = QString::fromUtf8(e.what());
            throw AgentBridgeError{-32005, "Failed to switch volume", data};
        }
    }

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

QJsonObject AgentBridgeServer::handleCatalogOpenSample(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString sampleId = p.value("sampleId").toString();
    if (sampleId.isEmpty()) {
        QJsonObject data;
        data["param"] = "sampleId";
        throw AgentBridgeError{-32602, "sampleId is required", data};
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
    if (!manifest->findSample(sampleId.toStdString())) {
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

    if (!mc->openOpenDataSampleById(sampleId)) {
        QJsonObject data;
        data["detail"] = QStringLiteral("failed to open open-data sample %1").arg(sampleId);
        throw AgentBridgeError{-32005, "Open Data sample open failed", data};
    }

    CState* state = _window->_state;
    QJsonObject result;
    result["opened"] = true;
    result["sampleId"] = sampleId;
    result["vpkgPath"] = (state && state->hasVpkg()) ? QJsonValue(state->vpkgPath())
                                                     : QJsonValue(QString());
    QJsonArray idArr;
    if (state && state->vpkg()) {
        for (const auto& id : state->vpkg()->volumeIDs())
            idArr.append(QString::fromStdString(id));
    }
    result["volumeIds"] = idArr;
    return result;
}

// ---------------------------------------------------------------------------
// Job tracking (SPEC §2.4, §3.17-3.18)
// ---------------------------------------------------------------------------

void AgentBridgeServer::subscribeJobSignals()
{
    if (!_window)
        return;

    if (_window->_cmdRunner) {
        CommandLineToolRunner* runner = _window->_cmdRunner;
        connect(runner, &CommandLineToolRunner::toolStarted, this,
                [this](CommandLineToolRunner::Tool, const QString& message) {
                    handleToolStarted(message);
                });
        connect(runner, &CommandLineToolRunner::toolFinished, this,
                [this](CommandLineToolRunner::Tool, bool success, const QString& message,
                       const QString& outputPath, bool) {
                    handleToolFinished(success, message, outputPath);
                });
        connect(runner, &CommandLineToolRunner::consoleOutputReceived, this,
                [this](const QString& output) { handleConsoleOutput(output); });
    }

    if (_window->_segmentationModule) {
        connect(_window->_segmentationModule.get(), &SegmentationModule::growthInProgressChanged,
                this, [this](bool running) { handleGrowthStatusChanged(running); });
    }
}

bool AgentBridgeServer::jobIsRunning() const
{
    if (_activeJob && _activeJob->state == QLatin1String("running"))
        return true;
    if (_window) {
        if (_window->_cmdRunner && _window->_cmdRunner->isRunning())
            return true;
        if (_window->_segmentationGrower && _window->_segmentationGrower->running())
            return true;
    }
    return false;
}

QString AgentBridgeServer::activeJobId() const
{
    return (_activeJob && _activeJob->state == QLatin1String("running")) ? _activeJob->id : QString();
}

QString AgentBridgeServer::beginJob(const QString& kind, const QString& label, bool broadcastStart)
{
    JobRecord job;
    job.id = QStringLiteral("job-%1").arg(_nextJobNum++);
    job.kind = kind;
    job.label = label;
    job.state = QStringLiteral("running");
    job.message = label;
    _activeJob = job;
    if (broadcastStart)
        broadcastJobProgress(*_activeJob, QStringLiteral("started"));
    return job.id;
}

void AgentBridgeServer::finishActiveJob(bool success, const QString& message, const QString& outputPath)
{
    if (!_activeJob)
        return;
    _activeJob->state = success ? QStringLiteral("succeeded") : QStringLiteral("failed");
    if (!message.isEmpty())
        _activeJob->message = message;
    if (!outputPath.isEmpty())
        _activeJob->outputPath = outputPath;

    broadcastJobProgress(*_activeJob, QStringLiteral("finished"), QString(), success);

    _recentJobs.push_back(*_activeJob);
    while (_recentJobs.size() > 8)
        _recentJobs.pop_front();
    _activeJob.reset();
}

void AgentBridgeServer::broadcastJobProgress(const JobRecord& job, const QString& phase,
                                             const QString& messageOverride,
                                             std::optional<bool> success)
{
    QJsonObject params;
    params["jobId"] = job.id;
    params["kind"] = job.kind;
    params["phase"] = phase;
    const QString msg = messageOverride.isEmpty() ? job.message : messageOverride;
    if (!msg.isEmpty())
        params["message"] = msg;
    if (success.has_value()) {
        params["success"] = *success;
        if (!job.outputPath.isEmpty())
            params["outputPath"] = job.outputPath;
    }
    broadcastNotification(QStringLiteral("job.progress"), params);
}

QJsonObject AgentBridgeServer::jobStatusJson(const JobRecord& job) const
{
    QJsonObject o;
    o["jobId"] = job.id;
    o["kind"] = job.kind;
    o["label"] = job.label;
    o["state"] = job.state;
    o["message"] = job.message;
    o["outputPath"] = job.outputPath.isEmpty() ? QJsonValue(QJsonValue::Null)
                                               : QJsonValue(job.outputPath);
    QJsonArray tail;
    for (const QString& line : job.consoleTail)
        tail.append(line);
    o["consoleTail"] = tail;
    return o;
}

void AgentBridgeServer::handleToolStarted(const QString& message)
{
    if (!_activeJob) {
        // Tool run initiated outside the bridge (e.g. a menu action): track it.
        beginJob(QStringLiteral("tool"), message, /*broadcastStart=*/true);
    } else {
        // The RPC that launched this already created + broadcast the job.
        _activeJob->message = message;
    }
}

void AgentBridgeServer::handleToolFinished(bool success, const QString& message,
                                           const QString& outputPath)
{
    finishActiveJob(success, message, outputPath);
}

void AgentBridgeServer::handleConsoleOutput(const QString& output)
{
    if (!_activeJob)
        return;

    const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    for (const QString& line : lines)
        _activeJob->consoleTail.append(line);
    while (_activeJob->consoleTail.size() > 50)
        _activeJob->consoleTail.removeFirst();

    // Rate-limit job.progress "output" to <=10/sec, coalescing (SPEC §3.18).
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(*_activeJob, QStringLiteral("output"), output.trimmed());
}

void AgentBridgeServer::handleGrowthStatusChanged(bool running)
{
    if (running) {
        if (!_activeJob)
            beginJob(QStringLiteral("segmentation.grow"),
                     QStringLiteral("Segmentation growth started"), /*broadcastStart=*/true);
    } else if (_activeJob && _activeJob->kind == QLatin1String("segmentation.grow")) {
        finishActiveJob(true, QStringLiteral("Segmentation growth finished"), QString());
    }
}

QJsonObject AgentBridgeServer::handleJobStatus(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString jobId = p.value("jobId").toString();

    const JobRecord* rec = nullptr;
    if (jobId.isEmpty()) {
        if (_activeJob)
            rec = &*_activeJob;
        else if (!_recentJobs.empty())
            rec = &_recentJobs.back();
    } else {
        if (_activeJob && _activeJob->id == jobId) {
            rec = &*_activeJob;
        } else {
            for (auto it = _recentJobs.rbegin(); it != _recentJobs.rend(); ++it) {
                if (it->id == jobId) {
                    rec = &*it;
                    break;
                }
            }
        }
    }

    if (!rec) {
        QJsonObject data;
        data["kind"] = "job";
        if (!jobId.isEmpty())
            data["id"] = jobId;
        throw AgentBridgeError{-32007, "No such job", data};
    }

    return jobStatusJson(*rec);
}
