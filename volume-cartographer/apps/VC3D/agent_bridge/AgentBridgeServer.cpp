#include "agent_bridge/AgentBridgeServer.hpp"

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
#include "CState.hpp"
#include "LasagnaServiceManager.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "MenuActionController.hpp"
#include "OpenDataManifest.hpp"
#include "OpenDataSampleProject.hpp"
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

QJsonObject vec3ToJson(const cv::Vec3f& v)
{
    QJsonObject o;
    o["x"] = static_cast<double>(v[0]);
    o["y"] = static_cast<double>(v[1]);
    o["z"] = static_cast<double>(v[2]);
    return o;
}

// --- Manual-add enum <-> string mappings (SPEC §9.4/§9.5) ---
QString linePreviewModeToString(ManualAddTool::LinePreviewMode mode)
{
    switch (mode) {
    case ManualAddTool::LinePreviewMode::VerticalOnly:   return QStringLiteral("vertical");
    case ManualAddTool::LinePreviewMode::HorizontalOnly: return QStringLiteral("horizontal");
    case ManualAddTool::LinePreviewMode::Cross:          return QStringLiteral("cross");
    case ManualAddTool::LinePreviewMode::CrossFill:      return QStringLiteral("cross_fill");
    }
    return QStringLiteral("cross");
}

QString interpolationModeToString(ManualAddTool::InterpolationMode mode)
{
    switch (mode) {
    case ManualAddTool::InterpolationMode::ThinPlateSpline:        return QStringLiteral("thin_plate_spline");
    case ManualAddTool::InterpolationMode::TracerRestrictedToFill: return QStringLiteral("tracer_restricted_to_fill");
    }
    return QStringLiteral("thin_plate_spline");
}

// --- Open Data representation-kind <-> string mappings (SPEC §10.2/10.3) ---
QString representationKindToJson(vc3d::opendata::OpenDataRepresentationKind kind)
{
    switch (kind) {
    case vc3d::opendata::OpenDataRepresentationKind::NormalGrids: return QStringLiteral("normal_grids");
    case vc3d::opendata::OpenDataRepresentationKind::Lasagna:     return QStringLiteral("lasagna");
    case vc3d::opendata::OpenDataRepresentationKind::Prediction:  return QStringLiteral("prediction");
    }
    return QStringLiteral("prediction");
}

// Parses a kind string; returns nullopt for an unknown value.
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

    // Line-annotation headless error reporting (SPEC §13 as-built, §1.3): the
    // fiber subsystem funnels virtually every failure — including failures of
    // ASYNCHRONOUS completions (line-optimization results, fiber-save jobs)
    // that land minutes after an RPC already returned — through
    // LineAnnotationController::showError (a blocking QMessageBox) or a
    // dataset-picker QFileDialog. A per-call guard cannot cover the async
    // completions, so suppression is set once for the bridge's lifetime:
    // messages are logged and recorded (takeLastSuppressedError) instead of
    // shown. The bridge is opt-in (--agent-bridge), so this never affects a
    // normal interactive session.
    if (_window && _window->_lineAnnotationController) {
        _window->_lineAnnotationController->setErrorDialogsSuppressed(true);
    }

    // Seeding widget headless dialog suppression (SPEC §15.2, §1.3): every
    // seeding action slot the bridge invokes (preview_rays, cast_rays) opens a
    // precondition QMessageBox::warning when no focus point is set; a static
    // QMessageBox spins a nested event loop, forbidden in a bridge handler. As
    // with the line-annotation valve above, set once for the bridge's lifetime
    // (the bridge is opt-in via --agent-bridge, so interactive sessions are
    // unaffected). NOTE: seeding.run / seeding.expand / seeding.analyze_paths
    // are deliberately NOT exposed — those slots spin a nested
    // QApplication::processEvents loop until child processes finish (SeedingWidget
    // .cpp run/expand: ~while(jobsRunning)... , analyzePaths per-path), which a
    // suppression flag cannot make safe.
    if (_window && _window->_seedingWidget) {
        _window->_seedingWidget->setDialogsSuppressed(true);
    }
}

AgentBridgeServer::~AgentBridgeServer()
{
    // Clean-shutdown removal of the discovery registry file. A hard kill skips
    // this; the reader-side stale-PID check reaps the orphan instead.
    removeRegistryFile();
}

bool AgentBridgeServer::listen(const QString& serverName)
{
    if (!_server) {
        _server = new QLocalServer(this);
        connect(_server, &QLocalServer::newConnection,
                this, &AgentBridgeServer::onNewConnection);
    }

    if (!_server->listen(serverName)) {
        // Stale socket file from a crashed run: remove once and retry (SPEC §1.1).
        QLocalServer::removeServer(serverName);
        if (!_server->listen(serverName))
            return false;
    }

    // Publish this process into the discovery registry so an MCP server can
    // attach without the human relaying the stdout handshake line.
    writeRegistryFile();
    return true;
}

// ---------------------------------------------------------------------------
// Discovery registry file (mirrors LasagnaServiceManager::discoverServices()'s
// ~/.fit_services convention). Pure QFile/QDir I/O -- no UI, no event loop.
// ---------------------------------------------------------------------------

void AgentBridgeServer::writeRegistryFile()
{
    if (!_server)
        return;

    const QString dirPath =
        QDir::homePath() + QStringLiteral("/.vc3d/agent_bridge");
    QDir dir;
    if (!dir.mkpath(dirPath))
        return;  // best-effort: discovery is an optimization, never fatal

    const qint64 pid = QCoreApplication::applicationPid();
    const QString filePath =
        QDir(dirPath).filePath(QStringLiteral("%1.json").arg(pid));

    QJsonObject obj;
    obj[QStringLiteral("pid")] = static_cast<double>(pid);
    obj[QStringLiteral("name")] = _server->serverName();
    obj[QStringLiteral("path")] = _server->fullServerName();
    // Epoch milliseconds: a plain sortable number the reader uses to pick the
    // newest live bridge.
    obj[QStringLiteral("startedAt")] =
        static_cast<double>(QDateTime::currentMSecsSinceEpoch());

    QFile f(filePath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return;
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
    f.close();

    _registryFilePath = filePath;
}

void AgentBridgeServer::removeRegistryFile()
{
    if (_registryFilePath.isEmpty())
        return;
    QFile::remove(_registryFilePath);
    _registryFilePath.clear();
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

    // Drop any deferred responses still pending for this connection so their
    // timers do not fire against a dead socket and the (connection, method)
    // guard does not leak (SPEC §8.4).
    for (auto it = _pendingDeferred.begin(); it != _pendingDeferred.end();) {
        if (it->socket.data() == socket) {
            if (it->timer) {
                it->timer->stop();
                it->timer->deleteLater();
            }
            it = _pendingDeferred.erase(it);
        } else {
            ++it;
        }
    }

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

    // Per-request context: handlers that defer their response (SPEC §8.4) read
    // this to stash the caller. Cleared after the handler returns/throws.
    _currentSocket = socket;
    _currentRequestId = id;
    _currentMethod = method;

    QJsonObject result;
    try {
        result = it.value()(request.value("params"));
    } catch (const AgentBridgeDeferred&) {
        // The handler took ownership of the reply; it will be written later by
        // a signal completion or the timeout (SPEC §8.4). Send nothing now.
        _currentSocket = nullptr;
        _currentRequestId = QJsonValue();
        _currentMethod.clear();
        return;
    } catch (const AgentBridgeError& e) {
        _currentSocket = nullptr;
        _currentRequestId = QJsonValue();
        _currentMethod.clear();
        if (!isNotification)
            sendError(socket, id, e.code, e.message, e.data);
        return;
    } catch (const std::exception& e) {
        _currentSocket = nullptr;
        _currentRequestId = QJsonValue();
        _currentMethod.clear();
        if (!isNotification) {
            QJsonObject data;
            data["detail"] = QString::fromUtf8(e.what());
            sendError(socket, id, -32010, "Internal error", data);
        }
        return;
    } catch (...) {
        _currentSocket = nullptr;
        _currentRequestId = QJsonValue();
        _currentMethod.clear();
        if (!isNotification)
            sendError(socket, id, -32010, "Internal error");
        return;
    }

    _currentSocket = nullptr;
    _currentRequestId = QJsonValue();
    _currentMethod.clear();

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
    _handlers.insert("segments.activate",
        [this](const QJsonValue& p) { return handleSegmentsActivate(p); });
    _handlers.insert("screenshot.capture",
        [this](const QJsonValue& p) { return handleScreenshotCapture(p); });
    _handlers.insert("canvas.get_cursor_volume_point",
        [this](const QJsonValue& p) { return handleCursorVolumePoint(p); });

    // --- Phase 2: canvas + mutating actions ---
    _handlers.insert("canvas.click",
        [this](const QJsonValue& p) { return handleCanvasClick(p, /*addShift=*/false); });
    _handlers.insert("canvas.shift_click",
        [this](const QJsonValue& p) { return handleCanvasClick(p, /*addShift=*/true); });
    _handlers.insert("canvas.drag",
        [this](const QJsonValue& p) { return handleCanvasDrag(p); });
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
    _handlers.insert("segmentation.manual_add.begin",
        [this](const QJsonValue& p) { return handleManualAddBegin(p); });
    _handlers.insert("segmentation.manual_add.finish",
        [this](const QJsonValue& p) { return handleManualAddFinish(p); });
    _handlers.insert("segmentation.manual_add.set_line_mode",
        [this](const QJsonValue& p) { return handleManualAddSetLineMode(p); });
    _handlers.insert("segmentation.manual_add.set_interpolation",
        [this](const QJsonValue& p) { return handleManualAddSetInterpolation(p); });
    _handlers.insert("segmentation.manual_add.undo_constraint",
        [this](const QJsonValue& p) { return handleManualAddUndoConstraint(p); });
    _handlers.insert("segmentation.corrections.set_point_mode",
        [this](const QJsonValue& p) { return handleCorrectionsSetPointMode(p); });
    _handlers.insert("points.commit",
        [this](const QJsonValue& p) { return handlePointsCommit(p); });
    _handlers.insert("points.list",
        [this](const QJsonValue& p) { return handlePointsList(p); });
    _handlers.insert("volume.open",
        [this](const QJsonValue& p) { return handleVolumeOpen(p); });
    _handlers.insert("volume.select",
        [this](const QJsonValue& p) { return handleVolumeSelect(p); });
    _handlers.insert("catalog.open_sample",
        [this](const QJsonValue& p) { return handleCatalogOpenSample(p); });
    _handlers.insert("catalog.list_samples",
        [this](const QJsonValue& p) { return handleCatalogListSamples(p); });
    _handlers.insert("catalog.describe_sample",
        [this](const QJsonValue& p) { return handleCatalogDescribeSample(p); });
    _handlers.insert("job.status",
        [this](const QJsonValue& p) { return handleJobStatus(p); });

    // --- Lasagna RPCs (SPEC §11) ---
    _handlers.insert("lasagna.service_status",
        [this](const QJsonValue& p) { return handleLasagnaServiceStatus(p); });
    _handlers.insert("lasagna.ensure_service",
        [this](const QJsonValue& p) { return handleLasagnaEnsureService(p); });
    _handlers.insert("lasagna.list_datasets",
        [this](const QJsonValue& p) { return handleLasagnaListDatasets(p); });
    _handlers.insert("lasagna.start_optimization",
        [this](const QJsonValue& p) { return handleLasagnaStartOptimization(p); });
    _handlers.insert("lasagna.jobs",
        [this](const QJsonValue& p) { return handleLasagnaJobs(p); });
    _handlers.insert("lasagna.cancel",
        [this](const QJsonValue& p) { return handleLasagnaCancel(p); });
    _handlers.insert("lasagna.select_output_segment",
        [this](const QJsonValue& p) { return handleLasagnaSelectOutputSegment(p); });
    _handlers.insert("lasagna.repeat_last",
        [this](const QJsonValue& p) { return handleLasagnaRepeatLast(p); });
    _handlers.insert("workspace.switch",
        [this](const QJsonValue& p) { return handleWorkspaceSwitch(p); });

    // --- Atlas RPCs (SPEC §12) ---
    _handlers.insert("atlas.open",
        [this](const QJsonValue& p) { return handleAtlasOpen(p); });
    _handlers.insert("atlas.status",
        [this](const QJsonValue& p) { return handleAtlasStatus(p); });
    _handlers.insert("atlas.search_start",
        [this](const QJsonValue& p) { return handleAtlasSearchStart(p); });
    _handlers.insert("atlas.search_cancel",
        [this](const QJsonValue& p) { return handleAtlasSearchCancel(p); });
    _handlers.insert("atlas.search_results",
        [this](const QJsonValue& p) { return handleAtlasSearchResults(p); });
    _handlers.insert("atlas.open_result",
        [this](const QJsonValue& p) { return handleAtlasOpenResult(p); });
    _handlers.insert("atlas.remap",
        [this](const QJsonValue& p) { return handleAtlasRemap(p); });
    _handlers.insert("atlas.optimize_snap_candidates",
        [this](const QJsonValue& p) { return handleAtlasOptimizeSnapCandidates(p); });

    // --- Line-annotation / fiber RPCs (SPEC §13) ---
    _handlers.insert("fiber.launch",
        [this](const QJsonValue& p) { return handleFiberLaunch(p); });
    _handlers.insert("fiber.list",
        [this](const QJsonValue& p) { return handleFiberList(p); });
    _handlers.insert("fiber.open",
        [this](const QJsonValue& p) { return handleFiberOpen(p); });
    _handlers.insert("fiber.set_follow",
        [this](const QJsonValue& p) { return handleFiberSetFollow(p); });
    _handlers.insert("fiber.save",
        [this](const QJsonValue& p) { return handleFiberSave(p); });
    _handlers.insert("fiber.delete",
        [this](const QJsonValue& p) { return handleFiberDelete(p); });
    _handlers.insert("fiber.set_tag",
        [this](const QJsonValue& p) { return handleFiberSetTag(p); });
    _handlers.insert("fiber.create_atlas",
        [this](const QJsonValue& p) { return handleFiberCreateAtlas(p); });
    _handlers.insert("fiber.export",
        [this](const QJsonValue& p) { return handleFiberExport(p); });
    _handlers.insert("fiber.import",
        [this](const QJsonValue& p) { return handleFiberImport(p); });

    // --- Stage 6 backlog surface (SPEC §15) ---
    _handlers.insert("tags.set",
        [this](const QJsonValue& p) { return handleTagsSet(p); });
    _handlers.insert("seeding.set_winding_annotation_mode",
        [this](const QJsonValue& p) { return handleSeedingSetWindingAnnotationMode(p); });
    _handlers.insert("seeding.preview_rays",
        [this](const QJsonValue& p) { return handleSeedingPreviewRays(p); });
    _handlers.insert("seeding.cast_rays",
        [this](const QJsonValue& p) { return handleSeedingCastRays(p); });
    _handlers.insert("seeding.reset_points",
        [this](const QJsonValue& p) { return handleSeedingResetPoints(p); });
    _handlers.insert("segmentation.push_pull.set_config",
        [this](const QJsonValue& p) { return handlePushPullSetConfig(p); });
    _handlers.insert("segmentation.push_pull.start",
        [this](const QJsonValue& p) { return handlePushPullStart(p); });
    _handlers.insert("segmentation.push_pull.stop",
        [this](const QJsonValue& p) { return handlePushPullStop(p); });
    _handlers.insert("tracer.run_trace",
        [this](const QJsonValue& p) { return handleTracerRunTrace(p); });

    // --- Rendering RPC (SPEC §19) ---
    _handlers.insert("render.tifxyz",
        [this](const QJsonValue& p) { return handleRenderTifxyz(p); });

    // --- Flattening RPCs (SPEC §20) ---
    _handlers.insert("flatten.slim",
        [this](const QJsonValue& p) { return handleFlattenSlim(p); });
    _handlers.insert("flatten.abf",
        [this](const QJsonValue& p) { return handleFlattenAbf(p); });
    _handlers.insert("flatten.straighten",
        [this](const QJsonValue& p) { return handleFlattenStraighten(p); });
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
        result["manualAddMode"] = false;
        result["manualAddLineMode"] = QJsonValue::Null;
        result["manualAddInterpolation"] = QJsonValue::Null;
        result["correctionsPointMode"] = false;
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

    // Manual-add (hole-fill) + corrections point-authoring state (SPEC §9.2-9.7;
    // reported here per Stage 2). Line/interpolation modes come from the panel
    // config (they persist whether or not manual-add is currently active).
    if (SegmentationModule* mod = _window->_segmentationModule.get()) {
        result["manualAddMode"] = mod->manualAddMode();
        result["correctionsPointMode"] = mod->correctionPointMode();
    } else {
        result["manualAddMode"] = false;
        result["correctionsPointMode"] = false;
    }
    if (SegmentationWidget* widget = _window->_segmentationWidget) {
        const ManualAddTool::Config cfg = widget->manualAddConfig();
        result["manualAddLineMode"] = linePreviewModeToString(cfg.linePreviewMode);
        result["manualAddInterpolation"] = interpolationModeToString(cfg.interpolationMode);
    } else {
        result["manualAddLineMode"] = QJsonValue::Null;
        result["manualAddInterpolation"] = QJsonValue::Null;
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

    const QJsonObject p = paramsObject(params);
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

    // SegmentationWidget::setEditingEnabled() is the *silent* sync setter (it
    // calls updateEditingState(enabled, /*notifyListeners=*/false)): it flips the
    // widget's own checkbox flag and returns without ever emitting
    // editingModeChanged. On its own that means this handler previously
    // reported {"enabled": true} without ever reaching
    // SegmentationModule::setEditingEnabled -> editingEnabledChanged ->
    // CWindow::onSegmentationEditingModeChanged -> beginEditingSession(): the
    // widget flag flipped but no real edit session was ever established, so
    // every editing-gated RPC downstream (manual_add.*, corrections.*) kept
    // failing with -32007 kind:"session" even after segments.activate. Drive
    // the module directly -- the same call CWindow itself makes to reconcile
    // widget/module drift (see CWindow.cpp's onSurfaceActivated, ~line 9795) --
    // so the real signal cascade (and beginEditingSession/endEditingSession)
    // actually runs.
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
    const QString methodStr = p.value("method").toString(QStringLiteral("tracer"));
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

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("segmentation.grow_patch_from_seed"),
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
    const bool apply = p.value("apply").toBool(true);
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
    const bool active = p.value("active").toBool();

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
        const QJsonValue resv = p.value("resources");
        if (!resv.isObject()) {
            QJsonObject data;
            data["param"] = "resources";
            throw AgentBridgeError{-32602, "resources must be an object", data};
        }
        const QJsonObject res = resv.toObject();
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
                const std::string vid = vv.toString().toStdString();
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
                const QString refStr = rv.toString();
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
                const QString ks = kv.toString();
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

void AgentBridgeServer::completeCatalogOpenJob(
    const MenuActionController::OpenDataSampleOpenOutcome& outcome)
{
    auto it = _activeJobs.find(QStringLiteral("catalog"));
    if (it == _activeJobs.end())
        return;  // job already reaped (e.g. shutdown); nothing to resolve.

    QString message;
    QString vpkgPath;
    if (outcome.success) {
        CState* state = _window ? _window->_state : nullptr;
        vpkgPath = (state && state->hasVpkg()) ? state->vpkgPath() : QString();

        QJsonObject body;
        body["opened"] = true;
        body["sampleId"] = _catalogOpenSampleId;
        body["vpkgPath"] = vpkgPath.isEmpty() ? QJsonValue(QString())
                                              : QJsonValue(vpkgPath);
        QJsonArray idArr;
        if (state && state->vpkg()) {
            for (const auto& id : state->vpkg()->volumeIDs())
                idArr.append(QString::fromStdString(id));
        }
        body["volumeIds"] = idArr;

        QJsonObject attached;
        attached["volumes"] = outcome.result.attachedVolumeEntries;
        attached["segments"] = outcome.result.attachedSegmentEntries;
        attached["normalGrids"] = outcome.result.attachedNormalGrids;
        attached["lasagnaDatasets"] = outcome.result.attachedLasagnaDatasets;
        body["attached"] = attached;

        QJsonArray messages;
        for (const auto& m : outcome.result.messages)
            messages.append(QString::fromStdString(m));
        body["messages"] = messages;

        it->resultJson = body;  // carried by finishJob's copy into history/wire.
        message = QStringLiteral("Opened sample %1").arg(_catalogOpenSampleId);
    } else {
        message = outcome.error.isEmpty()
            ? QStringLiteral("Open Data sample open failed")
            : outcome.error;
        // resultJson stays empty -> "result": null on the wire (SPEC §18.4).
    }

    _catalogOpenSampleId.clear();
    finishJob(QStringLiteral("catalog"), outcome.success, message, vpkgPath);
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
    const bool refresh = p.value("refresh").toBool(false);

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
    const QString sampleId = p.value("sampleId").toString();
    if (sampleId.isEmpty()) {
        QJsonObject data;
        data["param"] = "sampleId";
        throw AgentBridgeError{-32602, "sampleId is required", data};
    }
    const bool refresh = p.value("refresh").toBool(false);
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

    // Flattening lifecycle -> source:"flatten" jobs (SPEC §8.3, §20). Emitted by
    // the SlimJob / ABFJob / StraightenJob classes from BOTH the interactive
    // slots and the headless start* launchers, so human-initiated flattens are
    // registered as external jobs too.
    if (_window->_segmentationCommandHandler) {
        SegmentationCommandHandler* sch = _window->_segmentationCommandHandler.get();
        connect(sch, &SegmentationCommandHandler::flattenJobStarted, this,
                [this](const QString& kind, const QString& label) {
                    handleFlattenStarted(kind, label);
                });
        connect(sch, &SegmentationCommandHandler::flattenJobFinished, this,
                [this](bool success, const QString& message, const QString& outputPath) {
                    handleFlattenFinished(success, message, outputPath);
                });
    }

    // Lasagna optimization lifecycle -> source:"lasagna" jobs (SPEC §8.3, §11.4).
    // The singleton outlives the bridge, so the connections are lifetime-safe.
    LasagnaServiceManager* lasagna = &LasagnaServiceManager::instance();
    connect(lasagna, &LasagnaServiceManager::optimizationStarted, this,
            [this]() { handleLasagnaStarted(); });
    connect(lasagna, &LasagnaServiceManager::jobStarted, this,
            [this](const QString& jobId) { handleLasagnaJobStarted(jobId); });
    connect(lasagna, &LasagnaServiceManager::optimizationProgress, this,
            [this](const QString& stage, int step, int totalSteps, double loss,
                   double /*stageProgress*/, double overallProgress,
                   const QString& stageName) {
                handleLasagnaProgress(stageName.isEmpty() ? stage : stageName,
                                      step, totalSteps, loss, overallProgress);
            });
    connect(lasagna, &LasagnaServiceManager::optimizationFinished, this,
            [this](const QString& outputDir) {
                handleLasagnaFinished(true, QStringLiteral("Optimization finished"), outputDir);
            });
    connect(lasagna, &LasagnaServiceManager::optimizationError, this,
            [this](const QString& message) {
                handleLasagnaFinished(false, message, QString());
            });
    connect(lasagna, &LasagnaServiceManager::jobFinished, this,
            [this](const QString& /*jobId*/, const QString& outputDir) {
                handleLasagnaFinished(true, QStringLiteral("Job finished"), outputDir);
            });
    connect(lasagna, &LasagnaServiceManager::jobError, this,
            [this](const QString& /*jobId*/, const QString& message) {
                handleLasagnaFinished(false, message, QString());
            });
    connect(lasagna, &LasagnaServiceManager::resultsPlaced, this,
            [this](const QString& outputDir, const QStringList& segmentNames) {
                handleLasagnaResultsPlaced(outputDir, segmentNames);
            });

    // Atlas fiber-search lifecycle -> source:"atlas" jobs (SPEC §12.9).
    if (_window) {
        connect(_window, &CWindow::atlasSearchProgressChanged, this,
                [this](int phase, double fraction) {
                    handleAtlasSearchProgress(phase, fraction);
                });
        connect(_window, &CWindow::atlasSearchFinished, this,
                [this](bool success, int resultCount) {
                    handleAtlasSearchFinished(success, resultCount);
                });
    }
}

bool AgentBridgeServer::jobIsRunning(const QString& source) const
{
    if (auto it = _activeJobs.constFind(source);
        it != _activeJobs.constEnd() && it->state == QLatin1String("running"))
        return true;
    if (_window) {
        if (source == QLatin1String("tool") &&
            _window->_cmdRunner && _window->_cmdRunner->isRunning())
            return true;
        if (source == QLatin1String("growth") &&
            _window->_segmentationGrower && _window->_segmentationGrower->running())
            return true;
        // Lifecycle authority for atlas searches: the cancel flag is created
        // when a search launches and reset in its finished handler, so a
        // non-null flag means a search (bridge- or human-initiated) is live.
        if (source == QLatin1String("atlas") && _window->_atlasSearchCancelFlag)
            return true;
    }
    return false;
}

QString AgentBridgeServer::activeJobId(const QString& source) const
{
    if (auto it = _activeJobs.constFind(source);
        it != _activeJobs.constEnd() && it->state == QLatin1String("running"))
        return it->id;
    return QString();
}

void AgentBridgeServer::requireSourceIdle(const QString& source) const
{
    if (!jobIsRunning(source))
        return;
    QJsonObject data;
    const QString jid = activeJobId(source);
    if (!jid.isEmpty())
        data["jobId"] = jid;
    data["source"] = source;
    throw AgentBridgeError{-32004,
        QStringLiteral("A %1 job is already running").arg(source), data};
}

QString AgentBridgeServer::beginJob(const QString& source, const QString& kind,
                                    const QString& label, bool broadcastStart)
{
    JobRecord job;
    job.num = _nextJobNum++;
    job.id = QStringLiteral("job-%1").arg(job.num);
    job.source = source;
    job.kind = kind;
    job.label = label;
    job.state = QStringLiteral("running");
    job.message = label;
    job.startedAtMs = QDateTime::currentMSecsSinceEpoch();
    _activeJobs.insert(source, job);
    if (broadcastStart)
        broadcastJobProgress(_activeJobs.value(source), QStringLiteral("started"));
    return job.id;
}

void AgentBridgeServer::finishJob(const QString& source, bool success,
                                  const QString& message, const QString& outputPath)
{
    auto it = _activeJobs.find(source);
    if (it == _activeJobs.end())
        return;
    JobRecord job = it.value();
    _activeJobs.erase(it);

    job.state = success ? QStringLiteral("succeeded") : QStringLiteral("failed");
    if (!message.isEmpty())
        job.message = message;
    if (!outputPath.isEmpty())
        job.outputPath = outputPath;
    job.finishedAtMs = QDateTime::currentMSecsSinceEpoch();

    broadcastJobProgress(job, QStringLiteral("finished"), QString(), success);

    // Retain the last <=8 completed jobs per source (§8.3).
    std::deque<JobRecord>& hist = _recentJobs[source];
    hist.push_back(job);
    while (hist.size() > 8)
        hist.pop_front();
}

const AgentBridgeServer::JobRecord*
AgentBridgeServer::mostRecentJob(const QString& sourceFilter) const
{
    const JobRecord* best = nullptr;
    auto consider = [&](const JobRecord& rec) {
        if (!sourceFilter.isEmpty() && rec.source != sourceFilter)
            return;
        if (!best || rec.num > best->num)
            best = &rec;
    };
    for (const auto& rec : _activeJobs)
        consider(rec);
    for (const auto& hist : _recentJobs)
        for (const auto& rec : hist)
            consider(rec);
    return best;
}

const AgentBridgeServer::JobRecord*
AgentBridgeServer::jobById(const QString& jobId) const
{
    for (const auto& rec : _activeJobs)
        if (rec.id == jobId)
            return &rec;
    for (const auto& hist : _recentJobs)
        for (const auto& rec : hist)
            if (rec.id == jobId)
                return &rec;
    return nullptr;
}

void AgentBridgeServer::broadcastJobProgress(const JobRecord& job, const QString& phase,
                                             const QString& messageOverride,
                                             std::optional<bool> success)
{
    QJsonObject params;
    params["jobId"] = job.id;
    params["source"] = job.source;   // required in v2 (§8.3)
    params["kind"] = job.kind;
    params["phase"] = phase;
    const QString msg = messageOverride.isEmpty() ? job.message : messageOverride;
    if (!msg.isEmpty())
        params["message"] = msg;
    if (success.has_value()) {
        params["success"] = *success;
        if (!job.outputPath.isEmpty())
            params["outputPath"] = job.outputPath;
        // Terminal job.progress carries the result body too (SPEC §18.4).
        params["result"] = job.resultJson.isEmpty() ? QJsonValue(QJsonValue::Null)
                                                     : QJsonValue(job.resultJson);
    }
    broadcastNotification(QStringLiteral("job.progress"), params);
}

QJsonObject AgentBridgeServer::jobStatusJson(const JobRecord& job) const
{
    QJsonObject o;
    o["jobId"] = job.id;
    o["source"] = job.source;
    o["kind"] = job.kind;
    o["label"] = job.label;
    o["state"] = job.state;
    o["message"] = job.message;
    o["outputPath"] = job.outputPath.isEmpty() ? QJsonValue(QJsonValue::Null)
                                               : QJsonValue(job.outputPath);
    o["externalId"] = job.externalId.isEmpty() ? QJsonValue(QJsonValue::Null)
                                               : QJsonValue(job.externalId);
    QJsonArray tail;
    for (const QString& line : job.consoleTail)
        tail.append(line);
    o["consoleTail"] = tail;
    o["startedAtMs"] = static_cast<double>(job.startedAtMs);
    o["finishedAtMs"] = job.finishedAtMs == 0
                            ? QJsonValue(QJsonValue::Null)
                            : QJsonValue(static_cast<double>(job.finishedAtMs));
    // Additive result body (SPEC §18.4): obj for "catalog" jobs, null otherwise.
    o["result"] = job.resultJson.isEmpty() ? QJsonValue(QJsonValue::Null)
                                           : QJsonValue(job.resultJson);
    return o;
}

void AgentBridgeServer::handleToolStarted(const QString& message)
{
    auto it = _activeJobs.find(QStringLiteral("tool"));
    if (it == _activeJobs.end()) {
        // Tool run initiated outside the bridge (e.g. a menu action): track it
        // as an externally-initiated job (§8.3).
        beginJob(QStringLiteral("tool"), QStringLiteral("tool.external"), message,
                 /*broadcastStart=*/true);
    } else {
        // The RPC that launched this already created + broadcast the job.
        it.value().message = message;
    }
}

void AgentBridgeServer::handleToolFinished(bool success, const QString& message,
                                           const QString& outputPath)
{
    finishJob(QStringLiteral("tool"), success, message, outputPath);
}

void AgentBridgeServer::handleConsoleOutput(const QString& output)
{
    auto it = _activeJobs.find(QStringLiteral("tool"));
    if (it == _activeJobs.end())
        return;
    JobRecord& job = it.value();

    const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    for (const QString& line : lines)
        job.consoleTail.append(line);
    while (job.consoleTail.size() > 50)
        job.consoleTail.removeFirst();

    // Rate-limit job.progress "output" to <=10/sec, coalescing (SPEC §3.18).
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(job, QStringLiteral("output"), output.trimmed());
}

void AgentBridgeServer::handleGrowthStatusChanged(bool running)
{
    if (running) {
        if (!_activeJobs.contains(QStringLiteral("growth")))
            beginJob(QStringLiteral("growth"), QStringLiteral("growth.external"),
                     QStringLiteral("Segmentation growth started"), /*broadcastStart=*/true);
    } else if (_activeJobs.contains(QStringLiteral("growth"))) {
        finishJob(QStringLiteral("growth"), true,
                  QStringLiteral("Segmentation growth finished"), QString());
    }
}

// --- Flattening lifecycle (SPEC §8.3, §20) ---

void AgentBridgeServer::handleFlattenStarted(const QString& kind, const QString& label)
{
    auto it = _activeJobs.find(QStringLiteral("flatten"));
    if (it == _activeJobs.end()) {
        // A flatten launched outside the bridge (a human ran SLIM/ABF/Straighten
        // from the context menu): register it as an externally-initiated job so
        // -32004 / state.get reflect true app state (§8.3). The kind carries the
        // real flatten type (flatten.slim / flatten.abf / flatten.straighten).
        beginJob(QStringLiteral("flatten"), kind,
                 label.isEmpty() ? QStringLiteral("Flatten") : label,
                 /*broadcastStart=*/true);
    } else {
        // The RPC that launched this already created + broadcast the job; adopt
        // the concrete kind/label the job reported.
        it.value().kind = kind;
        if (!label.isEmpty())
            it.value().label = label;
    }
}

void AgentBridgeServer::handleFlattenFinished(bool success, const QString& message,
                                              const QString& outputPath)
{
    // An empty message on failure denotes a user cancel (SPEC §20); surface a
    // stable message so job.status is not blank.
    const QString msg = (!success && message.isEmpty())
                            ? QStringLiteral("Flatten cancelled")
                            : message;
    finishJob(QStringLiteral("flatten"), success, msg, outputPath);
}

// --- Lasagna optimization lifecycle (SPEC §8.3, §11.4) ---

void AgentBridgeServer::handleLasagnaStarted()
{
    // An optimization launched outside the bridge (a human clicked the panel):
    // register it as an externally-initiated job so -32004 / state.get reflect
    // true app state (§8.3). A bridge-submitted job already exists here.
    if (!_activeJobs.contains(QStringLiteral("lasagna"))) {
        beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.external"),
                 QStringLiteral("Lasagna optimization started"), /*broadcastStart=*/true);
    } else {
        auto it = _activeJobs.find(QStringLiteral("lasagna"));
        it.value().message = QStringLiteral("Lasagna optimization started");
    }
}

void AgentBridgeServer::handleLasagnaJobStarted(const QString& externalId)
{
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it == _activeJobs.end()) {
        // Started outside the bridge: register it, carrying the service job id.
        const QString jobId = beginJob(QStringLiteral("lasagna"),
                                       QStringLiteral("lasagna.external"),
                                       QStringLiteral("Lasagna optimization started"),
                                       /*broadcastStart=*/true);
        (void)jobId;
        it = _activeJobs.find(QStringLiteral("lasagna"));
    }
    if (it != _activeJobs.end())
        it.value().externalId = externalId;
}

void AgentBridgeServer::handleLasagnaProgress(const QString& stageName, int step,
                                              int totalSteps, double loss,
                                              double overallProgress)
{
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it == _activeJobs.end()) {
        // Progress before any registered start: adopt it as an external job.
        beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.external"),
                 QStringLiteral("Lasagna optimization running"), /*broadcastStart=*/true);
        it = _activeJobs.find(QStringLiteral("lasagna"));
        if (it == _activeJobs.end())
            return;
    }
    JobRecord& job = it.value();

    QString label = stageName.isEmpty() ? QStringLiteral("optimizing") : stageName;
    if (totalSteps > 0)
        label += QStringLiteral(" step %1/%2").arg(step).arg(totalSteps);
    if (std::isfinite(loss))
        label += QStringLiteral(" loss=%1").arg(loss, 0, 'g', 6);
    if (std::isfinite(overallProgress) && overallProgress > 0.0)
        label += QStringLiteral(" (%1%)").arg(overallProgress * 100.0, 0, 'f', 1);
    job.message = label;

    job.consoleTail.append(label);
    while (job.consoleTail.size() > 50)
        job.consoleTail.removeFirst();

    // Rate-limit job.progress "output" to <=10/sec (SPEC §3.18).
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(job, QStringLiteral("output"), label);
}

void AgentBridgeServer::handleLasagnaFinished(bool success, const QString& message,
                                              const QString& outputPath)
{
    // finishJob is a no-op if no lasagna job is active, so the redundant
    // optimizationFinished + jobFinished pair resolves the job exactly once.
    if (_activeJobs.contains(QStringLiteral("lasagna")))
        finishJob(QStringLiteral("lasagna"), success, message, outputPath);
}

void AgentBridgeServer::handleLasagnaResultsPlaced(const QString& outputDir,
                                                   const QStringList& segmentNames)
{
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it == _activeJobs.end())
        return;
    JobRecord& job = it.value();
    if (!outputDir.isEmpty())
        job.outputPath = outputDir;
    const QString label = segmentNames.isEmpty()
        ? QStringLiteral("Results placed: %1").arg(outputDir)
        : QStringLiteral("Results placed: %1 (%2)")
              .arg(outputDir, segmentNames.join(QStringLiteral(", ")));
    job.message = label;
    broadcastJobProgress(job, QStringLiteral("output"), label);
}

// --- Atlas fiber-search lifecycle (SPEC §12.9) ---

void AgentBridgeServer::handleAtlasSearchProgress(int phase, double fraction)
{
    // Only bridge-initiated searches are tracked (SPEC §12.2): do NOT
    // auto-register an external job here — FinishResults progress also fires
    // on pure UI re-population (group-by-fiber checkbox), which would leak a
    // job that never finishes.
    auto it = _activeJobs.find(QStringLiteral("atlas"));
    if (it == _activeJobs.end())
        return;
    JobRecord& job = it.value();

    const double clamped = std::clamp(fraction, 0.0, 1.0);
    const QString label = QStringLiteral("phase %1/5 (%2%)")
                              .arg(phase)
                              .arg(qRound(clamped * 100.0));
    job.message = label;

    // Rate-limit job.progress "output" to <=10/sec (SPEC §3.18).
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - _lastConsoleBroadcastMs < 100)
        return;
    _lastConsoleBroadcastMs = now;
    broadcastJobProgress(job, QStringLiteral("output"), label);
}

void AgentBridgeServer::handleAtlasSearchFinished(bool success, int resultCount)
{
    // finishJob is a no-op when no atlas job is active (e.g. a human-initiated
    // search, or an early-return failure before the bridge registered a job).
    if (!_activeJobs.contains(QStringLiteral("atlas")))
        return;
    finishJob(QStringLiteral("atlas"), success,
              success
                  ? QStringLiteral("Atlas fiber search finished: %1 result(s)").arg(resultCount)
                  : QStringLiteral("Atlas fiber search canceled or failed"),
              QString());
}

QJsonObject AgentBridgeServer::handleJobStatus(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString jobId = p.value("jobId").toString();
    const QString source = p.value("source").toString();

    const JobRecord* rec = jobId.isEmpty() ? mostRecentJob(source) : jobById(jobId);

    if (!rec) {
        QJsonObject data;
        data["kind"] = "job";
        if (!jobId.isEmpty())
            data["id"] = jobId;
        throw AgentBridgeError{-32007, "No such job", data};
    }

    return jobStatusJson(*rec);
}

// --- Lasagna RPCs (SPEC §11) + workspace switching (SPEC §11.9) ---

QJsonObject AgentBridgeServer::handleLasagnaServiceStatus(const QJsonValue&)
{
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();
    QJsonObject result;
    result["running"] = mgr.isRunning();
    result["external"] = mgr.isExternal();
    result["host"] = mgr.host();
    result["port"] = mgr.port();
    const QString err = mgr.lastError();
    result["lastError"] = err.isEmpty() ? QJsonValue() : QJsonValue(err);
    return result;
}

QJsonObject AgentBridgeServer::handleLasagnaEnsureService(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const bool hasHost = p.contains("host");
    const bool hasPort = p.contains("port");
    if (hasHost != hasPort) {
        QJsonObject data;
        data["detail"] = "host and port must be given together";
        throw AgentBridgeError{-32602, "host/port mismatch", data};
    }

    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();

    if (hasHost && hasPort) {
        // External mode: connectToExternal pings GET /health asynchronously;
        // completion is deferred (SPEC §8.4) on serviceStarted/serviceError.
        const QString host = p.value("host").toString();
        const int port = p.value("port").toInt();
        const int token = beginDeferred(15000, "Lasagna external service connect");
        connect(&mgr, &LasagnaServiceManager::serviceStarted, this,
                [this, token]() {
                    LasagnaServiceManager& m = LasagnaServiceManager::instance();
                    QJsonObject result;
                    result["running"] = true;
                    result["external"] = m.isExternal();
                    result["host"] = m.host();
                    result["port"] = m.port();
                    completeDeferredResult(token, result);
                }, Qt::SingleShotConnection);
        connect(&mgr, &LasagnaServiceManager::serviceError, this,
                [this, token](const QString& message) {
                    QJsonObject data;
                    data["detail"] = message;
                    completeDeferredError(token, -32005,
                                          "Lasagna service connection failed", data);
                }, Qt::SingleShotConnection);
        mgr.connectToExternal(host, port);
        throw AgentBridgeDeferred{};
    }

    // Internal mode: ensureServiceRunning() blocks until the process is up (or
    // fails) and returns synchronously -- no deferral needed.
    const QString pythonPath = p.value("pythonPath").toString();
    if (!mgr.ensureServiceRunning(pythonPath)) {
        QJsonObject data;
        data["detail"] = mgr.lastError();
        throw AgentBridgeError{-32005, "Failed to start lasagna service", data};
    }
    QJsonObject result;
    result["running"] = true;
    result["external"] = mgr.isExternal();
    result["host"] = mgr.host();
    result["port"] = mgr.port();
    return result;
}

QJsonObject AgentBridgeServer::handleLasagnaListDatasets(const QJsonValue&)
{
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();
    if (!mgr.isRunning()) {
        QJsonObject data;
        data["detail"] = "lasagna service is not running";
        throw AgentBridgeError{-32005, "Lasagna service not running", data};
    }
    const int token = beginDeferred(10000, "Lasagna datasets fetch");
    connect(&mgr, &LasagnaServiceManager::datasetsReceived, this,
            [this, token](const QJsonArray& datasets) {
                QJsonObject result;
                result["datasets"] = datasets;
                completeDeferredResult(token, result);
            }, Qt::SingleShotConnection);
    mgr.fetchDatasets();
    throw AgentBridgeDeferred{};
}

QJsonObject AgentBridgeServer::handleLasagnaStartOptimization(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString modeStr = p.value("mode").toString();
    SegmentationLasagnaPanel::LasagnaMode mode;
    if (modeStr == QLatin1String("reoptimize"))
        mode = SegmentationLasagnaPanel::ReOptimize;
    else if (modeStr == QLatin1String("new_model"))
        mode = SegmentationLasagnaPanel::NewModel;
    else if (modeStr == QLatin1String("offset"))
        mode = SegmentationLasagnaPanel::Offset;
    else if (modeStr == QLatin1String("atlas"))
        mode = SegmentationLasagnaPanel::Atlas;
    else {
        QJsonObject data;
        data["param"] = "mode";
        data["value"] = modeStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid mode: %1").arg(modeStr), data};
    }

    requireSourceIdle(QStringLiteral("lasagna"));

    SegmentationWidget* widget = _window->_segmentationWidget;
    SegmentationLasagnaPanel* panel = widget ? widget->lasagnaPanel() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "lasagna panel is not available";
        throw AgentBridgeError{-32009, "Lasagna panel unavailable", data};
    }

    const QString configPath = p.value("configPath").toString();
    const QString atlasPath = p.value("atlasPath").toString();
    std::optional<cv::Vec3i> seed;
    if (p.contains("seed")) {
        const cv::Vec3f v = jsonToVec3(p.value("seed"), "seed");
        seed = cv::Vec3i(qRound(v[0]), qRound(v[1]), qRound(v[2]));
    }

    QString errorMessage;
    const bool started =
        panel->startOptimizationHeadless(state, mode, configPath, seed, atlasPath, &errorMessage);
    if (!started) {
        QJsonObject data;
        data["detail"] = errorMessage;
        if (errorMessage.contains(QLatin1String("config"), Qt::CaseInsensitive)) {
            data["kind"] = "config";
            throw AgentBridgeError{-32007, "Lasagna config not found", data};
        }
        if (errorMessage.contains(QLatin1String("atlas"), Qt::CaseInsensitive)) {
            data["kind"] = "atlas";
            throw AgentBridgeError{-32007, "No atlas selected", data};
        }
        throw AgentBridgeError{-32005, "Lasagna optimization failed to start", data};
    }

    // The optimizationStarted/jobStarted signals may already have fired
    // synchronously (direct connection, same thread) and registered this as
    // an external job via handleLasagnaStarted/handleLasagnaJobStarted --
    // reuse that record rather than double-registering (SPEC §8.3).
    QString jobId;
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it != _activeJobs.end()) {
        it.value().kind = QStringLiteral("lasagna.optimize");
        jobId = it.value().id;
    } else {
        jobId = beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.optimize"),
                          QStringLiteral("Lasagna optimization started"),
                          /*broadcastStart=*/true);
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "lasagna.optimize";
    result["source"] = "lasagna";
    return result;
}

QJsonObject AgentBridgeServer::handleLasagnaJobs(const QJsonValue&)
{
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();
    if (!mgr.isRunning()) {
        QJsonObject data;
        data["detail"] = "lasagna service is not running";
        throw AgentBridgeError{-32005, "Lasagna service not running", data};
    }
    const int token = beginDeferred(10000, "Lasagna jobs fetch");
    connect(&mgr, &LasagnaServiceManager::jobsUpdated, this,
            [this, token](const QJsonArray& jobs) {
                QJsonObject result;
                result["jobs"] = jobs;
                completeDeferredResult(token, result);
            }, Qt::SingleShotConnection);
    mgr.fetchJobs();
    throw AgentBridgeDeferred{};
}

QJsonObject AgentBridgeServer::handleLasagnaCancel(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    LasagnaServiceManager& mgr = LasagnaServiceManager::instance();

    QString serviceJobId;
    if (!p.contains("jobId")) {
        if (!jobIsRunning(QStringLiteral("lasagna"))) {
            QJsonObject data;
            data["kind"] = "job";
            throw AgentBridgeError{-32007, "No active lasagna job", data};
        }
        if (!mgr.isRunning()) {
            QJsonObject data;
            data["detail"] = "lasagna service is not running";
            throw AgentBridgeError{-32005, "Lasagna service not running", data};
        }
        mgr.stopOptimization();
    } else {
        const QString jobId = p.value("jobId").toString();
        if (jobId.startsWith(QLatin1String("job-"))) {
            const JobRecord* job = jobById(jobId);
            if (!job) {
                QJsonObject data;
                data["kind"] = "job";
                data["id"] = jobId;
                throw AgentBridgeError{-32007, QStringLiteral("Unknown job id: %1").arg(jobId),
                                       data};
            }
            serviceJobId = job->externalId;
        } else {
            serviceJobId = jobId;  // raw service job id passthrough (SPEC §11.6).
        }
        if (!mgr.isRunning()) {
            QJsonObject data;
            data["detail"] = "lasagna service is not running";
            throw AgentBridgeError{-32005, "Lasagna service not running", data};
        }
        mgr.cancelJob(serviceJobId);
    }

    QJsonObject result;
    result["cancelRequested"] = true;
    result["serviceJobId"] = serviceJobId.isEmpty() ? QJsonValue() : QJsonValue(serviceJobId);
    return result;
}

QJsonObject AgentBridgeServer::handleLasagnaSelectOutputSegment(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString name = p.value("name").toString();
    if (name.isEmpty()) {
        QJsonObject data;
        data["param"] = "name";
        throw AgentBridgeError{-32602, "name is required", data};
    }
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    SurfacePanelController* panel = _window->_surfacePanel.get();
    if (!panel || !panel->selectSurfaceById(name.toStdString())) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = name;
        throw AgentBridgeError{-32007,
                               QStringLiteral("Unknown or unselectable segment: %1").arg(name),
                               data};
    }

    QJsonObject result;
    result["selected"] = true;
    result["name"] = name;
    return result;
}

QJsonObject AgentBridgeServer::handleLasagnaRepeatLast(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    requireSourceIdle(QStringLiteral("lasagna"));

    SegmentationWidget* widget = _window->_segmentationWidget;
    SegmentationLasagnaPanel* panel = widget ? widget->lasagnaPanel() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "lasagna panel is not available";
        throw AgentBridgeError{-32009, "Lasagna panel unavailable", data};
    }

    // repeatLastLasagnaAction() re-emits lasagnaOptimizeRequested, which
    // CWindow routes to the *interactive* startOptimization(state,
    // statusBar()) overload -- unsafe here (SPEC §1.3). Use the headless
    // twin, which calls startOptimizationHeadless directly instead.
    QString errorMessage;
    const bool started = panel->repeatLastLasagnaActionHeadless(state, &errorMessage);
    if (!started) {
        QJsonObject data;
        data["detail"] = errorMessage;
        throw AgentBridgeError{-32005, "Nothing to repeat", data};
    }

    // Same synchronous-registration caveat as handleLasagnaStartOptimization.
    QString jobId;
    auto it = _activeJobs.find(QStringLiteral("lasagna"));
    if (it != _activeJobs.end()) {
        it.value().kind = QStringLiteral("lasagna.optimize");
        jobId = it.value().id;
    } else {
        jobId = beginJob(QStringLiteral("lasagna"), QStringLiteral("lasagna.optimize"),
                          QStringLiteral("Lasagna optimization started"),
                          /*broadcastStart=*/true);
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "lasagna.optimize";
    result["source"] = "lasagna";
    return result;
}

QJsonObject AgentBridgeServer::handleWorkspaceSwitch(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    const QString name = p.value("name").toString();

    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    if (name == QLatin1String("lasagna")) {
        _window->switchToLasagnaWorkspace();
    } else if (name == QLatin1String("fiber_slice")) {
        _window->switchToFiberSliceWorkspace();
    } else {
        QJsonObject data;
        data["param"] = "name";
        data["value"] = name;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid workspace name: %1").arg(name),
                               data};
    }

    QJsonObject result;
    result["workspace"] = name;
    return result;
}

// ---------------------------------------------------------------------------
// Atlas RPCs (SPEC §12)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleAtlasOpen(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const QString atlasDirStr = p.value("atlasDir").toString();
    if (atlasDirStr.isEmpty()) {
        QJsonObject data;
        data["param"] = "atlasDir";
        throw AgentBridgeError{-32602, "atlasDir is required", data};
    }

    std::filesystem::path dir(atlasDirStr.toStdString());
    if (dir.is_relative()) {
        // Relative paths resolve against the volpkg root (SPEC §12.1), same
        // derivation as CWindow::loadAndDisplayAtlas.
        const std::filesystem::path volpkgRoot = vpkg->path().empty()
            ? std::filesystem::path(vpkg->getVolpkgDirectory())
            : vpkg->path().parent_path();
        dir = volpkgRoot / dir;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        QJsonObject data;
        data["kind"] = "atlas";
        data["path"] = QString::fromStdString(dir.string());
        throw AgentBridgeError{-32007, "Atlas directory not found", data};
    }

    // Headless open (SPEC §12.1 safety split): never the rebuild prompt or a
    // warning dialog — failures come back as the exception text.
    QString err;
    if (!_window->displayAtlasFromDirectoryHeadless(dir, &err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Atlas load failed", data};
    }

    QJsonObject result;
    result["opened"] = true;
    result["atlasDir"] = _window->_currentAtlasDir
        ? QString::fromStdString(_window->_currentAtlasDir->string())
        : QString::fromStdString(dir.string());
    result["atlasName"] = QString::fromStdString(_window->_currentAtlasName);
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasStatus(const QJsonValue&)
{
    QJsonObject result;
    if (_window && _window->_currentAtlasDir) {
        result["atlasDir"] = QString::fromStdString(_window->_currentAtlasDir->string());
        result["atlasName"] = QString::fromStdString(_window->_currentAtlasName);
    } else {
        result["atlasDir"] = QJsonValue::Null;
        result["atlasName"] = QJsonValue::Null;
    }

    QJsonObject search;
    search["running"] = jobIsRunning(QStringLiteral("atlas"));
    // 1-based phase numbering, mirroring atlasSearchPhaseNumber (CWindow.cpp).
    search["phase"] = _window
        ? static_cast<int>(_window->_atlasSearchProgressPhase) + 1 : 1;
    search["phaseCount"] = 5;  // ATLAS_SEARCH_PHASE_COUNT (CWindow.cpp)
    search["completed"] = _window
        ? static_cast<double>(_window->_atlasSearchPhaseCompleted) : 0.0;
    search["total"] = _window
        ? static_cast<double>(_window->_atlasSearchPhaseTotal) : 0.0;
    search["cancelRequested"] = _window ? _window->_atlasSearchCancelRequested : false;
    search["resultCount"] = _window
        ? static_cast<double>(_window->_atlasSearchResults.size()) : 0.0;
    result["search"] = search;
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasSearchStart(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);

    AtlasFiberSearchParams sp;
    const QString modeStr = p.contains("mode")
        ? p.value("mode").toString()
        : QStringLiteral("atlas_to_non_atlas");
    if (modeStr == QLatin1String("atlas_to_non_atlas")) {
        sp.searchMode = 0;  // ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS (CWindow.cpp)
    } else if (modeStr == QLatin1String("non_atlas_only")) {
        sp.searchMode = 1;  // ATLAS_SEARCH_MODE_NON_ATLAS_ONLY (CWindow.cpp)
    } else {
        QJsonObject data;
        data["param"] = "mode";
        data["value"] = modeStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid mode: %1").arg(modeStr), data};
    }

    auto readTags = [](const QJsonObject& obj, const char* key) -> QStringList {
        QStringList tags;
        const QJsonValue v = obj.value(QLatin1String(key));
        if (v.isUndefined() || v.isNull())
            return tags;
        if (!v.isArray()) {
            QJsonObject data;
            data["param"] = QString::fromLatin1(key);
            throw AgentBridgeError{-32602,
                QStringLiteral("%1 must be an array of strings").arg(QLatin1String(key)), data};
        }
        for (const QJsonValue& tv : v.toArray()) {
            const QString s = tv.toString();
            if (!s.isEmpty())
                tags.append(s);
        }
        return tags;
    };
    sp.requiredTags = readTags(p, "requiredTags");
    sp.excludedTags = readTags(p, "excludedTags");

    if (p.contains("maxDistance")) {
        const QJsonValue mv = p.value("maxDistance");
        const double md = mv.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!mv.isDouble() || !std::isfinite(md) || md < 0.0) {
            QJsonObject data;
            data["param"] = "maxDistance";
            throw AgentBridgeError{-32602, "maxDistance must be a non-negative number", data};
        }
        sp.maxDistance = md;
    }
    // else: keep the current spin-box value (SPEC §12.3) — the headless
    // launcher reads the persisted QSettings key.

    requireSourceIdle(QStringLiteral("atlas"));

    if (sp.searchMode == 0 && !_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        data["detail"] =
            "no atlas is open; call atlas.open first or use mode \"non_atlas_only\"";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    QString err;
    if (!_window->startAtlasFiberIntersectionSearchHeadless(sp, &err)) {
        QJsonObject data;
        data["detail"] = err;
        // -32007 for the atlas-shaped preconditions (SPEC §12.3); everything
        // else (no fibers / no lasagna dataset / already running) is -32005.
        if (err.contains(QLatin1String("no saved fiber mappings"), Qt::CaseInsensitive) ||
            err.contains(QLatin1String("Select an atlas"), Qt::CaseInsensitive)) {
            data["kind"] = "atlas";
            throw AgentBridgeError{-32007, "Atlas search preconditions not met", data};
        }
        throw AgentBridgeError{-32005, "Atlas search failed to start", data};
    }

    const QString jobId = beginJob(QStringLiteral("atlas"),
                                   QStringLiteral("atlas.fiber_search"),
                                   QStringLiteral("Atlas fiber search started"),
                                   /*broadcastStart=*/true);

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "atlas.fiber_search";
    result["source"] = "atlas";
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasSearchCancel(const QJsonValue&)
{
    if (!_window || !jobIsRunning(QStringLiteral("atlas"))) {
        QJsonObject data;
        data["kind"] = "job";
        throw AgentBridgeError{-32007, "No atlas search running", data};
    }
    // Dialog-free: sets the cancel flags and updates the progress bars. The
    // job still terminates through atlasSearchFinished (success=false).
    _window->cancelAtlasFiberIntersectionSearch();
    QJsonObject result;
    result["cancelRequested"] = true;
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasSearchResults(const QJsonValue& params)
{
    if (!_window)
        throw AgentBridgeError{-32010, "Window unavailable", {}};

    const QJsonObject p = paramsObject(params);
    int offset = 0;
    if (p.contains("offset")) {
        const QJsonValue ov = p.value("offset");
        const double od = ov.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!ov.isDouble() || !std::isfinite(od) || std::floor(od) != od || od < 0) {
            QJsonObject data;
            data["param"] = "offset";
            throw AgentBridgeError{-32602, "offset must be a non-negative integer", data};
        }
        offset = static_cast<int>(od);
    }
    int limit = 100;
    if (p.contains("limit")) {
        const QJsonValue lv = p.value("limit");
        const double ld = lv.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!lv.isDouble() || !std::isfinite(ld) || std::floor(ld) != ld || ld < 1) {
            QJsonObject data;
            data["param"] = "limit";
            throw AgentBridgeError{-32602, "limit must be a positive integer", data};
        }
        limit = static_cast<int>(std::min(ld, 1000.0));  // clamp to [1, 1000] (§12.5)
    }

    const auto& results = _window->_atlasSearchResults;
    const auto& signedWindings = _window->_atlasSearchSignedWindings;
    const int total = static_cast<int>(results.size());

    auto vec3dJson = [](const cv::Vec3d& v) {
        QJsonObject o;
        o["x"] = v[0];
        o["y"] = v[1];
        o["z"] = v[2];
        return o;
    };

    QJsonArray rows;
    for (int i = offset; i < total && i < offset + limit; ++i) {
        const auto& r = results[static_cast<size_t>(i)];
        QJsonObject row;
        row["index"] = i;  // vector order; the id atlas.open_result takes
        row["sourceFiberId"] = QString::number(r.sourceFiberId);
        row["targetFiberId"] = QString::number(r.targetFiberId);
        row["candidateDistance"] = r.candidateDistance;
        row["refinedScore"] = r.refinedScore;
        row["windingDistance"] = std::isfinite(r.windingDistance)
            ? QJsonValue(r.windingDistance)
            : QJsonValue(QJsonValue::Null);
        const bool haveSigned = static_cast<size_t>(i) < signedWindings.size() &&
                                std::isfinite(signedWindings[static_cast<size_t>(i)]);
        row["signedWinding"] = haveSigned
            ? QJsonValue(signedWindings[static_cast<size_t>(i)])
            : QJsonValue(QJsonValue::Null);
        row["sourcePoint"] = vec3dJson(r.sourcePoint);
        row["targetPoint"] = vec3dJson(r.targetPoint);
        row["sourceArclength"] = r.sourceArclength;
        row["targetArclength"] = r.targetArclength;
        row["converged"] = r.converged;
        row["message"] = QString::fromStdString(r.message);
        rows.push_back(row);
    }

    QJsonObject result;
    result["total"] = total;
    result["offset"] = offset;
    result["results"] = rows;
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasOpenResult(const QJsonValue& params)
{
    if (!_window)
        throw AgentBridgeError{-32010, "Window unavailable", {}};

    const QJsonObject p = paramsObject(params);
    const QJsonValue iv = p.value("index");
    const double id = iv.toDouble(std::numeric_limits<double>::quiet_NaN());
    if (!iv.isDouble() || !std::isfinite(id) || std::floor(id) != id) {
        QJsonObject data;
        data["param"] = "index";
        throw AgentBridgeError{-32602, "index must be an integer", data};
    }
    const int index = static_cast<int>(id);
    const auto& results = _window->_atlasSearchResults;
    if (index < 0 || index >= static_cast<int>(results.size())) {
        QJsonObject data;
        data["kind"] = "result";
        data["index"] = index;
        data["total"] = static_cast<int>(results.size());
        throw AgentBridgeError{-32007, "Atlas search result index out of range", data};
    }

    // Mirror the interactive slot's preconditions (CWindow::openAtlasSearchResult)
    // without its QMessageBoxes (SPEC §1.3).
    if (!_window->_lineAnnotationController || !_window->_intersectionsMdiArea) {
        QJsonObject data;
        data["detail"] = "intersections workspace is not available";
        throw AgentBridgeError{-32005, "Intersections workspace unavailable", data};
    }
    if (!_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    if (_window->_workspaceTabs && _window->_intersectionsWorkspaceWindow)
        _window->_workspaceTabs->setCurrentWidget(_window->_intersectionsWorkspaceWindow);

    QString err;
    if (!_window->_lineAnnotationController->showIntersectionInspectionHeadless(
            results[static_cast<size_t>(index)],
            _window->_intersectionsMdiArea,
            _window->_currentAtlasDir,
            &err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Could not open atlas search result", data};
    }

    QJsonObject result;
    result["opened"] = true;
    result["index"] = index;
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasRemap(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    QString err;
    if (!_window->startAtlasRemapHeadless(&err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Atlas remap failed to start", data};
    }

    QJsonObject result;
    result["remapped"] = true;
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasOptimizeSnapCandidates(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    QString err;
    if (!_window->optimizeAtlasSnapCandidatesHeadless(&err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Snap-candidate ranking failed to start", data};
    }

    QJsonObject result;
    result["requested"] = true;
    return result;
}

// ---------------------------------------------------------------------------
// Line-annotation / fiber RPCs (SPEC §13)
//
// All handlers rely on LineAnnotationController running with error dialogs
// suppressed (set in the constructor): the controller's public API reports
// most failures through showError(), which in headless mode records the
// message instead of raising a QMessageBox. Handlers clear the recorded
// message before a call (takeLastSuppressedError) and turn any message
// recorded during the call into a structured -32005 error.
// ---------------------------------------------------------------------------

namespace {

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
        if (std::isfinite(d) && d > 0 && std::floor(d) == d)
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

    const bool replaceOwning = p.value("replaceOwning").toBool(true);
    (void)ctrl->takeLastSuppressedError();
    ctrl->launchFromViewerAtPoint(chunked, scenePos, replaceOwning);
    const QString err = ctrl->takeLastSuppressedError();
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
        if (!v.isDouble() || !std::isfinite(d) || std::floor(d) != d || d < 0) {
            QJsonObject data;
            data["param"] = QString::fromLatin1(key);
            throw AgentBridgeError{-32602,
                QStringLiteral("%1 must be a non-negative integer").arg(QLatin1String(key)),
                data};
        }
        return static_cast<int>(d);
    };

    (void)ctrl->takeLastSuppressedError();
    if (p.contains("controlPointIndex")) {
        ctrl->openFiberAtControlPoint(fiberId, readIndex("controlPointIndex"));
    } else if (p.contains("linePointIndex")) {
        ctrl->openFiberAtLinePointIndex(fiberId, readIndex("linePointIndex"));
    } else if (p.contains("span")) {
        const QJsonValue sv = p.value("span");
        if (!sv.isArray() || sv.toArray().size() != 2) {
            QJsonObject data;
            data["param"] = "span";
            throw AgentBridgeError{-32602, "span must be [firstControlIndex, secondControlIndex]", data};
        }
        const QJsonArray sa = sv.toArray();
        auto readSpanIndex = [&](const QJsonValue& v, const char* which) -> int {
            const double d = v.toDouble(std::numeric_limits<double>::quiet_NaN());
            if (!v.isDouble() || !std::isfinite(d) || std::floor(d) != d || d < 0) {
                QJsonObject data;
                data["param"] = "span";
                data["detail"] = QStringLiteral("%1 span index must be a non-negative integer")
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

    const QString err = ctrl->takeLastSuppressedError();
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

    (void)ctrl->takeLastSuppressedError();
    // Headless split (SPEC §13.5 as-built): the interactive saveOpenFibers()
    // ends in waitForFiberSaves(), which spins a nested QEventLoop — forbidden
    // here (§1.3). The headless variant schedules the same saves; they
    // complete asynchronously on the fiber-save watcher.
    ctrl->saveOpenFibersHeadless();
    const QString err = ctrl->takeLastSuppressedError();
    if (!err.isEmpty()) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "fiber.save failed", data};
    }

    QJsonObject result;
    result["saved"] = true;
    return result;
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

    (void)ctrl->takeLastSuppressedError();
    ctrl->deleteFibers(ids);
    const QString err = ctrl->takeLastSuppressedError();

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
    const QString tag = p.value("tag").toString().trimmed();
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

    (void)ctrl->takeLastSuppressedError();
    ctrl->setFiberTag(fiberId, tag, ev.toBool());
    const QString err = ctrl->takeLastSuppressedError();
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

    // As-built deviation from §13.8's deferred design: createAtlasFromFiber is
    // fully SYNCHRONOUS (the atlasCreated signal fires before it returns), so
    // the deferred machinery adds nothing — and both its failure path
    // (showError QMessageBox) and its success path (atlasCreated ->
    // CWindow::displayAtlasFromDirectory -> possible rebuild
    // QMessageBox::question) violate §1.3. The headless split runs the same
    // core dialog-free, and the bridge then displays the created atlas via
    // the already-proven displayAtlasFromDirectoryHeadless (SPEC §12.1).
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

    const QString pathStr = p.value("path").toString();
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

    const QString pathStr = p.value("path").toString();
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

// ---------------------------------------------------------------------------
// Stage 6 backlog surface (SPEC §15): tags, seeding, push/pull, run-trace
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleTagsSet(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = paramsObject(params);
    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }
    const QString tagStr = p.value("tag").toString();
    if (!p.contains("enabled") || !p.value("enabled").isBool()) {
        QJsonObject data;
        data["param"] = "enabled";
        throw AgentBridgeError{-32602, "enabled (bool) is required", data};
    }
    const bool enabled = p.value("enabled").toBool();

    // §15.1: the four-value Tag enum; "revisit" does NOT exist -> -32602.
    SurfacePanelController::Tag tag;
    if (tagStr == QLatin1String("approved"))
        tag = SurfacePanelController::Tag::Approved;
    else if (tagStr == QLatin1String("defective"))
        tag = SurfacePanelController::Tag::Defective;
    else if (tagStr == QLatin1String("reviewed"))
        tag = SurfacePanelController::Tag::Reviewed;
    else if (tagStr == QLatin1String("inspect"))
        tag = SurfacePanelController::Tag::Inspect;
    else {
        QJsonObject data;
        data["param"] = "tag";
        data["value"] = tagStr;
        throw AgentBridgeError{-32602, QStringLiteral("Invalid tag: %1").arg(tagStr), data};
    }

    SurfacePanelController* panel = _window ? _window->_surfacePanel.get() : nullptr;
    if (!panel) {
        QJsonObject data;
        data["detail"] = "surface panel is not available";
        throw AgentBridgeError{-32010, "Surface panel unavailable", data};
    }

    // §15.1: select first (documented side effect: leaves segmentId selected,
    // which enables the tag checkboxes), then setTagChecked. selectSurfaceById
    // returns false for an unknown/unloaded id -> -32007.
    const std::string segmentId = segmentIdQ.toStdString();
    if (!panel->selectSurfaceById(segmentId)) {
        QJsonObject data;
        data["kind"] = "segment";
        data["id"] = segmentIdQ;
        throw AgentBridgeError{-32007, "Segment not found", data};
    }
    if (!panel->setTagChecked(tag, enabled)) {
        // Selection succeeded but the tag checkbox is unavailable/disabled.
        QJsonObject data;
        data["detail"] = "tag checkbox is not available for this segment";
        throw AgentBridgeError{-32010, "Tag could not be set", data};
    }

    QJsonObject result;
    result["segmentId"] = segmentIdQ;
    result["tag"] = tagStr;
    result["enabled"] = enabled;
    return result;
}

QJsonObject AgentBridgeServer::handleSeedingSetWindingAnnotationMode(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    const QJsonObject p = paramsObject(params);
    if (!p.contains("active") || !p.value("active").isBool()) {
        QJsonObject data;
        data["param"] = "active";
        throw AgentBridgeError{-32602, "active (bool) is required", data};
    }
    const bool active = p.value("active").toBool();
    widget->setRelWindingAnnotationMode(active);
    QJsonObject result;
    result["active"] = active;
    return result;
}

QJsonObject AgentBridgeServer::handleSeedingPreviewRays(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    widget->runPreviewRays();
    QJsonObject result;
    result["requested"] = true;
    return result;
}

QJsonObject AgentBridgeServer::handleSeedingCastRays(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    widget->runCastRays();
    QJsonObject result;
    result["requested"] = true;
    return result;
}

QJsonObject AgentBridgeServer::handleSeedingResetPoints(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    SeedingWidget* widget = _window ? _window->_seedingWidget : nullptr;
    if (!widget) {
        QJsonObject data;
        data["detail"] = "seeding widget is not available";
        throw AgentBridgeError{-32010, "Seeding widget unavailable", data};
    }
    widget->runResetPoints();
    QJsonObject result;
    result["reset"] = true;
    return result;
}

namespace {
QJsonObject pushPullConfigToJson(const AlphaPushPullConfig& c)
{
    QJsonObject o;
    o["start"] = static_cast<double>(c.start);
    o["stop"] = static_cast<double>(c.stop);
    o["step"] = static_cast<double>(c.step);
    o["low"] = static_cast<double>(c.low);
    o["high"] = static_cast<double>(c.high);
    o["blurRadius"] = c.blurRadius;
    o["computeScale"] = c.computeScale;
    o["perVertexLimit"] = static_cast<double>(c.perVertexLimit);
    o["perVertex"] = c.perVertex;
    return o;
}
}  // namespace

QJsonObject AgentBridgeServer::handlePushPullSetConfig(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!mod || !widget) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32010, "Segmentation module unavailable", data};
    }

    const QJsonObject p = paramsObject(params);

    // Read-modify-write over the current effective config (from the panel, which
    // always exists even before a session opens), then sanitize + apply.
    AlphaPushPullConfig cfg = widget->alphaPushPullConfig();

    auto readFloat = [&](const char* key, float& dst) {
        if (!p.contains(key)) return;
        const QJsonValue v = p.value(key);
        const double d = v.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!v.isDouble() || !std::isfinite(d)) {
            QJsonObject data;
            data["param"] = key;
            throw AgentBridgeError{-32602, QStringLiteral("%1 must be a finite number").arg(key), data};
        }
        dst = static_cast<float>(d);
    };
    auto readInt = [&](const char* key, int& dst) {
        if (!p.contains(key)) return;
        const QJsonValue v = p.value(key);
        if (!v.isDouble()) {
            QJsonObject data;
            data["param"] = key;
            throw AgentBridgeError{-32602, QStringLiteral("%1 must be an integer").arg(key), data};
        }
        dst = v.toInt();
    };
    auto readBool = [&](const char* key, bool& dst) {
        if (!p.contains(key)) return;
        const QJsonValue v = p.value(key);
        if (!v.isBool()) {
            QJsonObject data;
            data["param"] = key;
            throw AgentBridgeError{-32602, QStringLiteral("%1 must be a boolean").arg(key), data};
        }
        dst = v.toBool();
    };

    readFloat("start", cfg.start);
    readFloat("stop", cfg.stop);
    readFloat("step", cfg.step);
    readFloat("low", cfg.low);
    readFloat("high", cfg.high);
    readInt("blurRadius", cfg.blurRadius);
    readInt("computeScale", cfg.computeScale);
    readFloat("perVertexLimit", cfg.perVertexLimit);
    readBool("perVertex", cfg.perVertex);

    const AlphaPushPullConfig sanitized = SegmentationPushPullTool::sanitizeConfig(cfg);
    mod->setAlphaPushPullConfig(sanitized);  // sanitizes + updates tool + panel UI

    // Report the effective config as sanitized (matches what the panel now holds).
    return pushPullConfigToJson(sanitized);
}

QJsonObject AgentBridgeServer::handlePushPullStart(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    SegmentationWidget* widget = _window ? _window->_segmentationWidget : nullptr;
    if (!mod) {
        QJsonObject data;
        data["detail"] = "segmentation module is not available";
        throw AgentBridgeError{-32010, "Segmentation module unavailable", data};
    }

    const QJsonObject p = paramsObject(params);
    const QString dirStr = p.value("direction").toString();
    int direction = 0;
    if (dirStr == QLatin1String("push"))
        direction = 1;
    else if (dirStr == QLatin1String("pull"))
        direction = -1;
    else {
        QJsonObject data;
        data["param"] = "direction";
        data["value"] = dirStr;
        throw AgentBridgeError{-32602, "direction must be \"push\" or \"pull\"", data};
    }

    std::optional<bool> alphaOverride;
    if (p.contains("alpha")) {
        if (!p.value("alpha").isBool()) {
            QJsonObject data;
            data["param"] = "alpha";
            throw AgentBridgeError{-32602, "alpha must be a boolean", data};
        }
        alphaOverride = p.value("alpha").toBool();
    }

    // §15.3 preconditions: editing enabled + active edit session.
    if (!widget || !widget->isEditingEnabled())
        throw AgentBridgeError{-32008, "Segmentation editing is not enabled", {}};
    if (!mod->hasActiveSession()) {
        QJsonObject data;
        data["kind"] = "session";
        data["detail"] = "no active segmentation edit session";
        throw AgentBridgeError{-32007, "No active edit session", data};
    }

    // startPushPull returns false when there is no valid hover target (the agent
    // must position the cursor first with a buttonless canvas.drag, §15.3) — that
    // is reported as active:false, not an error.
    const bool active = mod->startPushPullMode(direction, alphaOverride);
    QJsonObject result;
    result["active"] = active;
    return result;
}

QJsonObject AgentBridgeServer::handlePushPullStop(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    SegmentationModule* mod = _window ? _window->_segmentationModule.get() : nullptr;
    if (mod)
        mod->stopPushPullAll();
    QJsonObject result;
    result["stopped"] = true;
    return result;
}

QJsonObject AgentBridgeServer::handleTracerRunTrace(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    // Run Trace runs vc_grow_seg_from_segments: the "tool" source (§8.3).
    requireSourceIdle(QStringLiteral("tool"));

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Run Trace unavailable", data};
    }

    SegmentationCommandHandler::RunTraceParams rt;
    const QJsonValue overrides = p.value("paramOverrides");
    if (overrides.isObject())
        rt.paramOverrides = overrides.toObject();
    else if (!overrides.isUndefined() && !overrides.isNull()) {
        QJsonObject data;
        data["param"] = "paramOverrides";
        throw AgentBridgeError{-32602, "paramOverrides must be an object", data};
    }
    if (p.contains("ompThreads")) {
        if (!p.value("ompThreads").isDouble()) {
            QJsonObject data;
            data["param"] = "ompThreads";
            throw AgentBridgeError{-32602, "ompThreads must be an integer", data};
        }
        rt.ompThreads = p.value("ompThreads").toInt();
    }
    rt.tgtDir = p.value("outputDir").toString();

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("tracer.run_trace"),
                                   QStringLiteral("Run Trace"),
                                   /*broadcastStart=*/false);

    // Suppress the runner's interactive "Operation Complete" QMessageBox for this
    // headless run so the modal dialog cannot starve the toolFinished slots that
    // transition the job out of "running" (auto-cleared on toolFinished; also
    // cleared on the synchronous-failure path below).
    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    QString err;
    QString outputDir;
    if (!handler->startRunTrace(segmentIdQ.toStdString(), rt, &err, &outputDir)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJobs.remove(QStringLiteral("tool"));
        // Map the distinct failure sentences from startRunTrace to codes (§15.4).
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentIdQ;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("trace_params.json not found"))) {
            QJsonObject data;
            data["kind"] = "file";
            data["detail"] = err;
            throw AgentBridgeError{-32007, "trace_params.json not found", data};
        }
        if (err.contains(QLatin1String("remote"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32009, "Remote volumes are unsupported by Run Trace", data};
        }
        if (err.contains(QLatin1String("Command line tools not available"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "vc_grow_seg_from_segments unavailable", data};
        }
        if (err.contains(QLatin1String("already running"))) {
            QJsonObject data;
            data["source"] = "tool";
            data["detail"] = err;
            throw AgentBridgeError{-32004, "A tool job is already running", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start Run Trace", data};
    }

    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "tracer.run_trace";
    result["source"] = "tool";
    result["outputDir"] = outputDir;
    return result;
}

// ---------------------------------------------------------------------------
// render.tifxyz (SPEC §19)
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleRenderTifxyz(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentIdQ = p.value("segmentId").toString();
    if (segmentIdQ.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    SegmentationCommandHandler::RenderSegmentParams rp;

    // outputFormat: required; the headline new capability over the GUI (which is
    // hardcoded to a per-slice TIFF stack).
    const QString outputFormat = p.value("outputFormat").toString();
    if (outputFormat == QLatin1String("zarr")) {
        rp.outputFormat = CommandLineToolRunner::RenderOutputFormat::Zarr;
    } else if (outputFormat == QLatin1String("tif_stack")) {
        rp.outputFormat = CommandLineToolRunner::RenderOutputFormat::TifStack;
    } else {
        QJsonObject data;
        data["param"] = "outputFormat";
        data["detail"] = "outputFormat must be \"zarr\" or \"tif_stack\"";
        throw AgentBridgeError{-32602, "Invalid outputFormat", data};
    }

    // volumeId: optional; validated here so a bad id is a clean -32007 before the
    // job is registered (mirrors segmentation.grow_patch_from_seed).
    rp.volumeId = p.value("volumeId").toString();
    if (!rp.volumeId.isEmpty()) {
        const auto ids = state->vpkg()->volumeIDs();
        if (std::find(ids.begin(), ids.end(), rp.volumeId.toStdString()) == ids.end()) {
            QJsonObject data;
            data["kind"] = "volume";
            data["id"] = rp.volumeId;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown volume id: %1").arg(rp.volumeId), data};
        }
    }

    // scale: optional, default 1.0; must be finite and > 0.
    if (p.contains("scale")) {
        if (!p.value("scale").isDouble()) {
            QJsonObject data;
            data["param"] = "scale";
            throw AgentBridgeError{-32602, "scale must be a number", data};
        }
        rp.scale = static_cast<float>(p.value("scale").toDouble(1.0));
    }
    if (!std::isfinite(rp.scale) || rp.scale <= 0.0f) {
        QJsonObject data;
        data["param"] = "scale";
        throw AgentBridgeError{-32602, "scale must be a finite value > 0", data};
    }

    // groupIdx: optional, default 0; OME-Zarr group index (>= 0).
    if (p.contains("groupIdx")) {
        if (!p.value("groupIdx").isDouble()) {
            QJsonObject data;
            data["param"] = "groupIdx";
            throw AgentBridgeError{-32602, "groupIdx must be an integer", data};
        }
        rp.groupIdx = p.value("groupIdx").toInt(0);
    }
    if (rp.groupIdx < 0) {
        QJsonObject data;
        data["param"] = "groupIdx";
        throw AgentBridgeError{-32602, "groupIdx must be >= 0", data};
    }

    // numSlices: optional, default 1; must be >= 1.
    if (p.contains("numSlices")) {
        if (!p.value("numSlices").isDouble()) {
            QJsonObject data;
            data["param"] = "numSlices";
            throw AgentBridgeError{-32602, "numSlices must be an integer", data};
        }
        rp.numSlices = p.value("numSlices").toInt(1);
    }
    if (rp.numSlices < 1) {
        QJsonObject data;
        data["param"] = "numSlices";
        throw AgentBridgeError{-32602, "numSlices must be >= 1", data};
    }

    // voxelSize: optional override; when omitted the tool derives it from volume
    // metadata (matching the interactive render path).
    if (p.contains("voxelSize") && !p.value("voxelSize").isNull()) {
        if (!p.value("voxelSize").isDouble()) {
            QJsonObject data;
            data["param"] = "voxelSize";
            throw AgentBridgeError{-32602, "voxelSize must be a number", data};
        }
        const double vs = p.value("voxelSize").toDouble();
        if (!std::isfinite(vs) || vs <= 0.0) {
            QJsonObject data;
            data["param"] = "voxelSize";
            throw AgentBridgeError{-32602, "voxelSize must be a finite value > 0", data};
        }
        rp.hasVoxelSize = true;
        rp.voxelSizeUm = vs;
    }

    rp.outputDir = p.value("outputDir").toString();

    // Render runs the external vc_render_tifxyz process: the "tool" source (§8.3).
    requireSourceIdle(QStringLiteral("tool"));

    SegmentationCommandHandler* handler = _window->_segmentationCommandHandler.get();
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Render unavailable", data};
    }

    const QString effectiveVolumeId = rp.volumeId.isEmpty()
        ? QString::fromStdString(state->currentVolumeId())
        : rp.volumeId;

    const QString jobId = beginJob(QStringLiteral("tool"),
                                   QStringLiteral("render.tifxyz"),
                                   QStringLiteral("Render segment"),
                                   /*broadcastStart=*/false);

    // Suppress the runner's interactive "Operation Complete" QMessageBox for this
    // headless run so the modal dialog cannot starve the toolFinished slots that
    // transition the job out of "running" (auto-cleared on toolFinished; also
    // cleared on the synchronous-failure path below). Same proven pattern as
    // segmentation.grow_patch_from_seed / tracer.run_trace.
    if (_window->_cmdRunner)
        _window->_cmdRunner->setSuppressCompletionDialogs(true);

    QString err;
    QString outputDir;
    if (!handler->startRenderSegment(segmentIdQ.toStdString(), rp, &err, &outputDir)) {
        if (_window->_cmdRunner)
            _window->_cmdRunner->setSuppressCompletionDialogs(false);
        _activeJobs.remove(QStringLiteral("tool"));
        // Map the distinct failure sentences from startRenderSegment (§19).
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentIdQ;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("Unknown volume id"))) {
            QJsonObject data;
            data["kind"] = "volume";
            data["detail"] = err;
            throw AgentBridgeError{-32007, err, data};
        }
        if (err.contains(QLatin1String("not found or not executable"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "vc_render_tifxyz unavailable", data};
        }
        if (err.contains(QLatin1String("already running"))) {
            QJsonObject data;
            data["source"] = "tool";
            data["detail"] = err;
            throw AgentBridgeError{-32004, "A tool job is already running", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start render", data};
    }

    if (auto it = _activeJobs.find(QStringLiteral("tool")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "render.tifxyz";
    result["source"] = "tool";
    result["outputDir"] = outputDir;
    result["outputFormat"] = outputFormat;
    result["volumeId"] = effectiveVolumeId;
    return result;
}

// ---------------------------------------------------------------------------
// Flattening RPCs (SPEC §20): flatten.slim / flatten.abf / flatten.straighten
// ---------------------------------------------------------------------------

// Shared launch body for all three flatten RPCs. The specific handler validates
// its params and builds `launch` (a closure over the concrete start* launcher);
// this registers the source:"flatten" job, invokes the launcher, and maps its
// distinct failure sentences to JSON-RPC codes. Reuses the exact
// beginJob->launch->broadcast pattern proven for render.tifxyz / tracer.run_trace,
// except completion is driven by SegmentationCommandHandler::flattenJobFinished
// rather than a CommandLineToolRunner signal (the flatten jobs own their own
// QProcess / QtConcurrent lifecycle).
QJsonObject AgentBridgeServer::launchFlattenJob(
    const QString& kind, const QString& label, const QString& segmentId,
    const std::function<bool(QString* err, QString* outDir)>& launch)
{
    // Only one flatten at a time (its own source, so it may run concurrently
    // with a "tool"/"growth"/etc. job, §8.3).
    requireSourceIdle(QStringLiteral("flatten"));

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Flatten unavailable", data};
    }

    const QString jobId = beginJob(QStringLiteral("flatten"), kind, label,
                                   /*broadcastStart=*/false);

    QString err;
    QString outputDir;
    if (!launch(&err, &outputDir)) {
        _activeJobs.remove(QStringLiteral("flatten"));
        // Map the distinct failure sentences the start* launchers produce (§20).
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentId;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("not found or not executable"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "Flatten tool unavailable", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start flatten", data};
    }

    // The job ctor already emitted flattenJobStarted (adopted by
    // handleFlattenStarted into the active record); attach the resolved output
    // path and broadcast the start now that we have it.
    if (auto it = _activeJobs.find(QStringLiteral("flatten")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = kind;
    result["source"] = "flatten";
    result["outputDir"] = outputDir;
    return result;
}

QJsonObject AgentBridgeServer::handleFlattenSlim(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (segmentId.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    SegmentationCommandHandler::SlimFlattenParams sp;  // headless defaults (§20)

    if (p.contains("iterations")) {
        if (!p.value("iterations").isDouble()) {
            QJsonObject data; data["param"] = "iterations";
            throw AgentBridgeError{-32602, "iterations must be an integer", data};
        }
        sp.iterations = p.value("iterations").toInt();
        if (sp.iterations < 1) {
            QJsonObject data; data["param"] = "iterations";
            throw AgentBridgeError{-32602, "iterations must be >= 1", data};
        }
    }
    if (p.contains("tolerance")) {
        if (!p.value("tolerance").isDouble()) {
            QJsonObject data; data["param"] = "tolerance";
            throw AgentBridgeError{-32602, "tolerance must be a number", data};
        }
        sp.tolerance = p.value("tolerance").toDouble();
        if (!std::isfinite(sp.tolerance) || sp.tolerance < 0.0) {
            QJsonObject data; data["param"] = "tolerance";
            throw AgentBridgeError{-32602, "tolerance must be a finite value >= 0", data};
        }
    }
    if (p.contains("energyType") && !p.value("energyType").isNull()) {
        const QString e = p.value("energyType").toString();
        if (e != QLatin1String("symmetric_dirichlet") && e != QLatin1String("conformal")) {
            QJsonObject data; data["param"] = "energyType";
            data["detail"] = "energyType must be \"symmetric_dirichlet\" or \"conformal\"";
            throw AgentBridgeError{-32602, "Invalid energyType", data};
        }
        sp.energyType = e;
    }
    if (p.contains("keepPercent")) {
        if (!p.value("keepPercent").isDouble()) {
            QJsonObject data; data["param"] = "keepPercent";
            throw AgentBridgeError{-32602, "keepPercent must be a number", data};
        }
        sp.keepPercent = p.value("keepPercent").toDouble();
        if (!std::isfinite(sp.keepPercent) || sp.keepPercent <= 0.0 || sp.keepPercent > 100.0) {
            QJsonObject data; data["param"] = "keepPercent";
            throw AgentBridgeError{-32602, "keepPercent must be in (0, 100]", data};
        }
    }
    if (p.contains("inpaintHoles")) {
        if (!p.value("inpaintHoles").isBool()) {
            QJsonObject data; data["param"] = "inpaintHoles";
            throw AgentBridgeError{-32602, "inpaintHoles must be a boolean", data};
        }
        sp.inpaintHoles = p.value("inpaintHoles").toBool();
    }
    sp.outputDir = p.value("outputDir").toString();

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.slim"), QStringLiteral("SLIM flatten"), segmentId,
        [handler, segmentId, sp](QString* err, QString* outDir) {
            return handler && handler->startSlimFlatten(segmentId.toStdString(), sp, err, outDir);
        });
}

QJsonObject AgentBridgeServer::handleFlattenAbf(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (segmentId.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    int iterations = 10;        // ABFFlattenDialog session default
    int downsampleFactor = 1;
    if (p.contains("iterations")) {
        if (!p.value("iterations").isDouble()) {
            QJsonObject data; data["param"] = "iterations";
            throw AgentBridgeError{-32602, "iterations must be an integer", data};
        }
        iterations = p.value("iterations").toInt();
        if (iterations < 1) {
            QJsonObject data; data["param"] = "iterations";
            throw AgentBridgeError{-32602, "iterations must be >= 1", data};
        }
    }
    if (p.contains("downsampleFactor")) {
        if (!p.value("downsampleFactor").isDouble()) {
            QJsonObject data; data["param"] = "downsampleFactor";
            throw AgentBridgeError{-32602, "downsampleFactor must be an integer", data};
        }
        downsampleFactor = p.value("downsampleFactor").toInt();
        if (downsampleFactor < 1) {
            QJsonObject data; data["param"] = "downsampleFactor";
            throw AgentBridgeError{-32602, "downsampleFactor must be >= 1", data};
        }
    }

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.abf"), QStringLiteral("ABF++ flatten"), segmentId,
        [handler, segmentId, iterations, downsampleFactor](QString* err, QString* outDir) {
            return handler && handler->startAbfFlatten(segmentId.toStdString(),
                                                       iterations, downsampleFactor, err, outDir);
        });
}

QJsonObject AgentBridgeServer::handleFlattenStraighten(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (segmentId.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    SegmentationCommandHandler::StraightenParams stp;  // defaults mirror the dialog

    if (p.contains("unbend")) {
        if (!p.value("unbend").isBool()) {
            QJsonObject data; data["param"] = "unbend";
            throw AgentBridgeError{-32602, "unbend must be a boolean", data};
        }
        stp.unbend = p.value("unbend").toBool();
    }
    if (p.contains("unbendSmoothCols")) {
        if (!p.value("unbendSmoothCols").isDouble()) {
            QJsonObject data; data["param"] = "unbendSmoothCols";
            throw AgentBridgeError{-32602, "unbendSmoothCols must be a number", data};
        }
        stp.unbendSmoothCols = p.value("unbendSmoothCols").toDouble();
        if (!std::isfinite(stp.unbendSmoothCols) || stp.unbendSmoothCols < 0.0) {
            QJsonObject data; data["param"] = "unbendSmoothCols";
            throw AgentBridgeError{-32602, "unbendSmoothCols must be a finite value >= 0", data};
        }
    }
    if (p.contains("overlapPasses")) {
        if (!p.value("overlapPasses").isDouble()) {
            QJsonObject data; data["param"] = "overlapPasses";
            throw AgentBridgeError{-32602, "overlapPasses must be an integer", data};
        }
        stp.overlapPasses = p.value("overlapPasses").toInt();
        if (stp.overlapPasses < 0) {
            QJsonObject data; data["param"] = "overlapPasses";
            throw AgentBridgeError{-32602, "overlapPasses must be >= 0", data};
        }
    }
    if (p.contains("orthogonalize")) {
        if (!p.value("orthogonalize").isBool()) {
            QJsonObject data; data["param"] = "orthogonalize";
            throw AgentBridgeError{-32602, "orthogonalize must be a boolean", data};
        }
        stp.orthogonalize = p.value("orthogonalize").toBool();
    }
    if (p.contains("trim")) {
        if (!p.value("trim").isBool()) {
            QJsonObject data; data["param"] = "trim";
            throw AgentBridgeError{-32602, "trim must be a boolean", data};
        }
        stp.trim = p.value("trim").toBool();
    }
    if (p.contains("trimMaxEdge")) {
        if (!p.value("trimMaxEdge").isDouble()) {
            QJsonObject data; data["param"] = "trimMaxEdge";
            throw AgentBridgeError{-32602, "trimMaxEdge must be a number", data};
        }
        stp.trimMaxEdge = p.value("trimMaxEdge").toDouble();
        if (!std::isfinite(stp.trimMaxEdge) || stp.trimMaxEdge < 0.0) {
            QJsonObject data; data["param"] = "trimMaxEdge";
            throw AgentBridgeError{-32602, "trimMaxEdge must be a finite value >= 0", data};
        }
    }
    stp.outputDir = p.value("outputDir").toString();

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.straighten"), QStringLiteral("Straighten"), segmentId,
        [handler, segmentId, stp](QString* err, QString* outDir) {
            return handler && handler->startStraighten(segmentId.toStdString(), stp, err, outDir);
        });
}

// NOTE: The deferred-response mechanism below (beginDeferred /
// completeDeferredResult / completeDeferredError, plus the AgentBridgeDeferred
// dispatch path) is bridge-core infrastructure per SPEC §8.4. It is proven out
// but not yet wired to a shipping method; the first real users (lasagna.* /
// atlas.* / fiber.create_atlas) arrive in later stages. It was validated
// during this stage against a temporary debug.defer self-test RPC (since
// removed) covering both the signal-completion and timeout (-32005) paths.

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
    int steps = 8;
    if (p.contains("steps")) {
        const QJsonValue sv = p.value("steps");
        const double sd = sv.toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!sv.isDouble() || !std::isfinite(sd) || std::floor(sd) != sd) {
            QJsonObject data;
            data["param"] = "steps";
            throw AgentBridgeError{-32602, "steps must be an integer", data};
        }
        steps = static_cast<int>(sd);
        if (steps < 1) {
            QJsonObject data;
            data["param"] = "steps";
            data["value"] = steps;
            throw AgentBridgeError{-32602, "steps must be >= 1", data};
        }
        if (steps > 256)
            steps = 256;
    }

    const QString space = p.value("space").toString(QStringLiteral("volume"));

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
            return QPointF(o.value("x").toDouble(), o.value("y").toDouble());
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

// ---------------------------------------------------------------------------
// Deferred responses (SPEC §8.4)
// ---------------------------------------------------------------------------

int AgentBridgeServer::beginDeferred(int timeoutMs, const QString& signalDesc)
{
    // At most one deferred call per (connection, method) may be in flight.
    for (const auto& pd : _pendingDeferred) {
        if (pd.socket.data() == _currentSocket && pd.method == _currentMethod) {
            QJsonObject data;
            data["detail"] = "deferred call already pending";
            data["method"] = _currentMethod;
            throw AgentBridgeError{-32004, "Deferred call already pending", data};
        }
    }

    const int token = _nextDeferredToken++;
    PendingDeferred pd;
    pd.socket = _currentSocket;
    pd.id = _currentRequestId;
    pd.method = _currentMethod;
    pd.signalDesc = signalDesc;
    pd.timer = new QTimer(this);
    pd.timer->setSingleShot(true);
    connect(pd.timer, &QTimer::timeout, this, [this, token, timeoutMs]() {
        auto it = _pendingDeferred.find(token);
        if (it == _pendingDeferred.end())
            return;
        const QString sig = it->signalDesc;
        QJsonObject data;
        data["detail"] = QStringLiteral("timed out after %1 ms waiting for %2")
                             .arg(timeoutMs).arg(sig);
        completeDeferredError(token, -32005, "Deferred response timed out", data);
    });
    _pendingDeferred.insert(token, pd);
    pd.timer->start(timeoutMs);
    return token;
}

void AgentBridgeServer::completeDeferredResult(int token, const QJsonObject& result)
{
    auto it = _pendingDeferred.find(token);
    if (it == _pendingDeferred.end())
        return;  // already completed or timed out
    PendingDeferred pd = it.value();
    _pendingDeferred.erase(it);
    if (pd.timer) {
        pd.timer->stop();
        pd.timer->deleteLater();
    }
    if (pd.socket && pd.socket->state() == QLocalSocket::ConnectedState)
        sendResponse(pd.socket.data(), pd.id, result);
}

void AgentBridgeServer::completeDeferredError(int token, int code, const QString& message,
                                              const QJsonObject& data)
{
    auto it = _pendingDeferred.find(token);
    if (it == _pendingDeferred.end())
        return;
    PendingDeferred pd = it.value();
    _pendingDeferred.erase(it);
    if (pd.timer) {
        pd.timer->stop();
        pd.timer->deleteLater();
    }
    if (pd.socket && pd.socket->state() == QLocalSocket::ConnectedState)
        sendError(pd.socket.data(), pd.id, code, message, data);
}
