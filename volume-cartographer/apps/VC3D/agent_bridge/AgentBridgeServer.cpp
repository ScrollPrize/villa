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
#include <QSaveFile>
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
#include <utility>

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

#include "agent_bridge/AgentBridgeCommandError.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"


namespace {
// Upper bound on a single framed request (and on the un-framed read buffer that
// precedes a newline). One oversized client is reported + dropped without
// affecting the others.
constexpr int kMaxLineBytes = 1 * 1024 * 1024;  // 1 MiB

class ScopedLineAnnotationDialogSuppression {
public:
    explicit ScopedLineAnnotationDialogSuppression(LineAnnotationController* controller)
        : _controller(controller),
          _wasSuppressed(controller && controller->errorDialogsSuppressed())
    {
        if (_controller) {
            _controller->setErrorDialogsSuppressed(true);
        }
    }

    ~ScopedLineAnnotationDialogSuppression()
    {
        if (_controller) {
            _controller->setErrorDialogsSuppressed(_wasSuppressed);
        }
    }

private:
    LineAnnotationController* _controller;
    bool _wasSuppressed;
};

}  // namespace


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

    // Restrict the endpoint to the current user: the bridge grants full
    // control of the running app, so no other account should be able to connect.
    _server->setSocketOptions(QLocalServer::UserAccessOption);

    // Probe the name BEFORE listening: a successful connect means a LIVE bridge
    // already owns the endpoint, so we must refuse without touching it. We cannot
    // rely on a listen() failure to detect this -- on Linux/Qt, QLocalServer::listen()
    // will reclaim (unlink + rebind) a name whose socket is still served by another
    // process, returning success and stranding that server. Probing first is the
    // only reliable guard: a live socket name is never unlinked as stale.
    {
        QLocalSocket probe;
        probe.connectToServer(serverName);
        const bool live = probe.waitForConnected(200);
        probe.abort();
        if (live)
            return false;  // name collision with a running bridge; leave it alone
    }

    if (!_server->listen(serverName)) {
        // No live server answered, but the endpoint still refused: a stale socket
        // file from a crashed run. Remove once and retry.
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

    // Publish atomically: a QSaveFile writes to a temp sibling and
    // rename()s into place on commit, so a concurrent reader can never observe
    // (and reap) a half-written JSON file.
    QSaveFile f(filePath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return;
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
    if (!f.commit())
        return;

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
        // Reject a complete line whose own length exceeds the limit.
        if (line.size() > kMaxLineBytes) {
            sendError(socket, QJsonValue(QJsonValue::Null), -32600,
                      QStringLiteral("Invalid Request: request exceeds %1-byte limit")
                          .arg(kMaxLineBytes));
            _buffers.remove(socket);  // `buffer` dangles after this; do not touch it
            socket->disconnectFromServer();
            return;
        }
        if (line.trimmed().isEmpty())
            continue;
        processLine(socket, line);
    }

    // Bound the unterminated remainder: an oversized request that never sends a
    // newline must not grow memory without limit. Any residual over the bound is
    // a partial oversized line -- report it best-effort and drop only this client.
    if (buffer.size() > kMaxLineBytes) {
        sendError(socket, QJsonValue(QJsonValue::Null), -32600,
                  QStringLiteral("Invalid Request: request exceeds %1-byte limit")
                      .arg(kMaxLineBytes));
        _buffers.remove(socket);  // `buffer` dangles after this; do not touch it
        socket->disconnectFromServer();
        return;
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
    // guard does not leak.
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
        // Batch requests are not supported.
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

    const QJsonValue params = request.value("params");
    if (!params.isUndefined() && !params.isNull() && !params.isObject()) {
        if (!isNotification) {
            QJsonObject data;
            data["param"] = "params";
            sendError(socket, id, -32602, "params must be an object", data);
        }
        return;
    }

    auto it = _handlers.find(method);
    if (it == _handlers.end()) {
        if (!isNotification)
            sendError(socket, id, -32601, QStringLiteral("Method not found: %1").arg(method));
        return;
    }

    ScopedLineAnnotationDialogSuppression suppressLineAnnotationDialogs(
        _window && _window->_lineAnnotationController
            ? _window->_lineAnnotationController.get()
            : nullptr);

    // Per-request context: handlers that defer their response read
    // this to stash the caller. Cleared after the handler returns/throws.
    _currentSocket = socket;
    _currentRequestId = id;
    _currentMethod = method;

    QJsonObject result;
    try {
        auto description = _methodDescriptions.constFind(method);
        if (description != _methodDescriptions.cend())
            description.value().validate(params);
        result = it.value()(params);
    } catch (const AgentBridgeDeferred&) {
        // The handler took ownership of the reply; it will be written later by
        // a signal completion or the timeout. Send nothing now.
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


[[noreturn]] void AgentBridgeServer::throwCommandLaunchError(
    const CommandLaunchError& error,
    const QString& fallbackMessage,
    const QString& segmentId,
    const QString& source)
{
    throw commandLaunchErrorToBridgeError(error, fallbackMessage, segmentId, source);
}


// ---------------------------------------------------------------------------
// Viewer registry
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
    if (!ref.isUndefined() && !ref.isNull() && !ref.isString())
        throwParamError("viewer", QStringLiteral("must be a string"));

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

QJsonObject AgentBridgeServer::handleRpcDescribe(const QJsonValue& params)
{
    const QString prefix = params.toObject().value("prefix").toString();

    QStringList describedNames = _methodDescriptions.keys();
    std::sort(describedNames.begin(), describedNames.end());

    QJsonObject methods;
    int describedCount = 0;
    for (const QString& name : describedNames) {
        if (prefix.isEmpty() || name.startsWith(prefix)) {
            methods[name] = _methodDescriptions.value(name).describe();
            ++describedCount;
        }
    }

    QStringList registeredNames = _handlers.keys();
    std::sort(registeredNames.begin(), registeredNames.end());

    QJsonArray undocumented;
    int registeredCount = 0;
    for (const QString& name : registeredNames) {
        if (!prefix.isEmpty() && !name.startsWith(prefix))
            continue;
        ++registeredCount;
        if (!_methodDescriptions.contains(name))
            undocumented.append(name);
    }

    QJsonObject coverage;
    coverage["described"] = describedCount;
    coverage["registered"] = registeredCount;
    coverage["complete"] = describedCount == registeredCount;

    QJsonObject result;
    result["methods"] = methods;
    result["undocumented"] = undocumented;
    result["coverage"] = coverage;
    return result;
}


void AgentBridgeServer::registerMethod(AgentBridgeMethod method, Handler handler)
{
    const QString name = method.name;
    const QString definitionError = method.definitionError();
    if (!definitionError.isEmpty()) {
        qFatal(
            "Invalid agent bridge method description: %s",
            qPrintable(definitionError));
    }
    if (_handlers.contains(name))
        qFatal("Duplicate agent bridge method: %s", qPrintable(name));

    _methodDescriptions.insert(name, std::move(method));
    _handlers.insert(name, std::move(handler));
}


void AgentBridgeServer::registerHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("rpc.describe"),
            .params = {{
                .name = QStringLiteral("prefix"),
                .type = AgentBridgeParamType::String,
            }},
            .errors = {-32602},
        },
        [this](const QJsonValue& p) { return handleRpcDescribe(p); });
    registerSessionHandlers();
    registerCanvasHandlers();
    registerViewerHandlers();
    registerEditingHandlers();
    registerWrapHandlers();
    registerPointsHandlers();
    registerJobHandlers();

    registerLasagnaHandlers();
    registerAtlasHandlers();
    registerFiberHandlers();
    registerSeedingHandlers();

    registerFlattenHandlers();

    registerSurfaceOpsHandlers();
}


// ---------------------------------------------------------------------------
// Deferred responses
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
    if (!pd.id.isUndefined() &&
        pd.socket && pd.socket->state() == QLocalSocket::ConnectedState) {
        sendResponse(pd.socket.data(), pd.id, result);
    }
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
    if (!pd.id.isUndefined() &&
        pd.socket && pd.socket->state() == QLocalSocket::ConnectedState) {
        sendError(pd.socket.data(), pd.id, code, message, data);
    }
}
