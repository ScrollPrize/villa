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

#include "agent_bridge/AgentBridgeInternal.hpp"


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
    // seeding action the bridge invokes (preview_rays, cast_rays, and the batch
    // run/expand entry points) opens a precondition QMessageBox::warning on a
    // failed precondition; a static QMessageBox spins a nested event loop,
    // forbidden in a bridge handler. As with the line-annotation valve above, set
    // once for the bridge's lifetime (the bridge is opt-in via --agent-bridge, so
    // interactive sessions are unaffected). The batch actions (seeding.run /
    // seeding.expand) are now exposed: their former nested-processEvents wait was
    // refactored away (SeedingWidget::runSegmentationHeadless / runExpandSeedsHeadless
    // launch the QProcess batch and return; it drains through the process finished
    // callbacks and resolves via the seedingBatch* signals — see subscribeJobSignals).
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
    _handlers.insert("segments.fetch",
        [this](const QJsonValue& p) { return handleSegmentsFetch(p); });
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
    _handlers.insert("viewer.rotate",
        [this](const QJsonValue& p) { return handleViewerRotate(p); });
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
    _handlers.insert("seeding.run",
        [this](const QJsonValue& p) { return handleSeedingRun(p); });
    _handlers.insert("seeding.expand",
        [this](const QJsonValue& p) { return handleSeedingExpand(p); });
    _handlers.insert("seeding.cancel",
        [this](const QJsonValue& p) { return handleSeedingCancel(p); });
    _handlers.insert("seeding.analyze_paths",
        [this](const QJsonValue& p) { return handleSeedingAnalyzePaths(p); });
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
