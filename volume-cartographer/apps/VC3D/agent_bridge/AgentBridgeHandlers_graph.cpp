#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QFutureWatcher>
#include <QtConcurrent>
#include <QTimer>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <optional>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "GraphDbController.hpp"

namespace {

GraphDbController* requireGraph(CWindow* window)
{
    GraphDbController* graph = window ? window->graphDbController() : nullptr;
    if (!graph || !graph->isOpen())
        throw AgentBridgeError{-32000, "No object database is open (start VC3D with --object-db <path> or set objects/db_path)", {}};
    return graph;
}

QString qs(const std::string& s) { return QString::fromStdString(s); }

QJsonObject commonJson(const vc::objects::RecordCommon& r)
{
    QJsonObject o;
    o["uuid"] = qs(r.uuid);
    o["kind"] = qs(r.kind);
    o["generation"] = double(r.generation);
    o["createdBy"] = qs(r.createdBy);
    o["createdAt"] = qs(r.createdAt);
    o["createdHost"] = qs(r.createdHost);
    o["changedBy"] = qs(r.changedBy);
    o["changedAt"] = qs(r.changedAt);
    o["changedHost"] = qs(r.changedHost);
    o["approvalState"] = qs(r.approvalState);
    o["approvedBy"] = r.approvedBy ? QJsonValue(qs(*r.approvedBy)) : QJsonValue::Null;
    QJsonArray tags;
    for (const auto& t : r.tags) tags.append(qs(t));
    o["tags"] = tags;
    const QJsonDocument metaDoc = QJsonDocument::fromJson(QByteArray::fromStdString(r.meta));
    o["meta"] = metaDoc.isObject() ? QJsonValue(metaDoc.object()) : QJsonValue(QJsonObject{});
    return o;
}

QJsonObject objectJson(const vc::objects::ObjectRecord& r)
{
    QJsonObject o = commonJson(r);
    o["name"] = qs(r.name);
    o["formatVersion"] = double(r.formatVersion);
    o["storageKind"] = qs(r.storageKind);
    o["storageRef"] = qs(r.storageRef);
    return o;
}

QJsonObject anchorJson(const vc::objects::Anchor& a)
{
    QJsonObject o;
    if (a.index) o["index"] = double(*a.index);
    if (a.position) o["position"] = QJsonArray{(*a.position)[0], (*a.position)[1], (*a.position)[2]};
    if (a.direction) o["direction"] = QJsonArray{(*a.direction)[0], (*a.direction)[1], (*a.direction)[2]};
    return o;
}

QJsonObject edgeJson(const vc::objects::EdgeRecord& e)
{
    QJsonObject o = commonJson(e);
    o["a"] = qs(e.aUuid);
    o["b"] = qs(e.bUuid);
    o["aAnchor"] = anchorJson(e.aAnchor);
    o["bAnchor"] = anchorJson(e.bAnchor);
    return o;
}

std::string paramUuid(const QJsonValue& params)
{
    return params.toObject().value("uuid").toString().toStdString();
}

std::optional<std::string> optParam(const QJsonValue& params, const char* key)
{
    const QJsonObject p = params.toObject();
    if (!p.contains(key) || p.value(key).isNull()) return std::nullopt;
    return p.value(key).toString().toStdString();
}

void requireObject(vc::objects::ObjectStore& db, const vc::objects::Txn& read, const std::string& uuid)
{
    if (!db.objectByUuid(read, uuid))
        throw AgentBridgeError{-32007, QStringLiteral("Unknown object uuid: %1").arg(qs(uuid)), {}};
}

}  // namespace

QJsonObject AgentBridgeServer::deferGraphQuery(std::function<QJsonObject(const std::atomic_bool*)> query)
{
    struct Result { QJsonObject value; std::optional<AgentBridgeError> error; };
    const int token = beginDeferred(30000, "graph query");
    auto cancel = std::make_shared<std::atomic_bool>(false);
    auto* watcher = new QFutureWatcher<Result>(this);
    // Deferred timers are destroyed on timeout, disconnect, or completion.
    connect(_pendingDeferred[token].timer, &QObject::destroyed, watcher, [cancel] { cancel->store(true); });
    connect(watcher, &QObject::destroyed, [cancel] { cancel->store(true); });
    connect(watcher, &QFutureWatcher<Result>::finished, this, [this, token, watcher] {
        const auto result = watcher->result();
        watcher->deleteLater();
        if (result.error) completeDeferredError(token, result.error->code, result.error->message, result.error->data);
        else completeDeferredResult(token, result.value);
    });
    watcher->setFuture(QtConcurrent::run([query = std::move(query), cancel] {
        Result result;
        try { result.value = query(cancel.get()); }
        catch (const AgentBridgeError& error) { result.error = error; }
        catch (const std::exception& error) { result.error = AgentBridgeError{-32010, QString::fromUtf8(error.what()), {}}; }
        catch (...) { result.error = AgentBridgeError{-32010, "Unknown graph query error", {}}; }
        return result;
    }));
    throw AgentBridgeDeferred{};
}

QJsonObject AgentBridgeServer::handleGraphStatus(const QJsonValue&)
{
    GraphDbController* graph = _window ? _window->graphDbController() : nullptr;
    if (!graph || !graph->isOpen()) return {{"open", false}};
    const auto db = graph->readHandle();
    const auto path = graph->path();
    return deferGraphQuery([db, path](const std::atomic_bool*) {
        auto read = db->read();
        QJsonObject counts;
        for (const char* kind : {"fiber", "patch", "track"}) counts[kind] = double(db->objectCount(read, kind));
        return QJsonObject{{"open", true}, {"path", path}, {"objectCounts", counts}};
    });
}

QJsonObject AgentBridgeServer::handleGraphObject(const QJsonValue& params)
{
    const auto db = requireGraph(_window)->readHandle();
    const auto uuid = paramUuid(params);
    return deferGraphQuery([db, uuid](const std::atomic_bool*) {
        auto read = db->read();
        const auto object = db->objectByUuid(read, uuid);
        if (!object) throw AgentBridgeError{-32007, QStringLiteral("Unknown object uuid: %1").arg(qs(uuid)), {}};
        return QJsonObject{{"object", objectJson(*object)}};
    });
}

QJsonObject AgentBridgeServer::handleGraphNeighbours(const QJsonValue& params)
{
    const auto db = requireGraph(_window)->readHandle();
    const auto uuid = paramUuid(params);
    const auto edgeKind = optParam(params, "edgeKind"), neighbourKind = optParam(params, "neighbourKind");
    return deferGraphQuery([db, uuid, edgeKind, neighbourKind](const std::atomic_bool*) {
        auto read = db->read();
        requireObject(*db, read, uuid);
        QJsonArray neighbours;
        for (const auto& n : db->neighbours(read, uuid, edgeKind, neighbourKind)) neighbours.append(qs(n));
        return QJsonObject{{"uuid", qs(uuid)}, {"neighbours", neighbours}};
    });
}

QJsonObject AgentBridgeServer::handleGraphEdges(const QJsonValue& params)
{
    const auto db = requireGraph(_window)->readHandle();
    const auto uuid = paramUuid(params);
    return deferGraphQuery([db, uuid](const std::atomic_bool*) {
        auto read = db->read();
        requireObject(*db, read, uuid);
        QJsonArray edges;
        for (const auto& edge : db->edgesOf(read, uuid)) edges.append(edgeJson(edge));
        return QJsonObject{{"uuid", qs(uuid)}, {"edges", edges}};
    });
}

QJsonObject AgentBridgeServer::handleGraphComponent(const QJsonValue& params)
{
    const auto db = requireGraph(_window)->readHandle();
    const auto uuid = paramUuid(params);
    std::vector<std::string> kinds;
    for (const auto& kind : params.toObject().value("edgeKinds").toArray()) kinds.push_back(kind.toString().toStdString());
    return deferGraphQuery([db, uuid, kinds](const std::atomic_bool* cancel) {
        auto read = db->read();
        requireObject(*db, read, uuid);
        QJsonArray members;
        for (const auto& member : db->componentOf(read, uuid, kinds, cancel)) members.append(qs(member));
        return QJsonObject{{"uuid", qs(uuid)}, {"component", members}};
    });
}
