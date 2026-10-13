#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

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
        throw AgentBridgeError{-32000, "No graph database is open (start VC3D with --graph-db <path> or set graph/db_path)", {}};
    return graph;
}

QString qs(const std::string& s) { return QString::fromStdString(s); }

QJsonObject commonJson(const vc::graph::RecordCommon& r)
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

QJsonObject objectJson(const vc::graph::ObjectRecord& r)
{
    QJsonObject o = commonJson(r);
    o["name"] = qs(r.name);
    o["formatVersion"] = double(r.formatVersion);
    o["storageKind"] = qs(r.storageKind);
    o["storageRef"] = qs(r.storageRef);
    return o;
}

QJsonObject anchorJson(const vc::graph::Anchor& a)
{
    QJsonObject o;
    if (a.index) o["index"] = double(*a.index);
    if (a.position) o["position"] = QJsonArray{(*a.position)[0], (*a.position)[1], (*a.position)[2]};
    if (a.direction) o["direction"] = QJsonArray{(*a.direction)[0], (*a.direction)[1], (*a.direction)[2]};
    return o;
}

QJsonObject edgeJson(const vc::graph::EdgeRecord& e)
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

void requireObject(GraphDbController* graph, const std::string& uuid)
{
    if (!graph->objectByUuid(uuid))
        throw AgentBridgeError{-32007, QStringLiteral("Unknown object uuid: %1").arg(qs(uuid)), {}};
}

}  // namespace

QJsonObject AgentBridgeServer::handleGraphStatus(const QJsonValue&)
{
    QJsonObject result;
    GraphDbController* graph = _window ? _window->graphDbController() : nullptr;
    result["open"] = graph && graph->isOpen();
    if (graph && graph->isOpen()) {
        result["path"] = graph->path();
        QJsonObject counts;
        for (const char* kind : {"fiber", "patch", "track"})
            counts[kind] = double(graph->objectsByKind(kind).size());
        result["objectCounts"] = counts;
    }
    return result;
}

QJsonObject AgentBridgeServer::handleGraphObject(const QJsonValue& params)
{
    GraphDbController* graph = requireGraph(_window);
    const std::string uuid = paramUuid(params);
    const auto obj = graph->objectByUuid(uuid);
    if (!obj)
        throw AgentBridgeError{-32007, QStringLiteral("Unknown object uuid: %1").arg(qs(uuid)), {}};
    QJsonObject result;
    result["object"] = objectJson(*obj);
    return result;
}

QJsonObject AgentBridgeServer::handleGraphNeighbours(const QJsonValue& params)
{
    GraphDbController* graph = requireGraph(_window);
    const std::string uuid = paramUuid(params);
    requireObject(graph, uuid);
    QJsonArray neighbours;
    for (const auto& n : graph->neighbours(uuid, optParam(params, "edgeKind"), optParam(params, "neighbourKind")))
        neighbours.append(qs(n));
    QJsonObject result;
    result["uuid"] = qs(uuid);
    result["neighbours"] = neighbours;
    return result;
}

QJsonObject AgentBridgeServer::handleGraphEdges(const QJsonValue& params)
{
    GraphDbController* graph = requireGraph(_window);
    const std::string uuid = paramUuid(params);
    requireObject(graph, uuid);
    QJsonArray edges;
    for (const auto& e : graph->edgesOf(uuid))
        edges.append(edgeJson(e));
    QJsonObject result;
    result["uuid"] = qs(uuid);
    result["edges"] = edges;
    return result;
}

QJsonObject AgentBridgeServer::handleGraphComponent(const QJsonValue& params)
{
    GraphDbController* graph = requireGraph(_window);
    const std::string uuid = paramUuid(params);
    requireObject(graph, uuid);
    std::vector<std::string> kinds;
    const QJsonObject p = params.toObject();
    if (p.contains("edgeKinds") && p.value("edgeKinds").isArray())
        for (const auto& k : p.value("edgeKinds").toArray())
            kinds.push_back(k.toString().toStdString());
    QJsonArray members;
    for (const auto& m : graph->componentOf(uuid, kinds))
        members.append(qs(m));
    QJsonObject result;
    result["uuid"] = qs(uuid);
    result["component"] = members;
    return result;
}
