#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;

namespace {

AgentBridgeMcp mcp(const QString& tool, bool snakeCaseParams = false)
{
    return {
        .tool = tool,
        .snakeCaseParams = snakeCaseParams,
    };
}

}  // namespace

void AgentBridgeServer::registerGraphHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("graph.status"),
            .errors = {-32602, -32000, -32004, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_graph_status")),
        },
        [this](const QJsonValue& p) { return handleGraphStatus(p); });

    registerMethod(
        {
            .name = QStringLiteral("graph.object"),
            .params = {
                Params::requiredString(QStringLiteral("uuid")),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_graph_object")),
        },
        [this](const QJsonValue& p) { return handleGraphObject(p); });

    registerMethod(
        {
            .name = QStringLiteral("graph.neighbours"),
            .params = {
                Params::requiredString(QStringLiteral("uuid")),
                Params::optionalString(QStringLiteral("edgeKind")),
                Params::optionalString(QStringLiteral("neighbourKind")),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_graph_neighbours")),
        },
        [this](const QJsonValue& p) { return handleGraphNeighbours(p); });

    registerMethod(
        {
            .name = QStringLiteral("graph.edges"),
            .params = {
                Params::requiredString(QStringLiteral("uuid")),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_graph_edges")),
        },
        [this](const QJsonValue& p) { return handleGraphEdges(p); });

    registerMethod(
        {
            .name = QStringLiteral("graph.component"),
            .params = {
                Params::requiredString(QStringLiteral("uuid")),
                Params::optionalArray(
                    QStringLiteral("edgeKinds"),
                    AgentBridgeParamType::String),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_graph_component")),
        },
        [this](const QJsonValue& p) { return handleGraphComponent(p); });
}
