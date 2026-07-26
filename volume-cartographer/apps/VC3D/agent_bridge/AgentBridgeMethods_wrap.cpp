#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;

namespace {

AgentBridgeParam optionalViewer()
{
    return Params::nullable(
        Params::optionalString(QStringLiteral("viewer")));
}

}  // namespace


void AgentBridgeServer::registerWrapHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("wrap_annotation.set_mode"),
            .params = {
                Params::requiredBoolean(QStringLiteral("enabled")),
            },
            .errors = {-32602, -32000},
            .mcp = {.tool = QStringLiteral("vc3d_set_wrap_annotation_mode")},
        },
        [this](const QJsonValue& p) { return handleWrapAnnotationSetMode(p); });

    registerMethod(
        {
            .name = QStringLiteral("wrap_annotation.commit"),
            .params = {optionalViewer()},
            .errors = {-32602, -32000, -32002, -32009},
            .mcp = {.tool = QStringLiteral("vc3d_commit_wrap_annotation")},
        },
        [this](const QJsonValue& p) { return handleWrapAnnotationCommit(p); });

    registerMethod(
        {
            .name = QStringLiteral("wrap_annotation.undo"),
            .params = {optionalViewer()},
            .errors = {-32602, -32000, -32002, -32009},
            .mcp = {.tool = QStringLiteral("vc3d_undo_wrap_annotation")},
        },
        [this](const QJsonValue& p) { return handleWrapAnnotationUndo(p); });
}
