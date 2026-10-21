#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

void AgentBridgeServer::registerProjectHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("project.attach"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("kind"),
                    {
                        QStringLiteral("umbilicus"),
                        QStringLiteral("normal_grid"),
                        QStringLiteral("lasagna_manifest"),
                        QStringLiteral("afv"),
                    }),
                Params::requiredString(QStringLiteral("location")),
                Params::optionalBoolean(QStringLiteral("fiberInference")),
            },
            .errors = {-32602, -32000, -32001, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_attach_to_project"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleProjectAttach(p); });

    registerMethod(
        {
            .name = QStringLiteral("project.detach"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("kind"),
                    {
                        QStringLiteral("umbilicus"),
                        QStringLiteral("normal_grid"),
                        QStringLiteral("lasagna_manifest"),
                        QStringLiteral("afv"),
                        QStringLiteral("volume"),
                        QStringLiteral("segments"),
                    }),
                Params::optionalString(QStringLiteral("location")),
                Params::optionalBoolean(QStringLiteral("confirm"), false),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_detach_from_project")),
        },
        [this](const QJsonValue& p) { return handleProjectDetach(p); });

    registerMethod(
        {
            .name = QStringLiteral("project.save_as"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
                Params::optionalBoolean(QStringLiteral("overwrite"), false),
            },
            .errors = {-32602, -32000, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_save_project_as")),
        },
        [this](const QJsonValue& p) { return handleProjectSaveAs(p); });
}
