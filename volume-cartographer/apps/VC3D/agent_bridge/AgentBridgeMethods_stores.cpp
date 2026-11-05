#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

void AgentBridgeServer::registerStoresHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("objects.import"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("kind"),
                    {
                        QStringLiteral("surface"),
                        QStringLiteral("fiber"),
                        QStringLiteral("point_collection"),
                    }),
                Params::requiredString(QStringLiteral("path")),
                Params::optionalArray(QStringLiteral("tags"), AgentBridgeParamType::String),
                Params::optionalObject(
                    QStringLiteral("bind"),
                    {
                        Params::requiredPositiveNumber(QStringLiteral("scale")),
                    }),
                Params::optionalString(QStringLiteral("uuidNamespace")),
                Params::optionalString(QStringLiteral("store")),
            },
            .errors = {-32602, -32000, -32007, -32009, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_import_objects"), {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleObjectsImport(p); });

    registerMethod(
        {
            .name = QStringLiteral("objects.export"),
            .params = {
                Params::requiredString(QStringLiteral("uuid")),
                Params::requiredString(QStringLiteral("path")),
                Params::optionalString(QStringLiteral("store")),
                Params::optionalBoolean(QStringLiteral("overwrite"), false),
                Params::optionalBoolean(QStringLiteral("validOnly"), false),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_export_object"), {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleObjectsExport(p); });

}
