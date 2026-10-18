#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

void AgentBridgeServer::registerSegmentOpsHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("segment.transform"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::requiredStringEnum(
                    QStringLiteral("op"),
                    {
                        QStringLiteral("flip_u"),
                        QStringLiteral("flip_v"),
                        QStringLiteral("rotate90"),
                    }),
                Params::optionalBoolean(QStringLiteral("confirm"), false),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_transform_segment")),
        },
        [this](const QJsonValue& p) { return handleSegmentTransform(p); });
}
