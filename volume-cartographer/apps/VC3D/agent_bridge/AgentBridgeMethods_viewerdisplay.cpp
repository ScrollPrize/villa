#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;

void AgentBridgeServer::registerViewerDisplayHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("viewer.get_ink_overlay"),
            .params = {Params::optionalBoolean(QStringLiteral("refresh"), false)},
            .errors = {-32010},
            .mcp = {.tool = QStringLiteral("vc3d_get_ink_overlay")},
        },
        [this](const QJsonValue& p) { return handleViewerGetInkOverlay(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.set_ink_overlay"),
            .params = {
                Params::nullable(Params::optionalString(QStringLiteral("path"))),
                [] {
                    AgentBridgeParam opacity =
                        Params::optionalNumber(QStringLiteral("opacity"));
                    opacity.minimum = 0.0;
                    opacity.maximum = 1.0;
                    return opacity;
                }(),
                Params::optionalString(QStringLiteral("colormap")),
                Params::optionalBoolean(QStringLiteral("flipHorizontal")),
                Params::optionalBoolean(QStringLiteral("flipVertical")),
            },
            .errors = {-32602, -32007, -32010},
            .mcp = {
                .tool = QStringLiteral("vc3d_set_ink_overlay"),
                .snakeCaseParams = true,
            },
        },
        [this](const QJsonValue& p) { return handleViewerSetInkOverlay(p); });
}
