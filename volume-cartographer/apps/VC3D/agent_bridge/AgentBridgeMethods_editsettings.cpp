#include "agent_bridge/AgentBridgeServer.hpp"

#include <vector>

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam brush(const QString& name, bool withStep = false)
{
    std::vector<AgentBridgeParam> properties{
        Params::optionalNumber(QStringLiteral("radius")),
        Params::optionalNumber(QStringLiteral("sigma")),
    };
    if (withStep)
        properties.push_back(Params::optionalNumber(QStringLiteral("step")));
    return Params::optionalObject(name, std::move(properties));
}

}  // namespace

void AgentBridgeServer::registerEditSettingsHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("segmentation.get_settings"),
            .errors = {-32009},
            .mcp = {.tool = QStringLiteral("vc3d_get_segmentation_settings")},
        },
        [this](const QJsonValue& p) { return handleSegmentationGetSettings(p); });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.set_settings"),
            .params = {
                brush(QStringLiteral("drag")),
                brush(QStringLiteral("pushPull"), true),
                Params::optionalNumber(QStringLiteral("editScale")),
                Params::optionalNumber(QStringLiteral("smoothingStrength")),
                Params::optionalInteger(QStringLiteral("smoothingIterations")),
                Params::optionalBoolean(QStringLiteral("showHoverMarker")),
                Params::optionalStringEnum(
                    QStringLiteral("growthMethod"),
                    {
                        QStringLiteral("tracer"),
                        QStringLiteral("corrections"),
                    }),
                Params::optionalInteger(QStringLiteral("growthSteps")),
                Params::optionalInteger(QStringLiteral("growthScale")),
                Params::optionalArray(
                    QStringLiteral("growthDirections"),
                    Params::requiredStringEnum(
                        QString(),
                        {
                            QStringLiteral("up"),
                            QStringLiteral("down"),
                            QStringLiteral("left"),
                            QStringLiteral("right"),
                        })),
                Params::optionalObject(
                    QStringLiteral("zRange"),
                    {
                        Params::requiredBoolean(QStringLiteral("enabled")),
                        Params::optionalInteger(QStringLiteral("min")),
                        Params::optionalInteger(QStringLiteral("max")),
                    }),
                // null, as get_settings reports when none is chosen, leaves
                // the growth volume unchanged.
                Params::nullable(Params::optionalString(QStringLiteral("growthVolumeId"))),
                Params::optionalArray(
                    QStringLiteral("directionFields"),
                    Params::optionalObject(
                        QString(),
                        {
                            Params::requiredString(QStringLiteral("path")),
                            Params::optionalStringEnum(
                                QStringLiteral("orientation"),
                                {
                                    QStringLiteral("normal"),
                                    QStringLiteral("horizontal"),
                                    QStringLiteral("vertical"),
                                }),
                            Params::optionalInteger(QStringLiteral("scale")),
                            Params::optionalNumber(QStringLiteral("weight")),
                        })),
            },
            .errors = {-32602, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_set_segmentation_settings")),
        },
        [this](const QJsonValue& p) { return handleSegmentationSetSettings(p); });
}
