#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam integerAtLeast(
    const QString& name,
    int defaultValue,
    int minimum)
{
    AgentBridgeParam param = Params::optionalInteger(name, defaultValue);
    param.minimum = minimum;
    return param;
}

AgentBridgeParam numberAtLeast(
    const QString& name,
    double defaultValue,
    double minimum)
{
    AgentBridgeParam param = Params::optionalNumber(name, defaultValue);
    param.minimum = minimum;
    return param;
}

AgentBridgeParam keepPercent()
{
    AgentBridgeParam param =
        Params::optionalNumber(QStringLiteral("keepPercent"), 100.0);
    param.minimum = 0.0;
    param.exclusiveMinimum = true;
    param.maximum = 100.0;
    return param;
}

}  // namespace

void AgentBridgeServer::registerFlattenHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("flatten.slim"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                integerAtLeast(QStringLiteral("iterations"), 50, 1),
                numberAtLeast(QStringLiteral("tolerance"), 0.0, 0.0),
                Params::nullable(
                    Params::optionalStringEnum(
                        QStringLiteral("energyType"),
                        {
                            QStringLiteral("symmetric_dirichlet"),
                            QStringLiteral("conformal"),
                        },
                        QStringLiteral("symmetric_dirichlet"))),
                keepPercent(),
                Params::optionalBoolean(
                    QStringLiteral("inpaintHoles"),
                    false),
                Params::optionalString(QStringLiteral("outputDir")),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007,
                -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_flatten_slim"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleFlattenSlim(p); });

    registerMethod(
        {
            .name = QStringLiteral("flatten.abf"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                integerAtLeast(QStringLiteral("iterations"), 10, 1),
                integerAtLeast(QStringLiteral("downsampleFactor"), 1, 1),
            },
            .errors = {
                -32602, -32000, -32004, -32005, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_flatten_abf"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleFlattenAbf(p); });

    registerMethod(
        {
            .name = QStringLiteral("flatten.straighten"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::optionalBoolean(QStringLiteral("unbend"), true),
                numberAtLeast(
                    QStringLiteral("unbendSmoothCols"),
                    300.0,
                    0.0),
                integerAtLeast(QStringLiteral("overlapPasses"), 2, 0),
                Params::optionalBoolean(
                    QStringLiteral("orthogonalize"),
                    true),
                Params::optionalBoolean(QStringLiteral("trim"), true),
                numberAtLeast(
                    QStringLiteral("trimMaxEdge"),
                    100.0,
                    0.0),
                Params::optionalString(QStringLiteral("outputDir")),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007,
                -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_flatten_straighten"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleFlattenStraighten(p); });
}
