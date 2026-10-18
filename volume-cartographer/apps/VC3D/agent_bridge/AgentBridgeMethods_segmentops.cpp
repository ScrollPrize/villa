#include "agent_bridge/AgentBridgeServer.hpp"

#include <optional>

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam boundedInteger(const QString& name, double minimum,
                                std::optional<double> maximum = std::nullopt)
{
    AgentBridgeParam param = Params::optionalInteger(name);
    param.minimum = minimum;
    if (maximum)
        param.maximum = *maximum;
    return param;
}

}  // namespace

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

    registerMethod(
        {
            .name = QStringLiteral("segment.export_obj"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::optionalString(QStringLiteral("outputPath")),
                Params::optionalBoolean(QStringLiteral("overwrite"), false),
                Params::optionalBoolean(QStringLiteral("normalizeUv"), false),
                Params::optionalBoolean(QStringLiteral("alignGrid"), false),
                boundedInteger(QStringLiteral("ompThreads"), 1),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_export_segment_obj"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentExportObj(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.rasterize"),
            .params = {
                Params::requiredArray(
                    QStringLiteral("segmentIds"),
                    Params::requiredString(QString())),
                boundedInteger(QStringLiteral("chunkSize"), 1, 4096),
                boundedInteger(QStringLiteral("zMin"), 0),
                boundedInteger(QStringLiteral("zMax"), -1),
            },
            .errors = {
                -32602, -32000, -32004, -32005, -32006, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_rasterize_segments"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentRasterize(p); });
}
