#include "agent_bridge/AgentBridgeServer.hpp"

#include <optional>
#include <vector>

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
                Params::optionalString(QStringLiteral("objName")),
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
            .name = QStringLiteral("segment.neighbor_copy"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::requiredStringEnum(
                    QStringLiteral("direction"),
                    {QStringLiteral("out"), QStringLiteral("in")}),
                Params::optionalString(QStringLiteral("volumeId")),
                Params::optionalString(QStringLiteral("outputTag")),
                boundedInteger(QStringLiteral("neighborMaxDistance"), 1, 500),
                boundedInteger(QStringLiteral("neighborMinClearance"), 1, 100),
                Params::optionalBoolean(QStringLiteral("neighborFill")),
                boundedInteger(QStringLiteral("neighborInterpWindow"), 1, 50),
                boundedInteger(QStringLiteral("generations"), 1, 10),
                boundedInteger(QStringLiteral("neighborSpikeWindow"), 1, 20),
                boundedInteger(QStringLiteral("resumeLocalOptStep"), 1, 512),
                boundedInteger(QStringLiteral("resumeLocalOptRadius"), 1, 2048),
                boundedInteger(QStringLiteral("resumeLocalMaxIters"), 1, 10000),
                Params::optionalBoolean(QStringLiteral("resumeLocalDenseQr")),
                boundedInteger(QStringLiteral("refineOmpThreads"), 1, 256),
                Params::optionalObject(QStringLiteral("refineParamOverrides"), {}),
            },
            .errors = {
                -32602, -32000, -32004, -32005, -32006, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_neighbor_copy_segment"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentNeighborCopy(p); });
}
