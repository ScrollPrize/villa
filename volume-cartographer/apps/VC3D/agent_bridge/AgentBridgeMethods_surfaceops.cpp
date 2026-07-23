#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam nonNegativeInteger(
    const QString& name,
    int defaultValue)
{
    AgentBridgeParam param = Params::optionalInteger(name, defaultValue);
    param.minimum = 0;
    return param;
}

}  // namespace

void AgentBridgeServer::registerSurfaceOpsHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("segment.crop_bounds"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32007, -32009,
            },
            .mcp =
                Mcp::snakeCase(QStringLiteral("vc3d_crop_segment_bounds")),
        },
        [this](const QJsonValue& p) { return handleSegmentCropBounds(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.recalc_area"),
            .params = {
                Params::requiredArray(
                    QStringLiteral("segmentIds"),
                    Params::requiredString(QString())),
            },
            .errors = {-32602, -32000, -32001},
            .mcp =
                Mcp::snakeCase(QStringLiteral("vc3d_recalc_segment_area")),
        },
        [this](const QJsonValue& p) { return handleSegmentRecalcArea(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.reoptimize"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::optionalString(QStringLiteral("volumeId")),
                nonNegativeInteger(QStringLiteral("ompThreads"), 1),
                Params::optionalObject(
                    QStringLiteral("paramOverrides"),
                    {}),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007,
                -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_reoptimize_segment"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentReoptimize(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.refine_alpha_comp"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::optionalBoolean(QStringLiteral("refine"), true),
                Params::optionalNumber(QStringLiteral("start"), -6.0),
                Params::optionalNumber(QStringLiteral("stop"), 30.0),
                Params::optionalNumber(QStringLiteral("step"), 2.0),
                Params::optionalInteger(QStringLiteral("low"), 26),
                Params::optionalInteger(QStringLiteral("high"), 255),
                Params::optionalNumber(QStringLiteral("borderOff"), 1.0),
                Params::optionalInteger(QStringLiteral("radius"), 3),
                Params::optionalBoolean(
                    QStringLiteral("genVertexColor"),
                    false),
                Params::optionalBoolean(QStringLiteral("overwrite"), true),
                Params::optionalNumber(QStringLiteral("readerScale"), 0.5),
                Params::optionalString(
                    QStringLiteral("scaleGroup"),
                    QStringLiteral("1")),
                Params::optionalInteger(QStringLiteral("ompThreads")),
                Params::optionalString(QStringLiteral("outputDir")),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007,
                -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_refine_segment_alpha_comp"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) {
            return handleSegmentRefineAlphaComp(p);
        });

    registerMethod(
        {
            .name = QStringLiteral("segment.generate_mask"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007},
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_generate_segment_mask")),
        },
        [this](const QJsonValue& p) { return handleSegmentGenerateMask(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.append_mask"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32007,
            },
            .mcp =
                Mcp::snakeCase(QStringLiteral("vc3d_append_segment_mask")),
        },
        [this](const QJsonValue& p) { return handleSegmentAppendMask(p); });
}
