#include "agent_bridge/AgentBridgeServer.hpp"

#include <limits>

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam optionalFloat(const QString& name)
{
    AgentBridgeParam param = Params::optionalNumber(name);
    param.minimum = -static_cast<double>(std::numeric_limits<float>::max());
    param.maximum = static_cast<double>(std::numeric_limits<float>::max());
    return param;
}

AgentBridgeParam nonNegativeInteger(
    const QString& name,
    int defaultValue)
{
    AgentBridgeParam param = Params::optionalInteger(name, defaultValue);
    param.minimum = 0;
    return param;
}

AgentBridgeParam positiveNumber(
    const QString& name,
    const QJsonValue& defaultValue = QJsonValue(QJsonValue::Undefined))
{
    AgentBridgeParam param = Params::optionalNumber(name, defaultValue);
    param.minimum = 0.0;
    param.exclusiveMinimum = true;
    return param;
}

AgentBridgeParam positiveFloat(
    const QString& name,
    const QJsonValue& defaultValue = QJsonValue(QJsonValue::Undefined))
{
    AgentBridgeParam param = positiveNumber(name, defaultValue);
    param.maximum = static_cast<double>(std::numeric_limits<float>::max());
    return param;
}

}  // namespace

void AgentBridgeServer::registerSeedingHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("tags.set"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::requiredStringEnum(
                    QStringLiteral("tag"),
                    {
                        QStringLiteral("approved"),
                        QStringLiteral("defective"),
                        QStringLiteral("reviewed"),
                        QStringLiteral("inspect"),
                    }),
                Params::requiredBoolean(QStringLiteral("enabled")),
            },
            .errors = {-32602, -32000, -32007, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_set_segment_tag")),
        },
        [this](const QJsonValue& p) { return handleTagsSet(p); });

    registerMethod(
        {
            .name =
                QStringLiteral("seeding.set_winding_annotation_mode"),
            .params = {
                Params::requiredBoolean(QStringLiteral("active")),
            },
            .errors = {-32602, -32000, -32001, -32010},
            .mcp = Mcp::exact(
                QStringLiteral(
                    "vc3d_seeding_set_winding_annotation_mode")),
        },
        [this](const QJsonValue& p) {
            return handleSeedingSetWindingAnnotationMode(p);
        });

    registerMethod(
        {
            .name = QStringLiteral("seeding.preview_rays"),
            .errors = {-32000, -32001, -32007, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_seeding_preview_rays")),
        },
        [this](const QJsonValue& p) { return handleSeedingPreviewRays(p); });

    registerMethod(
        {
            .name = QStringLiteral("seeding.cast_rays"),
            .errors = {-32000, -32001, -32007, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_seeding_cast_rays")),
        },
        [this](const QJsonValue& p) { return handleSeedingCastRays(p); });

    registerMethod(
        {
            .name = QStringLiteral("seeding.reset_points"),
            .errors = {-32000, -32001, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_seeding_reset_points")),
        },
        [this](const QJsonValue& p) { return handleSeedingResetPoints(p); });

    registerMethod(
        {
            .name = QStringLiteral("seeding.run"),
            .errors = {
                -32000, -32001, -32004, -32005, -32006, -32007, -32010,
            },
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_seeding_run"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSeedingRun(p); });

    registerMethod(
        {
            .name = QStringLiteral("seeding.expand"),
            .errors = {
                -32000, -32001, -32004, -32005, -32006, -32007, -32010,
            },
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_seeding_expand"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSeedingExpand(p); });

    registerMethod(
        {
            .name = QStringLiteral("seeding.cancel"),
            .errors = {-32007, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_seeding_cancel")),
        },
        [this](const QJsonValue& p) { return handleSeedingCancel(p); });

    registerMethod(
        {
            .name = QStringLiteral("seeding.analyze_paths"),
            .errors = {-32000, -32001, -32007, -32010},
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_seeding_analyze_paths")),
        },
        [this](const QJsonValue& p) {
            return handleSeedingAnalyzePaths(p);
        });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.push_pull.set_config"),
            .params = {
                optionalFloat(QStringLiteral("start")),
                optionalFloat(QStringLiteral("stop")),
                optionalFloat(QStringLiteral("step")),
                optionalFloat(QStringLiteral("low")),
                optionalFloat(QStringLiteral("high")),
                Params::optionalInteger(QStringLiteral("blurRadius")),
                Params::optionalInteger(QStringLiteral("computeScale")),
                optionalFloat(QStringLiteral("perVertexLimit")),
                Params::optionalBoolean(QStringLiteral("perVertex")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp =
                Mcp::snakeCase(QStringLiteral("vc3d_push_pull_set_config")),
        },
        [this](const QJsonValue& p) { return handlePushPullSetConfig(p); });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.push_pull.start"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("direction"),
                    {
                        QStringLiteral("push"),
                        QStringLiteral("pull"),
                    }),
                Params::optionalBoolean(QStringLiteral("alpha")),
            },
            .errors = {-32602, -32000, -32007, -32008, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_push_pull_start")),
        },
        [this](const QJsonValue& p) { return handlePushPullStart(p); });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.push_pull.stop"),
            .errors = {-32000},
            .mcp = Mcp::exact(QStringLiteral("vc3d_push_pull_stop")),
        },
        [this](const QJsonValue& p) { return handlePushPullStop(p); });

    registerMethod(
        {
            .name = QStringLiteral("tracer.run_trace"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::nullable(Params::optionalObject(
                    QStringLiteral("paramOverrides"),
                    {})),
                Params::optionalInteger(QStringLiteral("ompThreads")),
                Params::optionalString(QStringLiteral("outputDir")),
            },
            .errors = {
                -32602, -32000, -32004, -32005, -32006, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_run_trace"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleTracerRunTrace(p); });

    AgentBridgeParam numSlices =
        Params::optionalInteger(QStringLiteral("numSlices"), 1);
    numSlices.minimum = 1;
    registerMethod(
        {
            .name = QStringLiteral("render.tifxyz"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::requiredStringEnum(
                    QStringLiteral("outputFormat"),
                    {
                        QStringLiteral("zarr"),
                        QStringLiteral("tif_stack"),
                    }),
                Params::optionalString(QStringLiteral("volumeId")),
                Params::optionalString(QStringLiteral("outputDir")),
                positiveFloat(QStringLiteral("scale"), 1.0),
                nonNegativeInteger(QStringLiteral("groupIdx"), 0),
                std::move(numSlices),
                Params::nullable(
                    positiveNumber(QStringLiteral("voxelSize"))),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007,
                -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_render_tifxyz"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleRenderTifxyz(p); });
}
