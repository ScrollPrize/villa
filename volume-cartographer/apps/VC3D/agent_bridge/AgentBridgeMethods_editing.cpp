#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam boundedInteger(
    const QString& name,
    int defaultValue,
    int minimum,
    int maximum)
{
    AgentBridgeParam param = Params::optionalInteger(name, defaultValue);
    param.minimum = minimum;
    param.maximum = maximum;
    return param;
}

AgentBridgeParam nonNegativeNumber(const QString& name, double defaultValue)
{
    AgentBridgeParam param = Params::optionalNumber(name, defaultValue);
    param.minimum = 0.0;
    return param;
}

}  // namespace

void AgentBridgeServer::registerEditingHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("segmentation.enable_editing"),
            .params = {
                Params::requiredBoolean(QStringLiteral("enabled")),
            },
            .errors = {-32602, -32000, -32007, -32009},
            .mcp = Mcp::exact(QStringLiteral("vc3d_enable_editing")),
        },
        [this](const QJsonValue& p) {
            return handleSegmentationEnableEditing(p);
        });

    AgentBridgeParam steps = Params::requiredInteger(QStringLiteral("steps"));
    steps.minimum = 1;
    registerMethod(
        {
            .name = QStringLiteral("segmentation.grow"),
            .params = {
                // manual_add is parsed by the handler to preserve its specific
                // -32009 guidance instead of collapsing it into -32602.
                Params::optionalString(
                    QStringLiteral("method"),
                    QStringLiteral("tracer")),
                Params::optionalStringEnum(
                    QStringLiteral("direction"),
                    {
                        QStringLiteral("all"),
                        QStringLiteral("up"),
                        QStringLiteral("down"),
                        QStringLiteral("left"),
                        QStringLiteral("right"),
                        QStringLiteral("fill"),
                    },
                    QStringLiteral("all")),
                std::move(steps),
                Params::optionalBoolean(QStringLiteral("inpaintOnly"), false),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32007, -32008,
                -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_grow_segment"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentationGrow(p); });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.grow_patch_from_seed"),
            .params = {
                Params::requiredObject(
                    QStringLiteral("seed"),
                    {
                        Params::requiredNumber(QStringLiteral("x")),
                        Params::requiredNumber(QStringLiteral("y")),
                        Params::requiredNumber(QStringLiteral("z")),
                    }),
                Params::optionalString(QStringLiteral("volumeId")),
                boundedInteger(
                    QStringLiteral("iterations"),
                    200,
                    1,
                    100000),
                nonNegativeNumber(QStringLiteral("minAreaCm"), 0.002),
                Params::optionalString(QStringLiteral("outputDir")),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007,
                -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_grow_patch_from_seed"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) {
            return handleSegmentationGrowPatchFromSeed(p);
        });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.save"),
            .errors = {-32000, -32004},
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_save_segment"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentationSave(p); });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.manual_add.begin"),
            .errors = {-32000, -32004, -32005, -32007, -32008},
            .mcp = Mcp::exact(QStringLiteral("vc3d_manual_add_begin")),
        },
        [this](const QJsonValue& p) { return handleManualAddBegin(p); });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.manual_add.finish"),
            .params = {
                Params::optionalBoolean(QStringLiteral("apply"), true),
            },
            .errors = {-32602, -32000, -32007},
            .mcp = Mcp::exact(QStringLiteral("vc3d_manual_add_finish")),
        },
        [this](const QJsonValue& p) { return handleManualAddFinish(p); });

    registerMethod(
        {
            .name = QStringLiteral("segmentation.manual_add.set_line_mode"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("mode"),
                    {
                        QStringLiteral("vertical"),
                        QStringLiteral("horizontal"),
                        QStringLiteral("cross"),
                        QStringLiteral("cross_fill"),
                    }),
            },
            .errors = {-32602, -32000},
            .mcp =
                Mcp::exact(QStringLiteral("vc3d_manual_add_set_line_mode")),
        },
        [this](const QJsonValue& p) {
            return handleManualAddSetLineMode(p);
        });

    registerMethod(
        {
            .name =
                QStringLiteral("segmentation.manual_add.set_interpolation"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("mode"),
                    {
                        QStringLiteral("thin_plate_spline"),
                        QStringLiteral("tracer_restricted_to_fill"),
                    }),
            },
            .errors = {-32602, -32000},
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_manual_add_set_interpolation")),
        },
        [this](const QJsonValue& p) {
            return handleManualAddSetInterpolation(p);
        });

    registerMethod(
        {
            .name = QStringLiteral(
                "segmentation.manual_add.undo_constraint"),
            .errors = {-32000, -32007},
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_manual_add_undo_constraint")),
        },
        [this](const QJsonValue& p) {
            return handleManualAddUndoConstraint(p);
        });

    registerMethod(
        {
            .name =
                QStringLiteral("segmentation.corrections.set_point_mode"),
            .params = {
                Params::requiredBoolean(QStringLiteral("active")),
            },
            .errors = {-32602, -32000, -32004, -32007, -32008},
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_corrections_set_point_mode")),
        },
        [this](const QJsonValue& p) {
            return handleCorrectionsSetPointMode(p);
        });
}
