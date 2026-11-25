#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam fiberId(const QString& name, bool required = true)
{
    AgentBridgeParam param = Params::optionalString(name);
    param.alternateTypes = {AgentBridgeParamType::Integer};
    param.required = required;
    return param;
}

AgentBridgeParam nonNegativeIndex(const QString& name)
{
    AgentBridgeParam param = Params::optionalInteger(name);
    param.minimum = 0;
    return param;
}

}  // namespace


void AgentBridgeServer::registerFiberHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("fiber.launch"),
            .params = {
                Params::nullable(
                    Params::optionalString(QStringLiteral("viewer"))),
                Params::requiredObject(
                    QStringLiteral("position"),
                    {
                        Params::requiredNumber(QStringLiteral("x")),
                        Params::requiredNumber(QStringLiteral("y")),
                        Params::optionalNumber(QStringLiteral("z")),
                    }),
                Params::optionalStringEnum(
                    QStringLiteral("space"),
                    {
                        QStringLiteral("volume"),
                        QStringLiteral("scene"),
                    },
                    QStringLiteral("volume")),
                Params::optionalBoolean(
                    QStringLiteral("replaceOwning"),
                    true),
            },
            .errors = {
                -32602, -32000, -32010, -32001, -32002, -32003, -32005, -32009,
            },
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_fiber_launch")),
        },
        [this](const QJsonValue& p) { return handleFiberLaunch(p); });

    registerMethod(
        {
            .name = QStringLiteral("fiber.list"),
            .errors = {-32000, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_fiber_list")),
        },
        [this](const QJsonValue& p) { return handleFiberList(p); });

    registerMethod(
        {
            .name = QStringLiteral("fiber.open"),
            .params = {
                fiberId(QStringLiteral("fiberId")),
                nonNegativeIndex(QStringLiteral("controlPointIndex")),
                nonNegativeIndex(QStringLiteral("linePointIndex")),
                Params::optionalArray(
                    QStringLiteral("span"),
                    nonNegativeIndex(QString())),
                Params::optionalBoolean(QStringLiteral("editable"), false),
            },
            .errors = {-32602, -32000, -32010, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_fiber_open")),
        },
        [this](const QJsonValue& p) { return handleFiberOpen(p); });

    // The approval window (docs/fiber_approval_workflow.md): opened on
    // the solve whose normals the fiber map handed over for the fiber.
    registerMethod(
        {
            .name = QStringLiteral("fiber.open_approval"),
            .params = {
                fiberId(QStringLiteral("fiberId")),
                Params::optionalString(QStringLiteral("run")),
                nonNegativeIndex(QStringLiteral("lineFrom")),
                nonNegativeIndex(QStringLiteral("lineTo")),
            },
            .errors = {-32602, -32000, -32010, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_fiber_open_approval")),
        },
        [this](const QJsonValue& p) { return handleFiberOpenApproval(p); });

    // One edit in the fiber's approval window: a stretch state (from, to,
    // aspect, state) or a normal point (normalPoint, optional normal, remove).
    registerMethod(
        {
            .name = QStringLiteral("fiber.approval_edit"),
            .params = {
                fiberId(QStringLiteral("fiberId")),
                Params::optionalNumber(QStringLiteral("from")),
                Params::optionalNumber(QStringLiteral("to")),
                Params::optionalStringEnum(
                    QStringLiteral("aspect"),
                    {QStringLiteral("geometry"), QStringLiteral("normals")},
                    QStringLiteral("geometry")),
                Params::optionalStringEnum(
                    QStringLiteral("state"),
                    {QStringLiteral("undefined"), QStringLiteral("approved"), QStringLiteral("broken")},
                    QStringLiteral("approved")),
                Params::optionalNumber(QStringLiteral("normalPoint")),
                Params::optionalArray(QStringLiteral("normal"), Params::optionalNumber(QString())),
                Params::optionalBoolean(QStringLiteral("remove"), false),
            },
            .errors = {-32602, -32000, -32010, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_fiber_approval_edit")),
        },
        [this](const QJsonValue& p) { return handleFiberApprovalEdit(p); });

    registerMethod(
        {
            .name = QStringLiteral("fiber.set_follow"),
            .params = {
                Params::requiredBoolean(QStringLiteral("enabled")),
            },
            .errors = {-32602, -32000, -32010, -32007},
            .mcp = Mcp::exact(QStringLiteral("vc3d_fiber_set_follow")),
        },
        [this](const QJsonValue& p) { return handleFiberSetFollow(p); });

    registerMethod(
        {
            .name = QStringLiteral("fiber.save"),
            .errors = {-32000, -32010, -32005},
            .mcp = Mcp::exact(QStringLiteral("vc3d_fiber_save")),
        },
        [this](const QJsonValue& p) { return handleFiberSave(p); });

    registerMethod(
        {
            .name = QStringLiteral("fiber.delete"),
            .params = {
                Params::requiredArray(
                    QStringLiteral("fiberIds"),
                    fiberId(QString(), false)),
            },
            .errors = {-32602, -32000, -32010, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_fiber_delete")),
        },
        [this](const QJsonValue& p) { return handleFiberDelete(p); });

    registerMethod(
        {
            .name = QStringLiteral("fiber.set_tag"),
            .params = {
                fiberId(QStringLiteral("fiberId")),
                Params::requiredString(QStringLiteral("tag")),
                Params::requiredBoolean(QStringLiteral("enabled")),
            },
            .errors = {-32602, -32000, -32010, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_fiber_set_tag")),
        },
        [this](const QJsonValue& p) { return handleFiberSetTag(p); });

    registerMethod(
        {
            .name = QStringLiteral("fiber.export"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
            },
            .errors = {-32602, -32000, -32010, -32005},
            .mcp = Mcp::exact(QStringLiteral("vc3d_fiber_export")),
        },
        [this](const QJsonValue& p) { return handleFiberExport(p); });
}
