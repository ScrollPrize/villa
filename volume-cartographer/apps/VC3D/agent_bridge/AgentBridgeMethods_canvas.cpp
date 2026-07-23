#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;

namespace {

AgentBridgeParam optionalViewer()
{
    return Params::nullable(
        Params::optionalString(QStringLiteral("viewer")));
}

AgentBridgeParam point(const QString& name)
{
    return Params::requiredObject(
        name,
        {
            Params::requiredNumber(QStringLiteral("x")),
            Params::requiredNumber(QStringLiteral("y")),
            Params::optionalNumber(QStringLiteral("z")),
        });
}

AgentBridgeParam optionalScene()
{
    return Params::nullable(
        Params::optionalObject(
            QStringLiteral("scene"),
            {
                Params::requiredNumber(QStringLiteral("x")),
                Params::requiredNumber(QStringLiteral("y")),
            }));
}

AgentBridgeParam optionalModifiers()
{
    return Params::nullable(
        Params::optionalArray(
            QStringLiteral("modifiers"),
            Params::optionalStringEnum(
                QString(),
                {
                    QStringLiteral("shift"),
                    QStringLiteral("ctrl"),
                    QStringLiteral("alt"),
                    QStringLiteral("meta"),
                    QStringLiteral("keypad"),
                })));
}

AgentBridgeParam button(bool allowNone = false)
{
    QStringList values{
        QStringLiteral("left"),
        QStringLiteral("right"),
        QStringLiteral("middle"),
    };
    if (allowNone)
        values.append(QStringLiteral("none"));
    return Params::nullable(
        Params::optionalStringEnum(
            QStringLiteral("button"),
            std::move(values),
            QStringLiteral("left")));
}

AgentBridgeParam space()
{
    return Params::optionalStringEnum(
        QStringLiteral("space"),
        {
            QStringLiteral("volume"),
            QStringLiteral("scene"),
        },
        QStringLiteral("volume"));
}

std::vector<AgentBridgeParam> clickParams()
{
    return {
        optionalViewer(),
        point(QStringLiteral("position")),
        space(),
        button(),
        optionalModifiers(),
    };
}

AgentBridgeMcp canvasMcp(const QString& tool)
{
    return {
        .tool = tool,
        .snakeCaseParams = true,
    };
}

}  // namespace


void AgentBridgeServer::registerCanvasHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("canvas.get_cursor_volume_point"),
            .params = {
                optionalViewer(),
                optionalScene(),
            },
            .errors = {-32602, -32002, -32009, -32001, -32003},
            .mcp = canvasMcp(QStringLiteral("vc3d_get_cursor_point")),
        },
        [this](const QJsonValue& p) { return handleCursorVolumePoint(p); });

    registerMethod(
        {
            .name = QStringLiteral("canvas.click"),
            .params = clickParams(),
            .errors = {-32602, -32002, -32009, -32001, -32003},
            .mcp = canvasMcp(QStringLiteral("vc3d_click")),
        },
        [this](const QJsonValue& p) {
            return handleCanvasClick(p, /*addShift=*/false);
        });

    registerMethod(
        {
            .name = QStringLiteral("canvas.shift_click"),
            .params = clickParams(),
            .errors = {-32602, -32002, -32009, -32001, -32003},
            .mcp = canvasMcp(QStringLiteral("vc3d_shift_click")),
        },
        [this](const QJsonValue& p) {
            return handleCanvasClick(p, /*addShift=*/true);
        });

    AgentBridgeParam steps =
        Params::optionalInteger(QStringLiteral("steps"), 8);
    steps.minimum = 1;
    registerMethod(
        {
            .name = QStringLiteral("canvas.drag"),
            .params = {
                optionalViewer(),
                point(QStringLiteral("from")),
                point(QStringLiteral("to")),
                space(),
                button(true),
                optionalModifiers(),
                steps,
            },
            .errors = {-32602, -32002, -32009, -32001, -32003},
            .mcp = {
                .tool = QStringLiteral("vc3d_drag"),
                .snakeCaseParams = true,
                .paramRenames = {
                    {
                        QStringLiteral("from"),
                        QStringLiteral("from_point"),
                    },
                    {
                        QStringLiteral("to"),
                        QStringLiteral("to_point"),
                    },
                },
            },
        },
        [this](const QJsonValue& p) { return handleCanvasDrag(p); });
}
