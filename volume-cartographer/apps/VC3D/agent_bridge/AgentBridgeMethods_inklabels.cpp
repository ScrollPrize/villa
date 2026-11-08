#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

// A label-space {x, y} or volume-space {x, y, z} point.
AgentBridgeParam labelPoint()
{
    return Params::optionalObject(
        QString(),
        {
            Params::requiredNumber(QStringLiteral("x")),
            Params::requiredNumber(QStringLiteral("y")),
            Params::optionalNumber(QStringLiteral("z")),
        });
}

AgentBridgeParam pointList()
{
    return Params::requiredArray(QStringLiteral("points"), labelPoint());
}

AgentBridgeParam boolean(const QString& name, bool defaultValue)
{
    return Params::optionalBoolean(name, defaultValue);
}

AgentBridgeParam integerIn(const QString& name, int minimum, int maximum)
{
    AgentBridgeParam param = Params::optionalInteger(name);
    param.minimum = minimum;
    param.maximum = maximum;
    return param;
}

// A label-pixel rectangle that limits a fill, suggestion pass or cleanup;
// by default the visible part of the segment view, else the whole canvas.
AgentBridgeParam boundParam()
{
    return Params::optionalObject(
        QStringLiteral("bound"),
        {
            Params::requiredInteger(QStringLiteral("x")),
            Params::requiredInteger(QStringLiteral("y")),
            Params::requiredInteger(QStringLiteral("width")),
            Params::requiredInteger(QStringLiteral("height")),
        });
}

QStringList classNames()
{
    return {QStringLiteral("ink"), QStringLiteral("background"), QStringLiteral("validation")};
}

AgentBridgeParam spaceParam()
{
    return Params::optionalStringEnum(QStringLiteral("space"), {QStringLiteral("label"), QStringLiteral("volume")},
                                      QStringLiteral("label"));
}

AgentBridgeParam guideParam()
{
    return Params::optionalStringEnum(QStringLiteral("guide"), {QStringLiteral("prediction"), QStringLiteral("labels")});
}

AgentBridgeParam connectivityParam()
{
    AgentBridgeParam param = integerIn(QStringLiteral("connectivity"), 4, 8);
    return param;
}

}  // namespace


void AgentBridgeServer::registerInkLabelHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("ink_labels.state"),
            .errors = {-32000},
            .mcp = Mcp::exact(QStringLiteral("vc3d_ink_labels_state")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsState(p); });

    AgentBridgeParam version = Params::optionalInteger(QStringLiteral("version"));
    version.minimum = 1;
    version.maximum = 9999;
    registerMethod(
        {
            .name = QStringLiteral("ink_labels.open"),
            .params = {
                Params::optionalStringEnum(
                    QStringLiteral("recover"),
                    {
                        QStringLiteral("restore"),
                        QStringLiteral("discard"),
                        QStringLiteral("fail"),
                    },
                    QStringLiteral("fail")),
                boolean(QStringLiteral("readOnlyIfLocked"), true),
                version,
                boolean(QStringLiteral("startNewVersion"), false),
            },
            .errors = {-32602, -32000, -32007, -32009, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_open")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsOpen(p); });

    AgentBridgeParam radius = Params::requiredNumber(QStringLiteral("radius"));
    radius.minimum = 0.5;
    radius.maximum = 512.0;
    AgentBridgeParam serial = Params::optionalSafeId(QStringLiteral("serial"));
    registerMethod(
        {
            .name = QStringLiteral("ink_labels.paint"),
            .params = {
                Params::optionalStringEnum(QStringLiteral("class"), classNames()),
                boolean(QStringLiteral("erase"), false),
                boolean(QStringLiteral("backgroundKeepsInk"), false),
                Params::optionalStringEnum(
                    QStringLiteral("space"),
                    {
                        QStringLiteral("label"),
                        QStringLiteral("volume"),
                    },
                    QStringLiteral("label")),
                Params::optionalObject(QStringLiteral("stroke"), {pointList(), radius}),
                Params::optionalObject(QStringLiteral("polygon"), {pointList()}),
                Params::optionalObject(
                    QStringLiteral("mask"),
                    {
                        Params::requiredString(QStringLiteral("path")),
                        Params::requiredObject(
                            QStringLiteral("origin"),
                            {
                                Params::requiredInteger(QStringLiteral("x")),
                                Params::requiredInteger(QStringLiteral("y")),
                            }),
                        // Absent means "paint".
                        Params::optionalStringEnum(QStringLiteral("format"), {QStringLiteral("paint"), QStringLiteral("values")}),
                    }),
                Params::optionalObject(
                    QStringLiteral("expect"),
                    {
                        Params::optionalString(QStringLiteral("segment")),
                        serial,
                    }),
            },
            .errors = {-32602, -32000, -32002, -32003, -32007, -32009},
            .mcp = {
                .tool = QStringLiteral("vc3d_ink_labels_paint"),
                .snakeCaseParams = true,
                .paramRenames = {
                    {QStringLiteral("class"), QStringLiteral("label_class")},
                },
            },
        },
        [this](const QJsonValue& p) { return handleInkLabelsPaint(p); });

    AgentBridgeParam extent = Params::requiredInteger(QStringLiteral("width"));
    extent.minimum = 1;
    extent.maximum = 4096;
    AgentBridgeParam height = extent;
    height.name = QStringLiteral("height");
    registerMethod(
        {
            .name = QStringLiteral("ink_labels.read"),
            .params = {
                Params::requiredObject(
                    QStringLiteral("rect"),
                    {
                        Params::requiredInteger(QStringLiteral("x")),
                        Params::requiredInteger(QStringLiteral("y")),
                        extent,
                        height,
                    }),
                Params::optionalStringEnum(
                    QStringLiteral("format"),
                    {
                        QStringLiteral("values"),
                        QStringLiteral("ink"),
                        QStringLiteral("supervision"),
                        QStringLiteral("validation"),
                    },
                    QStringLiteral("values")),
                Params::requiredString(QStringLiteral("path")),
            },
            .errors = {-32602, -32000, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_read")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsRead(p); });

    AgentBridgeParam steps = Params::optionalInteger(QStringLiteral("steps"), 1);
    steps.minimum = 1;
    steps.maximum = 1000;
    registerMethod(
        {
            .name = QStringLiteral("ink_labels.history"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("action"),
                    {
                        QStringLiteral("undo"),
                        QStringLiteral("redo"),
                    }),
                steps,
                boolean(QStringLiteral("force"), false),
            },
            .errors = {-32602, -32000, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_history")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsHistory(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.save"),
            .params = {
                boolean(QStringLiteral("overwriteExternalEdits"), false),
                boolean(QStringLiteral("asNewVersion"), false),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_save")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsSave(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.versions"),
            .errors = {-32000, -32007},
            .mcp = Mcp::exact(QStringLiteral("vc3d_ink_labels_versions")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsVersions(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.dataset_entry"),
            .errors = {-32000, -32007},
            .mcp = Mcp::exact(QStringLiteral("vc3d_ink_labels_dataset_entry")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsDatasetEntry(p); });

    AgentBridgeParam toolRadius = Params::optionalNumber(QStringLiteral("radius"));
    toolRadius.minimum = 0.5;
    toolRadius.maximum = 512.0;
    AgentBridgeParam opacity = Params::optionalNumber(QStringLiteral("opacity"));
    opacity.minimum = 0.0;
    opacity.maximum = 1.0;
    registerMethod(
        {
            .name = QStringLiteral("ink_labels.set_tool_state"),
            .params = {
                Params::optionalBoolean(QStringLiteral("labelMode")),
                Params::optionalStringEnum(QStringLiteral("class"), classNames()),
                Params::optionalBoolean(QStringLiteral("erase")),
                Params::optionalBoolean(QStringLiteral("backgroundKeepsInk")),
                Params::optionalStringEnum(QStringLiteral("tool"),
                                           {QStringLiteral("brush"), QStringLiteral("polygon"), QStringLiteral("fill")}),
                toolRadius,
                Params::optionalObject(
                    QStringLiteral("display"),
                    {
                        Params::optionalBoolean(QStringLiteral("show")),
                        Params::optionalBoolean(QStringLiteral("ink")),
                        Params::optionalBoolean(QStringLiteral("supervision")),
                        Params::optionalBoolean(QStringLiteral("validation")),
                        opacity,
                    }),
                Params::optionalObject(
                    QStringLiteral("assist"),
                    {
                        guideParam(),
                        integerIn(QStringLiteral("threshold"), 0, 255),
                        Params::optionalBoolean(QStringLiteral("above")),
                        connectivityParam(),
                        Params::optionalBoolean(QStringLiteral("unlabeledOnly")),
                        integerIn(QStringLiteral("median"), 1, 5),
                        integerIn(QStringLiteral("minArea"), 0, 100000000),
                        integerIn(QStringLiteral("cleanupSize"), 1, 256),
                    }),
            },
            .errors = {-32602, -32000, -32005, -32007, -32009},
            .mcp = {
                .tool = QStringLiteral("vc3d_ink_labels_set_tool_state"),
                .snakeCaseParams = true,
                .paramRenames = {
                    {QStringLiteral("class"), QStringLiteral("label_class")},
                },
            },
        },
        [this](const QJsonValue& p) { return handleInkLabelsSetToolState(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.close"),
            .errors = {-32000},
            .mcp = Mcp::exact(QStringLiteral("vc3d_ink_labels_close")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsClose(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.fill"),
            .params = {
                Params::requiredObject(
                    QStringLiteral("point"),
                    {
                        Params::requiredNumber(QStringLiteral("x")),
                        Params::requiredNumber(QStringLiteral("y")),
                        Params::optionalNumber(QStringLiteral("z")),
                    }),
                spaceParam(),
                Params::optionalStringEnum(QStringLiteral("class"), classNames(), QStringLiteral("ink")),
                boolean(QStringLiteral("erase"), false),
                boolean(QStringLiteral("backgroundKeepsInk"), false),
                guideParam(),
                integerIn(QStringLiteral("threshold"), 0, 255),
                Params::optionalBoolean(QStringLiteral("above")),
                connectivityParam(),
                Params::optionalBoolean(QStringLiteral("unlabeledOnly")),
                boundParam(),
            },
            .errors = {-32602, -32000, -32002, -32003, -32004, -32005, -32007, -32009},
            .mcp = {
                .tool = QStringLiteral("vc3d_ink_labels_fill"),
                .snakeCaseParams = true,
                .paramRenames = {
                    {QStringLiteral("class"), QStringLiteral("label_class")},
                },
            },
        },
        [this](const QJsonValue& p) { return handleInkLabelsFill(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.suggest"),
            .params = {
                integerIn(QStringLiteral("threshold"), 0, 255),
                Params::optionalBoolean(QStringLiteral("above")),
                integerIn(QStringLiteral("median"), 1, 5),
                integerIn(QStringLiteral("minArea"), 0, 100000000),
                boundParam(),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_suggest")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsSuggest(p); });

    AgentBridgeParam suggestionId = Params::optionalInteger(QString());
    suggestionId.minimum = 1;
    registerMethod(
        {
            .name = QStringLiteral("ink_labels.accept_suggestions"),
            .params = {
                Params::optionalArray(QStringLiteral("ids"), suggestionId),
                boolean(QStringLiteral("all"), false),
            },
            .errors = {-32602, -32000, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_accept_suggestions")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsAcceptSuggestions(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.discard_suggestions"),
            .errors = {-32000},
            .mcp = Mcp::exact(QStringLiteral("vc3d_ink_labels_discard_suggestions")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsDiscardSuggestions(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.cleanup"),
            .params = {
                Params::requiredStringEnum(QStringLiteral("kind"),
                                           {
                                               QStringLiteral("remove_specks"),
                                               QStringLiteral("fill_holes"),
                                               QStringLiteral("grow"),
                                               QStringLiteral("shrink"),
                                               QStringLiteral("background_ring"),
                                           }),
                integerIn(QStringLiteral("size"), 1, 256),
                boundParam(),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_cleanup")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsCleanup(p); });

    registerMethod(
        {
            .name = QStringLiteral("ink_labels.import"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
                Params::optionalStringEnum(QStringLiteral("label"),
                                           {QStringLiteral("ink"), QStringLiteral("supervision"), QStringLiteral("validation")}),
                boolean(QStringLiteral("allowResample"), false),
                Params::optionalStringEnum(QStringLiteral("legacyMaskAs"),
                                           {QStringLiteral("supervision"), QStringLiteral("ignore")}),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_ink_labels_import")),
        },
        [this](const QJsonValue& p) { return handleInkLabelsImport(p); });
}
