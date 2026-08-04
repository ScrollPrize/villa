#include "agent_bridge/AgentBridgeServer.hpp"

#include "vc/core/render/Colormaps.hpp"

namespace {

AgentBridgeParam optionalViewer()
{
    AgentBridgeParam param =
        AgentBridgeParams::optionalString(QStringLiteral("viewer"));
    param.nullable = true;
    return param;
}

AgentBridgeParam requiredVec3(const QString& name)
{
    AgentBridgeParam param = AgentBridgeParams::optionalObject(
        name,
        {
            AgentBridgeParams::requiredNumber(QStringLiteral("x")),
            AgentBridgeParams::requiredNumber(QStringLiteral("y")),
            AgentBridgeParams::requiredNumber(QStringLiteral("z")),
        });
    param.required = true;
    return param;
}

AgentBridgeParam optionalWindow(const QString& name)
{
    return AgentBridgeParams::optionalObject(
        name,
        {
            AgentBridgeParams::requiredNumber(QStringLiteral("low")),
            AgentBridgeParams::requiredNumber(QStringLiteral("high")),
        });
}

QStringList overlayColormapIds()
{
    QStringList ids{QString()};
    for (const vc::OverlayColormapSpec& spec : vc::specs())
        ids.append(QString::fromStdString(spec.id));
    return ids;
}

}  // namespace

namespace Params = AgentBridgeParams;


void AgentBridgeServer::registerViewerHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("viewer.center_on_point"),
            .params = {
                optionalViewer(),
                requiredVec3(QStringLiteral("point")),
                Params::optionalBoolean(QStringLiteral("forceRender"), true),
            },
            .errors = {-32602, -32003, -32002, -32001},
            .mcp = {
                .tool = QStringLiteral("vc3d_center_viewer"),
                .snakeCaseParams = true,
            },
        },
        [this](const QJsonValue& p) { return handleViewerCenterOnPoint(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.zoom"),
            .params = {
                optionalViewer(),
                Params::requiredPositiveNumber(QStringLiteral("factor")),
            },
            .errors = {-32602, -32002},
            .mcp = {.tool = QStringLiteral("vc3d_zoom_viewer")},
        },
        [this](const QJsonValue& p) { return handleViewerZoom(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.rotate"),
            .params = {
                // The handler also accepts case/whitespace-normalized spellings,
                // so an exact enum here would reject valid raw RPC inputs.
                Params::requiredString(QStringLiteral("plane")),
                Params::requiredNumber(QStringLiteral("degrees")),
                Params::optionalBoolean(QStringLiteral("relative"), true),
            },
            .errors = {-32602, -32002, -32000},
            .mcp = {.tool = QStringLiteral("vc3d_rotate_viewer")},
        },
        [this](const QJsonValue& p) { return handleViewerRotate(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.set_axis_aligned_slices"),
            .params = {Params::requiredBoolean(QStringLiteral("enabled"))},
            .errors = {-32602, -32000},
            .mcp = {.tool = QStringLiteral("vc3d_set_axis_aligned_slices")},
        },
        [this](const QJsonValue& p) { return handleViewerSetAxisAlignedSlices(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.get_render_settings"),
            .errors = {-32010},
            .mcp = {.tool = QStringLiteral("vc3d_get_render_settings")},
        },
        [this](const QJsonValue& p) { return handleViewerGetRenderSettings(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.set_render_settings"),
            .params = {
                Params::optionalNumber(QStringLiteral("intersectionOpacity")),
                Params::optionalNumber(QStringLiteral("intersectionThickness")),
                Params::optionalNumber(QStringLiteral("overlayOpacity")),
                Params::optionalInteger(QStringLiteral("intersectionMaxSurfaces")),
                Params::optionalBoolean(QStringLiteral("planeIntersectionLinesVisible")),
                Params::optionalBoolean(QStringLiteral("showSurfaceNormals")),
                Params::optionalBoolean(QStringLiteral("showDirectionHints")),
                Params::optionalBoolean(QStringLiteral("surfaceOverlayEnabled")),
                Params::optionalArray(
                    QStringLiteral("highlightedSurfaceIds"),
                    AgentBridgeParamType::String),
                optionalWindow(QStringLiteral("volumeWindow")),
                Params::optionalNumber(QStringLiteral("normalArrowLengthScale")),
                Params::optionalInteger(QStringLiteral("normalMaxArrows")),
                Params::optionalBoolean(QStringLiteral("segmentationCursorMirroring")),
                Params::optionalInteger(QStringLiteral("samplingStride")),
                Params::optionalNumber(QStringLiteral("zScrollSensitivity")),
            },
            .errors = {-32602, -32010},
            .mcp = {
                .tool = QStringLiteral("vc3d_set_render_settings"),
                .snakeCaseParams = true,
            },
        },
        [this](const QJsonValue& p) { return handleViewerSetRenderSettings(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.get_overlay"),
            .errors = {-32010},
            .mcp = {.tool = QStringLiteral("vc3d_get_overlay")},
        },
        [this](const QJsonValue& p) { return handleViewerGetOverlay(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.set_overlay"),
            .params = {
                Params::optionalString(QStringLiteral("volumeId")),
                Params::optionalBoolean(QStringLiteral("clear")),
                Params::optionalStringEnum(
                    QStringLiteral("colormap"),
                    overlayColormapIds()),
                Params::optionalNumber(QStringLiteral("opacity")),
                Params::optionalNumber(QStringLiteral("threshold")),
                optionalWindow(QStringLiteral("window")),
                Params::optionalInteger(QStringLiteral("maxDisplayedResolution")),
                Params::optionalObject(
                    QStringLiteral("composite"),
                    {
                        Params::optionalBoolean(QStringLiteral("enabled")),
                        Params::optionalStringEnum(
                            QStringLiteral("method"),
                            {
                                QStringLiteral("max"),
                                QStringLiteral("mean"),
                                QStringLiteral("min"),
                            }),
                        Params::optionalInteger(QStringLiteral("layersFront")),
                        Params::optionalInteger(QStringLiteral("layersBehind")),
                    }),
            },
            .errors = {-32602, -32010, -32007, -32000},
            .mcp = {
                .tool = QStringLiteral("vc3d_set_overlay"),
                .snakeCaseParams = true,
            },
        },
        [this](const QJsonValue& p) { return handleViewerSetOverlay(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.list_overlay_volumes"),
            .errors = {-32010, -32000},
            .mcp = {.tool = QStringLiteral("vc3d_list_overlay_volumes")},
        },
        [this](const QJsonValue& p) { return handleViewerListOverlayVolumes(p); });

    registerMethod(
        {
            .name = QStringLiteral("viewer.set_intersects"),
            .params = {
                Params::requiredArray(
                    QStringLiteral("surfaceIds"),
                    AgentBridgeParamType::String),
                optionalViewer(),
            },
            .errors = {-32602, -32010, -32009, -32002},
            .mcp = {
                .tool = QStringLiteral("vc3d_set_intersects"),
                .snakeCaseParams = true,
            },
        },
        [this](const QJsonValue& p) { return handleViewerSetIntersects(p); });
}
