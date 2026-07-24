#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;

namespace {

AgentBridgeMcp mcp(
    const QString& tool,
    bool snakeCaseParams = false,
    QStringList extraParams = {})
{
    return {
        .tool = tool,
        .snakeCaseParams = snakeCaseParams,
        .extraParams = std::move(extraParams),
    };
}

AgentBridgeParam catalogResources()
{
    return Params::optionalObject(
        QStringLiteral("resources"),
        {
            Params::optionalArray(
                QStringLiteral("volumeIds"),
                AgentBridgeParamType::String),
            Params::optionalArray(
                QStringLiteral("representationRefs"),
                AgentBridgeParamType::String),
            Params::optionalArray(
                QStringLiteral("kinds"),
                Params::optionalStringEnum(
                    QString(),
                    {
                        QStringLiteral("normal_grids"),
                        QStringLiteral("lasagna"),
                        QStringLiteral("prediction"),
                    })),
        });
}

AgentBridgeParam reviewFilter()
{
    return Params::optionalObject(
        QStringLiteral("filter"),
        {
            Params::optionalBoolean(QStringLiteral("unreviewed")),
            Params::optionalBoolean(QStringLiteral("approved")),
            Params::optionalBoolean(QStringLiteral("defective")),
            Params::optionalBoolean(QStringLiteral("hideDefective")),
            Params::optionalBoolean(QStringLiteral("reviewed")),
            Params::optionalBoolean(QStringLiteral("inspect")),
            Params::optionalBoolean(QStringLiteral("partialReview")),
        });
}

}  // namespace


void AgentBridgeServer::registerSessionHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("ping"),
            .mcp = mcp(QStringLiteral("vc3d_ping")),
        },
        [this](const QJsonValue& p) { return handlePing(p); });

    registerMethod(
        {
            .name = QStringLiteral("state.get"),
            .mcp = mcp(QStringLiteral("vc3d_get_state")),
        },
        [this](const QJsonValue& p) { return handleStateGet(p); });

    registerMethod(
        {
            .name = QStringLiteral("segments.list"),
            .params = {
                Params::optionalBoolean(QStringLiteral("onlyLoaded"), false),
            },
            .errors = {-32602, -32000},
            .mcp = mcp(QStringLiteral("vc3d_list_segments"), true),
        },
        [this](const QJsonValue& p) { return handleSegmentsList(p); });

    registerMethod(
        {
            .name = QStringLiteral("segments.activate"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
            },
            .errors = {-32602, -32000, -32010, -32004, -32007, -32005},
            .mcp = mcp(
                QStringLiteral("vc3d_activate_segment"),
                true,
                {QStringLiteral("auto_fetch")}),
        },
        [this](const QJsonValue& p) { return handleSegmentsActivate(p); });

    registerMethod(
        {
            .name = QStringLiteral("segments.fetch"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
            },
            .errors = {-32602, -32000, -32010, -32004, -32007},
            .mcp = mcp(
                QStringLiteral("vc3d_fetch_segment"),
                true,
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentsFetch(p); });

    registerMethod(
        {
            .name = QStringLiteral("segments.delete"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::optionalBoolean(QStringLiteral("confirm"), false),
            },
            .errors = {-32602, -32000, -32004, -32007, -32010},
            .mcp = mcp(QStringLiteral("vc3d_delete_segment"), true),
        },
        [this](const QJsonValue& p) { return handleSegmentsDelete(p); });

    registerMethod(
        {
            .name = QStringLiteral("segments.rename"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::requiredString(QStringLiteral("newName")),
            },
            .errors = {-32602, -32000, -32004, -32007, -32010},
            .mcp = mcp(QStringLiteral("vc3d_rename_segment"), true),
        },
        [this](const QJsonValue& p) { return handleSegmentsRename(p); });

    registerMethod(
        {
            .name = QStringLiteral("segments.review"),
            .params = {
                Params::optionalBoolean(QStringLiteral("onlyLoaded"), false),
                reviewFilter(),
            },
            .errors = {-32602, -32000},
            .mcp = mcp(QStringLiteral("vc3d_review_segments"), true),
        },
        [this](const QJsonValue& p) { return handleSegmentsReview(p); });

    registerMethod(
        {
            .name = QStringLiteral("screenshot.capture"),
            .params = {
                Params::optionalString(QStringLiteral("target"), QStringLiteral("window")),
                Params::optionalString(QStringLiteral("filePath")),
                Params::optionalInteger(QStringLiteral("maxDim")),
            },
            .errors = {-32602, -32002, -32009, -32005},
            .mcp = mcp(QStringLiteral("vc3d_screenshot"), true),
        },
        [this](const QJsonValue& p) { return handleScreenshotCapture(p); });

    registerMethod(
        {
            .name = QStringLiteral("project.create"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
                Params::requiredString(QStringLiteral("volume")),
                Params::optionalString(QStringLiteral("name")),
                Params::optionalArray(
                    QStringLiteral("tags"),
                    AgentBridgeParamType::String),
                Params::optionalBoolean(QStringLiteral("overwrite"), false),
            },
            .errors = {-32602, -32007, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_create_project")),
        },
        [this](const QJsonValue& p) { return handleProjectCreate(p); });

    registerMethod(
        {
            .name = QStringLiteral("volume.open"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
                Params::optionalString(QStringLiteral("volumeId")),
            },
            .errors = {-32602, -32005, -32007, -32010},
            .mcp = mcp(QStringLiteral("vc3d_open_volume"), true),
        },
        [this](const QJsonValue& p) { return handleVolumeOpen(p); });

    registerMethod(
        {
            .name = QStringLiteral("volume.attach"),
            .params = {
                Params::requiredString(QStringLiteral("location")),
                Params::optionalArray(
                    QStringLiteral("tags"),
                    AgentBridgeParamType::String),
            },
            .errors = {-32602, -32000, -32004, -32007, -32005, -32010},
            .mcp = mcp(
                QStringLiteral("vc3d_attach_volume"),
                false,
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleVolumeAttach(p); });

    registerMethod(
        {
            .name = QStringLiteral("volume.select"),
            .params = {
                Params::requiredString(QStringLiteral("volumeId")),
            },
            .errors = {-32602, -32000, -32007, -32005},
            .mcp = mcp(QStringLiteral("vc3d_select_volume"), true),
        },
        [this](const QJsonValue& p) { return handleVolumeSelect(p); });

    registerMethod(
        {
            .name = QStringLiteral("volume.list"),
            .errors = {-32000},
            .mcp = mcp(QStringLiteral("vc3d_list_volumes")),
        },
        [this](const QJsonValue& p) { return handleVolumeList(p); });

    registerMethod(
        {
            .name = QStringLiteral("catalog.open_sample"),
            .params = {
                Params::requiredString(QStringLiteral("sampleId")),
                catalogResources(),
            },
            .errors = {-32602, -32007, -32005, -32010, -32004},
            .mcp = mcp(
                QStringLiteral("vc3d_open_catalog_sample"),
                true,
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleCatalogOpenSample(p); });

    registerMethod(
        {
            .name = QStringLiteral("catalog.list_samples"),
            .params = {
                Params::optionalBoolean(QStringLiteral("refresh"), false),
            },
            .errors = {-32602, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_list_catalog_samples")),
        },
        [this](const QJsonValue& p) { return handleCatalogListSamples(p); });

    registerMethod(
        {
            .name = QStringLiteral("catalog.describe_sample"),
            .params = {
                Params::requiredString(QStringLiteral("sampleId")),
                Params::optionalBoolean(QStringLiteral("refresh"), false),
            },
            .errors = {-32602, -32007, -32005, -32010},
            .mcp = mcp(QStringLiteral("vc3d_describe_catalog_sample"), true),
        },
        [this](const QJsonValue& p) { return handleCatalogDescribeSample(p); });
}
