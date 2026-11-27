#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

void AgentBridgeServer::registerStoresHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("objects.import"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("kind"),
                    {
                        QStringLiteral("surface"),
                        QStringLiteral("fiber"),
                        QStringLiteral("point_collection"),
                    }),
                Params::requiredString(QStringLiteral("path")),
                Params::optionalArray(QStringLiteral("tags"), AgentBridgeParamType::String),
                Params::optionalObject(
                    QStringLiteral("bind"),
                    {
                        Params::requiredPositiveNumber(QStringLiteral("scale")),
                    }),
                Params::optionalString(QStringLiteral("uuidNamespace")),
                Params::optionalString(QStringLiteral("store")),
            },
            .errors = {-32602, -32000, -32007, -32009, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_import_objects"), {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleObjectsImport(p); });

    registerMethod(
        {
            .name = QStringLiteral("objects.export"),
            .params = {
                Params::requiredString(QStringLiteral("uuid")),
                Params::requiredString(QStringLiteral("path")),
                Params::optionalString(QStringLiteral("store")),
                Params::optionalBoolean(QStringLiteral("overwrite"), false),
                Params::optionalBoolean(QStringLiteral("validOnly"), false),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_export_object"), {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleObjectsExport(p); });

    registerMethod(
        {
            .name = QStringLiteral("stores.attach"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
                Params::optionalBoolean(QStringLiteral("readOnly"), false),
            },
            .errors = {-32602, -32000, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_attach_store")),
        },
        [this](const QJsonValue& p) { return handleStoresAttach(p); });

    registerMethod(
        {
            .name = QStringLiteral("stores.create"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
            },
            .errors = {-32602, -32000, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_create_store")),
        },
        [this](const QJsonValue& p) { return handleStoresCreate(p); });

    registerMethod(
        {
            .name = QStringLiteral("stores.detach"),
            .params = {
                Params::requiredString(QStringLiteral("storeId")),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_detach_store")),
        },
        [this](const QJsonValue& p) { return handleStoresDetach(p); });

    registerMethod(
        {
            .name = QStringLiteral("stores.route"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("kind"),
                    {
                        QStringLiteral("surface"),
                        QStringLiteral("fiber"),
                        QStringLiteral("point_collection"),
                        QStringLiteral("umbilicus"),
                        QStringLiteral("volume_transform"),
                    }),
                Params::nullable(Params::requiredString(QStringLiteral("storeId"))),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_route_store")),
        },
        [this](const QJsonValue& p) { return handleStoresRoute(p); });

    registerMethod(
        {
            .name = QStringLiteral("project.save_as"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
            },
            .errors = {-32602, -32000, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_save_project_as")),
        },
        [this](const QJsonValue& p) { return handleProjectSaveAs(p); });

    registerMethod(
        {
            .name = QStringLiteral("project.attach_normal_grid"),
            .params = {
                Params::requiredString(QStringLiteral("location")),
            },
            .errors = {-32602, -32000, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_attach_normal_grid")),
        },
        [this](const QJsonValue& p) { return handleProjectAttachNormalGrid(p); });

    registerMethod(
        {
            .name = QStringLiteral("project.attach_lasagna_manifest"),
            .params = {
                Params::requiredString(QStringLiteral("location")),
                Params::requiredStringEnum(
                    QStringLiteral("role"),
                    {
                        QStringLiteral("regular"),
                        QStringLiteral("fiber"),
                    }),
            },
            .errors = {-32602, -32000, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_attach_lasagna_manifest"), {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleProjectAttachLasagnaManifest(p); });

    registerMethod(
        {
            .name = QStringLiteral("project.detach_entry"),
            .params = {
                Params::requiredString(QStringLiteral("location")),
            },
            .errors = {-32602, -32000, -32007, -32004, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_detach_project_entry")),
        },
        [this](const QJsonValue& p) { return handleProjectDetachEntry(p); });

    registerMethod(
        {
            .name = QStringLiteral("umbilicus.import"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
                [] {
                    AgentBridgeParam param = Params::optionalNumber(QStringLiteral("unitsPerVoxel"));
                    param.minimum = 0.0;
                    param.exclusiveMinimum = true;
                    return param;
                }(),
                Params::optionalBoolean(QStringLiteral("replace"), false),
            },
            .errors = {-32602, -32000, -32001, -32007, -32009, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_import_umbilicus")),
        },
        [this](const QJsonValue& p) { return handleUmbilicusImport(p); });

    registerMethod(
        {
            .name = QStringLiteral("umbilicus.remove"),
            .errors = {-32000, -32001, -32007, -32009, -32005},
            .mcp = Mcp::exact(QStringLiteral("vc3d_remove_umbilicus")),
        },
        [this](const QJsonValue& p) { return handleUmbilicusRemove(p); });
}
