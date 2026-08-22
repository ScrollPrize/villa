#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {
AgentBridgeParam opaqueObject(const QString& name, bool required = false)
{
    AgentBridgeParam param = Params::optionalObject(name, {});
    param.required = required;
    return param;
}

AgentBridgeParam positiveInteger(const QString& name)
{
    AgentBridgeParam param = Params::requiredInteger(name);
    param.minimum = 1;
    return param;
}
} // namespace

void AgentBridgeServer::registerSpiralHandlers()
{
    registerMethod(
        {.name = QStringLiteral("spiral.status"),
         .errors = {-32005},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_status"))},
        [this](const QJsonValue& p) { return handleSpiralStatus(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.profiles"),
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_list_profiles"))},
        [this](const QJsonValue& p) { return handleSpiralProfiles(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.connect"),
         .params = {Params::requiredString(QStringLiteral("profileId"))},
         .errors = {-32602, -32005, -32007},
         .mcp = Mcp::snakeCase(QStringLiteral("vc3d_spiral_connect"))},
        [this](const QJsonValue& p) { return handleSpiralConnect(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.disconnect"),
         .params = {Params::optionalBoolean(QStringLiteral("force"), false)},
         .errors = {-32005, -32010},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_disconnect"))},
        [this](const QJsonValue& p) { return handleSpiralDisconnect(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.reconnect"),
         .errors = {-32004, -32005},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_reconnect"))},
        [this](const QJsonValue& p) { return handleSpiralReconnect(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.dataset"),
         .errors = {-32005},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_get_dataset"))},
        [this](const QJsonValue& p) { return handleSpiralDataset(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.rebuild"),
         .params = {Params::requiredBoolean(QStringLiteral("confirm")),
                    Params::optionalBoolean(QStringLiteral("defaults")),
                    opaqueObject(QStringLiteral("request"))},
         .errors = {-32602, -32005},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_rebuild"))},
        [this](const QJsonValue& p) { return handleSpiralRebuild(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.run"),
         .params = {positiveInteger(QStringLiteral("iterations")),
                    opaqueObject(QStringLiteral("influenceConfig")),
                    opaqueObject(QStringLiteral("runConfig")),
                    opaqueObject(QStringLiteral("inputs"))},
         .errors = {-32602, -32005},
         .mcp = Mcp::snakeCase(QStringLiteral("vc3d_spiral_run"))},
        [this](const QJsonValue& p) { return handleSpiralRun(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.stop"),
         .errors = {-32004, -32005},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_stop"))},
        [this](const QJsonValue& p) { return handleSpiralStop(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.preview_export"),
         .errors = {-32004, -32005},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_export_preview"))},
        [this](const QJsonValue& p) { return handleSpiralPreviewExport(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.checkpoint_save"),
         .params = {Params::requiredString(QStringLiteral("name"))},
         .errors = {-32602, -32005},
         .mcp = Mcp::snakeCase(QStringLiteral("vc3d_spiral_save_checkpoint"))},
        [this](const QJsonValue& p) { return handleSpiralCheckpointSave(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.checkpoint_download"),
         .params = {Params::requiredString(QStringLiteral("localPath"))},
         .errors = {-32602, -32005, -32007},
         .mcp = Mcp::snakeCase(QStringLiteral("vc3d_spiral_download_checkpoint"))},
        [this](const QJsonValue& p) { return handleSpiralCheckpointDownload(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.checkpoint_load"),
         .params = {Params::optionalString(QStringLiteral("hostPath")),
                    Params::optionalString(QStringLiteral("localPath")),
                    Params::optionalBoolean(QStringLiteral("allowRebuild"), false)},
         .errors = {-32602, -32005, -32007, -32010},
         .mcp = Mcp::snakeCase(QStringLiteral("vc3d_spiral_load_checkpoint"))},
        [this](const QJsonValue& p) { return handleSpiralCheckpointLoad(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.input_upload"),
         .params = {Params::requiredStringEnum(
                        QStringLiteral("kind"),
                        {QStringLiteral("patch"), QStringLiteral("fiber"),
                         QStringLiteral("pcl")}),
                    Params::requiredString(QStringLiteral("localPath")),
                    Params::requiredString(QStringLiteral("inputId")),
                    Params::optionalString(QStringLiteral("role"))},
         .errors = {-32602, -32005, -32007},
         .mcp = Mcp::snakeCase(QStringLiteral("vc3d_spiral_upload_input"))},
        [this](const QJsonValue& p) { return handleSpiralInputUpload(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.input_remove"),
         .params = {Params::requiredStringEnum(
                        QStringLiteral("kind"),
                        {QStringLiteral("patch"), QStringLiteral("fiber"),
                         QStringLiteral("pcl")}),
                    Params::requiredString(QStringLiteral("inputId"))},
         .errors = {-32602, -32005, -32007},
         .mcp = Mcp::snakeCase(QStringLiteral("vc3d_spiral_remove_input"))},
        [this](const QJsonValue& p) { return handleSpiralInputRemove(p); });
    registerMethod(
        {.name = QStringLiteral("spiral.inputs_commit"),
         .errors = {-32005},
         .mcp = Mcp::exact(QStringLiteral("vc3d_spiral_commit_inputs"))},
        [this](const QJsonValue& p) { return handleSpiralInputsCommit(p); });
}
