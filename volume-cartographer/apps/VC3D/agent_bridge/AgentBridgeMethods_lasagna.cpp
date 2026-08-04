#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam optionalVec3(const QString& name)
{
    return Params::optionalObject(
        name,
        {
            Params::requiredNumber(QStringLiteral("x")),
            Params::requiredNumber(QStringLiteral("y")),
            Params::requiredNumber(QStringLiteral("z")),
        });
}

}  // namespace


void AgentBridgeServer::registerLasagnaHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("lasagna.service_status"),
            .mcp = Mcp::exact(QStringLiteral("vc3d_lasagna_service_status")),
        },
        [this](const QJsonValue& p) { return handleLasagnaServiceStatus(p); });

    registerMethod(
        {
            .name = QStringLiteral("lasagna.ensure_service"),
            .params = {
                Params::optionalString(QStringLiteral("pythonPath")),
                Params::optionalString(QStringLiteral("host")),
                Params::optionalInteger(QStringLiteral("port")),
            },
            .errors = {-32602, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_lasagna_ensure_service")),
        },
        [this](const QJsonValue& p) { return handleLasagnaEnsureService(p); });

    registerMethod(
        {
            .name = QStringLiteral("lasagna.list_datasets"),
            .errors = {-32005},
            .mcp = Mcp::exact(QStringLiteral("vc3d_lasagna_list_datasets")),
        },
        [this](const QJsonValue& p) { return handleLasagnaListDatasets(p); });

    registerMethod(
        {
            .name = QStringLiteral("lasagna.start_optimization"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("mode"),
                    {
                        QStringLiteral("reoptimize"),
                        QStringLiteral("new_model"),
                        QStringLiteral("offset"),
                        QStringLiteral("atlas"),
                    }),
                Params::optionalString(QStringLiteral("configPath")),
                optionalVec3(QStringLiteral("seed")),
                Params::optionalString(QStringLiteral("atlasPath")),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_lasagna_start_optimization"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleLasagnaStartOptimization(p); });

    registerMethod(
        {
            .name = QStringLiteral("lasagna.jobs"),
            .errors = {-32005},
            .mcp = Mcp::exact(QStringLiteral("vc3d_lasagna_jobs")),
        },
        [this](const QJsonValue& p) { return handleLasagnaJobs(p); });

    registerMethod(
        {
            .name = QStringLiteral("lasagna.cancel"),
            .params = {
                Params::optionalString(QStringLiteral("jobId")),
            },
            .errors = {-32602, -32007, -32005},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_lasagna_cancel")),
        },
        [this](const QJsonValue& p) { return handleLasagnaCancel(p); });

    registerMethod(
        {
            .name = QStringLiteral("lasagna.select_output_segment"),
            .params = {
                Params::requiredString(QStringLiteral("name")),
            },
            .errors = {-32602, -32000, -32007},
            .mcp = Mcp::exact(QStringLiteral("vc3d_lasagna_select_output")),
        },
        [this](const QJsonValue& p) {
            return handleLasagnaSelectOutputSegment(p);
        });

    registerMethod(
        {
            .name = QStringLiteral("lasagna.repeat_last"),
            .errors = {-32004, -32005, -32009, -32000},
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_lasagna_repeat_last"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleLasagnaRepeatLast(p); });

    registerMethod(
        {
            .name = QStringLiteral("workspace.switch"),
            .params = {
                Params::requiredStringEnum(
                    QStringLiteral("name"),
                    {
                        QStringLiteral("main"),
                        QStringLiteral("lasagna"),
                        QStringLiteral("fiber_slice"),
                    }),
            },
            .errors = {-32602, -32000},
            .mcp = Mcp::exact(QStringLiteral("vc3d_switch_workspace")),
        },
        [this](const QJsonValue& p) { return handleWorkspaceSwitch(p); });
}
