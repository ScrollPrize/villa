#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;


void AgentBridgeServer::registerJobHandlers()
{
    const std::vector<AgentBridgeParam> params{
        Params::optionalString(QStringLiteral("jobId")),
        Params::optionalString(QStringLiteral("source")),
    };

    registerMethod(
        {
            .name = QStringLiteral("job.status"),
            .params = params,
            .errors = {-32602, -32007},
            .mcp = {
                .tool = QStringLiteral("vc3d_job_status"),
                .snakeCaseParams = true,
            },
        },
        [this](const QJsonValue& p) { return handleJobStatus(p); });

    registerMethod(
        {
            .name = QStringLiteral("job.cancel"),
            .params = params,
            .errors = {-32602, -32007, -32010, -32005},
            .mcp = {
                .tool = QStringLiteral("vc3d_cancel_job"),
                .snakeCaseParams = true,
            },
        },
        [this](const QJsonValue& p) { return handleJobCancel(p); });
}
