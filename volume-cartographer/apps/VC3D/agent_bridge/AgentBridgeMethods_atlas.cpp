#include "agent_bridge/AgentBridgeServer.hpp"

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;


void AgentBridgeServer::registerAtlasHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("atlas.open"),
            .params = {
                Params::requiredString(QStringLiteral("atlasDir")),
            },
            .errors = {-32602, -32007, -32005, -32000},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_atlas_open")),
        },
        [this](const QJsonValue& p) { return handleAtlasOpen(p); });

    registerMethod(
        {
            .name = QStringLiteral("atlas.status"),
            .mcp = Mcp::exact(QStringLiteral("vc3d_atlas_status")),
        },
        [this](const QJsonValue& p) { return handleAtlasStatus(p); });

    AgentBridgeParam maxDistance =
        Params::optionalNumber(QStringLiteral("maxDistance"));
    maxDistance.minimum = 0.0;
    registerMethod(
        {
            .name = QStringLiteral("atlas.search_start"),
            .params = {
                Params::optionalStringEnum(
                    QStringLiteral("mode"),
                    {
                        QStringLiteral("atlas_to_non_atlas"),
                        QStringLiteral("non_atlas_only"),
                    },
                    QStringLiteral("atlas_to_non_atlas")),
                Params::nullable(
                    Params::optionalArray(
                        QStringLiteral("requiredTags"),
                        AgentBridgeParamType::String)),
                Params::nullable(
                    Params::optionalArray(
                        QStringLiteral("excludedTags"),
                        AgentBridgeParamType::String)),
                maxDistance,
            },
            .errors = {-32602, -32004, -32007, -32005, -32000},
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_atlas_search_start"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleAtlasSearchStart(p); });

    registerMethod(
        {
            .name = QStringLiteral("atlas.search_cancel"),
            .errors = {-32007},
            .mcp = Mcp::exact(QStringLiteral("vc3d_atlas_search_cancel")),
        },
        [this](const QJsonValue& p) { return handleAtlasSearchCancel(p); });

    AgentBridgeParam offset =
        Params::optionalInteger(QStringLiteral("offset"), 0);
    offset.minimum = 0;
    AgentBridgeParam limit =
        Params::optionalInteger(QStringLiteral("limit"), 100);
    limit.minimum = 1;
    registerMethod(
        {
            .name = QStringLiteral("atlas.search_results"),
            .params = {offset, limit},
            .errors = {-32602, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_atlas_search_results")),
        },
        [this](const QJsonValue& p) { return handleAtlasSearchResults(p); });

    registerMethod(
        {
            .name = QStringLiteral("atlas.open_result"),
            .params = {
                Params::requiredInteger(QStringLiteral("index")),
            },
            .errors = {-32602, -32007, -32005, -32010},
            .mcp = Mcp::exact(QStringLiteral("vc3d_atlas_open_result")),
        },
        [this](const QJsonValue& p) { return handleAtlasOpenResult(p); });

    registerMethod(
        {
            .name = QStringLiteral("atlas.remap"),
            .errors = {-32007, -32005, -32000},
            .mcp = Mcp::exact(QStringLiteral("vc3d_atlas_remap")),
        },
        [this](const QJsonValue& p) { return handleAtlasRemap(p); });

    registerMethod(
        {
            .name = QStringLiteral("atlas.optimize_snap_candidates"),
            .errors = {-32007, -32005, -32000},
            .mcp = Mcp::exact(
                QStringLiteral("vc3d_atlas_optimize_snap_candidates")),
        },
        [this](const QJsonValue& p) {
            return handleAtlasOptimizeSnapCandidates(p);
        });
}
