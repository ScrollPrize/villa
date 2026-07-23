#include "agent_bridge/AgentBridgeServer.hpp"

#include <iterator>

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam point3(const QString& name, bool required)
{
    AgentBridgeParam param = Params::optionalObject(
        name,
        {
            Params::requiredNumber(QStringLiteral("x")),
            Params::requiredNumber(QStringLiteral("y")),
            Params::requiredNumber(QStringLiteral("z")),
        });
    param.required = required;
    return param;
}

std::vector<AgentBridgeParam> collectionSelector()
{
    return {
        Params::optionalString(QStringLiteral("collection")),
        Params::optionalSafeId(QStringLiteral("collectionId")),
    };
}

std::vector<AgentBridgeParam> collectionSelectorWith(
    std::vector<AgentBridgeParam> params)
{
    std::vector<AgentBridgeParam> result = collectionSelector();
    result.insert(
        result.end(),
        std::make_move_iterator(params.begin()),
        std::make_move_iterator(params.end()));
    return result;
}

AgentBridgeParam windingMode()
{
    return Params::requiredStringEnum(
        QStringLiteral("mode"),
        {
            QStringLiteral("none"),
            QStringLiteral("incremental"),
            QStringLiteral("decremental"),
            QStringLiteral("constant"),
        });
}

std::vector<int> collectionErrors(bool invalidParams = true)
{
    std::vector<int> errors{-32000, -32010, -32007};
    if (invalidParams)
        errors.push_back(-32602);
    return errors;
}

}  // namespace

void AgentBridgeServer::registerPointsHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("points.commit"),
            .params = {
                Params::requiredString(QStringLiteral("collection")),
                Params::requiredArray(
                    QStringLiteral("points"),
                    point3(QString(), false)),
                Params::optionalNumber(QStringLiteral("winding")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_commit_points")),
        },
        [this](const QJsonValue& p) { return handlePointsCommit(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.list"),
            .params = {
                Params::optionalString(QStringLiteral("collection")),
            },
            .errors = {-32602, -32007},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_list_points")),
        },
        [this](const QJsonValue& p) { return handlePointsList(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.add_collection"),
            .params = {
                Params::optionalString(QStringLiteral("name")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_add_point_collection")),
        },
        [this](const QJsonValue& p) { return handlePointsAddCollection(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.update_point"),
            .params = {
                Params::requiredSafeId(QStringLiteral("pointId")),
                point3(QStringLiteral("position"), false),
                Params::nullable(
                    Params::optionalNumber(QStringLiteral("winding"))),
            },
            .errors = {-32602, -32000, -32010, -32007},
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_update_point"),
                {QStringLiteral("clear_winding")}),
        },
        [this](const QJsonValue& p) { return handlePointsUpdatePoint(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.remove_point"),
            .params = {
                Params::requiredSafeId(QStringLiteral("pointId")),
            },
            .errors = {-32602, -32000, -32010, -32007},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_remove_point")),
        },
        [this](const QJsonValue& p) { return handlePointsRemovePoint(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.clear_collection"),
            .params = collectionSelector(),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_clear_point_collection")),
        },
        [this](const QJsonValue& p) { return handlePointsClearCollection(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.clear_all"),
            .errors = {-32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_clear_all_points")),
        },
        [this](const QJsonValue& p) { return handlePointsClearAll(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.rename_collection"),
            .params = collectionSelectorWith({
                Params::requiredString(QStringLiteral("newName")),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_rename_point_collection")),
        },
        [this](const QJsonValue& p) { return handlePointsRenameCollection(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.set_collection_color"),
            .params = collectionSelectorWith({
                Params::requiredArray(
                    QStringLiteral("color"),
                    AgentBridgeParam{.type = AgentBridgeParamType::Number, .finite = true}),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_set_point_collection_color")),
        },
        [this](const QJsonValue& p) { return handlePointsSetCollectionColor(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.set_collection_metadata"),
            .params = collectionSelectorWith({
                Params::requiredBoolean(QStringLiteral("absoluteWindingNumber")),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_set_point_collection_metadata")),
        },
        [this](const QJsonValue& p) { return handlePointsSetCollectionMetadata(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.set_collection_tag"),
            .params = collectionSelectorWith({
                Params::requiredString(QStringLiteral("key")),
                Params::requiredString(QStringLiteral("value")),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_set_point_collection_tag")),
        },
        [this](const QJsonValue& p) { return handlePointsSetCollectionTag(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.remove_collection_tag"),
            .params = collectionSelectorWith({
                Params::requiredString(QStringLiteral("key")),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_remove_point_collection_tag")),
        },
        [this](const QJsonValue& p) { return handlePointsRemoveCollectionTag(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.set_windings_linked"),
            .params = collectionSelectorWith({
                Params::requiredArray(
                    QStringLiteral("linkedCollectionIds"),
                    Params::requiredSafeId(QString())),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_set_point_windings_linked")),
        },
        [this](const QJsonValue& p) { return handlePointsSetWindingsLinked(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.auto_fill_windings"),
            .params = collectionSelectorWith({
                windingMode(),
                Params::optionalNumber(QStringLiteral("constant")),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_auto_fill_windings")),
        },
        [this](const QJsonValue& p) { return handlePointsAutoFillWindings(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.set_auto_fill_mode"),
            .params = collectionSelectorWith({
                windingMode(),
                Params::optionalNumber(QStringLiteral("constant")),
            }),
            .errors = collectionErrors(),
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_set_auto_fill_mode")),
        },
        [this](const QJsonValue& p) { return handlePointsSetAutoFillMode(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.reset_windings"),
            .errors = {-32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_reset_windings")),
        },
        [this](const QJsonValue& p) { return handlePointsResetWindings(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.apply_anchor_offset"),
            .params = {
                Params::requiredNumber(QStringLiteral("offsetX")),
                Params::requiredNumber(QStringLiteral("offsetY")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_apply_anchor_offset")),
        },
        [this](const QJsonValue& p) { return handlePointsApplyAnchorOffset(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.save_json"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_save_points_json")),
        },
        [this](const QJsonValue& p) { return handlePointsSaveJson(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.load_json"),
            .params = {
                Params::requiredString(QStringLiteral("path")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_load_points_json")),
        },
        [this](const QJsonValue& p) { return handlePointsLoadJson(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.save_segment_path"),
            .params = {
                Params::requiredString(QStringLiteral("segmentPath")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_save_points_segment_path")),
        },
        [this](const QJsonValue& p) { return handlePointsSaveSegmentPath(p); });

    registerMethod(
        {
            .name = QStringLiteral("points.load_segment_path"),
            .params = {
                Params::requiredString(QStringLiteral("segmentPath")),
            },
            .errors = {-32602, -32000, -32010},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_load_points_segment_path")),
        },
        [this](const QJsonValue& p) { return handlePointsLoadSegmentPath(p); });
}
