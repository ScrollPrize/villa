#include "agent_bridge/AgentBridgeServer.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace Params = AgentBridgeParams;
namespace Mcp = AgentBridgeMcpTools;

namespace {

AgentBridgeParam boundedInteger(const QString& name, double minimum,
                                std::optional<double> maximum = std::nullopt)
{
    AgentBridgeParam param = Params::optionalInteger(name);
    param.minimum = minimum;
    if (maximum)
        param.maximum = *maximum;
    return param;
}

AgentBridgeParam boundedNumber(const QString& name, double minimum, double maximum)
{
    AgentBridgeParam param = Params::optionalNumber(name);
    param.minimum = minimum;
    param.maximum = maximum;
    return param;
}

// The RANSAC alignment tunables shared by vc_merge_tifxyz and vc_merge_patch,
// bounded like the dialogs' spin boxes.
void appendRansacParams(std::vector<AgentBridgeParam>& params)
{
    params.push_back(boundedInteger(QStringLiteral("ransacIters"), 1, 1000000));
    params.push_back(boundedNumber(QStringLiteral("ransacMinThresh"), 0.1, 1000.0));
    params.push_back(boundedNumber(QStringLiteral("ransacMaxThresh"), 0.1, 1000.0));
    params.push_back(boundedNumber(QStringLiteral("ransacMadK"), 0.1, 20.0));
    params.push_back(boundedInteger(QStringLiteral("ransacSeed"), 0));
    params.push_back(boundedInteger(QStringLiteral("anchorCap"), 0, 1000000));
    params.push_back(boundedInteger(QStringLiteral("ompThreads"), 1));
}

}  // namespace

void AgentBridgeServer::registerSegmentOpsHandlers()
{
    registerMethod(
        {
            .name = QStringLiteral("segment.transform"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::requiredStringEnum(
                    QStringLiteral("op"),
                    {
                        QStringLiteral("flip_u"),
                        QStringLiteral("flip_v"),
                        QStringLiteral("rotate90"),
                    }),
                Params::optionalBoolean(QStringLiteral("confirm"), false),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_transform_segment")),
        },
        [this](const QJsonValue& p) { return handleSegmentTransform(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.export_obj"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::optionalString(QStringLiteral("outputPath")),
                Params::optionalBoolean(QStringLiteral("overwrite"), false),
                Params::optionalBoolean(QStringLiteral("normalizeUv"), false),
                Params::optionalBoolean(QStringLiteral("alignGrid"), false),
                boundedInteger(QStringLiteral("ompThreads"), 1),
            },
            .errors = {
                -32602, -32000, -32001, -32004, -32005, -32006, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_export_segment_obj"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentExportObj(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.rasterize"),
            .params = {
                Params::requiredArray(
                    QStringLiteral("segmentIds"),
                    Params::requiredString(QString())),
                boundedInteger(QStringLiteral("chunkSize"), 1, 4096),
                boundedInteger(QStringLiteral("zMin"), 0),
                boundedInteger(QStringLiteral("zMax"), -1),
            },
            .errors = {
                -32602, -32000, -32004, -32005, -32006, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_rasterize_segments"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentRasterize(p); });

    registerMethod(
        {
            .name = QStringLiteral("segment.neighbor_copy"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::requiredStringEnum(
                    QStringLiteral("direction"),
                    {QStringLiteral("out"), QStringLiteral("in")}),
                Params::optionalString(QStringLiteral("volumeId")),
                Params::optionalString(QStringLiteral("outputDir")),
                boundedInteger(QStringLiteral("neighborMaxDistance"), 1, 500),
                boundedInteger(QStringLiteral("neighborMinClearance"), 1, 100),
                Params::optionalBoolean(QStringLiteral("neighborFill")),
                boundedInteger(QStringLiteral("neighborInterpWindow"), 1, 50),
                boundedInteger(QStringLiteral("generations"), 1, 10),
                boundedInteger(QStringLiteral("neighborSpikeWindow"), 1, 20),
                boundedInteger(QStringLiteral("resumeLocalOptStep"), 1, 512),
                boundedInteger(QStringLiteral("resumeLocalOptRadius"), 1, 2048),
                boundedInteger(QStringLiteral("resumeLocalMaxIters"), 1, 10000),
                Params::optionalBoolean(QStringLiteral("resumeLocalDenseQr")),
                boundedInteger(QStringLiteral("pass2OmpThreads"), 1, 256),
                Params::optionalObject(QStringLiteral("pass2ParamOverrides"), {}),
            },
            .errors = {
                -32602, -32000, -32004, -32005, -32006, -32007, -32009,
            },
            .mcp = Mcp::snakeCase(
                QStringLiteral("vc3d_neighbor_copy_segment"),
                {QStringLiteral("wait")}),
        },
        [this](const QJsonValue& p) { return handleSegmentNeighborCopy(p); });

    {
        std::vector<AgentBridgeParam> params{
            Params::optionalArray(
                QStringLiteral("segmentIds"),
                Params::requiredString(QString())),
            Params::optionalArray(
                QStringLiteral("grid"),
                Params::optionalArray(
                    QString(),
                    Params::nullable(Params::optionalString(QString())))),
            Params::optionalString(QStringLiteral("refSegmentId")),
            boundedInteger(QStringLiteral("stripCols"), 0, 1024),
        };
        appendRansacParams(params);
        registerMethod(
            {
                .name = QStringLiteral("segments.merge"),
                .params = std::move(params),
                .errors = {
                    -32602, -32000, -32004, -32005, -32006, -32007, -32009,
                },
                .mcp = Mcp::snakeCase(
                    QStringLiteral("vc3d_merge_segments"),
                    {QStringLiteral("wait")}),
            },
            [this](const QJsonValue& p) { return handleSegmentsMerge(p); });
    }

    {
        std::vector<AgentBridgeParam> params{
            Params::requiredArray(
                QStringLiteral("segmentIds"),
                Params::requiredString(QString())),
            Params::optionalString(QStringLiteral("parentId")),
            boundedInteger(QStringLiteral("borderCells"), 1, 256),
            boundedInteger(QStringLiteral("blendCells"), 1, 256),
            boundedInteger(QStringLiteral("idwK"), 1, 16),
            Params::optionalBoolean(QStringLiteral("confirm"), false),
        };
        appendRansacParams(params);
        registerMethod(
            {
                .name = QStringLiteral("segments.patch"),
                .params = std::move(params),
                .errors = {
                    -32602, -32000, -32004, -32005, -32006, -32007, -32009,
                },
                .mcp = Mcp::snakeCase(
                    QStringLiteral("vc3d_patch_segment"),
                    {QStringLiteral("wait")}),
            },
            [this](const QJsonValue& p) { return handleSegmentsPatch(p); });
    }

    registerMethod(
        {
            .name = QStringLiteral("segment.restore_backup"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                [] {
                    AgentBridgeParam index =
                        Params::requiredInteger(QStringLiteral("index"));
                    index.minimum = 0;
                    index.maximum = 9;
                    return index;
                }(),
                Params::optionalBoolean(QStringLiteral("confirm"), false),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_restore_segment_backup")),
        },
        [this](const QJsonValue& p) { return handleSegmentRestoreBackup(p); });

    registerMethod(
        {
            .name = QStringLiteral("segments.move_to_paths"),
            .params = {
                Params::requiredString(QStringLiteral("segmentId")),
                Params::optionalBoolean(QStringLiteral("replaceExisting"), false),
                Params::optionalBoolean(QStringLiteral("confirm"), false),
            },
            .errors = {-32602, -32000, -32004, -32005, -32007, -32009},
            .mcp = Mcp::snakeCase(QStringLiteral("vc3d_move_segment_to_paths")),
        },
        [this](const QJsonValue& p) { return handleSegmentsMoveToPaths(p); });
}
