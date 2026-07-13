#include "InspectionTools.hpp"
#include "PredictionService.hpp"

#include <fastmcpp/resources/resource.hpp>
#include <fastmcpp/tools/tool.hpp>

#include <QBuffer>
#include <QCryptographicHash>
#include <QFile>
#include <QImage>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>

namespace vc::mcp
{
namespace
{

constexpr const char* kInspectorUri = "ui://vc/inspector.html";

Json callResult(const Json& structured, const std::string& text)
{
    return {{"content", Json::array({Json{{"type", "text"}, {"text", text}}})}, {"structuredContent", structured}};
}

Json objectSchema(Json properties, Json required = Json::array())
{
    return {{"type", "object"}, {"properties", std::move(properties)}, {"required", std::move(required)}, {"additionalProperties", false}};
}

fastmcpp::AppConfig inspectorApp(std::vector<std::string> visibility = {"tool_result"})
{
    fastmcpp::AppConfig app;
    app.resource_uri = kInspectorUri;
    app.visibility = std::move(visibility);
    return app;
}

fastmcpp::tools::Tool inspectPredictionTool()
{
    auto schema = objectSchema(
        {{"prediction_uri", {{"type", "string"}, {"minLength", 1}}},
         {"prediction_space", {{"type", "string"}, {"enum", Json::array({"ct_l0_xyz", "ct_l2_xyz"})}}}},
        Json::array({"prediction_uri", "prediction_space"}));
    fastmcpp::tools::Tool tool{"vc_inspect_prediction", schema, Json::object(), [](const Json& input) {
                                   const auto output = inspectPrediction(input);
                                   return callResult(output, "Inspected prediction " + output.at("uri").get<std::string>());
                               }};
    tool.set_description("Inspect a surface or ink prediction Zarr and open the interactive seed picker")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(inspectorApp())
        .set_validate_args(true);
    return tool;
}

fastmcpp::tools::Tool findCandidatesTool()
{
    const Json point = objectSchema(
        {{"x", {{"type", "integer"}}}, {"y", {{"type", "integer"}}}, {"z", {{"type", "integer"}}}}, Json::array({"x", "y", "z"}));
    const Json region = objectSchema({{"center", point}, {"radius", point}}, Json::array({"center", "radius"}));
    auto schema = objectSchema(
        {{"prediction_uri", {{"type", "string"}, {"minLength", 1}}},
         {"prediction_space", {{"type", "string"}, {"enum", Json::array({"ct_l0_xyz", "ct_l2_xyz"})}}},
         {"ink_prediction_uri", {{"type", "string"}, {"minLength", 1}}},
         {"region", region},
         {"surface_threshold", {{"type", "integer"}, {"minimum", 0}, {"maximum", 255}}},
         {"ink_threshold", {{"type", "integer"}, {"minimum", 0}, {"maximum", 255}}},
         {"ink_weight", {{"type", "number"}, {"minimum", 0}, {"maximum", 1}}},
         {"max_candidates", {{"type", "integer"}, {"minimum", 1}, {"maximum", 100}}},
         {"minimum_separation_voxels", {{"type", "integer"}, {"minimum", 8}, {"maximum", 256}}}},
        Json::array({"prediction_uri", "prediction_space", "region"}));
    fastmcpp::tools::Tool
        tool{"vc_find_seed_candidates", schema, Json::object(), [](const Json& input) {
                 const auto output = findSeedCandidates(input);
                 return callResult(output, "Found " + std::to_string(output.at("candidates").size()) + " bounded seed candidates");
             }};
    tool.set_description("Find deterministic surface seed candidates with optional ink-prediction scoring")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(inspectorApp())
        .set_validate_args(true);
    return tool;
}

std::string pngDataUri(const std::filesystem::path& path)
{
    QImage source(QString::fromStdString(path.string()));
    if (source.isNull())
        return {};
    const int maximum = 720;
    if (source.width() > maximum || source.height() > maximum)
        source = source.scaled(maximum, maximum, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QImage preview(source.size(), QImage::Format_RGBA8888);
    for (int y = 0; y < source.height(); ++y)
        for (int x = 0; x < source.width(); ++x) {
            const int value = qGray(source.pixel(x, y));
            const double t = value / 255.0;
            preview.setPixelColor(x, y, QColor::fromRgbF(std::min(1.0, 0.12 + t * 1.2), std::min(1.0, 0.18 + t * 0.72), std::max(0.0, 0.32 - t * 0.24), 1.0));
        }
    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    preview.save(&buffer, "PNG");
    return "data:image/png;base64," + bytes.toBase64().toStdString();
}

fastmcpp::tools::Tool surfacePreviewTool(const std::shared_ptr<JobStore>& store)
{
    auto schema = objectSchema({{"job_id", {{"type", "string"}, {"minLength", 1}}}}, Json::array({"job_id"}));
    fastmcpp::tools::Tool tool{"vc_render_surface_preview", schema, Json::object(), [store](const Json& input) {
                                   const auto job = store->get(input.at("job_id").get<std::string>());
                                   if (!job.contains("artifacts") || job.at("artifacts").empty())
                                       throw std::runtime_error("job has no completed surface artifact");
                                   const std::filesystem::path surface = job.at("artifacts").at(0).at("path").get<std::string>();
                                   Json meta = Json::parse(std::ifstream(surface / "meta.json"));
                                   Json output =
                                       {{"job_id", job.at("job_id")},
                                        {"surface_path", surface.string()},
                                        {"meta", std::move(meta)},
                                        {"generation_preview", pngDataUri(surface / "generations.tif")}};
                                   return callResult(output, "Rendered surface preview for " + job.at("job_id").get<std::string>());
                               }};
    tool.set_description("Render a bounded generation preview and geometry summary for a completed surface")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(inspectorApp())
        .set_validate_args(true);
    return tool;
}

std::string sha256(const std::filesystem::path& path)
{
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::ReadOnly))
        return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(&file);
    return hash.result().toHex().toStdString();
}

fastmcpp::tools::Tool segmentationInspectorTool(const std::shared_ptr<JobStore>& store)
{
    auto schema = objectSchema({{"job_id", {{"type", "string"}, {"minLength", 1}}}}, Json::array({"job_id"}));
    fastmcpp::tools::Tool tool{"volume_inspect_segmentation", schema, Json::object(), [store](const Json& input) {
                                   const auto job = store->get(input.at("job_id").get<std::string>());
                                   if (job.at("state") != "succeeded" || !job.contains("artifacts") || job.at("artifacts").empty())
                                       throw std::runtime_error("segmentation job has not succeeded");
                                   const std::filesystem::path root = job.at("artifacts").at(0).at("path").get<std::string>();
                                   Json manifest = Json::parse(std::ifstream(root / "manifest.json"));
                                   Json output =
                                       {{"job_id", job.at("job_id")},
                                        {"state", job.at("state")},
                                        {"segmentation", true},
                                        {"manifest", std::move(manifest)},
                                        {"probability_preview", pngDataUri(root / "probability-preview.png")},
                                        {"mask_preview", pngDataUri(root / "mask-preview.png")}};
                                   return callResult(output, "Rendered segmentation result for " + job.at("job_id").get<std::string>());
                               }};
    tool.set_description("Show probability and mask previews for a completed volumetric segmentation job")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(inspectorApp())
        .set_validate_args(true);
    return tool;
}

fastmcpp::tools::Tool registeredSurfaceInspectorTool(const std::shared_ptr<JobStore>& store)
{
    auto schema = objectSchema({{"job_id", {{"type", "string"}, {"minLength", 1}}}}, Json::array({"job_id"}));
    fastmcpp::tools::Tool tool{"surface_inspect_registered_render", schema, Json::object(), [store](const Json& input) {
                                   const auto job = store->get(input.at("job_id").get<std::string>());
                                   if (job.at("state") != "succeeded" || !job.contains("artifacts") || job.at("artifacts").empty())
                                       throw std::runtime_error("registered surface job has not succeeded");
                                   const std::filesystem::path root = job.at("artifacts").at(0).at("path").get<std::string>();
                                   Json manifest = Json::parse(std::ifstream(root / "manifest.json"));
                                   Json output =
                                       {{"job_id", job.at("job_id")},
                                        {"state", job.at("state")},
                                        {"registered_surface", true},
                                        {"manifest", std::move(manifest)},
                                        {"intensity_preview", pngDataUri(root / "registered-intensity.png")},
                                        {"coverage_preview", pngDataUri(root / "registered-coverage.png")},
                                        {"depth_preview", pngDataUri(root / "surface-depth.png")}};
                                   return callResult(output, "Rendered registered surface evidence for " + job.at("job_id").get<std::string>());
                               }};
    tool.set_description("Show registered CT intensity, coverage, and surface-coordinate evidence")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(inspectorApp())
        .set_validate_args(true);
    return tool;
}

fastmcpp::tools::Tool surfaceEvidenceInspectorTool(
    const std::shared_ptr<JobStore>& store,
    const std::string& name,
    const std::string& expectedKind,
    const std::string& flag,
    std::array<std::pair<std::string, std::string>, 3> previews)
{
    auto schema = objectSchema({{"job_id", {{"type", "string"}, {"minLength", 1}}}}, Json::array({"job_id"}));
    fastmcpp::tools::Tool tool{name, schema, Json::object(), [store, expectedKind, flag, previews](const Json& input) {
                                   const auto job = store->get(input.at("job_id").get<std::string>());
                                   if (job.at("state") != "succeeded" || !job.contains("artifacts") || job.at("artifacts").empty())
                                       throw std::runtime_error("surface evidence job has not succeeded");
                                   const std::filesystem::path root = job.at("artifacts").at(0).at("path").get<std::string>();
                                   Json manifest = Json::parse(std::ifstream(root / "manifest.json"));
                                   if (manifest.value("kind", "") != expectedKind)
                                       throw std::runtime_error("surface evidence artifact kind does not match inspector");
                                   Json output = {{"job_id", job.at("job_id")}, {"state", job.at("state")}, {flag, true}, {"manifest", std::move(manifest)}};
                                   for (const auto& [key, file] : previews)
                                       output[key] = pngDataUri(root / file);
                                   return callResult(output, "Rendered surface evidence for " + job.at("job_id").get<std::string>());
                               }};
    tool.set_description("Show bounded surface evidence maps and their non-truth-claiming manifest")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(inspectorApp())
        .set_validate_args(true);
    return tool;
}

fastmcpp::tools::Tool artifactInspectorTool(const std::shared_ptr<JobStore>& store)
{
    auto schema = objectSchema({{"job_id", {{"type", "string"}, {"minLength", 1}}}}, Json::array({"job_id"}));
    fastmcpp::tools::Tool tool{"vc_inspect_artifacts", schema, Json::object(), [store](const Json& input) {
                                   const auto job = store->get(input.at("job_id").get<std::string>());
                                   Json files = Json::array();
                                   if (job.contains("artifacts"))
                                       for (const auto& artifact : job.at("artifacts")) {
                                           const std::filesystem::path root = artifact.at("path").get<std::string>();
                                           if (!std::filesystem::exists(root))
                                               continue;
                                           if (std::filesystem::is_regular_file(root)) {
                                               files.push_back(
                                                   {{"name", root.filename().string()},
                                                    {"path", root.string()},
                                                    {"size_bytes", std::filesystem::file_size(root)},
                                                    {"sha256", sha256(root)}});
                                               continue;
                                           }
                                           for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
                                               if (entry.is_regular_file())
                                                   files.push_back(
                                                       {{"name", std::filesystem::relative(entry.path(), root).string()},
                                                        {"path", entry.path().string()},
                                                        {"size_bytes", entry.file_size()},
                                                        {"sha256", sha256(entry.path())}});
                                       }
                                   Json output =
                                       {{"job_id", job.at("job_id")},
                                        {"state", job.at("state")},
                                        {"command_manifest", job.value("command_manifest", Json::object())},
                                        {"files", std::move(files)}};
                                   return callResult(output, "Inspected artifacts for " + job.at("job_id").get<std::string>());
                               }};
    tool.set_description("Inspect a job manifest and verify local artifact checksums")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(inspectorApp())
        .set_validate_args(true);
    return tool;
}

}  // namespace

void registerInspectionTools(fastmcpp::tools::ToolManager& tools, fastmcpp::resources::ResourceManager& resources, const std::shared_ptr<JobStore>& store)
{
    tools.register_tool(inspectPredictionTool());
    tools.register_tool(findCandidatesTool());
    tools.register_tool(surfacePreviewTool(store));
    tools.register_tool(segmentationInspectorTool(store));
    tools.register_tool(registeredSurfaceInspectorTool(store));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "surface_inspect_geometry",
        "vc_surface_geometry_diagnostics_v1",
        "surface_geometry",
        {{{"stretch_preview", "stretch-log-ratio.png"},
          {"normal_preview", "normal-change-degrees.png"},
          {"fold_preview", "fold-or-degenerate.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "surface_inspect_volume_alignment",
        "vc_surface_ct_alignment_v1",
        "surface_alignment",
        {{{"gradient_preview", "ct-peak-gradient.png"},
          {"offset_preview", "ct-peak-offset.png"},
          {"confidence_preview", "ct-alignment-confidence.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "text_inspect_grid_coherence",
        "vc_structural_grid_coherence_v1",
        "structural_grid",
        {{{"coherence_preview", "grid-coherence.png"},
          {"secondary_preview", "grid-coherence.png"},
          {"tertiary_preview", "grid-coherence.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "ink_inspect_registered_comparison",
        "vc_registered_structural_comparison_v1",
        "structural_comparison",
        {{{"agreement_preview", "structural-agreement.png"},
          {"divergence_preview", "structural-divergence.png"},
          {"priority_preview", "structural-review-priority.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "text_inspect_epoch_fold",
        "vc_epoch_fold_structure_v1",
        "epoch_fold",
        {{{"folded_preview", "folded-phase-profile.png"},
          {"search_preview", "period-search.png"},
          {"secondary_preview", "folded-phase-profile.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "surface_inspect_stability",
        "vc_surface_perturbation_stability_v1",
        "surface_stability",
        {{{"displacement_preview", "stability-displacement-mm.png"},
          {"angle_preview", "stability-normal-angle.png"},
          {"stability_preview", "stability-local-score.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "ink_inspect_registered_fusion",
        "vc_registered_ink_fusion_v1",
        "ink_fusion",
        {{{"combined_preview", "combined-score.png"},
          {"ink_model_preview", "ink-model-score.png"},
          {"dinovol_preview", "dinovol-similarity-normalized.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "surface_inspect_evidence_ranking",
        "vc_transparent_evidence_ranking_v1",
        "evidence_ranking",
        {{{"ranking_preview", "evidence-ranking.png"},
          {"secondary_preview", "evidence-ranking.png"},
          {"tertiary_preview", "evidence-ranking.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "review_inspect_queue",
        "vc_review_queue_v1",
        "review_queue",
        {{{"queue_preview", "review-queue.png"}, {"secondary_preview", "review-queue.png"}, {"tertiary_preview", "review-queue.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "review_inspect_assessment",
        "vc_review_assessment_v1",
        "review_assessment",
        {{{"assessment_preview", "assessment-summary.png"},
          {"secondary_preview", "assessment-summary.png"},
          {"tertiary_preview", "assessment-summary.png"}}}));
    tools.register_tool(surfaceEvidenceInspectorTool(
        store,
        "metric_inspect_label_evaluation",
        "vc_review_label_evaluation_v1",
        "label_evaluation",
        {{{"evaluation_preview", "evaluation-summary.png"},
          {"secondary_preview", "evaluation-summary.png"},
          {"tertiary_preview", "evaluation-summary.png"}}}));
    tools.register_tool(artifactInspectorTool(store));

    fastmcpp::resources::Resource ui;
    ui.uri = kInspectorUri;
    ui.name = "Volume Cartographer Inspector";
    ui.description = "Interactive prediction, seed, surface, and artifact inspector";
    ui.mime_type = "text/html;profile=mcp-app";
    fastmcpp::AppConfig app;
    app.prefers_border = false;
    app.csp = Json{{"connectDomains", Json::array()}, {"resourceDomains", Json::array()}};
    ui.app = app;
    ui.provider = [](const Json&) {
        QFile file(":/vc_mcp_ui/inspector.html");
        if (!file.open(QIODevice::ReadOnly))
            throw std::runtime_error("embedded MCP App resource is unavailable");
        return fastmcpp::resources::ResourceContent{kInspectorUri, "text/html;profile=mcp-app", file.readAll().toStdString()};
    };
    resources.register_resource(ui);
}

}  // namespace vc::mcp
