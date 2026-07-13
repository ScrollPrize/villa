#include "McpApplication.hpp"
#include "DiscoveryTools.hpp"
#include "InspectionTools.hpp"

#include <fastmcpp/mcp/handler.hpp>
#include <fastmcpp/resources/resource.hpp>
#include <fastmcpp/resources/template.hpp>
#include <fastmcpp/tools/tool.hpp>

#include <unordered_map>

namespace vc::mcp
{
namespace
{

Json callResult(const Json& structured, std::string text)
{
    return {{"content", Json::array({Json{{"type", "text"}, {"text", std::move(text)}}})}, {"structuredContent", structured}};
}

Json objectSchema(Json properties, Json required = Json::array())
{
    return {{"type", "object"}, {"properties", std::move(properties)}, {"required", std::move(required)}, {"additionalProperties", false}};
}

Json capabilities(const ServerConfig& config)
{
    Json operations = Json::array(
        {"grow_surface", "render_surface_diagnostics", "classical_features", "candidate_regions", "candidate_report", "text_layout"});
    if (config.discovery && config.discovery->nnunetAvailable())
        operations.push_back("volume_run_segmentation");
    if (config.discovery && config.discovery->surfaceBundleAvailable()) {
        operations.push_back("surface_render_registered_roi");
        operations.push_back("surface_validate_geometry");
        operations.push_back("surface_measure_volume_alignment");
    }
    if (config.discovery && config.discovery->structuralEvidenceAvailable()) {
        operations.push_back("text_measure_grid_coherence");
        operations.push_back("ink_compare_registered_predictions");
        operations.push_back("text_epoch_fold_structure");
    }
    if (config.discovery && config.discovery->evidenceFusionAvailable()) {
        operations.push_back("surface_test_stability");
        operations.push_back("ink_fuse_registered_scores");
        operations.push_back("surface_rank_evidence");
    }
    if (config.discovery && config.discovery->reviewAvailable()) {
        operations.push_back("review_create_queue");
        operations.push_back("review_record_assessment");
        operations.push_back("metric_evaluate_labels");
    }
    return {
        {"vc_version", config.vcVersion},
        {"vc_commit", config.vcCommit},
        {"container_digest", config.containerDigest},
        {"operations", std::move(operations)},
        {"volume_segmentation", config.discovery && config.discovery->nnunetAvailable()},
        {"registered_surface_rendering", config.discovery && config.discovery->surfaceBundleAvailable()},
        {"structural_evidence", config.discovery && config.discovery->structuralEvidenceAvailable()},
        {"evidence_fusion", config.discovery && config.discovery->evidenceFusionAvailable()},
        {"review_artifacts", config.discovery && config.discovery->reviewAvailable()},
        {"dinov3_exemplar_search", config.discovery && config.discovery->dinov3Available()},
        {"dinovol_exemplar_search", config.discovery && config.discovery->dinovolAvailable()},
        {"villa_ink_inference", config.discovery && config.discovery->villaAvailable()},
        {"coordinate_spaces", Json::array({"ct_l0_xyz", "ct_l2_xyz"})},
        {"profiles", Json::array({"scroll3-conservative-v1"})},
        {"execution_mode", config.worker ? config.worker->executionMode() : "fake"}};
}

fastmcpp::tools::Tool capabilitiesTool(const ServerConfig& config)
{
    fastmcpp::tools::Tool tool{
        "vc_capabilities",
        objectSchema(Json::object()),
        objectSchema(
            {{"vc_version", {{"type", "string"}}},
             {"vc_commit", {{"type", "string"}}},
             {"container_digest", {{"type", "string"}}},
             {"operations", {{"type", "array"}, {"items", {{"type", "string"}}}}},
             {"coordinate_spaces", {{"type", "array"}, {"items", {{"type", "string"}}}}},
             {"profiles", {{"type", "array"}, {"items", {{"type", "string"}}}}},
             {"execution_mode", {{"type", "string"}}},
             {"volume_segmentation", {{"type", "boolean"}}},
             {"registered_surface_rendering", {{"type", "boolean"}}},
             {"structural_evidence", {{"type", "boolean"}}},
             {"evidence_fusion", {{"type", "boolean"}}},
             {"review_artifacts", {{"type", "boolean"}}},
             {"dinov3_exemplar_search", {{"type", "boolean"}}},
             {"dinovol_exemplar_search", {{"type", "boolean"}}}},
            Json::array(
                {"vc_version",
                 "vc_commit",
                 "container_digest",
                 "operations",
                 "coordinate_spaces",
                 "profiles",
                 "execution_mode",
                 "volume_segmentation",
                 "registered_surface_rendering",
                 "structural_evidence",
                 "evidence_fusion",
                 "review_artifacts",
                 "dinov3_exemplar_search",
                 "dinovol_exemplar_search"})),
        [config](const Json&) { return callResult(capabilities(config), "Volume Cartographer adapter is available"); }};
    tool.set_description("Return this server's VC build and enabled operations")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_validate_args(true);
    return tool;
}

Json growInputSchema()
{
    const Json path = {{"type", "string"}, {"minLength", 1}};
    const Json seed = objectSchema(
        {{"x", {{"type", "number"}}},
         {"y", {{"type", "number"}}},
         {"z", {{"type", "number"}}},
         {"space", {{"type", "string"}, {"enum", Json::array({"ct_l0_xyz", "ct_l2_xyz"})}}}},
        Json::array({"x", "y", "z", "space"}));
    const Json limits = objectSchema(
        {{"max_generations", {{"type", "integer"}, {"minimum", 1}, {"maximum", 10000}}},
         {"min_area_cm2", {{"type", "number"}, {"minimum", 0}, {"maximum", 100}}}});
    Json schema = objectSchema(
        {{"prediction_path", path},
         {"prediction_uri", {{"type", "string"}, {"minLength", 1}}},
         {"prediction_space", {{"type", "string"}, {"enum", Json::array({"ct_l0_xyz", "ct_l2_xyz"})}}},
         {"seed", seed},
         {"profile", {{"type", "string"}, {"const", "scroll3-conservative-v1"}}},
         {"limits", limits},
         {"voxel_size_um", {{"type", "number"}, {"exclusiveMinimum", 0}, {"maximum", 10000}}},
         {"client_request_id", {{"type", "string"}, {"minLength", 1}, {"maxLength", 128}}}},
        Json::array({"seed", "prediction_space", "profile", "client_request_id"}));
    schema["oneOf"] = Json::array({Json{{"required", Json::array({"prediction_path"})}}, Json{{"required", Json::array({"prediction_uri"})}}});
    return schema;
}

Json jobSubmissionSchema()
{
    return objectSchema(
        {{"job_id", {{"type", "string"}}},
         {"state", {{"type", "string"}}},
         {"operation", {{"type", "string"}}},
         {"job_resource", {{"type", "string"}}},
         {"log_resource", {{"type", "string"}}},
         {"submitted_at", {{"type", "string"}}}},
        Json::array({"job_id", "state", "operation", "job_resource", "log_resource", "submitted_at"}));
}

fastmcpp::tools::Tool growTool(const std::shared_ptr<JobStore>& store)
{
    fastmcpp::tools::Tool tool{"vc_grow_surface", growInputSchema(), jobSubmissionSchema(), [store](const Json& input) {
                                   return store->submitGrow(input);
                               }};
    tool.set_description("Validate and enqueue local surface growth")
        .set_annotations({{"readOnlyHint", false}, {"destructiveHint", false}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_validate_args(true);
    return tool;
}

fastmcpp::tools::Tool generateSurfaceTool(const std::shared_ptr<JobStore>& store)
{
    fastmcpp::tools::Tool tool{"vc_generate_surface", growInputSchema(), jobSubmissionSchema(), [store](const Json& input) {
                                   return store->submitGenerateSurface(input);
                               }};
    fastmcpp::AppConfig app;
    app.resource_uri = "ui://vc/inspector.html";
    app.visibility = std::vector<std::string>{"tool_result"};
    tool.set_description("Generate, validate, preview, and hash a local TIFXYZ surface")
        .set_annotations({{"readOnlyHint", false}, {"destructiveHint", false}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_app(std::move(app))
        .set_validate_args(true);
    return tool;
}

fastmcpp::tools::Tool cancelJobTool(const std::shared_ptr<JobStore>& store)
{
    fastmcpp::tools::Tool
        tool{"vc_cancel_job", objectSchema({{"job_id", {{"type", "string"}, {"minLength", 1}}}}, Json::array({"job_id"})), Json::object(), [store](const Json& input) {
                 const auto job = store->cancel(input.at("job_id").get<std::string>());
                 return callResult(job, "Cancellation requested for " + job.at("job_id").get<std::string>());
             }};
    tool.set_description("Cancel a local VC job")
        .set_annotations({{"readOnlyHint", false}, {"destructiveHint", false}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_validate_args(true);
    return tool;
}

fastmcpp::tools::Tool getJobTool(const std::shared_ptr<JobStore>& store)
{
    fastmcpp::tools::Tool tool{
        "vc_get_job",
        objectSchema({{"job_id", {{"type", "string"}, {"minLength", 1}}}}, Json::array({"job_id"})),
        objectSchema(
            {{"job_id", {{"type", "string"}}},
             {"operation", {{"type", "string"}}},
             {"state", {{"type", "string"}}},
             {"progress", {{"type", "object"}}},
             {"created_at", {{"type", "string"}}},
             {"started_at", {{"type", Json::array({"string", "null"})}}},
             {"finished_at", {{"type", Json::array({"string", "null"})}}},
             {"job_revision", {{"type", "integer"}}},
             {"normalized_input", {{"type", "object"}}},
             {"command_manifest", {{"type", "object"}}},
             {"artifacts", {{"type", "array"}}},
             {"error", {{"type", "object"}}}},
            Json::array({"job_id", "operation", "state", "progress", "created_at", "job_revision"})),
        [store](const Json& input) {
            const auto job = store->get(input.at("job_id").get<std::string>());
            return callResult(job, "VC job " + job.at("job_id").get<std::string>() + " is " + job.at("state").get<std::string>());
        }};
    tool.set_description("Read an in-memory VC job")
        .set_annotations({{"readOnlyHint", true}, {"idempotentHint", true}, {"openWorldHint", false}})
        .set_validate_args(true);
    return tool;
}

void registerResources(fastmcpp::resources::ResourceManager& resources, const std::shared_ptr<JobStore>& store, const ServerConfig& config)
{
    fastmcpp::resources::Resource serverCapabilities;
    serverCapabilities.uri = "vc://server/capabilities";
    serverCapabilities.name = "Volume Cartographer capabilities";
    serverCapabilities.description = "Build metadata and Phase 1 operation availability";
    serverCapabilities.mime_type = "application/json";
    serverCapabilities.provider = [config](const Json&) {
        return fastmcpp::resources::ResourceContent{"vc://server/capabilities", "application/json", capabilities(config).dump(2)};
    };
    resources.register_resource(serverCapabilities);

    fastmcpp::resources::ResourceTemplate job;
    job.uri_template = "vc://jobs/{job_id}";
    job.name = "VC job";
    job.description = "Current state of a VC job";
    job.mime_type = "application/json";
    job.provider = [store](const Json& params) {
        const std::string id = params.at("job_id").get<std::string>();
        return fastmcpp::resources::ResourceContent{"vc://jobs/" + id, "application/json", store->get(id).dump(2)};
    };
    resources.register_template(std::move(job));

    fastmcpp::resources::ResourceTemplate logs;
    logs.uri_template = "vc://jobs/{job_id}/logs";
    logs.name = "VC job logs";
    logs.description = "At most the last 500 redacted lines for a VC job";
    logs.mime_type = "application/json";
    logs.provider = [store](const Json& params) {
        const std::string id = params.at("job_id").get<std::string>();
        return fastmcpp::resources::ResourceContent{"vc://jobs/" + id + "/logs", "application/json", store->logs(id).dump(2)};
    };
    resources.register_template(std::move(logs));
}

}  // namespace

McpApplication::McpApplication(ServerConfig config)
    : config_(std::move(config))
    , store_(std::make_shared<JobStore>(config_.worker, config_.discovery))
    , routing_("volume-cartographer", "0.1.0")
{
    tools_.register_tool(capabilitiesTool(config_));
    tools_.register_tool(growTool(store_));
    tools_.register_tool(generateSurfaceTool(store_));
    tools_.register_tool(getJobTool(store_));
    tools_.register_tool(cancelJobTool(store_));
    registerResources(resources_, store_, config_);
    registerInspectionTools(tools_, resources_, store_);
    if (config_.discovery)
        registerDiscoveryTools(tools_, store_, config_.discovery);

    const std::unordered_map<std::string, std::string> descriptions =
        {{"vc_capabilities", "Return this server's VC capabilities"},
         {"vc_grow_surface", "Enqueue local surface growth"},
         {"vc_generate_surface", "Generate and validate a TIFXYZ surface"},
         {"vc_get_job", "Read current job state"},
         {"vc_cancel_job", "Cancel a local VC job"}};
    auto core = fastmcpp::mcp::make_mcp_handler("volume-cartographer", "0.1.0", routing_, tools_, resources_, prompts_, descriptions);
    handler_ = [core = std::move(core)](const Json& request) mutable {
        Json response = core(request);
        const std::string method = request.value("method", "");
        if (method == "initialize" && response.contains("result"))
            response["result"]["capabilities"]["extensions"]["io.modelcontextprotocol/ui"] = Json::object();
        if (method == "resources/read" && request.value("params", Json::object()).value("uri", "") == "ui://vc/inspector.html" &&
            response.contains("result")) {
            for (auto& content : response["result"]["contents"])
                content["_meta"]["ui"] = {{"prefersBorder", false}};
        }
        return response;
    };
}

Json McpApplication::handle(const Json& request) const
{
    return handler_(request);
}

McpApplication::Handler McpApplication::handler() const
{
    return handler_;
}

}  // namespace vc::mcp
