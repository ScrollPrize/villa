#include "CpuDiscovery.hpp"
#include "McpApplication.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifndef VC_MCP_DINOV3_ADAPTER_PATH
#error VC_MCP_DINOV3_ADAPTER_PATH must be defined
#endif

using vc::mcp::Json;

namespace
{

Json request(int id, std::string method, Json params = Json::object())
{
    return {{"jsonrpc", "2.0"}, {"id", id}, {"method", std::move(method)}, {"params", std::move(params)}};
}

const Json& result(const Json& response)
{
    assert(response.at("jsonrpc") == "2.0");
    assert(response.contains("result"));
    return response.at("result");
}

const Json& findByName(const Json& values, const std::string& name)
{
    const auto found = std::find_if(values.begin(), values.end(), [&](const Json& value) { return value.value("name", "") == name; });
    assert(found != values.end());
    return *found;
}

Json growArguments(const std::filesystem::path& prediction)
{
    return {
        {"prediction_path", prediction.string()},
        {"prediction_space", "ct_l0_xyz"},
        {"seed", {{"x", 1}, {"y", 2}, {"z", 3}, {"space", "ct_l2_xyz"}}},
        {"profile", "scroll3-conservative-v1"},
        {"limits", {{"max_generations", 4}}},
        {"client_request_id", "protocol-test-grow-1"}};
}

}  // namespace

int main()
{
    const auto prediction = std::filesystem::temp_directory_path() / "vc-mcp-protocol-prediction";
    std::filesystem::create_directories(prediction);
    vc::mcp::CpuDiscoveryConfig discoveryConfig;
    discoveryConfig.workRoot = std::filesystem::temp_directory_path() / "vc-mcp-protocol-discovery";
    discoveryConfig.dinov3Executable = VC_MCP_DINOV3_ADAPTER_PATH;
    auto discovery = std::make_shared<vc::mcp::CpuDiscovery>(std::move(discoveryConfig));
    vc::mcp::McpApplication app({"test-version", "test-commit", "sha256:test", {}, discovery});

    const auto initialized = result(app.handle(request(
        1,
        "initialize",
        {{"protocolVersion", "2024-11-05"},
         {"capabilities", Json::object()},
         {"clientInfo", {{"name", "vc-mcp-test"}, {"version", "1"}}}})));
    assert(initialized.at("serverInfo").at("name") == "volume-cartographer");
    assert(initialized.at("capabilities").contains("tools"));
    assert(initialized.at("capabilities").contains("resources"));
    assert(!initialized.at("capabilities").contains("prompts"));
    assert(!initialized.at("capabilities").contains("sampling"));
    assert(initialized.at("capabilities").at("extensions").contains("io.modelcontextprotocol/ui"));

    const auto tools = result(app.handle(request(2, "tools/list"))).at("tools");
    assert(tools.size() >= 27);
    const auto& capabilitiesTool = findByName(tools, "vc_capabilities");
    assert(capabilitiesTool.contains("inputSchema"));
    assert(capabilitiesTool.contains("outputSchema"));
    assert(capabilitiesTool.at("annotations").at("readOnlyHint") == true);
    const auto& growTool = findByName(tools, "vc_grow_surface");
    assert(growTool.at("annotations").at("idempotentHint") == true);
    findByName(tools, "vc_get_job");
    findByName(tools, "vc_cancel_job");
    const auto& generateTool = findByName(tools, "vc_generate_surface");
    assert(generateTool.at("_meta").at("ui").at("resourceUri") == "ui://vc/inspector.html");
    const auto& inspectTool = findByName(tools, "vc_inspect_prediction");
    assert(inspectTool.at("_meta").at("ui").at("resourceUri") == "ui://vc/inspector.html");
    findByName(tools, "vc_find_seed_candidates");
    findByName(tools, "vc_render_surface_preview");
    findByName(tools, "volume_inspect_segmentation");
    findByName(tools, "surface_inspect_registered_render");
    findByName(tools, "surface_inspect_geometry");
    findByName(tools, "surface_inspect_volume_alignment");
    findByName(tools, "text_inspect_grid_coherence");
    findByName(tools, "ink_inspect_registered_comparison");
    findByName(tools, "text_inspect_epoch_fold");
    findByName(tools, "surface_inspect_stability");
    findByName(tools, "surface_inspect_evidence_ranking");
    findByName(tools, "review_inspect_queue");
    findByName(tools, "review_inspect_assessment");
    findByName(tools, "metric_inspect_label_evaluation");
    findByName(tools, "vc_inspect_artifacts");
    findByName(tools, "vc_render_surface_diagnostics");
    findByName(tools, "ink_compute_classical_features");
    findByName(tools, "ink_find_candidate_regions");
    findByName(tools, "ink_render_candidate_report");
    findByName(tools, "text_analyze_layout");
    const auto& dinov3Tool = findByName(tools, "dinov3_exemplar_search");
    assert(dinov3Tool.at("annotations").at("idempotentHint") == true);
    assert(std::none_of(tools.begin(), tools.end(), [](const Json& tool) { return tool.value("name", "") == "dinovol_exemplar_search"; }));

    const auto capabilityCall = result(app.handle(request(3, "tools/call", {{"name", "vc_capabilities"}, {"arguments", Json::object()}})));
    assert(capabilityCall.contains("content"));
    assert(capabilityCall.at("structuredContent").at("vc_commit") == "test-commit");

    const auto growCall = result(app.handle(request(4, "tools/call", {{"name", "vc_grow_surface"}, {"arguments", growArguments(prediction)}})));
    const auto& submission = growCall.at("structuredContent");
    assert(submission.at("state") == "queued");
    const std::string jobId = submission.at("job_id").get<std::string>();

    const auto duplicateCall =
        result(app.handle(request(5, "tools/call", {{"name", "vc_grow_surface"}, {"arguments", growArguments(prediction)}})));
    assert(duplicateCall.at("structuredContent").at("job_id") == jobId);

    const auto templates = result(app.handle(request(6, "resources/templates/list"))).at("resourceTemplates");
    assert(templates.size() == 2);
    findByName(templates, "VC job");
    findByName(templates, "VC job logs");

    const auto uiResource = result(app.handle(request(7, "resources/read", {{"uri", "ui://vc/inspector.html"}})));
    assert(uiResource.at("contents").at(0).at("mimeType") == "text/html;profile=mcp-app");
    assert(uiResource.at("contents").at(0).at("text").get<std::string>().find("Vellum Lens") != std::string::npos);
    assert(uiResource.at("contents").at(0).at("_meta").at("ui").at("prefersBorder") == false);

    const auto capabilityResource = result(app.handle(request(8, "resources/read", {{"uri", "vc://server/capabilities"}})));
    const auto capabilityDocument = Json::parse(capabilityResource.at("contents").at(0).at("text").get<std::string>());
    assert(capabilityDocument.at("container_digest") == "sha256:test");

    const auto jobResource = result(app.handle(request(9, "resources/read", {{"uri", "vc://jobs/" + jobId}})));
    const auto jobDocument = Json::parse(jobResource.at("contents").at(0).at("text").get<std::string>());
    assert(jobDocument.at("job_id") == jobId);
    assert(jobDocument.at("normalized_input").at("coordinates").at("vc_input").at("x") == 4.0);

    const auto logsResource = result(app.handle(request(10, "resources/read", {{"uri", "vc://jobs/" + jobId + "/logs"}})));
    const auto logsDocument = Json::parse(logsResource.at("contents").at(0).at("text").get<std::string>());
    assert(logsDocument.at("job_id") == jobId);
    assert(!logsDocument.at("lines").empty());

    std::cout << "ProtocolTest passed\n";
    return 0;
}
