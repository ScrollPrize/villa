#include "CpuDiscovery.hpp"
#include "McpApplication.hpp"
#include "VolumeCartographer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>

#ifndef VC_MCP_FAKE_GROW_PATH
#error VC_MCP_FAKE_GROW_PATH required
#endif
#ifndef VC_MCP_ANALYSIS_PYTHON
#error VC_MCP_ANALYSIS_PYTHON required
#endif
#ifndef VC_MCP_SURFACE_BUNDLE_ADAPTER
#error VC_MCP_SURFACE_BUNDLE_ADAPTER required
#endif
#ifndef VC_MCP_VOLUME_STAGER
#error VC_MCP_VOLUME_STAGER required
#endif
#ifndef VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER
#error VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER required
#endif
#ifndef VC_MCP_EVIDENCE_FUSION_ADAPTER
#error VC_MCP_EVIDENCE_FUSION_ADAPTER required
#endif
#ifndef VC_MCP_REVIEW_ADAPTER
#error VC_MCP_REVIEW_ADAPTER required
#endif

using vc::mcp::Json;
namespace
{
Json rpc(int id, std::string method, Json params = {})
{
    return {{"jsonrpc", "2.0"}, {"id", id}, {"method", std::move(method)}, {"params", std::move(params)}};
}
const Json& result(const Json& response)
{
    assert(response.contains("result"));
    return response.at("result");
}
Json waitForJob(vc::mcp::McpApplication& app, int& rpcId, const std::string& jobId)
{
    Json job;
    for (int attempt = 0; attempt < 500; ++attempt) {
        job = result(app.handle(rpc(rpcId++, "tools/call", {{"name", "vc_get_job"}, {"arguments", {{"job_id", jobId}}}})))
                  .at("structuredContent");
        if (job.at("state") == "succeeded" || job.at("state") == "failed")
            return job;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return job;
}
}  // namespace

int main()
{
    const auto root = std::filesystem::absolute(std::filesystem::temp_directory_path() / "vc-mcp-registered-surface-test").lexically_normal();
    std::filesystem::remove_all(root);
    const auto prediction = root / "prediction.zarr";
    const auto volume = root / "volume.zarr";
    std::filesystem::create_directories(prediction);
    std::filesystem::create_directories(volume / "0");
    std::ofstream(volume / ".zgroup") << R"({"zarr_format":2})";
    std::ofstream(volume / "0" / ".zarray") << R"({"zarr_format":2,"shape":[192,192,192],"chunks":[192,192,192],"dtype":"|u1","compressor":null,"fill_value":0,"filters":null,"order":"C","dimension_separator":"."})";
    std::ofstream chunk(volume / "0" / "0.0.0", std::ios::binary);
    for (int z = 0; z < 192; ++z)
        for (int y = 0; y < 192; ++y)
            for (int x = 0; x < 192; ++x)
                chunk.put(char((x * 7 + y * 5 + z * 3) % 256));
    chunk.close();

    auto worker = std::make_shared<vc::mcp::LocalVolumeCartographer>(
        vc::mcp::LocalWorkerConfig{VC_MCP_FAKE_GROW_PATH, root / "jobs", std::chrono::seconds(30)});
    vc::mcp::CpuDiscoveryConfig discoveryConfig;
    discoveryConfig.workRoot = root / "jobs";
    discoveryConfig.analysisPython = VC_MCP_ANALYSIS_PYTHON;
    discoveryConfig.surfaceBundleAdapter = VC_MCP_SURFACE_BUNDLE_ADAPTER;
    discoveryConfig.structuralEvidenceAdapter = VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER;
    discoveryConfig.evidenceFusionAdapter = VC_MCP_EVIDENCE_FUSION_ADAPTER;
    discoveryConfig.reviewAdapter = VC_MCP_REVIEW_ADAPTER;
    discoveryConfig.volumeStager = VC_MCP_VOLUME_STAGER;
    discoveryConfig.timeout = std::chrono::seconds(30);
    auto discovery = std::make_shared<vc::mcp::CpuDiscovery>(std::move(discoveryConfig));
    assert(discovery->surfaceBundleAvailable());
    vc::mcp::McpApplication app({"test", "test", "unverified", worker, discovery});
    int rpcId = 1;

    const Json growArguments =
        {{"prediction_path", prediction.string()},
         {"prediction_space", "ct_l0_xyz"},
         {"seed", {{"x", 12}, {"y", 24}, {"z", 32}, {"space", "ct_l0_xyz"}}},
         {"profile", "scroll3-conservative-v1"},
         {"limits", {{"max_generations", 1}}},
         {"client_request_id", "registered-surface-grow"}};
    const auto growCall = result(app.handle(rpc(rpcId++, "tools/call", {{"name", "vc_grow_surface"}, {"arguments", growArguments}})));
    const auto growJobId = growCall.at("structuredContent").at("job_id").get<std::string>();
    assert(waitForJob(app, rpcId, growJobId).at("state") == "succeeded");

    const auto surface = root / "jobs" / growJobId / "surface";
    cv::Mat x(80, 96, CV_32F), y(80, 96, CV_32F), z(80, 96, CV_32F, cv::Scalar(100));
    for (int v = 0; v < x.rows; ++v)
        for (int u = 0; u < x.cols; ++u) {
            x.at<float>(v, u) = float(u + 40);
            y.at<float>(v, u) = float(v + 50);
        }
    assert(cv::imwrite((surface / "x.tif").string(), x));
    assert(cv::imwrite((surface / "y.tif").string(), y));
    assert(cv::imwrite((surface / "z.tif").string(), z));
    std::ofstream(surface / "meta.json") << R"({"format":"tifxyz","scale":[1,1]})";

    const Json renderArguments =
        {{"surface", {{"job_id", growJobId}, {"artifact_id", "surface"}}},
         {"volume",
          {{"kind", "local_zarr"}, {"path", volume.string()}, {"array_path", "0"}, {"scale", 0}, {"voxel_spacing", {1000.0, 1000.0, 1000.0}}, {"origin_xyz", {0.0, 0.0, 0.0}}}},
         {"coordinate_space", "ct_l0_xyz"},
         {"uv_region", {{"u", 0}, {"v", 0}, {"width", 96}, {"height", 80}}},
         {"normal_padding_voxels", 32},
         {"client_request_id", "registered-surface-render"}};
    const auto renderCall =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "surface_render_registered_roi"}, {"arguments", renderArguments}})));
    const auto renderJobId = renderCall.at("structuredContent").at("job_id").get<std::string>();
    const auto renderJob = waitForJob(app, rpcId, renderJobId);
    if (renderJob.at("state") != "succeeded") {
        std::cerr << renderJob.dump(2) << '\n' << app.store()->logs(renderJobId).dump(2) << '\n';
        return 1;
    }
    const auto inspected =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "surface_inspect_registered_render"}, {"arguments", {{"job_id", renderJobId}}}})))
            .at("structuredContent");
    assert(inspected.at("registered_surface") == true);
    assert(inspected.at("manifest").at("surface_shape_vu") == Json::array({80, 96}));
    assert(inspected.at("manifest").at("coverage_fraction") == 1.0);
    assert(inspected.at("intensity_preview").get<std::string>().starts_with("data:image/png;base64,"));
    const auto artifact = app.store()->resolveArtifactReference({{"job_id", renderJobId}, {"artifact_id", "registered-surface"}});
    assert(artifact.at("media_type") == "application/vnd.vc.registered-surface+zarr");

    const Json registeredReference = {{"job_id", renderJobId}, {"artifact_id", "registered-surface"}};
    const auto validateNormalStack = [&](const std::string& profile, int channels, const std::string& requestId) {
        const auto call = result(app.handle(
            rpc(rpcId++,
                "tools/call",
                {{"name", "surface_render_normal_stack"},
                 {"arguments",
                  {{"surface", registeredReference}, {"model_profile", profile}, {"reverse_layers", false}, {"layer_step_voxels", 1.0}, {"client_request_id", requestId}}}})));
        const auto jobId = call.at("structuredContent").at("job_id").get<std::string>();
        const auto job = waitForJob(app, rpcId, jobId);
        if (job.at("state") != "succeeded") {
            std::cerr << job.dump(2) << '\n' << app.store()->logs(jobId).dump(2) << '\n';
            std::abort();
        }
        const auto stackArtifact = app.store()->resolveArtifactReference({{"job_id", jobId}, {"artifact_id", "surface-volume"}});
        assert(stackArtifact.at("media_type") == "application/vnd.vc.surface-volume+zarr");
        const std::filesystem::path output = stackArtifact.at("path").get<std::string>();
        const auto manifest = Json::parse(std::ifstream(output / "manifest.json"));
        const auto zarray = Json::parse(std::ifstream(output / "surface-volume.zarr" / ".zarray"));
        assert(manifest.at("shape_hwc") == Json::array({80, 96, channels}));
        assert(manifest.at("ink_model_loader_compatible") == true);
        assert(manifest.at("dtype") == "uint8");
        assert(zarray.at("shape") == Json::array({80, 96, channels}));
        assert(zarray.at("chunks").at(2) == 1);
        assert(std::distance(std::filesystem::directory_iterator(output / "layers"), std::filesystem::directory_iterator{}) == channels);
    };
    validateNormalStack("timesformer-26", 26, "registered-surface-timesformer-stack");
    validateNormalStack("resnet152-3d-decoder-62", 62, "registered-surface-resnet-stack");

    const auto geometryCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "surface_validate_geometry"},
             {"arguments", {{"surface", registeredReference}, {"client_request_id", "registered-surface-geometry"}}}})));
    const auto geometryJobId = geometryCall.at("structuredContent").at("job_id").get<std::string>();
    assert(waitForJob(app, rpcId, geometryJobId).at("state") == "succeeded");
    const auto geometryInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "surface_inspect_geometry"}, {"arguments", {{"job_id", geometryJobId}}}})))
            .at("structuredContent");
    assert(geometryInspection.at("surface_geometry") == true);
    assert(geometryInspection.at("manifest").at("fold_or_degenerate_cells") == 0);
    assert(geometryInspection.at("stretch_preview").get<std::string>().starts_with("data:image/png;base64,"));

    const auto alignmentCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "surface_measure_volume_alignment"},
             {"arguments",
              {{"surface", registeredReference}, {"maximum_offset_voxels", 1}, {"client_request_id", "registered-surface-alignment"}}}})));
    const auto alignmentJobId = alignmentCall.at("structuredContent").at("job_id").get<std::string>();
    const auto alignmentJob = waitForJob(app, rpcId, alignmentJobId);
    if (alignmentJob.at("state") != "succeeded") {
        std::cerr << alignmentJob.dump(2) << '\n' << app.store()->logs(alignmentJobId).dump(2) << '\n';
        return 1;
    }
    const auto alignmentInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "surface_inspect_volume_alignment"}, {"arguments", {{"job_id", alignmentJobId}}}})))
            .at("structuredContent");
    assert(alignmentInspection.at("surface_alignment") == true);
    assert(alignmentInspection.at("manifest").at("support_fraction") == 1.0);
    assert(alignmentInspection.at("confidence_preview").get<std::string>().starts_with("data:image/png;base64,"));

    const auto gridCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "text_measure_grid_coherence"},
             {"arguments",
              {{"surface", registeredReference},
               {"polarity", "bright"},
               {"letter_period_mm", 3.0},
               {"line_period_mm", 4.0},
               {"window_width_mm", 12.0},
               {"window_height_mm", 8.0},
               {"step_mm", 3.0},
               {"minimum_cycles", 3.0},
               {"null_trials", 4},
               {"null_seed", 7},
               {"client_request_id", "registered-surface-grid"}}}})));
    const auto gridJobId = gridCall.at("structuredContent").at("job_id").get<std::string>();
    const auto gridJob = waitForJob(app, rpcId, gridJobId);
    if (gridJob.at("state") != "succeeded") {
        std::cerr << gridJob.dump(2) << '\n' << app.store()->logs(gridJobId).dump(2) << '\n';
        return 1;
    }
    const auto gridInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "text_inspect_grid_coherence"}, {"arguments", {{"job_id", gridJobId}}}})))
            .at("structuredContent");
    assert(gridInspection.at("structural_grid") == true);
    assert(gridInspection.at("manifest").at("score_semantics") == "structural_periodicity_not_ink_probability_or_text_truth");

    const auto comparisonCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "ink_compare_registered_predictions"},
             {"arguments",
              {{"surface_a", registeredReference},
               {"surface_b", registeredReference},
               {"polarity", "bright"},
               {"letter_period_mm", 3.0},
               {"line_period_mm", 4.0},
               {"window_width_mm", 12.0},
               {"window_height_mm", 8.0},
               {"step_mm", 3.0},
               {"minimum_cycles", 3.0},
               {"null_trials", 4},
               {"client_request_id", "registered-surface-comparison"}}}})));
    const auto comparisonJobId = comparisonCall.at("structuredContent").at("job_id").get<std::string>();
    assert(waitForJob(app, rpcId, comparisonJobId).at("state") == "succeeded");
    const auto comparisonInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "ink_inspect_registered_comparison"}, {"arguments", {{"job_id", comparisonJobId}}}})))
            .at("structuredContent");
    assert(comparisonInspection.at("structural_comparison") == true);
    assert(comparisonInspection.at("manifest").at("maximum_xyz_difference") == 0.0);

    const Json gridReference = {{"job_id", gridJobId}, {"artifact_id", "grid-coherence"}};
    const auto foldCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "text_epoch_fold_structure"},
             {"arguments",
              {{"surface", registeredReference},
               {"grid", gridReference},
               {"polarity", "bright"},
               {"period_tolerance", 0.1},
               {"period_steps", 9},
               {"phase_bins", 16},
               {"null_trials", 4},
               {"null_seed", 11},
               {"client_request_id", "registered-surface-fold"}}}})));
    const auto foldJobId = foldCall.at("structuredContent").at("job_id").get<std::string>();
    const auto foldJob = waitForJob(app, rpcId, foldJobId);
    if (foldJob.at("state") != "succeeded") {
        std::cerr << foldJob.dump(2) << '\n' << app.store()->logs(foldJobId).dump(2) << '\n';
        return 1;
    }
    const auto foldInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "text_inspect_epoch_fold"}, {"arguments", {{"job_id", foldJobId}}}})))
            .at("structuredContent");
    assert(foldInspection.at("epoch_fold") == true);
    assert(foldInspection.at("manifest").at("score_semantics") == "periodic_line_structure_detection_not_text_or_transcription");

    const auto stabilityCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "surface_test_stability"},
             {"arguments",
              {{"baseline", registeredReference},
               {"variants", Json::array({registeredReference})},
               {"displacement_scale_mm", 0.05},
               {"normal_angle_scale_degrees", 10.0},
               {"signal_scale", 0.1},
               {"client_request_id", "registered-surface-stability"}}}})));
    const auto stabilityJobId = stabilityCall.at("structuredContent").at("job_id").get<std::string>();
    const auto stabilityJob = waitForJob(app, rpcId, stabilityJobId);
    if (stabilityJob.at("state") != "succeeded") {
        std::cerr << stabilityJob.dump(2) << '\n' << app.store()->logs(stabilityJobId).dump(2) << '\n';
        return 1;
    }
    const auto stabilityInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "surface_inspect_stability"}, {"arguments", {{"job_id", stabilityJobId}}}})))
            .at("structuredContent");
    assert(stabilityInspection.at("surface_stability") == true);
    assert(stabilityInspection.at("manifest").at("overall_stability_score") == 1.0);

    const Json rankingCandidate =
        {{"id", "candidate-a"},
         {"geometry", {{"job_id", geometryJobId}, {"artifact_id", "surface-geometry"}}},
         {"alignment", {{"job_id", alignmentJobId}, {"artifact_id", "surface-ct-alignment"}}},
         {"grid", gridReference},
         {"stability", {{"job_id", stabilityJobId}, {"artifact_id", "surface-stability"}}}};
    const auto rankingCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "surface_rank_evidence"},
             {"arguments",
              {{"candidates", Json::array({rankingCandidate})},
               {"weights", {{"geometry", 1.0}, {"alignment", 1.0}, {"grid", 1.0}, {"stability", 1.0}}},
               {"client_request_id", "registered-surface-ranking"}}}})));
    const auto rankingJobId = rankingCall.at("structuredContent").at("job_id").get<std::string>();
    const auto rankingJob = waitForJob(app, rpcId, rankingJobId);
    if (rankingJob.at("state") != "succeeded") {
        std::cerr << rankingJob.dump(2) << '\n' << app.store()->logs(rankingJobId).dump(2) << '\n';
        return 1;
    }
    const auto rankingInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "surface_inspect_evidence_ranking"}, {"arguments", {{"job_id", rankingJobId}}}})))
            .at("structuredContent");
    assert(rankingInspection.at("evidence_ranking") == true);
    assert(rankingInspection.at("manifest").at("ranked_candidates").size() == 1);
    assert(rankingInspection.at("ranking_preview").get<std::string>().starts_with("data:image/png;base64,"));

    const Json rankingReference = {{"job_id", rankingJobId}, {"artifact_id", "evidence-ranking"}};
    const Json comparisonReference = {{"job_id", comparisonJobId}, {"artifact_id", "structural-comparison"}};
    const auto queueCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "review_create_queue"},
             {"arguments",
              {{"ranking", rankingReference},
               {"comparison", comparisonReference},
               {"max_items", 10},
               {"divergence_percentile", 90.0},
               {"client_request_id", "review-queue"}}}})));
    const auto queueJobId = queueCall.at("structuredContent").at("job_id").get<std::string>();
    assert(waitForJob(app, rpcId, queueJobId).at("state") == "succeeded");
    const auto queueInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "review_inspect_queue"}, {"arguments", {{"job_id", queueJobId}}}})))
            .at("structuredContent");
    assert(queueInspection.at("review_queue") == true);
    assert(!queueInspection.at("manifest").at("items").empty());

    const Json queueReference = {{"job_id", queueJobId}, {"artifact_id", "review-queue"}};
    const auto assessmentCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "review_record_assessment"},
             {"arguments",
              {{"queue", queueReference},
               {"reviewer_id", "test-reviewer"},
               {"assessments",
                Json::array({Json{{"item_id", "candidate:candidate-a"}, {"decision", "accept"}, {"confidence", 0.9}, {"reason_codes", Json::array({"synthetic-fixture"})}}})},
               {"client_request_id", "review-assessment"}}}})));
    const auto assessmentJobId = assessmentCall.at("structuredContent").at("job_id").get<std::string>();
    assert(waitForJob(app, rpcId, assessmentJobId).at("state") == "succeeded");
    const auto assessmentInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "review_inspect_assessment"}, {"arguments", {{"job_id", assessmentJobId}}}})))
            .at("structuredContent");
    assert(assessmentInspection.at("review_assessment") == true);

    const Json assessmentReference = {{"job_id", assessmentJobId}, {"artifact_id", "review-assessment"}};
    const auto evaluationCall = result(app.handle(
        rpc(rpcId++,
            "tools/call",
            {{"name", "metric_evaluate_labels"},
             {"arguments", {{"assessments", Json::array({assessmentReference})}, {"client_request_id", "label-evaluation"}}}})));
    const auto evaluationJobId = evaluationCall.at("structuredContent").at("job_id").get<std::string>();
    assert(waitForJob(app, rpcId, evaluationJobId).at("state") == "succeeded");
    const auto evaluationInspection =
        result(app.handle(rpc(rpcId++, "tools/call", {{"name", "metric_inspect_label_evaluation"}, {"arguments", {{"job_id", evaluationJobId}}}})))
            .at("structuredContent");
    assert(evaluationInspection.at("label_evaluation") == true);
    assert(evaluationInspection.at("manifest").at("record_count") == 1);

    std::filesystem::remove_all(root);
    std::cout << "RegisteredSurfaceMcpTest passed\n";
    return 0;
}
