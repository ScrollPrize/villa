#include "CpuDiscovery.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

using vc::mcp::Json;

int main()
{
    const auto root = std::filesystem::absolute(std::filesystem::temp_directory_path() / "vc-mcp-cpu-discovery-test").lexically_normal();
    std::filesystem::remove_all(root);
    const auto stack = root / "stack";
    std::filesystem::create_directories(stack);
    for (int z = 0; z < 7; ++z) {
        cv::Mat image = cv::Mat::zeros(96, 128, CV_8U);
        cv::line(image, {15, 25 + z / 2}, {105, 70 + z / 2}, cv::Scalar(80 + z * 20), 3);
        cv::circle(image, {70, 45}, 9, cv::Scalar(180), -1);
        assert(cv::imwrite((stack / ("slice-" + std::to_string(z) + ".png")).string(), image));
    }

    vc::mcp::CpuDiscovery service({root / "jobs"});
    std::atomic<bool> cancelled{false};
    auto run = [&](const std::string& operation, const std::string& id, Json input) {
        input["client_request_id"] = id;
        return service.run(operation, id, service.validate(operation, input), cancelled, [](std::string) {});
    };

    auto diagnostics = run("vc_render_surface_diagnostics", "diagnostics", {{"surface_volume_path", stack.string()}});
    const auto diagnosticsPath = std::filesystem::path(diagnostics.artifacts.at(0).at("path").get<std::string>());
    assert(std::filesystem::exists(diagnosticsPath / "persistence.png"));

    auto features = run("ink_compute_classical_features", "features", {{"diagnostics_path", diagnosticsPath.string()}});
    const auto featuresPath = std::filesystem::path(features.artifacts.at(0).at("path").get<std::string>());
    assert(std::filesystem::exists(featuresPath / "candidate-score.png"));

    auto candidates =
        run("ink_find_candidate_regions", "candidates", {{"score_path", (featuresPath / "candidate-score.png").string()}, {"max_candidates", 20}});
    const auto candidatesPath = std::filesystem::path(candidates.artifacts.at(0).at("path").get<std::string>());
    Json candidateSet = Json::parse(std::ifstream(candidatesPath / "candidate-set.json"));
    assert(candidateSet.contains("candidates"));

    auto report = run("ink_render_candidate_report", "report", {{"candidate_set_path", candidatesPath.string()}, {"max_candidates", 10}});
    assert(std::filesystem::exists(std::filesystem::path(report.artifacts.at(0).at("path").get<std::string>()) / "report.json"));

    auto layout = run("text_analyze_layout", "layout", {{"mask_path", (candidatesPath / "candidate-mask.png").string()}});
    const auto layoutPath = std::filesystem::path(layout.artifacts.at(0).at("path").get<std::string>());
    Json analysis = Json::parse(std::ifstream(layoutPath / "layout-analysis.json"));
    assert(analysis.contains("text_like_score"));
    assert(analysis.at("score_semantics") == "heuristic_not_transcription");

    std::filesystem::remove_all(root);
    std::cout << "CpuDiscoveryTest passed\n";
    return 0;
}
