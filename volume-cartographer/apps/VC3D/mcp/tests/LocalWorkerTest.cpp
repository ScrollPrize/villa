#include "JobStore.hpp"
#include "VolumeCartographer.hpp"

#include <QCoreApplication>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>

#ifndef VC_MCP_FAKE_GROW_PATH
#error VC_MCP_FAKE_GROW_PATH must be defined
#endif

using vc::mcp::Json;

namespace
{

Json request(const std::filesystem::path& prediction, std::string requestId = "local-worker-test", int generations = 4)
{
    return {
        {"prediction_path", prediction.string()},
        {"prediction_space", "ct_l0_xyz"},
        {"seed", {{"x", 1}, {"y", 2}, {"z", 3}, {"space", "ct_l2_xyz"}}},
        {"profile", "scroll3-conservative-v1"},
        {"limits", {{"max_generations", generations}}},
        {"client_request_id", std::move(requestId)}};
}

}  // namespace

int main(int argc, char** argv)
{
    QCoreApplication application(argc, argv);
    const auto root = std::filesystem::temp_directory_path() / "vc-mcp-local-worker-test";
    std::filesystem::remove_all(root);
    const auto prediction = root / "prediction;touch injected.zarr";
    std::filesystem::create_directories(prediction);

    auto worker = std::make_shared<vc::mcp::LocalVolumeCartographer>(
        vc::mcp::LocalWorkerConfig{VC_MCP_FAKE_GROW_PATH, root / "jobs", std::chrono::seconds(10)});
    vc::mcp::JobStore store(worker);
    const auto submitted = store.submitGenerateSurface(request(prediction)).at("structuredContent");
    const std::string jobId = submitted.at("job_id").get<std::string>();

    Json job;
    for (int attempt = 0; attempt < 500; ++attempt) {
        job = store.get(jobId);
        if (job.at("state") == "succeeded" || job.at("state") == "failed")
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(job.at("state") == "succeeded");
    assert(job.at("command_manifest").at("executable") == std::filesystem::weakly_canonical(VC_MCP_FAKE_GROW_PATH).string());
    assert(job.at("command_manifest").at("profile").at("generations") == 4);
    assert(job.at("normalized_input").at("coordinates").at("vc_input").at("x") == 4.0);
    assert(std::filesystem::exists(root / "jobs" / jobId / "surface" / "meta.json"));
    assert(!store.logs(jobId).at("lines").empty());
    assert(!std::filesystem::exists(root / "injected.zarr"));
    assert(job.at("operation") == "generate_surface");
    assert(job.at("surface").at("format") == "tifxyz");
    for (const auto& stage : job.at("stages"))
        assert(stage.at("state") == "succeeded" || stage.at("state") == "skipped");
    const auto& files = job.at("artifacts").at(0).at("metadata").at("files");
    assert(!files.empty() && files.at(0).at("sha256").get<std::string>().size() == 64);

    const auto cancelSubmission = store.submitGrow(request(prediction, "local-worker-cancel", 9999)).at("structuredContent");
    const std::string cancelId = cancelSubmission.at("job_id").get<std::string>();
    for (int attempt = 0; attempt < 200; ++attempt) {
        job = store.get(cancelId);
        if (job.at("state") == "running")
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(job.at("state") == "running");
    store.cancel(cancelId);
    for (int attempt = 0; attempt < 500; ++attempt) {
        job = store.get(cancelId);
        if (job.at("state") == "cancelled")
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(job.at("state") == "cancelled");

    auto timeoutWorker = std::make_shared<vc::mcp::LocalVolumeCartographer>(
        vc::mcp::LocalWorkerConfig{VC_MCP_FAKE_GROW_PATH, root / "timeout-jobs", std::chrono::seconds(1)});
    vc::mcp::JobStore timeoutStore(timeoutWorker);
    const auto timeoutSubmission = timeoutStore.submitGrow(request(prediction, "local-worker-timeout", 9999)).at("structuredContent");
    const std::string timeoutId = timeoutSubmission.at("job_id").get<std::string>();
    for (int attempt = 0; attempt < 700; ++attempt) {
        job = timeoutStore.get(timeoutId);
        if (job.at("state") == "failed")
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(job.at("state") == "failed");
    assert(job.at("error").at("message").get<std::string>().find("timeout") != std::string::npos);

    std::filesystem::remove_all(root);
    std::cout << "LocalWorkerTest passed\n";
    return 0;
}
