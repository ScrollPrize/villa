#include "JobStore.hpp"

#include <fastmcpp/exceptions.hpp>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

using vc::mcp::JobStore;
using vc::mcp::Json;

namespace
{

Json request(const std::filesystem::path& prediction, std::string requestId = "test-grow-1")
{
    return {
        {"prediction_path", prediction.string()},
        {"prediction_space", "ct_l0_xyz"},
        {"seed", {{"x", 1}, {"y", 2}, {"z", 3}, {"space", "ct_l2_xyz"}}},
        {"profile", "scroll3-conservative-v1"},
        {"limits", {{"max_generations", 4}}},
        {"client_request_id", std::move(requestId)}};
}

Json structured(const Json& callResult)
{
    return callResult.at("structuredContent");
}

}  // namespace

int main()
{
    const auto prediction = std::filesystem::temp_directory_path() / "vc-mcp-test-prediction";
    std::filesystem::create_directories(prediction);
    const auto normalized = vc::mcp::validateAndNormalizeGrowRequest(request(prediction));
    assert(normalized.at("coordinates").at("vc_input").at("x") == 4.0);
    assert(normalized.at("coordinates").at("vc_input").at("y") == 8.0);
    assert(normalized.at("coordinates").at("vc_input").at("z") == 12.0);
    assert(normalized.at("coordinates").at("vc_input").at("space") == "ct_l0_xyz");

    auto sameLevel = request(prediction, "same-level");
    sameLevel["prediction_space"] = "ct_l2_xyz";
    const auto normalizedSameLevel = vc::mcp::validateAndNormalizeGrowRequest(sameLevel);
    assert(normalizedSameLevel.at("coordinates").at("vc_input").at("x") == 1.0);
    assert(normalizedSameLevel.at("coordinates").at("ct_l0").at("x") == 4.0);
    assert(normalizedSameLevel.at("coordinates").at("vc_input").at("space") == "ct_l2_xyz");

    bool rejected = false;
    try {
        auto invalid = request(prediction);
        invalid["prediction_path"] = "relative/prediction.zarr";
        vc::mcp::validateAndNormalizeGrowRequest(invalid);
    } catch (const fastmcpp::ValidationError&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        auto invalid = request(prediction);
        invalid["prediction_path"] = (prediction / "missing").string();
        vc::mcp::validateAndNormalizeGrowRequest(invalid);
    } catch (const fastmcpp::ValidationError&) {
        rejected = true;
    }
    assert(rejected);

    auto remote = request(prediction, "remote-validation");
    remote.erase("prediction_path");
    remote["prediction_uri"] = "https://dl.ash2txt.org/other/dev/scrolls/5/volumes/53keV_7.91um.zarr/";
    remote["voxel_size_um"] = 7.91;
    const auto normalizedRemote = vc::mcp::validateAndNormalizeGrowRequest(remote);
    assert(normalizedRemote.at("prediction_source") == remote.at("prediction_uri"));

    rejected = false;
    try {
        remote["prediction_uri"] = "https://attacker.invalid/prediction.zarr";
        vc::mcp::validateAndNormalizeGrowRequest(remote);
    } catch (const fastmcpp::ValidationError&) {
        rejected = true;
    }
    assert(rejected);

    JobStore store;
    const auto first = structured(store.submitGrow(request(prediction)));
    const auto duplicate = structured(store.submitGrow(request(prediction)));
    assert(first.at("job_id") == duplicate.at("job_id"));

    rejected = false;
    try {
        store.submitGenerateSurface(request(prediction));
    } catch (const fastmcpp::ValidationError&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        auto conflicting = request(prediction);
        conflicting["seed"]["x"] = 99;
        store.submitGrow(conflicting);
    } catch (const fastmcpp::ValidationError&) {
        rejected = true;
    }
    assert(rejected);

    const std::string jobId = first.at("job_id").get<std::string>();
    Json job;
    for (int attempt = 0; attempt < 100; ++attempt) {
        job = store.get(jobId);
        if (job.at("state") == "succeeded")
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(job.at("state") == "succeeded");
    assert(job.at("job_revision").get<int>() >= 3);
    assert(!store.logs(jobId).at("lines").empty());

    const auto cancellable = structured(store.submitGrow(request(prediction, "test-grow-cancel")));
    const std::string cancellableId = cancellable.at("job_id").get<std::string>();
    assert(store.cancel(cancellableId).at("state") == "cancelling");
    for (int attempt = 0; attempt < 100; ++attempt) {
        job = store.get(cancellableId);
        if (job.at("state") == "cancelled")
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(job.at("state") == "cancelled");

    std::filesystem::remove_all(prediction);
    std::cout << "JobStoreTest passed\n";
    return 0;
}
