#include "CpuDiscovery.hpp"
#include "McpApplication.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#ifndef VC_MCP_NNUNET_PYTHON
#error VC_MCP_NNUNET_PYTHON required
#endif
#ifndef VC_MCP_NNUNET_ADAPTER
#error VC_MCP_NNUNET_ADAPTER required
#endif
#ifndef VC_MCP_NNUNET_MODEL
#error VC_MCP_NNUNET_MODEL required
#endif
#ifndef VC_MCP_VOLUME_STAGER
#error VC_MCP_VOLUME_STAGER required
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
}  // namespace

int main()
{
    const auto root = std::filesystem::temp_directory_path() / "vc-mcp-segmentation-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto volume = root / "synthetic.npy";
    // NPY v1.0, little-endian uint16, shape 64^3. Values near the model's training mean.
    const std::string header = "{'descr': '<u2', 'fortran_order': False, 'shape': (64, 64, 64), }";
    std::string padded = header;
    const std::size_t preamble = 10;
    while ((preamble + padded.size() + 1) % 16)
        padded.push_back(' ');
    padded.push_back('\n');
    std::ofstream out(volume, std::ios::binary);
    out.write("\x93NUMPY", 6);
    out.put(1);
    out.put(0);
    const auto n = static_cast<std::uint16_t>(padded.size());
    out.put(char(n & 255));
    out.put(char(n >> 8));
    out.write(padded.data(), padded.size());
    for (int z = 0; z < 64; ++z)
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                std::uint16_t v = std::uint16_t(22000 + ((x + y + z) % 32) * 300);
                out.put(char(v & 255));
                out.put(char(v >> 8));
            }
    out.close();
    const auto zarr = root / "synthetic.zarr";
    std::filesystem::create_directories(zarr / "0");
    std::ofstream(zarr / ".zgroup") << R"({"zarr_format":2})";
    std::ofstream(zarr / "0" / ".zarray")
        << R"({"zarr_format":2,"shape":[64,64,64],"chunks":[64,64,64],"dtype":"<u2","compressor":null,"fill_value":0,"filters":null,"order":"C","dimension_separator":"."})";
    std::ofstream chunk(zarr / "0" / "0.0.0", std::ios::binary);
    for (int z = 0; z < 64; ++z)
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                std::uint16_t v = std::uint16_t(22000 + ((x + y + z) % 32) * 300);
                chunk.put(char(v & 255)); chunk.put(char(v >> 8));
            }
    chunk.close();

    vc::mcp::CpuDiscoveryConfig config;
    config.workRoot = root / "jobs";
    config.nnunetPython = VC_MCP_NNUNET_PYTHON;
    config.nnunetAdapter = VC_MCP_NNUNET_ADAPTER;
    config.volumeStager = VC_MCP_VOLUME_STAGER;
    config.nnunetModelDir = VC_MCP_NNUNET_MODEL;
    config.timeout = std::chrono::minutes(3);
    auto discovery = std::make_shared<vc::mcp::CpuDiscovery>(std::move(config));
    assert(discovery->nnunetAvailable());
    vc::mcp::McpApplication app({"test", "test", "unverified", {}, discovery});
    int rpcId = 1;
    const auto tools = result(app.handle(rpc(rpcId++, "tools/list"))).at("tools");
    const auto segmentationTool =
        std::find_if(tools.begin(), tools.end(), [](const Json& tool) { return tool.value("name", "") == "volume_run_segmentation"; });
    assert(segmentationTool != tools.end());
    assert(segmentationTool->at("_meta").at("ui").at("resourceUri") == "ui://vc/inspector.html");
    struct Case { std::string name; std::string device; Json input; };
    const Json region = {{"x",0},{"y",0},{"z",0},{"width",64},{"height",64},{"depth",64},{"space","ct_l0_xyz"}};
    const std::vector<Case> cases = {
        {"local-zarr-cpu", "cpu", {{"source",{{"kind","local_zarr"},{"path",zarr.string()},{"array_path","0"},{"scale",0},{"voxel_spacing",{2.0,3.0,4.0}},{"origin_xyz",{100.0,200.0,300.0}}}},{"region",region}}},
        {"local-npy-mps", "mps", {{"volume_path",volume.string()}}},
        {"remote-zarr-cpu", "cpu", {{"source",{{"kind","remote_zarr"},{"uri","https://dl.ash2txt.org/community-uploads/bruniss/test_vols/s5_test.zarr/"},{"array_path","0"}}},{"region",region}}}
    };
    for (const auto& testCase : cases) {
        Json args = testCase.input;
        args.update(
            {{"model", "vc-surface-nnunet-058"},
             {"device", testCase.device},
             {"tile_size", 64},
             {"overlap", 0.0},
             {"threshold", 0.5},
             {"cpu_threads", 4},
             {"checkpoint_sha256", "8b90543a3b8063d1158467364fcf825527fb18edc3af852ffcb91906f0e3e763"},
             {"client_request_id", "synthetic-segmentation-" + testCase.name}});
        const auto call = result(app.handle(rpc(rpcId++, "tools/call", {{"name", "volume_run_segmentation"}, {"arguments", args}})));
        const std::string jobId = call.at("structuredContent").at("job_id");
        Json job;
        for (int i = 0; i < 300; ++i) {
            job = result(app.handle(rpc(rpcId++, "tools/call", {{"name", "vc_get_job"}, {"arguments", {{"job_id", jobId}}}})))
                      .at("structuredContent");
            if (job.at("state") == "succeeded" || job.at("state") == "failed")
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (job.at("state") != "succeeded") {
            std::cerr << job.dump(2) << '\n' << app.store()->logs(jobId).dump(2) << '\n';
            return 1;
        }
        const auto inspected =
            result(app.handle(rpc(rpcId++, "tools/call", {{"name", "volume_inspect_segmentation"}, {"arguments", {{"job_id", jobId}}}})))
                .at("structuredContent");
        assert(inspected.at("segmentation") == true);
        assert(inspected.at("manifest").at("backend") == testCase.device);
        assert(inspected.at("manifest").at("input_shape_zyx") == Json::array({64, 64, 64}));
        if (testCase.input.contains("source")) {
            assert(inspected.at("manifest").at("submitted_region_xyz") == region);
            assert(inspected.at("manifest").at("array_path") == "0");
            assert(inspected.at("manifest").at("origin_xyz").size() == 3);
            assert(std::filesystem::is_regular_file(root / "jobs" / jobId / "volume_run_segmentation" / "spatial-metadata.json"));
        }
        assert(inspected.at("probability_preview").get<std::string>().starts_with("data:image/png;base64,"));
    }
    std::filesystem::remove_all(root);
    std::cout << "SegmentationMcpTest passed\n";
    return 0;
}
