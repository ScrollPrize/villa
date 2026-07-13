#include "McpApplication.hpp"

#include <fastmcpp/server/host_origin_guard.hpp>
#include <fastmcpp/server/stdio_server.hpp>
#include <fastmcpp/server/streamable_http_server.hpp>

#include <QCoreApplication>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef VC_MCP_VC_VERSION
#define VC_MCP_VC_VERSION "unknown"
#endif
#ifndef VC_MCP_VC_COMMIT
#define VC_MCP_VC_COMMIT "unknown"
#endif
#ifndef VC_MCP_CONTAINER_DIGEST
#define VC_MCP_CONTAINER_DIGEST "unverified"
#endif

namespace
{

volatile std::sig_atomic_t stopRequested = 0;

void requestStop(int)
{
    stopRequested = 1;
}

std::string environment(const char* name, std::string fallback = {})
{
    if (const char* value = std::getenv(name))
        return value;
    return fallback;
}

int httpPort()
{
    const std::string text = environment("VC_MCP_PORT", "18080");
    std::size_t consumed = 0;
    const int port = std::stoi(text, &consumed);
    if (consumed != text.size() || port < 1 || port > 65535)
        throw std::runtime_error("VC_MCP_PORT must be an integer from 1 to 65535");
    return port;
}

std::chrono::seconds workerTimeout()
{
    const std::string text = environment("VC_MCP_TIMEOUT_SECONDS", "21600");
    std::size_t consumed = 0;
    const long long seconds = std::stoll(text, &consumed);
    if (consumed != text.size() || seconds < 1 || seconds > 604800)
        throw std::runtime_error("VC_MCP_TIMEOUT_SECONDS must be from 1 to 604800");
    return std::chrono::seconds(seconds);
}

std::filesystem::path workRoot()
{
    std::filesystem::path root = environment("VC_MCP_WORK_ROOT");
    if (root.empty()) {
        const auto home = environment("HOME");
        if (home.empty())
            throw std::runtime_error("set VC_MCP_WORK_ROOT to an absolute directory");
        root = std::filesystem::path(home) / ".vc-mcp" / "jobs";
    }
    return root;
}

std::shared_ptr<vc::mcp::VolumeCartographer> localWorker(const char* argv0)
{
    std::filesystem::path executable = environment("VC_MCP_GROW_EXECUTABLE");
    if (executable.empty()) {
        std::error_code error;
        const auto serverPath = std::filesystem::weakly_canonical(argv0, error);
        if (error)
            throw std::runtime_error("set VC_MCP_GROW_EXECUTABLE to an absolute path");
        executable = serverPath.parent_path() / "vc_grow_seg_from_seed";
    }
    return std::make_shared<vc::mcp::LocalVolumeCartographer>(vc::mcp::LocalWorkerConfig{std::move(executable), workRoot(), workerTimeout()});
}

std::shared_ptr<vc::mcp::CpuDiscovery> cpuDiscovery()
{
    vc::mcp::CpuDiscoveryConfig config;
    config.workRoot = workRoot();
    config.timeout = workerTimeout();
    const auto nnunetPython = environment("VC_MCP_NNUNET_PYTHON");
    const auto nnunetAdapter = environment("VC_MCP_NNUNET_ADAPTER");
    const auto nnunetModel = environment("VC_MCP_NNUNET_MODEL_DIR");
    const auto volumeStager = environment("VC_MCP_VOLUME_STAGER");
    const auto analysisPython = environment("VC_MCP_ANALYSIS_PYTHON", nnunetPython);
    const auto surfaceBundleAdapter = environment("VC_MCP_SURFACE_BUNDLE_ADAPTER");
    const auto structuralEvidenceAdapter = environment("VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER");
    const auto evidenceFusionAdapter = environment("VC_MCP_EVIDENCE_FUSION_ADAPTER");
    const auto reviewAdapter = environment("VC_MCP_REVIEW_ADAPTER");
    if (!nnunetPython.empty())
        config.nnunetPython = nnunetPython;
    if (!nnunetAdapter.empty())
        config.nnunetAdapter = nnunetAdapter;
    if (!nnunetModel.empty())
        config.nnunetModelDir = nnunetModel;
    if (!volumeStager.empty())
        config.volumeStager = volumeStager;
    if (!analysisPython.empty())
        config.analysisPython = analysisPython;
    if (!surfaceBundleAdapter.empty())
        config.surfaceBundleAdapter = surfaceBundleAdapter;
    if (!structuralEvidenceAdapter.empty())
        config.structuralEvidenceAdapter = structuralEvidenceAdapter;
    if (!evidenceFusionAdapter.empty())
        config.evidenceFusionAdapter = evidenceFusionAdapter;
    if (!reviewAdapter.empty())
        config.reviewAdapter = reviewAdapter;
    const auto dinov3 = environment("VC_MCP_DINOV3_EXECUTABLE");
    if (!dinov3.empty())
        config.dinov3Executable = std::filesystem::path(dinov3);
    const auto dinovol = environment("VC_MCP_DINOVOL_EXECUTABLE");
    if (!dinovol.empty())
        config.dinovolExecutable = std::filesystem::path(dinovol);
    const auto dinovolPython = environment("VC_MCP_DINOVOL_PYTHON");
    const auto dinovolAdapter = environment("VC_MCP_DINOVOL_ADAPTER");
    const auto dinovolRepository = environment("VC_MCP_DINOVOL_REPOSITORY");
    const auto dinovolCheckpoint = environment("VC_MCP_DINOVOL_CHECKPOINT");
    config.dinovolRepositoryCommit = environment("VC_MCP_DINOVOL_REPOSITORY_COMMIT");
    if (!dinovolPython.empty())
        config.dinovolPython = dinovolPython;
    if (!dinovolAdapter.empty())
        config.dinovolAdapter = dinovolAdapter;
    if (!dinovolRepository.empty())
        config.dinovolRepository = dinovolRepository;
    if (!dinovolCheckpoint.empty())
        config.dinovolCheckpoint = dinovolCheckpoint;
    const auto inkModelPython = environment("VC_MCP_INK_MODEL_PYTHON");
    const auto inkModelAdapter = environment("VC_MCP_INK_MODEL_ADAPTER");
    const auto inkModelRepository = environment("VC_MCP_INK_MODEL_REPOSITORY");
    const auto inkModelCheckpoint = environment("VC_MCP_INK_MODEL_CHECKPOINT");
    config.inkModelRepositoryCommit = environment("VC_MCP_INK_MODEL_REPOSITORY_COMMIT");
    if (!inkModelPython.empty())
        config.inkModelPython = inkModelPython;
    if (!inkModelAdapter.empty())
        config.inkModelAdapter = inkModelAdapter;
    if (!inkModelRepository.empty())
        config.inkModelRepository = inkModelRepository;
    if (!inkModelCheckpoint.empty())
        config.inkModelCheckpoint = inkModelCheckpoint;
    return std::make_shared<vc::mcp::CpuDiscovery>(std::move(config));
}

int runHttp(const vc::mcp::McpApplication& app)
{
    const std::string host = environment("VC_MCP_HOST", "127.0.0.1");
    if (host != "127.0.0.1" && host != "::1" && host != "localhost")
        throw std::runtime_error(
            "Phase 2 preview only permits a loopback VC_MCP_HOST; deploy behind authenticated "
            "TLS after the production auth layer is implemented");

    const std::string token = environment("VC_MCP_AUTH_TOKEN");
    if (token.size() < 32)
        throw std::runtime_error("Streamable HTTP requires VC_MCP_AUTH_TOKEN with at least 32 characters");

    fastmcpp::server::StreamableHttpServerWrapper
        server(app.handler(), host, httpPort(), "/mcp", token, "", {{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"}});
    fastmcpp::server::HostOriginGuardOptions guard;
    guard.mode = fastmcpp::server::HostOriginProtectionMode::Auto;
    guard.allowed_hosts = std::vector<std::string>{host, "localhost", "127.0.0.1", "::1"};
    server.set_host_origin_guard(std::move(guard));

    if (!server.start())
        throw std::runtime_error("failed to start Streamable HTTP transport");

    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
    std::cerr << "VC MCP Streamable HTTP listening on http://" << server.host() << ':' << server.port() << server.mcp_path() << '\n';
    while (stopRequested == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.stop();
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        QCoreApplication qtApplication(argc, argv);
        if (argc < 1 || argv[0] == nullptr)
            throw std::runtime_error("cannot determine vc_mcp_server executable path");
        vc::mcp::McpApplication app({VC_MCP_VC_VERSION, VC_MCP_VC_COMMIT, VC_MCP_CONTAINER_DIGEST, localWorker(argv[0]), cpuDiscovery()});
        const std::string transport = environment("VC_MCP_TRANSPORT", "stdio");
        if (transport == "stdio") {
            fastmcpp::server::StdioServerWrapper server(app.handler());
            return server.run() ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (transport == "streamable-http")
            return runHttp(app);
        throw std::runtime_error("VC_MCP_TRANSPORT must be either 'stdio' or 'streamable-http'");
    } catch (const std::exception& error) {
        std::cerr << "vc_mcp_server: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
