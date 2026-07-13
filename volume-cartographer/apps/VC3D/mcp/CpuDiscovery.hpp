#pragma once

#include "VolumeCartographer.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace vc::mcp
{

struct CpuDiscoveryConfig {
    std::filesystem::path workRoot;
    std::optional<std::filesystem::path> nnunetPython;
    std::optional<std::filesystem::path> nnunetAdapter;
    std::optional<std::filesystem::path> volumeStager;
    std::optional<std::filesystem::path> analysisPython;
    std::optional<std::filesystem::path> surfaceBundleAdapter;
    std::optional<std::filesystem::path> structuralEvidenceAdapter;
    std::optional<std::filesystem::path> evidenceFusionAdapter;
    std::optional<std::filesystem::path> reviewAdapter;
    std::optional<std::filesystem::path> nnunetModelDir;
    std::optional<std::filesystem::path> dinov3Executable;
    std::optional<std::filesystem::path> dinovolExecutable;  // legacy wrapper
    std::optional<std::filesystem::path> dinovolPython;
    std::optional<std::filesystem::path> dinovolAdapter;
    std::optional<std::filesystem::path> dinovolRepository;
    std::optional<std::filesystem::path> dinovolCheckpoint;
    std::string dinovolRepositoryCommit;
    std::optional<std::filesystem::path> inkModelPython;
    std::optional<std::filesystem::path> inkModelAdapter;
    std::optional<std::filesystem::path> inkModelRepository;
    std::optional<std::filesystem::path> inkModelCheckpoint;
    std::string inkModelRepositoryCommit;
    std::chrono::seconds timeout{std::chrono::hours(2)};
};

class CpuDiscovery
{
public:
    using LogCallback = std::function<void(std::string)>;

    explicit CpuDiscovery(CpuDiscoveryConfig config);

    bool nnunetAvailable() const noexcept
    {
        return config_.nnunetPython && config_.nnunetAdapter && config_.volumeStager && config_.nnunetModelDir;
    }
    bool surfaceBundleAvailable() const noexcept { return config_.analysisPython && config_.surfaceBundleAdapter && config_.volumeStager; }
    bool structuralEvidenceAvailable() const noexcept { return config_.analysisPython && config_.structuralEvidenceAdapter; }
    bool evidenceFusionAvailable() const noexcept { return config_.analysisPython && config_.evidenceFusionAdapter; }
    bool reviewAvailable() const noexcept { return config_.analysisPython && config_.reviewAdapter; }
    bool dinov3Available() const noexcept { return config_.dinov3Executable.has_value(); }
    bool dinovolAvailable() const noexcept
    {
        return config_.dinovolPython && config_.dinovolAdapter && config_.dinovolRepository && config_.dinovolCheckpoint &&
               !config_.dinovolRepositoryCommit.empty();
    }
    bool inkModelAvailable() const noexcept
    {
        return config_.inkModelPython && config_.inkModelAdapter && config_.inkModelRepository && config_.inkModelCheckpoint &&
               !config_.inkModelRepositoryCommit.empty();
    }
    Json validate(const std::string& operation, const Json& request) const;
    WorkerResult run(const std::string& operation, const std::string& jobId, const Json& normalized, const std::atomic<bool>& cancelRequested, LogCallback log) const;

private:
    CpuDiscoveryConfig config_;
};

}  // namespace vc::mcp
