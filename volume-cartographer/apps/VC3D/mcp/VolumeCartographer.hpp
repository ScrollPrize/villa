#pragma once

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <string>

namespace vc::mcp
{

using Json = nlohmann::json;

struct LocalWorkerConfig {
    std::filesystem::path growExecutable;
    std::filesystem::path workRoot;
    std::chrono::seconds timeout{std::chrono::hours(6)};
};

struct WorkerResult {
    int exitCode{0};
    Json commandManifest;
    Json artifacts{Json::array()};
};

class VolumeCartographer
{
public:
    using LogCallback = std::function<void(std::string)>;

    virtual ~VolumeCartographer() = default;
    virtual std::string executionMode() const = 0;
    virtual Json validateAndNormalizeGrow(const Json& request) const = 0;
    virtual WorkerResult growSurface(const std::string& jobId, const Json& normalized, const std::atomic<bool>& cancelRequested, LogCallback log) const = 0;
};

class LocalVolumeCartographer final : public VolumeCartographer
{
public:
    explicit LocalVolumeCartographer(LocalWorkerConfig config);

    std::string executionMode() const override { return "local-process"; }
    Json validateAndNormalizeGrow(const Json& request) const override;
    WorkerResult growSurface(const std::string& jobId, const Json& normalized, const std::atomic<bool>& cancelRequested, LogCallback log) const override;

private:
    LocalWorkerConfig config_;
};

Json validateAndNormalizeLocalGrowRequest(const Json& request);

}  // namespace vc::mcp
