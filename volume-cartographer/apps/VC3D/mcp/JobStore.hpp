#pragma once

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vc::mcp
{

class VolumeCartographer;
class CpuDiscovery;

using Json = nlohmann::json;

// Process-local job state for the local MCP server. Jobs and artifacts remain
// on disk, but state is not restored after a server restart.
class JobStore
{
public:
    explicit JobStore(std::shared_ptr<VolumeCartographer> worker = {}, std::shared_ptr<CpuDiscovery> discovery = {});
    ~JobStore();

    JobStore(const JobStore&) = delete;
    JobStore& operator=(const JobStore&) = delete;

    Json submitGrow(const Json& request);
    Json submitGenerateSurface(const Json& request);
    Json submitDiscovery(const std::string& operation, const Json& request);
    Json get(const std::string& jobId) const;
    Json resolveArtifactReference(const Json& reference) const;
    Json logs(const std::string& jobId) const;
    Json cancel(const std::string& jobId);

private:
    struct Job {
        Json document;
        std::vector<std::string> logLines;
        std::shared_ptr<std::atomic<bool>> cancelRequested;
    };

    Json submitSurface(const Json& request, const std::string& operation);
    void runGrow(const std::string& jobId);
    void setStage(const std::string& jobId, const std::string& stage, const std::string& state);
    void runDiscovery(const std::string& jobId);
    void runFakeGrow(const std::string& jobId, const std::shared_ptr<std::atomic<bool>>& cancelled);
    void appendLog(const std::string& jobId, std::string line);
    void updateWorkerProgress(const std::string& jobId, const std::string& line);
    void transition(const std::string& jobId, const std::string& state, std::optional<Json> progress, std::string logLine);

    std::shared_ptr<VolumeCartographer> worker_;
    std::shared_ptr<CpuDiscovery> discovery_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Job> jobs_;
    std::unordered_map<std::string, std::string> requestIds_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stopping_{false};
    std::atomic<std::uint64_t> nextId_{1};
};

Json validateAndNormalizeGrowRequest(const Json& request);
std::string utcNow();

}  // namespace vc::mcp
