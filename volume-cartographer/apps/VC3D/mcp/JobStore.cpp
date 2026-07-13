#include "JobStore.hpp"
#include "CpuDiscovery.hpp"
#include "VolumeCartographer.hpp"

#include <fastmcpp/exceptions.hpp>

#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace vc::mcp
{
namespace
{

constexpr std::size_t kMaxLogLines = 500;

std::string makeId(std::string_view prefix, std::uint64_t sequence)
{
    const auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::ostringstream out;
    out << prefix << std::hex << ticks << '-' << sequence;
    return out.str();
}

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw fastmcpp::ValidationError(message);
}

Json callResult(const Json& structured, const std::string& text)
{
    return Json{{"content", Json::array({Json{{"type", "text"}, {"text", text}}})}, {"structuredContent", structured}};
}

}  // namespace

std::string utcNow()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

Json validateAndNormalizeGrowRequest(const Json& request)
{
    return validateAndNormalizeLocalGrowRequest(request);
}

JobStore::JobStore(std::shared_ptr<VolumeCartographer> worker, std::shared_ptr<CpuDiscovery> discovery)
    : worker_(std::move(worker)), discovery_(std::move(discovery))
{
}

JobStore::~JobStore()
{
    stopping_.store(true);
    {
        std::lock_guard lock(mutex_);
        for (auto& [id, job] : jobs_)
            job.cancelRequested->store(true);
    }
    for (auto& worker : workers_)
        if (worker.joinable())
            worker.join();
}

Json JobStore::submitGrow(const Json& request)
{
    return submitSurface(request, "grow_surface");
}

Json JobStore::submitGenerateSurface(const Json& request)
{
    return submitSurface(request, "generate_surface");
}

Json JobStore::submitSurface(const Json& request, const std::string& operation)
{
    const Json normalized = worker_ ? worker_->validateAndNormalizeGrow(request) : validateAndNormalizeGrowRequest(request);
    const std::string requestId = request.at("client_request_id").get<std::string>();

    std::string jobId;
    Json response;
    {
        std::lock_guard lock(mutex_);
        if (const auto existing = requestIds_.find(requestId); existing != requestIds_.end()) {
            const auto& job = jobs_.at(existing->second).document;
            require(job.at("operation") == operation, "client_request_id was already used for a different operation");
            require(job.at("normalized_input") == normalized, "client_request_id was already used with different input");
            response =
                {{"job_id", job.at("job_id")},
                 {"state", job.at("state")},
                 {"operation", job.at("operation")},
                 {"job_resource", "vc://jobs/" + existing->second},
                 {"log_resource", "vc://jobs/" + existing->second + "/logs"},
                 {"submitted_at", job.at("created_at")}};
            return callResult(response, "Existing VC growth job " + existing->second);
        }

        jobId = makeId("job_", nextId_.fetch_add(1));
        const std::string createdAt = utcNow();
        Json job =
            {{"job_id", jobId},
             {"operation", operation},
             {"state", "queued"},
             {"progress", {{"completed", 0}, {"total", 100}, {"unit", "percent"}, {"message", "Queued for worker"}}},
             {"created_at", createdAt},
             {"started_at", nullptr},
             {"finished_at", nullptr},
             {"job_revision", 1},
             {"normalized_input", normalized}};
        if (operation == "generate_surface")
            job["stages"] = Json::array(
                {{{"name", "validate_input"}, {"state", "succeeded"}},
                 {{"name", "grow_surface"}, {"state", "queued"}},
                 {{"name", "validate_surface"}, {"state", "pending"}},
                 {{"name", "render_preview"}, {"state", "pending"}},
                 {{"name", "hash_artifacts"}, {"state", "pending"}}});
        jobs_.emplace(jobId, Job{std::move(job), {"Job accepted by in-memory queue"}, std::make_shared<std::atomic<bool>>(false)});
        requestIds_.emplace(requestId, jobId);
        response =
            {{"job_id", jobId},
             {"state", "queued"},
             {"operation", operation},
             {"job_resource", "vc://jobs/" + jobId},
             {"log_resource", "vc://jobs/" + jobId + "/logs"},
             {"submitted_at", createdAt}};
        workers_.emplace_back([this, jobId] { runGrow(jobId); });
    }
    return callResult(response, "Queued VC growth job " + jobId);
}

Json JobStore::submitDiscovery(const std::string& operation, const Json& request)
{
    require(discovery_ != nullptr, "CPU discovery service is unavailable");
    Json resolved = request;
    for (const auto* key : {"surface", "surface_volume", "villa", "dinovol", "stability", "surface_a", "surface_b", "grid", "baseline", "ranking", "comparison", "queue"}) {
        if (!request.contains(key) || !request.at(key).is_object())
            continue;
        const auto artifact = resolveArtifactReference(request.at(key));
        resolved[std::string(key) + "_path"] = artifact.at("path");
        resolved[std::string(key) + "_media_type"] = artifact.value("media_type", "");
    }
    if (request.contains("variants") && request.at("variants").is_array()) {
        resolved["resolved_variants"] = Json::array();
        for (const auto& reference : request.at("variants")) {
            const auto artifact = resolveArtifactReference(reference);
            resolved["resolved_variants"].push_back(
                {{"reference", reference}, {"path", artifact.at("path")}, {"media_type", artifact.value("media_type", "")}});
        }
    }
    if (operation == "metric_evaluate_labels" && request.contains("assessments") && request.at("assessments").is_array()) {
        resolved["resolved_assessments"] = Json::array();
        for (const auto& reference : request.at("assessments")) {
            const auto artifact = resolveArtifactReference(reference);
            resolved["resolved_assessments"].push_back(
                {{"reference", reference}, {"path", artifact.at("path")}, {"media_type", artifact.value("media_type", "")}});
        }
    }
    if (request.contains("candidates") && request.at("candidates").is_array()) {
        resolved["resolved_candidates"] = Json::array();
        for (const auto& candidate : request.at("candidates")) {
            Json item = {{"id", candidate.at("id")}, {"resolved", Json::object()}};
            for (const auto* key : {"geometry", "alignment", "grid", "stability"}) {
                if (!candidate.contains(key))
                    continue;
                const auto artifact = resolveArtifactReference(candidate.at(key));
                item["resolved"][key] = artifact.at("path");
                item["resolved"][std::string(key) + "_media_type"] = artifact.value("media_type", "");
            }
            resolved["resolved_candidates"].push_back(std::move(item));
        }
    }
    const Json normalized = discovery_->validate(operation, resolved);
    const std::string requestId = operation + ":" + request.at("client_request_id").get<std::string>();
    std::string jobId;
    Json response;
    {
        std::lock_guard lock(mutex_);
        if (const auto existing = requestIds_.find(requestId); existing != requestIds_.end()) {
            const auto& job = jobs_.at(existing->second).document;
            require(job.at("normalized_input") == normalized, "client_request_id was already used with different input");
            response =
                {{"job_id", job.at("job_id")},
                 {"state", job.at("state")},
                 {"operation", operation},
                 {"job_resource", "vc://jobs/" + existing->second},
                 {"log_resource", "vc://jobs/" + existing->second + "/logs"},
                 {"submitted_at", job.at("created_at")}};
            return callResult(response, "Existing CPU discovery job " + existing->second);
        }
        jobId = makeId("job_", nextId_.fetch_add(1));
        const std::string createdAt = utcNow();
        Json job =
            {{"job_id", jobId},
             {"operation", operation},
             {"state", "queued"},
             {"progress", {{"completed", 0}, {"total", 100}, {"unit", "percent"}, {"message", "Queued for CPU worker"}}},
             {"created_at", createdAt},
             {"started_at", nullptr},
             {"finished_at", nullptr},
             {"job_revision", 1},
             {"normalized_input", normalized}};
        jobs_.emplace(jobId, Job{std::move(job), {"CPU discovery job accepted"}, std::make_shared<std::atomic<bool>>(false)});
        requestIds_.emplace(requestId, jobId);
        response =
            {{"job_id", jobId},
             {"state", "queued"},
             {"operation", operation},
             {"job_resource", "vc://jobs/" + jobId},
             {"log_resource", "vc://jobs/" + jobId + "/logs"},
             {"submitted_at", createdAt}};
        workers_.emplace_back([this, jobId] { runDiscovery(jobId); });
    }
    return callResult(response, "Queued CPU discovery job " + jobId);
}

Json JobStore::get(const std::string& jobId) const
{
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(jobId);
    if (found == jobs_.end())
        throw fastmcpp::NotFoundError("job not found: " + jobId);
    return found->second.document;
}

Json JobStore::resolveArtifactReference(const Json& reference) const
{
    require(reference.is_object(), "artifact reference must be an object");
    require(reference.contains("job_id") && reference.at("job_id").is_string(), "artifact reference job_id is required");
    require(reference.contains("artifact_id") && reference.at("artifact_id").is_string(), "artifact reference artifact_id is required");
    const auto jobId = reference.at("job_id").get<std::string>();
    const auto artifactId = reference.at("artifact_id").get<std::string>();
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(jobId);
    if (found == jobs_.end())
        throw fastmcpp::NotFoundError("artifact job not found: " + jobId);
    const auto& job = found->second.document;
    require(job.at("state") == "succeeded", "artifact job has not succeeded");
    require(job.contains("artifacts") && job.at("artifacts").is_array(), "job has no artifacts");
    for (const auto& artifact : job.at("artifacts")) {
        if (artifact.value("artifact_id", "") != artifactId)
            continue;
        require(artifact.contains("path") && artifact.at("path").is_string(), "artifact has no local path");
        std::error_code error;
        auto path = std::filesystem::weakly_canonical(artifact.at("path").get<std::string>(), error);
        require(!error && std::filesystem::exists(path), "artifact path no longer exists");
        Json resolved = artifact;
        resolved["path"] = path.string();
        resolved["job_id"] = jobId;
        return resolved;
    }
    throw fastmcpp::NotFoundError("artifact not found: " + artifactId);
}

Json JobStore::logs(const std::string& jobId) const
{
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(jobId);
    if (found == jobs_.end())
        throw fastmcpp::NotFoundError("job not found: " + jobId);
    Json lines = Json::array();
    const auto begin = found->second.logLines.size() > kMaxLogLines ? found->second.logLines.end() - kMaxLogLines : found->second.logLines.begin();
    for (auto it = begin; it != found->second.logLines.end(); ++it)
        lines.push_back(*it);
    return {{"job_id", jobId}, {"lines", std::move(lines)}, {"cursor", found->second.logLines.size()}};
}

Json JobStore::cancel(const std::string& jobId)
{
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(jobId);
    if (found == jobs_.end())
        throw fastmcpp::NotFoundError("job not found: " + jobId);
    auto& job = found->second;
    const std::string state = job.document.at("state").get<std::string>();
    if (state == "succeeded" || state == "failed" || state == "cancelled")
        return job.document;
    job.cancelRequested->store(true);
    job.document["state"] = "cancelling";
    job.document["job_revision"] = job.document.at("job_revision").get<std::uint64_t>() + 1;
    job.logLines.push_back(utcNow() + " Cancellation requested");
    return job.document;
}

void JobStore::appendLog(const std::string& jobId, std::string line)
{
    std::lock_guard lock(mutex_);
    auto& lines = jobs_.at(jobId).logLines;
    lines.push_back(utcNow() + " " + std::move(line));
    if (lines.size() > kMaxLogLines)
        lines.erase(lines.begin(), lines.begin() + (lines.size() - kMaxLogLines));
}

void JobStore::updateWorkerProgress(const std::string& jobId, const std::string& line)
{
    if (!line.starts_with("progress "))
        return;
    const auto slash = line.find('/', 9);
    if (slash == std::string::npos)
        return;
    try {
        const int completed = std::stoi(line.substr(9, slash - 9));
        const int total = std::stoi(line.substr(slash + 1));
        if (completed < 0 || total < 1 || completed > total)
            return;
        std::lock_guard lock(mutex_);
        auto& document = jobs_.at(jobId).document;
        document["progress"] = {{"completed", completed}, {"total", total}, {"unit", "patches"}, {"message", "Segmenting volume"}};
        document["job_revision"] = document.at("job_revision").get<std::uint64_t>() + 1;
    } catch (...) {
    }
}

void JobStore::setStage(const std::string& jobId, const std::string& stage, const std::string& state)
{
    std::lock_guard lock(mutex_);
    auto& document = jobs_.at(jobId).document;
    if (!document.contains("stages"))
        return;
    for (auto& item : document["stages"])
        if (item.value("name", "") == stage) {
            item["state"] = state;
            if (state == "running")
                item["started_at"] = utcNow();
            if (state == "succeeded" || state == "failed" || state == "skipped")
                item["finished_at"] = utcNow();
            break;
        }
    document["job_revision"] = document.at("job_revision").get<std::uint64_t>() + 1;
}

void JobStore::transition(const std::string& jobId, const std::string& state, std::optional<Json> progress, std::string logLine)
{
    std::lock_guard lock(mutex_);
    auto& job = jobs_.at(jobId);
    job.document["state"] = state;
    if (progress)
        job.document["progress"] = std::move(*progress);
    job.document["job_revision"] = job.document.at("job_revision").get<std::uint64_t>() + 1;
    if (state == "running" && job.document.at("started_at").is_null())
        job.document["started_at"] = utcNow();
    if (state == "succeeded" || state == "failed" || state == "cancelled")
        job.document["finished_at"] = utcNow();
    job.logLines.push_back(utcNow() + " " + std::move(logLine));
}

void JobStore::runGrow(const std::string& jobId)
{
    std::shared_ptr<std::atomic<bool>> cancelled;
    Json input;
    {
        std::lock_guard lock(mutex_);
        cancelled = jobs_.at(jobId).cancelRequested;
        input = jobs_.at(jobId).document.at("normalized_input");
    }
    if (worker_) {
        setStage(jobId, "grow_surface", "running");
        transition(jobId, "starting", std::nullopt, "Starting local VC process");
        if (cancelled->load()) {
            setStage(jobId, "grow_surface", "skipped");
            setStage(jobId, "validate_surface", "skipped");
            setStage(jobId, "render_preview", "skipped");
            setStage(jobId, "hash_artifacts", "skipped");
            transition(jobId, "cancelled", std::nullopt, "Cancelled before process start");
            return;
        }
        transition(jobId, "running", std::nullopt, "Local VC process started");
        try {
            auto result =
                worker_->growSurface(jobId, input, *cancelled, [this, jobId](std::string line) { appendLog(jobId, std::move(line)); });
            bool hasPreview = false;
            {
                std::lock_guard lock(mutex_);
                auto& document = jobs_.at(jobId).document;
                document["command_manifest"] = std::move(result.commandManifest);
                document["artifacts"] = std::move(result.artifacts);
                for (const auto& artifact : document["artifacts"]) {
                    if (artifact.value("artifact_id", "") == "surface" && artifact.contains("metadata"))
                        document["surface"] = artifact["metadata"].value("surface", Json::object());
                    if (artifact.value("artifact_id", "") == "generation-preview")
                        hasPreview = true;
                }
            }
            setStage(jobId, "grow_surface", "succeeded");
            setStage(jobId, "validate_surface", "succeeded");
            setStage(jobId, "render_preview", hasPreview ? "succeeded" : "skipped");
            setStage(jobId, "hash_artifacts", "succeeded");
            transition(
                jobId,
                "succeeded",
                Json{{"completed", 100}, {"total", 100}, {"unit", "percent"}, {"message", "VC growth complete"}},
                "Local VC process completed");
        } catch (const std::exception& error) {
            setStage(jobId, "grow_surface", cancelled->load() ? "skipped" : "failed");
            setStage(jobId, "validate_surface", "skipped");
            setStage(jobId, "render_preview", "skipped");
            setStage(jobId, "hash_artifacts", "skipped");
            if (cancelled->load()) {
                transition(jobId, "cancelled", std::nullopt, error.what());
            } else {
                {
                    std::lock_guard lock(mutex_);
                    jobs_.at(jobId).document["error"] =
                        {{"code", "VC_PROCESS_FAILED"}, {"message", error.what()}, {"retryable", false}, {"log_resource", "vc://jobs/" + jobId + "/logs"}};
                }
                transition(jobId, "failed", std::nullopt, error.what());
            }
        }
        return;
    }
    runFakeGrow(jobId, cancelled);
}

void JobStore::runDiscovery(const std::string& jobId)
{
    std::shared_ptr<std::atomic<bool>> cancelled;
    Json input;
    std::string operation;
    {
        std::lock_guard lock(mutex_);
        auto& document = jobs_.at(jobId).document;
        cancelled = jobs_.at(jobId).cancelRequested;
        input = document.at("normalized_input");
        operation = document.at("operation").get<std::string>();
    }
    transition(jobId, "starting", std::nullopt, "Starting CPU discovery worker");
    if (cancelled->load()) {
        transition(jobId, "cancelled", std::nullopt, "Cancelled before CPU work started");
        return;
    }
    transition(
        jobId,
        "running",
        Json{{"completed", 5}, {"total", 100}, {"unit", "percent"}, {"message", "Running CPU discovery"}},
        "CPU discovery started");
    try {
        auto result = discovery_->run(operation, jobId, input, *cancelled, [this, jobId](std::string line) {
            updateWorkerProgress(jobId, line);
            appendLog(jobId, std::move(line));
        });
        {
            std::lock_guard lock(mutex_);
            auto& document = jobs_.at(jobId).document;
            document["command_manifest"] = std::move(result.commandManifest);
            document["artifacts"] = std::move(result.artifacts);
        }
        transition(
            jobId,
            "succeeded",
            Json{{"completed", 100}, {"total", 100}, {"unit", "percent"}, {"message", "CPU discovery complete"}},
            "CPU discovery completed");
    } catch (const std::exception& error) {
        if (cancelled->load()) {
            transition(jobId, "cancelled", std::nullopt, error.what());
        } else {
            {
                std::lock_guard lock(mutex_);
                jobs_.at(jobId).document["error"] =
                    {{"code", "CPU_DISCOVERY_FAILED"}, {"message", error.what()}, {"retryable", false}, {"log_resource", "vc://jobs/" + jobId + "/logs"}};
            }
            transition(jobId, "failed", std::nullopt, error.what());
        }
    }
}

void JobStore::runFakeGrow(const std::string& jobId, const std::shared_ptr<std::atomic<bool>>& cancelled)
{
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(20ms);
    if (stopping_.load())
        return;
    if (cancelled->load()) {
        transition(jobId, "cancelled", std::nullopt, "Fake worker cancelled");
        return;
    }
    transition(jobId, "starting", std::nullopt, "Fake worker is starting");
    std::this_thread::sleep_for(20ms);
    if (stopping_.load())
        return;
    if (cancelled->load()) {
        transition(jobId, "cancelled", std::nullopt, "Fake worker cancelled");
        return;
    }
    transition(
        jobId,
        "running",
        Json{{"completed", 0}, {"total", 100}, {"unit", "percent"}, {"message", "Growing fake surface"}},
        "Fake surface growth started");
    for (int progress : {25, 50, 75, 100}) {
        std::this_thread::sleep_for(25ms);
        if (stopping_.load() || cancelled->load()) {
            transition(jobId, "cancelled", std::nullopt, "Fake worker cancelled");
            return;
        }
        transition(
            jobId,
            progress == 100 ? "succeeded" : "running",
            Json{{"completed", progress}, {"total", 100}, {"unit", "percent"}, {"message", progress == 100 ? "Fake growth complete" : "Growing fake surface"}},
            "Fake progress " + std::to_string(progress) + "%");
    }
}

}  // namespace vc::mcp
