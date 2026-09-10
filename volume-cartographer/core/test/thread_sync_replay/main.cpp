#include "thread_sync_replay/Replay.hpp"
#include "thread_sync_replay/RenderValgrindCli.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace replay = vc::thread_sync_replay;
using json = nlohmann::json;

namespace
{

constexpr int kProtocolVersion = 2;

struct LoadedGraph {
    explicit LoadedGraph(replay::Graph graph_) : graph(std::move(graph_)) {}

    replay::Graph graph;
    std::unordered_map<std::string, std::vector<double>> attributions;
};

std::vector<replay::Event> readEvents(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open event stream " + path);
    }
    std::vector<replay::Event> events;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const auto value = json::parse(line);
        const auto sequence = value.at("sequence").get<std::size_t>();
        if (sequence != events.size()) {
            throw std::runtime_error("event stream sequence is not contiguous");
        }
        replay::Event event{
            .thread = value.at("thread").get<std::int64_t>(),
            .kind = value.at("kind").get<std::string>(),
            .blocked = value.value("detail", json::object()).value("blocked", false),
        };
        for (const auto& dependency : value.value("dependencies", json::array())) {
            event.dependencies.push_back({
                .predecessor = dependency.at("sequence").get<std::size_t>(),
                .kind = dependency.at("kind").get<std::string>(),
            });
        }
        events.push_back(std::move(event));
    }
    return events;
}

json resultJson(const replay::ReplayResult& result)
{
    return {
        {"modeled_work", result.modeled_work},
        {"modeled_makespan", result.modeled_makespan},
        {"simulated_core_idle", result.simulated_core_idle},
        {"logical_sync_delay", result.logical_sync_delay},
        {"utilization", result.utilization},
        {"raw_replay_makespan", result.raw_replay_makespan},
        {"dependency_critical_path", result.dependency_critical_path},
        {"hard_dependency_critical_path", result.hard_dependency_critical_path},
        {"work_per_core_lower_bound", result.work_per_core_lower_bound},
        {"hard_schedule_lower_bound", result.hard_schedule_lower_bound},
        {"schedule_lower_bound", result.schedule_lower_bound},
        {"dependency_excess", result.dependency_excess},
        {"dependency_excess_scale", result.dependency_excess_scale},
        {"raw_replay_excess", result.raw_replay_excess},
        {"replay_idle_scale", result.replay_idle_scale},
        {"raw_simulated_core_idle", result.raw_simulated_core_idle},
    };
}

replay::ReplayOptions replayOptions(const json& value)
{
    return {
        .cores = value.at("cores").get<std::size_t>(),
        .tie_policy = value.at("tie_policy").get<std::string>(),
        .wake_latency = value.value("wake_latency", 0.0),
        .cross_thread_latency = value.value("cross_thread_latency", 0.0),
        .replay_idle_scale = value.value("replay_idle_scale", 1.0),
        .dependency_excess_scale = value.value("dependency_excess_scale", 1.0),
    };
}

replay::EventCostModel eventCostModel(const json& value)
{
    return {
        .feature_names = value.at("feature_names").get<std::vector<std::string>>(),
        .coefficients_ns = value.at("coefficients_ns").get<std::vector<double>>(),
        .stall_overlap_fraction = value.value("stall_overlap_fraction", 0.0),
    };
}

std::map<std::int64_t, replay::EventProfile> eventProfiles(const json& value)
{
    if (!value.is_object() || value.empty()) {
        throw std::runtime_error("event profiles must be a nonempty object");
    }
    std::map<std::int64_t, replay::EventProfile> profiles;
    for (auto thread = value.begin(); thread != value.end(); ++thread) {
        std::size_t consumed = 0;
        std::int64_t thread_id = 0;
        try {
            thread_id = std::stoll(thread.key(), &consumed);
        } catch (const std::exception&) {
            throw std::runtime_error("event profile has a malformed thread ID");
        }
        if (consumed != thread.key().size() || thread_id <= 0) {
            throw std::runtime_error("event profile has a malformed thread ID");
        }
        if (!thread.value().is_object()) {
            throw std::runtime_error("event profile is not an object");
        }
        replay::EventProfile events;
        for (auto event = thread.value().begin(); event != thread.value().end(); ++event) {
            if (!event.value().is_number_integer() && !event.value().is_number_unsigned()) {
                throw std::runtime_error("event profile counts must be nonnegative integers");
            }
            std::int64_t count = 0;
            if (event.value().is_number_unsigned()) {
                const auto unsigned_count = event.value().get<std::uint64_t>();
                if (unsigned_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                    throw std::runtime_error("event profile count is too large");
                }
                count = static_cast<std::int64_t>(unsigned_count);
            } else {
                count = event.value().get<std::int64_t>();
                if (count < 0) {
                    throw std::runtime_error("event profile counts must be nonnegative integers");
                }
            }
            events.emplace(event.key(), count);
        }
        if (!profiles.emplace(thread_id, std::move(events)).second) {
            throw std::runtime_error("event profile has duplicate numeric thread IDs");
        }
    }
    return profiles;
}

class Server
{
public:
    json handle(const json& request)
    {
        if (request.at("schema_version").get<int>() != kProtocolVersion) {
            throw std::runtime_error("unsupported native replay protocol version");
        }
        const auto command = request.at("command").get<std::string>();
        if (command == "load_graph") {
            return loadGraph(request);
        }
        if (command == "info") {
            return {
                {"compiler", VC_REPLAY_COMPILER_ID},
                {"compiler_version", VC_REPLAY_COMPILER_VERSION},
                {"build_type", VC_REPLAY_BUILD_TYPE},
                {"architecture", VC_REPLAY_ARCHITECTURE},
                {"profile_scoring", true},
            };
        }
        if (command == "model_profile_costs") {
            return modelProfileCosts(request);
        }
        if (command == "register_attributions") {
            return registerAttributions(request);
        }
        if (command == "replay_batch") {
            return replayBatch(request);
        }
        if (command == "stop") {
            _stop = true;
            return json::object();
        }
        throw std::runtime_error("unknown native replay command " + command);
    }

    [[nodiscard]] bool stopped() const noexcept { return _stop; }

private:
    json modelProfileCosts(const json& request)
    {
        const auto start = std::chrono::steady_clock::now();
        const auto costs = replay::modeledThreadCostsNs(eventProfiles(request.at("profiles")), eventCostModel(request.at("event_cost_model")));
        json encoded = json::object();
        double total = 0.0;
        for (const auto& [thread, cost] : costs) {
            encoded[std::to_string(thread)] = cost;
            total += cost;
            if (!std::isfinite(total)) {
                throw std::runtime_error("modeled thread-cost total is not finite");
            }
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return {
            {"thread_costs", std::move(encoded)},
            {"total_cost", total},
            {"native_profile_scoring_seconds", seconds},
        };
    }

    LoadedGraph& graph(const json& request)
    {
        const auto id = request.at("graph_id").get<std::string>();
        const auto found = _graphs.find(id);
        if (found == _graphs.end()) {
            throw std::runtime_error("unknown graph ID " + id);
        }
        return *found->second;
    }

    json loadGraph(const json& request)
    {
        const auto start = std::chrono::steady_clock::now();
        const auto id = request.at("graph_id").get<std::string>();
        if (id.empty() || _graphs.contains(id)) {
            throw std::runtime_error("graph ID is empty or already loaded");
        }
        auto loaded = std::make_unique<LoadedGraph>(replay::Graph(readEvents(request.at("event_path").get<std::string>())));
        const auto count = loaded->graph.size();
        _graphs.emplace(id, std::move(loaded));
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return {{"event_count", count}, {"native_load_seconds", seconds}};
    }

    json registerAttributions(const json& request)
    {
        const auto start = std::chrono::steady_clock::now();
        auto& loaded = graph(request);
        json registered = json::array();
        for (const auto& item : request.at("attributions")) {
            const auto id = item.at("attribution_id").get<std::string>();
            if (id.empty() || loaded.attributions.contains(id)) {
                throw std::runtime_error("attribution ID is empty or already registered");
            }
            std::map<std::int64_t, double> costs;
            for (auto entry = item.at("thread_costs").begin(); entry != item.at("thread_costs").end(); ++entry) {
                costs.emplace(std::stoll(entry.key()), entry.value().get<double>());
            }
            auto durations =
                loaded.graph.assignCosts(costs, item.at("residual_fraction").get<double>(), item.at("split_policy").get<std::string>());
            loaded.attributions.emplace(id, std::move(durations));
            registered.push_back(id);
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return {
            {"registered", std::move(registered)},
            {"native_attribution_seconds", seconds},
        };
    }

    json replayBatch(const json& request)
    {
        const auto start = std::chrono::steady_clock::now();
        auto& loaded = graph(request);
        json results = json::array();
        for (const auto& job : request.at("jobs")) {
            const auto attribution_id = job.at("attribution_id").get<std::string>();
            const auto attribution = loaded.attributions.find(attribution_id);
            if (attribution == loaded.attributions.end()) {
                throw std::runtime_error("unknown attribution ID " + attribution_id);
            }
            results.push_back({
                {"job_id", job.at("job_id").get<std::string>()},
                {"result", resultJson(loaded.graph.replayAdjusted(attribution->second, replayOptions(job)))},
            });
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return {
            {"results", std::move(results)},
            {"native_replay_seconds", seconds},
        };
    }

    std::unordered_map<std::string, std::unique_ptr<LoadedGraph>> _graphs;
    bool _stop{false};
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc >= 2 && std::string(argv[1]) == "evaluate-render") {
        try {
            return replay::renderValgrindCli(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "bench_thread_sync_replay: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 2 || std::string(argv[1]) != "--server") {
        std::cerr << "usage: bench_thread_sync_replay --server | evaluate-render ...\n";
        return 2;
    }

    Server server;
    std::string line;
    while (std::getline(std::cin, line)) {
        json request_id = nullptr;
        try {
            const auto request = json::parse(line);
            request_id = request.value("request_id", json(nullptr));
            auto response = server.handle(request);
            response["schema_version"] = kProtocolVersion;
            response["request_id"] = request_id;
            response["status"] = "ok";
            std::cout << response.dump() << '\n' << std::flush;
        } catch (const std::exception& error) {
            std::cout << json {
                {"schema_version", kProtocolVersion},
                {"request_id", request_id},
                {"status", "error"},
                {"error", error.what()},
            }.dump() << '\n' << std::flush;
        }
        if (server.stopped()) {
            return 0;
        }
    }
    return std::cin.eof() ? 0 : 1;
}
