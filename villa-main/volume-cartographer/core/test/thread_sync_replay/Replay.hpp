#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace vc::thread_sync_replay
{

struct Dependency {
    std::size_t predecessor;
    std::string kind;
};

struct Event {
    std::int64_t thread;
    std::string kind;
    bool blocked{false};
    std::vector<Dependency> dependencies;
};

struct ReplayOptions {
    std::size_t cores{1};
    std::string tie_policy{"fifo"};
    double wake_latency{0.0};
    double cross_thread_latency{0.0};
    double replay_idle_scale{1.0};
    double dependency_excess_scale{1.0};
};

struct ReplayResult {
    double modeled_work{0.0};
    double modeled_makespan{0.0};
    double simulated_core_idle{0.0};
    double logical_sync_delay{0.0};
    double utilization{0.0};
    double raw_replay_makespan{0.0};
    double dependency_critical_path{0.0};
    double hard_dependency_critical_path{0.0};
    double work_per_core_lower_bound{0.0};
    double hard_schedule_lower_bound{0.0};
    double schedule_lower_bound{0.0};
    double dependency_excess{0.0};
    double raw_replay_excess{0.0};
    double raw_simulated_core_idle{0.0};
    double replay_idle_scale{1.0};
    double dependency_excess_scale{1.0};
};

using EventProfile = std::map<std::string, std::int64_t>;

struct EventCostModel {
    std::vector<std::string> feature_names;
    std::vector<double> coefficients_ns;
    double stall_overlap_fraction{0.0};
};

[[nodiscard]] double modeledProfileCostNs(const EventProfile& events, const EventCostModel& model);

[[nodiscard]] std::map<std::int64_t, double> modeledThreadCostsNs(const std::map<std::int64_t, EventProfile>& profiles, const EventCostModel& model);

class Graph
{
public:
    explicit Graph(std::vector<Event> events);

    [[nodiscard]] std::size_t size() const noexcept { return _events.size(); }
    [[nodiscard]] const std::vector<Event>& events() const noexcept { return _events; }

    [[nodiscard]] std::vector<double> assignCosts(const std::map<std::int64_t, double>& thread_costs, double residual_fraction, const std::string& split_policy) const;

    [[nodiscard]] ReplayResult replayAdjusted(const std::vector<double>& durations, const ReplayOptions& options) const;

private:
    struct BasicResult {
        double work;
        double makespan;
        double idle;
        double sync_delay;
    };

    [[nodiscard]] BasicResult replay(const std::vector<double>& durations, const ReplayOptions& options) const;
    [[nodiscard]] double criticalPath(const std::vector<double>& durations, const ReplayOptions& options, bool exclude_drd) const;

    std::vector<Event> _events;
    std::vector<std::vector<std::size_t>> _successors;
    std::int64_t _max_thread{0};
};

}  // namespace vc::thread_sync_replay
