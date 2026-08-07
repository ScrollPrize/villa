#include "thread_sync_replay/Replay.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <iterator>
#include <unordered_map>

namespace vc::thread_sync_replay
{
namespace
{

bool isCrossThreadDependency(const std::string& kind)
{
    return kind == "drd_happens_before" || kind == "futex_wake";
}

void requireFiniteNonnegative(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(std::string(name) + " must be finite and nonnegative");
    }
}

std::vector<Dependency> uniqueDependencies(const std::vector<Dependency>& dependencies, std::size_t event_count)
{
    std::vector<Dependency> unique;
    std::unordered_map<std::size_t, std::size_t> positions;
    for (const auto& dependency : dependencies) {
        if (dependency.predecessor >= event_count) {
            throw std::runtime_error("dependency references an event outside the graph");
        }
        const auto [position, inserted] = positions.emplace(dependency.predecessor, unique.size());
        if (inserted) {
            unique.push_back(dependency);
        } else {
            unique[position->second].kind = dependency.kind;
        }
    }
    return unique;
}

double edgeFinish(const std::vector<Event>& events, const std::vector<double>& finish, std::size_t successor, const Dependency& dependency, const ReplayOptions& options)
{
    double value = finish[dependency.predecessor];
    const auto& predecessor = events[dependency.predecessor];
    const auto& event = events[successor];
    if (isCrossThreadDependency(dependency.kind) && predecessor.thread != event.thread) {
        value += options.cross_thread_latency;
    }
    if (dependency.kind == "futex_wake" && predecessor.thread == 1 && event.thread != 1) {
        value += options.wake_latency;
    }
    return value;
}

}  // namespace

Graph::Graph(std::vector<Event> events) : _events(std::move(events))
{
    _successors.resize(_events.size());
    for (std::size_t sequence = 0; sequence < _events.size(); ++sequence) {
        auto& event = _events[sequence];
        if (event.thread <= 0) {
            throw std::runtime_error("event thread IDs must be positive");
        }
        event.dependencies = uniqueDependencies(event.dependencies, _events.size());
        _max_thread = std::max(_max_thread, event.thread);
        for (const auto& dependency : event.dependencies) {
            _successors[dependency.predecessor].push_back(sequence);
        }
    }
}

std::vector<double> Graph::assignCosts(const std::map<std::int64_t, double>& thread_costs, double residual_fraction, const std::string& split_policy) const
{
    if (!std::isfinite(residual_fraction) || residual_fraction < 0.0 || residual_fraction > 1.0) {
        throw std::runtime_error("residual fraction must be between zero and one");
    }
    if (split_policy != "equal" && split_policy != "front" && split_policy != "back") {
        throw std::runtime_error("unknown split policy " + split_policy);
    }

    std::map<std::int64_t, std::vector<std::size_t>> by_thread;
    for (std::size_t sequence = 0; sequence < _events.size(); ++sequence) {
        by_thread[_events[sequence].thread].push_back(sequence);
    }

    std::vector<double> durations(_events.size(), 0.0);
    for (const auto& [thread, sequences] : by_thread) {
        const auto cost = thread_costs.find(thread);
        if (cost == thread_costs.end()) {
            throw std::runtime_error("trace thread " + std::to_string(thread) + " has no cost");
        }
        requireFiniteNonnegative(cost->second, "thread cost");

        struct Window {
            std::vector<std::size_t> candidates;
            double units;
        };
        std::vector<Window> windows;
        std::vector<std::size_t> candidates;
        bool blocked = false;
        bool current = false;
        for (const auto sequence : sequences) {
            const auto& event = _events[sequence];
            current = true;
            if (event.kind == "futex_resume") {
                blocked = false;
            }
            if (event.kind != "thread_start" && event.kind != "futex_resume" && !blocked) {
                candidates.push_back(sequence);
            }
            if (event.kind == "futex_wait" && event.blocked) {
                blocked = true;
            }
            if (event.kind == "work_quantum") {
                windows.push_back({std::move(candidates), 1.0});
                candidates.clear();
                current = false;
            }
        }
        if (current || !candidates.empty()) {
            windows.push_back({std::move(candidates), residual_fraction});
        }

        std::vector<Window*> eligible;
        double total_units = 0.0;
        for (auto& window : windows) {
            if (!window.candidates.empty()) {
                eligible.push_back(&window);
                total_units += window.units;
            }
        }
        if (eligible.empty()) {
            if (cost->second != 0.0) {
                throw std::runtime_error("trace thread " + std::to_string(thread) + " has positive cost but no eligible event");
            }
            continue;
        }
        if (total_units <= 0.0) {
            if (eligible.size() != 1) {
                throw std::runtime_error("trace thread " + std::to_string(thread) + " has no positive attribution weight");
            }
            eligible.front()->units = 1.0;
            total_units = 1.0;
        }

        const double unit_cost = cost->second / total_units;
        for (const auto* window : eligible) {
            const double window_cost = unit_cost * window->units;
            if (split_policy == "equal") {
                const double share = window_cost / window->candidates.size();
                for (const auto sequence : window->candidates) {
                    durations[sequence] += share;
                }
            } else if (split_policy == "front") {
                durations[window->candidates.front()] += window_cost;
            } else {
                durations[window->candidates.back()] += window_cost;
            }
        }

        double assigned = 0.0;
        for (const auto sequence : sequences) {
            assigned += durations[sequence];
        }
        const double tolerance = 1e-12 * std::max(1.0, cost->second);
        if (std::abs(assigned - cost->second) > tolerance) {
            throw std::runtime_error("cost attribution did not preserve thread total");
        }
    }
    return durations;
}

Graph::BasicResult Graph::replay(const std::vector<double>& durations, const ReplayOptions& options) const
{
    if (durations.size() != _events.size()) {
        throw std::runtime_error("duration count does not match event count");
    }
    if (options.cores == 0) {
        throw std::runtime_error("simulated core count must be positive");
    }
    if (options.tie_policy != "fifo" && options.tie_policy != "round_robin") {
        throw std::runtime_error("unknown tie policy " + options.tie_policy);
    }
    requireFiniteNonnegative(options.wake_latency, "wake latency");
    requireFiniteNonnegative(options.cross_thread_latency, "cross-thread latency");
    for (const auto duration : durations) {
        requireFiniteNonnegative(duration, "event duration");
    }

    std::vector<std::size_t> remaining;
    remaining.reserve(_events.size());
    for (const auto& event : _events) {
        remaining.push_back(event.dependencies.size());
    }
    std::vector<double> dependency_finish(_events.size(), 0.0);
    std::vector<double> finish(_events.size(), 0.0);
    std::set<std::size_t> ready;
    using Released = std::pair<double, std::size_t>;
    std::priority_queue<Released, std::vector<Released>, std::greater<>> ready_by_release;
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>> eligible_fifo;
    for (std::size_t sequence = 0; sequence < remaining.size(); ++sequence) {
        if (remaining[sequence] == 0) {
            if (options.tie_policy == "fifo") {
                ready_by_release.emplace(0.0, sequence);
            } else {
                ready.insert(sequence);
            }
        }
    }

    std::vector<double> core_available(options.cores, 0.0);
    std::int64_t last_thread = 0;
    std::size_t scheduled = 0;
    double idle = 0.0;
    double sync_delay = 0.0;
    while (!ready.empty() || !ready_by_release.empty() || !eligible_fifo.empty()) {
        const auto core = static_cast<std::size_t>(std::min_element(core_available.begin(), core_available.end()) - core_available.begin());
        const double core_time = core_available[core];
        std::size_t sequence = 0;
        if (options.tie_policy == "fifo") {
            while (!ready_by_release.empty() && ready_by_release.top().first <= core_time) {
                eligible_fifo.push(ready_by_release.top().second);
                ready_by_release.pop();
            }
            if (!eligible_fifo.empty()) {
                sequence = eligible_fifo.top();
                eligible_fifo.pop();
            } else {
                sequence = ready_by_release.top().second;
                ready_by_release.pop();
            }
        } else {
            auto best = ready.begin();
            auto best_key =
                std::tuple{std::max(core_time, dependency_finish[*best]), (_events[*best].thread - last_thread - 1 + _max_thread) % _max_thread, *best};
            for (auto candidate = std::next(ready.begin()); candidate != ready.end(); ++candidate) {
                const auto key =
                    std::tuple{std::max(core_time, dependency_finish[*candidate]), (_events[*candidate].thread - last_thread - 1 + _max_thread) % _max_thread, *candidate};
                if (key < best_key) {
                    best = candidate;
                    best_key = key;
                }
            }
            sequence = *best;
            ready.erase(best);
        }

        const auto& event = _events[sequence];
        double release = dependency_finish[sequence];
        double program_finish = 0.0;
        for (const auto& dependency : event.dependencies) {
            release = std::max(release, edgeFinish(_events, finish, sequence, dependency, options));
            if (dependency.kind == "program_order") {
                program_finish = std::max(program_finish, finish[dependency.predecessor]);
            }
        }
        const double event_start = std::max(core_time, release);
        finish[sequence] = event_start + durations[sequence];
        idle += event_start - core_time;
        sync_delay += std::max(0.0, release - program_finish);
        core_available[core] = finish[sequence];
        last_thread = event.thread;
        ++scheduled;
        for (const auto successor : _successors[sequence]) {
            --remaining[successor];
            dependency_finish[successor] = std::max(dependency_finish[successor], finish[sequence]);
            if (remaining[successor] == 0) {
                if (options.tie_policy == "fifo") {
                    ready_by_release.emplace(dependency_finish[successor], successor);
                } else {
                    ready.insert(successor);
                }
            }
        }
    }
    if (scheduled != _events.size()) {
        throw std::runtime_error("dependency graph is cyclic");
    }
    return {
        .work = std::accumulate(durations.begin(), durations.end(), 0.0),
        .makespan = finish.empty() ? 0.0 : *std::max_element(finish.begin(), finish.end()),
        .idle = idle,
        .sync_delay = sync_delay,
    };
}

double Graph::criticalPath(const std::vector<double>& durations, const ReplayOptions& options, bool exclude_drd) const
{
    std::vector<std::size_t> remaining(_events.size(), 0);
    std::vector<std::vector<std::size_t>> successors(_events.size());
    for (std::size_t sequence = 0; sequence < _events.size(); ++sequence) {
        for (const auto& dependency : _events[sequence].dependencies) {
            if (exclude_drd && dependency.kind == "drd_happens_before") {
                continue;
            }
            ++remaining[sequence];
            successors[dependency.predecessor].push_back(sequence);
        }
    }
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>> ready;
    for (std::size_t sequence = 0; sequence < remaining.size(); ++sequence) {
        if (remaining[sequence] == 0) {
            ready.push(sequence);
        }
    }
    std::vector<double> finish(_events.size(), 0.0);
    std::size_t completed = 0;
    while (!ready.empty()) {
        const auto sequence = ready.top();
        ready.pop();
        double release = 0.0;
        for (const auto& dependency : _events[sequence].dependencies) {
            if (exclude_drd && dependency.kind == "drd_happens_before") {
                continue;
            }
            release = std::max(release, edgeFinish(_events, finish, sequence, dependency, options));
        }
        finish[sequence] = release + durations[sequence];
        ++completed;
        for (const auto successor : successors[sequence]) {
            --remaining[successor];
            if (remaining[successor] == 0) {
                ready.push(successor);
            }
        }
    }
    if (completed != _events.size()) {
        throw std::runtime_error("dependency graph is cyclic");
    }
    return finish.empty() ? 0.0 : *std::max_element(finish.begin(), finish.end());
}

ReplayResult Graph::replayAdjusted(const std::vector<double>& durations, const ReplayOptions& options) const
{
    if (!std::isfinite(options.replay_idle_scale) || options.replay_idle_scale < 0.0 || options.replay_idle_scale > 1.0) {
        throw std::runtime_error("replay idle scale must be between zero and one");
    }
    if (!std::isfinite(options.dependency_excess_scale) || options.dependency_excess_scale < 0.0 || options.dependency_excess_scale > 1.0) {
        throw std::runtime_error("dependency excess scale must be between zero and one");
    }

    const auto basic = replay(durations, options);
    const double critical = criticalPath(durations, options, false);
    const double hard_critical = criticalPath(durations, options, true);
    const double work_bound = basic.work / static_cast<double>(options.cores);
    const double hard_lower = std::max(hard_critical, work_bound);
    double lower = std::max(hard_lower, critical);
    const double tolerance = 1e-9 * std::max({1.0, basic.makespan, lower});
    if (basic.makespan + tolerance < lower) {
        throw std::runtime_error("replay makespan is below its scheduling lower bound");
    }
    lower = std::min(lower, basic.makespan);
    const double replay_excess = std::max(0.0, basic.makespan - lower);
    const double dependency_excess = std::max(0.0, lower - hard_lower);
    const double adjusted = hard_lower + options.dependency_excess_scale * dependency_excess + options.replay_idle_scale * replay_excess;
    return {
        .modeled_work = basic.work,
        .modeled_makespan = adjusted,
        .simulated_core_idle = std::max(0.0, static_cast<double>(options.cores) * adjusted - basic.work),
        .logical_sync_delay = basic.sync_delay,
        .utilization = adjusted == 0.0 ? 0.0 : basic.work / (static_cast<double>(options.cores) * adjusted),
        .raw_replay_makespan = basic.makespan,
        .dependency_critical_path = critical,
        .hard_dependency_critical_path = hard_critical,
        .work_per_core_lower_bound = work_bound,
        .hard_schedule_lower_bound = hard_lower,
        .schedule_lower_bound = lower,
        .dependency_excess = dependency_excess,
        .raw_replay_excess = replay_excess,
        .raw_simulated_core_idle = basic.idle,
        .replay_idle_scale = options.replay_idle_scale,
        .dependency_excess_scale = options.dependency_excess_scale,
    };
}

}  // namespace vc::thread_sync_replay
