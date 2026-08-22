#include "thread_sync_replay/RenderValgrind.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace vc::thread_sync_replay
{
namespace
{

constexpr std::int64_t kMainThread = 1;

struct ParsedProfile {
    std::int64_t thread{0};
    EventProfile events;
};

struct MeasuredLog {
    std::vector<std::string> lines;
    std::size_t begin{0};
    std::size_t end{0};
};

std::vector<std::string> splitWords(const std::string& value)
{
    std::istringstream stream(value);
    std::vector<std::string> result;
    for (std::string word; stream >> word;) {
        result.push_back(std::move(word));
    }
    return result;
}

std::string regexEscape(const std::string& value)
{
    static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
    return std::regex_replace(value, special, R"(\$&)");
}

MeasuredLog readMeasuredLog(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open Valgrind trace " + path.string());
    }
    MeasuredLog result;
    for (std::string line; std::getline(stream, line);) {
        result.lines.push_back(std::move(line));
    }

    const std::regex clock_re(R"(SYSCALL\[\d+,(\d+)\]\(228\) sys_clock_gettime\( 1, (0x[0-9a-fA-F]+) \))");
    std::map<std::pair<std::int64_t, std::string>, std::vector<std::size_t>> clocks;
    for (std::size_t index = 0; index < result.lines.size(); ++index) {
        std::smatch match;
        if (std::regex_search(result.lines[index], match, clock_re)) {
            clocks[{std::stoll(match[1].str()), match[2].str()}].push_back(index);
        }
    }
    std::vector<std::pair<std::size_t, std::size_t>> candidates;
    for (const auto& [key, occurrences] : clocks) {
        if (key.first == kMainThread && occurrences.size() == 2) {
            candidates.emplace_back(occurrences[0], occurrences[1]);
        }
    }
    if (candidates.size() != 1) {
        throw std::runtime_error("Valgrind trace does not contain one unambiguous measured clock pair");
    }
    std::tie(result.begin, result.end) = candidates.front();
    if (result.begin + 1 >= result.end) {
        throw std::runtime_error("Valgrind measured clock window is empty");
    }
    return result;
}

SchedulerTrace schedulerFromMeasuredLog(const MeasuredLog& measured)
{
    const std::regex quantum_re(R"(SCHED\[(\d+)\]:\s+releasing lock \(VG_\(scheduler\):timeslice\))");
    SchedulerTrace result;
    result.begin_line = measured.begin + 1;
    result.end_line = measured.end + 1;
    for (std::size_t index = measured.begin + 1; index < measured.end; ++index) {
        std::smatch match;
        if (!std::regex_search(measured.lines[index], match, quantum_re)) {
            continue;
        }
        const auto thread = std::stoll(match[1].str());
        result.quantum_threads.push_back(thread);
        ++result.full_quanta[thread];
    }
    if (result.quantum_threads.empty()) {
        throw std::runtime_error("Valgrind measured window contains no scheduler quanta");
    }
    return result;
}

std::int64_t checkedAdd(std::int64_t left, std::int64_t right, const char* name)
{
    if (right < 0 || left > std::numeric_limits<std::int64_t>::max() - right) {
        throw std::runtime_error(std::string(name) + " counter overflow");
    }
    return left + right;
}

ParsedProfile parseProfile(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open Callgrind profile " + path.string());
    }
    std::optional<std::int64_t> thread;
    std::vector<std::string> names;
    std::vector<std::int64_t> totals;
    for (std::string line; std::getline(stream, line);) {
        if (line.starts_with("thread:")) {
            thread = std::stoll(line.substr(7));
        } else if (line.starts_with("events:")) {
            names = splitWords(line.substr(7));
        } else if (line.starts_with("totals:")) {
            totals.clear();
            for (const auto& word : splitWords(line.substr(7))) {
                const auto value = std::stoll(word);
                if (value < 0) {
                    throw std::runtime_error("negative Callgrind total in " + path.string());
                }
                totals.push_back(value);
            }
        }
    }
    if (!thread || *thread <= 0 || names.empty() || totals.size() > names.size()) {
        throw std::runtime_error("malformed Callgrind profile " + path.string());
    }
    totals.resize(names.size(), 0);
    EventProfile events;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (!events.emplace(names[index], totals[index]).second) {
            throw std::runtime_error("duplicate Callgrind event in " + path.string());
        }
    }
    return {*thread, std::move(events)};
}

EventProfile addProfiles(const EventProfile& left, const EventProfile& right)
{
    EventProfile result = left;
    for (const auto& [name, value] : right) {
        auto found = result.find(name);
        if (found == result.end()) {
            if (!left.empty()) {
                throw std::runtime_error("Callgrind slices use different event sets");
            }
            result.emplace(name, value);
        } else {
            found->second = checkedAdd(found->second, value, "Callgrind");
        }
    }
    if (!left.empty() && result.size() != right.size()) {
        throw std::runtime_error("Callgrind slices use different event sets");
    }
    return result;
}

bool profileIsZero(const EventProfile& profile)
{
    return std::all_of(profile.begin(), profile.end(), [](const auto& event) { return event.second == 0; });
}

std::int64_t profileValue(const EventProfile& profile, const std::string& name)
{
    const auto found = profile.find(name);
    if (found == profile.end()) {
        throw std::runtime_error("Callgrind profile is missing " + name);
    }
    return found->second;
}

std::vector<double> distributeSlices(const std::vector<EventProfile>& slices, const EventProfile& total, const EventCostModel& model, const std::vector<double>& weights)
{
    if (weights.empty()) {
        if (modeledProfileCostNs(total, model) != 0.0) {
            throw std::runtime_error("positive profile cost has no attribution window");
        }
        return {};
    }
    double total_weight = 0.0;
    for (const auto weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::runtime_error("invalid attribution-window weight");
        }
        total_weight += weight;
    }
    if (total_weight <= 0.0) {
        throw std::runtime_error("attribution windows have no positive weight");
    }

    const auto total_ir = profileValue(total, "Ir");
    const double target_cost = modeledProfileCostNs(total, model);
    if (total_ir <= 0) {
        if (target_cost != 0.0) {
            throw std::runtime_error("positive modeled cost has no instruction progress");
        }
        return std::vector<double>(weights.size(), 0.0);
    }

    std::vector<double> boundaries(weights.size() + 1, 0.0);
    for (std::size_t index = 0; index < weights.size(); ++index) {
        boundaries[index + 1] = boundaries[index] + static_cast<double>(total_ir) * weights[index] / total_weight;
    }
    boundaries.back() = static_cast<double>(total_ir);

    std::vector<double> result(weights.size(), 0.0);
    double progress = 0.0;
    double localized_total = 0.0;
    for (const auto& slice : slices) {
        const auto ir = profileValue(slice, "Ir");
        if (ir == 0) {
            continue;
        }
        const double begin = progress;
        const double end = progress + static_cast<double>(ir);
        const double cost = modeledProfileCostNs(slice, model);
        localized_total += cost;
        auto window = static_cast<std::size_t>(std::upper_bound(boundaries.begin(), boundaries.end(), begin) - boundaries.begin() - 1);
        window = std::min(window, weights.size() - 1);
        while (window < weights.size() && boundaries[window] < end) {
            const double overlap = std::max(0.0, std::min(end, boundaries[window + 1]) - std::max(begin, boundaries[window]));
            result[window] += cost * overlap / static_cast<double>(ir);
            ++window;
        }
        progress = end;
    }
    if (std::abs(progress - static_cast<double>(total_ir)) > 0.5) {
        throw std::runtime_error("Callgrind slice instructions do not equal the aggregate profile");
    }
    if (localized_total <= 0.0) {
        throw std::runtime_error("positive aggregate profile has no localized cost");
    }
    const double scale = target_cost / localized_total;
    for (auto& value : result) {
        value *= scale;
    }
    const double assigned = std::accumulate(result.begin(), result.end(), 0.0);
    result.back() += target_cost - assigned;
    return result;
}

std::map<std::int64_t, std::vector<double>> directWindowCosts(const CallgrindTrace& callgrind, const ThreadAttributionWindows& windows, const EventCostModel& model)
{
    std::map<std::int64_t, std::vector<double>> result;
    for (const auto& [thread, thread_windows] : windows) {
        const auto found_slices = callgrind.slices.find(thread);
        const auto found_total = callgrind.totals.find(thread);
        if (found_slices == callgrind.slices.end() || found_total == callgrind.totals.end()) {
            result.emplace(thread, std::vector<double>(thread_windows.size(), 0.0));
            continue;
        }
        std::vector<double> weights;
        weights.reserve(thread_windows.size());
        for (const auto& window : thread_windows) {
            weights.push_back(window.units);
        }
        result.emplace(thread, distributeSlices(found_slices->second, found_total->second, model, weights));
    }
    for (const auto& [thread, total] : callgrind.totals) {
        (void)total;
        if (!windows.contains(thread)) {
            throw std::runtime_error("Callgrind thread " + std::to_string(thread) + " has measured cost but no scheduler event");
        }
    }
    return result;
}

}  // namespace

CallgrindTrace parsePeriodicCallgrind(const std::filesystem::path& prefix)
{
    const auto directory = prefix.parent_path().empty() ? std::filesystem::path(".") : prefix.parent_path();
    const auto base = regexEscape(prefix.filename().string());
    const std::regex periodic("^" + base + R"(\.([0-9]+)-([0-9]+)$)");
    const std::regex residual("^" + base + R"(-([0-9]+)$)");
    std::map<std::int64_t, std::vector<std::pair<std::size_t, std::filesystem::path>>> files;
    std::map<std::int64_t, std::filesystem::path> residuals;
    std::size_t maximum_ordinal = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::smatch match;
        const auto name = entry.path().filename().string();
        if (std::regex_match(name, match, periodic)) {
            const auto ordinal = static_cast<std::size_t>(std::stoull(match[1].str()));
            const auto thread = std::stoll(match[2].str());
            files[thread].emplace_back(ordinal, entry.path());
            maximum_ordinal = std::max(maximum_ordinal, ordinal);
        } else if (std::regex_match(name, match, residual)) {
            residuals.emplace(std::stoll(match[1].str()), entry.path());
        }
    }
    if (files.empty() || maximum_ordinal == 0) {
        throw std::runtime_error("no periodic Callgrind profiles found at " + prefix.string());
    }

    CallgrindTrace result;
    result.periodic_dump_count = maximum_ordinal;
    for (auto& [thread, thread_files] : files) {
        std::sort(thread_files.begin(), thread_files.end());
        if (thread_files.size() != maximum_ordinal) {
            throw std::runtime_error("Callgrind thread " + std::to_string(thread) + " has missing periodic dumps");
        }
        EventProfile total;
        std::vector<EventProfile> slices;
        slices.reserve(thread_files.size() + 1);
        for (std::size_t index = 0; index < thread_files.size(); ++index) {
            if (thread_files[index].first != index + 1) {
                throw std::runtime_error("Callgrind periodic dump ordinals are not contiguous");
            }
            auto parsed = parseProfile(thread_files[index].second);
            if (parsed.thread != thread) {
                throw std::runtime_error("Callgrind filename and profile thread disagree");
            }
            total = addProfiles(total, parsed.events);
            slices.push_back(std::move(parsed.events));
        }
        const auto residual_file = residuals.find(thread);
        if (residual_file == residuals.end()) {
            throw std::runtime_error("Callgrind thread " + std::to_string(thread) + " has no residual profile");
        }
        auto parsed_residual = parseProfile(residual_file->second);
        total = addProfiles(total, parsed_residual.events);
        slices.push_back(std::move(parsed_residual.events));
        if (profileIsZero(total)) {
            continue;
        }
        result.slices.emplace(thread, std::move(slices));
        result.totals.emplace(thread, std::move(total));
    }
    for (const auto& [thread, path] : residuals) {
        if (files.contains(thread)) {
            continue;
        }
        const auto residual = parseProfile(path);
        if (residual.thread != thread || !profileIsZero(residual.events)) {
            throw std::runtime_error("Callgrind residual-only thread contains measured work");
        }
    }
    return result;
}

CallgrindSyncTrace parseMeasuredCallgrindSync(const std::filesystem::path& path)
{
    const auto measured = readMeasuredLog(path);
    const auto& lines = measured.lines;
    const auto begin = measured.begin;
    const auto end = measured.end;

    struct FutexCall {
        std::int64_t thread{0};
        std::string address;
        std::uint64_t operation{0};
        std::size_t start_line{0};
        std::size_t completion_line{0};
        bool complete{false};
        bool success{false};
        std::uint64_t result{0};
    };
    const std::regex start_re(R"(SYSCALL\[\d+,(\d+)\]\(202\) sys_futex \(\s*(0x[0-9a-fA-F]+),\s*([0-9]+),\s*([0-9]+).+\)\s*--> )");
    const std::regex completion_re(R"(SYSCALL\[\d+,(\d+)\]\(202\) \.\.\. \[async\] --> (Success|Failure)\((0x[0-9a-fA-F]+)\))");
    const std::regex result_re(R"(--> (Success|Failure)\((0x[0-9a-fA-F]+)\))");
    const std::regex quantum_re(R"(SCHED\[(\d+)\]:\s+releasing lock \(VG_\(scheduler\):timeslice\))");
    std::vector<FutexCall> calls;
    std::map<std::int64_t, std::size_t> pending;
    for (std::size_t index = 0; index < lines.size(); ++index) {
        std::smatch match;
        if (std::regex_search(lines[index], match, start_re)) {
            FutexCall call{
                .thread = std::stoll(match[1].str()),
                .address = match[2].str(),
                .operation = std::stoull(match[3].str()),
                .start_line = index,
                .completion_line = index,
            };
            std::smatch immediate;
            if (std::regex_search(lines[index], immediate, result_re)) {
                call.complete = true;
                call.success = immediate[1].str() == "Success";
                call.result = std::stoull(immediate[2].str(), nullptr, 0);
            }
            const auto call_index = calls.size();
            calls.push_back(std::move(call));
            if (!calls.back().complete && !pending.emplace(calls.back().thread, call_index).second) {
                throw std::runtime_error("Callgrind scheduler trace has overlapping futex syscalls for one thread");
            }
            continue;
        }
        if (!std::regex_search(lines[index], match, completion_re)) {
            continue;
        }
        const auto thread = std::stoll(match[1].str());
        const auto found = pending.find(thread);
        if (found == pending.end()) {
            continue;
        }
        auto& call = calls[found->second];
        call.complete = true;
        call.completion_line = index;
        call.success = match[2].str() == "Success";
        call.result = std::stoull(match[3].str(), nullptr, 0);
        pending.erase(found);
    }

    const auto command = [](const FutexCall& call) { return call.operation & 0x7fU; };
    const auto is_wait = [&](const FutexCall& call) { return command(call) == 0U || command(call) == 9U; };
    const auto is_wake = [&](const FutexCall& call) { return command(call) == 1U || command(call) == 10U; };
    const auto wait_blocks = [](const FutexCall& call) {
        constexpr std::uint64_t eagain = 11;
        return !call.complete || call.success || call.result != eagain;
    };
    const auto in_window = [&](std::size_t line) { return line > begin && line < end; };

    std::vector<std::uint64_t> wake_capacity(calls.size(), 0);
    for (std::size_t index = 0; index < calls.size(); ++index) {
        const auto& call = calls[index];
        if (in_window(call.start_line) && !is_wait(call) && !is_wake(call)) {
            throw std::runtime_error("Callgrind measured window contains unsupported futex operation " + std::to_string(command(call)));
        }
        if (is_wake(call) && call.complete && call.success) {
            wake_capacity[index] = call.result;
        }
    }

    std::vector<std::size_t> successful_waits;
    for (std::size_t index = 0; index < calls.size(); ++index) {
        if (is_wait(calls[index]) && calls[index].complete && calls[index].success) {
            successful_waits.push_back(index);
        }
    }
    std::sort(successful_waits.begin(), successful_waits.end(), [&](std::size_t left, std::size_t right) {
        return calls[left].completion_line < calls[right].completion_line;
    });

    std::map<std::size_t, std::size_t> wake_by_wait;
    std::size_t dropped_pre_window_edges = 0;
    for (const auto wait_index : successful_waits) {
        const auto& wait = calls[wait_index];
        std::optional<std::size_t> selected;
        for (std::size_t wake_index = 0; wake_index < calls.size(); ++wake_index) {
            const auto& wake = calls[wake_index];
            if (!is_wake(wake) || wake.address != wait.address || wake_capacity[wake_index] == 0 || wake.start_line < wait.start_line ||
                wake.start_line > wait.completion_line) {
                continue;
            }
            selected = wake_index;
            break;
        }
        if (!selected) {
            if (in_window(wait.completion_line) && wait.start_line <= begin) {
                ++dropped_pre_window_edges;
                continue;
            }
            if (in_window(wait.completion_line)) {
                throw std::runtime_error("Callgrind measured futex wait has no matching wake");
            }
            continue;
        }
        --wake_capacity[*selected];
        if (in_window(wait.completion_line)) {
            wake_by_wait.emplace(wait_index, *selected);
        }
    }

    enum class ActionKind {
        FutexStart,
        FutexResume,
        WorkQuantum,
    };
    struct Action {
        std::size_t line{0};
        ActionKind kind{ActionKind::FutexStart};
        std::int64_t thread{0};
        std::size_t call{0};
    };
    std::vector<Action> actions;
    for (std::size_t index = 0; index < calls.size(); ++index) {
        const auto& call = calls[index];
        if (in_window(call.start_line) && (is_wait(call) || is_wake(call))) {
            actions.push_back({call.start_line, ActionKind::FutexStart, call.thread, index});
        }
        if (is_wait(call) && call.complete && wait_blocks(call) && in_window(call.completion_line)) {
            actions.push_back({call.completion_line, ActionKind::FutexResume, call.thread, index});
        }
    }
    for (std::size_t index = begin + 1; index < end; ++index) {
        std::smatch match;
        if (std::regex_search(lines[index], match, quantum_re)) {
            actions.push_back({index, ActionKind::WorkQuantum, std::stoll(match[1].str()), 0});
        }
    }
    std::stable_sort(actions.begin(), actions.end(), [](const Action& left, const Action& right) {
        return std::tie(left.line, left.kind) < std::tie(right.line, right.kind);
    });

    CallgrindSyncTrace result;
    result.scheduler = schedulerFromMeasuredLog(measured);
    result.dropped_pre_window_edges = dropped_pre_window_edges;
    result.begin_line = begin + 1;
    result.end_line = end + 1;
    std::map<std::int64_t, std::size_t> previous;
    std::map<std::size_t, std::size_t> wake_event;
    const auto append = [&](std::int64_t thread, std::string kind, bool blocked = false) {
        Event event{.thread = thread, .kind = std::move(kind), .blocked = blocked};
        const auto found = previous.find(thread);
        if (found != previous.end()) {
            event.dependencies.push_back({found->second, "program_order"});
        }
        const auto sequence = result.events.size();
        result.events.push_back(std::move(event));
        previous[thread] = sequence;
        return sequence;
    };
    for (const auto& action : actions) {
        if (action.kind == ActionKind::WorkQuantum) {
            append(action.thread, "work_quantum");
            continue;
        }
        const auto& call = calls[action.call];
        if (action.kind == ActionKind::FutexStart) {
            ++result.parsed_futex_calls;
            if (is_wait(call)) {
                const bool blocked = wait_blocks(call);
                append(call.thread, "futex_wait", blocked);
                if (blocked) {
                    ++result.blocking_futex_waits;
                }
            } else {
                wake_event.emplace(action.call, append(call.thread, "futex_wake"));
            }
            continue;
        }

        const auto sequence = append(call.thread, "futex_resume");
        const auto matching_wake = wake_by_wait.find(action.call);
        if (matching_wake == wake_by_wait.end()) {
            continue;
        }
        const auto source = wake_event.find(matching_wake->second);
        if (source == wake_event.end()) {
            ++result.dropped_pre_window_edges;
            continue;
        }
        result.events[sequence].dependencies.push_back({source->second, "futex_wake"});
        ++result.futex_happens_before_edges;
    }
    if (result.events.empty()) {
        throw std::runtime_error("Callgrind measured window contains no replay events");
    }
    return result;
}

ReplayResult replayCallgrind(const CallgrindTrace& callgrind, const CallgrindSyncTrace& sync, const EventCostModel& model, const CallgrindReplayOptions& options)
{
    std::vector<Event> events = sync.events;
    std::map<std::int64_t, std::size_t> previous;
    for (std::size_t sequence = 0; sequence < events.size(); ++sequence) {
        previous[events[sequence].thread] = sequence;
    }

    Graph initial(events);
    const auto initial_windows = initial.attributionWindows(options.residual_fraction);
    for (const auto& [thread, total] : callgrind.totals) {
        (void)total;
        const auto found = initial_windows.find(thread);
        if (found != initial_windows.end() && !found->second.empty()) {
            continue;
        }
        Event residual{.thread = thread, .kind = "work_residual"};
        const auto predecessor = previous.find(thread);
        if (predecessor != previous.end()) {
            residual.dependencies.push_back({predecessor->second, "program_order"});
        }
        previous[thread] = events.size();
        events.push_back(std::move(residual));
    }

    Graph graph(std::move(events));
    const auto windows = graph.attributionWindows(options.residual_fraction);
    const auto costs = directWindowCosts(callgrind, windows, model);
    const auto durations = graph.assignWindowCosts(costs, options.residual_fraction, options.split_policy);
    return graph.replayAdjusted(durations, options.replay);
}

}  // namespace vc::thread_sync_replay
