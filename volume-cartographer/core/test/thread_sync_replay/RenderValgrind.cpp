#include "thread_sync_replay/RenderValgrind.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
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

std::vector<double> normalizedShape(const std::vector<EventProfile>& slices, const EventProfile& total, const EventCostModel& model, std::size_t bins)
{
    if (bins == 0) {
        throw std::runtime_error("equivalence shape must have at least one bin");
    }
    const auto shape = distributeSlices(slices, total, model, std::vector<double>(bins, 1.0));
    const double sum = std::accumulate(shape.begin(), shape.end(), 0.0);
    std::vector<double> normalized(shape.size(), 0.0);
    if (sum > 0.0) {
        std::transform(shape.begin(), shape.end(), normalized.begin(), [sum](double value) { return value / sum; });
    }
    return normalized;
}

struct SourceDescriptor {
    std::int64_t thread;
    std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t> signature;
    double cost;
    std::vector<double> shape;
};

struct TraceDescriptor {
    std::int64_t thread;
    std::pair<std::size_t, std::size_t> signature;
};

double relativeSpread(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    const double scale = std::max(1.0, std::accumulate(values.begin(), values.end(), 0.0) / values.size());
    return (*maximum - *minimum) / scale;
}

double shapeSpread(const std::vector<SourceDescriptor>& sources, std::size_t begin, std::size_t end)
{
    double result = 0.0;
    for (std::size_t left = begin; left < end; ++left) {
        for (std::size_t right = left + 1; right < end; ++right) {
            for (std::size_t bin = 0; bin < sources[left].shape.size(); ++bin) {
                result = std::max(result, std::abs(sources[left].shape[bin] - sources[right].shape[bin]));
            }
        }
    }
    return result;
}

std::map<std::int64_t, std::vector<double>> mappedWindowCosts(
    const std::map<std::int64_t, std::int64_t>& source_by_trace, const CallgrindTrace& callgrind, const ThreadAttributionWindows& windows, const EventCostModel& model)
{
    std::map<std::int64_t, std::vector<double>> result;
    for (const auto& [trace_thread, source_thread] : source_by_trace) {
        const auto found_windows = windows.find(trace_thread);
        const auto found_slices = callgrind.slices.find(source_thread);
        const auto found_total = callgrind.totals.find(source_thread);
        if (found_windows == windows.end() || found_slices == callgrind.slices.end() || found_total == callgrind.totals.end()) {
            throw std::runtime_error("paired attribution references an unknown thread");
        }
        std::vector<double> weights;
        weights.reserve(found_windows->second.size());
        for (const auto& window : found_windows->second) {
            weights.push_back(window.units);
        }
        result.emplace(trace_thread, distributeSlices(found_slices->second, found_total->second, model, weights));
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
        result.slices.emplace(thread, std::move(slices));
        result.totals.emplace(thread, std::move(total));
    }
    if (residuals.size() != files.size()) {
        throw std::runtime_error("Callgrind residual and periodic thread sets differ");
    }
    return result;
}

DrdTrace parseMeasuredDrd(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open DRD trace " + path.string());
    }
    std::vector<std::string> lines;
    for (std::string line; std::getline(stream, line);) {
        lines.push_back(std::move(line));
    }

    const std::regex clock_re(R"(SYSCALL\[\d+,(\d+)\]\(228\) sys_clock_gettime\( 1, (0x[0-9a-fA-F]+) \))");
    std::map<std::pair<std::int64_t, std::string>, std::vector<std::size_t>> clocks;
    for (std::size_t index = 0; index < lines.size(); ++index) {
        std::smatch match;
        if (std::regex_search(lines[index], match, clock_re)) {
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
        throw std::runtime_error("DRD trace does not contain one unambiguous measured clock pair");
    }
    const auto [begin, end] = candidates.front();
    if (begin + 1 >= end) {
        throw std::runtime_error("DRD measured clock window is empty");
    }

    const std::regex segment_re(R"(New segment for thread ([0-9]+) with vc \[ (.*) \])");
    const std::regex value_re(R"((\d+):\s*(\d+))");
    const std::regex quantum_re(R"(SCHED\[(\d+)\]:\s+releasing lock \(VG_\(scheduler\):timeslice\))");
    struct SegmentRecord {
        std::size_t line;
        std::int64_t thread;
        std::map<std::int64_t, std::int64_t> clock;
        std::optional<std::size_t> event;
    };
    std::vector<SegmentRecord> segments;
    std::map<std::pair<std::int64_t, std::int64_t>, std::size_t> segment_by_clock;
    for (std::size_t index = 0; index < end; ++index) {
        std::smatch match;
        if (!std::regex_search(lines[index], match, segment_re)) {
            continue;
        }
        SegmentRecord record{.line = index, .thread = std::stoll(match[1].str())};
        const auto clock_text = match[2].str();
        for (auto iterator = std::sregex_iterator(clock_text.begin(), clock_text.end(), value_re); iterator != std::sregex_iterator(); ++iterator) {
            record.clock.emplace(std::stoll((*iterator)[1].str()), std::stoll((*iterator)[2].str()));
        }
        const auto own = record.clock.find(record.thread);
        if (own == record.clock.end()) {
            throw std::runtime_error("DRD segment has no own vector-clock value");
        }
        const auto record_index = segments.size();
        if (!segment_by_clock.emplace(std::make_pair(record.thread, own->second), record_index).second) {
            throw std::runtime_error("duplicate DRD vector-clock segment");
        }
        segments.push_back(std::move(record));
    }

    DrdTrace result;
    result.parsed_segment_count = segments.size();
    result.begin_line = begin + 1;
    result.end_line = end + 1;
    std::map<std::int64_t, std::size_t> previous;
    std::map<std::int64_t, std::map<std::int64_t, std::int64_t>> previous_clock;
    auto append = [&](std::int64_t thread, std::string kind) -> std::size_t {
        Event event{.thread = thread, .kind = std::move(kind)};
        const auto found = previous.find(thread);
        if (found != previous.end()) {
            event.dependencies.push_back({found->second, "program_order"});
        }
        const auto sequence = result.events.size();
        result.events.push_back(std::move(event));
        previous[thread] = sequence;
        return sequence;
    };

    std::size_t segment_cursor = 0;
    for (std::size_t index = begin + 1; index < end; ++index) {
        while (segment_cursor < segments.size() && segments[segment_cursor].line < index) {
            ++segment_cursor;
        }
        if (segment_cursor < segments.size() && segments[segment_cursor].line == index) {
            auto& segment = segments[segment_cursor];
            const auto sequence = append(segment.thread, "hb_segment");
            ++result.retained_segment_count;
            segment.event = sequence;
            for (const auto& [owner, value] : segment.clock) {
                if (owner == segment.thread || value <= previous_clock[segment.thread][owner]) {
                    continue;
                }
                const auto predecessor = segment_by_clock.find({owner, value});
                if (predecessor == segment_by_clock.end() || segments[predecessor->second].line >= index) {
                    throw std::runtime_error("DRD trace has an unresolved measured-window dependency");
                }
                const auto predecessor_event = segments[predecessor->second].event;
                if (predecessor_event) {
                    result.events[sequence].dependencies.push_back({*predecessor_event, "drd_happens_before"});
                    ++result.happens_before_edges;
                } else {
                    ++result.dropped_pre_window_edges;
                }
            }
            previous_clock[segment.thread] = segment.clock;
            ++segment_cursor;
        }
        std::smatch quantum;
        if (std::regex_search(lines[index], quantum, quantum_re)) {
            const auto thread = std::stoll(quantum[1].str());
            append(thread, "work_quantum");
            ++result.full_quanta[thread];
        }
    }
    if (result.events.empty()) {
        throw std::runtime_error("DRD measured window contains no replay events");
    }
    return result;
}

PairedReplayResult replayPaired(const CallgrindTrace& callgrind, const DrdTrace& drd, const EventCostModel& model, const PairedReplayOptions& options)
{
    if (!std::isfinite(options.equivalent_cost_tolerance) || options.equivalent_cost_tolerance < 0.0 || options.equivalent_cost_tolerance >= 1.0 ||
        !std::isfinite(options.equivalent_shape_tolerance) || options.equivalent_shape_tolerance < 0.0 || options.equivalent_shape_tolerance >= 1.0) {
        throw std::runtime_error("equivalent-trace tolerances must be in [0, 1)");
    }
    if (options.maximum_mappings == 0) {
        throw std::runtime_error("maximum mapping count must be positive");
    }
    Graph graph(drd.events);
    const auto windows = graph.attributionWindows(options.residual_fraction);
    if (callgrind.totals.size() != windows.size() || !callgrind.totals.contains(kMainThread) || !windows.contains(kMainThread)) {
        throw std::runtime_error("Callgrind and DRD thread counts do not match");
    }

    std::vector<SourceDescriptor> sources;
    for (const auto& [thread, total] : callgrind.totals) {
        if (thread == kMainThread) {
            continue;
        }
        sources.push_back({
            .thread = thread,
            .signature =
                {
                    profileValue(total, "Ir"),
                    profileValue(total, "Dr"),
                    profileValue(total, "Dw"),
                    profileValue(total, "Bc"),
                    profileValue(total, "Bi"),
                    thread,
                },
            .cost = modeledProfileCostNs(total, model),
            .shape = normalizedShape(callgrind.slices.at(thread), total, model, options.equivalence_bins),
        });
    }
    std::sort(sources.begin(), sources.end(), [](const SourceDescriptor& left, const SourceDescriptor& right) {
        return left.signature < right.signature;
    });

    std::vector<TraceDescriptor> traces;
    for (const auto& [thread, thread_windows] : windows) {
        if (thread == kMainThread) {
            continue;
        }
        const auto quanta = drd.full_quanta.find(thread);
        traces.push_back({thread, {quanta == drd.full_quanta.end() ? 0 : quanta->second, thread_windows.size()}});
    }
    std::sort(traces.begin(), traces.end(), [](const TraceDescriptor& left, const TraceDescriptor& right) {
        return std::tie(left.signature, left.thread) < std::tie(right.signature, right.thread);
    });
    if (sources.size() != traces.size()) {
        throw std::runtime_error("Callgrind and DRD worker counts do not match");
    }

    struct Group {
        std::size_t begin;
        std::size_t end;
    };
    std::vector<Group> groups;
    PairedReplayResult result;
    for (std::size_t begin_index = 0; begin_index < traces.size();) {
        std::size_t end_index = begin_index + 1;
        while (end_index < traces.size() && traces[end_index].signature == traces[begin_index].signature) {
            ++end_index;
        }
        std::vector<double> costs;
        for (std::size_t index = begin_index; index < end_index; ++index) {
            costs.push_back(sources[index].cost);
        }
        const double cost_spread = relativeSpread(costs);
        const double trace_shape_spread = shapeSpread(sources, begin_index, end_index);
        result.maximum_equivalent_cost_spread = std::max(result.maximum_equivalent_cost_spread, cost_spread);
        result.maximum_equivalent_shape_spread = std::max(result.maximum_equivalent_shape_spread, trace_shape_spread);
        if (cost_spread > options.equivalent_cost_tolerance || trace_shape_spread > options.equivalent_shape_tolerance) {
            throw std::runtime_error(
                "tied DRD workers have non-equivalent Callgrind traces: ranks " + std::to_string(begin_index) + "-" + std::to_string(end_index - 1) +
                ", cost spread=" + std::to_string(cost_spread) + ", shape spread=" + std::to_string(trace_shape_spread));
        }
        groups.push_back({begin_index, end_index});
        begin_index = end_index;
    }

    std::map<std::int64_t, std::int64_t> mapping{{kMainThread, kMainThread}};
    result.minimum_makespan = std::numeric_limits<double>::infinity();
    double maximum_makespan = -1.0;
    std::function<void(std::size_t)> enumerate = [&](std::size_t group_index) {
        if (group_index == groups.size()) {
            if (++result.mapping_count > options.maximum_mappings) {
                throw std::runtime_error("equivalent worker mapping count exceeds limit");
            }
            const auto costs = mappedWindowCosts(mapping, callgrind, windows, model);
            const auto durations = graph.assignWindowCosts(costs, options.residual_fraction, options.split_policy);
            const auto replay = graph.replayAdjusted(durations, options.replay);
            result.minimum_makespan = std::min(result.minimum_makespan, replay.modeled_makespan);
            if (replay.modeled_makespan > maximum_makespan) {
                maximum_makespan = replay.modeled_makespan;
                result.conservative = replay;
                result.selected_source_by_trace_thread = mapping;
            }
            return;
        }
        const auto group = groups[group_index];
        std::vector<std::int64_t> permutation;
        for (std::size_t index = group.begin; index < group.end; ++index) {
            permutation.push_back(sources[index].thread);
        }
        std::sort(permutation.begin(), permutation.end());
        do {
            for (std::size_t index = group.begin; index < group.end; ++index) {
                mapping[traces[index].thread] = permutation[index - group.begin];
            }
            enumerate(group_index + 1);
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    };
    enumerate(0);
    if (result.mapping_count == 0) {
        throw std::runtime_error("no admissible paired worker mapping");
    }
    return result;
}

}  // namespace vc::thread_sync_replay
