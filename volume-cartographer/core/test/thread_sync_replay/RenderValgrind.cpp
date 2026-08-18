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

struct SchedulerDescriptor {
    double share{0.0};
    std::vector<double> cumulative_activity;
};

struct SchedulerDescriptors {
    std::map<std::int64_t, SchedulerDescriptor> threads;
    std::size_t total_quanta{0};
};

SchedulerDescriptors describeScheduler(const SchedulerTrace& scheduler, const std::vector<std::int64_t>& threads, std::size_t bins)
{
    if (threads.empty() || bins == 0) {
        throw std::runtime_error("scheduler attribution requires workers and activity bins");
    }
    const std::set<std::int64_t> expected(threads.begin(), threads.end());
    SchedulerDescriptors result;
    for (const auto thread : threads) {
        const auto found = scheduler.full_quanta.find(thread);
        if (found == scheduler.full_quanta.end() || found->second == 0) {
            throw std::runtime_error("scheduler trace has no measured work for worker " + std::to_string(thread));
        }
        result.threads.emplace(thread, SchedulerDescriptor{.cumulative_activity = std::vector<double>(bins, 0.0)});
    }
    for (const auto& [thread, quanta] : scheduler.full_quanta) {
        if (thread != kMainThread && quanta != 0 && !expected.contains(thread)) {
            throw std::runtime_error("scheduler trace contains an unmatched active worker " + std::to_string(thread));
        }
    }

    std::vector<std::int64_t> sequence;
    for (const auto thread : scheduler.quantum_threads) {
        if (expected.contains(thread)) {
            sequence.push_back(thread);
        }
    }
    result.total_quanta = sequence.size();
    if (result.total_quanta == 0) {
        throw std::runtime_error("scheduler trace contains no worker quanta");
    }

    std::map<std::int64_t, std::size_t> cumulative;
    std::size_t cursor = 0;
    for (std::size_t bin = 0; bin < bins; ++bin) {
        const std::size_t boundary = (bin + 1) * result.total_quanta / bins;
        while (cursor < boundary) {
            ++cumulative[sequence[cursor++]];
        }
        for (const auto thread : threads) {
            result.threads.at(thread).cumulative_activity[bin] = static_cast<double>(cumulative[thread]) / static_cast<double>(result.total_quanta);
        }
    }
    for (const auto thread : threads) {
        const auto counted = cumulative[thread];
        if (counted != scheduler.full_quanta.at(thread)) {
            throw std::runtime_error("scheduler trace quantum sequence and totals disagree");
        }
        result.threads.at(thread).share = static_cast<double>(counted) / static_cast<double>(result.total_quanta);
    }
    return result;
}

double schedulerDistance(const SchedulerDescriptor& source, const SchedulerDescriptor& target)
{
    if (source.cumulative_activity.size() != target.cumulative_activity.size() || source.cumulative_activity.empty()) {
        throw std::runtime_error("scheduler activity descriptors have incompatible shapes");
    }
    double cumulative_distance = 0.0;
    for (std::size_t index = 0; index < source.cumulative_activity.size(); ++index) {
        cumulative_distance += std::abs(source.cumulative_activity[index] - target.cumulative_activity[index]);
    }
    return std::abs(source.share - target.share) + cumulative_distance / source.cumulative_activity.size();
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

SchedulerTrace parseMeasuredScheduler(const std::filesystem::path& path)
{
    return schedulerFromMeasuredLog(readMeasuredLog(path));
}

DrdTrace parseMeasuredDrd(const std::filesystem::path& path)
{
    const auto measured = readMeasuredLog(path);
    const auto& lines = measured.lines;
    const auto begin = measured.begin;
    const auto end = measured.end;

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
    result.scheduler = schedulerFromMeasuredLog(measured);
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
        }
    }
    if (result.events.empty()) {
        throw std::runtime_error("DRD measured window contains no replay events");
    }
    return result;
}

PairedReplayResult replayPaired(
    const CallgrindTrace& callgrind, const SchedulerTrace& callgrind_scheduler, const DrdTrace& drd, const EventCostModel& model, const PairedReplayOptions& options)
{
    if (options.scheduler_bins == 0 || !std::isfinite(options.scheduler_quantum_slack) || options.scheduler_quantum_slack < 0.0) {
        throw std::runtime_error("scheduler assignment settings are invalid");
    }
    if (!std::isfinite(options.maximum_makespan_spread) || options.maximum_makespan_spread < 0.0 || options.maximum_makespan_spread >= 1.0) {
        throw std::runtime_error("maximum makespan spread must be in [0, 1)");
    }
    if (options.maximum_mappings == 0) {
        throw std::runtime_error("maximum mapping count must be positive");
    }
    Graph graph(drd.events);
    const auto windows = graph.attributionWindows(options.residual_fraction);
    if (callgrind.totals.size() != windows.size() || !callgrind.totals.contains(kMainThread) || !windows.contains(kMainThread)) {
        throw std::runtime_error("Callgrind and DRD thread counts do not match");
    }

    std::vector<std::int64_t> source_threads;
    for (const auto& [thread, total] : callgrind.totals) {
        (void)total;
        if (thread == kMainThread) {
            continue;
        }
        source_threads.push_back(thread);
    }
    std::vector<std::int64_t> trace_threads;
    for (const auto& [thread, thread_windows] : windows) {
        (void)thread_windows;
        if (thread == kMainThread) {
            continue;
        }
        trace_threads.push_back(thread);
    }
    if (source_threads.size() != trace_threads.size()) {
        throw std::runtime_error("Callgrind and DRD worker counts do not match");
    }

    const auto source_descriptors = describeScheduler(callgrind_scheduler, source_threads, options.scheduler_bins);
    const auto trace_descriptors = describeScheduler(drd.scheduler, trace_threads, options.scheduler_bins);
    const double quantum_tolerance =
        2.0 * options.scheduler_quantum_slack *
        (1.0 / static_cast<double>(source_descriptors.total_quanta) + 1.0 / static_cast<double>(trace_descriptors.total_quanta));

    struct Candidate {
        std::map<std::int64_t, std::int64_t> source_by_trace;
        double assignment_score{0.0};
    };
    std::vector<Candidate> candidates;
    std::vector<std::int64_t> permutation = source_threads;
    std::sort(permutation.begin(), permutation.end());
    do {
        if (candidates.size() >= options.maximum_mappings) {
            throw std::runtime_error("worker assignment count exceeds limit");
        }
        Candidate candidate{.source_by_trace = {{kMainThread, kMainThread}}};
        for (std::size_t index = 0; index < trace_threads.size(); ++index) {
            const auto trace_thread = trace_threads[index];
            const auto source_thread = permutation[index];
            candidate.source_by_trace.emplace(trace_thread, source_thread);
            candidate.assignment_score += schedulerDistance(source_descriptors.threads.at(source_thread), trace_descriptors.threads.at(trace_thread));
        }
        candidates.push_back(std::move(candidate));
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (candidates.empty()) {
        throw std::runtime_error("no worker assignments were generated");
    }

    PairedReplayResult result;
    result.evaluated_mapping_count = candidates.size();
    result.best_assignment_score = std::min_element(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
                                       return left.assignment_score < right.assignment_score;
                                   })->assignment_score;
    result.assignment_score_tolerance = quantum_tolerance;
    result.minimum_makespan = std::numeric_limits<double>::infinity();
    double maximum_makespan = -1.0;
    const double maximum_assignment_score = result.best_assignment_score + quantum_tolerance;
    for (const auto& candidate : candidates) {
        if (candidate.assignment_score > maximum_assignment_score + std::numeric_limits<double>::epsilon() * 16.0) {
            continue;
        }
        ++result.mapping_count;
        result.maximum_assignment_score = std::max(result.maximum_assignment_score, candidate.assignment_score);
        const auto costs = mappedWindowCosts(candidate.source_by_trace, callgrind, windows, model);
        const auto durations = graph.assignWindowCosts(costs, options.residual_fraction, options.split_policy);
        const auto replay = graph.replayAdjusted(durations, options.replay);
        result.minimum_makespan = std::min(result.minimum_makespan, replay.modeled_makespan);
        if (replay.modeled_makespan > maximum_makespan) {
            maximum_makespan = replay.modeled_makespan;
            result.conservative = replay;
            result.selected_source_by_trace_thread = candidate.source_by_trace;
        }
    }
    if (result.mapping_count == 0) {
        throw std::runtime_error("no scheduler-compatible worker assignment");
    }
    result.makespan_relative_spread = (maximum_makespan - result.minimum_makespan) / maximum_makespan;
    if (result.makespan_relative_spread > options.maximum_makespan_spread) {
        throw std::runtime_error(
            "assignment evidence insufficient: scheduler-compatible worker assignments exceed the makespan ambiguity limit: spread=" +
            std::to_string(result.makespan_relative_spread) + ", limit=" + std::to_string(options.maximum_makespan_spread));
    }
    return result;
}

}  // namespace vc::thread_sync_replay
