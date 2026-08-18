#pragma once

#include "thread_sync_replay/Replay.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

namespace vc::thread_sync_replay
{

struct CallgrindTrace {
    std::map<std::int64_t, std::vector<EventProfile>> slices;
    std::map<std::int64_t, EventProfile> totals;
    std::size_t periodic_dump_count{0};
};

struct DrdTrace {
    std::vector<Event> events;
    std::map<std::int64_t, std::size_t> full_quanta;
    std::size_t happens_before_edges{0};
    std::size_t dropped_pre_window_edges{0};
    std::size_t parsed_segment_count{0};
    std::size_t retained_segment_count{0};
    std::size_t begin_line{0};
    std::size_t end_line{0};
};

struct PairedReplayOptions {
    double residual_fraction{0.5};
    std::string split_policy{"equal"};
    double equivalent_cost_tolerance{0.05};
    double equivalent_shape_tolerance{0.02};
    std::size_t equivalence_bins{32};
    std::size_t maximum_mappings{100000};
    ReplayOptions replay;
};

struct PairedReplayResult {
    ReplayResult conservative;
    double minimum_makespan{0.0};
    std::size_t mapping_count{0};
    double maximum_equivalent_cost_spread{0.0};
    double maximum_equivalent_shape_spread{0.0};
    std::map<std::int64_t, std::int64_t> selected_source_by_trace_thread;
};

[[nodiscard]] CallgrindTrace parsePeriodicCallgrind(const std::filesystem::path& prefix);

[[nodiscard]] DrdTrace parseMeasuredDrd(const std::filesystem::path& path);

[[nodiscard]] PairedReplayResult replayPaired(const CallgrindTrace& callgrind, const DrdTrace& drd, const EventCostModel& model, const PairedReplayOptions& options);

}  // namespace vc::thread_sync_replay
