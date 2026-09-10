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

struct SchedulerTrace {
    std::vector<std::int64_t> quantum_threads;
    std::map<std::int64_t, std::size_t> full_quanta;
    std::size_t begin_line{0};
    std::size_t end_line{0};
};

struct CallgrindSyncTrace {
    std::vector<Event> events;
    SchedulerTrace scheduler;
    std::size_t futex_happens_before_edges{0};
    std::size_t dropped_pre_window_edges{0};
    std::size_t parsed_futex_calls{0};
    std::size_t blocking_futex_waits{0};
    std::size_t begin_line{0};
    std::size_t end_line{0};
};

struct CallgrindReplayOptions {
    double residual_fraction{0.5};
    std::string split_policy{"equal"};
    ReplayOptions replay;
};

[[nodiscard]] CallgrindTrace parsePeriodicCallgrind(const std::filesystem::path& prefix);

[[nodiscard]] CallgrindSyncTrace parseMeasuredCallgrindSync(const std::filesystem::path& path);

[[nodiscard]] ReplayResult replayCallgrind(
    const CallgrindTrace& callgrind, const CallgrindSyncTrace& sync, const EventCostModel& model, const CallgrindReplayOptions& options);

}  // namespace vc::thread_sync_replay
