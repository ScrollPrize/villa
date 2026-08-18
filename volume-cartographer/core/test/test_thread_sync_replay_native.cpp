#include "thread_sync_replay/Replay.hpp"
#include "thread_sync_replay/RenderValgrind.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace replay = vc::thread_sync_replay;

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        path = std::filesystem::temp_directory_path() /
               ("vc-thread-sync-replay-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() { std::filesystem::remove_all(path); }

    std::filesystem::path path;
};

void writeText(const std::filesystem::path& path, const std::string& value)
{
    std::ofstream stream(path);
    REQUIRE(stream.good());
    stream << value;
}

std::string callgrindProfile(std::int64_t thread, const std::vector<std::int64_t>& totals)
{
    std::string result = "thread: " + std::to_string(thread) + "\n";
    result += "events: Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw Bc Bcm Bi Bim\n";
    result += "totals:";
    for (const auto value : totals) {
        result += " " + std::to_string(value);
    }
    return result + "\n";
}

replay::EventProfile representativeProfile()
{
    return {
        {"Ir", 80},
        {"Dr", 20},
        {"Dw", 10},
        {"D1mr", 2},
        {"D1mw", 1},
        {"DLmr", 1},
        {"DLmw", 0},
        {"Bc", 20},
        {"Bcm", 2},
        {"Bi", 4},
        {"Bim", 1},
    };
}

replay::EventCostModel dataReadModel()
{
    return {
        .feature_names =
            {
                "non_data_instructions",
                "data_reads",
                "data_writes",
                "l1_data_misses",
                "last_level_data_misses",
                "branch_misses",
                "branch_weighted_l1_misses",
            },
        .coefficients_ns = std::vector<double>(7, 1.0),
    };
}

replay::SchedulerTrace schedulerTrace(std::vector<std::int64_t> quantum_threads)
{
    replay::SchedulerTrace result{.quantum_threads = std::move(quantum_threads)};
    for (const auto thread : result.quantum_threads) {
        ++result.full_quanta[thread];
    }
    return result;
}

replay::EventProfile scaledProfile(std::int64_t scale)
{
    auto result = representativeProfile();
    for (auto& [name, value] : result) {
        (void)name;
        value *= scale;
    }
    return result;
}

std::vector<std::int64_t> roundRobinSchedule(std::vector<std::pair<std::int64_t, std::size_t>> remaining)
{
    std::vector<std::int64_t> result;
    bool emitted = true;
    while (emitted) {
        emitted = false;
        for (auto& [thread, count] : remaining) {
            if (count == 0) {
                continue;
            }
            result.push_back(thread);
            --count;
            emitted = true;
        }
    }
    return result;
}

}  // namespace

TEST_CASE("native event-cost scoring matches the frozen feature arithmetic")
{
    const double expected = 50.0 + 20.0 + 10.0 + 3.0 + 1.0 + 3.0 + 9.0 / 80.0;
    CHECK(replay::modeledProfileCostNs(representativeProfile(), dataReadModel()) == doctest::Approx(expected).epsilon(1e-14));

    const auto costs = replay::modeledThreadCostsNs({{2, representativeProfile()}, {1, representativeProfile()}}, dataReadModel());
    CHECK(costs.size() == 2);
    CHECK(costs.at(1) == doctest::Approx(expected));
    CHECK(costs.at(2) == doctest::Approx(expected));
}

TEST_CASE("native event-cost interactions are overflow safe")
{
    auto profile = representativeProfile();
    const auto large = std::numeric_limits<std::int64_t>::max() / 4;
    profile["Ir"] = large;
    profile["D1mr"] = large;
    profile["Bcm"] = large;
    const auto cost = replay::modeledProfileCostNs(profile, dataReadModel());
    CHECK(std::isfinite(cost));
    CHECK(cost > 0.0);
}

TEST_CASE("native event-cost scoring rejects malformed profiles and models")
{
    auto profile = representativeProfile();
    auto model = dataReadModel();

    profile.erase("Ir");
    CHECK_THROWS_AS(replay::modeledProfileCostNs(profile, model), std::runtime_error);
    profile = representativeProfile();
    profile["Dr"] = -1;
    CHECK_THROWS_AS(replay::modeledProfileCostNs(profile, model), std::runtime_error);

    model = dataReadModel();
    model.feature_names[0] = "unknown";
    CHECK_THROWS_AS(replay::modeledProfileCostNs(representativeProfile(), model), std::runtime_error);
    model = dataReadModel();
    model.coefficients_ns.pop_back();
    CHECK_THROWS_AS(replay::modeledProfileCostNs(representativeProfile(), model), std::runtime_error);
    model = dataReadModel();
    model.coefficients_ns[0] = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(replay::modeledProfileCostNs(representativeProfile(), model), std::runtime_error);
    model = dataReadModel();
    model.stall_overlap_fraction = 0.5;
    CHECK_THROWS_AS(replay::modeledProfileCostNs(representativeProfile(), model), std::runtime_error);

    CHECK_THROWS_AS(replay::modeledThreadCostsNs({}, dataReadModel()), std::runtime_error);
    CHECK_THROWS_AS(replay::modeledThreadCostsNs({{0, representativeProfile()}}, dataReadModel()), std::runtime_error);
}

TEST_CASE("native replay attributes a sole zero-weight trailing window")
{
    replay::Graph graph({
        {.thread = 1, .kind = "thread_start"},
        {.thread = 1, .kind = "thread_finish", .dependencies = {{0, "program_order"}}},
    });
    const auto durations = graph.assignCosts({{1, 120.0}}, 0.0, "equal");
    CHECK(durations[0] == 0.0);
    CHECK(durations[1] == 120.0);
}

TEST_CASE("native replay attributes explicit chronological window costs")
{
    replay::Graph graph({
        {.thread = 1, .kind = "work"},
        {.thread = 1, .kind = "work_quantum"},
        {.thread = 1, .kind = "work"},
        {.thread = 1, .kind = "thread_finish"},
    });
    const auto windows = graph.attributionWindows(0.5);
    REQUIRE(windows.at(1).size() == 2);
    CHECK(windows.at(1)[0].units == 1.0);
    CHECK(windows.at(1)[1].units == 0.5);

    const auto durations = graph.assignWindowCosts({{1, {10.0, 30.0}}}, 0.5, "front");
    CHECK(durations[0] == 10.0);
    CHECK(durations[1] == 0.0);
    CHECK(durations[2] == 30.0);
    CHECK(durations[3] == 0.0);
}

TEST_CASE("native replay rejects malformed explicit window costs")
{
    replay::Graph graph({
        {.thread = 1, .kind = "work"},
        {.thread = 1, .kind = "work_quantum"},
        {.thread = 2, .kind = "work"},
    });
    CHECK_THROWS_AS(graph.assignWindowCosts({{1, {1.0}}}, 0.5, "front"), std::runtime_error);
    CHECK_THROWS_AS(graph.assignWindowCosts({{1, {1.0}}, {2, {2.0, 3.0}}}, 0.5, "front"), std::runtime_error);
    CHECK_THROWS_AS(graph.assignWindowCosts({{1, {-1.0}}, {2, {2.0}}}, 0.5, "front"), std::runtime_error);
}

TEST_CASE("native Callgrind parser preserves chronological deltas and totals")
{
    TemporaryDirectory temporary;
    const auto prefix = temporary.path / "callgrind.out";
    writeText(temporary.path / "callgrind.out.1-01", callgrindProfile(1, {10, 2, 1, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0}));
    writeText(temporary.path / "callgrind.out.2-01", callgrindProfile(1, {20, 4, 2, 0, 2, 0, 0, 1, 0, 4, 2, 0, 0}));
    writeText(temporary.path / "callgrind.out-01", callgrindProfile(1, {3, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}));

    const auto parsed = replay::parsePeriodicCallgrind(prefix);
    CHECK(parsed.periodic_dump_count == 2);
    REQUIRE(parsed.slices.at(1).size() == 3);
    CHECK(parsed.slices.at(1)[0].at("Ir") == 10);
    CHECK(parsed.slices.at(1)[1].at("Ir") == 20);
    CHECK(parsed.slices.at(1)[2].at("Ir") == 3);
    CHECK(parsed.totals.at(1).at("Ir") == 33);
    CHECK(parsed.totals.at(1).at("D1mr") == 3);
}

TEST_CASE("native Callgrind parser ignores zero residual-only idle threads")
{
    TemporaryDirectory temporary;
    const auto prefix = temporary.path / "callgrind.out";
    writeText(temporary.path / "callgrind.out.1-01", callgrindProfile(1, {10, 2, 1}));
    writeText(temporary.path / "callgrind.out.1-02", callgrindProfile(2, {0, 0, 0}));
    writeText(temporary.path / "callgrind.out-01", callgrindProfile(1, {0, 0, 0}));
    writeText(temporary.path / "callgrind.out-02", callgrindProfile(2, {0, 0, 0}));
    writeText(temporary.path / "callgrind.out-03", callgrindProfile(3, {0, 0, 0}));

    const auto parsed = replay::parsePeriodicCallgrind(prefix);
    REQUIRE(parsed.totals.size() == 1);
    CHECK(parsed.totals.contains(1));
    CHECK_FALSE(parsed.totals.contains(2));
    CHECK_FALSE(parsed.totals.contains(3));
}

TEST_CASE("native Callgrind parser rejects nonzero residual-only threads")
{
    TemporaryDirectory temporary;
    const auto prefix = temporary.path / "callgrind.out";
    writeText(temporary.path / "callgrind.out.1-01", callgrindProfile(1, {10, 2, 1}));
    writeText(temporary.path / "callgrind.out-01", callgrindProfile(1, {0, 0, 0}));
    writeText(temporary.path / "callgrind.out-02", callgrindProfile(2, {1, 0, 0}));

    CHECK_THROWS_WITH_AS(replay::parsePeriodicCallgrind(prefix), doctest::Contains("Callgrind residual-only thread contains measured work"), std::runtime_error);
}

TEST_CASE("native DRD parser trims to the passive measured clock pair")
{
    TemporaryDirectory temporary;
    const auto trace = temporary.path / "drd.log";
    writeText(
        trace,
        "-- New segment for thread 1 with vc [ 1: 1 ]\n"
        "SYSCALL[2,1](228) sys_clock_gettime( 1, 0xaaa )[sync] --> Success(0x0)\n"
        "-- New segment for thread 1 with vc [ 1: 2 ]\n"
        "-- New segment for thread 2 with vc [ 1: 2, 2: 1 ]\n"
        "-- SCHED[2]: releasing lock (VG_(scheduler):timeslice) -> VgTs_Yielding\n"
        "SYSCALL[2,1](228) sys_clock_gettime( 1, 0xaaa )[sync] --> Success(0x0)\n"
        "-- New segment for thread 1 with vc [ 1: 3, 2: 1 ]\n");

    const auto parsed = replay::parseMeasuredDrd(trace);
    CHECK(parsed.parsed_segment_count == 3);
    CHECK(parsed.retained_segment_count == 2);
    CHECK(parsed.events.size() == 3);
    CHECK(parsed.happens_before_edges == 1);
    CHECK(parsed.scheduler.full_quanta.at(2) == 1);
    CHECK(parsed.events[1].dependencies.back().kind == "drd_happens_before");
}

TEST_CASE("native scheduler parser preserves measured quantum order")
{
    TemporaryDirectory temporary;
    const auto trace = temporary.path / "scheduler.log";
    writeText(
        trace,
        "-- SCHED[9]: releasing lock (VG_(scheduler):timeslice) -> VgTs_Yielding\n"
        "SYSCALL[2,1](228) sys_clock_gettime( 1, 0xaaa )[sync] --> Success(0x0)\n"
        "-- SCHED[2]: releasing lock (VG_(scheduler):timeslice) -> VgTs_Yielding\n"
        "-- SCHED[3]: releasing lock (VG_(scheduler):timeslice) -> VgTs_Yielding\n"
        "-- SCHED[2]: releasing lock (VG_(scheduler):timeslice) -> VgTs_Yielding\n"
        "SYSCALL[2,1](228) sys_clock_gettime( 1, 0xaaa )[sync] --> Success(0x0)\n"
        "-- SCHED[9]: releasing lock (VG_(scheduler):timeslice) -> VgTs_Yielding\n");

    const auto parsed = replay::parseMeasuredScheduler(trace);
    CHECK(parsed.quantum_threads == std::vector<std::int64_t>{2, 3, 2});
    CHECK(parsed.full_quanta.at(2) == 2);
    CHECK(parsed.full_quanta.at(3) == 1);
    CHECK(parsed.begin_line == 2);
    CHECK(parsed.end_line == 6);
}

TEST_CASE("native paired replay uses same-run scheduler evidence")
{
    replay::CallgrindTrace callgrind;
    for (std::int64_t thread = 1; thread <= 3; ++thread) {
        callgrind.slices[thread] = {scaledProfile(thread)};
        callgrind.totals[thread] = scaledProfile(thread);
    }
    const auto callgrind_scheduler = schedulerTrace({2, 3, 2, 2, 2, 2, 2, 2});
    replay::DrdTrace drd{
        .events =
            {
                {.thread = 1, .kind = "work_quantum"},
                {.thread = 3, .kind = "work_quantum"},
                {.thread = 2, .kind = "work_quantum"},
            },
        .scheduler = schedulerTrace({3, 2, 3, 3, 3, 3, 3, 3}),
    };
    const auto result =
        replay::replayPaired(callgrind, callgrind_scheduler, drd, dataReadModel(), {.scheduler_bins = 4, .scheduler_quantum_slack = 0.0, .replay = {.cores = 3}});
    CHECK(result.evaluated_mapping_count == 2);
    CHECK(result.mapping_count == 1);
    CHECK(result.selected_source_by_trace_thread.at(2) == 3);
    CHECK(result.selected_source_by_trace_thread.at(3) == 2);
    CHECK(result.conservative.modeled_work > 0.0);
    CHECK(result.minimum_makespan == result.conservative.modeled_makespan);
}

TEST_CASE("native paired replay rejects material assignment ambiguity")
{
    replay::CallgrindTrace callgrind;
    callgrind.slices[1] = {scaledProfile(10)};
    callgrind.totals[1] = scaledProfile(10);
    callgrind.slices[2] = {scaledProfile(1)};
    callgrind.totals[2] = scaledProfile(1);
    callgrind.slices[3] = {scaledProfile(10)};
    callgrind.totals[3] = scaledProfile(10);
    const auto callgrind_scheduler = schedulerTrace({2, 3, 2, 3});
    replay::DrdTrace drd{
        .events =
            {
                {.thread = 2, .kind = "work_quantum"},
                {.thread = 3, .kind = "work_quantum"},
                {.thread = 1, .kind = "work_quantum", .dependencies = {{0, "drd_happens_before"}}},
            },
        .scheduler = schedulerTrace({2, 3, 2, 3}),
    };
    CHECK_THROWS_WITH_AS(replay::replayPaired(callgrind, callgrind_scheduler, drd, dataReadModel(), {.scheduler_bins = 1, .scheduler_quantum_slack = 0.0, .maximum_makespan_spread = 0.02, .replay = {.cores = 3}}), doctest::Contains("assignment evidence insufficient"), std::runtime_error);
}

TEST_CASE("native paired replay resolves a shifted DRD tie from Callgrind scheduling")
{
    replay::CallgrindTrace callgrind;
    callgrind.slices[1] = {scaledProfile(1)};
    callgrind.totals[1] = scaledProfile(1);
    for (const auto [thread, scale] : std::vector<std::pair<std::int64_t, std::int64_t>>{{2, 4}, {3, 5}, {4, 8}, {5, 10}}) {
        callgrind.slices[thread] = {scaledProfile(scale)};
        callgrind.totals[thread] = scaledProfile(scale);
    }
    const auto callgrind_scheduler = schedulerTrace(roundRobinSchedule({{2, 42}, {3, 43}, {4, 46}, {5, 50}}));
    replay::DrdTrace drd{
        .events =
            {
                {.thread = 1, .kind = "work_quantum"},
                {.thread = 6, .kind = "work_quantum"},
                {.thread = 7, .kind = "work_quantum"},
                {.thread = 8, .kind = "work_quantum"},
                {.thread = 9, .kind = "work_quantum"},
            },
        .scheduler = schedulerTrace(roundRobinSchedule({{6, 38}, {7, 38}, {8, 41}, {9, 45}})),
    };
    const auto result =
        replay::replayPaired(callgrind, callgrind_scheduler, drd, dataReadModel(), {.scheduler_bins = 16, .scheduler_quantum_slack = 1.0, .maximum_makespan_spread = 0.02, .replay = {.cores = 5}});
    CHECK(result.evaluated_mapping_count == 24);
    CHECK(result.mapping_count == 2);
    CHECK(result.selected_source_by_trace_thread.at(8) == 4);
    CHECK(result.selected_source_by_trace_thread.at(9) == 5);
    CHECK(result.makespan_relative_spread == 0.0);
}

TEST_CASE("native replay applies cross-thread and wake latency cumulatively")
{
    replay::Graph graph({
        {.thread = 1, .kind = "futex_wake"},
        {.thread = 2, .kind = "work", .dependencies = {{0, "futex_wake"}}},
    });
    const auto durations = graph.assignCosts({{1, 10.0}, {2, 5.0}}, 0.5, "front");
    const auto result = graph.replayAdjusted(
        durations,
        {
            .cores = 2,
            .wake_latency = 3.0,
            .cross_thread_latency = 7.0,
        });
    CHECK(result.modeled_makespan == 25.0);
    CHECK(result.dependency_critical_path == 25.0);
}

TEST_CASE("native replay preserves hard lower bound while scaling DRD excess")
{
    replay::Graph graph({
        {.thread = 1, .kind = "work"},
        {.thread = 2, .kind = "work", .dependencies = {{0, "drd_happens_before"}}},
    });
    const auto durations = graph.assignCosts({{1, 10.0}, {2, 10.0}}, 0.5, "front");
    const auto result = graph.replayAdjusted(
        durations,
        {
            .cores = 2,
            .dependency_excess_scale = 0.5,
        });
    CHECK(result.hard_schedule_lower_bound == 10.0);
    CHECK(result.dependency_excess == 10.0);
    CHECK(result.modeled_makespan == 15.0);
}

TEST_CASE("native replay uses the last duplicate dependency kind")
{
    replay::Graph graph({
        {.thread = 1, .kind = "work"},
        {.thread = 2,
         .kind = "work",
         .dependencies =
             {
                 {0, "drd_happens_before"},
                 {0, "program_order"},
             }},
    });
    const auto durations = graph.assignCosts({{1, 10.0}, {2, 5.0}}, 0.5, "front");
    const auto result = graph.replayAdjusted(
        durations,
        {
            .cores = 2,
            .cross_thread_latency = 100.0,
        });
    CHECK(result.modeled_makespan == 15.0);
}

TEST_CASE("native replay rejects malformed and cyclic graphs")
{
    CHECK_THROWS_AS(
        replay::Graph({
            {.thread = 1, .kind = "work", .dependencies = {{2, "program_order"}}},
        }),
        std::runtime_error);

    replay::Graph cyclic({
        {.thread = 1, .kind = "work", .dependencies = {{1, "program_order"}}},
        {.thread = 1, .kind = "work", .dependencies = {{0, "program_order"}}},
    });
    const auto durations = cyclic.assignCosts({{1, 2.0}}, 0.5, "equal");
    CHECK_THROWS_AS(cyclic.replayAdjusted(durations, {.cores = 1}), std::runtime_error);
}

TEST_CASE("native FIFO and round-robin replay are deterministic")
{
    replay::Graph graph({
        {.thread = 1, .kind = "work"},
        {.thread = 2, .kind = "work"},
        {.thread = 3, .kind = "work"},
    });
    const auto durations = graph.assignCosts({{1, 3.0}, {2, 3.0}, {3, 2.0}}, 0.5, "front");
    for (const std::string policy : {"fifo", "round_robin"}) {
        const replay::ReplayOptions options{.cores = 2, .tie_policy = policy};
        const auto first = graph.replayAdjusted(durations, options);
        const auto second = graph.replayAdjusted(durations, options);
        CHECK(first.modeled_makespan == second.modeled_makespan);
        CHECK(first.simulated_core_idle == second.simulated_core_idle);
    }
}
