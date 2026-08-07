#include "thread_sync_replay/Replay.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace replay = vc::thread_sync_replay;

namespace
{

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
        {"Bcm", 2},
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
