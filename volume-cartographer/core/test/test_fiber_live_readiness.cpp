#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberLiveBridge.hpp"
#include <atomic>
#include <future>

using namespace vc::fiber_tracer;

TEST_CASE("Publication proof is session scoped, bounded, and never inferred on failure") {
    FiberReadinessProfile profile;
    FiberCaseReadiness cache(1, &profile);
    const std::vector<FiberPredictionChunkDependency> a{{"/a", {0,0,0}}}, b{{"/b", {0,0,0}}};
    int waits=0;
    bool active=true;
    auto check=[&] { if(!active) throw std::runtime_error("cancelled"); };
    auto wait=[&](const auto&) { ++waits; };
    cache.await(a, wait, check);
    cache.await(a, wait, check);
    CHECK(waits==1);
    cache.await(b, wait, check);
    cache.await(a, wait, check);
    CHECK(waits==3);
    CHECK(profile.cacheClears==2);
    active=false;
    CHECK_THROWS_AS(cache.await(a, wait, check), std::runtime_error);
    CHECK(waits==3);
    active=true;
    FiberCaseReadiness fresh(1);
    CHECK_THROWS(fresh.await(a, [](const auto&) { throw std::runtime_error("failed"); }, check));
    fresh.await(a, wait, check);
    CHECK(waits==4);
    FiberCaseReadiness disabled(0);
    disabled.await(a, wait, check);
    disabled.await(a, wait, check);
    CHECK(waits==6);
}

TEST_CASE("Concurrent cases can share acknowledged proofs without profiling races") {
    FiberCaseReadiness cache(16);
    std::atomic<int> calls=0;
    std::vector<std::future<void>> jobs;
    for(uint32_t i=0;i<8;++i) jobs.push_back(std::async(std::launch::async,[&,i] {
        cache.await({{"/a",{0,0,i}}},[&](const auto&) { ++calls; },[] {});
    }));
    for(auto& job:jobs) job.get();
    for(uint32_t i=0;i<8;++i)
        cache.await({{"/a",{0,0,i}}},[&](const auto&) { ++calls; },[] {});
    CHECK(calls==8);
}
