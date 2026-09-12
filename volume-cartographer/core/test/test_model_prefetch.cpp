#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/ModelPrefetch.hpp"
#include "vc/core/render/ChunkFetch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <set>
#include <thread>

namespace {
using namespace vc::render;
using namespace vc::lasagna;

class RecordingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        std::lock_guard lock(mutex);
        keys.insert({key.iz, key.iy, key.ix});
        ChunkFetchResult result;
        result.status = fail ? ChunkFetchStatus::HttpError : ChunkFetchStatus::Found;
        result.bytes.assign(64, std::byte{17});
        return result;
    }
    std::mutex mutex;
    bool fail = false; // configured before requests start
    std::set<std::array<int, 3>> keys;
};

struct Source {
    std::shared_ptr<RecordingFetcher> fetcher = std::make_shared<RecordingFetcher>();
    std::shared_ptr<ChunkCache> cache;
    explicit Source(size_t capacity = 1024 * 1024)
    {
        ChunkCacheService::Options service;
        service.decodedByteCapacity = capacity;
        service.fetchConcurrency.maxConcurrentReads = 4;
        service.fetchConcurrency.adaptive = false;
        cache = std::make_shared<ChunkCache>(
            std::vector<ChunkCache::LevelInfo>{{{64, 64, 64}, {4, 4, 4}, {}}},
            std::vector<std::shared_ptr<IChunkFetcher>>{fetcher}, 0.0,
            ChunkDtype::UInt8, ChunkCache::Options{}, service);
    }
    ModelPrefetchSource source(double spacing = 1.0, bool remote = true) const
    { return {cache, spacing, remote}; }
};
}

TEST_CASE("model prefetch covers dense and sparse corridors identically")
{
    Source source;
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan sparse({source.source()}, {{4, 8, 8}, {60, 8, 8}}, options);
    CHECK(sparse.report().planned == 64);
    std::vector<cv::Vec3d> dense;
    for (int i = 0; i <= 10000; ++i)
        dense.emplace_back(4.0 + 56.0 * i / 10000.0, 8, 8);
    ModelPrefetchPlan plan({source.source()}, dense, options);
    CHECK(plan.report().planned == sparse.report().planned);
    CHECK_FALSE(plan.report().truncated);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool done = false;
    while (!(done = plan.pump()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(done);
    for (int z : {1, 2})
        for (int y : {1, 2})
            for (int x = 0; x < 16; ++x)
                REQUIRE(source.cache->getChunkBlocking(0, z, y, x).status == ChunkStatus::Data);
    std::lock_guard lock(source.fetcher->mutex);
    CHECK(source.fetcher->keys.size() == 64);
    CHECK(plan.report().submitted == 64);
}

TEST_CASE("model prefetch uses each channel's own coordinate scale")
{
    Source a, b;
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan first({a.source()}, {{4, 8, 8}, {60, 8, 8}}, options);
    ModelPrefetchPlan scaled({b.source(2)}, {{8, 16, 16}, {120, 16, 16}}, options);
    CHECK(first.report().planned == scaled.report().planned);
    ModelPrefetchPlan both({a.source(), b.source(2)}, {{8, 16, 16}}, options);
    CHECK(both.report().planned == 16);
    CHECK(both.report().plannedBytes == 16 * 64);
}

TEST_CASE("moving window is bounded and cancellation or local data submits nothing")
{
    Source source;
    ModelPrefetchWindowOptions options;
    options.lookahead = 8;
    options.refreshDistance = 4;
    options.corridor.radius = 0;
    options.corridor.maxRequests = 3;
    ModelPrefetchWindow local({source.source(1, false)}, options);
    CHECK(local.advance({8, 8, 8}, {1, 0, 0}, 100).submitted == 0);
    std::atomic<bool> cancelled{true};
    ModelPrefetchWindow window({source.source()}, options);
    CHECK(window.advance({8, 8, 8}, {1, 0, 0}, 100, &cancelled).submitted == 0);
    const auto first = window.advance({8, 8, 8}, {1, 0, 0}, 100);
    CHECK(first.submitted <= 3);
    REQUIRE(first.submitted > 0);
    CHECK(window.advance({8, 8, 8}, {1, 0, 0}, 100).submitted == 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (ChunkCache::speculativePrefetchStats().pendingRequests != 0 &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(ChunkCache::speculativePrefetchStats().pendingRequests == 0);
    const auto moved = window.advance({32, 8, 8}, {1, 0, 0}, 0);
    CHECK(moved.submitted <= 3);
    CHECK(moved.submitted > 0);
}

TEST_CASE("moving window ignores invalid options")
{
    Source source;
    ModelPrefetchWindowOptions options;
    options.lookahead = std::numeric_limits<double>::quiet_NaN();
    ModelPrefetchWindow invalid({source.source()}, options);
    CHECK(invalid.advance({8, 8, 8}, {1, 0, 0}, 100).submitted == 0);
    CHECK_NOTHROW(invalid.advance({8, 8, 8}, {1, 0, 0}, 100));
}

TEST_CASE("straight predictor preserves the old ray and distance-only refresh")
{
    ModelPrefetchWindowOptions options;
    options.lookahead = 20;
    options.refreshDistance = 8;
    ModelPrefetchPredictor predictor(options);
    auto first = predictor.update({1, 2, 3}, {2, 0, 0}, 12);
    REQUIRE(first);
    REQUIRE(first->points.size() == 2);
    CHECK(cv::norm(first->points.back() - cv::Vec3d{25, 2, 3}) == 0);
    CHECK_FALSE(first->reference);
    CHECK_FALSE(predictor.update({2, 2, 3}, {0, 1, 0}, 12));
    auto moved = predictor.update({9, 2, 3}, {0, 1, 0}, -1);
    REQUIRE(moved);
    CHECK(cv::norm(moved->points.back() - cv::Vec3d{9, 2, 3}) == 0);
    CHECK_FALSE(moved->turnRefresh);
}

TEST_CASE("guided predictor follows an oriented scaled copy and stops at its endpoint")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Guided;
    options.lookahead = 50;
    options.refreshDistance = 8;
    options.corridor.radius = 4;
    std::vector<cv::Vec3d> reference{{0, 0, 0}, {40, 0, 0}, {40, 80, 0}};
    ModelPrefetchPredictor forward(options, {reference, false, 0.5});
    ModelPrefetchPredictor reverse(options, {reference, true, 0.5});
    reference.clear(); // the bounded predictor owns its copy
    auto first = forward.update({0, 0, 0}, {1, 0, 0}, 100);
    REQUIRE(first);
    CHECK(first->reference);
    CHECK(cv::norm(first->points.back() - cv::Vec3d{20, 30, 0}) < 1e-10);
    auto backward = reverse.update({20, 40, 0}, {0, -1, 0}, 100);
    REQUIRE(backward);
    CHECK(backward->reference);
    CHECK(cv::norm(backward->points.back() - cv::Vec3d{10, 0, 0}) < 1e-10);
    auto corner = forward.update({20, 0, 0}, {0, 1, 0}, 100);
    REQUIRE(corner);
    CHECK(corner->reference);
    auto end = forward.update({20, 30, 0}, {0, 1, 0}, 100);
    REQUIRE(end);
    CHECK(end->reference);
    CHECK(cv::norm(end->points.back() - cv::Vec3d{20, 40, 0}) < 1e-10);
    auto past = forward.update({20, 42, 0}, {0, 1, 0}, 5);
    REQUIRE(past);
    CHECK_FALSE(past->reference);
    CHECK(past->referenceFallback);
    CHECK(cv::norm(past->points.back() - cv::Vec3d{20, 47, 0}) < 1e-10);
    auto stopped = forward.update({20, 50, 0}, {0, 1, 0}, 0);
    REQUIRE(stopped);
    CHECK_FALSE(stopped->referenceFallback);
}

TEST_CASE("guided matching rejects stale or opposed references and preserves origin continuity")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Guided;
    options.lookahead = 10;
    options.corridor.radius = 4;
    std::vector<cv::Vec3d> reference{{0, 0, 0}, {100, 0, 0}};
    ModelPrefetchPredictor nearby(options, {reference});
    auto offset = nearby.update({0, 3, 0}, {1, 0, 0}, 6);
    REQUIRE(offset);
    CHECK(offset->reference);
    CHECK(cv::norm(offset->points.front() - cv::Vec3d{0, 3, 0}) == 0);
    CHECK(cv::norm(offset->points.back() - cv::Vec3d{6, 3, 0}) < 1e-10);
    ModelPrefetchPredictor stale(options, {reference});
    auto distant = stale.update({0, 10, 0}, {1, 0, 0}, 6);
    REQUIRE(distant);
    CHECK_FALSE(distant->reference);
    CHECK(distant->referenceFallback);
    ModelPrefetchPredictor reversed(options, {reference});
    auto opposed = reversed.update({0, 0, 0}, {-1, 0, 0}, 6);
    REQUIRE(opposed);
    CHECK_FALSE(opposed->reference);
    CHECK(cv::norm(opposed->points.back() - cv::Vec3d{-6, 0, 0}) == 0);
}

TEST_CASE("reference work is bounded without bridging invalid or truncated input")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Guided;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<cv::Vec3d> broken{{0, 0, 0}, {0, 0, 0}, {2, 0, 0}, {nan, 0, 0}, {200, 0, 0}};
    ModelPrefetchPredictor prefix(options, {broken});
    auto valid = prefix.update({0, 0, 0}, {1, 0, 0}, 256);
    REQUIRE(valid);
    CHECK(valid->reference);
    CHECK(cv::norm(valid->points.back() - cv::Vec3d{2, 0, 0}) == 0);
    std::vector<cv::Vec3d> duplicates(2048, cv::Vec3d{0, 0, 0});
    duplicates.emplace_back(200, 100, 0); // must not inspect beyond the input cap
    ModelPrefetchPredictor capped(options, {duplicates});
    auto fallback = capped.update({0, 0, 0}, {1, 0, 0}, 256);
    REQUIRE(fallback);
    CHECK_FALSE(fallback->reference);
    std::vector<cv::Vec3d> dense;
    for (int i = 0; i < 3000; ++i) dense.emplace_back(i * 0.1, 0, 0);
    ModelPrefetchPredictor bounded(options, {dense});
    auto shortPlan = bounded.update({0, 0, 0}, {1, 0, 0}, 256);
    REQUIRE(shortPlan);
    CHECK(shortPlan->reference);
    CHECK(shortPlan->points.size() <= 129);
    CHECK(shortPlan->points.back()[0] <= 12.800001);
}

TEST_CASE("reference matching stays on the local branch at a self-approach")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Guided;
    options.refreshDistance = 8;
    options.lookahead = 16;
    options.corridor.radius = 4;
    std::vector<cv::Vec3d> reference{{0, 0, 0}, {20, 0, 0}, {20, 100, 0},
        {-20, 100, 0}, {-20, 1, 0}, {20, 1, 0}, {20, -100, 0}};
    ModelPrefetchPredictor predictor(options, {reference});
    REQUIRE(predictor.update({0, 0, 0}, {1, 0, 0}, 100));
    auto nearbyBranch = predictor.update({8, 1, 0}, {1, 0, 0}, 100);
    REQUIRE(nearbyBranch);
    CHECK(nearbyBranch->reference);
    CHECK(cv::norm(nearbyBranch->points.back() - cv::Vec3d{20, 5, 0}) < 1e-10);
    // Monotone progress will not follow the old curve backward after a reversal.
    auto backward = predictor.update({0, 0, 0}, {-1, 0, 0}, 100);
    REQUIRE(backward);
    CHECK_FALSE(backward->reference);
}

TEST_CASE("turn refresh accumulates relative to the plan and throttles stationary jitter")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Guided;
    options.refreshDistance = 64;
    ModelPrefetchPredictor predictor(options);
    REQUIRE(predictor.update({0, 0, 0}, {1, 0, 0}, 100));
    CHECK_FALSE(predictor.update({0, 0, 0}, {0, 1, 0}, 100));
    for (int i = 1; i <= 3; ++i)
        CHECK_FALSE(predictor.update({double(i * 5), 0, 0}, {std::cos(i * .08), std::sin(i * .08), 0}, 100));
    auto turn = predictor.update({20, 0, 0}, {std::cos(.32), std::sin(.32), 0}, 100);
    REQUIRE(turn);
    CHECK(turn->turnRefresh);
    CHECK_FALSE(predictor.update({21, 0, 0}, {0, 1, 0}, 100));
}

TEST_CASE("reference plans already contain bends and do not need turn-only refresh")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Guided;
    std::vector<cv::Vec3d> reference{{0, 0, 0}, {20, 0, 0}, {20, 300, 0}};
    ModelPrefetchPredictor predictor(options, {reference});
    auto first = predictor.update({0, 0, 0}, {1, 0, 0}, 300);
    REQUIRE(first);
    CHECK(first->reference);
    CHECK_FALSE(predictor.update({20, 0, 0}, {0, 1, 0}, 300));
    auto moved = predictor.update({20, 64, 0}, {0, 1, 0}, 300);
    REQUIRE(moved);
    CHECK(moved->reference);
    CHECK_FALSE(moved->turnRefresh);
    auto leftGuide = predictor.update({100, 64, 0}, {1, 0, 0}, 300);
    REQUIRE(leftGuide);
    CHECK(leftGuide->referenceFallback);
    auto turned = predictor.update({100, 84, 0}, {0, 1, 0}, 300);
    REQUIRE(turned);
    CHECK(turned->turnRefresh);
}

TEST_CASE("curvature uses spatial observations with bounded turn and resets on reversal")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Curved;
    options.lookahead = 64;
    options.refreshDistance = 32;
    ModelPrefetchPredictor predictor(options);
    std::optional<ModelPrefetchPrediction> latest;
    cv::Vec3d origin, direction;
    for (int i = 0; i <= 8; ++i) {
        const double theta = i * .05;
        origin = {200 * std::sin(theta), 200 * (1 - std::cos(theta)), 0};
        direction = {std::cos(theta), std::sin(theta), 0};
        if (auto next = predictor.update(origin, direction, 100)) latest = next;
    }
    REQUIRE(latest);
    REQUIRE(latest->curved);
    REQUIRE(latest->points.size() == 9);
    double arc = 0;
    for (size_t i = 1; i < latest->points.size(); ++i)
        arc += cv::norm(latest->points[i] - latest->points[i - 1]);
    CHECK(std::abs(arc - 64) < 1e-8);
    const auto finalDelta = latest->points.back() - latest->points[7];
    const double finalTurn = std::acos(std::clamp(finalDelta.dot(direction) / cv::norm(finalDelta), -1.0, 1.0));
    CHECK(finalTurn > 0);
    CHECK(finalTurn <= .2618);
    auto reversal = predictor.update(origin - direction * 20, -direction, 100);
    REQUIRE(reversal);
    CHECK_FALSE(reversal->curved);
}

TEST_CASE("curvature rejects alternating turns and invalid guided directions")
{
    ModelPrefetchWindowOptions options;
    options.projection = ModelPrefetchProjection::Curved;
    options.refreshDistance = 16;
    ModelPrefetchPredictor predictor(options);
    std::optional<ModelPrefetchPrediction> latest;
    for (int i = 0; i < 10; ++i) {
        const double angle = i % 2 ? .1 : -.1;
        latest = predictor.update({double(i * 20), 0, 0}, {std::cos(angle), std::sin(angle), 0}, 100);
    }
    REQUIRE(latest);
    CHECK_FALSE(latest->curved);
    CHECK_FALSE(predictor.update({300, 0, 0}, {0, 0, 0}, 100));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(predictor.update({nan, 0, 0}, {1, 0, 0}, 100));
    CHECK_FALSE(predictor.update({300, 0, 0}, {1, 0, 0}, nan));
}

TEST_CASE("model prefetch bounds planning and bytes independently")
{
    Source source;
    ModelPrefetchOptions options;
    options.maxRequests = 7;
    options.maxPlannedBytes = 3 * 64;
    ModelPrefetchPlan bytes({source.source()}, {{0, 0, 0}, {63, 63, 63}}, options);
    CHECK(bytes.report().planned <= 3);
    CHECK(bytes.report().plannedBytes <= options.maxPlannedBytes);
    CHECK(bytes.report().truncated);
    options.maxPlanningSteps = 1;
    ModelPrefetchPlan work({source.source()}, {{0, 0, 0}, {63, 63, 63}}, options);
    CHECK(work.report().planned == 0);
    CHECK(work.report().truncated);
}

TEST_CASE("model prefetch clips outside points and ignores invalid or local inputs")
{
    Source source;
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan clipped({source.source()}, {{-100000, 8, 8}, {100000, 8, 8}}, options);
    CHECK(clipped.report().planned == 64);
    ModelPrefetchPlan outside({source.source()}, {{-100, -100, -100}}, options);
    CHECK(outside.report().planned == 0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ModelPrefetchPlan invalid({source.source()}, {{nan, 8, 8}}, options);
    CHECK(invalid.report().planned == 0);
    ModelPrefetchPlan local({source.source(1, false), source.source(0), {nullptr, 1, true}},
                            {{8, 8, 8}}, options);
    CHECK(local.report().planned == 0);
}

TEST_CASE("model prefetch cancellation issues no new requests")
{
    Source source;
    ModelPrefetchPlan plan({source.source()}, {{8, 8, 8}});
    REQUIRE(plan.report().planned > 0);
    std::atomic<bool> cancelled{true};
    CHECK(plan.pump(&cancelled));
    CHECK(plan.report().submitted == 0);
    CHECK(plan.pump());
    std::lock_guard lock(source.fetcher->mutex);
    CHECK(source.fetcher->keys.empty());
}

TEST_CASE("model prefetch skips permanent omissions without stalling other channels")
{
    bool tooSmall = false;
    SUBCASE("decoded budget cannot fit a chunk") { tooSmall = true; }
    SUBCASE("an earlier required read cached an error") { tooSmall = false; }
    Source unavailable(tooSmall ? 64 : 1024 * 1024), available;
    if (!tooSmall) {
        unavailable.fetcher->fail = true;
        REQUIRE(unavailable.cache->getChunkBlocking(0, 1, 1, 1).status == ChunkStatus::Error);
    }
    ModelPrefetchOptions options;
    options.radius = 0;
    ModelPrefetchPlan plan({unavailable.source(), available.source()}, {{8, 8, 8}}, options);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool done = false;
    while (!(done = plan.pump()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(done);
    CHECK(plan.report().skipped >= (tooSmall ? 8 : 1));
    for (int z : {1, 2})
        for (int y : {1, 2})
            for (int x : {1, 2})
                CHECK(available.cache->getChunkBlocking(0, z, y, x).status == ChunkStatus::Data);
    std::lock_guard lock(available.fetcher->mutex);
    CHECK(available.fetcher->keys.size() == 8);
}
