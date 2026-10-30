#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberEmbedding.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>

using namespace vc::fiber_tracer;

namespace {
class ForkPrediction final : public FiberPredictionSource {
public:
    mutable size_t embeddingCalls = 0;
    bool constant = false, missing = false, wrongSize = false;
    bool supportsConcurrentSampling() const noexcept override { return true; }
    FiberPredictionSample sample(const cv::Vec3d& p, const cv::Vec3d&) const override {
        FiberPredictionSample out;
        // Ordinary presence favors the straight continuation; embedding agrees
        // only on the displaced branch. No GT is supplied to the tracer.
        out.options.push_back({{1,0,0}, p[1] > .5 ? .9f : 1.f, true});
        return out;
    }
    void sampleEmbeddings(const std::vector<cv::Vec3f>& points,
        std::vector<std::vector<float>>& values) const override {
        ++embeddingCalls;
        values.clear();
        if (wrongSize) return;
        for (const auto& p : points)
            values.push_back(missing && p[0] > 0 ? std::vector<float>{} :
                constant || p[0] == 0 || p[1] > .5 ? std::vector<float>{1,0} : std::vector<float>{0,1});
    }
};

FiberTraceConfig config() {
    FiberTraceConfig c;
    c.smoothnessWeight = c.smoothnessNormalWeight = c.smoothnessTangentWeight = 0;
    c.cumulativeSmoothnessTangentWeight = 0;
    c.stepVoxels = 4;
    c.coneAngleDegrees = 25;
    c.coneAngleStepDegrees = 5;
    c.beamWidth = 3;
    c.parallelThreads = 1;
    return c;
}
}

TEST_CASE("embedding history uses signed cosine, bounded arclength and stable falloff") {
    FiberEmbeddingHistory history{{0,{1,0}}, {32,{0,1}}};
    CHECK(fiberEmbeddingCost({1,0},history,64,64) == .5);
    CHECK(fiberEmbeddingCost({1,0},history,64,64,"uniform",true) == 0);
}

TEST_CASE("embedding history weighting and admission match the extracted formulas") {
    FiberEmbeddingHistory h{{0,{1,0}}, {32,{0,1}}};
    CHECK(fiberEmbeddingCost({-1,0},h,64,64) == 1.5);
    CHECK(fiberEmbeddingCost({1,0},h,64,64,"linear") == 1);
    const double w0=std::exp(-2.), w1=std::exp(-.5);
    CHECK(fiberEmbeddingCost({1,0},h,64,64,"gaussian") == doctest::Approx(w1/(w0+w1)));
    CHECK(fiberEmbeddingCost({1,0},h,64,64,"uniform",true) == 0);
    CHECK(fiberEmbeddingCost({1,0},h,97,64) == 0);
    auto sibling = h;
    advanceFiberEmbeddingHistory(h,{1,0},40,16,64,2);
    CHECK(h.size()==2);
    advanceFiberEmbeddingHistory(h,{1,0},48,16,64,2);
    CHECK(h.front().arcBase==32);
    CHECK(sibling.front().arcBase==0);
    advanceFiberEmbeddingHistory(h,{1,0},200,16,64,2);
    REQUIRE(h.size()==1);
    CHECK(h.front().arcBase==200);
    CHECK_THROWS_AS(fiberEmbeddingCost({1},sibling,64,64),std::invalid_argument);
    CHECK_THROWS_AS(fiberEmbeddingCost({1,0},sibling,64,64,"bad"),std::invalid_argument);
    CHECK_THROWS_AS(fiberEmbeddingCost({1,0},sibling,64,64,"linear",true),std::invalid_argument);
    CHECK_THROWS_AS(advanceFiberEmbeddingHistory(h,{1,0},100,16,64,2),std::invalid_argument);
    CHECK_THROWS_AS(fiberEmbeddingCosine({NAN},{1}),std::invalid_argument);
}

TEST_CASE("embedding costs affect beam decisions equally in serial and parallel scoring") {
    ForkPrediction source;
    auto c=config();
    const auto baseline=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
    CHECK(source.embeddingCalls==0);
    REQUIRE(baseline.points.size()>1);
    CHECK(baseline.points.back()[1]==0);
    c.embeddingWeight=2;
    const auto serial=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
    CHECK(source.embeddingCalls>0);
    REQUIRE(serial.points.size()>1);
    CHECK(serial.points.back()[1]>.5);
    c.parallelThreads=4;
    const auto parallel=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
    CHECK(serial.points==parallel.points);
    c.lazyLookahead=false;
    const auto eager=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
    CHECK(serial.points==eager.points);
    source.constant=true;
    const auto neutral=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
    CHECK(neutral.points==baseline.points);
}

TEST_CASE("missing embeddings reject candidates and malformed batches fail explicitly") {
    ForkPrediction source;
    auto c=config(); c.embeddingWeight=1;
    source.missing=true;
    const auto result=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
    CHECK(result.points.size()==1);
    source.wrongSize=true;
    CHECK_THROWS_AS(traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c),std::runtime_error);
    source.wrongSize=false;
    c.embeddingHistoryCount=0;
    CHECK_THROWS_AS(traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c),std::invalid_argument);
}

