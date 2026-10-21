#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/FiberWidth.hpp"
#include <cstdlib>
#include <iostream>

using namespace vc::fiber_tracer;
namespace {
class WidthPrediction final : public FiberPredictionSource {
public:
    mutable size_t sampled = 0;
    bool fork = false, malformed = false, invalidSeed = false;
    double end = 1000;
    bool supportsConcurrentSampling() const noexcept override { return true; }
    FiberPredictionSample sample(const cv::Vec3d& p, const cv::Vec3d&) const override {
        FiberPredictionSample out;
        if (p[0] < end) out.options.push_back({{1,0,0}, p[1] > .5 ? .9f : 1.f, true});
        return out;
    }
    float width(const cv::Vec3f& p) const {
        if (invalidSeed && p[0] == 0) return 0;
        return fork ? (p[0] == 0 || p[1] > .5 ? 10.f : 20.f) : 10 + p[0]*.01f;
    }
    void sampleWidths(const std::vector<cv::Vec3f>& points, std::vector<float>& out) const override {
        out.clear(); sampled += points.size();
        if (malformed) return;
        for (const auto& p : points) out.push_back(width(p));
    }
};
FiberTraceConfig config() {
    FiberTraceConfig c;
    c.smoothnessWeight = c.smoothnessNormalWeight = c.smoothnessTangentWeight = 0;
    c.cumulativeSmoothnessTangentWeight = 0;
    c.stepVoxels = 4; c.beamWidth = 3; c.parallelThreads = 1;
    return c;
}
void checkMean(const WidthPrediction& source, const FiberTraceOneWayResult& result, double initial = 0) {
    REQUIRE(!result.points.empty());
    double total = 0;
    for (size_t i=0; i<result.points.size(); ++i)
        total += i == 0 && initial > 0 ? initial : source.width(cv::Vec3f(result.points[i]));
    CHECK(result.widthSamples == result.points.size());
    CHECK(result.widthMean == doctest::Approx(total/result.points.size()));
}
}

TEST_CASE("width gates preserve the source linear and cosine formulas") {
    FiberWidthMean mean;
    mean.add(10); mean.add(12); mean.add(14);
    CHECK(mean.mean() == 12);
    CHECK(mean.count == 3);
    for (double sign : {-1., 1.}) {
        CHECK(fiberWidthPresenceScale(10+sign,10,.2) == doctest::Approx(.5));
        CHECK(fiberWidthPresenceScale(10+sign,10,.2,true) == doctest::Approx(std::sqrt(.5)));
        CHECK(fiberWidthPresenceScale(10+3*sign,10,.2,true) == 0);
    }
    CHECK(fiberWidthPresenceScale(10,10,.2) == 1);
    CHECK(fiberWidthPresenceScale(0,10,.2) == 0);
    CHECK(fiberWidthPresenceScale(NAN,10,.2) == 0);
    CHECK(fiberWidthPresenceScale(100,0,0) == 1);
    CHECK_THROWS(fiberWidthPresenceScale(10,0,.2));
    CHECK_THROWS(mean.add(0));
}

TEST_CASE("only committed widths enter the mean including intermediate lookahead points") {
    WidthPrediction source;
    auto c=config();
    const auto legacy=traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c);
    CHECK(source.sampled==0);
    for (int depth : {1,2,3}) {
        c.beamLookaheadSteps=depth; c.widthRelativeTolerance=.4;
        const auto result=traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c);
        checkMean(source,result);
        CHECK(source.sampled>result.widthSamples);
        c.parallelThreads=4;
        const auto parallel=traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c);
        CHECK(parallel.points==result.points);
        CHECK(parallel.widthMean==result.widthMean);
        c.parallelThreads=1;
    }
    c.initialWidthMean=11;
    const auto fitted=traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c);
    checkMean(source,fitted,11);
    c.widthReferenceMean=fitted.widthMean;
    const auto fixed=traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c);
    CHECK(fixed.widthSamples==0);
    CHECK(fixed.widthMean==c.widthReferenceMean);
    c.widthRelativeTolerance=0;
    CHECK(traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c).points==legacy.points);
}

TEST_CASE("width gating changes fork selection and eager lazy search agrees") {
    WidthPrediction source; source.fork=true;
    auto c=config();
    const auto baseline=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
    CHECK(baseline.points.back()[1]==0);
    c.widthRelativeTolerance=.4;
    for (bool cosine : {false,true}) {
        c.widthCosineWeight=cosine;
        c.lazyLookahead=true;
        const auto result=traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c);
        CHECK(result.points.back()[1]>.5);
        checkMean(source,result);
        c.lazyLookahead=false;
        CHECK(traceFiberExtrapolation(source,{0,0,0},{1,0,0},16,c).points==result.points);
    }
}

TEST_CASE("target exit and exhausted frontier retain only final path widths") {
    WidthPrediction source;
    auto c=config(); c.widthRelativeTolerance=.4;
    source.end=10;
    const auto exhausted=traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c);
    CHECK(exhausted.reason=="no_valid_candidates");
    checkMean(source,exhausted);
    source.end=1000;
    FiberTraceOneWayRequest request;
    request.startPoint={0,0,0}; request.targetPoint={8,0,0}; request.initialDirection={1,0,0};
    request.budgetSpanVoxels=8; request.config=c;
    request.snapTraceToSelectedCrossing=false;
    request.targetPlanes.push_back({"target",{8,0,0},{1,0,0}});
    const auto target=traceFiberOneWay(source,request);
    CHECK(target.reachedTargetPlane);
    checkMean(source,target);
}

TEST_CASE("width input errors fail explicitly") {
    WidthPrediction source;
    auto c=config(); c.widthRelativeTolerance=.4;
    source.malformed=true;
    CHECK_THROWS(traceFiberExtrapolation(source,{0,0,0},{1,0,0},8,c));
    c.widthReferenceMean=10; // bypass seed sampling, fail the candidate batch instead
    CHECK_THROWS(traceFiberExtrapolation(source,{0,0,0},{1,0,0},8,c));
    c.widthReferenceMean=0;
    source.malformed=false; source.invalidSeed=true;
    CHECK_THROWS(traceFiberExtrapolation(source,{0,0,0},{1,0,0},8,c));
    c.widthRelativeTolerance=-1;
    CHECK_THROWS_AS(traceFiberExtrapolation(source,{0,0,0},{1,0,0},8,c),std::invalid_argument);
}

TEST_CASE("real cached width predictions have reproducible committed means") {
    const char* manifest=std::getenv("VC_TEST_WIDTH_MANIFEST");
    const char* fiberPath=std::getenv("VC_TEST_WIDTH_FIBER");
    if (!manifest || !fiberPath) return;
    const auto dataset=vc::lasagna::LasagnaDataset::open(manifest);
    FiberPredictionFieldOptions options; options.enableWidth=true;
    FiberPredictionField field(dataset,options);
    const auto fiber=loadFiberJson(fiberPath);
    REQUIRE(fiber.controlPointLineIndices.size()>1);
    const size_t first=fiber.controlPointLineIndices.front();
    REQUIRE(first+1<fiber.linePointsXyzBase.size());
    const auto start=fiber.linePointsXyzBase[first];
    const auto direction=fiber.linePointsXyzBase[first+1]-start;
    auto c=config(); c.stepVoxels=16;
    const auto baseline=traceFiberExtrapolation(field,start,direction,64,c);
    c.widthRelativeTolerance=.4; c.widthCosineWeight=true;
    const auto serial=traceFiberExtrapolation(field,start,direction,64,c);
    c.parallelThreads=4;
    const auto parallel=traceFiberExtrapolation(field,start,direction,64,c);
    REQUIRE(serial.points.size()>1);
    CHECK(serial.points==parallel.points);
    CHECK(serial.widthMean==parallel.widthMean);
    std::vector<cv::Vec3f> points;
    for (const auto& p : serial.points) points.push_back(cv::Vec3f(p));
    std::vector<float> widths; field.sampleWidths(points,widths);
    double sum=0; for (float w : widths) sum+=w;
    CHECK(serial.widthSamples==widths.size());
    CHECK(serial.widthMean==doctest::Approx(sum/widths.size()));
    std::cout << "Real width trace: start=" << start[0] << ',' << start[1] << ',' << start[2]
        << "; baseline=" << baseline.points.size() << " points; gated="
        << serial.points.size() << "; committed mean=" << serial.widthMean
        << "; serial/parallel exact\n";
}
