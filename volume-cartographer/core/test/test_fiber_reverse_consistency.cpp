#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
#include <cstdlib>
#include <iostream>

using namespace vc::fiber_tracer;
namespace {
class Normals final : public vc::lasagna::NormalSampler {
public:
    bool valid = true;
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override {
        return {{0,0,1},valid};
    }
};
class Prediction final : public FiberPredictionSource {
public:
    bool missing = false, error = false;
    double end = 1000;
    float presence = 1;
    mutable std::optional<cv::Vec3d> first;
    bool supportsConcurrentSampling() const noexcept override { return true; }
    FiberPredictionSample sample(const cv::Vec3d& p, const cv::Vec3d&) const override {
        // This fixture records the seed before parallel candidate sampling.
        if (!first) first=p;
        if (error) throw std::runtime_error("sampling failed");
        FiberPredictionSample out;
        if (!missing && p[0]<end) out.options.push_back({{1,0,0},presence,true});
        return out;
    }
    void sampleWidths(const std::vector<cv::Vec3f>& points,std::vector<float>& widths) const override {
        widths.clear(); for (const auto& p : points) widths.push_back(10+p[0]*.01f);
    }
};
FiberTraceConfig config() {
    FiberTraceConfig c; c.stepVoxels=4; c.beamWidth=3;
    c.smoothnessNormalWeight=c.smoothnessTangentWeight=c.cumulativeSmoothnessTangentWeight=0;
    c.parallelThreads=1;
    return c;
}
}

TEST_CASE("reverse verifier uses anisotropic deviation and monotonic backward matching") {
    const auto line=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    Normals normals;
    ReverseSpanVerifier verifier(line,100,{100,0,0},normals,1,2,8);
    CHECK(verifier.observe({100,0,0})); // duplicate, not a failed advance
    CHECK(verifier.observe({90,4,0}));
    CHECK(verifier.remainingArc()==90);
    CHECK_FALSE(verifier.observe({80,0,3}));
    CHECK(verifier.remainingArc()==90); // failing point was not accepted
    ReverseSpanVerifier backward(line,100,{100,0,0},normals,1,2,8);
    CHECK(backward.observe({90,0,0}));
    CHECK_FALSE(backward.observe({95,0,0})); // cannot move forward again
    CHECK(backward.remainingArc()==90);
    ReverseSpanVerifier complete(line,100,{100,0,0},normals,1,2,8);
    CHECK_FALSE(complete.observe({0,0,0}));
    CHECK(complete.remainingArc()==0);
    normals.valid=false;
    ReverseSpanVerifier fallback(line,100,{100,0,0},normals,1,2,8);
    CHECK_FALSE(fallback.observe({90,4,0})); // Euclidean fallback, radius 2
}

TEST_CASE("committed callbacks match the returned path and can stop within lookahead") {
    Prediction prediction;
    auto c=config(); c.widthRelativeTolerance=.4;
    for (int depth : {1,2,3}) {
        c.beamLookaheadSteps=depth;
        const auto baseline=traceFiberExtrapolation(prediction,{0,0,0},{1,0,0},32,c);
        std::vector<cv::Vec3d> observed;
        double previousLoss=0;
        const auto traced=traceFiberExtrapolation(prediction,{0,0,0},{1,0,0},32,c,nullptr,{},false,
            [&](const cv::Vec3d& p,double loss,int step) {
                observed.push_back(p); CHECK(step==static_cast<int>(observed.size()));
                CHECK(loss>=previousLoss); previousLoss=loss; return true;
            });
        CHECK(traced.points==baseline.points);
        CHECK(traced.widthMean==baseline.widthMean);
        CHECK(observed==std::vector<cv::Vec3d>(traced.points.begin()+1,traced.points.end()));
        observed.clear();
        const auto stopped=traceFiberExtrapolation(prediction,{0,0,0},{1,0,0},32,c,nullptr,{},false,
            [&](const cv::Vec3d& p,double,int step) { observed.push_back(p); return step<3; });
        REQUIRE(stopped.points.size()==4);
        CHECK(stopped.reason=="observer_stop");
        CHECK(stopped.steps==3);
        CHECK(stopped.widthSamples==4);
        double sum=0; for (const auto& p : stopped.points) sum+=10+static_cast<float>(p[0])*.01f;
        CHECK(stopped.widthMean==doctest::Approx(sum/4));
        CHECK(observed==std::vector<cv::Vec3d>(stopped.points.begin()+1,stopped.points.end()));
    }
}

TEST_CASE("committed callbacks include exhausted fallback and propagate exceptions") {
    Prediction prediction; prediction.end=10;
    auto c=config();
    std::vector<cv::Vec3d> observed;
    const auto result=traceFiberExtrapolation(prediction,{0,0,0},{1,0,0},32,c,nullptr,{},false,
        [&](const cv::Vec3d& p,double,int) { observed.push_back(p); return true; });
    CHECK(result.reason=="no_valid_candidates");
    CHECK(observed==std::vector<cv::Vec3d>(result.points.begin()+1,result.points.end()));
    CHECK_THROWS_AS(traceFiberExtrapolation(prediction,{0,0,0},{1,0,0},32,c,nullptr,{},false,
        [](const cv::Vec3d&,double,int)->bool { throw std::runtime_error("observer failed"); }),std::runtime_error);
    CHECK_THROWS_AS(traceFiberExtrapolation(prediction,{0,0,0},{1,0,0},32,c,nullptr,{},false,{},{{0,0,0}}),std::invalid_argument);
}

TEST_CASE("native reverse tracing verifies forward paths and optional vertex history") {
    Prediction prediction;
    Normals normals;
    auto c=config(); c.widthRelativeTolerance=.4;
    const auto forward=traceFiberExtrapolation(prediction,{0,0,0},{1,0,0},100,c);
    const auto original=forward.points;
    const auto geometry=makePolylineArcGeometry(forward.points);
    c.widthReferenceMean=forward.widthMean;
    CHECK(verifyFiberReverseSpan(prediction,normals,c,geometry,geometry.length(),2,8)==doctest::Approx(0));
    CHECK(forward.points==original);
    Prediction recorded;
    const auto vertices=makePolylineArcGeometry({{0,0,0},{40,0,0},{80,0,0},{100,0,0}});
    CHECK(verifyFiberReverseSpan(recorded,normals,c,vertices,50,2,8,4)==doctest::Approx(0));
    REQUIRE(recorded.first.has_value()); CHECK((*recorded.first)[0]==80);
    c.traceToBaseScale=2;
    Prediction scaled;
    CHECK(verifyFiberReverseSpan(scaled,normals,c,vertices,100,2,8)==doctest::Approx(0));
    REQUIRE(scaled.first.has_value()); CHECK((*scaled.first)[0]==50);
    c.traceToBaseScale=1;
    const auto bent=makePolylineArcGeometry({{0,0,0},{50,0,0},{50,50,0}});
    CHECK(verifyFiberReverseSpan(prediction,normals,c,bent,100,2,8)>0);
}

TEST_CASE("reverse geometry checks do not inherit the forward presence cutoff") {
    Prediction prediction;prediction.presence=.2f;Normals normals;
    auto c=config();c.replayPresenceCutoff=.4;c.replayPresenceModulated=true;
    c.widthRelativeTolerance=.4;c.widthReferenceMean=10.5;
    const auto line=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    CHECK(verifyFiberReverseSpan(prediction,normals,c,line,100,2,8)==doctest::Approx(0));
    CHECK(c.replayPresenceCutoff==.4);
}

TEST_CASE("reverse checks fail missing seeds but preserve sampling and configuration errors") {
    Prediction prediction; prediction.missing=true;
    Normals normals;
    auto c=config();
    const auto line=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    CHECK(verifyFiberReverseSpan(prediction,normals,c,line,100,2,8)==100);
    prediction.error=true;
    CHECK_THROWS_AS(verifyFiberReverseSpan(prediction,normals,c,line,100,2,8),std::runtime_error);
    prediction.error=false;
    CHECK_THROWS_AS(verifyFiberReverseSpan(prediction,normals,c,line,100,0,8),std::invalid_argument);
    CHECK_THROWS_AS(verifyFiberReverseSpan(prediction,normals,c,line,101,2,8),std::invalid_argument);
    CHECK_THROWS_AS(verifyFiberReverseSpan(prediction,normals,c,line,100,2,8,65),std::invalid_argument);
    c.traceToBaseScale=0;
    CHECK_THROWS_AS(verifyFiberReverseSpan(prediction,normals,c,line,100,2,8),std::invalid_argument);
}

