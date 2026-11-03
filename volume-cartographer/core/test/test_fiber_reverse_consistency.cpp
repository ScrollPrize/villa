#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayCore.hpp"
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

TEST_CASE("cached real-data forward trace can be checked backward without ground truth") {
    const char* manifest=std::getenv("VC_TEST_WIDTH_MANIFEST");
    const char* fiberPath=std::getenv("VC_TEST_WIDTH_FIBER");
    const char* normalManifest=std::getenv("VC_TEST_REVERSE_NORMAL_MANIFEST");
    if (!manifest || !fiberPath || !normalManifest) return;
    const auto dataset=vc::lasagna::LasagnaDataset::open(manifest);
    const auto normalData=vc::lasagna::LasagnaDataset::open(normalManifest);
    FiberPredictionFieldOptions options; options.enableWidth=true;
    FiberPredictionField field(dataset,options);
    vc::lasagna::LasagnaNormalSampler normals(normalData);
    const auto fiber=loadFiberJson(fiberPath);
    REQUIRE(fiber.controlPointLineIndices.size()>1);
    const size_t first=fiber.controlPointLineIndices.front();
    REQUIRE(first+1<fiber.linePointsXyzBase.size());
    auto c=config(); c.stepVoxels=16; c.widthRelativeTolerance=.4; c.widthCosineWeight=true;
    const auto forward=traceFiberExtrapolation(field,fiber.linePointsXyzBase[first],
        fiber.linePointsXyzBase[first+1]-fiber.linePointsXyzBase[first],64,c,&normals);
    REQUIRE(forward.points.size()>1);
    const auto geometry=makePolylineArcGeometry(forward.points);
    c.widthReferenceMean=forward.widthMean;
    const double serial=verifyFiberReverseSpan(field,normals,c,geometry,geometry.length(),20,40);
    c.parallelThreads=4;
    const double parallel=verifyFiberReverseSpan(field,normals,c,geometry,geometry.length(),20,40);
    CHECK(serial==parallel);
    CHECK(serial>=0); CHECK(serial<=geometry.length());
    std::cout << "Real-data reverse check: forward=" << geometry.length()
        << " base voxels; remaining=" << serial << "; verified=" << geometry.length()-serial
        << "; serial/parallel exact\n";
}

TEST_CASE("future history enters checkpoint from the later forward side in reverse order") {
    const auto path=makePolylineArcGeometry({{0,0,0},{4,0,0},{8,0,0},{8,4,0},{12,4,0}});
    const auto history=fiberReverseFutureHistory(path,1,3);
    REQUIRE(history.size()==3);
    CHECK(history[0]==cv::Vec3d(-1,0,0));
    CHECK(history[1]==cv::Vec3d(0,-1,0));
    CHECK(history[2]==cv::Vec3d(-1,0,0));
    CHECK(fiberReverseFutureHistory(path,3,4).size()==1);
    CHECK(fiberReverseFutureHistory(path,4,4).empty());
    Prediction prediction;Normals normals;auto c=config();
    ReverseSpanOptions options;options.initialDirections=history;
    const auto result=verifyFiberReverseSpanDetailed(prediction,normals,c,path,4,2,8,0,options);
    REQUIRE(prediction.first.has_value());CHECK(*prediction.first==cv::Vec3d(4,0,0));
    CHECK(result.passed);CHECK(result.remainingArc==0);
}

TEST_CASE("convergence requires minimum travel sustained tight angle and previously verified prefix") {
    const auto line=makePolylineArcGeometry({{0,0,0},{1000,0,0}});Normals normals;
    ReverseSpanOptions o;o.minimumTravel=100;o.verifiedThroughArc=900;o.convergenceSteps=3;
    ReverseSpanVerifier v(line,1000,{1000,0,0},normals,1,20,40,o);
    CHECK(v.observe({950,0,0}));CHECK(v.observe({900,0,0}));
    CHECK(v.observe({850,0,0}));CHECK_FALSE(v.observe({800,0,0}));
    CHECK(v.result().passed);CHECK(v.result().converged);CHECK(v.remainingArc()==800);
    CHECK(v.result().travelled==200);
    o.verifiedThroughArc=0;
    ReverseSpanVerifier unverified(line,1000,{1000,0,0},normals,1,20,40,o);
    CHECK(unverified.observe({500,0,0}));CHECK_FALSE(unverified.result().passed);
    o.verifiedThroughArc=900;o.convergenceSteps=1;
    ReverseSpanVerifier crossing(line,1000,{1000,0,0},normals,1,20,40,o);
    CHECK_FALSE(crossing.observe({900,0,0})); // exact aligned case succeeds
    ReverseSpanVerifier angled(line,1000,{1000,0,0},normals,1,20,40,o);
    CHECK(angled.observe({950,10,0}));
    CHECK(angled.observe({900,0,0})); // on line, but >5degree direction mismatch
    CHECK_FALSE(angled.result().passed);
    CHECK_FALSE(angled.observe({850,0,0}));
    ReverseSpanVerifier distant(line,1000,{1000,0,0},normals,1,20,40,o);
    CHECK(distant.observe({900,5,0})); // broad ellipse passes, tight convergence fails
    CHECK_FALSE(distant.result().passed);
}

TEST_CASE("delayed replay certifies checkpoint only and flushes partial history at endpoint") {
    Normals normals;
    const auto reference=makePolylineArcGeometry({{0,0,0},{36,0,0}});
    ReplayConsistencyPolicy policy;policy.intervalBase=24;policy.delaySteps=4;
    policy.toStart=true;policy.allEndpoints=true;
    policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return std::nullopt;};
    std::vector<std::array<size_t,4>> checks;
    policy.verifyCheckpoint=[&](const auto& points,size_t target,size_t checked,bool terminal) {
        checks.push_back({points.size(),target,checked,terminal?1u:0u});
        return ReplayReverseCheck{true,4.*target,false,points.size()-1-target,0};
    };
    ReplaySegmentRunner run=[](const auto& seed,double,int,const auto& observe,const auto&) {
        for(int i=1;i<=9;++i) if(!observe(seed.point+cv::Vec3d(4*i,0,0),0,i)) return std::string("observer_stop");
        return std::string("budget");
    };
    const auto result=traceReferenceReplayCore(reference,0,36,4,3,20,0,2,1,normals,1,run,{},{},policy);
    REQUIRE(checks.size()==2);
    CHECK(checks[0]==std::array<size_t,4>{10,6,0,0}); // pending check has only3future edges
    CHECK(checks[1]==std::array<size_t,4>{10,9,6,1}); // endpoint explicitly checked, no future
    CHECK(result.reverseChecks==2);CHECK(result.failures.empty());
    REQUIRE(result.segments.size()==1);CHECK(result.segments[0].tracePointsBase.size()==10);
    policy.delaySteps=6;
    CHECK_THROWS_AS(traceReferenceReplayCore(reference,0,36,4,3,20,0,2,1,normals,1,run,{},{},policy),std::invalid_argument);
}

TEST_CASE("delayed checkpoint cadence and failure rollback exclude unchecked future suffix") {
    Normals normals;
    const auto reference=makePolylineArcGeometry({{0,0,0},{80,0,0}});
    ReplayConsistencyPolicy policy;policy.intervalBase=16;policy.delaySteps=2;
    policy.toStart=true;policy.allEndpoints=true;
    policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return std::nullopt;};
    int calls=0;
    policy.verifyCheckpoint=[&](const auto& points,size_t target,size_t checked,bool terminal) {
        ++calls;CHECK_FALSE(terminal);
        CHECK(target==size_t(calls*4));CHECK(points.size()==target+3);
        CHECK(checked==size_t((calls-1)*4));
        return ReplayReverseCheck{calls==1,4.*target,false,2,0};
    };
    ReplaySegmentRunner run=[](const auto& seed,double,int,const auto& observe,const auto&) {
        for(int i=1;i<=20;++i)if(!observe(seed.point+cv::Vec3d(4*i,0,0),0,i))return std::string("observer_stop");
        return std::string("budget");
    };
    const auto result=traceReferenceReplayCore(reference,0,80,4,3,20,0,2,1,normals,1,run,{},{},policy);
    CHECK(calls==2);REQUIRE(result.segments.size()==1);
    CHECK(result.segments[0].tracePointsBase.size()==5);
    CHECK(result.segments[0].endReferenceArcBase==16);
    REQUIRE(result.filteredStops.size()==1);CHECK(result.filteredStops[0][0]==16);
    CHECK(result.filteredStops[0][1]==40);CHECK(result.failures.empty());
}

TEST_CASE("presence and width stops verify from four accepted edges back and keep the history tail") {
    Normals normals;const auto reference=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    for(const std::string reason : {"presence_cutoff","width_cutoff","invalid_width"}) {
      for(int stopEdges : {12,15}) {
        ReplayConsistencyPolicy policy;policy.intervalBase=32;policy.delaySteps=4;policy.toStart=true;
        policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return std::nullopt;};
        int calls=0;
        policy.verifyCheckpoint=[&](const auto& points,size_t target,size_t checked,bool terminal) {
            ++calls;CHECK_FALSE(terminal);
            if(calls==1){CHECK(target==8);CHECK(checked==0);CHECK(points.size()==13);}
            else {CHECK(target==size_t(stopEdges-4));CHECK(checked==8);CHECK(points.size()==size_t(stopEdges+1));}
            return ReplayReverseCheck{true,4.*target,false,4,0};
        };
        ReplaySegmentRunner run=[&](const auto& seed,double,int,const auto& observe,const auto&) {
            for(int i=1;i<=stopEdges;++i)if(!observe(seed.point+cv::Vec3d(4*i,0,0),0,i))return std::string("observer_stop");
            return reason;
        };
        const auto result=traceReferenceReplayCore(reference,0,100,4,3,20,0,2,1,normals,1,run,{},{},policy);
        CHECK(calls==2);CHECK(result.reverseStopTailChecks==1);CHECK(result.reverseStopTailPassed==1);
        REQUIRE(result.segments.size()==1);CHECK(result.segments[0].tracePointsBase.size()==size_t(stopEdges+1));
        CHECK(result.segments[0].matches.size()==size_t(stopEdges));CHECK(result.segments[0].endReferenceArcBase==4*stopEdges);
        CHECK(result.segments[0].terminationReason=="presence_filtered");
        REQUIRE(result.filteredStops.size()==1);CHECK(result.filteredStops[0][0]==4*stopEdges);
      }
    }
}

TEST_CASE("stop-tail failure rolls back and observer rejected point never enters reverse history") {
    Normals normals;const auto reference=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    for(bool pass : {false,true}) {
        ReplayConsistencyPolicy policy;policy.intervalBase=32;policy.delaySteps=4;policy.toStart=true;
        policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return std::nullopt;};
        policy.acceptPoint=[](const auto& point,const auto&){return point[0]<64;};
        int calls=0;
        policy.verifyCheckpoint=[&](const auto& points,size_t target,size_t checked,bool) {
            ++calls;
            if(calls==2) {
                CHECK(points.size()==16);CHECK(points.back()[0]==60); // never rejected64
                CHECK(target==11);CHECK(checked==8);
            }
            return ReplayReverseCheck{calls==1 || pass,4.*target,false,4,0};
        };
        ReplaySegmentRunner run=[](const auto& seed,double,int,const auto& observe,const auto&) {
            for(int i=1;i<=16;++i)if(!observe(seed.point+cv::Vec3d(4*i,0,0),0,i))return std::string("observer_stop");
            return std::string("budget");
        };
        const auto result=traceReferenceReplayCore(reference,0,100,4,3,20,0,2,1,normals,1,run,{},{},policy);
        CHECK(calls==2);REQUIRE(result.segments.size()==1);
        const auto& segment=result.segments[0];
        CHECK(segment.tracePointsBase.size()==(pass ? 16:9));
        CHECK(segment.cumulativeLosses.size()==segment.tracePointsBase.size());
        CHECK(segment.matches.size()+1==segment.tracePointsBase.size());
        CHECK(segment.terminationReason==(pass ? "presence_filtered":"reverse_consistency_filtered"));
        CHECK(result.reverseStopTailPassed==(pass ? 1:0));
        REQUIRE(result.filteredStops.size()==1);CHECK(result.filteredStops[0][1]==64); // original detection retained
        CHECK(result.filteredStops[0][0]==(pass ? 60:32));
    }
}

TEST_CASE("short stop tail is retained without pretending a reverse trace happened") {
    Normals normals;const auto reference=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    ReplayConsistencyPolicy policy;policy.intervalBase=32;policy.delaySteps=4;policy.toStart=true;
    policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return std::nullopt;};
    policy.verifyCheckpoint=[](const auto&,size_t,size_t,bool) {
        FAIL("no preceding span to reverse-check");return ReplayReverseCheck{};
    };
    ReplaySegmentRunner run=[](const auto& seed,double,int,const auto& observe,const auto&) {
        for(int i=1;i<=3;++i)if(!observe(seed.point+cv::Vec3d(4*i,0,0),0,i))return std::string("observer_stop");
        return std::string("width_cutoff");
    };
    const auto result=traceReferenceReplayCore(reference,0,100,4,3,20,0,2,1,normals,1,run,{},{},policy);
    CHECK(result.reverseChecks==0);CHECK(result.reverseStopTailChecks==0);
    CHECK(result.reverseStopTailPassed==0);CHECK(result.reverseStopTailHistoryOnly==1);
    REQUIRE(result.segments.size()==1);CHECK(result.segments[0].tracePointsBase.size()==4);
}
