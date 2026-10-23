#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberReferenceReplayCore.hpp"
#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>

namespace {
class ReplayTestNormals final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override {
        return {{0,1,0},true,{}};
    }
};
}

TEST_CASE("distance failures retain diagnostics and make deterministic forward resets") {
    using namespace vc::fiber_tracer;
    ReplayTestNormals normals;
    const auto reference=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    std::vector<double> starts;
    size_t failures=0, progress=0;
    const ReplaySegmentRunner runner=[&](const PolylineArcSample& seed,double,int maximumSteps,
            const ReplayPointObserver& observe,const FiberTraceProgressCallback& report) {
        starts.push_back(seed.arc);
        CHECK(maximumSteps>0);
        (void)observe({seed.arc+10,3,0},7,1);
        if (report) report({});
        return std::string("observer_stop");
    };
    const auto result=traceReferenceReplayCore(reference,0,100,10,3,2,0,4,1,normals,1,runner,
        [&](const FiberTraceProgress& event) {
            REQUIRE(event.replaySegmentIndex.has_value());
            CHECK(*event.replaySegmentIndex==progress++);
            CHECK(event.referenceArcBase.has_value());
            CHECK(event.referenceArcFraction.has_value());
        }, [&](const FiberReplayFailure& failure) {
            CHECK(failure.index==failures++);
            CHECK(failure.reason=="distance_above_threshold");
            REQUIRE(failure.thresholdMeasurement.has_value());
            CHECK(failure.thresholdMeasurement->thresholdErrorRatio==doctest::Approx(1.5));
        });
    CHECK(result.completedReferenceArcBase==100);
    CHECK(result.failures.size()==10);
    CHECK(failures==10); CHECK(progress==10);
    for(size_t i=0;i<starts.size();++i) CHECK(starts[i]==10*i);
    CHECK(result.filteredStops.empty());
}

TEST_CASE("unfinished outside excursions fail at the terminal endpoint even within grace") {
    using namespace vc::fiber_tracer;
    ReplayTestNormals normals;
    const auto reference=makePolylineArcGeometry({{0,0,0},{40,0,0}});
    const ReplaySegmentRunner runner=[](const PolylineArcSample&,double,int,
            const ReplayPointObserver& observe,const FiberTraceProgressCallback&) {
        CHECK(observe({10,3,0},0,1));
        CHECK(observe({20,0,0},0,2)); // reset the excursion
        CHECK(observe({30,3,0},0,3));
        CHECK_FALSE(observe({40,3,0},0,4));
        return std::string("observer_stop");
    };
    const auto result=traceReferenceReplayCore(reference,0,40,10,3,2,100,4,1,normals,1,runner);
    REQUIRE(result.failures.size()==1);
    CHECK(result.failures[0].referenceArcBase==40);
    CHECK(result.failures[0].reason=="distance_above_threshold");
}

TEST_CASE("replay rejects invalid intervals and overflowing budgets before calling a backend") {
    using namespace vc::fiber_tracer;
    ReplayTestNormals normals;
    const auto reference=makePolylineArcGeometry({{0,0,0},{40,0,0}});
    const ReplaySegmentRunner runner=[](const auto&,double,int,const auto&,const auto&) {
        FAIL("invalid inputs must not reach the backend"); return std::string{};
    };
    for(double step : {0.,-1.,std::numeric_limits<double>::infinity(),1e-300})
        CHECK_THROWS_AS(traceReferenceReplayCore(reference,0,40,step,3,2,0,4,1,normals,1,runner),std::invalid_argument);
    CHECK_THROWS_AS(traceReferenceReplayCore(reference,0,41,10,3,2,0,4,1,normals,1,runner),std::invalid_argument);
    CHECK_THROWS_AS(traceReferenceReplayCore(reference,0,40,10,3,2,0,4,1,normals,0,runner),std::invalid_argument);
    CHECK_THROWS_AS(traceReferenceReplayCore(reference,0,40,10,3,2,0,4,1,normals,1,{}),std::invalid_argument);
}

TEST_CASE("real Paris4 native tracing plugs into shared reference replay with reverse checks") {
    using namespace vc::fiber_tracer;
    const char* manifest=std::getenv("VC_TEST_WIDTH_MANIFEST");
    const char* fiberPath=std::getenv("VC_TEST_WIDTH_FIBER");
    const char* normalManifest=std::getenv("VC_TEST_REVERSE_NORMAL_MANIFEST");
    if (!manifest || !fiberPath || !normalManifest) return;
    const auto dataset=vc::lasagna::LasagnaDataset::open(manifest);
    const auto normalData=vc::lasagna::LasagnaDataset::open(normalManifest);
    FiberPredictionField field(dataset);
    vc::lasagna::LasagnaNormalSampler normals(normalData);
    const auto fiber=loadFiberJson(fiberPath);
    REQUIRE(fiber.controlPointLineIndices.size()>1);
    const auto full=makePolylineArcGeometry(fiber.linePointsXyzBase);
    const double begin=full.vertexArcs.at(fiber.controlPointLineIndices.front());
    const auto reference=makePolylineArcGeometry(slicePolylineArc(full,begin,begin+64));
    FiberTraceConfig c; c.stepVoxels=16; c.beamWidth=3; c.parallelThreads=1;
    c.smoothnessNormalWeight=c.smoothnessTangentWeight=c.cumulativeSmoothnessTangentWeight=0;
    const ReplaySegmentRunner runner=[&](const PolylineArcSample& seed,double,int maximumSteps,
            const ReplayPointObserver& observe,const FiberTraceProgressCallback& progress) {
        return traceFiberExtrapolation(field,seed.point,seed.tangent,maximumSteps*c.stepVoxels,
            c,&normals,progress,false,observe).reason;
    };
    ReplayConsistencyPolicy policy;
    policy.intervalBase=32;
    policy.verify=[&](const auto& points) {
        const auto geometry=makePolylineArcGeometry(points);
        return verifyFiberReverseSpan(field,normals,c,geometry,geometry.length(),20,40)<=1e-12;
    };
    policy.nextSeed=[&](double arc)->std::optional<PolylineArcSample> {
        return samplePolylineArc(reference,arc);
    };
    const auto run=[&](const ReplayConsistencyPolicy& p) {
        return traceReferenceReplayCore(reference,0,reference.length(),16,3,20,0,2,1,normals,1,runner,{},{},p);
    };
    const auto baseline=run({});
    const auto serial=run(policy);
    c.parallelThreads=4;
    const auto parallel=run(policy);
    CHECK(serial.completedReferenceArcBase==reference.length());
    REQUIRE(!serial.segments.empty());
    REQUIRE(parallel.segments.size()==serial.segments.size());
    CHECK(serial.failures.size()==parallel.failures.size());
    CHECK(serial.filteredStops==parallel.filteredStops);
    CHECK(serial.reverseChecks==parallel.reverseChecks);
    for(size_t i=0;i<serial.segments.size();++i) {
        const auto& segment=serial.segments[i];
        CHECK(segment.tracePointsBase==parallel.segments[i].tracePointsBase);
        for(const auto& match:segment.matches) {
            const auto measured=measureFiberReplayThreshold(segment.tracePointsBase.at(match.tracePointIndex),
                match.matchedReferencePointBase,normals,1,20,2);
            CHECK(measured.thresholdErrorRatio==match.thresholdMeasurement.thresholdErrorRatio);
        }
    }
    std::cout << "Real Paris4 replay: " << reference.length() << " base voxels; baseline failures="
        << baseline.failures.size() << "; filtered failures=" << serial.failures.size()
        << "; filtered stops=" << serial.filteredStops.size() << "; reverse checks="
        << serial.reverseChecks << "; serial/parallel exact\n";
}

TEST_CASE("Periodic consistency checks final partial span and accounts rejected tail")
{
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override {
            return {{0, 1, 0}, true, {}};
        }
    } normals;
    const auto reference = makePolylineArcGeometry({{0,0,0},{250,0,0}});
    ReplaySegmentRunner runner = [](const PolylineArcSample& seed, double, int,
            const ReplayPointObserver& observer, const FiberTraceProgressCallback&) {
        for (double x = seed.arc + 10; x <= 250; x += 10)
            if (!observer({x,0,0}, 0, int(x/10))) break;
        return std::string("stopped");
    };
    ReplayConsistencyPolicy policy;
    policy.intervalBase = 100;
    policy.verify = [](const auto&) { return true; };
    policy.nextSeed = [](double) -> std::optional<PolylineArcSample> { return {}; };
    const auto run = [&] {
        return traceReferenceReplayCore(reference, 0, 250, 10, 3, 40, 160, 2, 1,
                                       normals, 1, runner, {}, {}, policy);
    };
    auto result = run();
    CHECK(result.reverseChecks == 3);
    CHECK(result.filteredStops.empty());
    CHECK(result.failures.empty());
    CHECK(result.completedReferenceArcBase == 250);
    size_t calls = 0;
    policy.verify = [&](const auto&) { return ++calls == 1; };
    result = run();
    REQUIRE(result.filteredStops.size() == 1);
    CHECK(result.filteredStops[0][0] == doctest::Approx(100));
    CHECK(result.filteredStops[0][1] == doctest::Approx(200));
    CHECK(result.filteredStops[0][2] == doctest::Approx(250));
    CHECK(result.segments[0].tracePointsBase.back()[0] == doctest::Approx(100));
    CHECK(result.failures.empty());
    CHECK(result.completedReferenceArcBase == 250);
    calls = 0;
    policy.nextSeed = [&](double arc) -> std::optional<PolylineArcSample> {
        return samplePolylineArc(reference, arc);
    };
    policy.verify = [&](const auto&) { return ++calls != 1; };
    result = run();
    REQUIRE(result.filteredStops.size() == 1);
    CHECK(result.filteredStops[0][2] == doctest::Approx(110));
    CHECK(result.segments.size() == 2);
    CHECK(result.failures.empty());
    CHECK(result.completedReferenceArcBase == 250);
    runner = [](const PolylineArcSample& seed, double, int,
            const ReplayPointObserver& observer, const FiberTraceProgressCallback&) {
        for (double x = seed.arc + 10; x <= 250; x += 10)
            if (!observer({x,200,0}, 0, int(x/10))) break;
        return std::string("stopped");
    };
    policy.verify = [](const auto&) { return false; };
    // A sustained excursion exceeds grace160. Keep the interval huge so
    // the GT failure necessarily arrives before a periodic check.
    policy.intervalBase = 100000;
    result = run();
    CHECK_FALSE(result.failures.empty());
    CHECK(result.filteredStops.empty());
    CHECK(result.reverseChecks == 0);
    policy.allEndpoints = true;
    result = run();
    CHECK(result.failures.empty());
    CHECK_FALSE(result.filteredStops.empty());
    REQUIRE(result.segments.front().filteredDistanceFailure.has_value());
    CHECK(result.reverseChecks > 0);
    policy.verify = [](const auto&) { return true; };
    result = run();
    CHECK_FALSE(result.failures.empty());
    CHECK(result.filteredStops.empty());
}

TEST_CASE("Full prefix consistency verifies to segment start including final partial endpoint")
{
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override {
            return {{0,1,0},true,{}};
        }
    } normals;
    const auto reference = makePolylineArcGeometry({{0,0,0},{250,0,0}});
    ReplaySegmentRunner runner = [](const PolylineArcSample& seed, double, int,
            const ReplayPointObserver& observe, const FiberTraceProgressCallback&) {
        for (double x=seed.arc+10; x<=250; x+=10)
            if (!observe({x,0,0},0,int(x/10))) break;
        return std::string("stopped");
    };
    std::vector<std::pair<double,double>> spans;
    ReplayConsistencyPolicy policy;
    policy.intervalBase=100;
    policy.toStart=true;
    policy.allEndpoints=true;
    policy.verify=[&](const auto& points) {
        spans.emplace_back(points.front()[0],points.back()[0]); return true;
    };
    policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return {};};
    const auto run=[&] {
        return traceReferenceReplayCore(reference,0,250,10,3,40,160,2,1,normals,1,runner,{},{},policy);
    };
    auto result=run();
    REQUIRE(spans.size()==3);
    CHECK(spans[0]==std::make_pair(0.,100.));
    CHECK(spans[1]==std::make_pair(0.,200.));
    CHECK(spans[2]==std::make_pair(0.,250.));
    CHECK(result.reverseChecks==3); // No duplicate terminal call.
    spans.clear(); policy.toStart=false;
    result=run();
    REQUIRE(spans.size()==3);
    CHECK(spans[1]==std::make_pair(100.,200.));
    CHECK(spans[2]==std::make_pair(200.,250.));
}

TEST_CASE("Presence filtering retains accepted prefix and shares safe reseeding") {
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0,1,0},true,{}}; }
    } normals;
    auto ref=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    ReplaySegmentRunner runner=[](const PolylineArcSample& seed,double,int,const ReplayPointObserver& observe,const FiberTraceProgressCallback&) {
        for(double x=seed.arc+10;x<=100;x+=10) if(!observe({x,0,0},0,int(x))) break;
        return std::string("stopped");
    };
    ReplayConsistencyPolicy policy;
    auto run=[&]{return traceReferenceReplayCore(ref,0,100,10,3,40,160,2,1,normals,1,runner,{},{},policy);};
    CHECK(run().filteredStops.empty());
    policy.acceptPoint=[](const auto& p,const auto&){return p[0]!=50;};
    policy.nextSeed=[&](double arc)->std::optional<PolylineArcSample>{return samplePolylineArc(ref,arc);};
    auto r=run();
    REQUIRE(r.filteredStops.size()==1);
    CHECK(r.filteredStops[0]==std::array<double,3>{40,50,60});
    CHECK(r.segments[0].terminationReason=="presence_filtered");
    CHECK(r.failures.empty()); CHECK(r.reverseChecks==0); CHECK(r.completedReferenceArcBase==100);
    policy.intervalBase=30; policy.allEndpoints=true;
    policy.verify=[](const auto&){return true;};
    r=run();
    CHECK(r.filteredStops[0][0]==30); // No terminal reverse verification of rejected tail.
    CHECK(r.reverseChecks==3);
    policy.intervalBase=0;
    policy.acceptPoint=[](const auto&,const auto&){return false;};
    r=run();
    CHECK(r.failures.empty()); CHECK(r.filteredStops.size()==5); CHECK(r.completedReferenceArcBase==100);
    policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return {};};
    r=run();
    CHECK(r.filteredStops.size()==1); CHECK(r.completedReferenceArcBase==100);
}

TEST_CASE("Unified prediction seeding covers initial error and filtered starts") {
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0,1,0},true,{}}; }
    } normals;
    const auto ref=makePolylineArcGeometry({{0,0,0},{100,0,0}});
    ReplayConsistencyPolicy policy;
    policy.seedAllStarts=true;
    std::vector<double> searches;
    policy.nextSeed=[&](double arc)->std::optional<PolylineArcSample> {
        searches.push_back(arc);
        if (arc+5>=100) return {};
        auto seed=samplePolylineArc(ref,arc+5);
        seed.point[1]=1; // Preserve prediction-derived point, not GT replacement.
        return seed;
    };
    int runs=0;
    ReplaySegmentRunner runner=[&](const PolylineArcSample& seed,double,int,const ReplayPointObserver& observe,const FiberTraceProgressCallback&) {
        CHECK(seed.point[1]==1);
        ++runs;
        for(double x=seed.arc+10;x<=105;x+=10)
            if(!observe({x,runs==1 ? 100. : 0.,0},0,int(x))) break;
        return std::string("stopped");
    };
    const auto run=[&]{return traceReferenceReplayCore(ref,0,100,10,3,40,0,2,1,normals,1,runner,{},{},policy);};
    auto r=run();
    REQUIRE(r.failures.size()==1);
    REQUIRE(searches.size()==2);
    CHECK(searches[0]==0);
    CHECK(searches[1]==doctest::Approx(r.failures[0].referenceArcBase+10));
    CHECK(r.segments[0].startReferenceArcBase==5);
    CHECK(r.segments[1].startReferenceArcBase==doctest::Approx(searches[1]+5));
    CHECK(r.filteredStops.empty());
    CHECK(r.completedReferenceArcBase==100);
    searches.clear(); runs=1;
    policy.acceptPoint=[](const auto& p,const auto&){return p[0]!=15;};
    r=run();
    REQUIRE(r.filteredStops.size()==1);
    CHECK(searches.size()==2); // Initial plus filtered; preselected seed not searched twice.
    CHECK(r.failures.empty());
    policy.nextSeed=[](double)->std::optional<PolylineArcSample>{return {};};
    r=run();
    CHECK(r.segments.empty()); CHECK(r.failures.empty()); CHECK(r.filteredStops.empty());
    CHECK(r.completedReferenceArcBase==100);
    for(double bad:{-1.,100.,std::numeric_limits<double>::quiet_NaN()}) {
        policy.nextSeed=[&,bad](double)->std::optional<PolylineArcSample>{auto s=samplePolylineArc(ref,0);s.arc=bad;return s;};
        CHECK_THROWS(run());
    }
}
