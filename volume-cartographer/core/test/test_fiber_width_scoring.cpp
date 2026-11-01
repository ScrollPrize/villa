#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/FiberWidth.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayCore.hpp"
#include "vc/lasagna/Dataset.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <fstream>
#include <random>
#include <cstdlib>
#include <iostream>

using namespace vc::fiber_tracer;

TEST_CASE("packed uint8 base widths are converted into tracing coordinates") {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("fiber_width_packed_" + std::to_string(std::random_device{}()));
    fs::create_directories(root / "sd1");
    vcr::zarr::CreateOptions create;
    create.shape = {4,4,4,4}; create.chunks = {4,4,4,4}; create.compressor.clear();
    auto array = vcr::zarr::Writer::create(root / "sd1/1", create);
    std::vector<std::uint8_t> bytes(4*64, 255);
    std::fill(bytes.begin()+128, bytes.begin()+192, 128);
    std::fill(bytes.begin()+192, bytes.end(), 80);
    array.writeChunk({0,0,0,0}, bytes.data(), bytes.size());
    vcr::zarr::writeAttributes(root, nlohmann::json::parse(R"({"lasagna":{"version":3,"source_to_base":1,"base_shape_zyx":[8,8,8],"validity_channel":"presence","width_encoding":"uint8-base-voxels","groups":{"sd1":{"zarr":"sd1/1","scaledown":1,"channels":["presence","nx","ny","width"]}}}})"));
    vc::lasagna::LasagnaDatasetOpenOptions open;
    open.workingToBaseScale = 2;
    const auto dataset = vc::lasagna::LasagnaDataset::open(root, open);
    FiberPredictionFieldOptions options; options.enableWidth = true;
    FiberPredictionField field(dataset, options);
    std::vector<float> widths;
    field.sampleWidths({cv::Vec3f{1,1,1}}, widths);
    REQUIRE(widths.size() == 1);
    CHECK(widths[0] == doctest::Approx(40));
    const auto sample = field.sample(cv::Vec3d{1,1,1}, cv::Vec3d{1,0,0});
    REQUIRE(sample.options.size() == 1);
    CHECK(sample.options[0].presence == doctest::Approx(1));
    fs::remove_all(root);
}
namespace {
class WidthPrediction final : public FiberPredictionSource {
public:
    mutable size_t sampled = 0;
    bool fork = false, malformed = false, invalidSeed = false;
    double end = 1000;
    double invalidWidthFrom = 1000;
    bool supportsConcurrentSampling() const noexcept override { return true; }
    FiberPredictionSample sample(const cv::Vec3d& p, const cv::Vec3d&) const override {
        FiberPredictionSample out;
        if (p[0] < end) out.options.push_back({{1,0,0}, p[1] > .5 ? .9f : 1.f, true});
        return out;
    }
    float width(const cv::Vec3f& p) const {
        if (invalidSeed && p[0] == 0) return 0;
        if (p[0] >= invalidWidthFrom) return 0;
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
    const auto missing=traceFiberExtrapolation(source,{0,0,0},{1,0,0},8,c);
    CHECK(missing.reason=="invalid_width");
    CHECK(missing.points==std::vector<cv::Vec3d>{{0,0,0}});
    CHECK(missing.widthSamples==0);
    c.widthRelativeTolerance=-1;
    CHECK_THROWS_AS(traceFiberExtrapolation(source,{0,0,0},{1,0,0},8,c),std::invalid_argument);
}

TEST_CASE("missing committed width stops at the last valid prefix") {
    WidthPrediction source; source.invalidWidthFrom=12;
    auto c=config(); c.widthRelativeTolerance=.4;
    for (int depth : {1,2,3}) {
        c.beamLookaheadSteps=depth;
        const auto result=traceFiberExtrapolation(source,{0,0,0},{1,0,0},32,c);
        CHECK(result.reason=="invalid_width");
        REQUIRE(result.points.size()>1);
        for (const auto& point : result.points) CHECK(point[0]<source.invalidWidthFrom);
        checkMean(source,result);
    }
}

TEST_CASE("reference replay counts missing widths as filtered stops not failures") {
    class Normals final : public vc::lasagna::NormalSampler {
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override {
            return {{0,1,0},true,{}};
        }
    } normals;
    WidthPrediction source; source.invalidWidthFrom=12;
    auto c=config(); c.widthRelativeTolerance=.4;
    const auto reference=makePolylineArcGeometry({{0,0,0},{32,0,0}});
    ReplayConsistencyPolicy policy;
    policy.nextSeed=[&](double arc)->std::optional<PolylineArcSample> {
        return samplePolylineArc(reference,arc);
    };
    const ReplaySegmentRunner runner=[&](const PolylineArcSample& seed,double,int maximumSteps,
            const ReplayPointObserver& observe,const FiberTraceProgressCallback& progress) {
        return traceFiberExtrapolation(source,seed.point,seed.tangent,maximumSteps*c.stepVoxels,
            c,&normals,progress,false,observe).reason;
    };
    const auto result=traceReferenceReplayCore(reference,0,32,4,3,20,0,2,1,normals,1,
        runner,{},{},policy);
    CHECK(result.failures.empty());
    CHECK_FALSE(result.filteredStops.empty());
    CHECK(result.completedReferenceArcBase==32);
    REQUIRE_FALSE(result.segments.empty());
    for (const auto& segment : result.segments) {
        // A rejected seed remains as a zero-length segment, never a traced step.
        for (size_t i=1; i<segment.tracePointsBase.size(); ++i)
            CHECK(segment.tracePointsBase[i][0]<12);
    }
}

