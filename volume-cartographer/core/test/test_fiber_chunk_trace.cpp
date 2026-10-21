#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>

using namespace vc::fiber_tracer;
namespace {
struct Temp {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("vc_chunk_trace_"+std::to_string(std::mt19937_64(std::random_device{}())()));
    ~Temp(){std::error_code e;std::filesystem::remove_all(path,e);}
};
class Normal final:public vc::lasagna::NormalSampler {
public:
    bool supportsConcurrentSampling() const noexcept override {return true;}
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override {return {{0,0,1},true};}
};
class Tubes final:public FiberPredictionSource {
public:
    mutable std::atomic<size_t> calls{0};
    mutable std::atomic<int> fail{0};
    bool low=false;
    bool supportsConcurrentSampling() const noexcept override {return true;}
    FiberPredictionSample sample(const cv::Vec3d& p,const cv::Vec3d&) const override {
        ++calls;
        if(fail.exchange(0)) throw std::runtime_error("transient input failure");
        double y=std::remainder(p[1]-8,16),z=std::remainder(p[2]-8,16);
        float presence=std::exp(-(y*y+z*z)/8);
        if(low&&p[0]>=10) presence=.01f;
        FiberPredictionSample s;s.options.push_back({{1,0,0},presence,true});return s;
    }
};
FiberChunkTraceConfig chunkConfig() {
    FiberChunkTraceConfig c;c.coreBase=32;c.haloBase=8;c.gridBase=4;c.nmsBase=4;
    c.tubeHalfBase=4;c.tubeStepBase=4;c.rimNormalBase=4;c.rimTangentBase=4;
    c.joinMinimumBase=8;c.pinSamplingRegion=false;return c;
}
FiberTraceConfig traceConfig() {
    FiberTraceConfig c;c.stepVoxels=4;c.beamWidth=3;c.beamLookaheadSteps=2;
    c.coneAngleDegrees=0;c.parallelThreads=1;c.smoothnessWeight=0;
    c.smoothnessNormalWeight=0;c.smoothnessTangentWeight=0;c.cumulativeSmoothnessTangentWeight=0;
    c.replayPresenceCutoff=.1;c.replayPresenceModulated=true;return c;
}
void parallel(size_t count,const std::function<void(size_t)>& run) {
    std::vector<std::future<void>> jobs;
    for(size_t worker=0;worker<4;++worker) jobs.push_back(std::async(std::launch::async,[&,worker]{
        for(size_t i=worker;i<count;i+=4) run(i);
    }));
    std::exception_ptr error;
    for(auto& job:jobs) try{job.get();}catch(...){if(!error)error=std::current_exception();}
    if(error) std::rethrow_exception(error);
}
}

TEST_CASE("chunk paths are deterministic in serial and parallel and stop at bounds") {
    Tubes source;Normal normals;const auto c=chunkConfig();const auto t=traceConfig();
    auto a=buildFiberChunkPaths({0,0,0},source,t,normals,c,1);
    auto b=buildFiberChunkPaths({0,0,0},source,t,normals,c,1,parallel);
    REQUIRE(a->lines.size()>1);
    CHECK(a->maxima==b->maxima);CHECK(a->seeds==b->seeds);
    CHECK(a->lines==b->lines);CHECK(a->endReasons==b->endReasons);
    for(size_t i=0;i<a->lines.size();++i) {
        CHECK(a->endReasons[i][0]=="chunk_boundary");CHECK(a->endReasons[i][1]=="chunk_boundary");
        for(size_t j=1;j+1<a->lines[i].size();++j)
            CHECK(a->lines[i][j][0]>=a->lo[0]&&a->lines[i][j][0]<a->hi[0]);
    }
}

TEST_CASE("chunk frontier stops on an exiting non-winning survivor") {
    class Source final:public FiberPredictionSource {
    public:
        mutable double maxX=0;
        FiberPredictionSample sample(const cv::Vec3d& p,const cv::Vec3d&) const override {
            maxX=std::max(maxX,p[0]);FiberPredictionSample s;s.options.push_back({{1,0,0},1,true});return s;
        }
    } source;
    auto c=traceConfig();c.beamWidth=8;c.beamPruneDistanceVoxels=0;c.coneAngleDegrees=25;c.coneAngleStepDegrees=10;
    const auto result=traceFiberChunkHalf(source,{0,0,0},{1,0,0},100,c,nullptr,
        {cv::Vec3d(-100,-.01,-.01),cv::Vec3d(100,.01,.01)});
    CHECK(result.reason=="chunk_boundary");
    CHECK(source.maxX<=8.00001);
    CHECK(std::abs(result.points.back()[1])<.01);
}

TEST_CASE("presence feasibility preserves valid lookahead and beats a boundary stop") {
    Tubes source;source.low=true;auto c=traceConfig();
    for(bool modulated:{false,true}) {
        c.replayPresenceModulated=modulated;
        const auto result=traceFiberChunkHalf(source,{0,8,8},{1,0,0},100,c,nullptr,
            {cv::Vec3d(-10,0,0),cv::Vec3d(10,16,16)});
        CHECK(result.reason=="presence_cutoff");
        REQUIRE(result.points.size()==3);
        CHECK(result.points.back()[0]==8);
    }
}

TEST_CASE("chunk cache shares builds persists checks configuration and retries failures") {
    Temp temp;Tubes source;Normal normals;auto t=traceConfig();auto c=chunkConfig();
    FiberChunkTraceCache cache(c,temp.path,"fixture-v1");
    source.fail=1;
    CHECK_THROWS(cache.get({0,0,0},source,t,normals,1));
    std::vector<std::future<std::shared_ptr<const FiberChunkPaths>>> jobs;
    for(int i=0;i<4;++i) jobs.push_back(std::async(std::launch::async,[&]{return cache.get({0,0,0},source,t,normals,1);}));
    const auto a=jobs.front().get();
    for(size_t i=1;i<jobs.size();++i) CHECK(jobs[i].get()==a);
    const size_t calls=source.calls;
    CHECK(cache.get({0,0,0},source,t,normals,1)==a);CHECK(source.calls==calls);
    FiberChunkTraceCache disk(c,temp.path,"fixture-v1");
    const auto b=disk.get({0,0,0},source,t,normals,1);
    CHECK(b->lines==a->lines);CHECK(b->endReasons==a->endReasons);CHECK(source.calls==calls);
    FiberChunkTraceCache wrong(c,temp.path,"different-model");
    CHECK_THROWS(wrong.get({0,0,0},source,t,normals,1));
    t.replayPresenceCutoff=.2;
    CHECK_THROWS(cache.get({0,0,0},source,t,normals,1));
    FiberChunkTraceCache changed(c,temp.path,"fixture-v1");
    CHECK_THROWS(changed.get({0,0,0},source,t,normals,1));
    CHECK(loadFiberChunkPaths(temp.path/"0_0_0.json")->lines==a->lines);
}

TEST_CASE("chunk configuration rejects invalid and unbounded allocations") {
    Tubes source;Normal normal;auto c=chunkConfig();auto t=traceConfig();
    c.gridBase=1e-10;CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normal,c,1));
    c=chunkConfig();c.tubeStepBase=0;CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normal,c,1));
    c=chunkConfig();t.stepVoxels=std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normal,c,1));
    CHECK(source.calls==0);
}

TEST_CASE("chunk matching uses orientation overlap and optional width not nearest endpoint alone") {
    FiberChunkPaths chunk;chunk.lo={-100,-100,-100};chunk.hi={200,100,100};
    chunk.lines={{{0,1,0},{100,1,0}},{{0,2,0},{100,2,0}},{{0,0,0},{0,100,0}}};
    const std::vector<cv::Vec3d> history{{0,0,0},{16,0,0},{32,0,0},{48,0,0}};
    auto c=chunkConfig();c.joinMinimumBase=8;
    auto match=matchFiberChunkPaths(chunk,history,{1,0,0},c,false);
    REQUIRE(match.valid);CHECK(match.line==0);CHECK(match.overlap>=8);
    c.joinWidthWeight=true;
    match=matchFiberChunkPaths(chunk,history,{1,0,0},c,false,nullptr,[](const auto& points){
        std::vector<float> values;for(const auto& p:points) values.push_back(p[1]==1?100:20);return values;
    });
    REQUIRE(match.valid);CHECK(match.line==1);
    FiberChunkInitialRegion region;region.center={32,0,0};region.metric=cv::Matx33d::eye()/4.0;
    match=matchFiberChunkPaths(chunk,{{32,0,0}},{-1,0,0},c,true,&region);
    REQUIRE(match.valid);CHECK(match.reverse);CHECK(match.line==0);
    region.metric(0,0)=-1;CHECK_THROWS(matchFiberChunkPaths(chunk,history,{1,0,0},c,true,&region));
}

TEST_CASE("real Paris4 chunk construction roundtrips with identical parallel paths") {
    const char* manifest=std::getenv("VC_TEST_WIDTH_MANIFEST");
    const char* fiberPath=std::getenv("VC_TEST_WIDTH_FIBER");
    const char* normalManifest=std::getenv("VC_TEST_REVERSE_NORMAL_MANIFEST");
    if(!manifest||!fiberPath||!normalManifest) return;
    const auto dataset=vc::lasagna::LasagnaDataset::open(manifest);
    FiberPredictionFieldOptions options;options.enableWidth=true;
    FiberPredictionField source(dataset,options);
    vc::lasagna::LasagnaNormalSampler normals(vc::lasagna::LasagnaDataset::open(normalManifest));
    const auto fiber=loadFiberJson(fiberPath);const auto seed=fiber.controlPointsXyzBase.front();
    FiberChunkTraceConfig c;c.coreBase=64;c.haloBase=32;c.gridBase=8;c.nmsBase=8;
    c.tubeHalfBase=16;c.tubeStepBase=8;c.seedPresence=.4;c.tubePresence=.2;c.rimDrop=.05;
    auto t=traceConfig();t.stepVoxels=16;t.coneAngleDegrees=25;t.coneAngleStepDegrees=15;
    t.widthRelativeTolerance=.4;t.widthCosineWeight=true;
    const std::array<int,3> key{int(std::floor(seed[0]/64)),int(std::floor(seed[1]/64)),int(std::floor(seed[2]/64))};
    Temp temp;FiberChunkTraceCache cache(c,temp.path,"real-cache-test");
    auto a=cache.get(key,source,t,normals,1);
    auto b=buildFiberChunkPaths(key,source,t,normals,c,1,parallel);
    REQUIRE(a->seeds>0);REQUIRE(!a->lines.empty());
    CHECK(a->lines==b->lines);CHECK(a->endReasons==b->endReasons);CHECK(a->seeds==b->seeds);
    c.pinSamplingRegion=false;
    const auto unpinned=buildFiberChunkPaths(key,source,t,normals,c,1);
    CHECK(a->lines==unpinned->lines);CHECK(a->endReasons==unpinned->endReasons);
    c.pinSamplingRegion=true;
    FiberChunkTraceCache disk(c,temp.path,"real-cache-test");
    CHECK(disk.get(key,source,t,normals,1)->lines==a->lines);
    std::cout<<"Real Paris4 chunk: maxima="<<a->maxima<<" seeds="<<a->seeds<<" paths="<<a->lines.size()
             <<"; serial/parallel/unpinned/disk paths exact\n";
}
