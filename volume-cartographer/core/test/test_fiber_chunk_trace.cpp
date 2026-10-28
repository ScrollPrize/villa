#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/FiberEvaluation.hpp"
#include "vc/fiber_tracer/detail/PolylineSegmentIndex.hpp"
#include "vc/fiber_tracer/detail/FiberRouteOwnership.hpp"
#include "vc/fiber_tracer/detail/FiberContinuationGraph.hpp"
#include "vc/fiber_tracer/detail/FiberPackedStorage.hpp"
#include "vc/fiber_tracer/FiberAttributes.hpp"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <zstd.h>

using namespace vc::fiber_tracer;

TEST_CASE("fiber attributes preserve actual segment scores through resampling reversal and cropping") {
    FiberPathAttributes a;
    a.widths={10,20,40};a.normals={{0,1,0},{0,-1,0},{0,1,0}};a.normalValid={1,1,1};
    a.quality={.25,.75};a.costs={2,6};a.lengths={16,16};
    const auto r=resampleFiberAttributes({0,16,32},a,{0,8,16,24,32});
    CHECK(r.widths==std::vector<double>{10,15,20,30,40});
    CHECK(r.costs==std::vector<double>{1,1,3,3});
    CHECK(r.quality==std::vector<double>{.25,.25,.75,.75});
    for(const auto& n:r.normals)CHECK(cv::norm(n)==doctest::Approx(1));
    nlohmann::json piece={{"points_base",{{8,0,0},{16,0,0},{24,0,0}}}};
    attachFiberAttributes(piece,r,1,4);
    const auto& f=piece["fiber_attributes"];
    CHECK(f["width_mean"]["values"].get<double>()==doctest::Approx(21.25));
    CHECK(f["modulated_presence_mean"]["values"].get<double>()==doctest::Approx(.5));
    CHECK(f["tracer_cost_sum"]["values"].get<double>()==doctest::Approx(4));
    const auto joined=joinFiberAttributes(r,r);
    CHECK(joined.costs==std::vector<double>{3,3,1,1,1,1,3,3});
    CHECK(joined.widths.size()==9);
    const auto clipped=resampleFiberAttributes({0,16,32},a,{0,16,20});
    CHECK(clipped.costs.back()==doctest::Approx(1.5));
    CHECK(clipped.quality.back()==doctest::Approx(.75));
    auto invalid=a;invalid.normalValid[1]=false;
    const auto withMissing=resampleFiberAttributes({0,16,32},invalid,{0,8,32});
    CHECK_FALSE(withMissing.normalValid[1]);
    // New optional fields survive the generic serializer unchanged.
    piece["point_attributes"]["future_field"]=fiberAttribute({1,2,3},"arbitrary","Future extension");
    validateStreamFiberAttributes(piece);
    const auto decoded=nlohmann::json::from_msgpack(nlohmann::json::to_msgpack(piece));
    CHECK(decoded==piece);
    piece["point_attributes"]["width"]["values"]={1};
    CHECK_THROWS(validateFiberAttributes(piece));
}

TEST_CASE("chunk segment R-tree conservatively contains every nearby segment") {
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> random(-100.,100.);
    for(double offset:{0.,9e9}) {
        std::vector<std::vector<cv::Vec3d>> paths;
        for(int line=0;line<12;++line) {
            paths.emplace_back();
            for(int i=0;i<2+line*11;++i)
                paths.back().emplace_back(offset+random(rng),offset+random(rng),offset+random(rng));
        }
        paths.push_back({{0,0,0},{100,0,0},{0,0,0},{100,0,0}});
        paths.push_back({{0,0,0},{1e-7,0,0},{2e-7,1e-7,0}});
        const detail::ChunkSegmentIndex index(paths,[](const auto& p)->const auto& {return p;});
        for(int query=0;query<600;++query) {
            cv::Vec3d p(offset+random(rng)*3,offset+random(rng)*3,offset+random(rng)*3);
            const double radius=query%2?20.:10.;
            if(query<12)p=paths[query][0]+cv::Vec3d(radius,0,0);
            std::set<size_t> hits;
            index.nearbyPaths(p,radius,[&](size_t id){hits.insert(id);});
            for(size_t line=0;line<paths.size();++line) {
                const auto& points=paths[line];
                std::vector<double> arcs{0};
                for(size_t s=1;s<points.size();++s)arcs.push_back(arcs.back()+cv::norm(points[s]-points[s-1]));
                for(size_t s=1;s<points.size();++s) {
                    auto v=points[s]-points[s-1];double len=arcs[s]-arcs[s-1];
                    double t=std::clamp((p-points[s-1]).dot(v)/(len*len),0.,1.);
                    auto d=p-(points[s-1]+t*v);
                    if(d.dot(d)<=radius*radius)CHECK(hits.count(line)==1);
                }
            }
        }
    }
    size_t visits=0;
    detail::ChunkSegmentIndex().nearbyPaths({0,0,0},20,[&](size_t){++visits;});
    CHECK(visits==0);
}
namespace {
nlohmann::json readPackedFixture(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);
    const std::string data((std::istreambuf_iterator<char>(in)),{});
    const auto size=ZSTD_getFrameContentSize(data.data(),data.size());
    REQUIRE(size!=ZSTD_CONTENTSIZE_ERROR);REQUIRE(size!=ZSTD_CONTENTSIZE_UNKNOWN);
    std::vector<uint8_t> decoded(size);
    REQUIRE(ZSTD_decompress(decoded.data(),decoded.size(),data.data(),data.size())==size);
    return nlohmann::json::from_msgpack(decoded);
}
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
    void sampleWidths(const std::vector<cv::Vec3f>& points,std::vector<float>& out) const override {
        out.assign(points.size(),std::numeric_limits<float>::quiet_NaN());
    }
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
    CHECK_FALSE(std::filesystem::exists(temp.path/"0_0_0.json"));
    const auto packed=temp.path/"0_0_0.msgpack.zst";
    CHECK(loadFiberChunkPaths(packed)->lines==a->lines);
    const auto document=readPackedFixture(packed);
    CHECK(document.at("timings")==a->timings);
    std::ofstream(temp.path/"0_0_0.json")<<document.dump();
    // A corrupt new-format file must not silently fall back to an old file.
    std::ofstream(packed,std::ios::binary|std::ios::trunc)<<"broken";
    CHECK_THROWS(loadFiberChunkPaths(packed));
    FiberChunkTraceCache corrupt(c,temp.path,"fixture-v1");
    CHECK_THROWS(corrupt.get({0,0,0},source,traceConfig(),normals,1));
}

TEST_CASE("chunk configuration rejects invalid and unbounded allocations") {
    Tubes source;Normal normal;auto c=chunkConfig();auto t=traceConfig();
    c.gridBase=1e-10;CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normal,c,1));
    c=chunkConfig();c.tubeStepBase=0;CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normal,c,1));
    c=chunkConfig();t.stepVoxels=std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normal,c,1));
    CHECK(source.calls==0);
}

TEST_CASE("online refinement stays disabled unless iterations are explicitly enabled") {
    const FiberChunkTraceConfig defaults;
    CHECK(defaults.onlineRefineIterations==0);
    CHECK(defaults.beamCommitMode=="shared_prefix");
    CHECK(fiberChunkConfigFromJson(nlohmann::json::object()).onlineRefineIterations==0);
    auto j=fiberChunkConfigJson(defaults);
    CHECK(j.at("online_refine_iterations")==0);
    j.erase("online_refine_iterations");j["online_refine_mode"]="analytic";
    const auto omitted=fiberChunkConfigFromJson(j);
    CHECK(omitted.onlineRefineIterations==0);
    CHECK(fiberChunkCacheSpecification(omitted,traceConfig(),1,"fixture")==
          fiberChunkCacheSpecification(defaults,traceConfig(),1,"fixture"));
    j["online_refine_iterations"]=1;
    CHECK(fiberChunkConfigFromJson(j).onlineRefineIterations==1);
}

TEST_CASE("rolling beam commits have separate identity even without refinement") {
    auto c=chunkConfig();auto t=traceConfig();
    const auto baseline=fiberChunkCacheSpecification(c,t,1,"fixture");
    c.beamCommitMode="rolling";
    CHECK(fiberChunkCacheSpecification(c,t,1,"fixture")!=baseline);
    auto j=fiberChunkConfigJson(c);auto decoded=fiberChunkConfigFromJson(j);
    CHECK(decoded.beamCommitMode=="rolling");
    Tubes source;Normal normals;
    CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normals,c,1));
    c.beamCommitMode="invalid";
    CHECK_THROWS(buildFiberChunkPaths({0,0,0},source,t,normals,c,1));
}

TEST_CASE("analytic refinement identity is separate but batching is execution only") {
    auto c=chunkConfig();const auto t=traceConfig();
    const auto disabled=fiberChunkCacheSpecification(c,t,1,"fixture");
    c.onlineRefineMode="analytic";c.gpuBatchHalves=4096;c.gpuBatchMemoryMiB=1024;
    CHECK(fiberChunkCacheSpecification(c,t,1,"fixture")==disabled);
    auto decoded=fiberChunkConfigFromJson(fiberChunkConfigJson(c));
    CHECK(decoded.onlineRefineMode=="analytic");CHECK(decoded.gpuBatchHalves==4096);
    CHECK(decoded.gpuBatchMemoryMiB==1024);
    c.onlineRefineIterations=1;
    const auto analytic=fiberChunkCacheSpecification(c,t,1,"fixture");
    c.gpuBatchHalves=1024;c.gpuBatchMemoryMiB=4096;
    CHECK(fiberChunkCacheSpecification(c,t,1,"fixture")==analytic);
    c.onlineRefineMode="legacy";
    CHECK(fiberChunkCacheSpecification(c,t,1,"fixture")!=analytic);
    c.onlineRefineMode="unknown";
    CHECK_THROWS(fiberChunkConfigFromJson(fiberChunkConfigJson(c)));
    c.onlineRefineMode="legacy";c.gpuBatchHalves=0;
    CHECK_THROWS(fiberChunkConfigFromJson(fiberChunkConfigJson(c)));
}

TEST_CASE("online refinement supports up to 64 iterations") {
    auto c=chunkConfig();c.onlineRefineMode="analytic";c.beamCommitMode="rolling";
    c.onlineRefineIterations=64;
    CHECK(fiberChunkConfigFromJson(fiberChunkConfigJson(c)).onlineRefineIterations==64);
    const auto sixtyFour=fiberChunkCacheSpecification(c,traceConfig(),1,"fixture");
    c.onlineRefineIterations=8;
    CHECK(fiberChunkCacheSpecification(c,traceConfig(),1,"fixture")!=sixtyFour);
    c.onlineRefineIterations=65;
    CHECK_THROWS(fiberChunkConfigFromJson(fiberChunkConfigJson(c)));
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

TEST_CASE("real-data chunk construction roundtrips with identical parallel paths") {
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
    const auto full=makePolylineArcGeometry(fiber.linePointsXyzBase);
    const double begin=full.vertexArcs.at(fiber.controlPointLineIndices.front());
    const auto reference=makePolylineArcGeometry(slicePolylineArc(full,begin,begin+256));
    FiberChunkReplayConfig replay;replay.errorThresholdBase=20;replay.tangentialThresholdFactor=2;
    size_t joins=0;
    const auto first=traceFiberChunkReferenceReplay(cache,source,t,1,normals,reference,0,reference.length(),
        normals,1,replay,{},{},{},{},{},[&](const auto&){++joins;});
    const auto second=traceFiberChunkReferenceReplay(disk,source,t,1,normals,reference,0,reference.length(),normals,1,replay);
    REQUIRE(!first.segments.empty());CHECK(joins>0);
    CHECK(first.completedReferenceArcBase==reference.length());
    REQUIRE(first.segments.size()==second.segments.size());
    CHECK(first.failures.size()==second.failures.size());
    for(size_t i=0;i<first.segments.size();++i) {
        CHECK(first.segments[i].tracePointsBase==second.segments[i].tracePointsBase);
        CHECK(first.segments[i].terminationReason==second.segments[i].terminationReason);
    }
    std::cout<<"Real-data chunk replay: length="<<reference.length()<<" segments="<<first.segments.size()
             <<" failures="<<first.failures.size()<<" joins="<<joins<<"; disk reload exact\n";
    std::cout<<"Real-data chunk: maxima="<<a->maxima<<" seeds="<<a->seeds<<" paths="<<a->lines.size()
             <<"; serial/parallel/unpinned/disk paths exact\n";
}

TEST_CASE("chunk route joins suffixes both ways and respects distance and observer budgets") {
    Tubes source;auto t=traceConfig();t.stepVoxels=16;
    auto c=chunkConfig();c.coreBase=64;c.haloBase=32;
    std::vector<std::array<int,3>> keys;
    FiberChunkProvider provider=[&](const std::array<int,3>& key) {
        keys.push_back(key);auto p=std::make_shared<FiberChunkPaths>();p->key=key;
        p->lo={key[0]*64.-32,-32,-32};p->hi={key[0]*64.+96,96,96};
        p->lines.emplace_back();
        for(int i=0;i<=8;++i) p->lines.back().push_back({p->lo[0]+16*i,8,8});
        p->endReasons.push_back({"chunk_boundary","chunk_boundary"});return p;
    };
    FiberChunkInitialRegion region;region.center={0,8,8};
    for(int sign:{-1,1}) {
        keys.clear();std::vector<cv::Vec3d> points;size_t joins=0;
        auto reason=traceFiberChunkRoute(provider,c,source,t,1,region,{double(sign),0,0},200,100,
            [&](const auto& p,double,int step){CHECK(step==int(points.size()+1));points.push_back(p);return true;},
            [&](const auto& j){CHECK(j.spliceDistanceBase==0);++joins;});
        CHECK(reason=="chunk_length_limit");REQUIRE(!points.empty());CHECK(points.back()[0]==sign*200);
        CHECK(joins>=2);CHECK(keys.size()==joins);
        double last=0;for(const auto& p:points) {CHECK(sign*p[0]>sign*last);last=p[0];}
    }
    int observed=0;keys.clear();
    CHECK(traceFiberChunkRoute(provider,c,source,t,1,region,{1,0,0},200,2,
        [&](const auto&,double,int){++observed;return true;})=="chunk_length_limit");
    CHECK(observed==2);CHECK(keys.size()==1);
    observed=0;
    CHECK(traceFiberChunkRoute(provider,c,source,t,1,region,{1,0,0},200,100,
        [&](const auto&,double,int){++observed;return false;})=="observer_stop");
    CHECK(observed==1);
    const FiberChunkProvider missing=[](const auto& key){
        auto chunk=std::make_shared<FiberChunkPaths>();chunk->key=key;
        chunk->lo={-32,-32,-32};chunk->hi={96,96,96};return chunk;
    };
    CHECK(traceFiberChunkRoute(missing,c,source,t,1,region,{1,0,0},200,100,
        [](const auto&,double,int){return true;})=="chunk_no_initial_path");
    t.stepVoxels=8;
    CHECK_THROWS(traceFiberChunkRoute(provider,c,source,t,1,region,{1,0,0},200,100,
        [](const auto&,double,int){return true;}));
}

TEST_CASE("volume routes join both ways, retain selected seed and resume deterministically") {
    Temp temp;const auto input=temp.path/"input";std::filesystem::create_directories(input);
    Tubes source;Normal normals;auto t=traceConfig();t.stepVoxels=16;
    auto c=chunkConfig();c.coreBase=64;c.haloBase=32;c.joinDistanceBase=.25;
    const std::vector<std::array<int,3>> keys{{0,0,0},{1,0,0},{2,0,0}};
    for(const auto& key:keys) {
        nlohmann::json lines=nlohmann::json::array();
        for(double y:{8.,9.,64.,-1.,50.}) { // upper-face and halo-only paths have no owner here
            nlohmann::json line=nlohmann::json::array();
            for(int i=0;i<=8;++i)line.push_back({key[0]*64.-32+16*i,y,8});
            line.insert(line.begin(),line.front());line.push_back(line.back()); // repeated endpoints must retain graph edges
            lines.push_back(line);
        }
        detail::writeFiberPacked(input/(std::to_string(key[0])+"_0_0.msgpack.zst"),
            {{"key",key},{"lo",{key[0]*64.-32,-32,-32}},{"hi",{key[0]*64.+96,96,96}},
             {"maxima",5},{"seeds",5},{"lines",lines},
             {"end_reasons",std::vector<std::array<std::string,2>>(5,{"chunk_boundary","chunk_boundary"})},
             {"specification",fiberChunkCacheSpecification(c,t,1,"route-test")}});
    }
    FiberChunkRouteVolumeConfig cfg;cfg.cacheBytes=1;cfg.pagePoints=8; // exercise eviction and page splitting
    auto run=[&](const std::filesystem::path& out,size_t workers,const std::string& identity="route-test") {
        FiberEvaluationControl control;
        exportFiberChunkRoutes(source,t,1,normals,c,input,out,identity,keys,keys,cfg,control,workers,2);
        return control.status();
    };
    const auto a=temp.path/"serial",b=temp.path/"parallel";
    FiberEvaluationControl cancelled;cancelled.cancel("test cancellation");
    CHECK_THROWS(exportFiberChunkRoutes(source,t,1,normals,c,input,temp.path/"cancelled","route-test",keys,keys,cfg,cancelled,4,2));
    CHECK_FALSE(std::filesystem::exists(temp.path/"cancelled"));
    const auto serialStatus=run(a,1),parallelStatus=run(b,4);
    CHECK(serialStatus.built==3);CHECK(parallelStatus.built==3);
    auto index=readPackedFixture(a/"component_0.routes.msgpack.zst");
    REQUIRE(index["seeds"].size()==3);REQUIRE(index["pages"].size()==1);
    for(int page=0;page<2;++page) {
        const auto name="component_"+std::to_string(page)+".routes.0.msgpack.zst";
        auto left=readPackedFixture(a/name),right=readPackedFixture(b/name);
        left.erase("generation");right.erase("generation");CHECK(left==right);
        const auto& fiber=left["fibers"][0];const auto& points=fiber["points_base"];
        CHECK(fiber["source_line"]==page);CHECK(fiber["length_base"]==256);
        CHECK(points.front()[0]==-32);CHECK(points.back()[0]==224);
        for(const auto& p:points)CHECK(p[1]==8.+page);
        CHECK(fiber["stop_reasons"]==nlohmann::json({"chunk_inventory_boundary","chunk_inventory_boundary"}));
        CHECK(fiber["joins"][0].size()>=1);CHECK(fiber["joins"][1].size()>=2);
        CHECK_FALSE(fiber["owned_arcs"].empty());
    }
    const auto modified=std::filesystem::last_write_time(a/"component_0.routes.msgpack.zst");
    CHECK(run(a,4).reused==3);
    CHECK(std::filesystem::last_write_time(a/"component_0.routes.msgpack.zst")==modified);
    CHECK_THROWS(run(a,2,"different"));
    // A source identity mismatch is rejected even in a completely fresh export.
    CHECK_THROWS(run(temp.path/"wrong-source",2,"different"));
    cfg.maximumDistanceBase=100;CHECK_THROWS(run(a,1));cfg.maximumDistanceBase=1000000;
    cfg.maximumSteps=1;
    CHECK(run(temp.path/"budget",4).built==3);
    CHECK(run(temp.path/"budget",1).reused==3);
    cfg.maximumSteps=100000;
    auto damaged=readPackedFixture(a/"component_0.routes.0.msgpack.zst");
    const auto saved=damaged;
    damaged["fibers"][0]["joins"][0][0]["arc"]=-1;
    detail::writeFiberPacked(a/"component_0.routes.0.msgpack.zst",damaged);CHECK_THROWS(run(a,1));
    detail::writeFiberPacked(a/"component_0.routes.0.msgpack.zst",saved);
    // Partial pages without the final index do not masquerade as complete.
    std::filesystem::remove(a/"component_0.routes.msgpack.zst");CHECK(run(a,4).built==1);
    std::ofstream(a/"component_0.routes.0.msgpack.zst",std::ios::trunc)<<"corrupt";
    CHECK_THROWS(run(a,1));
    std::filesystem::remove(input/"2_0_0.msgpack.zst");CHECK_THROWS(run(temp.path/"missing",4));
}

TEST_CASE("volume ownership includes a core-crossing segment with no in-core vertex") {
    Temp temp;const auto input=temp.path/"input";std::filesystem::create_directories(input);
    Tubes source;Normal normals;auto t=traceConfig();t.stepVoxels=16;
    auto c=chunkConfig();c.coreBase=64;c.haloBase=32;
    const std::array<int,3> key{0,0,0};
    detail::writeFiberPacked(input/"0_0_0.msgpack.zst",{
        {"key",key},{"lo",{-32,-32,-32}},{"hi",{96,96,96}},{"maxima",1},{"seeds",1},
        {"lines",{{{-1,1,8},{1,-1,8}}}},
        {"end_reasons",std::vector<std::array<std::string,2>>(1,{"presence_cutoff","presence_cutoff"})},
        {"specification",fiberChunkCacheSpecification(c,t,1,"corner")}});
    // Move the second endpoint so the segment crosses a positive core interval.
    auto data=readPackedFixture(input/"0_0_0.msgpack.zst");data["lines"][0][1]={2,-1,8};
    detail::writeFiberPacked(input/"0_0_0.msgpack.zst",data);
    FiberChunkRouteVolumeConfig cfg;FiberEvaluationControl control;
    exportFiberChunkRoutes(source,t,1,normals,c,input,temp.path/"out","corner",{key},{key},cfg,control,2,1);
    CHECK(control.status().built==1);
    const auto page=readPackedFixture(temp.path/"out/component_0.routes.0.msgpack.zst");
    CHECK(page["fibers"].size()==1);CHECK(page["fibers"][0]["length_base"].get<double>()==doctest::Approx(std::sqrt(13.)));
}

TEST_CASE("explicit route seed respects flags, coordinates and limits") {
    Tubes source;auto t=traceConfig();t.stepVoxels=16;auto c=chunkConfig();c.coreBase=64;c.haloBase=32;
    auto chunk=std::make_shared<FiberChunkPaths>();chunk->key={0,0,0};chunk->lo={-32,-32,-32};chunk->hi={96,96,96};
    chunk->lines={{{0,8,8},{16,8,8},{32,8,8}},{{0,9,8},{16,9,8},{32,9,8}}};
    chunk->endReasons={{{"presence_cutoff","presence_cutoff"}},{{"presence_cutoff","presence_cutoff"}}};
    FiberChunkInitialRegion region;region.center={16,9,8};FiberChunkRouteStart start{1,1,false};
    auto run=[&] {
        std::vector<cv::Vec3d> points;
        auto stop=traceFiberChunkRoute([&](const auto&){return chunk;},c,source,t,1,region,{1,0,0},8,100,
            [&](const auto& p,double,int){points.push_back(p);return true;},{},&start);
        return std::make_pair(stop,points);
    };
    auto result=run();CHECK(result.first=="chunk_length_limit");REQUIRE(result.second.size()==1);
    CHECK(result.second[0]==cv::Vec3d(24,9,8));
    region.center={16,8,8};CHECK_THROWS(run());region.center={16,9,8};
    c.conflictNormalBase=10;c.conflictTangentBase=20;CHECK_THROWS(run());
    chunk->conflictAnnotation={{"specification",{{"version",1},{"algorithm","chunk-conflict-v1"},
        {"normal",10},{"tangent",20},{"agreement",c.conflictAgreementBase},{"separation",c.conflictSeparationBase}}}};
    chunk->conflictFlags={false,true};CHECK(run().first=="chunk_seed_conflict");
    chunk->conflictAnnotation["specification"]["normal"]=11;CHECK_THROWS(run());
}

TEST_CASE("cross-chunk loops terminate with an explicit cycle reason") {
    Tubes source;auto t=traceConfig();t.stepVoxels=16;auto c=chunkConfig();c.coreBase=64;c.haloBase=32;
    auto left=std::make_shared<FiberChunkPaths>(),right=std::make_shared<FiberChunkPaths>();
    left->key={0,0,0};right->key={1,0,0};left->lo={-32,-32,-32};left->hi={96,96,96};
    right->lo={32,-32,-32};right->hi={160,96,96};
    left->lines={{{80,24,8},{64,24,8},{48,24,8},{32,24,8},{16,24,8},{0,24,8},
                   {0,8,8},{16,8,8},{32,8,8},{48,8,8},{64,8,8},{80,8,8},{96,8,8}}};
    right->lines={{{48,8,8},{64,8,8},{80,8,8},{96,8,8},{112,8,8},{128,8,8},
                    {128,24,8},{112,24,8},{96,24,8},{80,24,8},{64,24,8},{48,24,8}}};
    left->endReasons={{{"chunk_boundary","chunk_boundary"}}};right->endReasons=left->endReasons;
    const FiberChunkProvider provider=[&](const auto& key) {
        if(key==left->key)return left;if(key==right->key)return right;
        throw std::runtime_error("cycle unexpectedly left its two cores");
    };
    FiberChunkInitialRegion region;region.center={32,8,8};
    for(bool reverse:{false,true}) {
        FiberChunkRouteStart start{0,8,reverse};size_t points=0;
        auto reason=traceFiberChunkRoute(provider,c,source,t,1,region,reverse?cv::Vec3d(-1,0,0):cv::Vec3d(1,0,0),
            10000,1000,[&](const auto&,double,int){++points;return true;},{},&start);
        CHECK(reason=="chunk_route_cycle");CHECK(points>5);CHECK(points<50);
    }
}

TEST_CASE("chunk route preserves semantic stops and rejects unsupported splices") {
    Tubes source;auto t=traceConfig();t.stepVoxels=16;
    auto c=chunkConfig();c.coreBase=64;c.haloBase=32;
    FiberChunkInitialRegion region;region.center={0,8,8};
    std::string terminal="presence_cutoff";int requested=0;
    FiberChunkProvider provider=[&](const std::array<int,3>& key) {
        ++requested;auto p=std::make_shared<FiberChunkPaths>();p->key=key;
        p->lo={key[0]*64.-32,-32,-32};p->hi={key[0]*64.+96,96,96};
        const double y=key[0]==0?8:10;
        p->lines.emplace_back();for(int i=0;i<=8;++i) p->lines.back().push_back({p->lo[0]+16*i,y,8});
        p->endReasons.push_back({terminal,terminal});return p;
    };
    const auto observe=[](const auto&,double,int){return true;};
    for(const auto* reason:{"presence_cutoff","invalid_width"}) {
        terminal=reason;requested=0;
        CHECK(traceFiberChunkRoute(provider,c,source,t,1,region,{1,0,0},300,100,observe)==reason);
        CHECK(requested==1);
    }
    terminal="chunk_boundary";source.low=true;
    CHECK(traceFiberChunkRoute(provider,c,source,t,1,region,{1,0,0},300,100,observe)=="chunk_unsupported_splice");
    source.low=false;t.widthRelativeTolerance=.4;
    const auto invalidWidth=traceFiberChunkRoute(provider,c,source,t,1,region,{1,0,0},300,100,observe);
    CHECK(invalidWidth=="chunk_invalid_splice_width");
}

TEST_CASE("chunk reference replay shares resets metrics and reverse checks") {
    Temp tmp;Tubes source;Normal normals;auto c=chunkConfig();c.coreBase=64;c.haloBase=32;
    auto t=traceConfig();t.stepVoxels=16;
    FiberChunkTraceCache cache(c,tmp.path,"tube-replay");
    const auto reference=makePolylineArcGeometry({{0,8,8},{240,8,8}});
    FiberChunkReplayConfig config;config.errorThresholdBase=2;
    ReplayConsistencyPolicy policy;policy.intervalBase=32;int checks=0;
    policy.verify=[&](const auto& points){CHECK(points.size()>1);++checks;return true;};
    const auto result=traceFiberChunkReferenceReplay(cache,source,t,1,normals,reference,0,240,normals,1,
        config,{},{},{},{},policy);
    CHECK(result.completedReferenceArcBase==240);CHECK(result.failures.empty());CHECK(checks>0);
    REQUIRE(result.segments.size()==1);CHECK(result.segments[0].tracePointsBase.front()==cv::Vec3d(0,8,8));
    const auto endpoint=result.segments[0].tracePointsBase.back();
    // Retain the actual cached vertex (grid starts at x=4), not a point
    // fabricated at the GT endpoint. The shared metric projects it to arc240.
    CHECK(endpoint==cv::Vec3d(244,8,8));
    CHECK(result.segments[0].endReferenceArcBase==240);
    // A tiny acceptance region with no cached path must skip seeds, not emit a
    // fabricated bridge from reference geometry to a distant fiber.
    const auto absent=makePolylineArcGeometry({{0,0,0},{64,0,0}});
    const auto empty=traceFiberChunkReferenceReplay(cache,source,t,1,normals,absent,0,64,normals,1,config);
    CHECK(empty.segments.empty());CHECK(empty.seedSearchSkippedBase==64);
    // Historical replay first selects a prediction peak, and records a no-path
    // stop if the stricter cached-path ellipse cannot match it. Default mode
    // above continues to skip such starts, preserving main's generic behavior.
    auto historical=c;historical.presencePeakRestarts=true;
    FiberChunkTraceCache peakCache(historical,tmp.path,"tube-replay");
    const auto peakResult=traceFiberChunkReferenceReplay(peakCache,source,t,1,normals,
        absent,0,64,normals,1,config);
    REQUIRE(!peakResult.failures.empty());
    CHECK(peakResult.failures.front().reason=="chunk_no_initial_path");
    CHECK(peakResult.segments.front().tracePointsBase.front()!=cv::Vec3d(0,0,0));
    const auto bent=makePolylineArcGeometry({{0,8,8},{96,8,8},{144,40,8},{240,40,8}});
    const auto reset=traceFiberChunkReferenceReplay(cache,source,t,1,normals,bent,0,bent.length(),normals,1,config);
    CHECK(!reset.failures.empty());CHECK(reset.segments.size()>1);
    CHECK(reset.completedReferenceArcBase==bent.length());
}

TEST_CASE("prediction based beam replay seeds the first segment at a peak") {
    Tubes source;Normal normal;auto trace=traceConfig();
    FiberEvaluationCase input{{{0,7,8},{64,7,8}}};
    FiberChunkReplayConfig config;config.errorThresholdBase=20;
    const auto replay=replayFiberTrainingCase(source,trace,1,normal,normal,1,input,config,{},true);
    REQUIRE(!replay.segments.empty());
    CHECK(replay.segments.front().tracePointsBase.front()==cv::Vec3d(0,8,8));
    const auto reference=makePolylineArcGeometry(input.referenceBase);
    trace.widthRelativeTolerance=.4;
    // This source has valid presence/directions but invalid widths. A width-
    // modulated trace must never select such a peak as its starting point.
    CHECK_FALSE(findFiberReplayRestartSeed(source,trace,1,normal,1,reference,0,64,16));
}

TEST_CASE("Recovered conflict filter rejects branching partners, not duplicates") {
    Normal normal;FiberChunkTraceConfig cfg;
    cfg.conflictNormalBase=10;cfg.conflictTangentBase=20;
    cfg.conflictAgreementBase=64;cfg.conflictSeparationBase=32;
    FiberChunkPaths paths;paths.key={1,2,3};
    for(int line=0;line<3;++line) {
        std::vector<cv::Vec3d> points;
        for(int x=0;x<=320;x+=8)points.emplace_back(x,line==0?0.:line==1?std::max(0,x-128)*.5:100.,0);
        paths.lines.push_back(points);paths.endReasons.push_back({"chunk_boundary","chunk_boundary"});
    }
    const auto original=paths;
    auto report=filterFiberChunkConflicts(paths,cfg,normal,1);
    CHECK(report["rejected"]==nlohmann::json::array({0,1}));
    CHECK(paths.lines[0].empty());CHECK(paths.lines[1].empty());
    CHECK(paths.lines[2]==original.lines[2]);
    CHECK(paths.endReasons==original.endReasons);
    auto duplicate=original;duplicate.lines[1]=duplicate.lines[0];
    report=filterFiberChunkConflicts(duplicate,cfg,normal,1);
    CHECK(report["rejected"].empty());
    auto reversed=original;std::reverse(reversed.lines[1].begin(),reversed.lines[1].end());
    report=filterFiberChunkConflicts(reversed,cfg,normal,1);
    CHECK(report["rejected"]==nlohmann::json::array({0,1}));
    cfg.conflictNormalBase=std::numeric_limits<double>::infinity();
    CHECK_THROWS(filterFiberChunkConflicts(duplicate,cfg,normal,1));
}

TEST_CASE("boundary ambiguity reuses conflict grammar and permits equivalent copies") {
    Normal normals;FiberChunkTraceConfig c;c.conflictNormalBase=10;c.conflictTangentBase=20;
    FiberChunkPaths p;p.lo={-64,-64,-64};p.hi={576,576,576};
    p.conflictAnnotation={{"specification",{{"version",1},{"algorithm","chunk-conflict-v1"},
        {"normal",10},{"tangent",20},{"agreement",64},{"separation",32}}}};
    p.conflictFlags={false,false};
    std::vector<cv::Vec3d> history;
    for(int x=0;x<=128;x+=8)history.emplace_back(x,0,0);
    for(int i=0;i<2;++i){p.lines.emplace_back();for(int x=0;x<=320;x+=8)p.lines.back().emplace_back(x,0,0);}
    auto match=[&]{return matchFiberChunkPaths(p,history,{1,0,0},c,false,nullptr,{},.4,&normals,1);};
    CHECK(match().valid);CHECK_FALSE(match().ambiguous);
    for(auto& point:p.lines[1])point[1]=std::max(0.,point[0]-160)*.5;
    CHECK_FALSE(match().valid);CHECK(match().ambiguous);
    std::reverse(p.lines[1].begin(),p.lines[1].end());CHECK(match().ambiguous);
    p.conflictFlags[1]=true;CHECK(match().valid); // excluded alternatives cannot cause ambiguity
}

TEST_CASE("route interval ownership retains unvisited prefixes and budget tails") {
    detail::FiberRouteOwnership arcs(100);
    arcs.add(30,60);
    CHECK(arcs.available(50,false)==0);CHECK(arcs.available(50,true)==0);
    CHECK(arcs.available(20,false)==10);CHECK(arcs.available(80,true)==20);
    CHECK(arcs.available(30,false)==0);CHECK(arcs.available(60,true)==0);
    CHECK(std::isinf(arcs.available(30,true)));CHECK(std::isinf(arcs.available(60,false)));
    REQUIRE(arcs.uncovered(0,100));CHECK(arcs.uncovered(0,100)->second==30);
    arcs.add(0,30);CHECK(arcs.uncovered(0,100)->first==60);
    CHECK_THROWS(arcs.add(29,31));arcs.add(60,100);CHECK_FALSE(arcs.uncovered(0,100));
    CHECK_THROWS(arcs.add(-1,0));CHECK_THROWS(arcs.add(100,101));
    detail::FiberRouteOwnership tiny(100);tiny.add(0,50);tiny.add(50+1e-10,100);
    CHECK_FALSE(tiny.uncovered(0,100));CHECK_FALSE(tiny.uncovered(70,70+1e-10));
    detail::FiberRouteOwnership small(100);small.add(0,30);small.add(30+1.5e-9,100);
    const auto gap=small.uncovered(0,100);REQUIRE(gap);
    CHECK(small.available(gap->first,false)>small.tolerance());
    small.add(gap->first,gap->second);CHECK_FALSE(small.uncovered(0,100));
}

TEST_CASE("shared ownership policy converts both orientations and aliases only uncovered arcs") {
    for(bool reverse:{false,true}){
        detail::FiberRouteOwnership arcs(100);std::vector<std::tuple<double,double,bool>> recorded;
        auto policy=detail::fiberRouteOwnershipPolicy([&](const FiberChunkRouteJoin&){return std::pair{&arcs,100.};},
            [&](const auto&,double a,double b,bool alias){recorded.emplace_back(a,b,alias);});
        FiberChunkRouteJoin join;join.match.reverse=reverse;join.match.arc=20;
        CHECK(std::isinf(policy.available(join)));policy.consume(join,20,40);
        CHECK(policy.available(join)==0);policy.alias(join,10,30);
        REQUIRE(recorded.size()==2);
        CHECK(std::get<0>(recorded[0])==(reverse?60:20));
        CHECK(std::get<1>(recorded[0])==(reverse?80:40));
        CHECK(std::get<0>(recorded[1])==(reverse?80:10));
        CHECK(std::get<1>(recorded[1])==(reverse?90:20));
        CHECK(std::get<2>(recorded[1]));policy.alias(join,10,30);CHECK(recorded.size()==2);
    }
}

TEST_CASE("frozen replay preserves endpoints, directions and independent case ordering") {
    Tubes source;Normal normal;auto trace=traceConfig();trace.stepVoxels=16;
    auto config=chunkConfig();config.presencePeakRestarts=false;
    config.initialNormalRadiusBase=10;config.initialTangentialRadiusBase=40;
    FiberChunkReplayConfig replay;replay.errorThresholdBase=40;replay.tangentialThresholdFactor=2;
    const std::vector<cv::Vec3d> line{{0,8,8},{16,8,8},{32,8,8},{49,8,8}};
    FiberStoredRouteSet routes({line},{{"presence_cutoff","presence_cutoff"}});
    const auto forward=makePolylineArcGeometry(line);
    auto reversed=line;std::reverse(reversed.begin(),reversed.end());const auto backward=makePolylineArcGeometry(reversed);
    const auto run=[&](const auto& ref){return routes.replay(source,trace,1,ref,normal,1,config,replay);};
    auto a=run(forward),b=run(backward);REQUIRE(a.segments.size()==1);REQUIRE(b.segments.size()==1);
    CHECK(a.failures.empty());CHECK(b.failures.empty());
    CHECK(cv::norm(a.segments.back().tracePointsBase.back()-line.back())<1e-8);
    CHECK(cv::norm(b.segments.back().tracePointsBase.back()-line.front())<1e-8);
    const auto again=run(forward);CHECK(again.segments.size()==a.segments.size());
    CHECK(again.segments.front().tracePointsBase==a.segments.front().tracePointsBase);
    FiberStoredRouteSet empty({},{});CHECK(empty.replay(source,trace,1,forward,normal,1,config,replay).segments.empty());
}

TEST_CASE("continuation prefilter rejects forks and merges before traversal in either direction") {
    using End=detail::FiberPathEnd;
    const End a{{0,0,0},0,false},b{{0,0,0},1,false},c{{1,0,0},0,true},d{{1,0,0},1,true};
    detail::FiberContinuationGraph unique;unique.add(a,c);unique.add(c,a);
    CHECK(unique.edges().size()==1);CHECK(unique.reason(a,c).empty());CHECK(unique.reason(c,a).empty());
    CHECK(unique.reason(b,c)=="chunk_unqualified_continuation");
    auto merge=unique;merge.add(b,c);
    CHECK(merge.degree(c)==2);CHECK(merge.reason(a,c)=="chunk_ambiguous_continuation");
    CHECK(merge.reason(c,b)=="chunk_ambiguous_continuation");
    auto fork=unique;fork.add(a,d);
    CHECK(fork.reason(a,c)=="chunk_ambiguous_continuation");CHECK(fork.reason(d,a)=="chunk_ambiguous_continuation");
    // A rival's other ambiguity must not cause peeling that makes a link unique.
    merge.add(b,d);CHECK(merge.reason(a,c)=="chunk_ambiguous_continuation");
    CHECK(merge.reason(b,d)=="chunk_ambiguous_continuation");
}

TEST_CASE("export rules validate source and replay independently of case order") {
    Temp temp;const auto input=temp.path/"input";std::filesystem::create_directories(input);
    Tubes source;Normal normal;auto t=traceConfig();t.stepVoxels=16;
    auto c=chunkConfig();c.coreBase=64;c.haloBase=32;c.joinDistanceBase=2;
    std::vector<std::array<int,3>> keys{{0,0,0},{1,0,0}};
    for(const auto& key:keys){
        nlohmann::json line=nlohmann::json::array();
        for(int x=-32;x<=96;x+=16)line.push_back({key[0]*64+x,8,8});
        detail::writeFiberPacked(input/(std::to_string(key[0])+"_0_0.msgpack.zst"),{
            {"key",key},{"lo",{key[0]*64-32,-32,-32}},{"hi",{key[0]*64+96,96,96}},
            {"maxima",1},{"seeds",1},{"lines",nlohmann::json::array({line})},
            {"end_reasons",nlohmann::json::array({key[0]?nlohmann::json::array({"chunk_boundary","presence_cutoff"}):
                nlohmann::json::array({"presence_cutoff","chunk_boundary"})})},
            {"specification",fiberChunkCacheSpecification(c,t,1,"rules")}});
    }
    FiberEvaluationControl control;FiberChunkRouteVolumeConfig cfg;cfg.rejectAmbiguousPaths=false;
    const auto out=temp.path/"out";
    exportFiberChunkRoutes(source,t,1,normal,c,input,out,"rules",keys,keys,cfg,control,4,1);
    FiberChunkRoutingRules rules(out,input,c,t,1,"rules");
    auto changed=c;changed.joinDistanceBase=3;
    CHECK_THROWS(FiberChunkRoutingRules(out,input,changed,t,1,"rules"));
    CHECK_THROWS(FiberChunkRoutingRules(out,input,c,t,1,"wrong-source"));
    c.cacheOnly=true;FiberChunkTraceCache cache(c,input,"rules");
    FiberChunkTraceCache otherIdentity(c,input,"other");
    FiberChunkTraceCache otherDirectory(c,temp.path/"elsewhere","rules");
    FiberChunkTraceCache otherSettings(changed,input,"rules");
    rules.validate(cache,t,1);
    CHECK_THROWS(rules.validate(otherIdentity,t,1));
    CHECK_THROWS(rules.validate(otherDirectory,t,1));
    CHECK_THROWS(rules.validate(otherSettings,t,1));
    const auto reference=makePolylineArcGeometry({{0,8,8},{112,8,8}});
    const auto reverse=makePolylineArcGeometry({{112,8,8},{0,8,8}});
    FiberChunkReplayConfig replay;replay.errorThresholdBase=40;
    const auto run=[&](const auto& ref){return traceFiberChunkReferenceReplay(cache,source,t,1,normal,
        ref,0,ref.length(),normal,1,replay,{},{},{},{},{},{},&rules);};
    const auto a=run(reference),b=run(reverse),again=run(reference);
    REQUIRE(a.segments.size()==1);CHECK(a.failures.empty());CHECK(b.failures.empty());
    CHECK(a.segments.front().tracePointsBase==again.segments.front().tracePointsBase);
    auto f=std::async(std::launch::async,[&]{return run(reference);});const auto parallel=run(reverse);
    CHECK(f.get().segments.front().tracePointsBase==a.segments.front().tracePointsBase);
    CHECK(parallel.segments.front().tracePointsBase==b.segments.front().tracePointsBase);
}

TEST_CASE("runtime history cannot resurrect a continuation absent from the prefilter") {
    Tubes source;auto t=traceConfig();t.stepVoxels=16;auto c=chunkConfig();c.coreBase=64;c.haloBase=32;
    const FiberChunkProvider provider=[](const auto& key){
        auto chunk=std::make_shared<FiberChunkPaths>();chunk->key=key;
        chunk->lo={key[0]*64.-32,-32,-32};chunk->hi={key[0]*64.+96,96,96};chunk->lines.emplace_back();
        for(int x=-32;x<=96;x+=16)chunk->lines[0].push_back({key[0]*64.+x,8,8});
        chunk->endReasons.push_back({"chunk_boundary","chunk_boundary"});return chunk;
    };
    const auto target=provider({1,0,0});
    // Short canonical context cannot qualify; longer stitched runtime context can.
    const std::vector<cv::Vec3d> shortHistory{{95,8,8},{96,8,8}},stitched{{64,8,8},{80,8,8},{96,8,8}};
    CHECK(findFiberChunkMatches(*target,shortHistory,{1,0,0},c,false).empty());
    CHECK_FALSE(findFiberChunkMatches(*target,stitched,{1,0,0},c,false).empty());
    detail::FiberContinuationGraph graph;
    const FiberChunkContinuationGate gate=[&](const auto& a,const auto& b){
        return graph.reason({a.key,a.match.line,a.match.reverse},{b.key,b.match.line,!b.match.reverse});};
    FiberChunkInitialRegion region;region.center={0,8,8};FiberChunkRouteStart start{0,2,false};
    std::vector<cv::Vec3d> points;size_t joins=0;
    const auto stop=traceFiberChunkRoute(provider,c,source,t,1,region,{1,0,0},256,100,
        [&](const auto& p,double,int){points.push_back(p);return true;},[&](const auto&){++joins;},
        &start,nullptr,nullptr,gate);
    if(stop!="chunk_unqualified_continuation")std::cerr<<"unexpected gate stop: "<<stop<<"\n";
    CHECK(stop=="chunk_unqualified_continuation");
    REQUIRE(!points.empty());CHECK(points.back()[0]==96);CHECK(joins==1);
}

TEST_CASE("full export prefilters two incoming paths into one target, including reverse traversal") {
    Temp temp;const auto input=temp.path/"input";std::filesystem::create_directories(input);
    Tubes source;Normal normals;auto t=traceConfig();t.stepVoxels=16;
    auto c=chunkConfig();c.coreBase=64;c.haloBase=32;c.joinDistanceBase=2;
    std::vector<std::array<int,3>> keys{{0,0,0},{1,0,0}};
    for(const auto& key:keys){
        nlohmann::json lines=nlohmann::json::array();
        for(int l=0;l<(key[0]?1:2);++l){
            nlohmann::json line=nlohmann::json::array();
            for(int x=-32;x<=96;x+=16)line.push_back({key[0]*64+x,8+l,8});
            lines.push_back(line);
        }
        const std::array<std::string,2> reason=key[0]?std::array<std::string,2>{"chunk_boundary","presence_cutoff"}:
            std::array<std::string,2>{"presence_cutoff","chunk_boundary"};
        detail::writeFiberPacked(input/(std::to_string(key[0])+"_0_0.msgpack.zst"),{
            {"key",key},{"lo",{key[0]*64-32,-32,-32}},{"hi",{key[0]*64+96,96,96}},
            {"maxima",lines.size()},{"seeds",lines.size()},{"lines",lines},
            {"end_reasons",std::vector<std::array<std::string,2>>(lines.size(),reason)},
            {"specification",fiberChunkCacheSpecification(c,t,1,"merge")}});
    }
    FiberEvaluationControl control;FiberChunkRouteVolumeConfig cfg;cfg.reduceDuplicatePaths=false;
    const auto out=temp.path/"out";
    exportFiberChunkRoutes(source,t,1,normals,c,input,out,"merge",keys,keys,cfg,control,4,1);
    const auto graph=readPackedFixture(out/"continuations.msgpack.zst");
    CHECK(graph["candidate_connections"]==2);CHECK(graph["accepted_connections"]==0);CHECK(graph["ambiguous_ends"]==1);
    const auto page=readPackedFixture(out/"component_0.routes.0.msgpack.zst");
    REQUIRE(page["fibers"].size()==3);
    for(const auto& fiber:page["fibers"]){
        CHECK(fiber["length_base"].get<double>()==128);
        const auto& stops=fiber["stop_reasons"];
        CHECK((stops[0]=="chunk_ambiguous_continuation"||stops[1]=="chunk_ambiguous_continuation"));
        for(const auto& joins:fiber["joins"])CHECK(joins.size()==1);
    }
    FiberEvaluationControl resume;
    exportFiberChunkRoutes(source,t,1,normals,c,input,out,"merge",keys,keys,cfg,resume,1,1);
    CHECK(resume.status().reused==1);
    cfg.reduceDuplicatePaths=true;FiberEvaluationControl reduced;
    exportFiberChunkRoutes(source,t,1,normals,c,input,temp.path/"reduced","merge",keys,keys,cfg,reduced,4,1);
    const auto fusion=readPackedFixture(temp.path/"reduced/path_reduction.msgpack.zst");
    CHECK(fusion["fused_paths"]==1);CHECK(fusion["ambiguous_paths"]==0);
    const auto reducedGraph=readPackedFixture(temp.path/"reduced/continuations.msgpack.zst");
    CHECK(reducedGraph["candidate_connections"]==1);CHECK(reducedGraph["accepted_connections"]==1);
    const auto fusedPage=readPackedFixture(temp.path/"reduced/component_0.routes.0.msgpack.zst");
    REQUIRE(fusedPage["fibers"].size()==1);CHECK(fusedPage["fibers"][0]["length_base"].get<double>()==192);
    CHECK(readPackedFixture(input/"0_0_0.msgpack.zst")["lines"].size()==2);
    cfg.rejectAmbiguousPaths=false;FiberEvaluationControl wrongMode;
    CHECK_THROWS(exportFiberChunkRoutes(source,t,1,normals,c,input,temp.path/"reduced","merge",keys,keys,cfg,wrongMode,1,1));
}

TEST_CASE("in-chunk endpoint reduction shares continuation gates without the 16 voxel runway") {
    auto cfg=chunkConfig();cfg.coreBase=256;cfg.haloBase=64;cfg.joinMinimumBase=32;cfg.joinDistanceBase=12;
    FiberChunkPaths paths;paths.key={0,0,0};paths.lo={-64,-64,-64};paths.hi={320,320,320};
    auto straight=[](double y,int step=16){std::vector<cv::Vec3d> line;for(int x=0;x<=160;x+=step)line.emplace_back(x,y,8);return line;};
    paths.lines={straight(8),straight(8,8)};paths.endReasons.assign(2,{"chunk_boundary","chunk_boundary"});
    std::reverse(paths.lines[1].begin(),paths.lines[1].end());
    const std::vector<size_t> second{1};
    CHECK(findFiberChunkMatches(paths,paths.lines[0],{1,0,0},cfg,false,nullptr,{},.4,&second).empty());
    CHECK(findFiberChunkMatches(paths,paths.lines[0],{1,0,0},cfg,false,nullptr,{},.4,&second,
        FiberChunkMatchPurpose::TerminalRelation).size()==1);
    auto outside=paths;
    for(auto& line:outside.lines)for(auto& p:line)p[1]=outside.hi[1];
    CHECK(findFiberChunkMatches(outside,outside.lines[0],{1,0,0},cfg,false,nullptr,{},.4,&second,
        FiberChunkMatchPurpose::TerminalRelation).empty());
    const auto reduction=reduceFiberChunkPaths(paths,cfg);
    CHECK(reduction.fusedPaths==1);CHECK(reduction.ambiguousPaths==0);
    CHECK(reduction.representative==std::vector<int64_t>({0,0}));
    const auto fusionOnly=reduceFiberChunkPaths(paths,cfg,{},.4,false);
    CHECK(fusionOnly.representative==reduction.representative);
    CHECK(fusionOnly.endReasons==reduction.endReasons);
    CHECK(fusionOnly.fusedPaths==reduction.fusedPaths);
    auto view=paths;reduction.apply(view);CHECK(view.lines[0]==paths.lines[0]);CHECK(view.lines[1].empty());
    CHECK(view.endReasons[0][0].starts_with("fused_member_stop:")); // interior ends must remain stopped
    CHECK(!nextFiberChunk(view,view.lines[0].front(),view.lines[0][1],view.endReasons[0][0],cfg).stop.empty());
    // Same terminal region, but divergent opposite end: do not fuse or retain either.
    paths.lines={straight(8),straight(8)};
    for(auto& p:paths.lines[1])if(p[0]<80)p[1]+=(80-p[0])*.5;
    auto conflict=reduceFiberChunkPaths(paths,cfg);CHECK(conflict.fusedPaths==0);CHECK(conflict.ambiguousPaths==2);
    const auto retained=reduceFiberChunkPaths(paths,cfg,{},.4,false);
    CHECK(retained.ambiguousPaths==0);CHECK(retained.retainedAmbiguousPaths==2);
    CHECK(retained.representative==std::vector<int64_t>({0,1}));
    auto retainedView=paths;retained.apply(retainedView);
    CHECK(retainedView.lines==paths.lines);CHECK(retainedView.endReasons==paths.endReasons);
    // An interior excursion is not made equivalent merely by matching BOTH ends.
    paths.lines={straight(8),straight(8)};paths.lines[1][5][1]+=32;
    conflict=reduceFiberChunkPaths(paths,cfg);CHECK(conflict.ambiguousPaths==2);
    // Shifted, reversed endpoints within the same geometric gate are duplicates.
    paths.lines={straight(8),straight(8,8)};
    for(auto& p:paths.lines[1])p[0]+=6;
    std::reverse(paths.lines[1].begin(),paths.lines[1].end());
    CHECK(reduceFiberChunkPaths(paths,cfg).fusedPaths==1);
    for(auto& p:paths.lines[1])p[0]+=1e-8;
    CHECK(reduceFiberChunkPaths(paths,cfg).fusedPaths==1);
    // A tail exceeding the geometric gate cannot be fused.
    paths.lines={straight(8),straight(8)};paths.lines[1].pop_back();
    conflict=reduceFiberChunkPaths(paths,cfg);CHECK(conflict.ambiguousPaths==2);
    // Nontransitive A~B~C must not turn A and C into a single identity.
    paths.lines={straight(8),straight(18),straight(28)};paths.endReasons.assign(3,{"chunk_boundary","chunk_boundary"});
    conflict=reduceFiberChunkPaths(paths,cfg);CHECK(conflict.ambiguousPaths==3);CHECK(conflict.fusedPaths==0);
    const auto retainedChain=reduceFiberChunkPaths(paths,cfg,{},.4,false);
    CHECK(retainedChain.representative==std::vector<int64_t>({0,1,2}));
    CHECK(retainedChain.retainedAmbiguousPaths==3);CHECK(retainedChain.fusedPaths==0);
    paths.lines={straight(8),straight(33)};paths.endReasons.resize(2);
    auto separate=reduceFiberChunkPaths(paths,cfg);CHECK(separate.terminalPairs==0);CHECK(separate.ambiguousPaths==0);
    // Invalid width at even one point may permit a soft join, but never prove equivalence.
    paths.lines={straight(8),straight(8)};cfg.joinWidthWeight=true;
    const auto incomplete=[](const std::vector<cv::Vec3d>& points){
        std::vector<float> widths;for(const auto& p:points)widths.push_back(p[0]==80?std::numeric_limits<float>::quiet_NaN():40);return widths;};
    conflict=reduceFiberChunkPaths(paths,cfg,incomplete);CHECK(conflict.ambiguousPaths==2);
    const auto valid=[](const auto& points){return std::vector<float>(points.size(),40);};
    CHECK(reduceFiberChunkPaths(paths,cfg,valid).fusedPaths==1);
    // Retaining endpoint ambiguity does not resurrect the original conflict flags.
    cfg.conflictNormalBase=10;cfg.conflictTangentBase=20;
    paths.conflictFlags={true,false};
    paths.conflictAnnotation={{"specification",{{"version",1},{"algorithm","chunk-conflict-v1"},
        {"normal",cfg.conflictNormalBase},{"tangent",cfg.conflictTangentBase},
        {"agreement",cfg.conflictAgreementBase},{"separation",cfg.conflictSeparationBase}}}};
    CHECK(reduceFiberChunkPaths(paths,cfg,valid,.4,false).representative==std::vector<int64_t>({-1,1}));
}

TEST_CASE("conflict candidate shared-run boundary and gap preserve directed decisions") {
    Normal normal;FiberChunkTraceConfig cfg;
    cfg.conflictNormalBase=10;cfg.conflictTangentBase=20;
    cfg.conflictAgreementBase=64;cfg.conflictSeparationBase=32;
    for(int end:{48,56,64,72})for(bool reverse:{false,true}) {
        FiberChunkPaths paths;
        std::vector<cv::Vec3d> straight;
        for(int x=0;x<=256;x+=8)straight.emplace_back(x,0,0);
        if(reverse)std::reverse(straight.begin(),straight.end());
        // Keep the departing segment long: a second bend would introduce a
        // nearest-projection jump and intentionally fail the correspondence gate.
        paths.lines={straight,{{-32,0,0},{double(end),0,0},{512,double(512-end),0}}};
        const auto report=filterFiberChunkConflicts(paths,cfg,normal,1);
        const auto pair=nlohmann::json::array({0,1});
        const auto& pairs=report["directed_conflict_pairs"];
        CHECK((std::find(pairs.begin(),pairs.end(),pair)!=pairs.end())==(end>=64));
    }
    cfg.conflictNormalBase=cfg.conflictTangentBase=1;
    FiberChunkPaths gap;
    for(int x=0;x<=256;x+=8) {
        if(gap.lines.empty())gap.lines.emplace_back();
        gap.lines[0].emplace_back(x,0,0);
    }
    gap.lines.push_back({{-32,0,0},{32,0,0},{40,80,0},{48,0,0},{80,0,0},{160,80,0},{320,80,0}});
    const auto report=filterFiberChunkConflicts(gap,cfg,normal,1);
    const auto pair=nlohmann::json::array({0,1});
    const auto& pairs=report["directed_conflict_pairs"];
    CHECK(std::find(pairs.begin(),pairs.end(),pair)==pairs.end());
}

TEST_CASE("Recovered config round trips and cache-only never constructs") {
    const auto config=chunkConfig();const auto trace=traceConfig();Normal normals;Tubes source;Temp dir;
    auto json=fiberChunkConfigJson(config);
    CHECK(fiberChunkConfigJson(fiberChunkConfigFromJson(json))==json);
    json["unrecognized_option"]=1;CHECK_THROWS(fiberChunkConfigFromJson(json));
    auto readonly=config;readonly.cacheOnly=true;
    FiberChunkTraceCache cache(readonly,dir.path,"synthetic");
    CHECK_THROWS(cache.get({0,0,0},source,trace,normals,1));
    CHECK(source.calls==0);
    CHECK_FALSE(std::filesystem::exists(dir.path));
    CHECK(fiberChunkCacheSpecification(readonly,trace,1,"synthetic")==fiberChunkCacheSpecification(config,trace,1,"synthetic"));
    auto filter=config;filter.conflictNormalBase=10;filter.conflictTangentBase=20;
    CHECK(fiberChunkCacheSpecification(filter,trace,1,"synthetic")==fiberChunkCacheSpecification(config,trace,1,"synthetic"));
    auto refined=config;refined.refineSeed=true;refined.seedMovementBase=8;
    CHECK(fiberChunkCacheSpecification(refined,trace,1,"synthetic")!=fiberChunkCacheSpecification(config,trace,1,"synthetic"));
}

TEST_CASE("Recovered angular initial match penalizes a bending candidate") {
    FiberChunkPaths paths;
    paths.lines={{{0,0,0},{32,0,0},{64,0,0},{96,12,0},{128,24,0}},
                 {{0,2,0},{32,2,0},{64,2,0},{96,2,0},{128,2,0}}};
    FiberChunkInitialRegion region;region.center={64,0,0};region.metric=cv::Matx33d::eye()*(1./400);
    FiberChunkTraceConfig cfg;cfg.initialAlignmentPower=2;
    auto match=matchFiberChunkPaths(paths,{{64,0,0}},{1,0,0},cfg,true,&region);
    CHECK(match.valid);CHECK(match.line==1);CHECK(match.initialSamples==5);
}

TEST_CASE("saved conflict flags retain geometry and matching ignores rejected paths") {
    Temp dir;std::filesystem::create_directories(dir.path);
    auto c=chunkConfig();c.coreBase=512;c.haloBase=64;
    c.conflictNormalBase=10;c.conflictTangentBase=20;
    auto t=traceConfig();Normal normal;Tubes source;
    nlohmann::json lines=nlohmann::json::array();
    for(int line=0;line<3;++line) {
        nlohmann::json points=nlohmann::json::array();
        for(int x=0;x<=320;x+=8)points.push_back({x,line==0?0.:line==1?std::max(0,x-128)*.5:100.,0.});
        lines.push_back(points);
    }
    nlohmann::json doc={{"key",{0,0,0}},{"lo",{-64,-64,-64}},{"hi",{576,576,576}},
        {"maxima",3},{"seeds",3},{"lines",lines},
        {"end_reasons",std::vector<std::array<std::string,2>>(3,{"chunk_boundary","chunk_boundary"})},
        {"specification",fiberChunkCacheSpecification(c,t,1,"flags-test")}};
    std::ofstream(dir.path/"0_0_0.json")<<doc;
    std::string outcome;
    auto paths=prepareFiberChunkPaths({0,0,0},source,t,normal,c,1,dir.path,"flags-test",outcome);
    CHECK(outcome=="annotated");CHECK(source.calls==0);
    CHECK(paths->conflictFlags==std::vector<bool>{true,true,false});
    CHECK(paths->lines[0].size()==41);CHECK(paths->lines[1].size()==41);
    auto loaded=loadFiberChunkPaths(dir.path/"0_0_0.msgpack.zst");
    CHECK(loaded->lines==paths->lines);CHECK(loaded->conflictFlags==paths->conflictFlags);
    FiberChunkInitialRegion region;region.center={64,0,0};region.metric=cv::Matx33d::eye()/400.;
    auto stale=c;stale.conflictSeparationBase=48;
    CHECK_THROWS(matchFiberChunkPaths(*loaded,{{64,0,0}},{1,0,0},stale,true,&region));
    CHECK_FALSE(matchFiberChunkPaths(*loaded,{{64,0,0}},{1,0,0},c,true,&region).valid);
    c.conflictNormalBase=c.conflictTangentBase=0;
    CHECK(matchFiberChunkPaths(*loaded,{{64,0,0}},{1,0,0},c,true,&region).valid);
    c.conflictNormalBase=10;c.conflictTangentBase=20;
    const auto modified=std::filesystem::last_write_time(dir.path/"0_0_0.msgpack.zst");
    paths=prepareFiberChunkPaths({0,0,0},source,t,normal,c,1,dir.path,"flags-test",outcome);
    CHECK(outcome=="reused");CHECK(source.calls==0);
    CHECK(std::filesystem::last_write_time(dir.path/"0_0_0.msgpack.zst")==modified);
}
TEST_CASE("Original main cache defaults load without rewriting or reconstructing") {
    auto c=chunkConfig();const auto t=traceConfig();Normal normals;Tubes source;Temp dir;
    FiberChunkTraceCache writer(c,dir.path,"original-main");
    const auto built=writer.get({0,0,0},source,t,normals,1);
    const auto path=dir.path/"0_0_0.json";
    const auto packed=dir.path/"0_0_0.msgpack.zst";
    auto saved=readPackedFixture(packed);
    std::filesystem::remove(packed);
    for(const auto* key:{"refine_seed","seed_movement","backend","online_refinement"})
        saved["specification"]["chunk"].erase(key);
    const auto bytes=saved.dump();std::ofstream(path)<<bytes;
    c.cacheOnly=true;FiberChunkTraceCache reader(c,dir.path,"original-main");
    const auto calls=source.calls.load();
    CHECK(reader.get({0,0,0},source,t,normals,1)->lines==built->lines);
    CHECK(source.calls==calls);
    std::ifstream in(path);std::string after((std::istreambuf_iterator<char>(in)),{});
    CHECK(after==bytes);
}

TEST_CASE("real cached geometry survives MessagePack Zstd and rejects truncation") {
    const char* fixture=std::getenv("VC_TEST_CHUNK_PATH");
    if(!fixture)return;
    const auto original=loadFiberChunkPaths(fixture);
    std::ifstream input(fixture);nlohmann::json document;input>>document;
    const auto bytes=nlohmann::json::to_msgpack(document);
    std::string compressed(ZSTD_compressBound(bytes.size()),'\0');
    const auto size=ZSTD_compress(compressed.data(),compressed.size(),bytes.data(),bytes.size(),3);
    REQUIRE_FALSE(ZSTD_isError(size));compressed.resize(size);
    Temp dir;std::filesystem::create_directories(dir.path);
    const auto path=dir.path/"fixture.msgpack.zst";
    std::ofstream(path,std::ios::binary).write(compressed.data(),compressed.size());
    const auto decoded=loadFiberChunkPaths(path);
    CHECK(decoded->lines==original->lines);CHECK(decoded->endReasons==original->endReasons);
    CHECK(decoded->key==original->key);CHECK(decoded->lo==original->lo);CHECK(decoded->hi==original->hi);
    CHECK(decoded->seeds==original->seeds);CHECK(decoded->maxima==original->maxima);
    CHECK(decoded->timings==original->timings);
    CHECK(readPackedFixture(path)==document);
    compressed.pop_back();
    std::ofstream(path,std::ios::binary|std::ios::trunc).write(compressed.data(),compressed.size());
    CHECK_THROWS(loadFiberChunkPaths(path));
}
