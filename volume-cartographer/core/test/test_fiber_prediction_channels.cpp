#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/detail/FiberChunkTensor.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "utils/zarr.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace vc::fiber_tracer;
namespace fs = std::filesystem;

namespace {
void checkCombined(const detail::FiberChunkTensor& tensor,const std::vector<cv::Vec3f>& points)
{
    const std::vector<cv::Vec3f> dirs(points.size(),{.2f,-.7f,.4f});
    std::vector<FiberPredictionSample> predictions(points.size());
    for(auto& p:predictions)for(int i=0;i<3;++i)p.options.push_back({}); // Reused storage.
    std::vector<vc::lasagna::NormalSample> normals;
    std::vector<float> widths,expected;
    for(bool withNormals:{true,false})for(bool withWidths:{true,false}){
        tensor.sampleCandidateChannels(points,dirs,withNormals,withWidths,predictions,normals,widths);
        detail::FiberSamplingProfile profile;
        std::vector<FiberPredictionSample> measured;
        std::vector<vc::lasagna::NormalSample> measuredNormals;
        std::vector<float> measuredWidths;
        for(int repeat=0;repeat<2;++repeat){
            tensor.sampleCandidateChannels(points,dirs,withNormals,withWidths,measured,measuredNormals,measuredWidths,&profile);
            CHECK(profile.calls==repeat+1);
            int64_t sum=0;for(const auto ns:profile.nanoseconds){CHECK(ns>=0);sum+=ns;}
            CHECK(sum==profile.elapsedNanoseconds);
            CHECK(measuredWidths==widths);CHECK(measuredNormals.size()==normals.size());
            for(size_t i=0;i<points.size();++i){
                CHECK(measured[i].options[0].valid==predictions[i].options[0].valid);
                CHECK(measured[i].options[0].presence==predictions[i].options[0].presence);
                CHECK(measured[i].options[0].direction==predictions[i].options[0].direction);
                if(withNormals){CHECK(measuredNormals[i].valid==normals[i].valid);CHECK(measuredNormals[i].normal==normals[i].normal);}
            }
            if(!withWidths)CHECK(profile.entries[unsigned(detail::SamplingStage::WidthInterpolation)]==0);
            if(!withNormals)CHECK(profile.entries[unsigned(detail::SamplingStage::NormalTensor)]==0);
        }
        CHECK(normals.size()==(withNormals?points.size():0));
        CHECK(widths.size()==(withWidths?points.size():0));
        tensor.sampleWidths(points,expected);
        for(size_t i=0;i<points.size();++i){
            const auto old=tensor.sample(cv::Vec3d(points[i]),cv::Vec3d(dirs[i]));
            REQUIRE(predictions[i].options.size()==1);
            CHECK(predictions[i].options[0].valid==old.options[0].valid);
            CHECK(predictions[i].options[0].presence==old.options[0].presence);
            CHECK(predictions[i].options[0].direction==old.options[0].direction);
            if(withNormals){
                const auto n=tensor.sampleNormal(cv::Vec3d(points[i]));
                CHECK(normals[i].valid==n.valid);CHECK(normals[i].normal==n.normal);
            }
            if(withWidths)CHECK(widths[i]==expected[i]);
        }
    }
    CHECK_THROWS(tensor.sampleCandidateChannels(points,{},true,true,predictions,normals,widths));
    tensor.sampleCandidateChannels({}, {},true,true,predictions,normals,widths);
    CHECK(predictions.empty());CHECK(normals.empty());CHECK(widths.empty());
    detail::FiberSamplingProfile empty;
    tensor.sampleCandidateChannels({}, {},true,true,predictions,normals,widths,&empty);
    CHECK(empty.calls==1);
    CHECK(empty.nanoseconds[0]==empty.elapsedNanoseconds);
    CHECK_THROWS(tensor.sampleCandidateChannels(points,{},true,true,predictions,normals,widths,&empty));
    int64_t sum=0;for(auto ns:empty.nanoseconds)sum+=ns;
    CHECK(sum==empty.elapsedNanoseconds);
}

struct Fixture {
    fs::path path = fs::temp_directory_path() /
        ("vc_prediction_channels_" + std::to_string(std::mt19937_64(std::random_device{}())()));
    nlohmann::json groups;
    Fixture() {
        fs::create_directories(path);
        channel<uint8_t>("presence", {0,255,0,255,0,255,0,255}, utils::ZarrDtype::uint8);
        channel<uint8_t>("nx", std::vector<uint8_t>(8,128), utils::ZarrDtype::uint8);
        channel<uint8_t>("ny", std::vector<uint8_t>(8,128), utils::ZarrDtype::uint8);
    }
    ~Fixture() { std::error_code error; fs::remove_all(path, error); }
    template<class T>
    void channel(const std::string& name, const std::vector<T>& values,
                 utils::ZarrDtype dtype, int scale = 0) {
        utils::ZarrMetadata metadata;
        metadata.version = utils::ZarrVersion::v2;
        metadata.shape = {2,2,2}; metadata.chunks = {2,2,2};
        metadata.dtype = dtype; metadata.compressor_id.clear(); metadata.fill_value = 0;
        auto array = utils::ZarrArray::create(path/name, metadata);
        const std::array<size_t,3> index{0,0,0};
        array.write_chunk(index, std::as_bytes(std::span(values)));
        groups[name] = {{"zarr", name}, {"scaledown", scale}, {"channels", {name}}};
    }
    void auxiliary() {
        channel<float>("embedding_000", {-2,4,-2,4,-2,4,-2,4}, utils::ZarrDtype::float32);
        channel<float>("embedding_001", std::vector<float>(8,1), utils::ZarrDtype::float32);
        channel<float>("width", {0,20,0,20,0,20,0,20}, utils::ZarrDtype::float32);
    }
    vc::lasagna::LasagnaDataset dataset() const {
        return vc::lasagna::LasagnaDataset(vc::lasagna::LasagnaDatasetManifest::parseText(
            nlohmann::json{{"version",2}, {"source_to_base",1},
                {"grad_mag_encode_scale",1000}, {"grad_mag_factor",1},
                {"groups",groups}}.dump(), path/"manifest.json"));
    }
};

FiberPredictionFieldOptions auxiliaryOptions() {
    FiberPredictionFieldOptions options;
    options.enableEmbeddings = true;
    options.enableWidth = true;
    return options;
}
}

TEST_CASE("direct sampling window preserves traces and falls back for incompatible sources") {
    for(bool byteWidth:{false,true}) {
        Fixture fixture;
        fixture.channel<uint8_t>("presence",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
        fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
        if(byteWidth)fixture.channel<uint8_t>("width",std::vector<uint8_t>(8,20),utils::ZarrDtype::uint8);
        else fixture.channel<float>("width",std::vector<float>(8,20.125f),utils::ZarrDtype::float32);
        auto manifest=fixture.dataset().manifest();
        if(byteWidth)manifest.raw["width_encoding"]="uint8-base-voxels";
        vc::lasagna::LasagnaDataset dataset(manifest);
        FiberPredictionFieldOptions options;options.enableWidth=true;
        FiberPredictionField field(dataset,options),noWidth(dataset);
        vc::lasagna::LasagnaNormalSampler normals(dataset);
        CHECK(field.supportsDenseSampling(normals));CHECK_FALSE(noWidth.supportsDenseSampling(normals));
        int acknowledgements=0;
        field.setReadinessGate([&](const auto&){++acknowledgements;});
        FiberTraceConfig config;config.parallelThreads=1;config.stepVoxels=.1;
        config.coneAngleDegrees=5;config.widthRelativeTolerance=.3;
        auto baseline=traceFiberExtrapolation(field,{.5,.5,.2},{0,0,1},.6,config,&normals);
        FiberTraceProfile profile;config.profile=&profile;config.samplingWindowVoxels=8;
        const auto actual=traceFiberExtrapolation(field,{.5,.5,.2},{0,0,1},.6,config,&normals);
        CHECK(actual.points==baseline.points);CHECK(actual.widthMean==baseline.widthMean);
        CHECK(actual.reason==baseline.reason);CHECK(profile.samplingWindowLoads==1);
        CHECK(profile.samplingWindowFallbacks==0);CHECK(acknowledgements>0);
        CHECK(profile.samplingWindowCopiedBytes==0);
        CHECK(profile.samplingWindowPeakBytes>0);
        CHECK(profile.samplingWindowPeakTableBytes>0);
        config.widthRelativeTolerance=0;profile={};
        (void)traceFiberExtrapolation(noWidth,{.5,.5,.2},{0,0,1},.6,config,&normals);
        CHECK(profile.samplingWindowLoads==0);CHECK(profile.samplingWindowFallbacks>0);
        config.samplingWindowVoxels=7;CHECK_THROWS(validateFiberTraceConfig(config));
        config.samplingWindowVoxels=257;CHECK_THROWS(validateFiberTraceConfig(config));
    }
}

TEST_CASE("sampling window refills across chunks and preserves oversized and pinned fallback") {
    Fixture fixture;
    for(const std::string name:{"presence","nx","ny","grad_mag","width"}) {
        utils::ZarrMetadata meta;meta.version=utils::ZarrVersion::v2;
        meta.shape={80,16,16};meta.chunks={16,8,8};meta.dtype=utils::ZarrDtype::uint8;
        meta.compressor_id.clear();meta.fill_value=0;
        auto array=utils::ZarrArray::create(fixture.path/name,meta);
        const uint8_t value=(name=="nx"||name=="ny")?128:(name=="width"?20:255);
        std::vector<std::byte> bytes(16*8*8,std::byte(value));
        for(size_t z=0;z<5;++z)for(size_t y=0;y<2;++y)for(size_t x=0;x<2;++x)
            array.write_chunk(std::array<size_t,3>{z,y,x},bytes);
        fixture.groups[name]={{"zarr",name},{"scaledown",0},{"channels",{name}}};
    }
    auto manifest=fixture.dataset().manifest();manifest.raw["width_encoding"]="uint8-base-voxels";
    vc::lasagna::LasagnaDataset dataset(manifest);
    FiberPredictionFieldOptions options;options.enableWidth=true;
    FiberPredictionField field(dataset,options),other(dataset,options);
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    FiberTraceConfig config;config.parallelThreads=1;config.stepVoxels=1;
    config.coneAngleDegrees=5;config.beamWidth=2;config.widthRelativeTolerance=.3;
    const auto trace=[&]{return traceFiberExtrapolation(field,{8,8,2},{0,0,1},50,config,&normals);};
    const auto baseline=trace();FiberTraceProfile profile;
    config.profile=&profile;config.samplingWindowVoxels=8;
    const auto window=trace();CHECK(window.points==baseline.points);CHECK(window.reason==baseline.reason);
    CHECK(profile.samplingWindowLoads>1);CHECK(profile.samplingWindowFallbacks==0);
    config.stepVoxels=8;config.coneAngleDegrees=25;config.samplingWindowVoxels=0;
    const auto wide=trace();config.samplingWindowVoxels=8;profile={};
    const auto fallback=trace();CHECK(fallback.points==wide.points);CHECK(fallback.reason==wide.reason);
    CHECK(profile.samplingWindowFallbacks>0);
    auto pinned=field.pinRegion({0,0,0},{15,15,79});pinned->seal();
    config.pinnedRegion=pinned.get();profile={};
    const auto fixed=trace();CHECK(fixed.points==wide.points);CHECK(profile.samplingWindowLoads==0);
    CHECK(profile.samplingWindowFallbacks>0);
    auto wrong=other.pinRegion({0,0,0},{15,15,79});wrong->seal();config.pinnedRegion=wrong.get();
    CHECK_THROWS(trace());
}

TEST_CASE("dense chunk tensor matches scalar and batch interpolation with masked byte and float widths") {
    for(bool byteWidth:{false,true})for(bool masked:{false,true})for(bool retained:{false,true}) {
        Fixture fixture;
        fixture.channel<uint8_t>("presence",{0,255,203,204,205,255,128,0},utils::ZarrDtype::uint8);
        fixture.channel<uint8_t>("nx",{128,129,0,255,50,170,128,128},utils::ZarrDtype::uint8);
        fixture.channel<uint8_t>("ny",{128,200,0,255,170,100,128,128},utils::ZarrDtype::uint8);
        fixture.channel<uint8_t>("grad_mag",{0,255,0,255,255,0,255,255},utils::ZarrDtype::uint8);
        if(byteWidth)fixture.channel<uint8_t>("width",{0,20,40,80,120,200,255,0},utils::ZarrDtype::uint8);
        else fixture.channel<float>("width",{0,20.125f,-4.5f,80,120,200,255,0},utils::ZarrDtype::float32);
        auto manifest=fixture.dataset().manifest();
        manifest.workingToBaseScale=2;
        manifest.raw["mask_invalid_normal_corners"]=masked;
        if(byteWidth)manifest.raw["width_encoding"]="uint8-base-voxels";
        vc::lasagna::LasagnaDataset dataset(manifest);
        FiberPredictionFieldOptions options;options.enableWidth=true;
        FiberPredictionField field(dataset,options);vc::lasagna::LasagnaNormalSampler normals(dataset);
        detail::FiberChunkTensor dense(field,normals,{-1,-1,-1},{2,2,2},false,true,0,retained);
        if(retained){CHECK(dense.ownedVoxelBytes()==0);CHECK(dense.retainedVoxelBytes()>0);CHECK(dense.pointerTableBytes()>0);}
        const double s=field.chunkRegionChannels({0,0,0},{1,1,1}).first[0].spacing;
        std::vector<cv::Vec3d> points{{0,0,0},{s,s,s},{-.1,0,0},{s+.01,0,0},{2*s,0,0}};
        std::mt19937 rng(35);std::uniform_real_distribution<double> dist(0,s);
        for(int i=0;i<100;++i)points.emplace_back(dist(rng),dist(rng),dist(rng));
        std::vector<cv::Vec3f> candidatePoints;for(const auto& p:points)candidatePoints.emplace_back(p);
        checkCombined(dense,candidatePoints);
        for(const auto& p:points) {
            const cv::Vec3d dir{.2,-.7,.4};
            const auto a=field.sample(p,dir),b=dense.sample(p,dir);
            REQUIRE(a.options.size()==b.options.size());
            CHECK(a.options[0].valid==b.options[0].valid);
            CHECK(a.options[0].presence==b.options[0].presence);
            CHECK(a.options[0].direction==b.options[0].direction);
            for (int workers : {1, 4}) {
                std::vector<FiberPredictionSample> exact;
                field.sampleBatchScalarEquivalent({p,p},{dir,dir},workers,exact,nullptr);
                REQUIRE(exact.size()==2);
                for (const auto& value : exact) {
                    CHECK(value.options[0].valid==a.options[0].valid);
                    CHECK(value.options[0].presence==a.options[0].presence);
                    CHECK(value.options[0].direction==a.options[0].direction);
                }
                std::vector<vc::lasagna::NormalSampleWithDerivative> exactNormals;
                (void)normals.sampleNormalBatchScalarEquivalent({p,p},workers,exactNormals);
                const auto scalarNormal=normals.sampleNormal(p);
                for (const auto& value : exactNormals) {
                    CHECK(value.sample.valid==scalarNormal.valid);
                    if (scalarNormal.valid)
                        CHECK(value.sample.normal==scalarNormal.normal);
                }
            }
            std::vector<FiberPredictionSample> aa,bb;
            dense.FiberPredictionSource::sampleBatch({p},{dir},1,bb);
            CHECK(bb[0].options[0].valid==a.options[0].valid);
            CHECK(bb[0].options[0].presence==a.options[0].presence);
            CHECK(bb[0].options[0].direction==a.options[0].direction);
            field.sampleBatch({p},{dir},1,aa);dense.sampleBatch({p},{dir},1,bb);
            CHECK(aa[0].options[0].valid==bb[0].options[0].valid);
            CHECK(aa[0].options[0].presence==bb[0].options[0].presence);
            CHECK(aa[0].options[0].direction==bb[0].options[0].direction);
            field.sampleBatch({p},{{1,0,0}},1,aa);
            const auto old=aa[0].options[0];const float value=dense.seedPresence(p,.8);
            CHECK(value==(old.valid&&old.presence>=.8?old.presence:0));
            std::vector<float> wa,wb;field.sampleWidths({cv::Vec3f(p)},wa);dense.sampleWidths({cv::Vec3f(p)},wb);
            CHECK(wa==wb);
            const auto na=normals.sampleNormal(p),nb=dense.sampleNormal(p);
            CHECK(na.valid==nb.valid);if(na.valid)CHECK(na.normal==nb.normal);
        }
    }
}

TEST_CASE("GPU packing pads all physical boundary faces without changing interior samples") {
    for(bool byteWidth:{false,true}) {
        Fixture fixture;
        fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
        if(byteWidth)fixture.channel<uint8_t>("width",std::vector<uint8_t>(8,20),utils::ZarrDtype::uint8);
        else fixture.channel<float>("width",std::vector<float>(8,20.125f),utils::ZarrDtype::float32);
        auto manifest=fixture.dataset().manifest();
        if(byteWidth)manifest.raw["width_encoding"]="uint8-base-voxels";
        vc::lasagna::LasagnaDataset dataset(manifest);
        FiberPredictionFieldOptions options;options.enableWidth=true;
        FiberPredictionField field(dataset,options);vc::lasagna::LasagnaNormalSampler normals(dataset);
        detail::FiberChunkTensor interior(field,normals,{0,0,0},{1,1,1});
        std::vector<unsigned char> reference;std::vector<float> referenceWidths;
        interior.packGpu(reference,referenceWidths);
        // Includes all six faces, edges/corners, and an entirely external ROI.
        for(const auto lo:{cv::Vec3d{-2,-2,-2},cv::Vec3d{3,3,3}}) {
            detail::FiberChunkTensor padded(field,normals,lo,lo+cv::Vec3d::all(5));
            std::vector<unsigned char> bytes(216*6,123);std::vector<float> widths(216,99);
            padded.packGpu(bytes,widths);
            REQUIRE(bytes.size()==216*6);REQUIRE(widths.size()==216);
            size_t i=0;
            for(int z=0;z<6;++z)for(int y=0;y<6;++y)for(int x=0;x<6;++x,++i){
                const int xx=int(lo[0])+x,yy=int(lo[1])+y,zz=int(lo[2])+z;
                const bool inside=xx>=0&&xx<2&&yy>=0&&yy<2&&zz>=0&&zz<2;
                const size_t j=inside?size_t((zz*2+yy)*2+xx):0;
                for(size_t c=0;c<6;++c)CHECK(bytes[6*i+c]==(inside?reference[6*j+c]:0));
                CHECK(widths[i]==(inside?referenceWidths[j]:0));
            }
        }
    }
}

TEST_CASE("sparse regular seed grid exactly matches independent point interpolation") {
    for(int pattern=0;pattern<4;++pattern)for(double spacing:{1.,1.7}) {
        Fixture fixture;
        fixture.channel<uint8_t>("presence",pattern==0?std::vector<uint8_t>{0,255,203,204,205,255,128,0}:
            std::vector<uint8_t>(8,pattern==1?255:pattern==2?0:204),utils::ZarrDtype::uint8);
        fixture.channel<uint8_t>("nx",pattern==1?std::vector<uint8_t>(8,128):
            std::vector<uint8_t>{128,129,0,255,50,170,128,128},utils::ZarrDtype::uint8);
        fixture.channel<uint8_t>("ny",pattern==1?std::vector<uint8_t>(8,128):
            std::vector<uint8_t>{128,200,0,255,170,100,128,128},utils::ZarrDtype::uint8);
        fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
        fixture.auxiliary();
        auto manifest=fixture.dataset().manifest();manifest.sourceToBase=spacing;
        const vc::lasagna::LasagnaDataset dataset(manifest);
        FiberPredictionFieldOptions options;options.enableWidth=true;
        FiberPredictionField field(dataset,options);
        vc::lasagna::LasagnaNormalSampler normals(dataset);
        detail::FiberChunkTensor reference(field,normals,{-2,-2,-2},{4,4,4});
        const std::vector<double> thresholds{0,.8,double(std::nextafter(.8f,0.f)),
            double(std::nextafter(.8f,1.f)),1};
        for(double threshold:thresholds)for(double scale:{1.,2.,1.7})for(double grid:{.125,.137}) {
            detail::FiberChunkTensor candidate(field,normals,{-2,-2,-2},{4,4,4},true);
            CHECK_THROWS(candidate.seedPresence({0,0,0},threshold));
            const cv::Vec3d lo{-.123,-.25,0};const int side=20;
            std::vector<float> actual;
            const auto stats=candidate.seedGrid(lo,scale,grid,side,threshold,actual);
            REQUIRE(actual.size()==size_t(side*side*side));
            for(int z=0;z<side;++z)for(int y=0;y<side;++y)for(int x=0;x<side;++x){
                const cv::Vec3d point=(lo+cv::Vec3d(x*grid,y*grid,z*grid))/scale;
                CHECK(actual[(z*side+y)*side+x]==reference.seedPresence(point,threshold));
            }
            if(pattern==2 && threshold>0) {
                CHECK(stats["interpolated_points"]==0);
                CHECK(stats["presence_eligible_points"]==0);
                std::vector<unsigned char> bytes;std::vector<float> widths;
                CHECK_THROWS(candidate.packGpu(bytes,widths));
            }
        }
    }
}

TEST_CASE("dense ROI respects packed channel offsets and missing chunks") {
    Fixture fixture;
    utils::ZarrMetadata meta;meta.version=utils::ZarrVersion::v2;
    meta.shape={7,8,4,4};meta.chunks={7,4,4,4};meta.dtype=utils::ZarrDtype::uint8;
    meta.compressor_id.clear();meta.fill_value=0;
    auto array=utils::ZarrArray::create(fixture.path/"packed",meta);
    std::vector<uint8_t> data(7*64);
    const std::array<uint8_t,7> constants{255,128,200,40,180,130,255};
    for(size_t c=0;c<7;++c)std::fill(data.begin()+c*64,data.begin()+(c+1)*64,constants[c]);
    array.write_chunk(std::vector<size_t>{0,0,0,0},std::as_bytes(std::span(data)));
    const auto manifest=vc::lasagna::LasagnaDatasetManifest::parseText(nlohmann::json{
        {"version",3},{"source_to_base",1},{"base_shape_zyx",{8,4,4}},
        {"width_encoding","uint8-base-voxels"},{"validity_channel","presence"},
        {"groups",{{"sd0",{{"zarr","packed"},{"scaledown",0},
        {"channels",{"presence","nx","ny","width","sheet_nx","sheet_ny","sheet_grad_mag"}}}}}}}.dump(),fixture.path/"manifest.json");
    vc::lasagna::LasagnaDataset dataset(manifest);
    FiberPredictionFieldOptions options;options.enableWidth=true;
    FiberPredictionField field(dataset,options);
    vc::lasagna::LasagnaNormalSampler normals(vc::lasagna::LasagnaDataset(vc::lasagna::fiberNormalManifest(manifest)));
    detail::FiberChunkTensor dense(field,normals,{0,0,0},{3,3,7});
    std::vector<unsigned char> bytes;std::vector<float> widths;dense.packGpu(bytes,widths);
    REQUIRE(bytes.size()==6*128);REQUIRE(widths.size()==128);
    const std::array<int,6> order{0,1,2,6,4,5};
    for(size_t i=0;i<128;++i){
        for(size_t c=0;c<6;++c)CHECK(bytes[i*6+c]==(i<64?constants[order[c]]:0));
        CHECK(widths[i]==(i<64?40:0));
    }
    CHECK(dense.sampleNormal({1.1,1.2,1.3}).normal==normals.sampleNormal({1.1,1.2,1.3}).normal);
    detail::FiberChunkTensor retained(field,normals,{0,0,0},{3,3,7},false,true,0,true);
    CHECK(retained.ownedVoxelBytes()==0);
    CHECK(retained.retainedVoxelBytes()==7*64); // Packed allocation counted once.
    for(double z:{0.,1.3,3.,3.5,4.,7.}) {
        const cv::Vec3d point{1.1,1.2,z};
        const auto a=retained.sample(point,{0,0,1}),b=dense.sample(point,{0,0,1});
        CHECK(a.options[0].presence==b.options[0].presence);
        CHECK(a.options[0].direction==b.options[0].direction);
        CHECK(retained.sampleNormal(point).normal==dense.sampleNormal(point).normal);
        const std::vector<cv::Vec3f> points{cv::Vec3f(point)};
        checkCombined(retained,points);
        std::vector<float> expected,actual;
        dense.sampleWidths(points,expected);retained.sampleWidths(points,actual);
        CHECK(actual==expected);
    }
    // The last exported lattice point is not a volume edge: width sampling
    // still needs its upper interpolation corner outside the exported ROI.
    detail::FiberChunkTensor interior(field,normals,{1,1,1},{2,2,2});
    interior.packGpu(bytes,widths);
    REQUIRE(bytes.size()==6*8);
    REQUIRE(widths.size()==8);
    for(float value:widths)CHECK(value==40);
}

TEST_CASE("dense ROI allows independently chunked widths") {
    Fixture fixture;
    fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
    utils::ZarrMetadata meta;meta.version=utils::ZarrVersion::v2;
    meta.shape={2,2,2};meta.chunks={1,1,1};meta.dtype=utils::ZarrDtype::float32;
    meta.compressor_id.clear();meta.fill_value=0;
    auto array=utils::ZarrArray::create(fixture.path/"width",meta);
    for(size_t z=0;z<2;++z)for(size_t y=0;y<2;++y)for(size_t x=0;x<2;++x){
        const std::array<size_t,3> key{z,y,x};
        const std::array<float,1> value{float(10+z*4+y*2+x)};
        array.write_chunk(key,std::as_bytes(std::span(value)));
    }
    fixture.groups["width"]={{"zarr","width"},{"scaledown",0},{"channels",{"width"}}};
    const auto dataset=fixture.dataset();
    FiberPredictionFieldOptions options;options.enableWidth=true;
    FiberPredictionField field(dataset,options);
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    detail::FiberChunkTensor dense(field,normals,{0,0,0},{1,1,1});
    const std::vector<cv::Vec3f> points{{.2f,.3f,.4f},{1,1,1},{0,0,0}};
    std::vector<float> expected,actual;
    field.sampleWidths(points,expected);dense.sampleWidths(points,actual);
    CHECK(expected==actual);
    detail::FiberChunkTensor retained(field,normals,{0,0,0},{1,1,1},false,true,0,true);
    retained.sampleWidths(points,actual);CHECK(expected==actual);
    checkCombined(retained,points);
    checkCombined(dense,points);
}

TEST_CASE("retained chunk grid survives cache eviction and concurrent reads") {
    using namespace vc::render;
    Fixture fixture;fixture.auxiliary();
    FiberPredictionFieldOptions options;options.enableWidth=true;
    FiberPredictionField field(fixture.dataset(),options);
    const auto channels=field.chunkRegionChannels({0,0,0},{1,1,1}).first;
    auto partition=processChunkCacheService()->createDecodedPartition(1<<20);
    auto source=rebindChunkCache(channels[0].chunkSource,partition);
    auto owner=std::make_shared<PinnedChunkRegion>();
    owner->pin(source,{{0,0,0,0}});
    CHECK_THROWS(PinnedChunkGrid(owner,source,0,{0,0,0},{1,1,1}));
    owner->seal();
    PinnedChunkGrid grid(owner,source,0,{0,0,0},{1,1,1});
    owner.reset(); // The grid owns the lifetime, not its caller.
    partition->configureDecodedByteCapacity(1);
    CHECK(source->getChunkIfCached(0,0,0,0).status==ChunkStatus::MissQueued);
    const std::array<uint8_t,8> expected{0,255,0,255,0,255,0,255};
    std::vector<std::future<bool>> readers;
    for(int i=0;i<8;++i)readers.push_back(std::async(std::launch::async,[&]{
        for(int j=0;j<1000;++j)if(grid.corners<uint8_t>({0,0,0},{1,1,1})!=expected)return false;
        return true;
    }));
    for(auto& reader:readers)CHECK(reader.get());
    CHECK(source->getChunkIfCached(0,0,0,0).status==ChunkStatus::MissQueued);
    CHECK_THROWS_AS(grid.corners<uint8_t>({0,0,0},{2,1,1}),PinnedRegionMiss);
}

TEST_CASE("retained grid uses typed nonzero fills across partial edge chunks") {
    using namespace vc::render;
    for(bool floating:{false,true}) {
        Fixture fixture;
        utils::ZarrMetadata meta;meta.version=utils::ZarrVersion::v2;
        meta.shape={3,3,3};meta.chunks={2,2,2};meta.compressor_id.clear();
        meta.dtype=floating?utils::ZarrDtype::float32:utils::ZarrDtype::uint16;
        meta.fill_value=floating?17.25:1700.;
        auto array=std::make_shared<utils::ZarrArray>(utils::ZarrArray::create(fixture.path/"grid",meta));
        const std::array<size_t,3> key{0,0,0};
        if(floating){const std::array<float,8> data{1,2,3,4,5,6,7,8.5};array->write_chunk(key,std::as_bytes(std::span(data)));}
        else {const std::array<uint16_t,8> data{1,2,3,4,5,6,7,800};array->write_chunk(key,std::as_bytes(std::span(data)));}
        auto source=acquireProcessZarrChannel((fixture.path/"grid").string(),array,0);
        auto owner=std::make_shared<PinnedChunkRegion>();
        std::vector<ChunkKey> keys;
        for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x)keys.push_back({0,z,y,x});
        owner->pin(source,keys);owner->seal();
        PinnedChunkGrid grid(owner,source,0,{0,0,0},{2,2,2});
        const auto values=grid.corners<double>({1,1,1},{2,2,2});
        CHECK(values[0]==(floating?8.5:800.));
        for(size_t i=1;i<8;++i)CHECK(values[i]==(floating?17.25:1700.));
        CHECK_THROWS(grid.corners<uint8_t>({0,0,0},{1,1,1}));
        auto incomplete=std::make_shared<PinnedChunkRegion>();
        incomplete->pin(source,{{0,0,0,0}});incomplete->seal();
        CHECK_THROWS_AS(PinnedChunkGrid(incomplete,source,0,{0,0,0},{2,2,2}),PinnedRegionMiss);
    }
}

TEST_CASE("prepared chunk addresses share element offsets only across verified layouts") {
    using namespace vc::render;
    Fixture fixture;
    auto region=std::make_shared<PinnedChunkRegion>();
    std::array<std::shared_ptr<IChunkedArray>,3> sources;
    for(int c=0;c<3;++c){
        utils::ZarrMetadata meta;meta.version=utils::ZarrVersion::v2;
        meta.shape={4,2,2};meta.chunks={2,2,2};meta.compressor_id.clear();
        meta.dtype=c==0?utils::ZarrDtype::uint8:c==1?utils::ZarrDtype::uint16:utils::ZarrDtype::float32;
        meta.fill_value=c==0?17.:c==1?1700.:17.25;
        const auto path=fixture.path/("prepared"+std::to_string(c));
        auto array=std::make_shared<utils::ZarrArray>(utils::ZarrArray::create(path,meta));
        const std::array<size_t,3> key{0,0,0};
        if(c==0){const std::array<uint8_t,8> v{1,2,3,4,5,6,7,8};array->write_chunk(key,std::as_bytes(std::span(v)));}
        if(c==1){const std::array<uint16_t,8> v{1,2,3,4,5,6,7,8};array->write_chunk(key,std::as_bytes(std::span(v)));}
        if(c==2){const std::array<float,8> v{1,2,3,4,5,6,7,8};array->write_chunk(key,std::as_bytes(std::span(v)));}
        sources[c]=acquireProcessZarrChannel(path.string(),array,0);
        region->pin(sources[c],{{0,0,0,0},{0,1,0,0}});
    }
    region->seal();
    std::array<std::unique_ptr<PinnedChunkGrid>,3> grids;
    for(int c=0;c<3;++c)grids[c]=std::make_unique<PinnedChunkGrid>(region,sources[c],0,
        std::array{0,0,0},std::array{1,1,3},c?grids[0].get():nullptr);
    const auto plan=grids[0]->prepareCorners({0,0,1},{1,1,2});
    for(int c=0;c<3;++c){
        CHECK(grids[c]->accepts(plan));
        const auto v=grids[c]->corners<double>(plan);
        for(size_t i=0;i<4;++i)CHECK(v[i]==5.+i);
        for(size_t i=4;i<8;++i)CHECK(v[i]==(c==0?17.:c==1?1700.:17.25));
    }
    PinnedChunkGrid firstOnly(region,sources[0],0,{0,0,0},{1,1,1});
    PinnedChunkGrid shifted(region,sources[0],0,{0,0,2},{1,1,3},&firstOnly);
    // Same volume/chunk/count, but a different first chunk must not share offsets.
    const auto firstPlan=firstOnly.prepareCorners({0,0,0},{1,1,1});
    CHECK_FALSE(shifted.accepts(firstPlan));
    CHECK_THROWS_AS(shifted.corners<double>(firstPlan),std::invalid_argument);
    CHECK_FALSE(shifted.accepts(plan));
    CHECK_THROWS_AS(shifted.corners<double>(plan),std::invalid_argument);
    CHECK_THROWS_AS(grids[0]->corners<double>(PinnedChunkGrid::CornerAddress{}),std::invalid_argument);
    const PinnedChunkGrid::CornerAddress empty;
    const auto emptyCopy=empty;
    CHECK_FALSE(grids[0]->accepts(emptyCopy));
    auto moved=std::move(shifted);
    CHECK_FALSE(shifted.accepts(emptyCopy));
    CHECK_THROWS_AS(shifted.corners<double>(emptyCopy),std::invalid_argument);
    grids[0].reset(); // A compatible peer retains the layout identity and payload.
    CHECK(grids[1]->corners<double>(plan)[0]==5.);
    std::vector<std::future<bool>> readers;
    for(int i=0;i<8;++i)readers.push_back(std::async(std::launch::async,[&]{
        for(int j=0;j<1000;++j)if(grids[2]->corners<double>(plan)[7]!=17.25)return false;
        return true;
    }));
    for(auto& reader:readers)CHECK(reader.get());
}

TEST_CASE("auxiliary channels are opt in and missing channels fail explicitly") {
    Fixture fixture;
    const auto dataset = fixture.dataset();
    FiberPredictionField legacy(dataset);
    std::vector<float> widths;
    std::vector<std::vector<float>> embeddings;
    CHECK_THROWS(legacy.sampleWidths({}, widths));
    CHECK_THROWS(legacy.sampleEmbeddings({}, embeddings));
    CHECK_THROWS(FiberPredictionField(dataset, auxiliaryOptions()));
    fixture.auxiliary();
    const auto auxiliaryDataset = fixture.dataset();
    FiberPredictionField stillDisabled(auxiliaryDataset);
    CHECK_THROWS(stillDisabled.sampleWidths({}, widths));
    CHECK_THROWS(stillDisabled.sampleEmbeddings({}, embeddings));
}

TEST_CASE("embeddings interpolate signed components before normalization") {
    Fixture fixture; fixture.auxiliary();
    const auto dataset = fixture.dataset();
    FiberPredictionField field(dataset, auxiliaryOptions());
    const FiberPredictionSource& source = field;
    std::vector<std::vector<float>> embeddings;
    source.sampleEmbeddings({{.5f,.5f,.5f}, {0,0,0}, {-1,0,0}}, embeddings);
    REQUIRE(embeddings.size() == 3);
    REQUIRE(embeddings[0].size() == 2);
    CHECK(embeddings[0][0] == doctest::Approx(std::sqrt(.5)));
    CHECK(embeddings[0][1] == doctest::Approx(std::sqrt(.5)));
    CHECK(embeddings[1][0] == doctest::Approx(-2/std::sqrt(5.)));
    CHECK(embeddings[2].empty());
    source.sampleEmbeddings({}, embeddings);
    CHECK(embeddings.empty());
}

TEST_CASE("width interpolation ignores zero-presence corners and retains stored units") {
    Fixture fixture; fixture.auxiliary();
    const auto dataset = fixture.dataset();
    FiberPredictionField field(dataset, auxiliaryOptions());
    std::vector<float> widths;
    field.sampleWidths({{.5f,.5f,.5f}, {0,0,0}, {1,1,1}, {-1,0,0}}, widths);
    REQUIRE(widths.size() == 4);
    CHECK(widths[0] == 20); // Not 10: zero-presence corners are excluded.
    CHECK(widths[1] == 0);
    CHECK(widths[2] == 20);
    CHECK(widths[3] == 0);
    field.sampleWidths({}, widths);
    CHECK(widths.empty());
}

TEST_CASE("auxiliary channels reject incorrect types, scales and numbering") {
    for (int invalid = 0; invalid < 4; ++invalid) {
        Fixture fixture;
        if (invalid == 0) {
            fixture.channel<uint8_t>("width", std::vector<uint8_t>(8,20), utils::ZarrDtype::uint8);
        } else {
            fixture.channel<float>("width", std::vector<float>(8,20), utils::ZarrDtype::float32, invalid == 1 ? 1 : 0);
        }
        fixture.channel<float>("embedding_000", std::vector<float>(8,1), utils::ZarrDtype::float32, invalid == 2 ? 1 : 0);
        fixture.channel<float>(invalid == 3 ? "embedding_002" : "embedding_001", std::vector<float>(8,1), utils::ZarrDtype::float32);
        CHECK_THROWS(FiberPredictionField(fixture.dataset(), auxiliaryOptions()));
    }
}

TEST_CASE("auxiliary sampling is concurrent and leaves ordinary prediction samples unchanged") {
    Fixture fixture; fixture.auxiliary();
    const auto dataset = fixture.dataset();
    FiberPredictionField field(dataset, auxiliaryOptions()), legacy(dataset);
    const auto expected = legacy.sample({.5,.5,.5}, {0,0,1});
    std::vector<std::future<void>> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.push_back(std::async(std::launch::async, [&] {
            std::vector<float> widths;
            std::vector<std::vector<float>> embeddings;
            for (int repeat = 0; repeat < 10; ++repeat) {
                field.sampleWidths({{.5f,.5f,.5f}}, widths);
                field.sampleEmbeddings({{.5f,.5f,.5f}}, embeddings);
                if (widths.at(0) != 20 || embeddings.at(0).size() != 2)
                    throw std::runtime_error("concurrent auxiliary sample differs");
            }
        }));
    }
    for (auto& worker : workers) worker.get();
    const auto actual = field.sample({.5,.5,.5}, {0,0,1});
    REQUIRE(actual.options.size() == expected.options.size());
    CHECK(actual.options[0].direction == expected.options[0].direction);
    CHECK(actual.options[0].presence == expected.options[0].presence);
    CHECK(actual.options[0].valid == expected.options[0].valid);
}

TEST_CASE("real prediction cache remains identical through old and options constructors") {
    const char* manifest = std::getenv("VC_TEST_CHANNEL_PREDICTION_MANIFEST");
    const char* anchors = std::getenv("VC_TEST_METRIC_ANCHORS");
    if (!manifest || !anchors) return;
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifest);
    FiberPredictionField legacy(dataset), field(dataset, FiberPredictionFieldOptions{});
    std::ifstream input(anchors);
    REQUIRE(input.good());
    const auto data = nlohmann::json::parse(input);
    size_t count = 0;
    for (const auto& cell : data.at("cells")) {
        for (const auto& component : cell.at("components")) {
            if (!component.at("retained").get<bool>()) continue;
            const auto& p = component.at("position_base_xyz");
            const cv::Vec3d point{p[0].get<double>(),p[1].get<double>(),p[2].get<double>()};
            const auto expected = legacy.sample(point, {0,0,1});
            const auto actual = field.sample(point, {0,0,1});
            REQUIRE(actual.options.size() == expected.options.size());
            REQUIRE(!actual.options.empty());
            for (size_t i=0; i<actual.options.size(); ++i) {
                CHECK(actual.options[i].direction == expected.options[i].direction);
                CHECK(actual.options[i].presence == expected.options[i].presence);
                CHECK(actual.options[i].valid == expected.options[i].valid);
            }
            ++count;
        }
    }
    CHECK(count > 0);
    std::cout << "Compared prediction field constructors at " << count << " real anchor positions\n";
}

TEST_CASE("auxiliary scoring preserves the fused normal corner path") {
    Fixture fixture; fixture.auxiliary();
    fixture.channel<uint8_t>("grad_mag", std::vector<uint8_t>(8,255), utils::ZarrDtype::uint8);
    const auto dataset=fixture.dataset();
    FiberPredictionField field(dataset,auxiliaryOptions());
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    FiberTraceConfig config;
    config.stepVoxels=.1;
    config.beamWidth=3;
    config.beamPruneDistanceVoxels=0;
    config.embeddingWeight=.3;
    config.widthRelativeTolerance=.4;
    config.widthReferenceMean=22;
    config.widthCosineWeight=true;
    config.parallelThreads=1;
    const auto serial=traceFiberExtrapolation(field,{.5,.5,.5},{0,0,1},.2,config,&normals);
    size_t gateCalls=0;
    field.setReadinessGate([&](const auto& dependencies) {
        CHECK(dependencies.size()==6);
        ++gateCalls;
    });
    config.parallelThreads=4;
    FiberTraceProfile profile; config.profile=&profile;
    const auto fused=traceFiberExtrapolation(field,{.5,.5,.5},{0,0,1},.2,config,&normals);
    REQUIRE(serial.points.size()>1);
    CHECK(serial.points==fused.points);
    CHECK(profile.cornerPointCount>0);
    CHECK(gateCalls>0);
}

TEST_CASE("readiness keeps independently chunked auxiliary arrays separate") {
    Fixture fixture; fixture.auxiliary();
    fs::remove_all(fixture.path/"width");
    utils::ZarrMetadata metadata;
    metadata.version=utils::ZarrVersion::v2;
    metadata.shape={2,2,2}; metadata.chunks={1,1,1};
    metadata.dtype=utils::ZarrDtype::float32; metadata.compressor_id.clear();
    (void)utils::ZarrArray::create(fixture.path/"width",metadata);
    FiberPredictionField field(fixture.dataset(),auxiliaryOptions());
    field.setReadinessGate([&](const auto& dependencies) {
        CHECK(dependencies.size()==13); // Five single-chunk arrays plus eight width chunks.
        CHECK(std::count_if(dependencies.begin(),dependencies.end(),[](const auto& d) {
            return d.arrayPath.filename()=="width";
        })==8);
        throw std::runtime_error("width not published");
    });
    CHECK_THROWS(field.sample({.5,.5,.5},{0,0,1}));
}

TEST_CASE("readiness covers every sampling entrypoint and enabled channel") {
    Fixture fixture; fixture.auxiliary();
    fixture.channel<uint8_t>("grad_mag", std::vector<uint8_t>(8,255), utils::ZarrDtype::uint8);
    const auto dataset = fixture.dataset();
    FiberPredictionField field(dataset, auxiliaryOptions());
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    int calls = 0;
    field.setReadinessGate([&](const auto& dependencies) {
        ++calls;
        REQUIRE(dependencies.size() == 6);
        std::vector<std::string> names;
        for (const auto& dependency : dependencies) {
            names.push_back(dependency.arrayPath.filename().string());
            CHECK(dependency.indexZYX == std::array<uint32_t,3>{0,0,0});
        }
        CHECK(names == std::vector<std::string>{"embedding_000","embedding_001","nx","ny","presence","width"});
        throw std::runtime_error("not published");
    });
    std::vector<FiberPredictionSample> samples;
    std::vector<float> widths;
    std::vector<std::vector<float>> embeddings;
    std::vector<FiberStoredPredictionSample> stored;
    std::vector<FiberStoredPresenceSample> presence;
    vc::lasagna::LasagnaCornerBatch corners;
    const std::vector<cv::Vec3d> points{{.5,.5,.5}}, directions{{0,0,1}};
    const std::vector<cv::Vec3f> floatPoints{{.5f,.5f,.5f}}, floatDirections{{0,0,1}};
    CHECK_THROWS(field.sample({.5,.5,.5},{0,0,1}));
    CHECK_THROWS(field.prefetchSamples(points));
    CHECK_THROWS(field.sampleBatch(points,directions,4,samples));
    CHECK_THROWS(field.sampleBatch(points,directions,4,samples,nullptr));
    CHECK_THROWS(field.sampleBatch(floatPoints,floatDirections,4,samples,nullptr));
    CHECK_THROWS(field.sampleWidths(floatPoints,widths));
    CHECK_THROWS(field.sampleEmbeddings(floatPoints,embeddings));
    CHECK_THROWS(field.sampleStoredGridBatch({{0,0,0}},4,stored));
    CHECK_THROWS(field.sampleStoredPresenceBatch({{0,0,0}},4,presence));
    CHECK_THROWS(field.sampleCornerBatchWithNormals(normals,floatPoints,4,&corners,nullptr));
    CHECK_THROWS(field.visitCornerBatchWithNormals(normals,floatPoints,4,nullptr,nullptr,1,nullptr));
    CHECK(calls == 11);
    CHECK_THROWS(field.scanStoredPresenceChunks(1));
    CHECK(calls == 11); // Whole-volume scans must not discover a partial live cache.
    field.sampleBatch(std::vector<cv::Vec3f>{},std::vector<cv::Vec3f>{},1,samples,nullptr);
    CHECK(calls == 11);
    CHECK_THROWS(field.sampleBatch(points,{},1,samples));
    CHECK(calls == 11);
    field.setReadinessGate({});
    CHECK(field.sample({.5,.5,.5},{0,0,1}).options[0].valid);
}

TEST_CASE("readiness plans both float and double boundary corners and deduplicates") {
    Fixture fixture;
    for (const auto* name : {"presence","nx","ny"}) {
        fs::remove_all(fixture.path/name);
        utils::ZarrMetadata metadata;
        metadata.version=utils::ZarrVersion::v2;
        metadata.shape={128,128,128}; metadata.chunks={64,64,64};
        metadata.dtype=utils::ZarrDtype::uint8; metadata.compressor_id.clear();
        (void)utils::ZarrArray::create(fixture.path/name,metadata);
    }
    auto manifest=vc::lasagna::LasagnaDatasetManifest::parseText(
        nlohmann::json{{"version",2},{"source_to_base",.01},{"groups",fixture.groups}}.dump(),fixture.path/"manifest.json");
    FiberPredictionField field{vc::lasagna::LasagnaDataset(std::move(manifest))};
    field.setReadinessGate([&](const auto& dependencies) {
        // float(.63)/float(.01)==63; promotion to double alone misses x=64.
        for (const auto* name : {"presence","nx","ny"}) {
            CHECK(std::any_of(dependencies.begin(),dependencies.end(),[&](const auto& d) {
                return d.arrayPath.filename()==name && d.indexZYX==std::array<uint32_t,3>{0,0,1};
            }));
        }
        throw std::runtime_error("not published");
    });
    std::vector<FiberPredictionSample> samples;
    CHECK_THROWS(field.sampleBatch(std::vector<cv::Vec3f>{{.63f,.1f,.1f}},
        std::vector<cv::Vec3f>{{1,0,0}},1,samples,nullptr));
    field.setReadinessGate([&](const auto& dependencies) {
        CHECK(dependencies.size()==3);
        for (const auto& d : dependencies) CHECK(d.indexZYX==std::array<uint32_t,3>{1,1,1});
        throw std::runtime_error("not published");
    });
    FiberTraceProfile profile;
    CHECK_THROWS(field.sampleBatch(std::vector<cv::Vec3f>(1000,{.8f,.8f,.8f}),
        std::vector<cv::Vec3f>(1000,{1,0,0}),1,samples,&profile));
    CHECK(profile.readinessPlanPoints==1000);
    CHECK(profile.readinessPlanDependencies==3);
}

TEST_CASE("readiness precedes reads and does not remember failed acknowledgements") {
    Fixture fixture;
    fs::remove(fixture.path/"presence"/"0.0.0");
    const auto dataset=fixture.dataset();
    FiberPredictionField field(dataset);
    int calls=0;
    field.setReadinessGate([&](const auto&) {
        if (++calls==1) throw std::runtime_error("cancelled publication");
        fixture.channel<uint8_t>("presence",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
    });
    CHECK_THROWS(field.sample({.5,.5,.5},{0,0,1}));
    CHECK(field.sample({.5,.5,.5},{0,0,1}).options[0].presence==1);
    CHECK(calls==2);
    std::atomic<int> concurrentCalls{0};
    field.setReadinessGate([&](const auto&) { ++concurrentCalls; });
    std::vector<std::future<float>> workers;
    for (int i=0;i<4;++i) workers.push_back(std::async(std::launch::async,[&] {
        return field.sample({.5,.5,.5},{0,0,1}).options[0].presence;
    }));
    for (auto& worker : workers) CHECK(worker.get()==1);
    CHECK(concurrentCalls==4);
}

TEST_CASE("real cached fiber samples are identical with publication gating") {
    const char* manifest=std::getenv("VC_TEST_WIDTH_MANIFEST");
    const char* fiberPath=std::getenv("VC_TEST_WIDTH_FIBER");
    if (!manifest || !fiberPath) return;
    const auto dataset=vc::lasagna::LasagnaDataset::open(manifest);
    const auto fiber=loadFiberJson(fiberPath);
    FiberPredictionFieldOptions options; options.enableWidth=true;
    FiberPredictionField offline(dataset,options), gated(dataset,options);
    size_t calls=0, dependencies=0;
    gated.setReadinessGate([&](const auto& requested) {
        ++calls; dependencies+=requested.size();
        for (const auto& d : requested) REQUIRE(fs::is_directory(d.arrayPath));
    });
    std::vector<cv::Vec3d> points;
    const size_t first=fiber.controlPointLineIndices.front();
    for (size_t i=first;i<std::min(first+32,fiber.linePointsXyzBase.size());++i)
        points.push_back(fiber.linePointsXyzBase[i]);
    const std::vector<cv::Vec3d> directions(points.size(),{1,0,0});
    std::vector<FiberPredictionSample> expected,actual;
    offline.sampleBatch(points,directions,1,expected);
    gated.sampleBatch(points,directions,4,actual);
    REQUIRE(actual.size()==expected.size());
    for (size_t i=0;i<actual.size();++i) {
        REQUIRE(actual[i].options.size()==expected[i].options.size());
        for (size_t j=0;j<actual[i].options.size();++j) {
            CHECK(actual[i].options[j].presence==expected[i].options[j].presence);
            CHECK(actual[i].options[j].direction==expected[i].options[j].direction);
            CHECK(actual[i].options[j].valid==expected[i].options[j].valid);
        }
    }
    CHECK(calls==1);
    CHECK(dependencies>0);
    std::cout << "Real Paris4 readiness: " << points.size() << " points, " << dependencies
              << " dependencies in one gate; offline/live samples exact\n";
    cv::Vec3d lo=points.front(),hi=lo;
    std::vector<cv::Vec3f> floatPoints;
    for (const auto& p : points) {
        for (int axis=0;axis<3;++axis) {lo[axis]=std::min(lo[axis],p[axis]);hi[axis]=std::max(hi[axis],p[axis]);}
        floatPoints.emplace_back(p[0],p[1],p[2]);
    }
    auto region=gated.pinRegion(lo,hi);
    CHECK(calls==2);
    region->seal();
    std::vector<float> expectedWidths,actualWidths;
    offline.sampleWidths(floatPoints,expectedWidths);
    gated.sampleWidths(floatPoints,actualWidths,*region);
    CHECK(actualWidths==expectedWidths);
    CHECK(calls==2);
    CHECK(region->bytes()>0);
    std::cout << "Real Paris4 pinned widths exact; retained bytes=" << region->bytes() << '\n';
}

TEST_CASE("pinned prediction regions publish once and sample auxiliary data concurrently") {
    Fixture fixture; fixture.auxiliary();
    const auto dataset=fixture.dataset();
    FiberPredictionField field(dataset,auxiliaryOptions()), other(dataset,auxiliaryOptions());
    int calls=0;
    field.setReadinessGate([&](const auto& dependencies) {
        ++calls;
        CHECK(dependencies.size()==6);
        for (const auto& dependency : dependencies)
            CHECK((dependency.indexZYX == std::array<uint32_t,3>{0,0,0}));
    });
    // Negative crop bounds must not wrap signed keys into unsigned indices.
    auto region=field.pinRegion({-1,-1,-1},{1,1,1});
    CHECK(calls==1);
    std::vector<float> widths;
    CHECK_THROWS(field.sampleWidths({{.5f,.5f,.5f}},widths,*region)); // seal before sharing
    region->seal();
    CHECK_THROWS(other.sampleWidths({{.5f,.5f,.5f}},widths,*region));
    std::vector<std::future<void>> workers;
    for (int i=0;i<4;++i) workers.push_back(std::async(std::launch::async,[&] {
        std::vector<float> w;
        std::vector<std::vector<float>> e;
        field.sampleWidths({{.5f,.5f,.5f}},w,*region);
        field.sampleEmbeddings({{.5f,.5f,.5f}},e,*region);
        if (w.at(0)!=20 || std::abs(e.at(0).at(0)-std::sqrt(.5))>1e-6)
            throw std::runtime_error("pinned samples differ");
    }));
    for (auto& worker:workers) worker.get();
    CHECK(calls==1);
    CHECK(region->bytes()>0);
    CHECK_THROWS(field.pinRegion({1,0,0},{0,0,0}));
    CHECK_THROWS(field.pinRegion({0,0,0},{std::numeric_limits<double>::infinity(),0,0}));
    field.setReadinessGate([](const auto&) {throw std::runtime_error("cancelled");});
    CHECK_THROWS(field.pinRegion({0,0,0},{1,1,1}));
}

TEST_CASE("explicit pinned fused visitor matches ordinary corners") {
    Fixture fixture; fixture.auxiliary();
    fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
    const auto dataset=fixture.dataset();
    FiberPredictionField field(dataset,auxiliaryOptions());
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    const std::vector<cv::Vec3f> points{{.5f,.5f,.5f},{.7f,.4f,.3f}};
    struct Results {
        std::array<std::vector<std::array<uint8_t,8>>,2> corners;
        std::array<cv::Vec3f,2> fractions;
        std::array<bool,2> valid;
    } expected,actual;
    const auto visitor=+[](void* ptr,size_t i,const cv::Vec3f& f,bool valid,
                          std::span<const std::array<uint8_t,8>> corners) {
        auto& out=*static_cast<Results*>(ptr);
        out.corners[i].assign(corners.begin(),corners.end());out.fractions[i]=f;out.valid[i]=valid;
    };
    REQUIRE(field.visitCornerBatchWithNormals(normals,points,1,&expected,visitor,1,nullptr));
    int calls=0;
    field.setReadinessGate([&](const auto&) {++calls;});
    auto region=field.pinRegion({0,0,0},{1,1,1});
    normals.pinRegion(*region,{0,0,0},{1,1,1});
    region->seal();
    REQUIRE(field.visitCornerBatchWithNormals(normals,points,4,&actual,visitor,1,nullptr,region.get()));
    CHECK(actual.corners==expected.corners);
    CHECK(actual.fractions==expected.fractions);
    CHECK(actual.valid==expected.valid);
    CHECK(calls==1);
}

TEST_CASE("pinned sampling outside retained chunks fails instead of fetching") {
    Fixture fixture; fixture.auxiliary();
    for (const auto* name : {"presence","nx","ny","width"}) {
        fs::remove_all(fixture.path/name);
        utils::ZarrMetadata metadata;
        metadata.version=utils::ZarrVersion::v2;
        metadata.shape={8,8,8};metadata.chunks={2,2,2};
        metadata.dtype=std::string(name)=="width" ? utils::ZarrDtype::float32 : utils::ZarrDtype::uint8;
        metadata.fill_value=0;metadata.compressor_id.clear();
        (void)utils::ZarrArray::create(fixture.path/name,metadata);
    }
    FiberPredictionFieldOptions options;options.enableWidth=true;
    FiberPredictionField field(fixture.dataset(),options);
    int calls=0;
    field.setReadinessGate([&](const auto&) {++calls;});
    auto region=field.pinRegion({0,0,0},{0,0,0});
    region->seal();
    std::vector<float> widths;
    CHECK_THROWS_AS(field.sampleWidths({{6,6,6}},widths,*region),vc::render::PinnedRegionMiss);
    std::vector<FiberPredictionSample> samples;
    field.sampleBatch(std::vector<cv::Vec3f>{{0,0,0}},{{1,0,0}},1,samples,nullptr,region.get());
    field.sampleBatchScalarEquivalent({{0,0,0}},{{1,0,0}},1,samples,nullptr,region.get());
    CHECK_THROWS_AS(field.sampleBatchScalarEquivalent({{6,6,6}},{{1,0,0}},
        1,samples,nullptr,region.get()),vc::render::PinnedRegionMiss);
    CHECK_THROWS_AS(field.sampleBatch(std::vector<cv::Vec3f>{{6,6,6}},{{1,0,0}},
        1,samples,nullptr,region.get()),vc::render::PinnedRegionMiss);
    CHECK(calls==1);
}

TEST_CASE("single-worker prediction batches retain pinned reads without normal fusion") {
    Fixture fixture;
    fixture.channel<uint8_t>("presence",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
    FiberPredictionField field(fixture.dataset());
    int readiness=0;
    field.setReadinessGate([&](const auto&){++readiness;});
    auto region=field.pinRegion({0,0,0},{1,1,1}); region->seal();
    const int initial=readiness;
    FiberTraceOneWayRequest request;
    request.startPoint={.5,.5,.2};request.targetPoint={.5,.5,.8};
    request.initialDirection={0,0,1};request.budgetSpanVoxels=.6;
    request.targetPlanes={{"end",request.targetPoint,{0,0,1}}};
    request.config.stepVoxels=.1;request.config.parallelThreads=1;
    request.config.coneAngleDegrees=5;request.config.coneAngleStepDegrees=5;
    request.config.smoothnessNormalWeight=0;request.config.smoothnessTangentWeight=0;
    request.config.cumulativeSmoothnessTangentWeight=0;
    request.config.pinnedRegion=region.get();
    const auto result=traceFiberOneWay(field,request,nullptr);
    CHECK(result.reachedTargetPlane);
    // Initial seed validation uses the ordinary source once; candidate
    // frontiers must stay on the supplied pinned region thereafter.
    CHECK(readiness==initial+1);
}
