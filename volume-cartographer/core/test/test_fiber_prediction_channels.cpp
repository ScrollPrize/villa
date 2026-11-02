#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/detail/FiberChunkTensor.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>

#include <array>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
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
        channel<uint8_t>("presence", {0,255,0,255,0,255,0,255}, "|u1");
        channel<uint8_t>("nx", std::vector<uint8_t>(8,128), "|u1");
        channel<uint8_t>("ny", std::vector<uint8_t>(8,128), "|u1");
    }
    ~Fixture() { std::error_code error; fs::remove_all(path, error); }
    template<class T>
    void channel(const std::string& name, const std::vector<T>& values,
                 const std::string& dtype, int scale = 0) {
        auto writer = vcr::zarr::Writer::create(path/name, {.shape={2,2,2}, .chunks={2,2,2},
                                                            .dtype=dtype, .compressor=""});
        writer.writeChunk({0,0,0}, reinterpret_cast<const uint8_t*>(values.data()), values.size()*sizeof(T));
        groups[name] = {{"zarr", name}, {"scaledown", scale}, {"channels", {name}}};
    }
    // An empty array of shape^3 voxels in chunk^3 chunks.
    void empty(const std::string& name, int64_t shape, int64_t chunk, const std::string& dtype = "|u1") {
        fs::remove_all(path/name);
        (void)vcr::zarr::Writer::create(path/name, {.shape={shape,shape,shape}, .chunks={chunk,chunk,chunk},
                                                     .dtype=dtype, .compressor=""});
    }
    void auxiliary() {
        channel<float>("embedding_000", {-2,4,-2,4,-2,4,-2,4}, "<f4");
        channel<float>("embedding_001", std::vector<float>(8,1), "<f4");
        channel<float>("width", {0,20,0,20,0,20,0,20}, "<f4");
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
        fixture.channel<uint8_t>("presence",std::vector<uint8_t>(8,255),"|u1");
        fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),"|u1");
        if(byteWidth)fixture.channel<uint8_t>("width",std::vector<uint8_t>(8,20),"|u1");
        else fixture.channel<float>("width",std::vector<float>(8,20.125f),"<f4");
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
        CHECK(profile.samplingWindowCopiedBytes>0);
        CHECK(profile.samplingWindowPeakBytes>0);
        config.widthRelativeTolerance=0;profile={};
        (void)traceFiberExtrapolation(noWidth,{.5,.5,.2},{0,0,1},.6,config,&normals);
        CHECK(profile.samplingWindowLoads==0);CHECK(profile.samplingWindowFallbacks>0);
        config.samplingWindowVoxels=7;CHECK_THROWS(validateFiberTraceConfig(config));
        config.samplingWindowVoxels=257;CHECK_THROWS(validateFiberTraceConfig(config));
    }
}

TEST_CASE("dense chunk tensor matches scalar and batch interpolation with masked byte and float widths") {
    for(bool byteWidth:{false,true})for(bool masked:{false,true}) {
        Fixture fixture;
        fixture.channel<uint8_t>("presence",{0,255,203,204,205,255,128,0},"|u1");
        fixture.channel<uint8_t>("nx",{128,129,0,255,50,170,128,128},"|u1");
        fixture.channel<uint8_t>("ny",{128,200,0,255,170,100,128,128},"|u1");
        fixture.channel<uint8_t>("grad_mag",{0,255,0,255,255,0,255,255},"|u1");
        if(byteWidth)fixture.channel<uint8_t>("width",{0,20,40,80,120,200,255,0},"|u1");
        else fixture.channel<float>("width",{0,20.125f,-4.5f,80,120,200,255,0},"<f4");
        auto manifest=fixture.dataset().manifest();
        manifest.workingToBaseScale=2;
        manifest.raw["mask_invalid_normal_corners"]=masked;
        if(byteWidth)manifest.raw["width_encoding"]="uint8-base-voxels";
        vc::lasagna::LasagnaDataset dataset(manifest);
        FiberPredictionFieldOptions options;options.enableWidth=true;
        FiberPredictionField field(dataset,options);vc::lasagna::LasagnaNormalSampler normals(dataset);
        detail::FiberChunkTensor dense(field,normals,{-1,-1,-1},{2,2,2});
        const double s=field.chunkRegionChannels().first[0].spacing;
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
        fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),"|u1");
        if(byteWidth)fixture.channel<uint8_t>("width",std::vector<uint8_t>(8,20),"|u1");
        else fixture.channel<float>("width",std::vector<float>(8,20.125f),"<f4");
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
            std::vector<uint8_t>(8,pattern==1?255:pattern==2?0:204),"|u1");
        fixture.channel<uint8_t>("nx",pattern==1?std::vector<uint8_t>(8,128):
            std::vector<uint8_t>{128,129,0,255,50,170,128,128},"|u1");
        fixture.channel<uint8_t>("ny",pattern==1?std::vector<uint8_t>(8,128):
            std::vector<uint8_t>{128,200,0,255,170,100,128,128},"|u1");
        fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),"|u1");
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
    auto array=vcr::zarr::Writer::create(fixture.path/"packed",{.shape={7,8,4,4},.chunks={7,4,4,4},.dtype="|u1",.compressor=""});
    std::vector<uint8_t> data(7*64);
    const std::array<uint8_t,7> constants{255,128,200,40,180,130,255};
    for(size_t c=0;c<7;++c)std::fill(data.begin()+c*64,data.begin()+(c+1)*64,constants[c]);
    array.writeChunk({0,0,0,0},data.data(),data.size());
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
    fixture.channel<uint8_t>("grad_mag",std::vector<uint8_t>(8,255),"|u1");
    auto array=vcr::zarr::Writer::create(fixture.path/"width",{.shape={2,2,2},.chunks={1,1,1},.dtype="<f4",.compressor=""});
    for(size_t z=0;z<2;++z)for(size_t y=0;y<2;++y)for(size_t x=0;x<2;++x){
        const float value=float(10+z*4+y*2+x);
        array.writeChunk({int64_t(z),int64_t(y),int64_t(x)},reinterpret_cast<const uint8_t*>(&value),sizeof(value));
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
    checkCombined(dense,points);
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
            fixture.channel<uint8_t>("width", std::vector<uint8_t>(8,20), "|u1");
        } else {
            fixture.channel<float>("width", std::vector<float>(8,20), "<f4", invalid == 1 ? 1 : 0);
        }
        fixture.channel<float>("embedding_000", std::vector<float>(8,1), "<f4", invalid == 2 ? 1 : 0);
        fixture.channel<float>(invalid == 3 ? "embedding_002" : "embedding_001", std::vector<float>(8,1), "<f4");
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

TEST_CASE("auxiliary scoring preserves the fused normal corner path") {
    Fixture fixture; fixture.auxiliary();
    fixture.channel<uint8_t>("grad_mag", std::vector<uint8_t>(8,255), "|u1");
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
    std::atomic<size_t> gateCalls{0};
    field.setReadinessGate([&](const auto& dependencies) {
        CHECK(dependencies.size()==1);
        ++gateCalls;
    });
    config.parallelThreads=4;
    FiberTraceProfile profile; config.profile=&profile;
    const auto fused=traceFiberExtrapolation(field,{.5,.5,.5},{0,0,1},.2,config,&normals);
    REQUIRE(serial.points.size()>1);
    CHECK(serial.points==fused.points);
    CHECK(profile.predictionCornerSeconds>0);
    // One acknowledgement per chunk read: presence, nx, ny, two embeddings, width.
    CHECK(gateCalls==6);
}

TEST_CASE("readiness gates exactly the chunks a read touches") {
    Fixture fixture; fixture.auxiliary();
    fixture.empty("width",2,1,"<f4");
    FiberPredictionField field(fixture.dataset(),auxiliaryOptions());
    std::mutex mutex;
    std::vector<FiberPredictionChunkDependency> seen;
    field.setReadinessGate([&](const auto& dependencies) {
        std::lock_guard lock(mutex);
        seen.insert(seen.end(),dependencies.begin(),dependencies.end());
    });
    std::vector<float> widths;
    field.sampleWidths({{.5f,.5f,.5f}},widths);
    // Eight width chunks and the presence chunk that masks them.
    CHECK(seen.size()==9);
    CHECK(std::count_if(seen.begin(),seen.end(),[](const auto& d) {
        return d.arrayPath.filename()=="width";
    })==8);
}

TEST_CASE("readiness covers every sampling entrypoint and enabled channel") {
    Fixture fixture; fixture.auxiliary();
    fixture.channel<uint8_t>("grad_mag", std::vector<uint8_t>(8,255), "|u1");
    const auto dataset = fixture.dataset();
    FiberPredictionField field(dataset, auxiliaryOptions());
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    std::atomic<int> calls = 0;
    field.setReadinessGate([&](const auto& dependencies) {
        ++calls;
        REQUIRE(dependencies.size() == 1);
        CHECK(dependencies[0].indexZYX == std::array<uint32_t,3>{0,0,0});
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
    int expected = 0;
    const auto gated = [&](auto&& sample) {
        CHECK_THROWS(sample());
        CHECK(calls > expected);
        expected = calls;
    };
    gated([&] { (void)field.sample({.5,.5,.5},{0,0,1}); });
    gated([&] { field.sampleBatch(points,directions,4,samples); });
    gated([&] { field.sampleBatch(points,directions,4,samples,nullptr); });
    gated([&] { field.sampleBatch(floatPoints,floatDirections,4,samples,nullptr); });
    gated([&] { field.sampleWidths(floatPoints,widths); });
    gated([&] { field.sampleEmbeddings(floatPoints,embeddings); });
    gated([&] { field.sampleStoredGridBatch({{0,0,0}},4,stored); });
    gated([&] { field.sampleStoredPresenceBatch({{0,0,0}},4,presence); });
    gated([&] { (void)field.sampleCornerBatchWithNormals(normals,floatPoints,&corners,nullptr); });
    CHECK_THROWS(field.scanStoredPresenceChunks());
    CHECK(calls == expected); // Whole-volume scans must not discover a partial live cache.
    field.sampleBatch(std::vector<cv::Vec3f>{},std::vector<cv::Vec3f>{},1,samples,nullptr);
    CHECK(calls == expected);
    CHECK_THROWS(field.sampleBatch(points,{},1,samples));
    CHECK(calls == expected);
    field.setReadinessGate({});
    CHECK(field.sample({.5,.5,.5},{0,0,1}).options[0].valid);
}

TEST_CASE("readiness sees the chunks of float boundary corners") {
    Fixture fixture;
    for (const auto* name : {"presence","nx","ny"})
        fixture.empty(name,128,64);
    auto manifest=vc::lasagna::LasagnaDatasetManifest::parseText(
        nlohmann::json{{"version",2},{"source_to_base",.01},{"groups",fixture.groups}}.dump(),fixture.path/"manifest.json");
    FiberPredictionField field{vc::lasagna::LasagnaDataset(std::move(manifest))};
    std::mutex mutex;
    std::vector<FiberPredictionChunkDependency> seen;
    field.setReadinessGate([&](const auto& dependencies) {
        std::lock_guard lock(mutex);
        seen.insert(seen.end(),dependencies.begin(),dependencies.end());
    });
    std::vector<FiberPredictionSample> samples;
    // float(.63)/float(.01)==63: the cube reaches x=64, the next chunk.
    field.sampleBatch(std::vector<cv::Vec3f>{{.63f,.1f,.1f}},
        std::vector<cv::Vec3f>{{1,0,0}},1,samples,nullptr);
    for (const auto* name : {"presence","nx","ny"}) {
        CHECK(std::any_of(seen.begin(),seen.end(),[&](const auto& d) {
            return d.arrayPath.filename()==name && d.indexZYX==std::array<uint32_t,3>{0,0,1};
        }));
    }
}

TEST_CASE("readiness precedes reads and does not remember failed acknowledgements") {
    Fixture fixture;
    fs::remove(fixture.path/"presence"/"0.0.0");
    const auto dataset=fixture.dataset();
    FiberPredictionField field(dataset);
    std::atomic<int> calls=0;
    field.setReadinessGate([&](const auto& dependencies) {
        if (++calls==1) throw std::runtime_error("cancelled publication");
        if (dependencies.at(0).arrayPath.filename()=="presence")
            fixture.channel<uint8_t>("presence",std::vector<uint8_t>(8,255),"|u1");
    });
    CHECK_THROWS(field.sample({.5,.5,.5},{0,0,1}));
    CHECK(field.sample({.5,.5,.5},{0,0,1}).options[0].presence==1);
    CHECK(calls==4); // the failed presence chunk again, then nx and ny
    // A new gate is a new session: acknowledged chunks are not reused.
    std::atomic<int> concurrentCalls{0};
    field.setReadinessGate([&](const auto&) { ++concurrentCalls; });
    std::vector<std::future<float>> workers;
    for (int i=0;i<4;++i) workers.push_back(std::async(std::launch::async,[&] {
        return field.sample({.5,.5,.5},{0,0,1}).options[0].presence;
    }));
    for (auto& worker : workers) CHECK(worker.get()==1);
    // Each reader asks for a chunk the memory cache does not hold yet itself (the
    // gate waits on its thread), and no chunk more than once per reader.
    CHECK(concurrentCalls>=3);
    CHECK(concurrentCalls<=12);
}
