#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
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
    CHECK(calls==1);
}
