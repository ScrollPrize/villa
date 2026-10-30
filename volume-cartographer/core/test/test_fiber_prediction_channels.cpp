#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include <vc_reader/zarr.hpp>

#include <array>
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
    CHECK_THROWS(field.scanStoredPresenceChunks(1));
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
