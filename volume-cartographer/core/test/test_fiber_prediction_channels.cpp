#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "utils/zarr.hpp"

#include <array>
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

TEST_CASE("embedding scoring preserves the fused normal corner path") {
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
    config.parallelThreads=1;
    const auto serial=traceFiberExtrapolation(field,{.5,.5,.5},{0,0,1},.2,config,&normals);
    config.parallelThreads=4;
    FiberTraceProfile profile; config.profile=&profile;
    const auto fused=traceFiberExtrapolation(field,{.5,.5,.5},{0,0,1},.2,config,&normals);
    REQUIRE(serial.points.size()>1);
    CHECK(serial.points==fused.points);
    CHECK(profile.cornerPointCount>0);
}
