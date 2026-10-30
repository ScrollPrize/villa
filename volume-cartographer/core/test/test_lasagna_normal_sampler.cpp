#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineOptimizer.hpp"

#include <vc_reader/zarr.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <barrier>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto dir = fs::temp_directory_path() /
        ("vc_lasagna_normal_sampler_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

void createU8Zarr(
    const fs::path& path,
    std::vector<size_t> shape,
    std::vector<size_t> chunks,
    const std::vector<uint8_t>* payload,
    std::optional<double> fillValue = 0.0)
{
    vcr::zarr::CreateOptions options;
    options.shape.assign(shape.begin(), shape.end());
    options.chunks.assign(chunks.begin(), chunks.end());
    options.compressor = "";
    options.fillValue = fillValue ? nlohmann::json(*fillValue) : nlohmann::json();
    auto writer = vcr::zarr::Writer::create(path, options);
    if (payload != nullptr)
        writer.writeChunk(vcr::Index(shape.size(), 0), payload->data(), payload->size());
}

void createConstantChunkedU8Zarr(
    const fs::path& path,
    const std::array<size_t, 3>& shape,
    const std::array<size_t, 3>& chunks,
    uint8_t value)
{
    auto writer = vcr::zarr::Writer::create(path, {
        .shape = {int64_t(shape[0]), int64_t(shape[1]), int64_t(shape[2])},
        .chunks = {int64_t(chunks[0]), int64_t(chunks[1]), int64_t(chunks[2])},
        .compressor = ""});
    const std::vector<uint8_t> bytes(chunks[0] * chunks[1] * chunks[2], value);
    for (size_t z = 0; z < (shape[0] + chunks[0] - 1) / chunks[0]; ++z)
        for (size_t y = 0; y < (shape[1] + chunks[1] - 1) / chunks[1]; ++y)
            for (size_t x = 0; x < (shape[2] + chunks[2] - 1) / chunks[2]; ++x)
                writer.writeChunk({int64_t(z), int64_t(y), int64_t(x)}, bytes.data(), bytes.size());
}

void writeText(const fs::path& path, const std::string& text)
{
    std::ofstream out(path);
    out << text;
}

} // namespace

TEST_CASE("LasagnaNormalSampler rejects 4D channel groups")
{
    const auto dir = tmpDir("4d");
    const auto zarrPath = dir / "pred.zarr";
    std::vector<uint8_t> payload(3 * 2 * 2 * 2, 128);
    createU8Zarr(zarrPath, {3, 2, 2, 2}, {3, 2, 2, 2}, &payload);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "source_to_base": 1.0,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "pred": {
                "zarr": "pred.zarr",
                "scaledown": 0,
                "channels": ["grad_mag", "nx", "ny"]
            }
        }
    })");

    const vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_WITH_AS(
        vc::lasagna::LasagnaNormalSampler(dataset),
        doctest::Contains("must reference a 3D (Z,Y,X) zarr"),
        std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler supports 3D per-channel zarr groups and coordinate scaling")
{
    const auto dir = tmpDir("3d");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    std::vector<uint8_t> nx(2 * 2 * 2, 255);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "source_to_base": 2.0,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 1, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 1, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 1, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    const auto sample = sampler.sampleNormal({4.0, 4.0, 4.0});

    REQUIRE(sample.valid);
    CHECK(sample.normal[0] == doctest::Approx(1.0));
    CHECK(sample.normal[1] == doctest::Approx(0.0));
    CHECK(sample.normal[2] == doctest::Approx(0.0));

    vc::lasagna::LasagnaDataset scaledDataset =
        vc::lasagna::LasagnaDataset::open(manifestPath, {4.0});
    vc::lasagna::LasagnaNormalSampler scaledSampler(scaledDataset);
    const auto scaledSample = scaledSampler.sampleNormal({1.0, 1.0, 1.0});
    REQUIRE(scaledSample.valid);
    CHECK(scaledSample.normal[0] == doctest::Approx(sample.normal[0]));
    CHECK(scaledSample.normal[1] == doctest::Approx(sample.normal[1]));
    CHECK(scaledSample.normal[2] == doctest::Approx(sample.normal[2]));
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler integrates decoded grad_mag as winding distance")
{
    const auto dir = tmpDir("winding_distance");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 100);
    std::vector<uint8_t> nx(2 * 2 * 2, 255);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 100.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    const auto density = sampler.sampleWindingDensity({0.5, 0.5, 0.5});
    REQUIRE(density.has_value());
    CHECK(*density == doctest::Approx(1.0));
    CHECK(sampler.windingDistance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.25) ==
          doctest::Approx(1.0));
    CHECK(sampler.normalAlignedWindingDistance(
              {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.25) ==
          doctest::Approx(1.0));
    CHECK(sampler.windingDistance(
              {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, 0.25) ==
          doctest::Approx(1.0));
    CHECK(sampler.normalAlignedWindingDistance(
              {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, 0.25) ==
          doctest::Approx(0.0).epsilon(1.0e-9));
    const std::vector<std::pair<cv::Vec3d, cv::Vec3d>> connectors{
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
        {{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}},
    };
    const auto batchOne = sampler.normalAlignedWindingDistancesBatch(
        connectors, 0.25, 1);
    const auto batchFour = sampler.normalAlignedWindingDistancesBatch(
        connectors, 0.25, 4);
    REQUIRE(batchOne.size() == connectors.size());
    REQUIRE(batchFour.size() == batchOne.size());
    CHECK(batchOne[0] == doctest::Approx(1.0).epsilon(1.0e-6));
    CHECK(batchOne[1] == doctest::Approx(0.0).epsilon(1.0e-6));
    CHECK(batchOne[2] == doctest::Approx(0.0));
    for (std::size_t index = 0; index < batchOne.size(); ++index)
        CHECK(batchFour[index] == batchOne[index]);
    fs::remove_all(dir);
}

TEST_CASE("Batched aligned winding tracks scalar sampling on varying fields")
{
    const auto dir = tmpDir("winding_batch_varying");
    std::vector<uint8_t> gradMag(4 * 4 * 4);
    std::vector<uint8_t> nx(4 * 4 * 4);
    std::vector<uint8_t> ny(4 * 4 * 4);
    for (size_t index = 0; index < gradMag.size(); ++index) {
        gradMag[index] = static_cast<uint8_t>(40 + index * 3 % 180);
        nx[index] = static_cast<uint8_t>(80 + index * 5 % 150);
        ny[index] = static_cast<uint8_t>(50 + index * 7 % 180);
    }
    createU8Zarr(dir / "grad_mag.zarr", {4, 4, 4}, {4, 4, 4}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {4, 4, 4}, {4, 4, 4}, &nx);
    createU8Zarr(dir / "ny.zarr", {4, 4, 4}, {4, 4, 4}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 100.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    const std::vector<std::pair<cv::Vec3d, cv::Vec3d>> connectors{
        {{0.17, 0.31, 0.44}, {2.73, 1.82, 2.19}},
        {{2.61, 0.23, 1.11}, {0.29, 2.54, 2.77}},
        {{0.41, 2.49, 0.38}, {2.45, 0.52, 2.63}},
    };
    const auto batchOne = sampler.normalAlignedWindingDistancesBatch(
        connectors, 0.37, 1);
    const auto batchFour = sampler.normalAlignedWindingDistancesBatch(
        connectors, 0.37, 4);
    REQUIRE(batchOne.size() == connectors.size());
    REQUIRE(batchFour.size() == connectors.size());
    for (size_t index = 0; index < connectors.size(); ++index) {
        const double scalar = sampler.normalAlignedWindingDistance(
            connectors[index].first, connectors[index].second, 0.37);
        CHECK(batchOne[index] == doctest::Approx(scalar).epsilon(2.0e-5));
        CHECK(batchFour[index] == batchOne[index]);
    }
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler interpolates unoriented normal tensors")
{
    const auto dir = tmpDir("tensor_interp");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    std::vector<uint8_t> nx{
        255, 1,
        255, 1,
        255, 1,
        255, 1,
    };
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    const auto sample = sampler.sampleNormal({0.5, 0.5, 0.5});

    REQUIRE(sample.valid);
    CHECK(std::abs(sample.normal[0]) == doctest::Approx(1.0).epsilon(1.0e-9));
    CHECK(sample.normal[1] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(sample.normal[2] == doctest::Approx(0.0).epsilon(1.0e-9));
    fs::remove_all(dir);
}

TEST_CASE("validity masks companion corners without diluting width or presence")
{
    using namespace vc::lasagna;
    const auto dir=tmpDir("valid_corners");
    std::vector<uint8_t> mask(8,0),nx(8,0),ny(8,0),width(8,0);
    mask[0]=255; nx[0]=255; ny[0]=128; width[0]=40;
    createU8Zarr(dir/"mask.zarr",{2,2,2},{2,2,2},&mask);
    createU8Zarr(dir/"nx.zarr",{2,2,2},{2,2,2},&nx);
    createU8Zarr(dir/"ny.zarr",{2,2,2},{2,2,2},&ny);
    createU8Zarr(dir/"width.zarr",{2,2,2},{2,2,2},&width);
    writeText(dir/"dataset.lasagna.json",R"({"version":2,"mask_invalid_normal_corners":true,
        "grad_mag_encode_scale":255,"grad_mag_factor":1,"groups":{
        "m":{"zarr":"mask.zarr","scaledown":0,"channels":["grad_mag"]},
        "x":{"zarr":"nx.zarr","scaledown":0,"channels":["nx"]},
        "y":{"zarr":"ny.zarr","scaledown":0,"channels":["ny"]},
        "w":{"zarr":"width.zarr","scaledown":0,"channels":["width"]}}})");
    auto dataset=LasagnaDataset::open(dir/"dataset.lasagna.json");
    LasagnaGroupArrays arrays(1<<20);
    const auto m=bindLasagnaChannel(dataset.manifest(),"grad_mag",arrays);
    const auto w=bindLasagnaChannel(dataset.manifest(),"width",arrays);
    std::vector<std::vector<float>> out;
    sampleLasagnaVectorBatch({w},{{.5f,.5f,.5f},{1,1,1},{0,0,0}},out,false,&m);
    REQUIRE(out[0].size()==1); CHECK(out[0][0]==doctest::Approx(40));
    CHECK(out[1].empty()); REQUIRE(out[2].size()==1); CHECK(out[2][0]==doctest::Approx(40));
    CHECK(*sampleLasagnaChannel(m,cv::Vec3d(.5,.5,.5))==doctest::Approx(255./8));
    LasagnaNormalSampler sampler(dataset);
    auto n=sampler.sampleNormal({.5,.5,.5});
    REQUIRE(n.valid); CHECK(std::abs(n.normal[0])==doctest::Approx(1));
    CHECK_FALSE(sampler.sampleNormal({1,1,1}).valid);
    std::array<uint8_t,8> a{},b{},v{};
    a[0]=255;b[0]=128;v[0]=255;
    auto weights=lasagnaCornerWeights({.5f,.5f,.5f});
    auto axis=interpolateLasagnaCompactAxisCorners(a,b,validLasagnaCornerWeights(weights,v));
    CHECK(std::abs(axis[0])==doctest::Approx(1)); CHECK(std::abs(axis[1])<1e-6);
    auto zero=interpolateLasagnaCompactAxisCorners(a,b,validLasagnaCornerWeights(lasagnaCornerWeights({1,1,1}),v));
    CHECK(zero.dot(zero)==doctest::Approx(0));
    v.fill(255); CHECK(validLasagnaCornerWeights(weights,v)==weights);
    CHECK(sampler.masksInvalidCorners());
    auto legacyManifest=dataset.manifest();
    legacyManifest.raw["mask_invalid_normal_corners"]=false;
    LasagnaDataset legacyDataset(std::move(legacyManifest));
    LasagnaNormalSampler legacy(legacyDataset);
    CHECK_FALSE(legacy.masksInvalidCorners());
    const auto legacyNormal=legacy.sampleNormal({.5,.5,.5});
    REQUIRE(legacyNormal.valid);
    const auto legacyAxis=interpolateLasagnaCompactAxisCorners(a,b,weights);
    CHECK(std::abs(cv::Vec3d(legacyAxis).dot(legacyNormal.normal))==doctest::Approx(1).epsilon(1e-5));
    CHECK(std::abs(legacyNormal.normal[1])>.1);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler requires grad_mag channel")
{
    const auto dir = tmpDir("missing_grad_mag_channel");
    std::vector<uint8_t> nx(2 * 2 * 2, 128);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "groups": {
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_AS(vc::lasagna::LasagnaNormalSampler(dataset), std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler requires explicit grad_mag scale metadata")
{
    const auto dir = tmpDir("missing_grad_mag_scale");
    std::vector<uint8_t> payload(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &payload);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &payload);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &payload);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_AS(vc::lasagna::LasagnaNormalSampler(dataset), std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler reports invalid samples for missing chunks and zero grad_mag")
{
    const auto dir = tmpDir("invalid");
    std::vector<uint8_t> zeroGradMag(2 * 2 * 2, 0);
    std::vector<uint8_t> normals(2 * 2 * 2, 128);
    createU8Zarr(dir / "zero_grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &zeroGradMag);
    createU8Zarr(dir / "zero_nx.zarr", {2, 2, 2}, {2, 2, 2}, &normals);
    createU8Zarr(dir / "zero_ny.zarr", {2, 2, 2}, {2, 2, 2}, &normals);
    createU8Zarr(dir / "missing_grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, nullptr);
    createU8Zarr(dir / "missing_nx.zarr", {2, 2, 2}, {2, 2, 2}, nullptr);
    createU8Zarr(dir / "missing_ny.zarr", {2, 2, 2}, {2, 2, 2}, nullptr);

    const auto zeroManifest = dir / "zero.lasagna.json";
    writeText(zeroManifest, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "zero_grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "zero_nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "zero_ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");
    vc::lasagna::LasagnaDataset zeroDataset = vc::lasagna::LasagnaDataset::open(zeroManifest);
    vc::lasagna::LasagnaNormalSampler zeroSampler(zeroDataset);
    CHECK_FALSE(zeroSampler.sampleNormal({1.0, 1.0, 1.0}).valid);

    const auto missingManifest = dir / "missing.lasagna.json";
    writeText(missingManifest, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "missing_grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "missing_nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "missing_ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");
    vc::lasagna::LasagnaDataset missingDataset = vc::lasagna::LasagnaDataset::open(missingManifest);
    vc::lasagna::LasagnaNormalSampler missingSampler(missingDataset);
    CHECK_FALSE(missingSampler.sampleNormal({1.0, 1.0, 1.0}).valid);

    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler integrates with LineOptimizer")
{
    const auto dir = tmpDir("optimizer");
    std::vector<uint8_t> gradMag(4 * 4 * 4, 255);
    std::vector<uint8_t> normals(4 * 4 * 4, 128);
    createU8Zarr(dir / "grad_mag.zarr", {4, 4, 4}, {4, 4, 4}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {4, 4, 4}, {4, 4, 4}, &normals);
    createU8Zarr(dir / "ny.zarr", {4, 4, 4}, {4, 4, 4}, &normals);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "source_to_base": 100.0,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 50.0;
    config.maxIterations = 5;
    const auto result = optimizer.optimizeFromSeed({100.0, 100.0, 100.0}, config);

    CHECK(result.line.points.size() == 3);
    CHECK(result.report.validNormalSamples > 0);
    CHECK(result.line.points[1].sampledNormal.valid);
    CHECK(result.line.points[1].sampledNormal.normal[2] == doctest::Approx(1.0));
    fs::remove_all(dir);
}
