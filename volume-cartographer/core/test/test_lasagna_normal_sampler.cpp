#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "vc/atlas/Atlas.hpp"

#include "utils/zarr.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
    double fillValue = 0.0)
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v2;
    meta.shape = std::move(shape);
    meta.chunks = std::move(chunks);
    meta.dtype = utils::ZarrDtype::uint8;
    meta.compressor_id.clear();
    meta.fill_value = fillValue;
    auto array = utils::ZarrArray::create(path, meta);
    if (payload == nullptr) {
        return;
    }
    std::vector<std::byte> bytes(payload->size());
    for (size_t i = 0; i < payload->size(); ++i) {
        bytes[i] = static_cast<std::byte>((*payload)[i]);
    }
    std::vector<size_t> zero(meta.shape.size(), 0);
    array.write_chunk(zero, bytes);
}

void createConstantChunkedU8Zarr(
    const fs::path& path,
    const std::array<size_t, 3>& shape,
    const std::array<size_t, 3>& chunks,
    uint8_t value)
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v2;
    meta.shape = {shape[0], shape[1], shape[2]};
    meta.chunks = {chunks[0], chunks[1], chunks[2]};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.compressor_id.clear();
    meta.fill_value = 0.0;
    auto array = utils::ZarrArray::create(path, meta);
    std::vector<std::byte> bytes(chunks[0] * chunks[1] * chunks[2],
                                 static_cast<std::byte>(value));
    for (size_t z = 0; z < (shape[0] + chunks[0] - 1) / chunks[0]; ++z) {
        for (size_t y = 0; y < (shape[1] + chunks[1] - 1) / chunks[1]; ++y) {
            for (size_t x = 0; x < (shape[2] + chunks[2] - 1) / chunks[2]; ++x) {
                const std::array<size_t, 3> indices{z, y, x};
                array.write_chunk(indices, bytes);
            }
        }
    }
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
    REQUIRE(scaledSampler.predDtSpacing() == std::nullopt);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler samples pred_dt channel with accepted threshold")
{
    const auto dir = tmpDir("pred_dt");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    std::vector<uint8_t> nx(2 * 2 * 2, 128);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    std::vector<uint8_t> predDtValues(2 * 2 * 2, 170);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    createU8Zarr(dir / "pred_dt.zarr", {2, 2, 2}, {2, 2, 2}, &predDtValues);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]},
            "pred_dt_group": {"zarr": "pred_dt.zarr", "scaledown": 0, "channels": ["pred_dt"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    REQUIRE(sampler.hasPredDtChannel());
    REQUIRE(sampler.predDtSpacing().has_value());
    CHECK(*sampler.predDtSpacing() == doctest::Approx(1.0));
    const auto predDt = sampler.samplePredDt({1.0, 1.0, 1.0});
    REQUIRE(predDt.has_value());
    CHECK(*predDt == doctest::Approx(170.0));
    CHECK(vc::atlas::atlasPredDtIsInside(*predDt));
    CHECK(vc::atlas::atlasPredDtIsInside(110.0));
    CHECK_FALSE(vc::atlas::atlasPredDtIsInside(109.0));
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler integrates decoded grad_mag as winding distance")
{
    const auto dir = tmpDir("winding_distance");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 100);
    std::vector<uint8_t> nx(2 * 2 * 2, 128);
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

TEST_CASE("LasagnaNormalSampler fetches only interpolation source chunks across boundaries")
{
    const auto dir = tmpDir("chunk_boundary");
    createConstantChunkedU8Zarr(dir / "grad_mag.zarr", {4, 4, 4}, {2, 2, 2}, 255);
    createConstantChunkedU8Zarr(dir / "nx.zarr", {4, 4, 4}, {2, 2, 2}, 128);
    createConstantChunkedU8Zarr(dir / "ny.zarr", {4, 4, 4}, {2, 2, 2}, 128);
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

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
    const auto report = sampler.sampleNormalBatch({{1.5, 1.5, 1.5}}, false, samples);

    REQUIRE(samples.size() == 1);
    REQUIRE(samples.front().sample.valid);
    CHECK(samples.front().sample.normal[2] == doctest::Approx(1.0));
    CHECK(report.prefetch.requestedChunks == 24);
    CHECK(report.prefetch.chunksRead == 24);
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

TEST_CASE("LasagnaNormalSampler reads absent chunks from the Zarr fill value")
{
    const auto dir = tmpDir("fill_value");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, nullptr, 128.0);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, nullptr, 128.0);

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

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    const auto sample = sampler.sampleNormal({1.0, 1.0, 1.0});

    REQUIRE(sample.valid);
    CHECK(sample.normal[0] == doctest::Approx(0.0));
    CHECK(sample.normal[1] == doctest::Approx(0.0));
    CHECK(sample.normal[2] == doctest::Approx(1.0));
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
