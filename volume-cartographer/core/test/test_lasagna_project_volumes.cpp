#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"
#include "utils/zarr.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <random>

namespace fs = std::filesystem;

namespace
{

fs::path temporaryDirectory()
{
    std::mt19937_64 rng(std::random_device{}());
    auto path = fs::temp_directory_path() / ("vc_lasagna_project_volumes_" + std::to_string(rng()));
    fs::create_directories(path);
    return path;
}

void createZyx(const fs::path& path,
               std::array<std::size_t, 3> shape,
               unsigned char firstValue)
{
    utils::ZarrMetadata metadata;
    metadata.version = utils::ZarrVersion::v2;
    metadata.shape = {shape[0], shape[1], shape[2]};
    metadata.chunks = {shape[0], shape[1], shape[2]};
    metadata.dtype = utils::ZarrDtype::uint8;
    metadata.compressor_id.clear();
    metadata.fill_value = 0.0;
    auto array = utils::ZarrArray::create(path, metadata);
    std::vector<std::byte> bytes(shape[0] * shape[1] * shape[2]);
    for (std::size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<std::byte>(firstValue + i);
    const std::array<std::size_t, 3> key{0, 0, 0};
    array.write_chunk(key, bytes);
}

void createZyx(const fs::path& path, unsigned char firstValue)
{
    createZyx(path, {2, 2, 2}, firstValue);
}

void createCzyx(const fs::path& path)
{
    utils::ZarrMetadata metadata;
    metadata.version = utils::ZarrVersion::v2;
    metadata.shape = {2, 2, 2, 2};
    metadata.chunks = {2, 2, 2, 2};
    metadata.dtype = utils::ZarrDtype::uint8;
    metadata.compressor_id.clear();
    metadata.fill_value = 0.0;
    (void)utils::ZarrArray::create(path, metadata);
}

}  // namespace

TEST_CASE("Lasagna project preparation exposes a ZYX group as a 3D volume")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "pred.zarr", 100);
    const auto manifestPath = dir / "fiber.lasagna.json";
    {
        std::ofstream manifest(manifestPath);
        manifest << R"({
          "version":2,
          "source_to_base":2.0,
          "groups":{
            "pred":{
              "zarr":"pred.zarr",
              "scaledown":1,
              "channels":["presence"]
            }
          }
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto prepared = vc::lasagna::prepareLasagnaProjectVolumes(dataset);
    REQUIRE(prepared.size() == 1);
    CHECK(prepared[0].volume->shape() == std::array<int, 3>{4, 4, 4});
    CHECK_FALSE(prepared[0].volume->hasScaleLevel(0));
    CHECK(prepared[0].volume->hasScaleLevel(1));
    CHECK(prepared[0].volume->levelShape(1) == std::array<int, 3>{2, 2, 2});
    CHECK(prepared[0].volume->voxelSize() == doctest::Approx(4.0));
    CHECK(prepared[0].location ==
          fs::absolute(dir / "pred.zarr").lexically_normal().string());
    CHECK(std::find(
              prepared[0].tags.begin(), prepared[0].tags.end(),
              "vc-lasagna-derived:" + manifestPath.string()) !=
          prepared[0].tags.end());
    CHECK(prepared[0].tags == std::vector<std::string>{
          "vc-lasagna-derived:" + manifestPath.string()});

    const auto chunk = prepared[0].volume->chunkedCache()->getChunkBlocking(1, 0, 0, 0);
    REQUIRE(chunk.status == vc::render::ChunkStatus::Data);
    REQUIRE(chunk.bytes);
    REQUIRE(chunk.bytes->size() == 8);
    for (std::size_t i = 0; i < 8; ++i)
        CHECK(static_cast<unsigned char>((*chunk.bytes)[i]) == 100 + i);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna project preparation preserves manifest base shape for scaled groups")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "pred.zarr", {2, 3, 3}, 50);
    const auto manifestPath = dir / "data.lasagna.json";
    {
        std::ofstream manifest(manifestPath);
        manifest << R"({
          "version":2,
          "source_to_base":1.0,
          "base_shape_zyx":[8,10,12],
          "groups":{
            "pred":{
              "zarr":"pred.zarr",
              "scaledown":2,
              "channels":["presence"]
            }
          }
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto prepared = vc::lasagna::prepareLasagnaProjectVolumes(dataset);
    REQUIRE(prepared.size() == 1);
    const auto& volume = prepared[0].volume;
    CHECK(volume->shape() == std::array<int, 3>{8, 10, 12});
    CHECK(volume->shapeXyz() == std::array<int, 3>{12, 10, 8});
    CHECK_FALSE(volume->hasScaleLevel(0));
    CHECK_FALSE(volume->hasScaleLevel(1));
    CHECK(volume->hasScaleLevel(2));
    CHECK(volume->levelShape(2) == std::array<int, 3>{2, 3, 3});

    const auto chunk = volume->chunkedCache()->getChunkBlocking(2, 0, 0, 0);
    REQUIRE(chunk.status == vc::render::ChunkStatus::Data);
    REQUIRE(chunk.bytes);
    REQUIRE(chunk.bytes->size() == 18);
    CHECK(static_cast<unsigned char>((*chunk.bytes)[0]) == 50);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna project preparation rejects CZYX arrays")
{
    const auto dir = temporaryDirectory();
    createCzyx(dir / "pred.zarr");
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::
        parseText(R"({"version":2,"groups":{"pred":{"zarr":"pred.zarr","channels":["presence"]}}})", dir / "data.lasagna.json");
    CHECK_THROWS_WITH_AS(
        vc::lasagna::prepareLasagnaProjectVolumes(vc::lasagna::LasagnaDataset(manifest)),
        doctest::Contains("must reference a 3D (Z,Y,X) array"),
        std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna project preparation rejects ambiguous 3D multi-channel groups")
{
    const auto dir = temporaryDirectory();
    utils::ZarrMetadata metadata;
    metadata.version = utils::ZarrVersion::v2;
    metadata.shape = {2, 2, 2};
    metadata.chunks = {2, 2, 2};
    metadata.dtype = utils::ZarrDtype::uint8;
    metadata.compressor_id.clear();
    (void)utils::ZarrArray::create(dir / "pred.zarr", metadata);
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::
        parseText(R"({"version":2,"groups":{"pred":{"zarr":"pred.zarr","channels":["a","b"]}}})", dir / "data.lasagna.json");
    CHECK_THROWS_AS(vc::lasagna::prepareLasagnaProjectVolumes(vc::lasagna::LasagnaDataset(manifest)), std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna project reconciliation restores prepared volumes after reload")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "presence.zarr", 1);
    createZyx(dir / "nx.zarr", 100);
    const auto manifestPath = dir / "fiber.lasagna.json";
    {
        std::ofstream manifest(manifestPath);
        manifest << R"({"version":2,"groups":{
          "presence":{"zarr":"presence.zarr","channels":["presence"]},
          "nx":{"zarr":"nx.zarr","channels":["nx"]}
        }})";
    }
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    std::vector<VolumePkg::PreparedVolumeAttachment> attachments;
    for (auto& item : vc::lasagna::prepareLasagnaProjectVolumes(dataset)) {
        attachments.push_back({std::move(item.location), std::move(item.tags), std::move(item.volume)});
    }

    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto package = VolumePkg::newDetached(options);
    REQUIRE(package->attachPreparedLasagnaDataset(manifestPath.string(), {}, false, attachments) == VolumePkg::AttachLasagnaResult::Attached);
    const auto projectPath = dir / "project.volpkg.json";
    package->save(projectPath);

    auto reloaded = VolumePkg::load(projectPath, options);
    CHECK(reloaded->numberOfVolumes() == 0);
    const auto diagnostics = vc::lasagna::reconcileLasagnaProjectVolumes(*reloaded);
    CHECK_MESSAGE(diagnostics.empty(), (diagnostics.empty() ? std::string{} : diagnostics.front()));
    CHECK(reloaded->numberOfVolumes() == 2);
    CHECK(reloaded->volumeEntries().size() == 2);
    fs::remove_all(dir);
}
