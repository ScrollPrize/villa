#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"
#include "support/TestProject.hpp"
#include <vc_reader/zarr.hpp>

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

void createArray(const fs::path& path, vcr::Index shape)
{
    vcr::zarr::CreateOptions options;
    options.shape = shape;
    options.chunks = shape;
    options.compressor = "";
    vcr::zarr::Writer::create(path, options);
}

void createZyx(const fs::path& path, vcr::Index shape = {2, 2, 2})
{
    createArray(path, std::move(shape));
}

}  // namespace

TEST_CASE("Lasagna project preparation attaches group zarrs as regular volumes")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "pred.zarr");
    const auto manifestPath = dir / "fiber.lasagna.json";
    {
        std::ofstream manifest(manifestPath);
        manifest << R"({
          "version":2,
          "groups":{
            "pred":{
              "zarr":"pred.zarr",
              "channels":["presence"]
            }
          }
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto prepared = vc::lasagna::prepareLasagnaProjectVolumes(dataset);
    REQUIRE(prepared.size() == 1);
    CHECK(prepared[0].volume->shape() == std::array<int, 3>{2, 2, 2});
    CHECK(prepared[0].volume->hasScaleLevel(0));
    CHECK(prepared[0].volume->shape(0) == std::array<int, 3>{2, 2, 2});
    CHECK(prepared[0].location ==
          fs::absolute(dir / "pred.zarr").lexically_normal().string());
    CHECK(prepared[0].tags == std::vector<std::string>{
          "vc-lasagna-manifest:" + manifestPath.string(),
          "vc-lasagna-group:pred"});

    fs::remove_all(dir);
}

TEST_CASE("Lasagna project preparation deduplicates shared source zarrs")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "pred.zarr");
    const auto manifestPath = dir / "fiber.lasagna.json";
    {
        std::ofstream manifest(manifestPath);
        manifest << R"({
          "version":2,
          "groups":{
            "presence":{
              "zarr":"pred.zarr",
              "channels":["presence"]
            },
            "nx":{
              "zarr":"pred.zarr",
              "channels":["nx"]
            },
            "ny":{
              "zarr":"pred.zarr",
              "channels":["ny"]
            }
          }
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto prepared = vc::lasagna::prepareLasagnaProjectVolumes(dataset);
    REQUIRE(prepared.size() == 1);
    CHECK(prepared[0].location ==
          fs::absolute(dir / "pred.zarr").lexically_normal().string());
    CHECK(std::find(prepared[0].tags.begin(), prepared[0].tags.end(),
                    "vc-lasagna-manifest:" + manifestPath.string()) !=
          prepared[0].tags.end());
    CHECK(std::find(prepared[0].tags.begin(), prepared[0].tags.end(),
                    "vc-lasagna-group:presence") != prepared[0].tags.end());
    CHECK(std::find(prepared[0].tags.begin(), prepared[0].tags.end(),
                    "vc-lasagna-group:nx") != prepared[0].tags.end());
    CHECK(std::find(prepared[0].tags.begin(), prepared[0].tags.end(),
                    "vc-lasagna-group:ny") != prepared[0].tags.end());

    std::vector<VolumePkg::PreparedVolumeAttachment> attachments;
    for (auto& item : vc::lasagna::prepareLasagnaProjectVolumes(dataset)) {
        attachments.push_back({
            std::move(item.location),
            std::move(item.tags),
            std::move(item.volume)});
    }
    auto package = vc::test::emptyProject();
    const auto result = package->attachPreparedLasagnaDataset(
        manifestPath.string(), {}, true, attachments);
    REQUIRE(result == VolumePkg::AttachLasagnaResult::Attached);
    CHECK(package->numberOfVolumes() == 1);
    CHECK(package->volumeEntries().size() == 1);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna project volume tags preserve anonymous remote authentication")
{
    vc::lasagna::LasagnaChannelGroup group;
    group.name = "pred";
    group.sourceLocation = "s3://bucket/artifact/pred.zarr";
    group.discoverAwsCredentials = false;

    const auto tags = vc::lasagna::lasagnaProjectVolumeTags(
        group, "s3://bucket/artifact/dataset.lasagna.json");
    CHECK(std::find(tags.begin(), tags.end(),
                    vc::project::kAnonymousRemoteAuthTag) != tags.end());

    group.discoverAwsCredentials = true;
    const auto signedTags = vc::lasagna::lasagnaProjectVolumeTags(
        group, "s3://bucket/artifact/dataset.lasagna.json");
    CHECK(std::find(signedTags.begin(), signedTags.end(),
                    vc::project::kAnonymousRemoteAuthTag) == signedTags.end());
}

TEST_CASE("Lasagna project preparation keeps numeric scale group ids unique")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "fiber_sd1_exp002_crop_presence.ome.zarr" / "3");
    createZyx(dir / "fiber_sd1_exp002_crop_nx.ome.zarr" / "3");
    createZyx(dir / "fiber_sd1_exp002_crop_ny.ome.zarr" / "3");
    const auto manifestPath = dir / "fiber.lasagna.json";
    {
        std::ofstream manifest(manifestPath);
        manifest << R"({
          "version":2,
          "source_to_base":1.0,
          "groups":{
            "presence":{
              "zarr":"fiber_sd1_exp002_crop_presence.ome.zarr/3",
              "scaledown":3,
              "channels":["presence"]
            },
            "nx":{
              "zarr":"fiber_sd1_exp002_crop_nx.ome.zarr/3",
              "scaledown":3,
              "channels":["nx"]
            },
            "ny":{
              "zarr":"fiber_sd1_exp002_crop_ny.ome.zarr/3",
              "scaledown":3,
              "channels":["ny"]
            }
          }
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto prepared = vc::lasagna::prepareLasagnaProjectVolumes(dataset);
    REQUIRE(prepared.size() == 3);
    auto requirePreparedGroup = [&](const std::string& group,
                                    const fs::path& expectedRoot) {
        auto it = std::find_if(
            prepared.begin(), prepared.end(), [&](const auto& item) {
                return std::find(item.tags.begin(), item.tags.end(),
                                 "vc-lasagna-group:" + group) !=
                       item.tags.end();
        });
        REQUIRE(it != prepared.end());
        CHECK(it->location == fs::absolute(expectedRoot).lexically_normal().string());
        CHECK(fs::path(it->location).filename() != "3");
    };
    requirePreparedGroup("presence",
                         dir / "fiber_sd1_exp002_crop_presence.ome.zarr");
    requirePreparedGroup("nx",
                         dir / "fiber_sd1_exp002_crop_nx.ome.zarr");
    requirePreparedGroup("ny",
                         dir / "fiber_sd1_exp002_crop_ny.ome.zarr");

    std::vector<VolumePkg::PreparedVolumeAttachment> attachments;
    for (auto& item : prepared) {
        attachments.push_back({
            std::move(item.location),
            std::move(item.tags),
            std::move(item.volume)});
    }
    REQUIRE(attachments.size() == 3);
    auto requireGroup = [&](const std::string& group,
                            const std::string& expectedName) {
        auto it = std::find_if(
            attachments.begin(), attachments.end(), [&](const auto& item) {
                return std::find(item.tags.begin(), item.tags.end(),
                                 "vc-lasagna-group:" + group) !=
                       item.tags.end();
            });
        REQUIRE(it != attachments.end());
        CHECK(it->volume->name() == expectedName);
        CHECK(it->volume->id().rfind(expectedName + "-", 0) == 0);
        CHECK(it->volume->id().find("/3") == std::string::npos);
    };
    requireGroup("presence", "fiber_sd1_exp002_crop_presence.ome.zarr");
    requireGroup("nx", "fiber_sd1_exp002_crop_nx.ome.zarr");
    requireGroup("ny", "fiber_sd1_exp002_crop_ny.ome.zarr");
    CHECK(attachments[0].volume->id() != attachments[1].volume->id());
    CHECK(attachments[0].volume->id() != attachments[2].volume->id());
    CHECK(attachments[1].volume->id() != attachments[2].volume->id());

    auto package = vc::test::emptyProject();
    REQUIRE(package->attachPreparedLasagnaDataset(
                manifestPath.string(), {}, true, attachments) ==
            VolumePkg::AttachLasagnaResult::Attached);
    CHECK(package->numberOfVolumes() == 3);
    CHECK(package->volumeEntries().size() == 3);
    for (const auto& entry : package->volumeEntries())
        CHECK(fs::path(entry.location).filename() != "3");
    fs::remove_all(dir);
}

TEST_CASE("Lasagna project preparation leaves scale handling to native volume loading")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "pred.zarr" / "2", {2, 3, 3});
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
    CHECK(prepared[0].location ==
          fs::absolute(dir / "pred.zarr").lexically_normal().string());
    CHECK(fs::path(prepared[0].location).filename() != "2");
    const auto& volume = prepared[0].volume;
    CHECK(volume->shape() == std::array<int, 3>{8, 12, 12});
    CHECK(volume->shapeXyz() == std::array<int, 3>{12, 12, 8});
    CHECK_FALSE(volume->hasScaleLevel(0));
    CHECK_FALSE(volume->hasScaleLevel(1));
    CHECK(volume->hasScaleLevel(2));
    CHECK(volume->shape(2) == std::array<int, 3>{2, 3, 3});

    fs::remove_all(dir);
}

TEST_CASE("Lasagna project preparation uses native volume validation")
{
    const auto dir = temporaryDirectory();
    createArray(dir / "pred.zarr", {2, 2, 2, 2});
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::
        parseText(R"({"version":2,"groups":{"pred":{"zarr":"pred.zarr","channels":["presence"]}}})", dir / "data.lasagna.json");
    CHECK_THROWS_WITH_AS(
        vc::lasagna::prepareLasagnaProjectVolumes(vc::lasagna::LasagnaDataset(manifest)),
        doctest::Contains("must be 3D"),
        std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna project volumes reload without Lasagna-specific reconciliation")
{
    const auto dir = temporaryDirectory();
    createZyx(dir / "presence.zarr");
    createZyx(dir / "nx.zarr");
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
    const auto store = vc::test::createTestProject();
    auto package = store.open(options);
    REQUIRE(package->attachPreparedLasagnaDataset(manifestPath.string(), {}, false, attachments) == VolumePkg::AttachLasagnaResult::Attached);
    REQUIRE(package->volumeEntries().size() == 2);

    auto reloaded = store.open(options);
    CHECK(reloaded->numberOfVolumes() == 0);
    reloaded->resolveDeferredEntries();
    CHECK(reloaded->numberOfVolumes() == 2);
    CHECK(reloaded->volumeEntries().size() == 2);
    fs::remove_all(dir);
}
