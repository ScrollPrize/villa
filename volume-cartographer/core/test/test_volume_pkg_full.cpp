// VolumePkg coverage with real attached resources: a local zarr volume
// and an empty normal-grid dir. Exercises addSingleVolume/removeSingleVolume,
// reload variants, volume(), etc.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
#include "support/TestProject.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_vpkg_full_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

// The project at `path`, opened in a session of its own (as another process would).
std::shared_ptr<VolumePkg> openElsewhere(const fs::path& path, bool readOnly = false,
                                         const vc::project::LoadOptions& opts = {})
{
    if (readOnly) fs::permissions(path, fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read);
    return VolumePkg::open(path, opts);
}

// Build a small local zarr volume in `root/volumes/<id>/`. Returns the
// volume path.
fs::path makeLocalVolume(const fs::path& root,
                         const std::string& id = "vol1")
{
    auto volDir = root / "volumes" / id;
    vc::test::ZarrVolumeSpec o;
    o.shapeZYX = {16, 16, 16};
    o.chunkShapeZYX = {16, 16, 16};
    o.numLevels = 1;
    o.uuid = id;
    o.name = id;
    auto v = vc::test::createZarrVolume(volDir, o);
    REQUIRE(v);
    return volDir;
}

} // namespace

TEST_CASE("VolumePkg::addVolumeEntry auto-loads vol1 from the volumes dir")
{
    auto d = tmpDir("addvol");
    makeLocalVolume(d);
    auto p = vc::test::emptyProject();
    p->addVolumeEntry((d / "volumes").string());
    // resolveVolumeEntry already loaded vol1; addSingleVolume returns false
    // for an already-present id. Either way, vol1 must be reachable.
    (void)p->addSingleVolume("vol1");
    auto v = p->volume("vol1");
    CHECK(v != nullptr);
    auto vDefault = p->volume();
    CHECK(vDefault != nullptr);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::removeSingleVolume drops a volume by id")
{
    auto d = tmpDir("rmvol");
    makeLocalVolume(d);
    auto p = vc::test::emptyProject();
    p->addVolumeEntry((d / "volumes").string());
    REQUIRE(p->volume("vol1") != nullptr);
    CHECK(p->removeSingleVolume("vol1"));
    CHECK(p->volume("vol1") == nullptr);
    // Second remove is a no-op.
    CHECK_FALSE(p->removeSingleVolume("vol1"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::attachPreparedVolume persists one loaded entry idempotently")
{
    auto d = tmpDir("attach_prepared");
    const auto firstPath = makeLocalVolume(d, "vol1");
    const auto secondRoot = d / "second";
    const auto secondPath = makeLocalVolume(secondRoot, "vol1");
    const auto store = vc::test::createTestProject(d / "project.vcobj");

    auto package = store.open();
    auto first = Volume::New(firstPath);
    REQUIRE(first);
    CHECK(
        package->attachPreparedVolume(
            firstPath.string(), {"overlay"}, first) ==
        VolumePkg::AttachVolumeResult::Attached);
    CHECK(package->volume("vol1") == first);
    REQUIRE(package->volumeEntries().size() == 1);
    CHECK(package->volumeEntries().front().tags ==
          std::vector<std::string>{"overlay"});

    CHECK(
        package->attachPreparedVolume(
            firstPath.string(), {"ignored-on-retry"}, first) ==
        VolumePkg::AttachVolumeResult::AlreadyAttached);
    CHECK(package->volumeEntries().size() == 1);
    CHECK(package->volumeEntries().front().tags ==
          std::vector<std::string>{"overlay"});

    auto conflicting = Volume::New(secondPath);
    REQUIRE(conflicting);
    CHECK(
        package->attachPreparedVolume(
            secondPath.string(), {}, conflicting) ==
        VolumePkg::AttachVolumeResult::VolumeIdConflict);
    CHECK(package->volumeEntries().size() == 1);

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto reloaded = store.open(deferred);
    REQUIRE(reloaded->volumeEntries().size() == 1);
    CHECK(vc::project::resolveLocalPath(reloaded->volumeEntries().front().location, reloaded->directory()) == firstPath);
    CHECK(reloaded->volumeEntries().front().tags ==
          std::vector<std::string>{"overlay"});
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::attachPreparedVolume recognizes a loaded relative entry")
{
    auto d = tmpDir("attach_relative");
    const auto volumePath = makeLocalVolume(d, "vol1");
    const auto store = vc::test::createTestProject(d / "project.vcobj");

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    REQUIRE(store.open(deferred)->addVolumeEntry("volumes/vol1", {"original"}));

    // Relative entries resolve against the project's directory.
    auto package = openElsewhere(fs::relative(store.path, fs::current_path()));
    auto prepared = Volume::New(volumePath);
    REQUIRE(prepared);
    CHECK(
        package->attachPreparedVolume(
            volumePath.string(), {"ignored"}, prepared) ==
        VolumePkg::AttachVolumeResult::AlreadyAttached);
    REQUIRE(package->volumeEntries().size() == 1);
    CHECK(package->volumeEntries().front().location == "volumes/vol1");
    CHECK(package->volumeEntries().front().tags ==
          std::vector<std::string>{"original"});
    fs::remove_all(d);
}

// The relative-path case above only reaches fs::equivalent on macOS, where the
// temp directory sits under a symlinked /var. These two create the symlink
// themselves, so the behaviour is exercised on every platform.
namespace
{
// Returns an alias directory symlinked to `target`, or an empty path when the
// filesystem will not create one (Windows without developer mode, some CIFS).
fs::path aliasFor(const fs::path& target)
{
    auto alias = target.parent_path() / (target.filename().string() + "_alias");
    std::error_code ec;
    fs::create_directory_symlink(target, alias, ec);
    if (ec) return {};
    return alias;
}
}  // namespace

TEST_CASE("VolumePkg::attachPreparedVolume sees through a directory symlink")
{
    auto d = tmpDir("attach_symlink");
    const auto volumePath = makeLocalVolume(d, "vol1");
    const auto store = vc::test::createTestProject(d / "project.vcobj");

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    REQUIRE(store.open(deferred)->addVolumeEntry("volumes/vol1", {"original"}));

    const auto alias = aliasFor(d);
    if (alias.empty()) {
        fs::remove_all(d);
        return;
    }

    // Package opened through the alias, volume attached through the real path.
    auto package = openElsewhere(alias / "project.vcproj");
    auto prepared = Volume::New(volumePath);
    REQUIRE(prepared);
    CHECK(
        package->attachPreparedVolume(
            volumePath.string(), {"ignored"}, prepared) ==
        VolumePkg::AttachVolumeResult::AlreadyAttached);
    CHECK(package->volumeEntries().size() == 1);

    fs::remove(alias);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::attachPreparedVolume matches a container entry through a symlink")
{
    auto d = tmpDir("attach_container_symlink");
    const auto volumePath = makeLocalVolume(d, "vol1");
    const auto store = vc::test::createTestProject(d / "project.vcobj");

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    // "volumes" is a collection entry: it holds zarr volumes but is not one.
    REQUIRE(store.open(deferred)->addVolumeEntry("volumes", {"collection"}));

    const auto alias = aliasFor(d);
    if (alias.empty()) {
        fs::remove_all(d);
        return;
    }

    auto package = openElsewhere(alias / "project.vcproj");
    auto prepared = Volume::New(volumePath);
    REQUIRE(prepared);
    CHECK(
        package->attachPreparedVolume(
            volumePath.string(), {"ignored"}, prepared) ==
        VolumePkg::AttachVolumeResult::AlreadyAttached);
    CHECK(package->volumeEntries().size() == 1);

    fs::remove(alias);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::attachPreparedVolume persists a directly loaded volume")
{
    auto d = tmpDir("attach_loaded_only");
    const auto volumePath = makeLocalVolume(d, "vol1");
    auto package = vc::test::emptyProject();
    auto prepared = Volume::New(volumePath);
    REQUIRE(prepared);

    CHECK(
        package->attachPreparedVolume(
            volumePath.string(), {"source:direct"}, prepared) ==
        VolumePkg::AttachVolumeResult::Attached);
    REQUIRE(package->volumeEntries().size() == 1);
    CHECK(package->volumeEntries().front().location == volumePath.string());
    CHECK(package->volumeEntries().front().tags ==
          std::vector<std::string>{"source:direct"});
    fs::remove_all(d);
}

TEST_CASE("VolumePkg does not treat a direct Zarr as a collection entry")
{
    auto d = tmpDir("attach_nested_direct");
    const auto parentPath = makeLocalVolume(d, "parent");
    const auto stagedChild = makeLocalVolume(d / "staging", "child");
    const auto childPath = parentPath / "child";
    fs::rename(stagedChild, childPath);
    auto package = vc::test::emptyProject();
    REQUIRE(package->addVolumeEntry(parentPath.string()));
    CHECK_FALSE(package->matchingVolumeEntry(childPath.string()));

    auto prepared = Volume::New(childPath);
    REQUIRE(prepared);
    CHECK(
        package->attachPreparedVolume(childPath.string(), {}, prepared) ==
        VolumePkg::AttachVolumeResult::Attached);
    CHECK(package->volumeEntries().size() == 2);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg matches legacy remote entries by resolved identity")
{
    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto package = vc::test::emptyProject(deferred);
    const std::string legacy = "s3://example-bucket/path/volume.zarr";
    REQUIRE(package->addVolumeEntry(
        legacy, {"vc-open-data-voxel-size-um:7.91"}));

    const auto canonical = vc::parseRemoteVolumeSpec(legacy).portableLocator;
    const auto matching = package->matchingVolumeEntry(canonical);
    REQUIRE(matching);
    CHECK(matching->location == legacy);
    CHECK(matching->tags ==
          std::vector<std::string>{"vc-open-data-voxel-size-um:7.91"});
}

TEST_CASE("VolumePkg::attachPreparedVolume leaves the project as it was when the write fails")
{
    auto d = tmpDir("attach_rollback");
    const auto volumePath = makeLocalVolume(d, "vol1");
    const auto store = vc::test::createTestProject(d / "project.vcobj");

    auto package = openElsewhere(store.path, /*readOnly=*/true);
    auto prepared = Volume::New(volumePath);
    REQUIRE(prepared);
    CHECK_THROWS(
        package->attachPreparedVolume(volumePath.string(), {}, prepared));
    CHECK(package->volumeEntries().empty());
    CHECK(package->volume("vol1") == nullptr);
    CHECK(store.open()->volumeEntries().empty());

    fs::remove_all(d);
}

TEST_CASE("A catalog gives remote entries their catalog volume identity at every level")
{
    const auto catalog = vcr::catalog::parseOpenDataManifest(R"({"samples": {"s": {
      "volumes": {"vol": {"id": "vol", "suffix": "scan.zarr", "properties": {"pixel_size_um": 2.5},
        "data": [
          {"type": "zarr", "origins": [{"path": "s/vol.zarr", "access_roots": [
            {"type": "https", "url": "https://example.test/", "usage": "public-read"}]}]},
          {"type": "surface-prediction-zarr", "parameters": {"level": 1}, "origins": [{"path": "s/pred.zarr",
            "access_roots": [{"type": "https", "url": "https://example.test/", "usage": "public-read"}]}]}]}}}}})");
    const auto source = vc::project::catalogVolumeMetadata(&catalog, "https://example.test/s/vol.zarr");
    CHECK(source.at("uuid").get_string() == "vol");
    CHECK_FALSE(source.contains("voxelsize"));  // the store's metadata decides, the catalog only fills gaps
    const auto level = vc::project::catalogVolumeMetadata(&catalog, "https://example.test/s/vol.zarr#vc-base-scale=2");
    CHECK(level.at("uuid").get_string() == "vol");
    CHECK(level.at("name").get_string() == "scan.zarr [L2]");
    const auto prediction = vc::project::catalogVolumeMetadata(&catalog, "https://example.test/s/pred.zarr");
    CHECK_FALSE(prediction.contains("uuid"));
    CHECK(prediction.at("name").get_string() == "surface-prediction-zarr of scan.zarr [L1]");
    CHECK(vc::project::catalogVolumeMetadata(&catalog, "https://example.test/elsewhere.zarr").empty());
    CHECK(vc::project::catalogVolumeMetadata(nullptr, "https://example.test/s/vol.zarr").empty());
}

