// VolumePkg coverage with real attached resources: a local zarr volume
// and an empty normal-grid dir. Exercises addSingleVolume/removeSingleVolume,
// reload variants, volume(), etc.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
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

struct TestAutosaveRoot {
    TestAutosaveRoot()
        : previous(VolumePkg::autosaveRoot())
        , root(tmpDir("autosave_root"))
    {
        VolumePkg::setAutosaveRoot(root);
    }

    ~TestAutosaveRoot()
    {
        VolumePkg::setAutosaveRoot(previous);
        fs::remove_all(root);
    }

    fs::path previous;
    fs::path root;
};

TestAutosaveRoot testAutosaveRoot;

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
    auto p = VolumePkg::newEmpty();
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
    auto p = VolumePkg::newEmpty();
    p->addVolumeEntry((d / "volumes").string());
    REQUIRE(p->volume("vol1") != nullptr);
    CHECK(p->removeSingleVolume("vol1"));
    CHECK(p->volume("vol1") == nullptr);
    // Second remove is a no-op.
    CHECK_FALSE(p->removeSingleVolume("vol1"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::reloadSingleVolume returns true after a successful reload")
{
    auto d = tmpDir("reloadvol");
    makeLocalVolume(d);
    auto p = VolumePkg::newEmpty();
    p->addVolumeEntry((d / "volumes").string());
    p->addSingleVolume("vol1");
    CHECK(p->reloadSingleVolume("vol1"));
    CHECK_FALSE(p->reloadSingleVolume("unknown"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::addVolume directly inserts a constructed Volume")
{
    auto d = tmpDir("addvol_direct");
    auto volDir = makeLocalVolume(d);
    auto v = Volume::New(volDir);
    REQUIRE(v);
    auto p = VolumePkg::newEmpty();
    CHECK(p->addVolume(v));
    CHECK(p->volume(v->id()) != nullptr);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::attachPreparedVolume persists one loaded entry idempotently")
{
    auto d = tmpDir("attach_prepared");
    const auto firstPath = makeLocalVolume(d, "vol1");
    const auto secondRoot = d / "second";
    const auto secondPath = makeLocalVolume(secondRoot, "vol1");
    const auto project = d / "project.volpkg.json";

    auto package = VolumePkg::newEmpty();
    package->save(project);
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
    auto reloaded = VolumePkg::load(project, deferred);
    REQUIRE(reloaded->volumeEntries().size() == 1);
    CHECK(reloaded->volumeEntries().front().location == firstPath.string());
    CHECK(reloaded->volumeEntries().front().tags ==
          std::vector<std::string>{"overlay"});
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::attachPreparedVolume recognizes a loaded relative entry")
{
    auto d = tmpDir("attach_relative");
    const auto volumePath = makeLocalVolume(d, "vol1");
    const auto project = d / "project.volpkg.json";

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto detached = VolumePkg::newDetached(deferred);
    REQUIRE(detached->addVolumeEntry("volumes/vol1", {"original"}));
    detached->save(project);

    const auto relativeProject = fs::relative(project, fs::current_path());
    auto package = VolumePkg::load(relativeProject);
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
    const auto project = d / "project.volpkg.json";

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto detached = VolumePkg::newDetached(deferred);
    REQUIRE(detached->addVolumeEntry("volumes/vol1", {"original"}));
    detached->save(project);

    const auto alias = aliasFor(d);
    if (alias.empty()) {
        fs::remove_all(d);
        return;
    }

    // Package opened through the alias, volume attached through the real path.
    auto package = VolumePkg::load(alias / "project.volpkg.json");
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
    const auto project = d / "project.volpkg.json";

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto detached = VolumePkg::newDetached(deferred);
    // "volumes" is a collection entry: it holds zarr volumes but is not one.
    REQUIRE(detached->addVolumeEntry("volumes", {"collection"}));
    detached->save(project);

    const auto alias = aliasFor(d);
    if (alias.empty()) {
        fs::remove_all(d);
        return;
    }

    auto package = VolumePkg::load(alias / "project.volpkg.json");
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
    const auto project = d / "project.volpkg.json";

    auto package = VolumePkg::newEmpty();
    package->save(project);
    auto prepared = Volume::New(volumePath);
    REQUIRE(prepared);
    REQUIRE(package->addVolume(prepared));

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
    const auto project = d / "project.volpkg.json";

    auto package = VolumePkg::newEmpty();
    package->save(project);
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
    auto package = VolumePkg::newDetached(deferred);
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

TEST_CASE("VolumePkg::attachPreparedVolume restores autosave after write failure")
{
    auto d = tmpDir("attach_rollback");
    const auto volumePath = makeLocalVolume(d, "vol1");
    const auto project = d / "project.volpkg.json";

    auto package = VolumePkg::newEmpty();
    package->save(project);
    fs::remove(project);
    fs::create_directory(project);

    auto prepared = Volume::New(volumePath);
    REQUIRE(prepared);
    CHECK_THROWS(
        package->attachPreparedVolume(volumePath.string(), {}, prepared));
    CHECK(package->volumeEntries().empty());
    CHECK(package->volume("vol1") == nullptr);

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto autosave = VolumePkg::load(VolumePkg::autosaveFile(), deferred);
    CHECK(autosave->volumeEntries().empty());

    fs::remove_all(d);
}

TEST_CASE("A catalog gives remote entries their catalog volume identity at their level")
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
    CHECK(source.at("voxelsize").get_double() == doctest::Approx(2.5));
    const auto level = vc::project::catalogVolumeMetadata(&catalog, "https://example.test/s/vol.zarr#vc-base-scale=2");
    CHECK(level.at("uuid").get_string() == "vol@L2");
    CHECK(level.at("voxelsize").get_double() == doctest::Approx(10.0));
    CHECK(level.at("name").get_string() == "scan.zarr [L2]");
    const auto prediction = vc::project::catalogVolumeMetadata(&catalog, "https://example.test/s/pred.zarr");
    CHECK_FALSE(prediction.contains("uuid"));
    CHECK(prediction.at("voxelsize").get_double() == doctest::Approx(5.0));
    CHECK(vc::project::catalogVolumeMetadata(&catalog, "https://example.test/elsewhere.zarr").empty());
    CHECK(vc::project::catalogVolumeMetadata(nullptr, "https://example.test/s/vol.zarr").empty());
}

