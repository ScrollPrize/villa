// More VolumePkg coverage:
// - focus bounds (the author's view)
// - directory
// - volumeTags
// - normalGridPaths

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
#include "support/TestProject.hpp"

#include <filesystem>
#include <fstream>
#include <random>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_vpkg_more_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

TEST_CASE("Project focus bounds persist independently of enabled state")
{
    const auto store = vc::test::createTestProject();
    auto project = store.open();
    CHECK_FALSE(project->focusBounds().has_value());
    CHECK_FALSE(project->focusBoundsEnabled());
    project->setFocusBounds(Rect3D{{30, 20, 10}, {1, 2, 3}}, true);
    auto loaded = store.open();
    REQUIRE(loaded->focusBounds().has_value());
    CHECK(loaded->focusBounds()->low == cv::Vec3f(1, 2, 3));
    CHECK(loaded->focusBounds()->high == cv::Vec3f(30, 20, 10));
    CHECK(loaded->focusBoundsEnabled());

    loaded->setFocusBounds(loaded->focusBounds(), false);
    auto disabled = store.open();
    REQUIRE(disabled->focusBounds().has_value());
    CHECK_FALSE(disabled->focusBoundsEnabled());
    CHECK(disabled->focusBounds()->high == cv::Vec3f(30, 20, 10));
    disabled->setFocusBounds(std::nullopt, false);
    auto cleared = store.open();
    CHECK_FALSE(cleared->focusBounds().has_value());
    CHECK_FALSE(cleared->focusBoundsEnabled());
    CHECK_FALSE(vc::test::emptyProject()->focusBounds().has_value());
}

TEST_CASE("The volume an author last viewed is kept with the project")
{
    const auto store = vc::test::createTestProject();
    CHECK(store.open()->currentVolumeId().empty());
    store.open()->setCurrentVolumeId("scan-a");
    CHECK(store.open()->currentVolumeId() == "scan-a");
    CHECK(VolumePkg::open(store.session, store.attachment, "someone-else")->currentVolumeId().empty());
}

fs::path makeLocalVolume(const fs::path& root, const std::string& id = "vol1")
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

TEST_CASE("directory is the project store's directory")
{
    auto d = tmpDir("vpkg_dir");
    const auto store = vc::test::createTestProject(d / "project.vcobj");
    CHECK(store.open()->directory() == d);
    CHECK(store.open()->path() == d / "project.vcobj");
    fs::remove_all(d);
}

TEST_CASE("volumeTags return tags by id")
{
    auto d = tmpDir("tags");
    makeLocalVolume(d);
    auto p = vc::test::emptyProject();
    p->addVolumeEntry((d / "volumes").string(), {"voltag1", "voltag2"});
    // Volume ID is the uuid from meta.json
    auto v = p->volume("vol1");
    REQUIRE(v);
    auto vt = p->volumeTags("vol1");
    CHECK(vt.size() == 2);
    fs::remove_all(d);
}

TEST_CASE("isRemote is false for purely-local projects")
{
    auto d = tmpDir("isremote");
    makeLocalVolume(d);
    auto p = vc::test::emptyProject();
    p->addVolumeEntry((d / "volumes").string());
    CHECK_FALSE(p->isRemote());
    fs::remove_all(d);
}

TEST_CASE("normalGridPaths: empty without entries")
{
    auto p = vc::test::emptyProject();
    CHECK(p->normalGridPaths().empty());
}

TEST_CASE("volumeIDs returns loaded volume ids")
{
    auto d = tmpDir("vol_ids");
    makeLocalVolume(d);
    auto p = vc::test::emptyProject();
    p->addVolumeEntry((d / "volumes").string());
    auto ids = p->volumeIDs();
    CHECK_FALSE(ids.empty());
    CHECK(ids[0] == "vol1");
    fs::remove_all(d);
}
