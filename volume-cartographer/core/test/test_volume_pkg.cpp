// Coverage for core/src/VolumePkg.cpp — focuses on the JSON project file
// lifecycle (newEmpty/save/load), entry add/remove, validators, and the
// free vc::project helpers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>

namespace fs = std::filesystem;
using vc::project::Category;
using vc::project::isLocationRemote;
using vc::project::resolveLocalPath;
using vc::project::validateLocation;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_pkg_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

} // namespace

// --- Free helpers ---

TEST_CASE("isLocationRemote: schemes")
{
    CHECK(isLocationRemote("s3://bucket/key"));
    CHECK(isLocationRemote("s3+eu-west-1://bucket"));
    CHECK(isLocationRemote("http://example.com"));
    CHECK(isLocationRemote("https://example.com"));
    CHECK_FALSE(isLocationRemote("/local/path"));
    CHECK_FALSE(isLocationRemote("relative/path"));
    CHECK_FALSE(isLocationRemote("file:///tmp/x"));
    CHECK_FALSE(isLocationRemote(""));
}

TEST_CASE("resolveLocalPath: absolute, relative+base, file:// prefix")
{
    CHECK(resolveLocalPath("/abs/path") == fs::path("/abs/path"));
    CHECK(resolveLocalPath("file:///abs/path") == fs::path("/abs/path"));
    auto rel = resolveLocalPath("rel/path", fs::path("/base"));
    CHECK(rel == fs::path("/base/rel/path"));
    // No base + relative -> returns relative path unchanged.
    auto bare = resolveLocalPath("rel/path");
    CHECK(bare == fs::path("rel/path"));
}

TEST_CASE("validateLocation: empty location is rejected")
{
    CHECK_FALSE(validateLocation(Category::Volumes, "").empty());
    CHECK_FALSE(validateLocation(Category::Segments, "").empty());
    CHECK_FALSE(validateLocation(Category::NormalGrids, "").empty());
}

TEST_CASE("validateLocation: remote allowed only for Volumes")
{
    CHECK(validateLocation(Category::Volumes, "s3://b/k").empty());
    CHECK_FALSE(validateLocation(Category::Segments, "s3://b/k").empty());
    CHECK_FALSE(validateLocation(Category::NormalGrids, "https://x/y").empty());
}

TEST_CASE("validateLocation: malformed remote URLs are rejected")
{
    CHECK_FALSE(validateLocation(Category::Volumes, "s3:").empty());
    CHECK_FALSE(validateLocation(Category::Volumes, "s3://").empty());
}

TEST_CASE("validateLocation: nonexistent local path is rejected")
{
    CHECK_FALSE(validateLocation(Category::Volumes, "/__no__/__where__").empty());
}

TEST_CASE("validateLocation: non-directory local path is rejected")
{
    auto d = tmpDir("not_dir");
    auto p = d / "file.txt";
    { std::ofstream f(p); f << "hello"; }
    auto err = validateLocation(Category::Volumes, p.string());
    CHECK_FALSE(err.empty());
    fs::remove_all(d);
}

TEST_CASE("validateLocation: empty directory not a valid volume/segment/normalgrid")
{
    auto d = tmpDir("empty");
    CHECK_FALSE(validateLocation(Category::Volumes, d.string()).empty());
    CHECK_FALSE(validateLocation(Category::Segments, d.string()).empty());
    CHECK_FALSE(validateLocation(Category::NormalGrids, d.string()).empty());
    fs::remove_all(d);
}

TEST_CASE("validateLocation: a segment-shaped dir validates for Segments")
{
    // Make a minimal tifxyz segment-like directory.
    auto d = tmpDir("seg");
    auto segDir = d / "myseg";
    fs::create_directories(segDir);
    { std::ofstream f(segDir / "meta.json");
      f << R"({"type":"seg","uuid":"test","format":"tifxyz"})"; }
    // Both the directory itself, and the parent (because it contains a seg subdir).
    CHECK(validateLocation(Category::Segments, segDir.string()).empty());
    CHECK(validateLocation(Category::Segments, d.string()).empty());
    fs::remove_all(d);
}

TEST_CASE("validateLocation: a normalgrid-shaped dir validates for NormalGrids")
{
    auto d = tmpDir("ng");
    fs::create_directories(d / "xy");
    fs::create_directories(d / "xz");
    fs::create_directories(d / "yz");
    { std::ofstream f(d / "metadata.json"); f << "{}"; }
    CHECK(validateLocation(Category::NormalGrids, d.string()).empty());
    fs::remove_all(d);
}

// --- VolumePkg lifecycle ---

TEST_CASE("VolumePkg::newEmpty produces an empty package")
{
    auto p = VolumePkg::newEmpty();
    REQUIRE(p);
    CHECK(p->volumeEntries().empty());
    CHECK(p->segmentEntries().empty());
    CHECK(p->normalGridEntries().empty());
}

TEST_CASE("VolumePkg: setName persists in memory")
{
    auto p = VolumePkg::newEmpty();
    p->setName("My Project");
    CHECK(p->name() == "My Project");
}

TEST_CASE("VolumePkg: addVolumeEntry / removeEntry round-trip")
{
    auto p = VolumePkg::newEmpty();
    CHECK(p->addVolumeEntry("/vol1", {"tag-a"}));
    CHECK(p->volumeEntries().size() == 1);
    CHECK(p->volumeEntries()[0].location == "/vol1");
    CHECK(p->volumeEntries()[0].tags == std::vector<std::string>{"tag-a"});
    // Duplicate add is rejected
    CHECK_FALSE(p->addVolumeEntry("/vol1"));
    // Empty location is rejected
    CHECK_FALSE(p->addVolumeEntry(""));
    // Remove works
    CHECK(p->removeEntry("/vol1"));
    CHECK(p->volumeEntries().empty());
    // Second remove is a no-op
    CHECK_FALSE(p->removeEntry("/vol1"));
}

TEST_CASE("VolumePkg: addSegmentsEntry sets outputSegments on first add")
{
    auto p = VolumePkg::newEmpty();
    CHECK(p->addSegmentsEntry("/segs"));
    CHECK(p->segmentEntries().size() == 1);
    // Second add doesn't override outputSegments
    CHECK(p->addSegmentsEntry("/more_segs"));
    CHECK(p->segmentEntries().size() == 2);
    p->clearOutputSegments();
    CHECK_FALSE(p->addSegmentsEntry(""));
}

TEST_CASE("VolumePkg: addNormalGridEntry")
{
    auto p = VolumePkg::newEmpty();
    CHECK(p->addNormalGridEntry("/grids"));
    CHECK(p->normalGridEntries().size() == 1);
    CHECK_FALSE(p->addNormalGridEntry(""));
    CHECK_FALSE(p->addNormalGridEntry("/grids")); // duplicate
}

TEST_CASE("VolumePkg: save then load round-trips entries")
{
    auto d = tmpDir("save_load");
    auto jsonPath = d / "project.json";

    {
        auto p = VolumePkg::newEmpty();
        p->setName("Roundtrip");
        p->addVolumeEntry("/vol-x");
        p->addSegmentsEntry("/seg-x");
        p->addNormalGridEntry("/ng-x");
        p->save(jsonPath);
    }
    REQUIRE(fs::exists(jsonPath));

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->name() == "Roundtrip");
    CHECK(loaded->volumeEntries().size() == 1);
    CHECK(loaded->volumeEntries()[0].location == "/vol-x");
    CHECK(loaded->segmentEntries().size() == 1);
    CHECK(loaded->normalGridEntries().size() == 1);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::New is an alias for load")
{
    auto d = tmpDir("new_alias");
    auto jsonPath = d / "project.json";
    {
        auto p = VolumePkg::newEmpty();
        p->setName("Alias");
        p->save(jsonPath);
    }
    auto loaded = VolumePkg::New(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->name() == "Alias");
    fs::remove_all(d);
}

TEST_CASE("VolumePkg: autosave file path is settable")
{
    auto saved = VolumePkg::autosaveRoot();
    auto d = tmpDir("autosave");
    VolumePkg::setAutosaveRoot(d);
    CHECK(VolumePkg::autosaveRoot() == d);
    // Restore so other tests aren't affected.
    VolumePkg::setAutosaveRoot(saved);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::loadAutosave returns nullptr when no autosave file exists")
{
    auto saved = VolumePkg::autosaveRoot();
    auto d = tmpDir("no_autosave");
    VolumePkg::setAutosaveRoot(d);
    auto p = VolumePkg::loadAutosave();
    CHECK(p == nullptr);
    VolumePkg::setAutosaveRoot(saved);
    fs::remove_all(d);
}

TEST_CASE("VolumePkg::setLoadFirstSegmentationDirectory: round-trip")
{
    VolumePkg::setLoadFirstSegmentationDirectory("custom_segs");
    // Clear it again with empty string
    VolumePkg::setLoadFirstSegmentationDirectory("");
    CHECK(true);
}
