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
using vc::project::validateSingleVolumeLocation;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_pkg_" + tag + "_" + std::to_string(rng()));
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
    CHECK_FALSE(validateLocation(
        Category::Volumes,
        "https://example.test/volume.zarr#unknown=2").empty());
}

TEST_CASE("validateSingleVolumeLocation requires exactly one zarr")
{
    auto root = tmpDir("single_volume");
    auto volume = root / "volume";
    fs::create_directories(volume / "0");
    { std::ofstream(volume / "meta.json") << "{}"; }
    { std::ofstream(volume / "0" / ".zarray") << "{}"; }

    CHECK(validateSingleVolumeLocation(volume.string()).empty());
    CHECK_FALSE(validateSingleVolumeLocation(root.string()).empty());
    CHECK(validateSingleVolumeLocation(
        "s3://bucket/volume.zarr#vc-base-scale=2").empty());
    CHECK_FALSE(validateSingleVolumeLocation(
        "https://example.test/not-a-zarr").empty());
    CHECK_FALSE(validateSingleVolumeLocation(
        "https://example.test/volume.zarr#unknown=2").empty());
    fs::remove_all(root);
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

TEST_CASE("VolumePkg::newDetached writes only when explicitly saved")
{
    const auto autosave = VolumePkg::autosaveFile();
    fs::create_directories(autosave.parent_path());
    {
        std::ofstream out(autosave);
        out << "current session";
    }

    auto d = tmpDir("detached");
    auto target = d / "created.volpkg.json";
    vc::project::LoadOptions opts;
    opts.deferResolution = true;
    auto p = VolumePkg::newDetached(opts);
    p->setName("Created");
    CHECK(p->addVolumeEntry("/volume", {"source:test"}));

    std::ifstream beforeSave(autosave);
    CHECK(std::string(
              std::istreambuf_iterator<char>(beforeSave),
              std::istreambuf_iterator<char>()) == "current session");
    CHECK_FALSE(fs::exists(target));

    p->save(target);
    REQUIRE(fs::exists(target));
    auto loaded = VolumePkg::load(target, opts);
    CHECK(loaded->name() == "Created");
    REQUIRE(loaded->volumeEntries().size() == 1);
    CHECK(loaded->volumeEntries().front().location == "/volume");

    p->setName("Changed later");
    CHECK(VolumePkg::load(target, opts)->name() == "Created");

    std::ifstream afterSave(autosave);
    CHECK(std::string(
              std::istreambuf_iterator<char>(afterSave),
              std::istreambuf_iterator<char>()) == "current session");
    fs::remove_all(d);
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

TEST_CASE("VolumePkg: segment discovery skips transient cache directories")
{
    auto d = tmpDir("seg_transients");
    auto writeSegMeta = [](const fs::path& segDir, const std::string& uuid) {
        fs::create_directories(segDir);
        std::ofstream f(segDir / "meta.json");
        f << R"({"type":"seg","uuid":")" << uuid << R"(","format":"tifxyz"})";
    };

    writeSegMeta(d / "stable-seg", "stable-seg");
    writeSegMeta(d / "stable-seg.tmp-12345", "stable-seg.tmp-12345");
    writeSegMeta(d / "stable-seg.previous", "stable-seg.previous");

    auto p = VolumePkg::newEmpty();
    CHECK(p->addSegmentsEntry(d.string()));
    const auto ids = p->segmentationIDs();
    REQUIRE(ids.size() == 1);
    CHECK(ids.front() == "stable-seg");

    fs::remove_all(d);
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

TEST_CASE("VolumePkg: saving over an existing project replaces it")
{
    auto d = tmpDir("save_overwrite");
    auto jsonPath = d / "project.json";

    auto p = VolumePkg::newEmpty();
    p->setName("First");
    p->save(jsonPath);
    REQUIRE(fs::exists(jsonPath));

    p->setName("Second");
    p->save(jsonPath);

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->name() == "Second");
    fs::remove_all(d);
}

TEST_CASE("VolumePkg: missing selected_lasagna_dataset loads as empty")
{
    auto d = tmpDir("lasagna_missing");
    auto jsonPath = d / "project.json";
    {
        auto p = VolumePkg::newEmpty();
        p->save(jsonPath);
    }

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->selectedLasagnaDataset().empty());
    CHECK(loaded->selectedLasagnaDatasetPath().empty());
    fs::remove_all(d);
}

TEST_CASE("VolumePkg: selected_lasagna_dataset round-trips through save/load")
{
    auto d = tmpDir("lasagna_roundtrip");
    auto jsonPath = d / "project.json";
    const std::string manifest = (d / "dataset.lasagna.json").string();
    {
        auto p = VolumePkg::newEmpty();
        p->setSelectedLasagnaDataset(manifest);
        CHECK(p->selectedLasagnaDataset() == manifest);
        p->save(jsonPath);
    }

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->selectedLasagnaDataset() == manifest);
    CHECK(loaded->selectedLasagnaDatasetPath() == fs::path(manifest));

    loaded->clearSelectedLasagnaDataset();
    CHECK(loaded->selectedLasagnaDataset().empty());
    CHECK(loaded->selectedLasagnaDatasetPath().empty());
    fs::remove_all(d);
}

TEST_CASE("VolumePkg: selectedLasagnaDatasetPath resolves relative to project file")
{
    auto d = tmpDir("lasagna_relative");
    auto jsonPath = d / "project.json";
    {
        auto p = VolumePkg::newEmpty();
        p->save(jsonPath);
        p->setSelectedLasagnaDataset("datasets/reference.lasagna.json");
    }

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->selectedLasagnaDataset() == "datasets/reference.lasagna.json");
    CHECK(loaded->selectedLasagnaDatasetPath() ==
          d / "datasets" / "reference.lasagna.json");
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

TEST_CASE("VolumePkg reconciles and relocates coordinate-bearing asset entries")
{
    auto d = tmpDir("asset_reconcile");
    const auto oldSegments = (d / "segments-old").string();
    const auto newSegments = (d / "segments-new").string();
    const auto grids = (d / "grids").string();

    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addSegmentsEntry(
        oldSegments,
        {"user-tag", "vc-open-data-coordinate-space:sample/source@L0"}));
    CHECK(pkg->reconcileSegmentsEntryTags(
        oldSegments,
        {"vc-open-data-coordinate-space:sample/source@L2",
         "vc-open-data-source-coordinate-level:2"},
        {"vc-open-data-coordinate-space:",
         "vc-open-data-source-coordinate-level:"}));
    REQUIRE(pkg->segmentEntries().size() == 1);
    CHECK(std::find(pkg->segmentEntries()[0].tags.begin(),
                    pkg->segmentEntries()[0].tags.end(),
                    "user-tag") != pkg->segmentEntries()[0].tags.end());
    CHECK(std::find(pkg->segmentEntries()[0].tags.begin(),
                    pkg->segmentEntries()[0].tags.end(),
                    "vc-open-data-coordinate-space:sample/source@L0") ==
          pkg->segmentEntries()[0].tags.end());
    CHECK(pkg->relocateSegmentsEntry(oldSegments, newSegments));
    CHECK(pkg->segmentEntries()[0].location == newSegments);

    REQUIRE(pkg->addNormalGridEntry(
        grids, {"vc-open-data-source-coordinate-level:0"}));
    CHECK(pkg->reconcileNormalGridEntryTags(
        grids, {"vc-open-data-source-coordinate-level:2"},
        {"vc-open-data-source-coordinate-level:"}));
    CHECK(pkg->normalGridEntries()[0].tags.back() ==
          "vc-open-data-source-coordinate-level:2");
    fs::remove_all(d);
}

TEST_CASE("VolumePkg persists manifest-backed Lasagna entries independently of normal grids")
{
    auto d = tmpDir("lasagna_entries");
    const auto project = d / "project.json";
    const auto lasagna = (d / "cache" / "data.lasagna.json").string();
    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addLasagnaDatasetEntry(
        lasagna,
        {"vc-open-data-lasagna", "vc-open-data-volume-id:vol-a"}));
    CHECK(pkg->normalGridEntries().empty());
    pkg->save(project);

    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto loaded = VolumePkg::load(project, options);
    REQUIRE(loaded);
    REQUIRE(loaded->lasagnaDatasetEntries().size() == 1);
    CHECK(loaded->lasagnaDatasetEntries().front().location == lasagna);
    CHECK(loaded->normalGridEntries().empty());
    fs::remove_all(d);
}

TEST_CASE("VolumePkg canonicalizes virtual locators and deduplicates explicit base zero")
{
    auto d = tmpDir("remote_selector_identity");
    const auto jsonPath = d / "project.json";
    {
        std::ofstream out(jsonPath);
        out << R"({"name":"selectors","version":1,"volumes":[{"location":"s3://bucket/source.zarr","tags":[]}]})";
    }
    vc::project::LoadOptions opts;
    opts.deferResolution = true;
    auto pkg = VolumePkg::load(jsonPath, opts);
    REQUIRE(pkg);
    CHECK_FALSE(pkg->addVolumeEntry(
        "s3://bucket/source.zarr#vc-base-scale=0"));
    CHECK(pkg->addVolumeEntry(
        "s3://bucket/source.zarr/#vc-base-scale=02"));
    CHECK_FALSE(pkg->addVolumeEntry(
        "https://bucket.s3.us-east-1.amazonaws.com/source.zarr#vc-base-scale=2"));
    REQUIRE(pkg->volumeEntries().size() == 2);
    CHECK(pkg->volumeEntries()[0].location == "s3://bucket/source.zarr");
    CHECK(pkg->volumeEntries()[1].location ==
          "https://bucket.s3.us-east-1.amazonaws.com/source.zarr#vc-base-scale=2");
    fs::remove_all(d);
}
