// Coverage for core/src/VolumePkg.cpp — focuses on the JSON project file
// lifecycle (newEmpty/save/load), entry add/remove, validators, and the
// free vc::project helpers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"

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

TEST_CASE("VolumePkg: segment entries match normalized local paths")
{
    auto d = tmpDir("segment_entry_identity");
    fs::create_directories(d / "segments");

    auto p = VolumePkg::newEmpty();
    p->save(d / "project.volpkg.json");
    REQUIRE(p->addSegmentsEntry("segments", {"source:test"}));

    const auto matching = p->matchingSegmentsEntry((d / "segments").string() + "/");
    REQUIRE(matching);
    CHECK(matching->location == "segments");
    CHECK(matching->tags == std::vector<std::string>{"source:test"});
    CHECK_FALSE(p->addSegmentsEntry((d / "segments").string()));
    CHECK(p->segmentEntries().size() == 1);

    std::error_code symlinkError;
    fs::create_directory_symlink(
        d / "segments", d / "segments-alias", symlinkError);
    if (!symlinkError) {
        CHECK(p->matchingSegmentsEntry((d / "segments-alias").string()));
        CHECK_FALSE(p->addSegmentsEntry((d / "segments-alias").string()));
    }

#ifndef _WIN32
    fs::create_directories(d / "segments\\");
    CHECK_FALSE(p->matchingSegmentsEntry((d / "segments\\").string()));
    CHECK(p->addSegmentsEntry((d / "segments\\").string()));
#endif

    fs::remove_all(d);
}

TEST_CASE("VolumePkg: segment directory-name lookup is case-insensitive")
{
    auto d = tmpDir("segment_source_name");
    fs::create_directories(d / "one" / "User-Segments");

    auto p = VolumePkg::newEmpty();
    p->save(d / "project.volpkg.json");
    REQUIRE(p->addSegmentsEntry(
        (d / "one" / "User-Segments").string()));

    const auto matching =
        p->matchingSegmentsEntryByDirectoryName("user-segments");
    REQUIRE(matching);
    CHECK(matching->location ==
          (d / "one" / "User-Segments").string());

    fs::remove_all(d);
}

TEST_CASE("VolumePkg: segment attachment rolls back after a write failure")
{
    auto d = tmpDir("segment_attach_rollback");
    auto makeSegment = [](const fs::path& path, const std::string& id) {
        fs::create_directories(path);
        std::ofstream meta(path / "meta.json");
        meta << R"({"type":"seg","uuid":")" << id
             << R"(","format":"tifxyz"})";
    };
    makeSegment(d / "initial", "initial");
    makeSegment(d / "new", "new");

    const auto project = d / "project.volpkg.json";
    auto p = VolumePkg::newEmpty();
    p->save(project);
    REQUIRE(p->addSegmentsEntry((d / "initial").string()));
    fs::remove(project);
    fs::create_directory(project);

    CHECK_THROWS(p->attachSegmentsEntry(
        (d / "new").string(), {"source:test"}, true));
    REQUIRE(p->segmentEntries().size() == 1);
    CHECK(p->segmentEntries().front().location == (d / "initial").string());
    CHECK(p->outputSegmentsPath() == d / "initial");
    CHECK(p->segmentationIDs() == std::vector<std::string>{"initial"});

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto autosave = VolumePkg::load(VolumePkg::autosaveFile(), deferred);
    REQUIRE(autosave->segmentEntries().size() == 1);
    CHECK(autosave->segmentEntries().front().location ==
          (d / "initial").string());

    fs::remove_all(d);
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

TEST_CASE("VolumePkg: missing umbilicus loads as empty")
{
    auto d = tmpDir("umbilicus_missing");
    auto jsonPath = d / "project.json";
    {
        auto p = VolumePkg::newEmpty();
        p->save(jsonPath);
    }

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->umbilicus().empty());
    CHECK(loaded->umbilicusPath().empty());
    fs::remove_all(d);
}

TEST_CASE("VolumePkg: umbilicus round-trips through save/load")
{
    auto d = tmpDir("umbilicus_roundtrip");
    auto jsonPath = d / "project.json";
    const std::string location = (d / "umbilicus.json").string();
    {
        auto p = VolumePkg::newEmpty();
        p->setUmbilicus(location);
        CHECK(p->umbilicus() == location);
        p->save(jsonPath);
    }

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->umbilicus() == location);
    CHECK(loaded->umbilicusPath() == fs::path(location));

    // An empty location clears the field, and the cleared state persists.
    loaded->setUmbilicus("");
    CHECK(loaded->umbilicus().empty());
    CHECK(loaded->umbilicusPath().empty());
    loaded->save(jsonPath);
    auto reloaded = VolumePkg::load(jsonPath);
    REQUIRE(reloaded);
    CHECK(reloaded->umbilicus().empty());
    fs::remove_all(d);
}

TEST_CASE("VolumePkg: umbilicusPath resolves relative to project file")
{
    auto d = tmpDir("umbilicus_relative");
    auto jsonPath = d / "project.json";
    {
        auto p = VolumePkg::newEmpty();
        p->save(jsonPath);
        p->setUmbilicus("annotations/umbilicus.json");
    }

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->umbilicus() == "annotations/umbilicus.json");
    CHECK(loaded->umbilicusPath() == d / "annotations" / "umbilicus.json");
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

TEST_CASE("VolumePkg stores regular and fiber Lasagna entries in one canonical array")
{
    auto d = tmpDir("canonical_lasagna_roles");
    const auto project = d / "project.json";
    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addLasagnaDatasetEntry("regular.lasagna.json", {"user-tag"}));
    REQUIRE(pkg->addFiberInferenceDatasetEntry("fiber.lasagna.json", {"model:v1"}));
    pkg->setSelectedLasagnaDataset("regular.lasagna.json");
    pkg->setSelectedFiberInferenceDataset("fiber.lasagna.json");
    pkg->save(project);

    const auto json = utils::Json::parse_file(project);
    REQUIRE(json.contains("lasagna_datasets"));
    CHECK_FALSE(json.contains("fiber_inference_datasets"));
    REQUIRE(json.at("lasagna_datasets").size() == 2);

    vc::project::LoadOptions options;
    options.deferResolution = true;
    const auto loaded = VolumePkg::load(project, options);
    REQUIRE(loaded->lasagnaDatasetEntries().size() == 1);
    REQUIRE(loaded->fiberInferenceDatasetEntries().size() == 1);
    CHECK(loaded->fiberInferenceDatasetEntries().front().location == "fiber.lasagna.json");
    CHECK(vc::project::isFiberLasagnaEntry(loaded->fiberInferenceDatasetEntries().front()));
    CHECK(loaded->selectedLasagnaDataset() == "regular.lasagna.json");
    CHECK(loaded->selectedFiberInferenceDataset() == "fiber.lasagna.json");
    fs::remove_all(d);
}

TEST_CASE("VolumePkg migrates the legacy fiber inference array without losing tags")
{
    auto d = tmpDir("legacy_fiber_lasagna");
    const auto project = d / "project.json";
    {
        std::ofstream out(project);
        out << R"({
          "name":"legacy",
          "version":1,
          "lasagna_datasets":[
            {"location":"shared.lasagna.json","tags":["canonical-tag"]}
          ],
          "fiber_inference_datasets":[
            {"location":"shared.lasagna.json","tags":["legacy-tag"]},
            {"location":"fiber.lasagna.json","tags":[]}
          ],
          "selected_fiber_inference_dataset":"fiber.lasagna.json"
        })";
    }
    vc::project::LoadOptions options;
    options.deferResolution = true;
    const auto loaded = VolumePkg::load(project, options);
    REQUIRE(loaded->fiberInferenceDatasetEntries().size() == 2);
    REQUIRE(loaded->allLasagnaDatasetEntries().size() == 2);
    const auto& shared = loaded->allLasagnaDatasetEntries().front();
    CHECK(std::find(shared.tags.begin(), shared.tags.end(), "canonical-tag") != shared.tags.end());
    CHECK(std::find(shared.tags.begin(), shared.tags.end(), "legacy-tag") != shared.tags.end());
    CHECK(vc::project::isFiberLasagnaEntry(shared));

    loaded->save(project);
    const auto saved = utils::Json::parse_file(project);
    CHECK_FALSE(saved.contains("fiber_inference_datasets"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg attaches Lasagna manifest volumes as regular volume entries")
{
    auto d = tmpDir("lasagna_batch_attach");
    Volume::ZarrCreateOptions create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "lasagna-view";
    create.name = "presence";
    create.compressor.clear();
    auto volume = Volume::New(d / "backing.zarr", create);

    VolumePkg::PreparedVolumeAttachment derived{
        (d / "backing.zarr").string(),
        {"vc-lasagna-manifest:fiber.lasagna.json", "vc-lasagna-group:presence"},
        volume,
    };
    auto pkg = VolumePkg::newEmpty();
    CHECK(pkg->attachPreparedLasagnaDataset("fiber.lasagna.json", {"user-tag"}, true, {derived}) == VolumePkg::AttachLasagnaResult::Attached);
    REQUIRE(pkg->fiberInferenceDatasetEntries().size() == 1);
    REQUIRE(pkg->volumeEntries().size() == 1);
    CHECK(pkg->hasVolume("lasagna-view"));
    CHECK(pkg->selectedFiberInferenceDataset() == "fiber.lasagna.json");
    CHECK(pkg->allLasagnaDatasetEntries().front().tags ==
          std::vector<std::string>{"user-tag", "vc-lasagna-fiber"});

    CHECK(pkg->removeEntry("fiber.lasagna.json"));
    CHECK(pkg->allLasagnaDatasetEntries().empty());
    REQUIRE(pkg->volumeEntries().size() == 1);
    CHECK(pkg->hasVolume("lasagna-view"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg Lasagna attach fills but does not replace selected datasets")
{
    auto pkg = VolumePkg::newEmpty();

    REQUIRE(pkg->attachPreparedLasagnaDataset(
                "first.lasagna.json", {}, false, {}) ==
            VolumePkg::AttachLasagnaResult::Attached);
    CHECK(pkg->selectedLasagnaDataset() == "first.lasagna.json");

    REQUIRE(pkg->attachPreparedLasagnaDataset(
                "second.lasagna.json", {}, false, {}) ==
            VolumePkg::AttachLasagnaResult::Attached);
    CHECK(pkg->selectedLasagnaDataset() == "first.lasagna.json");

    REQUIRE(pkg->attachPreparedLasagnaDataset(
                "fiber-a.lasagna.json", {}, true, {}) ==
            VolumePkg::AttachLasagnaResult::Attached);
    CHECK(pkg->selectedFiberInferenceDataset() == "fiber-a.lasagna.json");

    REQUIRE(pkg->attachPreparedLasagnaDataset(
                "fiber-b.lasagna.json", {}, true, {}) ==
            VolumePkg::AttachLasagnaResult::Attached);
    CHECK(pkg->selectedFiberInferenceDataset() == "fiber-a.lasagna.json");
}

TEST_CASE("VolumePkg Lasagna reattachment updates manifest role without owning volumes")
{
    auto d = tmpDir("lasagna_batch_reconcile");
    Volume::ZarrCreateOptions create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "old-channel";
    create.name = "old";
    create.compressor.clear();
    auto oldVolume = Volume::New(d / "old.zarr", create);
    create.uuid = "new-channel";
    create.name = "new";
    auto newVolume = Volume::New(d / "new.zarr", create);

    const std::vector<VolumePkg::PreparedVolumeAttachment> oldAttachments{{
        (d / "old.zarr").string(),
        {"vc-lasagna-manifest:data.lasagna.json", "vc-lasagna-group:old"},
        oldVolume,
    }};
    const std::vector<VolumePkg::PreparedVolumeAttachment> newAttachments{{
        (d / "new.zarr").string(),
        {"vc-lasagna-manifest:data.lasagna.json", "vc-lasagna-group:new"},
        newVolume,
    }};

    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->attachPreparedLasagnaDataset("data.lasagna.json", {"model:old"}, true, oldAttachments) == VolumePkg::AttachLasagnaResult::Attached);
    REQUIRE(
        pkg->attachPreparedLasagnaDataset("data.lasagna.json", {"model:new"}, false, newAttachments, {}, true, true, {"model:"}) ==
        VolumePkg::AttachLasagnaResult::Attached);

    REQUIRE(pkg->lasagnaDatasetEntries().size() == 1);
    CHECK(pkg->fiberInferenceDatasetEntries().empty());
    CHECK(pkg->selectedLasagnaDataset() == "data.lasagna.json");
    CHECK(pkg->selectedFiberInferenceDataset().empty());
    const auto& manifestTags = pkg->allLasagnaDatasetEntries().front().tags;
    CHECK(std::find(manifestTags.begin(), manifestTags.end(), "model:old") == manifestTags.end());
    CHECK(std::find(manifestTags.begin(), manifestTags.end(), "model:new") != manifestTags.end());
    REQUIRE(pkg->volumeEntries().size() == 2);
    CHECK(pkg->hasVolume("old-channel"));
    CHECK(pkg->hasVolume("new-channel"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg merges Lasagna manifest breadcrumbs onto shared regular volumes")
{
    auto d = tmpDir("lasagna_shared_ownership");
    Volume::ZarrCreateOptions create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "shared-lasagna-view";
    create.name = "shared";
    create.compressor.clear();
    auto volume = Volume::New(d / "shared.zarr", create);

    const VolumePkg::PreparedVolumeAttachment first{
        (d / "shared.zarr").string(),
        {"vc-lasagna-manifest:first.lasagna.json", "vc-lasagna-group:first"},
        volume,
    };
    const VolumePkg::PreparedVolumeAttachment second{
        (d / "shared.zarr").string(),
        {"vc-lasagna-manifest:second.lasagna.json", "vc-lasagna-group:second"},
        volume,
    };
    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->attachPreparedLasagnaDataset("first.lasagna.json", {}, false, {first}) == VolumePkg::AttachLasagnaResult::Attached);
    REQUIRE(pkg->attachPreparedLasagnaDataset("second.lasagna.json", {}, false, {second}) == VolumePkg::AttachLasagnaResult::Attached);

    REQUIRE(pkg->removeEntry("first.lasagna.json"));
    REQUIRE(pkg->volumeEntries().size() == 1);
    CHECK(pkg->hasVolume("shared-lasagna-view"));
    CHECK(std::find(pkg->volumeEntries().front().tags.begin(), pkg->volumeEntries().front().tags.end(), "vc-lasagna-manifest:first.lasagna.json") !=
          pkg->volumeEntries().front().tags.end());
    CHECK(std::find(pkg->volumeEntries().front().tags.begin(), pkg->volumeEntries().front().tags.end(), "vc-lasagna-manifest:second.lasagna.json") !=
          pkg->volumeEntries().front().tags.end());
    fs::remove_all(d);
}

TEST_CASE("VolumePkg Lasagna attach can tag an independently attached source volume")
{
    auto d = tmpDir("lasagna_independent_ownership");
    Volume::ZarrCreateOptions create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "independent-lasagna-source";
    create.name = "source";
    create.compressor.clear();
    const auto location = (d / "source.zarr").string();
    auto volume = Volume::New(location, create);

    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->attachPreparedVolume(location, {}, volume) ==
            VolumePkg::AttachVolumeResult::Attached);
    const VolumePkg::PreparedVolumeAttachment derived{
        location,
        {"vc-lasagna-manifest:data.lasagna.json", "vc-lasagna-group:source"},
        volume,
    };
    REQUIRE(pkg->attachPreparedLasagnaDataset(
                "data.lasagna.json", {}, false, {derived}) ==
            VolumePkg::AttachLasagnaResult::Attached);
    REQUIRE(pkg->volumeEntries().size() == 1);
    CHECK(std::find(pkg->volumeEntries().front().tags.begin(),
                    pkg->volumeEntries().front().tags.end(),
                    "vc-lasagna-manifest:data.lasagna.json") !=
          pkg->volumeEntries().front().tags.end());

    REQUIRE(pkg->removeEntry("data.lasagna.json"));
    REQUIRE(pkg->volumeEntries().size() == 1);
    CHECK(pkg->volumeEntries().front().location == location);
    CHECK(pkg->hasVolume("independent-lasagna-source"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg rejects Lasagna prepared volumes with mismatched locations")
{
    auto d = tmpDir("lasagna_mismatched_location");
    Volume::ZarrCreateOptions create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "source";
    create.name = "source";
    create.compressor.clear();
    const auto location = (d / "source.zarr").string();
    auto prepared = Volume::New(d / "other.zarr", create);

    auto pkg = VolumePkg::newEmpty();
    const VolumePkg::PreparedVolumeAttachment derived{
        location,
        {"vc-lasagna-manifest:data.lasagna.json", "vc-lasagna-group:source"},
        prepared,
    };

    CHECK_THROWS_WITH_AS(
        pkg->attachPreparedLasagnaDataset(
            "data.lasagna.json", {}, false, {derived}),
        doctest::Contains("does not match its attachment location"),
        std::invalid_argument);
    CHECK(pkg->allLasagnaDatasetEntries().empty());
    CHECK(pkg->volumeEntries().empty());
    CHECK_FALSE(pkg->hasVolume("source"));
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
