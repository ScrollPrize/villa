// Coverage for core/src/VolumePkg.cpp — focuses on the JSON project file
// lifecycle (newEmpty/save/load), entry add/remove, validators, and the
// free vc::project helpers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"

#include <algorithm>
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
    CHECK_FALSE(validateLocation(Category::NormalGrids, "").empty());
}

TEST_CASE("validateLocation: remote allowed only for Volumes")
{
    CHECK(validateLocation(Category::Volumes, "s3://b/k").empty());
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

TEST_CASE("validateLocation: empty directory not a valid volume/normalgrid")
{
    auto d = tmpDir("empty");
    CHECK_FALSE(validateLocation(Category::Volumes, d.string()).empty());
    CHECK_FALSE(validateLocation(Category::NormalGrids, d.string()).empty());
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
        p->addNormalGridEntry("/ng-x");
        p->save(jsonPath);
    }
    REQUIRE(fs::exists(jsonPath));

    auto loaded = VolumePkg::load(jsonPath);
    REQUIRE(loaded);
    CHECK(loaded->name() == "Roundtrip");
    CHECK(loaded->volumeEntries().size() == 1);
    CHECK(loaded->volumeEntries()[0].location == "/vol-x");
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

    loaded->clearSelectedLasagnaDataset();
    CHECK(loaded->selectedLasagnaDataset().empty());
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
    REQUIRE(pkg->addLasagnaDatasetEntry("fiber.lasagna.json", {"model:v1", std::string(vc::project::kFiberLasagnaTag)}));
    pkg->setSelectedLasagnaDataset("regular.lasagna.json");
    pkg->setSelectedFiberInferenceDataset("fiber.lasagna.json");
    pkg->save(project);

    const auto json = utils::Json::parse_file(project);
    REQUIRE(json.contains("lasagna_datasets"));
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

TEST_CASE("VolumePkg attaches Lasagna manifest volumes as regular volume entries")
{
    auto d = tmpDir("lasagna_batch_attach");
    vc::test::ZarrVolumeSpec create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "lasagna-view";
    create.name = "presence";
    auto volume = vc::test::createZarrVolume(d / "backing.zarr", create);

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
    vc::test::ZarrVolumeSpec create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "old-channel";
    create.name = "old";
    auto oldVolume = vc::test::createZarrVolume(d / "old.zarr", create);
    create.uuid = "new-channel";
    create.name = "new";
    auto newVolume = vc::test::createZarrVolume(d / "new.zarr", create);

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
        pkg->attachPreparedLasagnaDataset("data.lasagna.json", {"model:new"}, false, newAttachments) ==
        VolumePkg::AttachLasagnaResult::Attached);

    REQUIRE(pkg->lasagnaDatasetEntries().size() == 1);
    CHECK(pkg->fiberInferenceDatasetEntries().empty());
    CHECK(pkg->selectedLasagnaDataset() == "data.lasagna.json");
    CHECK(pkg->selectedFiberInferenceDataset().empty());
    const auto& manifestTags = pkg->allLasagnaDatasetEntries().front().tags;
    CHECK(std::find(manifestTags.begin(), manifestTags.end(), "model:new") != manifestTags.end());
    REQUIRE(pkg->volumeEntries().size() == 2);
    CHECK(pkg->hasVolume("old-channel"));
    CHECK(pkg->hasVolume("new-channel"));
    fs::remove_all(d);
}

TEST_CASE("VolumePkg merges Lasagna manifest breadcrumbs onto shared regular volumes")
{
    auto d = tmpDir("lasagna_shared_ownership");
    vc::test::ZarrVolumeSpec create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "shared-lasagna-view";
    create.name = "shared";
    auto volume = vc::test::createZarrVolume(d / "shared.zarr", create);

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
    vc::test::ZarrVolumeSpec create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "independent-lasagna-source";
    create.name = "source";
    const auto location = (d / "source.zarr").string();
    auto volume = vc::test::createZarrVolume(location, create);

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
    vc::test::ZarrVolumeSpec create;
    create.shapeZYX = {2, 2, 2};
    create.chunkShapeZYX = {2, 2, 2};
    create.numLevels = 1;
    create.uuid = "source";
    create.name = "source";
    const auto location = (d / "source.zarr").string();
    auto prepared = vc::test::createZarrVolume(d / "other.zarr", create);

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

