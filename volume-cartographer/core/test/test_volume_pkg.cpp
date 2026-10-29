// Coverage for core/src/VolumePkg.cpp — a project kept in a standalone file
// (open/reopen), entry add/remove, validators, and the free vc::project helpers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
#include "support/TestProject.hpp"

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

TEST_CASE("A new project is empty")
{
    auto p = vc::test::emptyProject();
    REQUIRE(p);
    CHECK(p->volumeEntries().empty());
    CHECK(p->normalGridEntries().empty());
    CHECK(p->name() == "test");
}

TEST_CASE("VolumePkg: setName persists in the project file")
{
    const auto store = vc::test::createTestProject();
    auto p = store.open();
    p->setName("My Project");
    CHECK(p->name() == "My Project");
    CHECK(store.open()->name() == "My Project");
}

TEST_CASE("VolumePkg: addVolumeEntry / removeEntry round-trip")
{
    auto p = vc::test::emptyProject();
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
    auto p = vc::test::emptyProject();
    CHECK(p->addNormalGridEntry("/grids"));
    CHECK(p->normalGridEntries().size() == 1);
    CHECK_FALSE(p->addNormalGridEntry(""));
    CHECK_FALSE(p->addNormalGridEntry("/grids")); // duplicate
}

TEST_CASE("VolumePkg: reopening the project returns its entries")
{
    const auto store = vc::test::createTestProject();
    {
        auto p = store.open();
        p->setName("Roundtrip");
        p->addVolumeEntry("/vol-x");
        p->addNormalGridEntry("/ng-x");
    }
    auto loaded = store.open();
    REQUIRE(loaded);
    CHECK(loaded->name() == "Roundtrip");
    CHECK(loaded->volumeEntries().size() == 1);
    CHECK(loaded->volumeEntries()[0].location == "/vol-x");
    CHECK(loaded->normalGridEntries().size() == 1);
}

TEST_CASE("VolumePkg: the selected Lasagna dataset is kept in the project file")
{
    const auto store = vc::test::createTestProject();
    CHECK(store.open()->selectedLasagnaDataset().empty());
    const std::string manifest = "/data/dataset.lasagna.json";
    store.open()->setSelectedLasagnaDataset(manifest);
    auto loaded = store.open();
    CHECK(loaded->selectedLasagnaDataset() == manifest);
    // Reopening the document restores the same view.
    CHECK(VolumePkg::open(store.path)->selectedLasagnaDataset() == manifest);
    loaded->clearSelectedLasagnaDataset();
    CHECK(store.open()->selectedLasagnaDataset().empty());
}

TEST_CASE("VolumePkg resources belong to the project file")
{
    const auto store = vc::test::createTestProject();
    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto first = store.open(options);
    REQUIRE(first->addVolumeEntry("s3://bucket/shared.zarr"));
    REQUIRE(first->addVolumeEntry("/home/alice/predictions.zarr"));
    auto second = store.open(options);
    CHECK(second->volumeEntries() == first->volumeEntries());
    REQUIRE(second->removeEntry("/home/alice/predictions.zarr"));
    CHECK(first->reload());
    CHECK(first->volumeEntries().size() == 1);
}

TEST_CASE("VolumePkg follows entries another session adds or removes")
{
    const auto store = vc::test::createTestProject();
    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto mine = store.open(options);
    auto theirs = store.open(options);
    CHECK_FALSE(mine->reload());
    REQUIRE(theirs->addVolumeEntry("/their-volume"));
    CHECK(mine->reload());
    REQUIRE(mine->volumeEntries().size() == 1);
    CHECK(mine->volumeEntries().front().location == "/their-volume");
    REQUIRE(theirs->removeEntry("/their-volume"));
    CHECK(mine->reload());
    CHECK(mine->volumeEntries().empty());
}

TEST_CASE("VolumePkg persists manifest-backed Lasagna entries independently of normal grids")
{
    auto d = tmpDir("lasagna_entries");
    const auto store = vc::test::createTestProject();
    const auto lasagna = (d / "cache" / "data.lasagna.json").string();
    auto pkg = store.open();
    REQUIRE(pkg->addLasagnaDatasetEntry(
        lasagna,
        {"vc-open-data-lasagna", "vc-open-data-volume-id:vol-a"}));
    CHECK(pkg->normalGridEntries().empty());

    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto loaded = store.open(options);
    REQUIRE(loaded);
    REQUIRE(loaded->lasagnaDatasetEntries().size() == 1);
    CHECK(loaded->lasagnaDatasetEntries().front().location == lasagna);
    CHECK(loaded->normalGridEntries().empty());
    fs::remove_all(d);
}

TEST_CASE("VolumePkg stores regular and fiber Lasagna entries in one resource kind")
{
    const auto store = vc::test::createTestProject();
    auto pkg = store.open();
    REQUIRE(pkg->addLasagnaDatasetEntry("regular.lasagna.json", {"user-tag"}));
    REQUIRE(pkg->addLasagnaDatasetEntry("fiber.lasagna.json", {"model:v1", std::string(vc::project::kFiberLasagnaTag)}));
    pkg->setSelectedLasagnaDataset("regular.lasagna.json");
    pkg->setSelectedFiberInferenceDataset("fiber.lasagna.json");
    store.session->read(store.attachment, [](const auto& db, const auto& txn, const auto&) {
        CHECK(vc::objects::projectResources(db, txn, vc::objects::kLasagnaDatasetResource).empty());
    });

    vc::project::LoadOptions options;
    options.deferResolution = true;
    const auto loaded = store.open(options);
    REQUIRE(loaded->lasagnaDatasetEntries().size() == 1);
    REQUIRE(loaded->fiberInferenceDatasetEntries().size() == 1);
    CHECK(loaded->fiberInferenceDatasetEntries().front().location == "fiber.lasagna.json");
    CHECK(vc::project::isFiberLasagnaEntry(loaded->fiberInferenceDatasetEntries().front()));
    CHECK(loaded->selectedLasagnaDataset() == "regular.lasagna.json");
    CHECK(loaded->selectedFiberInferenceDataset() == "fiber.lasagna.json");
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
    auto pkg = vc::test::emptyProject();
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
    auto pkg = vc::test::emptyProject();

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

    auto pkg = vc::test::emptyProject();
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
    auto pkg = vc::test::emptyProject();
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

    auto pkg = vc::test::emptyProject();
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

    auto pkg = vc::test::emptyProject();
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

TEST_CASE("VolumePkg keeps one entry per volume at any working level")
{
    vc::project::LoadOptions opts;
    opts.deferResolution = true;
    auto pkg = vc::test::emptyProject(opts);
    REQUIRE(pkg->addVolumeEntry("s3://bucket/source.zarr"));
    CHECK_FALSE(pkg->addVolumeEntry(
        "s3://bucket/source.zarr#vc-base-scale=0"));
    CHECK_FALSE(pkg->addVolumeEntry(
        "s3://bucket/source.zarr/#vc-base-scale=02"));
    CHECK_FALSE(pkg->addVolumeEntry(
        "https://bucket.s3.us-east-1.amazonaws.com/source.zarr#vc-base-scale=2"));
    REQUIRE(pkg->volumeEntries().size() == 1);
    CHECK(pkg->volumeEntries()[0].location == "s3://bucket/source.zarr");
    CHECK(pkg->addVolumeEntry("s3://bucket/other.zarr#vc-base-scale=1"));
    CHECK(pkg->volumeEntries().back().location ==
          "https://bucket.s3.us-east-1.amazonaws.com/other.zarr#vc-base-scale=1");
}


TEST_CASE("Project documents preserve store references and resource ownership on Save As")
{
    const auto root = tmpDir("document_copy");
    const auto project = vc::test::createTestProject(root / "original.vcproj");
    auto document = vc::project::ProjectDocument::load(project.path);
    auto data = document.data();
    data["volumes"].push_back({{"location", "volumes/scan.zarr"}, {"tags", {"vc-lasagna-derived:predictions/manifest.json"}}});
    data["lasagna_datasets"].push_back({{"location", "predictions/manifest.json"}, {"tags", nlohmann::json::array()}});
    data["view"]["selected_lasagna_dataset"] = "predictions/manifest.json";
    data["normal_grids"].push_back({{"location", "s3://bucket/grid"}, {"tags", nlohmann::json::array()}});
    document.replace(data);
    fs::create_directory(root / "copies");
    auto copy = document.saveAs(root / "copies" / "copy.vcproj");
    CHECK(copy.resolve(copy.data()["stores"][0]["location"]) == project.storePath);
    CHECK(copy.data()["volumes"][0]["location"] == "../volumes/scan.zarr");
    CHECK(copy.data()["volumes"][0]["tags"][0] == "vc-lasagna-derived:../predictions/manifest.json");
    CHECK(copy.data()["view"]["selected_lasagna_dataset"] == "../predictions/manifest.json");
    CHECK(copy.data()["normal_grids"][0]["location"] == "s3://bucket/grid");
    auto changed = copy.data();
    changed["name"] = "Independent name";
    copy.replace(changed);
    CHECK(vc::project::ProjectDocument::load(project.path).data() == data);
    CHECK_THROWS(copy.saveAs(project.path));
    CHECK_THROWS(vc::project::ProjectDocument::load(project.storePath));
    fs::remove_all(root);
}

TEST_CASE("Project document validation and failed saves leave existing state intact")
{
    const auto project = vc::test::createTestProject();
    auto document = vc::project::ProjectDocument::load(project.path);
    const auto original = document.data();
    auto invalid = original;
    invalid["version"] = 2;
    CHECK_THROWS(document.replace(invalid));
    invalid = original;
    invalid["stores"] = nlohmann::json::array();
    CHECK_THROWS(document.replace(invalid));
    invalid = original;
    invalid["selections"]["fiber"] = "missing-id";
    CHECK_THROWS(document.replace(invalid));
    CHECK(document.data() == original);
    CHECK(vc::project::ProjectDocument::load(project.path).data() == original);
    fs::permissions(project.path, fs::perms::owner_read);
    CHECK_FALSE(document.writable());
    auto changed = original;
    changed["name"] = "Should fail";
    CHECK_THROWS(document.replace(changed));
    CHECK(document.data() == original);
    CHECK(vc::project::ProjectDocument::load(project.path).data() == original);
    fs::permissions(project.path, fs::perms::owner_all);
}
