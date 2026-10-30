// Live S3 tests against PHerc 0172 and PHerc 0139 for Volume / Zarr / VcDataset
// and catalog identity coverage. Set VC_TEST_REQUIRE_NETWORK=1 to make
// failures hard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "support/TestProject.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_reader/catalog.hpp>

#include <cstdlib>
#include <atomic>
#include <exception>
#include <memory>
#include <string>

namespace {

constexpr const char* kVolumeUrl =
    "s3://vesuvius-challenge-open-data/PHerc0172/volumes/"
    "20241024131838-7.910um-53keV-masked.zarr";

bool requireNetwork()
{
    const char* env = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return env && env[0] && env[0] != '0';
}

std::shared_ptr<Volume> openOrSkip()
{
    try {
        return Volume::NewFromUrl(kVolumeUrl);
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("Volume::NewFromUrl failed: " << e.what());
        MESSAGE("Skipping (network unavailable?): " << e.what());
        return nullptr;
    }
}

} // namespace

TEST_CASE("Catalog PHerc0139 scan reopens with the object-store volume identity")
{
    const std::string url =
        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/"
        "20260102150214-2.399um-0.2m-78keV-masked.zarr";
    std::shared_ptr<Volume> untagged;
    try {
        untagged = Volume::NewFromUrl(url, {}, utils::Json::object(), false);
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("PHerc0139 remote open failed: " << e.what());
        MESSAGE("Skipping (network unavailable?): " << e.what());
        return;
    }
    const nlohmann::json fiber = {
        {"coordinates", {{"volume_uuid", "20260102150214"}, {"scale_factor", 1}}}};
    // Reproduce the URL-derived identity mismatch before the catalog names the volume.
    CHECK_FALSE(vc::objects::geometryScaleToVolume(fiber, untagged->id()).has_value());

    const auto store = vc::test::createTestProject();
    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    deferred.failOnRemoteError = true;
    deferred.catalog = std::make_shared<const vcr::catalog::OpenDataManifest>(vcr::catalog::parseOpenDataManifest(R"({
      "samples": {"PHerc0139": {"volumes": {"20260102150214": {"id": "20260102150214",
        "suffix": "20260102150214-2.399um-0.2m-78keV-masked.zarr", "properties": {"pixel_size_um": 2.399},
        "data": [{"type": "zarr", "origins": [{"path": "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr",
          "access_roots": [{"type": "https", "url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/",
                            "usage": "public-read"}]}]}]}}}}})"));
    auto project = store.open(deferred);
    REQUIRE(project->addVolumeEntry(url, {"vc-remote-auth:anonymous"}));
    // A working level is the same entry, not a second volume.
    CHECK_FALSE(project->addVolumeEntry(url + "#vc-base-scale=2", {"vc-remote-auth:anonymous"}));
    project = store.open(deferred);
    project->resolveDeferredEntries();
    const auto volume = project->volume("20260102150214");
    REQUIRE(volume);
    CHECK(volume->shape() == untagged->shape());
    // Changing the annotation identity must reuse the pre-existing URL cache,
    // including the process cache shared with manually opened remote volumes.
    CHECK(untagged->remotePersistentCachePath() == untagged->remoteCacheRoot() / untagged->id());
    CHECK(volume->remotePersistentCachePath() == untagged->remotePersistentCachePath());
    // vc_reader opens both at the same source with the same VC3D cache directory.
    CHECK(volume->sourceLocation() == untagged->sourceLocation());
    CHECK(volume->sourceOpenOptions().vc3dVolumeDir == untagged->sourceOpenOptions().vc3dVolumeDir);
    CHECK(vc::objects::geometryScaleToVolume(fiber, volume->id()) == 1.0);
    CHECK(vc::objects::volumeCoordinates(volume->id(), 1) == fiber.at("coordinates"));
    REQUIRE(project->volumeIDs() == std::vector<std::string>{"20260102150214"});
    CHECK(vc::project::annotationVolume(project.get(), *volume).volume == "20260102150214");

    // The same volume at working level 2: its ID, entry and objects stay; its
    // voxels are 4 of the volume's own.
    const auto levelZeroCache = volume->remotePersistentCachePath();
    REQUIRE(project->setVolumeLevel("20260102150214", 2));
    CHECK_FALSE(project->setVolumeLevel("20260102150214", 2));
    const auto coarse = project->volume("20260102150214");
    REQUIRE(coarse);
    CHECK(coarse != volume);
    CHECK(coarse->baseScaleLevel() == 2);
    CHECK(project->volumeEntries().size() == 1);
    CHECK(project->volumeEntries().front().location == url + "#vc-base-scale=2");
    const auto view = project->catalogView(coarse->id());
    REQUIRE(view);
    CHECK((view->sampleId == "PHerc0139" && view->volumeId == "20260102150214" && view->level == 2 && view->source));
    CHECK(project->overlayCompatible(coarse->id(), coarse->id()));
    CHECK(coarse->voxelSize() == doctest::Approx(4 * 2.399));
    CHECK(coarse->remotePersistentCachePath() != levelZeroCache);
    // Objects made on it are kept in the volume's own voxels, at its scale.
    const auto annotation = vc::project::annotationVolume(project.get(), *coarse);
    CHECK(annotation.volume == "20260102150214");
    CHECK(annotation.scale == 0.25);
    REQUIRE(annotation.voxelSizeUm);
    CHECK(*annotation.voxelSizeUm == doctest::Approx(2.399));
    // The project reopens at the level, and goes back to level 0.
    project = store.open(deferred);
    project->resolveDeferredEntries();
    REQUIRE(project->volume("20260102150214"));
    CHECK(project->volume("20260102150214")->baseScaleLevel() == 2);
    REQUIRE(project->setVolumeLevel("20260102150214", 0));
    CHECK(project->volumeEntries().front().location == url);
    CHECK(project->volume("20260102150214")->remotePersistentCachePath() == levelZeroCache);

    // Optional read-only validation against a local replica of the PHerc0139 hub.
    if (const char* path = std::getenv("VC_TEST_CATALOG_FIBER_STORE")) {
        vc::objects::OpenOptions options;
        options.readOnly = true;
        auto store = vc::objects::ObjectStore::open(path, options);
        auto txn = store.read();
        vc::objects::ObjectSelection selection;
        selection.kind = "fiber";
        const auto all = store.selectObjects(txn, selection);
        REQUIRE_FALSE(all.empty());
        selection.volume = untagged->id();
        CHECK(store.selectObjects(txn, selection).empty());
        selection.volume = volume->id();
        const auto matched = store.selectObjects(txn, selection);
        CHECK(matched.size() == all.size());
        CHECK(store.unboundObjectCount(txn, "fiber", volume->id()) == 0);
        for (const auto& object : matched)
            CHECK(vc::objects::geometryScaleToVolume(nlohmann::json::parse(object.meta), volume->id()) == 1.0);
        MESSAGE("Matched " << matched.size() << " real hub fibers to " << volume->id());
    }
}

TEST_CASE("Volume::NewFromUrl: opens the PHerc 0172 zarr")
{
    auto v = openOrSkip();
    if (!v) return;
    CHECK(v->isRemote());
    CHECK_FALSE(v->remoteUrl().empty());
    CHECK_FALSE(v->id().empty());
    CHECK(v->remotePersistentCachePath() == v->remoteCacheRoot() / v->id());
}

TEST_CASE("Volume::NewFromUrl: invalid credentials fall back to anonymous public S3")
{
    vc::HttpAuth invalid;
    invalid.access_key = "INVALIDACCESSKEY";
    invalid.secret_key = "INVALIDSECRET";
    invalid.session_token = "INVALIDSESSIONTOKEN";
    invalid.region = "us-east-1";

    try {
        const auto volume = Volume::NewFromUrl(kVolumeUrl, invalid);
        REQUIRE(volume != nullptr);
        CHECK(volume->isRemote());
        CHECK(volume->remoteAuth().empty());
    } catch (const std::exception& e) {
        if (requireNetwork()) {
            FAIL("invalid-credential anonymous fallback failed: " << e.what());
        }
        MESSAGE("Skipping invalid-credential fallback (network unavailable?): "
                << e.what());
    }
}
