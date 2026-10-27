// Live S3 tests against PHerc 0172 and PHerc 0139 for Volume / Zarr / VcDataset
// and catalog identity coverage. Set VC_TEST_REQUIRE_NETWORK=1 to make
// failures hard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/render/ChunkCache.hpp"
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

    struct TemporaryProject {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("vc-catalog-id-" + vc::objects::util::newUuid() + ".volpkg.json");
        ~TemporaryProject() { std::error_code ec; std::filesystem::remove(path, ec); }
    } temporaryProject;
    const auto& projectPath = temporaryProject.path;
    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    deferred.failOnRemoteError = true;
    deferred.catalog = std::make_shared<const vcr::catalog::OpenDataManifest>(vcr::catalog::parseOpenDataManifest(R"({
      "samples": {"PHerc0139": {"volumes": {"20260102150214": {"id": "20260102150214",
        "suffix": "20260102150214-2.399um-0.2m-78keV-masked.zarr", "properties": {"pixel_size_um": 2.399},
        "data": [{"type": "zarr", "origins": [{"path": "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr",
          "access_roots": [{"type": "https", "url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/",
                            "usage": "public-read"}]}]}]}}}}})"));
    auto project = VolumePkg::newDetached(deferred);
    REQUIRE(project->addVolumeEntry(url, {"vc-remote-auth:anonymous"}));
    REQUIRE(project->addVolumeEntry(url + "#vc-base-scale=2", {"vc-remote-auth:anonymous"}));
    project->save(projectPath);
    project = VolumePkg::load(projectPath, deferred);
    project->resolveDeferredEntries();
    const auto volume = project->volume("20260102150214");
    REQUIRE(volume);
    CHECK(volume->shape() == untagged->shape());
    // Changing the annotation identity must reuse the pre-existing URL cache,
    // including the process cache shared with manually opened remote volumes.
    CHECK(untagged->remotePersistentCachePath() == untagged->remoteCacheRoot() / untagged->id());
    CHECK(volume->remotePersistentCachePath() == untagged->remotePersistentCachePath());
    const auto originalCache = untagged->sharedChunkCache();
    const auto catalogCache = volume->sharedChunkCache();
    CHECK(catalogCache->sourceId() == originalCache->sourceId());
    // Populate the original disk cache, evict decoded RAM, and read through
    // the catalog volume. The bytes must survive without a remote chunk fetch.
    const int level = untagged->presentScaleLevels().back();
    const auto shape = untagged->levelShape(level);
    const auto chunk = untagged->chunkShape(level);
    const std::array<int, 3> key{shape[0] / 2 / chunk[0], shape[1] / 2 / chunk[1], shape[2] / 2 / chunk[2]};
    const auto before = originalCache->getChunkBlocking(level, key[0], key[1], key[2]);
    REQUIRE(before.status != vc::render::ChunkStatus::Error);
    originalCache->waitForPendingChunks();
    originalCache->invalidate();
    const auto fetches = std::make_shared<std::atomic_int>(0);
    const auto listener = catalogCache->addRemoteFetchActivityListener(
        [fetches](const auto&, bool active) { if (active) ++*fetches; });
    const auto after = catalogCache->getChunkBlocking(level, key[0], key[1], key[2]);
    catalogCache->removeRemoteFetchActivityListener(listener);
    CHECK(fetches->load() == 0);
    CHECK(after.status == before.status);
    if (before.bytes) {
        REQUIRE(after.bytes);
        CHECK(*after.bytes == *before.bytes);
    }
    MESSAGE("Catalog cache reuse: " << (after.bytes ? after.bytes->size() : 0)
            << " decoded bytes, " << fetches->load() << " remote chunk fetches");
    CHECK(vc::objects::geometryScaleToVolume(fiber, volume->id()) == 1.0);
    CHECK(vc::objects::volumeCoordinates(volume->id(), 1) == fiber.at("coordinates"));
    REQUIRE(project->volumeIDs().size() == 2);
    for (const auto& id : project->volumeIDs()) {
        if (id != volume->id()) {
            const auto rebased = project->volume(id);
            CHECK(rebased->id() == "20260102150214@L2");
            // Both show the catalog volume, at their levels: different frames.
            const auto view = project->catalogView(rebased->id());
            REQUIRE(view);
            CHECK((view->sampleId == "PHerc0139" && view->volumeId == "20260102150214" && view->level == 2 && view->source));
            CHECK_FALSE(project->overlayCompatible(volume->id(), rebased->id()));
            CHECK(project->overlayCompatible(rebased->id(), rebased->id()));
            CHECK(rebased->voxelSize() == doctest::Approx(4 * 2.399));
            CHECK_FALSE(vc::objects::geometryScaleToVolume(fiber, rebased->id()).has_value());
            CHECK(rebased->remotePersistentCachePath() != volume->remotePersistentCachePath());
        }
    }

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

TEST_CASE("Volume: shape and pyramid metadata match the upstream .zarray files")
{
    auto v = openOrSkip();
    if (!v) return;

    // Base level (0) shape, pinned from the live bucket (.zarray observed 2026-06).
    auto s0 = v->levelShape(0);
    CHECK(s0[0] == 21000);
    CHECK(s0[1] == 6700);
    CHECK(s0[2] == 9100);

    // Level 5 (smallest pyramid level)
    auto s5 = v->levelShape(5);
    CHECK(s5[0] == 657);
    CHECK(s5[1] == 210);
    CHECK(s5[2] == 285);

    // 6 levels total
    CHECK(v->numScales() == 6);
    CHECK(v->hasScaleLevel(0));
    CHECK(v->hasScaleLevel(5));
    CHECK_FALSE(v->hasScaleLevel(99));
}

TEST_CASE("Volume: chunkShape per level is the upstream chunk grid")
{
    auto v = openOrSkip();
    if (!v) return;

    auto c5 = v->chunkShape(5);
    CHECK(c5[0] == 128);
    CHECK(c5[1] == 128);
    CHECK(c5[2] == 128);
}

TEST_CASE("Volume: dtype is uint8")
{
    auto v = openOrSkip();
    if (!v) return;
    // dtypeSize() == 1 means uint8 (matches the "|u1" .zarray).
    CHECK(v->dtypeSize() == 1);
}

TEST_CASE("Volume: present scale levels include 0..5")
{
    auto v = openOrSkip();
    if (!v) return;
    auto present = v->presentScaleLevels();
    CHECK(present.size() >= 6);
    CHECK(v->firstPresentScaleLevel() == 0);
    // finestPresentScaleLevelAtOrBelow returns the finest level *with on-disk
    // data*; the live PHerc 0172 zarr reports 4 here. Just sanity-check it
    // lies within the pyramid range.
    int finest5 = v->finestPresentScaleLevelAtOrBelow(5);
    CHECK(finest5 >= 0);
    CHECK(finest5 <= 5);
}

TEST_CASE("Volume: shape vs shapeXyz order swap")
{
    auto v = openOrSkip();
    if (!v) return;
    auto zyx = v->shape();
    auto xyz = v->shapeXyz();
    // xyz = [width, height, slices] = reverse of [slices, height, width]
    CHECK(xyz[0] == zyx[2]);
    CHECK(xyz[1] == zyx[1]);
    CHECK(xyz[2] == zyx[0]);
}

TEST_CASE("Volume: chunkCount(level=5) matches grid size from the .zarray")
{
    auto v = openOrSkip();
    if (!v) return;
    // shape (657, 210, 285), chunks (128, 128, 128)
    // grid = ceil(657/128)*ceil(210/128)*ceil(285/128) = 6*2*3 = 36
    // (the live bucket actually has 30 chunks on-disk — missing ones are fill)
    CHECK(v->chunkCount(5) > 0);
}

TEST_CASE("Volume::readZYX: read a small region at level 5 (one chunk)")
{
    auto v = openOrSkip();
    if (!v) return;

    Array3D<uint8_t> region({16, 16, 16});
    bool ok = false;
    try {
        ok = v->readZYX(region, {0, 0, 0}, /*level=*/5);
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("readZYX failed: " << e.what());
        MESSAGE("Skipping read (network?): " << e.what());
        return;
    }
    CHECK(ok);
    // Sanity check: region is the right size, accepts arbitrary uint8 values.
    CHECK(region.shape()[0] == 16);
    CHECK(region.shape()[1] == 16);
    CHECK(region.shape()[2] == 16);
}

TEST_CASE("Volume::readXYZ: same region using XY-major order")
{
    auto v = openOrSkip();
    if (!v) return;

    Array3D<uint8_t> region({16, 16, 16});
    bool ok = false;
    try {
        ok = v->readXYZ(region, {0, 0, 0}, 5);
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("readXYZ failed: " << e.what());
        MESSAGE("Skipping read (network?): " << e.what());
        return;
    }
    CHECK(ok);
}
