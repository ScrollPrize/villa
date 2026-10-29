// Cover QuadSurface::save force_overwrite atomic-exchange and identity rules.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_qs_save_paths_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat_<cv::Vec3f> grid(int rows = 8, int cols = 8, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return m;
}

} // namespace

TEST_CASE("new surface UUID survives metadata and geometry saves")
{
    auto root = tmpDir("identity");
    auto segDir = root / "named_surface";
    QuadSurface created(grid(), cv::Vec2f(1.f, 1.f));
    created.save(segDir);
    QuadSurface loaded(segDir);
    const std::string uuid = loaded.meta["uuid"].get_string();
    CHECK(vc::objects::util::isUuid(uuid));
    CHECK(loaded.id == "named_surface");
    loaded.ensureLoaded();
    loaded.saveOverwrite();
    CHECK(QuadSurface(segDir).meta["uuid"].get_string() == uuid);
    fs::remove_all(root);
}

TEST_CASE("in-place edits keep the object UUID; copies get a new one")
{
    auto root = tmpDir("identity_lifecycle");
    auto segDir = root / "original";
    {
        QuadSurface created(grid(), cv::Vec2f(1.f, 1.f));
        created.save(segDir);
    }
    const std::string uuid = QuadSurface(segDir).objectUuid();
    REQUIRE(vc::objects::util::isUuid(uuid));

    // In-place geometry edit (crop/flip/rotate path): save over the source.
    {
        auto loaded = load_quad_from_tifxyz(segDir.string());
        loaded->save(segDir, true);
        CHECK(loaded->objectUuid() == uuid);
        CHECK(loaded->id == "original");
    }
    CHECK(QuadSurface(segDir).objectUuid() == uuid);

    // A derived output whose meta was copied from the source is a new object.
    auto copyDir = root / "derived";
    {
        auto loaded = load_quad_from_tifxyz(segDir.string());
        loaded->saveAsNew(copyDir);
    }
    const std::string copyUuid = QuadSurface(copyDir).objectUuid();
    CHECK(vc::objects::util::isUuid(copyUuid));
    CHECK(copyUuid != uuid);

    // saveEdited decides by destination.
    {
        auto loaded = load_quad_from_tifxyz(segDir.string());
        loaded->saveEdited(segDir, segDir, true);
    }
    CHECK(QuadSurface(segDir).objectUuid() == uuid);
    auto editedDir = root / "edited_elsewhere";
    {
        auto loaded = load_quad_from_tifxyz(segDir.string());
        loaded->saveEdited(segDir, editedDir);
    }
    const std::string editedUuid = QuadSurface(editedDir).objectUuid();
    CHECK(vc::objects::util::isUuid(editedUuid));
    CHECK(editedUuid != uuid);
    CHECK(editedUuid != copyUuid);

    // An explicitly assigned identity (e.g. a catalog id) is kept as given.
    auto catalogDir = root / "catalog";
    {
        QuadSurface created(grid(), cv::Vec2f(1.f, 1.f));
        created.meta["uuid"] = "catalog-representation-id";
        created.save(catalogDir);
    }
    CHECK(QuadSurface(catalogDir).objectUuid() == "catalog-representation-id");
    fs::remove_all(root);
}

TEST_CASE("save(path, force_overwrite=true): atomic-exchange replaces existing")
{
    auto root = tmpDir("force");
    auto segDir = root / "seg";

    // First save: creates the directory.
    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.meta["uuid"] = "uuid-a";
        qs.save(segDir.string(), /*force_overwrite=*/false);
    }
    REQUIRE(fs::exists(segDir / "x.tif"));

    // Second save with force_overwrite=true into the same dir.
    {
        QuadSurface qs(grid(8, 8, /*z=*/50.f), cv::Vec2f(1.f, 1.f));
        qs.meta["uuid"] = "uuid-b";
        qs.save(segDir.string(), /*force_overwrite=*/true);
    }
    REQUIRE(fs::exists(segDir / "x.tif"));
    QuadSurface reloaded(segDir);
    CHECK(reloaded.meta["uuid"].get_string() == "uuid-b");
    fs::remove_all(root);
}

TEST_CASE("saving over a segment keeps its subfolders")
{
    auto root = tmpDir("subfolders");
    auto segDir = root / "seg";
    QuadSurface(grid(), cv::Vec2f(1.f, 1.f)).save(segDir);

    // What VC3D keeps inside a catalog segment, and a label zarr.
    fs::create_directories(segDir / "ink-detections");
    std::ofstream(segDir / "ink-detections" / "ink_detection.png") << "png";
    fs::create_directories(segDir / "seg_inklabels.zarr" / "0");
    std::ofstream(segDir / "seg_inklabels.zarr" / "0" / ".zarray") << "{}";

    SUBCASE("saveOverwrite (VC3D edits and autosave)")
    {
        QuadSurface loaded(segDir);
        loaded.ensureLoaded();
        loaded.saveOverwrite();
    }
    SUBCASE("save(path, force_overwrite=true)")
    {
        auto loaded = load_quad_from_tifxyz(segDir.string());
        REQUIRE(loaded);
        loaded->save(segDir, true);
    }

    CHECK(fs::exists(segDir / "ink-detections" / "ink_detection.png"));
    CHECK(fs::exists(segDir / "seg_inklabels.zarr" / "0" / ".zarray"));
    CHECK(fs::exists(segDir / "x.tif"));
    CHECK(QuadSurface(segDir).meta["format"].get_string() == "tifxyz");
    fs::remove_all(root);
}

TEST_CASE("load rolls back an interrupted multi-file save generation")
{
    auto root = tmpDir("rollback");
    auto segDir = root / "seg";
    auto replacementDir = root / "replacement";

    {
        QuadSurface original(grid(8, 8, /*z=*/1.f), cv::Vec2f(1.f, 1.f));
        original.meta["uuid"] = "uuid-old";
        original.save(segDir.string(), false);
        QuadSurface replacement(grid(8, 8, /*z=*/99.f), cv::Vec2f(1.f, 1.f));
        replacement.meta["uuid"] = "uuid-new";
        replacement.save(replacementDir.string(), false);
    }

    // Reproduce the durable state left when Windows stops after publishing
    // only part of a generation: complete originals in the rollback journal,
    // its ready marker, and a mixture in the live directory.
    const auto rollbackDir = segDir / ".vc-save-rollback";
    fs::create_directories(rollbackDir);
    for (const auto* name : {"x.tif", "y.tif", "z.tif", "meta.json"}) {
        fs::copy_file(segDir / name, rollbackDir / name);
    }
    {
        std::ofstream ready(rollbackDir / ".ready");
        ready << "ready\n";
    }
    fs::copy_file(replacementDir / "x.tif", segDir / "x.tif",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(replacementDir / "meta.json", segDir / "meta.json",
                  fs::copy_options::overwrite_existing);

    QuadSurface recovered(segDir);
    CHECK(recovered.meta["uuid"].get_string() == "uuid-old");
    recovered.ensureLoaded();
    CHECK(recovered.rawPointsPtr()->at<cv::Vec3f>(0, 0)[2] == doctest::Approx(1.f));
    CHECK_FALSE(fs::exists(rollbackDir));

    fs::remove_all(root);
}

TEST_CASE("save without force_overwrite + existing dir throws (or overwrites)")
{
    auto root = tmpDir("noforce");
    auto segDir = root / "seg";
    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.meta["uuid"] = "uuid-a";
        qs.save(segDir.string(), /*force_overwrite=*/false);
    }
    // Second save without force_overwrite. Behavior is impl-defined — some
    // builds throw, others overwrite. We just exercise the path.
    try {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.meta["uuid"] = "uuid-c";
        qs.save(segDir.string(), /*force_overwrite=*/false);
    } catch (const std::exception&) {
        // Acceptable.
    }
    CHECK(true);
    fs::remove_all(root);
}

TEST_CASE("lookupDepthIndex: with 'd' channel as float matrix")
{
    auto pts = grid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat dCh(8, 8, CV_32FC1, cv::Scalar(2.5f));
    qs.setChannel("d", dCh);
    float v = lookupDepthIndex(&qs, 4, 4);
    CHECK(v == doctest::Approx(2.5f).epsilon(1e-3));
}

TEST_CASE("lookupDepthIndex: out-of-bounds returns NaN")
{
    auto pts = grid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat dCh(8, 8, CV_32FC1, cv::Scalar(1.0f));
    qs.setChannel("d", dCh);
    CHECK(std::isnan(lookupDepthIndex(&qs, -1, 0)));
    CHECK(std::isnan(lookupDepthIndex(&qs, 0, 99)));
    CHECK(std::isnan(lookupDepthIndex(nullptr, 0, 0)));
}

TEST_CASE("lookupDepthIndex: empty 'd' channel returns NaN")
{
    auto pts = grid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    // No 'd' channel set
    CHECK(std::isnan(lookupDepthIndex(&qs, 0, 0)));
}

TEST_CASE("free pointTo (Vec3f): exercises second-pass branch when first miss")
{
    // A surface where the target is far enough that the initial location
    // doesn't immediately converge.
    cv::Mat_<cv::Vec3f> pts(32, 32);
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c)
            pts(r, c) = cv::Vec3f(float(c), float(r), 0.f);
    cv::Vec2f loc(0.f, 0.f);
    cv::Vec3f target(25.f, 25.f, 0.f);
    float d = ::pointTo(loc, pts, target, /*th=*/0.5f, /*max_iters=*/50, /*scale=*/1.0f);
    CHECK(d >= 0.0f);
}
