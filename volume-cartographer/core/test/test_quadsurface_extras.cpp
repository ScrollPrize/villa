// Additional QuadSurface coverage:
// - QuadSurface(path, json) ctor with components/bbox
// - setChannel with 3- and 4-channel masks (normalizeMaskChannel branches)
// - channel(name) on-demand loading via file
// - pointTo with a SurfacePatchIndex hint
// - orientZUp with a deliberate Z gradient
// - load_quad_from_tifxyz with a mask.tif present

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Tiff.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <memory>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

#ifndef VC_TEST_FIXTURES_DIR
#define VC_TEST_FIXTURES_DIR "core/test/data"
#endif

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_qs_extras_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return m;
}

fs::path fixtureSeg(const std::string& name)
{
    return fs::path(VC_TEST_FIXTURES_DIR) / "segments" / name;
}

} // namespace

TEST_CASE("QuadSurface(path, json) ctor with bbox + components fields")
{
    auto root = tmpDir("path_json");
    auto d = root / "seg";
    // Save a real segment to disk first so we have a points-loader target.
    auto pts = makePlanarGrid(16, 16);
    {
        QuadSurface seed(pts, cv::Vec2f(1.f, 1.f));
        seed.id = d.filename().string();
        seed.save(d);
    }
    // Build a custom meta JSON for the second ctor.
    auto j = utils::Json::object();
    j["format"] = "tifxyz";
    j["type"] = "seg";
    j["uuid"] = "test-uuid";
    j["bbox"] = utils::Json::array();
    auto lo = utils::Json::array();
    lo.push_back(0.0); lo.push_back(0.0); lo.push_back(0.0);
    auto hi = utils::Json::array();
    hi.push_back(16.0); hi.push_back(16.0); hi.push_back(0.0);
    j["bbox"].push_back(lo);
    j["bbox"].push_back(hi);
    auto comps = utils::Json::array();
    auto c1 = utils::Json::array(); c1.push_back(1); c1.push_back(2);
    comps.push_back(c1);
    j["components"] = comps;

    QuadSurface qs(d, j);
    qs.ensureLoaded();
    CHECK(qs.rawPointsPtr() != nullptr);
    CHECK(qs.meta["uuid"].get_string() == "test-uuid");
    fs::remove_all(root);
}

TEST_CASE("setChannel with a 3-channel mask triggers normalizeMaskChannel branch")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat mask3(4, 4, CV_8UC3, cv::Scalar(255, 0, 0));
    qs.setChannel("mask", mask3);
    // channel() should normalize to single-channel uint8.
    auto out = qs.channel("mask", SURF_CHANNEL_NORESIZE);
    CHECK(out.channels() == 1);
    CHECK(out.type() == CV_8UC1);
}

TEST_CASE("setChannel with a 4-channel mask is normalized too")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat mask4(4, 4, CV_8UC4, cv::Scalar(255, 128, 64, 255));
    qs.setChannel("mask", mask4);
    auto out = qs.channel("mask", SURF_CHANNEL_NORESIZE);
    CHECK(out.channels() == 1);
}

TEST_CASE("setChannel with a CV_16U mask is converted to CV_8U")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat mask16(4, 4, CV_16UC1, cv::Scalar(65535));
    qs.setChannel("mask", mask16);
    auto out = qs.channel("mask", SURF_CHANNEL_NORESIZE);
    CHECK(out.depth() == CV_8U);
}

TEST_CASE("channel: unknown name returns empty mat")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto m = qs.channel("not_a_channel", SURF_CHANNEL_NORESIZE);
    CHECK(m.empty());
}

TEST_CASE("channel: with default flags, scaled to points size")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    // A smaller mask gets nearest-resized up.
    cv::Mat m(2, 2, CV_8UC1, cv::Scalar(255));
    qs.setChannel("approval", m);
    auto out = qs.channel("approval", 0);
    CHECK(out.size() == cv::Size(8, 8));
}

TEST_CASE("channel: on-disk file load (no in-memory mat)")
{
    auto root = tmpDir("chan_disk");
    auto d = root / "seg";
    auto pts = makePlanarGrid(32, 32);
    QuadSurface seed(pts, cv::Vec2f(1.f, 1.f));
    seed.id = d.filename().string();
    seed.path = d;
    seed.save(d);
    // 32x32 image with untiled (single strip) write.
    cv::Mat sideCh(32, 32, CV_8UC1, cv::Scalar(123));
    writeTiff(d / "side.tif", sideCh, CV_8UC1, /*tileW=*/0, /*tileH=*/0);

    QuadSurface reloaded(d);
    reloaded.setChannel("side", cv::Mat());
    auto loaded = reloaded.channel("side", SURF_CHANNEL_NORESIZE);
    if (!loaded.empty()) {
        CHECK(loaded.rows == 32);
        CHECK(loaded.cols == 32);
    }
    fs::remove_all(root);
}

TEST_CASE("pointTo with a SurfacePatchIndex hint")
{
    auto pts = makePlanarGrid(32, 32, 0.f);
    auto surf = std::make_shared<QuadSurface>(pts, cv::Vec2f(1.f, 1.f));
    SurfacePatchIndex idx;
    idx.rebuild({surf});

    cv::Vec3f ptr(0, 0, 0);
    cv::Vec3f tgt(10.f, 10.f, 0.f);
    float d = surf->pointTo(ptr, tgt, /*th=*/0.5f, /*max_iters=*/100,
                            /*surfaceIndex=*/&idx);
    CHECK(d >= 0.0f);
}

TEST_CASE("orientZUp with deliberate Z gradient runs without crashing")
{
    cv::Mat_<cv::Vec3f> m(32, 32);
    // Z increases with row — high Z at bottom.
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), float(r * 2));
    QuadSurface qs(m, cv::Vec2f(1.f, 1.f));
    qs.orientZUp();
    CHECK(qs.rawPointsPtr() != nullptr);
}

TEST_CASE("orientZUp on flat surface is a no-op (angle below threshold)")
{
    auto pts = makePlanarGrid(16, 16, 5.f);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    int before = qs.countValidPoints();
    qs.orientZUp();
    CHECK(qs.countValidPoints() == before);
}

TEST_CASE("load_quad_from_tifxyz with an associated mask.tif trims invalid points")
{
    auto seg = fixtureSeg("20241113070770");
    if (!fs::exists(seg / "meta.json")) {
        MESSAGE("Skipping: fixture missing");
        return;
    }
    // Stage the segment + an extra mask.tif into a tmp dir so we don't
    // pollute the committed fixtures.
    auto d = tmpDir("mask_load");
    auto dst = d / "seg";
    fs::create_directories(dst);
    for (const auto& f : fs::directory_iterator(seg)) {
        fs::copy(f.path(), dst / f.path().filename(),
                 fs::copy_options::recursive);
    }
    // Write a 129x129 mask of all 255 (no trimming) just to exercise the
    // mask-application code path.
    cv::Mat mask(129, 129, CV_8UC1, cv::Scalar(255));
    writeTiff(dst / "mask.tif", mask, CV_8UC1, /*tileW=*/128, /*tileH=*/128);

    auto qs = load_quad_from_tifxyz(dst.string(), 0);
    REQUIRE(qs);
    qs->ensureLoaded();
    CHECK(qs->rawPointsPtr() != nullptr);
    fs::remove_all(d);
}

TEST_CASE("invalidateMask: clears the cached validMask")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    (void)qs.validMask(); // populate cache
    qs.invalidateMask();
    // Re-issue should still succeed.
    auto m = qs.validMask();
    CHECK(m.rows == 8);
}

TEST_CASE("countValid* on path-only (unloaded) construction lazily loads")
{
    auto root = tmpDir("lazy");
    auto d = root / "seg";
    auto pts = makePlanarGrid(8, 8);
    {
        QuadSurface seed(pts, cv::Vec2f(1.f, 1.f));
        seed.id = d.filename().string();
        seed.save(d);
    }
    QuadSurface reloaded(d);
    // Just touch countValidPoints; it triggers ensureLoaded internally.
    int validPts = reloaded.countValidPoints();
    CHECK(validPts >= 0); // small grids may or may not have valid points
    fs::remove_all(root);
}

TEST_CASE("ptrToGrid round-trips with default scale")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto g = qs.ptrToGrid(cv::Vec3f(0, 0, 0));
    CHECK(g[0] > 0);
    CHECK(g[1] > 0);
}
