// The VC3D viewer samples each frame from the caches alone (vcr::grid) and reads
// what was missing in the background: frames never read the volume's store, the
// background read makes the next frame complete, and coarser levels fill holes
// meanwhile. Planes and QuadSurface grids are sampled the same way.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../../apps/VC3D/OpenedVolumes.hpp"
#include "../../apps/VC3D/volume_viewers/ViewerSampling.hpp"
#include "support/TestZarrVolume.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <vc_reader/zarr.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Fixture {
    fs::path dir = fs::temp_directory_path() /
                   ("vc-viewer-sampling-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::shared_ptr<Volume> volume;
    std::vector<std::uint8_t> level0 = std::vector<std::uint8_t>(32 * 32 * 32);
    std::vector<std::uint8_t> level1 = std::vector<std::uint8_t>(16 * 16 * 16);

    Fixture()
    {
        vc::test::ZarrVolumeSpec spec;
        spec.shapeZYX = {32, 32, 32};
        spec.chunkShapeZYX = {8, 8, 8};
        spec.numLevels = 2;
        volume = vc::test::createZarrVolume(dir, spec);
        for (std::size_t i = 0; i < level0.size(); ++i)
            level0[i] = std::uint8_t(1 + i % 250);
        for (std::size_t i = 0; i < level1.size(); ++i)
            level1[i] = std::uint8_t(1 + 7 * i % 250);
        vcr::zarr::Writer::open(dir / "0").writeRegion({0, 0, 0}, {32, 32, 32}, level0.data());
        vcr::zarr::Writer::open(dir / "1").writeRegion({0, 0, 0}, {16, 16, 16}, level1.data());
    }
    ~Fixture() { fs::remove_all(dir); }

    std::uint8_t at0(int z, int y, int x) const { return level0[std::size_t((z * 32 + y) * 32 + x)]; }
};

// The z plane through level-0 voxel centres, one pixel per voxel.
vc3d::ViewerGeometry planeAt(float z, int width = 32, int height = 32)
{
    vc3d::ViewerGeometry g;
    g.plane = vcr::grid::Plane{{0, 0, z}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, width, height};
    return g;
}

std::shared_ptr<const vc3d::ViewerLevels> openLevels(OpenedVolumes& opened, const Volume& volume)
{
    return std::make_shared<const vc3d::ViewerLevels>(vc3d::openViewerLevels(opened, volume));
}

}  // namespace

TEST_CASE("viewer levels and level choice")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = vc3d::openViewerLevels(opened, *f.volume);
    REQUIRE(levels.count() == 2);
    CHECK(levels.has(0));
    CHECK(levels.has(1));
    CHECK(levels.voxelExtent(1)[0] == doctest::Approx(2.0));
    CHECK(vc3d::maximumBaseVoxelExtent(levels, 1) == doctest::Approx(2.0));
    // One screen pixel per voxel wants level 0; a quarter pixel per voxel takes level 1.
    CHECK(vc3d::sourceLevelForView(levels, 1.0f, 1.0f) == 0);
    CHECK(vc3d::sourceLevelForView(levels, 0.25f, 1.0f) == 1);
    CHECK(vc3d::sourceLevelForView(levels, 8.0f, 1.0f) == 0);
    // 8-voxel chunks of level 0 do not span a 32-pixel viewport at 1 px/voxel; level 1's 16 do not either.
    CHECK(vc3d::fallbackLevelCountForViewport(levels, 0, 32, 32, 1.0f) == 1);
    // At 4 px/voxel the viewport spans 8 voxels: a level-0 chunk covers it.
    CHECK(vc3d::fallbackLevelCountForViewport(levels, 0, 32, 32, 4.0f) == 0);
    // Level-1 indices of level-0 voxel 4.5: its centre, 2.
    const auto g = levels.gridLevel(1);
    CHECK(g.scale[0] * 4.5 + g.offset[0] == doctest::Approx(2.0));
}

TEST_CASE("frames sample the caches only; the background read completes them")
{
    Fixture f;
    for (const auto sampling : {vc::Sampling::Nearest, vc::Sampling::Trilinear}) {
        CAPTURE(int(sampling));
        OpenedVolumes opened(64 << 20);
        const auto levels = openLevels(opened, *f.volume);
        const vc3d::ViewerSample what{levels, 0, 1, sampling, {0.0f}, std::nullopt};
        vc3d::ViewerLoad load;
        auto r = vc3d::sampleCached(planeAt(5), what, &load);
        // Nothing is in memory: nothing is covered and the store was not read; the
        // frame asks for its start level and, for a preview, its coarsest one.
        for (const auto level : r.level)
            REQUIRE(level == vcr::grid::Result::kMissing);
        CHECK(levels->levels[0].array->array->stats().disk_hits == 0);
        REQUIRE(load.target.size() == 1);
        REQUIRE(load.coarse.size() == 1);

        vc3d::loadPending(load, [] { return true; });
        vc3d::ViewerLoad again;
        r = vc3d::sampleCached(planeAt(5), what, &again);
        CHECK(again.empty());
        cv::Mat_<uint8_t> values(32, 32, uint8_t(0)), coverage(32, 32, uint8_t(0));
        vc3d::toDisplay(r, 0, values.ptr(), coverage.ptr());
        CHECK(cv::countNonZero(coverage) == 32 * 32);
        // Voxel centres: nearest and linear both give the stored values.
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 32; ++x)
                REQUIRE(values(y, x) == f.at0(5, y, x));
    }
}

TEST_CASE("a coarser level in memory fills a frame while the start level loads")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    const vc3d::ViewerSample what{levels, 0, 1, vc::Sampling::Nearest, {0.0f}, std::nullopt};
    {
        vc3d::ViewerLoad load;
        vc3d::sampleCached(planeAt(4), what, &load);
        load.target.clear();  // only the coarse preview arrives
        vc3d::loadPending(load, [] { return true; });
    }
    vc3d::ViewerLoad load;
    const auto r = vc3d::sampleCached(planeAt(4), what, &load);
    for (const auto level : r.level)
        REQUIRE(level == 1);
    CHECK(load.coarse.empty());
    REQUIRE(load.target.size() == 1);
    // Level-0 voxel (4, 6, 10) lies in level-1 voxel (2, 3, 5).
    CHECK(r.values[6 * 32 + 10] == float(f.level1[(2 * 16 + 3) * 16 + 5]));
}

TEST_CASE("pixels outside the volume have no sample and are not read")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    vc3d::ViewerGeometry g;
    g.plane = vcr::grid::Plane{{28, 3, 3}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 8, 1};  // x 28 .. 35
    const vc3d::ViewerSample what{levels, 0, 0, vc::Sampling::Nearest, {0.0f}, std::nullopt};
    vc3d::ViewerLoad load;
    vc3d::sampleCached(g, what, &load);
    REQUIRE(load.target.size() == 1);
    CHECK(load.target[0].pixels == std::vector<std::uint8_t>{1, 1, 1, 1, 0, 0, 0, 0});
    vc3d::loadPending(load, [] { return true; });
    const auto r = vc3d::sampleCached(g, what, nullptr);
    CHECK(r.level[3] == 0);
    CHECK(r.level[4] == vcr::grid::Result::kNone);
    CHECK(r.values[0] == float(f.at0(3, 3, 28)));
}

TEST_CASE("a QuadSurface grid samples like the plane it lies on, composites reduce its layers")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    // A flat surface at z = 6 with a vertex every 4 voxels: grid scale 0.25.
    cv::Mat_<cv::Vec3f> points(9, 9);
    for (int r = 0; r < 9; ++r)
        for (int c = 0; c < 9; ++c)
            points(r, c) = cv::Vec3f(4.0f * c, 4.0f * r, 6.0f);
    QuadSurface quad(points, cv::Vec2f(0.25f, 0.25f));
    vc3d::ViewerGeometry g;
    g.surface = vc3d::gridSurface(quad);
    REQUIRE(g.surface);
    g.raster = {0.25, 0.25, 0.25, 0.25, 30, 30};  // voxels 1 .. 30 in x and y
    vc3d::ViewerSample what{levels, 0, 0, vc::Sampling::Nearest, {0.0f}, std::nullopt};
    vc3d::ViewerLoad load;
    vc3d::sampleCached(g, what, &load);
    vc3d::loadPending(load, [] { return true; });
    auto r = vc3d::sampleCached(g, what, nullptr);
    CHECK(r.values[0] == float(f.at0(6, 1, 1)));
    CHECK(r.values[29 * 30 + 29] == float(f.at0(6, 30, 30)));

    // Max of the layers z = 5, 6, 7 (the surface normal is +-z).
    CompositeParams params;
    params.method = "max";
    what.offsets = {-1.0f, 0.0f, 1.0f};
    what.composite = vc3d::compositeOptions(params);
    REQUIRE(what.composite);
    vc3d::ViewerLoad more;
    vc3d::sampleCached(g, what, &more);
    vc3d::loadPending(more, [] { return true; });
    r = vc3d::sampleCached(g, what, nullptr);
    REQUIRE(r.values.size() == 30 * 30);
    const float expected = float(std::max({f.at0(5, 10, 10), f.at0(6, 10, 10), f.at0(7, 10, 10)}));
    CHECK(r.values[9 * 30 + 9] == expected);
    params.method = "volumetric";
    CHECK_FALSE(vc3d::compositeOptions(params));
}
