#include <vc_reader/display.hpp>
// The VC3D viewer samples each frame from the caches alone (vcr::grid) and reads
// what was missing in the background: frames never read the volume's store, the
// background read makes the next frame complete, and coarser levels fill holes
// meanwhile. Planes and QuadSurface grids are sampled the same way.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../../apps/VC3D/OpenedVolumes.hpp"
#include "../../apps/VC3D/volume_viewers/ViewerSampling.hpp"
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <filesystem>
#include <limits>
#include <fstream>
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <vc_reader/array.hpp>

#include <opencv2/core.hpp>

#include <chrono>
#include <vc_reader/native/cuda_available.hpp>
#include "vc/core/util/CudaSampling.hpp"

#include <cmath>
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
        const std::filesystem::path volumeRoot = dir;
        const vcr::zarr::CreateOptions levelZero{.shape = {32, 32, 32}, .chunks = {8, 8, 8}, .compressor = "", .overwrite = true};
        vcr::zarr::Writer::create(volumeRoot / "0", levelZero);
        vcr::zarr::createPyramidLevels(volumeRoot, levelZero, {.levels = 1});
        vcr::zarr::createGroup(volumeRoot, nlohmann::json{{"multiscales", nlohmann::json::array({vcr::zarr::omeMultiscales({.levels = 1})})}});
        volume = Volume::New(volumeRoot);
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

TEST_CASE("flattened LOD follows the actual voxel spacing of the surface")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    // UV units span only a quarter voxel: nominal zoom alone chooses L1,
    // although the actual viewport requires L0.
    cv::Mat_<cv::Vec3f> points(9, 9);
    for (int r = 0; r < 9; ++r)
        for (int c = 0; c < 9; ++c) points(r, c) = {float(c), float(r), 6};
    QuadSurface quad(points, {0.25f, 0.25f});
    const auto raster = quad.gridRaster(-quad.center()[0], -quad.center()[1], 0.5, {12, 12});
    const auto zoom = vc3d::surfacePixelsPerVoxel(quad, raster, 0.5f);
    CHECK(zoom == doctest::Approx(2));
    CHECK(vc3d::sourceLevelForView(*levels, 0.5f, 1) == 1);
    CHECK(vc3d::sourceLevelForView(*levels, zoom, 1) == 0);

    vc3d::ViewerGeometry geometry;
    geometry.surface = vc3d::gridSurface(quad);
    geometry.raster = raster;
    vc3d::ViewerLoad preview;
    vc3d::sampleCached(geometry, {levels, 1, 0, 0}, &preview);
    vc3d::loadPending(preview, [] { return true; });
    vc3d::ViewerLoad detail;
    vc3d::sampleCached(geometry, {levels, 0, 1, 0}, &detail);
    REQUIRE(detail.target.size() == 1);
    vc3d::loadPending(detail, [] { return true; });
    const auto fine = vc3d::sampleCached(geometry, {levels, 0, 1, 0}, nullptr);
    CHECK(fine.level[0] == 0);
    CHECK(fine.values[0] == float(f.at0(6, 0, 0)));
}

TEST_CASE("frames sample the caches only; the background read completes them")
{
    Fixture f;
    for (const auto sampling : {0, 1}) {
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
        vcr::display::toUint8(r.values.data(), std::size_t(r.width) * r.height, values.ptr(), coverage.ptr());
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
    const vc3d::ViewerSample what{levels, 0, 1, 0, {0.0f}, std::nullopt};
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

TEST_CASE("pixels outside the volume read as fill and are not read in the background")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    vc3d::ViewerGeometry g;
    g.plane = vcr::grid::Plane{{28, 3, 3}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 8, 1};  // x 28 .. 35
    const vc3d::ViewerSample what{levels, 0, 0, 0, {0.0f}, std::nullopt};
    vc3d::ViewerLoad load;
    vc3d::sampleCached(g, what, &load);
    REQUIRE(load.target.size() == 1);
    CHECK(load.target[0].pixels == std::vector<std::uint8_t>{1, 1, 1, 1, 0, 0, 0, 0});
    vc3d::loadPending(load, [] { return true; });
    const auto r = vc3d::sampleCached(g, what, nullptr);
    CHECK(r.level[3] == 0);
    CHECK(r.level[4] == 0);
    CHECK(r.values[4] == 0.0f);
    CHECK(r.values[0] == float(f.at0(3, 3, 28)));
}

TEST_CASE("a GPU frame that fails renders on the CPU")
{
    if (!vcr::cuda::available()) {
        MESSAGE("Skipped: no CUDA device");
        return;
    }
    Fixture f;
    OpenedVolumes cpu(64 << 20);
    OpenedVolumes tooSmall(64 << 20, 1);  // no brick fits this device budget
    const auto expected = openLevels(cpu, *f.volume);
    const auto levels = openLevels(tooSmall, *f.volume);
    REQUIRE(levels->levels[0].cuda);
    vc3d::ViewerGeometry g;
    g.plane = vcr::grid::Plane{{0, 0, 6}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 16, 16};
    const auto rendered = [&](const std::shared_ptr<const vc3d::ViewerLevels>& from) {
        vc3d::ViewerSample what{from, 0, 0, 0, {0.0f}, std::nullopt};
        vc3d::ViewerLoad load;
        vc3d::sampleCached(g, what, &load);
        vc3d::loadPending(load, [] { return true; });
        return vc3d::sampleCached(g, what, nullptr).values;
    };
    const auto values = rendered(levels);
    CHECK(!values.empty());
    CHECK(values == rendered(expected));
}

TEST_CASE("viewer and QuadSurface default to bicubic positions and patch normals")
{
    cv::Mat_<cv::Vec3f> points(5, 5);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
            points(r, c) = cv::Vec3f(float(c), float(r), float(c * c));
    QuadSurface quad(points, cv::Vec2f(1, 1));
    const auto viewer = vc3d::gridSurface(quad);
    REQUIRE(viewer);
    for (bool useViewer : {true, false}) {
        float p[3], n[3];
        if (useViewer) {
            viewer->resample({1.25, 1.5, 1, 1, 1, 1}, p, n);
        } else {
            const auto uv = quad.gridToSurface({1.25, 1.5});
            const cv::Vec3f ptr(float(uv[0]), float(uv[1]), 0);
            const auto position = quad.coord(ptr);
            const auto normal = quad.normal(ptr);
            for (int a = 0; a < 3; ++a) {
                p[a] = position[a];
                n[a] = normal[a];
            }
        }
        // The interior bicubic patch reproduces z = x^2, with dz/dx = 2x.
        // Bilinear interpolation would give z = 1.75 and dz/dx = 3.
        CHECK(p[0] == doctest::Approx(1.25f));
        CHECK(p[1] == doctest::Approx(1.5f));
        CHECK(p[2] == doctest::Approx(1.5625f));
        CHECK(n[0] == doctest::Approx(-2.5f / std::sqrt(7.25f)));
        CHECK(n[1] == doctest::Approx(0.0f));
        CHECK(n[2] == doctest::Approx(1.0f / std::sqrt(7.25f)));
    }
}

TEST_CASE("vertex spacing is the working level's voxel shape")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    auto levels = *openLevels(opened, *f.volume);
    REQUIRE(levels.count() > 0);
    CHECK(levels.vertexSpacing() == std::array<float, 3>{1, 1, 1});
    levels.levels[0].working.scale = {4, 4, 4};
    CHECK(levels.vertexSpacing() == std::array<float, 3>{1, 1, 1});
    levels.levels[0].working.scale = {1, 2, 4};  // ZYX
    CHECK(levels.vertexSpacing() == std::array<float, 3>{4, 2, 1});
}

TEST_CASE("a frame on an anisotropic level samples where the level's placement says")
{
    // Level 1 halves x and y only: z stays at level 0's spacing.
    Fixture f;
    vcr::zarr::removeNode(f.dir / "1");
    vcr::zarr::CreateOptions options;
    options.shape = {32, 16, 16};
    options.chunks = {8, 8, 8};
    auto coarse = vcr::zarr::Writer::create(f.dir / "1", options);
    const auto ramp = [](double z, double y, double x) { return 10.0 + 2.0 * z + 1.0 * y + 3.0 * x; };
    std::vector<std::uint8_t> values0(32 * 32 * 32), values1(32 * 16 * 16);
    for (int z = 0; z < 32; ++z)
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 32; ++x)
                values0[std::size_t((z * 32 + y) * 32 + x)] = std::uint8_t(ramp(z, y, x));
    for (int z = 0; z < 32; ++z)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)  // the block's centre, at (2x + 0.5, 2y + 0.5) of level 0
                values1[std::size_t((z * 16 + y) * 16 + x)] = std::uint8_t(ramp(z, 2 * y + 0.5, 2 * x + 0.5));
    vcr::zarr::Writer::open(f.dir / "0").writeRegion({0, 0, 0}, {32, 32, 32}, values0.data());
    coarse.writeRegion({0, 0, 0}, {32, 16, 16}, values1.data());
    using vcr::zarr::Json;
    const auto level = [](const char* path, Json scale, Json translation) {
        return Json{{"path", path},
                    {"coordinateTransformations", Json::array({Json{{"type", "scale"}, {"scale", scale}},
                                                               Json{{"type", "translation"}, {"translation", translation}}})}};
    };
    vcr::zarr::writeAttributes(f.dir, {{"multiscales", Json::array({Json{
        {"version", "0.4"},
        {"axes", Json::array({Json{{"name", "z"}, {"type", "space"}}, Json{{"name", "y"}, {"type", "space"}}, Json{{"name", "x"}, {"type", "space"}}})},
        {"datasets", Json::array({level("0", {1, 1, 1}, {0, 0, 0}), level("1", {1, 2, 2}, {0, 0.5, 0.5})})}}})}});

    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    REQUIRE(levels->has(1));
    CHECK(levels->voxelExtent(1) == std::array<double, 3>{1, 2, 2});
    CHECK(vc3d::maximumBaseVoxelExtent(*levels, 1) == doctest::Approx(2.0));
    const auto frame = [&](int start, int fallback) {
        const vc3d::ViewerSample what{levels, start, fallback, 1, {0.0f}, std::nullopt};
        vc3d::ViewerLoad load;
        (void)vc3d::sampleCached(planeAt(5, 28, 28), what, &load);
        vc3d::loadPending(load, [] { return true; });
        return vc3d::sampleCached(planeAt(5, 28, 28), what, nullptr);
    };
    const auto fine = frame(0, 0);
    const auto wide = frame(1, 0);
    REQUIRE(wide.values.size() == std::size_t(28 * 28));
    for (int y = 1; y < 28; ++y)
        for (int x = 1; x < 28; ++x) {
            const auto q = std::size_t(y) * 28 + std::size_t(x);
            // A linear ramp: level 1 gives what level 0 does, within the stored values' rounding.
            REQUIRE(std::abs(wide.values[q] - fine.values[q]) <= 1.0f);
            REQUIRE(fine.values[q] == doctest::Approx(float(ramp(5, y, x))).epsilon(0.02));
        }
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
    vc3d::ViewerSample what{levels, 0, 0, 0, {0.0f}, std::nullopt};
    vc3d::ViewerLoad load;
    vc3d::sampleCached(g, what, &load);
    vc3d::loadPending(load, [] { return true; });
    auto r = vc3d::sampleCached(g, what, nullptr);
    CHECK(r.values[0] == float(f.at0(6, 1, 1)));
    CHECK(r.values[29 * 30 + 29] == float(f.at0(6, 30, 30)));

    // Max of the layers z = 5, 6, 7 (the surface normal is +-z).
    CompositeParams params;
    REQUIRE(setCompositeMethod(params, "max"));
    what.offsets = {-1.0f, 0.0f, 1.0f};
    what.composite = (params.volumetric ? std::nullopt : std::optional(params.options));
    REQUIRE(what.composite);
    vc3d::ViewerLoad more;
    vc3d::sampleCached(g, what, &more);
    vc3d::loadPending(more, [] { return true; });
    r = vc3d::sampleCached(g, what, nullptr);
    REQUIRE(r.values.size() == 30 * 30);
    const float expected = float(std::max({f.at0(5, 10, 10), f.at0(6, 10, 10), f.at0(7, 10, 10)}));
    CHECK(r.values[9 * 30 + 9] == expected);
    REQUIRE(setCompositeMethod(params, "volumetric"));
    CHECK_FALSE((params.volumetric ? std::nullopt : std::optional(params.options)));
    CHECK_FALSE(setCompositeMethod(params, "nothing"));
    CHECK(compositeMethodName(params) == "volumetric");
}

TEST_CASE("a zoomed-out frame takes resident finer data and does not read its own level")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    // Level 0 in memory (as after zooming in); the frame wants level 1.
    {
        const vc3d::ViewerSample fine{levels, 0, 0, 0, {0.0f}, std::nullopt};
        vc3d::ViewerLoad load;
        vc3d::sampleCached(planeAt(5), fine, &load);
        vc3d::loadPending(load, [] { return true; });
    }
    const vc3d::ViewerSample coarse{levels, 1, 0, 0, {0.0f}, std::nullopt};
    vc3d::ViewerLoad load;
    const auto r = vc3d::sampleCached(planeAt(5), coarse, &load);
    CHECK(load.empty());
    CHECK(r.level[0] == 1);  // the second level of the request: level 0
    CHECK(r.values[7 * 32 + 9] == float(f.at0(5, 7, 9)));
}

#ifdef VC_READER_CUDA
TEST_CASE("GPU frames match CPU frames: resident bricks only, background reads load them")
{
    if (!vcr::cuda::available()) {
        MESSAGE("Skipped: no CUDA device");
        return;
    }
    Fixture f;
    for (const auto sampling : {0, 1}) {
        CAPTURE(int(sampling));
        OpenedVolumes cpu(64 << 20), gpu(64 << 20, 64 << 20);
        const auto cpuLevels = openLevels(cpu, *f.volume);
        const auto gpuLevels = openLevels(gpu, *f.volume);
        REQUIRE(gpuLevels->levels[0].cuda);
        const auto geometry = planeAt(5.25f);
        const vc3d::ViewerSample cpuWhat{cpuLevels, 0, 1, sampling, {0.0f, 1.0f}, std::nullopt};
        const vc3d::ViewerSample gpuWhat{gpuLevels, 0, 1, sampling, {0.0f, 1.0f}, std::nullopt};

        vc3d::ViewerLoad load;
        for (const auto level : vc3d::sampleCached(geometry, gpuWhat, &load).level)
            REQUIRE(level == vcr::grid::Result::kMissing);
        REQUIRE(load.target.size() == 1);
        vc3d::loadPending(load, [] { return true; });

        vc3d::ViewerLoad again, cpuLoad, cpuAgain;
        const auto onGpu = vc3d::sampleCached(geometry, gpuWhat, &again);
        CHECK(again.empty());
        vc3d::sampleCached(geometry, cpuWhat, &cpuLoad);
        vc3d::loadPending(cpuLoad, [] { return true; });
        const auto onCpu = vc3d::sampleCached(geometry, cpuWhat, &cpuAgain);
        REQUIRE(onGpu.values.size() == onCpu.values.size());
        CHECK(onGpu.level == onCpu.level);
        float worst = 0.0f;
        for (std::size_t i = 0; i < onCpu.values.size(); ++i)
            worst = std::max(worst, std::abs(onGpu.values[i] - onCpu.values[i]));
        CHECK(worst <= 1e-3f);
    }
}

TEST_CASE("vc::cuda point sampling matches the CPU, and tools get a reader only with --cuda")
{
    if (!vcr::cuda::available()) {
        MESSAGE("Skipped: no CUDA device");
        return;
    }
    Fixture f;
    OpenedVolumes cpu(64 << 20), gpu(64 << 20, 64 << 20);
    const auto level = gpu.level(*f.volume, 0);
    REQUIRE(level.cuda);
    CHECK_FALSE(cpu.level(*f.volume, 0).cuda);
    auto& array = *level.array->array;
    const std::vector<double> zyx{5.25, 6.5, 7.75, 0.0, 0.0, 0.0, 31.0, 31.0, 31.0, 40.0, 1.0, 1.0};
    const std::size_t n = zyx.size() / 3;
    std::vector<float> linearCpu(n), linearGpu(n);
    vc::cuda::sampleLinear(array, level.array->meta.type, nullptr, zyx.data(), n, linearCpu.data());
    vc::cuda::sampleLinear(array, level.array->meta.type, level.cuda.get(), zyx.data(), n, linearGpu.data());
    std::vector<std::uint8_t> nearestCpu(n), nearestGpu(n);
    vc::cuda::sampleNearest(array, nullptr, zyx.data(), n, nearestCpu.data());
    vc::cuda::sampleNearest(array, level.cuda.get(), zyx.data(), n, nearestGpu.data());
    CHECK(nearestCpu == nearestGpu);
    for (std::size_t i = 0; i < n; ++i)
        CHECK(std::abs(linearCpu[i] - linearGpu[i]) <= 1e-3f);
    const std::vector<std::int64_t> voxels{3, 4, 5, 31, 31, 31};
    std::vector<std::uint8_t> read(2);
    vc::cuda::readVoxels(*level.cuda, voxels.data(), 2, read.data());
    CHECK(read[0] == f.at0(3, 4, 5));
    CHECK(read[1] == f.at0(31, 31, 31));

    // A request larger than the budget still gets its values, on the CPU.
    const auto tiny = vc::cuda::makeReader(array, level.array->meta.type, 1);
    REQUIRE(tiny);
    std::vector<float> fallen(n);
    vc::cuda::sampleLinear(array, level.array->meta.type, tiny.get(), zyx.data(), n, fallen.data());
    CHECK(fallen == linearCpu);

    // Off by default; --cuda-cache-gb implies --cuda and is taken out of argv.
    CHECK(vc::cuda::toolReader(*level.array) == nullptr);
    std::string program = "tool", first = "input", flag = "--cuda-cache-gb", gb = "1", last = "--other";
    char* args[] = {program.data(), first.data(), flag.data(), gb.data(), last.data(), nullptr};
    int argc = 5;
    vc::cuda::applyToolOptions(argc, args);
    REQUIRE(argc == 3);
    CHECK(std::string(args[1]) == "input");
    CHECK(std::string(args[2]) == "--other");
    CHECK(args[3] == nullptr);
    CHECK(vc::cuda::enabledForTool());
    auto* reader = vc::cuda::toolReader(*level.array);
    REQUIRE(reader);
    CHECK(vc::cuda::toolReaderOf(array) == reader);
    // Sampled through its own handle: it outlives the caller's array.
    std::vector<float> viaTool(n);
    vc::cuda::sampleLinear(array, level.array->meta.type, reader, zyx.data(), n, viaTool.data());
    for (std::size_t i = 0; i < n; ++i)
        CHECK(std::abs(linearCpu[i] - viaTool[i]) <= 1e-3f);
}
#endif

TEST_CASE("the background read stops when told to, and volumetric views have no reduction")
{
    Fixture f;
    OpenedVolumes opened(64 << 20);
    const auto levels = openLevels(opened, *f.volume);
    const vc3d::ViewerSample what{levels, 0, 1, 0, {0.0f}, std::nullopt};
    vc3d::ViewerLoad load;
    vc3d::sampleCached(planeAt(5), what, &load);
    REQUIRE(load.coarse.size() == 1);
    REQUIRE(load.target.size() == 1);

    int parts = 0;
    vc3d::loadPending(load, [&] { ++parts; return false; });
    CHECK(parts == 1);
    vc3d::ViewerLoad remaining;
    vc3d::sampleCached(planeAt(5), what, &remaining);
    CHECK(remaining.target.size() == 1);  // only the coarse part was read
    CHECK(remaining.coarse.empty());

    CompositeParams params;
    REQUIRE(setCompositeMethod(params, "max"));
    CHECK((params.volumetric ? std::nullopt : std::optional(params.options)));
    params.volumetric = true;
    CHECK_FALSE((params.volumetric ? std::nullopt : std::optional(params.options)));
}

TEST_CASE("toDisplay reads a composite's single value per pixel whichever layer is asked for")
{
    vcr::grid::Result composite;
    composite.width = 2;
    composite.height = 1;
    composite.values = {10.0f, std::numeric_limits<float>::quiet_NaN()};
    uint8_t values[2] = {}, coverage[2] = {};
    vcr::display::toUint8(composite.values.data(), std::size_t(composite.width) * composite.height, values, coverage);
    CHECK(values[0] == 10);
    CHECK(coverage[0] != 0);
    CHECK(coverage[1] == 0);
}

TEST_CASE("a cursor's offset from a surface is the frame's layer offset, in physical units")
{
    cv::Mat_<cv::Vec3f> points(4, 4);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) points(r, c) = cv::Vec3f(float(c), float(r), 8.f);
    QuadSurface quad(points, {1.f, 1.f});
    const auto surface = vc3d::gridSurface(quad, {1.f, 1.f, 2.f});  // z voxels are twice as long
    float xy[]{1.5f,1.5f};
    cv::Vec3f position, normal;
    surface->pointsAt(xy,1,position.val,normal.val);
    CHECK(position == cv::Vec3f(1.5f,1.5f,8.f));
    const auto layer=position+normal*4.f;
    CHECK(std::abs(layer[2]-8.f)==doctest::Approx(2.f));
    CHECK((layer-position).dot(normal)/normal.dot(normal)==doctest::Approx(4.f));
    xy[0]=10; surface->pointsAt(xy,1,position.val,normal.val);
    CHECK_FALSE(std::isfinite(position[0]));
}

TEST_CASE("grid windows and point queries match the whole grid without rebuilding it")
{
    constexpr int rows = 40, cols = 60;
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            points(y, x) = {float(x) + std::sin(y * 0.3f), float(y) + std::cos(x * 0.2f), 4 * std::sin(x * 0.11f + y * 0.17f)};
    points(10, 20) = {-1, -1, -1};
    points(25, 41) = {-1, -1, -1};
    QuadSurface quad(points, {1, 1});
    quad.setComponents({{0, 30}, {30, 60}});
    const auto whole = quad.pointSurface();

    const auto same = [](const float* a, const float* b) {
        for (int i = 0; i < 3; ++i)
            if (std::isnan(a[i]) != std::isnan(b[i]) || (!std::isnan(a[i]) && std::abs(a[i] - b[i]) > 1e-4f))
                return false;
        return true;
    };
    bool windowsMatch = true, pointsMatch = true;
    const cv::Rect window(12, 6, 30, 26);
    const auto cropped = quad.gridWindow(window);
    const std::array<std::array<float, 2>, 7> edges{{{-1, 3}, {59, 39}, {59.5f, 3}, {0, 0}, {29.5f, 5}, {20.25f, 10.5f}, {41, 25}}};
    std::vector<std::array<float, 2>> queries(edges.begin(), edges.end());
    for (float y = 0.25f; y < rows; y += 1.5f)
        for (float x = 0.5f; x < cols; x += 1.25f)
            queries.push_back({x, y});
    for (const auto& [x, y] : queries) {
        const float xy[]{x, y};
        cv::Vec3f xyz, normal, du, dv, expectedXyz, expectedNormal, expectedDu, expectedDv;
        whole->pointsAt(xy, 1, expectedXyz.val, expectedNormal.val, expectedDu.val, expectedDv.val);
        quad.dropPointSurface();
        quad.pointAt(x, y, xyz.val, normal.val, du.val, dv.val);
        pointsMatch = pointsMatch && same(xyz.val, expectedXyz.val) && same(normal.val, expectedNormal.val) &&
                      same(du.val, expectedDu.val) && same(dv.val, expectedDv.val);
        // Inside the window, by its pad.
        const int pad = QuadSurface::kGridWindowPad + 1;
        if (x >= window.x + pad && x < window.x + window.width - pad && y >= window.y + pad && y < window.y + window.height - pad) {
            const float local[]{x - window.x, y - window.y};
            cropped->pointsAt(local, 1, xyz.val, normal.val);
            windowsMatch = windowsMatch && same(xyz.val, expectedXyz.val) && same(normal.val, expectedNormal.val);
        }
    }
    CHECK(pointsMatch);
    CHECK(windowsMatch);

    // A frame's window covers its raster, with the pad, and stays inside the grid.
    const vcr::grid::Raster raster{10.5, 8.25, 0.5, 0.5, 20, 10};
    CHECK(vc3d::gridWindowFor(quad, raster, 2, 0, 0, 0) == cv::Rect(5, 4, 21, 15));
    CHECK(vc3d::gridWindowFor(quad, {-500, -500, 1, 1, 2000, 2000}, 0, 0, 0, 0) == cv::Rect(0, 0, cols, rows));
    CHECK(vc3d::gridWindowFor(quad, {5000, 5000, 1, 1, 10, 10}, 0, 0, 0, 0).empty() == false);  // clamped to the edge
}

TEST_CASE("retained native snapshots survive edits and owner destruction")
{
    std::shared_ptr<const vcr::grid::Surface> old;
    {
        cv::Mat_<cv::Vec3f> points(3,3);
        for(int y=0;y<3;++y) for(int x=0;x<3;++x) points(y,x)={float(x),float(y),5.f};
        QuadSurface quad(points,{1,1});
        old=vc3d::gridSurface(quad);
        for(auto& p:*quad.rawPointsPtr()) p[2]=9;
        quad.invalidateCache();
        const auto current=vc3d::gridSurface(quad);
        const float xy[]{1,1};float xyz[3];
        current->pointsAt(xy,1,xyz,nullptr);CHECK(xyz[2]==9);
    }
    const float xy[]{1,1};float xyz[3];
    old->pointsAt(xy,1,xyz,nullptr);CHECK(xyz[2]==5);
}
