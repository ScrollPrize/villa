// vc_transform_geom's tifxyz transforms: an affine between uniform scales with
// regridding, and uniform scaling that keeps the grid (in memory and streamed).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/TransformSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Tiff.hpp"
#include <vc_object_store/Uuid.hpp>

#include <opencv2/core.hpp>

#include <cmath>
#include <memory>
#include <optional>
#include <chrono>
#include <fstream>
#include <limits>

using namespace vc::core::util;

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float spacing = 1.0f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c) * spacing,
                                static_cast<float>(r) * spacing,
                                0.f);
    return m;
}

} // namespace

TEST_CASE("no-resample scaling preserves samples, sentinels and physical units")
{
    auto pts = makePlanarGrid(4, 5, 20);
    pts(0, 0) = {-1, -1, -1};
    pts(0, 1) = {20, -1, 3}; // -1 is a sentinel only in the X band.
    QuadSurface surface(pts, {0.05f, 0.05f});
    surface.meta = utils::Json::parse(R"({"base_shape_zyx":[10,10,10],"output_step_vx":20,"voxel_size_um":9.6,"area_vx2":100,"area_cm2":2})");
    CHECK_THROWS(scaleSurfaceWithoutResampling(&surface, 4));
    CHECK_THROWS(scaleSurfaceWithoutResampling(&surface, 0));
    scaleSurfaceWithoutResampling(&surface, 4, std::array<int, 3>{40, 39, 39});
    CHECK(surface.rawPointsPtr()->rows == 4);
    CHECK(surface.rawPointsPtr()->cols == 5);
    CHECK((*surface.rawPointsPtr())(0, 0) == cv::Vec3f(-1, -1, -1));
    CHECK((*surface.rawPointsPtr())(0, 1) == cv::Vec3f(80, -4, 12));
    CHECK(surface.scale()[0] == doctest::Approx(0.0125));
    CHECK(surface.meta["output_step_vx"].get_double() == 80);
    CHECK(surface.meta["voxel_size_um"].get_double() == doctest::Approx(2.4));
    CHECK(surface.meta["area_vx2"].get_double() == 1600);
    CHECK(surface.meta["area_cm2"].get_double() == 2);
    CHECK(surface.meta["base_shape_zyx"][1].get_int() == 39);
}

TEST_CASE("streamed no-resample scaling reads tiled and stripped XYZ, publishes complete output")
{
    namespace fs = std::filesystem;
    struct Scratch {
        fs::path path = fs::current_path() / ("scale-tifxyz-test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Scratch() { fs::create_directory(path); }
        ~Scratch() { std::error_code ec; fs::remove_all(path, ec); }
    } scratch;
    for (int format = 0; format < 4; ++format) {
        const bool tiled = format == 1;
        const auto input = scratch.path / ("input-" + std::to_string(format));
        const auto output = scratch.path / ("output-" + std::to_string(format));
        fs::create_directory(input);
        auto pts = makePlanarGrid(19, 35, 20);
        for (auto& point : pts) point[2] = 10;
        pts(0, 0) = {-1, -1, -1};
        pts(0, 1) = {20, -1, 3};
        std::vector<cv::Mat> bands;
        cv::split(pts, bands);
        const std::array<const char*, 3> names{"x.tif", "y.tif", "z.tif"};
        for (int axis = 0; axis < 3; ++axis)
            writeTiff(input / names[axis], bands[axis], -1, tiled ? 16 : 0, tiled ? 16 : 0,
                -1, (tiled || format == 2) ? COMPRESSION_LZW : COMPRESSION_NONE);
        if (format == 3) {
            // Exercise direct byte-swapped strip reads independently of host endianness.
            for (int axis = 0; axis < 3; ++axis) {
                TIFF* out = TIFFOpen((input / names[axis]).string().c_str(), "wb");
                REQUIRE(out);
                TIFFSetField(out, TIFFTAG_IMAGEWIDTH, pts.cols);
                TIFFSetField(out, TIFFTAG_IMAGELENGTH, pts.rows);
                TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);
                TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
                TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
                TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
                TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 3);
                auto band = bands[axis].clone(); // libtiff may byte-swap the write buffer.
                for (int y = 0; y < pts.rows; ++y)
                    CHECK(TIFFWriteScanline(out, band.ptr(y), y) == 1);
                TIFFClose(out);
            }
        }
        std::ofstream(input / "meta.json") << R"({"type":"seg","format":"tifxyz","uuid":"original","name":"original","scale":[0.05,0.05],"base_shape_zyx":[10,10,10],"grid_shape":[19,35],"output_step_vx":20,"voxel_size_um":9.6,"bbox":[[0,0,0],[680,360,3]]})";
        std::ofstream(input / "ancillary.txt") << "unchanged";
        CHECK_THROWS(scaleTifxyzWithoutResampling(input, output, 4));
        CHECK_FALSE(fs::exists(output));
        scaleTifxyzWithoutResampling(input, output / "", 4, std::array<int, 3>{40,39,39});
        auto loaded = load_quad_from_tifxyz(output.string());
        REQUIRE(loaded);
        const auto* result = loaded->rawPointsPtr();
        REQUIRE(result->size() == pts.size());
        for (int y = 0; y < pts.rows; ++y)
            for (int x = 0; x < pts.cols; ++x)
                CHECK((*result)(y,x) == (pts(y,x)[0] == -1 ? pts(y,x) : pts(y,x) * 4));
        CHECK(loaded->scale()[0] == doctest::Approx(0.0125));
        CHECK(fs::file_size(output / "ancillary.txt") == 9);
        TIFF* tif = TIFFOpen((output / "x.tif").string().c_str(), "r");
        REQUIRE(tif);
        CHECK_FALSE(TIFFIsTiled(tif));
        uint16_t compression = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &compression);
        CHECK(compression == COMPRESSION_NONE);
        TIFFClose(tif);
        CHECK_THROWS(scaleTifxyzWithoutResampling(input, output, 4, std::array<int, 3>{40,39,39}));
        // Failure after staging starts must not leave a partial destination or temp directory.
        bands[2].setTo(std::numeric_limits<float>::max());
        writeTiff(input / "z.tif", bands[2], -1, 0, 0, -1, COMPRESSION_NONE);
        CHECK_THROWS(scaleTifxyzWithoutResampling(input, scratch.path / "overflow", 4,
            std::array<int, 3>{40,39,39}));
        CHECK_FALSE(fs::exists(scratch.path / "overflow"));
        for (const auto& entry : fs::directory_iterator(scratch.path))
            CHECK(entry.path().extension() != ".tmp");
    }
}

TEST_CASE("streamed scaling gives a new identity but retains coordinate provenance")
{
    namespace fs = std::filesystem;
    struct Scratch {
        fs::path path = fs::current_path() / ("scale-identity-test-" + vc::objects::util::newUuid());
        Scratch() { fs::create_directory(path); }
        ~Scratch() { std::error_code ec; fs::remove_all(path, ec); }
    } scratch;
    const auto input = scratch.path / "catalog";
    QuadSurface source(makePlanarGrid(4, 5, 20), {0.05f, 0.05f});
    source.meta = utils::Json::parse(R"({
        "base_shape_zyx":[10,10,10],
        "fit_config":{"voxel_size_um":9.6,"source":"original"},
        "vc_open_data_coordinate_space":"Paris4/scan@L2"
    })");
    source.save(input);
    const auto sourceUuid = source.objectUuid();
    REQUIRE(vc::objects::util::isUuid(sourceUuid));
    std::ofstream(input / "ancillary.txt") << "unchanged";
    const auto original = utils::Json::parse_file(input / "meta.json");
    std::string previousUuid;
    for (const auto* parent : {"first", "second"}) {
        const auto output = scratch.path / parent / "same-name";
        scaleTifxyzWithoutResampling(input, output, 4, std::array<int, 3>{40,39,39});
        const auto metadata = utils::Json::parse_file(output / "meta.json");
        const auto uuid = metadata["uuid"].get_string();
        CHECK(vc::objects::util::isUuid(uuid));
        CHECK(uuid != sourceUuid);
        CHECK(uuid != previousUuid);
        CHECK(metadata["name"].get_string() == "same-name");
        CHECK(metadata["vc_open_data_coordinate_space"].get_string() == "Paris4/scan@L2");
        CHECK(metadata["fit_config"].dump() == original["fit_config"].dump());
        CHECK(metadata["base_shape_zyx"][1].get_int() == 39);
        CHECK(metadata["voxel_size_um"].get_double() == doctest::Approx(2.4));
        CHECK(fs::file_size(output / "ancillary.txt") == 9);
        previousUuid = uuid;
    }
    source.saveAsNew(scratch.path / "gui-copy");
    const auto guiMeta = utils::Json::parse_file(scratch.path / "gui-copy/meta.json");
    CHECK(vc::objects::util::isUuid(guiMeta["uuid"].get_string()));
    CHECK(guiMeta["uuid"].get_string() != sourceUuid);
    CHECK(guiMeta["fit_config"].dump() == original["fit_config"].dump());
    CHECK(utils::Json::parse_file(input / "meta.json").dump() == original.dump());
}

TEST_CASE("a rigid registration maps every point exactly and keeps the grid")
{
    auto pts = makePlanarGrid(24, 24, 2.f);
    pts(3, 4) = cv::Vec3f(-1.f, -1.f, -1.f);
    QuadSurface qs(pts, cv::Vec2f(0.5f, 0.5f));
    // A quarter turn about z, then a shift.
    vcr::catalog::Affine m;
    m(0, 0) = 0; m(0, 1) = -1; m(1, 0) = 1; m(1, 1) = 0;
    m(0, 3) = 100; m(1, 3) = -20; m(2, 3) = 7;
    transformSurfacePoints(&qs, 1.0, m, 1.0);
    const auto& p = *qs.rawPointsPtr();
    REQUIRE(p.size() == pts.size());
    CHECK(p(3, 4) == cv::Vec3f(-1.f, -1.f, -1.f));
    for (int r = 0; r < p.rows; ++r)
        for (int c = 0; c < p.cols; ++c) {
            if (r == 3 && c == 4) continue;
            const auto q = m.apply({pts(r, c)[0], pts(r, c)[1], pts(r, c)[2]});
            CHECK(p(r, c) == cv::Vec3f(float(q[0]), float(q[1]), float(q[2])));
        }
}

TEST_CASE("scales before and after the affine compose, and the grid follows the spacing")
{
    auto pts = makePlanarGrid(24, 24);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 2.0, vcr::catalog::Affine::scale(1.5), 2.0);
    const auto& p = *qs.rawPointsPtr();
    // Six times the spacing: six times the samples per side, minus the last
    // interval, and the corners keep their scaled positions.
    CHECK(p.cols > 5 * pts.cols);
    CHECK(qs.scale() == cv::Vec2f(1.f, 1.f));
    CHECK(p(0, 0) == cv::Vec3f(0, 0, 0));
    const auto far = p(p.rows - 1, p.cols - 1);
    CHECK(far[0] == doctest::Approx(23 * 6).epsilon(0.05));
    CHECK(far[1] == doctest::Approx(23 * 6).epsilon(0.05));
}

TEST_CASE("points the affine takes out of float range become invalid")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    vcr::catalog::Affine m;
    m(0, 3) = 1e39; // beyond float range
    transformSurfacePoints(&qs, 1.0, m, 1.0);
    const auto& p = *qs.rawPointsPtr();
    for (int r = 0; r < p.rows; ++r)
        for (int c = 0; c < p.cols; ++c) CHECK(p(r, c) == cv::Vec3f(-1.f, -1.f, -1.f));
}

TEST_CASE("refreshTransformedSurfaceState writes bbox + scale into meta")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.5f, 2.0f));
    refreshTransformedSurfaceState(&qs);
    CHECK(qs.meta.is_object());
    CHECK(qs.meta.contains("bbox"));
    CHECK(qs.meta.contains("scale"));
    CHECK(qs.meta["scale"].is_array());
}

TEST_CASE("refreshTransformedSurfaceState preserves an existing object meta")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.meta = utils::Json::object();
    qs.meta["custom"] = "value";
    refreshTransformedSurfaceState(&qs);
    CHECK(qs.meta.contains("custom"));
    CHECK(qs.meta.contains("bbox"));
}
