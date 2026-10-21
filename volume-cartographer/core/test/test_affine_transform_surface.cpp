// Coverage gap-filler for AffineTransform.cpp's QuadSurface-touching code:
// measureGridAxisSpacing, transformSurfacePoints, refreshTransformedSurfaceState,
// cloneSurfaceForTransform (lines ~24-100 + ~375-495).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/util/Uuid.hpp"

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

cv::Matx44d scaleMatrix(double s)
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = s; m(1, 1) = s; m(2, 2) = s;
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

TEST_CASE("streamed scaling resets catalog identity but retains coordinate provenance")
{
    namespace fs = std::filesystem;
    struct Scratch {
        fs::path path = fs::current_path() / ("scale-identity-test-" + vc::util::newUuid());
        Scratch() { fs::create_directory(path); }
        ~Scratch() { std::error_code ec; fs::remove_all(path, ec); }
    } scratch;
    const auto input = scratch.path / "catalog";
    QuadSurface source(makePlanarGrid(4, 5, 20), {0.05f, 0.05f});
    source.meta = utils::Json::parse(R"({
        "vc_open_data_representation":{"sample_id":"Paris4"},
        "base_shape_zyx":[10,10,10],
        "fit_config":{"voxel_size_um":9.6,"source":"original"},
        "vc_open_data_coordinate_space":"Paris4/scan@L2"
    })");
    source.save(input);
    const auto sourceUuid = source.objectUuid();
    REQUIRE(vc::util::isUuid(sourceUuid));
    std::ofstream(input / "catalog-origin.json") << "{}";
    std::ofstream(input / "ancillary.txt") << "unchanged";
    const auto original = utils::Json::parse_file(input / "meta.json");
    std::string previousUuid;
    for (const auto* parent : {"first", "second"}) {
        const auto output = scratch.path / parent / "same-name";
        scaleTifxyzWithoutResampling(input, output, 4, std::array<int, 3>{40,39,39});
        const auto metadata = utils::Json::parse_file(output / "meta.json");
        const auto uuid = metadata["uuid"].get_string();
        CHECK(vc::util::isUuid(uuid));
        CHECK(uuid != sourceUuid);
        CHECK(uuid != previousUuid);
        CHECK(metadata["name"].get_string() == "same-name");
        CHECK_FALSE(metadata.contains("vc_open_data_representation"));
        CHECK_FALSE(fs::exists(output / "catalog-origin.json"));
        CHECK(metadata["vc_open_data_coordinate_space"].get_string() == "Paris4/scan@L2");
        CHECK(metadata["fit_config"].dump() == original["fit_config"].dump());
        CHECK(metadata["base_shape_zyx"][1].get_int() == 39);
        CHECK(metadata["voxel_size_um"].get_double() == doctest::Approx(2.4));
        CHECK(fs::file_size(output / "ancillary.txt") == 9);
        previousUuid = uuid;
    }
    source.saveAsNew(scratch.path / "gui-copy");
    const auto guiMeta = utils::Json::parse_file(scratch.path / "gui-copy/meta.json");
    CHECK(vc::util::isUuid(guiMeta["uuid"].get_string()));
    CHECK(guiMeta["uuid"].get_string() != sourceUuid);
    CHECK_FALSE(guiMeta.contains("vc_open_data_representation"));
    CHECK(guiMeta["fit_config"].dump() == original["fit_config"].dump());
    CHECK(utils::Json::parse_file(input / "meta.json").dump() == original.dump());
    CHECK(fs::exists(input / "catalog-origin.json"));
}

TEST_CASE("transformSurfacePoints (int scale): no-op on identity + scale=1")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 1, std::nullopt);
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    CHECK((*p)(0, 0) == cv::Vec3f(0, 0, 0));
}

TEST_CASE("transformSurfacePoints applies pre-scale factor to all points")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 2, std::nullopt); // pre-scale by 2x
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    // Spacing measurement should fire and resample back to "approximate
    // original spacing"; we mainly check no NaN and points are still valid.
    CHECK(std::isfinite((*p)(0, 0)[0]));
    CHECK(std::isfinite((*p)(p->rows - 1, p->cols - 1)[0]));
}

TEST_CASE("transformSurfacePoints with a 4x4 affine matrix (uniform scale)")
{
    auto pts = makePlanarGrid(20, 20); // big enough to skip the small-grid path
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto m = scaleMatrix(2.0);
    transformSurfacePoints(&qs, 1.0, m, 1.0);
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    CHECK(std::isfinite((*p)(0, 0)[0]));
}

TEST_CASE("transformSurfacePoints handles a tiny grid (small-grid branch)")
{
    // < 20 rows/cols triggers the "use whole grid" branch in
    // measureGridAxisSpacing.
    auto pts = makePlanarGrid(5, 5);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 1.0, std::nullopt, 1.0);
    CHECK(qs.rawPointsPtr() != nullptr);
}

TEST_CASE("transformSurfacePoints with sentinel points leaves them as sentinels")
{
    auto pts = makePlanarGrid(8, 8);
    pts(1, 1) = cv::Vec3f(-1.f, -1.f, -1.f);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 1, std::nullopt);
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    CHECK((*p)(1, 1)[0] == -1.f);
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

TEST_CASE("cloneSurfaceForTransform deep-copies points + metadata")
{
    auto pts = makePlanarGrid(4, 4);
    auto src = std::make_shared<QuadSurface>(pts, cv::Vec2f(1.f, 1.f));
    src->id = "src-id";
    src->meta = utils::Json::object();
    src->meta["k"] = "v";
    auto clone = cloneSurfaceForTransform(src);
    REQUIRE(clone != nullptr);
    CHECK(clone->id == "src-id");
    CHECK(clone->meta["k"].get_string() == "v");
    // Mutating the clone must not affect the source.
    auto* cp = clone->rawPointsPtr();
    REQUIRE(cp);
    (*cp)(0, 0) = cv::Vec3f(-999, -999, -999);
    CHECK((*src->rawPointsPtr())(0, 0)[0] != -999);
}

TEST_CASE("cloneSurfaceForTransform with null source returns null")
{
    CHECK(cloneSurfaceForTransform(nullptr) == nullptr);
}

TEST_CASE("cloneSurfaceForTransform handles null meta on the source")
{
    auto pts = makePlanarGrid(2, 2);
    auto src = std::make_shared<QuadSurface>(pts, cv::Vec2f(1.f, 1.f));
    // src->meta is default-constructed (null Json); clone should make it {}.
    auto clone = cloneSurfaceForTransform(src);
    REQUIRE(clone != nullptr);
    CHECK(clone->meta.is_object());
}
