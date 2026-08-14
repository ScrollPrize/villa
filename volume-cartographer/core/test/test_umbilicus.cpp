// Coverage for core/src/Umbilicus.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Umbilicus.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vc::core::util::Umbilicus;

namespace {

fs::path tmpFile(const std::string& tag, const std::string& ext)
{
    std::mt19937_64 rng(std::random_device{}());
    return fs::temp_directory_path() /
           ("vc_umb_" + tag + "_" + std::to_string(rng()) + ext);
}

void writeFile(const fs::path& p, const std::string& content)
{
    std::ofstream f(p);
    f << content;
}

} // namespace

TEST_CASE("FromPoints: requires positive volume shape")
{
    std::vector<cv::Vec3f> pts{{0, 0, 0}};
    CHECK_THROWS_AS(Umbilicus::FromPoints(pts, cv::Vec3i(0, 100, 100)), std::invalid_argument);
    CHECK_THROWS_AS(Umbilicus::FromPoints(pts, cv::Vec3i(100, 0, 100)), std::invalid_argument);
    CHECK_THROWS_AS(Umbilicus::FromPoints(pts, cv::Vec3i(100, 100, 0)), std::invalid_argument);
}

TEST_CASE("FromPoints: requires at least one control point")
{
    std::vector<cv::Vec3f> empty;
    CHECK_THROWS_AS(Umbilicus::FromPoints(empty, cv::Vec3i(10, 10, 10)), std::invalid_argument);
}

TEST_CASE("Single-point umbilicus: dense_centers populated, volume_shape preserved")
{
    std::vector<cv::Vec3f> pts{{5.f, 7.f, 3.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 20, 30));
    CHECK(u.volume_shape() == cv::Vec3i(10, 20, 30));
    CHECK_FALSE(u.centers().empty());
}

TEST_CASE("center_at: valid index returns interpolated center; out-of-range throws")
{
    std::vector<cv::Vec3f> pts{{0, 0, 0}, {5, 5, 10}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(20, 20, 20));
    CHECK(u.center_at(0)[0] == doctest::Approx(0.0f));
    CHECK_THROWS_AS(u.center_at(-1), std::out_of_range);
    CHECK_THROWS_AS(u.center_at(10000), std::out_of_range);
}

TEST_CASE("vector_to_umbilicus + distance_to_umbilicus")
{
    std::vector<cv::Vec3f> pts{{10.f, 10.f, 5.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(20, 20, 10));
    auto v = u.vector_to_umbilicus(cv::Vec3f(0, 0, 5));
    CHECK(v[0] == doctest::Approx(10.0f));
    CHECK(v[1] == doctest::Approx(10.0f));
    CHECK(u.distance_to_umbilicus(cv::Vec3f(0, 0, 5)) == doctest::Approx(std::sqrt(200.0)));
}

TEST_CASE("set_seam (cardinal directions) — has_seam and seam_direction round-trip")
{
    std::vector<cv::Vec3f> pts{{0, 0, 0}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 10, 10));
    CHECK_FALSE(u.has_seam());

    u.set_seam(Umbilicus::SeamDirection::PositiveX);
    CHECK(u.has_seam());
    CHECK(u.seam_direction() == Umbilicus::SeamDirection::PositiveX);

    u.set_seam(Umbilicus::SeamDirection::NegativeY);
    CHECK(u.seam_direction() == Umbilicus::SeamDirection::NegativeY);
}

TEST_CASE("seam_direction throws when only a free-form seam is set")
{
    std::vector<cv::Vec3f> pts{{5.f, 5.f, 5.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 10, 10));
    u.set_seam_from_point(cv::Vec3f(10.f, 5.f, 5.f));
    CHECK(u.has_seam());
    // seam_direction() is only valid for cardinal seams.
    CHECK_THROWS_AS(u.seam_direction(), std::logic_error);
}

TEST_CASE("set_seam_from_point rejects coincident point")
{
    std::vector<cv::Vec3f> pts{{5.f, 5.f, 5.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 10, 10));
    CHECK_THROWS_AS(u.set_seam_from_point(cv::Vec3f(5.f, 5.f, 5.f)),
                    std::invalid_argument);
}

TEST_CASE("seam_segment / seam_endpoints require a seam to be set")
{
    std::vector<cv::Vec3f> pts{{0, 0, 0}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 10, 10));
    CHECK_THROWS_AS(u.seam_segment(0), std::logic_error);
    CHECK_THROWS_AS(u.seam_endpoints(), std::logic_error);

    u.set_seam(Umbilicus::SeamDirection::PositiveX);
    auto seg = u.seam_segment(0);
    CHECK(seg.first[2] == doctest::Approx(seg.second[2]));
    CHECK_FALSE(u.seam_endpoints().empty());
    CHECK_THROWS_AS(u.seam_segment(-1), std::out_of_range);
    CHECK_THROWS_AS(u.seam_segment(10000), std::out_of_range);
}

TEST_CASE("theta requires a seam direction")
{
    std::vector<cv::Vec3f> pts{{0, 0, 0}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 10, 10));
    CHECK_THROWS_AS(u.theta(cv::Vec3f(1, 1, 1)), std::logic_error);
}

TEST_CASE("theta returns wrap_count*360 at coincident point")
{
    std::vector<cv::Vec3f> pts{{5.f, 5.f, 5.f}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(10, 10, 10));
    u.set_seam(Umbilicus::SeamDirection::PositiveX);
    CHECK(u.theta(cv::Vec3f(5.f, 5.f, 5.f)) == doctest::Approx(0.0));
    CHECK(u.theta(cv::Vec3f(5.f, 5.f, 5.f), 2) == doctest::Approx(720.0));
}

TEST_CASE("theta is 0 along seam, ~90 perpendicular")
{
    std::vector<cv::Vec3f> pts{{0, 0, 5}};
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(20, 20, 10));
    u.set_seam(Umbilicus::SeamDirection::PositiveX);
    CHECK(u.theta(cv::Vec3f(5.f, 0.f, 5.f)) == doctest::Approx(0.0));
    CHECK(u.theta(cv::Vec3f(0.f, 5.f, 5.f)) == doctest::Approx(90.0));
}

// ------- File loaders -------

TEST_CASE("FromFile (text): basic parse with comments and trimming")
{
    auto p = tmpFile("text_basic", ".txt");
    writeFile(p,
        "# comment\n"
        "0, 5, 10\n"
        "\n"
        "  10, 6, 11  \n"
        "20, 7, 12\n");
    auto u = Umbilicus::FromFile(p, cv::Vec3i(30, 30, 30));
    CHECK_FALSE(u.centers().empty());
    fs::remove(p);
}

TEST_CASE("FromFile (text): missing columns throws")
{
    auto p = tmpFile("text_missing", ".txt");
    writeFile(p, "0, 5\n"); // only 2 cols
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("FromFile (text): too many columns throws")
{
    auto p = tmpFile("text_extra", ".txt");
    writeFile(p, "0, 5, 10, 99\n");
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("FromFile (text): non-numeric token throws")
{
    auto p = tmpFile("text_nan", ".txt");
    writeFile(p, "0, hello, 10\n");
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("FromFile (text): empty file throws")
{
    auto p = tmpFile("text_empty", ".txt");
    writeFile(p, "\n# only a comment\n");
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("FromFile (missing path) throws")
{
    CHECK_THROWS_AS(Umbilicus::FromFile("/__nonexistent__/x.txt", cv::Vec3i(10, 10, 10)),
                    std::runtime_error);
}

TEST_CASE("FromFile (json array of [z,y,x]) parses")
{
    auto p = tmpFile("json_arr", ".json");
    writeFile(p, "[[0,5,10],[5,6,11],[10,7,12]]");
    auto u = Umbilicus::FromFile(p, cv::Vec3i(30, 30, 30));
    CHECK_FALSE(u.centers().empty());
    fs::remove(p);
}

TEST_CASE("FromFile (json object form with z/y/x keys) parses")
{
    auto p = tmpFile("json_obj", ".json");
    writeFile(p, R"([{"z":0,"y":5,"x":10},{"z":5,"y":6,"x":11}])");
    auto u = Umbilicus::FromFile(p, cv::Vec3i(30, 30, 30));
    CHECK_FALSE(u.centers().empty());
    fs::remove(p);
}

TEST_CASE("FromFile (json with 'points' wrapper) parses")
{
    auto p = tmpFile("json_wrap", ".json");
    writeFile(p, R"({"points":[[0,5,10],[5,6,11]]})");
    auto u = Umbilicus::FromFile(p, cv::Vec3i(30, 30, 30));
    CHECK_FALSE(u.centers().empty());
    fs::remove(p);
}

TEST_CASE("FromFile (json malformed root) throws")
{
    auto p = tmpFile("json_bad_root", ".json");
    writeFile(p, R"({"other":"value"})");
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("FromFile (json points wrapper not array) throws")
{
    auto p = tmpFile("json_bad_wrap", ".json");
    writeFile(p, R"({"points":"nope"})");
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("FromFile (json array entry with <3 fields) throws")
{
    auto p = tmpFile("json_short", ".json");
    writeFile(p, "[[0,5]]");
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("FromFile (json object missing key) throws")
{
    auto p = tmpFile("json_missing", ".json");
    writeFile(p, R"([{"z":0,"y":5}])");
    CHECK_THROWS_AS(Umbilicus::FromFile(p, cv::Vec3i(10, 10, 10)), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("centers interpolate between control points")
{
    std::vector<cv::Vec3f> pts{
        cv::Vec3f(0.f, 0.f, 0.f),
        cv::Vec3f(10.f, 10.f, 10.f)
    };
    auto u = Umbilicus::FromPoints(pts, cv::Vec3i(20, 20, 20));
    // dense_centers should fill at least range [0, max_z_of_controls]
    REQUIRE(u.centers().size() >= 11);
    // At z=5, x and y should be ~5 (linear interp)
    auto mid = u.center_at(5);
    CHECK(mid[0] == doctest::Approx(5.0f).epsilon(1e-3));
    CHECK(mid[1] == doctest::Approx(5.0f).epsilon(1e-3));
}

TEST_CASE("control_points json format loads via the shared loader")
{
    const auto path = tmpFile("control_points", ".json");
    writeFile(path,
              R"({"control_points": [)"
              R"({"x": 10, "y": 20, "z": 5, "score": 100},)"
              R"({"x": 12, "y": 22, "z": 15, "score": 100}],)"
              R"("metadata": {"total_points": 2}})");

    const auto points = Umbilicus::LoadControlPoints(path);
    REQUIRE(points.size() == 2);
    CHECK(points[0] == cv::Vec3f(10.0f, 20.0f, 5.0f));
    CHECK(points[1] == cv::Vec3f(12.0f, 22.0f, 15.0f));

    // FromFile shares the same parser, so the format works there too.
    const auto umbilicus = Umbilicus::FromFile(path, cv::Vec3i(20, 40, 40));
    CHECK(umbilicus.centers().size() == 20);

    fs::remove(path);
}

// ------- LoadFileInfo (frame metadata) -------

TEST_CASE("LoadFileInfo: stamped file reports all three metadata fields")
{
    const auto path = tmpFile("info_stamped", ".json");
    writeFile(path,
              R"({"metadata": {"total_points": 2, "voxelsize_um": 9.6,)"
              R"( "volume": "scroll.zarr", "zarr_level": 2},)"
              R"("control_points": [)"
              R"({"x": 10, "y": 20, "z": 5, "score": 100},)"
              R"({"x": 12, "y": 22, "z": 15, "score": 100}]})");

    const auto info = Umbilicus::LoadFileInfo(path);
    REQUIRE(info.controlPoints.size() == 2);
    CHECK(info.controlPoints[0] == cv::Vec3f(10.0f, 20.0f, 5.0f));
    REQUIRE(info.voxelsizeUm.has_value());
    CHECK(*info.voxelsizeUm == doctest::Approx(9.6));
    REQUIRE(info.volume.has_value());
    CHECK(*info.volume == "scroll.zarr");
    REQUIRE(info.zarrLevel.has_value());
    CHECK(*info.zarrLevel == 2);

    fs::remove(path);
}

TEST_CASE("LoadFileInfo: metadata without frame keys leaves every field unset")
{
    const auto path = tmpFile("info_unstamped", ".json");
    writeFile(path,
              R"({"metadata": {"total_points": 1, "timestamp": "2026-04-20T13:16:29Z"},)"
              R"("control_points": [{"x": 1, "y": 2, "z": 3, "score": 100}]})");

    const auto info = Umbilicus::LoadFileInfo(path);
    REQUIRE(info.controlPoints.size() == 1);
    CHECK_FALSE(info.voxelsizeUm.has_value());
    CHECK_FALSE(info.volume.has_value());
    CHECK_FALSE(info.zarrLevel.has_value());

    fs::remove(path);
}

TEST_CASE("LoadFileInfo: bare-array json parses points with no metadata")
{
    const auto path = tmpFile("info_bare", ".json");
    writeFile(path, "[[0,5,10],[10,6,11]]");

    const auto info = Umbilicus::LoadFileInfo(path);
    REQUIRE(info.controlPoints.size() == 2);
    CHECK(info.controlPoints[0] == cv::Vec3f(10.0f, 5.0f, 0.0f));
    CHECK_FALSE(info.voxelsizeUm.has_value());
    CHECK_FALSE(info.volume.has_value());
    CHECK_FALSE(info.zarrLevel.has_value());

    fs::remove(path);
}

TEST_CASE("LoadFileInfo: text file parses points with no metadata")
{
    const auto path = tmpFile("info_text", ".txt");
    writeFile(path, "0, 5, 10\n10, 6, 11\n");

    const auto info = Umbilicus::LoadFileInfo(path);
    REQUIRE(info.controlPoints.size() == 2);
    CHECK_FALSE(info.voxelsizeUm.has_value());
    CHECK_FALSE(info.volume.has_value());
    CHECK_FALSE(info.zarrLevel.has_value());

    fs::remove(path);
}

TEST_CASE("LoadFileInfo: a malformed field only costs that field")
{
    const std::string points =
        R"("control_points": [{"x": 1, "y": 2, "z": 3, "score": 100}]})";

    SUBCASE("malformed voxelsize_um")
    {
        const auto path = tmpFile("info_bad_voxelsize", ".json");
        writeFile(path,
                  R"({"metadata": {"voxelsize_um": "nine point six",)"
                  R"( "volume": "scroll.zarr", "zarr_level": 2},)" + points);

        const auto info = Umbilicus::LoadFileInfo(path);
        CHECK_FALSE(info.voxelsizeUm.has_value());
        CHECK(info.volume.value_or("") == "scroll.zarr");
        CHECK(info.zarrLevel.value_or(-1) == 2);
        fs::remove(path);
    }

    SUBCASE("non-positive voxelsize_um")
    {
        const auto path = tmpFile("info_zero_voxelsize", ".json");
        writeFile(path,
                  R"({"metadata": {"voxelsize_um": 0, "zarr_level": 1},)" + points);

        const auto info = Umbilicus::LoadFileInfo(path);
        CHECK_FALSE(info.voxelsizeUm.has_value());
        CHECK(info.zarrLevel.value_or(-1) == 1);
        fs::remove(path);
    }

    SUBCASE("malformed volume")
    {
        const auto path = tmpFile("info_bad_volume", ".json");
        writeFile(path,
                  R"({"metadata": {"voxelsize_um": 2.4, "volume": 7,)"
                  R"( "zarr_level": 0},)" + points);

        const auto info = Umbilicus::LoadFileInfo(path);
        CHECK(info.voxelsizeUm.value_or(0.0) == doctest::Approx(2.4));
        CHECK_FALSE(info.volume.has_value());
        CHECK(info.zarrLevel.value_or(-1) == 0);
        fs::remove(path);
    }

    SUBCASE("malformed zarr_level")
    {
        const auto path = tmpFile("info_bad_level", ".json");
        writeFile(path,
                  R"({"metadata": {"voxelsize_um": 2.4,)"
                  R"( "volume": "scroll.zarr", "zarr_level": "two"},)" + points);

        const auto info = Umbilicus::LoadFileInfo(path);
        CHECK(info.voxelsizeUm.value_or(0.0) == doctest::Approx(2.4));
        CHECK(info.volume.value_or("") == "scroll.zarr");
        CHECK_FALSE(info.zarrLevel.has_value());
        fs::remove(path);
    }

    SUBCASE("metadata is not an object")
    {
        const auto path = tmpFile("info_bad_metadata", ".json");
        writeFile(path, R"({"metadata": "nope", )" + points);

        const auto info = Umbilicus::LoadFileInfo(path);
        CHECK_FALSE(info.voxelsizeUm.has_value());
        CHECK_FALSE(info.volume.has_value());
        CHECK_FALSE(info.zarrLevel.has_value());
        fs::remove(path);
    }
}
