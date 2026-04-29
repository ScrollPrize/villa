#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "utils/Json.hpp"
#include "vc/core/util/Zarr.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

struct TmpDir {
    fs::path path;

    TmpDir()
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        path = fs::temp_directory_path() / ("vc_zarr_attrs_" + std::to_string(rng()));
        fs::create_directories(path);
    }

    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

}  // namespace

TEST_CASE("writeZarrAttrs preserves anisotropic OME-Zarr physical scale")
{
    TmpDir dir;

    writeZarrAttrs(
        dir.path, "scan.zarr", 2,
        5, 3.0, 0.0, "max", 0,
        cv::Size(80, 40), 5, 16, 16,
        23.73, 15.82, "micrometer");

    const auto attrs = utils::Json::parse_file(dir.path / ".zattrs");
    REQUIRE(attrs.contains("multiscales"));

    const auto& multiscale = attrs["multiscales"][0];
    CHECK(multiscale["axes"][0]["name"].get_string() == "z");
    CHECK(multiscale["axes"][0]["unit"].get_string() == "micrometer");
    CHECK(multiscale["axes"][1]["name"].get_string() == "y");
    CHECK(multiscale["axes"][2]["name"].get_string() == "x");

    const auto& level0Scale = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"];
    CHECK(level0Scale[0].get_double() == doctest::Approx(23.73));
    CHECK(level0Scale[1].get_double() == doctest::Approx(15.82));
    CHECK(level0Scale[2].get_double() == doctest::Approx(15.82));

    const auto& level2Scale = multiscale["datasets"][2]["coordinateTransformations"][0]["scale"];
    CHECK(level2Scale[0].get_double() == doctest::Approx(23.73));
    CHECK(level2Scale[1].get_double() == doctest::Approx(63.28));
    CHECK(level2Scale[2].get_double() == doctest::Approx(63.28));
}
