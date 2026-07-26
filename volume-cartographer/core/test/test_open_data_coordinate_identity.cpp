#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "OpenDataCoordinateIdentity.hpp"

TEST_CASE("open-data coordinate identity emits the complete output schema")
{
    const std::vector<std::string> tags{
        "vc-open-data-coordinate-space:PHerc1451/20260319101107@L2",
        "vc-open-data-source-path:s3://path/to/the/source/volume",
        "vc-open-data-source-coordinate-level:2",
        "vc-open-data-source-original-resolution:2.4",
    };

    const auto identity = vc3d::opendata::coordinateIdentityFromTags(tags);
    REQUIRE(identity.has_value());
    CHECK(identity->coordinateSpace == "PHerc1451/20260319101107@L2");
    CHECK(identity->sourcePath == "s3://path/to/the/source/volume");
    CHECK(identity->sourceCoordinateLevel == 2);
    CHECK(identity->sourceCoordinateScaleFactor == 4);
    CHECK(identity->sourceOriginalResolution == doctest::Approx(2.4));

    const auto json = vc3d::opendata::coordinateIdentityJson(identity);
    CHECK(json["vc_open_data_coordinate_space"].get_string() ==
          "PHerc1451/20260319101107@L2");
    CHECK(json["vc_open_data_source_path"].get_string() ==
          "s3://path/to/the/source/volume");
    CHECK(json["vc_open_data_source_coordinate_level"].get_int() == 2);
    CHECK(json["vc_open_data_source_coordinate_scale_factor"].get_uint64() == 4);
    CHECK(json["vc_open_data_source_original_resolution"].get_double() ==
          doctest::Approx(2.4));
}

TEST_CASE("coordinate identity rejects incomplete coordinate tags")
{
    CHECK_FALSE(vc3d::opendata::coordinateIdentityFromTags({
        "vc-open-data-source-coordinate-level:2",
        "vc-open-data-source-path:s3://path/to/volume",
    }).has_value());
    CHECK_FALSE(vc3d::opendata::coordinateIdentityFromTags({
        "vc-open-data-coordinate-space:PHerc1451/20260319101107@L2",
        "vc-open-data-source-coordinate-level:not-an-int",
    }).has_value());
}
