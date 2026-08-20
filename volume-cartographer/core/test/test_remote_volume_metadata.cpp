#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/RemoteVolumeMetadata.hpp"

#include "utils/Json.hpp"

TEST_CASE("samplePixelSize is found in published nested layouts")
{
    const auto legacy = utils::Json::parse(R"({
        "scan": {"tomo": {"acquisition": {"detector": {
            "samplePixelSize": 0.00791
        }}}}
    })");
    const auto legacyValue = vc::findUnambiguousSamplePixelSizeMillimeters(legacy);
    REQUIRE(legacyValue);
    CHECK(*legacyValue == doctest::Approx(0.00791));

    const auto surface = utils::Json::parse(R"({
        "source": {"metadata": {"detector": {
            "samplePixelSize": 0.0024
        }}}
    })");
    const auto surfaceValue = vc::findUnambiguousSamplePixelSizeMillimeters(surface);
    REQUIRE(surfaceValue);
    CHECK(*surfaceValue == doctest::Approx(0.0024));

    const auto noScanWrapper = utils::Json::parse(R"({
        "tomo": {"acquisition": {"detector": {
            "samplePixelSize": 0.001129
        }}}
    })");
    const auto noScanValue = vc::findUnambiguousSamplePixelSizeMillimeters(noScanWrapper);
    REQUIRE(noScanValue);
    CHECK(*noScanValue == doctest::Approx(0.001129));
}

TEST_CASE("samplePixelSize search traverses arrays")
{
    const auto metadata = utils::Json::parse(R"({
        "sources": [{"metadata": {"samplePixelSize": 0.00324}}]
    })");
    const auto value = vc::findUnambiguousSamplePixelSizeMillimeters(metadata);
    REQUIRE(value);
    CHECK(*value == doctest::Approx(0.00324));
}

TEST_CASE("identical repeated samplePixelSize values are accepted")
{
    const auto metadata = utils::Json::parse(R"({
        "detector": {"samplePixelSize": 0.00791},
        "provenance": {"detector": {"samplePixelSize": 0.00791}}
    })");
    const auto value = vc::findUnambiguousSamplePixelSizeMillimeters(metadata);
    REQUIRE(value);
    CHECK(*value == doctest::Approx(0.00791));
}

TEST_CASE("conflicting samplePixelSize values are rejected")
{
    const auto metadata = utils::Json::parse(R"({
        "detector": {"samplePixelSize": 0.00791},
        "source": {"metadata": {"samplePixelSize": 0.00324}}
    })");
    CHECK_FALSE(vc::findUnambiguousSamplePixelSizeMillimeters(metadata));
}

TEST_CASE("invalid or absent samplePixelSize values are ignored")
{
    const auto invalid = utils::Json::parse(R"({
        "samplePixelSize": -1,
        "metadata": {"samplePixelSize": "0.00791"}
    })");
    CHECK_FALSE(vc::findUnambiguousSamplePixelSizeMillimeters(invalid));
    CHECK_FALSE(vc::findUnambiguousSamplePixelSizeMillimeters(utils::Json::object()));
}
