#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/RemoteUrl.hpp"

#include <string>


TEST_CASE("remote volume selector is parsed and canonicalized")
{
    const auto spec = vc::parseRemoteVolumeSpec(
        "https://example.test/source.zarr/#vc-base-scale=02");
    CHECK(spec.sourceUrl == "https://example.test/source.zarr");
    CHECK(spec.portableLocator ==
          "https://example.test/source.zarr#vc-base-scale=2");
    CHECK(spec.baseScaleLevel == 2);
}

TEST_CASE("base scale zero canonicalizes to native locator")
{
    const auto spec = vc::parseRemoteVolumeSpec(
        "s3://bucket/source.zarr#vc-base-scale=0");
    CHECK(spec.sourceUrl ==
          "https://bucket.s3.us-east-1.amazonaws.com/source.zarr");
    CHECK(spec.portableLocator == spec.sourceUrl);
    CHECK(spec.baseScaleLevel == 0);
    CHECK(spec.useAwsSigv4);
}

TEST_CASE("selector parsing rejects malformed fragments")
{
    for (const std::string& locator : {
             "https://e.test/a.zarr#",
             "https://e.test/a.zarr#unknown=2",
             "https://e.test/a.zarr#vc-base-scale",
             "https://e.test/a.zarr#vc-base-scale=-1",
             "https://e.test/a.zarr#vc-base-scale=1.5",
             "https://e.test/a.zarr#vc-base-scale=6",
             "https://e.test/a.zarr#vc-base-scale=1&vc-base-scale=2"}) {
        CAPTURE(locator);
        CHECK_THROWS_AS(vc::parseRemoteVolumeSpec(locator), std::invalid_argument);
    }
}


TEST_CASE("fragment-free remote volume specs retain existing resolution")
{
    const std::string input = "https://example.test/source.zarr/?token=x/";
    const auto spec = vc::parseRemoteVolumeSpec(input);
    // Slashes ending the path go; a query's bytes (a signed token) stay as they are.
    CHECK(spec.sourceUrl == "https://example.test/source.zarr?token=x/");
    CHECK(spec.portableLocator == spec.sourceUrl);
    CHECK(spec.baseScaleLevel == 0);
}
