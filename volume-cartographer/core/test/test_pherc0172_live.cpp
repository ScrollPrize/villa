// Live S3 smoke test against PHerc 0172 on the Vesuvius open-data bucket.
//
// Pulls a tiny piece of the smallest pyramid level (level 5, ~58 MB total,
// chunks 128^3 uint8) every time it runs. Anonymous read — bucket is public
// (us-east-1, --no-sign-request equivalent).
//
// The test soft-skips if the network fetch fails (so offline CI doesn't break).
// To force a hard fail on a missing fetch, set VC_TEST_REQUIRE_NETWORK=1.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include "utils/Json.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

constexpr const char* kZarrRoot =
    "s3://vesuvius-challenge-open-data/PHerc0172/volumes/"
    "20241024131838-7.910um-53keV-masked.zarr";

bool requireNetwork()
{
    const char* env = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return env && env[0] && env[0] != '0';
}

std::string fetchOrSoftSkip(const std::string& url)
{
    try {
        auto body = vc::httpGetString(url);
        if (body.empty()) {
            if (requireNetwork()) FAIL("Empty response from " << url);
            MESSAGE("Skipping: empty response from " << url);
            return {};
        }
        return body;
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("Network fetch failed: " << e.what());
        MESSAGE("Skipping (no network?): " << e.what());
        return {};
    }
}

} // namespace

TEST_CASE("resolveRemoteUrl produces the expected HTTPS form for PHerc 0172")
{
    auto r = vc::resolveRemoteUrl(std::string(kZarrRoot) + "/5/.zarray");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "us-east-1");
    CHECK(r.httpsUrl.find("vesuvius-challenge-open-data") != std::string::npos);
    CHECK(r.httpsUrl.find(".zarray") != std::string::npos);
}

TEST_CASE("PHerc 0172 zarr level 5 .zarray is reachable + has expected shape")
{
    auto r = vc::resolveRemoteUrl(std::string(kZarrRoot) + "/5/.zarray");
    auto body = fetchOrSoftSkip(r.httpsUrl);
    if (body.empty()) return; // soft-skipped

    auto j = utils::Json::parse(body);
    REQUIRE(j.contains("shape"));
    REQUIRE(j.contains("chunks"));
    REQUIRE(j.contains("dtype"));
    // Pinned values from the live bucket as observed 2026-05.
    CHECK(j["dtype"].get_string() == "|u1");
    CHECK(j["shape"].at(0).get_int64() == 651);
    CHECK(j["shape"].at(1).get_int64() == 210);
    CHECK(j["shape"].at(2).get_int64() == 285);
    CHECK(j["chunks"].at(0).get_int64() == 128);
}

TEST_CASE("PHerc 0172 zarr group .zattrs has multiscales metadata")
{
    auto r = vc::resolveRemoteUrl(std::string(kZarrRoot) + "/.zattrs");
    auto body = fetchOrSoftSkip(r.httpsUrl);
    if (body.empty()) return;

    auto j = utils::Json::parse(body);
    REQUIRE(j.contains("multiscales"));
    CHECK(j["multiscales"].is_array());
}

TEST_CASE("PHerc 0172 metadata.json carries zarr_export workflow info")
{
    auto r = vc::resolveRemoteUrl(std::string(kZarrRoot) + "/metadata.json");
    auto body = fetchOrSoftSkip(r.httpsUrl);
    if (body.empty()) return;

    auto j = utils::Json::parse(body);
    CHECK(j.contains("zarr_export"));
}
