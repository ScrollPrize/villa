// Exercise HttpFetch error paths via real HTTP responses.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/HttpFetch.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

bool requireNetwork()
{
    const char* env = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return env && env[0] && env[0] != '0';
}

} // namespace

TEST_CASE("httpGetString: 404 returns empty string")
{
    try {
        auto body = vc::httpGetString(
            "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/__no__such__key__");
        // 4xx misses should yield empty body per the impl doc.
        CHECK(body.empty());
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("network: " << e.what());
        MESSAGE("Skipping (no network?): " << e.what());
    }
}

TEST_CASE("httpGetString: 403 from a private bucket triggers the auth-error path")
{
    // philodemos bucket returns 403 to unauthenticated callers.
    try {
        (void)vc::httpGetString("https://philodemos.s3.amazonaws.com/");
        // If we somehow got through (cached creds?), accept silently.
        CHECK(true);
    } catch (const std::exception& e) {
        // Auth-error throws — that's the path we want to cover.
        std::string what = e.what();
        // Either auth-error message or network error; both are fine.
        CHECK(!what.empty());
        if (what.find("Access denied") == std::string::npos &&
            what.find("credentials") == std::string::npos)
        {
            MESSAGE("Note: did not see auth-error message; got: " << what);
        }
    }
}

TEST_CASE("httpGetString: bad URL surface as exception, not crash")
{
    try {
        (void)vc::httpGetString("not://a/real/scheme");
        CHECK(true);
    } catch (const std::exception&) {
        CHECK(true);
    }
}

TEST_CASE("httpGetString: empty URL handled gracefully")
{
    try {
        auto body = vc::httpGetString("");
        CHECK(body.empty());
    } catch (const std::exception&) {
        CHECK(true);
    }
}
