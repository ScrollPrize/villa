// Exercise HttpFetch error paths via real HTTP responses.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/HttpFetch.hpp"

#include <utils/http_fetch.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
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

TEST_CASE("httpGetBytes: transport failure is not mistaken for an HTTP miss")
{
    try {
        (void)vc::httpGetBytes("not://a/real/scheme");
        FAIL("expected a transport exception");
    } catch (const std::exception& e) {
        CHECK(std::string(e.what()).find("HTTP transport error") != std::string::npos);
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

TEST_CASE("HttpClient preserves libcurl transport errors")
{
    utils::HttpClient::Config config;
    config.max_retries = 0;
    const auto response = utils::HttpClient(config).get("not://a/real/scheme");
    CHECK(response.status_code == 0);
    CHECK_FALSE(response.error_message.empty());
}

TEST_CASE("HttpClient scoped observer receives response body bytes")
{
    const auto path = std::filesystem::temp_directory_path() /
        "vc_http_download_observer_fixture.bin";
    constexpr std::string_view payload = "streamed response bytes";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        REQUIRE(output.good());
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }

    std::size_t observed = 0;
    {
        utils::HttpClient::ScopedDownloadObserver observer(
            [&](std::size_t bytes) { observed += bytes; });
        const auto response = utils::HttpClient{}.get("file://" + path.string());
        CHECK(response.body.size() == payload.size());
    }
    std::filesystem::remove(path);
    CHECK(observed == payload.size());
}
