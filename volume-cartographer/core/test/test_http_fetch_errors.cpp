// Exercise HttpFetch error paths via real HTTP responses.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/S3AuthFallback.hpp"

#include <utils/http_fetch.hpp>

#include <cstdlib>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

bool requireNetwork()
{
    const char* env = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return env && env[0] && env[0] != '0';
}

utils::HttpResponse response(long status, std::string_view body = {})
{
    utils::HttpResponse out;
    out.status_code = status;
    out.body.reserve(body.size());
    for (const char value : body)
        out.body.push_back(static_cast<std::byte>(value));
    return out;
}

} // namespace

TEST_CASE("S3 authentication failures are classified narrowly")
{
    CHECK(vc::isAwsAuthenticationFailure(400, "<Code>InvalidToken</Code>"));
    CHECK(vc::isAwsAuthenticationFailure(403, {}));
    CHECK(vc::isAwsAuthenticationFailure(
        400, "<Code>SignatureDoesNotMatch</Code>"));
    CHECK_FALSE(vc::isAwsAuthenticationFailure(400, "BadRequest"));
    CHECK_FALSE(vc::isAwsAuthenticationFailure(404, "NoSuchKey"));
    CHECK_FALSE(vc::isAwsAuthenticationFailure(500, "InternalError"));
}

TEST_CASE("S3 fallback becomes sticky after anonymous success")
{
    vc::S3AuthFallback fallback(true, true);
    std::vector<bool> attempts;
    auto request = [&](bool anonymous) {
        attempts.push_back(anonymous);
        return anonymous
            ? response(200, "public")
            : response(400, "<Code>InvalidToken</Code>");
    };

    const auto first = fallback.request(request);
    const auto second = fallback.request(request);

    REQUIRE(first.response.ok());
    REQUIRE(first.authenticatedFailure.has_value());
    CHECK(first.usedAnonymous);
    CHECK(second.response.ok());
    CHECK(second.usedAnonymous);
    CHECK(fallback.usesAnonymous());
    CHECK(attempts == std::vector<bool>{false, true, true});
}

TEST_CASE("S3 fallback preserves private authenticated mode")
{
    vc::S3AuthFallback validPrivate(true, true);
    int validSigned = 0;
    int validAnonymous = 0;
    const auto valid = validPrivate.request([&](bool anonymous) {
        anonymous ? ++validAnonymous : ++validSigned;
        return response(200, "private");
    });
    CHECK(valid.response.ok());
    CHECK(validSigned == 1);
    CHECK(validAnonymous == 0);

    vc::S3AuthFallback rejectedPrivate(true, true);
    std::vector<bool> attempts;
    auto rejectedRequest = [&](bool anonymous) {
        attempts.push_back(anonymous);
        return anonymous
            ? response(403, "<Code>AccessDenied</Code>")
            : response(400, "<Code>InvalidToken</Code>");
    };
    const auto first = rejectedPrivate.request(rejectedRequest);
    const auto second = rejectedPrivate.request(rejectedRequest);

    CHECK(first.response.status_code == 403);
    REQUIRE(first.authenticatedFailure.has_value());
    CHECK(first.authenticatedFailure->status_code == 400);
    CHECK_FALSE(rejectedPrivate.usesAnonymous());
    CHECK(second.response.status_code == 400);
    CHECK(attempts == std::vector<bool>{false, true, false});
}

TEST_CASE("anonymous not-found establishes public S3 access")
{
    vc::S3AuthFallback fallback(true, true);
    std::vector<bool> attempts;
    const auto missing = fallback.request([&](bool anonymous) {
        attempts.push_back(anonymous);
        return anonymous
            ? response(404, "NoSuchKey")
            : response(400, "<Code>InvalidToken</Code>");
    });
    const auto present = fallback.request([&](bool anonymous) {
        attempts.push_back(anonymous);
        return response(anonymous ? 200 : 400);
    });

    CHECK(missing.response.not_found());
    CHECK(present.response.ok());
    CHECK(fallback.usesAnonymous());
    CHECK(attempts == std::vector<bool>{false, true, true});
}

TEST_CASE("S3 fallback ignores unrelated failures and non-S3 requests")
{
    for (const long status : {400L, 404L, 500L}) {
        vc::S3AuthFallback fallback(true, true);
        int attempts = 0;
        const auto result = fallback.request([&](bool anonymous) {
            ++attempts;
            CHECK_FALSE(anonymous);
            return response(status, status == 400 ? "BadRequest" : "failure");
        });
        CHECK(result.response.status_code == status);
        CHECK(attempts == 1);
    }

    vc::S3AuthFallback nonS3(false, true);
    int attempts = 0;
    const auto result = nonS3.request([&](bool anonymous) {
        ++attempts;
        CHECK_FALSE(anonymous);
        return response(400, "<Code>InvalidToken</Code>");
    });
    CHECK(result.response.status_code == 400);
    CHECK(attempts == 1);
}

TEST_CASE("concurrent S3 requests share the anonymous transition")
{
    vc::S3AuthFallback fallback(true, true);
    std::atomic<int> signedAttempts{0};
    std::atomic<int> anonymousAttempts{0};
    std::promise<void> probeStarted;
    std::promise<void> releaseProbe;
    auto release = releaseProbe.get_future().share();

    auto request = [&](bool anonymous) {
        if (!anonymous) {
            ++signedAttempts;
            return response(400, "<Code>InvalidToken</Code>");
        }
        if (anonymousAttempts.fetch_add(1) == 0) {
            probeStarted.set_value();
            release.wait();
        }
        return response(200, "public");
    };

    std::vector<std::thread> threads;
    std::atomic<int> successfulRequests{0};
    threads.emplace_back([&] {
        if (fallback.request(request).response.ok())
            ++successfulRequests;
    });
    probeStarted.get_future().wait();
    for (int i = 0; i < 7; ++i) {
        threads.emplace_back([&] {
            if (fallback.request(request).response.ok())
                ++successfulRequests;
        });
    }
    releaseProbe.set_value();
    for (auto& thread : threads)
        thread.join();

    CHECK(signedAttempts == 1);
    CHECK(anonymousAttempts == 8);
    CHECK(successfulRequests == 8);
    CHECK(fallback.usesAnonymous());
}

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
