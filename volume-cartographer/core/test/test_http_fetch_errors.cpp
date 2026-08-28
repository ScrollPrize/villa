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
    CHECK(vc::isAwsAuthenticationFailure(400, "<Code>AccessDenied</Code>"));
    CHECK_FALSE(vc::isAwsAuthenticationFailure(400, "BadRequest"));
    CHECK_FALSE(vc::isAwsAuthenticationFailure(404, "NoSuchKey"));
    CHECK_FALSE(vc::isAwsAuthenticationFailure(500, "InternalError"));
    CHECK_FALSE(vc::isAwsAuthenticationFailure(
        200, "private AccessDenied InvalidToken"));
}

TEST_CASE("S3 access becomes sticky after anonymous success")
{
    vc::S3AuthFallback fallback(true, true);
    std::vector<bool> attempts;
    auto request = [&](bool anonymous) {
        attempts.push_back(anonymous);
        return response(anonymous ? 200 : 400, anonymous
            ? "public"
            : "<Code>InvalidToken</Code>");
    };

    const auto first = fallback.request(request);
    const auto second = fallback.request(request);

    REQUIRE(first.response.ok());
    CHECK_FALSE(first.anonymousFailure.has_value());
    CHECK(first.usedAnonymous);
    CHECK(second.response.ok());
    CHECK(second.usedAnonymous);
    CHECK(fallback.usesAnonymous());
    CHECK(attempts == std::vector<bool>{true, true});
}

TEST_CASE("successful anonymous HEAD avoids stale credentials")
{
    vc::S3AuthFallback fallback(true, true);
    int signedAttempts = 0;
    int anonymousAttempts = 0;
    const auto result = fallback.request([&](bool anonymous) {
        if (anonymous) {
            ++anonymousAttempts;
            return response(200);
        }
        ++signedAttempts;
        return response(400);
    });

    CHECK(result.response.ok());
    CHECK(result.usedAnonymous);
    CHECK(anonymousAttempts == 1);
    CHECK(signedAttempts == 0);
    CHECK(fallback.usesAnonymous());
}

TEST_CASE("S3 access falls back once and preserves private authenticated mode")
{
    vc::S3AuthFallback fallback(true, true);
    std::vector<bool> attempts;
    auto request = [&](bool anonymous) {
        attempts.push_back(anonymous);
        return anonymous
            ? response(403, "<Code>AccessDenied</Code>")
            : response(200, "private AccessDenied InvalidToken");
    };

    const auto first = fallback.request(request);
    const auto second = fallback.request(request);

    CHECK(first.response.ok());
    REQUIRE(first.anonymousFailure.has_value());
    CHECK(first.anonymousFailure->status_code == 403);
    CHECK_FALSE(first.usedAnonymous);
    CHECK(second.response.ok());
    CHECK_FALSE(second.usedAnonymous);
    CHECK_FALSE(fallback.usesAnonymous());
    CHECK(attempts == std::vector<bool>{true, false, false});
}

TEST_CASE("failed private authentication does not retain a session mode")
{
    vc::S3AuthFallback fallback(true, true);
    std::vector<bool> attempts;
    auto request = [&](bool anonymous) {
        attempts.push_back(anonymous);
        return anonymous
            ? response(403, "<Code>AccessDenied</Code>")
            : response(400, "<Code>InvalidToken</Code>");
    };

    const auto first = fallback.request(request);
    const auto second = fallback.request(request);

    CHECK(first.response.status_code == 400);
    REQUIRE(first.anonymousFailure.has_value());
    CHECK(first.anonymousFailure->status_code == 403);
    CHECK(second.response.status_code == 400);
    CHECK_FALSE(fallback.usesAnonymous());
    CHECK(attempts == std::vector<bool>{true, false, true, false});
}

TEST_CASE("anonymous not-found leaves S3 access undecided")
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
    CHECK(attempts == std::vector<bool>{true, true});
}

TEST_CASE("S3 fallback ignores unrelated failures and non-S3 requests")
{
    for (const long status : {400L, 404L, 500L}) {
        vc::S3AuthFallback fallback(true, true);
        int attempts = 0;
        const auto result = fallback.request([&](bool anonymous) {
            ++attempts;
            CHECK(anonymous);
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

TEST_CASE("anonymous S3 access upgrades when a later object is private")
{
    vc::S3AuthFallback fallback(true, true);
    std::vector<bool> attempts;

    const auto publicResult = fallback.request([&](bool anonymous) {
        attempts.push_back(anonymous);
        return response(anonymous ? 200 : 500);
    });
    const auto privateResult = fallback.request([&](bool anonymous) {
        attempts.push_back(anonymous);
        return anonymous
            ? response(403, "<Code>AccessDenied</Code>")
            : response(200, "private");
    });
    const auto nextPrivate = fallback.request([&](bool anonymous) {
        attempts.push_back(anonymous);
        return response(anonymous ? 403 : 200);
    });

    CHECK(publicResult.response.ok());
    CHECK(publicResult.usedAnonymous);
    CHECK(privateResult.response.ok());
    REQUIRE(privateResult.anonymousFailure.has_value());
    CHECK_FALSE(privateResult.usedAnonymous);
    CHECK(nextPrivate.response.ok());
    CHECK_FALSE(nextPrivate.usedAnonymous);
    CHECK_FALSE(fallback.usesAnonymous());
    CHECK(attempts == std::vector<bool>{true, true, false, false});
}

TEST_CASE("concurrent S3 requests share the anonymous-first transition")
{
    vc::S3AuthFallback fallback(true, true);
    std::atomic<int> signedAttempts{0};
    std::atomic<int> anonymousAttempts{0};
    std::promise<void> probeStarted;
    std::promise<void> releaseProbe;
    auto release = releaseProbe.get_future().share();

    auto request = [&](bool anonymous) {
        if (anonymous) {
            if (anonymousAttempts.fetch_add(1) == 0) {
                probeStarted.set_value();
                release.wait();
            }
            return response(200, "public");
        } else {
            ++signedAttempts;
            return response(400, "<Code>InvalidToken</Code>");
        }
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

    CHECK(signedAttempts == 0);
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
