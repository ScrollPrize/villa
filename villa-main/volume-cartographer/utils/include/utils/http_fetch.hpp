#pragma once

// Only compile if curl is available
#if __has_include(<curl/curl.h>)
#define UTILS_HAS_CURL 1

#include <string>
#include <string_view>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <chrono>
#include <span>
#include <atomic>
#include <filesystem>

// NOTE: <curl/curl.h> is intentionally NOT included here. All curl-typed
// code lives in http_fetch.cpp so the ~88 TUs that pull this header in
// (mostly transitively via zarr.hpp) don't each re-parse curl.h.

namespace utils {

// ---------------------------------------------------------------------------
// HttpResponse
// ---------------------------------------------------------------------------
struct HttpResponse {
    long status_code = 0;
    std::vector<std::byte> body;
    std::string content_type;
    std::size_t content_length = 0;

    [[nodiscard]] bool ok() const noexcept { return status_code >= 200 && status_code < 300; }
    [[nodiscard]] bool not_found() const noexcept { return status_code == 404; }

    [[nodiscard]] std::string_view body_string() const noexcept {
        return {reinterpret_cast<const char*>(body.data()), body.size()};
    }
};

// ---------------------------------------------------------------------------
// HttpAuth
// ---------------------------------------------------------------------------
struct HttpAuth {
    std::string bearer_token;
    std::string username;
    std::string password;

    [[nodiscard]] bool empty() const noexcept {
        return bearer_token.empty() && username.empty() && password.empty();
    }
};

// ---------------------------------------------------------------------------
// S3 URL utilities
// ---------------------------------------------------------------------------
struct S3Url {
    std::string bucket;
    std::string key;
    std::string region; // empty if not specified
};

/// Recognises "s3://", "S3://", and "s3+REGION://" (e.g. "s3+us-west-2://").
[[nodiscard]] bool is_s3_url(std::string_view url) noexcept;

/// Parse an S3 URL. Supports "s3://bucket/key" and "s3+REGION://bucket/key".
[[nodiscard]] std::optional<S3Url> parse_s3_url(std::string_view url);

/// Convert an S3 URL to HTTPS. Uses the region from the URL if present,
/// otherwise falls back to path-style without a region subdomain.
[[nodiscard]] std::string s3_to_https(const S3Url& parsed);

/// Convert an S3 URL string to HTTPS.
[[nodiscard]] std::string s3_to_https(std::string_view s3_url);

// ---------------------------------------------------------------------------
// AwsAuth -- AWS SigV4 credentials
// ---------------------------------------------------------------------------
struct AwsAuth {
    std::string access_key;
    std::string secret_key;
    std::string session_token; // optional (STS temporary credentials)
    std::string region;        // e.g. "us-east-1"

    [[nodiscard]] bool empty() const noexcept {
        return access_key.empty() && secret_key.empty();
    }

    /// Read credentials from environment variables:
    ///   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    ///   AWS_SESSION_TOKEN, AWS_DEFAULT_REGION
    [[nodiscard]] static AwsAuth from_env() {
        AwsAuth auth;
        if (auto* v = std::getenv("AWS_ACCESS_KEY_ID"))      auth.access_key = v;
        if (auto* v = std::getenv("AWS_SECRET_ACCESS_KEY"))   auth.secret_key = v;
        if (auto* v = std::getenv("AWS_SESSION_TOKEN"))       auth.session_token = v;
        if (auto* v = std::getenv("AWS_DEFAULT_REGION"))      auth.region = v;
        return auth;
    }

    /// Full credential resolution. Tries in order:
    ///   1. `aws configure export-credentials` (resolves SSO, assume-role, etc.)
    ///   2. ~/.aws/credentials + ~/.aws/config INI files
    ///   3. Environment variables
    /// Respects AWS_PROFILE for methods 1 & 2.
    [[nodiscard]] static AwsAuth load(const std::string& profile = "default");
};

// ---------------------------------------------------------------------------
// HttpClient
// ---------------------------------------------------------------------------
class HttpClient final {
public:
    struct Config {
        HttpAuth auth{};
        AwsAuth aws_auth{};  // AWS SigV4 authentication (takes precedence over auth if non-empty)
        std::chrono::seconds connect_timeout{10};
        std::chrono::seconds transfer_timeout{30};
        bool follow_redirects{true};
        std::size_t max_retries{3};
        std::string user_agent{"utils-http/1.0"};
    };

    HttpClient() = default;

    explicit HttpClient(Config config)
        : config_(std::move(config))
    {
    }

    // Flip the process-global abort flag. Any in-flight curl_easy_perform
    // returns CURLE_ABORTED_BY_CALLBACK on the next progress tick (sub-
    // millisecond on an active socket) and pending retries bail. Use to
    // make app shutdown effectively instantaneous regardless of S3 timeout.
    static void abortAll() noexcept
    {
        abort_flag().store(true, std::memory_order_release);
    }

    // Reset the abort flag (e.g. for tests that re-use the process).
    static void resetAbort() noexcept
    {
        abort_flag().store(false, std::memory_order_release);
    }

    [[nodiscard]] static bool isAborted() noexcept
    {
        return abort_flag().load(std::memory_order_acquire);
    }

    // GET request
    [[nodiscard]] HttpResponse get(std::string_view url) const;

    // GET with byte range
    [[nodiscard]] HttpResponse get_range(std::string_view url,
                                          std::size_t offset, std::size_t length) const;

    // HEAD request (metadata only)
    [[nodiscard]] HttpResponse head(std::string_view url) const;

    // PUT request
    [[nodiscard]] HttpResponse put(std::string_view url,
                                    std::span<const std::byte> data,
                                    std::string_view content_type = "application/octet-stream") const;

    // PUT from file (streams from disk, constant memory)
    [[nodiscard]] HttpResponse put_file(std::string_view url,
                                         const std::filesystem::path& file_path,
                                         std::string_view content_type = "application/octet-stream") const;

    [[nodiscard]] static std::atomic<bool>& abort_flag() noexcept
    {
        static std::atomic<bool> flag{false};
        return flag;
    }

private:
    Config config_;
};

} // namespace utils

#endif // __has_include(<curl/curl.h>)
