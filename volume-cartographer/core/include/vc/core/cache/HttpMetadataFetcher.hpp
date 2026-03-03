#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vc::cache {

// Optional auth config for HTTP requests (e.g. AWS SigV4 signing).
struct HttpAuth {
    bool awsSigv4 = false;
    std::string region;        // e.g. "us-east-1"
    std::string accessKey;     // AWS_ACCESS_KEY_ID
    std::string secretKey;     // AWS_SECRET_ACCESS_KEY
    std::string sessionToken;  // AWS_SESSION_TOKEN (optional, for STS)
};

struct RemoteZarrInfo {
    std::string url;
    std::filesystem::path stagingDir;  // local dir with .zarray files
    std::string delimiter = ".";
    int numLevels = 0;
};

// Result of S3 ListObjectsV2 with delimiter.
struct S3ListResult {
    std::vector<std::string> prefixes;  // subdirectory prefixes (CommonPrefixes)
    std::vector<std::string> objects;   // object keys (Contents)
    bool authError = false;             // true if request failed due to auth (expired token, access denied)
    std::string errorMessage;           // S3 error message if any
};

// Fetch zarr metadata from a remote URL, write to local staging dir.
// Probes 0/.zarray, 1/.zarray, ... until 404.
// Creates: <stagingRoot>/<volumeId>/.zgroup, <volumeId>/0/.zarray, etc.
// Also synthesizes meta.json with dimensions from level 0's shape.
RemoteZarrInfo fetchRemoteZarrMetadata(
    const std::string& url,
    const std::filesystem::path& stagingRoot,
    const HttpAuth& auth = {});

// Fetch URL body as string. Empty on failure.
std::string httpGetString(const std::string& url, const HttpAuth& auth = {});

// List objects under an S3 prefix using ListObjectsV2.
// httpsBaseUrl should be the bucket URL (https://bucket.s3.region.amazonaws.com/prefix/).
// Returns prefixes (subdirectories) and object keys.
S3ListResult s3ListObjects(const std::string& httpsBaseUrl, const HttpAuth& auth = {});

// Download a URL to a local file. Returns true on success.
// Uses atomic write (temp file + rename).
bool httpDownloadFile(const std::string& url, const std::filesystem::path& dest, const HttpAuth& auth = {});

// ---- Shared curl auth helper (VC_USE_CURL only) -----------------------------
#ifdef VC_USE_CURL
#include <curl/curl.h>

// RAII guard that applies AWS SigV4 auth to a CURL handle.
// The guard owns the header slist and userpwd string, which must outlive
// curl_easy_perform(). Construct before perform, destroy after.
struct CurlAuthGuard {
    struct curl_slist* headers = nullptr;
    std::string userpwd;
    std::string sigv4;

    CurlAuthGuard() = default;
    ~CurlAuthGuard() { if (headers) curl_slist_free_all(headers); }

    CurlAuthGuard(const CurlAuthGuard&) = delete;
    CurlAuthGuard& operator=(const CurlAuthGuard&) = delete;
    CurlAuthGuard(CurlAuthGuard&& o) noexcept
        : headers(o.headers), userpwd(std::move(o.userpwd)), sigv4(std::move(o.sigv4)) { o.headers = nullptr; }
    CurlAuthGuard& operator=(CurlAuthGuard&& o) noexcept {
        if (this != &o) {
            if (headers) curl_slist_free_all(headers);
            headers = o.headers; o.headers = nullptr;
            userpwd = std::move(o.userpwd);
            sigv4 = std::move(o.sigv4);
        }
        return *this;
    }
};

// Apply AWS SigV4 auth options to a CURL handle if auth.awsSigv4 is set.
// Returns a guard whose lifetime must span the curl_easy_perform() call.
[[nodiscard]] inline CurlAuthGuard applyCurlAuth(CURL* curl, const HttpAuth& auth)
{
    CurlAuthGuard guard;
    if (!auth.awsSigv4) return guard;

    guard.sigv4 = "aws:amz:" + auth.region + ":s3";
    curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, guard.sigv4.c_str());
    guard.userpwd = auth.accessKey + ":" + auth.secretKey;
    curl_easy_setopt(curl, CURLOPT_USERPWD, guard.userpwd.c_str());
    if (!auth.sessionToken.empty()) {
        std::string hdr = "x-amz-security-token: " + auth.sessionToken;
        guard.headers = curl_slist_append(guard.headers, hdr.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, guard.headers);
    }
    return guard;
}
#endif  // VC_USE_CURL

}  // namespace vc::cache
