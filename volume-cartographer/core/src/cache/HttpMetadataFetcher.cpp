#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <cstdio>
#include <fstream>
#include <future>
#include <optional>
#include <nlohmann/json.hpp>

#ifdef VC_USE_CURL
#include <curl/curl.h>
#endif

namespace vc::cache {

// ---- curl helpers -----------------------------------------------------------

#ifdef VC_USE_CURL
static size_t stringWriteCallback(
    char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* str = static_cast<std::string*>(userdata);
    str->append(ptr, size * nmemb);
    return size * nmemb;
}
#endif

std::string httpGetString(const std::string& url, const HttpAuth& auth)
{
#ifdef VC_USE_CURL
    // Thread-local CURL handle: reuses TCP+TLS connections across requests
    thread_local CURL* curl = [] {
        CURL* c = curl_easy_init();
        if (c) {
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        }
        return c;
    }();
    if (!curl) return {};

    // Reset clears all options but keeps the connection alive
    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

    std::string response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stringWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    auto authGuard = applyCurlAuth(curl, auth);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) return {};
    return response;
#else
    (void)url;
    (void)auth;
    return {};
#endif
}

// ---- S3 listing -------------------------------------------------------------

// Simple XML tag extraction — finds all occurrences of <tag>...</tag> in xml
static std::vector<std::string> extractXmlTags(const std::string& xml, const std::string& tag)
{
    std::vector<std::string> results;
    const std::string openTag = "<" + tag + ">";
    const std::string closeTag = "</" + tag + ">";
    size_t pos = 0;
    while (true) {
        pos = xml.find(openTag, pos);
        if (pos == std::string::npos) break;
        pos += openTag.size();
        auto end = xml.find(closeTag, pos);
        if (end == std::string::npos) break;
        results.push_back(xml.substr(pos, end - pos));
        pos = end + closeTag.size();
    }
    return results;
}

// Parse S3 HTTPS URL into bucket host and prefix.
// Input: "https://bucket.s3.region.amazonaws.com/some/prefix/"
// Output: bucketHost = "https://bucket.s3.region.amazonaws.com"
//         prefix = "some/prefix/"
static bool parseS3Url(const std::string& url, std::string& bucketHost, std::string& prefix)
{
    // Find scheme
    auto schemeEnd = url.find("://");
    if (schemeEnd == std::string::npos) return false;
    auto pathStart = url.find('/', schemeEnd + 3);
    if (pathStart == std::string::npos) {
        bucketHost = url;
        prefix = "";
    } else {
        bucketHost = url.substr(0, pathStart);
        prefix = url.substr(pathStart + 1);
    }
    return true;
}

S3ListResult s3ListObjects(const std::string& httpsBaseUrl, const HttpAuth& auth)
{
    S3ListResult result;

#ifdef VC_USE_CURL
    std::string bucketHost, prefix;
    if (!parseS3Url(httpsBaseUrl, bucketHost, prefix)) {
        return result;
    }

    // Build ListObjectsV2 URL
    std::string listUrl = bucketHost + "/?list-type=2&delimiter=/";
    if (!prefix.empty()) {
        // URL-encode is not needed for simple path prefixes
        listUrl += "&prefix=" + prefix;
    }

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[S3] ListObjects: %s\n", listUrl.c_str());

    // Thread-local CURL handle without FAILONERROR so we can log the status
    thread_local CURL* curl = [] {
        CURL* c = curl_easy_init();
        if (c) {
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        }
        return c;
    }();

    std::string xml;
    {
        if (!curl) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[S3] Failed to init curl\n");
            return result;
        }

        // Reset clears all options but keeps the connection alive
        curl_easy_reset(curl);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

        curl_easy_setopt(curl, CURLOPT_URL, listUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stringWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &xml);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

        auto authGuard = applyCurlAuth(curl, auth);

        CURLcode res = curl_easy_perform(curl);

        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

        if (res != CURLE_OK) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[S3] ListObjects curl error: %s\n",
                             curl_easy_strerror(res));
            return result;
        }

        if (httpCode != 200) {
            if (auto* log = cacheDebugLog()) {
                std::fprintf(log, "[S3] ListObjects HTTP %ld\n", httpCode);
                if (!xml.empty()) {
                    std::fprintf(log, "[S3] Response: %.500s\n", xml.c_str());
                }
            }

            // Detect auth errors so callers can prompt for fresh credentials
            if (httpCode == 400 || httpCode == 401 || httpCode == 403) {
                // Check for known S3 auth error codes in the XML body
                auto codes = extractXmlTags(xml, "Code");
                for (const auto& code : codes) {
                    if (code == "ExpiredToken" || code == "AccessDenied" ||
                        code == "InvalidAccessKeyId" || code == "SignatureDoesNotMatch" ||
                        code == "TokenRefreshRequired") {
                        result.authError = true;
                        auto msgs = extractXmlTags(xml, "Message");
                        if (!msgs.empty()) result.errorMessage = msgs.front();
                        break;
                    }
                }
                // If no specific code found but it's a 401/403, still flag as auth
                if (!result.authError && (httpCode == 401 || httpCode == 403)) {
                    result.authError = true;
                    result.errorMessage = "HTTP " + std::to_string(httpCode);
                }
            }

            return result;
        }
    }

    if (xml.empty()) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[S3] ListObjects returned empty response\n");
        return result;
    }

    // Parse <CommonPrefixes><Prefix>...</Prefix></CommonPrefixes>
    // The S3 response nests Prefix inside CommonPrefixes, but our simple
    // extractor just gets all <Prefix> tags directly. Filter for the ones
    // that start with our prefix.
    for (const auto& p : extractXmlTags(xml, "Prefix")) {
        if (prefix.empty() || p.rfind(prefix, 0) == 0) {
            // Strip the parent prefix to get just the subdirectory name
            std::string relative = p.substr(prefix.size());
            // Remove trailing slash
            while (!relative.empty() && relative.back() == '/') {
                relative.pop_back();
            }
            if (!relative.empty()) {
                result.prefixes.push_back(relative);
            }
        }
    }

    // Parse <Contents><Key>...</Key></Contents>
    for (const auto& k : extractXmlTags(xml, "Key")) {
        if (prefix.empty() || k.rfind(prefix, 0) == 0) {
            std::string relative = k.substr(prefix.size());
            if (!relative.empty()) {
                result.objects.push_back(relative);
            }
        }
    }

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[S3] Found %zu prefixes, %zu objects\n",
                     result.prefixes.size(), result.objects.size());
#else
    (void)httpsBaseUrl;
    (void)auth;
#endif

    return result;
}

// ---- HTTP file download -----------------------------------------------------

#ifdef VC_USE_CURL
static size_t fileWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* fp = static_cast<std::FILE*>(userdata);
    return std::fwrite(ptr, size, nmemb, fp);
}
#endif

bool httpDownloadFile(const std::string& url, const std::filesystem::path& dest, const HttpAuth& auth)
{
#ifdef VC_USE_CURL
    namespace fs = std::filesystem;

    // Write to temp file, then atomic rename
    auto tempPath = dest;
    tempPath += ".tmp";
    fs::create_directories(dest.parent_path());

    std::FILE* fp = std::fopen(tempPath.c_str(), "wb");
    if (!fp) return false;

    // Thread-local CURL handle: reuses TCP+TLS connections across requests
    thread_local CURL* curl = [] {
        CURL* c = curl_easy_init();
        if (c) {
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        }
        return c;
    }();
    if (!curl) {
        std::fclose(fp);
        return false;
    }

    // Reset clears all options but keeps the connection alive
    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fileWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    auto authGuard = applyCurlAuth(curl, auth);

    CURLcode res = curl_easy_perform(curl);
    std::fclose(fp);

    if (res != CURLE_OK) {
        std::error_code ec;
        fs::remove(tempPath, ec);
        return false;
    }

    // Atomic rename
    std::error_code ec;
    fs::rename(tempPath, dest, ec);
    return !ec;
#else
    (void)url;
    (void)dest;
    (void)auth;
    return false;
#endif
}

// ---- metadata fetcher -------------------------------------------------------

static std::string deriveVolumeId(const std::string& url)
{
    // Extract last non-empty path component from URL
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    auto pos = u.rfind('/');
    if (pos != std::string::npos) {
        return u.substr(pos + 1);
    }
    return u;
}

static void writeFile(const std::filesystem::path& path, const std::string& content)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
}

static std::string readFile(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Check if staging dir already has valid cached metadata (meta.json + at least 0/.zarray).
// If so, return the info without hitting the network.
static std::optional<RemoteZarrInfo> tryLoadCachedMetadata(
    const std::string& baseUrl,
    const std::filesystem::path& stagingDir)
{
    namespace fs = std::filesystem;

    auto metaPath = stagingDir / "meta.json";
    auto level0Zarray = stagingDir / "0" / ".zarray";
    if (!fs::exists(metaPath) || !fs::exists(level0Zarray)) {
        return std::nullopt;
    }

    // Count cached levels
    int numLevels = 0;
    for (int lvl = 0; lvl < 20; lvl++) {
        if (fs::exists(stagingDir / std::to_string(lvl) / ".zarray")) {
            numLevels++;
        } else {
            break;
        }
    }

    if (numLevels == 0) return std::nullopt;

    // Parse delimiter from level 0
    std::string delimiter = ".";
    auto zarray0 = readFile(level0Zarray);
    if (!zarray0.empty()) {
        try {
            auto j = nlohmann::json::parse(zarray0);
            if (j.contains("dimension_separator")) {
                delimiter = j["dimension_separator"].get<std::string>();
            }
        } catch (...) {}
    }

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Using cached metadata from %s (%d levels)\n",
                     stagingDir.c_str(), numLevels);

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels
    };
}

RemoteZarrInfo fetchRemoteZarrMetadata(
    const std::string& url,
    const std::filesystem::path& stagingRoot,
    const HttpAuth& auth)
{
    // Normalize URL: strip trailing slashes
    std::string baseUrl = url;
    while (!baseUrl.empty() && baseUrl.back() == '/') baseUrl.pop_back();

    std::string volumeId = deriveVolumeId(baseUrl);
    auto stagingDir = stagingRoot / volumeId;

    // Clean up stale sibling .chunks dir from older code path
    {
        auto staleChunks = stagingRoot / (volumeId + ".chunks");
        std::error_code ec;
        if (std::filesystem::exists(staleChunks, ec)) {
            std::filesystem::remove_all(staleChunks, ec);
        }
    }

    // Try cached metadata first — avoids network round-trips on subsequent opens
    if (auto cached = tryLoadCachedMetadata(baseUrl, stagingDir)) {
        return *cached;
    }

    std::filesystem::create_directories(stagingDir);

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Fetching metadata for %s -> %s\n",
                     baseUrl.c_str(), stagingDir.c_str());

    // Fetch .zgroup
    auto zgroup = httpGetString(baseUrl + "/.zgroup", auth);
    if (!zgroup.empty()) {
        writeFile(stagingDir / ".zgroup", zgroup);
    } else {
        // Synthesize minimal .zgroup
        writeFile(stagingDir / ".zgroup", R"({"zarr_format":2})");
    }

    // Fetch .zattrs (optional, may 404)
    auto zattrs = httpGetString(baseUrl + "/.zattrs", auth);
    if (!zattrs.empty()) {
        writeFile(stagingDir / ".zattrs", zattrs);
    }

    // Probe levels concurrently in batches of 8.
    // Most volumes have fewer than 8 pyramid levels, so the first batch
    // typically discovers all of them in a single round-trip.
    std::string delimiter = ".";
    int numLevels = 0;
    nlohmann::json level0Meta;

    constexpr int kBatchSize = 8;
    constexpr int kMaxLevels = 20;

    for (int batchStart = 0; batchStart < kMaxLevels; batchStart += kBatchSize) {
        int batchEnd = std::min(batchStart + kBatchSize, kMaxLevels);

        // Launch async probes for this batch
        std::vector<std::future<std::string>> futures;
        futures.reserve(batchEnd - batchStart);
        for (int lvl = batchStart; lvl < batchEnd; lvl++) {
            std::string lvlUrl = baseUrl + "/" + std::to_string(lvl) + "/.zarray";
            futures.push_back(std::async(std::launch::async,
                [lvlUrl, &auth]() { return httpGetString(lvlUrl, auth); }));
        }

        // Collect results in order
        bool batchHadMiss = false;
        for (int i = 0; i < static_cast<int>(futures.size()); i++) {
            int lvl = batchStart + i;
            auto zarray = futures[i].get();

            if (zarray.empty()) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Level %d: no .zarray\n", lvl);
                batchHadMiss = true;
                break;  // Stop at first gap — levels must be contiguous
            }

            auto levelDir = stagingDir / std::to_string(lvl);
            writeFile(levelDir / ".zarray", zarray);
            numLevels++;

            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[REMOTE] Level %d: fetched .zarray (%zu bytes)\n",
                             lvl, zarray.size());

            // Parse level 0 for shape and delimiter
            if (lvl == 0) {
                try {
                    level0Meta = nlohmann::json::parse(zarray);
                    if (level0Meta.contains("dimension_separator")) {
                        delimiter = level0Meta["dimension_separator"].get<std::string>();
                    }
                } catch (const std::exception& e) {
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "[REMOTE] Warning: failed to parse level 0 .zarray: %s\n", e.what());
                }
            }
        }

        // If any level in this batch was missing, no need to probe further
        if (batchHadMiss) break;
    }

    if (numLevels == 0) {
        throw std::runtime_error("No zarr levels found at " + baseUrl);
    }

    // Synthesize meta.json from level 0 shape
    int width = 0, height = 0, slices = 0;
    if (level0Meta.contains("shape") && level0Meta["shape"].is_array() &&
        level0Meta["shape"].size() >= 3) {
        // zarr shape is [z, y, x]
        slices = level0Meta["shape"][0].get<int>();
        height = level0Meta["shape"][1].get<int>();
        width  = level0Meta["shape"][2].get<int>();
    }

    nlohmann::json meta;
    meta["uuid"] = volumeId;
    meta["name"] = volumeId;
    meta["type"] = "vol";
    meta["width"] = width;
    meta["height"] = height;
    meta["slices"] = slices;
    meta["format"] = "zarr";
    meta["voxelsize"] = 0;
    meta["min"] = 0;
    meta["max"] = 255;

    writeFile(stagingDir / "meta.json", meta.dump(2));

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Metadata complete: %d levels, shape=[%d, %d, %d] delimiter='%s'\n",
                     numLevels, slices, height, width, delimiter.c_str());

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels
    };
}

}  // namespace vc::cache
