#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include "utils/Json.hpp"

#include <utils/hash.hpp>

#ifdef VC_USE_CURL
#include <curl/curl.h>
#endif

namespace vc::cache {

#ifdef VC_USE_CURL
struct CurlDeleter { void operator()(CURL* c) const { if (c) curl_easy_cleanup(c); } };
using CurlHandle = std::unique_ptr<CURL, CurlDeleter>;
#endif

static constexpr const char* kRemoteSourceFile = ".remote_source.json";

// ---- curl helpers -----------------------------------------------------------

#ifdef VC_USE_CURL
static size_t stringWriteCallback(
    char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* str = static_cast<std::string*>(userdata);
    str->append(ptr, size * nmemb);
    return size * nmemb;
}

// Set common CURL options after curl_easy_reset. Call this once per request.
static void configureCurlDefaults(CURL* c)
{
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
#if CURL_AT_LEAST_VERSION(7, 85, 0)
    curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR, "http,https");
#else
    curl_easy_setopt(c, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
#endif
}
#endif

std::string httpGetString(const std::string& url, const HttpAuth& auth)
{
#ifdef VC_USE_CURL
    // Thread-local CURL handle: reuses TCP+TLS connections, cleaned up on thread exit
    thread_local CurlHandle curlOwner = [] {
        CurlHandle p(curl_easy_init());
        if (p) configureCurlDefaults(p.get());
        return p;
    }();
    CURL* curl = curlOwner.get();
    if (!curl) return {};

    // Reset clears all options but keeps the connection alive
    curl_easy_reset(curl);
    configureCurlDefaults(curl);

    std::string response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stringWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    // Don't use FAILONERROR — we check HTTP codes ourselves to distinguish
    // 403 (auth error) from 404 (not found)

    auto authGuard = applyCurlAuth(curl, auth);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::fprintf(stderr, "[httpGetString] curl error %d for %s: %s\n",
                     res, url.c_str(), curl_easy_strerror(res));
        return {};
    }

    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    if (httpCode >= 400) {
        // Check for S3 auth/token errors in response body
        // S3 returns 400 for expired tokens, 403 for access denied
        bool isAuthError = (httpCode == 401 || httpCode == 403);
        if (!isAuthError && !response.empty()) {
            // S3 returns XML errors like <Code>ExpiredToken</Code>
            isAuthError = response.find("ExpiredToken") != std::string::npos ||
                          response.find("AccessDenied") != std::string::npos ||
                          response.find("InvalidAccessKeyId") != std::string::npos ||
                          response.find("SignatureDoesNotMatch") != std::string::npos ||
                          response.find("TokenRefreshRequired") != std::string::npos ||
                          response.find("InvalidToken") != std::string::npos;
        }

        if (isAuthError) {
            // Try to extract the S3 error message
            std::string errMsg = "Access denied (HTTP " + std::to_string(httpCode) + ")";
            auto msgStart = response.find("<Message>");
            auto msgEnd = response.find("</Message>");
            if (msgStart != std::string::npos && msgEnd != std::string::npos) {
                msgStart += 9;  // strlen("<Message>")
                errMsg = response.substr(msgStart, msgEnd - msgStart) +
                         " (HTTP " + std::to_string(httpCode) + ")";
            }
            throw std::runtime_error(errMsg + ". Check your AWS credentials.");
        }

        // 5xx — server error, throw so callers know it's transient
        if (httpCode >= 500) {
            throw std::runtime_error(
                "HTTP server error " + std::to_string(httpCode) + " fetching " + url);
        }

        // 404, 400, etc. — treat as "not found"
        return {};
    }

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

    // Build ListObjectsV2 URL — use a temporary curl handle for URL encoding only
    CURL* tmpCurl = curl_easy_init();
    std::string listUrl = bucketHost + "/?list-type=2&delimiter=/";
    if (!prefix.empty() && tmpCurl) {
        char* encoded = curl_easy_escape(
            tmpCurl, prefix.c_str(), static_cast<int>(prefix.size()));
        listUrl += "&prefix=";
        listUrl += encoded;
        curl_free(encoded);
    } else if (!prefix.empty()) {
        listUrl += "&prefix=" + prefix;
    }
    if (tmpCurl) curl_easy_cleanup(tmpCurl);

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[S3] ListObjects: %s\n", listUrl.c_str());

    // Thread-local CURL handle without FAILONERROR so we can log the status
    thread_local CurlHandle curlOwner = [] {
        CurlHandle p(curl_easy_init());
        if (p) configureCurlDefaults(p.get());
        return p;
    }();
    CURL* curl = curlOwner.get();

    std::string xml;
    {
        if (!curl) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[S3] Failed to init curl\n");
            return result;
        }

        // Reset clears all options but keeps the connection alive
        curl_easy_reset(curl);
        configureCurlDefaults(curl);

        curl_easy_setopt(curl, CURLOPT_URL, listUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stringWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &xml);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

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
                        code == "TokenRefreshRequired" || code == "InvalidToken") {
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

    // Thread-local CURL handle: reuses TCP+TLS connections, cleaned up on thread exit
    thread_local CurlHandle curlOwner = [] {
        CurlHandle p(curl_easy_init());
        if (p) configureCurlDefaults(p.get());
        return p;
    }();
    CURL* curl = curlOwner.get();
    if (!curl) {
        std::fclose(fp);
        return false;
    }

    // Reset clears all options but keeps the connection alive
    curl_easy_reset(curl);
    configureCurlDefaults(curl);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fileWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
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

// ---- zarr v3 helpers --------------------------------------------------------

// Convert a zarr v3 zarr.json into a zarr v2-compatible .zarray JSON string.
// This lets the rest of the codebase (VcDataset, openZarrLevels) consume v3
// volumes without any changes — they only ever see synthesized v2 metadata.
static std::string synthesize_v2_metadata(const utils::Json& v3)
{
    utils::Json v2;
    v2["zarr_format"] = 2;

    // Shape
    v2["shape"] = v3["shape"];

    // Chunk shape: v3 uses chunk_grid.configuration.chunk_shape
    // For sharded arrays, this is the SHARD shape; the inner chunk shape
    // is in the sharding_indexed codec. We need the inner chunk shape for v2.
    utils::Json chunkShape;
    if (v3.contains("chunk_grid") && v3["chunk_grid"].contains("configuration")) {
        auto gridCfg = v3["chunk_grid"]["configuration"];
        if (gridCfg.contains("chunk_shape"))
            chunkShape = gridCfg["chunk_shape"];
    }
    // Check if sharding_indexed codec overrides the chunk shape
    if (v3.contains("codecs") && v3["codecs"].is_array()) {
        for (auto& codec : v3["codecs"]) {
            if (codec.value("name", std::string("")) == "sharding_indexed") {
                if (codec.contains("configuration")) {
                    auto shardCfg = codec["configuration"];
                    if (shardCfg.contains("chunk_shape"))
                        chunkShape = shardCfg["chunk_shape"];
                }
                break;
            }
        }
    }
    v2["chunks"] = chunkShape;

    // Data type: v3 uses string like "uint8", "uint16"; map to v2 dtype codes
    std::string dtype = v3["data_type"].get_string();
    if (dtype == "uint8")       v2["dtype"] = "|u1";
    else if (dtype == "uint16") v2["dtype"] = "<u2";
    else                        v2["dtype"] = dtype;  // pass through, VcDataset will validate

    // Fill value
    v2["fill_value"] = v3.value("fill_value", 0);

    // Order
    v2["order"] = "C";

    // Dimension separator: v3 uses "/" by default
    v2["dimension_separator"] = "/";

    // Compressor: map v3 codecs array to v2 compressor
    v2["compressor"] = nullptr;
    if (v3.contains("codecs") && v3["codecs"].is_array()) {
        for (auto& codec : v3["codecs"]) {
            std::string name = codec.value("name", std::string(""));
            if (name == "blosc") {
                auto& cfg = codec["configuration"];
                utils::Json comp;
                comp["id"] = "blosc";
                comp["cname"] = cfg.value("cname", std::string("lz4"));
                comp["clevel"] = cfg.value("clevel", 5);
                comp["shuffle"] = cfg.value("shuffle", 1);
                if (cfg.contains("typesize"))
                    comp["typesize"] = cfg["typesize"];
                if (cfg.contains("blocksize"))
                    comp["blocksize"] = cfg["blocksize"];
                v2["compressor"] = comp;
                break;
            } else if (name == "zstd") {
                utils::Json comp;
                comp["id"] = "zstd";
                comp["level"] = codec["configuration"].value("level", 3);
                v2["compressor"] = comp;
                break;
            } else if (name == "gzip" || name == "zlib") {
                utils::Json comp;
                comp["id"] = "gzip";
                comp["level"] = codec["configuration"].value("level", 5);
                v2["compressor"] = comp;
                break;
            } else if (name == "lz4") {
                utils::Json comp;
                comp["id"] = "lz4";
                comp["acceleration"] = codec["configuration"].value("acceleration", 1);
                v2["compressor"] = comp;
                break;
            }
            // Unknown codecs (e.g. "vc3d_h265") — leave compressor null.
            // The VcDecompressor handles these via magic-header detection.
        }
    }

    v2["filters"] = nullptr;

    return v2.dump(2);
}

// Parse shard configuration from zarr v3 metadata.
// Checks both storage_transformers (older v3 draft) and codecs (current v3 spec).
// Returns a ShardConfig with enabled=true if sharding is found.
static ShardConfig parse_v3_shard_config(
    const utils::Json& v3,
    const std::array<int, 3>& chunkShape)
{
    ShardConfig config;

    // Helper to parse shard config from a sharding codec/transformer config
    auto tryParseShard = [&](const utils::Json& cfg) -> bool {
        if (cfg.contains("chunks_per_shard") && cfg["chunks_per_shard"].is_array() &&
            cfg["chunks_per_shard"].size() >= 3) {
            int cz = cfg["chunks_per_shard"][0].get_int();
            int cy = cfg["chunks_per_shard"][1].get_int();
            int cx = cfg["chunks_per_shard"][2].get_int();
            config.enabled = true;
            config.shardShape = {
                cz * chunkShape[0],
                cy * chunkShape[1],
                cx * chunkShape[2]
            };
            return true;
        }
        // Standard v3: chunk_grid gives shard shape, inner chunk_shape is in codec config
        if (cfg.contains("chunk_shape") && cfg["chunk_shape"].is_array() &&
            cfg["chunk_shape"].size() >= 3) {
            // chunk_grid.chunk_shape = shard extents, codec.chunk_shape = inner chunk extents
            // Compute shard shape from chunk_grid if available
            if (v3.contains("chunk_grid") && v3["chunk_grid"].contains("configuration")) {
                auto gridCfg = v3["chunk_grid"]["configuration"];
                if (gridCfg.contains("chunk_shape") && gridCfg["chunk_shape"].is_array()) {
                    config.enabled = true;
                    config.shardShape = {
                        gridCfg["chunk_shape"][0].get_int(),
                        gridCfg["chunk_shape"][1].get_int(),
                        gridCfg["chunk_shape"][2].get_int()
                    };
                    return true;
                }
            }
        }
        return false;
    };

    // Check storage_transformers (older v3 draft format)
    if (v3.contains("storage_transformers") && v3["storage_transformers"].is_array()) {
        for (auto& t : v3["storage_transformers"]) {
            std::string name = t.value("name", std::string(""));
            if (name == "chunk_manifest_sharding" && t.contains("configuration")) {
                if (tryParseShard(t["configuration"])) return config;
            }
        }
    }

    // Check codecs array (current v3 spec: sharding_indexed codec)
    if (v3.contains("codecs") && v3["codecs"].is_array()) {
        for (auto& codec : v3["codecs"]) {
            std::string name = codec.value("name", std::string(""));
            if (name == "sharding_indexed" && codec.contains("configuration")) {
                if (tryParseShard(codec["configuration"])) return config;
            }
        }
    }

    return config;
}

// ---- metadata fetcher -------------------------------------------------------

std::string normalizeRemoteUrl(const std::string& url)
{
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    return u;
}

static std::string deriveRemoteVolumeName(const std::string& url)
{
    std::string u = normalizeRemoteUrl(url);
    auto pos = u.rfind('/');
    if (pos != std::string::npos) {
        return u.substr(pos + 1);
    }
    return u;
}

std::string deriveRemoteVolumeId(const std::string& url)
{
    const auto normalized = normalizeRemoteUrl(url);
    const auto name = deriveRemoteVolumeName(normalized);
    const auto hash = utils::fnv1a(std::string_view(normalized));

    std::ostringstream oss;
    oss << name << "-" << std::hex << std::nouppercase << std::setw(16)
        << std::setfill('0') << hash;
    return oss.str();
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

static std::optional<std::string> readRemoteSourceMarker(
    const std::filesystem::path& stagingDir)
{
    auto markerJson = readFile(stagingDir / kRemoteSourceFile);
    if (markerJson.empty()) {
        return std::nullopt;
    }

    try {
        auto marker = utils::Json::parse(markerJson);
        if (!marker.contains("url") || !marker["url"].is_string()) {
            return std::nullopt;
        }
        return normalizeRemoteUrl(marker["url"].get_string());
    } catch (...) {
        return std::nullopt;
    }
}

static void writeRemoteSourceMarker(
    const std::filesystem::path& stagingDir,
    const std::string& baseUrl)
{
    utils::Json marker;
    marker["url"] = baseUrl;
    writeFile(stagingDir / kRemoteSourceFile, marker.dump(2));
}

// Check if staging dir already has valid cached metadata (meta.json + at least 0/.zarray).
// If so, return the info without hitting the network.
static std::optional<RemoteZarrInfo> tryLoadCachedMetadata(
    const std::string& baseUrl,
    const std::filesystem::path& stagingDir)
{
    namespace fs = std::filesystem;

    if (auto markerUrl = readRemoteSourceMarker(stagingDir)) {
        if (*markerUrl != baseUrl) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log,
                             "[REMOTE] Rejecting cached metadata from %s: marker URL %s does not match %s\n",
                             stagingDir.c_str(), markerUrl->c_str(), baseUrl.c_str());
            return std::nullopt;
        }
    }

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
            auto j = utils::Json::parse(zarray0);
            if (j.contains("dimension_separator")) {
                delimiter = j["dimension_separator"].get_string();
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
    const std::string baseUrl = normalizeRemoteUrl(url);

    const std::string volumeName = deriveRemoteVolumeName(baseUrl);
    const std::string volumeId = deriveRemoteVolumeId(baseUrl);
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
        writeRemoteSourceMarker(stagingDir, baseUrl);
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
    //
    // Strategy: try zarr v2 (.zarray) first. If that yields zero levels,
    // fall back to zarr v3 (zarr.json). This keeps v2 as the fast path.
    std::string delimiter = ".";
    int numLevels = 0;
    utils::Json level0Meta;
    ShardConfig shardConfig;
    bool isV3 = false;

    constexpr int kBatchSize = 8;
    constexpr int kMaxLevels = 20;

    // Probe levels in batches, calling onLevel for each response.
    // Returns number of levels found. Stops at first gap (levels must be contiguous).
    // urlSuffix: appended to baseUrl/<level>/ (e.g. ".zarray" or "zarr.json")
    auto probeLevels = [&](const char* urlSuffix, auto&& onLevel) -> int {
        int found = 0;
        for (int batchStart = 0; batchStart < kMaxLevels; batchStart += kBatchSize) {
            int batchEnd = std::min(batchStart + kBatchSize, kMaxLevels);

            std::vector<std::future<std::string>> futures;
            futures.reserve(batchEnd - batchStart);
            for (int lvl = batchStart; lvl < batchEnd; lvl++) {
                std::string lvlUrl = baseUrl + "/" + std::to_string(lvl) + "/" + urlSuffix;
                futures.push_back(std::async(std::launch::async,
                    [lvlUrl, &auth]() { return httpGetString(lvlUrl, auth); }));
            }

            bool batchHadMiss = false;
            for (int i = 0; i < static_cast<int>(futures.size()); i++) {
                int lvl = batchStart + i;
                auto response = futures[i].get();

                if (response.empty()) {
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "[REMOTE] Level %d: no %s\n", lvl, urlSuffix);
                    batchHadMiss = true;
                    break;
                }

                if (!onLevel(lvl, std::move(response))) {
                    batchHadMiss = true;
                    break;
                }
                found++;
            }

            if (batchHadMiss) break;
        }
        return found;
    };

    // --- Phase 1: try zarr v2 (.zarray) ---
    numLevels = probeLevels(".zarray", [&](int lvl, std::string zarray) -> bool {
        auto levelDir = stagingDir / std::to_string(lvl);
        writeFile(levelDir / ".zarray", zarray);

        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[REMOTE] Level %d: fetched .zarray (%zu bytes)\n",
                         lvl, zarray.size());

        if (lvl == 0) {
            try {
                level0Meta = utils::Json::parse(zarray);
                if (level0Meta.contains("dimension_separator")) {
                    delimiter = level0Meta["dimension_separator"].get_string();
                }
            } catch (const std::exception& e) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Warning: failed to parse level 0 .zarray: %s\n", e.what());
            }
        }
        return true;
    });

    // --- Phase 2: if v2 found nothing, try zarr v3 (zarr.json) ---
    if (numLevels == 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[REMOTE] No zarr v2 levels found, trying zarr v3...\n");

        numLevels = probeLevels("zarr.json", [&](int lvl, std::string zarrJson) -> bool {
            utils::Json v3;
            try {
                v3 = utils::Json::parse(zarrJson);
            } catch (const std::exception& e) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Level %d: failed to parse zarr.json: %s\n", lvl, e.what());
                return false;
            }

            if (v3.value("zarr_format", 0) != 3 ||
                v3.value("node_type", std::string("")) != "array") {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Level %d: zarr.json is not zarr v3 array\n", lvl);
                return false;
            }

            std::string synthesized = synthesize_v2_metadata(v3);
            auto levelDir = stagingDir / std::to_string(lvl);
            writeFile(levelDir / ".zarray", synthesized);
            isV3 = true;

            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[REMOTE] Level %d: fetched zarr.json (v3), synthesized .zarray (%zu bytes)\n",
                             lvl, synthesized.size());

            if (lvl == 0) {
                try {
                    level0Meta = utils::Json::parse(synthesized);
                    delimiter = "/";
                    if (level0Meta.contains("dimension_separator")) {
                        delimiter = level0Meta["dimension_separator"].get_string();
                    }

                    std::array<int, 3> chunkShape = {128, 128, 128};
                    if (v3.contains("chunk_grid")) {
                        auto grid = v3["chunk_grid"];
                        if (grid.contains("configuration")) {
                            auto gridCfg = grid["configuration"];
                            if (gridCfg.contains("chunk_shape") && gridCfg["chunk_shape"].is_array()) {
                                auto cs = gridCfg["chunk_shape"];
                                chunkShape = {cs[0].get_int(), cs[1].get_int(), cs[2].get_int()};
                            }
                        }
                    }
                    shardConfig = parse_v3_shard_config(v3, chunkShape);

                    if (shardConfig.enabled) {
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "[REMOTE] Shard config: shape=[%d, %d, %d]\n",
                                         shardConfig.shardShape[0],
                                         shardConfig.shardShape[1],
                                         shardConfig.shardShape[2]);
                    }
                } catch (const std::exception& e) {
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "[REMOTE] Warning: failed to parse level 0 synthesized .zarray: %s\n", e.what());
                }
            }
            return true;
        });
    }

    if (numLevels == 0) {
        throw std::runtime_error("No zarr levels found at " + baseUrl +
                                 " (tried both v2 .zarray and v3 zarr.json)");
    }

    // Synthesize meta.json from level 0 shape
    int width = 0, height = 0, slices = 0;
    if (level0Meta.contains("shape") && level0Meta["shape"].is_array() &&
        level0Meta["shape"].size() >= 3) {
        // zarr shape is [z, y, x]
        slices = level0Meta["shape"][0].get_int();
        height = level0Meta["shape"][1].get_int();
        width  = level0Meta["shape"][2].get_int();
    }

    utils::Json meta;
    meta["uuid"] = volumeId;
    meta["name"] = volumeName;
    meta["type"] = "vol";
    meta["width"] = width;
    meta["height"] = height;
    meta["slices"] = slices;
    meta["format"] = "zarr";
    meta["voxelsize"] = 0;
    meta["min"] = 0;
    meta["max"] = 255;

    writeFile(stagingDir / "meta.json", meta.dump(2));
    writeRemoteSourceMarker(stagingDir, baseUrl);

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Metadata complete: %d levels, shape=[%d, %d, %d] delimiter='%s'%s\n",
                     numLevels, slices, height, width, delimiter.c_str(),
                     isV3 ? " (zarr v3)" : "");

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels,
        .shardConfig = shardConfig
    };
}

}  // namespace vc::cache
