#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/CacheUtils.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>

namespace vc::cache {

// --- Shared metadata helpers for both FileSystem and Http chunk sources ---

using LevelMeta = FileSystemChunkSource::LevelMeta;

static int levelsNumLevels(const std::vector<LevelMeta>& levels)
{
    return static_cast<int>(levels.size());
}

static std::array<int, 3> levelsChunkShape(
    const std::vector<LevelMeta>& levels, int level)
{
    if (level < 0 || level >= static_cast<int>(levels.size()))
        return {0, 0, 0};
    return levels[level].chunkShape;
}

static std::array<int, 3> levelsLevelShape(
    const std::vector<LevelMeta>& levels, int level)
{
    if (level < 0 || level >= static_cast<int>(levels.size()))
        return {0, 0, 0};
    return levels[level].shape;
}

// =============================================================================
// FileSystemChunkSource
// =============================================================================

FileSystemChunkSource::FileSystemChunkSource(
    const std::filesystem::path& zarrRoot,
    const std::string& delimiter)
    : root_(zarrRoot), delimiter_(delimiter)
{
    discoverLevels();
}

FileSystemChunkSource::FileSystemChunkSource(
    const std::filesystem::path& zarrRoot,
    const std::string& delimiter,
    std::vector<LevelMeta> levels)
    : root_(zarrRoot), delimiter_(delimiter), levels_(std::move(levels))
{
}

void FileSystemChunkSource::discoverLevels()
{
    // Discover pyramid levels by scanning for numbered subdirectories
    // with .zarray metadata files.
    std::vector<int> levelNums;
    for (auto& entry : std::filesystem::directory_iterator(root_)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        // Check if directory name is a number
        bool isNum = !name.empty() &&
                     std::all_of(name.begin(), name.end(), ::isdigit);
        if (isNum) {
            levelNums.push_back(std::stoi(name));
        }
    }
    std::sort(levelNums.begin(), levelNums.end());

    levels_.clear();
    for (int lvl : levelNums) {
        auto zarrayPath = root_ / std::to_string(lvl) / ".zarray";
        if (!std::filesystem::exists(zarrayPath)) continue;

        std::ifstream f(zarrayPath);
        if (!f.is_open()) continue;

        nlohmann::json meta;
        try {
            f >> meta;
        } catch (const std::exception& e) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[CHUNK_SOURCE] Warning: failed to parse %s: %s\n",
                             zarrayPath.c_str(), e.what());
            continue;
        }

        LevelMeta lm{};
        if (meta.contains("shape") && meta["shape"].is_array()) {
            auto& s = meta["shape"];
            if (s.size() >= 3) {
                lm.shape = {s[0].get<int>(), s[1].get<int>(), s[2].get<int>()};
            }
        }
        if (meta.contains("chunks") && meta["chunks"].is_array()) {
            auto& c = meta["chunks"];
            if (c.size() >= 3) {
                lm.chunkShape = {
                    c[0].get<int>(), c[1].get<int>(), c[2].get<int>()};
            }
        }
        levels_.push_back(lm);
    }
}

std::filesystem::path FileSystemChunkSource::chunkPath(
    const ChunkKey& key) const
{
    return root_ / std::to_string(key.level) / chunkFilename(key, delimiter_);
}

std::vector<uint8_t> FileSystemChunkSource::fetch(const ChunkKey& key)
{
    auto result = readFileToVector(chunkPath(key));
    return result ? std::move(*result) : std::vector<uint8_t>{};
}

int FileSystemChunkSource::numLevels() const { return levelsNumLevels(levels_); }
std::array<int, 3> FileSystemChunkSource::chunkShape(int level) const { return levelsChunkShape(levels_, level); }
std::array<int, 3> FileSystemChunkSource::levelShape(int level) const { return levelsLevelShape(levels_, level); }

// =============================================================================
// HttpChunkSource
// =============================================================================

#ifdef VC_USE_CURL
#include <curl/curl.h>

static size_t curlWriteCallback(
    char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* vec = static_cast<std::vector<uint8_t>*>(userdata);
    size_t bytes = size * nmemb;
    vec->insert(vec->end(), ptr, ptr + bytes);
    return bytes;
}
#endif

HttpChunkSource::HttpChunkSource(
    const std::string& baseUrl,
    const std::string& delimiter,
    std::vector<LevelMeta> levels,
    HttpAuth auth)
    : baseUrl_(baseUrl), delimiter_(delimiter), levels_(std::move(levels)), auth_(std::move(auth))
{
    // Remove trailing slash from base URL
    while (!baseUrl_.empty() && baseUrl_.back() == '/') {
        baseUrl_.pop_back();
    }

#ifdef VC_USE_CURL
    static std::once_flag curlOnce;
    std::call_once(curlOnce, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
#endif
}

HttpChunkSource::~HttpChunkSource() = default;

std::string HttpChunkSource::chunkUrl(const ChunkKey& key) const
{
    std::string url;
    url.reserve(baseUrl_.size() + 32);
    url += baseUrl_;
    url += '/';
    url += std::to_string(key.level);
    url += '/';
    url += std::to_string(key.iz);
    url += delimiter_;
    url += std::to_string(key.iy);
    url += delimiter_;
    url += std::to_string(key.ix);
    return url;
}

std::vector<uint8_t> HttpChunkSource::fetch(const ChunkKey& key)
{
#ifdef VC_USE_CURL
    // Thread-local CURL handle: reuses TCP+TLS connections across requests
    // on the same IOPool worker thread.
    thread_local CURL* curl = [] {
        CURL* c = curl_easy_init();
        if (c) {
            // One-time options that survive curl_easy_reset()
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
            // Enable HTTP/2 — allows multiplexing over shared connections
            curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        }
        return c;
    }();
    if (!curl) return {};

    // Reset clears per-request state but keeps the connection alive
    curl_easy_reset(curl);
    // curl_easy_reset clears all options — re-apply protocol preferences
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

    std::string url = chunkUrl(key);
    std::vector<uint8_t> response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
#if CURL_AT_LEAST_VERSION(7, 85, 0)
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
#else
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
#endif
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    auto authGuard = applyCurlAuth(curl, auth_);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

        if (res == CURLE_HTTP_RETURNED_ERROR &&
            (httpCode == 404 || httpCode == 403 || httpCode == 400)) {
            return {};  // chunk doesn't exist at source
        }

        // Transient error — throw so IOPool skips negative caching.
        // Include URL and HTTP code for diagnostics.
        char msg[512];
        std::snprintf(msg, sizeof(msg),
                      "HTTP fetch failed: %s (curl=%d http=%ld) url=%s",
                      curl_easy_strerror(res), static_cast<int>(res),
                      httpCode, url.c_str());
        throw std::runtime_error(msg);
    }
    return response;
#else
    (void)key;
    return {};
#endif
}

int HttpChunkSource::numLevels() const { return levelsNumLevels(levels_); }
std::array<int, 3> HttpChunkSource::chunkShape(int level) const { return levelsChunkShape(levels_, level); }
std::array<int, 3> HttpChunkSource::levelShape(int level) const { return levelsLevelShape(levels_, level); }

}  // namespace vc::cache
