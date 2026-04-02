#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/CacheUtils.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <mutex>
#include "utils/Json.hpp"

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

        utils::Json meta;
        try {
            meta = utils::Json::parse_file(zarrayPath);
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
                lm.shape = {s[0].get_int(), s[1].get_int(), s[2].get_int()};
            }
        }
        if (meta.contains("chunks") && meta["chunks"].is_array()) {
            auto& c = meta["chunks"];
            if (c.size() >= 3) {
                lm.chunkShape = {
                    c[0].get_int(), c[1].get_int(), c[2].get_int()};
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

    buildCachedAuth();

#ifdef VC_USE_CURL
    static std::once_flag curlOnce;
    std::call_once(curlOnce, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
#endif
}

void HttpChunkSource::buildCachedAuth()
{
    if (!auth_.awsSigv4) return;
    cachedSigv4_ = "aws:amz:" + auth_.region + ":s3";
    cachedUserpwd_ = auth_.accessKey + ":" + auth_.secretKey;
    if (!auth_.sessionToken.empty())
        cachedTokenHeader_ = "x-amz-security-token: " + auth_.sessionToken;
}

void HttpChunkSource::applyCachedAuth(
    CURL* curl, struct curl_slist*& outHeaders) const
{
    if (!auth_.awsSigv4) return;
    curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, cachedSigv4_.c_str());
    curl_easy_setopt(curl, CURLOPT_USERPWD, cachedUserpwd_.c_str());
    if (!cachedTokenHeader_.empty()) {
        outHeaders = curl_slist_append(outHeaders, cachedTokenHeader_.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, outHeaders);
    }
}

HttpChunkSource::~HttpChunkSource() = default;

void HttpChunkSource::setShardConfig(const ShardConfig& config)
{
    sharded_ = config.enabled;
    shardShape_ = config.shardShape;
    if (sharded_ && !levels_.empty()) {
        // Compute chunks per shard from shard shape and chunk shape of level 0
        auto& cs = levels_[0].chunkShape;
        for (int d = 0; d < 3; d++) {
            chunksPerShard_[d] = (cs[d] > 0 && shardShape_[d] > 0)
                ? shardShape_[d] / cs[d] : 1;
            if (chunksPerShard_[d] < 1) chunksPerShard_[d] = 1;
        }
    }
}

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

std::string HttpChunkSource::shardUrl(const ChunkKey& key) const
{
    int sz = key.iz / chunksPerShard_[0];
    int sy = key.iy / chunksPerShard_[1];
    int sx = key.ix / chunksPerShard_[2];
    std::string url;
    url.reserve(baseUrl_.size() + 32);
    url += baseUrl_;
    url += '/';
    url += std::to_string(key.level);
    url += "/c/";
    url += std::to_string(sz);
    url += '/';
    url += std::to_string(sy);
    url += '/';
    url += std::to_string(sx);
    return url;
}

int HttpChunkSource::innerChunkIndex(const ChunkKey& key) const
{
    int iz = key.iz % chunksPerShard_[0];
    int iy = key.iy % chunksPerShard_[1];
    int ix = key.ix % chunksPerShard_[2];
    return (iz * chunksPerShard_[1] + iy) * chunksPerShard_[2] + ix;
}

int HttpChunkSource::totalChunksPerShard() const
{
    return chunksPerShard_[0] * chunksPerShard_[1] * chunksPerShard_[2];
}

// Helper: perform a full HTTP GET, returning raw bytes.
// Uses the pre-computed auth strings from the HttpChunkSource instance.
//
// The thread-local CURL handle is initialized once with constant options
// (HTTP/2, keep-alive, timeouts, etc.) and never reset. Only per-request
// options (URL, write callback/data, auth headers) are set each call.
// This avoids curl_easy_reset() which, while it preserves the connection
// pool, forces us to re-set ~18 constant options on every request.
#ifdef VC_USE_CURL
static std::vector<uint8_t> httpGetBytes(
    const std::string& url, const HttpChunkSource& src, size_t reserveHint = 2 * 1024 * 1024)
{
    thread_local CURL* curl = [] {
        CURL* c = curl_easy_init();
        if (!c) return c;
        // Constant options — set once, never cleared
        curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        curl_easy_setopt(c, CURLOPT_PIPEWAIT, 1L);
        curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(c, CURLOPT_MAXREDIRS, 5L);
#if CURL_AT_LEAST_VERSION(7, 85, 0)
        curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR, "http,https");
#else
        curl_easy_setopt(c, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
#endif
        curl_easy_setopt(c, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 5L);
        curl_easy_setopt(c, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(c, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(c, CURLOPT_TCP_KEEPIDLE, 30L);
        curl_easy_setopt(c, CURLOPT_TCP_KEEPINTVL, 15L);
        curl_easy_setopt(c, CURLOPT_DNS_CACHE_TIMEOUT, 600L);
        curl_easy_setopt(c, CURLOPT_BUFFERSIZE, 512L * 1024L);
        curl_easy_setopt(c, CURLOPT_TCP_NODELAY, 1L);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curlWriteCallback);
        return c;
    }();
    if (!curl) return {};

    // Per-request options only
    std::vector<uint8_t> response;
    response.reserve(reserveHint);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    struct curl_slist* authHeaders = nullptr;
    src.applyCachedAuth(curl, authHeaders);
    CURLcode res = curl_easy_perform(curl);
    if (authHeaders) {
        curl_slist_free_all(authHeaders);
        // Clear custom headers so they don't leak into next request
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, nullptr);
    }

    if (res != CURLE_OK) {
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

        if (res == CURLE_HTTP_RETURNED_ERROR &&
            (httpCode == 404 || httpCode == 403 || httpCode == 400))
            return {};

        char msg[512];
        std::snprintf(msg, sizeof(msg),
                      "HTTP fetch failed: %s (curl=%d http=%ld) url=%s",
                      curl_easy_strerror(res), static_cast<int>(res),
                      httpCode, url.c_str());
        throw std::runtime_error(msg);
    }
    return response;
}
#endif

std::vector<uint8_t> HttpChunkSource::fetchFromShard(const ChunkKey& key)
{
    std::string url = shardUrl(key);

    // Check shard cache (shared_ptr so multiple threads can read concurrently)
    std::shared_ptr<std::vector<uint8_t>> shardData;
    {
        std::lock_guard<std::mutex> lock(shardCacheMutex_);
        auto it = shardCache_.find(url);
        if (it != shardCache_.end())
            shardData = it->second;
    }

    if (!shardData) {
#ifdef VC_USE_CURL
        auto raw = httpGetBytes(url, *this, 8 * 1024 * 1024);
        if (raw.empty()) return {};
        shardData = std::make_shared<std::vector<uint8_t>>(std::move(raw));

        std::lock_guard<std::mutex> lock(shardCacheMutex_);
        // Evict entire cache if over limit (shards are re-fetchable)
        if (shardCache_.size() >= 16)
            shardCache_.clear();

        // Another thread may have inserted while we fetched — keep first
        auto [it, inserted] = shardCache_.emplace(url, shardData);
        if (!inserted) shardData = it->second;

        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[SHARD] Cached %s (%zu bytes, %d entries, cache=%zu)\n",
                         url.c_str(), shardData->size(), totalChunksPerShard(),
                         shardCache_.size());
#else
        return {};
#endif
    }

    // Parse the binary index at the end of the shard.
    // Format: N entries of [uint64_le offset, uint64_le nbytes], 16 bytes each.
    int nChunks = totalChunksPerShard();
    uint64_t indexSize = static_cast<uint64_t>(nChunks) * 16;
    if (shardData->size() < indexSize) return {};

    int inner = innerChunkIndex(key);
    if (inner < 0 || inner >= nChunks) return {};

    const uint8_t* indexBase = shardData->data() + shardData->size() - indexSize;
    const uint8_t* entry = indexBase + inner * 16;

    uint64_t offset = 0, nbytes = 0;
    std::memcpy(&offset, entry, 8);
    std::memcpy(&nbytes, entry + 8, 8);

    // Empty sentinel: all-ones means chunk not present
    if (offset == UINT64_MAX && nbytes == UINT64_MAX) return {};

    if (offset + nbytes > shardData->size() - indexSize) return {};

    return {shardData->data() + offset, shardData->data() + offset + nbytes};
}

std::vector<uint8_t> HttpChunkSource::fetch(const ChunkKey& key)
{
    if (sharded_)
        return fetchFromShard(key);

#ifdef VC_USE_CURL
    return httpGetBytes(chunkUrl(key), *this);
#else
    (void)key;
    return {};
#endif
}

int HttpChunkSource::numLevels() const { return levelsNumLevels(levels_); }
std::array<int, 3> HttpChunkSource::chunkShape(int level) const { return levelsChunkShape(levels_, level); }
std::array<int, 3> HttpChunkSource::levelShape(int level) const { return levelsLevelShape(levels_, level); }

}  // namespace vc::cache
