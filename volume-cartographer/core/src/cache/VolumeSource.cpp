#include "vc/core/cache/VolumeSource.hpp"
#include "vc/core/cache/BlockCache.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/CacheUtils.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include "utils/Json.hpp"
#include <utils/http_fetch.hpp>
#include <utils/zarr.hpp>

namespace vc::cache {

// --- Shared metadata helpers ---

using LevelMeta = FileSystemSource::LevelMeta;

static int levelsNumLevels(const std::vector<LevelMeta>& levels) noexcept
{
    return static_cast<int>(levels.size());
}

static std::array<int, 3> levelsChunkShape(
    const std::vector<LevelMeta>& levels, int level) noexcept
{
    if (level < 0 || level >= static_cast<int>(levels.size()))
        return {0, 0, 0};
    return levels[level].chunkShape;
}

static std::array<int, 3> levelsLevelShape(
    const std::vector<LevelMeta>& levels, int level) noexcept
{
    if (level < 0 || level >= static_cast<int>(levels.size()))
        return {0, 0, 0};
    return levels[level].shape;
}

// =============================================================================
// FileSystemSource
// =============================================================================

FileSystemSource::FileSystemSource(
    const std::filesystem::path& zarrRoot,
    const std::string& delimiter)
    : root_(zarrRoot), delimiter_(delimiter)
{
    discoverLevels();
}

FileSystemSource::FileSystemSource(
    const std::filesystem::path& zarrRoot,
    const std::string& delimiter,
    std::vector<LevelMeta> levels)
    : root_(zarrRoot), delimiter_(delimiter), levels_(std::move(levels))
{
}

void FileSystemSource::discoverLevels()
{
    std::vector<int> levelNums;
    for (auto& entry : std::filesystem::directory_iterator(root_)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        bool isNum = !name.empty() &&
                     std::all_of(name.begin(), name.end(), ::isdigit);
        if (isNum)
            levelNums.push_back(std::stoi(name));
    }
    std::sort(levelNums.begin(), levelNums.end());

    levels_.clear();
    for (int lvl : levelNums) {
        auto levelPath = root_ / std::to_string(lvl);
        try {
            auto meta = utils::ZarrArray::open(levelPath).metadata();
            LevelMeta lm{};
            lm.dirName = std::to_string(lvl);
            // Finest granularity: inner chunks for sharded v3, chunks otherwise.
            const auto& cs = meta.shard_config ? meta.shard_config->sub_chunks
                                               : meta.chunks;
            if (meta.shape.size() >= 3)
                lm.shape = {int(meta.shape[0]), int(meta.shape[1]), int(meta.shape[2])};
            if (cs.size() >= 3)
                lm.chunkShape = {int(cs[0]), int(cs[1]), int(cs[2])};
            // 16³ blocks are the fixed storage unit; chunks must tile cleanly.
            // Arbitrary multiples of 16 on each axis are fine (128³, 64³,
            // 192³, non-cubic 32x128x128, etc.) — just not 100, 50, 96 etc.
            for (int d = 0; d < 3; ++d) {
                if (lm.chunkShape[d] <= 0 || lm.chunkShape[d] % kBlockSize != 0) {
                    throw std::runtime_error(
                        "zarr level " + std::to_string(lvl) + " at " +
                        levelPath.string() + " has chunk shape " +
                        std::to_string(lm.chunkShape[0]) + "x" +
                        std::to_string(lm.chunkShape[1]) + "x" +
                        std::to_string(lm.chunkShape[2]) +
                        "; each axis must be a positive multiple of " +
                        std::to_string(kBlockSize));
                }
            }
            levels_.push_back(lm);
        } catch (const std::exception& e) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[CHUNK_SOURCE] Warning: failed to open %s: %s\n",
                             levelPath.c_str(), e.what());
            continue;
        }
    }
}

std::filesystem::path FileSystemSource::chunkPath(const ChunkKey& key) const
{
    const auto& dir = (key.level >= 0 && key.level < int(levels_.size()) && !levels_[key.level].dirName.empty())
        ? levels_[key.level].dirName
        : std::to_string(key.level);
    return root_ / dir / chunkFilename(key, delimiter_);
}

FetchResult FileSystemSource::fetch(const ChunkKey& key)
{
    FetchResult out;
    auto result = readFileToVector(chunkPath(key));
    if (result) {
        out.bytes = std::move(*result);
    } else {
        out.wasAbsent = true;
    }
    return out;
}

int FileSystemSource::numLevels() const noexcept { return levelsNumLevels(levels_); }
std::array<int, 3> FileSystemSource::chunkShape(int level) const noexcept { return levelsChunkShape(levels_, level); }
std::array<int, 3> FileSystemSource::levelShape(int level) const noexcept { return levelsLevelShape(levels_, level); }

// =============================================================================
// HttpSource
// =============================================================================

HttpSource::HttpSource(
    const std::string& baseUrl,
    const std::string& delimiter,
    std::vector<LevelMeta> levels,
    HttpAuth auth)
    : baseUrl_(baseUrl), delimiter_(delimiter), levels_(std::move(levels))
{
    while (!baseUrl_.empty() && baseUrl_.back() == '/')
        baseUrl_.pop_back();

    utils::HttpClient::Config cfg;
    cfg.aws_auth = std::move(auth);
    cfg.transfer_timeout = std::chrono::seconds{60};
    cfg.connect_timeout = std::chrono::seconds{5};
    cfg.max_retries = 2;
    client_ = std::make_shared<utils::HttpClient>(std::move(cfg));
}

HttpSource::~HttpSource() = default;

void HttpSource::setShardConfig(const ShardConfig& config)
{
    sharded_ = config.enabled;
    shardShape_ = config.shardShape;
    if (sharded_ && !levels_.empty()) {
        auto& cs = levels_[0].chunkShape;
        for (int d = 0; d < 3; d++) {
            chunksPerShard_[d] = (cs[d] > 0 && shardShape_[d] > 0)
                ? shardShape_[d] / cs[d] : 1;
            if (chunksPerShard_[d] < 1) chunksPerShard_[d] = 1;
        }
    }
}

std::string HttpSource::chunkUrl(const ChunkKey& key) const
{
    const auto& dir = (key.level >= 0 && key.level < int(levels_.size()) && !levels_[key.level].dirName.empty())
        ? levels_[key.level].dirName
        : std::to_string(key.level);
    std::string url;
    url.reserve(baseUrl_.size() + 32);
    url += baseUrl_;
    url += '/';
    url += dir;
    url += '/';
    url += std::to_string(key.iz);
    url += delimiter_;
    url += std::to_string(key.iy);
    url += delimiter_;
    url += std::to_string(key.ix);
    return url;
}

std::string HttpSource::shardUrl(const ChunkKey& key) const
{
    int sz = key.iz / chunksPerShard_[0];
    int sy = key.iy / chunksPerShard_[1];
    int sx = key.ix / chunksPerShard_[2];
    const auto& dir = (key.level >= 0 && key.level < int(levels_.size()) && !levels_[key.level].dirName.empty())
        ? levels_[key.level].dirName
        : std::to_string(key.level);
    std::string url;
    url.reserve(baseUrl_.size() + 32);
    url += baseUrl_;
    url += '/';
    url += dir;
    url += "/c/";
    url += std::to_string(sz);
    url += '/';
    url += std::to_string(sy);
    url += '/';
    url += std::to_string(sx);
    return url;
}

int HttpSource::innerChunkIndex(const ChunkKey& key) const noexcept
{
    int iz = key.iz % chunksPerShard_[0];
    int iy = key.iy % chunksPerShard_[1];
    int ix = key.ix % chunksPerShard_[2];
    return (iz * chunksPerShard_[1] + iy) * chunksPerShard_[2] + ix;
}

int HttpSource::totalChunksPerShard() const noexcept
{
    return chunksPerShard_[0] * chunksPerShard_[1] * chunksPerShard_[2];
}

FetchResult HttpSource::httpGet(const std::string& url)
{
    FetchResult out;
    auto resp = client_->get(url);
    out.wasAbsent = (resp.status_code == 404);
    if (resp.status_code == 404) {
        std::fprintf(stderr, "[HTTP 404] GET %s\n", url.c_str());
    }
    if (!resp.ok()) {
        if (resp.status_code != 404) {
            out.transientError = true;
            transientError_.store(true, std::memory_order_relaxed);
            static std::atomic<int> errCount{0};
            int n = errCount.fetch_add(1);
            if (n < 5) {
                std::fprintf(stderr, "[HTTP] GET %s -> status=%ld (%s)\n",
                             url.c_str(),
                             long(resp.status_code),
                             resp.body.empty()
                                 ? "(no body)"
                                 : std::string(
                                     reinterpret_cast<const char*>(resp.body.data()),
                                     std::min(resp.body.size(), size_t(200))).c_str());
            }
        }
        return out;
    }

    transientError_.store(false, std::memory_order_relaxed);
    out.bytes.resize(resp.body.size());
    if (!out.bytes.empty())
        std::memcpy(out.bytes.data(), resp.body.data(), out.bytes.size());
    return out;
}

FetchResult HttpSource::httpGetRange(const std::string& url,
                                     std::size_t offset,
                                     std::size_t length)
{
    FetchResult out;
    if (length == 0) return out;
    auto resp = client_->get_range(url, offset, length);
    out.wasAbsent = (resp.status_code == 404);
    if (resp.status_code == 404) {
        std::fprintf(stderr, "[HTTP 404] GET_RANGE %s [%zu..%zu)\n",
            url.c_str(), offset, offset + length);
    }
    if (!resp.ok()) {
        if (resp.status_code != 404) {
            out.transientError = true;
            transientError_.store(true, std::memory_order_relaxed);
            static std::atomic<int> errCount{0};
            int n = errCount.fetch_add(1);
            if (n < 5) {
                std::fprintf(stderr,
                             "[HTTP] GET_RANGE %s [%zu..%zu) -> status=%ld\n",
                             url.c_str(), offset, offset + length,
                             long(resp.status_code));
            }
        }
        return out;
    }
    transientError_.store(false, std::memory_order_relaxed);

    const auto& body = resp.body;
    if (body.size() > length && body.size() >= offset + length) {
        out.bytes.resize(length);
        std::memcpy(out.bytes.data(), body.data() + offset, length);
    } else {
        out.bytes.resize(body.size());
        if (!body.empty())
            std::memcpy(out.bytes.data(), body.data(), out.bytes.size());
    }
    return out;
}

FetchResult HttpSource::fetchFromShard(const ChunkKey& key)
{
    FetchResult out;
    std::string url = shardUrl(key);
    const int nChunks = totalChunksPerShard();
    if (nChunks <= 0) return out;

    const int inner = innerChunkIndex(key);
    if (inner < 0 || inner >= nChunks) return out;

    std::shared_ptr<utils::detail::ShardIndex> shardIndex;
    {
        std::lock_guard<std::mutex> lock(shardCacheMutex_);
        if (auto it = shardCacheMap_.find(url); it != shardCacheMap_.end()) {
            shardCacheLru_.splice(shardCacheLru_.begin(), shardCacheLru_, it->second);
            shardIndex = it->second->entry.index;
        }
    }

    if (!shardIndex) {
        const std::size_t indexBytes = std::size_t(nChunks) * 16;
        auto raw = httpGetRange(url, 0, indexBytes);
        if (raw.bytes.size() != indexBytes) {
            out.wasAbsent = raw.wasAbsent;
            out.transientError = raw.transientError || (!raw.wasAbsent);
            return out;
        }

        std::span<const std::byte> span(
            reinterpret_cast<const std::byte*>(raw.bytes.data()), raw.bytes.size());
        auto parsed = std::make_shared<utils::detail::ShardIndex>(
            utils::detail::ShardIndex::deserialize(span, size_t(nChunks)));

        constexpr size_t kShardIndexBudget = 64ull << 20;
        std::lock_guard<std::mutex> lock(shardCacheMutex_);
        if (auto it = shardCacheMap_.find(url); it != shardCacheMap_.end()) {
            if (!it->second->entry.index) it->second->entry.index = parsed;
            shardIndex = it->second->entry.index;
            shardCacheLru_.splice(shardCacheLru_.begin(), shardCacheLru_, it->second);
        } else {
            shardCacheLru_.push_front({url, {{}, parsed}});
            shardCacheMap_[url] = shardCacheLru_.begin();
            shardCacheBytes_ += indexBytes;
            shardIndex = parsed;

            while (shardCacheBytes_ > kShardIndexBudget
                   && shardCacheLru_.size() > 1) {
                auto& victim = shardCacheLru_.back();
                shardCacheBytes_ -= std::size_t(nChunks) * 16;
                shardCacheMap_.erase(victim.url);
                shardCacheLru_.pop_back();
            }

            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[SHARD] Cached index %s (%zu entries, cache=%zu)\n",
                             url.c_str(), std::size_t(nChunks),
                             shardCacheMap_.size());
        }
    }

    const auto& entry = shardIndex->entries[inner];

    if (entry.is_missing() || entry.is_empty() || entry.is_verified_absent()) {
        const char* why = entry.is_missing() ? "missing"
                        : entry.is_empty() ? "zarr-empty"
                        : "verified-absent";
        std::fprintf(stderr,
            "[fetchFromShard ABSENT] %s inner=%d -> %s\n",
            url.c_str(), inner, why);
        out.wasAbsent = true;
        return out;
    }
    if (entry.nbytes == 0) return out;

    return httpGetRange(url, entry.offset, entry.nbytes);
}

FetchResult HttpSource::fetch(const ChunkKey& key)
{
    if (sharded_) return fetchFromShard(key);
    return httpGet(chunkUrl(key));
}

std::vector<uint8_t> HttpSource::fetchWholeShard(int level, int sz, int sy, int sx)
{
    std::string url;
    url.reserve(baseUrl_.size() + 32);
    url += baseUrl_;
    url += '/';
    url += std::to_string(level);
    url += "/c/";
    url += std::to_string(sz);
    url += '/';
    url += std::to_string(sy);
    url += '/';
    url += std::to_string(sx);
    return std::move(httpGet(url).bytes);
}

std::array<int, 3> HttpSource::shardsPerAxis(int level) const noexcept
{
    auto shape = levelsLevelShape(levels_, level);
    return {
        (shape[0] + shardShape_[0] - 1) / std::max(shardShape_[0], 1),
        (shape[1] + shardShape_[1] - 1) / std::max(shardShape_[1], 1),
        (shape[2] + shardShape_[2] - 1) / std::max(shardShape_[2], 1),
    };
}

int HttpSource::numLevels() const noexcept { return levelsNumLevels(levels_); }
std::array<int, 3> HttpSource::chunkShape(int level) const noexcept { return levelsChunkShape(levels_, level); }
std::array<int, 3> HttpSource::levelShape(int level) const noexcept { return levelsLevelShape(levels_, level); }

}  // namespace vc::cache
