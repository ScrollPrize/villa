#include "vc/core/cache/VolumeSource.hpp"
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
            // Finest granularity: inner chunks for sharded v3, chunks otherwise.
            const auto& cs = meta.shard_config ? meta.shard_config->sub_chunks
                                               : meta.chunks;
            if (meta.shape.size() >= 3)
                lm.shape = {int(meta.shape[0]), int(meta.shape[1]), int(meta.shape[2])};
            if (cs.size() >= 3)
                lm.chunkShape = {int(cs[0]), int(cs[1]), int(cs[2])};
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
    return root_ / std::to_string(key.level) / chunkFilename(key, delimiter_);
}

std::vector<uint8_t> FileSystemSource::fetch(const ChunkKey& key)
{
    auto result = readFileToVector(chunkPath(key));
    return result ? std::move(*result) : std::vector<uint8_t>{};
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

std::string HttpSource::shardUrl(const ChunkKey& key) const
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

std::vector<uint8_t> HttpSource::httpGet(const std::string& url)
{
    auto resp = client_->get(url);
    if (!resp.ok()) return {};

    // Convert from vector<byte> to vector<uint8_t>
    std::vector<uint8_t> result(resp.body.size());
    if (!result.empty())
        std::memcpy(result.data(), resp.body.data(), result.size());
    return result;
}

std::vector<uint8_t> HttpSource::fetchFromShard(const ChunkKey& key)
{
    std::string url = shardUrl(key);

    std::shared_ptr<std::vector<uint8_t>> shardData;
    {
        std::lock_guard<std::mutex> lock(shardCacheMutex_);
        auto it = shardCache_.find(url);
        if (it != shardCache_.end())
            shardData = it->second;
    }

    if (!shardData) {
        auto raw = httpGet(url);
        if (raw.empty()) return {};
        shardData = std::make_shared<std::vector<uint8_t>>(std::move(raw));

        std::lock_guard<std::mutex> lock(shardCacheMutex_);
        if (shardCache_.size() >= 16)
            shardCache_.clear();

        auto [it, inserted] = shardCache_.emplace(url, shardData);
        if (!inserted) shardData = it->second;

        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[SHARD] Cached %s (%zu bytes, %d entries, cache=%zu)\n",
                         url.c_str(), shardData->size(), totalChunksPerShard(),
                         shardCache_.size());
    }

    int nChunks = totalChunksPerShard();
    uint64_t indexSize = static_cast<uint64_t>(nChunks) * 16;
    if (shardData->size() < indexSize) return {};

    int inner = innerChunkIndex(key);
    if (inner < 0 || inner >= nChunks) return {};

    // Index is always at the start of the shard
    const uint8_t* entry = shardData->data() + inner * 16;

    uint64_t offset = 0, nbytes = 0;
    std::memcpy(&offset, entry, 8);
    std::memcpy(&nbytes, entry + 8, 8);

    // Missing chunk sentinel
    if (offset == UINT64_MAX && nbytes == UINT64_MAX) return {};
    // Zero chunk sentinel
    if (offset == (UINT64_MAX - 1) && nbytes == 0) return {};
    if (offset + nbytes > shardData->size()) return {};

    return {shardData->data() + offset, shardData->data() + offset + nbytes};
}

std::vector<uint8_t> HttpSource::fetch(const ChunkKey& key)
{
    if (sharded_)
        return fetchFromShard(key);
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
    return httpGet(url);
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
