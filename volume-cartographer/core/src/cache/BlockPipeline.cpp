#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <utils/video_codec.hpp>

namespace vc::cache {

// Convert ChunkKey to zarr chunk indices (z, y, x)
static std::array<size_t, 3> chunkIndices(const ChunkKey& key) {
    return {static_cast<size_t>(key.iz), static_cast<size_t>(key.iy), static_cast<size_t>(key.ix)};
}

static std::optional<std::vector<std::byte>> zarrReadChunk(
    utils::ZarrArray& zarr, const ChunkKey& key)
{
    auto idx = chunkIndices(key);
    return zarr.read_inner_chunk_from_shard(idx);
}

// Write compressed chunk to ZarrArray (handles both sharded and unsharded)
static void zarrWriteChunk(
    utils::ZarrArray& zarr, const ChunkKey& key,
    const uint8_t* data, size_t size)
{
    auto idx = chunkIndices(key);
    std::span<const std::byte> byteSpan(
        reinterpret_cast<const std::byte*>(data), size);
    zarr.write_inner_chunk_to_shard(idx, byteSpan);
}

// Check if chunk exists in ZarrArray
static bool zarrChunkExists(const utils::ZarrArray& zarr, const ChunkKey& key)
{
    auto idx = chunkIndices(key);
    return zarr.inner_chunk_exists(idx);
}

static bool isAllZero(const uint8_t* data, size_t size) noexcept
{
    const auto* p = reinterpret_cast<const uint64_t*>(data);
    size_t n8 = size / 8;
    for (size_t i = 0; i < n8; i++)
        if (p[i] != 0) return false;
    for (size_t i = n8 * 8; i < size; i++)
        if (data[i] != 0) return false;
    return true;
}

// Helper to build LRUCache config for the hot tier
static auto makeHotConfig(const BlockPipeline::Config& cfg) {
    using HotCache = utils::LRUCache<ChunkKey, ChunkDataPtr, ChunkKeyHash>;
    typename HotCache::Config c;
    c.max_bytes = cfg.hotMaxBytes;
    c.evict_target = 15.0 / 16.0;
    c.promote_on_read = false;  // VC3D pattern: no LRU churn on reads
    c.size_fn = [](const ChunkDataPtr& p) -> std::size_t {
        return p ? p->totalBytes() : 0;
    };
    return c;
}

BlockPipeline::BlockPipeline(
    Config config,
    std::unique_ptr<ChunkSource> source,
    DecompressFn decompress,
    std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels)
    : hotCache_(makeHotConfig(config))
    , config_(std::move(config))
    , diskLevels_(std::move(diskLevels))
    , source_(std::move(source))
    , decompress_(std::move(decompress))
    , ioPool_(config_.ioThreads)
    , blockCache_(BlockCache::Config{config_.hotMaxBytes})
{
    auto* httpSource = dynamic_cast<HttpChunkSource*>(source_.get());
    bool sharded = httpSource && httpSource->isSharded();

    // --- Shard mapper: ChunkKey → ShardKey ---
    if (sharded) {
        auto cps = httpSource->chunksPerShard();
        ioPool_.setShardMapper([cps](const ChunkKey& key) -> ShardKey {
            return {key.level, key.iz / cps[0], key.iy / cps[1], key.ix / cps[2]};
        });
    } else {
        // Non-sharded: each chunk is its own "shard"
        ioPool_.setShardMapper([](const ChunkKey& key) -> ShardKey {
            return {key.level, key.iz, key.iy, key.ix};
        });
    }

    // --- Fetch function: ShardKey → all chunks from that shard ---
    if (sharded) {
        auto cps = httpSource->chunksPerShard();
        ioPool_.setFetchFunc([this, cps](const ShardKey& shard) -> IOPool::FetchResult {
            using Clock = std::chrono::steady_clock;

            auto* dz = (shard.level < static_cast<int>(diskLevels_.size()))
                ? diskLevels_[shard.level].get() : nullptr;
            auto* http = dynamic_cast<HttpChunkSource*>(source_.get());

            // Ensure shard is on disk
            if (dz) {
                std::vector<std::size_t> shard_idx = {
                    static_cast<std::size_t>(shard.sz),
                    static_cast<std::size_t>(shard.sy),
                    static_cast<std::size_t>(shard.sx)};
                auto shard_path = dz->chunk_path(shard_idx);

                if (!std::filesystem::exists(shard_path)) {
                    auto t0 = Clock::now();
                    auto shardBytes = http->fetchWholeShard(
                        shard.level, shard.sz, shard.sy, shard.sx);
                    auto fetchMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        Clock::now() - t0).count();

                    if (!shardBytes.empty()) {
                        std::filesystem::create_directories(shard_path.parent_path());
                        std::ofstream f(shard_path, std::ios::binary | std::ios::trunc);
                        f.write(reinterpret_cast<const char*>(shardBytes.data()),
                                static_cast<std::streamsize>(shardBytes.size()));

                        auto n = statIceFetches_.fetch_add(1, std::memory_order_relaxed);
                        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
                        if (n < 20 || n % 100 == 0)
                            std::fprintf(stderr, "[Cache] shard-fetch #%lu lvl=%d shard(%d,%d,%d) %zuB %ldms\n",
                                         n + 1, shard.level, shard.sz, shard.sy, shard.sx,
                                         shardBytes.size(), fetchMs);
                    }
                }
            }

            // Read ALL inner chunks from the shard on disk
            IOPool::FetchResult result;
            if (!dz) return result;

            auto ls = http->levelShape(shard.level);
            auto cs = http->chunkShape(shard.level);
            int gridZ = (ls[0] + cs[0] - 1) / cs[0];
            int gridY = (ls[1] + cs[1] - 1) / cs[1];
            int gridX = (ls[2] + cs[2] - 1) / cs[2];

            int baseZ = shard.sz * cps[0];
            int baseY = shard.sy * cps[1];
            int baseX = shard.sx * cps[2];

            for (int dz_ = 0; dz_ < cps[0]; dz_++) {
                for (int dy = 0; dy < cps[1]; dy++) {
                    for (int dx = 0; dx < cps[2]; dx++) {
                        int iz = baseZ + dz_, iy = baseY + dy, ix = baseX + dx;
                        if (iz >= gridZ || iy >= gridY || ix >= gridX) continue;

                        ChunkKey ck{shard.level, iz, iy, ix};
                        auto diskData = zarrReadChunk(*dz, ck);
                        if (diskData && !diskData->empty()) {
                            auto& bytes = *diskData;
                            result.emplace_back(ck, std::vector<uint8_t>(
                                reinterpret_cast<const uint8_t*>(bytes.data()),
                                reinterpret_cast<const uint8_t*>(bytes.data() + bytes.size())));
                            statColdHits_.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
            return result;
        });
    } else {
        // Non-sharded: download individual chunk
        ioPool_.setFetchFunc([this](const ShardKey& shard) -> IOPool::FetchResult {
            ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};

            // Check disk cache first
            auto* dz = (key.level < static_cast<int>(diskLevels_.size()))
                ? diskLevels_[key.level].get() : nullptr;
            if (dz) {
                auto diskData = zarrReadChunk(*dz, key);
                if (diskData && !diskData->empty()) {
                    statColdHits_.fetch_add(1, std::memory_order_relaxed);
                    auto& bytes = *diskData;
                    return {{key, {reinterpret_cast<const uint8_t*>(bytes.data()),
                                   reinterpret_cast<const uint8_t*>(bytes.data() + bytes.size())}}};
                }
            }

            // Fetch from remote
            if (!source_) return {{key, {}}};
            std::vector<uint8_t> data;
            try {
                data = source_->fetch(key);
            } catch (const std::exception&) {
                return {{key, {}}};
            }
            if (data.empty() || isAllZero(data.data(), data.size()))
                return {{key, {}}};

            statIceFetches_.fetch_add(1, std::memory_order_relaxed);
            return {{key, std::move(data)}};
        });
    }

    // --- Completion callback: decompress all chunks, hotPut, recompress if needed ---
    ioPool_.setCompletionCallback(
        [this, sharded](IOPool::FetchResult&& chunks) {
            bool anyArrived = false;
            ChunkKey lastKey{};

            for (auto& [key, compressed] : chunks) {
                if (compressed.empty()) {
                    // Non-sharded: single chunk was empty, negative-cache it
                    if (!sharded) {
                        bloomAdd(key);
                        std::lock_guard lock(negativeMutex_);
                        negativeCache_.insert(key);
                    }
                    continue;
                }

                if (hotCache_.contains(key)) continue;

                if (decompress_) {
                    auto data = decompress_(compressed, key);
                    if (data) {
                        insertChunkAsBlocks(key, *data, key.level == residentLevel_);
                        hotPut(key, data);
                        anyArrived = true;
                        lastKey = key;

                        // Non-sharded: recompress to h265 and write to disk shard
                        if (!sharded) {
                            auto* dz = (key.level < static_cast<int>(diskLevels_.size()))
                                ? diskLevels_[key.level].get() : nullptr;
                            if (dz) {
                                auto hotData = hotGet(key);
                                if (hotData) {
                                    auto cs = source_->chunkShape(key.level);
                                    utils::VideoCodecParams vp;
                                    vp.depth = cs[0]; vp.height = cs[1]; vp.width = cs[2];
                                    vp.qp = 0;  // lossless
                                    auto h265 = utils::video_encode(
                                        {reinterpret_cast<const std::byte*>(hotData->rawData()),
                                         hotData->totalBytes()}, vp);
                                    zarrWriteChunk(*dz, key,
                                        reinterpret_cast<const uint8_t*>(h265.data()), h265.size());
                                    statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                        }
                    }
                }
            }

            // Notify listeners once per shard completion
            if (anyArrived && !chunkArrivedFlag_.exchange(true, std::memory_order_acq_rel)) {
                std::lock_guard cbLock(callbackMutex_);
                for (const auto& [id, cb] : chunkReadyListeners_)
                    cb(lastKey);
            }
        });

    // Start IO workers after callbacks are set to avoid data races.
    ioPool_.start();

    loadNegativeCache();
}

BlockPipeline::~BlockPipeline()
{
    ioPool_.stop();

    // Print final fetch stats summary
    auto cold = statColdHits_.load(std::memory_order_relaxed);
    auto ice = statIceFetches_.load(std::memory_order_relaxed);
    if (cold > 0 || ice > 0) {
        std::fprintf(stderr, "[Cache] session summary: coldHits=%lu iceFetches=%lu (%.0f%% from disk)\n",
                     cold, ice, (cold + ice) > 0 ? 100.0 * cold / (cold + ice) : 0.0);
    }
    saveNegativeCache();
}

// =============================================================================
// Bloom filter for negative cache (lock-free fast path)
// =============================================================================

void BlockPipeline::bloomAdd(const ChunkKey& key) noexcept
{
    auto h = ChunkKeyHash{}(key);
    // Two hash functions derived from the single hash via golden ratio mixing
    auto h1 = h;
    auto h2 = h * 0x9E3779B97F4A7C15ULL;
    auto idx1 = h1 % kBloomBits;
    auto idx2 = h2 % kBloomBits;
    negativeBloom_[idx1 / 64].fetch_or(1ULL << (idx1 % 64), std::memory_order_relaxed);
    negativeBloom_[idx2 / 64].fetch_or(1ULL << (idx2 % 64), std::memory_order_relaxed);
}

bool BlockPipeline::bloomMayContain(const ChunkKey& key) const noexcept
{
    auto h = ChunkKeyHash{}(key);
    auto h1 = h;
    auto h2 = h * 0x9E3779B97F4A7C15ULL;
    auto idx1 = h1 % kBloomBits;
    auto idx2 = h2 % kBloomBits;
    auto b1 = negativeBloom_[idx1 / 64].load(std::memory_order_relaxed) & (1ULL << (idx1 % 64));
    auto b2 = negativeBloom_[idx2 / 64].load(std::memory_order_relaxed) & (1ULL << (idx2 % 64));
    return b1 && b2;
}

void BlockPipeline::bloomClear() noexcept
{
    for (auto& word : negativeBloom_)
        word.store(0, std::memory_order_relaxed);
}

// =============================================================================
// Non-blocking reads
// =============================================================================

ChunkDataPtr BlockPipeline::get(const ChunkKey& key)
{
    // Check hot tier
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
    }

    statMisses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

std::pair<ChunkDataPtr, int> BlockPipeline::getBestAvailable(
    const ChunkKey& key)
{
    int maxLevel = source_ ? source_->numLevels() - 1 : 0;

    // Try the requested level first, then progressively coarser
    for (int lvl = key.level; lvl <= maxLevel; lvl++) {
        ChunkKey coarsened =
            (lvl == key.level) ? key : key.coarsen(lvl);

        auto data = get(coarsened);
        if (data) return {data, lvl};
    }

    // Nothing available at any level
    return {nullptr, -1};
}

// =============================================================================
// Blocking reads
// =============================================================================

ChunkDataPtr BlockPipeline::getBlocking(const ChunkKey& key)
{
    // Fast path: hot cache hit. This is the common case during rendering —
    // just hash + shard + lock_guard + flat-map probe + return.
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
    }

    // Known non-existent?
    if (isNegativeCached(key)) return nullptr;

    // Full promotion chain: cold → hot, or ice → cold → hot
    auto data = loadFull(key);
    if (!data) {
        bloomAdd(key);
        std::lock_guard lock(negativeMutex_);
        negativeCache_.insert(key);
    }
    return data;
}

// =============================================================================
// Interactive fetch
// =============================================================================

void BlockPipeline::fetchInteractive(const std::vector<ChunkKey>& keys)
{
    if (keys.empty()) return;
    std::vector<ChunkKey> submit;
    submit.reserve(keys.size());
    for (const auto& key : keys) {
        if (!isReadyForNonBlockingRead(key))
            submit.push_back(key);
    }
    if (!submit.empty())
        ioPool_.updateInteractive(submit);
}


// =============================================================================
// Cache management
// =============================================================================

void BlockPipeline::clearMemory()
{
    hotCache_.clear();
}

void BlockPipeline::clearAll()
{
    ioPool_.cancelPending();
    clearMemory();
    bloomClear();
    {
        std::lock_guard lock(negativeMutex_);
        negativeCache_.clear();
    }
    if (!diskLevels_.empty()) {
        std::error_code ec;
        std::filesystem::remove(
            diskLevels_[0]->path().parent_path() / (config_.volumeId + ".negative"), ec);
    }
}

int BlockPipeline::numLevels() const noexcept
{
    return source_ ? source_->numLevels() : 0;
}

std::array<int, 3> BlockPipeline::chunkShape(int level) const noexcept
{
    return source_ ? source_->chunkShape(level) : std::array<int, 3>{0, 0, 0};
}

std::array<int, 3> BlockPipeline::levelShape(int level) const noexcept
{
    return source_ ? source_->levelShape(level) : std::array<int, 3>{0, 0, 0};
}

void BlockPipeline::setDataBounds(int minX, int maxX, int minY, int maxY, int minZ, int maxZ)
{
    std::lock_guard lock(dataBoundsMutex_);
    dataBoundsL0_ = {minX, maxX, minY, maxY, minZ, maxZ, true};
}

BlockPipeline::DataBoundsL0 BlockPipeline::dataBounds() const
{
    std::lock_guard lock(dataBoundsMutex_);
    return dataBoundsL0_;
}

bool BlockPipeline::isNegativeCached(const ChunkKey& key) const
{
    // Bloom filter fast-reject: if bloom says no, definitely not cached.
    if (!bloomMayContain(key)) return false;
    std::lock_guard lock(negativeMutex_);
    return negativeCache_.count(key) > 0;
}

bool BlockPipeline::areAllCachedInRegion(
    int level,
    int iz0, int iy0, int ix0,
    int iz1, int iy1, int ix1) const
{
    // Only count chunks in HOT cache (RAM) or negative-cached.
    // Disk-only chunks still require blocking reads — don't count them
    // as "cached" for best-effort level selection.
    for (int iz = iz0; iz <= iz1; iz++) {
        for (int iy = iy0; iy <= iy1; iy++) {
            for (int ix = ix0; ix <= ix1; ix++) {
                ChunkKey key{level, iz, iy, ix};
                if (!isReadyForNonBlockingRead(key)) return false;
            }
        }
    }
    return true;
}

size_t BlockPipeline::countAvailable(const std::vector<ChunkKey>& keys) const
{
    size_t available = 0;
    for (const auto& key : keys) {
        if (isAvailableWithoutRemoteFetch(key)) {
            available++;
        }
    }
    return available;
}

BlockPipeline::ChunkReadyCallbackId
BlockPipeline::addChunkReadyListener(ChunkReadyCallback cb)
{
    std::lock_guard lock(callbackMutex_);
    auto id = nextListenerId_.fetch_add(1, std::memory_order_relaxed);
    chunkReadyListeners_.emplace_back(id, std::move(cb));
    return id;
}

void BlockPipeline::removeChunkReadyListener(ChunkReadyCallbackId id)
{
    std::lock_guard lock(callbackMutex_);
    auto it = std::remove_if(chunkReadyListeners_.begin(), chunkReadyListeners_.end(),
        [id](const auto& p) { return p.first == id; });
    chunkReadyListeners_.erase(it, chunkReadyListeners_.end());
}

void BlockPipeline::clearChunkArrivedFlag() noexcept
{
    chunkArrivedFlag_.store(false, std::memory_order_release);
}

// =============================================================================
// Stats
// =============================================================================

auto BlockPipeline::stats() const -> Stats
{
    Stats s;
    s.hotHits = statHotHits_.load(std::memory_order_relaxed);
    s.coldHits = statColdHits_.load(std::memory_order_relaxed);
    s.iceFetches = statIceFetches_.load(std::memory_order_relaxed);
    s.misses = statMisses_.load(std::memory_order_relaxed);
    s.hotEvictions = hotCache_.evictions();
    s.hotBytes = hotCache_.byte_size();
    s.ioPending = ioPool_.pendingCount();
    s.diskWrites = statDiskWrites_.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(negativeMutex_);
        s.negativeCount = negativeCache_.size();
    }
    s.totalSubmitted = statTotalSubmitted_.load(std::memory_order_relaxed);
    // Disk cache stats: scan shard files across all levels
    for (const auto& dz : diskLevels_) {
        if (!dz) continue;
        namespace fs = std::filesystem;
        std::error_code ec;
        for (auto& entry : fs::recursive_directory_iterator(dz->path(), ec)) {
            if (entry.is_regular_file(ec) && entry.path().filename() != "zarr.json") {
                s.diskBytes += entry.file_size(ec);
                s.diskShards++;
            }
        }
    }
    if (auto* http = dynamic_cast<HttpChunkSource*>(source_.get()))
        s.sharded = http->isSharded();
    // Also check disk levels — if any is sharded, the dataset is sharded
    if (!s.sharded) {
        for (const auto& dz : diskLevels_) {
            if (dz && dz->is_sharded()) { s.sharded = true; break; }
        }
    }
    return s;
}

// =============================================================================
// Hot tier — delegates to utils::LRUCache
// =============================================================================

ChunkDataPtr BlockPipeline::hotGet(const ChunkKey& key)
{
    return hotCache_.get_or(key, nullptr);
}

void BlockPipeline::hotPut(const ChunkKey& key, ChunkDataPtr data)
{
    if (data) insertChunkAsBlocks(key, *data, key.level == residentLevel_);
    hotCache_.put(key, std::move(data));
}

void BlockPipeline::insertChunkAsBlocks(const ChunkKey& key,
                                           const ChunkData& chunk,
                                           bool resident)
{
    const int cz = chunk.shape[0];
    const int cy = chunk.shape[1];
    const int cx = chunk.shape[2];
    if (cz <= 0 || cy <= 0 || cx <= 0) return;
    // Blocks are 16^3; chunks must align.
    const int bzN = cz / kBlockSize;
    const int byN = cy / kBlockSize;
    const int bxN = cx / kBlockSize;
    if (bzN * kBlockSize != cz || byN * kBlockSize != cy || bxN * kBlockSize != cx) return;

    const uint8_t* src = chunk.rawData();
    const int strideZ = chunk.strideZ();
    const int strideY = chunk.strideY();

    // Starting block coord of this chunk at this level.
    const int baseBz = key.iz * bzN;
    const int baseBy = key.iy * byN;
    const int baseBx = key.ix * bxN;

    uint8_t tmp[kBlockBytes];

    for (int bi = 0; bi < bzN; ++bi) {
        for (int bj = 0; bj < byN; ++bj) {
            for (int bk = 0; bk < bxN; ++bk) {
                // Gather contiguous block out of the chunk buffer.
                uint8_t* dst = tmp;
                for (int lz = 0; lz < kBlockSize; ++lz) {
                    const uint8_t* rowBase = src
                        + (bi * kBlockSize + lz) * strideZ
                        + (bj * kBlockSize) * strideY;
                    for (int ly = 0; ly < kBlockSize; ++ly) {
                        const uint8_t* p = rowBase + ly * strideY + bk * kBlockSize;
                        std::memcpy(dst, p, kBlockSize);
                        dst += kBlockSize;
                    }
                }
                BlockKey bkKey{key.level, baseBz + bi, baseBy + bj, baseBx + bk};
                if (resident)
                    blockCache_.putResident(bkKey, tmp);
                else
                    blockCache_.put(bkKey, tmp);
            }
        }
    }
}

BlockPtr BlockPipeline::blockAt(const BlockKey& key) noexcept
{
    return blockCache_.get(key);
}

void BlockPipeline::loadResidentLevel(int level)
{
    residentLevel_ = level;
    int nScales = numLevels();
    if (level < 0 || level >= nScales) return;
    auto shape = levelShape(level);
    auto chunks = chunkShape(level);
    int gridZ = (shape[0] + chunks[0] - 1) / chunks[0];
    int gridY = (shape[1] + chunks[1] - 1) / chunks[1];
    int gridX = (shape[2] + chunks[2] - 1) / chunks[2];

    for (int iz = 0; iz < gridZ; ++iz)
        for (int iy = 0; iy < gridY; ++iy)
            for (int ix = 0; ix < gridX; ++ix) {
                ChunkKey k{level, iz, iy, ix};
                (void)getBlocking(k);  // populates blockCache_ as resident via hotPut
            }
}

// =============================================================================
// Promotion helpers
// =============================================================================

ChunkDataPtr BlockPipeline::promoteFromCold(const ChunkKey& key)
{
    if (!decompress_) return nullptr;
    auto* dz = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    if (!dz) return nullptr;

    auto raw = zarrReadChunk(*dz, key);
    if (!raw || raw->empty()) return nullptr;

    statColdHits_.fetch_add(1, std::memory_order_relaxed);

    std::vector<uint8_t> compressed(
        reinterpret_cast<const uint8_t*>(raw->data()),
        reinterpret_cast<const uint8_t*>(raw->data() + raw->size()));
    auto data = decompress_(compressed, key);
    if (!data) return nullptr;
    // Move into cache; shared_ptr copy kept for return
    auto ret = data;
    hotPut(key, std::move(data));
    return ret;
}

ChunkDataPtr BlockPipeline::promoteFromIce(const ChunkKey& key)
{
    if (!source_) return nullptr;

    auto compressed = source_->fetch(key);
    if (compressed.empty()) return nullptr;

    statIceFetches_.fetch_add(1, std::memory_order_relaxed);

    // Store to cold (disk cache) — per-level zarr
    auto* dzIce = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    if (dzIce) {
        zarrWriteChunk(*dzIce, key, compressed.data(), compressed.size());
        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
    }

    // Decompress and promote to hot; release compressed bytes immediately after
    if (!decompress_) return nullptr;
    auto data = decompress_(compressed, key);
    compressed.clear();
    compressed.shrink_to_fit();
    if (!data) return nullptr;
    // Move into cache; shared_ptr copy kept for return
    auto ret = data;
    hotPut(key, std::move(data));
    return ret;
}

ChunkDataPtr BlockPipeline::loadFull(const ChunkKey& key)
{
    // Try cold (disk cache)
    auto data = promoteFromCold(key);
    if (data) return data;

    // Try ice (remote/filesystem)
    return promoteFromIce(key);
}

bool BlockPipeline::isReadyForNonBlockingRead(const ChunkKey& key) const
{
    if (hotCache_.contains(key)) return true;

    if (isNegativeCached(key)) return true;

    return false;
}

bool BlockPipeline::isAvailableWithoutRemoteFetch(const ChunkKey& key) const
{
    if (isReadyForNonBlockingRead(key)) return true;

    // Check cold tier (local zarr): if in shard, no remote fetch needed.
    auto* dzAvail = (key.level < static_cast<int>(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    if (dzAvail && zarrChunkExists(*dzAvail, key))
        return true;
    return false;
}

void BlockPipeline::flushPersistentState()
{
    saveNegativeCache();
}

// =============================================================================
// Negative cache persistence
// =============================================================================

void BlockPipeline::loadNegativeCache()
{
    if (diskLevels_.empty()) return;
    auto negRoot = diskLevels_[0]->path().parent_path();

    auto path = negRoot / (config_.volumeId + ".negative");
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return;

    int32_t level, iz, iy, ix;
    size_t count = 0;
    while (f.read(reinterpret_cast<char*>(&level), 4) &&
           f.read(reinterpret_cast<char*>(&iz), 4) &&
           f.read(reinterpret_cast<char*>(&iy), 4) &&
           f.read(reinterpret_cast<char*>(&ix), 4)) {
        ChunkKey k{level, iz, iy, ix};
        negativeCache_.insert(k);
        bloomAdd(k);
        count++;
    }
    if (count > 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "Loaded %zu negative cache entries from disk\n", count);
    }
}

void BlockPipeline::saveNegativeCache() const
{
    if (negativeCache_.empty()) return;
    if (diskLevels_.empty()) return;
    auto negRoot = diskLevels_[0]->path().parent_path();

    auto path = negRoot / (config_.volumeId + ".negative");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return;

    std::lock_guard lock(negativeMutex_);
    for (const auto& key : negativeCache_) {
        int32_t level = key.level, iz = key.iz, iy = key.iy, ix = key.ix;
        f.write(reinterpret_cast<const char*>(&level), 4);
        f.write(reinterpret_cast<const char*>(&iz), 4);
        f.write(reinterpret_cast<const char*>(&iy), 4);
        f.write(reinterpret_cast<const char*>(&ix), 4);
    }
    if (auto* log = cacheDebugLog())
        std::fprintf(log, "Saved %zu negative cache entries to disk\n",
                     negativeCache_.size());
}

}  // namespace vc::cache
