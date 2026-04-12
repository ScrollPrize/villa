#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/VolumeSource.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <utils/video_codec.hpp>

namespace vc::cache {

static std::array<size_t, 3> chunkIndices(const ChunkKey& key) {
    return {size_t(key.iz), size_t(key.iy), size_t(key.ix)};
}

static std::optional<std::vector<std::byte>> zarrReadChunk(
    utils::ZarrArray& zarr, const ChunkKey& key) {
    return zarr.read_inner_chunk_from_shard(chunkIndices(key));
}

static void zarrWriteChunk(
    utils::ZarrArray& zarr, const ChunkKey& key,
    const uint8_t* data, size_t size) {
    std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data), size);
    zarr.write_inner_chunk_to_shard(chunkIndices(key), bytes);
}

static bool isAllZero(const uint8_t* data, size_t size) noexcept {
    const auto* p = reinterpret_cast<const uint64_t*>(data);
    size_t n8 = size / 8;
    for (size_t i = 0; i < n8; i++) if (p[i] != 0) return false;
    for (size_t i = n8 * 8; i < size; i++) if (data[i] != 0) return false;
    return true;
}

BlockPipeline::BlockPipeline(
    Config config,
    std::unique_ptr<VolumeSource> source,
    DecompressFn decompress,
    std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels)
    : config_(std::move(config))
    , diskLevels_(std::move(diskLevels))
    , source_(std::move(source))
    , decompress_(std::move(decompress))
    , ioPool_(config_.ioThreads)
    , blockCache_(BlockCache::Config{config_.blockCacheBytes})
{
    auto* httpSource = dynamic_cast<HttpSource*>(source_.get());
    bool sharded = httpSource && httpSource->isSharded();

    if (sharded) {
        auto cps = httpSource->chunksPerShard();
        ioPool_.setShardMapper([cps](const ChunkKey& key) -> ShardKey {
            return {key.level, key.iz / cps[0], key.iy / cps[1], key.ix / cps[2]};
        });
    } else {
        ioPool_.setShardMapper([](const ChunkKey& key) -> ShardKey {
            return {key.level, key.iz, key.iy, key.ix};
        });
    }

    if (sharded) {
        auto cps = httpSource->chunksPerShard();
        ioPool_.setFetchFunc([this, cps](const ShardKey& shard) -> IOPool::FetchResult {
            using Clock = std::chrono::steady_clock;

            auto* dz = (shard.level < int(diskLevels_.size()))
                ? diskLevels_[shard.level].get() : nullptr;
            auto* http = dynamic_cast<HttpSource*>(source_.get());

            bool shardMissing = false;
            if (dz) {
                std::vector<size_t> shard_idx = {size_t(shard.sz), size_t(shard.sy), size_t(shard.sx)};
                auto shard_path = dz->chunk_path(shard_idx);
                if (!std::filesystem::exists(shard_path)) {
                    auto t0 = Clock::now();
                    auto shardBytes = http->fetchWholeShard(shard.level, shard.sz, shard.sy, shard.sx);
                    auto fetchMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
                    if (!shardBytes.empty()) {
                        std::filesystem::create_directories(shard_path.parent_path());
                        std::ofstream f(shard_path, std::ios::binary | std::ios::trunc);
                        f.write(reinterpret_cast<const char*>(shardBytes.data()),
                                std::streamsize(shardBytes.size()));
                        auto n = statIceFetches_.fetch_add(1, std::memory_order_relaxed);
                        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
                        if (n < 20 || n % 100 == 0)
                            std::fprintf(stderr, "[Cache] shard-fetch #%lu lvl=%d shard(%d,%d,%d) %zuB %ldms\n",
                                         n + 1, shard.level, shard.sz, shard.sy, shard.sx,
                                         shardBytes.size(), fetchMs);
                    } else {
                        // Shard doesn't exist remotely: materialize an empty-sentinel
                        // shard on disk so future sessions short-circuit without
                        // another remote round-trip.
                        std::vector<size_t> sidx = {size_t(shard.sz), size_t(shard.sy), size_t(shard.sx)};
                        dz->write_empty_shard(sidx);
                        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
                        shardMissing = true;
                    }
                }
            }

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

            std::vector<ChunkKey> emptyKeys;
            for (int dz_ = 0; dz_ < cps[0]; dz_++)
                for (int dy = 0; dy < cps[1]; dy++)
                    for (int dx = 0; dx < cps[2]; dx++) {
                        int iz = baseZ + dz_, iy = baseY + dy, ix = baseX + dx;
                        if (iz >= gridZ || iy >= gridY || ix >= gridX) continue;
                        ChunkKey ck{shard.level, iz, iy, ix};
                        auto diskData = zarrReadChunk(*dz, ck);
                        if (diskData && !diskData->empty()) {
                            auto& b = *diskData;
                            result.emplace_back(ck, std::vector<uint8_t>(
                                reinterpret_cast<const uint8_t*>(b.data()),
                                reinterpret_cast<const uint8_t*>(b.data() + b.size())));
                            statColdHits_.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            // Shard index flags this inner chunk as empty or missing.
                            // Record in the in-memory negative cache so future fetches skip it.
                            emptyKeys.push_back(ck);
                        }
                    }
            if (!emptyKeys.empty()) {
                std::lock_guard lock(negativeMutex_);
                for (auto& k : emptyKeys) {
                    bloomAdd(k);
                    negativeCache_.insert(k);
                }
            }
            return result;
        });
    } else {
        ioPool_.setFetchFunc([this](const ShardKey& shard) -> IOPool::FetchResult {
            ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};
            auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
            if (dz) {
                auto diskData = zarrReadChunk(*dz, key);
                if (diskData && !diskData->empty()) {
                    statColdHits_.fetch_add(1, std::memory_order_relaxed);
                    auto& b = *diskData;
                    return {{key, {reinterpret_cast<const uint8_t*>(b.data()),
                                   reinterpret_cast<const uint8_t*>(b.data() + b.size())}}};
                }
            }
            if (!source_) return {{key, {}}};
            std::vector<uint8_t> data;
            try { data = source_->fetch(key); } catch (...) { return {{key, {}}}; }
            if (data.empty() || isAllZero(data.data(), data.size()))
                return {{key, {}}};
            statIceFetches_.fetch_add(1, std::memory_order_relaxed);
            return {{key, std::move(data)}};
        });
    }

    ioPool_.setCompletionCallback(
        [this, sharded](IOPool::FetchResult&& chunks) {
            bool anyArrived = false;
            ChunkKey lastKey{};
            for (auto& [key, compressed] : chunks) {
                if (compressed.empty()) {
                    if (!sharded) {
                        bloomAdd(key);
                        std::lock_guard lock(negativeMutex_);
                        negativeCache_.insert(key);
                    }
                    continue;
                }
                if (!decompress_) continue;
                auto data = decompress_(compressed, key);
                if (!data) continue;

                insertChunkAsBlocks(key, *data, key.level == residentLevel_);
                anyArrived = true;
                lastKey = key;

                // Non-sharded: recompress to h265 and write to disk shard
                if (!sharded) {
                    auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
                    if (dz) {
                        auto cs = source_->chunkShape(key.level);
                        utils::VideoCodecParams vp;
                        vp.depth = cs[0]; vp.height = cs[1]; vp.width = cs[2];
                        vp.qp = 0;
                        auto h265 = utils::video_encode(
                            {reinterpret_cast<const std::byte*>(data->rawData()),
                             data->totalBytes()}, vp);
                        zarrWriteChunk(*dz, key,
                            reinterpret_cast<const uint8_t*>(h265.data()), h265.size());
                        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            if (anyArrived && !chunkArrivedFlag_.exchange(true, std::memory_order_acq_rel)) {
                std::lock_guard cbLock(callbackMutex_);
                for (const auto& [id, cb] : chunkReadyListeners_)
                    cb(lastKey);
            }
        });

    ioPool_.start();
    loadNegativeCache();
}

BlockPipeline::~BlockPipeline() {
    ioPool_.stop();
    auto cold = statColdHits_.load();
    auto ice = statIceFetches_.load();
    if (cold > 0 || ice > 0) {
        std::fprintf(stderr, "[Cache] session summary: coldHits=%lu iceFetches=%lu (%.0f%% from disk)\n",
                     cold, ice, (cold + ice) > 0 ? 100.0 * cold / (cold + ice) : 0.0);
    }
    saveNegativeCache();
}

void BlockPipeline::bloomAdd(const ChunkKey& key) noexcept {
    auto h = ChunkKeyHash{}(key);
    uint64_t h1 = h, h2 = h * 0x9E3779B97F4A7C15ULL;
    size_t i1 = h1 % kBloomBits, i2 = h2 % kBloomBits;
    negativeBloom_[i1 / 64].fetch_or(1ULL << (i1 % 64), std::memory_order_relaxed);
    negativeBloom_[i2 / 64].fetch_or(1ULL << (i2 % 64), std::memory_order_relaxed);
}

bool BlockPipeline::bloomMayContain(const ChunkKey& key) const noexcept {
    auto h = ChunkKeyHash{}(key);
    uint64_t h1 = h, h2 = h * 0x9E3779B97F4A7C15ULL;
    size_t i1 = h1 % kBloomBits, i2 = h2 % kBloomBits;
    auto b1 = negativeBloom_[i1 / 64].load(std::memory_order_relaxed) & (1ULL << (i1 % 64));
    auto b2 = negativeBloom_[i2 / 64].load(std::memory_order_relaxed) & (1ULL << (i2 % 64));
    return b1 && b2;
}

void BlockPipeline::bloomClear() noexcept {
    for (auto& w : negativeBloom_) w.store(0, std::memory_order_relaxed);
}

void BlockPipeline::fetchInteractive(const std::vector<ChunkKey>& keys) {
    if (keys.empty()) return;
    std::vector<ChunkKey> submit;
    submit.reserve(keys.size());
    for (const auto& key : keys) {
        if (isNegativeCached(key)) continue;
        submit.push_back(key);
    }
    if (!submit.empty()) ioPool_.updateInteractive(submit);
}

BlockPtr BlockPipeline::blockAt(const BlockKey& key) noexcept {
    auto b = blockCache_.get(key);
    if (b) statBlockHits_.fetch_add(1, std::memory_order_relaxed);
    else   statMisses_.fetch_add(1, std::memory_order_relaxed);
    return b;
}

ChunkDataPtr BlockPipeline::fetchChunkBlocking(const ChunkKey& key) {
    if (isNegativeCached(key)) return nullptr;

    // Try cold first.
    auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    if (dz && decompress_) {
        // If the shard index already flags this inner chunk as empty/missing,
        // negative-cache and short-circuit.
        if (dz->is_sharded()) {
            auto idx = chunkIndices(key);
            if (!dz->inner_chunk_exists(idx)) {
                bloomAdd(key);
                std::lock_guard lock(negativeMutex_);
                negativeCache_.insert(key);
                return nullptr;
            }
        }
        auto raw = zarrReadChunk(*dz, key);
        if (raw && !raw->empty()) {
            statColdHits_.fetch_add(1, std::memory_order_relaxed);
            std::vector<uint8_t> compressed(
                reinterpret_cast<const uint8_t*>(raw->data()),
                reinterpret_cast<const uint8_t*>(raw->data() + raw->size()));
            auto data = decompress_(compressed, key);
            if (data) return data;
        }
    }

    // Ice.
    if (!source_) return nullptr;
    std::vector<uint8_t> compressed;
    try { compressed = source_->fetch(key); } catch (...) { return nullptr; }
    if (compressed.empty()) {
        bloomAdd(key);
        std::lock_guard lock(negativeMutex_);
        negativeCache_.insert(key);
        return nullptr;
    }
    statIceFetches_.fetch_add(1, std::memory_order_relaxed);

    if (dz) {
        zarrWriteChunk(*dz, key, compressed.data(), compressed.size());
        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
    }

    if (!decompress_) return nullptr;
    return decompress_(compressed, key);
}

BlockPtr BlockPipeline::getBlockingBlock(const BlockKey& key) {
    if (auto b = blockCache_.get(key)) {
        statBlockHits_.fetch_add(1, std::memory_order_relaxed);
        return b;
    }

    auto cs = chunkShape(key.level);
    if (cs[0] <= 0 || cs[1] <= 0 || cs[2] <= 0) return nullptr;
    int bpcZ = cs[0] / kBlockSize;
    int bpcY = cs[1] / kBlockSize;
    int bpcX = cs[2] / kBlockSize;
    if (bpcZ <= 0 || bpcY <= 0 || bpcX <= 0) return nullptr;

    ChunkKey ck{key.level, key.bz / bpcZ, key.by / bpcY, key.bx / bpcX};
    auto data = fetchChunkBlocking(ck);
    if (!data) return nullptr;
    insertChunkAsBlocks(ck, *data, ck.level == residentLevel_);
    return blockCache_.get(key);
}

void BlockPipeline::insertChunkAsBlocks(const ChunkKey& key,
                                        const ChunkData& chunk,
                                        bool resident) {
    const int cz = chunk.shape[0];
    const int cy = chunk.shape[1];
    const int cx = chunk.shape[2];
    if (cz <= 0 || cy <= 0 || cx <= 0) return;
    const int bzN = cz / kBlockSize;
    const int byN = cy / kBlockSize;
    const int bxN = cx / kBlockSize;
    if (bzN * kBlockSize != cz || byN * kBlockSize != cy || bxN * kBlockSize != cx) return;

    const uint8_t* src = chunk.rawData();
    const int strideZ = chunk.strideZ();
    const int strideY = chunk.strideY();

    const int baseBz = key.iz * bzN;
    const int baseBy = key.iy * byN;
    const int baseBx = key.ix * bxN;

    uint8_t tmp[kBlockBytes];
    for (int bi = 0; bi < bzN; ++bi) {
        for (int bj = 0; bj < byN; ++bj) {
            for (int bk = 0; bk < bxN; ++bk) {
                uint8_t* dst = tmp;
                for (int lz = 0; lz < kBlockSize; ++lz) {
                    const uint8_t* zRow = src + (bi * kBlockSize + lz) * strideZ;
                    for (int ly = 0; ly < kBlockSize; ++ly) {
                        const uint8_t* p = zRow + (bj * kBlockSize + ly) * strideY + bk * kBlockSize;
                        std::memcpy(dst, p, kBlockSize);
                        dst += kBlockSize;
                    }
                }
                BlockKey bkKey{key.level, baseBz + bi, baseBy + bj, baseBx + bk};
                if (resident) blockCache_.putResident(bkKey, tmp);
                else          blockCache_.put(bkKey, tmp);
            }
        }
    }
}

void BlockPipeline::loadResidentLevel(int level) {
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
                auto data = fetchChunkBlocking(k);
                if (data) insertChunkAsBlocks(k, *data, true);
            }
}

void BlockPipeline::clearMemory() {
    blockCache_.clearEvictable();
}

void BlockPipeline::clearAll() {
    ioPool_.cancelPending();
    blockCache_.clearEvictable();
    bloomClear();
    std::lock_guard lock(negativeMutex_);
    negativeCache_.clear();
}

int BlockPipeline::numLevels() const noexcept {
    return source_ ? source_->numLevels() : 0;
}

std::array<int, 3> BlockPipeline::chunkShape(int level) const noexcept {
    return source_ ? source_->chunkShape(level) : std::array<int, 3>{0, 0, 0};
}

std::array<int, 3> BlockPipeline::levelShape(int level) const noexcept {
    return source_ ? source_->levelShape(level) : std::array<int, 3>{0, 0, 0};
}

void BlockPipeline::setDataBounds(int minX, int maxX, int minY, int maxY, int minZ, int maxZ) {
    std::lock_guard lock(dataBoundsMutex_);
    dataBoundsL0_ = {minX, maxX, minY, maxY, minZ, maxZ, true};
}

BlockPipeline::DataBoundsL0 BlockPipeline::dataBounds() const {
    std::lock_guard lock(dataBoundsMutex_);
    return dataBoundsL0_;
}

bool BlockPipeline::isNegativeCached(const ChunkKey& key) const {
    if (!bloomMayContain(key)) return false;
    std::lock_guard lock(negativeMutex_);
    return negativeCache_.count(key) > 0;
}

size_t BlockPipeline::countAvailable(const std::vector<ChunkKey>& keys) const {
    size_t n = 0;
    auto cs = source_ ? source_->chunkShape(0) : std::array<int, 3>{0, 0, 0};
    for (const auto& key : keys) {
        if (isNegativeCached(key)) { n++; continue; }
        auto csk = source_ ? source_->chunkShape(key.level) : cs;
        if (csk[0] <= 0) continue;
        int bpcZ = csk[0] / kBlockSize;
        int bpcY = csk[1] / kBlockSize;
        int bpcX = csk[2] / kBlockSize;
        if (bpcZ <= 0 || bpcY <= 0 || bpcX <= 0) continue;
        BlockKey bk{key.level, key.iz * bpcZ, key.iy * bpcY, key.ix * bpcX};
        if (const_cast<BlockCache&>(blockCache_).get(bk)) n++;
    }
    return n;
}

BlockPipeline::ChunkReadyCallbackId
BlockPipeline::addChunkReadyListener(ChunkReadyCallback cb) {
    std::lock_guard lock(callbackMutex_);
    auto id = nextListenerId_.fetch_add(1, std::memory_order_relaxed);
    chunkReadyListeners_.emplace_back(id, std::move(cb));
    return id;
}

void BlockPipeline::removeChunkReadyListener(ChunkReadyCallbackId id) {
    std::lock_guard lock(callbackMutex_);
    auto it = std::remove_if(chunkReadyListeners_.begin(), chunkReadyListeners_.end(),
        [id](const auto& p) { return p.first == id; });
    chunkReadyListeners_.erase(it, chunkReadyListeners_.end());
}

void BlockPipeline::clearChunkArrivedFlag() noexcept {
    chunkArrivedFlag_.store(false, std::memory_order_release);
}

auto BlockPipeline::stats() const -> Stats {
    Stats s;
    s.blockHits = statBlockHits_.load(std::memory_order_relaxed);
    s.coldHits = statColdHits_.load(std::memory_order_relaxed);
    s.iceFetches = statIceFetches_.load(std::memory_order_relaxed);
    s.misses = statMisses_.load(std::memory_order_relaxed);
    s.blocksResident = blockCache_.residentSize();
    s.blocksEvictable = blockCache_.evictableSize();
    s.ioPending = ioPool_.pendingCount();
    s.diskWrites = statDiskWrites_.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(negativeMutex_);
        s.negativeCount = negativeCache_.size();
    }
    s.totalSubmitted = statTotalSubmitted_.load(std::memory_order_relaxed);
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
    if (auto* http = dynamic_cast<HttpSource*>(source_.get()))
        s.sharded = http->isSharded();
    if (!s.sharded) {
        for (const auto& dz : diskLevels_)
            if (dz && dz->is_sharded()) { s.sharded = true; break; }
    }
    return s;
}

// Negative info now lives in the on-disk shard index (empty chunks marked
// there directly); the in-memory bloom + set is a session-scoped speedup.
void BlockPipeline::loadNegativeCache() {}
void BlockPipeline::saveNegativeCache() const {}
void BlockPipeline::flushPersistentState() {}

std::unique_ptr<BlockPipeline> openFilesystemPipeline(
    VcDataset* ds, size_t maxBytes, const std::filesystem::path& datasetPath)
{
    FileSystemSource::LevelMeta lm;
    const auto& shape = ds->shape();
    const auto& chunks = ds->defaultChunkShape();
    lm.shape = {int(shape[0]), int(shape[1]), int(shape[2])};
    lm.chunkShape = {int(chunks[0]), int(chunks[1]), int(chunks[2])};
    auto source = std::make_unique<FileSystemSource>(
        datasetPath.parent_path(), ds->delimiter(), std::vector{lm});
    auto decompress = makeVcDecompressor(ds);
    BlockPipeline::Config cfg;
    cfg.blockCacheBytes = maxBytes;
    return std::make_unique<BlockPipeline>(
        std::move(cfg), std::move(source), std::move(decompress));
}

}  // namespace vc::cache
