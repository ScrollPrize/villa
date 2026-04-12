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

// Decode a canonical h265 chunk from disk bytes. Uses the video header for
// dims; independent of any source VcDataset.
static ChunkDataPtr decodeCanonicalH265(const std::vector<uint8_t>& compressed) {
    std::span<const std::byte> bytes(
        reinterpret_cast<const std::byte*>(compressed.data()), compressed.size());
    if (!utils::is_video_compressed(bytes)) return nullptr;
    auto dims = utils::video_header_dims(bytes);
    utils::VideoCodecParams vp;
    vp.depth = dims[0]; vp.height = dims[1]; vp.width = dims[2];
    size_t n = size_t(dims[0]) * dims[1] * dims[2];
    auto decoded = utils::video_decode(bytes, n, vp);
    auto out = std::make_shared<ChunkData>();
    out->shape = {int(dims[0]), int(dims[1]), int(dims[2])};
    out->elementSize = 1;
    out->bytes.resize(decoded.size());
    std::memcpy(out->bytes.data(), decoded.data(), decoded.size());
    return out;
}

// Encode decoded chunk bytes as canonical h265. qp=36 is the canonical
// disk-storage quality for this codebase.
static std::vector<std::byte> encodeCanonicalH265(const ChunkData& chunk) {
    utils::VideoCodecParams vp;
    vp.depth = chunk.shape[0];
    vp.height = chunk.shape[1];
    vp.width = chunk.shape[2];
    vp.qp = 36;
    return utils::video_encode(
        {reinterpret_cast<const std::byte*>(chunk.rawData()), chunk.totalBytes()},
        vp);
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
    // Per-chunk IOPool granularity. Shard mapper is identity; each canonical
    // chunk is its own work unit. When the HTTP source is itself sharded,
    // HttpSource's internal shard cache amortizes S3 GETs across chunks that
    // fall in the same source shard.
    ioPool_.setShardMapper([](const ChunkKey& key) -> ShardKey {
        return {key.level, key.iz, key.iy, key.ix};
    });

    ioPool_.setFetchFunc([this](const ShardKey& shard) -> IOPool::FetchResult {
        ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};
        if (isNegativeCached(key)) return {};

        auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;

        // Canonical disk tier first (remote-sourced volumes write transcoded
        // h265 chunks here; local-source volumes have no disk tier).
        if (dz) {
            if (dz->is_sharded() && !dz->inner_chunk_exists(chunkIndices(key))) {
                bloomAdd(key);
                std::lock_guard lock(negativeMutex_);
                negativeCache_.insert(key);
                return {};
            }
            auto diskBytes = zarrReadChunk(*dz, key);
            if (diskBytes && !diskBytes->empty()) {
                statColdHits_.fetch_add(1, std::memory_order_relaxed);
                std::vector<uint8_t> buf(
                    reinterpret_cast<const uint8_t*>(diskBytes->data()),
                    reinterpret_cast<const uint8_t*>(diskBytes->data() + diskBytes->size()));
                return {{key, std::move(buf)}};
            }
        }

        if (!source_) return {};

        // Local source (no disk tier): pass source bytes through; the
        // completion callback decompresses with the source codec.
        if (!dz) {
            std::vector<uint8_t> compressed;
            try { compressed = source_->fetch(key); } catch (...) { return {}; }
            if (compressed.empty() || isAllZero(compressed.data(), compressed.size())) {
                bloomAdd(key);
                std::lock_guard lock(negativeMutex_);
                negativeCache_.insert(key);
                return {};
            }
            statIceFetches_.fetch_add(1, std::memory_order_relaxed);
            return {{key, std::move(compressed)}};
        }

        // Remote source with canonical disk tier. Assemble the canonical 128^3
        // chunk (rechunking if source granularity differs), h265-encode, write
        // to canonical disk, return canonical h265 bytes.
        auto data = assembleCanonicalChunk(key);
        if (!data) {
            bloomAdd(key);
            {
                std::lock_guard lock(negativeMutex_);
                negativeCache_.insert(key);
            }
            if (dz->is_sharded()) dz->mark_inner_chunk_empty(chunkIndices(key));
            return {};
        }
        statIceFetches_.fetch_add(1, std::memory_order_relaxed);

        auto h265 = encodeCanonicalH265(*data);
        zarrWriteChunk(*dz, key,
            reinterpret_cast<const uint8_t*>(h265.data()), h265.size());
        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);

        std::vector<uint8_t> out(h265.size());
        std::memcpy(out.data(), h265.data(), h265.size());
        return {{key, std::move(out)}};
    });

    ioPool_.setCompletionCallback(
        [this](IOPool::FetchResult&& chunks) {
            bool anyArrived = false;
            ChunkKey lastKey{};
            for (auto& [key, compressed] : chunks) {
                if (compressed.empty()) continue;

                // For volumes with a canonical disk tier, the fetch func always
                // returns canonical h265 (either read from disk or freshly
                // transcoded). For local-source volumes there is no disk tier
                // and bytes are raw source-codec.
                ChunkDataPtr data = decodeCanonicalH265(compressed);
                if (!data && decompress_) data = decompress_(compressed, key);
                if (!data) continue;

                insertChunkAsBlocks(key, *data, key.level == residentLevel_);
                anyArrived = true;
                lastKey = key;
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

// Rechunk source chunks into a canonical 128^3 chunk. Returns a decoded
// ChunkData for the canonical chunk, or null if the canonical region is
// entirely absent from the source.
ChunkDataPtr BlockPipeline::assembleCanonicalChunk(const ChunkKey& canonKey) {
    if (!source_ || !decompress_) return nullptr;
    constexpr int C = 128;
    auto scs = source_->chunkShape(canonKey.level);
    if (scs[0] <= 0 || scs[1] <= 0 || scs[2] <= 0) return nullptr;

    int cz0 = canonKey.iz * C, cy0 = canonKey.iy * C, cx0 = canonKey.ix * C;
    int cz1 = cz0 + C,         cy1 = cy0 + C,         cx1 = cx0 + C;
    int sz0 = cz0 / scs[0], sz1 = (cz1 + scs[0] - 1) / scs[0];
    int sy0 = cy0 / scs[1], sy1 = (cy1 + scs[1] - 1) / scs[1];
    int sx0 = cx0 / scs[2], sx1 = (cx1 + scs[2] - 1) / scs[2];

    auto out = std::make_shared<ChunkData>();
    out->shape = {C, C, C};
    out->elementSize = 1;
    out->bytes.assign(size_t(C) * C * C, 0);
    bool anyData = false;

    for (int siz = sz0; siz < sz1; ++siz)
    for (int siy = sy0; siy < sy1; ++siy)
    for (int six = sx0; six < sx1; ++six) {
        ChunkKey srcKey{canonKey.level, siz, siy, six};
        std::vector<uint8_t> compressed;
        try { compressed = source_->fetch(srcKey); } catch (...) { continue; }
        if (compressed.empty() || isAllZero(compressed.data(), compressed.size())) continue;
        auto data = decompress_(compressed, srcKey);
        if (!data) continue;
        anyData = true;

        int svz = siz * scs[0], svy = siy * scs[1], svx = six * scs[2];
        int oz0 = std::max(svz,          cz0), oz1 = std::min(svz + scs[0], cz1);
        int oy0 = std::max(svy,          cy0), oy1 = std::min(svy + scs[1], cy1);
        int ox0 = std::max(svx,          cx0), ox1 = std::min(svx + scs[2], cx1);
        if (oz1 <= oz0 || oy1 <= oy0 || ox1 <= ox0) continue;

        const uint8_t* src = data->rawData();
        int srcStrideZ = data->strideZ(), srcStrideY = data->strideY();
        int runLen = ox1 - ox0;
        uint8_t* dst = out->rawData();

        for (int z = oz0; z < oz1; ++z)
        for (int y = oy0; y < oy1; ++y) {
            int srcLz = z - svz, srcLy = y - svy, srcLx0 = ox0 - svx;
            int canLz = z - cz0, canLy = y - cy0, canLx0 = ox0 - cx0;
            const uint8_t* s = src + size_t(srcLz) * srcStrideZ
                                   + size_t(srcLy) * srcStrideY + srcLx0;
            uint8_t*       d = dst + size_t(canLz) * C * C
                                   + size_t(canLy) * C + canLx0;
            std::memcpy(d, s, runLen);
        }
    }

    if (!anyData) return nullptr;
    return out;
}

ChunkDataPtr BlockPipeline::fetchChunkBlocking(const ChunkKey& key) {
    if (isNegativeCached(key)) return nullptr;

    auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;

    // Canonical disk tier first.
    if (dz) {
        if (dz->is_sharded() && !dz->inner_chunk_exists(chunkIndices(key))) {
            bloomAdd(key);
            std::lock_guard lock(negativeMutex_);
            negativeCache_.insert(key);
            return nullptr;
        }
        auto raw = zarrReadChunk(*dz, key);
        if (raw && !raw->empty()) {
            statColdHits_.fetch_add(1, std::memory_order_relaxed);
            std::vector<uint8_t> compressed(
                reinterpret_cast<const uint8_t*>(raw->data()),
                reinterpret_cast<const uint8_t*>(raw->data() + raw->size()));
            auto data = decodeCanonicalH265(compressed);
            if (data) return data;
        }
    }

    // No canonical disk tier (local source): fetch source chunk directly.
    if (!dz) {
        if (!source_ || !decompress_) return nullptr;
        std::vector<uint8_t> compressed;
        try { compressed = source_->fetch(key); } catch (...) { return nullptr; }
        if (compressed.empty() || isAllZero(compressed.data(), compressed.size())) {
            bloomAdd(key);
            std::lock_guard lock(negativeMutex_);
            negativeCache_.insert(key);
            return nullptr;
        }
        statIceFetches_.fetch_add(1, std::memory_order_relaxed);
        return decompress_(compressed, key);
    }

    // Canonical disk tier: rechunk from source and persist as h265.
    auto data = assembleCanonicalChunk(key);
    if (!data) {
        bloomAdd(key);
        {
            std::lock_guard lock(negativeMutex_);
            negativeCache_.insert(key);
        }
        if (dz->is_sharded()) dz->mark_inner_chunk_empty(chunkIndices(key));
        return nullptr;
    }
    statIceFetches_.fetch_add(1, std::memory_order_relaxed);

    auto h265 = encodeCanonicalH265(*data);
    zarrWriteChunk(*dz, key,
        reinterpret_cast<const uint8_t*>(h265.data()), h265.size());
    statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
    return data;
}

BlockPtr BlockPipeline::getBlockingBlock(const BlockKey& key) {
    if (auto b = blockCache_.get(key)) {
        statBlockHits_.fetch_add(1, std::memory_order_relaxed);
        return b;
    }

    // Canonical disk tier implies canonical 128^3 chunks; else use source.
    auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
    std::array<int, 3> cs = dz ? std::array<int, 3>{128, 128, 128}
                               : chunkShape(key.level);
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
