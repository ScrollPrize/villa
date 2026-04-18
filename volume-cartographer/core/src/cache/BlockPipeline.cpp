#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/VolumeSource.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <utils/http_fetch.hpp>
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

// Two zarr sharded-index sentinel states we care about:
//   (0xFF..F, 0xFF..F) — "missing" : the default fill for a freshly-created
//       shard file. Semantically "dunno, not yet fetched" — we should retry.
//   (0xFF..E, 0)        — "empty"   : we positively confirmed the chunk is
//       absent/zero remotely. Skip re-fetching.
// inner_chunk_is_empty() specifically tests for the (0xFF..E, 0) sentinel,
// so it only returns true for the "confirmed empty" state.
static bool diskShardMarksChunkEmpty(utils::ZarrArray& dz, const ChunkKey& key) {
    if (!dz.is_sharded()) return false;
    return dz.inner_chunk_is_empty(chunkIndices(key));
}

ShardKey BlockPipeline::canonicalShardKey(const ChunkKey& key) const noexcept
{
    if (key.level < 0 || key.level >= int(diskLevels_.size())) return {};
    const auto* dz = diskLevels_[key.level].get();
    if (!dz || !dz->is_sharded()) return {key.level, 0, 0, 0};
    const auto& m = dz->metadata();
    auto bpcZ = m.sub_chunks_per_shard(0);
    auto bpcY = m.sub_chunks_per_shard(1);
    auto bpcX = m.sub_chunks_per_shard(2);
    ShardKey sk{key.level, 0, 0, 0};
    sk.sz = bpcZ ? key.iz / int(bpcZ) : key.iz;
    sk.sy = bpcY ? key.iy / int(bpcY) : key.iy;
    sk.sx = bpcX ? key.ix / int(bpcX) : key.ix;
    return sk;
}

size_t BlockPipeline::shardCacheBucketIndex(const ShardKey& sk) noexcept {
    // ShardKeyHash is already a good mix; take low bits mod bucket count.
    return ShardKeyHash{}(sk) & (kShardCacheBuckets - 1);
}

void BlockPipeline::shardCacheInsertLocked(
    ShardCacheBucket& b,
    const ShardKey& sk,
    std::shared_ptr<std::vector<std::byte>> bytes)
{
    if (!bytes || bytes->empty()) return;
    const size_t totalBudget = config_.shardCacheBytes;
    if (totalBudget == 0) return;
    const size_t entrySize = bytes->size();
    if (entrySize > totalBudget) return;  // single shard too big for total

    // Replace any existing entry for this key — release its bytes from
    // both bucket and global counters before accounting the new one.
    if (auto it = b.map.find(sk); it != b.map.end()) {
        const size_t oldSize = it->second->bytes ? it->second->bytes->size() : 0;
        b.bytes -= oldSize;
        shardCacheGlobalBytes_.fetch_sub(oldSize, std::memory_order_relaxed);
        b.lru.erase(it->second);
        b.map.erase(it);
    }
    // Evict LRU from THIS bucket until global total + new entry fits budget.
    // Evicting from our own bucket (rather than scanning all) preserves LRU
    // within a bucket; buckets that have been recently active naturally
    // retain their entries because unrelated buckets aren't touched here.
    while (!b.lru.empty()
        && shardCacheGlobalBytes_.load(std::memory_order_relaxed) + entrySize > totalBudget) {
        auto& victim = b.lru.back();
        const size_t vSize = victim.bytes ? victim.bytes->size() : 0;
        b.bytes -= vSize;
        shardCacheGlobalBytes_.fetch_sub(vSize, std::memory_order_relaxed);
        b.map.erase(victim.key);
        b.lru.pop_back();
    }
    b.lru.push_front({sk, std::move(bytes)});
    b.map[sk] = b.lru.begin();
    b.bytes += entrySize;
    shardCacheGlobalBytes_.fetch_add(entrySize, std::memory_order_relaxed);
}

const std::vector<std::byte>* BlockPipeline::shardBytesFor(
    const ChunkKey& key, utils::ZarrArray& dz)
{
    if (config_.shardCacheBytes == 0) return nullptr;
    const ShardKey sk = canonicalShardKey(key);
    auto& bucket = shardCacheBuckets_[shardCacheBucketIndex(sk)];

    // Thread-local "last shard seen" cache. Loader workers routinely
    // process runs of adjacent chunks that fall in the same shard; with
    // 256 loader threads contending on the cache, skipping the map+lock
    // for same-shard hits saves ~1-2% total CPU per profile. The
    // shared_ptr keeps bytes alive even if the LRU evicts them — we
    // return a raw pointer to the bytes so callers don't bump the
    // shared_ptr refcount (that was ~20% of CPU under 12-thread decode
    // because all threads hit the same canonical shard's control block).
    // Qualify with `this` so a volume swap (pipeline destroyed + recreated)
    // doesn't serve stale data from the old pipeline's shard cache.
    thread_local const BlockPipeline* tlOwner = nullptr;
    thread_local ShardKey tlLastShard{-1, -1, -1, -1};
    thread_local std::shared_ptr<std::vector<std::byte>> tlLastBytes;
    if (tlOwner == this && tlLastShard == sk && tlLastBytes) {
        statShardHits_.fetch_add(1, std::memory_order_relaxed);
        return tlLastBytes.get();
    }

    // Fast path: hit under the bucket mutex, move entry to LRU head.
    {
        std::lock_guard lk(bucket.mutex);
        auto it = bucket.map.find(sk);
        if (it != bucket.map.end()) {
            bucket.lru.splice(bucket.lru.begin(), bucket.lru, it->second);
            statShardHits_.fetch_add(1, std::memory_order_relaxed);
            tlOwner = this;
            tlLastShard = sk;
            tlLastBytes = it->second->bytes;
            return tlLastBytes.get();
        }
    }

    // Miss — do the disk read outside the cache lock so concurrent hits
    // on other shards aren't blocked behind our I/O.
    statShardMisses_.fetch_add(1, std::memory_order_relaxed);
    auto raw = dz.read_whole_shard(chunkIndices(key));
    if (!raw || raw->empty()) return nullptr;
    auto bytes = std::make_shared<std::vector<std::byte>>(std::move(*raw));

    std::lock_guard lk(bucket.mutex);
    // Another thread may have raced us to populate this shard; prefer
    // the entry already in the cache to keep `shared_ptr` identity
    // consistent across concurrent reads.
    if (auto it = bucket.map.find(sk); it != bucket.map.end()) {
        bucket.lru.splice(bucket.lru.begin(), bucket.lru, it->second);
        tlOwner = this;
        tlLastShard = sk;
        tlLastBytes = it->second->bytes;
        return tlLastBytes.get();
    }
    shardCacheInsertLocked(bucket, sk, bytes);
    tlOwner = this;
    tlLastShard = sk;
    tlLastBytes = std::move(bytes);
    return tlLastBytes.get();
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
    const size_t n = size_t(dims[0]) * dims[1] * dims[2];
    auto out = std::make_unique<ChunkData>();
    out->shape = {int(dims[0]), int(dims[1]), int(dims[2])};
    out->elementSize = 1;
    out->bytes.resize(n);
    // Zero-copy: decoder writes straight into ChunkData::bytes. Saves one
    // ~2 MiB std::vector<std::byte> allocation + zero-init + memcpy per
    // chunk vs. the std::vector-returning video_decode — this path runs
    // on every decoded chunk so the allocation churn was a primary
    // driver of worker-thread page-fault / swap pressure.
    utils::video_decode_into(
        bytes,
        std::span<std::byte>(reinterpret_cast<std::byte*>(out->bytes.data()), n),
        vp);
    return out;
}

// Encode decoded chunk bytes as canonical h265 into a thread-local output
// buffer, returned by reference. Caller must consume the bytes before
// the next call on the same thread. Capacity grows once to the largest
// chunk ever seen on this thread and stays — no per-chunk allocation.
static const std::vector<std::byte>& encodeCanonicalH265(
    const ChunkData& chunk, const utils::VideoCodecParams& base) {
    thread_local std::vector<std::byte> tlEncoded;
    utils::VideoCodecParams vp = base;
    vp.depth = chunk.shape[0];
    vp.height = chunk.shape[1];
    vp.width = chunk.shape[2];
    utils::video_encode_into(
        {reinterpret_cast<const std::byte*>(chunk.rawData()), chunk.totalBytes()},
        vp, tlEncoded);
    return tlEncoded;
}

// Does `bytes` already carry the canonical VC3D/h265 magic header?
// If so we can skip the decode+re-encode cycle and passthrough the bytes
// to the canonical disk directly.
static bool bytesAreCanonicalH265(const std::vector<uint8_t>& bytes) {
    return utils::is_video_compressed(
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()));
}

BlockPipeline::BlockPipeline(
    Config config,
    std::unique_ptr<VolumeSource> source,
    DecompressFn decompress,
    std::vector<std::unique_ptr<utils::ZarrArray>> diskLevels)
    : config_(std::move(config))
    , diskLevels_(std::move(diskLevels))
    , source_(std::move(source))
    , decompress_(std::move(decompress))
    // Thread sizing. Loader is pure I/O wait (disk read or shard-cache
    // memcpy) — oversubscribe heavily so it can keep the disk queue full
    // even when many threads are blocked on io_wait. The CPU-bound pools
    // (encode, decode) stay near hw so concurrent decode never runs more
    // threads than cores. CLI tools (e.g. vc_zarr_prefetch) can still
    // override the downloader count via setIOThreads().
    //   downloader: HTTP fetch + re-chunk (I/O bound, keep-alive pool)
    //   encode    : h265 encode → disk (CPU, cold-miss only)
    //   loader    : shard-cache memcpy or disk read → staged bytes (I/O)
    //   decode    : h265 decode + block insert (pure CPU, caps work rate)
    , downloaderPool_([&] {
        if (config_.ioThreads > 0) return config_.ioThreads;
        unsigned hw = std::thread::hardware_concurrency();
        return hw ? static_cast<int>(hw) : 8;
    }())
    , encodePool_([] {
        unsigned hw = std::thread::hardware_concurrency();
        return hw ? static_cast<int>(hw) : 8;
    }())
    , loaderPool_([] {
        unsigned hw = std::thread::hardware_concurrency();
        // 16× hw: loader is almost entirely pread() io_wait. Observed
        // disk utilisation was ~1% at 4× — we're leaving most of the
        // device's queue depth on the table. Each worker's stack is
        // virtual-only until touched, so going wide is cheap.
        return hw ? 16 * static_cast<int>(hw) : 128;
    }())
    , decodePool_([] {
        unsigned hw = std::thread::hardware_concurrency();
        return hw ? static_cast<int>(hw) : 8;
    }())
    , blockCache_([&] {
        // Reserve a small residency floor per pyramid level so a coarse
        // fallback image is always available even under heavy fine-level
        // pressure. 4096 blocks = 16 MiB per level = ~32 canonical 128^3
        // chunks — enough for a viewport-worth at any level. Floors are
        // clamped inside BlockCache if they'd overwhelm the arena.
        BlockCache::Config bcfg;
        bcfg.bytes = config_.bytes;
        for (auto& f : bcfg.levelFloor) f = 4096;
        return bcfg;
      }())
{
    // Clear any stale process-wide HTTP abort flag from a previous
    // BlockPipeline's destructor. Without this, a volume swap during the
    // same session would leave the flag set and every subsequent curl
    // request would return CURLE_ABORTED_BY_CALLBACK → silent failure.
    utils::HttpClient::resetAbort();

    // Scan the on-disk cache once at startup so the stats bar reports
    // actual usage instead of "0 GB / 0 shards" until we write something.
    // Counts every regular file under each level root (each shard is one
    // file in zarr v3 sharded layout). zarr.json is excluded.
    for (const auto& dz : diskLevels_) {
        if (!dz) continue;
        const auto& root = dz->path();
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) continue;
        std::filesystem::recursive_directory_iterator it(
            root, std::filesystem::directory_options::skip_permission_denied, ec);
        std::filesystem::recursive_directory_iterator end;
        for (; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            if (it->path().filename() == "zarr.json") continue;
            auto sz = it->file_size(ec);
            if (ec) { ec.clear(); continue; }
            initialDiskBytes_ += sz;
            initialDiskShards_ += 1;
        }
    }
    if (initialDiskShards_ > 0) {
        fprintf(stderr, "[BlockPipeline] disk cache scan: %zu shards, %.1f MB\n",
                initialDiskShards_, double(initialDiskBytes_) / (1024.0 * 1024.0));
    }

    // Shard mapper is identity for all three pools: each canonical chunk
    // is its own work unit. HttpSource's internal shard cache still
    // amortizes S3 GETs across chunks that fall in the same source shard.
    auto shardMapper = [](const ChunkKey& key) -> ShardKey {
        return {key.level, key.iz, key.iy, key.ix};
    };
    downloaderPool_.setShardMapper(shardMapper);
    encodePool_.setShardMapper(shardMapper);
    loaderPool_.setShardMapper(shardMapper);

    // Downloader: network fetch + source-codec decode + re-chunk into a
    // canonical 128³ ChunkData. Stages the decoded buffer and hands the
    // key to the encoder. Does NOT touch disk, h265, or the block cache —
    // so one thread can return to fetching the next chunk as quickly as
    // possible.
    downloaderPool_.setFetchFunc([this](const ShardKey& shard) -> IOPool::FetchResult {
        ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};
        if (isNegativeCached(key)) return {};

        auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        if (!dz || !source_) return {};

        if (diskShardMarksChunkEmpty(*dz, key)) {
            bloomAdd(key);
            std::lock_guard lock(negativeMutex_);
            negativeCache_.insert(key);
            return {};
        }

        // Pull source chunks over the network and assemble a canonical
        // 128³ buffer. Source decode happens here too because the
        // re-chunking needs the voxels; it's a small fraction of the
        // overall work compared to x265 encode, which is now off-thread.
        auto decoded = assembleCanonicalChunk(key);
        if (!decoded) {
            // Negative-cache *only* when the source confirms the chunk is
            // genuinely absent — a real S3 404, or a sharded v3 missing/
            // zero-placeholder index entry. Transient errors / curl
            // failures / auth issues must not poison the cache.
            const bool isHttp = dynamic_cast<HttpSource*>(source_.get()) != nullptr;
            const bool absent = !isHttp || HttpSource::lastFetchWasAbsent();
            if (absent) {
                bloomAdd(key);
                std::lock_guard lock(negativeMutex_);
                negativeCache_.insert(key);
                if (dz->is_sharded()) dz->mark_inner_chunk_empty(chunkIndices(key));
            }
            return {};
        }
        statIceFetches_.fetch_add(1, std::memory_order_relaxed);

        // Stage for the encoder.
        {
            std::lock_guard lk(encodeStagingMutex_);
            encodeStaging_[key] = std::move(decoded);
        }
        IOPool::FetchResult result;
        result.emplace_back(key, std::vector<uint8_t>{});
        return result;
    });

    downloaderPool_.setCompletionCallback(
        [this](IOPool::FetchResult&& res) {
            if (res.empty()) return;
            std::vector<ChunkKey> encodeKeys;
            encodeKeys.reserve(res.size());
            for (auto& [key, _] : res) encodeKeys.push_back(key);
            // submit appends (O(N) in new keys); updateInteractive would
            // reshuffle the whole encoder queue (O(Q)) and reset served
            // counters, neither of which makes sense for inter-stage
            // handoffs — the viewport priority comes from the renderer's
            // fetchInteractive call upstream, not from completions.
            encodePool_.submit(encodeKeys);
        });

    // Encoder: take staged ChunkData → h265 encode → disk write → forward
    // the key to the loader. Pure CPU work plus a small disk write;
    // oversubscribing this pool just thrashes cores, so it runs at 1×
    // hardware_concurrency.
    encodePool_.setFetchFunc([this](const ShardKey& shard) -> IOPool::FetchResult {
        ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};
        auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        if (!dz) return {};

        ChunkDataPtr decoded;
        {
            std::lock_guard lk(encodeStagingMutex_);
            auto it = encodeStaging_.find(key);
            if (it == encodeStaging_.end()) return {};
            decoded = std::move(it->second);
            encodeStaging_.erase(it);
        }
        if (!decoded) return {};

        const auto& h265 = encodeCanonicalH265(*decoded, config_.encodeParams);
        zarrWriteChunk(*dz, key,
            reinterpret_cast<const uint8_t*>(h265.data()), h265.size());
        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
        statDiskBytes_.fetch_add(h265.size(), std::memory_order_relaxed);
        if (dz->is_sharded()) {
            const ShardKey sk = canonicalShardKey(key);
            {
                std::lock_guard lk(writtenShardsMutex_);
                writtenShards_.insert(sk);
            }
            // Shard file grew on disk; drop the stale cached copy so the
            // loader re-reads it next time.
            auto& bucket = shardCacheBuckets_[shardCacheBucketIndex(sk)];
            std::lock_guard lk(bucket.mutex);
            if (auto it = bucket.map.find(sk); it != bucket.map.end()) {
                const size_t sz = it->second->bytes ? it->second->bytes->size() : 0;
                bucket.bytes -= sz;
                shardCacheGlobalBytes_.fetch_sub(sz, std::memory_order_relaxed);
                bucket.lru.erase(it->second);
                bucket.map.erase(it);
            }
        }

        IOPool::FetchResult result;
        result.emplace_back(key, std::vector<uint8_t>{});
        return result;
    });

    encodePool_.setCompletionCallback(
        [this](IOPool::FetchResult&& res) {
            if (res.empty()) return;
            std::vector<ChunkKey> loaderKeys;
            loaderKeys.reserve(res.size());
            for (auto& [key, _] : res) loaderKeys.push_back(key);
            loaderPool_.submit(loaderKeys);
        });

    // Loader: read compressed bytes for `key` (shard RAM cache or disk),
    // stage them for decodePool_. Pure I/O + memcpy — no CPU-heavy work
    // so this pool can be oversubscribed to keep the disk queue full.
    loaderPool_.setFetchFunc([this](const ShardKey& shard) -> IOPool::FetchResult {
        ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};
        if (isNegativeCached(key)) return {};

        auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        std::vector<uint8_t> compressed;

        if (dz) {
            if (diskShardMarksChunkEmpty(*dz, key)) {
                bloomAdd(key);
                std::lock_guard lock(negativeMutex_);
                negativeCache_.insert(key);
                return {};
            }
            // Route through the shard cache. Subsequent inner chunks from
            // the same shard served from RAM with no syscalls.
            std::optional<std::vector<std::byte>> innerBytes;
            if (dz->is_sharded() && config_.shardCacheBytes > 0) {
                auto shardBytes = shardBytesFor(key, *dz);
                if (shardBytes) {
                    auto idx = chunkIndices(key);
                    const auto& m = dz->metadata();
                    std::vector<std::size_t> inner(idx.size());
                    for (size_t d = 0; d < idx.size(); ++d) {
                        auto ips = m.sub_chunks_per_shard(d);
                        inner[d] = ips ? idx[d] % ips : idx[d];
                    }
                    innerBytes = dz->extract_inner_chunk(*shardBytes, inner);
                }
            } else {
                innerBytes = zarrReadChunk(*dz, key);
            }
            if (innerBytes && !innerBytes->empty()) {
                statColdHits_.fetch_add(1, std::memory_order_relaxed);
                compressed.assign(
                    reinterpret_cast<const uint8_t*>(innerBytes->data()),
                    reinterpret_cast<const uint8_t*>(innerBytes->data() + innerBytes->size()));
            }
        } else if (source_) {
            // Local source, no disk tier: treat the source files as disk.
            try { compressed = source_->fetch(key); } catch (...) { return {}; }
            if (compressed.empty() || isAllZero(compressed.data(), compressed.size())) {
                // Same rule as the downloader: only poison the negative
                // cache when the source confirms genuine absence.
                const bool isHttp = dynamic_cast<HttpSource*>(source_.get()) != nullptr;
                const bool absent = !isHttp || HttpSource::lastFetchWasAbsent();
                if (absent) {
                    bloomAdd(key);
                    std::lock_guard lock(negativeMutex_);
                    negativeCache_.insert(key);
                }
                return {};
            }
            statColdHits_.fetch_add(1, std::memory_order_relaxed);
        }

        if (compressed.empty()) return {};

        // Stage compressed bytes for decodePool_ so the loader worker can
        // immediately serve the next I/O request while CPU decode proceeds
        // in parallel. Decode concurrency is bounded by decodePool_ size.
        {
            std::lock_guard stage(decodeStagingMutex_);
            decodeStaging_[key] = std::move(compressed);
        }
        // Non-empty FetchResult so IOPool fires the completion callback,
        // which forwards this key to decodePool_. Payload unused.
        IOPool::FetchResult result;
        result.emplace_back(key, std::vector<uint8_t>{});
        return result;
    });

    loaderPool_.setCompletionCallback(
        [this](IOPool::FetchResult&& res) {
            if (res.empty()) return;
            std::vector<ChunkKey> decodeKeys;
            decodeKeys.reserve(res.size());
            for (auto& [k, _] : res) decodeKeys.push_back(k);
            decodePool_.submit(decodeKeys);
        });

    decodePool_.setShardMapper(shardMapper);

    // Decoder: staged compressed bytes → h265 decode → block cache insert →
    // fire chunk-ready callbacks. Pure CPU; concurrency here caps CPU load.
    decodePool_.setFetchFunc([this](const ShardKey& shard) -> IOPool::FetchResult {
        ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};
        std::vector<uint8_t> compressed;
        {
            std::lock_guard stage(decodeStagingMutex_);
            auto it = decodeStaging_.find(key);
            if (it == decodeStaging_.end()) return {};
            compressed = std::move(it->second);
            decodeStaging_.erase(it);
        }
        if (compressed.empty()) return {};

        ChunkDataPtr decoded;
        auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        if (dz) {
            decoded = decodeCanonicalH265(compressed);
        } else if (decompress_) {
            decoded = decompress_(compressed, key);
        }
        if (!decoded) return {};

        insertChunkAsBlocks(key, *decoded);
        if (!chunkArrivedFlag_.exchange(true, std::memory_order_acq_rel)) {
            // Snapshot callbacks under lock and release before firing so
            // a slow listener can't serialize the decode hot path or
            // deadlock with add/remove calls that need the same mutex.
            std::vector<ChunkReadyCallback> snapshot;
            {
                std::lock_guard cbLock(callbackMutex_);
                snapshot.reserve(chunkReadyListeners_.size());
                for (const auto& [id, cb] : chunkReadyListeners_)
                    snapshot.push_back(cb);
            }
            for (const auto& cb : snapshot) cb(key);
        }
        return {};
    });

    decodePool_.setCompletionCallback(
        [](IOPool::FetchResult&&) {
            // Work already done inside the fetch func.
        });

    // Canonical-passthrough override. When the source is byte-identical to
    // our local layout (zarr v3, 128^3 inner H.265 chunks, matching shard
    // shape), the downloader skips decode/re-encode entirely: pull the
    // chunk bytes from source (HttpSource's internal shardCache_ amortises
    // to one S3 GET per source shard), write verbatim to the local sharded
    // zarr, forward the chunk key straight to the loader. Counters stay
    // chunk-denominated because the shardMapper remains identity.
    if (config_.canonicalSourceShard[0] != 0
        && config_.canonicalSourceShard[1] != 0
        && config_.canonicalSourceShard[2] != 0)
    {
        const std::array<int, 3> srcShard = config_.canonicalSourceShard;
        for (auto& dz : diskLevels_) {
            if (!dz || !dz->is_sharded()) continue;
            const auto& m = dz->metadata();
            if (m.chunks.size() < 3
                || int(m.chunks[0]) != srcShard[0]
                || int(m.chunks[1]) != srcShard[1]
                || int(m.chunks[2]) != srcShard[2]) {
                std::fprintf(stderr,
                    "[BlockPipeline] canonicalSourceShard %dx%dx%d does "
                    "not match local shard shape %zux%zux%zu — disabling "
                    "passthrough\n",
                    srcShard[0], srcShard[1], srcShard[2],
                    m.chunks.size() > 0 ? m.chunks[0] : 0,
                    m.chunks.size() > 1 ? m.chunks[1] : 0,
                    m.chunks.size() > 2 ? m.chunks[2] : 0);
                goto skipPassthrough;
            }
        }

        downloaderPool_.setFetchFunc(
            [this](const ShardKey& shard) -> IOPool::FetchResult {
                ChunkKey key{shard.level, shard.sz, shard.sy, shard.sx};
                if (isNegativeCached(key)) return {};
                auto* dz = (key.level < int(diskLevels_.size()))
                    ? diskLevels_[key.level].get() : nullptr;
                if (!dz || !source_) return {};
                if (diskShardMarksChunkEmpty(*dz, key)) {
                    bloomAdd(key);
                    std::lock_guard lock(negativeMutex_);
                    negativeCache_.insert(key);
                    return {};
                }

                // Pull the canonical h265 bytes from source.
                // HttpSource::fetch caches the whole source shard
                // internally, so concurrent siblings hit the cache.
                std::vector<uint8_t> bytes;
                try { bytes = source_->fetch(key); }
                catch (...) { return {}; }
                if (bytes.empty()) {
                    // Negative-cache only on confirmed source-side absence
                    // (real 404 or sharded-v3 missing/zero placeholder).
                    if (HttpSource::lastFetchWasAbsent()) {
                        bloomAdd(key);
                        std::lock_guard lock(negativeMutex_);
                        negativeCache_.insert(key);
                        if (dz->is_sharded()) dz->mark_inner_chunk_empty(chunkIndices(key));
                    }
                    return {};
                }

                // Sanity: bytes must already be VC3D/H.265. If not, the
                // source advertised canonical structure but serves blosc
                // or raw — we can't use it. Mark the chunk negative so
                // we stop re-fetching it every render; otherwise every
                // fetchInteractive would re-queue it forever.
                if (!utils::is_video_compressed(std::span<const std::byte>(
                        reinterpret_cast<const std::byte*>(bytes.data()),
                        bytes.size()))) {
                    // Always log: these are source-level corruption events,
                    // not transient noise. Rate-limiting to 5 previously hid
                    // systemic codec-mismatch issues from users.
                    std::fprintf(stderr,
                        "[BlockPipeline] passthrough: chunk lvl=%d "
                        "(%d,%d,%d) lacks VC3D magic — marking absent\n",
                        key.level, key.iz, key.iy, key.ix);
                    bloomAdd(key);
                    {
                        std::lock_guard lock(negativeMutex_);
                        negativeCache_.insert(key);
                    }
                    if (dz->is_sharded()) dz->mark_inner_chunk_empty(chunkIndices(key));
                    return {};
                }

                statIceFetches_.fetch_add(1, std::memory_order_relaxed);
                zarrWriteChunk(*dz, key, bytes.data(), bytes.size());
                statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
                statDiskBytes_.fetch_add(bytes.size(), std::memory_order_relaxed);
                if (dz->is_sharded()) {
                    const ShardKey sk = canonicalShardKey(key);
                    {
                        std::lock_guard lk(writtenShardsMutex_);
                        writtenShards_.insert(sk);
                    }
                    auto& bucket = shardCacheBuckets_[shardCacheBucketIndex(sk)];
                    std::lock_guard lk(bucket.mutex);
                    if (auto it = bucket.map.find(sk); it != bucket.map.end()) {
                        const size_t sz = it->second->bytes ? it->second->bytes->size() : 0;
                        bucket.bytes -= sz;
                        shardCacheGlobalBytes_.fetch_sub(sz, std::memory_order_relaxed);
                        bucket.lru.erase(it->second);
                        bucket.map.erase(it);
                    }
                }

                IOPool::FetchResult result;
                result.emplace_back(key, std::vector<uint8_t>{});
                return result;
            });

        // Skip the encoder entirely — bytes are already canonical.
        downloaderPool_.setCompletionCallback(
            [this](IOPool::FetchResult&& res) {
                if (res.empty()) return;
                std::vector<ChunkKey> keys;
                keys.reserve(res.size());
                for (auto& [k, _] : res) keys.push_back(k);
                loaderPool_.submit(keys);
            });
    }
skipPassthrough:

    downloaderPool_.start();
    encodePool_.start();
    loaderPool_.start();
    decodePool_.start();
}

BlockPipeline::~BlockPipeline() {
    // Cancel any in-flight curl requests so workers don't sit inside
    // libcurl waiting for S3 timeouts during shutdown.
    utils::HttpClient::abortAll();
    // Drop queued work before joining so we don't waste cycles starting
    // new fetches that are about to be aborted anyway.
    downloaderPool_.cancelPending();
    encodePool_.cancelPending();
    loaderPool_.cancelPending();
    decodePool_.cancelPending();
    // Stop upstream-first so no new work lands in downstream queues while
    // we're trying to drain them.
    downloaderPool_.stop();
    encodePool_.stop();
    loaderPool_.stop();
    decodePool_.stop();
    auto cold = statColdHits_.load();
    auto ice = statIceFetches_.load();
    if (cold > 0 || ice > 0) {
        std::fprintf(stderr, "[Cache] session summary: coldHits=%lu iceFetches=%lu (%.0f%% from disk)\n",
                     cold, ice, (cold + ice) > 0 ? 100.0 * cold / (cold + ice) : 0.0);
    }
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

void BlockPipeline::fetchInteractive(const std::vector<ChunkKey>& keys, int targetLevel) {
    if (keys.empty()) return;
    // Dedup: the renderer calls this every frame, and viewport-idle frames
    // pass the same keys + targetLevel as the previous call. When the
    // BlockCache hasn't evicted anything since the last submit, nothing
    // downstream would change — the expensive containsBatch, emptyChunks
    // snapshot, classification, and (most importantly) IOPool queue
    // rebuilds would reproduce their previous outputs. Skip.
    {
        // Commutative hash so order-insensitive dedup works across
        // render paths that enumerate chunks in different orders.
        uint64_t h = uint64_t(uint32_t(targetLevel)) * 0x9E3779B97F4A7C15ULL;
        ChunkKeyHash kh;
        for (const auto& k : keys) h ^= kh(k);
        const uint64_t evictionNow = blockCache_.evictionVersion();
        std::lock_guard lk(fetchInteractiveDedupMutex_);
        if (haveLastFetchInteractive_
            && lastFetchInteractiveHash_ == h
            && lastFetchInteractiveEviction_ == evictionNow
            && lastFetchInteractiveTargetLevel_ == targetLevel) {
            return;
        }
        lastFetchInteractiveHash_ = h;
        lastFetchInteractiveEviction_ = evictionNow;
        lastFetchInteractiveTargetLevel_ = targetLevel;
        haveLastFetchInteractive_ = true;
    }
    // Triage by disk presence: chunks already on disk go straight to the
    // loader pool (fast, CPU-bound decode), chunks that need fetching go
    // to the downloader pool (slow, network-bound). The two pools are
    // fully independent so the loader can't be starved by in-flight S3
    // work. Before either, skip chunks already resident in the block
    // cache — otherwise IOPool re-queues every viewport chunk every frame
    // (Done shards are re-queueable for eviction recovery) and loader
    // workers endlessly re-decode data we already have.
    constexpr int kMaxL = 16;
    std::array<int, kMaxL> bpcZ{}, bpcY{}, bpcX{};
    std::array<bool, kMaxL> shapeCached{};
    auto ensureShapeCached = [&](int level) {
        if (level < 0 || level >= kMaxL) return;
        if (!shapeCached[level] && source_) {
            auto csk = chunkShape(level);
            if (csk[0] > 0) {
                bpcZ[level] = csk[0] / kBlockSize;
                bpcY[level] = csk[1] / kBlockSize;
                bpcX[level] = csk[2] / kBlockSize;
            }
            shapeCached[level] = true;
        }
    };

    // Build per-key BlockKey pairs (first + last block of each chunk) and
    // batch-query the block cache. Checking two blocks instead of one
    // catches partial eviction: if the first block survives the clock
    // sweep but interior blocks were reclaimed, a single-block check
    // would falsely skip the chunk, leaving visual holes until the first
    // block is also evicted.
    std::vector<BlockKey> probeKeys;
    probeKeys.reserve(keys.size() * 2);
    for (const auto& key : keys) {
        if (key.level < 0 || key.level >= kMaxL) {
            probeKeys.push_back({-1, 0, 0, 0});
            probeKeys.push_back({-1, 0, 0, 0});
            continue;
        }
        ensureShapeCached(key.level);
        const int z = bpcZ[key.level], y = bpcY[key.level], x = bpcX[key.level];
        if (z <= 0 || y <= 0 || x <= 0) {
            probeKeys.push_back({-1, 0, 0, 0});
            probeKeys.push_back({-1, 0, 0, 0});
            continue;
        }
        // First block of the chunk
        probeKeys.push_back({key.level, key.iz * z, key.iy * y, key.ix * x});
        // Last block of the chunk
        probeKeys.push_back({key.level, key.iz * z + z - 1,
                                        key.iy * y + y - 1,
                                        key.ix * x + x - 1});
    }
    std::vector<uint8_t> resident;
    blockCache_.containsBatch(probeKeys, resident);

    // Empty-chunks lookup is now lock-free per probe — no snapshot needed.
    std::vector<ChunkKey> loaderKeys, downloaderKeys;
    loaderKeys.reserve(keys.size());
    downloaderKeys.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto& key = keys[i];
        if (isNegativeCached(key)) continue;
        if (isEmptyChunk(key)) continue;
        // Both first and last block of the chunk must be resident to
        // consider the chunk fully cached. See comment above probeKeys.
        if (probeKeys[i * 2].level >= 0
            && resident[i * 2] && resident[i * 2 + 1]) continue;

        auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        const bool diskPresent = dz
            && dz->is_sharded()
            && dz->inner_chunk_exists(chunkIndices(key));
        if (diskPresent || !dz) {
            // Present on canonical disk, OR no canonical disk tier at all
            // (local filesystem source — the "disk" is the source files).
            loaderKeys.push_back(key);
        } else {
            downloaderKeys.push_back(key);
        }
    }
    if (!loaderKeys.empty())
        loaderPool_.updateInteractive(loaderKeys, targetLevel);
    if (!downloaderKeys.empty())
        downloaderPool_.updateInteractive(downloaderKeys, targetLevel);
}

BlockPtr BlockPipeline::blockAt(const BlockKey& key) noexcept {
    // Hot path: try the block cache first. Most samples hit a resident
    // block, so emptyChunksMutex_ never needs to be acquired — previously
    // every blockAt() call took that mutex before the cache lookup.
    // BlockCache::get() now owns the hit counter (per-shard, no contention);
    // we no longer bump statBlockHits_ here on the fast path.
    if (auto b = blockCache_.get(key); b) return b;
    // Miss: could be an "empty chunk" (all-zero canonical chunk that we
    // don't store). Canonical chunks are 128³ = 8x8x8 blocks of 16³ —
    // reverse-map the block coord to its enclosing chunk and check via
    // the lock-free hash set. No rwlock — every blockAt miss used to hit
    // pthread_rwlock_rdlock here.
    const ChunkKey chunkKey{key.level, key.bz / 8, key.by / 8, key.bx / 8};
    if (isEmptyChunk(chunkKey)) {
        // One canonical zero block shared by every caller asking for
        // a block inside any empty chunk — no arena consumption.
        static constinit Block kZeroBlock{};
        // Per-thread local accumulator flushed every 1024 hits. A single
        // global atomic fetch_add here burned ~5%+ of CPU on hot render
        // workloads (12 threads ping-ponging one cacheline per miss).
        thread_local uint64_t localEmptyHits = 0;
        if ((++localEmptyHits & 1023) == 0)
            statEmptyHits_.fetch_add(1024, std::memory_order_relaxed);
        return &kZeroBlock;
    }
    thread_local uint64_t localMisses = 0;
    if ((++localMisses & 1023) == 0)
        statMisses_.fetch_add(1024, std::memory_order_relaxed);
    return nullptr;
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

    auto out = std::make_unique<ChunkData>();
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


void BlockPipeline::insertChunkAsBlocks(const ChunkKey& key,
                                        const ChunkData& chunk) {
    const int cz = chunk.shape[0];
    const int cy = chunk.shape[1];
    const int cx = chunk.shape[2];
    if (cz <= 0 || cy <= 0 || cx <= 0) return;
    const int bzN = cz / kBlockSize;
    const int byN = cy / kBlockSize;
    const int bxN = cx / kBlockSize;
    // Blocks are the fixed 16³ storage/render unit; chunk dims must be a
    // multiple of kBlockSize so blocks tile the chunk cleanly. Enforced
    // up-front in FileSystemSource::discoverLevels — reaching here with a
    // misaligned chunk means a new source path slipped past the check.
    if (bzN * kBlockSize != cz || byN * kBlockSize != cy || bxN * kBlockSize != cx) {
        std::fprintf(stderr,
                     "[FATAL] chunk dims must be multiples of %d, got %dx%dx%d (key L%d %d/%d/%d)\n",
                     kBlockSize, cz, cy, cx, key.level, key.iz, key.iy, key.ix);
        std::abort();
    }

    // Zero-chunk shortcut: VcDecompressor already scanned the decoded bytes
    // and set isEmpty when every voxel is zero. Record the canonical chunk
    // key and skip copying 512 identical zero blocks into the arena —
    // blockAt() will hand out a shared static zero block for every inner
    // block of this chunk. Saves ~2 MB of arena per empty 128³ chunk.
    if (chunk.isEmpty) {
        addEmptyChunk(key);
        return;
    }

    const uint8_t* src = chunk.rawData();
    const int strideZ = chunk.strideZ();
    const int strideY = chunk.strideY();

    const int baseBz = key.iz * bzN;
    const int baseBy = key.iy * byN;
    const int baseBx = key.ix * bxN;

    // One unique_lock covers all 512 inserts for a 128³ canonical chunk
    // instead of 512 separate lock/unlock pairs. acquire() returns the
    // arena slot directly so we write the 16³ block straight into its
    // final destination — no tmp buffer, no double copy.
    BlockCache::BatchPut batch(blockCache_);
    for (int bi = 0; bi < bzN; ++bi) {
        for (int bj = 0; bj < byN; ++bj) {
            for (int bk = 0; bk < bxN; ++bk) {
                BlockKey bkKey{key.level, baseBz + bi, baseBy + bj, baseBx + bk};
                uint8_t* dst = batch.acquire(bkKey);
                if (!dst) continue;
                for (int lz = 0; lz < kBlockSize; ++lz) {
                    const uint8_t* zRow = src + (bi * kBlockSize + lz) * strideZ;
                    for (int ly = 0; ly < kBlockSize; ++ly) {
                        const uint8_t* p = zRow + (bj * kBlockSize + ly) * strideY + bk * kBlockSize;
                        std::memcpy(dst, p, kBlockSize);
                        dst += kBlockSize;
                    }
                }
            }
        }
    }
}


void BlockPipeline::clearMemory() {
    blockCache_.clear();
    clearEmptyChunks();
    for (auto& bucket : shardCacheBuckets_) {
        std::lock_guard lk(bucket.mutex);
        bucket.lru.clear();
        bucket.map.clear();
        bucket.bytes = 0;
    }
    shardCacheGlobalBytes_.store(0, std::memory_order_relaxed);
}

void BlockPipeline::clearAll() {
    downloaderPool_.cancelPending();
    encodePool_.cancelPending();
    loaderPool_.cancelPending();
    decodePool_.cancelPending();
    {
        std::lock_guard lk(encodeStagingMutex_);
        encodeStaging_.clear();
    }
    {
        std::lock_guard lk(decodeStagingMutex_);
        decodeStaging_.clear();
    }
    for (auto& bucket : shardCacheBuckets_) {
        std::lock_guard lk(bucket.mutex);
        bucket.lru.clear();
        bucket.map.clear();
        bucket.bytes = 0;
    }
    shardCacheGlobalBytes_.store(0, std::memory_order_relaxed);
    blockCache_.clear();
    clearEmptyChunks();
    bloomClear();
    std::lock_guard lock(negativeMutex_);
    negativeCache_.clear();
}

int BlockPipeline::numLevels() const noexcept {
    return source_ ? source_->numLevels() : 0;
}

std::array<int, 3> BlockPipeline::chunkShape(int level) const noexcept {
    if (!source_) return {0, 0, 0};
    // When a disk-tier exists for this level, chunks are canonicalized to
    // 128³ by assembleCanonicalChunk / insertChunkAsBlocks. The source's
    // native chunk size may differ (e.g., 256³); returning it would make
    // Slicing.cpp's chunk-key enumeration compute the wrong grid.
    if (level >= 0 && level < int(diskLevels_.size()) && diskLevels_[level])
        return {128, 128, 128};
    return source_->chunkShape(level);
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
    if (!source_) return 0;
    // Cache chunk shape per level. chunkShape() is virtual and typically
    // hits a std::array lookup; avoiding the per-key call matters when
    // countAvailable is invoked with thousands of keys per frame.
    constexpr int kMaxL = 16;
    std::array<int, kMaxL> bpcZ{}, bpcY{}, bpcX{};
    std::array<bool, kMaxL> shapeCached{};
    // Empty-chunk probe is now lock-free — no snapshot needed.
    for (const auto& key : keys) {
        if (isNegativeCached(key)) { n++; continue; }
        if (isEmptyChunk(key)) { n++; continue; }
        if (key.level < 0 || key.level >= kMaxL) continue;
        if (!shapeCached[key.level]) {
            auto csk = chunkShape(key.level);
            if (csk[0] <= 0) { shapeCached[key.level] = true; continue; }
            bpcZ[key.level] = csk[0] / kBlockSize;
            bpcY[key.level] = csk[1] / kBlockSize;
            bpcX[key.level] = csk[2] / kBlockSize;
            shapeCached[key.level] = true;
        }
        const int z = bpcZ[key.level], y = bpcY[key.level], x = bpcX[key.level];
        if (z <= 0 || y <= 0 || x <= 0) continue;
        // Check first + last block (matches fetchInteractive's two-block
        // probe) so partial eviction doesn't report the chunk as available.
        BlockKey first{key.level, key.iz * z, key.iy * y, key.ix * x};
        BlockKey last{key.level, key.iz * z + z - 1, key.iy * y + y - 1, key.ix * x + x - 1};
        if (blockCache_.contains(first) && blockCache_.contains(last)) n++;
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
    // Hot path hits live in BlockCache's per-shard counters; empty-chunk
    // canonical-zero hits are separate (cold path).
    s.blockHits = blockCache_.blockHits()
                + statEmptyHits_.load(std::memory_order_relaxed);
    s.coldHits = statColdHits_.load(std::memory_order_relaxed);
    s.iceFetches = statIceFetches_.load(std::memory_order_relaxed);
    s.misses = statMisses_.load(std::memory_order_relaxed);
    s.blocks = blockCache_.size();
    s.downloadPending = downloaderPool_.pendingCount();
    s.encodePending = encodePool_.pendingCount();
    s.loadPending = loaderPool_.pendingCount();
    s.decodePending = decodePool_.pendingCount();
    s.ioPending = s.downloadPending + s.encodePending + s.loadPending + s.decodePending;
    s.shardHits = statShardHits_.load(std::memory_order_relaxed);
    s.shardMisses = statShardMisses_.load(std::memory_order_relaxed);
    // Global byte count is an atomic so we don't need the per-bucket locks
    // to read it. Entry count still needs the per-bucket walk.
    s.shardCacheBytes = shardCacheGlobalBytes_.load(std::memory_order_relaxed);
    s.shardCacheEntries = 0;
    for (auto& bucket : shardCacheBuckets_) {
        std::lock_guard lk(bucket.mutex);
        s.shardCacheEntries += bucket.lru.size();
    }
    s.diskWrites = statDiskWrites_.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(negativeMutex_);
        s.negativeCount = negativeCache_.size();
    }
    s.totalSubmitted = statTotalSubmitted_.load(std::memory_order_relaxed);
    s.diskBytes = initialDiskBytes_
                + statDiskBytes_.load(std::memory_order_relaxed);
    {
        std::lock_guard lk(writtenShardsMutex_);
        s.diskShards = initialDiskShards_ + writtenShards_.size();
    }
    if (auto* http = dynamic_cast<HttpSource*>(source_.get()))
        s.sharded = http->isSharded();
    if (!s.sharded) {
        for (const auto& dz : diskLevels_)
            if (dz && dz->is_sharded()) { s.sharded = true; break; }
    }
    return s;
}

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
    cfg.bytes = maxBytes;
    return std::make_unique<BlockPipeline>(
        std::move(cfg), std::move(source), std::move(decompress));
}

}  // namespace vc::cache
