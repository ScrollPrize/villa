#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/TickCoordinator.hpp"
#include "vc/core/cache/VolumeSource.hpp"
#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <thread>
#include <utils/http_fetch.hpp>

namespace vc::cache {

static std::array<size_t, 3> chunkIndices(const ChunkKey& key) {
    return {size_t(key.iz), size_t(key.iy), size_t(key.ix)};
}

static std::optional<std::vector<std::byte>> zarrReadChunk(
    utils::ZarrArray& zarr, const ChunkKey& key) {
    return zarr.read_chunk(chunkIndices(key));
}

static void zarrWriteChunk(
    utils::ZarrArray& zarr, const ChunkKey& key,
    const uint8_t* data, size_t size) {
    std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data), size);
    zarr.write_chunk(chunkIndices(key), bytes);
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
    std::shared_ptr<utils::ShardBytes> bytes)
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

const utils::ShardBytes* BlockPipeline::shardBytesFor(
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
    thread_local std::shared_ptr<utils::ShardBytes> tlLastBytes;
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
    // Gate concurrent whole-shard reads. Each call returns a ~256 MiB
    // buffer; without the gate, 16×hw loader workers all missing at
    // once hold tens of GiB of in-flight shard bytes. RAII guard
    // decrements on every return path.
    {
        std::unique_lock lk(inflightShardMutex_);
        inflightShardCv_.wait(lk, [this] {
            return shuttingDown_.load(std::memory_order_acquire)
                || inflightShardReads_ < config_.maxConcurrentShardReads;
        });
        if (shuttingDown_.load(std::memory_order_acquire)) return nullptr;
        ++inflightShardReads_;
    }
    struct InflightGuard {
        BlockPipeline* self;
        size_t bytes = 0;
        ~InflightGuard() {
            self->inflightShardBytes_.fetch_sub(bytes, std::memory_order_relaxed);
            {
                std::lock_guard lk(self->inflightShardMutex_);
                --self->inflightShardReads_;
            }
            self->inflightShardCv_.notify_one();
        }
    } guard{this};
    auto raw = dz.read_whole_shard(chunkIndices(key));
    if (!raw || raw->empty()) return nullptr;
    guard.bytes = raw->size();
    inflightShardBytes_.fetch_add(guard.bytes, std::memory_order_relaxed);
    auto bytes = std::make_shared<utils::ShardBytes>(std::move(*raw));

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

// Canonical-chunk side is fixed at 256: c3d's codec atom size. The chunk
// grid enumeration in Slicing.cpp derives from this constant.
static constexpr int kCanonicalChunkSide = 256;

// Decode a canonical c3d chunk from disk bytes. Returns null if the bytes
// don't carry the C3DC magic (source corruption or wrong codec).
static ChunkDataPtr decodeCanonicalChunk(const std::vector<uint8_t>& compressed) {
    std::span<const std::byte> bytes(
        reinterpret_cast<const std::byte*>(compressed.data()), compressed.size());
    if (!utils::is_c3d_compressed(bytes)) return nullptr;
    utils::C3dCodecParams p;
    p.skip_denoise = true;
    const std::size_t n = size_t(kCanonicalChunkSide)
                        * size_t(kCanonicalChunkSide)
                        * size_t(kCanonicalChunkSide);
    auto decoded = utils::c3d_decode(bytes, n, p);
    auto out = vc::cache::acquireChunkData();
    out->shape = {kCanonicalChunkSide, kCanonicalChunkSide, kCanonicalChunkSide};
    out->elementSize = 1;
    out->bytes.resize(decoded.size());
    std::memcpy(out->bytes.data(), decoded.data(), decoded.size());
    return out;
}

// Encode decoded chunk bytes as canonical c3d into a thread-local output
// buffer, returned by reference. Caller must consume the bytes before the
// next call on the same thread. Capacity grows once to the largest chunk
// ever seen on this thread and stays — no per-chunk allocation.
static const std::vector<std::byte>& encodeCanonicalChunk(
    const ChunkData& chunk, const BlockPipeline::Config& cfg) {
    thread_local std::vector<std::byte> tlEncoded;
    std::span<const std::byte> raw(
        reinterpret_cast<const std::byte*>(chunk.rawData()), chunk.totalBytes());
    utils::C3dCodecParams p = cfg.c3dEncodeParams;
    p.depth = chunk.shape[0];
    p.height = chunk.shape[1];
    p.width = chunk.shape[2];
    tlEncoded = utils::c3d_encode(raw, p);
    return tlEncoded;
}

// Does `bytes` carry the C3DC magic header? If so we can skip
// decode+re-encode and passthrough verbatim.
static bool bytesAreCanonical(const std::vector<uint8_t>& bytes) {
    std::span<const std::byte> s(
        reinterpret_cast<const std::byte*>(bytes.data()), bytes.size());
    return utils::is_c3d_compressed(s);
}

BlockPipeline::BlockPipeline(
    Config config,
    BlockCache& blockCache,
    std::unique_ptr<VolumeSource> source,
    DecompressFn decompress,
    std::vector<std::unique_ptr<utils::ZarrArray>> diskLevels)
    : config_(std::move(config))
    , diskLevels_(std::move(diskLevels))
    , source_(std::move(source))
    , decompress_(std::move(decompress))
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
        return hw ? 4 * static_cast<int>(hw) : 32;
    }())
    , decodePool_([] {
        unsigned hw = std::thread::hardware_concurrency();
        return hw ? static_cast<int>(hw) : 8;
    }())
    , blockCache_(blockCache)
{
    fprintf(stderr, "[BlockPipeline] constructor %p: bytes=%zu\n", (void*)this, config_.bytes);

    // Clear any stale process-wide HTTP abort flag from a previous
    // BlockPipeline's destructor.
    utils::HttpClient::resetAbort();

    // Don't clear blockCache_ here — CState::setCurrentVolume already
    // called clearMemory() + _blockCache->clear() before we got here.
    // A second clear() would bump the generation and invalidate in-flight
    // inserts that read the gen between the two clears.

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
    // Label pool threads so perf / top can tell them apart. TASK_COMM_LEN
    // caps names at 15 chars so short prefixes + worker index are best.
    downloaderPool_.setThreadLabel("bpDl");
    encodePool_.setThreadLabel("bpEnc");
    loaderPool_.setThreadLabel("bpLd");
    decodePool_.setThreadLabel("bpDec");

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

        {
            std::unique_lock lk(encodeStagingMutex_);
            encodeStagingCv_.wait(lk, [this] {
                return shuttingDown_.load(std::memory_order_acquire)
                    || encodeStaging_.size() < config_.maxEncodeStagingChunks;
            });
            if (shuttingDown_.load(std::memory_order_acquire)) return {};
        }

        auto decoded = config_.compressed
            ? assembleCanonicalChunk(key)
            : fetchAndDecompressSourceChunk(key);
        if (!decoded) {
            const bool isHttp = dynamic_cast<HttpSource*>(source_.get()) != nullptr;
            const bool absent = !isHttp || HttpSource::lastFetchWasAbsent();
            if (absent) {
                bloomAdd(key);
                std::lock_guard lock(negativeMutex_);
                negativeCache_.insert(key);
                if (dz->is_sharded()) dz->mark_inner_chunk_empty(chunkIndices(key));
            } else if (isHttp && HttpSource::lastFetchHadTransientError()) {
                throw std::runtime_error("transient HTTP chunk fetch failure");
            }
            return {};
        }
        statIceFetches_.fetch_add(1, std::memory_order_relaxed);

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
        encodeStagingCv_.notify_one();
        if (!decoded) return {};

        size_t writeBytes;
        if (config_.compressed) {
            const auto& encoded = encodeCanonicalChunk(*decoded, config_);
            zarrWriteChunk(*dz, key,
                reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
            writeBytes = encoded.size();
        } else {
            // Raw mode: write decoded voxels directly
            zarrWriteChunk(*dz, key, decoded->rawData(),
                           decoded->bytes.size());
            writeBytes = decoded->bytes.size();
        }
        statDiskWrites_.fetch_add(1, std::memory_order_relaxed);
        statDiskBytes_.fetch_add(writeBytes, std::memory_order_relaxed);
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
                    innerBytes = dz->extract_inner_chunk(shardBytes->span(), inner);
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
                const bool isHttp = dynamic_cast<HttpSource*>(source_.get()) != nullptr;
                const bool absent = !isHttp || HttpSource::lastFetchWasAbsent();
                if (absent) {
                    bloomAdd(key);
                    std::lock_guard lock(negativeMutex_);
                    negativeCache_.insert(key);
                } else if (isHttp && HttpSource::lastFetchHadTransientError()) {
                    throw std::runtime_error("transient HTTP chunk fetch failure");
                }
                return {};
            }
            statColdHits_.fetch_add(1, std::memory_order_relaxed);
        }

        if (compressed.empty()) return {};

        // Backpressure: wait if the decoder is already sitting on more
        // compressed bytes than it can chew through. Gating here blocks
        // one loader worker at a time (the bytes are already on its
        // stack, but that's bounded by worker count × per-chunk size,
        // vastly less than the unbounded queue would cost).
        {
            std::unique_lock stage(decodeStagingMutex_);
            decodeStagingCv_.wait(stage, [this] {
                return shuttingDown_.load(std::memory_order_acquire)
                    || decodeStagingBytesAtomic_.load(std::memory_order_relaxed)
                         < config_.maxDecodeStagingBytes;
            });
            if (shuttingDown_.load(std::memory_order_acquire)) return {};
        }

        // Stage compressed bytes for decodePool_ so the loader worker can
        // immediately serve the next I/O request while CPU decode proceeds
        // in parallel. Decode concurrency is bounded by decodePool_ size.
        // Replacing an existing entry must decrement its size from the
        // atomic first — otherwise repeated loader re-queues of the same
        // key leak the counter upward and eventually stall every loader
        // worker on the backpressure CV with a map that's actually empty.
        const size_t compressedSize = compressed.size();
        {
            std::lock_guard stage(decodeStagingMutex_);
            auto [it, inserted] = decodeStaging_.try_emplace(
                key, std::vector<uint8_t>{});
            if (!inserted) {
                decodeStagingBytesAtomic_.fetch_sub(
                    it->second.size(), std::memory_order_relaxed);
            }
            it->second = std::move(compressed);
        }
        decodeStagingBytesAtomic_.fetch_add(compressedSize,
                                            std::memory_order_relaxed);
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
        decodeStagingBytesAtomic_.fetch_sub(compressed.size(),
                                            std::memory_order_relaxed);
        decodeStagingCv_.notify_one();
        if (compressed.empty()) return {};

        ChunkDataPtr decoded;
        auto* dz = (key.level < int(diskLevels_.size())) ? diskLevels_[key.level].get() : nullptr;
        if (!config_.compressed && dz) {
            // Raw mode: bytes are already decoded voxels
            auto scs = source_ ? source_->chunkShape(key.level)
                               : std::array<int,3>{0,0,0};
            decoded = acquireChunkData();
            decoded->shape = {scs[0], scs[1], scs[2]};
            decoded->elementSize = 1;
            decoded->bytes.resize(compressed.size());
            std::memcpy(decoded->bytes.data(), compressed.data(),
                        compressed.size());
        } else if (dz) {
            decoded = decodeCanonicalChunk(compressed);
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
                    } else if (HttpSource::lastFetchHadTransientError()) {
                        throw std::runtime_error("transient HTTP chunk fetch failure");
                    }
                    return {};
                }

                // Sanity: bytes must carry the C3DC magic. If not, the
                // source advertised canonical structure but serves blosc
                // or raw — we can't use it. Mark the chunk negative so
                // we stop re-fetching it every render; otherwise every
                // fetchInteractive would re-queue it forever.
                if (!bytesAreCanonical(bytes)) {
                    // Always log: these are source-level corruption events,
                    // not transient noise. Rate-limiting to 5 previously hid
                    // systemic codec-mismatch issues from users.
                    std::fprintf(stderr,
                        "[BlockPipeline] passthrough: chunk lvl=%d "
                        "(%d,%d,%d) lacks C3DC magic — marking absent\n",
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

    // Precompute blocks-per-chunk for each level (used by blockAt).
    for (int lvl = 0; lvl < kMaxLevels; ++lvl) {
        auto cs = chunkShape(lvl);
        if (cs[0] > 0 && cs[1] > 0 && cs[2] > 0) {
            blocksPerChunk_[lvl] = {cs[0] / kBlockSize,
                                    cs[1] / kBlockSize,
                                    cs[2] / kBlockSize};
        }
    }

    downloaderPool_.start();
    encodePool_.start();
    loaderPool_.start();
    decodePool_.start();
}

void BlockPipeline::shutdown() {
    // Atomic exchange: if already shutting down (destructor or prior shutdown()
    // call), skip. This makes shutdown() + ~BlockPipeline() idempotent.
    if (shuttingDown_.exchange(true, std::memory_order_acq_rel)) {
        fprintf(stderr, "[BlockPipeline] shutdown %p: already shut down\n", (void*)this);
        return;
    }
    fprintf(stderr, "[BlockPipeline] shutdown %p: stopping pools...\n", (void*)this);
    // Release any downloader workers blocked on the backpressure CV so
    // pool.stop() can actually join them.
    encodeStagingCv_.notify_all();
    decodeStagingCv_.notify_all();
    inflightShardCv_.notify_all();
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
    // All workers have joined — safe to clear the process-global abort
    // flag so a new pipeline can use curl without seeing a stale abort.
    utils::HttpClient::resetAbort();
    fprintf(stderr, "[BlockPipeline] shutdown %p: pools stopped, abort cleared\n", (void*)this);
    auto cold = statColdHits_.load();
    auto ice = statIceFetches_.load();
    if (cold > 0 || ice > 0) {
        std::fprintf(stderr, "[Cache] session summary: coldHits=%lu iceFetches=%lu (%.0f%% from disk)\n",
                     cold, ice, (cold + ice) > 0 ? 100.0 * cold / (cold + ice) : 0.0);
    }
}

BlockPipeline::~BlockPipeline() {
    fprintf(stderr, "[BlockPipeline] destructor %p\n", (void*)this);
    shutdown();
    fprintf(stderr, "[BlockPipeline] destructor %p: done\n", (void*)this);
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
    static int fetchDbgCount = 0;
    if (fetchDbgCount++ < 5 || fetchDbgCount % 100 == 0)
        fprintf(stderr, "[BlockPipeline] fetchInteractive %p: %zu keys, targetLevel=%d\n",
                (void*)this, keys.size(), targetLevel);
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
        const bool diskPresent = dz && (
            dz->is_sharded()
                ? dz->inner_chunk_exists(chunkIndices(key))
                : dz->chunk_exists(chunkIndices(key)));
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
    // don't store). Canonical chunks are (canonical_side / block_side)
    // blocks along each axis — 8 for h265 (128³ / 16³), 16 for c3d
    // (256³ / 16³). Reverse-map the block coord to its enclosing
    // canonical-chunk index and query via the lock-free hash set. No
    // rwlock on the miss path — every blockAt miss used to hit
    // pthread_rwlock_rdlock here.
    const auto& bpc = (key.level >= 0 && key.level < kMaxLevels)
        ? blocksPerChunk_[key.level]
        : blocksPerChunk_[0];
    const int bpcZ = bpc[0] > 0 ? bpc[0] : (kCanonicalChunkSide / kBlockSize);
    const int bpcY = bpc[1] > 0 ? bpc[1] : (kCanonicalChunkSide / kBlockSize);
    const int bpcX = bpc[2] > 0 ? bpc[2] : (kCanonicalChunkSide / kBlockSize);
    const ChunkKey chunkKey{key.level, key.bz / bpcZ, key.by / bpcY, key.bx / bpcX};
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
    const int C = kCanonicalChunkSide;
    auto scs = source_->chunkShape(canonKey.level);
    if (scs[0] <= 0 || scs[1] <= 0 || scs[2] <= 0) return nullptr;

    int cz0 = canonKey.iz * C, cy0 = canonKey.iy * C, cx0 = canonKey.ix * C;
    int cz1 = cz0 + C,         cy1 = cy0 + C,         cx1 = cx0 + C;
    int sz0 = cz0 / scs[0], sz1 = (cz1 + scs[0] - 1) / scs[0];
    int sy0 = cy0 / scs[1], sy1 = (cy1 + scs[1] - 1) / scs[1];
    int sx0 = cx0 / scs[2], sx1 = (cx1 + scs[2] - 1) / scs[2];

    auto out = vc::cache::acquireChunkData();
    out->shape = {C, C, C};
    out->elementSize = 1;
    out->bytes.assign(size_t(C) * C * C, 0);
    bool anyData = false;
    bool sawTransientFetchFailure = false;
    const bool isHttp = dynamic_cast<HttpSource*>(source_.get()) != nullptr;

    for (int siz = sz0; siz < sz1; ++siz)
    for (int siy = sy0; siy < sy1; ++siy)
    for (int six = sx0; six < sx1; ++six) {
        ChunkKey srcKey{canonKey.level, siz, siy, six};
        std::vector<uint8_t> compressed;
        try {
            compressed = source_->fetch(srcKey);
        } catch (...) {
            if (isHttp) sawTransientFetchFailure = true;
            continue;
        }
        if (compressed.empty() || isAllZero(compressed.data(), compressed.size())) {
            if (isHttp && HttpSource::lastFetchHadTransientError())
                sawTransientFetchFailure = true;
            continue;
        }
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

    if (sawTransientFetchFailure)
        throw std::runtime_error("transient HTTP chunk fetch failure");
    if (!anyData) return nullptr;
    return out;
}

ChunkDataPtr BlockPipeline::fetchAndDecompressSourceChunk(const ChunkKey& key) {
    if (!source_ || !decompress_) return nullptr;
    std::vector<uint8_t> compressed;
    try { compressed = source_->fetch(key); }
    catch (...) { return nullptr; }
    if (compressed.empty() || isAllZero(compressed.data(), compressed.size())) {
        return nullptr;
    }
    return decompress_(compressed, key);
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
        TickCoordinator::notifyEmptyChunkNoted(key);
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
    BlockCache::BatchPut batch(blockCache_, blockCache_.generation());
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
    TickCoordinator::notifyChunkLanded(this, key);
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
    // the codec's native chunk size (H265=128³, C3d=256³) by
    // assembleCanonicalChunk / insertChunkAsBlocks. The source's native
    // chunk size may differ (e.g., 128³ h265 source re-canonicalized as
    // 256³ c3d); returning the source shape would make Slicing.cpp's
    // chunk-key enumeration compute the wrong grid.
    // In uncompressed mode, source chunk size is used directly (no rechunking).
    if (level >= 0 && level < int(diskLevels_.size()) && diskLevels_[level]) {
        if (config_.compressed) {
            const int C = kCanonicalChunkSide;
            return {C, C, C};
        }
        return source_->chunkShape(level);
    }
    return source_->chunkShape(level);
}

std::array<int, 3> BlockPipeline::levelShape(int level) const noexcept {
    if (!source_) return {0, 0, 0};
    auto shape = source_->levelShape(level);
    // If this level doesn't physically exist (shape is zero), synthesize
    // the expected shape from a level that does exist using scale factors.
    // This allows bounds checks and coordinate transforms to work correctly
    // for missing fine scales (e.g., scales 0-1 absent, only 2+ present).
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        float sf = levelScaleFactor(level);
        int nLevels = source_->numLevels();
        for (int i = 0; i < nLevels; i++) {
            auto refShape = source_->levelShape(i);
            if (refShape[0] > 0 && refShape[1] > 0 && refShape[2] > 0) {
                float refSf = levelScaleFactor(i);
                float ratio = refSf / sf;  // e.g. 4.0/1.0 = 4 if ref is coarser
                shape = {
                    static_cast<int>(std::ceil(refShape[0] * ratio)),
                    static_cast<int>(std::ceil(refShape[1] * ratio)),
                    static_cast<int>(std::ceil(refShape[2] * ratio))
                };
                break;
            }
        }
    }
    return shape;
}

float BlockPipeline::levelScaleFactor(int vectorIndex) const noexcept {
    // Power-of-2 scaling: level 0 → 1, level 1 → 2, level 2 → 4, etc.
    // With padded level vectors (index = actual scale), this is always correct.
    return static_cast<float>(size_t{1} << vectorIndex);
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

std::vector<ChunkKey> BlockPipeline::chunksMissingFromCache(
    const std::vector<ChunkKey>& keys) const
{
    std::vector<ChunkKey> missing;
    if (!source_) return keys;

    constexpr int kMaxL = 16;
    std::array<int, kMaxL> bpcZ{}, bpcY{}, bpcX{};
    std::array<bool, kMaxL> shapeCached{};

    auto ensureShapeCached = [&](int level) {
        if (level < 0 || level >= kMaxL || shapeCached[level]) return;
        auto csk = chunkShape(level);
        if (csk[0] > 0) {
            bpcZ[level] = csk[0] / kBlockSize;
            bpcY[level] = csk[1] / kBlockSize;
            bpcX[level] = csk[2] / kBlockSize;
        }
        shapeCached[level] = true;
    };

    for (const auto& key : keys) {
        if (isNegativeCached(key) || isEmptyChunk(key)) continue;

        bool resident = false;
        if (key.level >= 0 && key.level < kMaxL) {
            ensureShapeCached(key.level);
            const int z = bpcZ[key.level], y = bpcY[key.level], x = bpcX[key.level];
            if (z > 0 && y > 0 && x > 0) {
                const BlockKey first{key.level, key.iz * z, key.iy * y, key.ix * x};
                const BlockKey last{key.level, key.iz * z + z - 1,
                                               key.iy * y + y - 1,
                                               key.ix * x + x - 1};
                resident = blockCache_.contains(first) && blockCache_.contains(last);
            }
        }
        if (resident) continue;

        auto* dz = (key.level >= 0 && key.level < int(diskLevels_.size()))
            ? diskLevels_[key.level].get()
            : nullptr;
        const bool diskPresent = dz
            && dz->is_sharded()
            && dz->inner_chunk_exists(chunkIndices(key));
        if (diskPresent) continue;

        missing.push_back(key);
    }

    return missing;
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
    {
        std::lock_guard lk(encodeStagingMutex_);
        s.encodeStagingChunks = encodeStaging_.size();
    }
    s.decodeStagingBytes = decodeStagingBytesAtomic_.load(
        std::memory_order_relaxed);
    {
        std::lock_guard lk(inflightShardMutex_);
        s.inflightShardReads = inflightShardReads_;
    }
    s.inflightShardBytes = inflightShardBytes_.load(std::memory_order_relaxed);
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
    VcDataset* ds, size_t maxBytes, const std::filesystem::path& datasetPath,
    BlockCache* sharedCache)
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

    std::unique_ptr<BlockCache> ownedCache;
    if (!sharedCache) {
        BlockCache::Config bcfg;
        bcfg.bytes = maxBytes;
        for (auto& f : bcfg.levelFloor) f = 4096;
        ownedCache = std::make_unique<BlockCache>(bcfg);
        sharedCache = ownedCache.get();
    }
    auto pipeline = std::make_unique<BlockPipeline>(
        std::move(cfg), *sharedCache, std::move(source), std::move(decompress));
    pipeline->ownBlockCache(std::move(ownedCache));
    return pipeline;
}

}  // namespace vc::cache
