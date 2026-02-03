#include "vc/core/util/ChunkCache.hpp"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "z5/dataset.hxx"
#include "z5/types/types.hxx"

#include <algorithm>

// Helper to read a chunk from disk via z5::Dataset
template<typename T>
static std::shared_ptr<xt::xarray<T>> readChunkFromSource(z5::Dataset& ds, size_t iz, size_t iy, size_t ix)
{
    z5::types::ShapeType chunkId = {iz, iy, ix};

    if (!ds.chunkExists(chunkId))
        return nullptr;

    if (ds.getDtype() != z5::types::Datatype::uint8 && ds.getDtype() != z5::types::Datatype::uint16)
        throw std::runtime_error("only uint8_t/uint16 zarrs supported currently!");

    const auto& maxChunkShape = ds.defaultChunkShape();
    const std::size_t maxChunkSize = ds.defaultChunkSize();

    auto out = std::make_shared<xt::xarray<T>>(xt::empty<T>(maxChunkShape));

    if (ds.getDtype() == z5::types::Datatype::uint8) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            ds.readChunk(chunkId, out->data());
        } else {
            throw std::runtime_error("Cannot read uint8 dataset into uint16 array");
        }
    }
    else if (ds.getDtype() == z5::types::Datatype::uint16) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            ds.readChunk(chunkId, out->data());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            xt::xarray<uint16_t> tmp = xt::empty<uint16_t>(maxChunkShape);
            ds.readChunk(chunkId, tmp.data());

            uint8_t* p8 = out->data();
            uint16_t* p16 = tmp.data();
            for (size_t i = 0; i < maxChunkSize; i++)
                p8[i] = p16[i] / 257;
        }
    }

    return out;
}

template<typename T>
ChunkCache<T>::ChunkCache(size_t maxBytes) : _maxBytes(maxBytes)
{
}

template<typename T>
ChunkCache<T>::~ChunkCache()
{
    clear();
}

template<typename T>
void ChunkCache<T>::setMaxBytes(size_t maxBytes)
{
    _maxBytes = maxBytes;
}

template<typename T>
auto ChunkCache<T>::get(z5::Dataset* ds, int iz, int iy, int ix) -> ChunkPtr
{
    ChunkKey key{ds, iz, iy, ix};

    // Fast path: shared lock read
    {
        std::shared_lock<std::shared_mutex> rlock(_mapMutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
            it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
            _hits.fetch_add(1, std::memory_order_relaxed);
            return it->second.chunk;
        }
    }

    _misses.fetch_add(1, std::memory_order_relaxed);

    // Slow path: load from disk (per-key lock to avoid duplicate reads)
    std::lock_guard<std::mutex> diskLock(_lockPool[lockIndex(key)]);

    // Re-check after acquiring disk lock
    {
        std::shared_lock<std::shared_mutex> rlock(_mapMutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
            it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
            return it->second.chunk;
        }
    }

    ChunkPtr newChunk = loadChunk(ds, iz, iy, ix);
    if (!newChunk) return nullptr;

    size_t chunkBytes = newChunk->size() * sizeof(T);
    _bytesRead.fetch_add(chunkBytes, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> evictLock(_evictionMutex);
        if (_maxBytes > 0 && _storedBytes.load(std::memory_order_relaxed) + chunkBytes > _maxBytes) {
            evictIfNeeded();
        }
    }

    {
        std::unique_lock<std::shared_mutex> wlock(_mapMutex);
        auto [it, inserted] = _map.try_emplace(key, CacheEntry{newChunk, chunkBytes, _generation.fetch_add(1, std::memory_order_relaxed)});
        if (inserted) {
            _storedBytes.fetch_add(chunkBytes, std::memory_order_relaxed);
            _cachedCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Another thread inserted while we were loading — use theirs
            return it->second.chunk;
        }
    }

    return newChunk;
}

template<typename T>
auto ChunkCache<T>::getIfCached(z5::Dataset* ds, int iz, int iy, int ix) const -> ChunkPtr
{
    ChunkKey key{ds, iz, iy, ix};
    std::shared_lock<std::shared_mutex> rlock(_mapMutex);
    auto it = _map.find(key);
    if (it != _map.end()) {
        _hits.fetch_add(1, std::memory_order_relaxed);
        return it->second.chunk;
    }
    return nullptr;
}

template<typename T>
void ChunkCache<T>::prefetch(z5::Dataset* ds, int minIz, int minIy, int minIx, int maxIz, int maxIy, int maxIx)
{
    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int ix = minIx; ix <= maxIx; ix++) {
        for (int iy = minIy; iy <= maxIy; iy++) {
            for (int iz = minIz; iz <= maxIz; iz++) {
                get(ds, iz, iy, ix);
            }
        }
    }
}

template<typename T>
void ChunkCache<T>::clear()
{
    std::unique_lock<std::shared_mutex> wlock(_mapMutex);
    _map.clear();
    _storedBytes.store(0, std::memory_order_relaxed);
    _cachedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);
}

template<typename T>
void ChunkCache<T>::flush()
{
    clear();
}

template<typename T>
void ChunkCache<T>::evictIfNeeded()
{
    // Called with _evictionMutex held
    if (_maxBytes == 0) return;

    size_t currentBytes = _storedBytes.load(std::memory_order_relaxed);
    if (currentBytes <= _maxBytes) return;

    struct EvictCandidate {
        ChunkKey key;
        size_t bytes;
        uint64_t lastAccess;
    };

    std::vector<EvictCandidate> candidates;

    {
        std::shared_lock<std::shared_mutex> rlock(_mapMutex);
        candidates.reserve(_map.size());
        for (auto& [key, entry] : _map) {
            candidates.push_back({key, entry.bytes, entry.lastAccess});
        }
    }

    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(),
              [](const EvictCandidate& a, const EvictCandidate& b) {
                  return a.lastAccess < b.lastAccess;
              });

    size_t target = _maxBytes * 15 / 16;
    size_t evictedBytes = 0;
    size_t evictedCount = 0;

    std::vector<ChunkKey> toRemove;
    for (auto& c : candidates) {
        if (currentBytes - evictedBytes <= target) break;
        toRemove.push_back(c.key);
        evictedBytes += c.bytes;
        evictedCount++;
    }

    if (!toRemove.empty()) {
        std::unique_lock<std::shared_mutex> wlock(_mapMutex);
        for (auto& key : toRemove) {
            _map.erase(key);
        }
        _storedBytes.fetch_sub(evictedBytes, std::memory_order_relaxed);
        _cachedCount.fetch_sub(evictedCount, std::memory_order_relaxed);
        _evictions.fetch_add(evictedCount, std::memory_order_relaxed);
    }
}

template<typename T>
auto ChunkCache<T>::stats() const -> Stats
{
    return {
        _hits.load(std::memory_order_relaxed),
        _misses.load(std::memory_order_relaxed),
        _evictions.load(std::memory_order_relaxed),
        _bytesRead.load(std::memory_order_relaxed)
    };
}

template<typename T>
void ChunkCache<T>::resetStats()
{
    _hits.store(0, std::memory_order_relaxed);
    _misses.store(0, std::memory_order_relaxed);
    _evictions.store(0, std::memory_order_relaxed);
    _bytesRead.store(0, std::memory_order_relaxed);
}

template<typename T>
auto ChunkCache<T>::loadChunk(z5::Dataset* ds, int iz, int iy, int ix) -> ChunkPtr
{
    if (!ds) return nullptr;
    try {
        return readChunkFromSource<T>(*ds, static_cast<size_t>(iz), static_cast<size_t>(iy), static_cast<size_t>(ix));
    } catch (const std::exception&) {
        return nullptr;
    }
}

// Explicit template instantiations
template class ChunkCache<uint8_t>;
template class ChunkCache<uint16_t>;
