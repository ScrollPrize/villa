#include "vc/core/util/ChunkCache.hpp"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "z5/dataset.hxx"
#include "z5/types/types.hxx"

#include <algorithm>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Read compressed chunk bytes using POSIX I/O — 3 syscalls:
// open, fstat, read, close. Buffer is reused across calls.
// O_NOATIME avoids updating file access time (saves inode write-back).
static bool readChunkFileRaw(const char* path, std::vector<char>& buffer)
{
    int fd = ::open(path, O_RDONLY | O_NOATIME);
    if (fd < 0) {
        // O_NOATIME requires ownership or CAP_FOWNER; fall back
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
    }

    struct stat sb;
    if (::fstat(fd, &sb) != 0) { ::close(fd); return false; }

    const size_t fileSize = static_cast<size_t>(sb.st_size);
    if (fileSize == 0) { ::close(fd); return false; }
    buffer.resize(fileSize);

    size_t total = 0;
    while (total < fileSize) {
        ssize_t n = ::read(fd, buffer.data() + total, fileSize - total);
        if (n <= 0) { ::close(fd); return false; }
        total += static_cast<size_t>(n);
    }
    ::close(fd);
    return true;
}

// Chunk path helper — caches the base path string and delimiter to avoid
// repeated z5::Dataset virtual calls and fs::path construction.
struct ChunkPathBuilder {
    std::string basePath;
    std::string delimiter;
    bool isZarr;
    // Reusable string buffer for path construction
    std::string pathBuf;

    explicit ChunkPathBuilder(z5::Dataset& ds) {
        basePath = ds.path().string();
        isZarr = ds.isZarr();
        // Extract delimiter by probing a known chunk path
        if (isZarr) {
            z5::types::ShapeType probe = {0, 0, 1};
            std::filesystem::path p;
            ds.chunkPath(probe, p);
            // Path is basePath + "/" + "0" + delim + "0" + delim + "1"
            // Extract delimiter from the suffix
            std::string suffix = p.string().substr(basePath.size() + 1);
            // suffix is "0<delim>0<delim>1"
            size_t pos = suffix.find_first_not_of("0123456789");
            if (pos != std::string::npos) {
                size_t end = suffix.find_first_of("0123456789", pos);
                delimiter = suffix.substr(pos, end - pos);
            } else {
                delimiter = ".";
            }
        }
    }

    const char* build(size_t iz, size_t iy, size_t ix) {
        pathBuf.clear();
        pathBuf.reserve(basePath.size() + 20);
        pathBuf.append(basePath);
        pathBuf.push_back('/');
        if (isZarr) {
            appendUint(iz);
            pathBuf.append(delimiter);
            appendUint(iy);
            pathBuf.append(delimiter);
            appendUint(ix);
        } else {
            // N5: reversed axis order
            appendUint(ix);
            pathBuf.push_back('/');
            appendUint(iy);
            pathBuf.push_back('/');
            appendUint(iz);
        }
        return pathBuf.c_str();
    }

private:
    void appendUint(size_t v) {
        // Fast uint-to-string without allocation
        char tmp[20];
        int len = 0;
        if (v == 0) { pathBuf.push_back('0'); return; }
        while (v > 0) { tmp[len++] = '0' + static_cast<char>(v % 10); v /= 10; }
        for (int i = len - 1; i >= 0; i--) pathBuf.push_back(tmp[i]);
    }
};

// Helper to read a chunk from disk via z5::Dataset.
// Uses direct POSIX I/O to bypass z5's redundant filesystem checks,
// with thread-local buffer reuse and kernel readahead hints.
template<typename T>
static std::shared_ptr<xt::xarray<T>> readChunkFromSource(
    z5::Dataset& ds, size_t iz, size_t iy, size_t ix,
    ChunkPathBuilder& pathBuilder, std::vector<char>& compressedBuf)
{
    const char* path = pathBuilder.build(iz, iy, ix);

    if (!readChunkFileRaw(path, compressedBuf))
        return nullptr;  // chunk doesn't exist or read failed

    const auto& maxChunkShape = ds.defaultChunkShape();
    const std::size_t maxChunkSize = ds.defaultChunkSize();

    auto out = std::make_shared<xt::xarray<T>>(xt::empty<T>(maxChunkShape));

    if (ds.getDtype() == z5::types::Datatype::uint8) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            ds.decompress(compressedBuf, out->data(), maxChunkSize);
        } else {
            throw std::runtime_error("Cannot read uint8 dataset into uint16 array");
        }
    }
    else if (ds.getDtype() == z5::types::Datatype::uint16) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            ds.decompress(compressedBuf, out->data(), maxChunkSize);
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            xt::xarray<uint16_t> tmp = xt::empty<uint16_t>(maxChunkShape);
            ds.decompress(compressedBuf, tmp.data(), maxChunkSize);

            uint8_t* p8 = out->data();
            uint16_t* p16 = tmp.data();
            for (size_t i = 0; i < maxChunkSize; i++)
                p8[i] = p16[i] / 257;
        }
    }

    return out;
}

// Backward-compatible overload (allocates fresh buffer each call)
template<typename T>
static std::shared_ptr<xt::xarray<T>> readChunkFromSource(z5::Dataset& ds, size_t iz, size_t iy, size_t ix)
{
    thread_local ChunkPathBuilder* tlPathBuilder = nullptr;
    thread_local z5::Dataset* tlDs = nullptr;
    thread_local std::vector<char> tlBuffer;

    // Lazily create/update thread-local path builder
    if (tlDs != &ds) {
        delete tlPathBuilder;
        tlPathBuilder = new ChunkPathBuilder(ds);
        tlDs = &ds;
    }

    return readChunkFromSource<T>(ds, iz, iy, ix, *tlPathBuilder, tlBuffer);
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

        // Track re-reads: if we've loaded this chunk before, it was evicted
        // and is now being re-read — wasted I/O
        auto [it, firstTime] = _everLoaded.insert(key);
        if (!firstTime) {
            _reReads.fetch_add(1, std::memory_order_relaxed);
            _reReadBytes.fetch_add(chunkBytes, std::memory_order_relaxed);
        }

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
        _bytesRead.load(std::memory_order_relaxed),
        _reReads.load(std::memory_order_relaxed),
        _reReadBytes.load(std::memory_order_relaxed)
    };
}

template<typename T>
void ChunkCache<T>::resetStats()
{
    _hits.store(0, std::memory_order_relaxed);
    _misses.store(0, std::memory_order_relaxed);
    _evictions.store(0, std::memory_order_relaxed);
    _bytesRead.store(0, std::memory_order_relaxed);
    _reReads.store(0, std::memory_order_relaxed);
    _reReadBytes.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> evictLock(_evictionMutex);
        _everLoaded.clear();
    }
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
