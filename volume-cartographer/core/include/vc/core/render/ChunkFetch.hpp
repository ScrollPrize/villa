#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vc::render {

class DecodedChunkPayload {
public:
    virtual ~DecodedChunkPayload() = default;
    [[nodiscard]] virtual std::size_t residentBytes() const noexcept = 0;
};

inline constexpr std::string_view kPersistentSourcePayloadExtension = ".source";

// Zarr store keys are relative slash-separated object names. Reject platform
// roots, empty components, and traversal before joining them to a cache root.
inline bool isSafeZarrStoreKey(std::string_view key) noexcept
{
    if (key.empty() || key.front() == '/' || key.find('\\') != std::string_view::npos ||
        (key.size() >= 2 && key[1] == ':'))
        return false;
    std::size_t begin = 0;
    while (begin < key.size()) {
        const auto end = key.find('/', begin);
        const auto component = key.substr(begin, end - begin);
        if (component.empty() || component == "." || component == "..")
            return false;
        if (end == std::string_view::npos)
            return true;
        begin = end + 1;
    }
    return false;
}

struct VolumeSourceId {
    std::uint64_t value = 0;

    explicit operator bool() const noexcept { return value != 0; }
    friend bool operator==(const VolumeSourceId&, const VolumeSourceId&) = default;
};

struct ChunkKey {
    int level = 0;
    int iz = 0;
    int iy = 0;
    int ix = 0;
    VolumeSourceId sourceId{};

    friend bool operator==(const ChunkKey&, const ChunkKey&) = default;
};

struct ChunkKeyHash {
    std::size_t operator()(const ChunkKey& key) const noexcept
    {
        std::size_t seed = std::hash<std::uint64_t>{}(key.sourceId.value);
        auto combine = [&seed](int value) {
            seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        combine(key.level);
        combine(key.iz);
        combine(key.iy);
        combine(key.ix);
        return seed;
    }
};

enum class ChunkFetchStatus {
    Found,
    Missing,
    HttpError,
    IoError,
    DecodeError
};

struct ChunkFetchResult {
    ChunkFetchStatus status = ChunkFetchStatus::Missing;
    std::vector<std::byte> bytes;
    std::vector<std::byte> persistentBytes;
    bool hasPersistentBytes = false;
    // An independent exact-source write owns persistence for this fetch.
    bool persistentWriteHandled = false;
    int httpStatus = 0;
    std::string message;
    // Application-decoded opaque chunks live directly in ChunkCache's LRU.
    // Dense tensor chunks continue to use `bytes`.
    std::shared_ptr<const DecodedChunkPayload> payload;
};

// A logical decoded chunk may be an inner member of one physical storage
// object (a Zarr shard). Persistent probing, downloading, and writing use this
// identity; decoded RAM remains keyed by ChunkKey.
struct ChunkStorageObject {
    ChunkKey representativeKey;
    int outerZ = 0;
    int outerY = 0;
    int outerX = 0;
    std::array<int, 3> innerIndices{};
    std::array<int, 3> innerChunksPerObject{1, 1, 1};
    std::string sourceKey;

    [[nodiscard]] bool sharded() const noexcept
    {
        return innerChunksPerObject != std::array<int, 3>{1, 1, 1};
    }
};

class IChunkFetcher {
public:
    using DownloadProgressCallback = std::function<void(std::size_t)>;

    virtual ~IChunkFetcher() = default;
    virtual ChunkFetchResult fetch(const ChunkKey& key) = 0;

    // True only when fetchEncoded() issues a remote HTTP payload request and
    // reports its response-body bytes through DownloadProgressCallback.
    // The scheduler uses this before the call so connection and TTFB time are
    // part of the measured network interval.
    [[nodiscard]] virtual bool measuresRemoteTransfer() const noexcept
    {
        return false;
    }

    // Split transfer from CPU decoding for schedulers which provide a
    // dedicated decode stage. The compatibility default preserves existing
    // fetchers whose fetch() already returns decoded bytes.
    virtual ChunkFetchResult fetchEncoded(const ChunkKey& key)
    {
        return fetch(key);
    }

    // Remote fetchers report encoded response-body bytes while they are
    // received. The default intentionally emits no synthetic progress.
    virtual ChunkFetchResult fetchEncoded(
        const ChunkKey& key,
        const DownloadProgressCallback&)
    {
        return fetchEncoded(key);
    }

    virtual ChunkFetchResult decodeFetched(
        const ChunkKey&,
        ChunkFetchResult fetched) const
    {
        return fetched;
    }

    virtual std::string persistentCacheExtension(const ChunkKey&) const
    {
        return ".bin";
    }

    virtual std::optional<std::string> sourceChunkKey(const ChunkKey&) const
    {
        return std::nullopt;
    }

    // Exact physical-object access used by native Zarr mirror caches. Generic
    // fetchers remain on the legacy logical-chunk persistence path.
    virtual std::optional<ChunkStorageObject>
    storageObject(const ChunkKey&) const
    {
        return std::nullopt;
    }

    virtual ChunkFetchResult fetchStorageObject(
        const ChunkStorageObject& object,
        const DownloadProgressCallback& progress)
    {
        return fetchEncoded(object.representativeKey, progress);
    }

    virtual ChunkFetchResult decodeStorageObject(
        const ChunkKey& key,
        std::span<const std::byte> objectBytes) const
    {
        return decodeSourcePayload(
            key, std::vector<std::byte>(objectBytes.begin(), objectBytes.end()));
    }

    virtual std::optional<ChunkKey> logicalRepresentativeForStorageKey(
        int,
        std::string_view) const
    {
        return std::nullopt;
    }

    virtual bool sourcePayloadMatchesPersistentCache(const ChunkKey&) const
    {
        return false;
    }

    // Persistence maintenance stores the exact source payload without
    // decoding. Fetchers opt in only when decodeSourcePayload() can reconstruct
    // the decoded chunk from those bytes later.
    virtual bool supportsSourcePayloadPersistence(const ChunkKey&) const
    {
        return false;
    }

    virtual ChunkFetchResult decodeSourcePayload(
        const ChunkKey& key,
        std::vector<std::byte> bytes) const
    {
        ChunkFetchResult fetched;
        fetched.status = ChunkFetchStatus::Found;
        fetched.bytes = std::move(bytes);
        return decodeFetched(key, std::move(fetched));
    }

    virtual ChunkFetchResult decodePersistentBytes(
        const ChunkKey&,
        std::vector<std::byte> bytes) const
    {
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::move(bytes);
        return result;
    }
};

} // namespace vc::render
