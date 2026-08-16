#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vc::render {

inline constexpr std::string_view kPersistentSourcePayloadExtension = ".source";

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
