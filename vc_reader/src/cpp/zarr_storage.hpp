// Zarr v3 objects and indexed shards, over the shared HTTP/local I/O path.
#pragma once
#include "chunk_io.hpp"
#include "http.hpp"
#include "shard_index.hpp"
#include "channel_decode.hpp"
#include <array>
#include <atomic>
#include <limits>
#include <optional>

namespace vcr {

class ZarrStorage {
    struct IndexData {
        std::vector<std::uint8_t> bytes;
        std::uint64_t total = 0, start = 0;
        std::string etag;
    };
public:
    ZarrStorage(std::string root, std::string prefix, std::string separator, Index shards,
                bool index_end, bool index_crc, std::size_t chunk_bytes, HttpClient* http,
                std::atomic<std::uint64_t>& requests, std::atomic<std::uint64_t>& bytes)
        : root_(std::move(root)), prefix_(std::move(prefix)), separator_(std::move(separator)),
          layout_(std::move(shards), index_end, index_crc, chunk_bytes), http_(http),
          requests_(requests), bytes_(bytes)
    {}

    bool read(const std::string& key, std::vector<std::uint8_t>& body)
    {
        const auto [path, position] = location(key);
        if (layout_.ratios.empty()) {
            if (http_) {
                ++requests_;
                auto result = http_->get_stream(path, [&] { body.clear(); }, [&](const char* p, std::size_t n) {
                    if (n > layout_.maxPayload-body.size()) throw std::runtime_error("encoded chunk exceeds bounded payload size");
                    body.insert(body.end(), p, p+n);
                });
                bytes_ += result.bytes;
                return result.status == 200;
            }
            return local_read(path, body, std::nullopt, 0, layout_.maxPayload);
        }
        if (!http_) return LocalShard(path, layout_).readEntry(*position, body);
        auto index = shard_index(path);
        if (!index) return false;
        const auto entry = layout_.entry(index->bytes, *position, index->total);
        if (!entry) return false;
        const auto [offset, length] = *entry;
        ++requests_;
        auto response = http_->get_range(path, body, std::to_string(offset)+"-"+std::to_string(offset+length-1),
                                         length, index->etag);
        bytes_ += response.bytes;
        if (response.status == 404) throw std::runtime_error("shard disappeared after index read");
        validate_range(response, offset, length, index->total);
        return true;
    }

    std::optional<ChannelDecodeResult> read_channels(const std::string& key, const ChunkCodec& codec,
        std::size_t decodedBytes, std::size_t channelBytes, std::span<const std::size_t> requested,
        std::span<const std::size_t> relevant)
    {
        const auto [path, position] = location(key);
        if (!http_) {
            if (position) {
                LocalShard shard(path, layout_);
                if (shard.index.empty()) return std::nullopt;
                const auto entry = layout_.entry(shard.index, *position, shard.total);
                if (!entry) return std::nullopt;
                return decode_channels(codec, decodedBytes, channelBytes, requested, relevant, entry->second,
                    [&](auto offset, auto length, auto dst) { shard.readRange(entry->first+offset, length, dst); });
            }
            File file = File::open_read(path);
            if (!file) return std::nullopt;
            platform::FileStat stat;
            if (platform::fstat(file.fd(), stat) != 0)
                throw std::system_error(errno, std::generic_category(), "stat " + path);
            if (stat.size < 0 || std::uint64_t(stat.size) > layout_.maxPayload)
                throw std::runtime_error("oversized channel payload: " + path);
            return decode_channels(codec, decodedBytes, channelBytes, requested, relevant, std::size_t(stat.size),
                [&](auto offset, auto length, auto dst) { pread_exact(file.fd(), dst, length, offset, path); });
        }
        std::uint64_t base = 0, length = 0, total = 0;
        std::string etag;
        std::vector<std::uint8_t> prefix;
        if (position) {
            auto index = shard_index(path);
            if (!index) return std::nullopt;
            const auto entry = layout_.entry(index->bytes, *position, index->total);
            if (!entry) return std::nullopt;
            base = entry->first; length = entry->second; total = index->total; etag = index->etag;
        } else {
            // Discover size and revision without a HEAD request. For Blosc this
            // is also the complete codec header, reused by the range planner.
            const std::size_t probe = codec == Codec::Blosc ? BLOSC_MIN_HEADER_LENGTH : 1;
            ++requests_;
            auto response = http_->get_range(path, prefix, "0-"+std::to_string(probe-1), probe);
            bytes_ += response.bytes;
            if (response.status == 404) return std::nullopt;
            unsigned long long first = 0, last = 0, size = 0;
            if (std::sscanf(response.content_range.c_str(), "bytes %llu-%llu/%llu", &first, &last, &size) != 3 ||
                first != 0 || last != probe-1 || size < probe || size > layout_.maxPayload)
                throw std::runtime_error("invalid channel object Content-Range");
            length = total = size; etag = response.etag;
        }
        return decode_channels(codec, decodedBytes, channelBytes, requested, relevant, length,
            [&](std::size_t offset, std::size_t size, std::uint8_t* dst) {
                if (offset < prefix.size()) {
                    const auto reused = std::min(size, prefix.size()-offset);
                    std::memcpy(dst, prefix.data()+offset, reused);
                    dst += reused; offset += reused; size -= reused;
                }
                if (!size) return;
                std::vector<std::uint8_t> body;
                ++requests_;
                auto response = http_->get_range(path, body, std::to_string(base+offset)+"-"+
                    std::to_string(base+offset+size-1), size, etag);
                bytes_ += response.bytes;
                if (response.status == 404) throw std::runtime_error("channel object disappeared during ranged read");
                validate_range(response, base+offset, size, total);
                if (!etag.empty() && !response.etag.empty() && etag != response.etag)
                    throw std::runtime_error("channel object changed during ranged read");
                std::memcpy(dst, body.data(), size);
            });
    }

private:
    std::pair<std::string, std::optional<std::size_t>> location(const std::string& key) const
    {
        if (layout_.ratios.empty()) return {joinLocation(root_, prefix_ + key), std::nullopt};
        Index coordinates;
        std::size_t begin = 0;
        for (std::size_t axis = 0; axis < layout_.ratios.size(); ++axis) {
            const auto stop = key.find(separator_, begin);
            coordinates.push_back(std::stoll(key.substr(begin, stop - begin)));
            begin = stop == std::string::npos ? key.size() : stop + separator_.size();
        }
        const auto [shard, position] = layout_.locate(coordinates);
        std::string shard_key;
        for (auto coordinate : shard) {
            if (!shard_key.empty()) shard_key += separator_;
            shard_key += std::to_string(coordinate);
        }
        return {joinLocation(root_, prefix_ + shard_key), position};
    }
    static void validate_range(const HttpResponse& response, std::uint64_t offset,
                               std::uint64_t length, std::uint64_t total)
    {
        unsigned long long first = 0, last = 0, size = 0;
        if (std::sscanf(response.content_range.c_str(), "bytes %llu-%llu/%llu", &first, &last, &size) != 3 ||
            first != offset || last < first || last-first+1 != length || size != total)
            throw std::runtime_error("invalid HTTP Content-Range for shard read");
    }

    static bool local_read(const std::string& path, std::vector<std::uint8_t>& body,
                           std::optional<std::uint64_t> offset, std::size_t length, std::size_t maximum)
    {
        int fd = platform::open_file(path, O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT) return false;
            throw std::system_error(errno, std::generic_category(), "open " + path);
        }
        try {
            platform::FileStat stat;
            if (platform::fstat(fd, stat) != 0) throw std::system_error(errno, std::generic_category(), "fstat " + path);
            if (!offset) length = static_cast<std::size_t>(stat.size);
            if (length > maximum || offset.value_or(0) > std::uint64_t(stat.size) ||
                length > std::uint64_t(stat.size)-offset.value_or(0))
                throw std::runtime_error("truncated or oversized Zarr object: " + path);
            body.resize(length);
            pread_exact(fd, body.data(), length, static_cast<std::int64_t>(offset.value_or(0)), path);
        } catch (...) { ::close(fd); throw; }
        ::close(fd);
        return true;
    }

    std::shared_ptr<const IndexData> shard_index(const std::string& path)
    {
        std::lock_guard lock(index_mutex_);
        // Remote stores can reuse one index (at most 8 MiB).
        if (cached_path_ == path && cached_index_) return cached_index_;
        auto index = std::make_shared<IndexData>();
        ++requests_;
        auto response = http_->get_range(path, index->bytes,
            layout_.indexEnd ? "-"+std::to_string(layout_.indexBytes) : "0-"+std::to_string(layout_.indexBytes-1), layout_.indexBytes);
        bytes_ += response.bytes;
        if (response.status == 404) return nullptr;
        unsigned long long first = 0, last = 0, total = 0;
        if (std::sscanf(response.content_range.c_str(), "bytes %llu-%llu/%llu", &first, &last, &total) != 3 || total < layout_.indexBytes)
            throw std::runtime_error("invalid HTTP shard index Content-Range");
        index->start = layout_.indexEnd ? total-layout_.indexBytes : 0;
        index->total = total;
        index->etag = response.etag;
        validate_range(response, index->start, layout_.indexBytes, total);
        layout_.validate(index->bytes, index->total);
        if (layout_.indexBytes <= (8ull<<20)) {
            cached_path_ = path;
            cached_index_ = index;
        }
        return index;
    }

    std::string root_, prefix_, separator_;
    ShardLayout layout_;
    HttpClient* http_;
    std::atomic<std::uint64_t>& requests_;
    std::atomic<std::uint64_t>& bytes_;
    std::mutex index_mutex_;
    std::string cached_path_;
    std::shared_ptr<const IndexData> cached_index_;
};
} // namespace vcr
