// Zarr v3 objects and indexed shards, over the shared HTTP/local I/O path.
#pragma once
#include "chunk_io.hpp"
#include "http.hpp"
#include "lru.hpp"
#include "thread_pool.hpp"
#include "shard_index.hpp"
#include "channel_decode.hpp"
#include <array>
#include <atomic>
#include <limits>
#include <map>
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
        if (!http_) return LocalShard(utf8Path(path), layout_).readEntry(*position, body);
        auto index = shard_index(path);
        if (!index) return false;
        const auto entry = layout_.entry(index->bytes, *position, index->total);
        if (!entry) return false;
        const auto [offset, length] = *entry;
        read_shard_range(path, index->total, index->etag, offset, length, body);
        return true;
    }

    static constexpr std::size_t max_merged_bytes = 512 << 10;
    static constexpr std::size_t max_batch_chunks = 16;

    bool can_batch() const
    {
        return http_ && !layout_.ratios.empty() && !sftp::isSftpUrl(root_);
    }

    // The caller owns these remote misses. Keep the index alive only while
    // planning its shard, then read the bounded groups on the existing pool.
    // An empty payload denotes a missing shard or chunk, never valid data.
    std::vector<std::vector<std::uint8_t>> read_batch(const std::vector<std::string>& keys, ThreadPool& pool)
    {
        if (keys.size() > max_batch_chunks)
            throw std::invalid_argument("shard batch exceeds chunk limit");
        struct Member {
            std::uint64_t start, stop;
            std::size_t slot;
        };
        struct Range {
            const std::string* path;
            std::uint64_t total, start, stop;
            std::string etag;
            std::vector<Member> members;
        };
        std::map<std::string, std::vector<std::pair<std::size_t, std::size_t>>> shards;
        for (std::size_t slot = 0; slot < keys.size(); ++slot) {
            auto [path, position] = location(keys[slot]);
            shards[std::move(path)].emplace_back(*position, slot);
        }

        std::vector<Range> ranges;
        for (const auto& [path, positions] : shards) {
            auto index = shard_index(path);
            if (!index) continue;
            std::vector<Member> members;
            for (const auto& [position, slot] : positions) {
                if (auto entry = layout_.entry(index->bytes, position, index->total))
                    members.push_back({entry->first, entry->first + entry->second, slot});
            }
            std::sort(members.begin(), members.end(), [](const Member& a, const Member& b) {
                return a.start < b.start;
            });
            const auto first_range = ranges.size();
            const bool strong_etag = !index->etag.empty() && !index->etag.starts_with("W/");
            for (const auto& member : members) {
                if (strong_etag && ranges.size() > first_range) {
                    auto& last = ranges.back();
                    const auto stop = std::max(last.stop, member.stop);
                    if (member.start <= last.stop && stop - last.start <= max_merged_bytes) {
                        last.stop = stop;
                        last.members.push_back(member);
                        continue;
                    }
                }
                ranges.push_back({&path, index->total, member.start, member.stop, index->etag, {member}});
            }
        }

        std::vector<std::vector<std::uint8_t>> bodies(keys.size());
        pool.parallel_for(ranges.size(), [&](std::size_t i) {
            const auto& range = ranges[i];
            std::vector<std::uint8_t> body;
            try {
                read_shard_range(*range.path, range.total, range.etag, range.start, range.stop - range.start, body);
            } catch (const HttpRangeRejected&) {
                if (range.members.size() == 1) throw;
                for (const auto& member : range.members)
                    read_shard_range(*range.path, range.total, range.etag, member.start,
                                     member.stop - member.start, bodies[member.slot]);
                return;
            }
            if (range.members.size() == 1) {
                bodies[range.members.front().slot] = std::move(body);
                return;
            }
            for (const auto& member : range.members) {
                bodies[member.slot].assign(body.data() + (member.start - range.start),
                                           body.data() + (member.stop - range.start));
            }
        });
        return bodies;
    }

    std::optional<ChannelDecodeResult> read_channels(const std::string& key, const ChunkCodec& codec,
        std::size_t decodedBytes, std::size_t channelBytes, std::span<const std::size_t> requested,
        std::span<const std::size_t> relevant)
    {
        const auto [path, position] = location(key);
        if (!http_) {
            if (position) {
                LocalShard shard(utf8Path(path), layout_);
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
        int end = 0;
        if (std::sscanf(response.content_range.c_str(), "bytes %llu-%llu/%llu%n", &first, &last, &size, &end) != 3 ||
            std::size_t(end) != response.content_range.size() || first != offset ||
            last < first || last >= size || last-first+1 != length || size != total)
            throw std::runtime_error("invalid HTTP Content-Range for shard read");
    }

    void read_shard_range(const std::string& path, std::uint64_t total, const std::string& etag,
                          std::uint64_t offset, std::size_t length, std::vector<std::uint8_t>& body)
    {
        ++requests_;
        body.reserve(length);
        auto response = http_->get_range(path, body, std::to_string(offset) + "-" + std::to_string(offset + length - 1),
                                         length, etag);
        bytes_ += response.bytes;
        if (response.status == 404) throw std::runtime_error("shard disappeared after index read");
        validate_range(response, offset, length, total);
        if (!etag.empty() && !etag.starts_with("W/") && !response.etag.empty() && response.etag != etag)
            throw std::runtime_error("shard changed during ranged read");
    }

    static bool local_read(const std::string& path, std::vector<std::uint8_t>& body,
                           std::optional<std::uint64_t> offset, std::size_t length, std::size_t maximum)
    {
        int fd = platform::open_file(utf8Path(path), O_RDONLY);
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
        if (auto* cached = indexes_.get(path)) return *cached;
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
        indexes_.put(path, index, layout_.indexBytes);
        return index;
    }

    std::string root_, prefix_, separator_;
    ShardLayout layout_;
    HttpClient* http_;
    std::atomic<std::uint64_t>& requests_;
    std::atomic<std::uint64_t>& bytes_;
    std::mutex index_mutex_;
    Lru<std::shared_ptr<const IndexData>> indexes_{8 << 20};
};
} // namespace vcr
