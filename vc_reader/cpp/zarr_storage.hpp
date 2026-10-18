// Zarr v3 objects and indexed shards, over the shared HTTP/local I/O path.
#pragma once
#include "chunk_io.hpp"
#include "http.hpp"
#include <array>
#include <atomic>
#include <limits>
#include <optional>

namespace vcr {

inline std::uint64_t little_uint(const std::uint8_t* p, int bytes)
{
    std::uint64_t value = 0;
    for (int i = 0; i < bytes; ++i) value |= std::uint64_t(p[i]) << (8*i);
    return value;
}

inline std::uint32_t crc32c(const std::uint8_t* data, std::size_t size)
{
    static const auto table = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            auto c = i;
            for (int bit = 0; bit < 8; ++bit) c = (c >> 1) ^ ((c & 1) ? 0x82f63b78u : 0u);
            values[i] = c;
        }
        return values;
    }();
    std::uint32_t crc = ~0u;
    for (std::size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 255] ^ (crc >> 8);
    return ~crc;
}

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
          shards_(std::move(shards)), end_(index_end), crc_(index_crc), http_(http),
          requests_(requests), bytes_(bytes)
    {
        // Encoded compression may add overhead, but cannot allocate a shard-sized
        // buffer from an unchecked index. Native chunks retain their normal budget.
        max_payload_ = chunk_bytes > (std::numeric_limits<std::size_t>::max()-(1<<20))/2 ?
                       chunk_bytes : 2*chunk_bytes+(1<<20);
        if (!shards_.empty()) {
            index_size_ = 16;
            for (auto n : shards_) {
                if (n <= 0 || std::uint64_t(n) > (64ull<<20)/index_size_)
                    throw std::invalid_argument("shard index exceeds 64 MiB");
                index_size_ *= static_cast<std::size_t>(n);
            }
            index_size_ += crc_ ? 4 : 0;
            if (index_size_ > (64ull<<20)) throw std::invalid_argument("shard index exceeds 64 MiB");
        }
    }

    bool read(const std::string& key, std::vector<std::uint8_t>& body)
    {
        if (shards_.empty()) {
            const auto path = root_ + "/" + prefix_ + key;
            if (http_) {
                ++requests_;
                auto result = http_->get_stream(path, [&] { body.clear(); }, [&](const char* p, std::size_t n) {
                    if (n > max_payload_-body.size()) throw std::runtime_error("encoded chunk exceeds bounded payload size");
                    body.insert(body.end(), p, p+n);
                });
                bytes_ += result.bytes;
                return result.status == 200;
            }
            return local_read(path, body, std::nullopt, 0, max_payload_);
        }
        std::string shard_key;
        std::size_t position = 0, begin = 0;
        for (std::size_t axis = 0; axis < shards_.size(); ++axis) {
            const auto stop = key.find(separator_, begin);
            const auto coordinate = std::stoull(key.substr(begin, stop-begin));
            if (axis) shard_key += separator_;
            shard_key += std::to_string(coordinate/shards_[axis]);
            position = position*shards_[axis] + coordinate%shards_[axis];
            begin = stop == std::string::npos ? key.size() : stop+separator_.size();
        }
        const auto path = root_ + "/" + prefix_ + shard_key;
        auto index = shard_index(path);
        if (!index) return false;
        const auto offset = little_uint(index->bytes.data()+16*position, 8);
        const auto length = little_uint(index->bytes.data()+16*position+8, 8);
        constexpr auto missing = std::numeric_limits<std::uint64_t>::max();
        if (offset == missing && length == missing) return false;
        const auto data_end = end_ ? index->start : index->total;
        if (offset > data_end || length > data_end-offset || (!end_ && offset < index_size_) ||
            !length || length > max_payload_)
            throw std::runtime_error("invalid or oversized shard chunk offset/length");
        if (!http_) return local_read(path, body, offset, length, max_payload_);
        ++requests_;
        auto response = http_->get_range(path, body, std::to_string(offset)+"-"+std::to_string(offset+length-1),
                                         length, index->etag);
        bytes_ += response.bytes;
        if (response.status == 404) throw std::runtime_error("shard disappeared after index read");
        validate_range(response, offset, length, index->total);
        return true;
    }

private:
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
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            if (errno == ENOENT) return false;
            throw std::system_error(errno, std::generic_category(), "open " + path);
        }
        try {
            struct stat stat{};
            if (::fstat(fd, &stat) != 0) throw std::system_error(errno, std::generic_category(), "fstat " + path);
            if (!offset) length = static_cast<std::size_t>(stat.st_size);
            if (length > maximum || offset.value_or(0) > std::uint64_t(stat.st_size) ||
                length > std::uint64_t(stat.st_size)-offset.value_or(0))
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
        // Immutable remote stores can reuse one index (at most 8 MiB). Local
        // indexes are reread to respect modifications between reads.
        if (http_ && cached_path_ == path && cached_index_) return cached_index_;
        auto index = std::make_shared<IndexData>();
        if (http_) {
            ++requests_;
            auto response = http_->get_range(path, index->bytes,
                end_ ? "-"+std::to_string(index_size_) : "0-"+std::to_string(index_size_-1), index_size_);
            bytes_ += response.bytes;
            if (response.status == 404) return nullptr;
            unsigned long long first = 0, last = 0, total = 0;
            if (std::sscanf(response.content_range.c_str(), "bytes %llu-%llu/%llu", &first, &last, &total) != 3 || total < index_size_)
                throw std::runtime_error("invalid HTTP shard index Content-Range");
            index->start = end_ ? total-index_size_ : 0;
            index->total = total;
            index->etag = response.etag;
            validate_range(response, index->start, index_size_, total);
        } else {
            std::error_code error;
            auto total = std::filesystem::file_size(path, error);
            if (error == std::errc::no_such_file_or_directory) return nullptr;
            if (error) throw std::system_error(error, path);
            if (total < index_size_) throw std::runtime_error("truncated shard index");
            index->total = total;
            index->start = end_ ? total-index_size_ : 0;
            if (!local_read(path, index->bytes, index->start, index_size_, index_size_)) return nullptr;
        }
        if (crc_ && crc32c(index->bytes.data(), index_size_-4) != little_uint(index->bytes.data()+index_size_-4, 4))
            throw std::runtime_error("Zarr shard index CRC32C mismatch");
        if (http_ && index_size_ <= (8ull<<20)) {
            cached_path_ = path;
            cached_index_ = index;
        }
        return index;
    }

    std::string root_, prefix_, separator_;
    Index shards_;
    bool end_, crc_;
    HttpClient* http_;
    std::atomic<std::uint64_t>& requests_;
    std::atomic<std::uint64_t>& bytes_;
    std::size_t index_size_ = 0, max_payload_;
    std::mutex index_mutex_;
    std::string cached_path_;
    std::shared_ptr<const IndexData> cached_index_;
};
} // namespace vcr
