// Shared indexed-shard geometry, checksums and payload validation.
#pragma once
#include "chunk_io.hpp"
#include <array>
#include <fstream>
#include <limits>
#include <optional>
#include <utility>

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


class ShardLayout {
public:
    Index ratios;
    bool indexEnd, indexCrc;
    std::size_t indexBytes = 0, maxPayload;

    ShardLayout(Index chunks, bool end, bool crc, std::size_t chunkBytes)
        : ratios(std::move(chunks)), indexEnd(end), indexCrc(crc),
          maxPayload(chunkBytes > (SIZE_MAX - (1 << 20)) / 2 ? chunkBytes : 2 * chunkBytes + (1 << 20))
    {
        if (ratios.empty()) return;
        indexBytes = 16;
        for (auto n : ratios) {
            if (n <= 0 || std::uint64_t(n) > (64ull << 20) / indexBytes)
                throw std::invalid_argument("shard index exceeds 64 MiB");
            indexBytes *= std::size_t(n);
        }
        indexBytes += indexCrc ? 4 : 0;
        if (indexBytes > (64ull << 20))
            throw std::invalid_argument("shard index exceeds 64 MiB");
    }

    std::pair<Index, std::size_t> locate(const Index& chunk) const
    {
        if (chunk.size() != ratios.size())
            throw std::invalid_argument("shard coordinate rank mismatch");
        Index shard(chunk.size());
        std::size_t position = 0;
        for (std::size_t axis = 0; axis < chunk.size(); ++axis) {
            if (chunk[axis] < 0) throw std::invalid_argument("negative shard coordinate");
            shard[axis] = chunk[axis] / ratios[axis];
            position = position * ratios[axis] + chunk[axis] % ratios[axis];
        }
        return {std::move(shard), position};
    }

    void validate(const std::vector<std::uint8_t>& index, std::uint64_t total) const
    {
        if (index.size() != indexBytes || total < indexBytes)
            throw std::runtime_error("truncated shard index");
        if (indexCrc && crc32c(index.data(), indexBytes - 4) != little_uint(index.data() + indexBytes - 4, 4))
            throw std::runtime_error("Zarr shard index CRC32C mismatch");
    }

    using Entry = std::pair<std::uint64_t, std::uint64_t>;
    std::optional<Entry> entry(const std::vector<std::uint8_t>& index, std::size_t position,
                               std::uint64_t total) const
    {
        if (position >= indexBytes / 16 || index.size() != indexBytes)
            throw std::out_of_range("shard index position outside grid");
        const auto offset = little_uint(index.data() + 16 * position, 8);
        const auto length = little_uint(index.data() + 16 * position + 8, 8);
        constexpr auto missing = std::numeric_limits<std::uint64_t>::max();
        if (offset == missing && length == missing) return std::nullopt;
        if (total < indexBytes) throw std::runtime_error("truncated shard index");
        const auto end = indexEnd ? total - indexBytes : total;
        if (offset > end || length > end - offset || (!indexEnd && offset < indexBytes) ||
            !length || length > maxPayload)
            throw std::runtime_error("invalid or oversized shard chunk offset/length");
        return Entry{offset, length};
    }
};

// Keep index and payload on the same opened file across atomic replacements.
class LocalShard {
public:
    LocalShard(const std::filesystem::path& path, const ShardLayout& layout)
        : layout_(layout)
    {
        errno = 0;
        file_.open(path, std::ios::binary);
        if (!file_) {
            // Use the failed open's result: checking existence afterward races
            // a writer publishing the first version of this shard.
            if (errno == ENOENT) return;
            throw std::runtime_error("cannot open shard " + path.string());
        }
        file_.seekg(0, std::ios::end);
        const auto size = file_.tellg();
        if (size < 0 || std::uint64_t(size) < layout_.indexBytes)
            throw std::runtime_error("truncated shard index");
        total = std::uint64_t(size);
        read(layout_.indexEnd ? total - layout_.indexBytes : 0, layout_.indexBytes, index);
        layout_.validate(index, total);
    }

    bool readEntry(std::size_t position, std::vector<std::uint8_t>& body)
    {
        if (index.empty()) return false;
        const auto e = layout_.entry(index, position, total);
        if (!e) return false;
        read(e->first, e->second, body);
        return true;
    }

    std::vector<std::uint8_t> index;
    std::uint64_t total = 0;

private:
    void read(std::uint64_t offset, std::size_t length, std::vector<std::uint8_t>& body)
    {
        body.resize(length);
        file_.seekg(std::streamoff(offset));
        file_.read(reinterpret_cast<char*>(body.data()), std::streamsize(length));
        if (!file_) throw std::runtime_error("truncated shard payload");
    }
    const ShardLayout& layout_;
    std::ifstream file_;
};
}  // namespace vcr
