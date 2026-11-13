// Select contiguous channel planes without retaining another encoded-byte cache.
#pragma once
#include "vc_reader/exception_type.hpp"
#include "codecs.hpp"
#include <functional>
#include <numeric>

namespace vcr {

class VCR_EXCEPTION_TYPE ChannelDecodeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct DecodedChannel {
    std::size_t index;
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
};

struct ChannelDecodeResult {
    std::vector<DecodedChannel> channels;
    std::size_t bytesRead = 0;
    std::size_t bytesDecoded = 0; // decoded blocks, including unavoidable block padding
    bool partial = false;
};

namespace channel_detail {
inline std::uint32_t u32(const std::uint8_t* p)
{
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 |
           std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24;
}
inline void put32(std::uint8_t* p, std::size_t value)
{
    for (unsigned i = 0; i != 4; ++i) p[i] = std::uint8_t(value >> (8*i));
}
inline std::vector<std::size_t> checked(std::span<const std::size_t> channels, std::size_t count)
{
    std::vector<std::size_t> out(channels.begin(), channels.end());
    for (auto c : out) if (c >= count) throw std::out_of_range("channel outside decoded chunk");
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
} // namespace channel_detail

// readRange must fill exactly `length` bytes, or throw. It may use pread, an HTTP
// Range request or an already fetched payload. Channel indices are chunk-local.
// Raw/Blosc fetch only requested blocks; other codecs decode once, retaining the
// requested channels and every relevant channel encountered during that decode.
template<class ReadRange>
ChannelDecodeResult decode_channels(const ChunkCodec& codec, std::size_t decodedBytes,
    std::size_t channelBytes, std::span<const std::size_t> requested,
    std::span<const std::size_t> relevant, std::size_t encodedBytes, ReadRange&& readRange)
{
    using namespace channel_detail;
    if (!channelBytes || !decodedBytes || decodedBytes % channelBytes)
        throw std::invalid_argument("channel planes must partition the decoded chunk");
    const auto count = decodedBytes / channelBytes;
    auto selected = checked(requested, count);
    const auto tagged = checked(relevant, count);
    ChannelDecodeResult result;
    if (selected.empty()) return result;
    const auto read = [&](std::size_t offset, std::size_t length, std::uint8_t* dst) {
        if (offset > encodedBytes || length > encodedBytes-offset)
            throw ChannelDecodeError("truncated channel payload");
        if (length) readRange(offset, length, dst);
        result.bytesRead += length;
    };
    const auto allocate = [&] {
        for (auto c : selected)
            result.channels.push_back({c, std::make_shared<std::vector<std::uint8_t>>(channelBytes)});
    };
    const auto writable = [](DecodedChannel& c) {
        return std::const_pointer_cast<std::vector<std::uint8_t>>(c.bytes)->data();
    };
    const auto direct = [&](std::size_t offset) {
        allocate();
        for (std::size_t begin = 0; begin < selected.size();) {
            std::size_t end = begin+1;
            while (end < selected.size() && selected[end] == selected[end-1]+1) ++end;
            const auto size = (end-begin)*channelBytes;
            if (end == begin+1) read(offset+selected[begin]*channelBytes, size, writable(result.channels[begin]));
            else {
                std::vector<std::uint8_t> bytes(size);
                read(offset+selected[begin]*channelBytes, size, bytes.data());
                for (auto i = begin; i < end; ++i)
                    std::memcpy(writable(result.channels[i]), bytes.data()+(i-begin)*channelBytes, channelBytes);
            }
            begin = end;
        }
        result.bytesDecoded = selected.size()*channelBytes;
        result.partial = selected.size() < count;
    };
    if (codec == Codec::Raw) {
        if (encodedBytes != decodedBytes) throw ChannelDecodeError("raw channel payload size mismatch");
        direct(0);
        return result;
    }
    if (codec == Codec::Blosc) {
        constexpr std::size_t headerSize = BLOSC_MIN_HEADER_LENGTH;
        std::array<std::uint8_t, headerSize> header{};
        read(0, headerSize, header.data());
        const std::size_t nbytes = u32(header.data()+4), blockSize = u32(header.data()+8),
                          cbytes = u32(header.data()+12);
        if (header[0] != BLOSC_VERSION_FORMAT || (header[2] & 0x08) || !header[3] ||
            nbytes != decodedBytes || !blockSize || blockSize > nbytes || blockSize > BLOSC_MAX_BLOCKSIZE ||
            cbytes < headerSize || cbytes > encodedBytes || cbytes > std::size_t(INT_MAX) ||
            nbytes > std::size_t(INT_MAX))
            throw ChannelDecodeError("invalid Blosc channel header");
        if (header[2] & BLOSC_MEMCPYED) {
            if (cbytes != headerSize+nbytes) throw ChannelDecodeError("invalid Blosc memcpy size");
            direct(headerSize);
            return result;
        }
        const auto blocks = (nbytes-1)/blockSize+1;
        if (blocks > (cbytes-headerSize)/4)
            throw ChannelDecodeError("truncated Blosc block directory");
        const auto payloadStart = headerSize+blocks*4;
        std::vector<std::uint8_t> directory(blocks*4);
        read(headerSize, directory.size(), directory.data());
        std::vector<std::size_t> starts(blocks), ends(blocks), physical(blocks);
        for (std::size_t b = 0; b < blocks; ++b) {
            starts[b] = u32(directory.data()+4*b);
            if (starts[b] < payloadStart || starts[b] >= cbytes)
                throw ChannelDecodeError("invalid Blosc block offset");
        }
        std::iota(physical.begin(), physical.end(), 0);
        std::sort(physical.begin(), physical.end(), [&](auto a, auto b) { return starts[a] < starts[b]; });
        for (std::size_t i = 0; i < blocks; ++i) {
            const auto b = physical[i];
            ends[b] = i+1 < blocks ? starts[physical[i+1]] : cbytes;
            if (ends[b] <= starts[b]) throw ChannelDecodeError("overlapping Blosc blocks");
        }
        std::vector<bool> needed(blocks, false);
        for (auto c : selected)
            for (auto b = c*channelBytes/blockSize; b <= ((c+1)*channelBytes-1)/blockSize; ++b)
                needed[b] = true;
        // A relevant plane wholly covered by the minimum required blocks is free
        // to retain. Do not pull in another block merely to prefetch that plane.
        for (auto c : tagged) {
            bool covered = true;
            for (auto b = c*channelBytes/blockSize; b <= ((c+1)*channelBytes-1)/blockSize; ++b)
                covered &= needed[b];
            if (covered) selected.push_back(c);
        }
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        allocate();
        std::vector<std::uint8_t> frame, decoded;
        // Physical block order may differ from logical order (parallel encoders).
        // Coalesce adjacent selected blocks into one I/O operation.
        for (std::size_t i = 0; i < blocks;) {
            if (!needed[physical[i]]) { ++i; continue; }
            auto end = i+1;
            while (end < blocks && needed[physical[end]]) ++end;
            const auto firstByte = starts[physical[i]], lastByte = ends[physical[end-1]];
            std::vector<std::uint8_t> compressed(lastByte-firstByte);
            read(firstByte, compressed.size(), compressed.data());
            for (auto j = i; j < end; ++j) {
                const auto b = physical[j], plainStart = b*blockSize,
                           plainSize = std::min(blockSize, nbytes-plainStart),
                           compressedSize = ends[b]-starts[b];
                // A one-block frame preserves the original typesize, filters,
                // compressor and blocksize (including a short final block).
                // The existing bounded decoder handles all split codecs safely.
                frame.resize(headerSize+4+compressedSize);
                std::copy(header.begin(), header.end(), frame.begin());
                put32(frame.data()+4, plainSize);
                put32(frame.data()+12, frame.size());
                put32(frame.data()+headerSize, headerSize+4);
                std::memcpy(frame.data()+headerSize+4, compressed.data()+starts[b]-firstByte, compressedSize);
                decoded.resize(blockSize);
                if (blosc_decompress_ctx(frame.data(), decoded.data(), decoded.size(), 1) != int(plainSize))
                    throw ChannelDecodeError("Blosc channel block decode failed");
                result.bytesDecoded += plainSize;
                for (auto& channel : result.channels) {
                    const auto lo = std::max(plainStart, channel.index*channelBytes),
                               hi = std::min(plainStart+plainSize, (channel.index+1)*channelBytes);
                    if (lo < hi) std::memcpy(writable(channel)+lo-channel.index*channelBytes,
                                             decoded.data()+lo-plainStart, hi-lo);
                }
            }
            i = end;
        }
        result.partial = result.bytesDecoded < decodedBytes;
        return result;
    }
    selected.insert(selected.end(), tagged.begin(), tagged.end());
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    std::vector<std::uint8_t> encoded(encodedBytes), decoded(decodedBytes);
    read(0, encoded.size(), encoded.data());
    try { decode(codec, encoded.data(), encoded.size(), decoded.data(), decoded.size()); }
    catch (const std::runtime_error& error) { throw ChannelDecodeError(error.what()); }
    allocate();
    for (auto& channel : result.channels)
        std::memcpy(writable(channel), decoded.data()+channel.index*channelBytes, channelBytes);
    result.bytesDecoded = decodedBytes;
    return result;
}

} // namespace vcr
