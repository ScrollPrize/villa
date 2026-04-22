#include "vc/core/types/VcDataset.hpp"

#include <utils/zarr.hpp>
#include "utils/Json.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <numeric>
#include <stdexcept>

#include <blosc.h>
#include <zstd.h>
#include <lz4.h>
#include <zlib.h>

namespace vc {

// ============================================================================
// Compressor configuration (parsed from .zarray JSON)
// ============================================================================

enum class CompressorId { None, Blosc, Zstd, Lz4, Gzip };

struct CompressorConfig {
    CompressorId id = CompressorId::None;
    // Blosc params
    std::string blosc_cname = "lz4";
    int blosc_clevel = 5;
    int blosc_shuffle = 1;
    int blosc_typesize = 1;
    int blosc_blocksize = 0;
    // Zstd/Gzip level
    int level = 3;
};

namespace {

constexpr int kCompressionThreads = 1;

void ensureBloscInitialized()
{
    static std::once_flag once;
    std::call_once(once, []() {
        blosc_init();
        std::atexit([]() { blosc_destroy(); });
    });
}

int checkedInt(size_t value, const char* name)
{
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds supported size");
    }
    return static_cast<int>(value);
}

uInt checkedUInt(size_t value, const char* name)
{
    if (value > static_cast<size_t>(std::numeric_limits<uInt>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds supported size");
    }
    return static_cast<uInt>(value);
}

std::vector<std::byte> bloscCompress(std::span<const std::byte> input,
                                     const CompressorConfig& cfg)
{
    ensureBloscInitialized();

    std::vector<std::byte> output(input.size() + BLOSC_MAX_OVERHEAD);
    const int rc = blosc_compress_ctx(cfg.blosc_clevel,
                                      cfg.blosc_shuffle,
                                      cfg.blosc_typesize,
                                      input.size(),
                                      input.data(),
                                      output.data(),
                                      output.size(),
                                      cfg.blosc_cname.c_str(),
                                      cfg.blosc_blocksize,
                                      kCompressionThreads);
    if (rc <= 0) {
        throw std::runtime_error("blosc_compress_ctx failed with code " + std::to_string(rc));
    }
    output.resize(static_cast<size_t>(rc));
    return output;
}

std::vector<std::byte> bloscDecompress(std::span<const std::byte> input, size_t outputSize)
{
    ensureBloscInitialized();

    std::vector<std::byte> output(outputSize);
    const int rc = blosc_decompress(input.data(), output.data(), outputSize);
    if (rc < 0) {
        if (input.size() == outputSize) {
            std::memcpy(output.data(), input.data(), outputSize);
            return output;
        }
        throw std::runtime_error("blosc_decompress failed with code " + std::to_string(rc));
    }
    return output;
}

std::vector<std::byte> zstdCompress(std::span<const std::byte> input, const CompressorConfig& cfg)
{
    const size_t bound = ZSTD_compressBound(input.size());
    std::vector<std::byte> output(bound);
    const size_t rc = ZSTD_compress(output.data(), bound, input.data(), input.size(), cfg.level);
    if (ZSTD_isError(rc)) {
        throw std::runtime_error(std::string("ZSTD_compress failed: ") + ZSTD_getErrorName(rc));
    }
    output.resize(rc);
    return output;
}

std::vector<std::byte> zstdDecompress(std::span<const std::byte> input, size_t outputSize)
{
    std::vector<std::byte> output(outputSize);
    const size_t rc = ZSTD_decompress(output.data(), outputSize, input.data(), input.size());
    if (ZSTD_isError(rc)) {
        throw std::runtime_error(std::string("ZSTD_decompress failed: ") + ZSTD_getErrorName(rc));
    }
    if (rc != outputSize) {
        throw std::runtime_error("ZSTD_decompress returned unexpected byte count");
    }
    return output;
}

std::vector<std::byte> lz4Compress(std::span<const std::byte> input, const CompressorConfig& cfg)
{
    const int inputSize = checkedInt(input.size(), "LZ4 input");
    const int bound = LZ4_compressBound(inputSize);
    std::vector<std::byte> output(sizeof(uint32_t) + static_cast<size_t>(bound));

    const uint32_t originalSize = static_cast<uint32_t>(input.size());
    std::memcpy(output.data(), &originalSize, sizeof(originalSize));

    const int rc = LZ4_compress_fast(reinterpret_cast<const char*>(input.data()),
                                     reinterpret_cast<char*>(output.data() + sizeof(uint32_t)),
                                     inputSize,
                                     bound,
                                     std::max(cfg.level, 1));
    if (rc <= 0) {
        throw std::runtime_error("LZ4_compress_fast failed");
    }
    output.resize(sizeof(uint32_t) + static_cast<size_t>(rc));
    return output;
}

std::vector<std::byte> lz4Decompress(std::span<const std::byte> input, size_t outputSize)
{
    if (input.size() < sizeof(uint32_t)) {
        throw std::runtime_error("LZ4 compressed data too short");
    }

    uint32_t originalSize = 0;
    std::memcpy(&originalSize, input.data(), sizeof(originalSize));
    if (originalSize > outputSize) {
        throw std::runtime_error("LZ4 original size exceeds output buffer");
    }

    std::vector<std::byte> output(outputSize);
    const int rc = LZ4_decompress_safe(
        reinterpret_cast<const char*>(input.data() + sizeof(uint32_t)),
        reinterpret_cast<char*>(output.data()),
        checkedInt(input.size() - sizeof(uint32_t), "LZ4 compressed payload"),
        checkedInt(originalSize, "LZ4 output"));
    if (rc < 0) {
        throw std::runtime_error("LZ4_decompress_safe failed");
    }
    return output;
}

std::vector<std::byte> gzipCompress(std::span<const std::byte> input, const CompressorConfig& cfg)
{
    z_stream stream{};
    if (deflateInit2(&stream, cfg.level, Z_DEFLATED, 16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("deflateInit2 failed");
    }

    std::vector<std::byte> output(deflateBound(&stream, checkedUInt(input.size(), "gzip input")));
    stream.avail_in = checkedUInt(input.size(), "gzip input");
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(input.data()));
    stream.avail_out = checkedUInt(output.size(), "gzip output");
    stream.next_out = reinterpret_cast<Bytef*>(output.data());

    const int rc = deflate(&stream, Z_FINISH);
    deflateEnd(&stream);
    if (rc != Z_STREAM_END) {
        throw std::runtime_error("deflate failed with code " + std::to_string(rc));
    }

    output.resize(stream.total_out);
    return output;
}

std::vector<std::byte> gzipDecompress(std::span<const std::byte> input, size_t outputSize)
{
    z_stream stream{};
    if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) {
        throw std::runtime_error("inflateInit2 failed");
    }

    std::vector<std::byte> output(outputSize);
    stream.avail_in = checkedUInt(input.size(), "gzip input");
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(input.data()));
    stream.avail_out = checkedUInt(output.size(), "gzip output");
    stream.next_out = reinterpret_cast<Bytef*>(output.data());

    const int rc = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
    if (rc != Z_STREAM_END && rc != Z_OK) {
        throw std::runtime_error("gzip inflate failed with code " + std::to_string(rc));
    }
    return output;
}

std::vector<std::byte> decompressBytes(const CompressorConfig& cfg,
                                       std::span<const std::byte> input,
                                       size_t outputSize)
{
    switch (cfg.id) {
    case CompressorId::None:
        return std::vector<std::byte>(input.begin(), input.end());
    case CompressorId::Blosc:
        return bloscDecompress(input, outputSize);
    case CompressorId::Zstd:
        return zstdDecompress(input, outputSize);
    case CompressorId::Lz4:
        return lz4Decompress(input, outputSize);
    case CompressorId::Gzip:
        return gzipDecompress(input, outputSize);
    }

    throw std::runtime_error("unsupported zarr compressor");
}

std::vector<std::byte> compressBytes(const CompressorConfig& cfg,
                                     std::span<const std::byte> input)
{
    switch (cfg.id) {
    case CompressorId::None:
        return std::vector<std::byte>(input.begin(), input.end());
    case CompressorId::Blosc:
        return bloscCompress(input, cfg);
    case CompressorId::Zstd:
        return zstdCompress(input, cfg);
    case CompressorId::Lz4:
        return lz4Compress(input, cfg);
    case CompressorId::Gzip:
        return gzipCompress(input, cfg);
    }

    throw std::runtime_error("unsupported zarr compressor");
}

} // namespace

static void fillTypedElements(uint8_t* dst,
                              size_t count,
                              const std::vector<uint8_t>& fillBytes)
{
    if (count == 0) return;
    if (fillBytes.size() == 1) {
        std::memset(dst, fillBytes[0], count);
        return;
    }

    const size_t elemSize = fillBytes.size();
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(dst + i * elemSize, fillBytes.data(), elemSize);
    }
}

static CompressorConfig compressorFromMeta(const utils::ZarrMetadata& meta, int dtypeSize)
{
    CompressorConfig cfg;
    cfg.blosc_typesize = dtypeSize;

    // v2: meta.compressor_id set by parse_zarray
    if (!meta.compressor_id.empty()) {
        if (meta.compressor_id == "blosc") {
            cfg.id = CompressorId::Blosc;
            cfg.blosc_clevel = meta.compression_level > 0 ? meta.compression_level : 5;
        } else if (meta.compressor_id == "zstd") {
            cfg.id = CompressorId::Zstd;
            cfg.level = meta.compression_level > 0 ? meta.compression_level : 3;
        } else if (meta.compressor_id == "lz4") {
            cfg.id = CompressorId::Lz4;
            cfg.level = meta.compression_level > 0 ? meta.compression_level : 1;
        } else if (meta.compressor_id == "gzip" || meta.compressor_id == "zlib") {
            cfg.id = CompressorId::Gzip;
            cfg.level = meta.compression_level > 0 ? meta.compression_level : 5;
        } else {
            throw std::runtime_error("Unsupported zarr compressor: " + meta.compressor_id);
        }
        return cfg;
    }

    // v3: walk codec pipeline for a bytes→bytes codec
    for (const auto& cc : meta.codecs) {
        if (cc.name == "blosc") { cfg.id = CompressorId::Blosc; return cfg; }
        if (cc.name == "zstd")  { cfg.id = CompressorId::Zstd;  return cfg; }
        if (cc.name == "lz4")   { cfg.id = CompressorId::Lz4;   return cfg; }
        if (cc.name == "gzip" || cc.name == "zlib") { cfg.id = CompressorId::Gzip; return cfg; }
    }

    cfg.id = CompressorId::None;
    return cfg;
}

static utils::ZarrArray::Codec codecFromConfig(const CompressorConfig& cfg)
{
    if (cfg.id == CompressorId::None) {
        return {};
    }

    utils::ZarrArray::Codec codec;
    codec.compress = [cfg](std::span<const std::byte> data) {
        return compressBytes(cfg, data);
    };
    codec.decompress = [cfg](std::span<const std::byte> data, std::size_t outSize) {
        return decompressBytes(cfg, data, outSize);
    };
    return codec;
}

// ============================================================================
// VcDataset::Impl
// ============================================================================

struct VcDataset::Impl {
    std::filesystem::path fsPath;
    std::vector<size_t> shape_;
    std::vector<size_t> chunkShape_;
    size_t chunkSize_ = 0;
    VcDtype dtype_ = VcDtype::uint8;
    size_t dtypeSize_ = 1;
    std::string delimiter_ = ".";
    CompressorConfig compressor_;
    std::vector<uint8_t> fillValueBytes_;

    // utils zarr array for chunk I/O
    std::shared_ptr<utils::FileSystemStore> store_;
    std::unique_ptr<utils::ZarrArray> zarrArray_;

    // Build a codec registry covering the compressors we decode (blosc,
    // zstd, lz4, gzip). ZarrArray::open picks the right codec from meta.
    static utils::ZarrArray::CodecRegistry buildCodecRegistry(int dtypeSize) {
        utils::ZarrArray::CodecRegistry reg;
        for (const char* name : {"blosc", "zstd", "lz4", "gzip", "zlib"}) {
            CompressorConfig cfg;
            if      (std::string(name) == "blosc") cfg.id = CompressorId::Blosc;
            else if (std::string(name) == "zstd")  cfg.id = CompressorId::Zstd;
            else if (std::string(name) == "lz4")   cfg.id = CompressorId::Lz4;
            else                                   cfg.id = CompressorId::Gzip;
            cfg.blosc_typesize = dtypeSize;
            reg[name] = codecFromConfig(cfg);
        }
        return reg;
    }

    void open(const std::filesystem::path& path) {
        fsPath = path;
        // Auto-detects v2 (.zarray) and v3 (zarr.json).
        zarrArray_ = std::make_unique<utils::ZarrArray>(
            utils::ZarrArray::open(path, buildCodecRegistry(/*dtypeSize guess*/1)));
        const auto& meta = zarrArray_->metadata();

        shape_.assign(meta.shape.begin(), meta.shape.end());

        // Finest chunk granularity: v3 sharded uses inner (sub_chunks); v2 and
        // v3 unsharded use meta.chunks directly.
        if (meta.shard_config) {
            chunkShape_.assign(meta.shard_config->sub_chunks.begin(),
                               meta.shard_config->sub_chunks.end());
        } else {
            chunkShape_.assign(meta.chunks.begin(), meta.chunks.end());
        }
        chunkSize_ = 1;
        for (auto c : chunkShape_) chunkSize_ *= c;

        if (meta.dtype == utils::ZarrDtype::uint8) {
            dtype_ = VcDtype::uint8;
            dtypeSize_ = 1;
        } else if (meta.dtype == utils::ZarrDtype::uint16) {
            dtype_ = VcDtype::uint16;
            dtypeSize_ = 2;
        } else {
            throw std::runtime_error("Unsupported zarr dtype");
        }

        delimiter_ = meta.dimension_separator.empty() ? "/" : meta.dimension_separator;

        fillValueBytes_.assign(dtypeSize_, 0);
        if (meta.fill_value.has_value()) {
            std::int64_t raw = static_cast<std::int64_t>(*meta.fill_value);
            if (dtype_ == VcDtype::uint8) {
                if (raw < 0) raw = 0;
                if (raw > std::numeric_limits<uint8_t>::max())
                    raw = std::numeric_limits<uint8_t>::max();
                fillValueBytes_[0] = static_cast<uint8_t>(raw);
            } else {
                if (raw < 0) raw = 0;
                if (raw > std::numeric_limits<uint16_t>::max())
                    raw = std::numeric_limits<uint16_t>::max();
                const auto v = static_cast<uint16_t>(raw);
                std::memcpy(fillValueBytes_.data(), &v, sizeof(v));
            }
        }

        compressor_ = compressorFromMeta(meta, static_cast<int>(dtypeSize_));
    }
};

// ============================================================================
// VcDataset public API
// ============================================================================

VcDataset::VcDataset(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>())
{
    impl_->open(path);
}

VcDataset::~VcDataset() = default;
VcDataset::VcDataset(VcDataset&&) noexcept = default;
VcDataset& VcDataset::operator=(VcDataset&&) noexcept = default;

const std::vector<size_t>& VcDataset::shape() const { return impl_->shape_; }
const std::vector<size_t>& VcDataset::defaultChunkShape() const { return impl_->chunkShape_; }
size_t VcDataset::defaultChunkSize() const { return impl_->chunkSize_; }
VcDtype VcDataset::getDtype() const { return impl_->dtype_; }
size_t VcDataset::dtypeSize() const { return impl_->dtypeSize_; }
const std::filesystem::path& VcDataset::path() const { return impl_->fsPath; }
const std::string& VcDataset::delimiter() const { return impl_->delimiter_; }

void VcDataset::decompress(std::span<const uint8_t> compressed,
                            void* output, size_t nElements) const
{
    const size_t outBytes = nElements * impl_->dtypeSize_;
    const auto input = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(compressed.data()),
        compressed.size());

    switch (impl_->compressor_.id) {
        case CompressorId::None:
            std::memcpy(output, compressed.data(), outBytes);
            break;

        case CompressorId::Blosc: {
            ensureBloscInitialized();
            int ret = blosc_decompress(compressed.data(), output, outBytes);
            if (ret < 0) {
                if (compressed.size() == outBytes) {
                    std::memcpy(output, compressed.data(), outBytes);
                    break;
                }
                throw std::runtime_error("blosc_decompress failed with code " +
                                          std::to_string(ret));
            }
            break;
        }

        case CompressorId::Zstd: {
            size_t ret = ZSTD_decompress(output, outBytes, compressed.data(), compressed.size());
            if (ZSTD_isError(ret)) {
                throw std::runtime_error(
                    std::string("ZSTD_decompress failed: ") + ZSTD_getErrorName(ret));
            }
            break;
        }

        case CompressorId::Lz4: {
            const auto bytes = decompressBytes(impl_->compressor_, input, outBytes);
            std::memcpy(output, bytes.data(), outBytes);
            break;
        }

        case CompressorId::Gzip: {
            const auto bytes = decompressBytes(impl_->compressor_, input, outBytes);
            std::memcpy(output, bytes.data(), outBytes);
            break;
        }
    }
}

bool VcDataset::chunkExists(size_t iz, size_t iy, size_t ix) const
{
    // Build chunk path: <basepath>/<iz><delim><iy><delim><ix>
    auto p = impl_->fsPath /
        (std::to_string(iz) + impl_->delimiter_ +
         std::to_string(iy) + impl_->delimiter_ +
         std::to_string(ix));
    return std::filesystem::exists(p);
}

bool VcDataset::readChunk(size_t iz, size_t iy, size_t ix, void* output) const
{
    std::array<size_t, 3> indices = {iz, iy, ix};
    auto result = impl_->zarrArray_->read_chunk(indices);
    if (!result) return false;

    const auto& bytes = *result;
    const size_t expectedBytes = impl_->chunkSize_ * impl_->dtypeSize_;
    if (bytes.size() < expectedBytes) return false;

    std::memcpy(output, bytes.data(), expectedBytes);
    return true;
}

bool VcDataset::readChunkOrFill(size_t iz, size_t iy, size_t ix, void* output) const
{
    if (readChunk(iz, iy, ix, output)) {
        return true;
    }

    auto* outBytes = static_cast<uint8_t*>(output);
    fillTypedElements(outBytes, impl_->chunkSize_, impl_->fillValueBytes_);
    return false;
}

bool VcDataset::writeChunk(size_t iz, size_t iy, size_t ix,
                            const void* input, size_t nbytes)
{
    std::array<size_t, 3> indices = {iz, iy, ix};
    auto data = std::span<const std::byte>(
        static_cast<const std::byte*>(input), nbytes);
    impl_->zarrArray_->write_chunk(indices, data);
    return true;
}

bool VcDataset::removeChunk(size_t iz, size_t iy, size_t ix)
{
    auto p = impl_->fsPath /
        (std::to_string(iz) + impl_->delimiter_ +
         std::to_string(iy) + impl_->delimiter_ +
         std::to_string(ix));

    std::error_code ec;
    const bool removed = std::filesystem::remove(p, ec);
    if (ec) {
        throw std::runtime_error("failed removing chunk: " + p.string());
    }
    return removed;
}

bool VcDataset::readRegion(const std::vector<size_t>& offset,
                            const std::vector<size_t>& regionShape,
                            void* output) const
{
    for (size_t d = 0; d < regionShape.size(); ++d) {
        if (regionShape[d] == 0) return true;
    }
    const size_t ndim = offset.size();
    const auto& chunkShape = impl_->chunkShape_;
    const size_t elemSize = impl_->dtypeSize_;
    auto* outBytes = static_cast<uint8_t*>(output);

    // Total elements per chunk
    size_t chunkElems = 1;
    for (size_t d = 0; d < ndim; ++d) chunkElems *= chunkShape[d];

    // Compute chunk index ranges
    std::vector<size_t> chunkStart(ndim), chunkEnd(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        chunkStart[d] = offset[d] / chunkShape[d];
        chunkEnd[d] = (offset[d] + regionShape[d] - 1) / chunkShape[d];
    }

    // Region strides (C-order)
    std::vector<size_t> regionStrides(ndim);
    regionStrides[ndim - 1] = elemSize;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d)
        regionStrides[d] = regionStrides[d + 1] * regionShape[d + 1];

    // Chunk strides (C-order)
    std::vector<size_t> chunkStrides(ndim);
    chunkStrides[ndim - 1] = elemSize;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d)
        chunkStrides[d] = chunkStrides[d + 1] * chunkShape[d + 1];

    // Iterate over all chunks that overlap the region
    std::vector<size_t> ci(ndim);
    std::function<bool(size_t)> iterChunks = [&](size_t dim) -> bool {
        if (dim == ndim) {
            // Read this chunk
            std::array<size_t, 3> indices;
            for (size_t d = 0; d < ndim && d < 3; ++d) indices[d] = ci[d];
            auto result = impl_->zarrArray_->read_chunk(
                std::span<const size_t>(indices.data(), ndim));
            const uint8_t* src = nullptr;
            if (result) {
                if (result->size() >= chunkElems * elemSize) {
                    src = reinterpret_cast<const uint8_t*>(result->data());
                }
            }

            // Copy overlapping portion to output
            // Compute overlap in each dimension
            std::vector<size_t> copyStart(ndim), copySize(ndim);
            for (size_t d = 0; d < ndim; ++d) {
                size_t chunkGlobalStart = ci[d] * chunkShape[d];
                size_t chunkGlobalEnd = chunkGlobalStart + chunkShape[d];
                size_t regStart = offset[d];
                size_t regEnd = offset[d] + regionShape[d];
                size_t overlapStart = std::max(chunkGlobalStart, regStart);
                size_t overlapEnd = std::min(chunkGlobalEnd, regEnd);
                copyStart[d] = overlapStart;
                copySize[d] = overlapEnd - overlapStart;
            }

            // Copy element rows (innermost dimension contiguous)
            std::vector<size_t> pos(ndim, 0);
            std::function<void(size_t)> copyLoop = [&](size_t d) {
                if (d == ndim - 1) {
                    // Copy a contiguous row
                    size_t srcOff = 0, dstOff = 0;
                    for (size_t dd = 0; dd < ndim; ++dd) {
                        size_t chunkLocal = copyStart[dd] + pos[dd] - ci[dd] * chunkShape[dd];
                        size_t regLocal = copyStart[dd] + pos[dd] - offset[dd];
                        srcOff += chunkLocal * chunkStrides[dd];
                        dstOff += regLocal * regionStrides[dd];
                    }
                    if (src) {
                        std::memcpy(outBytes + dstOff, src + srcOff, copySize[d] * elemSize);
                    } else {
                        fillTypedElements(outBytes + dstOff, copySize[d], impl_->fillValueBytes_);
                    }
                    return;
                }
                for (size_t i = 0; i < copySize[d]; ++i) {
                    pos[d] = i;
                    copyLoop(d + 1);
                }
            };
            copyLoop(0);
            return true;
        }
        for (ci[dim] = chunkStart[dim]; ci[dim] <= chunkEnd[dim]; ++ci[dim]) {
            if (!iterChunks(dim + 1)) return false;
        }
        return true;
    };
    return iterChunks(0);
}

bool VcDataset::writeRegion(const std::vector<size_t>& offset,
                             const std::vector<size_t>& regionShape,
                             const void* data)
{
    for (size_t d = 0; d < regionShape.size(); ++d) {
        if (regionShape[d] == 0) return true;
    }
    const size_t ndim = offset.size();
    const auto& chunkShape = impl_->chunkShape_;
    const size_t elemSize = impl_->dtypeSize_;
    const auto* inBytes = static_cast<const uint8_t*>(data);

    size_t chunkElems = 1;
    for (size_t d = 0; d < ndim; ++d) chunkElems *= chunkShape[d];

    std::vector<size_t> chunkStart(ndim), chunkEnd(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        chunkStart[d] = offset[d] / chunkShape[d];
        chunkEnd[d] = (offset[d] + regionShape[d] - 1) / chunkShape[d];
    }

    std::vector<size_t> regionStrides(ndim);
    regionStrides[ndim - 1] = elemSize;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d)
        regionStrides[d] = regionStrides[d + 1] * regionShape[d + 1];

    std::vector<size_t> chunkStrides(ndim);
    chunkStrides[ndim - 1] = elemSize;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d)
        chunkStrides[d] = chunkStrides[d + 1] * chunkShape[d + 1];

    std::vector<uint8_t> chunkBuf(chunkElems * elemSize);

    std::vector<size_t> ci(ndim);
    std::function<bool(size_t)> iterChunks = [&](size_t dim) -> bool {
        if (dim == ndim) {
            std::array<size_t, 3> indices;
            for (size_t d = 0; d < ndim && d < 3; ++d) indices[d] = ci[d];
            auto idxSpan = std::span<const size_t>(indices.data(), ndim);

            // Check if we're writing a full chunk
            bool fullChunk = true;
            for (size_t d = 0; d < ndim; ++d) {
                size_t chunkGlobalStart = ci[d] * chunkShape[d];
                if (chunkGlobalStart < offset[d] ||
                    chunkGlobalStart + chunkShape[d] > offset[d] + regionShape[d]) {
                    fullChunk = false;
                    break;
                }
            }

            // If partial, read existing chunk first
            if (!fullChunk) {
                auto existing = impl_->zarrArray_->read_chunk(idxSpan);
                if (existing && existing->size() >= chunkElems * elemSize) {
                    std::memcpy(chunkBuf.data(), existing->data(), chunkElems * elemSize);
                } else {
                    std::memset(chunkBuf.data(), 0, chunkElems * elemSize);
                }
            }

            // Compute overlap
            std::vector<size_t> copyStart(ndim), copySize(ndim);
            for (size_t d = 0; d < ndim; ++d) {
                size_t chunkGlobalStart = ci[d] * chunkShape[d];
                size_t chunkGlobalEnd = chunkGlobalStart + chunkShape[d];
                size_t regStart = offset[d];
                size_t regEnd = offset[d] + regionShape[d];
                copyStart[d] = std::max(chunkGlobalStart, regStart);
                copySize[d] = std::min(chunkGlobalEnd, regEnd) - copyStart[d];
            }

            // Copy from input to chunk buffer
            std::vector<size_t> pos(ndim, 0);
            std::function<void(size_t)> copyLoop = [&](size_t d) {
                if (d == ndim - 1) {
                    size_t srcOff = 0, dstOff = 0;
                    for (size_t dd = 0; dd < ndim; ++dd) {
                        size_t regLocal = copyStart[dd] + pos[dd] - offset[dd];
                        size_t chunkLocal = copyStart[dd] + pos[dd] - ci[dd] * chunkShape[dd];
                        srcOff += regLocal * regionStrides[dd];
                        dstOff += chunkLocal * chunkStrides[dd];
                    }
                    std::memcpy(chunkBuf.data() + dstOff, inBytes + srcOff, copySize[d] * elemSize);
                    return;
                }
                for (size_t i = 0; i < copySize[d]; ++i) {
                    pos[d] = i;
                    copyLoop(d + 1);
                }
            };

            copyLoop(0);

            // Write the chunk
            auto byteSpan = std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(chunkBuf.data()),
                chunkElems * elemSize);
            impl_->zarrArray_->write_chunk(idxSpan, byteSpan);
            return true;
        }
        for (ci[dim] = chunkStart[dim]; ci[dim] <= chunkEnd[dim]; ++ci[dim]) {
            if (!iterChunks(dim + 1)) return false;
        }
        return true;
    };
    return iterChunks(0);
}

// ============================================================================
// Factory functions
// ============================================================================

std::vector<std::unique_ptr<VcDataset>> openZarrLevels(
    const std::filesystem::path& zarrRoot)
{
    std::vector<std::string> levelNames;

    // Prefer OME-Zarr .zattrs multiscales metadata for level discovery.
    auto zattrs = readZarrAttributes(zarrRoot);
    bool gotFromAttrs = false;
    if (zattrs.contains("multiscales") && zattrs["multiscales"].is_array()
        && zattrs["multiscales"].size() > 0) {
        auto ms0 = zattrs["multiscales"][0];
        if (ms0.contains("datasets") && ms0["datasets"].is_array()) {
            for (const auto& ds : ms0["datasets"]) {
                if (ds.contains("path")) {
                    std::string p = ds["path"].get_string();
                    // Verify the array actually exists on disk
                    if (std::filesystem::exists(zarrRoot / p / ".zarray")) {
                        levelNames.push_back(std::move(p));
                    }
                }
            }
            gotFromAttrs = !levelNames.empty();
        }
    }

    // Fallback: scan subdirectories for .zarray (legacy / non-OME zarrs).
    if (!gotFromAttrs) {
        for (auto& entry : std::filesystem::directory_iterator(zarrRoot)) {
            if (!entry.is_directory()) continue;
            auto p = entry.path();
            if (std::filesystem::exists(p / ".zarray")) {
                levelNames.push_back(p.filename().string());
            }
        }
        // Sort numerically where possible, lexicographically otherwise.
        std::sort(levelNames.begin(), levelNames.end(),
                  [](const std::string& a, const std::string& b) {
                      int ia = 0, ib = 0;
                      bool aNum = false, bNum = false;
                      try { ia = std::stoi(a); aNum = true; } catch (...) {}
                      try { ib = std::stoi(b); bNum = true; } catch (...) {}
                      if (aNum && bNum) return ia < ib;
                      if (aNum != bNum) return aNum;
                      return a < b;
                  });
    }

    std::vector<std::unique_ptr<VcDataset>> result;
    result.reserve(levelNames.size());
    for (auto& name : levelNames) {
        result.push_back(std::make_unique<VcDataset>(zarrRoot / name));
    }
    return result;
}

utils::Json readZarrAttributes(const std::filesystem::path& groupPath)
{
    auto attrsPath = groupPath / ".zattrs";
    if (!std::filesystem::exists(attrsPath)) {
        return utils::Json::object();
    }
    return utils::Json::parse_file(attrsPath);
}

void writeZarrAttributes(const std::filesystem::path& groupPath,
                          const utils::Json& attrs)
{
    auto attrsPath = groupPath / ".zattrs";
    std::filesystem::create_directories(groupPath);
    std::ofstream f(attrsPath);
    f << attrs.dump(2) << '\n';
}

std::unique_ptr<VcDataset> createZarrDataset(
    const std::filesystem::path& parentPath,
    const std::string& name,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& chunks,
    VcDtype dtype,
    const std::string& compressor,
    const std::string& dimensionSeparator,
    std::int64_t fillValue)
{
    namespace fs = std::filesystem;
    fs::path dsPath = parentPath / name;

    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v2;
    meta.shape.assign(shape.begin(), shape.end());
    meta.chunks.assign(chunks.begin(), chunks.end());
    meta.dtype = (dtype == VcDtype::uint8) ? utils::ZarrDtype::uint8
                                           : utils::ZarrDtype::uint16;
    meta.fill_value = static_cast<double>(fillValue);
    meta.dimension_separator = dimensionSeparator;
    if (compressor == "blosc") {
        meta.compressor_id = "blosc";
        meta.compression_level = 3;
    } else if (compressor == "zstd") {
        meta.compressor_id = "zstd";
        meta.compression_level = 3;
    } else if (compressor.empty() || compressor == "none") {
        meta.compressor_id.clear();
    } else {
        meta.compressor_id = compressor;
    }

    // ZarrArray::create writes the .zarray file for us.
    utils::ZarrArray::create(dsPath, meta);

    auto zgroupPath = parentPath / ".zgroup";
    if (!fs::exists(zgroupPath)) {
        std::ofstream g(zgroupPath);
        g << R"({"zarr_format": 2})" << '\n';
    }

    return std::make_unique<VcDataset>(dsPath);
}

}  // namespace vc
