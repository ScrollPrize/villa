#include "vc/core/types/VcDataset.hpp"

#include <utils/zarr.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
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

static CompressorConfig parseCompressor(const nlohmann::json& zarray, int dtypeSize)
{
    CompressorConfig cfg;
    cfg.blosc_typesize = dtypeSize;

    if (!zarray.contains("compressor") || zarray["compressor"].is_null()) {
        cfg.id = CompressorId::None;
        return cfg;
    }

    const auto& comp = zarray["compressor"];
    std::string id = comp.value("id", "");

    if (id == "blosc") {
        cfg.id = CompressorId::Blosc;
        cfg.blosc_cname = comp.value("cname", "lz4");
        cfg.blosc_clevel = comp.value("clevel", 5);
        cfg.blosc_shuffle = comp.value("shuffle", 1);
        if (comp.contains("blocksize"))
            cfg.blosc_blocksize = comp["blocksize"].get<int>();
        if (comp.contains("typesize"))
            cfg.blosc_typesize = comp["typesize"].get<int>();
        else
            cfg.blosc_typesize = dtypeSize;
    } else if (id == "zstd") {
        cfg.id = CompressorId::Zstd;
        cfg.level = comp.value("level", 3);
    } else if (id == "lz4") {
        cfg.id = CompressorId::Lz4;
        cfg.level = comp.value("acceleration", 1);
    } else if (id == "gzip" || id == "zlib") {
        cfg.id = CompressorId::Gzip;
        cfg.level = comp.value("level", 5);
    } else {
        throw std::runtime_error("Unsupported zarr compressor: " + id);
    }

    return cfg;
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

    // utils zarr array for chunk I/O
    std::shared_ptr<utils::FileSystemStore> store_;
    std::unique_ptr<utils::ZarrArray> zarrArray_;

    void parseZarray(const std::filesystem::path& path)
    {
        auto zarrayPath = path / ".zarray";
        if (!std::filesystem::exists(zarrayPath)) {
            throw std::runtime_error("Missing .zarray in " + path.string());
        }

        std::ifstream f(zarrayPath);
        nlohmann::json zarray = nlohmann::json::parse(f);

        // Shape
        auto shapeJson = zarray["shape"];
        shape_.clear();
        for (auto& v : shapeJson)
            shape_.push_back(v.get<size_t>());

        // Chunks
        auto chunksJson = zarray["chunks"];
        chunkShape_.clear();
        for (auto& v : chunksJson)
            chunkShape_.push_back(v.get<size_t>());

        // Chunk size (product)
        chunkSize_ = 1;
        for (auto c : chunkShape_)
            chunkSize_ *= c;

        // Dtype
        std::string dtypeStr = zarray["dtype"].get<std::string>();
        if (dtypeStr == "|u1" || dtypeStr == "<u1" || dtypeStr == "uint8") {
            dtype_ = VcDtype::uint8;
            dtypeSize_ = 1;
        } else if (dtypeStr == "<u2" || dtypeStr == "|u2" || dtypeStr == "uint16") {
            dtype_ = VcDtype::uint16;
            dtypeSize_ = 2;
        } else {
            throw std::runtime_error("Unsupported zarr dtype: " + dtypeStr);
        }

        // Dimension separator
        if (zarray.contains("dimension_separator")) {
            delimiter_ = zarray["dimension_separator"].get<std::string>();
        }

        // Compressor
        compressor_ = parseCompressor(zarray, static_cast<int>(dtypeSize_));
    }

    void openZarrArray()
    {
        // Open the zarr array directly from its path using utils
        zarrArray_ = std::make_unique<utils::ZarrArray>(
            utils::ZarrArray::open(fsPath));
    }
};

// ============================================================================
// VcDataset public API
// ============================================================================

VcDataset::VcDataset(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>())
{
    impl_->fsPath = path;
    impl_->parseZarray(path);
    impl_->openZarrArray();
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
    const auto* src = reinterpret_cast<const char*>(compressed.data());

    switch (impl_->compressor_.id) {
        case CompressorId::None:
            std::memcpy(output, compressed.data(), outBytes);
            break;

        case CompressorId::Blosc: {
            int ret = blosc_decompress(src, output, outBytes);
            if (ret < 0) {
                throw std::runtime_error("blosc_decompress failed with code " +
                                          std::to_string(ret));
            }
            break;
        }

        case CompressorId::Zstd: {
            size_t ret = ZSTD_decompress(output, outBytes,
                                          compressed.data(), compressed.size());
            if (ZSTD_isError(ret)) {
                throw std::runtime_error(
                    std::string("ZSTD_decompress failed: ") + ZSTD_getErrorName(ret));
            }
            break;
        }

        case CompressorId::Lz4: {
            // numcodecs lz4 format: 4-byte LE original size prefix, then lz4 block
            if (compressed.size() < 4) {
                throw std::runtime_error("LZ4 compressed data too short");
            }
            uint32_t origSize;
            std::memcpy(&origSize, compressed.data(), 4);
            if (origSize > outBytes) {
                throw std::runtime_error(
                    "LZ4 origSize (" + std::to_string(origSize) +
                    ") exceeds output buffer (" + std::to_string(outBytes) + ")");
            }
            int ret = LZ4_decompress_safe(
                src + 4,
                static_cast<char*>(output),
                static_cast<int>(compressed.size() - 4),
                static_cast<int>(origSize));
            if (ret < 0) {
                throw std::runtime_error("LZ4_decompress_safe failed");
            }
            break;
        }

        case CompressorId::Gzip: {
            z_stream strm{};
            if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
                throw std::runtime_error("inflateInit2 failed");
            }
            strm.avail_in = static_cast<uInt>(compressed.size());
            strm.next_in = const_cast<Bytef*>(
                reinterpret_cast<const Bytef*>(compressed.data()));
            strm.avail_out = static_cast<uInt>(outBytes);
            strm.next_out = static_cast<Bytef*>(output);
            int ret = inflate(&strm, Z_FINISH);
            inflateEnd(&strm);
            if (ret != Z_STREAM_END && ret != Z_OK) {
                throw std::runtime_error("gzip inflate failed with code " +
                                          std::to_string(ret));
            }
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

bool VcDataset::writeChunk(size_t iz, size_t iy, size_t ix,
                            const void* input, size_t nbytes)
{
    std::array<size_t, 3> indices = {iz, iy, ix};
    auto data = std::span<const std::byte>(
        static_cast<const std::byte*>(input), nbytes);
    impl_->zarrArray_->write_chunk(indices, data);
    return true;
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
                        std::memset(outBytes + dstOff, 0, copySize[d] * elemSize);
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

    for (auto& entry : std::filesystem::directory_iterator(zarrRoot)) {
        if (!entry.is_directory()) continue;
        auto p = entry.path();
        if (std::filesystem::exists(p / ".zarray")) {
            levelNames.push_back(p.filename().string());
        }
    }

    // Sort numerically where possible, lexicographically otherwise.
    // Numeric names sort before non-numeric names.
    std::sort(levelNames.begin(), levelNames.end(),
              [](const std::string& a, const std::string& b) {
                  int ia = 0, ib = 0;
                  bool aNum = false, bNum = false;
                  try { ia = std::stoi(a); aNum = true; } catch (...) {}
                  try { ib = std::stoi(b); bNum = true; } catch (...) {}
                  if (aNum && bNum) return ia < ib;
                  if (aNum != bNum) return aNum;  // numeric before non-numeric
                  return a < b;
              });

    std::vector<std::unique_ptr<VcDataset>> result;
    result.reserve(levelNames.size());
    for (auto& name : levelNames) {
        result.push_back(std::make_unique<VcDataset>(zarrRoot / name));
    }
    return result;
}

nlohmann::json readZarrAttributes(const std::filesystem::path& groupPath)
{
    auto attrsPath = groupPath / ".zattrs";
    if (!std::filesystem::exists(attrsPath)) {
        return nlohmann::json::object();
    }
    std::ifstream f(attrsPath);
    return nlohmann::json::parse(f);
}

void writeZarrAttributes(const std::filesystem::path& groupPath,
                          const nlohmann::json& attrs)
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
    const std::string& dimensionSeparator)
{
    namespace fs = std::filesystem;

    // Create the directory structure
    fs::path dsPath = parentPath / name;
    fs::create_directories(dsPath);

    // Write .zarray metadata
    nlohmann::json zarray;
    zarray["zarr_format"] = 2;
    zarray["shape"] = shape;
    zarray["chunks"] = chunks;
    zarray["dtype"] = (dtype == VcDtype::uint8) ? "|u1" : "<u2";
    zarray["fill_value"] = 0;
    zarray["order"] = "C";
    zarray["dimension_separator"] = dimensionSeparator;

    if (compressor == "blosc") {
        zarray["compressor"] = {
            {"id", "blosc"},
            {"cname", "lz4"},
            {"clevel", 5},
            {"shuffle", 1},
            {"blocksize", 0}
        };
    } else if (compressor == "zstd") {
        zarray["compressor"] = {
            {"id", "zstd"},
            {"level", 3}
        };
    } else if (compressor.empty() || compressor == "none") {
        zarray["compressor"] = nullptr;
    } else {
        zarray["compressor"] = {{"id", compressor}};
    }
    zarray["filters"] = nullptr;

    std::ofstream f(dsPath / ".zarray");
    f << zarray.dump(2) << '\n';
    f.close();

    // Also write .zgroup in parent if it doesn't exist
    auto zgroupPath = parentPath / ".zgroup";
    if (!fs::exists(zgroupPath)) {
        std::ofstream g(zgroupPath);
        g << R"({"zarr_format": 2})" << '\n';
    }

    return std::make_unique<VcDataset>(dsPath);
}

}  // namespace vc
