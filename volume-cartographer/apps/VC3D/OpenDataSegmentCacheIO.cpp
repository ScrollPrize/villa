#include "OpenDataSegmentCacheIO.hpp"

#include "OpenDataManifest.hpp"

#include "vc/core/util/HttpFetch.hpp"

#include <tiffio.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace vc3d::opendata::detail {
namespace {

struct TiffInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint16_t bits = 0;
    uint16_t sampleFormat = SAMPLEFORMAT_UINT;
    uint16_t samplesPerPixel = 1;
    uint16_t compression = COMPRESSION_NONE;
    uint32_t rowsPerStrip = 0;
    tstrip_t strips = 0;
    bool tiled = false;
    bool byteSwapped = false;
};

TiffInfo readTiffInfo(const std::filesystem::path& path)
{
    TIFF* tif = TIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        throw std::runtime_error("failed to open TIFF: " + path.string());
    }
    TiffInfo info;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &info.width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &info.height);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &info.bits);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &info.sampleFormat);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &info.samplesPerPixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &info.compression);
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &info.rowsPerStrip);
    info.strips = TIFFNumberOfStrips(tif);
    info.tiled = TIFFIsTiled(tif);
    info.byteSwapped = TIFFIsByteSwapped(tif);
    TIFFClose(tif);
    return info;
}

bool isMmapCompatibleTiff(const TiffInfo& info)
{
    return !info.tiled && !info.byteSwapped && info.samplesPerPixel == 1 &&
           info.bits == 32 && info.sampleFormat == SAMPLEFORMAT_IEEEFP &&
           info.compression == COMPRESSION_NONE && info.strips == 1 &&
           info.rowsPerStrip >= info.height && info.width > 0 && info.height > 0;
}

float tiffSampleToFloat(const uint8_t* p, uint16_t sampleFormat, uint16_t bits)
{
    switch (sampleFormat) {
        case SAMPLEFORMAT_IEEEFP:
            if (bits == 32) {
                float value = 0.0F;
                std::memcpy(&value, p, sizeof(value));
                return value;
            }
            if (bits == 64) {
                double value = 0.0;
                std::memcpy(&value, p, sizeof(value));
                return static_cast<float>(value);
            }
            break;
        case SAMPLEFORMAT_UINT:
            if (bits == 8) return static_cast<float>(*p);
            if (bits == 16) {
                uint16_t value = 0;
                std::memcpy(&value, p, sizeof(value));
                return static_cast<float>(value);
            }
            if (bits == 32) {
                uint32_t value = 0;
                std::memcpy(&value, p, sizeof(value));
                return static_cast<float>(value);
            }
            break;
        case SAMPLEFORMAT_INT:
            if (bits == 8) {
                return static_cast<float>(*reinterpret_cast<const int8_t*>(p));
            }
            if (bits == 16) {
                int16_t value = 0;
                std::memcpy(&value, p, sizeof(value));
                return static_cast<float>(value);
            }
            if (bits == 32) {
                int32_t value = 0;
                std::memcpy(&value, p, sizeof(value));
                return static_cast<float>(value);
            }
            break;
        default:
            break;
    }
    throw std::runtime_error("unsupported TIFF sample type");
}

std::vector<float> readTiffAsFloat(const std::filesystem::path& path,
                                   TiffInfo* outInfo)
{
    TIFF* tif = TIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        throw std::runtime_error("failed to open TIFF: " + path.string());
    }

    const TiffInfo info = readTiffInfo(path);
    if (info.width == 0 || info.height == 0 || info.samplesPerPixel != 1) {
        TIFFClose(tif);
        throw std::runtime_error("unsupported TIFF geometry: " + path.string());
    }
    if (info.bits != 8 && info.bits != 16 && info.bits != 32 && info.bits != 64) {
        TIFFClose(tif);
        throw std::runtime_error("unsupported TIFF bits per sample: " + path.string());
    }

    const int bytesPerSample = (info.bits + 7) / 8;
    std::vector<float> pixels(static_cast<std::size_t>(info.width) * info.height);
    if (TIFFIsTiled(tif)) {
        uint32_t tileWidth = 0;
        uint32_t tileHeight = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
        if (tileWidth == 0 || tileHeight == 0) {
            TIFFClose(tif);
            throw std::runtime_error("invalid TIFF tile geometry: " + path.string());
        }
        const tmsize_t tileBytes = TIFFTileSize(tif);
        std::vector<uint8_t> tileBuffer(static_cast<std::size_t>(tileBytes));
        for (uint32_t y0 = 0; y0 < info.height; y0 += tileHeight) {
            const uint32_t height = std::min(tileHeight, info.height - y0);
            for (uint32_t x0 = 0; x0 < info.width; x0 += tileWidth) {
                const uint32_t width = std::min(tileWidth, info.width - x0);
                const ttile_t tile = TIFFComputeTile(tif, x0, y0, 0, 0);
                if (TIFFReadEncodedTile(
                        tif, tile, tileBuffer.data(), tileBytes) < 0) {
                    TIFFClose(tif);
                    throw std::runtime_error("failed reading tile: " + path.string());
                }
                for (uint32_t y = 0; y < height; ++y) {
                    const uint8_t* row =
                        tileBuffer.data() +
                        static_cast<std::size_t>(y) * tileWidth * bytesPerSample;
                    for (uint32_t x = 0; x < width; ++x) {
                        pixels[static_cast<std::size_t>(y0 + y) * info.width + x0 + x] =
                            tiffSampleToFloat(
                                row + static_cast<std::size_t>(x) * bytesPerSample,
                                info.sampleFormat, info.bits);
                    }
                }
            }
        }
    } else {
        const tmsize_t scanlineBytes = TIFFScanlineSize(tif);
        std::vector<uint8_t> scanline(static_cast<std::size_t>(scanlineBytes));
        for (uint32_t y = 0; y < info.height; ++y) {
            if (TIFFReadScanline(tif, scanline.data(), y, 0) != 1) {
                TIFFClose(tif);
                throw std::runtime_error("failed reading scanline: " + path.string());
            }
            for (uint32_t x = 0; x < info.width; ++x) {
                pixels[static_cast<std::size_t>(y) * info.width + x] =
                    tiffSampleToFloat(
                        scanline.data() + static_cast<std::size_t>(x) * bytesPerSample,
                        info.sampleFormat, info.bits);
            }
        }
    }

    TIFFClose(tif);
    if (outInfo) {
        *outInfo = info;
    }
    return pixels;
}

void writeMmapCompatibleTiff(const std::filesystem::path& path,
                             const TiffInfo& info,
                             const std::vector<float>& pixels)
{
    const auto temporaryPath = path.string() + ".mmap_tmp";
    const std::uint64_t pixelBytes =
        static_cast<std::uint64_t>(pixels.size()) * sizeof(float);
    TIFF* output = TIFFOpen(
        temporaryPath.c_str(), pixelBytes > 0xffff0000ULL ? "w8" : "w");
    if (!output) {
        throw std::runtime_error("failed to create TIFF: " + temporaryPath);
    }

    TIFFSetField(output, TIFFTAG_IMAGEWIDTH, info.width);
    TIFFSetField(output, TIFFTAG_IMAGELENGTH, info.height);
    TIFFSetField(output, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(output, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(output, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(output, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(output, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(output, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(output, TIFFTAG_ROWSPERSTRIP, info.height);

    const tsize_t bytes = static_cast<tsize_t>(pixelBytes);
    if (TIFFWriteEncodedStrip(
            output, 0, const_cast<float*>(pixels.data()), bytes) < 0) {
        TIFFClose(output);
        std::error_code error;
        std::filesystem::remove(temporaryPath, error);
        throw std::runtime_error("failed writing TIFF: " + temporaryPath);
    }
    TIFFClose(output);
    std::filesystem::rename(temporaryPath, path);
}

void normalizeTiffForMmap(const std::filesystem::path& path)
{
    const TiffInfo info = readTiffInfo(path);
    if (isMmapCompatibleTiff(info)) {
        return;
    }
    TiffInfo readInfo;
    auto pixels = readTiffAsFloat(path, &readInfo);
    writeMmapCompatibleTiff(path, readInfo, pixels);
}

} // namespace

bool isNonEmptyFile(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
           std::filesystem::file_size(path, error) > 0 && !error;
}

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void writeBytesAtomic(const std::filesystem::path& path,
                      const std::vector<std::byte>& bytes)
{
    std::filesystem::create_directories(path.parent_path());
    const auto temporaryPath = path.string() + ".tmp";
    {
        std::ofstream output(
            temporaryPath, std::ios::binary | std::ios::trunc);
        if (!bytes.empty()) {
            output.write(reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()));
        }
        if (!output) {
            throw std::runtime_error("failed to write " + temporaryPath);
        }
    }
    std::filesystem::rename(temporaryPath, path);
}

void writeStringAtomic(const std::filesystem::path& path,
                       const std::string& text)
{
    const auto bytes = std::vector<std::byte>(
        reinterpret_cast<const std::byte*>(text.data()),
        reinterpret_cast<const std::byte*>(text.data() + text.size()));
    writeBytesAtomic(path, bytes);
}

void writeCachedTifxyzBand(const std::string& baseUrl,
                           const std::string& fileName,
                           const std::filesystem::path& target)
{
    const auto url = joinOpenDataUrl(baseUrl, fileName);
    auto bytes = vc::httpGetBytes(url);
    if (bytes.empty()) {
        throw std::runtime_error("missing or empty " + fileName +
                                 " at " + url);
    }
    writeBytesAtomic(target, bytes);
    normalizeTiffForMmap(target);
}

bool cacheOptionalFile(const std::string& baseUrl,
                       const std::string& fileName,
                       const std::filesystem::path& target)
{
    try {
        auto bytes = vc::httpGetBytes(joinOpenDataUrl(baseUrl, fileName));
        if (bytes.empty()) return false;
        writeBytesAtomic(target, bytes);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace vc3d::opendata::detail
