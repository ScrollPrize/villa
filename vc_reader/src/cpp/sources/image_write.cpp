#include "vc_reader/image_write.hpp"

#include <tiffio.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace vcr::sources {
namespace {
struct Sample {
    std::uint16_t format;
    std::uint16_t bits;
};
Sample sampleOf(const std::string& dtype)
{
    if (dtype == "|u1") return {SAMPLEFORMAT_UINT, 8};
    if (dtype == "<u2") return {SAMPLEFORMAT_UINT, 16};
    if (dtype == "<i2") return {SAMPLEFORMAT_INT, 16};
    if (dtype == "<u4") return {SAMPLEFORMAT_UINT, 32};
    if (dtype == "<i4") return {SAMPLEFORMAT_INT, 32};
    if (dtype == "<f4") return {SAMPLEFORMAT_IEEEFP, 32};
    if (dtype == "<f8") return {SAMPLEFORMAT_IEEEFP, 64};
    throw std::invalid_argument("writeTiff does not support dtype " + dtype);
}
std::uint16_t compressionOf(const std::string& name)
{
    if (name == "none") return COMPRESSION_NONE;
    if (name == "lzw") return COMPRESSION_LZW;
    if (name == "deflate") return COMPRESSION_ADOBE_DEFLATE;
    throw std::invalid_argument("writeTiff compression must be none, lzw or deflate, got " + name);
}
void write(TIFF* tif, const std::uint8_t* data, std::int64_t rows, std::int64_t cols, std::size_t pixel, const TiffOptions& options)
{
    const auto failed = [] { throw std::runtime_error("writing a TIFF image failed"); };
    const std::size_t line = std::size_t(cols) * pixel;
    if (options.tile && rows >= 16 && cols >= 16) {
        const int th = std::max(16, (std::min<int>((*options.tile)[0], int(std::min<std::int64_t>(rows, 1 << 30))) / 16) * 16);
        const int tw = std::max(16, (std::min<int>((*options.tile)[1], int(std::min<std::int64_t>(cols, 1 << 30))) / 16) * 16);
        TIFFSetField(tif, TIFFTAG_TILELENGTH, std::uint32_t(th));
        TIFFSetField(tif, TIFFTAG_TILEWIDTH, std::uint32_t(tw));
        std::vector<std::uint8_t> tile(std::size_t(th) * std::size_t(tw) * pixel);
        for (std::int64_t r = 0; r < rows; r += th)
            for (std::int64_t c = 0; c < cols; c += tw) {
                std::fill(tile.begin(), tile.end(), std::uint8_t(0));
                const auto h = std::min<std::int64_t>(th, rows - r), w = std::min<std::int64_t>(tw, cols - c);
                for (std::int64_t y = 0; y < h; ++y)
                    std::copy_n(data + std::size_t(r + y) * line + std::size_t(c) * pixel, std::size_t(w) * pixel, tile.data() + std::size_t(y) * std::size_t(tw) * pixel);
                if (TIFFWriteEncodedTile(tif, TIFFComputeTile(tif, std::uint32_t(c), std::uint32_t(r), 0, 0), tile.data(), tmsize_t(tile.size())) < 0) failed();
            }
    } else {
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, std::uint32_t(rows));
        if (TIFFWriteEncodedStrip(tif, 0, const_cast<std::uint8_t*>(data), tmsize_t(std::size_t(rows) * line)) < 0) failed();
    }
}
}  // namespace

void writeTiff(const std::filesystem::path& path, const void* data, const std::string& dtype, std::int64_t rows, std::int64_t cols, std::int64_t channels, const TiffOptions& options)
{
    const auto sample = sampleOf(dtype);
    const auto compression = compressionOf(options.compression);
    if (rows <= 0 || cols <= 0 || channels <= 0 || rows >= (1LL << 31) || cols >= (1LL << 31) || channels > 4)
        throw std::invalid_argument("writeTiff needs positive rows and columns and up to 4 channels");
    const std::size_t pixel = std::size_t(channels) * sample.bits / 8;
    const auto bytes = std::size_t(rows) * std::size_t(cols) * pixel;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += "." + std::to_string(::getpid()) + ".tmp";
    {
        // Large output, or compressed output that may not shrink, needs 64-bit offsets.
        std::unique_ptr<TIFF, decltype(&TIFFClose)> tif(TIFFOpen(temporary.c_str(), bytes >= (1ULL << 31) ? "w8" : "w"), TIFFClose);
        if (!tif) throw std::runtime_error("cannot create " + temporary.string());
        TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, std::uint32_t(rows));
        TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, std::uint32_t(cols));
        TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, std::uint16_t(channels));
        TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, sample.bits);
        TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT, sample.format);
        TIFFSetField(tif.get(), TIFFTAG_COMPRESSION, compression);
        TIFFSetField(tif.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, channels >= 3 && sample.format == SAMPLEFORMAT_UINT ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
        if (channels == 2 || channels == 4) {
            const std::uint16_t extra[1]{EXTRASAMPLE_UNSPECIFIED};
            TIFFSetField(tif.get(), TIFFTAG_EXTRASAMPLES, std::uint16_t(1), extra);
        }
        try {
            write(tif.get(), static_cast<const std::uint8_t*>(data), rows, cols, pixel, options);
            if (!TIFFFlush(tif.get())) throw std::runtime_error("writing a TIFF image failed");
        } catch (...) {
            tif.reset();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw;
        }
    }
    std::filesystem::rename(temporary, path);
}

}  // namespace vcr::sources
