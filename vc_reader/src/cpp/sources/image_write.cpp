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
    if (dtype == "|i1") return {SAMPLEFORMAT_INT, 8};
    if (dtype == "<u2") return {SAMPLEFORMAT_UINT, 16};
    if (dtype == "<i2") return {SAMPLEFORMAT_INT, 16};
    if (dtype == "<u4") return {SAMPLEFORMAT_UINT, 32};
    if (dtype == "<i4") return {SAMPLEFORMAT_INT, 32};
    if (dtype == "<u8") return {SAMPLEFORMAT_UINT, 64};
    if (dtype == "<i8") return {SAMPLEFORMAT_INT, 64};
    if (dtype == "<f4") return {SAMPLEFORMAT_IEEEFP, 32};
    if (dtype == "<f8") return {SAMPLEFORMAT_IEEEFP, 64};
    throw std::invalid_argument("TiffWriter does not support dtype " + dtype);
}
std::uint16_t compressionOf(const std::string& name)
{
    const std::uint16_t scheme = name == "none" ? COMPRESSION_NONE : name == "lzw" ? COMPRESSION_LZW : name == "deflate" ? COMPRESSION_ADOBE_DEFLATE : name == "packbits" ? COMPRESSION_PACKBITS : name == "zstd" ? COMPRESSION_ZSTD : 0;
    if (!scheme) throw std::invalid_argument("TiffWriter compression must be none, lzw, deflate, packbits or zstd, got " + name);
    if (!TIFFIsCODECConfigured(scheme)) throw std::runtime_error("this libtiff cannot write " + name);
    return scheme;
}
int tileSide(int requested, std::int64_t side)
{
    return std::max(16, (std::min<int>(requested, int(std::min<std::int64_t>(side, 1 << 30))) / 16) * 16);
}
constexpr std::size_t kStripBytes = 256 * 1024;
}  // namespace

struct TiffWriter::State {
    std::filesystem::path path, temporary;
    std::string dtype;
    Sample sample{};
    std::uint16_t compression = COMPRESSION_NONE;
    std::int64_t rows = 0, cols = 0, channels = 1;
    TiffOptions options;
    std::size_t pixel = 0;
    std::array<int, 2> tile{};
    std::int64_t pagesWritten = 0;
    bool tiling = false;
    std::vector<bool> written;  // tiles of the single page
    std::unique_ptr<TIFF, decltype(&TIFFClose)> tif{nullptr, TIFFClose};

    void tags() const
    {
        TIFF* t = tif.get();
        TIFFSetField(t, TIFFTAG_IMAGELENGTH, std::uint32_t(rows));
        TIFFSetField(t, TIFFTAG_IMAGEWIDTH, std::uint32_t(cols));
        TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, std::uint16_t(channels));
        TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, sample.bits);
        TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, sample.format);
        TIFFSetField(t, TIFFTAG_COMPRESSION, compression);
        TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(t, TIFFTAG_PHOTOMETRIC, channels >= 3 && sample.format == SAMPLEFORMAT_UINT ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
        if (channels == 2 || channels == 4) {
            const std::uint16_t extra[1]{EXTRASAMPLE_UNSPECIFIED};
            TIFFSetField(t, TIFFTAG_EXTRASAMPLES, std::uint16_t(1), extra);
        }
    }
    void tileTags() const
    {
        TIFFSetField(tif.get(), TIFFTAG_TILELENGTH, std::uint32_t(tile[0]));
        TIFFSetField(tif.get(), TIFFTAG_TILEWIDTH, std::uint32_t(tile[1]));
    }
    void encodeTile(std::int64_t row, std::int64_t col, const std::uint8_t* tiled)
    {
        if (TIFFWriteEncodedTile(tif.get(), TIFFComputeTile(tif.get(), std::uint32_t(col), std::uint32_t(row), 0, 0), const_cast<std::uint8_t*>(tiled), tmsize_t(std::size_t(tile[0]) * std::size_t(tile[1]) * pixel)) < 0)
            throw std::runtime_error("writing a TIFF image failed");
    }
    void nextPage()
    {
        if (!tif) throw std::logic_error("TiffWriter is finished");
        if (tiling) throw std::logic_error("TiffWriter cannot mix pages and tiles");
        if (pagesWritten >= options.pages) throw std::logic_error("TiffWriter has written all its pages");
        if (pagesWritten) {
            if (!TIFFWriteDirectory(tif.get())) throw std::runtime_error("writing a TIFF image failed");
            tags();
        }
    }
    // `data` null: zeros.
    void strips(const std::uint8_t* data) const
    {
        const std::size_t line = std::size_t(cols) * pixel;
        const auto perStrip = std::clamp<std::int64_t>(std::int64_t(kStripBytes / line), 1, rows);
        TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, std::uint32_t(perStrip));
        std::vector<std::uint8_t> zeros(data ? 0 : std::size_t(perStrip) * line);
        for (std::int64_t r = 0; r < rows; r += perStrip) {
            const auto height = std::min(perStrip, rows - r);
            auto* from = data ? const_cast<std::uint8_t*>(data) + std::size_t(r) * line : zeros.data();
            if (TIFFWriteEncodedStrip(tif.get(), std::uint32_t(r / perStrip), from, tmsize_t(std::size_t(height) * line)) < 0)
                throw std::runtime_error("writing a TIFF image failed");
        }
    }
    void copyTile(std::vector<std::uint8_t>& out, const std::uint8_t* data, std::int64_t lines, std::int64_t width, std::size_t stride) const
    {
        std::fill(out.begin(), out.end(), std::uint8_t(0));
        for (std::int64_t y = 0; y < lines; ++y)
            std::copy_n(data + std::size_t(y) * stride, std::size_t(width) * pixel, out.data() + std::size_t(y) * std::size_t(tile[1]) * pixel);
    }
};

TiffWriter::TiffWriter(std::filesystem::path path, const std::string& dtype, std::int64_t rows, std::int64_t cols, std::int64_t channels, TiffOptions options)
    : state_(std::make_unique<State>())
{
    auto& s = *state_;
    s.sample = sampleOf(dtype);
    s.dtype = dtype;
    s.compression = compressionOf(options.compression);
    if (rows <= 0 || cols <= 0 || channels <= 0 || rows >= (1LL << 31) || cols >= (1LL << 31) || channels > 4)
        throw std::invalid_argument("TiffWriter needs positive rows and columns and up to 4 channels");
    if (options.pages <= 0) throw std::invalid_argument("TiffWriter needs at least one page");
    s.rows = rows;
    s.cols = cols;
    s.channels = channels;
    s.pixel = std::size_t(channels) * s.sample.bits / 8;
    const auto requested = options.tile.value_or(std::array<int, 2>{256, 256});
    s.tile = {tileSide(requested[0], 1 << 30), tileSide(requested[1], 1 << 30)};
    s.options = std::move(options);
    s.path = std::move(path);
    if (s.path.has_parent_path()) std::filesystem::create_directories(s.path.parent_path());
    s.temporary = s.path;
    s.temporary += "." + std::to_string(::getpid()) + ".tmp";
    const auto bytes = std::size_t(s.options.pages) * std::size_t(rows) * std::size_t(cols) * s.pixel;
    // Large output, or compressed output that may not shrink, needs 64-bit offsets.
    s.tif.reset(TIFFOpen(s.temporary.c_str(), bytes >= (1ULL << 31) ? "w8" : "w"));
    if (!s.tif) throw std::runtime_error("cannot create " + s.temporary.string());
    s.tags();
}

TiffWriter::~TiffWriter()
{
    abandon();
}

std::array<int, 2> TiffWriter::tile() const
{
    return state_->tile;
}

const std::string& TiffWriter::dtype() const
{
    return state_->dtype;
}

std::array<std::int64_t, 3> TiffWriter::shape() const
{
    return {state_->rows, state_->cols, state_->channels};
}

void TiffWriter::writePage(const void* data)
{
    auto& s = *state_;
    s.nextPage();
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    const std::size_t line = std::size_t(s.cols) * s.pixel;
    if (s.options.tile && s.rows >= 16 && s.cols >= 16) {
        s.tile = {tileSide(s.tile[0], s.rows), tileSide(s.tile[1], s.cols)};
        s.tileTags();
        std::vector<std::uint8_t> tile(std::size_t(s.tile[0]) * std::size_t(s.tile[1]) * s.pixel);
        for (std::int64_t r = 0; r < s.rows; r += s.tile[0])
            for (std::int64_t c = 0; c < s.cols; c += s.tile[1]) {
                s.copyTile(tile, bytes + std::size_t(r) * line + std::size_t(c) * s.pixel, std::min<std::int64_t>(s.tile[0], s.rows - r), std::min<std::int64_t>(s.tile[1], s.cols - c), line);
                s.encodeTile(r, c, tile.data());
            }
    } else {
        s.strips(bytes);
    }
    ++s.pagesWritten;
}

void TiffWriter::writeBlankPage()
{
    auto& s = *state_;
    s.nextPage();
    s.strips(nullptr);
    ++s.pagesWritten;
}

void TiffWriter::writeTile(std::int64_t row, std::int64_t col, const void* data, std::int64_t rows, std::int64_t cols)
{
    auto& s = *state_;
    if (!s.tif) throw std::logic_error("TiffWriter is finished");
    if (s.options.pages != 1 || s.pagesWritten) throw std::logic_error("TiffWriter writes tiles of a single page only");
    if (row < 0 || col < 0 || row >= s.rows || col >= s.cols || row % s.tile[0] || col % s.tile[1])
        throw std::invalid_argument("TiffWriter tile position is outside the image or not aligned to the tile grid");
    if (rows <= 0 || cols <= 0 || rows > s.tile[0] || cols > s.tile[1]) throw std::invalid_argument("TiffWriter tile is larger than the tile size");
    if (!s.tiling) {
        s.tileTags();
        s.tiling = true;
        s.written.assign(std::size_t(TIFFNumberOfTiles(s.tif.get())), false);
    }
    const auto index = std::size_t(row / s.tile[0]) * std::size_t((s.cols + s.tile[1] - 1) / s.tile[1]) + std::size_t(col / s.tile[1]);
    if (s.written[index]) throw std::logic_error("TiffWriter tile written twice");
    if (rows == s.tile[0] && cols == s.tile[1]) {
        s.encodeTile(row, col, static_cast<const std::uint8_t*>(data));
    } else {
        std::vector<std::uint8_t> tile(std::size_t(s.tile[0]) * std::size_t(s.tile[1]) * s.pixel);
        s.copyTile(tile, static_cast<const std::uint8_t*>(data), rows, cols, std::size_t(cols) * s.pixel);
        s.encodeTile(row, col, tile.data());
    }
    s.written[index] = true;
}

void TiffWriter::finish()
{
    auto& s = *state_;
    if (!s.tif) throw std::logic_error("TiffWriter is finished");
    if (s.tiling) {
        const std::vector<std::uint8_t> zeros(std::size_t(s.tile[0]) * std::size_t(s.tile[1]) * s.pixel);
        const auto across = std::size_t((s.cols + s.tile[1] - 1) / s.tile[1]);
        for (std::size_t i = 0; i < s.written.size(); ++i)
            if (!s.written[i]) s.encodeTile(std::int64_t(i / across) * s.tile[0], std::int64_t(i % across) * s.tile[1], zeros.data());
    } else if (s.pagesWritten != s.options.pages) {
        throw std::logic_error("TiffWriter finished with " + std::to_string(s.pagesWritten) + " of " + std::to_string(s.options.pages) + " pages");
    }
    if (!TIFFFlush(s.tif.get())) throw std::runtime_error("writing a TIFF image failed");
    s.tif.reset();
    std::filesystem::rename(s.temporary, s.path);
}

void TiffWriter::abandon()
{
    auto& s = *state_;
    if (!s.tif) return;
    s.tif.reset();
    std::error_code ignored;
    std::filesystem::remove(s.temporary, ignored);
}

void writeTiff(const std::filesystem::path& path, const void* data, const std::string& dtype, std::int64_t rows, std::int64_t cols, std::int64_t channels, const TiffOptions& options)
{
    TiffWriter writer(path, dtype, rows, cols, channels, options);
    const std::size_t page = std::size_t(rows) * std::size_t(cols) * std::size_t(channels) * sampleOf(dtype).bits / 8;
    for (std::int64_t i = 0; i < options.pages; ++i) writer.writePage(static_cast<const std::uint8_t*>(data) + std::size_t(i) * page);
    writer.finish();
}

}  // namespace vcr::sources
