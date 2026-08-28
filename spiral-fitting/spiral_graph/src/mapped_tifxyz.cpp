#include <spiral_graph/surface_index.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <tiffio.h>

namespace surfcore {
namespace {

class FileMapping {
public:
    explicit FileMapping(const std::filesystem::path& path)
    {
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "cannot open " + path.string());
        }
        struct stat status {};
        if (::fstat(descriptor, &status) != 0 || status.st_size <= 0) {
            const int error = errno;
            ::close(descriptor);
            throw std::system_error(error, std::generic_category(),
                                    "cannot stat " + path.string());
        }
        size_ = static_cast<size_t>(status.st_size);
        data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
        const int error = errno;
        ::close(descriptor);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            throw std::system_error(error, std::generic_category(),
                                    "cannot map " + path.string());
        }
    }

    ~FileMapping()
    {
        if (data_) ::munmap(data_, size_);
    }

    FileMapping(const FileMapping&) = delete;
    FileMapping& operator=(const FileMapping&) = delete;

    const std::uint8_t* bytes(std::uint64_t offset, size_t count) const
    {
        if (offset > size_ || count > size_ - static_cast<size_t>(offset)) {
            throw std::runtime_error("TIFF strip points outside mapped file");
        }
        return static_cast<const std::uint8_t*>(data_) + offset;
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

class MappedFloatBand {
public:
    explicit MappedFloatBand(const std::filesystem::path& path)
        : mapping_(path)
    {
        TIFF* tif = TIFFOpen(path.c_str(), "r");
        if (!tif) throw std::runtime_error("cannot open TIFF " + path.string());

        std::uint32_t width = 0, height = 0, rows_per_strip = 0;
        std::uint16_t samples = 1, bits = 0, format = SAMPLEFORMAT_UINT;
        std::uint16_t compression = COMPRESSION_NONE;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
        TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
        TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &format);
        TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &compression);
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
        const tstrip_t strip_count = TIFFNumberOfStrips(tif);

        if (!width || !height || TIFFIsTiled(tif) || TIFFIsByteSwapped(tif)
            || samples != 1 || bits != 32 || format != SAMPLEFORMAT_IEEEFP
            || compression != COMPRESSION_NONE || rows_per_strip == 0
            || strip_count <= 0) {
            std::ostringstream message;
            message << "TIFF is not directly mappable: " << path;
            TIFFClose(tif);
            throw std::runtime_error(message.str());
        }

        std::uint64_t* offsets = nullptr;
        std::uint64_t* byte_counts = nullptr;
        if (!TIFFGetField(tif, TIFFTAG_STRIPOFFSETS, &offsets)
            || !TIFFGetField(tif, TIFFTAG_STRIPBYTECOUNTS, &byte_counts)
            || !offsets || !byte_counts) {
            TIFFClose(tif);
            throw std::runtime_error("TIFF has no strip offsets: " + path.string());
        }
        offsets_.resize(static_cast<size_t>(strip_count));
        for (tstrip_t strip = 0; strip < strip_count; ++strip) {
            const std::uint32_t first_row = strip * rows_per_strip;
            const std::uint32_t rows = first_row < height
                ? std::min(rows_per_strip, height - first_row) : 0;
            const std::uint64_t expected = static_cast<std::uint64_t>(width)
                * rows * sizeof(float);
            if (!rows || byte_counts[strip] < expected) {
                TIFFClose(tif);
                throw std::runtime_error("TIFF strip is truncated: " + path.string());
            }
            offsets_[static_cast<size_t>(strip)] = offsets[strip];
        }
        TIFFClose(tif);
        width_ = width;
        height_ = height;
        rows_per_strip_ = rows_per_strip;
    }

    size_t rows() const noexcept { return height_; }
    size_t cols() const noexcept { return width_; }

    float at(size_t row, size_t col) const
    {
        const size_t strip = row / rows_per_strip_;
        const size_t strip_row = row - strip * rows_per_strip_;
        const std::uint64_t offset = offsets_[strip]
            + (static_cast<std::uint64_t>(strip_row) * width_ + col) * sizeof(float);
        float value = 0.0f;
        std::memcpy(&value, mapping_.bytes(offset, sizeof(value)), sizeof(value));
        return value;
    }

private:
    FileMapping mapping_;
    std::vector<std::uint64_t> offsets_;
    size_t width_ = 0;
    size_t height_ = 0;
    size_t rows_per_strip_ = 0;
};

class MappedTifxyzPointSource final : public SurfacePointSource {
public:
    MappedTifxyzPointSource(
        const std::filesystem::path& directory, size_t rows, size_t cols)
        : x_(directory / "x.tif"), y_(directory / "y.tif"),
          z_(directory / "z.tif")
    {
        if (x_.rows() != rows || y_.rows() != rows || z_.rows() != rows
            || x_.cols() != cols || y_.cols() != cols || z_.cols() != cols) {
            throw std::runtime_error(
                "mapped tifxyz dimensions differ from patch metadata: "
                + directory.string());
        }
    }

    Vec3 at(size_t row, size_t col) const override
    {
        return {x_.at(row, col), y_.at(row, col), z_.at(row, col)};
    }

private:
    MappedFloatBand x_;
    MappedFloatBand y_;
    MappedFloatBand z_;
};

} // namespace

std::shared_ptr<const SurfacePointSource> open_mapped_tifxyz_point_source(
    const std::filesystem::path& directory, size_t rows, size_t cols)
{
    return std::make_shared<MappedTifxyzPointSource>(directory, rows, cols);
}

std::vector<std::uint8_t> read_tifxyz_mask(
    const std::filesystem::path& path, size_t rows, size_t cols)
{
    TIFF* raw = TIFFOpen(path.c_str(), "r");
    if (!raw) throw std::runtime_error("cannot open mask TIFF " + path.string());
    const auto close = [](TIFF* value) { TIFFClose(value); };
    std::unique_ptr<TIFF, decltype(close)> tif(raw, close);
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t samples = 1;
    std::uint16_t bits = 0;
    std::uint16_t planar = PLANARCONFIG_CONTIG;
    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &height);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
    if (width != cols || height != rows || samples < 1
        || planar != PLANARCONFIG_CONTIG
        || (bits != 8 && bits != 16 && bits != 32)) {
        throw std::runtime_error(
            "mask TIFF dimensions or type do not match coordinates: "
            + path.string());
    }
    const std::size_t bytes_per_value = bits / 8;
    const std::size_t bytes_per_pixel = bytes_per_value * samples;
    std::vector<std::uint8_t> result(rows * cols, 0);
    const auto nonzero = [bytes_per_value](const std::uint8_t* pointer) {
        std::uint32_t value = 0;
        std::memcpy(&value, pointer, bytes_per_value);
        return value != 0;
    };
    if (!TIFFIsTiled(tif.get())) {
        std::vector<std::uint8_t> line(
            static_cast<std::size_t>(TIFFScanlineSize(tif.get())));
        for (std::size_t row = 0; row < rows; ++row) {
            if (TIFFReadScanline(tif.get(), line.data(), row) < 0) {
                throw std::runtime_error("failed reading mask TIFF " + path.string());
            }
            for (std::size_t column = 0; column < cols; ++column) {
                result[row * cols + column] = nonzero(
                    line.data() + column * bytes_per_pixel) ? 1 : 0;
            }
        }
        return result;
    }
    std::uint32_t tile_width = 0;
    std::uint32_t tile_height = 0;
    TIFFGetField(tif.get(), TIFFTAG_TILEWIDTH, &tile_width);
    TIFFGetField(tif.get(), TIFFTAG_TILELENGTH, &tile_height);
    if (tile_width == 0 || tile_height == 0) {
        throw std::runtime_error("mask TIFF has invalid tile dimensions");
    }
    std::vector<std::uint8_t> tile(
        static_cast<std::size_t>(TIFFTileSize(tif.get())));
    for (std::size_t row = 0; row < rows; row += tile_height) {
        for (std::size_t column = 0; column < cols; column += tile_width) {
            const ttile_t index = TIFFComputeTile(
                tif.get(), static_cast<std::uint32_t>(column),
                static_cast<std::uint32_t>(row), 0, 0);
            if (TIFFReadEncodedTile(
                    tif.get(), index, tile.data(),
                    static_cast<tmsize_t>(tile.size())) < 0) {
                throw std::runtime_error("failed reading tiled mask TIFF " + path.string());
            }
            const std::size_t copy_rows = std::min<std::size_t>(tile_height, rows - row);
            const std::size_t copy_cols = std::min<std::size_t>(tile_width, cols - column);
            for (std::size_t local_row = 0; local_row < copy_rows; ++local_row) {
                for (std::size_t local_column = 0;
                     local_column < copy_cols; ++local_column) {
                    const std::size_t tile_offset
                        = (local_row * tile_width + local_column) * bytes_per_pixel;
                    result[(row + local_row) * cols + column + local_column]
                        = nonzero(tile.data() + tile_offset) ? 1 : 0;
                }
            }
        }
    }
    return result;
}

} // namespace surfcore
