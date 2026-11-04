#pragma once
// Writing TIFF images; reading them is ImageFile, SourceReader and readImage.
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vcr::sources {

// Interleaved contiguous pixels; RGB(A) color order.
struct ImageView {
    const void* data;
    std::string dtype;
    std::int64_t rows, cols, channels = 1;
};
std::vector<std::uint8_t> encodePng(ImageView image);
std::vector<std::uint8_t> encodeJpeg(ImageView image, int quality = 95);
// PNG/JPEG use uint8 gray/RGB(A); TIFF retains the input type. Atomic publication.
void writeImage(const std::filesystem::path& path, ImageView image, int jpegQuality = 95);

struct TiffOptions {
    std::string compression = "none";  // none, lzw, deflate, packbits, zstd
    // Rows and columns of a tile, rounded down to a multiple of 16; a whole page's are capped to the image.
    // None: strips of about 256 KiB.
    std::optional<std::array<int, 2>> tile;
    std::int64_t pages = 1;
    std::optional<double> pixelUm;  // written as the resolution tags
};

// `dtype` is a typestr: |u1, |i1, <u2, <i2, <u4, <i4, <u8, <i8, <f4, <f8. Samples are row-major with
// channels last. Written through a temporary beside `path`, renamed by finish(); a writer
// destroyed before finish() removes it.
class TiffWriter {
public:
    TiffWriter(std::filesystem::path path, const std::string& dtype, std::int64_t rows, std::int64_t cols, std::int64_t channels = 1, TiffOptions options = {});
    ~TiffWriter();
    TiffWriter(const TiffWriter&) = delete;
    TiffWriter& operator=(const TiffWriter&) = delete;

    // The tile writeTile takes, whether or not the options asked for one.
    [[nodiscard]] std::array<int, 2> tile() const;
    [[nodiscard]] const std::string& dtype() const;
    [[nodiscard]] std::array<std::int64_t, 3> shape() const;  // rows, columns, channels
    // The next page, whole: tiled when the options ask for it and the image reaches 16 pixels on both sides.
    // `name` is its PageName tag, the layer name image editors show.
    void writePage(const void* data, const std::string& name = {});
    // The next page, all zeros, in strips even if the options ask for tiles.
    void writeBlankPage();
    // One tile of a single-page image, at a tile-aligned position, in any order. A tile at the
    // image's end may hold fewer rows and columns; a tile never written reads as zeros.
    void writeTile(std::int64_t row, std::int64_t col, const void* data, std::int64_t rows, std::int64_t cols);
    void finish();
    void abandon();

private:
    struct State;
    std::unique_ptr<State> state_;
};

// One image, or with `options.pages` that many, of rows x cols x channels values at `data`.
void writeTiff(const std::filesystem::path& path, const void* data, const std::string& dtype, std::int64_t rows,
               std::int64_t cols, std::int64_t channels = 1, const TiffOptions& options = {});

inline void writeTiff(const std::filesystem::path& path, ImageView image, const TiffOptions& options = {}) {
    writeTiff(path,image.data,image.dtype,image.rows,image.cols,image.channels,options);
}
}  // namespace vcr::sources
