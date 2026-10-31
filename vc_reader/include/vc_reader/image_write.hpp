#pragma once
// Writing TIFF images; reading them is ImageFile and SourceReader.
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace vcr::sources {

struct TiffOptions {
    std::string compression = "none";  // none, lzw, deflate
    // Rows and columns of a tile, rounded down to a multiple of 16 and capped to the image;
    // an image under 16 pixels on a side is written in strips. None: one strip.
    std::optional<std::array<int, 2>> tile;
};

// `data` holds rows x cols x channels values of `dtype` (typestr: |u1, <u2, <i2, <u4, <i4,
// <f4, <f8), channels last, row-major. Written through a temporary beside `path`, then renamed.
void writeTiff(const std::filesystem::path& path, const void* data, const std::string& dtype, std::int64_t rows,
               std::int64_t cols, std::int64_t channels = 1, const TiffOptions& options = {});

}  // namespace vcr::sources
