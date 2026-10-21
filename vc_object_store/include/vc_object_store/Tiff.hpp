#pragma once

#include <opencv2/core/mat.hpp>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <tiffio.h>

// Write single-channel image (8U, 16U, 32F) as TIFF.
// cvType: output type (-1 = same as input). If different, values are scaled:
//         8U↔16U: scale by 257, 8U↔32F: scale by 1/255, 16U↔32F: scale by 1/65535
// tileW/tileH: tile dimensions. If either is 0, write one untiled strip.
// compression: libtiff compression constant (e.g. COMPRESSION_LZW, COMPRESSION_PACKBITS)
// padValue: value for padding partial tiles (default -1.0f, used for float; int types use 0)
// dpi: resolution in dots per inch (0 = don't set).
void writeTiff(
    const std::filesystem::path& outPath,
    const cv::Mat& img,
    int cvType = -1,
    uint32_t tileW = 1024,
    uint32_t tileH = 1024,
    float padValue = -1.0f,
    uint16_t compression = COMPRESSION_LZW,
    float dpi = 0.f);

// Decode the first directory completely to check for truncated or corrupt data.
// Reads the entire payload and temporarily suppresses libtiff diagnostics.
bool isReadableTiff(const std::filesystem::path& path);

// Native sample bytes in host byte order, interleaved in TIFF component order.
// No color conversion, scaling, mask normalization or float conversion.
struct TiffSamples {
    uint32_t width = 0, height = 0;
    uint16_t bits = 0, sampleFormat = 0, components = 0;
    std::vector<uint8_t> bytes;
};
TiffSamples readTiffSamples(const std::filesystem::path&, bool strictLayout = true);
