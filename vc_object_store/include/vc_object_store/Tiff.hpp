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

// True if `path` opens as a TIFF whose first directory is present and whose
// every tile (or strip) decodes to its full size. libtiff writes the directory
// when closing a file, so a file left behind by a
// writer that was killed mid-way has a header and tile data but no
// directory: exists() says "done", this says "torn". Decoding the payload
// additionally rejects a file whose directory survived but whose tile data
// is truncated or does not decompress; it reads the whole file, so this
// costs about as much as one pass over the output. Pixel values are not
// checked against anything. libtiff diagnostics are suppressed while
// probing; the caller decides what to report.
bool isReadableTiff(const std::filesystem::path& path);

// Atomic multi-page TIFF write: write all pages to <outPath>.tmp, then
// rename over outPath. A crash mid-write leaves the previous file (or
// nothing, if there was none) intact rather than a torn half-written
// TIFF. Throws on failure (write or rename). The .tmp is written
// inside the same directory as outPath to avoid EXDEV from cross-mount
// renames.
void atomicImwriteMulti(const std::filesystem::path& outPath,
                        const std::vector<cv::Mat>& pages);

// Native sample bytes in host byte order, interleaved in TIFF component order.
// No color conversion, scaling, mask normalization or float conversion.
struct TiffSamples {
    uint32_t width = 0, height = 0;
    uint16_t bits = 0, sampleFormat = 0, components = 0;
    std::vector<uint8_t> bytes;
};
TiffSamples readTiffSamples(const std::filesystem::path&, bool strictLayout = true);
