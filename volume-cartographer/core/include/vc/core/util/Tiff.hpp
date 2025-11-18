#pragma once

#include <opencv2/calib3d.hpp>

// Use libtiff for BigTIFF; fall back to OpenCV if not present.
#include <filesystem>
#include <tiffio.h>

// Options for writing TIFF from QuadSurface
struct TiffWriteOptions {
    enum class Compression { NONE, LZW, DEFLATE };
    enum class Predictor  { NONE, HORIZONTAL, FLOATINGPOINT };

    bool forceBigTiff = false;
    int  tileSize = 1024;                 // square tiles
    Compression compression = Compression::LZW;
    Predictor  predictor   = Predictor::FLOATINGPOINT; // for float32
};

// Write a 32-bit float single-channel image as tiled BigTIFF with LZW compression
void writeFloatBigTiff(const std::filesystem::path& outPath,
                              const cv::Mat& img,
                              uint32_t tileW = 1024,
                              uint32_t tileH = 1024);

// Write a single-channel image (8U, 16U, or 32F) as tiled BigTIFF with LZW compression
void writeSingleChannelBigTiff(const std::filesystem::path& outPath,
                                      const cv::Mat& img,
                                      uint32_t tileW = 1024,
                                      uint32_t tileH = 1024);