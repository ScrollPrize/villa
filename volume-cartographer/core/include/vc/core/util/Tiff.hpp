#pragma once

#include <opencv2/calib3d.hpp>

// Use libtiff for BigTIFF; fall back to OpenCV if not present.
#include <filesystem>
#include <tiffio.h>

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