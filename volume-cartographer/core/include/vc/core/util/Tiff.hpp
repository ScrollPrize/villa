#pragma once

#include <opencv2/core.hpp>
#include <filesystem>

// Write a 32-bit float single-channel image as tiled TIFF with LZW compression
void writeFloatTiff(const std::filesystem::path& outPath,
                    const cv::Mat& img,
                    uint32_t tileW = 1024,
                    uint32_t tileH = 1024);

// Write a single-channel image (8U, 16U, or 32F) as tiled TIFF with LZW compression
void writeSingleChannelTiff(const std::filesystem::path& outPath,
                            const cv::Mat& img,
                            uint32_t tileW = 1024,
                            uint32_t tileH = 1024);