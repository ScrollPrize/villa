#pragma once

#include <opencv2/core.hpp>
#include <vector>

void customThinning(const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<std::vector<cv::Point>>* traces = nullptr);
