#pragma once

#include <opencv2/core.hpp>

void customThinning(const cv::Mat& inputImage, cv::Mat& outputImage, int iterations);
