#pragma once

#include <qimage.h>

#include <opencv2/core.hpp>



cv::Mat QImage2Mat(const QImage& nSrc);

QImage Mat2QImage(const cv::Mat& nSrc);

