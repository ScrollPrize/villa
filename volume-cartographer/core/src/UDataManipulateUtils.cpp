#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/ui/UDataManipulateUtils.hpp"

// Convert from QImage to cv::Mat
auto QImage2Mat(const QImage& nSrc) -> cv::Mat
{
    cv::Mat tmp(
        nSrc.height(), nSrc.width(), CV_8UC3, const_cast<uchar*>(nSrc.bits()),
        nSrc.bytesPerLine());
    cv::Mat result;  // deep copy
    cvtColor(tmp, result, cv::COLOR_BGR2RGB);
    return result;
}

// Convert from cv::Mat to QImage
auto Mat2QImage(const cv::Mat& nSrc) -> QImage
{
    cv::Mat tmp;
    cvtColor(nSrc, tmp, cv::COLOR_BGR2RGB);  // copy and convert color space
    QImage result(
        static_cast<const std::uint8_t*>(tmp.data), tmp.cols, tmp.rows,
        tmp.step, QImage::Format_RGB888);
    result.bits();  // enforce depp copy, see documentation of
    // QImage::QImage( const uchar *dta, int width, int height, Format format )
    return result;
}

