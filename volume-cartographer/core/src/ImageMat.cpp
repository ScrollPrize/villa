#include "vc/core/util/ImageMat.hpp"

#include <vc_reader/source_reader.hpp>

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace vc::core::util {

const char* tiffDtype(int depth)
{
    switch (depth) {
    case CV_8U: return "|u1";
    case CV_8S: return "|i1";
    case CV_16U: return "<u2";
    case CV_16S: return "<i2";
    case CV_32S: return "<i4";
    case CV_32F: return "<f4";
    case CV_64F: return "<f8";
    }
    return nullptr;
}

cv::Mat imageMat(const vcr::sources::Block& block)
{
    if (block.shape.size() < 2 || block.shape.size() > 3)
        return {};
    const int channels = block.shape.size() == 3 ? int(block.shape[2]) : 1;
    if (channels > CV_CN_MAX)
        return {};
    const int rows = int(block.shape[0]), cols = int(block.shape[1]);
    cv::Mat stored;
    if (block.dtype == "|b1") {
        stored = cv::Mat(rows, cols, CV_MAKETYPE(CV_8U, channels), block.data.get()) * 255;
    } else {
        for (const int depth : {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F})
            if (block.dtype == tiffDtype(depth))
                stored = cv::Mat(rows, cols, CV_MAKETYPE(depth, channels), block.data.get());
        if (stored.empty())
            return {};
    }
    cv::Mat image;
    if (channels == 3) cv::cvtColor(stored, image, cv::COLOR_RGB2BGR);
    else if (channels == 4) cv::cvtColor(stored, image, cv::COLOR_RGBA2BGRA);
    else image = stored.clone();
    return image;
}

vcr::sources::ImageView ImageBuffer::view() const
{
    return {pixels.data,tiffDtype(pixels.depth()),pixels.rows,pixels.cols,pixels.channels()};
}

ImageBuffer imageBuffer(const cv::Mat& mat)
{
    if (mat.empty() || !tiffDtype(mat.depth()))
        throw std::invalid_argument("unsupported image buffer");
    cv::Mat stored;
    if (mat.channels() == 3) cv::cvtColor(mat, stored, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 4) cv::cvtColor(mat, stored, cv::COLOR_BGRA2RGBA);
    else stored = mat.isContinuous() ? mat : mat.clone();
    return {std::move(stored)};
}

}  // namespace vc::core::util
