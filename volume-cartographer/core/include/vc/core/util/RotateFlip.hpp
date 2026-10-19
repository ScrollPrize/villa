#pragma once

#include <opencv2/core.hpp>
#include <algorithm>

// rotQuad: 1 = 90 degrees CCW, 2 = 180, 3 = 90 degrees CW; 0 or -1 = none.
// flipAxis: 0 = vertical, 1 = horizontal, 2 = both; -1 = none. The flip applies after the rotation.
namespace vc {

inline cv::Size rotatedSize(cv::Size srcSize, int rotQuad)
{
    return (rotQuad == 1 || rotQuad == 3) ? cv::Size(srcSize.height, srcSize.width) : srcSize;
}

inline void rotateFlipIfNeeded(cv::Mat& m, int rotQuad, int flipAxis)
{
    if (rotQuad == 1) cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (rotQuad == 2) cv::rotate(m, m, cv::ROTATE_180);
    else if (rotQuad == 3) cv::rotate(m, m, cv::ROTATE_90_CLOCKWISE);
    if (flipAxis == 0) cv::flip(m, m, 0);
    else if (flipAxis == 1) cv::flip(m, m, 1);
    else if (flipAxis == 2) cv::flip(m, m, -1);
}

inline cv::Rect srcRectForDst(const cv::Rect& dst, cv::Size srcSize, int rotQuad, int flipAxis)
{
    const cv::Size outSize = rotatedSize(srcSize, rotQuad);
    const int outW = outSize.width, outH = outSize.height;
    auto sourcePoint = [&](int x, int y) {
        if (flipAxis == 1 || flipAxis == 2) x = outW - 1 - x;
        if (flipAxis == 0 || flipAxis == 2) y = outH - 1 - y;
        switch (rotQuad) {
        case 1: return cv::Point(srcSize.width - 1 - y, x);
        case 2: return cv::Point(srcSize.width - 1 - x, srcSize.height - 1 - y);
        case 3: return cv::Point(y, srcSize.height - 1 - x);
        default: return cv::Point(x, y);
        }
    };
    const cv::Point p0 = sourcePoint(dst.x, dst.y);
    const cv::Point p1 = sourcePoint(dst.x + dst.width - 1, dst.y);
    const cv::Point p2 = sourcePoint(dst.x, dst.y + dst.height - 1);
    const cv::Point p3 = sourcePoint(dst.x + dst.width - 1, dst.y + dst.height - 1);
    const int x0 = std::min({p0.x, p1.x, p2.x, p3.x});
    const int y0 = std::min({p0.y, p1.y, p2.y, p3.y});
    const int x1 = std::max({p0.x, p1.x, p2.x, p3.x});
    const int y1 = std::max({p0.y, p1.y, p2.y, p3.y});
    return cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

} // namespace vc
