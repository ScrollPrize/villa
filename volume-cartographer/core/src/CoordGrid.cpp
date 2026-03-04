#include "vc/core/util/CoordGrid.hpp"

namespace vc {

cv::Mat_<cv::Vec3f> makeCoordGrid(
    const cv::Vec3f& origin, const cv::Vec3f& axisU, const cv::Vec3f& axisV, int w, int h)
{
    cv::Mat_<cv::Vec3f> coords(h, w);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            coords(r, c) = origin + axisU * static_cast<float>(c)
                                  + axisV * static_cast<float>(r);
    return coords;
}

}  // namespace vc
