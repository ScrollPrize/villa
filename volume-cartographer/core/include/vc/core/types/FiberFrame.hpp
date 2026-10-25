#pragma once

#include <vc_object_store/objects/VolumeTransforms.hpp>
#include <opencv2/core/matx.hpp>
#include <cmath>

namespace vc::fibers
{
// Fiber attributes between frames. Directions follow the affine's linear
// part and normals its inverse transpose, both kept unit length; a uniform
// scale changes neither, so they are kept exactly. Widths scale by the
// uniform factor (the cube root of the determinant's magnitude otherwise).
inline bool uniformScale(const vc::objects::VolumeAffine& m)
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            if (r != c && m(r, c) != 0) return false;
    return m(0, 0) > 0 && m(0, 0) == m(1, 1) && m(1, 1) == m(2, 2);
}
inline cv::Vec3d mapFiberDirection(const vc::objects::VolumeAffine& m, const cv::Vec3d& d)
{
    if (uniformScale(m)) return d;
    const auto q = m.apply({d[0], d[1], d[2]}, true);
    const cv::Vec3d v{q[0], q[1], q[2]};
    const double n = std::sqrt(v.dot(v));
    return n > 0 ? v / n : v;
}
inline cv::Vec3d mapFiberNormal(const vc::objects::VolumeAffine& m, const cv::Vec3d& normal)
{
    if (uniformScale(m)) return normal;
    const auto inverse = m.inverse();
    cv::Vec3d v;
    for (int i = 0; i < 3; ++i) v[i] = inverse(0, i) * normal[0] + inverse(1, i) * normal[1] + inverse(2, i) * normal[2];
    const double n = std::sqrt(v.dot(v));
    return n > 0 ? v / n : v;
}
inline double mapFiberWidth(const vc::objects::VolumeAffine& m, double width)
{
    if (uniformScale(m)) return width * m(0, 0);
    const double det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
                       m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    return width * std::cbrt(std::abs(det));
}

} // namespace vc::fibers
