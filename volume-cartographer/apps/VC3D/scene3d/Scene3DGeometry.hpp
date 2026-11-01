#pragma once
#include <array>
#include <cmath>

namespace vc3d::scene3d
{
// Level-0 CT coordinates; subtract origins in double precision before upload.
using Point = std::array<double, 3>;
struct Bounds { Point lower{}, upper{}; };
inline double distance(const Point& a, const Point& b)
{
    return std::hypot(b[0]-a[0], b[1]-a[1], b[2]-a[2]);
}
}
