#pragma once

#include <opencv2/core/matx.hpp>

namespace vc::grow_seed {

// Whether the seed parsed from --seed selects mode "explicit_seed".
// origin starts at zero, so 0 0 0 is read as no seed; any other seed is
// kept, including one with a single coordinate at zero (-s 0 3936 830).
inline bool is_explicit_seed(const cv::Vec3d& origin)
{
    return origin[0] != 0 || origin[1] != 0 || origin[2] != 0;
}

} // namespace vc::grow_seed
