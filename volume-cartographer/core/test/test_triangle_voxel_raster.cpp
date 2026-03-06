#include "vc/core/util/TriangleVoxelRaster.hpp"

#include <algorithm>
#include <cassert>
#include <set>
#include <tuple>
#include <vector>

namespace {

using Shape3 = std::array<size_t, 3>;

std::set<std::tuple<int, int, int>> rasterize(const cv::Vec3f& a,
                                              const cv::Vec3f& b,
                                              const cv::Vec3f& c,
                                              vc::core::util::RasterSliceAxis axis,
                                              float planeOffset,
                                              const Shape3& shape)
{
    std::set<std::tuple<int, int, int>> out;
    vc::core::util::rasterizeTriangleOnAxisSlices(a, b, c, axis, planeOffset, shape, [&](const auto& voxel) {
        out.emplace(voxel.z, voxel.y, voxel.x);
    });
    return out;
}

}  // namespace

int main()
{
    const Shape3 shape{32, 32, 32};

    {
        const cv::Vec3f a{2.0f, 2.0f, 10.2f};
        const cv::Vec3f b{10.0f, 2.0f, 10.8f};
        const cv::Vec3f c{2.0f, 10.0f, 10.6f};

        const auto integerZ = rasterize(a, b, c, vc::core::util::RasterSliceAxis::Z, 0.0f, shape);
        const auto halfZ = rasterize(a, b, c, vc::core::util::RasterSliceAxis::Z, 0.5f, shape);
        assert(integerZ.empty());
        assert(!halfZ.empty());
    }

    {
        const cv::Vec3f a{2.0f, 8.2f, 10.2f};
        const cv::Vec3f b{18.0f, 8.8f, 10.4f};
        const cv::Vec3f c{4.0f, 8.5f, 18.0f};

        const auto integerZ = rasterize(a, b, c, vc::core::util::RasterSliceAxis::Z, 0.0f, shape);
        const auto integerY = rasterize(a, b, c, vc::core::util::RasterSliceAxis::Y, 0.0f, shape);
        assert(integerZ.empty());
        assert(!integerY.empty());
    }

    {
        const cv::Vec3f a{6.2f, 2.0f, 3.0f};
        const cv::Vec3f b{6.8f, 14.0f, 3.2f};
        const cv::Vec3f c{6.4f, 4.0f, 12.0f};

        const auto integerZ = rasterize(a, b, c, vc::core::util::RasterSliceAxis::Z, 0.0f, shape);
        const auto integerX = rasterize(a, b, c, vc::core::util::RasterSliceAxis::X, 0.0f, shape);
        assert(integerZ.empty());
        assert(!integerX.empty());
    }

    return 0;
}
