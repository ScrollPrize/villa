#include "Fiber3DSurfaceMesh.hpp"
#include "Fiber3DCT.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::fibers3d::surface
{
std::shared_ptr<Texture> sampleTexture(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f scale,
                                     Volume& volume, const std::atomic_bool& cancelled, int level, int maxEdge)
{
    if (points.rows < 2 || points.cols < 2 || maxEdge < 2 || maxEdge > 2048 ||
        !std::isfinite(scale[0]) || !std::isfinite(scale[1]) || scale[0] <= 0 || scale[1] <= 0)
        throw std::invalid_argument("Invalid surface texture dimensions or scale");
    if (!volume.hasScaleLevel(level))
        throw std::invalid_argument("Requested surface CT texture level is unavailable");
    const double width = (points.cols-1) / double(scale[0]), height = (points.rows-1) / double(scale[1]);
    const double step = std::max({1., width/(maxEdge-1), height/(maxEdge-1)});
    auto texture = std::make_shared<Texture>();
    texture->level = level;
    texture->width = std::clamp(int(std::ceil(width/step))+1, 2, maxEdge);
    texture->height = std::clamp(int(std::ceil(height/step))+1, 2, maxEdge);
    texture->pixels.resize(size_t(texture->width)*texture->height*2);
    // Share the immutable worker snapshot; reuse QuadSurface's four-corner
    // validity and interpolation instead of inventing another hole policy.
    QuadSurface snapshot(new cv::Mat_<cv::Vec3f>(points), {1,1});
    for (int first=0; first<texture->height; first+=16) {
        if (cancelled.load()) return {};
        const int rows = std::min(16, texture->height-first);
        cv::Mat_<cv::Vec3f> coords(rows, texture->width, cv::Vec3f{-1,-1,-1});
        for (int y=0; y<rows; ++y)
            for (int x=0; x<texture->width; ++x) {
                const cv::Vec2d grid{double(x)*(points.cols-1)/(texture->width-1), double(first+y)*(points.rows-1)/(texture->height-1)};
                const auto sample = snapshot.sampleAtSurface(snapshot.gridToSurface(grid));
                if (sample) coords(y,x) = sample.volume;
            }
        const auto samples = ct::sampleCoordinates(coords, volume, level);
        for (int y=0; y<rows; ++y)
            std::copy_n(reinterpret_cast<const uint8_t*>(samples.ptr(y)), size_t(texture->width)*2,
                        texture->pixels.data()+size_t(first+y)*texture->width*2);
    }
    return texture;
}
}
