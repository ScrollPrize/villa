#include "Fiber3DSurfaceMesh.hpp"
#include "Fiber3DCT.hpp"
#include "OpenedVolumes.hpp"
#include "vc/core/types/Volume.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::fibers3d::surface
{
std::shared_ptr<Texture> sampleTexture(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f scale,
                                     OpenedVolumes& volumes, const Volume& volume, const std::atomic_bool& cancelled, int level, int maxEdge)
{
    if (points.rows < 2 || points.cols < 2 || maxEdge < 2 || maxEdge > 2048 ||
        !std::isfinite(scale[0]) || !std::isfinite(scale[1]) || scale[0] <= 0 || scale[1] <= 0)
        throw std::invalid_argument("Invalid surface texture dimensions or scale");
    if (!volumes.pyramid(volume).has(level))
        throw std::invalid_argument("Requested surface CT texture level is unavailable");
    const double width = (points.cols-1) / double(scale[0]), height = (points.rows-1) / double(scale[1]);
    const double step = std::max({1., width/(maxEdge-1), height/(maxEdge-1)});
    auto texture = std::make_shared<Texture>();
    texture->level = level;
    texture->width = std::clamp(int(std::ceil(width/step))+1, 2, maxEdge);
    texture->height = std::clamp(int(std::ceil(height/step))+1, 2, maxEdge);
    texture->pixels.resize(size_t(texture->width)*texture->height*2);
    // A texel needs all four corners of its quad to be valid vertices.
    const auto bilinear = [&](const cv::Vec2d& grid, cv::Vec3f& out) {
        const int c = std::min(int(grid[0]), points.cols-2), r = std::min(int(grid[1]), points.rows-2);
        const auto valid = [](const cv::Vec3f& p) { return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) && p[0] >= 0 && p[1] >= 0 && p[2] >= 0; };
        if (!valid(points(r,c)) || !valid(points(r,c+1)) || !valid(points(r+1,c)) || !valid(points(r+1,c+1)))
            return false;
        const double fx = grid[0]-c, fy = grid[1]-r;
        const cv::Vec3d top = cv::Vec3d(points(r,c))*(1-fx)+cv::Vec3d(points(r,c+1))*fx;
        const cv::Vec3d bottom = cv::Vec3d(points(r+1,c))*(1-fx)+cv::Vec3d(points(r+1,c+1))*fx;
        out = cv::Vec3f(top*(1-fy)+bottom*fy);
        return true;
    };
    for (int first=0; first<texture->height; first+=16) {
        if (cancelled.load()) return {};
        const int rows = std::min(16, texture->height-first);
        cv::Mat_<cv::Vec3f> coords(rows, texture->width, cv::Vec3f{-1,-1,-1});
        for (int y=0; y<rows; ++y)
            for (int x=0; x<texture->width; ++x) {
                const cv::Vec2d grid{double(x)*(points.cols-1)/(texture->width-1), double(first+y)*(points.rows-1)/(texture->height-1)};
                bilinear(grid, coords(y,x));
            }
        const auto samples = ct::sampleCoordinates(coords, volumes, volume, level);
        for (int y=0; y<rows; ++y)
            std::copy_n(reinterpret_cast<const uint8_t*>(samples.ptr(y)), size_t(texture->width)*2,
                        texture->pixels.data()+size_t(first+y)*texture->width*2);
    }
    return texture;
}
}
