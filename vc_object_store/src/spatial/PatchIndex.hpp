#pragma once
#include "spatial/PatchGrid.hpp"
#include "spatial/BoundsIndex.hpp"
#include <memory>
#include <span>
#include <optional>
#include <unordered_set>
namespace vc::object_links {
class PatchIndex {
public:
    using SurfacePtr=std::shared_ptr<PatchGrid>;
    struct SurfaceFilter {const std::unordered_set<PatchGrid*>* includeRaw=nullptr;};
    struct PointQuery {cv::Vec3f worldPoint;float tolerance=1;SurfaceFilter surfaces;};
    struct LookupResult {SurfacePtr surface; cv::Vec2f grid; float distance;};
    void rebuild(const std::vector<SurfacePtr>& surfaces);
    std::vector<LookupResult> locateAll(const PointQuery&) const;
    std::optional<LookupResult> locate(const PointQuery&) const;
    template<class Visitor> void locateAllBatch(std::span<const cv::Vec3f> points,float tolerance,
        const SurfaceFilter& surfaces,Visitor visitor) const {
        for(size_t i=0;i<points.size();++i)visitor(i,locateAll({points[i],tolerance,surfaces}));
    }
private:
    struct Tile {size_t surface;int row,col;};
    std::vector<SurfacePtr> surfaces_;
    std::vector<Tile> tiles_;
    objects::detail::BoundsIndex bounds_;
};
}
