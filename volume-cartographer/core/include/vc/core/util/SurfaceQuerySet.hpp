#pragma once
#include <vc_object_store/geometry/SurfaceIndex.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>
class QuadSurface;

// Associates application surfaces with immutable resident geometry. The store owns
// the spatial queries; this set owns application identity and ptr-space conversion.
class SurfaceQuerySet {
public:
    using SurfacePtr = std::shared_ptr<QuadSurface>;
    struct LookupResult { SurfacePtr surface; cv::Vec3f ptr; float distance; };
    struct SurfaceFilter { SurfacePtr only; };
    struct PointQuery { cv::Vec3f worldPoint{}; float tolerance=0; SurfaceFilter surfaces; };
    void rebuild(const std::vector<SurfacePtr>& surfaces);
    bool empty() const { return index_.empty(); }
    size_t tileCount() const { return index_.tileCount(); }
    std::optional<LookupResult> locate(const PointQuery& query) const;
    std::vector<LookupResult> locateAll(const PointQuery& query) const;
private:
    vc::object_links::SurfaceIndex index_;
    std::unordered_map<vc::object_links::SurfaceGrid*,SurfacePtr> identities_;
    std::unordered_map<QuadSurface*,std::shared_ptr<vc::object_links::SurfaceGrid>> geometry_;
    vc::object_links::SurfaceIndex::PointQuery queryFor(const PointQuery& query) const;
};
