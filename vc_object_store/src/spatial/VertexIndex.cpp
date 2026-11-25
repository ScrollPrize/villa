#include <vc_object_store/geometry/VertexIndex.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <cmath>
#include <stdexcept>
namespace vc::geometry {
void VertexIndex::rebuild(const cv::Mat_<cv::Vec3f>& points, std::span<const std::uint8_t> valid) {
    if (!valid.empty() && valid.size()!=points.total()) throw std::invalid_argument("vertex validity shape mismatch");
    points_.clear(); indices_.clear();
    std::vector<objects::detail::BoundsIndex::Box> boxes;
    for (int r=0;r<points.rows;++r) for(int c=0;c<points.cols;++c) {
        const auto& p=points(r,c);
        const std::size_t i=std::size_t(r)*points.cols+c;
        if (!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2]) ||
            (valid.empty() ? !validSurfacePoint(p) : !valid[i])) continue;
        points_.push_back(p); indices_.push_back(i);
        boxes.push_back({p[0],p[1],p[2],p[0],p[1],p[2]});
    }
    bounds_.build(std::move(boxes));
}
std::optional<VertexIndex::Hit> VertexIndex::nearest(const cv::Vec3f& p,float maxDistance) const {
    if (!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2])||std::isnan(maxDistance)||maxDistance<0) return {};
    double distanceSquared=double(maxDistance)*maxDistance;
    std::optional<Hit> hit;
    bounds_.nearest({p[0],p[1],p[2]},distanceSquared,[&](std::size_t i) {
        // Use the same precision as bounds traversal so pruning cannot discard
        // a closer vertex after a candidate distance rounded down to float.
        const cv::Vec3d delta=cv::Vec3d(points_[i])-cv::Vec3d(p);
        const double d=delta.dot(delta);
        if (d<distanceSquared || (d==distanceSquared && (!hit || indices_[i]<hit->index))) {
            distanceSquared=d; hit=Hit{indices_[i],points_[i],float(d)};
        }
    });
    return hit;
}
}
