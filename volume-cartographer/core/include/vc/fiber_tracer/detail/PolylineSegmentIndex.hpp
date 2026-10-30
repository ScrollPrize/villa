#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <opencv2/core/types.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace vc::fiber_tracer::detail {

// Immutable segment-box broad phase using VC3D's existing Boost.Geometry R-tree.
// Returns conservative path candidates, NOT nearest vertices/segments. The
// caller retains its exact projection, acceptance tests and original tie order.
class ChunkSegmentIndex {
    using Point = boost::geometry::model::point<double,3,boost::geometry::cs::cartesian>;
    using Box = boost::geometry::model::box<Point>;
    using Entry = std::pair<Box,size_t>;
    using Tree = boost::geometry::index::rtree<Entry,boost::geometry::index::quadratic<16>>;
    Tree tree_;

public:
    ChunkSegmentIndex()=default;
    template<class Paths,class PointsOf>
    ChunkSegmentIndex(const Paths& paths,PointsOf pointsOf) {
        std::vector<Entry> entries;
        for(size_t path=0;path<paths.size();++path) {
            const auto& points=pointsOf(paths[path]);
            for(size_t s=1;s<points.size();++s) {
                cv::Vec3d lo,hi;
                for(int a=0;a<3;++a) {
                    // Outward-only numerical padding; it can admit extra
                    // candidates, never relax the downstream exact thresholds.
                    const double pad=64*std::numeric_limits<double>::epsilon()*
                        std::max({1.,std::abs(points[s-1][a]),std::abs(points[s][a])});
                    lo[a]=std::nextafter(std::min(points[s-1][a],points[s][a])-pad,-std::numeric_limits<double>::infinity());
                    hi[a]=std::nextafter(std::max(points[s-1][a],points[s][a])+pad,std::numeric_limits<double>::infinity());
                }
                entries.emplace_back(Box(Point(lo[0],lo[1],lo[2]),Point(hi[0],hi[1],hi[2])),path);
            }
        }
        tree_=Tree(entries.begin(),entries.end());
    }

    template<class Visitor>
    void nearbyPaths(const cv::Vec3d& p,double radius,Visitor&& visit) const {
        const double pad=64*std::numeric_limits<double>::epsilon()*
            std::max({1.,radius,std::abs(p[0]),std::abs(p[1]),std::abs(p[2])});
        const double r=radius+pad;
        const Box query(Point(p[0]-r,p[1]-r,p[2]-r),Point(p[0]+r,p[1]+r,p[2]+r));
        for(auto it=tree_.qbegin(boost::geometry::index::intersects(query));it!=tree_.qend();++it)
            visit(it->second); // A path may occur more than once for a query.
    }
};
} // namespace vc::fiber_tracer::detail
