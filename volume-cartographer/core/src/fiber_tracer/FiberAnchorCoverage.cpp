#include "vc/fiber_tracer/FiberAnchorCoverage.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace vc::fiber_tracer {
nlohmann::json measureFiberAnchorCoverage(
    const std::vector<cv::Vec3d>& points, std::span<const cv::Vec3d> anchors,
    const vc::lasagna::NormalSampler& normals, double normalScale,
    double radius, double factor, double cellSide, std::span<const cv::Vec3d> fittedAxes)
{
    using Json = nlohmann::json;
    if (!fittedAxes.empty() && fittedAxes.size()!=anchors.size())
        throw std::invalid_argument("anchor fitted-axis count differs");
    if (!(radius > 0 && factor > 0 && cellSide > 0) || !std::isfinite(radius + factor + cellSide))
        throw std::invalid_argument("invalid anchor coverage dimensions");
    const auto line = makePolylineArcGeometry(points);
    if (!(line.length() > 0)) throw std::invalid_argument("anchor reference has zero length");
    const double broadRadius = radius * std::max(1.0, factor);
    // Index short segment intervals in a uniform grid. Queries inspect only
    // nearby intervals; projection arithmetic remains the shared implementation.
    using Key = std::array<int64_t, 3>;
    auto key = [&](const cv::Vec3d& p) {
        return Key{int64_t(std::floor(p[0]/broadRadius)),
                   int64_t(std::floor(p[1]/broadRadius)),
                   int64_t(std::floor(p[2]/broadRadius))};
    };
    std::vector<std::pair<double,double>> intervals;
    std::map<Key,std::vector<size_t>> buckets;
    for (size_t s=0; s+1<line.points.size(); ++s) {
        const double begin=line.vertexArcs[s], length=line.vertexArcs[s+1]-begin;
        const size_t count=std::max(size_t{1},size_t(std::ceil(length/broadRadius)));
        for (size_t j=0;j<count;++j) {
            const double lo=begin+length*j/count, hi=begin+length*(j+1)/count;
            if (!(hi>lo)) continue;
            const auto a=samplePolylineArc(line,lo).point, b=samplePolylineArc(line,hi).point;
            cv::Vec3d lower,upper;
            for (int d=0;d<3;++d) {lower[d]=std::min(a[d],b[d]);upper[d]=std::max(a[d],b[d]);}
            const auto low=key(lower), high=key(upper);
            for(auto x=low[0];x<=high[0];++x) for(auto y=low[1];y<=high[1];++y)
                for(auto z=low[2];z<=high[2];++z) buckets[{x,y,z}].push_back(intervals.size());
            intervals.emplace_back(lo,hi);
        }
    }
    Json accepted=Json::array();
    std::vector<double> arcs;
    size_t fallback=0, endRejected=0, tested=0;
    for(size_t i=0;i<anchors.size();++i) {
        const auto& anchor=anchors[i];
        for (double v: {anchor[0],anchor[1],anchor[2]})
            if (!std::isfinite(v)) throw std::invalid_argument("nonfinite anchor");
        const auto k=key(anchor);
        std::vector<size_t> nearby;
        for(int x=-1;x<=1;++x) for(int y=-1;y<=1;++y) for(int z=-1;z<=1;++z)
            if(auto it=buckets.find({k[0]+x,k[1]+y,k[2]+z});it!=buckets.end())
                nearby.insert(nearby.end(),it->second.begin(),it->second.end());
        std::sort(nearby.begin(),nearby.end());
        nearby.erase(std::unique(nearby.begin(),nearby.end()),nearby.end());
        PolylineArcProjection best;
        best.distance=std::numeric_limits<double>::infinity();
        for(auto id:nearby) {
            const auto [lo,hi]=intervals[id];
            const auto p=projectPointToPolylineArc(line,anchor,lo,hi);
            if(p.distance<best.distance-1e-12 ||
                (std::abs(p.distance-best.distance)<=1e-12 && p.arc<best.arc)) best=p;
        }
        if(best.distance>broadRadius) continue;
        // Flat caps: do not count anchors beyond the eligible CP/Lasagna span.
        if ((best.arc<=0 && (anchor-line.points.front()).dot(samplePolylineArc(line,0).tangent)<0) ||
            (best.arc>=line.length() && (anchor-line.points.back()).dot(samplePolylineArc(line,line.length()).tangent)>0)) {
            ++endRejected; continue;
        }
        const auto measurement=measureFiberReplayThreshold(anchor,best.point,normals,normalScale,radius,factor);
        ++tested;
        fallback+=!measurement.localNormalValid;
        if(fiberReplayThresholdExceeded(measurement,radius)) continue;
        arcs.push_back(best.arc);
        accepted.push_back({{"anchor_index",i},{"position_base_xyz",{anchor[0],anchor[1],anchor[2]}},
            {"reference_arc_base",best.arc},{"reference_point_base_xyz",{best.point[0],best.point[1],best.point[2]}},
            {"threshold_ratio",measurement.thresholdErrorRatio},{"normal_valid",measurement.localNormalValid}});
        accepted.back()["direction_error_degrees"]=nullptr;
        if (!fittedAxes.empty()) {
            const auto tangent=samplePolylineArc(line,best.arc).tangent;
            const auto axis=fittedAxes[i];
            const double norm=cv::norm(axis)*cv::norm(tangent);
            if (std::isfinite(norm) && norm>0) {
                const double cosine=std::clamp(std::abs(axis.dot(tangent))/norm,0.0,1.0);
                accepted.back()["direction_error_degrees"]=std::acos(cosine)*180.0/std::acos(-1.0);
            }
        }
    }
    std::sort(arcs.begin(),arcs.end());
    arcs.erase(std::unique(arcs.begin(),arcs.end()),arcs.end());
    std::vector<double> gaps;
    for(size_t i=1;i<arcs.size();++i) gaps.push_back(arcs[i]-arcs[i-1]);
    const size_t binCount=std::max(size_t{1},size_t(std::ceil(line.length()/cellSide)));
    std::vector<bool> occupied(binCount,false);
    for(auto arc:arcs) occupied[std::min(binCount-1,size_t(arc/cellSide))]=true;
    std::vector<size_t> emptyRuns;
    size_t empty=0,run=0;
    double emptyLength=0;
    for(size_t i=0;i<binCount;++i) {
        if(!occupied[i]) {
            ++empty; ++run;
            emptyLength+=std::min(cellSide,line.length()-i*cellSide);
        } else if(run) {emptyRuns.push_back(run);run=0;}
    }
    if(run) emptyRuns.push_back(run);
    return {{"protocol","anchor-nearest-projection-flat-caps-v1"},{"reference_length_base",line.length()},
        {"cell_side_base",cellSide},{"candidate_anchors",anchors.size()},{"ellipse_tests",tested},
        {"invalid_normal_tests",fallback},{"end_cap_rejections",endRejected},{"accepted_anchors",accepted},
        {"unique_arcs_base",arcs},{"interior_gaps_base",gaps},
        {"leading_gap_base",arcs.empty()?Json(nullptr):Json(arcs.front())},
        {"trailing_gap_base",arcs.empty()?Json(nullptr):Json(line.length()-arcs.back())},
        {"no_anchor_length_base",arcs.empty()?line.length():0.0},
        {"arc_bins",binCount},{"empty_arc_bins",empty},{"empty_arc_bin_length_base",emptyLength},
        {"empty_arc_bin_runs",emptyRuns},{"threshold",fiberReplayThresholdDescriptorJson(radius,factor)}};
}
}
