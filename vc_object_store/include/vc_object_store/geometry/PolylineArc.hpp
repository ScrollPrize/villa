#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace vc::geometry {

struct PolylineArcLocations {
    std::vector<int64_t> segment;
    std::vector<double> fraction;
};
struct PolylineArcLocation {int64_t segment;double fraction;};

// A non-owning index over packed polylines. Each interval [offset[i],offset[i+1])
// has at least two finite, nondecreasing arc coordinates, beginning at zero.
// A stride permits an arc column in an interleaved memory-mapped record array.
// The caller keeps both buffers alive and does not mutate them during queries.
class PolylineArcIndex {
public:
    PolylineArcIndex(std::span<const int64_t> offsets,const double* arcs,
                     size_t arcCount,ptrdiff_t stride=1)
        :offsets_(offsets),arcs_(arcs),stride_(stride)
    {
        if(arcCount>uint64_t(std::numeric_limits<int64_t>::max())||offsets.empty()||
           offsets.front()!=0||offsets.back()!=int64_t(arcCount)||stride<1||(!arcs&&arcCount)||
           (arcCount&&arcCount-1>uint64_t(std::numeric_limits<ptrdiff_t>::max()/stride)))
            throw std::invalid_argument("invalid packed polyline offsets/arc stride");
        for(size_t i=1;i<offsets.size();++i)
            if(offsets[i]<0||uint64_t(offsets[i])>arcCount||offsets[i]<offsets[i-1]||offsets[i]-offsets[i-1]<2)
                throw std::invalid_argument("packed polylines require at least two points");
    }

    PolylineArcLocations locate(std::span<const int64_t> owners,std::span<const double> positions)const
    {
        if(owners.size()!=positions.size())throw std::invalid_argument("arc owner/query sizes differ");
        PolylineArcLocations out;out.segment.resize(owners.size());out.fraction.resize(owners.size());
        for(size_t i=0;i<owners.size();++i){
            const auto value=locate(owners[i],positions[i]);
            out.segment[i]=value.segment;out.fraction[i]=value.fraction;
        }
        return out;
    }

    PolylineArcLocation locate(int64_t owner,double value)const
    {
            if(owner<0||uint64_t(owner)>=offsets_.size()-1)
                throw std::invalid_argument("unknown packed polyline index");
            const auto first=offsets_[owner],end=offsets_[owner+1];
            const auto length=arc(end-1);
            if(!std::isfinite(value)||!std::isfinite(length)||length<0||
               value < -1e-4 || value>length+std::max(1e-4,2e-7*length))
                throw std::invalid_argument("endpoint outside fiber");
            const auto clipped=std::clamp(value,0.,length);
            // upper_bound within this fiber, not a search over all packed points.
            int64_t lo=first,hi=end;
            while(lo<hi){const auto mid=lo+(hi-lo)/2;if(clipped<arc(mid))hi=mid;else lo=mid+1;}
            const auto j=std::min(lo-1,end-2);
            if(j<first)throw std::invalid_argument("invalid packed arc search");
            return {j,(clipped-arc(j))/std::max(arc(j+1)-arc(j),1e-300)};
    }
private:
    double arc(int64_t i)const{return arcs_[i*stride_];}
    std::span<const int64_t> offsets_;
    const double* arcs_;
    ptrdiff_t stride_;
};
} // namespace vc::geometry
