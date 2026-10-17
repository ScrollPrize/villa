#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct CLAHE {Scalar clip_limit=2.;Shape tile_shape{16,16,16};int bins=256;ChannelOptions options;};
inline ChannelPlan channel_parameters(const CLAHE& t,const Array& a,Random& r) {
    volume_detail::Grid3 grid(a.spatial());double clip=draw(t.clip_limit,r);require(clip>=0,"clip_limit must be nonnegative");
    volume_detail::HistogramGrid layout(grid,t.tile_shape,t.bins,clip);ChannelPlan p;p.values={clip};return p;
}
struct CLAHEKernel {
    volume_detail::HistogramGrid layout;
    template<class T> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T>& c) const {
        using namespace volume_detail;double span=c.input.high-c.input.low;if(span==0)return v[i];
        return c.input.low+span*interpolate_histograms(c.grid,layout,i,bin(v[i],c.input.low,c.input.high,layout.bins),c.table);
    }
};
inline void channel_apply(const CLAHE& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::Grid3 grid(g.shape);volume_detail::HistogramGrid layout(grid,t.tile_shape,t.bins,p.values[0]);
    auto table=volume_detail::channel_histograms(v,grid,layout);volume_detail::map_channel(v,g,CLAHEKernel{layout},table.data());
}
inline void apply(const CLAHE& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
