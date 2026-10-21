#pragma once
#include "../ops/channel_misalignment.hpp"
#include "../shared/voxel_transform.hpp"
namespace vcr::transforms {
namespace volume_detail {
inline Parameters misalignment_parameters(const ChannelMisalignment& t,const Sample& s,Random& r) {
    Parameters p;
    for(const auto& [key,f]:s)if(f.target.role==Role::Image && (t.options.on.empty() || std::find(t.options.on.begin(),t.options.on.end(),key)!=t.options.on.end())){
        require_image(f.array);auto channels=t.options.channels;if(channels.empty()){channels.resize(f.array.shape[0]);std::iota(channels.begin(),channels.end(),0);}
        Parameters shared;if(t.options.shared)shared=sample(t.affine,s,r);
        for(auto c:channels){require(c<f.array.shape[0],"channel out of bounds");if(!r.chance(t.options.p_channel))continue;auto q=t.options.shared?shared:sample(t.affine,s,r);ChannelPlan cp;cp.key=key;cp.channel=c;cp.values=q.matrix;cp.values.insert(cp.values.end(),q.offset.begin(),q.offset.end());p.channels.push_back(std::move(cp));}}
    return p;
}
inline Parameters channel_geometry(const ChannelPlan& c,const Shape& shape,bool border){auto q=geometry(shape);q.matrix.assign(c.values.begin(),c.values.begin()+shape.size()*shape.size());q.offset.assign(c.values.begin()+shape.size()*shape.size(),c.values.end());q.border=border;return q;}
}
inline Parameters sample(const ChannelMisalignment& t,const Sample& s,Random& r){return volume_detail::misalignment_parameters(t,s,r);}
inline void apply(const ChannelMisalignment& t,Sample& s,const Parameters& p){
    for(const auto& c:p.channels){auto& f=s.at(c.key);auto shape=f.array.shape;shape[0]=1;Array one(shape,'f',f.array.bytes);one.channel(0,f.array.channel(c.channel));auto q=volume_detail::channel_geometry(c,f.array.spatial(),t.affine.border);auto out=resample(one,q,false,f.target.fill);f.array.channel(c.channel,out.channel(0));}restore_valid(s);
}

} // namespace vcr::transforms
