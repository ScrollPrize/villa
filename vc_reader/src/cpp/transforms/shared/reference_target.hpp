#pragma once
#include "voxel_transform.hpp"
namespace vcr::transforms::volume_detail {
inline const Array& reference_target(const std::string& key,const Sample& s,const Parameters& p) {
    auto it=s.find(key);require(it!=s.end(),"reference target is missing");
    const auto& ref=it->second.array;
    require(ref.kind=='f' && (ref.bytes==4 || ref.bytes==8),"reference target must be floating point");
    for(const auto& c:p.channels){
        require(c.key!=key,"exclude the reference target using on=");
        const auto& image=s.at(c.key).array;
        require(ref.spatial()==image.spatial(),"reference must match spatial shape");
        require(ref.shape[0]==1 || ref.shape[0]==image.shape[0],"reference needs one channel or matching image channels");
    }
    return ref;
}
template<class Transform> void apply_reference_channels(const Transform& t,Sample& s,const Parameters& p) {
    if(t.reference_key.empty()){apply_checked_channels(t,s,p);return;}
    const auto& ref=reference_target(t.reference_key,s,p);
    auto resolved=t;resolved.reference.shape=ref.shape;resolved.reference.wide=ref.bytes==8;
    resolved.reference.values=Array(ref.shape,ref.kind,ref.bytes,ref.data());
    apply_checked_channels(resolved,s,p);
}
} // namespace vcr::transforms::volume_detail
