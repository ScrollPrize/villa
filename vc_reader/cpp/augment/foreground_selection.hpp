#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment::volume_detail {
// Reservoir sampling visits foreground in a fixed order. Only the resulting
// index crosses from device to host for geometric parameter construction.
template<class T> VCR_HD std::size_t foreground_index(const T* mask,std::size_t n,double threshold,std::uint64_t seed) {
    std::size_t chosen=n,count=0;for(std::size_t i=0;i<n;++i)if(double(mask[i])>threshold){++count;if(uniform(seed,i)<1./double(count))chosen=i;}return chosen;
}
inline const Field& mask_field(const Sample& s,const std::string& name,std::size_t channel) {
    require(s.count(name),"mask target is missing: "+name);const auto& f=s.at(name);
    require((f.target.role==Role::Label || f.target.role==Role::Valid) && f.array.shape.size()==4 && channel<f.array.shape[0],"mask must be a volumetric Label/ValidMask with the requested channel");return f;
}
}
