#include "../cpu/labels.hpp"
#include "geometry.cuh"
#include "label_values.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const DeepSupervision& t,Sample& s,const Parameters&,Session& session) {
    auto out=s.at(t.on);require(!t.scales.empty(),"deep supervision requires scales");
    const auto source=session.arrays.at(t.on);std::vector<DeviceArray> levels;
    for(auto scale:t.scales) {
        auto shape=source.spatial();if(scale.size()==1)scale.resize(shape.size(),scale[0]);require(scale.size()==shape.size(),"invalid deep-supervision scale dimensions");
        for(std::size_t a=0;a<shape.size();++a){require(std::isfinite(scale[a]) && scale[a]>0 && scale[a]<=1,"deep-supervision scales must be in (0, 1]");shape[a]=std::max<std::size_t>(1,std::size_t(round_even(shape[a]*scale[a])));}
        levels.push_back(resize_cuda(source,shape,true,false));out.pyramid.push_back(array_metadata(levels.back()));
    }
    const auto key=t.output.empty()?t.on:t.output;out.source=t.on;s[key]=std::move(out);
    session.arrays.insert_or_assign(key,source);session.pyramids.insert_or_assign(key,std::move(levels));
}
}
