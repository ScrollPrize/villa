#pragma once
#include <vc_reader/components.hpp>
#include "../distance/common.hpp"
namespace vcr::components::detail {
inline std::size_t validate(const void* input,void* output,const std::vector<std::size_t>& shape) {
    const auto n=distance::detail::validate(input,output,shape);
    if(n>UINT32_MAX) throw std::invalid_argument("component labels require at most UINT32_MAX voxels");
    return n;
}
}
