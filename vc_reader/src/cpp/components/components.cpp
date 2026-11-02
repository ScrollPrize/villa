#include "common.hpp"
#include "../transforms/shared/component_traversal.hpp"
namespace vcr::components {
void label(const std::uint8_t* input,std::uint32_t* output,const std::vector<std::size_t>& shape,bool full) {
    const auto n=detail::validate(input,output,shape);if(!n)return;
    transforms::Grid grid(shape);
    std::vector<std::uint8_t> visited(n);
    std::vector<std::size_t> queue(n),offsets(n+1);
    const auto count=transforms::component_detail::components(input,transforms::component_detail::GridView(grid),full,visited.data(),queue.data(),offsets.data());
    std::fill_n(output,n,0);
    for(std::size_t c=0;c<count;++c) {
        const auto id=std::uint32_t(queue[offsets[c]]+1);
        for(auto j=offsets[c];j<offsets[c+1];++j)output[queue[j]]=id;
    }
}
}
