#include "common.hpp"
namespace vcr::distance {
void manhattan(const std::uint8_t* input, std::uint8_t* output, const std::vector<std::size_t>& shape) {
    const auto n=detail::validate(input,output,shape);
    if (!n) return;
    for (std::size_t i=0;i<n;++i) output[i]=input[i] ? 255 : 0;
    std::size_t stride=1;
    for (std::size_t a=shape.size();a-->0;) {
        const auto extent=shape[a];
        if (stride==1) {
            for(std::size_t i=0;i<n;i+=extent) detail::line(output+i,extent,1);
        } else {
            for(std::size_t base=0;base<n;base+=extent*stride) {
                for(std::size_t q=1;q<extent;++q) {
                    auto* row=output+base+q*stride;
                    for(std::size_t i=0;i<stride;++i) {
                        const int v=int((row-stride)[i])+1;
                        if(v<row[i]) row[i]=std::uint8_t(v);
                    }
                }
                for(std::size_t q=extent-1;q-->0;) {
                    auto* row=output+base+q*stride;
                    for(std::size_t i=0;i<stride;++i) {
                        const int v=int(row[i+stride])+1;
                        if(v<row[i]) row[i]=std::uint8_t(v);
                    }
                }
            }
        }
        stride*=extent;
    }
}
}
