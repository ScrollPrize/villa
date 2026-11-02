#include "mask.hpp"
#include <algorithm>
#include <stdexcept>
namespace vcr::projection {
void projectMask(const MaskBand& g,std::uint32_t* output,int id){
    if(g.rows<1||g.cols<1||std::int64_t(g.rows)*g.cols>INT32_MAX||!std::isfinite(g.halfThickness)||g.halfThickness<0||g.halfThickness>100000)throw std::invalid_argument("invalid mask band");
    std::size_t n=1;for(int a=0;a<3;++a){if(g.shape[a]<1)throw std::invalid_argument("invalid projection crop");n*=g.shape[a];}
#ifdef VC_READER_CUDA
    if(id>=0){projectMaskCuda(g,output,id);return;}
#else
    if(id>=0)throw std::runtime_error("vc_reader built without CUDA projection");
#endif
    std::fill_n(output,n,0u);
    auto mark=[&](V p){auto i=maskIndex(g,p);if(i>=0)output[i]=1;};
    for(int i=0;i<g.rows*g.cols;++i)maskCell(g,i,mark);
}
}
