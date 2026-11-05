#include "mask.hpp"
#include <algorithm>
#include <atomic>
#include "../thread_pool.hpp"
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
    const int cells=g.rows*g.cols;
    if(cells<64){
        const auto mark=[&](std::int64_t i){output[i]=1;};
        for(int i=0;i<cells;++i)maskCell(g,i,mark);
        return;
    }
    // Every writer stores the same value. Atomic stores avoid data races without
    // a read/modify/write operation; the pool joins before output is consumed.
    const auto mark=[&](std::int64_t i){std::atomic_ref<std::uint32_t>(output[i]).store(1,std::memory_order_relaxed);};
    samplingPool().parallel_for((std::size_t(cells)+15)/16,[&](std::size_t block){
        for(int i=int(block)*16;i<std::min(cells,int(block+1)*16);++i)maskCell(g,i,mark);
    });
}
}
